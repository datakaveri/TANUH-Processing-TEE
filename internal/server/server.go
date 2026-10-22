// Package server wires the Processing TEE's external HTTP surface: the
// RA-TLS connect endpoint, the job intake (/api/load-model, the path the
// Buffer TEE dispatch client POSTs to), and the health check the buffer
// polls while provisioning. Everything runs in-process — the former
// Flask :4000 hop and the :8081 loopback key API are gone.
package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/pipeline"
	"github.com/datakaveri/tanuh-processing-tee/internal/ratls"
)

// readIdleTimeout is how long a payload upload may stall before the
// connection is dropped. There is no limit on the whole upload: a multi-GB
// model at a steady rate takes as long as it needs. A variable so tests can
// shorten it.
var readIdleTimeout = 5 * time.Minute

// runJob starts an accepted job; a variable so tests can stop short of the
// pipeline.
var runJob = func(ctx context.Context, mgr *pipeline.Manager, req *pipeline.Request) { mgr.RunJob(ctx, req) }

// New builds the ServeMux for the external RA-TLS listener.
func New(rs *ratls.Server, mgr *pipeline.Manager, jobCtx context.Context) http.Handler {
	mux := http.NewServeMux()
	mux.Handle(ratls.ConnectPath, rs.ConnectHandler())
	mux.HandleFunc("POST /api/load-model", handleLoadModel(mgr, jobCtx))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	return mux
}

// handleLoadModel receives a payload v3 stream (pipeline.ReceiveV3) and starts
// the pipeline on its own goroutine. 202 once every artifact has arrived and
// verified; 409 while another job is running (answered before the artifacts
// are read); any other 4xx means the payload can never be accepted.
func handleLoadModel(mgr *pipeline.Manager, jobCtx context.Context) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil || mediaType != "multipart/form-data" || params["boundary"] == "" {
			jsonError(w, http.StatusUnsupportedMediaType,
				"payload must be multipart/form-data (payload v3); deploy the matching Buffer TEE")
			return
		}
		started := time.Now()
		log.Printf("server: load-model stream from %s", r.RemoteAddr)

		rc := http.NewResponseController(w)
		body := &idleDeadlineReader{r: r.Body, rc: rc}
		req, err := mgr.ReceiveV3(body, params["boundary"])
		if err != nil {
			var ie *pipeline.IntakeError
			if !errors.As(err, &ie) {
				log.Printf("server: load-model intake failed: %v", err)
				jsonError(w, http.StatusInternalServerError, "intake failed")
				return
			}
			log.Printf("server: load-model refused (%d): %s", ie.Status, ie.Msg)
			status := "error"
			if ie.Status == http.StatusConflict {
				status = "busy"
			}
			writeJSON(w, ie.Status, map[string]string{"status": status, "message": ie.Msg})
			return
		}
		log.Printf("server: secure job %s received in %s (%s)", req.JobID,
			time.Since(started).Round(time.Millisecond), memoryUsage())

		// The job outlives this request — run it on the server's lifetime
		// context, not the request's.
		go runJob(jobCtx, mgr, req)

		// The upload may have outlasted the server's WriteTimeout.
		_ = rc.SetWriteDeadline(time.Now().Add(time.Minute))
		writeJSON(w, http.StatusAccepted, map[string]string{
			"status": "accepted",
			"job_id": req.JobID,
		})
	}
}

// idleDeadlineReader pushes the connection's read deadline readIdleTimeout
// ahead while bytes keep arriving (at most once a second, and at least four
// times per timeout), replacing the server's fixed ReadTimeout for this
// request. The write deadline moves with it: over HTTP/2 the server's
// WriteTimeout ends the stream even while the handler is still reading.
type idleDeadlineReader struct {
	r    io.Reader
	rc   *http.ResponseController
	next time.Time
}

func (d *idleDeadlineReader) Read(p []byte) (int, error) {
	if now := time.Now(); now.After(d.next) {
		_ = d.rc.SetReadDeadline(now.Add(readIdleTimeout))
		_ = d.rc.SetWriteDeadline(now.Add(readIdleTimeout + time.Minute))
		d.next = now.Add(min(time.Second, readIdleTimeout/4))
	}
	return d.r.Read(p)
}

// memoryUsage reports this process's current and peak resident memory.
func memoryUsage() string {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return "memory unknown"
	}
	defer f.Close()
	var out []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		if line := s.Text(); strings.HasPrefix(line, "VmRSS:") || strings.HasPrefix(line, "VmHWM:") {
			out = append(out, strings.Join(strings.Fields(line), " "))
		}
	}
	return strings.Join(out, ", ")
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

func jsonError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"status": "error", "message": msg})
}
