package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"testing"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
	"github.com/datakaveri/tanuh-processing-tee/internal/pipeline"
)

// v3Body builds a payload v3 body for an ONNX model and adaptor.
func v3Body(t *testing.T, jobID string) (*bytes.Buffer, string) {
	t.Helper()
	uploads := []struct {
		slot  string
		plain []byte
	}{
		{"model", append([]byte{0x08}, bytes.Repeat([]byte("m"), 3<<20)...)}, // several 1 MiB chunks
		{"adaptor", []byte("print('adaptor')")},
	}
	artifacts := map[string]any{}
	streams := map[string][]byte{}
	for i, u := range uploads {
		key := bytes.Repeat([]byte{byte(i + 1)}, 32)
		stream, err := crypto.EncodeChunked(key, make([]byte, 12), u.slot, u.plain, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(u.plain)
		artifacts[u.slot] = map[string]any{
			"encoding":        crypto.ChunkedEncoding,
			"sha256":          hex.EncodeToString(sum[:]),
			"ciphertext_size": len(stream),
			"aes_key":         base64.StdEncoding.EncodeToString(key),
		}
		streams[u.slot] = stream
	}
	meta, _ := json.Marshal(map[string]any{
		"payload_version": 3, "job_id": jobID, "dataset_id": "d", "model_format": "onnx", "artifacts": artifacts,
	})
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	write := func(name string, data []byte) {
		h := textproto.MIMEHeader{}
		h.Set("Content-Disposition", fmt.Sprintf(`form-data; name=%q`, name))
		w, _ := mw.CreatePart(h)
		w.Write(data) //nolint:errcheck
	}
	write("metadata", meta)
	for _, u := range uploads {
		write(u.slot, streams[u.slot])
	}
	mw.Close()
	return &buf, mw.FormDataContentType()
}

// startServer serves the load-model handler; accepted jobs are sent on the
// returned channel instead of being run.
func startServer(t *testing.T) (*httptest.Server, *pipeline.Manager, chan *pipeline.Request) {
	t.Helper()
	mgr, err := pipeline.NewManager(config.Config{BaseDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	started := make(chan *pipeline.Request, 1)
	saved := runJob
	runJob = func(_ context.Context, _ *pipeline.Manager, req *pipeline.Request) { started <- req }
	t.Cleanup(func() { runJob = saved })
	srv := httptest.NewServer(handleLoadModel(mgr, context.Background()))
	t.Cleanup(srv.Close)
	return srv, mgr, started
}

func post(t *testing.T, url, contentType string, body *bytes.Buffer) (int, map[string]string) {
	t.Helper()
	resp, err := http.Post(url, contentType, body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]string
	json.NewDecoder(resp.Body).Decode(&out) //nolint:errcheck
	return resp.StatusCode, out
}

func TestLoadModelAcceptsV3(t *testing.T) {
	srv, _, started := startServer(t)
	body, ct := v3Body(t, "job-1")
	code, out := post(t, srv.URL, ct, body)
	if code != http.StatusAccepted || out["status"] != "accepted" || out["job_id"] != "job-1" {
		t.Fatalf("%d %v", code, out)
	}
	req := <-started
	if req.JobID != "job-1" || req.Artifacts["model"].Size != 3<<20+1 {
		t.Fatalf("request = %+v", req)
	}
}

func TestLoadModelBusy(t *testing.T) {
	srv, mgr, _ := startServer(t)
	mgr.TryStartJob("job-running")
	body, ct := v3Body(t, "job-2")
	code, out := post(t, srv.URL, ct, body)
	if code != http.StatusConflict || out["status"] != "busy" {
		t.Fatalf("%d %v", code, out)
	}
}

// pacedReader hands out data at a steady rate: perByte of delay per byte.
type pacedReader struct {
	data    []byte
	perByte time.Duration
}

func (p *pacedReader) Read(b []byte) (int, error) {
	if len(p.data) == 0 {
		return 0, io.EOF
	}
	n := copy(b[:min(len(b), 64<<10)], p.data)
	p.data = p.data[n:]
	time.Sleep(time.Duration(n) * p.perByte)
	return n, nil
}

// A payload that takes longer than the server's ReadTimeout and WriteTimeout
// is received while bytes keep arriving, over TLS as the Buffer sends it.
func TestLoadModelOutlivesServerTimeouts(t *testing.T) {
	saved := readIdleTimeout
	readIdleTimeout = 600 * time.Millisecond
	defer func() { readIdleTimeout = saved }()

	mgr, err := pipeline.NewManager(config.Config{BaseDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	savedRun := runJob
	runJob = func(context.Context, *pipeline.Manager, *pipeline.Request) {}
	defer func() { runJob = savedRun }()
	srv := httptest.NewUnstartedServer(handleLoadModel(mgr, context.Background()))
	srv.EnableHTTP2 = true
	srv.Config.ReadTimeout = time.Second
	srv.Config.WriteTimeout = time.Second
	srv.StartTLS()
	defer srv.Close()

	body, ct := v3Body(t, "job-slow")
	slow := &pacedReader{data: body.Bytes(), perByte: 2 * time.Second / time.Duration(body.Len())} // ~2 s in all
	resp, err := srv.Client().Post(srv.URL, ct, slow)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("slow payload over %s: %d", resp.Proto, resp.StatusCode)
	}
}

func TestLoadModelRejects(t *testing.T) {
	srv, _, _ := startServer(t)
	code, out := post(t, srv.URL, "application/json", bytes.NewBufferString(`{"payload_version": 2}`))
	if code != http.StatusUnsupportedMediaType || out["status"] != "error" {
		t.Fatalf("v2 JSON: %d %v", code, out)
	}
	body, ct := v3Body(t, "../evil")
	if code, out := post(t, srv.URL, ct, body); code != http.StatusBadRequest || out["message"] != "invalid job_id" {
		t.Fatalf("bad job id: %d %v", code, out)
	}
}
