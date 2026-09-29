// Package leaderboard builds and posts evaluation results to the external
// benchmark leaderboard (POST /leaderboard/submit-solution, see
// tanuh-leaderboards-apis). Submission is best-effort: a failed submit never
// fails the pipeline. Authentication is the submitting user's Keycloak Bearer
// JWT, forwarded through the dispatch payload; the leaderboard only decodes it
// for the `sub` claim. Attestation claims ride in the body so each entry
// carries its TEE provenance.
//
// The leaderboard validates dataset_id against the catalogue and, for a
// succeeded submission, requires every metric the dataset's catalogue entry
// defines (see ForLeaderboard), ranking entries by the primary metric.
package leaderboard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/attest"
)

// Submission carries everything needed for one leaderboard POST.
type Submission struct {
	JobID         string
	DatasetID     string
	Claims        attest.Claims
	Succeeded     bool
	KeycloakToken string

	// Succeeded submissions.
	NumSamples     int
	ElapsedSeconds float64
	ModelSHA256    string
	Providers      []string       // sent as onnx_runtime_providers (torch formats: "torch:cpu"/"torch:cuda")
	Metrics        map[string]any // catalogue-keyed, from ForLeaderboard

	// Failed submissions.
	Error *ErrorInfo
}

// Submit POSTs one evaluation result. Never returns an error — outcomes are
// logged, and the pipeline's success does not depend on it.
func Submit(ctx context.Context, submitURL string, s Submission) {
	// The leaderboard authenticates the caller's Keycloak Bearer JWT. If the
	// browser did not forward one, skip rather than send a token the
	// leaderboard will reject.
	if s.KeycloakToken == "" {
		log.Printf("leaderboard: no Keycloak token forwarded for job %s; skipping submit", s.JobID)
		return
	}

	attestation := map[string]any{
		"hwmodel":      s.Claims.HWModel,
		"swname":       s.Claims.SwName,
		"image_digest": s.Claims.ImageDigest,
		"secboot":      s.Claims.Secboot,
		"iss":          s.Claims.Iss,
	}

	body := map[string]any{
		"job_id":      UUID5("tanuh:" + s.JobID),
		"dataset_id":  s.DatasetID,
		"attestation": attestation,
	}
	status := "failed"
	if s.Succeeded {
		status = "succeeded"
		providers := s.Providers
		if providers == nil {
			providers = []string{}
		}
		body["num_samples"] = s.NumSamples
		body["elapsed_seconds"] = s.ElapsedSeconds
		body["model_sha256"] = s.ModelSHA256
		body["onnx_runtime_providers"] = providers
		body["metrics"] = s.Metrics
	} else if s.Error != nil {
		body["error"] = s.Error
	}
	body["status"] = status

	payload, err := json.Marshal(body)
	if err != nil {
		log.Printf("leaderboard: marshal submit body for job %s: %v", s.JobID, err)
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, submitURL, bytes.NewReader(payload))
	if err != nil {
		log.Printf("leaderboard: build submit request for job %s: %v", s.JobID, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.KeycloakToken)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("leaderboard: submit failed for job %s (non-fatal): %v", s.JobID, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusCreated {
		log.Printf("leaderboard: %s submitted for job %s (dataset %s) → %d", status, s.JobID, s.DatasetID, resp.StatusCode)
	} else {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		log.Printf("leaderboard: submit for job %s returned %d: %s", s.JobID, resp.StatusCode, string(snippet))
	}
}

// String implements fmt.Stringer for logging.
func (e ErrorInfo) String() string {
	return fmt.Sprintf("code=%d type=%s msg=%s", e.Code, e.Type, e.Message)
}
