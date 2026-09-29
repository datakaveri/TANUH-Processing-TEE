// Package catalogue fetches a dataset's declaration (task_type, modality,
// num_classes, class_names, preprocessing hints) from the TANUH catalogue by
// UUID and renders it as the dataset_spec.json the generic evaluation engine
// reads. The catalogue is the single source of truth for dataset metadata; the
// engine never calls it. The Processing TEE fetches at job time so the
// declaration is authoritative and the job submitter cannot influence scoring.
package catalogue

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Spec is the engine's dataset_spec.json contract.
type Spec struct {
	TaskType   string         `json:"task_type"`
	Modality   string         `json:"modality"`
	NumClasses int            `json:"num_classes,omitempty"`
	ClassNames []string       `json:"class_names,omitempty"`
	Input      map[string]any `json:"input,omitempty"`
	DataFile   string         `json:"data_file,omitempty"`
}

// FetchSpec GETs {baseURL}/controlplane/iudx/v2/cat/item?id=<uuid> and maps the
// catalogue item to a Spec. token is the bearer forwarded with the request.
//
// NOTE (placeholder): task_type / num_classes / class_names are NEW catalogue
// fields still being added (plan B3c). This maps them when present; the exact
// field names should be reconciled with the finalised catalogue schema, and the
// bearer should move to the TEE's own identity (fallback: the forwarded keycloak
// token, used here for v1).
func FetchSpec(ctx context.Context, baseURL, uuid, token string) (Spec, error) {
	u := fmt.Sprintf("%s/controlplane/iudx/v2/cat/item?id=%s&auditEnabled=false", baseURL, uuid)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return Spec{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return Spec{}, fmt.Errorf("catalogue: fetch item %s: %w", uuid, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return Spec{}, fmt.Errorf("catalogue: item %s status %d: %s", uuid, resp.StatusCode, string(raw))
	}
	var body struct {
		Result []map[string]any `json:"result"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return Spec{}, fmt.Errorf("catalogue: parse item %s: %w", uuid, err)
	}
	if len(body.Result) == 0 {
		return Spec{}, fmt.Errorf("catalogue: item %s not found", uuid)
	}
	return specFromItem(body.Result[0]), nil
}

func specFromItem(item map[string]any) Spec {
	s := Spec{
		TaskType: str(item["task_type"]),
		Modality: str(item["modality"]),
		DataFile: str(item["data_file"]),
	}
	if n, ok := toInt(item["num_classes"]); ok {
		s.NumClasses = n
	}
	if names, ok := item["class_names"].([]any); ok {
		for _, v := range names {
			s.ClassNames = append(s.ClassNames, str(v))
		}
	}
	if in, ok := item["input"].(map[string]any); ok {
		s.Input = in
	}
	return s
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case string:
		var i int
		if _, err := fmt.Sscanf(n, "%d", &i); err == nil {
			return i, true
		}
	}
	return 0, false
}
