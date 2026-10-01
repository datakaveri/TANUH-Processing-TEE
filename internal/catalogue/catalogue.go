// Package catalogue reads a dataset's entry from the TANUH catalogue by UUID:
// the problem bucket (problemStatement) and class_names needed to run a job,
// and the metric definitions (datasetMetrics, primaryMetric) the leaderboard
// validates a submission against.
//
// The Processing TEE fetches the entry at job time, so what drives scoring and
// submission is the catalogue's own record — the job submitter cannot change it.
package catalogue

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Spec is what a job needs from the catalogue item.
type Spec struct {
	TaskType        string         // problem bucket slug, e.g. binary_classification
	ClassNames      []string       // in label order (label i = ClassNames[i])
	DatasetMetrics  map[string]any // raw datasetMetrics block (metrics + descriptive fields)
	PrimaryMetric   string         // the leaderboard ranks by metrics[PrimaryMetric]
	SecondaryMetric string
}

// FetchSpec GETs {baseURL}/controlplane/iudx/v2/cat/item?id=<uuid>.
// token is the bearer forwarded with the request.
//
// The token is the job submitter's Keycloak access token, which lives only a
// few minutes and can expire while the Processing VM boots. The controlplane
// then answers 401 even for an open item, so on a 401 the request is retried
// once without the token: an open item is served, a restricted one stays 401.
func FetchSpec(ctx context.Context, baseURL, uuid, token string) (Spec, error) {
	status, raw, err := getItem(ctx, baseURL, uuid, token)
	if err == nil && status == http.StatusUnauthorized && token != "" {
		log.Printf("catalogue: item %s: 401 with the submitter's token (%s); retrying without it", uuid, snippet(raw))
		status, raw, err = getItem(ctx, baseURL, uuid, "")
	}
	if err != nil {
		return Spec{}, fmt.Errorf("catalogue: fetch item %s: %w", uuid, err)
	}
	if status != http.StatusOK {
		return Spec{}, fmt.Errorf("catalogue: item %s status %d: %s", uuid, status, snippet(raw))
	}
	return ParseItem(raw)
}

func getItem(ctx context.Context, baseURL, uuid, token string) (int, []byte, error) {
	u := fmt.Sprintf("%s/controlplane/iudx/v2/cat/item?id=%s&auditEnabled=false", baseURL, uuid)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return 0, nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return resp.StatusCode, raw, nil
}

// ParseItem maps a cat/item response body to a Spec.
func ParseItem(raw []byte) (Spec, error) {
	var body struct {
		Result []map[string]any `json:"result"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return Spec{}, fmt.Errorf("catalogue: parse item: %w", err)
	}
	if len(body.Result) == 0 {
		return Spec{}, fmt.Errorf("catalogue: item not found")
	}
	item := body.Result[0]
	// The UI's metadata form names the bucket "problemStatement" (values are
	// the bucket slugs, e.g. binary_classification); "task_type" is the name
	// this TEE used first and is still accepted.
	bucket := strings.TrimSpace(str(item["problemStatement"]))
	if bucket == "" {
		bucket = strings.TrimSpace(str(item["task_type"]))
	}
	s := Spec{
		TaskType:        bucket,
		ClassNames:      classNames(item["class_names"]),
		PrimaryMetric:   str(item["primaryMetric"]),
		SecondaryMetric: str(item["secondaryMetric"]),
	}
	s.DatasetMetrics, _ = item["datasetMetrics"].(map[string]any)
	return s, nil
}

// Validate checks the fields a job cannot run without.
func (s Spec) Validate() error {
	if s.TaskType == "" {
		return fmt.Errorf("catalogue: the dataset has no problemStatement (problem bucket, e.g. binary_classification)")
	}
	min := 2 // classification and segmentation (background + 1)
	if s.TaskType == "object_detection" {
		min = 1 // no background class: a single object type is a valid detection task
	}
	if len(s.ClassNames) < min {
		return fmt.Errorf("catalogue: the dataset needs at least %d class_names, has %d", min, len(s.ClassNames))
	}
	return nil
}

// MetricSpec is one metric the leaderboard requires for this dataset.
type MetricSpec struct {
	Name     string
	IsMatrix bool
	Min, Max float64
}

var rangePattern = regexp.MustCompile(`^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$`)

// RequiredMetrics returns the metric keys a submission must include, using
// the leaderboard's own rule (tanuh-leaderboards-apis internal/catalogue):
// a datasetMetrics key whose value is a bare number is required with range
// [0,1]; a key whose value is a "lo-hi" string is required within that range;
// every other key is descriptive metadata. confusion_matrix is a matrix.
func (s Spec) RequiredMetrics() map[string]MetricSpec {
	out := map[string]MetricSpec{}
	for key, v := range s.DatasetMetrics {
		switch val := v.(type) {
		case float64:
			out[key] = MetricSpec{Name: key, Min: 0, Max: 1}
		case string:
			if m := rangePattern.FindStringSubmatch(val); m != nil {
				lo, _ := strconv.ParseFloat(m[1], 64)
				hi, _ := strconv.ParseFloat(m[2], 64)
				out[key] = MetricSpec{Name: key, Min: lo, Max: hi}
			}
		}
	}
	if spec, ok := out["confusion_matrix"]; ok {
		spec.IsMatrix = true
		out["confusion_matrix"] = spec
	}
	return out
}

// classNames accepts a JSON array or a comma-separated string.
func classNames(v any) []string {
	var names []string
	switch t := v.(type) {
	case []any:
		for _, x := range t {
			if s := strings.TrimSpace(str(x)); s != "" {
				names = append(names, s)
			}
		}
	case string:
		for _, s := range strings.Split(t, ",") {
			if s = strings.TrimSpace(s); s != "" {
				names = append(names, s)
			}
		}
	}
	return names
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func snippet(b []byte) string {
	if len(b) > 300 {
		return string(b[:300]) + "…"
	}
	return string(b)
}
