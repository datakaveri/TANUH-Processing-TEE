package catalogue

import (
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func load(t *testing.T, name string) Spec {
	t.Helper()
	raw, err := os.ReadFile("testdata/item_" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	s, err := ParseItem(raw)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func keys(m map[string]MetricSpec) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// The fixtures are the live catalogue entries (metric fields only), so these
// pin exactly what the leaderboard will require for each dataset.
func TestRequiredMetricsFromLiveEntries(t *testing.T) {
	cases := map[string]struct {
		primary string
		want    []string
	}{
		"ocs": {"f2_score", []string{"f2_score", "npv_negative_predictive_value", "ppv_positive_predictive_value",
			"sensitivity_recall", "specificity_true_negative_rate"}},
		"bcd": {"macro_f1", []string{"accuracy", "auc_area_under_curve", "macro_f1", "macro_f2",
			"npv_negative_predictive_value", "ppv_positive_predictive_value", "qwk_quadratic_weight_kappa",
			"sensitivity_recall", "specificity_true_negative_rate", "weighted_f2"}},
		"glaucoma": {"f2_score", []string{"auc_area_under_curve", "confusion_matrix", "f2_score",
			"fnr_false_negative_rate", "fpr_false_positive_rate", "ppv_positive_predictive_value",
			"sensitivity_recall", "specificity_true_negative_rate"}},
	}
	for name, tc := range cases {
		s := load(t, name)
		req := s.RequiredMetrics()
		if got := keys(req); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s required\n got %v\nwant %v", name, got, tc.want)
		}
		if s.PrimaryMetric != tc.primary {
			t.Errorf("%s primary = %q", name, s.PrimaryMetric)
		}
		for k, m := range req {
			if k != "confusion_matrix" && (m.Min != 0 || m.Max != 1) {
				t.Errorf("%s %s range [%g,%g], want [0,1]", name, k, m.Min, m.Max)
			}
		}
		if m, ok := req["confusion_matrix"]; ok && !m.IsMatrix {
			t.Errorf("%s confusion_matrix not flagged as a matrix", name)
		}
	}
}

func TestRangeStringsAndDescriptiveFields(t *testing.T) {
	s := Spec{DatasetMetrics: map[string]any{
		"qwk":                "-1-1",
		"mae":                "0-100",
		"accuracy":           0.0,
		"modality":           "RGB image",
		"outputColumns":      "label",
		"customMetricRanges": map[string]any{"x": "0-1"},
	}}
	req := s.RequiredMetrics()
	if got := keys(req); !reflect.DeepEqual(got, []string{"accuracy", "mae", "qwk"}) {
		t.Fatalf("required = %v", got)
	}
	if req["qwk"].Min != -1 || req["qwk"].Max != 1 || req["mae"].Max != 100 {
		t.Fatalf("ranges = %+v", req)
	}
}

func TestBucketFromProblemStatement(t *testing.T) {
	cases := map[string]struct {
		item string
		want string
	}{
		// As the UI's metadata form writes it (the live entries on 30 Sep 2026).
		"problemStatement":                  {`{"problemStatement":"multiclass_classification","class_names":["A","B","C","D"]}`, "multiclass_classification"},
		"wins over task_type":               {`{"problemStatement":"binary_classification","task_type":"multiclass_classification"}`, "binary_classification"},
		"task_type fallback":                {`{"task_type":"binary_classification"}`, "binary_classification"},
		"empty problemStatement falls back": {`{"problemStatement":"  ","task_type":"binary_classification"}`, "binary_classification"},
		"neither":                           {`{"label":"Oral Cancer"}`, ""},
	}
	for name, c := range cases {
		s, err := ParseItem([]byte(`{"result":[` + c.item + `]}`))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if s.TaskType != c.want {
			t.Errorf("%s: bucket = %q, want %q", name, s.TaskType, c.want)
		}
	}
	s, _ := ParseItem([]byte(`{"result":[{"label":"Oral Cancer","class_names":["a","b"]}]}`))
	if err := s.Validate(); err == nil || !strings.Contains(err.Error(), "problemStatement") {
		t.Fatalf("missing bucket error = %v, want it to name problemStatement", err)
	}
}

func TestParseItemAndValidate(t *testing.T) {
	raw := []byte(`{"result":[{"task_type":" binary_classification ","class_names":"Non-Suspicious, Suspicious","primaryMetric":"f2_score"}]}`)
	s, err := ParseItem(raw)
	if err != nil {
		t.Fatal(err)
	}
	if s.TaskType != "binary_classification" || !reflect.DeepEqual(s.ClassNames, []string{"Non-Suspicious", "Suspicious"}) {
		t.Fatalf("spec = %+v", s)
	}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	// These fixtures predate the problemStatement / class_names fields.
	if err := load(t, "ocs").Validate(); err == nil {
		t.Fatal("entry without a bucket passed validation")
	}
	if _, err := ParseItem([]byte(`{"result":[]}`)); err == nil {
		t.Fatal("empty result accepted")
	}
}
