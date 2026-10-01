package leaderboard

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/datakaveri/tanuh-processing-tee/internal/catalogue"
	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
	"github.com/datakaveri/tanuh-processing-tee/internal/modelpkg"
)

// Golden from Python: str(uuid.uuid5(uuid.NAMESPACE_URL, "tanuh:job-test-123"))
func TestUUID5MatchesPython(t *testing.T) {
	got := UUID5("tanuh:job-test-123")
	const golden = "1640b625-aa06-5f8c-a77a-befd07cc4020"
	if got != golden {
		t.Fatalf("uuid5 diverged from Python golden:\n got  %s\n want %s", got, golden)
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		name     string
		err      error
		wantCode int
		wantType string
	}{
		{"model will not load", &eval.StageError{Stage: eval.StageInfer, ExitCode: 13}, 2, "ModelError"},
		{"dataset file will not decode", &eval.StageError{Stage: eval.StageInfer, ExitCode: 14}, 1, "DatasetDecodeError"},
		{"cuda", &eval.StageError{Stage: eval.StageInfer, ExitCode: 11}, 3, "CudaError"},
		{"inference timeout", &eval.StageError{Stage: eval.StageInfer, TimedOut: true}, 3, "InferenceTimeout"},
		{"inference crash", &eval.StageError{Stage: eval.StageInfer, ExitCode: 1}, 3, "InferenceError"},
		{"adaptor crash", &eval.StageError{Stage: eval.StageAdaptor, ExitCode: 1}, 1, "AdaptorError"},
		{"adaptor timeout", &eval.StageError{Stage: eval.StageAdaptor, TimedOut: true}, 1, "AdaptorError"},
		{"bad predictions", &eval.StageError{Stage: eval.StageEvaluator, ExitCode: 12}, 1, "PredictionsInvalidError"},
		{"evaluator crash", &eval.StageError{Stage: eval.StageEvaluator, ExitCode: 1}, 3, "EvaluatorError"},
		{"no predictions file", &eval.AdaptorOutputError{Msg: "predictions.csv missing"}, 1, "AdaptorOutputError"},
		{"wrong model format", fmt.Errorf("materialize: %w", &modelpkg.FormatError{Msg: "not an ONNX file"}), 2, "ModelFormatError"},
		{"metric mapping", &MetricError{Kind: "MetricMappingError", Msg: "x"}, 3, "MetricMappingError"},
		{"deps user error", &eval.DepsError{Output: "no such package: nonexistent-lib"}, 1, "AdaptorDepsError"},
		{"deps network error", &eval.DepsError{Output: "connection reset while downloading"}, 3, "EnvironmentError"},
		{"env keyword", fmt.Errorf("KMS asymmetricDecrypt: connection refused"), 3, "EnvironmentError"},
		{"generic", fmt.Errorf("something odd happened"), 3, "PipelineError"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			info := Classify(tc.err)
			if info.Code != tc.wantCode || info.Type != tc.wantType {
				t.Fatalf("got (%d,%s), want (%d,%s)", info.Code, info.Type, tc.wantCode, tc.wantType)
			}
			if info.Code < 1 || info.Code > 3 || info.Message == "" {
				t.Fatalf("leaderboard would reject %+v", info)
			}
		})
	}
}

func TestSanitizeMsg(t *testing.T) {
	got := SanitizeMsg(`open /app/cvm_workflow/secure_jobs/j1/artifacts/model.onnx: no such file`)
	want := `open <path> no such file`
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func catalogueSpec(t *testing.T, name string) catalogue.Spec {
	t.Helper()
	raw, err := os.ReadFile("../catalogue/testdata/item_" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	s, err := catalogue.ParseItem(raw)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func evalMetrics(t *testing.T, name string) map[string]any {
	t.Helper()
	raw, err := os.ReadFile("testdata/results_" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	var r struct {
		Metrics map[string]any `json:"metrics"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r.Metrics
}

func mapKeys(m map[string]any) []string {
	var k []string
	for key := range m {
		k = append(k, key)
	}
	sort.Strings(k)
	return k
}

// Real evaluator output from the parity runs against the live catalogue entries.
func TestForLeaderboardMatchesLiveCatalogue(t *testing.T) {
	cases := []struct {
		dataset, results string
	}{
		{"ocs", "binary_ocs"},
		{"bcd", "multiclass_bcd"},
		{"glaucoma", "binary_ocs"}, // a binary result mapped onto the Glaucoma entry (fnr/fpr/confusion_matrix)
	}
	for _, tc := range cases {
		spec := catalogueSpec(t, tc.dataset)
		required := spec.RequiredMetrics()
		got, err := ForLeaderboard(evalMetrics(t, tc.results), required)
		if err != nil {
			t.Fatalf("%s: %v", tc.dataset, err)
		}
		var want []string
		for k := range required {
			want = append(want, k)
		}
		sort.Strings(want)
		if !reflect.DeepEqual(mapKeys(got), want) {
			t.Fatalf("%s keys\n got %v\nwant %v", tc.dataset, mapKeys(got), want)
		}
		if _, ok := got[spec.PrimaryMetric]; !ok {
			t.Fatalf("%s: primary metric %q missing", tc.dataset, spec.PrimaryMetric)
		}
	}
	// Spot-check the translation: OCS f2_score is the evaluator's f2.
	ocs, _ := ForLeaderboard(evalMetrics(t, "binary_ocs"), catalogueSpec(t, "ocs").RequiredMetrics())
	if ocs["f2_score"] != evalMetrics(t, "binary_ocs")["f2"] {
		t.Fatalf("f2_score = %v", ocs["f2_score"])
	}
}

func TestForLeaderboardNegativeQWKAgainstDefaultRange(t *testing.T) {
	m := evalMetrics(t, "multiclass_bcd")
	m["qwk"] = -0.037
	_, err := ForLeaderboard(m, catalogueSpec(t, "bcd").RequiredMetrics())
	var me *MetricError
	if !errors.As(err, &me) || me.Kind != "MetricOutOfRangeError" || !strings.Contains(me.Msg, "qwk_quadratic_weight_kappa") {
		t.Fatalf("err = %v", err)
	}
	// Declared as "-1-1" in the catalogue, the same value is accepted.
	req := catalogueSpec(t, "bcd").RequiredMetrics()
	q := req["qwk_quadratic_weight_kappa"]
	q.Min, q.Max = -1, 1
	req["qwk_quadratic_weight_kappa"] = q
	if _, err := ForLeaderboard(m, req); err != nil {
		t.Fatalf("with a -1..1 range: %v", err)
	}
}

func TestForLeaderboardErrors(t *testing.T) {
	required := map[string]catalogue.MetricSpec{"dice_coefficient": {Name: "dice_coefficient", Max: 1}}
	_, err := ForLeaderboard(evalMetrics(t, "binary_ocs"), required)
	var me *MetricError
	if !errors.As(err, &me) || me.Kind != "MetricMappingError" {
		t.Fatalf("unmapped metric err = %v", err)
	}
	m := evalMetrics(t, "binary_ocs")
	m["auc"] = nil
	_, err = ForLeaderboard(m, map[string]catalogue.MetricSpec{"auc_area_under_curve": {Max: 1}})
	if !errors.As(err, &me) || me.Kind != "MetricOutOfRangeError" {
		t.Fatalf("undefined AUC err = %v", err)
	}
}

// The kidney-detection legend's metric names, slugged as the UI stores them,
// map onto the detection evaluator's names.
func TestForLeaderboardDetectionKeys(t *testing.T) {
	eval := map[string]any{"map": 0.41, "ap50": 0.72, "ap75": 0.39, "map_giou": 0.37, "recall_50": 0.8, "mean_iou": 0.66,
		"per_class": map[string]any{}}
	required := map[string]catalogue.MetricSpec{}
	for _, k := range []string{"map_0_5_0_95", "giou_matched_map_0_5_0_95", "ap_0_5", "ap_0_75", "mean_iou"} {
		required[k] = catalogue.MetricSpec{Name: k, Max: 1}
	}
	got, err := ForLeaderboard(eval, required)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"map_0_5_0_95": 0.41, "giou_matched_map_0_5_0_95": 0.37, "ap_0_5": 0.72, "ap_0_75": 0.39, "mean_iou": 0.66}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v", got)
	}
}

func TestForLeaderboardWithoutRequirementsSendsFlatMetrics(t *testing.T) {
	got, err := ForLeaderboard(evalMetrics(t, "multiclass_bcd"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, nested := got["per_class"]; nested {
		t.Fatal("nested per_class sent")
	}
	if got["qwk"] == nil {
		t.Fatal("metrics missing")
	}
}
