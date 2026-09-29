package leaderboard

import (
	"fmt"
	"testing"

	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
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
		{"user code exit 10", &eval.ExitCodeError{Code: 10}, 1, "EvalUserCodeError"},
		{"cuda exit 11", &eval.ExitCodeError{Code: 11}, 3, "EvalEnvironmentError"},
		{"other exit", &eval.ExitCodeError{Code: 7}, 2, "EvalScriptError"},
		{"env keyword", fmt.Errorf("connection to Secret Manager timed out"), 3, "EnvironmentError"},
		{"deps user error", &eval.DepsError{Output: "no such package: nonexistent-lib"}, 1, "PreprocessingDepsError"},
		{"deps network error", &eval.DepsError{Output: "connection reset while downloading"}, 3, "EnvironmentError"},
		{"generic", fmt.Errorf("something odd happened"), 2, "PipelineError"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			info := Classify(tc.err)
			if info.Code != tc.wantCode || info.Type != tc.wantType {
				t.Fatalf("got (%d,%s), want (%d,%s)", info.Code, info.Type, tc.wantCode, tc.wantType)
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
