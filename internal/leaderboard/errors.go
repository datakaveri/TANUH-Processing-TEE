package leaderboard

import (
	"errors"
	"fmt"
	"io/fs"
	"regexp"
	"strings"

	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
	"github.com/datakaveri/tanuh-processing-tee/internal/modelpkg"
)

var pathRe = regexp.MustCompile(`/[^\s"']+`)

// SanitizeMsg strips filesystem paths from an error message before external
// reporting.
func SanitizeMsg(msg string) string {
	return pathRe.ReplaceAllString(msg, "<path>")
}

// ErrorInfo is the error payload sent with a failed leaderboard submission.
type ErrorInfo struct {
	Code    int    `json:"error_code"`
	Type    string `json:"error_type"`
	Message string `json:"error_message"`
}

// envKeywords indicate environment/infrastructure failures.
var envKeywords = []string{
	"cuda", "gpu", "nvidia", "cudnn",
	"gcs", "kms", "secret manager", "connection", "timeout",
	"no space", "out of memory", "oom",
}

// Classify maps a pipeline error onto the leaderboard's error categories
// (tanuh-leaderboards-apis models.SubmissionError):
//
//	1 = data loading / pre- or post-processing: a dataset file that will not
//	    decode, the adaptor, its dependencies, or its predictions
//	2 = model loading: the model file, its declared format, or running it
//	3 = container / runtime: CUDA, timeouts, the evaluator, metric mapping,
//	    GCS/KMS/catalogue, anything else in the platform
//
// Stage failures get fixed messages; stage output never leaves the TEE.
func Classify(err error) ErrorInfo {
	var st *eval.StageError
	if errors.As(err, &st) {
		return classifyStage(st)
	}
	var fe *modelpkg.FormatError
	if errors.As(err, &fe) {
		return ErrorInfo{2, "ModelFormatError", SanitizeMsg(fe.Msg)}
	}
	var ao *eval.AdaptorOutputError
	if errors.As(err, &ao) {
		return ErrorInfo{1, "AdaptorOutputError", SanitizeMsg(ao.Msg)}
	}
	var me *MetricError
	if errors.As(err, &me) {
		return ErrorInfo{3, me.Kind, SanitizeMsg(me.Msg)}
	}

	msg := err.Error()
	lower := strings.ToLower(msg)
	var depsErr *eval.DepsError
	isDeps := errors.As(err, &depsErr)
	for _, kw := range envKeywords {
		if strings.Contains(lower, kw) {
			return ErrorInfo{3, "EnvironmentError", SanitizeMsg(msg)}
		}
	}
	// A dependency install that failed for a non-network reason (unknown
	// package, resolution conflict) is the adaptor's imports.
	if isDeps {
		return ErrorInfo{1, "AdaptorDepsError", SanitizeMsg(msg)}
	}
	if errors.Is(err, fs.ErrNotExist) {
		return ErrorInfo{3, "FileNotFoundError", SanitizeMsg(msg)}
	}
	if errors.Is(err, fs.ErrPermission) {
		return ErrorInfo{3, "PermissionError", SanitizeMsg(msg)}
	}
	return ErrorInfo{3, "PipelineError", SanitizeMsg(msg)}
}

func classifyStage(st *eval.StageError) ErrorInfo {
	switch st.Stage {
	case eval.StageInfer:
		switch {
		case st.TimedOut:
			return ErrorInfo{3, "InferenceTimeout", "Model inference exceeded its time limit."}
		case st.ExitCode == 11:
			return ErrorInfo{3, "CudaError", "CUDA/GPU runtime error during inference."}
		case st.ExitCode == 13:
			return ErrorInfo{2, "ModelError", "The model could not be loaded, or does not accept the platform's input."}
		case st.ExitCode == 14:
			return ErrorInfo{1, "DatasetDecodeError", "A dataset file could not be decoded."}
		default:
			return ErrorInfo{3, "InferenceError", fmt.Sprintf("Inference exited with code %d.", st.ExitCode)}
		}
	case eval.StageAdaptor:
		if st.TimedOut {
			return ErrorInfo{1, "AdaptorError", "The adaptor exceeded its time limit."}
		}
		return ErrorInfo{1, "AdaptorError", fmt.Sprintf("The adaptor exited with code %d.", st.ExitCode)}
	case eval.StageEvaluator:
		switch {
		case st.ExitCode == 12:
			return ErrorInfo{1, "PredictionsInvalidError",
				"predictions.csv does not match the bucket's format (missing, duplicate or unknown files, or invalid values)."}
		case st.TimedOut:
			return ErrorInfo{3, "EvaluatorError", "The evaluator exceeded its time limit."}
		default:
			return ErrorInfo{3, "EvaluatorError", fmt.Sprintf("The evaluator exited with code %d.", st.ExitCode)}
		}
	}
	return ErrorInfo{3, "PipelineError", st.Error()}
}
