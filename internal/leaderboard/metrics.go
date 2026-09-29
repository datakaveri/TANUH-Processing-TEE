package leaderboard

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/datakaveri/tanuh-processing-tee/internal/catalogue"
)

// metricAliases maps each metric the bucket evaluators produce (standard
// names) to the keys a catalogue entry may use for it. Catalogue keys are
// chosen at onboarding: the UI's built-in keys (e.g. sensitivity) or slugs of
// the display name (e.g. sensitivity_recall). Identity matches always work.
var metricAliases = map[string][]string{
	"f2":          {"f2_score", "f2score"},
	"f1":          {"f1_score", "f1score"},
	"sensitivity": {"sensitivity_recall", "recall", "tpr", "true_positive_rate"},
	"specificity": {"specificity_true_negative_rate", "tnr", "true_negative_rate"},
	"ppv":         {"ppv_positive_predictive_value", "precision", "positive_predictive_value"},
	"npv":         {"npv_negative_predictive_value", "negative_predictive_value"},
	"auc":         {"auc_area_under_curve", "roc_auc", "auroc"},
	"qwk":         {"qwk_quadratic_weight_kappa", "qwk_quadratic_weighted_kappa", "quadratic_weighted_kappa"},
	"fnr":         {"fnr_false_negative_rate", "false_negative_rate"},
	"fpr":         {"fpr_false_positive_rate", "false_positive_rate"},
}

var aliasToStandard = func() map[string]string {
	m := map[string]string{}
	for std, aliases := range metricAliases {
		for _, a := range aliases {
			m[a] = std
		}
	}
	return m
}()

// MetricError explains why the evaluator's metrics cannot satisfy the
// dataset's catalogue-defined requirements.
type MetricError struct {
	Kind string // MetricMappingError or MetricOutOfRangeError
	Msg  string
}

func (e *MetricError) Error() string { return e.Kind + ": " + e.Msg }

// ForLeaderboard builds the submission's metrics object: exactly the keys the
// catalogue requires (the leaderboard validates against them and the UI shows
// every key it receives), each taken from the evaluator's standard metric.
// With no catalogue requirements it passes the evaluator's flat metrics through.
func ForLeaderboard(evalMetrics map[string]any, required map[string]catalogue.MetricSpec) (map[string]any, error) {
	if len(required) == 0 {
		out := map[string]any{}
		for k, v := range evalMetrics {
			if _, nested := v.(map[string]any); !nested {
				out[k] = v
			}
		}
		return out, nil
	}

	out := map[string]any{}
	var unmapped, outOfRange []string
	for _, key := range sortedKeys(required) {
		spec := required[key]
		value, ok := evalMetrics[key]
		if !ok {
			if std, known := aliasToStandard[normalizeKey(key)]; known {
				value, ok = evalMetrics[std]
			}
		}
		if !ok {
			unmapped = append(unmapped, key)
			continue
		}
		if spec.IsMatrix {
			if !isMatrix(value) {
				outOfRange = append(outOfRange, key+" (not a matrix)")
				continue
			}
			out[key] = value
			continue
		}
		num, isNum := value.(float64)
		switch {
		case value == nil:
			outOfRange = append(outOfRange, key+" (undefined for this dataset)")
		case !isNum || math.IsNaN(num) || math.IsInf(num, 0):
			outOfRange = append(outOfRange, key+" (not a number)")
		case num < spec.Min || num > spec.Max:
			outOfRange = append(outOfRange, fmt.Sprintf("%s = %.6g, outside the declared range [%g, %g]", key, num, spec.Min, spec.Max))
		default:
			out[key] = num
		}
	}
	if len(unmapped) > 0 {
		return nil, &MetricError{Kind: "MetricMappingError", Msg: "the dataset requires metrics this bucket's evaluator does not produce: " +
			strings.Join(unmapped, ", ")}
	}
	if len(outOfRange) > 0 {
		return nil, &MetricError{Kind: "MetricOutOfRangeError", Msg: strings.Join(outOfRange, "; ")}
	}
	return out, nil
}

func normalizeKey(k string) string { return strings.ToLower(strings.TrimSpace(k)) }

func isMatrix(v any) bool {
	rows, ok := v.([]any)
	if !ok || len(rows) == 0 {
		return false
	}
	for _, r := range rows {
		cells, ok := r.([]any)
		if !ok {
			return false
		}
		for _, c := range cells {
			if _, ok := c.(float64); !ok {
				return false
			}
		}
	}
	return true
}

func sortedKeys(m map[string]catalogue.MetricSpec) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
