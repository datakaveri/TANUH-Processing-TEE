package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/attest"
	"github.com/datakaveri/tanuh-processing-tee/internal/catalogue"
	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
	"github.com/datakaveri/tanuh-processing-tee/internal/groundtruth"
	"github.com/datakaveri/tanuh-processing-tee/internal/leaderboard"
	"github.com/datakaveri/tanuh-processing-tee/internal/modelpkg"
)

// payloadVersion is the Buffer → Processing TEE payload layout this build
// understands (Buffer: scheduler.PayloadVersion). Both ship together.
const payloadVersion = 2

// Size caps for files the Python stages hand back.
const (
	maxPredictionsBytes = 64 << 20
	maxResultsBytes     = 1 << 20
)

// gtKind is what a bucket's ground truth gives per sample.
type gtKind string

const (
	gtClass gtKind = "class" // a class name or index (groundtruth.Match)
	gtMask  gtKind = "mask"  // a mask file in the dataset (groundtruth.MatchMasks)
	gtBox   gtKind = "box"   // zero or more boxes (groundtruth.MatchBoxes); inference runs per sample
)

// bucketKinds lists the buckets whose ground truth is not a class per sample.
var bucketKinds = map[string]gtKind{"segmentation": gtMask, "object_detection": gtBox}

func kindOf(bucket string) gtKind {
	if k, ok := bucketKinds[bucket]; ok {
		return k
	}
	return gtClass
}

// materialized holds everything RunJob needs after payload validation.
type materialized struct {
	jobID         string
	datasetUUID   string
	modelFormat   string
	jobDir        string
	runtimeDir    string
	modelDir      string // laid out by modelpkg for infer.py --model-dir
	adaptorPath   string
	modelSHA256   string
	weightsSHA256 string
	adaptorSHA256 string
	// inputSpecSHA256 is set when an input_spec.json was uploaded.
	inputSpecSHA256 string
	keycloakToken   string
	bufferJobURL    string
}

// RunJob executes the full secure-job pipeline for an accepted payload and
// always releases the job slot. Meant to run on its own goroutine.
func (m *Manager) RunJob(ctx context.Context, payload map[string]any) {
	jobID := stringField(payload, "job_id")
	defer m.releaseJobSlot()

	job, err := m.materialize(payload)
	if err == nil {
		err = m.runSteps(ctx, job)
	}
	if err == nil {
		return
	}

	// Failure path: log the full error inside the TEE (serial log only),
	// report a sanitised classification externally, notify the buffer, and
	// deallocate so a failed job never leaves the VM running.
	log.Printf("pipeline: secure job %s failed: %v", jobID, err)
	m.setState(func(s *State) {
		s.Status = "error"
		s.CurrentJobID = ""
		s.LastJobID = jobID
		s.LastError = err.Error()
	})

	errInfo := leaderboard.Classify(err)
	claims := attest.FetchClaims(ctx, m.cfg.AttestAudience)
	leaderboard.Submit(ctx, m.cfg.LeaderboardURL, leaderboard.Submission{
		JobID:         jobID,
		DatasetID:     stringField(payload, "dataset_id"),
		Claims:        claims,
		Succeeded:     false,
		Error:         &errInfo,
		KeycloakToken: stringField(payload, "keycloak_token"),
	})
	m.notifyBuffer(ctx, stringField(payload, "buffer_job_url"), jobID, "failed", &errInfo)

	if m.cfg.DeallocAfterJob {
		m.RequestDeallocation(ctx, fmt.Sprintf("job %s failed", jobID))
	}
}

// runSteps is the happy path:
//
//	catalogue entry → evaluator → adaptor deps → dataset decrypt → ground truth
//	→ infer → adaptor → evaluator → leaderboard metrics → submit → callback
func (m *Manager) runSteps(ctx context.Context, job *materialized) error {
	started := time.Now()
	m.setState(func(s *State) {
		s.Status = "running"
		s.CurrentJobID = job.jobID
		s.LastJobID = job.jobID
		s.LastError = ""
	})
	log.Printf("pipeline: secure job %s: dataset %s, model format %s", job.jobID, job.datasetUUID, job.modelFormat)

	timings := map[string]float64{}
	timed := func(name string, fn func() error) error {
		t0 := time.Now()
		err := fn()
		timings[name] = roundTo(time.Since(t0).Seconds(), 3)
		return err
	}

	// 1. The dataset's catalogue entry: problem bucket, classes, and the
	//    metrics the leaderboard will require. Read at job time, so the job
	//    submitter cannot influence how their model is scored.
	spec, err := catalogue.FetchSpec(ctx, m.cfg.CatalogueBaseURL, job.datasetUUID, job.keycloakToken)
	if err != nil {
		return fmt.Errorf("pipeline: fetch catalogue entry: %w", err)
	}
	if err := spec.Validate(); err != nil {
		return err
	}
	required := spec.RequiredMetrics()

	// 2. The bucket's evaluator, fetched before any expensive step so a
	//    missing one fails fast.
	ev, err := m.evaluators.Fetch(ctx, spec.TaskType, filepath.Join(job.runtimeDir, "evaluator"))
	if err != nil {
		return err
	}
	log.Printf("pipeline: secure job %s: bucket %s, evaluator %s (sha256 %s…)",
		job.jobID, ev.Bucket, ev.Origin, ev.SHA256[:12])

	// 3. Third-party imports the adaptor needs.
	if err := timed("deps", func() error {
		return eval.InstallDeps(ctx, m.cfg.BaseDir, filepath.Join(m.cfg.BaseDir, "dep_scanner.py"),
			job.adaptorPath, m.cfg.DepsTimeout)
	}); err != nil {
		return err
	}

	// 4. Decrypt the ground truth, match it to the dataset's files, and decrypt
	//    those files into a per-job temp folder. It holds the data provider's
	//    data, so it is wiped when the job finishes either way.
	datasetDir := filepath.Join(job.runtimeDir, "dataset")
	defer os.RemoveAll(datasetDir)
	kind := kindOf(spec.TaskType)
	var ds *dataset
	if err := timed("dataset", func() (err error) {
		ds, err = m.fetchAndDecryptDataset(ctx, job.datasetUUID, datasetDir, spec.ClassNames, kind)
		return err
	}); err != nil {
		return err
	}

	// 5. The canonical ground truth for the evaluator (file = path in the
	//    dataset, then the class index; for segmentation the decrypted mask;
	//    for detection one row per box), and the input list for inference.
	rows := ds.Match.Rows
	groundTruthPath := filepath.Join(job.runtimeDir, "ground_truth.csv")
	switch kind {
	case gtMask:
		err = groundtruth.WriteCanonicalMasks(groundTruthPath, rows, ds.Match.Masks, func(f string) string { return ds.Paths[f] })
	case gtBox:
		err = groundtruth.WriteCanonicalBoxes(groundTruthPath, rows, ds.Match.Boxes)
	default:
		err = groundtruth.WriteCanonical(groundTruthPath, rows)
	}
	if err != nil {
		return err
	}
	inputsPath := filepath.Join(job.runtimeDir, "inputs.txt")
	if err := groundtruth.WriteInputs(inputsPath, rows, func(f string) string { return ds.Paths[f] }); err != nil {
		return fmt.Errorf("pipeline: %w", err)
	}
	specPath := filepath.Join(job.runtimeDir, "dataset_spec.json")
	if err := writeJSON(specPath, map[string]any{
		"task_type":   spec.TaskType,
		"modality":    ds.Modality,
		"class_names": spec.ClassNames,
		"num_classes": len(spec.ClassNames),
	}); err != nil {
		return err
	}
	log.Printf("pipeline: secure job %s: %d %s samples, %d classes (ground truth %s: %s, id column %q, label column %q, labels by %s, matched by %v)",
		job.jobID, len(rows), ds.Modality, len(spec.ClassNames), ds.GroundTruth, ds.Table.Format,
		ds.Table.IDColumn, ds.Table.LabelColumn, ds.Match.LabelMode, ds.Match.MatchedBy)

	// 6. The three Python stages.
	rawDir := filepath.Join(job.runtimeDir, "raw")
	predDir := filepath.Join(job.runtimeDir, "predictions")
	resultsPath := filepath.Join(job.runtimeDir, "results.json")
	inferArgs := []string{"--format", job.modelFormat, "--model-dir", job.modelDir,
		"--inputs", inputsPath, "--output-dir", rawDir}
	if kind == gtBox {
		inferArgs = append(inferArgs, "--per-sample") // detectors return a different number of boxes per image
	}
	if err := timed(eval.StageInfer, func() error {
		return eval.RunStage(ctx, eval.Stage{
			Name:    eval.StageInfer,
			Script:  filepath.Join(m.cfg.BaseDir, "infer.py"),
			Args:    inferArgs,
			Dir:     job.runtimeDir,
			Env:     []string{"HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"},
			Timeout: m.cfg.InferTimeout,
		})
	}); err != nil {
		return err
	}
	if err := timed(eval.StageAdaptor, func() error {
		return eval.RunStage(ctx, eval.Stage{
			Name:    eval.StageAdaptor,
			Script:  job.adaptorPath,
			Args:    []string{"--raw-dir", rawDir, "--spec", specPath, "--output-dir", predDir},
			Dir:     job.runtimeDir,
			Timeout: m.cfg.AdaptorTimeout,
		})
	}); err != nil {
		return err
	}
	predictionsPath, err := findPredictions(predDir)
	if err != nil {
		return err
	}
	if err := checkPredictions(predictionsPath); err != nil {
		return err
	}
	if err := timed(eval.StageEvaluator, func() error {
		return eval.RunStage(ctx, eval.Stage{
			Name:   eval.StageEvaluator,
			Script: ev.Path,
			Args: []string{"--predictions", predictionsPath, "--ground-truth", groundTruthPath,
				"--spec", specPath, "--results", resultsPath},
			Dir:     job.runtimeDir,
			Timeout: m.cfg.EvaluatorTimeout,
		})
	}); err != nil {
		return err
	}

	// 7. Results → the metrics object the leaderboard requires for this dataset.
	results, numSamples, evalMetrics, err := readResults(resultsPath)
	if err != nil {
		return err
	}
	lbMetrics, err := leaderboard.ForLeaderboard(evalMetrics, required)
	if err != nil {
		return err
	}
	meta := readInferMeta(filepath.Join(rawDir, "meta.json"))
	elapsed := roundTo(time.Since(started).Seconds(), 3)

	results["job_id"] = job.jobID
	results["dataset_id"] = job.datasetUUID
	results["payload_version"] = payloadVersion
	results["bucket"] = spec.TaskType
	results["model_format"] = job.modelFormat
	results["runtime"] = meta.Runtime
	results["runtime_version"] = meta.RuntimeVersion
	results["device"] = meta.Device
	results["onnx_runtime_providers"] = meta.Providers
	results["model_sha256"] = job.modelSHA256
	results["weights_sha256"] = job.weightsSHA256
	results["adaptor_sha256"] = job.adaptorSHA256
	results["input_spec_sha256"] = job.inputSpecSHA256
	results["evaluator"] = map[string]any{"origin": ev.Origin, "sha256": ev.SHA256}
	results["ground_truth"] = map[string]any{ // counts only: no sample names leave the TEE log
		"file":         ds.GroundTruth,
		"format":       ds.Table.Format,
		"id_column":    ds.Table.IDColumn,
		"label_column": ds.Table.LabelColumn,
		"label_mode":   ds.Match.LabelMode,
		"rows":         ds.Match.Total,
		"scored":       len(rows),
		"unmatched":    len(ds.Match.Unmatched),
		"unlabeled":    ds.Match.Unlabeled,
		"matched_by":   ds.Match.MatchedBy,
		"boxes":        ds.Match.BoxCount(),
	}
	results["modality"] = ds.Modality
	results["leaderboard_metrics"] = lbMetrics
	results["stage_seconds"] = timings
	results["elapsed_seconds"] = elapsed
	if err := writeJSON(resultsPath, results); err != nil {
		return err
	}
	log.Printf("pipeline: secure job %s: scored %d samples in %.1fs (%s)", job.jobID, numSamples, elapsed, formatTimings(timings))

	// 8. Report — attestation claims ride in the leaderboard body.
	claims := attest.FetchClaims(ctx, m.cfg.AttestAudience)
	leaderboard.Submit(ctx, m.cfg.LeaderboardURL, leaderboard.Submission{
		JobID:          job.jobID,
		DatasetID:      job.datasetUUID,
		Claims:         claims,
		Succeeded:      true,
		NumSamples:     numSamples,
		ElapsedSeconds: elapsed,
		ModelSHA256:    job.modelSHA256,
		Providers:      meta.Providers,
		Metrics:        lbMetrics,
		KeycloakToken:  job.keycloakToken,
	})
	m.notifyBuffer(ctx, job.bufferJobURL, job.jobID, "succeeded", nil)

	m.setState(func(s *State) {
		s.Status = "complete"
		s.CurrentJobID = ""
		s.LastJobID = job.jobID
		s.LastError = ""
	})
	if m.cfg.DeallocAfterJob {
		m.RequestDeallocation(ctx, fmt.Sprintf("job %s finished successfully", job.jobID))
	}
	return nil
}

// materialize validates a version-2 payload and lays the artifacts out:
//
//	{"payload_version": 2, "job_id", "dataset_id", "model_format",
//	 "keycloak_token", "buffer_job_url",
//	 "artifacts": {"model"|"weights"|"adaptor"|"input_spec": {"sha256", "base64"}}}
//
// Every artifact must carry a SHA-256 (committed by the browser at submit and
// re-checked by the Buffer) that matches its bytes. Confidentiality in transit
// is the RA-TLS channel.
func (m *Manager) materialize(payload map[string]any) (*materialized, error) {
	if v, _ := payload["payload_version"].(float64); int(v) != payloadVersion {
		return nil, fmt.Errorf("pipeline: unsupported payload_version %v (this Processing TEE needs %d; "+
			"deploy the matching Buffer TEE)", payload["payload_version"], payloadVersion)
	}
	job := &materialized{
		jobID:         stringField(payload, "job_id"),
		datasetUUID:   stringField(payload, "dataset_id"),
		modelFormat:   stringField(payload, "model_format"),
		keycloakToken: stringField(payload, "keycloak_token"),
		bufferJobURL:  stringField(payload, "buffer_job_url"),
	}
	if job.jobID == "" || job.jobID != filepath.Base(job.jobID) {
		return nil, fmt.Errorf("pipeline: invalid job_id")
	}
	if job.datasetUUID == "" {
		return nil, fmt.Errorf("pipeline: empty dataset_id")
	}
	switch job.modelFormat {
	case modelpkg.ONNX, modelpkg.TorchScript, modelpkg.HuggingFace:
	default:
		return nil, &modelpkg.FormatError{Msg: fmt.Sprintf("unsupported model_format %q", job.modelFormat)}
	}

	artifacts, _ := payload["artifacts"].(map[string]any)
	model, modelSHA, err := decodeArtifact(artifacts, "model", true)
	if err != nil {
		return nil, err
	}
	weights, weightsSHA, err := decodeArtifact(artifacts, "weights", false)
	if err != nil {
		return nil, err
	}
	adaptor, adaptorSHA, err := decodeArtifact(artifacts, "adaptor", true)
	if err != nil {
		return nil, err
	}
	inputSpec, inputSpecSHA, err := decodeArtifact(artifacts, "input_spec", false)
	if err != nil {
		return nil, err
	}
	job.modelSHA256, job.weightsSHA256, job.adaptorSHA256 = modelSHA, weightsSHA, adaptorSHA
	job.inputSpecSHA256 = inputSpecSHA

	job.jobDir = m.jobDir(job.jobID)
	job.runtimeDir = filepath.Join(job.jobDir, "runtime")
	job.modelDir = filepath.Join(job.jobDir, "artifacts", "model")
	scriptsDir := filepath.Join(job.jobDir, "artifacts", "scripts")
	for _, d := range []string{job.runtimeDir, scriptsDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	if err := modelpkg.Place(job.modelFormat, model, weights, inputSpec, job.modelDir); err != nil {
		return nil, err
	}
	job.adaptorPath = filepath.Join(scriptsDir, "adaptor.py")
	if err := os.WriteFile(job.adaptorPath, adaptor, 0o644); err != nil {
		return nil, err
	}

	if raw, err := json.MarshalIndent(payload, "", "  "); err == nil {
		if err := os.WriteFile(filepath.Join(job.jobDir, "incoming_payload.json"), raw, 0o644); err != nil {
			log.Printf("pipeline: write incoming_payload.json: %v", err)
		}
	}
	log.Printf("pipeline: secure job %s materialized (%s model %d bytes, weights %d bytes, adaptor %d bytes)",
		job.jobID, job.modelFormat, len(model), len(weights), len(adaptor))
	return job, nil
}

func stringField(payload map[string]any, key string) string {
	s, _ := payload[key].(string)
	return s
}

// decodeArtifact base64-decodes artifacts[slot] and checks it against its
// SHA-256 (fail closed: a present artifact without a valid hash is rejected).
// A missing optional artifact returns nil bytes.
func decodeArtifact(artifacts map[string]any, slot string, required bool) ([]byte, string, error) {
	entry, ok := artifacts[slot].(map[string]any)
	if !ok {
		if required {
			return nil, "", fmt.Errorf("pipeline: payload has no %s artifact", slot)
		}
		return nil, "", nil
	}
	data, err := base64.StdEncoding.DecodeString(stringField(entry, "base64"))
	if err != nil {
		return nil, "", fmt.Errorf("pipeline: %s artifact is not valid base64: %w", slot, err)
	}
	expected := strings.ToLower(strings.TrimSpace(stringField(entry, "sha256")))
	if len(expected) != 64 {
		return nil, "", fmt.Errorf("pipeline: %s artifact has no valid SHA-256", slot)
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != expected {
		return nil, "", fmt.Errorf("pipeline: %s artifact SHA256 mismatch in secure payload", slot)
	}
	return data, expected, nil
}

// findPredictions returns the adaptor's output table: predictions.csv (or
// .tsv) in the predictions folder, else the only .csv / .tsv file there.
func findPredictions(dir string) (string, error) {
	for _, name := range []string{"predictions.csv", "predictions.tsv"} {
		if p := filepath.Join(dir, name); fileExists(p) {
			return p, nil
		}
	}
	var tables []string
	for _, pattern := range []string{"*.csv", "*.tsv"} {
		matches, _ := filepath.Glob(filepath.Join(dir, pattern))
		tables = append(tables, matches...)
	}
	switch len(tables) {
	case 1:
		return tables[0], nil
	case 0:
		return "", &eval.AdaptorOutputError{Msg: "the adaptor did not write predictions/predictions.csv"}
	}
	return "", &eval.AdaptorOutputError{Msg: fmt.Sprintf(
		"the adaptor wrote %d tables to predictions/ and none is named predictions.csv", len(tables))}
}

func fileExists(p string) bool {
	_, err := os.Lstat(p)
	return err == nil
}

// checkPredictions confirms the adaptor produced a usable predictions table.
func checkPredictions(path string) error {
	info, err := os.Lstat(path)
	switch {
	case err != nil:
		return &eval.AdaptorOutputError{Msg: "the adaptor did not write predictions/predictions.csv"}
	case !info.Mode().IsRegular():
		return &eval.AdaptorOutputError{Msg: "predictions.csv is not a regular file"}
	case info.Size() == 0:
		return &eval.AdaptorOutputError{Msg: "predictions.csv is empty"}
	case info.Size() > maxPredictionsBytes:
		return &eval.AdaptorOutputError{Msg: fmt.Sprintf("predictions.csv is larger than %d bytes", maxPredictionsBytes)}
	}
	return nil
}

// readResults loads the evaluator's results.json and returns the whole map,
// the sample count and the metrics object.
func readResults(path string) (map[string]any, int, map[string]any, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, nil, &eval.StageError{Stage: eval.StageEvaluator, ExitCode: -1}
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, maxResultsBytes+1))
	if err != nil {
		return nil, 0, nil, err
	}
	if len(raw) > maxResultsBytes {
		return nil, 0, nil, fmt.Errorf("pipeline: results.json is larger than %d bytes", maxResultsBytes)
	}
	var results map[string]any
	if err := json.Unmarshal(raw, &results); err != nil {
		return nil, 0, nil, fmt.Errorf("pipeline: parse results.json: %w", err)
	}
	metrics, ok := results["metrics"].(map[string]any)
	if !ok {
		return nil, 0, nil, fmt.Errorf("pipeline: results.json has no metrics object")
	}
	n, _ := results["num_samples"].(float64)
	if n < 1 || n != math.Trunc(n) {
		return nil, 0, nil, fmt.Errorf("pipeline: results.json num_samples must be a positive integer")
	}
	return results, int(n), metrics, nil
}

// inferMeta is what infer.py reports about how the model was run.
type inferMeta struct {
	Runtime        string   `json:"runtime"`
	RuntimeVersion string   `json:"runtime_version"`
	Device         string   `json:"device"`
	Providers      []string `json:"providers"`
}

func readInferMeta(path string) inferMeta {
	var meta inferMeta
	raw, err := os.ReadFile(path)
	if err == nil {
		err = json.Unmarshal(raw, &meta)
	}
	if err != nil {
		log.Printf("pipeline: read inference meta.json: %v", err)
	}
	if meta.Providers == nil {
		meta.Providers = []string{}
	}
	return meta
}

func writeJSON(path string, v any) error {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

func roundTo(v float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.Round(v*p) / p
}

func formatTimings(t map[string]float64) string {
	parts := make([]string, 0, len(t))
	for _, k := range []string{"deps", "dataset", eval.StageInfer, eval.StageAdaptor, eval.StageEvaluator} {
		if v, ok := t[k]; ok {
			parts = append(parts, fmt.Sprintf("%s %.1fs", k, v))
		}
	}
	return strings.Join(parts, ", ")
}
