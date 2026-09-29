package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/eval"
	"github.com/datakaveri/tanuh-processing-tee/internal/modelpkg"
)

func artifact(data []byte) map[string]any {
	sum := sha256.Sum256(data)
	return map[string]any{"sha256": hex.EncodeToString(sum[:]), "base64": base64.StdEncoding.EncodeToString(data)}
}

func v2Payload(format string, artifacts map[string]any) map[string]any {
	return map[string]any{
		"payload_version": float64(2), // numbers decode as float64 from JSON
		"job_id":          "job-abc123",
		"dataset_id":      "d8c50b6b-af58-4f9d-9af8-4d42bfb1c4bb",
		"model_format":    format,
		"artifacts":       artifacts,
	}
}

func testManager(t *testing.T) *Manager {
	return &Manager{cfg: config.Config{BaseDir: t.TempDir()}}
}

var onnxModel = []byte{0x08, 0x0a, 0x12, 0x07}

func TestMaterializeONNX(t *testing.T) {
	m := testManager(t)
	job, err := m.materialize(v2Payload("onnx", map[string]any{
		"model":   artifact(onnxModel),
		"weights": artifact([]byte("weights")),
		"adaptor": artifact([]byte("print('adapt')")),
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{filepath.Join(job.modelDir, "model.onnx"), filepath.Join(job.modelDir, "model.onnx.data"), job.adaptorPath} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s not written", p)
		}
	}
	if job.modelFormat != "onnx" || len(job.modelSHA256) != 64 || len(job.adaptorSHA256) != 64 {
		t.Fatalf("job = %+v", job)
	}
}

func TestMaterializeRejects(t *testing.T) {
	good := func() map[string]any {
		return map[string]any{"model": artifact(onnxModel), "adaptor": artifact([]byte("a"))}
	}
	tampered := good()
	tampered["model"].(map[string]any)["sha256"] = strings.Repeat("0", 64)
	noHash := good()
	delete(noHash["adaptor"].(map[string]any), "sha256")
	noAdaptor := good()
	delete(noAdaptor, "adaptor")

	cases := map[string]map[string]any{
		"old payload (v1)":          {"payload_version": nil, "job_id": "j", "dataset_id": "d", "model_onnx_base64": "x"},
		"missing adaptor":           v2Payload("onnx", noAdaptor),
		"tampered model":            v2Payload("onnx", tampered),
		"adaptor without hash":      v2Payload("onnx", noHash),
		"unknown format":            v2Payload("keras", good()),
		"path in job_id":            func() map[string]any { p := v2Payload("onnx", good()); p["job_id"] = "../evil"; return p }(),
		"onnx bytes as torchscript": v2Payload("torchscript", good()),
	}
	for name, payload := range cases {
		if _, err := testManager(t).materialize(payload); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	var fe *modelpkg.FormatError
	if _, err := testManager(t).materialize(v2Payload("torchscript", good())); !errors.As(err, &fe) {
		t.Errorf("format mismatch err = %v, want a model-format error", err)
	}
}

func TestPlanDataset(t *testing.T) {
	names := []string{
		"u/a.jpg", "u/a.jpg.manifest.json",
		"u/b.jpg", "u/b.jpg.manifest.json",
		"u/ground_truth.csv", "u/ground_truth.csv.manifest.json",
		"u/evaluation_script.py", "u/evaluation_script.py.manifest.json", // legacy upload: skipped, never decrypted
		"u/notes.txt", // plaintext, no manifest: ignored
	}
	data, gt, modality, err := planDataset(names)
	if err != nil {
		t.Fatal(err)
	}
	if modality != "image" || len(data) != 2 || data[0].Name != "a.jpg" || gt.Name != "ground_truth.csv" {
		t.Fatalf("data=%+v gt=%+v modality=%s", data, gt, modality)
	}
	for _, d := range data {
		if strings.HasSuffix(d.Name, ".py") {
			t.Fatal("python file planned for decryption")
		}
	}
}

func TestPlanDatasetGroundTruthFallbackAndDICOM(t *testing.T) {
	_, gt, modality, err := planDataset([]string{
		"u/s1.dcm", "u/s1.dcm.manifest.json",
		"u/labels.csv", "u/labels.csv.manifest.json", // the only CSV, uploaded under its own name
	})
	if err != nil || gt.Name != "labels.csv" || modality != "dicom" {
		t.Fatalf("gt=%+v modality=%s err=%v", gt, modality, err)
	}
}

func TestPlanDatasetRejects(t *testing.T) {
	m := func(n ...string) []string {
		var out []string
		for _, x := range n {
			out = append(out, x, x+".manifest.json")
		}
		return out
	}
	cases := map[string][]string{
		"no ground truth":           m("u/a.jpg"),
		"two csvs, no ground_truth": m("u/a.jpg", "u/x.csv", "u/y.csv"),
		"mixed image and dicom":     m("u/a.jpg", "u/b.dcm", "u/ground_truth.csv"),
		"duplicate names":           m("u/a.jpg", "u/sub/a.jpg", "u/ground_truth.csv"),
		"no data files":             m("u/ground_truth.csv"),
		"manifest without object":   {"u/a.jpg.manifest.json", "u/ground_truth.csv", "u/ground_truth.csv.manifest.json"},
	}
	for name, names := range cases {
		if _, _, _, err := planDataset(names); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestRunParallelProcessesEverything(t *testing.T) {
	var n atomic.Int64
	items := make([]int, 240)
	if err := runParallel(context.Background(), 16, items, func(context.Context, int) error {
		n.Add(1)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 240 {
		t.Fatalf("processed %d of 240", n.Load())
	}
}

func TestRunParallelFirstErrorCancelsTheRest(t *testing.T) {
	var started atomic.Int64
	items := make([]int, 1000)
	for i := range items {
		items[i] = i
	}
	boom := errors.New("KMS unwrap failed")
	err := runParallel(context.Background(), 4, items, func(ctx context.Context, i int) error {
		started.Add(1)
		if i == 3 {
			return boom
		}
		<-ctx.Done() // a slow download, cut short by the cancellation
		return ctx.Err()
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the first error", err)
	}
	if started.Load() > 50 {
		t.Fatalf("%d items started after the failure; the rest should be skipped", started.Load())
	}
}

func TestCheckPredictions(t *testing.T) {
	dir := t.TempDir()
	var ao *eval.AdaptorOutputError
	if err := checkPredictions(filepath.Join(dir, "predictions.csv")); !errors.As(err, &ao) {
		t.Fatalf("missing file err = %v", err)
	}
	empty := filepath.Join(dir, "empty.csv")
	os.WriteFile(empty, nil, 0o644) //nolint:errcheck
	if err := checkPredictions(empty); !errors.As(err, &ao) {
		t.Fatalf("empty file err = %v", err)
	}
	ok := filepath.Join(dir, "ok.csv")
	os.WriteFile(ok, []byte("file,label,score\na.jpg,1,0.9\n"), 0o644) //nolint:errcheck
	if err := checkPredictions(ok); err != nil {
		t.Fatal(err)
	}
}

func TestReadResults(t *testing.T) {
	dir := t.TempDir()
	write := func(body string) string {
		p := filepath.Join(dir, "results.json")
		os.WriteFile(p, []byte(body), 0o644) //nolint:errcheck
		return p
	}
	_, n, metrics, err := readResults(write(`{"num_samples": 127, "metrics": {"f2": 0.99}}`))
	if err != nil || n != 127 || metrics["f2"] != 0.99 {
		t.Fatalf("n=%d metrics=%v err=%v", n, metrics, err)
	}
	for _, bad := range []string{`{"num_samples": 0, "metrics": {}}`, `{"num_samples": 5}`, `not json`} {
		if _, _, _, err := readResults(write(bad)); err == nil {
			t.Errorf("accepted %s", bad)
		}
	}
}
