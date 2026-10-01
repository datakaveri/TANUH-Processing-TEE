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

// encrypted lists each name with its manifest, as the UI uploads them.
func encrypted(n ...string) []string {
	var out []string
	for _, x := range n {
		out = append(out, x, x+".manifest.json")
	}
	return out
}

func rels(objs []datasetObject) []string {
	out := make([]string, len(objs))
	for i, o := range objs {
		out[i] = o.Rel
	}
	return out
}

func TestPlanDatasetNestedFolders(t *testing.T) {
	names := append(encrypted(
		"u/ground_truth.csv",
		"u/top.jpg",                     // next to the ground truth
		"u/Suspicious/a.jpg",            // one folder deep
		"u/batch1/Non-Suspicious/a.jpg", // same file name, two folders deep
		"u/subject_00001/IM0001",        // DICOM without an extension
		"u/sub/other.csv",               // a table in a subfolder is not the ground truth
	), "u/notes.txt", "u/dataset.json.enc", // no manifest: not an encrypted object, ignored
		"u/gone.jpg.manifest.json") // manifest whose object is missing: ignored
	data, gt, err := planDataset(names, "u/")
	if err != nil {
		t.Fatal(err)
	}
	want := "Suspicious/a.jpg,batch1/Non-Suspicious/a.jpg,sub/other.csv,subject_00001/IM0001,top.jpg"
	if gt.Rel != "ground_truth.csv" || strings.Join(rels(data), ",") != want {
		t.Fatalf("gt=%s data=%v", gt.Rel, rels(data))
	}
	if data[0].Cipher != "u/Suspicious/a.jpg" || data[0].Manifest != "u/Suspicious/a.jpg.manifest.json" {
		t.Fatalf("object names %+v", data[0])
	}
}

func TestPlanDatasetGroundTruthChoice(t *testing.T) {
	cases := map[string]struct {
		names []string
		want  string
	}{
		"json ground truth":          {encrypted("u/a.jpg", "u/ground_truth.json"), "ground_truth.json"},
		"named among several tables": {encrypted("u/a.jpg", "u/metadata.csv", "u/Ground-Truth.CSV"), "Ground-Truth.CSV"},
		"labels file":                {encrypted("u/a.jpg", "u/stats.json", "u/labels.jsonl"), "labels.jsonl"},
		"the only table":             {encrypted("u/a.dcm", "u/ocs_ground_truth_v2.csv"), "ocs_ground_truth_v2.csv"},
	}
	for name, c := range cases {
		_, gt, err := planDataset(c.names, "u/")
		if err != nil || gt.Rel != c.want {
			t.Errorf("%s: gt=%q err=%v", name, gt.Rel, err)
		}
	}
}

func TestPlanDatasetRejects(t *testing.T) {
	cases := map[string][]string{
		"no ground truth":                 encrypted("u/a.jpg"),
		"ground truth only in subfolder":  encrypted("u/a.jpg", "u/meta/ground_truth.csv"),
		"two tables, none named":          encrypted("u/a.jpg", "u/x.csv", "u/y.json"),
		"two ground truths":               encrypted("u/a.jpg", "u/ground_truth.csv", "u/ground_truth.json"),
		"no data files":                   encrypted("u/ground_truth.csv"),
		"only unusable paths besides gt":  encrypted("u/ground_truth.csv", "u/../x.jpg", "u/a//b.jpg"),
		"manifests without their objects": {"u/a.jpg.manifest.json", "u/ground_truth.csv.manifest.json"},
	}
	for name, names := range cases {
		if _, _, err := planDataset(names, "u/"); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestSafeRel(t *testing.T) {
	for rel, want := range map[string]bool{
		"a.jpg": true, "x/y/a.jpg": true, "a b/c.dcm": true,
		"": false, "../a.jpg": false, "x/../a.jpg": false, "./a.jpg": false, "x//a.jpg": false, "/a.jpg": false, "x/": false,
	} {
		if safeRel(rel) != want {
			t.Errorf("safeRel(%q) = %v", rel, !want)
		}
	}
}

func TestFileKind(t *testing.T) {
	dicom := append(make([]byte, 128), []byte("DICM....")...)
	cases := []struct {
		head []byte
		rel  string
		want string
	}{
		{dicom, "IM0001", "dicom"}, // no extension: found by content
		{dicom, "scan.jpg", "dicom"},
		{[]byte("\x00\x00\x00raw"), "old.dcm", "dicom"}, // DICOM without preamble: by extension
		{[]byte{0xFF, 0xD8, 0xFF, 0xE0}, "a", "image"},
		{[]byte("\x89PNG\r\n"), "a.png", "image"},
		{[]byte("II*\x00...."), "a.tif", "image"},
		{[]byte("BM......"), "a.bmp", "image"},
		{[]byte("RIFF\x00\x00\x00\x00WEBPVP8 "), "a.webp", "image"},
		{[]byte("hello"), "notes.bin", "unknown"},
	}
	for _, c := range cases {
		if got := fileKind(c.head, c.rel); got != c.want {
			t.Errorf("fileKind(%q) = %s, want %s", c.rel, got, c.want)
		}
	}
	if modalityOf(map[string]int{"image": 3}) != "image" || modalityOf(map[string]int{"image": 1, "dicom": 1}) != "mixed" ||
		modalityOf(map[string]int{"dicom": 2, "unknown": 1}) != "dicom" {
		t.Fatal("modalityOf")
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

func TestFindPredictions(t *testing.T) {
	write := func(dir string, names ...string) {
		for _, n := range names {
			os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644) //nolint:errcheck
		}
	}
	cases := map[string]struct {
		files []string
		want  string // "" = error
	}{
		"standard name":            {[]string{"predictions.csv", "debug.csv"}, "predictions.csv"},
		"tsv":                      {[]string{"predictions.tsv"}, "predictions.tsv"},
		"the only table":           {[]string{"preds.csv", "log.txt"}, "preds.csv"},
		"nothing":                  {[]string{"log.txt"}, ""},
		"several tables, no names": {[]string{"a.csv", "b.csv"}, ""},
	}
	for name, c := range cases {
		dir := t.TempDir()
		write(dir, c.files...)
		got, err := findPredictions(dir)
		var ao *eval.AdaptorOutputError
		switch {
		case c.want == "" && !errors.As(err, &ao):
			t.Errorf("%s: err = %v, want an adaptor output error", name, err)
		case c.want != "" && (err != nil || filepath.Base(got) != c.want):
			t.Errorf("%s: got %q, %v", name, got, err)
		}
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
