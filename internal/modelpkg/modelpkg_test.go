package modelpkg

import (
	"archive/zip"
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type entry struct {
	name string
	body string
	mode os.FileMode
}

func makeZip(t *testing.T, entries ...entry) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, e := range entries {
		h := &zip.FileHeader{Name: e.name, Method: zip.Deflate}
		if e.mode != 0 {
			h.SetMode(e.mode)
		}
		w, err := zw.CreateHeader(h)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(e.body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func isFormatErr(err error) bool {
	var fe *FormatError
	return errors.As(err, &fe)
}

var hfGood = []entry{
	{name: "config.json", body: `{"model_type":"mobilevitv2","architectures":["MobileViTV2ForImageClassification"]}`},
	{name: "preprocessor_config.json", body: `{}`},
	{name: "model.safetensors", body: "weights"},
}

func TestONNX(t *testing.T) {
	dir := t.TempDir()
	if err := Place(ONNX, []byte{0x08, 0x0a, 0x12}, []byte("w"), dir); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"model.onnx", "model.onnx.data"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Fatalf("%s not written", f)
		}
	}
	if err := Place(ONNX, []byte("PK\x03\x04 a zip"), nil, t.TempDir()); !isFormatErr(err) {
		t.Fatalf("non-ONNX accepted: %v", err)
	}
}

func TestWeightsOnlyForONNX(t *testing.T) {
	ts := makeZip(t, entry{name: "m/code/__torch__.py"}, entry{name: "m/constants.pkl"}, entry{name: "m/extra/tanuh.json"})
	if err := Place(TorchScript, ts, []byte("w"), t.TempDir()); !isFormatErr(err) {
		t.Fatalf("weights with TorchScript accepted: %v", err)
	}
}

func TestTorchScript(t *testing.T) {
	good := makeZip(t, entry{name: "m/code/__torch__.py"}, entry{name: "m/constants.pkl"}, entry{name: "m/extra/tanuh.json", body: "{}"})
	dir := t.TempDir()
	if err := Place(TorchScript, good, nil, dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "model.pt")); err != nil {
		t.Fatal("model.pt not written")
	}
	cases := map[string][]byte{
		"no tanuh.json":          makeZip(t, entry{name: "m/code/__torch__.py"}, entry{name: "m/constants.pkl"}),
		"torch.save state dict":  makeZip(t, entry{name: "archive/data.pkl"}, entry{name: "archive/data/0"}),
		"not a zip (raw pickle)": []byte("\x80\x02pickle"),
	}
	for name, data := range cases {
		if err := Place(TorchScript, data, nil, t.TempDir()); !isFormatErr(err) {
			t.Errorf("%s: err = %v, want FormatError", name, err)
		}
	}
}

func TestHuggingFaceExtracts(t *testing.T) {
	dir := t.TempDir()
	if err := Place(HuggingFace, makeZip(t, hfGood...), nil, dir); err != nil {
		t.Fatal(err)
	}
	for _, e := range hfGood {
		if _, err := os.Stat(filepath.Join(dir, "hf", e.name)); err != nil {
			t.Fatalf("%s not extracted", e.name)
		}
	}
}

func TestHuggingFaceFolderZipped(t *testing.T) {
	var wrapped []entry
	for _, e := range hfGood {
		wrapped = append(wrapped, entry{name: "my-model/" + e.name, body: e.body})
	}
	dir := t.TempDir()
	if err := Place(HuggingFace, makeZip(t, wrapped...), nil, dir); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "hf", "config.json")); err != nil {
		t.Fatal("wrapper folder not stripped")
	}
}

func TestHuggingFaceRejects(t *testing.T) {
	with := func(extra ...entry) []byte { return makeZip(t, append(append([]entry{}, hfGood...), extra...)...) }
	cases := map[string][]byte{
		"pickle weights":  with(entry{name: "pytorch_model.bin", body: "x"}),
		"python code":     with(entry{name: "modeling_custom.py", body: "import os"}),
		"auto_map":        makeZip(t, entry{name: "config.json", body: `{"model_type":"x","auto_map":{"AutoModel":"m.M"}}`}, hfGood[1], hfGood[2]),
		"no preprocessor": makeZip(t, hfGood[0], hfGood[2]),
		"no safetensors":  makeZip(t, hfGood[0], hfGood[1]),
		"not a zip":       []byte("config.json"),
		"zip slip":        with(entry{name: "../../escape.json", body: "x"}),
		"symlink entry":   with(entry{name: "link", body: "/etc/passwd", mode: os.ModeSymlink | 0o777}),
		"no model_type":   makeZip(t, entry{name: "config.json", body: `{}`}, hfGood[1], hfGood[2]),
		"bad config JSON": makeZip(t, entry{name: "config.json", body: `{`}, hfGood[1], hfGood[2]),
	}
	for name, data := range cases {
		if err := Place(HuggingFace, data, nil, t.TempDir()); !isFormatErr(err) {
			t.Errorf("%s: err = %v, want FormatError", name, err)
		}
	}
	if err := Place(HuggingFace, makeZip(t, hfGood...), []byte("w"), t.TempDir()); !isFormatErr(err) {
		t.Errorf("weights with HF accepted: %v", err)
	}
}

func TestUnknownFormat(t *testing.T) {
	if err := Place("keras", []byte("x"), nil, t.TempDir()); !isFormatErr(err) {
		t.Fatalf("unknown format accepted: %v", err)
	}
}
