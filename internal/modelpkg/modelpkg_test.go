package modelpkg

import (
	"archive/zip"
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
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
	if err := Place(ONNX, []byte{0x08, 0x0a, 0x12}, []byte("w"), nil, dir); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"model.onnx", "model.onnx.data"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Fatalf("%s not written", f)
		}
	}
	if err := Place(ONNX, []byte("PK\x03\x04 a zip"), nil, nil, t.TempDir()); !isFormatErr(err) {
		t.Fatalf("non-ONNX accepted: %v", err)
	}
}

func TestInputSpec(t *testing.T) {
	dir := t.TempDir()
	if err := Place(ONNX, []byte{0x08, 0x0a}, nil, []byte(`{"input_size": [224, 224]}`), dir); err != nil {
		t.Fatal(err)
	}
	if raw, _ := os.ReadFile(filepath.Join(dir, "input_spec.json")); string(raw) != `{"input_size": [224, 224]}` {
		t.Fatalf("input_spec.json = %q", raw)
	}
	cases := map[string]struct {
		format string
		model  []byte
		spec   []byte
	}{
		"with Hugging Face": {HuggingFace, makeZip(t, hfGood...), []byte(`{"input_size": 224}`)},
		"not an object":     {ONNX, []byte{0x08}, []byte(`[224, 224]`)},
		"not JSON":          {ONNX, []byte{0x08}, []byte(`size=224`)},
		"too large":         {ONNX, []byte{0x08}, []byte(`{"x": "` + strings.Repeat("a", 70<<10) + `"}`)},
	}
	for name, c := range cases {
		if err := Place(c.format, c.model, nil, c.spec, t.TempDir()); !isFormatErr(err) {
			t.Errorf("%s: err = %v, want FormatError", name, err)
		}
	}
}

func TestWeightsOnlyForONNX(t *testing.T) {
	ts := makeZip(t, entry{name: "m/code/__torch__.py"}, entry{name: "m/constants.pkl"}, entry{name: "m/extra/tanuh.json"})
	if err := Place(TorchScript, ts, []byte("w"), nil, t.TempDir()); !isFormatErr(err) {
		t.Fatalf("weights with TorchScript accepted: %v", err)
	}
}

func TestTorchScript(t *testing.T) {
	code := []entry{{name: "m/code/__torch__.py"}, {name: "m/constants.pkl"}}
	plain := makeZip(t, code...)
	accepted := map[string]struct {
		model []byte
		spec  []byte
	}{
		"embedded tanuh.json":      {makeZip(t, append(code, entry{name: "m/extra/tanuh.json", body: "{}"})...), nil},
		"embedded input_spec.json": {makeZip(t, append(code, entry{name: "m/extra/input_spec.json", body: "{}"})...), nil},
		"plain + uploaded spec":    {plain, []byte(`{"input_size": 224}`)},
	}
	for name, c := range accepted {
		dir := t.TempDir()
		if err := Place(TorchScript, c.model, nil, c.spec, dir); err != nil {
			t.Errorf("%s: %v", name, err)
		} else if _, err := os.Stat(filepath.Join(dir, "model.pt")); err != nil {
			t.Errorf("%s: model.pt not written", name)
		}
	}
	rejected := map[string][]byte{
		"no input size anywhere": plain,
		"torch.save state dict":  makeZip(t, entry{name: "archive/data.pkl"}, entry{name: "archive/data/0"}),
		"not a zip (raw pickle)": []byte("\x80\x02pickle"),
	}
	for name, data := range rejected {
		if err := Place(TorchScript, data, nil, nil, t.TempDir()); !isFormatErr(err) {
			t.Errorf("%s: err = %v, want FormatError", name, err)
		}
	}
}

// extracted returns the files Place wrote into hf/, name -> content.
func extracted(t *testing.T, data []byte) map[string]string {
	t.Helper()
	dir := t.TempDir()
	if err := Place(HuggingFace, data, nil, nil, dir); err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	ents, _ := os.ReadDir(filepath.Join(dir, "hf"))
	for _, e := range ents {
		raw, _ := os.ReadFile(filepath.Join(dir, "hf", e.Name()))
		out[e.Name()] = string(raw)
	}
	return out
}

func TestHuggingFaceExtracts(t *testing.T) {
	got := extracted(t, makeZip(t, hfGood...))
	if len(got) != 3 || got["model.safetensors"] != "weights" {
		t.Fatalf("extracted %v", got)
	}
}

func TestHuggingFaceFindsTheModelFolder(t *testing.T) {
	nest := func(prefix string) []entry {
		var out []entry
		for _, e := range hfGood {
			out = append(out, entry{name: prefix + e.name, body: e.body})
		}
		return out
	}
	cases := map[string][]entry{
		"one wrapper folder": nest("my-model/"),
		"deeper":             nest("outputs/final/"),
		"with checkpoints":   append(nest("run/"), nest("run/checkpoint-500/")...), // shallowest wins
		"macOS zip":          append(nest("my-model/"), entry{name: "__MACOSX/my-model/._config.json", body: "junk"}),
	}
	for name, entries := range cases {
		if got := extracted(t, makeZip(t, entries...)); len(got) != 3 {
			t.Errorf("%s: extracted %v", name, got)
		}
	}
}

func TestHuggingFaceIgnoresWhatItDoesNotLoad(t *testing.T) {
	data := makeZip(t, append(append([]entry{}, hfGood...),
		entry{name: "training_args.bin", body: "pickle"}, // Trainer output
		entry{name: "optimizer.pt", body: "pickle"},
		entry{name: "modeling_custom.py", body: "import os"},
		entry{name: "README.md", body: "# model"},
		entry{name: "model.safetensors.index.json", body: `{"weight_map": {}}`},
		entry{name: "link", body: "/etc/passwd", mode: os.ModeSymlink | 0o777},
		entry{name: "../../escape.json", body: "x"},
	)...)
	got := extracted(t, data)
	want := []string{"config.json", "model.safetensors", "model.safetensors.index.json", "preprocessor_config.json"}
	if len(got) != len(want) {
		t.Fatalf("extracted %v, want only %v", got, want)
	}
	for _, w := range want {
		if _, ok := got[w]; !ok {
			t.Fatalf("%s missing from %v", w, got)
		}
	}
}

func TestHuggingFaceDropsAutoMap(t *testing.T) {
	got := extracted(t, makeZip(t,
		entry{name: "config.json", body: `{"model_type":"vit","auto_map":{"AutoModel":"evil.Model"}}`},
		entry{name: "preprocessor_config.json", body: `{"auto_map":{"AutoImageProcessor":"evil.P"},"do_resize":true}`},
		hfGood[2]))
	if strings.Contains(got["config.json"], "auto_map") || strings.Contains(got["preprocessor_config.json"], "auto_map") ||
		!strings.Contains(got["config.json"], `"vit"`) || !strings.Contains(got["preprocessor_config.json"], "do_resize") {
		t.Fatalf("configs %v", got)
	}
}

func TestHuggingFaceRejects(t *testing.T) {
	cases := map[string][]byte{
		"pickle weights only": makeZip(t, hfGood[0], hfGood[1], entry{name: "pytorch_model.bin", body: "x"}),
		"no preprocessor":     makeZip(t, hfGood[0], hfGood[2]),
		"no safetensors":      makeZip(t, hfGood[0], hfGood[1]),
		"no config":           makeZip(t, hfGood[1], hfGood[2]),
		"two models":          makeZip(t, entry{name: "a/config.json", body: hfGood[0].body}, entry{name: "b/config.json", body: hfGood[0].body}, hfGood[1], hfGood[2]),
		"not a zip":           []byte("config.json"),
		"symlinked weights":   makeZip(t, hfGood[0], hfGood[1], entry{name: "model.safetensors", body: "/etc/passwd", mode: os.ModeSymlink | 0o777}),
		"no model_type":       makeZip(t, entry{name: "config.json", body: `{}`}, hfGood[1], hfGood[2]),
		"bad config JSON":     makeZip(t, entry{name: "config.json", body: `{`}, hfGood[1], hfGood[2]),
	}
	for name, data := range cases {
		if err := Place(HuggingFace, data, nil, nil, t.TempDir()); !isFormatErr(err) {
			t.Errorf("%s: err = %v, want FormatError", name, err)
		}
	}
	if err := Place(HuggingFace, makeZip(t, hfGood...), []byte("w"), nil, t.TempDir()); !isFormatErr(err) {
		t.Errorf("weights with HF accepted: %v", err)
	}
}

func TestUnknownFormat(t *testing.T) {
	if err := Place("keras", []byte("x"), nil, nil, t.TempDir()); !isFormatErr(err) {
		t.Fatalf("unknown format accepted: %v", err)
	}
}
