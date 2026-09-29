// Package modelpkg lays a submitted model out on disk for the inference step
// and checks that the files really are the declared format before anything
// loads them.
//
//	onnx         <dir>/model.onnx (+ <dir>/model.onnx.data for external weights)
//	torchscript  <dir>/model.pt   (TorchScript archive carrying extra/tanuh.json)
//	huggingface  <dir>/hf/        (config.json, *.safetensors, preprocessor_config.json)
//
// tools/infer/infer.py repeats the format checks when it loads the model; these
// run first so a wrong or unsafe upload fails as a model-format error without
// starting Python.
package modelpkg

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// Formats.
const (
	ONNX        = "onnx"
	TorchScript = "torchscript"
	HuggingFace = "huggingface"
)

// Limits for the Hugging Face zip.
const (
	maxZipEntries        = 10000
	maxZipUncompressed   = 8 << 30
	hfFolder             = "hf"
	torchscriptSpecEntry = "/extra/tanuh.json"
)

// Hugging Face packages must be data only: files that can carry code are refused.
var hfForbiddenSuffixes = map[string]bool{
	".py": true, ".pyc": true, ".bin": true, ".pt": true, ".pth": true, ".pkl": true,
	".pickle": true, ".ckpt": true, ".h5": true, ".joblib": true,
}

// FormatError means the upload is not a valid package of its declared format.
type FormatError struct{ Msg string }

func (e *FormatError) Error() string { return "model format: " + e.Msg }

func formatErr(format string, a ...any) error { return &FormatError{Msg: fmt.Sprintf(format, a...)} }

// Place writes the model into dir for its format. weights is only valid for ONNX.
func Place(format string, model, weights []byte, dir string) error {
	if len(model) == 0 {
		return formatErr("model file is empty")
	}
	if weights != nil && format != ONNX {
		return formatErr("a weights file is only used with ONNX models")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	switch format {
	case ONNX:
		// An ONNX file is a protobuf ModelProto; exporters write field 1
		// (ir_version, a varint) first, i.e. the tag byte 0x08.
		if model[0] != 0x08 {
			return formatErr("model is not an ONNX file")
		}
		if err := os.WriteFile(filepath.Join(dir, "model.onnx"), model, 0o644); err != nil {
			return err
		}
		if weights != nil {
			return os.WriteFile(filepath.Join(dir, "model.onnx.data"), weights, 0o644)
		}
		return nil
	case TorchScript:
		if err := checkTorchScript(model); err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(dir, "model.pt"), model, 0o644)
	case HuggingFace:
		return extractHuggingFace(model, filepath.Join(dir, hfFolder))
	default:
		return formatErr("unsupported model format %q", format)
	}
}

func checkTorchScript(model []byte) error {
	zr, err := zip.NewReader(bytes.NewReader(model), int64(len(model)))
	if err != nil {
		return formatErr("model is not a TorchScript archive (save it with torch.jit.save)")
	}
	var code, constants, spec bool
	for _, f := range zr.File {
		code = code || strings.Contains(f.Name, "/code/")
		constants = constants || strings.HasSuffix(f.Name, "constants.pkl")
		spec = spec || strings.HasSuffix(f.Name, torchscriptSpecEntry)
	}
	if !code || !constants {
		return formatErr("model is not a TorchScript archive (save it with torch.jit.save, not torch.save)")
	}
	if !spec {
		return formatErr("TorchScript model has no tanuh.json; save it with torch.jit.save(m, f, _extra_files={\"tanuh.json\": ...})")
	}
	return nil
}

// extractHuggingFace validates and extracts a save_pretrained() zip. A zip of
// the folder itself (one top-level directory) is accepted too.
func extractHuggingFace(data []byte, dest string) error {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return formatErr("Hugging Face model must be a .zip of a save_pretrained() folder")
	}
	if len(zr.File) > maxZipEntries {
		return formatErr("Hugging Face zip has more than %d entries", maxZipEntries)
	}
	prefix := commonTopDir(zr.File)

	var total uint64
	var hasConfig, hasPreprocessor, hasSafetensors bool
	var configFile *zip.File
	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}
		name := strings.TrimPrefix(f.Name, prefix)
		if hfForbiddenSuffixes[strings.ToLower(path.Ext(name))] {
			return formatErr("Hugging Face package contains %s, which can carry code; use safetensors weights only", path.Base(name))
		}
		total += f.UncompressedSize64
		switch {
		case name == "config.json":
			hasConfig, configFile = true, f
		case name == "preprocessor_config.json":
			hasPreprocessor = true
		case strings.HasSuffix(name, ".safetensors"):
			hasSafetensors = true
		}
	}
	if total > maxZipUncompressed {
		return formatErr("Hugging Face zip expands beyond %d bytes", uint64(maxZipUncompressed))
	}
	if !hasConfig || !hasPreprocessor || !hasSafetensors {
		return formatErr("Hugging Face package needs config.json, preprocessor_config.json and *.safetensors weights")
	}
	if err := checkHFConfig(configFile); err != nil {
		return err
	}
	return extractZip(zr, dest, prefix)
}

func checkHFConfig(f *zip.File) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	var cfg map[string]any
	if err := json.NewDecoder(io.LimitReader(rc, 1<<20)).Decode(&cfg); err != nil {
		return formatErr("config.json is not valid JSON")
	}
	if _, ok := cfg["auto_map"]; ok {
		return formatErr("config.json uses auto_map (custom model code), which is not allowed")
	}
	if mt, _ := cfg["model_type"].(string); mt == "" {
		return formatErr("config.json has no model_type")
	}
	return nil
}

// commonTopDir returns "dir/" when every entry lives under one top-level
// directory that holds config.json, otherwise "".
func commonTopDir(files []*zip.File) string {
	var top string
	for _, f := range files {
		first, _, found := strings.Cut(f.Name, "/")
		if !found {
			return "" // a file at the root
		}
		if top == "" {
			top = first
		} else if top != first {
			return ""
		}
	}
	if top == "" {
		return ""
	}
	for _, f := range files {
		if f.Name == top+"/config.json" {
			return top + "/"
		}
	}
	return ""
}

// extractZip writes the zip's regular files under destDir, stripping prefix,
// refusing entries that would escape destDir and capping each file's actual
// size (declared sizes in a zip can lie).
func extractZip(zr *zip.Reader, destDir, prefix string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	cleanDest := filepath.Clean(destDir)
	var written int64
	for _, f := range zr.File {
		name := strings.TrimPrefix(f.Name, prefix)
		if name == "" || f.FileInfo().IsDir() {
			continue
		}
		if !f.Mode().IsRegular() {
			return formatErr("zip entry %q is not a regular file", f.Name)
		}
		target := filepath.Join(cleanDest, filepath.FromSlash(name))
		if !strings.HasPrefix(filepath.Clean(target), cleanDest+string(os.PathSeparator)) {
			return formatErr("zip entry escapes destination: %q", f.Name)
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		n, err := extractFile(f, target, maxZipUncompressed-written)
		if err != nil {
			return fmt.Errorf("extract %s: %w", f.Name, err)
		}
		written += n
	}
	return nil
}

var errTooLarge = errors.New("zip expands beyond the size limit")

func extractFile(f *zip.File, target string, remaining int64) (int64, error) {
	src, err := f.Open()
	if err != nil {
		return 0, err
	}
	defer src.Close()
	dst, err := os.Create(target)
	if err != nil {
		return 0, err
	}
	defer dst.Close()
	n, err := io.Copy(dst, io.LimitReader(src, remaining+1))
	if err != nil {
		return n, err
	}
	if n > remaining {
		return n, &FormatError{Msg: errTooLarge.Error()}
	}
	return n, nil
}
