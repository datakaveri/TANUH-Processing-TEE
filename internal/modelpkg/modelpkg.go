// Package modelpkg lays a submitted model out on disk for the inference step
// and checks that the files really are the declared format before anything
// loads them.
//
//	onnx         <dir>/model.onnx (+ <dir>/model.onnx.data for external weights)
//	torchscript  <dir>/model.pt   (TorchScript archive saved with torch.jit.save)
//	huggingface  <dir>/hf/        (config.json, *.safetensors, preprocessor_config.json)
//	             <dir>/input_spec.json when one was uploaded (ONNX / TorchScript)
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
	"log"
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
	maxZipEntries      = 10000
	maxZipUncompressed = 8 << 30
	maxInputSpecBytes  = 64 << 10
	hfFolder           = "hf"
	inputSpecFile      = "input_spec.json"
)

// torchscriptSpecEntries are the files a TorchScript archive may carry its
// input size in (torch.jit.save(..., _extra_files={name: ...})).
var torchscriptSpecEntries = []string{"/extra/tanuh.json", "/extra/input_spec.json"}

// Weight formats that load through pickle, i.e. can run code. A Hugging Face
// package that has only these instead of safetensors is refused.
var pickleWeightExts = map[string]bool{".bin": true, ".pt": true, ".pth": true, ".pkl": true,
	".pickle": true, ".ckpt": true, ".h5": true, ".msgpack": true}

// FormatError means the upload is not a valid package of its declared format.
type FormatError struct{ Msg string }

func (e *FormatError) Error() string { return "model format: " + e.Msg }

func formatErr(format string, a ...any) error { return &FormatError{Msg: fmt.Sprintf(format, a...)} }

// Place writes the model into dir for its format. weights is only valid for
// ONNX; inputSpec (an optional input_spec.json) for ONNX and TorchScript.
func Place(format string, model, weights, inputSpec []byte, dir string) error {
	if len(model) == 0 {
		return formatErr("model file is empty")
	}
	if weights != nil && format != ONNX {
		return formatErr("a weights file is only used with ONNX models")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if inputSpec != nil {
		if err := placeInputSpec(format, inputSpec, dir); err != nil {
			return err
		}
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
		if err := checkTorchScript(model, inputSpec != nil); err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(dir, "model.pt"), model, 0o644)
	case HuggingFace:
		return extractHuggingFace(model, filepath.Join(dir, hfFolder))
	default:
		return formatErr("unsupported model format %q", format)
	}
}

// placeInputSpec writes an uploaded input_spec.json next to the model after
// checking it is a small JSON object; infer.py reads its keys.
func placeInputSpec(format string, spec []byte, dir string) error {
	if format != ONNX && format != TorchScript {
		return formatErr("an input spec is only used with ONNX and TorchScript models " +
			"(Hugging Face models use their preprocessor_config.json)")
	}
	if len(spec) > maxInputSpecBytes {
		return formatErr("input_spec.json is larger than %d bytes", maxInputSpecBytes)
	}
	var obj map[string]any
	if err := json.Unmarshal(bytes.TrimPrefix(spec, []byte("\xef\xbb\xbf")), &obj); err != nil {
		return formatErr("input_spec.json must be a JSON object: %v", err)
	}
	return os.WriteFile(filepath.Join(dir, inputSpecFile), spec, 0o644)
}

// checkTorchScript confirms a torch.jit.save archive. TorchScript records no
// input size, so it must be embedded in the archive or uploaded alongside.
func checkTorchScript(model []byte, specUploaded bool) error {
	zr, err := zip.NewReader(bytes.NewReader(model), int64(len(model)))
	if err != nil {
		return formatErr("model is not a TorchScript archive (save it with torch.jit.save)")
	}
	var code, constants, spec bool
	for _, f := range zr.File {
		code = code || strings.Contains(f.Name, "/code/")
		constants = constants || strings.HasSuffix(f.Name, "constants.pkl")
		for _, e := range torchscriptSpecEntries {
			spec = spec || strings.HasSuffix(f.Name, e)
		}
	}
	if !code || !constants {
		return formatErr("model is not a TorchScript archive (save it with torch.jit.save, not torch.save)")
	}
	if !spec && !specUploaded {
		return formatErr("the TorchScript model's input size is unknown: give it on the submit form, " +
			"or embed it with torch.jit.save(m, f, _extra_files={\"tanuh.json\": '{\"input_size\": [224, 224]}'})")
	}
	return nil
}

// isJunk reports zip entries archivers add that are never part of a model:
// macOS resource forks and folder metadata.
func isJunk(name string) bool {
	base := path.Base(name)
	return strings.HasPrefix(name, "__MACOSX/") || strings.Contains(name, "/__MACOSX/") ||
		base == ".DS_Store" || strings.HasPrefix(base, "._") || base == "Thumbs.db"
}

// hfNeeded reports whether a file in the model folder is one transformers
// loads: the config, the image processor config, and safetensors weights
// (single file or shards + index). Anything else in the zip — READMEs,
// training_args.bin, optimizer state, custom code — is left out, so it can
// never be loaded.
func hfNeeded(base string) bool {
	return base == "config.json" || base == "preprocessor_config.json" ||
		strings.HasSuffix(base, ".safetensors") || strings.HasSuffix(base, ".safetensors.index.json")
}

// extractHuggingFace finds the save_pretrained() folder in a zip (the
// shallowest folder holding config.json — at the root, inside one wrapper
// folder, or deeper) and extracts only the files transformers needs from it.
func extractHuggingFace(data []byte, dest string) error {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return formatErr("Hugging Face model must be a .zip of a save_pretrained() folder")
	}
	if len(zr.File) > maxZipEntries {
		return formatErr("Hugging Face zip has more than %d entries", maxZipEntries)
	}
	root, err := hfRoot(zr.File)
	if err != nil {
		return err
	}

	var keep []*zip.File
	var total uint64
	var hasPreprocessor, hasSafetensors bool
	var pickled []string
	ignored := 0
	for _, f := range zr.File {
		if f.FileInfo().IsDir() || isJunk(f.Name) {
			continue
		}
		base := path.Base(f.Name)
		if path.Dir(f.Name) != root || !hfNeeded(base) {
			if path.Dir(f.Name) == root && pickleWeightExts[strings.ToLower(path.Ext(base))] {
				pickled = append(pickled, base)
			}
			ignored++
			continue
		}
		if !f.Mode().IsRegular() {
			return formatErr("zip entry %q is not a regular file", f.Name)
		}
		hasPreprocessor = hasPreprocessor || base == "preprocessor_config.json"
		hasSafetensors = hasSafetensors || strings.HasSuffix(base, ".safetensors")
		total += f.UncompressedSize64
		keep = append(keep, f)
	}
	switch {
	case !hasSafetensors && len(pickled) > 0:
		return formatErr("the Hugging Face model's weights are %s, which load via pickle and can run code; "+
			"save them as safetensors: model.save_pretrained(dir, safe_serialization=True)", pickled[0])
	case !hasSafetensors:
		return formatErr("the Hugging Face package has no *.safetensors weights next to config.json")
	case !hasPreprocessor:
		return formatErr("the Hugging Face package has no preprocessor_config.json; save it with " +
			"image_processor.save_pretrained(dir) — it is the model's preprocessing")
	case total > maxZipUncompressed:
		return formatErr("Hugging Face model expands beyond %d bytes", uint64(maxZipUncompressed))
	}
	if ignored > 0 {
		log.Printf("modelpkg: Hugging Face zip: using %d file(s) from %q, ignoring %d other(s)", len(keep), root, ignored)
	}

	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	var written int64
	for _, f := range keep {
		base := path.Base(f.Name) // flat: nothing can land outside dest
		target := filepath.Join(dest, base)
		n, err := extractFile(f, target, maxZipUncompressed-written)
		if err != nil {
			return fmt.Errorf("extract %s: %w", f.Name, err)
		}
		written += n
		if base == "config.json" || base == "preprocessor_config.json" {
			if err := sanitizeHFConfig(target, base == "config.json"); err != nil {
				return err
			}
		}
	}
	return nil
}

// hfRoot returns the zip folder ("." for the top level) that holds the model:
// the shallowest one with a config.json.
func hfRoot(files []*zip.File) (string, error) {
	best, bestDepth, count := "", -1, 0
	for _, f := range files {
		if f.FileInfo().IsDir() || isJunk(f.Name) || path.Base(f.Name) != "config.json" {
			continue
		}
		dir := path.Dir(f.Name)
		depth := 0
		if dir != "." {
			depth = strings.Count(dir, "/") + 1
		}
		switch {
		case bestDepth < 0 || depth < bestDepth:
			best, bestDepth, count = dir, depth, 1
		case depth == bestDepth:
			count++
		}
	}
	switch {
	case bestDepth < 0:
		return "", formatErr("the Hugging Face zip has no config.json; zip the folder save_pretrained() wrote")
	case count > 1:
		return "", formatErr("the Hugging Face zip holds several models side by side; upload one")
	}
	return best, nil
}

// sanitizeHFConfig drops auto_map (a pointer to custom model code) so only
// transformers' built-in classes are ever used, and checks config.json names
// a model_type.
func sanitizeHFConfig(file string, isModelConfig bool) error {
	raw, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	var cfg map[string]any
	if err := json.Unmarshal(bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf")), &cfg); err != nil {
		return formatErr("%s is not valid JSON", filepath.Base(file))
	}
	if isModelConfig {
		if mt, _ := cfg["model_type"].(string); mt == "" {
			return formatErr("config.json has no model_type")
		}
	}
	if _, ok := cfg["auto_map"]; !ok {
		return nil
	}
	log.Printf("modelpkg: removed auto_map from %s; the built-in transformers class for the model type is used", filepath.Base(file))
	delete(cfg, "auto_map")
	out, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(file, out, 0o644)
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
