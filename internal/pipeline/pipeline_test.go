package pipeline

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestDecodeAndVerify(t *testing.T) {
	data := []byte("model-bytes")
	sum := sha256.Sum256(data)
	payload := map[string]any{
		"model_onnx_base64": base64.StdEncoding.EncodeToString(data),
		"model_sha256":      hex.EncodeToString(sum[:]),
	}

	got, err := decodeAndVerify(payload, "model_onnx_base64", "model_sha256")
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Fatal("decoded bytes mismatch")
	}

	payload["model_sha256"] = "deadbeef"
	if _, err := decodeAndVerify(payload, "model_onnx_base64", "model_sha256"); err == nil {
		t.Fatal("sha mismatch not rejected — must fail closed")
	}
}

func TestFileNameFieldRejectsTraversal(t *testing.T) {
	payload := map[string]any{"model_file": "../../etc/passwd"}
	if got := fileNameField(payload, "model_file", "model.onnx"); got != "model.onnx" {
		t.Fatalf("traversal name accepted: %q", got)
	}
	payload["model_file"] = "custom.onnx"
	if got := fileNameField(payload, "model_file", "model.onnx"); got != "custom.onnx" {
		t.Fatalf("plain name rejected: %q", got)
	}
}

func TestUnzipRejectsZipSlip(t *testing.T) {
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "evil.zip")

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("../escape.txt")
	w.Write([]byte("pwned")) //nolint:errcheck
	zw.Close()
	if err := os.WriteFile(zipPath, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := unzip(zipPath, filepath.Join(dir, "out")); err == nil {
		t.Fatal("zip-slip entry extracted without error")
	}
}

func TestUnzipExtracts(t *testing.T) {
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "ok.zip")

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("sub/file.txt")
	w.Write([]byte("hello")) //nolint:errcheck
	zw.Close()
	if err := os.WriteFile(zipPath, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	out := filepath.Join(dir, "out")
	if err := unzip(zipPath, out); err != nil {
		t.Fatalf("unzip: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(out, "sub", "file.txt"))
	if err != nil || string(got) != "hello" {
		t.Fatalf("extracted content wrong: %q err=%v", got, err)
	}
}
