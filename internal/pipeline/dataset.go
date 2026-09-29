package pipeline

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
)

// fetchAndDecryptDataset lists gs://<DatasetsBucket>/<uuid>/ and, for every
// <object>.manifest.json it finds, keeps the manifest in destDir, unwraps the
// DEK via KMS asymmetricDecrypt, decrypts + authenticates the object
// (tanuh-enc-dataset-v1), verifies the plaintext SHA-256, and lays the result
// into destDir: a DATA zip is extracted in place (zip-slip safe); the
// ground-truth table is written as-is. destDir is a per-job temp folder the
// caller deletes when the job completes.
func (m *Manager) fetchAndDecryptDataset(ctx context.Context, uuid, destDir string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	bucket := m.cfg.DatasetsBucket
	prefix := uuid + "/"
	log.Printf("pipeline: listing dataset objects gs://%s/%s", bucket, prefix)
	names, err := gcp.ListObjects(ctx, bucket, prefix)
	if err != nil {
		return fmt.Errorf("pipeline: list dataset folder: %w", err)
	}
	present := make(map[string]bool, len(names))
	for _, n := range names {
		present[n] = true
	}
	var manifests []string
	for _, n := range names {
		if strings.HasSuffix(n, ".manifest.json") {
			manifests = append(manifests, n)
		}
	}
	if len(manifests) == 0 {
		return fmt.Errorf("pipeline: no *.manifest.json under gs://%s/%s", bucket, prefix)
	}
	for _, manifestObj := range manifests {
		cipherObj := strings.TrimSuffix(manifestObj, ".manifest.json")
		if !present[cipherObj] {
			return fmt.Errorf("pipeline: manifest %q has no ciphertext object %q", manifestObj, cipherObj)
		}
		if err := m.decryptDatasetObject(ctx, bucket, manifestObj, cipherObj, destDir); err != nil {
			return fmt.Errorf("pipeline: decrypt %s: %w", cipherObj, err)
		}
	}
	return nil
}

// decryptDatasetObject downloads a manifest + its ciphertext, keeps the manifest
// in destDir (retained for audit and future encrypted-manifest support), unwraps
// the DEK via KMS, decrypts + verifies, and lays the plaintext into destDir — a
// zip is extracted in place, anything else written under its base name.
func (m *Manager) decryptDatasetObject(ctx context.Context, bucket, manifestObj, cipherObj, destDir string) error {
	mbytes, err := gcp.DownloadObjectBytes(ctx, bucket, manifestObj)
	if err != nil {
		return fmt.Errorf("download manifest: %w", err)
	}
	if err := os.WriteFile(filepath.Join(destDir, path.Base(manifestObj)), mbytes, 0o644); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	var man crypto.ManifestV1
	if err := json.Unmarshal(mbytes, &man); err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}
	if man.Schema != "tanuh-enc-dataset-v1" {
		return fmt.Errorf("unsupported manifest schema %q", man.Schema)
	}
	dek, err := gcp.AsymmetricDecrypt(ctx, man.KMSKeyVersion, man.WrappedDEK)
	if err != nil {
		return fmt.Errorf("KMS unwrap: %w", err)
	}
	if len(dek) != 32 {
		return fmt.Errorf("unwrapped DEK is %d bytes, want 32", len(dek))
	}
	ct, err := gcp.DownloadObjectBytes(ctx, bucket, cipherObj)
	if err != nil {
		return fmt.Errorf("download ciphertext: %w", err)
	}
	pt, err := crypto.DecryptDatasetV1(dek, man, ct)
	if err != nil {
		return err
	}
	// The DATA object is a zip — extract it in Go (zip-slip safe) so the engine
	// gets a ready directory. The ground-truth table is written as-is.
	if strings.HasSuffix(strings.ToLower(cipherObj), ".zip") {
		if err := unzipBytes(pt, destDir); err != nil {
			return fmt.Errorf("extract data zip: %w", err)
		}
		log.Printf("pipeline: decrypted + extracted %s (%d bytes) into %s", cipherObj, len(pt), destDir)
		return nil
	}
	outPath := filepath.Join(destDir, path.Base(cipherObj))
	if err := os.WriteFile(outPath, pt, 0o644); err != nil {
		return fmt.Errorf("write plaintext: %w", err)
	}
	log.Printf("pipeline: decrypted %s (%d bytes) -> %s", cipherObj, len(pt), outPath)
	return nil
}

// unzip extracts an archive file with zip-slip protection.
func unzip(zipPath, destDir string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()
	return extractZip(&r.Reader, destDir)
}

// unzipBytes extracts an in-memory zip (the decrypted DATA object) with the same
// zip-slip protection as unzip.
func unzipBytes(data []byte, destDir string) error {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return err
	}
	return extractZip(zr, destDir)
}

func extractZip(r *zip.Reader, destDir string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}
	cleanDest := filepath.Clean(destDir)
	for _, f := range r.File {
		target := filepath.Join(cleanDest, f.Name)
		if !strings.HasPrefix(filepath.Clean(target), cleanDest+string(os.PathSeparator)) {
			return fmt.Errorf("zip entry escapes destination: %q", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		if err := extractFile(f, target); err != nil {
			return fmt.Errorf("extract %s: %w", f.Name, err)
		}
	}
	return nil
}

func extractFile(f *zip.File, target string) error {
	src, err := f.Open()
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(target)
	if err != nil {
		return err
	}
	defer dst.Close()
	_, err = io.Copy(dst, src)
	return err
}
