package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
)

const manifestSuffix = ".manifest.json"

// modalityByExt maps data-file extensions to the decoder infer.py uses.
var modalityByExt = map[string]string{
	".jpg": "image", ".jpeg": "image", ".png": "image",
	".dcm": "dicom",
}

// dataset is a decrypted dataset in the job's temp folder.
type dataset struct {
	Modality    string            // "image" or "dicom", from the data files' extensions
	GroundTruth string            // path to the decrypted ground_truth.csv
	Files       map[string]string // data file name -> path
}

// datasetObject is one encrypted object and its manifest.
type datasetObject struct {
	Name     string // file name the plaintext is written under
	Manifest string // GCS object names
	Cipher   string
}

// planDataset chooses what to decrypt from the folder listing alone, before
// any KMS call: data files by extension, and the ground truth
// (ground_truth.csv, else the only .csv). Everything else — for example a
// legacy evaluation_script.py — is skipped and never decrypted.
func planDataset(names []string) (data []datasetObject, gt datasetObject, modality string, err error) {
	present := make(map[string]bool, len(names))
	for _, n := range names {
		present[n] = true
	}
	seen := map[string]bool{}
	var csvs []datasetObject
	skipped := 0
	for _, n := range names {
		if !strings.HasSuffix(n, manifestSuffix) {
			continue
		}
		cipher := strings.TrimSuffix(n, manifestSuffix)
		if !present[cipher] {
			return nil, gt, "", fmt.Errorf("manifest %q has no matching encrypted object", n)
		}
		obj := datasetObject{Name: path.Base(cipher), Manifest: n, Cipher: cipher}
		if seen[obj.Name] {
			return nil, gt, "", fmt.Errorf("two dataset objects are named %q", obj.Name)
		}
		seen[obj.Name] = true
		ext := strings.ToLower(path.Ext(obj.Name))
		switch m, isData := modalityByExt[ext]; {
		case isData:
			if modality != "" && m != modality {
				return nil, gt, "", fmt.Errorf("dataset mixes %s and %s files", modality, m)
			}
			modality = m
			data = append(data, obj)
		case ext == ".csv":
			csvs = append(csvs, obj)
		default:
			skipped++
		}
	}
	if skipped > 0 {
		log.Printf("pipeline: skipping %d dataset object(s) that are neither data files nor ground truth", skipped)
	}
	if len(data) == 0 {
		return nil, gt, "", fmt.Errorf("no encrypted data files (.jpg/.jpeg/.png/.dcm with a manifest) in the dataset")
	}
	gt, err = pickGroundTruth(csvs)
	if err != nil {
		return nil, gt, "", err
	}
	sort.Slice(data, func(i, j int) bool { return data[i].Name < data[j].Name })
	return data, gt, modality, nil
}

func pickGroundTruth(csvs []datasetObject) (datasetObject, error) {
	for _, c := range csvs {
		if strings.EqualFold(c.Name, "ground_truth.csv") {
			return c, nil
		}
	}
	switch len(csvs) {
	case 0:
		return datasetObject{}, fmt.Errorf("the dataset has no ground_truth.csv")
	case 1:
		return csvs[0], nil
	}
	return datasetObject{}, fmt.Errorf("the dataset has %d .csv files and none is named ground_truth.csv", len(csvs))
}

// fetchAndDecryptDataset lists gs://<DatasetsBucket>/<uuid>/, decrypts the
// planned objects in parallel (each: KMS-unwrap its key, decrypt + verify
// tanuh-enc-dataset-v1) into destDir, and keeps each manifest next to its file.
// destDir is a per-job temp folder the caller deletes when the job completes.
func (m *Manager) fetchAndDecryptDataset(ctx context.Context, uuid, destDir string) (*dataset, error) {
	bucket := m.cfg.DatasetsBucket
	prefix := uuid + "/"
	names, err := gcp.ListObjects(ctx, bucket, prefix)
	if err != nil {
		return nil, fmt.Errorf("pipeline: list dataset folder: %w", err)
	}
	data, gt, modality, err := planDataset(names)
	if err != nil {
		return nil, fmt.Errorf("pipeline: dataset gs://%s/%s: %w", bucket, prefix, err)
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, err
	}

	started := time.Now()
	all := append(append([]datasetObject{}, data...), gt)
	err = runParallel(ctx, m.cfg.DecryptWorkers, all, func(ctx context.Context, obj datasetObject) error {
		if err := m.decryptDatasetObject(ctx, bucket, obj, destDir); err != nil {
			return fmt.Errorf("pipeline: decrypt %s: %w", obj.Cipher, err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	log.Printf("pipeline: decrypted %d %s files + ground truth in %s", len(data), modality,
		time.Since(started).Round(time.Millisecond))

	ds := &dataset{Modality: modality, GroundTruth: filepath.Join(destDir, gt.Name), Files: map[string]string{}}
	for _, d := range data {
		ds.Files[d.Name] = filepath.Join(destDir, d.Name)
	}
	return ds, nil
}

// decryptDatasetObject downloads a manifest + its ciphertext, keeps the
// manifest in destDir, unwraps the key via KMS, decrypts + verifies, and
// writes the plaintext under the object's file name.
func (m *Manager) decryptDatasetObject(ctx context.Context, bucket string, obj datasetObject, destDir string) error {
	mbytes, err := gcp.DownloadObjectBytes(ctx, bucket, obj.Manifest)
	if err != nil {
		return fmt.Errorf("download manifest: %w", err)
	}
	if err := os.WriteFile(filepath.Join(destDir, obj.Name+manifestSuffix), mbytes, 0o644); err != nil {
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
	ct, err := gcp.DownloadObjectBytes(ctx, bucket, obj.Cipher)
	if err != nil {
		return fmt.Errorf("download ciphertext: %w", err)
	}
	pt, err := crypto.DecryptDatasetV1(dek, man, ct)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(destDir, obj.Name), pt, 0o644)
}

// runParallel calls fn for every item on up to `workers` goroutines. The first
// error cancels the rest (items not yet started are skipped) and is returned.
func runParallel[T any](ctx context.Context, workers int, items []T, fn func(context.Context, T) error) error {
	if workers < 1 {
		workers = 1
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)
	queue := make(chan T)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for it := range queue {
				if ctx.Err() != nil {
					continue
				}
				if err := fn(ctx, it); err != nil {
					once.Do(func() { firstErr = err; cancel() })
				}
			}
		}()
	}
feed:
	for _, it := range items {
		select {
		case queue <- it:
		case <-ctx.Done():
			break feed
		}
	}
	close(queue)
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
}
