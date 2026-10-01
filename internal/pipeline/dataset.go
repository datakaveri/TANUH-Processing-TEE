package pipeline

import (
	"bytes"
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
	"github.com/datakaveri/tanuh-processing-tee/internal/groundtruth"
)

const manifestSuffix = ".manifest.json"

// groundTruthExts are the ground-truth formats groundtruth.Read understands.
var groundTruthExts = map[string]bool{".csv": true, ".tsv": true, ".json": true, ".jsonl": true, ".ndjson": true}

// groundTruthNames are file names (without extension, lower case, letters
// and digits only) that mark the ground truth when the root folder holds
// several tables.
var groundTruthNames = []string{"groundtruth", "gt", "labels", "label", "annotations", "annotation"}

// dataset is a decrypted dataset in the job's temp folder.
type dataset struct {
	Modality    string              // image, dicom, or mixed — from the files' content
	GroundTruth string              // the ground-truth object's name in the dataset
	Table       *groundtruth.Table  // the ground truth as read
	Match       *groundtruth.Result // ... and matched to the dataset's files
	Paths       map[string]string   // path in the dataset -> decrypted file
}

// datasetObject is one encrypted object and its manifest.
type datasetObject struct {
	Rel      string // path inside the dataset folder; the plaintext is written there
	Manifest string // GCS object names
	Cipher   string
}

// safeRel reports whether an object's path inside the dataset folder can be
// written under the job's temp folder: no empty, "." or ".." parts.
func safeRel(rel string) bool {
	if rel == "" || strings.ContainsRune(rel, 0) {
		return false
	}
	for _, part := range strings.Split(rel, "/") {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

// planDataset reads the folder listing gs://bucket/<prefix>... before any KMS
// call. Every encrypted object (one with a manifest) at any depth is a
// candidate data file, named by its path inside the folder; the ground truth
// is a table in the folder itself (not a subfolder). Objects without a
// manifest (e.g. an old-format dataset.json.enc) are skipped.
func planDataset(names []string, prefix string) (objects []datasetObject, gt datasetObject, err error) {
	present := make(map[string]bool, len(names))
	for _, n := range names {
		present[n] = true
	}
	var tables []datasetObject
	unsafe, orphans := 0, 0
	for _, n := range names {
		if !strings.HasSuffix(n, manifestSuffix) {
			continue
		}
		cipher := strings.TrimSuffix(n, manifestSuffix)
		rel := strings.TrimPrefix(cipher, prefix)
		switch {
		case !present[cipher]:
			orphans++
			continue
		case !safeRel(rel):
			unsafe++
			continue
		}
		obj := datasetObject{Rel: rel, Manifest: n, Cipher: cipher}
		if !strings.Contains(rel, "/") && groundTruthExts[strings.ToLower(path.Ext(rel))] {
			tables = append(tables, obj)
		}
		objects = append(objects, obj)
	}
	if orphans > 0 {
		log.Printf("pipeline: skipping %d manifest(s) whose encrypted object is missing", orphans)
	}
	if unsafe > 0 {
		log.Printf("pipeline: skipping %d object(s) with an unusable path (empty, . or .. parts)", unsafe)
	}
	gt, err = pickGroundTruth(tables)
	if err != nil {
		return nil, gt, err
	}
	data := objects[:0]
	for _, o := range objects {
		if o.Rel != gt.Rel {
			data = append(data, o)
		}
	}
	if len(data) == 0 {
		return nil, gt, fmt.Errorf("the dataset has no encrypted files besides the ground truth %s", gt.Rel)
	}
	sort.Slice(data, func(i, j int) bool { return data[i].Rel < data[j].Rel })
	return data, gt, nil
}

// pickGroundTruth chooses among the tables in the dataset's root folder: the
// one named like a ground truth (ground_truth.csv, labels.json, gt.tsv, ...),
// else the only table.
func pickGroundTruth(tables []datasetObject) (datasetObject, error) {
	var named []datasetObject
	for _, want := range groundTruthNames {
		for _, t := range tables {
			if normBase(t.Rel) == want {
				named = append(named, t)
			}
		}
		if len(named) > 0 {
			break
		}
	}
	switch {
	case len(named) == 1:
		return named[0], nil
	case len(named) > 1:
		return datasetObject{}, fmt.Errorf("the dataset has several ground-truth files (%s); keep one", joinRels(named))
	case len(tables) == 1:
		return tables[0], nil
	case len(tables) == 0:
		return datasetObject{}, fmt.Errorf("the dataset folder has no ground truth (.csv, .tsv, .json or .jsonl next to the data)")
	}
	return datasetObject{}, fmt.Errorf("the dataset folder has %d tables (%s) and none is named ground_truth", len(tables), joinRels(tables))
}

func normBase(rel string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(strings.TrimSuffix(rel, path.Ext(rel))) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func joinRels(objs []datasetObject) string {
	rels := make([]string, len(objs))
	for i, o := range objs {
		rels[i] = o.Rel
	}
	return strings.Join(rels, ", ")
}

// fileKind tells DICOM from images by content, falling back to the extension
// for DICOM files without the "DICM" preamble.
func fileKind(head []byte, rel string) string {
	switch {
	case len(head) >= 132 && string(head[128:132]) == "DICM":
		return "dicom"
	case bytes.HasPrefix(head, []byte{0xFF, 0xD8, 0xFF}), // JPEG
		bytes.HasPrefix(head, []byte("\x89PNG")),
		bytes.HasPrefix(head, []byte("II*\x00")), bytes.HasPrefix(head, []byte("MM\x00*")), // TIFF
		bytes.HasPrefix(head, []byte("BM")),
		bytes.HasPrefix(head, []byte("GIF8")),
		len(head) >= 12 && string(head[:4]) == "RIFF" && string(head[8:12]) == "WEBP":
		return "image"
	}
	switch strings.ToLower(path.Ext(rel)) {
	case ".dcm", ".dicom", ".dic":
		return "dicom"
	}
	return "unknown"
}

// fetchAndDecryptDataset lists gs://<DatasetsBucket>/<uuid>/, decrypts the
// ground truth, matches it to the listed files, then decrypts just the
// matched files in parallel (each: KMS-unwrap its key, decrypt + verify
// tanuh-enc-dataset-v1) into destDir at their paths in the dataset, keeping
// each manifest next to its file. destDir is a per-job temp folder the caller
// deletes when the job completes.
func (m *Manager) fetchAndDecryptDataset(ctx context.Context, uuid, destDir string, classNames []string) (*dataset, error) {
	bucket := m.cfg.DatasetsBucket
	prefix := uuid + "/"
	names, err := gcp.ListObjects(ctx, bucket, prefix)
	if err != nil {
		return nil, fmt.Errorf("pipeline: list dataset folder: %w", err)
	}
	objects, gtObj, err := planDataset(names, prefix)
	if err != nil {
		return nil, fmt.Errorf("pipeline: dataset gs://%s/%s: %w", bucket, prefix, err)
	}
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, err
	}

	started := time.Now()
	gtBytes, err := m.decryptDatasetObject(ctx, bucket, gtObj, destDir)
	if err != nil {
		return nil, fmt.Errorf("pipeline: decrypt ground truth %s: %w", gtObj.Rel, err)
	}
	table, err := groundtruth.Read(gtObj.Rel, gtBytes)
	if err != nil {
		return nil, fmt.Errorf("pipeline: %w", err)
	}
	rels := make([]string, len(objects))
	byRel := make(map[string]datasetObject, len(objects))
	for i, o := range objects {
		rels[i] = o.Rel
		byRel[o.Rel] = o
	}
	match, err := groundtruth.Match(table, classNames, rels)
	if err != nil {
		return nil, fmt.Errorf("pipeline: %w", err)
	}
	if len(match.Unmatched) > 0 {
		shown := match.Unmatched
		if len(shown) > 3 {
			shown = shown[:3]
		}
		log.Printf("pipeline: %d ground-truth row(s) have no file in the dataset and are skipped, e.g. %q",
			len(match.Unmatched), shown)
	}

	// Only files the ground truth refers to are decrypted.
	needed := make([]datasetObject, len(match.Rows))
	for i, r := range match.Rows {
		needed[i] = byRel[r.File]
	}
	var mu sync.Mutex
	kinds := map[string]int{}
	err = runParallel(ctx, m.cfg.DecryptWorkers, needed, func(ctx context.Context, obj datasetObject) error {
		pt, err := m.decryptDatasetObject(ctx, bucket, obj, destDir)
		if err != nil {
			return fmt.Errorf("pipeline: decrypt %s: %w", obj.Cipher, err)
		}
		kind := fileKind(pt[:min(len(pt), 132)], obj.Rel)
		mu.Lock()
		kinds[kind]++
		mu.Unlock()
		return nil
	})
	if err != nil {
		return nil, err
	}

	ds := &dataset{GroundTruth: gtObj.Rel, Table: table, Match: match, Paths: make(map[string]string, len(needed))}
	for _, o := range needed {
		ds.Paths[o.Rel] = filepath.Join(destDir, filepath.FromSlash(o.Rel))
	}
	ds.Modality = modalityOf(kinds)
	log.Printf("pipeline: decrypted ground truth %s + %d of %d dataset files (%v) in %s", gtObj.Rel, len(needed),
		len(objects), kinds, time.Since(started).Round(time.Millisecond))
	return ds, nil
}

// modalityOf summarises per-file kinds: one kind, or "mixed". Files of
// unknown kind are left for infer.py to try decoding.
func modalityOf(kinds map[string]int) string {
	var known []string
	for k := range kinds {
		if k != "unknown" {
			known = append(known, k)
		}
	}
	switch len(known) {
	case 0:
		return "unknown"
	case 1:
		return known[0]
	}
	return "mixed"
}

// decryptDatasetObject downloads a manifest + its ciphertext, unwraps the key
// via KMS, decrypts + verifies, and writes the plaintext and its manifest
// under the object's path in destDir. It returns the plaintext.
func (m *Manager) decryptDatasetObject(ctx context.Context, bucket string, obj datasetObject, destDir string) ([]byte, error) {
	target := filepath.Join(destDir, filepath.FromSlash(obj.Rel))
	if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
		return nil, fmt.Errorf("object path %q leaves the dataset folder", obj.Rel)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return nil, err
	}
	mbytes, err := gcp.DownloadObjectBytes(ctx, bucket, obj.Manifest)
	if err != nil {
		return nil, fmt.Errorf("download manifest: %w", err)
	}
	if err := os.WriteFile(target+manifestSuffix, mbytes, 0o644); err != nil {
		return nil, fmt.Errorf("write manifest: %w", err)
	}
	var man crypto.ManifestV1
	if err := json.Unmarshal(mbytes, &man); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}
	if man.Schema != "tanuh-enc-dataset-v1" {
		return nil, fmt.Errorf("unsupported manifest schema %q", man.Schema)
	}
	dek, err := gcp.AsymmetricDecrypt(ctx, man.KMSKeyVersion, man.WrappedDEK)
	if err != nil {
		return nil, fmt.Errorf("KMS unwrap: %w", err)
	}
	if len(dek) != 32 {
		return nil, fmt.Errorf("unwrapped DEK is %d bytes, want 32", len(dek))
	}
	ct, err := gcp.DownloadObjectBytes(ctx, bucket, obj.Cipher)
	if err != nil {
		return nil, fmt.Errorf("download ciphertext: %w", err)
	}
	pt, err := crypto.DecryptDatasetV1(dek, man, ct)
	if err != nil {
		return nil, err
	}
	return pt, os.WriteFile(target, pt, 0o644)
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
