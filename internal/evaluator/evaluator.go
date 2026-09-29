// Package evaluator locates the evaluation script for a problem bucket.
//
// Each bucket (binary_classification, multiclass_classification, ...) has one
// platform-owned evaluate.py. For now it is fetched per job from a GCS bucket
// baked into the image (GCSSource); later it will ship inside the image
// (ImageSource). Either way the job records the script's SHA-256 and origin, so
// every leaderboard entry can be traced to the exact scorer that produced it.
package evaluator

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
)

// MaxScriptBytes caps an evaluator script's size.
const MaxScriptBytes = 1 << 20

var slugRe = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,62}$`)

// ErrNotFound means no evaluator exists for the bucket.
var ErrNotFound = errors.New("no evaluator for this bucket")

// ValidateSlug checks a bucket name before it becomes part of a storage path.
func ValidateSlug(slug string) error {
	if !slugRe.MatchString(slug) {
		return fmt.Errorf("evaluator: invalid bucket name %q", slug)
	}
	return nil
}

// Evaluator is a fetched evaluation script ready to run.
type Evaluator struct {
	Bucket string // problem bucket slug
	Path   string // local script path
	SHA256 string
	Origin string // where it came from, e.g. gs://bucket/slug/evaluate.py#<generation>
}

// Source fetches the evaluator for a bucket into destDir.
type Source interface {
	Fetch(ctx context.Context, slug, destDir string) (Evaluator, error)
}

// GCSSource reads gs://<Bucket>/<slug>/evaluate.py.
type GCSSource struct {
	Bucket string
	// Download is gcp.DownloadObjectWithGeneration; replaceable in tests.
	Download func(ctx context.Context, bucket, object string, maxBytes int64) ([]byte, string, error)
}

// NewGCSSource returns a GCSSource for bucket.
func NewGCSSource(bucket string) *GCSSource {
	return &GCSSource{Bucket: bucket, Download: gcp.DownloadObjectWithGeneration}
}

// Fetch implements Source.
func (s *GCSSource) Fetch(ctx context.Context, slug, destDir string) (Evaluator, error) {
	if err := ValidateSlug(slug); err != nil {
		return Evaluator{}, err
	}
	object := slug + "/evaluate.py"
	data, generation, err := s.Download(ctx, s.Bucket, object, MaxScriptBytes)
	if errors.Is(err, gcp.ErrObjectNotFound) {
		return Evaluator{}, fmt.Errorf("evaluator: %s (gs://%s/%s): %w", slug, s.Bucket, object, ErrNotFound)
	}
	if err != nil {
		return Evaluator{}, fmt.Errorf("evaluator: fetch %s: %w", slug, err)
	}
	origin := fmt.Sprintf("gs://%s/%s", s.Bucket, object)
	if generation != "" {
		origin += "#" + generation
	}
	return write(slug, destDir, data, origin)
}

// ImageSource reads <Root>/<slug>/evaluate.py from inside the image.
type ImageSource struct {
	Root string
}

// Fetch implements Source.
func (s ImageSource) Fetch(_ context.Context, slug, destDir string) (Evaluator, error) {
	if err := ValidateSlug(slug); err != nil {
		return Evaluator{}, err
	}
	path := filepath.Join(s.Root, slug, "evaluate.py")
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Evaluator{}, fmt.Errorf("evaluator: %s (%s): %w", slug, path, ErrNotFound)
	}
	if err != nil {
		return Evaluator{}, err
	}
	if len(data) > MaxScriptBytes {
		return Evaluator{}, fmt.Errorf("evaluator: %s is larger than %d bytes", path, MaxScriptBytes)
	}
	return write(slug, destDir, data, "image:"+path)
}

func write(slug, destDir string, data []byte, origin string) (Evaluator, error) {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return Evaluator{}, err
	}
	path := filepath.Join(destDir, "evaluate.py")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return Evaluator{}, err
	}
	sum := sha256.Sum256(data)
	return Evaluator{Bucket: slug, Path: path, SHA256: hex.EncodeToString(sum[:]), Origin: origin}, nil
}
