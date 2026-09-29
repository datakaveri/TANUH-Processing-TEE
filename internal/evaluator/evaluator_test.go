package evaluator

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/datakaveri/tanuh-processing-tee/internal/gcp"
)

func TestValidateSlug(t *testing.T) {
	for _, ok := range []string{"binary_classification", "multiclass_classification", "seg-v2", "a"} {
		if err := ValidateSlug(ok); err != nil {
			t.Errorf("%q rejected: %v", ok, err)
		}
	}
	for _, bad := range []string{"", "../x", "a/b", "Binary", "1abc", "a%2Fb", "a.b", strings.Repeat("a", 64)} {
		if err := ValidateSlug(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestGCSSourceRecordsHashAndGeneration(t *testing.T) {
	var gotBucket, gotObject string
	src := &GCSSource{Bucket: "tanuh-evaluators", Download: func(_ context.Context, b, o string, max int64) ([]byte, string, error) {
		gotBucket, gotObject = b, o
		if max != MaxScriptBytes {
			t.Fatalf("max = %d", max)
		}
		return []byte("print('score')"), "1717171717", nil
	}}
	ev, err := src.Fetch(context.Background(), "binary_classification", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if gotBucket != "tanuh-evaluators" || gotObject != "binary_classification/evaluate.py" {
		t.Fatalf("fetched gs://%s/%s", gotBucket, gotObject)
	}
	if ev.Origin != "gs://tanuh-evaluators/binary_classification/evaluate.py#1717171717" || len(ev.SHA256) != 64 {
		t.Fatalf("evaluator = %+v", ev)
	}
	if b, _ := os.ReadFile(ev.Path); string(b) != "print('score')" {
		t.Fatalf("script content %q", b)
	}
}

func TestGCSSourceMissingEvaluatorFailsClosed(t *testing.T) {
	src := &GCSSource{Bucket: "b", Download: func(context.Context, string, string, int64) ([]byte, string, error) {
		return nil, "", gcp.ErrObjectNotFound
	}}
	if _, err := src.Fetch(context.Background(), "regression", t.TempDir()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestGCSSourceRejectsBadSlugBeforeFetching(t *testing.T) {
	src := &GCSSource{Bucket: "b", Download: func(context.Context, string, string, int64) ([]byte, string, error) {
		t.Fatal("download called for an invalid slug")
		return nil, "", nil
	}}
	if _, err := src.Fetch(context.Background(), "../secrets", t.TempDir()); err == nil {
		t.Fatal("invalid slug accepted")
	}
}

func TestImageSource(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "binary_classification"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "binary_classification", "evaluate.py"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	ev, err := ImageSource{Root: root}.Fetch(context.Background(), "binary_classification", t.TempDir())
	if err != nil || !strings.HasPrefix(ev.Origin, "image:") {
		t.Fatalf("ev=%+v err=%v", ev, err)
	}
	if _, err := (ImageSource{Root: root}).Fetch(context.Background(), "regression", t.TempDir()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing evaluator err = %v", err)
	}
}
