// Package groundtruth reads a dataset's canonical ground_truth.csv and
// matches it to the decrypted data files.
//
// Format (one header row, then one row per sample):
//
//	file,label
//	02da7fc6-fb22-42d5-95d2-26034ef84006.jpg,0
//
// `file` is the data object's file name; `label` is the 0-based class index,
// in class_names order. Every row must name a file present in the dataset, so
// every model is scored on exactly the same samples.
package groundtruth

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Row is one ground-truth sample.
type Row struct {
	File  string
	Label int
}

// Parse reads the CSV, requiring `file` and `label` columns, unique files and
// labels in [0, numClasses).
func Parse(r io.Reader, numClasses int) ([]Row, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	raw = bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf")) // UTF-8 BOM written by spreadsheet tools
	cr := csv.NewReader(bytes.NewReader(raw))
	cr.TrimLeadingSpace = true
	header, err := cr.Read()
	if err != nil {
		return nil, fmt.Errorf("ground truth: read header: %w", err)
	}
	fileCol, labelCol := -1, -1
	for i, h := range header {
		switch strings.ToLower(strings.TrimSpace(h)) {
		case "file":
			fileCol = i
		case "label":
			labelCol = i
		}
	}
	if fileCol < 0 || labelCol < 0 {
		return nil, fmt.Errorf("ground truth: header must contain file and label columns, got %v", header)
	}

	var rows []Row
	seen := map[string]int{}
	for line := 2; ; line++ {
		rec, err := cr.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("ground truth line %d: %w", line, err)
		}
		if len(rec) <= fileCol || len(rec) <= labelCol {
			return nil, fmt.Errorf("ground truth line %d: missing columns", line)
		}
		file := strings.TrimSpace(rec[fileCol])
		if file == "" {
			return nil, fmt.Errorf("ground truth line %d: empty file", line)
		}
		if prev, dup := seen[file]; dup {
			return nil, fmt.Errorf("ground truth line %d: file %q already listed on line %d", line, file, prev)
		}
		seen[file] = line
		label, err := strconv.Atoi(strings.TrimSpace(rec[labelCol]))
		if err != nil || label < 0 || label >= numClasses {
			return nil, fmt.Errorf("ground truth line %d: label must be 0..%d, got %q", line, numClasses-1, rec[labelCol])
		}
		rows = append(rows, Row{File: file, Label: label})
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("ground truth has no rows")
	}
	return rows, nil
}

// ParseFile is Parse for a file on disk.
func ParseFile(path string, numClasses int) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Parse(f, numClasses)
}

// Resolve maps every row to its data file path (files is name -> path) and
// returns the paths in ground-truth order. Any row without a file is an error.
func Resolve(rows []Row, files map[string]string) ([]string, error) {
	paths := make([]string, 0, len(rows))
	var missing []string
	for _, r := range rows {
		p, ok := files[r.File]
		if !ok {
			missing = append(missing, r.File)
			continue
		}
		paths = append(paths, p)
	}
	if len(missing) > 0 {
		shown := missing
		if len(shown) > 3 {
			shown = shown[:3]
		}
		return nil, fmt.Errorf("ground truth lists %d file(s) not in the dataset, e.g. %v", len(missing), shown)
	}
	return paths, nil
}

// WriteInputs writes the resolved paths, one per line, for the inference step.
func WriteInputs(path string, paths []string) error {
	return os.WriteFile(path, []byte(strings.Join(paths, "\n")+"\n"), 0o644)
}
