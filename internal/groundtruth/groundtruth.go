// Package groundtruth reads a dataset's ground truth, in whatever reasonable
// shape the data provider uploaded it, and matches it to the dataset's files.
//
// Accepted files (in the dataset's root folder):
//
//	CSV / TSV   any delimiter (, ; tab |); header names are flexible:
//	              Unique_case_ID,Label     file_name;class     image,target ...
//	            a file with no header and two columns is read as id,label
//	JSON        [{"file": "a.jpg", "label": 1}, ...]      (any key names, as above)
//	            {"annotations": [ ...records... ]}         (records under one key)
//	            {"a.jpg": 1, "b.jpg": "Suspicious"}        (id -> label map)
//	            [["a.jpg", 1], ...]                        (pairs)
//	JSON Lines  one record per line
//
// Labels are either the 0-based class index or the class name (case and
// punctuation ignored), decided once for the whole file.
//
// An id is matched to a dataset file by, in order: its path in the dataset,
// a path suffix, its file name, its file name without extension, or the name
// of a folder that holds exactly one data file. Rows whose file is not in the
// dataset are skipped and counted; an id that fits several files is an error,
// because a guess could score the wrong image.
//
// The pipeline then writes the canonical form the evaluators read:
//
//	file,label
//	Suspicious/02da7fc6.jpg,1        (file = path in the dataset folder)
//
// Segmentation (MatchMasks): the label column names each sample's mask file
// in the dataset instead of a class, matched by the same rules. Masks are
// resolved first and the sample ids only against the remaining files, so
// images/a.jpg and masks/a.png never collide. The canonical form is then
//
//	file,mask
//	images/a.jpg,/job/dataset/masks/a.png    (mask = the decrypted mask on disk)
package groundtruth

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// MinMatchedFraction is the share of labelled ground-truth rows that must
// match a dataset file. Below it the ids almost certainly name files some
// other way than the dataset does, and scoring the few that match would
// quietly benchmark on the wrong subset.
const MinMatchedFraction = 0.5

// Entry is one ground-truth record exactly as the data provider wrote it.
type Entry struct {
	ID    string
	Label string
	Line  int // CSV line or JSON item number, for error messages
}

// Table is a parsed ground-truth file.
type Table struct {
	Format      string // csv, tsv, json or jsonl
	IDColumn    string // the column / key used as the id ("" = positional)
	LabelColumn string
	Entries     []Entry
}

// Row is one labelled sample matched to a dataset file.
type Row struct {
	File  string // the file's path inside the dataset folder
	Label int
}

// Result is the ground truth matched against a dataset.
type Result struct {
	Rows      []Row             // matched samples, sorted by File
	Total     int               // entries in the file
	Unlabeled int               // entries with an empty label, skipped
	Unmatched []string          // ids with no file in the dataset, skipped
	LabelMode string            // "index", "name", "mask" (segmentation), "single" (one-class boxes)
	MatchedBy map[string]int    // matching rule -> count
	Masks     map[string]string // segmentation: row File -> its mask's path in the dataset
	Boxes     map[string][]Box  // object detection: row File -> its boxes (none: no objects)
}

// ── reading ─────────────────────────────────────────────────────────────────

// Read parses a ground-truth file. name picks the parser by extension
// (.json, .jsonl/.ndjson, anything else delimited text); data is the
// decrypted content.
func Read(name string, data []byte) (*Table, error) {
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf")) // BOM written by spreadsheet tools
	var (
		t   *Table
		err error
	)
	switch strings.ToLower(path.Ext(name)) {
	case ".json":
		t, err = readJSON(data)
	case ".jsonl", ".ndjson":
		t, err = readJSONLines(data)
	default:
		t, err = readDelimited(name, data)
	}
	if err != nil {
		return nil, fmt.Errorf("ground truth %s: %w", name, err)
	}
	if len(t.Entries) == 0 {
		return nil, fmt.Errorf("ground truth %s has no rows", name)
	}
	return t, nil
}

// Column names recognised as the sample id and the label, after normalising
// (lower case, runs of other characters -> "_"). Exact names win over hints;
// hints are substrings.
var (
	idNames = []string{"file", "filename", "file_name", "filepath", "file_path", "path", "relative_path",
		"image", "image_id", "image_name", "image_path", "image_file", "img", "img_id", "img_name",
		"dicom", "dicom_file", "sample", "sample_id", "case_id", "unique_case_id", "case", "subject",
		"subject_id", "folder", "folder_id", "id", "name", "uid"}
	idHints    = []string{"file", "path", "image", "img", "dicom", "case", "subject", "folder", "sample", "id", "name"}
	labelNames = []string{"label", "labels", "class", "class_id", "class_label", "class_name", "category",
		"category_id", "target", "y", "gt", "ground_truth", "groundtruth", "truth", "diagnosis",
		"annotation", "grade", "outcome",
		"mask", "mask_path", "mask_file", "mask_name", "segmentation", "seg", "seg_mask"} // segmentation
	labelHints = []string{"label", "class", "target", "category", "diagnos", "truth", "grade", "mask"}
)

func normName(s string) string {
	var b strings.Builder
	under := false
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			under = false
		} else if !under && b.Len() > 0 {
			b.WriteByte('_')
			under = true
		}
	}
	return strings.TrimSuffix(b.String(), "_")
}

// pickColumns chooses the id and label columns from a header, -1 for one it
// cannot identify. If only one is recognised and exactly one other named
// column exists, that column is the other.
func pickColumns(header []string) (idCol, labelCol int) {
	norm := make([]string, len(header))
	for i, h := range header {
		norm[i] = normName(h)
	}
	find := func(names, hints []string, skip int) int {
		for _, n := range names {
			for i, h := range norm {
				if i != skip && h == n {
					return i
				}
			}
		}
		for _, n := range hints {
			for i, h := range norm {
				if i != skip && h != "" && strings.Contains(h, n) {
					return i
				}
			}
		}
		return -1
	}
	only := func(skip int) int {
		found := -1
		for i, h := range norm {
			if i == skip || h == "" {
				continue
			}
			if found >= 0 {
				return -1
			}
			found = i
		}
		return found
	}
	labelCol = find(labelNames, labelHints, -1)
	idCol = find(idNames, idHints, labelCol)
	if idCol < 0 && labelCol >= 0 {
		idCol = only(labelCol)
	}
	if labelCol < 0 && idCol >= 0 {
		labelCol = only(idCol)
	}
	return idCol, labelCol
}

// looksLikeFile reports whether a cell reads as a file reference rather than
// a column name, to spot a CSV without a header row ("image_001.jpg,1").
func looksLikeFile(s string) bool {
	s = strings.TrimSpace(s)
	for _, n := range idNames {
		if normName(s) == n {
			return false
		}
	}
	ext := path.Ext(s)
	return strings.ContainsAny(s, "/\\") || (ext != "" && len(ext) <= 6 && ext != s)
}

func readDelimited(name string, data []byte) (*Table, error) {
	delim := sniffDelimiter(data)
	if strings.EqualFold(path.Ext(name), ".tsv") {
		delim = '\t'
	}
	t := &Table{Format: "csv"}
	if delim == '\t' {
		t.Format = "tsv"
	}
	cr := csv.NewReader(bytes.NewReader(data))
	cr.Comma = delim
	cr.FieldsPerRecord = -1
	cr.LazyQuotes = true
	cr.TrimLeadingSpace = true
	records, err := cr.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, errors.New("file is empty")
	}
	header := records[0]
	idCol, labelCol := pickColumns(header)
	start := 1
	switch {
	case len(header) == 2 && looksLikeFile(header[0]):
		idCol, labelCol, start = 0, 1, 0 // no header row: id,label
	case idCol >= 0 && labelCol >= 0:
		t.IDColumn, t.LabelColumn = strings.TrimSpace(header[idCol]), strings.TrimSpace(header[labelCol])
	case len(header) == 2:
		idCol, labelCol = 0, 1
		t.IDColumn, t.LabelColumn = strings.TrimSpace(header[0]), strings.TrimSpace(header[1])
	default:
		return nil, fmt.Errorf("cannot tell which columns hold the file and the label in header %q "+
			"(name them e.g. file and label)", header)
	}
	for i := start; i < len(records); i++ {
		rec := records[i]
		e := Entry{Line: i + 1}
		if idCol < len(rec) {
			e.ID = strings.TrimSpace(rec[idCol])
		}
		if labelCol < len(rec) {
			e.Label = strings.TrimSpace(rec[labelCol])
		}
		if e.ID == "" && e.Label == "" {
			continue // blank line
		}
		t.Entries = append(t.Entries, e)
	}
	return t, nil
}

// sniffDelimiter picks the most frequent of , tab ; | on the first line,
// outside quotes.
func sniffDelimiter(data []byte) rune {
	line := data
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		line = data[:i]
	}
	counts := map[rune]int{}
	inQuote := false
	for _, r := range string(line) {
		switch {
		case r == '"':
			inQuote = !inQuote
		case !inQuote && (r == ',' || r == ';' || r == '\t' || r == '|'):
			counts[r]++
		}
	}
	best, bestN := ',', 0
	for _, r := range []rune{',', '\t', ';', '|'} {
		if counts[r] > bestN {
			best, bestN = r, counts[r]
		}
	}
	return best
}

// Keys under which a JSON object commonly nests its list of records.
var recordListKeys = []string{"annotations", "data", "items", "samples", "images", "records", "rows",
	"entries", "ground_truth", "groundtruth", "labels", "files"}

func readJSON(data []byte) (*Table, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("not valid JSON: %w", err)
	}
	if dec.More() { // several top-level values: JSON Lines saved as .json
		return readJSONLines(data)
	}
	t := &Table{Format: "json"}
	return t, t.fromJSON(v)
}

func readJSONLines(data []byte) (*Table, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var items []any
	for {
		var v any
		err := dec.Decode(&v)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("not valid JSON Lines (item %d): %w", len(items)+1, err)
		}
		items = append(items, v)
	}
	t := &Table{Format: "jsonl"}
	return t, t.fromRecords(items)
}

func (t *Table) fromJSON(v any) error {
	switch x := v.(type) {
	case []any:
		return t.fromRecords(x)
	case map[string]any:
		keys := sortedKeys(x)
		for _, want := range recordListKeys {
			for _, k := range keys {
				if list, ok := x[k].([]any); ok && normName(k) == want {
					return t.fromRecords(list)
				}
			}
		}
		var lists []string
		for _, k := range keys {
			if list, ok := x[k].([]any); ok && len(list) > 0 {
				if _, isObj := list[0].(map[string]any); isObj {
					lists = append(lists, k)
				}
			}
		}
		if len(lists) == 1 {
			return t.fromRecords(x[lists[0]].([]any))
		}
		return t.fromMap(x)
	}
	return errors.New("expected a list of records or an object mapping file -> label")
}

// fromMap reads {"a.jpg": 1, ...} or {"a.jpg": {"label": 1, ...}, ...}.
func (t *Table) fromMap(m map[string]any) error {
	for i, k := range sortedKeys(m) {
		label, ok := scalarString(m[k])
		if !ok {
			obj, isObj := m[k].(map[string]any)
			if !isObj {
				return fmt.Errorf("value for %q is neither a label nor a record", k)
			}
			keys := sortedKeys(obj)
			_, labelKey := pickColumns(keys)
			if labelKey < 0 {
				return fmt.Errorf("record for %q has no label key", k)
			}
			label, _ = scalarString(obj[keys[labelKey]])
		}
		t.Entries = append(t.Entries, Entry{ID: strings.TrimSpace(k), Label: strings.TrimSpace(label), Line: i + 1})
	}
	return nil
}

func (t *Table) fromRecords(items []any) error {
	if len(items) == 0 {
		return nil
	}
	if _, isPair := items[0].([]any); isPair {
		for i, it := range items {
			pair, ok := it.([]any)
			if !ok || len(pair) < 2 {
				return fmt.Errorf("item %d is not an [id, label] pair", i+1)
			}
			id, _ := scalarString(pair[0])
			label, _ := scalarString(pair[1])
			t.Entries = append(t.Entries, Entry{ID: strings.TrimSpace(id), Label: strings.TrimSpace(label), Line: i + 1})
		}
		return nil
	}
	// Records: pick the id and label keys from the union of all keys.
	keySet := map[string]any{}
	for i, it := range items {
		obj, ok := it.(map[string]any)
		if !ok {
			return fmt.Errorf("item %d is not a record", i+1)
		}
		for k := range obj {
			keySet[k] = nil
		}
	}
	keys := sortedKeys(keySet)
	idCol, labelCol := pickColumns(keys)
	if idCol < 0 || labelCol < 0 {
		return fmt.Errorf("cannot tell which keys hold the file and the label in %q (name them e.g. file and label)", keys)
	}
	t.IDColumn, t.LabelColumn = keys[idCol], keys[labelCol]
	for i, it := range items {
		obj := it.(map[string]any)
		id, _ := scalarString(obj[t.IDColumn])
		label, _ := scalarString(obj[t.LabelColumn])
		if id == "" && label == "" {
			continue
		}
		t.Entries = append(t.Entries, Entry{ID: strings.TrimSpace(id), Label: strings.TrimSpace(label), Line: i + 1})
	}
	return nil
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// scalarString renders a JSON scalar as text; ok is false for arrays/objects.
func scalarString(v any) (string, bool) {
	switch x := v.(type) {
	case nil:
		return "", true
	case string:
		return x, true
	case json.Number:
		return x.String(), true
	case bool:
		return strconv.FormatBool(x), true
	}
	return "", false
}

// ── labels ──────────────────────────────────────────────────────────────────

func normLabel(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// labelParser decides, once for the whole file, whether labels are class
// names or 0-based class indexes, so one file never mixes the two readings.
// Names are tried first: with class names like "1","2","3" a file of 1/2/3
// means the names, not indexes.
func labelParser(labels, classNames []string) (mode string, parse func(string) int, err error) {
	byName := map[string]int{}
	for i, n := range classNames {
		k := normLabel(n)
		if _, dup := byName[k]; dup || k == "" {
			byName = nil // names collide once normalised: indexes only
			break
		}
		byName[k] = i
	}
	nameOf := func(s string) (int, bool) {
		i, ok := byName[normLabel(s)]
		return i, ok && byName != nil
	}
	indexOf := func(s string) (int, bool) {
		if n, err := strconv.Atoi(s); err == nil {
			return n, n >= 0 && n < len(classNames)
		}
		if f, err := strconv.ParseFloat(s, 64); err == nil && f == math.Trunc(f) { // 1.0 from spreadsheets
			return int(f), f >= 0 && int(f) < len(classNames)
		}
		if len(classNames) == 2 {
			switch strings.ToLower(s) {
			case "true", "yes", "positive":
				return 1, true
			case "false", "no", "negative":
				return 0, true
			}
		}
		return 0, false
	}
	failing := func(f func(string) (int, bool)) []string {
		var bad []string
		for _, l := range labels {
			if _, ok := f(l); !ok {
				bad = append(bad, l)
			}
		}
		return bad
	}
	if len(failing(nameOf)) == 0 {
		return "name", func(s string) int { i, _ := nameOf(s); return i }, nil
	}
	bad := failing(indexOf)
	if len(bad) == 0 {
		return "index", func(s string) int { i, _ := indexOf(s); return i }, nil
	}
	bad = uniqueStrings(bad)
	if len(bad) > 5 {
		bad = bad[:5]
	}
	return "", nil, fmt.Errorf("labels must be class indexes 0..%d or class names %q; not recognised: %q",
		len(classNames)-1, classNames, bad)
}

// ── matching ────────────────────────────────────────────────────────────────

// Extensions that mark a file as documentation or metadata rather than data:
// such files are matched only when the ground truth names them exactly.
var docExts = map[string]bool{".csv": true, ".tsv": true, ".json": true, ".jsonl": true, ".txt": true,
	".md": true, ".xlsx": true, ".xls": true, ".xml": true, ".yaml": true, ".yml": true, ".pdf": true,
	".doc": true, ".docx": true, ".ini": true, ".py": true, ".zip": true, ".log": true}

func cleanID(id string) string {
	id = strings.ReplaceAll(strings.TrimSpace(id), "\\", "/")
	for strings.HasPrefix(id, "./") {
		id = id[2:]
	}
	return strings.Trim(id, "/")
}

func stem(name string) string { return strings.TrimSuffix(name, path.Ext(name)) }

func uniqueStrings(s []string) []string {
	sort.Strings(s)
	out := s[:0]
	for i, v := range s {
		if i == 0 || v != s[i-1] {
			out = append(out, v)
		}
	}
	return out
}

// index maps one kind of key to the dataset files carrying it.
type index map[string][]string

func (ix index) add(k, file string) { ix[k] = append(ix[k], file) }

// matcher holds one lookup table per matching rule.
type matcher struct {
	exact, suffix, base, baseFold, stemName, dirs index
}

func newMatcher(files []string) *matcher {
	m := &matcher{exact: index{}, suffix: index{}, base: index{}, baseFold: index{}, stemName: index{}, dirs: index{}}
	dataInDir := map[string][]string{}
	for _, f := range files {
		b := path.Base(f)
		m.exact.add(f, f)
		parts := strings.Split(f, "/")
		for i := 1; i < len(parts)-1; i++ {
			m.suffix.add(strings.Join(parts[i:], "/"), f) // "a/b/c.jpg" -> "b/c.jpg"
		}
		m.base.add(b, f)
		m.baseFold.add(strings.ToLower(b), f)
		if docExts[strings.ToLower(path.Ext(b))] {
			continue
		}
		m.stemName.add(stem(b), f)
		for d := path.Dir(f); d != "."; d = path.Dir(d) {
			dataInDir[d] = append(dataInDir[d], f)
		}
	}
	for d, fs := range dataInDir {
		if len(fs) == 1 { // a folder per sample, e.g. subject_00001/<scan>.dcm
			m.dirs.add(d, fs[0])
			m.dirs.add(path.Base(d), fs[0])
		}
	}
	for k, fs := range m.dirs {
		m.dirs[k] = uniqueStrings(fs) // the same folder name at several places
	}
	return m
}

// errAmbiguous is returned when an id fits more than one dataset file.
type errAmbiguous struct {
	id    string
	files []string
}

func (e *errAmbiguous) Error() string {
	shown := e.files
	if len(shown) > 3 {
		shown = shown[:3]
	}
	return fmt.Sprintf("%q matches %d files %v", e.id, len(e.files), shown)
}

// match returns the dataset file an id refers to and the rule that matched;
// file is "" when nothing matches. Rules run from most to least specific and
// the first one that finds anything decides.
func (m *matcher) match(rawID string) (file, rule string, err error) {
	id := cleanID(rawID)
	if id == "" {
		return "", "", nil
	}
	// The id is a trailing part of a file's path ("Suspicious/a.jpg" for
	// "ocs/Suspicious/a.jpg"), or has extra leading folders itself.
	var suffix []string
	if parts := strings.Split(id, "/"); len(parts) > 1 {
		suffix = append(suffix, m.suffix[id]...)
		for i := 1; i < len(parts); i++ {
			suffix = append(suffix, m.exact[strings.Join(parts[i:], "/")]...)
		}
		suffix = uniqueStrings(suffix)
	}
	b := path.Base(id)
	for _, try := range []struct {
		rule  string
		files []string
	}{
		{"path", m.exact[id]},
		{"path_suffix", suffix},
		{"file_name", m.base[b]},
		{"file_name_case", m.baseFold[strings.ToLower(b)]},
		{"name_without_extension", m.stemName[b]},
		{"name_without_extension", m.stemName[stem(b)]},
		{"folder", m.dirs[id]},
		{"folder", m.dirs[b]},
	} {
		switch len(try.files) {
		case 0:
			continue
		case 1:
			return try.files[0], try.rule, nil
		}
		return "", "", &errAmbiguous{id: rawID, files: try.files}
	}
	return "", "", nil
}

// Match labels every entry and matches it to one of files (paths inside the
// dataset folder, the ground-truth file itself excluded).
func Match(t *Table, classNames []string, files []string) (*Result, error) {
	res := &Result{Total: len(t.Entries), MatchedBy: map[string]int{}}
	var labelled []Entry
	var labels []string
	for _, e := range t.Entries {
		if e.Label == "" {
			res.Unlabeled++
			continue
		}
		if e.ID == "" {
			return nil, fmt.Errorf("ground truth line %d: empty file id", e.Line)
		}
		labelled = append(labelled, e)
		labels = append(labels, e.Label)
	}
	if len(labelled) == 0 {
		return nil, errors.New("ground truth has no labelled rows")
	}
	mode, parse, err := labelParser(labels, classNames)
	if err != nil {
		return nil, fmt.Errorf("ground truth: %w", err)
	}
	res.LabelMode = mode

	m := newMatcher(files)
	type hit struct {
		label, line int
		id          string
	}
	byFile := map[string]hit{}
	matched := 0
	var ambiguous []string
	for _, e := range labelled {
		f, rule, err := m.match(e.ID)
		var amb *errAmbiguous
		switch {
		case errors.As(err, &amb):
			ambiguous = append(ambiguous, amb.Error())
			continue
		case f == "":
			res.Unmatched = append(res.Unmatched, e.ID)
			continue
		}
		matched++
		label := parse(e.Label)
		if prev, dup := byFile[f]; dup {
			if prev.label != label {
				return nil, fmt.Errorf("ground truth lines %d (%q) and %d (%q) both refer to %s with different labels",
					prev.line, prev.id, e.Line, e.ID, f)
			}
			continue // the same sample listed twice
		}
		byFile[f] = hit{label: label, line: e.Line, id: e.ID}
		res.MatchedBy[rule]++
	}
	if len(ambiguous) > 0 {
		if len(ambiguous) > 3 {
			ambiguous = append(ambiguous[:3], "…")
		}
		return nil, fmt.Errorf("ground truth: some ids fit more than one dataset file, name them by their path "+
			"in the dataset instead: %s", strings.Join(ambiguous, "; "))
	}
	if matched == 0 || float64(matched) < MinMatchedFraction*float64(len(labelled)) {
		shown := res.Unmatched
		if len(shown) > 3 {
			shown = shown[:3]
		}
		return nil, fmt.Errorf("ground truth: only %d of %d labelled rows match a dataset file (unmatched e.g. %q); "+
			"ids must name files by path, file name, name without extension, or a folder holding one file",
			matched, len(labelled), shown)
	}
	for f, h := range byFile {
		res.Rows = append(res.Rows, Row{File: f, Label: h.label})
	}
	sort.Slice(res.Rows, func(i, j int) bool { return res.Rows[i].File < res.Rows[j].File })
	return res, nil
}

// MatchMasks is Match for segmentation: every entry's label names the
// sample's mask file. Masks are matched against all files first; the sample
// ids then only against the files that are not masks. A row whose sample or
// mask has no file is skipped and counted, like an unmatched row in Match.
func MatchMasks(t *Table, files []string) (*Result, error) {
	res := &Result{Total: len(t.Entries), MatchedBy: map[string]int{}, LabelMode: "mask", Masks: map[string]string{}}
	var labelled []Entry
	for _, e := range t.Entries {
		if e.Label == "" {
			res.Unlabeled++
			continue
		}
		if e.ID == "" {
			return nil, fmt.Errorf("ground truth line %d: empty file id", e.Line)
		}
		labelled = append(labelled, e)
	}
	if len(labelled) == 0 {
		return nil, errors.New("ground truth has no rows with a mask")
	}

	var ambiguous []string
	note := func(err error) bool {
		var amb *errAmbiguous
		if errors.As(err, &amb) {
			ambiguous = append(ambiguous, amb.Error())
			return true
		}
		return false
	}
	all := newMatcher(files)
	maskOf := make([]string, len(labelled))
	isMask := map[string]bool{}
	for i, e := range labelled {
		f, _, err := all.match(e.Label)
		if note(err) {
			continue
		}
		maskOf[i] = f
		if f != "" {
			isMask[f] = true
		}
	}
	var inputs []string
	for _, f := range files {
		if !isMask[f] {
			inputs = append(inputs, f)
		}
	}
	samples := newMatcher(inputs)

	type hit struct {
		mask, id string
		line     int
	}
	byFile := map[string]hit{}
	matched := 0
	for i, e := range labelled {
		if maskOf[i] == "" {
			res.Unmatched = append(res.Unmatched, e.ID)
			continue
		}
		f, rule, err := samples.match(e.ID)
		if note(err) {
			continue
		}
		if f == "" {
			res.Unmatched = append(res.Unmatched, e.ID)
			continue
		}
		matched++
		if prev, dup := byFile[f]; dup {
			if prev.mask != maskOf[i] {
				return nil, fmt.Errorf("ground truth lines %d (%q) and %d (%q) give %s different masks",
					prev.line, prev.id, e.Line, e.ID, f)
			}
			continue
		}
		byFile[f] = hit{mask: maskOf[i], id: e.ID, line: e.Line}
		res.MatchedBy[rule]++
	}
	if len(ambiguous) > 0 {
		if len(ambiguous) > 3 {
			ambiguous = append(ambiguous[:3], "…")
		}
		return nil, fmt.Errorf("ground truth: some files or masks fit more than one dataset file, name them by "+
			"their path in the dataset instead: %s", strings.Join(ambiguous, "; "))
	}
	if matched == 0 || float64(matched) < MinMatchedFraction*float64(len(labelled)) {
		shown := res.Unmatched
		if len(shown) > 3 {
			shown = shown[:3]
		}
		return nil, fmt.Errorf("ground truth: only %d of %d rows match both a dataset file and its mask "+
			"(unmatched e.g. %q); name files and masks by path, file name or name without extension",
			matched, len(labelled), shown)
	}
	for f, h := range byFile {
		res.Rows = append(res.Rows, Row{File: f})
		res.Masks[f] = h.mask
	}
	sort.Slice(res.Rows, func(i, j int) bool { return res.Rows[i].File < res.Rows[j].File })
	return res, nil
}

// ── output for the Python stages ────────────────────────────────────────────

// WriteCanonical writes the matched rows as file,label for the evaluator.
func WriteCanonical(filePath string, rows []Row) error {
	var buf bytes.Buffer
	w := csv.NewWriter(&buf)
	_ = w.Write([]string{"file", "label"})
	for _, r := range rows {
		_ = w.Write([]string{r.File, strconv.Itoa(r.Label)})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return err
	}
	return os.WriteFile(filePath, buf.Bytes(), 0o644)
}

// WriteCanonicalMasks writes segmentation rows as file,mask for the
// evaluator; mask is the decrypted mask's path on disk.
func WriteCanonicalMasks(filePath string, rows []Row, masks map[string]string, pathOf func(file string) string) error {
	var buf bytes.Buffer
	w := csv.NewWriter(&buf)
	_ = w.Write([]string{"file", "mask"})
	for _, r := range rows {
		m := masks[r.File]
		if m == "" {
			return fmt.Errorf("ground truth row %s has no mask", r.File)
		}
		_ = w.Write([]string{r.File, pathOf(m)})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return err
	}
	return os.WriteFile(filePath, buf.Bytes(), 0o644)
}

// WriteInputs writes one "<file>\t<path on disk>" line per row for the
// inference step; <file> is the sample id the adaptor and evaluator see.
func WriteInputs(filePath string, rows []Row, pathOf func(file string) string) error {
	var b strings.Builder
	for _, r := range rows {
		if strings.ContainsAny(r.File, "\t\n\r") {
			return fmt.Errorf("dataset file name %q contains a tab or line break", r.File)
		}
		b.WriteString(r.File)
		b.WriteByte('\t')
		b.WriteString(pathOf(r.File))
		b.WriteByte('\n')
	}
	return os.WriteFile(filePath, []byte(b.String()), 0o644)
}
