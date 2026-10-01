package groundtruth

// Object detection: the ground truth lists boxes, many per image, and images
// with no objects at all. Accepted files (in the dataset's root folder):
//
//	CSV / TSV   one row per box; header names are flexible:
//	              file,x_min,y_min,x_max,y_max,class        (also xmin / x1 / left, ...)
//	              image,x,y,w,h,label                       (top-left corner + size)
//	            an image with no objects: a row with the box fields empty
//	JSON        [{"file": "a.png", "x_min": 10, ..., "class": "kidney"}, ...]   (records)
//	COCO JSON   {"images": [{"id", "file_name"}], "annotations": [{"image_id",
//	             "bbox": [x, y, w, h], "category_id"}], "categories": [{"id", "name"}]}
//	            images without annotations are images with no objects
//
// Coordinates are the original image's pixels. Classes are class names or
// 0-based indexes, decided once for the whole file (as for classification); a
// file without a class column is accepted when the dataset has one class.
// The canonical form for the evaluator is file,x_min,y_min,x_max,y_max,class.

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"
	"sort"
	"strconv"
	"strings"
)

// Box is one ground-truth box in original-image pixels.
type Box struct {
	X1, Y1, X2, Y2 float64
	Class          int
}

// BoxEntry is one ground-truth row as the data provider wrote it.
type BoxEntry struct {
	ID     string
	Coords [4]float64 // x_min, y_min, x_max, y_max (already converted from x,y,w,h)
	Label  string
	Empty  bool // an image listed with no objects
	Line   int
}

// BoxTable is a parsed detection ground-truth file.
type BoxTable struct {
	Format      string // csv, tsv, json or coco
	IDColumn    string
	BoxFormat   string // xyxy or xywh, as written
	LabelColumn string // "" when the file has no class column
	Entries     []BoxEntry
}

var (
	boxX1 = []string{"x_min", "xmin", "x1", "left", "x0", "bbox_x_min", "box_x_min"}
	boxY1 = []string{"y_min", "ymin", "y1", "top", "y0", "bbox_y_min", "box_y_min"}
	boxX2 = []string{"x_max", "xmax", "x2", "right", "bbox_x_max", "box_x_max"}
	boxY2 = []string{"y_max", "ymax", "y2", "bottom", "bbox_y_max", "box_y_max"}
	boxX  = []string{"x", "bbox_x", "box_x"}
	boxY  = []string{"y", "bbox_y", "box_y"}
	boxW  = []string{"w", "width", "bbox_w", "bbox_width", "box_w", "box_width"}
	boxH  = []string{"h", "height", "bbox_h", "bbox_height", "box_h", "box_height"}
	// Box columns win over these when picking the file and class columns.
	boxIDNames = []string{"file", "filename", "file_name", "filepath", "file_path", "path", "image", "image_id",
		"image_name", "image_path", "img", "id", "name", "sample", "sample_id", "case_id"}
	boxClassNames = []string{"class", "label", "category", "class_id", "category_id", "class_name", "category_name",
		"class_index", "label_id", "cls", "target"}
)

// ReadBoxes parses a detection ground-truth file.
func ReadBoxes(name string, data []byte) (*BoxTable, error) {
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	var (
		t   *BoxTable
		err error
	)
	switch strings.ToLower(path.Ext(name)) {
	case ".json":
		t, err = readBoxesJSON(data)
	default:
		t, err = readBoxesDelimited(name, data)
	}
	if err != nil {
		return nil, fmt.Errorf("ground truth %s: %w", name, err)
	}
	if len(t.Entries) == 0 {
		return nil, fmt.Errorf("ground truth %s has no rows", name)
	}
	return t, nil
}

func readBoxesDelimited(name string, data []byte) (*BoxTable, error) {
	delim := sniffDelimiter(data)
	if strings.EqualFold(path.Ext(name), ".tsv") {
		delim = '\t'
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
	format := "csv"
	if delim == '\t' {
		format = "tsv"
	}
	return boxesFromRows(format, records[0], records[1:], 2)
}

// boxesFromRows reads box rows given a header; firstLine numbers rows in errors.
func boxesFromRows(format string, header []string, rows [][]string, firstLine int) (*BoxTable, error) {
	norm := make([]string, len(header))
	for i, h := range header {
		norm[i] = normName(h)
	}
	used := map[int]bool{}
	find := func(names []string) int {
		for _, n := range names {
			for i, h := range norm {
				if !used[i] && h == n {
					return i
				}
			}
		}
		return -1
	}
	t := &BoxTable{Format: format, BoxFormat: "xyxy"}
	cols := []int{find(boxX1), find(boxY1), find(boxX2), find(boxY2)}
	if cols[0] < 0 || cols[1] < 0 || cols[2] < 0 || cols[3] < 0 {
		t.BoxFormat = "xywh"
		x, y := cols[0], cols[1]
		if x < 0 {
			x = find(boxX)
		}
		if y < 0 {
			y = find(boxY)
		}
		cols = []int{x, y, find(boxW), find(boxH)}
		if cols[0] < 0 || cols[1] < 0 || cols[2] < 0 || cols[3] < 0 {
			return nil, fmt.Errorf("cannot find the box columns in header %q (name them x_min,y_min,x_max,y_max "+
				"or x,y,w,h)", header)
		}
	}
	for _, c := range cols {
		used[c] = true
	}
	idCol := find(boxIDNames)
	if idCol < 0 {
		return nil, fmt.Errorf("cannot find the file column in header %q (name it e.g. file or image)", header)
	}
	used[idCol] = true
	t.IDColumn = strings.TrimSpace(header[idCol])
	labelCol := find(boxClassNames)
	if labelCol >= 0 {
		t.LabelColumn = strings.TrimSpace(header[labelCol])
	}
	cell := func(rec []string, i int) string {
		if i >= 0 && i < len(rec) {
			return strings.TrimSpace(rec[i])
		}
		return ""
	}
	for n, rec := range rows {
		line := firstLine + n
		e := BoxEntry{ID: cell(rec, idCol), Label: cell(rec, labelCol), Line: line}
		vals := []string{cell(rec, cols[0]), cell(rec, cols[1]), cell(rec, cols[2]), cell(rec, cols[3])}
		if e.ID == "" && strings.Join(vals, "") == "" && e.Label == "" {
			continue // blank line
		}
		if strings.Join(vals, "") == "" {
			e.Empty = true
			t.Entries = append(t.Entries, e)
			continue
		}
		for k, v := range vals {
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: %s is not a number: %q", line, header[cols[k]], v)
			}
			e.Coords[k] = f
		}
		if t.BoxFormat == "xywh" {
			e.Coords[2] += e.Coords[0]
			e.Coords[3] += e.Coords[1]
		}
		t.Entries = append(t.Entries, e)
	}
	return t, nil
}

func readBoxesJSON(data []byte) (*BoxTable, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("not valid JSON: %w", err)
	}
	if obj, ok := v.(map[string]any); ok {
		if _, hasImages := obj["images"]; hasImages {
			if _, hasAnns := obj["annotations"]; hasAnns {
				return readCOCO(obj)
			}
		}
		for _, k := range sortedKeys(obj) {
			if list, ok := obj[k].([]any); ok && len(list) > 0 {
				if _, isObj := list[0].(map[string]any); isObj {
					v = list
					break
				}
			}
		}
	}
	items, ok := v.([]any)
	if !ok {
		return nil, errors.New("expected COCO JSON or a list of box records")
	}
	keySet := map[string]any{}
	for i, it := range items {
		obj, ok := it.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("item %d is not a record", i+1)
		}
		for k := range obj {
			keySet[k] = nil
		}
	}
	header := sortedKeys(keySet)
	rows := make([][]string, len(items))
	for i, it := range items {
		obj := it.(map[string]any)
		rows[i] = make([]string, len(header))
		for j, k := range header {
			rows[i][j], _ = scalarString(obj[k])
		}
	}
	t, err := boxesFromRows("json", header, rows, 1)
	return t, err
}

func readCOCO(obj map[string]any) (*BoxTable, error) {
	t := &BoxTable{Format: "coco", IDColumn: "images.file_name", BoxFormat: "xywh", LabelColumn: "categories.name"}
	num := func(v any) (float64, bool) {
		if n, ok := v.(json.Number); ok {
			f, err := n.Float64()
			return f, err == nil
		}
		return 0, false
	}
	key := func(v any) string { s, _ := scalarString(v); return s }
	catName := map[string]string{}
	cats, _ := obj["categories"].([]any)
	for _, c := range cats {
		if m, ok := c.(map[string]any); ok {
			if name := strings.TrimSpace(key(m["name"])); name != "" {
				catName[key(m["id"])] = name
			}
		}
	}
	if len(catName) == 0 {
		t.LabelColumn = "annotations.category_id"
	}
	files := map[string]string{}
	var order []string
	images, _ := obj["images"].([]any)
	for i, im := range images {
		m, ok := im.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("images[%d] is not an object", i)
		}
		id, file := key(m["id"]), strings.TrimSpace(key(m["file_name"]))
		if file == "" {
			return nil, fmt.Errorf("images[%d] has no file_name", i)
		}
		files[id] = file
		order = append(order, id)
	}
	withBoxes := map[string]bool{}
	anns, _ := obj["annotations"].([]any)
	for i, a := range anns {
		m, ok := a.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("annotations[%d] is not an object", i)
		}
		file, ok := files[key(m["image_id"])]
		if !ok {
			return nil, fmt.Errorf("annotations[%d] refers to image_id %s, which is not in images", i, key(m["image_id"]))
		}
		bbox, _ := m["bbox"].([]any)
		if len(bbox) != 4 {
			return nil, fmt.Errorf("annotations[%d]: bbox must be [x, y, width, height]", i)
		}
		e := BoxEntry{ID: file, Line: i + 1}
		for k := range bbox {
			f, ok := num(bbox[k])
			if !ok {
				return nil, fmt.Errorf("annotations[%d]: bbox value %v is not a number", i, bbox[k])
			}
			e.Coords[k] = f
		}
		e.Coords[2] += e.Coords[0]
		e.Coords[3] += e.Coords[1]
		cat := key(m["category_id"])
		if name, ok := catName[cat]; ok {
			e.Label = name
		} else {
			e.Label = cat
		}
		t.Entries = append(t.Entries, e)
		withBoxes[file] = true
	}
	for _, id := range order {
		if !withBoxes[files[id]] {
			t.Entries = append(t.Entries, BoxEntry{ID: files[id], Empty: true})
		}
	}
	return t, nil
}

// MatchBoxes matches every entry's image to one of files and reads its class.
// Result.Rows has one row per image (Label unused); Result.Boxes its boxes,
// empty for an image with no objects.
func MatchBoxes(t *BoxTable, classNames []string, files []string) (*Result, error) {
	res := &Result{Total: len(t.Entries), MatchedBy: map[string]int{}, Boxes: map[string][]Box{}}
	var labels []string
	for _, e := range t.Entries {
		if e.ID == "" {
			return nil, fmt.Errorf("ground truth line %d: empty file id", e.Line)
		}
		if e.Empty {
			continue
		}
		if e.Coords[2] < e.Coords[0] || e.Coords[3] < e.Coords[1] {
			return nil, fmt.Errorf("ground truth line %d: box has x_max < x_min or y_max < y_min", e.Line)
		}
		labels = append(labels, e.Label)
	}
	parse := func(string) int { return 0 }
	switch {
	case t.LabelColumn == "" && len(classNames) == 1:
		res.LabelMode = "single"
	case t.LabelColumn == "":
		return nil, fmt.Errorf("ground truth: no class column, and the dataset has %d classes", len(classNames))
	case len(labels) == 0:
		res.LabelMode = "none"
	default:
		mode, p, err := labelParser(labels, classNames)
		if err != nil {
			return nil, fmt.Errorf("ground truth: %w", err)
		}
		res.LabelMode, parse = mode, p
	}

	m := newMatcher(files)
	fileOf := map[string]string{} // id as written -> dataset file
	var ambiguous []string
	unmatched := map[string]bool{}
	for _, e := range t.Entries {
		if _, done := fileOf[e.ID]; done || unmatched[e.ID] {
			continue
		}
		f, rule, err := m.match(e.ID)
		var amb *errAmbiguous
		switch {
		case errors.As(err, &amb):
			ambiguous = append(ambiguous, amb.Error())
			unmatched[e.ID] = true
			continue
		case f == "":
			unmatched[e.ID] = true
			res.Unmatched = append(res.Unmatched, e.ID)
			continue
		}
		fileOf[e.ID] = f
		res.MatchedBy[rule]++
	}
	if len(ambiguous) > 0 {
		if len(ambiguous) > 3 {
			ambiguous = append(ambiguous[:3], "…")
		}
		return nil, fmt.Errorf("ground truth: some ids fit more than one dataset file, name them by their path "+
			"in the dataset instead: %s", strings.Join(ambiguous, "; "))
	}
	images := len(fileOf) + len(unmatched)
	if len(fileOf) == 0 || float64(len(fileOf)) < MinMatchedFraction*float64(images) {
		shown := res.Unmatched
		if len(shown) > 3 {
			shown = shown[:3]
		}
		return nil, fmt.Errorf("ground truth: only %d of %d images match a dataset file (unmatched e.g. %q); "+
			"ids must name files by path, file name, name without extension, or a folder holding one file",
			len(fileOf), images, shown)
	}
	for _, e := range t.Entries {
		f, ok := fileOf[e.ID]
		if !ok {
			continue
		}
		if _, seen := res.Boxes[f]; !seen {
			res.Boxes[f] = nil
			res.Rows = append(res.Rows, Row{File: f})
		}
		if !e.Empty {
			res.Boxes[f] = append(res.Boxes[f], Box{X1: e.Coords[0], Y1: e.Coords[1], X2: e.Coords[2], Y2: e.Coords[3],
				Class: parse(e.Label)})
		}
	}
	sort.Slice(res.Rows, func(i, j int) bool { return res.Rows[i].File < res.Rows[j].File })
	return res, nil
}

// WriteCanonicalBoxes writes the matched boxes as file,x_min,y_min,x_max,y_max,class
// for the evaluator; an image without objects gets one row with empty box fields.
func WriteCanonicalBoxes(filePath string, rows []Row, boxes map[string][]Box) error {
	var buf bytes.Buffer
	w := csv.NewWriter(&buf)
	_ = w.Write([]string{"file", "x_min", "y_min", "x_max", "y_max", "class"})
	f := func(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
	for _, r := range rows {
		if len(boxes[r.File]) == 0 {
			_ = w.Write([]string{r.File, "", "", "", "", ""})
			continue
		}
		for _, b := range boxes[r.File] {
			_ = w.Write([]string{r.File, f(b.X1), f(b.Y1), f(b.X2), f(b.Y2), strconv.Itoa(b.Class)})
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return err
	}
	return os.WriteFile(filePath, buf.Bytes(), 0o644)
}

// BoxCount is the number of boxes in a matched detection ground truth.
func (r *Result) BoxCount() int {
	n := 0
	for _, b := range r.Boxes {
		n += len(b)
	}
	return n
}
