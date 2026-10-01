package groundtruth

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

var binary = []string{"Non-Suspicious", "Suspicious"}

// entries reads a ground-truth file and returns its (id, label) pairs.
func entries(t *testing.T, name, content string) [][2]string {
	t.Helper()
	tab, err := Read(name, []byte(content))
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	var out [][2]string
	for _, e := range tab.Entries {
		out = append(out, [2]string{e.ID, e.Label})
	}
	return out
}

func TestReadFormats(t *testing.T) {
	want := [][2]string{{"a.jpg", "1"}, {"b.jpg", "0"}}
	cases := map[string]struct{ name, content string }{
		"canonical with BOM":        {"gt.csv", "\xef\xbb\xbffile,label\n a.jpg , 1\nb.jpg,0\n"},
		"OCS original headers":      {"gt.csv", "Unique_case_ID,Label\na.jpg,1\nb.jpg,0\n"},
		"other names, extra column": {"gt.csv", "note,Class,Image Name\nx,1,a.jpg\ny,0,b.jpg\n"},
		"semicolons":                {"gt.csv", "filename;target\na.jpg;1\nb.jpg;0\n"},
		"tab separated":             {"gt.tsv", "file\tlabel\na.jpg\t1\nb.jpg\t0\n"},
		"no header row":             {"gt.csv", "a.jpg,1\nb.jpg,0\n"},
		"dotted header":             {"gt.csv", "file.name,label\na.jpg,1\nb.jpg,0\n"},
		"pandas index column":       {"gt.csv", ",new_file_path,tissueden\n0,a.jpg,1\n1,b.jpg,0\n"},
		"blank lines":               {"gt.csv", "file,label\n\na.jpg,1\n\nb.jpg,0\n\n"},
		"json records":              {"gt.json", `[{"image": "a.jpg", "class": 1}, {"image": "b.jpg", "class": 0}]`},
		"json records under a key":  {"gt.json", `{"version": 2, "annotations": [{"file_name": "a.jpg", "label": 1}, {"file_name": "b.jpg", "label": 0}]}`},
		"json map":                  {"gt.json", `{"b.jpg": 0, "a.jpg": 1}`},
		"json map of records":       {"gt.json", `{"a.jpg": {"label": 1, "site": "x"}, "b.jpg": {"label": 0}}`},
		"json pairs":                {"gt.json", `[["a.jpg", 1], ["b.jpg", 0]]`},
		"json lines":                {"gt.jsonl", "{\"file\": \"a.jpg\", \"label\": 1}\n{\"file\": \"b.jpg\", \"label\": 0}\n"},
		"json lines saved as .json": {"gt.json", "{\"file\": \"a.jpg\", \"label\": 1}\n{\"file\": \"b.jpg\", \"label\": 0}\n"},
	}
	for name, c := range cases {
		if got := entries(t, c.name, c.content); !reflect.DeepEqual(got, want) {
			t.Errorf("%s: got %v", name, got)
		}
	}
}

func TestReadHeaderlessWithHintedName(t *testing.T) {
	// "image_001.jpg" contains the id hint "image" but is data, not a header.
	got := entries(t, "gt.csv", "image_001.jpg,1\nimage_002.jpg,0\n")
	if len(got) != 2 || got[0] != [2]string{"image_001.jpg", "1"} {
		t.Fatalf("got %v", got)
	}
}

func TestReadRejects(t *testing.T) {
	cases := map[string]struct{ name, content string }{
		"unrecognisable columns": {"gt.csv", "alpha,beta,gamma\n1,2,3\n"},
		"no rows":                {"gt.csv", "file,label\n"},
		"bad json":               {"gt.json", `[{"file": "a.jpg"`},
		"json without label key": {"gt.json", `[{"file": "a.jpg", "site": "x", "age": 3}]`},
	}
	for name, c := range cases {
		if _, err := Read(c.name, []byte(c.content)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func match(t *testing.T, gt string, classes, files []string) *Result {
	t.Helper()
	tab, err := Read("gt.csv", []byte(gt))
	if err != nil {
		t.Fatal(err)
	}
	res, err := Match(tab, classes, files)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestMatchNestedFolders(t *testing.T) {
	files := []string{
		"root.jpg",
		"Suspicious/a.jpg",
		"Non-Suspicious/b.jpg",
		"batch1/x/c.png",
		"batch2/d.jpeg",
		"subject_00001/scan.dcm", // one file per folder
		"notes/readme.txt",
	}
	gt := "id,label\n" +
		"root.jpg,1\n" + // at the ground truth's level
		"Suspicious/a.jpg,1\n" + // path in the dataset
		"b.jpg,0\n" + // file name only
		"x/c.png,0\n" + // path suffix
		"d,1\n" + // name without extension
		"subject_00001,0\n" // folder holding one file
	res := match(t, gt, binary, files)
	want := []Row{
		{"Non-Suspicious/b.jpg", 0}, {"Suspicious/a.jpg", 1}, {"batch1/x/c.png", 0},
		{"batch2/d.jpeg", 1}, {"root.jpg", 1}, {"subject_00001/scan.dcm", 0},
	}
	if !reflect.DeepEqual(res.Rows, want) {
		t.Fatalf("rows %v", res.Rows)
	}
	if res.MatchedBy["folder"] != 1 || res.MatchedBy["name_without_extension"] != 1 {
		t.Fatalf("matched by %v", res.MatchedBy)
	}
}

func TestMatchSkipsRowsWithoutFiles(t *testing.T) {
	res := match(t, "file,label\na.jpg,1\nb.jpg,0\ngone.jpg,1\nunlabelled.jpg,\n", binary, []string{"a.jpg", "b.jpg", "unlabelled.jpg"})
	if len(res.Rows) != 2 || !reflect.DeepEqual(res.Unmatched, []string{"gone.jpg"}) || res.Unlabeled != 1 {
		t.Fatalf("rows %v unmatched %v unlabeled %d", res.Rows, res.Unmatched, res.Unlabeled)
	}
}

func TestMatchLabels(t *testing.T) {
	files := []string{"a.dcm", "b.dcm", "c.dcm"}
	abcd := []string{"A", "B", "C", "D"}
	if res := match(t, "file,label\na.dcm,A\nb.dcm,d\nc.dcm, c \n", abcd, files); res.LabelMode != "name" ||
		!reflect.DeepEqual(res.Rows, []Row{{"a.dcm", 0}, {"b.dcm", 3}, {"c.dcm", 2}}) {
		t.Fatalf("names: %+v", res)
	}
	if res := match(t, "file,label\na.dcm,0\nb.dcm,3.0\nc.dcm,2\n", abcd, files); res.LabelMode != "index" ||
		res.Rows[1].Label != 3 {
		t.Fatalf("indexes: %+v", res)
	}
	// Names that are numbers: the file means the names, not indexes.
	if res := match(t, "file,label\na.dcm,1\nb.dcm,3\nc.dcm,2\n", []string{"1", "2", "3"}, files); res.LabelMode != "name" ||
		!reflect.DeepEqual(res.Rows, []Row{{"a.dcm", 0}, {"b.dcm", 2}, {"c.dcm", 1}}) {
		t.Fatalf("numeric names: %+v", res)
	}
	if res := match(t, "file,label\na.dcm,Non Suspicious\nb.dcm,suspicious\nc.dcm,SUSPICIOUS\n", binary, files); res.Rows[0].Label != 0 || res.Rows[2].Label != 1 {
		t.Fatalf("name normalisation: %+v", res.Rows)
	}
}

func TestMatchRejects(t *testing.T) {
	files := []string{"one/a.jpg", "two/a.jpg", "b.jpg", "c.jpg"}
	cases := map[string]string{
		"ambiguous file name":        "file,label\na.jpg,1\nb.jpg,0\n",
		"conflicting duplicate":      "file,label\nb.jpg,1\nb.jpg,0\nc.jpg,1\n",
		"unknown label":              "file,label\nb.jpg,maybe\nc.jpg,1\n",
		"label out of range":         "file,label\nb.jpg,2\nc.jpg,1\n",
		"most rows match no file":    "file,label\nb.jpg,1\nx.jpg,0\ny.jpg,1\nz.jpg,0\n",
		"nothing labelled":           "file,label\nb.jpg,\n",
		"labelled row without an id": "file,label\n,1\nb.jpg,0\n",
	}
	for name, gt := range cases {
		tab, err := Read("gt.csv", []byte(gt))
		if err == nil {
			_, err = Match(tab, binary, files)
		}
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// The same sample listed twice with one label is fine.
	if res := match(t, "file,label\nb.jpg,1\nb.jpg,1\nc.jpg,0\n", binary, files); len(res.Rows) != 2 {
		t.Fatalf("duplicate rows %v", res.Rows)
	}
	// A path disambiguates.
	if res := match(t, "file,label\none/a.jpg,1\ntwo/a.jpg,0\n", binary, files); len(res.Rows) != 2 {
		t.Fatalf("paths %v", res.Rows)
	}
}

func TestMatchMasks(t *testing.T) {
	files := []string{"images/a.jpg", "images/b.jpg", "images/c.jpg", "masks/a.png", "masks/b.png", "masks/c.png"}
	cases := map[string]string{
		"paths":                        "image,mask\nimages/a.jpg,masks/a.png\nimages/b.jpg,masks/b.png\nimages/c.jpg,masks/c.png\n",
		"file names":                   "image_path,mask_path\na.jpg,a.png\nb.jpg,b.png\nc.jpg,c.png\n",
		"ids without extension":        "id,segmentation\na,a.png\nb,b.png\nc,c.png\n", // "a" alone fits only images/a.jpg
		"json records":                 `[{"file": "a.jpg", "mask": "masks/a.png"}, {"file": "b.jpg", "mask": "masks/b.png"}, {"file": "c.jpg", "mask": "masks/c.png"}]`,
		"no header, file + mask names": "a.jpg,a.png\nb.jpg,b.png\nc.jpg,c.png\n",
	}
	want := map[string]string{"images/a.jpg": "masks/a.png", "images/b.jpg": "masks/b.png", "images/c.jpg": "masks/c.png"}
	for name, content := range cases {
		file := "gt.csv"
		if content[0] == '[' {
			file = "gt.json"
		}
		tab, err := Read(file, []byte(content))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		res, err := MatchMasks(tab, files)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(res.Rows) != 3 || !reflect.DeepEqual(res.Masks, want) || res.LabelMode != "mask" {
			t.Fatalf("%s: rows %v masks %v", name, res.Rows, res.Masks)
		}
	}

	// Rows whose mask or image is missing are skipped and counted.
	tab, _ := Read("gt.csv", []byte("file,mask\na.jpg,a.png\nb.jpg,b.png\nc.jpg,gone.png\ngone.jpg,c.png\n"))
	res, err := MatchMasks(tab, files)
	if err != nil || len(res.Rows) != 2 || len(res.Unmatched) != 2 {
		t.Fatalf("partial: %v %+v", err, res)
	}

	// A mask named without an extension fits both a.jpg and a.png: an error, not a guess.
	tab, _ = Read("gt.csv", []byte("file,mask\nimages/a.jpg,a\nimages/b.jpg,b.png\n"))
	if _, err := MatchMasks(tab, files); err == nil {
		t.Fatal("ambiguous mask accepted")
	}

	// Too few rows match both: refused.
	tab, _ = Read("gt.csv", []byte("file,mask\na.jpg,x.png\nb.jpg,y.png\nc.jpg,c.png\n"))
	if _, err := MatchMasks(tab, files); err == nil {
		t.Fatal("1 of 3 matched accepted")
	}

	// The same image given two different masks is refused.
	tab, _ = Read("gt.csv", []byte("file,mask\na.jpg,a.png\nimages/a.jpg,b.png\n"))
	if _, err := MatchMasks(tab, files); err == nil {
		t.Fatal("conflicting masks accepted")
	}
}

func TestWriters(t *testing.T) {
	dir := t.TempDir()
	rows := []Row{{"Suspicious/a, b.jpg", 1}, {"c.jpg", 0}}
	if err := WriteCanonical(filepath.Join(dir, "gt.csv"), rows); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(dir, "gt.csv"))
	if string(raw) != "file,label\n\"Suspicious/a, b.jpg\",1\nc.jpg,0\n" {
		t.Fatalf("canonical %q", raw)
	}
	if err := WriteInputs(filepath.Join(dir, "in.txt"), rows, func(f string) string { return "/d/" + f }); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(filepath.Join(dir, "in.txt"))
	if string(raw) != "Suspicious/a, b.jpg\t/d/Suspicious/a, b.jpg\nc.jpg\t/d/c.jpg\n" {
		t.Fatalf("inputs %q", raw)
	}
	if err := WriteInputs(filepath.Join(dir, "bad.txt"), []Row{{"a\tb.jpg", 0}}, func(f string) string { return f }); err == nil {
		t.Fatal("tab in a file name accepted")
	}

	masks := map[string]string{"img/a.jpg": "masks/a.png"}
	if err := WriteCanonicalMasks(filepath.Join(dir, "seg.csv"), []Row{{File: "img/a.jpg"}}, masks,
		func(f string) string { return "/d/" + f }); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(filepath.Join(dir, "seg.csv"))
	if string(raw) != "file,mask\nimg/a.jpg,/d/masks/a.png\n" {
		t.Fatalf("canonical masks %q", raw)
	}
	if err := WriteCanonicalMasks(filepath.Join(dir, "seg2.csv"), []Row{{File: "b.jpg"}}, masks,
		func(f string) string { return f }); err == nil {
		t.Fatal("row without a mask accepted")
	}
}
