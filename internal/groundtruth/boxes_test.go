package groundtruth

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

var boxFiles = []string{"scans/a.png", "scans/b.png", "scans/c.png"}

func matchBoxes(t *testing.T, name, content string, classes []string) *Result {
	t.Helper()
	tab, err := ReadBoxes(name, []byte(content))
	if err != nil {
		t.Fatalf("%s: read: %v", name, err)
	}
	res, err := MatchBoxes(tab, classes, boxFiles)
	if err != nil {
		t.Fatalf("%s: match: %v", name, err)
	}
	return res
}

func TestBoxFormats(t *testing.T) {
	classes := []string{"kidney", "cyst"}
	want := map[string][]Box{
		"scans/a.png": {{X1: 10, Y1: 20, X2: 50, Y2: 80, Class: 0}, {X1: 5, Y1: 5, X2: 15, Y2: 25, Class: 1}},
		"scans/b.png": nil, // no objects
		"scans/c.png": {{X1: 0, Y1: 0, X2: 100, Y2: 40, Class: 0}},
	}
	cases := map[string]struct{ name, content string }{
		"xyxy, names": {"gt.csv", "file,x_min,y_min,x_max,y_max,class\n" +
			"a.png,10,20,50,80,kidney\na.png,5,5,15,25,cyst\nb.png,,,,,\nc.png,0,0,100,40,Kidney\n"},
		"xyxy, indexes, other names": {"gt.csv", "image;xmin;ymin;xmax;ymax;label\n" +
			"scans/a.png;10;20;50;80;0\nscans/a.png;5;5;15;25;1\nscans/b.png;;;;;\nscans/c.png;0;0;100;40;0\n"},
		"xywh": {"gt.tsv", "image_id\tx\ty\tw\th\tcategory\n" +
			"a\t10\t20\t40\t60\tkidney\na\t5\t5\t10\t20\tcyst\nb\t\t\t\t\t\nc\t0\t0\t100\t40\tkidney\n"},
		"json records": {"gt.json", `[{"file":"a.png","x_min":10,"y_min":20,"x_max":50,"y_max":80,"class":"kidney"},
			{"file":"a.png","x_min":5,"y_min":5,"x_max":15,"y_max":25,"class":"cyst"},
			{"file":"b.png"},
			{"file":"c.png","x_min":0,"y_min":0,"x_max":100,"y_max":40,"class":"kidney"}]`},
		"coco": {"gt.json", `{"images":[{"id":1,"file_name":"a.png"},{"id":2,"file_name":"b.png"},{"id":3,"file_name":"c.png"}],
			"annotations":[{"id":1,"image_id":1,"bbox":[10,20,40,60],"category_id":7},
			               {"id":2,"image_id":1,"bbox":[5,5,10,20],"category_id":9},
			               {"id":3,"image_id":3,"bbox":[0,0,100,40],"category_id":7}],
			"categories":[{"id":7,"name":"kidney"},{"id":9,"name":"cyst"}]}`},
	}
	for name, c := range cases {
		res := matchBoxes(t, c.name, c.content, classes)
		if !reflect.DeepEqual(res.Boxes, want) || len(res.Rows) != 3 {
			t.Fatalf("%s: boxes %+v rows %v", name, res.Boxes, res.Rows)
		}
	}
}

func TestBoxesSingleClassWithoutClassColumn(t *testing.T) {
	res := matchBoxes(t, "gt.csv", "file,x_min,y_min,x_max,y_max\na.png,1,2,3,4\nb.png,,,,\n", []string{"kidney"})
	if res.LabelMode != "single" || len(res.Boxes["scans/a.png"]) != 1 || res.BoxCount() != 1 {
		t.Fatalf("%+v", res)
	}
}

func TestBoxesRejects(t *testing.T) {
	two := []string{"kidney", "cyst"}
	cases := map[string]string{
		"no class column, two classes": "file,x_min,y_min,x_max,y_max\na.png,1,2,3,4\n",
		"inverted box":                 "file,x_min,y_min,x_max,y_max,class\na.png,10,2,3,4,kidney\n",
		"unknown class":                "file,x_min,y_min,x_max,y_max,class\na.png,1,2,3,4,liver\n",
		"too few images match":         "file,x_min,y_min,x_max,y_max,class\nx.png,1,2,3,4,kidney\ny.png,1,2,3,4,kidney\na.png,1,2,3,4,kidney\n",
	}
	for name, content := range cases {
		tab, err := ReadBoxes("gt.csv", []byte(content))
		if err == nil {
			_, err = MatchBoxes(tab, two, boxFiles)
		}
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := ReadBoxes("gt.csv", []byte("file,class\na.png,kidney\n")); err == nil {
		t.Error("table without box columns accepted")
	}
	if _, err := ReadBoxes("gt.csv", []byte("file,x_min,y_min,x_max,y_max\na.png,1,two,3,4\n")); err == nil {
		t.Error("non-numeric coordinate accepted")
	}
}

func TestWriteCanonicalBoxes(t *testing.T) {
	dir := t.TempDir()
	rows := []Row{{File: "a.png"}, {File: "b.png"}}
	boxes := map[string][]Box{"a.png": {{X1: 1.5, Y1: 2, X2: 30, Y2: 40.25, Class: 1}}}
	if err := WriteCanonicalBoxes(filepath.Join(dir, "gt.csv"), rows, boxes); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(filepath.Join(dir, "gt.csv"))
	if string(raw) != "file,x_min,y_min,x_max,y_max,class\na.png,1.5,2,30,40.25,1\nb.png,,,,,\n" {
		t.Fatalf("%q", raw)
	}
}

// Renal Health's benchmark_ground_truth.csv (columns from its Croissant
// card): the image is named by dcm_path, which carries an extra leading
// folder; image_width must not be taken for the file column; one class and
// no class column.
func TestBoxesRenalHealthLayout(t *testing.T) {
	files := []string{
		"automated_labels_benchmarking/RHV_0040_RIGHT/RHV_0040_RIGHT.dcm",
		"manual_annotation_finalized/RHV_0042_LEFT/RHV_0042_LEFT.dcm",
	}
	gt := "patient_id,side,hospital,source,dcm_path,image_width,image_height,xmin,ymin,xmax,ymax,depth_cm,cone_angle_deg\n" +
		"RHV_0040,RIGHT,Hosp4,automated,Benchmarking_Confirmed/automated_labels_benchmarking/RHV_0040_RIGHT/RHV_0040_RIGHT.dcm,860,725,191.5,194.6,610.6,412.8,19.4,55.0\n" +
		"RHV_0042,LEFT,Hosp4,manual,Benchmarking_Confirmed/manual_annotation_finalized/RHV_0042_LEFT/RHV_0042_LEFT.dcm,900,700,100,120,500,400,,62.8\n"
	tab, err := ReadBoxes("benchmark_ground_truth.csv", []byte(gt))
	if err != nil {
		t.Fatal(err)
	}
	if tab.IDColumn != "dcm_path" || tab.LabelColumn != "" {
		t.Fatalf("id column %q, label column %q", tab.IDColumn, tab.LabelColumn)
	}
	res, err := MatchBoxes(tab, []string{"kidney"}, files)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]Box{
		files[0]: {{X1: 191.5, Y1: 194.6, X2: 610.6, Y2: 412.8}},
		files[1]: {{X1: 100, Y1: 120, X2: 500, Y2: 400}},
	}
	if !reflect.DeepEqual(res.Boxes, want) || res.MatchedBy["path_suffix"] != 2 {
		t.Fatalf("boxes %v, matched by %v", res.Boxes, res.MatchedBy)
	}
}
