package groundtruth

import (
	"strings"
	"testing"
)

func TestParseCanonical(t *testing.T) {
	in := "\xef\xbb\xbffile,label\n a.jpg , 1\nb.jpg,0\n"
	rows, err := Parse(strings.NewReader(in), 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0] != (Row{"a.jpg", 1}) || rows[1] != (Row{"b.jpg", 0}) {
		t.Fatalf("rows = %+v", rows)
	}
}

func TestParseColumnOrderAndExtraColumns(t *testing.T) {
	rows, err := Parse(strings.NewReader("label,note,File\n3,x,s1.dcm\n"), 4)
	if err != nil || rows[0] != (Row{"s1.dcm", 3}) {
		t.Fatalf("rows=%+v err=%v", rows, err)
	}
}

func TestParseRejects(t *testing.T) {
	cases := map[string]string{
		"missing file column":  "name,label\na.jpg,0\n",
		"missing label column": "file,class\na.jpg,0\n",
		"duplicate file":       "file,label\na.jpg,0\na.jpg,1\n",
		"label out of range":   "file,label\na.jpg,2\n",
		"negative label":       "file,label\na.jpg,-1\n",
		"letter label":         "file,label\na.jpg,A\n",
		"empty file":           "file,label\n,0\n",
		"no rows":              "file,label\n",
	}
	for name, in := range cases {
		if _, err := Parse(strings.NewReader(in), 2); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestResolve(t *testing.T) {
	rows := []Row{{"b.jpg", 0}, {"a.jpg", 1}}
	files := map[string]string{"a.jpg": "/d/a.jpg", "b.jpg": "/d/b.jpg", "extra.jpg": "/d/extra.jpg"}
	paths, err := Resolve(rows, files)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(paths, ",") != "/d/b.jpg,/d/a.jpg" {
		t.Fatalf("paths %v not in ground-truth order", paths)
	}
	if _, err := Resolve([]Row{{"missing.jpg", 0}}, files); err == nil || !strings.Contains(err.Error(), "missing.jpg") {
		t.Fatalf("missing file not reported: %v", err)
	}
}
