package pipeline

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/textproto"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/datakaveri/tanuh-processing-tee/internal/config"
	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
)

// v3Payload is a payload v3 under construction: tests change the metadata or
// the parts, then build the body.
type v3Payload struct {
	meta  map[string]any
	parts []v3Part // after the metadata part, in order
}

type v3Part struct {
	name string
	data []byte
}

// newV3Payload encrypts each upload the way the browser does (small chunks so
// every stream has several) and lists it in the metadata.
func newV3Payload(t *testing.T, uploads map[string][]byte) *v3Payload {
	t.Helper()
	p := &v3Payload{meta: map[string]any{
		"payload_version": 3,
		"job_id":          "job-abc123",
		"dataset_id":      "d8c50b6b-af58-4f9d-9af8-4d42bfb1c4bb",
		"model_format":    "onnx",
		"keycloak_token":  "secret-token",
		"buffer_job_url":  "https://buffer/v1/jobs/job-abc123",
	}}
	artifacts := map[string]any{}
	for _, slot := range uploadSlots { // the Buffer's order
		plain, ok := uploads[slot.Name]
		if !ok {
			continue
		}
		key, iv := make([]byte, 32), make([]byte, 12)
		rand.Read(key) //nolint:errcheck
		rand.Read(iv)  //nolint:errcheck
		stream, err := crypto.EncodeChunked(key, iv, "file-"+slot.Name, plain, max(7, len(plain)/4))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(plain)
		artifacts[slot.Name] = map[string]any{
			"encoding":        crypto.ChunkedEncoding,
			"sha256":          hex.EncodeToString(sum[:]),
			"ciphertext_size": len(stream),
			"aes_key":         base64.StdEncoding.EncodeToString(key),
		}
		p.parts = append(p.parts, v3Part{slot.Name, stream})
	}
	p.meta["artifacts"] = artifacts
	return p
}

func (p *v3Payload) artifact(slot string) map[string]any {
	return p.meta["artifacts"].(map[string]any)[slot].(map[string]any)
}

// body returns the multipart body and its boundary; metadataAt places the
// metadata part (0 = first, as the Buffer sends it).
func (p *v3Payload) body(t *testing.T, metadataAt int) ([]byte, string) {
	t.Helper()
	meta, err := json.Marshal(p.meta)
	if err != nil {
		t.Fatal(err)
	}
	parts := append([]v3Part{}, p.parts...)
	parts = append(parts[:metadataAt], append([]v3Part{{"metadata", meta}}, parts[metadataAt:]...)...)
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for _, part := range parts {
		h := textproto.MIMEHeader{}
		h.Set("Content-Disposition", fmt.Sprintf(`form-data; name=%q`, part.name))
		w, err := mw.CreatePart(h)
		if err != nil {
			t.Fatal(err)
		}
		w.Write(part.data) //nolint:errcheck
	}
	mw.Close()
	return buf.Bytes(), mw.Boundary()
}

func intakeManager(t *testing.T) *Manager {
	t.Helper()
	m, err := NewManager(config.Config{BaseDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

var testUploads = map[string][]byte{
	"model":      append([]byte{0x08}, bytes.Repeat([]byte("onnx-model"), 10)...),
	"weights":    bytes.Repeat([]byte("w"), 50),
	"adaptor":    []byte("print('adaptor')"),
	"input_spec": []byte(`{"input_size": [224, 224]}`),
}

func TestReceiveV3(t *testing.T) {
	m := intakeManager(t)
	body, boundary := newV3Payload(t, testUploads).body(t, 0)
	req, err := m.ReceiveV3(bytes.NewReader(body), boundary)
	if err != nil {
		t.Fatal(err)
	}
	if req.JobID != "job-abc123" || req.KeycloakToken != "secret-token" || req.ModelFormat != "onnx" {
		t.Fatalf("request = %+v", req)
	}
	for slot, want := range testUploads {
		a := req.Artifacts[slot]
		got, err := os.ReadFile(a.Path)
		if err != nil || !bytes.Equal(got, want) || a.Size != int64(len(want)) {
			t.Errorf("%s: got %q (size %d), want %q", slot, got, a.Size, want)
		}
		if filepath.Dir(a.Path) != m.receivedDir(req.JobID) {
			t.Errorf("%s written to %s", slot, a.Path)
		}
	}
	if ok, _ := m.TryStartJob("another"); ok {
		t.Fatal("job slot not held after a successful intake")
	}

	// The request goes on to materialize as RunJob would.
	if _, err := m.materialize(req); err != nil {
		t.Fatal(err)
	}
}

func TestReceiveV3Rejects(t *testing.T) {
	cases := map[string]func(p *v3Payload) (metadataAt int){
		"metadata not first": func(p *v3Payload) int { return 1 },
		"unknown part": func(p *v3Payload) int {
			p.parts = append(p.parts, v3Part{"extra", []byte("x")})
			return 0
		},
		"duplicate part": func(p *v3Payload) int {
			p.parts = append(p.parts, p.parts[0])
			return 0
		},
		"listed part not sent": func(p *v3Payload) int {
			p.parts = p.parts[:len(p.parts)-1]
			return 0
		},
		"unknown slot in metadata": func(p *v3Payload) int {
			p.meta["artifacts"].(map[string]any)["evil"] = p.artifact("model")
			return 0
		},
		"no adaptor": func(p *v3Payload) int {
			delete(p.meta["artifacts"].(map[string]any), "adaptor")
			p.parts = p.parts[:1]
			return 0
		},
		"sha256 mismatch": func(p *v3Payload) int {
			p.artifact("adaptor")["sha256"] = strings.Repeat("0", 64)
			return 0
		},
		"sha256 missing": func(p *v3Payload) int {
			delete(p.artifact("model"), "sha256")
			return 0
		},
		"ciphertext_size wrong": func(p *v3Payload) int {
			p.artifact("model")["ciphertext_size"] = p.artifact("model")["ciphertext_size"].(int) + 1
			return 0
		},
		"trailing bytes in a part": func(p *v3Payload) int {
			p.parts[0].data = append(p.parts[0].data, 0)
			p.artifact("model")["ciphertext_size"] = len(p.parts[0].data)
			return 0
		},
		"flipped ciphertext byte": func(p *v3Payload) int {
			p.parts[0].data[len(p.parts[0].data)-1] ^= 1
			return 0
		},
		"wrong key": func(p *v3Payload) int {
			p.artifact("model")["aes_key"] = p.artifact("adaptor")["aes_key"]
			return 0
		},
		"short key": func(p *v3Payload) int {
			p.artifact("model")["aes_key"] = base64.StdEncoding.EncodeToString(make([]byte, 16))
			return 0
		},
		"other encoding": func(p *v3Payload) int {
			p.artifact("model")["encoding"] = "AES-256-GCM"
			return 0
		},
		"payload v2": func(p *v3Payload) int {
			p.meta["payload_version"] = 2
			return 0
		},
		"path in job_id": func(p *v3Payload) int {
			p.meta["job_id"] = "../evil"
			return 0
		},
		"ciphertext_size zero": func(p *v3Payload) int {
			p.artifact("model")["ciphertext_size"] = 0
			return 0
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			m := intakeManager(t)
			p := newV3Payload(t, map[string][]byte{"model": testUploads["model"], "adaptor": testUploads["adaptor"]})
			body, boundary := p.body(t, mutate(p))
			_, err := m.ReceiveV3(bytes.NewReader(body), boundary)
			var ie *IntakeError
			if !errors.As(err, &ie) || ie.Status != 400 {
				t.Fatalf("err = %v, want a 400 intake error", err)
			}
			t.Log(err)
			if ok, _ := m.TryStartJob("next"); !ok {
				t.Error("job slot still held after a rejected intake")
			}
			if _, err := os.Stat(m.jobDir("job-abc123")); !os.IsNotExist(err) {
				t.Errorf("job directory left behind: %v", err)
			}
		})
	}
}

// File sizes are not capped at intake (the UI caps uploads): a 2 MiB adaptor,
// over the Buffer's old 1 MiB limit, is accepted.
func TestReceiveV3NoFileSizeCap(t *testing.T) {
	m := intakeManager(t)
	big := bytes.Repeat([]byte("# comment\n"), 200<<10)
	body, boundary := newV3Payload(t, map[string][]byte{"model": testUploads["model"], "adaptor": big}).body(t, 0)
	req, err := m.ReceiveV3(bytes.NewReader(body), boundary)
	if err != nil {
		t.Fatal(err)
	}
	if req.Artifacts["adaptor"].Size != int64(len(big)) {
		t.Fatalf("adaptor size %d", req.Artifacts["adaptor"].Size)
	}
}

// readCounter counts how much of the body the intake consumed.
type readCounter struct {
	r io.Reader
	n int
}

func (c *readCounter) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}

func TestReceiveV3BusyBeforeBulkBytes(t *testing.T) {
	m := intakeManager(t)
	if ok, _ := m.TryStartJob("job-running"); !ok {
		t.Fatal("could not take the slot")
	}
	body, boundary := newV3Payload(t, map[string][]byte{"model": append([]byte{0x08}, make([]byte, 4<<20)...),
		"adaptor": testUploads["adaptor"]}).body(t, 0)
	r := &readCounter{r: bytes.NewReader(body)}
	_, err := m.ReceiveV3(r, boundary)
	var ie *IntakeError
	if !errors.As(err, &ie) || ie.Status != 409 || !strings.Contains(ie.Msg, "job-running") {
		t.Fatalf("err = %v, want 409 naming the running job", err)
	}
	if r.n > 64<<10 {
		t.Errorf("read %d of %d bytes before answering busy", r.n, len(body))
	}
	if _, err := os.Stat(m.jobDir("job-abc123")); !os.IsNotExist(err) {
		t.Error("a busy intake created the job directory")
	}
}

func TestReceiveV3DiskFull(t *testing.T) {
	saved := diskFree
	diskFree = func(string) (uint64, error) { return 100, nil }
	defer func() { diskFree = saved }()

	m := intakeManager(t)
	body, boundary := newV3Payload(t, testUploads).body(t, 0)
	_, err := m.ReceiveV3(bytes.NewReader(body), boundary)
	var ie *IntakeError
	if !errors.As(err, &ie) || ie.Status != 507 {
		t.Fatalf("err = %v, want 507", err)
	}
	if ok, _ := m.TryStartJob("next"); !ok {
		t.Error("job slot taken by a refused intake")
	}
}
