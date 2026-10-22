package pipeline

// intake.go receives a payload v3 stream from the Buffer TEE:
//
//	part "metadata"  JSON: job fields + per artifact {encoding, sha256,
//	                 ciphertext_size, aes_key}
//	part "<slot>"    the artifact's AES-256-GCM-CHUNKED-v1 stream, exactly as
//	                 the browser uploaded it to the Buffer
//
// Each artifact is decrypted chunk by chunk straight to disk and checked
// against the SHA-256 the browser committed to at submit, so memory stays at a
// couple of chunks whatever the model's size. The AES keys arrive inside the
// RA-TLS channel and live only for the duration of the intake.
//
// Only memory is bounded (the metadata part here, the header and chunk sizes
// in crypto.DecryptChunkedStream). File sizes are not: the free-disk check
// refuses what cannot fit, and the UI caps what users upload.

import (
	"bufio"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/datakaveri/tanuh-processing-tee/internal/crypto"
)

// maxMetadataBytes bounds the metadata part, which is parsed whole. Real
// metadata (a token and a few hashes) is about 2 KB.
const maxMetadataBytes = 64 << 10

// IntakeError is a payload the TEE refuses before taking the job on. Status is
// the HTTP status returned to the Buffer TEE: 409 means "busy, try later";
// any other 4xx means the payload can never be accepted.
type IntakeError struct {
	Status int
	Msg    string
}

func (e *IntakeError) Error() string { return e.Msg }

func reject(status int, format string, a ...any) error {
	return &IntakeError{Status: status, Msg: fmt.Sprintf(format, a...)}
}

// v3Metadata is the payload's first part.
type v3Metadata struct {
	PayloadVersion  int                   `json:"payload_version"`
	JobID           string                `json:"job_id"`
	DatasetID       string                `json:"dataset_id"`
	ModelFormat     string                `json:"model_format"`
	KeycloakToken   string                `json:"keycloak_token"`
	BufferJobURL    string                `json:"buffer_job_url"`
	SubmittedBy     string                `json:"submitted_by"`
	SubmittedAtUnix int64                 `json:"submitted_at_unix"`
	Artifacts       map[string]v3Artifact `json:"artifacts"`
}

type v3Artifact struct {
	Encoding       string `json:"encoding"`
	SHA256         string `json:"sha256"` // of the plaintext
	CiphertextSize int64  `json:"ciphertext_size"`
	AESKey         string `json:"aes_key"` // base64, 32 bytes
}

// diskFree reports the bytes available to an unprivileged writer at path.
// A variable so tests can simulate a full disk.
var diskFree = func(path string) (uint64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	return st.Bavail * uint64(st.Bsize), nil
}

// ReceiveV3 reads a payload v3 body (boundary from its Content-Type). In order:
// the metadata is validated, free disk space is checked, the job slot is taken
// (a busy TEE answers 409 before any bulk bytes are read), then each artifact
// is decrypted into the job's received/ folder. On success the caller owns the
// slot and must hand the request to RunJob, which releases it; on any error
// the slot is released and nothing is left on disk.
func (m *Manager) ReceiveV3(body io.Reader, boundary string) (*Request, error) {
	mr := multipart.NewReader(body, boundary)
	meta, keys, err := readMetadata(mr)
	if err != nil {
		return nil, err
	}
	defer func() {
		for _, k := range keys {
			clear(k)
		}
	}()
	if err := m.checkDiskSpace(meta); err != nil {
		return nil, err
	}
	if ok, current := m.TryStartJob(meta.JobID); !ok {
		return nil, reject(http.StatusConflict, "Already running job %s", current)
	}

	req, err := m.receiveArtifacts(mr, meta, keys)
	if err != nil {
		os.RemoveAll(m.jobDir(meta.JobID))
		m.setState(func(s *State) {
			s.Status = "waiting_for_job"
			s.CurrentJobID = ""
			s.LastError = "intake failed: " + err.Error()
		})
		m.releaseJobSlot()
		return nil, err
	}
	return req, nil
}

// readMetadata reads and validates the first part, returning the decoded AES
// key per slot.
func readMetadata(mr *multipart.Reader) (*v3Metadata, map[string][]byte, error) {
	part, err := mr.NextRawPart()
	if err != nil {
		return nil, nil, reject(http.StatusBadRequest, "read metadata part: %v", err)
	}
	if part.FormName() != "metadata" {
		return nil, nil, reject(http.StatusBadRequest, "the first part must be \"metadata\", got %q", part.FormName())
	}
	raw, err := io.ReadAll(io.LimitReader(part, maxMetadataBytes+1))
	if err != nil {
		return nil, nil, reject(http.StatusBadRequest, "read metadata part: %v", err)
	}
	if len(raw) > maxMetadataBytes {
		return nil, nil, reject(http.StatusBadRequest, "metadata is larger than %d bytes", maxMetadataBytes)
	}
	var meta v3Metadata
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, nil, reject(http.StatusBadRequest, "metadata is not valid JSON: %v", err)
	}
	if meta.PayloadVersion != payloadVersion {
		return nil, nil, reject(http.StatusBadRequest, "payload_version %d is not supported (this Processing TEE "+
			"accepts %d); deploy the matching Buffer TEE", meta.PayloadVersion, payloadVersion)
	}
	if !validJobID(meta.JobID) {
		return nil, nil, reject(http.StatusBadRequest, "invalid job_id")
	}

	keys := map[string][]byte{}
	fail := func(status int, format string, a ...any) (*v3Metadata, map[string][]byte, error) {
		for _, k := range keys {
			clear(k)
		}
		return nil, nil, reject(status, format, a...)
	}
	for _, slot := range uploadSlots {
		if _, ok := meta.Artifacts[slot.Name]; slot.Required && !ok {
			return fail(http.StatusBadRequest, "metadata lists no %s artifact", slot.Name)
		}
	}
	for name, a := range meta.Artifacts {
		if !isUploadSlot(name) {
			return fail(http.StatusBadRequest, "metadata lists unknown artifact %q", name)
		}
		if a.Encoding != crypto.ChunkedEncoding {
			return fail(http.StatusBadRequest, "%s: encoding %q is not %s", name, a.Encoding, crypto.ChunkedEncoding)
		}
		a.SHA256 = strings.ToLower(strings.TrimSpace(a.SHA256))
		if b, err := hex.DecodeString(a.SHA256); err != nil || len(b) != 32 {
			return fail(http.StatusBadRequest, "%s: sha256 must be 64 hex characters", name)
		}
		if a.CiphertextSize <= 0 {
			return fail(http.StatusBadRequest, "%s: ciphertext_size %d must be positive", name, a.CiphertextSize)
		}
		key, err := base64.StdEncoding.DecodeString(a.AESKey)
		if err != nil || len(key) != 32 {
			clear(key)
			return fail(http.StatusBadRequest, "%s: aes_key must be 32 bytes of base64", name)
		}
		meta.Artifacts[name] = a
		keys[name] = key
	}
	return &meta, keys, nil
}

// checkDiskSpace refuses a payload whose artifacts would not fit on disk
// (plaintext is never larger than its ciphertext), with 10% to spare.
func (m *Manager) checkDiskSpace(meta *v3Metadata) error {
	var need uint64
	for _, a := range meta.Artifacts {
		need += uint64(a.CiphertextSize)
	}
	need += need / 10
	free, err := diskFree(m.cfg.BaseDir)
	if err != nil {
		log.Printf("pipeline: free disk space check skipped: %v", err)
		return nil
	}
	if free < need {
		return reject(http.StatusInsufficientStorage, "artifacts need %d bytes of disk, %d free", need, free)
	}
	return nil
}

// receiveArtifacts decrypts every remaining part into received/<slot>.
func (m *Manager) receiveArtifacts(mr *multipart.Reader, meta *v3Metadata, keys map[string][]byte) (*Request, error) {
	dir := m.receivedDir(meta.JobID)
	if err := os.RemoveAll(dir); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	req := &Request{
		JobID:           meta.JobID,
		DatasetID:       meta.DatasetID,
		ModelFormat:     meta.ModelFormat,
		KeycloakToken:   meta.KeycloakToken,
		BufferJobURL:    meta.BufferJobURL,
		SubmittedBy:     meta.SubmittedBy,
		SubmittedAtUnix: meta.SubmittedAtUnix,
		Artifacts:       map[string]Artifact{},
	}
	for {
		part, err := mr.NextRawPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, reject(http.StatusBadRequest, "read payload part: %v", err)
		}
		slot := part.FormName()
		a, ok := meta.Artifacts[slot]
		if !ok {
			return nil, reject(http.StatusBadRequest, "part %q is not listed in the metadata", slot)
		}
		if _, dup := req.Artifacts[slot]; dup {
			return nil, reject(http.StatusBadRequest, "part %q was sent twice", slot)
		}
		art, err := receivePart(part, filepath.Join(dir, slot), slot, a, keys[slot])
		if err != nil {
			return nil, err
		}
		req.Artifacts[slot] = art
	}
	for slot := range meta.Artifacts {
		if _, ok := req.Artifacts[slot]; !ok {
			return nil, reject(http.StatusBadRequest, "artifact %q is listed in the metadata but was not sent", slot)
		}
	}
	return req, nil
}

// receivePart decrypts one artifact stream to path and checks its size and
// SHA-256.
func receivePart(part io.Reader, path, slot string, a v3Artifact, key []byte) (Artifact, error) {
	started := time.Now()
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return Artifact{}, err
	}
	defer f.Close()

	// Never read more than the declared (and disk-checked) size. One byte
	// past it, so a longer part is caught as trailing data, not silently cut.
	ct := &countingReader{r: io.LimitReader(part, a.CiphertextSize+1)}
	out := &errWriter{w: bufio.NewWriterSize(f, 1<<20)}
	sum, n, err := crypto.DecryptChunkedStream(ct, key, out)
	switch {
	case out.err != nil:
		return Artifact{}, fmt.Errorf("pipeline: write %s: %w", slot, out.err)
	case err != nil:
		return Artifact{}, reject(http.StatusBadRequest, "%s: %v", slot, err)
	case ct.n != a.CiphertextSize:
		return Artifact{}, reject(http.StatusBadRequest, "%s: received %d bytes of ciphertext, metadata says %d",
			slot, ct.n, a.CiphertextSize)
	case hex.EncodeToString(sum[:]) != a.SHA256:
		return Artifact{}, reject(http.StatusBadRequest, "%s does not match its submit-time SHA-256", slot)
	}
	if err := out.w.Flush(); err != nil {
		return Artifact{}, fmt.Errorf("pipeline: write %s: %w", slot, err)
	}
	if err := f.Close(); err != nil {
		return Artifact{}, fmt.Errorf("pipeline: write %s: %w", slot, err)
	}
	elapsed := time.Since(started)
	log.Printf("pipeline: received %s: %d bytes in %s (%.1f MB/s)", slot, n, elapsed.Round(time.Millisecond),
		float64(n)/1e6/max(elapsed.Seconds(), 1e-3))
	return Artifact{Path: path, SHA256: a.SHA256, Size: n}, nil
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// errWriter remembers a write failure, so a full disk is told apart from a
// bad stream.
type errWriter struct {
	w   *bufio.Writer
	err error
}

func (e *errWriter) Write(p []byte) (int, error) {
	n, err := e.w.Write(p)
	if err != nil && e.err == nil {
		e.err = err
	}
	return n, err
}
