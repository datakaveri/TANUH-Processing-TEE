package pipeline

import "path/filepath"

// payloadVersion is the Buffer → Processing TEE payload layout this build
// accepts (Buffer: scheduler.PayloadVersion): a multipart stream of metadata
// then each artifact's ciphertext, decrypted to disk at intake (intake.go).
// Both TEEs ship together.
const payloadVersion = 3

// Upload slots, in the order the Buffer sends them (the Buffer's
// jobs.artifactSpecs). File sizes are not capped here: intake streams to disk
// and checks free space first; the UI caps what users can upload.
var uploadSlots = []struct {
	Name     string
	Required bool
}{
	{"model", true},
	{"weights", false},
	{"adaptor", true},
	{"input_spec", false},
}

func isUploadSlot(name string) bool {
	for _, s := range uploadSlots {
		if s.Name == name {
			return true
		}
	}
	return false
}

// Artifact is one upload on disk: plaintext, already checked against the
// SHA-256 the browser committed to at submit.
type Artifact struct {
	Path   string
	SHA256 string
	Size   int64
}

// Request is an accepted job as RunJob consumes it: the metadata from the
// payload and the decrypted, hash-checked uploads by slot.
type Request struct {
	JobID           string
	DatasetID       string
	ModelFormat     string
	KeycloakToken   string
	BufferJobURL    string
	SubmittedBy     string
	SubmittedAtUnix int64
	Artifacts       map[string]Artifact
}

// validJobID reports whether id is usable as a directory name.
func validJobID(id string) bool {
	return id != "" && id != "." && id != ".." && id == filepath.Base(id)
}

// path returns a slot's file, or "" when it was not uploaded.
func (r *Request) path(slot string) string {
	return r.Artifacts[slot].Path
}

// receipt is what the TEE keeps on disk about an accepted request: no token,
// no keys and no file contents.
func (r *Request) receipt() map[string]any {
	artifacts := map[string]any{}
	for slot, a := range r.Artifacts {
		artifacts[slot] = map[string]any{"sha256": a.SHA256, "size": a.Size}
	}
	return map[string]any{
		"payload_version":   payloadVersion,
		"job_id":            r.JobID,
		"dataset_id":        r.DatasetID,
		"model_format":      r.ModelFormat,
		"submitted_by":      r.SubmittedBy,
		"submitted_at_unix": r.SubmittedAtUnix,
		"buffer_job_url":    r.BufferJobURL,
		"artifacts":         artifacts,
	}
}
