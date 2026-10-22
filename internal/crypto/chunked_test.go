package crypto

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
)

// goldenVector is testdata/chunked_v1/vector.json, the same file the Buffer
// TEE's parser is tested against.
type goldenVector struct {
	Key       []byte
	Stream    []byte
	Plaintext []byte
	SHA256    string
}

func loadGolden(t *testing.T) goldenVector {
	t.Helper()
	raw, err := os.ReadFile("testdata/chunked_v1/vector.json")
	if err != nil {
		t.Fatal(err)
	}
	var v struct {
		KeyB64       string `json:"key_b64"`
		StreamB64    string `json:"stream_b64"`
		PlaintextB64 string `json:"plaintext_b64"`
		SHA256       string `json:"plaintext_sha256"`
	}
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	dec := func(s string) []byte {
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	return goldenVector{dec(v.KeyB64), dec(v.StreamB64), dec(v.PlaintextB64), v.SHA256}
}

// chunkedStream builds an AES-256-GCM-CHUNKED-v1 stream the way the browser
// does, returning the framed header and each framed chunk separately so tests
// can reorder or drop them.
func chunkedStream(t *testing.T, key, baseIV []byte, fileID string, plain []byte, chunkSize int) ([]byte, [][]byte) {
	t.Helper()
	total := max(1, (len(plain)+chunkSize-1)/chunkSize)
	hdr, _ := json.Marshal(ChunkedHeader{FileID: fileID, BaseIV: base64.StdEncoding.EncodeToString(baseIV), TotalChunks: total})
	aead, err := newAESGCM(key, 12)
	if err != nil {
		t.Fatal(err)
	}
	frames := make([][]byte, total)
	for i := range frames {
		end := min((i+1)*chunkSize, len(plain))
		ct := aead.Seal(nil, ivForChunk(baseIV, i), plain[i*chunkSize:end], fmt.Appendf(nil, "%d:%d:%s", i, total, fileID))
		frames[i] = append(binary.BigEndian.AppendUint32(nil, uint32(len(ct))), ct...)
	}
	return append(binary.BigEndian.AppendUint32(nil, uint32(len(hdr))), hdr...), frames
}

func join(hdr []byte, frames ...[]byte) []byte {
	return bytes.Join(append([][]byte{hdr}, frames...), nil)
}

func TestDecryptChunkedStreamGolden(t *testing.T) {
	g := loadGolden(t)
	var out bytes.Buffer
	sum, n, err := DecryptChunkedStream(bytes.NewReader(g.Stream), g.Key, &out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), g.Plaintext) || n != int64(len(g.Plaintext)) || hex.EncodeToString(sum[:]) != g.SHA256 {
		t.Fatalf("got %q (%d bytes, sha %x)", out.Bytes(), n, sum)
	}
}

func TestDecryptChunkedStreamRejects(t *testing.T) {
	g := loadGolden(t)
	key := bytes.Repeat([]byte{7}, 32)
	iv := bytes.Repeat([]byte{0xff}, 12) // every chunk after the first carries
	plain := bytes.Repeat([]byte("0123456789abcdef"), 4)
	hdr, frames := chunkedStream(t, key, iv, "f", plain, 16)
	good := join(hdr, frames...)

	flipped := bytes.Clone(good)
	flipped[len(hdr)+4] ^= 1
	wrongKey := bytes.Clone(key)
	wrongKey[0] ^= 1
	moreChunks := func() []byte {
		h, _ := json.Marshal(ChunkedHeader{FileID: "f", BaseIV: base64.StdEncoding.EncodeToString(iv), TotalChunks: 5})
		return join(append(binary.BigEndian.AppendUint32(nil, uint32(len(h))), h...), frames...)
	}()
	oversize := join(hdr, binary.BigEndian.AppendUint32(nil, MaxChunkBytes+1))

	zeroChunks := func() []byte {
		h, _ := json.Marshal(ChunkedHeader{FileID: "f", BaseIV: base64.StdEncoding.EncodeToString(iv), TotalChunks: 0})
		return append(binary.BigEndian.AppendUint32(nil, uint32(len(h))), h...)
	}()

	cases := map[string]struct {
		stream []byte
		key    []byte
	}{
		"wrong key":               {good, wrongKey},
		"flipped byte":            {flipped, key},
		"swapped chunks":          {join(hdr, frames[1], frames[0], frames[2], frames[3]), key},
		"dropped last chunk":      {join(hdr, frames[:3]...), key},
		"truncated mid-chunk":     {good[:len(good)-5], key},
		"trailing bytes":          {append(bytes.Clone(good), 0), key},
		"header claims 5 chunks":  {moreChunks, key},
		"header claims 0 chunks":  {zeroChunks, key},
		"chunk over 64 MiB + tag": {oversize, key},
		"short key":               {good, key[:16]},
		"empty stream":            {nil, key},
		"golden with other key":   {g.Stream, key},
	}
	for name, c := range cases {
		if _, _, err := DecryptChunkedStream(bytes.NewReader(c.stream), c.key, io.Discard); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestEncodeChunkedMatchesGolden(t *testing.T) {
	g := loadGolden(t)
	// The golden header has empty enc/wrapped_key; rebuild it with the same
	// base IV and file id and compare the chunk frames byte for byte.
	baseIV := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0xff, 0xfe}
	stream, err := EncodeChunked(g.Key, baseIV, "golden-file", g.Plaintext, 16)
	if err != nil {
		t.Fatal(err)
	}
	hdrLen := func(s []byte) int { return 4 + int(binary.BigEndian.Uint32(s)) }
	if !bytes.Equal(stream[hdrLen(stream):], g.Stream[hdrLen(g.Stream):]) {
		t.Fatal("EncodeChunked's chunks differ from the golden vector's")
	}
	var out bytes.Buffer
	if _, _, err := DecryptChunkedStream(bytes.NewReader(stream), g.Key, &out); err != nil || !bytes.Equal(out.Bytes(), g.Plaintext) {
		t.Fatalf("round trip: %v", err)
	}
}

// A sender may use chunks of up to 64 MiB of plaintext (the UI uses 16 MiB).
func TestDecryptChunkedStreamMaxChunk(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 64 MiB")
	}
	key := bytes.Repeat([]byte{9}, 32)
	plain := make([]byte, 64<<20)
	hdr, frames := chunkedStream(t, key, make([]byte, 12), "big", plain, len(plain))
	sum, n, err := DecryptChunkedStream(bytes.NewReader(join(hdr, frames...)), key, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if want := sha256.Sum256(plain); sum != want || n != int64(len(plain)) {
		t.Fatalf("n=%d sum mismatch", n)
	}
}
