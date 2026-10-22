// chunked.go streams the AES-256-GCM-CHUNKED-v1 upload format: the framing the
// browser uses for model uploads to the Buffer TEE, which the Buffer stores and
// forwards unchanged (payload v3). The Buffer TEE has its own copy of this
// parser (internal/chunked); both are tested against testdata/chunked_v1.
//
//	[u32 BE header length][header JSON]
//	for each chunk i in 0..total_chunks-1:
//	  [u32 BE ciphertext length][ciphertext = plaintext + 16-byte GCM tag]
//
// Chunk i uses IV = base_iv + i (96-bit big-endian) and AAD
// "{i}:{total_chunks}:{file_id}". The chunk size is whatever the sender chose;
// only an upper bound (MaxChunkBytes) is enforced, as it bounds memory. The
// stream's total size is not limited here: callers bound it by what they read.
package crypto

import (
	"crypto/cipher"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// ChunkedEncoding names the format in payload v3 metadata.
const ChunkedEncoding = "AES-256-GCM-CHUNKED-v1"

const (
	// maxChunkedHeaderBytes bounds the header JSON, which is parsed whole.
	maxChunkedHeaderBytes = 8192
	// MaxChunkBytes bounds one chunk's ciphertext (64 MiB of plaintext + tag).
	// GCM authenticates a chunk only once all of it is read, so this is the
	// memory one chunk can take. It is a safety bound, not the sender's chunk
	// size: senders may use any chunk size up to it.
	MaxChunkBytes = 64<<20 + gcmTagSize
)

// ChunkedHeader is frame 0 of the stream. Enc and WrappedKey carry the AES key
// HPKE-wrapped to the Buffer TEE; the Processing TEE receives the key in the
// payload metadata instead and ignores them.
type ChunkedHeader struct {
	FileID          string `json:"file_id"`
	Enc             string `json:"enc"`
	WrappedKey      string `json:"wrapped_key"`
	BaseIV          string `json:"base_iv"`
	TotalChunks     int    `json:"total_chunks"`
	PlaintextSHA256 string `json:"plaintext_sha256"`
}

// DecryptChunkedStream decrypts a whole AES-256-GCM-CHUNKED-v1 stream from r
// into w, one chunk at a time: memory stays at about two chunks whatever the
// stream's size. It fails closed on any authentication failure and on bytes
// after the last chunk. It returns the plaintext's SHA-256 and length; the
// caller checks them against its commitment and must discard w's contents on
// error.
func DecryptChunkedStream(r io.Reader, key []byte, w io.Writer) (sum [32]byte, n int64, err error) {
	hdr, err := readChunkedHeader(r)
	if err != nil {
		return sum, 0, err
	}
	baseIV, err := base64.StdEncoding.DecodeString(hdr.BaseIV)
	if err != nil || len(baseIV) != 12 {
		return sum, 0, fmt.Errorf("crypto: chunked header base_iv must be 12 bytes of base64")
	}
	if len(key) != 32 {
		return sum, 0, fmt.Errorf("crypto: chunked stream key is %d bytes, want 32", len(key))
	}
	aead, err := newAESGCM(key, 12)
	if err != nil {
		return sum, 0, err
	}

	h := sha256.New()
	var ct, pt []byte
	for i := 0; i < hdr.TotalChunks; i++ {
		ct, err = readChunk(r, ct, i)
		if err != nil {
			return sum, n, err
		}
		pt, err = openChunk(aead, baseIV, hdr, i, ct, pt)
		if err != nil {
			return sum, n, err
		}
		h.Write(pt)
		if _, err := w.Write(pt); err != nil {
			return sum, n, err
		}
		n += int64(len(pt))
	}
	if err := expectEOF(r); err != nil {
		return sum, n, err
	}
	h.Sum(sum[:0])
	return sum, n, nil
}

// EncodeChunked encrypts plain into an AES-256-GCM-CHUNKED-v1 stream the way
// the browser does. The TEE never encrypts uploads; this is for tests and
// tools that need valid streams.
func EncodeChunked(key, baseIV []byte, fileID string, plain []byte, chunkSize int) ([]byte, error) {
	if chunkSize <= 0 || len(baseIV) != 12 {
		return nil, fmt.Errorf("crypto: chunk size must be positive and base IV 12 bytes")
	}
	aead, err := newAESGCM(key, 12)
	if err != nil {
		return nil, err
	}
	total := max(1, (len(plain)+chunkSize-1)/chunkSize)
	sum := sha256.Sum256(plain)
	hdr, err := json.Marshal(ChunkedHeader{
		FileID:          fileID,
		BaseIV:          base64.StdEncoding.EncodeToString(baseIV),
		TotalChunks:     total,
		PlaintextSHA256: fmt.Sprintf("%x", sum),
	})
	if err != nil {
		return nil, err
	}
	out := binary.BigEndian.AppendUint32(nil, uint32(len(hdr)))
	out = append(out, hdr...)
	for i := 0; i < total; i++ {
		end := min((i+1)*chunkSize, len(plain))
		ct := aead.Seal(nil, ivForChunk(baseIV, i), plain[i*chunkSize:end], fmt.Appendf(nil, "%d:%d:%s", i, total, fileID))
		out = binary.BigEndian.AppendUint32(out, uint32(len(ct)))
		out = append(out, ct...)
	}
	return out, nil
}

func readChunkedHeader(r io.Reader) (ChunkedHeader, error) {
	var hdr ChunkedHeader
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return hdr, fmt.Errorf("crypto: read chunked header length: %w", err)
	}
	hlen := binary.BigEndian.Uint32(lenBuf[:])
	if hlen == 0 || hlen > maxChunkedHeaderBytes {
		return hdr, fmt.Errorf("crypto: chunked header length %d out of range", hlen)
	}
	raw := make([]byte, hlen)
	if _, err := io.ReadFull(r, raw); err != nil {
		return hdr, fmt.Errorf("crypto: read chunked header: %w", err)
	}
	if err := json.Unmarshal(raw, &hdr); err != nil {
		return hdr, fmt.Errorf("crypto: parse chunked header: %w", err)
	}
	if hdr.TotalChunks <= 0 {
		return hdr, fmt.Errorf("crypto: chunked header total_chunks %d must be positive", hdr.TotalChunks)
	}
	if hdr.FileID == "" {
		return hdr, fmt.Errorf("crypto: chunked header has no file_id")
	}
	return hdr, nil
}

// readChunk reads chunk i's length prefix and ciphertext into buf, growing it
// only as far as the largest chunk seen.
func readChunk(r io.Reader, buf []byte, i int) ([]byte, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return buf, fmt.Errorf("crypto: read chunk %d length: %w", i, err)
	}
	clen := binary.BigEndian.Uint32(lenBuf[:])
	if clen < gcmTagSize || clen > MaxChunkBytes {
		return buf, fmt.Errorf("crypto: chunk %d length %d out of range (max %d)", i, clen, MaxChunkBytes)
	}
	if cap(buf) < int(clen) {
		buf = make([]byte, clen)
	}
	buf = buf[:clen]
	if _, err := io.ReadFull(r, buf); err != nil {
		return buf, fmt.Errorf("crypto: read chunk %d: %w", i, err)
	}
	return buf, nil
}

func openChunk(aead cipher.AEAD, baseIV []byte, hdr ChunkedHeader, i int, ct, dst []byte) ([]byte, error) {
	aad := fmt.Appendf(nil, "%d:%d:%s", i, hdr.TotalChunks, hdr.FileID)
	pt, err := aead.Open(dst[:0], ivForChunk(baseIV, i), ct, aad)
	if err != nil {
		return dst, fmt.Errorf("crypto: chunk %d/%d authentication failed (corrupt, reordered, or wrong key)",
			i+1, hdr.TotalChunks)
	}
	return pt, nil
}

func expectEOF(r io.Reader) error {
	var b [1]byte
	n, err := io.ReadFull(r, b[:])
	if n > 0 {
		return fmt.Errorf("crypto: unexpected bytes after the last chunk")
	}
	if !errors.Is(err, io.EOF) {
		return fmt.Errorf("crypto: read after the last chunk: %w", err)
	}
	return nil
}
