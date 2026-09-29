// Package crypto decrypts TANUH datasets: the tanuh-enc-dataset-v1 format the
// UI writes (see dataset_v1.go and ui-dx encrypted-dataset-upload.ts).
package crypto

import (
	"crypto/aes"
	"crypto/cipher"
	"fmt"
)

func newAESGCM(key []byte, nonceLen int) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("crypto: aes.NewCipher: %w", err)
	}
	if nonceLen == 12 {
		return cipher.NewGCM(block)
	}
	return cipher.NewGCMWithNonceSize(block, nonceLen)
}
