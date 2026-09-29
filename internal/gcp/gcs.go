package gcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// ErrObjectNotFound is returned (wrapped) when a GCS object does not exist.
var ErrObjectNotFound = errors.New("gcs object not found")

// apiClient is shared by the GCS and KMS calls so parallel dataset downloads
// reuse keep-alive connections instead of opening a TLS session per object
// (the default transport keeps only 2 idle connections per host). Timeouts
// are applied per call through the request context.
var apiClient = &http.Client{Transport: &http.Transport{
	Proxy:               http.ProxyFromEnvironment,
	ForceAttemptHTTP2:   true,
	MaxIdleConns:        64,
	MaxIdleConnsPerHost: 32,
	IdleConnTimeout:     90 * time.Second,
	TLSHandshakeTimeout: 15 * time.Second,
}}

// ListObjects returns the names of all objects under prefix in bucket
// (following pagination), using the attached service-account token.
func ListObjects(ctx context.Context, bucket, prefix string) ([]string, error) {
	token, err := AccessToken(ctx)
	if err != nil {
		return nil, err
	}
	var names []string
	pageToken := ""
	for {
		u := fmt.Sprintf("https://storage.googleapis.com/storage/v1/b/%s/o?prefix=%s",
			url.PathEscape(bucket), url.QueryEscape(prefix))
		if pageToken != "" {
			u += "&pageToken=" + url.QueryEscape(pageToken)
		}
		raw, status, err := getWithTimeout(ctx, u, token, 30*time.Second, 1<<20)
		if err != nil {
			return nil, fmt.Errorf("gcp: GCS list gs://%s/%s: %w", bucket, prefix, err)
		}
		if status != http.StatusOK {
			return nil, fmt.Errorf("gcp: GCS list gs://%s/%s: status %d: %s", bucket, prefix, status, string(raw))
		}
		var page struct {
			Items []struct {
				Name string `json:"name"`
			} `json:"items"`
			NextPageToken string `json:"nextPageToken"`
		}
		if err := json.Unmarshal(raw, &page); err != nil {
			return nil, fmt.Errorf("gcp: parse GCS list: %w", err)
		}
		for _, it := range page.Items {
			names = append(names, it.Name)
		}
		if page.NextPageToken == "" {
			return names, nil
		}
		pageToken = page.NextPageToken
	}
}

// DownloadObjectBytes downloads a GCS object fully into memory using the
// attached service-account token. Suitable for manifests and dataset objects
// that fit in RAM.
func DownloadObjectBytes(ctx context.Context, bucket, object string) ([]byte, error) {
	data, _, err := download(ctx, bucket, object, 0)
	return data, err
}

// DownloadObjectWithGeneration downloads an object of at most maxBytes and
// returns its GCS generation (the version number GCS assigns on every write),
// so a job can record exactly which version of a file it used.
func DownloadObjectWithGeneration(ctx context.Context, bucket, object string, maxBytes int64) ([]byte, string, error) {
	data, hdr, err := download(ctx, bucket, object, maxBytes)
	if err != nil {
		return nil, "", err
	}
	return data, hdr.Get("X-Goog-Generation"), nil
}

// download fetches an object; maxBytes > 0 caps its size.
func download(ctx context.Context, bucket, object string, maxBytes int64) ([]byte, http.Header, error) {
	token, err := AccessToken(ctx)
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	u := fmt.Sprintf("https://storage.googleapis.com/download/storage/v1/b/%s/o/%s?alt=media",
		url.PathEscape(bucket), url.QueryEscape(object))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := apiClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("gcp: GCS download gs://%s/%s: %w", bucket, object, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, nil, fmt.Errorf("gcp: gs://%s/%s: %w", bucket, object, ErrObjectNotFound)
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, nil, fmt.Errorf("gcp: GCS download gs://%s/%s: status %d: %s",
			bucket, object, resp.StatusCode, string(raw))
	}
	body := io.Reader(resp.Body)
	if maxBytes > 0 {
		body = io.LimitReader(resp.Body, maxBytes+1)
	}
	data, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, fmt.Errorf("gcp: GCS download gs://%s/%s: %w", bucket, object, err)
	}
	if maxBytes > 0 && int64(len(data)) > maxBytes {
		return nil, nil, fmt.Errorf("gcp: gs://%s/%s is larger than %d bytes", bucket, object, maxBytes)
	}
	return data, resp.Header, nil
}

// getWithTimeout GETs u with a bearer token through the shared client and
// returns up to limit bytes of the body plus the status code.
func getWithTimeout(ctx context.Context, u, token string, timeout time.Duration, limit int64) ([]byte, int, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := apiClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, limit))
	return raw, resp.StatusCode, err
}
