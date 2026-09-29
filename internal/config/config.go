// Package config centralises all runtime configuration for the Processing
// TEE. Everything is parsed once in main from the environment (the Dockerfile
// is the single source of default values for deployment).
package config

import (
	"os"
	"strconv"
	"time"
)

// Config is the full runtime configuration.
type Config struct {
	BaseDir       string
	ListenAddr    string
	RATLSAudience string

	ProjectID        string
	DatasetsBucket   string // GCS bucket holding per-UUID dataset folders
	CatalogueBaseURL string // base URL for the catalogue (cat/item) + leaderboard host

	// Self-deallocation target (this VM). INSTANCE must be overridden per VM
	// (gpu-cs-tdx-h100 / cpu-cs-tdx) via tee-env metadata.
	StopProject  string
	StopZone     string
	StopInstance string

	IdleTimeout      time.Duration
	DeallocAfterJob  bool
	EvalTimeout      time.Duration // 0 disables the timeout
	DepsTimeout      time.Duration // pre-eval uv install cap; 0 disables
	LeaderboardURL   string
	PolicyPath       string
	AttestAudience   string // audience for leaderboard attestation claims
	CallbackAudience string // audience for the buffer completion-callback token
	CallbackTimeout  time.Duration
}

// FromEnv builds the Config from the environment.
func FromEnv() Config {
	base := getEnv("BASE_DIR", "/app")
	return Config{
		BaseDir:       base,
		ListenAddr:    getEnv("LISTEN_ADDR", ":443"),
		RATLSAudience: getEnv("RATLS_AUDIENCE", "ratls-buffer-tee"),

		ProjectID:        getEnv("GCP_PROJECT_ID", "proj-tanuh-benchmark-ptfm"),
		DatasetsBucket:   getEnv("DATASETS_BUCKET", "file-server-data"),
		CatalogueBaseURL: getEnv("CATALOGUE_BASE_URL", "https://benchmark.tanuh.ai"),

		StopProject:  getEnv("PROJECT", "p3dx-depa-sandbox"),
		StopZone:     getEnv("ZONE", "us-central1-a"),
		StopInstance: getEnv("INSTANCE", "gpu-cs-tdx-h100"),

		IdleTimeout:     secondsEnv("PROCESSING_IDLE_TIMEOUT_SECONDS", 300),
		DeallocAfterJob: getEnv("PROCESSING_DEALLOCATE_AFTER_JOB", "1") == "1",
		EvalTimeout:     secondsEnv("PROCESSING_EVAL_TIMEOUT_SECONDS", 3600),
		DepsTimeout:     secondsEnv("PROCESSING_DEPS_TIMEOUT_SECONDS", 600),
		LeaderboardURL: getEnv("LEADERBOARD_SUBMIT_URL",
			"https://benchmark.tanuh.ai/leaderboard/submit-solution"),
		PolicyPath:       getEnv("NETWORK_POLICY_PATH", base+"/policy/network_policy.json"),
		AttestAudience:   "https://tanuh-processing-tee",
		CallbackAudience: getEnv("CALLBACK_AUDIENCE", "tanuh-buffer-callback"),
		CallbackTimeout:  10 * time.Second,
	}
}

// WorkflowDir returns BASE_DIR/cvm_workflow.
func (c Config) WorkflowDir() string { return c.BaseDir + "/cvm_workflow" }

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func secondsEnv(key string, def int) time.Duration {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return time.Duration(n) * time.Second
		}
	}
	return time.Duration(def) * time.Second
}
