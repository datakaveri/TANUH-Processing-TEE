// Package eval runs the Python stages of a job as subprocesses:
//
//	infer      platform inference (/app/infer.py): model + data -> raw outputs
//	adaptor    the model provider's adaptor.py: raw outputs -> predictions.csv
//	evaluator  the bucket's evaluate.py: predictions + ground truth -> results.json
//
// A failure is reported as a *StageError naming the stage, its exit code and
// whether it timed out; leaderboard.Classify turns that into the leaderboard's
// error categories. Stage output streams to the TEE log.
package eval

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"
)

// Stage names.
const (
	StageInfer     = "infer"
	StageAdaptor   = "adaptor"
	StageEvaluator = "evaluator"
)

// StageError reports a stage that exited non-zero or timed out.
type StageError struct {
	Stage    string
	ExitCode int
	TimedOut bool
}

func (e *StageError) Error() string {
	if e.TimedOut {
		return fmt.Sprintf("%s stage timed out", e.Stage)
	}
	return fmt.Sprintf("%s stage failed with exit code %d", e.Stage, e.ExitCode)
}

// AdaptorOutputError means the adaptor exited cleanly but did not produce a
// usable predictions.csv (missing, empty or too large).
type AdaptorOutputError struct{ Msg string }

func (e *AdaptorOutputError) Error() string { return "adaptor output: " + e.Msg }

// Stage describes one Python subprocess.
type Stage struct {
	Name    string
	Script  string
	Args    []string
	Dir     string        // working directory
	Env     []string      // added to the TEE's environment
	Timeout time.Duration // 0 disables the deadline
}

// RunStage runs `python3 <Script> <Args...>` and waits for it.
func RunStage(ctx context.Context, st Stage) error {
	if st.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, st.Timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(ctx, "python3", append([]string{st.Script}, st.Args...)...)
	cmd.Dir = st.Dir
	cmd.Env = append(os.Environ(), st.Env...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	started := time.Now()
	log.Printf("eval: %s stage starting", st.Name)
	err := cmd.Run()
	if err == nil {
		log.Printf("eval: %s stage done in %s", st.Name, time.Since(started).Round(time.Millisecond))
		return nil
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return &StageError{Stage: st.Name, TimedOut: true}
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return &StageError{Stage: st.Name, ExitCode: exitErr.ExitCode()}
	}
	return fmt.Errorf("eval: start %s stage: %w", st.Name, err)
}
