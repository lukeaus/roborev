package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// DroidAgent runs code reviews using Factory's Droid CLI
type DroidAgent struct {
	Command   string         // The droid command to run (default: "droid")
	Model     string         // Model ID passed via -m (empty = droid default)
	Reasoning ReasoningLevel // Reasoning level for the agent
	Agentic   bool           // Whether agentic mode is enabled (allow file edits)
}

// droidLogPath is droid's own diagnostic log. Droid writes the provider-side
// cause of a failed run there while stderr only reports "Exec failed", so the
// failure path reads the region this run appended. Tests override the variable.
var droidLogPath = defaultDroidLogPath()

// defaultDroidLogPath locates droid's log relative to the user's home
// directory. An empty result disables cause recovery rather than guessing.
func defaultDroidLogPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".factory", "logs", "droid-log-single.log")
}

// NewDroidAgent creates a new Droid agent with standard reasoning
func NewDroidAgent(command string) *DroidAgent {
	if command == "" {
		command = "droid"
	}
	return &DroidAgent{Command: command, Reasoning: ReasoningStandard}
}

func (a *DroidAgent) clone(opts ...agentCloneOption) *DroidAgent {
	cfg := newAgentCloneConfig(
		a.Command,
		a.Model,
		a.Reasoning,
		a.Agentic,
		"",
		opts...,
	)
	return &DroidAgent{
		Command:   cfg.Command,
		Model:     cfg.Model,
		Reasoning: cfg.Reasoning,
		Agentic:   cfg.Agentic,
	}
}

// WithReasoning returns a copy of the agent with the specified reasoning level
func (a *DroidAgent) WithReasoning(level ReasoningLevel) Agent {
	return a.clone(withClonedReasoning(level))
}

// WithAgentic returns a copy of the agent configured for agentic mode.
func (a *DroidAgent) WithAgentic(agentic bool) Agent {
	return a.clone(withClonedAgentic(agentic))
}

// WithModel returns a copy of the agent configured to use the given model.
// An empty model leaves the agent unchanged so a built-in default survives.
func (a *DroidAgent) WithModel(model string) Agent {
	if model == "" {
		return a
	}
	return a.clone(withClonedModel(model))
}

// droidReasoningEffort maps ReasoningLevel to droid-specific effort values
func (a *DroidAgent) droidReasoningEffort() string {
	switch a.Reasoning {
	case ReasoningMaximum, ReasoningThorough, ReasoningHigh:
		return "high"
	case ReasoningMedium:
		return "medium"
	case ReasoningFast, ReasoningLow:
		return "low"
	default:
		return "" // use droid default
	}
}

func (a *DroidAgent) Name() string {
	return "droid"
}

func (a *DroidAgent) CommandName() string {
	return a.Command
}

func (a *DroidAgent) CommandLine() string {
	agenticMode := a.Agentic || AllowUnsafeAgents()
	args := a.buildArgs(agenticMode)
	return a.Command + " " + strings.Join(args, " ")
}

func (a *DroidAgent) buildArgs(agenticMode bool) []string {
	args := []string{"exec", "--tag", "roborev"}

	if a.Model != "" {
		args = append(args, "-m", a.Model)
	}

	// Set autonomy level based on agentic mode
	if agenticMode {
		args = append(args, "--auto", "medium")
	} else {
		args = append(args, "--auto", "low")
	}

	// Set reasoning effort if specified
	if effort := a.droidReasoningEffort(); effort != "" {
		args = append(args, "--reasoning-effort", effort)
	}

	return args
}

func (a *DroidAgent) Review(ctx context.Context, repoPath, commitSHA, prompt string, output io.Writer) (string, error) {
	// Use agentic mode if either per-job setting or global setting enables it
	agenticMode := a.Agentic || AllowUnsafeAgents()

	args := a.buildArgs(agenticMode)

	cmd := exec.CommandContext(ctx, a.Command, args...)
	cmd.Dir = repoPath
	cmd.Stdin = strings.NewReader(prompt)
	tracker := configureSubprocess(ctx, cmd)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	if sw := newSyncWriter(output); sw != nil {
		cmd.Stderr = io.MultiWriter(&stderr, sw)
	} else {
		cmd.Stderr = &stderr
	}

	// Record how much of droid's log already existed so the failure path can
	// read only what this run appended.
	logPath := droidLogPath
	logOffset := droidLogSize(logPath)

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("droid failed: %w", err)
	}
	pid := 0
	if cmd.Process != nil {
		pid = cmd.Process.Pid
	}

	if err := cmd.Wait(); err != nil {
		if ctxErr := contextProcessError(ctx, tracker, err, nil); ctxErr != nil {
			return "", ctxErr
		}
		return "", droidExecError(err, pid, stderr.String(), logPath, logOffset)
	}

	result := stdout.String()
	if len(result) == 0 {
		return "No review output generated", nil
	}

	return result, nil
}

// droidExecError builds the error for a non-zero droid exit.
//
// Real droid prints only "Error during droid execution: Exec failed" on stderr
// and writes the provider-side cause (a 403 allowance message, an auth failure,
// a connection error) to its own log. The cause is recovered from there and put
// in the returned error so `roborev show` and limit classification see why the
// run failed instead of the opaque wrapper.
//
// Droid's stdout is the review text, not a diagnostic, and is deliberately left
// out. Folding it in would bloat the stored job error with up to a full review
// and would let review content that merely mentions quota wording trip the
// limit classifier into a false cooldown.
func droidExecError(runErr error, pid int, stderr, logPath string, logOffset int64) error {
	detail := strings.TrimSpace(stderr)
	if cause := droidLogCause(logPath, logOffset, pid); cause != "" {
		if detail == "" {
			detail = "cause: " + cause
		} else {
			detail += "\ncause: " + cause
		}
	}

	var err error
	if detail == "" {
		err = fmt.Errorf("droid failed: %w", runErr)
	} else {
		err = fmt.Errorf("droid failed: %w\n%s", runErr, detail)
	}
	// Classify from stderr and the recovered provider cause only, never from
	// review text on stdout.
	return WithLimitClassification(err, ClassifyLimit("droid", detail))
}

// droidLogSize returns the current length of droid's log, or 0 when it is
// missing or unreachable.
func droidLogSize(path string) int64 {
	if path == "" {
		return 0
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}

// droidLogCause recovers the provider-side message droid wrote for the process
// that just exited. Only the region appended since the run started is read, and
// only lines tagged with that process's PID are considered, so other droid
// sessions on the machine are neither inspected nor reported. An empty string
// means nothing useful was found and the caller falls back to stderr alone.
func droidLogCause(logPath string, offset int64, pid int) string {
	if logPath == "" || pid <= 0 {
		return ""
	}
	f, err := os.Open(logPath)
	if err != nil {
		return ""
	}
	defer f.Close()

	if offset > 0 {
		if _, err := f.Seek(offset, io.SeekStart); err != nil {
			return ""
		}
	}

	// Droid log lines embed full JSON contexts, including stack traces, and can
	// far exceed bufio.Scanner's default token limit. ReadString has no such
	// cap, so no line is silently dropped.
	reader := bufio.NewReader(f)
	cause := ""
	for {
		line, readErr := reader.ReadString('\n')
		if line != "" && droidLogLinePID(line) == pid {
			if msg := droidInformativeCause(droidLogLineCause(line)); msg != "" {
				// Later lines describe the terminal failure, so keep looking.
				cause = msg
			}
		}
		if readErr != nil {
			return cause
		}
	}
}

// droidLogLinePID returns the process ID a droid log line is tagged with, or 0
// when the line carries none.
func droidLogLinePID(line string) int {
	const key = `"process.pid":`
	_, rest, ok := strings.Cut(line, key)
	if !ok {
		return 0
	}
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0
	}
	pid, err := strconv.Atoi(rest[:end])
	if err != nil {
		return 0
	}
	return pid
}

// droidLogLineCause pulls the provider-side message out of one droid log line's
// JSON context. Shapes observed from droid 0.228.0 on real failures:
//
//	"error":{"name":"Error","message":"401 User not found."}
//	"errorMessage":"401 User not found."
//	"cause":{"name":"TypeError","message":"Unable to connect. ..."}
//	"error":"Error: 401 User not found."
func droidLogLineCause(line string) string {
	const marker = "| Context: "
	_, ctxJSON, ok := strings.Cut(line, marker)
	if !ok {
		return ""
	}
	var ctx map[string]json.RawMessage
	if err := json.Unmarshal([]byte(ctxJSON), &ctx); err != nil {
		return ""
	}
	for _, key := range []string{"errorMessage", "error", "cause"} {
		raw, ok := ctx[key]
		if !ok {
			continue
		}
		if msg := droidJSONMessage(raw); msg != "" {
			return msg
		}
	}
	return ""
}

// droidJSONMessage reads a log context value that is either a bare string or an
// object carrying a "message" field.
func droidJSONMessage(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var obj struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil {
		return obj.Message
	}
	return ""
}

// droidInformativeCause drops the wrapper droid stamps on every failure so the
// recovered text is the provider's own reason rather than a repeat of what
// stderr already reports.
func droidInformativeCause(msg string) string {
	msg = strings.TrimSpace(msg)
	if msg == "" {
		return ""
	}
	lower := strings.ToLower(msg)
	if lower == "exec failed" || strings.HasPrefix(lower, "metaerror:") {
		return ""
	}
	return msg
}

func init() {
	Register(NewDroidAgent(""))
}
