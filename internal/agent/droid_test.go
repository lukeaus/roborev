package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDroidBuildArgs(t *testing.T) {
	tests := []struct {
		name string
		// agentic is the agentic-mode input to buildArgs.
		agentic bool
		setup   func(*DroidAgent) *DroidAgent
		// wantArgs must each appear somewhere in argv.
		wantArgs []string
		// wantJoined must appear in the space-joined argv, which is how a
		// flag and its value are proven to arrive as a consecutive pair.
		wantJoined []string
		dontWant   []string
	}{
		{
			name:     "Non-agentic default",
			agentic:  false,
			wantArgs: []string{"exec", "--tag", "roborev", "--auto", "low"},
			dontWant: []string{"medium"},
		},
		{
			name:  "Model set",
			setup: func(a *DroidAgent) *DroidAgent { return a.WithModel("custom:foo").(*DroidAgent) },
			// Acceptance is `droid exec ... -m custom:foo`, so assert the
			// pair is consecutive; membership alone would still pass if
			// buildArgs emitted `-m --auto low custom:foo`.
			wantJoined: []string{"-m custom:foo"},
		},
		{
			name:     "No model",
			dontWant: []string{"-m"},
		},
		{
			name:     "Agentic mode",
			agentic:  true,
			wantArgs: []string{"--auto", "medium"},
		},
		{
			name:     "Reasoning Thorough",
			setup:    func(a *DroidAgent) *DroidAgent { return a.WithReasoning(ReasoningThorough).(*DroidAgent) },
			wantArgs: []string{"--reasoning-effort", "high"},
		},
		{
			name:     "Reasoning Fast",
			setup:    func(a *DroidAgent) *DroidAgent { return a.WithReasoning(ReasoningFast).(*DroidAgent) },
			wantArgs: []string{"--reasoning-effort", "low"},
		},
		{
			name:     "Reasoning Standard",
			setup:    func(a *DroidAgent) *DroidAgent { return a.WithReasoning(ReasoningStandard).(*DroidAgent) },
			dontWant: []string{"--reasoning-effort"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert := assert.New(t)
			a := NewDroidAgent("droid")
			if tt.setup != nil {
				a = tt.setup(a)
			}

			args := a.buildArgs(tt.agentic)
			joined := strings.Join(args, " ")

			for _, want := range tt.wantArgs {
				assertContainsArg(t, args, want)
			}
			for _, want := range tt.wantJoined {
				assert.Contains(joined, want, "args: %s", joined)
			}
			for _, dont := range tt.dontWant {
				assertNotContainsArg(t, args, dont)
			}
		})
	}
}

func TestDroidCommandLineUsesBuildArgs(t *testing.T) {
	a := NewDroidAgent("droid").WithReasoning(ReasoningThorough).WithAgentic(true).(*DroidAgent)

	want := strings.Join(a.buildArgs(true), " ")
	assert.Equal(t, "droid "+want, a.CommandLine())
}

func TestDroidName(t *testing.T) {
	a := NewDroidAgent("")
	require.Equal(t, "droid", a.Name(), "expected name 'droid', got %s", a.Name())
	require.Equal(t, "droid", a.CommandName(), "expected command name 'droid', got %s", a.CommandName())
}

func TestDroidWithAgentic(t *testing.T) {
	a := NewDroidAgent("droid")
	require.False(t, a.Agentic, "expected non-agentic by default")

	a2 := a.WithAgentic(true).(*DroidAgent)
	require.True(t, a2.Agentic, "expected agentic after WithAgentic(true)")
	require.False(t, a.Agentic, "original should be unchanged")
}

func TestDroidReviewOutcomes(t *testing.T) {
	tests := []struct {
		name        string
		mockOpts    MockCLIOpts
		wantError   bool
		errContains string
		wantResult  string
		exactMatch  bool
	}{
		{
			name:       "Success",
			mockOpts:   MockCLIOpts{StdoutLines: []string{"Review feedback from Droid"}},
			wantResult: "Review feedback from Droid\n",
			exactMatch: true,
		},
		{
			name:        "Failure",
			mockOpts:    MockCLIOpts{StderrLines: []string{"error: something went wrong"}, ExitCode: 1},
			wantError:   true,
			errContains: "droid failed",
		},
		{
			name:       "Empty Output",
			mockOpts:   MockCLIOpts{},
			wantResult: "No review output generated",
			exactMatch: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := mockAgentCLI(t, tt.mockOpts)
			a := NewDroidAgent(mock.CmdPath)

			result, err := a.Review(context.Background(), t.TempDir(), "HEAD", "review this commit", nil)

			if tt.wantError {
				require.Error(t, err)

				if !strings.Contains(err.Error(), tt.errContains) {
					require.ErrorContains(t, err, tt.errContains, "expected error to contain %q, got %v", tt.errContains, err)
				}
				return
			}
			require.NoError(t, err)

			if tt.exactMatch {
				require.Equal(t, tt.wantResult, result, "expected exact result %q, got %q", tt.wantResult, result)
			} else if !strings.Contains(result, tt.wantResult) {
				require.Contains(t, result, tt.wantResult, "expected result to contain %q, got %q", tt.wantResult, result)
			}
		})
	}
}

func TestDroidReviewWithProgress(t *testing.T) {
	skipIfWindows(t)
	tmpDir := t.TempDir()
	progressFile := filepath.Join(tmpDir, "progress.txt")

	mock := mockAgentCLI(t, MockCLIOpts{
		StderrLines: []string{"Processing..."},
		StdoutLines: []string{"Done"},
	})
	a := NewDroidAgent(mock.CmdPath)

	f, err := os.Create(progressFile)
	require.NoError(t, err, "create progress file: %v")

	defer f.Close()

	_, err = a.Review(context.Background(), tmpDir, "deadbeef", "review this commit", f)
	require.NoError(t, err)

	progress, _ := os.ReadFile(progressFile)
	if !strings.Contains(string(progress), "Processing") {
		require.Contains(t, string(progress), "Processing", "expected progress output, got %q", string(progress))
	}
}

func TestDroidReviewPipesPromptViaStdin(t *testing.T) {
	skipIfWindows(t)

	mock := mockAgentCLI(t, MockCLIOpts{
		CaptureArgs:  true,
		CaptureStdin: true,
		StdoutLines:  []string{"ok"},
	})

	a := NewDroidAgent(mock.CmdPath)
	prompt := "Review this commit carefully"
	_, err := a.Review(
		context.Background(), t.TempDir(), "HEAD", prompt, nil,
	)
	require.NoError(t, err, "Review failed: %v")

	assertFileContent(t, mock.StdinFile, prompt)

	assertFileNotContains(t, mock.ArgsFile, prompt)
}

func TestDroidReviewAgenticModeFromGlobal(t *testing.T) {
	withUnsafeAgents(t, true)

	mock := mockAgentCLI(t, MockCLIOpts{
		CaptureArgs: true,
		StdoutLines: []string{"result"},
	})

	a := NewDroidAgent(mock.CmdPath)
	if _, err := a.Review(context.Background(), t.TempDir(), "deadbeef", "prompt", nil); err != nil {
		require.NoError(t, err)
	}

	args, err := os.ReadFile(mock.ArgsFile)
	require.NoError(t, err, "read args: %v")

	if !strings.Contains(string(args), "medium") {
		require.Contains(t, strings.TrimSpace(string(args)), "--auto", "expected '--auto medium' in args when global unsafe enabled, got %s", strings.TrimSpace(string(args)))
	}
}

func TestDroidWithModel(t *testing.T) {
	assert := assert.New(t)
	a := NewDroidAgent("droid")

	a2 := a.WithModel("custom:foo").(*DroidAgent)
	assert.Equal("custom:foo", a2.Model)
	assert.Empty(a.Model, "original should be unchanged")

	a3 := a2.WithReasoning(ReasoningThorough).(*DroidAgent)
	assert.Equal("custom:foo", a3.Model, "model should survive other clones")
	assert.Contains(a3.CommandLine(), "-m custom:foo")
}

func TestDroidWithModelEmptyKeepsExisting(t *testing.T) {
	assert := assert.New(t)
	a := NewDroidAgent("droid").WithModel("custom:foo").(*DroidAgent)

	// The Agent contract: an empty model returns the agent unchanged so a
	// configured model is not wiped by an empty job/model override.
	assert.Same(a, a.WithModel(""))

	bare := NewDroidAgent("droid")
	assert.Same(bare, bare.WithModel(""))
}

// withDroidLogPath points droid's diagnostic log at path for one test.
func withDroidLogPath(t *testing.T, path string) {
	t.Helper()
	prev := droidLogPath
	droidLogPath = path
	t.Cleanup(func() { droidLogPath = prev })
}

// writeDroidFailureScript writes a fake droid that records a log line tagged
// with its own PID ($$, which is the same process exec.Command exposes as
// cmd.Process.Pid) and then fails the way real droid does: the provider cause
// on the log, "Exec failed" on stderr. logLine must contain one %s for the PID.
func writeDroidFailureScript(t *testing.T, dir, logPath, logLine string) string {
	t.Helper()
	script := filepath.Join(dir, "droid")
	body := "#!/bin/sh\n" +
		"printf " + shellSingleQuote(logLine+"\\n") + " \"$$\" >> " + shellSingleQuote(logPath) + "\n" +
		"echo 'Error during droid execution: Exec failed' >&2\n" +
		"exit 1\n"
	require.NoError(t, os.WriteFile(script, []byte(body), 0o755), "write fake droid: %v")
	return script
}

// shellSingleQuote quotes s for a POSIX shell.
func shellSingleQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func TestDroidFailureRecoversProviderCauseFromLog(t *testing.T) {
	skipIfWindows(t)
	assert := assert.New(t)
	tmp := t.TempDir()
	logPath := filepath.Join(tmp, "droid-log-single.log")

	// Pre-run history from a different droid process: it must not leak into
	// this failure, which is what the offset + PID filters guarantee.
	history := `[2026-01-01T00:00:00.000Z] WARN: [LLM] Chat route failure | Context: {"process.pid":99999,"error":{"message":"403 allowance for this period is used up (other process)"}}` + "\n"
	require.NoError(t, os.WriteFile(logPath, []byte(history), 0o644), "seed log history: %v")
	withDroidLogPath(t, logPath)

	// Shape captured from droid 0.228.0 on a real provider failure.
	logLine := `[2026-01-01T00:00:01.000Z] WARN: [LLM] Chat route failure | Context: {"process.pid":%s,"error":{"name":"Error","message":"403 Your subscription allowance for this period is used up, and your plan does not allow spending wallet balance on top of it."}}`
	cmdPath := writeDroidFailureScript(t, tmp, logPath, logLine)
	a := NewDroidAgent(cmdPath)

	_, err := a.Review(context.Background(), tmp, "HEAD", "review", nil)
	require.Error(t, err)

	assert.Contains(err.Error(), "Exec failed")
	assert.Contains(err.Error(), "403 Your subscription allowance for this period is used up")
	assert.NotContains(err.Error(), "other process", "history from another PID must not leak")
	assert.Equal(LimitKindQuota, ClassifyLimit("droid", err.Error()).Kind)

	cls, attached := LimitClassificationFromError(err)
	assert.True(attached, "droid failures carry their own classification")
	assert.Equal(LimitKindQuota, cls.Kind)
}

func TestDroidFailureOmitsStdoutFromError(t *testing.T) {
	assert := assert.New(t)
	withDroidLogPath(t, filepath.Join(t.TempDir(), "droid-log-single.log"))

	// stdout is the review text. If it were folded into the error, a commit
	// that merely mentions quota wording would cool the agent down.
	mock := mockAgentCLI(t, MockCLIOpts{
		StderrLines: []string{"Error during droid execution: Exec failed"},
		StdoutLines: []string{"## Summary", "This change handles quota exceeded and exhausts your capacity."},
		ExitCode:    1,
	})
	a := NewDroidAgent(mock.CmdPath)

	_, err := a.Review(context.Background(), t.TempDir(), "HEAD", "review", nil)
	require.Error(t, err)

	assert.NotContains(err.Error(), "exhausts your capacity")
	assert.NotContains(err.Error(), "quota exceeded")
	assert.Equal(LimitKindNone, ClassifyLimit("droid", err.Error()).Kind)
}

func TestDroidFailureWithoutLogFallsBackToStderr(t *testing.T) {
	assert := assert.New(t)
	withDroidLogPath(t, filepath.Join(t.TempDir(), "no-such-log.log"))

	mock := mockAgentCLI(t, MockCLIOpts{
		StderrLines: []string{"Error during droid execution: Exec failed"},
		ExitCode:    1,
	})
	a := NewDroidAgent(mock.CmdPath)

	_, err := a.Review(context.Background(), t.TempDir(), "HEAD", "review", nil)
	require.Error(t, err)

	assert.Contains(err.Error(), "Exec failed")
	assert.Equal(LimitKindNone, ClassifyLimit("droid", err.Error()).Kind)
}

func TestDroidLogLineCause(t *testing.T) {
	tests := []struct {
		name string
		line string
		want string
	}{
		{
			name: "error object with message",
			line: `[T] WARN: [LLM] Chat route failure | Context: {"process.pid":1,"error":{"name":"Error","message":"401 User not found."}}`,
			want: "401 User not found.",
		},
		{
			name: "errorMessage field",
			line: `[T] INFO: [metrics_log_agent_error_count] | Context: {"reason":"[Agent] runAgent error","errorMessage":"401 User not found.","process.pid":1}`,
			want: "401 User not found.",
		},
		{
			name: "cause object with message",
			line: `[T] WARN: x | Context: {"cause":{"name":"TypeError","message":"Unable to connect. Is the computer able to access the url?"},"process.pid":1}`,
			want: "Unable to connect. Is the computer able to access the url?",
		},
		{
			name: "error as bare string",
			line: `[T] WARN: x | Context: {"error":"Error: 401 User not found.","process.pid":1}`,
			want: "Error: 401 User not found.",
		},
		{
			name: "wrapper text is not a cause",
			line: `[T] ERROR: Error running droid command | Context: {"error":"MetaError: Exec failed","process.pid":1}`,
			want: "",
		},
		{
			name: "no context json",
			line: `[T] ERROR: Error running droid command`,
			want: "",
		},
		{
			name: "unrelated field ignored",
			line: `[T] WARN: x | Context: {"errorName":"YAMLException","process.pid":1}`,
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, droidInformativeCause(droidLogLineCause(tt.line)))
		})
	}
}

func TestDroidLogLinePID(t *testing.T) {
	assert := assert.New(t)
	assert.Equal(79981, droidLogLinePID(`{"process.pid":79981}`))
	assert.Equal(79981, droidLogLinePID(`x {"process.pid":79981,"y":1}`))
	assert.Equal(799810, droidLogLinePID(`{"process.pid":799810,"y":1}`))
	assert.Equal(0, droidLogLinePID(`{"sessionId":"abc"}`))
}

func TestDroidLogCauseSkipsOtherProcesses(t *testing.T) {
	assert := assert.New(t)
	tmp := t.TempDir()
	logPath := filepath.Join(tmp, "droid-log-single.log")

	// A longer PID that shares a prefix with ours must not be picked up.
	content := `[T] WARN: x | Context: {"process.pid":799810,"error":{"message":"403 allowance for this period is used up (prefix collision)"}}` + "\n" +
		`[T] WARN: x | Context: {"process.pid":99999,"error":{"message":"403 allowance for this period is used up (other process)"}}` + "\n"
	require.NoError(t, os.WriteFile(logPath, []byte(content), 0o644), "seed log: %v")

	assert.Empty(droidLogCause(logPath, 0, 79981), "no line is tagged with the process we ran")
	assert.Empty(droidLogCause(logPath, 0, 0), "pid 0 matches nothing")

	content += `[T] WARN: x | Context: {"process.pid":79981,"error":{"message":"403 Your subscription allowance for this period is used up."}}` + "\n"
	require.NoError(t, os.WriteFile(logPath, []byte(content), 0o644), "append our line: %v")

	assert.Equal(
		"403 Your subscription allowance for this period is used up.",
		droidLogCause(logPath, 0, 79981),
	)
}

func TestDroidLogCauseReadsOnlyAppendedRegion(t *testing.T) {
	assert := assert.New(t)
	tmp := t.TempDir()
	logPath := filepath.Join(tmp, "droid-log-single.log")

	// History carries a cause for the same PID; the region this run appended
	// does not. Reading from the pre-run offset must therefore find nothing.
	history := `[T] WARN: x | Context: {"process.pid":1234,"error":{"message":"403 allowance for this period is used up (history)"}}` + "\n"
	appended := `[T] INFO: x | Context: {"process.pid":1234,"tags":{"clientType":"cli"}}` + "\n"
	require.NoError(t, os.WriteFile(logPath, []byte(history+appended), 0o644), "seed log: %v")

	assert.Empty(droidLogCause(logPath, int64(len(history)), 1234))
	assert.Equal(
		"403 allowance for this period is used up (history)",
		droidLogCause(logPath, 0, 1234),
		"without an offset the earlier cause is visible",
	)
}
