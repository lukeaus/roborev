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
		name     string
		agentic  bool
		setup    func(*DroidAgent) *DroidAgent
		wantArgs []string
		dontWant []string
	}{
		{
			name:     "Non-agentic default",
			agentic:  false,
			wantArgs: []string{"exec", "--tag", "roborev", "--auto", "low"},
			dontWant: []string{"medium"},
		},
		{
			name:     "Model set",
			setup:    func(a *DroidAgent) *DroidAgent { return a.WithModel("custom:foo").(*DroidAgent) },
			wantArgs: []string{"-m", "custom:foo"},
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
			a := NewDroidAgent("droid")
			if tt.setup != nil {
				a = tt.setup(a)
			}

			args := a.buildArgs(tt.agentic)

			for _, want := range tt.wantArgs {
				assertContainsArg(t, args, want)
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
	a := NewDroidAgent("droid")

	a2 := a.WithModel("custom:foo").(*DroidAgent)
	assert.Equal(t, "custom:foo", a2.Model)
	assert.Empty(t, a.Model, "original should be unchanged")

	a3 := a2.WithReasoning(ReasoningThorough).(*DroidAgent)
	assert.Equal(t, "custom:foo", a3.Model, "model should survive other clones")
	assert.Contains(t, a3.CommandLine(), "-m custom:foo")
}

func TestDroidFailureIncludesStdoutAndClassifiesQuota(t *testing.T) {
	mock := mockAgentCLI(t, MockCLIOpts{
		StderrLines: []string{"Error during droid execution: Exec failed"},
		StdoutLines: []string{"403 Your subscription allowance for this period is used up."},
		ExitCode:    1,
	})
	a := NewDroidAgent(mock.CmdPath)

	_, err := a.Review(context.Background(), t.TempDir(), "HEAD", "review", nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "Exec failed")
	assert.Contains(t, err.Error(), "allowance for this period is used up")
	assert.Equal(t, LimitKindQuota, ClassifyLimit("droid", err.Error()).Kind)
}
