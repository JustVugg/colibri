//go:build !windows

package main

import (
	"os/exec"
	"syscall"
)

// isolateProcessGroup puts the child in its own process group so a terminal
// Ctrl-C (delivered to the whole foreground group) does not reach it directly;
// coli relays signals to it explicitly instead, so the gateway shuts down once
// and cleanly — matching the single-SIGINT the in-process Python `coli serve`
// receives.
func isolateProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// prepareEngineProc is a no-op on Unix: the chat engine shares coli's process
// group (matching the Python launcher), and interruptChild targets it by pid.
func prepareEngineProc(cmd *exec.Cmd) {}

// interruptChild forwards SIGINT (the first Ctrl-C during a stream).
func interruptChild(cmd *exec.Cmd) error {
	return cmd.Process.Signal(syscall.SIGINT)
}

// terminateChild forwards SIGTERM (graceful shutdown of the gateway).
func terminateChild(cmd *exec.Cmd) error {
	return cmd.Process.Signal(syscall.SIGTERM)
}
