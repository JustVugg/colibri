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
