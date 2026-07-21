//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

// kernel32 is declared in diskfree_other.go (same package, Windows build).
var procGenerateConsoleCtrlEvent = kernel32.NewProc("GenerateConsoleCtrlEvent")

const ctrlBreakEvent = 1 // CTRL_BREAK_EVENT

// newProcessGroup makes the child a new process-group leader. Windows has no
// POSIX process groups; CREATE_NEW_PROCESS_GROUP both detaches the child from the
// console's Ctrl-C (coli relays explicitly) and makes it addressable as a group
// by GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid).
func newProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

// isolateProcessGroup and prepareEngineProc both need the child in its own group
// so a Ctrl-Break can be targeted at it (the gateway and the chat engine
// respectively).
func isolateProcessGroup(cmd *exec.Cmd) { newProcessGroup(cmd) }
func prepareEngineProc(cmd *exec.Cmd)   { newProcessGroup(cmd) }

// interruptChild / terminateChild deliver CTRL_BREAK_EVENT to the child's process
// group. Windows cannot deliver POSIX SIGINT/SIGTERM to another process, so the
// old Process.Signal(SIGINT/SIGTERM) calls were silent no-ops here. Ctrl-Break is
// the console-control equivalent a process-group leader can receive.
//
// Note: whether the child shuts down *gracefully* on Ctrl-Break depends on its
// own console-control handler — the engine (glm.c) and the Python gateway
// (openai_server.py, which today handles SIGTERM). Graceful Windows shutdown for
// those is tracked with their own ports; here coli delivers a real event instead
// of failing silently.
func interruptChild(cmd *exec.Cmd) error { return generateConsoleCtrlEvent(cmd.Process.Pid) }
func terminateChild(cmd *exec.Cmd) error { return generateConsoleCtrlEvent(cmd.Process.Pid) }

func generateConsoleCtrlEvent(pid int) error {
	r, _, err := procGenerateConsoleCtrlEvent.Call(uintptr(ctrlBreakEvent), uintptr(pid))
	if r == 0 {
		return err
	}
	return nil
}
