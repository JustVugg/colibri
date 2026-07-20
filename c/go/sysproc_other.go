//go:build windows

package main

import "os/exec"

// Windows has no POSIX process groups; the gateway child is signalled directly.
func isolateProcessGroup(cmd *exec.Cmd) {}
