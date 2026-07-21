//go:build !windows

package main

import (
	"os"
	"syscall"
	"unsafe"
)

type winsize struct{ Row, Col, Xpixel, Ypixel uint16 }

// terminalCols returns the controlling terminal's column count via the
// TIOCGWINSZ ioctl on stdout, or 0 when stdout is not a terminal.
func terminalCols() int {
	ws := &winsize{}
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, os.Stdout.Fd(),
		uintptr(syscall.TIOCGWINSZ), uintptr(unsafe.Pointer(ws)))
	if errno != 0 {
		return 0
	}
	return int(ws.Col)
}
