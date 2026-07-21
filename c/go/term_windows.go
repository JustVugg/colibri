//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

// kernel32 is declared in diskfree_other.go (same package, Windows build).
var procGetConsoleScreenBufferInfo = kernel32.NewProc("GetConsoleScreenBufferInfo")

type coord struct{ X, Y int16 }
type smallRect struct{ Left, Top, Right, Bottom int16 }
type consoleScreenBufferInfo struct {
	size              coord
	cursorPosition    coord
	attributes        uint16
	window            smallRect
	maximumWindowSize coord
}

// terminalCols returns the console width (window right-left+1) via
// GetConsoleScreenBufferInfo on stdout, or 0 when stdout is not a console.
func terminalCols() int {
	h, err := syscall.GetStdHandle(syscall.STD_OUTPUT_HANDLE)
	if err != nil {
		return 0
	}
	var info consoleScreenBufferInfo
	r, _, _ := procGetConsoleScreenBufferInfo.Call(uintptr(h), uintptr(unsafe.Pointer(&info)))
	if r == 0 {
		return 0
	}
	return int(info.window.Right-info.window.Left) + 1
}
