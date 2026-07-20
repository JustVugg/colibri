//go:build windows

package main

// Windows has no RLIMIT_NOFILE; the Python original skips this on win32 too.
func raiseFileLimit() {}
