//go:build windows

package main

// diskFree is best-effort; on Windows the info command reports it as unavailable
// (matching the Python OSError branch), avoiding a Win32 dependency.
func diskFree(path string) (uint64, bool) { return 0, false }
