//go:build !windows

package main

import "syscall"

// diskFree returns the free bytes available on the filesystem containing path.
func diskFree(path string) (uint64, bool) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, false
	}
	return st.Bavail * uint64(st.Bsize), true
}
