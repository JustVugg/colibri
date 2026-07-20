//go:build !windows

package main

import "syscall"

// raiseFileLimit mirrors the RLIMIT_NOFILE bump in the Python `coli` (lines
// 25-34): the engine mmaps every shard (144+ files) and macOS' default soft
// limit of 256 is not enough. Best-effort; failures are ignored, exactly as in
// the Python original.
func raiseFileLimit() {
	const want = 65536
	var lim syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &lim); err != nil {
		return
	}
	target := uint64(want)
	if lim.Max != 0 && lim.Max < target {
		target = lim.Max
	}
	if lim.Cur < target {
		lim.Cur = target
		_ = syscall.Setrlimit(syscall.RLIMIT_NOFILE, &lim)
	}
}
