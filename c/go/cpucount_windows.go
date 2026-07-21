//go:build windows

package main

import (
	"runtime"
	"unsafe"
)

// kernel32 is declared in diskfree_other.go (same package, Windows build).
var procGetLogicalProcessorInformationEx = kernel32.NewProc("GetLogicalProcessorInformationEx")

// relationProcessorCore selects physical-core records from
// GetLogicalProcessorInformationEx (LOGICAL_PROCESSOR_RELATIONSHIP).
const relationProcessorCore = 0

// physicalCPUCount counts physical cores via GetLogicalProcessorInformationEx,
// matching the Python physical_cpu_count() so the Windows OMP_NUM_THREADS default
// is the physical (not logical) count — on a hyperthreaded box the logical count
// would over-subscribe the memory-bandwidth-bound engine. Falls back to the
// logical count if the query fails.
func physicalCPUCount() int {
	var needed uint32
	// First call with a nil buffer returns the required byte length in `needed`.
	procGetLogicalProcessorInformationEx.Call(uintptr(relationProcessorCore), 0, uintptr(unsafe.Pointer(&needed)))
	if needed == 0 {
		return runtime.NumCPU()
	}
	buf := make([]byte, needed)
	r, _, _ := procGetLogicalProcessorInformationEx.Call(
		uintptr(relationProcessorCore), uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&needed)))
	if r == 0 {
		return runtime.NumCPU()
	}
	// The buffer is a packed sequence of variable-length records; each record's
	// Size field (uint32 at offset+4) gives its length. Every RelationProcessorCore
	// record is one physical core.
	count := 0
	for off := uint32(0); off+8 <= needed; {
		relationship := *(*uint32)(unsafe.Pointer(&buf[off]))
		size := *(*uint32)(unsafe.Pointer(&buf[off+4]))
		if size == 0 {
			break
		}
		if relationship == relationProcessorCore {
			count++
		}
		off += size
	}
	if count == 0 {
		return runtime.NumCPU()
	}
	return count
}
