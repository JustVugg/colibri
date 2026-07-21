//go:build !windows

package main

import "runtime"

// physicalCPUCount is only consulted on the Windows OMP_NUM_THREADS path; on
// other systems the launcher never calls it (the engine tunes OMP itself). The
// logical count is a fine placeholder here — it keeps the cross-platform build
// whole without a Unix-specific core enumeration the launcher would never use.
func physicalCPUCount() int { return runtime.NumCPU() }
