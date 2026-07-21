package main

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestDriverErrorPaths exercises the embedded Python drivers against the real
// resource_plan.py / doctor.py via their deterministic error paths. If those
// modules' APIs drift (renamed/re-signatured build_plan, run_doctor, etc.) the
// import or the call fails before the expected sys.exit, so the exit code no
// longer matches — which is exactly the drift we want to catch. No fixture model
// is required: the errors trigger before any model is read.
func TestDriverErrorPaths(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	src, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	// runDriver uses the package globals PYTHON/SRC/GLM; point them at the source
	// tree for the duration of the test.
	oldPY, oldSRC, oldGLM := PYTHON, SRC, GLM
	PYTHON, SRC, GLM = "python3", src, filepath.Join(src, "glm")
	defer func() { PYTHON, SRC, GLM = oldPY, oldSRC, oldGLM }()

	t.Run("plan_invalid_model", func(t *testing.T) {
		a := &Args{Model: "/nonexistent-model-xyz", Policy: "quality", provided: map[string]bool{}}
		_, stderr, code, _ := runDriver(planDriver, a, nil)
		if code != 3 {
			t.Fatalf("plan exit code = %d, want 3 (import/build_plan drift?); stderr: %s", code, stderr)
		}
		if strings.TrimSpace(stderr) == "" {
			t.Errorf("plan wrote no error text")
		}
	})

	t.Run("doctor_negative_vram", func(t *testing.T) {
		a := &Args{Model: "/nonexistent-model-xyz", Policy: "quality", Vram: -1, provided: map[string]bool{}}
		out, stderr, code, _ := runDriver(doctorDriver, a, nil)
		if code != 2 {
			t.Fatalf("doctor exit code = %d, want 2 (import/validation drift?); stderr: %s", code, stderr)
		}
		if !strings.Contains(out, "--vram cannot be negative") {
			t.Errorf("doctor output missing the vram error, got: %s", out)
		}
	})
}
