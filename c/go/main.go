// Command coli is a dependency-free Go port of the colibrì `coli` launcher
// (issue #310). It reproduces the Python CLI's argument parsing, environment
// setup and engine process management. The build/info/run/chat commands run
// natively (Python-free); serve/web/plan/doctor/bench/convert still delegate to
// the existing Python support code, which stays authoritative.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

var isWindows = runtime.GOOS == "windows"

// Resolved once at startup, mirroring c/coli lines 42-68.
var (
	exeSuffix string // ".exe" on Windows
	GLM       string // engine binary
	TOOLS     string // tools/ directory (Python helper scripts)
	SRC       string // directory holding the Python support modules
	PYTHON    string // interpreter for delegated commands
)

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func resolvePaths() {
	if isWindows {
		exeSuffix = ".exe"
	}
	exe, err := os.Executable()
	if err == nil {
		if resolved, e := filepath.EvalSymlinks(exe); e == nil {
			exe = resolved
		}
	}
	here := filepath.Dir(exe)

	// SRC is where the Python support modules (openai_server.py, resource_plan.py,
	// doctor.py) live — resolved via the source layout, independent of
	// COLI_ENGINE (which the Python coli finds through sys.path, not the engine
	// dir). The Go binary typically sits in c/go/ or c/ while the support files
	// and engine sit in c/. Fall back to the installed libexec layout.
	found := false
	for _, base := range []string{here, filepath.Dir(here)} {
		if exists(filepath.Join(base, "glm"+exeSuffix)) || exists(filepath.Join(base, "openai_server.py")) {
			SRC = base
			found = true
			break
		}
	}
	if !found {
		SRC = filepath.Join(filepath.Dir(here), "libexec", "colibri")
	}

	// Engine/tools resolution mirrors c/coli lines 53-66: COLI_ENGINE overrides
	// the engine (tools sit next to it); else the engine next to the support
	// files if it has been built; else the installed libexec layout. When the
	// engine is not built, this reports the libexec path exactly like the Python
	// coli (so `info`/`doctor` show the same "not built" location).
	libexec := filepath.Join(filepath.Dir(SRC), "libexec", "colibri")
	switch {
	case os.Getenv("COLI_ENGINE") != "":
		GLM = os.Getenv("COLI_ENGINE")
		TOOLS = filepath.Join(filepath.Dir(GLM), "tools")
	case exists(filepath.Join(SRC, "glm"+exeSuffix)):
		GLM = filepath.Join(SRC, "glm"+exeSuffix)
		TOOLS = filepath.Join(SRC, "tools")
	default:
		GLM = filepath.Join(libexec, "glm"+exeSuffix)
		TOOLS = filepath.Join(libexec, "tools")
	}
	PYTHON = resolvePython(SRC)
}

func main() {
	raiseFileLimit()
	resolvePaths()

	a, err := parseArgs(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "coli: "+err.Error())
		os.Exit(2)
	}
	if a.Cmd == "" {
		banner("")
		fmt.Println(usageDoc)
		return
	}

	handlers := map[string]func(*Args) int{
		"build":   cmdBuild,
		"info":    cmdInfo,
		"run":     cmdRun,
		"chat":    cmdChat,
		"serve":   cmdServe,
		"web":     cmdWeb,
		"plan":    cmdPlan,
		"doctor":  cmdDoctor,
		"bench":   cmdBench,
		"convert": cmdConvert,
	}
	if h := handlers[a.Cmd]; h != nil {
		os.Exit(h(a))
	}
	banner("")
	fmt.Println(usageDoc)
}
