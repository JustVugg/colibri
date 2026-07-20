package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// runInherit runs cmd with the parent's stdio and returns its exit code.
func runInherit(cmd *exec.Cmd) int {
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintln(os.Stderr, "coli: "+err.Error())
		return 1
	}
	return 0
}

func cmdBuild(a *Args) int {
	banner("build")
	if !exists(filepath.Join(SRC, "Makefile")) {
		fatal(fmt.Sprintf("%scoli build%s only works from a source checkout (this is an installed copy).\n"+
			"  Clone https://github.com/JustVugg/colibri and run ./setup.sh, or make -C c glm.", C.yel, C.r))
	}
	return runInherit(exec.Command("make", "-C", SRC, "glm"))
}

func infoRow(k, v string) { fmt.Printf("   %s%-10s%s %s\n", C.gray, k, C.r, v) }

// jsonGet returns the config value as a string, or "None" when absent — matching
// Python's f-string interpolation of a missing dict key.
func jsonGet(m map[string]json.RawMessage, key string) string {
	raw, ok := m[key]
	if !ok {
		return "None"
	}
	return strings.Trim(string(raw), `"`)
}

func cmdInfo(a *Args) int {
	banner("info")
	cfgPath := filepath.Join(a.Model, "config.json")
	if data, err := os.ReadFile(cfgPath); err == nil {
		var cfg map[string]json.RawMessage
		if json.Unmarshal(data, &cfg) == nil {
			infoRow("model", a.Model)
			infoRow("arch", fmt.Sprintf("hidden %s · %s layer · %s expert/layer · top-%s",
				jsonGet(cfg, "hidden_size"), jsonGet(cfg, "num_hidden_layers"),
				jsonGet(cfg, "n_routed_experts"), jsonGet(cfg, "num_experts_per_tok")))
			var count int
			var total int64
			if entries, err := os.ReadDir(a.Model); err == nil {
				for _, e := range entries {
					if strings.HasSuffix(e.Name(), ".safetensors") {
						count++
						if fi, err := e.Info(); err == nil {
							total += fi.Size()
						}
					}
				}
			}
			infoRow("shards", fmt.Sprintf("%d files · %.0f GB on disk", count, float64(total)/1e9))
		}
	} else {
		fmt.Printf("   %sconfig.json is missing (incomplete conversion?)%s\n", C.yel, C.r)
	}

	if mi, err := os.ReadFile("/proc/meminfo"); err == nil {
		tot := meminfoField(mi, "MemTotal")
		av := meminfoField(mi, "MemAvailable")
		if tot > 0 {
			infoRow("RAM", fmt.Sprintf("%.0f GB total · %.1f GB available", tot/1e6, av/1e6))
		}
	}

	target := SRC
	if fi, err := os.Stat(a.Model); err == nil && fi.IsDir() {
		target = a.Model
	}
	if free, ok := diskFree(target); ok {
		infoRow("disk", fmt.Sprintf("%.0f GB free", float64(free)/1e9))
	} else {
		infoRow("disk", "? GB (unavailable)")
	}

	if exists(GLM) {
		infoRow("engine", "ready ✓")
	} else {
		infoRow("engine", "not built (coli build)")
	}

	var knobs []string
	if a.Ram != 0 {
		knobs = append(knobs, fmt.Sprintf("ram %dGB", a.Ram))
	}
	if a.Topp != 0 {
		knobs = append(knobs, "topp "+ftoa(a.Topp))
	}
	if a.Topk != 0 {
		knobs = append(knobs, "topk "+strconv.Itoa(a.Topk))
	}
	if len(knobs) > 0 {
		infoRow("tuning", strings.Join(knobs, " · "))
	}
	fmt.Println()
	return 0
}

var meminfoRe = regexp.MustCompile(`(\d+)`)

func meminfoField(data []byte, key string) float64 {
	for _, line := range bytes.Split(data, []byte("\n")) {
		if bytes.HasPrefix(line, []byte(key+":")) {
			if m := meminfoRe.Find(line[len(key)+1:]); m != nil {
				v, _ := strconv.ParseFloat(string(m), 64)
				return v
			}
		}
	}
	return 0
}
