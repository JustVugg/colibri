package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// oracleScript loads the Python `c/coli` as a module and calls its env_for with a
// namespace built from the JSON spec, then dumps the resulting child environment.
// It is the authoritative reference the Go envFor must reproduce byte-for-byte.
const oracleScript = `
import sys, os, json, types, importlib.machinery, importlib.util
spec = json.loads(sys.argv[1])
os.environ.clear()
os.environ.update(spec["_env"])
loader = importlib.machinery.SourceFileLoader("coli_cli", spec["_coli"])
mspec = importlib.util.spec_from_loader("coli_cli", loader)
coli = importlib.util.module_from_spec(mspec)
loader.exec_module(coli)
ns = types.SimpleNamespace(
    model=spec["model"], policy=spec["policy"],
    ram=int(spec["ram"]), ngen=int(spec["ngen"]),
    topp=float(spec["topp"]), topk=int(spec["topk"]),
    temp=(None if spec["temp"] is None else float(spec["temp"])),
    repin=int(spec["repin"]), ctx=int(spec["ctx"]),
    auto_tier=bool(spec["auto_tier"]), gpu=spec["gpu"], vram=float(spec["vram"]),
)
json.dump(coli.env_for(ns), sys.stdout)
`

// pythonEnvFor runs the oracle for one case and returns the reference env map.
func pythonEnvFor(t *testing.T, spec map[string]any, base map[string]string) map[string]string {
	t.Helper()
	coli, err := filepath.Abs(filepath.Join("..", "coli"))
	if err != nil {
		t.Fatal(err)
	}
	spec["_coli"] = coli
	spec["_env"] = base
	payload, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("python3", "-", string(payload))
	cmd.Stdin = strings.NewReader(oracleScript)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		t.Fatalf("oracle failed: %v\nstderr: %s", err, errBuf.String())
	}
	var m map[string]string
	if err := json.Unmarshal(out.Bytes(), &m); err != nil {
		t.Fatalf("oracle output not a string map: %v\n%s", err, out.String())
	}
	return m
}

// withEnv clears the process environment, sets base, runs fn, then restores.
// envFor reads os.Environ() through currentEnvMap, so this pins its input.
func withEnv(base map[string]string, fn func()) {
	saved := os.Environ()
	os.Clearenv()
	for k, v := range base {
		os.Setenv(k, v)
	}
	defer func() {
		os.Clearenv()
		for _, kv := range saved {
			if i := strings.IndexByte(kv, '='); i >= 0 {
				os.Setenv(kv[:i], kv[i+1:])
			}
		}
	}()
	fn()
}

// argsFromSpec builds the Go Args matching a Python oracle spec. Only the fields
// env_for reads are populated; provided[] mirrors argparse's `is not None`.
func argsFromSpec(spec map[string]any) *Args {
	a := &Args{provided: map[string]bool{}}
	a.Model = spec["model"].(string)
	a.Policy = spec["policy"].(string)
	a.Ram = spec["ram"].(int)
	a.Ngen = spec["ngen"].(int)
	a.Topp = spec["topp"].(float64)
	a.Topk = spec["topk"].(int)
	if spec["temp"] != nil {
		a.Temp = spec["temp"].(float64)
		a.provided["temp"] = true
	}
	a.Repin = spec["repin"].(int)
	a.Ctx = spec["ctx"].(int)
	a.AutoTier = spec["auto_tier"].(bool)
	if spec["gpu"] != nil {
		a.Gpu = spec["gpu"].(string)
		a.provided["gpu"] = true
	}
	a.Vram = spec["vram"].(float64)
	return a
}

func TestEnvForMatchesPython(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	if isWindows {
		t.Skip("golden covers the non-Windows env_for branch")
	}

	// A base env that also exercises the gpu=none pop branch.
	base := map[string]string{
		"PATH":           "/usr/bin:/bin",
		"COLI_GPU":       "0",
		"CUDA_DENSE":     "1",
		"CUDA_EXPERT_GB": "5",
		"UNRELATED":      "keep-me",
	}

	def := func(over map[string]any) map[string]any {
		s := map[string]any{
			"model": "/models/glm", "policy": "quality", "ram": 0, "ngen": 0,
			"topp": 0.0, "topk": 0, "temp": nil, "repin": 0, "ctx": 0,
			"auto_tier": false, "gpu": nil, "vram": 0.0,
		}
		for k, v := range over {
			s[k] = v
		}
		return s
	}

	cases := []struct {
		name string
		spec map[string]any
	}{
		{"defaults", def(nil)},
		{"ram", def(map[string]any{"ram": 8})},
		{"ngen", def(map[string]any{"ngen": 512})},
		{"topp", def(map[string]any{"topp": 0.95})},
		{"topk", def(map[string]any{"topk": 40})},
		{"temp_greedy", def(map[string]any{"temp": 0.0})},
		{"temp_set", def(map[string]any{"temp": 0.7})},
		{"repin", def(map[string]any{"repin": 64})},
		{"ctx", def(map[string]any{"ctx": 8192})},
		{"policy_balanced", def(map[string]any{"policy": "balanced"})},
		{"gpu_none", def(map[string]any{"gpu": "none"})},
		{"gpu_none_vram_ignored", def(map[string]any{"gpu": "none", "vram": 12.0})},
		{"combo", def(map[string]any{
			"ram": 16, "ngen": 256, "topp": 0.9, "topk": 20, "temp": 0.3,
			"repin": 32, "ctx": 4096, "policy": "experimental-fast",
		})},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := pythonEnvFor(t, tc.spec, base)
			var got map[string]string
			withEnv(base, func() { got = envFor(argsFromSpec(tc.spec)) })
			if !reflect.DeepEqual(got, want) {
				t.Errorf("env mismatch\n go: %v\n py: %v", got, want)
			}
		})
	}
}

func TestFtoa(t *testing.T) {
	cases := []struct {
		in   float64
		want string
	}{
		{0.0, "0.0"}, {300.0, "300.0"}, {0.95, "0.95"}, {1.0, "1.0"},
		{12.5, "12.5"}, {0.1, "0.1"}, {-1.0, "-1.0"},
	}
	for _, c := range cases {
		if got := ftoa(c.in); got != c.want {
			t.Errorf("ftoa(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestParseStat(t *testing.T) {
	s := parseStat("STAT 10 5.5 80 12.3", false)
	if s.Tok != 10 || s.Tps != 5.5 || s.Hit != 80 || s.Rss != 12.3 || s.Interrupted {
		t.Errorf("parseStat full = %+v", s)
	}
	if s := parseStat("garbage line", true); s.Tok != 0 || !s.Interrupted {
		t.Errorf("parseStat garbage = %+v, want zero+interrupted", s)
	}
	if s := parseStat("STAT 1 2 3", false); s.Tok != 0 {
		t.Errorf("parseStat short line = %+v, want zero", s)
	}
}
