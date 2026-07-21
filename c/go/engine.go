package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Engine byte-protocol sentinels — byte-identical to c/coli and openai_server.py.
var (
	sentinelEND   = []byte("\x01\x01END\x01\x01\n")
	sentinelREADY = []byte("\x01\x01READY\x01\x01\n")
)

// errEngineDead mirrors the Python streamTurn returning None: the engine's
// stdout closed, i.e. the process died. errInterruptQuit is the second Ctrl-C.
var (
	errEngineDead    = errors.New("engine stdout closed")
	errInterruptQuit = errors.New("interrupted")
)

// Stat is one parsed "STAT tok tps hit rss" line.
type Stat struct {
	Tok         int
	Tps         float64
	Hit         float64
	Rss         float64
	Interrupted bool
}

func resolvePython(src string) string {
	sub := "bin"
	if isWindows {
		sub = "Scripts"
	}
	venv := filepath.Join(src, "mio_env", sub, "python3")
	if isWindows {
		venv += ".exe"
	}
	if exists(venv) {
		return venv
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	if p, err := exec.LookPath("python"); err == nil {
		return p
	}
	return "python3"
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(1)
}

func needModel(model string) {
	fi, err := os.Stat(model)
	if err != nil || !fi.IsDir() {
		fatal(fmt.Sprintf("%smodel not found:%s %s\n  set COLI_MODEL or use --model", C.yel, C.r, model))
	}
	if !exists(filepath.Join(model, "tokenizer.json")) {
		fatal(fmt.Sprintf("%stokenizer.json is missing from %s%s", C.yel, model, C.r))
	}
	if !exists(GLM) {
		fatal(fmt.Sprintf("%sengine is not built.%s Run: coli build", C.yel, C.r))
	}
}

// cudaBinary reports whether the engine is linked against a resolvable CUDA
// runtime (Linux only), matching cuda_binary() in c/coli.
func cudaBinary() bool {
	if !exists(GLM) || runtime.GOOS != "linux" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ldd", GLM).Output()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "libcudart") && !strings.Contains(line, "not found") {
			return true
		}
	}
	return false
}

func currentEnvMap() map[string]string {
	e := map[string]string{}
	for _, kv := range os.Environ() {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			e[kv[:i]] = kv[i+1:]
		}
	}
	return e
}

func envToSlice(e map[string]string) []string {
	out := make([]string, 0, len(e))
	for k, v := range e {
		out = append(out, k+"="+v)
	}
	sort.Strings(out) // deterministic order (aids testing/debugging)
	return out
}

func setdefault(e map[string]string, k, v string) {
	if _, ok := e[k]; !ok {
		e[k] = v
	}
}

// ftoa formats a float to match Python's str(float): whole values keep a
// trailing ".0" (str(0.0) == "0.0", str(300.0) == "300.0"), so the environment
// passed to the engine is byte-identical to what the Python coli produced.
func ftoa(f float64) string {
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eEnN") { // no decimal point, exponent, inf or nan
		s += ".0"
	}
	return s
}

// envFor reproduces env_for() in c/coli: the child engine's environment. It may
// terminate the process (like sys.exit) when --gpu/--vram are used without a
// CUDA build. --auto-tier is resolved through the Python resource_plan driver.
func envFor(a *Args) map[string]string {
	e := currentEnvMap()
	e["SNAP"] = a.Model
	if isWindows {
		// Measured Windows defaults: presence-based, all setdefault. COLI_NO_OMP_TUNE
		// (any non-empty value) disables ONLY the OMP block; the I/O defaults keep
		// their own kill-switches (DIRECT=0 etc.).
		if e["COLI_NO_OMP_TUNE"] == "" {
			setdefault(e, "OMP_WAIT_POLICY", "active")
			setdefault(e, "GOMP_SPINCOUNT", "200000")
			setdefault(e, "OMP_DYNAMIC", "FALSE")
			setdefault(e, "OMP_NUM_THREADS", strconv.Itoa(physicalCPUCount()))
		}
		setdefault(e, "DIRECT", "1")
		setdefault(e, "PIPE", "1")
		setdefault(e, "PILOT_REAL", "1")
	}
	e["COLI_POLICY"] = a.Policy
	if a.Ram != 0 {
		e["RAM_GB"] = strconv.Itoa(a.Ram)
	}
	if a.Ngen != 0 {
		e["NGEN"] = strconv.Itoa(a.Ngen)
	}
	if a.Topp != 0 {
		e["TOPP"] = ftoa(a.Topp)
	}
	if a.Topk != 0 {
		e["TOPK"] = strconv.Itoa(a.Topk)
	}
	if a.provided["temp"] { // 0 = greedy; the engine default is 1.0 + nucleus 0.95
		e["TEMP"] = ftoa(a.Temp)
	}
	if a.Repin != 0 {
		e["REPIN"] = strconv.Itoa(a.Repin)
	}
	if a.Ctx != 0 {
		e["CTX"] = strconv.Itoa(a.Ctx)
	}
	if a.AutoTier {
		return autoTierEnv(a, e)
	}
	// --gpu/--vram WITHOUT --auto-tier: previously ignored silently and the run
	// started CPU-only with no warning (#121).
	if a.provided["gpu"] {
		delete(e, "COLI_GPU")
		delete(e, "COLI_GPUS")
		if a.Gpu == "none" {
			e["COLI_CUDA"] = "0"
			delete(e, "CUDA_EXPERT_GB")
			delete(e, "CUDA_DENSE")
		} else {
			if !cudaBinary() {
				fatal(fmt.Sprintf("%s--gpu needs the CUDA build:%s make glm CUDA=1 (this binary is CPU-only)", C.yel, C.r))
			}
			e["COLI_CUDA"] = "1"
			if a.Gpu != "auto" {
				e["COLI_GPUS"] = a.Gpu
			}
			setdefault(e, "CUDA_DENSE", "1")
		}
	}
	if a.Vram != 0 && a.Gpu != "none" {
		if !cudaBinary() {
			fatal(fmt.Sprintf("%s--vram needs the CUDA build:%s make glm CUDA=1 (this binary is CPU-only)", C.yel, C.r))
		}
		e["COLI_CUDA"] = "1"
		e["CUDA_EXPERT_GB"] = ftoa(a.Vram)
	}
	return e
}

// engineProc wraps the engine subprocess and a single persistent stdout reader
// goroutine feeding a byte channel, so streamTurn can consume the byte protocol
// and be interrupted by SIGINT even during a minutes-long cold prefill.
type engineProc struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr io.ReadCloser
	bytes  chan byte
}

func (ep *engineProc) startReader() {
	ep.bytes = make(chan byte, 4096)
	r := bufio.NewReader(ep.stdout)
	go func() {
		defer close(ep.bytes)
		for {
			b, err := r.ReadByte()
			if err != nil {
				return
			}
			ep.bytes <- b
		}
	}()
}

// readLine consumes bytes up to and including a newline, returning the line
// without the trailing '\n'. ok is false if the stream closed first.
func (ep *engineProc) readLine() (string, bool) {
	var sb strings.Builder
	for b := range ep.bytes {
		if b == '\n' {
			return sb.String(), true
		}
		sb.WriteByte(b)
	}
	return sb.String(), false
}

func (ep *engineProc) writeControl(s string) error {
	_, err := io.WriteString(ep.stdin, s)
	return err
}

func parseStat(line string, interrupted bool) *Stat {
	fields := strings.Fields(line)
	if len(fields) < 5 || fields[0] != "STAT" {
		return &Stat{Interrupted: interrupted}
	}
	tok, _ := strconv.Atoi(fields[1])
	tps, _ := strconv.ParseFloat(fields[2], 64)
	hit, _ := strconv.ParseFloat(fields[3], 64)
	rss, _ := strconv.ParseFloat(fields[4], 64)
	return &Stat{Tok: tok, Tps: tps, Hit: hit, Rss: rss, Interrupted: interrupted}
}

// streamTurn reads until the sentinel, handing response chunks to onBytes, then
// reads the STAT line. Faithful port of stream_turn() in c/coli: the first
// Ctrl-C during a stream forwards SIGINT to the engine and keeps draining; a
// second Ctrl-C quits. sigCh delivers os.Interrupt (nil disables handling).
func streamTurn(ep *engineProc, sentinel []byte, sigCh <-chan os.Signal, onBytes func([]byte)) (*Stat, error) {
	pend := make([]byte, 0, len(sentinel)+64)
	interrupted := false
	for {
		var b byte
		var ok bool
		if sigCh != nil {
			select {
			case sig := <-sigCh:
				_ = sig
				if interrupted || ep.cmd.ProcessState != nil {
					return nil, errInterruptQuit
				}
				interrupted = true
				_ = interruptChild(ep.cmd)
				fmt.Printf("\n  %s⏹ stopping… (Ctrl-C again to quit)%s\n", C.yel, C.r)
				continue
			case b, ok = <-ep.bytes:
			}
		} else {
			b, ok = <-ep.bytes
		}
		if !ok {
			return nil, errEngineDead
		}
		pend = append(pend, b)
		if bytes.HasSuffix(pend, sentinel) {
			rest := pend[:len(pend)-len(sentinel)]
			if len(rest) > 0 {
				onBytes(rest)
			}
			line, _ := ep.readLine()
			return parseStat(line, interrupted), nil
		}
		if len(pend) > len(sentinel) {
			out := pend[:len(pend)-len(sentinel)]
			onBytes(out)
			pend = append(pend[:0], pend[len(pend)-len(sentinel):]...)
		}
	}
}

// engineDiag explains why the engine died (mirrors engine_diag in c/coli): a
// silent SIGKILL is almost always the kernel OOM-killer.
func engineDiag(ep *engineProc, errlogPath string) {
	_ = ep.cmd.Wait()
	ps := ep.cmd.ProcessState
	why := "its output closed but the process is still alive"
	var killedBySIGKILL bool
	if ps != nil {
		if ws, ok := ps.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
			sig := ws.Signal()
			why = "killed by " + sig.String()
			killedBySIGKILL = sig == syscall.SIGKILL
		} else if code := ps.ExitCode(); code > 0 {
			why = "exit code " + strconv.Itoa(code)
		} else if code == 0 {
			why = "exited cleanly"
		}
	}
	fmt.Printf("\n  %s[engine terminated: %s]%s\n", C.yel, why, C.r)
	if killedBySIGKILL {
		fmt.Printf("  %snothing in the engine sends SIGKILL to itself: this is the kernel's\n"+
			"  OOM-killer. The peak RSS exceeded the machine's free memory.\n"+
			"  Lower --ram, lower PIN_GB, or shorten the context.%s\n", C.yel, C.r)
	}
	if errlogPath != "" {
		for i := 0; i < 20; i++ {
			if data, err := os.ReadFile(errlogPath); err == nil {
				tail := strings.TrimSpace(string(data))
				if tail != "" {
					lines := strings.Split(tail, "\n")
					if len(lines) > 6 {
						lines = lines[len(lines)-6:]
					}
					fmt.Printf("  %s%s%s\n", C.dgray, strings.Join(lines, "\n  "), C.r)
					break
				}
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
}

// notifyInterrupt returns a channel receiving os.Interrupt and a stop func.
func notifyInterrupt() (chan os.Signal, func()) {
	ch := make(chan os.Signal, 4)
	signal.Notify(ch, os.Interrupt)
	return ch, func() { signal.Stop(ch) }
}
