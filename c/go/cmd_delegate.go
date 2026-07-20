package main

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// runChildForwardingSignals runs cmd attached to the parent's stdio and relays
// SIGINT/SIGTERM to it, so the Python gateway shuts down gracefully.
func runChildForwardingSignals(cmd *exec.Cmd) int {
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "coli: "+err.Error())
		return 1
	}
	sigCh := make(chan os.Signal, 8)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)
	go func() {
		first := true
		for range sigCh {
			// openai_server.py only shuts down on its SIGTERM handler (it ignores
			// SIGINT — PEP 475 retries the serve loop). So relay a SIGTERM for the
			// first stop signal — the same clean shutdown the in-process Python
			// `coli serve` performs on Ctrl-C — and hard-kill on a second.
			if first {
				_ = cmd.Process.Signal(syscall.SIGTERM)
				first = false
			} else {
				_ = cmd.Process.Kill()
			}
		}
	}()
	err := cmd.Wait()
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	if err != nil {
		return 1
	}
	return 0
}

// spawnGateway launches the Python OpenAI-compatible gateway (openai_server.py),
// which stays authoritative. The child inherits env_for(a) so it — and the engine
// it spawns — see the same environment the Python coli would have set up.
func spawnGateway(a *Args) int {
	args := []string{
		filepath.Join(SRC, "openai_server.py"),
		"--model", a.Model, "--engine", GLM,
		"--host", a.Host, "--port", strconv.Itoa(a.Port),
		"--model-id", a.ModelID,
		"--cap", strconv.Itoa(a.Cap), "--max-tokens", strconv.Itoa(a.Ngen),
		"--max-queue", strconv.Itoa(a.MaxQueue),
		"--queue-timeout", ftoa(a.QueueTimeout),
		"--kv-slots", strconv.Itoa(a.KvSlots),
	}
	if a.APIKey != "" {
		args = append(args, "--api-key", a.APIKey)
	}
	for _, o := range a.CorsOrigin {
		args = append(args, "--cors-origin", o)
	}
	cmd := exec.Command(PYTHON, args...)
	cmd.Env = envToSlice(envFor(a))
	isolateProcessGroup(cmd)
	return runChildForwardingSignals(cmd)
}

func cmdServe(a *Args) int {
	needModel(a.Model)
	return spawnGateway(a)
}

func cmdWeb(a *Args) int {
	needModel(a.Model)
	dist := filepath.Join(filepath.Dir(SRC), "web", "dist")
	if !exists(filepath.Join(dist, "index.html")) {
		fmt.Printf("%sweb UI not built:%s run  cd web && npm install && npm run build  first;\n", C.yel, C.r)
		fmt.Println("serving the API anyway (the dashboard will 404 until built).")
	}
	url := fmt.Sprintf("http://%s:%d/", a.Host, a.Port)
	if !a.NoBrowser {
		go browserOpener(a.Host, a.Port, url)
	}
	fmt.Printf("dashboard: %s  (opens automatically when the engine is ready)\n", url)
	return spawnGateway(a)
}

// browserOpener polls /health (the 744B engine takes minutes to load) and opens
// the dashboard once it answers. Mirrors the opener thread in cmd_web.
func browserOpener(host string, port int, url string) {
	client := &http.Client{Timeout: 2 * time.Second}
	health := fmt.Sprintf("http://%s:%d/health", host, port)
	for i := 0; i < 600; i++ {
		time.Sleep(2 * time.Second)
		resp, err := client.Get(health)
		if err == nil {
			resp.Body.Close()
			openBrowser(url)
			return
		}
	}
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch {
	case isWindows:
		cmd = exec.Command("cmd", "/c", "start", "", url)
	case fileExistsInPath("xdg-open"):
		cmd = exec.Command("xdg-open", url)
	default:
		cmd = exec.Command("open", url) // macOS
	}
	_ = cmd.Start()
}

func fileExistsInPath(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

func cmdPlan(a *Args) int {
	out, errText, code, _ := runDriver(planDriver, a, nil)
	if code != 0 {
		fatal(fmt.Sprintf("%scannot create resource plan:%s %s", C.yel, C.r, strings.TrimSpace(errText)))
	}
	if a.JSON {
		fmt.Print(out)
		return 0
	}
	banner("plan · Disk / RAM / VRAM")
	for _, line := range strings.Split(strings.TrimRight(out, "\n"), "\n") {
		fmt.Println("  " + line)
	}
	fmt.Println()
	return 0
}

func cmdDoctor(a *Args) int {
	// doctor prints its report to stdout and encodes status in the exit code;
	// stream it straight through.
	payload := driverPayloadJSON(a)
	cmd := exec.Command(PYTHON, "-", payload)
	cmd.Stdin = strings.NewReader(doctorDriver)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "coli: "+err.Error())
		return 1
	}
	return 0
}

func cmdBench(a *Args) int {
	needModel(a.Model)
	banner("bench")
	tasks := "hellaswag,arc_challenge,mmlu"
	if len(a.Tasks) > 0 {
		tasks = strings.Join(a.Tasks, ",")
	}
	taskList := strings.Split(tasks, ",")

	var missing []string
	for _, t := range taskList {
		if !exists(filepath.Join(a.Data, t+".jsonl")) {
			missing = append(missing, t)
		}
	}
	if len(missing) > 0 {
		fmt.Printf("  %sdownloading missing datasets: %s%s\n", C.dim, strings.Join(missing, ", "), C.r)
		lim := a.Limit
		if lim < 200 {
			lim = 200
		}
		runInherit(exec.Command(PYTHON, filepath.Join(TOOLS, "fetch_benchmarks.py"),
			"--out", a.Data, "--tasks", strings.Join(missing, ","), "--limit", strconv.Itoa(lim)))
		var still []string
		for _, t := range taskList {
			if !exists(filepath.Join(a.Data, t+".jsonl")) {
				still = append(still, t)
			}
		}
		if len(still) > 0 {
			var keep []string
			for _, t := range taskList {
				if !contains(still, t) {
					keep = append(keep, t)
				}
			}
			tasks = strings.Join(keep, ",")
			fmt.Printf("  %sskipping (download failed, rerun later): %s%s\n", C.yel, strings.Join(still, ", "), C.r)
			if tasks == "" {
				fmt.Printf("  %sno datasets available — nothing to bench%s\n", C.yel, C.r)
				return 1
			}
		}
	}

	args := []string{filepath.Join(TOOLS, "eval_glm.py"), "--glm", GLM, "--snap", a.Model,
		"--tasks", tasks, "--limit", strconv.Itoa(a.Limit), "--data", a.Data}
	if a.Ram != 0 {
		args = append(args, "--ram", strconv.Itoa(a.Ram))
	}
	fmt.Printf("  %sdecode is disk-bound: this takes HOURS on slow hardware. Raise --limit on faster machines.%s\n\n", C.dim, C.r)
	cmd := exec.Command(PYTHON, args...)
	cmd.Env = envToSlice(envFor(a))
	return runInherit(cmd)
}

func cmdConvert(a *Args) int {
	banner("convert")
	base := []string{filepath.Join(TOOLS, "convert_fp8_to_int4.py"),
		"--repo", a.Repo, "--outdir", a.Model,
		"--ebits", strconv.Itoa(a.Ebits), "--io-bits", strconv.Itoa(a.IoBits)}
	if a.Xbits != 0 {
		base = append(base, "--xbits", strconv.Itoa(a.Xbits))
	}
	// step 1: main model (78 layers). Resumable: restarts from the missing shards.
	fmt.Printf("  %s[1/2] model: %s %s%s\n", C.dim, PYTHON, strings.Join(base, " "), C.r)
	if rc := runInherit(exec.Command(PYTHON, base...)); rc != 0 {
		return rc
	}
	if a.NoMtp {
		return 0
	}
	// step 2: MTP head (layer 78). ALWAYS int8: at int4 the drafts are almost
	// always wrong (issue #8) and speculation never starts.
	mtp := append([]string{}, base...)
	for i, v := range mtp {
		if v == "--ebits" {
			mtp[i+1] = strconv.Itoa(max(8, a.Ebits))
			break
		}
	}
	fmt.Printf("  %s[2/2] int8 MTP head (speculative drafts)%s\n", C.dim, C.r)
	mtp = append(mtp, "--mtp")
	return runInherit(exec.Command(PYTHON, mtp...))
}
