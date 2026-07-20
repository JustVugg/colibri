package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// utf8Decoder is an incremental UTF-8 decoder: it holds back an incomplete
// trailing multibyte sequence until the rest of its bytes arrive (the engine can
// split a rune across chunks), matching Python's codecs incremental decoder.
type utf8Decoder struct{ buf []byte }

func (d *utf8Decoder) decode(b []byte) string {
	d.buf = append(d.buf, b...)
	// Emit the longest valid prefix, holding back an incomplete trailing rune.
	// Find the start of the last rune (skip continuation bytes 0b10xxxxxx); if the
	// sequence from there isn't a complete rune, keep it for the next chunk.
	valid := len(d.buf)
	i := valid - 1
	for i >= 0 && d.buf[i]&0xC0 == 0x80 {
		i--
	}
	if i >= 0 && !utf8.FullRune(d.buf[i:valid]) {
		valid = i
	}
	out := string(d.buf[:valid])
	d.buf = append(d.buf[:0], d.buf[valid:]...)
	return out
}

var (
	reLoaded = regexp.MustCompile(`loaded in ([0-9.]+)s \| resident dense: ([0-9.]+) MB`)
	rePath   = regexp.MustCompile(` ?\(?/[^ )]+\)?`)
	reFrom   = regexp.MustCompile(` from$`)
)

func statusPrefixed(line string) bool {
	for _, p := range []string{"[RAM_GB", "[PIN]", "[MTP]", "[USAGE]", "[DSA]", "[KV]"} {
		if strings.HasPrefix(line, p) {
			return true
		}
	}
	return false
}

// wrapText is a minimal word wrapper for status lines (textwrap.wrap parity).
func wrapText(s string, width int) []string {
	if width < 1 {
		width = 1
	}
	words := strings.Fields(s)
	if len(words) == 0 {
		return []string{s}
	}
	var lines []string
	cur := words[0]
	for _, w := range words[1:] {
		if len(cur)+1+len(w) > width {
			lines = append(lines, cur)
			cur = w
		} else {
			cur += " " + w
		}
	}
	return append(lines, cur)
}

// waitDrain waits up to ~1s for the stderr snapshot to complete, matching the
// Python 20×0.05s poll for the drain thread.
func waitDrain(done <-chan struct{}) {
	select {
	case <-done:
	case <-time.After(time.Second):
	}
}

func cmdChat(a *Args) int {
	needModel(a.Model)
	ram := "-"
	if a.Ram != 0 {
		ram = strconv.Itoa(a.Ram)
	}
	topp := "off"
	if a.Topp != 0 {
		topp = ftoa(a.Topp)
	}
	banner(fmt.Sprintf("chat · %s · ram %sGB · topp %s", filepath.Base(a.Model), ram, topp))

	errlog, err := os.CreateTemp("", "coli-*.log")
	if err != nil {
		fatal("cannot create a temp log: " + err.Error())
	}
	defer os.Remove(errlog.Name())

	e := envFor(a)
	e["SERVE"] = "1"
	cmd := exec.Command(GLM, strconv.Itoa(a.Cap))
	cmd.Env = envToSlice(e)
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		fatal("cannot start the engine: " + err.Error())
	}
	ep := &engineProc{cmd: cmd, stdin: stdin, stdout: stdout, stderr: stderr}
	ep.startReader()
	// Snapshot stderr into errlog once it reaches EOF, exactly like the Python
	// original: the engine emits ~400 bytes of load status then (on the real
	// engine) closes stderr, so the read completes and drainDone fires. While the
	// engine keeps stderr open, errlog stays empty and no status is shown — the
	// same observable behaviour as the Python drain thread (which blocks in
	// read() until EOF). A Go goroutine cannot deadlock two child pipes the way
	// CPython does on Windows, so this is a plain read-to-EOF.
	drainDone := make(chan struct{})
	go func() {
		defer close(drainDone)
		if data, _ := io.ReadAll(stderr); len(data) > 0 {
			errlog.Write(data)
			errlog.Sync()
		}
	}()

	sp := NewSpinner("waking the giant (744B)…", nil)
	sp.Start()
	st, turnErr := streamTurn(ep, sentinelREADY, nil, func([]byte) {})
	sp.Stop()
	if turnErr == errEngineDead {
		waitDrain(drainDone)
		if data, err := os.ReadFile(errlog.Name()); err == nil {
			tail := string(data)
			if len(tail) > 1500 {
				tail = tail[len(tail)-1500:]
			}
			fmt.Println(tail)
		}
		engineDiag(ep, "")
		fatal("the engine exited while loading")
	}
	_ = st
	ep.readLine() // TIERS line (web-dashboard protocol); leaks into the answer if unread

	// Up to ~1s for the load-status snapshot (only present if stderr hit EOF).
	waitDrain(drainDone)
	var elog string
	if data, err := os.ReadFile(errlog.Name()); err == nil {
		elog = string(data)
	}
	if m := reLoaded.FindStringSubmatch(elog); m != nil {
		mb, _ := strconv.ParseFloat(m[2], 64)
		fmt.Printf("  %s✓%s ready in %ss %s· resident %.1f GB · RSS %s GB%s\n",
			C.grn, C.r, m[1], C.dim, mb/1000, ftoa(st.Rss), C.r)
	}
	for _, l := range strings.Split(elog, "\n") {
		if statusPrefixed(l) {
			l = rePath.ReplaceAllString(strings.TrimSpace(l), "")
			l = reFrom.ReplaceAllString(l, "")
			chunks := wrapText(l, termWidth()-4)
			for _, chunk := range chunks {
				fmt.Printf("  %s%s%s\n", C.dgray, chunk, C.r)
			}
		}
	}
	fmt.Printf("  %stype and press Enter · Ctrl-C stops the answer · :more continues · :reset clears memory · :q exits%s\n\n", C.dim, C.r)

	sigCh, stopSig := notifyInterrupt()
	defer stopSig()
	reader := bufio.NewReader(os.Stdin)
	w := termWidth() - 4

	exitCode := 0
	prefillTick := func() string {
		data, err := os.ReadFile(errlog.Name())
		if err != nil {
			return ""
		}
		s := string(data)
		if len(s) > 1500 {
			s = s[len(s)-1500:]
		}
		last := ""
		for _, l := range strings.Split(s, "\n") {
			if strings.HasPrefix(l, "[prefill]") {
				last = strings.Replace(l, "[prefill] ", "prefill ", 1)
			}
		}
		return last
	}

replLoop:
	for {
		if TTY {
			fmt.Printf("  %s╭%s╮%s\n", C.dgray, strings.Repeat("─", w), C.r)
			fmt.Printf("  %s│%s %s%s›%s ", C.dgray, C.r, C.teal, C.b, C.r)
		}
		lineCh := make(chan string, 1)
		eofCh := make(chan struct{}, 1)
		go func() {
			l, e := reader.ReadString('\n')
			if e != nil && l == "" {
				eofCh <- struct{}{}
				return
			}
			lineCh <- l
		}()

		var msg string
		select {
		case <-sigCh:
			fmt.Printf("\n  %sinterrupted%s\n", C.dim, C.r)
			break replLoop
		case <-eofCh:
			fmt.Println()
			break replLoop
		case l := <-lineCh:
			msg = strings.TrimSpace(l)
		}
		if TTY {
			fmt.Printf("  %s╰%s╯%s\n", C.dgray, strings.Repeat("─", w), C.r)
		}

		switch msg {
		case ":q", ":quit", "exit":
			break replLoop
		case "":
			continue
		case ":reset":
			ep.writeControl("\x02RESET\n")
			streamTurn(ep, sentinelEND, nil, func([]byte) {})
			fmt.Printf("  %s✦ memory cleared%s\n\n", C.dim, C.r)
			continue
		case ":piu", ":più", ":more", ":continua":
			ep.writeControl("\x02MORE\n")
		default:
			ep.writeControl(strings.ReplaceAll(msg, "\n", " ") + "\n")
		}

		fmt.Printf("\n  %s◆ colibrì%s\n", C.teal, C.r)
		dec := &utf8Decoder{}
		md := NewMDStream("  ")
		raw := os.Getenv("COLI_RAW") == "1"
		sp2 := NewSpinner("thinking…", prefillTick)
		sp2.Start()
		first := true
		echo := func(bs []byte) {
			if first {
				sp2.Stop()
				first = false
				if raw {
					fmt.Print("  ")
				}
			}
			s := dec.decode(bs)
			if s == "" {
				return
			}
			if raw {
				fmt.Print(strings.ReplaceAll(s, "\n", "\n  "))
			} else {
				md.Feed(s)
			}
		}
		t0 := time.Now()
		st, turnErr := streamTurn(ep, sentinelEND, sigCh, echo)
		if !raw {
			md.Close()
		}
		if first {
			sp2.Stop()
		}
		if turnErr == errEngineDead {
			engineDiag(ep, errlog.Name())
			break replLoop
		}
		if turnErr == errInterruptQuit {
			fmt.Printf("\n  %sinterrupted%s\n", C.dim, C.r)
			break replLoop
		}
		el := time.Since(t0).Seconds()
		if st.Tok > 0 {
			fmt.Printf("\r  %s└─ %d tok · %.2f tok/s · hit %.0f%% · RSS %.1f GB · %.0fs%s\n",
				C.dgray, st.Tok, st.Tps, st.Hit, st.Rss, el, C.r)
			if st.Interrupted {
				fmt.Printf("  %s⏹ interrupted; type :more to continue the response%s\n", C.yel, C.r)
			} else if st.Tok >= a.Ngen {
				fmt.Printf("  %s…stopped at --ngen (%d); type :more to continue the response%s\n", C.yel, a.Ngen, C.r)
			}
			fmt.Println()
		} else {
			if st.Interrupted {
				fmt.Printf("  %s⏹ interrupted%s\n", C.yel, C.r)
			}
			fmt.Println()
		}
	}

	ep.stdin.Close()
	_ = ep.cmd.Process.Kill()
	fmt.Printf("  %sgoodbye%s %s— the hummingbird returns to its nest%s 🐦\n\n", C.teal, C.r, C.dim, C.r)
	return exitCode
}
