package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// C holds the ANSI palette, mirroring the Python `C` class. Fields are cleared
// (set to "") when output is not a TTY, so all the format strings that embed
// them collapse to plain text — matching the Python `C.off()` behaviour.
type palette struct {
	teal, cyan, mag, org, grn, yel string
	dim, b, r, gray, dgray         string
}

func c256(n int) string { return "\033[38;5;" + strconv.Itoa(n) + "m" }

var C = palette{
	teal: c256(37), cyan: c256(80), mag: c256(170), org: c256(208),
	grn: c256(78), yel: c256(179),
	dim: "\033[2m", b: "\033[1m", r: "\033[0m", gray: c256(242), dgray: c256(238),
}

// TTY reports whether colored/interactive output should be produced. Matches
// `sys.stdout.isatty() or os.environ.get("COLI_COLOR")=="1"`.
var TTY = detectTTY()

func detectTTY() bool {
	if os.Getenv("COLI_COLOR") == "1" {
		return true
	}
	fi, err := os.Stdout.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

func init() {
	if !TTY {
		C = palette{} // all fields empty: color off
	}
}

// termWidth mirrors term_w() = min(columns, 100). It follows the same precedence
// as Python's shutil.get_terminal_size: honour $COLUMNS when set and valid, else
// query the controlling terminal (terminalCols, per-OS: TIOCGWINSZ on Unix,
// GetConsoleScreenBufferInfo on Windows), else fall back to 80.
func termWidth() int {
	cols := 0
	if v := os.Getenv("COLUMNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cols = n
		}
	}
	if cols == 0 {
		cols = terminalCols()
	}
	if cols <= 0 {
		cols = 80
	}
	if cols > 100 {
		cols = 100
	}
	return cols
}

func hline(w int) string { return C.dgray + strings.Repeat("─", w) + C.r }

// ---- colibrì 8-bit sprite (2 vertical pixels per character) ----
var sprite = []string{
	"....MMM.........",
	"...MMMMM..w.....",
	"....MMMM.ww.....",
	"OOOOTTeTCC......",
	"....TTTTTCC.....",
	".....TTTTCC.....",
	"......TTCC......",
	".......TC.......",
	"........C.......",
	"................",
}

var spritePal = map[byte]int{'M': 170, 'T': 37, 'C': 80, 'O': 208, 'e': 231, 'w': 80}

func spriteLines() []string {
	if !TTY {
		return []string{"  (\\   ", "   )·>  ", "  / \\   ", "        ", "        "}
	}
	pal := func(ch byte) (int, bool) { v, ok := spritePal[ch]; return v, ok }
	var out []string
	for y := 0; y < len(sprite); y += 2 {
		top := sprite[y]
		var bot string
		if y+1 < len(sprite) {
			bot = sprite[y+1]
		} else {
			bot = strings.Repeat(".", len(top))
		}
		var row strings.Builder
		for x := 0; x < len(top); x++ {
			ct, okt := pal(top[x])
			cb, okb := pal(bot[x])
			switch {
			case !okt && !okb:
				row.WriteString("\033[0m ")
			case okt && !okb:
				fmt.Fprintf(&row, "\033[38;5;%dm\033[49m▀", ct)
			case !okt && okb:
				fmt.Fprintf(&row, "\033[38;5;%dm\033[49m▄", cb)
			default:
				fmt.Fprintf(&row, "\033[38;5;%dm\033[48;5;%dm▀", ct, cb)
			}
		}
		out = append(out, row.String()+"\033[0m")
	}
	return out
}

func banner(sub string) {
	sp := spriteLines()
	txt := []string{
		C.teal + C.b + "colibrì" + C.r + " " + C.dim + "v1.0" + C.r,
		C.dim + "tiny engine, immense model" + C.r,
		C.gray + "GLM-5.2 · 744B MoE · int4 · streaming CPU" + C.r,
		"",
		"",
	}
	if sub != "" {
		txt[3] = C.dgray + sub + C.r
	}
	fmt.Println()
	for i, s := range sp {
		t := ""
		if i < len(txt) {
			t = txt[i]
		}
		fmt.Printf("  %s   %s\n", s, t)
	}
	fmt.Printf("  %s\n", hline(58))
}
