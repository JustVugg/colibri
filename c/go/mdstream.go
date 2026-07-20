package main

import (
	"fmt"
	"os"
	"strings"
	"unicode"
)

// MDStream interprets the answer's markdown as it streams: ``` fences become
// framed boxes, **x** real bold, `x` colored, # headings, - bullets. Markers are
// never shown. It tolerates chunks split mid-marker (hold-back) and dirty output
// (doubled ```). Faithful port of the Python MDStream; operates on runes so
// multibyte UTF-8 in the stream is never sliced mid-character.
type MDStream struct {
	ind        string
	cur        string
	code       bool
	lang       string
	bold       bool
	icode      bool
	justClosed bool
	printed    int // runes of the current line already emitted
}

func NewMDStream(indent string) *MDStream { return &MDStream{ind: indent} }

func (m *MDStream) w(s string) { fmt.Fprint(os.Stdout, s) }

func lstrip(s string) string { return strings.TrimLeftFunc(s, unicode.IsSpace) }

func (m *MDStream) fence(line string) {
	lang := strings.Trim(strings.TrimSpace(strings.TrimSpace(line)[3:]), "`")
	if !m.code {
		if lang == "" && m.justClosed {
			return // orphan ``` after a close: noise, ignore
		}
		m.code = true
		m.lang = lang
		label := lang
		if label == "" {
			label = "code"
		}
		m.w(m.ind + C.dgray + "╭─ " + label + C.r + "\n")
	} else if lang != "" { // ```lang while already in code: close and reopen
		m.w(m.ind + C.dgray + "╰─" + C.r + "\n" + m.ind + C.dgray + "╭─ " + lang + C.r + "\n")
		m.lang = lang
	} else {
		m.code = false
		m.justClosed = true
		m.w(m.ind + C.dgray + "╰─" + C.r + "\n")
	}
}

func (m *MDStream) inline(txt string, out *strings.Builder) {
	r := []rune(txt)
	i := 0
	for i < len(r) {
		ch := r[i]
		if ch == '`' {
			m.icode = !m.icode
			if m.icode {
				out.WriteString(C.org)
			} else {
				out.WriteString(C.r)
			}
			i++
			continue
		}
		if ch == '*' {
			j := i
			for j < len(r) && r[j] == '*' {
				j++
			}
			if j-i >= 2 { // **/***: bold toggles, asterisks vanish
				m.bold = !m.bold
				if m.bold {
					out.WriteString(C.b)
				} else {
					out.WriteString(C.r)
				}
			} else {
				out.WriteRune('*') // lone *: keep it (multiplication etc.)
			}
			i = j
			continue
		}
		out.WriteRune(ch)
		i++
	}
}

func (m *MDStream) line(lineStr string, partial bool) {
	if !partial && strings.HasPrefix(lstrip(lineStr), "```") {
		m.fence(lineStr)
		m.printed = 0
		return
	}
	if strings.TrimSpace(lineStr) != "" {
		m.justClosed = false
	}
	runes := []rune(lineStr)
	seg := string(runes[min(m.printed, len(runes)):]) // emit only the new part
	var out strings.Builder
	if m.code {
		if m.printed == 0 {
			out.WriteString(m.ind + C.dgray + "│" + C.r + " " + C.cyan)
		}
		out.WriteString(seg)
	} else {
		if m.printed == 0 {
			out.WriteString(m.ind)
			st := lstrip(seg)
			switch {
			case strings.HasPrefix(st, "#"): // heading: drop #, bold teal
				seg = strings.TrimSpace(strings.TrimLeft(st, "#"))
				out.WriteString(C.teal + C.b)
				m.bold = true
			case strings.HasPrefix(st, "- ") || strings.HasPrefix(st, "* "): // bullet
				seg = st[2:]
				out.WriteString(C.teal + "•" + C.r + " ")
			}
		}
		m.inline(seg, &out)
	}
	m.w(out.String())
	m.printed = len(runes)
	if !partial { // end of line: reset inline state (robust to orphan markers)
		m.w(C.r + "\n")
		m.bold, m.icode = false, false
		m.printed = 0
	}
}

func (m *MDStream) Feed(s string) {
	m.cur += s
	for {
		idx := strings.IndexByte(m.cur, '\n')
		if idx < 0 {
			break
		}
		lineStr := m.cur[:idx]
		m.cur = m.cur[idx+1:]
		m.line(lineStr, false)
	}
	st := lstrip(m.cur)
	if st != "" && (strings.HasPrefix(st, "```") || (len([]rune(st)) < 3 && strings.HasPrefix("```", st))) {
		return // partial line: possible fence, wait for newline
	}
	if strings.HasPrefix(st, "#") && m.printed == 0 {
		return // heading: render the whole line at newline
	}
	// hold back trailing marker chars that might be split across chunks
	cr := []rune(m.cur)
	hold := 0
	for hold < len(cr) && (cr[len(cr)-1-hold] == '*' || cr[len(cr)-1-hold] == '`') {
		hold++
	}
	safe := m.cur
	if hold > 0 {
		safe = string(cr[:len(cr)-hold])
	}
	if len([]rune(safe)) > m.printed {
		m.line(safe, true)
	}
}

func (m *MDStream) Close() {
	if m.cur != "" {
		m.line(m.cur, false)
		m.cur = ""
	}
	if m.code {
		m.w("\n" + m.ind + C.dgray + "╰─" + C.r)
		m.code = false
	}
	m.w(C.r)
}
