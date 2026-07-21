package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
)

func TestUtf8Decoder(t *testing.T) {
	// A rune split across chunks must be held back until complete.
	t.Run("ascii", func(t *testing.T) {
		d := &utf8Decoder{}
		if got := d.decode([]byte("abc")); got != "abc" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("two_byte_split", func(t *testing.T) {
		d := &utf8Decoder{}
		// "é" = 0xC3 0xA9
		if got := d.decode([]byte{0xC3}); got != "" {
			t.Errorf("partial should emit nothing, got %q", got)
		}
		if got := d.decode([]byte{0xA9}); got != "é" {
			t.Errorf("completion got %q, want é", got)
		}
	})
	t.Run("emoji_byte_by_byte", func(t *testing.T) {
		d := &utf8Decoder{}
		bird := []byte("🐦") // 4 bytes
		var out strings.Builder
		for _, b := range bird {
			out.WriteString(d.decode([]byte{b}))
		}
		if out.String() != "🐦" {
			t.Errorf("got %q, want 🐦", out.String())
		}
	})
	t.Run("mixed_split", func(t *testing.T) {
		d := &utf8Decoder{}
		got := d.decode([]byte("a")) + d.decode([]byte{0xC3}) + d.decode([]byte{0xA9})
		if got != "aé" {
			t.Errorf("got %q, want aé", got)
		}
	})
}

// captureStdout swaps os.Stdout for a pipe while fn runs (MDStream writes there).
func captureStdout(fn func()) string {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	fn()
	w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	io.Copy(&buf, r)
	return buf.String()
}

func TestMDStream(t *testing.T) {
	// Force color off so assertions are on plain text, independent of TTY.
	savedC := C
	C = palette{}
	defer func() { C = savedC }()

	feed := func(chunks ...string) string {
		return captureStdout(func() {
			m := NewMDStream("  ")
			for _, c := range chunks {
				m.Feed(c)
			}
			m.Close()
		})
	}

	t.Run("plain_line", func(t *testing.T) {
		if got := feed("hello\n"); got != "  hello\n" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("bold_markers_vanish", func(t *testing.T) {
		got := feed("**bold**\n")
		if strings.Contains(got, "*") || !strings.Contains(got, "bold") {
			t.Errorf("got %q, want no asterisks", got)
		}
	})
	t.Run("bold_split_across_chunks", func(t *testing.T) {
		got := feed("**bo", "ld**\n")
		if strings.Contains(got, "*") || !strings.Contains(got, "bold") {
			t.Errorf("got %q", got)
		}
	})
	t.Run("inline_code_markers_vanish", func(t *testing.T) {
		got := feed("`code`\n")
		if strings.Contains(got, "`") || !strings.Contains(got, "code") {
			t.Errorf("got %q", got)
		}
	})
	t.Run("heading_hash_dropped", func(t *testing.T) {
		got := feed("# Title\n")
		if strings.Contains(got, "#") || !strings.Contains(got, "Title") {
			t.Errorf("got %q", got)
		}
	})
	t.Run("bullet_rendered", func(t *testing.T) {
		got := feed("- item\n")
		if !strings.Contains(got, "•") || !strings.Contains(got, "item") {
			t.Errorf("got %q, want bullet", got)
		}
	})
	t.Run("fence_body_preserved", func(t *testing.T) {
		got := feed("```py\n", "x = 1\n", "```\n")
		if strings.Contains(got, "```") || !strings.Contains(got, "x = 1") {
			t.Errorf("got %q", got)
		}
	})
	t.Run("fence_split_backticks", func(t *testing.T) {
		got := feed("``", "`py\n", "y=2\n", "``", "`\n")
		if strings.Contains(got, "```") || !strings.Contains(got, "y=2") {
			t.Errorf("got %q", got)
		}
	})
}
