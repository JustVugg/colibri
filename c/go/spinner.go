package main

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// Spinner mirrors the Python Spinner: a background ticker that overwrites a
// single terminal line. The optional tick callback (~1 Hz) supplies a live
// suffix (e.g. prefill progress read from the engine log). No-op when not a TTY.
type Spinner struct {
	label  string
	tick   func() string
	suffix string
	stop   chan struct{}
	done   chan struct{}
	once   sync.Once
}

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

func NewSpinner(label string, tick func() string) *Spinner {
	return &Spinner{label: label, tick: tick, stop: make(chan struct{}), done: make(chan struct{})}
}

func (s *Spinner) Start() {
	if !TTY {
		close(s.done)
		return
	}
	t0 := time.Now()
	go func() {
		defer close(s.done)
		i := 0
		for {
			select {
			case <-s.stop:
				return
			default:
			}
			el := time.Since(t0).Seconds()
			if s.tick != nil && i%8 == 0 {
				if v := s.tick(); v != "" {
					s.suffix = v
				}
			}
			suf := ""
			if s.suffix != "" {
				suf = fmt.Sprintf(" %s· %s%s", C.dgray, s.suffix, C.r)
			}
			fmt.Fprintf(os.Stdout, "\r  %s%s%s %s%s %.0fs%s%s\033[K",
				C.teal, spinnerFrames[i%10], C.r, C.dim, s.label, el, C.r, suf)
			i++
			time.Sleep(120 * time.Millisecond)
		}
	}()
}

func (s *Spinner) Stop() {
	s.once.Do(func() { close(s.stop) })
	<-s.done
	if TTY {
		fmt.Fprint(os.Stdout, "\r\033[K")
	}
}
