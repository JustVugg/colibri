package main

import "testing"

func TestParseArgs(t *testing.T) {
	cases := []struct {
		name  string
		argv  []string
		check func(t *testing.T, a *Args)
	}{
		{"no_subcommand", []string{}, func(t *testing.T, a *Args) {
			if a.Cmd != "" {
				t.Errorf("Cmd = %q, want empty", a.Cmd)
			}
		}},
		{"global_flag_only", []string{"--ram", "8"}, func(t *testing.T, a *Args) {
			if a.Cmd != "" || a.Ram != 8 {
				t.Errorf("Cmd=%q Ram=%d, want ''/8", a.Cmd, a.Ram)
			}
		}},
		{"bare_chat", []string{"chat"}, func(t *testing.T, a *Args) {
			if a.Cmd != "chat" {
				t.Errorf("Cmd = %q", a.Cmd)
			}
		}},
		{"flag_before_subcommand", []string{"--ram", "8", "chat"}, func(t *testing.T, a *Args) {
			if a.Cmd != "chat" || a.Ram != 8 {
				t.Errorf("Cmd=%q Ram=%d", a.Cmd, a.Ram)
			}
		}},
		{"flag_after_subcommand", []string{"chat", "--ram", "8"}, func(t *testing.T, a *Args) {
			if a.Cmd != "chat" || a.Ram != 8 {
				t.Errorf("Cmd=%q Ram=%d", a.Cmd, a.Ram)
			}
		}},
		{"run_positionals", []string{"run", "hello", "world"}, func(t *testing.T, a *Args) {
			if a.Cmd != "run" || len(a.Prompt) != 2 || a.Prompt[0] != "hello" || a.Prompt[1] != "world" {
				t.Errorf("Cmd=%q Prompt=%v", a.Cmd, a.Prompt)
			}
		}},
		{"bench_intermixed", []string{"bench", "hellaswag", "--limit", "10"}, func(t *testing.T, a *Args) {
			if a.Cmd != "bench" || len(a.Tasks) != 1 || a.Tasks[0] != "hellaswag" || a.Limit != 10 {
				t.Errorf("Cmd=%q Tasks=%v Limit=%d", a.Cmd, a.Tasks, a.Limit)
			}
		}},
		{"serve_port", []string{"serve", "--port", "9000"}, func(t *testing.T, a *Args) {
			if a.Cmd != "serve" || a.Port != 9000 {
				t.Errorf("Cmd=%q Port=%d", a.Cmd, a.Port)
			}
		}},
		{"convert_ebits", []string{"convert", "--ebits", "8"}, func(t *testing.T, a *Args) {
			if a.Cmd != "convert" || a.Ebits != 8 {
				t.Errorf("Cmd=%q Ebits=%d", a.Cmd, a.Ebits)
			}
		}},
		{"policy_valid", []string{"chat", "--policy", "balanced"}, func(t *testing.T, a *Args) {
			if a.Policy != "balanced" || !a.provided["policy"] {
				t.Errorf("Policy=%q provided=%v", a.Policy, a.provided["policy"])
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a, err := parseArgs(tc.argv)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tc.check(t, a)
		})
	}
}

func TestParseArgsErrors(t *testing.T) {
	cases := []struct {
		name string
		argv []string
	}{
		{"invalid_policy", []string{"chat", "--policy", "bogus"}},
		{"unknown_command", []string{"frobnicate"}},
		{"unexpected_positional", []string{"info", "extra"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseArgs(tc.argv); err == nil {
				t.Errorf("parseArgs(%v) = nil error, want error", tc.argv)
			}
		})
	}
}
