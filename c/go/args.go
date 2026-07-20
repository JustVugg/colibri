package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Args holds every parsed flag/positional, mirroring the single argparse
// Namespace of the Python `coli`. Common flags are accepted both before and
// after the subcommand (parents=[common] in argparse); see parseArgs.
type Args struct {
	Cmd string

	// common
	Model    string
	Ram      int
	AutoTier bool
	Ctx      int
	Gpu      string
	Vram     float64
	Policy   string
	Repin    int
	Cap      int
	Ngen     int
	Topp     float64
	Topk     int
	Temp     float64

	// plan / doctor
	JSON bool

	// run
	Prompt []string

	// serve / web
	Host         string
	Port         int
	ModelID      string
	APIKey       string
	CorsOrigin   stringSlice
	MaxQueue     int
	QueueTimeout float64
	KvSlots      int
	NoBrowser    bool

	// bench
	Tasks []string
	Limit int
	Data  string

	// convert
	Repo   string
	Ebits  int
	IoBits int
	Xbits  int
	NoMtp  bool

	// which flags the user actually supplied (argparse `is not None` semantics)
	provided map[string]bool
}

type stringSlice []string

func (s *stringSlice) String() string { return strings.Join(*s, ",") }
func (s *stringSlice) Set(v string) error {
	*s = append(*s, v)
	return nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOr(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloatOr(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// benchCacheDir mirrors the XDG/LOCALAPPDATA bench cache default in main().
func benchCacheDir() string {
	var root string
	home, _ := os.UserHomeDir()
	if isWindows {
		root = envOr("LOCALAPPDATA", filepath.Join(home, "AppData", "Local"))
	} else {
		root = envOr("XDG_CACHE_HOME", filepath.Join(home, ".cache"))
	}
	return filepath.Join(root, "colibri", "bench")
}

func newArgs() *Args {
	return &Args{
		Model:        envOr("COLI_MODEL", "/home/vincenzo/glm52_i4"),
		Policy:       envOr("COLI_POLICY", "quality"),
		Cap:          8,
		Ngen:         1024,
		Host:         "127.0.0.1",
		Port:         8000,
		ModelID:      envOr("COLI_MODEL_ID", "glm-5.2-colibri"),
		APIKey:       os.Getenv("COLI_API_KEY"),
		MaxQueue:     envIntOr("COLI_MAX_QUEUE", 8),
		QueueTimeout: envFloatOr("COLI_QUEUE_TIMEOUT", 300),
		KvSlots:      envIntOr("COLI_KV_SLOTS", 1),
		Limit:        40,
		Data:         benchCacheDir(),
		Repo:         "zai-org/GLM-5.2-FP8",
		Ebits:        4,
		IoBits:       8,
		provided:     map[string]bool{},
	}
}

const usageDoc = `colibrì — tiny engine, immense model.
Run GLM-5.2 (744B) locally on CPU with roughly 15-26 GB of RAM.

  coli chat                 interactive chat (loads the model once)
  coli serve                OpenAI-compatible HTTP API (persistent engine)
  coli run "prompt"         one-shot generation
  coli info                 model, RAM, disk, and configuration status
  coli plan                 Disk / RAM / VRAM resource plan
  coli doctor               installation and execution-plan diagnostics
  coli bench [task...]      quality benchmarks (MMLU/HellaSwag/...)
  coli convert              convert GLM-5.2-FP8 to int4, one shard at a time
  coli build                build the engine
  coli web                  serve + open the dashboard in a browser

Configuration through environment variables or flags (also valid after the subcommand):
  COLI_MODEL=<dir>   model directory (default /home/vincenzo/glm52_i4)
  --ram N            RAM budget in GB (automatically sizes the expert cache)
  --repin N          adapt RAM/VRAM experts every N tokens
  --topp P           adaptive expert top-p             --topk N   fixed top-k
  --ngen N           maximum response tokens           --cap N    cache slots/layer`

func registerCommon(fs *flag.FlagSet, a *Args) {
	fs.StringVar(&a.Model, "model", a.Model, "model directory")
	fs.IntVar(&a.Ram, "ram", a.Ram, "RAM budget in GB (0=auto)")
	fs.BoolVar(&a.AutoTier, "auto-tier", a.AutoTier, "automatically apply the RAM/VRAM plan")
	fs.IntVar(&a.Ctx, "ctx", a.Ctx, "context length (0=auto)")
	fs.StringVar(&a.Gpu, "gpu", a.Gpu, "auto, none, or a device list such as 0,1")
	fs.Float64Var(&a.Vram, "vram", a.Vram, "total VRAM budget in GB (0=auto)")
	fs.StringVar(&a.Policy, "policy", a.Policy, "resource policy: quality, balanced, or experimental-fast")
	fs.IntVar(&a.Repin, "repin", a.Repin, "adapt RAM/VRAM experts every N tokens")
	fs.IntVar(&a.Cap, "cap", a.Cap, "cache slots/layer")
	fs.IntVar(&a.Ngen, "ngen", a.Ngen, "maximum response tokens")
	fs.Float64Var(&a.Topp, "topp", a.Topp, "adaptive expert top-p")
	fs.IntVar(&a.Topk, "topk", a.Topk, "fixed top-k")
	fs.Float64Var(&a.Temp, "temp", a.Temp, "token temperature (0=greedy)")
}

// registerServe adds the flags shared by `serve` and `web`.
func registerServe(fs *flag.FlagSet, a *Args) {
	fs.StringVar(&a.Host, "host", a.Host, "bind address")
	fs.IntVar(&a.Port, "port", a.Port, "bind port")
	fs.StringVar(&a.ModelID, "model-id", a.ModelID, "advertised model id")
	fs.StringVar(&a.APIKey, "api-key", a.APIKey, "bearer token required from clients")
	fs.Var(&a.CorsOrigin, "cors-origin", "allowed browser origin; repeat as needed ('*' for any)")
	fs.IntVar(&a.MaxQueue, "max-queue", a.MaxQueue, "max queued requests")
	fs.Float64Var(&a.QueueTimeout, "queue-timeout", a.QueueTimeout, "queue wait timeout (seconds)")
	fs.IntVar(&a.KvSlots, "kv-slots", a.KvSlots, "independent KV cache slots")
}

// parseArgs reproduces argparse's parents=[common] behaviour: common flags are
// registered on both a leading global set (so `coli --ram 8 chat` works) and on
// the subcommand set (so `coli chat --ram 8` works). The subcommand set reuses
// the post-global values as its defaults, so a value given after the subcommand
// wins and one given before is preserved.
func parseArgs(argv []string) (*Args, error) {
	a := newArgs()

	global := flag.NewFlagSet("coli", flag.ContinueOnError)
	global.SetOutput(os.Stdout)
	registerCommon(global, a)
	global.Usage = func() {
		fmt.Fprintln(os.Stdout, "coli — colibrì — run GLM-5.2 locally")
		fmt.Fprintln(os.Stdout, "\nGlobal flags (also valid after the subcommand):")
		global.PrintDefaults()
		fmt.Fprintln(os.Stdout, "\n"+usageDoc)
	}
	if err := global.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			os.Exit(0)
		}
		return nil, err
	}
	global.Visit(func(f *flag.Flag) { a.provided[f.Name] = true })

	rest := global.Args()
	if len(rest) == 0 {
		return a, nil // no subcommand: caller prints banner + doc
	}
	a.Cmd = rest[0]

	sub := flag.NewFlagSet(a.Cmd, flag.ContinueOnError)
	sub.SetOutput(os.Stdout)
	registerCommon(sub, a)
	switch a.Cmd {
	case "plan", "doctor":
		sub.BoolVar(&a.JSON, "json", a.JSON, "emit a versioned JSON report")
	case "serve":
		registerServe(sub, a)
	case "web":
		registerServe(sub, a)
		sub.BoolVar(&a.NoBrowser, "no-browser", a.NoBrowser, "don't auto-open the browser")
	case "bench":
		sub.IntVar(&a.Limit, "limit", a.Limit, "questions per task")
		sub.StringVar(&a.Data, "data", a.Data, "dataset cache directory")
	case "convert":
		sub.StringVar(&a.Repo, "repo", a.Repo, "source FP8 repo")
		sub.IntVar(&a.Ebits, "ebits", a.Ebits, "expert bits")
		sub.IntVar(&a.IoBits, "io-bits", a.IoBits, "input/output bits")
		sub.IntVar(&a.Xbits, "xbits", a.Xbits, "extra bits")
		sub.BoolVar(&a.NoMtp, "no-mtp", a.NoMtp, "skip the MTP head (no speculative drafts)")
	case "build", "info", "run", "chat":
		// only common flags (+ positional prompt for run)
	default:
		return nil, fmt.Errorf("unknown command: %s", a.Cmd)
	}
	// argparse allows flags and positionals to be intermixed (e.g. `bench
	// hellaswag --model X`). Go's flag package stops at the first positional, so
	// we parse, collect one positional, and re-parse the remainder until nothing
	// is left.
	var positionals []string
	remaining := rest[1:]
	for len(remaining) > 0 {
		if err := sub.Parse(remaining); err != nil {
			if err == flag.ErrHelp {
				os.Exit(0)
			}
			return nil, err
		}
		after := sub.Args()
		if len(after) == 0 {
			break
		}
		positionals = append(positionals, after[0])
		remaining = after[1:]
	}
	sub.Visit(func(f *flag.Flag) { a.provided[f.Name] = true })

	switch a.Cmd {
	case "run":
		a.Prompt = positionals
	case "bench":
		a.Tasks = positionals
	default:
		if len(positionals) > 0 {
			return nil, fmt.Errorf("unrecognized arguments: %s", strings.Join(positionals, " "))
		}
	}

	// argparse validates `choices` only for values given on the command line, not
	// for a default (e.g. from COLI_POLICY) — mirror that by checking only when
	// --policy was actually provided.
	if a.provided["policy"] {
		switch a.Policy {
		case "quality", "balanced", "experimental-fast":
		default:
			return nil, fmt.Errorf("argument --policy: invalid choice: %q (choose from quality, balanced, experimental-fast)", a.Policy)
		}
	}
	return a, nil
}
