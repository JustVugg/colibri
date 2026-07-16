#!/usr/bin/env python3
"""
diag_harness.py — Comprehensive model diagnostic harness for the colibri GLM-5.2 engine.

Runs a full campaign of tests against any model snapshot:
  Phase 0 (system):     startup telemetry — GPU, RAM, cache cap, load time, MTP, idot kernel
  Phase 1 (smoke):      correctness — curated prompts, coherence checks, corruption detection
  Phase 2 (diagnostic): deep x-ray — full PROFILE breakdown, routing, MTP, disk-split, CUDA tier
  Phase 3 (quality):    benchmark accuracy — hellaswag/arc_challenge/mmlu via eval_glm.py SCORE
  Phase 4 (throughput): tok/s with and without MTP speculation
  Phase 5 (report):     structured JSON + human-readable Markdown summary

Usage:
  python tools/diag_harness.py --snap /path/to/model --phase all
  python tools/diag_harness.py --snap /path/to/model --phase smoke --ngen 64
  python tools/diag_harness.py --snap /path/to/model --phase quality --quality-limit 200

Output goes to --out (default ./diag_results/<timestamp>/).  Each phase writes a raw log
(<phase>_<run>.log) and all metrics are collected into report.json + report.md.

Design notes:
  - stdout and stderr are captured separately via subprocess.PIPE.  The engine streams
    generated text to stdout (interleaved with prompt + PROFILE stats), but TOKENS=1 dumps
    clean token-id lists to stderr — that is the primary text-capture path.
  - Every regex is anchored to the exact printf format strings in glm.c (verified against
    profile_print line 3853, run_text line 3948, the banner line 5299, etc.).
  - Subprocess calls have a hard timeout (default 600s) and are killed cleanly on expiry.
  - A single phase can be run standalone; results accumulate in the output dir.
"""
import os, sys, re, json, time, argparse, subprocess, signal, traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# PROMPT SUITE — curated across categories.  "expect" is a case-insensitive
# substring checked against the generated text (None = coherence-only check).
# ---------------------------------------------------------------------------
PROMPTS = [
    {"id":"fact_capital",  "cat":"factual",    "prompt":"The capital of France is",
     "expect":"Paris",  "note":"basic world knowledge"},
    {"id":"fact_boiling",  "cat":"factual",    "prompt":"What is the boiling point of water in Celsius?",
     "expect":"100",    "note":"basic science"},
    {"id":"fact_planet",   "cat":"factual",    "prompt":"What planet is closest to the Sun?",
     "expect":"Mercury","note":"astronomy fact"},
    {"id":"math_mult",     "cat":"math",       "prompt":"What is 15 times 12?",
     "expect":"180",    "note":"2-digit multiplication"},
    {"id":"math_add",      "cat":"math",       "prompt":"What is 847 plus 153?",
     "expect":"1000",   "note":"3-digit addition"},
    {"id":"reason_train",  "cat":"reasoning",  "prompt":"If a train travels 60 mph for 2.5 hours, how far does it go?",
     "expect":"150",    "note":"rate-time-distance"},
    {"id":"code_factorial","cat":"code",       "prompt":"Write a Python function that computes the factorial of a number.",
     "expect":"def",    "note":"code generation"},
    {"id":"explain_nn",    "cat":"explanation","prompt":"Explain what a neural network is in one sentence.",
     "expect":None,     "note":"coherence check"},
    {"id":"creative_story","cat":"creative",   "prompt":"Write a one-sentence story about a lighthouse.",
     "expect":None,     "note":"creative coherence"},
    {"id":"edge_hello",    "cat":"edge",       "prompt":"Hello, how are you today?",
     "expect":None,     "note":"conversational opener"},
    {"id":"edge_the",      "cat":"edge",       "prompt":"The weather today is",
     "expect":None,     "note":"simple continuation"},
    {"id":"edge_repeat",   "cat":"edge",       "prompt":"The quick brown fox jumps over the lazy dog. The quick brown fox",
     "expect":None,     "note":"repetition-bait"},
]

# ---------------------------------------------------------------------------
# METRIC EXTRACTION — each parser is (name, compiled_regex, group_index, cast).
# Regexes match the EXACT printf format strings in glm.c.
# ---------------------------------------------------------------------------
def _f1(g): return float(g)
def _i1(g): return int(g)

# Build parsers as (regex, lambda(match)->value) for clarity
RX = {
    # stdout: run_text summary line (line 3948)
    "decode_toks":   (re.compile(r"decode (\d+) tokens in"),              lambda m: int(m.group(1))),
    "decode_secs":   (re.compile(r"decode \d+ tokens in ([\d.]+)s"),      lambda m: float(m.group(1))),
    "decode_tps":    (re.compile(r"decode \d+ tokens in [\d.]+s \(([\d.]+) tok/s\)"), lambda m: float(m.group(1))),
    "prefill_toks":  (re.compile(r"prefill (\d+) tokens in"),             lambda m: int(m.group(1))),
    "prefill_secs":  (re.compile(r"prefill \d+ tokens in ([\d.]+)s"),     lambda m: float(m.group(1))),
    "hit_rate":      (re.compile(r"expert hit rate ([\d.]+)%"),           lambda m: float(m.group(1))),
    "rss_gb":        (re.compile(r"RSS ([\d.]+) GB"),                     lambda m: float(m.group(1))),
    "experts_per_tok":(re.compile(r"experts loaded/token: ([\d.]+)"),     lambda m: float(m.group(1))),
    "mtp_accept":    (re.compile(r"MTP acceptance ([\d.]+)%"),            lambda m: float(m.group(1))),
    "mtp_acc_cnt":   (re.compile(r"MTP acceptance \d+% \((\d+)/(\d+)\)"), lambda m: (int(m.group(1)), int(m.group(2)))),
    "spec_tok_per_fw":(re.compile(r"speculation: ([\d.]+) tokens/forward"),lambda m: float(m.group(1))),
    # stdout: PROFILE lines (profile_print, line 3853-3864)
    "prof_expert_disk":  (re.compile(r"expert-disk ([\d.]+)s service"),   lambda m: float(m.group(1))),
    "prof_expert_wait":  (re.compile(r"service / ([\d.]+)s wait"),        lambda m: float(m.group(1))),
    "prof_expert_mm":    (re.compile(r"expert-matmul ([\d.]+)s"),         lambda m: float(m.group(1))),
    "prof_attention":    (re.compile(r"\| attention ([\d.]+)s"),          lambda m: float(m.group(1))),
    "prof_kvb":          (re.compile(r"including kvb ([\d.]+)s"),         lambda m: float(m.group(1))),
    "prof_lm_head":      (re.compile(r"lm_head ([\d.]+)s"),               lambda m: float(m.group(1))),
    "prof_other":        (re.compile(r"\| other ([\d.]+)s"),              lambda m: float(m.group(1))),
    # stdout: banner (line 5299)
    "load_secs":     (re.compile(r"loaded in ([\d.]+)s"),                 lambda m: float(m.group(1))),
    "resident_mb":   (re.compile(r"resident dense: ([\d.]+) MB"),         lambda m: float(m.group(1))),
    "mtp_status":    (re.compile(r"MTP (ACTIVE|absent)"),                 lambda m: m.group(1)),
    "n_layers":      (re.compile(r"layers=(\d+) experts=(\d+)"),          lambda m: int(m.group(1))),
    "n_experts":     (re.compile(r"layers=(\d+) experts=(\d+)"),          lambda m: int(m.group(2))),
    # stdout: banner idot kernel
    "idot_kernel":   (re.compile(r"idot: (\S+) =="),                      lambda m: m.group(1)),
    "cache_cap":     (re.compile(r"cache=(\d+) experts/layer"),           lambda m: int(m.group(1))),
    # stderr: RAM_GB (line 5086-5112)
    "ram_budget":    (re.compile(r"\[RAM_GB=([\d.]+)"),                   lambda m: float(m.group(1))),
    "cap_lowered":   (re.compile(r"cap lowered (\d+)->(\d+)"),            lambda m: (int(m.group(1)), int(m.group(2)))),
    "cap_raised":    (re.compile(r"cap raised (\d+)->(\d+)"),             lambda m: (int(m.group(1)), int(m.group(2)))),
    "cap_ok":        (re.compile(r"cap=(\d+) ok"),                        lambda m: int(m.group(1))),
    # stderr: CUDA (backend_cuda.cu:389)
    "cuda_device":   (re.compile(r"\[CUDA\] device (\d+): (.*?), ([\d.]+) GB VRAM, sm_(\d)(\d)"),
                      lambda m: {"id":int(m.group(1)),"name":m.group(2).strip(),"vram_gb":float(m.group(3)),
                                 "sm":f"{m.group(4)}.{m.group(5)}"}),
    "cuda_mode":     (re.compile(r"\[CUDA\] mode: (.+)"),                 lambda m: m.group(1)),
    "cuda_tier":     (re.compile(r"CUDA expert tier: (\d+) resident experts \(([\d.]+) GB\)"),
                      lambda m: {"resident":int(m.group(1)),"vram_gb":float(m.group(2))}),
    # stderr: TOKENS dump (line 4010-4012)
    "tokens_dump":   (re.compile(r"^\[TOKENS\] (\d+) generated:(.*)$", re.MULTILINE),
                      lambda m: [int(x) for x in m.group(2).split()]),
    # stderr: per-16-token progress (emit_stream line 3742)
    "progress_tps":  (re.compile(r"t=(\d+)\s+RSS ([\d.]+) GB\s+hit ([\d.]+)%\s+([\d.]+) tok/s\s+([\d.]+) tok/fw"),
                      lambda m: {"tok":int(m.group(1)),"rss":float(m.group(2)),"hit":float(m.group(3)),
                                 "tps":float(m.group(4)),"tpf":float(m.group(5))}),
    # stderr: DSA, USAGE, KV startup lines
    "usage_loaded":  (re.compile(r"\[USAGE\].*?(\d+) selections"),        lambda m: int(m.group(1))),
    "kv_slots":      (re.compile(r"\[KV\].*?(\d+) context slots"),        lambda m: int(m.group(1))),
}

def extract_metrics(stdout: str, stderr: str) -> dict:
    """Parse all metrics from the engine's stdout+stderr output."""
    text_out = stdout or ""
    text_err = stderr or ""
    metrics = {}
    # For most parsers we search BOTH streams (engine is inconsistent about which
    # channel a given line lands on).  Some are stream-specific (noted below).
    combined = text_out + "\n" + text_err
    for name, (rx, fn) in RX.items():
        m = rx.search(combined)
        if m:
            try: metrics[name] = fn(m)
            except (ValueError, IndexError): pass
    # TOKENS dump is stderr-only and may appear once; grab it explicitly
    m = RX["tokens_dump"][0].search(text_err)
    if m:
        try: metrics["tokens_dump"] = RX["tokens_dump"][1](m)
        except: pass
    # Multiple CUDA devices — collect all
    cuda_devs = []
    for m in re.finditer(r"\[CUDA\] device (\d+): (.*?), ([\d.]+) GB VRAM, sm_(\d)(\d)", text_err):
        cuda_devs.append({"id":int(m.group(1)),"name":m.group(2).strip(),
                          "vram_gb":float(m.group(3)),"sm":f"{m.group(4)}.{m.group(5)}"})
    if cuda_devs: metrics["cuda_devices"] = cuda_devs
    # Multiple progress checkpoints — collect the full curve
    progress = []
    for m in RX["progress_tps"][0].finditer(text_err):
        try:
            progress.append(RX["progress_tps"][1](m))
        except: pass
    if progress: metrics["progress_curve"] = progress
    # Multiple PROFILE lines — there are two (prefill + decode).  Keep both.
    prof_lines = [(i, line) for i, line in enumerate(text_out.splitlines()) if line.startswith("PROFILE:")]
    for idx, (line_no, line) in enumerate(prof_lines):
        label = "prefill_profile" if idx == 0 else "decode_profile"
        p = {}
        for pname, (rx, fn) in RX.items():
            if not pname.startswith("prof_"): continue
            mm = rx.search(line)
            if mm:
                try: p[pname] = fn(mm)
                except: pass
        if p: metrics[label] = p
    return metrics


# ---------------------------------------------------------------------------
# ENGINE RUNNER — subprocess wrapper with timeout, signal handling, logging.
# ---------------------------------------------------------------------------
class EngineRunner:
    def __init__(self, glm_path, snap, out_dir, default_env=None, timeout=600):
        self.glm = str(glm_path)
        self.snap = str(snap)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.default_env = default_env or {}
        self.timeout = timeout
        self.default_cap = 75  # production default (matches bench_full.sh)

    def run_prompt(self, prompt, ngen=64, env_extra=None, log_name=None, cap=None):
        """Run the engine in PROMPT mode and return (stdout, stderr, returncode, elapsed)."""
        env = dict(os.environ, SNAP=self.snap, PROMPT=prompt, NGEN=str(ngen))
        env.update(self.default_env)
        if env_extra: env.update(env_extra)
        env["TOKENS"] = "1"   # always capture token ids for reliable text extraction
        cmd = [self.glm, str(cap if cap is not None else self.default_cap)]
        return self._exec(cmd, env, log_name)

    def run_score(self, score_file, cap=None, env_extra=None, log_name=None):
        """Run the engine in SCORE (log-likelihood) mode."""
        env = dict(os.environ, SNAP=self.snap, SCORE=score_file)
        env.update(self.default_env)
        if env_extra: env.update(env_extra)
        cmd = [self.glm, str(cap if cap is not None else self.default_cap)]
        return self._exec(cmd, env, log_name)

    def _exec(self, cmd, env, log_name):
        t0 = time.time()
        log_path = self.out_dir / (log_name or f"run_{int(t0)}.log")
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8", errors="replace")
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                rc = -1
                stderr = (stderr or "") + f"\n[TIMEOUT after {self.timeout}s]\n"
        except Exception as e:
            stdout, stderr, rc = "", f"[EXCEPTION] {e}\n{traceback.format_exc()}", -2
        elapsed = time.time() - t0
        # Write raw log (both streams, clearly delimited)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== CMD: {' '.join(cmd)}\n=== ELAPSED: {elapsed:.1f}s\n=== RC: {rc}\n\n")
            f.write("--- STDOUT ---\n"); f.write(stdout or ""); f.write("\n")
            f.write("--- STDERR ---\n"); f.write(stderr or ""); f.write("\n")
        return stdout or "", stderr or "", rc, elapsed


# ---------------------------------------------------------------------------
# TEXT RECOVERY — decode the [TOKENS] ids back to text using the tokenizer.
# Falls back to stdout extraction if tokenizer/tokens unavailable.
# ---------------------------------------------------------------------------
def recover_text(stdout, stderr, prompt, tokenizer=None):
    """Extract generated text.  Primary: TOKENS dump decoded.  Fallback: stdout parse."""
    # Method 1: decode TOKENS ids via tokenizer
    if tokenizer:
        m = re.search(r"^\[TOKENS\] \d+ generated:(.*)$", stderr, re.MULTILINE)
        if m:
            ids = [int(x) for x in m.group(1).split()]
            if ids:
                try:
                    return tokenizer.decode(ids), "tokens"
                except Exception: pass
    # Method 2: stdout — text is between the prompt string and "PROFILO" or "\n---"
    text = stdout or ""
    # Find the prompt in stdout, take everything after it up to PROFILO
    idx = text.find(prompt)
    if idx >= 0:
        after = text[idx + len(prompt):]
        cut = after.find("PROFILO")
        if cut >= 0:
            return after[:cut].strip(), "stdout"
        cut = after.find("\n---")
        if cut >= 0:
            return after[:cut].strip(), "stdout"
    return "", "none"


# ---------------------------------------------------------------------------
# CORRUPTION / COHERENCE CHECKS
# ---------------------------------------------------------------------------
def check_repetition(token_ids):
    """Detect degenerate repetition: same token 3+ consecutive times."""
    if not token_ids or len(token_ids) < 6:
        return False, 0
    max_run = 1; cur = 1
    for i in range(1, len(token_ids)):
        if token_ids[i] == token_ids[i-1]: cur += 1; max_run = max(max_run, cur)
        else: cur = 1
    return max_run >= 3, max_run

def check_expected(text, expect):
    """Check if expected substring appears case-insensitively in the first 200 chars."""
    if not expect or not text: return None
    return expect.lower() in text[:200].lower()


# ---------------------------------------------------------------------------
# PHASES
# ---------------------------------------------------------------------------
class DiagnosticHarness:
    def __init__(self, args):
        self.args = args
        self.snap = args.snap
        self.glm = args.glm
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(args.out or f"./diag_results/{ts}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Base env for all runs
        base_env = {"TEMP": "0", "PIPE": "1", "PIPE_WORKERS": "8", "DIRECT": "1"}
        if args.ram: base_env["RAM_GB"] = str(args.ram)
        if args.cuda:
            base_env.update({
                "COLI_CUDA": "1",
                "COLI_GPU": str(args.gpu) if args.gpu is not None else "0",
                "CUDA_DENSE": "1",
                "COLI_CUDA_ATTN": "1",
                "COLI_CUDA_PIPE": "2",
                "COLI_CUDA_PIPE_S_MIN": "1",
            })
        self.runner = EngineRunner(self.glm, self.snap, self.out_dir, base_env, timeout=args.timeout)
        self.runner.default_cap = args.cap
        # Load tokenizer for text recovery
        self.tokenizer = None
        tok_path = os.path.join(self.snap, "tokenizer.json")
        if os.path.exists(tok_path):
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(tok_path)
            except Exception as e:
                print(f"[warn] could not load tokenizer: {e}", file=sys.stderr)
        self.results = {"meta": {"snap": self.snap, "glm": self.glm,
                                 "timestamp": ts, "args": vars(args)},
                        "phases": {}}

    def phase_system(self):
        """Phase 0: minimal run to capture startup telemetry."""
        print("\n" + "="*60 + "\nPHASE 0: SYSTEM PROBE\n" + "="*60)
        stdout, stderr, rc, elapsed = self.runner.run_prompt(
            "Hello", ngen=1, log_name="system_probe.log")
        metrics = extract_metrics(stdout, stderr)
        result = {
            "rc": rc, "elapsed": elapsed,
            "load_secs": metrics.get("load_secs"),
            "resident_mb": metrics.get("resident_mb"),
            "n_layers": metrics.get("n_layers"),
            "n_experts": metrics.get("n_experts"),
            "mtp_status": metrics.get("mtp_status"),
            "idot_kernel": metrics.get("idot_kernel"),
            "cache_cap_requested": metrics.get("cache_cap"),
            "ram_budget_gb": metrics.get("ram_budget"),
            "cap_lowered": metrics.get("cap_lowered"),
            "cap_raised": metrics.get("cap_raised"),
            "cap_final": metrics.get("cap_ok"),
            "cuda_devices": metrics.get("cuda_devices", []),
            "cuda_mode": metrics.get("cuda_mode"),
        }
        # Print summary
        print(f"  load time:    {result['load_secs']:.2f}s" if result['load_secs'] else "  load time:    FAILED")
        print(f"  resident:     {result['resident_mb']:.0f} MB" if result['resident_mb'] else "")
        print(f"  layers/exp:   {result['n_layers']}/{result['n_experts']}" if result['n_layers'] else "")
        print(f"  MTP:          {result['mtp_status']}")
        print(f"  idot kernel:  {result['idot_kernel']}")
        print(f"  RAM budget:   {result['ram_budget_gb']} GB" if result['ram_budget_gb'] else "  RAM budget:   auto")
        if result['cap_lowered']:
            print(f"  cache cap:    {result['cap_lowered'][0]} -> {result['cap_lowered'][1]} (RAM-lowered)")
        elif result['cap_final']:
            print(f"  cache cap:    {result['cap_final']} (ok)")
        else:
            print(f"  cache cap:    {result['cache_cap_requested']}")
        for d in result['cuda_devices']:
            print(f"  GPU {d['id']}:       {d['name']}, {d['vram_gb']:.1f} GB, sm_{d['sm']}")
        if rc != 0 and rc != -1:
            print(f"  WARNING: engine returned rc={rc}")
        self.results["phases"]["system"] = result
        return result

    def phase_smoke(self):
        """Phase 1: correctness smoke test across curated prompts."""
        print("\n" + "="*60 + "\nPHASE 1: CORRECTNESS SMOKE TEST\n" + "="*60)
        ngen = self.args.ngen
        prompt_results = []
        pass_count = 0; total = len(PROMPTS)
        for p in PROMPTS:
            stdout, stderr, rc, elapsed = self.runner.run_prompt(
                p["prompt"], ngen=ngen, log_name=f"smoke_{p['id']}.log")
            metrics = extract_metrics(stdout, stderr)
            token_ids = metrics.get("tokens_dump", [])
            text, method = recover_text(stdout, stderr, p["prompt"], self.tokenizer)
            is_rep, max_run = check_repetition(token_ids)
            expect_ok = check_expected(text, p.get("expect"))
            # Verdict: pass if text is non-empty, no severe repetition, and (if factual) expected found
            has_text = bool(text.strip())
            ok = has_text and not is_rep
            if p.get("expect") and expect_ok is False: ok = False
            if ok: pass_count += 1
            # Diagnose failure mode for reporting
            if not has_text:
                fail_reason = "no tokens generated (immediate EOS)"
            elif is_rep:
                fail_reason = f"repetition loop (max run={max_run})"
            elif p.get("expect") and expect_ok is False:
                fail_reason = f"expected '{p['expect']}' not found"
            else:
                fail_reason = ""
            prompt_results.append({
                "id": p["id"], "cat": p["cat"], "prompt": p["prompt"],
                "expect": p.get("expect"), "generated": text[:300],
                "expect_match": expect_ok, "repetition": is_rep, "max_run": max_run,
                "toks_generated": len(token_ids), "decode_tps": metrics.get("decode_tps"),
                "hit_rate": metrics.get("hit_rate"), "rc": rc, "elapsed": elapsed,
                "text_method": method, "pass": ok, "fail_reason": fail_reason,
            })
            status = "PASS" if ok else "FAIL"
            extra = f" expect={'Y' if expect_ok else ('N' if expect_ok is False else '-')}" if p.get("expect") else ""
            tps = f" {metrics.get('decode_tps',0):.2f}t/s" if metrics.get("decode_tps") else ""
            print(f"  [{status}] {p['id']:<22} ({p['cat']:<11}) rep={'Y' if is_rep else 'N'}{extra}{tps}")
            if text:
                preview = text.replace("\n", " ")[:80]
                print(f"         -> {preview}")
            elif rc != 0:
                print(f"         -> [engine rc={rc}]")
            else:
                print(f"         -> [{fail_reason}]")
        result = {"prompts": prompt_results, "pass": pass_count, "total": total,
                  "pass_rate": 100.0 * pass_count / total if total else 0}
        print(f"\n  SMOKE SUMMARY: {pass_count}/{total} passed ({result['pass_rate']:.0f}%)")
        self.results["phases"]["smoke"] = result
        return result

    def phase_diagnostic(self):
        """Phase 2: single deep-instrumented run — full PROFILE + routing + MTP."""
        print("\n" + "="*60 + "\nPHASE 2: FULL SYSTEM DIAGNOSTIC\n" + "="*60)
        diag_env = {
            "LOOKA": "1", "DISK_SPLIT": "1", "ROUTE_AGREE": "1",
        }
        if self.args.cuda:
            diag_env["COLI_CUDA_PROFILE"] = "1"
        prompt = ("Write a short paragraph explaining how photosynthesis works. "
                  "Include the roles of sunlight, water, and carbon dioxide.")
        stdout, stderr, rc, elapsed = self.runner.run_prompt(
            prompt, ngen=self.args.ngen, env_extra=diag_env, log_name="diagnostic.log")
        metrics = extract_metrics(stdout, stderr)
        text, _ = recover_text(stdout, stderr, prompt, self.tokenizer)
        result = {
            "rc": rc, "elapsed": elapsed,
            "generated_text": text[:500],
            "prefill_profile": metrics.get("prefill_profile", {}),
            "decode_profile": metrics.get("decode_profile", {}),
            "decode_tps": metrics.get("decode_tps"),
            "prefill_secs": metrics.get("prefill_secs"),
            "hit_rate": metrics.get("hit_rate"),
            "rss_gb": metrics.get("rss_gb"),
            "experts_per_tok": metrics.get("experts_per_tok"),
            "mtp_accept": metrics.get("mtp_accept"),
            "mtp_counts": metrics.get("mtp_acc_cnt"),
            "spec_tok_per_fw": metrics.get("spec_tok_per_fw"),
            "cuda_tier": metrics.get("cuda_tier"),
            "progress_curve": metrics.get("progress_curve", []),
        }
        # Print the PROFILE breakdown
        dp = result["decode_profile"]
        print(f"\n  DECODE TIMING BREAKDOWN (per-bucket, seconds):")
        print(f"    expert-disk:    {dp.get('prof_expert_disk', '?'):>8}")
        print(f"    expert-matmul:  {dp.get('prof_expert_mm', '?'):>8}")
        print(f"    attention:      {dp.get('prof_attention', '?'):>8}")
        print(f"    lm_head:        {dp.get('prof_lm_head', '?'):>8}")
        print(f"    other:          {dp.get('prof_other', '?'):>8}")
        print(f"\n  PERFORMANCE:")
        print(f"    prefill:   {result['prefill_secs']:.2f}s" if result['prefill_secs'] else "    prefill:   ?")
        print(f"    decode:    {result['decode_tps']:.3f} tok/s" if result['decode_tps'] else "    decode:    ?")
        print(f"    hit rate:  {result['hit_rate']:.1f}%" if result.get('hit_rate') is not None else "    hit rate:  ?")
        print(f"    RSS:       {result['rss_gb']:.1f} GB" if result.get('rss_gb') else "    RSS:       ?")
        print(f"    exp/tok:   {result['experts_per_tok']:.1f}" if result.get('experts_per_tok') else "    exp/tok:   ?")
        print(f"    MTP:       {result['mtp_accept']:.0f}% accept" if result.get('mtp_accept') is not None else "    MTP:       ?")
        if result.get('cuda_tier'):
            ct = result['cuda_tier']
            print(f"    CUDA tier: {ct['resident']} experts, {ct['vram_gb']:.1f} GB VRAM")
        # Show generated text preview
        if text:
            print(f"\n  GENERATED TEXT (first 200 chars):")
            print(f"    {text[:200].replace(chr(10), ' ')}")
        else:
            print(f"\n  GENERATED TEXT: [none recovered]")
        self.results["phases"]["diagnostic"] = result
        return result

    def phase_quality(self):
        """Phase 3: benchmark accuracy via eval_glm.py SCORE mode."""
        print("\n" + "="*60 + "\nPHASE 3: QUALITY BENCHMARKS\n" + "="*60)
        eval_script = os.path.join(os.path.dirname(__file__), "eval_glm.py")
        bench_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench")
        if not os.path.exists(eval_script):
            print(f"  [SKIP] eval_glm.py not found at {eval_script}")
            self.results["phases"]["quality"] = {"error": "eval_glm.py not found"}
            return None
        tasks = ["hellaswag", "arc_challenge", "mmlu"]
        missing = [t for t in tasks if not os.path.exists(os.path.join(bench_dir, f"{t}.jsonl"))]
        if missing:
            print(f"  [SKIP] benchmark data missing: {missing}")
            print(f"         run: python tools/fetch_benchmarks.py --out {bench_dir}")
            self.results["phases"]["quality"] = {"error": f"missing benchmark data: {missing}"}
            return None
        limit = self.args.quality_limit
        py = sys.executable
        cmd = [py, eval_script, "--snap", self.snap, "--glm", self.glm,
               "--data", bench_dir, "--tasks", ",".join(tasks), "--limit", str(limit)]
        env = dict(os.environ)
        if self.args.ram: env["RAM_GB"] = str(self.args.ram)
        print(f"  Running eval_glm.py (tasks={tasks}, n={limit})...")
        print(f"  This takes ~{limit*3*4/0.05:.0f}s at 0.05 tok/s (worst case)...")
        t0 = time.time()
        log_path = self.out_dir / "quality_eval.log"
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=self.args.timeout*3,
                                      encoding="utf-8", errors="replace")
                logf.write(proc.stdout); logf.write("\n--- STDERR ---\n"); logf.write(proc.stderr)
            elapsed = time.time() - t0
            # Parse the acc/acc_norm table from eval_glm.py output
            scores = {}
            for line in proc.stdout.splitlines():
                # lines like: "hellaswag           40   45.0%     50.0%"
                m = re.match(r"(\w+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%", line.strip())
                if m:
                    scores[m.group(1)] = {"n": int(m.group(2)), "acc": float(m.group(3)),
                                          "acc_norm": float(m.group(4))}
            mean_m = re.search(r"MEAN acc_norm:\s*([\d.]+)%", proc.stdout)
            result = {"scores": scores, "mean_acc_norm": float(mean_m.group(1)) if mean_m else None,
                      "elapsed": elapsed, "rc": proc.returncode}
            print(f"  Completed in {elapsed:.0f}s\n")
            print(f"  {'task':<18} {'n':>4} {'acc':>7} {'acc_norm':>9}")
            for t, s in scores.items():
                print(f"  {t:<18} {s['n']:>4} {s['acc']:>6.1f}% {s['acc_norm']:>8.1f}%")
            if result["mean_acc_norm"] is not None:
                print(f"\n  MEAN acc_norm: {result['mean_acc_norm']:.1f}%")
        except subprocess.TimeoutExpired:
            result = {"error": f"eval timed out after {self.args.timeout*3}s"}
            print(f"  [TIMEOUT] eval_glm.py exceeded {self.args.timeout*3}s")
        except Exception as e:
            result = {"error": str(e)}
            print(f"  [ERROR] {e}")
        self.results["phases"]["quality"] = result
        return result

    def phase_throughput(self):
        """Phase 4: tok/s comparison — MTP on vs off."""
        print("\n" + "="*60 + "\nPHASE 4: THROUGHPUT BENCHMARK\n" + "="*60)
        prompt = "Summarize the plot of Romeo and Juliet in three sentences."
        ngen = self.args.ngen
        results = {}
        # Run 1: with MTP (default)
        print(f"  [1/2] MTP ON  (draft=3)...")
        stdout, stderr, rc, t = self.runner.run_prompt(
            prompt, ngen=ngen, log_name="throughput_mtp_on.log")
        m_on = extract_metrics(stdout, stderr)
        results["mtp_on"] = {"tps": m_on.get("decode_tps"), "hit": m_on.get("hit_rate"),
                             "mtp_accept": m_on.get("mtp_accept"), "elapsed": t}
        print(f"       {m_on.get('decode_tps',0):.3f} tok/s | hit {m_on.get('hit_rate',0):.1f}% | "
              f"MTP {m_on.get('mtp_accept',0):.0f}%")
        # Run 2: MTP off
        print(f"  [2/2] MTP OFF (MTP=0)...")
        stdout, stderr, rc, t = self.runner.run_prompt(
            prompt, ngen=ngen, env_extra={"MTP": "0"}, log_name="throughput_mtp_off.log")
        m_off = extract_metrics(stdout, stderr)
        results["mtp_off"] = {"tps": m_off.get("decode_tps"), "hit": m_off.get("hit_rate"),
                              "elapsed": t}
        print(f"       {m_off.get('decode_tps',0):.3f} tok/s | hit {m_off.get('hit_rate',0):.1f}%")
        # Compute speedup
        if results["mtp_on"]["tps"] and results["mtp_off"]["tps"] and results["mtp_off"]["tps"] > 0:
            sp = results["mtp_on"]["tps"] / results["mtp_off"]["tps"]
            results["mtp_speedup"] = sp
            print(f"\n  MTP speedup: {sp:.2f}x ({'MTP helps' if sp > 1.05 else 'MTP hurts' if sp < 0.95 else 'no effect'})")
        else:
            print(f"\n  MTP speedup: (insufficient data)")
        self.results["phases"]["throughput"] = results
        return results

    def write_report(self):
        """Write report.json and report.md."""
        ts = self.results["meta"]["timestamp"]
        json_path = self.out_dir / "report.json"
        md_path = self.out_dir / "report.md"
        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)
        # Markdown
        lines = [
            f"# Diagnostic Report — {ts}",
            f"",
            f"**Model:** `{self.snap}`",
            f"**Engine:** `{self.glm}`",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
        ]
        phases = self.results["phases"]
        # System
        if "system" in phases:
            s = phases["system"]
            lines += ["## Phase 0: System Probe", ""]
            lines += [f"- Load time: **{s.get('load_secs','?')}s**"]
            lines += [f"- Layers/experts: {s.get('n_layers','?')}/{s.get('n_experts','?')}"]
            lines += [f"- MTP: {s.get('mtp_status','?')}"]
            lines += [f"- idot kernel: `{s.get('idot_kernel','?')}`"]
            lines += [f"- RAM budget: {s.get('ram_budget_gb','auto')} GB"]
            if s.get("cap_lowered"):
                lines += [f"- Cache cap: {s['cap_lowered'][0]}→{s['cap_lowered'][1]} (RAM-lowered)"]
            elif s.get("cap_final"):
                lines += [f"- Cache cap: {s['cap_final']}"]
            for d in s.get("cuda_devices", []):
                lines += [f"- GPU {d['id']}: {d['name']}, {d['vram_gb']:.1f} GB, sm_{d['sm']}"]
            lines.append("")
        # Smoke
        if "smoke" in phases:
            sm = phases["smoke"]
            lines += [f"## Phase 1: Correctness Smoke", "",
                      f"**{sm['pass']}/{sm['total']} prompts passed ({sm['pass_rate']:.0f}%)**", "",
                      "| ID | Category | Pass | Expect | Repetition | tok/s | Generated (first 80 chars) |",
                      "|---|---|---|---|---|---|---|"]
            for p in sm["prompts"]:
                gen = p["generated"][:80].replace("|", "\\|").replace("\n", " ") if p["generated"] else ""
                exp = "Y" if p.get("expect_match") else ("N" if p.get("expect_match") is False else "-")
                tps = f"{p.get('decode_tps',0):.2f}" if p.get("decode_tps") else "-"
                lines.append(f"| {p['id']} | {p['cat']} | {'✅' if p['pass'] else '❌'} | {exp} | "
                             f"{'⚠️' if p['repetition'] else '-'} (run={p.get('max_run',0)}) | {tps} | {gen} |")
            lines.append("")
        # Diagnostic
        if "diagnostic" in phases:
            d = phases["diagnostic"]
            lines += ["## Phase 2: Full System Diagnostic", ""]
            dp = d.get("decode_profile", {})
            lines += ["### Decode Timing Breakdown", "",
                      "| Bucket | Seconds |", "|---|---|"]
            for k, label in [("prof_expert_disk","expert-disk"),("prof_expert_mm","expert-matmul"),
                             ("prof_attention","attention"),("prof_lm_head","lm_head"),
                             ("prof_other","other")]:
                v = dp.get(k, "?")
                lines.append(f"| {label} | {v} |")
            lines += [f"", f"### Performance", f"- Decode: **{d.get('decode_tps','?')} tok/s**",
                      f"- Prefill: {d.get('prefill_secs','?')}s",
                      f"- Expert hit rate: {d.get('hit_rate','?')}%",
                      f"- RSS: {d.get('rss_gb','?')} GB",
                      f"- Experts/token: {d.get('experts_per_tok','?')}",
                      f"- MTP acceptance: {d.get('mtp_accept','?')}%", ""]
            if d.get("generated_text"):
                lines += [f"### Generated Text", f"```\n{d['generated_text'][:300]}\n```", ""]
        # Quality
        if "quality" in phases:
            q = phases["quality"]
            lines += ["## Phase 3: Quality Benchmarks", ""]
            if "error" in q:
                lines += [f"⚠️ {q['error']}", ""]
            else:
                lines += [f"**Mean acc_norm: {q.get('mean_acc_norm','?')}%**", "",
                          "| Task | n | acc | acc_norm |", "|---|---|---|---|"]
                for t, s in q.get("scores", {}).items():
                    lines.append(f"| {t} | {s['n']} | {s['acc']:.1f}% | {s['acc_norm']:.1f}% |")
                lines.append("")
        # Throughput
        if "throughput" in phases:
            th = phases["throughput"]
            lines += ["## Phase 4: Throughput", "",
                      "| Mode | tok/s | hit% | MTP accept |", "|---|---|---|---|"]
            on = th.get("mtp_on", {}); off = th.get("mtp_off", {})
            lines.append(f"| MTP ON | {on.get('tps','?')} | {on.get('hit','?')}% | {on.get('mtp_accept','?')}% |")
            lines.append(f"| MTP OFF | {off.get('tps','?')} | {off.get('hit','?')}% | — |")
            if th.get("mtp_speedup"):
                lines.append(f"\n**MTP speedup: {th['mtp_speedup']:.2f}x**")
            lines.append("")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n{'='*60}")
        print(f"Report written:")
        print(f"  JSON: {json_path}")
        print(f"  MD:   {md_path}")
        print(f"  Logs: {self.out_dir}/*.log")
        print(f"{'='*60}")

    def run(self):
        phase = self.args.phase
        if phase == "all":
            self.phase_system()
            self.phase_smoke()
            self.phase_diagnostic()
            self.phase_quality()
            self.phase_throughput()
        elif phase == "system":     self.phase_system()
        elif phase == "smoke":       self.phase_smoke()
        elif phase == "diagnostic":  self.phase_diagnostic()
        elif phase == "quality":     self.phase_quality()
        elif phase == "throughput":  self.phase_throughput()
        else:
            print(f"Unknown phase: {phase}", file=sys.stderr); sys.exit(1)
        self.write_report()


def main():
    ap = argparse.ArgumentParser(description="Comprehensive model diagnostic harness for colibri GLM-5.2")
    ap.add_argument("--snap", required=True, help="model snapshot directory")
    ap.add_argument("--glm", default=None, help="engine binary path (default: ./glm.exe or ./glm)")
    ap.add_argument("--phase", default="all",
                    choices=["all","system","smoke","diagnostic","quality","throughput"],
                    help="which test phase to run")
    ap.add_argument("--out", default=None, help="output directory (default: ./diag_results/<timestamp>)")
    ap.add_argument("--ngen", type=int, default=64, help="generation length for smoke/throughput (default 64)")
    ap.add_argument("--quality-limit", type=int, default=40, help="questions per benchmark task (default 40)")
    ap.add_argument("--ram", type=float, default=0, help="RAM_GB override (0=auto)")
    ap.add_argument("--cuda", action="store_true", help="enable COLI_CUDA GPU tier")
    ap.add_argument("--gpu", type=int, default=None, help="GPU device ordinal (with --cuda)")
    ap.add_argument("--cap", type=int, default=75, help="experts-per-layer cache cap (default 75)")
    ap.add_argument("--timeout", type=int, default=600, help="per-run timeout in seconds (default 600)")
    a = ap.parse_args()
    # Auto-detect engine binary
    if a.glm is None:
        for cand in ["./glm.exe", "./glm", "../glm.exe", os.path.join(os.path.dirname(__file__), "..", "glm.exe")]:
            if os.path.exists(cand): a.glm = os.path.abspath(cand); break
        if a.glm is None:
            print("ERROR: could not find glm/glm.exe. Specify with --glm", file=sys.stderr); sys.exit(1)
    if not os.path.isdir(a.snap):
        print(f"ERROR: snapshot dir does not exist: {a.snap}", file=sys.stderr); sys.exit(1)
    harness = DiagnosticHarness(a)
    harness.run()

if __name__ == "__main__":
    main()
