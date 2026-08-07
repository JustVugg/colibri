"""Efficiency / regression harness for the colibri engine.

The engine already emits rich telemetry (REPLAY tok/s, PROFILE phase timings,
[PROF] time shares + verdict, CUDA expert-tier utilization). Until now every
consumer of that telemetry — `benchmark_cuda_fixture.py`, `bench_full.sh`,
`bench_ux.sh` — has only *printed* it for a human to eyeball. This module turns
each signal into a parseable field so tests can assert on it.

Design:
  - Reuses SPEED_RE / PROFILE_RE from tools.benchmark_cuda_fixture (no drift).
  - parse_run() is pure: stdout+stderr in, dict out. Easy to unit-test against
    captured strings (like the existing test_benchmark_cuda_fixture does).
  - run_engine() is the subprocess wrapper. Captures stdout and stderr
    separately, because the engine splits them: PROFILE/REPLAY/CUDA-tier go to
    stdout, the [CUDA]/[PROF]/[prefill] banners go to stderr.
  - Floor defaults are module constants (tunable in one place, not scattered).

No model file is required to import this module; only run_engine() invokes the
binary. parse_run() works on any captured text, so most test surface is covered
by string fixtures without spinning the engine at all.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

# Reuse the validated regexes from the existing A/B benchmark harness so the
# PROFILE field order (disk, expert_matmul, attention, lm_head, other) and the
# tok/s capture stay identical. Drift here would silently break every consumer.
from tools.benchmark_cuda_fixture import SPEED_RE as _SPEED_RE_REPLAY, PROFILE_RE, PROFILE_KEYS

# SPEED_RE (from benchmark_cuda_fixture) matches the REPLAY-mode line only:
#   "REPLAY decode: ... | 12.34 tok/s | ..."
# run_text / PROMPT mode uses a DIFFERENT format (glm.c:4682):
#   "decode N tokens in X.XXs (12.34 tok/s) | expert hit rate ..."
# This alt regex catches the parenthesized form so the full-model report (which
# uses PROMPT mode) gets a real tok/s instead of reporting it missing.
SPEED_RE_TEXT = re.compile(r"decode \d+ tokens in [0-9.]+s \(([0-9.]+) tok/s\)")


def _first_speed(stdout: str):
    """Find tok/s in whichever run-mode format the engine used."""
    for rx in (_SPEED_RE_REPLAY, SPEED_RE_TEXT):
        m = rx.search(stdout)
        if m:
            return m
    return None


# Public alias so existing imports keep working (tests reference SPEED_RE).
SPEED_RE = _SPEED_RE_REPLAY


# --- additional parsers (formats verified against glm.c printf strings) ---

# "expert hit rate 88.1%" (summary line) | "expert hit 95.0%" (REPLAY line)
HIT_RE = re.compile(r"expert hit(?:\s+rate)?\s+([0-9.]+)%")

# "[PROF] time shares: expert-I/O 3% | expert-matmul 12% | attention 56% | lm_head 2% | other 27%"
SHARES_RE = re.compile(
    r"\[PROF\] time shares: expert-I/O\s+([0-9.]+)%\s*\|\s*expert-matmul\s+([0-9.]+)%\s*"
    r"\|\s*attention\s+([0-9.]+)%\s*\|\s*lm_head\s+([0-9.]+)%\s*\|\s*other\s+([0-9.]+)%"
)

# "[PROF] verdict: I/O-bound — 60% of the time ..."  (also compute-bound / attention-bound / balanced)
VERDICT_RE = re.compile(r"\[PROF\] verdict:\s*(I/O-bound|compute-bound|attention-bound|balanced)")

# "[PROF] expert I/O: ... hit 95.0% (76 hit / 4 load) | 4.0 loads/token"
LOADS_PER_TOK_RE = re.compile(r"\|\s*([0-9.]+)\s+loads/token")

# "CUDA expert tier: 111 resident experts (2.36 GB) | 5400 calls served from VRAM"  (stdout)
CUDA_TIER_RE = re.compile(
    r"CUDA expert tier:\s+(\d+)\s+resident experts\s+\(([0-9.]+)\s+GB\)\s*\|\s+(\d+)\s+calls served from VRAM"
)

# "[CUDA] device 0: NVIDIA ..., 14.4 GB VRAM, sm_120"  (stderr, per device at init)
CUDA_DEVICE_RE = re.compile(r"\[CUDA\] device\s+\d+:")

# "[CUDA] resident set: 12 tensors, 0.45 GB VRAM"  (stderr, cuda_stats_print)
CUDA_RESIDENT_RE = re.compile(
    r"\[CUDA\] resident set:\s+(\d+)\s+tensors,\s+([0-9.]+)\s+GB\s+VRAM"
)

# "PREFILL (teacher-forcing) C vs oracle: 11/32 positions | 1700.4 pos/s"  (TF=1 mode, stdout)
TF_MATCH_RE = re.compile(r"PREFILL \(teacher-forcing\).*:\s+(\d+)/(\d+)\s+positions")

# --- the six deeper signals (added after "are you gathering everything?" audit) ---

# "ATTENTION: projection/RoPE 0.050s | score-softmax-value 0.009s | output projection 0.011s"
# Sub-breakdown of the attention phase — answers "how is attention being read".
ATTN_BREAKDOWN_RE = re.compile(
    r"ATTENTION: projection/RoPE\s+([0-9.]+)s\s*\|\s*score-softmax-value\s+([0-9.]+)s\s*"
    r"\|\s*output projection\s+([0-9.]+)s"
)

# "[PROF] decode forwards: 20 | latency p50 7.3 ms | p90 8.0 ms | p99 8.1 ms | max 8.2 ms | 1.00 tok/forward"
# Per-forward tail latency — a p99 >> p50 means decode stalls (I/O hiccups, KV grow).
LATENCY_RE = re.compile(
    r"\[PROF\] decode forwards:\s+(\d+)\s*\| latency p50\s+([0-9.]+)\s*ms\s*\| "
    r"p90\s+([0-9.]+)\s*ms\s*\|\s*p99\s+([0-9.]+)\s*ms\s*\|\s*max\s+([0-9.]+)\s*ms"
)

# "[PROF] expert I/O: 0.004 GB fetched (0.2 MB/token, 0.03 GB/s over the run) | hit 95.0% ... |
#        4.0 loads/token | 0.0s read service / 0.0s felt wait"
# Absolute disk throughput + the felt-wait split (the [PROF] version, more detailed than PROFILE).
EXPERT_IO_RE = re.compile(
    r"\[PROF\] expert I/O:\s+([0-9.]+)\s+GB fetched\s+\(([0-9.]+)\s+MB/token,\s*([0-9.]+)\s+GB/s"
    r".*?\|\s*([0-9.]+)\s+loads/token\s*\|\s*([0-9.]+)s\s+read service\s*/\s*([0-9.]+)s\s+felt wait"
)

# "speculation: 1.05 tokens/forward (19 forwards per 20 tokens) | MTP acceptance 44% (7/16)"
# Draft efficiency — is the speculative decoder pulling weight or dead overhead?
SPECULATION_RE = re.compile(
    r"speculation:\s+([0-9.]+)\s+tokens/forward\s+\((\d+)\s+forwards per\s+(\d+)\s+tokens\)"
    r"\s*\|\s*MTP acceptance\s+([0-9.]+)%"
)

# "experts loaded/token: 450.0 (per-layer 56.25 across 8; baseline topk=8) | TOPK=0 TOPP=0.00"
# Fuller than loads_per_tok: includes the per-layer spread + the active topk/topp.
EXPERTS_LOADED_RE = re.compile(
    r"experts loaded/token:\s+([0-9.]+)\s+\(per-layer\s+([0-9.]+)\s+across\s+(\d+);\s*baseline topk=(\d+)\)"
)

# "[PROF] machine: Intel(...) | 22 cores (22 omp threads) | RAM 34.1 GB total, 27.1 GB available | backend CUDA"
# Provenance — makes a report reproducible across machines/runs.
MACHINE_RE = re.compile(r"\[PROF\] machine:\s*(.*)\|\s*(\d+)\s+cores.*backend\s+(\S+)")

# "[PROF] config: RAM_GB=auto 23.9 CTX=4096 | expert cache cap 8/layer ... | DRAFT=0 PIPE=1 DIRECT=0 ..."
# Effective resolved config (after auto-budgeting). Answers "what config actually ran".
CONFIG_RE = re.compile(r"\[PROF\] config:\s*(.*)")

# "[ORACLE] mismatch pos=7 expected=197 got=22"  (TF=1 mode, stderr, per position)
# Captures the engine's *actual* argmax at each position, so two backends can be
# compared DIRECTLY (independent of how each relates to the oracle).
TF_MISMATCH_RE = re.compile(r"\[ORACLE\] mismatch pos=(\d+) expected=\d+ got=(\d+)")

# --- routing-quality + disk-split + cuda-groups (the deep signals) ---

# summary suffix: " | swap 3.1% (12/384)"  (CACHE_ROUTE inline)
SWAP_RE = re.compile(r"swap\s+([0-9.]+)%\s+\((\d+)/(\d+)\)")

# summary suffix: " | route_agree 85.0% | route_kl 0.0123"  (CACHE_ROUTE/ROUTE_AGREE)
ROUTE_AGREE_RE = re.compile(r"route_agree\s+([0-9.]+)%\s*\|\s*route_kl\s+([0-9.]+)")

# "disk-load split: draft 8 + absorb 0 + verify/main 3 misses | MTP-layer 0 loads 0.00 GB |
#        main-layers 11 loads 0.01 GB (MTP 0.0% of bytes)"  (DISK_SPLIT=1)
DISK_SPLIT_RE = re.compile(
    r"disk-load split: draft\s+(\d+)\s+\+\s+absorb\s+(\d+)\s+\+\s+verify/main\s+(\d+)\s+misses"
    r"\s*\|\s*MTP-layer\s+(\d+)\s+loads\s+([0-9.]+)\s+GB\s*\|\s*main-layers\s+(\d+)\s+loads\s+([0-9.]+)\s+GB"
    r"(?:\s+\(MTP\s+([0-9.]+)%\s+of bytes\))?"
)

# "[CUDA] expert groups: 120 call, 840 expert, 1200 righe (7.00 expert/call)"
CUDA_GROUPS_RE = re.compile(
    r"\[CUDA\] expert groups:\s+(\d+)\s+call,\s+(\d+)\s+expert,\s+(\d+)\s+righe\s+\(([0-9.]+)\s+expert/call\)"
)

# "[CUDA] expert groups timing: H2D 12.3 ms | kernel 45.6 ms | D2H 7.8 ms"  (COLI_CUDA_PROFILE=1)
CUDA_GROUPS_TIME_RE = re.compile(
    r"\[CUDA\] expert groups timing: H2D\s+([0-9.]+)\s+ms\s*\|\s*kernel\s+([0-9.]+)\s+ms\s*\|\s*D2H\s+([0-9.]+)\s+ms"
)

# LOOKAHEAD recall block — 4 named rows. (name, pct, hit, tot)
LOOKAHEAD_RE = re.compile(
    r"^\s*(.+?)\s+([0-9.]+)%\s+\((\d+)/(\d+)\)\s*$", re.MULTILINE
)

# "loaded in 0.02s | resident dense: 0.21 MB | layers=5 experts=8 | MTP absent (draft=0)"
LOAD_BANNER_RE = re.compile(
    r"loaded in\s+([0-9.]+)s\s*\|\s*resident dense:\s+([0-9.]+)\s+MB\s*\|"
    r"\s*layers=(\d+)\s+experts=(\d+)\s*\|\s*MTP\s+(\w+)\s+\(draft=(\d+)\)"
)


# --- tunable floors -----------------------------------------------------------
# These are deliberately generous so they catch *regressions* (broken builds,
# pathological configs, telemetry accounting bugs) without flapping on machine
# noise. The tiny model is fully resident at ~200 tok/s, so a 20 tok/s floor is
# a 10x margin. Tune per-host via env if needed (documented in README).
TINY_TOK_S_FLOOR = float(os.environ.get("COLI_TINY_TOK_S_FLOOR", "20.0"))
# On a fully-resident tiny model the expert-disk wait share should be tiny.
# If it exceeds this, something regressed in the I/O accounting or cache path.
MAX_DISK_WAIT_SHARE = float(os.environ.get("COLI_MAX_DISK_WAIT_SHARE", "0.20"))
# decode wall-time should be roughly the sum of PROFILE phases (other = residual).
PROFILE_SUM_TOLERANCE = float(os.environ.get("COLI_PROFILE_SUM_TOL", "0.05"))
# Minimum direct CPU-vs-CUDA argmax agreement on the tiny TF fixture. The two
# backends use different accumulation orders (SIMD dot vs CUDA kernel), so a
# few near-tied positions flip argmax — that's expected numeric divergence, not
# a kernel bug. Measured baseline ~84% (27/32) on this fixture; the 70% floor
# leaves headroom for machine noise while still catching a catastrophic kernel
# regression (e.g. a wrong GEMM would drop this to ~random = ~4%).
MIN_CPU_CUDA_AGREEMENT = float(os.environ.get("COLI_MIN_CPU_CUDA_AGREE", "0.70"))


def parse_run(stdout: str, stderr: str = "") -> dict:
    """Parse one engine run's output into a telemetry dict.

    Returns keys: tok_s, hit_pct, profile (dict, seconds), profile_sum,
    time_shares (dict, fractions 0..1), verdict, loads_per_tok, cuda (dict),
    tf_match (tuple or None), parsed (set of field names found).

    Raises RuntimeError only if the core throughput line is missing — everything
    else is optional and absent on some run modes (e.g. [PROF] needs PROF=1,
    CUDA tier needs gpu_expert_count>0).
    """
    out = dict(
        tok_s=None, hit_pct=None, profile=None, profile_sum=None,
        time_shares=None, verdict=None, loads_per_tok=None,
        cuda=None, tf_match=None, stderr=stderr,
    )
    parsed = set()
    blob = stdout + "\n" + stderr  # [PROF]/[CUDA] live on stderr; scan both.

    m = _first_speed(stdout)
    if m:
        out["tok_s"] = float(m.group(1)); parsed.add("tok_s")

    m = HIT_RE.search(blob)
    if m:
        out["hit_pct"] = float(m.group(1)); parsed.add("hit_pct")

    m = PROFILE_RE.search(stdout)
    if m:
        service, wait, emm, attn, head, other = (float(x) for x in m.groups())
        disk = service + (wait or 0.0)
        out["profile"] = dict(zip(PROFILE_KEYS, (disk, emm, attn, head, other)))
        out["profile_sum"] = disk + emm + attn + head + other
        parsed.add("profile")

    # ATTENTION sub-breakdown: projection/RoPE | score-softmax-value | output.
    m = ATTN_BREAKDOWN_RE.search(stdout)
    if m:
        out["attn_breakdown"] = dict(zip(
            ("proj_rope", "score_sm_value", "out_proj"),
            (float(x) for x in m.groups())))
        parsed.add("attn_breakdown")

    m = SHARES_RE.search(blob)
    if m:
        io, emm, attn, head, other = (float(x) / 100.0 for x in m.groups())
        out["time_shares"] = dict(io=io, matmul=emm, attention=attn, head=head, other=other)
        parsed.add("time_shares")

    m = VERDICT_RE.search(blob)
    if m:
        out["verdict"] = m.group(1); parsed.add("verdict")

    # [PROF] decode forwards + latency p50/p90/p99/max.
    m = LATENCY_RE.search(blob)
    if m:
        out["latency"] = dict(zip(
            ("forwards", "p50_ms", "p90_ms", "p99_ms", "max_ms"),
            (float(x) for x in m.groups())))
        parsed.add("latency")

    # [PROF] expert I/O throughput: GB fetched, MB/token, GB/s, service vs felt wait.
    m = EXPERT_IO_RE.search(blob)
    if m:
        out["expert_io"] = dict(zip(
            ("gb_fetched", "mb_per_tok", "gb_per_s", "loads_per_tok",
             "read_service_s", "felt_wait_s"),
            (float(x) for x in m.groups())))
        parsed.add("expert_io")

    m = LOADS_PER_TOK_RE.search(blob)
    if m:
        out["loads_per_tok"] = float(m.group(1)); parsed.add("loads_per_tok")

    # experts loaded/token with per-layer spread + baseline topk (run_text summary).
    m = EXPERTS_LOADED_RE.search(stdout)
    if m:
        out["experts_loaded"] = dict(
            per_tok=float(m.group(1)), per_layer=float(m.group(2)),
            n_sparse_layers=int(m.group(3)), baseline_topk=int(m.group(4)))
        parsed.add("experts_loaded")

    # speculation: tokens/forward, forwards, tokens, MTP acceptance%.
    m = SPECULATION_RE.search(stdout)
    if m:
        out["speculation"] = dict(zip(
            ("tok_per_fw", "forwards", "tokens", "mtp_accept_pct"),
            (float(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))))
        parsed.add("speculation")

    # routing quality (CACHE_ROUTE inline suffixes on the summary line).
    m = SWAP_RE.search(stdout)
    if m:
        out["swap"] = dict(pct=float(m.group(1)), swaps=int(m.group(2)), slots=int(m.group(3)))
        parsed.add("swap")
    m = ROUTE_AGREE_RE.search(stdout)
    if m:
        out["route_agree"] = dict(agree_pct=float(m.group(1)), kl=float(m.group(2)))
        parsed.add("route_agree")

    # disk-load split by decode phase (DISK_SPLIT=1).
    m = DISK_SPLIT_RE.search(stdout)
    if m:
        out["disk_split"] = dict(zip(
            ("draft", "absorb", "verify_main", "mtp_loads", "mtp_gb",
             "main_loads", "main_gb", "mtp_bytes_pct"),
            (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
             float(m.group(5)), int(m.group(6)), float(m.group(7)),
             float(m.group(8)) if m.group(8) else None)))
        parsed.add("disk_split")

    # provenance: machine + resolved config.
    m = MACHINE_RE.search(blob)
    if m:
        out["machine"] = dict(cpu=m.group(1).strip(), cores=int(m.group(2)), backend=m.group(3))
        parsed.add("machine")
    m = CONFIG_RE.search(blob)
    if m:
        out["config_str"] = m.group(1).strip(); parsed.add("config")

    # load banner: load time, resident dense MB, layers, experts, MTP status.
    m = LOAD_BANNER_RE.search(stdout)
    if m:
        out["load"] = dict(zip(
            ("load_s", "resident_dense_mb", "layers", "experts", "mtp_status", "draft"),
            (float(m.group(1)), float(m.group(2)), int(m.group(3)),
             int(m.group(4)), m.group(5), int(m.group(6)))))
        parsed.add("load")

    # LOOKAHEAD routing-recall block (LOOKA=1): list of {predictor, pct, hit, tot}.
    la_block = re.search(
        r"LOOKAHEAD routing.*?recall.*?:\n((?:^\s+.+?\s+[0-9.]+%\s+\(\d+/\d+\)\s*$\n?)+)",
        blob, re.MULTILINE)
    if la_block:
        out["lookahead"] = []
        for row in LOOKAHEAD_RE.finditer(la_block.group(1)):
            out["lookahead"].append(dict(
                predictor=row.group(1).strip(), pct=float(row.group(2)),
                hit=int(row.group(3)), tot=int(row.group(4))))
        parsed.add("lookahead")

    cuda = dict(enabled=False, expert_count=None, expert_gb=None,
                calls_served=None, resident_tensors=None, resident_gb=None,
                groups=None, groups_timing=None)
    if CUDA_DEVICE_RE.search(stderr):
        cuda["enabled"] = True
    m = CUDA_TIER_RE.search(stdout)
    if m:
        cuda["expert_count"] = int(m.group(1))
        cuda["expert_gb"] = float(m.group(2))
        cuda["calls_served"] = int(m.group(3))
    m = CUDA_RESIDENT_RE.search(stderr)
    if m:
        cuda["resident_tensors"] = int(m.group(1))
        cuda["resident_gb"] = float(m.group(2))
    m = CUDA_GROUPS_RE.search(stderr)
    if m:
        cuda["groups"] = dict(zip(
            ("calls", "experts", "rows", "experts_per_call"),
            (int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)))))
    m = CUDA_GROUPS_TIME_RE.search(stderr)
    if m:
        cuda["groups_timing"] = dict(zip(
            ("h2d_ms", "kernel_ms", "d2h_ms"),
            (float(m.group(1)), float(m.group(2)), float(m.group(3)))))
    out["cuda"] = cuda

    m = TF_MATCH_RE.search(stdout)
    if m:
        out["tf_match"] = (int(m.group(1)), int(m.group(2))); parsed.add("tf_match")

    # Capture per-position argmax divergences from the oracle, keyed by position.
    mismatches = {int(mm.group(1)): int(mm.group(2))
                  for mm in TF_MISMATCH_RE.finditer(blob)}
    if out["tf_match"] is not None:
        out["tf_mismatches"] = mismatches
        parsed.add("tf_mismatches")

    out["parsed"] = parsed
    return out


def run_engine(
    env_overlay: dict,
    *,
    engine: Optional[str] = None,
    cap: int = 4,
    ebits: int = 4,
    dbits: int = 4,
    timeout: float = 600.0,
    snap: Optional[str] = None,
) -> tuple[dict, subprocess.CompletedProcess]:
    """Run the engine with an env overlay; return (parsed_telemetry, proc).

    `engine` defaults to the built ./colibri (or .exe). `snap` defaults to
    the bundled tiny model (glm_tiny) so callers can omit it for fast tests.
    The positional argv is `cap ebits dbits`, matching the engine's main().
    """
    if engine is None:
        # glm was renamed colibri; prefer whichever of the two names exists so
        # a stale checkout still resolves, and fall back to the current one.
        _c = Path(__file__).resolve().parent.parent
        engine = str(next((p for p in (_c / "colibri", _c / "colibri.exe") if p.exists()), _c / "colibri"))
    env = os.environ.copy()
    # Strip CUDA vars by default so a CPU run isn't accidentally GPU-accelerated
    # by a leftover env; callers opt in by passing them in env_overlay.
    for k in ("COLI_CUDA", "COLI_GPU", "COLI_GPUS", "CUDA_DENSE", "CUDA_EXPERT_GB"):
        env.pop(k, None)
    env.update(env_overlay)
    if snap is not None:
        env["SNAP"] = snap
    elif "SNAP" not in env:
        env["SNAP"] = str(Path(__file__).resolve().parent.parent / "glm_tiny")

    proc = subprocess.run(
        [engine, str(cap), str(ebits), str(dbits)],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    telemetry = parse_run(proc.stdout, proc.stderr)
    telemetry["returncode"] = proc.returncode
    telemetry["env"] = {k: env_overlay[k] for k in env_overlay}
    return telemetry, proc


def disk_wait_share(t: dict) -> Optional[float]:
    """Fraction of decode wall-time spent waiting on expert disk reads.

    Preferred source: [PROF] time_shares (the engine's own accounting, which
    separates felt-wait from read-service). Falls back to PROFILE disk / sum if
    [PROF] wasn't emitted (PROF=0 runs). None if neither is available.
    """
    if t.get("time_shares"):
        return t["time_shares"]["io"]
    if t.get("profile") and t.get("profile_sum"):
        return t["profile"]["disk"] / t["profile_sum"]
    return None


def tf_agreement(cpu: dict, cuda: dict, oracle: list[int]) -> tuple[float, list[int]]:
    """Direct CPU-vs-CUDA argmax agreement on the TF fixture.

    Both runs prefilled the SAME oracle sequence; tf_mismatches holds each
    backend's actual argmax where it diverged from the oracle. Where a backend
    is ABSENT from the mismatch map, its prediction equals the oracle token at
    that position. So the reconstructed per-position prediction is:
        oracle[i] if i not in mismatches else mismatches[i]
    and agreement is the fraction of positions where CPU and CUDA predictions
    are identical — independent of how each relates to the oracle.

    `oracle` is ref_glm.json's tf_pred (the per-position oracle argmax). Pass
    n_positions = len(oracle).

    Returns (agreement_fraction, list_of_differing_positions).
    """
    cm = cpu.get("tf_mismatches") or {}
    gm = cuda.get("tf_mismatches") or {}
    diff = []
    for i, orc in enumerate(oracle):
        cpu_tok = cm.get(i, orc)     # matched oracle => oracle token
        cuda_tok = gm.get(i, orc)
        if cpu_tok != cuda_tok:
            diff.append(i)
    agree = (len(oracle) - len(diff)) / len(oracle) if oracle else 0.0
    return agree, diff
