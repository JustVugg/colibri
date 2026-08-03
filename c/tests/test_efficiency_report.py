#!/usr/bin/env python3
"""Exhaustive optimization dossier for a colibri engine run.

This is NOT a pass/fail test. It runs the engine with every instrumentation flag
on (PROF, COLI_CUDA_PROFILE, CACHE_ROUTE, DISK_SPLIT, LOOKA) and prints a section-
by-section report answering, for each subsystem:

    WHAT is doing it    — which phase/kernel/tier
    WHEN it is doing it — how much of decode wall-time it owns
    WITH WHAT           — the config/weights/tier it used
    IS IT INEFFICIENT?  — a verdict, with the threshold
    HOW TO IMPROVE      — the concrete knob, named

Activation (opt-in only — NOT in `make test`):
    COLI_EFFICIENCY_MODEL=<model_dir> python tests/test_efficiency_report.py

Optional env:
    COLI_EFFICIENCY_CUDA=1        also exercise the CUDA dense/expert tiers
    COLI_EFFICIENCY_NGEN=N        decode tokens (default 24)
    COLI_EFFICIENCY_RAM_GB=N      RAM budget (default 28)
    COLI_EFFICIENCY_VRAM_GB=N     CUDA expert-tier budget GB (default 4)
    COLI_EFFICIENCY_PROMPT=...    prompt (default: a code-gen prompt)

Exit code is always 0 (it's a dossier, not a gate). Lines marked FLAG point at
the most likely lever to move tok/s for the observed bottleneck.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.efficiency import run_engine, disk_wait_share  # noqa: E402


# --- advisory thresholds (the "IS IT INEFFICIENT?" lines) ---
DISK_WAIT_DOMINANT = 0.40     # >40% decode waiting on expert reads -> I/O-bound
LOW_HIT_RATE = 0.30           # <30% cache hit -> thrashing (cap too small)
LOW_ROUTE_AGREE = 0.80        # <80% routing overlap -> prefetch is guessing wrong
HIGH_TAIL_RATIO = 3.0         # p99 > 3x p50 -> decode stalls (I/O hiccups / KV grow)
LOW_MTP_ACCEPT = 0.20         # <20% MTP acceptance -> draft decoder is dead weight
VRAM_WASTE_CALLS = 0          # experts pinned in VRAM but 0 calls served


def _flag(ok): return "OK " if ok else "FLAG"


def _bar(frac, width=24):
    """A simple ASCII bar for share visualization."""
    n = max(0, min(width, round(frac * width)))
    return "#" * n + "." * (width - n)


def _line(label, value, flag=None, note=""):
    tag = f"  [{flag}]" if flag else ""
    print(f"  {label:<22} {value}{tag}  {note}" if note else f"  {label:<22} {value}{tag}")


def main() -> int:
    model = os.environ.get("COLI_EFFICIENCY_MODEL")
    if not model:
        print(__doc__)
        print("\nNot activated: set COLI_EFFICIENCY_MODEL=<model_dir> to run.")
        return 0
    model = str(Path(model).resolve())
    if not Path(model).is_dir():
        print(f"ERROR: {model} is not a directory", file=sys.stderr)
        return 0

    ngen = int(os.environ.get("COLI_EFFICIENCY_NGEN", "24"))
    ram_gb = os.environ.get("COLI_EFFICIENCY_RAM_GB", "28")
    vram_gb = os.environ.get("COLI_EFFICIENCY_VRAM_GB", "4")
    prompt = os.environ.get(
        "COLI_EFFICIENCY_PROMPT",
        "Write a Python function that computes the factorial of a number. "
        "Include error handling and a docstring.")
    use_cuda = os.environ.get("COLI_EFFICIENCY_CUDA") == "1"

    # Turn ON every instrumentation flag so the dossier has maximum detail.
    # These are all observability toggles (PROF/COLI_CUDA_PROFILE/CACHE_ROUTE/
    # DISK_SPLIT/LOOKA); none change the computed output.
    overlay = dict(
        NGEN=str(ngen), TEMP="0", RAM_GB=ram_gb, PROMPT=prompt,
        PROF="1", CACHE_ROUTE="1", DISK_SPLIT="1", LOOKA="1", ROUTE_AGREE="1",
    )
    if use_cuda:
        overlay.update(COLI_CUDA="1", COLI_GPU="0", CUDA_DENSE="1",
                       COLI_CUDA_PROFILE="1", CUDA_EXPERT_GB=vram_gb)

    print("=" * 78)
    print(f"OPTIMIZATION DOSSIER — {Path(model).name}")
    print(f"  mode : {'CUDA (dense+expert tiers)' if use_cuda else 'CPU-only'}   "
          f"ngen : {ngen}   ram : {ram_gb} GB" +
          (f"   vram : {vram_gb} GB" if use_cuda else ""))
    print("=" * 78)

    t0 = time.time()
    t, proc = run_engine(overlay, snap=model, timeout=3600.0)
    wall = time.time() - t0
    flags = []  # collected FLAG lines for the summary

    print("\n[0] RUN")
    _line("wall clock", f"{wall:.0f}s")
    _line("exit code", proc.returncode,
          None if proc.returncode == 0 else "FLAG",
          "" if proc.returncode == 0 else "non-zero exit")
    if proc.returncode != 0:
        print("    stderr tail:")
        for ln in proc.stderr.strip().splitlines()[-8:]:
            print(f"      {ln}")
        return 0

    # ---------------------------------------------------------------- [1] WHO ----
    print("\n[1] PROVENANCE  — what is running, on what, with what config")
    if t.get("machine"):
        m = t["machine"]
        _line("CPU", m["cpu"])
        _line("cores / omp", f"{m['cores']} cores")
        _line("backend", m["backend"])
    if t.get("load"):
        ld = t["load"]
        _line("model load time", f"{ld['load_s']:.2f}s")
        _line("resident dense", f"{ld['resident_dense_mb']:.1f} MB")
        _line("layers / experts", f"{ld['layers']} layers, {ld['experts']} experts")
        _line("MTP", f"{ld['mtp_status']} (draft={ld['draft']})")
    if t.get("config_str"):
        _line("resolved config", t["config_str"])
        print("    (this is the EFFECTIVE config after auto-budgeting — not your env verbatim)")

    # ---------------------------------------------------------------- [2] SPEED --
    print("\n[2] THROUGHPUT  — is it fast, is the tail healthy")
    if t.get("tok_s") is not None:
        _line("tok/s", f"{t['tok_s']:.3f}")
    else:
        flags.append("throughput line missing — engine output format may have changed")
    if t.get("latency"):
        la = t["latency"]
        _line("decode forwards", f"{int(la['forwards'])}")
        _line("p50 / p90", f"{la['p50_ms']:.1f} / {la['p90_ms']:.1f} ms")
        _line("p99 / max", f"{la['p99_ms']:.1f} / {la['max_ms']:.1f} ms")
        tail_ok = la["p99_ms"] <= HIGH_TAIL_RATIO * la["p50_ms"]
        _line("tail ratio (p99/p50)", f"{la['p99_ms']/max(la['p50_ms'],1e-9):.2f}x",
              _flag(tail_ok),
              "high tail = decode stalls (I/O hiccups, KV growth, re-pin)")
        if not tail_ok:
            flags.append(f"tail latency p99={la['p99_ms']:.1f}ms >> p50={la['p50_ms']:.1f}ms "
                         "(look for REPIN swaps or disk stalls)")

    # ---------------------------------------------------------------- [3] TIME ---
    print("\n[3] WHERE TIME GOES  — what is doing it, when (share of decode)")
    ts = t.get("time_shares")
    prof = t.get("profile")
    if ts:
        order = [("io", "expert-disk I/O", DISK_WAIT_DOMINANT, "the cache is too small / disk is slow"),
                 ("matmul", "expert matmul", 0.40, "compute-bound; more cores or a GPU expert tier"),
                 ("attention", "attention", 0.35, "context length is the cost; lower CTX"),
                 ("head", "lm_head", 0.10, "vocab projection; unusual to dominate"),
                 ("other", "other", 0.30, "scheduling / KV bookkeeping overhead")]
        for key, name, thresh, lever in order:
            f = ts[key]
            ok = f < thresh
            _line(name, f"{f:5.0%}  {_bar(f)}", _flag(ok),
                  "" if ok else f"->{lever}")
            if not ok:
                flags.append(f"{name} dominates ({f:.0%}) -> {lever}")
        if t.get("verdict"):
            print(f"  engine verdict         : {t['verdict']}")
    elif prof:
        print("  (no [PROF] time shares — set PROF=1 for phase percentages)")
    if prof:
        print("  absolute seconds       :")
        for k in ("disk", "expert_matmul", "attention", "lm_head", "other"):
            _line(k, f"{prof[k]:.3f}s")

    # attention sub-breakdown: how is attention being read
    ab = t.get("attn_breakdown")
    if ab:
        print("\n[3a] ATTENTION BREAKDOWN  — how the attention phase is spent")
        atot = sum(ab.values()) or 1.0
        for k, label in (("proj_rope", "projection + RoPE"),
                         ("score_sm_value", "score-softmax-value"),
                         ("out_proj", "output projection")):
            _line(label, f"{ab[k]:.3f}s  ({ab[k]/atot:.0%} of attn)")

    # ---------------------------------------------------------------- [4] CACHE --
    print("\n[4] EXPERT CACHE  — is the cache efficient")
    hit = t.get("hit_pct")
    if hit is not None:
        ok = hit >= LOW_HIT_RATE * 100
        _line("hit rate", f"{hit:.1f}%", _flag(ok),
              "" if ok else "<30% = thrashing; raise RAM_GB or cap")
        if not ok:
            flags.append(f"cache hit {hit:.1f}% is low -> raise RAM_GB (or cap), add PIN_GB")
    el = t.get("experts_loaded")
    if el:
        per_tok = el["per_tok"]
        _line("experts loaded/token", f"{per_tok:.1f}")
        _line("  per-layer", f"{el['per_layer']:.2f} across {el['n_sparse_layers']} sparse layers")
        _line("  baseline", f"topk={el['baseline_topk']} active experts/token")
        base_topk = el["baseline_topk"]
        if base_topk > 0 and per_tok > 2 * base_topk:
            flags.append(f"loading {per_tok:.0f} experts/token vs topk={base_topk} "
                         "-> redundant I/O; cache is re-fetching evicted experts")

    # ---------------------------------------------------------------- [5] DISK ---
    print("\n[5] DISK I/O  — is I/O the bottleneck, and where")
    eio = t.get("expert_io")
    if eio:
        _line("total fetched", f"{eio['gb_fetched']:.3f} GB")
        _line("per token", f"{eio['mb_per_tok']:.1f} MB/token")
        _line("disk throughput", f"{eio['gb_per_s']:.2f} GB/s over the run")
        _line("read service", f"{eio['read_service_s']:.2f}s (on I/O threads)")
        _line("felt wait", f"{eio['felt_wait_s']:.2f}s (stall compute felt)")
        if eio["felt_wait_s"] > eio["read_service_s"] * 0.5 and eio["read_service_s"] > 0:
            flags.append("felt wait is a large fraction of read service -> PIPE=1 may not be "
                         "overlapping fully, or DIRECT=1 on NVMe")
    ds = t.get("disk_split")
    if ds:
        print("\n[5a] DISK-LOAD SPLIT  — which decode phase reads the bytes")
        _line("draft phase", f"{ds['draft']} loads")
        _line("absorb phase", f"{ds['absorb']} loads")
        _line("verify/main", f"{ds['verify_main']} loads")
        _line("MTP-layer bytes", f"{ds['mtp_loads']} loads, {ds['mtp_gb']:.2f} GB")
        _line("main-layer bytes", f"{ds['main_loads']} loads, {ds['main_gb']:.2f} GB")
        if ds.get("mtp_bytes_pct") is not None:
            _line("MTP share of bytes", f"{ds['mtp_bytes_pct']:.1f}%")
    share = disk_wait_share(t)
    if share is not None:
        ok = share < DISK_WAIT_DOMINANT
        _line("disk-wait share", f"{share:.0%}", _flag(ok),
              "" if ok else "I/O-bound (see levers in [3])")

    # ---------------------------------------------------------------- [6] ROUTE --
    print("\n[6] ROUTING QUALITY  — is the router / prefetch accurate")
    ra = t.get("route_agree")
    if ra:
        ok = ra["agree_pct"] >= LOW_ROUTE_AGREE * 100
        _line("route_agree", f"{ra['agree_pct']:.1f}% overlap with true top-K",
              _flag(ok),
              "" if ok else "prefetch is guessing wrong; CACHE_ROUTE params may need tuning")
        _line("route_kl", f"{ra['kl']:.4f} mean KL (lower = closer to true routing)")
        if not ok:
            flags.append(f"route_agree {ra['agree_pct']:.1f}% low -> tune ROUTE_J/M/P, "
                         "or prefetch is hurting more than helping")
    sw = t.get("swap")
    if sw:
        _line("cache swaps", f"{sw['swaps']}/{sw['slots']} ({sw['pct']:.1f}%)",
              None, "high swap = churn between turns")
    la = t.get("lookahead")
    if la:
        print("\n[6a] ROUTING PREDICTABILITY  — recall of true experts in predicted top-8")
        print("    (which predictor should drive prefetch? highest recall wins)")
        for row in la:
            _line(row["predictor"][:34], f"{row['pct']:5.1f}%  ({row['hit']}/{row['tot']})")

    # ---------------------------------------------------------------- [7] SPEC ---
    print("\n[7] SPECULATION  — is the draft decoder pulling weight")
    sp = t.get("speculation")
    if sp:
        _line("tokens/forward", f"{sp['tok_per_fw']:.2f}  (>1.0 means speculation helps)")
        _line("forwards/tokens", f"{sp['forwards']} forwards for {sp['tokens']} tokens")
        # Speculation helps only if acceptance is high enough that tok/forward > 1.
        # tok_per_fw already == 1.0 when nothing verifies, so judge by acceptance.
        ok = sp["mtp_accept_pct"] >= LOW_MTP_ACCEPT * 100
        _line("MTP acceptance", f"{sp['mtp_accept_pct']:.0f}%", _flag(ok),
              "" if ok else "<20% -> drafts rarely verify; DRAFT=0 may be faster")
        if not ok:
            flags.append(f"MTP acceptance {sp['mtp_accept_pct']:.0f}% low -> "
                         "drafts cost more I/O than they save; try DRAFT=0")

    # ---------------------------------------------------------------- [8] GPU ----
    print("\n[8] GPU TIERS  — is the GPU actually used")
    cuda = t.get("cuda") or {}
    if not cuda.get("enabled"):
        print("  (CUDA not enabled — CPU-only run)")
    else:
        if cuda.get("resident_tensors") is not None:
            _line("resident dense tensors", f"{cuda['resident_tensors']} tensors, "
                  f"{cuda['resident_gb']:.2f} GB")
        if cuda.get("expert_count") is not None:
            waste = cuda["calls_served"] <= VRAM_WASTE_CALLS
            _line("expert tier", f"{cuda['expert_count']} experts pinned "
                  f"({cuda['expert_gb']:.2f} GB)", _flag(not waste))
            _line("  calls served", f"{cuda['calls_served']} from VRAM",
                  _flag(not waste),
                  "" if not waste else "WASTE: pinned but never routed -> lower CUDA_EXPERT_GB")
            if waste:
                flags.append("VRAM expert tier has 0 calls served -> experts pinned but unused; "
                             "PIN stats may not match this workload")
        if cuda.get("groups"):
            g = cuda["groups"]
            _line("expert groups", f"{g['calls']} calls, {g['experts']} experts, "
                  f"{g['rows']} rows ({g['experts_per_call']:.1f} experts/call)")
        if cuda.get("groups_timing"):
            gt = cuda["groups_timing"]
            _line("GPU timing", f"H2D {gt['h2d_ms']:.1f} ms | kernel {gt['kernel_ms']:.1f} ms | "
                  f"D2H {gt['d2h_ms']:.1f} ms")
            if gt["h2d_ms"] + gt["d2h_ms"] > gt["kernel_ms"]:
                flags.append("CUDA H2D+D2H > kernel time -> transfer-bound; "
                             "consider larger expert tier to keep weights resident")

    # ---------------------------------------------------------------- summary ----
    print("\n" + "=" * 78)
    if flags:
        print(f"  {len(flags)} FLAG(s) — the most likely levers to move tok/s:")
        for f in flags:
            print(f"    - {f}")
    else:
        print("  no flags — every measured subsystem is within advisory thresholds.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
