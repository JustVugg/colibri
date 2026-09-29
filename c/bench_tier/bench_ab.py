#!/usr/bin/env python3
"""Interleaved A/B benchmark harness for OLMoE tier-cache/PILOT experiments.

Drives ./olmoe in SERVE=1 mode over the real wire protocol (READY handshake,
SUBMIT/CANCEL, DATA/DONE/PROF/HITS/TIERS/EMAP frames) under a fixed expert
cache cap, and interleaves repetitions across configs so page-cache and
thermal drift cancel:

    for rep in 1..R: for config in configs: run

Configs are plain env-var dicts passed through to the engine untouched:

    --configs 'baseline:PILOT=0|HOT=0|RECENT_RING=0|RECENT_BLOOM=0|MARKOV=0,
               phase1:PILOT=0|HOT=0|RECENT_RING=1|RECENT_BLOOM=1|MARKOV=0,
               phase12:PILOT=0|HOT=0|RECENT_RING=1|RECENT_BLOOM=1|MARKOV=1'

Per serve run: decode-only tok/s = 1000 / mean(DATA-frame step deltas, the
199 deltas after the first DATA), P50/P90/P99/max step ms, DONE STAT
tokens/tok_s/hit%, PROF disk/matmul/attn seconds, SHA-256[:16] + byte length
of the concatenated DATA payloads, startup seconds, wall seconds. After the
serve matrix every config is validated token-exactly with the standalone
harness from c/:

    SNAP=... <config env> ./olmoe 8 8 bench_tier/ref200.json

Outputs: appends JSONL to c/bench_tier/ab_<label>.jsonl and writes
c/bench_tier/ab_<label>_summary.md (markdown table + the exact commands used).

Measurement only: no engine code is touched. $SNAP/hot_pinned.bin is removed
before every run so persistent pinning cannot leak across configs. Every run
has a wall-clock timeout (default 600s) and a deterministic fallback: if the
READY handshake or DONE never arrives the engine is killed, the failure is
recorded, and the harness continues.

Usage:
  python3 c/bench_tier/bench_ab.py [--configs SPEC] [--reps 3] [--label ab]
                                   [--cap 8] [--max-tokens 200] [--warmup 1]
                                   [--timeout 600]
"""
import argparse
import hashlib
import json
import os
import re
import select
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(C_DIR)
BENCH_DIR = os.path.join(C_DIR, "bench_tier")
BIN = os.path.join(C_DIR, "olmoe")
PROMPT_PATH = os.path.join(BENCH_DIR, "prompt.txt")
REF_PATH = os.path.join(BENCH_DIR, "ref200.json")
SNAP = os.environ.get("SNAP", "/Users/khalid/models/olmoe_merged")
PIN_FILE = os.path.join(SNAP, "hot_pinned.bin")
CTX = os.environ.get("CTX", "2048")
BITS = "8"
GRID_IDLE_S = 3.0

DEFAULT_CONFIGS = [
    ("baseline", {"PILOT": "0", "HOT": "0", "RECENT_RING": "0", "RECENT_BLOOM": "0", "MARKOV": "0"}),
    ("phase1", {"PILOT": "0", "HOT": "0", "RECENT_RING": "1", "RECENT_BLOOM": "1", "MARKOV": "0"}),
    ("phase12", {"PILOT": "0", "HOT": "0", "RECENT_RING": "1", "RECENT_BLOOM": "1", "MARKOV": "1"}),
]


class RunTimeout(Exception):
    pass


class EngineGone(Exception):
    pass


class FrameReader:
    """Buffered non-blocking reader over the engine's stdout pipe.

    Deadlines are absolute time.time() values; RunTimeout is raised when the
    deadline passes with no complete line, EngineGone on EOF mid-frame.
    """

    def __init__(self, fd):
        self.fd = fd
        self.buf = bytearray()
        self.eof = False

    def _fill(self, deadline):
        while True:
            if self.eof:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RunTimeout("no data before deadline")
            ready, _, _ = select.select([self.fd], [], [], min(0.25, remaining))
            if not ready:
                continue
            chunk = os.read(self.fd, 65536)
            if not chunk:
                self.eof = True
                return
            self.buf += chunk
            return

    def read_line(self, deadline):
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line = bytes(self.buf[:i])
                del self.buf[:i + 1]
                return line
            if self.eof:
                if self.buf:
                    line = bytes(self.buf)
                    self.buf.clear()
                    return line
                return None
            self._fill(deadline)

    def read_exact(self, n, deadline):
        while len(self.buf) < n:
            if self.eof:
                raise EngineGone("short frame read (%d of %d bytes)" % (len(self.buf), n))
            self._fill(deadline)
        data = bytes(self.buf[:n])
        del self.buf[:n]
        return data


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def fmt(v, prec=3):
    if v is None or isinstance(v, bool):
        return "n/a"
    try:
        return ("%%.%df" % prec) % float(v)
    except (TypeError, ValueError):
        return "n/a"


def parse_configs(spec):
    """Parse --configs into [(name, {ENV: VAL}), ...].

    Items are comma separated; each item is 'name:KEY=VAL|KEY=VAL' or a bare
    'name' (a name matching a built-in default reuses that default's env).
    """
    if not spec:
        return [(n, dict(e)) for n, e in DEFAULT_CONFIGS]
    defaults = dict(DEFAULT_CONFIGS)
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, body = item.partition(":")
        name = name.strip()
        if not name:
            raise SystemExit("bad --configs item: %r" % item)
        env = {}
        if sep:
            for kv in body.split("|"):
                kv = kv.strip()
                if not kv:
                    continue
                key, eq, val = kv.partition("=")
                key = key.strip()
                if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    raise SystemExit("bad --configs override %r in %r" % (kv, item))
                env[key] = val.strip()
        elif name in defaults:
            env = dict(defaults[name])
        out.append((name, env))
    if not out:
        raise SystemExit("--configs parsed to zero configs")
    return out


def cmd_line(overrides, extra_env, argv):
    parts = ["SNAP=%s" % shlex.quote(SNAP)]
    for k in sorted(extra_env):
        parts.append("%s=%s" % (k, shlex.quote(extra_env[k])))
    for k in sorted(overrides):
        parts.append("%s=%s" % (k, shlex.quote(overrides[k])))
    parts.append(" ".join(shlex.quote(a) for a in argv))
    return " ".join(parts)


def _terminate(p, grace=5.0):
    if p is None:
        return
    if p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                p.terminate()
            except Exception:
                pass
        try:
            p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
    for stream in (p.stdin, p.stdout):
        try:
            if stream is not None and not stream.closed:
                stream.close()
        except Exception:
            pass


def _read_tail(f, limit=1500):
    try:
        f.flush()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit))
        data = f.read(limit)
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        return data
    except Exception:
        return ""


def popcount_hex(hexs):
    n = 0
    for i in range(0, len(hexs) - 1, 2):
        try:
            n += bin(int(hexs[i:i + 2], 16)).count("1")
        except ValueError:
            pass
    return n


def parse_tiers(line):
    f = line.split()
    out = {"raw": line.decode("ascii", "replace")}
    if len(f) >= 6:
        try:
            out.update({"filled": int(f[2]), "ondisk": int(f[3]), "gb": float(f[5])})
        except ValueError:
            pass
    return out


def parse_emap(line):
    f = line.split()
    if len(f) < 4:
        return {"raw": line.decode("ascii", "replace")}
    hexs = f[3].decode("ascii", "replace")
    ram = hot = 0
    max_heat = 0
    for i in range(0, len(hexs) - 1, 2):
        try:
            b = int(hexs[i:i + 2], 16)
        except ValueError:
            continue
        if b >> 6:
            ram += 1
        heat = b & 63
        if heat:
            hot += 1
        if heat > max_heat:
            max_heat = heat
    return {"layers": int(f[1]), "experts": int(f[2]), "hex_len": len(hexs),
            "ram_experts": ram, "hot_experts": hot, "max_heat": max_heat,
            "sha16": hashlib.sha256(hexs.encode("ascii")).hexdigest()[:16]}


def run_serve_once(name, env_overrides, args, rep, label):
    rec = {
        "kind": "serve",
        "label": label,
        "config": name,
        "env": dict(env_overrides),
        "rep": rep,
        "cap": args.cap,
        "bits": int(BITS),
        "max_tokens": args.max_tokens,
        "ctx": int(CTX),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    env = dict(os.environ)
    env.update(env_overrides)
    env["SNAP"] = SNAP
    env["SERVE"] = "1"
    env["CTX"] = CTX
    argv = [BIN, str(args.cap), BITS]
    rec["cmd"] = cmd_line(env_overrides, {"SERVE": "1", "CTX": CTX},
                          ["./olmoe", str(args.cap), BITS])

    prompt = open(PROMPT_PATH, "rb").read()
    hdr = ("SUBMIT bench 0 %d %d 0.0 1.0\n" % (len(prompt), args.max_tokens)).encode("ascii")
    rec["submit_header"] = hdr.decode("ascii").rstrip("\n")

    if os.path.exists(PIN_FILE):
        os.remove(PIN_FILE)

    stderr_file = tempfile.TemporaryFile()
    p = None
    reader = None
    error = None
    warnings = []
    phase = "spawn"
    t_spawn = time.time()
    deadline = t_spawn + float(args.timeout)
    t_submit = None
    t_first_data = None
    t_last = None
    step_times = []
    data_bytes = bytearray()
    n_data = 0
    done = prof = hits = tiers = emap = engine_error = ready_stat = None
    post = {"tiers": False, "emap": False}

    try:
        p = subprocess.Popen(argv, cwd=C_DIR, env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=stderr_file,
                             bufsize=0, start_new_session=True)
        reader = FrameReader(p.stdout.fileno())

        phase = "ready"
        while True:
            line = reader.read_line(deadline)
            if line is None:
                raise EngineGone("engine exited before READY")
            if b"READY" in line:
                break
        rec["ready"] = True
        rec["startup_s"] = round(time.time() - t_spawn, 3)

        phase = "submit"
        t_submit = time.time()
        p.stdin.write(hdr + prompt + b"\n")
        p.stdin.flush()

        phase = "frames"
        idle_deadline = None
        while True:
            if done is not None and prof is not None and hits is not None \
                    and post["tiers"] and post["emap"]:
                break
            dl = deadline if idle_deadline is None else min(deadline, idle_deadline)
            try:
                line = reader.read_line(dl)
            except RunTimeout:
                now = time.time()
                if idle_deadline is not None and now < deadline:
                    warnings.append("post-DONE drain idle timeout; final TIERS/EMAP missing")
                    break
                raise
            if line is None:
                if done is None:
                    raise EngineGone("engine exited before DONE (n_data=%d)" % n_data)
                warnings.append("engine exited before final TIERS/EMAP")
                break

            if line.startswith(b"DATA "):
                fields = line.split()
                n = int(fields[2])
                payload = reader.read_exact(n, dl)
                now = time.time()
                reader.read_exact(1, dl)
                if t_first_data is None:
                    t_first_data = now
                if t_last is not None:
                    step_times.append((now - t_last) * 1000.0)
                t_last = now
                data_bytes += payload
                n_data += 1
                continue

            if line.startswith(b"DONE "):
                m = re.match(rb"DONE (\S+) STAT (\d+) ([\d.]+) ([\d.]+) ([\d.]+) (\d+) (\d+)", line)
                if m:
                    done = {"tokens": int(m.group(2)), "tok_s": float(m.group(3)),
                            "hit_pct": float(m.group(4)), "rss_gb": float(m.group(5)),
                            "prompt_tokens": int(m.group(6)), "limited": int(m.group(7))}
                idle_deadline = time.time() + GRID_IDLE_S
                continue

            if line.startswith(b"PROF "):
                f = line.split()
                if len(f) >= 10:
                    prof = {"dt_s": float(f[1]), "np": int(f[2]), "gen": int(f[3]),
                            "disk_s": float(f[4]), "wait_s": float(f[5]),
                            "matmul_s": float(f[6]), "attn_s": float(f[7]),
                            "head_s": float(f[8]), "forwards": int(f[9])}
                if idle_deadline is not None:
                    idle_deadline = time.time() + GRID_IDLE_S
                continue

            if line.startswith(b"HITS "):
                f = line.split()
                if len(f) >= 3:
                    hits = {"rows": int(f[1]), "cols": int(f[2])}
                    if len(f) >= 4:
                        hexs = f[3].decode("ascii", "replace")
                        hits["hex_len"] = len(hexs)
                        hits["routed_experts"] = popcount_hex(hexs)
                if idle_deadline is not None:
                    idle_deadline = time.time() + GRID_IDLE_S
                continue

            if line.startswith(b"TIERS "):
                tiers = parse_tiers(line)
                if done is not None:
                    post["tiers"] = True
                    idle_deadline = time.time() + GRID_IDLE_S
                continue

            if line.startswith(b"EMAP "):
                emap = parse_emap(line)
                if done is not None:
                    post["emap"] = True
                    idle_deadline = time.time() + GRID_IDLE_S
                continue

            if line.startswith(b"ERROR "):
                engine_error = line.decode("utf-8", "replace")
                if idle_deadline is not None:
                    idle_deadline = time.time() + GRID_IDLE_S
                continue

            if line.startswith(b"STAT "):
                ready_stat = line.decode("utf-8", "replace")
                continue

            if idle_deadline is not None:
                idle_deadline = time.time() + GRID_IDLE_S

        rec["run_wall_s"] = round(time.time() - t_submit, 3)

        phase = "shutdown"
        try:
            p.stdin.close()
        except Exception:
            pass
        try:
            rc = p.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            warnings.append("engine did not exit within 30s of stdin close")
            _terminate(p)
            rc = p.returncode
        rec["exit_code"] = rc

    except RunTimeout as exc:
        error = "timeout(%s): %s" % (phase, exc)
    except EngineGone as exc:
        error = "%s (phase=%s)" % (exc, phase)
    except Exception as exc:
        error = "%s: %s (phase=%s)" % (type(exc).__name__, exc, phase)
    finally:
        _terminate(p)
        if error:
            rec["stderr_tail"] = _read_tail(stderr_file)
        stderr_file.close()

    rec["n_data"] = n_data
    rec["n_steps"] = len(step_times)
    rec["out_len"] = len(data_bytes)
    rec["sha16"] = hashlib.sha256(bytes(data_bytes)).hexdigest()[:16]
    rec["out_sha"] = rec["sha16"]
    rec["wall_s"] = round(time.time() - t_spawn, 3)
    if t_first_data is not None and t_submit is not None:
        rec["prefill_s"] = round(t_first_data - t_submit, 3)

    st = sorted(step_times)
    if st:
        mean_ms = sum(st) / len(st)
        rec["mean_ms"] = round(mean_ms, 3)
        rec["decode_tps"] = round(1000.0 / mean_ms, 4)
        rec["p50_ms"] = round(percentile(st, 50), 3)
        rec["p90_ms"] = round(percentile(st, 90), 3)
        rec["p99_ms"] = round(percentile(st, 99), 3)
        rec["max_ms"] = round(st[-1], 3)
    if prof and step_times:
        rec["disk_ms_per_step"] = round(prof["disk_s"] * 1000.0 / len(step_times), 3)

    rec["ready_stat"] = ready_stat
    rec["done"] = done
    rec["prof"] = prof
    rec["hits"] = hits
    rec["tiers"] = tiers
    rec["emap"] = emap
    rec["engine_error"] = engine_error
    rec["warnings"] = warnings
    rec["error"] = error
    rec["ok"] = (error is None and done is not None and engine_error is None
                 and len(step_times) >= 1)
    if not rec["ok"] and error is None and engine_error is None:
        rec["error"] = "incomplete: done=%s steps=%d" % (done is not None, len(step_times))
    return rec


def run_standalone_once(name, env_overrides, args, label):
    rec = {
        "kind": "standalone",
        "label": label,
        "config": name,
        "env": dict(env_overrides),
        "cap": args.cap,
        "bits": int(BITS),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    env = dict(os.environ)
    env.update(env_overrides)
    env["SNAP"] = SNAP
    env.pop("SERVE", None)
    ref_rel = os.path.relpath(REF_PATH, C_DIR)
    argv = [BIN, str(args.cap), BITS, ref_rel]
    rec["cmd"] = cmd_line(env_overrides, {},
                          ["./olmoe", str(args.cap), BITS, ref_rel])

    if os.path.exists(PIN_FILE):
        os.remove(PIN_FILE)

    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=C_DIR, env=env, capture_output=True,
                           text=True, timeout=float(args.timeout))
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        rec["wall_s"] = round(time.time() - t0, 3)
        rec["exit_code"] = p.returncode
        m = re.search(r"Matching tokens: (\d+)/(\d+)", out)
        if m:
            rec["match"] = int(m.group(1))
            rec["n_new"] = int(m.group(2))
            rec["exact"] = (rec["match"] == rec["n_new"] == 200)
        m = re.search(r"Expert cache hit rate: ([\d.]+)%\s*\(hit=(\d+) miss=(\d+)\)", out)
        if m:
            rec["hit_pct"] = float(m.group(1))
            rec["hits"] = int(m.group(2))
            rec["misses"] = int(m.group(3))
        m = re.search(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)", out)
        if m:
            rec["tok_s"] = float(m.group(1))
            rec["seconds"] = float(m.group(2))
        m = re.search(r"PEAK RSS: ([\d.]+) GB", out)
        if m:
            rec["peak_rss_gb"] = float(m.group(1))
        if p.returncode != 0:
            rec["error"] = "exit code %d" % p.returncode
            rec["tail"] = out[-500:]
        elif "match" not in rec:
            rec["error"] = "no 'Matching tokens' line in output"
            rec["tail"] = out[-500:]
    except subprocess.TimeoutExpired:
        rec["wall_s"] = round(time.time() - t0, 3)
        rec["error"] = "timeout after %.0fs" % float(args.timeout)
    except Exception as exc:
        rec["wall_s"] = round(time.time() - t0, 3)
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
    return rec


def _brief(rec):
    kind = rec.get("kind")
    cfg = rec.get("config")
    if kind == "serve":
        if rec.get("ok"):
            d = rec.get("done") or {}
            return ("[serve] %-10s rep %d  decode %8.3f tok/s  p50 %6.1f  p90 %6.1f  "
                    "p99 %7.1f ms  hit %5.1f%%  disk/step %6.1f ms  sha %s  wall %6.1fs"
                    % (cfg, rec.get("rep", 0), rec.get("decode_tps") or 0.0,
                       rec.get("p50_ms") or 0.0, rec.get("p90_ms") or 0.0,
                       rec.get("p99_ms") or 0.0, d.get("hit_pct") or 0.0,
                       rec.get("disk_ms_per_step") or 0.0, rec.get("sha16") or "-",
                       rec.get("wall_s") or 0.0))
        return "[serve] %-10s rep %d  FAILED: %s" % (cfg, rec.get("rep", 0),
                                                     rec.get("error") or "incomplete")
    if kind == "standalone":
        if rec.get("exact"):
            return ("[gold]  %-10s %s/%s exact  hit %5.1f%%  %6.2f tok/s  wall %6.1fs"
                    % (cfg, rec.get("match"), rec.get("n_new"), rec.get("hit_pct") or 0.0,
                       rec.get("tok_s") or 0.0, rec.get("wall_s") or 0.0))
        return "[gold]  %-10s FAILED: %s" % (
            cfg, rec.get("error") or ("%s/%s tokens" % (rec.get("match"), rec.get("n_new"))))
    return json.dumps(rec)


def build_summary(args, configs, serve_records, standalone_by_cfg, jsonl_path,
                  summary_path, invoke):
    by_cfg = {}
    for name, _env in configs:
        by_cfg[name] = [r for r in serve_records if r.get("config") == name]
    prompt_len = os.path.getsize(PROMPT_PATH)
    rel = lambda p: os.path.relpath(p, ROOT_DIR)

    L = []
    L.append("# OLMoE tier A/B benchmark: `%s`" % args.label)
    L.append("")
    L.append("- date: %s" % time.strftime("%Y-%m-%dT%H:%M:%S"))
    L.append("- engine: `%s` (argv `%d %s`), SNAP: `%s`" % (rel(BIN), args.cap, BITS, SNAP))
    L.append("- serve env: `SERVE=1 CTX=%s`; config env passed through as-is" % CTX)
    L.append("- reps: %d; warmup: %d discarded serve run(s) of `%s`; per-run timeout: %.0fs"
             % (args.reps, args.warmup, configs[0][0], args.timeout))
    L.append("- primary metric: decode-only tok/s = 1000 / mean(DATA frame delta), "
             "the %d deltas after the first DATA" % (args.max_tokens - 1))
    L.append("- interleaving: rep-major (`for rep in 1..%d: for config in configs`), "
             "so page-cache/thermal drift cancels" % args.reps)
    L.append("- JSONL: `%s`" % rel(jsonl_path))
    L.append("")

    table = []
    table.append("| config | decode tok/s (med) | P50 ms | P90 ms | P99 ms | hit% | "
                 "disk ms/step | SHA-256(16) | standalone exact |")
    table.append("|---|---|---|---|---|---|---|---|---|")
    notes = []
    for name, _env in configs:
        recs = by_cfg[name]
        ok = [r for r in recs if r.get("ok")]
        dtps = _median([r.get("decode_tps") for r in ok])
        p50 = _median([r.get("p50_ms") for r in ok])
        p90 = _median([r.get("p90_ms") for r in ok])
        p99 = _median([r.get("p99_ms") for r in ok])
        hit = _median([(r.get("done") or {}).get("hit_pct") for r in ok])
        disk = _median([r.get("disk_ms_per_step") for r in ok])
        shas = sorted({r.get("sha16") for r in ok if r.get("sha16")})
        if len(shas) == 1:
            sha = shas[0]
        elif len(shas) > 1:
            sha = "MIXED(%d)" % len(shas)
            notes.append("- `%s`: DATA SHA differs across reps: %s" % (name, ", ".join(shas)))
        else:
            sha = "n/a"
        st = standalone_by_cfg.get(name) or {}
        if st.get("exact"):
            exact = "%d/%d OK" % (st.get("match"), st.get("n_new"))
        elif st.get("match") is not None:
            exact = "%d/%d FAIL" % (st.get("match"), st.get("n_new"))
        elif st.get("error"):
            exact = "FAIL (%s)" % str(st["error"])[:24]
        else:
            exact = "n/a"
        table.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, fmt(dtps), fmt(p50, 1), fmt(p90, 1), fmt(p99, 1), fmt(hit, 1),
            fmt(disk, 1), sha, exact))
        nfail = len(recs) - len(ok)
        if nfail:
            reasons = sorted({(r.get("error") or "incomplete") for r in recs if not r.get("ok")})
            notes.append("- `%s`: %d/%d serve runs failed (%s)"
                         % (name, nfail, len(recs), "; ".join(reasons)))
        if st and st.get("exact") is False:
            notes.append("- `%s`: standalone golden check not exact: %s/%s"
                         % (name, st.get("match"), st.get("n_new")))
    L.extend(table)
    if notes:
        L.append("")
        L.extend(notes)

    L.append("")
    L.append("## Standalone golden check (`./olmoe %d %s bench_tier/ref200.json`)" % (args.cap, BITS))
    L.append("")
    L.append("| config | match | hit% | tok/s | wall s |")
    L.append("|---|---|---|---|---|")
    for name, _env in configs:
        st = standalone_by_cfg.get(name) or {}
        match = ("%s/%s" % (st.get("match"), st.get("n_new"))) if st.get("match") is not None else "n/a"
        L.append("| %s | %s | %s | %s | %s |" % (
            name, match, fmt(st.get("hit_pct"), 1), fmt(st.get("tok_s"), 2),
            fmt(st.get("wall_s"), 1)))

    L.append("")
    L.append("## Exact commands")
    L.append("")
    L.append("Harness invocation:")
    L.append("")
    L.append("```")
    L.append(invoke)
    L.append("```")
    L.append("")
    L.append("Before every run (serve and standalone): `rm -f %s`" % PIN_FILE)
    L.append("")
    L.append("Serve run (once per config per rep; cwd `%s`):" % rel(C_DIR))
    L.append("")
    L.append("```")
    for name, env in configs:
        L.append("# %s" % name)
        L.append(cmd_line(env, {"SERVE": "1", "CTX": CTX}, ["./olmoe", str(args.cap), BITS]))
        L.append("stdin: %s + payload %s (%d bytes) + LF"
                 % ("SUBMIT bench 0 %d %d 0.0 1.0" % (prompt_len, args.max_tokens),
                    rel(PROMPT_PATH), prompt_len))
    L.append("```")
    L.append("")
    L.append("Standalone golden check (once per config, after the serve matrix):")
    L.append("")
    L.append("```")
    for name, env in configs:
        L.append("# %s" % name)
        L.append(cmd_line(env, {},
                          ["./olmoe", str(args.cap), BITS, "bench_tier/ref200.json"]))
    L.append("```")

    L.append("")
    L.append("## Per-run detail")
    L.append("")
    L.append("| run | config | rep | decode tok/s | n DATA | SHA-256(16) | status |")
    L.append("|---|---|---|---|---|---|---|")
    for r in serve_records:
        status = "ok" if r.get("ok") else (r.get("error") or "incomplete")
        L.append("| serve | %s | %s | %s | %s | %s | %s |" % (
            r.get("config"), r.get("rep"), fmt(r.get("decode_tps")), r.get("n_data"),
            r.get("sha16") or "n/a", status))
    for name, _env in configs:
        st = standalone_by_cfg.get(name) or {}
        status = "exact" if st.get("exact") else (st.get("error") or "not exact")
        L.append("| gold | %s | - | %s | - | - | %s |" % (name, fmt(st.get("tok_s")), status))

    text = "\n".join(L) + "\n"
    with open(summary_path, "w") as f:
        f.write(text)
    return "\n".join(table), text


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bench_ab.py",
        description="Interleaved A/B benchmark harness for OLMoE tier-cache/PILOT work "
                    "(serve mode + standalone golden check, cap=8 slots/layer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="config syntax:\n"
               "  --configs 'name:KEY=VAL|KEY=VAL,name2:KEY=VAL,...'\n"
               "  a bare name matching a built-in default reuses that default's env;\n"
               "  keys are env var names passed through to the engine as-is\n"
               "  (PILOT, HOT, RECENT_RING, RECENT_BLOOM, MARKOV, ...).\n"
               "  default set: baseline (all off), phase1 (RECENT_RING+RECENT_BLOOM),\n"
               "  phase12 (phase1 + MARKOV). Quote the spec: it contains '|'.\n"
               "\n"
               "examples:\n"
               "  python3 c/bench_tier/bench_ab.py\n"
               "  python3 c/bench_tier/bench_ab.py --reps 5 --label ab2\n"
               "  python3 c/bench_tier/bench_ab.py --configs 'baseline,phase1,phase12,pilot2:PILOT=2|HOT=0|RECENT_RING=1|RECENT_BLOOM=1|MARKOV=1'\n")
    ap.add_argument("--configs", default=None,
                    help="comma-separated 'name:KEY=VAL|KEY=VAL' list (default: built-in set)")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per config (default 3)")
    ap.add_argument("--label", default="ab", help="output label (default ab)")
    ap.add_argument("--cap", type=int, default=8, help="expert cache slots/layer (default 8)")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=200,
                    help="tokens to generate per serve request (default 200)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded serve runs of the first config (default 1)")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="per-run wall-clock timeout in seconds (default 600)")
    args = ap.parse_args(argv)

    if args.reps < 1:
        ap.error("--reps must be >= 1")
    if args.cap < 1:
        ap.error("--cap must be >= 1")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be >= 1")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.timeout <= 0:
        ap.error("--timeout must be > 0")

    configs = parse_configs(args.configs)
    raw_argv = argv if argv is not None else sys.argv[1:]
    invoke = "python3 c/bench_tier/bench_ab.py " + " ".join(shlex.quote(a) for a in raw_argv)
    jsonl_path = os.path.join(BENCH_DIR, "ab_%s.jsonl" % args.label)
    summary_path = os.path.join(BENCH_DIR, "ab_%s_summary.md" % args.label)

    if not os.path.exists(BIN):
        print("warning: %s not found (engine not built?)" % BIN, file=sys.stderr)

    print("configs: %s" % ", ".join(n for n, _ in configs))
    print("reps: %d  cap: %d  max_tokens: %d  warmup: %d  timeout: %.0fs"
          % (args.reps, args.cap, args.max_tokens, args.warmup, args.timeout))

    for i in range(args.warmup):
        w = run_serve_once(configs[0][0], configs[0][1], args, 0, args.label)
        print("warmup %d/%d (discarded): %s" % (i + 1, args.warmup, _brief(w)))

    serve_records = []
    standalone_by_cfg = {}
    with open(jsonl_path, "a") as fout:
        for rep in range(1, args.reps + 1):
            for name, env in configs:
                rec = run_serve_once(name, env, args, rep, args.label)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                serve_records.append(rec)
                print(_brief(rec))
        for name, env in configs:
            rec = run_standalone_once(name, env, args, args.label)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            standalone_by_cfg[name] = rec
            print(_brief(rec))

    table, _text = build_summary(args, configs, serve_records, standalone_by_cfg,
                                 jsonl_path, summary_path, invoke)
    print("")
    print(table)
    print("")
    print("jsonl:   %s" % os.path.relpath(jsonl_path, ROOT_DIR))
    print("summary: %s" % os.path.relpath(summary_path, ROOT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
