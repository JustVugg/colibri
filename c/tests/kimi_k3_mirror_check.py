"""kimi_k3: the routed experts read from COLI_MODEL_MIRROR copies (mirror_policy.h).

Runs the engine on the tiny fixture with copies of its shard in temporary
mirror directories, and checks:
  * the tokens are ref.json's (Moonshot's own modeling code), the same as the
    run without a mirror, with one mirror (weights unset and 1,3), with two
    (1,1,1), through the loader pool (K3_PIPE=1) and the OpenMP reads
    (K3_PIPE=0), O_DIRECT and buffered (K3_DIRECT=0);
  * every drive serves reads (the closing "[K3 MIRROR] reads:" line), and the
    round-robin of the O_DIRECT reads gives each drive its share of all reads
    to within one read, whatever the size of each batch;
  * a copy whose shard differs from the primary's (one header byte) is
    skipped, the model directory in the list is ignored, the rest serves;
  * COLI_DISK_WEIGHTS with the wrong count, a zero or junk gives a message,
    and the probe sets the split instead (the tokens do not change);
  * the copies are never written, and a run without COLI_MODEL_MIRROR prints
    no mirror line.

  python3 tests/kimi_k3_mirror_check.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHARD = "model.safetensors"
ROUND = 240   # mirror_policy.h: MIRROR_ROUND


def run(binary, fixture, case, engine_env, **env):
    full = dict(os.environ, OMP_NUM_THREADS="2", **engine_env, **env)
    for key in ("COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS", "K3_PIPE", "K3_DIRECT"):
        if key not in env:
            full.pop(key, None)
    p = subprocess.run([str(binary), str(fixture), "--ids", " ".join(map(str, case["prompt_ids"])),
                        "--ngen", str(case["max_new_tokens"])],
                       env=full, capture_output=True, text=True, timeout=600)
    tokens = [int(t) for t in p.stdout.split("TUNE")[0].split()] if p.returncode == 0 else None
    return tokens, p.stderr


def reads(stderr):
    """[reads per drive] from the closing line, or None."""
    line = re.search(r"^\[K3 MIRROR\] reads:(.*)$", stderr, re.M)
    return [int(n) for n in re.findall(r"(?:primary|mirror\d+) (\d+) \(", line.group(1))] if line else None


def shares(weights):
    """The round-robin shares that mirror_cuts gives these weights, written again here."""
    total, total_so_far, prev, out = sum(weights), 0.0, 0, []
    n = len(weights)
    for r, w in enumerate(weights):
        total_so_far += w
        cut = ROUND if r == n - 1 else int(ROUND * total_so_far / total + 0.5)
        cut = min(max(cut, prev + 1), ROUND - (n - 1 - r))
        out.append(cut - prev)
        prev = cut
    return out


def even(counts, weights):
    """Each drive within one read of its share of all reads (the round-robin)."""
    total = sum(counts)
    return all(abs(c - total * s / ROUND) <= 1.0 + 1e-9 for c, s in zip(counts, shares(weights)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--fixture", required=True, type=Path, help="kimi_k3_tiny (one shard, ref.json)")
    a = ap.parse_args()
    binary, fixture = a.binary.resolve(), a.fixture.resolve()
    ref = json.loads((fixture / "ref.json").read_text())
    engine_env = ref["engine_env"]
    failures = 0

    def check(ok, what):
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        failures += not ok

    tmp = Path(tempfile.mkdtemp())
    try:
        m1, m2, m3, empty = tmp / "m1", tmp / "m2", tmp / "m3", tmp / "empty"
        for m in (m1, m2, m3):
            m.mkdir()
            shutil.copyfile(fixture / SHARD, m / SHARD)
        empty.mkdir()
        bad = bytearray((m3 / SHARD).read_bytes())   # one header byte (the padding) differs
        bad[8 + int.from_bytes(bad[:8], "little") - 1] ^= 0x01
        (m3 / SHARD).write_bytes(bytes(bad))
        two = f"{m1};{m2}"
        # (what, env, drives, the weights the round-robin follows or None for the hash).
        # The fixture's shard is smaller than the probe reads, so unset weights give an even split.
        cases = [
            ("one mirror, weights unset", {"COLI_MODEL_MIRROR": str(m1)}, 2, [1, 1]),
            ("one mirror, COLI_DISK_WEIGHTS=1,3",
             {"COLI_MODEL_MIRROR": str(m1), "COLI_DISK_WEIGHTS": "1,3"}, 2, [1, 3]),
            ("two mirrors, COLI_DISK_WEIGHTS=1,1,1",
             {"COLI_MODEL_MIRROR": two, "COLI_DISK_WEIGHTS": "1,1,1"}, 3, [1, 1, 1]),
            ("two mirrors, ',' between them, OpenMP reads (K3_PIPE=0)",
             {"COLI_MODEL_MIRROR": f"{m1},{m2}", "K3_PIPE": "0"}, 3, [1, 1, 1]),
            ("two mirrors, buffered reads (K3_DIRECT=0), one drive per expert",
             {"COLI_MODEL_MIRROR": two, "K3_DIRECT": "0"}, 3, None),
        ]
        for name, case in ref["cases"].items():
            want = case["greedy_new_ids"]
            base, err = run(binary, fixture, case, engine_env)
            if base != want:
                print(err[-2000:])
                check(False, f"{name}: the run without a mirror gives ref.json's tokens")
                continue
            check("[K3 MIRROR]" not in err, f"{name}: without COLI_MODEL_MIRROR no mirror line")
            for what, env, n, weights in cases:
                got, err = run(binary, fixture, case, engine_env, **env)
                counts = reads(err)
                ok = (got == want and counts is not None and len(counts) == n and all(counts)
                      and (weights is None or even(counts, weights)))
                check(ok, f"{name}, {what}: ref.json's tokens, reads per drive {counts}")
                if got != want:
                    print(err[-2000:])
        case = ref["cases"]["long"]
        got, err = run(binary, fixture, case, engine_env, COLI_MODEL_MIRROR=f"{m3};{empty};{fixture};{m1}")
        check(got == case["greedy_new_ids"] and "header differs" in err
              and err.count("no shard equal to the primary's: skipped") == 2
              and "is the model directory itself: ignored" in err
              and reads(err) is not None and len(reads(err)) == 2,
              "a differing copy and an empty directory are skipped, the model directory is ignored, "
              "the rest serves")
        for spec in ("1", "1,1,1", "1,0", "1,x", "9GB,3", "-1,1"):
            got, err = run(binary, fixture, case, engine_env, COLI_MODEL_MIRROR=str(m1), COLI_DISK_WEIGHTS=spec)
            check(got == case["greedy_new_ids"] and f"COLI_DISK_WEIGHTS='{spec}' is not 2 positive weights" in err
                  and reads(err) is not None,
                  f"COLI_DISK_WEIGHTS={spec} with one mirror: a message, the probe splits, same tokens")
        for m in (m1, m2):
            check(sorted(p.name for p in m.iterdir()) == [SHARD]
                  and (m / SHARD).read_bytes() == (fixture / SHARD).read_bytes(),
                  f"{m.name} holds only its unchanged shard (copies are never written)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("kimi_k3_mirror_check: " + ("ok" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
