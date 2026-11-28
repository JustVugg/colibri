"""Turns the output of bench.sh into raw evidence and two experiment manifests.

  python3 summarize.py RUN_DIR OUT_DIR --base-commit SHA --commit SHA --command TEXT \
      --model TEXT --cpu TEXT --ram TEXT --storage TEXT --os TEXT --mirrors TEXT \
      [--uri-prefix PATH] [--shorten DIR=TEXT ...]

RUN_DIR holds <arm>-<round>.out/.err/.disks from bench.sh. OUT_DIR gets, for each
arm, <arm>-raw.txt (the engine's summary lines, the drive bytes and the SHA-256 of
the output of every run), summary.md (all arms in RUN_DIR) and two manifests:
  manifest.json         pr-off against pr-mirror (COLI_MODEL_MIRROR)
  engine-manifest.json  base-off against pr-off (the engine, both without a mirror)
--shorten replaces a local directory in the kept lines with TEXT (for example the
model directory in the "[K3 MIRROR]" lines). Each raw file names every replacement
in its first line.
The quality gate passes when every run of these three arms generated the same
text (stdout up to the timing line). tok_s is N / X from the engine's line "[K3] decode N tokens in Xs (Y tok/s)"
(Y itself is rounded to 0.01).
"""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

ARMS = ("base-off", "pr-off", "pr-mirror")
DECODE = re.compile(r"^\[K3\] decode (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)", re.M)
PREFILL = re.compile(r"^\[K3\] prefill done in ([\d.]+)s \(([\d.]+) tok/s\)", re.M)
KEEP = re.compile(r"^\[K3( MIRROR)?\] (decode|prefill done|time:|reads:|\d+ drives|/)", re.M)


def runs(run_dir, arm, shorten):
    out = []
    for err in sorted(run_dir.glob(f"{arm}-[0-9]*.err"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        stem = err.with_suffix("")
        text = err.read_text(errors="replace")
        m = DECODE.search(text)
        if not m:
            raise SystemExit(f"{err}: no decode line")
        p = PREFILL.search(text)
        out.append({
            "name": stem.name,
            "tok_s": round(int(m.group(1)) / float(m.group(2)), 4),   # the line rounds to 0.01
            "prefill_tok_s": float(p.group(2)) if p else None,
            "lines": [replaced(line, shorten) for line in text.splitlines() if KEEP.match(line)],
            "disks": stem.with_suffix(".disks").read_text().strip(),
            "out_sha256": hashlib.sha256(generated(stem.with_suffix(".out"))).hexdigest(),
        })
    if len(out) < 3:
        raise SystemExit(f"{arm}: {len(out)} runs, the validator wants at least 3")
    return out


def replaced(line, shorten):
    for old, new in shorten:
        line = line.replace(old, new)
    return line


def generated(path):
    """The generated text: stdout up to the engine's "TUNE decode:" timing line,
    which follows the text on the same line."""
    return path.read_bytes().split(b"TUNE decode:")[0]


def raw_file(out_dir, arm, data, shorten):
    lines = []
    if shorten:
        lines.append("# directories replaced by summarize.py --shorten: " + ", ".join(new for _, new in shorten))
    for run in data:
        lines.append(f"== {run['name']}")
        lines += run["lines"]
        lines.append(f"drive bytes: {run['disks']}")
        lines.append(f"output sha256: {run['out_sha256']}")
    path = out_dir / f"{arm}-raw.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def side(config, data, quality, path, uri):
    speeds = [r["tok_s"] for r in data]
    return {
        "config": config,
        "samples": {"tok_s": speeds},
        "median_tok_s": statistics.median(speeds),
        "quality": quality,
        "evidence": {"uri": uri, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }


def outcome(base, trial):
    b, t = statistics.median(base), statistics.median(trial)
    if max(base) < min(trial):
        return "improvement"
    if max(trial) < min(base) and t < 0.97 * b:
        return "regression"
    return "no-change"


def summary(out_dir, every, data):
    """summary.md: median decode and prefill rate, GB per drive, same output as the PR arms."""
    ref = data["pr-off"][0]["out_sha256"]
    lines = ["| arm | decode tok/s (runs) | median | prefill tok/s median | GB read per drive (median run) | output |",
             "|---|---|---|---|---|---|"]
    for arm, rows in every.items():
        speeds = [r["tok_s"] for r in rows]
        pre = [r["prefill_tok_s"] for r in rows if r["prefill_tok_s"]]
        mid = sorted(rows, key=lambda r: r["tok_s"])[len(rows) // 2]
        disks = mid["disks"].split()
        n = len(disks) // 2
        gb = " / ".join(f"{int(b) / 1e9:.0f}" for b in disks[n:])
        same = "same" if all(r["out_sha256"] == ref for r in rows) else "differs"
        lines.append(f"| {arm} | {', '.join(f'{v:.3f}' for v in speeds)} | {statistics.median(speeds):.3f} | "
                     f"{statistics.median(pre) if pre else 0:.2f} | {gb} | {same} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--uri-prefix", default="docs/experiments/kimi-k3-mirror-2026-11-27/")
    ap.add_argument("--shorten", action="append", default=[], metavar="DIR=TEXT")
    for key in ("base-commit", "commit", "command", "model", "cpu", "ram", "storage", "os", "mirrors"):
        ap.add_argument(f"--{key}", required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    shorten = [tuple(item.split("=", 1)) for item in a.shorten]
    if any(len(pair) != 2 or not pair[0] for pair in shorten):
        raise SystemExit("--shorten wants DIR=TEXT")
    names = sorted({p.stem.rsplit("-", 1)[0] for p in a.run_dir.glob("*-[0-9]*.err")})
    every = {arm: runs(a.run_dir, arm, shorten) for arm in names}
    data = {arm: every[arm] for arm in ARMS}
    summary(a.out_dir, every, data)
    outputs = {r["out_sha256"] for arm in ARMS for r in data[arm]}
    method = ("byte-identical output: all {} runs of the three arms wrote the same text "
              "(greedy, COLI_TEMP=0, 64 tokens), output sha256 {}").format(
                  sum(len(v) for v in data.values()), next(iter(outputs)))
    quality = {"method": method, "passed": len(outputs) == 1}
    paths = {arm: raw_file(a.out_dir, arm, data[arm], shorten) for arm in ARMS}
    prompt = (a.run_dir / "prompt.txt").read_bytes()
    common = {
        "version": 1,
        "model": a.model,
        "command": a.command,
        "prompt_hash": hashlib.sha256(prompt).hexdigest(),
        "hardware": {"cpu": a.cpu, "ram": a.ram, "storage": a.storage, "os": a.os},
        "warmup_runs": 0,
    }

    def uri(arm):
        return a.uri_prefix + paths[arm].name

    mirror = dict(common)
    mirror.update({
        "hypothesis": "Reading the routed experts of each batch from byte-identical copies on two more "
                      "drives (round-robin by weight) raises kimi_k3's expert read rate, and so its decode "
                      "rate, above what the one drive that holds a layer's shard delivers.",
        "commit": a.commit,
        "changed_variables": ["COLI_MODEL_MIRROR"],
        "baseline": side({"COLI_DISK_WEIGHTS": "1,1,1"}, data["pr-off"], quality, paths["pr-off"], uri("pr-off")),
        "trial": side({"COLI_DISK_WEIGHTS": "1,1,1", "COLI_MODEL_MIRROR": a.mirrors}, data["pr-mirror"], quality,
                      paths["pr-mirror"], uri("pr-mirror")),
    })
    mirror["outcome"] = outcome(mirror["baseline"]["samples"]["tok_s"], mirror["trial"]["samples"]["tok_s"])
    off = dict(common)
    off.update({
        "hypothesis": "Without COLI_MODEL_MIRROR, the engine of this branch decodes at the rate of its base "
                      "commit: every read takes drive 0, whose descriptors are the primary's own.",
        "commit": a.commit,
        "changed_variables": ["engine"],
        "baseline": side({"engine": f"kimi_k3 at {a.base_commit}", "COLI_DISK_WEIGHTS": "1,1,1"},
                         data["base-off"], quality, paths["base-off"], uri("base-off")),
        "trial": side({"engine": f"kimi_k3 at {a.commit}", "COLI_DISK_WEIGHTS": "1,1,1"},
                      data["pr-off"], quality, paths["pr-off"], uri("pr-off")),
    })
    off["outcome"] = outcome(off["baseline"]["samples"]["tok_s"], off["trial"]["samples"]["tok_s"])
    for name, record in (("manifest.json", mirror), ("engine-manifest.json", off)):
        (a.out_dir / name).write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: {record['outcome']}, median {record['baseline']['median_tok_s']} -> "
              f"{record['trial']['median_tok_s']} tok/s, quality {quality['passed']}")


if __name__ == "__main__":
    main()
