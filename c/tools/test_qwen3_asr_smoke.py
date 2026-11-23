#!/usr/bin/env python3
"""Smoke/oracle check for qwen3_asr.

Runs the C engine on the hello.wav oracle case and compares transcript.
Exits 0 on pass, 1 on mismatch.
"""
import argparse, base64, json, os, subprocess, sys, time

def norm(s):
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s).split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./c/qwen3_asr")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", default="/tmp/opencode/bench_wavs/hello.wav")
    ap.add_argument("--oracle", default="c/testdata/qwen3_asr_hello/transcript.txt")
    ap.add_argument("--manifest")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    def R(p):
        if os.path.isabs(p): return p
        if os.path.exists(p): return os.path.abspath(p)
        return os.path.join(repo, p)
    args.engine = R(args.engine)
    args.wav = R(args.wav)
    args.oracle = R(args.oracle)
    if args.manifest:
        args.manifest = R(args.manifest)
        cases = json.load(open(args.manifest))
        for c in cases: c["wav"] = R(c["wav"])
    else:
        expected = open(args.oracle).read().strip() if os.path.exists(args.oracle) else None
        cases = [{"wav": args.wav, "ref": expected}]
    cmd = [args.engine, "--model", args.model, "--serve"]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ready = False
    for _ in range(1200):
        line = p.stdout.readline()
        if b"READY" in line: ready = True; break
        if not line: break
    load_s = time.time() - t0
    if not ready:
        sys.stderr.write(p.stderr.read().decode(errors='ignore')[-2000:])
        print("FAIL: engine not READY")
        return 1
    results = []; all_ok = True
    for i, case in enumerate(cases):
        if not os.path.exists(case["wav"]):
            results.append({"wav": case["wav"], "ok": False, "exact_match": False, "normalized_match": False, "wall_s": 0, "text": None, "expected": case.get("ref"), "error": "missing wav"})
            all_ok = False
            continue
        audio_b64 = base64.b64encode(open(case["wav"], "rb").read()).decode()
        payload = json.dumps({"audio_b64": audio_b64}).encode()
        req = f"TRANSCRIBE t{i} 0 {len(payload)}\n".encode() + payload + b"\n"
        t0 = time.time(); p.stdin.write(req); p.stdin.flush()
        text = None; err = None
        while True:
            line = p.stdout.readline()
            if not line: err = "eof"; break
            if line.startswith(f"TRANSCRIPT t{i} ".encode()):
                n = int(line.split()[-1]); pl = p.stdout.read(n); p.stdout.readline()
                text = json.loads(pl.decode()).get("text", "")
            if line.startswith(f"DONE t{i} ".encode()): break
            if line.startswith(f"ERROR t{i} ".encode()): err = line.decode(errors='ignore').strip(); break
        wall = time.time() - t0
        expected = case.get("ref")
        ok = err is None and text is not None
        exact = ok and expected is not None and text.strip() == expected
        norm_ok = ok and expected is not None and norm(text) == norm(expected)
        all_ok = all_ok and (norm_ok or expected is None)
        results.append({"wav": case["wav"], "ok": ok, "exact_match": exact, "normalized_match": norm_ok, "wall_s": round(wall,3), "text": text, "expected": expected, "error": err})
    p.terminate()
    try: p.wait(timeout=5)
    except Exception: p.kill()
    print(json.dumps({"ok": all_ok, "load_s": round(load_s,3), "results": results}, indent=2))
    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
