#!/usr/bin/env python3
"""Reproducible ASR benchmark: Colibri qwen3_asr vs llama.cpp.

Reports WER, CER, RTF, wall time, audio duration, load time, and peak RSS.
Uses deterministic flite-generated WAV samples by default.

Example:
  OMP_NUM_THREADS=16 python3 c/tools/benchmark_asr.py \
    --colibri ./c/qwen3_asr --model <hf-snapshot> \
    --gguf <text.gguf> --mmproj <mmproj.gguf> \
    --llama-cli <llama-cli> --json --repeats 1
"""
from __future__ import annotations
import argparse, base64, json, os, platform, re, resource, subprocess, sys, time, wave

DEFAULT_SAMPLES = [
    ("hello", "Hello world, this is a test of speech recognition."),
    ("weather", "The weather today is sunny with a chance of rain."),
    ("numbers", "One two three four five six seven eight nine ten."),
    ("combined", "Hello world. This is a test of speech recognition. The weather today is sunny with a chance of rain. One two three four five six seven eight nine ten"),
]

def norm(t: str):
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split()

def wer(ref: str, hyp: str):
    r, h = norm(ref), norm(hyp)
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[len(r)][len(h)], len(r)

def cer(ref: str, hyp: str):
    r = "".join(norm(ref)); h = "".join(norm(hyp))
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[len(r)][len(h)], len(r)

def wav_dur(path):
    with wave.open(path, 'rb') as w: return w.getnframes()/w.getframerate()

def ensure_samples(workdir):
    os.makedirs(workdir, exist_ok=True)
    out = []
    for name, text in DEFAULT_SAMPLES:
        path = os.path.join(workdir, f"{name}.wav")
        if not os.path.exists(path):
            if name == "combined":
                # build from component samples for stable audio
                parts = [os.path.join(workdir, f"{n}.wav") for n, _ in DEFAULT_SAMPLES if n in ("hello", "weather", "numbers")]
                for p in parts:
                    if not os.path.exists(p):
                        ensure_samples(workdir)
                        break
                lst = os.path.join(workdir, "combined.txt")
                with open(lst, "w") as f:
                    for p in parts: f.write(f"file '{os.path.abspath(p)}'\n")
                subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"flite=text='{text}':voice=slt", "-ar", "16000", "-ac", "1", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out.append((name, text, path))
    return out

def proc_rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None

def bench_colibri(args, samples, env, gguf=None, mmproj=None, engine_name=None):
    cmd=[args.colibri, "--model", args.model]
    if gguf: cmd += ["--gguf", gguf]
    if mmproj: cmd += ["--mmproj", mmproj]
    cmd += ["--serve"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    ready = False
    t0 = time.time()
    for _ in range(1200):
        line = p.stdout.readline()
        if not line: break
        if b"READY" in line: ready = True; break
    load_s = time.time() - t0
    if not ready:
        err = p.stderr.read().decode(errors="ignore")[-2000:]
        p.kill(); p.wait(timeout=5)
        raise RuntimeError(f"colibri engine not READY: {err}")
    rows = []
    if engine_name is None: engine_name = "colibri-gguf" if gguf else "colibri"
    for rep in range(args.repeats):
        for si, (name, ref, wav) in enumerate(samples):
            audio_b64 = base64.b64encode(open(wav, "rb").read()).decode()
            payload = json.dumps({"audio_b64": audio_b64}).encode()
            req = f"TRANSCRIBE r{rep}_{si} 0 {len(payload)}\n".encode() + payload + b"\n"
            t0 = time.time(); p.stdin.write(req); p.stdin.flush()
            text = ""; err = None
            while True:
                line = p.stdout.readline()
                if not line: err = "eof"; break
                if line.startswith(f"TRANSCRIPT r{rep}_{si} ".encode()):
                    n = int(line.split()[-1]); pl = p.stdout.read(n); p.stdout.readline()
                    try: text = json.loads(pl.decode()).get("text", "")
                    except Exception as e: err = str(e)
                if line.startswith(f"DONE r{rep}_{si} ".encode()): break
                if line.startswith(f"ERROR r{rep}_{si} ".encode()): err = line.decode(errors="ignore").strip(); break
            dt = time.time() - t0; dur = wav_dur(wav)
            we, wn = wer(ref, text); ce, cn = cer(ref, text)
            rows.append({"engine":engine_name,"sample":name,"rep":rep,"cold":rep==0 and si==0,"wer":f"{we}/{wn}","cer":f"{ce}/{cn}","rtf":round(dt/dur,4),"wall_s":round(dt,4),"audio_s":round(dur,4),"load_s":round(load_s,4),"rss_peak_kb":proc_rss_kb(p.pid),"text":text,"error":err})
    p.terminate()
    try: p.wait(timeout=5)
    except Exception: p.kill()
    return rows, cmd

def bench_llama(args, samples, env):
    if not args.llama_cli: return [], []
    rows = []
    cmd_base=[args.llama_cli, "-m", args.gguf, "--mmproj", args.mmproj]
    for rep in range(args.repeats):
        for si, (name, ref, wav) in enumerate(samples):
            cmd = cmd_base + ["--audio", wav, "-p", "Transcribe this audio.", "-n", "128", "--no-display-prompt", "-st"]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
            dt = time.time() - t0
            text = r.stdout
            if "<asr_text>" in text: text = text.split("<asr_text>")[-1]
            for cut in ["[ Prompt:", "\n>", "Exiting"]:
                if cut in text: text = text.split(cut)[0]
            text = text.strip(); dur = wav_dur(wav)
            we, wn = wer(ref, text); ce, cn = cer(ref, text)
            # llama.cpp runs per process; ru_maxrss is children peak in KB on Linux
            rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            rows.append({"engine":"llama.cpp","sample":name,"rep":rep,"cold":True,"wer":f"{we}/{wn}","cer":f"{ce}/{cn}","rtf":round(dt/dur,4),"wall_s":round(dt,4),"audio_s":round(dur,4),"load_s":None,"rss_peak_kb":rss,"text":text,"error":None if r.returncode==0 else f"rc={r.returncode}"})
    return rows, cmd_base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colibri", default="./qwen3_asr")
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-cli")
    ap.add_argument("--gguf")
    ap.add_argument("--mmproj")
    ap.add_argument("--colibri-gguf")
    ap.add_argument("--colibri-mmproj")
    ap.add_argument("--workdir", default="/tmp/opencode/bench_wavs")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--omp-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "0")) or None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    env = os.environ.copy()
    if args.omp_threads: env["OMP_NUM_THREADS"] = str(args.omp_threads)
    samples = ensure_samples(args.workdir)
    c_gguf = args.colibri_gguf or args.gguf
    c_mmproj = args.colibri_mmproj or args.mmproj
    meta = {"platform": platform.platform(), "python": sys.version.split()[0], "machine": platform.machine(), "processor": platform.processor(),
            "omp_num_threads": env.get("OMP_NUM_THREADS"), "cpu_count": os.cpu_count(), "repeats": args.repeats,
            "model": args.model, "gguf": args.gguf, "mmproj": args.mmproj, "colibri_gguf": c_gguf, "colibri_mmproj": c_mmproj,
            "omp_guidance": "Set OMP_NUM_THREADS or --omp-threads. Start near physical cores or nproc-2/4; returns diminish past a reasonable core count."}
    colibri_rows, colibri_cmd = bench_colibri(args, samples, env, gguf=None, mmproj=None, engine_name="colibri-bf16")
    colibri_gguf_rows, colibri_gguf_cmd = ([], None)
    if c_gguf:
        colibri_gguf_rows, colibri_gguf_cmd = bench_colibri(args, samples, env, gguf=c_gguf, mmproj=c_mmproj, engine_name="colibri-gguf")
    llama_rows, llama_cmd = bench_llama(args, samples, env)
    meta["colibri_cmd"] = colibri_cmd
    meta["colibri_gguf_cmd"] = colibri_gguf_cmd
    meta["llama_cmd"] = llama_cmd
    rows = colibri_rows + colibri_gguf_rows + llama_rows
    report = {"meta": meta, "rows": rows}
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f: f.write(text + "\n")
    if args.json:
        print(text)
    else:
        print(f"# ASR benchmark\nplatform={meta['platform']} machine={meta['machine']} omp={meta['omp_num_threads']}")
        print("engine,sample,rep,cold,wer,cer,rtf,wall_s,audio_s,load_s,rss_peak_kb,text")
        for r in rows:
            print(f"{r['engine']},{r['sample']},{r['rep']},{r['cold']},{r['wer']},{r['cer']},{r['rtf']},{r['wall_s']},{r['audio_s']},{r['load_s']},{r['rss_peak_kb']},{json.dumps(r['text'])}")

if __name__ == "__main__":
    main()
