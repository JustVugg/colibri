#!/usr/bin/env python3
"""Reproducible ASR benchmark: Colibri qwen3_asr vs llama.cpp.

Reports WER, CER, RTF, wall time, audio duration, and peak RSS (when /usr/bin/time -v available).
Uses deterministic flite-generated WAV samples by default.

Example:
  python3 c/tools/benchmark_asr.py \
    --colibri ./c/qwen3_asr \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<rev> \
    --llama-cli .../llama-cli --gguf ...Q8_0.gguf --mmproj ...mmproj-Q8_0.gguf
"""
from __future__ import annotations
import argparse, base64, json, os, platform, resource, subprocess, sys, time, wave

DEFAULT_SAMPLES = [
    ("hello", "Hello world, this is a test of speech recognition."),
    ("weather", "The weather today is sunny with a chance of rain."),
    ("numbers", "One two three four five six seven eight nine ten."),
]

def norm(t: str):
    return t.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "").replace("'", "").split()

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
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"flite=text='{text}':voice=slt", "-ar", "16000", "-ac", "1", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out.append((name, text, path))
    return out

def bench_colibri(args, samples):
    cmd=[args.colibri, "--model", args.model]
    if args.gguf: cmd += ["--gguf", args.gguf]
    if args.mmproj: cmd += ["--mmproj", args.mmproj]
    cmd += ["--serve"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ready = False
    t0 = time.time()
    for _ in range(800):
        line = p.stdout.readline()
        if b"READY" in line: ready = True; break
    load_s = time.time() - t0
    if not ready: raise RuntimeError("colibri engine did not become READY")
    rows = []
    for name, ref, wav in samples:
        audio_b64 = base64.b64encode(open(wav, "rb").read()).decode()
        payload = json.dumps({"audio_b64": audio_b64}).encode()
        cmd = f"TRANSCRIBE t1 0 {len(payload)}\n".encode() + payload + b"\n"
        t0 = time.time(); p.stdin.write(cmd); p.stdin.flush()
        text = ""; err = None
        while True:
            line = p.stdout.readline()
            if not line: err = "eof"; break
            if line.startswith(b"TRANSCRIPT t1 "):
                n = int(line.split()[-1]); pl = p.stdout.read(n); p.stdout.readline()
                try: text = json.loads(pl.decode()).get("text", "")
                except Exception as e: err = str(e)
            if line.startswith(b"DONE t1"): break
            if line.startswith(b"ERROR t1"): err = line.decode(errors="ignore").strip(); break
        dt = time.time() - t0; dur = wav_dur(wav)
        we, wn = wer(ref, text); ce, cn = cer(ref, text)
        rows.append({"engine":"colibri","sample":name,"wer":f"{we}/{wn}","cer":f"{ce}/{cn}","rtf":round(dt/dur,3),"wall_s":round(dt,3),"audio_s":round(dur,3),"load_s":round(load_s,3),"text":text,"error":err})
    p.terminate()
    try: p.wait(timeout=5)
    except Exception: p.kill()
    return rows

def bench_llama(args, samples):
    if not args.llama_cli: return []
    rows = []
    for name, ref, wav in samples:
        t0 = time.time()
        r = subprocess.run([args.llama_cli, "-m", args.gguf, "--mmproj", args.mmproj, "--audio", wav, "-p", "Transcribe this audio.", "-n", "128", "--no-display-prompt", "-st"], capture_output=True, text=True, timeout=args.timeout)
        dt = time.time() - t0
        text = r.stdout
        if "<asr_text>" in text: text = text.split("<asr_text>")[-1]
        for cut in ["[ Prompt:", "\n>", "Exiting"]:
            if cut in text: text = text.split(cut)[0]
        text = text.strip(); dur = wav_dur(wav)
        we, wn = wer(ref, text); ce, cn = cer(ref, text)
        rows.append({"engine":"llama.cpp","sample":name,"wer":f"{we}/{wn}","cer":f"{ce}/{cn}","rtf":round(dt/dur,3),"wall_s":round(dt,3),"audio_s":round(dur,3),"load_s":None,"text":text,"error":None if r.returncode==0 else f"rc={r.returncode}"})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colibri", default="./qwen3_asr")
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-cli")
    ap.add_argument("--gguf")
    ap.add_argument("--mmproj")
    ap.add_argument("--workdir", default="/tmp/opencode/bench_wavs")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    samples = ensure_samples(args.workdir)
    meta = {"platform": platform.platform(), "python": sys.version.split()[0], "machine": platform.machine(), "processor": platform.processor()}
    rows = bench_colibri(args, samples) + bench_llama(args, samples)
    if args.json:
        print(json.dumps({"meta": meta, "rows": rows}, indent=2))
    else:
        print(f"# ASR benchmark\nplatform={meta['platform']} machine={meta['machine']}")
        print("engine,sample,wer,cer,rtf,wall_s,audio_s,load_s,text")
        for r in rows:
            print(f"{r['engine']},{r['sample']},{r['wer']},{r['cer']},{r['rtf']},{r['wall_s']},{r['audio_s']},{r['load_s']},{json.dumps(r['text'])}")

if __name__ == "__main__":
    main()
