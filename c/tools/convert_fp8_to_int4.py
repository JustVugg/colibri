"""
Convertitore GLM-5.2-FP8 -> nostro container int4 (STADIO B).

Strategia DISK-SAFE (richiesta dell'utente): scarica UNO shard (~5 GB), lo converte in
int4, lo CANCELLA, passa al prossimo. Il disco non si riempie mai: picco = 1 shard + l'output
int4 che cresce fino a ~372 GB. Controllo di spazio che si ferma se manca margine.

Cosa fa per ogni tensore:
  - pesi FP8 (e4m3) con `*.weight_scale_inv`  -> dequant a blocchi 128x128 -> f32
  - pesi BF16 (norme/embed/lm_head/...)        -> f32
  poi:
  - attn/mlp/shared/expert/embed/lm_head -> QUANTIZZATO int4 (o int8) con la STESSA matematica
    del motore C (np.rint = lrintf, stesse soglie, stesso packing dei nibble) -> token identici
  - norme / router (mlp.gate.weight) / bias / e_score_correction_bias -> tenuti F32
  - indexer DSA / layer MTP (78) / shared_head / eh_proj / *norm dell'indexer -> SALTATI

Output: una dir di safetensors leggibile dal motore C (per ogni peso quantizzato: `nome` U8 =
dati impacchettati, `nome.qs` F32 = scale per riga).

USO:
  # test locale (oracolo tiny, niente download): converte una dir gia' presente
  python3 tools/convert_fp8_to_int4.py --indir glm_tiny --outdir glm_tiny_i4 --ebits 4 --io-bits 4
  # selftest del dequant fp8 (richiede torch)
  python3 tools/convert_fp8_to_int4.py --selftest
  # reale: scarica+converte+cancella shard per shard
  python3 tools/convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 --outdir /home/vincenzo/glm52_i4
"""
import os, sys, glob, json, shutil, argparse
import numpy as np

# ---------- quantizzazione: identica al C (glm.c) ----------
def quant_int8(w, bits):                       # w: [O,I] f32 -> (qbytes U8 [O*I], scale f32 [O])
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

def quant_int4(w, bits):                        # -> (qbytes U8 [O*ceil(I/2)], scale f32 [O])
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)  # nibble [-8,7]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def quant_int2(w, bits):                        # -> (qbytes U8 [O*ceil(I/4)], scale f32 [O]); 4/byte
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1                 # bits=2 -> qmax=1, valori [-2,1]
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -2, qmax).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                           # impacchetta 4 valori per byte (identico a pack_int2 in C)
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), s[:, 0].astype(np.float32)

# ---------- classificazione dei tensori ----------
def layer_idx(name):
    p = name.split(".")
    if len(p) > 2 and p[0] == "model" and p[1] == "layers":
        try: return int(p[2])
        except ValueError: return -1
    return -1

def classify(name, n_layers, keep_mtp=False, keep_idx=False):
    if name.endswith("_scale_inv"): return "consumed"   # gestito col suo peso
    li = layer_idx(name)
    if keep_idx:
        # modalita' --indexer: SOLO i pesi del DSA lightning indexer dei layer principali
        if li < 0 or li >= n_layers or "indexer" not in name: return "skip"
        if name.endswith("norm.weight"): return "f32"
        return "q"                                       # int8 consigliato (--ebits 8): pesi di scoring
    if keep_mtp:
        if li != n_layers: return "skip"                 # solo il layer MTP
        if "indexer" in name: return "skip"              # il DSA indexer resta un no-op
    else:
        if li >= n_layers: return "skip"                 # layer MTP (78)
        if any(k in name for k in ["indexer", "indexers_proj", "eh_proj",
                                    "enorm", "hnorm", "shared_head"]): return "skip"
    if name.endswith("e_score_correction_bias"): return "f32"
    if name.endswith("mlp.gate.weight"): return "f32"    # router (NON gate_proj)
    if name.endswith("norm.weight") or name == "model.norm.weight": return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"): return "io"
    if ".mlp.experts." in name and name.endswith(".weight"): return "x"  # expert ROUTED (streaming)
    if name.endswith(".weight"): return "q"              # attn/dense-mlp/shared (residente)
    return "f32"

# ---------- dequant di un tensore (fp8+scale a blocchi / bf16 / f32) ----------
def dequant(f, name):
    import torch
    sl = f.get_slice(name); dt = sl.get_dtype()
    if dt in ("F8_E4M3", "float8_e4m3fn"):
        w = f.get_tensor(name).to(torch.float32)
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)   # [ceil(O/128),ceil(I/128)]
        O, I = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        return (w * sc).numpy()
    return f.get_tensor(name).to(torch.float32).numpy()

def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits, keep_mtp=False, keep_idx=False):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            kind = classify(name, n_layers, keep_mtp, keep_idx)
            if kind in ("skip", "consumed"): continue
            w = dequant(f, name)
            if kind == "f32":
                out_dict[name] = w.astype(np.float32)
            else:
                bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
                if w.ndim != 2:        # es. bias 1D non previsto come 'q' -> tienilo f32
                    out_dict[name] = w.astype(np.float32); continue
                q, s = (quant_int2(w, bits) if bits <= 2 else
                        quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
                out_dict[name] = q
                out_dict[name + ".qs"] = s

def free_gb(p): return shutil.disk_usage(p).free / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=None)   # bit residenti (default 4; 8 per --mtp/--indexer)
    ap.add_argument("--io-bits", type=int, default=8)    # bit di embed/lm_head
    ap.add_argument("--xbits", type=int, default=None)   # bit degli expert ROUTED (streaming); default=ebits
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mtp", action="store_true",
        help="download and convert ONLY the MTP head (model.layers.<n_layers>.*) -> out-mtp-*.safetensors")
    ap.add_argument("--indexer", action="store_true",
        help="extract ONLY the DSA lightning-indexer weights -> out-idx-*.safetensors. WARNING: "
             "indexer tensors are spread across nearly every shard, so this re-downloads the whole "
             "repository (~756 GB of traffic) to retain only a few GB. Resumable per shard. "
             "Recommended: --ebits 8.")
    a = ap.parse_args()
    if a.ebits is None:
        # testa MTP a int4 = acceptance ~0-4% (misurato, issue #8): il draft sbaglia sempre
        # e la speculazione non parte mai. A int8: 39-59%, 2.2-2.8 token/forward.
        a.ebits = 8 if (a.mtp or a.indexer) else 4
    if a.xbits is None: a.xbits = a.ebits

    if a.selftest:
        import torch
        w = (torch.randn(256, 256) * 0.3)
        O, I = w.shape; bs = 128
        sc = torch.zeros(O // bs, I // bs)
        for bi in range(O // bs):
            for bj in range(I // bs):
                blk = w[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                sc[bi, bj] = blk.abs().max() / 448.0
        q = (w / sc.repeat_interleave(bs,0).repeat_interleave(bs,1)).to(torch.float8_e4m3fn)
        deq = (q.to(torch.float32) * sc.repeat_interleave(bs,0).repeat_interleave(bs,1))
        rel = (deq - w).abs().mean() / w.abs().mean()
        print(f"[selftest fp8 block-dequant] mean relative error = {rel:.4f}  "
              f"({'OK' if rel < 0.05 else 'HIGH'})")
        return

    os.makedirs(a.outdir, exist_ok=True)
    if a.indir:    # conversione locale (test)
        shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
        from safetensors.numpy import save_file
        for i, sp in enumerate(shards):
            out = {}; convert_shard(sp, out, a.n_layers, a.ebits, a.io_bits, a.xbits)
            save_file(out, os.path.join(a.outdir, f"out-{i:05d}.safetensors"))
        # copia config + tokenizer
        for fn in ["config.json"]:
            src = os.path.join(a.indir, fn)
            if os.path.exists(src): shutil.copy(src, a.outdir)
        print(f"converted {len(shards)} shards -> {a.outdir}")
        return

    # reale: scarica shard per shard, converte, cancella
    # EN: real: download shard by shard, convert, delete
    #
    # ROBUSTEZZA RETE: timeout brevi sulle read cosi' un download appeso FALLISCE invece
    # di restare fermo per sempre. 8s, non 30: "timeout" = ZERO byte ricevuti in quella
    # finestra; su un transfer vivo i chunk arrivano di continuo, quindi 8s e' sicuro e
    # uno stallo costa 8s invece di 30.
    # EN: NETWORK ROBUSTNESS: short read timeouts so a hung download FAILS instead of
    # EN: sitting there forever. 8s, not 30: a "timeout" means ZERO bytes received in that
    # EN: window; a live transfer delivers chunks constantly, so 8s is safe and a stall
    # EN: costs 8s instead of 30.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "8")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    # log con timestamp: i messaggi "Trying to resume" di hf_hub diventano databili.
    # EN: timestamped logs: hf_hub's "Trying to resume" messages become datable.
    import logging
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    # hf_xet si blocca quando la rete si riavvia (connessioni zombie senza timeout):
    # forza la via HTTP classica, che curl ha dimostrato funzionare. (misurato 2026-07-02)
    # EN: hf_xet hangs when the network restarts (zombie connections with no timeout):
    # EN: force the classic HTTP path, which curl proved works (measured 2026-07-02).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # =0 per riabilitare xet / to re-enable xet
    from huggingface_hub import HfApi, hf_hub_download

    # lock anti-doppione: DUE convertitori sulla stessa outdir si corrompono a vicenda.
    # EN: anti-duplicate lock: TWO converters on the same outdir corrupt each other.
    import fcntl
    lock = open(os.path.join(a.outdir, ".convert.lock"), "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("ERROR: another converter is already using this output directory. Exiting."); return

    # dimensioni note dei file, riempite dopo repo_info: il downloader multi-stream le usa
    # per calcolare i confini dei segmenti e per sapere quando un file e' completo.
    # EN: known file sizes, filled after repo_info: the multi-stream downloader uses them
    # EN: to compute segment boundaries and to know when a file is complete.
    SIZES = {}

    def download_retry(repo, fn, dest, tries=999):
        """Downloader multi-stream con resume via Range. Apre N segmenti concorrenti
        (default 2, COLI_DL_STREAMS per cambiarli) e salva lo stato per-segmento in un
        sidecar .seg -> NESSUN byte perso comunque muoia la connessione. Un singolo stream
        HF e' limitato a ~2 MB/s (misurato); 2 stream ~ raddoppiano il throughput senza
        saturare una linea domestica. File piccoli, COLI_DL_STREAMS=1 o un vecchio .part
        legacy -> percorso a stream singolo (_download_single).
        EN: multi-stream Range-resume downloader. Opens N concurrent segments (default 2,
        EN: COLI_DL_STREAMS to change) and saves per-segment state in a .seg sidecar -> NO
        EN: byte is lost however the connection dies. A single HF stream is paced at
        EN: ~2 MB/s (measured); 2 streams roughly double throughput without saturating a
        EN: home line. Small files, COLI_DL_STREAMS=1 or a legacy .part -> single-stream
        EN: path (_download_single)."""
        import time as _t, threading, urllib.request, urllib.error
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        out = os.path.join(dest, fn); part = out + ".part"; side = part + ".seg"
        os.makedirs(dest, exist_ok=True)
        expected = SIZES.get(fn)
        if os.path.exists(out) and (expected is None or os.path.getsize(out) == expected):
            return out
        NS = max(1, min(8, int(os.environ.get("COLI_DL_STREAMS", "2"))))
        # un .part senza sidecar l'ha scritto una versione precedente a stream singolo.
        # EN: a .part without a sidecar was written by an older single-stream version.
        legacy = os.path.exists(part) and not os.path.exists(side)
        if expected is None or expected < (256 << 20) or NS == 1 or legacy:
            return _download_single(url, fn, out, part, expected)
        # ---- multi-stream ----
        segs = [(expected * t // NS, expected * (t + 1) // NS) for t in range(NS)]
        done = [0] * NS
        # riprendi lo stato dei segmenti se il sidecar combacia (stesso N, stessa size).
        # EN: resume per-segment progress if the sidecar matches (same N, same size).
        if os.path.exists(side):
            try:
                st = json.loads(open(side).read())
                if st.get("n") == NS and st.get("size") == expected: done = st["done"]
            except Exception: pass
        if not os.path.exists(part):
            with open(part, "wb") as f: f.truncate(expected)   # file sparse / sparse file
        fd = os.open(part, os.O_WRONLY)
        t0 = _t.time(); nres = [0]; log_lock = threading.Lock(); stopfail = []
        def worker(t):
            s0, s1 = segs[t]
            while done[t] < s1 - s0 and not stopfail:
                pos = s0 + done[t]
                req = urllib.request.Request(url, headers={"User-Agent": "colibri-convert",
                                                           "Range": f"bytes={pos}-{s1-1}"})
                try:
                    with urllib.request.urlopen(req, timeout=8) as r:
                        if r.status != 206:               # Range ignorato: multi-stream impossibile
                            stopfail.append(t); return    # EN: Range ignored: multi-stream impossible
                        while done[t] < s1 - s0:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            rem = (s1 - s0) - done[t]     # mai oltre il segmento / never past the segment
                            if len(chunk) > rem: chunk = chunk[:rem]
                            os.pwrite(fd, chunk, s0 + done[t])
                            done[t] += len(chunk)
                except KeyboardInterrupt: raise
                except Exception as ex:
                    with log_lock:
                        nres[0] += 1
                        print(f"    [dl] s{t}: {type(ex).__name__} at {(s0+done[t])/1e9:.2f} GB: "
                              f"resuming (#{nres[0]})", flush=True)
                    _t.sleep(min(15, 1 + nres[0] // NS))
        th = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(NS)]
        for x in th: x.start()
        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected: {NS} streams, "
              f"{sum(done)/1e9:.2f} of {expected/1e9:.2f} GB", flush=True)
        mark = sum(done); tmark = t0
        while any(x.is_alive() for x in th):
            _t.sleep(5)
            have = sum(done)
            tmpside = side + ".tmp"                       # checkpoint atomico / atomic checkpoint
            open(tmpside, "w").write(json.dumps({"n": NS, "size": expected, "done": list(done)}))
            os.replace(tmpside, side)
            now = _t.time()
            if now - tmark >= 30:
                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s, {NS} stream)", flush=True)
                mark = have; tmark = now
        os.close(fd)
        if stopfail:                                      # il server non onora il Range: fallback
            for f2 in (part, side):                       # EN: server won't honor Range: fall back
                if os.path.exists(f2): os.remove(f2)
            return _download_single(url, fn, out, part, expected)
        assert sum(done) == expected
        if os.path.exists(side): os.remove(side)
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9)
        print(f"    [dl] {fn}: {expected/1e9:.2f} GB in {dt/60:.1f} min "
              f"({expected/dt/1e6:.1f} MB/s avg, {NS} streams, {nres[0]} resumes)", flush=True)
        return out

    def _download_single(url, fn, out, part, expected):
        """Percorso a stream singolo con resume via Range (file piccoli / .part legacy /
        COLI_DL_STREAMS=1). Un EOF corto ma pulito conta come ripresa; se non arriva
        NESSUN byte nuovo, backoff invece di girare a vuoto.
        EN: single-stream path with Range resume (small files / legacy .part /
        EN: COLI_DL_STREAMS=1). A clean short EOF counts as a resume; if NO new byte
        EN: arrives, back off instead of spinning."""
        import time as _t, urllib.request, urllib.error
        t0 = _t.time(); nres = 0; mark = 0; tmark = t0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if expected is not None and have >= expected: break
            have0 = have
            req = urllib.request.Request(url, headers={"User-Agent": "colibri-convert"})
            if have: req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    if have and r.status == 200:          # server ha ignorato il Range: riparti pulito
                        have = 0                          # EN: server ignored Range: restart clean
                    if expected is None:
                        cl = r.headers.get("Content-Length")
                        if cl: expected = have + int(cl)
                    if have == 0 or nres:                 # segnale di vita subito / immediate sign of life
                        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected"
                              f"{f' @ {have/1e9:.2f} GB' if have else ''}"
                              f"{f' of {expected/1e9:.2f} GB' if expected else ''}", flush=True)
                    with open(part, "ab" if have else "wb") as f:
                        if not have: f.truncate(0)
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            f.write(chunk); have += len(chunk)
                            if have - mark >= 512 * 1024 * 1024 or _t.time() - tmark >= 30:
                                now = _t.time()
                                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s)", flush=True)
                                mark = have; tmark = now
                if expected is None: break                # lunghezza ignota: passata singola / unknown length
                if have < expected:                       # EOF corto ma pulito: conta come ripresa
                    nres += 1                             # EN: clean short EOF: counts as a resume
                    if have == have0: _t.sleep(min(15, 1 + nres))   # zero progresso -> backoff / zero progress -> back off
            except KeyboardInterrupt: raise
            except urllib.error.HTTPError as ex:
                if ex.code == 416: break                  # gia' completo / already complete
                nres += 1
                print(f"    [dl] HTTP {ex.code} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
            except Exception as ex:
                nres += 1
                print(f"    [dl] {type(ex).__name__} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9); sz = os.path.getsize(out)
        print(f"    [dl] {fn}: {sz/1e9:.2f} GB in {dt/60:.1f} min "
              f"({sz/dt/1e6:.1f} MB/s avg, {nres} resumes)", flush=True)
        return out

    from safetensors.numpy import save_file
    import time as _t
    for att in range(999):
        try:
            info = HfApi().repo_info(a.repo, files_metadata=True)
            # dimensioni note dallo store: abilitano il download multi-stream a segmenti.
            # EN: sizes known from the store: enable segmented multi-stream download.
            SIZES.update({s.rfilename: s.size for s in info.siblings if s.size})
            break
        except KeyboardInterrupt: raise
        except Exception as ex:
            w = min(60, 5*(att+1)); print(f"repo_info failed ({type(ex).__name__}); retrying in {w}s", flush=True); _t.sleep(w)
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        try: shutil.copy(hf_hub_download(a.repo, fn, local_dir=a.outdir+"/_meta"), a.outdir)
        except Exception: pass
    tmp = os.path.join(a.outdir, "_inflight"); os.makedirs(tmp, exist_ok=True)
    if a.mtp:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        pref = f"model.layers.{a.n_layers}."
        mtp_shards = sorted(set(v for k, v in idx.items() if k.startswith(pref)))
        print(f"[MTP] head at layer {a.n_layers}: {len(mtp_shards)} shards to process: {mtp_shards}")
        for i, sh in enumerate(mtp_shards):
            outp = os.path.join(a.outdir, f"out-mtp-{i:05d}.safetensors")
            if os.path.exists(outp): print(f"[MTP] {outp} already done"); continue
            print(f"[MTP {i+1}/{len(mtp_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_mtp=True)
            save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB, {len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[MTP] DONE."); return
    if a.indexer:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        idx_shards = sorted(set(v for k, v in idx.items()
                                if "indexer" in k and 0 <= layer_idx(k) < a.n_layers))
        tot_gb = len(idx_shards) * 5.4
        print(f"[IDX] indexer weights across {len(idx_shards)} shards (~{tot_gb:.0f} GB total download, resumable)")
        for i, sh in enumerate(idx_shards):
            outp = os.path.join(a.outdir, f"out-idx-{i:05d}.safetensors")
            if os.path.exists(outp): continue             # gia' fatto -> ripartibile
            print(f"[IDX {i+1}/{len(idx_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_idx=True)
            if out: save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[IDX] DONE."); return
    for i, sh in enumerate(shards):
        if free_gb(a.outdir) < a.min_free_gb:
            print(f"STOP: free space is below {a.min_free_gb} GB. Free space and rerun to resume."); break
        outp = os.path.join(a.outdir, f"out-{i:05d}.safetensors")
        if os.path.exists(outp): continue                 # gia' fatto -> ripartibile
        print(f"[{i+1}/{len(shards)}] downloading {sh} ({free_gb(a.outdir):.0f} GB free)...", flush=True)
        p = download_retry(a.repo, sh, tmp)
        out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits)
        save_file(out, outp)
        os.remove(p)                                       # <-- cancella subito lo shard fp8
        for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
            if os.path.isfile(blob): os.remove(blob)
        print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("DONE." if i == len(shards)-1 else "INTERRUPTED (rerun to resume).")

if __name__ == "__main__":
    main()
