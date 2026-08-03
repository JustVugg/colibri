#!/usr/bin/env python3
"""Inkling: quantizza la DENSA RESIDENTE (bf16 -> int4-gs64) in un container a parte.

Serve solo se la RAM non basta. Il container int4 di Inkling
(nbeerbower/Inkling-colibri-int4) ha gli esperti routed gia' a int4 ma tiene la
densa in **bf16**: 49.4 GB che devono stare in RAM per ogni token. Su un host con
meno di ~64 GB non ci stanno — e il caricamento li espande pure a f32 (~99 GB al
picco), quindi il processo muore prima di generare. A int4-gs64 la densa scende a
**15.3 GB** e il modello gira su una macchina da 25 GB.

Con RAM abbondante NON serve: `inkling.c` carica la densa bf16 come sempre.

Cosa fa:
  - legge SOLO gli header degli shard (pochi MB) e costruisce un indice
  - quantizza un tensore alla volta, a blocchi di righe (picco RAM ~1 GB)
  - scrive UN file nuovo; gli esperti routed (464 GB) non vengono toccati
  - gli originali sono aperti in sola lettura: se qualcosa va storto si rifa'
  - riporta l'errore di quantizzazione L2 per classe, misurato sui pesi veri

Formato di output (stessa convenzione .qs che inkling.c usa gia' per gli esperti):
    <nome>      U8   [O, ceil(I/2)]   nibble packed, low = colonna pari, offset +8
    <nome>.qs   F32  [O, ceil(I/64)]  una scala ogni 64 elementi lungo l'INPUT
    __metadata__ {"<nome>": "fmt=4;gs=64"}  /  "fmt=1" per int8  /  "fmt=raw"

Politica di precisione (misurata: vedi docs/inkling.md):
    attention / shared_experts / mlp denso   -> int4-gs64    ~11% errore L2
    embed_tokens / lm_head                   -> int8 per-row ~0.9% (entrano in OGNI token)
    norm / bias / A_log / conv1d / router    -> passthrough  (minuscoli e delicati)
ATTN_BITS=8 porta anche l'attention a int8 (+4 GB, errore 11% -> 1.1%): non serve
per la coerenza dell'output, ma e' li' se vuoi spendere RAM per la qualita'.

Uso:
    python3 convert_inkling_dense_int4.py --dir /path/al/modello --plan   # stima
    python3 convert_inkling_dense_int4.py --dir /path/al/modello          # converte
    mkdir -p /path/al/modello/dense-int4g64
    mv /path/al/modello/dense-int4g64.safetensors \\
       /path/al/modello/dense-int4g64/dense.safetensors
Il motore trova il container da solo (INK_DENSE_Q4=0 lo ignora).
"""
import argparse, json, os, re, struct, sys, time
import numpy as np

DIR = os.environ.get("INKLING_DIR", ".")
OUT = os.path.join(DIR, "dense-int4g64.safetensors")
GS = 64
ROWS_PER_CHUNK = 4096          # blocchi di righe: tiene il picco RAM ~<1 GB

# ---------- classificazione (per nome) ----------
RE_EXPERT = re.compile(r"\.experts\.")            # routed: NON toccare (shared_experts escluso sotto)
RE_SHARED = re.compile(r"shared_experts")
RE_ATTN   = re.compile(r"self_attn")
RE_MLPD   = re.compile(r"\.mlp\.(gate|up|down)_proj")
RE_EMB    = re.compile(r"(embed_tokens|lm_head)")
RE_SMALL  = re.compile(r"(norm|bias|A_log|dt_bias|conv1d|\.gate\.|rotary|scale$)", re.I)

MIN_ELEMS = 1 << 20        # sotto ~1M elementi non vale il rischio: passthrough
ATTN_BITS = int(os.environ.get("ATTN_BITS", "8"))   # attention: 8 = int8 (default), 4 = int4-gs64

def classify(name, shape, dtype):
    """-> 'skip' (expert routed) | 'gs4' | 'int8' | 'raw'

    Regola GENERALE, non una whitelist di nomi: qualunque peso float grande e
    2-D/3-D con l'ultima dimensione (quella di contrazione) >= GS va a int4-gs64.
    Una whitelist mi aveva gia' fatto perdere shared_experts (3-D [2,3072,6144])
    e mtp.*.input_proj, 14.5 GB finiti in passthrough senza accorgersene."""
    if RE_EXPERT.search(name) and not RE_SHARED.search(name):
        return "skip"                              # esperti routed: restano negli shard
    if dtype not in ("BF16", "F16", "F32"):
        return "raw"                               # gia' quantizzato o non-float
    if RE_EMB.search(name):
        return "int8"                              # embed/lm_head: sensibili, ogni token
    if RE_SMALL.search(name):
        return "raw"                               # norm/bias/A_log/conv1d/router: delicati
    # ATTN_BITS=8 (default): l'attention a 4 bit rende il modello incoerente —
    # misurato: 11% di errore per matmul x ~5 matmul x 66 layer e l'output degenera
    # (il container e il kernel sono verificati corretti, e' la precisione a mancare).
    # ATTN_BITS=4 la rimette a int4-gs64 per confronto.
    if RE_ATTN.search(name) and ATTN_BITS == 8:
        return "int8"
    n = 1
    for d in shape:
        n *= d
    if len(shape) < 2 or shape[-1] < GS or n < MIN_ELEMS:
        return "raw"                               # 1-D, contrazione corta o troppo piccolo
    return "gs4"

# ---------- I/O safetensors ----------
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def read_rows(path, base, off0, dtype, cols, r0, r1):
    """Legge le righe [r0,r1) di un tensore 2-D e le restituisce in f32."""
    itemsize = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1}[dtype]
    start = base + off0 + r0 * cols * itemsize
    nbytes = (r1 - r0) * cols * itemsize
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(nbytes)
    if dtype == "BF16":                            # bf16 -> f32: shift a sinistra di 16 bit
        u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(r1 - r0, cols)
    if dtype == "F16":
        return np.frombuffer(raw, np.float16).astype(np.float32).reshape(r1 - r0, cols)
    if dtype == "F32":
        return np.frombuffer(raw, np.float32).reshape(r1 - r0, cols)
    raise ValueError(f"dtype non gestito per la quantizzazione: {dtype}")

# ---------- quantizzatori ----------
def quant_gs4(w):
    """w [R, I] f32 -> (packed U8 [R, ceil(I/2)], scales f32 [R, ng]).
       Gruppi lungo I (dimensione di contrazione), simmetrico, nibble +8."""
    R, I = w.shape
    ng = (I + GS - 1) // GS
    pad = ng * GS - I
    wp = np.concatenate([w, np.zeros((R, pad), np.float32)], 1) if pad else w
    g = wp.reshape(R, ng, GS)
    amax = np.abs(g).max(-1, keepdims=True)
    scale = (amax / 7.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(g / scale), -8, 7).astype(np.int8) + 8      # 0..15
    q = q.reshape(R, ng * GS)[:, :I].astype(np.uint8)
    if I % 2:                                                       # coda dispari
        q = np.concatenate([q, np.full((R, 1), 8, np.uint8)], 1)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.reshape(R, ng).astype(np.float32)

def dequant_gs4(packed, scale, I):
    R = packed.shape[0]
    q = np.empty((R, packed.shape[1] * 2), np.float32)
    q[:, 0::2] = (packed & 15).astype(np.float32) - 8.0
    q[:, 1::2] = (packed >> 4).astype(np.float32) - 8.0
    q = q[:, :I]
    s = np.repeat(scale, GS, axis=1)[:, :I]
    return q * s

def quant_i8(w):
    amax = np.abs(w).max(1, keepdims=True)
    scale = (amax / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale[:, 0].astype(np.float32)

def dequant_i8(q, scale):
    return q.astype(np.float32) * scale[:, None]

# ---------- main ----------
def main():
    global DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="stima e piano, non scrive nulla")
    ap.add_argument("--dir", default=DIR, help="dir del modello")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    DIR = a.dir
    if a.out == OUT: a.out = os.path.join(DIR, "dense-int4g64.safetensors")
    shards = sorted(f for f in os.listdir(DIR) if f.endswith(".safetensors") and not f.startswith("dense-"))
    print(f"[1/3] indicizzo {len(shards)} shard (solo header)…", flush=True)
    index = {}                                     # name -> (path, base, off0, off1, dtype, shape)
    for sh in shards:
        p = os.path.join(DIR, sh)
        hdr, base = read_header(p)
        for k, t in hdr.items():
            if k == "__metadata__":
                continue
            o = t["data_offsets"]
            index[k] = (p, base, o[0], o[1], t["dtype"], t["shape"])

    def flat2d(shape):
        """[E,R,I] o [R,I] -> (righe, I): i gruppi vanno sull'ULTIMA dim (contrazione)."""
        I = shape[-1]
        R = 1
        for d in shape[:-1]:
            R *= d
        return R, I

    plan, tot_in, tot_out = [], 0, 0
    for name, (p, base, o0, o1, dt, shape) in sorted(index.items()):
        kind = classify(name, shape, dt)
        if kind == "skip":
            continue
        nbytes_in = o1 - o0
        tot_in += nbytes_in
        if kind == "gs4":
            R, I = flat2d(shape)
            ng = (I + GS - 1) // GS
            out_b = R * ((I + 1) // 2) + R * ng * 4
        elif kind == "int8":
            R, I = flat2d(shape)
            out_b = R * I + R * 4
        else:
            out_b = nbytes_in
        tot_out += out_b
        plan.append((name, kind, p, base, o0, o1, dt, tuple(shape), out_b))

    by = {}
    for name, kind, *_rest in plan:
        by[kind] = by.get(kind, 0) + 1
    print(f"      densa da convertire: {len(plan)} tensori  {by}")
    print(f"      input  bf16/f32 : {tot_in/1e9:7.2f} GB")
    print(f"      output stimato   : {tot_out/1e9:7.2f} GB   ({100*tot_out/max(tot_in,1):.0f}%)")
    free = os.statvfs(DIR).f_bavail * os.statvfs(DIR).f_frsize
    print(f"      spazio libero    : {free/1e9:7.2f} GB")
    if a.plan:
        print("\n[--plan] nessuna scrittura. Rilancia senza --plan per convertire.")
        return
    if free < tot_out * 1.15:
        sys.exit(f"[ERR] spazio insufficiente: servono ~{tot_out*1.15/1e9:.0f} GB")

    # ---- header di output: shape/offset noti in anticipo ----
    print("[2/3] costruisco l'header di output…", flush=True)
    hdr, off, meta = {}, 0, {}
    for name, kind, p, base, o0, o1, dt, shape, _ob in plan:
        if kind == "gs4":
            R, I = flat2d(shape)
            ng = (I + GS - 1) // GS
            nb = R * ((I + 1) // 2)
            hdr[name] = {"dtype": "U8", "shape": [R, (I + 1) // 2], "data_offsets": [off, off + nb]}; off += nb
            nb = R * ng * 4
            hdr[name + ".qs"] = {"dtype": "F32", "shape": [R, ng], "data_offsets": [off, off + nb]}; off += nb
            # orig= conserva la shape logica (3-D per gli shared_experts): il loader
            # ricostruisce [E,R,I] dai byte piatti senza doverla indovinare.
            meta[name] = f"fmt=4;gs={GS};I={I};orig={'x'.join(str(d) for d in shape)}"
        elif kind == "int8":
            R, I = flat2d(shape)
            nb = R * I
            hdr[name] = {"dtype": "I8", "shape": [R, I], "data_offsets": [off, off + nb]}; off += nb
            nb = R * 4
            hdr[name + ".qs"] = {"dtype": "F32", "shape": [R], "data_offsets": [off, off + nb]}; off += nb
            meta[name] = f"fmt=1;orig={'x'.join(str(d) for d in shape)}"
        else:
            nb = o1 - o0
            hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}; off += nb
            meta[name] = "fmt=raw"
    meta["colibri.dense"] = "inkling-int4g64-v1"
    hdr["__metadata__"] = meta
    hjson = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (8 - (len(hjson) % 8)) % 8               # header allineato a 8 byte
    hjson += b" " * pad

    tmp = a.out + ".part"
    print(f"[3/3] scrivo {a.out}  ({off/1e9:.2f} GB payload)…", flush=True)
    err_by_kind = {}
    t0 = time.time()
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(hjson))); w.write(hjson)
        done = 0
        for name, kind, p, base, o0, o1, dt, shape, _ob in plan:
            if kind == "raw":
                with open(p, "rb") as f:
                    f.seek(base + o0)
                    left = o1 - o0
                    while left > 0:
                        chunk = f.read(min(left, 8 << 20)); w.write(chunk); left -= len(chunk)
            else:
                R, I = flat2d(shape)
                packs, scales, errs = [], [], []
                for r0 in range(0, R, ROWS_PER_CHUNK):
                    r1 = min(r0 + ROWS_PER_CHUNK, R)
                    blk = read_rows(p, base, o0, dt, I, r0, r1)
                    if kind == "gs4":
                        pk, sc = quant_gs4(blk); rec = dequant_gs4(pk, sc, I)
                    else:
                        pk, sc = quant_i8(blk);  rec = dequant_i8(pk, sc)
                    n = float(np.linalg.norm(blk))
                    if n > 0: errs.append(float(np.linalg.norm(blk - rec)) / n)
                    packs.append(pk); scales.append(sc)
                w.write(np.concatenate(packs, 0).tobytes())
                w.write(np.concatenate(scales, 0).astype(np.float32).tobytes())
                if errs:
                    k = "attention" if RE_ATTN.search(name) else \
                        "shared"    if RE_SHARED.search(name) else \
                        "embed/head"if RE_EMB.search(name) else "mlp denso"
                    err_by_kind.setdefault(k, []).append(float(np.mean(errs)))
            done += 1
            if done % 100 == 0:
                print(f"      {done}/{len(plan)}  ({time.time()-t0:.0f}s)", flush=True)
    os.replace(tmp, a.out)
    sz = os.path.getsize(a.out)
    print(f"\n✅ scritto {a.out}  {sz/1e9:.2f} GB  in {time.time()-t0:.0f}s")
    print("\n=== errore di quantizzazione (L2 relativo, sui pesi VERI) ===")
    for k, v in sorted(err_by_kind.items()):
        print(f"  {k:12} {100*float(np.mean(v)):.3f}%   (su {len(v)} tensori)")

if __name__ == "__main__":
    main()
