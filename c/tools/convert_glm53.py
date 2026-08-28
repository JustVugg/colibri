"""
GLM-5.3-Flash (zai-org/GLM-5.3-Flash) -> contenitore colibri.

Strategia DISK-SAFE, la stessa di convert_fp8_to_int4.py: scarica UNO shard,
lo converte, lo CANCELLA, passa al prossimo. Il picco su disco e' l'output che
cresce piu' UNO shard sorgente (~5 GB), mai i 328 GB del repo.

Cosa fa, per gruppo di tensori (i 92 tipi del checkpoint sono tutti classificati
esplicitamente: un nome non riconosciuto FERMA la conversione invece di essere
saltato in silenzio):

  - esperti routed (FP8 e4m3, blocchi 128x128, 97% dei byte)
        -> dequant -> int4 group-scaled gs64, stessa matematica del motore C
           (np.rint = lrintf, stesse soglie, stesso packing dei nibble)
        -> `nome` U8 + `nome.qs` F32 (fmt=4)
  - tutto il resto (densi, MLA, KDA, indexer, mHC, MTP, embed/head, vision)
        -> dequantizzato se FP8, salvato in BF16 (o F32 dove l'originale e' F32)

Perche' il resto resta BF16 e non int4: sono 9,7 B parametri su 321 (3%), cioe'
~18 GB su disco. Tenerli in alta precisione costa poco e lascia al motore la
scelta dei bit a load time (lo schema di kimi_k3.c, K3_BITS), cosi' ritarare la
precisione dei densi NON richiede di riscaricare 328 GB.

USO:
  python3 tools/convert_glm53.py --outdir /path/glm53_i4 --min-free-gb 30
  python3 tools/convert_glm53.py --outdir ... --limit-shards 1   # prova su 1 shard
  python3 tools/convert_glm53.py --indir /path/raw --outdir ...  # shard gia' locali
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = "zai-org/GLM-5.3-Flash"
# chat_template.jinja e' il template ufficiale delle conversazioni: senza,
# `coli chat` renderebbe i turni a modo suo e il modello risponderebbe a un
# prompt che non e' quello per cui e' stato addestrato.
META_FILES = ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "processor_config.json",
              "chat_template.jinja")


# ---------------------------------------------------------------- quantizzatore
def quant_int4_grouped(w, gs=64):
    """int4 group-scaled: una scala ogni `gs` elementi lungo l'asse di input.

    Copia bit-identica di quant_int4_grouped() in convert_fp8_to_int4.py: stesso
    ordine di operazioni, stesso np.rint, stesso packing dei nibble, cosi' i due
    contenitori si decodificano con lo stesso quant.h."""
    O, I = w.shape
    qmax = 7
    ngroups = (I + gs - 1) // gs
    Ipad = ngroups * gs
    wpad = np.zeros((O, Ipad), np.float32)
    wpad[:, :I] = w
    wr = wpad.reshape(O, ngroups, gs)
    amax = np.abs(wr).max(axis=2, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(wr / s), -8, qmax).astype(np.int32)
    q = q.reshape(O, Ipad)[:, :I]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, :, 0].astype(np.float32).reshape(-1)


# ---------------------------------------------------------------- classificazione
# I 92 tipi di tensore del checkpoint, per gruppo. Le voci sono suffissi o
# frammenti; l'ordine conta (il primo che combacia vince).
EXPERT_MARK = ".mlp.experts."
KEEP_F32 = (
    "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
    "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
    "dt_bias", "A_log", "e_score_correction_bias",
    "index_kpool_compress_ape",
    "norm.weight", "norm.bias", "layernorm.weight", "o_norm.weight",
    ".bias",
)
KNOWN_BF16 = (
    "embed_tokens.weight", "lm_head.weight",
    "mlp.gate.weight",                       # router
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "shared_experts.gate_proj.weight", "shared_experts.up_proj.weight",
    "shared_experts.down_proj.weight",
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "q_a_proj.weight", "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
    "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
    "f_a_proj.weight", "f_b_proj.weight", "b_proj.weight",
    "g_a_proj.weight", "g_b_proj.weight",
    "indexer.wq_b.weight", "indexer.wk.weight",
    "indexer.weights_proj.weight", "index_kpool_compress_gate",
    "eh_proj.weight",
    "visual.patch_embed.proj.weight", "visual.downsample.weight",
    "visual.merger.proj.weight", "visual.merger.gate_proj.weight",
    "visual.merger.up_proj.weight", "visual.merger.down_proj.weight",
    "attn.qkv.weight", "attn.proj.weight",
    "blocks.", "post_layernorm.weight",
)


def classify(name):
    """expert | f32 | bf16 | consumed. Solleva su un nome sconosciuto."""
    if name.endswith("_scale_inv"):
        return "consumed"                     # scala FP8: consumata col suo peso
    if EXPERT_MARK in name and name.endswith(".weight"):
        return "expert"
    for k in KEEP_F32:
        if name.endswith(k) or k in name:
            return "f32"
    for k in KNOWN_BF16:
        if name.endswith(k) or k in name:
            return "bf16"
    raise KeyError(name)


# ---------------------------------------------------------------- I/O tensori
def dequant(f, name, keys):
    """FP8 e4m3 con scale_inv a blocchi 128x128 -> f32; altrimenti f32 diretto."""
    import torch
    sl = f.get_slice(name)
    if sl.get_dtype() in ("F8_E4M3", "float8_e4m3fn"):
        w = f.get_tensor(name).to(torch.float32)
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)
        O, I = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        return (w * sc).numpy()
    return f.get_tensor(name).to(torch.float32).numpy()


def to_bf16(a):
    """f32 -> bf16 con round-to-nearest-even, come torch."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16)


def _partial_bytes(root):
    """Byte gia' scritti da hf_hub_download per lo shard in corso."""
    total = 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".incomplete") or name.endswith(".safetensors"):
                try:
                    total += os.path.getsize(os.path.join(base, name))
                except OSError:
                    pass
    return total


def fetch_hf(repo, filename, dest_dir, token, expected_size=0,
             stall_seconds=900, tries=30):
    """hf_hub_download (protocollo Xet nativo) sotto un guardiano di stallo.

    Le due strade hanno difetti opposti e complementari: il bridge HTTP di Xet
    (quello che vede curl) regge la ripresa ma serve ~40 KB/s, mentre il client
    nativo va a MB/s e pero' si e' appeso a meta' shard restando vivo un'ora
    senza scrivere un byte. Qui il download gira in un processo figlio e il
    padre guarda crescere i byte su disco: se non crescono per `stall_seconds`
    il figlio viene ucciso e si riprova, e hf_hub_download riparte dal
    `.incomplete` che ha lasciato.

    La finestra e' larga (15 minuti) di proposito: hf_xet NON riprende un
    `.incomplete` lasciato da un processo ucciso, ne apre uno nuovo, quindi un
    falso positivo non costa una riconnessione ma l'intero shard. Meglio
    accorgersi con qualche minuto di ritardo di un blocco vero che buttare via
    un'ora di download perche' la rete ha rallentato."""
    import subprocess
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, filename)
    code = (
        "import os,sys\n"
        "from huggingface_hub import hf_hub_download\n"
        "hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3],\n"
        "                token=os.environ.get('COLI_HF_TOKEN') or False)\n"
    )
    environment = dict(os.environ)
    if token:
        environment["COLI_HF_TOKEN"] = token
    def complete():
        """Un file c'e' solo se e' INTERO: un troncamento (un tentativo
        precedente interrotto) sarebbe accettato come buono e poi esploderebbe
        in safe_open, che e' esattamente com'e' stato scoperto."""
        if not os.path.exists(target):
            return False
        if expected_size and os.path.getsize(target) != expected_size:
            os.remove(target)
            return False
        return True

    for attempt in range(tries):
        if complete():
            return target
        child = subprocess.Popen([sys.executable, "-c", code, repo, filename, dest_dir],
                                 env=environment,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        last_bytes, last_move = _partial_bytes(dest_dir), time.time()
        while child.poll() is None:
            time.sleep(15)
            now = _partial_bytes(dest_dir)
            if now != last_bytes:
                last_bytes, last_move = now, time.time()
            elif time.time() - last_move > stall_seconds:
                print(f"[stallo] {filename}: fermo da {stall_seconds}s, riparto",
                      flush=True)
                child.kill()
                child.wait()
                break
        if complete():
            return target
        time.sleep(5)
    raise RuntimeError(f"download fallito dopo {tries} tentativi: {filename}")


def fetch_curl(repo, filename, dest_dir, expected_size, token=None,
               chunk_bytes=256 << 20, idle_limit=40):
    """Scarica per BLOCCHI espliciti, appendendo, su una rete che cade spesso.

    La macchina su cui gira questo ha una scheda di rete che si reinizializza
    ogni tanto: una singola connessione lunga muore a meta' shard, e ogni ripresa
    con `-C -` dipende da come il server tratta la Range su un redirect firmato.
    Chiedere invece blocchi da 256 MB con `-r inizio-fine` rende ogni pezzo
    indipendente e verificabile: una caduta costa al massimo un blocco (~un
    minuto a 4,5 MB/s), mai le ore gia' scaricate, e i byte sul disco sono
    sempre esattamente quelli che sappiamo di avere.

    Si arrende solo dopo `idle_limit` tentativi consecutivi che non aggiungono
    NEMMENO un byte: un link che oscilla continua, un link morto no."""
    import subprocess
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, filename)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    if not expected_size:
        raise RuntimeError(f"dimensione attesa sconosciuta per {filename}")
    idle = 0
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == expected_size:
            return path
        if have > expected_size:            # sorgente cambiato o file sporco
            os.remove(path)
            continue
        stop = min(have + chunk_bytes, expected_size) - 1
        command = ["curl", "-sL", "-4", "--retry", "3", "--retry-delay", "5",
                   "--speed-limit", "50000", "--speed-time", "120",
                   "-r", f"{have}-{stop}", "--output", "-", url]
        if token:
            command[1:1] = ["-H", f"Authorization: Bearer {token}"]
        with open(path, "ab") as sink:
            subprocess.run(command, stdout=sink, check=False)
        grown = (os.path.getsize(path) if os.path.exists(path) else 0) - have
        if grown <= 0:
            idle += 1
            if idle >= idle_limit:
                raise RuntimeError(
                    f"{filename}: nessun progresso in {idle_limit} tentativi "
                    f"({have}/{expected_size} byte)")
            time.sleep(min(5 * idle, 60))
        else:
            idle = 0


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def guard_space(outdir, min_free, min_free_c):
    """Ferma la conversione PRIMA di riempire il disco: WSL e, se montato, C:.

    Le scritture dentro WSL riusano i blocchi gia' liberati dentro il vhdx; il
    vhdx cresce (mangiando C:) solo oltre il suo massimo storico, ed e' proprio
    quel caso che questa guardia deve intercettare."""
    while True:
        wsl = free_gb(outdir)
        win = free_gb("/mnt/c") if os.path.isdir("/mnt/c") else float("inf")
        if wsl >= min_free and win >= min_free_c:
            return
        print(f"[WAIT] spazio: WSL {wsl:.0f} GB (min {min_free:.0f}), "
              f"C: {win:.0f} GB (min {min_free_c:.0f}) — libera spazio…",
              flush=True)
        time.sleep(60)


# ---------------------------------------------------------------- conversione
def convert_shard(path, out_path, group_size):
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    out = {}
    stats = {"expert": 0, "bf16": 0, "f32": 0}
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in sorted(keys):
            kind = classify(name)
            if kind == "consumed":
                continue
            w = dequant(f, name, keys)
            if kind == "expert":
                if w.ndim != 2:
                    raise ValueError(f"expert non 2D: {name} {w.shape}")
                q, s = quant_int4_grouped(w, group_size)
                out[name] = torch.from_numpy(q)
                out[name + ".qs"] = torch.from_numpy(s)
                stats["expert"] += 1
            elif kind == "f32":
                out[name] = torch.from_numpy(np.ascontiguousarray(w))
                stats["f32"] += 1
            else:
                out[name] = to_bf16(w)
                stats["bf16"] += 1
    save_file(out, out_path, metadata={"format": "pt"})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--indir", default=None, help="shard gia' scaricati")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    ap.add_argument("--min-free-c-gb", type=float, default=40.0,
                    help="spazio minimo da lasciare su C: (il vhdx cresce li')")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="converte solo i primi N shard (prova)")
    ap.add_argument("--keep-source", action="store_true",
                    help="non cancella lo shard sorgente dopo la conversione")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    done_path = os.path.join(a.outdir, ".converted.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()

    if a.indir:
        shards = sorted(x for x in os.listdir(a.indir) if x.endswith(".safetensors"))
        fetch = lambda fn: os.path.join(a.indir, fn)
    else:
        from huggingface_hub import hf_hub_download
        # Il token salvato in ~/.cache/huggingface puo' essere un OAuth scaduto
        # (lo era: "signature verification failed"). Si prova prima ~/.hf_token,
        # e in mancanza si scarica senza autenticazione: il repo e' pubblico.
        def _meta_token():
            path = os.path.expanduser("~/.hf_token")
            if os.environ.get("HF_TOKEN"):
                return os.environ["HF_TOKEN"]
            if os.path.isfile(path):
                return open(path).read().strip()
            return False

        def pull(fn, sub):
            return hf_hub_download(a.repo, fn, token=_meta_token(),
                                   local_dir=os.path.join(a.outdir, sub))
        index = json.load(open(pull("model.safetensors.index.json", "_meta")))
        shards = sorted(set(index["weight_map"].values()))
        # Dimensioni dichiarate dall'API: servono a distinguere "finito" da
        # "troncato", cosa che il solo exit code di curl non dice.
        sizes = {}
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(a.repo, files_metadata=True, token=False)
            sizes = {s.rfilename: s.size for s in info.siblings if s.size}
        except Exception as exc:                           # noqa: BLE001
            print(f"[size] metadati non disponibili ({exc}); "
                  f"nessun controllo di troncamento", flush=True)
        token = None
        for candidate in (os.environ.get("HF_TOKEN"),
                          os.path.expanduser("~/.hf_token")):
            if not candidate:
                continue
            if os.path.isfile(candidate):
                token = open(candidate).read().strip()
            elif candidate.startswith("hf_"):
                token = candidate
            if token:
                break
        print(f"token HF: {'presente' if token else 'assente (throttling)'}", flush=True)
        # curl sul bridge HTTP, non il client Xet nativo. Misurato in stato
        # calmo: 4,3 MB/s, cioe' la banda piena della macchina. Il client Xet
        # va altrettanto veloce ma NON riprende un file interrotto, e ogni
        # riavvio ricominciava da zero: con shard da 5 GB, una singola
        # connessione che riprende batte una veloce che riparte.
        fetch = lambda fn: fetch_curl(a.repo, fn, os.path.join(a.outdir, "_src"),
                                      sizes.get(fn, 0), token)
        for meta in META_FILES:
            try:
                shutil.copy(pull(meta, "_meta"), os.path.join(a.outdir, meta))
            except Exception as exc:                       # noqa: BLE001
                print(f"[meta] {meta}: {exc}", flush=True)

    if a.limit_shards:
        shards = shards[:a.limit_shards]
    todo = [s for s in shards if s not in done]
    print(f"shard: {len(shards)} totali, {len(todo)} da fare", flush=True)

    t_start = time.time()
    for i, shard in enumerate(todo, 1):
        guard_space(a.outdir, a.min_free_gb, a.min_free_c_gb)
        t0 = time.time()
        src = fetch(shard)
        t_dl = time.time() - t0
        out_path = os.path.join(a.outdir, shard)
        t1 = time.time()
        stats = convert_shard(src, out_path, a.group_size)
        if not a.indir and not a.keep_source:
            os.remove(src)
        size = os.path.getsize(out_path) / 1e9
        print(f"[{i}/{len(todo)}] {shard}  dl {t_dl:5.0f}s  conv {time.time()-t1:5.0f}s"
              f"  out {size:5.2f} GB  experts {stats['expert']:4d}"
              f"  bf16 {stats['bf16']:3d}  f32 {stats['f32']:3d}"
              f"  | WSL {free_gb(a.outdir):.0f} GB  C: {free_gb('/mnt/c'):.0f} GB",
              flush=True)
        done.add(shard)
        json.dump(sorted(done), open(done_path, "w"))

    total = sum(os.path.getsize(os.path.join(a.outdir, s)) / 1e9
                for s in done if os.path.exists(os.path.join(a.outdir, s)))
    print(f"\nfatto: {len(done)} shard, {total:.1f} GB in {(time.time()-t_start)/60:.0f} min",
          flush=True)


if __name__ == "__main__":
    main()
