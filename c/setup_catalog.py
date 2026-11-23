#!/usr/bin/env python3
"""Ready-to-run downloads for the one-step setup, and which of them fit a machine.

family_registry.py knows every engine and how to plan a model it can see on
disk. It has no notion of a model that is not downloaded yet: where to fetch it,
how big it is, how much RAM it needs before its shards exist to be measured.
That is this table. Each entry names a registry family, so the engine, build
target and accelerator support come from the registry, never from here
(tests/test_setup_catalog.py holds the two together).

Only checkpoints that run straight after a plain download are listed: a
pre-converted colibri container, or an official checkpoint the engine reads
natively. Models that need a conversion (GLM-5.3-Flash, OLMoE, Qwen3.8-27B) or a
post-download step with third-party packages (DeepSeek V4.1 Flash) stay in the
README's manual path.

Sizes are the repository listings (with the excludes applied), in GB of 10^9
bytes, checked against the Hugging Face API on 2026-10-02. The RAM figures come
from the measurements quoted in README.md and the per-model docs:

- ram_min_gb: the always-resident weights (the dense part) plus the smallest
  expert cache the engine runs with, plus a little for the operating system.
  Below it the model does not start or swaps.
- ram_good_gb: where it runs as the docs measured it: the whole expert set in
  RAM for the small models, a comfortable cache for the large ones.

After the download, `coli setup` runs the real planner (resource_plan.py) on the
files, which is the authority; these numbers only decide what to offer.
"""
import json
import os
from dataclasses import dataclass, field, fields

GB = 1_000_000_000


@dataclass(frozen=True)
class CatalogModel:
    id: str
    family: str                 # family_registry id
    name: str
    repo: str
    disk_gb: float
    ram_min_gb: float
    ram_good_gb: float
    dense_gb: float             # always-resident part, for the "what fits" line
    rank: int                   # higher = more capable
    size_class: str             # "small": fits in RAM on a normal PC; "large": streams from the SSD
    summary: str
    revision: str = "main"
    include: tuple = ()
    exclude: tuple = ()
    modality: str = "text"
    license_note: str = ""
    # (tool under tools/ run with the model folder, the file it creates)
    post_install: tuple = ()
    # The first published release whose prebuilt engines run this checkpoint.
    # Older release archives predate the engine support; build from source then.
    prebuilt_since: str = "1.12.1"
    doc: str = ""
    notes: tuple = field(default_factory=tuple)
    # What a token costs, as the engine holds the weights (GB; model_fit.py): always resident
    # (dense part, embeddings, heads), every routed expert, the dense bytes and all the bytes a
    # decode step reads, the layers it runs (a loop's passes counted), the tokens a forward gives
    # (0: the family's, model_fit.FAMILY), and the FAMILY entry when it is not the family's.
    # From the tensor headers of the files the setup downloads (ui/models.py's geometry): the
    # 16-bit dense bytes halved where the engine quantizes them to int8 as it loads them.
    resident_gb: float = 0.0
    experts_gb: float = 0.0
    dense_read_gb: float = 0.0
    active_gb: float = 0.0
    layers: int = 0
    tpf: float = 0.0
    cost: str = ""


CATALOG = (
    CatalogModel(
        id="qwen36-35b", family="qwen36", name="Qwen3.6-35B-A3B",
        repo="Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64", disk_gb=23.1,
        # docs/qwen36.md: peak RSS 17 GB with every expert cached (cap 256);
        # the minimum follows the same engine's measured 6.5 GB at 32 experts/layer.
        ram_min_gb=10, ram_good_gb=20, dense_gb=3.5, rank=30, size_class="small",
        summary="general chat with thinking and tools; int4-gs64 container",
        doc="docs/qwen36.md",
        resident_gb=2.96, experts_gb=18.12, dense_read_gb=1.94, active_gb=2.51, layers=40),
    CatalogModel(
        id="qwen3-coder-30b", family="qwen36", name="Qwen3-Coder-30B-A3B",
        repo="Justvugg/Qwen3-Coder-30B-A3B-colibri-int4", disk_gb=19.4,
        # docs/qwen36.md: 6.5 GB resident at 32 experts/layer, 15.2 GB with all 128.
        ram_min_gb=8, ram_good_gb=18, dense_gb=3.6, rank=20, size_class="small",
        summary="coding model with tool calls, no thinking; int4-gs64 container",
        prebuilt_since="2.0.0", doc="docs/qwen36.md#qwen3-coder-30b-a3b",
        resident_gb=1.85, experts_gb=16.31, dense_read_gb=1.23, active_gb=2.25, layers=48),
    CatalogModel(
        id="inkling2-mini", family="inkling", name="Inkling-2-Mini UD-Q4_K_XL",
        repo="unsloth/Inkling-2-Mini-GGUF",
        include=("Inkling-2-Mini-UD-Q4_K_XL.gguf", "mmproj-F16.gguf"), disk_gb=18.3,
        # docs/inkling.md: the dense part and the vision tower 2.5 GB, every expert 15.8 GB
        ram_min_gb=6, ram_good_gb=21, dense_gb=2.5, rank=30, size_class="small",
        summary="26.4B with pictures and tools, its middle layers run three passes; unsloth's GGUF",
        post_install=(("gguf_link", "gguf.json"),), prebuilt_since="2.2.0",
        doc="docs/inkling.md#inkling-2-mini",
        resident_gb=2.46, experts_gb=15.8, dense_read_gb=1.33, active_gb=2.32, layers=48, cost="inkling2"),
    CatalogModel(
        id="inkling2-mini-iq2", family="inkling", name="Inkling-2-Mini UD-IQ2_XXS",
        repo="unsloth/Inkling-2-Mini-GGUF",
        include=("Inkling-2-Mini-UD-IQ2_XXS.gguf", "mmproj-F16.gguf"), disk_gb=9.1,
        ram_min_gb=4, ram_good_gb=12, dense_gb=1.9, rank=25, size_class="small",
        summary="Inkling-2-Mini with 2-bit experts, for 8 to 16 GB of RAM; unsloth's GGUF",
        post_install=(("gguf_link", "gguf.json"),), prebuilt_since="2.2.0",
        doc="docs/inkling.md#inkling-2-mini",
        resident_gb=1.86, experts_gb=7.25, dense_read_gb=0.98, active_gb=1.43, layers=48, cost="inkling2"),
    CatalogModel(
        id="qwen3-asr-1.7b", family="qwen3_asr", name="Qwen3-ASR-1.7B",
        repo="Qwen/Qwen3-ASR-1.7B", disk_gb=4.5,
        ram_min_gb=6, ram_good_gb=8, dense_gb=4.0, rank=2, size_class="small",
        summary="speech-to-text ASR (52 languages/dialects), BF16 safetensors; MVP in progress",
        doc="docs/plans/2026-11-22-qwen3-asr-port.md"),
    CatalogModel(
        id="deepseek-v4-flash-reap", family="deepseek_v4", name="DeepSeek V4 Flash REAP 150B",
        repo="puwaer/DeepSeek-V4-Flash-0731-reap-150b", disk_gb=84.7,
        ram_min_gb=16, ram_good_gb=32, dense_gb=7.9, rank=40, size_class="large",
        summary="pruned DeepSeek V4 Flash (132 of 256 experts), official fp4/fp8 weights",
        doc="docs/deepseek-v4.md",
        resident_gb=7.74, experts_gb=75.88, dense_read_gb=7.74, active_gb=11.19, layers=43),
    CatalogModel(
        id="deepseek-v4-flash", family="deepseek_v4", name="DeepSeek V4 Flash",
        repo="deepseek-ai/DeepSeek-V4-Flash-0731", disk_gb=166.9,
        ram_min_gb=16, ram_good_gb=32, dense_gb=7.9, rank=50, size_class="large",
        summary="284B, official checkpoint, no conversion",
        doc="docs/deepseek-v4.md",
        resident_gb=18.65, experts_gb=147.17, dense_read_gb=7.79, active_gb=11.24, layers=43),
    CatalogModel(
        id="qwen38-flash-next", family="qwen38", name="Qwen3.8-Flash-Next",
        repo="Qwen/Qwen3.8-Flash-Next-FP8", revision="bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        disk_gb=185.6, ram_min_gb=24, ram_good_gb=32, dense_gb=8.0, rank=50, size_class="large",
        summary="125B + 51B n-gram, official FP8 checkpoint (pinned revision)",
        doc="docs/qwen38.md",
        resident_gb=13.49, experts_gb=120.81, dense_read_gb=8.62, active_gb=10.98, layers=48),
    CatalogModel(
        id="qwen38-flash-next-iq2xs", family="qwen38", name="Qwen3.8-Flash-Next UD-IQ2_XS",
        repo="unsloth/Qwen3.8-Flash-Next-GGUF",
        include=("UD-IQ2_XS/*", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"), disk_gb=80.0,
        # the dense part and the MTP head 8.9 GB, the experts 42.3 GB; the 28.8 GB of n-gram
        # tables are read a few rows a token from the file
        ram_min_gb=16, ram_good_gb=54, dense_gb=8.9, rank=44, size_class="large",
        summary="Qwen3.8-Flash-Next with 2-bit experts and its MTP head; unsloth's GGUF",
        post_install=(("gguf_link", "gguf.json"),), prebuilt_since="2.2.0", doc="docs/qwen38.md",
        resident_gb=8.88, experts_gb=42.28, dense_read_gb=4.4, active_gb=5.23, layers=48),
    CatalogModel(
        id="qwen38-flash-next-iq3", family="qwen38", name="Qwen3.8-Flash-Next UD-IQ3_XXS",
        repo="unsloth/Qwen3.8-Flash-Next-GGUF",
        include=("UD-IQ3_XXS/*", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"), disk_gb=87.8,
        ram_min_gb=16, ram_good_gb=62, dense_gb=8.7, rank=47, size_class="large",
        summary="Qwen3.8-Flash-Next with 3-bit experts and its MTP head; unsloth's GGUF",
        post_install=(("gguf_link", "gguf.json"),), prebuilt_since="2.2.0", doc="docs/qwen38.md",
        resident_gb=8.66, experts_gb=50.29, dense_read_gb=4.0, active_gb=4.98, layers=48),
    CatalogModel(
        id="mimo-v2.6-flash", family="mimo", name="MiMo-V2.6 Flash",
        repo="XiaomiMiMo/MiMo-V2.6-Flash-MOPD", disk_gb=171.8,
        # docs/mimo.md: 30.1 GB resident with 32 experts cached per layer, 49.8 GB with 64.
        ram_min_gb=32, ram_good_gb=52, dense_gb=24.0, rank=60, size_class="large",
        summary="309B with vision and tools, official checkpoint",
        # The engine never loads these three (docs/mimo.md, measured on Pro,
        # the same engine and layout).
        exclude=("dflash/*", "audio_tokenizer/*", "model_mtp.safetensors"),
        prebuilt_since="2.0.0", doc="docs/mimo.md",
        resident_gb=10.82, experts_gb=160.86, dense_read_gb=8.11, active_gb=13.14, layers=48),
    CatalogModel(
        id="glm-5.2", family="glm", name="GLM-5.2",
        repo="mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp", disk_gb=429.3,
        ram_min_gb=16, ram_good_gb=24, dense_gb=9.9, rank=80, size_class="large",
        summary="744B, the reference model; int4-gs64 with the int8 MTP head",
        doc="README.md#2-get-the-model",
        resident_gb=11.88, experts_gb=417.36, dense_read_gb=10.93, active_gb=23.97, layers=78),
    CatalogModel(
        id="glm-5.3", family="glm", name="GLM-5.3",
        repo="Justvugg/GLM-5.3-colibri-int4-g64", disk_gb=419.3,
        ram_min_gb=16, ram_good_gb=24, dense_gb=9.9, rank=80, size_class="large",
        summary="744B, int4-gs64, no MTP head (no speculative decoding)",
        doc="README.md#2-get-the-model",
        resident_gb=11.6, experts_gb=407.69, dense_read_gb=10.64, active_gb=23.38, layers=78),
    CatalogModel(
        id="mimo-v2.6-pro", family="mimo", name="MiMo-V2.6 Pro",
        repo="XiaomiMiMo/MiMo-V2.6-Pro-MOPD", disk_gb=563.6,
        ram_min_gb=54, ram_good_gb=64, dense_gb=32.4, rank=90, size_class="large",
        summary="1.02T with vision and tools, official checkpoint",
        exclude=("dflash/*", "audio_tokenizer/*", "model_mtp.safetensors"),
        prebuilt_since="2.0.0", doc="docs/mimo.md#pro",
        resident_gb=32.16, experts_gb=531.35, dense_read_gb=28.81, active_gb=39.88, layers=70),
    CatalogModel(
        id="inkling", family="inkling", name="Inkling",
        repo="nbeerbower/Inkling-colibri-int4", disk_gb=514.1,
        # README: ~120 GB of RAM as downloaded (bf16 dense, 49.4 GB resident);
        # the 25 GB path needs docs/inkling.md's dense conversion first.
        ram_min_gb=120, ram_good_gb=128, dense_gb=49.4, rank=90, size_class="large",
        summary="975B, int4 experts with bf16 dense weights",
        doc="docs/inkling.md",
        resident_gb=49.36, experts_gb=464.66, dense_read_gb=36.37, active_gb=47.26, layers=66),
    CatalogModel(
        id="kimi-k3", family="kimi", name="Kimi K3",
        repo="moonshotai/Kimi-K3", disk_gb=1561.0,
        ram_min_gb=32, ram_good_gb=64, dense_gb=20.0, rank=100, size_class="large",
        summary="2.8T, original checkpoint with native MXFP4 experts",
        # The repo ships tiktoken.model only; the engines read tokenizer.json.
        post_install=(("k3_tokenizer", "tokenizer.json"),),
        doc="docs/kimi_k3.md",
        resident_gb=114.4, experts_gb=1446.46, dense_read_gb=111.16, active_gb=136.99, layers=93),
    CatalogModel(
        id="kimi-k3.1-pro", family="kimi", name="Kimi K3.1 Pro",
        repo="moonshotai/Kimi-K3.1-Pro", disk_gb=1561.3,
        # K3's weights plus the DSA indexer (120 BF16 tensors, 0.36 GB)
        ram_min_gb=32, ram_good_gb=64, dense_gb=20.2, rank=110, size_class="large",
        summary="2.8T, K3 with sparse attention past 2048 tokens, native MXFP4 experts",
        # the same tiktoken.model as K3 (one sha256): the same tokenizer.json
        post_install=(("k3_tokenizer", "tokenizer.json"),),
        prebuilt_since="2.1.1", doc="docs/kimi_k3.md#kimi-k31-pro",
        resident_gb=114.76, experts_gb=1446.46, dense_read_gb=111.52, active_gb=137.35, layers=93),
    CatalogModel(
        id="qwen-image-2.1", family="qwen_image", name="Qwen-Image-2.1",
        repo="Qwen/Qwen-Image-2.1", disk_gb=33.1,
        # docs/qwen-image.md: 9.0 GB peak for one 768x512 image with the text
        # encoder loaded per prompt, 16.0 GB with everything resident.
        ram_min_gb=12, ram_good_gb=18, dense_gb=8.5, rank=10, size_class="small",
        modality="image", summary="text to image (pictures, not chat)",
        license_note="Qwen Research License: non-commercial use only",
        prebuilt_since="2.0.0", doc="docs/qwen-image.md"),
    CatalogModel(
        id="qwen-image-2.1-turbo", family="qwen_image", name="Qwen-Image-2.1-Turbo",
        repo="Qwen/Qwen-Image-2.1-Turbo", disk_gb=32.5,
        # the base model's text encoder (byte for byte) and the same DiT shapes, so the
        # same RAM; its VAE is the bf16 one (0.7 GB on disk, f32 in RAM as the base's)
        ram_min_gb=12, ram_good_gb=18, dense_gb=8.5, rank=10, size_class="small",
        modality="image", summary="text to image in 8 steps (Qwen-Image-2.1 distilled)",
        license_note="Qwen Research License: non-commercial use only",
        prebuilt_since="2.1.1", doc="docs/qwen-image.md#turbo"),
)


def catalog():
    """The built-in entries, plus any from the JSON list COLI_SETUP_CATALOG names:
    a mirror, a private repository, or a test fixture served by a fake hub. An
    extra entry with a built-in id replaces it."""
    entries = {entry.id: entry for entry in CATALOG}
    path = os.environ.get("COLI_SETUP_CATALOG")
    if path:
        with open(path, encoding="utf-8") as handle:
            extra = json.load(handle)
        known = {f.name for f in fields(CatalogModel)}
        geo = ("resident_gb", "experts_gb", "dense_read_gb", "active_gb", "layers", "tpf", "cost")
        for item in extra if isinstance(extra, list) else []:
            values = {k: tuple(v) if isinstance(v, list) else v
                      for k, v in item.items() if k in known}
            built_in = entries.get(values.get("id"))
            if built_in is not None and not values.get("active_gb"):   # a mirror: the same weights
                values.update({k: getattr(built_in, k) for k in geo if k not in values})
            entry = CatalogModel(**values)
            entries[entry.id] = entry
    return tuple(entries.values())


def by_id(model_id):
    for entry in catalog():
        if entry.id == model_id:
            return entry
    raise KeyError(model_id)


def version_tuple(text):
    parts = []
    for piece in str(text or "").lstrip("v").split("."):
        # Leading digits only. "0-rc1" is patch 0, not 1.
        digits = []
        for ch in piece:
            if not ch.isdigit():
                break
            digits.append(ch)
        parts.append(int("".join(digits)) if digits else 0)
    return tuple((parts + [0, 0, 0])[:3])


def platform_supported(entry, os_name, machine):
    """Engine builds the Makefile itself refuses (COLI_V4_SUPPORTED)."""
    machine = (machine or "").lower()
    if entry.family == "deepseek_v4":
        x86 = machine in ("x86_64", "amd64")
        arm = machine in ("aarch64", "arm64")
        return (x86 and os_name in ("linux", "win32")) or (arm and os_name in ("linux", "darwin"))
    return True


def assess(entry, ram_total, disk_free, *, os_name="linux", machine="x86_64",
           already_downloaded=0):
    """One entry against one machine. Returns (status, reason).

    status is one of: good, tight, needs-ram, needs-disk, unsupported."""
    if not platform_supported(entry, os_name, machine):
        return "unsupported", f"no {entry.family} engine for {os_name}/{machine}"
    need_disk = max(0.0, entry.disk_gb * GB * 1.02 + 2 * GB - (already_downloaded or 0))
    if ram_total is not None and ram_total < entry.ram_min_gb * GB:
        return "needs-ram", (f"needs {entry.ram_min_gb:g} GB of RAM "
                             f"(dense part {entry.dense_gb:g} GB plus the expert cache); "
                             f"this machine has {ram_total / GB:.1f} GB")
    if disk_free is not None and disk_free < need_disk:
        return "needs-disk", (f"needs {need_disk / GB:.0f} GB free for the download; "
                              f"{disk_free / GB:.0f} GB free there")
    if ram_total is not None and ram_total < entry.ram_good_gb * GB:
        return "tight", (f"runs with a small expert cache (needs {entry.ram_min_gb:g} GB, "
                         f"runs best with {entry.ram_good_gb:g} GB)")
    if entry.size_class == "small":
        return "good", f"the whole model fits in RAM ({entry.ram_good_gb:g} GB)"
    return "good", (f"dense part {entry.dense_gb:g} GB stays in RAM, the experts "
                    "stream from the SSD (a fast NVMe drive makes it faster)")


def recommend(ram_total, disk_free, *, os_name="linux", machine="x86_64",
              downloaded=None, include_images=True, fit=None, backends=None):
    """Every catalog entry assessed, the default marked.

    With `fit` (model_fit.machine_from_hw: the RAM and its measured bandwidth, the cards and
    theirs, the disk) and `backends` ({family: the backend choose_backend gives it here}), each
    text model that fits gets its estimated decode speed (row["est"], model_fit.estimate) and
    the default is model_fit.pick's: the one worth the most here, its capability discounted by how
    far below smooth chat it decodes (model_fit.worth). The menu lists the others after it in
    that order.

    Without `fit` (a caller with no machine to measure), the CPU alone with ram_total of RAM at
    model_fit.DEFAULT_RAM_GBS."""
    downloaded = downloaded or {}
    rows = []
    for entry in catalog():
        if entry.modality != "text" and not include_images:
            continue
        status, reason = assess(entry, ram_total, disk_free, os_name=os_name, machine=machine,
                                already_downloaded=downloaded.get(entry.id, 0))
        rows.append({"entry": entry, "status": status, "reason": reason,
                     "fits": status in ("good", "tight"), "recommended": False})
    if fit is None:   # nothing measured: the CPU alone, with model_fit's assumed RAM bandwidth
        fit = {"ram": ram_total or 0, "ram_avail": None, "ram_gbs": None, "cards": [], "integrated": None,
               "disk": None, "disk_gbs": None}
    return _recommend_measured(rows, fit, backends or {})


def _recommend_measured(rows, fit, backends):
    import model_fit
    for r in rows:
        if r["fits"] and r["entry"].modality == "text":
            backend = backends.get(r["entry"].family, "cpu")
            est = model_fit.estimate(r["entry"], fit, backend)
            if est.get("tier"):
                r["est"] = dict(est, backend=backend)
            elif r["entry"].active_gb:   # the figures say it cannot run here after all
                r.update(status="needs-ram", reason=est["why"], fits=False)
    chosen = model_fit.pick(rows)
    if chosen is not None:
        chosen["recommended"] = True

    def order(r):   # the pick, then what each is worth here (model_fit.worth), the images last
        e, est = r["entry"], r.get("est")
        return (not r["recommended"], e.modality != "text", not est,
                -(model_fit.worth(r) if est else 0), -((est or {}).get("tok_s") or 0), e.disk_gb)
    fitting = sorted((r for r in rows if r["fits"]), key=order)
    others = sorted((r for r in rows if not r["fits"]), key=lambda r: r["entry"].disk_gb)
    return fitting + others


def as_dict(row):
    """A recommendation row as plain data (the MCP server and --json)."""
    entry = row["entry"]
    return {
        "id": entry.id, "name": entry.name, "family": entry.family, "repo": entry.repo,
        "revision": entry.revision, "download_gb": entry.disk_gb,
        "ram_min_gb": entry.ram_min_gb, "ram_good_gb": entry.ram_good_gb,
        "dense_gb": entry.dense_gb, "kind": entry.modality,
        "runs": "from RAM" if entry.size_class == "small" else "streams experts from the SSD",
        "summary": entry.summary, "status": row["status"], "fits": row["fits"],
        "reason": row["reason"], "recommended": row["recommended"],
        "license_note": entry.license_note, "doc": entry.doc,
        "estimate": row.get("est"),
    }


FITS_EXPLAINED = ("fits = the dense part, which always stays in RAM, plus a minimum expert "
                  "cache fit in RAM, and the download fits on the disk")
