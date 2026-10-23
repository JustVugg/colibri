#!/usr/bin/env python3
"""Qwen-Image-2.1 text-to-image reference, one component at a time.

The C engine (c/qwenimage) computes in float32, so this oracle does too by
default: the bf16 weights are upcast and every op runs in f32. The pipeline is
never loaded whole. The text encoder is loaded, runs and is freed, then the
transformer, then the VAE, so the real model peaks at about 31 GB (the f32
language model) instead of the ~75 GB the three f32 components take together.
The vision tower and the LM head are dropped before the upcast: text-to-image
reads neither.

Every stage calls the diffusers code itself (QwenImage21Pipeline.encode_prompt,
QwenImage21Pipeline.__call__ with output_type="latent", then the pipeline's own
VAE tail), and forward hooks record what goes through. The initial latents are
explicit and saved, so no RNG has to be matched.

  # primary oracle, f32, all dumps and the step-to-step redundancy table
  python3 tools/qwenimage_ref.py --model /path/qwen-image-2.1 --out ref_fox \\
      --prompt "a red fox sitting in the snow, photograph" --size 512x512 \\
      --steps 8 --seed 42 --dump --redundancy

  # steps sweep in bf16 compute, reusing the oracle's prompt embeddings and latents
  python3 tools/qwenimage_ref.py --model ... --out sweep --size 512x512 \\
      --steps 8,16,30 --dtype bf16 --embeds ref_fox/ref.safetensors \\
      --latents ref_fox/ref.safetensors

Output (per run directory): image.png, ref.json (config values a C
implementation needs, ids, schedule, timings, peak RSS, versions) and, with
--dump, ref.safetensors. make_qwenimage_tiny.py writes the same set for the
tiny random model. Tensor layouts in ref.safetensors (batch axis dropped):
  input_ids [T] i32            full ids fed to the text encoder
  te_hidden_NN [T][hidden]     text encoder hidden states (00 = embeddings,
                               last = final layer output before the norm)
  prompt_embeds [T-drop][hidden]  what the transformer reads
  latents_init [N][C]          packed initial latents, N = (H/16)*(W/16)
  sigmas [steps+1], timesteps [steps], timestep_in [steps] (t/1000 as fed)
  noise_pred_NNN [N][C], latents_NNN [N][C] (after step NNN), latents_final
  step0_*                      step 0 internals (see ref.json "tensors")
  vae_in [C][h][w]             latents after de-normalization
  vae_out_raw [4][H][W]        decoder output before the clamp
  vae_out [4][H][W]            after the clamp to [-1, 1]
  rgba [H][W][4] u8            the pipeline's postprocess, top row first

Reference stack (the ds venv this was written and run against):
  diffusers 0.41.0.dev0, git commit 80c7ed262aeffbeb43ef13ae04baeb9b84515a69
  (pip install "git+https://github.com/huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69")
  transformers 5.17.0, torch 2.14.0+cpu, tokenizers 0.23.2, safetensors 0.8.0,
  numpy 2.5.3, pillow 12.3.0, python 3.14.
"""
import argparse
import ctypes
import gc
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

VAE_SCALE = 16          # QwenImage21Pipeline.vae_scale_factor: one latent token per 16x16 pixels
RATIO_EDGES = (0.01, 0.05, 0.10)


# ---------------------------------------------------------------- memory and timing

def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return float("inf")


def rss_gb(kind: str = "RssAnon") -> float:
    """Anonymous RSS by default: safetensors can be memory-mapped, and clean file pages the kernel can drop
    at will must not count against the cap."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(kind + ":"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def release_memory():
    gc.collect()
    try:  # glibc keeps freed arenas otherwise, and the next component would stack on top
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (anon {rss_gb():.1f} GB, file {rss_gb('RssFile'):.1f} GB, "
          f"avail {mem_available_gb():.1f} GB, load {os.getloadavg()[0]:.1f})", flush=True)


class MemoryGuard:
    """Waits until MemAvailable leaves room for a component, and aborts an upcast that overshoots.

    ds is shared: other jobs come and go, and swapping would slow them all down. A component is only
    loaded once MemAvailable exceeds what it needs plus a margin; the upcast loops call check() so a
    wrong estimate stops the run instead of pushing the machine into swap."""

    def __init__(self, max_rss_gb: float, wait_minutes: float, margin_gb: float = 4.0):
        self.max_rss_gb = max_rss_gb
        self.wait_s = wait_minutes * 60.0
        self.margin_gb = margin_gb

    def wait_for(self, need_gb: float, what: str) -> bool:
        deadline = time.time() + self.wait_s
        while True:
            avail = mem_available_gb()
            if avail >= need_gb + self.margin_gb:
                return True
            if time.time() >= deadline:
                log(f"{what}: needs {need_gb:.1f} GB + {self.margin_gb:.0f} GB margin, only {avail:.1f} GB available")
                return False
            log(f"{what}: waiting for memory (need {need_gb:.1f} + {self.margin_gb:.0f} GB)")
            time.sleep(30)

    def check(self, what: str):
        rss = rss_gb()
        if rss > self.max_rss_gb:
            raise MemoryError(f"{what}: anonymous RSS {rss:.1f} GB is over the {self.max_rss_gb:.0f} GB cap")


def dir_size_gb(path: Path, pattern: str = "*.safetensors") -> float:
    return sum(p.stat().st_size for p in Path(path).glob(pattern)) / 1073741824.0


# ---------------------------------------------------------------- component loading

def load_text_encoder(model_dir: Path, dtype: torch.dtype, guard: MemoryGuard | None = None):
    from transformers import Qwen3VLForConditionalGeneration

    te = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir / "text_encoder"), dtype=torch.bfloat16)
    te.eval()
    # Text-to-image never runs the vision tower or the LM head (only the hidden states are read). Dropping them
    # before the upcast saves ~5 GB of f32 on the real model. The forward still calls lm_head, so it becomes
    # an identity instead of disappearing.
    te.model.visual = None
    te.lm_head = torch.nn.Identity()
    release_memory()
    if dtype == torch.float32:
        lm = te.model.language_model
        lm.embed_tokens.float()
        for i, layer in enumerate(lm.layers):
            layer.float()
            if guard is not None and i % 4 == 3:
                release_memory()
                guard.check(f"text encoder upcast, layer {i}")
        lm.norm.float()
        rot = lm.rotary_emb
        if rot.rope_type != "default":
            raise SystemExit(f"unexpected rope_type {rot.rope_type!r} in the text encoder")
        # inv_freq is a non-persistent buffer: recompute it in f32 rather than trust whatever dtype the load left
        inv_freq, _ = rot.compute_default_rope_parameters(rot.config)
        rot.inv_freq = inv_freq.float()
        rot.original_inv_freq = inv_freq.float().clone()
    release_memory()
    return te


def load_transformer(model_dir: Path, dtype: torch.dtype, guard: MemoryGuard | None = None):
    from diffusers import QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TemporalTimesteps

    tr = QwenImage21Transformer2DModel.from_pretrained(str(model_dir / "transformer"), torch_dtype=torch.bfloat16)
    tr.eval()
    if dtype == torch.float32:
        # block by block, so each bf16 block is released as its f32 copy appears
        for name, child in tr.named_children():
            if name == "transformer_blocks":
                for i, block in enumerate(child):
                    block.float()
                    if guard is not None and i % 4 == 3:
                        release_memory()
                        guard.check(f"transformer upcast, block {i}")
            else:
                child.float()
        # the sinusoid frequencies are a non-persistent buffer: rebuild them in f32 exactly as __init__ does
        tp = tr.time_text_embed.time_proj
        fresh = QwenImage21TemporalTimesteps(tp.timestep_dim, time_factor=tp.time_factor)
        tp.freqs = fresh.freqs.float()
    release_memory()
    return tr


def load_vae(model_dir: Path):
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(str(model_dir / "vae"), torch_dtype=torch.float32)
    vae.eval()
    return vae


PROCESSOR_NOTE = {"how": "Qwen3VLProcessor.from_pretrained"}


def load_processor(model_dir: Path):
    """Qwen3VLProcessor as the pipeline loads it, or, when torchvision is missing (the video processor and the
    fast image processor need it), the same class assembled from the tokenizer, the PIL image processor and
    the chat template with no video processor. Text-to-image calls neither image nor video processor, so the
    ids, the attention mask and the chat template rendering are the same; make_qwenimage_tiny.py checks the
    ids against the raw tokenizers library either way."""
    from transformers import Qwen3VLProcessor

    d = model_dir / "processor"
    try:
        proc = Qwen3VLProcessor.from_pretrained(str(d))
        PROCESSOR_NOTE["how"] = "Qwen3VLProcessor.from_pretrained"
        return proc
    except ImportError as e:
        if "orchvision" not in str(e):
            raise
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from transformers.processing_utils import ProcessorMixin

    tokenizer = AutoTokenizer.from_pretrained(str(d))
    image_processor = Qwen2VLImageProcessorPil.from_pretrained(str(d))
    strict = ProcessorMixin.check_argument_for_proper_class

    def allow_missing_video(self, name, arg):
        if arg is None and name == "video_processor":
            return None
        return strict(self, name, arg)

    ProcessorMixin.check_argument_for_proper_class = allow_missing_video
    try:
        proc = Qwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=None,
                                chat_template=(d / "chat_template.jinja").read_text(encoding="utf-8"))
    finally:
        ProcessorMixin.check_argument_for_proper_class = strict
    PROCESSOR_NOTE["how"] = ("Qwen3VLProcessor(tokenizer, Qwen2VLImageProcessorPil, video_processor=None, "
                             "chat_template.jinja): torchvision is not installed, t2i never calls the image or "
                             "video processor")
    return proc


def load_scheduler(model_dir: Path):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_dir / "scheduler"))


def make_pipeline(model_dir: Path, processor, text_encoder=None, transformer=None, vae=None):
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline(scheduler=load_scheduler(model_dir), vae=vae, text_encoder=text_encoder,
                               processor=processor, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------- stages

def f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).contiguous().clone()


def encode_prompts(model_dir: Path, prompts: list[str], dtype: torch.dtype, dump: bool,
                   guard: MemoryGuard | None = None) -> list[dict]:
    """Runs QwenImage21Pipeline.encode_prompt for each prompt and records ids and hidden states."""
    processor = load_processor(model_dir)
    t0 = time.time()
    te = load_text_encoder(model_dir, dtype, guard)
    load_s = time.time() - t0
    log(f"text encoder loaded ({dtype}) in {load_s:.1f} s")
    pipe = make_pipeline(model_dir, processor, text_encoder=te)

    captured = {}
    lm = te.model.language_model

    def pre_hook(module, args, kwargs):
        captured["input_ids"] = kwargs["input_ids"].detach().clone()
        captured["attention_mask"] = kwargs["attention_mask"].detach().clone()

    def post_hook(module, args, kwargs, output):
        if dump:
            captured["hidden"] = [f32(h[0]) for h in output.hidden_states]

    def last_layer_hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["last_layer_out"] = f32(out[0])

    handles = [te.register_forward_pre_hook(pre_hook, with_kwargs=True),
               te.register_forward_hook(post_hook, with_kwargs=True),
               lm.layers[-1].register_forward_hook(last_layer_hook)]
    results = []
    try:
        for prompt in prompts:
            captured.clear()
            t1 = time.time()
            with torch.no_grad():
                prompt_embeds, mask, image_pad_mask = pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"))
            enc_s = time.time() - t1
            enc_load = os.getloadavg()[0]
            if mask is not None and not bool(mask.all()):
                raise SystemExit("unexpected padding mask for a single prompt")
            if bool(image_pad_mask.any()):
                raise SystemExit("unexpected image tokens in a text-to-image prompt")
            drop = pipe._drop_idx
            ids = captured["input_ids"][0]
            # The pipeline feeds the transformer the last decoder layer's output BEFORE the final norm; check
            # that is really what came out, since transformers 5 changed what hidden_states[-1] means.
            pre_norm = torch.equal(f32(prompt_embeds[0]), captured["last_layer_out"][drop:])
            if not pre_norm:
                raise SystemExit("prompt_embeds is not the pre-norm last layer output: the norm hook did not apply")
            rec = {
                "prompt": prompt,
                "template_text": pipe.prompt_template_t2i.format(prompt if prompt else " "),
                "system_prompt": pipe.sys_prompt,
                "input_ids": [int(v) for v in ids],
                "drop_idx": int(drop),
                "prompt_embeds": f32(prompt_embeds[0]),
                "prompt_embeds_is_prenorm_last_layer": pre_norm,
                "encode_s": enc_s,
                "encode_loadavg": enc_load,
            }
            if dump:
                rec["te_hidden"] = captured["hidden"]
            results.append(rec)
            log(f"encoded {len(ids)} tokens (drop {drop}) in {enc_s:.2f} s: {prompt!r}")
    finally:
        for h in handles:
            h.remove()
    info = {"load_s": load_s, "dtype": str(dtype).replace("torch.", ""),
            "config": text_encoder_config(te), "peak_rss_gb": peak_rss_gb()}
    del pipe, te, lm, handles
    release_memory()
    log("text encoder freed")
    return results, info


def text_encoder_config(te) -> dict:
    c = te.config.text_config
    rp = dict(c.rope_parameters)
    return {
        "hidden_size": c.hidden_size, "num_hidden_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads, "num_key_value_heads": c.num_key_value_heads,
        "head_dim": c.head_dim, "intermediate_size": c.intermediate_size, "vocab_size": c.vocab_size,
        "rms_norm_eps": c.rms_norm_eps, "hidden_act": c.hidden_act,
        "rope_theta": rp.get("rope_theta"), "mrope_section": rp.get("mrope_section"),
        "mrope_interleaved": rp.get("mrope_interleaved"),
        "note": "text-only input: the three M-RoPE position components are equal (0..T-1), so the rotation is "
                "plain RoPE over the full head_dim with rope_theta, rotate_half pairing (i, i + head_dim/2); "
                "q_norm/k_norm are per-head RMSNorm before RoPE; attention is causal; the output read is the "
                "last layer before the final norm",
    }


class DenoiseRecorder:
    """Forward hooks on the transformer for one pipeline call.

    Per call (= per step): the timestep as fed and the model output for the target tokens. Step 0 also
    records txt_in, temb, the shared modulation, the RoPE table and block 0 in/out over the whole joint
    sequence. With redundancy on, every block's target-token output (and its residual delta, output minus
    input) is compared with the previous step's."""

    def __init__(self, transformer, n_target: int, dump: bool, redundancy: bool):
        self.tr = transformer
        self.n = n_target
        self.dump = dump
        self.redundancy = redundancy
        self.step = -1
        self.timestep_in = []
        self.noise_pred = []
        self.latents = []
        self.step_s = []
        self.step_load = []
        self.step0 = {}
        self.stats = {"blocks": {}, "noise_pred": [], "latents": []}
        self.prev_out = {}
        self.prev_delta = {}
        self.cur_in = {}
        self.prev_noise = None
        self.prev_latents = None
        self._t0 = None
        self.handles = []
        tr = transformer
        self.handles.append(tr.register_forward_pre_hook(self._model_pre, with_kwargs=True))
        self.handles.append(tr.register_forward_hook(self._model_post))
        if dump:
            self.handles.append(tr.txt_in.register_forward_hook(self._step0("txt_in")))
            self.handles.append(tr.time_text_embed.register_forward_hook(self._step0("temb")))
            self.handles.append(tr.modulation.register_forward_hook(self._step0("modulation")))
            self.handles.append(tr.pos_embed.register_forward_hook(self._rope))
        for k, block in enumerate(tr.transformer_blocks):
            if redundancy or (dump and k == 0):
                self.handles.append(block.register_forward_pre_hook(self._block_pre(k), with_kwargs=True))
                self.handles.append(block.register_forward_hook(self._block_post(k)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _model_pre(self, module, args, kwargs):
        self.step += 1
        self.timestep_in.append(float(kwargs["timestep"].float()[0]))
        self._t0 = time.time()

    def _model_post(self, module, args, output):
        self.step_s.append(time.time() - self._t0)
        self.step_load.append(os.getloadavg()[0])
        noise = f32(output[0][0, -self.n:])
        if self.dump:
            self.noise_pred.append(noise)
        if self.redundancy:
            if self.prev_noise is not None:
                self.stats["noise_pred"].append({"step": self.step, **change_stats(noise, self.prev_noise)})
            self.prev_noise = noise
        log(f"step {self.step} transformer {self.step_s[-1]:.2f} s")

    def _step0(self, name):
        def hook(module, args, output):
            if self.step == 0:
                self.step0[name] = f32(output[0] if name == "txt_in" else output)
        return hook

    def _rope(self, module, args, output):
        if self.step == 0:
            self.step0["rope_cos"] = f32(output.real)
            self.step0["rope_sin"] = f32(output.imag)

    def _block_pre(self, k):
        def hook(module, args, kwargs):
            h = kwargs["hidden_states"][0]
            if self.dump and k == 0 and self.step == 0:
                self.step0["block0_in"] = f32(h)
            if self.redundancy:
                self.cur_in[k] = f32(h[-self.n:])
        return hook

    def _block_post(self, k):
        def hook(module, args, output):
            h = output[0]
            if self.dump and k == 0 and self.step == 0:
                self.step0["block0_out"] = f32(h)
            if self.redundancy:
                out = f32(h[-self.n:])
                delta = out - self.cur_in.pop(k)
                if k in self.prev_out:
                    entry = {"step": self.step, "out": change_stats(out, self.prev_out[k]),
                             "delta": change_stats(delta, self.prev_delta[k])}
                    self.stats["blocks"].setdefault(k, []).append(entry)
                self.prev_out[k] = out
                self.prev_delta[k] = delta
        return hook

    def on_step_end(self, pipe, i, t, kwargs):
        lat = f32(kwargs["latents"][0])
        if self.dump:
            self.latents.append(lat)
        if self.redundancy:
            if self.prev_latents is not None:
                self.stats["latents"].append({"step": i, **change_stats(lat, self.prev_latents)})
            self.prev_latents = lat
        return {}


def change_stats(cur: torch.Tensor, prev: torch.Tensor) -> dict:
    """Step-to-step change of a [tokens][dim] tensor: global cosine, and the per-token relative L2 change."""
    cur = cur.double()
    prev = prev.double()
    cos = float((cur * prev).sum() / (cur.norm() * prev.norm()).clamp_min(1e-30))
    rel = (cur - prev).norm(dim=-1) / prev.norm(dim=-1).clamp_min(1e-30)
    q = torch.quantile(rel, torch.tensor([0.5, 0.9], dtype=rel.dtype))
    out = {"cos": cos, "rel_mean": float(rel.mean()), "rel_median": float(q[0]), "rel_p90": float(q[1]),
           "rel_global": float((cur - prev).norm() / prev.norm().clamp_min(1e-30))}
    for edge in RATIO_EDGES:
        out[f"frac_below_{int(round(edge * 100))}pct"] = float((rel < edge).double().mean())
    return out


def pack_initial_latents(seed: int, channels: int, width: int, height: int) -> torch.Tensor:
    """The pipeline's prepare_latents with an explicit CPU generator: randn (1, 1, C, h, w), then packed."""
    from diffusers import QwenImage21Pipeline

    h = 2 * (height // (VAE_SCALE * 2))
    w = 2 * (width // (VAE_SCALE * 2))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn((1, 1, channels, h, w), generator=gen, dtype=torch.float32)
    return QwenImage21Pipeline._pack_latents(lat, 1, channels, h, w)[0].contiguous()


def denoise(model_dir: Path, runs: list[dict], dtype: torch.dtype, dump: bool, redundancy: bool,
            guard: MemoryGuard | None = None):
    """runs: dicts with prompt_embeds [T][D] f32, latents_init [N][C] f32, width, height, steps.
    One transformer load serves every run. Returns per-run outputs and load info."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    processor = load_processor(model_dir)
    t0 = time.time()
    tr = load_transformer(model_dir, dtype, guard)
    load_s = time.time() - t0
    log(f"transformer loaded ({dtype}) in {load_s:.1f} s")
    pipe = make_pipeline(model_dir, processor, transformer=tr)
    outs = []
    for run in runs:
        width, height, steps = run["width"], run["height"], run["steps"]
        n = run["latents_init"].shape[0]
        rec = DenoiseRecorder(tr, n, dump and run.get("dump", True), redundancy and run.get("redundancy", True))
        t1 = time.time()
        try:
            with torch.no_grad():
                final = pipe(prompt_embeds=run["prompt_embeds"][None].to(dtype),
                             latents=run["latents_init"][None].to(dtype),
                             width=width, height=height, num_inference_steps=steps, true_cfg_scale=1.0,
                             output_type="latent", callback_on_step_end=rec.on_step_end,
                             return_dict=False)[0]
        finally:
            rec.remove()
        total_s = time.time() - t1
        sc = pipe.scheduler.config
        mu = calculate_shift(n, sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096),
                             sc.get("base_shift", 0.5), sc.get("max_shift", 1.15))
        outs.append({
            "final": f32(final[0]), "sigmas": f32(pipe.scheduler.sigmas), "timesteps": f32(pipe.scheduler.timesteps),
            "mu": float(mu), "timestep_in": rec.timestep_in, "noise_pred": rec.noise_pred, "latents": rec.latents,
            "step0": rec.step0, "step_s": rec.step_s, "step_load": rec.step_load, "denoise_s": total_s,
            "redundancy": rec.stats if rec.redundancy else None,
        })
        rec = None  # it holds the transformer; the model is freed below only if nothing else does
        log(f"denoised {width}x{height}, {steps} steps in {total_s:.1f} s")
    info = {"load_s": load_s, "dtype": str(dtype).replace("torch.", ""), "config": dict(tr.config),
            "peak_rss_gb": peak_rss_gb()}
    del pipe, tr, rec
    release_memory()
    log("transformer freed")
    return outs, info


def decode(model_dir: Path, items: list[dict], dump: bool):
    """items: dicts with final [N][C] packed latents, width, height. The tail of QwenImage21Pipeline.__call__,
    verbatim, plus a hook on the decoder for the pre-clamp output."""
    processor = load_processor(model_dir)
    t0 = time.time()
    vae = load_vae(model_dir)
    load_s = time.time() - t0
    log(f"VAE loaded in {load_s:.1f} s")
    pipe = make_pipeline(model_dir, processor, vae=vae)
    captured = {}
    handle = vae.decoder.register_forward_hook(lambda m, a, out: captured.__setitem__("raw", f32(out[0, :, 0])))
    outs = []
    try:
        for item in items:
            width, height = item["width"], item["height"]
            t1 = time.time()
            with torch.no_grad():
                latents = pipe._unpack_latents(item["final"][None], height, width, pipe.vae_scale_factor)
                latents = latents.to(pipe.vae.dtype)
                latents_mean = (torch.tensor(pipe.vae.config.latents_mean)
                                .view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype))
                latents_std = (torch.tensor(pipe.vae.config.latents_std)
                               .view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype))
                latents = latents * latents_std + latents_mean
                image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
                pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
            dec_s = time.time() - t1
            out = {"pil": pil, "rgba": torch.from_numpy(np.array(pil)).contiguous(), "decode_s": dec_s,
                   "loadavg": os.getloadavg()[0]}
            if dump:
                out["vae_in"] = f32(latents[0, :, 0])
                out["vae_out"] = f32(image[0])
                out["vae_out_raw"] = captured["raw"]
            outs.append(out)
            log(f"decoded {width}x{height} in {dec_s:.1f} s")
    finally:
        handle.remove()
    vcfg = dict(vae.config)
    info = {"load_s": load_s, "dtype": "float32", "config": vcfg, "peak_rss_gb": peak_rss_gb()}
    del pipe, vae, handle
    release_memory()
    return outs, info


# ---------------------------------------------------------------- writing

def versions() -> dict:
    import diffusers
    import safetensors
    import tokenizers
    import transformers

    commit = None
    try:
        from importlib.metadata import distribution
        du = distribution("diffusers").read_text("direct_url.json")
        if du:
            commit = json.loads(du).get("vcs_info", {}).get("commit_id")
    except Exception:
        pass
    return {"diffusers": diffusers.__version__, "diffusers_commit": commit, "transformers": transformers.__version__,
            "torch": torch.__version__, "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__,
            "numpy": np.__version__, "python": platform.python_version()}


def machine() -> dict:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"host": platform.node(), "cpu": cpu, "torch_threads": torch.get_num_threads()}


def model_constants(te_cfg: dict, tr_cfg: dict, vae_cfg: dict, sched_cfg: dict) -> dict:
    """Everything beyond the config files a C implementation needs, stated once."""
    inner = tr_cfg["num_attention_heads"] * tr_cfg["attention_head_dim"]
    return {
        "text_encoder": te_cfg,
        "transformer": {
            **{k: tr_cfg[k] for k in ("in_channels", "out_channels", "num_layers", "attention_head_dim",
                                     "num_attention_heads", "context_in_dim", "mlp_ratio", "axes_dims_rope",
                                     "eps", "causal_condition", "patch_size")},
            "inner_dim": inner, "mlp_hidden": inner * tr_cfg["mlp_ratio"],
            "rope_theta": 10000, "rope_pos_table": "index i >= 0 for 0..8191, negative index -k at row 8192+1024-k",
            "timestep_embedding": {"dim": 256, "max_period": 10000, "time_factor": 1000.0,
                                   "layout": "[cos(args) | sin(args)], args = 1000 * t * exp(-ln(10000) * i / 128)",
                                   "input": "t = timestep / 1000 as fed (timestep_in); linear_1, SiLU, linear_2, "
                                            "no biases"},
            "modulation": "SiLU(temb) @ modulation.1.weight^T -> [scale1 | gate1 | scale2 | gate2], shared by "
                          "every block; row 0 = sampled t (target tokens), row 1 = t = 0 (text tokens)",
            "block": "x += tanh(gate1) * attn(LN(x) * (1 + scale1)); x += tanh(gate2) * mlp(LN(x) * (1 + scale2)); "
                     "LN without affine, eps; mlp = out(silu(gate_layer(x)) * proj(x))",
            "attention": "to_q/to_k/to_v, per-head RMSNorm (norm_q/norm_k, weight, eps), 3-axis RoPE as complex "
                         "pairs (x[2i], x[2i+1]), SDPA scale 1/sqrt(head_dim), to_out.0; no biases anywhere",
            "mask": "text tokens causal among themselves and never see the image; image tokens see all text and "
                    "all image tokens. So the text K/V are step-independent (the pipeline's KV cache)",
            "txt_in": "zero-centered RMSNorm (x * rsqrt(mean(x^2) + eps) * (1 + w)), in_layer, GELU tanh, out_layer",
            "norm_out": "LN(x) * (1 + linear(SiLU(temb))), then proj_out",
            "rope_positions": "text token j: (j, j, j); target image token (r, c) of an h x w grid: frame = T_text, "
                              "height = r - (h - h//2), width = c - (w - w//2)",
            "sequence": "[prompt_embeds (after drop_idx) | target latents]",
        },
        "vae": {
            **{k: vae_cfg[k] for k in ("z_dim", "base_dim", "decoder_base_dim", "dim_mult", "num_res_blocks",
                                      "in_channels", "out_channels", "is_residual", "temperal_downsample",
                                      "scale_factor_spatial", "latents_mean", "latents_std", "attn_scales")},
            "denormalize": "z = latents * latents_std + latents_mean (per channel)",
            "rms_norm": "F.normalize over channels (eps 1e-12) * sqrt(C) * gamma",
            "clamp": [-1.0, 1.0],
            "postprocess": "u8 = round(clamp(x / 2 + 0.5, 0, 1) * 255), RGBA, top row first",
        },
        "scheduler": dict(sched_cfg),
        "pipeline": {"vae_scale_factor": VAE_SCALE, "multiple_of": 32, "true_cfg_scale": 1.0, "use_kv_cache": True,
                     "calculate_shift": "mu = N * m + b, m = (max_shift - base_shift) / (max_image_seq_len - "
                                        "base_image_seq_len), b = base_shift - m * base_image_seq_len",
                     "sigmas": "s = linspace(1, 1/steps, steps) as f32; s = e^mu / (e^mu + (1/s - 1)); "
                               "stretch so the last is shift_terminal; append 0",
                     "euler": "latents += (sigma[i+1] - sigma[i]) * noise_pred",
                     "timestep_fed": "t = sigma * 1000, fed as t / 1000 (see timestep_in)"},
    }


def tensor_table(tensors: dict) -> dict:
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in tensors.items()}


def write_run(out_dir: Path, enc: dict, den: dict, dec: dict, meta: dict, dump: bool):
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    dec["pil"].save(out_dir / "image.png")
    tensors = {}
    if dump:
        tensors["input_ids"] = torch.tensor(enc["input_ids"], dtype=torch.int32)
        for i, h in enumerate(enc.get("te_hidden", [])):
            tensors[f"te_hidden_{i:02d}"] = h
        tensors["prompt_embeds"] = enc["prompt_embeds"]
        tensors["latents_init"] = meta["latents_init"]
        tensors["sigmas"] = den["sigmas"]
        tensors["timesteps"] = den["timesteps"]
        tensors["timestep_in"] = torch.tensor(den["timestep_in"], dtype=torch.float32)
        for i, (npred, lat) in enumerate(zip(den["noise_pred"], den["latents"])):
            tensors[f"noise_pred_{i:03d}"] = npred
            tensors[f"latents_{i:03d}"] = lat
        tensors["latents_final"] = den["final"]
        for k, v in den["step0"].items():
            tensors[f"step0_{k}"] = v
        for k in ("vae_in", "vae_out_raw", "vae_out"):
            tensors[k] = dec[k]
        tensors["rgba"] = dec["rgba"]
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(out_dir / "ref.safetensors"))
    n_text = len(enc["input_ids"]) - enc["drop_idx"]
    n = meta["latents_init"].shape[0]
    ref = {
        "model": meta["model"],
        "prompt": enc["prompt"],
        "template_text": enc["template_text"],
        "system_prompt": enc["system_prompt"],
        "input_ids": enc["input_ids"],
        "drop_idx": enc["drop_idx"],
        "prompt_embeds_is_prenorm_last_layer": enc["prompt_embeds_is_prenorm_last_layer"],
        "width": meta["width"], "height": meta["height"],
        "latent_h": 2 * (meta["height"] // 32), "latent_w": 2 * (meta["width"] // 32),
        "n_text_tokens": n_text, "n_image_tokens": n, "joint_seq_len": n_text + n,
        "steps": meta["steps"], "seed": meta["seed"], "latents_seed_note": meta.get("latents_note"),
        "mu": den["mu"],
        "sigmas": [float(v) for v in den["sigmas"]],
        "timesteps": [float(v) for v in den["timesteps"]],
        "timestep_in": den["timestep_in"],
        "compute_dtype": meta["compute_dtype"],
        "processor": dict(PROCESSOR_NOTE),
        "constants": meta["constants"],
        "step0_notes": {
            "step0_txt_in": "txt_in output [T_text][inner]",
            "step0_temb": "time_text_embed output [2][inner]: row 0 sampled t, row 1 t = 0",
            "step0_modulation": "modulation output [2][4 * inner]",
            "step0_rope_cos/sin": "RoPE table for the joint sequence [T_text + N][head_dim / 2], axes concatenated",
            "step0_block0_in/out": "block 0 input/output over the whole joint sequence [T_text + N][inner]; "
                                   "the target tokens are the last N rows",
        },
        "timings_s": meta["timings"],
        "peak_rss_gb": meta["peak_rss"],
        "machine": machine(),
        "versions": versions(),
        "tensors": tensor_table(tensors) if dump else None,
    }
    (out_dir / "ref.json").write_text(json.dumps(ref, indent=1))


def summarize_redundancy(stats: dict, n_layers: int) -> dict:
    blocks = []
    for k in range(n_layers):
        entries = stats["blocks"].get(k, [])
        blocks.append({"block": k, "steps": entries})
    return {"blocks": blocks, "noise_pred": stats["noise_pred"], "latents": stats["latents"],
            "definitions": {
                "out": "block output, target image tokens only",
                "delta": "block output minus block input (what a block-skip cache would reuse)",
                "cos": "cosine similarity of step t vs step t-1 over the whole [N][D] matrix",
                "rel": "per-token ||h_t - h_{t-1}|| / ||h_{t-1}||",
                "frac_below_Xpct": "fraction of target tokens whose rel is under X percent",
            }}


# ---------------------------------------------------------------- main

def load_tensor(path: Path, key: str) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(str(path))[key].float()


def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", default="a red fox sitting in the snow, photograph")
    ap.add_argument("--size", default="512x512", help="WIDTHxHEIGHT")
    ap.add_argument("--steps", default="8", help="one count, or a comma list (one run each, subdirs steps_N)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32", help="transformer compute dtype")
    ap.add_argument("--te-dtype", choices=("f32", "bf16"), default="f32", help="text encoder compute dtype")
    ap.add_argument("--embeds", type=Path, help="reuse prompt_embeds (and ids) from a previous ref.safetensors")
    ap.add_argument("--latents", type=Path, help="reuse latents_init from a previous ref.safetensors")
    ap.add_argument("--dump", action="store_true", help="write ref.safetensors with every intermediate")
    ap.add_argument("--redundancy", action="store_true", help="measure step-to-step change of every block")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-rss-gb", type=float, default=35.0)
    ap.add_argument("--wait-mem-minutes", type=float, default=30.0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    guard = MemoryGuard(args.max_rss_gb, args.wait_mem_minutes)
    width, height = parse_size(args.size)
    if width % 32 or height % 32:
        raise SystemExit("width and height must be multiples of 32")
    steps_list = [int(s) for s in args.steps.split(",")]
    dt = {"f32": torch.float32, "bf16": torch.bfloat16}
    model_dir = args.model
    timings = {}
    peak = {}

    # 1. text encoder
    if args.embeds:
        from safetensors.torch import load_file

        prev = load_file(str(args.embeds))
        prev_json = json.loads((args.embeds.parent / "ref.json").read_text())
        enc = {"prompt": prev_json["prompt"], "template_text": prev_json["template_text"],
               "system_prompt": prev_json["system_prompt"], "input_ids": prev_json["input_ids"],
               "drop_idx": prev_json["drop_idx"], "prompt_embeds": prev["prompt_embeds"].float(),
               "prompt_embeds_is_prenorm_last_layer": prev_json["prompt_embeds_is_prenorm_last_layer"]}
        te_cfg = prev_json["constants"]["text_encoder"]
        if args.prompt != enc["prompt"]:
            log(f"--embeds overrides --prompt: using {enc['prompt']!r}")
        timings["encode"] = {"reused_from": str(args.embeds)}
    else:
        te_need = dir_size_gb(model_dir / "text_encoder") * (2.0 if args.te_dtype == "f32" else 1.0)
        te_dtype = dt[args.te_dtype]
        if not guard.wait_for(te_need, "text encoder"):
            if te_dtype == torch.float32:
                log("falling back to bf16 compute for the text encoder")
                te_dtype = torch.bfloat16
                if not guard.wait_for(te_need / 2, "text encoder bf16"):
                    raise SystemExit("not enough memory for the text encoder")
            else:
                raise SystemExit("not enough memory for the text encoder")
        encs, te_info = encode_prompts(model_dir, [args.prompt], te_dtype, args.dump, guard)
        enc = encs[0]
        te_cfg = te_info["config"]
        timings["encode"] = {"load_s": te_info["load_s"], "compute_s": enc["encode_s"],
                             "loadavg_1min": enc["encode_loadavg"], "dtype": te_info["dtype"]}
        peak["after_encode"] = te_info["peak_rss_gb"]

    # 2. transformer
    from diffusers import QwenImage21Transformer2DModel

    tr_cfg = dict(QwenImage21Transformer2DModel.load_config(str(model_dir / "transformer")))
    channels = tr_cfg["in_channels"]
    if args.latents:
        latents_init = load_tensor(args.latents, "latents_init")
        latents_note = f"reused from {args.latents}"
    else:
        latents_init = pack_initial_latents(args.seed, channels, width, height)
        latents_note = (f"torch.randn((1, 1, {channels}, h, w), generator=torch.Generator('cpu').manual_seed("
                        f"{args.seed}), float32), then QwenImage21Pipeline._pack_latents")
    n = (height // VAE_SCALE) * (width // VAE_SCALE)
    if latents_init.shape != (n, channels):
        raise SystemExit(f"latents_init has shape {tuple(latents_init.shape)}, expected {(n, channels)}")
    tr_need = dir_size_gb(model_dir / "transformer") * (2.0 if args.dtype == "f32" else 1.0) + 1.5
    if not guard.wait_for(tr_need, "transformer"):
        raise SystemExit("not enough memory for the transformer")
    runs = [{"prompt_embeds": enc["prompt_embeds"], "latents_init": latents_init, "width": width, "height": height,
             "steps": s} for s in steps_list]
    dens, tr_info = denoise(model_dir, runs, dt[args.dtype], args.dump, args.redundancy, guard)
    peak["after_denoise"] = tr_info["peak_rss_gb"]

    # 3. VAE
    if not guard.wait_for(dir_size_gb(model_dir / "vae") + 3.0, "VAE"):
        raise SystemExit("not enough memory for the VAE")
    decs, vae_info = decode(model_dir, [{"final": d["final"], "width": width, "height": height} for d in dens],
                            args.dump)
    peak["after_decode"] = vae_info["peak_rss_gb"]

    sched_cfg = dict(load_scheduler(model_dir).config)
    constants = model_constants(te_cfg, tr_info["config"], vae_info["config"], sched_cfg)
    for steps, den, dec in zip(steps_list, dens, decs):
        out_dir = args.out if len(steps_list) == 1 else args.out / f"steps_{steps}"
        t = {"encode": timings["encode"],
             "denoise": {"load_s": tr_info["load_s"], "total_s": den["denoise_s"], "per_step_s": den["step_s"],
                         "loadavg_1min_after_step": den["step_load"], "dtype": tr_info["dtype"]},
             "decode": {"load_s": vae_info["load_s"], "compute_s": dec["decode_s"], "loadavg_1min": dec["loadavg"],
                        "dtype": vae_info["dtype"]},
             "threads": args.threads}
        meta = {"model": str(model_dir), "width": width, "height": height, "steps": steps, "seed": args.seed,
                "latents_init": latents_init, "latents_note": latents_note,
                "compute_dtype": {"text_encoder": timings["encode"].get("dtype", "reused"),
                                  "transformer": tr_info["dtype"], "vae": "float32"},
                "constants": constants, "timings": t, "peak_rss": peak}
        write_run(out_dir, enc, den, dec, meta, args.dump)
        if den["redundancy"] is not None:
            red = summarize_redundancy(den["redundancy"], tr_info["config"]["num_layers"])
            red.update({"width": width, "height": height, "steps": steps, "sigmas": [float(v) for v in den["sigmas"]],
                        "compute_dtype": tr_info["dtype"]})
            (out_dir / "redundancy.json").write_text(json.dumps(red, indent=1))
        log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
