# Qwen3-ASR-1.7B full port plan

Status: in progress (scaffolding started 2026-11-22)

## Goal

Fully port `Qwen/Qwen3-ASR-1.7B` into Colibri as a pure-C engine, expose a stable transcription API usable by other projects, and provide reproducible benchmarks against llama.cpp.

Target model: https://huggingface.co/Qwen/Qwen3-ASR-1.7B (`qwen3_asr`, Apache-2.0, BF16 safetensors).

## Verified model shape

From HF `config.json`, `preprocessor_config.json`, and transformers `modeling_qwen3_asr.py`:

- Top-level: `Qwen3ASRForConditionalGeneration`
- Audio frontend: Whisper-style log-mel, 16 kHz, `n_fft=400`, `hop_length=160`, `feature_size=128`, 30s chunks, attention mask
- Audio encoder: `qwen3_asr_audio_encoder`, 24 layers, `d_model=1024`, 16 heads, `encoder_ffn_dim=4096`, 3x stride-2 Conv2d, sinusoidal pos emb, windowed attention (`n_window=50`, `n_window_infer=800`), `output_dim=2048`
- Projector: Linear(1024->1024), GELU, Linear(1024->2048)
- Text decoder: dense Qwen3, 28 layers, hidden 2048, 16 attn / 8 KV heads, head_dim 128, intermediate 6144, SwiGLU, RMSNorm, RoPE theta 1e6, mRoPE interleaved `[24,20,20]`, vocab 151936
- Fusion: audio embeddings replace `audio_token_id=151676` in input embeddings; autoregressive generation

## Compatibility targets

Primary external API (for other projects, including voicy-offline-llama-style local clients):

- `POST /v1/audio/transcriptions` - OpenAI-style multipart/form-data and/or JSON audio input
- `GET /v1/models` - model metadata

Secondary:

- Chat/completions with `audio_url` / `input_audio` for multimodal chat clients
- Streaming transcription (phase after offline correctness)

## Phases

### Phase 0 - Spec and fixtures
- [ ] Freeze MVP contract: offline 16kHz WAV transcription, greedy decode, BF16 safetensors, no timestamps/streaming/ForcedAligner
- [ ] Add reference fixture generator (`c/tools/make_qwen3_asr_tiny.py` or reference extraction script)
- [ ] Capture transformers oracle outputs for short audio clips
- [ ] Define benchmark manifest format (audio path, duration, reference text, language)

### Phase 1 - Audio frontend
- [ ] C WAV/PCM reader and mono 16kHz conversion/resampling policy
- [ ] Whisper-compatible log-mel extractor in C with parity tests
- [ ] Chunking/attention-mask handling matching preprocessor

### Phase 2 - Audio encoder + projector
- [ ] Conv2d frontend, pos emb, windowed attention, FFN, ln_post
- [ ] Multimodal projector
- [ ] Encoder output parity against transformers on tiny fixture

### Phase 3 - Text decoder integration
- [ ] Dense Qwen3 backbone reuse/adaptation in `qwen3_asr` engine
- [ ] Audio token placeholder replacement
- [ ] Greedy generation, stop/eos handling, prompt/template formatting
- [ ] End-to-end transcription parity on tiny fixture

### Phase 4 - Gateway and CLI integration
- [ ] `c/family_registry.py` descriptor/capabilities
- [ ] `c/setup_catalog.py` model entry
- [ ] `c/Makefile` target
- [ ] `coli` launcher/family wiring
- [ ] `c/openai_server.py` `/v1/audio/transcriptions` endpoint
- [ ] Model/capability reporting

### Phase 5 - Correctness, quantization, performance
- [ ] Long audio chunking
- [ ] BF16/f16/int quant paths as appropriate
- [ ] CPU/GPU backend decisions
- [ ] Memory/load-time optimizations

### Phase 6 - Benchmarks and docs
- [ ] `c/tools/benchmark_asr.py` harness
- [ ] Colibri vs llama.cpp protocol and report
- [ ] User docs: setup, API, benchmark reproduction

## Benchmark protocol (Colibri vs llama.cpp)

Report per run:

- Commit/version: Colibri commit, llama.cpp commit, model revision/quant
- Hardware: CPU/GPU, RAM, OS, threads, BLAS/GPU flags
- Audio manifest: file, sample rate, duration, language, reference text
- Quality: WER, CER, language ID accuracy (if enabled)
- Speed: wall time, RTF = audio_duration / wall_time, load time, first-result latency
- Resources: peak RSS, GPU memory if applicable

Rules:

- Same audio inputs and reference transcripts
- Same decoding settings where possible (greedy, temperature 0)
- Warm and cold runs reported separately
- No hidden preprocessing differences; document resampling/chunking

## Status (MVP)

- [x] `c/qwen3_asr.c` engine (BF16 safetensors + optional GGUF/mmproj), serve protocol, `/v1/audio/transcriptions`.
- [x] Decoder parity: neox/half-split RoPE, English forced prefill, mel-floor padding, windowed audio encoder attention; rope_theta resolved to HF-effective value.
- [x] Oracle smoke: `make -C c test-qwen3-asr` (skippable without model) — hello/weather/numbers/combined exact-or-normalized HF refs.
- [x] Definitive benchmark (3-repeat, OMP=16, same machine): Colibri BF16 4.23/3.62/5.06s med, Colibri Q8_0 GGUF 3.06/2.60/3.35s, llama.cpp Q8_0 11.20/11.05/11.79s; WER 0.
- [x] Plug-and-play benchmark: `c/tools/run_qwen3_asr_benchmark.sh`.

## Perf posture

- **Recommended fast path:** Q8_0 GGUF (`--gguf` + `--mmproj`) — lowest latency/RSS in definitive benchmark.
- **Supported/reference:** BF16 safetensors — correct, slower decode/prefill (bandwidth-bound).
- GPU encoder handoff deferred; no timestamps/streaming/ForcedAligner in MVP.

## Notes

- Inkling DMel audio input in Colibri is not reusable for Whisper-mel ASR frontend.
- Qwen3 dense support exists in qwen36-related paths; reuse only after exact config/mRoPE/template parity checks.
- ForcedAligner (`Qwen/Qwen3-ForcedAligner-0.6B`) is out of MVP unless explicitly pulled in.
- External project compatibility should prefer `/v1/audio/transcriptions`; confirm voicy-offline-llama expected endpoint before finalizing (deferred).