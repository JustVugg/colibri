#!/usr/bin/env bash
set -euo pipefail

# Plug-and-play Qwen3-ASR benchmark runner (Colibri vs llama.cpp).
# Auto-detects CPU threads, allows manual override, writes JSON report.

usage() {
  cat <<'USAGE'
Usage: run_qwen3_asr_benchmark.sh [options] [-- extra benchmark_asr.py args]

Options:
  --threads N         OMP threads (default: auto, ~physical cores or nproc-2/4)
  --repeats N         Benchmark repeats (default: 3)
  --out PATH          JSON report output (default: c/tools/benchmark_qwen3_asr_hello3.json)
  --workdir PATH      WAV workdir (default: /tmp/opencode/bench_wavs)
  --colibri PATH      Colibri engine binary (default: ./c/qwen3_asr)
  --model PATH        HF snapshot/model dir (default: detected from HF cache)
  --llama-cli PATH    llama-cli binary (default: detected common location)
  --gguf PATH         GGUF text model for llama/colibri-gguf
  --mmproj PATH       mmproj for llama/colibri-gguf
  --no-llama          Skip llama.cpp comparison
  --json              Print JSON to stdout (also writes --out)
  -h, --help          Show this help

Environment:
  OMP_NUM_THREADS     Used if --threads not given

Examples:
  c/tools/run_qwen3_asr_benchmark.sh --threads 16 --repeats 3
  OMP_NUM_THREADS=8 c/tools/run_qwen3_asr_benchmark.sh --out /tmp/report.json
USAGE
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

detect_threads() {
  local n
  n="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
  # sensible default: leave headroom
  if [ "$n" -le 4 ]; then echo "$n"; elif [ "$n" -le 8 ]; then echo $((n-2)); else echo $((n-4)); fi
}

THREADS="${OMP_NUM_THREADS:-}"
REPEATS=3
OUT="c/tools/benchmark_qwen3_asr_hello3.json"
WORKDIR="/tmp/opencode/bench_wavs"
COLIBRI="./c/qwen3_asr"
MODEL=""
LLAMA_CLI="/media/dorin/more_data/work/xllama-agent/vendor/llama.cpp/build/bin/llama-cli"
GGUF=""
MMPROJ=""
NO_LLAMA=0
PRINT_JSON=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --workdir) WORKDIR="$2"; shift 2;;
    --colibri) COLIBRI="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --llama-cli) LLAMA_CLI="$2"; shift 2;;
    --gguf) GGUF="$2"; shift 2;;
    --mmproj) MMPROJ="$2"; shift 2;;
    --no-llama) NO_LLAMA=1; shift;;
    --json) PRINT_JSON=1; shift;;
    -h|--help) usage; exit 0;;
    --) shift; EXTRA+=("$@"); break;;
    *) EXTRA+=("$1"); shift;;
  esac
done

if [ -z "$THREADS" ]; then THREADS="$(detect_threads)"; fi
if [ -z "$MODEL" ]; then
  MODEL="$(ls -1d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/* 2>/dev/null | tail -1 || true)"
fi
if [ -z "$GGUF" ]; then
  GGUF="$(ls -1 "$HOME"/.cache/huggingface/hub/models--ggml-org--Qwen3-ASR-1.7B-GGUF/snapshots/*/Qwen3-ASR-1.7B-Q8_0.gguf 2>/dev/null | tail -1 || true)"
fi
if [ -z "$MMPROJ" ]; then
  MMPROJ="$(ls -1 "$HOME"/.cache/huggingface/hub/models--ggml-org--Qwen3-ASR-1.7B-GGUF/snapshots/*/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf 2>/dev/null | tail -1 || true)"
fi

if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
  echo "error: model not found; pass --model <hf-snapshot>" >&2; exit 1
fi
if [ ! -x "$COLIBRI" ]; then
  echo "error: colibri binary not found/executable: $COLIBRI" >&2; exit 1
fi

CMD=(python3 c/tools/benchmark_asr.py --colibri "$COLIBRI" --model "$MODEL" --workdir "$WORKDIR" --repeats "$REPEATS" --omp-threads "$THREADS" --out "$OUT")
if [ "$NO_LLAMA" -eq 0 ] && [ -x "$LLAMA_CLI" ] && [ -n "$GGUF" ] && [ -n "$MMPROJ" ]; then
  CMD+=(--llama-cli "$LLAMA_CLI" --gguf "$GGUF" --mmproj "$MMPROJ")
else
  echo "note: llama.cpp comparison skipped (missing llama-cli/gguf/mmproj or --no-llama)" >&2
fi
if [ "$PRINT_JSON" -eq 1 ]; then CMD+=(--json); fi
if [ ${#EXTRA[@]} -gt 0 ]; then CMD+=("${EXTRA[@]}"); fi

echo "== Qwen3-ASR benchmark =="
echo "threads=$THREADS repeats=$REPEATS out=$OUT"
echo "model=$MODEL"
echo "colibri=$COLIBRI"
if [ "$NO_LLAMA" -eq 0 ] && [ -x "$LLAMA_CLI" ] && [ -n "$GGUF" ] && [ -n "$MMPROJ" ]; then
  echo "llama=$LLAMA_CLI"
fi
echo

OMP_NUM_THREADS="$THREADS" "${CMD[@]}"
