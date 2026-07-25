#!/bin/zsh
# Milestone 7 staged smoke test (docs/plans/2026-07-19-m7-real-model-runbook.md §3b).
# Fully offline once the int4 snapshot is on disk.
#
#   COLI_MODEL=~/Work/models/glm52_i4 ./scripts/m7_smoke.sh
#
# Stages: build -> doctor -> 1 full training step -> 10 steps -> adapter loads
# in inference. Abort criteria are enforced by coli_train itself (budget
# manager: footprint ceiling + swap growth). Watch the [mem] lines.
set -euo pipefail
cd "$(dirname "$0")/.."

# stage timing: printed after every stage and summed at the end
T_START=$(date +%s); T_LAST=$(date +%s)
stamp() { local now=$(date +%s)
  echo "⏱  [$1] took $((now-T_LAST))s (elapsed total $((now-T_START))s)"; T_LAST=$now; }

MODEL="${COLI_MODEL:-$HOME/Work/models/glm52_i4}"
OUT="${ADAPTER_OUT:-adapters/m7-smoke}"
RAM="${RAM:-36}"   # conservative default: leaves room for your apps + macOS file cache; raise to 52 on a dedicated run

[[ -f "$MODEL/config.json" ]] || { echo "!! no model at $MODEL (set COLI_MODEL)"; exit 1; }

echo "== stage 0: build =="
make colibri METAL=1
make coli_train

stamp "stage 0: build"

echo "== stage 0b: read-only readiness =="
COLI_MODEL="$MODEL" ./coli doctor || echo "(doctor warnings above — review before continuing)"

stamp "stage 0b: doctor"

echo "== stage 1: ONE full training step (fwd + bwd + optimizer) =="
./coli_train --model "$MODEL" --data data/m8_tokenized --adapter-out "$OUT" \
  --ram "$RAM" --seq-len 128 --grad-accum 1 --rank 4 --alpha 8 \
  --lr 1e-4 --steps 1 --save-every 1 --seed 0

stamp "stage 1: single step"

echo "== stage 2: 10 steps with gradient accumulation (resumes stage 1) =="
RESUME_ARGS=()
[[ -f "$OUT/train_state.bin" ]] && RESUME_ARGS=(--resume "$OUT") || echo "(no stage-1 state — starting fresh)"
./coli_train --model "$MODEL" --data data/m8_tokenized --adapter-out "$OUT" \
  "${RESUME_ARGS[@]}" \
  --ram "$RAM" --seq-len 128 --grad-accum 4 --rank 4 --alpha 8 \
  --lr 1e-4 --steps 10 --save-every 5 --seed 0

stamp "stage 2: 10 steps"

echo "== stage 3: adapter loads in inference (M7 success criterion) =="
PROMPT="$(cat data/m8_eval/who.txt)"
echo "--- with adapter ---"
ADAPTER="$OUT" SNAP="$MODEL" PROMPT="$PROMPT" NGEN=48 TEMP=0 ./colibri 64
stamp "stage 3: inference eval"
echo ""
echo "== M7 smoke complete. Checklist (AGENTS.md §16/§27):"
echo "   - every [step] line finite loss?          (scroll up)"
echo "   - [mem] footprint stayed under $RAM GB, swap-delta ~0?"
echo "   - expert loads/hits and MB/step recorded?"
echo "   - checkpoint written to $OUT and loaded by inference above?"
