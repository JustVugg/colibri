#!/bin/zsh
# Milestone 8: deliberate overfit on the Kolibrík persona set + on/off eval
# (docs/plans/2026-07-19-m7-real-model-runbook.md §3c). Offline once the
# snapshot is local. Run AFTER m7_smoke.sh passes.
#
#   COLI_MODEL=~/Work/models/glm52_i4 ./scripts/m8_overfit.sh
#   STEPS=600 LR=2e-3 ./scripts/m8_overfit.sh        # knobs
#
# Success evidence (§16-M8): train loss drops materially; generation on
# training prompts moves toward the persona (terse + 🐦); the held-out prompt
# shows generalized style; unsetting ADAPTER restores base behavior.
set -euo pipefail
cd "$(dirname "$0")/.."

T_START=$(date +%s); T_LAST=$(date +%s)
stamp() { local now=$(date +%s)
  echo "⏱  [$1] took $((now-T_LAST))s (elapsed total $((now-T_START))s)"; T_LAST=$now; }

MODEL="${COLI_MODEL:-$HOME/Work/models/glm52_i4}"
OUT="${ADAPTER_OUT:-adapters/m8-persona}"
RAM="${RAM:-52}"
STEPS="${STEPS:-300}"
LR="${LR:-1e-3}"

[[ -f "$MODEL/config.json" ]] || { echo "!! no model at $MODEL (set COLI_MODEL)"; exit 1; }
[[ -x coli_train ]] || make coli_train

RESUME_ARGS=()
if [[ -f "$OUT/train_state.bin" ]]; then RESUME_ARGS=(--resume "$OUT"); echo "(resuming $OUT)"
else echo "(fresh run — no $OUT state)"; fi

echo "== training: $STEPS steps on data/m8_tokenized (rank 8, lr $LR) =="
./coli_train --model "$MODEL" --data data/m8_tokenized --adapter-out "$OUT" \
  "${RESUME_ARGS[@]}" \
  --ram "$RAM" --seq-len 128 --grad-accum 4 --rank 8 --alpha 16 \
  --lr "$LR" --steps "$STEPS" --save-every 50 --seed 0

stamp "training ($STEPS steps)"

echo ""
echo "== eval: base vs adapter (greedy, 48 tokens) =="
for f in data/m8_eval/*.txt; do
  name="$(basename "$f" .txt)"
  P="$(cat "$f")"
  echo "----------------------------------------------------------"
  # generation text is everything BEFORE the trailing "---" stats block
  # (tail -3 used to grab the profile summary instead of the answer)
  echo ">> [$name] BASE (no adapter):"
  SNAP="$MODEL" PROMPT="$P" NGEN=48 TEMP=0 ./colibri 64 2>/dev/null | sed -n '1,/^---$/p' | grep -v '^\[t='
  echo ">> [$name] ADAPTER:"
  ADAPTER="$OUT" SNAP="$MODEL" PROMPT="$P" NGEN=48 TEMP=0 ./colibri 64 2>/dev/null | sed -n '1,/^---$/p' | grep -v '^\[t='
done
stamp "eval (all prompts, base+adapter)"
echo "----------------------------------------------------------"
echo "Expect: adapter answers terse, persona-shaped, ending with the bird."
echo "Base answers must be unchanged (adapter off = base behavior, §16-M8)."
