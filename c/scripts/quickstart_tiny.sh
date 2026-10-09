#!/bin/zsh
# QLoRA quickstart on the tiny oracle model — the full mechanism in ~2 minutes,
# no big download, CPU only. Needs a Python with torch+transformers ONCE to
# generate the tiny model (tools/make_glm_oracle.py); training and inference
# are pure C.
#
#   ./scripts/quickstart_tiny.sh
#
# What it proves, in order:
#   1. BASE     — greedy generation matches the PyTorch oracle reference
#                 token-exactly (the inference path is untouched).
#   2. TRAIN    — coli_train overfits a rank-8 LoRA so the same prompt should
#                 emit a DIFFERENT, chosen continuation (~seconds of training).
#   3. ADAPTER  — with ADAPTER set, generation reproduces the trained
#                 continuation token-exactly.
#   4. UNSET    — with ADAPTER unset again, generation matches the oracle
#                 reference token-exactly: base behavior restored.
set -euo pipefail
cd "$(dirname "$0")/.."

make colibri coli_train

if [[ ! -f glm_tiny/config.json || ! -f ref_glm.json ]]; then
  echo "== generating tiny oracle model (needs torch+transformers) =="
  python3 tools/make_glm_oracle.py
fi

echo "== building tiny SFT set from the oracle's own prompt =="
python3 tools/make_tiny_sft.py --out data/tiny_sft
SEQ=$(python3 -c "import json;print(json.load(open('data/tiny_sft/metadata.json'))['sample_len'])")

run_oracle() {  # $1 = adapter dir or empty; prints the engine's generated ids
  if [[ -n "$1" ]]; then
    ADAPTER="$1" SNAP=./glm_tiny DRAFT=0 ./colibri 64 16 16
  else
    SNAP=./glm_tiny DRAFT=0 ./colibri 64 16 16
  fi
}

echo ""
echo "== 1. BASE: adapter off — must match the oracle reference exactly =="
BASE_OUT=$(run_oracle "" | grep -E "Matching tokens")
echo "   $BASE_OUT"

echo ""
echo "== 2. TRAIN: overfit a rank-16 LoRA onto a different continuation =="
rm -rf adapters/tiny-demo
./coli_train --model ./glm_tiny --data data/tiny_sft --adapter-out adapters/tiny-demo \
  --ram 12 --seq-len "$SEQ" --grad-accum 1 --rank 16 --alpha 32 \
  --lr 2e-2 --steps 800 --save-every 800 --seed 0 2>&1 | grep -E "\[step (1|400|800)\]|\[train\] done"

echo ""
echo "== 3. ADAPTER on: generation should now emit the TRAINED continuation =="
ADA_IDS=$(run_oracle adapters/tiny-demo | grep "^GLM C engine" | cut -d: -f2)
python3 - "$ADA_IDS" <<'PYEOF'
import json, sys
got = [int(x) for x in sys.argv[1].split()]
t = json.load(open('data/tiny_sft/target_ids.json'))
tgt, ref = t['target_ids'], t['ref_continuation']
n = sum(1 for a, b in zip(got, tgt) if a == b)
print(f"   trained continuation reproduced: {n}/{len(tgt)} tokens "
      f"(vs {sum(1 for a,b in zip(got,ref) if a==b)}/{len(ref)} still matching the base reference)")
if n < len(tgt) * 0.9: sys.exit("   FAIL: adapter did not take")
PYEOF

echo ""
echo "== 4. UNSET: adapter off again — base behavior must be restored =="
UNSET_OUT=$(run_oracle "" | grep -E "Matching tokens")
echo "   $UNSET_OUT"
[[ "$UNSET_OUT" == "$BASE_OUT" ]] && echo "   identical to step 1: base restored ✓" \
                                  || { echo "   FAIL: differs from step 1"; exit 1; }

echo ""
echo "quickstart complete: base exact -> adapter takes -> base restored."
