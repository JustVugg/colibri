#!/bin/zsh
# IQ2_XXS Metal dispatch validation runner.
MODEL=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf
REPO=/Users/creemac/colibri/colibri
ENGINE=$REPO/c/qwen36
LOGS=$REPO/logs/iq2_validation

cd $REPO
tag=$1
shift
out=$LOGS/run_$tag.log

echo "== run $tag : $(date) | env IQ2=${COLI_METAL_IQ2_XXS:-unset} =="
SNAP=$MODEL N_NEW=${N_NEW:-32} DUMP=$LOGS/logits_$tag.f32 \
  ./c/qwen36 0 4 "$LOGS/prompt.txt" 2>"$out" 1>"$LOGS/run_$tag.stdout.txt"
rc=$?
echo "exit=$rc"
grep -E "resident weights|PEAK RSS|Speed:|TTFT|Generated|Text|IQ2|metal|\[dump\]|Error|failed" "$out" | head -20
