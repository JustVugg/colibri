#!/bin/bash
# Unified user-experience benchmark: fixed scenarios, fixed decoding, medians.
#
# Reports, per scenario: TTFT (prefill wall), decode tok/s, and the first line
# of the generated text (drift check). Runs each scenario REPS times and prints
# every sample; judge by the median, not the best.
#
# Usage:  SNAP=/path/to/model [REPS=3] [PIPE=2] tools/bench_ux.sh [engine-binary]
# The engine env (COLI_CUDA, PIN, ...) is inherited from the caller so the same
# script exercises CPU-only, CUDA and pipeline configurations.
#
# Discipline (docs/experiments/glm52-6x5090-2026-07-12.md):
#   - TEMP=0 DRAFT=0 always: greedy, no speculation, one variable at a time.
#   - Same binary for every configuration under comparison.
#   - .coli_usage drifts placement between runs: compare medians of >=3 reps,
#     or snapshot/restore the usage file around the battery.
set -u
GLM="${1:-./glm}"
REPS="${REPS:-3}"
export TEMP=0 DRAFT=0
[ -n "${PIPE:-}" ] && export COLI_CUDA_PIPE="$PIPE"

SHORT_PROMPT="Explain why the sky is blue in simple terms."
LONGDOC_PROMPT=$(python3 - <<'PY'
base=("蜂鸟是世界上最小的鸟类之一，主要分布在美洲大陆。它们的翅膀每秒可以扇动五十到八十次，"
"使它们能够在空中悬停、倒飞，甚至垂直起降。蜂鸟的新陈代谢极快，心跳每分钟可达一千两百次，"
"因此它们必须不断进食来维持能量。它们的主要食物是花蜜，同时也会捕食小型昆虫和蜘蛛来补充蛋白质。"
"蜂鸟的喙细长，适合深入花朵内部吸食花蜜，舌头呈管状，可以快速伸缩。在授粉过程中，"
"蜂鸟扮演着重要的角色，许多美洲植物依赖蜂鸟传粉。")
print((base*12)+"请根据上文回答：蜂鸟的主要食物是什么？")
PY
)

scenario(){ # name prompt ngen
  local name=$1 prompt=$2 ngen=$3
  for r in $(seq 1 "$REPS"); do
    local log; log=$(mktemp)
    NGEN=$ngen PROMPT="$prompt" "$GLM" 64 4 4 >"$log" 2>&1
    local line; line=$(grep -aE "prefill .* decode .*tok/s" "$log" | tail -1)
    local head; head=$(grep -av "^\[" "$log" | grep -avE "PROFILE|PROFILO|ATTENTION|expert|CUDA|prefill|specul|^---|TOPP|stop|Motore|caricato|prompt:|token" \
                       | tail -1 | tr -d '\n' | cut -c1-48)
    printf "%-10s r%d  %s\n" "$name" "$r" "$line"
    printf "%-10s r%d  text: %s\n" "$name" "$r" "$head"
    rm -f "$log"
  done
}

echo "== bench_ux: reps=$REPS pipe=${COLI_CUDA_PIPE:-unset} $(date +%F' '%T) =="
scenario chat     "$SHORT_PROMPT"   96
scenario longdoc  "$LONGDOC_PROMPT" 96
echo "== bench_ux done $(date +%T) =="
