#!/bin/zsh
set -e
MODEL=${MODEL:-$HOME/models/qwen36_i4_gs64}
RUNS=${RUNS:-$HOME/a1-m1-runs}
N=${N:-3}
CAP=${CAP:-2}
THREADS=${THREADS:-4}
mkdir -p "$RUNS"
printf 'The three primary colors are' > "$RUNS/prompt.txt"

run_block() {
  local label=$1 bin=$2
  for i in $(seq 1 "$N"); do
    echo "=== $label-$i ==="
    env -u OMP_NUM_THREADS -u OMP_DYNAMIC SNAP="$MODEL" N_NEW=16 COLI_TIMERS=1 \
      /usr/bin/time -l "$bin" "$CAP" "$THREADS" "$RUNS/prompt.txt" \
      > "$RUNS/${label}-${i}.out" 2> "$RUNS/${label}-${i}.err"
  done
}

run_block base1 "$RUNS/qwen36-base"
run_block sdot1 "$RUNS/qwen36-sdot"
run_block sdot2 "$RUNS/qwen36-sdot"
run_block base2 "$RUNS/qwen36-base"
echo "DONE"
