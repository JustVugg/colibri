#!/usr/bin/env bash
# Interleaved runs of kimi_k3 for the COLI_MODEL_MIRROR experiment (#812).
#
#   MODEL=/path/to/model bench.sh OUT_DIR ARM [ARM ...]
#   ARM = NAME@BINARY[@VAR=VALUE]...
#
# Example (the three arms of the manifests in this directory):
#   MODEL=/nvme0/kimi bench.sh out \
#     base-off@./base/c/kimi_k3 \
#     pr-off@./pr/c/kimi_k3 \
#     'pr-mirror@./pr/c/kimi_k3@COLI_MODEL_MIRROR=/nvme2/kimi;/nvme3/kimi'
#
# RUNS rounds. Each round starts one arm later than the round before, so no arm
# always follows the same one. Every run gets COMMON below (COLI_DISK_WEIGHTS
# included), then the arm's own variables. Before each run the page cache is
# dropped (needs root). For each run, OUT_DIR gets the engine's stdout (the
# tokens as text), its stderr, and the bytes each drive read (/proc/diskstats).
# summarize.py turns OUT_DIR into the manifests.
set -euo pipefail
umask 022

OUT=${1:?usage: MODEL=... bench.sh OUT_DIR ARM [ARM ...]}
shift
(($#)) || { echo "no arm given" >&2; exit 2; }
: "${MODEL:?}"
RUNS=${RUNS:-3}
NGEN=${NGEN:-64}
DRIVES=${DRIVES:-nvme0n1 nvme1n1 nvme2n1 nvme3n1}
PROMPT=${PROMPT:-"Write a connected adventure story in 40 chapters. The main character is a lighthouse keeper, and the story takes place on a remote island in the North Sea. Each chapter starts with its own line 'Chapter <number>: <title>' and has about 150 words. Write all 40 chapters in full, with no summary at the end."}
# shellcheck disable=SC2054  # the comma belongs to the value of COLI_DISK_WEIGHTS
COMMON=(COLI_TEMP=0 OMP_NUM_THREADS=24 OMP_WAIT_POLICY=active GOMP_SPINCOUNT=200000 OMP_PROC_BIND=close
        OMP_PLACES=cores K3_EXPERT_GB=24 K3_CHUNK=512 K3_LOAD_THREADS=16 USAGE_SAVE=0 COLI_DISK_WEIGHTS=1,1,1)
ARMS=("$@")

mkdir -p "$OUT"
printf '%s' "$PROMPT" >"$OUT/prompt.txt"
printf '%s\n' "${COMMON[@]}" >"$OUT/common.env"
printf '%s\n' "${ARMS[@]}" >"$OUT/arms.txt"

sectors() {   # sectors read by each drive in DRIVES, in that order
  awk -v want="$DRIVES" 'BEGIN {n = split(want, d, " ")} {s[$3] = $6}
    END {for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? " " : ""), s[d[i]]; print ""}' /proc/diskstats
}

run() {   # run ARM ROUND
  local name bin before after i
  local -a parts vars bytes=()
  IFS=@ read -r -a parts <<<"$1"
  name=${parts[0]}-$2
  bin=${parts[1]}
  vars=("${parts[@]:2}")
  sync
  echo 3 >/proc/sys/vm/drop_caches
  before=$(sectors)
  env "${COMMON[@]}" ${vars[@]+"${vars[@]}"} "$bin" "$MODEL" "$PROMPT" --chat --ngen "$NGEN" \
    >"$OUT/$name.out" 2>"$OUT/$name.err"
  after=$(sectors)
  read -r -a b <<<"$before"
  read -r -a a <<<"$after"
  for i in "${!a[@]}"; do bytes+=("$(((a[i] - b[i]) * 512))"); done
  printf '%s %s\n' "$DRIVES" "${bytes[*]}" >"$OUT/$name.disks"
  printf '%s %s: %s\n' "$(date -u +%H:%M:%S)" "$name" "$(grep -h '^\[K3\] decode' "$OUT/$name.err")"
}

for arm in "${ARMS[@]}"; do
  IFS=@ read -r -a parts <<<"$arm"
  sha256sum "${parts[1]}"
done >"$OUT/binaries.sha256"
n=${#ARMS[@]}
for r in $(seq 1 "$RUNS"); do
  for k in $(seq 0 $((n - 1))); do run "${ARMS[$(((r - 1 + k) % n))]}" "$r"; done
done
