#!/usr/bin/env bash
# GLM-5.2 Expert Atlas — probe sweep (#175).
#
#   cd c && ./tools/expert_atlas/sweep.sh [probes.json] [outdir]
#
# Env: COLI_MODEL, NGEN (default 64), COLI (default ./coli)
#
# ---------------------------------------------------------------------------------------------
# THE CONFOUNDS THIS SCRIPT EXISTS TO CONTROL. Each one silently corrupts the atlas.
#
#   TOPP=0      --topp prunes experts by cumulative probability. Measured on GLM-5.2 int4,
#               same prompt, only top-p changed:
#                   topp=0    -> 21,000 selections across 7,587 distinct experts
#                   topp=0.7  -> 11,944 selections across 4,687 distinct experts
#               It hides 38% of the experts. It is also the RECOMMENDED SPEED SETTING, so this
#               is very easy to walk into: with top-p on you profile the pruner, not the model.
#
#   MTP=0       Speculative drafts route experts for tokens that are later REJECTED and never
#   DRAFT=0     emitted (eusage is incremented inside moe(), before verification). Those counts
#               would describe text the model never produced.
#
#   --gpu none  The CUDA expert tier is not run-to-run deterministic (VRAM placement shifts
#               between runs). Routing on the CPU path is reproducible. The tier only decides
#               where weights live, not what the router picks, so CPU costs nothing here.
#
#   --temp 0    Greedy: deterministic continuation, reproducible atlas.
#
#   rm .coli_usage before EVERY run
#               eusage is LOADED from <model>/.coli_usage at startup and written back at exit, so
#               a naive STATS dump contains ALL PRIOR HISTORY, not this run. Removing it per run
#               makes each dump exactly one probe. The user's learned cache is backed up and
#               restored on exit.
#
# Prompt lengths are kept in a narrow band across categories: prefill routes the prompt tokens
# too, so a verbose category would otherwise simply look "busier".
# ---------------------------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROBES="${1:-$HERE/probes.json}"
OUT="${2:-./atlas_out}"
COLI="${COLI:-./coli}"
NGEN="${NGEN:-64}"
MODEL="${COLI_MODEL:?set COLI_MODEL to the snapshot directory}"

[ -f "$PROBES" ] || { echo "no probe file: $PROBES" >&2; exit 1; }
[ -x "$COLI" ]   || { echo "no coli at $COLI (run from c/, or set COLI=)" >&2; exit 1; }
mkdir -p "$OUT/stats"

export COLI_MODEL="$MODEL" MTP=0 DRAFT=0 TOPP=0

USAGE="$MODEL/.coli_usage"
BACKUP="$OUT/.coli_usage.backup"
[ -f "$USAGE" ] && cp "$USAGE" "$BACKUP" && echo "backed up $USAGE"
restore(){ [ -f "$BACKUP" ] && cp "$BACKUP" "$USAGE" && echo "restored $USAGE"; }
trap restore EXIT

python3 - "$PROBES" > "$OUT/runlist.tsv" <<'PY'
import json, sys
for cat, prompts in json.load(open(sys.argv[1])).items():
    if cat.startswith('_'):
        continue
    for i, p in enumerate(prompts):
        print(f"{cat}\t{i}\t{p}")
PY

n=$(wc -l < "$OUT/runlist.tsv"); i=0
echo "$n probes -> $OUT/stats"
while IFS=$'\t' read -r cat idx prompt; do
  i=$((i+1))
  dst="$OUT/stats/${cat}_${idx}.txt"
  [ -s "$dst" ] && { echo "  [$i/$n] $cat/$idx (cached)"; continue; }
  rm -f "$USAGE"                                   # start from an EMPTY routing history
  STATS="$dst" "$COLI" run "$prompt" --ngen "$NGEN" --ctx 4096 --gpu none --temp 0 \
      > "$OUT/stats/${cat}_${idx}.log" 2>&1
  echo "  [$i/$n] $cat/$idx  $(grep -aoE '[0-9]+ selections across [0-9]+ distinct experts' \
        "$OUT/stats/${cat}_${idx}.log" | head -1)"
done < "$OUT/runlist.tsv"

echo
echo "next:"
echo "  python3 $HERE/analyze.py  --stats $OUT/stats --out $OUT/experts.json"
echo "  python3 $HERE/validate.py $OUT/stats 200"
