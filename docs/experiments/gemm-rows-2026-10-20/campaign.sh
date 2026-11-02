#!/bin/bash
# One-variable A/B for COLI_VK_CHAIN_GEMM_ROWS (0 vs -1): 3 interleaved samples
# per arm, fixed .coli_usage, same RAM/threads/prompt, greedy; hashes the logs
# for docs/experiments. COLI_VK_TIER_HOST=0 is held fixed in BOTH arms (this
# branch predates the host-pool bound; without it the pool takes 25.75 GiB and
# the device is lost — that variable is not the one under test).
# Run scripts/build-vk.sh first.
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(cd -- "$HERE/../.." && pwd)
MODEL=${MODEL:?point MODEL at your DeepSeek V4 Flash checkpoint}
PROMPT=$REPO/bench/prompt_2k.txt
USAGE=${USAGE:?set USAGE=/path/to/a/fixed/.coli_usage snapshot -- the campaign pins expert history so arms are comparable}
OUT=${OUT:-$HERE/run-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
RAM=28; OMP=16; TOK=32

run() {
  arm=$1; rep=$2; log=$OUT/$arm-$rep.log
  if [ "$arm" = trial ]; then rows=-1; else rows=0; fi
  cp "$USAGE" "$MODEL/.coli_usage"
  systemd-run --scope --quiet --unit=gemmab-$arm-$rep --collect \
    -p MemoryMax=$((RAM+6))G -p MemorySwapMax=2G \
    env COLI_TEMP=0 KVSAVE=0 V4_PREFIX_CKPT=0 OMP_NUM_THREADS=$OMP \
        OMP_PROC_BIND=close OMP_PLACES=cores COLI_VULKAN=1 \
        COLI_VK_TIER_HOST=0 COLI_VK_CHAIN_GEMM_ROWS=$rows \
        timeout 3600 \
    "$REPO/c/deepseek_v4" "$MODEL" --prompt-file "$PROMPT" \
        --max-tokens $TOK --memory-gb $RAM >"$log" 2>&1
  awk -v tag="$arm-$rep" '
    /time_to_first_token=/ { split($0,a,"time_to_first_token="); split(a[2],b," "); ttft=b[1] }
    /TUNE decode:/         { split($0,c,"TUNE decode:"); n=split(c[2],d," "); for(i=1;i<=n;i++) if(d[i]=="in") sec=d[i+1] }
    /generated_text=/      { sub(/^generated_text=/,""); text=substr($0,1,60) }
    END { printf "%s ttft=%.3f wall=%.3f text=%s\n", tag, ttft, sec, text }' "$log"
}

for rep in 1 2 3; do run baseline $rep; run trial $rep; done | tee "$OUT/samples.txt"
echo "logs: $OUT"; sha256sum "$OUT"/*.log | tee "$OUT/logs.sha256"
