#!/bin/bash
# run-arm2276.sh — one benchmark arm of the 2026-10-21 upstream-tip campaign.
# Mirrors scripts/ab-forks.sh controls (same model, prompt, state snapshot,
# cgroup ceiling, interleaving handled by the caller) but takes the engine dir
# so arms can point at any worktree build.
#
# Usage: run-arm2276.sh <engine-c-dir> <tag> [extra env assignments...]
# Output: /root/bench-colibri/up2276/<tag>.log  (parse line appended to summary)
set -u
DIR=$1; TAG=$2; shift 2
BIN=$DIR/deepseek_v4
MODEL=/root/Projects/colibri-v4-docker/models/DeepSeek-V4-Flash
PROMPT=/root/Projects/colibri-upstream/bench/prompt_2k.txt
USAGE_SNAP=/root/Projects/colibri-upstream/bench/coli_usage.fixed
MFU_SNAP=/root/Projects/colibri-upstream/bench/coli_mfu.fixed
OUT=/tmp/selftune/st-out
RAM_GB=${RAM_GB:-28}; OMP=16; TOKENS=64
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "missing $BIN" >&2; exit 2; }
if ! nm "$BIN" 2>/dev/null | grep -q coli_vk; then
  echo "$TAG: refusing a CPU relink (no Vulkan symbols in $BIN)" >&2; exit 1
fi

# same expert-history state before every run
cp "$USAGE_SNAP" "$MODEL/.coli_usage"
[ -f "$MFU_SNAP" ] && cp "$MFU_SNAP" "$MODEL/.coli_mfu" || true
rm -f "$MODEL/.coli_ckpt" 2>/dev/null || true   # no cold-KV restore in perf arms

LOG=$OUT/$TAG.log

# Device-state guard. The chain fit and the expert tier size themselves from the
# DEVICE-WIDE free memory (VK_EXT_memory_budget: the heap less every VRAM byte
# on the card, ours and any other holder's), and both are decided once at open,
# so an arm started beside leftover VRAM -- another process's buffers, a killed
# test's pages the kernel still holds -- silently shrinks the tier for the whole
# run. The 2026-10-22 mm8-on arm lost 2.1 GiB this way (fit 14347628544 B free
# against the clean arms' 16624951295 B, tier 325 against 469 experts) with the
# engine itself allocating nothing: COLI_VK_MM8 is lazy, 0 device bytes at the
# fit print in both arms. Refuse above the idle floor (this card's ~600 MiB) so
# a confounded arm cannot reach summary.txt; VRAM_MAX_USED_MB overrides a
# co-tenant you know about (<=0 disables). The pre-run counter lands in the log
# either way (after the run's own output -- the run truncates the file), so an
# anomaly stays attributable.
VRAM_SYSFS=$(ls /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
VRAM_MAX_MB=${VRAM_MAX_USED_MB:-1024}
if [ -n "$VRAM_SYSFS" ]; then
  VRAM0=$(cat "$VRAM_SYSFS")
  if [ "$VRAM_MAX_MB" -gt 0 ] && [ "$VRAM0" -gt $((VRAM_MAX_MB * 1048576)) ]; then
    echo "vram0=$VRAM0 B ($VRAM_SYSFS)" >>"$LOG"
    echo "$TAG: refusing an arm beside $((VRAM0 / 1048576)) MiB of VRAM already in use" \
      "(the card's idle floor is ~600 MiB; VRAM_MAX_USED_MB overrides)" | tee -a "$LOG" >&2
    exit 2
  fi
fi

systemd-run --scope --quiet --unit=arm2276-$TAG --collect \
  -p MemoryMax=$((RAM_GB + 6))G -p MemorySwapMax=2G \
  env COLI_TEMP=0 KVSAVE=0 V4_PREFIX_CKPT=0 COLI_VULKAN=1 COLI_V4_PREWARM=0 \
      OMP_NUM_THREADS=$OMP OMP_PROC_BIND=close OMP_PLACES=cores \
      "$@" \
  timeout 3600 "$BIN" "$MODEL" --prompt-file "$PROMPT" --max-tokens "$TOKENS" \
      --memory-gb "$RAM_GB" >"$LOG" 2>&1
echo "rc=$? $TAG" >>"$LOG"
[ -n "${VRAM0:-}" ] && echo "vram0=$VRAM0 B ($VRAM_SYSFS)" >>"$LOG"

awk '
  /time_to_first_token=/ { split($0,a,"time_to_first_token="); split(a[2],b," "); ttft=b[1] }
  /TUNE decode:/         { split($0,c,"TUNE decode:"); n=split(c[2],d," "); toks=d[1]; for(i=1;i<=n;i++) if(d[i]=="in") sec=d[i+1] }
  END {
    if (sec=="" || ttft=="") { print "PARSE-FAIL"; exit }
    dec = (toks-1) / (sec-ttft)
    printf "%.3f %.4f %.3f\n", ttft, dec, sec
  }' "$LOG" | tee -a "$OUT/summary.txt" | sed "s/^/   $TAG  ttft decode wall: /"
