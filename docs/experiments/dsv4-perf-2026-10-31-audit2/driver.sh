#!/bin/bash
# driver.sh — benchmark-reproduction driver, audit-wt lane 06 (fresh builds).
# PR binary  = /root/audit-wt/pr-bench/c/deepseek_v4   @ cb0a0a91
# BASE binary= /root/audit-wt/base-tests/c/deepseek_v4 @ d67ebd8b
# Arms, interleaved R=1..3, order shuffled per rep:
#   audit2-legacy-rN   PR, COLI_VK_LEGACY=1
#   audit2-defaults-rN PR, env-free
#   audit2-oneline-rN  PR, COLI_VK_PROFILE=RDNA2 COLI_VK_TIER_HOST_GB=2
#   audit2-uvk-r1/r2   BASE, env-free (mixed into reps 1-2)
#   audit2-mm8-r1      PR, oneline env + COLI_VK_CHAIN_MM8_NDBUF=2 (last, optional)
# Regime every leg: VRAM_MAX_USED_MB=3584 RAM_GB=16 COLI_BENCH_MEMORY_GB=24 via
# /root/audit-wt/bench/run-audit.sh (imp3-stack run-arm2276.sh copy, OUT changed only).
# Modeled on audit/2026-10-31 evidence/bench/audit-driver.sh: pre-leg VRAM gate
# (7 samples over 70s, band<=64MiB max<=3544MiB avail>=2GiB, NO re-gate, blocked
# >10 min -> SKIPPED-GATE and continue — a concurrent GPU session is expected),
# one retry per failed leg (-r2 suffix), resumable status file, lock+window log,
# device-lost stop. Gate-skipped legs stay retryable on the next driver run
# (terminal after 4 gate-skips so a driver relaunch can recover contention).
set -u
BENCH=/root/audit-wt/bench
OUT=$BENCH
RUN=$BENCH/run-audit.sh
WIN=$OUT/window.log
REP=$OUT/interim.md
STATUS=$OUT/legs
PR=/root/audit-wt/pr-bench/c
BASE=/root/audit-wt/base-tests/c
MODEL=/root/Projects/colibri-v4-docker/models/DeepSeek-V4-Flash
PROMPT=/root/Projects/colibri-upstream/bench/prompt_2k.txt
VRAM_SYSFS=$(ls /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
CEIL_MB=3584          # VRAM_MAX_USED_MB handed to the harness (regime, every leg)
GATE_MAX_MB=3544      # pre-leg co-tenant ceiling (64 MiB under the harness's own)
T0=$SECONDS
mkdir -p "$STATUS"
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo "$(date) another driver holds $OUT/driver.lock — exiting" >>"$WIN"; exit 4; }

vram(){ cat "$VRAM_SYSFS" 2>/dev/null || echo 0; }
availg(){ free -g | awk 'NR==2{print $7}'; }
freeg(){ free -g | awk 'NR==2{print "total "$2" used "$3" free "$4" avail "$7}'; }

wlog(){ echo "$*" >>"$WIN"; }

# ---- pre-flight (stop/ask conditions) -----------------------------------------------
preflight(){
  local ok=1
  for b in "$BASE/deepseek_v4" "$PR/deepseek_v4"; do
    [ -x "$b" ] || { wlog "PREFLIGHT-FAIL: missing $b"; ok=0; }
    nm "$b" 2>/dev/null | grep -q coli_vk || { wlog "PREFLIGHT-FAIL: no Vulkan symbols in $b"; ok=0; }
  done
  [ -d "$MODEL" ] && [ -r "$MODEL" ] || { wlog "PREFLIGHT-FAIL: model dir unreadable $MODEL"; ok=0; }
  [ -r "$PROMPT" ] || { wlog "PREFLIGHT-FAIL: prompt unreadable $PROMPT"; ok=0; }
  [ -x "$RUN" ] || { wlog "PREFLIGHT-FAIL: harness missing $RUN"; ok=0; }
  [ "$ok" = 1 ] || exit 3
  wlog "preflight ok $(date): binaries+nm, model, prompt, harness; vram_sysfs=$VRAM_SYSFS vram0=$(vram)"
}

# ---- pre-leg device gate (gap3 pattern, no re-gate; SKIPPED-GATE + continue) ----------
# 0=go, 1=skip leg (blocked >10 min — e.g. a concurrent GPU session holds the card)
gate(){ # tag
  local tag=$1 t0=$SECONDS w=0 i mn mx av bandMB maxMB
  while :; do
    local s=()
    for i in 1 2 3 4 5 6 7; do s+=("$(vram)"); [ "$i" -lt 7 ] && sleep 10; done
    mn=$(printf '%s\n' "${s[@]}" | sort -n | head -1)
    mx=$(printf '%s\n' "${s[@]}" | sort -n | tail -1)
    av=$(availg); bandMB=$(( (mx-mn)/1048576 )); maxMB=$(( mx/1048576 ))
    if [ $((mx-mn)) -le 67108864 ] && [ "$mx" -le $((GATE_MAX_MB*1048576)) ] && [ "$av" -ge 2 ]; then
      wlog "  gate[$tag]: GO (band=${bandMB}MiB max=${maxMB}MiB avail=${av}GiB)"
      return 0
    fi
    if [ $((SECONDS-t0)) -gt 600 ]; then
      wlog "  gate[$tag]: BLOCKED >10 min (band=${bandMB}MiB max=${maxMB}MiB avail=${av}GiB — need band<=64MiB max<=$GATE_MAX_MB avail>=2GiB) — leg SKIPPED-GATE (retry on next driver run) $(date)"
      return 1
    fi
    w=$((w+1))
    wlog "  gate[$tag]: w$w retry (band=${bandMB}MiB max=${maxMB}MiB avail=${av}GiB — need band<=64MiB max<=$GATE_MAX_MB avail>=2GiB)"
    sleep 60
  done
}

# ---- mem-peak sampler (driver-side; the imp3-stack harness has no memtee) ------------
memsampler(){ # harness-pid tag
  local hpid=$1 tag=$2 peak=0 hwm cg scope
  scope=/sys/fs/cgroup/system.slice/arm2276-$tag.scope/memory.peak
  while kill -0 "$hpid" 2>/dev/null; do
    for p in /proc/[0-9]*/stat; do
      p=${p#/proc/}; p=${p%/stat}
      [ -r "/proc/$p/comm" ] || continue
      [ "$(cat "/proc/$p/comm" 2>/dev/null)" = deepseek_v4 ] || continue
      hwm=$(awk '/^VmHWM:/{print $2}' "/proc/$p/status" 2>/dev/null)
      [ -n "${hwm:-}" ] && [ "$hwm" -gt "$peak" ] && peak=$hwm
    done
    sleep 5
  done
  cg=$(cat "$scope" 2>/dev/null || echo 0)
  awk -v pk="$peak" -v cg="$cg" 'BEGIN{printf "mem peak=%.2fGiB (VmHWM %d kB, scope memory.peak %d B)\n", pk/1048576, pk, cg}' >"$OUT/$tag.mempeak"
}

# ---- one harness attempt --------------------------------------------------------------
attempt(){ # tag dir [envs...] ; rc0 = green (rc=0 and parse ok)
  local tag=$1 dir=$2; shift 2
  local t0=$SECONDS
  wlog "audit2-$tag: launch $(date +%H:%M:%S) dir=$dir vram0=$(vram) free-g: $(freeg) envs: $*"
  env VRAM_MAX_USED_MB=$CEIL_MB RAM_GB=16 COLI_BENCH_MEMORY_GB=24 \
      "$RUN" "$dir" "audit2-$tag" "$@" >"$OUT/audit2-$tag.parse" 2>&1 &
  local hpid=$!
  memsampler "$hpid" "audit2-$tag" &
  local spid=$!
  wait "$hpid" 2>/dev/null
  local w=0
  while [ ! -s "$OUT/audit2-$tag.mempeak" ] && kill -0 "$spid" 2>/dev/null && [ $w -lt 10 ]; do sleep 1; w=$((w+1)); done
  kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  local el=$((SECONDS-t0)) L="$OUT/audit2-$tag.log"
  local rc; rc=$(grep -o "^rc=[0-9]*" "$L" 2>/dev/null | tail -1 | cut -d= -f2); rc=${rc:-NA}
  local parse; parse=$(grep -vE '^\s*$' "$OUT/audit2-$tag.parse" 2>/dev/null | tail -1)
  local mp; mp=$(cat "$OUT/audit2-$tag.mempeak" 2>/dev/null || echo "mem peak=n/a")
  wlog "audit2-$tag: done rc=$rc $(printf '%02d:%02d:%02d' $((el/3600)) $(((el/60)%60)) $((el%60))); parse:$parse; $mp"
  grep -m1 "tier deepseek_v4: on," "$L" 2>/dev/null | cut -c1-190 | sed 's/^/  tier-boot: /' >>"$WIN"
  grep -m1 -o "chain: [0-9]* of [0-9]* layers on the device" "$L" 2>/dev/null | sed 's/^/  chain: /' >>"$WIN"
  grep -m1 "window probe" "$L" 2>/dev/null | sed 's/^/  windowprobe: /' >>"$WIN"
  grep -m1 "attention FA" "$L" 2>/dev/null | cut -c1-120 | sed 's/^/  fa: /' >>"$WIN"
  grep -m1 "decode-path=fork.*engaged\|slab registry armed" "$L" 2>/dev/null | cut -c1-150 | sed 's/^/  fork: /' >>"$WIN"
  grep -m1 "s1gemv.*armed" "$L" 2>/dev/null | cut -c1-120 | sed 's/^/  s1gemv: /' >>"$WIN"
  grep -m1 "cs experts armed" "$L" 2>/dev/null | cut -c1-120 | sed 's/^/  cs: /' >>"$WIN"
  grep -m1 "decode grow armed" "$L" 2>/dev/null | cut -c1-120 | sed 's/^/  grow: /' >>"$WIN"
  grep -m1 "legacy (COLI_VK_LEGACY=1)" "$L" 2>/dev/null | cut -c1-120 | sed 's/^/  legacy1: /' >>"$WIN"
  if grep -qiE "device lost|VK_ERROR_DEVICE_LOST" "$L" 2>/dev/null; then
    wlog "  DEVICE-LOST in audit2-$tag — DRIVER STOPPED $(date)"
    exit 9
  fi
  [ "$rc" = 0 ] && ! grep -q "PARSE-FAIL" "$OUT/audit2-$tag.parse" 2>/dev/null
}

# ---- one leg (gate, attempt, retry-once) ----------------------------------------------
leg(){ # tag dir [envs...]
  local tag=$1 dir=$2; shift 2
  local st="$STATUS/audit2-$tag.status" cur=""
  [ -f "$st" ] && cur=$(head -1 "$st")
  case "$cur" in
    GREEN*)   wlog "audit2-$tag: SKIP (already green: $cur)"; return 0 ;;
    FAILED*)  wlog "audit2-$tag: SKIP (terminal FAILED)"; return 0 ;;
  esac
  if ! gate "$tag"; then
    local n=$(sed -n 's/^GATESKIP \([0-9]*\).*/\1/p' "$st" 2>/dev/null); n=${n:-0}
    echo "GATESKIP $((n+1)) $(date)" >"$st"
    if [ $((n+1)) -ge 4 ]; then wlog "audit2-$tag: SKIPPED-GATE terminal (four gate-skips)"; else wlog "audit2-$tag: SKIPPED-GATE (logged; retry on next driver run)"; fi
    return 0
  fi
  if [ "$cur" = "FAIL 1" ]; then
    # first attempt already burned (this driver run or a previous one): go to -r2
    if attempt "$tag-r2" "$dir" "$@"; then echo "GREEN via r2 $(date)" >"$st"; wlog "audit2-$tag: GREEN via retry -r2"; else echo "FAILED $(date) (2 attempts)" >"$st"; wlog "audit2-$tag: FAILED (2 attempts; tails in audit2-$tag.log / audit2-$tag-r2.log)"; fi
    return 0
  fi
  if attempt "$tag" "$dir" "$@"; then
    echo "GREEN $(date) :: $(grep -vE '^\s*$' "$OUT/audit2-$tag.parse" 2>/dev/null | tail -1)" >"$st"
    wlog "audit2-$tag: GREEN"
  else
    echo "FAIL 1 $(date)" >"$st"
    if attempt "$tag-r2" "$dir" "$@"; then echo "GREEN via r2 $(date) :: $(grep -vE '^\s*$' "$OUT/audit2-$tag-r2.parse" 2>/dev/null | tail -1)" >"$st"; wlog "audit2-$tag: GREEN via retry -r2"; else echo "FAILED $(date) (2 attempts)" >"$st"; wlog "audit2-$tag: FAILED (2 attempts; tails in audit2-$tag.log / audit2-$tag-r2.log)"; fi
  fi
  return 0
}

# ---- INTERIM checkpoint (idempotent per rep; driver-local) ----------------------------
legrow(){ # tag (no audit2- prefix)
  local L="$OUT/audit2-$1.log" rc ttft dec wall v0 mp ch fa md5
  rc=$(grep -o "^rc=[0-9]*" "$L" 2>/dev/null | tail -1 | cut -d= -f2); rc=${rc:-NA}
  read -r ttft dec wall <<<"$(awk '
    /time_to_first_token=/ { split($0,a,"time_to_first_token="); split(a[2],b," "); ttft=b[1] }
    /TUNE decode:/         { split($0,c,"TUNE decode:"); n=split(c[2],d," "); toks=d[1]; for(i=1;i<=n;i++) if(d[i]=="in") sec=d[i+1] }
    END { if (sec=="" || ttft=="") print "NA NA NA"; else printf "%.3f %.4f %.3f", ttft, (toks-1)/(sec-ttft), sec }' "$L" 2>/dev/null)"
  v0=$(grep -o "vram0=[0-9]*" "$L" 2>/dev/null | tail -1 | cut -d= -f2)
  mp=$(sed -n 's/^mem peak=\([0-9.]*\)GiB.*/\1/p' "$OUT/audit2-$1.mempeak" 2>/dev/null)
  ch=$(grep -m1 -o "chain: [0-9]* of [0-9]* layers" "$L" 2>/dev/null)
  fa=$(grep -m1 -o "attention FA [a-z]*" "$L" 2>/dev/null)
  md5=$(grep -m1 -o "generated_text=.*" "$L" 2>/dev/null | head -c 3000000 | md5sum | cut -d' ' -f1)
  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$1" "$rc" "${ttft:-NA}" "${dec:-NA}" "${wall:-NA}" "$([ -n "$v0" ] && echo $((v0/1048576)) || echo NA)" "${mp:-NA}" "${ch:-NA} ${fa:-noFA}" "${md5:-NA}"
}

checkpoint(){ # phase-label tag...
  local phase=$1; shift
  if grep -q "INTERIM (auto-checkpoint) — $phase complete" "$REP" 2>/dev/null; then return 0; fi
  {
    echo ""
    echo "## INTERIM (auto-checkpoint) — $phase complete — $(date) (driver elapsed $((SECONDS/60)) min)"
    echo ""
    echo "| leg | rc | ttft s | decode tok/s | wall s | vram0 MiB | mem peak GiB | chain/FA | gen md5 |"
    echo "|---|---|---|---|---|---|---|---|---|"
    for t in "$@"; do legrow "$t"; done
  } >>"$REP"
  wlog "checkpoint: appended INTERIM $phase to $REP"
}

# ---- schedule ---------------------------------------------------------------------------
wlog "=== AUDIT2 DRIVER START $(date) (pid $$) — regime: VRAM_MAX_USED_MB=$CEIL_MB RAM_GB=16 COLI_BENCH_MEMORY_GB=24 — PR=$PR BASE=$BASE ==="
preflight

ALL_LEG=""
DONE_LEG=""
for R in 1 2 3; do
  order=(legacy defaults oneline)
  [ "$R" -le 2 ] && order=(uvk "${order[@]}")
  order=($(printf '%s\n' "${order[@]}" | shuf))
  wlog "rep $R shuffled order: ${order[*]}"
  for a in "${order[@]}"; do
    case "$a" in
      uvk)      leg "uvk-r$R"     "$BASE" ;;
      legacy)   leg "legacy-r$R"  "$PR" COLI_VK_LEGACY=1 ;;
      defaults) leg "defaults-r$R" "$PR" ;;
      oneline)  leg "oneline-r$R"  "$PR" COLI_VK_PROFILE=RDNA2 COLI_VK_TIER_HOST_GB=2 ;;
    esac
    ALL_LEG="$ALL_LEG $a-r$R"
    DONE_LEG="$DONE_LEG $a-r$R"
  done
  checkpoint "Rep $R" $ALL_LEG
done
# optional tail leg
leg "mm8-r1" "$PR" COLI_VK_PROFILE=RDNA2 COLI_VK_TIER_HOST_GB=2 COLI_VK_CHAIN_MM8_NDBUF=2
checkpoint "Tail (mm8)" $ALL_LEG mm8-r1

wlog "=== AUDIT2 DRIVER COMPLETE $(date) (elapsed $((SECONDS/60)) min) ==="
