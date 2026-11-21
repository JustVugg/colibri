#!/bin/bash
# q38_bench.sh <engine-binary> <model-dir> <arm-name> <runs>
# One JSON object per arm on stdout (progress on stderr). Greedy only (the engine is argmax).
# Optional env: ARM_ENV="Q38_MTP=0 ..." extra engine env (recorded in JSON "env");
#               PREFILL_RUNS=<n> (default <runs>; 0 = decode only, prefill_median null); NGEN=64; CAP=32; BITS=8 (reserved bits 2..8);
#               WARMUP=1 (default; one discarded decode run per arm, 0 disables);
#               EFFORT_NOTE="..." (recorded as "effort"); BUILD=<build label> and METAL=<0|1> (recorded);
#               BENCH_RAW=<dir> parent for raw logs; each arm gets its own unique <dir>/<arm>.<timestamp>.<pid>.
#               Test hooks: TIME_BIN (default /usr/bin/time), SWAPUSAGE_CMD (default "sysctl -n vm.swapusage").
# Engine invocation: SNAP=<model-dir> COLI_TEMP=0 Q38_MAXT=32768 N_NEW=<n> <binary> <CAP> <BITS> <prompt.txt>
#   (argv is `<cache/layer> <reserved bits 2..8> <prompt file>`; N_NEW sets tokens generated.)
# Parsing (stderr): "[enc] prompt tokens: N", "TTFT: X s", "Speed: Y tok/s (Z s for K tokens)".
#   prefill_s = TTFT of N_NEW=1 run on long12k.txt.
#   decode_tok_s = (NGEN-1)/(Z - TTFT) on short.txt: tokens after the first over time after TTFT.
# Footprint: "peak memory footprint" from /usr/bin/time -l (max over runs). Swap: sysctl vm.swapusage used (K/M/G), after-before.
# Every parse is checked; a miss or a non-positive decode time exits 2 with a message.
set -u
BIN=$1; MODEL=$2; ARM=$3; RUNS=$4
PREFILL_RUNS=${PREFILL_RUNS:-$RUNS}; NGEN=${NGEN:-64}; CAP=${CAP:-32}
DIR=$(cd "$(dirname "$0")" && pwd); P=$DIR/prompts
BITS=${BITS:-8}; WARMUP=${WARMUP:-1}; BUILD=${BUILD:-unset}; METAL=${METAL:-unset}
TIME_BIN=${TIME_BIN:-/usr/bin/time}; SWAPUSAGE_CMD=${SWAPUSAGE_CMD:-sysctl -n vm.swapusage}
EFFORT_NOTE=${EFFORT_NOTE:-low (system prompt in short.txt via openai_server.render_chat_qwen38; no engine env)}
die() { echo "bench error: $*" >&2; exit 2; }
case $BITS in [2-8]) ;; *) die "BITS must be 2..8, got '$BITS'";; esac
case $WARMUP in 0|1) ;; *) die "WARMUP must be 0 or 1, got '$WARMUP'";; esac
isnum() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
isint() { [[ "$1" =~ ^[0-9]+$ ]]; }
isint "$RUNS" && [ "$RUNS" -ge 1 ] || die "runs must be an integer >= 1, got '$RUNS'"
isint "$PREFILL_RUNS" || die "PREFILL_RUNS must be an integer >= 0, got '$PREFILL_RUNS'"
RAWBASE=${BENCH_RAW:-$(mktemp -d)}; RAW=$RAWBASE/$ARM.$(date +%Y%m%d-%H%M%S).$$
mkdir -p "$RAWBASE" && mkdir "$RAW" || die "cannot create unique raw dir $RAW"
unset OMP_NUM_THREADS
swap_mb() { # used swap in MB (unit K/M/G honoured); prints nothing on a parse miss
  $SWAPUSAGE_CMD | sed -nE 's/.*used = ([0-9.]+)([KMG]).*/\1 \2/p' | awk 'NF==2{u=$2; printf "%.2f", (u=="G")?$1*1024:(u=="K")?$1/1024:$1}'; }
preflight() { # exit 3 if port 8000 is listening or any qwen38 / qwen38-cpu / qwen38-metal / <basename BIN> process exists (checked before launch, so never our own child)
  local busy; busy=$(lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null)
  if [ -n "$busy" ]; then echo "PREFLIGHT FAILED: something is listening on TCP 8000:" >&2; echo "$busy" >&2; exit 3; fi
  local n; for n in qwen38 qwen38-cpu qwen38-metal "$(basename "$BIN")"; do
    busy=$(pgrep -lx "$n" 2>/dev/null)
    if [ -n "$busy" ]; then echo "PREFLIGHT FAILED: a $n process is running (not killing it):" >&2; echo "$busy" >&2; exit 3; fi
  done
}
run1() { # prompt nnew tag -> log path
  local log=$RAW/$ARM.$3.log
  env SNAP="$MODEL" COLI_TEMP=0 Q38_MAXT=32768 N_NEW=$2 ${ARM_ENV:-} "$TIME_BIN" -l "$BIN" "$CAP" "$BITS" "$1" >"$log" 2>&1
  echo "$log"
}
SWAP0=$(swap_mb); isnum "$SWAP0" || die "cannot parse numeric swap used (before) from vm.swapusage"
if [ "$WARMUP" = 1 ]; then
  echo "[$ARM] warmup decode run (discarded)" >&2
  preflight
  L=$(run1 "$P/short.txt" "$NGEN" "warm")
  grep -q '^Speed: ' "$L" || die "warmup run produced no Speed line in $L"
fi
DEC=(); PRE=(); FP=0; PT=0; PTL=0
for ((i = 1; i <= RUNS; i++)); do  # not seq: BSD `seq 1 0` counts down
  echo "[$ARM] decode run $i/$RUNS" >&2
  preflight
  L=$(run1 "$P/short.txt" "$NGEN" "dec$i")
  TT=$(sed -nE 's/^TTFT: ([0-9.]+) s.*/\1/p' "$L" | tail -1)
  SZ=$(sed -nE 's/^Speed: [0-9.]+ tok\/s \(([0-9.]+)s for ([0-9]+) tokens\)/\1/p' "$L" | tail -1)
  KK=$(sed -nE 's/^Speed: [0-9.]+ tok\/s \(([0-9.]+)s for ([0-9]+) tokens\)/\2/p' "$L" | tail -1)
  PT=$(sed -nE 's/.*\[enc\] prompt tokens: ([0-9]+).*/\1/p' "$L" | head -1)
  F=$(awk '/peak memory footprint/{print $1}' "$L" | tail -1)
  isnum "$TT" && isnum "$SZ" && isint "$KK" || die "parse failure (TTFT/Speed) in $L"
  isint "$PT" || die "parse failure (prompt tokens) in $L"
  isint "$F" || die "parse failure (peak memory footprint) in $L"
  awk -v z="$SZ" -v t="$TT" 'BEGIN{exit !(z-t>0)}' || die "non-positive decode time (Z=$SZ, TTFT=$TT) in $L"
  [ "$KK" -gt 1 ] || die "generated only $KK token(s); decode rate undefined in $L"
  DEC+=("$(awk -v k="$KK" -v z="$SZ" -v t="$TT" 'BEGIN{printf "%.4f",(k-1)/(z-t)}')")
  FP=$(awk -v a="$FP" -v b="$F" 'BEGIN{print (b>a)?b:a}')
done
for ((i = 1; i <= PREFILL_RUNS; i++)); do
  echo "[$ARM] prefill run $i/$PREFILL_RUNS" >&2
  preflight
  L=$(run1 "$P/long12k.txt" 1 "pre$i")
  TT=$(sed -nE 's/^TTFT: ([0-9.]+) s.*/\1/p' "$L" | tail -1)
  PTL=$(sed -nE 's/.*\[enc\] prompt tokens: ([0-9]+).*/\1/p' "$L" | head -1)
  F=$(awk '/peak memory footprint/{print $1}' "$L" | tail -1)
  isnum "$TT" || die "parse failure (TTFT) in $L"
  isint "$PTL" || die "parse failure (prompt tokens) in $L"
  isint "$F" || die "parse failure (peak memory footprint) in $L"
  [ "$PTL" -gt 0 ] || die "zero prompt tokens in $L"
  PRE+=("$TT")
  FP=$(awk -v a="$FP" -v b="$F" 'BEGIN{print (b>a)?b:a}')
done
SWAP1=$(swap_mb); isnum "$SWAP1" || die "cannot parse numeric swap used (after) from vm.swapusage"
GD=$(dirname "$BIN")/..
COMMIT=$(git -C "$GD" rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ "$COMMIT" != unknown ] && [ -n "$(git -C "$GD" status --porcelain --untracked-files=no 2>/dev/null)" ]; then COMMIT=$COMMIT-dirty; fi
python3 -I - "$ARM" "$COMMIT" "${ARM_ENV:-}" "$CAP" "$NGEN" "$P" "$PT" "$PTL" "$FP" "$SWAP0" "$SWAP1" "$RAW" "$MODEL" "$BITS" "$EFFORT_NOTE" "$BUILD" "$METAL" "$WARMUP" "${DEC[*]:-}" "|" "${PRE[*]:-}" <<'PY'
import sys,json,hashlib,statistics as st
a=sys.argv[1:]
arm,commit,aenv,cap,ngen,P,pt,ptl,fp,s0,s1,raw,model,bits,effort,build,metal,warmup=a[:18]
rest=a[18:]; k=rest.index("|")
dec=[float(x) for x in " ".join(rest[:k]).split()]; pre=[float(x) for x in " ".join(rest[k+1:]).split()]
env={"SNAP":model,"COLI_TEMP":"0","Q38_MAXT":"32768","N_NEW_decode":ngen,"N_NEW_prefill":"1","OMP_NUM_THREADS":"unset",
     "argv":f"{cap} {bits} <prompt>","BITS":bits,"effort":effort,"BUILD":build,"METAL":metal,"WARMUP":warmup}
for kv in aenv.split():
    n,v=kv.split("=",1); env[n]=v
sha=lambda f:hashlib.sha256(open(f"{P}/{f}","rb").read()).hexdigest()
print(json.dumps({"arm":arm,"commit":commit,"env":env,
 "prompt_sha256":{"short.txt":sha("short.txt"),"long12k.txt":sha("long12k.txt")},
 "prompt_tokens":{"short.txt":int(pt),"long12k.txt":int(ptl)},
 "decode_tok_s":dec,"decode_median":st.median(dec),
 "prefill_s":pre,"prefill_median":st.median(pre) if pre else None,
 "peak_footprint_gb":round(int(fp)/2**30,2),
 "swap_used_delta_mb":round(float(s1)-float(s0),2),"raw_logs":raw}))
PY
