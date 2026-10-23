# Sourced by tests/vulkan_engines.sh: qwen38 (Qwen3.8 Flash Next) with a partial dense
# chain (docs/vulkan.md, "A partial chain"): its first N layers on the device, the CPU
# running the other layers, the final mixer and lm_head from the four streams the chain
# hands back after layer N-1. On the tiny fixtures (four layers: a DeltaNet layer with the
# PLE, QSA, DeltaNet, QSA), against the CPU run of the same build:
#   - COLI_VK_CHAIN_LAYERS=k for k = 0..4, every resident format (bf16, the int8 trunk), the
#     int4-g64 experts, the PLE layer on either side of the handoff (a fixture with it at
#     layer 2), the CPU's prefill batching off, and the tier off (the per-matrix path on:
#     nothing that is not on the device already may go up);
#   - COLI_VK_DEVICE_CAP_MB under which exactly k layers fit (from a probe run's fit line):
#     the fit's N is the one the line's numbers give, and the run keeps it;
#   - an upload refused inside layer k's setup (COLI_VK_STAGED_FAULT, aimed from a probe's
#     marks), on the first forward's setup (the full chain), at startup (a partial chain)
#     and in the dense-host pass: N = k, the device holding the k layers' matrices only;
#   - with N < L: prompt chunks of 3 and a long prompt streaming its cold experts, MTP
#     drafts (accepted, rejected, mixed, three a verify) and prompt-lookup ones, the
#     speculative harness's byte-for-byte gate, serve sessions with prefix reuse and
#     pins, the KV split, prompts only, and the device lost mid-decode and in a verify;
#   - COLI_VK_DENSE_HOST=0 with N < L: only the N layers' host copies dropped, none read
#     back on a healthy run, and after a lost device only theirs;
#   - nothing forced on a device where everything fits: N = L with the head, as before, and
#     the same logits as with COLI_VK_CHAIN_LAYERS=4; coli plan's N, per-layer, fixed and
#     tail bytes the engine's (under the caps above).
# The CPU's tokens everywhere, logits within 1e-4 of the largest (bf16 trunk).

# ptl_q38_res <log>: the matrices the engine holds on the device at exit (the report line)
ptl_q38_res() {
  local n; n=$(sed -n 's/^\[VK\] qwen38: [0-9]* matmuls on the GPU (\([0-9]*\) matrices resident.*/\1/p' "$1" | tail -1)
  echo "${n:--1}"
}
# ptl_q38_setup_res <log>: the matrices resident when the chain's setup ended (its line)
ptl_q38_setup_res() {
  local n; n=$(sed -n 's/^\[VK\] qwen38 chain: [0-9]* layers on the device ([0-9]* QSA), \([0-9]*\) matrices resident.*/\1/p' "$1" | tail -1)
  echo "${n:--1}"
}
# ptl_q38_gate <tag> <N> <tol 0|1> <env...> -- <argv...>: the CPU run and the chain's
# (COLI_VK_CHAIN=${CHAINMODE:-1}) with the same settings. Gates: the CPU's tokens; logits
# within 1e-4 of the largest with tol 1; the final N; with N > 0 the chain ran, the device
# held the N layers' matrices after setup (the placed line), and with N < L nothing more
# went up afterwards (the matrices resident at exit are the setup's); with N = 0 the chain
# never ran and no matrix is on the device. CPUENV: the CPU arm's own settings.
ptl_q38_gate() {
  local tag=$1 want=$2 tol=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage cpu.f32 vk.f32
  local rc_cpu=0
  env "${envs[@]}" ${CPUENV:-} DUMP=cpu.f32 ./qwen38 "$@" > cpu.log 2>&1 || rc_cpu=$?
  env "${envs[@]}" DUMP=vk.f32 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./qwen38 "$@" > vk.log 2>&1 || { [ $rc_cpu != 0 ] || { cat vk.log; fail "$tag: the chain misses the oracle the CPU passes"; }; }
  rm -f chain.usage
  same_tokens cpu.log vk.log "$tag"
  ptl_check_n qwen38 vk.log "$want" "$tag"
  local L fw; L=$(ptl_L qwen38 vk.log); fw=$(chain_count qwen38 vk.log)
  if [ "$want" -gt 0 ]; then
    [ "$fw" -gt 0 ] || { cat vk.log; fail "$tag: the chain never ran"; }
    ptl_check_placed qwen38 vk.log "$tag"
    if [ "$want" -lt "$L" ]; then
      [ "$(ptl_q38_res vk.log)" = "$(ptl_q38_setup_res vk.log)" ] ||
        { grep -a '^\[VK\] qwen38' vk.log; fail "$tag: matrices went up past the chain's setup"; }
    fi
  else
    [ "$fw" = 0 ] || { grep -a '^\[VK\]' vk.log; fail "$tag: the chain ran with no layer on the device"; }
    [ "$(ptl_q38_res vk.log)" = 0 ] || { grep -a '^\[VK\] qwen38' vk.log; fail "$tag: matrices on the device with N = 0"; }
  fi
  local lg=""
  if [ "$tol" = 1 ]; then lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }; lg=", $lg"; fi
  echo "OK $tag: tokens = CPU$lg, $want of $L layers on the device, $fw chain forwards"
}
# ptl_q38_run <log> <env...> -- <argv...>: one chain run (probes, sanitized runs)
ptl_q38_run() {
  local log=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./qwen38 "$@" > "$log" 2>&1 || true
  rm -f chain.usage
}
# ptl_q38_cap <tag> <k> <env...> -- <argv...>: COLI_VK_DEVICE_CAP_MB from a probe's fit
# line under which exactly k layers fit; the run's fit gives k by the rule, the final N is
# k, the CPU's tokens (logits within 1e-4 of the largest unless PTL_Q38_TOL=0), and coli
# plan's prediction the same as the engine's. The probe runs under a 256 MiB cap: up to
# there the pools take 64 KiB blocks, so what the probe held at the fit is what the run
# holds; a larger probe cap takes larger blocks, and on these layers of tens of KiB that
# difference alone decides N.
ptl_q38_cap() {
  local tag=$1 k=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local cap; cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap qwen38 probe.log "$k") || { grep -a '^\[VK\] qwen38 chain' probe.log; fail "$tag: no cap"; }
  ptl_q38_gate "$tag" "$k" "${PTL_Q38_TOL:-1}" "${envs[@]}" COLI_VK_DEVICE_CAP_MB="$cap" -- "$@"
  local p; p=$(ptl_calc predict qwen38 vk.log)
  [ "$p" = "$k" ] || { grep -a '^\[VK\] qwen38 chain' vk.log; fail "$tag: the fit line gives $p layers, not $k"; }
  echo "   $tag: COLI_VK_DEVICE_CAP_MB=$cap, the fit's rule gives $p"
  local snap; snap=$(printf '%s\n' "${envs[@]}" | sed -n 's/^SNAP=//p' | tail -1)
  ptl_q38_plan "$tag" "$snap" vk.log "${envs[@]}" COLI_VK_DEVICE_CAP_MB="$cap"
}
# ptl_q38_plan <tag> <fixture> <log> <env...>: coli plan's prediction (resource_plan.py,
# vk_chain_fit with qwen38's layout) for the same device and settings: the engine's free
# bytes, per-layer bytes, fixed bytes, tail and N, from the checkpoint's header and config
ptl_q38_plan() {
  local tag=$1 fx=$2 log=$3; shift 3
  $PY - "$fx" "$log" "$@" <<'PY' || fail "$tag: coli plan predicts otherwise"
import re, sys
sys.path.insert(0, ".")
import resource_plan as rp
fx, log = sys.argv[1], sys.argv[2]
env = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
env.update(COLI_VULKAN="1", COLI_VK_CHAIN="1")
lines = open(log, errors="replace").read().splitlines()
f = [l for l in lines if l.startswith("[VK] qwen38 chain fit: ")][-1]
num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
n = int(re.findall(r"\[VK\] qwen38 chain: (\d+) of \d+ layers on the device", "\n".join(lines))[0])
info = rp.analyze_model(fx)
fit = rp.vk_chain_fit(info, "qwen38", env, {"type": "cpu"})
tail = rp._VK_CHAIN_LAYOUT["qwen38"](info, env, {"type": "cpu"}).tail
ok = (fit["layers"], fit["fixed"], fit["free"], tail, fit["n"]) == (layers, num("fixed"), num("free"), num("tail"), n)
print(f"   coli plan: N = {fit['n']} of {fit['L']} ({'as the engine' if ok else 'the engine: %d' % n}), "
      f"layers {'= the engine' if fit['layers'] == layers else fit['layers']}, fixed {fit['fixed']} B, "
      f"tail {tail} B, free {fit['free']} B")
sys.exit(0 if ok else 1)
PY
}
# ptl_q38_fault <tag> <k> <probe log> <env...> -- <argv...>: COLI_VK_STAGED_FAULT aimed
# inside layer k's setup (the probe's marks); the run keeps the k layers before it.
ptl_q38_fault() {
  local tag=$1 k=$2 probe=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local n; n=$(ptl_calc fault qwen38 "$probe" "$k") || { grep -a '^\[VK\] qwen38 chain' "$probe"; fail "$tag: no fault point in layer $k"; }
  ptl_q38_gate "$tag" "$k" 1 "${envs[@]}" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:"$n" -- "$@"
  grep -qa "COLI_VK_STAGED_FAULT: submit #$n fails" vk.log || { cat vk.log; fail "$tag: the fault never fired"; }
  if [ "$k" -gt 0 ]; then
    grep -qa "^\[VK\] qwen38 chain: $k of [0-9]* layers on the device .*(layer $k did not reach the device" vk.log ||
      { grep -a '^\[VK\] qwen38 chain' vk.log; fail "$tag: no shrink line"; }
  fi
  echo "   $tag: COLI_VK_STAGED_FAULT=submit:$n"
}

ptl_q38_fixtures() {
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_ple3 --ple-layer 3
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_mtp --mtp
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_tiny_mtp   # serve speaks text
}

ptl_family_qwen38() {
  export OMP_NUM_THREADS=2
  make qwen38 VK=1
  ptl_q38_fixtures
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4 --fp8-experts --int4-experts --expert-gain 3
  local ids; ids=$(pf_ids 110 1 60)
  $PY tools/make_qwen38_tiny.py --out qwen38_long --prompt-ids "$ids"
  kv_qwen38_fixtures
  local T=qwen38_tiny/ref.json M=qwen38_tiny_mtp/ref.json k f
  # 1. N forced, every k, and the formats, the experts and the PLE on either side
  for k in 0 1 2 3 4; do
    ptl_q38_gate "partial qwen38 COLI_VK_CHAIN_LAYERS=$k" $k 1 COLI_VK_CHAIN_LAYERS=$k SNAP=qwen38_tiny -- 4 8 $T
  done
  ptl_q38_gate "partial qwen38 int8 trunk, 2 layers" 2 0 Q38_TRUNK_MIN_KB=0 COLI_VK_CHAIN_LAYERS=2 SNAP=qwen38_tiny -- 4 8 $T
  ptl_q38_gate "partial qwen38 f32 trunk, 3 layers" 3 1 Q38_TRUNK_CPU_INT8=0 Q38_NATIVE_BF16=0 COLI_VK_CHAIN_LAYERS=3 \
    SNAP=qwen38_tiny -- 4 8 $T
  ptl_q38_gate "partial qwen38 int4-g64 experts, 2 layers" 2 1 COLI_VK_CHAIN_LAYERS=2 SNAP=qwen38_tiny_int4 -- 2 8 \
    qwen38_tiny_int4/ref_int4.json
  for k in 1 2 3; do   # the PLE at layer 2: on the CPU with 1 and 2 layers, on the device with 3
    ptl_q38_gate "partial qwen38 the PLE at layer 2, $k layers" $k 1 COLI_VK_CHAIN_LAYERS=$k SNAP=qwen38_tiny_ple3 -- 4 8 \
      qwen38_tiny_ple3/ref.json
  done
  ptl_q38_gate "partial qwen38 prefill batching off, 2 layers" 2 1 Q38_PREFILL_BATCH=0 COLI_VK_CHAIN_LAYERS=2 SNAP=qwen38_tiny -- 4 8 $T
  # the tier off puts the per-matrix path on: the CPU's layers, the head and the MTP head
  # stay on the CPU all the same (the gate checks nothing went up past the setup)
  ptl_q38_gate "partial qwen38 tier off, MTP mixed, 2 layers" 2 1 COLI_VK_TIER=0 Q38_MTP=1 Q38_MTP_FORCE=mixed \
    COLI_VK_CHAIN_LAYERS=2 SNAP=qwen38_tiny_mtp -- 2 8 $M
  grep -qa '^\[VK\] qwen38: dense matrices on the device' vk.log || grep -qa 'device ready, dense matrices on the device' vk.log ||
    { grep -a '^\[VK\] qwen38' vk.log; fail "partial qwen38 tier off: the per-matrix path was not on"; }

  # 2. the fit under a device cap: a probe, then the caps under which 1, 3 and 4 layers fit
  local R=COLI_VK_TIER_RESERVE_GB=0.04
  ptl_q38_run probe.log $R COLI_VK_DEVICE_CAP_MB=256 SNAP=qwen38_tiny -- 4 8 $T
  for k in 1 3 4; do
    ptl_q38_cap "partial qwen38 a device holding $k layers" $k $R SNAP=qwen38_tiny -- 4 8 $T
  done
  # the int8 trunk and the MTP head's tail, in the engine's fit and in coli plan's
  ptl_q38_run probe.log $R COLI_VK_DEVICE_CAP_MB=256 Q38_TRUNK_MIN_KB=0 Q38_MTP=1 SNAP=qwen38_tiny_mtp -- 2 8 $M
  PTL_Q38_TOL=0 ptl_q38_cap "partial qwen38 int8 trunk, MTP, a device holding 2 layers" 2 $R Q38_TRUNK_MIN_KB=0 Q38_MTP=1 \
    SNAP=qwen38_tiny_mtp -- 2 8 $M

  # 3. an upload refused inside layer k's setup: on the full chain's first forward (k = 0
  # turns the chain off), at startup for a partial chain, and in the dense-host pass
  ptl_q38_run probe.log COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=4 SNAP=qwen38_tiny -- 4 8 $T
  for k in 0 1 2; do
    ptl_q38_fault "partial qwen38 upload refused in layer $k (first forward)" $k probe.log COLI_VK_CHAIN_LAYERS=4 \
      SNAP=qwen38_tiny -- 4 8 $T
  done
  ptl_q38_run probe.log COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=3 SNAP=qwen38_tiny -- 4 8 $T
  ptl_q38_fault "partial qwen38 upload refused in layer 1 (startup)" 1 probe.log COLI_VK_CHAIN_LAYERS=3 SNAP=qwen38_tiny -- 4 8 $T
  ptl_q38_run probe.log COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN_LAYERS=4 \
    SNAP=qwen38_tiny -- 4 8 $T
  ptl_q38_fault "partial qwen38 upload refused in layer 2 (dense-host pass)" 2 probe.log COLI_VK_DENSE_HOST=0 \
    COLI_VK_CHAIN_LAYERS=4 SNAP=qwen38_tiny -- 4 8 $T
  [ "$(dho_dropped vk.log)" = 32 ] && [ "$(dho_reloaded vk.log)" = 0 ] ||
    { grep -a '^\[VK\] qwen38' vk.log; fail "partial qwen38 dense-host pass refused: not the 2 layers' 32 host copies dropped"; }

  # 4. the forward paths with 2 of 4 layers: prompt chunks, streaming, drafts, serve, the
  # KV split, prompts only, a lost device
  local P=COLI_VK_CHAIN_LAYERS=2
  ptl_q38_gate "partial qwen38 prompt in chunks of 3" 2 1 COLI_VK_CHAIN_ROWS=3 $P SNAP=qwen38_tiny -- 4 8 $T
  ptl_q38_gate "partial qwen38 long prompt, cold experts streamed" 2 1 $PF_FORCE COLI_VK_TIER_GB=0.00005 COLI_VK_CHAIN_ROWS=40 \
    $P SNAP=qwen38_long -- 4 8 qwen38_long/ref.json
  [ "$(stream_count qwen38 vk.log)" -gt 0 ] || { grep -a '\[VK\] tier' vk.log; fail "partial qwen38 long prompt: nothing streamed"; }
  for f in "" reject accept mixed; do
    ptl_q38_gate "partial qwen38 MTP ${f:-drafting}" 2 1 Q38_MTP=1 Q38_MTP_FORCE=$f $P SNAP=qwen38_tiny_mtp -- 2 8 $M
  done
  ptl_q38_gate "partial qwen38 MTP depth 3 cycle, 3 layers" 3 1 Q38_MTP=1 Q38_MTP_DRAFTS=3 Q38_MTP_FORCE=cycle \
    COLI_VK_CHAIN_LAYERS=3 SNAP=qwen38_tiny_mtp -- 2 8 $M
  ptl_q38_gate "partial qwen38 lookup row4" 2 1 COLI_LOOKUP=1 COLI_SPEC_GATE=0 COLI_LOOKUP_FORCE=row4 $P SNAP=qwen38_tiny_mtp -- 2 8 $M
  local h="$PY tests/spec_drafts_harness.py"
  $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp --quick \
    --env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 --env $P
  local E="qwen38 chain: 2 of 4 layers on the device"
  CHAIN_SERVE_EXPECT="$E" $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 $P
  CHAIN_SERVE_EXPECT="$E" $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp $P
  CHAIN_SERVE_EXPECT="$E" COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 $P
  for k in 2 3; do
    kv_gate qwen38 "partial qwen38 KV split, $k layers" 1 16 4 COLI_VK_CHAIN_LAYERS=$k SNAP=qwen38_kv -- 4 8 qwen38_kv/ref.json
    ptl_check_n qwen38 vk.log $k "partial qwen38 KV split, $k layers"
  done
  kv_gate qwen38 "partial qwen38 KV split, MTP mixed" 1 16 4 Q38_MTP=1 Q38_MTP_FORCE=mixed $P SNAP=qwen38_kv_mtp -- 2 8 \
    qwen38_kv_mtp/ref.json
  ptl_check_n qwen38 vk.log 2 "partial qwen38 KV split, MTP mixed"
  CHAINMODE=2 ptl_q38_gate "partial qwen38 prompts only, MTP mixed" 2 1 Q38_MTP=1 Q38_MTP_FORCE=mixed $P SNAP=qwen38_tiny_mtp -- 2 8 $M
  lost_late qwen38 "partial qwen38 device lost mid-decode" 10 rebuild $P SNAP=qwen38_tiny -- 4 8 $T
  ptl_check_n qwen38 vk.log 2 "partial qwen38 device lost mid-decode"
  lost_late qwen38 "partial qwen38 device lost in a verify" 15 rebuild Q38_MTP=1 Q38_MTP_FORCE=mixed $P SNAP=qwen38_tiny_mtp -- 2 8 $M
  lost_late qwen38 "partial qwen38 device lost, the PLE on the CPU" 10 rebuild COLI_VK_CHAIN_LAYERS=1 SNAP=qwen38_tiny_ple3 -- 4 8 \
    qwen38_tiny_ple3/ref.json

  # 5. the host copies with 2 of 4 layers on the device: theirs only (15 matrices a layer,
  # 2 more with the PLE), none read back on a healthy run, only theirs after a lost device
  dho_gate qwen38 "partial qwen38 dense-only, 2 layers" 1 $P SNAP=qwen38_tiny -- 4 8 $T
  ptl_check_n qwen38 vk.log 2 "partial qwen38 dense-only"
  [ "$(dho_dropped vk.log)" = 32 ] || { grep -a '^\[VK\] qwen38' vk.log; fail "partial qwen38 dense-only: $(dho_dropped vk.log) host copies dropped, not 32"; }
  grep -qa 'dense matrices on the device only, .*(the 2 of 4 layers on the device; the 2 on the CPU keep theirs)' vk.log ||
    { grep -a '^\[VK\] qwen38' vk.log; fail "partial qwen38 dense-only: the placed line does not name the layers"; }
  dho_gate qwen38 "partial qwen38 dense-only, MTP mixed, 3 layers" 1 Q38_MTP=1 Q38_MTP_FORCE=mixed COLI_VK_CHAIN_LAYERS=3 \
    SNAP=qwen38_tiny_mtp -- 2 8 $M
  [ "$(dho_dropped vk.log)" = 47 ] || { grep -a '^\[VK\] qwen38' vk.log; fail "partial qwen38 dense-only MTP: $(dho_dropped vk.log) dropped, not 47"; }
  dho_lost qwen38 "partial qwen38 dense-only, device lost" 10 $P SNAP=qwen38_tiny -- 4 8 $T
  [ "$(dho_reloaded vk.log)" -le 32 ] || { grep -a '^\[VK\] qwen38' vk.log; fail "partial qwen38: a CPU layer's matrix was read back"; }
  $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp --n-new 32 --device-loss \
    --env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 --env COLI_VK_DENSE_HOST=0 --env $P
  CHAIN_SERVE_EXPECT="$E" COLI_VK_DENSE_HOST=0 $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 $P

  # 6. nothing forced on a device where everything fits: every layer and the head, as before
  # (the tier's balance off in these two: their logits must be the same bytes)
  ptl_q38_gate "partial qwen38 everything fits" 4 1 COLI_VK_TIER_BALANCE=0 SNAP=qwen38_tiny -- 4 8 $T
  grep -qa '^\[VK\] qwen38 chain: 4 of 4 layers on the device ([^)]*), 0 on the CPU (free' vk.log ||
    { grep -a '^\[VK\] qwen38 chain' vk.log; fail "partial qwen38 everything fits: not the full chain with its head"; }
  cp vk.f32 fits.f32
  ptl_q38_gate "partial qwen38 every layer forced" 4 1 COLI_VK_TIER_BALANCE=0 COLI_VK_CHAIN_LAYERS=4 SNAP=qwen38_tiny -- 4 8 $T
  cmp -s fits.f32 vk.f32 || fail "partial qwen38: the logits with COLI_VK_CHAIN_LAYERS=4 differ from the fit's full chain"
  rm -f fits.f32
  dho_gate qwen38 "partial qwen38 everything fits, dense-only" 1 SNAP=qwen38_tiny -- 4 8 $T
  [ "$(dho_dropped vk.log)" = 65 ] || { grep -a '^\[VK\] qwen38' vk.log; fail "partial qwen38 everything fits: $(dho_dropped vk.log) dropped, not 65"; }
  rm -f chain.usage chain-serve.usage probe.log
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan: a forced middle N, a cap, an upload refused mid-setup,
# the device lost mid-decode (the host copies dropped, read back from disk), a serve
# session and the speculative harness's quick subset with 2 of 4 layers on the device.
ptl_family_qwen38_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen38 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  ptl_q38_fixtures
  local T=qwen38_tiny/ref.json M=qwen38_tiny_mtp/ref.json P=COLI_VK_CHAIN_LAYERS=2 tag
  ptl_q38_san() {  # <tag> <want N> <env...> -- <argv...>
    local tag=$1 want=$2; shift 2
    local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    ptl_q38_run san.log "${envs[@]}" -- "$@"
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan $tag: sanitizer diagnostic"; fi
    ptl_check_n qwen38 san.log "$want" "asan $tag"
    grep -qa '^C engine' san.log || { cat san.log; fail "asan $tag: no tokens"; }
    echo "OK asan $tag: sanitizers clean, $want of $(ptl_L qwen38 san.log) layers on the device"
  }
  ptl_q38_san "partial qwen38 2 layers, MTP mixed" 2 Q38_MTP=1 Q38_MTP_FORCE=mixed $P SNAP=qwen38_tiny_mtp -- 2 8 $M
  ptl_q38_san "partial qwen38 the PLE on the CPU, chunks of 3" 1 COLI_VK_CHAIN_ROWS=3 COLI_VK_CHAIN_LAYERS=1 \
    SNAP=qwen38_tiny_ple3 -- 4 8 qwen38_tiny_ple3/ref.json
  local R=COLI_VK_TIER_RESERVE_GB=0.04 cap n
  ptl_q38_run probe.log $R COLI_VK_DEVICE_CAP_MB=256 SNAP=qwen38_tiny -- 4 8 $T
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap qwen38 probe.log 1)
  ptl_q38_san "partial qwen38 a device holding 1 layer" 1 $R COLI_VK_DEVICE_CAP_MB="$cap" SNAP=qwen38_tiny -- 4 8 $T
  ptl_q38_run probe.log COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=4 SNAP=qwen38_tiny -- 4 8 $T
  n=$(ptl_calc fault qwen38 probe.log 1)
  ptl_q38_san "partial qwen38 upload refused in layer 1" 1 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:"$n" \
    COLI_VK_CHAIN_LAYERS=4 SNAP=qwen38_tiny -- 4 8 $T
  ptl_check_placed qwen38 san.log "asan partial qwen38 upload refused in layer 1"
  # the device lost 10 frames before the end (a probe counts them), the host copies dropped
  ptl_q38_run probe.log COLI_VK_DENSE_HOST=0 $P SNAP=qwen38_tiny -- 4 8 $T
  n=$(sed -n "s/^\[VK\] qwen38 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p" probe.log | tail -1)
  [ -n "$n" ] && [ "$n" -gt 10 ] || { cat probe.log; fail "asan partial qwen38: the probe reports no frames"; }
  ptl_q38_san "partial qwen38 device lost, host copies dropped" 2 COLI_VK_CHAIN_FAULT=$((n - 10)) COLI_VK_DENSE_HOST=0 $P \
    SNAP=qwen38_tiny -- 4 8 $T
  grep -q "rebuilding the state of [1-9]" san.log || { cat san.log; fail "asan partial qwen38 device lost: no state rebuilt"; }
  [ "$(dho_reloaded san.log)" -gt 0 ] || { cat san.log; fail "asan partial qwen38 device lost: nothing read back"; }
  CHAIN_SERVE_EXPECT="qwen38 chain: 2 of 4 layers on the device" \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 $P > san.log 2>&1 || { cat san.log; fail "asan partial qwen38 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan partial qwen38 serve: sanitizer diagnostic"; fi
  echo "OK asan partial qwen38 serve: $(tail -1 san.log)"
  $PY tests/spec_drafts_harness.py --sanitize --quick --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp \
    --env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 --env $P
  rm -f chain.usage chain-serve.usage probe.log
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
