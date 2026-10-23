# Sourced by tests/vulkan_engines.sh: the partial chain (docs/vulkan.md, "A partial
# chain") of qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B dense, Clef's backbone) and olmoe:
# the first N layers on the device, the rest and the head on the CPU.
#
#   partial-qwen36-olmoe            every case below, on Lavapipe
#   partial-qwen36-olmoe-sanitize   a subset under ASan and UBSan
#
# Per engine: COLI_VK_CHAIN_LAYERS=k for k in {0, 1, about L/2, L-1, L} on every geometry
# (the CPU's tokens, logits within 1e-4 of the largest, the line's N, the matrices on the
# device after setup the N layers' exactly); COLI_VK_DEVICE_CAP_MB set from a probe so
# that exactly k layers fit (the line's N, and the N the rule gives from the fit line);
# an upload failing inside layer k's setup (COLI_VK_STAGED_FAULT): N = k, nothing of
# layer k left on the device; and with N < L the forward paths the chain covers: prompt
# chunks with expert streaming, prompt-lookup verifies (accepted and rejected, against
# the CPU and byte for byte against the same run without drafts), serve sessions with
# pins and prefix reuse, the KV split, prompts only, a lost device mid-decode, and the
# dense weights on the device only (only the N layers' host copies dropped, none read
# back on a healthy run, only theirs after a lost one). Nothing forced: N = L.

# ptlqo_resident <engine> <snapshot> <k>: the matrices the device holds with the chain's
# first k layers and the head on the CPU: qwen36's q/k/v/o or the DeltaNet's qkv/z/out,
# the router, the shared expert's (or a dense model's MLP) three, the DeltaNet's b|a rows
# and the shared expert's gate row; olmoe's q/k/v/o and router
ptlqo_resident() {
  $PY - "$@" <<'PY'
import json, sys
eng, snap, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
if eng == "olmoe":
    print(5 * k); sys.exit(0)
m = json.load(open(snap + "/qwen36_meta.json"))
E, SI = int(m.get("num_experts") or 0), int(m.get("shared_inter") or 0)
# q/k/v/o, or qkv/z/out and the b|a rows: four either way
print(k * (4 + (1 if E else 0) + (3 if SI else 0) + (1 if E and SI else 0)))
PY
}
# ptlqo_exit_resident <engine> <log>: the matrices resident at exit ("[VK] <engine>: N
# matmuls on the GPU (M matrices resident, ...)")
ptlqo_exit_resident() {
  local n; n=$(sed -n "s/^\[VK\] $1: [0-9]* matmuls on the GPU (\([0-9]*\) matrices resident.*/\1/p" "$2" | tail -1)
  echo "${n:--1}"
}
# ptlqo_gate <engine> <tag> <tol 0|1> <ks> <env...> -- <argv...>: the CPU once (CPUENV:
# its own settings), then the chain with COLI_VK_CHAIN_LAYERS=k for each k of ks
# (CHAINMODE: COLI_VK_CHAIN, default 1): the CPU's tokens, logits within 1e-4 of the
# largest (tol 1), N = k, the chain ran (k > 0) or not (k = 0), the placed check, and with
# k < L the per-matrix path put nothing more on the device (the matrices resident at exit
# are the k layers', ptlqo_resident; SNAP from the arguments).
ptlqo_gate() {
  local eng=$1 tag=$2 tol=$3 ks=$4; shift 4
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage cpu.f32
  local rc_cpu=0 k lg
  env "${envs[@]}" ${CPUENV:-} DUMP=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || rc_cpu=$?
  for k in $ks; do
    rm -f chain.usage vk.f32
    env "${envs[@]}" DUMP=vk.f32 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
      COLI_VK_CHAIN_LAYERS=$k ./"$eng" "$@" > vk.log 2>&1 || { [ $rc_cpu != 0 ] || { cat vk.log; fail "$tag k=$k: misses the oracle the CPU passes"; }; }
    rm -f chain.usage
    same_tokens cpu.log vk.log "$tag k=$k"
    ptl_check_n "$eng" vk.log "$k" "$tag k=$k"
    if [ "$k" = 0 ]; then
      [ "$(chain_count "$eng" vk.log)" = 0 ] || { cat vk.log; fail "$tag k=0: the chain ran"; }
    else
      [ "$(chain_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag k=$k: the chain never ran"; }
    fi
    ptl_check_placed "$eng" vk.log "$tag k=$k"
    local L snap want got; L=$(ptl_L "$eng" vk.log)
    if [ "$k" -lt "$L" ]; then
      snap=$(printf '%s\n' "${envs[@]}" | sed -n 's/^SNAP=//p' | tail -1)
      want=$(ptlqo_resident "$eng" "$snap" "$k"); got=$(ptlqo_exit_resident "$eng" vk.log)
      [ "$got" = "$want" ] || { grep '^\[VK\]' vk.log; fail "$tag k=$k: $got matrices on the device at exit, the $k layers hold $want"; }
    fi
    lg=""
    if [ "$tol" = 1 ]; then lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag k=$k: logits"; }; lg=", $lg"; fi
    echo "OK $tag k=$k: tokens = CPU$lg, $k of $(ptl_L "$eng" vk.log) layers on the device, $(chain_count "$eng" vk.log) chain forwards"
  done
}

# ptlqo_plan <engine> <tag> <log> <env...>: coli plan's prediction (resource_plan.py,
# vk_chain_fit with the run's settings, the model its SNAP) against the engine's fit line
# in <log>: the same free bytes, per-layer bytes, fixed bytes and N.
ptlqo_plan() {
  local eng=$1 tag=$2 log=$3; shift 3
  $PY - "$eng" "$log" "$@" <<'PY' || fail "$tag: coli plan predicts otherwise"
import re, sys
sys.path.insert(0, ".")
import resource_plan as rp
eng, log = sys.argv[1], sys.argv[2]
env = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
env.update(COLI_VULKAN="1", COLI_VK_CHAIN="1")
lines = open(log, errors="replace").read().splitlines()
f = [l for l in lines if l.startswith(f"[VK] {eng} chain fit: ")][-1]
num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
n = int(re.findall(rf"\[VK\] {eng} chain: (\d+) of \d+ layers on the device", "\n".join(lines))[0])
info = rp.analyze_model(env["SNAP"])
fit = rp.vk_chain_fit(info, info["resolved_family"].descriptor.id, env, {"type": "cpu"})
ok = (fit["layers"], fit["fixed"], fit["free"], fit["n"]) == (layers, num("fixed"), num("free"), n)
print(f"   coli plan: N = {fit['n']} of {fit['L']} ({'as the engine' if ok else 'the engine: %d' % n}), "
      f"layers {'= the engine' if fit['layers'] == layers else fit['layers']}, fixed {fit['fixed']} B, free {fit['free']} B")
sys.exit(0 if ok else 1)
PY
}
# ptlqo_cap <engine> <tag> <ks> <env...> -- <argv...>: a probe under a 256 MiB cap (at or
# below 256 MiB the pools' blocks are 64 KiB whatever the cap, ptl_calc cap recomputing
# their share), then for each k the COLI_VK_DEVICE_CAP_MB under which exactly k layers fit:
# the fit's own N, the rule's N from its line and coli plan's (ptlqo_plan) are all k, the
# CPU's tokens and logits. COLI_VK_TIER_RESERVE_GB=0.04: the reserve holds the first
# chunk's scratch and the KV mirrors once the tier has taken the rest.
ptlqo_cap() {
  local eng=$1 tag=$2 ks=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_RESERVE_GB=0.04)
  rm -f chain.usage cpu.f32
  env "${envs[@]}" DUMP=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_DEVICE_CAP_MB=256 ./"$eng" "$@" > probe.log 2>&1 || true
  rm -f chain.usage
  [ "$(ptl_n "$eng" probe.log)" = "$(ptl_L "$eng" probe.log)" ] || { grep '^\[VK\]' probe.log; fail "$tag: the probe does not hold every layer"; }
  local k cap p lg
  for k in $ks; do
    cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap "$eng" probe.log "$k") || fail "$tag k=$k: no cap"
    rm -f vk.f32
    env "${envs[@]}" "${vk[@]}" DUMP=vk.f32 COLI_USAGE=chain.usage COLI_VK_DEVICE_CAP_MB=$cap ./"$eng" "$@" > vk.log 2>&1 || true
    rm -f chain.usage
    p=$(ptl_calc predict "$eng" vk.log) || { cat vk.log; fail "$tag k=$k: no fit line"; }
    [ "$p" = "$k" ] || { grep '^\[VK\]' vk.log; fail "$tag k=$k: the rule gives $p from the fit line under $cap MiB"; }
    ptl_check_n "$eng" vk.log "$k" "$tag k=$k"
    same_tokens cpu.log vk.log "$tag k=$k"
    [ "$k" = 0 ] || ptl_check_placed "$eng" vk.log "$tag k=$k"
    lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag k=$k: logits"; }
    echo "OK $tag k=$k: COLI_VK_DEVICE_CAP_MB=$cap holds $k layers (the rule and the fit agree), tokens = CPU, $lg"
    ptlqo_plan "$eng" "$tag k=$k" vk.log "${envs[@]}" "${vk[@]}" COLI_VK_DEVICE_CAP_MB=$cap
  done
}

# ptlqo_fault <engine> <tag> <ks> <env...> -- <argv...>: a probe with every layer forced
# (COLI_VK_CHAIN_LAYERS=L) and staged uploads counting the submit point, then for each k a
# submit failing inside layer k's setup: N = k (k = 0: the chain off), the CPU's tokens,
# and the matrices on the device after setup the k layers' (nothing of layer k).
ptlqo_fault() {
  local eng=$1 tag=$2 ks=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_STAGED=1)
  rm -f chain.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_CHAIN_LAYERS=1000 COLI_VK_STAGED_FAULT=submit:1000000 ./"$eng" "$@" > probe.log 2>&1 || true
  rm -f chain.usage
  local L; L=$(ptl_L "$eng" probe.log)
  local k f
  for k in $ks; do
    f=$(ptl_calc fault "$eng" probe.log "$k") || { grep '^\[VK\]' probe.log; fail "$tag k=$k: no fault point inside layer $k"; }
    env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_CHAIN_LAYERS=$L COLI_VK_STAGED_FAULT=submit:$f ./"$eng" "$@" > vk.log 2>&1 || true
    rm -f chain.usage
    grep -q "COLI_VK_STAGED_FAULT: submit #$f fails" vk.log || { cat vk.log; fail "$tag k=$k: the fault did not fire"; }
    ptl_check_n "$eng" vk.log "$k" "$tag k=$k"
    grep -q "^\[VK\] $eng chain: $k of $L layers .*layer $k did not reach the device" vk.log || { grep '^\[VK\]' vk.log; fail "$tag k=$k: no shrink line"; }
    same_tokens cpu.log vk.log "$tag k=$k"
    ptl_check_placed "$eng" vk.log "$tag k=$k"
    if [ "$k" = 0 ]; then [ "$(chain_count "$eng" vk.log)" = 0 ] || { cat vk.log; fail "$tag k=0: the chain ran"; }
    else [ "$(chain_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag k=$k: the chain never ran"; }; fi
    echo "OK $tag k=$k: submit #$f failed inside layer $k, $k of $L layers on the device, $(ptl_calc matrices "$eng" vk.log | cut -d' ' -f1) B of matrices placed, tokens = CPU"
  done
}

# ptlqo_same <engine> <tag> <setting> <env...> -- <argv...>: nothing forced against
# `setting`, the tier deterministic (COLI_VK_TIER_SYNC=1, COLI_VK_TIER_BALANCE=0): the
# same N and every forward's logits byte for byte
ptlqo_same() {
  local eng=$1 tag=$2 set=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1)
  rm -f chain.usage a.f32 b.f32
  env "${envs[@]}" "${vk[@]}" DUMP=a.f32 COLI_USAGE=chain.usage ./"$eng" "$@" > a.log 2>&1 || true
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" "$set" DUMP=b.f32 COLI_USAGE=chain.usage ./"$eng" "$@" > b.log 2>&1 || true
  rm -f chain.usage
  [ "$(ptl_n "$eng" a.log)" = "$(ptl_n "$eng" b.log)" ] || { grep '^\[VK\]' a.log b.log; fail "$tag: N differs"; }
  [ -s a.f32 ] && cmp -s a.f32 b.f32 || fail "$tag: the logits differ"
  same_tokens a.log b.log "$tag"
  echo "OK $tag: N = $(ptl_n "$eng" a.log) both ways, the same logits byte for byte"
}
# ptlqo_dho <engine> <tag> <k> <want> <env...> -- <argv...>: the dense weights on the device
# only (COLI_VK_DENSE_HOST=0) with k of L layers: the CPU's tokens and logits, `want`
# matrices dropped (the k layers' alone), the placed line naming the k layers, none read
# back; then the device lost mid-decode: the CPU's tokens, and only the k layers'
# matrices read back (at most `want`).
ptlqo_dho() {
  local eng=$1 tag=$2 k=$3 want=$4; shift 4
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN_LAYERS=$k)
  rm -f chain.usage cpu.f32 vk.f32
  env "${envs[@]}" DUMP=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" "${vk[@]}" DUMP=vk.f32 COLI_USAGE=chain.usage ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  same_tokens cpu.log vk.log "$tag"
  ptl_check_n "$eng" vk.log "$k" "$tag"
  local n lg L; n=$(dho_dropped vk.log); L=$(ptl_L "$eng" vk.log)
  [ "$n" = "$want" ] || { grep '^\[VK\]' vk.log; fail "$tag: $n matrices dropped, the $k layers hold $want"; }
  grep -q "dense matrices on the device only, .*(the $k of $L layers on the device; the $((L - k)) on the CPU keep theirs)" vk.log || \
    { grep '^\[VK\]' vk.log; fail "$tag: the placed line does not name the $k layers"; }
  [ "$(dho_reloaded vk.log)" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: a matrix was read back without a lost device"; }
  [ "$(chain_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the chain never ran"; }
  lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }
  echo "OK $tag: tokens = CPU, $lg, $n matrices on the device only (the $k layers'), none read back"
  local f; f=$(sed -n "s/^\[VK\] $eng chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p" vk.log | tail -1)
  [ -n "$f" ] && [ "$f" -gt 6 ] || { cat vk.log; fail "$tag: no frames to count"; }
  env "${envs[@]}" "${vk[@]}" COLI_VK_CHAIN_FAULT=$((f - 6)) COLI_USAGE=chain.usage ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  same_tokens cpu.log vk.log "$tag, device lost"
  local r; r=$(dho_reloaded vk.log)
  { [ "$r" -gt 0 ] && [ "$r" -le "$want" ]; } || { grep '^\[VK\]' vk.log; fail "$tag, device lost: $r matrices read back (the $k layers hold $want)"; }
  echo "OK $tag, device lost: tokens = CPU, $r of the $k layers' $want matrices read back"
}

# qwen36's matrices a layer drops with the dense weights on the device only, for its first k
# layers: q/k/v/o or the DeltaNet's qkv/z/out, the router, the shared expert's three
ptlqo_q36_drops() {  # <snapshot> <k>
  $PY - "$1" "$2" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json")); c = c.get("text_config", c)
k = int(sys.argv[2]); types = c["layer_types"]
per = lambda t: (4 if t == "full_attention" else 3) + (1 if c.get("num_experts", 0) else 0) + (3 if c.get("shared_expert_intermediate_size", 0) else 0)
print(sum(per(t) for t in types[:k]))
PY
}

ptlqo_fixtures() {
  kv_qwen36_fixtures   # qwen36_tiny and the 40+40-token qwen36_kv
  kv_olmoe_fixtures    # olmoe_tiny with its tokenizer and the 40+40-token reference
  $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "tests"); from prefix_serve_harness import ensure_byte_tokenizer; ensure_byte_tokenizer(Path("qwen36_tiny_c"))'   # serve speaks text
  $PY tools/make_qwen36_tiny.py --geometry qwen3-coder-30b --out qwen3_coder_tiny --ref-mode full --emit-ref qwen3_coder_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen3_coder_tiny --out qwen3_coder_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen38_27b_tiny --out qwen38_27b_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/make_qwen36_vl_tiny.py --out qwen38_27b_vl_tiny
  $PY tools/convert_qwen36.py --model qwen38_27b_vl_tiny --out qwen38_27b_vl_tiny_c --ebits 8
  make qwen36-tiny-v128-generate
}

ptl_family_qwen36_olmoe() {
  export OMP_NUM_THREADS=2
  make qwen36 olmoe VK=1
  ptlqo_fixtures
  local R=qwen36_tiny/ref_full.json Q=(COLI_DENSE_I8=0 SNAP=qwen36_tiny_c)
  # ---- qwen36: N forced on every geometry: the hybrid (8 layers, attention at 3 and 7),
  # Qwen3-Coder (6 attention layers, no shared expert), the 27B dense (no routed experts:
  # no host step), int8 rows (fmt 1, f32 activations on the CPU too), int4-g64 rows
  # (fmt 4: the CPU's integer dot rounds the activations, tokens only), an image
  ptlqo_gate qwen36 "partial qwen36 hybrid f32" 1 "0 1 4 7 8" "${Q[@]}" -- 8 8 $R
  ptlqo_gate qwen36 "partial qwen36 Qwen3-Coder" 1 "0 1 3 5 6" COLI_DENSE_I8=0 SNAP=qwen3_coder_tiny_c -- 8 8 qwen3_coder_tiny/ref_full.json
  ptlqo_gate qwen36 "partial qwen36 27B dense" 1 "0 1 4 7 8" COLI_DENSE_I8=0 SNAP=qwen38_27b_tiny_c -- 8 8 qwen38_27b_tiny/ref_full.json
  ptlqo_gate qwen36 "partial qwen36 int8 rows" 1 "4" COLI_DENSE_IDOT=0 SNAP=qwen36_tiny_c -- 8 8 $R
  ptlqo_gate qwen36 "partial qwen36 27B dense int8 rows" 1 "4" COLI_DENSE_IDOT=0 SNAP=qwen38_27b_tiny_c -- 8 8 qwen38_27b_tiny/ref_full.json
  ptlqo_gate qwen36 "partial qwen36 int4-g64 rows" 0 "4" COLI_DENSE_BITS=4 SNAP=qwen36_tiny64_c -- 8 4 qwen36_tiny64/ref_full.json
  ptlqo_gate qwen36 "partial qwen36 int4-g64 experts, cap 1" 1 "4" QWEN_EXPERT_ACT=f32 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- 1 4 qwen36_tiny64/ref_full.json
  ptlqo_gate qwen36 "partial qwen36 vision" 1 "4" COLI_DENSE_I8=0 SNAP=qwen38_27b_vl_tiny_c -- 8 8 qwen38_27b_vl_tiny/ref.json
  # the device's budget: a cap that holds exactly k layers, then a failed upload inside layer k
  ptlqo_cap qwen36 "partial qwen36 cap" "0 1 4 7" "${Q[@]}" -- 8 8 $R
  ptlqo_cap qwen36 "partial qwen36 27B dense cap" "1 7" COLI_DENSE_I8=0 SNAP=qwen38_27b_tiny_c -- 8 8 qwen38_27b_tiny/ref_full.json
  ptlqo_fault qwen36 "partial qwen36 fault" "0 1 4 7" "${Q[@]}" -- 8 8 $R
  # the forward paths with 4 of 8 layers on the device: prompt chunks of 3 rows (and the
  # tiled GEMM from 2), the tier off, prompts only
  ptlqo_gate qwen36 "partial qwen36 prefill in chunks of 3" 1 "4" COLI_VK_CHAIN_ROWS=3 "${Q[@]}" -- 8 8 $R
  ptlqo_gate qwen36 "partial qwen36 tiled GEMM" 1 "4" COLI_VK_GEMM_MIN_S=2 "${Q[@]}" -- 8 8 $R
  ptlqo_gate qwen36 "partial qwen36 tier off" 1 "4" COLI_VK_TIER=0 "${Q[@]}" -- 8 8 $R
  CHAINMODE=2 ptlqo_gate qwen36 "partial qwen36 prompts only" 1 "4" "${Q[@]}" -- 8 8 $R
  # decode on the per-matrix path beside the chain: the 4 layers' tensors on the device,
  # the CPU's layers and the head still on the CPU (the resident check above)
  CHAINMODE=2 ptlqo_gate qwen36 "partial qwen36 prompts only, the per-matrix path beside" 1 "4" COLI_VK_DENSE=1 "${Q[@]}" -- 8 8 $R
  [ "$(vk_count qwen36 vk.log)" -gt 0 ] || { grep '^\[VK\]' vk.log; fail "partial qwen36 prompts only, the per-matrix path beside: no matmul on the device"; }
  ptlqo_gate qwen36 "partial qwen36 beside the per-matrix trunk" 1 "4" COLI_VK_DENSE=1 "${Q[@]}" -- 8 8 $R
  # prompt-lookup verifies (the device's DeltaNet copies roll back its layers, the CPU's
  # snapshots the CPU's): against the CPU with the tier on, and byte for byte against the
  # same run without drafts with the tier off (tests/spec_drafts_harness.py)
  local f
  for f in cycle row2 row3; do
    ptlqo_gate qwen36 "partial qwen36 lookup $f" 1 "4" COLI_DENSE_I8=0 COLI_LOOKUP=1 COLI_SPEC_GATE=0 COLI_LOOKUP_FORCE=$f \
      SNAP=qwen36_tiny_v128_c -- 8 8 qwen36_tiny_v128/ref_full.json
  done
  $PY tests/spec_drafts_harness.py --engine ./qwen36 --kind qwen36 --fixture ./qwen36_tiny_v128_c --ref ./qwen36_tiny_v128/ref_full.json \
    --cap 8 --n-new 64 --quick --env COLI_DENSE_I8=0 --env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 \
    --env COLI_VK_CHAIN_LAYERS=4
  # the KV split across the device's attention layer (the CPU's keeps its rows on the host)
  ptlqo_gate qwen36 "partial qwen36 KV split" 1 "4" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 qwen36_kv/ref_full.json
  [ "$(kv_hostparts qwen36 vk.log)" -gt 0 ] || { cat vk.log; fail "partial qwen36 KV split: no host part"; }
  # the device lost mid-decode: the 4 layers' DeltaNet state rebuilt on the CPU, the rest
  # already the host's
  lost_late qwen36 "partial qwen36 device lost" 20 rebuild COLI_VK_CHAIN_LAYERS=4 "${Q[@]}" -- 8 8 $R
  ptl_check_n qwen36 vk.log 4 "partial qwen36 device lost"
  # serve: pins, prompt-cache extensions, a divergent prompt, logprobs, frame for frame;
  # the prefix-reuse contract; Clef's oracle (its head reads every row the CPU finishes)
  CHAIN_SERVE_EXPECT='qwen36 chain: 4 of 8 layers on the device' $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0 COLI_VK_CHAIN_LAYERS=4
  CHAIN_SERVE_EXPECT='qwen36 chain: 4 of 8 layers on the device' COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0 COLI_VK_CHAIN_LAYERS=4
  QWEN36_TINY=$PWD/qwen36_tiny_c COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=4 COLI_USAGE=$PWD/chain.usage $PY tests/test_qwen36_prefix_serve.py
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=4 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage make clef-tiny-check VK=1
  rm -f chain.usage chain-serve.usage
  # the dense weights on the device only, 4 of 8 layers
  ptlqo_dho qwen36 "partial qwen36 dense-only" 4 "$(ptlqo_q36_drops qwen36_tiny_c 4)" "${Q[@]}" -- 8 8 $R
  # nothing forced, everything fits: the full chain, the head with it, as before
  chain_gate qwen36 "partial qwen36 everything fits" 1 "${Q[@]}" -- 8 8 $R
  ptl_check_n qwen36 vk.log 8 "partial qwen36 everything fits"
  grep -q "qwen36 chain: 8 layers on the device (2 attention), " vk.log || { grep '^\[VK\]' vk.log; fail "partial qwen36 everything fits: the head stayed on the CPU"; }
  ptlqo_same qwen36 "partial qwen36 the fit = every layer asked for" COLI_VK_CHAIN_LAYERS=8 "${Q[@]}" -- 8 8 $R

  # ---- olmoe (4 layers)
  local OR=olmoe_tiny/ref_olmoe.json O=(SNAP=olmoe_tiny_c)
  ptlqo_gate olmoe "partial olmoe" 1 "0 1 2 3 4" "${O[@]}" -- 8 8 $OR
  ptlqo_gate olmoe "partial olmoe cap 1" 1 "2" "${O[@]}" -- 1 8 $OR
  ptlqo_gate olmoe "partial olmoe 4-bit experts" 1 "2" "${O[@]}" -- 8 4 $OR
  ptlqo_gate olmoe "partial olmoe PILOT cap 1" 1 "1 3" PILOT=1 WIDE=2 "${O[@]}" -- 1 8 $OR
  ptlqo_gate olmoe "partial olmoe PILOT=3" 1 "2" PILOT=3 "${O[@]}" -- 8 8 $OR
  ptlqo_cap olmoe "partial olmoe cap" "0 1 2 3" "${O[@]}" -- 8 8 $OR
  ptlqo_fault olmoe "partial olmoe fault" "0 1 2 3" "${O[@]}" -- 8 8 $OR
  ptlqo_gate olmoe "partial olmoe prefill in chunks of 3" 1 "2" COLI_VK_CHAIN_ROWS=3 "${O[@]}" -- 8 8 $OR
  ptlqo_gate olmoe "partial olmoe tiled GEMM" 1 "2" COLI_VK_GEMM_MIN_S=2 "${O[@]}" -- 8 8 $OR
  ptlqo_gate olmoe "partial olmoe tier off" 1 "2" COLI_VK_TIER=0 "${O[@]}" -- 8 8 $OR
  CHAINMODE=2 ptlqo_gate olmoe "partial olmoe prompts only" 1 "2" "${O[@]}" -- 8 8 $OR
  CHAINMODE=2 ptlqo_gate olmoe "partial olmoe prompts only, the per-matrix path beside" 1 "2" COLI_VK_DENSE=1 "${O[@]}" -- 8 8 $OR
  [ "$(vk_count olmoe vk.log)" -gt 0 ] || { grep '^\[VK\]' vk.log; fail "partial olmoe prompts only, the per-matrix path beside: no matmul on the device"; }
  ptlqo_gate olmoe "partial olmoe KV split" 1 "2" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 "${O[@]}" -- 8 8 olmoe_tiny/ref_olmoe_long.json
  [ "$(kv_hostparts olmoe vk.log)" -gt 0 ] || { cat vk.log; fail "partial olmoe KV split: no host part"; }
  lost_late olmoe "partial olmoe device lost" 7 redo COLI_VK_CHAIN_LAYERS=2 "${O[@]}" -- 8 8 $OR
  ptl_check_n olmoe vk.log 2 "partial olmoe device lost"
  CHAIN_SERVE_EXPECT='olmoe chain: 2 of 4 layers on the device' $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c COLI_VK_CHAIN_LAYERS=2
  CHAIN_SERVE_EXPECT='olmoe chain: 2 of 4 layers on the device' $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c PILOT=1 WIDE=2 COLI_VK_CHAIN_LAYERS=2
  CHAIN_SERVE_EXPECT='olmoe chain: 2 of 4 layers on the device' COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c COLI_VK_CHAIN_LAYERS=2
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=2 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_ATTN_BLOCK=0 OLMOE_TINY=olmoe_tiny_c \
    $PY -m unittest tests.test_olmoe_prefix_serve tests.test_olmoe_dashboard_hits tests.test_brio_serve.BrioServe
  ptlqo_dho olmoe "partial olmoe dense-only" 2 10 "${O[@]}" -- 8 8 $OR
  chain_gate olmoe "partial olmoe everything fits" 1 "${O[@]}" -- 8 8 $OR
  ptl_check_n olmoe vk.log 4 "partial olmoe everything fits"
  grep -q "olmoe chain: 4 layers on the device, " vk.log || { grep '^\[VK\]' vk.log; fail "partial olmoe everything fits: the head stayed on the CPU"; }
  ptlqo_same olmoe "partial olmoe the fit = every layer asked for" COLI_VK_CHAIN_LAYERS=4 "${O[@]}" -- 8 8 $OR
  rm -f chain.usage chain-serve.usage
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan: memory safety is the gate (a sanitized build vectorizes
# differently, so tokens are not compared): a forced middle N, a cap, a failed upload in
# a layer's setup, the device lost, a lookup verify, the dense weights on the device only
# and a serve session, per engine; each run's N is the one asked for.
ptl_family_qwen36_olmoe_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 olmoe VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "tests"); from prefix_serve_harness import ensure_byte_tokenizer; ensure_byte_tokenizer(Path("qwen36_tiny_c"))'
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen38_27b_tiny --out qwen38_27b_tiny_c --ebits 8
  make qwen36-tiny-v128-generate
  kv_olmoe_fixtures
  local R=qwen36_tiny/ref_full.json OR=olmoe_tiny/ref_olmoe.json
  psan() {  # <engine> <tag> <N> <env and argv...>: a sanitized run, N layers on the device
    local eng=$1 tag=$2 n=$3; shift 3
    rm -f chain.usage
    env COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
    rm -f chain.usage
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    ptl_check_n "$eng" san.log "$n" "$tag"
    [ "$n" = 0 ] || [ "$(chain_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: the chain never ran"; }
    echo "OK $tag: sanitizers clean, $n of $(ptl_L "$eng" san.log) layers on the device, $(chain_count "$eng" san.log) chain forwards"
  }
  psan qwen36 "asan partial qwen36 4 of 8" 4 COLI_VK_CHAIN_LAYERS=4 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R
  psan qwen36 "asan partial qwen36 dense model, chunks of 3" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_ROWS=3 SNAP=qwen38_27b_tiny_c ./qwen36 8 8 qwen38_27b_tiny/ref_full.json
  psan qwen36 "asan partial qwen36 lookup verifies" 4 COLI_VK_CHAIN_LAYERS=4 COLI_DENSE_I8=0 COLI_LOOKUP=1 COLI_SPEC_GATE=0 \
    COLI_LOOKUP_FORCE=cycle SNAP=qwen36_tiny_v128_c ./qwen36 8 8 qwen36_tiny_v128/ref_full.json
  psan qwen36 "asan partial qwen36 dense-only" 4 COLI_VK_CHAIN_LAYERS=4 COLI_VK_DENSE_HOST=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R
  psan qwen36 "asan partial qwen36 device lost" 4 COLI_VK_CHAIN_LAYERS=4 COLI_VK_CHAIN_FAULT=60 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R
  grep -q "rebuilding the state of [1-9]" san.log || { cat san.log; fail "asan partial qwen36 device lost: no state rebuilt"; }
  local cap f
  psan qwen36 "asan partial qwen36 cap probe" 8 COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap qwen36 san.log 3)
  psan qwen36 "asan partial qwen36 cap $cap MiB" 3 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R
  psan qwen36 "asan partial qwen36 fault probe" 8 COLI_VK_CHAIN_LAYERS=8 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R
  f=$(ptl_calc fault qwen36 san.log 5)
  psan qwen36 "asan partial qwen36 fault in layer 5" 5 COLI_VK_CHAIN_LAYERS=8 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$f COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R
  psan olmoe "asan partial olmoe 2 of 4, PILOT" 2 COLI_VK_CHAIN_LAYERS=2 PILOT=1 WIDE=2 SNAP=olmoe_tiny_c ./olmoe 1 8 $OR
  psan olmoe "asan partial olmoe KV split, chunks of 3" 2 COLI_VK_CHAIN_LAYERS=2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_VK_CHAIN_ROWS=3 \
    SNAP=olmoe_tiny_c ./olmoe 8 8 olmoe_tiny/ref_olmoe_long.json
  psan olmoe "asan partial olmoe dense-only" 1 COLI_VK_CHAIN_LAYERS=1 COLI_VK_DENSE_HOST=0 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  psan olmoe "asan partial olmoe device lost" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_FAULT=20 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  grep -q "the CPU redoes the step from position" san.log || { cat san.log; fail "asan partial olmoe device lost: no step redone"; }
  psan olmoe "asan partial olmoe cap probe" 4 COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap olmoe san.log 1)
  psan olmoe "asan partial olmoe cap $cap MiB" 1 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  psan olmoe "asan partial olmoe fault probe" 4 COLI_VK_CHAIN_LAYERS=4 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  f=$(ptl_calc fault olmoe san.log 2)
  psan olmoe "asan partial olmoe fault in layer 2" 2 COLI_VK_CHAIN_LAYERS=4 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$f SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  local eng snap n
  for eng in qwen36 olmoe; do
    snap=qwen36_tiny_c; n=4; [ $eng = olmoe ] && { snap=olmoe_tiny_c; n=2; }
    CHAIN_SERVE_EXPECT="$eng chain: $n of [0-9]+ layers on the device" \
      $PY tests/vulkan_chain_serve.py ./$eng $snap COLI_DENSE_I8=0 COLI_VK_CHAIN_LAYERS=$n > san.log 2>&1 || { cat san.log; fail "asan partial $eng serve"; }
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan partial $eng serve: sanitizer diagnostic"; fi
    echo "OK asan partial $eng serve: $(tail -1 san.log)"
  done
  rm -f chain.usage chain-serve.usage
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
