# Sourced by tests/vulkan_engines.sh: inkling's and MiMo's partial dense chain (the first N
# layers on the device, the others and the head on the CPU; docs/vulkan.md, "A partial
# chain"), run as `bash tests/vulkan_engines.sh partial-inkling-mimo` (and `...-sanitize`).
# Every configuration against the CPU's tokens (and logits within 1e-4 of the largest, as
# the chain families hold them), with the engine's lines checked: N by COLI_VK_CHAIN_LAYERS,
# by a device cap (COLI_VK_DEVICE_CAP_MB) the test computes from the fit line, and by an
# upload failing inside a layer (COLI_VK_STAGED_FAULT); the matrices on the device after
# setup the N layers' (ptl_check_placed). Then the chain's forward paths with N < L, the
# dense weights on the device only for the N layers, and the full chain unforced.
#
# The cap's probe runs under a 256 MiB cap: below 1 GiB the pools take 64 KiB blocks
# (coli_vk_block_bytes), so the probe's fixed bytes are the capped run's and ptl_calc cap
# lands where the tiny fixtures' layers (a few hundred KiB) say.

# ---- inkling: the tiny model (8 layers: 2 dense, 6 MoE, a global layer at 5), its dense
# forms (f32, the dense-int4g64 container, bf16), its expert containers, and the model at
# D = 6144 (2 layers)
ptl_ink_fixtures() {
  kv_inkling_fixtures
  mkdir -p tiny_inkling_bf16
  cp tiny_inkling/config.json tiny_inkling/generation_config.json tiny_inkling_bf16/
  $PY - <<'EOF'
import json, struct, numpy as np
src, dst = "tiny_inkling/model.safetensors", "tiny_inkling_bf16/model.safetensors"
raw = open(src, "rb").read(); n = struct.unpack("<Q", raw[:8])[0]; hdr = json.loads(raw[8:8 + n])
out, blobs, off = {}, [], 0
for k, v in hdr.items():
    if k == "__metadata__": out[k] = v; continue
    a = np.frombuffer(raw[8 + n + v["data_offsets"][0]: 8 + n + v["data_offsets"][1]], np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    out[k] = {"dtype": "BF16", "shape": v["shape"], "data_offsets": [off, off + len(b)]}
    blobs.append(b); off += len(b)
h = json.dumps(out).encode(); h += b" " * (-len(h) % 8)
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h); [f.write(b) for b in blobs]
EOF
}

# ptl_ink <tag> <k> <env...> -- <argv...>: inkling with COLI_VK_CHAIN_LAYERS=k ("-": unset,
# PTL_N the N expected): chain_gate's gates (the CPU's tokens, every forward's logits within
# 1e-4 of the largest), the final N, and the matrices on the device after setup the N
# layers'. k = 0: the chain stays off, nothing goes up (the per-matrix path neither), the
# CPU's tokens. CHAINMODE: COLI_VK_CHAIN's value.
ptl_ink() {
  local tag=$1 k=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ "$k" = - ] || envs+=(COLI_VK_CHAIN_LAYERS=$k)
  if [ "$k" = 0 ]; then
    rm -f chain.usage
    env "${envs[@]}" ./inkling "$@" > cpu.log 2>&1 || true
    env "${envs[@]}" COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} ./inkling "$@" > vk.log 2>&1 || true
    same_tokens cpu.log vk.log "$tag"
    [ "$(chain_count inkling vk.log)" = 0 ] || { cat vk.log; fail "$tag: the chain ran with N = 0"; }
    [ "$(vk_count inkling vk.log)" = 0 ] || { cat vk.log; fail "$tag: the per-matrix path ran with N = 0"; }
    echo "OK $tag: tokens = CPU, the chain off, nothing on the device"
  else
    chain_gate inkling "$tag" 1 "${envs[@]}" -- "$@"
  fi
  ptl_check_n inkling vk.log "${PTL_N:-$k}" "$tag"
  ptl_check_placed inkling vk.log "$tag"
  echo "   $tag: $(grep -a '^\[VK\] inkling chain: [0-9]* of [0-9]* layers on the device' vk.log | tail -1 | sed 's/^\[VK\] inkling chain: //')"
}
# ptl_ink_bf16 <tag> <k> <env...> -- <argv...>: the bf16 snapshot. Where this CPU's bf16 dot
# rounds the activations (AVX512-BF16) no matrix goes to the device in bf16: no fit, the
# chain declines as before ("bf16 stays on the CPU"), the CPU's tokens; else ptl_ink.
ptl_ink_bf16() {
  local tag=$1 k=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" COLI_VK_CHAIN_LAYERS=$k COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    ./inkling "$@" > vk.log 2>&1 || true
  if grep -qa 'bf16 stays on the CPU' vk.log; then
    env "${envs[@]}" ./inkling "$@" > cpu.log 2>&1 || true
    same_tokens cpu.log vk.log "$tag"
    if grep -qa '^\[VK\] inkling chain fit:' vk.log; then cat vk.log; fail "$tag: a fit for matrices the device does not take"; fi
    echo "OK $tag: tokens = CPU, the chain declined without a fit (this CPU's bf16 dot keeps bf16 on the CPU)"
  else
    ptl_ink "$tag" "$k" "${envs[@]}" -- "$@"
  fi
}
# ptl_imk_cap <engine> <tag> <probe log> <k> <run...>: COLI_VK_DEVICE_CAP_MB from the probe's fit
# line (ptl_calc cap, PTL_PROBE_CAP_MB=256) for exactly k layers; <run> is a function the
# cap goes to in PTL_CAP, writing vk.log: N = k by the line, and by vkc_fit's rule from the
# run's own fit line
ptl_imk_cap() {
  local eng=$1 tag=$2 probe=$3 k=$4; shift 4
  local cap; cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap "$eng" "$probe" "$k") || { cat "$probe"; fail "$tag: no cap from the probe"; }
  PTL_CAP=$cap "$@"
  PTL_CAP_LAST=$cap
  local p; p=$(ptl_calc predict "$eng" vk.log) || { cat vk.log; fail "$tag: no fit line"; }
  [ "$p" = "$k" ] || { grep -a "^\[VK\] $eng chain" vk.log; fail "$tag: under COLI_VK_DEVICE_CAP_MB=$cap the rule says $p layers, not $k"; }
  ptl_check_n "$eng" vk.log "$k" "$tag"
  ptl_check_placed "$eng" vk.log "$tag"
  echo "   $tag: COLI_VK_DEVICE_CAP_MB=$cap: $(grep -a "^\[VK\] $eng chain: [0-9]* of [0-9]* layers on the device" vk.log | tail -1 | sed "s/^\[VK\] $eng chain: //")"
}
# ptl_imk_head_fault <engine> <probe log>: the COLI_VK_STAGED_FAULT=submit:n n of the head's
# upload, the first time the point is reached after the last layer's setup (the probe's
# placed line, as for ptl_calc fault; one weight block holds the fixture, so no zero fill)
ptl_imk_head_fault() {
  sed -n "s/^\[VK\] $1 chain: [0-9]* of [0-9]* layers placed:.*reached \([0-9,]*\) times.*/\1/p" "$2" | tail -1 |
    awk -F, '{ print $NF + 1 }'
}
# The reserve the capped runs keep for a chunk's scratch and the frames' staging (the tier
# takes the rest of the device): 0.04 GiB.
PTL_RESERVE=0.04
# ptl_imk_plan <engine> <tag> <fixture> <log> <env...>: coli plan's prediction (resource_plan.py,
# vk_chain_fit) for the same device and settings, from the checkpoint's header and config
# alone: the engine's free bytes, per-layer bytes, fixed bytes and N. The tiny inkling's
# config says model_type inkling_text, which the family registry does not take: the plan
# reads a copy that says inkling (the real checkpoint's), the tensors linked. The plan
# counts a buffer's alignment as 256 bytes, Lavapipe's: a driver that aligns them wider
# (Dozen: 64 KiB) gives a tiny fixture's layers other bytes, so only Lavapipe compares.
ptl_imk_plan() {
  local eng=$1 tag=$2 fx=$3 log=$4; shift 4
  case "${VK_ICD_FILENAMES:-}" in
    *lvp_icd*) ;;
    *) echo "   coli plan: not compared on this driver (its buffers' alignment is not the plan's 256 bytes)"; return 0 ;;
  esac
  $PY - "$eng" "$fx" "$log" "$@" <<'PY' || fail "$tag: coli plan predicts otherwise"
import json, os, re, shutil, sys, tempfile
sys.path.insert(0, ".")
import resource_plan as rp
eng, fx, log = sys.argv[1], sys.argv[2], sys.argv[3]
env = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
env.update(COLI_VULKAN="1", COLI_VK_CHAIN="1")
lines = open(log, errors="replace").read().splitlines()
f = [l for l in lines if l.startswith(f"[VK] {eng} chain fit: ")][-1]
num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
n = int(re.findall(rf"\[VK\] {eng} chain: (\d+) of \d+ layers on the device", "\n".join(lines))[0])
with tempfile.TemporaryDirectory() as tmp:
    model = fx
    config = json.load(open(os.path.join(fx, "config.json")))
    if config.get("model_type") == "inkling_text":
        model = os.path.join(tmp, "model")
        os.mkdir(model)
        json.dump(dict(config, model_type="inkling"), open(os.path.join(model, "config.json"), "w"))
        for name in os.listdir(fx):
            if name != "config.json":
                os.symlink(os.path.abspath(os.path.join(fx, name)), os.path.join(model, name))
    fit = rp.vk_chain_fit(rp.analyze_model(model), eng, env, {"type": "cpu"})
ok = fit is not None and (fit["layers"], fit["fixed"], fit["free"], fit["n"]) == (layers, num("fixed"), num("free"), n)
if fit is None:
    sys.exit(f"   coli plan: no fit for {eng}")
print(f"   coli plan: N = {fit['n']} of {fit['L']} ({'as the engine' if ok else 'the engine: %d' % n}), "
      f"layers {'= the engine' if fit['layers'] == layers else fit['layers']}, fixed {fit['fixed']} B, free {fit['free']} B")
sys.exit(0 if ok else 1)
PY
}

ptl_inkling() {
  ptl_ink_fixtures
  local R=tiny_inkling/ref_inkling.json W=tiny_inkling_wide/ref_inkling.json L=tiny_inkling/ref_long.json k
  # ---- COLI_VK_CHAIN_LAYERS: every N from 0 to L on the f32 model; the dense-int4g64
  # container, the int4 experts, runtime int8 experts and bf16 at a middle N; D = 6144
  for k in 0 1 4 7 8; do ptl_ink "partial inkling N=$k" $k SNAP=tiny_inkling -- 8 0 $R; done
  ptl_ink "partial inkling N=3, the dense-int4g64 container" 3 SNAP=tiny_inkling_q -- 8 0 $R
  ptl_ink "partial inkling N=5, int4 experts, cap 1" 5 SNAP=tiny_inkling_x-i4 -- 1 0 $R
  ptl_ink "partial inkling N=2, runtime int8 experts" 2 SNAP=tiny_inkling -- 2 8 $R
  ptl_ink "partial inkling N=4, the trunk shared with the per-matrix path" 4 COLI_VK_DENSE=1 SNAP=tiny_inkling_q -- 8 0 $R
  ptl_ink "partial inkling N=0, the per-matrix path asked for" 0 COLI_VK_DENSE=1 SNAP=tiny_inkling_q -- 8 0 $R
  ptl_ink_bf16 "partial inkling N=4 bf16" 4 SNAP=tiny_inkling_bf16 -- 8 0 $R
  for k in 0 1 2; do ptl_ink "partial inkling D=6144 N=$k" $k SNAP=tiny_inkling_wide -- 8 0 $W; done

  # ---- the device's memory capped so that exactly k layers fit (the probe: the same run
  # under 256 MiB)
  ink_cap_run() {
    rm -f chain.usage
    env COLI_VK_DEVICE_CAP_MB=$PTL_CAP COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 \
      COLI_VULKAN=1 COLI_VK_CHAIN=1 SNAP=tiny_inkling ./inkling 8 0 $R > vk.log 2>&1 || true
    same_tokens cpu.log vk.log "partial inkling cap"
  }
  rm -f chain.usage
  SNAP=tiny_inkling ./inkling 8 0 $R > cpu.log 2>&1 || true
  PTL_CAP=256 ink_cap_run; cp vk.log ptl-probe.log
  ptl_check_n inkling ptl-probe.log 8 "partial inkling cap probe"
  for k in 1 4 7; do
    ptl_imk_cap inkling "partial inkling cap for N=$k" ptl-probe.log $k ink_cap_run
    ptl_imk_plan inkling "partial inkling cap for N=$k" tiny_inkling vk.log COLI_VK_DEVICE_CAP_MB=$PTL_CAP_LAST COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE
    echo "OK partial inkling cap for N=$k: tokens = CPU, coli plan's N the engine's"
  done
  # the dense-int4g64 container's forms, and D = 6144, in the plan
  rm -f chain.usage
  env COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE COLI_VK_CHAIN_LAYERS=0 COLI_USAGE=chain.usage COLI_VULKAN=1 \
    COLI_VK_CHAIN=1 SNAP=tiny_inkling_q ./inkling 8 0 $R > vk.log 2>&1 || true
  ptl_imk_plan inkling "partial inkling plan, the dense-int4g64 container" tiny_inkling_q vk.log COLI_VK_DEVICE_CAP_MB=256 \
    COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE COLI_VK_CHAIN_LAYERS=0
  rm -f chain.usage
  env COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE COLI_USAGE=chain.usage COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    SNAP=tiny_inkling_wide ./inkling 8 0 $W > vk.log 2>&1 || true
  ptl_imk_plan inkling "partial inkling plan, D=6144" tiny_inkling_wide vk.log COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE
  echo "OK partial inkling plan: coli plan's numbers the engine's (the container, D = 6144)"

  # ---- an upload failing inside layer k's setup (COLI_VK_STAGED_FAULT, aimed from a probe's
  # counts): the layers before it on the device, nothing of layer k; the full chain's setup
  # (at the first forward) and the device-only one (at start-up)
  ink_fault() {  # <tag> <k> <probe log> <env...>
    local tag=$1 k=$2 probe=$3 f; shift 3
    f=$(ptl_calc fault inkling "$probe" "$k") || { cat "$probe"; fail "$tag: no fault count for layer $k"; }
    rm -f chain.usage
    env "$@" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$f COLI_VK_CHAIN_LAYERS=8 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 \
      COLI_VULKAN=1 COLI_VK_CHAIN=1 SNAP=tiny_inkling ./inkling 8 0 $R > vk.log 2>&1 || true
    grep -qa "COLI_VK_STAGED_FAULT: submit #$f fails" vk.log || { cat vk.log; fail "$tag: the fault never fired"; }
    same_tokens cpu.log vk.log "$tag"
    ptl_check_n inkling vk.log "$k" "$tag"
    ptl_check_placed inkling vk.log "$tag"
    echo "OK $tag: tokens = CPU, $(grep -a '^\[VK\] inkling chain: [0-9]* of [0-9]* layers on the device' vk.log | tail -1 | sed 's/^\[VK\] inkling chain: //')"
  }
  local h
  for h in 1 0; do
    rm -f chain.usage
    env COLI_VK_DENSE_HOST=$h COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=8 COLI_USAGE=chain.usage \
      COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 SNAP=tiny_inkling ./inkling 8 0 $R > ptl-fprobe.log 2>&1 || true
    for k in 0 3 7; do ink_fault "partial inkling upload failing in layer $k (COLI_VK_DENSE_HOST=$h)" $k ptl-fprobe.log COLI_VK_DENSE_HOST=$h; done
    if [ $h = 1 ]; then   # lm_head's upload failing: every layer on the device, the head on the CPU
      f=$(ptl_imk_head_fault inkling ptl-fprobe.log)
      rm -f chain.usage
      env COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$f COLI_VK_CHAIN_LAYERS=8 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 \
        COLI_VULKAN=1 COLI_VK_CHAIN=1 SNAP=tiny_inkling ./inkling 8 0 $R > vk.log 2>&1 || true
      grep -qa "lm_head did not reach the device; it runs on the CPU" vk.log || { cat vk.log; fail "partial inkling head fault: not the head's"; }
      same_tokens cpu.log vk.log "partial inkling head fault"
      ptl_check_n inkling vk.log 8 "partial inkling head fault"
      ptl_check_placed inkling vk.log "partial inkling head fault"
      echo "OK partial inkling lm_head's upload failing (submit #$f): tokens = CPU, every layer on the device, the head on the CPU"
    fi
    if [ $h = 0 ]; then   # the dropped host copies are the 7 layers' only, none read back
      [ "$(dho_dropped vk.log)" = 56 ] || { grep -a '^\[VK\]' vk.log; fail "partial inkling fault, device only: $(dho_dropped vk.log) matrices dropped, not the 7 layers' 56"; }
      [ "$(dho_reloaded vk.log)" = 0 ] || { grep -a '^\[VK\]' vk.log; fail "partial inkling fault, device only: a matrix read back"; }
    fi
  done

  # ---- the forward paths with N < L: prompt chunks of 3 rows, prompts only, the tiled
  # GEMM, the KV split (N = 6: the global layer 5 on the device), the device lost
  # mid-decode (the device's layers rebuilt, the CPU's keep their state)
  ptl_ink "partial inkling N=4, prompt chunks of 3" 4 COLI_VK_CHAIN_ROWS=3 SNAP=tiny_inkling -- 8 0 $R
  CHAINMODE=2 ptl_ink "partial inkling N=4, prompts only" 4 SNAP=tiny_inkling -- 8 0 $R
  ptl_ink "partial inkling N=4, tiled GEMM" 4 COLI_VK_GEMM_MIN_S=2 SNAP=tiny_inkling -- 8 0 $R
  kv_gate inkling "partial inkling N=6, the KV split" 1 16 4 COLI_VK_CHAIN_LAYERS=6 SNAP=tiny_inkling -- 8 0 $L
  ptl_check_n inkling vk.log 6 "partial inkling N=6, the KV split"
  kv_gate inkling "partial inkling N=6, the KV split, chunks of 3" 1 16 4 COLI_VK_CHAIN_LAYERS=6 COLI_VK_CHAIN_ROWS=3 SNAP=tiny_inkling -- 8 0 $L
  lost_late inkling "partial inkling N=4, device lost in a shared-expert frame" 8 rebuild COLI_VK_CHAIN_LAYERS=4 SNAP=tiny_inkling -- 8 0 $R
  grep -qa "the device's layers; the CPU's have theirs" vk.log || { cat vk.log; fail "partial inkling lost: not a partial rebuild"; }
  lost_late inkling "partial inkling N=6, device lost, the KV split" 8 rebuild COLI_VK_CHAIN_LAYERS=6 COLI_VK_KV_DEVICE_ROWS=16 \
    COLI_VK_KV_BLOCK=4 SNAP=tiny_inkling -- 8 0 $L

  # ---- the dense weights on the device only for the N layers: their host copies dropped
  # (8 tensors a layer on the fixture: q, k, v, r, o and the MLP's or the shared experts'
  # three), the CPU's layers keep theirs and never read one back; a lost device reads back
  # the device's layers' only
  local n
  for k in 2 5; do
    dho_gate inkling "partial inkling N=$k, device only" 2 COLI_VK_CHAIN_LAYERS=$k SNAP=tiny_inkling -- 8 0 $R
    n=$(dho_dropped vk.log)
    [ "$n" = $((8 * k)) ] || { grep -a '^\[VK\]' vk.log; fail "partial inkling N=$k device only: $n matrices dropped, not the $k layers' $((8 * k))"; }
    grep -qa "the $k of 8 layers on the device; the $((8 - k)) on the CPU keep theirs" vk.log ||
      { grep -a '^\[VK\]' vk.log; fail "partial inkling N=$k device only: the placed line does not say the N layers"; }
    ptl_check_n inkling vk.log $k "partial inkling N=$k device only"
  done
  dho_lost inkling "partial inkling N=4, device only, device lost" 8 COLI_VK_CHAIN_LAYERS=4 SNAP=tiny_inkling -- 8 0 $R
  n=$(dho_reloaded vk.log)
  [ "$n" -le 32 ] || { grep -a '^\[VK\]' vk.log; fail "partial inkling device only lost: $n matrices read back, more than the 4 layers' 32"; }
  # the auto cap measures RAM after the N layers' host copies went
  rm -f chain.usage
  env COLI_USAGE=chain.usage COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN_LAYERS=1 SNAP=tiny_inkling_wide \
    ./inkling 0 0 $W > vk.log 2>&1 || true
  grep -q '^\[cap auto\] dense weights on the device only: RAM measured after 0\.[0-9]*[1-9][0-9]* GB of host copies went' vk.log ||
    { grep 'cap auto\|^\[VK\]' vk.log; fail "partial inkling auto cap: not measured after the drop"; }
  echo "OK partial inkling auto cap, N=1 of 2 on the device only: $(grep -o 'RAM measured after.*' vk.log)"

  # ---- serve sessions (pins, prompt-cache extensions, a divergent prompt, logprobs)
  # frame for frame, both modes, and the prefix-reuse and dashboard contracts, with N < L
  $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_CHAIN_LAYERS=4
  COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_CHAIN_LAYERS=6
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_CHAIN_LAYERS=6 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=4 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_ATTN_BLOCK=0 \
    INKLING_TINY=tiny_inkling $PY -m unittest tests.test_inkling_prefix_serve tests.test_inkling_dashboard_hits

  # ---- nothing forced, everything fits: the full chain, as before
  PTL_N=8 ptl_ink "partial inkling unforced" - SNAP=tiny_inkling -- 8 0 $R
  if grep -qa 'COLI_VK_CHAIN_LAYERS=' vk.log; then fail "partial inkling unforced: the line says forced"; fi
  PTL_N=2 ptl_ink "partial inkling unforced, D=6144" - SNAP=tiny_inkling_wide -- 8 0 $W
}

# ---- MiMo-V2.6: the tiny fixture (6 layers: full attention at 0 and 3, sliding windows
# elsewhere, the dense layer 0), its dense forms (MIMO_DENSE_BITS 32, 0: the release's FP8
# and BF16, 8), text and picture
# ptl_mimo <tag> <case> <k> <env...>: mimo_chain_gate (the CPU's text, the teacher-forced
# logits within 1e-4) with COLI_VK_CHAIN_LAYERS=k ("-": unset, PTL_N the N expected), N by
# the line, the placed check. k = 0: the chain off, nothing on the device, the CPU's text.
ptl_mimo() {
  local tag=$1 c=$2 k=$3 x=(); shift 3
  [ "$k" = - ] || set -- "$@" COLI_VK_CHAIN_LAYERS=$k
  if [ "$k" = 0 ]; then
    [ "$c" = image ] && x=(--image mimo_tiny/patches.f32 --grid $MGRID)
    env "$@" COLI_TEMP=0 ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${x[@]}" > mimo-cpu.txt 2>/dev/null
    env "$@" COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} COLI_VK_TIER_SYNC=1 \
      ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${x[@]}" > mimo-vk.txt 2> mimo-vk.err
    [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "$tag $c: tokens differ from the CPU"; }
    [ "$(chain_count mimo mimo-vk.err)" = 0 ] || { cat mimo-vk.err; fail "$tag $c: the chain ran with N = 0"; }
    [ "$(vk_count mimo mimo-vk.err)" = 0 ] || { cat mimo-vk.err; fail "$tag $c: the per-matrix path ran with N = 0"; }
    echo "OK $tag $c: tokens = CPU, the chain off, nothing on the device"
  else
    mimo_chain_gate "$tag" "$c" "$@"
  fi
  ptl_check_n mimo mimo-vk.err "${PTL_N:-$k}" "$tag $c"
  ptl_check_placed mimo mimo-vk.err "$tag $c"
  echo "   $tag $c: $(grep -a '^\[VK\] mimo chain: [0-9]* of [0-9]* layers on the device' mimo-vk.err | tail -1 | sed 's/^\[VK\] mimo chain: //')"
}

ptl_mimo_all() {
  kv_mimo_fixtures
  local k c bits
  # ---- COLI_VK_CHAIN_LAYERS: every N from 0 to L, text and picture; the dense forms at a
  # middle N; one token at a time, blocks of 3, the tower on the per-matrix path (the tail:
  # on the CPU with N < L), the tier off
  for k in 0 1 3 5 6; do for c in long image; do ptl_mimo "partial mimo N=$k" $c $k MIMO_DENSE_BITS=32; done; done
  for bits in 0 8; do ptl_mimo "partial mimo N=3 bits=$bits" long 3 MIMO_DENSE_BITS=$bits; done
  ptl_mimo "partial mimo N=2, one token at a time" window 2 MIMO_DENSE_BITS=32 MIMO_CHUNK=1
  ptl_mimo "partial mimo N=4, blocks of 3" short 4 MIMO_DENSE_BITS=0 MIMO_CHUNK=3
  ptl_mimo "partial mimo N=3, the tower asked on the device" image 3 MIMO_DENSE_BITS=0 COLI_VK_DENSE=1
  ptl_mimo "partial mimo N=0, the per-matrix path asked for" image 0 MIMO_DENSE_BITS=0 COLI_VK_DENSE=1
  ptl_mimo "partial mimo N=3, tier off" long 3 MIMO_DENSE_BITS=0 COLI_VK_TIER=0

  # ---- the device capped so that exactly k layers fit
  local P; P=$(mimo_ids long prompt_ids)
  COLI_TEMP=0 MIMO_DENSE_BITS=32 ./mimo mimo_tiny --ids "$P" --ngen 6 > mimo-cpu.txt 2>/dev/null
  mimo_cap_run() {
    env COLI_VK_DEVICE_CAP_MB=$PTL_CAP COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
      COLI_VK_TIER_SYNC=1 MIMO_DENSE_BITS=32 ./mimo mimo_tiny --ids "$P" --ngen 6 > mimo-vk.txt 2> vk.log
    [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt vk.log; fail "partial mimo cap: tokens differ from the CPU"; }
  }
  PTL_CAP=256 mimo_cap_run; cp vk.log ptl-probe.log
  ptl_check_n mimo ptl-probe.log 6 "partial mimo cap probe"
  for k in 1 3 5; do
    ptl_imk_cap mimo "partial mimo cap for N=$k" ptl-probe.log $k mimo_cap_run
    ptl_imk_plan mimo "partial mimo cap for N=$k" mimo_tiny vk.log COLI_VK_DEVICE_CAP_MB=$PTL_CAP_LAST COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE MIMO_DENSE_BITS=32
    echo "OK partial mimo cap for N=$k: tokens = CPU, coli plan's N the engine's"
  done
  for bits in 0 8; do   # the release's FP8/BF16 and int8 rows, a block of 3, in the plan
    env COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE COLI_VK_CHAIN_LAYERS=0 MIMO_CHUNK=3 COLI_TEMP=0 COLI_VULKAN=1 \
      COLI_VK_CHAIN=1 MIMO_DENSE_BITS=$bits ./mimo mimo_tiny --ids "$P" --ngen 1 > /dev/null 2> vk.log
    ptl_imk_plan mimo "partial mimo plan bits=$bits" mimo_tiny vk.log COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE \
      COLI_VK_CHAIN_LAYERS=0 MIMO_CHUNK=3 MIMO_DENSE_BITS=$bits
  done
  echo "OK partial mimo plan: coli plan's numbers the engine's (MIMO_DENSE_BITS 0 and 8, blocks of 3)"

  # ---- an upload failing inside layer k (the full chain's setup, after the tier; the
  # device-only one, before it)
  mimo_fault() {  # <tag> <k> <probe log> <env...>
    local tag=$1 k=$2 probe=$3 f; shift 3
    f=$(ptl_calc fault mimo "$probe" "$k") || { cat "$probe"; fail "$tag: no fault count for layer $k"; }
    env "$@" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$f COLI_VK_CHAIN_LAYERS=6 COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
      COLI_VK_TIER_SYNC=1 MIMO_DENSE_BITS=32 ./mimo mimo_tiny --ids "$P" --ngen 6 > mimo-vk.txt 2> vk.log
    grep -qa "COLI_VK_STAGED_FAULT: submit #$f fails" vk.log || { cat vk.log; fail "$tag: the fault never fired"; }
    [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt vk.log; fail "$tag: tokens differ from the CPU"; }
    ptl_check_n mimo vk.log "$k" "$tag"
    ptl_check_placed mimo vk.log "$tag"
    echo "OK $tag: tokens = CPU, $(grep -a '^\[VK\] mimo chain: [0-9]* of [0-9]* layers on the device' vk.log | tail -1 | sed 's/^\[VK\] mimo chain: //')"
  }
  local h
  for h in 1 0; do
    env COLI_VK_DENSE_HOST=$h COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=6 COLI_TEMP=0 COLI_VULKAN=1 \
      COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 MIMO_DENSE_BITS=32 ./mimo mimo_tiny --ids "$P" --ngen 6 > /dev/null 2> ptl-fprobe.log
    for k in 0 2 5; do mimo_fault "partial mimo upload failing in layer $k (COLI_VK_DENSE_HOST=$h)" $k ptl-fprobe.log COLI_VK_DENSE_HOST=$h; done
    if [ $h = 1 ]; then   # the head's upload failing: every layer on the device, the head on the CPU
      f=$(ptl_imk_head_fault mimo ptl-fprobe.log)
      env COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$f COLI_VK_CHAIN_LAYERS=6 COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
        COLI_VK_TIER_SYNC=1 MIMO_DENSE_BITS=32 ./mimo mimo_tiny --ids "$P" --ngen 6 > mimo-vk.txt 2> vk.log
      grep -qa "the head did not reach the device; it runs on the CPU" vk.log || { cat vk.log; fail "partial mimo head fault: not the head's"; }
      [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt vk.log; fail "partial mimo head fault: tokens differ from the CPU"; }
      ptl_check_n mimo vk.log 6 "partial mimo head fault"
      ptl_check_placed mimo vk.log "partial mimo head fault"
      echo "OK partial mimo the head's upload failing (submit #$f): tokens = CPU, every layer on the device, the head on the CPU"
    fi
    if [ $h = 0 ]; then   # the dropped host copies are the 5 layers' only (layer 0's five, two a layer after it)
      [ "$(dho_dropped vk.log)" = 13 ] || { grep -a '^\[VK\]' vk.log; fail "partial mimo fault, device only: $(dho_dropped vk.log) matrices dropped, not the 5 layers' 13"; }
      [ "$(dho_reloaded vk.log)" = 0 ] || { grep -a '^\[VK\]' vk.log; fail "partial mimo fault, device only: a matrix read back"; }
    fi
  done

  # ---- the forward paths with N < L: a block in chunks of 3 rows, prompts only, the KV
  # split (the full layer 0 on the device, 3 on the CPU), the device lost mid-decode and
  # inside a chunked prompt (the host's caches hold whole chunks on both sides)
  ptl_mimo "partial mimo N=3, chunks of 3 rows" long 3 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3
  CHAINMODE=2 ptl_mimo "partial mimo N=3, prompts only" long 3 MIMO_DENSE_BITS=32
  mimo_kv_gate "partial mimo N=3, the KV split" long 16 4 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_LAYERS=3
  ptl_check_n mimo mimo-vk.err 3 "partial mimo N=3, the KV split"
  mimo_kv_gate "partial mimo N=4, the KV split, a picture" image 16 4 MIMO_DENSE_BITS=0 COLI_VK_CHAIN_LAYERS=4
  mimo_lost_gate "partial mimo N=3, device lost mid-decode" long 2 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_LAYERS=3
  mimo_lost_gate "partial mimo N=3, device lost inside a chunked prompt" long 30 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3 COLI_VK_CHAIN_LAYERS=3
  mimo_lost_gate "partial mimo N=4, device lost inside a picture's prompt" image 30 MIMO_DENSE_BITS=0 COLI_VK_CHAIN_ROWS=3 COLI_VK_CHAIN_LAYERS=4
  grep -q "the device was lost at position [1-9]" mimo-vk.err || { cat mimo-vk.err; fail "partial mimo picture: not lost after the prompt's first chunk"; }

  # ---- the dense weights on the device only for the N layers (layer 0's five matrices,
  # two a layer after it); a lost device reads back the device's layers' only
  local n want
  for k in 2 4; do
    mimo_dho_gate "partial mimo N=$k, device only" long MIMO_DENSE_BITS=0 COLI_VK_CHAIN_LAYERS=$k
    n=$(dho_dropped mimo-vk0.err); want=$((5 + 2 * (k - 1)))
    [ "$n" = "$want" ] || { grep -a '^\[VK\]' mimo-vk0.err; fail "partial mimo N=$k device only: $n matrices dropped, not the $k layers' $want"; }
    grep -qa "the $k of 6 layers on the device; the $((6 - k)) on the CPU keep theirs" mimo-vk0.err ||
      { grep -a '^\[VK\]' mimo-vk0.err; fail "partial mimo N=$k device only: the placed line does not say the N layers"; }
  done
  mimo_dho_gate "partial mimo N=3, device only, a picture" image MIMO_DENSE_BITS=8 COLI_VK_CHAIN_LAYERS=3
  mimo_dho_lost "partial mimo N=3, device only, device lost" long 2 MIMO_DENSE_BITS=0 COLI_VK_CHAIN_LAYERS=3
  n=$(dho_reloaded mimo-vk.err)
  [ "$n" -le 9 ] || { grep -a '^\[VK\]' mimo-vk.err; fail "partial mimo device only lost: $n matrices read back, more than the 3 layers' 9"; }

  # ---- serve sessions frame for frame within the family's 1e-3 (photos restored, prompt
  # cache extensions, a divergent prompt, logprobs), both modes, a lost device, the split;
  # the prefix-reuse contract and Brio photos against a cold engine
  for k in 1 2; do CHAIN_SERVE_TOL=1e-3 COLI_VK_CHAIN=$k $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_CHAIN_LAYERS=3; done
  CHAIN_SERVE_EXPECT='the device was lost at position' CHAIN_SERVE_TOL=1e-3 \
    $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_CHAIN_LAYERS=4 COLI_VK_CHAIN_FAULT=30
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' CHAIN_SERVE_TOL=1e-3 \
    $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_CHAIN_LAYERS=4 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=3 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=0 COLI_VK_ATTN_BLOCK=0 \
    $PY -m unittest tests.test_mimo_prefix_serve

  # ---- nothing forced, everything fits: the full chain, as before
  for c in long image; do PTL_N=6 ptl_mimo "partial mimo unforced" $c - MIMO_DENSE_BITS=0; done
  if grep -qa 'COLI_VK_CHAIN_LAYERS=' mimo-vk.err; then fail "partial mimo unforced: the line says forced"; fi
}

ptl_family_inkling_mimo() {
  export OMP_NUM_THREADS=2   # tiny models: a wide team only waits for itself (and for Lavapipe's)
  make inkling mimo VK=1
  ptl_inkling
  ptl_mimo_all
  rm -f chain.usage chain-serve.usage ptl-probe.log ptl-fprobe.log
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan: memory safety is the gate (a sanitized build vectorizes
# differently, so tokens are not compared), each run's N as asked: a middle N, a cap, an
# upload failing inside a layer, a lost device, a serve session, for each engine.
ptl_family_inkling_mimo_sanitize() {
  make clean >/dev/null 2>&1 || true
  make inkling mimo VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  ptl_ink_fixtures
  kv_mimo_fixtures
  psan() {  # <engine> <tag> <N> <env and argv...>: no diagnostic, N as the line says
    local eng=$1 tag=$2 k=$3; shift 3
    rm -f chain.usage
    env COLI_USAGE=chain.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan $tag: sanitizer diagnostic"; fi
    ptl_check_n "$eng" san.log "$k" "asan $tag"
    ptl_check_placed "$eng" san.log "asan $tag"
    echo "OK asan $tag: sanitizers clean, N = $k, $(chain_count "$eng" san.log) chain forwards"
  }
  local R=tiny_inkling/ref_inkling.json P; P=$(mimo_ids long prompt_ids)
  psan inkling "partial inkling N=4" 4 COLI_VK_CHAIN_LAYERS=4 SNAP=tiny_inkling ./inkling 8 0 $R
  psan inkling "partial inkling N=3, device only, chunks of 3" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN_ROWS=3 \
    SNAP=tiny_inkling_q ./inkling 8 0 $R
  psan inkling "partial inkling N=1, D=6144" 1 COLI_VK_CHAIN_LAYERS=1 SNAP=tiny_inkling_wide ./inkling 8 0 tiny_inkling_wide/ref_inkling.json
  psan inkling "partial inkling cap probe" 8 COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE SNAP=tiny_inkling ./inkling 8 0 $R
  cp san.log ptl-probe.log
  local cap; cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap inkling ptl-probe.log 5)
  psan inkling "partial inkling cap for N=5" 5 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE SNAP=tiny_inkling ./inkling 8 0 $R
  psan inkling "partial inkling fault probe" 8 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=8 \
    SNAP=tiny_inkling ./inkling 8 0 $R
  cp san.log ptl-fprobe.log
  psan inkling "partial inkling upload failing in layer 4" 4 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$(ptl_calc fault inkling ptl-fprobe.log 4) \
    COLI_VK_CHAIN_LAYERS=8 SNAP=tiny_inkling ./inkling 8 0 $R
  psan inkling "partial inkling N=4, device lost" 4 COLI_VK_CHAIN_LAYERS=4 COLI_VK_CHAIN_FAULT=40 SNAP=tiny_inkling ./inkling 8 0 $R
  grep -qa "rebuilding the state of [1-9]" san.log || { cat san.log; fail "asan partial inkling device lost: no rebuild"; }
  $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_CHAIN_LAYERS=4 > san.log 2>&1 ||
    { cat san.log; fail "asan partial inkling serve"; }
  echo "OK asan partial inkling serve: $(tail -1 san.log)"

  psan mimo "partial mimo N=3" 3 COLI_VK_CHAIN_LAYERS=3 MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 6
  psan mimo "partial mimo N=2, device only, a picture" 2 COLI_VK_CHAIN_LAYERS=2 COLI_VK_DENSE_HOST=0 MIMO_DENSE_BITS=0 COLI_TEMP=0 \
    ./mimo mimo_tiny --ids "$(mimo_ids image prompt_ids)" --ngen 4 --image mimo_tiny/patches.f32 --grid $MGRID
  psan mimo "partial mimo cap probe" 6 COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE MIMO_DENSE_BITS=32 COLI_TEMP=0 \
    ./mimo mimo_tiny --ids "$P" --ngen 6
  cp san.log ptl-probe.log
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap mimo ptl-probe.log 4)
  psan mimo "partial mimo cap for N=4" 4 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=$PTL_RESERVE MIMO_DENSE_BITS=32 COLI_TEMP=0 \
    ./mimo mimo_tiny --ids "$P" --ngen 6
  psan mimo "partial mimo fault probe" 6 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=6 MIMO_DENSE_BITS=32 \
    COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 6
  cp san.log ptl-fprobe.log
  psan mimo "partial mimo upload failing in layer 3" 3 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$(ptl_calc fault mimo ptl-fprobe.log 3) \
    COLI_VK_CHAIN_LAYERS=6 MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 6
  psan mimo "partial mimo N=3, device lost in a chunked prompt" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_ROWS=3 COLI_VK_CHAIN_FAULT=12 \
    MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 6
  grep -qa "the device was lost at position [1-9]" san.log || { cat san.log; fail "asan partial mimo device lost: not lost inside the prompt"; }
  CHAIN_SERVE_TOL=1e-3 $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_CHAIN_LAYERS=3 > san.log 2>&1 ||
    { cat san.log; fail "asan partial mimo serve"; }
  echo "OK asan partial mimo serve: $(tail -1 san.log)"
  rm -f chain.usage chain-serve.usage ptl-probe.log ptl-fprobe.log
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
