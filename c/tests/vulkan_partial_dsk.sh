# Sourced by tests/vulkan_engines.sh: the partial chain (docs/vulkan.md, "A partial chain")
# on DeepSeek V4.1 Flash (deepseek_v41) and Kimi K3 (kimi_k3): the first N layers on the
# device, the CPU the rest and the head (partial-dsk, partial-dsk-sanitize; the template
# is tests/vulkan_partial_deepseek.sh).
#
# Every configuration runs the engine's chain gate (v41_gate: the CPU's tokens and exit,
# every logits row within 1e-4 of the largest; k3c_gate: the CPU's tokens, every logits
# row within 2e-3) with the partial chain's settings on top, then checks the lines (ptl_*
# helpers in tests/vulkan_engines.sh):
#   - COLI_VK_CHAIN_LAYERS=k for every k of the six-layer fixtures: N = k, the matrices on
#     the device after setup are those k layers' (the placed line), and with k < L the
#     per-matrix path put nothing more there (the resident matrices at exit = at setup).
#     V4.1's cuts at 2 and 4 hand the CPU's layers the index list and the candidate mask a
#     chain layer made; Kimi K3's at 1 and 3 cut inside an AttnRes block;
#   - COLI_VK_DEVICE_CAP_MB aimed at k layers from a probe's fit line (and at every layer
#     but the head): the line's N, the rule's from the run's own numbers, and coli plan's
#     (resource_plan.py: free, per-layer and fixed bytes too);
#   - a staged upload failing inside layer k's setup: N = k, nothing of layer k on the
#     device; with the dense weights on the device only, nothing read back from disk;
#   - the forward paths with N < L: prompt chunks, expert streaming, DSpark drafts (V4.1),
#     prompts only, the tier off, the per-matrix path beside, the KV split, a lost device
#     (V4.1: the forward again on the CPU; Kimi K3: the chain layers' KDA state rebuilt),
#     serve sessions with prefix reuse (and Kimi K3's recurrent-state photos), images;
#   - the dense weights on the device only with N < L: only the N layers' matrices give back
#     their host copies, none is read back on a healthy run (after a loss, only theirs);
#   - everything fitting: N = L, the same bits with and without COLI_VK_CHAIN_LAYERS=L.

# ---- deepseek_v41 ----------------------------------------------------------------------
# ptl_v41_resident <log>: the matrices the device holds at setup and at exit, "S X"
ptl_v41_resident() {
  local s x
  s=$(sed -n 's/^\[VK\] deepseek_v41 chain: [0-9]* layers on the device (.*, \([0-9]*\) matrices (.* on the device$/\1/p' "$1" | tail -1)
  x=$(sed -n 's/^\[VK\] deepseek_v41: \([0-9]*\) resident matrices on the device.*/\1/p' "$1" | tail -1)
  echo "${s:-0} ${x:-0}"
}
# ptl_v41_off <tag> <env...> -- <argv...>: N = 0, the chain off: the CPU's tokens and exit,
# the logits within 1e-4, no chain forward, no dense matrix on the device
ptl_v41_off() {
  local tag=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local rc_cpu=0 rc_vk=0
  rm -f chain.usage cpu.f32 vk.f32
  env "${envs[@]}" DUMP=cpu.f32 ./deepseek_v41 "$@" > cpu.txt 2> cpu.log || rc_cpu=$?
  env "${envs[@]}" DUMP=vk.f32 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./deepseek_v41 "$@" > vk.txt 2> vk.log || rc_vk=$?
  [ "$rc_cpu" = "$rc_vk" ] && [ -s cpu.txt ] && cmp -s cpu.txt vk.txt || { cat cpu.txt vk.txt; tail -20 vk.log; fail "$tag: tokens"; }
  ptl_check_n deepseek_v41 vk.log 0 "$tag"
  [ "$(chain_count deepseek_v41 vk.log)" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: the chain ran"; }
  if grep -q '^\[VK\] deepseek_v41 chain: [0-9]* of [0-9]* layers placed' vk.log; then ptl_check_placed deepseek_v41 vk.log "$tag"; fi
  set -- $(ptl_v41_resident vk.log)
  [ "$2" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: $2 dense matrices on the device with the chain off"; }
  local lg; lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }
  echo "OK $tag: N = 0, the chain off, tokens = CPU (exit $rc_cpu), $lg, no dense matrix on the device"
}
# ptl_v41 <tag> <k|-> <env...> -- <argv...>: v41_gate with COLI_VK_CHAIN_LAYERS=k (-: the
# fit decides), then N = PTL_WANT (default k), the placed check, and with N < L nothing
# more on the device at exit than at setup
ptl_v41() {
  local tag=$1 k=$2; shift 2
  local want=${PTL_WANT:-$k} e=()
  [ "$k" != - ] && e=(COLI_VK_CHAIN_LAYERS=$k)
  if [ "$want" = 0 ]; then ptl_v41_off "$tag" "${e[@]}" "$@"; return; fi
  v41_gate "$tag" "${e[@]}" "$@"
  ptl_check_n deepseek_v41 vk.log "$want" "$tag"
  ptl_check_placed deepseek_v41 vk.log "$tag"
  local L; L=$(ptl_L deepseek_v41 vk.log)
  if [ "$want" -lt "$L" ] && [ -z "${FAULT_BACK:-}" ]; then
    set -- $(ptl_v41_resident vk.log)
    [ "$1" = "$2" ] || { grep '^\[VK\]' vk.log; fail "$tag: $1 matrices on the device at setup, $2 at exit: the CPU's layers went up"; }
  fi
  echo "   N = $want of $L, $(ptl_calc matrices deepseek_v41 vk.log | cut -d' ' -f1) B of matrices on the device after setup"
}
# ptl_v41_probe <log> <env...> -- <argv...>: one chained run, its stderr kept
ptl_v41_probe() {
  local log=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./deepseek_v41 "$@" > /dev/null 2> "$log" || true
  rm -f chain.usage
  grep -q '^\[VK\] deepseek_v41 chain fit: ' "$log" || { tail -20 "$log"; fail "probe run: no fit line"; }
}
# ptl_plan <engine> <family> <tag> <snapshot> <log> <env...>: coli plan's prediction
# (resource_plan.py, vk_chain_fit) for the same device and settings: the engine's free
# bytes, per-layer bytes, fixed bytes and N, from the checkpoint's header and config alone
ptl_plan() {
  local eng=$1 fam=$2 tag=$3 snap=$4 log=$5; shift 5
  $PY - "$eng" "$fam" "$snap" "$log" "$@" <<'PY' || fail "$tag: coli plan predicts otherwise"
import re, sys
sys.path.insert(0, ".")
import resource_plan as rp
eng, fam, snap, log = sys.argv[1:5]
env = dict(a.split("=", 1) for a in sys.argv[5:] if "=" in a)
env.update(COLI_VULKAN="1", COLI_VK_CHAIN=env.get("COLI_VK_CHAIN", "1"))
lines = open(log, errors="replace").read().splitlines()
f = [l for l in lines if l.startswith(f"[VK] {eng} chain fit: ")][-1]
num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
n = int(re.findall(rf"\[VK\] {eng} chain: (\d+) of \d+ layers on the device", "\n".join(lines))[0])
fit = rp.vk_chain_fit(rp.analyze_model(snap), fam, env, {"type": "cpu"})
ok = (fit["layers"], fit["fixed"], fit["free"], fit["n"]) == (layers, num("fixed"), num("free"), n)
print(f"   coli plan: N = {fit['n']} of {fit['L']} ({'as the engine' if ok else 'the engine: %d' % n}), "
      f"layers {'= the engine' if fit['layers'] == layers else fit['layers']}, fixed {fit['fixed']} B, free {fit['free']} B")
sys.exit(0 if ok else 1)
PY
}
# ptl_notail <engine> <probe log>: the COLI_VK_DEVICE_CAP_MB under which every layer fits
# but not the tail (ptl_calc cap's arithmetic, half of the tail instead of all of it)
ptl_notail() {
  $PY - "$1" "$2" <<'PY'
import os, re, sys
eng, log = sys.argv[1], sys.argv[2]
f = [l for l in open(log, errors="replace").read().splitlines() if l.startswith(f"[VK] {eng} chain fit: ")][-1]
num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
free, res, tail, engine = num("free"), num("reserve"), num("tail"), int(re.search(r"the engine's (\d+) B", f)[1])
layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
if tail < 2: sys.exit("no tail")
def blk(cap, d):
    b = 64 << 10
    while b < cap // 4096: b <<= 1
    return min(b, d)
pools = lambda cap: blk(cap, 256 << 20) + 3 * blk(cap, 64 << 20) + 4 * (4 << 20)
held = int(float(os.environ["PTL_PROBE_CAP_MB"]) * 1048576) - free
cap = held + free
for _ in range(4):
    cap = held + res + engine + pools(cap) + sum(layers) + tail // 2
print(f"{cap / 1048576:.6f}")
PY
}
# the snapshot of a run's settings (SNAP=...)
ptl_snap() { local a; for a in "$@"; do case $a in SNAP=*) echo "${a#SNAP=}"; return;; esac; done; }
# ptl_v41_cap <tag> <k|notail> <env...> -- <argv...>: COLI_VK_DEVICE_CAP_MB aimed at k layers
# (notail: every layer but not the tail) from a probe at 256 MiB with the same settings;
# the reserve holds the first chunk's scratch and the frames' staging after the tier took
# the rest
ptl_v41_cap() {
  local tag=$1 k=$2; shift 2
  local R=(COLI_VK_TIER_RESERVE_GB=0.04) cap pred envs=() snap
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  snap=$(ptl_snap "${envs[@]}")
  ptl_v41_probe ptl-probe.log COLI_VK_DEVICE_CAP_MB=256 "${R[@]}" "${envs[@]}" -- "$@"
  if [ "$k" = notail ]; then
    cap=$(PTL_PROBE_CAP_MB=256 ptl_notail deepseek_v41 ptl-probe.log) || fail "$tag: no cap from the probe"
    k=$(ptl_L deepseek_v41 ptl-probe.log)
  else
    cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap deepseek_v41 ptl-probe.log "$k") || fail "$tag: no cap from the probe"
  fi
  PTL_WANT=$k ptl_v41 "$tag (cap $cap MiB)" - COLI_VK_DEVICE_CAP_MB="$cap" "${R[@]}" "${envs[@]}" -- "$@"
  pred=$(ptl_calc predict deepseek_v41 vk.log)
  [ "$pred" = "$k" ] || { grep '^\[VK\] deepseek_v41 chain' vk.log; fail "$tag: the rule gives $pred from the run's own numbers"; }
  ptl_plan deepseek_v41 deepseek_v41 "$tag" "$snap" vk.log COLI_VK_DEVICE_CAP_MB="$cap" "${R[@]}" "${envs[@]}"
}
# ptl_v41_fault <tag> <k> <env...> -- <argv...>: a staged upload fails inside layer k's
# setup (every layer asked for): the chain keeps the layers before it
ptl_v41_fault() {
  local tag=$1 k=$2; shift 2
  local envs=() L n
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ptl_v41_probe ptl-probe.log COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=99 "${envs[@]}" -- "$@"
  L=$(ptl_L deepseek_v41 ptl-probe.log)
  n=$(ptl_calc fault deepseek_v41 ptl-probe.log "$k") || { grep '^\[VK\] deepseek_v41 chain' ptl-probe.log; fail "$tag: no fault count from the probe"; }
  PTL_WANT=$k ptl_v41 "$tag (submit #$n)" "$L" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:"$n" "${envs[@]}" -- "$@"
  grep -q "deepseek_v41 chain: $k of $L layers on the device.*layer $k did not reach the device" vk.log ||
    { grep '^\[VK\] deepseek_v41 chain' vk.log; fail "$tag: no line for layer $k's failure"; }
}
# ptl_v41_droppable <snapshot> <k>: the matrices of layers 0..k-1 the device holds alone with
# the dense weights on the device only (v41_dho_bytes' set), from the checkpoint's header
ptl_v41_droppable() {
  $PY - "$1" "$2" <<'PY'
import json, re, struct, sys
with open(sys.argv[1] + "/model.safetensors", "rb") as f:
    h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
pat = re.compile(r"layers\.(\d+)\.(?:attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b|compressor\.(?:wkv|wgate)|indexer\.(?:wk|wq_b|weights_proj))"
                 r"|ffn\.shared_experts\.w[123]|engram\.wkv)\.weight")
print(sum(1 for name in h if (m := pat.fullmatch(name)) and int(m[1]) < int(sys.argv[2])))
PY
}
# ptl_v41_dho <tag> <k> <env...> -- <argv...>: the dense weights on the device only with
# N = k: the k layers' matrices alone give back their host copies (the count from the
# header), none comes back on a healthy run (after a loss: some of theirs, no other)
ptl_v41_dho() {
  local tag=$1 k=$2; shift 2
  local snap; snap=$(ptl_snap "$@")
  ptl_v41 "$tag" "$k" COLI_VK_DENSE_HOST=0 "$@"
  local n want L r
  n=$(dho_dropped vk.log); want=$(ptl_v41_droppable "$snap" "$k"); L=$(ptl_L deepseek_v41 vk.log); r=$(dho_reloaded vk.log)
  [ "$n" = "$want" ] || { grep '^\[VK\]' vk.log; fail "$tag: $n host copies dropped, the $k layers have $want"; }
  if [ "$k" -lt "$L" ]; then
    grep -q "dense matrices on the device only, .* (the $k of $L layers on the device; the $((L - k)) on the CPU keep theirs)" vk.log ||
      { grep '^\[VK\]' vk.log; fail "$tag: the placed line does not say the $k of $L layers"; }
  fi
  if [ -n "${FAULT_BACK:-}" ]; then
    [ "$r" -gt 0 ] && [ "$r" -le "$n" ] || { grep '^\[VK\]' vk.log; fail "$tag: $r matrices read back after the loss ($n on the device only)"; }
  else [ "$r" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: $r matrices read back from disk on a healthy run"; }; fi
  echo "   $n host copies dropped (the $k layers' $want), $r read back"
}
# ptl_v41_same <tag> <a> <b> <env...> -- <argv...>: the chain's logits under settings a and b
# (each one word, X=1 for none) the same bits, the tier deterministic
ptl_v41_same() {
  local tag=$1 a=$2 b=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local D=(COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_TIER_STREAM_ROWS=16 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1})
  rm -f chain.usage same-a.f32 same-b.f32
  env "${envs[@]}" "${D[@]}" $a COLI_USAGE=chain.usage DUMP=same-a.f32 ./deepseek_v41 "$@" > /dev/null 2> same-a.log || true
  rm -f chain.usage
  env "${envs[@]}" "${D[@]}" $b COLI_USAGE=chain.usage DUMP=same-b.f32 ./deepseek_v41 "$@" > /dev/null 2> same-b.log || true
  rm -f chain.usage
  [ -s same-a.f32 ] && cmp -s same-a.f32 same-b.f32 || { grep '^\[VK\] deepseek_v41 chain' same-a.log same-b.log; fail "$tag: the logits differ"; }
  echo "OK $tag: the same bits ($a; $b)"
}

ptl_v41_cases() {
  local k f L=SNAP=dsv41_long T=SNAP=dsv41_tiny
  # ---- N forced: every k of the six layers (2: the index list crosses, 4: the candidate mask)
  for k in 0 1 2 3 4 5 6; do ptl_v41 "partial deepseek_v41 COLI_VK_CHAIN_LAYERS=$k" $k $L -- 8 dsv41_long/ref.json; done
  ptl_v41 "partial deepseek_v41 12-token prompt, 3 of 6 layers" 3 $T -- 8 dsv41_tiny/ref.json
  ptl_v41 "partial deepseek_v41 one cache slot, 2 of 6 layers" 2 $L -- 1 dsv41_long/ref.json
  # ---- everything fits: every layer, the same bits as asking for all of them
  PTL_WANT=6 ptl_v41 "partial deepseek_v41 the fit, everything fitting" - $L -- 8 dsv41_long/ref.json
  ptl_v41_same "partial deepseek_v41 the fit = every layer asked for" X=1 COLI_VK_CHAIN_LAYERS=6 $L -- 8 dsv41_long/ref.json
  # ---- a small device: N from the budget
  for k in 1 3 5 0; do ptl_v41_cap "partial deepseek_v41 a device for $k layers" $k $L -- 8 dsv41_long/ref.json; done
  # every layer and not the head: the per-matrix path (COLI_VK_DENSE) leaves it on the CPU
  ptl_v41_cap "partial deepseek_v41 a device for every layer, not the head" notail $L COLI_VK_DENSE=1 -- 8 dsv41_long/ref.json
  grep -q 'deepseek_v41 chain: 6 of 6 layers on the device (.*), 0 on the CPU, the head and what goes with it' vk.log ||
    { grep '^\[VK\] deepseek_v41 chain' vk.log; fail "partial deepseek_v41 every layer, not the head: no line for the head on the CPU"; }
  # ---- a staged upload failing in the middle of the setup
  for k in 0 2 4; do ptl_v41_fault "partial deepseek_v41 an upload failing in layer $k" $k $L -- 8 dsv41_long/ref.json; done
  # ---- the forward paths with N < L
  ptl_v41 "partial deepseek_v41 prompt in chunks of 3" 2 $L COLI_VK_CHAIN_ROWS=3 -- 8 dsv41_long/ref.json
  ptl_v41 "partial deepseek_v41 prompt in chunks of 7" 4 $L COLI_VK_CHAIN_ROWS=7 -- 2 dsv41_long/ref.json
  ptl_v41 "partial deepseek_v41 cold experts streamed" 3 SNAP=dsv41_vlong $PF_FORCE COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=64 -- 2 dsv41_vlong/ref.json
  pf_check deepseek_v41 vk.log "partial deepseek_v41 cold experts streamed" forced
  for k in 2 4; do for f in 1 3 5; do
    ptl_v41 "partial deepseek_v41 DSpark spec=$f, $k of 6 layers" $k $T V41_DSPARK=1 V41_SPEC_FORCE=$f -- 8 dsv41_tiny/ref.json
  done; done
  CHAINMODE=2 ptl_v41 "partial deepseek_v41 prompts only" 3 $L -- 8 dsv41_long/ref.json
  CHAINMODE=2 ptl_v41 "partial deepseek_v41 prompts only, DSpark spec=2" 2 $T V41_DSPARK=1 V41_SPEC_FORCE=2 -- 8 dsv41_tiny/ref.json
  ptl_v41 "partial deepseek_v41 tier off" 3 $L COLI_VK_TIER=0 -- 8 dsv41_long/ref.json
  ptl_v41 "partial deepseek_v41 beside the per-matrix trunk" 3 $L COLI_VK_DENSE=1 -- 8 dsv41_long/ref.json
  ptl_v41 "partial deepseek_v41 the per-row GEMV" 3 $T COLI_VK_CHAIN_GEMV=0 -- 8 dsv41_tiny/ref.json
  ptl_v41 "partial deepseek_v41 V41_INDEX_OWNER=1" 4 $L V41_INDEX_OWNER=1 -- 8 dsv41_long/ref.json
  kv_v41_gate "partial deepseek_v41 the KV split" 8 2 COLI_VK_CHAIN_LAYERS=3 $L -- 8 dsv41_long/ref.json
  ptl_check_n deepseek_v41 vk.log 3 "partial deepseek_v41 the KV split"
  kv_v41_gate "partial deepseek_v41 the KV split, DSpark spec=3" 16 4 COLI_VK_CHAIN_LAYERS=4 $L V41_DSPARK=1 V41_SPEC_FORCE=3 -- 8 dsv41_long/ref.json
  ptl_check_n deepseek_v41 vk.log 4 "partial deepseek_v41 the KV split, DSpark spec=3"
  FAULT_BACK=5 ptl_v41 "partial deepseek_v41 device lost mid-decode" 3 $L -- 8 dsv41_long/ref.json
  FAULT_BACK=60 ptl_v41 "partial deepseek_v41 device lost in the prompt" 3 $L COLI_VK_CHAIN_ROWS=7 -- 8 dsv41_long/ref.json
  FAULT_BACK=8 ptl_v41 "partial deepseek_v41 device lost between drafts" 2 $T V41_DSPARK=1 V41_SPEC_FORCE=5 -- 8 dsv41_tiny/ref.json
  # ---- the dense weights on the device only
  for k in 1 3 6; do ptl_v41_dho "partial deepseek_v41 device only, $k of 6 layers" $k $L -- 8 dsv41_long/ref.json; done
  ptl_v41_dho "partial deepseek_v41 device only, DSpark spec=3" 2 $T V41_DSPARK=1 V41_SPEC_FORCE=3 -- 8 dsv41_tiny/ref.json
  ptl_v41_dho "partial deepseek_v41 device only, staged" 3 $L COLI_VK_STAGED=1 -- 8 dsv41_long/ref.json
  CHAINMODE=2 ptl_v41_dho "partial deepseek_v41 device only, prompts only" 3 $L -- 8 dsv41_long/ref.json
  FAULT_BACK=5 ptl_v41_dho "partial deepseek_v41 device only, device lost mid-decode" 3 $L -- 8 dsv41_long/ref.json
  ptl_v41_fault "partial deepseek_v41 device only, an upload failing in layer 3" 3 $L COLI_VK_DENSE_HOST=0 -- 8 dsv41_long/ref.json
  [ "$(dho_reloaded vk.log)" = 0 ] || { grep '^\[VK\]' vk.log; fail "partial deepseek_v41 device only, an upload failing: read back from disk"; }
  # the trunk's placement asks with N from the budget (the fit runs there, before any upload)
  ptl_v41_cap "partial deepseek_v41 device only, a device for 3 layers" 3 $L COLI_VK_DENSE_HOST=0 -- 8 dsv41_long/ref.json
  [ "$(dho_dropped vk.log)" = "$(ptl_v41_droppable dsv41_long 3)" ] && [ "$(dho_reloaded vk.log)" = 0 ] ||
    { grep '^\[VK\]' vk.log; fail "partial deepseek_v41 device only, a device for 3 layers: the host copies dropped are not the 3 layers'"; }
  ptl_v41_same "partial deepseek_v41 device only = host copies" COLI_VK_DENSE_HOST=1 COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN_LAYERS=3 $L -- 8 dsv41_long/ref.json
  # ---- serve sessions, prefix reuse and images with N < L
  CHAIN_SERVE_EXPECT='deepseek_v41 chain: 2 of 6 layers on the device' \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 COLI_VK_CHAIN_LAYERS=2
  CHAIN_SERVE_EXPECT='deepseek_v41 chain: 4 of 6 layers on the device' \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0 COLI_VK_CHAIN_LAYERS=4
  CHAIN_SERVE_EXPECT='deepseek_v41 chain: 3 of 6 layers on the device' COLI_VK_CHAIN=2 \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 COLI_VK_CHAIN_ROWS=3 COLI_VK_CHAIN_LAYERS=3
  CHAIN_SERVE_EXPECT='dense matrices on the device only, .*the 3 of 6 layers on the device' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 COLI_VK_CHAIN_LAYERS=3
  $PY tests/vulkan_chain_v41_image.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0 COLI_VK_CHAIN_LAYERS=3
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=3 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage \
    $PY -m unittest tests.test_dsv41_prefix_serve tests.test_dsv41_dspark_serve
  rm -f chain.usage chain-serve.usage chain-image.usage
}

# ---- kimi_k3 ---------------------------------------------------------------------------
PTL_K3_O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0"
ptl_k3_resident() {
  local s x
  s=$(sed -n 's/^\[VK\] kimi_k3 chain: [0-9]* layers on the device (.*, \([0-9]*\) matrices (.* on the device$/\1/p' "$1" | tail -1)
  x=$(sed -n 's/^\[VK\] kimi_k3: \([0-9]*\) resident matrices on the device.*/\1/p' "$1" | tail -1)
  echo "${s:-0} ${x:-0}"
}
# ptl_k3_off <tag> <case> <env...>: N = 0, the chain off: the CPU's tokens, every logits row
# within k3c_close, no chain forward, no dense matrix on the device
ptl_k3_off() {
  local tag=$1 c=$2; shift 2
  rm -f k3c.usage cpu.f32 vk.f32
  env "$@" COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=cpu.f32 ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 \
    2> cpu.log | sed 's/ *TUNE.*//' > cpu.tok
  env "$@" COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=vk.f32 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag $c: tokens"; }
  ptl_check_n kimi_k3 vk.log 0 "$tag"
  [ "$(chain_count kimi_k3 vk.log)" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: the chain ran"; }
  if grep -q '^\[VK\] kimi_k3 chain: [0-9]* of [0-9]* layers placed' vk.log; then ptl_check_placed kimi_k3 vk.log "$tag"; fi
  set -- $(ptl_k3_resident vk.log)
  [ "$2" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: $2 dense matrices on the device with the chain off"; }
  local lg; lg=$(k3c_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag $c: logits"; }
  echo "OK $tag $c: N = 0, the chain off, tokens = CPU, $lg, no dense matrix on the device"
}
# ptl_k3 <tag> <case> <k|-> <env...>: k3c_gate on one oracle case with COLI_VK_CHAIN_LAYERS=k
# (-: the fit decides), then N = PTL_WANT (default k), the placed check, and with N < L
# nothing more on the device at exit than at setup
ptl_k3() {
  local tag=$1 c=$2 k=$3; shift 3
  local want=${PTL_WANT:-$k} e=()
  [ "$k" != - ] && e=(COLI_VK_CHAIN_LAYERS=$k)
  if [ "$want" = 0 ]; then ptl_k3_off "$tag" "$c" "${e[@]}" "$@"; return; fi
  K3C_CASES=$c k3c_gate "$tag" "${e[@]}" "$@"
  ptl_check_n kimi_k3 vk.log "$want" "$tag"
  ptl_check_placed kimi_k3 vk.log "$tag"
  local L; L=$(ptl_L kimi_k3 vk.log)
  if [ "$want" -lt "$L" ] && [ -z "${FAULT_BACK:-}" ]; then
    set -- $(ptl_k3_resident vk.log)
    [ "$1" = "$2" ] || { grep '^\[VK\]' vk.log; fail "$tag: $1 matrices on the device at setup, $2 at exit: the CPU's layers went up"; }
  fi
  echo "   N = $want of $L, $(ptl_calc matrices kimi_k3 vk.log | cut -d' ' -f1) B of matrices on the device after setup"
}
# ptl_k3_probe <log> <case> <env...>: one chained run, its stderr kept
ptl_k3_probe() {
  local log=$1 c=$2; shift 2
  rm -f k3c.usage
  env "$@" COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 > /dev/null 2> "$log" || true
  grep -q '^\[VK\] kimi_k3 chain fit: ' "$log" || { tail -20 "$log"; fail "probe run: no fit line"; }
}
# ptl_k3_cap <tag> <case> <k|notail> <env...>: COLI_VK_DEVICE_CAP_MB aimed at k layers (notail:
# every layer, not the head) from a probe at 256 MiB with the same settings
ptl_k3_cap() {
  local tag=$1 c=$2 k=$3; shift 3
  local R=(COLI_VK_TIER_RESERVE_GB=0.04) cap pred
  ptl_k3_probe ptl-probe.log "$c" COLI_VK_DEVICE_CAP_MB=256 "${R[@]}" "$@"
  if [ "$k" = notail ]; then
    cap=$(PTL_PROBE_CAP_MB=256 ptl_notail kimi_k3 ptl-probe.log) || fail "$tag: no cap from the probe"
    k=$(ptl_L kimi_k3 ptl-probe.log)
  else
    cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap kimi_k3 ptl-probe.log "$k") || fail "$tag: no cap from the probe"
  fi
  PTL_WANT=$k ptl_k3 "$tag (cap $cap MiB)" "$c" - COLI_VK_DEVICE_CAP_MB="$cap" "${R[@]}" "$@"
  pred=$(ptl_calc predict kimi_k3 vk.log)
  [ "$pred" = "$k" ] || { grep '^\[VK\] kimi_k3 chain' vk.log; fail "$tag: the rule gives $pred from the run's own numbers"; }
  ptl_plan kimi_k3 kimi "$tag" kimi_k3_tiny vk.log COLI_VK_DEVICE_CAP_MB="$cap" "${R[@]}" "$@"
}
# ptl_k3_fault <tag> <case> <k> <env...>: a staged upload fails inside layer k's setup
ptl_k3_fault() {
  local tag=$1 c=$2 k=$3; shift 3
  local L n
  ptl_k3_probe ptl-probe.log "$c" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=99 "$@"
  L=$(ptl_L kimi_k3 ptl-probe.log)
  n=$(ptl_calc fault kimi_k3 ptl-probe.log "$k") || { grep '^\[VK\] kimi_k3 chain' ptl-probe.log; fail "$tag: no fault count from the probe"; }
  PTL_WANT=$k ptl_k3 "$tag (submit #$n)" "$c" "$L" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:"$n" "$@"
  grep -q "kimi_k3 chain: $k of $L layers on the device.*layer $k did not reach the device" vk.log ||
    { grep '^\[VK\] kimi_k3 chain' vk.log; fail "$tag: no line for layer $k's failure"; }
}
# ptl_k3_droppable <k>: the matrices of layers 0..k-1 the device holds alone with the
# dense weights on the device only and the chain on: every W (k3_dho_each) and the f32
# routers, decay pairs and beta projections the chain holds (k3_dho_place)
ptl_k3_droppable() {
  $PY - kimi_k3_tiny "$1" <<'PY'
import json, re, struct, sys
with open(sys.argv[1] + "/model.safetensors", "rb") as f:
    h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
pat = re.compile(r"model\.layers\.(\d+)\.(?:self_attn\.(?:[qkvog]_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|f_a_proj|f_b_proj|b_proj)"
                 r"|block_sparse_moe\.(?:gate|routed_expert_down_proj|routed_expert_up_proj|shared_experts\.(?:gate|up|down)_proj)"
                 r"|mlp\.(?:gate|up|down)_proj)\.weight")
print(sum(1 for name in h if (m := pat.fullmatch(name)) and int(m[1]) < int(sys.argv[2])))
PY
}
# ptl_k3_dho <tag> <case> <k> <env...>: the dense weights on the device only with N = k
ptl_k3_dho() {
  local tag=$1 c=$2 k=$3; shift 3
  ptl_k3 "$tag" "$c" "$k" COLI_VK_DENSE_HOST=0 "$@"
  local n want L r
  n=$(dho_dropped vk.log); want=$(ptl_k3_droppable "$k"); L=$(ptl_L kimi_k3 vk.log); r=$(dho_reloaded vk.log)
  # RELOAD_MLA=1 (prompts only: decode runs on the CPU, whose MLA attention reads kv_b by
  # row): each MLA layer on the device reads its kv_b back once, as in the dense-only family
  local reloads=0
  [ "${RELOAD_MLA:-0}" = 1 ] && reloads=$(sed -n 's/^\[VK\] kimi_k3 chain: [0-9]* layers on the device ([0-9]* KDA, \([0-9]*\) MLA.*/\1/p' vk.log | tail -1)
  [ "$k" = "$L" ] && want=$((want + 1))   # the head, with every layer
  [ "$n" = "$want" ] || { grep '^\[VK\]' vk.log; fail "$tag: $n host copies dropped, the $k layers have $want"; }
  if [ "$k" -lt "$L" ]; then
    grep -q "dense matrices on the device only, .* (the $k of $L layers on the device; the $((L - k)) on the CPU keep theirs)" vk.log ||
      { grep '^\[VK\]' vk.log; fail "$tag: the placed line does not say the $k of $L layers"; }
  fi
  if [ -n "${FAULT_BACK:-}" ]; then
    [ "$r" -gt 0 ] && [ "$r" -le "$n" ] || { grep '^\[VK\]' vk.log; fail "$tag: $r matrices read back after the loss ($n on the device only)"; }
  else [ "$r" = "${reloads:-0}" ] || { grep '^\[VK\]' vk.log; fail "$tag: $r matrices read back from disk on a healthy run, expected ${reloads:-0}"; }; fi
  echo "   $n host copies dropped (the $k layers' $want), $r read back"
}
# ptl_k3_same <tag> <case> <a> <b> <env...>: the chain's logits under settings a and b the
# same bits, the tier deterministic
ptl_k3_same() {
  local tag=$1 c=$2 a=$3 b=$4; shift 4
  local D=(COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_TIER_STREAM_ROWS=16 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1})
  rm -f k3c.usage same-a.f32 same-b.f32
  env "$@" "${D[@]}" $a COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=same-a.f32 ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 > /dev/null 2> same-a.log || true
  env "$@" "${D[@]}" $b COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=same-b.f32 ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 > /dev/null 2> same-b.log || true
  [ -s same-a.f32 ] && cmp -s same-a.f32 same-b.f32 || { grep '^\[VK\] kimi_k3 chain' same-a.log same-b.log; fail "$tag: the logits differ"; }
  echo "OK $tag: the same bits ($a; $b)"
}

ptl_k3_cases() {
  local O=$PTL_K3_O k c out frames
  # ---- N forced: every k (1, 3, 5 cut inside an AttnRes block of 2 layers)
  for k in 0 1 2 3 4 5 6; do ptl_k3 "partial kimi_k3 COLI_VK_CHAIN_LAYERS=$k" long $k $O; done
  for c in short chunk; do ptl_k3 "partial kimi_k3 COLI_VK_CHAIN_LAYERS=3" $c 3 $O; done
  # ---- everything fits: every layer and the head, the same bits as asking for all of them
  PTL_WANT=6 ptl_k3 "partial kimi_k3 the fit, everything fitting" long - $O
  ptl_k3_same "partial kimi_k3 the fit = every layer asked for" long X=1 COLI_VK_CHAIN_LAYERS=6 $O
  # ---- a small device: N from the budget
  for k in 1 3 5 0; do ptl_k3_cap "partial kimi_k3 a device for $k layers" long $k $O; done
  # every layer and not the head: the chain hands the CPU its final rows, the CPU the head
  ptl_k3_cap "partial kimi_k3 a device for every layer, not the head" long notail $O
  grep -q 'kimi_k3 chain: 6 of 6 layers on the device (.*), 0 on the CPU, the head and what goes with it' vk.log ||
    { grep '^\[VK\] kimi_k3 chain' vk.log; fail "partial kimi_k3 every layer, not the head: no line for the head on the CPU"; }
  # ---- a staged upload failing in the middle of the setup
  for k in 0 2 4; do ptl_k3_fault "partial kimi_k3 an upload failing in layer $k" long $k $O; done
  # ---- the forward paths with N < L
  ptl_k3 "partial kimi_k3 chain chunks of 3" long 3 $O COLI_VK_CHAIN_ROWS=3
  ptl_k3 "partial kimi_k3 prefill one token at a time" long 2 $O K3_CHUNK=1
  K3C_IDS=$(pf_ids 200 1 300 | tr ',' ' ') ptl_k3 "partial kimi_k3 cold experts streamed" ids 3 $O $PF_FORCE COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=50
  pf_check kimi_k3 vk.log "partial kimi_k3 cold experts streamed" forced
  for k in 1 4; do ptl_k3 "partial kimi_k3 int4 KDA and MoE, int8 MLA and head, $k of 6 layers" long $k K3_BITS=4 K3_IDOT=0 COLI_TEMP=0; done
  ptl_k3 "partial kimi_k3 tier off" long 3 $O COLI_VK_TIER=0
  ptl_k3 "partial kimi_k3 shared experts by COLI_VK_DENSE=1" long 3 $O COLI_VK_DENSE=1
  ptl_k3 "partial kimi_k3 K3_TOPP=0.6" long 3 $O K3_TOPP=0.6
  CHAINMODE=2 ptl_k3 "partial kimi_k3 prompts only" long 3 $O
  K3C_CASES=long kv_k3c_gate "partial kimi_k3 the KV split" 8 2 COLI_VK_CHAIN_LAYERS=3 $O
  ptl_check_n kimi_k3 vk.log 3 "partial kimi_k3 the KV split"
  FAULT_BACK=3 REBUILD=1 ptl_k3 "partial kimi_k3 device lost mid-decode" long 3 $O
  FAULT_BACK=29 ptl_k3 "partial kimi_k3 device lost in the first prompt chunk" long 3 $O K3_CHUNK=32
  FAULT_BACK=26 REBUILD=1 ptl_k3 "partial kimi_k3 device lost in a later prompt chunk" long 3 $O K3_CHUNK=32
  # ---- the dense weights on the device only
  for k in 1 3 6; do ptl_k3_dho "partial kimi_k3 device only, $k of 6 layers" long $k $O; done
  ptl_k3_dho "partial kimi_k3 device only, staged" long 3 $O COLI_VK_STAGED=1
  CHAINMODE=2 RELOAD_MLA=1 ptl_k3_dho "partial kimi_k3 device only, prompts only" long 3 $O
  FAULT_BACK=3 REBUILD=1 ptl_k3_dho "partial kimi_k3 device only, device lost mid-decode" long 3 $O
  # device only: the dense weights go up through the per-matrix placement before the chain
  # (k3_dho_place), not with each chain layer as deepseek_v41's; an upload failing there
  # keeps that matrix's host copy and the chain puts it up itself, so the run keeps every
  # layer: N = L, one host copy fewer dropped, nothing read back, the CPU's tokens
  local all fn L
  ptl_k3_probe ptl-probe.log long COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=99 $O COLI_VK_DENSE_HOST=0
  all=$(dho_dropped ptl-probe.log); L=$(ptl_L kimi_k3 ptl-probe.log)
  fn=$(ptl_calc fault kimi_k3 ptl-probe.log 3) || { grep '^\[VK\] kimi_k3 chain' ptl-probe.log; fail "partial kimi_k3 device only: no fault count from the probe"; }
  PTL_WANT=$L ptl_k3 "partial kimi_k3 device only, an upload failing in layer 3's placement (submit #$fn)" long "$L" \
    COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:"$fn" $O COLI_VK_DENSE_HOST=0
  grep -q "staged upload failed (submit: .*): a matrix stays on the CPU" vk.log && [ "$(dho_dropped vk.log)" = $((all - 1)) ] &&
    [ "$(dho_reloaded vk.log)" = 0 ] ||
    { grep '^\[VK\]' vk.log; fail "partial kimi_k3 device only, an upload failing: not one host copy kept with every layer on the device"; }
  ptl_k3_cap "partial kimi_k3 device only, a device for 3 layers" long 3 $O COLI_VK_DENSE_HOST=0
  [ "$(dho_dropped vk.log)" = "$(ptl_k3_droppable 3)" ] && [ "$(dho_reloaded vk.log)" = 0 ] ||
    { grep -E '^\[VK\]|^\[K3\]' vk.log; fail "partial kimi_k3 device only, a device for 3 layers: the host copies dropped are not the 3 layers'"; }
  grep -q '^\[K3\] dense weights on the device only: ' vk.log ||
    { grep '^\[K3\]' vk.log; fail "partial kimi_k3 device only: the expert cache's plan did not count the dropped copies"; }
  ptl_k3_same "partial kimi_k3 device only = host copies" long COLI_VK_DENSE_HOST=1 COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN_LAYERS=3 $O
  # ---- serve sessions frame for frame with N < L: pins, the prompt cache, prefix reuse,
  # recurrent-state photos taken from the chain layers' device state and restored to it
  local S="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0"
  export CHAIN_SERVE_TOL=2e-2
  # shellcheck disable=SC2086
  {
    CHAIN_SERVE_EXPECT='kimi_k3 chain: 2 of 6 layers on the device' $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_VK_CHAIN_LAYERS=2
    out=$(CHAIN_SERVE_EXPECT='kimi_k3 chain: 3 of 6 layers on the device' \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4 COLI_VK_CHAIN_LAYERS=3); echo "$out"
    CHAIN_SERVE_EXPECT='kimi_k3 chain: 4 of 6 layers on the device' \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve K3_BITS=4 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0 COLI_VK_CHAIN_ROWS=3 COLI_VK_CHAIN_LAYERS=4
    CHAIN_SERVE_EXPECT='kimi_k3 chain: 3 of 6 layers on the device' COLI_VK_CHAIN=2 \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4 COLI_VK_CHAIN_LAYERS=3
    CHAIN_SERVE_EXPECT='dense matrices on the device only, .*the 3 of 6 layers on the device' COLI_VK_DENSE_HOST=0 \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=2 COLI_VK_CHAIN_LAYERS=3
    # a device lost mid-session, 40 frames before the end of the same session without a fault
    frames=$(echo "$out" | sed -n 's/.*kimi_k3 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p')
    [ -n "$frames" ] && [ "$frames" -gt 40 ] || fail "partial kimi_k3 serve: no fault-free session to count frames from"
    CHAIN_SERVE_EXPECT='kimi_k3 chain: the device was lost' \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_FAULT=$((frames - 40))
  }
  unset CHAIN_SERVE_TOL
  # recurrent-state checkpoints and the dashboard's lines with N < L
  COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=3 COLI_VK_TIER_SYNC=1 \
    $PY tests/test_kimi_k3_ckpt.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=3 COLI_VK_TIER_SYNC=1 \
    $PY tests/test_kimi_k3_dashboard.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  rm -f k3c.usage chain-serve.usage
}

ptl_dsk_fixtures() {
  v41_chain_fixtures
  $PY tools/make_dsv41_tiny.py --out dsv41_vlong --emit-ref dsv41_vlong/ref.json --prompt-len 160 --max-new 6 > /dev/null
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
}
ptl_family_dsk() {
  export OMP_NUM_THREADS=2
  make deepseek_v41 kimi_k3 VK=1
  ptl_dsk_fixtures
  ptl_v41_cases
  ptl_k3_cases
  rm -rf dsv41_vlong kimi_k3_serve
  rm -f cpu.f32 vk.f32 cpu.txt vk.txt cpu.tok vk.tok same-a.* same-b.* ptl-probe.log
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan: no diagnostic, the chain ran with the N the line says
# (or handled the loss), the matrices placed the N layers' exactly.
ptl_dsk_san() {  # <engine> <tag> <want N|lost> <env and argv...>
  local eng=$1 tag=$2 want=$3; shift 3
  rm -f chain.usage k3c.usage
  env COLI_USAGE=$PWD/chain.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
  if [ "$want" = lost ]; then
    grep -q "$eng chain: the device was lost" san.log || { cat san.log; fail "$tag: no loss was handled"; }
  else
    grep -q "$eng chain: [1-9][0-9]* forwards" san.log || { cat san.log; fail "$tag: the chain never ran"; }
    ptl_check_n "$eng" san.log "$want" "$tag"
    ptl_check_placed "$eng" san.log "$tag"
  fi
  echo "OK $tag: sanitizers clean, N = $(ptl_n "$eng" san.log), $(chain_count "$eng" san.log) chain forwards"
}
ptl_family_dsk_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make clean >/dev/null 2>&1 || true
  make deepseek_v41 kimi_k3 VK=1 EXTRA_CFLAGS="$SAN"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  ptl_dsk_fixtures
  local cap n frames ids O=$PTL_K3_O
  # ---- deepseek_v41
  local V=(SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json)
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 3 of 6 layers" 3 COLI_VK_CHAIN_LAYERS=3 "${V[@]}"
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 2 of 6 layers, chunks of 3" 2 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_ROWS=3 "${V[@]}"
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 4 of 6 layers, DSpark spec=5" 4 COLI_VK_CHAIN_LAYERS=4 SNAP=dsv41_tiny \
    V41_DSPARK=1 V41_SPEC_FORCE=5 ./deepseek_v41 1 dsv41_tiny/ref.json
  ptl_v41_probe ptl-probe.log COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04 SNAP=dsv41_long -- 8 dsv41_long/ref.json
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap deepseek_v41 ptl-probe.log 2)
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 a device for 2 layers" 2 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 "${V[@]}"
  ptl_v41_probe ptl-probe.log COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=6 SNAP=dsv41_long -- 8 dsv41_long/ref.json
  n=$(ptl_calc fault deepseek_v41 ptl-probe.log 3)
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 an upload failing in layer 3" 3 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$n COLI_VK_CHAIN_LAYERS=6 "${V[@]}"
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 device only, an upload failing in layer 3" 3 COLI_VK_DENSE_HOST=0 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$n COLI_VK_CHAIN_LAYERS=6 "${V[@]}"
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 device only, 3 of 6 layers" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_DENSE_HOST=0 "${V[@]}"
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 the KV split" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2 "${V[@]}"
  ptl_dsk_san deepseek_v41 "asan partial deepseek_v41 device lost" lost COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_FAULT=30 "${V[@]}"
  CHAIN_SERVE_EXPECT='deepseek_v41 chain: 2 of 6 layers on the device' \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 COLI_VK_CHAIN_LAYERS=2 > san.log 2>&1 || { cat san.log; fail "asan partial deepseek_v41 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan partial deepseek_v41 serve: sanitizer diagnostic"; fi
  echo "OK asan partial deepseek_v41 serve: $(tail -1 san.log | cut -c1-120)"
  # ---- kimi_k3
  ids=$(k3c_ids long)
  local K=(./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8)
  # shellcheck disable=SC2086
  {
    ptl_dsk_san kimi_k3 "asan partial kimi_k3 3 of 6 layers" 3 $O COLI_VK_CHAIN_LAYERS=3 "${K[@]}"
    frames=$(sed -n 's/^\[VK\] kimi_k3 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' san.log | tail -1)
    ptl_dsk_san kimi_k3 "asan partial kimi_k3 1 of 6 layers, int4 trunk, chunks of 3" 1 K3_BITS=4 K3_IDOT=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_ROWS=3 "${K[@]}"
    ptl_k3_probe ptl-probe.log long COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04 $O
    cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap kimi_k3 ptl-probe.log 2)
    ptl_dsk_san kimi_k3 "asan partial kimi_k3 a device for 2 layers" 2 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 $O "${K[@]}"
    ptl_k3_probe ptl-probe.log long COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=6 $O
    n=$(ptl_calc fault kimi_k3 ptl-probe.log 3)
    ptl_dsk_san kimi_k3 "asan partial kimi_k3 an upload failing in layer 3" 3 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$n COLI_VK_CHAIN_LAYERS=6 $O "${K[@]}"
    ptl_dsk_san kimi_k3 "asan partial kimi_k3 device only, 3 of 6 layers" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_DENSE_HOST=0 $O "${K[@]}"
    ptl_dsk_san kimi_k3 "asan partial kimi_k3 the KV split" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2 $O "${K[@]}"
    # the device lost mid-decode: 2 frames before the end of the run above, the chain
    # layers' KDA state rebuilt on the CPU
    ptl_dsk_san kimi_k3 "asan partial kimi_k3 device lost, rebuilt" lost COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_FAULT=$((frames - 2)) $O "${K[@]}"
    grep -q "rebuilding the state of [1-9]" san.log || { cat san.log; fail "asan partial kimi_k3 device lost: no state was rebuilt"; }
  }
  CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT='kimi_k3 chain: 2 of 6 layers on the device' $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve \
    K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0 COLI_K3_CKPT=4 COLI_VK_CHAIN_LAYERS=2 > san.log 2>&1 ||
    { cat san.log; fail "asan partial kimi_k3 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan partial kimi_k3 serve: sanitizer diagnostic"; fi
  echo "OK asan partial kimi_k3 serve: $(tail -1 san.log | cut -c1-120)"
  rm -rf dsv41_vlong kimi_k3_serve
  rm -f san.log ptl-probe.log chain.usage k3c.usage chain-serve.usage
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
