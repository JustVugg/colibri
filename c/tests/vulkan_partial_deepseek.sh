# Sourced by tests/vulkan_engines.sh: the partial chain (docs/vulkan.md, "A partial chain")
# on DeepSeek V4 (deepseek_v4): the first N layers on the device, the CPU the rest and the
# head. The template the other engine groups follow (one file each, ptl_family_<group> and
# ptl_family_<group>_sanitize, run as partial-<group> and partial-<group>-sanitize).
#
# Every configuration runs the deepseek-chain family's gate (v4_chain_gate: the CPU's
# ids, the reference's, every logits row within V4's bound, decode rows = teacher-forced
# rows) with the partial chain's settings on top, then checks the lines (ptl_* helpers in
# tests/vulkan_engines.sh):
#   - COLI_VK_CHAIN_LAYERS=k for k in 0, 1, 3, 5, 6 on a six-layer fixture (and the
#     other geometries at one k): N = k, and the matrices on the device after setup are
#     those k layers' (the placed line); with k < L the per-matrix path put nothing more
#     there (the resident matrices at exit = at setup);
#   - COLI_VK_DEVICE_CAP_MB aimed at k layers from a probe's fit line: the rule's N from
#     the run's own numbers, and the line's, are k, and coli plan (resource_plan.py)
#     predicts the same free, per-layer and fixed bytes and the same N;
#   - a staged upload failing inside layer k's setup (COLI_VK_STAGED_FAULT=submit:n, n
#     from a probe's per-layer counts): N = k, nothing of layer k on the device;
#   - the forward paths with N < L: prompt chunks, prefill blocks, expert streaming,
#     n-gram drafts accepted and rejected, the MTP flag, prompts only, the per-matrix
#     path beside, the tier off, the KV split, a lost device mid-decode, in the prompt and
#     between drafts, serve sessions and the prefix-reuse tests (on the three-layer
#     fixture, whose served prompts have no near tie the roundings flip); the chain against itself
#     (chunks, prefill blocks, drafts) with N < L;
#   - the dense weights on the device only with N < L: only the N layers' matrices drop
#     their host copies, none is read back on a healthy run, the plan counts them alone;
#   - everything fitting: N = L, the same bits with and without COLI_VK_CHAIN_LAYERS=L.

PTL_V4_L6=deepseek_v4_tiny_l6
# Six layers: windows only (0, 3, 4), two indexers (1, 5), a heavy ratio (2). On a random
# six-layer fixture a near tie can decide a greedy token: DeepSeek V4's roundings can flip
# it between the device's and the CPU's sums, the full chain's as much as a partial one's,
# and between the CPU and the reference too. Measured on Lavapipe (tokens of the three
# oracle cases, N = 1, 3, 5, 6): ratios 0,4,8,0,4,8 moved the long case's first token on
# the device at every N, with the chain before this change as well (its top two logits 6%
# apart); 0,4,0,8,4,0 and four other orders put the CPU off the reference on a case; these
# keep the CPU on the reference and the device on the CPU in every case.
PTL_V4_RATIOS=0,4,8,0,0,4
# ptl_v4_fixture <dir> <layers> <ratios>: the tiny fixture with more layers (the generator's
# layer kinds follow the ratios: 0 a window only, 4 the indexer's, else the heavy ratio)
ptl_v4_fixture() {
  $PY - tools/make_deepseek_v4_tiny.py "$1" "$2" "$3" <<'PY' > /dev/null
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
gen.LAYERS = int(sys.argv[3]); gen.COMPRESS_RATIOS = [int(x) for x in sys.argv[4].split(",")]
kinds = {0: "sliding_attention", 4: "compressed_sparse_attention"}
hf = gen.make_hf_config
def patched(C, hf=hf):
    def make(**kw):
        kw["layer_types"] = [kinds.get(r, "heavily_compressed_attention") for r in gen.COMPRESS_RATIOS]
        kw["mlp_layer_types"] = ["hash_moe"] + ["moe"] * (gen.LAYERS - 1)
        return C(**kw)
    return hf(make)
gen.make_hf_config = patched
sys.argv = ["make_deepseek_v4_tiny.py", "--output", sys.argv[2], "--force"]
gen.main()
PY
}
# ptl_v4_run <err> <fixture> <case> <env...>: one chained run of a case, its stderr kept
ptl_v4_run() {
  local err=$1 fx=$2 c=$3; shift 3
  local p mt
  p=$(v4_chain_prompt "$fx" "$c")
  mt=$($PY -c 'import json,sys; c=sys.argv[2]; print(c.split(":")[2] if c.startswith("ids:") else json.load(open(sys.argv[1]+"/ref.json"))["cases"][c]["max_new_tokens"])' "$fx" "$c")
  rm -f "$fx/.coli_usage"
  env "$@" COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" > /dev/null 2> "$err" ||
    { tail -20 "$err"; fail "probe run on $fx $c"; }
  rm -f "$fx/.coli_usage"
}
# ptl_v4_resident <log>: the matrices the device holds at setup and at exit, "S X"
ptl_v4_resident() {
  local s x
  s=$(sed -n 's/^\[VK\] deepseek_v4 chain: [0-9]* layers on the device (.*, \([0-9]*\) matrices (.* on the device$/\1/p' "$1" | tail -1)
  x=$(sed -n 's/^\[VK\] deepseek_v4: \([0-9]*\) resident matrices on the device.*/\1/p' "$1" | tail -1)
  echo "${s:-0} ${x:-0}"
}
# ptl_v4_off <tag> <fixture> <case> <env...>: N = 0, the chain off: the CPU's ids, no chain
# forward, no dense matrix on the device
ptl_v4_off() {
  local tag=$1 fx=$2 c=$3; shift 3
  local p mt
  p=$(v4_chain_prompt "$fx" "$c")
  mt=$($PY -c 'import json,sys; c=sys.argv[2]; print(c.split(":")[2] if c.startswith("ids:") else json.load(open(sys.argv[1]+"/ref.json"))["cases"][c]["max_new_tokens"])' "$fx" "$c")
  rm -f "$fx/.coli_usage" v4-cpu.json v4-vk.json
  env "$@" ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" --record-oracle v4-cpu.json > /dev/null 2> v4-cpu.err || { cat v4-cpu.err; fail "$tag: CPU run"; }
  rm -f "$fx/.coli_usage"
  env "$@" COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" \
    --record-oracle v4-vk.json > /dev/null 2> v4-vk.err || { tail -20 v4-vk.err; fail "$tag: device run"; }
  rm -f "$fx/.coli_usage"
  $PY -c 'import json,sys; a=json.load(open("v4-cpu.json")); b=json.load(open("v4-vk.json")); sys.exit(a["full_ids"] != b["full_ids"])' ||
    fail "$tag: the ids differ from the CPU's"
  ptl_check_n deepseek_v4 v4-vk.err 0 "$tag"
  [ "$(chain_count deepseek_v4 v4-vk.err)" = 0 ] || { grep '^\[VK\]' v4-vk.err; fail "$tag: the chain ran"; }
  if grep -q '^\[VK\] deepseek_v4 chain: [0-9]* of [0-9]* layers placed' v4-vk.err; then ptl_check_placed deepseek_v4 v4-vk.err "$tag"; fi
  set -- $(ptl_v4_resident v4-vk.err)
  [ "$2" = 0 ] || { grep '^\[VK\]' v4-vk.err; fail "$tag: $2 dense matrices on the device with the chain off"; }
  echo "OK $tag: N = 0, the chain off, ids = CPU, no dense matrix on the device"
}
# ptl_v4 <tag> <fixture> <case> <k|-> <env...>: v4_chain_gate with COLI_VK_CHAIN_LAYERS=k
# (-: the fit decides), then N = PTL_WANT (default k), the placed check, and with N < L
# nothing more on the device at exit than at setup
ptl_v4() {
  local tag=$1 fx=$2 c=$3 k=$4; shift 4
  local want=${PTL_WANT:-$k} e=()
  [ "$k" != - ] && e=(COLI_VK_CHAIN_LAYERS=$k)
  if [ "$want" = 0 ]; then ptl_v4_off "$tag" "$fx" "$c" "${e[@]}" "$@"; return; fi
  v4_chain_gate "$tag" "$fx" "$c" "${e[@]}" "$@"
  ptl_check_n deepseek_v4 v4-vk.err "$want" "$tag"
  ptl_check_placed deepseek_v4 v4-vk.err "$tag"
  local L; L=$(ptl_L deepseek_v4 v4-vk.err)
  if [ "$want" -lt "$L" ]; then
    set -- $(ptl_v4_resident v4-vk.err)
    [ "$1" = "$2" ] || { grep '^\[VK\]' v4-vk.err; fail "$tag: $1 matrices on the device at setup, $2 at exit: the CPU's layers went up"; }
  fi
  echo "   N = $want of $L, $(ptl_calc matrices deepseek_v4 v4-vk.err | cut -d' ' -f1) B of matrices on the device after setup"
}
# ptl_v4_plan <tag> <fixture> <log> <env...>: coli plan's prediction (resource_plan.py,
# vk_chain_fit) for the same device and settings: the engine's free bytes, per-layer bytes,
# fixed bytes and N, from the checkpoint's header and config alone
ptl_v4_plan() {
  local tag=$1 fx=$2 log=$3; shift 3
  $PY - "$fx" "$log" "$@" <<'PY' || fail "$tag: coli plan predicts otherwise"
import re, sys
sys.path.insert(0, ".")
import resource_plan as rp
fx, log = sys.argv[1], sys.argv[2]
env = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
env.update(COLI_VULKAN="1", COLI_VK_CHAIN="1")
lines = open(log, errors="replace").read().splitlines()
f = [l for l in lines if l.startswith("[VK] deepseek_v4 chain fit: ")][-1]
num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
n = int(re.findall(r"\[VK\] deepseek_v4 chain: (\d+) of \d+ layers on the device", "\n".join(lines))[0])
fit = rp.vk_chain_fit(rp.analyze_model(fx), "deepseek_v4", env, {"type": "cpu"})
ok = (fit["layers"], fit["fixed"], fit["free"], fit["n"]) == (layers, num("fixed"), num("free"), n)
print(f"   coli plan: N = {fit['n']} of {fit['L']} ({'as the engine' if ok else 'the engine: %d' % n}), "
      f"layers {'= the engine' if fit['layers'] == layers else fit['layers']}, fixed {fit['fixed']} B, free {fit['free']} B")
sys.exit(0 if ok else 1)
PY
}
# ptl_v4_cap <tag> <fixture> <case> <k> <env...>: COLI_VK_DEVICE_CAP_MB aimed at k layers
# from a probe at 256 MiB with the same settings; the reserve holds the first chunk's
# scratch and the frames' staging after the tier took the rest
ptl_v4_cap() {
  local tag=$1 fx=$2 c=$3 k=$4; shift 4
  local R=(COLI_VK_TIER_RESERVE_GB=0.04) cap pred
  ptl_v4_run ptl-probe.err "$fx" "$c" COLI_VK_DEVICE_CAP_MB=256 "${R[@]}" "$@"
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap deepseek_v4 ptl-probe.err "$k") || fail "$tag: no cap from the probe"
  PTL_WANT=$k ptl_v4 "$tag (cap $cap MiB)" "$fx" "$c" - COLI_VK_DEVICE_CAP_MB="$cap" "${R[@]}" "$@"
  pred=$(ptl_calc predict deepseek_v4 v4-vk.err)
  [ "$pred" = "$k" ] || { grep '^\[VK\] deepseek_v4 chain' v4-vk.err; fail "$tag: the rule gives $pred from the run's own numbers"; }
  ptl_v4_plan "$tag" "$fx" v4-vk.err COLI_VK_DEVICE_CAP_MB="$cap" "${R[@]}" "$@"
}
# ptl_v4_fault <tag> <fixture> <case> <k> <env...>: a staged upload fails inside layer k's
# setup (every layer asked for): the chain keeps the layers before it
ptl_v4_fault() {
  local tag=$1 fx=$2 c=$3 k=$4; shift 4
  local L n
  L=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["num_hidden_layers"])' "$fx")
  ptl_v4_run ptl-probe.err "$fx" "$c" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS="$L" "$@"
  n=$(ptl_calc fault deepseek_v4 ptl-probe.err "$k") || { grep '^\[VK\] deepseek_v4 chain' ptl-probe.err; fail "$tag: no fault count from the probe"; }
  PTL_WANT=$k ptl_v4 "$tag (submit #$n)" "$fx" "$c" "$L" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:"$n" "$@"
  grep -q "deepseek_v4 chain: $k of $L layers on the device.*layer $k did not reach the device" v4-vk.err ||
    { grep '^\[VK\] deepseek_v4 chain' v4-vk.err; fail "$tag: no line for layer $k's failure"; }
}
# ptl_v4_droppable <fixture> <k>: the tensors of layers 0..k-1 the device may hold alone
# (coli_v4_dense_device_only_tensor), from the checkpoint's header
ptl_v4_droppable() {
  $PY - "$1" "$2" <<'PY'
import json, re, struct, sys
with open(sys.argv[1] + "/model.safetensors", "rb") as f:
    h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
n = 0
for name, t in h.items():
    m = re.match(r"layers\.(\d+)\.(.*)$", name)
    if not m or int(m[1]) >= int(sys.argv[2]) or len(t.get("shape", [])) != 2: continue
    if t["dtype"] == "F8_E4M3" or (t["dtype"] == "BF16" and m[2].endswith(("compressor.wkv.weight", "compressor.wgate.weight"))): n += 1
print(n)
PY
}
# ptl_v4_dho <tag> <fixture> <case> <k> <env...>: the dense weights on the device only with
# N = k: the k layers' tensors alone drop their host copies (the count from the header),
# none comes back on a healthy run (after a loss: the device layers' do), the plan counts
# them on the device
ptl_v4_dho() {
  local tag=$1 fx=$2 c=$3 k=$4; shift 4
  ptl_v4 "$tag" "$fx" "$c" "$k" COLI_VK_DENSE_HOST=0 "$@"
  local n want L r
  n=$(dho_dropped v4-vk.err); want=$(ptl_v4_droppable "$fx" "$k"); L=$(ptl_L deepseek_v4 v4-vk.err); r=$(dho_reloaded v4-vk.err)
  [ "$n" = "$want" ] || { grep '^\[VK\]' v4-vk.err; fail "$tag: $n host copies dropped, the $k layers have $want"; }
  grep -q '^ram_tiers .*dense=device(' v4-vk.err || { grep ram_tiers v4-vk.err; fail "$tag: the plan did not count the device's layers"; }
  if [ "$k" -lt "$L" ]; then
    grep -q "dense matrices on the device only, .* (the $k of $L layers on the device; the $((L - k)) on the CPU keep theirs)" v4-vk.err ||
      { grep '^\[VK\]' v4-vk.err; fail "$tag: the placed line does not say the $k of $L layers"; }
  fi
  if [ -n "${FAULT_BACK:-}" ]; then
    [ "$r" -gt 0 ] && grep -q 'the device that held the dense layers is gone' v4-vk.err || { grep '^\[VK\]' v4-vk.err; fail "$tag: no read-back after the loss"; }
  else [ "$r" = 0 ] || { grep '^\[VK\]' v4-vk.err; fail "$tag: $r matrices read back from disk on a healthy run"; }; fi
  echo "   $n host copies dropped (the $k layers' $want), $r read back, $(grep -o 'dense=device([^)]*)' v4-vk.err | head -1)"
}

ptl_family_deepseek() {
  export OMP_NUM_THREADS=2
  make deepseek-v4 VK=1
  v4_chain_fixtures
  ptl_v4_fixture $PTL_V4_L6 6 $PTL_V4_RATIOS
  local F=$PTL_V4_L6 c k
  # ---- N forced: k = 0, 1, L/2, L-1, L
  for k in 0 1 3 5 6; do ptl_v4 "partial deepseek_v4 COLI_VK_CHAIN_LAYERS=$k" $F long $k; done
  for c in short compressed; do ptl_v4 "partial deepseek_v4 COLI_VK_CHAIN_LAYERS=3" $F $c 3; done
  ptl_v4 "partial deepseek_v4 4 experts, 1 of 3 layers" deepseek_v4_tiny_t long 1
  ptl_v4 "partial deepseek_v4 8 experts pinned rows16, 2 of 3 layers" deepseek_v4_tiny_e8 long 2
  ptl_v4 "partial deepseek_v4 2 output groups, 1 of 3 layers" deepseek_v4_tiny_g2 long 1
  ptl_v4 "partial deepseek_v4 a window of 4, ratios 4 and 2, 2 of 3 layers" deepseek_v4_tiny_w4 long 2
  ptl_v4 "partial deepseek_v4 an indexer of 64 heads of 128, 1 of 3 layers" deepseek_v4_tiny_ix long 1
  # ---- everything fits: the fit takes every layer, the same bits as asking for all of them
  PTL_WANT=6 ptl_v4 "partial deepseek_v4 the fit, everything fitting" $F long -
  v4_chain_same "partial deepseek_v4 the fit = every layer asked for" $F long "X=1" "COLI_VK_CHAIN_LAYERS=6"
  # ---- a small device: N from the budget
  for k in 1 3 5; do ptl_v4_cap "partial deepseek_v4 a device for $k layers" $F long $k; done
  ptl_v4_cap "partial deepseek_v4 a device for no layer" $F long 0
  ptl_v4_cap "partial deepseek_v4 a device for 2 layers, 8 experts" deepseek_v4_tiny_e8 long 2
  # ---- a staged upload failing in the middle of the setup
  for k in 0 2 4; do ptl_v4_fault "partial deepseek_v4 an upload failing in layer $k" $F long $k; done
  ptl_v4_fault "partial deepseek_v4 an upload failing in layer 1, 2 output groups" deepseek_v4_tiny_g2 long 1
  # ---- the forward paths with N < L
  ptl_v4 "partial deepseek_v4 chain chunks of 3" $F long 3 COLI_VK_CHAIN_ROWS=3
  ptl_v4 "partial deepseek_v4 prefill blocks of 5" $F long 3 V4_PREFILL_CHUNK=5
  ptl_v4 "partial deepseek_v4 cold experts streamed" $F long 3 $PF_FORCE COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=64
  pf_check deepseek_v4 v4-vk.err "partial deepseek_v4 cold experts streamed" forced
  ptl_v4 "partial deepseek_v4 the tier off" $F long 3 COLI_VK_TIER=0 COLI_VK_DENSE=0
  ptl_v4 "partial deepseek_v4 beside the per-matrix trunk" $F long 3 COLI_VK_DENSE=1
  CHAINMODE=2 ptl_v4 "partial deepseek_v4 prompts only" $F long 3
  CHAINMODE=2 ptl_v4 "partial deepseek_v4 prompts only, the per-matrix trunk" $F long 3 COLI_VK_DENSE=1
  [ "$(vk_count deepseek_v4 v4-vk.err)" -gt 0 ] || { grep '^\[VK\]' v4-vk.err; fail "partial deepseek_v4 prompts only: the device's layers did not decode on the device"; }
  local A=ids:20,21,22,23,24,25,26,27,28,29,20,21,22:12 R=ids:20,21,22,23,50,51,52,20,21,22:12
  local M=ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32:12
  ptl_v4 "partial deepseek_v4 n-gram drafts accepted and rejected" $F $A 3 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ptl_v4 "partial deepseek_v4 an n-gram draft rejected" $F $R 3 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ptl_v4 "partial deepseek_v4 n-gram drafts, 1 of 6 layers" $F $M 1 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ptl_v4 "partial deepseek_v4 n-gram drafts, a window of 4" deepseek_v4_tiny_w4 $M 2 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ptl_v4 "partial deepseek_v4 V4_MTP=1 (one MTP layer: target only)" $F long 3 V4_MTP=1 V4_DRAFT=3
  kv_v4_gate "partial deepseek_v4 the KV split" 8 2 $F long COLI_VK_CHAIN_LAYERS=3
  ptl_check_n deepseek_v4 v4-vk.err 3 "partial deepseek_v4 the KV split"
  kv_v4_gate "partial deepseek_v4 the KV split, drafts" 4 1 $F $A COLI_VK_CHAIN_LAYERS=4 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ptl_check_n deepseek_v4 v4-vk.err 4 "partial deepseek_v4 the KV split, drafts"
  FAULT_BACK=3 ptl_v4 "partial deepseek_v4 device lost mid-decode" $F long 3
  FAULT_BACK=40 ptl_v4 "partial deepseek_v4 device lost in the prompt" $F long 3 COLI_VK_CHAIN_ROWS=7
  FAULT_BACK=10 ptl_v4 "partial deepseek_v4 device lost between drafts" $F $M 3 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  # ---- the chain against itself with N < L: chunks, prefill blocks, drafts cut otherwise
  v4_chain_same "partial deepseek_v4 chunks" $F long "X=1" "COLI_VK_CHAIN_ROWS=1" COLI_VK_CHAIN_LAYERS=3
  v4_chain_same "partial deepseek_v4 prefill blocks" $F long "X=1" "V4_PREFILL_CHUNK=3" COLI_VK_CHAIN_LAYERS=3
  v4_chain_same "partial deepseek_v4 a draft rejected, chunks of 2" $F $R "X=1" "COLI_VK_CHAIN_ROWS=2" COLI_VK_CHAIN_LAYERS=3 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_same "partial deepseek_v4 the tier off" $F long "X=1" "COLI_VK_TIER=0 COLI_VK_DENSE=0" COLI_VK_CHAIN_LAYERS=3
  # ---- the dense weights on the device only
  for k in 1 3 6; do ptl_v4_dho "partial deepseek_v4 device only, $k of 6 layers" $F long $k; done
  ptl_v4_dho "partial deepseek_v4 device only, drafts" $F $M 3 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  COLI_VK_STAGED=1 ptl_v4_dho "partial deepseek_v4 device only, staged" $F long 3
  FAULT_BACK=3 ptl_v4_dho "partial deepseek_v4 device only, device lost mid-decode" $F long 3
  # the RAM plan asks with N from the budget (the fit runs there, before any upload)
  ptl_v4_cap "partial deepseek_v4 device only, a device for 3 layers" $F long 3 COLI_VK_DENSE_HOST=0
  [ "$(dho_dropped v4-vk.err)" = "$(ptl_v4_droppable $F 3)" ] && [ "$(dho_reloaded v4-vk.err)" = 0 ] ||
    { grep -E '^\[VK\]|ram_tiers' v4-vk.err; fail "partial deepseek_v4 device only, a device for 3 layers: the host copies dropped are not the 3 layers'"; }
  v4_chain_same "partial deepseek_v4 device only = host copies" $F long "COLI_VK_DENSE_HOST=1" "COLI_VK_DENSE_HOST=0" COLI_VK_CHAIN_LAYERS=3
  # ---- serve sessions and the prefix-reuse contract with N < L. On the three-layer
  # fixture: the served prompts (prefixes of the long case) meet near ties on the six-layer
  # one that the full chain flips as well (measured: COLI_VK_CHAIN_LAYERS=6 and 3 move a
  # served token there, 1 does not)
  COLI_VK_CHAIN_LAYERS=1 v4_chain_serve deepseek_v4_tiny_t
  COLI_VK_CHAIN_LAYERS=2 COLI_VK_DENSE_HOST=0 v4_chain_serve deepseek_v4_tiny_t
  COLI_VK_CHAIN_LAYERS=1 V4_CHAIN_MODE=2 v4_chain_serve deepseek_v4_tiny_t
  COLI_VK_CHAIN_LAYERS=1 v4_chain_serve $F
  local t
  for t in test_deepseek_v4_prefix test_deepseek_v4_brio; do
    COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=1 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage \
      $PY tests/$t.py --binary "$PWD/deepseek_v4" --fixture "$PWD/deepseek_v4_tiny_t" > v4-test.log 2>&1 ||
      { cat v4-test.log; fail "deepseek_v4 $t with 1 of 3 layers on the device"; }
    echo "OK deepseek_v4 $t with 1 of 3 layers on the device: $(tail -1 v4-test.log)"
  done
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix $F chain.usage
  rm -f cpu.f32 vk.f32 v4-*.f32 v4-*.json v4-*.err v4-tf.txt v4-test.log ptl-probe.err
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan (LTO off, as the chain's sanitize family builds it): no
# diagnostic, the chain ran with the N the line says (or handled the loss).
ptl_v4_san() {  # <tag> <want N> <env and argv...>
  local tag=$1 want=$2; shift 2
  v4_chain_san "$tag" "$@"
  ptl_check_n deepseek_v4 san.log "$want" "$tag"
  ptl_check_placed deepseek_v4 san.log "$tag"
}
ptl_family_deepseek_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make deepseek-v4-clean >/dev/null 2>&1 || true
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  v4_chain_fixtures
  ptl_v4_fixture $PTL_V4_L6 6 $PTL_V4_RATIOS
  local F=$PTL_V4_L6 p cap n
  p=$(v4_chain_prompt $F long)
  ptl_v4_san "asan partial deepseek_v4 3 of 6 layers" 3 COLI_VK_CHAIN_LAYERS=3 ./deepseek_v4 ./$F "$p" --raw-prompt --max-tokens 4 --record-oracle san.json
  ptl_v4_san "asan partial deepseek_v4 1 of 6 layers, chunks of 3" 1 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_ROWS=3 ./deepseek_v4 ./$F "$p" --raw-prompt --max-tokens 4
  ptl_v4_run ptl-probe.err $F long COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap deepseek_v4 ptl-probe.err 2)
  ptl_v4_san "asan partial deepseek_v4 a device for 2 layers" 2 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 ./deepseek_v4 ./$F "$p" --raw-prompt --max-tokens 4
  ptl_v4_run ptl-probe.err $F long COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=6
  n=$(ptl_calc fault deepseek_v4 ptl-probe.err 3)
  ptl_v4_san "asan partial deepseek_v4 an upload failing in layer 3" 3 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$n COLI_VK_CHAIN_LAYERS=6 ./deepseek_v4 ./$F "$p" --raw-prompt --max-tokens 4
  ptl_v4_san "asan partial deepseek_v4 device only, 3 of 6 layers" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_DENSE_HOST=0 ./deepseek_v4 ./$F "$p" --raw-prompt --max-tokens 4
  ptl_v4_san "asan partial deepseek_v4 drafts, 2 of 6 layers" 2 COLI_VK_CHAIN_LAYERS=2 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1 ./deepseek_v4 ./$F \
    "$(v4_chain_prompt x ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32)" --raw-prompt --max-tokens 12
  ptl_v4_san "asan partial deepseek_v4 the KV split" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2 ./deepseek_v4 ./$F "$p" --raw-prompt --max-tokens 4
  FAULT_BACK=10 v4_chain_san "asan partial deepseek_v4 device lost" COLI_VK_CHAIN_LAYERS=3 ./deepseek_v4 ./$F "$p" --raw-prompt --max-tokens 4
  COLI_VK_CHAIN_LAYERS=2 v4_chain_serve deepseek_v4_tiny_t > san.log 2>&1 || { cat san.log; fail "asan partial deepseek_v4 served"; }
  echo "OK asan partial deepseek_v4 served: $(tail -1 san.log | cut -c1-120)"
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix $F san.json san.log ptl-probe.err chain.usage
  unset OMP_NUM_THREADS
  make deepseek-v4-clean >/dev/null 2>&1 || true
}
