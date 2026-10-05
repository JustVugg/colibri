# Sourced by tests/vulkan_engines.sh: the partial chain (docs/vulkan.md, "A partial chain")
# on GLM-5.2 (colibri) and GLM-5.3 Flash (glm53): the first N layers on the device, the CPU
# the rest and the head. Run as partial-glm and partial-glm-sanitize.
#
# Every configuration runs the glm-chain family's gate (mla_gate: the CPU's token lines,
# every logits row within 1e-4 of the largest, chain forwards) with the partial chain's
# settings on top, then checks the lines (ptl_* helpers in tests/vulkan_engines.sh):
#   - COLI_VK_CHAIN_LAYERS=k for k in 0, 1, about L/2, L-1, L: N = k, the matrices on the
#     device after setup are those k layers' (the placed line), and with k < L the
#     per-matrix path put nothing more there (the held line at exit);
#   - COLI_VK_DEVICE_CAP_MB aimed at k layers from a probe's fit line: the rule's N from
#     the run's own numbers, and the line's, are k;
#   - a staged upload failing inside layer k's setup (COLI_VK_STAGED_FAULT=submit:n, n
#     from a probe's per-layer counts): N = k, nothing of layer k on the device;
#   - the forward paths with N < L: prompt chunks, expert streaming, the DSA selection
#     handed to the CPU at a shared indexer layer, n-gram and MTP drafts, prompts only,
#     the per-matrix path beside, the tier off, the KV split, a lost device mid-decode, in
#     the prompt, between drafts and after an image (glm53's KDA state rebuilt for the
#     device's layers), serve sessions with the prompt cache and two KV slots, glm53's
#     image and its pin-branch harness;
#   - the dense weights on the device only with N < L: only the N layers' matrices drop
#     their host copies (counted from the config), none is read back on a healthy run,
#     the device layers' come back after a loss, the RAM the experts get grows with N;
#   - everything fitting: the fit takes every layer, the same bits as asking for all.
# The fixtures: glm-chain's, glm_tiny with shared indexer layers (layers 1, 3 and 4 reuse
# the selection of the full layer before them) and a six-layer GLM-5.3 (KDA and MLA
# alternating, so MLA layers run on both sides of the handoff).

# ptl_glm_shx <src> <dst>: a copy of glm_tiny whose layers 1, 3 and 4 reuse the DSA
# selection of the full layer before them (the CPU's engine reads indexer_types)
ptl_glm_shx() {
  rm -rf "$2" && cp -r "$1" "$2"
  $PY - "$2/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["indexer_types"] = ["full", "shared", "full", "shared", "shared"]
json.dump(c, open(sys.argv[1], "w"))
PY
}
# ptl_g53_l6 <dir>: the GLM-5.3 tiny with six layers, KDA and MLA alternating, the first
# MLP dense; no oracle (the engine's runs are compared with each other)
ptl_g53_l6() {
  $PY - tools/make_glm53_tiny.py "$1" <<'PY' > /dev/null
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
gen.LAYERS = 6
kinds = ["linear_attention", "deepseek_sparse_attention"] * 3
ck, rc = gen.config_kwargs, gen.runtime_config
def config_kwargs():
    k = ck()
    k["layer_types"], k["mlp_layer_types"], k["indexer_types"] = kinds, ["dense"] + ["sparse"] * 5, ["full"] * 6
    return k
def runtime_config(version, fp8_mlp=False):
    r = rc(version, fp8_mlp)
    r["text_config"]["linear_attn_config"]["kda_layers"] = [i for i, t in enumerate(kinds) if t == "linear_attention"]
    return r
gen.config_kwargs, gen.runtime_config = config_kwargs, runtime_config
gen.oracle = lambda *a: {"note": "no reference: the engine's runs are compared with each other"}
sys.argv = ["make_glm53_tiny.py", "--output", sys.argv[2], "--force"]
gen.main()
PY
}
ptl_glm_fixtures() {
  glm_chain_fixtures
  pf_ref glm_tiny/ref_long.json 200 1 250
  ptl_glm_shx glm_tiny glm_tiny_shx
  ptl_g53_l6 glm53_l6
  $PY tools/make_glm53_streaming_pair.py --fixture glm53_l6 --output glm53_l6s > /dev/null
  rm -rf glm53_l6s-deq glm53_l6_serve && cp -r glm53_l6s-i4 glm53_l6_serve
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 128 ./glm53_l6_serve > /dev/null
}

# ptl_mla_run <out> <engine> <env...> -- <argv...>: one chained run, its stderr kept (a probe)
ptl_mla_run() {
  local out=$1 eng=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./"$eng" "$@" > /dev/null 2> "$out" || true
  rm -f chain.usage
  grep -q "^\[VK\] $eng chain fit:" "$out" || { tail -20 "$out"; fail "probe run of $eng: no fit line"; }
}
# ptl_mla_held <engine> <log> <tag>: N < L, the matrices on the device at exit are the ones
# the setup placed (the held line)
ptl_mla_held() {
  local s x
  s=$(ptl_calc matrices "$1" "$2" | cut -d' ' -f1)
  x=$(sed -n "s/^\[VK\] $1 chain: [0-9]* of [0-9]* layers held at exit: \([0-9]*\) B of matrices on the device$/\1/p" "$2" | tail -1)
  { [ -n "$x" ] && [ "$s" = "$x" ]; } || { grep -a '^\[VK\]' "$2"; fail "$3: $s B of matrices at setup, ${x:-no line} at exit: a CPU layer or the tail went up"; }
}
# ptl_mla_off <engine> <tag> <env...> -- <argv...>: N = 0, the chain off: the CPU's tokens
# and logits, no chain forward, nothing of the trunk on the device at setup or at exit
ptl_mla_off() {
  local eng=$1 tag=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage cpu.f32 vk.f32
  env "${envs[@]}" COLI_USAGE=chain.usage USAGE_SAVE=0 DUMP=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || true
  rm -f chain.usage
  env "${envs[@]}" COLI_USAGE=chain.usage USAGE_SAVE=0 DUMP=vk.f32 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
    COLI_VK_CHAIN=${CHAINMODE:-1} ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  mla_toks "$eng" cpu.log > cpu.tok; mla_toks "$eng" vk.log > vk.tok
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the tokens differ from the CPU"; }
  local lg; lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }
  ptl_check_n "$eng" vk.log 0 "$tag"
  [ "$(chain_count "$eng" vk.log)" = 0 ] || { grep -a '^\[VK\]' vk.log; fail "$tag: the chain ran"; }
  ptl_check_placed "$eng" vk.log "$tag"
  ptl_mla_held "$eng" vk.log "$tag"
  [ "$(ptl_calc matrices "$eng" vk.log | cut -d' ' -f1)" = 0 ] || { grep -a '^\[VK\]' vk.log; fail "$tag: a matrix of the trunk on the device with the chain off"; }
  echo "OK $tag: N = 0, the chain off, tokens = CPU, $lg, no matrix of the trunk on the device"
}
# ptl_mla <engine> <tag> <k|-> <env...> -- <argv...>: mla_gate with COLI_VK_CHAIN_LAYERS=k
# (-: the fit decides), then N = PTL_WANT (default k), the placed check, and with N < L
# nothing more on the device at exit than at setup. KV=<rows>:<block>: kv_mla_gate (the
# KV split) instead of mla_gate.
ptl_mla() {
  local eng=$1 tag=$2 k=$3; shift 3
  local want=${PTL_WANT:-$k} e=()
  [ "$k" != - ] && e=(COLI_VK_CHAIN_LAYERS=$k)
  if [ "$want" = 0 ]; then ptl_mla_off "$eng" "$tag" "${e[@]}" "$@"; return; fi
  if [ -n "${KV:-}" ]; then kv_mla_gate "$eng" "$tag" 1 "${KV%%:*}" "${KV##*:}" "${e[@]}" "$@"
  else mla_gate "$eng" "$tag" 1 "${e[@]}" "$@"; fi
  ptl_check_n "$eng" vk.log "$want" "$tag"
  ptl_check_placed "$eng" vk.log "$tag"
  local L; L=$(ptl_L "$eng" vk.log)
  [ "$want" -ge "$L" ] || ptl_mla_held "$eng" vk.log "$tag"
  echo "   N = $want of $L, $(ptl_calc matrices "$eng" vk.log | cut -d' ' -f1) B of matrices on the device after setup"
}
# ptl_mla_plan <tag> <model dir> <log> <env...>: coli plan's prediction (resource_plan.py,
# vk_chain_fit) for glm53 on the same device and settings: the engine's free bytes,
# per-layer bytes, fixed bytes and N, from the checkpoint's header and config alone.
# colibri has no layout (its dense formats are the command line's): nothing to compare.
ptl_mla_plan() {
  local tag=$1 fx=$2 log=$3; shift 3
  $PY - "$fx" "$log" "$@" <<'PY' || fail "$tag: coli plan predicts otherwise"
import re, sys
sys.path.insert(0, ".")
import resource_plan as rp
fx, log = sys.argv[1], sys.argv[2]
env = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
env.update(COLI_VULKAN="1", COLI_VK_CHAIN="1")
lines = open(log, errors="replace").read().splitlines()
f = [l for l in lines if l.startswith("[VK] glm53 chain fit: ")][-1]
num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
n = int(re.findall(r"\[VK\] glm53 chain: (\d+) of \d+ layers on the device", "\n".join(lines))[0])
fit = rp.vk_chain_fit(rp.analyze_model(fx), "glm53", env, {"type": "cpu"})
ok = (fit["layers"], fit["fixed"], fit["free"], fit["n"]) == (layers, num("fixed"), num("free"), n)
print(f"   coli plan: N = {fit['n']} of {fit['L']} ({'as the engine' if ok else 'the engine: %d' % n}), "
      f"layers {'= the engine' if fit['layers'] == layers else fit['layers']}, fixed {fit['fixed']} B, free {fit['free']} B")
sys.exit(0 if ok else 1)
PY
}
# ptl_mla_cap <engine> <tag> <k> <env...> -- <argv...>: COLI_VK_DEVICE_CAP_MB aimed at k
# layers from a probe at 256 MiB with the same settings; the reserve holds the device's
# own scratch beside what the tier takes. glm53: coli plan predicts the same N (PTL_MODEL
# names the checkpoint)
ptl_mla_cap() {
  local eng=$1 tag=$2 k=$3; shift 3
  local R=COLI_VK_TIER_RESERVE_GB=0.04 cap pred
  ptl_mla_run ptl-probe.err "$eng" COLI_VK_DEVICE_CAP_MB=256 $R "$@"
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap "$eng" ptl-probe.err "$k") || fail "$tag: no cap from the probe"
  PTL_WANT=$k ptl_mla "$eng" "$tag (cap $cap MiB)" - COLI_VK_DEVICE_CAP_MB="$cap" $R "$@"
  pred=$(ptl_calc predict "$eng" vk.log)
  [ "$pred" = "$k" ] || { grep -a "^\[VK\] $eng chain" vk.log; fail "$tag: the rule gives $pred from the run's own numbers"; }
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  if [ -n "${PTL_MODEL:-}" ]; then ptl_mla_plan "$tag" "$PTL_MODEL" vk.log COLI_VK_DEVICE_CAP_MB="$cap" $R "${envs[@]}"; fi
}
# ptl_mla_fault <engine> <tag> <k> <env...> -- <argv...>: a staged upload fails inside
# layer k's setup (every layer asked for): the chain keeps the layers before it
ptl_mla_fault() {
  local eng=$1 tag=$2 k=$3; shift 3
  local L n
  ptl_mla_run ptl-probe.err "$eng" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=99 "$@"
  L=$(ptl_L "$eng" ptl-probe.err)
  n=$(ptl_calc fault "$eng" ptl-probe.err "$k") || { grep -a "^\[VK\] $eng chain" ptl-probe.err; fail "$tag: no fault count from the probe"; }
  PTL_WANT=$k ptl_mla "$eng" "$tag (submit #$n)" "$L" COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:"$n" "$@"
  grep -q "$eng chain: $k of $L layers on the device.*layer $k did not reach the device" vk.log ||
    { grep -a "^\[VK\] $eng chain" vk.log; fail "$tag: no line for layer $k's failure"; }
}
# ptl_mla_droppable <engine> <snapshot> <k>: the matrices of layers 0..k-1 that drop their
# host copies (glm_dho_layer, g53_dho_layer): colibri's five attention matrices, three of
# the MLP or the shared expert, the DSA indexer's three on a full layer; glm53's KDA layer
# twelve, its MLA layer thirteen (the absorbed kv_b's two halves among them). KEEP_KVB=1:
# the chain does not run every forward (prompts only): kv_b and the indexer stay
ptl_mla_droppable() {
  $PY - "$1" "$2/config.json" "$3" "${KEEP_KVB:-0}" <<'PY'
import json, sys
eng, c, k, keep = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3]), sys.argv[4] == "1"
c = c.get("text_config", c)
if eng == "colibri":
    kinds = c.get("indexer_types") or ["full"] * c["num_hidden_layers"]
    print(sum(7 + (0 if keep else 1 + (3 if kinds[i] == "full" else 0)) for i in range(k)))
else:
    print(sum(12 if c["layer_types"][i] == "linear_attention" else 11 + (0 if keep else 2) for i in range(k)))
PY
}
# ptl_mla_dho <engine> <tag> <k> <snapshot> <env...> -- <argv...>: the dense weights on the
# device only with N = k < L: the k layers' matrices alone drop their host copies, the
# placed line says so, none comes back on a healthy run (FAULT_BACK: the device layers' do)
ptl_mla_dho() {
  local eng=$1 tag=$2 k=$3 snap=$4; shift 4
  ptl_mla "$eng" "$tag" "$k" COLI_VK_DENSE_HOST=0 "$@"
  local n want L r
  n=$(dho_dropped vk.log); want=$(ptl_mla_droppable "$eng" "$snap" "$k"); L=$(ptl_L "$eng" vk.log); r=$(dho_reloaded vk.log)
  [ "$n" = "$want" ] || { grep -a '^\[VK\]' vk.log; fail "$tag: $n host copies dropped, the $k layers have $want"; }
  grep -q "dense matrices on the device only, .* (the $k of $L layers on the device; the $((L - k)) on the CPU keep theirs)" vk.log ||
    { grep -a '^\[VK\]' vk.log; fail "$tag: the placed line does not say the $k of $L layers"; }
  if [ -n "${FAULT_BACK:-}" ]; then
    { [ "$r" -gt 0 ] && [ "$r" -le "$n" ]; } || { grep -a '^\[VK\]' vk.log; fail "$tag: $r read back after the loss (the device held $n alone)"; }
  else [ "$r" = 0 ] || { grep -a '^\[VK\]' vk.log; fail "$tag: $r matrices read back from disk on a healthy run"; }; fi
  echo "   $n host copies dropped (the $k layers' $want), $r read back"
}
# ptl_mla_same <engine> <tag> <env A> <env B> <env...> -- <argv...>: two chained runs that
# differ only in env A / env B (each one string, KEY=VALUE words): the same tokens and the
# same logits, bit for bit (COLI_VK_TIER_BALANCE=0: the tier's split of the routed experts
# with the CPU follows the timings, and with it which side's bits a row gets); TOL=1: the
# logits within 1e-4 of the largest (logits_close)
ptl_mla_same() {
  local eng=$1 tag=$2 a=$3 b=$4; shift 4
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local x
  for x in a b; do
    local v; [ $x = a ] && v=$a || v=$b
    rm -f chain.usage
    # shellcheck disable=SC2086
    env "${envs[@]}" $v COLI_USAGE=chain.usage USAGE_SAVE=0 DUMP=same-$x.f32 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VULKAN=1 \
      COLI_VK_CHAIN=${CHAINMODE:-1} ./"$eng" "$@" > same-$x.log 2>&1 || true
    mla_toks "$eng" same-$x.log > same-$x.tok
  done
  rm -f chain.usage
  { [ -s same-a.tok ] && cmp -s same-a.tok same-b.tok; } || { cat same-a.tok same-b.tok; fail "$tag: the tokens differ"; }
  if [ "${TOL:-0}" = 1 ]; then
    local lg; lg=$(logits_close same-a.f32 same-b.f32) || { echo "$lg"; fail "$tag: logits"; }
    echo "OK $tag: the same tokens, $lg ($a | $b)"
  else
    cmp -s same-a.f32 same-b.f32 || fail "$tag: the logits differ"
    echo "OK $tag: the same tokens and logits ($a | $b)"
  fi
}
# ptl_mla_resident <log>: colibri's "resident dense" MB after the load (what cap_for_ram counts)
ptl_mla_resident() { grep -ao 'resident dense: [0-9.]* MB' "$1" | head -1 | sed 's/[^0-9.]//g'; }

ptl_family_glm() {
  make colibri glm53 VK=1
  ptl_glm_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  local G="SNAP=glm_tiny REF=ref_glm.json" X="SNAP=glm_tiny_shx REF=ref_glm.json" T="SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json" k
  # ---- colibri: N forced, k = 0, 1, L/2, L-1, L, and every format
  for k in 0 1 2 4 5; do ptl_mla colibri "partial colibri COLI_VK_CHAIN_LAYERS=$k" $k $G -- 64 16 16; done
  ptl_mla colibri "partial colibri 8-bit trunk, 2 of 5 layers" 2 $G IDOT=0 -- 2 8 8
  ptl_mla colibri "partial colibri 4-bit trunk, 3 of 5 layers" 3 $G IDOT=0 -- 2 4 4
  ptl_mla colibri "partial colibri i4 container, 1 of 5 layers" 1 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 -- 1 4 4
  # the DSA selection handed over: layer N reuses the device's selection (shared indexer
  # layers 1, 3 and 4), and a full layer N makes its own (2)
  for k in 1 2 3 4; do ptl_mla colibri "partial colibri DSA top-4, shared indexers, $k of 5 layers" $k $X DSA_TOPK=4 -- 64 16 16; done
  ptl_mla colibri "partial colibri DSA top-4 prefill in chunks of 5, 1 of 5 layers" 1 $X DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  ptl_mla colibri "partial colibri DSA_FORCE, 3 of 5 layers" 3 $X DSA_FORCE=1 -- 64 16 16
  # ---- everything fits: the fit takes every layer, the same bits as asking for all of them
  PTL_WANT=5 ptl_mla colibri "partial colibri the fit, everything fitting" - $G -- 64 16 16
  ptl_mla_same colibri "partial colibri the fit = every layer asked for" "X=1" "COLI_VK_CHAIN_LAYERS=5" $G -- 64 16 16
  # ---- a small device: N from the budget
  for k in 1 4 0; do ptl_mla_cap colibri "partial colibri a device for $k layers" $k $G -- 64 16 16; done
  # ---- a staged upload failing in the middle of the setup
  for k in 0 2 4; do ptl_mla_fault colibri "partial colibri an upload failing in layer $k" $k $G -- 64 16 16; done
  # ---- the forward paths with N < L
  ptl_mla colibri "partial colibri prefill in chunks of 3" 2 $G TF=1 COLI_VK_CHAIN_ROWS=3 -- 64 16 16
  ptl_mla colibri "partial colibri cold experts streamed" 3 $PF_FORCE COLI_VK_TIER_GB=0.0003 COLI_VK_CHAIN_ROWS=64 SNAP=glm_tiny REF=glm_tiny/ref_long.json TF=1 -- 2 16 16
  pf_check colibri vk.log "partial colibri cold experts streamed" forced
  ptl_mla colibri "partial colibri n-gram drafts" 2 $G DRAFT=3 -- 64 16 16
  for k in 1 3; do ptl_mla colibri "partial colibri MTP depth 3, $k of 5 layers" $k $T DRAFT=3 -- 64 16 16; done
  ptl_mla colibri "partial colibri MTP with DSA top-4" 2 $T DSA_TOPK=4 -- 64 16 16
  # a declined step: the exact verify of COLI_EXACT_VERIFY runs on the CPU (every layer),
  # the other forwards take the device's layers
  ptl_mla colibri "partial colibri MTP, the exact verify declined" 2 $T DRAFT=2 COLI_EXACT_VERIFY=1 -- 64 16 16
  ptl_mla colibri "partial colibri tier off" 2 $G COLI_VK_TIER=0 -- 64 16 16
  ptl_mla colibri "partial colibri beside COLI_VK_DENSE=1 COLI_VK_ATTN=1" 2 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 COLI_VK_DENSE=1 COLI_VK_ATTN=1 -- 2 4 4
  CHAINMODE=2 ptl_mla colibri "partial colibri prompts only, drafts" 2 $G DRAFT=3 -- 64 16 16
  KV=8:4 ptl_mla colibri "partial colibri the KV split" 3 $G -- 64 16 16
  KV=8:4 ptl_mla colibri "partial colibri the KV split, DSA top-4 prefill in chunks of 5" 3 $X DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  FAULT_BACK=12 ptl_mla colibri "partial colibri device lost mid-decode" 2 $G -- 64 16 16
  FAULT_BACK=2 ptl_mla colibri "partial colibri device lost in the prompt" 3 $G TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  FAULT_BACK=8 ptl_mla colibri "partial colibri device lost with MTP" 2 $T DRAFT=2 -- 64 16 16
  # ---- the chain against itself with N < L: chunks
  ptl_mla_same colibri "partial colibri chunks of 1 = 5" "COLI_VK_CHAIN_ROWS=1" "COLI_VK_CHAIN_ROWS=5" COLI_VK_CHAIN_LAYERS=2 $X DSA_TOPK=4 TF=1 -- 64 16 16
  # ---- the dense weights on the device only
  for k in 1 3; do ptl_mla_dho colibri "partial colibri device only, $k of 5 layers" $k glm_tiny $G -- 64 16 16; done
  ptl_mla_dho colibri "partial colibri device only, MTP" 2 glm_tiny_mtp $T DRAFT=2 -- 64 16 16
  ptl_mla_dho colibri "partial colibri device only, DSA top-4 shared" 1 glm_tiny_shx $X DSA_TOPK=4 -- 64 16 16
  # prompts only: decode runs per matrix, the device layers' kv_b and indexer stay on the
  # host (the CPU's attention reads them by row)
  CHAINMODE=2 KEEP_KVB=1 ptl_mla_dho colibri "partial colibri device only, prompts only, drafts" 2 glm_tiny $G DRAFT=3 -- 64 16 16
  COLI_VK_STAGED=1 ptl_mla_dho colibri "partial colibri device only, staged" 2 glm_tiny $G -- 64 16 16
  FAULT_BACK=12 ptl_mla_dho colibri "partial colibri device only, device lost mid-decode" 2 glm_tiny $G -- 64 16 16
  # the RAM cap_for_ram counts: the host copies dropped are the N layers', resident dense
  # falls with N (the pins and the cap count only what was dropped)
  local r0 r2 r5
  for k in 0 2 5; do
    rm -f chain.usage
    env $G USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 \
      COLI_VK_CHAIN_LAYERS=$k ./colibri 64 16 16 > ptl-ram$k.log 2>&1 || true
  done
  rm -f chain.usage
  r0=$(ptl_mla_resident ptl-ram0.log); r2=$(ptl_mla_resident ptl-ram2.log); r5=$(ptl_mla_resident ptl-ram5.log)
  $PY -c "import sys; a,b,c=map(float,sys.argv[1:]); sys.exit(not (a > b > c))" "$r0" "$r2" "$r5" ||
    { grep -a 'resident dense\|dense matrices' ptl-ram*.log; fail "partial colibri device only: resident dense $r0 / $r2 / $r5 MB with 0 / 2 / 5 layers"; }
  echo "OK partial colibri device only: resident dense $r0 MB with 0 layers on the device, $r2 MB with 2, $r5 MB with 5"
  # serve sessions frame for frame with 2 of 5 layers: pins, the prompt cache, the prefill
  # read-out, drafts, two KV slots (the ragged batch declines the chain), prompts only
  export CHAIN_SERVE_EXPECT='colibri chain: 2 of 5 layers on the device'
  CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 COLI_VK_CHAIN_LAYERS=2
  CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4 DRAFT=3 COLI_VK_CHAIN_LAYERS=2
  CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2 COLI_VK_CHAIN_LAYERS=2
  COLI_VK_CHAIN=2 CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4 COLI_VK_CHAIN_LAYERS=2
  CHAIN_SERVE_EXPECT='colibri: [0-9]+ dense matrices on the device only.*the 2 of 5 layers on the device' COLI_VK_DENSE_HOST=0 \
    CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 COLI_VK_CHAIN_LAYERS=2
  unset CHAIN_SERVE_EXPECT

  # ---- glm53: the six-layer fixture (KDA and MLA alternating), N forced
  local ids lids F="--model glm53_l6s-i4" M="--model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4" bits
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  lids=$(pf_ids 200 2 120)
  for k in 0 1 3 5 6; do ptl_mla glm53 "partial glm53 COLI_VK_CHAIN_LAYERS=$k" $k GLM53_BITS=32 -- $F --ids $ids --greedy 4; done
  for bits in 8 4; do ptl_mla glm53 "partial glm53 ${bits}-bit trunk, 3 of 6 layers" 3 GLM53_BITS=$bits -- $F --ids $ids --greedy 4; done
  ptl_mla glm53 "partial glm53 decode, 2 of 6 layers" 2 GLM53_BITS=32 -- $F --ids 5,7,9,11,13,17,19,23 --greedy 8
  ptl_mla glm53 "partial glm53 resident experts, 4 of 6 layers" 4 GLM53_BITS=32 -- --model glm53_l6 --ids $ids --greedy 6
  ptl_mla glm53 "partial glm53 four layers, 2 on the device" 2 GLM53_BITS=32 -- --model glm53_stream-i4 --ids $ids --greedy 4
  ptl_mla glm53 "partial glm53 swiglu_limit 0, 2 of 4 layers" 2 GLM53_BITS=32 -- --model glm53_lim0 --ids $ids --greedy 4
  # ---- everything fits
  PTL_WANT=6 ptl_mla glm53 "partial glm53 the fit, everything fitting" - GLM53_BITS=32 -- $F --ids $ids --greedy 4
  ptl_mla_same glm53 "partial glm53 the fit = every layer asked for" "X=1" "COLI_VK_CHAIN_LAYERS=6" GLM53_BITS=32 -- $F --ids $ids --greedy 4
  # ---- a small device; an upload failing in the middle of the setup
  for k in 1 4 0; do PTL_MODEL=glm53_l6s-i4 ptl_mla_cap glm53 "partial glm53 a device for $k layers" $k GLM53_BITS=32 -- $F --ids $ids --greedy 4; done
  PTL_MODEL=glm53_l6s-i4 ptl_mla_cap glm53 "partial glm53 a device for 3 layers, 4-bit trunk, chunks of 7" 3 GLM53_BITS=4 COLI_VK_CHAIN_ROWS=7 -- $F --ids $ids --greedy 4
  for k in 0 2 5; do ptl_mla_fault glm53 "partial glm53 an upload failing in layer $k" $k GLM53_BITS=4 -- $F --ids $ids --greedy 4; done
  # ---- the forward paths with N < L
  ptl_mla glm53 "partial glm53 prefill chunks of 7, one cache slot" 3 GLM53_BITS=4 GLM53_PREFILL_CHUNK=7 GLM53_EXPERT_GB=0.000001 -- $F --ids $ids --greedy 4
  ptl_mla glm53 "partial glm53 chain chunks of 3" 4 GLM53_BITS=32 COLI_VK_CHAIN_ROWS=3 -- $F --ids $ids --greedy 4
  ptl_mla glm53 "partial glm53 cold experts streamed" 3 $PF_FORCE COLI_VK_TIER_GB=0.0002 COLI_VK_CHAIN_ROWS=64 GLM53_BITS=4 GLM53_PREFILL_CHUNK=200 -- $F --ids "$lids" --greedy 4
  pf_check glm53 vk.log "partial glm53 cold experts streamed" forced
  ptl_mla glm53 "partial glm53 tier off" 3 GLM53_BITS=32 COLI_VK_TIER=0 -- $F --ids $ids --greedy 6
  ptl_mla glm53 "partial glm53 beside COLI_VK_DENSE=1" 3 GLM53_BITS=4 COLI_VK_DENSE=1 -- $F --ids $ids --greedy 4
  ptl_mla glm53 "partial glm53 an image, 2 of 4 layers" 2 GLM53_BITS=32 -- $M --greedy 4
  CHAINMODE=2 ptl_mla glm53 "partial glm53 prompts only" 3 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- $F --ids $ids --greedy 6
  KV=16:4 ptl_mla glm53 "partial glm53 the KV split" 4 GLM53_BITS=32 -- $F --ids $ids --greedy 8
  KV=16:4 ptl_mla glm53 "partial glm53 the KV split, chain chunks of 3" 2 GLM53_BITS=32 COLI_VK_CHAIN_ROWS=3 -- $F --ids $ids --greedy 8
  FAULT_BACK=7 REBUILD=1 ptl_mla glm53 "partial glm53 device lost, the device layers' KDA state rebuilt" 3 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- $F --ids $ids --greedy 6
  FAULT_BACK=2 REBUILD=1 ptl_mla glm53 "partial glm53 device lost after an image" 2 GLM53_BITS=32 -- $M --greedy 4
  # glm53's chain rounds differently with the chunk's rows with every layer on the device
  # too (the binary before this branch: 3.6e-7 of 1.54 between chunks of 1 and 7): the
  # chunks compare within the bound, the tokens exactly
  TOL=1 ptl_mla_same glm53 "partial glm53 chunks of 1 = 7" "COLI_VK_CHAIN_ROWS=1" "COLI_VK_CHAIN_ROWS=7" COLI_VK_CHAIN_LAYERS=4 GLM53_BITS=32 -- $F --ids $ids --greedy 4
  # ---- the dense weights on the device only
  for k in 1 3; do ptl_mla_dho glm53 "partial glm53 device only, $k of 6 layers" $k glm53_l6s-i4 GLM53_BITS=32 -- $F --ids $ids --greedy 4; done
  ptl_mla_dho glm53 "partial glm53 device only, 4-bit trunk" 4 glm53_l6s-i4 GLM53_BITS=4 -- $F --ids $ids --greedy 4
  ptl_mla_dho glm53 "partial glm53 device only, an image" 2 glm53_mm_tiny GLM53_BITS=32 -- $M --greedy 4
  CHAINMODE=2 KEEP_KVB=1 ptl_mla_dho glm53 "partial glm53 device only, prompts only" 4 glm53_l6s-i4 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- $F --ids $ids --greedy 6
  COLI_VK_STAGED=1 ptl_mla_dho glm53 "partial glm53 device only, staged" 3 glm53_l6s-i4 GLM53_BITS=4 -- $F --ids $ids --greedy 4
  FAULT_BACK=7 REBUILD=1 ptl_mla_dho glm53 "partial glm53 device only, device lost" 3 glm53_l6s-i4 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- $F --ids $ids --greedy 6
  # the expert cache sizes itself after the N layers' host copies went
  rm -f chain.usage
  env GLM53_VERBOSE=1 GLM53_BITS=32 USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN_LAYERS=3 ./glm53 $F --ids 5,7,9 --greedy 2 > vk.log 2>&1 || true
  rm -f chain.usage
  local given budget
  given=$(grep -n 'dense matrices on the device only' vk.log | head -1 | cut -d: -f1); budget=$(grep -n '^expert budget:' vk.log | head -1 | cut -d: -f1)
  { [ -n "$given" ] && [ -n "$budget" ] && [ "$given" -lt "$budget" ]; } || { cat vk.log; fail "partial glm53 device only: the expert cache sized itself before the host copies went"; }
  echo "OK partial glm53 device only: the expert cache sized after the 3 layers' host copies went ($(grep -a '^expert budget:' vk.log | head -1))"
  # serve sessions with 3 of 6 layers: pins, the prompt cache, read-outs, chain chunks, two
  # sessions taking the device in turn, prompts only, device only
  export CHAIN_SERVE_EXPECT='glm53 chain: 3 of 6 layers on the device'
  CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=32 COLI_VK_CHAIN_LAYERS=3
  CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3 COLI_VK_CHAIN_LAYERS=3
  CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=32 KV_SLOTS=2 COLI_VK_CHAIN_LAYERS=3
  COLI_VK_CHAIN=2 CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=32 COLI_VK_CHAIN_LAYERS=3
  CHAIN_SERVE_EXPECT='glm53: [0-9]+ dense matrices on the device only.*the 3 of 6 layers on the device' COLI_VK_DENSE_HOST=0 \
    CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=32 COLI_VK_CHAIN_LAYERS=3
  unset CHAIN_SERVE_EXPECT
  # a pin restored over rows another branch rewrote, 2 of 4 layers on the device
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_CHAIN_LAYERS=2 COLI_USAGE=$PWD/chain.usage $PY tests/glm53_pin_branch_harness.py --binary ./glm53 --fixture glm53_mm_tiny
  rm -rf glm_tiny_shx glm53_l6 glm53_l6s-i4 glm53_l6_serve chain.usage
  rm -f ptl-probe.err ptl-ram*.log same-*.log same-*.tok same-*.f32
  unset OMP_NUM_THREADS CAP_RAISE
}

# The same under ASan and UBSan (the glm-chain-sanitize family's build): no diagnostic, the
# chain ran with the N the line says (or handled the loss).
ptl_mla_san() {  # <engine> <tag> <want N> <env and argv...>
  local eng=$1 tag=$2 want=$3; shift 3
  rm -f chain.usage
  env COLI_USAGE=chain.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} "$@" > san.log 2>&1 || true
  rm -f chain.usage
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
  grep -q "$eng chain: \([1-9][0-9]* forwards\|the device was lost\)" san.log || [ "$want" = 0 ] || { cat san.log; fail "$tag: the chain never ran"; }
  ptl_check_n "$eng" san.log "$want" "$tag"
  ptl_check_placed "$eng" san.log "$tag"
  echo "OK $tag: sanitizers clean, N = $want, $(chain_count "$eng" san.log) chain forwards"
}
ptl_family_glm_sanitize() {
  make clean >/dev/null 2>&1 || true
  make colibri glm53 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ptl_glm_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  local G="SNAP=glm_tiny REF=ref_glm.json" X="SNAP=glm_tiny_shx REF=ref_glm.json" cap n ids F="--model glm53_l6s-i4"
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  ptl_mla_san colibri "asan partial colibri 2 of 5 layers" 2 COLI_VK_CHAIN_LAYERS=2 $G ./colibri 64 16 16
  ptl_mla_san colibri "asan partial colibri DSA top-4 handed over, chunks of 3" 1 COLI_VK_CHAIN_LAYERS=1 $X DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=3 ./colibri 64 16 16
  ptl_mla_run ptl-probe.err colibri COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04 $G -- 64 16 16
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap colibri ptl-probe.err 3)
  ptl_mla_san colibri "asan partial colibri a device for 3 layers" 3 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 $G ./colibri 64 16 16
  ptl_mla_run ptl-probe.err colibri COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=5 $G -- 64 16 16
  n=$(ptl_calc fault colibri ptl-probe.err 2)
  ptl_mla_san colibri "asan partial colibri an upload failing in layer 2" 2 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$n COLI_VK_CHAIN_LAYERS=5 $G ./colibri 64 16 16
  ptl_mla_san colibri "asan partial colibri device only, MTP" 2 COLI_VK_CHAIN_LAYERS=2 COLI_VK_DENSE_HOST=0 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=2 ./colibri 64 16 16
  ptl_mla_san colibri "asan partial colibri the KV split" 3 COLI_VK_CHAIN_LAYERS=3 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 $G ./colibri 64 16 16
  ptl_mla_san colibri "asan partial colibri device lost" 2 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_FAULT=12 $G ./colibri 64 16 16
  ptl_mla_san glm53 "asan partial glm53 3 of 6 layers, chain chunks of 3" 3 COLI_VK_CHAIN_LAYERS=3 GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3 ./glm53 $F --ids $ids --greedy 4
  ptl_mla_run ptl-probe.err glm53 COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04 GLM53_BITS=32 -- $F --ids $ids --greedy 4
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap glm53 ptl-probe.err 2)
  ptl_mla_san glm53 "asan partial glm53 a device for 2 layers" 2 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 GLM53_BITS=32 ./glm53 $F --ids $ids --greedy 4
  ptl_mla_run ptl-probe.err glm53 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:1000000 COLI_VK_CHAIN_LAYERS=6 GLM53_BITS=4 -- $F --ids $ids --greedy 4
  n=$(ptl_calc fault glm53 ptl-probe.err 3)
  ptl_mla_san glm53 "asan partial glm53 an upload failing in layer 3" 3 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=submit:$n COLI_VK_CHAIN_LAYERS=6 GLM53_BITS=4 ./glm53 $F --ids $ids --greedy 4
  ptl_mla_san glm53 "asan partial glm53 device only, an image" 2 COLI_VK_CHAIN_LAYERS=2 COLI_VK_DENSE_HOST=0 GLM53_BITS=32 ./glm53 --model glm53_mm_tiny \
    --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  ptl_mla_san glm53 "asan partial glm53 the KV split" 4 COLI_VK_CHAIN_LAYERS=4 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 GLM53_BITS=32 ./glm53 $F --ids $ids --greedy 6
  ptl_mla_san glm53 "asan partial glm53 device lost, rebuilt" 3 COLI_VK_CHAIN_LAYERS=3 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 COLI_VK_CHAIN_FAULT=8 ./glm53 $F --ids $ids --greedy 6
  local args
  for args in "./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2 COLI_VK_CHAIN_LAYERS=2" \
              "./glm53 glm53_l6_serve GLM53_BITS=32 KV_SLOTS=2 COLI_VK_CHAIN_LAYERS=3"; do
    # shellcheck disable=SC2086
    CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=$([ "${args%% *}" = ./colibri ] && echo colibri || echo numeric) \
      $PY tests/vulkan_chain_serve.py $args > san.log 2>&1 || { cat san.log; fail "asan partial serve $args"; }
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan partial serve $args: sanitizer diagnostic"; fi
    echo "OK asan partial serve ${args%% *}: $(tail -1 san.log | cut -c1-120)"
  done
  rm -rf glm_tiny_shx glm53_l6 glm53_l6s-i4 glm53_l6_serve chain.usage
  rm -f ptl-probe.err san.log
  unset OMP_NUM_THREADS CAP_RAISE
  make clean >/dev/null 2>&1 || true
}
