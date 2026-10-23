# Sourced by tests/vulkan_engines.sh: colibri (GLM-5.2) and glm53 (GLM-5.3 Flash) with
# their dense weights on the device only (COLI_VK_DENSE_HOST=0, docs/vulkan.md "Dense
# weights on the device only"). The fixtures and the token lines are the glm-chain
# family's (glm_chain_fixtures, mla_toks): every expert and trunk format, prefill in
# chunks, the DSA selection, n-gram and MTP drafts, the tier off, prompts only and the
# per-matrix path alone (the steps the chain does not run take the device: the CPU has no
# copy), the device lost (the matrices read back from the checkpoint, glm53's KDA state
# rebuilt), serve sessions frame for frame (two KV slots, prompts only), glm53's image and
# pin-branch harness, staged uploads, the RAM the experts' cache gets back, and the
# resident set at exit on a fixture whose head is large enough to see.

# dho_mla_gate <engine> <tag> <tol 0|1> <env...> -- <argv...>: the CPU, the device keeping
# its host copies (COLI_VK_DENSE_HOST=1) and the device only (=0), the tier deterministic
# and each run from a fresh history. Gates: the CPU's tokens in both device runs; matrices
# on the device only and none read back; chain forwards (CHAINMODE, default 1; 0 = the
# per-matrix path alone with COLI_VK_DENSE=1); tol 1: the device-only logits within 1e-4
# of the CPU's largest (logits_close).
dho_mla_gate() {
  local eng=$1 tag=$2 tol=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local mode=${CHAINMODE:-1}
  local vk=(USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VULKAN=1 COLI_VK_CHAIN=$mode)
  [ "$mode" = 0 ] && vk+=(COLI_VK_DENSE=1)
  rm -f chain.usage cpu.f32 vk.f32
  env "${envs[@]}" USAGE_SAVE=0 COLI_USAGE=chain.usage DUMP=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || true
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" COLI_VK_DENSE_HOST=1 ./"$eng" "$@" > host.log 2>&1 || true
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" COLI_VK_DENSE_HOST=0 DUMP=vk.f32 ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  mla_toks "$eng" cpu.log > cpu.tok; mla_toks "$eng" host.log > host.tok; mla_toks "$eng" vk.log > vk.tok
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the device-only run's tokens differ from the CPU"; }
  cmp -s cpu.tok host.tok || { cat cpu.tok host.tok; tail -20 host.log; fail "$tag: the host-copy run's tokens differ from the CPU"; }
  local n; n=$(dho_dropped vk.log)
  [ "$n" -gt 0 ] || { cat vk.log; fail "$tag: no dense matrix went to the device only"; }
  [ "$(dho_reloaded vk.log)" = 0 ] || { grep -a '^\[VK\]' vk.log; fail "$tag: a matrix was read back from disk without a lost device"; }
  if [ "$mode" != 0 ]; then [ "$(chain_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the chain did not stay on"; }; fi
  local lg=""
  if [ "$tol" = 1 ]; then lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }; lg=", $lg"; fi
  echo "OK $tag: tokens = CPU$lg, $n matrices on the device only, $(chain_count "$eng" vk.log) chain forwards"
}
# dho_mla_lost <engine> <tag> <frames back> <env...> -- <argv...>: device only, the device
# lost `back` frames before the end of a run (COLI_VK_CHAIN_FAULT, counted on a fault-free
# run of the same configuration): the CPU's tokens, the loss handled, matrices read back
# from disk for the CPU; REBUILD=1: glm53's KDA state rebuilt on the CPU.
dho_mla_lost() {
  local eng=$1 tag=$2 back=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local vk=(USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0)
  rm -f chain.usage
  env "${envs[@]}" USAGE_SAVE=0 COLI_USAGE=chain.usage ./"$eng" "$@" > cpu.log 2>&1 || true
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" ./"$eng" "$@" > vk.log 2>&1 || true
  local f; f=$(sed -n "s/^\[VK\] $eng chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p" vk.log | tail -1)
  { [ -n "$f" ] && [ "$f" -gt "$back" ]; } || { cat vk.log; fail "$tag: no fault-free run to count frames from"; }
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" COLI_VK_CHAIN_FAULT=$((f - back + 1)) ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  mla_toks "$eng" cpu.log > cpu.tok; mla_toks "$eng" vk.log > vk.tok
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the tokens after the loss differ from the CPU"; }
  grep -q "$eng chain: the device was lost" vk.log || { cat vk.log; fail "$tag: no loss was handled"; }
  if [ "${REBUILD:-0}" = 1 ]; then grep -q "rebuilding the state of [1-9]" vk.log || { cat vk.log; fail "$tag: no state was rebuilt"; }; fi
  local r; r=$(dho_reloaded vk.log)
  [ "$r" -gt 0 ] || { grep -a '^\[VK\]' vk.log; fail "$tag: nothing was read back for the CPU"; }
  echo "OK $tag: tokens = CPU after the loss, $r matrices read back from disk$(grep -o 'rebuilding the state of [0-9]* positions' vk.log | sed 's/^/, /')"
}
# dho_widen <src> <dst> <vocab>: a copy of a fixture whose embedding and head have `vocab`
# rows (the rows past the original vocabulary repeat the first ones): a head of tens of MiB
# whose host copy shows in the resident set; the prompts keep their ids, so the runs are
# the original's model with a wider (never chosen) vocabulary tail.
dho_widen() {
  $PY - "$1" "$2" "$3" <<'PY'
import json, shutil, struct, sys
from pathlib import Path
src, dst, vocab = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
cfg = json.loads((dst / "config.json").read_text())
(cfg["text_config"] if "text_config" in cfg else cfg)["vocab_size"] = vocab
(dst / "config.json").write_text(json.dumps(cfg))
for shard in sorted(dst.glob("*.safetensors")):
    raw = shard.read_bytes(); n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n]); body = raw[8 + n:]
    meta = head.pop("__metadata__", None)
    out, off, blobs = {}, 0, []
    for name, t in sorted(head.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a, b = t["data_offsets"]; data = body[a:b]; shape = list(t["shape"])
        if (name.endswith("embed_tokens.weight") or name == "lm_head.weight") and t["dtype"] == "F32":
            row = len(data) // shape[0]
            data = b"".join(data[(r % shape[0]) * row:(r % shape[0] + 1) * row] for r in range(vocab))
            shape[0] = vocab
        out[name] = {"dtype": t["dtype"], "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data); off += len(data)
    if meta is not None: out["__metadata__"] = meta
    h = json.dumps(out).encode(); h += b" " * (-len(h) % 8)
    shard.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
PY
}

dho_family_glm() {
  make colibri glm53 VK=1
  glm_chain_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  local cap b fx d
  # colibri: the f32 trunk and experts, 8/4/3-bit trunks (IDOT=0: the device multiplies f32
  # activations), the expert containers, prefill, the DSA selection, drafts
  for cap in 1 64; do dho_mla_gate colibri "dense-only colibri f32 cap=$cap" 1 SNAP=glm_tiny REF=ref_glm.json -- $cap 16 16; done
  dho_mla_gate colibri "dense-only colibri f32 prefill, chunks of 3" 1 SNAP=glm_tiny REF=ref_glm.json TF=1 COLI_VK_CHAIN_ROWS=3 -- 64 16 16
  for b in 8 4 3; do
    dho_mla_gate colibri "dense-only colibri ${b}-bit trunk and experts" 1 SNAP=glm_tiny REF=ref_glm.json IDOT=0 -- 2 $b $b
  done
  dho_mla_gate colibri "dense-only colibri int4-g64 experts" 1 SNAP=glm_tiny_fmt4 REF=glm_tiny_fmt4/ref_glm.json -- 2 16 16
  dho_mla_gate colibri "dense-only colibri E8/IQ3 experts on the CPU" 1 SNAP=glm_tiny_fmt6 REF=glm_tiny_fmt6/ref_glm.json -- 2 16 16
  for fx in i4 i4r i3 d3; do
    dho_mla_gate colibri "dense-only colibri $fx container" 1 SNAP=glm_tiny_$fx REF=glm_tiny_$fx/ref_glm.json IDOT=0 -- 1 4 4
  done
  dho_mla_gate colibri "dense-only colibri DSA top-4" 1 SNAP=glm_tiny REF=ref_glm.json DSA_TOPK=4 -- 64 16 16
  dho_mla_gate colibri "dense-only colibri DSA top-4, prefill in chunks of 5" 1 SNAP=glm_tiny REF=ref_glm.json DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  dho_mla_gate colibri "dense-only colibri n-gram drafts" 1 SNAP=glm_tiny REF=ref_glm.json DRAFT=3 -- 64 16 16
  # the MTP head runs per matrix: its layer, eh_proj and lm_head take the device here
  for d in 1 3; do dho_mla_gate colibri "dense-only colibri MTP depth $d" 1 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=$d -- 64 16 16; done
  dho_mla_gate colibri "dense-only colibri MTP, 4-bit" 1 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json IDOT=0 DRAFT=2 -- 2 4 4
  dho_mla_gate colibri "dense-only colibri tier off" 1 SNAP=glm_tiny REF=ref_glm.json COLI_VK_TIER=0 -- 64 16 16
  dho_mla_gate colibri "dense-only colibri beside COLI_VK_ATTN=1" 1 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 COLI_VK_ATTN=1 -- 2 4 4
  # prompts only: decode runs per matrix on the device; kv_b and the indexer stay on the
  # host (the CPU's attention reads them by row)
  CHAINMODE=2 dho_mla_gate colibri "dense-only colibri prompts only, drafts" 1 SNAP=glm_tiny REF=ref_glm.json DRAFT=3 -- 64 16 16
  CHAINMODE=0 dho_mla_gate colibri "dense-only colibri per-matrix path, absorb core" 1 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 COLI_VK_ATTN=1 -- 2 4 4
  CHAINMODE=0 dho_mla_gate colibri "dense-only colibri per-matrix path, MTP" 1 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=2 -- 64 16 16
  # staged uploads (a card without Resizable BAR): the drop happens after the copy returned
  COLI_VK_STAGED=1 dho_mla_gate colibri "dense-only colibri staged" 1 SNAP=glm_tiny REF=ref_glm.json DSA_TOPK=4 -- 64 16 16
  grep -q '^\[VK\] memory: staged uploads' vk.log || { grep '^\[VK\]' vk.log; fail "dense-only colibri staged: not staged"; }
  # the device lost: mid-decode, in the prompt (one teacher-forced forward), with MTP drafts
  dho_mla_lost colibri "dense-only colibri device lost mid-decode" 12 SNAP=glm_tiny REF=ref_glm.json -- 64 16 16
  dho_mla_lost colibri "dense-only colibri device lost in the prompt" 3 SNAP=glm_tiny REF=ref_glm.json TF=1 -- 64 16 16
  dho_mla_lost colibri "dense-only colibri device lost with MTP" 12 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json -- 64 16 16
  COLI_VK_STAGED=1 dho_mla_lost colibri "dense-only colibri device lost, staged" 12 SNAP=glm_tiny REF=ref_glm.json -- 64 16 16
  # serve sessions: pins, the prompt cache, the prefill read-out, drafts, two KV slots,
  # several conversations at once
  export COLI_VK_DENSE_HOST=0 CHAIN_SERVE_EXPECT='dense matrices on the device only'
  CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0
  CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4 DRAFT=3
  CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2
  COLI_VK_CHAIN=2 CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4
  CHAIN_MUX_EXPECT="$CHAIN_SERVE_EXPECT" $PY tests/vulkan_chain_mux.py ./colibri glm_tiny_serve 3 SERVE_BATCH=1 IDOT=0 DSA_TOPK=4
  # the steps the chain does not take (MUX=0): per matrix on the device, kv_b on the host
  CHAIN_MUX_EXPECT="$CHAIN_SERVE_EXPECT" CHAIN_MUX_STEPS=0 $PY tests/vulkan_chain_mux.py ./colibri glm_tiny_serve 3 SERVE_BATCH=1 IDOT=0 COLI_VK_CHAIN_MUX=0
  rm -f chain-mux.usage
  unset COLI_VK_DENSE_HOST CHAIN_SERVE_EXPECT
  # the RAM the experts' cache gets back: RAM_GB just above the fixture's fixed reserve, the
  # cap cap_for_ram leaves with the host copies against the one it leaves without them
  local c1 c0 h
  for h in 1 0; do
    rm -f chain.usage
    env RAM_GB=3.7145 SNAP=glm_tiny REF=ref_glm.json USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
      COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=$h ./colibri 64 16 16 > vk$h.log 2>&1 || true
  done
  rm -f chain.usage
  c1=$(sed -n 's/.*cap lowered 64->\([0-9]*\).*/\1/p' vk1.log | tail -1); c0=$(sed -n 's/.*cap lowered 64->\([0-9]*\).*/\1/p' vk0.log | tail -1)
  { [ -n "$c1" ] && [ -n "$c0" ] && [ "$c0" -gt "$c1" ]; } || { grep -a 'RAM_GB\|resident dense' vk1.log vk0.log; fail "dense-only colibri RAM_GB: the cap did not grow"; }
  mla_toks colibri vk1.log > host.tok; mla_toks colibri vk0.log > vk.tok
  cmp -s host.tok vk.tok || fail "dense-only colibri RAM_GB: the tokens differ between the two caps"
  echo "OK dense-only colibri RAM_GB=3.7145: cap $c1 with the host copies, $c0 on the device only ($(grep -ao 'resident dense: [0-9.]* MB' vk1.log) -> $(grep -ao '[0-9.]* MB' <<< "$(grep -ao 'resident dense: [0-9.]* MB' vk0.log)"))"
  # the resident set at exit: a 65536-row f32 head (32 MiB) on the device only
  dho_widen glm_tiny glm_tiny_wide 65536
  dho_rss_gate colibri "dense-only colibri resident set" 24 SNAP=glm_tiny_wide REF=ref_glm.json USAGE_SAVE=0 -- 64 16 16
  # unset, on a CPU device the host copies stay (the auto rule), and the line says why
  rm -f chain.usage
  env SNAP=glm_tiny REF=ref_glm.json USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VULKAN=1 COLI_VK_CHAIN=1 ./colibri 64 16 16 > vk.log 2>&1 || true
  rm -f chain.usage
  grep -q '^\[VK\] colibri: dense weights on the device and in host RAM (a CPU device' vk.log || { grep '^\[VK\]' vk.log; fail "colibri: the auto rule dropped the host copies on a CPU device"; }
  [ "$(dho_dropped vk.log)" = 0 ] || fail "colibri: matrices dropped without COLI_VK_DENSE_HOST=0 on a CPU device"
  echo "OK dense-only colibri auto on a CPU device: host copies kept"

  # glm53: f32, int8 and int4 trunks, streamed and resident experts, prefill and chain
  # chunks, the tier off, an image, swiglu_limit 0, prompts only, the per-matrix path alone
  local ids bits
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  dho_mla_gate glm53 "dense-only glm53 decode" 1 GLM53_BITS=32 -- --model glm53_stream-i4 --ids 5,7,9,11,13,17,19,23 --greedy 8
  for bits in 32 8 4; do
    dho_mla_gate glm53 "dense-only glm53 100-token prompt, ${bits}-bit trunk" 1 GLM53_BITS=$bits -- --model glm53_stream-i4 --ids $ids --greedy 4
  done
  dho_mla_gate glm53 "dense-only glm53 prefill chunks of 7, one cache slot" 1 GLM53_BITS=4 GLM53_PREFILL_CHUNK=7 GLM53_EXPERT_GB=0.000001 -- --model glm53_stream-i4 --ids $ids --greedy 4
  dho_mla_gate glm53 "dense-only glm53 chain chunks of 3" 1 GLM53_BITS=32 COLI_VK_CHAIN_ROWS=3 -- --model glm53_stream-i4 --ids $ids --greedy 4
  dho_mla_gate glm53 "dense-only glm53 resident experts" 1 GLM53_BITS=32 -- --model glm53_tiny --ids $ids --greedy 6
  dho_mla_gate glm53 "dense-only glm53 tier off" 1 GLM53_BITS=32 COLI_VK_TIER=0 -- --model glm53_stream-i4 --ids $ids --greedy 6
  dho_mla_gate glm53 "dense-only glm53 an image" 1 GLM53_BITS=32 -- --model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  dho_mla_gate glm53 "dense-only glm53 swiglu_limit 0" 1 GLM53_BITS=32 -- --model glm53_lim0 --ids $ids --greedy 4
  CHAINMODE=2 dho_mla_gate glm53 "dense-only glm53 prompts only" 1 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- --model glm53_stream-i4 --ids $ids --greedy 6
  CHAINMODE=0 dho_mla_gate glm53 "dense-only glm53 per-matrix path, 4-bit trunk" 1 GLM53_BITS=4 -- --model glm53_stream-i4 --ids $ids --greedy 6
  COLI_VK_STAGED=1 dho_mla_gate glm53 "dense-only glm53 staged" 1 GLM53_BITS=4 -- --model glm53_stream-i4 --ids $ids --greedy 4
  grep -q '^\[VK\] memory: staged uploads' vk.log || { grep '^\[VK\]' vk.log; fail "dense-only glm53 staged: not staged"; }
  REBUILD=1 dho_mla_lost glm53 "dense-only glm53 device lost, the KDA state rebuilt" 7 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- --model glm53_stream-i4 --ids $ids --greedy 6
  REBUILD=1 dho_mla_lost glm53 "dense-only glm53 device lost, 4-bit trunk" 7 GLM53_BITS=4 GLM53_PREFILL_CHUNK=16 -- --model glm53_stream-i4 --ids $ids --greedy 6
  REBUILD=1 dho_mla_lost glm53 "dense-only glm53 device lost after an image" 5 GLM53_BITS=32 -- --model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  export COLI_VK_DENSE_HOST=0 CHAIN_SERVE_EXPECT='dense matrices on the device only'
  CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32
  CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3
  CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32 KV_SLOTS=2
  COLI_VK_CHAIN=2 CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32
  unset CHAIN_SERVE_EXPECT
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_USAGE=$PWD/chain.usage COLI_VK_TIER=0 $PY tests/glm53_pin_branch_harness.py --binary ./glm53 --fixture glm53_mm_tiny --tol 1e-5
  unset COLI_VK_DENSE_HOST
  # the expert cache sizes itself from the free memory after the host copies went (on a
  # device that shares the RAM the device's copy takes the same RAM: no gain to show there;
  # on a discrete GPU the RAM given back is free memory the cache measures)
  rm -f chain.usage
  env GLM53_VERBOSE=1 GLM53_BITS=32 USAGE_SAVE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    COLI_VK_DENSE_HOST=0 ./glm53 --model glm53_stream-i4 --ids 5,7,9 --greedy 2 > vk.log 2>&1 || true
  rm -f chain.usage
  local given budget
  given=$(grep -n 'dense matrices on the device only' vk.log | head -1 | cut -d: -f1); budget=$(grep -n '^expert budget:' vk.log | head -1 | cut -d: -f1)
  { [ -n "$given" ] && [ -n "$budget" ] && [ "$given" -lt "$budget" ]; } || { cat vk.log; fail "dense-only glm53: the expert cache sized itself before the host copies went"; }
  echo "OK dense-only glm53: the expert cache sized after the host copies went ($(grep -a '^expert budget:' vk.log | head -1))"
  # The log rounds RSS to 0.01 GiB. A 64 MiB head leaves enough separation
  # for that rounding and allocator bookkeeping without weakening the gate.
  dho_widen glm53_stream-i4 glm53_wide 131072
  dho_rss_gate glm53 "dense-only glm53 resident set" 48 GLM53_BITS=32 USAGE_SAVE=0 -- --model glm53_wide --ids 5,7,9,11,13 --greedy 4
  unset OMP_NUM_THREADS CAP_RAISE
}

# The same under ASan and UBSan: memory safety is the gate, with the host copies dropped,
# the read-back after a lost device, the CPU's attention reading kv_b back (prompts only),
# drafts, an image and serve sessions.
dho_family_glm_sanitize() {
  make clean >/dev/null 2>&1 || true
  make colibri glm53 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  glm_chain_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  dsan() {  # <engine> <tag> <lost 0|1> <env and argv...>
    local eng=$1 tag=$2 lost=$3; shift 3
    rm -f chain.usage
    env COLI_USAGE=chain.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} COLI_VK_DENSE_HOST=0 "$@" > san.log 2>&1 || true
    rm -f chain.usage
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(dho_dropped san.log)" -gt 0 ] || { cat san.log; fail "$tag: nothing on the device only"; }
    if [ "$lost" = 1 ]; then [ "$(dho_reloaded san.log)" -gt 0 ] || { cat san.log; fail "$tag: nothing read back"; }; fi
    echo "OK $tag: sanitizers clean, $(dho_dropped san.log) matrices on the device only, $(dho_reloaded san.log) read back"
  }
  local ids
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  dsan colibri "asan dense-only colibri f32, DSA top-4 prefill" 0 SNAP=glm_tiny REF=ref_glm.json TF=1 DSA_TOPK=4 COLI_VK_CHAIN_ROWS=3 ./colibri 64 16 16
  dsan colibri "asan dense-only colibri i4 container" 0 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 ./colibri 2 4 4
  dsan colibri "asan dense-only colibri MTP depth 2" 0 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=2 ./colibri 64 16 16
  CHAINMODE=2 dsan colibri "asan dense-only colibri prompts only" 0 SNAP=glm_tiny REF=ref_glm.json DRAFT=3 ./colibri 64 16 16
  dsan colibri "asan dense-only colibri device lost" 1 SNAP=glm_tiny REF=ref_glm.json COLI_VK_CHAIN_FAULT=30 ./colibri 64 16 16
  dsan glm53 "asan dense-only glm53 int4 trunk, chain chunks of 3" 0 GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 4
  dsan glm53 "asan dense-only glm53 an image" 0 GLM53_BITS=32 ./glm53 --model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  CHAINMODE=2 dsan glm53 "asan dense-only glm53 prompts only" 0 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 6
  dsan glm53 "asan dense-only glm53 device lost, rebuilt" 1 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 COLI_VK_CHAIN_FAULT=20 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 6
  local args
  for args in "./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2" "./glm53 glm53_serve GLM53_BITS=32 KV_SLOTS=2"; do
    local dialect=colibri; [ "${args%% *}" = ./glm53 ] && dialect=numeric
    # shellcheck disable=SC2086
    COLI_VK_DENSE_HOST=0 CHAIN_SERVE_EXPECT='dense matrices on the device only' CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=$dialect \
      $PY tests/vulkan_chain_serve.py $args > san.log 2>&1 || { cat san.log; fail "asan dense-only serve ${args%% *}"; }
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan dense-only serve ${args%% *}: sanitizer diagnostic"; fi
    echo "OK asan dense-only serve ${args%% *}: $(tail -1 san.log)"
  done
  unset OMP_NUM_THREADS CAP_RAISE
  make clean >/dev/null 2>&1 || true
}
