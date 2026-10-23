# Sourced by tests/vulkan_engines.sh: qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B dense,
# Clef) and olmoe with their dense matrices on the device only (COLI_VK_DENSE_HOST=0,
# docs/vulkan.md "Dense weights on the device only").
#
# qwen36: f32, int8 and int4-g64 dense rows, int4-g64 experts, the Qwen3-Coder and the
# dense 27B geometries, prefill in chunks, the tier off, prompts only and the per-matrix
# path alone (the steps the chain does not run take the device too: the CPU has no copy),
# an image, the device lost (the DeltaNet state rebuilt on the CPU with every matrix read
# back from disk), a serve session frame for frame, and Clef's oracle.
# olmoe: caps 1 and 8, PILOT, the device lost (the step redone on the CPU from matrices
# read back), a serve session, an automatic cache that takes the RAM the dense weights
# held, and the resident set falling by what was dropped (a wide fixture).

# A wide OLMoE: a 32768-row lm_head of 512 columns (64 MiB of f32) and small experts, so
# the dense weights are most of the process and an expert slot is small beside them.
dho_olmoe_wide_fixture() {
  [ -f olmoe_wide_c/config.json ] && return 0
  rm -rf olmoe_wide olmoe_wide_c
  $PY - <<'PY'
import json, torch
from pathlib import Path
from transformers import OlmoeConfig, OlmoeForCausalLM
torch.manual_seed(20261004); torch.set_num_threads(2)
cfg = OlmoeConfig(vocab_size=32768, hidden_size=512, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=8, num_key_value_heads=8, max_position_embeddings=128,
                  num_experts=64, num_experts_per_tok=2, norm_topk_prob=True,
                  eos_token_id=None, pad_token_id=None, tie_word_embeddings=False)
model = OlmoeForCausalLM(cfg).eval().float()
prompt = [3, 11, 29, 7, 41, 19]
with torch.no_grad():
    full = model.generate(torch.tensor([prompt]), max_new_tokens=6, do_sample=False, use_cache=True)[0].tolist()
out = Path("olmoe_wide"); out.mkdir()
model.save_pretrained(out, safe_serialization=True)
(out / "ref_olmoe.json").write_text(json.dumps({"prompt_ids": prompt, "full_ids": full}) + "\n")
PY
  $PY tools/convert_olmoe_merged.py --model olmoe_wide --out olmoe_wide_c > /dev/null
}

dho_qwen36_olmoe_fixtures() {
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "tests"); from prefix_serve_harness import ensure_byte_tokenizer; ensure_byte_tokenizer(Path("qwen36_tiny_c"))'   # serve speaks text
  $PY tools/make_qwen36_tiny.py --geometry qwen3-coder-30b --out qwen3_coder_tiny --ref-mode full --emit-ref qwen3_coder_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen3_coder_tiny --out qwen3_coder_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen38_27b_tiny --out qwen38_27b_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/make_qwen36_vl_tiny.py --out qwen38_27b_vl_tiny
  $PY tools/convert_qwen36.py --model qwen38_27b_vl_tiny --out qwen38_27b_vl_tiny_c --ebits 8
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny --force
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 128 olmoe_tiny_c
  dho_olmoe_wide_fixture
}

# dho_olmoe_cap_gate: olmoe's automatic cache (cap 0) under a RAM_GB that leaves the
# experts half of their bytes beside the dense weights (the room is RAM_GB less what is
# resident less 0.5 GB; the KV is not set aside): with the host copies kept it stays
# below every expert of a layer, on the device only it takes them all (the 64 MiB the
# dense weights held hold them many times over).
dho_olmoe_cap_gate() {
  local R=olmoe_wide/ref_olmoe.json
  SNAP=olmoe_wide_c ./olmoe 8 8 $R > cpu.log 2>&1 || true
  local rss half ram
  rss=$(sed -n 's/.*RSS after load: \([0-9.]*\) GB.*/\1/p' cpu.log | tail -1)
  [ -n "$rss" ] || { cat cpu.log; fail "olmoe automatic cache: no RSS after load"; }
  half=$($PY -c 'import json; c=json.load(open("olmoe_wide_c/config.json")); h,i=c["hidden_size"],c["intermediate_size"]; print((h*i*3+(2*i+h)*4)*c["num_experts"]*c["num_hidden_layers"]/2/1e9)')
  ram=$($PY -c "print(f'{$rss + 0.5 + $half:.4f}')")
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 RAM_GB=$ram SNAP=olmoe_wide_c)
  rm -f chain.usage
  env "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_DENSE_HOST=1 ./olmoe 0 8 $R > host.log 2>&1 || true
  rm -f chain.usage
  env "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_DENSE_HOST=0 ./olmoe 0 8 $R > vk.log 2>&1 || true
  rm -f chain.usage
  same_tokens cpu.log vk.log "olmoe automatic cache"
  local a b
  a=$(sed -n 's/^\[cache\] \([0-9]*\) slots\/layer of .*/\1/p' host.log | tail -1)
  b=$(sed -n 's/^\[cache\] \([0-9]*\) slots\/layer of .*/\1/p' vk.log | tail -1)
  { [ -n "$a" ] && [ -n "$b" ] && [ "$b" -gt "$a" ]; } || { grep '\[cache\]' host.log vk.log; fail "olmoe automatic cache: the cache did not grow on the device only"; }
  grep -q '^\[cache\] .* (was [0-9]*): the dense weights on the device only gave' vk.log || { grep '\[cache\]' vk.log; fail "olmoe automatic cache: no line for the growth"; }
  echo "OK dense-only olmoe automatic cache, RAM_GB=$ram: $a slots/layer with the host copies, $b on the device only"
}

dho_family_qwen36_olmoe() {
  export OMP_NUM_THREADS=2
  make qwen36 olmoe VK=1
  dho_qwen36_olmoe_fixtures
  local fx R=qwen36_tiny/ref_full.json
  # qwen36: every geometry with f32 dense rows, then int8 rows (fmt 1; the CPU arm with
  # f32 activations, as the device) and the int4-g64 copy (COLI_DENSE_BITS=4, fmt 4)
  for fx in qwen36_tiny qwen3_coder_tiny qwen38_27b_tiny; do
    dho_gate qwen36 "dense-only qwen36 $fx f32" 2 COLI_DENSE_I8=0 SNAP=${fx}_c -- 8 8 $fx/ref_full.json
    CPUENV=COLI_DENSE_IDOT=0 dho_gate qwen36 "dense-only qwen36 $fx int8" 2 SNAP=${fx}_c -- 8 8 $fx/ref_full.json
  done
  dho_gate qwen36 "dense-only qwen36 int4-g64 dense" 2 COLI_DENSE_BITS=4 SNAP=qwen36_tiny64_c -- 8 4 qwen36_tiny64/ref_full.json
  dho_gate qwen36 "dense-only qwen36 int4-g64 experts" 2 QWEN_EXPERT_ACT=f32 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- 8 4 qwen36_tiny64/ref_full.json
  dho_gate qwen36 "dense-only qwen36 cap 1" 2 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 1 8 $R
  dho_gate qwen36 "dense-only qwen36 prefill in chunks of 3" 2 COLI_VK_CHAIN_ROWS=3 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 $R
  dho_gate qwen36 "dense-only qwen36 tier off" 2 COLI_VK_TIER=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 $R
  # prompts on the chain, decode on the per-matrix path (the device's in this mode, the
  # CPU's with the host copies kept on this device): the CPU's logits
  CHAINMODE=2 dho_gate qwen36 "dense-only qwen36 prompts only" 1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 $R
  CHAINMODE=0 dho_gate qwen36 "dense-only qwen36 per-matrix path" 2 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 $R
  # PILOT keeps the chain off: every matrix per matrix on the device
  CHAINMODE=0 dho_gate qwen36 "dense-only qwen36 PILOT" 2 PILOT=1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 $R
  dho_gate qwen36 "dense-only qwen36 vision" 1 COLI_DENSE_I8=0 SNAP=qwen38_27b_vl_tiny_c -- 8 8 qwen38_27b_vl_tiny/ref.json
  # f16 acquired a Vulkan form with the decision models. Import requested alongside
  # host removal must copy first: the device cannot retain pages that the engine frees.
  dho_gate qwen36 "dense-only qwen36 f16 dense" 2 COLI_DENSE_BITS=16 COLI_VK_IMPORT=1 SNAP=qwen36_tiny_c -- 8 8 $R
  dho_gate qwen36 "dense-only qwen36 int8 import requested" 2 COLI_DENSE_IDOT=0 COLI_VK_IMPORT=1 SNAP=qwen36_tiny_c -- 8 8 $R
  dho_lost qwen36 "dense-only qwen36 f16 device lost" 20 COLI_DENSE_BITS=16 COLI_VK_IMPORT=1 SNAP=qwen36_tiny_c -- 8 8 $R
  dho_lost qwen36 "dense-only qwen36 device lost" 20 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 $R
  dho_lost qwen36 "dense-only qwen36 int8 device lost" 20 COLI_DENSE_IDOT=0 SNAP=qwen36_tiny_c -- 8 8 $R
  CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0
  # Clef's oracle (the decision head on the CPU, the backbone on the device only)
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage \
    make clef-tiny-check VK=1
  rm -f chain.usage chain-serve.usage

  # olmoe
  local OR=olmoe_tiny/ref_olmoe.json cap
  for cap in 1 8; do dho_gate olmoe "dense-only olmoe cap=$cap" 2 SNAP=olmoe_tiny_c -- $cap 8 $OR; done
  dho_gate olmoe "dense-only olmoe PILOT" 2 PILOT=1 WIDE=2 SNAP=olmoe_tiny_c -- 1 8 $OR
  dho_gate olmoe "dense-only olmoe prefill in chunks of 3" 2 COLI_VK_CHAIN_ROWS=3 SNAP=olmoe_tiny_c -- 8 8 $OR
  CHAINMODE=2 dho_gate olmoe "dense-only olmoe prompts only" 1 SNAP=olmoe_tiny_c -- 8 8 $OR
  CHAINMODE=0 dho_gate olmoe "dense-only olmoe per-matrix path" 2 SNAP=olmoe_tiny_c -- 8 8 $OR
  dho_lost olmoe "dense-only olmoe device lost" 10 SNAP=olmoe_tiny_c -- 8 8 $OR
  CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c
  dho_gate olmoe "dense-only olmoe wide" 2 SNAP=olmoe_wide_c -- 8 8 olmoe_wide/ref_olmoe.json
  dho_olmoe_cap_gate
  # the RSS: on Lavapipe the device copies are this process's memory too, so the
  # resident set falls by the host copies alone (64 MiB of lm_head, 4 MiB of layers)
  dho_rss_gate olmoe "dense-only olmoe RSS" 48 SNAP=olmoe_wide_c -- 8 8 olmoe_wide/ref_olmoe.json
  rm -f chain.usage chain-serve.usage
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan: memory safety is the gate, with the host copies dropped,
# a lost device's read-back, PILOT, the grown automatic cache and a serve session each.
dho_family_qwen36_olmoe_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 olmoe VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "tests"); from prefix_serve_harness import ensure_byte_tokenizer; ensure_byte_tokenizer(Path("qwen36_tiny_c"))'
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny --force
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 128 olmoe_tiny_c
  local tag args eng R=qwen36_tiny/ref_full.json OR=olmoe_tiny/ref_olmoe.json
  for tag in "qwen36 int8" "qwen36 int4-g64 dense" "qwen36 PILOT" "qwen36 device lost" \
             "olmoe cap 1" "olmoe PILOT" "olmoe device lost" "olmoe automatic cache"; do
    case $tag in
      "qwen36 int8") args=(SNAP=qwen36_tiny_c ./qwen36 8 8 $R) ;;
      "qwen36 int4-g64 dense") args=(COLI_DENSE_BITS=4 SNAP=qwen36_tiny64_c ./qwen36 8 4 qwen36_tiny64/ref_full.json) ;;
      "qwen36 PILOT") args=(PILOT=1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R) ;;
      "qwen36 device lost") args=(COLI_VK_CHAIN_FAULT=60 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 $R) ;;
      "olmoe cap 1") args=(SNAP=olmoe_tiny_c ./olmoe 1 8 $OR) ;;
      "olmoe PILOT") args=(PILOT=1 WIDE=2 SNAP=olmoe_tiny_c ./olmoe 1 8 $OR) ;;
      "olmoe device lost") args=(COLI_VK_CHAIN_FAULT=30 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR) ;;
      "olmoe automatic cache") args=(RAM_GB=0.7 SNAP=olmoe_tiny_c ./olmoe 0 8 $OR) ;;
    esac
    rm -f chain.usage
    env COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 "${args[@]}" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan dense-only $tag: sanitizer diagnostic"; fi
    [ "$(dho_dropped san.log)" -gt 0 ] || { cat san.log; fail "asan dense-only $tag: nothing on the device only"; }
    case $tag in *lost) [ "$(dho_reloaded san.log)" -gt 0 ] || { cat san.log; fail "asan dense-only $tag: nothing read back"; } ;; esac
    echo "OK asan dense-only $tag: sanitizers clean, $(dho_dropped san.log) matrices on the device only, $(dho_reloaded san.log) read back"
  done
  for eng in qwen36 olmoe; do
    local snap=qwen36_tiny_c; [ $eng = olmoe ] && snap=olmoe_tiny_c
    CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
      $PY tests/vulkan_chain_serve.py ./$eng $snap COLI_DENSE_I8=0 > san.log 2>&1 || { cat san.log; fail "asan dense-only $eng serve"; }
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan dense-only $eng serve: sanitizer diagnostic"; fi
    echo "OK asan dense-only $eng serve: $(tail -1 san.log)"
  done
  rm -f chain.usage chain-serve.usage
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
