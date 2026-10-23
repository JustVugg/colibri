# Sourced by tests/vulkan_engines.sh: qwen38 (Qwen3.8 Flash Next) with its trunk on the
# device only (COLI_VK_DENSE_HOST=0, docs/vulkan.md "Dense weights on the device only").
# Every resident format, the int4-g64 experts, prefill in chunks, the tier off, an MTP
# verify, prompts only and the per-matrix path alone (the steps the chain does not run
# take the device too: the CPU has no copy), the device lost (the trunk read back from
# the checkpoint, the CPU's tokens from there), and a serve session frame for frame.
dho_family_qwen38() {
  export OMP_NUM_THREADS=2
  make qwen38 VK=1
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4 --fp8-experts --int4-experts --expert-gain 3
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_mtp --mtp
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_tiny_mtp   # serve speaks text
  local bf16
  for bf16 in 0 1; do
    dho_gate qwen38 "dense-only qwen38 bf16=$bf16" 2 Q38_NATIVE_BF16=$bf16 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  done
  # the int8 trunk (fmt 1): every matrix of the tiny model through the int8 conversion
  dho_gate qwen38 "dense-only qwen38 int8 trunk" 2 Q38_TRUNK_MIN_KB=0 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  dho_gate qwen38 "dense-only qwen38 int4-g64 experts" 2 SNAP=qwen38_tiny_int4 -- 2 8 qwen38_tiny_int4/ref_int4.json
  dho_gate qwen38 "dense-only qwen38 prefill in chunks of 3" 2 COLI_VK_CHAIN_ROWS=3 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  dho_gate qwen38 "dense-only qwen38 tier off" 2 COLI_VK_TIER=0 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  # the MTP head's matrices run per matrix on the device here (on the CPU with the host
  # copies kept on this device): tokens and the CPU's logits
  dho_gate qwen38 "dense-only qwen38 MTP mixed" 1 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  # prompts on the chain, decode on the per-matrix path: the device's in this mode
  CHAINMODE=2 dho_gate qwen38 "dense-only qwen38 prompts only" 1 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  CHAINMODE=0 dho_gate qwen38 "dense-only qwen38 per-matrix path" 2 Q38_TRUNK_MIN_KB=0 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  # the device lost mid-decode and inside an MTP verify: the trunk comes back from disk
  dho_lost qwen38 "dense-only qwen38 device lost" 10 Q38_TRUNK_MIN_KB=0 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  dho_lost qwen38 "dense-only qwen38 device lost in a verify" 15 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1
  # unset on a CPU device the host copies stay (the auto rule), and the line says why
  rm -f chain.usage
  env COLI_USAGE=chain.usage COLI_VULKAN=1 COLI_VK_CHAIN=1 SNAP=qwen38_tiny ./qwen38 4 8 qwen38_tiny/ref.json > vk.log 2>&1
  grep -q '^\[VK\] qwen38: dense weights on the device and in host RAM (a CPU device' vk.log || { grep '^\[VK\]' vk.log; fail "qwen38: the auto rule dropped the host copies on a CPU device"; }
  [ "$(dho_dropped vk.log)" = 0 ] || fail "qwen38: matrices dropped without COLI_VK_DENSE_HOST=0 on a CPU device"
  echo "OK dense-only qwen38 auto on a CPU device: host copies kept"
  rm -f chain.usage
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan: memory safety is the gate, with the host copies dropped,
# a lost device's read-back and a serve session.
dho_family_qwen38_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen38 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_mtp --mtp
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_tiny_mtp
  local tag args f
  for tag in "int8 trunk" "MTP mixed" "device lost"; do
    case $tag in
      "int8 trunk") args=(Q38_TRUNK_MIN_KB=0 SNAP=qwen38_tiny ./qwen38 4 8 qwen38_tiny/ref.json) ;;
      "MTP mixed") args=(Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_tiny_mtp ./qwen38 2 8 qwen38_tiny_mtp/ref.json) ;;
      "device lost") args=(COLI_VK_CHAIN_FAULT=40 Q38_TRUNK_MIN_KB=0 SNAP=qwen38_tiny ./qwen38 4 8 qwen38_tiny/ref.json) ;;
    esac
    rm -f chain.usage
    env COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 "${args[@]}" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan dense-only qwen38 $tag: sanitizer diagnostic"; fi
    [ "$(dho_dropped san.log)" -gt 0 ] || { cat san.log; fail "asan dense-only qwen38 $tag: nothing on the device only"; }
    if [ "$tag" = "device lost" ]; then [ "$(dho_reloaded san.log)" -gt 0 ] || { cat san.log; fail "asan dense-only qwen38 $tag: nothing read back"; }; fi
    echo "OK asan dense-only qwen38 $tag: sanitizers clean, $(dho_dropped san.log) matrices on the device only, $(dho_reloaded san.log) read back"
  done
  CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 > san.log 2>&1 || { cat san.log; fail "asan dense-only qwen38 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan dense-only qwen38 serve: sanitizer diagnostic"; fi
  echo "OK asan dense-only qwen38 serve: $(tail -1 san.log)"
  rm -f chain.usage
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
