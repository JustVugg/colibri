#!/usr/bin/env bash
# Every engine's Vulkan path against its own CPU run, on Lavapipe (Mesa's software
# Vulkan), one family per call so CI can run them side by side:
#
#   bash tests/vulkan_engines.sh qwen | inkling-olmoe | mimo-qwenimage | deepseek
#   bash tests/vulkan_engines.sh shader    # the qmatmul formats alone, no engine
#
# Needs libvulkan-dev, glslc and mesa-vulkan-drivers, plus the Python packages of
# the family's tiny fixtures (see the vulkan-engines job in .github/workflows/ci.yml).
#
# Lavapipe is a CPU rasteriser: nothing here says anything about speed. What it does
# prove is that every resident format an engine uploads reaches the shader and comes
# back as the CPU computes it. Each configuration is gated on two things:
#   - the Vulkan run gives the tokens of the CPU run with the same snapshot and
#     settings (and, where the engine has one, passes its own oracle);
#   - its "[VK] <engine>: N matmuls on the GPU" line has N > 0, because a hook that
#     declines every matrix would otherwise pass the first gate trivially.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}
export COLI_NO_OMP_TUNE=1
PY=${PY:-python3}

fail() { echo "FAIL: $*"; exit 1; }

# vk_count <engine> <log>: N from the last "[VK] <engine>: N matmuls on the GPU" line
vk_count() {
  local n
  n=$(sed -n "s/^\[VK\] $1: \([0-9][0-9]*\) matmuls on the GPU.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
need_gpu() {  # <engine> <log> <tag>
  [ "$(vk_count "$1" "$2")" -gt 0 ] || { cat "$2"; fail "$3: no matmul ran on the device"; }
}
same_tokens() {  # <cpu log> <vk log> <tag>: the engines' "C engine" token lines
  grep -a '^C engine' "$1" > cpu.tok || true
  grep -a '^C engine' "$2" > vk.tok || true
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; fail "$3: Vulkan tokens differ from the CPU"; }
}

# The shader itself: every weight format against a CPU reference, before any engine.
shader_formats() {
  cc -O2 -DVK_TEST backend_vulkan.c -o vk_test -lvulkan -lm
  COLI_VK_TEST_MATMUL_ONLY=1 ./vk_test shaders/qmatmul.spv | tee vk_test.log
  tail -1 vk_test.log | grep -qx PASS || fail "qmatmul format cases"
}

# vk_gate <engine> <placed-regex> <tag> <env...> -- <argv...>
# CPU arm and Vulkan arm of one configuration; both must exit 0 (the engine's own
# oracle), give the same tokens, run matmuls on the device, and report the expected
# placement ("placed int8 a, int4 b, f32 c" / "int8 a, bf16 b, f32 c").
vk_gate() {
  local eng=$1 placed=$2 tag=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || { cat cpu.log; fail "$tag: CPU run"; }
  env "${envs[@]}" COLI_VULKAN=1 ./"$eng" "$@" > vk.log 2>&1 || { cat vk.log; fail "$tag: Vulkan run misses the oracle"; }
  same_tokens cpu.log vk.log "$tag"
  need_gpu "$eng" vk.log "$tag"
  grep -qE "\[VK\] $eng: .*placed .*$placed" vk.log || { grep '\[VK\]' vk.log; fail "$tag: expected placement '$placed'"; }
  echo "OK $tag: $(grep -a -o 'Matching tokens: [0-9/]*' vk.log), tokens = CPU, $(grep -o '[0-9]* matmuls on the GPU.*' vk.log | tail -1)"
}

family_qwen() {
  make qwen36 qwen38 VK=1
  # qwen36: the hybrid, Qwen3-Coder (qwen3_moe), the 27B dense and the 2.4T geometry
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen3-coder-30b --out qwen3_coder_tiny --ref-mode full --emit-ref qwen3_coder_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-2p4t --seed 3 --out qwen38_2p4t_tiny --ref-mode full --emit-ref qwen38_2p4t_tiny/ref_full.json
  local fx cap caps
  for fx in qwen36_tiny qwen3_coder_tiny qwen38_27b_tiny qwen38_2p4t_tiny; do
    $PY tools/convert_qwen36.py --model $fx --out ${fx}_c --ebits 8
    # cap=1 evicts on every routed expert; on the 92-layer 2.4T geometry that costs
    # ~90 s on Lavapipe for nothing the device path adds, so that one runs at cap=8.
    caps="1 8"; [ $fx = qwen38_2p4t_tiny ] && caps=8
    for cap in $caps; do
      # the CPU job's own configuration: f32 dense weights (fmt 10), token-exact
      # against transformers and equal to the CPU run
      vk_gate qwen36 'f32 [1-9]' "qwen36 $fx f32 cap=$cap" COLI_DENSE_I8=0 SNAP=${fx}_c -- $cap 8 $fx/ref_full.json
    done
    # int8 dense rows (fmt 1). int8 weights alone miss the torch oracle on some
    # fixtures, CPU or GPU alike, so this arm gates on the CPU's tokens only, with
    # COLI_DENSE_IDOT=0 giving the CPU the shader's f32 activations.
    COLI_DENSE_IDOT=0 SNAP=${fx}_c ./qwen36 8 8 $fx/ref_full.json > cpu.log 2>&1 || true
    COLI_DENSE_IDOT=0 COLI_VULKAN=1 SNAP=${fx}_c ./qwen36 8 8 $fx/ref_full.json > vk.log 2>&1 || true
    same_tokens cpu.log vk.log "qwen36 $fx int8"
    grep -qE '\[VK\] qwen36: [1-9][0-9]* matmuls on the GPU.*placed int8 [1-9]' vk.log || { cat vk.log; fail "qwen36 $fx int8: nothing placed"; }
    echo "OK qwen36 $fx int8: tokens = CPU, $(grep -o '[0-9]* matmuls on the GPU.*' vk.log | tail -1)"
  done

  # qwen38: one fixture, every resident format, with and without prefill batching
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  local batch O
  for batch in 0 1; do
    O="OMP_NUM_THREADS=2 SNAP=qwen38_tiny Q38_PREFILL_BATCH=$batch"
    # the default: every fixture matrix is under 1 MiB, so the trunk stays BF16 (fmt 11)
    vk_gate qwen38 'bf16 [1-9]' "qwen38 bf16 batch=$batch" $O -- 1 8 qwen38_tiny/ref.json
    # Q38_TRUNK_MIN_KB=0: the int8 trunk (fmt 1); what stays outside it is BF16
    vk_gate qwen38 'int8 [1-9].*bf16 [1-9]' "qwen38 int8 batch=$batch" $O Q38_TRUNK_MIN_KB=0 -- 1 8 qwen38_tiny/ref.json
    # Q38_NATIVE_BF16=0 expands the rows to f32 at load (fmt 10)
    vk_gate qwen38 'f32 [1-9]' "qwen38 f32 batch=$batch" $O Q38_TRUNK_CPU_INT8=0 Q38_NATIVE_BF16=0 -- 1 8 qwen38_tiny/ref.json
  done
}

family_inkling_olmoe() {
  make inkling olmoe VK=1
  local cap
  # inkling, f32 fixture (fmt 10): the oracle and the CPU's ids, at three caps
  $PY tools/make_tiny_inkling.py tiny_inkling
  for cap in 1 2 8; do
    SNAP=tiny_inkling ./inkling $cap 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1
    COLI_VULKAN=1 SNAP=tiny_inkling ./inkling $cap 0 tiny_inkling/ref_inkling.json > vk.log 2>&1
    grep -qE 'Matching tokens: ([0-9]+)/\1$' vk.log || { cat vk.log; fail "inkling f32 cap=$cap: oracle"; }
    same_tokens cpu.log vk.log "inkling f32 cap=$cap"
    need_gpu inkling vk.log "inkling f32 cap=$cap"
    echo "OK inkling f32 cap=$cap: $(grep -o 'Matching tokens: [0-9/]*' vk.log), $(vk_count inkling vk.log) matmuls on the GPU"
  done

  # the same weights stored as bf16 (fmt 11). A build with the AVX512-BF16 dot rounds
  # activations to bf16 on the CPU and keeps those matrices there ("0 bf16" in the
  # placement line), so the count is only required when the line places any.
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
  SNAP=tiny_inkling_bf16 ./inkling 8 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1 || true
  COLI_VULKAN=1 SNAP=tiny_inkling_bf16 ./inkling 8 0 tiny_inkling/ref_inkling.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "inkling bf16"
  grep -q ', 0 bf16)' vk.log || need_gpu inkling vk.log "inkling bf16"
  echo "OK inkling bf16: tokens = CPU, $(vk_count inkling vk.log) matmuls on the GPU"

  # the dense-int4g64 container (int8 fmt 1, int4-g64 fmt 4, f32 fmt 10 side by side)
  rm -rf tiny_inkling_q && cp -r tiny_inkling tiny_inkling_q
  $PY - tools/convert_inkling_dense_int4.py tiny_inkling_q <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("conv", sys.argv[1])
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
conv.MIN_ELEMS = 0
conv.ATTN_BITS = 4
base = conv.classify
conv.classify = lambda n, s, d: "int8" if n.endswith(".mlp.down_proj.weight") else base(n, s, d)
sys.argv = ["convert_inkling_dense_int4.py", "--dir", sys.argv[2]]
conv.main()
EOF
  mkdir -p tiny_inkling_q/dense-int4g64
  mv tiny_inkling_q/dense-int4g64.safetensors tiny_inkling_q/dense-int4g64/dense.safetensors
  SNAP=tiny_inkling_q ./inkling 8 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1 || true
  COLI_VULKAN=1 SNAP=tiny_inkling_q ./inkling 8 0 tiny_inkling/ref_inkling.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "inkling int4-g64 container"
  need_gpu inkling vk.log "inkling int4-g64 container"
  echo "OK inkling int4-g64 container: tokens = CPU, $(vk_count inkling vk.log) matmuls on the GPU"

  # olmoe: f32 residents (fmt 10), the oracle and the CPU's ids
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  SNAP=olmoe_tiny_c ./olmoe 8 8 olmoe_tiny/ref_olmoe.json > cpu.log 2>&1
  COLI_VULKAN=1 SNAP=olmoe_tiny_c ./olmoe 8 8 olmoe_tiny/ref_olmoe.json > vk.log 2>&1
  grep -qE 'Matching tokens: ([0-9]+)/\1$' vk.log || { cat vk.log; fail "olmoe: oracle"; }
  same_tokens cpu.log vk.log "olmoe"
  need_gpu olmoe vk.log "olmoe"
  echo "OK olmoe: $(grep -o 'Matching tokens: [0-9/]*' vk.log), $(vk_count olmoe vk.log) matmuls on the GPU"
}

family_mimo_qwenimage() {
  make mimo qwenimage VK=1
  # mimo: Xiaomi's vendor oracle on the GPU (its engine_env is the f32 dense
  # configuration; its variants include the native FP8/BF16 one and the BF16 vision
  # tower), once with the experts on the CPU, once all on the GPU (MXFP4, fmt 7).
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  COLI_VULKAN=1 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  COLI_VULKAN=1 MIMO_VK_EXPERTS=4096 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  # MIMO_DENSE_BITS 0 (native fp8/bf16: fmt 12, 11), 8 (fmt 1) and 32 (fmt 10):
  # the CPU's tokens for every case of ref.json and for the picture.
  ids() { $PY -c "import json,sys;r=json.load(open('mimo_tiny/ref.json'));c=r['image'] if sys.argv[1]=='image' else r['cases'][sys.argv[1]];print(' '.join(map(str,c['prompt_ids'])))" "$1"; }
  local grid bits c x extra
  grid=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  for bits in 0 8 32; do
    for c in short window long image; do
      extra=(); [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $grid)
      MIMO_DENSE_BITS=$bits COLI_TEMP=0 ./mimo mimo_tiny --ids "$(ids $c)" --ngen 6 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
      for x in 0 4096; do
        COLI_VULKAN=1 MIMO_VK_EXPERTS=$x MIMO_DENSE_BITS=$bits COLI_TEMP=0 \
          ./mimo mimo_tiny --ids "$(ids $c)" --ngen 6 "${extra[@]}" > mimo-vk.txt 2> mimo-vk.err
        cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-vk.err; fail "mimo bits=$bits $c MIMO_VK_EXPERTS=$x differs from the CPU"; }
        need_gpu mimo mimo-vk.err "mimo bits=$bits $c MIMO_VK_EXPERTS=$x"
      done
    done
    echo "OK mimo MIMO_DENSE_BITS=$bits: the CPU's tokens, text and image, experts on the CPU and on the GPU"
  done

  # qwenimage: at 8, 16 and 32 bits (fmt 1, 11, 10) every oracle stage of the
  # Lavapipe run against the CPU run with the same bits and f32 activations. int8 is
  # outside the oracle's own tolerance on both sides, so the reports are compared
  # with each other, not with the reference; at 16 bits the Vulkan run must also pass.
  $PY tools/make_qwenimage_tiny.py qwenimage_tiny
  for bits in 8 16 32; do
    COLI_IMG_BITS=$bits COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --ref qwenimage_tiny/ref > qi-cpu.log 2>&1 || true
    local rc=0
    COLI_VULKAN=1 COLI_IMG_BITS=$bits COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --ref qwenimage_tiny/ref > qi-vk.log 2>&1 || rc=$?
    [ $bits != 16 ] || [ $rc = 0 ] || { cat qi-vk.log; fail "qwenimage bf16 oracle on Vulkan"; }
    need_gpu qwenimage qi-vk.log "qwenimage oracle bits=$bits"
    BITS=$bits $PY - <<'PY'
import os, re
def stages(p):
    return {k.strip(): float(r) for k, r in
            re.findall(r"\[oracle\] (.+?)\s+n=\d+\s+max\|err\| \S+\s+rel (\S+)", open(p).read())}
c, v = stages("qi-cpu.log"), stages("qi-vk.log")
assert len(c) >= 10 and c.keys() == v.keys(), (c, v)
for k in c:   # relative errors against the reference, CPU vs GPU: equal up to summation order
    assert abs(c[k] - v[k]) <= 0.25 * c[k] + 2e-7, (k, c[k], v[k])
print(f"OK qwenimage oracle: {len(c)} stages of the Lavapipe run match the CPU run at {os.environ['BITS']} bits")
PY
  done
  # a generated picture at the default int8 weights, Lavapipe against the CPU
  COLI_IMG_BITS=8 COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --prompt "a red fox in the snow" \
    --width 256 --height 256 --steps 2 --seed 1 --out qi-cpu.png
  COLI_VULKAN=1 COLI_IMG_BITS=8 COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --prompt "a red fox in the snow" \
    --width 256 --height 256 --steps 2 --seed 1 --out qi-vk.png 2> qi-vk-gen.err
  need_gpu qwenimage qi-vk-gen.err "qwenimage picture"
  $PY - <<'PY'
import sys; sys.path.insert(0, ".")
import image_engine as e
_, _, _, a = e.decode_png(open("qi-cpu.png", "rb").read())
_, _, _, b = e.decode_png(open("qi-vk.png", "rb").read())
d = [abs(x - y) for x, y in zip(a, b)]
assert len(a) == len(b) and max(d) <= 2 and sum(t > 0 for t in d) <= len(d) // 1000, (max(d), sum(t > 0 for t in d))
print(f"OK qwenimage picture: {sum(t > 0 for t in d)} of {len(d)} bytes differ from the CPU's, max {max(d)}")
PY
  # the serve protocol with the device on
  COLI_VULKAN=1 QWENIMAGE_TINY=qwenimage_tiny $PY -m unittest tests.test_qwenimage_engine_serve
}

family_deepseek() {
  make deepseek_v41 VK=1
  # deepseek_v41: fp8 dense in 32x32 ue8m0 tiles (fmt 12, gs 32, the tile scale
  # repeated over its rows) and bf16 (fmt 11). The engine exits non-zero on any
  # token mismatch with the reference; the CPU run must print the same stream.
  $PY tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json
  $PY tools/make_dsv41_tiny.py --out dsv41_long --emit-ref dsv41_long/ref.json --prompt-len 40 --max-new 6
  v41() {  # <tag> <env and argv...>
    local tag=$1; shift
    env "$@" > v41-cpu.txt 2> v41-cpu.err || { cat v41-cpu.err; fail "deepseek_v41 $tag: CPU run"; }
    env COLI_VULKAN=1 "$@" > v41-vk.txt 2> v41-vk.err || { cat v41-vk.err; fail "deepseek_v41 $tag: Vulkan run misses the oracle"; }
    cmp -s v41-cpu.txt v41-vk.txt || { diff v41-cpu.txt v41-vk.txt | head; fail "deepseek_v41 $tag: Vulkan output differs from the CPU"; }
    need_gpu deepseek_v41 v41-vk.err "deepseek_v41 $tag"
    echo "OK deepseek_v41 $tag: output = CPU, $(vk_count deepseek_v41 v41-vk.err) matmuls on the GPU"
  }
  local cap force
  for cap in 1 2 8; do v41 "cap=$cap" SNAP=dsv41_tiny ./deepseek_v41 $cap dsv41_tiny/ref.json; done
  for cap in 2 8; do v41 "40-token prompt cap=$cap" SNAP=dsv41_long ./deepseek_v41 $cap dsv41_long/ref.json; done
  for force in 1 2 3 4 5; do
    v41 "DSpark spec=$force" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$force ./deepseek_v41 8 dsv41_tiny/ref.json
  done

  # deepseek_v4: fp8 128x128 blocks (fmt 12, gs 128) and the bf16 router, compressors
  # and head (fmt 11). The GPU gets the activations after the CPU's own E4M3 rounding,
  # so the two runs do the same arithmetic. The tiny check builds the VK=1 binary and
  # keeps passing with the device open; its --oracle path reloads the dense weights
  # every forward and so stays on the CPU, which is why the device is checked on the
  # session path below: ids and teacher-forced predictions equal to the CPU's and to
  # the reference's greedy stream.
  COLI_VULKAN=1 make deepseek-v4-tiny-check VK=1
  local prompt
  prompt=$($PY -c 'import json; c=json.load(open("deepseek_v4_tiny/ref.json"))["cases"]["long"]; print("".join("<t%03d>" % t for t in c["prompt_ids"]))')
  ./deepseek_v4 ./deepseek_v4_tiny "$prompt" --raw-prompt --max-tokens 4 --record-oracle v4-cpu.json > /dev/null
  COLI_VULKAN=1 ./deepseek_v4 ./deepseek_v4_tiny "$prompt" --raw-prompt --max-tokens 4 --record-oracle v4-vk.json > /dev/null 2> v4-vk.err
  $PY - <<'PY' || fail "deepseek_v4: the Vulkan session differs from the CPU's"
import json, sys
a, b = json.load(open("v4-cpu.json")), json.load(open("v4-vk.json"))
ref = json.load(open("deepseek_v4_tiny/ref.json"))["cases"]["long"]["greedy_full_ids"]
ok = a["full_ids"] == b["full_ids"] == ref and a["tf_pred"] == b["tf_pred"]
print("OK deepseek_v4 session: ids = CPU = reference" if ok else ("CPU", a, "VK", b, "ref", ref))
sys.exit(0 if ok else 1)
PY
  need_gpu deepseek_v4 v4-vk.err "deepseek_v4 session"
  echo "OK deepseek_v4: $(vk_count deepseek_v4 v4-vk.err) matmuls on the GPU"
}

case "${1:-}" in
  shader)         shader_formats ;;
  qwen)           family_qwen ;;
  inkling-olmoe)  family_inkling_olmoe ;;
  mimo-qwenimage) family_mimo_qwenimage ;;
  deepseek)       family_deepseek ;;
  *) echo "usage: $0 shader|qwen|inkling-olmoe|mimo-qwenimage|deepseek" >&2; exit 2 ;;
esac
