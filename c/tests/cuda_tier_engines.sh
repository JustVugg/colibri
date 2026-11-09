#!/bin/bash
# The routed-expert tier on a CUDA device (cu_tier.h) against each engine's CPU run, on the
# engines' tiny fixtures, with the criteria of tests/vulkan_engines.sh's tier gates. It needs a
# CUDA card, so CI does not run it:
#
#   cd c && PY=python3 bash tests/cuda_tier_engines.sh
#
# PY has numpy, torch and gguf-py (the fixtures' generators). The engines are built with CUDA=1.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
make -j"$(nproc)" glm53 deepseek_v41 mimo kimi_k3 colibri qwen36 CUDA=1 > cuda_tier_build.log 2>&1 || { tail cuda_tier_build.log; exit 1; }
[ -d glm53_stream-i4 ] || { $PY tools/make_glm53_tiny.py --output glm53_tiny --force > /dev/null &&
                             $PY tools/make_glm53_streaming_pair.py --fixture glm53_tiny --output glm53_stream > /dev/null; } || exit 1
[ -d dsv41_tiny ] || $PY tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json > /dev/null || exit 1
[ -f mimo_tiny/model-00001-of-00002.safetensors ] || $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision > /dev/null || exit 1
if [ ! -d k3_gguf_tiny ]; then
  $PY - <<'PY' || exit 1
import sys
sys.path.insert(0, "tests")
import numpy as np
from test_gguf_import import write_kimik3
write_kimik3("k3_gguf_tiny.gguf", np.random.default_rng(6))
PY
  $PY tools/gguf_import.py k3_gguf_tiny.gguf k3_gguf_tiny > /dev/null || exit 1
fi
if [ ! -d ds41_gguf_tiny ]; then   # a published deepseek41 GGUF, small (tests/test_gguf_import.py)
  $PY - <<'PY' || exit 1
import sys
sys.path.insert(0, "tests")
import numpy as np
from test_gguf_import import write_deepseek41
write_deepseek41("ds41_gguf_tiny.gguf", np.random.default_rng(3))
PY
  $PY tools/gguf_import.py ds41_gguf_tiny.gguf ds41_gguf_tiny > /dev/null || exit 1
fi
if [ ! -d glm_gguf_tiny ]; then   # llama.cpp's glm-dsa (GLM-5.2, GLM-5.3), small (tests/test_gguf_import.py)
  $PY - <<'PY' || exit 1
import sys
sys.path.insert(0, "tests")
import numpy as np
from test_gguf_import import write_glmdsa
write_glmdsa("glm_gguf_tiny.gguf", np.random.default_rng(9))
PY
  $PY tools/gguf_import.py glm_gguf_tiny.gguf glm_gguf_tiny > /dev/null || exit 1
fi
if [ ! -d glm_gguf_sh ]; then   # the same with layers 2 and 3 holding layer 1's indexer: shared layers, as unsloth's GLM-5.3
  $PY - <<'PY' || exit 1
import sys
sys.path.insert(0, "tests")
import numpy as np
from test_gguf_import import write_glmdsa
write_glmdsa("glm_gguf_sh.gguf", np.random.default_rng(9), copies=(2, 3))
PY
  $PY tools/gguf_import.py glm_gguf_sh.gguf glm_gguf_sh > /dev/null || exit 1
fi
fails=0
ok() { echo "OK $*"; }
bad() { echo "FAIL $*"; fails=$((fails + 1)); }
dev_count() { grep -aoE "\[CUDA\] tier $1 (run|turn): device [0-9]+ of [0-9]+" "$2" | tail -1 | grep -oE "device [0-9]+" | grep -oE "[0-9]+" || echo 0; }
X="COLI_CUDA=1 COLI_VULKAN=0 COLI_VK_TIER_SYNC=1"   # COLI_VULKAN=0: a build with Vulkan too stays on CUDA

# glm53: the greedy and teacher-forced lines, the int4-g64 streaming container
for v in "decode:GLM53_BITS=32:5,7,9,11,13,17,19,23:20" "evict:GLM53_BITS=32 COLI_VK_TIER_GB=0.00006:5,7,9,11,13,17,19,23:6"; do
  IFS=: read -r tag envs ids n <<< "$v"
  env $envs USAGE_SAVE=0 COLI_USAGE=g53.usage ./glm53 --model glm53_stream-i4 --ids $ids --greedy $n > g53-cpu.log 2>&1
  env $envs $X USAGE_SAVE=0 COLI_USAGE=g53.usage ./glm53 --model glm53_stream-i4 --ids $ids --greedy $n > g53-cu.log 2>&1
  grep -aE '^teacher_forcing|^greedy' g53-cpu.log > a.tok; grep -aE '^teacher_forcing|^greedy' g53-cu.log > b.tok
  d=$(dev_count glm53 g53-cu.log)
  if [ -s a.tok ] && cmp -s a.tok b.tok && [ "$d" -gt 0 ]; then ok "glm53 $tag: tokens = CPU, device $d"; else bad "glm53 $tag (device $d)"; diff a.tok b.tok | head -4; tail -3 g53-cu.log; fi
done

# glm53 on the CUDA chain (glm53_cuchain.h): every layer on the card, the greedy and teacher-forced lines the
# CPU's and the logits within 1e-4 of the largest with the exact GEMMs (COLI_CUDA_TC_GEMM=0), 1e-3 on the tensor cores
lclose() {  # <cpu.f32> <cu.f32> <tolerance>: max |diff| within tolerance of the largest |logit|
  $PY - "$1" "$2" "$3" <<'PY'
import array, sys
try:
    a = array.array("f", open(sys.argv[1], "rb").read()); b = array.array("f", open(sys.argv[2], "rb").read())
except OSError:
    print("no logits"); sys.exit(1)
d = max(abs(x - y) for x, y in zip(a, b)) if a and len(a) == len(b) else float("inf")
m = max(abs(x) for x in a) if a else 0.0
print(f"max |logit diff| {d:.2e} of {m:.2e}")
sys.exit(0 if d <= float(sys.argv[3]) * m else 1)
PY
}
g53ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
for v in "chain decode|1e-4|GLM53_BITS=32|5,7,9,11,13,17,19,23|8" \
         "chain 100-token prompt, 32-bit trunk|1e-4|GLM53_BITS=32|$g53ids|4" \
         "chain 100-token prompt, 4-bit trunk, exact GEMMs|1e-4|GLM53_BITS=4 COLI_CUDA_TC_GEMM=0|$g53ids|4" \
         "chain 100-token prompt, 4-bit trunk, tensor cores|1e-3|GLM53_BITS=4|$g53ids|4" \
         "chain prefill chunks of 7, one cache slot|1e-4|GLM53_BITS=4 COLI_CUDA_TC_GEMM=0 GLM53_PREFILL_CHUNK=7 GLM53_EXPERT_GB=0.000001|$g53ids|4" \
         "chain chunks of 3|1e-4|GLM53_BITS=32 COLI_CUDA_G53_ROWS=3|$g53ids|4"; do
  IFS='|' read -r tag tol envs ids n <<< "$v"
  rm -f g53.usage g53-cpu.f32 g53-cu.f32
  env $envs COLI_SPEC_GATE=0 USAGE_SAVE=0 COLI_USAGE=g53.usage DUMP=g53-cpu.f32 ./glm53 --model glm53_stream-i4 --ids $ids --greedy $n > g53-cpu.log 2>&1
  rm -f g53.usage
  env $envs $X COLI_SPEC_GATE=0 USAGE_SAVE=0 COLI_USAGE=g53.usage DUMP=g53-cu.f32 ./glm53 --model glm53_stream-i4 --ids $ids --greedy $n > g53-cu.log 2>&1
  grep -aE '^teacher_forcing|^greedy' g53-cpu.log > a.tok; grep -aE '^teacher_forcing|^greedy' g53-cu.log > b.tok
  fw=$(sed -n 's/^\[CUDA\] glm53 chain: \([0-9][0-9]*\) forwards.*/\1/p' g53-cu.log | tail -1)
  lg=$(lclose g53-cpu.f32 g53-cu.f32 "$tol"); lok=$?
  if [ -s a.tok ] && cmp -s a.tok b.tok && [ "${fw:-0}" -gt 0 ] && [ $lok = 0 ]; then ok "glm53 $tag: tokens = CPU, $lg, $fw chain forwards"
  else bad "glm53 $tag (${fw:-0} chain forwards, $lg)"; diff a.tok b.tok | head -4; tail -3 g53-cu.log; fi
done

# qwen36 on the CUDA chain from a dense GGUF (architecture qwen35) whose DeltaNet layers give qkv and the gate
# two types, as unsloth's dynamic mixes of Qwen3.8-27B do in about half of them: layer 0 Q8_0 and Q4_K, layer 1
# the other way, layer 2 Q8_0 twice. The two-type layers' in_proj is two tensors (qt_dnproj_z_init_fmt); before,
# they stayed on the CPU and the chain with them. A 12-token prompt (one step of 12 rows, or chunks of 5) and
# four decode steps: the tokens are the CPU's, the last logits within 1e-2 of the largest, the chain ran.
if [ ! -d q35d_gguf_tiny ]; then
  $PY - <<'PY' || exit 1
import sys, numpy as np
sys.path.insert(0, "tests")
from test_gguf_import import write_model
write_model("q35d_gguf_tiny.gguf", np.random.default_rng(12), dense=True,
            dn_types={0: ("Q8_0", "Q4_K"), 1: ("Q4_K", "Q8_0"), 2: ("Q8_0", "Q8_0")})
PY
  $PY tools/gguf_import.py q35d_gguf_tiny.gguf q35d_gguf_tiny > /dev/null || exit 1
  printf '{"prompt_ids": [5, 6, 7, 8, 9, 31, 17, 2, 44, 12, 3, 29], "full_ids": [5, 6, 7, 8, 9, 31, 17, 2, 44, 12, 3, 29, 0, 0, 0, 0]}' > q35d_gguf_tiny/ref.json
fi
q36tok() { grep -a '^C engine' "$1" | tail -1 | cut -d: -f2 | sed 's/Text.*//'; }
for v in "a 12-row step|" "chunks of 5|COLI_CUDA_CHAIN_ROWS=5"; do
  IFS='|' read -r tag envs <<< "$v"
  rm -f q35-cpu.f32 q35-cu.f32
  env $envs DUMP=q35-cpu.f32 SNAP=q35d_gguf_tiny OMP_NUM_THREADS=2 COLI_NO_OMP_TUNE=1 USAGE_SAVE=0 ./qwen36 2 8 q35d_gguf_tiny/ref.json > q35-cpu.log 2>&1
  env $envs $X DUMP=q35-cu.f32 SNAP=q35d_gguf_tiny OMP_NUM_THREADS=2 COLI_NO_OMP_TUNE=1 USAGE_SAVE=0 ./qwen36 2 8 q35d_gguf_tiny/ref.json > q35-cu.log 2>&1
  a=$(q36tok q35-cpu.log); b=$(q36tok q35-cu.log); lg=$(lclose q35-cpu.f32 q35-cu.f32 1e-2); lok=$?
  two=$(sed -n 's/^\[dnp\] \([0-9]*\) of them as two tensors.*/\1/p' q35-cu.log | tail -1)
  if [ -n "$a" ] && [ "$a" = "$b" ] && [ $lok = 0 ] && [ "${two:-0}" = 2 ] && ! grep -qa "the chain stays off" q35-cu.log; then
    ok "qwen36 GGUF dense, DeltaNet qkv and gate in two types, $tag: tokens = CPU, $lg, $two in_proj as two tensors, the chain on"
  else bad "qwen36 GGUF dense two types, $tag (two tensors: ${two:-none}, $lg)"; echo "cpu:$a"; echo "cu: $b"; grep -aE '^\[(chain|dnp|dense)\]' q35-cu.log | head -6; fi
done

# qwen36's hybrid steps on the CUDA chain (a tier too small for the experts: the host computes the slots the device
# lacks while the device waits for its word), from a small qwen35moe GGUF; then the host's word of the second hybrid
# step never written (COLI_CUDA_HOST_WAIT_FAULT=2): the device stops waiting after COLI_CUDA_HOST_WAIT_S=1, the frame is
# lost and the CPU goes on (#2696: the card spun on that word with the answer stopped). Between them the same steps
# with every signal sent to the card at once (COLI_CUDA_FLUSH=1, Windows' default without TCC). The tokens are the CPU's.
if [ ! -d q35m_gguf_tiny ]; then
  $PY - <<'PY' || exit 1
import sys, numpy as np
sys.path.insert(0, "tests")
from test_gguf_import import write_model
write_model("q35m_gguf_tiny.gguf", np.random.default_rng(11), same_gate_up=True)
PY
  $PY tools/gguf_import.py q35m_gguf_tiny.gguf q35m_gguf_tiny > /dev/null || exit 1
fi
# the reference is the CPU's own twelve tokens (gate and up in one type: the chain's experts take the layer)
printf '{"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}' > q35m_gguf_tiny/ref.json
env SNAP=q35m_gguf_tiny OMP_NUM_THREADS=2 COLI_NO_OMP_TUNE=1 USAGE_SAVE=0 ./qwen36 2 8 q35m_gguf_tiny/ref.json > q35m-cpu.log 2>&1
cputok=$(grep -a '^C engine' q35m-cpu.log | tail -1 | sed 's/^C engine : //; s/ *Text .*//' | tr -s ' ' | sed 's/ *$//; s/ /, /g')
printf '{"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8, %s]}' "$cputok" > q35m_gguf_tiny/ref.json
q36hyb() { sed -n 's/^\[chain\] qwen36: [0-9]* layer steps with the routed experts in the frame, \([0-9]*\) of them hybrid.*/\1/p' "$1" | tail -1; }
for v in "hybrid steps|" "hybrid steps, each signal sent at once|COLI_CUDA_FLUSH=1" \
         "the host's word lost|COLI_CUDA_HOST_WAIT_FAULT=2 COLI_CUDA_HOST_WAIT_S=1"; do
  IFS='|' read -r tag envs <<< "$v"
  env SNAP=q35m_gguf_tiny OMP_NUM_THREADS=2 COLI_NO_OMP_TUNE=1 USAGE_SAVE=0 ./qwen36 2 8 q35m_gguf_tiny/ref.json > q35m-cpu.log 2>&1
  env $envs $X CUDA_EXPERT_GB=0.005 SNAP=q35m_gguf_tiny OMP_NUM_THREADS=2 COLI_NO_OMP_TUNE=1 USAGE_SAVE=0 timeout 120 ./qwen36 2 8 q35m_gguf_tiny/ref.json > q35m-cu.log 2>&1; rc=$?
  a=$(grep -a '^C engine' q35m-cpu.log | tail -1); b=$(grep -a '^C engine' q35m-cu.log | tail -1); h=$(q36hyb q35m-cu.log)
  if [ -z "$envs" ]; then good=$([ "${h:-0}" -gt 0 ] && ! grep -qa "a signal for the host goes to the device at once" q35m-cu.log && echo 1)
  elif [ "$envs" = COLI_CUDA_FLUSH=1 ]; then
    good=$([ "${h:-0}" -gt 0 ] && grep -qa "a signal for the host goes to the device at once (COLI_CUDA_FLUSH" q35m-cu.log && echo 1)
  else good=$(grep -qa "the device waited 1 s for the host's share of hybrid step" q35m-cu.log && echo 1); fi
  if [ $rc = 0 ] && [ -n "$a" ] && [ "$a" = "$b" ] && [ "$good" = 1 ]; then ok "qwen36 GGUF $tag: tokens = CPU (${h:-0} hybrid layer steps)"
  else bad "qwen36 GGUF $tag (rc $rc, ${h:-0} hybrid layer steps)"; echo "cpu: $a"; echo "cu:  $b"; grep -aE '^\[(chain|qtier|CUDA)\]' q35m-cu.log | tail -5; fi
done

# deepseek_v41: the oracle passes and the output is the CPU's
for cap in 1 2 8; do
  env SNAP=dsv41_tiny ./deepseek_v41 $cap dsv41_tiny/ref.json > v41-cpu.txt 2> v41-cpu.err; rc0=$?
  env $X COLI_USAGE=tier.usage SNAP=dsv41_tiny ./deepseek_v41 $cap dsv41_tiny/ref.json > v41-cu.txt 2> v41-cu.err; rc=$?
  d=$(dev_count deepseek_v41 v41-cu.err)
  if [ $rc0 = 0 ] && [ $rc = 0 ] && cmp -s v41-cpu.txt v41-cu.txt && [ "$d" -gt 0 ]; then ok "deepseek_v41 cap=$cap: output = CPU, oracle passed, device $d"
  else bad "deepseek_v41 cap=$cap (rc $rc0/$rc, device $d)"; diff v41-cpu.txt v41-cu.txt | head -4; tail -3 v41-cu.err; fi
done
rm -f tier.usage

# deepseek_v41 from a GGUF: the routed experts on the tier and the dense matrices on the card
# (cu_dense.h), in the float forms; the tokens the CPU's
ids="5, 6, 7, 8, 20, 21, 22, 23, 30, 31, 32, 33"
echo "{\"prompt_ids\": [$ids], \"output_ids\": [0, 0, 0, 0, 0, 0, 0, 0]}" > ds41_gguf_tiny/ref.json
env GQ_EXACT=1 V41_DSPARK=0 SNAP=ds41_gguf_tiny ./deepseek_v41 8 ds41_gguf_tiny/ref.json > v41g-cpu.txt 2> v41g-cpu.err
env GQ_EXACT=1 V41_DSPARK=0 SNAP=ds41_gguf_tiny $X COLI_USAGE=tier.usage ./deepseek_v41 8 ds41_gguf_tiny/ref.json > v41g-cu.txt 2> v41g-cu.err
d=$(dev_count deepseek_v41 v41g-cu.err)
mats=$(grep -aoE "deepseek_v41: [0-9]+ dense matrices on the device" v41g-cu.err | sed -E 's/.*: ([0-9]+) dense.*/\1/' | head -1)
if [ -s v41g-cpu.txt ] && cmp -s v41g-cpu.txt v41g-cu.txt && [ "$d" -gt 0 ] && [ "${mats:-0}" -gt 0 ]; then
  ok "deepseek_v41 GGUF: tokens = CPU, device $d, $mats dense matrices on the card"
else bad "deepseek_v41 GGUF (device $d, dense ${mats:-0})"; diff v41g-cpu.txt v41g-cu.txt | head -4; tail -3 v41g-cu.err; fi
rm -f tier.usage

# colibri (GLM-5.2, GLM-5.3) from a GGUF on the CUDA chain (colibri_cuchain.h): every layer on the card, the
# experts on the tier as the file has them; the oracle's lines and the prompt's the CPU's, the logits within 1e-4
# of the largest with the exact GEMMs (COLI_CUDA_TC_GEMM=0), 1e-3 on the tensor cores. The CPU's float form of the
# file's blocks (GQ_EXACT=1) and a float32 KV cache on both sides (KV16=0: CUDA's own). The refs' tokens are none
# the model makes: the gate compares the chain with the CPU. An 8-token prompt and a 40-token one, in one chunk or in
# chunks of 3 and 5 (COLI_CUDA_GLM_ROWS); DSA_TOPK=4 turns the indexer's selection on from the fifth position,
# DSA_FORCE=1 at every position. The dense part at 8 bits: the CPU's exact int8 kernel (IDOT=0: the integer one moves
# the index scores, and four picks out of forty follow), no drafts (DRAFT=0: the two sides verify the same steps).
echo '{"prompt_ids": [5, 6, 7, 8, 20, 21, 22, 23], "full_ids": [5, 6, 7, 8, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8]}' > glm_gguf_tiny/ref.json
$PY -c "import json; p=[(i*37+5)%290+2 for i in range(40)]; print(json.dumps({'prompt_ids': p, 'full_ids': p + [1, 2, 3, 4, 5, 6, 7, 8]}))" > glm_gguf_tiny/ref40.json
cp glm_gguf_tiny/ref.json glm_gguf_tiny/ref40.json glm_gguf_sh/
gltok() { grep -aE '^GLM C engine|^PREFILL|^\[ORACLE\] mismatch' "$1" | sed 's/ | [0-9.]* pos\/s//'; }
for v in "chain decode|1e-4|COLI_CUDA_TC_GEMM=0|64 16 16|ref" \
         "chain decode, cap=1|1e-4|COLI_CUDA_TC_GEMM=0|1 16 16|ref" \
         "chain DSA top-4, the dense part at 8 bits (the serve's: the indexer's F32 head weights in int8)|1e-4|COLI_CUDA_TC_GEMM=0 DSA_TOPK=4 DRAFT=0 IDOT=0|64 16 8|ref40" \
         "chain 40-token prompt|1e-4|COLI_CUDA_TC_GEMM=0|64 16 16|ref40" \
         "chain 40-token prompt in chunks of 3|1e-4|COLI_CUDA_TC_GEMM=0 COLI_CUDA_GLM_ROWS=3|64 16 16|ref40" \
         "chain DSA top-4|1e-4|COLI_CUDA_TC_GEMM=0 DSA_TOPK=4|64 16 16|ref" \
         "chain DSA top-4, 40-token prompt in chunks of 5|1e-4|COLI_CUDA_TC_GEMM=0 DSA_TOPK=4 COLI_CUDA_GLM_ROWS=5|64 16 16|ref40" \
         "chain DSA_FORCE|1e-4|COLI_CUDA_TC_GEMM=0 DSA_FORCE=1|64 16 16|ref40" \
         "chain 40-token prompt, tensor cores|1e-3|DSA_TOPK=4|64 16 16|ref40" \
         "chain DSA top-4 over shared indexer layers (layers 2 and 3 reuse layer 1's selection)|1e-4|COLI_CUDA_TC_GEMM=0 DSA_TOPK=4|64 16 16|ref|glm_gguf_sh" \
         "chain DSA top-4 over shared indexer layers, 40-token prompt in chunks of 5|1e-4|COLI_CUDA_TC_GEMM=0 DSA_TOPK=4 COLI_CUDA_GLM_ROWS=5|64 16 16|ref40|glm_gguf_sh"; do
  IFS='|' read -r tag tol envs cap ref snap <<< "$v"
  snap=${snap:-glm_gguf_tiny}
  G="SNAP=$snap REF=$snap/$ref.json GQ_EXACT=1 KV16=0 COLI_SPEC_GATE=0 USAGE_SAVE=0"
  rm -f gl.usage gl-cpu.f32 gl-cu.f32
  env $G $envs COLI_USAGE=gl.usage DUMP=gl-cpu.f32 ./colibri $cap > gl-cpu.log 2>&1
  rm -f gl.usage
  env $G $envs $X COLI_USAGE=gl.usage DUMP=gl-cu.f32 ./colibri $cap > gl-cu.log 2>&1
  gltok gl-cpu.log > a.tok; gltok gl-cu.log > b.tok
  fw=$(sed -n 's/^\[CUDA\] colibri chain: \([0-9][0-9]*\) forwards.*/\1/p' gl-cu.log | tail -1)
  d=$(dev_count colibri gl-cu.log)
  lg=$(lclose gl-cpu.f32 gl-cu.f32 "$tol"); lok=$?
  if [ -s a.tok ] && cmp -s a.tok b.tok && [ "${fw:-0}" -gt 0 ] && [ "$d" -gt 0 ] && [ $lok = 0 ]; then
    ok "colibri GGUF $tag: tokens = CPU, $lg, $fw chain forwards, device $d"
  else bad "colibri GGUF $tag (${fw:-0} chain forwards, device $d, $lg)"; diff a.tok b.tok | head -4; tail -3 gl-cu.log; fi
done
rm -f gl.usage

# mimo: the greedy tokens, MXFP4 experts with an f32 trunk and with the release's
for bits in 32 0; do
  env MIMO_DENSE_BITS=$bits COLI_TEMP=0 ./mimo mimo_tiny --ids "1 2 3 4 5 6 7 8" --ngen 6 > mimo-cpu.txt 2> mimo-cpu.err
  env MIMO_DENSE_BITS=$bits COLI_TEMP=0 $X ./mimo mimo_tiny --ids "1 2 3 4 5 6 7 8" --ngen 6 > mimo-cu.txt 2> mimo-cu.err
  d=$(dev_count mimo mimo-cu.err)
  mats=$(grep -aoE "mimo: [0-9]+ dense matrices on the device" mimo-cu.err | sed -E 's/.*: ([0-9]+) dense.*/\1/' | head -1)
  # the f32 trunk goes to the card (cu_dense.h); the release's bf16 and FP8 stay on the CPU
  if [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-cu.txt && [ "$d" -gt 0 ] && { [ $bits != 32 ] || [ "${mats:-0}" -gt 0 ]; }; then
    ok "mimo dense bits $bits: tokens = CPU, device $d, ${mats:-0} dense matrices on the card"
  else bad "mimo dense bits $bits (device $d)"; diff mimo-cpu.txt mimo-cu.txt | head -4; tail -3 mimo-cu.err; fi
done

# kimi_k3: a GGUF's experts, the tokens and every logits row
ids="5 6 7 8 20 21 22 23"; G="GQ_EXACT=1 K3_MLA_BITS=32 COLI_TEMP=0 USAGE_SAVE=0"
for v in "all:" "budget of two:COLI_VK_TIER_GB=0.0001" "host pool:COLI_VK_TIER_GB=0.0001 COLI_VK_TIER_HOST=1"; do
  IFS=: read -r tag envs <<< "$v"
  env $G $envs K3_CUDA=0 K3_VAL_LOGITS=cpu.f32 ./kimi_k3 k3_gguf_tiny --ids "$ids" --ngen 8 2> k3-cpu.err | sed 's/ *TUNE.*//' > k3-cpu.tok
  env $G $envs K3_CUDA=1 $X K3_VAL_LOGITS=cu.f32 ./kimi_k3 k3_gguf_tiny --ids "$ids" --ngen 8 2> k3-cu.err | sed 's/ *TUNE.*//' > k3-cu.tok
  d=$(dev_count kimi_k3 k3-cu.err)
  lg=$(python3 -c "
import struct
a=open('cpu.f32','rb').read(); b=open('cu.f32','rb').read()
n=min(len(a),len(b))//4
x=struct.unpack(f'{n}f',a[:4*n]); y=struct.unpack(f'{n}f',b[:4*n])
print(len(a)==len(b) and max(abs(p-q) for p,q in zip(x,y))/max(abs(v) for v in x) < 3e-5, '%.1e' % (max(abs(p-q) for p,q in zip(x,y))/max(abs(v) for v in x)))")
  if [ -s k3-cpu.tok ] && cmp -s k3-cpu.tok k3-cu.tok && [ "${lg%% *}" = True ] && [ "$d" -gt 0 ]; then ok "kimi_k3 GGUF $tag: tokens = CPU, logits within ${lg#* } of the largest, device $d"
  else bad "kimi_k3 GGUF $tag (device $d, logits $lg)"; tail -3 k3-cu.err; fi
done
# the tier's second device (COLI_VK_DEV2=0: the same card as a second logical device, the test mode of
# tests/vulkan_dev2.sh), the primary held to a few experts so the second one holds the rest, against the CPU:
# decode and evict (COLI_VK_EXPERTS2=1) the CPU's tokens, the second device took experts and served rows; fault
# (COLI_VK_DEV2_FAULT=2: its second batch fails) the CPU's tokens, the second device stopped, the tier went on
d2_line() { sed -n "s/^\[CUDA\] tier $1 run: .* | second device \(.*\)/\1/p" "$2" | tail -1; }
d2_ok() {   # <engine> <log> <case>
  local l n; l=$(d2_line "$1" "$2"); n=$(echo "$l" | sed -n 's/.*), \([0-9][0-9]*\) rows in .*/\1/p')
  grep -qaE "^\[CUDA\] tier $1: second device .*, budget .* = [1-9][0-9]* experts" "$2" || return 1
  if [ "$3" = fault ]; then
    grep -qa "^\[CUDA\] tier $1: the second device's batch failed, its experts go back to the CPU" "$2" && echo "$l" | grep -q '(stopped)$'
  else [ "${n:-0}" -gt 0 ]; fi
}
d2ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
for k in decode evict fault; do
  x="COLI_VK_DEV2=0"; case $k in evict) x="$x COLI_VK_EXPERTS2=1";; fault) x="$x COLI_VK_DEV2_FAULT=2";; esac
  rm -f d2.usage; env GLM53_BITS=32 USAGE_SAVE=0 COLI_USAGE=d2.usage ./glm53 --model glm53_stream-i4 --ids $d2ids --greedy 6 > d2-cpu.log 2>&1
  rm -f d2.usage; env GLM53_BITS=32 USAGE_SAVE=0 COLI_USAGE=d2.usage $X COLI_VK_TIER_GB=0.00006 $x ./glm53 --model glm53_stream-i4 --ids $d2ids --greedy 6 > d2-cu.log 2>&1
  grep -aE '^teacher_forcing|^greedy' d2-cpu.log > a.tok; grep -aE '^teacher_forcing|^greedy' d2-cu.log > b.tok
  if [ -s a.tok ] && cmp -s a.tok b.tok && d2_ok glm53 d2-cu.log $k; then ok "glm53 dev2 $k: tokens = CPU, second device $(d2_line glm53 d2-cu.log)"
  else bad "glm53 dev2 $k"; diff a.tok b.tok | head -4; grep -a "tier glm53" d2-cu.log | tail -3; fi
done
for k in decode fault; do
  x="COLI_VK_DEV2=0"; [ $k = fault ] && x="$x COLI_VK_DEV2_FAULT=2"
  env SNAP=dsv41_tiny ./deepseek_v41 8 dsv41_tiny/ref.json > d2-v41-cpu.txt 2> d2-v41-cpu.err; rc0=$?
  rm -f d2.usage; env $X $x COLI_USAGE=d2.usage COLI_VK_TIER_GB=0.00005 SNAP=dsv41_tiny ./deepseek_v41 8 dsv41_tiny/ref.json > d2-v41-cu.txt 2> d2-v41-cu.err; rc=$?
  if [ $rc0 = 0 ] && [ $rc = 0 ] && cmp -s d2-v41-cpu.txt d2-v41-cu.txt && d2_ok deepseek_v41 d2-v41-cu.err $k; then
    ok "deepseek_v41 dev2 $k: output = CPU, oracle passed, second device $(d2_line deepseek_v41 d2-v41-cu.err)"
  else bad "deepseek_v41 dev2 $k (rc $rc0/$rc)"; diff d2-v41-cpu.txt d2-v41-cu.txt | head -4; grep -a "tier deepseek_v41" d2-v41-cu.err | tail -3; fi
done

echo "fails: $fails"
[ "$fails" = 0 ]
