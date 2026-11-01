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
make -j"$(nproc)" glm53 deepseek_v41 mimo kimi_k3 CUDA=1 > cuda_tier_build.log 2>&1 || { tail cuda_tier_build.log; exit 1; }
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
echo "fails: $fails"
[ "$fails" = 0 ]
