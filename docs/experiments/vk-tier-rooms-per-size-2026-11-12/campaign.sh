#!/usr/bin/env bash
# Three interleaved runs per arm, each server started on a copy of one history.
#   BASE=<tree at origin/dev b92eccd7>  TRIAL=<tree with this change>  (make -C c qwen36 VK=1 in both)
#   MODEL=<flint.colibri, from tools/gguf_import.py flint-UD-Q4_K_XL.gguf flint.colibri --mode link>
#   HIST=<a copy of MODEL after one earlier session, its .coli_usage included>
#   VKLIB=<the directory of libvulkan.so.1>  S=<where the logs go>
port=18110
for i in 1 2 3; do
  for arm in base trial; do
    tree=$BASE; [ $arm = trial ] && tree=$TRIAL
    cp -a $HIST/. $MODEL/
    port=$((port+1))
    echo "=== $arm run $i (MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB)"
    (cd $tree/c && nohup env LD_LIBRARY_PATH=$VKLIB:/run/opengl-driver/lib \
      VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json \
      MESA_VK_DEVICE_SELECT=1002:7590 \
      COLI_VULKAN=1 COLI_VK_STAGED=1 COLI_VK_TIER_HOST=0 COLI_VK_SHADERS=$tree/c/shaders \
      python3 ./coli serve --model $MODEL --ctx 16384 --no-tune-profile --no-think --ngen 200 \
        --host 127.0.0.1 --port $port > $S/c-$arm-$i.log 2>&1 &)
    for t in $(seq 180); do curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1 && break; sleep 1; done
    python3 bench.py http://127.0.0.1:$port/v1/chat/completions 1 $i
    (cd $tree/c && python3 ./coli stop --port $port | tail -1); sleep 3
    sed 's/\x1b\[[0-9;]*m//g' $S/c-$arm-$i.log | grep -i 'tier qwen36: on\|tier qwen36 turn'
  done
done
