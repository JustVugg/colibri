#!/usr/bin/env bash
# coli_serve.sh <ctx> <port> <tag> [ENV=val ...]: colibri dev serving Inkling 2 Mini Q3_K_XL on the RX via Vulkan,
# on a fresh copy of the step's history; waits until it answers, prints the engine's PID
T=${TREE:-$TREE}
E=$KIT
M=$MODEL
ctx=$1; port=$2; tag=$3; shift 3
rm -rf $M; cp -a ${HIST:-$E/step1/hist} $M
( cd $T/c && exec env LD_LIBRARY_PATH=$VKLIB:/run/opengl-driver/lib \
  VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json MESA_VK_DEVICE_SELECT=1002:7590 \
  COLI_VULKAN=1 COLI_VK_SHADERS=$T/c/shaders OMP_NUM_THREADS=8 "$@" \
  python3 ./coli serve --model $M --ctx $ctx --no-tune-profile --host 127.0.0.1 --port $port ) > $E/logs/coli-$tag.log 2>&1 < /dev/null &
for i in $(seq 400); do curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1 && break; sleep 1; done
curl -s -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1 || { echo DIED; tail -5 $E/logs/coli-$tag.log >&2; exit 1; }
pgrep -n -f "/c/inkling [0-9]"
