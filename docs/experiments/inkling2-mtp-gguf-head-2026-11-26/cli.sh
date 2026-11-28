#!/usr/bin/env bash
# cli.sh <tag> <prompt> <n> [ENV=val ...]: the engine's own CLI (greedy, --chat), on a fresh copy of the drafter's history
E=$KIT; T=${TREE:-$DEV}
M=$MODEL
tag=$1; pf=$2; n=$3; shift 3
rm -rf $M; cp -a $E/mtp/hist $M
( cd $T/c && exec env SNAP=$M LD_LIBRARY_PATH=$VKLIB:/run/opengl-driver/lib \
  VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json MESA_VK_DEVICE_SELECT=1002:7590 \
  COLI_VULKAN=1 COLI_VK_SHADERS=$T/c/shaders COLI_VK_STAGED=1 OMP_NUM_THREADS=8 CTX_MAX=16384 "$@" \
  ./inkling --chat -f $pf -n $n ) > $E/mtp/cli-$tag.out 2> $E/logs/cli-$tag.err < /dev/null &
pid=$!
$E/step1/watchdog.sh $pid $E/logs/cli-$tag.err & w=$!
wait $pid; rc=$?; kill $w 2>/dev/null
echo "== $tag rc=$rc"
grep -E '^\[prefill' $E/mtp/cli-$tag.out
grep -E 'INK2_MTP|tier inkling: (on|no room)' $E/logs/cli-$tag.err | cut -c1-260
