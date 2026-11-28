#!/usr/bin/env bash
# ab.sh <rounds>: drafter off / dev's default / the fix's default / the fix with the target's head, interleaved
E=$KIT; P=$E/mtp/prompts
DEV=$DEV; FIX=$FIX
for r in $(seq 1 $1); do for pr in code prose; do
  for arm in off dev fix fixbf16; do
    case $arm in
      off)     tree=$FIX; env="INK2_MTP=0";;
      dev)     tree=$DEV; env="INK2_MTP=1";;
      fix)     tree=$FIX; env="INK2_MTP=1";;
      fixbf16) tree=$FIX; env="INK2_MTP=1 INK2_MTP_HEAD=bf16";;
    esac
    TREE=$tree $E/mtp/cli.sh ab-$pr-$arm-r$r $P/$pr.txt 512 $env 2>&1 | grep -E "^\[prefill|drafts kept|its head|not a float" | sed "s/^/$pr $arm r$r: /" | cut -c1-210
  done
done; done
echo "done $(date +%T) Tctl $(sensors | awk '/Tctl/{print $2}')"
