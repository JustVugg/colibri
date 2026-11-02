#!/bin/sh
# Build the Vulkan engine and refuse a silent CPU relink: after a VK=0 build the
# binary keeps its name, loads Vulkan with dlopen and answers no ldd question,
# so a benchmark would quietly measure the CPU path. A VK-only string is the
# tell (LTO can inline symbols away, but the messages stay).
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$here/../../c"
make -f Makefile.deepseek-v4 VK=1 ARCH="${ARCH:-native}" -j"${JOBS:-8}" deepseek-v4
if ! strings deepseek_v4 | grep -q 'VK_EXT_memory_budget'; then
    echo "build-vk: deepseek_v4 has no Vulkan strings (a VK=0 relink overwrote it?)" >&2
    exit 1
fi
echo "build-vk: deepseek_v4 is a Vulkan build"
