#!/usr/bin/env bash
# Fetch the sm120 DeepGEMM headers the DeepSeek V4 CUDA tier's DeepGEMM flavour
# needs (cuda-dsv4-dg-dll on Windows, Makefile.deepseek-v4 DEEPGEMM=1 on Linux)
# at a pinned commit, verify the pin, and apply patches-deepgemm-sm120-msvc.patch.
#
#   tools/fetch_deepgemm.sh [<pinned-sha>] [<dest>]
#
# Defaults: the pin below and c/third_party/deepgemm (gitignored). Nothing is
# vendored in this repository: the tree is a git checkout of
# https://github.com/bvolpato/DeepGEMM (the community sm120 port — upstream
# DeepGEMM has no sm120 kernels; MIT) plus its NVIDIA CUTLASS/CuTe submodule
# (BSD-3-Clause), both at the commits recorded here. Attribution lives in
# THIRD_PARTY_NOTICES.md regardless of whether the tree is present.
#
# Idempotent: a tree already at the pin with the patch applied is left alone.
# A tree at a different commit is refetched (bump the pin in one place). The
# checkout is shallow (one commit each), ~40 MB.
set -euo pipefail

DEEPGEMM_REPO=${DEEPGEMM_REPO:-https://github.com/bvolpato/DeepGEMM.git}
DEEPGEMM_PIN_DEFAULT=39fb4447a062b418fd08ce17cd308adb28559417   # branch sm120-full, "Add SM120 heuristic calibration harness"
CUTLASS_PIN_DEFAULT=f3fde58372d33e9a5650ba7b80fc48b3b49d40c8    # third-party/cutlass gitlink at that commit

pin=${1:-${DEEPGEMM_PIN:-$DEEPGEMM_PIN_DEFAULT}}
here=$(cd "$(dirname "$0")/.." && pwd)          # c/
dest=${2:-${DEEPGEMM_HOME:-$here/third_party/deepgemm}}
case "$dest" in /*|[A-Za-z]:*) ;; *) dest="$here/$dest" ;; esac
patch="$here/patches-deepgemm-sm120-msvc.patch"
cutlass_pin=${CUTLASS_PIN:-$CUTLASS_PIN_DEFAULT}
marker="$dest/.coli-deepgemm-fetched"

die() { echo "fetch_deepgemm: $*" >&2; exit 1; }
command -v git >/dev/null 2>&1 || die "git is required (MSYS2: pacman -S git)"
[ -f "$patch" ] || die "missing $patch"
# The patch is LF in git; on a core.autocrlf checkout it arrives CRLF, so
# normalise before hashing/applying (git apply is strict about CR).
patch_sum=$(tr -d '\r' < "$patch" | git hash-object --stdin)

if [ -f "$marker" ] && [ "$(cat "$marker" 2>/dev/null)" = "$pin $cutlass_pin $patch_sum" ] \
   && [ -f "$dest/deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh" ]; then
    echo "fetch_deepgemm: $dest already at $pin (patched)"
    exit 0
fi

if [ -e "$dest" ] && [ ! -d "$dest/.git" ]; then
    die "$dest exists but is not a git checkout; move it away or set DEEPGEMM_HOME"
fi
if [ -d "$dest/.git" ]; then
    echo "fetch_deepgemm: $dest is not at $pin — refetching"
    rm -rf "$dest"
fi

mkdir -p "$dest"
git -C "$dest" init -q
# The checkout must stay LF regardless of a global core.autocrlf (Windows/MSYS2
# users): the patch is LF and nvcc/MSVC do not care about the endings.
git -C "$dest" config core.autocrlf false
git -C "$dest" config core.eol lf
git -C "$dest" remote add origin "$DEEPGEMM_REPO"
echo "fetch_deepgemm: fetching $DEEPGEMM_REPO @ $pin"
git -C "$dest" fetch -q --depth 1 origin "$pin"
git -C "$dest" checkout -q FETCH_HEAD
git -C "$dest" -c core.autocrlf=false -c core.eol=lf submodule update -q --init --depth 1 third-party/cutlass

# Verify the pins: the commit ids are the checksums (content-addressed), for
# the port and for the CUTLASS gitlink it records.
got=$(git -C "$dest" rev-parse HEAD)
[ "$got" = "$pin" ] || die "DeepGEMM checkout is $got, expected $pin"
gotc=$(git -C "$dest/third-party/cutlass" rev-parse HEAD)
[ "$gotc" = "$cutlass_pin" ] || die "CUTLASS submodule is $gotc, expected $cutlass_pin"
[ -f "$dest/deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh" ] || die "no sm120 kernels in $dest (wrong pin?)"

# MSVC ignores alignas(64) on by-value CUtensorMap kernel parameters; the
# patch pads them (and a few constexpr/inline-asm host fixes for cl.exe).
tr -d '\r' < "$patch" | git -C "$dest" apply --check || die "patch does not apply cleanly at $pin"
tr -d '\r' < "$patch" | git -C "$dest" apply
echo "$pin $cutlass_pin $patch_sum" > "$marker"
echo "fetch_deepgemm: ready at $dest (DeepGEMM $pin, CUTLASS $cutlass_pin, patch $patch_sum)"
