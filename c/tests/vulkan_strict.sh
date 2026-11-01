#!/usr/bin/env bash
# The strict capability gate (COLI_VK_STRICT=1, docs/vulkan.md) end to end, on any
# Vulkan device (CI: Lavapipe over VK_ICD_FILENAMES): what a silent degradation
# costs under strict (exit 70 naming the op), and — as important — what it must NOT
# cost: legitimate refusals (a format with no device form) and fallbacks continue.
#
#   make vk-strict-check VK=1        # builds the binaries, then this script
#   bash tests/vulkan_strict.sh      # the same, from c/
#
# The cases:
#   the helper itself (tests/test_vk_degrade): no device needed, runs first;
#   the core shader absent: exit 70, op=core shader (a host with no Vulkan
#        device skips the rest: "no Vulkan device" in this run's own log);
#   an optional shader absent (qmatmul_gate_up.spv): exit 70, op=the shader;
#   a batch's wait gives up mid-run (COLI_VK_WAIT_FAULT): exit 70,
#        op=expert batch fence;
#   the default is untouched: unset and 0 give the same stderr (cmp), exit 0,
#        and the old fault line — nothing about strict anywhere;
#   the hatch is the base: all five gap-wave4 default levers at =0, and the
#        self-tuning knobs at the base's own numbers (MM8=0 CHAIN_GEMM_ROWS=0
#        TIER_GEMM_ROWS=16 STREAM_SLOTS=64 STREAM_BYPASS=0), vs COLI_VK_LEGACY=1
#        (every knob unset) agree byte for byte (stdout exact, stderr past the
#        hatch's own legacy lines) — per-lever =0 still wins, and the hatch
#        reproduces exactly that base;
#   the refusal is not fatal: a strict run of test_vk_tier serves the layers
#        whose extra form has no device form ("[VK] refused:", the run continues)
#        and aborts only at its deliberate final wait fault — exactly one
#        "strict: aborting" line, that one;
#   a strict run with no faults stays green: test_vk_chain exits 0 (this is
#        the anti-fatal-everything check: a Lavapipe with no cooperative matrices
#        and modest shared memory must pass strict);
#   exactly one "[VK] strict: on" boot line per run.
set -euo pipefail
cd "$(dirname "$0")/.."

TIER=tests/test_vk_tier
CHAIN=tests/test_vk_chain
DG=tests/test_vk_degrade
SPV=shaders/qmatmul.spv
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
rc_of() { set +e; "$@" >"$TMP/out" 2>"$TMP/err"; local rc=$?; set -e; echo "$rc"; }

for b in "$DG" "$TIER" "$CHAIN"; do
    [ -x "$b" ] || make "$b" VK=1 || fail "build $b"
done

#   — the helper's class table, no device.
"$DG" || fail "case: test_vk_degrade"

#   — the core shader absent from an empty shader directory (no argv: with one,
# test_vk_tier takes it over COLI_VK_SHADERS). A host with no Vulkan device says so
# here ("no Vulkan device"); that is a skip, not a failure.
mkdir "$TMP/none"
rc=$(rc_of env COLI_VK_STRICT=1 COLI_VK_SHADERS="$TMP/none" "$TIER")
if grep -q "no Vulkan device" "$TMP/out" "$TMP/err"; then
    echo "SKIP: no Vulkan device (case-case need one; case passed)"
    exit 0
fi
[ "$rc" = 70 ] || fail "case: exit 70 expected, got $rc"
grep -q "\[VK\] strict: on" "$TMP/err" || fail "case: the strict boot line"
grep -q "op=core shader" "$TMP/err" || fail "case: op=core shader"

#   — an optional shader absent: the deployment ships the full set, strict says
# which one is missing. case: exactly one boot line.
mkdir "$TMP/shv" && cp shaders/*.spv "$TMP/shv/" && rm "$TMP/shv/qmatmul_gate_up.spv"
rc=$(rc_of env COLI_VK_STRICT=1 COLI_VK_SHADERS="$TMP/shv" "$TIER")
[ "$rc" = 70 ] || fail "case: exit 70 expected, got $rc"
grep -q "op=qmatmul_gate_up.spv" "$TMP/err" || fail "case: op=qmatmul_gate_up.spv"
[ "$(grep -c '\[VK\] strict: on' "$TMP/err")" = 1 ] || fail "case: exactly one boot line"

#   — a wait gives up mid-run: the old line, then the gate's abort.
rc=$(rc_of env COLI_VK_STRICT=1 COLI_VK_WAIT_FAULT=1 "$TIER" "$SPV")
[ "$rc" = 70 ] || fail "case: exit 70 expected, got $rc"
grep -q "expert batch: fence wait failed" "$TMP/err" || fail "case: the old fault line kept"
grep -q "op=expert batch fence" "$TMP/err" || fail "case: op=expert batch fence"

#   — the default is byte-identical: unset and 0 agree with each other, exit 0,
# keep the old fault line, and print nothing about strict. The tier's exit lines
# carry wall-clock timings (ms, GB/s, %) and the decode-growth arm's live free-
# VRAM reading (GiB/MiB/KiB) that jitter between ANY two runs — the baseline
# against itself differs the same way — so those fields are masked for the
# comparison; everything else must match byte for byte.
norm() { sed -E 's/[0-9]+(\.[0-9]+)? (ms|us|GB\/s)/X/g; s/[0-9]+%/X%/g; s/[0-9]+(\.[0-9]+)? (GiB|MiB|KiB)/X/g; s/frag [0-9]+(\.[0-9]+)?/frag X/g' "${1:--}"; }
rc=$(rc_of env -u COLI_VK_STRICT "$TIER" "$SPV"); mv "$TMP/out" "$TMP/a.out"; mv "$TMP/err" "$TMP/a.err"
[ "$rc" = 0 ] || fail "case: unset must pass (rc $rc)"
rc=$(rc_of env COLI_VK_STRICT=0 "$TIER" "$SPV"); mv "$TMP/out" "$TMP/b.out"; mv "$TMP/err" "$TMP/b.err"
[ "$rc" = 0 ] || fail "case: 0 must pass (rc $rc)"
cmp -s "$TMP/a.out" "$TMP/b.out" || fail "case: unset and 0 stdout differ"
cmp -s <(norm "$TMP/a.err") <(norm "$TMP/b.err") || fail "case: unset and 0 stderr differ"
grep -q "expert batch: fence wait failed" "$TMP/a.err" || fail "case: the old fault line kept"
if grep -q "strict" "$TMP/a.out" "$TMP/a.err"; then fail "case: strict must not appear unset"; fi

#   — the hatch is the base (gap-wave4): the five default levers at =0 are the
# pre-flip standing, and COLI_VK_LEGACY=1 must reproduce exactly that — stdout
# byte for byte, stderr past the hatch's own "legacy (COLI_VK_LEGACY=1)" lines
# (an explicit env always wins over the hatch, so the =0 side prints nothing).
# The self-tuning defaults' five knobs join the leg the same way, each pinned to
# the BASE's own value (not =0: the tier's GEMM rows were 16, the stream slots
# 64), so the hatch case keeps covering the whole measured-shape flip.
OFF="COLI_VK_CHAIN_ATTN_FA=0 COLI_VK_TIER_EXPERTS=0 COLI_VK_TIER_DECODE_GROW=0 COLI_VK_TIER_DECODE_GROW_HEADROOM=0 COLI_VK_V4_DECODE_PATH=0 COLI_VK_V4_S1GEMV=0 COLI_VK_MM8=0 COLI_VK_CHAIN_GEMM_ROWS=0 COLI_VK_TIER_GEMM_ROWS=16 COLI_VK_TIER_STREAM_SLOTS=64 COLI_VK_TIER_STREAM_BYPASS=0"
rc=$(rc_of env -u COLI_VK_STRICT $OFF "$TIER" "$SPV"); mv "$TMP/out" "$TMP/c.out"; mv "$TMP/err" "$TMP/c.err"
[ "$rc" = 0 ] || fail "case: the =0 base must pass (rc $rc)"
rc=$(rc_of env -u COLI_VK_STRICT COLI_VK_LEGACY=1 "$TIER" "$SPV"); mv "$TMP/out" "$TMP/d.out"; mv "$TMP/err" "$TMP/d.err"
[ "$rc" = 0 ] || fail "case: LEGACY=1 must pass (rc $rc)"
cmp -s "$TMP/c.out" "$TMP/d.out" || fail "case: =0 and LEGACY=1 stdout differ"
cmp -s <(norm "$TMP/c.err") <(sed -E '/legacy \(COLI_VK_LEGACY=1\)/d' "$TMP/d.err" | norm) \
    || fail "case: =0 and LEGACY=1 stderr differ (past the hatch's own lines)"
grep -q "legacy (COLI_VK_LEGACY=1)" "$TMP/d.err" || fail "case: the hatch said its legacy lines"
grep -q "legacy (COLI_VK_LEGACY=1)" "$TMP/c.err" && fail "case: the =0 base must not print legacy lines"

#   — the refusal is not fatal: the extra layers' form has no device form, the
# tier says so and serves the rest; the ONLY abort of the run is its deliberate
# final wait fault (that is case's mechanism, driven by test_vk_tier's own timeout
# section, so the exit here is 70 by design).
rc=$(rc_of env COLI_VK_STRICT=1 "$TIER" "$SPV")
[ "$rc" = 70 ] || fail "case: the deliberate final fault aborts (rc $rc)"
[ "$(grep -c 'strict: aborting' "$TMP/err")" = 1 ] || fail "case: exactly one abort line"
grep -q "op=expert batch fence" "$TMP/err" || fail "case: the abort is the timeout fault"
grep -q "has no device form" "$TMP/err" || fail "case: the no-device-form refusal ran"
grep -q "\[VK\] refused:" "$TMP/err" || fail "case: the refusal is named, not fatal"

#   — no faults: strict stays green (Lavapipe passes: its missing cooperative
# matrices and small shared memory are refusals, not degradations). case again.
rc=$(rc_of env COLI_VK_STRICT=1 "$CHAIN" "$SPV")
[ "$rc" = 0 ] || fail "case: test_vk_chain under strict must pass (rc $rc)"
if grep -q "strict: aborting" "$TMP/err"; then fail "case: nothing aborted"; fi
[ "$(grep -c '\[VK\] strict: on' "$TMP/err")" = 1 ] || fail "case: exactly one boot line"

echo "PASS: strict gate"
