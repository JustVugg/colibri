#!/bin/sh
# The qwen38 engine built with METAL=1 (make test-qwen38-metal-engine), on the tiny
# fixtures. Run from c/ with ./qwen38 already built with METAL=1; PYTHON is the
# interpreter make passes on for the fixture generator.
#
# decide reasons: every startup case the Metal chain does not run turns it off with
#   its one line, and the CPU engine's tokens are the reference's.
# cpu unchanged: qwen38-tiny-check passes on the METAL=1 binary with COLI_METAL=0.
# gguf fixture cpu: the CPU engine on the tiny GGUF (qwen38_tiny_256_gguf, imported in
#   link mode from the safetensors fixture qwen38_tiny_256) gives the same tokens as on
#   its safetensors source, and matches the GGUF's own ref.json (rebased onto the CPU).
# weights match: with Q38M_CHECK_WEIGHTS=1 the chain's start runs one random row through
#   every dense matrix of layers 0 and 3 on the GPU (q38m_mm) and through the CPU's
#   q38_dense_matmul, and prints `[q38m-check] <name> worst=<e>` (Ruling 9 scale, FAIL over
#   2e-5): the BF16 safetensors qwen38_tiny, and the tiny GGUF (Q8_0 trunk, BF16 indexer,
#   F32), in fast and exact mode.
set -u
cd "$(dirname "$0")/.." || exit 1
fail=0
PREFIX='[qwen38-metal] the Metal chain is off: '
SUFFIX='; the CPU engine runs the model'

# run <fixture> <expected off reason, or "" for none> <env...>
run() {
    fx=$1; reason=$2; shift 2
    out=$(env "$@" OMP_NUM_THREADS=2 SNAP=./$fx ./qwen38 1 8 ./$fx/ref.json 2>&1)
    rc=$?
    what="$fx $*"
    if [ $rc -ne 0 ]; then echo "FAIL [$what] exit $rc"; echo "$out" | tail -5; fail=1; return; fi
    if [ -n "$reason" ]; then
        n=$(printf '%s\n' "$out" | grep -cF "$PREFIX$reason$SUFFIX")
        all=$(printf '%s\n' "$out" | grep -cF "$PREFIX")
        if [ "$n" -ne 1 ] || [ "$all" -ne 1 ]; then
            echo "FAIL [$what] expected exactly one line: $PREFIX$reason$SUFFIX"
            printf '%s\n' "$out" | grep -F '[qwen38-metal]'; fail=1; return
        fi
    fi
    if ! printf '%s\n' "$out" | grep -qF 'Matching tokens: 8/8'; then
        echo "FAIL [$what] tokens differ from the reference"
        printf '%s\n' "$out" | grep -E 'Matching tokens|C engine'; fail=1; return
    fi
    echo "ok   [$what] ${reason:-chain decided on}"
}

echo "== decide reasons"
make -s qwen38-tiny-generate PYTHON="${PYTHON:-python3}" >/dev/null 2>&1 || { echo "FAIL fixture generation"; exit 1; }
make -s qwen38-tiny-mtp-generate PYTHON="${PYTHON:-python3}" >/dev/null 2>&1 || { echo "FAIL mtp fixture generation"; exit 1; }
run qwen38_tiny "COLI_METAL=0" COLI_METAL=0
run qwen38_tiny "KV_SLOTS>1 (one conversation on the GPU)" KV_SLOTS=2
# qwen38_tiny has no MTP head (Q38_MTP=1 refuses it before the chain decides): its
# twin with the head
run qwen38_tiny_mtp "MTP (Q38_MTP=1) needs the CPU engine until M5" Q38_MTP=1

echo "== cpu unchanged"
log=$(mktemp) || exit 1
if COLI_METAL=0 make -s qwen38-tiny-check METAL=1 PYTHON="${PYTHON:-python3}" > "$log" 2>&1; then
    n=$(grep -c 'Matching tokens: 8/8' "$log")
    if [ "$n" -eq 16 ]; then echo "ok   [qwen38-tiny-check COLI_METAL=0] 16/16 configurations"
    else echo "FAIL [qwen38-tiny-check COLI_METAL=0] $n of 16 configurations match"; fail=1; fi
else
    echo "FAIL [qwen38-tiny-check COLI_METAL=0]"; tail -5 "$log"; fail=1
fi
rm -f "$log"

echo "== gguf fixture cpu"
if make -s qwen38-tiny-gguf-generate PYTHON="${PYTHON:-python3}" >/dev/null 2>&1; then
    # the GGUF against the safetensors source: the same tokens from the CPU engine
    # (the source's ref.json is the PyTorch reference, which the engine itself does not
    # reproduce past token 5 on this geometry; the GGUF's ref.json is the CPU's on it)
    toks() { env COLI_METAL=0 OMP_NUM_THREADS=2 SNAP=./$1 ./qwen38 1 8 ./$1/ref.json 2>&1 | grep '^C engine'; }
    a=$(toks qwen38_tiny_256); b=$(toks qwen38_tiny_256_gguf)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "ok   [gguf fixture cpu] same tokens as the safetensors source: $b"
    else echo "FAIL [gguf fixture cpu] source '$a' vs gguf '$b'"; fail=1; fi
    run qwen38_tiny_256_gguf "COLI_METAL=0" COLI_METAL=0
else
    echo "FAIL gguf fixture generation"; fail=1
fi

echo "== weights match"
# check <tokens: 1 = assert the reference's> <fixture> <env...>: check lines for both layers,
# none over the tolerance, and the CPU engine's tokens unchanged (the chain still declines
# every step)
check() {
    tok=$1; fx=$2; shift 2
    out=$(env Q38M_CHECK_WEIGHTS=1 "$@" OMP_NUM_THREADS=2 SNAP=./$fx ./qwen38 1 8 ./$fx/ref.json 2>&1)
    rc=$?
    what="$fx $*"
    # a ref.json mismatch exits 1 after the run: only a crash counts when tokens are not asserted
    if [ $rc -ne 0 ] && { [ "$tok" = 1 ] || [ $rc -ne 1 ]; }; then
        echo "FAIL [weights $what] exit $rc"; echo "$out" | tail -5; fail=1; return
    fi
    n0=$(printf '%s\n' "$out" | grep -c '^\[q38m-check\] L0\.')
    n3=$(printf '%s\n' "$out" | grep -c '^\[q38m-check\] L3\.')
    bad=$(printf '%s\n' "$out" | grep '^\[q38m-check\]' | grep -F 'FAIL')
    if [ "$n0" -lt 5 ] || [ "$n3" -lt 5 ]; then
        echo "FAIL [weights $what] check lines: layer 0 $n0, layer 3 $n3"; printf '%s\n' "$out" | grep -F 'q38m'; fail=1; return
    fi
    if [ -n "$bad" ]; then echo "FAIL [weights $what]"; printf '%s\n' "$bad"; fail=1; return; fi
    if [ "$tok" = 1 ] && ! printf '%s\n' "$out" | grep -qF 'Matching tokens: 8/8'; then
        echo "FAIL [weights $what] tokens differ from the reference"; fail=1; return
    fi
    # a GGUF's blocks are the CPU's own bytes, mapped in place (q38m_weight), not copied
    case $fx in *_gguf)
        if ! printf '%s\n' "$out" | grep -qE '^\[qwen38-metal\] dense weights: [1-9][0-9]* mapped in place'; then
            echo "FAIL [weights $what] no matrix mapped in place"; printf '%s\n' "$out" | grep -F 'dense weights'; fail=1; return
        fi;;
    esac
    worst=$(printf '%s\n' "$out" | grep '^\[q38m-check\]' | sed 's/.*worst=\([^ ]*\).*/\1/' | sort -g | tail -1)
    echo "ok   [weights $what] $n0 + $n3 matrices, worst $worst"
}
for exact in 0 1; do
    check 1 qwen38_tiny QWEN36_METAL_EXACT=$exact
    # GQ_EXACT=1: the CPU's float sums for the GGUF's blocks, as q36_dot_gq's. By default
    # gq_matmul quantizes the activation to int8 per 32 (gq_act), which q36_dot_gq does not:
    # about 1e-3 on this scale, the activation's rounding and not the weights'. The GGUF's
    # ref.json is the CPU's default mode, which GQ_EXACT=1 leaves at token 6: no token check
    check 0 qwen38_tiny_256_gguf QWEN36_METAL_EXACT=$exact GQ_EXACT=1
done

[ $fail -eq 0 ] && echo "metal_qwen38: PASS" || { echo "metal_qwen38: FAIL"; exit 1; }
