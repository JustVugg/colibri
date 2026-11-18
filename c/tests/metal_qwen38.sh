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

[ $fail -eq 0 ] && echo "metal_qwen38: PASS" || { echo "metal_qwen38: FAIL"; exit 1; }
