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
#
# The M2 forward (the trunk on the GPU, the routed experts on the CPU), against the CPU
# engine on the same fixture (Ruling P2-3) in reference precision (Ruling P2-4: GQ_EXACT=1,
# Q38_TRUNK_CPU_INT8=0, Q38_NATIVE_BF16=1, on both sides, so the chain's CPU experts are
# the reference's too). Token equality against the default-mode CPU is reported (info).
# m2 tiny: qwen38_tiny (BF16) and the tiny GGUF, exact mode (QWEN38_METAL_EXACT=1: the same
#   tokens, final logits within 1e-4 x max|logit|) and fast mode (the same tokens); no off
#   line, the chain's report with forwards > 0.
# metal_state_handoff: Q38M_DECLINE_EVERY=2 (debug) declines every second step, so the CPU
#   and the chain hand the state back and forth (q38m_cpu_step, the KV watermarks): the
#   CPU's tokens.
# kv growth: a 1,300-token prompt (qwen38_tiny with max_position_embeddings raised, a
#   generated prompt): the device caches grow 256, 512, 1024, 2048; the CPU's tokens.
# layer test: Q38M_LAYER_CHECK=1 (debug) runs the CPU's q38_layer_forward beside the first
#   step and compares each layer's output streams, `[q38m-layer] L<i> worst=<e>` (max
#   |gpu-cpu| over max |cpu|), FAIL over 1e-4, exact mode.
# serve: tests/metal_qwen38_serve.py (two turns, a regenerated answer, a parked conversation:
#   every served step's logits against a fresh engine's; KV_SLOTS=2 in a serve), then
#   tests/test_qwen38_prefix_serve.py and test_qwen38_prefix_store.py with the chain on.
set -u
cd "$(dirname "$0")/.." || exit 1
fail=0
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
REFP="GQ_EXACT=1 Q38_TRUNK_CPU_INT8=0 Q38_NATIVE_BF16=1"   # Ruling P2-4
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
# KV_SLOTS is the serve's (g_q38_mux_slots): a run that is not a serve keeps the chain on
# (m2 tiny below runs one with KV_SLOTS=2); the serve case is in tests/metal_qwen38_serve.py
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
if make -s qwen38-tiny-gguf-generate METAL=1 PYTHON="${PYTHON:-python3}" >/dev/null 2>&1; then
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

# eng <name> <fixture dir> <ref.json> <env...>: one run, its output in $tmp/<name>.log; a
# reference mismatch exits 1 after the run (the gates compare runs, not the ref): only
# another code fails
eng() {
    nm=$1; dir=$2; ref=$3; shift 3
    env "$@" OMP_NUM_THREADS=2 SNAP=$dir ./qwen38 1 8 "$ref" > "$tmp/$nm.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && [ $rc -ne 1 ]; then echo "FAIL [$nm $*] exit $rc"; tail -5 "$tmp/$nm.log"; fail=1; return 1; fi
    return 0
}
toks() { grep '^C engine' "$tmp/$1.log"; }
forwards() { sed -n 's/^\[metal\] qwen38 chain: \([0-9]*\) forwards, \([0-9]*\) declined.*/\1 \2/p' "$tmp/$1.log" | tail -1; }
# relerr <a.bin> <b.bin>: max |a-b| / max |a| over two DUMP files (float32)
relerr() {
    "${PYTHON:-python3}" -c '
import array, sys
a = array.array("f"); b = array.array("f")
a.frombytes(open(sys.argv[1], "rb").read()); b.frombytes(open(sys.argv[2], "rb").read())
if len(a) != len(b) or not a: print("inf"); sys.exit()
m = max(abs(x) for x in a)
print("%.3g" % (max(abs(x - y) for x, y in zip(a, b)) / (m or 1.0)))' "$1" "$2"
}
# chain_on <name>: no off line, the report with forwards > 0
chain_on() {
    if grep -qF "$PREFIX" "$tmp/$1.log"; then echo "FAIL [$1] the chain is off:"; grep -F "$PREFIX" "$tmp/$1.log"; fail=1; return 1; fi
    f=$(forwards "$1")
    if [ -z "$f" ] || [ "${f%% *}" -eq 0 ]; then echo "FAIL [$1] the chain ran no forward (report: '${f:-none}')"; fail=1; return 1; fi
    return 0
}

echo "== m2 tiny"
for fx in qwen38_tiny qwen38_tiny_256_gguf; do
    eng cpu_$fx ./$fx ./$fx/ref.json COLI_METAL=0 $REFP DUMP=$tmp/cpu_$fx.bin || continue
    eng cpud_$fx ./$fx ./$fx/ref.json COLI_METAL=0 || continue
    want=$(toks cpu_$fx)
    for exact in 1 0; do
        nm=m2_${fx}_exact$exact
        eng $nm ./$fx ./$fx/ref.json QWEN38_METAL_EXACT=$exact QWEN38_METAL_EXPERTS=0 $REFP DUMP=$tmp/$nm.bin || continue
        chain_on $nm || continue
        got=$(toks $nm)
        if [ -z "$want" ] || [ "$got" != "$want" ]; then echo "FAIL [$nm] tokens '$got' vs the CPU's '$want'"; fail=1; continue; fi
        if [ $exact = 1 ]; then
            e=$(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin")
            if ! awk -v e="$e" 'BEGIN { exit !(e <= 1e-4) }'; then echo "FAIL [$nm] final logits off by $e x max|logit|"; fail=1; continue; fi
            echo "ok   [$nm] the CPU's tokens, logits within $e x max, forwards $(forwards $nm)"
        else
            echo "ok   [$nm] the CPU's tokens, forwards $(forwards $nm) (logits within $(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin") x max, not gated)"
        fi
        [ "$got" = "$(toks cpud_$fx)" ] && same=same || same=differ
        echo "info [$nm] against the default-mode CPU: $same"
    done
done
# KV_SLOTS outside a serve: the chain stays on
eng m2_kvslots ./qwen38_tiny ./qwen38_tiny/ref.json KV_SLOTS=2 && chain_on m2_kvslots && echo "ok   [m2 KV_SLOTS=2 not a serve] chain on"

echo "== metal_state_handoff"
for fx in qwen38_tiny qwen38_tiny_256_gguf; do
    nm=handoff_$fx
    eng $nm ./$fx ./$fx/ref.json Q38M_DECLINE_EVERY=2 QWEN38_METAL_EXACT=1 $REFP || continue
    chain_on $nm || continue
    f=$(forwards $nm)
    if [ "${f#* }" -eq 0 ]; then echo "FAIL [$nm] nothing declined ($f)"; fail=1; continue; fi
    if [ "$(toks $nm)" != "$(toks cpu_$fx)" ]; then echo "FAIL [$nm] tokens '$(toks $nm)' vs '$(toks cpu_$fx)'"; fail=1; continue; fi
    echo "ok   [$nm] the CPU's tokens (forwards, declined: $f)"
done

echo "== kv growth"
mkdir -p "$tmp/long"
for f in qwen38_tiny/*; do ln -s "$PWD/$f" "$tmp/long/"; done
rm -f "$tmp/long/config.json" "$tmp/long/ref.json" "$tmp/long/.coli_usage"
"${PYTHON:-python3}" -c '
import json, sys
c = json.load(open("qwen38_tiny/config.json"))
(c.get("text_config") or c)["max_position_embeddings"] = 2048
json.dump(c, open(sys.argv[1] + "/config.json", "w"))
x, ids = 12345, []
for _ in range(1300):
    x = (x * 1103515245 + 12345) % 2**31
    ids.append(3 + x % 61)
json.dump({"schema_version": 1, "prompt_ids": ids, "full_ids": ids + [3] * 8}, open(sys.argv[1] + "/ref.json", "w"))' "$tmp/long"
if eng long_cpu "$tmp/long" "$tmp/long/ref.json" COLI_METAL=0 $REFP &&
   eng long_mtl "$tmp/long" "$tmp/long/ref.json" QWEN38_METAL_EXACT=1 $REFP && chain_on long_mtl; then
    if [ -n "$(toks long_cpu)" ] && [ "$(toks long_mtl)" = "$(toks long_cpu)" ]; then echo "ok   [kv growth 1300 tokens] the CPU's tokens: $(toks long_mtl)"
    else echo "FAIL [kv growth 1300 tokens] '$(toks long_mtl)' vs the CPU's '$(toks long_cpu)'"; fail=1; fi
fi

echo "== layer test"
for fx in qwen38_tiny qwen38_tiny_256_gguf; do
    nm=layers_$fx
    eng $nm ./$fx ./$fx/ref.json Q38M_LAYER_CHECK=1 QWEN38_METAL_EXACT=1 $REFP || continue
    n=$(grep -c '^\[q38m-layer\] L' "$tmp/$nm.log")
    if [ "$n" -lt 4 ] || grep '^\[q38m-layer\]' "$tmp/$nm.log" | grep -qF FAIL; then
        echo "FAIL [$nm] $n layer lines"; grep '^\[q38m-layer\]' "$tmp/$nm.log"; fail=1; continue
    fi
    worst=$(grep '^\[q38m-layer\] L' "$tmp/$nm.log" | sed 's/.*worst=\([^ ]*\).*/\1/' | sort -g | tail -1)
    echo "ok   [$nm] $n layers, worst $worst"
done

echo "== serve"
"${PYTHON:-python3}" tests/metal_qwen38_serve.py || fail=1
# the engine's own prefix suites (the live state, the snapshot, the pins, the parked
# conversation, the cancels; the store's checkpoints and rows) with the chain on: they reach
# every state hook (q38m_sync_host, q38m_host_wrote, q38m_rollback)
mkdir -p "$tmp/tok"
for f in qwen38_tiny/*; do ln -s "$PWD/$f" "$tmp/tok/"; done
"${PYTHON:-python3}" tools/make_edge_tiny_tokenizer.py --vocab-size 64 "$tmp/tok" >/dev/null
for t in test_qwen38_prefix_serve test_qwen38_prefix_store; do
    if QWEN38_TINY="$tmp/tok" "${PYTHON:-python3}" -W ignore tests/$t.py > "$tmp/$t.log" 2>&1; then
        echo "ok   [$t on the chain] $(grep '^Ran' "$tmp/$t.log")"
    else
        echo "FAIL [$t on the chain]"; grep -E '^(FAIL|ERROR):' "$tmp/$t.log"; fail=1
    fi
done

[ $fail -eq 0 ] && echo "metal_qwen38: PASS" || { echo "metal_qwen38: FAIL"; exit 1; }
