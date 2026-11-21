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
# m3 gpu experts (Plan 2 Task 8): the routed experts on the GPU from the GPU expert cache, on
#   the tiny GGUF (Q8_0 experts), against the same CPU run: exact mode the same tokens and
#   logits within 1e-4, fast mode the same tokens; the report's `experts on GPU: N, loads: L,
#   hits: H` with N, L > 0 and a layer with more experts than one group of top-k; a cache of two
#   payloads (QWEN38_EXPERT_CACHE_BYTES, one group: every group evicts the last) within 120 s;
#   Q38M_FAIL_SLOT_ALLOC=1 (debug) and QWEN38_METAL_EXPERTS=0 one off line each and the CPU's
#   experts; Q38M_REFUSE_AT=3 (debug) turns the chain off mid-run: the GPU cache released, the
#   CPU's expert cache back to its size; the usage history and route trace the CPU's; the
#   release's expert shape on a variant (16 experts, top-k 10, IQ4_XS/IQ4_NL: sub-groups of 8 and
#   2), exact, fast and the layer check; a 1,300-token prompt on the GGUF (chunks of 256 rows),
#   exact and fast.
# ahead fail: Q38M_AHEAD_FAIL=5 (debug: the fifth ahead router declines, mid-decode) and =start
#   (its buffers refused at the start): one `read-ahead off` line, the chain and the GPU experts
#   on, logits bit-identical to QWEN38_METAL_AHEAD=0 and the CPU's tokens.
# metal_state_handoff: Q38M_DECLINE_EVERY=2 (debug) declines every second step, so the CPU
#   and the chain hand the state back and forth (q38m_cpu_step, the KV watermarks): the
#   CPU's tokens and final logits (within 1e-4 x max), also on qwen38_decay (the tiny fixture
#   with its DeltaNet decay raised to about 0.9, so the recurrent state counts).
# lookup policy (Ruling P2-5): COLI_LOOKUP unset beside the chain gives one note line and no
#   drafts; COLI_LOOKUP=1 with forced drafts runs its verifies on the CPU: the CPU's tokens.
# kv growth: a 1,300-token prompt (qwen38_tiny with max_position_embeddings raised, a
#   generated prompt): the device caches grow 256, 512, 1024, 2048; the CPU's tokens and logits.
#   Then growth with valid rows: qwen38_decay served a 1,020-token prompt in segments of 300 rows
#   and 8 tokens across position 1024 (tests/metal_qwen38_serve.py --segments): two growths drop
#   valid device rows and upload them again; every step's logits the CPU's.
# long prompt, fast mode: 3,000 tokens on qwen38_decay with the fast kernels: the CPU's tokens
#   and logits, and Q38M_LAYER_CHECK=2 (each layer, the routes, the CPU's own one-ulp lines).
# layer test: Q38M_LAYER_CHECK=1 (debug) runs the CPU's q38_layer_forward beside the first
#   step and compares each layer's output streams, `[q38m-layer] L<i> worst=<e>` (max
#   |gpu-cpu| over max |cpu|), FAIL over 1e-4, exact mode (on the GGUF with the GPU experts,
#   whose GPU routes must be the CPU's rule on the same logits in every row); then Q38M_FORCE_ROUTES=1 and 2 on the
#   1,300-token prompt: rows held to the reference run's routes and selections (counted > 0).
# serve: tests/metal_qwen38_serve.py (two turns, a regenerated answer, a parked conversation:
#   every served step's logits against a fresh engine's; KV_SLOTS=2 in a serve), then
#   tests/test_qwen38_prefix_serve.py and test_qwen38_prefix_store.py with the chain on.
# serve metal frame: tests/metal_qwen38_serve.py --metal-frame on the tiny GGUF (the gateway's
#   METAL frame: one a turn, per-turn deltas, GPU-resident experts as EMAP tier 2; COLI_METAL=0).
# serve gpu error: tests/metal_qwen38_serve.py --gpu-error on the tiny GGUF (Q38M_FAIL_AT_STEP=3,
#   debug: a chain failure with the state on the GPU only ends the turn with ERROR GPU_FAILED, the
#   engine keeps serving and reads the next turn on the CPU as a CPU-only engine does); not a
#   serve, the same failure still exits 1.
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
    env "$@" OMP_NUM_THREADS=2 SNAP=$dir ./qwen38 "${ENG_CAP:-1}" 8 "$ref" > "$tmp/$nm.log" 2>&1
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
# within <relerr>: a plain non-negative number, at most 1e-4 (empty, nan or inf fail: macOS awk
# compares a NaN equal to anything, so the format is checked first)
within() { [ -n "$1" ] && awk -v e="$1" 'BEGIN { exit !(e ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && e + 0 <= 1e-4) }'; }
# logits_ok <cpu name> <metal name>: DUMP logits within 1e-4 x max|logit|, or a FAIL line
logits_ok() {
    e=$(relerr "$tmp/$1.bin" "$tmp/$2.bin")
    if within "$e"; then return 0; fi
    echo "FAIL [$2] final logits off by $e x max|logit| against $1"; fail=1; return 1
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
            if ! within "$e"; then echo "FAIL [$nm] final logits off by $e x max|logit|"; fail=1; continue; fi
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

echo "== m3 gpu experts"
# M3 (Plan 2 Task 8): the routed experts on the GPU, from a GPU expert cache, on the tiny GGUF
# (its Q8_0 experts), against the same CPU run as m2 tiny (reference precision, Ruling P2-4).
# The chain's report says `experts on GPU: N, loads: L, hits: H` and the cache's groups.
GX='[qwen38-metal] GPU experts are off: '
gx_n() { sed -n 's/.*experts on GPU: \([0-9]*\), loads: \([0-9]*\), hits: \([0-9]*\).*/\1 \2 \3/p' "$tmp/$1.log" | tail -1; }
gx_most() { sed -n 's/.*most experts in one layer \([0-9]*\).*/\1/p' "$tmp/$1.log" | tail -1; }
gx_on() {   # <name>: no off line for the GPU experts, N > 0 experts computed on the GPU, L > 0 loads
    if grep -qF "$GX" "$tmp/$1.log"; then echo "FAIL [$1] GPU experts off:"; grep -F "$GX" "$tmp/$1.log"; fail=1; return 1; fi
    c=$(gx_n "$1")
    if [ -z "$c" ] || [ "${c%% *}" -eq 0 ] || [ "$(echo "$c" | cut -d' ' -f2)" -eq 0 ]; then
        echo "FAIL [$1] experts on GPU, loads, hits: '${c:-no report}'"; fail=1; return 1; fi
    return 0
}
fx=qwen38_tiny_256_gguf
want=$(toks cpu_$fx)
for exact in 1 0; do
    nm=m3_exact$exact
    eng $nm ./$fx ./$fx/ref.json QWEN38_METAL_EXACT=$exact $REFP DUMP=$tmp/$nm.bin || continue
    chain_on $nm || continue
    gx_on $nm || continue
    got=$(toks $nm)
    if [ -z "$want" ] || [ "$got" != "$want" ]; then echo "FAIL [$nm] tokens '$got' vs the CPU's '$want'"; fail=1; continue; fi
    most=$(gx_most $nm)
    # the prompt's rows (5 rows, top-k 2 of 8 experts) touch more experts than one group of top-k
    if [ -z "$most" ] || [ "$most" -le 2 ]; then echo "FAIL [$nm] no layer with more experts than one group ('${most:-none}')"; fail=1; continue; fi
    if [ $exact = 1 ]; then logits_ok cpu_$fx $nm || continue; fi
    echo "ok   [$nm] the CPU's tokens, logits within $(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin") x max$([ $exact = 1 ] || echo ' (not gated)'); experts on GPU, loads, hits: $(gx_n $nm); most in one layer $most"
done
# tiny cache: two payloads (one group of top-k 2), smaller than one step's working set: every
# group evicts the last; the CPU's tokens, and it finishes (120 s)
pay=$(sed -n 's/^\[qwen38-metal\] GPU experts: .* slots of \([0-9]*\) bytes.*/\1/p' "$tmp/m3_exact1.log" | tail -1)
if [ -z "$pay" ]; then echo "FAIL [m3 tiny cache] no payload in the GPU experts' line"; fail=1
else
    nm=m3_tinycache
    if eng $nm ./$fx ./$fx/ref.json perl -e 'alarm shift; exec @ARGV' 120 env QWEN38_EXPERT_CACHE_BYTES=$((2 * pay)) QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/$nm.bin &&
       chain_on $nm && gx_on $nm; then
        if ! grep -qE '^\[qwen38-metal\] GPU experts: 2 slots' "$tmp/$nm.log"; then echo "FAIL [$nm] not 2 slots:"; grep -F 'GPU experts' "$tmp/$nm.log"; fail=1
        elif [ "$(toks $nm)" != "$want" ]; then echo "FAIL [$nm] tokens '$(toks $nm)' vs the CPU's '$want'"; fail=1
        elif logits_ok cpu_$fx $nm; then echo "ok   [$nm 2 slots] the CPU's tokens, logits within $(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin") x max; experts on GPU, loads, hits: $(gx_n $nm)"; fi
    fi
fi
# slot alloc failure (debug): the GPU experts off with one line, the chain on, the CPU's tokens
nm=m3_allocfail
if eng $nm ./$fx ./$fx/ref.json Q38M_FAIL_SLOT_ALLOC=1 QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/$nm.bin && chain_on $nm; then
    n=$(grep -cF "${GX}a slot of the GPU expert cache could not be allocated (Q38M_FAIL_SLOT_ALLOC); the routed experts run on the CPU" "$tmp/$nm.log")
    c=$(gx_n $nm)
    if [ "$n" -ne 1 ] || [ "$(grep -cF "$GX" "$tmp/$nm.log")" -ne 1 ]; then echo "FAIL [$nm] expected one off line:"; grep -F 'GPU experts' "$tmp/$nm.log"; fail=1
    elif [ -n "$c" ] && [ "${c%% *}" -ne 0 ]; then echo "FAIL [$nm] experts on the GPU: $c"; fail=1
    elif [ "$(toks $nm)" != "$want" ]; then echo "FAIL [$nm] tokens '$(toks $nm)' vs the CPU's '$want'"; fail=1
    elif logits_ok cpu_$fx $nm; then echo "ok   [$nm] one off line, the CPU's tokens, logits within $(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin") x max"; fi
fi
# QWEN38_METAL_EXPERTS=0: the M2 path, with its line
nm=m3_off
if eng $nm ./$fx ./$fx/ref.json QWEN38_METAL_EXPERTS=0 && chain_on $nm; then
    if [ "$(grep -cF "${GX}QWEN38_METAL_EXPERTS=0; the routed experts run on the CPU" "$tmp/$nm.log")" -ne 1 ]; then echo "FAIL [$nm] expected one off line"; grep -F 'GPU experts' "$tmp/$nm.log"; fail=1
    else echo "ok   [$nm] QWEN38_METAL_EXPERTS=0 keeps the routed experts on the CPU, one line"; fi
fi
# off mid-run (Q38M_REFUSE_AT=3, debug: the third step's device memory refused): the chain's
# off line, the GPU experts release their cache and the CPU's expert cache gets its slots back
# (4 a layer here), whose hits the CPU's steps then use; the CPU's tokens and logits
nm=m3_refused
if ENG_CAP=4 eng $nm ./$fx ./$fx/ref.json Q38M_REFUSE_AT=3 QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/$nm.bin; then
    hits=$(sed -n 's/^Expert cache hit rate: .*(hit=\([0-9]*\) miss=.*/\1/p' "$tmp/$nm.log" | tail -1)
    if [ "$(grep -cF "${PREFIX}device memory for a step of 1 rows at position 6 was refused$SUFFIX" "$tmp/$nm.log")" -ne 1 ]; then
        echo "FAIL [$nm] no off line for the refused step"; grep -F '[qwen38-metal]' "$tmp/$nm.log"; fail=1
    elif ! grep -qF "[qwen38-metal] GPU experts released (the chain is off): the CPU's expert cache is back to 4 slots a layer" "$tmp/$nm.log"; then
        echo "FAIL [$nm] the GPU experts not released, or the CPU's cache not back to 4"; grep -F 'GPU experts' "$tmp/$nm.log"; fail=1
    elif [ -z "$hits" ] || [ "$hits" -eq 0 ]; then echo "FAIL [$nm] the CPU's expert cache has no hit ('${hits:-no line}')"; fail=1
    elif [ "$(toks $nm)" != "$want" ]; then echo "FAIL [$nm] tokens '$(toks $nm)' vs the CPU's '$want'"; fail=1
    elif logits_ok cpu_$fx $nm; then echo "ok   [$nm] off at step 3: GPU cache released, the CPU's cache back to 4 slots ($hits hits), the CPU's tokens and logits ($(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin"))"; fi
fi
# usage accounting: the GPU-routed steps count in the usage history and the route trace as the
# CPU's do (rt_route): fresh model dirs, USAGE_SAVE=1, ROUTE_TRACE; the same history file and the
# same routes (row, layer, experts and gates; the call numbers aside)
for u in cpu m3; do
    mkdir -p "$tmp/use_$u"
    for f in qwen38_tiny_256_gguf/*; do ln -s "$PWD/$f" "$tmp/use_$u/"; done
    rm -f "$tmp/use_$u/.coli_usage"
done
eng use_cpu "$tmp/use_cpu" ./$fx/ref.json COLI_METAL=0 $REFP USAGE_SAVE=1 ROUTE_TRACE=$tmp/use_cpu.tr
eng use_m3 "$tmp/use_m3" ./$fx/ref.json QWEN38_METAL_EXACT=1 $REFP USAGE_SAVE=1 ROUTE_TRACE=$tmp/use_m3.tr
if chain_on use_m3 && gx_on use_m3; then
    a=$(sed -n 's/^\[STATS\] \([0-9]* selections across [0-9]* distinct experts\).*/\1/p' "$tmp/use_cpu.log")
    b=$(sed -n 's/^\[STATS\] \([0-9]* selections across [0-9]* distinct experts\).*/\1/p' "$tmp/use_m3.log")
    for u in cpu m3; do awk '{$1 = ""; print}' "$tmp/use_$u.tr" | sort > "$tmp/use_$u.routes"; done
    if [ -z "$a" ] || [ "$a" != "$b" ]; then echo "FAIL [m3 usage] the chain's '$b' vs the CPU's '$a'"; fail=1
    elif ! cmp -s "$tmp/use_cpu/.coli_usage" "$tmp/use_m3/.coli_usage"; then echo "FAIL [m3 usage] the usage files differ"; fail=1
    elif [ ! -s "$tmp/use_m3.routes" ] || ! cmp -s "$tmp/use_cpu.routes" "$tmp/use_m3.routes"; then
        echo "FAIL [m3 usage] route traces differ ($(wc -l < "$tmp/use_cpu.tr") vs $(wc -l < "$tmp/use_m3.tr") lines)"; fail=1
    else echo "ok   [m3 usage] $b, the same usage file and the same $(wc -l < "$tmp/use_m3.tr" | tr -d ' ') route-trace lines as the CPU"; fi
fi
# the release's expert shape (Task 8 review): a variant of the tiny GGUF with 16 experts, top-k 10
# (one group of 10 = kernel sub-groups of 8 and 2; the prompt's rows touch more than one group) and
# IQ4_XS gate/up, IQ4_NL down experts, its ref.json the CPU's on it; exact mode the CPU's tokens
# and logits (1e-4), fast mode its tokens, and the layer check: every layer within 1e-4 and the
# GPU's routes the CPU's rule
iq=$tmp/qwen38_tiny_256_gguf_iq4
mkdir -p "$iq"
if "${PYTHON:-python3}" tools/make_qwen38_tiny_gguf.py --experts 16 --topk 10 --iq4 ./qwen38_tiny_256 "$iq/model.gguf" >/dev/null 2>&1 &&
   "${PYTHON:-python3}" tools/gguf_import.py "$iq/model.gguf" "$iq" --mode link >/dev/null 2>&1 &&
   cp qwen38_tiny_256/ref.json "$iq/ref.json" &&
   "${PYTHON:-python3}" tools/make_qwen38_tiny_gguf.py --rebase-ref ./qwen38 "$iq" >/dev/null 2>&1; then
    eng iq4_cpu "$iq" "$iq/ref.json" COLI_METAL=0 $REFP DUMP=$tmp/iq4_cpu.bin
    if ! grep -qF 'routed experts: GGUF IQ4_XS/IQ4_XS/IQ4_NL' "$tmp/iq4_cpu.log"; then echo "FAIL [iq4 fixture] not IQ4 experts"; fail=1; fi
    for exact in 1 0; do
        nm=iq4_m3_exact$exact
        eng $nm "$iq" "$iq/ref.json" QWEN38_METAL_EXACT=$exact $REFP DUMP=$tmp/$nm.bin && chain_on $nm && gx_on $nm || continue
        most=$(gx_most $nm)
        if [ -z "$(toks iq4_cpu)" ] || [ "$(toks $nm)" != "$(toks iq4_cpu)" ]; then echo "FAIL [$nm] '$(toks $nm)' vs the CPU's '$(toks iq4_cpu)'"; fail=1; continue; fi
        if ! grep -qE '^\[qwen38-metal\] GPU experts: .*groups of 10' "$tmp/$nm.log" || [ -z "$most" ] || [ "$most" -le 10 ]; then
            echo "FAIL [$nm] not groups of 10 or no layer past one group ('${most:-none}')"; fail=1; continue; fi
        if [ $exact = 1 ]; then logits_ok iq4_cpu $nm || continue; fi
        echo "ok   [$nm 16 experts, top-k 10, IQ4] the CPU's tokens, logits within $(relerr "$tmp/iq4_cpu.bin" "$tmp/$nm.bin") x max; experts on GPU, loads, hits: $(gx_n $nm); most in one layer $most"
    done
    nm=iq4_layers
    if eng $nm "$iq" "$iq/ref.json" Q38M_LAYER_CHECK=1 QWEN38_METAL_EXACT=1 $REFP; then
        n=$(grep -c '^\[q38m-layer\] L.*worst=' "$tmp/$nm.log")
        g=$(grep -c '^\[q38m-layer\] L[0-9]* GPU routes: 0 rows unlike' "$tmp/$nm.log")
        if [ "$n" -lt 4 ] || [ "$g" -lt 4 ] || grep '^\[q38m-layer\]' "$tmp/$nm.log" | grep -qF FAIL; then
            echo "FAIL [$nm] $n layer lines, $g clean GPU-route lines"; grep '^\[q38m-layer\]' "$tmp/$nm.log"; fail=1
        else echo "ok   [$nm] 4 layers, worst $(grep '^\[q38m-layer\] L.*worst=' "$tmp/$nm.log" | sed 's/.*worst=\([^ ]*\).*/\1/' | sort -g | tail -1), the GPU's routes the CPU's rule"; fi
    fi
else
    echo "FAIL [iq4 fixture] generation"; fail=1
fi
# a long prompt on the GGUF (1,300 tokens: chunks of 256 rows, every expert of a layer in a chunk,
# groups of top-k through the cache), exact and fast: the CPU's tokens; exact, its logits
mkdir -p "$tmp/glong"
for f in qwen38_tiny_256_gguf/*; do ln -s "$PWD/$f" "$tmp/glong/"; done
rm -f "$tmp/glong/config.json" "$tmp/glong/ref.json" "$tmp/glong/.coli_usage"
"${PYTHON:-python3}" -c '
import json, sys
c = json.load(open("qwen38_tiny_256_gguf/config.json"))
(c.get("text_config") or c)["max_position_embeddings"] = 2048
json.dump(c, open(sys.argv[1] + "/config.json", "w"))
x, ids = 4242, []
for _ in range(1300):
    x = (x * 1103515245 + 12345) % 2**31
    ids.append(3 + x % 61)
json.dump({"schema_version": 1, "prompt_ids": ids, "full_ids": ids + [3] * 8}, open(sys.argv[1] + "/ref.json", "w"))' "$tmp/glong"
if eng glong_cpu "$tmp/glong" "$tmp/glong/ref.json" COLI_METAL=0 $REFP DUMP=$tmp/glong_cpu.bin; then
    for exact in 1 0; do
        nm=glong_m3_exact$exact
        eng $nm "$tmp/glong" "$tmp/glong/ref.json" QWEN38_METAL_EXACT=$exact $REFP DUMP=$tmp/$nm.bin && chain_on $nm && gx_on $nm || continue
        if [ -z "$(toks glong_cpu)" ] || [ "$(toks $nm)" != "$(toks glong_cpu)" ]; then echo "FAIL [$nm] '$(toks $nm)' vs the CPU's '$(toks glong_cpu)'"; fail=1; continue; fi
        if [ $exact = 1 ]; then logits_ok glong_cpu $nm || continue; fi
        echo "ok   [$nm 1300 tokens] the CPU's tokens, logits within $(relerr "$tmp/glong_cpu.bin" "$tmp/$nm.bin") x max; experts on GPU, loads, hits: $(gx_n $nm); most in one layer $(gx_most $nm)"
    done
fi

echo "== ahead tokens"
# Plan 2 Task 9: QWEN38_METAL_AHEAD=1 runs the next layer's router on each decode layer's MLP
# rows and announces the uncached guesses' GGUF bytes to the OS (F_RDADVISE): advice only, so
# the tokens and the final logits are bit-identical to QWEN38_METAL_AHEAD=0, fast and exact, on
# the tiny GGUF (top-k 2: two guesses) and the IQ4 variant (top-k 10: six guesses). With it on
# the report says `guesses: G, right: R` with G > 0 and R <= G; off, no guess.
ah_n() { sed -n 's/.*guesses: \([0-9]*\), right: \([0-9]*\).*/\1 \2/p' "$tmp/$1.log" | tail -1; }
for fx in gguf iq4; do
    case $fx in gguf) dir=./qwen38_tiny_256_gguf;; *) dir=$iq;; esac
    [ -f "$dir/ref.json" ] || { echo "FAIL [ahead $fx] no fixture"; fail=1; continue; }
    for exact in 1 0; do
        an=ahead_${fx}_exact$exact   # eng sets nm
        eng ${an}_off "$dir" "$dir/ref.json" QWEN38_METAL_AHEAD=0 QWEN38_METAL_EXACT=$exact $REFP DUMP=$tmp/${an}_off.bin &&
            eng ${an}_on "$dir" "$dir/ref.json" QWEN38_METAL_AHEAD=1 QWEN38_METAL_EXACT=$exact $REFP DUMP=$tmp/${an}_on.bin &&
            chain_on ${an}_off && chain_on ${an}_on && gx_on ${an}_on || continue
        g=$(ah_n ${an}_on); g0=$(ah_n ${an}_off)
        if [ -z "$(toks ${an}_off)" ] || [ "$(toks ${an}_on)" != "$(toks ${an}_off)" ]; then
            echo "FAIL [$an] tokens with read-ahead '$(toks ${an}_on)' vs without '$(toks ${an}_off)'"; fail=1
        elif ! cmp -s "$tmp/${an}_off.bin" "$tmp/${an}_on.bin"; then
            echo "FAIL [$an] final logits differ with read-ahead ($(relerr "$tmp/${an}_off.bin" "$tmp/${an}_on.bin") x max)"; fail=1
        elif [ -z "$g" ] || [ "${g%% *}" -eq 0 ] || [ "${g#* }" -gt "${g%% *}" ]; then
            echo "FAIL [$an] guesses, right: '${g:-no report}'"; fail=1
        elif [ -n "$g0" ] && [ "${g0%% *}" -ne 0 ]; then
            echo "FAIL [$an] guesses with QWEN38_METAL_AHEAD=0: $g0"; fail=1
        else echo "ok   [$an] the same tokens and bit-identical logits with read-ahead; guesses: ${g%% *}, right: ${g#* }"; fi
    done
done

# read-ahead that cannot run degrades (final review): Q38M_AHEAD_FAIL=5 (debug) makes the fifth
# ahead router decline (the second decode step: 3 a step on 4 layers), =start refuses its buffers
# at the start: one `read-ahead off` line, the chain and the GPU experts on, and the tokens and
# final logits bit-identical to QWEN38_METAL_AHEAD=0 (so the CPU's, within 1e-4)
AOFF='[qwen38-metal] read-ahead off: '
for af in 5 start; do
    nm=ahead_fail_$af; aref=ahead_gguf_exact1_off; fx=qwen38_tiny_256_gguf
    [ -s "$tmp/$aref.bin" ] || { echo "FAIL [$nm] no read-ahead-off run to compare"; fail=1; continue; }
    eng $nm ./$fx ./$fx/ref.json Q38M_AHEAD_FAIL=$af QWEN38_METAL_AHEAD=1 QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/$nm.bin &&
        chain_on $nm && gx_on $nm || continue
    case $af in start) line="${AOFF}no device memory for its router (Q38M_AHEAD_FAIL=start); the GPU experts run without it";;
                *) line="${AOFF}its router did not run on the GPU; the GPU experts run without it";; esac
    g=$(ah_n $nm)
    if [ "$(grep -cF "$line" "$tmp/$nm.log")" -ne 1 ] || [ "$(grep -cF "$AOFF" "$tmp/$nm.log")" -ne 1 ]; then
        echo "FAIL [$nm] expected one line: $line"; grep -F '[qwen38-metal]' "$tmp/$nm.log"; fail=1
    elif [ -z "$(toks cpu_$fx)" ] || [ "$(toks $nm)" != "$(toks cpu_$fx)" ] || [ "$(toks $nm)" != "$(toks $aref)" ]; then
        echo "FAIL [$nm] tokens '$(toks $nm)' vs the CPU's '$(toks cpu_$fx)'"; fail=1
    elif ! cmp -s "$tmp/$aref.bin" "$tmp/$nm.bin"; then
        echo "FAIL [$nm] final logits differ from QWEN38_METAL_AHEAD=0 ($(relerr "$tmp/$aref.bin" "$tmp/$nm.bin") x max)"; fail=1
    elif [ $af = start ] && [ -n "$g" ] && [ "${g%% *}" -ne 0 ]; then echo "FAIL [$nm] guesses without read-ahead: $g"; fail=1
    elif [ $af != start ] && { [ -z "$g" ] || [ "${g%% *}" -eq 0 ]; }; then echo "FAIL [$nm] no guess before the decline ('${g:-no report}')"; fail=1
    elif logits_ok cpu_$fx $nm; then
        echo "ok   [$nm] one read-ahead-off line, the chain and GPU experts on, logits bit-identical to read-ahead off and within $(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin") of the CPU (guesses, right: ${g:-0})"
    fi
done

echo "== metal_state_handoff"
# qwen38_decay: qwen38_tiny with A_log = -1.88 and dt_bias = 0 in its DeltaNet layers, so the
# recurrent state decays by about 0.9 a token (the fixture's own: about 1e-6, the state hardly
# matters): a state the hand-off loses or applies twice moves the logits
"${PYTHON:-python3}" -c '
import struct, sys
sys.path.insert(0, "tests")
import safetensors_edit as se
d = se.copy_fixture("qwen38_tiny", sys.argv[1])
(d / ".coli_usage").unlink(missing_ok=True)
bf = lambda x: struct.pack("<H", struct.unpack("<I", struct.pack("<f", x))[0] >> 16)
for i in (0, 2):
    for name, v in (("A_log", -1.88), ("dt_bias", 0.0)):
        t = "model.layers.%d.linear_attn.%s" % (i, name)
        n = se.shape_of(d, t)[0]
        se.put(d, t, "BF16", [n], bf(v) * n)' "$tmp/qwen38_decay"
eng cpu_decay "$tmp/qwen38_decay" "$tmp/qwen38_decay/ref.json" COLI_METAL=0 $REFP DUMP=$tmp/cpu_decay.bin
for fx in qwen38_tiny qwen38_tiny_256_gguf decay; do
    nm=handoff_$fx
    case $fx in decay) dir=$tmp/qwen38_decay;; *) dir=./$fx;; esac
    eng $nm $dir $dir/ref.json Q38M_DECLINE_EVERY=2 QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/$nm.bin || continue
    chain_on $nm || continue
    f=$(forwards $nm)
    if [ "${f#* }" -eq 0 ]; then echo "FAIL [$nm] nothing declined ($f)"; fail=1; continue; fi
    if [ -z "$(toks cpu_$fx)" ] || [ "$(toks $nm)" != "$(toks cpu_$fx)" ]; then echo "FAIL [$nm] tokens '$(toks $nm)' vs '$(toks cpu_$fx)'"; fail=1; continue; fi
    logits_ok cpu_$fx $nm || continue
    echo "ok   [$nm] the CPU's tokens, logits within $(relerr "$tmp/cpu_$fx.bin" "$tmp/$nm.bin") x max (forwards, declined: $f)"
done
# the decay fixture's chain without declines, against the same CPU run
if eng m2_decay $tmp/qwen38_decay $tmp/qwen38_decay/ref.json QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/m2_decay.bin && chain_on m2_decay; then
    if [ -z "$(toks cpu_decay)" ] || [ "$(toks m2_decay)" != "$(toks cpu_decay)" ]; then
        echo "FAIL [m2_decay] tokens '$(toks m2_decay)' vs the CPU's '$(toks cpu_decay)'"; fail=1
    elif logits_ok cpu_decay m2_decay; then
        echo "ok   [m2_decay] the CPU's tokens, logits within $(relerr "$tmp/cpu_decay.bin" "$tmp/m2_decay.bin") x max"
    fi
fi

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
if eng long_cpu "$tmp/long" "$tmp/long/ref.json" COLI_METAL=0 $REFP DUMP=$tmp/long_cpu.bin &&
   eng long_mtl "$tmp/long" "$tmp/long/ref.json" QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/long_mtl.bin && chain_on long_mtl; then
    if [ -z "$(toks long_cpu)" ] || [ "$(toks long_mtl)" != "$(toks long_cpu)" ]; then
        echo "FAIL [kv growth 1300 tokens] '$(toks long_mtl)' vs the CPU's '$(toks long_cpu)'"; fail=1
    elif logits_ok long_cpu long_mtl; then
        echo "ok   [kv growth 1300 tokens] the CPU's tokens, logits within $(relerr "$tmp/long_cpu.bin" "$tmp/long_mtl.bin") x max: $(toks long_mtl)"
    fi
fi
# growth with valid rows (final review): qwen38_decay with 2048 positions and a byte tokenizer, served
# with a 1,020-token prompt in segments of 300 rows and 8 tokens decoded across position 1024: the
# device caches grow twice with valid rows (dropped, uploaded again from the host); every served
# step's logits the CPU's (tests/metal_qwen38_serve.py --segments)
mkdir -p "$tmp/dseg"
for f in "$tmp"/qwen38_decay/*; do ln -s "$f" "$tmp/dseg/"; done
rm -f "$tmp/dseg/config.json" "$tmp/dseg/ref.json" "$tmp/dseg/.coli_usage" "$tmp/dseg/tokenizer.json"
"${PYTHON:-python3}" -c '
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
(c.get("text_config") or c)["max_position_embeddings"] = 2048
json.dump(c, open(sys.argv[2] + "/config.json", "w"))' "$tmp/qwen38_decay" "$tmp/dseg"
"${PYTHON:-python3}" tools/make_edge_tiny_tokenizer.py --vocab-size 64 "$tmp/dseg" >/dev/null
"${PYTHON:-python3}" tests/metal_qwen38_serve.py --segments "$tmp/dseg" || fail=1

echo "== long prompt, fast mode"
# Plan 2 Task 7a: the fast kernels (QWEN38_METAL_EXPERTS=0, QWEN38_METAL_EXACT unset: one DeltaNet
# state plane) over a 3,000-token prompt on qwen38_decay (its recurrent state counts): 12 chunks
# of 256 rows, the QSA past its budget, the caches grown to 4096. The CPU's tokens and final
# logits (1e-4 x max), and the layer check with Q38M_LAYER_CHECK=2: every layer of the chain
# within 1e-4 of the CPU, with the routes the reference run took, and the CPU's own one-ulp
# lines printed (the floor a real model's long prompt is read against: a top-k near-tie the
# CPU itself flips from one ulp, see docs/ENVIRONMENT.md Q38M_LAYER_CHECK)
mkdir -p "$tmp/lfast"
for f in "$tmp"/qwen38_decay/*; do ln -s "$f" "$tmp/lfast/"; done
rm -f "$tmp/lfast/config.json" "$tmp/lfast/ref.json" "$tmp/lfast/.coli_usage"
"${PYTHON:-python3}" -c '
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
(c.get("text_config") or c)["max_position_embeddings"] = 4096
json.dump(c, open(sys.argv[2] + "/config.json", "w"))
x, ids = 777, []
for _ in range(3000):
    x = (x * 1103515245 + 12345) % 2**31
    ids.append(3 + x % 61)
json.dump({"schema_version": 1, "prompt_ids": ids, "full_ids": ids + [3] * 8}, open(sys.argv[2] + "/ref.json", "w"))' "$tmp/qwen38_decay" "$tmp/lfast"
# the same run with no debug switch: the CPU's tokens and logits
if eng lfast_cpu "$tmp/lfast" "$tmp/lfast/ref.json" COLI_METAL=0 $REFP DUMP=$tmp/lfast_cpu.bin &&
   eng lfast_plain "$tmp/lfast" "$tmp/lfast/ref.json" QWEN38_METAL_EXPERTS=0 $REFP DUMP=$tmp/lfast_plain.bin && chain_on lfast_plain; then
    if [ -z "$(toks lfast_cpu)" ] || [ "$(toks lfast_plain)" != "$(toks lfast_cpu)" ]; then
        echo "FAIL [long fast 3000 tokens, no debug] '$(toks lfast_plain)' vs the CPU's '$(toks lfast_cpu)'"; fail=1
    elif grep -qF '[q38m-' "$tmp/lfast_plain.log"; then
        echo "FAIL [long fast 3000 tokens, no debug] debug lines with the switches unset"; fail=1
    elif logits_ok lfast_cpu lfast_plain; then
        echo "ok   [long fast 3000 tokens, no debug] the CPU's tokens, logits within $(relerr "$tmp/lfast_cpu.bin" "$tmp/lfast_plain.bin") x max"
    fi
fi
if [ -s "$tmp/lfast_cpu.bin" ] &&
   eng lfast_mtl "$tmp/lfast" "$tmp/lfast/ref.json" QWEN38_METAL_EXPERTS=0 Q38M_LAYER_CHECK=2 $REFP DUMP=$tmp/lfast_mtl.bin &&
   chain_on lfast_mtl; then
    n=$(grep -c '^\[q38m-layer\] L.*routes off the reference run.s in 0 rows' "$tmp/lfast_mtl.log")
    u=$(grep -c '^\[q38m-cpu-ulp\] L' "$tmp/lfast_mtl.log")
    if [ -z "$(toks lfast_cpu)" ] || [ "$(toks lfast_mtl)" != "$(toks lfast_cpu)" ]; then
        echo "FAIL [long fast 3000 tokens] '$(toks lfast_mtl)' vs the CPU's '$(toks lfast_cpu)'"; fail=1
    elif [ "$n" -lt 4 ] || [ "$u" -lt 4 ] || grep '^\[q38m-layer\] L' "$tmp/lfast_mtl.log" | grep -qF FAIL; then
        echo "FAIL [long fast 3000 tokens] layer check: $n clean layer lines, $u one-ulp lines"
        grep -E '^\[q38m-(layer|cpu-ulp)\]' "$tmp/lfast_mtl.log"; fail=1
    elif logits_ok lfast_cpu lfast_mtl; then
        worst=$(grep '^\[q38m-layer\] L' "$tmp/lfast_mtl.log" | sed 's/.*worst=\([^ ]*\).*/\1/' | sort -g | tail -1)
        echo "ok   [long fast 3000 tokens] the CPU's tokens, logits within $(relerr "$tmp/lfast_cpu.bin" "$tmp/lfast_mtl.bin") x max, layers within $worst, routes the reference's"
    fi
fi

echo "== lookup policy"
# Ruling P2-5: beside the chain, COLI_LOOKUP unset turns prompt-lookup drafts off with one note
# (their verifies would run on the CPU); COLI_LOOKUP=1 keeps them, the verifies on the CPU
NOTE='[qwen38-metal] prompt-lookup drafts: off'
eng lk_default ./qwen38_tiny ./qwen38_tiny/ref.json
n=$(grep -cF "$NOTE" "$tmp/lk_default.log")
if [ "$n" -eq 1 ] && chain_on lk_default; then echo "ok   [lookup unset beside the chain] one note line"
else echo "FAIL [lookup unset beside the chain] note lines: $n"; fail=1; fi
eng lk_cpu ./qwen38_tiny ./qwen38_tiny/ref.json COLI_METAL=0 COLI_LOOKUP=1 COLI_LOOKUP_FORCE=mixed $REFP DUMP=$tmp/lk_cpu.bin
eng lk_on ./qwen38_tiny ./qwen38_tiny/ref.json COLI_LOOKUP=1 COLI_LOOKUP_FORCE=mixed QWEN38_METAL_EXACT=1 $REFP DUMP=$tmp/lk_on.bin
if grep -qF "$NOTE" "$tmp/lk_on.log" || grep -qF "$NOTE" "$tmp/lk_cpu.log"; then echo "FAIL [COLI_LOOKUP=1] the note with lookup asked for"; fail=1
elif chain_on lk_on; then
    f=$(forwards lk_on)
    if [ "${f#* }" -eq 0 ]; then echo "FAIL [COLI_LOOKUP=1] no verify declined ($f)"; fail=1
    elif [ -z "$(toks lk_cpu)" ] || [ "$(toks lk_on)" != "$(toks lk_cpu)" ]; then echo "FAIL [COLI_LOOKUP=1] '$(toks lk_on)' vs the CPU's '$(toks lk_cpu)'"; fail=1
    elif logits_ok lk_cpu lk_on; then echo "ok   [COLI_LOOKUP=1 forced mixed drafts] the CPU's tokens, verifies on the CPU (forwards, declined: $f)"; fi
fi

echo "== layer test"
for fx in qwen38_tiny qwen38_tiny_256_gguf; do
    nm=layers_$fx
    eng $nm ./$fx ./$fx/ref.json Q38M_LAYER_CHECK=1 QWEN38_METAL_EXACT=1 $REFP || continue
    n=$(grep -c '^\[q38m-layer\] L.*worst=' "$tmp/$nm.log")
    if [ "$n" -lt 4 ] || grep '^\[q38m-layer\]' "$tmp/$nm.log" | grep -qF FAIL; then
        echo "FAIL [$nm] $n layer lines"; grep '^\[q38m-layer\]' "$tmp/$nm.log"; fail=1; continue
    fi
    worst=$(grep '^\[q38m-layer\] L.*worst=' "$tmp/$nm.log" | sed 's/.*worst=\([^ ]*\).*/\1/' | sort -g | tail -1)
    case $fx in *_gguf)   # the GPU experts: the GPU's routes are the CPU's rule on the same logits
        g=$(grep -c '^\[q38m-layer\] L[0-9]* GPU routes: 0 rows unlike' "$tmp/$nm.log")
        if [ "$g" -lt 4 ]; then echo "FAIL [$nm] GPU routes lines: $g clean"; grep '^\[q38m-layer\] L[0-9]* GPU routes' "$tmp/$nm.log"; fail=1; continue; fi;;
    esac
    echo "ok   [$nm] $n layers, worst $worst"
done
# Q38M_FORCE_ROUTES on the kv-growth prompt (1,300 tokens on qwen38_tiny: its QSA budget is 4
# positions, so nearly every row is past it): =1 holds rows to the reference run's routes (some
# rows' router logits moved), =2 also writes the reference run's selection into every row past
# the budget; the chain's own picks are the reference's here, so no selection may differ (the
# two formats agree) and nothing FAILs
for force in 1 2; do
    nm=forced$force
    eng $nm "$tmp/long" "$tmp/long/ref.json" Q38M_LAYER_CHECK=1 Q38M_FORCE_ROUTES=$force QWEN38_METAL_EXACT=1 $REFP || continue
    n=$(grep -c '^\[q38m-layer\] L.*worst=.*; forced$' "$tmp/$nm.log")
    held=$(sed -n 's/^\[q38m-layer\] L[0-9]* forced: \([0-9]*\) rows held.*/\1/p' "$tmp/$nm.log" | awk '{s += $1} END {print s + 0}')
    q=$(sed -n "s/^\[q38m-layer\] QSA forced: \([0-9]*\) rows past the budget held; \([0-9]*\) rows' selections differed.*/\1 \2/p" "$tmp/$nm.log")
    bad=""
    [ "$n" -lt 4 ] && bad="$n forced layer lines"
    [ "$held" -gt 0 ] || bad="$bad; no row held to the reference run's routes"
    grep '^\[q38m-layer\]' "$tmp/$nm.log" | grep -qF FAIL && bad="$bad; a layer FAILs"
    if [ $force = 2 ]; then
        if [ -z "$q" ] || [ "${q%% *}" -eq 0 ]; then bad="$bad; no row past the budget held ('${q:-no QSA line}')"
        elif [ "${q#* }" -ne 0 ]; then bad="$bad; ${q#* } rows' selections differ from the reference run's"; fi
    elif [ -n "$q" ]; then bad="$bad; a QSA line with Q38M_FORCE_ROUTES=1"; fi
    if [ -n "$bad" ]; then echo "FAIL [$nm]${bad#;}"; grep '^\[q38m-layer\]' "$tmp/$nm.log" | grep -v ' by chunk' | head -20; fail=1; continue; fi
    worst=$(grep '^\[q38m-layer\] L.*worst=' "$tmp/$nm.log" | sed 's/.*worst=\([^ ]*\).*/\1/' | sort -g | tail -1)
    echo "ok   [$nm 1300 tokens] $n layers, worst $worst, $held rows held to the reference's routes${q:+, ${q%% *} QSA rows held, ${q#* } differing}"
done

echo "== serve"
"${PYTHON:-python3}" tests/metal_qwen38_serve.py || fail=1
echo "== serve metal frame"
"${PYTHON:-python3}" tests/metal_qwen38_serve.py --metal-frame ./qwen38_tiny_256_gguf || fail=1
echo "== serve gpu error"
"${PYTHON:-python3}" tests/metal_qwen38_serve.py --gpu-error ./qwen38_tiny_256_gguf || fail=1
# not a serve: the same failure stops the run loudly, as before (Q38M_FAIL_AT_STEP, debug)
out=$(env Q38M_FAIL_AT_STEP=3 OMP_NUM_THREADS=2 SNAP=./qwen38_tiny_256_gguf ./qwen38 1 8 ./qwen38_tiny_256_gguf/ref.json 2>&1); rc=$?
if [ $rc -eq 1 ] && printf '%s\n' "$out" | grep -qF "the recurrent state is on the GPU only, stopping"; then
    echo "ok   [gpu error not a serve] exit 1 with the stopping line"
else
    echo "FAIL [gpu error not a serve] exit $rc"; printf '%s\n' "$out" | grep -F '[qwen38-metal]'; fail=1
fi
# the engine's own prefix suites (the live state, the snapshot, the pins, the parked
# conversation, the cancels; the store's checkpoints and rows) with the chain on: they reach
# every state hook (q38m_sync_host, q38m_host_wrote, q38m_rollback)
mkdir -p "$tmp/tok"
for f in qwen38_tiny/*; do ln -s "$PWD/$f" "$tmp/tok/"; done
"${PYTHON:-python3}" tools/make_edge_tiny_tokenizer.py --vocab-size 64 "$tmp/tok" >/dev/null
# serve gpu error, the store's files on disk: a lost GPU state removes them (the one Metal case of the disk suite)
if QWEN38_TINY="$tmp/tok" "${PYTHON:-python3}" -W ignore -m unittest \
        tests.test_qwen38_prefix_disk.Qwen38PrefixDiskTest.test_a_lost_gpu_state_removes_the_files > "$tmp/psd_lost.log" 2>&1 \
        && ! grep -q "skipped" "$tmp/psd_lost.log"; then
    echo "ok   [serve gpu error, files on disk] removed"
else
    echo "FAIL [serve gpu error, files on disk]"; grep -E '^(FAIL|ERROR)|Error|skipped' "$tmp/psd_lost.log" | head -5; fail=1
fi
for t in test_qwen38_prefix_serve test_qwen38_prefix_store; do
    if QWEN38_TINY="$tmp/tok" "${PYTHON:-python3}" -W ignore tests/$t.py > "$tmp/$t.log" 2>&1; then
        echo "ok   [$t on the chain] $(grep '^Ran' "$tmp/$t.log")"
    else
        echo "FAIL [$t on the chain]"; grep -E '^(FAIL|ERROR):' "$tmp/$t.log"; fail=1
    fi
done

[ $fail -eq 0 ] && echo "metal_qwen38: PASS" || { echo "metal_qwen38: FAIL"; exit 1; }
