#!/usr/bin/env bash
# vk_golden_parity.sh — issue #19: three-leg golden-path parity.
#
# PORT ADAPTATION (D11, 2026-10-24; Source: colibri-ds4-vulkan@3207b6bf
# scripts/vk_golden_parity.sh, copied verbatim then minimally re-rooted):
#   - this copy lives in the port repo (colibri-upstream, port/up-2276); legs
#     B/C build THIS tree, and leg A pins GOLDEN_UPSTREAM_REF — for the D11
#     re-record the pin is the port HEAD itself (073a03b6), so the banked
#     fixtures are this tree's own pre-Option-B CPU-path outputs (the surgery
#     baseline later legs validate against; the fork's .golden/cache at
#     dd7df2c9 in /root/Projects/colibri-ds4-vulkan is untouched).
#   - leg C arms the GPU with COLI_VULKAN=1 (the port's deployed enable,
#     bench/up2276/run-arm2276.sh); the fork's COLI_DSV4_VK_DENSE/EXPERTS
#     levers were not ported. The split contract stands: --oracle validation
#     calls target_batch(NULL,…) so every oracle leg is CPU-bound by
#     construction (docs/golden-parity-cost.md §3 in the fork); the port has
#     no cell-hash A/B, so this harness's comparison class is the whole gate.
#   - BASE_ENV stays the fork's compose env verbatim; the real-checkpoint legs
#     add the port campaign's no-warm-start/runner knobs V4_PREFIX_CKPT=0
#     COLI_V4_PREWARM=0 (bench/up2276/run-arm2276.sh env; KVSAVE in the port
#     is server-side, the engine's cold-tier off-switch is V4_PREFIX_CKPT=0).
#     The tiny legs run BASE_ENV bare: the port's own prefix test exercises
#     the cold tier and fails with it disabled.
#   - the fork header documents GOLDEN_UPSTREAM_REF as an env override but the
#     parser only reads the --upstream-ref flag (fork doc drift, unchanged
#     here); pass --upstream-ref REF.
#
#   A. upstream_cpu   — JustVugg/colibri CPU-only, built from the fork base
#                       (default: merge-base of HEAD and upstream/main; the
#                       exact upstream commit this repo's CPU path derives
#                       from — override with GOLDEN_UPSTREAM_REF=upstream/main
#                       to measure pure upstream drift instead).
#   B. fork_cpu       — this repo's binary, Vulkan tiers OFF (backend off).
#   C. fork_gpu       — this repo's binary, tiers ON as deployed:
#                       COLI_DSV4_VK_DENSE=1 + COLI_DSV4_VK_EXPERTS=1 (v5
#                       ExpertCache), auto RAM (no --memory-gb — the
#                       deployment's RAM_GB=0 config), CTX/lanes/union/O_DIRECT
#                       per colibri-v4-docker compose.
#
# Gates:
#   tiny   — all three legs token-exact vs c/deepseek_v4_tiny/ref.json (the
#            generated torch-reference oracle; tests/test_deepseek_v4_tiny.py
#            + test_deepseek_v4_prefix.py, 12/12 + 3/3).
#   real   — the real checkpoint: leg A RECORDS oracle fixtures (teacher-
#            forcing predictions + greedy ids) for each prompt; legs B and C
#            VALIDATE against them token-exact (engine --oracle mode). Two
#            passes:
#              deploy — representative env (deployment knobs, autopin history
#                       machinery as each tree ships it). Mismatch is REPORTED
#                       (deviation-source analysis + stability probe), not
#                       fatal: docs/deepseek-v4.md §CPU-only audit already
#                       showed greedy text is not run-stable across configs
#                       even upstream-vs-upstream with history on.
#              pinned — + COLI_V4_ROWS16=0 COLI_V4_AUTOPIN=0
#                       COLI_V4_SAVE_USAGE=0 (reference kernels, no history —
#                       the audit's determinism envelope). Mismatch is FATAL:
#                       this is the hard golden-path gate.
#
# Quiescing (built in): stops the colibri-v4 deployment container if running
# and kills stray deepseek_v4/coli processes before any run (co-tenant RAM/
# VRAM otherwise poisons auto-sizing — issues #2's lesson). The container is
# NOT restarted afterwards; `docker start colibri-v4` when done. The model
# dir's .coli_usage history is snapshotted and restored on exit.
#
# Usage:
#   ./scripts/vk_golden_parity.sh [--quick] [--skip-real]
#                              [--skip-tiny] [--keep-container]
#                              [--restore-container] [--upstream-ref REF]
#                              [--record-only] [--pinned-only]
#                              [--no-cache] [--smoke[=SEED]]
#
#   0.2 fixture cache: leg-A oracle fixtures are cached under
#   .golden/cache/<upstream_ref>/ and reused, so a GPU-only change re-runs no
#   leg A. The fork-CPU leg is canary-validated (the shortest prompt is always
#   re-run; the long-prompt cache is trusted only if the canary matches).
#   --no-cache forces a full re-record. --smoke[=SEED] runs ONE leg-C validate
#   on a random (seed-logged) fixture from SMOKE_POOL (default: short) — the
#   per-commit loop. The long prompt is a slice-boundary leg.
#
#   Clean builds of BOTH legs are ALWAYS enforced (deepseek-v4-clean first,
#   no reuse path): the unit objects do not track the VK flag — a stale
#   VK=0 tree silently relinking into a "VK=1" parity run is exactly the
#   class of silent wrong-answer this harness exists to catch.
#
#   --record-only captures the leg-A oracle fixtures (and the prompt text)
#   without running legs B/C. Use it to bank goldens on a quiet box (issue
#   #42 / plan 0.1a); the per-leg `[time]` lines are the cost model.
#   --pinned-only skips the deploy pass entirely (pinned is the hard gate;
#   deploy is non-fatal attribution and doubles a long prompt's record cost).
#   Typically combined: --record-only --pinned-only.
#
# Long prompt (0.1a, issues #19/#42): targets >= GOLDEN_LONG_TARGET tokens
# (default 2100). The historical default measured only ~580 engine tokens —
# far BELOW the issue #40/#41 join crossover (position 2051) — so the gate
# could not see that bug class. Blocks derive from the MEASURED ~145
# tokens/block (11 blocks -> 1599 tokens, run-20260916-101159); the record
# path parses the engine's own `v4_tokens prompt=` and fails if short.
# GOLDEN_LONG_BLOCKS pins the exact paragraph-block count for reproduction.
#
# Env:
#   MODEL              real checkpoint dir (default: colibri-v4-docker path)
#   GOLDEN_UPSTREAM_REF upstream ref for leg A (default: fork merge-base)
#   GOLDEN_PROMPT      extra short prompt (added to the default set)
#   GOLDEN_MAX_TOKENS  greedy tokens for short prompts (default 16)
#   GOLDEN_LONG_TOKENS greedy tokens for the long prompt (default 24)
#   GOLDEN_TF          teacher-forcing positions for short prompts (default 32)
#   GOLDEN_LONG_TARGET target tokens for the long prompt (default 2100)
#   GOLDEN_LONG_BLOCKS explicit block count (overrides GOLDEN_LONG_TARGET)
#   GOLDEN_ARCH        -march for builds (default native)
#   SMOKE_POOL         smoke prompt pool: short (default) | any
#
# Exit: 0 = tiny + pinned-real gates green (deploy-real findings, if any, are
# in the report) — or, under --record-only, the leg-A records captured;
# 1 = a hard gate failed; 2 = inconclusive (the GPU leg could not boot its
# tier, e.g. co-tenant VRAM). Artifacts: .golden/run-*/ (report.md, per-run
# stdout/stderr, oracle fixtures, console.log with `[time]` per-leg walls).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="$ROOT/c"
GOLDEN="$ROOT/.golden"
MODEL="${MODEL:-/root/Projects/colibri-v4-docker/models/DeepSeek-V4-Flash}"
GOLDEN_ARCH="${GOLDEN_ARCH:-native}"
GOLDEN_MAX_TOKENS="${GOLDEN_MAX_TOKENS:-16}"
GOLDEN_LONG_TOKENS="${GOLDEN_LONG_TOKENS:-24}"
GOLDEN_TF="${GOLDEN_TF:-32}"
GOLDEN_LONG_TARGET="${GOLDEN_LONG_TARGET:-2100}"
GOLDEN_LONG_BLOCKS="${GOLDEN_LONG_BLOCKS:-}"
SMOKE_POOL="${SMOKE_POOL:-short}"
UPSTREAM_REF=""
QUICK=0; SKIP_REAL=0; SKIP_TINY=0; KEEP_CONTAINER=0; RESTORE_CONTAINER=0
RECORD_ONLY=0; PINNED_ONLY=0; NO_CACHE=0; SMOKE=0; SMOKE_SEED=""

for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --skip-real) SKIP_REAL=1 ;;
        --skip-tiny) SKIP_TINY=1 ;;
        --keep-container) KEEP_CONTAINER=1 ;;
        --restore-container) RESTORE_CONTAINER=1 ;;
        --record-only) RECORD_ONLY=1 ;;
        --pinned-only) PINNED_ONLY=1 ;;
        --no-cache) NO_CACHE=1 ;;
        --smoke) SMOKE=1 ;;
        --smoke=*) SMOKE=1; SMOKE_SEED="${arg#*=}" ;;
        --upstream-ref) : ;;   # consumed with its value below
        --upstream-ref=*) UPSTREAM_REF="${arg#*=}" ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done
# `--upstream-ref REF` (two tokens)
prev=""
for arg in "$@"; do
    if [ "$prev" = "--upstream-ref" ]; then UPSTREAM_REF="$arg"; fi
    prev="$arg"
done

# Leg selection: default record+cpu+gpu; --record-only = record; --smoke = gpu
# only (one prompt). --smoke implies --pinned-only (one GPU validate, one pass).
LEGS="record,cpu,gpu"
[ "$RECORD_ONLY" = 1 ] && LEGS="record"
[ "$SMOKE" = 1 ] && { LEGS="gpu"; PINNED_ONLY=1; }
leg_on() { case ",$LEGS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$GOLDEN/run-$STAMP"
mkdir -p "$OUT"
SCRIPT_T0="$(date +%s.%N)"

step() { printf '\n=== %s ===\n' "$*"; tee -a "$OUT/console.log" >/dev/null; }
say()  { printf '%s\n' "$*" | tee -a "$OUT/console.log"; }
# 0.1a per-leg wall logging (the #42 cost model). Prints to stdout (the run
# log) and appends the same line to console.log.
elapsed() { # label start-epoch-seconds
    local end; end="$(date +%s.%N)"
    awk -v a="$2" -v b="$end" -v l="$1" \
        'BEGIN{printf "[time] %-24s %9.1fs\n", l, b-a}' | tee -a "$OUT/console.log"
}
# 0.2 fixture-cache key (stable hash of an arbitrary string).
sig_of() { # string
    if command -v sha1sum >/dev/null 2>&1; then
        printf '%s' "$1" | sha1sum | cut -c1-12
    else
        printf '%s' "$1" | cksum | tr -d ' '
    fi
}

# ---------------------------------------------------------------- 0. quiesce
step "0 quiesce the host (co-tenant RAM/VRAM poison auto-sizing)"
if command -v docker >/dev/null 2>&1 && [ "$KEEP_CONTAINER" = 0 ]; then
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx colibri-v4; then
        say "stopping deployment container colibri-v4 (docker start colibri-v4 to restore)"
        docker stop colibri-v4 >/dev/null
    else
        say "deployment container not running"
    fi
fi
if pkill -x deepseek_v4 2>/dev/null; then say "killed stray deepseek_v4 process(es)"; sleep 2; fi
say "RAM now: $(free -h | awk '/^Mem:/{print $7" available"}')"

# Snapshot .coli_usage histories (upstream's autopin writes <model>/.coli_usage;
# restore on exit so the harness never permanently alters deployment state —
# including deleting any .coli_usage the harness's own runs created).
USAGE_SNAPSHOT="$OUT/coli_usage.snapshot"
mkdir -p "$USAGE_SNAPSHOT"
USAGE_DIRS=("$MODEL" "$C/deepseek_v4_tiny")
snapshot_usage() {
    : > "$USAGE_SNAPSHOT/manifest"
    local i=0 d
    for d in "${USAGE_DIRS[@]}"; do
        if [ -f "$d/.coli_usage" ]; then
            cp "$d/.coli_usage" "$USAGE_SNAPSHOT/$i.bin"
            echo "$d/.coli_usage" >> "$USAGE_SNAPSHOT/manifest"
        fi
        i=$((i+1))
    done
    say "snapshotted $(wc -l < "$USAGE_SNAPSHOT/manifest") .coli_usage file(s)"
}
restore_usage() {
    local i=0 line d
    if [ -f "$USAGE_SNAPSHOT/manifest" ]; then
        while IFS= read -r line; do
            cp "$USAGE_SNAPSHOT/$i.bin" "$line" && say "restored $line"
            i=$((i+1))
        done < "$USAGE_SNAPSHOT/manifest"
    fi
    # remove files the harness's own runs created (not in the manifest)
    for d in "${USAGE_DIRS[@]}"; do
        if [ -f "$d/.coli_usage" ] && ! grep -qxF "$d/.coli_usage" "$USAGE_SNAPSHOT/manifest" 2>/dev/null; then
            rm -f "$d/.coli_usage" && say "removed harness-created $d/.coli_usage"
        fi
    done
}
cleanup() {
    restore_usage || true
    if [ "$RESTORE_CONTAINER" = 1 ] && command -v docker >/dev/null 2>&1; then
        docker start colibri-v4 >/dev/null 2>&1 && say "restarted colibri-v4 container" || true
    fi
}
trap cleanup EXIT
snapshot_usage || true

# ---------------------------------------------------------------- 1. builds
step "1 build the three legs"
if [ -z "$UPSTREAM_REF" ]; then
    UPSTREAM_REF="$(git -C "$ROOT" merge-base HEAD upstream/main 2>/dev/null || true)"
    [ -n "$UPSTREAM_REF" ] || { echo "cannot compute merge-base with upstream/main" >&2; exit 2; }
    say "leg A ref = fork merge-base $UPSTREAM_REF (GOLDEN_UPSTREAM_REF to override)"
fi
# 0.2 fixture cache, keyed by the exact upstream ref (a fixed ref is a fixed
# program). A GPU-only change then re-runs no leg A; --no-cache re-records.
CACHE_DIR="$GOLDEN/cache/$UPSTREAM_REF"
mkdir -p "$CACHE_DIR"
say "fixture cache: $CACHE_DIR$([ "$NO_CACHE" = 1 ] && echo ' (--no-cache: forcing re-record)' || echo '')"
UP="$GOLDEN/upstream"
if git -C "$ROOT" worktree list --porcelain 2>/dev/null | grep -qx "worktree $UP"; then
    git -C "$UP" checkout --detach --quiet "$UPSTREAM_REF"
    git -C "$UP" clean -fdq c
else
    git -C "$ROOT" worktree add --detach "$UP" "$UPSTREAM_REF" >/dev/null
fi
say "leg A: clean-build upstream CPU-only at $(git -C "$UP" log -1 --oneline)"
_t_build="$(date +%s.%N)"
make -C "$UP/c" deepseek-v4-clean >/dev/null 2>&1 || true
make -C "$UP/c" deepseek-v4 ARCH="$GOLDEN_ARCH" -j"$(nproc)"
elapsed "build.upstream_cpu" "$_t_build"
UPBIN="$UP/c/deepseek_v4"
UP_NM="$(nm "$UPBIN" 2>/dev/null || true)"
if grep -q "coli_vk_init" <<< "$UP_NM"; then
    echo "ERROR: upstream binary has Vulkan symbols — wrong tree?" >&2; exit 2
fi

say "leg B/C: clean-build fork VK=1 at $(git -C "$ROOT" log -1 --oneline)"
# Clean ALWAYS: unit objects do not track the VK flag (same .o names) — a
# VK=0-built tree makes a "VK=1" link silently relink the VK=0 objects.
_t_build="$(date +%s.%N)"
make -C "$C" -f Makefile.deepseek-v4 deepseek-v4-clean
make -C "$C" -f Makefile.deepseek-v4 deepseek-v4 VK=1 ARCH="$GOLDEN_ARCH" -j"$(nproc)"
elapsed "build.fork_vk1" "$_t_build"
FORKBIN="$C/deepseek_v4"
FORK_NM="$(nm "$FORKBIN" 2>/dev/null || true)"
grep -q "coli_vk_init" <<< "$FORK_NM" || { echo "ERROR: fork binary lacks Vulkan symbols" >&2; exit 2; }
say "builds ok: $UPBIN / $FORKBIN"

# ---------------------------------------------------------------- env legs
# Representative engine env (colibri-v4-docker compose): auto RAM (no cap),
# CTX 4096, expert-union prefill, 9 loader lanes, O_DIRECT, greedy, no
# .coli_kv warm start. OMP pinned identical across legs (deployment value).
BASE_ENV=(CTX=4096 OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
          COLI_TEMP=0 KVSAVE=0 V4_EXPERT_UNION=1 V4_LOADER_LANES=9
          COLI_V4_DIRECT=1)
# Port real-leg additions (see header): no cold-KV warm start, no prewarm.
PORT_REAL_ENV=(V4_PREFIX_CKPT=0 COLI_V4_PREWARM=0)
# Determinism envelope (docs/deepseek-v4.md CPU-only audit): reference
# kernels, no autopin history read/write. No-ops on trees without the knobs.
PIN_ENV=(COLI_V4_ROWS16=0 COLI_V4_AUTOPIN=0 COLI_V4_SAVE_USAGE=0)

TINY_PASS=0; REAL_PINNED_PASS=0; DEPLOY_FINDINGS=0; RECORD_CRASH=0; LONG_TOO_SHORT=0; GPU_INCONCLUSIVE=0

# ---------------------------------------------------------------- 2. tiny
if [ "$SKIP_TINY" = 0 ]; then
step "2 tiny oracle — three legs, token-exact vs ref.json"
FIXTURE="$C/deepseek_v4_tiny"
if [ ! -f "$FIXTURE/model.safetensors" ]; then
    make -C "$C" deepseek-v4-tiny-generate
fi
run_tiny() { # leg-name binary extra-env...
    local name="$1" bin="$2"; shift 2
    local log="$OUT/tiny.$name.log"
    local _t_tiny; _t_tiny="$(date +%s.%N)"
    say "--- tiny / $name ---"
    if env "${BASE_ENV[@]}" "$@" \
        python3 "$C/tests/test_deepseek_v4_tiny.py" --binary "$bin" --fixture "$FIXTURE" \
        >"$log" 2>&1 && \
       env "${BASE_ENV[@]}" "$@" \
        python3 "$C/tests/test_deepseek_v4_prefix.py" --binary "$bin" --fixture "$FIXTURE" \
        >>"$log" 2>&1; then
        say "PASS tiny/$name ($(grep -c '^PASS' "$log") checks)"
        elapsed "tiny.$name" "$_t_tiny"
        return 0
    fi
    say "FAIL tiny/$name — tail:"; tail -15 "$log" | sed 's/^/    /' | tee -a "$OUT/console.log"
    elapsed "tiny.$name" "$_t_tiny"
    return 1
}
tiny_fail=0
run_tiny upstream_cpu "$UPBIN" || tiny_fail=1
run_tiny fork_cpu    "$FORKBIN" || tiny_fail=1
run_tiny fork_gpu    "$FORKBIN" COLI_VULKAN=1 || tiny_fail=1
if [ "$tiny_fail" = 0 ]; then TINY_PASS=1; say "tiny gate GREEN (3/3 legs token-exact)"; else say "tiny gate RED"; fi
fi

# ---------------------------------------------------------------- 3. real
if [ "$SKIP_REAL" = 0 ]; then
if [ ! -d "$MODEL" ]; then
    say "MODEL not found at $MODEL — skipping the real-checkpoint gate"
else
step "3 real checkpoint — oracle record (A) then validate (B, C)"

# Prompt set. The long prompt is the deterministic paragraph tiling used by
# colibri-v4-docker's bench/benchmark.py; 0.1a raised the target above the
# issue #40/#41 join crossover (position 2051). Measured 2026-09-16: 11
# blocks == 1599 engine tokens == ~145 tokens/block, so the historical
# "826-token" label was low by ~1.4x; the divisor below is that measured
# value and the record path re-checks the engine's actual count.
PARAGRAPH="The deep sea is the largest and least explored habitat on Earth, a cold, dark realm where pressure increases by roughly one atmosphere for every ten meters of depth. Sunlight fades entirely within the first two hundred meters, leaving the vast majority of this world in permanent darkness. Yet life thrives here: bioluminescent organisms produce their own light to hunt, confuse predators, or communicate with potential mates across vast distances. Hydrothermal vents support entire ecosystems that rely on chemosynthesis rather than photosynthesis, drawing energy from mineral-rich water superheated by volcanic activity beneath the ocean floor. Submersibles must withstand crushing forces, and every expedition returns with new species and new questions about how life persists in such extreme conditions."
LONG_PROMPT="$(python3 - "$PARAGRAPH" "$GOLDEN_LONG_TARGET" "$GOLDEN_LONG_BLOCKS" <<'EOF'
import sys, math
p, target, blocks = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n = int(blocks) if blocks else max(1, math.ceil(target / 145.0))
print("\n\n".join(f"[{i}] {p}" for i in range(n)), end="")
EOF
)"
printf '%s' "$LONG_PROMPT" > "$OUT/long-prompt.txt"
say "long prompt: ${#LONG_PROMPT} chars (target >= $GOLDEN_LONG_TARGET tokens; GOLDEN_LONG_BLOCKS='${GOLDEN_LONG_BLOCKS:-auto}')"

declare -a PROMPTS=() PTOKS=() PTF=()
add_prompt() { PROMPTS+=("$1"); PTOKS+=("$2"); PTF+=("$3"); }
add_prompt "The capital of France is" "$GOLDEN_MAX_TOKENS" "$GOLDEN_TF"
if [ -n "${GOLDEN_PROMPT:-}" ]; then add_prompt "$GOLDEN_PROMPT" "$GOLDEN_MAX_TOKENS" "$GOLDEN_TF"; fi
if [ "$QUICK" = 0 ]; then
    # --prompt-file via a temp copy (the oracle tool takes --prompt only)
    add_prompt "@$OUT/long-prompt.txt" "$GOLDEN_LONG_TOKENS" 900
fi
SMOKE_IDX=0
if [ "$SMOKE" = 1 ]; then
    SMOKE_N=${#PROMPTS[@]}
    # E1 (#42): default the per-commit smoke pool to the SHORT prompts only.
    # A uniform pick over all prompts landed on the long leg (~42 min,
    # CPU-bound) in ~1/3 of runs; long legs belong to slice boundaries.
    if [ "$SMOKE_POOL" != any ] && [ "$QUICK" = 0 ]; then
        SMOKE_N=$((SMOKE_N - 1))   # the long prompt is the last entry
    fi
    [ "$SMOKE_N" -lt 1 ] && SMOKE_N=1
    case "$SMOKE_SEED" in ''|*[!0-9]*) SMOKE_SEED="" ;; esac
    SMOKE_SEED="${SMOKE_SEED:-$RANDOM}"
    RANDOM="$SMOKE_SEED"
    SMOKE_IDX=$((RANDOM % SMOKE_N))
    say "--- smoke: seed=$SMOKE_SEED -> prompt p$SMOKE_IDX of $SMOKE_N (leg C only, pool=$SMOKE_POOL) ---"
fi

run_engine() { # out-prefix binary env... -- engine args...
    local prefix="$1" bin="$2"; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local _t_run rc=0; _t_run="$(date +%s.%N)"
    ( cd "$OUT" && env "${envs[@]}" "$bin" "$@" \
        >"$OUT/$prefix.out" 2>"$OUT/$prefix.err" ) || rc=$?
    elapsed "$prefix" "$_t_run"
    return $rc
}

validate_leg() { # pass leg-name fixture tf greedy binary env... (env list LAST)
    local pass="$1" name="$2" fixture="$3" tf="$4" greedy="$5" bin="$6"; shift 6
    say "--- $pass / $name : validate vs $(basename "$fixture") (tf=$tf greedy=$greedy) ---"
    if run_engine "$pass.$name" "$bin" "$@" -- \
            "$MODEL" --oracle "$fixture" --teacher-forcing "$tf" --greedy "$greedy"; then
        grep -E 'PREFILL \(teacher-forcing\)|GREEDY C vs oracle' "$OUT/$pass.$name.out" \
            | sed 's/^/    /' | tee -a "$OUT/console.log"
        say "PASS $pass/$name"
        return 0
    fi
    say "MISMATCH $pass/$name — first divergences:"
    grep -E '\[ORACLE\].*mismatch' "$OUT/$pass.$name.err" | head -3 | sed 's/^/    /' | tee -a "$OUT/console.log"
    grep -E 'PREFILL \(teacher-forcing\)|GREEDY C vs oracle' "$OUT/$pass.$name.out" | sed 's/^/    /' | tee -a "$OUT/console.log"
    return 1
}

real_pass() { # pass-name extra-env... (legs per $LEGS; --smoke filters prompts)
    local pass="$1"; shift
    local envs=("$@")
    local pin_fail=0
    local tagi=0
    local first=1
    local canary_ok=0
    for p in "${PROMPTS[@]}"; do
        local tokens="${PTOKS[$tagi]}" tf="${PTF[$tagi]}"
        local tag="p$tagi"
        if [ "$SMOKE" = 1 ] && [ "$tagi" != "$SMOKE_IDX" ]; then
            tagi=$((tagi+1)); first=0; continue
        fi
        local ptext="$p"
        if [[ "$p" == @* ]]; then ptext="$(cat "${p#@}")"; fi
        local psig; psig="$(sig_of "$pass|$tokens|$tf|$ptext")"
        local fixture="$OUT/oracle.$pass.$tag.json"

        # ---- leg A: upstream record (fixture cache keyed by ref+prompt+env) --
        if leg_on record || [ "$SMOKE" = 1 ]; then
            local cA="$CACHE_DIR/oracle.$pass.$tag.$psig.json"
            local ctok="$CACHE_DIR/oracle.$pass.$tag.$psig.tokens"
            if [ "$NO_CACHE" = 0 ] && [ -f "$cA" ]; then
                cp "$cA" "$fixture"
                say "--- $pass / leg A CACHE HIT $tag (sig=$psig) ---"
            else
                say "--- $pass / record leg A (upstream_cpu) [$tag, ${#ptext} chars] ---"
                if ! run_engine "$pass.record" "$UPBIN" "${envs[@]}" -- \
                        "$MODEL" "$ptext" --max-tokens "$tokens" --record-oracle "$fixture"; then
                    say "FAIL: leg A record crashed (see $OUT/$pass.record.err)"
                    RECORD_CRASH=1
                    [ "$pass" = pinned ] && pin_fail=1 || DEPLOY_FINDINGS=$((DEPLOY_FINDINGS+1))
                    tagi=$((tagi+1)); first=0; continue
                fi
                if [ "$NO_CACHE" = 0 ]; then
                    cp "$fixture" "$cA"
                    grep -ao 'v4_tokens prompt=[0-9]*' "$OUT/$pass.record.err" 2>/dev/null \
                        | tail -1 | grep -o '[0-9]*$' > "$ctok" || true
                fi
            fi
            local rtok=""; [ -f "$ctok" ] && rtok="$(cat "$ctok")"
            if [ -n "$rtok" ]; then
                say "  prompt_tokens=$rtok (greedy=$tokens, tf=$tf)"
                if [ "$tf" -ge 900 ] && [ "$rtok" -lt "$GOLDEN_LONG_TARGET" ]; then
                    say "  WARNING: long prompt is $rtok tokens (< $GOLDEN_LONG_TARGET) — join crossover NOT exercised"
                    LONG_TOO_SHORT=1
                fi
            fi
        fi

        # ---- leg B: fork CPU (canary-validated cache) -------------------
        if leg_on cpu; then
            local cB="$CACHE_DIR/cpu.$pass.$tag.$psig.txt"
            if [ "$first" = 1 ]; then
                if validate_leg "$pass" fork_cpu "$fixture" "$tf" "$tokens" "$FORKBIN" "${envs[@]}"; then
                    if [ "$NO_CACHE" = 0 ]; then
                        if [ -f "$cB" ] && ! cmp -s "$OUT/$pass.fork_cpu.out" "$cB"; then
                            say "  canary MISMATCH — leg B cache invalidated for this pass"
                            canary_ok=0
                        else
                            say "  canary OK (leg B cache valid for this pass)"
                            canary_ok=1
                        fi
                        cp "$OUT/$pass.fork_cpu.out" "$cB"
                    else
                        canary_ok=1
                    fi
                else
                    [ "$pass" = pinned ] && pin_fail=1 || DEPLOY_FINDINGS=$((DEPLOY_FINDINGS+1))
                fi
            elif [ "$canary_ok" = 1 ] && [ -f "$cB" ] && [ "$NO_CACHE" = 0 ]; then
                say "--- $pass / fork_cpu CACHE HIT $tag (canary-validated) ---"
            else
                if validate_leg "$pass" fork_cpu "$fixture" "$tf" "$tokens" "$FORKBIN" "${envs[@]}"; then
                    [ "$NO_CACHE" = 0 ] && cp "$OUT/$pass.fork_cpu.out" "$cB"
                else
                    [ "$pass" = pinned ] && pin_fail=1 || DEPLOY_FINDINGS=$((DEPLOY_FINDINGS+1))
                fi
            fi
        fi

        # ---- leg C: fork GPU (always run; the thing under test) ---------
        if leg_on gpu; then
            if ! validate_leg "$pass" fork_gpu "$fixture" "$tf" "$tokens" "$FORKBIN" "${envs[@]}" \
                    COLI_VULKAN=1; then
                if grep -qE 'tier=(off|unavailable)|init-failed|backend-unavailable' \
                        "$OUT/$pass.fork_gpu.err" 2>/dev/null; then
                    say "  INCONCLUSIVE: fork_gpu could not boot the tier (co-tenant VRAM?)"
                    GPU_INCONCLUSIVE=1
                else
                    [ "$pass" = pinned ] && pin_fail=1 || DEPLOY_FINDINGS=$((DEPLOY_FINDINGS+1))
                fi
            fi
        fi

        tagi=$((tagi+1)); first=0
    done
    if [ "$pin_fail" = 1 ]; then REAL_PINNED_PASS=0; else REAL_PINNED_PASS=1; fi
    return $pin_fail
}

# Deploy pass: representative env. Non-fatal by design (see header); on a
# mismatch, re-validate A against its own fixture to test upstream's own
# run-to-run stability in this config (deviation-source attribution).
if [ "$PINNED_ONLY" = 1 ]; then
    step "3a real / deploy pass SKIPPED (--pinned-only)"
    say "deploy pass skipped (--pinned-only; pinned is the hard gate)"
else
step "3a real / deploy pass (representative env — findings recorded, not fatal)"
deploy_rc=0
real_pass deploy "${BASE_ENV[@]}" "${PORT_REAL_ENV[@]}" || deploy_rc=1
if [ "$RECORD_ONLY" = 0 ] && [ "$deploy_rc" = 1 ]; then
    say "deploy-pass mismatch — probing upstream self-stability (A vs A, deploy env)"
    idx=0
    for p in "${PROMPTS[@]}"; do
        local tf="${PTF[$idx]}" tokens="${PTOKS[$idx]}"
        validate_leg deploy upstream_cpu "$OUT/oracle.deploy.p$idx.json" "$tf" "$tokens" \
            "$UPBIN" "${BASE_ENV[@]}" \
            && say "upstream self-stable on p$idx → divergence is cross-config" \
            || say "upstream NOT self-stable on p$idx → history/kernel nondeterminism dominates"
        idx=$((idx+1))
    done
fi
fi

# Pinned pass: the hard gate.
step "3b real / pinned pass (determinism envelope — HARD gate)"
pinned_rc=0
real_pass pinned "${BASE_ENV[@]}" "${PIN_ENV[@]}" "${PORT_REAL_ENV[@]}" || pinned_rc=1
say "pinned real gate $([ "$REAL_PINNED_PASS" = 1 ] && echo GREEN || echo RED)"
fi
fi

# ---------------------------------------------------------------- 4. report
step "4 summary"
{
    echo "# golden parity report — $STAMP"
    echo
    echo "- fork HEAD: $(git -C "$ROOT" log -1 --oneline)"
    echo "- leg A ref: $UPSTREAM_REF — $(git -C "$ROOT" log -1 --oneline "$UPSTREAM_REF")"
    echo "- model: $MODEL"
    echo "- tiny gate: $([ "$TINY_PASS" = 1 ] && echo GREEN || echo RED)"
    echo "- real deploy pass: $([ "$DEPLOY_FINDINGS" = 0 ] && echo GREEN || echo "findings: $DEPLOY_FINDINGS")"
    echo "- real pinned gate: $([ "$REAL_PINNED_PASS" = 1 ] && echo GREEN || echo RED)"
    echo "- deploy-pass findings: $DEPLOY_FINDINGS"
    echo "- mode: legs=$LEGS$([ "$PINNED_ONLY" = 1 ] && echo ' deploy-skipped' || echo '')$([ "$SMOKE" = 1 ] && echo " smoke seed=$SMOKE_SEED idx=$SMOKE_IDX pool=$SMOKE_POOL" || echo '')"
    echo "- oracle validation: CPU-path (target_batch(NULL,…)); GPU parity = scripts/vk_prefill_cell_ab.sh"
    echo "- cache: $CACHE_DIR$([ "$NO_CACHE" = 1 ] && echo ' (disabled)' || echo '')"
    echo "- gpu-inconclusive: $GPU_INCONCLUSIVE"
    echo "- long prompt: >= $GOLDEN_LONG_TARGET tokens (GOLDEN_LONG_BLOCKS='${GOLDEN_LONG_BLOCKS:-auto}')"
    echo "- total elapsed: $(awk -v a="$SCRIPT_T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f s", b-a}')"
    echo "- per-leg walls: see console.log ([time] lines)"
    echo
    echo "Artifacts: $OUT (per-run stdout/stderr, oracle fixtures, console.log)"
} | tee "$OUT/report.md"

say "report: $OUT/report.md"
if [ "$RECORD_ONLY" = 1 ]; then
    if [ "$RECORD_CRASH" = 0 ] && [ "$LONG_TOO_SHORT" = 0 ]; then
        say "RECORD-ONLY OK (leg-A goldens captured; validation skipped)"
        exit 0
    fi
    if [ "$RECORD_CRASH" = 1 ]; then say "RECORD-ONLY FAILED (a leg-A record crashed)"; fi
    if [ "$LONG_TOO_SHORT" = 1 ]; then say "RECORD-ONLY FAILED (long prompt below GOLDEN_LONG_TARGET)"; fi
    exit 1
fi
if [ "$GPU_INCONCLUSIVE" = 1 ] && [ "$REAL_PINNED_PASS" = 1 ]; then
    say "INCONCLUSIVE: the GPU tier could not boot (co-tenant VRAM?) — no divergence seen; exit 2"
    exit 2
fi
[ "$TINY_PASS" = 1 ] && [ "$REAL_PINNED_PASS" = 1 ] && say "ALL HARD GATES GREEN" || { say "A HARD GATE FAILED"; exit 1; }
