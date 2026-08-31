/* route_trace.h — engine-agnostic routing telemetry: the ROUTE_TRACE stream and the
 * .coli_usage expert history, in one place so every engine emits the same bytes.
 *
 * Why a separate header and not part of telemetry.h: telemetry.h is bound to GLM's
 * structures ("include after Model/Cfg/QT/ESlot/shards ... requires qt_bytes(), now_s(),
 * rss_gb()"), so kimi_k3.c and olmoe.c cannot include it — their Model/Cfg are different
 * types. Routing telemetry does not need shared types: an engine knows its layer index,
 * the expert ids it selected and their gates, and that is the whole input.
 *
 * Dependencies are the C library plus compat.h, and compat.h is NOT optional here: it
 * redirects rename() to MoveFileEx(MOVEFILE_REPLACE_EXISTING), because the CRT rename
 * fails with EEXIST when the destination exists and would therefore silently stop
 * updating the history after its first write on Windows. That shim exists in compat.h
 * *because of* this very code path — see the comment above compat_rename(). Including
 * only <stdio.h> here compiles fine and passes on Linux; it reintroduces that bug.
 * compat.h itself is self-contained (system headers only, no engine types), which is why
 * st.h can include it too.
 *
 * ---- history format (.coli_usage) ----
 * Text, one record per line: "<layer> <expert> <count>", sparse (non-zero only).
 * Two header records carry the dimensions and the writing engine's identity:
 *
 *     -1 <n_layers>       <n_experts>
 *     -2 <format_version> <engine_id>
 *
 * The layer field is NEGATIVE on purpose. Every reader written against the old format is
 * a `while(fscanf(f,"%d %d %u",&l,&e,&c)==3)` loop guarded by `l>=0`, so it parses these
 * two records, discards them, and continues into the data — which is what lets a file
 * written here still load in an engine built before this header existed.
 *
 * Two constraints that field layout is forced by, and they are easy to get wrong:
 *   - a string cannot go in any slot: a non-numeric field makes fscanf return < 3, which
 *     ends the loop and silently drops every record after it. Hence engine_id is a hash.
 *   - engine_id must sit in the THIRD field, the one old readers parse with "%u". A 32-bit
 *     hash routinely exceeds INT_MAX and the second field is read with "%d", where that
 *     is undefined behaviour. So the version (small) goes second, the hash third.
 *
 * Two older layouts are also read, so histories accumulated before this existed keep
 * working — that ranking is the entire value of PIN=auto:
 *   - legacy text: the same triples with no header records. It cannot be validated, so it
 *     is accepted as-is, exactly as before.
 *   - inkling's "IKU1": uint32 {magic, n_layers, n_experts} then a dense
 *     uint32[n_layers][n_experts] block. Only inkling reads it; any other engine refuses
 *     it by name rather than loading another model's ranking.
 *
 * ---- trace format (ROUTE_TRACE=<path>) ----
 * One line per (moe call, batch row): "<call> <row> <layer> <id>:<gate.4f> ...".
 * Byte-identical to what colibri.c emitted before this header; tools/route_pairs.py and
 * the .coli_pairs pipeline consume it unchanged.
 */
#ifndef ROUTE_TRACE_H
#define ROUTE_TRACE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "compat.h"                    /* rename() -> MoveFileEx on Windows; see above */

#define RT_FORMAT_VERSION 1
#define RT_IKU1_MAGIC 0x31554B49u      /* "IKU1" — inkling.c's usage_save/pins_load */

/* engines that write this format; the table exists so a mismatch names both sides */
static const char *rt_engine_names[] = { "glm_moe_dsa", "inkling", "olmoe", "kimi_k3", "deepseek_v4", "qwen38", NULL };

static uint32_t rt_hash(const char *s){        /* FNV-1a 32, stable across builds */
    uint32_t h = 2166136261u;
    for(; s && *s; s++){ h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}
static const char *rt_engine_of(uint32_t id){
    for(int i = 0; rt_engine_names[i]; i++)
        if(rt_hash(rt_engine_names[i]) == id) return rt_engine_names[i];
    return NULL;                                /* unknown writer: report the id */
}

/* ---- state: owned here so a new engine defines no storage and no format ---- */
static uint32_t **rt_c;                         /* [n_layers+1][n_experts] counters */
static int rt_nl = -1, rt_ne;                   /* rows 0..rt_nl inclusive */
static uint32_t rt_id;                          /* this engine's id */
static const char *rt_engine = "";
static FILE *rt_fp;                             /* ROUTE_TRACE stream, NULL = off */
static int rt_call;                             /* moe-call counter, first trace field */

/* n_layers is inclusive: rows 0..n_layers exist, matching the extra MTP row GLM keeps at
 * index n_layers. Engines without one simply never touch it. A failed row allocation
 * leaves that row NULL and is handled the same way an engine's own failed calloc is. */
/* Open the trace stream. Separate from rt_init because it needs no dimensions, so an
 * engine that reads ROUTE_TRACE before it has parsed its config (colibri.c does) keeps
 * the message exactly where it was. Idempotent; a failed open is not retried. */
static void rt_trace_open(void){
    static int tried;
    if(tried) return;
    tried = 1;
    const char *p = getenv("ROUTE_TRACE");
    if(!p || !*p) return;
    rt_fp = fopen(p, "w");
    if(!rt_fp) fprintf(stderr, "[ROUTE_TRACE] cannot open %s\n", p);
    else       fprintf(stderr, "[ROUTE_TRACE] logging routing to %s\n", p);
}

static void rt_init(const char *engine, int n_layers, int n_experts){
    if(n_layers < 0 || n_experts < 1) return;
    rt_engine = engine ? engine : "";
    rt_id = rt_hash(rt_engine);
    rt_c = (uint32_t**)calloc((size_t)n_layers + 1, sizeof(uint32_t*));
    if(!rt_c) return;
    rt_nl = n_layers; rt_ne = n_experts;
    for(int i = 0; i <= n_layers; i++)
        rt_c[i] = (uint32_t*)calloc((size_t)n_experts, sizeof(uint32_t));
    rt_trace_open();                            /* no-op if the engine opened it already */
}

static uint32_t **rt_counts_all(void){ return rt_c; }
static uint32_t  *rt_counts(int layer){
    return (rt_c && layer >= 0 && layer <= rt_nl) ? rt_c[layer] : NULL;
}
static int rt_tracing(void){ return rt_fp != NULL; }

/* Long-lived Segment hosts already own model-specific teardown, while the
 * standalone engines historically relied on process exit.  New engines can
 * use this explicit cleanup so sanitizer runs also cover a complete lifecycle. */
static void rt_destroy(void){
    if(rt_fp){ fclose(rt_fp); rt_fp = NULL; }
    if(rt_c){
        for(int i = 0; i <= rt_nl; i++) free(rt_c[i]);
        free(rt_c);
    }
    rt_c = NULL; rt_nl = -1; rt_ne = 0; rt_id = 0; rt_engine = ""; rt_call = 0;
}

/* Release a layer's counter row, so rt_counts(layer) is NULL for a layer that does not
 * route. rt_init cannot know which those are — an engine learns its own sparsity while it
 * builds its layers, after it has told us its dimensions — so it drops them afterwards.
 *
 * This is not an optimisation, it is the load admission rule. Every reader here treats a
 * NULL row as "this layer has no experts to attribute a count to", which is precisely what
 * the code being replaced expressed by leaving eusage[i] NULL on dense layers. Skip the
 * drop and a history record naming a dense layer is silently absorbed and written back. */
static void rt_drop_row(int layer){
    if(rt_c && layer >= 0 && layer <= rt_nl){ free(rt_c[layer]); rt_c[layer] = NULL; }
}

/* count only — for a router that bumps its other per-expert state in the same loop */
static void rt_count(int layer, const int *ids, int k){
    uint32_t *row = rt_counts(layer);
    if(!row || !ids) return;
    for(int i = 0; i < k; i++)
        if(ids[i] >= 0 && ids[i] < rt_ne) row[ids[i]]++;
}

/* trace only — once per (row), with the gates the layer will actually apply */
static void rt_trace(int layer, int row, const int *ids, const float *gates, int k){
    if(!rt_fp || !ids || !gates) return;
    fprintf(rt_fp, "%d %d %d", rt_call, row, layer);
    for(int i = 0; i < k; i++) fprintf(rt_fp, " %d:%.4f", ids[i], gates[i]);
    fputc('\n', rt_fp);
}
/* advance the call counter: once per moe() invocation, after its rows are traced */
static void rt_trace_end(void){ if(rt_fp) rt_call++; }

/* both, for an engine that does not need them at separate points in its router */
static void rt_route(int layer, int row, const int *ids, const float *gates, int k){
    rt_count(layer, ids, k);
    rt_trace(layer, row, ids, gates, k);
}

/* A typed path that overrides an identity mismatch says so, loudly and once. A mistyped path
 * and a deliberate cross-engine experiment are indistinguishable at the call site but not to
 * the person who typed it, and this line is the only thing separating them. It also travels
 * with any benchmark pasted into the tracker, so a run whose pins came from another engine
 * carries its own explanation for looking slow. Requested in #700.
 *
 * Only placement is at stake, which is why an override exists here at all and does not for a
 * mismatched weight container (#529): pins decide which bytes are resident, never what the
 * model computes. The worst case is slower, self-inflicted, and visible in the hit rate. */
static void rt_say_override(const char *path, const char *writer){
    static int said = 0;
    if(said) return;                            /* once per process, not once per read */
    said = 1;
    fprintf(stderr, "[USAGE] %s: written by %s, this engine is %s\n"
        "        — honouring it because you named it explicitly\n"
        "          (placement only; expert ids do not transfer between engines)\n",
        path, writer, rt_engine);
}

/* ---- one reader, three layouts. cb() receives every (layer, expert, count) record.
 * Returns the total read, or -1 if the file was refused (unreadable, or a dimension /
 * engine mismatch in a file that identifies itself). A file that does not identify
 * itself is accepted without validation: that is the pre-existing behaviour, and the
 * reason the header records were added. ---- */
/* A record callback returns 1 when it accepted the record, and rt_read totals only the
 * accepted ones. This is not a style choice: both readers this replaces added to their
 * total INSIDE the same `if` that admitted the record, and their admission tests differ
 * from each other (usage_load bounds the layer, pin_load bounds it implicitly through its
 * sparsity test). Totalling every parsed record instead would inflate the
 * "[USAGE] expert history: N selections" line for any history whose records fall outside
 * this model's shape — which a legacy headerless history from a larger model does, and a
 * headerless file has no dimension record to be refused by. */
typedef int (*rt_cb)(int layer, int expert, uint32_t count, void *ud);

/* trusted=1 relaxes the IDENTITY check and nothing else. The identity refusal is the one
 * that says "pass PIN=<path> to use it anyway", so a path the user typed has to be honoured
 * or that sentence sends them in a circle; it is also the only way to try one engine's
 * placement priors on another on purpose. The format version and an IKU1 file's dimensions
 * are parse geometry rather than identity and are never relaxed — see each of them below.
 *
 * What earns trust is HOW THE PATH WAS REACHED, not what the file contains. PIN=<file> was
 * typed by a person; PIN=auto and the AUTOPIN seed are paths the engine found by itself and
 * nobody vouched for, so they take the full check. The caller has to tell us which, because
 * one function reads both kinds of path and the file cannot say which it was. */
static int64_t rt_read_ex(const char *path, rt_cb cb, void *ud, int trusted){
    FILE *f = fopen(path, "rb");                /* binary: the magic sniff needs raw bytes,
                                                 * and %d/%u skip \r, so text parses fine */
    if(!f) return -1;
    int64_t tot = 0;
    uint32_t magic = 0;
    if(fread(&magic, 4, 1, f) == 1 && magic == RT_IKU1_MAGIC){
        /* no false positive is possible from our own text: it starts with "-1 " */
        if(strcmp(rt_engine, "inkling") != 0){
            if(!trusted){
                fprintf(stderr, "[USAGE] %s: written by inkling, this engine is %s — refusing "
                    "(pass PIN=<path> to use it anyway)\n", path, rt_engine);
                fclose(f); return -1;
            }
            rt_say_override(path, "inkling");
        }
        uint32_t dim[2];
        if(fread(dim, 4, 2, f) != 2){ fclose(f); return -1; }
        /* Not bypassed by trust: here the dimensions ARE the parse geometry, because a
         * record's layer and expert come from its position in the file rather than from the
         * record itself. Wrong dimensions mean the bytes are being cut up wrongly, which is
         * misreading rather than misattributing, and this refusal never promised PIN=. */
        if((int)dim[0] != rt_nl || (int)dim[1] != rt_ne){
            fprintf(stderr, "[USAGE] %s: history is %u layers x %u experts, this engine is "
                "%d x %d — refusing rather than misreading it\n",
                path, dim[0], dim[1], rt_nl, rt_ne);
            fclose(f); return -1;
        }
        uint32_t *rowbuf = (uint32_t*)malloc((size_t)dim[1] * 4);
        if(!rowbuf){ fclose(f); return -1; }
        for(uint32_t l = 0; l < dim[0]; l++){
            if(fread(rowbuf, 4, dim[1], f) != dim[1]) break;
            for(uint32_t e = 0; e < dim[1]; e++)
                if(rowbuf[e] && cb((int)l, (int)e, rowbuf[e], ud)) tot += rowbuf[e];
        }
        free(rowbuf); fclose(f); return tot;
    }
    rewind(f);
    int l, ver; uint32_t v3;
    while(fscanf(f, "%d %d %u", &l, &ver, &v3) == 3){
        if(l == -1){                                        /* -1 <n_layers> <n_experts> */
            /* Unconditional, like the version check: a file that declares dimensions this
             * model does not have is a mistake, not a preference, and this refusal never
             * offered PIN= as a way past it. Nor is there prior behaviour to keep — the old
             * format carried no dimensions, so no file could previously declare wrong ones.
             * Only identity, below, bends to trust, because only identity promises PIN=. */
            if(ver != rt_nl || (int)v3 != rt_ne){
                fprintf(stderr, "[USAGE] %s: history is %d layers x %u experts, this engine "
                    "is %d x %d — refusing rather than misreading it\n",
                    path, ver, v3, rt_nl, rt_ne);
                fclose(f); return -1;
            }
            continue;
        }
        if(l == -2){                                        /* -2 <version> <engine_id> */
            /* Not bypassed by trust, unlike the identity check below. Trust says whose
             * history the user accepts, not that this build can parse the bytes: a future
             * layout read as v1 is misread, which is the failure this format exists to
             * avoid. There is also no prior behaviour to preserve — no v2 file exists. */
            if(ver > RT_FORMAT_VERSION){
                fprintf(stderr, "[USAGE] %s: format version %d, this build reads %d — "
                    "refusing rather than misreading it\n", path, ver, RT_FORMAT_VERSION);
                fclose(f); return -1;
            }
            if(v3 != rt_id){
                const char *w = rt_engine_of(v3);
                if(!trusted){
                    fprintf(stderr, "[USAGE] %s: written by %s, this engine is %s — refusing "
                        "(pass PIN=<path> to use it anyway)\n",
                        path, w ? w : "an unknown engine", rt_engine);
                    fclose(f); return -1;
                }
                rt_say_override(path, w ? w : "an unknown engine");
            }
            continue;
        }
        if(l < 0) continue;                     /* room for future header records */
        if(cb(l, ver, v3, ud)) tot += v3;       /* data row: ver=expert, v3=count */
    }
    fclose(f);
    return tot;
}
static int64_t rt_read(const char *path, rt_cb cb, void *ud){
    return rt_read_ex(path, cb, ud, 0);
}

/* accumulate a history file into the counters owned here. The admission test is the one
 * usage_load used: in range, and the layer has a counter row — which for an engine that
 * dropped its dense rows (see rt_drop_row) means the layer actually routes. */
static int rt_acc_cb(int l, int e, uint32_t c, void *ud){
    (void)ud;
    uint32_t *row = rt_counts(l);
    if(!row || e < 0 || e >= rt_ne) return 0;
    row[e] += c;
    return 1;
}
static int64_t rt_load(const char *path){
    int64_t t = rt_read(path, rt_acc_cb, NULL);
    return t < 0 ? 0 : t;                       /* callers already treat 0 as "no history" */
}

/* atomic write of the current counters. quiet=0 prints the one-line summary. */
/* COLI_USAGE_DECAY: the persistent history is a cumulative, never-decayed histogram,
 * so its responsiveness falls as it grows. Measured on a host with 18.2M recorded
 * selections: a typical turn contributes ~38k, i.e. it moves the ranking by 0.2%, and
 * the pinned set can no longer follow a change of workload. Multiplying the counters
 * by a factor at each save gives the history a half-life in turns (0.99 -> ~69 turns)
 * while leaving the default (1.0) byte-identical to current behaviour.
 * Placement only: routing math and outputs are untouched, so the token-exact oracle
 * is unaffected. Rounding keeps a count of 1 alive -- forgetting targets the large
 * counters that dominate the ranking, not the long tail. */
static void rt_decay(void){
    static double d = -1.0;
    if(d < 0.0){ const char *e = getenv("COLI_USAGE_DECAY"); d = e ? atof(e) : 1.0;
                 if(d <= 0.0 || d > 1.0) d = 1.0; }
    if(d >= 1.0 || !rt_c) return;
    for(int i = 0; i <= rt_nl; i++){
        if(!rt_c[i]) continue;
        for(int e = 0; e < rt_ne; e++)
            if(rt_c[i][e]) rt_c[i][e] = (unsigned)(rt_c[i][e] * d + 0.5);
    }
}

static int rt_save(const char *path, int quiet){
    if(!rt_c || !path || !*path) return 0;
    /* USAGE_SAVE=0: read-only run — the history is loaded but never written back
     * (#1039: benchmark loops must not skew the very profile they are measuring).
     * Checked HERE so every engine honours the same switch with the same
     * truthiness; a requested skip is a success, not a save failure. */
    { const char *sv = getenv("USAGE_SAVE");
      if(sv && atoi(sv)==0) return 1; }
    rt_decay();
    int64_t tot = 0, nz = 0;
    for(int i = 0; i <= rt_nl; i++){
        if(!rt_c[i]) continue;
        for(int e = 0; e < rt_ne; e++) if(rt_c[i][e]){ tot += rt_c[i][e]; nz++; }
    }
    char tmp[2100];
    /* Truncation here is not cosmetic: fopen/rename below would then write and land on a
     * DIFFERENT path than the caller asked for, silently, with no indication the intended
     * history was never touched. Refuse instead of truncating. A NEGATIVE return
     * (encoding error) is the same hazard in a worse coat: C leaves tmp indeterminate,
     * so proceeding would fopen whatever bytes happen to be there. Same refusal path. */
    int tl = snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    if (tl < 0 || tl >= (int)sizeof(tmp)) {
        if (!quiet) {
            if (tl < 0) fprintf(stderr, "[STATS] snprintf error building temp path, not saved: %s\n", path);
            else        fprintf(stderr, "[STATS] path too long (>%d bytes), not saved: %s\n",
                                (int)sizeof(tmp) - 5, path);
        }
        return 0;
    }
    FILE *f = fopen(tmp, "w");
    if(!f){ if(!quiet) perror(tmp); return 0; }
    /* An all-zero history stays a ZERO-BYTE file, the way it was before this header
     * existed. PIN=auto decides whether a history is usable by testing the file SIZE and
     * falls back to stats.txt when it is empty, so writing two header lines into an empty
     * history would silently cost that fallback. */
    if(nz){
        fprintf(f, "-1 %d %d\n", rt_nl, rt_ne);
        fprintf(f, "-2 %d %u\n", RT_FORMAT_VERSION, rt_id);
        for(int i = 0; i <= rt_nl; i++){
            if(!rt_c[i]) continue;
            for(int e = 0; e < rt_ne; e++)
                if(rt_c[i][e]) fprintf(f, "%d %d %u\n", i, e, rt_c[i][e]);
        }
    }
    fclose(f);
    if(rename(tmp, path) != 0) return 0;
    if(!quiet) fprintf(stderr, "[STATS] %lld selections across %lld distinct experts -> %s\n",
                       (long long)tot, (long long)nz, path);
    return 1;
}

/* ---- router safety: a top-k pick that is always a valid expert id ----------
 *
 * Every engine selects experts with the same loop:
 *
 *     int best = -1; float bv = -1e30f;
 *     for (e ...) if (!taken && score[e] > bv) { bv = score[e]; best = e; }
 *     idx[kk] = best;
 *
 * If every candidate score is NaN, no comparison succeeds -- NaN > bv is false
 * for any bv -- and `best` stays -1. It is then used directly as an index.
 * Downstream that becomes score[-1] (heap read), usage[-1]++ (heap write) and,
 * because an expert id also picks a file offset, a pread at a negative offset.
 * In a release build none of that faults: the process finishes and returns
 * wrong numbers over quietly corrupted memory, which is worse than a crash.
 *
 * NaN reaches the router from a corrupt or hostile expert tile, or from an fp
 * overflow at an eviction boundary; c/tests/test_logit_nan.c documents both and
 * hardened only the sampling side, which is downstream of this.
 *
 * colibri.c has carried this guard as a private router_best_or_fallback() since
 * that test was written. inkling.c, kimi_k3.c and olmoe.c never received it --
 * the recurring shape of defects in this tree, a fix that lands in one engine
 * and not its siblings. It lives here now because route_trace.h was the one
 * header all four original engines already included (newer integrated engines
 * inherit it too), and because "which expert did we pick" is exactly what this
 * file is about.
 *
 * Degrading deterministically (to kk, the slot's own index) keeps the selection
 * reproducible and in range; the warning fires once so a poisoned tile is
 * visible without flooding a long generation. */
static int rt_router_pick(int best, int kk, int experts, int layer) {
    if (best >= 0) return best;
    static int rt_router_warned;
    if (!rt_router_warned) {
        rt_router_warned = 1;
        fprintf(stderr, "[router] non-finite logits at layer %d: selection degraded "
                        "(corrupt expert tile, or overflow at an eviction boundary)\n",
                layer);
    }
    return kk < experts ? kk : 0;
}

#endif /* ROUTE_TRACE_H */
