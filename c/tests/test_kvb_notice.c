/* kv_b fmt-gate NOTICE test (see kvb_fmt_gate_notice in colibri.c, called once from
 * model_init after all layer tensors are resolved).
 *
 * The trap: a v1-class mixed-precision container can mint kv_b_proj at a format outside
 * the pair the fused Metal attention kernels actually serve. BOTH fused entries
 * (attention_rows' fused decode and layer_forward_rows' fused layer decode) hard-gate on
 * `l->kv_b.fmt==2||l->kv_b.fmt==4` (per-row or grouped int4 -- fmt=4 became a valid
 * GPU-served kv_b configuration once #587's grouped-int4 kv_b support landed), so a kv_b
 * outside {2,4} silently forces that layer's decode attention onto the CPU absorb path --
 * with no notice at all if this fires. This test proves the fix (a stderr notice, NOT a
 * behavior change): it fires exactly once when it should, and stays silent for BOTH valid
 * formats (fmt=2 and fmt=4) as well as for the MTP head (excluded by design: the notice
 * only scans m->L[0..n_layers), never m->mtpL). It calls kvb_fmt_gate_notice() directly
 * against a hand-built Model, so it needs no real snapshot on disk and no Metal backend
 * link -- g_metal_enabled is declared unconditionally in colibri.c precisely so this
 * stays portable.
 *
 * Portable stderr capture: freopen(stderr) onto a temp file, dup()/dup2() to save and
 * restore the real fd 2 around it. Deliberately no child-process spawn/reap primitives
 * anywhere in this file -- those are POSIX-only and the earlier version of this kind of
 * test broke Windows CI. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* Redirects stderr to `path` (truncated, read+write) and returns a dup of the original
 * fd 2 so restore_stderr() can put it back. */
static int redirect_stderr(const char *path){
    fflush(stderr);
    int saved = dup(fileno(stderr));
    if(saved<0){ printf("FAIL: dup(stderr) failed\n"); fails++; }
    if(!freopen(path, "w+", stderr)){ printf("FAIL: freopen(%s) failed\n", path); fails++; }
    return saved;
}
/* Reads back everything written to stderr since redirect_stderr(), restores the real
 * stderr fd, and NUL-terminates the captured text into buf. */
static void restore_stderr(int saved, char *buf, size_t bufsz){
    fflush(stderr);
    long n = ftell(stderr);
    if(n<0) n=0;
    rewind(stderr);
    size_t want = (size_t)n < bufsz-1 ? (size_t)n : bufsz-1;
    size_t got = fread(buf, 1, want, stderr);
    buf[got] = 0;
    fflush(stderr);
    dup2(saved, fileno(stderr));      /* fd 2 back to the real stderr */
    close(saved);
}

static Model make_model(int n_layers){
    Model m; memset(&m,0,sizeof m);
    m.c.n_layers = n_layers;
    m.L = calloc((size_t)n_layers, sizeof(Layer));
    return m;
}

int main(void){
    char buf[4096];

    /* (a) fmt=1 kv_b on 2/3 layers (the other is fmt=2) + Metal enabled -> notice fires
     * exactly once, for a format OUTSIDE {2,4} -- fmt=1 (v1-class INT8 container) is
     * chosen specifically because it's neither of the two now-valid GPU-served formats. */
    {
        Model m = make_model(3);
        m.L[0].kv_b.fmt = 1;             /* v1-class INT8 container: outside {2,4} */
        m.L[1].kv_b.fmt = 2;
        m.L[2].kv_b.fmt = 1;
        g_metal_enabled = 1;

        int saved = redirect_stderr("tests/tmp_kvb_notice_a.stderr");
        kvb_fmt_gate_notice(&m);
        restore_stderr(saved, buf, sizeof buf);
        remove("tests/tmp_kvb_notice_a.stderr");

        int occurrences = 0; const char *p = buf;
        while((p = strstr(p, "[METAL] kv_b_proj"))){ occurrences++; p++; }
        CHECK(occurrences == 1);                          /* exactly one notice, not one per layer */
        CHECK(strstr(buf, "fmt=1") != NULL);               /* which fmt kv_b actually has */
        CHECK(strstr(buf, "2/3 layer") != NULL);            /* count of affected layers */
        CHECK(strstr(buf, "fmt=2") != NULL);               /* mentions the per-row option */
        CHECK(strstr(buf, "fmt=4") != NULL);               /* mentions the grouped option -- fmt=4 is now valid */
        CHECK(strstr(buf, "CPU absorb path") != NULL);      /* attention runs the CPU path */
        CHECK(strstr(buf, "--kvb-bits 4") != NULL);          /* remedy hint */
        /* the pre-#587 message's PER-ROW-only warning and fmt=4-trap sentence must be
         * gone -- fmt=4 is a valid GPU-served kv_b now, not a second way to miss the gate. */
        CHECK(strstr(buf, "PER-ROW") == NULL);
        CHECK(strstr(buf, "keeping kv_b ungrouped") == NULL);
        CHECK(strstr(buf, "also misses this fmt=2 gate") == NULL);
        free(m.L);
    }

    /* (b) fmt=2 kv_b on every layer -> no notice, even with Metal enabled */
    {
        Model m = make_model(3);
        for(int i=0;i<3;i++) m.L[i].kv_b.fmt = 2;
        g_metal_enabled = 1;

        int saved = redirect_stderr("tests/tmp_kvb_notice_b.stderr");
        kvb_fmt_gate_notice(&m);
        restore_stderr(saved, buf, sizeof buf);
        remove("tests/tmp_kvb_notice_b.stderr");

        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (b2) fmt=4 (grouped int4) kv_b on every layer -> no notice either: fmt=4 is a valid
     * GPU-served kv_b configuration post-#587, not a trap (this is the semantic change
     * under test -- the pre-#587 notice fired here). */
    {
        Model m = make_model(3);
        for(int i=0;i<3;i++) m.L[i].kv_b.fmt = 4;
        g_metal_enabled = 1;

        int saved = redirect_stderr("tests/tmp_kvb_notice_b2.stderr");
        kvb_fmt_gate_notice(&m);
        restore_stderr(saved, buf, sizeof buf);
        remove("tests/tmp_kvb_notice_b2.stderr");

        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (b3) mixed fmt=2/fmt=4 across layers -> still no notice (the OR gate, not just
     * either format alone, is what the fused path actually requires per-layer). */
    {
        Model m = make_model(4);
        m.L[0].kv_b.fmt = 2; m.L[1].kv_b.fmt = 4; m.L[2].kv_b.fmt = 4; m.L[3].kv_b.fmt = 2;
        g_metal_enabled = 1;

        int saved = redirect_stderr("tests/tmp_kvb_notice_b3.stderr");
        kvb_fmt_gate_notice(&m);
        restore_stderr(saved, buf, sizeof buf);
        remove("tests/tmp_kvb_notice_b3.stderr");

        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (c) fmt=1 kv_b but Metal not enabled -> no notice (the fused paths this warns
     * about are Metal-only, so with Metal off there is nothing to warn about) */
    {
        Model m = make_model(2);
        m.L[0].kv_b.fmt = 1; m.L[1].kv_b.fmt = 1;
        g_metal_enabled = 0;

        int saved = redirect_stderr("tests/tmp_kvb_notice_c.stderr");
        kvb_fmt_gate_notice(&m);
        restore_stderr(saved, buf, sizeof buf);
        remove("tests/tmp_kvb_notice_c.stderr");

        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (d) MTP-exclusion: the notice scans m->L[0..n_layers) only. A bad format planted
     * on m->mtpL (the separate MTP-head Layer, never part of m->L) must NOT be counted or
     * reported, even with every real layer at a valid format and Metal enabled -- the
     * fused layer-decode gate itself already excludes the MTP head (li<n_layers), and
     * common containers ship it at INT8 by design, so warning about it here would be both
     * wrong (this notice's gate doesn't apply to it) and noise on every ordinary load. */
    {
        Model m = make_model(3);
        for(int i=0;i<3;i++) m.L[i].kv_b.fmt = 2;
        m.mtpL.kv_b.fmt = 1;             /* bad format, but on the excluded MTP head */
        g_metal_enabled = 1;

        int saved = redirect_stderr("tests/tmp_kvb_notice_d.stderr");
        kvb_fmt_gate_notice(&m);
        restore_stderr(saved, buf, sizeof buf);
        remove("tests/tmp_kvb_notice_d.stderr");

        CHECK(buf[0] == 0);
        free(m.L);
    }

    if(fails){ printf("kvb fmt-gate notice tests: %d FAILED\n", fails); return 1; }
    printf("kvb fmt-gate notice tests: ok\n");
    return 0;
}
