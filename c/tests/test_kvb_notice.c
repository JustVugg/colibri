/* Fused-Metal format-gate NOTICE test (see metal_fmt_gate_notice in colibri.c, called
 * once from model_init after all layer tensors are resolved).
 *
 * The trap: a v1-class mixed-precision container can mint any fused-path weight tensor
 * at a format the fused Metal decode kernels cannot bind. BOTH fused entries
 * (attention_rows' fused decode and layer_forward_rows' fused layer decode) gate per
 * layer on metal_fused_layer_fmt_miss(): kv_b_proj on the gates' two-format+mode term
 * (int4 per-row fmt==2 always; grouped fmt==4 only while g_moe_exact is off), the other
 * bound tensors (q_a/q_b/kv_a/o and, on sparse layers, sh_gate/sh_up/sh_down) on the
 * metal_fused_fmt_ok() allowlist (fmt 1/2/3/4). A miss silently drops the layer off the
 * fused path onto the CPU implementation -- historically with no notice at all, and the
 * kv_b-only notice this one replaced had itself drifted (it ignored g_moe_exact and
 * stayed silent on fmt=4 kv_b under COLI_METAL_MOE_EXACT=1 while both gates closed).
 * This test proves the notice (stderr only, NOT a behavior change): one line per
 * offending tensor KIND (bounded, never per-layer), silent when clean, silent without
 * Metal, MTP head excluded, and mode-aware exactly like the gates. It also pins the
 * shared predicate's truth table and the gate masks' bit membership, since the gates
 * consume the exact same function. It calls the functions directly against a hand-built
 * Model, so it needs no real snapshot on disk and no Metal backend link --
 * g_metal_enabled and g_moe_exact are declared unconditionally in colibri.c precisely so
 * this stays portable.
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
 * fd 2 so restore_stderr() can put it back. A failed dup/freopen leaves the capture seam
 * (and stderr itself) in an undefined state every later CHECK would silently depend on,
 * so abort the whole test loudly instead of limping on. */
static int redirect_stderr(const char *path){
    fflush(stderr);
    int saved = dup(fileno(stderr));
    if(saved<0){ printf("FAIL: dup(stderr) failed -- capture seam unusable, aborting\n"); exit(1); }
    if(!freopen(path, "w+", stderr)){
        printf("FAIL: freopen(%s) failed -- capture seam unusable, aborting\n", path);
        exit(1);
    }
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

/* Runs the notice against `m` with stderr captured into buf. */
static void run_notice(Model *m, char *buf, size_t bufsz){
    int saved = redirect_stderr("tests/tmp_kvb_notice.stderr");
    metal_fmt_gate_notice(m);
    restore_stderr(saved, buf, bufsz);
    remove("tests/tmp_kvb_notice.stderr");
}

static int count_sub(const char *hay, const char *needle){
    int n=0; const char *p=hay;
    while((p=strstr(p,needle))){ n++; p++; }
    return n;
}

static Model make_model(int n_layers){
    Model m; memset(&m,0,sizeof m);
    m.c.n_layers = n_layers;
    m.L = calloc((size_t)n_layers, sizeof(Layer));
    return m;
}

/* Every fused-bound tensor at an allowlisted format -- deliberately a MIX of all four
 * admitted formats (1/2/3/4) so the allowlist itself is what keeps this silent. */
static void set_clean(Model *m){
    for(int i=0;i<m->c.n_layers;i++){
        Layer *l=&m->L[i];
        l->sparse=1;
        l->kv_b.fmt=2;
        l->q_a.fmt=1; l->q_b.fmt=4; l->kv_a.fmt=3; l->o.fmt=1;
        l->sh_gate.fmt=1; l->sh_up.fmt=2; l->sh_down.fmt=4;
    }
}

/* Kind index k follows the METAL_FUSED_* bit order (1..7 = the allowlist kinds). */
static void set_kind_fmt(Layer *l, int k, int fmt){
    switch(k){
        case 1: l->q_a.fmt=fmt; break;
        case 2: l->q_b.fmt=fmt; break;
        case 3: l->kv_a.fmt=fmt; break;
        case 4: l->o.fmt=fmt; break;
        case 5: l->sh_gate.fmt=fmt; break;
        case 6: l->sh_up.fmt=fmt; break;
        case 7: l->sh_down.fmt=fmt; break;
    }
}
static const char *const kind_str[8] = { "kv_b_proj",
    "q_a_proj","q_b_proj","kv_a_proj_with_mqa","o_proj",
    "shared_experts.gate_proj","shared_experts.up_proj","shared_experts.down_proj" };

int main(void){
    char buf[8192];
    g_moe_exact = 0;                 /* default mode unless a case says otherwise */

    /* (a) fmt=1 kv_b on 2/3 layers + Metal enabled -> the kv_b notice appears exactly
     * once, with fmt, count, requirement, consequence, remedy -- and nothing else.
     * fmt=1 (v1-class INT8 container) is chosen specifically because it misses the
     * kv_b term in BOTH modes. */
    {
        Model m = make_model(3);
        set_clean(&m);
        m.L[0].kv_b.fmt = 1;             /* v1-class INT8 container: not int4 */
        m.L[2].kv_b.fmt = 1;
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);

        CHECK(count_sub(buf, "[METAL] kv_b_proj") == 1);  /* exactly one notice, not one per layer */
        CHECK(count_sub(buf, "[METAL]") == 1);            /* and no allowlist lines: kv_b only */
        CHECK(strstr(buf, "fmt=1") != NULL);               /* which fmt kv_b actually has */
        CHECK(strstr(buf, "2/3 layer") != NULL);            /* count of affected layers */
        CHECK(strstr(buf, "fmt=2") != NULL);               /* mentions the per-row option */
        CHECK(strstr(buf, "fmt=4") != NULL);               /* mentions the grouped option -- fmt=4 is valid outside MOE-exact */
        CHECK(strstr(buf, "CPU absorb path") != NULL);      /* attention runs the CPU path */
        CHECK(strstr(buf, "--kvb-bits 4") != NULL);          /* remedy hint */
        /* mode OFF -> the PLAIN remedy: no --group-size opt-out steering (that flag is
         * the MOE-exact cure; suggesting the accuracy-costing ungrouped requant on an
         * ordinary load would be wrong advice) */
        CHECK(strstr(buf, "--group-size 0") == NULL);
        /* the pre-#587 message's PER-ROW-only warning and fmt=4-trap sentence must be
         * gone -- fmt=4 is a valid GPU-served kv_b outside MOE-exact mode, not a second
         * way to miss the gate. (These pins are carried from dev's test verbatim.) */
        CHECK(strstr(buf, "PER-ROW") == NULL);
        CHECK(strstr(buf, "keeping kv_b ungrouped") == NULL);
        CHECK(strstr(buf, "also misses this fmt=2 gate") == NULL);
        free(m.L);
    }

    /* (a2) fmt=2 kv_b on every layer -> no notice (dev-compat case (b)) */
    {
        Model m = make_model(3);
        set_clean(&m);
        for(int i=0;i<3;i++) m.L[i].kv_b.fmt = 2;
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (a3) fmt=4 (grouped int4) kv_b on every layer, default mode -> no notice: fmt=4
     * is a valid GPU-served kv_b configuration post-#587 while g_moe_exact is off
     * (dev-compat case (b2)). */
    {
        Model m = make_model(3);
        set_clean(&m);
        for(int i=0;i<3;i++) m.L[i].kv_b.fmt = 4;
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (a4) mixed fmt=2/fmt=4 across layers, default mode -> still no notice (the OR
     * term, not just either format alone, is what the fused path requires per-layer;
     * dev-compat case (b3)). */
    {
        Model m = make_model(4);
        set_clean(&m);
        m.L[0].kv_b.fmt = 2; m.L[1].kv_b.fmt = 4; m.L[2].kv_b.fmt = 4; m.L[3].kv_b.fmt = 2;
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (a5) MODE TERM (the r4 drift cure): under g_moe_exact=1 (COLI_METAL_MOE_EXACT=1)
     * the gates' kv_b term is fmt==2 EXACTLY -- grouped fmt=4 closes both gates, so the
     * notice must fire for it. Mixed 2/4: only the fmt=4 layers are misses. The
     * pre-r4 notice (fmt!=2 && fmt!=4) stayed silent here: gates closed, no signal. */
    {
        Model m = make_model(4);
        set_clean(&m);
        m.L[0].kv_b.fmt = 2; m.L[1].kv_b.fmt = 4; m.L[2].kv_b.fmt = 4; m.L[3].kv_b.fmt = 2;
        g_metal_enabled = 1;
        g_moe_exact = 1;
        run_notice(&m, buf, sizeof buf);

        CHECK(count_sub(buf, "[METAL] kv_b_proj") == 1);
        CHECK(count_sub(buf, "[METAL]") == 1);
        CHECK(strstr(buf, "fmt=4") != NULL);               /* the offending format is grouped int4 */
        CHECK(strstr(buf, "2/4 layer") != NULL);            /* only the fmt=4 layers count */
        CHECK(strstr(buf, "COLI_METAL_MOE_EXACT") != NULL); /* names the mode that excludes fmt=4 */
        /* mode ON -> the MODE-AWARE remedy: `--kvb-bits 4` alone is circular here (the
         * converter's default --group-size 64 mints fmt=4 again), so the message must
         * name the ungrouped opt-out and the option of unsetting the mode. */
        CHECK(strstr(buf, "--group-size 0") != NULL);
        CHECK(strstr(buf, "unsetting COLI_METAL_MOE_EXACT") != NULL);

        /* ... and all-fmt=4 under the same mode: 3/3 miss */
        for(int i=0;i<4;i++) m.L[i].kv_b.fmt = 4;
        m.c.n_layers = 3;                                   /* reuse first 3 layers */
        run_notice(&m, buf, sizeof buf);
        CHECK(count_sub(buf, "[METAL] kv_b_proj") == 1);
        CHECK(strstr(buf, "3/3 layer") != NULL);

        /* fmt=2 stays served under MOE-exact: silent */
        for(int i=0;i<3;i++) m.L[i].kv_b.fmt = 2;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);

        g_moe_exact = 0;
        free(m.L);
    }

    /* (b) each allowlist tensor kind individually off-allowlist (fmt=8) on 2/4 layers
     * -> exactly ONE line, naming that kind, its fmt, the layer count, the allowlist
     * requirement, and the CPU consequence. No kv_b line, no other kind named. */
    for(int k=1;k<8;k++){
        Model m = make_model(4);
        set_clean(&m);
        set_kind_fmt(&m.L[1], k, 8);
        set_kind_fmt(&m.L[3], k, 8);
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);

        char head[128];
        snprintf(head, sizeof head, "[METAL] %s is fmt=8", kind_str[k]);
        CHECK(count_sub(buf, "[METAL]") == 1);            /* one line for the one bad kind */
        CHECK(strstr(buf, head) != NULL);                  /* names the kind + actual fmt */
        /* count of affected layers: sh_* kinds count over the sparse population (all 4
         * layers are sparse here), every other kind over all main layers */
        CHECK(strstr(buf, k>=5 ? "2/4 sparse layer" : "2/4 layer") != NULL);
        if(k<5) CHECK(strstr(buf, "sparse") == NULL);      /* attn-bound kinds never say sparse */
        CHECK(strstr(buf, "fmt 1/2/3/4") != NULL);          /* the allowlist requirement */
        CHECK(strstr(buf, "CPU path") != NULL);             /* the fused-path consequence */
        CHECK(strstr(buf, "kv_b_proj") == NULL);           /* clean kv_b stays unmentioned */
        for(int j=1;j<8;j++) if(j!=k)
            CHECK(strstr(buf, kind_str[j]) == NULL);       /* no other kind named */
        free(m.L);
    }

    /* (c) EVERYTHING bad on EVERY layer of a 40-layer model -> bounded output: exactly
     * 8 lines (kv_b + 7 kinds), each kind named exactly once, counts say 40/40 --
     * never per-layer spam (which would be 320 lines here). */
    {
        Model m = make_model(40);
        set_clean(&m);
        for(int i=0;i<40;i++){
            m.L[i].kv_b.fmt = 1;
            for(int k=1;k<8;k++) set_kind_fmt(&m.L[i], k, 9);
        }
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);

        CHECK(count_sub(buf, "[METAL]") == 8);            /* bounded: one line per kind */
        for(int k=0;k<8;k++) CHECK(count_sub(buf, kind_str[k]) == 1);
        CHECK(count_sub(buf, "40/40 layer") == 5);         /* kv_b + q_a/q_b/kv_a/o: all layers */
        CHECK(count_sub(buf, "40/40 sparse layer") == 3);  /* sh_*: all 40 layers are sparse */
        free(m.L);
    }

    /* (d) all-pass silence: every fused-bound tensor on the allowlist -- across all
     * four admitted formats (1/2/3/4, incl. fmt=3) -- and kv_b at fmt=2 -> no output. */
    {
        Model m = make_model(3);
        set_clean(&m);
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (e) Metal not enabled -> no notice even with everything bad (the fused paths this
     * warns about are Metal-only, so with Metal off there is nothing to warn about) */
    {
        Model m = make_model(2);
        set_clean(&m);
        for(int i=0;i<2;i++){
            m.L[i].kv_b.fmt = 1;
            for(int k=1;k<8;k++) set_kind_fmt(&m.L[i], k, 8);
        }
        g_metal_enabled = 0;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (f) MTP-head exclusion: main layers clean, the MTP head (m->mtpL, ships INT8 by
     * design) entirely off-allowlist -> still silent. The notice scans main layers only
     * by ratified decision: the fused attention gate DOES evaluate the MTP row and its
     * INT8 kv_b closes it on every ordinary load (carried, correct CPU fallback), so a
     * notice line about it would be noise on every load, not signal. */
    {
        Model m = make_model(3);
        set_clean(&m);
        m.has_mtp = 1;
        m.mtpL.sparse = 1;
        m.mtpL.kv_b.fmt = 1;
        for(int k=1;k<8;k++) set_kind_fmt(&m.mtpL, k, 8);
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (g) dense-layer sh_* exemption: dense layers never load the shared-expert MLP
     * (sh_* fmt stays 0) and never reach the fused layer CB, so their sh_* must NOT
     * be reported -- a real GLM container has dense layer(s) and would false-positive
     * on every load otherwise. */
    {
        Model m = make_model(3);
        set_clean(&m);
        m.L[0].sparse = 0;
        m.L[0].sh_gate.fmt = 0; m.L[0].sh_up.fmt = 0; m.L[0].sh_down.fmt = 0;
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);
        CHECK(buf[0] == 0);
        free(m.L);
    }

    /* (h) shared-predicate truth table: the gates consume metal_fused_layer_fmt_miss
     * directly, so pin its bit semantics (this is what "the notice cannot drift from
     * the gates" rests on) -- including the kv_b mode term in BOTH g_moe_exact states. */
    {
        Layer l; memset(&l,0,sizeof l);
        l.sparse=1;
        l.kv_b.fmt=2; l.q_a.fmt=1; l.q_b.fmt=2; l.kv_a.fmt=3; l.o.fmt=4;
        l.sh_gate.fmt=1; l.sh_up.fmt=1; l.sh_down.fmt=1;
        CHECK(metal_fused_layer_fmt_miss(&l) == 0);                     /* fully eligible */
        l.kv_b.fmt=4;                                                    /* grouped int4: */
        CHECK(metal_fused_layer_fmt_miss(&l) == 0);                     /* served while MOE-exact off */
        g_moe_exact = 1;
        CHECK(metal_fused_layer_fmt_miss(&l) == METAL_FUSED_KV_B);      /* MOE-exact: fmt==2 exactly */
        l.kv_b.fmt=2;
        CHECK(metal_fused_layer_fmt_miss(&l) == 0);                     /* fmt=2 serves in both modes */
        g_moe_exact = 0;
        l.kv_b.fmt=1;
        CHECK(metal_fused_layer_fmt_miss(&l) == METAL_FUSED_KV_B);      /* non-int4 misses in both modes */
        l.kv_b.fmt=2; l.q_a.fmt=8;
        CHECK(metal_fused_layer_fmt_miss(&l) == METAL_FUSED_Q_A);
        l.q_a.fmt=1; l.o.fmt=0;
        CHECK(metal_fused_layer_fmt_miss(&l) == METAL_FUSED_O);         /* fmt=0 never fused */
        l.o.fmt=4; l.sh_down.fmt=5;
        CHECK(metal_fused_layer_fmt_miss(&l) == METAL_FUSED_SH_DOWN);
        CHECK(!(metal_fused_layer_fmt_miss(&l) & METAL_FUSED_ATTN_TENSORS)); /* attn mask blind to sh_* */
        l.sparse=0;
        CHECK(metal_fused_layer_fmt_miss(&l) == 0);                     /* dense: sh_* exempt */
    }

    /* (i) mixed dense/sparse, GLM shape: 61 layers, layer 0 dense, everything bad on
     * every layer it can be bad on -> still exactly 8 bounded lines; attn-bound kinds
     * count over ALL 61 layers, sh_* kinds over the 60-layer sparse population only
     * (a 61/61 there would claim a miss on a layer that never loads the tensor). */
    {
        Model m = make_model(61);
        set_clean(&m);
        m.L[0].sparse = 0;
        m.L[0].sh_gate.fmt = 0; m.L[0].sh_up.fmt = 0; m.L[0].sh_down.fmt = 0;
        for(int i=0;i<61;i++){
            m.L[i].kv_b.fmt = 1;
            for(int k=1;k<5;k++) set_kind_fmt(&m.L[i], k, 8);
            if(m.L[i].sparse) for(int k=5;k<8;k++) set_kind_fmt(&m.L[i], k, 8);
        }
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);

        CHECK(count_sub(buf, "[METAL]") == 8);             /* bounded on the mixed shape too */
        CHECK(count_sub(buf, "61/61 layer") == 5);          /* kv_b + q_a/q_b/kv_a/o: all layers */
        CHECK(count_sub(buf, "60/60 sparse layer") == 3);   /* sh_*: sparse population only */
        free(m.L);
    }

    /* (j) mask BIT COMPOSITION: the truth table in (h) proves the helper's bits, but
     * not which bits each GATE mask contains -- a mask silently dropping a member
     * (e.g. METAL_FUSED_O falling out of METAL_FUSED_ATTN_TENSORS) would fail open on
     * the fused path with every other test still green. Assert, per kind, the masked
     * gate booleans the gates actually compute against membership stated independently
     * of the enum definitions: attn mask = kv_b+q_a+q_b+kv_a+o and NOT sh_*; layer
     * mask = all 8. */
    {
        static const int in_attn[8]  = {1,1,1,1,1,0,0,0};
        static const int in_layer[8] = {1,1,1,1,1,1,1,1};
        for(int k=0;k<8;k++){
            Layer l; memset(&l,0,sizeof l);
            l.sparse=1;
            l.kv_b.fmt=2; l.q_a.fmt=1; l.q_b.fmt=2; l.kv_a.fmt=3; l.o.fmt=4;
            l.sh_gate.fmt=1; l.sh_up.fmt=1; l.sh_down.fmt=1;
            if(k==0) l.kv_b.fmt=1; else set_kind_fmt(&l, k, 8);   /* exactly kind k misses */
            unsigned miss = metal_fused_layer_fmt_miss(&l);
            int attn_pass  = !(miss & METAL_FUSED_ATTN_TENSORS);   /* the gates' own tests */
            int layer_pass = !(miss & METAL_FUSED_LAYER_TENSORS);
            if(attn_pass != !in_attn[k] || layer_pass != !in_layer[k])
                printf("FAIL mask membership: kind %s attn_pass=%d layer_pass=%d\n",
                       kind_str[k], attn_pass, layer_pass);
            CHECK(attn_pass == !in_attn[k]);
            CHECK(layer_pass == !in_layer[k]);
        }
        /* the kv_b MODE miss flows through the same masks: fmt=4 + MOE-exact closes
         * both masked gates, exactly like the fused gates themselves. */
        {
            Layer l; memset(&l,0,sizeof l);
            l.sparse=1;
            l.kv_b.fmt=4; l.q_a.fmt=1; l.q_b.fmt=2; l.kv_a.fmt=3; l.o.fmt=4;
            l.sh_gate.fmt=1; l.sh_up.fmt=1; l.sh_down.fmt=1;
            g_moe_exact = 1;
            unsigned miss = metal_fused_layer_fmt_miss(&l);
            CHECK((miss & METAL_FUSED_ATTN_TENSORS) != 0);
            CHECK((miss & METAL_FUSED_LAYER_TENSORS) != 0);
            g_moe_exact = 0;
            miss = metal_fused_layer_fmt_miss(&l);
            CHECK(!(miss & METAL_FUSED_ATTN_TENSORS));
            CHECK(!(miss & METAL_FUSED_LAYER_TENSORS));
        }
        /* and a fully clean layer passes both masked gates */
        Layer l; memset(&l,0,sizeof l);
        l.sparse=1;
        l.kv_b.fmt=2; l.q_a.fmt=1; l.q_b.fmt=2; l.kv_a.fmt=3; l.o.fmt=4;
        l.sh_gate.fmt=1; l.sh_up.fmt=1; l.sh_down.fmt=1;
        unsigned miss = metal_fused_layer_fmt_miss(&l);
        CHECK(!(miss & METAL_FUSED_ATTN_TENSORS));
        CHECK(!(miss & METAL_FUSED_LAYER_TENSORS));
    }

    /* (k) ACCEPTED BEHAVIOR (documented, not guarded): runtime-quant dense configs land
     * every fused-bound tensor on one qt_alloc rung, and the notice reports that
     * truthfully under FORMAT-ONLY SEMANTICS. dense@16 (bits>=16 -> fmt=0 everywhere):
     * fmt=0 never fused, so ALL 8 kinds miss -> exactly 8 accurate lines, every one
     * naming fmt=0 -- bounded, correct, and deliberate (a load that can never fuse for
     * format reasons deserves to say so once per kind). dense@3 (bits==3 -> fmt=3
     * everywhere): fmt=3 is ON the allowlist, so only kv_b misses (an int2 kv_b is not
     * int4) -> exactly 1 line. This case pins the behavior so it is claimed and tested
     * rather than accidental. */
    {
        Model m = make_model(4);
        for(int i=0;i<4;i++){
            Layer *l=&m.L[i]; l->sparse=1;
            l->kv_b.fmt=0; l->q_a.fmt=0; l->q_b.fmt=0; l->kv_a.fmt=0; l->o.fmt=0;
            l->sh_gate.fmt=0; l->sh_up.fmt=0; l->sh_down.fmt=0;
        }
        g_metal_enabled = 1;
        run_notice(&m, buf, sizeof buf);
        CHECK(count_sub(buf, "[METAL]") == 8);             /* bounded: one line per kind */
        for(int k=0;k<8;k++) CHECK(count_sub(buf, kind_str[k]) == 1);
        CHECK(count_sub(buf, "fmt=0") == 8);                /* every line names the real fmt */
        CHECK(count_sub(buf, "4/4 layer") == 5);            /* kv_b + q_a/q_b/kv_a/o */
        CHECK(count_sub(buf, "4/4 sparse layer") == 3);     /* sh_* */

        for(int i=0;i<4;i++){
            Layer *l=&m.L[i];
            l->kv_b.fmt=3; l->q_a.fmt=3; l->q_b.fmt=3; l->kv_a.fmt=3; l->o.fmt=3;
            l->sh_gate.fmt=3; l->sh_up.fmt=3; l->sh_down.fmt=3;
        }
        run_notice(&m, buf, sizeof buf);
        CHECK(count_sub(buf, "[METAL]") == 1);              /* allowlist kinds all pass */
        CHECK(count_sub(buf, "[METAL] kv_b_proj") == 1);    /* the one miss is kv_b */
        CHECK(strstr(buf, "fmt=3") != NULL);
        CHECK(strstr(buf, "4/4 layer") != NULL);
        free(m.L);
    }

    if(fails){ printf("fused-metal fmt-gate notice tests: %d FAILED\n", fails); return 1; }
    printf("fused-metal fmt-gate notice tests: ok\n");
    return 0;
}
