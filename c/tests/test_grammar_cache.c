/* schema->GBNF compile-cache correctness (#7).
 *
 * mux_submit now skips re-compiling a slot's grammar when the next request sends
 * the IDENTICAL schema/grammar text: it keeps an owned copy in GrDraft.src and,
 * on a match, does grammar_reset() instead of grammar_teardown()+setup. That is
 * only correct if two claims hold, which this test pins on the REAL functions
 * (via the include-colibri.c pattern):
 *
 *   (1) EQUIVALENCE: grammar_reset() on an already-compiled grammar yields the
 *       exact same walker behaviour as a fresh grammar_setup_text() of the same
 *       text -- same forced spans, same accept/reject at every byte.
 *   (2) OWNERSHIP: the GrDraft.src lifecycle (set on compile, freed on teardown)
 *       never leaks or double-frees across the miss/hit/fail/no-grammar paths.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#define CHECK(cond) do { if(!(cond)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#cond); return 1; } } while(0)

/* feed a byte string through a walker; return bytes accepted before a reject */
static int feed(GrState *S, const char *s){
    int n=0; while(s[n]){ if(gr_accept(S,(unsigned char)s[n])!=1) break; n++; } return n;
}

/* Compare fresh-setup vs reset behaviour of the SAME compiled grammar. */
static int check_equivalence(const char *gbnf, const char *sample){
    GrDraft a; memset(&a,0,sizeof a); a.max=24;
    char *t1=malloc(strlen(gbnf)+1); strcpy(t1,gbnf);
    CHECK(grammar_setup_text(&a,NULL,t1,"A")==0);   /* frees t1 */
    char fa[512]; int fa_n=gr_forced(&a.st,fa,sizeof fa);
    int acc_a=feed(&a.st,sample);

    /* reset the SAME draft and re-run: must match itself exactly */
    grammar_reset(&a);
    char fr[512]; int fr_n=gr_forced(&a.st,fr,sizeof fr);
    int acc_r=feed(&a.st,sample);
    CHECK(fr_n==fa_n && !memcmp(fr,fa,(size_t)fa_n) && acc_r==acc_a);

    /* a completely fresh draft from the same text: must also match */
    GrDraft b; memset(&b,0,sizeof b); b.max=24;
    char *t2=malloc(strlen(gbnf)+1); strcpy(t2,gbnf);
    CHECK(grammar_setup_text(&b,NULL,t2,"B")==0);
    char fb[512]; int fb_n=gr_forced(&b.st,fb,sizeof fb);
    int acc_b=feed(&b.st,sample);
    CHECK(fb_n==fa_n && !memcmp(fb,fa,(size_t)fa_n) && acc_b==acc_a);

    grammar_teardown(&a); grammar_teardown(&b);
    return 0;
}

int main(void){
    /* (1) equivalence across representative grammars */
    CHECK(check_equivalence("root ::= \"{\\\"id\\\":\"", "{\"id\":")==0);
    CHECK(check_equivalence("root ::= \"a\" (\"b\" | \"c\") \"d\"", "abd")==0);
    CHECK(check_equivalence("root ::= \"[\" [0-9]+ \"]\"", "[123]")==0);

    /* (2) ownership: replicate the exact mux_submit miss/hit/fail/none flow,
     * setting GrDraft.src exactly as mux_submit does, and exercising teardown. */
    GrDraft gd; memset(&gd,0,sizeof gd); gd.max=24;
    const char *A="root ::= \"x\"";
    for(int turn=0; turn<3; turn++){
        /* HIT path if src matches, else MISS (compile) path */
        if(gd.on && gd.src && !strcmp(gd.src,A)){
            grammar_reset(&gd);                 /* cache hit: reset only */
        } else {
            grammar_teardown(&gd);
            char *keep=malloc(strlen(A)+1); strcpy(keep,A);
            char *txt=malloc(strlen(A)+1); strcpy(txt,A);
            if(grammar_setup_text(&gd,NULL,txt,"req")==0) gd.src=keep; else free(keep);
        }
        CHECK(gd.on && gd.src && !strcmp(gd.src,A));
    }
    /* switch to a different grammar: teardown must free the old src */
    grammar_teardown(&gd);
    CHECK(gd.src==NULL);
    /* a grammar that fails to compile: left recursion -> gr_state_init not alive,
     * grammar_setup_text returns -1. src must stay NULL, no crash. */
    const char *BAD="root ::= root \"a\"";
    char *bad=malloc(strlen(BAD)+1); strcpy(bad,BAD);
    int rc=grammar_setup_text(&gd,NULL,bad,"bad");
    CHECK(rc!=0 && gd.src==NULL);
    grammar_teardown(&gd);                       /* NULL-safe free(src) */

    printf("test_grammar_cache: reset==fresh-setup equivalence + src ownership OK\n");
    return 0;
}
