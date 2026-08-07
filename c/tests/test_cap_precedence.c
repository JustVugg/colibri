/* test_cap_precedence.c -- #379 expert-cache-cap precedence contract.
 * coli_resolve_cap() (colibri.c) must implement exactly: explicit CLI positional >
 * explicit CAP env > platform default (Metal+darwin+fast SSD, cap=1) >
 * historic default. The historic default depends on how the engine was
 * invoked: a bare `./glm` (no positional at all) keeps the engine's old 64,
 * the coli wrapper's "0 = auto" sentinel keeps the 8 coli always forced.
 * It is a pure function (no I/O, no globals), so this exercises it directly
 * with fabricated inputs -- no model, no probe, no Metal hardware needed.
 * Portable: builds and runs on every platform, same as test_idot.c. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int failures = 0;

static void check(const char *label, int cli_given, int cli, int env, int platform_fast,
                   int want_cap, int want_explicit){
    int got_explicit = -1;
    int got_cap = coli_resolve_cap(cli_given, cli, env, platform_fast, &got_explicit);
    if(got_cap != want_cap || got_explicit != want_explicit){
        fprintf(stderr, "FAIL %-36s given=%d cli=%d env=%d fast=%d -> cap=%d explicit=%d "
                "(want cap=%d explicit=%d)\n",
                label, cli_given, cli, env, platform_fast, got_cap, got_explicit,
                want_cap, want_explicit);
        failures++;
    }
}

static void check_autopin(const char *label, double planned, double available,
                          double lru, double expected){
    double got=autopin_preserve_lru(planned,available,lru);
    if(fabs(got-expected)>1e-9){
        fprintf(stderr,"FAIL %-36s -> %.3f (want %.3f)\n",label,got,expected);
        failures++;
    }
}

static void check_lru_reserve(const char *label, double available, double slot,
                              int requested, int expected_cap, double expected_bytes){
    int cap=-1;
    double got=autopin_lru_reserve(available,slot,requested,&cap);
    if(cap!=expected_cap || fabs(got-expected_bytes)>1e-9){
        fprintf(stderr,"FAIL %-36s -> cap %d, %.3f (want %d, %.3f)\n",
                label,cap,got,expected_cap,expected_bytes);
        failures++;
    }
}

int main(void){
    /* bare invocation (argc<=1, no positional): byte-identical to the old
     * argc>1?atoi(argv[1]):64 fallback on every non-qualifying path. */
    check("bare, slow/no-metal",              0, 0, 0, 0, 64, 0);
    check("bare, fast SSD",                   0, 0, 0, 1,  1, 0);
    check("bare + CAP env, slow",             0, 0, 4, 0,  4, 1);
    check("bare + CAP env, fast",             0, 0, 4, 1,  4, 1);
    /* wrapper sentinel (coli passes 0 = "auto"): coli's historic 8, or the
     * platform default when it engages. */
    check("sentinel, slow/no-metal",          1, 0, 0, 0,  8, 0);
    check("sentinel, fast SSD",               1, 0, 0, 1,  1, 0);
    check("sentinel + CAP env, slow",         1, 0, 6, 0,  6, 1);
    check("sentinel + CAP env, fast",         1, 0, 6, 1,  6, 1);
    /* explicit CLI always wins, whatever the platform says. */
    check("CLI wins over fast platform",      1, 16, 0, 1, 16, 1);
    check("CLI wins over slow platform",      1, 16, 0, 0, 16, 1);
    /* explicit CAP env loses to CLI. */
    check("CLI wins over env",                1, 16, 4, 1, 16, 1);
    /* cap=0 from either source is "not given", never a literal request for
     * zero cache slots (undocumented and unused repo-wide before #379). */
    check("sentinel does not read as explicit",1, 0, 0, 1,  1, 0);
    /* negative isn't 0, so it counts as explicit even though it is a
     * nonsensical cap -- the engine has never validated cap's range; this
     * test pins the precedence contract, not value sanity. */
    check("negative CLI still explicit",      1, -1, 0, 1, -1, 1);

    check_autopin("nine slots preserve eight",4.5,9.0,8.0,1.0);
    check_autopin("sixteen slots keep half",8.0,16.0,8.0,8.0);
    check_autopin("twenty slots keep plan",10.0,20.0,8.0,10.0);
    check_autopin("insufficient for requested LRU",4.0,6.0,8.0,0.0);
    check_autopin("small plan capped",2.0,9.0,8.0,1.0);
    check_autopin("zero plan",0.0,9.0,8.0,0.0);
    check_lru_reserve("requested cap fits",16.0,1.0,8,8,8.0);
    check_lru_reserve("reserve only affordable cap",9.0,2.0,8,4,8.0);
    check_lru_reserve("zero slot size",9.0,0.0,8,0,0.0);
    check_lru_reserve("zero requested cap",9.0,1.0,0,0,0.0);

    if(failures){
        fprintf(stderr, "cap precedence tests: %d FAILURE(S)\n", failures);
        return 1;
    }
    puts("cap precedence tests: ok");
    return 0;
}
