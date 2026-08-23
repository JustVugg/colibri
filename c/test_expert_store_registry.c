/*
 * Unit test for the pluggable expert-store backend registry.
 *
 * Verifies:
 *  - the built-in "auto" backend is registered at link time (constructor);
 *  - lookup hits for "auto", misses for an unknown name;
 *  - open_selected dispatches to COLI_EXPERT_STORE, errors cleanly on an
 *    unregistered backend, and falls back to "auto" when the env is unset.
 *
 * The real auto open (coli_v4_expert_store_open_planned) lives in deepseek_v4.c;
 * we stub it here so this test links without the engine. The stub records that
 * it was called and returns a sentinel so we can observe dispatch.
 *
 *   gcc -O2 test_expert_store_registry.c expert_store_registry.c -o test_expert_store_registry
 *   ./test_expert_store_registry
 */
#include "expert_store_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_auto_called = 0;

/* Stub for the built-in backend's open fn. */
int coli_v4_expert_store_open_planned(ColiV4Engine *engine,
                                      const ColiDeepSeekV4Config *config,
                                      const ColiDeepSeekV4ExpertStoreOptions *opts,
                                      ColiExpertStore **out,
                                      char *error, size_t error_size) {
    (void)engine; (void)config; (void)opts; (void)out; (void)error; (void)error_size;
    g_auto_called = 1;
    return -1; /* sentinel: dispatched but did not open */
}

int main(void) {
    int n = coli_expert_store_backend_count();
    if (n != 1) {
        printf("FAIL: expected 1 backend (auto) on startup, got %d\n", n);
        return 1;
    }
    if (!coli_expert_store_backend_lookup("auto")) {
        printf("FAIL: 'auto' not registered\n");
        return 1;
    }
    if (coli_expert_store_backend_lookup("does-not-exist")) {
        printf("FAIL: unknown backend unexpectedly found\n");
        return 1;
    }

    char err[128] = {0};
    ColiExpertStore *out = NULL;

    /* Unregistered backend selected by env -> clean error, no dispatch. */
    setenv("COLI_EXPERT_STORE", "example-not-linked", 1);
    g_auto_called = 0;
    int rc = coli_expert_store_backend_open_selected(NULL, NULL, NULL, &out, err, sizeof(err));
    if (rc == 0 || g_auto_called) {
        printf("FAIL: unregistered backend should error without dispatch (rc=%d, auto_called=%d)\n",
               rc, g_auto_called);
        return 1;
    }
    if (!strstr(err, "example-not-linked") || !strstr(err, "not registered")) {
        printf("FAIL: error message wrong: %s\n", err);
        return 1;
    }
    printf("unregistered backend -> clean error: %s\n", err);

    /* Env unset -> default 'auto' -> dispatch to the stub. */
    unsetenv("COLI_EXPERT_STORE");
    g_auto_called = 0;
    err[0] = 0;
    rc = coli_expert_store_backend_open_selected(NULL, NULL, NULL, &out, err, sizeof(err));
    if (!g_auto_called) {
        printf("FAIL: default 'auto' did not dispatch\n");
        return 1;
    }
    /* rc is the stub's -1 sentinel; what matters is that auto was dispatched. */
    (void)rc;
    printf("default (COLI_EXPERT_STORE unset) -> dispatched to 'auto'\n");

    /* Explicit 'auto' also dispatches. */
    setenv("COLI_EXPERT_STORE", "auto", 1);
    g_auto_called = 0;
    coli_expert_store_backend_open_selected(NULL, NULL, NULL, &out, err, sizeof(err));
    if (!g_auto_called) {
        printf("FAIL: explicit 'auto' did not dispatch\n");
        return 1;
    }
    unsetenv("COLI_EXPERT_STORE");
    printf("explicit 'auto' -> dispatched to 'auto'\n");

    printf("ALL OK\n");
    return 0;
}