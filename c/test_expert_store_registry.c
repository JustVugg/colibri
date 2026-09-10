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

/* Windows has no setenv/unsetenv at all -- MinGW fails this file at link time
 * with "undefined reference to `setenv'".  compat.h's SetEnvironmentVariableA
 * shim fixes the link but not the test: coli_expert_store_backend_open_selected
 * reads COLI_EXPERT_STORE with getenv(), and getenv() reads the CRT's own copy
 * of the environment, which that shim does not update -- the same trap
 * test_qwen36_ctx.c and test_inkling_shared_batch.c document. _putenv_s is the
 * one that updates the copy getenv() reads. */
#ifdef _WIN32
static void env_set(const char *name, const char *value) { _putenv_s(name, value); }
static void env_unset(const char *name) { _putenv_s(name, ""); }
#else
static void env_set(const char *name, const char *value) { setenv(name, value, 1); }
static void env_unset(const char *name) { unsetenv(name); }
#endif

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
    env_set("COLI_EXPERT_STORE", "example-not-linked");
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
    env_unset("COLI_EXPERT_STORE");
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
    env_set("COLI_EXPERT_STORE", "auto");
    g_auto_called = 0;
    coli_expert_store_backend_open_selected(NULL, NULL, NULL, &out, err, sizeof(err));
    if (!g_auto_called) {
        printf("FAIL: explicit 'auto' did not dispatch\n");
        return 1;
    }
    env_unset("COLI_EXPERT_STORE");
    printf("explicit 'auto' -> dispatched to 'auto'\n");

    printf("ALL OK\n");
    return 0;
}