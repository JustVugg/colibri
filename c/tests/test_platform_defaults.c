/* platform_defaults.h: unset and empty variables take the table's values, set ones keep theirs, COLI_PLATFORM_DEFAULTS=0
 * sets nothing, the returned count and list are right, a list longer than its buffer is cut and terminated; a group is
 * set whole or not at all.
 *   cc -std=c11 -Wall -Wextra -I. tests/test_platform_defaults.c -o test_platform_defaults && ./test_platform_defaults */
#define _POSIX_C_SOURCE 200809L
#include "../platform_defaults.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static const char *env(const char *n) { const char *v = getenv(n); return v ? v : "(unset)"; }

int main(void) {
    static const ColiDefaultKV t[] = { {"TPD_A", "4"}, {"TPD_B", "1"}, {"TPD_C", "f16"}, {"TPD_D", "128"} };
    char got[256];

    unsetenv("COLI_PLATFORM_DEFAULTS");
    unsetenv("TPD_A"); setenv("TPD_B", "0", 1); setenv("TPD_C", "", 1); unsetenv("TPD_D");
    int n = coli_platform_defaults_apply(t, 4, got, sizeof got);
    CHECK(n == 3);
    CHECK(!strcmp(env("TPD_A"), "4"));
    CHECK(!strcmp(env("TPD_B"), "0"));          /* an explicit value wins, even "0" */
    CHECK(!strcmp(env("TPD_C"), "f16"));        /* empty counts as unset */
    CHECK(!strcmp(env("TPD_D"), "128"));
    CHECK(!strcmp(got, "TPD_A=4 TPD_C=f16 TPD_D=128"));

    /* a second application sets nothing and leaves an empty list */
    n = coli_platform_defaults_apply(t, 4, got, sizeof got);
    CHECK(n == 0 && got[0] == 0);

    /* COLI_PLATFORM_DEFAULTS=0: nothing */
    unsetenv("TPD_A"); unsetenv("TPD_D");
    setenv("COLI_PLATFORM_DEFAULTS", "0", 1);
    n = coli_platform_defaults_apply(t, 4, got, sizeof got);
    CHECK(n == 0 && !strcmp(env("TPD_A"), "(unset)") && !strcmp(env("TPD_D"), "(unset)") && got[0] == 0);
    setenv("COLI_PLATFORM_DEFAULTS", "1", 1);   /* anything else leaves them on */
    n = coli_platform_defaults_apply(t, 4, got, sizeof got);
    CHECK(n == 2 && !strcmp(env("TPD_A"), "4") && !strcmp(env("TPD_D"), "128"));

    /* a short buffer: the count is still every variable set, the list is cut and terminated */
    unsetenv("COLI_PLATFORM_DEFAULTS");
    unsetenv("TPD_A"); unsetenv("TPD_B"); unsetenv("TPD_C"); unsetenv("TPD_D");
    char small[10];
    n = coli_platform_defaults_apply(t, 4, small, sizeof small);
    CHECK(n == 4);
    CHECK(strlen(small) == sizeof small - 1 && !strncmp(small, "TPD_A=4 T", sizeof small - 1));
    /* no list wanted */
    unsetenv("TPD_A");
    CHECK(coli_platform_defaults_apply(t, 4, NULL, 0) == 1 && !strcmp(env("TPD_A"), "4"));

    /* a group: all of it when every member is unset or empty, none of it when one is set */
    static const ColiDefaultKV g[] = { {"TPD_G1", "1"}, {"TPD_G2", "1"}, {"TPD_G3", "x=0,y=all"} };
    unsetenv("TPD_G1"); setenv("TPD_G2", "", 1); unsetenv("TPD_G3");
    n = coli_platform_defaults_apply_group(g, 3, got, sizeof got);
    CHECK(n == 3 && !strcmp(env("TPD_G1"), "1") && !strcmp(env("TPD_G2"), "1") && !strcmp(env("TPD_G3"), "x=0,y=all"));
    CHECK(!strcmp(got, "TPD_G1=1 TPD_G2=1 TPD_G3=x=0,y=all"));
    unsetenv("TPD_G1"); unsetenv("TPD_G2"); setenv("TPD_G3", "y=0", 1);   /* one set by hand: none */
    strcpy(got, "stale");
    n = coli_platform_defaults_apply_group(g, 3, got, sizeof got);
    CHECK(n == 0 && got[0] == 0 && !strcmp(env("TPD_G1"), "(unset)") && !strcmp(env("TPD_G2"), "(unset)"));
    CHECK(!strcmp(env("TPD_G3"), "y=0"));
    setenv("TPD_G1", "0", 1); unsetenv("TPD_G3");                          /* even "0" counts as set */
    CHECK(coli_platform_defaults_apply_group(g, 3, got, sizeof got) == 0 && !strcmp(env("TPD_G3"), "(unset)"));
    unsetenv("TPD_G1");
    setenv("COLI_PLATFORM_DEFAULTS", "0", 1);                              /* the switch turns groups off too */
    CHECK(coli_platform_defaults_apply_group(g, 3, got, sizeof got) == 0 && !strcmp(env("TPD_G1"), "(unset)"));
    unsetenv("COLI_PLATFORM_DEFAULTS");
    /* whole or nothing: a member that cannot be set (a name with '=' is refused) unsets the ones set before it */
    static const ColiDefaultKV bad[] = { {"TPD_R1", "1"}, {"TPD_R2=X", "1"} };
    unsetenv("TPD_R1");
    strcpy(got, "stale");
    CHECK(coli_platform_defaults_apply_group(bad, 2, got, sizeof got) == 0 && got[0] == 0);
    CHECK(!strcmp(env("TPD_R1"), "(unset)"));
    /* a short buffer: the whole group is still set, the list is cut and terminated */
    unsetenv("TPD_G1"); unsetenv("TPD_G2"); unsetenv("TPD_G3");
    char tiny[8];
    CHECK(coli_platform_defaults_apply_group(g, 3, tiny, sizeof tiny) == 3 && strlen(tiny) == sizeof tiny - 1);
    CHECK(!strcmp(env("TPD_G3"), "x=0,y=all"));

    printf("%s: %d checks, %d failed\n", fails ? "FAILED" : "PASSED", checks, fails);
    return fails != 0;
}
