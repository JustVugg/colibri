/* platform_defaults.h: unset and empty variables take the table's values, set ones keep theirs, COLI_PLATFORM_DEFAULTS=0
 * sets nothing, the returned count and list are right, a list longer than its buffer is cut and terminated.
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

    printf("%s: %d checks, %d failed\n", fails ? "FAILED" : "PASSED", checks, fails);
    return fails != 0;
}
