/* setenv() must be visible to getenv() in the same process.
 *
 * On Windows the two are different objects: SetEnvironmentVariableA writes the
 * Win32 environment block that children inherit, getenv() reads the CRT's own
 * copy, and the shim used to write only the first. A value set by the engine
 * and read back by the engine was therefore the OLD value, with no error
 * anywhere -- the failure mode that made six test files grow their own
 * _putenv_s helper (#1416, #1417, #1420).
 *
 * The assertions below are the POSIX contract, so they hold on every platform
 * and the Windows shim is held to the same one. The empty-value case is the
 * single place the platforms genuinely differ, and it is asserted as what it
 * is rather than papered over.
 */
/* stdio before compat.h: the header uses FILE and stderr without including
 * them, as every engine that pulls it in already has. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"

static int fails = 0;

static void ck(int cond, const char *what) {
    if (cond) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

static const char *NAME = "COLI_COMPAT_ENV_PROBE";

int main(void) {
    printf("compat env shims\n");
    unsetenv(NAME);
    ck(getenv(NAME) == NULL, "unset to start with");

    ck(setenv(NAME, "first", 1) == 0, "setenv reports success");
    const char *seen = getenv(NAME);
    ck(seen != NULL && !strcmp(seen, "first"),
       "getenv sees what setenv just wrote, in this process");

    ck(setenv(NAME, "second", 0) == 0, "setenv without overwrite reports success");
    seen = getenv(NAME);
    ck(seen != NULL && !strcmp(seen, "first"),
       "overwrite 0 leaves the existing value alone");

    ck(setenv(NAME, "second", 1) == 0, "setenv with overwrite reports success");
    seen = getenv(NAME);
    ck(seen != NULL && !strcmp(seen, "second"), "overwrite 1 replaces it");

#ifdef _WIN32
    /* The other view: what a child process would inherit. Both have to carry
     * the value, or the engine's re-exec (omp_tune.h) loses its settings. */
    char block[64];
    DWORD got = GetEnvironmentVariableA(NAME, block, (DWORD)sizeof(block));
    ck(got > 0 && got < sizeof(block) && !strcmp(block, "second"),
       "the Win32 block a child inherits carries it too");
#endif

    ck(unsetenv(NAME) == 0, "unsetenv reports success");
    ck(getenv(NAME) == NULL, "getenv no longer sees it");

    /* Windows has no representation for an empty value: setenv(name, "", 1)
     * removes the variable there. Assert each platform's real behaviour so the
     * difference is documented by a test rather than discovered by a user. */
    setenv(NAME, "", 1);
    seen = getenv(NAME);
#ifdef _WIN32
    ck(seen == NULL, "empty value removes the variable (Windows CRT limitation)");
#else
    ck(seen != NULL && *seen == '\0', "empty value is an empty value");
#endif
    unsetenv(NAME);

    printf("%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
