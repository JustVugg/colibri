/* max_tokens is a ceiling. The remaining serve engines (kimi, inkling, olmoe)
 * used to refuse when prompt + budget exceeded the context, so coli chat's
 * interactive default (16384) 400'd every Kimi/Inkling turn at the default
 * 8192-token window. Only a prompt that does not fit is refused. */
#include "../serve_budget.h"
#include <limits.h>
#include <stdio.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return 1; \
} } while (0)

int main(void) {
    CHECK(coli_serve_budget(2, 16384, 8192, 0) == 8190);
    CHECK(coli_serve_budget(2, 8192, 8192, 0) == 8190);
    CHECK(coli_serve_budget(100, 50, 8192, 0) == 50);
    CHECK(coli_serve_budget(8191, 1, 8192, 0) == 1);
    CHECK(coli_serve_budget(8192, 1, 8192, 0) == -1);
    CHECK(coli_serve_budget(9000, 1, 8192, 0) == -1);
    CHECK(coli_serve_budget(0, 1, 8192, 0) == -1);
    CHECK(coli_serve_budget(8192, 0, 8192, 1) == 0);
    CHECK(coli_serve_budget(8193, 0, 8192, 1) == -1);
    CHECK(coli_serve_budget(20, 0, 8192, 1) == 0);
    CHECK(coli_serve_budget(8191, 100, 8192, 1) == 1);
    CHECK(coli_serve_budget(20, INT_MAX, 8192, 0) == 8172);
    CHECK(coli_serve_budget(3500, 1024, 4096, 0) == 596);
    puts("serve budget ceiling: ok");
    return 0;
}
