/* The serve contract accepts a fitting prompt even if the requested output
 * budget is too large, and never turns a read-only request into generation. */
#define main dsv41_main_unused
#include "../deepseek_v41.c"
#undef main
#include <limits.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return 1; \
} } while (0)

int main(void) {
    CHECK(serve_budget(20, 5000, 4096, 0) == 4076);
    CHECK(serve_budget(20, 10, 4096, 0) == 10);
    CHECK(serve_budget(4095, 5000, 4096, 0) == 1);
    CHECK(serve_budget(4096, 1, 4096, 0) == -1);
    CHECK(serve_budget(4097, 1, 4096, 0) == -1);
    CHECK(serve_budget(0, 1, 4096, 0) == -1);
    CHECK(serve_budget(20, 0, 4096, 0) == 256);
    CHECK(serve_budget(4095, 0, 4096, 0) == 1);
    CHECK(serve_budget(4096, 0, 4096, 1) == 0);
    CHECK(serve_budget(4097, 0, 4096, 1) == -1);
    CHECK(serve_budget(20, 0, 4096, 1) == 0);
    CHECK(serve_budget(4095, 100, 4096, 1) == 1);
    CHECK(serve_budget(20, INT_MAX, 4096, 0) == 4076);
    puts("dsv41 serve budget: ok");
    return 0;
}
