/* qpack_parse_layout when one of its tables cannot be allocated: the parse
 * must fail with "out of memory parsing layout", never crash. Its cleanup used
 * to walk a NULL sections table (qpack_free_layout read layout->sections[i] for
 * every counted section), a SEGV whenever that calloc was the one that failed.
 *
 * The failure is targeted by the exact (count, size) of the table, not by call
 * ordinal: json.h callocs too, and the number of those calls follows the input. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t fail_count, fail_size;
static int failed;

static void *oom_calloc(size_t n, size_t size)
{
    if (!failed && n == fail_count && size == fail_size) {
        failed = 1;
        return NULL;
    }
    return calloc(n, size);
}

#define calloc oom_calloc
#include "../qpack.c"
#undef calloc

/* three sections, five layers: neither table's (count, size) matches a json.h
 * node or the other table */
static const char layout_json[] =
    "{\"expertCount\":1,\"layerCount\":5,\"expertStride\":16384,"
    "\"sections\":[{\"name\":\"w\",\"dtype\":\"U8\",\"shape\":[4],\"offset\":0,\"size\":4},"
    "{\"name\":\"s\",\"dtype\":\"F16\",\"shape\":[2],\"offset\":4,\"size\":4},"
    "{\"name\":\"b\",\"dtype\":\"F32\",\"shape\":[1],\"offset\":8,\"size\":4}],"
    "\"linearLayers\":[false,true,false,true,false]}";

static int parse_failing(size_t count, size_t size, const char *what)
{
    char error[256] = "";
    ColiQpackLayout layout;
    memset(&layout, 0, sizeof(layout));
    fail_count = count;
    fail_size = size;
    failed = 0;
    int result = qpack_parse_layout(&layout, layout_json, error, sizeof(error));
    if (!failed) {
        fprintf(stderr, "%s: the targeted calloc was never made\n", what);
        return 1;
    }
    if (result == 0 || strcmp(error, "out of memory parsing layout") != 0) {
        fprintf(stderr, "%s: result %d, error '%s'\n", what, result, error);
        return 1;
    }
    if (layout.sections || layout.linear_layers || layout.section_count) {
        fprintf(stderr, "%s: the layout was left half built\n", what);
        return 1;
    }
    return 0;
}

int main(void)
{
    char error[256];
    ColiQpackLayout layout;
    memset(&layout, 0, sizeof(layout));
    fail_count = 0;
    failed = 1;
    if (qpack_parse_layout(&layout, layout_json, error, sizeof(error)) != 0) {
        fprintf(stderr, "the layout does not parse: %s\n", error);
        return 1;
    }
    qpack_free_layout(&layout);

    if (parse_failing(3, sizeof(ColiQpackSection), "sections table") ||
        parse_failing(5, 1, "linearLayers table"))
        return 1;
    puts("qpack layout allocation failures: ok");
    return 0;
}
