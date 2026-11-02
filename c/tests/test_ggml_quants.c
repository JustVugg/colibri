/* Drives ggml_quants.h for tests/test_ggml_quants.py: reads
 *   int32 type, int64 n_elements, then the raw row bytes
 * from stdin and writes n_elements float32 to stdout (gq_dequant_row). */
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include "../ggml_quants.h"

int main(void) {
#ifdef _WIN32   /* the blocks and floats go through stdin/stdout: no CRLF translation */
    _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
#endif
    int32_t t; int64_t n;
    if (fread(&t, 4, 1, stdin) != 1 || fread(&n, 8, 1, stdin) != 1) return 2;
    int64_t rb = gq_row_bytes(t, n);
    if (rb < 0) { fprintf(stderr, "bad type/len %d %lld\n", t, (long long)n); return 3; }
    uint8_t *src = malloc((size_t)rb); float *dst = malloc((size_t)n * 4);
    if (!src || !dst || fread(src, 1, (size_t)rb, stdin) != (size_t)rb) return 4;
    if (gq_dequant_row(t, src, dst, n)) { fprintf(stderr, "unsupported %s\n", gq_type_name(t)); return 5; }
    /* the integer form must reproduce the same floats bit for bit */
    if (gq_has_int(t)) {
        int8_t *w8 = malloc((size_t)n); float *sw = malloc((size_t)(n / GQ_G) * 4), *mw = malloc((size_t)(n / GQ_G) * 4);
        if (!w8 || !sw || !mw || gq_unpack_row(t, src, n, w8, sw, mw)) return 6;
        for (int64_t i = 0; i < n; i++) {
            float v = sw[i / GQ_G] * (float)w8[i] - mw[i / GQ_G];
            uint32_t a, b; memcpy(&a, &v, 4); memcpy(&b, &dst[i], 4);
            if (a != b && !(v != v && dst[i] != dst[i])) {
                fprintf(stderr, "%s: integer form differs at %lld: %.9g vs %.9g (w8 %d sw %.9g mw %.9g)\n",
                        gq_type_name(t), (long long)i, v, dst[i], w8[i], sw[i / GQ_G], mw[i / GQ_G]);
                return 7;
            }
        }
        free(w8); free(sw); free(mw);
    }
    fwrite(dst, 4, (size_t)n, stdout);
    free(src); free(dst);
    return 0;
}
