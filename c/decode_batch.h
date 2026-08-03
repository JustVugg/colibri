#ifndef COLIBRI_DECODE_BATCH_H
#define COLIBRI_DECODE_BATCH_H

#include <stddef.h>
#include <stdio.h>
#include <math.h>

/* `base` belongs to one sequence's KV state.  Keeping this arithmetic in a
 * model-independent seam makes ragged decode row ownership directly testable. */
static inline float *coli_kv_row(float *base, int position, int width) {
    return base + (size_t)position * (size_t)width;
}

typedef struct {
    unsigned long long id, bytes, gbytes;
    int slot, max_tokens;
    float temperature, top_p;
} ColiSubmit;

/* Parse the textual header. The payload is read separately using `bytes`, so
 * it may contain newlines. Reject trailing fields to keep framing unambiguous.
 * Optional 7th field `gbytes`: length of a per-request grammar (raw GBNF, or a
 * JSON-Schema compiled engine-side) appended to the payload AFTER the prompt
 * bytes. 6-field headers remain valid (gbytes = 0). */
static inline int coli_submit_parse(const char *line, ColiSubmit *s) {
    char tail;
    if (!line || !s) return 0;
    s->gbytes = 0;
    if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %llu %c", &s->id, &s->slot, &s->bytes, &s->max_tokens,
               &s->temperature, &s->top_p, &s->gbytes, &tail) != 7) {
        s->gbytes = 0;
        if (sscanf(line, "SUBMIT %llu %d %llu %d %f %f %c", &s->id, &s->slot, &s->bytes, &s->max_tokens,
                   &s->temperature, &s->top_p, &tail) != 6)
            return 0;
    }
    return s->id > 0 && s->bytes <= (16u << 20) && s->gbytes <= (1u << 20) && s->slot >= 0 && s->max_tokens >= 1 &&
           isfinite(s->temperature) && isfinite(s->top_p) && s->temperature >= 0 && s->temperature <= 2 &&
           s->top_p > 0 && s->top_p <= 1;
}

#endif
