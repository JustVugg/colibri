#ifndef COLI_SERVE_BUDGET_H
#define COLI_SERVE_BUDGET_H

/* max_tokens is a ceiling, not a target (#260/#382/#1641).
 * Returns the tokens the request may generate, or -1 when the PROMPT does
 * not fit. Generation needs one free position; a read-only logprobs request
 * may fill the context exactly. Refusing when prompt + budget exceeded the
 * context turned coli chat's interactive default (16384) into a 400 on every
 * Kimi/Inkling message against the default 8192-token window. */

static inline int coli_serve_budget(int prompt, int requested, int context,
                                    int read_only)
{
    if (prompt < 1) return -1;
    int room = context - prompt;
    if (read_only) return room < 0 ? -1 : (requested < room ? requested : room);
    if (room < 1) return -1;
    return requested > room ? room : requested;
}

#endif
