/* prompt_break.h: a served prompt that a STOP or CANCEL stops while it is read.
 *
 * The gateway sends CANCEL as soon as a client goes away, also while the engine is still
 * reading the prompt (#908). An engine that looks at its input only between two generated
 * tokens reads the whole prompt first: a 40,000-token prompt kept an 8-core CPU busy for
 * about 17 minutes after the client that sent it had given up.
 *
 * An engine points g_coli_prompt_break at one of these around a served prompt's forward and
 * calls coli_prompt_break() between two layers (or two prompt blocks) of its CPU path. The
 * poll reads the commands waiting on stdin with the engine's own reader and says whether this
 * request is to stop; once it has said so the answer stays. The engine then leaves the forward
 * before it moves its context length or its kv_prefix record, and puts its context in a state
 * the next request can start from: its own comment says which. NULL everywhere else, so decode,
 * verifies and Segment never read stdin from inside a forward.
 *
 * control: what stopped it, in the engine's own codes (nonzero). eof: the input closed while
 * the prompt was read; the turn goes on, as the decode loops treat it. ctx: whatever else the
 * engine's poll needs (colibri.c's numeric request id).
 *
 * COLI_PROMPT_BREAK_AFTER=n (tests): stdin is read from the (n+1)-th check on, so a CANCEL
 * written together with its SUBMIT stops the prompt after n layers or blocks, the stop and
 * the restore in the middle of a forward that timing alone would make a race. */
#ifndef COLI_PROMPT_BREAK_H
#define COLI_PROMPT_BREAK_H

#include <stdio.h>
#include <stdlib.h>

typedef struct ColiPromptBreak {
    int (*poll)(struct ColiPromptBreak *b);   /* nonzero: set b->control and stop */
    const char *id;                           /* the request being read */
    int control, eof;
    void *ctx;
    int checks;                               /* how many times it was asked (the test knob) */
} ColiPromptBreak;

static ColiPromptBreak *g_coli_prompt_break;

/* One check of b: reads stdin unless it already stopped, or the test knob holds it back. */
static inline int coli_prompt_poll(ColiPromptBreak *b) {
    static int after = -1;
    if (after < 0) { const char *e = getenv("COLI_PROMPT_BREAK_AFTER"); after = e ? atoi(e) : 0; if (after < 0) after = 0; }
    if (b->control) return 1;
    if (b->eof || b->checks++ < after) return 0;
    b->poll(b);
    return b->control != 0;
}

/* Between two layers or blocks of a forward: the hook's check, 0 when none is set. */
static inline int coli_prompt_break(void) {
    return g_coli_prompt_break && coli_prompt_poll(g_coli_prompt_break);
}

/* COLI_PREFIX_LOG=1: the line the serve tests look for, the same for every engine. held is the
 * prefix the conversation keeps for the next request (0 when the engine starts over). */
static inline void coli_prompt_break_log(int cancelled, int held, int prompt_tokens) {
    if (!getenv("COLI_PREFIX_LOG")) return;
    fprintf(stderr, "[PREFIX] %s while reading the prompt: %d of %d prompt tokens held\n",
            cancelled ? "cancelled" : "stopped", held, prompt_tokens);
    fflush(stderr);
}

/* The rows of a served prompt's segment, for an engine that reads it in several forwards and
 * checks between them (a chain on the device, which is not stopped inside its forward, or a
 * recurrent state copied once per segment): COLI_PROMPT_SEGMENT, else the engine's own. */
static inline int coli_prompt_segment_rows(int engine_default) {
    static int rows = -1;
    if (rows < 0) { const char *e = getenv("COLI_PROMPT_SEGMENT"); rows = e && atoi(e) > 0 ? atoi(e) : 0; }
    return rows ? rows : engine_default;
}

/* After the layers: did the poll stop this forward? It does not read stdin again. */
static inline int coli_prompt_broken(void) {
    return g_coli_prompt_break && g_coli_prompt_break->control != 0;
}

#endif
