/* The check-and-use guard must live in ONE place, and every sibling that
 * touches tier storage has to go through it (#1564).
 *
 * The reader defect was twelve copies of
 *
 *     if(!G.on) return 0;
 *     pthread_mutex_lock(&G.mx);
 *     ... G.slot ...
 *
 * Fixing qt_is_resident alone would have left eleven of them, and eleven is
 * exactly the number that a test walking the hot path never visits. So the
 * claim under test is not about one function: it is that
 *
 *   - no admission decision is made from G.on any more, anywhere in the file
 *     (G.on is written twice -- set by a successful init, cleared by the
 *     teardown -- and read by NOTHING);
 *   - there is no once-only teardown claim left to re-arm;
 *   - each of the fourteen entry points that can reach G.slot, G_lmh.t or
 *     G.mx enters the lifecycle gate exactly once, and has no `return` between
 *     that entry and its leave;
 *   - qt_shutdown clears G.on INSIDE its exclusive section, not at the top,
 *     which is the whole defect 2.
 *
 * This is a source-shape test on purpose. The behavioural reproductions
 * (test_qwen36_tier_reader_race, test_qwen36_tier_reinit_window) walk the
 * interleavings; this one walks the file, so a fourteenth sibling added later
 * without the gate is a failure here rather than a SIGSEGV in the field.
 *
 * Run from c/ (as the Makefile does); pass a path to override. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int fails;
static void check(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL: %s\n", what);
    fails++;
}

/* ---- read + strip comments ----------------------------------------------
 * Comments quote the old code back at the reader, so the checks below run on
 * code only. Strings are tracked too, so a comment opener inside a printf
 * format cannot swallow the rest of the file. */
static char *slurp_stripped(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)n + 2);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = 0; buf[got + 1] = 0;
    long o = 0; int in_str = 0, in_chr = 0;
    for (long i = 0; i < (long)got; i++) {
        char c = buf[i], nx = (i + 1 < (long)got) ? buf[i + 1] : 0;
        if (in_str) { buf[o++] = c; if (c == '\\') { buf[o++] = nx; i++; } else if (c == '"') in_str = 0; continue; }
        if (in_chr) { buf[o++] = c; if (c == '\\') { buf[o++] = nx; i++; } else if (c == '\'') in_chr = 0; continue; }
        if (c == '"')  { in_str = 1;  buf[o++] = c; continue; }
        if (c == '\'') { in_chr = 1;  buf[o++] = c; continue; }
        if (c == '/' && nx == '*') {
            /* Collapse to a space but KEEP THE NEWLINES: body_of() finds
             * definitions by their position at the start of a line, and a block
             * comment that swallowed its own line breaks would join the code
             * before it to the code after it. */
            int nls = 0;
            i += 2;
            while (i < (long)got && !(buf[i] == '*' && i + 1 < (long)got && buf[i + 1] == '/')) {
                if (buf[i] == '\n') nls++;
                i++;
            }
            buf[o++] = ' ';
            while (nls-- > 0) buf[o++] = '\n';
            i++;
            continue;
        }
        if (c == '/' && nx == '/') {
            while (i < (long)got && buf[i] != '\n') i++;
            buf[o++] = '\n';
            continue;
        }
        buf[o++] = c;
    }
    buf[o] = 0;
    if (out_len) *out_len = o;
    return buf;
}

static int count_of(const char *hay, const char *needle) {
    int n = 0; size_t l = strlen(needle);
    for (const char *p = hay; (p = strstr(p, needle)) != NULL; p += l) n++;
    return n;
}

/* ---- the body of a definition, by brace matching -------------------------
 * Found by the name followed by `(`, then whichever comes first: the `{` that
 * opens the body, or the `;` that ends a call. A definition sits behind its
 * return type (`int qt_is_resident(...)`), so "at the start of a line" is not
 * the rule -- and a call site is rejected by the `;` rather than by luck. */
static char *body_of(const char *src, const char *name) {
    size_t nl = strlen(name);
    for (const char *p = src; (p = strstr(p, name)) != NULL; p += nl) {
        char prev = (p > src) ? p[-1] : ' ';
        if (p[nl] != '(') continue;
        if (!(prev == ' ' || prev == '\t' || prev == '\n' || prev == '*')) continue;
        for (const char *q = p; *q; q++) {
            if (*q == ';') break;                       /* a call, not a definition */
            if (*q == '{') {
                int d = 0;
                for (const char *r = q; *r; r++) {
                    if (*r == '{') d++;
                    else if (*r == '}' && --d == 0) {
                        size_t len = (size_t)(r - q) + 1;
                        char *b = (char *)malloc(len + 1);
                        if (!b) return NULL;
                        memcpy(b, q, len); b[len] = 0;
                        return b;
                    }
                }
                return NULL;
            }
        }
    }
    return NULL;
}

/* Compare by the literal's own length: hand-counting strlen() of a 13-character
 * identifier is how this check silently stops matching anything. */
static int is_tok(const char *w, size_t n, const char *s) {
    return n == strlen(s) && !strncmp(w, s, n);
}

/* Every `return` in the body must happen with the gate NOT held. The refusal
 * path -- `if(!qt_gate_enter()) return ...;` -- is the one return that legally
 * happens inside the entry expression, so `pending` distinguishes it from a
 * return that would skip a leave. */
static int returns_only_outside_the_gate(const char *body, int *enters) {
    int held = 0, pending = 0, bad = 0, ne = 0;
    for (const char *p = body; *p; ) {
        if (isalpha((unsigned char)*p) || *p == '_') {
            const char *w = p;
            while (isalnum((unsigned char)*p) || *p == '_') p++;
            size_t n = (size_t)(p - w);
            if (is_tok(w, n, "return")) {
                if (pending) pending = 0;              /* refused at the door: legal */
                else if (held) bad = 1;                /* would skip a leave */
            } else if (is_tok(w, n, "qt_gate_enter")) { held++; pending = 1; ne++; }
            else if (is_tok(w, n, "qt_gate_leave")) { if (held) held--; }
            continue;
        }
        if (*p == ';') pending = 0;
        p++;
    }
    if (enters) *enters = ne;
    return !bad && held == 0;
}

/* ---- the entry points that can reach storage the teardown frees ---------- */
static const char *SITES[] = {
    "qt_lmhead_init", "qt_lmhead_matmul",
    "qt_is_resident", "qt_resident_count", "qt_resident_bytes",
    "qt_note", "qt_note_block", "qt_note_planned",
    "qt_fill_next", "qt_plan_fill", "qt_fill_wait",
    "qt_issue", "qt_take", "qt_stats",
    NULL
};

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "qwen36_tier.c";
    long len = 0;
    char *src = slurp_stripped(path, &len);
    if (!src) { printf("  FAIL: cannot read %s\n", path); return 1; }
    printf("qwen36 tier gate coverage: one check-and-use guard, every sibling\n");

    /* ---- no admission decision may come from G.on any more ---------------- */
    check(count_of(src, "G.on") == 2,
          "G.on is written twice (a successful init, and the teardown) and read nowhere");
    check(count_of(src, "int teardown;") == 0 &&
          count_of(src, "G.teardown") == 0 &&
          count_of(src, "__atomic_exchange_n(&G.teardown") == 0,
          "no once-only teardown claim is left to re-arm");
    check(count_of(src, "if(G.on)") == 0,
          "no `if(G.on)` guard survives: it is blind to the teardown window");
    check(count_of(src, "if(!G.on)") == 0,
          "no `if(!G.on)` guard survives: it is read with nothing held");

    /* ---- the gate itself is a state machine, not a latch ------------------ */
    check(count_of(src, "QT_TEARING_DOWN") >= 3,
          "the lifecycle has a TEARING_DOWN state that something reads and moves");
    check(count_of(src, "qt_gate_enter(void)") == 1 &&
          count_of(src, "qt_gate_xenter(void)") == 1,
          "there is exactly one shared entry helper and one exclusive entry helper");

    /* ---- every sibling goes through it ------------------------------------ */
    for (int i = 0; SITES[i]; i++) {
        char *b = body_of(src, SITES[i]);
        if (!b) { printf("  FAIL: %s: no definition found in %s\n", SITES[i], path); fails++; continue; }
        int enters = 0;
        int ok = returns_only_outside_the_gate(b, &enters);
        int gated = enters >= 1;
        int reads_flag = strstr(b, "G.on") != NULL;
        char msg[256];
        snprintf(msg, sizeof msg, "%s enters the gate (%d) and never returns while holding it",
                 SITES[i], enters);
        check(gated && ok && !reads_flag, msg);
        if (enters != 1)
            printf("       note: %s has %d gate entries\n", SITES[i], enters);
        free(b);
    }

    /* ---- defect 2's mechanism, as an ordering in the source -------------- */
    char *sd = body_of(src, "qt_shutdown");
    if (!sd) { printf("  FAIL: qt_shutdown definition not found\n"); fails++; }
    else {
        char *x = strstr(sd, "qt_gate_xenter()");
        char *off = strstr(sd, "G.on=0");
        char *td = strstr(sd, "QT_TEARING_DOWN");
        check(x && off && x < off,
              "qt_shutdown clears G.on INSIDE the exclusive section, not at the top");
        check(td && x && td < x,
              "qt_shutdown moves to TEARING_DOWN before it waits for the callers inside");
        check(strstr(sd, "while(qg_get()==QT_TEARING_DOWN) pthread_cond_wait") != NULL,
              "a losing qt_shutdown WAITS for the teardown instead of returning");
        free(sd);
    }

    free(src);
    if (fails) { printf("test_qwen36_tier_gate_coverage: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_gate_coverage: ok (14 sites, one guard, no latch)\n");
    return 0;
}
