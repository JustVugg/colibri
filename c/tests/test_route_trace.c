/* route_trace.h: the history format, and the property the whole design rests on —
 * that a reader written against the OLD format still recovers every record from a file
 * written by the new one.
 *
 * Case 7 is the important one. The header records are encoded as triples with a negative
 * layer precisely so that the pre-existing readers (telemetry.h's usage_load and
 * colibri.c's pin_load, both `while(fscanf(f,"%d %d %u",&l,&e,&c)==3)` guarded by l>=0)
 * parse them, discard them, and keep going. That contract is invisible in the code and
 * easy to break by "tidying" the header into a comment line or moving the engine hash
 * into the second field, so it is asserted here against a literal copy of that loop.
 *
 * This test includes route_trace.h and nothing else: no Model, no Cfg, no st.h. That is
 * the point of the header — any engine can use it — and compiling this file proves it. */
#include "../route_trace.h"

static int g_nfails = 0;
static void check(int cond, const char *what){
    if(!cond){ printf("FAIL: %s\n", what); g_nfails++; }
}

#define TMP "test_route_trace.tmp"

static long fsize(const char *p){
    FILE *f = fopen(p, "rb");
    if(!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}
static void zero_counts(void){
    for(int l = 0; l <= rt_nl; l++)
        if(rt_counts(l)) memset(rt_counts(l), 0, (size_t)rt_ne * sizeof(uint32_t));
}
static int64_t g_sum; static int g_recs;
/* accepts everything, to observe what the reader hands out before any admission rule */
static int sum_cb(int l, int e, uint32_t c, void *ud){
    (void)l; (void)e; (void)ud; g_sum += c; g_recs++; return 1;
}

int main(void){
    rt_init("glm_moe_dsa", 5, 8);                /* 5 layers + the MTP row, 8 experts */
    check(rt_counts(0) != NULL, "rt_init allocates row 0");
    check(rt_counts(5) != NULL, "rt_init allocates the inclusive last row");
    check(rt_counts(6) == NULL, "rt_init does not allocate past n_layers");
    check(rt_counts(-1) == NULL, "a negative layer has no row");

    /* 1. an all-zero history must stay a ZERO-BYTE file: PIN=auto decides whether a
     * history is usable by testing the file size, and falls back to stats.txt when it is
     * empty. Two header lines in an empty history would silently cost that fallback. */
    remove(TMP);
    check(rt_save(TMP, 1) == 1, "empty history writes");
    check(fsize(TMP) == 0, "empty history is a zero-byte file");

    /* 2. with counts, the header appears and the data section is the sparse triples */
    int ids[3] = {2, 5, 7};
    rt_count(3, ids, 3);
    rt_count(3, ids, 1);                          /* expert 2 twice */
    check(rt_save(TMP, 1) == 1, "history with counts writes");
    {
        char buf[512] = {0};
        /* text mode on purpose: the writer uses fopen("w"), so on Windows the records are
         * CRLF-terminated (as they have always been — stats_dump_q wrote them the same
         * way). Reading back in text mode collapses that, which keeps these assertions
         * about the RECORDS rather than about the platform's line terminator. */
        FILE *f = fopen(TMP, "r");
        size_t n = f ? fread(buf, 1, sizeof(buf) - 1, f) : 0;
        if(f) fclose(f);
        check(n > 0, "history is non-empty");
        check(strstr(buf, "-1 5 8\n") == buf, "first record is the dimensions");
        check(strstr(buf, "\n-2 1 ") != NULL, "second record is version then engine id");
        check(strstr(buf, "\n3 2 2\n") != NULL, "expert 2 counted twice");
        check(strstr(buf, "\n3 5 1\n") != NULL, "expert 5 counted once");
        check(strstr(buf, "\n3 7 1\n") != NULL, "expert 7 counted once");
    }

    /* 3. round trip through the reader */
    zero_counts();
    check(rt_load(TMP) == 4, "round trip recovers every selection");
    check(rt_counts(3)[2] == 2, "round trip recovers the per-expert count");

    /* 4. a legacy file — same triples, no header — is accepted as before */
    {
        FILE *f = fopen(TMP, "w");
        fprintf(f, "3 2 2\n3 5 1\n3 7 1\n");
        fclose(f);
        zero_counts();
        check(rt_load(TMP) == 4, "legacy text without a header still loads");
    }

    /* 4b. a history written on Windows arrives with CRLF endings. The reader opens "rb"
     * (the magic sniff needs raw bytes), so the \r survives into fscanf — which skips it
     * as whitespace. Asserted rather than assumed, and this runs on the MinGW CI job too. */
    {
        FILE *f = fopen(TMP, "wb");
        fprintf(f, "-1 5 8\r\n-2 1 %u\r\n3 2 2\r\n3 5 1\r\n", rt_hash("glm_moe_dsa"));
        fclose(f);
        zero_counts();
        check(rt_load(TMP) == 3, "a CRLF history loads (Windows-written file)");
        check(rt_counts(3)[2] == 2, "CRLF: per-expert count intact");
    }

    /* The cases below exercise the refusal paths, so they print eight [USAGE] lines to stderr
     * on a PASSING run: seven refusals (5, 6, 7, two in 10, two in 10b) and one three-line
     * override notice in 10. Say so, or a CI log looks like something went wrong. */
    printf("route_trace: the [USAGE] refusal and override lines below are expected\n");

    /* 5. a file that identifies itself with the wrong dimensions is refused, not misread */
    {
        FILE *f = fopen(TMP, "w");
        fprintf(f, "-1 99 512\n3 2 2\n");
        fclose(f);
        g_sum = 0; g_recs = 0;
        check(rt_read(TMP, sum_cb, NULL) == -1, "wrong dimensions are refused");
        check(g_recs == 0, "a refused file yields no records");
    }

    /* 6. a file from another engine is refused, and the message can name it */
    {
        FILE *f = fopen(TMP, "w");
        fprintf(f, "-1 5 8\n-2 1 %u\n3 2 2\n", rt_hash("kimi_k3"));
        fclose(f);
        check(rt_read(TMP, sum_cb, NULL) == -1, "another engine's history is refused");
        check(rt_engine_of(rt_hash("kimi_k3")) != NULL, "the writer can be named");
        check(strcmp(rt_engine_of(rt_hash("kimi_k3")), "kimi_k3") == 0, "named correctly");
        check(rt_engine_of(12345u) == NULL, "an unknown id has no name");
        /* deepseek_v4 writes through this same header (its private writer is gone,
         * #700 completed), so its id must resolve to a name like every sibling's */
        check(strcmp(rt_engine_of(rt_hash("deepseek_v4")), "deepseek_v4") == 0,
              "the deepseek_v4 writer can be named");
    }

    /* 7. inkling's IKU1 layout is refused by any engine that is not inkling */
    {
        FILE *f = fopen(TMP, "wb");
        uint32_t hdr[3] = { RT_IKU1_MAGIC, 5, 8 };
        uint32_t body[5 * 8];
        memset(body, 0, sizeof(body));
        fwrite(hdr, 4, 3, f);
        fwrite(body, 4, 5 * 8, f);
        fclose(f);
        check(rt_read(TMP, sum_cb, NULL) == -1, "IKU1 is refused by a non-inkling engine");
    }

    /* 7b. …and inkling itself still reads it. Same reader, engine identity flipped: without
     * this the binary path is only ever exercised on its refusal, so a broken IKU1 decode
     * would look like a pass. Same translation unit, so the identity is directly settable. */
    {
        const char *save_engine = rt_engine;
        uint32_t save_id = rt_id;
        rt_engine = "inkling"; rt_id = rt_hash("inkling");

        FILE *f = fopen(TMP, "wb");
        uint32_t hdr[3] = { RT_IKU1_MAGIC, 5, 8 };      /* must match rt_nl / rt_ne */
        uint32_t body[5 * 8];
        memset(body, 0, sizeof(body));
        body[3 * 8 + 2] = 7;                            /* layer 3, expert 2 -> 7 */
        body[1 * 8 + 5] = 4;                            /* layer 1, expert 5 -> 4 */
        fwrite(hdr, 4, 3, f);
        fwrite(body, 4, 5 * 8, f);
        fclose(f);

        zero_counts();
        check(rt_load(TMP) == 11, "inkling accepts its own IKU1 layout");
        check(rt_counts(3)[2] == 7, "IKU1 decode places layer 3 / expert 2");
        check(rt_counts(1)[5] == 4, "IKU1 decode places layer 1 / expert 5");
        check(rt_counts(0)[0] == 0, "IKU1 decode leaves the zeros alone");

        rt_engine = save_engine; rt_id = save_id;
    }

    /* 8. THE CONTRACT: a reader written against the old format must recover exactly the
     * same records from a new-format file. This is a literal copy of the loop in
     * telemetry.h's usage_load and colibri.c's pin_load. */
    {
        zero_counts();
        rt_count(3, ids, 3);
        rt_count(3, ids, 1);
        rt_count(5, ids, 2);
        check(rt_save(TMP, 1) == 1, "contract fixture writes");

        int64_t legacy_tot = 0; int legacy_recs = 0;
        FILE *f = fopen(TMP, "r");                /* exactly as the old readers open it */
        int l, e; uint32_t cnt;
        while(f && fscanf(f, "%d %d %u", &l, &e, &cnt) == 3)
            if(l >= 0 && l <= 5 && e >= 0 && e < 8){ legacy_tot += cnt; legacy_recs++; }
        if(f) fclose(f);

        g_sum = 0; g_recs = 0;
        rt_read(TMP, sum_cb, NULL);
        check(legacy_tot == g_sum,   "an old reader recovers the same total");
        check(legacy_recs == g_recs, "an old reader recovers the same record count");
        check(legacy_tot == 6,       "and that total is the one that was written");
    }

    /* 9. the total counts ACCEPTED records only. Both readers this replaces added to their
     * total inside the same `if` that admitted the record, so a history holding records
     * outside this model's shape reported the sum of the usable ones. A legacy headerless
     * history from a larger model is exactly that, and it carries no dimension record to be
     * refused by — so getting this wrong inflates the count the user is shown. */
    {
        FILE *f = fopen(TMP, "w");
        fprintf(f, "3 2 5\n9 1 100\n1 99 200\n");   /* valid, layer OOR, expert OOR */
        fclose(f);
        zero_counts();
        check(rt_load(TMP) == 5, "out-of-range records are not counted in the total");
        check(rt_counts(3)[2] == 5, "the in-range record still lands");
        check(rt_counts(1)[7] == 0, "an out-of-range expert wrote nothing");
    }

    /* 10. THE ESCAPE HATCH. Every refusal message says "pass PIN=<path> to use it anyway",
     * and pin_load reads through here, so a trusted read must not re-apply the check the
     * message is telling the user to get around — otherwise that advice cannot work. The
     * old pin_load had no identity or dimension check at all, so this also keeps it. */
    {
        FILE *f = fopen(TMP, "w");
        fprintf(f, "-1 5 8\n-2 1 %u\n3 2 6\n", rt_hash("kimi_k3"));
        fclose(f);
        check(rt_read(TMP, sum_cb, NULL) == -1, "an untrusted read refuses another engine");
        g_sum = 0; g_recs = 0;
        check(rt_read_ex(TMP, sum_cb, NULL, 1) == 6, "a trusted read takes it anyway");
        check(g_recs == 1, "trusted: the data record reached the callback");
        /* the override announces itself once per process (#700), which must not turn a
         * repeated trusted read into a different answer — the guard is on the message only */
        g_sum = 0; g_recs = 0;
        check(rt_read_ex(TMP, sum_cb, NULL, 1) == 6, "a second trusted read still honours it");
        check(g_recs == 1, "the once-only notice does not gate the read");

        /* but only identity bends to trust. Wrong dimensions are a mistake rather than a
         * preference, and that refusal never offered PIN= as the way past it. */
        f = fopen(TMP, "w");
        fprintf(f, "-1 99 512\n-2 1 %u\n3 2 6\n", rt_hash("glm_moe_dsa"));
        fclose(f);
        g_sum = 0; g_recs = 0;
        check(rt_read_ex(TMP, sum_cb, NULL, 1) == -1, "trust does not excuse wrong dimensions");
        check(g_recs == 0, "no record from a wrong-dimension file reached the callback");
    }

    /* 10b. …but trust does not extend to the parse geometry. A format this build cannot read
     * is refused even on a trusted read: trust says whose history the user accepts, not that
     * the bytes can be cut up correctly, and a future layout read as v1 is misread silently.
     * Nothing to preserve here either — the refusal for a version never promised PIN=. */
    {
        FILE *f = fopen(TMP, "w");
        fprintf(f, "-1 5 8\n-2 %d %u\n3 2 9\n", RT_FORMAT_VERSION + 1, rt_hash("glm_moe_dsa"));
        fclose(f);
        g_sum = 0; g_recs = 0;
        check(rt_read(TMP, sum_cb, NULL) == -1, "a future format version is refused");
        check(rt_read_ex(TMP, sum_cb, NULL, 1) == -1, "and a trusted read cannot override it");
        check(g_recs == 0, "no record from a future version reached the callback");
    }

    /* 11. every record the writer emits has EXACTLY three fields. Not "at least three": a
     * fourth field leaves one value unconsumed, which desynchronises every old reader and
     * makes it fabricate a record out of that leftover plus the start of the next line. The
     * l>=0 guard does not save anyone, because the invented record has a positive layer — so
     * the count lands on a layer that was never routed and nothing reports it. The writer is
     * the only place this can be enforced, so it is asserted here. */
    {
        zero_counts();
        rt_count(3, ids, 3);
        rt_count(1, ids, 2);
        check(rt_save(TMP, 1) == 1, "three-field fixture writes");
        FILE *f = fopen(TMP, "r");
        char line[256]; int bad = 0, lines = 0;
        while(f && fgets(line, sizeof(line), f)){
            int n = 0; const char *p = line;
            for(;;){
                while(*p == ' ' || *p == '\t') p++;
                if(*p == '\0' || *p == '\n' || *p == '\r') break;
                n++;
                while(*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
            }
            if(n){ lines++; if(n != 3) bad++; }
        }
        if(f) fclose(f);
        check(lines > 0, "the fixture produced records");
        check(bad == 0, "every record the writer emits has exactly three fields");
    }

    /* 12. a dropped row is the load admission rule, not an optimisation: a layer that does
     * not route must not absorb a count. This is what the replaced code said by leaving
     * eusage[i] NULL on dense layers, and it runs last because the drop is permanent. */
    {
        rt_drop_row(2);
        check(rt_counts(2) == NULL, "rt_drop_row clears the row");
        FILE *f = fopen(TMP, "w");
        fprintf(f, "2 3 50\n3 4 7\n");
        fclose(f);
        zero_counts();
        check(rt_load(TMP) == 7, "a record for a non-routing layer is not counted");
        check(rt_counts(3)[4] == 7, "the routing layer's record still lands");
        check(rt_save(TMP, 1) == 1, "saving after a drop works");
        {
            char buf[512] = {0};
            FILE *g = fopen(TMP, "r");
            size_t n = g ? fread(buf, 1, sizeof(buf) - 1, g) : 0;
            if(g) fclose(g);
            check(n > 0, "post-drop history is non-empty");
            check(strstr(buf, "\n2 3 ") == NULL, "the dropped layer is not written back");
        }
    }

    /* 8. USAGE_SAVE=0 is a read-only run (#1039): rt_save reports success but the
     * file must not be touched — a benchmark loop relies on the profile it measures
     * staying frozen. Enforced in rt_save itself so every engine gets it. */
    {
        rt_counts(3)[4] = 99;
#ifdef _WIN32
        _putenv_s("USAGE_SAVE", "0");
#else
        setenv("USAGE_SAVE", "0", 1);
#endif
        long before = fsize(TMP);
        check(rt_save(TMP, 1) == 1, "USAGE_SAVE=0: a requested skip is not a failure");
        check(fsize(TMP) == before, "USAGE_SAVE=0: the history file is not rewritten");
#ifdef _WIN32
        _putenv_s("USAGE_SAVE", "1");
#else
        setenv("USAGE_SAVE", "1", 1);
#endif
        check(rt_save(TMP, 1) == 1, "USAGE_SAVE=1: saving works again");
        check(fsize(TMP) != before, "USAGE_SAVE=1: the new counter reaches the file");
#ifdef _WIN32
        _putenv_s("USAGE_SAVE", "");
#else
        unsetenv("USAGE_SAVE");
#endif
    }

    remove(TMP);
    if(g_nfails){ printf("route_trace: %d FAILED\n", g_nfails); return 1; }
    printf("route_trace: empty-history size, round trip, legacy read, refusals, "
           "admitted-only totals, trusted read, dropped rows, old-reader contract, "
           "USAGE_SAVE=0 read-only ok\n");
    return 0;
}
