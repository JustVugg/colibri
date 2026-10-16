/* test_draft_vocab -- the token list of a draft head: which tokens are in it,
 * how counts halve, what a request and a pick change, and the counts file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include "../draft_vocab.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

enum { VOCAB = 200 };

static int listed(const DraftVocab *v, int id) { return v->in_list[id]; }

/* a prompt with `n` copies of `tok` appended at *len */
static void append_copies(int *prompt, int *len, int tok, int n) {
    while (n-- > 0) prompt[(*len)++] = tok;
}

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    fputs(text, f);
    fclose(f);
}

/* 1 when the file contains exactly `text` */
static int file_is(const char *path, const char *text) {
    char got[256] = "";
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(got, 1, sizeof got - 1, f);
    fclose(f);
    got[n] = 0;
    return !strcmp(got, text);
}

static int list_is_sorted(const DraftVocab *v, int n) {
    for (int i = 1; i < n; i++) if (v->list[i - 1] >= v->list[i]) return 0;
    return 1;
}

/* the rule written out for a state: sort by count (higher first, lower id first), take
 * `top`, add the 3/4 rule over counts above 0, add the conversation */
static void reference_list(const DraftVocab *v, uint8_t *want) {
    int *ids = malloc((size_t)v->vocab * sizeof *ids);
    for (int i = 0; i < v->vocab; i++) ids[i] = i;
    for (int i = 1; i < v->vocab; i++) {            /* insertion sort: slow, plainly right */
        int id = ids[i], j = i;
        while (j > 0 && (v->count[ids[j - 1]] < v->count[id])) { ids[j] = ids[j - 1]; j--; }
        ids[j] = id;
    }
    memset(want, 0, (size_t)v->vocab);
    for (int i = 0; i < v->top; i++) want[ids[i]] = 1;
    double nth = v->count[ids[v->top - 1]];
    for (int i = 0; i < v->vocab; i++)
        if (v->count[i] > 0 && draft_vocab_near_top(v->count[i], nth)) want[i] = 1;
    for (int i = 0; i < v->vocab; i++) if (v->in_conversation[i]) want[i] = 1;
    free(ids);
}

static void test_fresh_list_is_the_lowest_ids(void) {
    DraftVocab *v = draft_vocab_new(VOCAB, 5);
    CHECK(v->list_n == 5, "a new list has %d ids, want 5", v->list_n);
    for (int id = 0; id < VOCAB; id++)
        CHECK(listed(v, id) == (id < 5), "id %d: listed %d on a new list", id, listed(v, id));
    draft_vocab_free(v);
}

static void test_fewer_counted_tokens_than_top(void) {
    DraftVocab *v = draft_vocab_new(VOCAB, 4);
    int prompt[8], len = 0;
    append_copies(prompt, &len, 150, 3);
    append_copies(prompt, &len, 160, 2);
    draft_vocab_begin(v, prompt, len, 0);
    int want[] = {0, 1, 150, 160};   /* the two counted tokens, the lowest ids for the rest */
    CHECK(v->list_n == 4, "list_n %d, want 4", v->list_n);
    for (int i = 0; i < 4; i++) CHECK(listed(v, want[i]), "id %d should be listed", want[i]);
    draft_vocab_free(v);
}

static void test_close_counts_and_conversation(void) {
    DraftVocab *v = draft_vocab_new(VOCAB, 6);
    int prompt[200], len = 0;
    int counts[][2] = {{50, 20}, {51, 19}, {52, 18}, {53, 17}, {54, 16}, {55, 14}, {56, 11}, {57, 8}, {58, 8}};
    for (int i = 0; i < 9; i++) append_copies(prompt, &len, counts[i][0], counts[i][1]);
    draft_vocab_begin(v, prompt, len, 0);
    for (int id = 50; id <= 55; id++) CHECK(listed(v, id), "id %d is among the top 6", id);
    CHECK(listed(v, 56), "11 is at least 3/4 of 14 (10.5)");
    CHECK(listed(v, 57) && listed(v, 58), "both are in the conversation of the first request");
    CHECK(list_is_sorted(v, v->list_n), "a rebuilt list is ascending");

    /* a second request whose prompt is the token 57 only, read from a cache */
    int next[] = {57};
    draft_vocab_begin(v, next, 1, 1);
    CHECK(v->count[57] == 8, "a restored token is not counted again (%g)", v->count[57]);
    CHECK(listed(v, 57), "57 is in the conversation");
    CHECK(!listed(v, 58), "58 is below 3/4 of 14 and not in this conversation");
    CHECK(listed(v, 56), "56 is listed through the 3/4 rule");
    draft_vocab_free(v);
}

static void test_restored_tokens_are_not_counted(void) {
    DraftVocab *v = draft_vocab_new(VOCAB, 3);
    int prompt[] = {120, 121, 122};
    draft_vocab_begin(v, prompt, 3, 2);
    CHECK(v->count[120] == 0 && v->count[121] == 0, "restored tokens counted");
    CHECK(v->count[122] == 1, "a read token has count %g, want 1", v->count[122]);
    CHECK(listed(v, 120) && listed(v, 121) && listed(v, 122), "the whole prompt is the conversation");
    draft_vocab_free(v);
}

static void test_pick_appends_and_counts(void) {
    DraftVocab *v = draft_vocab_new(VOCAB, 3);
    int prompt[] = {7};
    draft_vocab_begin(v, prompt, 1, 0);
    CHECK(!listed(v, 150), "150 starts outside the list");
    int before = v->list_n;
    draft_vocab_count_token(v, 150);
    CHECK(listed(v, 150) && v->list_n == before + 1 && v->list[v->list_n - 1] == 150,
          "a picked token joins at the end at once");
    CHECK(v->count[150] == 1, "a pick counts");
    draft_vocab_count_token(v, 7);
    CHECK(v->run.picks == 2 && v->run.picks_inside == 1 && v->run.appended == 1,
          "picks %llu inside %llu appended %llu, want 2 1 1",
          (unsigned long long)v->run.picks, (unsigned long long)v->run.picks_inside,
          (unsigned long long)v->run.appended);
    DraftVocabSummary summary = draft_vocab_summary(v);
    CHECK(summary.average_list_tokens == before + 0.5 && summary.inside_percent == 50 &&
          summary.appended == 1 && summary.rebuilds == 0,
          "summary %g %g %llu %llu, want %g 50 1 0", summary.average_list_tokens, summary.inside_percent,
          (unsigned long long)summary.appended, (unsigned long long)summary.rebuilds, before + 0.5);
    draft_vocab_begin(v, prompt, 1, 1);
    CHECK(!v->in_conversation[150], "a new request starts its conversation again");
    CHECK(v->run.picks == 0, "a new request starts its report again");
    summary = draft_vocab_summary(v);
    CHECK(summary.average_list_tokens == 0 && summary.inside_percent == 0, "a request without picks reports zeros");
    draft_vocab_free(v);
}

static void test_counts_halve_and_old_topics_fade(void) {
    DraftVocab *v = draft_vocab_new(VOCAB, 1);
    int prompt[] = {100};
    draft_vocab_begin(v, prompt, 1, 0);                    /* counted: 1 */
    for (int i = 0; i < DRAFT_VOCAB_HALVE_EVERY - 1; i++) draft_vocab_count_token(v, 101);
    CHECK(v->run.rebuilds == 1, "one halving after %d counted tokens, got %llu rebuilds",
          DRAFT_VOCAB_HALVE_EVERY, (unsigned long long)v->run.rebuilds);
    CHECK(v->count[100] == 0.5f && v->count[101] == (DRAFT_VOCAB_HALVE_EVERY - 1) / 2.f,
          "counts after the halving: %g %g", v->count[100], v->count[101]);
    CHECK(v->counted == 0, "the period starts again");

    /* a new topic: the old token loses its place once the conversation is another one */
    int other[] = {102};
    draft_vocab_begin(v, other, 1, 0);
    for (int i = 0; i < DRAFT_VOCAB_HALVE_EVERY - 1; i++) draft_vocab_count_token(v, 103);
    CHECK(v->count[101] == (DRAFT_VOCAB_HALVE_EVERY - 1) / 4.f, "101 halved twice: %g", v->count[101]);
    CHECK(listed(v, 103), "103 has the highest count");
    CHECK(!listed(v, 101), "101 is below 3/4 of the top count and not in the conversation");
    draft_vocab_free(v);
}

static void test_rebuild_matches_the_rule(void) {
    uint32_t seed = 12345;
    for (int round = 0; round < 40; round++) {
        int top = 1 + (int)(seed % 150);
        DraftVocab *v = draft_vocab_new(VOCAB, top);
        for (int id = 0; id < VOCAB; id++) {
            seed = seed * 1103515245u + 12345u;
            uint32_t r = seed >> 8;
            v->count[id] = r % 7 == 0 ? 0.f : r % 5 == 0 ? 3e9f - (float)(r % 2048) * 1e6f
                         : r % 11 == 0 ? 1e-30f * (float)(r % 9) : (float)((r >> 4) % 40) * 0.5f;   /* ties, zeros, large, tiny */
            seed = seed * 1103515245u + 12345u;
            v->in_conversation[id] = ((seed >> 12) % 17) == 0;
        }
        draft_vocab_rebuild(v);
        uint8_t want[VOCAB];
        reference_list(v, want);
        for (int id = 0; id < VOCAB; id++)
            CHECK(listed(v, id) == want[id], "round %d top %d id %d: listed %d, rule gives %d",
                  round, top, id, listed(v, id), want[id]);
        draft_vocab_free(v);
        seed = seed * 1103515245u + 12345u;
    }
}

#ifndef _WIN32
/* 1 when enter(v, tok) ends the process with status 1; the text it wrote to stderr is in message */
static int ends_the_run(void (*enter)(DraftVocab *, int), DraftVocab *v, int tok, char *message, size_t size) {
    int fds[2];
    if (pipe(fds) != 0) return 0;
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) { dup2(fds[1], 2); close(fds[0]); close(fds[1]); enter(v, tok); _exit(0); }
    close(fds[1]);
    ssize_t n = read(fds[0], message, size - 1);
    message[n > 0 ? n : 0] = 0;
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 1;
}

static void begin_with_token(DraftVocab *v, int tok) {
    int prompt[] = {5, tok};
    draft_vocab_begin(v, prompt, 2, 0);
}

static void test_token_outside_the_vocabulary_ends_the_run(void) {
    DraftVocab *v = draft_vocab_new(VOCAB, 3);
    int bad[] = {-1, VOCAB, VOCAB + 5};
    for (int i = 0; i < 3; i++) {
        char message[256];
        CHECK(ends_the_run(begin_with_token, v, bad[i], message, sizeof message) && strstr(message, "outside"),
              "a prompt with the id %d should end the run with a message, got \"%s\"", bad[i], message);
        CHECK(ends_the_run(draft_vocab_count_token, v, bad[i], message, sizeof message) && strstr(message, "outside"),
              "a picked id %d should end the run with a message, got \"%s\"", bad[i], message);
    }
    draft_vocab_free(v);
}
#endif

static void test_counts_file(void) {
    const char *path = "test_draft_vocab_counts.tmp";
    DraftVocab *a = draft_vocab_new(VOCAB, 10);
    int prompt[300], len = 0;
    append_copies(prompt, &len, 33, 120);
    append_copies(prompt, &len, 77, 40);
    draft_vocab_begin(a, prompt, len, 0);
    a->count[91] = 1e-30f;   /* a count that has been halved many times */
    CHECK(draft_vocab_load(a, strdup(path)) == DRAFT_VOCAB_NO_FILE, "a counts file before the first save");
    CHECK(draft_vocab_save(a) == DRAFT_VOCAB_SAVED, "save failed");
    CHECK(draft_vocab_save(a) == DRAFT_VOCAB_SAVED, "a second save replaces the first");

    DraftVocab *b = draft_vocab_new(VOCAB, 10);
    CHECK(draft_vocab_load(b, strdup(path)) == DRAFT_VOCAB_LOADED, "load failed");
    CHECK(!memcmp(a->count, b->count, VOCAB * sizeof *a->count), "counts differ after a round trip (halved counts included)");
    CHECK(listed(b, 33) && listed(b, 77), "the list follows the loaded counts");
    double tokens; int distinct;
    draft_vocab_totals(b, &tokens, &distinct);
    CHECK(distinct == 3 && tokens > 159.99 && tokens < 160.01, "totals %g %d, want 160 3", tokens, distinct);

    DraftVocab *other = draft_vocab_new(VOCAB + 1, 10);
    CHECK(draft_vocab_load(other, strdup(path)) == DRAFT_VOCAB_OTHER_SIZE, "a file for another size is read");
    CHECK(other->count[33] == 0, "a refused file changed the counts");

    CHECK(draft_vocab_load(b, strdup("test_draft_vocab_missing.tmp")) == DRAFT_VOCAB_NO_FILE, "a missing file");

    /* records that are not counts are skipped, and the first one that cannot be read ends the file */
    write_file(path, "-1 200\n5 7\n-3 4\n200 9\n6 0\n7 -2\n10 nan\n11 inf\n12 0.25\n8 x\n9 9\n");
    DraftVocab *c = draft_vocab_new(VOCAB, 10);
    CHECK(draft_vocab_load(c, strdup(path)) == DRAFT_VOCAB_DAMAGED, "the unreadable record is reported");
    CHECK(c->records == 8, "records read %d, want the 8 before the unreadable one", c->records);
    CHECK(c->count[5] == 7 && c->count[12] == 0.25f, "records before the damage count (%g %g)", c->count[5], c->count[12]);
    CHECK(c->count[6] == 0 && c->count[7] == 0 && c->count[10] == 0 && c->count[11] == 0,
          "a zero, negative or non-finite count is not a count");
    CHECK(c->count[9] == 0, "records after the damage are not read");

    remove(path);
    draft_vocab_free(a); draft_vocab_free(b); draft_vocab_free(c); draft_vocab_free(other);
}

/* a file that the run did not read as counts is never replaced, whatever the run counted */
static void test_unusable_file_is_left_alone(void) {
    const char *path = "test_draft_vocab_foreign.tmp";
    struct { const char *what; const char *text; DraftVocabLoad found; int records; } cases[] = {
        {"text without the first line", "not a counts file\n", DRAFT_VOCAB_NOT_COUNTS, 0},
        {"an empty file", "", DRAFT_VOCAB_NOT_COUNTS, 0},
        {"counts for another vocabulary size", "-1 7\n3 2\n", DRAFT_VOCAB_OTHER_SIZE, 0},
        {"a record that is not a count", "-1 200\n5 7\n9 3\n8 x\n12 4\n", DRAFT_VOCAB_DAMAGED, 2},
        {"a record cut off", "-1 200\n5 7\n9", DRAFT_VOCAB_DAMAGED, 1},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        write_file(path, cases[i].text);
        DraftVocab *v = draft_vocab_new(VOCAB, 10);
        CHECK(draft_vocab_load(v, strdup(path)) == cases[i].found, "%s: found %d, want %d",
              cases[i].what, v->found, cases[i].found);
        CHECK(v->records == cases[i].records, "%s: %d records read, want %d", cases[i].what, v->records, cases[i].records);
        v->count[33] = 4;   /* what a run counts before it ends */
        CHECK(draft_vocab_save(v) == DRAFT_VOCAB_LEFT_ALONE, "%s: the save should leave the file alone", cases[i].what);
        CHECK(file_is(path, cases[i].text), "%s: the file changed", cases[i].what);
        draft_vocab_free(v);
    }
    remove(path);
#ifndef _WIN32
    CHECK(mkdir(path, 0700) == 0, "a directory at the path");
    DraftVocab *v = draft_vocab_new(VOCAB, 10);
    CHECK(draft_vocab_load(v, strdup(path)) == DRAFT_VOCAB_NOT_COUNTS, "a directory cannot be read as counts");
    CHECK(draft_vocab_save(v) == DRAFT_VOCAB_LEFT_ALONE, "the save should leave the directory alone");
    struct stat st;
    CHECK(stat(path, &st) == 0 && S_ISDIR(st.st_mode), "the directory is still there");
    draft_vocab_free(v);
    rmdir(path);
#endif
}

int main(void) {
    test_fresh_list_is_the_lowest_ids();
    test_fewer_counted_tokens_than_top();
    test_close_counts_and_conversation();
    test_restored_tokens_are_not_counted();
    test_pick_appends_and_counts();
    test_counts_halve_and_old_topics_fade();
    test_rebuild_matches_the_rule();
#ifndef _WIN32
    test_token_outside_the_vocabulary_ends_the_run();
#endif
    test_counts_file();
    test_unusable_file_is_left_alone();
    if (failures) { fprintf(stderr, "test_draft_vocab: %d failure(s)\n", failures); return 1; }
    printf("test_draft_vocab: ok\n");
    return 0;
}
