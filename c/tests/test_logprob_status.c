/* test_logprob_status.c — the classified logit-row reduction in sample.h and
 * the evidence digest in evidence_digest.h.
 *
 * Both are foundations for engine modes that write artifacts an offline tool
 * re-checks, so both need to be right for inputs the sampling path never sees:
 * rows containing NaN or an infinity, empty rows, and a target whose distance
 * from the row maximum does not fit in a float.  A wrong answer there is worse
 * than a crash, because it is written into a file and read back later as if it
 * were a measurement.
 *
 * Required properties:
 *   P1 AGREEMENT — on a well-behaved row the classified path returns exactly
 *      the value and argmax flag the plain logprob_target() returns, so adding
 *      the classified path changes no existing number.
 *   P2 CLASSIFICATION — each exceptional row shape reports its own cause, not a
 *      generic failure, no partition function is invented for it, and the
 *      precedence between the causes is the documented one.
 *   P3 PROPAGATION — reading a target out of a row that did not reduce cleanly
 *      reports the row's original cause, a caller mistake is reported as such
 *      rather than as a usable value, and every refusal leaves a defined
 *      number behind rather than whatever the caller's variable held.
 *   P4 DIGEST — the digest matches published SHA-256 values, including the
 *      lengths that exercise the block boundary and both padding branches, and
 *      the streaming form agrees with the one-shot form.
 * Exit 0 = all pass.
 */
#include <float.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "../evidence_digest.h"

static int fails = 0;
#define CHECK(cond,msg) do{ if(!(cond)){ printf("  FAIL: %s\n", msg); fails++; } \
                            else printf("  ok:   %s\n", msg); }while(0)

static int close_enough(double a, double b){
    double d = a - b;
    if(d < 0) d = -d;
    return d <= 1e-12;
}

/* P1: the classified path must not move a single existing number. */
static void t_agreement(void){
    printf("P1 agreement with the plain path\n");
    static const float rows[3][5] = {
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        { 1.0f, 2.0f, 3.0f, -1.0f, 0.5f },
        { -30.0f, -31.0f, -29.5f, -100.0f, -29.75f },
    };
    for(int r=0;r<3;r++){
        LogprobRow row;
        CHECK(logprob_row_checked(rows[r],5,&row)==LOGPROB_FINITE,
              "finite row reduces to FINITE");
        for(int target=0;target<5;target++){
            double classified = 0;
            CHECK(logprob_from_row_checked(rows[r],target,&row,&classified)==LOGPROB_FINITE,
                  "finite row yields a finite target value");
            int plain_argmax = 0;
            double plain = logprob_target(rows[r],5,target,&plain_argmax);
            CHECK(close_enough(classified,plain),
                  "classified value equals the plain logprob_target value");
            CHECK(plain_argmax==(row.argmax==target),
                  "classified argmax flag equals the plain argmax flag");
        }
    }
    /* A two-entry uniform row has a hand-checkable answer: log(1/2). */
    static const float uniform[2] = { 0.0f, 0.0f };
    LogprobRow row;
    logprob_row_checked(uniform,2,&row);
    double value = 0;
    logprob_from_row_checked(uniform,0,&row,&value);
    CHECK(close_enough(value,-log(2.0)), "uniform pair gives log(1/2)");
    CHECK(close_enough(row.logZ,log(2.0)), "uniform pair has logZ = log(2)");
    CHECK(row.argmax==0, "uniform pair takes the first maximum");
}

/* P2: every exceptional row shape names its own cause. */
static void t_classification(void){
    printf("P2 exceptional rows are classified\n");
    LogprobRow row;

    float nan_row[3] = { 1.0f, NAN, 3.0f };
    CHECK(logprob_row_checked(nan_row,3,&row)==LOGPROB_NAN, "NaN present -> NAN");
    CHECK(row.status==LOGPROB_NAN, "the returned row carries the same status");
    CHECK(row.logZ==0 && row.max==0, "no partition function is invented");

    float pos_row[3] = { 1.0f, INFINITY, 3.0f };
    CHECK(logprob_row_checked(pos_row,3,&row)==LOGPROB_POS_INF, "+inf present -> POS_INF");

    float neg_row[3] = { 1.0f, -INFINITY, 3.0f };
    CHECK(logprob_row_checked(neg_row,3,&row)==LOGPROB_NEG_INF, "-inf present -> NEG_INF");

    /* Precedence, stated as a fixed order rather than the order of appearance.
     * Each row below contains more than one exceptional class, so a change to
     * the ranking changes an answer here. */
    float mixed_row[3] = { NAN, INFINITY, -INFINITY };
    CHECK(logprob_row_checked(mixed_row,3,&row)==LOGPROB_ALL_NONFINITE,
          "no finite entry at all -> ALL_NONFINITE, whatever the classes are");

    float finite_nan_pos[3] = { 1.0f, NAN, INFINITY };
    CHECK(logprob_row_checked(finite_nan_pos,3,&row)==LOGPROB_NAN,
          "a NaN outranks a positive infinity");
    float finite_pos_nan[3] = { 1.0f, INFINITY, NAN };
    CHECK(logprob_row_checked(finite_pos_nan,3,&row)==LOGPROB_NAN,
          "a NaN outranks a positive infinity that appears before it");
    float finite_nan_neg[3] = { 1.0f, NAN, -INFINITY };
    CHECK(logprob_row_checked(finite_nan_neg,3,&row)==LOGPROB_NAN,
          "a NaN outranks a negative infinity");
    float finite_pos_neg[3] = { 1.0f, INFINITY, -INFINITY };
    CHECK(logprob_row_checked(finite_pos_neg,3,&row)==LOGPROB_POS_INF,
          "a positive infinity outranks a negative one");
    float finite_neg_pos[3] = { 1.0f, -INFINITY, INFINITY };
    CHECK(logprob_row_checked(finite_neg_pos,3,&row)==LOGPROB_POS_INF,
          "a positive infinity outranks a negative one that appears first");
    float finite_neg[3] = { 1.0f, 2.0f, -INFINITY };
    CHECK(logprob_row_checked(finite_neg,3,&row)==LOGPROB_NEG_INF,
          "a negative infinity alongside finite entries -> NEG_INF");
    float all_nan[2] = { NAN, NAN };
    CHECK(logprob_row_checked(all_nan,2,&row)==LOGPROB_ALL_NONFINITE,
          "a row of NaN alone is still ALL_NONFINITE");

    CHECK(logprob_row_checked(NULL,3,&row)==LOGPROB_INVALID, "no row -> INVALID");
    CHECK(logprob_row_checked(nan_row,0,&row)==LOGPROB_INVALID, "empty row -> INVALID");
    CHECK(logprob_row_checked(nan_row,-1,&row)==LOGPROB_INVALID, "negative length -> INVALID");
    CHECK(logprob_row_checked(nan_row,3,NULL)==LOGPROB_NAN,
          "the status is returned even with no output row");

    CHECK(strcmp(logprob_status_name(LOGPROB_FINITE),"FINITE")==0, "FINITE names itself");
    CHECK(strcmp(logprob_status_name(LOGPROB_NAN),"NAN")==0, "NAN names itself");
    CHECK(strcmp(logprob_status_name(LOGPROB_POS_INF),"POS_INF")==0, "POS_INF names itself");
    CHECK(strcmp(logprob_status_name(LOGPROB_NEG_INF),"NEG_INF")==0, "NEG_INF names itself");
    CHECK(strcmp(logprob_status_name(LOGPROB_ALL_NONFINITE),"ALL_NONFINITE")==0,
          "ALL_NONFINITE names itself");
    CHECK(strcmp(logprob_status_name(LOGPROB_FINITE_OVERFLOW),"FINITE_OVERFLOW")==0,
          "FINITE_OVERFLOW names itself");
    CHECK(strcmp(logprob_status_name(LOGPROB_INVALID),"INVALID")==0, "INVALID names itself");
}

/* P3: a target read out of a bad row reports the row's cause, and a target the
 * float subtraction cannot represent is refused rather than rounded. */
static void t_propagation(void){
    printf("P3 target reads propagate the row's cause\n");
    LogprobRow row;
    float nan_row[3] = { 1.0f, NAN, 3.0f };
    logprob_row_checked(nan_row,3,&row);
    double value = 12345.0;
    CHECK(logprob_from_row_checked(nan_row,0,&row,&value)==LOGPROB_NAN,
          "a NaN row propagates NAN to the target read");
    CHECK(isnan(value), "a refused target read leaves a defined value behind");

    CHECK(logprob_from_row_checked(nan_row,0,NULL,&value)==LOGPROB_INVALID,
          "no row -> INVALID");

    /* A caller mistake on a perfectly good row must not read back as a usable
     * value: the earlier form returned FINITE here and never wrote the output,
     * so the caller consumed whatever its own variable happened to hold. */
    static const float good_row[3] = { 1.0f, 2.0f, 3.0f };
    LogprobRow good;
    CHECK(logprob_row_checked(good_row,3,&good)==LOGPROB_FINITE, "the good row reduces");
    value = 12345.0;
    CHECK(logprob_from_row_checked(NULL,0,&good,&value)==LOGPROB_INVALID,
          "no logit vector on a good row -> INVALID, never FINITE");
    CHECK(isnan(value), "no logit vector leaves a defined value behind");
    value = 12345.0;
    CHECK(logprob_from_row_checked(good_row,-1,&good,&value)==LOGPROB_INVALID,
          "a negative target on a good row -> INVALID, never FINITE");
    CHECK(isnan(value), "a negative target leaves a defined value behind");

    logprob_row_checked(nan_row,3,&row);
    CHECK(logprob_from_row_checked(NULL,0,&row,&value)==LOGPROB_INVALID,
          "no logit vector is a caller mistake even when the row is bad");

    /* The widest representable spread: the float difference saturates even
     * though every logit in the row is finite. */
    float wide[2] = { FLT_MAX, -FLT_MAX };
    CHECK(logprob_row_checked(wide,2,&row)==LOGPROB_FINITE, "an extreme finite row still reduces");
    CHECK(logprob_from_row_checked(wide,1,&row,&value)==LOGPROB_FINITE_OVERFLOW,
          "a target the float subtraction cannot hold -> FINITE_OVERFLOW");
    CHECK(logprob_from_row_checked(wide,0,&row,&value)==LOGPROB_FINITE,
          "the maximum itself still reads back finite");
    CHECK(close_enough(value,0.0), "the maximum of a saturated row has log-probability 0");

    logprob_row_checked(nan_row,0,&row);
    CHECK(logprob_from_row_checked(nan_row,-1,&row,&value)==LOGPROB_INVALID,
          "a negative target on an invalid row -> INVALID");
}

/* P4: published SHA-256 answers, chosen to cover the block boundary and both
 * padding branches (a message whose tail leaves no room for the length field
 * needs an extra block). */
static void t_digest(void){
    printf("P4 evidence digest matches published SHA-256 values\n");
    static const struct { const char *text; const char *hex; } vectors[] = {
        { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
        { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
        { "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno",
          "2ff100b36c386c65a1afc462ad53e25479bec9498ed00aa5a04de584bc25301b" },
    };
    char out[65];
    for(unsigned v=0;v<sizeof(vectors)/sizeof(vectors[0]);v++){
        evidence_sha256_hex(vectors[v].text,strlen(vectors[v].text),out);
        CHECK(strcmp(out,vectors[v].hex)==0, "one-shot digest matches the published value");
        CHECK(strlen(out)==64, "the digest is 64 characters and NUL-terminated");
    }

    /* 65 bytes: one full block plus a remainder, fed one byte at a time, so a
     * mishandled carry between updates would show up here and not above. */
    char sixty_five[66];
    memset(sixty_five,'a',65); sixty_five[65]=0;
    EvidenceSha256 stream;
    unsigned char raw[32];
    evidence_sha256_init(&stream);
    for(int i=0;i<65;i++) evidence_sha256_update(&stream,sixty_five+i,1);
    evidence_sha256_final(&stream,raw);
    char streamed[65];
    static const char hex[]="0123456789abcdef";
    for(int i=0;i<32;i++){ streamed[2*i]=hex[raw[i]>>4]; streamed[2*i+1]=hex[raw[i]&15]; }
    streamed[64]=0;
    CHECK(strcmp(streamed,
                 "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0")==0,
          "a byte-at-a-time stream matches the published value");
    evidence_sha256_hex(sixty_five,65,out);
    CHECK(strcmp(streamed,out)==0, "the streaming and one-shot forms agree");
}

int main(void){
    printf("test_logprob_status\n");
    t_agreement();
    t_classification();
    t_propagation();
    t_digest();
    printf(fails ? "FAILED (%d)\n" : "PASSED (%d failures)\n", fails);
    return fails ? 1 : 0;
}
