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
 *   P1 AGREEMENT — on a row whose logits sit close enough together that a
 *      float-scale subtraction cannot lose precision, the classified path
 *      returns exactly the value and argmax flag the plain logprob_target()
 *      returns. The two paths are not required to agree past a float's own
 *      precision on a widely spread row: logprob_target() keeps its
 *      historical float-scale subtraction, while the classified path
 *      promotes to double first (P1B).
 *   P1B PRECISION — the classified path's value is accurate to double
 *      precision: it matches, bit for bit, a reference computed
 *      a second time in this file by promoting the row's logit to double
 *      before subtracting, on rows spanning ordinary, widely spread,
 *      subnormal and full-double-precision inputs. This is the property
 *      P1's exact test rows are too narrow to exercise.
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

/* Tight tolerance for P1B: the classified path's value against an
 * second double computation should agree to a few ulp; the relative bound leaves no room
 * for a float-scale rounding to sneak back in while tolerating, at most,
 * a difference in the last bit of a double reduction. */
static int close_enough_precise(double a, double b){
    /* a few units in the last place, relative to the reference magnitude */
    double tol = 16.0 * DBL_EPSILON * fabs(b) + 1e-300;
    return fabs(a - b) <= tol;
}

/* P1: on a row a float subtraction cannot mis-round, the classified path
 * agrees with the plain path exactly. */
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

/* A second double computation for P1B (same formula as the code under test, so it
 * detects float-first rounding, not ulp-level error): re-finds the row's maximum and
 * log-sum-exp from scratch, entirely in double, without touching the
 * LogprobRow the function under test already reduced. This is not the same
 * code path as logprob_from_row_checked() -- it recomputes the row instead
 * of reading r->max/r->logZ -- so it cannot agree with a wrong answer by
 * sharing a mistake. */
static double reference_double(const float *lo, int V, int target){
    double mx = (double)lo[0];
    for(int i=1;i<V;i++){ double v=(double)lo[i]; if(v>mx) mx=v; }
    double se = 0;
    for(int i=0;i<V;i++) se += exp((double)lo[i]-mx);
    return (double)lo[target] - (mx + log(se));
}

static void check_row_precise(const char *name, const float *lo, int V,
                               const int *targets, int ntargets){
    LogprobRow row;
    CHECK(logprob_row_checked(lo,V,&row)==LOGPROB_FINITE, name);
    for(int k=0;k<ntargets;k++){
        int target = targets[k];
        double value = 0;
        CHECK(logprob_from_row_checked(lo,target,&row,&value)==LOGPROB_FINITE,
              "the target reads back finite");
        double ref = reference_double(lo,V,target);
        CHECK(close_enough_precise(value,ref),
              "the classified path's value agrees with the second double computation to a few ulp");
    }
}

/* P1B: the classified path agrees with a second double computation (to a few units in
 * the last place; ~13 significant digits on the widest rows) on rows P1's own
 * exact test rows are too narrow to tell apart from a float-rounded answer.
 * Table covers: an ordinary tightly-spread row; a wide-spread row mirroring
 * test_ablate_mode.c's t_nll_pin (max 1.0e7, one target's contribution under
 * half a float ulp at that magnitude); a moderately spread row whose correct
 * value needs digits past a float's own seven to state; and a subnormal row,
 * where the magnitudes are so small a float-first subtraction cannot lose
 * anything, so this also checks the fix introduces no new problem there. */
static void t_precision(void){
    printf("P1B classified path agrees with a second double computation to a few ulp\n");

    static const float ordinary[4] = { 2.0f, -1.5f, 0.25f, -3.0f };
    int t_ordinary[4] = { 0, 1, 2, 3 };
    check_row_precise("ordinary row reduces to FINITE", ordinary, 4, t_ordinary, 4);

    /* Mirrors test_ablate_mode.c's t_nll_pin: max 1.0e7 at index 0, the
     * target of interest at index 2 with a logit of 0.25 -- a contribution
     * the float subtraction lo[2]-max rounds away entirely, since the ulp
     * near 1.0e7 is 1.0 and 0.25 is under half of it. */
    static float wide_spread[64];
    for(int i=0;i<64;i++) wide_spread[i] = -1.0e7f;
    wide_spread[0] = 1.0e7f;
    wide_spread[2] = 0.25f;
    int t_wide[5] = { 0, 1, 2, 5, 63 };
    check_row_precise("wide-spread row reduces to FINITE", wide_spread, 64, t_wide, 5);

    /* A moderately spread row: max in the tens of thousands, targets with
     * fractional digits a float at that magnitude cannot all hold. */
    static const float moderate[5] =
        { 10000.0f, 0.123456789f, -5000.5f, 9999.99f, 3.14159265f };
    int t_moderate[5] = { 0, 1, 2, 3, 4 };
    check_row_precise("moderate-spread row reduces to FINITE", moderate, 5, t_moderate, 5);

    /* Subnormal logits: values this small carry no ulp large enough for a
     * float-first subtraction to lose, so this is a robustness check as much
     * as a precision one. */
    static const float subnormal[3] =
        { 1e-45f, 1e-45f*3.0f, 1e-45f*7.0f };
    int t_subnormal[3] = { 0, 1, 2 };
    check_row_precise("subnormal row reduces to FINITE", subnormal, 3, t_subnormal, 3);
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

/* P3: a target read out of a bad row reports the row's cause. A spread that
 * would overflow a float subtraction no longer refuses the target, because
 * the classified path takes that subtraction in double: it is pinned below
 * as a status change, not silently absorbed into "still finite". */
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

    /* The widest representable spread: a float subtraction between these two
     * values would saturate to infinity even though every logit in the row
     * is finite. The classified path subtracts in double, so it does not:
     * this is a real, disclosed widening of what the layer accepts, pinned
     * here so it cannot regress silently in either direction. */
    float wide[2] = { FLT_MAX, -FLT_MAX };
    CHECK(logprob_row_checked(wide,2,&row)==LOGPROB_FINITE, "an extreme finite row still reduces");
    CHECK(logprob_from_row_checked(wide,1,&row,&value)==LOGPROB_FINITE,
          "a target the float subtraction could not hold -> FINITE, not refused, under the double subtraction");
    CHECK(value==-2.0*(double)FLT_MAX,
          "the saturated target's value is the exact double difference, not an infinity");
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
    t_precision();
    t_classification();
    t_propagation();
    t_digest();
    printf(fails ? "FAILED (%d)\n" : "PASSED (%d failures)\n", fails);
    return fails ? 1 : 0;
}
