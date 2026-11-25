/* Next-layer prefetcher queue policy, token-start, depth-2, gating, router
 * thread, and V4_PREFETCH_NO_CREDIT (deepseek_v4.c, block unit + hot store):
 *   - Default replace vs V4_NEXT_PREFETCH_KEEP (replace/prune/reset, stale)
 *   - V4_NEXT_PREFETCH_TOKSTART (leading hash layers queued at token start)
 *   - V4_NEXT_PREFETCH_DEPTH=2 / V4_NEXT_PREFETCH_K2 (L+2 prediction + KEEP)
 *   - Gating (V4_NP_MIN_CONF, V4_NP_TOKEN_MB): confidences, admission, budget
 *   - Depth-1 router worker (unbiased and biased bf16 gate)
 *   - Reader work on a tiny 7-expert safetensors fixture store
 *   - V4_PREFETCH_NO_CREDIT: prefetch lookups skip usage/requests/hit/heat
 * Runs against a block unit built with -DCOLI_V4_NP_TEST (hooks only; no
 * reader threads, so the queue holds exactly what was issued). No model,
 * no GPU. Prints "v4 np queue: ok". */
#include "../deepseek_v4_internal.h"
#include "../compat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void coli_v4_np_test_setup(int keep, int tokstart, int stats);
void coli_v4_np_test_depth(int depth, int k2);
void coli_v4_np_test_set_forward(uint32_t forward);
void coli_v4_np_test_issue(ColiExpertStore *store, int layer, const int *experts, int count,
                           int mode, uint32_t stamp);
void coli_v4_np_test_predict(ColiExpertStore *store, const ColiDeepSeekV4Config *config,
                             int layer, const float *input, int token);
int coli_v4_np_test_queue(int *layers, int *experts, uint32_t *stamps, int max);
void coli_v4_np_test_stats(unsigned long long *queue_stats, int layer, int *n,
                           uint32_t *stamp, int *ids);
void coli_v4_np_test_pred2(int layer, int *n, uint32_t *stamp, int *ids);
void coli_v4_np_test_gate(float min_conf, double token_mb);
int coli_v4_np_test_admit(float conf, uint32_t stamp, uint64_t bytes);
void coli_v4_np_test_gate_stats(unsigned long long *gate_stats, uint32_t *stamp,
                                uint64_t *spent);
int coli_v4_np_test_confs(float *conf, int max);
void coli_v4_np_test_score_conf(ColiExpertStore *store, const ColiDeepSeekV4LayerWeights *layer,
                                const int *guess, const float *weights, int k, int topk,
                                const int *routed, int selected, unsigned long long *hist);
void coli_v4_np_test_router(int k);
void coli_v4_np_test_work(ColiExpertStore *store, ColiExpertKey key, int rank, float conf,
                          uint32_t stamp);

enum { APPEND, REPLACE, PRUNE, RESET };   /* = V4_NP_APPEND.. in the unit */
enum { QUEUED, DROPPED, KEPT, STALE, TOKSTART, FULL, NQ };   /* = V4_NPQ_* */
enum { ADMITTED, GATED_CONF, GATED_BUDGET, BYTES, NG };   /* = V4_NPG_* */
#define BUCKETS 10   /* = V4_NP_CONF_BUCKETS */

static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; fprintf(stderr, "FAIL %s:%d: ", \
    __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

#define LAYERS 43
#define VOCAB 64
#define TOPK 6
static ColiDeepSeekV4LayerWeights layers[LAYERS];
static unsigned char ready[LAYERS];
static int64_t tid2eid[3][VOCAB * TOPK];
static ColiDeepSeekV4Config config;
static int store_stub;
#define STORE ((ColiExpertStore *)(void *)&store_stub)
static float input[64];

static int expert_of(int layer, int token, int i) { return (layer * 37 + token * 11 + i * 5) % 256; }

static void setup_layers(void) {
    for (int l = 0; l < LAYERS; l++) {
        ColiDeepSeekV4LayerWeights *w = &layers[l];
        memset(w, 0, sizeof *w);
        w->plan.layer = l;
        w->plan.uses_hash_router = l < 3;
        if (l < 3) {
            for (int t = 0; t < VOCAB; t++)
                for (int i = 0; i < TOPK; i++)
                    tid2eid[l][t * TOPK + i] = expert_of(l, t, i);
            w->plan.tensor_count = 1;
            snprintf(w->plan.tensors[0].name, sizeof w->plan.tensors[0].name,
                     "layers.%d.ffn.gate.tid2eid", l);
            w->data[0] = tid2eid[l];
        }
        ready[l] = 1;
    }
    g_v4_next_layers = layers;
    g_v4_next_ready = ready;
    g_v4_next_nlayers = LAYERS;
    memset(&config, 0, sizeof config);
    config.num_hidden_layers = LAYERS;
    config.n_routed_experts = 256;
    config.num_experts_per_tok = TOPK;
    config.hidden_size = 64;
    config.vocab_size = VOCAB;
    config.routed_scaling_factor = 1.5f;
}

typedef struct { int n, layer[256], expert[256]; uint32_t stamp[256]; } Queue;
static Queue snap(void) {
    Queue q;
    q.n = coli_v4_np_test_queue(q.layer, q.expert, q.stamp, 256);
    return q;
}
/* the queue, as "layer:count" runs head first, e.g. "1:6 2:6" */
static const char *shape(void) {
    static char out[512];
    Queue q = snap();
    out[0] = 0;
    for (int i = 0; i < q.n && i < 256;) {
        int j = i;
        while (j < q.n && q.layer[j] == q.layer[i]) j++;
        snprintf(out + strlen(out), sizeof out - strlen(out), "%s%d:%d", i ? " " : "",
                 q.layer[i], j - i);
        i = j;
    }
    return out;
}
static unsigned long long counter(int which) {
    unsigned long long q[NQ];
    int n, ids[16];
    uint32_t stamp;
    coli_v4_np_test_stats(q, 0, &n, &stamp, ids);
    return q[which];
}

static void test_default_replace(void) {
    /* KEEP off: replace wipes every unstarted read, append keeps them */
    coli_v4_np_test_setup(0, 0, 1);
    coli_v4_np_test_set_forward(5);
    int a[8] = {1, 2, 3, 4, 5, 6, 7, 8}, b[6] = {9, 10, 11, 12, 13, 14};
    coli_v4_np_test_issue(STORE, 3, a, 8, REPLACE, 5);
    coli_v4_np_test_issue(STORE, 4, b, 6, APPEND, 5);
    CHECK(!strcmp(shape(), "3:8 4:6"), "default: %s", shape());
    coli_v4_np_test_issue(STORE, 4, a, 8, REPLACE, 5);
    CHECK(!strcmp(shape(), "4:8"), "default replace wipes all: %s", shape());
    /* no staleness without KEEP: an old forward's guess is still queued */
    coli_v4_np_test_issue(STORE, 9, b, 6, APPEND, 1);
    CHECK(!strcmp(shape(), "4:8 9:6"), "default keeps an old forward's guess: %s", shape());
    Queue q = snap();
    CHECK(q.expert[0] == 1 && q.expert[7] == 8 && q.stamp[0] == 5 && q.stamp[8] == 1,
          "default order/stamps");
    CHECK(counter(STALE) == 0 && counter(DROPPED) == 14 && counter(QUEUED) == 28, "default counters "
          "stale=%llu dropped=%llu queued=%llu", counter(STALE), counter(DROPPED), counter(QUEUED));
    /* hash-routed layer prediction publishes exact tid2eid experts */
    coli_v4_np_test_predict(STORE, &config, 0, input, 4);
    CHECK(!strcmp(shape(), "1:6"), "hash publish: %s", shape());
    coli_v4_np_test_predict(STORE, &config, 1, input, 4);
    CHECK(!strcmp(shape(), "2:6"), "hash publish replaces: %s", shape());
    q = snap();
    CHECK(q.n == TOPK && q.expert[0] == expert_of(2, 4, 0), "hash expert mismatch");
}

static void test_keep(void) {
    coli_v4_np_test_setup(1, 0, 1);
    coli_v4_np_test_set_forward(7);
    int d1[8] = {20, 21, 22, 23, 24, 25, 26, 27}, d2[6] = {30, 31, 32, 33, 34, 35};
    int n1[8] = {40, 41, 42, 43, 44, 45, 46, 47};
    coli_v4_np_test_issue(STORE, 3, d1, 8, REPLACE, 7);
    coli_v4_np_test_issue(STORE, 4, d2, 6, APPEND, 7);
    CHECK(!strcmp(shape(), "3:8 4:6"), "keep start: %s", shape());
    coli_v4_np_test_issue(STORE, 5, d2, 6, APPEND, 7);
    coli_v4_np_test_issue(STORE, 4, n1, 8, REPLACE, 7);
    CHECK(!strcmp(shape(), "4:8 5:6"), "keep replace: %s", shape());
    Queue q = snap();
    CHECK(q.expert[0] == 40 && q.expert[7] == 47 && q.expert[8] == 30, "keep order");
    /* prune for 5 (layer 4 routed): drops 4, keeps 5 */
    coli_v4_np_test_issue(STORE, 5, NULL, 0, PRUNE, 7);
    CHECK(!strcmp(shape(), "5:6"), "keep prune: %s", shape());
    /* a guess whose forward is over is not queued */
    coli_v4_np_test_issue(STORE, 9, d1, 8, REPLACE, 6);
    coli_v4_np_test_issue(STORE, 9, d1, 8, APPEND, 6);
    CHECK(!strcmp(shape(), "5:6"), "keep stale: %s", shape());
    CHECK(counter(STALE) == 16, "stale counter %llu", counter(STALE));
    /* a newer forward's items survive an older forward's late replace */
    coli_v4_np_test_issue(STORE, 0, d2, 6, APPEND, 8);
    coli_v4_np_test_issue(STORE, 42, d1, 8, REPLACE, 7);
    CHECK(!strcmp(shape(), "42:8 0:6"), "keep newer forward: %s", shape());
    /* reset drops everything */
    coli_v4_np_test_issue(STORE, 0, n1, 6, RESET, 8);
    CHECK(!strcmp(shape(), "0:6"), "reset: %s", shape());
    /* full queue: 255 slots; the overflow is counted */
    int many[16];
    for (int i = 0; i < 16; i++) many[i] = 100 + i;
    for (int l = 1; l <= 20; l++) coli_v4_np_test_issue(STORE, l, many, 16, APPEND, 8);
    Queue full = snap();
    CHECK(full.n == 255, "full queue holds %d", full.n);
    CHECK(counter(FULL) == 6 + 320 - 255, "full counter %llu", counter(FULL));
    /* a replace on a full queue keeps the later layers that still fit */
    coli_v4_np_test_issue(STORE, 10, many, 16, REPLACE, 8);
    Queue after = snap();
    CHECK(after.layer[0] == 10 && after.layer[16] == 11 && after.n <= 255,
          "replace on a full queue: head %d, then %d, n %d", after.layer[0], after.layer[16],
          after.n);
}

/* Gating: confidences in the queue and the published guess, and the
 * used/guessed histogram (hash layers left out). */
static void test_conf(void) {
    coli_v4_np_test_setup(0, 0, 1);
    coli_v4_np_test_gate(0.0f, 0.0);
    coli_v4_np_test_set_forward(40);
    int guess[4] = {10, 11, 12, 13};
    float w[4] = {2.0f, 1.0f, 1.0f, 0.5f};   /* conf 0.5 0.25 0.25 0.125 over topk=3 */
    int routed[6] = {10, 12, 21, 50, 51, 52};
    unsigned long long hist[BUCKETS][2];
    coli_v4_np_test_score_conf(STORE, &layers[5], guess, w, 4, 3, routed, 6, &hist[0][0]);
    float conf[8];
    int n = coli_v4_np_test_confs(conf, 8);
    CHECK(n == 4 && conf[0] == 0.5f && conf[1] == 0.25f && conf[2] == 0.25f &&
          conf[3] == 0.125f, "queued confidences: n %d, %g %g %g %g", n, conf[0], conf[1],
          conf[2], conf[3]);
    /* buckets of 0.05: 0.5 -> 9 (the open last), 0.25 -> 5, 0.125 -> 2 */
    CHECK(hist[9][0] == 1 && hist[9][1] == 1 && hist[5][0] == 1 &&
          hist[5][1] == 2 && hist[2][0] == 0 && hist[2][1] == 1,
          "depth-1 histogram");
    /* a hash layer is not counted */
    coli_v4_np_test_score_conf(STORE, &layers[1], guess, w, 4, 3, routed, 6, &hist[0][0]);
    CHECK(hist[2][1] == 1 && hist[9][1] == 1, "hash layer counted");
    /* an exact list (hash layers) has confidence 1 */
    coli_v4_np_test_issue(STORE, 6, guess, 3, APPEND, 40);
    n = coli_v4_np_test_confs(conf, 8);
    CHECK(n == 7 && conf[4] == 1.0f && conf[6] == 1.0f, "exact list: n %d conf %g", n, conf[4]);
    /* all-zero weights: no confidence, and no division by zero */
    float zero[4] = {0, 0, 0, 0};
    coli_v4_np_test_score_conf(STORE, &layers[7], guess, zero, 4, 3, routed, 6, &hist[0][0]);
    n = coli_v4_np_test_confs(conf, 8);
    CHECK(n == 4 && conf[0] == 0.0f && conf[3] == 0.0f, "zero weights: %g", conf[0]);
}

static void gate_state(unsigned long long *g, uint32_t *stamp, uint64_t *spent) {
    coli_v4_np_test_gate_stats(g, stamp, spent);
}

/* Gating: threshold, per-forward budget, its reset on a newer stamp. */
static void test_budget(void) {
    unsigned long long g[NG];
    uint32_t stamp;
    uint64_t spent;
    const uint64_t mb = 1048576;
    coli_v4_np_test_gate(0.2f, 1.0);
    CHECK(coli_v4_np_test_admit(0.1f, 1, 400000) == 1, "below the threshold");
    gate_state(g, &stamp, &spent);
    CHECK(stamp == 1 && spent == 0, "a refused guess charged %llu", (unsigned long long)spent);
    CHECK(coli_v4_np_test_admit(0.5f, 1, 400000) == 0, "first within budget");
    CHECK(coli_v4_np_test_admit(0.5f, 1, 400000) == 0, "second within budget");
    CHECK(coli_v4_np_test_admit(0.5f, 1, 400000) == 2, "past the budget");
    gate_state(g, &stamp, &spent);
    CHECK(spent == 800000, "over budget charged: %llu", (unsigned long long)spent);
    /* the threshold itself passes; the budget is reached exactly */
    CHECK(coli_v4_np_test_admit(0.2f, 1, 200000) == 0, "conf at the threshold");
    CHECK(coli_v4_np_test_admit(0.9f, 0, mb - 1000000) == 0,
          "an older forward's guess, charged to this one, up to the budget");
    gate_state(g, &stamp, &spent);
    CHECK(stamp == 1 && spent == mb, "budget full: stamp %u spent %llu", stamp,
          (unsigned long long)spent);
    CHECK(coli_v4_np_test_admit(0.9f, 1, 1) == 2, "one byte past the budget");
    /* a newer forward starts over */
    CHECK(coli_v4_np_test_admit(0.9f, 2, mb) == 0, "new forward");
    CHECK(coli_v4_np_test_admit(0.9f, 2, 1) == 2, "new forward full");
    gate_state(g, &stamp, &spent);
    CHECK(stamp == 2 && spent == mb, "new forward: stamp %u", stamp);
    CHECK(g[ADMITTED] == 5 && g[GATED_CONF] == 1 && g[GATED_BUDGET] == 3 &&
          g[BYTES] == 2 * mb, "counters admitted %llu conf %llu budget %llu bytes %llu",
          g[ADMITTED], g[GATED_CONF], g[GATED_BUDGET], g[BYTES]);
    /* the stamp wraps: 0xfffffffe is older than 2, 1 is newer than 0xfffffffe */
    CHECK(coli_v4_np_test_admit(0.9f, 0xfffffffeu, 1) == 2, "a much older stamp");
    CHECK(coli_v4_np_test_admit(0.9f, 0x80000001u, 1) == 0, "half a cycle on: newer");
    CHECK(coli_v4_np_test_admit(0.9f, 0xfffffffeu, mb) == 0, "before the wrap");
    CHECK(coli_v4_np_test_admit(0.9f, 0xfffffffeu, 1) == 2, "before the wrap, full");
    CHECK(coli_v4_np_test_admit(0.9f, 1, mb) == 0, "after the wrap: a new forward");
    gate_state(g, &stamp, &spent);
    CHECK(stamp == 1 && spent == mb, "after the wrap: stamp %u", stamp);
    /* threshold alone: no byte limit; budget alone: any confidence */
    coli_v4_np_test_gate(0.3f, 0.0);
    CHECK(coli_v4_np_test_admit(0.5f, 3, 100 * mb) == 0 &&
          coli_v4_np_test_admit(0.5f, 3, 100 * mb) == 0, "no budget");
    CHECK(coli_v4_np_test_admit(0.29f, 3, 1) == 1, "threshold alone");
    coli_v4_np_test_gate(0.0f, 0.5);
    CHECK(coli_v4_np_test_admit(0.0f, 4, mb / 2) == 0, "budget alone, zero confidence");
    CHECK(coli_v4_np_test_admit(1.0f, 4, 1) == 2, "budget alone, full");
    coli_v4_np_test_gate(0.0f, 0.0);
}

/* Depth-1 router thread, K 8 > topk 6, on layer 5 given a bf16 gate:
 * publishes the top 8 with confidences normalized over top-6. */
static uint16_t gate[256 * 64];
static float hidden[64], gate_bias[256];
static void test_router(int experts, int biased, uint32_t forward) {
    float logit[256];
    for (int r = 0; r < experts; r++) {
        logit[r] = (float)((r * 73) % 256 - 128) / 32.0f;   /* distinct, exact in bf16 */
        uint32_t bits;
        memcpy(&bits, &logit[r], 4);
        gate[r * 64] = (uint16_t)(bits >> 16);
        gate_bias[r] = biased ? (float)((r * 29) % 16) / 4.0f : 0.0f;
    }
    hidden[0] = 1.0f;
    layers[5].plan.tensor_count = biased ? 2 : 1;
    snprintf(layers[5].plan.tensors[0].name, sizeof layers[5].plan.tensors[0].name,
             "layers.5.ffn.gate.weight");
    layers[5].data[0] = gate;
    snprintf(layers[5].plan.tensors[1].name, sizeof layers[5].plan.tensors[1].name,
             "layers.5.ffn.gate.bias");
    layers[5].data[1] = gate_bias;
    config.n_routed_experts = experts;
    coli_v4_np_test_setup(0, 0, 1);
    coli_v4_np_test_gate(0.0f, 0.0);
    coli_v4_np_test_router(8);
    coli_v4_np_test_set_forward(forward);
    coli_v4_np_test_predict(STORE, &config, 4, hidden, 0);
    float conf[16];
    int got = 0;
    for (int waited = 0; waited < 5000 && (got = coli_v4_np_test_confs(conf, 16)) < 8; waited++)
        usleep(1000);
    Queue q = snap();
    CHECK(got == 8 && !strcmp(shape(), "5:8"), "router %d/%d queue: n %d %s", experts, biased, got,
          shape());
    config.n_routed_experts = 256;
    layers[5].plan.tensor_count = 0;
    if (got != 8) return;
    /* expected ranking and confidences by sqrt(softplus) */
    int order[8], rises = 0;
    float score[8], top = 0.0f;
    for (int i = 0; i < 8; i++) {
        int best = -1;
        float best_key = 0.0f;
        for (int r = 0; r < experts; r++) {
            int taken = 0;
            for (int j = 0; j < i; j++) taken |= order[j] == r;
            float s = sqrtf(fmaxf(logit[r], 0.0f) + log1pf(expf(-fabsf(logit[r]))));
            if (!taken && (best < 0 || s + gate_bias[r] > best_key)) {
                best = r;
                best_key = s + gate_bias[r];
            }
        }
        order[i] = best;
        score[i] = sqrtf(fmaxf(logit[best], 0.0f) + log1pf(expf(-fabsf(logit[best]))));
        if (i < TOPK) top += score[i];
        rises += i > 0 && score[i] > score[i - 1];
    }
    CHECK(biased ? rises > 0 : rises == 0, "router %d/%d: %d confidence rises by rank", experts,
          biased, rises);
    for (int i = 0; i < 8; i++) {
        CHECK(q.expert[i] == order[i] && order[i] < experts, "router %d/%d depth-1 rank %d: %d, want %d",
              experts, biased, i, q.expert[i], order[i]);
        CHECK(fabsf(conf[i] - score[i] / top) < 1e-5f, "router %d/%d depth-1 conf %d: %g, want %g",
              experts, biased, i, conf[i], score[i] / top);
    }
}

/* A one-layer expert checkpoint like tests/test_deepseek_v4.c's: per expert
 * w1/w2/w3, an E8M0 [1,1] scale and an I8 [1,16] weight, one 51-byte record */
static int write_experts(const char *path, int experts) {
    static const char *m[3] = {"w1", "w2", "w3"};
    static char header[16384];
    static unsigned char payload[64 * 51];
    if (experts > 64) return -1;
    int used = snprintf(header, sizeof header, "{");
    for (int e = 0; e < experts; e++)
        for (int i = 0; i < 3; i++) {
            size_t s = (size_t)e * 51 + i, w = (size_t)e * 51 + 3 + i * 16;
            used += snprintf(header + used, sizeof header - used,
                             "%s\"layers.0.ffn.experts.%d.%s.scale\":{\"dtype\":\"F8_E8M0\","
                             "\"shape\":[1,1],\"data_offsets\":[%zu,%zu]},"
                             "\"layers.0.ffn.experts.%d.%s.weight\":{\"dtype\":\"I8\","
                             "\"shape\":[1,16],\"data_offsets\":[%zu,%zu]}",
                             used > 1 ? "," : "", e, m[i], s, s + 1, e, m[i], w, w + 16);
        }
    used += snprintf(header + used, sizeof header - used, "}");
    for (int i = 0; i < experts * 51; i++) payload[i] = (unsigned char)i;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t length = (uint64_t)used;
    int bad = fwrite(&length, 8, 1, f) != 1 || fwrite(header, (size_t)used, 1, f) != 1 ||
              fwrite(payload, 51, (size_t)experts, f) != (size_t)experts;
    return fclose(f) || bad ? -1 : 0;
}

/* A reader on a real store (no GPU tier): gating admits a guess at its first
 * read, so one RAM holds is looked up uncharged even below the threshold, and
 * one past the threshold or the budget reads nothing. */
static void test_work(void) {
    char dir[] = "np-work-XXXXXX", path[64], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    if (!mkdtemp(dir)) { CHECK(0, "mkdtemp"); return; }
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    ColiExpertStore *store = NULL;
    ColiDeepSeekV4ExpertStoreOptions options = {dir, 1, 7, 6 * 51, -1, UINT64_MAX, 0, 0};
    if (write_experts(path, 7) ||
        coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof error)) {
        CHECK(0, "fixture store: %s", error);
        remove(path);
        rmdir(dir);
        return;
    }
    ColiExpertStoreStats s0, s;
    unsigned long long g[NG];
    uint32_t stamp;
    uint64_t spent;
#define K(e) ((ColiExpertKey){0, (e)})
    ColiExpertView view;
    CHECK(!coli_expert_lookup(store, K(0), &view), "demand lookup");
    coli_expert_release(store, &view);
    CHECK(coli_v4_expert_store_resident(store, K(0)) == 1 &&
          coli_v4_expert_store_resident(store, K(1)) == 0, "fixture residency");
    coli_v4_np_test_gate(0.5f, 51.0 / 1048576.0);   /* one record per forward */
    store->ops->stats(store, &s0);
    /* in RAM, below the threshold: looked up, nothing charged or gated */
    coli_v4_np_test_work(store, K(0), 0, 0.1f, 60);
    store->ops->stats(store, &s);
    gate_state(g, &stamp, &spent);
    CHECK(s.requests == s0.requests + 1 && s.hits == s0.hits + 1 && s.bytes_read == s0.bytes_read,
          "in RAM: requests %llu hits %llu", (unsigned long long)(s.requests - s0.requests),
          (unsigned long long)(s.hits - s0.hits));
    CHECK(g[ADMITTED] == 0 && g[GATED_CONF] == 0 && spent == 0, "in RAM charged or gated");
    /* outside RAM, below the threshold: no read */
    coli_v4_np_test_work(store, K(1), 0, 0.1f, 60);
    s0 = s;
    store->ops->stats(store, &s);
    gate_state(g, &stamp, &spent);
    CHECK(s.requests == s0.requests && g[GATED_CONF] == 1 &&
          coli_v4_expert_store_resident(store, K(1)) == 0, "below the threshold read");
    /* outside RAM, confident: read, one record charged */
    coli_v4_np_test_work(store, K(2), 0, 0.9f, 60);
    s0 = s;
    store->ops->stats(store, &s);
    gate_state(g, &stamp, &spent);
    CHECK(s.requests == s0.requests + 1 && s.misses == s0.misses + 1 && g[ADMITTED] == 1 &&
          spent == 51 && coli_v4_expert_store_resident(store, K(2)) == 1,
          "confident read: spent %llu", (unsigned long long)spent);
    /* the forward's budget is spent: no read; one RAM holds still goes, free */
    coli_v4_np_test_work(store, K(3), 0, 0.9f, 60);
    coli_v4_np_test_work(store, K(2), 0, 0.9f, 60);
    s0 = s;
    store->ops->stats(store, &s);
    gate_state(g, &stamp, &spent);
    CHECK(s.requests == s0.requests + 1 && s.hits == s0.hits + 1 && g[GATED_BUDGET] == 1 &&
          spent == 51 && coli_v4_expert_store_resident(store, K(3)) == 0,
          "past the budget: requests %llu budget %llu", (unsigned long long)(s.requests - s0.requests),
          g[GATED_BUDGET]);
    /* the next forward: a new budget */
    coli_v4_np_test_work(store, K(3), 0, 0.9f, 61);
    gate_state(g, &stamp, &spent);
    CHECK(stamp == 61 && spent == 51 && g[ADMITTED] == 2 &&
          coli_v4_expert_store_resident(store, K(3)) == 1, "next forward: spent %llu",
          (unsigned long long)spent);
    /* gating off: every guess is read */
    coli_v4_np_test_gate(0.0f, 0.0);
    coli_v4_np_test_work(store, K(4), 0, 0.0f, 61);
    gate_state(g, &stamp, &spent);
    CHECK(coli_v4_expert_store_resident(store, K(4)) == 1 && g[ADMITTED] == 0,
          "gating off");
#undef K
    store->ops->destroy(store);
    remove(path);
    CHECK(rmdir(dir) == 0, "%s left behind (a .coli_usage?)", dir);
}

/* V4_PREFETCH_NO_CREDIT: prefetch lookups do not credit usage, layer_requests,
 * hit, or heat when enabled, while demand lookups still do; the setting is
 * latched at store open. */
static void test_no_credit(void) {
    char dir[] = "np-cred-XXXXXX", path[64], error[256];
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    if (!mkdtemp(dir)) { CHECK(0, "mkdtemp"); return; }
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    if (write_experts(path, 7)) {
        CHECK(0, "write_experts");
        remove(path);
        rmdir(dir);
        return;
    }
    for (int enabled = 0; enabled <= 1; enabled++) {
        setenv("V4_PREFETCH_NO_CREDIT", enabled ? "1" : "0", 1);
        ColiDeepSeekV4ExpertStoreOptions options = {dir, 1, 7, 7 * 51, -1, UINT64_MAX, 0, 0};
        ColiExpertStore *store = NULL;
        if (coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof error)) {
            CHECK(0, "open credit store: %s", error);
            break;
        }
        /* Changing the env after open must not alter the store's latched policy. */
        setenv("V4_PREFETCH_NO_CREDIT", enabled ? "0" : "1", 1);
        ColiExpertView view;
        coli_v4_expert_lookup_set_prefetch(1);
        for (int e = 0; e < 2; e++) {
            CHECK(!coli_expert_lookup(store, (ColiExpertKey){0, e}, &view), "prefetch lookup %d", e);
            coli_expert_release(store, &view);
        }
        coli_v4_expert_lookup_set_prefetch(0);
        uint64_t usage[7], reqs = 0;
        unsigned char hit[7], heat[7];
        int pins[16];
        CHECK(coli_v4_test_expert_books(store, 0, usage, hit, heat, &reqs, pins) >= 0, "books after prefetch");
        uint64_t credit = enabled ? 0 : 1;
        CHECK(usage[0] == credit && usage[1] == credit && reqs == 2 * credit &&
              hit[0] == credit && hit[1] == credit && heat[0] == credit && heat[1] == credit,
              "no_credit=%d after prefetch: usage=(%llu,%llu) reqs=%llu hit=(%u,%u) heat=(%u,%u)",
              enabled, (unsigned long long)usage[0], (unsigned long long)usage[1],
              (unsigned long long)reqs, hit[0], hit[1], heat[0], heat[1]);
        /* Demand lookup on expert 0 always credits it */
        CHECK(!coli_expert_lookup(store, (ColiExpertKey){0, 0}, &view), "demand lookup 0");
        coli_expert_release(store, &view);
        CHECK(coli_v4_test_expert_books(store, 0, usage, hit, heat, &reqs, pins) >= 0, "books after demand");
        CHECK(usage[0] == credit + 1 && usage[1] == credit && reqs == 2 * credit + 1 &&
              hit[0] == 1 && hit[1] == credit && heat[0] == credit + 1 && heat[1] == credit,
              "no_credit=%d after demand: usage=(%llu,%llu) reqs=%llu",
              enabled, (unsigned long long)usage[0], (unsigned long long)usage[1],
              (unsigned long long)reqs);
        ColiExpertStoreStats st;
        store->ops->stats(store, &st);
        CHECK(st.requests == 3 && st.hits == 1 && st.misses == 2 && st.bytes_read == 102,
              "no_credit=%d stats: req=%llu hit=%llu miss=%llu bytes=%llu",
              enabled, (unsigned long long)st.requests, (unsigned long long)st.hits,
              (unsigned long long)st.misses, (unsigned long long)st.bytes_read);
        store->ops->destroy(store);
    }
    unsetenv("V4_PREFETCH_NO_CREDIT");
    remove(path);
    CHECK(rmdir(dir) == 0, "%s left behind", dir);
}

static void test_tokstart(void) {
    coli_v4_np_test_setup(0, 1, 1);   /* token start implies KEEP */
    coli_v4_np_test_set_forward(20);
    int old[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    coli_v4_np_test_issue(STORE, 42, old, 8, REPLACE, 20);   /* last forward's leftovers */
    int token = 9;
    coli_v4_np_token_start(STORE, &config, token);
    CHECK(!strcmp(shape(), "0:6 1:6 2:6"), "token start queue: %s", shape());
    Queue q = snap();
    for (int i = 0; i < q.n; i++) {
        CHECK(q.stamp[i] == 21, "token start stamp %u", q.stamp[i]);
        CHECK(q.expert[i] == expert_of(q.layer[i], token, i % TOPK), "token start expert");
    }
    unsigned long long qs[NQ];
    int n, ids[16];
    uint32_t stamp;
    for (int l = 0; l < 3; l++) {
        coli_v4_np_test_stats(qs, l, &n, &stamp, ids);
        CHECK(n == TOPK && stamp == 21 && ids[0] == expert_of(l, token, 0),
              "token start publish layer %d: n %d stamp %u", l, n, stamp);
    }
    CHECK(qs[TOKSTART] == 18, "tokstart counter %llu", qs[TOKSTART]);
    /* the forward starts: layer 0 routes; predicting 1 drops layer 0's unstarted
     * reads and does not queue layer 1's list again */
    coli_v4_np_test_set_forward(21);
    coli_v4_np_test_predict(STORE, &config, 0, input, token);
    CHECK(!strcmp(shape(), "1:6 2:6"), "after layer 0: %s", shape());
    coli_v4_np_test_predict(STORE, &config, 1, input, token);
    CHECK(!strcmp(shape(), "2:6"), "after layer 1: %s", shape());
    /* layer 2 routes; 3 is gate-routed (no gate tensor here: nothing published) */
    coli_v4_np_test_predict(STORE, &config, 2, input, token);
    CHECK(!strcmp(shape(), ""), "after layer 2: %s", shape());
    /* the same layers again in this forward (a prefill row, say) with another
     * token: the lists are no longer marked, so they are queued */
    coli_v4_np_test_predict(STORE, &config, 0, input, token + 1);
    CHECK(!strcmp(shape(), "1:6"), "unmarked layer 1: %s", shape());
    q = snap();
    CHECK(q.expert[0] == expert_of(1, token + 1, 0), "unmarked layer 1 is the new token's");
    /* next token start: a different token, leftovers dropped */
    coli_v4_np_token_start(STORE, &config, token + 2);
    CHECK(!strcmp(shape(), "0:6 1:6 2:6"), "second token start: %s", shape());
    coli_v4_np_test_set_forward(22);
    /* a token mismatch (another forward on another token) queues normally */
    coli_v4_np_test_predict(STORE, &config, 0, input, token + 3);
    q = snap();
    CHECK(q.n == 12 && q.layer[0] == 1 && q.expert[0] == expert_of(1, token + 3, 0) &&
          q.layer[6] == 2, "token mismatch: %s", shape());
    /* out-of-vocabulary token: nothing queued, nothing dropped */
    int before = snap().n;
    coli_v4_np_token_start(STORE, &config, VOCAB + 5);
    CHECK(snap().n == before, "bad token start changed the queue");
}

static void test_tokstart_off(void) {
    /* prefetch on, token start off: token start is a no-op, the hash publish
     * at layer 0 replaces as before */
    coli_v4_np_test_setup(0, 0, 0);
    coli_v4_np_test_set_forward(30);
    coli_v4_np_token_start(STORE, &config, 4);
    CHECK(!strcmp(shape(), ""), "token start off queued: %s", shape());
    coli_v4_np_test_set_forward(31);
    coli_v4_np_test_predict(STORE, &config, 0, input, 4);
    CHECK(!strcmp(shape(), "1:6"), "hash publish: %s", shape());
    coli_v4_np_test_predict(STORE, &config, 1, input, 4);
    CHECK(!strcmp(shape(), "2:6"), "hash publish replaces: %s", shape());
}

static uint16_t gate2[256 * 64];
static void test_depth2(void) {
    const int experts = 256, k1 = 6, k2 = 4;
    float logit1[256], logit2[256];
    memset(gate, 0, sizeof gate);
    memset(gate2, 0, sizeof gate2);
    for (int r = 0; r < experts; r++) {
        logit1[r] = (float)((r * 73) % 256 - 128) / 32.0f;
        logit2[r] = (float)((r * 131 + 17) % 256 - 128) / 32.0f;
        uint32_t b1, b2;
        memcpy(&b1, &logit1[r], 4);
        memcpy(&b2, &logit2[r], 4);
        gate[r * 64] = (uint16_t)(b1 >> 16);
        gate2[r * 64] = (uint16_t)(b2 >> 16);
    }
    memset(hidden, 0, sizeof hidden);
    hidden[0] = 1.0f;
    layers[5].plan.tensor_count = 1;
    snprintf(layers[5].plan.tensors[0].name, sizeof layers[5].plan.tensors[0].name,
             "layers.5.ffn.gate.weight");
    layers[5].data[0] = gate;
    layers[6].plan.tensor_count = 1;
    snprintf(layers[6].plan.tensors[0].name, sizeof layers[6].plan.tensors[0].name,
             "layers.6.ffn.gate.weight");
    layers[6].data[0] = gate2;
    config.n_routed_experts = experts;

    coli_v4_np_test_setup(1, 0, 1);
    coli_v4_np_test_gate(0.0f, 0.0);
    coli_v4_np_test_router(k1);
    coli_v4_np_test_depth(2, k2);
    coli_v4_np_test_set_forward(70);
    coli_v4_np_test_predict(STORE, &config, 4, hidden, 0);

    float conf[16];
    int got = 0;
    for (int waited = 0; waited < 5000 && (got = coli_v4_np_test_confs(conf, 16)) < k1 + k2; waited++)
        usleep(1000);
    Queue q = snap();
    CHECK(got == k1 + k2 && !strcmp(shape(), "5:6 6:4"),
          "depth2 queue: n %d %s", got, shape());

    int n2 = 0, ids2[16];
    uint32_t stamp2 = 0;
    coli_v4_np_test_pred2(6, &n2, &stamp2, ids2);
    CHECK(n2 == k2 && stamp2 == 70, "depth2 pred2: n=%d stamp=%u", n2, stamp2);

    /* Expected top-k2 for layer 6 (unbiased) */
    int order2[6];
    float score2[6], top2 = 0.0f;
    for (int i = 0; i < TOPK; i++) {
        int best = -1;
        float best_s = 0.0f;
        for (int r = 0; r < experts; r++) {
            int taken = 0;
            for (int j = 0; j < i; j++) taken |= order2[j] == r;
            float s = sqrtf(fmaxf(logit2[r], 0.0f) + log1pf(expf(-fabsf(logit2[r]))));
            if (!taken && (best < 0 || s > best_s)) {
                best = r;
                best_s = s;
            }
        }
        order2[i] = best;
        score2[i] = best_s;
        top2 += best_s;
    }
    if (got == k1 + k2 && n2 == k2) {
        for (int i = 0; i < k2; i++) {
            CHECK(q.layer[k1 + i] == 6 && q.expert[k1 + i] == order2[i] &&
                  ids2[i] == order2[i], "depth2 rank %d: q=%d pred2=%d want=%d",
                  i, q.expert[k1 + i], ids2[i], order2[i]);
            CHECK(fabsf(conf[k1 + i] - score2[i] / top2) < 1e-5f,
                  "depth2 conf %d: %g want %g", i, conf[k1 + i], score2[i] / top2);
        }
    }

    /* KEEP=1: when layer 5 routes, layer 5's unstarted reads are pruned while
     * layer 6's depth-2 reads stay queued until layer 6's depth-1 replaces them. */
    coli_v4_np_test_issue(STORE, 6, NULL, 0, PRUNE, 70);
    CHECK(!strcmp(shape(), "6:4"), "depth2 keep after layer 5 prune: %s", shape());

    layers[5].plan.tensor_count = 0;
    layers[6].plan.tensor_count = 0;
    coli_v4_np_test_depth(1, 4);
}

int main(void) {
    setup_layers();
    test_default_replace();
    test_keep();
    test_tokstart();
    test_tokstart_off();
    test_conf();
    test_budget();
    test_router(256, 0, 50);
    test_router(256, 1, 52);
    test_depth2();
    test_work();
    test_no_credit();
    if (failures) {
        printf("v4 np queue: %d FAILURES\n", failures);
        return 1;
    }
    printf("v4 np queue: ok\n");
    return 0;
}
