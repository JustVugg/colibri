/* The values qwen38 sets for the MTP head on an integrated GPU with many compute units
 * (q38_mtp_igpu_defaults), the check that decides them and the RAM the expert tier reserves
 * for them, with no model and no device. A value is not set when the matching variable
 * (Q38_MTP_DRAFTS, Q38_MTP_CAP) is set. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

enum { LAYERS = 3, EXPERTS = 8, LAYER_CAP = 2, EXPERT_BYTES = 100, RAM_EXPERT = 1000 };

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* A model with an MTP head whose cache has LAYER_CAP slots and whose experts take EXPERT_BYTES. */
static void head_model(Model *m, LCache *cache) {
    memset(m, 0, sizeof *m);
    m->mtp = 1; m->mtp_drafts = 2; m->mtp_expert_bytes = EXPERT_BYTES;
    m->c.layers = LAYERS; m->c.experts = EXPERTS;
    m->cache = cache;
    CHECK(!q38_cache_slots_set(&cache[LAYERS], LAYER_CAP), "the starting cache");
}

static void settings(const char *drafts, const char *cap) {
    if (drafts) setenv("Q38_MTP_DRAFTS", drafts, 1); else unsetenv("Q38_MTP_DRAFTS");
    if (cap) setenv("Q38_MTP_CAP", cap, 1); else unsetenv("Q38_MTP_CAP");
}

static void defaults(const char *what, const char *drafts, const char *cap, int want_drafts, int want_cap) {
    Model m;
    LCache cache[LAYERS + 1] = {0};
    head_model(&m, cache);
    settings(drafts, cap);
    q38_mtp_igpu_defaults(&m);
    CHECK(m.mtp_drafts == want_drafts, "%s: %d drafts per verify, want %d", what, m.mtp_drafts, want_drafts);
    CHECK(cache[LAYERS].cap == want_cap && cache[LAYERS].slots, "%s: head cache of %d, want %d", what, cache[LAYERS].cap, want_cap);
    free(cache[LAYERS].slots);
}

static void filled_cache(void) {
    Model m;
    LCache cache[LAYERS + 1] = {0};
    head_model(&m, cache);
    Slot *slots = cache[LAYERS].slots;
    cache[LAYERS].n = 1;
    settings(NULL, NULL);
    q38_mtp_igpu_defaults(&m);
    CHECK(cache[LAYERS].cap == LAYER_CAP && cache[LAYERS].slots == slots, "a cache in use is resized");
    free(slots);
}

static void wanted(void) {
    Model m = {0};
    m.mtp = 1;
    CHECK(q38_mtp_igpu_wanted(&m, 1, 1), "head, many units and the chain on");
    CHECK(!q38_mtp_igpu_wanted(&m, 0, 1), "fewer units");
    CHECK(!q38_mtp_igpu_wanted(&m, 1, 0), "chain off or for prompts only");
    m.mtp = 0;
    CHECK(!q38_mtp_igpu_wanted(&m, 1, 1), "no head");
}

static void reserve(void) {
    Model m;
    LCache cache[LAYERS + 1] = {0};
    head_model(&m, cache);
    size_t layers = (size_t)RAM_EXPERT * LAYER_CAP * LAYERS;
    size_t more = (size_t)(EXPERTS - LAYER_CAP) * EXPERT_BYTES;
    settings(NULL, NULL);
    CHECK(q38_vk_tier_ram_reserve(&m, LAYER_CAP, RAM_EXPERT, 1) == layers + more, "the slots the defaults add");
    CHECK(q38_vk_tier_ram_reserve(&m, LAYER_CAP, RAM_EXPERT, 0) == layers, "without the defaults");
    settings(NULL, "5");
    CHECK(q38_vk_tier_ram_reserve(&m, LAYER_CAP, RAM_EXPERT, 1) == layers, "Q38_MTP_CAP set");
    settings(NULL, NULL);
    cache[LAYERS].cap = EXPERTS;
    CHECK(q38_vk_tier_ram_reserve(&m, LAYER_CAP, RAM_EXPERT, 1) == layers, "a slot for every expert already");
    cache[LAYERS].cap = LAYER_CAP;
    free(cache[LAYERS].slots);
}

int main(void) {
    defaults("no setting", NULL, NULL, 1, EXPERTS);
    defaults("Q38_MTP_DRAFTS", "3", NULL, 2, EXPERTS);
    defaults("Q38_MTP_CAP", NULL, "5", 1, LAYER_CAP);
    defaults("both settings", "3", "5", 2, LAYER_CAP);
    filled_cache();
    wanted();
    reserve();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
