#include "../expert_store.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    int held;
    unsigned active_leases;
    ColiExpertStoreStats stats;
} MockState;

static int mock_lookup(ColiExpertStore *store, ColiExpertKey key,
                       ColiExpertView *view) {
    static const unsigned char weights[4] = {1, 2, 3, 4};
    MockState *state = (MockState *)store->state;
    if (state->held || !view) return -1;
    memset(view, 0, sizeof(*view));
    view->key = key;
    view->gate.format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->gate.scale_format = COLI_SCALE_UE8M0;
    view->gate.data = weights;
    view->gate.data_bytes = sizeof(weights);
    view->lease = state;
    state->held = 1;
    state->active_leases++;
    state->stats.requests++;
    state->stats.misses++;
    state->stats.bytes_read += sizeof(weights);
    return 0;
}

static void mock_release(ColiExpertStore *store, ColiExpertView *view) {
    MockState *state = (MockState *)store->state;
    if (view && view->lease == state) {
        state->held = 0;
        if (state->active_leases) state->active_leases--;
        view->lease = NULL;
    }
}

static int mock_prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                         size_t count) {
    MockState *state = (MockState *)store->state;
    (void)keys;
    state->stats.prefetched += count;
    return (int)count;
}

static void mock_stats(const ColiExpertStore *store,
                       ColiExpertStoreStats *stats) {
    *stats = ((const MockState *)store->state)->stats;
}

static void mock_destroy(ColiExpertStore *store) {
    MockState *state = (MockState *)store->state;
    if (state->active_leases != 0)
        fprintf(stderr, "destroy with active leases=%u\n", state->active_leases);
}

static int view_is_cleared(const ColiExpertView *view) {
    static const ColiExpertView zero;
    return memcmp(view, &zero, sizeof(*view)) == 0;
}

int main(void) {
    static const ColiExpertStoreOps ops = {
        mock_lookup, mock_release, mock_prefetch, mock_stats, mock_destroy
    };
    MockState state = {0};
    ColiExpertStore store = {&ops, &state};
    ColiExpertView view;
    ColiExpertView other;
    ColiExpertKey key = {7, 19};
    ColiExpertStoreStats stats;

    if (coli_expert_lookup(&store, key, &view) != 0) return 1;
    if (view.key.layer != 7 || view.key.expert != 19) return 1;
    if (view.gate.format != COLI_TENSOR_FP4_NATIVE_BLOCK) return 1;
    if (view.gate.scale_format != COLI_SCALE_UE8M0) return 1;

    /* A second lookup while the store is busy must fail and clear *other*. */
    memset(&other, 0x5a, sizeof(other));
    if (coli_expert_lookup(&store, key, &other) == 0) return 1;
    if (!view_is_cleared(&other)) return 1;
    if (view.lease != &state || !state.held) return 1;

    coli_expert_release(&store, &view);
    if (state.held || state.active_leases) return 1;
    if (!view_is_cleared(&view)) return 1;

    /* Double release and release of a zero view are no-ops. */
    coli_expert_release(&store, &view);
    memset(&other, 0, sizeof(other));
    coli_expert_release(&store, &other);
    if (!view_is_cleared(&other)) return 1;

    if (store.ops->prefetch(&store, &key, 1) != 1) return 1;
    store.ops->stats(&store, &stats);
    if (stats.requests != 1 || stats.misses != 1 ||
        stats.prefetched != 1 || stats.bytes_read != 4) return 1;

    store.ops->destroy(&store);
    if (state.active_leases != 0) return 1;
    puts("expert store ops tests: ok");
    return 0;
}
