/* kv_persist_dsv4.h — .coli_kv on-disk KV persistence for the DeepSeek V4 engine.
 *
 * Conversations reopen warm across engine restarts: the per-layer attention
 * state (sliding-window kv + compressed slots + recurrent compressor/indexer
 * state) is snapshotted to disk after every turn and restored at serve start,
 * so the next request skips re-prefilling the whole history (prefix reuse).
 *
 * The file is a FULL rewrite via temp+rename (atomic, crash-safe): unlike GLM's
 * per-position MLA records, the DS4 state is a window ring plus recurrent
 * state, not per-position rows, so incremental append is not well-defined.
 * Cost at 8k tokens is ~85-90 MiB (~30-60 ms on NVMe), negligible at ~1 tok/s.
 *
 * Serialization reuses the existing snapshot machinery (ColiV4AttentionSnapshot
 * create/restore, deepseek_v4.c) as the validation layer: restore() checks
 * shapes and presence parity against the live state, so a corrupt or
 * mismatched file can never corrupt a session — it degrades to a fresh start.
 *
 * Include AFTER the snapshot struct definitions in deepseek_v4.c (the
 * ColiV4AttentionSnapshot/Compressor/Indexer structs are private to the TU).
 * KVSAVE=0 disables saving and resume. Same header layout as GLM's kv_persist.h
 * (nrec at int32 index 6) so c/coli's kv_resume_notice reads both magics.
 */
#ifndef KV_PERSIST_DSV4_H
#define KV_PERSIST_DSV4_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kv_prefix.h"

#define V4_KV_MAGIC "COLIDV41\0"

static int v4_kv_enabled(void) {
    const char *e = getenv("KVSAVE");
    return !(e && atoi(e) == 0);
}

static double v4_kv_now(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* FNV-1a over the per-layer compression ratios: a checkpoint whose
 * compress_ratios differ (different model version) gets its file ignored. */
static uint32_t v4_kv_ratio_checksum(const ColiDeepSeekV4Config *c) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < c->num_hidden_layers; i++) {
        int r = c->compress_ratios[i];
        for (int b = 0; b < 4; b++) {
            hash ^= (uint32_t)((r >> (8 * b)) & 0xFF);
            hash *= 16777619u;
        }
    }
    return hash;
}

static int v4_kv_write_all(FILE *f, const void *data, size_t bytes) {
    return bytes == 0 || fwrite(data, 1, bytes, f) == bytes ? 0 : -1;
}

/* Replicate prepare_compressed_state's lazy init through the PUBLIC API: the
 * compressor/indexer/compressed buffers do not exist at session_create (they
 * are created at the first token). The restore path needs them before the
 * first request, so create them here exactly like the forward path does.
 * Idempotent: the engine's own lazy init at the first token sees layer >= 0
 * and just re-binds weights. */
static int v4_kv_prepare_layer(ColiV4Engine *engine, ColiV4Session *session,
                               ColiDeepSeekV4WindowAttentionState *state, int layer_id,
                               char *error, size_t error_size) {
    const ColiDeepSeekV4Config *config = &session->config;
    ColiDeepSeekV4LayerWeights layer;
    if (coli_v4_layer_load(engine, &layer, config, coli_v4_engine_target_index(engine),
                           layer_id, error, error_size)) return -1;
    int ratio = layer.plan.compression_ratio;
    int rc = 0;
    if (ratio && state->layer < 0) {
        state->layer = layer.plan.layer;
        state->ratio = ratio;
        state->compressed_capacity = 16;
        state->compressed = calloc((size_t)state->compressed_capacity * state->head_dim,
                                   sizeof(*state->compressed));
        if (!state->compressed) rc = -1;
        else if (coli_v4_compressor_create(&state->compressor, &layer, config,
                                           error, error_size)) rc = -1;
        if (!rc && ratio == 4 && coli_v4_indexer_create(
                &state->indexer, &layer, config, config->max_position_embeddings,
                error, error_size)) rc = -1;
    }
    if (!rc && state->compressor &&
        coli_v4_compressor_bind_weights(state->compressor, &layer, error, error_size)) rc = -1;
    if (!rc && state->indexer &&
        coli_v4_indexer_bind_weights(state->indexer, &layer, error, error_size)) rc = -1;
    coli_v4_layer_free(engine, &layer);
    return rc;
}

/* ------------------------------------------------------------------ save */
/* Non-static: implemented in the ATTENTION_TRANSACTION object (where the
 * snapshot structs are defined) but called from the GENERATE_STATS object
 * (serve protocol); declared extern before the serve section. */

int v4_kv_save(ColiV4Session *session, const char *path) {
    if (!session || !path || !v4_kv_enabled()) return 0;
    const ColiDeepSeekV4Config *c = &session->config;
    char tmp[4608];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) return -1;
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    int nrec = session->fed.len;
    int32_t header[8] = {
        c->num_hidden_layers, c->sliding_window, c->head_dim,
        c->index_head_dim, c->qk_rope_head_dim, c->vocab_size, nrec,
        (int32_t)v4_kv_ratio_checksum(c)
    };
    int rc = 0;
    if (v4_kv_write_all(f, V4_KV_MAGIC, 8) || v4_kv_write_all(f, header, sizeof(header)))
        rc = -1;
    if (!rc && nrec > 0 &&
        v4_kv_write_all(f, session->fed.fed, (size_t)nrec * sizeof(int))) rc = -1;
    for (int layer = 0; !rc && layer < c->num_hidden_layers; layer++) {
        ColiDeepSeekV4WindowAttentionState *state =
            session->attention ? session->attention[layer] : NULL;
        if (!state) { rc = -1; break; }
        int32_t count = state->compressed_count;
        if (v4_kv_write_all(f, &count, 4)) { rc = -1; break; }
        size_t kv_n = (size_t)state->window_size * state->head_dim;
        if (v4_kv_write_all(f, state->kv, kv_n * sizeof(float))) { rc = -1; break; }
        if (count > 0 && v4_kv_write_all(f, state->compressed,
                                         (size_t)count * state->head_dim * sizeof(float))) {
            rc = -1; break;
        }
        if (state->compressor) {
            ColiV4CompressorSnapshot *snap = NULL;
            int32_t has = 1;
            if (v4_kv_write_all(f, &has, 4) ||
                coli_v4_compressor_snapshot_create(state->compressor, &snap) ||
                coli_v4_compressor_snapshot_write(snap, f)) rc = -1;
            if (snap) coli_v4_compressor_snapshot_destroy(snap);
            if (rc) break;
        } else {
            int32_t has = 0;
            if (v4_kv_write_all(f, &has, 4)) { rc = -1; break; }
        }
        if (state->indexer) {
            ColiV4IndexerSnapshot *snap = NULL;
            int32_t has = 1;
            if (v4_kv_write_all(f, &has, 4) ||
                coli_v4_indexer_snapshot_create(state->indexer, &snap) ||
                coli_v4_indexer_snapshot_write(snap, f)) rc = -1;
            if (snap) coli_v4_indexer_snapshot_destroy(snap);
            if (rc) break;
        } else {
            int32_t has = 0;
            if (v4_kv_write_all(f, &has, 4)) { rc = -1; break; }
        }
    }
    if (rc || fflush(f) != 0) { fclose(f); remove(tmp); return -1; }
    if (fclose(f) != 0) { remove(tmp); return -1; }
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}

/* ------------------------------------------------------------------ load */

int v4_kv_load(ColiV4Engine *engine, ColiV4Session *session, const char *path) {
    if (!session || !path || !v4_kv_enabled()) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    const ColiDeepSeekV4Config *c = &session->config;
    double t0 = v4_kv_now();
    char mg[8];
    int32_t header[8];
    if (fread(mg, 1, 8, f) != 8 || memcmp(mg, V4_KV_MAGIC, 8) ||
        fread(header, 4, 8, f) != 8 ||
        header[0] != c->num_hidden_layers || header[1] != c->sliding_window ||
        header[2] != c->head_dim || header[3] != c->index_head_dim ||
        header[4] != c->qk_rope_head_dim || header[5] != c->vocab_size ||
        header[7] != (int32_t)v4_kv_ratio_checksum(c)) {
        fprintf(stderr, "[KV] ignoring .coli_kv from a different model or version\n");
        fclose(f); return 0;
    }
    int nrec = header[6];
    if (nrec < 1) { fclose(f); return 0; }
    if (nrec >= session->max_prompt_tokens) {
        fprintf(stderr, "[KV] saved conversation (%d tokens) exceeds the context: starting over\n",
                nrec);
        fclose(f); return 0;
    }
    int *ids = malloc((size_t)nrec * sizeof(int));
    if (!ids) { fclose(f); return 0; }
    if (fread(ids, sizeof(int), (size_t)nrec, f) != (size_t)nrec) { free(ids); fclose(f); return 0; }

    int rc = 0;
    char error[512] = {0};
    for (int layer = 0; layer < c->num_hidden_layers; layer++) {
        ColiDeepSeekV4WindowAttentionState *state = session->attention[layer];
        int32_t count = 0;
        if (fread(&count, 4, 1, f) != 1 || count < 0 || count > nrec + 16) { rc = -1; break; }
        if (v4_kv_prepare_layer(engine, session, state, layer, error, sizeof(error))) {
            rc = -1; break;
        }
        while (state->compressed_capacity < count) {   /* grow to fit, like grow_compressed_state */
            int capacity = state->compressed_capacity > 0
                ? state->compressed_capacity * 2 : 16;
            float *grown = realloc(state->compressed,
                                   (size_t)capacity * state->head_dim * sizeof(float));
            if (!grown) { rc = -1; break; }
            state->compressed = grown;
            state->compressed_capacity = capacity;
        }
        if (rc) break;
        ColiV4AttentionSnapshot snap = {0};
        snap.window_size = state->window_size;
        snap.head_dim = state->head_dim;
        snap.compressed_count = count;
        snap.kv = malloc((size_t)state->window_size * state->head_dim * sizeof(float));
        if (!snap.kv ||
            fread(snap.kv, sizeof(float), (size_t)state->window_size * state->head_dim, f) !=
                (size_t)state->window_size * state->head_dim) { rc = -1; break; }
        if (count > 0) {
            snap.compressed = malloc((size_t)count * state->head_dim * sizeof(float));
            if (!snap.compressed ||
                fread(snap.compressed, sizeof(float), (size_t)count * state->head_dim, f) !=
                    (size_t)count * state->head_dim) { rc = -1; break; }
        }
        int32_t has = 0;
        if (fread(&has, 4, 1, f) != 1) { rc = -1; break; }
        if (has) {
            if (!state->compressor ||
                coli_v4_compressor_snapshot_read(f, &snap.compressor)) { rc = -1; break; }
        }
        if (fread(&has, 4, 1, f) != 1) { rc = -1; break; }
        if (has) {
            int indexer_count = 0;
            if (!state->indexer ||
                coli_v4_indexer_snapshot_read(f, c->index_head_dim, &snap.indexer,
                                              &indexer_count) ||
                coli_v4_indexer_grow(state->indexer, indexer_count,
                                     error, sizeof(error))) { rc = -1; break; }
        }
        if (rc) break;
        if (coli_v4_attention_snapshot_restore(state, &snap)) rc = -1;
        free(snap.kv);
        free(snap.compressed);
        if (snap.compressor) coli_v4_compressor_snapshot_destroy(snap.compressor);
        if (snap.indexer) coli_v4_indexer_snapshot_destroy(snap.indexer);
        if (rc) break;
    }
    if (rc) {
        free(ids); fclose(f);
        fprintf(stderr, "[KV] saved conversation ignored (restore failed)\n");
        return 0;
    }
    fclose(f);
    /* Commit the token ids only after every layer restored: a partial commit
     * would claim coverage the states do not have (kv_prefix INVARIANT). */
    if (!kv_prefix_alloc(&session->fed, nrec)) { free(ids); return 0; }
    memcpy(session->fed.fed, ids, (size_t)nrec * sizeof(int));
    session->fed.len = nrec;
    session->fed.tainted = 0;
    free(ids);
    fprintf(stderr, "[KV] resumed conversation from disk: %d tokens in %.1fs (no re-prefill)\n",
            nrec, v4_kv_now() - t0);
    return 1;
}

#endif /* KV_PERSIST_DSV4_H */
