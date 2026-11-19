#include "expert_store_wex.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <errno.h>
#include <limits.h>

#include "compat.h"

/* The 16 ms pause before the second read of WEX_READ_RETRY=1. Outside Windows colibri's compat.h
 * has no Sleep(), so the store does not take it from there. */
#ifndef _WIN32
#include <time.h>
#endif
static void wex_retry_pause(void) {
#ifdef _WIN32
    Sleep(16);
#else
    struct timespec pause = {0, 16L * 1000000L};
    while (nanosleep(&pause, &pause) != 0 && errno == EINTR) {}
#endif
}

#ifdef _WIN32
typedef HANDLE WexFile;
#define WEX_BAD_FILE INVALID_HANDLE_VALUE
#else
typedef int WexFile;
#define WEX_BAD_FILE (-1)
#endif

static void *wex_alloc_aligned(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, WEX_ALIGN);
#else
    return aligned_alloc(WEX_ALIGN, bytes);
#endif
}

static void wex_free_aligned(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void wex_close_file(WexFile file) {
#ifdef _WIN32
    if (file != WEX_BAD_FILE) CloseHandle(file);
#else
    if (file != WEX_BAD_FILE) close(file);
#endif
}

static int wex_file_bytes(WexFile file, uint64_t *bytes) {
#ifdef _WIN32
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size) || size.QuadPart < 0) return -1;
    *bytes = (uint64_t)size.QuadPart;
#else
    struct stat st;
    if (fstat(file, &st) || st.st_size < 0) return -1;
    *bytes = (uint64_t)st.st_size;
#endif
    return 0;
}

static const char *wex_path_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    if (!slash || (backslash && backslash > slash)) slash = backslash;
    return slash ? slash + 1 : path;
}

/* Every direct read has aligned source, destination and size. Short reads,
 * including a truncated file after open, never publish a cache entry. */
static int wex_read_exact(WexFile file, void *buffer, size_t bytes, uint64_t offset) {
    unsigned char *dst = (unsigned char *)buffer;
#ifdef _WIN32
    HANDLE event = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!event) return -1;
    while (bytes) {
        DWORD chunk = bytes > 0x7ffff000U ? 0x7ffff000U : (DWORD)bytes;
        OVERLAPPED ov = {0};
        ov.Offset = (DWORD)offset;
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent = event;
        ResetEvent(event);
        DWORD got = 0;
        BOOL ok = ReadFile(file, dst, chunk, &got, &ov);
        if (!ok && GetLastError() == ERROR_IO_PENDING)
            ok = GetOverlappedResult(file, &ov, &got, TRUE);
        if (!ok || got != chunk) { CloseHandle(event); return -1; }
        dst += chunk; offset += chunk; bytes -= chunk;
    }
    CloseHandle(event);
#else
    while (bytes) {
        size_t chunk = bytes > 0x7ffff000U ? 0x7ffff000U : bytes;
        ssize_t got = pread(file, dst, chunk, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got != (ssize_t)chunk) return -1;
        dst += chunk; offset += chunk; bytes -= chunk;
    }
#endif
    return 0;
}

typedef struct WexSlot {
    int layer;
    int expert;
    unsigned references;
    uint64_t used;
    unsigned char *slab;
    int lru_prev;
    int lru_next;
    ColiExpertView cached_view; /* Pre-filled view for instant O(1) cache hit */
} WexSlot;

typedef struct WexStoreState {
#ifdef _WIN32
    HANDLE hFilePrimary;
    HANDLE hFileMirror;
#else
    int fd_primary;
    int fd_mirror;
#endif
    int layers;
    int experts_per_layer;
    int slots_total;
    int slots_per_layer;
    uint64_t record_bytes;
    WexEntry *entries;
    uint32_t header_flags;   /* the WNR3 records' canonical-zero check */

    WexSlot *slots;
    int *slot_by_expert; /* [layer * experts_per_layer + expert] -> slot_idx or -1 */
    int *allocated_per_layer; /* [layer] -> next unallocated slot (0..slots_per_layer) for O(1) allocation */
    int *lru_head;
    int *lru_tail;
    uint64_t clock;
    unsigned active_leases;

    pthread_mutex_t mutex;
    ColiExpertStoreStats stats;
    int disk_weights_primary;
    int disk_weights_mirror;
    uint32_t mirror_cut_256; /* Knuth multiplicative hash split threshold in [0..256] */
} WexStoreState;

static inline int wex_expert_index(const WexStoreState *state, int layer, int expert) {
    return (state->experts_per_layer == 256)
        ? ((layer << 8) | expert)
        : (layer * state->experts_per_layer + expert);
}

static inline WexEntry *wex_get_entry(WexStoreState *state, int layer, int expert) {
    if ((unsigned)layer >= (unsigned)state->layers || (unsigned)expert >= (unsigned)state->experts_per_layer)
        return NULL;
    return &state->entries[wex_expert_index(state, layer, expert)];
}

/*
 * Dual-NVMe Knuth Multiplicative Striping:
 * Linear index (layer * experts_per_layer + expert) multiplied by the golden ratio
 * prime 2654435761u, upper 8 bits (16-23) compared against precomputed mirror_cut_256.
 * Guaranteed 1-cycle O(1) evaluation, no modulo or division on hot path,
 * perfectly uniform distribution for any top-k subset, and 100% bit-exact parity
 * with DeepSeek-V4's v4_expert_route / COLI_DISK_WEIGHTS.
 */
static inline int wex_route_to_mirror(const WexStoreState *state, int layer, int expert) {
    if (state->mirror_cut_256 >= 256) return 0;
    uint32_t flat = (uint32_t)wex_expert_index(state, layer, expert);
    uint32_t h = flat * 2654435761u;
    uint32_t hv = (h >> 16) & 255;
    return (hv >= state->mirror_cut_256);
}

#ifdef _WIN32
static inline HANDLE wex_get_file_handle(const WexStoreState *state, int layer, int expert) {
    if (state->hFileMirror != INVALID_HANDLE_VALUE && wex_route_to_mirror(state, layer, expert)) {
        return state->hFileMirror;
    }
    return state->hFilePrimary;
}
#else
static inline int wex_get_file_fd(const WexStoreState *state, int layer, int expert) {
    if (state->fd_mirror >= 0 && wex_route_to_mirror(state, layer, expert)) {
        return state->fd_mirror;
    }
    return state->fd_primary;
}
#endif

static inline void wex_touch_lru(WexStoreState *state, int slot_idx) {
    WexSlot *s = &state->slots[slot_idx];
    int l = s->layer;
    if (state->lru_tail[l] == slot_idx) {
        /* Already at the MRU tail: fast exit, zero list mutations */
        return;
    }
    /* Unlink from current position */
    if (state->lru_head[l] == slot_idx) {
        state->lru_head[l] = s->lru_next;
        if (state->lru_head[l] >= 0)
            state->slots[state->lru_head[l]].lru_prev = -1;
    } else if (s->lru_prev >= 0) {
        state->slots[s->lru_prev].lru_next = s->lru_next;
    }
    if (s->lru_next >= 0) {
        state->slots[s->lru_next].lru_prev = s->lru_prev;
    }

    /* Append to tail */
    s->lru_prev = state->lru_tail[l];
    s->lru_next = -1;
    if (state->lru_tail[l] >= 0)
        state->slots[state->lru_tail[l]].lru_next = slot_idx;
    state->lru_tail[l] = slot_idx;
    if (state->lru_head[l] < 0)
        state->lru_head[l] = slot_idx;
}

static int wex_find_or_evict_slot(WexStoreState *state, int layer) {
    /* 1. Fast O(1) allocation: grab next free slot in layer without linear scan */
    if (state->allocated_per_layer[layer] < state->slots_per_layer) {
        int slot_idx = layer * state->slots_per_layer + state->allocated_per_layer[layer]++;
        return slot_idx;
    }

    /* 2. Fast O(1) LRU eviction: inspect head of LRU list */
    int victim = state->lru_head[layer];
    while (victim >= 0) {
        if (state->slots[victim].references == 0) {
            int old_exp = state->slots[victim].expert;
            if (old_exp >= 0) {
                int old_idx = wex_expert_index(state, layer, old_exp);
                state->slot_by_expert[old_idx] = -1;
            }
            return victim;
        }
        victim = state->slots[victim].lru_next;
    }
    return -1;
}

static void wex_fill_views(WexSlot *slot, const WexEntry *entry, ColiExpertView *view) {
    view->lease = slot;

    /* W1: Gate */
    view->gate.format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->gate.scale_format = COLI_SCALE_UE8M0;
    view->gate.scales = slot->slab + entry->scale_offsets[0];
    view->gate.data = slot->slab + entry->weight_offsets[0];
    view->gate.scale_bytes = (size_t)entry->scale_shapes[0][0] * entry->scale_shapes[0][1];
    view->gate.data_bytes = (size_t)entry->weight_shapes[0][0] * entry->weight_shapes[0][1];
    view->gate.rows = entry->weight_shapes[0][0];
    view->gate.columns = entry->weight_shapes[0][1] * 2;
    view->gate.block_rows = 1;
    view->gate.block_columns = 32;
    view->gate.gpu = NULL;

    /* W2: Down */
    view->down.format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->down.scale_format = COLI_SCALE_UE8M0;
    view->down.scales = slot->slab + entry->scale_offsets[1];
    view->down.data = slot->slab + entry->weight_offsets[1];
    view->down.scale_bytes = (size_t)entry->scale_shapes[1][0] * entry->scale_shapes[1][1];
    view->down.data_bytes = (size_t)entry->weight_shapes[1][0] * entry->weight_shapes[1][1];
    view->down.rows = entry->weight_shapes[1][0];
    view->down.columns = entry->weight_shapes[1][1] * 2;
    view->down.block_rows = 1;
    view->down.block_columns = 32;
    view->down.gpu = NULL;

    /* W3: Up */
    view->up.format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->up.scale_format = COLI_SCALE_UE8M0;
    view->up.scales = slot->slab + entry->scale_offsets[2];
    view->up.data = slot->slab + entry->weight_offsets[2];
    view->up.scale_bytes = (size_t)entry->scale_shapes[2][0] * entry->scale_shapes[2][1];
    view->up.data_bytes = (size_t)entry->weight_shapes[2][0] * entry->weight_shapes[2][1];
    view->up.rows = entry->weight_shapes[2][0];
    view->up.columns = entry->weight_shapes[2][1] * 2;
    view->up.block_rows = 1;
    view->up.block_columns = 32;
    view->up.gpu = NULL;
}

static int wex_load_slot(WexStoreState *state, int slot_idx, int layer, int expert,
                         const WexEntry *entry) {
    WexSlot *slot = &state->slots[slot_idx];
    size_t read_offset = wex_record_read_offset(entry);
    if (read_offset > state->record_bytes ||
        entry->total_bytes > state->record_bytes - read_offset) return -1;
#ifdef _WIN32
    WexFile file = wex_get_file_handle(state, layer, expert);
#else
    WexFile file = wex_get_file_fd(state, layer, expert);
#endif
    if (wex_read_exact(file, slot->slab + read_offset, entry->total_bytes,
                       entry->file_offset) != 0 ||
        !wex_nr3_owner_ok(entry, slot->slab + read_offset, (uint32_t)layer, (uint32_t)expert,
                          state->header_flags) ||
        coli_wex_decode_record(entry, slot->slab, (size_t)state->record_bytes) != 0) {
        const char *wtry = getenv("WEX_READ_RETRY");
        if (wtry && *wtry == '1') {   /* opt-in: default stays exactly upstream */
            wex_retry_pause();
            if (wex_read_exact(file, slot->slab + read_offset, entry->total_bytes,
                               entry->file_offset) == 0 &&
                wex_nr3_owner_ok(entry, slot->slab + read_offset, (uint32_t)layer, (uint32_t)expert,
                                 state->header_flags) &&
                coli_wex_decode_record(entry, slot->slab, (size_t)state->record_bytes) == 0)
                return 0;
        }
        /* The victim was already unmapped. Keep this slot reusable even when
         * the first load into a newly allocated slot fails. */
        slot->expert = -1;
        slot->references = 0;
        wex_touch_lru(state, slot_idx);
        return -1;
    }
    return 0;
}

static int wex_lookup(ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view) {
    if (!store || !store->state || !view) return -1;
    WexStoreState *state = (WexStoreState *)store->state;

    if ((unsigned)key.layer >= (unsigned)state->layers ||
        (unsigned)key.expert >= (unsigned)state->experts_per_layer) {
        memset(view, 0, sizeof(*view));
        return -1;
    }

    pthread_mutex_lock(&state->mutex);
    state->stats.requests++;

    int global_idx = wex_expert_index(state, key.layer, key.expert);
    int slot_idx = state->slot_by_expert[global_idx];

    if (slot_idx >= 0) {
        /* CACHE HIT: 0 ms stall, instant O(1) cached view transfer */
        WexSlot *slot = &state->slots[slot_idx];
        slot->references++;
        slot->used = ++state->clock;
        state->active_leases++;
        wex_touch_lru(state, slot_idx);
        state->stats.hits++;

        *view = slot->cached_view;
        view->key = key;
        view->lease = slot;
        pthread_mutex_unlock(&state->mutex);
        return 0;
    }

    /* CACHE MISS: Single direct I/O read */
    state->stats.misses++;
    const WexEntry *entry = &state->entries[global_idx];
    if (!entry->total_bytes || !WEX_IS_ALIGNED(entry->file_offset, WEX_ALIGN)) {
        pthread_mutex_unlock(&state->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }

    uint32_t aligned_read_bytes = entry->total_bytes;
    if (aligned_read_bytes > state->record_bytes) {
        pthread_mutex_unlock(&state->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }

    slot_idx = wex_find_or_evict_slot(state, key.layer);
    if (slot_idx < 0) {
        pthread_mutex_unlock(&state->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }

    WexSlot *slot = &state->slots[slot_idx];
    assert(WEX_IS_ALIGNED(slot->slab, WEX_ALIGN));

    if (wex_load_slot(state, slot_idx, key.layer, key.expert, entry) != 0) {
        pthread_mutex_unlock(&state->mutex);
        memset(view, 0, sizeof(*view));
        return -1;
    }

    slot->expert = key.expert;
    slot->layer = key.layer;
    slot->references = 1;
    slot->used = ++state->clock;
    state->active_leases++;
    state->slot_by_expert[global_idx] = slot_idx;
    wex_fill_views(slot, entry, &slot->cached_view);
    wex_touch_lru(state, slot_idx);

    state->stats.bytes_read += aligned_read_bytes;

    *view = slot->cached_view;
    view->key = key;
    view->lease = slot;
    pthread_mutex_unlock(&state->mutex);
    return 0;
}

static void wex_release(ColiExpertStore *store, ColiExpertView *view) {
    if (!store || !store->state || !view || !view->lease) return;
    WexStoreState *state = (WexStoreState *)store->state;
    WexSlot *slot = (WexSlot *)view->lease;

    pthread_mutex_lock(&state->mutex);
    if (slot->references > 0) slot->references--;
    if (state->active_leases > 0) state->active_leases--;
    pthread_mutex_unlock(&state->mutex);

    memset(view, 0, sizeof(*view));
}

static int wex_prefetch(ColiExpertStore *store, const ColiExpertKey *keys, size_t count) {
    if (!store || !store->state || !keys || !count) return 0;
    WexStoreState *state = (WexStoreState *)store->state;

    int accepted = 0;
    pthread_mutex_lock(&state->mutex);
    for (size_t i = 0; i < count; i++) {
        if ((unsigned)keys[i].layer >= (unsigned)state->layers ||
            (unsigned)keys[i].expert >= (unsigned)state->experts_per_layer) {
            continue;
        }

        int global_idx = wex_expert_index(state, keys[i].layer, keys[i].expert);
        if (state->slot_by_expert[global_idx] >= 0) continue; /* Already resident */

        const WexEntry *entry = &state->entries[global_idx];
        if (!entry->total_bytes || !WEX_IS_ALIGNED(entry->file_offset, WEX_ALIGN)) continue;

        uint32_t aligned_read_bytes = entry->total_bytes;
        if (aligned_read_bytes > state->record_bytes) continue;

        int slot_idx = wex_find_or_evict_slot(state, keys[i].layer);
        if (slot_idx < 0) continue;

        WexSlot *slot = &state->slots[slot_idx];
        assert(WEX_IS_ALIGNED(slot->slab, WEX_ALIGN));

        if (wex_load_slot(state, slot_idx, keys[i].layer, keys[i].expert, entry) != 0)
            continue;
        slot->expert = keys[i].expert;
        slot->layer = keys[i].layer;
        slot->references = 0;
        slot->used = ++state->clock;
        wex_fill_views(slot, entry, &slot->cached_view);
        state->slot_by_expert[global_idx] = slot_idx;
        wex_touch_lru(state, slot_idx);

        accepted++;
        state->stats.prefetched++;
        state->stats.bytes_read += aligned_read_bytes;
    }
    pthread_mutex_unlock(&state->mutex);
    return accepted;
}

static void wex_stats(const ColiExpertStore *store, ColiExpertStoreStats *out_stats) {
    if (!store || !store->state || !out_stats) return;
    WexStoreState *state = (WexStoreState *)store->state;
    pthread_mutex_lock(&state->mutex);
    *out_stats = state->stats;
    pthread_mutex_unlock(&state->mutex);
}

static void wex_destroy(ColiExpertStore *store) {
    if (!store) return;
    WexStoreState *state = (WexStoreState *)store->state;
    if (state) {
        pthread_mutex_lock(&state->mutex);
        for (int i = 0; i < state->slots_total; i++) {
            if (state->slots[i].slab) {
#ifdef _WIN32
                _aligned_free(state->slots[i].slab);
#else
                free(state->slots[i].slab);
#endif
            }
        }
        free(state->slots);
        free(state->slot_by_expert);
        free(state->allocated_per_layer);
        free(state->lru_head);
        free(state->lru_tail);
#ifdef _WIN32
        _aligned_free(state->entries);
#else
        free(state->entries);
#endif

#ifdef _WIN32
        if (state->hFilePrimary != INVALID_HANDLE_VALUE) CloseHandle(state->hFilePrimary);
        if (state->hFileMirror != INVALID_HANDLE_VALUE) CloseHandle(state->hFileMirror);
#else
        if (state->fd_primary >= 0) close(state->fd_primary);
        if (state->fd_mirror >= 0) close(state->fd_mirror);
#endif
        pthread_mutex_unlock(&state->mutex);
        pthread_mutex_destroy(&state->mutex);
        free(state);
    }
    free(store);
}

static const ColiExpertStoreOps s_wex_ops = {
    wex_lookup,
    wex_release,
    wex_prefetch,
    wex_stats,
    wex_destroy
};

int coli_v4_expert_store_open_wex(
    ColiV4Engine *engine,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **output,
    char *error, size_t error_size) {
    if (engine) {
        /* Delegate directly to coli_v4_expert_store_open_planned so the complete
         * memory tiering, dense layer residency (43 layers loaded once on GPU/RAM),
         * resident LM head cache, 40+ GB expert cache budgeting, and V4HotPolicy
         * are fully activated. The planned store natively reads from the WEX container
         * using 4KB-aligned Direct I/O via v4_read_expert_record. */
        return coli_v4_expert_store_open_planned(engine, config, options, output, error, error_size);
    }
    if (!options || !output || !options->model_dir) {
        if (error && error_size) snprintf(error, error_size, "invalid WEX options");
        return -1;
    }

    *output = NULL;
    char path_primary[1024] = {0};
    char custom_path[1024] = {0};
#ifdef _WIN32
    if (GetEnvironmentVariableA("COLI_WEX_PATH", custom_path, sizeof(custom_path)) == 0) {
        custom_path[0] = '\0';
    }
#else
    const char *env_p = getenv("COLI_WEX_PATH");
    if (env_p) snprintf(custom_path, sizeof(custom_path), "%s", env_p);
#endif

    if (custom_path[0]) {
        snprintf(path_primary, sizeof(path_primary), "%s", custom_path);
    } else {
        size_t len = strlen(options->model_dir);
        if (len > 4 && strcmp(options->model_dir + len - 4, ".wex") == 0) {
            snprintf(path_primary, sizeof(path_primary), "%s", options->model_dir);
        } else {
            snprintf(path_primary, sizeof(path_primary), "%s\\v4_flash_experts.wex", options->model_dir);
        }
    }

#ifdef _WIN32
    HANDLE hFilePrimary = CreateFileA(
        path_primary,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );
    if (hFilePrimary == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        fprintf(stderr, "[WEX] Container '%s' not found (err %lu). Falling back to standard planned store.\n",
                path_primary, (unsigned long)err);
        return coli_v4_expert_store_open_planned(engine, config, options, output, error, error_size);
    }

    HANDLE hFileMirror = INVALID_HANDLE_VALUE;
    const char *mirror_dir = getenv("COLI_MODEL_MIRROR");
    if (!options->skip_mirror_setup && mirror_dir && *mirror_dir) {
        char path_mirror[1024];
        snprintf(path_mirror, sizeof(path_mirror), "%s\\%s", mirror_dir, wex_path_basename(path_primary));
        if (_stricmp(path_primary, path_mirror) != 0) {
            hFileMirror = CreateFileA(
                path_mirror,
                GENERIC_READ,
                FILE_SHARE_READ,
                NULL,
                OPEN_EXISTING,
                FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                NULL
            );
        }
    }
#else
    int fd_primary = open(path_primary, O_RDONLY);
    if (fd_primary < 0) {
        if (error && error_size) snprintf(error, error_size, "cannot open WEX at '%s'", path_primary);
        return -1;
    }
    int fd_mirror = -1;
#endif

    WexFile primary =
#ifdef _WIN32
        hFilePrimary;
    WexFile mirror = hFileMirror;
#else
        fd_primary;
    WexFile mirror = fd_mirror;
    const char *mirror_dir = getenv("COLI_MODEL_MIRROR");
    if (!options->skip_mirror_setup && mirror_dir && *mirror_dir) {
        char path_mirror[1024];
        snprintf(path_mirror, sizeof(path_mirror), "%s/%s", mirror_dir, wex_path_basename(path_primary));
        if (strcmp(path_primary, path_mirror) != 0)
            mirror = open(path_mirror, O_RDONLY);
    }
#endif
    uint64_t file_bytes = 0, mirror_bytes = 0;
    size_t table_bytes = 0, aligned_table_bytes = 0;
    unsigned char *header_page = NULL, *mirror_header = NULL;
    WexEntry *entries = NULL, *mirror_entries = NULL;
    WexStoreState *state = NULL;
    ColiExpertStore *store = NULL;
    int mutex_ready = 0;
    const char *failure = "invalid WEX container";

    if (options->layers < 0 || options->experts_per_layer < 0 ||
        wex_file_bytes(primary, &file_bytes) != 0) goto fail;
    header_page = (unsigned char *)wex_alloc_aligned(WEX_ALIGN);
    if (!header_page) { failure = "out of memory for WEX header"; goto fail; }
    if (wex_read_exact(primary, header_page, WEX_ALIGN, 0) != 0) {
        failure = "failed to read complete WEX header"; goto fail;
    }
    const WexHeader *hdr = (const WexHeader *)header_page;
    if (wex_validate_header(hdr, file_bytes, options->layers,
                            options->experts_per_layer,
                            &table_bytes, &aligned_table_bytes) != 0) goto fail;
    int layers = (int)hdr->num_layers;
    int experts = (int)hdr->num_experts;
    uint64_t expert_bytes = hdr->aligned_expert_bytes;
    uint64_t table_off = hdr->table_offset;
    uint32_t header_flags = hdr->flags;
    size_t entry_count = (size_t)layers * (size_t)experts;
    if (entry_count > INT_MAX || layers > INT_MAX / 24 ||
        (size_t)layers * 24 > SIZE_MAX / sizeof(WexSlot) ||
        expert_bytes > UINT64_MAX / ((uint64_t)layers * 24)) goto fail;
    entries = (WexEntry *)wex_alloc_aligned(aligned_table_bytes);
    if (!entries) { failure = "out of memory for WEX entries"; goto fail; }
    if (wex_read_exact(primary, entries, aligned_table_bytes, table_off) != 0) {
        failure = "failed to read complete WEX table"; goto fail;
    }
    if (wex_validate_table(hdr, entries, file_bytes) != 0) {
        failure = "invalid WEX table/entry"; goto fail;
    }
    if (hdr->flags & (WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3)) {
        uint64_t model_offset, aligned_model;
        uint32_t model_bytes, model_id, registered = 0;
        unsigned char *model;
        int bad;
        if (wex_nr2_model_location(hdr, header_page, &model_offset, &model_bytes, &model_id) ||
            wex_aligned_size(model_bytes, &aligned_model)) {
            failure = "invalid WNR2 model descriptor"; goto fail;
        }
        model = (unsigned char *)wex_alloc_aligned((size_t)aligned_model);
        if (!model) { failure = "out of memory for the WNR2 model"; goto fail; }
        bad = wex_read_exact(primary, model, (size_t)aligned_model, model_offset) != 0 ||
              coli_wex_register_model(model, model_bytes, &registered) != 0 || registered != model_id;
        wex_free_aligned(model);
        if (bad) { failure = "invalid or unreadable WNR2 model"; goto fail; }
    }
    if (coli_wex_container_refused(header_flags, "wex")) {
        failure = "canonical-zero container on DSV4_CUDA_MOE_MASKED"; goto fail;
    }

    /* A matching layout does not identify a generation: WEX2 stores a build
     * identifier in the header page. Compare all 4096 bytes before striping. */
    if (mirror != WEX_BAD_FILE) {
        if (wex_file_bytes(mirror, &mirror_bytes) != 0 || mirror_bytes != file_bytes) {
            failure = "WEX mirror file size differs"; goto fail;
        }
        mirror_header = (unsigned char *)wex_alloc_aligned(WEX_ALIGN);
        mirror_entries = (WexEntry *)wex_alloc_aligned(aligned_table_bytes);
        if (!mirror_header || !mirror_entries) {
            failure = "out of memory validating WEX mirror"; goto fail;
        }
        if (wex_read_exact(mirror, mirror_header, WEX_ALIGN, 0) != 0 ||
            memcmp(header_page, mirror_header, WEX_ALIGN) != 0 ||
            wex_read_exact(mirror, mirror_entries, aligned_table_bytes, table_off) != 0 ||
            memcmp(entries, mirror_entries, aligned_table_bytes) != 0) {
            failure = "WEX mirror header or table differs"; goto fail;
        }
    }
    wex_free_aligned(mirror_header); mirror_header = NULL;
    wex_free_aligned(mirror_entries); mirror_entries = NULL;
    wex_free_aligned(header_page); header_page = NULL;

    state = (WexStoreState *)calloc(1, sizeof(*state));
    store = (ColiExpertStore *)calloc(1, sizeof(*store));
    if (!state || !store) { failure = "out of memory for WEX store"; goto fail; }
#ifdef _WIN32
    state->hFilePrimary = primary;
    state->hFileMirror = mirror;
#else
    state->fd_primary = primary;
    state->fd_mirror = mirror;
#endif
    state->layers = layers;
    state->experts_per_layer = experts;
    state->record_bytes = expert_bytes;
    state->entries = entries;
    state->header_flags = header_flags;
    if (pthread_mutex_init(&state->mutex, NULL) != 0) {
        failure = "failed to initialize WEX cache mutex"; goto fail;
    }
    mutex_ready = 1;
    state->slots_per_layer = 24;
    state->slots_total = state->layers * state->slots_per_layer;
    state->slots = (WexSlot *)calloc((size_t)state->slots_total, sizeof(WexSlot));
    state->slot_by_expert = (int *)malloc(entry_count * sizeof(int));
    state->allocated_per_layer = (int *)calloc((size_t)layers, sizeof(int));
    state->lru_head = (int *)malloc((size_t)layers * sizeof(int));
    state->lru_tail = (int *)malloc((size_t)layers * sizeof(int));
    if (!state->slots || !state->slot_by_expert || !state->allocated_per_layer ||
        !state->lru_head || !state->lru_tail) {
        failure = "out of memory for WEX cache"; goto fail;
    }
    for (size_t i = 0; i < entry_count; i++) state->slot_by_expert[i] = -1;
    for (int l = 0; l < layers; l++) {
        state->lru_head[l] = -1;
        state->lru_tail[l] = -1;
    }
    for (int i = 0; i < state->slots_total; i++) {
        state->slots[i].layer = i / state->slots_per_layer;
        state->slots[i].expert = -1;
        state->slots[i].lru_prev = -1;
        state->slots[i].lru_next = -1;
        state->slots[i].slab = (unsigned char *)wex_alloc_aligned((size_t)state->record_bytes);
        if (!state->slots[i].slab) {
            failure = "out of memory for WEX expert slab"; goto fail;
        }
    }

    /* Dual-NVMe read weights (Crucial T700 C: vs Samsung 970 U:) */
    state->disk_weights_primary = 58;
    state->disk_weights_mirror = 42;
    const char *dw = getenv("COLI_DISK_WEIGHTS");
    if (dw && *dw) {
        int w1 = 0, w2 = 0;
        if (sscanf(dw, "%d%*[, ]%d", &w1, &w2) == 2 && w1 > 0 && w2 > 0) {
            state->disk_weights_primary = w1;
            state->disk_weights_mirror = w2;
        }
    }
    int64_t total_wt = (int64_t)state->disk_weights_primary + state->disk_weights_mirror;
    if (total_wt <= 0) {
        state->disk_weights_primary = 58;
        state->disk_weights_mirror = 42;
        total_wt = 100;
    }
    /* Precompute Knuth hash threshold in [0..256] matching DeepSeek-V4 cut */
    state->mirror_cut_256 = (uint32_t)((state->disk_weights_primary * 256ULL + (total_wt / 2)) / total_wt);
    if (state->mirror_cut_256 > 256) state->mirror_cut_256 = 256;

    state->stats.capacity_bytes = (uint64_t)state->slots_total * state->record_bytes;
    state->stats.resident_bytes = state->stats.capacity_bytes;

    store->ops = &s_wex_ops;
    store->state = state;
    store->gpu = NULL;

    *output = store;
    return 0;

fail:
    if (error && error_size) snprintf(error, error_size, "%s in '%s'", failure, path_primary);
    wex_free_aligned(header_page);
    wex_free_aligned(mirror_header);
    wex_free_aligned(mirror_entries);
    if (state) {
        if (state->slots) {
            for (int i = 0; i < state->slots_total; i++)
                wex_free_aligned(state->slots[i].slab);
        }
        free(state->slots);
        free(state->slot_by_expert);
        free(state->allocated_per_layer);
        free(state->lru_head);
        free(state->lru_tail);
        if (mutex_ready) pthread_mutex_destroy(&state->mutex);
        free(state);
    }
    free(store);
    wex_free_aligned(entries);
    wex_close_file(primary);
    wex_close_file(mirror);
    return -1;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
static void coli_register_wex_backend(void) {
    coli_expert_store_backend_register("wex", coli_v4_expert_store_open_wex);
}
#endif
