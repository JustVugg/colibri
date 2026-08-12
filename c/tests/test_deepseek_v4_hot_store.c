#include "../compat.h"
#include "../deepseek_v4_internal.h"
#include "../native_quant_fp4_rows16.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef COLI_FP4_ROWS16_KERNEL
int main(void) {
    puts("SKIP DeepSeek-V4 hot ExpertStore tests: rows16 kernel unavailable");
    return 0;
}
#else

enum {
    HOT_EXPERTS = 7,
    HOT_MATRICES = 3,
    HOT_ROWS = 16,
    HOT_PACKED_COLUMNS = 64,
    HOT_SCALE_BYTES = HOT_ROWS * 4,
    HOT_WEIGHT_BYTES = HOT_ROWS * HOT_PACKED_COLUMNS,
    HOT_RECORD_BYTES = HOT_MATRICES * (HOT_SCALE_BYTES + HOT_WEIGHT_BYTES),
    HOT_WORKERS = 8,
    HOT_LOOKUPS_PER_WORKER = 100
};

static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = data;
    while (length) {
        ssize_t count = write(fd, bytes, length);
        if (count <= 0) return -1;
        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int append_json(char *header, size_t capacity, size_t *used,
                       const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    int written = vsnprintf(header + *used, capacity - *used, format,
                            arguments);
    va_end(arguments);
    if (written < 0 || (size_t)written >= capacity - *used) return -1;
    *used += (size_t)written;
    return 0;
}

static int write_hot_fixture(const char *path) {
    char header[8192];
    size_t used = 0;
    uint64_t offset = 0;
    if (append_json(header, sizeof(header), &used, "{") != 0) return -1;
    for (int expert = 0; expert < HOT_EXPERTS; expert++) {
        for (int matrix = 1; matrix <= HOT_MATRICES; matrix++) {
            uint64_t end = offset + HOT_SCALE_BYTES;
            if (append_json(
                    header, sizeof(header), &used,
                    "\"layers.0.ffn.experts.%d.w%d.scale\":{"
                    "\"dtype\":\"F8_E8M0\",\"shape\":[16,4],"
                    "\"data_offsets\":[%llu,%llu]},",
                    expert, matrix, (unsigned long long)offset,
                    (unsigned long long)end) != 0)
                return -1;
            offset = end;
        }
        for (int matrix = 1; matrix <= HOT_MATRICES; matrix++) {
            uint64_t end = offset + HOT_WEIGHT_BYTES;
            if (append_json(
                    header, sizeof(header), &used,
                    "\"layers.0.ffn.experts.%d.w%d.weight\":{"
                    "\"dtype\":\"I8\",\"shape\":[16,64],"
                    "\"data_offsets\":[%llu,%llu]},",
                    expert, matrix, (unsigned long long)offset,
                    (unsigned long long)end) != 0)
                return -1;
            offset = end;
        }
    }
    if (!used || header[used - 1] != ',') return -1;
    header[used - 1] = '}';

    unsigned char payload[HOT_EXPERTS * HOT_RECORD_BYTES];
    for (size_t i = 0; i < sizeof(payload); i++)
        payload[i] = (unsigned char)(i * 31u + 7u);
    uint64_t header_length = used;
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, used) ||
                 write_all(fd, payload, sizeof(payload));
    close(fd);
    return result ? -1 : 0;
}

typedef struct {
    ColiExpertStore *store;
    int failed;
} HotLookupWorker;

static void *lookup_hot_expert(void *argument) {
    HotLookupWorker *worker = argument;
    for (int i = 0; i < HOT_LOOKUPS_PER_WORKER; i++) {
        ColiExpertView view;
        if (coli_expert_lookup(worker->store, (ColiExpertKey){0, 0}, &view) ||
            view.gate.block_rows != 16 || view.down.block_rows != 16 ||
            view.up.block_rows != 16) {
            worker->failed = 1;
            return NULL;
        }
        coli_expert_release(worker->store, &view);
    }
    return NULL;
}

int main(void) {
    char directory[] = "colibri-v4-hot-store-XXXXXX";
    char path[256], usage_path[256], error[256];
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    snprintf(usage_path, sizeof(usage_path), "%s/.coli_usage", directory);
    if (write_hot_fixture(path) != 0) {
        perror("write_hot_fixture");
        unlink(path); rmdir(directory);
        return 1;
    }

    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, HOT_EXPERTS, HOT_EXPERTS * HOT_RECORD_BYTES, -1, 2
    };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        unlink(path); rmdir(directory);
        return 1;
    }

    ColiExpertView held = {0}, deferred = {0}, initial = {0};
    int failed = coli_expert_lookup(store, (ColiExpertKey){0, 0}, &held) ||
        held.gate.block_rows != 1 || held.down.block_rows != 1 ||
        held.up.block_rows != 1 ||
        coli_expert_lookup(store, (ColiExpertKey){0, 0}, &deferred) ||
        deferred.gate.block_rows != 1 || deferred.down.block_rows != 1 ||
        deferred.up.block_rows != 1;
    coli_expert_release(store, &deferred);
    coli_expert_release(store, &held);
    if (!failed)
        failed = coli_expert_lookup(store, (ColiExpertKey){0, 0}, &initial) ||
            initial.gate.block_rows != 16 || initial.down.block_rows != 16 ||
            initial.up.block_rows != 16;
    coli_expert_release(store, &initial);

    pthread_t threads[HOT_WORKERS];
    HotLookupWorker workers[HOT_WORKERS];
    memset(workers, 0, sizeof(workers));
    for (int i = 0; !failed && i < HOT_WORKERS; i++) {
        workers[i].store = store;
        if (pthread_create(&threads[i], NULL, lookup_hot_expert, &workers[i])) {
            failed = 1;
            for (int joined = 0; joined < i; joined++)
                pthread_join(threads[joined], NULL);
            break;
        }
    }
    if (!failed)
        for (int i = 0; i < HOT_WORKERS; i++) {
            pthread_join(threads[i], NULL);
            failed |= workers[i].failed;
        }

    store->ops->destroy(store);
    unlink(path);
    unlink(usage_path);
    rmdir(directory);
    if (failed) return 1;
    puts("DeepSeek-V4 hot ExpertStore tests: ok");
    return 0;
}
#endif
