#ifndef COLI_V4_GPU_DEVICES_H
#define COLI_V4_GPU_DEVICES_H

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#define COLI_V4_GPU_MAX_DEVICES 16

/* CUDA ordinals are relative to CUDA_VISIBLE_DEVICES. Preserve list order. */
static inline int coli_v4_gpu_devices_parse(const char *text, int *devices) {
    int count = 0;
    if (!text || !devices) return -1;
    for (;;) {
        while (isspace((unsigned char)*text)) text++;
        if (!isdigit((unsigned char)*text) || count == COLI_V4_GPU_MAX_DEVICES)
            return -1;
        char *end;
        errno = 0;
        long device = strtol(text, &end, 10);
        if (errno || device > INT_MAX) return -1;
        for (int i = 0; i < count; i++)
            if (devices[i] == device) return -1;
        devices[count++] = (int)device;
        text = end;
        while (isspace((unsigned char)*text)) text++;
        if (!*text) return count;
        if (*text++ != ',') return -1;
    }
}

static inline int coli_v4_gpu_layer_begin(int slot, int count, int layers) {
    return slot * layers / count;
}

static inline int coli_v4_gpu_layer_owner(int layer, int count, int layers) {
    if (count < 1 || count > layers || layer < 0 || layer >= layers) return -1;
    return ((layer + 1) * count - 1) / layers;
}
#endif
