/* The two CUDA devices the dsv4 CUDA tests run on: DSV4_TEST_DEVICES="a,b"
 * (ordinals relative to CUDA_VISIBLE_DEVICES), default "0,1". */
#ifndef DSV4_TEST_DEVICES_H
#define DSV4_TEST_DEVICES_H

#include <stdio.h>
#include <stdlib.h>
#include "../deepseek_v4_gpu_devices.h"

static inline void dsv4_test_devices(int devices[2]) {
    const char *text = getenv("DSV4_TEST_DEVICES");
    int list[COLI_V4_GPU_MAX_DEVICES];
    if (coli_v4_gpu_devices_parse(text ? text : "0,1", list) != 2) {
        fprintf(stderr, "DSV4_TEST_DEVICES must name two CUDA devices, e.g. 0,1\n");
        exit(2);
    }
    devices[0] = list[0];
    devices[1] = list[1];
}

#endif
