/* blockdec_host.h -- a GGUF's block formats as the GPU backends hold them (backend_vulkan.c,
 * backend_cuda.cu): fmt 100 + t is ggml block type t, its rows as the GGUF file stores
 * them. F16 (101) and BF16 (130) are blocks of 32 values here (64 bytes, no scale): the
 * same rows, so a 16-bit matrix takes the block formats' GEMVs and tiled GEMMs.
 * Elements and bytes per block by ggml type, and the table their decoders read
 * (shaders/blockdec_table.h's layout: block sizes, ksigns, kvalues, the IQ grids from
 * ggml_grids.h, vq2's codebooks), built once here for both backends so the two read the
 * same words. */
#ifndef BLOCKDEC_HOST_H
#define BLOCKDEC_HOST_H
#include <stdint.h>
#include <string.h>
#include "ggml_grids.h"
#include "shaders/blockdec_table.h"

static const uint16_t BD_BLK[43] = {1,32,32,32,0,0,32,32,32,0,256,256,256,256,256,0,256,256,256,256,32,256,256,256,
                                    0,0,0,0,0,256,32,0,0,0,256,256,0,0,0,32,64,128,64};
static const uint16_t BD_BYTES[43] = {4,64,18,20,0,0,22,24,34,0,84,110,144,176,210,0,66,74,98,50,18,110,82,136,
                                      0,0,0,0,0,56,64,0,0,0,54,66,0,0,0,17,36,18,18};

/* the table's BDT_WORDS words; vq2: its BDT_WORDS - BDT_VQ2 codebook words, or NULL (zeros) */
static inline void bdtab_fill(uint32_t *t, const uint32_t *vq2) {
    memset(t, 0, (size_t)BDT_WORDS * 4);
    if (vq2) memcpy(t + BDT_VQ2, vq2, (size_t)(BDT_WORDS - BDT_VQ2) * 4);
    for (int i = 0; i < 43; i++) { t[BDT_BLK + i] = BD_BLK[i]; t[BDT_BYTES + i] = BD_BYTES[i]; }
    for (int i = 0; i < 128; i++) t[BDT_KSIGNS + i] = GGML_KSIGNS_IQ2XS[i];
    for (int i = 0; i < 16; i++) {
        t[BDT_IQ4NL + i] = (uint32_t)(int32_t)GGML_KVALUES_IQ4NL[i];
        t[BDT_MXFP4 + i] = (uint32_t)(int32_t)GGML_KVALUES_MXFP4[i];
    }
    /* the grids' bytes as little-endian words, as the decoders read them */
    memcpy(t + BDT_IQ2XXS, GGML_IQ2XXS_GRID, sizeof GGML_IQ2XXS_GRID);
    memcpy(t + BDT_IQ2XS, GGML_IQ2XS_GRID, sizeof GGML_IQ2XS_GRID);
    memcpy(t + BDT_IQ2S, GGML_IQ2S_GRID, sizeof GGML_IQ2S_GRID);
    memcpy(t + BDT_IQ3XXS, GGML_IQ3XXS_GRID, sizeof GGML_IQ3XXS_GRID);
    memcpy(t + BDT_IQ3S, GGML_IQ3S_GRID, sizeof GGML_IQ3S_GRID);
    memcpy(t + BDT_IQ1S, GGML_IQ1S_GRID, sizeof GGML_IQ1S_GRID);
}
#ifdef __cplusplus
static_assert(
#else
_Static_assert(
#endif
               BDT_IQ2XS - BDT_IQ2XXS == sizeof GGML_IQ2XXS_GRID / 4 && BDT_IQ2S - BDT_IQ2XS == sizeof GGML_IQ2XS_GRID / 4 &&
               BDT_IQ3XXS - BDT_IQ2S == sizeof GGML_IQ2S_GRID / 4 && BDT_IQ3S - BDT_IQ3XXS == sizeof GGML_IQ3XXS_GRID / 4 &&
               BDT_IQ1S - BDT_IQ3S == sizeof GGML_IQ3S_GRID / 4 && BDT_VQ2 - BDT_IQ1S == sizeof GGML_IQ1S_GRID / 4 &&
               BDT_WORDS - BDT_VQ2 == 2 * 256 * 8 / 4,
               "shaders/blockdec_table.h does not match ggml_grids.h");
#endif
