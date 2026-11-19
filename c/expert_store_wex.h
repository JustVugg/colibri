#ifndef COLI_EXPERT_STORE_WEX_H
#define COLI_EXPERT_STORE_WEX_H

/*
 * DeepSeek-V4 WEX (Windows Expert Container) Storage Backend.
 *
 * Provides page-aligned, contiguous single-read Direct I/O streaming
 * on Windows bypassing the NTFS page cache, bounce buffers, and unaligned reads.
 */

#include "expert_store.h"
#include "deepseek_v4_internal.h"
#include "wex_format.h"
/* coli_wex_decode_record / coli_wex_cuda_decode_count live in wex_decode.c */
#include "wex_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Open function registered in the pluggable ExpertStore registry */
int coli_v4_expert_store_open_wex(
    ColiV4Engine *engine,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **output,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* COLI_EXPERT_STORE_WEX_H */
