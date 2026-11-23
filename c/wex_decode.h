#ifndef COLI_WEX_DECODE_H
#define COLI_WEX_DECODE_H

/*
 * WEX record reconstruction, shared by the engines that read WEX containers
 * (docs/wex.md): the process's WNM2 model registry and the decode of one record
 * into its slab. WNR3 records (codec 5) name a container-level model by id; a
 * container registers its model blob at open, and the decode looks the model up
 * by the id the record carries.
 *
 * The optional CUDA helper (tools/wex_cuda_bridge.cu, `make wex-cuda-helper`)
 * decodes WNR2/WNR3 records on an NVIDIA GPU. It is opened at run time, the
 * first time a model is registered or a neural record decoded, on Windows and
 * Linux, and only when COLI_WEX_CUDA_LIB=<path> names it (kaira_wex_cuda.dll /
 * kaira_wex_cuda.so); unset or "0", no library is looked for and every record
 * decodes on the CPU. coli_wex_decode_record then sends a neural record to the helper
 * and decodes it on the CPU when the helper is absent, lacks the feature, or does
 * not decode it: the bytes are the same either way. On the CPU a WNR3 record goes
 * to the fast decoder of wex_fast_decode.h (AVX2 where CPUID says the CPU has it,
 * NEON on ARM64) unless COLI_WEX_FAST_DECODE=0; to the reference decoder of
 * wex_neural_rans.h then, on other CPUs and for WNR2 records. Same bytes, same
 * verdicts. The coli_wex_cuda_* calls below expose the helper's other entry points.
 * COLI_WEX_STATS=1 prints one "[wex] stats: helper=... fast=... reference=..." line
 * at exit: which decoder took the records.
 *
 * Linkage: an engine that is one translation unit includes wex_decode.c with
 * COLI_WEX_DECODE_API defined as `static inline`, so every engine has its own
 * registry and no two engines linked into one library define the same symbol.
 * Built on its own (cc -c wex_decode.c) the functions are external.
 */

#include <stddef.h>
#include <stdint.h>

#include "wex_format.h"

#ifndef COLI_WEX_DECODE_API
#define COLI_WEX_DECODE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Register a WNM2 model blob (WEX_NR2_MODEL_BYTES). The registry keeps its own
 * copy until the process exits; registering the same model again is a no-op.
 * Returns 0 and the model id, or -1 for an invalid blob, a different blob with
 * the same id, or a full registry (16 models). */
COLI_WEX_DECODE_API int coli_wex_register_model(const unsigned char *blob, size_t bytes, uint32_t *id);

/* The model blob of a WEX4 container, read from the file into `stored` (its
 * `bytes` followed by the zero padding to `stored_bytes`): checked against the
 * SHA-256 the container's index gives it (sha) and its padding, then
 * registered. 0 when it is the index's blob and its id is `id`; -1 otherwise
 * (nothing registered). */
COLI_WEX_DECODE_API int coli_wex4_register_model(const unsigned char *stored, size_t bytes, size_t stored_bytes,
                                                 const unsigned char sha[32], uint32_t id);

/* Decode one record in place. The caller has read the record's total_bytes at
 * slab + wex_record_read_offset(entry) (0 for a RAW record); capacity is the
 * slab's size, at least the entry's aligned decoded size. On 0 the slab starts
 * with the decoded record (scales, then weights, as wex_entry_raw_sizes lays
 * them out); on -1 (an invalid record, a stream whose check fails, a model that
 * was not registered, no memory) the decoded bytes are not there and the caller
 * must not use the slab. WNR3 records: the caller checks the owner and the
 * canonical-zero flag first (wex_nr3_owner_ok). */
COLI_WEX_DECODE_API int coli_wex_decode_record(const WexEntry *entry, unsigned char *slab, size_t capacity);

/* Decode one neural record (WNR2 / WNR3) straight into the caller's buffers, with
 * no slab: `record` holds its entry->total_bytes as stored (no alignment needed,
 * not modified); matrix m's scales go to scales[m] (scale_shapes[m] bytes), its
 * weights to weights[m] (weight_shapes[m] bytes). The same decoders, checks and
 * bytes as coli_wex_decode_record, without the decoded page and the copy out of
 * it (the CUDA helper, where it takes the record, still decodes through a slab of
 * its own). 0 decoded; -1 as coli_wex_decode_record, and then the buffers may
 * hold part of a decode: the caller must not use them. WNR3 records: the caller
 * checks the owner and the canonical-zero flag first (wex_nr3_owner_ok). */
COLI_WEX_DECODE_API int coli_wex_decode_record_to(const WexEntry *entry, const unsigned char *record,
                                                  unsigned char *const scales[3], unsigned char *const weights[3]);

/* The streams the fast decoder interleaves in a record (1, 2, 3, 4, 6 or 8):
 * COLI_WEX_FAST_LANES when set to one of them, else the count measured on the
 * process's first WNR3 records, 0 while that is still being measured. */
COLI_WEX_DECODE_API int coli_wex_fast_lanes(void);

/* The CPU decoder of WNR3 records now: "AVX2" or "NEON" (the fast decoder), or
 * "reference" (COLI_WEX_FAST_DECODE=0, or no kernel for this CPU). And the WNR3
 * records the fast decoder has decoded so far. */
COLI_WEX_DECODE_API const char *coli_wex_cpu_decoder(void);
COLI_WEX_DECODE_API uint64_t coli_wex_fast_decode_count(void);

/* Every container open path calls this once: 1 (and a stderr line, prefixed by
 * `who`) when a canonical-zero container (header flag 64; for WEX4, any record
 * with the value-exact flag) must be refused because DSV4_CUDA_MOE_MASKED != 0
 * selects the masked MoE path, which is not proven value-exact; otherwise 0
 * (logging "value-exact" for such a container). */
COLI_WEX_DECODE_API int coli_wex_container_refused(uint32_t header_flags, const char *who);

/* ---- the CUDA helper ----------------------------------------------------------
 * Its feature bits (docs/wex.md), 0 when no helper is loaded: 1 decode to device,
 * 2 host register, 4 WNR2 records and the model registry, 8 batch to device, 16
 * WNR3 records, 32 batch to host. Opens the helper on the first call. */
#define COLI_WEX_CUDA_DEVICE 1u
#define COLI_WEX_CUDA_REGISTER 2u
#define COLI_WEX_CUDA_NR2 4u
#define COLI_WEX_CUDA_BATCH 8u
#define COLI_WEX_CUDA_NR3 16u
#define COLI_WEX_CUDA_BATCH_HOST 32u
COLI_WEX_DECODE_API unsigned int coli_wex_cuda_features(void);

/* Records decoded by the helper so far: into host slabs (coli_wex_decode_record and
 * the host batch) and into device buffers. */
COLI_WEX_DECODE_API uint64_t coli_wex_cuda_decode_count(void);
COLI_WEX_DECODE_API uint64_t coli_wex_cuda_device_decode_count(void);

/* One neural record held in host memory (entry->total_bytes at record, page-locked
 * for full speed) decoded straight into caller-owned device buffers on `device`:
 * matrix m's scales to dst_scale[m], its weights to dst_weight[m]. 0 decoded and
 * written; -1 an invalid record or one that fails its checks; 1 no helper, no
 * feature 1 (or 16 for WNR3), its model not registered, or CUDA unavailable. On
 * any status but 0 the destination is untouched. The calling thread is left on
 * `device`. */
COLI_WEX_DECODE_API int coli_wex_cuda_decode_to_device(const WexEntry *entry, const unsigned char *record, int device,
                                                       unsigned char *const dst_scale[3],
                                                       unsigned char *const dst_weight[3]);

/* n records in one call (feature 8 / 32): record i's matrix m to dst_scale[3*i+m] /
 * dst_weight[3*i+m] (device), or its decoded slab (scales then weights, the layout
 * coli_wex_decode_record leaves) to the host buffer dst[i], which may overlap record
 * i itself but no other record. status[i] as coli_wex_cuda_decode_to_device; a
 * destination whose status is not 0 is untouched, except after a CUDA failure in
 * the host batch's copy back, so a caller treats such a slot as empty and decodes
 * that record again on its own. Returns the number of records whose status is not
 * 0: all n, each status 1, when the feature is missing. */
COLI_WEX_DECODE_API int coli_wex_cuda_decode_batch_to_device(int n, const WexEntry *const *entries,
                                                             const unsigned char *const *records, int device,
                                                             unsigned char *const *dst_scale,
                                                             unsigned char *const *dst_weight, int *status);
COLI_WEX_DECODE_API int coli_wex_cuda_decode_batch_to_host(int n, const WexEntry *const *entries,
                                                           const unsigned char *const *records, int device,
                                                           unsigned char *const *dst, int *status);

/* Page-lock a staging buffer through the helper's CUDA runtime (feature 2): 0 done,
 * 1 not available or refused, -1 a NULL pointer or zero bytes. */
COLI_WEX_DECODE_API int coli_wex_cuda_host_register(void *pointer, size_t bytes);
COLI_WEX_DECODE_API int coli_wex_cuda_host_unregister(void *pointer);

#ifdef __cplusplus
}
#endif

#endif /* COLI_WEX_DECODE_H */
