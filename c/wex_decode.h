#ifndef COLI_WEX_DECODE_H
#define COLI_WEX_DECODE_H

/*
 * WEX record reconstruction on the CPU, shared by the engines that read WEX
 * containers (docs/wex.md): the process's WNM2 model registry and the decode
 * of one record into its slab. WNR3 records (codec 5) name a container-level
 * model by id; a container registers its model blob at open, and the decode
 * looks the model up by the id the record carries.
 *
 * Wex-WRN2's c/wex_decode.c also loads an optional CUDA helper that decodes
 * neural records on the GPU; that part is not in this file (series 3 of the
 * WEX upstreaming), so every record here decodes on the CPU, with the
 * reference decoder of wex_neural_rans.h.
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

/* Every container open path calls this once: 1 (and a stderr line, prefixed by
 * `who`) when a canonical-zero container (header flag 64; for WEX4, any record
 * with the value-exact flag) must be refused because DSV4_CUDA_MOE_MASKED != 0
 * selects the masked MoE path, which is not proven value-exact; otherwise 0
 * (logging "value-exact" for such a container). */
COLI_WEX_DECODE_API int coli_wex_container_refused(uint32_t header_flags, const char *who);

#ifdef __cplusplus
}
#endif

#endif /* COLI_WEX_DECODE_H */
