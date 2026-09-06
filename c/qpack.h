/* qpack.h -- strict reader for Swiftlet qpack v1 expert containers. */
#ifndef COLI_QPACK_H
#define COLI_QPACK_H

#include <stddef.h>
#include <stdint.h>

#include "affine_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_QPACK_MANIFEST_VERSION 1
#define COLI_QPACK_PAGE_ALIGNMENT 16384u

typedef struct {
    char *name;
    char *dtype;
    size_t *shape;
    size_t rank;
    uint64_t offset;
    uint64_t size;
} ColiQpackSection;

typedef struct {
    size_t expert_count;
    size_t layer_count;
    size_t expert_stride;
    ColiQpackSection *sections;
    size_t section_count;
    unsigned char *linear_layers;
} ColiQpackLayout;

typedef struct {
    ColiQpackLayout layout;
    int quant_bits;
    int quant_group_size;
    char *packed_dir;
    int *layer_fds;
} ColiQpackReader;

/* Opens and validates manifest.json, packed_experts/layout.json, and every
 * fixed-stride layer file. The reader owns all returned metadata until close.
 * Do not reopen a live reader. Close is safe after either success or failure.
 *
 * Every layer descriptor is opened and fstat-validated before success. The
 * resulting reader is immutable and concurrent expert reads are safe because
 * they use explicit-offset pread. Close must not race with a read.
 */
int coli_qpack_open(ColiQpackReader *reader, const char *container_dir,
                    char *error, size_t error_capacity);
void coli_qpack_close(ColiQpackReader *reader);

const ColiQpackSection *
coli_qpack_find_section(const ColiQpackReader *reader, const char *name);

/* Reads exactly one expert_stride-byte blob with one pread attempt. */
int coli_qpack_read_expert(const ColiQpackReader *reader,
                           size_t layer, size_t expert,
                           void *buffer, size_t buffer_size,
                           char *error, size_t error_capacity);

/* Resolves projection.{weight,scales,biases} inside a caller-owned expert
 * blob and constructs Colibri's checked MLX affine descriptor. The view and
 * its pointers remain valid only as long as the blob remains valid.
 */
int coli_qpack_affine_view(const ColiQpackReader *reader,
                           const void *expert_blob, size_t blob_size,
                           const char *projection,
                           ColiAffineQuantizedView *view,
                           char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
