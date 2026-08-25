#ifndef COLIBRI_SEGMENT_CONFORMANCE_FIXTURES_H
#define COLIBRI_SEGMENT_CONFORMANCE_FIXTURES_H

#include "../segment_runtime.h"

#include <stddef.h>
#include <stdint.h>

/*
 * Dependency-free contract fixtures for every Colibri model family.
 *
 * These are deliberately not model adapters and do not execute model math.
 * They model the distinct remote-state topologies so the common Segment ABI
 * cannot accidentally collapse to the conventional-KV OLMoE case. A real
 * adapter must later pass the same lifecycle checks with its generated tiny
 * checkpoint and its own numerical oracle.
 */
typedef enum {
    COLI_SEGMENT_FIXTURE_KV             = UINT32_C(1) << 0,
    COLI_SEGMENT_FIXTURE_SLIDING_RING   = UINT32_C(1) << 1,
    COLI_SEGMENT_FIXTURE_MLA_LATENT     = UINT32_C(1) << 2,
    COLI_SEGMENT_FIXTURE_DSA_INDEXER    = UINT32_C(1) << 3,
    COLI_SEGMENT_FIXTURE_CONVOLUTION    = UINT32_C(1) << 4,
    COLI_SEGMENT_FIXTURE_RECURRENT      = UINT32_C(1) << 5,
    COLI_SEGMENT_FIXTURE_ATTN_RESIDUAL  = UINT32_C(1) << 6,
    COLI_SEGMENT_FIXTURE_MHC            = UINT32_C(1) << 7,
    COLI_SEGMENT_FIXTURE_COMPRESSOR     = UINT32_C(1) << 8,
    COLI_SEGMENT_FIXTURE_DEVICE_CACHE   = UINT32_C(1) << 9,
} ColiSegmentFixtureState;

typedef struct {
    const char *family_id;
    const char *display_name;
    const char *state_schema;
    const char *state_description;
    const char *tiny_generator;
    uint32_t state_kinds;
    uint32_t state_lanes;
    uint32_t state_width;
    uint32_t num_layers;
    uint32_t max_context_tokens;
    uint32_t fixture_tag;
} ColiSegmentConformanceFixture;

size_t coli_segment_conformance_fixture_count(void);
const ColiSegmentConformanceFixture *
coli_segment_conformance_fixture_at(size_t index);

/* Explicit registration mirrors the consumer lifecycle required by the
 * production ABI and remains portable to MSVC. */
int coli_segment_conformance_register_fixtures(void);

/* Run lifecycle, isolation, ordering, range, snapshot and restore checks for
 * one registered fixture. Returns zero on success and writes a diagnostic on
 * failure. */
int coli_segment_conformance_run_fixture(
    const ColiSegmentConformanceFixture *fixture,
    char *error, size_t error_size);

#endif /* COLIBRI_SEGMENT_CONFORMANCE_FIXTURES_H */
