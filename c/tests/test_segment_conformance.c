#include "segment_conformance_fixtures.h"

#include <stdio.h>
#include <string.h>

static int expected_family(const char *family_id) {
    static const char *const expected[] = {
        "glm", "glm53", "inkling", "kimi", "olmoe", "qwen36", "deepseek_v4",
    };
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++)
        if (strcmp(expected[i], family_id) == 0) return 1;
    return 0;
}

int main(void) {
    const size_t required_families = 7;
    size_t count = coli_segment_conformance_fixture_count();
    if (count != required_families) {
        fprintf(stderr, "segment conformance requires %zu families, found %zu\n",
                required_families, count);
        return 1;
    }
    if (coli_segment_conformance_register_fixtures() != 0) {
        fprintf(stderr, "could not register all Segment conformance fixtures\n");
        return 1;
    }
    if ((size_t)coli_segment_adapter_count() != count) {
        fprintf(stderr, "not every required family registered a fixture adapter\n");
        return 1;
    }

    uint32_t covered_state = 0;
    for (size_t i = 0; i < count; i++) {
        const ColiSegmentConformanceFixture *fixture =
            coli_segment_conformance_fixture_at(i);
        if (!fixture || !expected_family(fixture->family_id) ||
            !fixture->state_kinds || !fixture->state_lanes ||
            !fixture->state_width || !fixture->num_layers ||
            !fixture->tiny_generator || !*fixture->tiny_generator) {
            fprintf(stderr, "incomplete Segment fixture at index %zu\n", i);
            return 1;
        }
        for (size_t j = 0; j < i; j++) {
            const ColiSegmentConformanceFixture *previous =
                coli_segment_conformance_fixture_at(j);
            if (strcmp(previous->family_id, fixture->family_id) == 0 ||
                strcmp(previous->state_schema, fixture->state_schema) == 0) {
                fprintf(stderr, "duplicate Segment fixture identity: %s\n",
                        fixture->family_id);
                return 1;
            }
        }

        char error[256] = "";
        if (coli_segment_conformance_run_fixture(fixture,
                                                 error, sizeof(error)) != 0) {
            fprintf(stderr, "Segment conformance failed for %s: %s\n",
                    fixture->display_name, error[0] ? error : "unknown error");
            return 1;
        }
        covered_state |= fixture->state_kinds;
        printf("ok %-12s %s\n", fixture->family_id,
               fixture->state_description);
    }

    const uint32_t required_state =
        COLI_SEGMENT_FIXTURE_KV |
        COLI_SEGMENT_FIXTURE_SLIDING_RING |
        COLI_SEGMENT_FIXTURE_MLA_LATENT |
        COLI_SEGMENT_FIXTURE_DSA_INDEXER |
        COLI_SEGMENT_FIXTURE_CONVOLUTION |
        COLI_SEGMENT_FIXTURE_RECURRENT |
        COLI_SEGMENT_FIXTURE_ATTN_RESIDUAL |
        COLI_SEGMENT_FIXTURE_MHC |
        COLI_SEGMENT_FIXTURE_COMPRESSOR |
        COLI_SEGMENT_FIXTURE_DEVICE_CACHE;
    if ((covered_state & required_state) != required_state) {
        fprintf(stderr, "Segment state-topology matrix is incomplete\n");
        return 1;
    }

    puts("segment all-family contract conformance: ok");
    return 0;
}
