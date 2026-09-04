#include "qpack.h"

#include <inttypes.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.qpack\n", argv[0]);
        return 2;
    }

    char error[512] = {0};
    ColiQpackReader reader;
    if (coli_qpack_open(&reader, argv[1], error, sizeof(error))) {
        fprintf(stderr, "qpack validation failed: %s\n", error);
        return 1;
    }

    size_t linear_layers = 0;
    for (size_t layer = 0; layer < reader.layout.layer_count; layer++)
        linear_layers += reader.layout.linear_layers[layer] != 0;

    printf("QPACK v1: %zu layers (%zu linear), %zu experts/layer, "
           "%zu-byte stride\n",
           reader.layout.layer_count, linear_layers,
           reader.layout.expert_count, reader.layout.expert_stride);
    if ((reader.quant_bits == 4 || reader.quant_bits == 8) &&
        reader.quant_group_size > 0) {
        printf("quantization: %d-bit affine, group %d\n",
               reader.quant_bits, reader.quant_group_size);
    } else {
        printf("quantization: unspecified or unsupported by affine adapter\n");
    }

    for (size_t i = 0; i < reader.layout.section_count; i++) {
        const ColiQpackSection *section = &reader.layout.sections[i];
        printf("  %-24s %-5s offset=%" PRIu64 " size=%" PRIu64 " shape=[",
               section->name, section->dtype,
               section->offset, section->size);
        for (size_t dimension = 0; dimension < section->rank; dimension++)
            printf("%s%zu", dimension ? "," : "", section->shape[dimension]);
        puts("]");
    }

    coli_qpack_close(&reader);
    return 0;
}
