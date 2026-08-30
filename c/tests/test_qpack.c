#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../qpack.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../compat.h"

#define CHECK(condition) do {                                                    \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                          \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

static const char valid_layout_bf16[] =
    "{\n"
    "  \"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,\n"
    "  \"linearLayers\":[true,false],\"sections\":[\n"
    "    {\"name\":\"gate_proj.weight\",\"dtype\":\"U32\","
         "\"shape\":[2,1],\"offset\":0,\"size\":8},\n"
    "    {\"name\":\"gate_proj.scales\",\"dtype\":\"BF16\","
         "\"shape\":[2,1],\"offset\":8,\"size\":4},\n"
    "    {\"name\":\"gate_proj.biases\",\"dtype\":\"BF16\","
         "\"shape\":[2,1],\"offset\":12,\"size\":4}\n"
    "  ]\n}";

static const char valid_layout_f16[] =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\","
    "\"shape\":[2,1],\"offset\":0,\"size\":8},"
    "{\"name\":\"gate_proj.scales\",\"dtype\":\"F16\","
    "\"shape\":[2,1],\"offset\":8,\"size\":4},"
    "{\"name\":\"gate_proj.biases\",\"dtype\":\"F16\","
    "\"shape\":[2,1],\"offset\":12,\"size\":4}]}";

static const char valid_layout_f32[] =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\","
    "\"shape\":[2,1],\"offset\":0,\"size\":8},"
    "{\"name\":\"gate_proj.scales\",\"dtype\":\"F32\","
    "\"shape\":[2,1],\"offset\":8,\"size\":8},"
    "{\"name\":\"gate_proj.biases\",\"dtype\":\"F32\","
    "\"shape\":[2,1],\"offset\":16,\"size\":8}]}";

static const char *misaligned_layout =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":8192,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"w\",\"dtype\":\"U32\",\"shape\":[1],"
    "\"offset\":0,\"size\":4}]}";

static const char *overlap_layout =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"a\",\"dtype\":\"U32\",\"shape\":[2],"
    "\"offset\":0,\"size\":8},"
    "{\"name\":\"b\",\"dtype\":\"U32\",\"shape\":[2],"
    "\"offset\":4,\"size\":8}]}";

static const char *duplicate_layout =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"a\",\"dtype\":\"U32\",\"shape\":[1],"
    "\"offset\":0,\"size\":4},"
    "{\"name\":\"a\",\"dtype\":\"U32\",\"shape\":[1],"
    "\"offset\":4,\"size\":4}]}";

static const char *overflow_layout =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"w\",\"dtype\":\"U32\","
    "\"shape\":[4294967296,4294967296],\"offset\":0,\"size\":4}]}";

static const char *bad_affine_shape_layout =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\","
    "\"shape\":[2,1],\"offset\":0,\"size\":8},"
    "{\"name\":\"gate_proj.scales\",\"dtype\":\"BF16\","
    "\"shape\":[1,2],\"offset\":8,\"size\":4},"
    "{\"name\":\"gate_proj.biases\",\"dtype\":\"BF16\","
    "\"shape\":[2,1],\"offset\":12,\"size\":4}]}";

static void make_path(char *output, size_t capacity,
                      const char *root, const char *name) {
    int count = snprintf(output, capacity, "%s/%s", root, name);
    CHECK(count > 0 && (size_t)count < capacity);
}

static void make_dir(const char *path) {
#ifdef _WIN32
    CHECK(_mkdir(path) == 0);
#else
    CHECK(mkdir(path, 0700) == 0);
#endif
}

static void remove_dir(const char *path) {
#ifdef _WIN32
    CHECK(_rmdir(path) == 0);
#else
    CHECK(rmdir(path) == 0);
#endif
}

static void must_write(const char *path, const void *data, size_t size) {
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(data, 1, size, file) == size);
    CHECK(fclose(file) == 0);
}

static void write_layer(const char *root, size_t layer, size_t size,
                        ColiAffineScalarFormat scalar_format) {
    char path[1024], name[64];
    int count = snprintf(name, sizeof(name),
                         "packed_experts/layer_%02zu.bin", layer);
    CHECK(count > 0 && (size_t)count < sizeof(name));
    make_path(path, sizeof(path), root, name);
    unsigned char *bytes = (unsigned char *)calloc(size ? size : 1, 1);
    CHECK(bytes != NULL);
    if (size >= 32768) {
        static const unsigned char weights[] = {
            0x10, 0x32, 0x54, 0x76, 0x11, 0x11, 0x11, 0x11
        };
        static const unsigned char bf16_scalars[] = {
            0x00, 0x3f, 0x80, 0x3f, 0x00, 0x3e, 0x00, 0xbf
        };
        static const unsigned char f16_scalars[] = {
            0x00, 0x38, 0x00, 0x3c, 0x00, 0x30, 0x00, 0xb8
        };
        static const unsigned char f32_scalars[] = {
            0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00, 0xbf
        };
        memcpy(bytes + 16384, weights, sizeof(weights));
        if (scalar_format == COLI_AFFINE_SCALAR_F32)
            memcpy(bytes + 16384 + sizeof(weights), f32_scalars,
                   sizeof(f32_scalars));
        else if (scalar_format == COLI_AFFINE_SCALAR_F16)
            memcpy(bytes + 16384 + sizeof(weights), f16_scalars,
                   sizeof(f16_scalars));
        else
            memcpy(bytes + 16384 + sizeof(weights), bf16_scalars,
                   sizeof(bf16_scalars));
    }
    must_write(path, bytes, size);
    free(bytes);
}

static void write_fixture_sized(const char *root,
                                const void *layout, size_t layout_size,
                                const char *magic, size_t layer_size,
                                int quant_bits, int group_size,
                                ColiAffineScalarFormat scalar_format) {
    char packed[1024], path[1024];
    make_path(packed, sizeof(packed), root, "packed_experts");
    make_dir(packed);
    make_path(path, sizeof(path), root, "packed_experts/layout.json");
    must_write(path, layout, layout_size);
    write_layer(root, 0, layer_size, scalar_format);
    write_layer(root, 1, layer_size, scalar_format);

    char manifest[2048];
    int count = snprintf(
        manifest, sizeof(manifest),
        "{\"magic\":\"%s\",\"version\":1,\"modelName\":\"fixture\","
        "\"sourceCheckpoint\":\"fixture\",\"quantBits\":%d,"
        "\"quantGroupSize\":%d,\"files\":{"
        "\"packed_experts/layout.json\":%zu,"
        "\"packed_experts/layer_00.bin\":%zu,"
        "\"packed_experts/layer_01.bin\":%zu}}",
        magic, quant_bits, group_size, layout_size, layer_size, layer_size);
    CHECK(count > 0 && (size_t)count < sizeof(manifest));
    make_path(path, sizeof(path), root, "manifest.json");
    must_write(path, manifest, (size_t)count);
}

static void write_fixture(const char *root, const char *layout,
                          const char *magic, size_t layer_size,
                          int quant_bits, int group_size,
                          ColiAffineScalarFormat scalar_format) {
    write_fixture_sized(root, layout, strlen(layout), magic, layer_size,
                        quant_bits, group_size, scalar_format);
}

static void cleanup(const char *root) {
    char path[1024];
    for (size_t layer = 0; layer < 2; layer++) {
        char name[64];
        int count = snprintf(name, sizeof(name),
                             "packed_experts/layer_%02zu.bin", layer);
        CHECK(count > 0 && (size_t)count < sizeof(name));
        make_path(path, sizeof(path), root, name);
        CHECK(remove(path) == 0);
    }
    make_path(path, sizeof(path), root, "packed_experts/layout.json");
    CHECK(remove(path) == 0);
    make_path(path, sizeof(path), root, "manifest.json");
    CHECK(remove(path) == 0);
    make_path(path, sizeof(path), root, "packed_experts");
    remove_dir(path);
    remove_dir(root);
}

static void expect_open_failure(const char *layout, const char *magic,
                                size_t layer_size, const char *message) {
    char root[] = "test_qpack_bad_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    write_fixture(root, layout, magic, layer_size, 4, 8,
                  COLI_AFFINE_SCALAR_BF16);
    ColiQpackReader reader;
    char error[512] = {0};
    CHECK(coli_qpack_open(&reader, root, error, sizeof(error)) != 0);
    CHECK(strstr(error, message) != NULL);
    CHECK(reader.layer_fds == NULL && reader.layout.sections == NULL);
    cleanup(root);
}

static void test_valid_container(const char *layout,
                                 ColiAffineScalarFormat scalar_format,
                                 int quant_bits, int group_size,
                                 size_t expected_input,
                                 float expected0, float expected1) {
    char root[] = "test_qpack_ok_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    write_fixture(root, layout, "QPACK", 32768,
                  quant_bits, group_size, scalar_format);

    ColiQpackReader reader;
    char error[512] = {0};
    CHECK(coli_qpack_open(&reader, root, error, sizeof(error)) == 0);
    CHECK(reader.layout.layer_count == 2);
    CHECK(reader.layout.expert_count == 2);
    CHECK(reader.layout.expert_stride == 16384);
    CHECK(reader.quant_bits == quant_bits);
    CHECK(reader.quant_group_size == group_size);
    CHECK(reader.layout.linear_layers[0] == 1);
    CHECK(reader.layout.linear_layers[1] == 0);
    CHECK(reader.layer_fds[0] >= 0 && reader.layer_fds[1] >= 0);

    const ColiQpackSection *section =
        coli_qpack_find_section(&reader, "gate_proj.weight");
    CHECK(section && section->offset == 0 && section->size == 8);
    CHECK(section->rank == 2 && section->shape[0] == 2 &&
          section->shape[1] == 1);

    unsigned char *blob = (unsigned char *)malloc(reader.layout.expert_stride);
    CHECK(blob != NULL);
    CHECK(coli_qpack_read_expert(&reader, 1, 1, blob,
                                 reader.layout.expert_stride,
                                 error, sizeof(error)) == 0);
    CHECK(blob[7] == 0x11);

    ColiAffineQuantizedView view;
    CHECK(coli_qpack_affine_view(&reader, blob, reader.layout.expert_stride,
                                 "gate_proj", &view,
                                 error, sizeof(error)) == 0);
    size_t scalar_bytes = scalar_format == COLI_AFFINE_SCALAR_F32 ? 8 : 4;
    CHECK(view.weights == blob && view.scales == blob + 8 &&
          view.biases == blob + 8 + scalar_bytes);
    CHECK(view.weight_bytes == 8 && view.scale_bytes == scalar_bytes &&
          view.bias_bytes == scalar_bytes);
    CHECK(view.output_dim == 2 && view.input_dim == expected_input &&
          view.group_size == (size_t)group_size);
    CHECK(coli_affine_bits(view.format) == (unsigned)quant_bits);
    CHECK(view.scalar_format == scalar_format);

    float input[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float output[2];
    CHECK(coli_affine_matmul_ref(output, input, 1, &view) == COLI_AFFINE_OK);
    CHECK(fabsf(output[0] - expected0) < 1e-5f);
    CHECK(fabsf(output[1] - expected1) < 1e-5f);

    CHECK(coli_qpack_affine_view(&reader, blob, reader.layout.expert_stride,
                                 "up_proj", &view,
                                 error, sizeof(error)) != 0);
    CHECK(coli_qpack_affine_view(&reader, blob, 16, "gate_proj", &view,
                                 error, sizeof(error)) != 0);
    CHECK(coli_qpack_read_expert(&reader, 2, 0, blob,
                                 reader.layout.expert_stride,
                                 error, sizeof(error)) != 0);
    CHECK(coli_qpack_read_expert(&reader, 0, 0, blob, 16,
                                 error, sizeof(error)) != 0);
    free(blob);
    coli_qpack_close(&reader);
    CHECK(reader.layer_fds == NULL && reader.layout.sections == NULL);
    cleanup(root);
}

static void test_affine_shape_refusal(void) {
    char root[] = "test_qpack_shape_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    write_fixture(root, bad_affine_shape_layout, "QPACK", 32768, 4, 8,
                  COLI_AFFINE_SCALAR_BF16);
    ColiQpackReader reader;
    char error[512] = {0};
    CHECK(coli_qpack_open(&reader, root, error, sizeof(error)) == 0);
    unsigned char *blob = (unsigned char *)calloc(reader.layout.expert_stride, 1);
    CHECK(blob != NULL);
    ColiAffineQuantizedView view;
    CHECK(coli_qpack_affine_view(&reader, blob, reader.layout.expert_stride,
                                 "gate_proj", &view,
                                 error, sizeof(error)) != 0);
    CHECK(strstr(error, "shape does not match") != NULL);
    free(blob);
    coli_qpack_close(&reader);
    cleanup(root);
}

static void test_post_open_truncation(void) {
    char root[] = "test_qpack_short_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    write_fixture(root, valid_layout_bf16, "QPACK", 32768, 4, 8,
                  COLI_AFFINE_SCALAR_BF16);
    ColiQpackReader reader;
    char error[512] = {0};
    CHECK(coli_qpack_open(&reader, root, error, sizeof(error)) == 0);
    write_layer(root, 1, 16384, COLI_AFFINE_SCALAR_BF16);
    unsigned char *blob = (unsigned char *)malloc(reader.layout.expert_stride);
    CHECK(blob != NULL);
    CHECK(coli_qpack_read_expert(&reader, 1, 1, blob,
                                 reader.layout.expert_stride,
                                 error, sizeof(error)) != 0);
    CHECK(strstr(error, "short read") != NULL);
    free(blob);
    coli_qpack_close(&reader);
    cleanup(root);
}

static void test_eager_layer_validation_cleanup(void) {
    char root[] = "test_qpack_eager_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    write_fixture(root, valid_layout_bf16, "QPACK", 32768, 4, 8,
                  COLI_AFFINE_SCALAR_BF16);
    write_layer(root, 1, 16384, COLI_AFFINE_SCALAR_BF16);
    ColiQpackReader reader;
    char error[512] = {0};
    CHECK(coli_qpack_open(&reader, root, error, sizeof(error)) != 0);
    CHECK(strstr(error, "unexpected size") != NULL);
    CHECK(reader.layer_fds == NULL && reader.layout.sections == NULL);
    cleanup(root);
}

static void test_embedded_nul_metadata(void) {
    char root[] = "test_qpack_nul_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    size_t prefix_size = strlen(valid_layout_bf16);
    static const char suffix[] = "trailing";
    size_t layout_size = prefix_size + 1 + sizeof(suffix) - 1;
    char *layout = (char *)malloc(layout_size);
    CHECK(layout != NULL);
    memcpy(layout, valid_layout_bf16, prefix_size);
    layout[prefix_size] = 0;
    memcpy(layout + prefix_size + 1, suffix, sizeof(suffix) - 1);
    write_fixture_sized(root, layout, layout_size, "QPACK", 32768, 4, 8,
                        COLI_AFFINE_SCALAR_BF16);
    free(layout);

    ColiQpackReader reader;
    char error[512] = {0};
    CHECK(coli_qpack_open(&reader, root, error, sizeof(error)) != 0);
    CHECK(strstr(error, "embedded NUL") != NULL);
    CHECK(reader.layer_fds == NULL && reader.layout.sections == NULL);
    cleanup(root);
}

int main(void) {
    static const struct {
        const char *layout;
        ColiAffineScalarFormat scalar_format;
    } formats[] = {
        {valid_layout_bf16, COLI_AFFINE_SCALAR_BF16},
        {valid_layout_f16, COLI_AFFINE_SCALAR_F16},
        {valid_layout_f32, COLI_AFFINE_SCALAR_F32}
    };
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
        test_valid_container(formats[i].layout, formats[i].scalar_format,
                             4, 8, 8, 15.0f, 4.0f);
        test_valid_container(formats[i].layout, formats[i].scalar_format,
                             8, 4, 4, 134.5f, 66.0f);
    }
    expect_open_failure(valid_layout_bf16, "NOTQPACK", 32768,
                        "not a supported QPACK v1");
    expect_open_failure(misaligned_layout, "QPACK", 16384,
                        "16 KiB aligned");
    expect_open_failure(overlap_layout, "QPACK", 32768, "overlap");
    expect_open_failure(duplicate_layout, "QPACK", 32768,
                        "duplicate section");
    expect_open_failure(overflow_layout, "QPACK", 32768, "invalid shape");
    expect_open_failure(valid_layout_bf16, "QPACK", 32767,
                        "manifest size mismatch");
    test_affine_shape_refusal();
    test_eager_layer_validation_cleanup();
    test_embedded_nul_metadata();
    test_post_open_truncation();
    puts("test_qpack: strict v1 metadata, fixed-stride read, affine views: ok");
    return 0;
}
