#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "qpack.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "compat.h"
#include "json.h"

#define COLI_QPACK_MAX_JSON (16u * 1024u * 1024u)
#define COLI_QPACK_MAX_LAYERS 4096u
#define COLI_QPACK_MAX_EXPERTS 1048576u
#define COLI_QPACK_MAX_SECTIONS 256u
#define COLI_QPACK_MAX_FILES 8192u
#define COLI_QPACK_MAX_RANK 16u
#define COLI_QPACK_MAX_EXPERT_STRIDE (1u << 30)

static int qpack_fail(char *error, size_t capacity, const char *format, ...) {
    if (error && capacity) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, capacity, format, args);
        va_end(args);
    }
    return -1;
}

static char *qpack_copy_string(const char *source) {
    size_t size = strlen(source) + 1;
    char *copy = (char *)malloc(size);
    if (copy) memcpy(copy, source, size);
    return copy;
}

static char *qpack_join_path(const char *left, const char *right) {
    size_t left_size = strlen(left), right_size = strlen(right);
    int slash = left_size > 0 && left[left_size - 1] != '/' &&
                left[left_size - 1] != '\\';
    size_t extra = (size_t)slash + 1;
    if (right_size > SIZE_MAX - extra ||
        left_size > SIZE_MAX - right_size - extra)
        return NULL;
    char *path = (char *)malloc(left_size + (size_t)slash + right_size + 1);
    if (!path) return NULL;
    memcpy(path, left, left_size);
    if (slash) path[left_size++] = '/';
    memcpy(path + left_size, right, right_size + 1);
    return path;
}

static char *qpack_read_text(const char *path, size_t *size_out,
                             char *error, size_t error_capacity) {
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0)
        return qpack_fail(error, error_capacity, "open %s: %s",
                          path, strerror(errno)), NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        int saved = errno;
        close(fd);
        return qpack_fail(error, error_capacity, "fstat %s: %s",
                          path, strerror(saved)), NULL;
    }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0 ||
        (uint64_t)st.st_size > COLI_QPACK_MAX_JSON) {
        close(fd);
        return qpack_fail(error, error_capacity,
                          "%s is not a bounded regular JSON file", path), NULL;
    }
    size_t size = (size_t)st.st_size;
    char *text = (char *)malloc(size + 1);
    if (!text) {
        close(fd);
        return qpack_fail(error, error_capacity,
                          "out of memory reading %s", path), NULL;
    }
    ssize_t count;
    do {
        count = pread(fd, text, size, 0);
    } while (count < 0 && errno == EINTR);
    int saved = errno;
    close(fd);
    if (count < 0 || (size_t)count != size) {
        free(text);
        if (count < 0)
            return qpack_fail(error, error_capacity, "pread %s: %s",
                              path, strerror(saved)), NULL;
        return qpack_fail(error, error_capacity, "short read from %s", path), NULL;
    }
    if (memchr(text, 0, size)) {
        free(text);
        return qpack_fail(error, error_capacity,
                          "%s contains an embedded NUL", path), NULL;
    }
    text[size] = 0;
    if (size_out) *size_out = size;
    return text;
}

static int qpack_json_u64(jval *value, uint64_t *output) {
    if (!value || value->t != J_NUM || !isfinite(value->num) ||
        value->num < 0.0 || floor(value->num) != value->num ||
        value->num > 9007199254740991.0)
        return -1;
    *output = (uint64_t)value->num;
    return 0;
}

static int qpack_object_has_duplicate_keys(jval *object) {
    if (!object || object->t != J_OBJ) return 0;
    for (int i = 0; i < object->len; i++)
        for (int j = 0; j < i; j++)
            if (strcmp(object->keys[i], object->keys[j]) == 0) return 1;
    return 0;
}

static int qpack_dtype_size(const char *dtype, size_t *size) {
    static const struct { const char *name; size_t size; } types[] = {
        {"BOOL", 1}, {"U8", 1}, {"I8", 1}, {"U16", 2}, {"I16", 2},
        {"F16", 2}, {"BF16", 2}, {"U32", 4}, {"I32", 4}, {"F32", 4},
        {"U64", 8}, {"I64", 8}, {"F64", 8}
    };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        if (strcmp(dtype, types[i].name) == 0) {
            *size = types[i].size;
            return 0;
        }
    }
    return -1;
}

static void qpack_free_layout(ColiQpackLayout *layout) {
    if (!layout) return;
    for (size_t i = 0; i < layout->section_count; i++) {
        free(layout->sections[i].name);
        free(layout->sections[i].dtype);
        free(layout->sections[i].shape);
    }
    free(layout->sections);
    free(layout->linear_layers);
    memset(layout, 0, sizeof(*layout));
}

static int qpack_parse_layout(ColiQpackLayout *layout, const char *text,
                              char *error, size_t error_capacity) {
    jval *root = json_parse_exact(text, NULL);
    if (!root || root->t != J_OBJ || qpack_object_has_duplicate_keys(root)) {
        json_free(root);
        return qpack_fail(error, error_capacity,
                          "layout.json must be an object with unique keys");
    }

    uint64_t expert_count, layer_count, expert_stride;
    if (qpack_json_u64(json_get(root, "expertCount"), &expert_count) ||
        !expert_count || expert_count > COLI_QPACK_MAX_EXPERTS ||
        qpack_json_u64(json_get(root, "layerCount"), &layer_count) ||
        !layer_count || layer_count > COLI_QPACK_MAX_LAYERS ||
        qpack_json_u64(json_get(root, "expertStride"), &expert_stride) ||
        !expert_stride || expert_stride > COLI_QPACK_MAX_EXPERT_STRIDE ||
        expert_stride > SIZE_MAX) {
        json_free(root);
        return qpack_fail(error, error_capacity,
                          "layout has invalid counts or stride");
    }
    if (expert_stride % COLI_QPACK_PAGE_ALIGNMENT != 0) {
        json_free(root);
        return qpack_fail(error, error_capacity,
                          "expertStride must be 16 KiB aligned");
    }

    jval *sections = json_get(root, "sections");
    jval *linear = json_get(root, "linearLayers");
    if (!sections || sections->t != J_ARR || sections->len <= 0 ||
        sections->len > (int)COLI_QPACK_MAX_SECTIONS ||
        !linear || linear->t != J_ARR ||
        (uint64_t)linear->len != layer_count) {
        json_free(root);
        return qpack_fail(error, error_capacity,
                          "layout sections or linearLayers are invalid");
    }

    layout->expert_count = (size_t)expert_count;
    layout->layer_count = (size_t)layer_count;
    layout->expert_stride = (size_t)expert_stride;
    layout->section_count = (size_t)sections->len;
    layout->sections = (ColiQpackSection *)calloc(
        layout->section_count, sizeof(ColiQpackSection));
    layout->linear_layers = (unsigned char *)calloc(layout->layer_count, 1);
    if (!layout->sections || !layout->linear_layers) {
        json_free(root);
        qpack_free_layout(layout);
        return qpack_fail(error, error_capacity,
                          "out of memory parsing layout");
    }

    for (size_t i = 0; i < layout->layer_count; i++) {
        jval *value = linear->kids[i];
        if (!value || value->t != J_BOOL) {
            json_free(root);
            qpack_free_layout(layout);
            return qpack_fail(error, error_capacity,
                              "linearLayers[%zu] is not boolean", i);
        }
        layout->linear_layers[i] = (unsigned char)value->boolean;
    }

    for (size_t i = 0; i < layout->section_count; i++) {
        jval *object = sections->kids[i];
        jval *name = json_get(object, "name");
        jval *dtype = json_get(object, "dtype");
        jval *shape = json_get(object, "shape");
        uint64_t offset, size;
        if (!object || object->t != J_OBJ ||
            qpack_object_has_duplicate_keys(object) ||
            !name || name->t != J_STR || !name->str[0] ||
            !dtype || dtype->t != J_STR || !dtype->str[0] ||
            !shape || shape->t != J_ARR || shape->len <= 0 ||
            shape->len > (int)COLI_QPACK_MAX_RANK ||
            qpack_json_u64(json_get(object, "offset"), &offset) ||
            qpack_json_u64(json_get(object, "size"), &size) || !size ||
            offset > expert_stride || size > expert_stride - offset) {
            json_free(root);
            qpack_free_layout(layout);
            return qpack_fail(error, error_capacity,
                              "section %zu is invalid", i);
        }

        ColiQpackSection *section = &layout->sections[i];
        section->name = qpack_copy_string(name->str);
        section->dtype = qpack_copy_string(dtype->str);
        section->rank = (size_t)shape->len;
        section->shape = (size_t *)calloc(section->rank, sizeof(size_t));
        section->offset = offset;
        section->size = size;
        if (!section->name || !section->dtype || !section->shape) {
            json_free(root);
            qpack_free_layout(layout);
            return qpack_fail(error, error_capacity,
                              "out of memory parsing section %zu", i);
        }

        size_t element_size;
        uint64_t element_count = 1;
        if (qpack_dtype_size(section->dtype, &element_size)) {
            int result = qpack_fail(error, error_capacity,
                                    "section %s has unsupported dtype %s",
                                    section->name, section->dtype);
            json_free(root);
            qpack_free_layout(layout);
            return result;
        }
        for (size_t d = 0; d < section->rank; d++) {
            uint64_t dimension;
            if (qpack_json_u64(shape->kids[d], &dimension) || !dimension ||
                dimension > SIZE_MAX ||
                element_count > UINT64_MAX / dimension) {
                int result = qpack_fail(error, error_capacity,
                                        "section %s has invalid shape",
                                        section->name);
                json_free(root);
                qpack_free_layout(layout);
                return result;
            }
            section->shape[d] = (size_t)dimension;
            element_count *= dimension;
        }
        if (element_count > UINT64_MAX / element_size ||
            element_count * element_size != section->size) {
            int result = qpack_fail(
                error, error_capacity,
                "section %s size does not match dtype and shape", section->name);
            json_free(root);
            qpack_free_layout(layout);
            return result;
        }

        for (size_t prior = 0; prior < i; prior++) {
            const ColiQpackSection *other = &layout->sections[prior];
            if (strcmp(section->name, other->name) == 0) {
                int result = qpack_fail(error, error_capacity,
                                        "duplicate section name %s", section->name);
                json_free(root);
                qpack_free_layout(layout);
                return result;
            }
            uint64_t end = section->offset + section->size;
            uint64_t other_end = other->offset + other->size;
            if (section->offset < other_end && other->offset < end) {
                int result = qpack_fail(error, error_capacity,
                                        "sections %s and %s overlap",
                                        section->name, other->name);
                json_free(root);
                qpack_free_layout(layout);
                return result;
            }
        }
    }

    json_free(root);
    return 0;
}

static int qpack_parse_manifest(const char *text,
                                int *quant_bits, int *quant_group_size,
                                jval **files_out, jval **root_out,
                                char *error, size_t error_capacity) {
    jval *root = json_parse_exact(text, NULL);
    jval *magic = json_get(root, "magic");
    jval *model = json_get(root, "modelName");
    jval *source = json_get(root, "sourceCheckpoint");
    uint64_t version;
    if (!root || root->t != J_OBJ || qpack_object_has_duplicate_keys(root) ||
        !magic || magic->t != J_STR || strcmp(magic->str, "QPACK") != 0 ||
        !model || model->t != J_STR || !model->str[0] ||
        !source || source->t != J_STR || !source->str[0] ||
        qpack_json_u64(json_get(root, "version"), &version) ||
        version != COLI_QPACK_MANIFEST_VERSION) {
        json_free(root);
        return qpack_fail(error, error_capacity,
                          "manifest is not a supported QPACK v1 container");
    }
    jval *files = json_get(root, "files");
    if (!files || files->t != J_OBJ || files->len <= 0 ||
        files->len > (int)COLI_QPACK_MAX_FILES ||
        qpack_object_has_duplicate_keys(files)) {
        json_free(root);
        return qpack_fail(error, error_capacity,
                          "manifest files table is invalid");
    }
    for (int i = 0; i < files->len; i++) {
        uint64_t ignored;
        if (!files->keys[i][0] || qpack_json_u64(files->kids[i], &ignored)) {
            json_free(root);
            return qpack_fail(error, error_capacity,
                              "manifest file size is invalid");
        }
    }

    *quant_bits = 0;
    *quant_group_size = 0;
    jval *bits = json_get(root, "quantBits");
    jval *group = json_get(root, "quantGroupSize");
    uint64_t number;
    if (bits && bits->t != J_NULL) {
        if (qpack_json_u64(bits, &number) || number > INT_MAX) {
            json_free(root);
            return qpack_fail(error, error_capacity,
                              "manifest quantBits is invalid");
        }
        *quant_bits = (int)number;
    }
    if (group && group->t != J_NULL) {
        if (qpack_json_u64(group, &number) || number > INT_MAX) {
            json_free(root);
            return qpack_fail(error, error_capacity,
                              "manifest quantGroupSize is invalid");
        }
        *quant_group_size = (int)number;
    }
    *files_out = files;
    *root_out = root;
    return 0;
}

static int qpack_verify_manifest_size(jval *files, const char *relative,
                                      uint64_t expected_size,
                                      char *error, size_t error_capacity) {
    uint64_t manifest_size;
    jval *entry = json_get(files, relative);
    if (!entry || qpack_json_u64(entry, &manifest_size))
        return qpack_fail(error, error_capacity,
                          "manifest has no valid size for %s", relative);
    if (manifest_size != expected_size)
        return qpack_fail(error, error_capacity,
                          "manifest size mismatch for %s", relative);
    return 0;
}

int coli_qpack_open(ColiQpackReader *reader, const char *container_dir,
                    char *error, size_t error_capacity) {
    if (!reader)
        return qpack_fail(error, error_capacity, "qpack reader is null");
    memset(reader, 0, sizeof(*reader));
    if (!container_dir || !container_dir[0])
        return qpack_fail(error, error_capacity, "qpack path is empty");

    char *manifest_path = qpack_join_path(container_dir, "manifest.json");
    reader->packed_dir = qpack_join_path(container_dir, "packed_experts");
    char *layout_path = reader->packed_dir ?
        qpack_join_path(reader->packed_dir, "layout.json") : NULL;
    if (!manifest_path || !reader->packed_dir || !layout_path) {
        free(manifest_path);
        free(layout_path);
        coli_qpack_close(reader);
        return qpack_fail(error, error_capacity,
                          "out of memory building qpack paths");
    }

    size_t layout_size;
    char *manifest_text = qpack_read_text(
        manifest_path, NULL, error, error_capacity);
    char *layout_text = qpack_read_text(
        layout_path, &layout_size, error, error_capacity);
    free(manifest_path);
    if (!manifest_text || !layout_text) {
        free(manifest_text);
        free(layout_text);
        free(layout_path);
        coli_qpack_close(reader);
        return -1;
    }

    jval *manifest_root = NULL, *files = NULL;
    if (qpack_parse_manifest(manifest_text, &reader->quant_bits,
                             &reader->quant_group_size, &files,
                             &manifest_root, error, error_capacity) ||
        qpack_parse_layout(&reader->layout, layout_text,
                           error, error_capacity)) {
        free(manifest_text);
        free(layout_text);
        free(layout_path);
        json_free(manifest_root);
        coli_qpack_close(reader);
        return -1;
    }
    free(manifest_text);
    free(layout_text);

    if (qpack_verify_manifest_size(files, "packed_experts/layout.json",
                                   layout_size, error, error_capacity)) {
        free(layout_path);
        json_free(manifest_root);
        coli_qpack_close(reader);
        return -1;
    }
    free(layout_path);

    if (reader->layout.expert_count >
        (uint64_t)INT64_MAX / reader->layout.expert_stride) {
        json_free(manifest_root);
        coli_qpack_close(reader);
        return qpack_fail(error, error_capacity,
                          "expert layer size overflows off_t");
    }
    uint64_t layer_size = (uint64_t)reader->layout.expert_count *
                          reader->layout.expert_stride;
    reader->layer_fds = (int *)malloc(reader->layout.layer_count * sizeof(int));
    if (!reader->layer_fds) {
        json_free(manifest_root);
        coli_qpack_close(reader);
        return qpack_fail(error, error_capacity,
                          "out of memory allocating layer descriptors");
    }
    for (size_t layer = 0; layer < reader->layout.layer_count; layer++)
        reader->layer_fds[layer] = -1;

    for (size_t layer = 0; layer < reader->layout.layer_count; layer++) {
        char name[64], relative[96];
        int name_count = snprintf(name, sizeof(name), "layer_%02zu.bin", layer);
        int relative_count = snprintf(relative, sizeof(relative),
                                      "packed_experts/%s", name);
        char *path = qpack_join_path(reader->packed_dir, name);
        if (name_count < 0 || (size_t)name_count >= sizeof(name) ||
            relative_count < 0 || (size_t)relative_count >= sizeof(relative)) {
            free(path);
            json_free(manifest_root);
            coli_qpack_close(reader);
            return qpack_fail(error, error_capacity,
                              "layer filename is too long");
        }
        if (!path) {
            json_free(manifest_root);
            coli_qpack_close(reader);
            return qpack_fail(error, error_capacity,
                              "out of memory building layer path");
        }
        if (qpack_verify_manifest_size(files, relative, layer_size,
                                       error, error_capacity)) {
            free(path);
            json_free(manifest_root);
            coli_qpack_close(reader);
            return -1;
        }
        int fd = open(path, COMPAT_O_RDONLY);
        if (fd < 0) {
            int saved = errno;
            free(path);
            json_free(manifest_root);
            coli_qpack_close(reader);
            return qpack_fail(error, error_capacity, "open %s: %s",
                              relative, strerror(saved));
        }
        reader->layer_fds[layer] = fd;
        struct stat st;
        if (fstat(fd, &st) != 0) {
            int saved = errno;
            free(path);
            json_free(manifest_root);
            coli_qpack_close(reader);
            return qpack_fail(error, error_capacity, "fstat %s: %s",
                              relative, strerror(saved));
        }
        if (!S_ISREG(st.st_mode) || st.st_size < 0 ||
            (uint64_t)st.st_size != layer_size) {
            free(path);
            json_free(manifest_root);
            coli_qpack_close(reader);
            return qpack_fail(error, error_capacity,
                              "%s has unexpected size", relative);
        }
        free(path);
    }

    json_free(manifest_root);
    return 0;
}

void coli_qpack_close(ColiQpackReader *reader) {
    if (!reader) return;
    if (reader->layer_fds) {
        for (size_t i = 0; i < reader->layout.layer_count; i++)
            if (reader->layer_fds[i] >= 0) close(reader->layer_fds[i]);
    }
    free(reader->layer_fds);
    free(reader->packed_dir);
    qpack_free_layout(&reader->layout);
    memset(reader, 0, sizeof(*reader));
}

const ColiQpackSection *
coli_qpack_find_section(const ColiQpackReader *reader, const char *name) {
    if (!reader || !name) return NULL;
    for (size_t i = 0; i < reader->layout.section_count; i++)
        if (strcmp(reader->layout.sections[i].name, name) == 0)
            return &reader->layout.sections[i];
    return NULL;
}

int coli_qpack_read_expert(const ColiQpackReader *reader,
                           size_t layer, size_t expert,
                           void *buffer, size_t buffer_size,
                           char *error, size_t error_capacity) {
    if (!reader || !buffer || !reader->layer_fds)
        return qpack_fail(error, error_capacity, "qpack reader is not open");
    if (layer >= reader->layout.layer_count ||
        expert >= reader->layout.expert_count)
        return qpack_fail(error, error_capacity,
                          "expert coordinate is out of range");
    if (buffer_size < reader->layout.expert_stride)
        return qpack_fail(error, error_capacity,
                          "expert destination buffer is too small");

    off_t offset = (off_t)((uint64_t)expert * reader->layout.expert_stride);
    ssize_t count;
    do {
        count = pread(reader->layer_fds[layer], buffer,
                      reader->layout.expert_stride, offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0)
        return qpack_fail(error, error_capacity,
                          "pread layer %zu expert %zu: %s",
                          layer, expert, strerror(errno));
    if ((size_t)count != reader->layout.expert_stride)
        return qpack_fail(error, error_capacity,
                          "short read for layer %zu expert %zu", layer, expert);
    return 0;
}

static const ColiQpackSection *
qpack_projection_section(const ColiQpackReader *reader,
                         const char *projection, const char *suffix) {
    char name[256];
    int count = snprintf(name, sizeof(name), "%s.%s", projection, suffix);
    if (count < 0 || (size_t)count >= sizeof(name)) return NULL;
    return coli_qpack_find_section(reader, name);
}

static int qpack_scalar_format(const char *dtype,
                               ColiAffineScalarFormat *format) {
    if (strcmp(dtype, "F32") == 0) *format = COLI_AFFINE_SCALAR_F32;
    else if (strcmp(dtype, "F16") == 0) *format = COLI_AFFINE_SCALAR_F16;
    else if (strcmp(dtype, "BF16") == 0) *format = COLI_AFFINE_SCALAR_BF16;
    else return -1;
    return 0;
}

int coli_qpack_affine_view(const ColiQpackReader *reader,
                           const void *expert_blob, size_t blob_size,
                           const char *projection,
                           ColiAffineQuantizedView *view,
                           char *error, size_t error_capacity) {
    if (!reader || !expert_blob || !projection || !projection[0] || !view)
        return qpack_fail(error, error_capacity,
                          "affine descriptor received a null pointer");
    memset(view, 0, sizeof(*view));
    if (blob_size < reader->layout.expert_stride)
        return qpack_fail(error, error_capacity,
                          "expert blob is shorter than expertStride");
    if ((reader->quant_bits != 4 && reader->quant_bits != 8) ||
        reader->quant_group_size <= 0)
        return qpack_fail(error, error_capacity,
                          "qpack affine quantization metadata is unsupported");

    const ColiQpackSection *weight = qpack_projection_section(
        reader, projection, "weight");
    const ColiQpackSection *scales = qpack_projection_section(
        reader, projection, "scales");
    const ColiQpackSection *biases = qpack_projection_section(
        reader, projection, "biases");
    if (!weight || !scales || !biases)
        return qpack_fail(error, error_capacity,
                          "projection is missing weight, scales, or biases");
    if (strcmp(weight->dtype, "U32") != 0 || weight->rank < 2)
        return qpack_fail(error, error_capacity,
                          "affine weight must be a rank-2+ U32 tensor");
    if (strcmp(scales->dtype, biases->dtype) != 0 ||
        scales->rank != weight->rank || biases->rank != weight->rank)
        return qpack_fail(error, error_capacity,
                          "affine scale and bias metadata is incompatible");

    ColiAffineScalarFormat scalar_format;
    if (qpack_scalar_format(scales->dtype, &scalar_format))
        return qpack_fail(error, error_capacity,
                          "affine scales must be F32, F16, or BF16");

    size_t output_dim = 1;
    for (size_t i = 0; i + 1 < weight->rank; i++) {
        if (output_dim > SIZE_MAX / weight->shape[i])
            return qpack_fail(error, error_capacity,
                              "affine output dimension overflows");
        output_dim *= weight->shape[i];
        if (scales->shape[i] != weight->shape[i] ||
            biases->shape[i] != weight->shape[i])
            return qpack_fail(error, error_capacity,
                              "affine scale or bias shape does not match weights");
    }
    const unsigned per_word = 32u / (unsigned)reader->quant_bits;
    if (weight->shape[weight->rank - 1] > SIZE_MAX / per_word)
        return qpack_fail(error, error_capacity,
                          "affine input dimension overflows");
    size_t input_dim = weight->shape[weight->rank - 1] * per_word;
    size_t group_size = (size_t)reader->quant_group_size;
    if (!output_dim || !input_dim || input_dim % group_size != 0)
        return qpack_fail(error, error_capacity,
                          "affine dimensions do not match quantGroupSize");
    size_t groups = input_dim / group_size;
    if (scales->shape[scales->rank - 1] != groups ||
        biases->shape[biases->rank - 1] != groups)
        return qpack_fail(error, error_capacity,
                          "affine scale or bias shape does not match weights");

    const unsigned char *blob = (const unsigned char *)expert_blob;
    view->weights = blob + weight->offset;
    view->scales = blob + scales->offset;
    view->biases = blob + biases->offset;
    view->weight_bytes = (size_t)weight->size;
    view->scale_bytes = (size_t)scales->size;
    view->bias_bytes = (size_t)biases->size;
    view->output_dim = output_dim;
    view->input_dim = input_dim;
    view->group_size = group_size;
    view->format = reader->quant_bits == 4 ?
        COLI_AFFINE_MLX_Q4 : COLI_AFFINE_MLX_Q8;
    view->scalar_format = scalar_format;
    ColiAffineStatus status = coli_affine_validate(view);
    if (status != COLI_AFFINE_OK) {
        memset(view, 0, sizeof(*view));
        return qpack_fail(error, error_capacity,
                          "affine descriptor is invalid: %s",
                          coli_affine_status_string(status));
    }
    return 0;
}
