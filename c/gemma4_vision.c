#define _CRT_SECURE_NO_WARNINGS

#include "gemma4_vision.h"

#include <math.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(COLI_GEMMA4_VISION_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define COLI_VISION_HAS_AVX2 1
#endif

#ifdef COLI_GEMMA4_LIBPNG
#include <png.h>
#endif
#ifdef COLI_GEMMA4_LIBJPEG
#include <setjmp.h>
#include <jpeglib.h>
#endif

#ifdef _WIN32
#define COBJMACROS
#include <windows.h>
#include <wincodec.h>
#endif

static void ppm_error(char *error, size_t capacity, const char *format, ...) {
    va_list arguments;
    if (!error || !capacity) return;
    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
}

static int ppm_token(FILE *file, char *token, size_t capacity) {
    int character;
    size_t length = 0;
    do {
        character = fgetc(file);
        if (character == '#')
            while ((character = fgetc(file)) != '\n' && character != EOF) {}
    } while (character == ' ' || character == '\t' ||
             character == '\r' || character == '\n');
    if (character == EOF) return -1;
    do {
        if (length + 1 >= capacity) return -1;
        token[length++] = (char)character;
        character = fgetc(file);
    } while (character != EOF && character != ' ' && character != '\t' &&
             character != '\r' && character != '\n');
    if (character == '\r') {
        character = fgetc(file);
        if (character != '\n' && character != EOF) ungetc(character, file);
    }
    token[length] = '\0';
    return 0;
}

int coli_gemma4_vision_load_ppm(const char *path, uint8_t **rgb,
                                uint32_t *width, uint32_t *height,
                                char *error, size_t error_capacity) {
    FILE *file = NULL;
    uint8_t *pixels = NULL;
    char token[64], *end = NULL;
    unsigned long parsed_width, parsed_height, maximum;
    size_t pixel_bytes;
    int status = -1;
    if (error && error_capacity) error[0] = '\0';
    if (!path || !rgb || !width || !height) return -1;
    *rgb = NULL;
    *width = 0;
    *height = 0;
    file = fopen(path, "rb");
    if (!file) {
        ppm_error(error, error_capacity, "cannot open %s: %s", path,
                  strerror(errno));
        return -1;
    }
    if (ppm_token(file, token, sizeof(token)) != 0 || strcmp(token, "P6") != 0) {
        ppm_error(error, error_capacity,
                  "%s is not a binary PPM (expected P6)", path);
        goto cleanup;
    }
    if (ppm_token(file, token, sizeof(token)) != 0) goto invalid_header;
    errno = 0;
    parsed_width = strtoul(token, &end, 10);
    if (errno || !end || *end || !parsed_width || parsed_width > UINT32_MAX)
        goto invalid_header;
    if (ppm_token(file, token, sizeof(token)) != 0) goto invalid_header;
    errno = 0;
    parsed_height = strtoul(token, &end, 10);
    if (errno || !end || *end || !parsed_height || parsed_height > UINT32_MAX)
        goto invalid_header;
    if (ppm_token(file, token, sizeof(token)) != 0) goto invalid_header;
    errno = 0;
    maximum = strtoul(token, &end, 10);
    if (errno || !end || *end || maximum != 255) {
        ppm_error(error, error_capacity,
                  "%s must use 8-bit PPM samples (maxval 255)", path);
        goto cleanup;
    }
    if ((uint64_t)parsed_width * (uint64_t)parsed_height > SIZE_MAX / 3) {
        ppm_error(error, error_capacity, "%s dimensions are too large", path);
        goto cleanup;
    }
    pixel_bytes = (size_t)parsed_width * (size_t)parsed_height * 3;
    pixels = (uint8_t *)malloc(pixel_bytes);
    if (!pixels) {
        ppm_error(error, error_capacity, "out of memory reading %s", path);
        goto cleanup;
    }
    if (fread(pixels, 1, pixel_bytes, file) != pixel_bytes) {
        ppm_error(error, error_capacity, "%s has a truncated pixel payload", path);
        goto cleanup;
    }
    *rgb = pixels;
    *width = (uint32_t)parsed_width;
    *height = (uint32_t)parsed_height;
    pixels = NULL;
    status = 0;
    goto cleanup;

invalid_header:
    ppm_error(error, error_capacity, "%s has an invalid PPM header", path);
cleanup:
    free(pixels);
    fclose(file);
    return status;
}

#ifdef COLI_GEMMA4_LIBPNG
static int load_png_image(const char *path, uint8_t **rgb,
                          uint32_t *width, uint32_t *height,
                          char *error, size_t error_capacity) {
    png_image image;
    uint8_t *pixels = NULL;
    size_t pixel_bytes;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&image, path)) {
        ppm_error(error, error_capacity, "cannot decode PNG %s: %s", path,
                  image.message);
        return -1;
    }
    image.format = PNG_FORMAT_RGB;
    if (!image.width || !image.height || image.width > SIZE_MAX / 3U ||
        image.height > SIZE_MAX / ((size_t)image.width * 3U)) {
        ppm_error(error, error_capacity, "%s dimensions are too large", path);
        png_image_free(&image);
        return -1;
    }
    pixel_bytes = PNG_IMAGE_SIZE(image);
    pixels = (uint8_t *)malloc(pixel_bytes);
    if (!pixels || !png_image_finish_read(&image, NULL, pixels, 0, NULL)) {
        ppm_error(error, error_capacity, "cannot decode PNG %s: %s", path,
                  pixels ? image.message : "out of memory");
        free(pixels);
        png_image_free(&image);
        return -1;
    }
    *rgb = pixels;
    *width = (uint32_t)image.width;
    *height = (uint32_t)image.height;
    png_image_free(&image);
    return 0;
}
#endif

#ifdef COLI_GEMMA4_LIBJPEG
typedef struct {
    struct jpeg_error_mgr base;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
} coli_jpeg_error;

static void jpeg_failure(j_common_ptr common) {
    coli_jpeg_error *failure = (coli_jpeg_error *)common->err;
    (*common->err->format_message)(common, failure->message);
    longjmp(failure->jump, 1);
}

static int load_jpeg_image(const char *path, uint8_t **rgb,
                           uint32_t *width, uint32_t *height,
                           char *error, size_t error_capacity) {
    struct jpeg_decompress_struct decoder;
    coli_jpeg_error failure;
    FILE * volatile file = NULL;
    uint8_t * volatile pixels = NULL;
    size_t stride, pixel_bytes;
    volatile int created = 0;
    int status = -1;
    memset(&decoder, 0, sizeof(decoder));
    memset(&failure, 0, sizeof(failure));
    decoder.err = jpeg_std_error(&failure.base);
    failure.base.error_exit = jpeg_failure;
    if (setjmp(failure.jump)) {
        ppm_error(error, error_capacity, "cannot decode JPEG %s: %s", path,
                  failure.message[0] ? failure.message : "invalid image");
        goto cleanup;
    }
    file = fopen(path, "rb");
    if (!file) {
        ppm_error(error, error_capacity, "cannot open %s: %s", path,
                  strerror(errno));
        goto cleanup;
    }
    jpeg_create_decompress(&decoder);
    created = 1;
    jpeg_stdio_src(&decoder, (FILE *)file);
    if (jpeg_read_header(&decoder, TRUE) != JPEG_HEADER_OK) goto cleanup;
    decoder.out_color_space = JCS_RGB;
    if (!jpeg_start_decompress(&decoder) || !decoder.output_width ||
        !decoder.output_height || decoder.output_components != 3 ||
        decoder.output_width > SIZE_MAX / 3U ||
        decoder.output_height >
            SIZE_MAX / ((size_t)decoder.output_width * 3U)) {
        ppm_error(error, error_capacity, "%s has unsupported dimensions", path);
        goto cleanup;
    }
    stride = (size_t)decoder.output_width * 3U;
    pixel_bytes = stride * (size_t)decoder.output_height;
    pixels = (uint8_t *)malloc(pixel_bytes);
    if (!pixels) {
        ppm_error(error, error_capacity, "out of memory reading %s", path);
        goto cleanup;
    }
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = (uint8_t *)pixels +
                       (size_t)decoder.output_scanline * stride;
        if (jpeg_read_scanlines(&decoder, &row, 1) != 1) goto cleanup;
    }
    if (!jpeg_finish_decompress(&decoder)) goto cleanup;
    *rgb = (uint8_t *)pixels;
    *width = (uint32_t)decoder.output_width;
    *height = (uint32_t)decoder.output_height;
    pixels = NULL;
    status = 0;
cleanup:
    free((uint8_t *)pixels);
    if (created) jpeg_destroy_decompress(&decoder);
    if (file) fclose((FILE *)file);
    return status;
}
#endif

#ifdef _WIN32
static int load_wic_image(const char *path, uint8_t **rgb,
                          uint32_t *width, uint32_t *height,
                          char *error, size_t error_capacity) {
    IWICImagingFactory *factory = NULL;
    IWICBitmapDecoder *decoder = NULL;
    IWICBitmapFrameDecode *frame = NULL;
    IWICFormatConverter *converter = NULL;
    wchar_t *wide_path = NULL;
    uint8_t *pixels = NULL;
    UINT image_width = 0, image_height = 0, stride = 0, pixel_bytes = 0;
    HRESULT initialized, result;
    int wide_length, status = -1;
    initialized = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(initialized) && initialized != RPC_E_CHANGED_MODE) {
        ppm_error(error, error_capacity, "cannot initialize image decoder (0x%08lx)",
                  (unsigned long)initialized);
        return -1;
    }
    wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                      path, -1, NULL, 0);
    if (!wide_length ||
        (size_t)wide_length > SIZE_MAX / sizeof(*wide_path)) {
        ppm_error(error, error_capacity, "image path is not valid UTF-8");
        goto cleanup;
    }
    wide_path = (wchar_t *)malloc((size_t)wide_length * sizeof(*wide_path));
    if (!wide_path || !MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, path, -1,
            wide_path, wide_length)) {
        ppm_error(error, error_capacity, "image path is not valid UTF-8");
        goto cleanup;
    }
    result = CoCreateInstance(&CLSID_WICImagingFactory, NULL,
                              CLSCTX_INPROC_SERVER, &IID_IWICImagingFactory,
                              (void **)&factory);
    if (SUCCEEDED(result))
        result = IWICImagingFactory_CreateDecoderFromFilename(
            factory, wide_path, NULL, GENERIC_READ,
            WICDecodeMetadataCacheOnLoad, &decoder);
    if (SUCCEEDED(result))
        result = IWICBitmapDecoder_GetFrame(decoder, 0, &frame);
    if (SUCCEEDED(result))
        result = IWICBitmapFrameDecode_GetSize(frame, &image_width,
                                                &image_height);
    if (SUCCEEDED(result) && (!image_width || !image_height ||
        image_width > UINT32_MAX || image_height > UINT32_MAX ||
        image_width > UINT_MAX / 3U ||
        image_height > UINT_MAX / (image_width * 3U)))
        result = E_INVALIDARG;
    if (SUCCEEDED(result))
        result = IWICImagingFactory_CreateFormatConverter(factory, &converter);
    if (SUCCEEDED(result))
        result = IWICFormatConverter_Initialize(
            converter, (IWICBitmapSource *)frame, &GUID_WICPixelFormat24bppRGB,
            WICBitmapDitherTypeNone, NULL, 0.0, WICBitmapPaletteTypeCustom);
    if (SUCCEEDED(result)) {
        stride = image_width * 3U;
        pixel_bytes = stride * image_height;
        pixels = (uint8_t *)malloc(pixel_bytes);
        if (!pixels) result = E_OUTOFMEMORY;
    }
    if (SUCCEEDED(result))
        result = IWICFormatConverter_CopyPixels(
            converter, NULL, stride, pixel_bytes, pixels);
    if (FAILED(result)) {
        ppm_error(error, error_capacity,
                  "cannot decode image %s (WIC error 0x%08lx)", path,
                  (unsigned long)result);
        goto cleanup;
    }
    *rgb = pixels;
    *width = (uint32_t)image_width;
    *height = (uint32_t)image_height;
    pixels = NULL;
    status = 0;
cleanup:
    free(pixels);
    free(wide_path);
    if (converter) IWICFormatConverter_Release(converter);
    if (frame) IWICBitmapFrameDecode_Release(frame);
    if (decoder) IWICBitmapDecoder_Release(decoder);
    if (factory) IWICImagingFactory_Release(factory);
    if (SUCCEEDED(initialized)) CoUninitialize();
    return status;
}
#endif

int coli_gemma4_vision_load_image(const char *path, uint8_t **rgb,
                                  uint32_t *width, uint32_t *height,
                                  char *error, size_t error_capacity) {
    FILE *file;
    unsigned char signature[8] = {0};
    if (!path || !rgb || !width || !height) return -1;
    file = fopen(path, "rb");
    if (!file) {
        ppm_error(error, error_capacity, "cannot open %s: %s", path,
                  strerror(errno));
        return -1;
    }
    (void)fread(signature, 1, sizeof(signature), file);
    fclose(file);
    if (signature[0] == 'P' && signature[1] == '6')
        return coli_gemma4_vision_load_ppm(
            path, rgb, width, height, error, error_capacity);
#ifdef _WIN32
    return load_wic_image(path, rgb, width, height, error, error_capacity);
#else
#ifdef COLI_GEMMA4_LIBPNG
    if (png_sig_cmp(signature, 0, sizeof(signature)) == 0)
        return load_png_image(path, rgb, width, height,
                              error, error_capacity);
#endif
#ifdef COLI_GEMMA4_LIBJPEG
    if (signature[0] == 0xffU && signature[1] == 0xd8U)
        return load_jpeg_image(path, rgb, width, height,
                               error, error_capacity);
#endif
    ppm_error(error, error_capacity,
              "%s is not a supported PPM/PNG/JPEG image", path);
    return -1;
#endif
}

typedef struct {
    uint16_t *q;
    uint16_t *k;
    uint16_t *v;
    uint16_t *output;
    uint16_t *ff_up;
    uint16_t *ff_gate;
    uint16_t *ff_down;
    float *ln1;
    float *ln2;
    float *q_norm;
    float *k_norm;
    float *attention_post_norm;
    float *ffn_post_norm;
} vision_block;

static void vision_error(coli_gemma4_vision *vision, const char *format, ...) {
    va_list arguments;
    if (!vision) return;
    va_start(arguments, format);
    vsnprintf(vision->last_error, sizeof(vision->last_error), format, arguments);
    va_end(arguments);
}

static int vision_trace_f32(coli_gemma4_vision *vision, uint32_t layer,
                            const char *name, const float *values,
                            size_t count) {
    char path[1024];
    FILE *file;
    int path_length;
    int close_status;
    if (!vision->trace_directory || vision->trace_layer != layer) return 0;
    path_length = snprintf(path, sizeof(path),
                           "%s/native-vision-layer%u-%s.f32",
                           vision->trace_directory, layer, name);
    if (path_length < 0 || (size_t)path_length >= sizeof(path)) {
        vision_error(vision, "vision trace path is too long");
        return -1;
    }
    file = fopen(path, "wb");
    if (!file) {
        vision_error(vision, "cannot write vision trace %s", path);
        return -1;
    }
    if (fwrite(values, sizeof(*values), count, file) != count) {
        fclose(file);
        vision_error(vision, "cannot write vision trace %s", path);
        return -1;
    }
    close_status = fclose(file);
    if (close_status != 0) {
        vision_error(vision, "cannot write vision trace %s", path);
        return -1;
    }
    return 0;
}

static int tensor_shape(const coli_gemma4_tensor *tensor, uint32_t type,
                        uint32_t dimensions, const uint64_t *shape) {
    uint32_t dimension;
    if (!tensor || tensor->type != type || tensor->n_dims != dimensions)
        return 0;
    for (dimension = 0; dimension < dimensions; ++dimension)
        if (tensor->dims[dimension] != shape[dimension]) return 0;
    return 1;
}

static float bf16_value(uint16_t value) {
    union { uint32_t bits; float value; } converted;
    converted.bits = (uint32_t)value << 16;
    return converted.value;
}

static uint16_t bf16_rounded_bits(float value) {
    union { uint32_t bits; float value; } converted;
    uint16_t rounded;
    converted.value = value;
    if ((converted.bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000))
        rounded = (uint16_t)((converted.bits >> 16) | UINT32_C(64));
    else
        rounded = (uint16_t)((converted.bits + UINT32_C(0x7fff) +
                              ((converted.bits >> 16) & UINT32_C(1))) >> 16);
    return rounded;
}

static const char *vision_compat_mode(void) {
    const char *mode = getenv("COLI_GEMMA4_VISION_COMPAT");
    return mode ? mode : "";
}

static int vision_compat_has(const char *feature) {
    const char *mode = vision_compat_mode();
    if (strcmp(mode, "llama-avx2") == 0)
        return strcmp(feature, "rms64") == 0 ||
               strcmp(feature, "ropeiter") == 0 ||
               strcmp(feature, "fma8") == 0 ||
               strcmp(feature, "patchvecdot") == 0 ||
               strcmp(feature, "flashtiled") == 0 ||
               strcmp(feature, "gelu16") == 0;
    return strstr(mode, feature) != NULL;
}

static float bf16_dot_fma8(const uint16_t *left, const uint16_t *right,
                           uint32_t count) {
    float sums[8] = { 0.0F, 0.0F, 0.0F, 0.0F,
                      0.0F, 0.0F, 0.0F, 0.0F };
    uint32_t index = 0;
    for (; index + 8U <= count; index += 8U) {
        uint32_t lane;
        for (lane = 0; lane < 8U; ++lane)
            sums[lane] = fmaf(bf16_value(left[index + lane]),
                              bf16_value(right[index + lane]), sums[lane]);
    }
    {
        float pair0 = sums[0] + sums[4];
        float pair1 = sums[1] + sums[5];
        float pair2 = sums[2] + sums[6];
        float pair3 = sums[3] + sums[7];
        float total = (pair0 + pair2) + (pair1 + pair3);
        for (; index < count; ++index)
            total += bf16_value(left[index]) * bf16_value(right[index]);
        return total;
    }
}

static float bf16_dot_vecdot(const uint16_t *left, const uint16_t *right,
                             uint32_t count) {
    float sums[4][8] = { { 0.0F } };
    uint32_t index = 0;
    for (; index + 32U <= count; index += 32U) {
        uint32_t vector, lane;
        for (vector = 0; vector < 4U; ++vector)
            for (lane = 0; lane < 8U; ++lane) {
                uint32_t position = index + vector * 8U + lane;
                float product = bf16_value(left[position]) *
                                bf16_value(right[position]);
                sums[vector][lane] += product;
            }
    }
    {
        float lanes[8];
        float pair0, pair1, pair2, pair3, total;
        uint32_t lane;
        for (lane = 0; lane < 8U; ++lane)
            lanes[lane] = (sums[0][lane] + sums[2][lane]) +
                          (sums[1][lane] + sums[3][lane]);
        pair0 = lanes[0] + lanes[4];
        pair1 = lanes[1] + lanes[5];
        pair2 = lanes[2] + lanes[6];
        pair3 = lanes[3] + lanes[7];
        total = (pair0 + pair2) + (pair1 + pair3);
        for (; index < count; ++index)
            total += bf16_value(left[index]) * bf16_value(right[index]);
        return total;
    }
}

static float f32_from_bits(uint32_t bits) {
    union { uint32_t bits; float value; } converted;
    converted.bits = bits;
    return converted.value;
}

/* Scalar transcription of ggml_v_expf's AVX2/FMA polynomial. The vision
   softmax only evaluates non-positive values close enough to zero to use the
   fast path; extreme inputs retain libm's range handling. */
static float vision_fast_expf(float x) {
    const float r = f32_from_bits(UINT32_C(0x4b400000));
    float z = fmaf(x, f32_from_bits(UINT32_C(0x3fb8aa3b)), r);
    float n = z - r;
    float b, u, polynomial, tail;
    uint32_t exponent;
    if (!isfinite(x)) return x < 0.0F ? 0.0F : expf(x);
    if (fabsf(n) > 126.0F) return expf(x);
    b = fmaf(-n, f32_from_bits(UINT32_C(0x35bfbe8e)),
             fmaf(-n, f32_from_bits(UINT32_C(0x3f317200)), x));
    {
        union { uint32_t bits; float value; } converted;
        converted.value = z;
        exponent = converted.bits << 23;
    }
    u = b * b;
    polynomial = fmaf(f32_from_bits(UINT32_C(0x3c072010)), b,
                      f32_from_bits(UINT32_C(0x3d2b9f17)));
    tail = fmaf(f32_from_bits(UINT32_C(0x3e2aaf33)), b,
                f32_from_bits(UINT32_C(0x3efffedb)));
    polynomial = fmaf(polynomial, u, tail);
    polynomial = fmaf(polynomial, u,
                      f32_from_bits(UINT32_C(0x3f7ffff6)) * b);
    {
        float scale = f32_from_bits(exponent + UINT32_C(0x3f800000));
        return fmaf(polynomial, scale, scale);
    }
}

#ifdef COLI_VISION_HAS_AVX2
/* Match ggml's AVX2 ggml_v_expf fast path lane-for-lane.  Keeping this as an
   intrinsic implementation matters here: a scalar transcription changes the
   fused-operation and horizontal-reduction boundaries used by flash attention. */
static __m256 vision_fast_exp8(__m256 x) {
    const __m256 r = _mm256_set1_ps(12582912.0F);
    const __m256 z = _mm256_fmadd_ps(
        x, _mm256_set1_ps(f32_from_bits(UINT32_C(0x3fb8aa3b))), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(
        n, _mm256_set1_ps(f32_from_bits(UINT32_C(0x35bfbe8e))),
        _mm256_fnmadd_ps(
            n, _mm256_set1_ps(f32_from_bits(UINT32_C(0x3f317200))), x));
    const __m256i exponent = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 scale = _mm256_castsi256_ps(_mm256_add_epi32(
        exponent, _mm256_castps_si256(_mm256_set1_ps(1.0F))));
    const __m256 square = _mm256_mul_ps(b, b);
    const __m256 polynomial = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32_from_bits(UINT32_C(0x3c072010))), b,
                _mm256_set1_ps(f32_from_bits(UINT32_C(0x3d2b9f17)))),
            square,
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32_from_bits(UINT32_C(0x3e2aaf33))), b,
                _mm256_set1_ps(f32_from_bits(UINT32_C(0x3efffedb))))),
        square,
        _mm256_mul_ps(
            _mm256_set1_ps(f32_from_bits(UINT32_C(0x3f7ffff6))), b));
    return _mm256_fmadd_ps(polynomial, scale, scale);
}

#endif

static float f32_dot_vecdot(const float *left, const float *right,
                            uint32_t count) {
#ifdef COLI_VISION_HAS_AVX2
    __m256 sums[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    uint32_t index = 0;
    float total;
    for (; index + 32U <= count; index += 32U) {
        uint32_t vector;
        for (vector = 0; vector < 4U; ++vector) {
            uint32_t offset = index + vector * 8U;
            sums[vector] = _mm256_fmadd_ps(
                _mm256_loadu_ps(left + offset),
                _mm256_loadu_ps(right + offset), sums[vector]);
        }
    }
    sums[0] = _mm256_add_ps(sums[0], sums[2]);
    sums[1] = _mm256_add_ps(sums[1], sums[3]);
    sums[0] = _mm256_add_ps(sums[0], sums[1]);
    {
        __m128 reduced = _mm_add_ps(
            _mm256_castps256_ps128(sums[0]),
            _mm256_extractf128_ps(sums[0], 1));
        reduced = _mm_hadd_ps(reduced, reduced);
        reduced = _mm_hadd_ps(reduced, reduced);
        total = _mm_cvtss_f32(reduced);
    }
    for (; index < count; ++index)
        total += left[index] * right[index];
    return total;
#else
    float sums[4][8] = { { 0.0F } };
    uint32_t index = 0;
    for (; index + 32U <= count; index += 32U) {
        uint32_t vector, lane;
        for (vector = 0; vector < 4U; ++vector)
            for (lane = 0; lane < 8U; ++lane) {
                uint32_t position = index + vector * 8U + lane;
                sums[vector][lane] = fmaf(
                    left[position], right[position], sums[vector][lane]);
            }
    }
    {
        float lanes[8];
        float low0, low1, high0, high1, total;
        uint32_t lane;
        for (lane = 0; lane < 8U; ++lane)
            lanes[lane] = (sums[0][lane] + sums[2][lane]) +
                          (sums[1][lane] + sums[3][lane]);
        low0 = (lanes[0] + lanes[4]) + (lanes[1] + lanes[5]);
        low1 = (lanes[2] + lanes[6]) + (lanes[3] + lanes[7]);
        high0 = low0;
        high1 = low1;
        total = high0 + high1;
        for (; index < count; ++index)
            total += left[index] * right[index];
        return total;
    }
#endif
}

/* llamafile's AVX2 tinyBLAS F32 kernel accumulates consecutive groups into a
   single eight-lane register, then performs the same low/high horizontal sum
   used by hsum(__m256). */
static float f32_dot_fma8(const float *left, const float *right,
                          uint32_t count) {
    float sums[8] = { 0.0F, 0.0F, 0.0F, 0.0F,
                      0.0F, 0.0F, 0.0F, 0.0F };
    uint32_t index = 0;
    for (; index + 8U <= count; index += 8U) {
        uint32_t lane;
        for (lane = 0; lane < 8U; ++lane)
            sums[lane] = fmaf(left[index + lane], right[index + lane],
                              sums[lane]);
    }
    {
        float pair0 = sums[0] + sums[4];
        float pair1 = sums[1] + sums[5];
        float pair2 = sums[2] + sums[6];
        float pair3 = sums[3] + sums[7];
        float total = (pair0 + pair2) + (pair1 + pair3);
        for (; index < count; ++index)
            total += left[index] * right[index];
        return total;
    }
}

static uint16_t vision_fp16_bits(float value) {
    union { float value; uint32_t bits; } converted;
    const float scale_to_inf = f32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = f32_from_bits(UINT32_C(0x08800000));
    float base = (fabsf(value) * scale_to_inf) * scale_to_zero;
    uint32_t shifted, sign, bias, bits, exponent, mantissa, nonsign;
    converted.value = value;
    shifted = converted.bits + converted.bits;
    sign = converted.bits & UINT32_C(0x80000000);
    bias = shifted & UINT32_C(0xff000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
    base = f32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    converted.value = base;
    bits = converted.bits;
    exponent = (bits >> 13) & UINT32_C(0x00007c00);
    mantissa = bits & UINT32_C(0x00000fff);
    nonsign = exponent + mantissa;
    return (uint16_t)((sign >> 16) |
        (shifted > UINT32_C(0xff000000) ? UINT32_C(0x7e00) : nonsign));
}

static float vision_fp16_value(uint16_t value) {
    uint32_t word = (uint32_t)value << 16;
    uint32_t sign = word & UINT32_C(0x80000000);
    uint32_t twice = word + word;
    float normalized = f32_from_bits(
        (twice >> 4) + UINT32_C(0x70000000)) *
        f32_from_bits(UINT32_C(0x07800000));
    float denormalized = f32_from_bits(
        (twice >> 17) | UINT32_C(0x3f000000)) - 0.5F;
    uint32_t magnitude;
    union { float value; uint32_t bits; } converted;
    converted.value = twice < UINT32_C(0x08000000) ?
        denormalized : normalized;
    magnitude = converted.bits;
    return f32_from_bits(sign | magnitude);
}

static float fp16_dot_vecdot(const uint16_t *left, const uint16_t *right,
                             uint32_t count) {
    float sums[4][8] = { { 0.0F } };
    uint32_t index = 0;
    for (; index + 32U <= count; index += 32U) {
        uint32_t vector, lane;
        for (vector = 0; vector < 4U; ++vector)
            for (lane = 0; lane < 8U; ++lane) {
                uint32_t position = index + vector * 8U + lane;
                sums[vector][lane] = fmaf(
                    vision_fp16_value(left[position]),
                    vision_fp16_value(right[position]), sums[vector][lane]);
            }
    }
    {
        float lanes[8];
        float low0, low1;
        double total;
        uint32_t lane;
        for (lane = 0; lane < 8U; ++lane)
            lanes[lane] = (sums[0][lane] + sums[2][lane]) +
                          (sums[1][lane] + sums[3][lane]);
        low0 = (lanes[0] + lanes[4]) + (lanes[1] + lanes[5]);
        low1 = (lanes[2] + lanes[6]) + (lanes[3] + lanes[7]);
        total = (double)(low0 + low1);
        for (; index < count; ++index)
            total += (double)(vision_fp16_value(left[index]) *
                              vision_fp16_value(right[index]));
        return (float)total;
    }
}

static int vision_flash_attention_f16(float *output, const float *q,
                                      const float *k, const float *v,
                                      uint32_t patch_count,
                                      uint32_t head_count,
                                      uint32_t head_dim) {
    int64_t item;
    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (item = 0; item < (int64_t)patch_count * head_count; ++item) {
        uint32_t query_index = (uint32_t)item / head_count;
        uint32_t head_index = (uint32_t)item % head_count;
        const float *query = q +
            ((size_t)query_index * head_count + head_index) * head_dim;
        uint16_t *query16 = (uint16_t *)malloc(
            (size_t)head_dim * sizeof(*query16));
        uint16_t *key16 = (uint16_t *)malloc(
            (size_t)head_dim * sizeof(*key16));
        uint16_t *accumulator16 = (uint16_t *)calloc(
            head_dim, sizeof(*accumulator16));
        float maximum = -INFINITY;
        float sum = 0.0F;
        uint32_t dimension, key_index;
        if (!query16 || !key16 || !accumulator16) {
            free(query16); free(key16); free(accumulator16);
            failed = 1;
            continue;
        }
        for (dimension = 0; dimension < head_dim; ++dimension)
            query16[dimension] = vision_fp16_bits(query[dimension]);
        for (key_index = 0; key_index < patch_count; ++key_index) {
            const float *key = k +
                ((size_t)key_index * head_count + head_index) * head_dim;
            float score, old_maximum, maximum_scale = 1.0F;
            float value_scale = 1.0F;
            for (dimension = 0; dimension < head_dim; ++dimension)
                key16[dimension] = vision_fp16_bits(key[dimension]);
            score = fp16_dot_vecdot(key16, query16, head_dim);
            old_maximum = maximum;
            if (score > maximum) {
                maximum = score;
                maximum_scale = expf(old_maximum - maximum);
                for (dimension = 0; dimension < head_dim; ++dimension)
                    accumulator16[dimension] = vision_fp16_bits(
                        vision_fp16_value(accumulator16[dimension]) *
                        maximum_scale);
            } else {
                value_scale = expf(score - maximum);
            }
            for (dimension = 0; dimension < head_dim; ++dimension) {
                float value = vision_fp16_value(vision_fp16_bits(v[
                    ((size_t)key_index * head_count + head_index) *
                    head_dim + dimension]));
                accumulator16[dimension] = vision_fp16_bits(fmaf(
                    value, value_scale,
                    vision_fp16_value(accumulator16[dimension])));
            }
            sum = sum * maximum_scale + value_scale;
        }
        {
            float inverse = sum == 0.0F ? 0.0F : 1.0F / sum;
            for (dimension = 0; dimension < head_dim; ++dimension)
                output[((size_t)query_index * head_count + head_index) *
                       head_dim + dimension] =
                    vision_fp16_value(accumulator16[dimension]) * inverse;
        }
        free(query16); free(key16); free(accumulator16);
    }
    return failed ? -1 : 0;
}

static double vision_softmax_tile64(float scores[64], float maximum) {
    double sum = 0.0;
    uint32_t index;
    for (index = 0; index < 64U; index += 8U) {
#ifdef COLI_VISION_HAS_AVX2
        __m256 input = _mm256_loadu_ps(scores + index);
        __m256 padded = _mm256_cmp_ps(
            input, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ);
        __m256 lanes = vision_fast_exp8(_mm256_sub_ps(
            input, _mm256_set1_ps(maximum)));
        __m128 reduced;
        lanes = _mm256_andnot_ps(padded, lanes);
        _mm256_storeu_ps(scores + index, lanes);
        reduced = _mm_add_ps(
            _mm256_extractf128_ps(lanes, 1), _mm256_castps256_ps128(lanes));
        reduced = _mm_add_ps(reduced, _mm_movehl_ps(reduced, reduced));
        reduced = _mm_add_ss(reduced, _mm_movehdup_ps(reduced));
        sum += (double)_mm_cvtss_f32(reduced);
#else
        float lanes[8];
        float pair0, pair1, pair2, pair3;
        uint32_t lane;
        for (lane = 0; lane < 8U; ++lane) {
            lanes[lane] = vision_fast_expf(scores[index + lane] - maximum);
            scores[index + lane] = lanes[lane];
        }
        pair0 = lanes[0] + lanes[4];
        pair1 = lanes[1] + lanes[5];
        pair2 = lanes[2] + lanes[6];
        pair3 = lanes[3] + lanes[7];
        sum += (double)((pair0 + pair2) + (pair1 + pair3));
#endif
    }
    return sum;
}

static int vision_flash_attention_tiled(float *output, const float *q,
                                        const float *k, const float *v,
                                        uint32_t patch_count,
                                        uint32_t head_count,
                                        uint32_t head_dim) {
    int64_t item;
    int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (item = 0; item < (int64_t)patch_count * head_count; ++item) {
        uint32_t query_index = (uint32_t)item / head_count;
        uint32_t head_index = (uint32_t)item % head_count;
        const float *query = q +
            ((size_t)query_index * head_count + head_index) * head_dim;
        float *accumulator = (float *)calloc(head_dim, sizeof(*accumulator));
        float maximum = -INFINITY;
        float sum = 0.0F;
        uint32_t tile_start, dimension;
        if (!accumulator) {
            failed = 1;
            continue;
        }
        for (tile_start = 0; tile_start < patch_count; tile_start += 64U) {
            uint32_t tile_count = patch_count - tile_start;
            float scores[64];
            float tile_maximum = -INFINITY;
            float new_maximum, maximum_scale = 1.0F;
            uint32_t tile_index;
            if (tile_count > 64U) tile_count = 64U;
#ifdef COLI_VISION_HAS_AVX2
            for (tile_index = 0; tile_index < 64U; tile_index += 8U) {
                __m256 score8 = _mm256_setzero_ps();
                for (dimension = 0; dimension < head_dim; ++dimension) {
                    float key_lanes[8];
                    uint32_t lane;
                    for (lane = 0; lane < 8U; ++lane) {
                        uint32_t key_in_tile = tile_index + lane;
                        if (key_in_tile < tile_count) {
                            const float *key = k +
                                ((size_t)(tile_start + key_in_tile) *
                                 head_count + head_index) * head_dim;
                            key_lanes[lane] = vision_fp16_value(
                                vision_fp16_bits(key[dimension]));
                        } else {
                            key_lanes[lane] = 0.0F;
                        }
                    }
                    score8 = _mm256_fmadd_ps(
                        _mm256_set1_ps(query[dimension]),
                        _mm256_loadu_ps(key_lanes), score8);
                }
                _mm256_storeu_ps(scores + tile_index, score8);
            }
            for (tile_index = 0; tile_index < tile_count; ++tile_index)
                if (scores[tile_index] > tile_maximum)
                    tile_maximum = scores[tile_index];
#else
            for (tile_index = 0; tile_index < tile_count; ++tile_index) {
                const float *key = k +
                    ((size_t)(tile_start + tile_index) * head_count +
                     head_index) * head_dim;
                float score = 0.0F;
                for (dimension = 0; dimension < head_dim; ++dimension)
                    score = fmaf(query[dimension],
                                 vision_fp16_value(
                                     vision_fp16_bits(key[dimension])),
                                 score);
                scores[tile_index] = score;
                if (score > tile_maximum) tile_maximum = score;
            }
#endif
            for (tile_index = tile_count; tile_index < 64U; ++tile_index)
                scores[tile_index] = -INFINITY;
            new_maximum = fmaxf(maximum, tile_maximum);
            if (new_maximum > maximum) {
                maximum_scale = expf(maximum - new_maximum);
                for (dimension = 0; dimension < head_dim; ++dimension)
                    accumulator[dimension] *= maximum_scale;
                sum *= maximum_scale;
            }
            maximum = new_maximum;
            sum = (float)((double)sum +
                          vision_softmax_tile64(scores, maximum));
#ifdef COLI_VISION_HAS_AVX2
            for (dimension = 0; dimension + 8U <= head_dim;
                 dimension += 8U) {
                __m256 accumulated = _mm256_loadu_ps(
                    accumulator + dimension);
                for (tile_index = 0; tile_index < 64U; ++tile_index) {
                    __m256 value8 = _mm256_setzero_ps();
                    if (tile_index < tile_count) {
                        const float *value = v +
                            ((size_t)(tile_start + tile_index) * head_count +
                             head_index) * head_dim + dimension;
                        value8 = _mm256_cvtph_ps(_mm256_cvtps_ph(
                            _mm256_loadu_ps(value), _MM_FROUND_TO_NEAREST_INT));
                    }
                    accumulated = _mm256_fmadd_ps(
                        _mm256_set1_ps(scores[tile_index]), value8,
                        accumulated);
                }
                _mm256_storeu_ps(accumulator + dimension, accumulated);
            }
            for (; dimension < head_dim; ++dimension)
                for (tile_index = 0; tile_index < 64U; ++tile_index) {
                    float value = 0.0F;
                    if (tile_index < tile_count)
                        value = vision_fp16_value(vision_fp16_bits(v[
                            ((size_t)(tile_start + tile_index) * head_count +
                             head_index) * head_dim + dimension]));
                    accumulator[dimension] = fmaf(
                        scores[tile_index], value, accumulator[dimension]);
                }
#else
            for (dimension = 0; dimension < head_dim; ++dimension)
                for (tile_index = 0; tile_index < 64U; ++tile_index) {
                    float value = 0.0F;
                    if (tile_index < tile_count)
                        value = vision_fp16_value(vision_fp16_bits(v[
                            ((size_t)(tile_start + tile_index) * head_count +
                             head_index) * head_dim + dimension]));
                    accumulator[dimension] = fmaf(
                        scores[tile_index], value, accumulator[dimension]);
                }
#endif
        }
        {
            float inverse = sum == 0.0F ? 0.0F : 1.0F / sum;
            for (dimension = 0; dimension < head_dim; ++dimension)
                output[((size_t)query_index * head_count + head_index) *
                       head_dim + dimension] =
                    accumulator[dimension] * inverse;
        }
        free(accumulator);
    }
    return failed ? -1 : 0;
}

static int read_named(const coli_gemma4_vision *vision, const char *name,
                      uint32_t type, uint32_t dimensions,
                      const uint64_t *shape, void **storage) {
    const coli_gemma4_tensor *tensor =
        coli_gemma4_gguf_find(&vision->gguf, name);
    void *values;
    if (!tensor_shape(tensor, type, dimensions, shape) ||
        tensor->nbytes > SIZE_MAX) return -1;
    values = malloc((size_t)tensor->nbytes);
    if (!values || coli_gemma4_gguf_read(
            &vision->gguf, tensor, values, (size_t)tensor->nbytes) != 0) {
        free(values);
        return -1;
    }
    *storage = values;
    return 0;
}

static void vision_block_close(vision_block *block) {
    if (!block) return;
    free(block->q); free(block->k); free(block->v); free(block->output);
    free(block->ff_up); free(block->ff_gate); free(block->ff_down);
    free(block->ln1); free(block->ln2); free(block->q_norm); free(block->k_norm);
    free(block->attention_post_norm); free(block->ffn_post_norm);
    memset(block, 0, sizeof(*block));
}

static int vision_block_open(const coli_gemma4_vision *vision,
                             uint32_t layer, vision_block *block) {
    char name[128];
    uint64_t square[2], up[2], down[2], vector[1], head[1];
    uint32_t width = vision->gguf.vision_embedding_length;
    uint32_t ff = vision->gguf.vision_feed_forward_length;
    uint32_t head_dim;
#define READ_BLOCK(field, suffix, type, dims, shape) do { \
    if (snprintf(name, sizeof(name), "v.blk.%u.%s", layer, suffix) < 0 || \
        read_named(vision, name, type, dims, shape, (void **)&block->field) != 0) \
        goto failure; \
} while (0)
    if (!block || !width || !ff || !vision->gguf.vision_head_count ||
        width % vision->gguf.vision_head_count) return -1;
    head_dim = width / vision->gguf.vision_head_count;
    memset(block, 0, sizeof(*block));
    square[0] = width; square[1] = width;
    up[0] = width; up[1] = ff;
    down[0] = ff; down[1] = width;
    vector[0] = width; head[0] = head_dim;
    READ_BLOCK(ln1, "ln1.weight", COLI_GGML_TYPE_F32, 1, vector);
    READ_BLOCK(ln2, "ln2.weight", COLI_GGML_TYPE_F32, 1, vector);
    READ_BLOCK(q_norm, "attn_q_norm.weight", COLI_GGML_TYPE_F32, 1, head);
    READ_BLOCK(k_norm, "attn_k_norm.weight", COLI_GGML_TYPE_F32, 1, head);
    READ_BLOCK(attention_post_norm, "attn_post_norm.weight",
               COLI_GGML_TYPE_F32, 1, vector);
    READ_BLOCK(ffn_post_norm, "ffn_post_norm.weight",
               COLI_GGML_TYPE_F32, 1, vector);
    READ_BLOCK(q, "attn_q.weight", COLI_GGML_TYPE_BF16, 2, square);
    READ_BLOCK(k, "attn_k.weight", COLI_GGML_TYPE_BF16, 2, square);
    READ_BLOCK(v, "attn_v.weight", COLI_GGML_TYPE_BF16, 2, square);
    READ_BLOCK(output, "attn_out.weight", COLI_GGML_TYPE_BF16, 2, square);
    READ_BLOCK(ff_up, "ffn_up.weight", COLI_GGML_TYPE_BF16, 2, up);
    READ_BLOCK(ff_gate, "ffn_gate.weight", COLI_GGML_TYPE_BF16, 2, up);
    READ_BLOCK(ff_down, "ffn_down.weight", COLI_GGML_TYPE_BF16, 2, down);
#undef READ_BLOCK
    return 0;
failure:
#undef READ_BLOCK
    vision_block_close(block);
    return -1;
}

static void bf16_matmul(float *outputs, const float *inputs,
                        const uint16_t *weights,
                        uint32_t rows, uint32_t input_width,
                        uint32_t output_width) {
    uint16_t *rounded_inputs = NULL;
    const char *compat_mode = vision_compat_mode();
    int compat_fma8 = strcmp(compat_mode, "llama-avx2") == 0 ||
                      strstr(compat_mode, "fma8") != NULL;
    int compat_vecdot = strstr(compat_mode, "vecdot") != NULL;
    size_t input_values = (size_t)rows * input_width;
    int64_t output_row;
    if (input_values <= SIZE_MAX / sizeof(*rounded_inputs))
        rounded_inputs = (uint16_t *)malloc(input_values * sizeof(*rounded_inputs));
    if (rounded_inputs) {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)input_values; ++index)
            rounded_inputs[index] = bf16_rounded_bits(inputs[index]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (output_row = 0; output_row < (int64_t)output_width; ++output_row) {
        const uint16_t *weight = weights + output_row * input_width;
        uint32_t row;
        for (row = 0; row < rows; ++row) {
            const float *input = inputs + (size_t)row * input_width;
            const uint16_t *rounded = rounded_inputs ?
                rounded_inputs + (size_t)row * input_width : NULL;
            /* ggml's BF16 mul_mat converts the F32 activation row to the
               weight's BF16 vec-dot type before evaluating the product. */
            float sum;
            if (rounded && compat_fma8) {
                sum = bf16_dot_fma8(rounded, weight, input_width);
            } else if (rounded && compat_vecdot) {
                sum = bf16_dot_vecdot(rounded, weight, input_width);
            } else {
                uint32_t column;
                sum = 0.0F;
                for (column = 0; column < input_width; ++column)
                    sum += bf16_value(rounded ? rounded[column] :
                                      bf16_rounded_bits(input[column])) *
                           bf16_value(weight[column]);
            }
            outputs[(size_t)row * output_width + (size_t)output_row] = sum;
        }
    }
    free(rounded_inputs);
}

static void vision_rmsnorm(float *outputs, const float *inputs,
                           const float *weights, uint32_t rows,
                           uint32_t width, float epsilon) {
    int64_t row;
    int compat_rms64 = vision_compat_has("rms64");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < (int64_t)rows; ++row) {
        const float *input = inputs + row * width;
        float *output = outputs + row * width;
        double square_sum64 = 0.0;
        float square_sum = 0.0F;
        uint32_t column;
        if (compat_rms64) {
            for (column = 0; column < width; ++column)
                square_sum64 += (double)(input[column] * input[column]);
        } else {
            for (column = 0; column < width; ++column)
                square_sum += input[column] * input[column];
        }
        {
            float mean = compat_rms64 ?
                (float)(square_sum64 / (double)width) :
                square_sum / (float)width;
            float scale = 1.0F / sqrtf(mean + epsilon);
            for (column = 0; column < width; ++column)
                output[column] = input[column] * scale *
                                 (weights ? weights[column] : 1.0F);
        }
    }
}

static void vision_rope_2d(float *values, uint32_t patch_count,
                           uint32_t patch_columns, uint32_t head_count,
                           uint32_t head_dim, float theta) {
    int64_t item;
    uint32_t half = head_dim / 2U;
    uint32_t pairs = half / 2U;
    int compat_rope_iter = vision_compat_has("ropeiter");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (item = 0; item < (int64_t)patch_count * head_count; ++item) {
        uint32_t patch = (uint32_t)item / head_count;
        uint32_t position[2] = {
            patch % patch_columns,
            patch / patch_columns
        };
        float *head = values + (size_t)item * head_dim;
        uint32_t section;
        for (section = 0; section < 2U; ++section) {
            float *part = head + section * half;
            float theta_scale = powf(theta, -2.0F / (float)half);
            float iterated_angle = (float)position[section];
            uint32_t pair;
            for (pair = 0; pair < pairs; ++pair) {
                float angle;
                if (compat_rope_iter) {
                    angle = iterated_angle;
                    iterated_angle *= theta_scale;
                } else {
                    float frequency = powf(theta,
                        -2.0F * (float)pair / (float)half);
                    angle = (float)position[section] * frequency;
                }
                float cosine = cosf(angle);
                float sine = sinf(angle);
                float first = part[pair];
                float second = part[pair + pairs];
                part[pair] = first * cosine - second * sine;
                part[pair + pairs] = first * sine + second * cosine;
            }
        }
    }
}

static int vision_attention(float *output, const float *q, const float *k,
                            const float *v, uint32_t patch_count,
                            uint32_t head_count, uint32_t head_dim) {
    int64_t item;
    int failed = 0;
    if (vision_compat_has("flashtiled"))
        return vision_flash_attention_tiled(
            output, q, k, v, patch_count, head_count, head_dim);
    if (strstr(vision_compat_mode(), "flash16"))
        return vision_flash_attention_f16(
            output, q, k, v, patch_count, head_count, head_dim);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (item = 0; item < (int64_t)patch_count * head_count; ++item) {
        uint32_t query_index = (uint32_t)item / head_count;
        uint32_t head_index = (uint32_t)item % head_count;
        const float *query = q +
            ((size_t)query_index * head_count + head_index) * head_dim;
        float *scores = (float *)malloc((size_t)patch_count * sizeof(float));
        float *value_column = NULL;
        float maximum = -INFINITY;
        double total64 = 0.0;
        float total = 0.0F;
        int compat_fast_exp =
            strstr(vision_compat_mode(), "fastexp") != NULL;
        int compat_attn_vecdot =
            strstr(vision_compat_mode(), "attnvec") != NULL;
        int compat_attn_q_fma8 =
            strstr(vision_compat_mode(), "attnqfma8") != NULL;
        int compat_soft64 = compat_fast_exp ||
            strstr(vision_compat_mode(), "soft64") != NULL;
        uint32_t key_index, dimension;
        if (!scores) {
            failed = 1;
            continue;
        }
        if (compat_attn_vecdot) {
            value_column = (float *)malloc(
                (size_t)patch_count * sizeof(*value_column));
            if (!value_column) {
                free(scores);
                failed = 1;
                continue;
            }
        }
        for (key_index = 0; key_index < patch_count; ++key_index) {
            const float *key = k +
                ((size_t)key_index * head_count + head_index) * head_dim;
            float score;
            if (compat_attn_q_fma8)
                score = f32_dot_fma8(query, key, head_dim);
            else if (compat_attn_vecdot)
                score = f32_dot_vecdot(query, key, head_dim);
            else {
                score = 0.0F;
                for (dimension = 0; dimension < head_dim; ++dimension)
                    score += query[dimension] * key[dimension];
            }
            scores[key_index] = score;
            if (score > maximum) maximum = score;
        }
        if (compat_fast_exp) {
            for (key_index = 0; key_index + 8U <= patch_count;
                 key_index += 8U) {
#ifdef COLI_VISION_HAS_AVX2
                __m256 lanes = vision_fast_exp8(_mm256_sub_ps(
                    _mm256_loadu_ps(scores + key_index),
                    _mm256_set1_ps(maximum)));
                __m128 reduced;
                _mm256_storeu_ps(scores + key_index, lanes);
                reduced = _mm_add_ps(
                    _mm256_extractf128_ps(lanes, 1),
                    _mm256_castps256_ps128(lanes));
                reduced = _mm_add_ps(
                    reduced, _mm_movehl_ps(reduced, reduced));
                reduced = _mm_add_ss(reduced, _mm_movehdup_ps(reduced));
                total64 += (double)_mm_cvtss_f32(reduced);
#else
                float lanes[8];
                float pair0, pair1, pair2, pair3;
                uint32_t lane;
                for (lane = 0; lane < 8U; ++lane) {
                    lanes[lane] = vision_fast_expf(
                        scores[key_index + lane] - maximum);
                    scores[key_index + lane] = lanes[lane];
                }
                pair0 = lanes[0] + lanes[4];
                pair1 = lanes[1] + lanes[5];
                pair2 = lanes[2] + lanes[6];
                pair3 = lanes[3] + lanes[7];
                total64 += (double)((pair0 + pair2) + (pair1 + pair3));
#endif
            }
            for (; key_index < patch_count; ++key_index) {
                scores[key_index] = expf(scores[key_index] - maximum);
                total64 += (double)scores[key_index];
            }
        } else for (key_index = 0; key_index < patch_count; ++key_index) {
            scores[key_index] = expf(scores[key_index] - maximum);
            if (compat_soft64) total64 += (double)scores[key_index];
            else total += scores[key_index];
        }
        if (compat_soft64) total = (float)total64;
        if (!(total > 0.0F) || !isfinite(total)) {
            free(value_column);
            free(scores);
            failed = 1;
            continue;
        }
        {
            float inverse = compat_soft64 ?
                (float)(1.0 / total64) : 1.0F / total;
            for (key_index = 0; key_index < patch_count; ++key_index)
                scores[key_index] *= inverse;
        }
        for (dimension = 0; dimension < head_dim; ++dimension) {
            float sum;
            if (compat_attn_vecdot) {
                for (key_index = 0; key_index < patch_count; ++key_index)
                    value_column[key_index] = v[
                        ((size_t)key_index * head_count + head_index) *
                        head_dim + dimension];
                sum = f32_dot_vecdot(scores, value_column, patch_count);
            } else {
                sum = 0.0F;
                for (key_index = 0; key_index < patch_count; ++key_index) {
                    const float *value = v +
                        ((size_t)key_index * head_count + head_index) * head_dim;
                    sum += scores[key_index] * value[dimension];
                }
            }
            output[((size_t)query_index * head_count + head_index) *
                   head_dim + dimension] = sum;
        }
        free(value_column);
        free(scores);
    }
    return failed ? -1 : 0;
}

static int vision_block_apply(coli_gemma4_vision *vision,
                              const vision_block *block,
                              float *hidden, uint32_t patch_count,
                              uint32_t patch_columns, uint32_t layer) {
    uint32_t width = vision->gguf.vision_embedding_length;
    uint32_t ff = vision->gguf.vision_feed_forward_length;
    uint32_t heads = vision->gguf.vision_head_count;
    uint32_t head_dim;
    float epsilon = vision->gguf.vision_epsilon;
    size_t hidden_values = (size_t)patch_count * width;
    size_t ff_values = (size_t)patch_count * ff;
    float *norm = NULL, *q = NULL, *k = NULL, *v = NULL;
    float *context = NULL, *branch = NULL, *up = NULL, *gate = NULL;
    int compat_gelu16 = vision_compat_has("gelu16");
    int64_t index;
    int result = -1;
    if (!width || !ff || !heads || width % heads ||
        (size_t)patch_count > SIZE_MAX / width ||
        (size_t)patch_count > SIZE_MAX / ff ||
        hidden_values > SIZE_MAX / sizeof(float) ||
        ff_values > SIZE_MAX / sizeof(float))
        return -1;
    head_dim = width / heads;
    norm = (float *)malloc(hidden_values * sizeof(float));
    q = (float *)malloc(hidden_values * sizeof(float));
    k = (float *)malloc(hidden_values * sizeof(float));
    v = (float *)malloc(hidden_values * sizeof(float));
    context = (float *)malloc(hidden_values * sizeof(float));
    branch = (float *)malloc(hidden_values * sizeof(float));
    up = (float *)malloc(ff_values * sizeof(float));
    gate = (float *)malloc(ff_values * sizeof(float));
    if (!norm || !q || !k || !v || !context || !branch || !up || !gate)
        goto cleanup;

    if (vision_trace_f32(vision, layer, "input", hidden, hidden_values) != 0)
        goto cleanup;
    vision_rmsnorm(norm, hidden, block->ln1, patch_count, width, epsilon);
    if (vision_trace_f32(vision, layer, "attention-input-norm", norm,
                         hidden_values) != 0) goto cleanup;
    bf16_matmul(q, norm, block->q, patch_count, width, width);
    bf16_matmul(k, norm, block->k, patch_count, width, width);
    bf16_matmul(v, norm, block->v, patch_count, width, width);
    vision_rmsnorm(q, q, block->q_norm, patch_count * heads,
                   head_dim, epsilon);
    vision_rmsnorm(k, k, block->k_norm, patch_count * heads,
                   head_dim, epsilon);
    if (vision_trace_f32(vision, layer, "query-norm", q, hidden_values) != 0 ||
        vision_trace_f32(vision, layer, "key-norm", k, hidden_values) != 0 ||
        vision_trace_f32(vision, layer, "value", v, hidden_values) != 0)
        goto cleanup;
    vision_rope_2d(q, patch_count, patch_columns, heads, head_dim, 100.0F);
    vision_rope_2d(k, patch_count, patch_columns, heads, head_dim, 100.0F);
    vision_rmsnorm(v, v, NULL, patch_count * heads, head_dim, epsilon);
    if (vision_trace_f32(vision, layer, "query-rope", q, hidden_values) != 0 ||
        vision_trace_f32(vision, layer, "key-rope", k, hidden_values) != 0 ||
        vision_trace_f32(vision, layer, "value-norm", v, hidden_values) != 0)
        goto cleanup;
    if (vision_attention(context, q, k, v, patch_count,
                         heads, head_dim) != 0)
        goto cleanup;
    if (vision_trace_f32(vision, layer, "attention-context", context,
                         hidden_values) != 0) goto cleanup;
    bf16_matmul(branch, context, block->output,
                patch_count, width, width);
    if (vision_trace_f32(vision, layer, "attention-output", branch,
                         hidden_values) != 0) goto cleanup;
    vision_rmsnorm(branch, branch, block->attention_post_norm,
                   patch_count, width, epsilon);
    if (vision_trace_f32(vision, layer, "attention-post-norm", branch,
                         hidden_values) != 0) goto cleanup;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index = 0; index < (int64_t)hidden_values; ++index)
        hidden[index] += branch[index];
    if (vision_trace_f32(vision, layer, "attention-residual", hidden,
                         hidden_values) != 0) goto cleanup;

    vision_rmsnorm(norm, hidden, block->ln2, patch_count, width, epsilon);
    if (vision_trace_f32(vision, layer, "ffn-input-norm", norm,
                         hidden_values) != 0) goto cleanup;
    bf16_matmul(up, norm, block->ff_up, patch_count, width, ff);
    bf16_matmul(gate, norm, block->ff_gate, patch_count, width, ff);
    if (vision_trace_f32(vision, layer, "ffn-up", up, ff_values) != 0 ||
        vision_trace_f32(vision, layer, "ffn-gate", gate, ff_values) != 0)
        goto cleanup;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index = 0; index < (int64_t)ff_values; ++index) {
        float value = gate[index];
        if (compat_gelu16) {
            float rounded = vision_fp16_value(vision_fp16_bits(value));
            float activated = rounded *
                (1.0F / (1.0F + expf(-1.702F * rounded)));
            gate[index] = up[index] *
                vision_fp16_value(vision_fp16_bits(activated));
        } else {
            gate[index] = up[index] * value /
                          (1.0F + expf(-1.702F * value));
        }
    }
    if (vision_trace_f32(vision, layer, "ffn-activated", gate,
                         ff_values) != 0) goto cleanup;
    bf16_matmul(branch, gate, block->ff_down, patch_count, ff, width);
    if (vision_trace_f32(vision, layer, "ffn-output", branch,
                         hidden_values) != 0) goto cleanup;
    vision_rmsnorm(branch, branch, block->ffn_post_norm,
                   patch_count, width, epsilon);
    if (vision_trace_f32(vision, layer, "ffn-post-norm", branch,
                         hidden_values) != 0) goto cleanup;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index = 0; index < (int64_t)hidden_values; ++index)
        hidden[index] += branch[index];
    if (vision_trace_f32(vision, layer, "output", hidden,
                         hidden_values) != 0) goto cleanup;
    result = 0;
cleanup:
    free(norm); free(q); free(k); free(v); free(context); free(branch);
    free(up); free(gate);
    return result;
}

int coli_gemma4_vision_open(coli_gemma4_vision *vision, const char *path) {
    const coli_gemma4_tensor *patch, *position, *projection, *bias, *scale;
    uint64_t patch_shape[4], projection_shape[2], vector_shape[1];
    if (!vision || !path) return -1;
    memset(vision, 0, sizeof(*vision));
    vision->merge_size = 3;
    vision->minimum_tokens = 40;
    vision->maximum_tokens = 280;
    if (coli_gemma4_gguf_open(&vision->gguf, path) != 0) {
        vision_error(vision, "%s", coli_gemma4_gguf_last_error(&vision->gguf));
        return -1;
    }
    patch_shape[0] = vision->gguf.vision_patch_size;
    patch_shape[1] = vision->gguf.vision_patch_size;
    patch_shape[2] = 3;
    patch_shape[3] = vision->gguf.vision_embedding_length;
    projection_shape[0] = vision->gguf.vision_embedding_length;
    projection_shape[1] = vision->gguf.vision_projection_dim;
    vector_shape[0] = vision->gguf.vision_embedding_length;
    patch = coli_gemma4_gguf_find(&vision->gguf, "v.patch_embd.weight");
    position = coli_gemma4_gguf_find(&vision->gguf, "v.position_embd.weight");
    projection = coli_gemma4_gguf_find(&vision->gguf, "mm.input_projection.weight");
    bias = coli_gemma4_gguf_find(&vision->gguf, "v.std_bias");
    scale = coli_gemma4_gguf_find(&vision->gguf, "v.std_scale");
    if (!tensor_shape(patch, COLI_GGML_TYPE_F32, 4, patch_shape) ||
        !position || position->type != COLI_GGML_TYPE_F32 ||
        position->n_dims != 3 ||
        position->dims[0] != vision->gguf.vision_embedding_length ||
        !position->dims[1] || position->dims[2] != 2 ||
        !tensor_shape(projection, COLI_GGML_TYPE_BF16, 2, projection_shape) ||
        !tensor_shape(bias, COLI_GGML_TYPE_F32, 1, vector_shape) ||
        !tensor_shape(scale, COLI_GGML_TYPE_F32, 1, vector_shape)) {
        vision_error(vision, "Gemma 4 vision core tensors have unexpected shapes");
        coli_gemma4_gguf_close(&vision->gguf);
        return -1;
    }
    return 0;
}

void coli_gemma4_vision_close(coli_gemma4_vision *vision) {
    if (!vision) return;
    coli_gemma4_gguf_close(&vision->gguf);
    memset(vision, 0, sizeof(*vision));
}

const char *coli_gemma4_vision_last_error(const coli_gemma4_vision *vision) {
    return vision ? vision->last_error : "invalid Gemma 4 vision handle";
}

static uint32_t aligned_round(uint32_t value, uint32_t factor) {
    return (uint32_t)(roundf((float)value / (float)factor) * (float)factor);
}

static uint32_t aligned_floor(float value, uint32_t factor) {
    return (uint32_t)(floorf(value / (float)factor) * (float)factor);
}

static uint32_t aligned_ceil(float value, uint32_t factor) {
    return (uint32_t)(ceilf(value / (float)factor) * (float)factor);
}

int coli_gemma4_vision_target_size(const coli_gemma4_vision *vision,
                                   uint32_t source_width,
                                   uint32_t source_height,
                                   uint32_t *target_width,
                                   uint32_t *target_height,
                                   uint32_t *token_count) {
    uint32_t alignment, width, height;
    uint64_t area, patch_area, minimum_pixels, maximum_pixels;
    if (!vision || !vision->gguf.vision_patch_size || !vision->merge_size ||
        !source_width || !source_height || !target_width || !target_height ||
        !token_count ||
        source_width > 1000000U || source_height > 1000000U ||
        vision->gguf.vision_patch_size > UINT32_MAX / vision->merge_size)
        return -1;
    alignment = vision->gguf.vision_patch_size * vision->merge_size;
    patch_area = (uint64_t)alignment * alignment;
    minimum_pixels = patch_area * vision->minimum_tokens;
    maximum_pixels = patch_area * vision->maximum_tokens;
    width = aligned_round(source_width, alignment);
    height = aligned_round(source_height, alignment);
    if (width < alignment) width = alignment;
    if (height < alignment) height = alignment;
    area = (uint64_t)width * height;
    if (area > maximum_pixels) {
        float beta = sqrtf((float)((uint64_t)source_width * source_height) /
                           (float)maximum_pixels);
        width = aligned_floor((float)source_width / beta, alignment);
        height = aligned_floor((float)source_height / beta, alignment);
        if (width < alignment) width = alignment;
        if (height < alignment) height = alignment;
    } else if (area < minimum_pixels) {
        float beta = sqrtf((float)minimum_pixels /
                           (float)((uint64_t)source_width * source_height));
        width = aligned_ceil((float)source_width * beta, alignment);
        height = aligned_ceil((float)source_height * beta, alignment);
    }
    if (!width || !height || width % alignment || height % alignment ||
        (uint64_t)(width / alignment) * (height / alignment) > UINT32_MAX)
        return -1;
    *target_width = width;
    *target_height = height;
    *token_count = (width / alignment) * (height / alignment);
    return 0;
}

int coli_gemma4_vision_prepare_rgb(const coli_gemma4_vision *vision,
                                   const uint8_t *rgb,
                                   uint32_t source_width,
                                   uint32_t source_height,
                                   coli_gemma4_vision_image *image) {
    uint32_t width, height, tokens, x, y, channel;
    uint64_t values;
    float x_ratio, y_ratio;
    if (!vision || !rgb || !image || !source_width || !source_height ||
        source_width > SIZE_MAX / 3U / source_height) return -1;
    memset(image, 0, sizeof(*image));
    if (coli_gemma4_vision_target_size(vision, source_width, source_height,
                                      &width, &height, &tokens) != 0)
        return -1;
    values = (uint64_t)width * height * 3;
    if (values > SIZE_MAX / sizeof(float)) return -1;
    image->pixels = (float *)malloc((size_t)values * sizeof(float));
    if (!image->pixels) return -1;
    x_ratio = width > 1 ?
        (float)(source_width - 1) / (float)(width - 1) : 0.0F;
    y_ratio = height > 1 ?
        (float)(source_height - 1) / (float)(height - 1) : 0.0F;
    for (y = 0; y < height; ++y) {
        float py = (float)y * y_ratio;
        uint32_t y0 = (uint32_t)py;
        uint32_t y1 = y0 + 1 < source_height ? y0 + 1 : y0;
        float yf = py - (float)y0;
        for (x = 0; x < width; ++x) {
            float px = (float)x * x_ratio;
            uint32_t x0 = (uint32_t)px;
            uint32_t x1 = x0 + 1 < source_width ? x0 + 1 : x0;
            float xf = px - (float)x0;
            for (channel = 0; channel < 3; ++channel) {
                float p00 = rgb[((size_t)y0 * source_width + x0) * 3 + channel];
                float p10 = rgb[((size_t)y0 * source_width + x1) * 3 + channel];
                float p01 = rgb[((size_t)y1 * source_width + x0) * 3 + channel];
                float p11 = rgb[((size_t)y1 * source_width + x1) * 3 + channel];
                float top = p00 + (p10 - p00) * xf;
                float bottom = p01 + (p11 - p01) * xf;
                uint8_t resized = (uint8_t)(top + (bottom - top) * yf);
                float unit = (float)resized / 255.0F;
                image->pixels[((size_t)y * width + x) * 3 + channel] =
                    (unit - vision->gguf.vision_image_mean[channel]) /
                    vision->gguf.vision_image_std[channel];
            }
        }
    }
    image->width = width;
    image->height = height;
    image->patch_columns = width / vision->gguf.vision_patch_size;
    image->patch_rows = height / vision->gguf.vision_patch_size;
    image->token_columns = image->patch_columns / vision->merge_size;
    image->token_rows = image->patch_rows / vision->merge_size;
    image->value_count = (size_t)values;
    (void)tokens;
    return 0;
}

int coli_gemma4_vision_patch_embeddings(
    const coli_gemma4_vision *vision,
    const coli_gemma4_vision_image *image,
    float **embeddings, uint32_t *patch_count) {
    const coli_gemma4_tensor *patch, *position;
    float *weights = NULL, *pos_x = NULL, *pos_y = NULL, *output = NULL;
    uint32_t patch_size, width, columns, rows;
    int64_t patch_index;
    uint64_t patches, output_values, row_bytes, y_offset;
    size_t patch_values;
    int result = -1;
    int failed = 0;
    int compat_patch_fma8 = vision_compat_has("patchfma8");
    int compat_patch_vecdot = vision_compat_has("patchvecdot");
    if (!vision || !image || !image->pixels || !embeddings || !patch_count ||
        !image->width || !image->height) return -1;
    *embeddings = NULL;
    *patch_count = 0;
    patch_size = vision->gguf.vision_patch_size;
    width = vision->gguf.vision_embedding_length;
    columns = image->patch_columns;
    rows = image->patch_rows;
    if (!patch_size || !width || !columns || !rows ||
        columns != image->width / patch_size ||
        rows != image->height / patch_size) return -1;
    patches = (uint64_t)columns * rows;
    output_values = patches * width;
    patch_values = (size_t)patch_size * patch_size * 3U * width;
    if (patches > UINT32_MAX || output_values > SIZE_MAX / sizeof(float) ||
        patch_values > SIZE_MAX / sizeof(float)) return -1;
    patch = coli_gemma4_gguf_find(&vision->gguf, "v.patch_embd.weight");
    position = coli_gemma4_gguf_find(&vision->gguf, "v.position_embd.weight");
    if (!patch || !position || columns > position->dims[1] ||
        rows > position->dims[1]) return -1;
    weights = (float *)malloc(patch_values * sizeof(float));
    pos_x = (float *)malloc((size_t)columns * width * sizeof(float));
    pos_y = (float *)malloc((size_t)rows * width * sizeof(float));
    output = (float *)malloc((size_t)output_values * sizeof(float));
    if (!weights || !pos_x || !pos_y || !output) goto cleanup;
    row_bytes = (uint64_t)width * sizeof(float);
    y_offset = position->dims[1] * row_bytes;
    if (coli_gemma4_gguf_read(&vision->gguf, patch, weights,
                              patch_values * sizeof(float)) != 0 ||
        coli_gemma4_gguf_read_slice(
            &vision->gguf, position, 0, pos_x,
            (size_t)columns * (size_t)row_bytes) != 0 ||
        coli_gemma4_gguf_read_slice(
            &vision->gguf, position, y_offset, pos_y,
            (size_t)rows * (size_t)row_bytes) != 0)
        goto cleanup;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
    for (patch_index = 0; patch_index < (int64_t)patches; ++patch_index) {
        uint32_t y = (uint32_t)patch_index / columns;
        uint32_t x = (uint32_t)patch_index % columns;
        uint32_t out;
        float *flattened = NULL;
        if (compat_patch_fma8 || compat_patch_vecdot) {
            uint32_t channel, py, px;
            flattened = (float *)malloc(
                (size_t)patch_size * patch_size * 3U * sizeof(*flattened));
            if (!flattened) {
                failed = 1;
                continue;
            }
            for (channel = 0; channel < 3; ++channel)
                for (py = 0; py < patch_size; ++py)
                    for (px = 0; px < patch_size; ++px) {
                        size_t pixel_index =
                            ((size_t)(y * patch_size + py) * image->width +
                             x * patch_size + px) * 3U + channel;
                        size_t flattened_index =
                            ((size_t)channel * patch_size + py) *
                            patch_size + px;
                        flattened[flattened_index] =
                            2.0F * image->pixels[pixel_index] - 1.0F;
                    }
        }
            for (out = 0; out < width; ++out) {
                const float *kernel = weights +
                    (size_t)out * patch_size * patch_size * 3U;
                float sum;
                uint32_t py, px, channel;
                if (flattened) {
                    sum = compat_patch_vecdot ?
                        f32_dot_vecdot(flattened, kernel,
                                      patch_size * patch_size * 3U) :
                        f32_dot_fma8(flattened, kernel,
                                     patch_size * patch_size * 3U);
                    sum += pos_x[(size_t)x * width + out];
                    sum += pos_y[(size_t)y * width + out];
                } else {
                    sum = pos_x[(size_t)x * width + out] +
                          pos_y[(size_t)y * width + out];
                    for (channel = 0; channel < 3; ++channel)
                        for (py = 0; py < patch_size; ++py)
                            for (px = 0; px < patch_size; ++px) {
                                size_t pixel_index =
                                    ((size_t)(y * patch_size + py) * image->width +
                                     x * patch_size + px) * 3U + channel;
                                size_t kernel_index =
                                    ((size_t)channel * patch_size + py) *
                                    patch_size + px;
                                sum += (2.0F * image->pixels[pixel_index] - 1.0F) *
                                       kernel[kernel_index];
                            }
                }
                output[((size_t)y * columns + x) * width + out] = sum;
            }
        free(flattened);
    }
    if (failed) goto cleanup;
    *embeddings = output;
    *patch_count = (uint32_t)patches;
    output = NULL;
    result = 0;
cleanup:
    free(weights);
    free(pos_x);
    free(pos_y);
    free(output);
    return result;
}

int coli_gemma4_vision_transform(coli_gemma4_vision *vision,
                                 float *embeddings,
                                 uint32_t patch_count,
                                 uint32_t patch_columns,
                                 uint32_t layer_count) {
    return coli_gemma4_vision_transform_range(
        vision, embeddings, patch_count, patch_columns, 0, layer_count);
}

int coli_gemma4_vision_transform_range(coli_gemma4_vision *vision,
                                       float *embeddings,
                                       uint32_t patch_count,
                                       uint32_t patch_columns,
                                       uint32_t first_layer,
                                       uint32_t layer_count) {
    uint32_t layer, width, heads, head_dim;
    if (!vision || !embeddings || !patch_count || !patch_columns ||
        patch_count % patch_columns ||
        first_layer > layer_count ||
        layer_count > vision->gguf.vision_block_count)
        return -1;
    width = vision->gguf.vision_embedding_length;
    heads = vision->gguf.vision_head_count;
    if (!width || !heads || width % heads) return -1;
    head_dim = width / heads;
    if (!head_dim || head_dim % 4U) {
        vision_error(vision,
                     "Gemma 4 vision head dimension must be divisible by four");
        return -1;
    }
    for (layer = first_layer; layer < layer_count; ++layer) {
        vision_block block;
        memset(&block, 0, sizeof(block));
        if (vision_block_open(vision, layer, &block) != 0) {
            vision_error(vision, "cannot load Gemma 4 vision block %u", layer);
            return -1;
        }
        if (vision_block_apply(vision, &block, embeddings,
                               patch_count, patch_columns, layer) != 0) {
            vision_block_close(&block);
            vision_error(vision, "cannot evaluate Gemma 4 vision block %u",
                         layer);
            return -1;
        }
        vision_block_close(&block);
    }
    return 0;
}

void coli_gemma4_vision_trace_layer(coli_gemma4_vision *vision,
                                    uint32_t layer,
                                    const char *directory) {
    if (!vision) return;
    vision->trace_layer = layer;
    vision->trace_directory = directory;
}

int coli_gemma4_vision_encode(coli_gemma4_vision *vision,
                              const coli_gemma4_vision_image *image,
                              float **embeddings, uint32_t *token_count) {
    uint32_t patch_count = 0, width, projection_width;
    uint32_t merge, token_columns, token_rows;
    uint64_t output_tokens, output_values;
    float pool_scale;
    float *hidden = NULL, *pooled = NULL, *projected = NULL;
    float *bias = NULL, *scale = NULL;
    uint16_t *projection = NULL;
    uint64_t vector_shape[1], projection_shape[2];
    int64_t token;
    int result = -1;
    if (!vision || !image || !embeddings || !token_count) return -1;
    *embeddings = NULL;
    *token_count = 0;
    width = vision->gguf.vision_embedding_length;
    projection_width = vision->gguf.vision_projection_dim;
    merge = vision->merge_size;
    if (!width || !projection_width || !merge ||
        !image->patch_columns || !image->patch_rows ||
        image->patch_columns % merge || image->patch_rows % merge)
        return -1;
    token_columns = image->patch_columns / merge;
    token_rows = image->patch_rows / merge;
    output_tokens = (uint64_t)token_columns * token_rows;
    output_values = output_tokens * projection_width;
    if (!output_tokens || output_tokens > UINT32_MAX ||
        output_values > SIZE_MAX / sizeof(float) ||
        output_tokens > SIZE_MAX / width ||
        output_tokens * width > SIZE_MAX / sizeof(float))
        return -1;
    if (coli_gemma4_vision_patch_embeddings(
            vision, image, &hidden, &patch_count) != 0) {
        vision_error(vision, "cannot create Gemma 4 vision patch embeddings");
        goto cleanup;
    }
    if (coli_gemma4_vision_transform(
            vision, hidden, patch_count, image->patch_columns,
            vision->gguf.vision_block_count) != 0)
        goto cleanup;

    pooled = (float *)malloc((size_t)output_tokens * width * sizeof(float));
    projected = (float *)malloc((size_t)output_values * sizeof(float));
    if (!pooled || !projected) {
        vision_error(vision, "out of memory completing Gemma 4 vision graph");
        goto cleanup;
    }
    pool_scale = sqrtf((float)width) / (float)(merge * merge);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (token = 0; token < (int64_t)output_tokens; ++token) {
        uint32_t output_y = (uint32_t)token / token_columns;
        uint32_t output_x = (uint32_t)token % token_columns;
        uint32_t dimension;
        for (dimension = 0; dimension < width; ++dimension) {
            float sum = 0.0F;
            uint32_t y, x;
            for (y = 0; y < merge; ++y)
                for (x = 0; x < merge; ++x) {
                    uint32_t patch_y = output_y * merge + y;
                    uint32_t patch_x = output_x * merge + x;
                    sum += hidden[((size_t)patch_y * image->patch_columns +
                                   patch_x) * width + dimension];
                }
            pooled[(size_t)token * width + dimension] =
                sum * pool_scale;
        }
    }
    vector_shape[0] = width;
    projection_shape[0] = width;
    projection_shape[1] = projection_width;
    if (read_named(vision, "v.std_bias", COLI_GGML_TYPE_F32,
                   1, vector_shape, (void **)&bias) != 0 ||
        read_named(vision, "v.std_scale", COLI_GGML_TYPE_F32,
                   1, vector_shape, (void **)&scale) != 0 ||
        read_named(vision, "mm.input_projection.weight", COLI_GGML_TYPE_BF16,
                   2, projection_shape, (void **)&projection) != 0) {
        vision_error(vision, "cannot load Gemma 4 vision projection tensors");
        goto cleanup;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (token = 0; token < (int64_t)output_tokens; ++token) {
        uint32_t dimension;
        float *row = pooled + (size_t)token * width;
        for (dimension = 0; dimension < width; ++dimension)
            row[dimension] = (row[dimension] - bias[dimension]) *
                             scale[dimension];
    }
    vision_rmsnorm(pooled, pooled, NULL, (uint32_t)output_tokens,
                   width, vision->gguf.vision_epsilon);
    bf16_matmul(projected, pooled, projection,
                (uint32_t)output_tokens, width, projection_width);
    *embeddings = projected;
    *token_count = (uint32_t)output_tokens;
    projected = NULL;
    result = 0;
cleanup:
    free(hidden); free(pooled); free(projected);
    free(bias); free(scale); free(projection);
    return result;
}

void coli_gemma4_vision_image_close(coli_gemma4_vision_image *image) {
    if (!image) return;
    free(image->pixels);
    memset(image, 0, sizeof(*image));
}
