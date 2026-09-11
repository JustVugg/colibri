#define _CRT_SECURE_NO_WARNINGS

#include "gemma4_vision.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(int condition, const char *message) {
    if (!condition) fprintf(stderr, "FAIL: %s\n", message);
    return condition ? 0 : 1;
}

int main(void) {
    coli_gemma4_vision vision;
    coli_gemma4_vision_image image;
    uint32_t width, height, tokens;
    float transform_probe[32] = {0};
    const uint8_t rgb[6] = {255, 0, 0, 0, 0, 255};
    int failures = 0;
    const char *ppm_path = "gemma4-test-image.ppm";
    uint8_t *loaded_rgb = NULL;
    char ppm_error[256];
    FILE *ppm;
    memset(&vision, 0, sizeof(vision));
    vision.gguf.vision_patch_size = 16;
    vision.gguf.vision_image_mean[0] = 0.0F;
    vision.gguf.vision_image_mean[1] = 0.0F;
    vision.gguf.vision_image_mean[2] = 0.0F;
    vision.gguf.vision_image_std[0] = 1.0F;
    vision.gguf.vision_image_std[1] = 1.0F;
    vision.gguf.vision_image_std[2] = 1.0F;
    vision.merge_size = 3;
    vision.minimum_tokens = 40;
    vision.maximum_tokens = 280;
    failures += check(coli_gemma4_vision_target_size(
        &vision, 640, 480, &width, &height, &tokens) == 0,
        "ordinary target size failed");
    failures += check(width == 624 && height == 480 && tokens == 130,
                      "ordinary target size differs from smart resize");
    failures += check(coli_gemma4_vision_target_size(
        &vision, 4000, 2000, &width, &height, &tokens) == 0,
        "maximum target size failed");
    failures += check(width == 1104 && height == 528 && tokens == 253,
                      "maximum target size differs from smart resize");
    vision.minimum_tokens = 1;
    vision.maximum_tokens = 1;
    memset(&image, 0, sizeof(image));
    failures += check(coli_gemma4_vision_prepare_rgb(
        &vision, rgb, 2, 1, &image) == 0, "bilinear RGB preparation failed");
    failures += check(image.width == 48 && image.height == 48 &&
                      image.patch_columns == 3 && image.patch_rows == 3 &&
                      image.token_columns == 1 && image.token_rows == 1,
                      "prepared image geometry is wrong");
    failures += check(image.value_count == 48U * 48U * 3U,
                      "prepared image value count is wrong");
    if (image.pixels) {
        size_t last = image.value_count - 3;
        failures += check(fabsf(image.pixels[0] - 1.0F) < 1e-7F &&
                          image.pixels[1] == 0.0F && image.pixels[2] == 0.0F,
                          "left endpoint pixel is wrong");
        failures += check(image.pixels[last] == 0.0F &&
                          image.pixels[last + 1] == 0.0F &&
                          fabsf(image.pixels[last + 2] -
                                (254.0F / 255.0F)) < 1e-7F,
                          "right endpoint pixel is wrong");
    }
    vision.gguf.vision_embedding_length = 8;
    vision.gguf.vision_head_count = 2;
    vision.gguf.vision_block_count = 0;
    failures += check(coli_gemma4_vision_transform(
        &vision, transform_probe, 4, 2, 0) == 0,
        "zero-block transformer boundary failed");
    failures += check(coli_gemma4_vision_transform(
        &vision, transform_probe, 4, 3, 0) != 0,
        "invalid transformer geometry was accepted");
    ppm = fopen(ppm_path, "wb");
    failures += check(ppm != NULL, "could not create PPM fixture");
    if (ppm) {
        const char header[] = "P6\n# fixture\n2 1\n255\n";
        failures += check(fwrite(header, 1, sizeof(header) - 1, ppm) ==
                          sizeof(header) - 1 &&
                          fwrite(rgb, 1, sizeof(rgb), ppm) == sizeof(rgb) &&
                          fclose(ppm) == 0, "could not write PPM fixture");
        failures += check(coli_gemma4_vision_load_image(
            ppm_path, &loaded_rgb, &width, &height,
            ppm_error, sizeof(ppm_error)) == 0, "PPM loading failed");
        failures += check(width == 2 && height == 1 && loaded_rgb &&
                          memcmp(loaded_rgb, rgb, sizeof(rgb)) == 0,
                          "PPM pixels differ from fixture");
    }
    remove(ppm_path);
    free(loaded_rgb);
    coli_gemma4_vision_image_close(&image);
    if (failures) return 1;
    puts("Gemma 4 vision sizing and RGB preprocessing passed");
    return 0;
}
