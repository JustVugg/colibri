/* Compare the C image+prompt encoder with the pinned Diffusers tiny oracle. */
#define main qwenimage_cli_main
#include "../qwenimage.c"
#undef main

static int compare(shards *reference, const char *key, const float *value, size_t n, double tolerance)
{
    int64_t count;
    float *want = ref_tensor(reference, key, &count);
    int pass = want && count == (int64_t)n && cmp(key, value, want, n) <= tolerance;
    if (!pass) fprintf(stderr, "[vision oracle] %s failed\n", key);
    free(want);
    return !pass;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR (make_qwenimage_tiny.py --vision-conditioning)\n", argv[0]);
        return 2;
    }
    char dir[2048], path[2080], *arena = NULL;
    snprintf(dir, sizeof dir, "%s/ref-vision", argv[1]);
    jval *meta = read_json(dir, "ref.json", &arena);
    jval *prompt = json_get(meta, "prompt");
    int width = jnum(meta, "width", 0), height = jnum(meta, "height", 0);
    if (!prompt || prompt->t != J_STR || width != 96 || height != 64) {
        free(arena);
        return 2;
    }
    snprintf(path, sizeof path, "%s/image.rgba", dir);
    FILE *image = fopen(path, "rb");
    uint8_t rgba[96 * 64 * 4];
    int complete = image && fread(rgba, 1, sizeof rgba, image) == sizeof rgba && fgetc(image) == EOF;
    if (image) fclose(image);
    if (!complete) {
        free(arena);
        return 2;
    }

    g_bits = 16;  /* Exact BF16 checkpoint weights, with FP32 arithmetic. */
    qi_parts_init(argv[1]);
    Te te = {0};
    te_config(&te, argv[1]);
    if (!te_image_supported(&te, argv[1]) || te.loaded) return 1;
    te_tokenizer(&te, argv[1]);
    shards reference = {0};
    st_init(&reference, dir);
    int fails = 0;
    int gh = height / QI_PATCH, gw = width / QI_PATCH;
    int slots = gh / QI_MERGE * (gw / QI_MERGE);
    int tokens, drop, start;
    int *ids = qi_image_tokens(&te, prompt->str, slots, &tokens, &drop, &start);
    if (!ids) return 1;
    float *ids_float = fmalloc(tokens);
    for (int i = 0; i < tokens; i++) ids_float[i] = (float)ids[i];
    fails += compare(&reference, "input_ids", ids_float, tokens, 0);
    fails += drop != jnum(meta, "drop", -1) || start - drop != jnum(meta, "image_slot_start", -1);
    free(ids_float);
    free(ids);

    float *pixels = qi_image_pixels(rgba, width, gh, gw);
    fails += compare(&reference, "pixel_values", pixels, (size_t)gh * gw * QI_PATCH_FEATURES, 5e-7);
    size_t feature_size = (size_t)slots * te.hidden;
    float *features = fmalloc(4 * feature_size);
    if (qi_vision_encode(&te, argv[1], pixels, gh, gw, features) == slots) {
        fails += compare(&reference, "merged", features, feature_size, 2e-5);
        for (int i = 0; i < 3; i++) {
            char tier[16];
            snprintf(tier, sizeof tier, "deepstack_%d", i);
            fails += compare(&reference, tier, features + (size_t)(i + 1) * feature_size, feature_size, 2e-5);
        }
    } else fails++;
    free(features);
    free(pixels);

    size_t rope_size = (size_t)tokens * (te.hd / 2);
    float *cosines = fmalloc(2 * rope_size), *sines = cosines + rope_size;
    qi_image_rope(&te, tokens, start, gh / QI_MERGE, gw / QI_MERGE, cosines, sines);
    fails += compare(&reference, "cosines", cosines, rope_size, 2e-6);
    fails += compare(&reference, "sines", sines, rope_size, 2e-6);
    free(cosines);

    QiImagePrompt result;
    if (te_encode_image_prompt(&te, argv[1], prompt->str, rgba, width, height, &result) == 0) {
        fails += result.tokens != tokens - drop || result.image_slot_start != start - drop ||
                 result.image_slot_count != jnum(meta, "image_slot_count", -1) ||
                 result.grid_h != gh || result.grid_w != gw;
        fails += compare(&reference, "hidden", result.hidden, (size_t)result.tokens * te.hidden, 2e-5);
        free(result.hidden);
    } else fails++;
    fails += te_encode_image_prompt(&te, argv[1], prompt->str, rgba, width - 1, height, &result) != -1 ||
             result.hidden != NULL;
    fails += te_encode_image_prompt(&te, argv[1], "<|image_pad|>", rgba, width, height, &result) != -1 ||
             result.hidden != NULL;

    st_destroy(&reference);
    te_unload(&te);
    st_destroy(&te.S);
    tok_free(&te.tok);
    free(arena);
    fprintf(stderr, "[vision oracle] %d failures\n", fails);
    return fails ? 1 : 0;
}
