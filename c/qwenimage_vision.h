#ifndef COLI_QWENIMAGE_VISION_H
#define COLI_QWENIMAGE_VISION_H

#include "qwen38_vision.h"

/* One still reference. The caller supplies the RGBA8 image resized to the same
 * 32-aligned dimensions as the VAE input. The vision tower reads the reference
 * composited over white; only the VAE sees alpha. */
typedef struct {
    float *hidden;  /* owned, [tokens][te->hidden], before the final RMS norm */
    int tokens, image_slot_start, image_slot_count;  /* after system-prefix drop */
    int grid_h, grid_w;  /* vision patch grid before 2x2 merge; temporal size = 1 */
} QiImagePrompt;

enum { QI_PATCH = 16, QI_MERGE = 2, QI_PATCH_FEATURES = 3 * 2 * 16 * 16 };

static unsigned qi_paste_channel(unsigned rgb, unsigned alpha)
{
    /* Pillow's rounded 8-bit paste over white. */
    return 255 - alpha + (rgb * alpha + 127) / 255;
}

static float *qi_image_pixels(const uint8_t *rgba, int width, int grid_h, int grid_w)
{
    int patches = grid_h * grid_w;
    float *pixels = fmalloc((size_t)patches * QI_PATCH_FEATURES);
    for (int i = 0; i < patches; i++) {
        int y, x;
        q38v_row_col(i, grid_w, QI_MERGE, &y, &x);
        float *row = pixels + (size_t)i * QI_PATCH_FEATURES;
        for (int c = 0; c < 3; c++)
            for (int py = 0; py < QI_PATCH; py++)
                for (int px = 0; px < QI_PATCH; px++) {
                    const uint8_t *p = rgba + 4 * ((size_t)(y * QI_PATCH + py) * width + x * QI_PATCH + px);
                    float v = (float)qi_paste_channel(p[c], p[3]) / 255.f * 2.f - 1.f;
                    int at = (c * 2 * QI_PATCH + py) * QI_PATCH + px;
                    row[at] = row[at + QI_PATCH * QI_PATCH] = v;
                }
    }
    return pixels;
}

static int *qi_image_tokens(Te *te, const char *prompt, int slots, int *tokens,
                            int *drop, int *image_start)
{
    int n;
    int *ids = te_encode_prompt_impl(te, prompt, 1, &n, drop);
    int pad = tok_id_of(&te->tok, "<|image_pad|>");
    int start = tok_id_of(&te->tok, "<|vision_start|>");
    int end = tok_id_of(&te->tok, "<|vision_end|>");
    int video = tok_id_of(&te->tok, "<|video_pad|>");
    int at = -1, markers = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] == pad) at = i;
        if (ids[i] == pad || ids[i] == start || ids[i] == end || ids[i] == video)
            markers++;
    }
    if (pad < 0 || start < 0 || end < 0 || markers != 3 ||
        at < 1 || at < *drop || at >= n - 1 || ids[at - 1] != start ||
        ids[at + 1] != end || n - *drop - 1 > QI_MAX_PROMPT_TOKENS ||
        n > 16384 - slots) {
        free(ids);
        return NULL;
    }
    *tokens = n + slots - 1;
    *image_start = at;
    int *expanded = xmalloc((size_t)*tokens * sizeof(int));
    memcpy(expanded, ids, (size_t)at * sizeof(int));
    for (int i = 0; i < slots; i++) expanded[at + i] = pad;
    memcpy(expanded + at + slots, ids + at + 1, (size_t)(n - at - 1) * sizeof(int));
    free(ids);
    return expanded;
}

/* Within the image run, the position advances on the height and width axes.
 * After it, text resumes one beyond the highest image coordinate. */
static void qi_image_rope(const Te *te, int tokens, int start, int gh, int gw,
                          float *cosines, float *sines)
{
    const int half = te->hd / 2, slots = gh * gw, span = gh > gw ? gh : gw;
    for (int i = 0; i < half; i++) {
        int axis = i % 3;
        if (i >= te->mrope[axis] * 3) axis = 0;
        float inv = 1.f / powf(te->theta, (float)(2 * i) / (float)te->hd);
        for (int t = 0; t < tokens; t++) {
            int pos = t;
            if (t >= start + slots) pos += span - slots;
            else if (t >= start) {
                int slot = t - start;
                pos = start + (axis == 1 ? slot / gw : axis == 2 ? slot % gw : 0);
            }
            float angle = (float)pos * inv;
            cosines[(size_t)t * half + i] = cosf(angle);
            sines[(size_t)t * half + i] = sinf(angle);
        }
    }
}

typedef struct {
    Q38Vision tower;
    Q38VisionTap taps[3];
    Q38VBlock *blocks;
    Q38VMerger final;
} QiVision;

/* With no destination, validate the metadata without reading tensor bytes. */
static int qi_vision_weight(shards *store, const char *prefix, const char *suffix,
                            int rows, int width, const float **destination)
{
    char key[160];
    snprintf(key, sizeof key, "%s.%s", prefix, suffix);
    st_tensor *t = st_find(store, key);
    int64_t cells = (int64_t)rows * width;
    int type = t ? st_ggml_type_any(t->dtype) : -1;
    if (!t || (type != GQ_F32 && type != GQ_F16 && type != GQ_BF16) ||
        t->rank < 1 || t->shape[0] != rows ||
        t->nbytes != cells * (type == GQ_F32 ? 4 : 2)) {
        if (destination) fprintf(stderr, "[qwenimage] vision tensor %s missing or unsupported\n", key);
        return 0;
    }
    if (destination) {
        float *out = fmalloc((size_t)cells);
        st_read_f32_cap(store, key, out, cells, 1);
        *destination = out;
    }
    return 1;
}

static int qi_vision_norm(shards *store, const char *prefix, int width, Q38Norm *norm,
                          int load)
{
    return qi_vision_weight(store, prefix, "weight", width, 1, load ? &norm->w : NULL) &&
           qi_vision_weight(store, prefix, "bias", width, 1, load ? &norm->b : NULL);
}

static int qi_vision_linear(shards *store, const char *prefix, int in, int out, Q38Linear *linear,
                            int load)
{
    linear->in = in;
    linear->out = out;
    return qi_vision_weight(store, prefix, "weight", out, in, load ? &linear->w : NULL) &&
           qi_vision_weight(store, prefix, "bias", out, 1, load ? &linear->b : NULL);
}

static void qi_vision_free_norm(const Q38Norm *norm)
{
    free((void *)norm->w);
    free((void *)norm->b);
}

static void qi_vision_free_linear(const Q38Linear *linear)
{
    free((void *)linear->w);
    free((void *)linear->b);
}

static int qi_vision_merger(shards *store, const char *prefix, int hidden, int output,
                            int postshuffle, Q38VMerger *merger, int load)
{
    int wide = hidden * QI_MERGE * QI_MERGE;
    char key[128];
    snprintf(key, sizeof key, "%s.norm", prefix);
    if (!qi_vision_norm(store, key, postshuffle ? wide : hidden, &merger->norm, load)) return 0;
    snprintf(key, sizeof key, "%s.linear_fc1", prefix);
    if (!qi_vision_linear(store, key, wide, wide, &merger->fc1, load)) return 0;
    snprintf(key, sizeof key, "%s.linear_fc2", prefix);
    return qi_vision_linear(store, key, wide, output, &merger->fc2, load);
}

static void qi_vision_free_merger(const Q38VMerger *merger)
{
    qi_vision_free_norm(&merger->norm);
    qi_vision_free_linear(&merger->fc1);
    qi_vision_free_linear(&merger->fc2);
}

static void qi_vision_free(const QiVision *vision)
{
    const Q38Vision *tower = &vision->tower;
    for (int i = 0; vision->blocks && i < tower->depth; i++) {
        const Q38VBlock *b = &vision->blocks[i];
        qi_vision_free_norm(&b->norm1);
        qi_vision_free_norm(&b->norm2);
        qi_vision_free_linear(&b->qkv);
        qi_vision_free_linear(&b->proj);
        qi_vision_free_linear(&b->fc1);
        qi_vision_free_linear(&b->fc2);
    }
    free(vision->blocks);
    qi_vision_free_linear(&tower->patch_embed);
    free((void *)tower->pos_embed);
    qi_vision_free_merger(&vision->final);
    for (int i = 0; i < 3; i++) qi_vision_free_merger(&vision->taps[i].merger);
}

static int qi_vision_dim(jval *config, const char *key, int maximum)
{
    double value = jnum(config, key, 0);
    return value >= 1 && value <= maximum && value == floor(value) ? (int)value : 0;
}

static int qi_vision_config(Te *te, const char *model, QiVision *vision)
{
    char dir[2048], *arena = NULL;
    qi_part(model, "text_encoder", dir, sizeof dir);
    jval *root = read_json(dir, "config.json", &arena);
    jval *v = json_get(root, "vision_config");
    jval *indexes = v ? json_get(v, "deepstack_visual_indexes") : NULL;
    jval *act = v ? json_get(v, "hidden_act") : NULL;
    Q38Vision *tower = &vision->tower;
    tower->depth = qi_vision_dim(v, "depth", 64);
    tower->hidden = qi_vision_dim(v, "hidden_size", 4096);
    tower->heads = qi_vision_dim(v, "num_heads", tower->hidden);
    tower->inter = qi_vision_dim(v, "intermediate_size", 16384);
    tower->out_hidden = qi_vision_dim(v, "out_hidden_size", te->hidden);
    tower->num_pos = qi_vision_dim(v, "num_position_embeddings", 16384);
    int ok = tower->depth > 0 && tower->hidden > 0 &&
             tower->heads > 0 && tower->hidden % tower->heads == 0 &&
             (tower->hidden / tower->heads) % 4 == 0 &&
             tower->inter > 0 && tower->out_hidden == te->hidden && tower->num_pos > 0 &&
             jnum(v, "patch_size", 0) == QI_PATCH &&
             jnum(v, "spatial_merge_size", 0) == QI_MERGE &&
             jnum(v, "temporal_patch_size", 0) == 2 && jnum(v, "in_channels", 0) == 3 &&
             act && act->t == J_STR && !strcmp(act->str, "gelu_pytorch_tanh") &&
             indexes && indexes->t == J_ARR && indexes->len == 3;
    int layer = -1;
    for (int i = 0; ok && i < 3; i++) {
        jval *entry = indexes->kids[i];
        if (entry->t != J_NUM || !(entry->num > layer && entry->num < tower->depth) ||
            entry->num != floor(entry->num))
            ok = 0;
        else
            vision->taps[i].layer = layer = (int)entry->num;
    }
    json_free(root);
    free(arena);
    if (!ok) return 0;
    tower->side = (int)sqrt(tower->num_pos);
    if (tower->side * tower->side != tower->num_pos) return 0;
    tower->head_dim = tower->hidden / tower->heads;
    tower->merge = QI_MERGE;
    tower->eps = 1e-6f;
    return 1;
}

static int qi_vision_load(Te *te, const char *model, QiVision *vision, int load)
{
    if (!qi_vision_config(te, model, vision)) return 0;
    Q38Vision *tower = &vision->tower;
    int hidden = tower->hidden, inter = tower->inter, output = tower->out_hidden;
    shards *store = &te->S;
    const char *patch_key = "model.visual.patch_embed.proj";
    st_tensor *patch = st_find(store, "model.visual.patch_embed.proj.weight");
    if (!patch || patch->rank != 5 || patch->shape[1] != 3 || patch->shape[2] != 2 ||
        patch->shape[3] != QI_PATCH || patch->shape[4] != QI_PATCH)
        return 0;
    if (!qi_vision_linear(store, patch_key, QI_PATCH_FEATURES, hidden, &tower->patch_embed, load) ||
        !qi_vision_weight(store, "model.visual.pos_embed", "weight", tower->num_pos,
                          hidden, load ? &tower->pos_embed : NULL))
        return 0;
    vision->blocks = calloc((size_t)tower->depth, sizeof(Q38VBlock));
    if (!vision->blocks) return 0;
    tower->blocks = vision->blocks;
    for (int i = 0; i < tower->depth; i++) {
        Q38VBlock *b = &vision->blocks[i];
        char key[128];
        snprintf(key, sizeof key, "model.visual.blocks.%d.norm1", i);
        if (!qi_vision_norm(store, key, hidden, &b->norm1, load)) return 0;
        snprintf(key, sizeof key, "model.visual.blocks.%d.norm2", i);
        if (!qi_vision_norm(store, key, hidden, &b->norm2, load)) return 0;
        snprintf(key, sizeof key, "model.visual.blocks.%d.attn.qkv", i);
        if (!qi_vision_linear(store, key, hidden, 3 * hidden, &b->qkv, load)) return 0;
        snprintf(key, sizeof key, "model.visual.blocks.%d.attn.proj", i);
        if (!qi_vision_linear(store, key, hidden, hidden, &b->proj, load)) return 0;
        snprintf(key, sizeof key, "model.visual.blocks.%d.mlp.linear_fc1", i);
        if (!qi_vision_linear(store, key, hidden, inter, &b->fc1, load)) return 0;
        snprintf(key, sizeof key, "model.visual.blocks.%d.mlp.linear_fc2", i);
        if (!qi_vision_linear(store, key, inter, hidden, &b->fc2, load)) return 0;
    }
    if (!qi_vision_merger(store, "model.visual.merger", hidden, output, 0, &vision->final, load))
        return 0;
    tower->merger_norm = vision->final.norm;
    tower->merger_fc1 = vision->final.fc1;
    tower->merger_fc2 = vision->final.fc2;
    for (int i = 0; i < 3; i++) {
        char key[128];
        snprintf(key, sizeof key, "model.visual.deepstack_merger_list.%d", i);
        if (!qi_vision_merger(store, key, hidden, output, 1, &vision->taps[i].merger, load))
            return 0;
    }
    return 1;
}

static int qi_vision_encode(Te *te, const char *model, const float *pixels,
                            int gh, int gw, float *features)
{
    QiVision vision = {0};
    int result = -1;
    int slots = gh / QI_MERGE * (gw / QI_MERGE);
    te_open(te, model);
    if (qi_vision_load(te, model, &vision, 1)) {
        for (int i = 0; i < 3; i++) vision.taps[i].out = features + (size_t)(i + 1) * slots * te->hidden;
        result = q38_vision_forward_tapped(&vision.tower, pixels, gh, gw, features,
                                           NULL, vision.taps, 3);
    }
    qi_vision_free(&vision);
    return result;
}

static int qi_image_text_supported(const Te *te)
{
    return te->layers >= 3 && te->mrope[0] + te->mrope[1] + te->mrope[2] == te->hd / 2;
}

static int te_image_supported(Te *te, const char *model)
{
    if (!qi_image_text_supported(te)) return 0;
    QiVision vision = {0};
    te_open(te, model);
    int supported = qi_vision_load(te, model, &vision, 0);
    qi_vision_free(&vision);
    return supported;
}

/* Leaves the shared text encoder available to the caller's existing lifecycle.
 * No model weights are retained by the vision stage itself. */
static int te_encode_image_prompt(Te *te, const char *model, const char *prompt,
                                  const uint8_t *rgba, int width, int height,
                                  QiImagePrompt *out)
{
    *out = (QiImagePrompt){0};
    if (!rgba || width <= 0 || height <= 0 || width % 32 || height % 32 ||
        (int64_t)width * height > 4096 * 4096 || !qi_image_text_supported(te))
        return -1;
    int gh = height / QI_PATCH, gw = width / QI_PATCH;
    int slots = gh / QI_MERGE * (gw / QI_MERGE);
    int tokens, drop, start;
    te_tokenizer(te, model);
    int *ids = qi_image_tokens(te, prompt, slots, &tokens, &drop, &start);
    if (!ids) return -1;

    size_t feature_size = (size_t)slots * te->hidden;
    float *features = fmalloc(4 * feature_size);
    float *pixels = qi_image_pixels(rgba, width, gh, gw);
    int encoded = qi_vision_encode(te, model, pixels, gh, gw, features);
    free(pixels);
    if (encoded != slots) {
        free(features);
        free(ids);
        return -1;
    }

    size_t rope_size = (size_t)tokens * (te->hd / 2);
    float *cosines = fmalloc(2 * rope_size), *sines = cosines + rope_size;
    qi_image_rope(te, tokens, start, gh / QI_MERGE, gw / QI_MERGE, cosines, sines);
    TeVisual visual = {
        .start = start, .count = slots, .merged = features,
        .deepstack = {features + feature_size, features + 2 * feature_size, features + 3 * feature_size},
        .cosines = cosines, .sines = sines,
    };
    te_load(te, model);
    float *hidden = te_forward_impl(te, ids, tokens, &visual);
    int kept = tokens - drop;
    memmove(hidden, hidden + (size_t)drop * te->hidden, (size_t)kept * te->hidden * sizeof(float));
    *out = (QiImagePrompt){
        .hidden = hidden, .tokens = kept, .image_slot_start = start - drop,
        .image_slot_count = slots, .grid_h = gh, .grid_w = gw,
    };
    free(cosines);
    free(features);
    free(ids);
    return 0;
}

#endif
