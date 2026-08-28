/* Il tower vision contro l'oracolo generato da transformers
 * (tools/make_glm53_vision_tiny.py). Legge la fixture con st.h, cosi' il test
 * esercita anche il percorso di caricamento reale invece di dati incollati. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../json.h"
#include "../st.h"
#include "../vision_tower.h"

static const float *need(shards *S, const char *name, int64_t expect) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "manca il tensore %s\n", name); exit(1); }
    if (expect > 0 && t->numel != expect) {
        fprintf(stderr, "%s: %lld elementi, attesi %lld\n", name,
                (long long)t->numel, (long long)expect); exit(1);
    }
    float *buffer = malloc((size_t)t->numel * sizeof(float));
    if (!buffer) { fprintf(stderr, "OOM su %s\n", name); exit(1); }
    st_read_f32_cap(S, name, buffer, t->numel, 0);
    return buffer;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "non riesco ad aprire %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "%s: lettura corta\n", path); exit(1); }
    b[n] = 0; fclose(f);
    return b;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "glm53_vision_tiny";
    char path[1024];
    shards S;
    st_init(&S, dir);

    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *text = slurp(path), *arena = NULL;
    jval *root = json_parse(text, &arena);
    jval *vc = json_get(root, "vision_config");
    if (!vc) { fprintf(stderr, "config.json senza vision_config\n"); return 1; }
#define NUM(key) ((int)json_get(vc, key)->num)
    ColiVisionConfig cfg = {
        .depth = NUM("depth"), .hidden = NUM("hidden_size"), .heads = NUM("num_heads"),
        .intermediate = NUM("intermediate_size"), .patch = NUM("patch_size"),
        .temporal = NUM("temporal_patch_size"), .merge = NUM("spatial_merge_size"),
        .in_channels = NUM("in_channels"), .out_hidden = NUM("out_hidden_size"),
        .proj_intermediate = NUM("projection_intermediate_size"),
        .eps = (float)json_get(vc, "rms_norm_eps")->num,
        .swiglu_limit = (float)json_get(vc, "swiglu_limit")->num,
        .rope_theta = 10000.0f,
    };
#undef NUM
    cfg.head_dim = cfg.hidden / cfg.heads;

    ColiVisionBlock *blocks = calloc((size_t)cfg.depth, sizeof(*blocks));
    for (int b = 0; b < cfg.depth; b++) {
        char n[256];
#define GET(field, suffix, expect) \
        snprintf(n, sizeof(n), "blocks.%d." suffix, b); blocks[b].field = need(&S, n, expect)
        GET(norm1, "norm1.weight", cfg.hidden);
        GET(norm2, "norm2.weight", cfg.hidden);
        GET(qkv_w, "attn.qkv.weight", (int64_t)3 * cfg.hidden * cfg.hidden);
        GET(qkv_b, "attn.qkv.bias", 3 * cfg.hidden);
        GET(q_norm, "attn.q_norm.weight", cfg.head_dim);
        GET(k_norm, "attn.k_norm.weight", cfg.head_dim);
        GET(proj_w, "attn.proj.weight", (int64_t)cfg.hidden * cfg.hidden);
        GET(proj_b, "attn.proj.bias", cfg.hidden);
        GET(gate_w, "mlp.gate_proj.weight", (int64_t)cfg.intermediate * cfg.hidden);
        GET(gate_b, "mlp.gate_proj.bias", cfg.intermediate);
        GET(up_w, "mlp.up_proj.weight", (int64_t)cfg.intermediate * cfg.hidden);
        GET(up_b, "mlp.up_proj.bias", cfg.intermediate);
        GET(down_w, "mlp.down_proj.weight", (int64_t)cfg.hidden * cfg.intermediate);
        GET(down_b, "mlp.down_proj.bias", cfg.hidden);
#undef GET
    }
    ColiVisionTower tower = {
        .config = cfg, .blocks = blocks,
        .patch_w = need(&S, "patch_embed.proj.weight", 0),
        .patch_b = need(&S, "patch_embed.proj.bias", cfg.hidden),
        .post_norm = need(&S, "post_layernorm.weight", cfg.hidden),
        .down_w = need(&S, "downsample.weight", 0),
        .down_b = need(&S, "downsample.bias", cfg.out_hidden),
        .merger_proj = need(&S, "merger.proj.weight", (int64_t)cfg.out_hidden * cfg.out_hidden),
        .merger_norm_w = need(&S, "merger.post_projection_norm.weight", cfg.out_hidden),
        .merger_norm_b = need(&S, "merger.post_projection_norm.bias", cfg.out_hidden),
        .merger_gate = need(&S, "merger.gate_proj.weight", 0),
        .merger_up = need(&S, "merger.up_proj.weight", 0),
        .merger_down = need(&S, "merger.down_proj.weight", 0),
    };

    snprintf(path, sizeof(path), "%s/ref.json", dir);
    char *rtext = slurp(path), *rarena = NULL;
    jval *ref = json_parse(rtext, &rarena);
    jval *grid = json_get(ref, "grid_thw");
    int gh = (int)grid->kids[0]->kids[1]->num, gw = (int)grid->kids[0]->kids[2]->num;
    jval *want = json_get(ref, "output");

    const float *pixels = need(&S, "input.pixel_values", 0);
    int out_tokens = coli_vision_output_tokens(&cfg, gh, gw);
    float *out = malloc((size_t)out_tokens * cfg.out_hidden * sizeof(float));
    if (coli_vision_forward(out, &tower, pixels, gh, gw)) {
        fprintf(stderr, "coli_vision_forward ha fallito\n"); return 1;
    }

    int n = out_tokens * cfg.out_hidden;
    if (want->len != n) {
        fprintf(stderr, "ref ha %d valori, il tower ne produce %d\n", want->len, n);
        return 1;
    }
    float worst = 0.0f; int at = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(out[i] - (float)want->kids[i]->num);
        if (d > worst) { worst = d; at = i; }
    }
    printf("vision tower: %d token x %d, max|delta| = %.3e (indice %d: %.9g vs %.9g)\n",
           out_tokens, cfg.out_hidden, worst, at, out[at], want->kids[at]->num);
    if (worst > 2e-5f) { printf("FALLITO\n"); return 1; }
    printf("PASS tower vision: combacia con l'oracolo transformers\n");
    return 0;
}
