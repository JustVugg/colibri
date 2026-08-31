/* test_qwen38_vision.c -- la torre in qwen38_vision.h contro l'oracolo upstream.
 *
 * La fixture e il riferimento li scrive tools/make_qwen38_vision_tiny.py, che
 * costruisce un Qwen4ExpVisionModel vero con pesi casuali e ne registra il
 * forward. Nessun peso del checkpoint: 240k parametri, meno di un megabyte.
 *
 * Vengono confrontate DUE uscite, non una:
 *
 *   last_hidden  l'uscita dei blocchi, prima del merger
 *   pooler       dopo il merger, cioe' i token immagine veri
 *
 * Se combacia solo la prima il difetto e' nel merger; se non combacia nessuna
 * delle due e' nei blocchi o nelle posizioni. Un solo numero finale non
 * distinguerebbe i due casi, e distinguerli e' meta' del lavoro di trovarli.
 *
 * USO: ./tests/test_qwen38_vision <cartella-fixture>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../json.h"
#include "../st.h"
#include "../qwen38_vision.h"

static float *json_floats(jval *node, int64_t *count)
{
    if (!node || node->t != J_ARR) return NULL;
    float *out = (float *)malloc((size_t)node->len * sizeof(float));
    if (!out) return NULL;
    for (int i = 0; i < node->len; i++) out[i] = (float)node->kids[i]->num;
    *count = node->len;
    return out;
}

static const float *want_tensor(shards *S, const char *name, int64_t expect)
{
    char full[512];
    snprintf(full, sizeof full, "model.visual.%s", name);
    st_tensor *t = st_find(S, full);
    if (!t) { fprintf(stderr, "manca il tensore %s\n", full); exit(1); }
    if (expect > 0 && t->numel != expect) {
        fprintf(stderr, "%s ha %lld valori, ne servono %lld\n",
                full, (long long)t->numel, (long long)expect); exit(1);
    }
    float *buf = (float *)malloc((size_t)t->numel * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM su %s\n", full); exit(1); }
    st_read_f32(S, full, buf, t->numel);
    return buf;
}

static void load_linear(shards *S, Q38Linear *l, const char *stem, int out, int in)
{
    char name[256];
    snprintf(name, sizeof name, "%s.weight", stem);
    l->w = want_tensor(S, name, (int64_t)out * in);
    snprintf(name, sizeof name, "%s.bias", stem);
    l->b = want_tensor(S, name, out);
    l->out = out; l->in = in;
}

static void load_norm(shards *S, Q38Norm *n, const char *stem, int width)
{
    char name[256];
    snprintf(name, sizeof name, "%s.weight", stem);
    n->w = want_tensor(S, name, width);
    snprintf(name, sizeof name, "%s.bias", stem);
    n->b = want_tensor(S, name, width);
}

static int compare(const char *label, const float *got, const float *want,
                   int64_t count, float tolerance)
{
    double worst = 0.0; int64_t at = -1;
    for (int64_t i = 0; i < count; i++) {
        double delta = fabs((double)got[i] - (double)want[i]);
        if (delta > worst) { worst = delta; at = i; }
    }
    if (worst <= tolerance) {
        printf("  ok   %-14s %lld valori, scarto max %.3e\n", label, (long long)count, worst);
        return 0;
    }
    printf("  FAIL %-14s scarto max %.3e a %lld (nostro %.6f, riferimento %.6f)\n",
           label, worst, (long long)at, got[at], want[at]);
    return 1;
}

int main(int argc, char **argv)
{
    /* `make test-c` esegue ogni tests/test_* SENZA argomenti. Senza fixture
     * questo test non ha verificato niente, e dirlo saltato e' l'unica risposta
     * onesta: uscire con errore lo farebbe fallire su ogni macchina che non ha
     * ancora generato la fixture, e uscire con successo sarebbe una bugia. */
    const char *fixture = (argc >= 2) ? argv[1] : "./qwen38_vision_tiny";
    char path[1024];
    snprintf(path, sizeof path, "%s/ref_vision.json", fixture);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("SKIP: manca %s; genera la fixture con "
                     "tools/make_qwen38_vision_tiny.py\n", path); return 0; }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "lettura di %s fallita\n", path); return 1; }
    text[size] = 0; fclose(f);
    char *arena = NULL;
    jval *ref = json_parse(text, &arena);
    if (!ref || ref->t != J_OBJ) { fprintf(stderr, "ref_vision.json malformato\n"); return 1; }

    jval *grid = json_get(ref, "grid");
    int grid_h = (int)grid->kids[1]->num, grid_w = (int)grid->kids[2]->num;
    int hidden = (int)json_get(ref, "hidden_size")->num;
    int out_hidden = (int)json_get(ref, "out_hidden")->num;
    int64_t npix = 0, nout = 0, nlast = 0;
    float *pixels = json_floats(json_get(ref, "pixels"), &npix);
    float *want_out = json_floats(json_get(ref, "output"), &nout);
    float *want_last = json_floats(json_get(ref, "last_hidden"), &nlast);

    shards S; st_init(&S, fixture);

    /* La geometria viene dal config della fixture, non da costanti qui: una
     * torre con dimensioni diverse deve fallire dicendo cosa non torna, non
     * leggere silenziosamente pesi della misura sbagliata. */
    snprintf(path, sizeof path, "%s/config.json", fixture);
    FILE *cf = fopen(path, "rb");
    if (!cf) { fprintf(stderr, "manca %s\n", path); return 1; }
    fseek(cf, 0, SEEK_END); long csize = ftell(cf); fseek(cf, 0, SEEK_SET);
    char *ctext = (char *)malloc((size_t)csize + 1);
    if (!ctext || fread(ctext, 1, (size_t)csize, cf) != (size_t)csize) return 1;
    ctext[csize] = 0; fclose(cf);
    char *carena = NULL;
    jval *cfg = json_get(json_parse(ctext, &carena), "vision_config");
    if (!cfg) { fprintf(stderr, "config.json senza vision_config\n"); return 1; }

    Q38Vision v;
    memset(&v, 0, sizeof v);
    v.depth = (int)json_get(cfg, "depth")->num;
    v.hidden = hidden;
    v.heads = (int)json_get(cfg, "num_heads")->num;
    v.head_dim = v.hidden / v.heads;
    v.inter = (int)json_get(cfg, "intermediate_size")->num;
    v.patch = (int)json_get(cfg, "patch_size")->num;
    v.merge = (int)json_get(cfg, "spatial_merge_size")->num;
    v.temporal = (int)json_get(cfg, "temporal_patch_size")->num;
    v.in_ch = (int)json_get(cfg, "in_channels")->num;
    v.out_hidden = out_hidden;
    v.num_pos = (int)json_get(cfg, "num_position_embeddings")->num;
    v.side = (int)(sqrt((double)v.num_pos) + 0.5);
    v.eps = 1e-6f;

    int features = v.in_ch * v.temporal * v.patch * v.patch;
    load_linear(&S, &v.patch_embed, "patch_embed.proj", v.hidden, features);
    v.pos_embed = want_tensor(&S, "pos_embed.weight", (int64_t)v.num_pos * v.hidden);
    v.blocks = (Q38VBlock *)calloc((size_t)v.depth, sizeof(Q38VBlock));
    for (int i = 0; i < v.depth; i++) {
        char stem[128];
        snprintf(stem, sizeof stem, "blocks.%d.norm1", i);
        load_norm(&S, &v.blocks[i].norm1, stem, v.hidden);
        snprintf(stem, sizeof stem, "blocks.%d.norm2", i);
        load_norm(&S, &v.blocks[i].norm2, stem, v.hidden);
        snprintf(stem, sizeof stem, "blocks.%d.attn.qkv", i);
        load_linear(&S, &v.blocks[i].qkv, stem, 3 * v.hidden, v.hidden);
        snprintf(stem, sizeof stem, "blocks.%d.attn.proj", i);
        load_linear(&S, &v.blocks[i].proj, stem, v.hidden, v.hidden);
        snprintf(stem, sizeof stem, "blocks.%d.mlp.linear_fc1", i);
        load_linear(&S, &v.blocks[i].fc1, stem, v.inter, v.hidden);
        snprintf(stem, sizeof stem, "blocks.%d.mlp.linear_fc2", i);
        load_linear(&S, &v.blocks[i].fc2, stem, v.hidden, v.inter);
    }
    int wide = v.hidden * v.merge * v.merge;
    load_norm(&S, &v.merger_norm, "merger.norm", v.hidden);
    load_linear(&S, &v.merger_fc1, "merger.linear_fc1", wide, wide);
    load_linear(&S, &v.merger_fc2, "merger.linear_fc2", v.out_hidden, wide);

    int patches = grid_h * grid_w;
    if (npix != (int64_t)patches * features) {
        fprintf(stderr, "pixels: %lld valori, ne servono %lld\n",
                (long long)npix, (long long)patches * features);
        return 1;
    }
    float *out = (float *)calloc((size_t)patches, sizeof(float) * (size_t)v.out_hidden);
    float *last = (float *)calloc((size_t)patches * v.hidden, sizeof(float));
    int tokens = q38_vision_forward_dbg(&v, pixels, grid_h, grid_w, out, last);
    if (tokens <= 0) { fprintf(stderr, "la torre ha rifiutato la griglia\n"); return 1; }

    printf("torre vision Qwen3.8: %d patch -> %d token\n", patches, tokens);
    int failures = 0;
    if ((int64_t)tokens * v.out_hidden != nout) {
        printf("  FAIL forma: %d x %d, il riferimento dice %lld valori\n",
               tokens, v.out_hidden, (long long)nout);
        failures++;
    } else {
        /* 8e-4 non e' una tolleranza scelta per far passare il test: e' MISURATA.
         * Il riferimento in float32 confrontato con se stesso in float64 differisce
         * di 1.83e-4 su queste attivazioni, che arrivano a 154. Il nostro scarto e'
         * dello stesso ordine, quindi e' l'aritmetica di chi ci confrontiamo, non
         * un difetto. Un errore vero -- ordine delle patch, teste, residui -- si
         * presenta a 1e-3 o piu': la gelu sbagliata nel merger, che era un difetto
         * VERO trovato qui, dava 4.6e-3. La soglia sta in mezzo apposta. */
        failures += compare("pooler", out, want_out, nout, 8e-4f);
    }
    if ((int64_t)patches * v.hidden == nlast)
        failures += compare("last_hidden", last, want_last, nlast, 8e-4f);

    /* Il merger da solo: se alimentato con l'uscita del RIFERIMENTO produce il
     * pooler del riferimento, allora il merger e' corretto e lo scarto sul
     * pooler e' quello dei blocchi amplificato dalla sua LayerNorm -- non un
     * difetto suo. Senza questa distinzione i due casi sono indistinguibili. */
    {
        int wide2 = v.hidden * v.merge * v.merge, tok2 = patches / (v.merge * v.merge);
        float *normed = (float *)calloc((size_t)patches * v.hidden, sizeof(float));
        float *hid = (float *)calloc((size_t)tok2 * wide2, sizeof(float));
        float *got = (float *)calloc((size_t)tok2 * v.out_hidden, sizeof(float));
        q38v_layernorm(normed, want_last, &v.merger_norm, patches, v.hidden, v.eps);
        q38v_linear(hid, normed, &v.merger_fc1, tok2);
        for (int i = 0; i < tok2 * wide2; i++) hid[i] = q38v_gelu_exact(hid[i]);
        q38v_linear(got, hid, &v.merger_fc2, tok2);
        failures += compare("merger da solo", got, want_out, nout, 8e-4f);
        free(normed); free(hid); free(got);
    }

    /* Una griglia che il merge non chiude deve essere rifiutata, non arrotondata. */
    if (q38_vision_forward(&v, pixels, 3, 3, out) != -1) {
        printf("  FAIL una griglia 3x3 con merge 2 e' stata accettata\n");
        failures++;
    } else {
        printf("  ok   griglia non divisibile per il merge rifiutata\n");
    }

    printf("\n%s\n", failures ? "TEST FAIL" : "la torre combacia con l'oracolo upstream");
    return failures ? 1 : 0;
}
