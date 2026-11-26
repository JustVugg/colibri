/* qwen3_asr.c - Qwen3-ASR engine for colibri (MVP implementation in progress).
 *
 * Target: Qwen/Qwen3-ASR-1.7B (qwen3_asr), BF16 safetensors.
 * Pipeline: WAV/PCM16 -> 16kHz -> Whisper-style log-mel -> audio encoder ->
 * projector -> audio token injection -> Qwen3 dense decoder -> text.
 */

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "json.h"
#include "st.h"
#include "tok.h"
#include "serve_codec.h"
#include "qwen3_asr_audio.h"
#include "matmul_f16.h"
#include "ggml_view.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { float *w; float *b; int out, in, kh, kw, stride, pad; } Conv2d;
typedef struct { uint16_t *w; void *ggml; int ggml_type; float *b; int out, in; } Linear;
typedef struct { float *w; float *b; int n; } LayerNorm;

typedef struct {
    LayerNorm attn_ln, ffn_ln;
    Linear q, k, v, o;
    Linear qkv; /* fused q+k+v when available */
    Linear fc1, fc2;
    int n_heads, head_dim, d_model, ffn_dim;
} AsrEncLayer;

typedef struct {
    int d_model, n_heads, head_dim, n_layers, ffn_dim;
    int n_mels, n_window, n_window_infer, max_pos, output_dim;
    Conv2d conv1, conv2, conv3;
    Linear conv_out;
    float *pos_emb; /* [max_pos, d_model] */
    AsrEncLayer *layers;
    LayerNorm ln_post;
    Linear proj1, proj2;
} AudioTower;

typedef struct {
    LayerNorm ln1, ln2;
    Linear q, k, v, o;
    Linear gate, up, down;
    float *q_norm, *k_norm;
    int n_heads, n_kv_heads, head_dim, hidden, inter;
} TextLayer;

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab;
    int max_pos;
    float rms_eps, rope_theta;
    int mrope_section[4];
    uint16_t *embed; /* [vocab, hidden] bf16 */
    void *embed_ggml; int embed_ggml_type;
    TextLayer *layers;
    float *final_norm;
    uint16_t *lm_head; /* [vocab, hidden] bf16 or tied to embed */
    void *lm_head_ggml; int lm_head_ggml_type;
    int lm_head_tied;
} TextModel;

typedef struct {
    char model_dir[1024];
    char gguf_path[1024];
    char mmproj_path[1024];
    AudioTower audio;
    TextModel text;
    int audio_token_id;
    shards S;
    gguf_ctx Gguf;
    gguf_ctx Mmg;
    int has_gguf;
    int has_mmproj;
    Tok tok;
    int tok_loaded;
    int loaded;
} Qwen3Asr;

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "qwen3_asr: OOM\n"); exit(1); }
    return p;
}
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "qwen3_asr: OOM\n"); exit(1); }
    return p;
}

static void write_all(FILE *out, const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char *)buf;
    while (n > 0) {
        size_t w = fwrite(p, 1, n, out);
        if (w == 0) { perror("qwen3_asr: fwrite"); exit(1); }
        p += w; n -= w;
    }
}

static void emit_error(FILE *out, const char *id, const char *code, const char *msg) {
    char line[1024];
    int n = snprintf(line, sizeof(line), "ERROR %s %s %s\n", id, code ? code : "ERROR", msg ? msg : "error");
    if (n < 0) return;
    write_all(out, line, (size_t)n);
    fflush(out);
}

static void emit_transcript(FILE *out, const char *id, const char *text) {
    /* JSON encode minimal text. */
    size_t tlen = text ? strlen(text) : 0;
    size_t cap = tlen * 6 + 64;
    char *payload = (char *)xmalloc(cap);
    size_t o = 0;
    o += snprintf(payload + o, cap - o, "{\"text\":\"");
    for (size_t i = 0; i < tlen && o + 8 < cap; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"' || c == '\\') { payload[o++]='\\'; payload[o++]=c; }
        else if (c == '\n') { payload[o++]='\\'; payload[o++]='n'; }
        else if (c == '\r') { payload[o++]='\\'; payload[o++]='r'; }
        else if (c == '\t') { payload[o++]='\\'; payload[o++]='t'; }
        else if (c < 0x20) { o += snprintf(payload+o, cap-o, "\\u%04x", c); }
        else payload[o++] = (char)c;
    }
    o += snprintf(payload + o, cap - o, "\"}");
    char hdr[128];
    int hn = snprintf(hdr, sizeof(hdr), "TRANSCRIPT %s %zu\n", id, o);
    if (hn > 0) { write_all(out, hdr, (size_t)hn); write_all(out, payload, o); write_all(out, "\n", 1); }
    hn = snprintf(hdr, sizeof(hdr), "DONE %s STAT 0 0.0 0.0 0.0 0 0\n", id);
    if (hn > 0) write_all(out, hdr, (size_t)hn);
    fflush(out);
    free(payload);
}

static char *read_file(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = (char *)xmalloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[sz] = 0;
    if (out_n) *out_n = (size_t)sz;
    return buf;
}

static int json_get_int(jval *root, const char *key, int def) {
    jval *v = json_get(root, key);
    if (v && v->t == J_NUM) return (int)v->num;
    return def;
}
static float json_get_float(jval *root, const char *key, float def) {
    jval *v = json_get(root, key);
    if (v && v->t == J_NUM) return (float)v->num;
    return def;
}

static int load_linear(shards *S, const char *name, Linear *L, int out, int in, int want_bias) {
    char wn[512], bn[512];
    snprintf(wn, sizeof(wn), "%s.weight", name);
    L->out = out; L->in = in;
    st_tensor *wt = st_find(S, wn);
    if (!wt) { fprintf(stderr, "qwen3_asr: missing tensor %s\n", wn); return -1; }
    size_t n = (size_t)out * in;
    L->w = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    if (wt->dtype == 0) { /* BF16 */
        st_read_raw_cap(S, wn, L->w, (int64_t)n * 2, 0);
    } else if (wt->dtype == 1) { /* F16 -> store as bf16? keep f16 not supported by matmul_bf16w; convert to bf16 */
        uint16_t *tmp = (uint16_t *)xmalloc(n * sizeof(uint16_t));
        st_read_raw_cap(S, wn, tmp, (int64_t)n * 2, 0);
        for (size_t i = 0; i < n; i++) {
            float f = f16_to_f32(tmp[i]);
            uint32_t u; memcpy(&u, &f, 4);
            uint32_t lsb = (u >> 16) & 1;
            uint32_t rounding_bias = 0x7fff + lsb;
            L->w[i] = (uint16_t)((u + rounding_bias) >> 16);
        }
        free(tmp);
    } else { /* F32 -> convert to bf16 */
        float *tmp = (float *)xmalloc(n * sizeof(float));
        st_read_f32(S, wn, tmp, 0);
        for (size_t i = 0; i < n; i++) {
            uint32_t u; memcpy(&u, &tmp[i], 4);
            uint32_t lsb = (u >> 16) & 1;
            uint32_t rounding_bias = 0x7fff + lsb;
            L->w[i] = (uint16_t)((u + rounding_bias) >> 16);
        }
        free(tmp);
    }
    L->b = NULL;
    if (want_bias) {
        snprintf(bn, sizeof(bn), "%s.bias", name);
        st_tensor *t = st_find(S, bn);
        if (t) {
            L->b = (float *)xcalloc((size_t)out, sizeof(float));
            if (st_read_f32(S, bn, L->b, 0) < 0) { free(L->b); L->b = NULL; }
        }
    }
    return 0;
}

static const gguf_tensor *gguf_require(const gguf_ctx *g, const char *name) {
    const gguf_tensor *t = gguf_find(g, name);
    if (!t) fprintf(stderr, "qwen3_asr: missing gguf tensor %s\n", name);
    return t;
}

static int load_linear_gguf(gguf_ctx *g, const char *name, Linear *L, int want_bias) {
    char wn[512], bn[512];
    snprintf(wn, sizeof(wn), "%s.weight", name);
    const gguf_tensor *wt = gguf_require(g, wn);
    if (!wt) return -1;
    L->in = (int)wt->ne[0];
    L->out = (int)wt->ne[1];
    L->w = NULL; L->ggml = NULL; L->ggml_type = wt->type; L->b = NULL;
    size_t nbytes = (size_t)wt->nbytes;
    L->ggml = xmalloc(nbytes);
    if (gguf_read(g, wt, L->ggml, 0, (int64_t)nbytes) != 0) {
        fprintf(stderr, "qwen3_asr: failed to read gguf tensor %s\n", wn);
        return -1;
    }
    if (want_bias) {
        snprintf(bn, sizeof(bn), "%s.bias", name);
        const gguf_tensor *bt = gguf_find(g, bn);
        if (bt) {
            L->b = (float *)xcalloc((size_t)L->out, sizeof(float));
            if (gguf_read_f32(g, bt, L->b) != 0) { free(L->b); L->b = NULL; }
        }
    }
    return 0;
}


static int fuse_linear_qkv(Linear *dst, const Linear *q, const Linear *k, const Linear *v) {
    if (q->in!=k->in || q->in!=v->in || q->out!=k->out || q->out!=v->out) return -1;
    int in=q->in, out=q->out;
    memset(dst, 0, sizeof(*dst));
    dst->in=in; dst->out=out*3;
    if (q->ggml && k->ggml && v->ggml && q->ggml_type==k->ggml_type && q->ggml_type==v->ggml_type) {
        size_t rb = (size_t)gq_row_bytes(q->ggml_type, in);
        size_t total = rb * (size_t)out * 3;
        dst->ggml_type=q->ggml_type; dst->w=NULL;
        dst->ggml = xmalloc(total);
        memcpy((uint8_t*)dst->ggml, q->ggml, rb*out);
        memcpy((uint8_t*)dst->ggml + rb*out, k->ggml, rb*out);
        memcpy((uint8_t*)dst->ggml + rb*out*2, v->ggml, rb*out);
    } else if (q->w && k->w && v->w) {
        size_t row = (size_t)in;
        dst->w = (uint16_t*)xmalloc(row*(size_t)out*3*sizeof(uint16_t));
        memcpy(dst->w, q->w, row*out*sizeof(uint16_t));
        memcpy(dst->w + row*out, k->w, row*out*sizeof(uint16_t));
        memcpy(dst->w + row*out*2, v->w, row*out*sizeof(uint16_t));
    } else return -1;
    if (q->b || k->b || v->b) {
        dst->b = (float*)xcalloc((size_t)out*3, sizeof(float));
        if (q->b) memcpy(dst->b, q->b, (size_t)out*sizeof(float));
        if (k->b) memcpy(dst->b + out, k->b, (size_t)out*sizeof(float));
        if (v->b) memcpy(dst->b + out*2, v->b, (size_t)out*sizeof(float));
    }
    return 0;
}

static int load_f32_gguf(gguf_ctx *g, const char *name, float *out, int n) {
    const gguf_tensor *t = gguf_require(g, name);
    if (!t) return -1;
    if ((int)t->ne[0] != n && (int64_t)t->ne[0]*t->ne[1]*t->ne[2]*t->ne[3] != n) {
        /* allow if total matches */
    }
    if (gguf_read_f32(g, t, out) != 0) return -1;
    return 0;
}

static int load_layernorm(shards *S, const char *name, LayerNorm *LN, int n) {
    char wn[512], bn[512];
    snprintf(wn, sizeof(wn), "%s.weight", name);
    snprintf(bn, sizeof(bn), "%s.bias", name);
    LN->n = n;
    LN->w = (float *)xcalloc((size_t)n, sizeof(float));
    LN->b = (float *)xcalloc((size_t)n, sizeof(float));
    if (st_read_f32(S, wn, LN->w, 0) < 0) return -1;
    st_tensor *t = st_find(S, bn);
    if (t) st_read_f32(S, bn, LN->b, 0);
    return 0;
}

static int load_conv2d(shards *S, const char *name, Conv2d *C, int out, int in, int kh, int kw, int stride, int pad) {
    char wn[512], bn[512];
    snprintf(wn, sizeof(wn), "%s.weight", name);
    snprintf(bn, sizeof(bn), "%s.bias", name);
    C->out=out; C->in=in; C->kh=kh; C->kw=kw; C->stride=stride; C->pad=pad;
    size_t n = (size_t)out*in*kh*kw;
    C->w = (float *)xcalloc(n, sizeof(float));
    C->b = (float *)xcalloc((size_t)out, sizeof(float));
    if (st_read_f32(S, wn, C->w, 0) < 0) return -1;
    st_tensor *t = st_find(S, bn);
    if (t) st_read_f32(S, bn, C->b, 0);
    return 0;
}

static int load_config(Qwen3Asr *m) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/config.json", m->model_dir);
    size_t n=0; char *txt = read_file(path, &n);
    if (!txt) { fprintf(stderr, "qwen3_asr: cannot read %s\n", path); return -1; }
    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    free(txt);
    if (!root) { fprintf(stderr, "qwen3_asr: bad config.json\n"); return -1; }
    jval *thinker = json_get(root, "thinker_config");
    if (!thinker) thinker = root;
    jval *ac = json_get(thinker, "audio_config");
    jval *tc = json_get(thinker, "text_config");
    if (!ac || !tc) { free(arena); fprintf(stderr, "qwen3_asr: missing audio/text config\n"); return -1; }
    AudioTower *a = &m->audio;
    a->d_model = json_get_int(ac, "d_model", 1024);
    a->n_heads = json_get_int(ac, "encoder_attention_heads", 16);
    a->n_layers = json_get_int(ac, "encoder_layers", 24);
    a->ffn_dim = json_get_int(ac, "encoder_ffn_dim", 4096);
    a->n_mels = json_get_int(ac, "num_mel_bins", 128);
    a->n_window = json_get_int(ac, "n_window", 50);
    a->n_window_infer = json_get_int(ac, "n_window_infer", 800);
    a->max_pos = json_get_int(ac, "max_source_positions", 1500);
    a->output_dim = json_get_int(ac, "output_dim", 2048);
    a->head_dim = a->d_model / a->n_heads;
    TextModel *t = &m->text;
    t->hidden = json_get_int(tc, "hidden_size", 2048);
    t->n_layers = json_get_int(tc, "num_hidden_layers", 28);
    t->n_heads = json_get_int(tc, "num_attention_heads", 16);
    t->n_kv_heads = json_get_int(tc, "num_key_value_heads", 8);
    t->head_dim = json_get_int(tc, "head_dim", 128);
    t->inter = json_get_int(tc, "intermediate_size", 6144);
    t->vocab = json_get_int(tc, "vocab_size", 151936);
    t->max_pos = json_get_int(tc, "max_position_embeddings", 65536);
    t->rms_eps = json_get_float(tc, "rms_norm_eps", 1e-6f);
    jval *rope = json_get(tc, "rope_scaling");
    float rs_theta = rope ? json_get_float(rope, "rope_theta", -1.0f) : -1.0f;
    if (rs_theta > 0) t->rope_theta = rs_theta;
    else if (rope) t->rope_theta = 10000.0f; /* transformers 5.x default when rope_scaling present without rope_theta */
    else t->rope_theta = json_get_float(tc, "rope_theta", 10000.0f);
    if(getenv("QWEN3_ASR_ROPE_THETA")) t->rope_theta = (float)atof(getenv("QWEN3_ASR_ROPE_THETA"));
    jval *mrs = rope ? json_get(rope, "mrope_section") : NULL;
    for (int i=0;i<4;i++) t->mrope_section[i] = 0;
    if (mrs && mrs->t == J_ARR) {
        for (int i=0; i<mrs->len && i<4; i++) {
            jval *it = mrs->kids[i];
            if (it && it->t == J_NUM) t->mrope_section[i] = (int)it->num;
        }
    }
    m->audio_token_id = json_get_int(thinker, "audio_token_id", 151676);
    free(arena);
    return 0;
}
static float *build_sinusoids(int length, int channels) {
    float *pe = (float *)xcalloc((size_t)length * channels, sizeof(float));
    int half = channels / 2;
    float log_timescale_increment = logf(10000.0f) / (float)(half - 1 > 0 ? half - 1 : 1);
    for (int p = 0; p < length; p++) {
        for (int i = 0; i < half; i++) {
            float inv = expf(-log_timescale_increment * (float)i);
            float v = (float)p * inv;
            pe[p * channels + i] = sinf(v);
            pe[p * channels + half + i] = cosf(v);
        }
    }
    return pe;
}

static int load_audio_tower(Qwen3Asr *m) {
    AudioTower *a = &m->audio;
    if (load_conv2d(&m->S, "thinker.audio_tower.conv2d1", &a->conv1, a->d_model/2?480:a->d_model, 1, 3, 3, 2, 1) < 0) {
        /* config d_model for conv channels comes from downsample_hidden_size; try 480 */
    }
    /* conv channels: downsample_hidden_size is 480 per config */
    int ds = 480;
    free(a->conv1.w); free(a->conv1.b);
    if (load_conv2d(&m->S, "thinker.audio_tower.conv2d1", &a->conv1, ds, 1, 3, 3, 2, 1) < 0) return -1;
    if (load_conv2d(&m->S, "thinker.audio_tower.conv2d2", &a->conv2, ds, ds, 3, 3, 2, 1) < 0) return -1;
    if (load_conv2d(&m->S, "thinker.audio_tower.conv2d3", &a->conv3, ds, ds, 3, 3, 2, 1) < 0) return -1;
    /* conv_out: Linear(ds * (((n_mels+1)/2+1)/2+1)/2, d_model, bias=False) */
    int freq_after = ((((a->n_mels + 1) / 2 + 1) / 2 + 1) / 2);
    if (load_linear(&m->S, "thinker.audio_tower.conv_out", &a->conv_out, a->d_model, ds * freq_after, 0) < 0) return -1;
    a->pos_emb = build_sinusoids(a->max_pos, a->d_model);
    a->layers = (AsrEncLayer *)xcalloc((size_t)a->n_layers, sizeof(AsrEncLayer));
    for (int i = 0; i < a->n_layers; i++) {
        char pfx[256];
        snprintf(pfx, sizeof(pfx), "thinker.audio_tower.layers.%d", i);
        AsrEncLayer *L = &a->layers[i];
        L->n_heads = a->n_heads; L->head_dim = a->head_dim; L->d_model = a->d_model; L->ffn_dim = a->ffn_dim;
        char n[300];
        snprintf(n,sizeof(n),"%s.self_attn_layer_norm",pfx); if (load_layernorm(&m->S,n,&L->attn_ln,a->d_model)<0) return -1;
        snprintf(n,sizeof(n),"%s.final_layer_norm",pfx); if (load_layernorm(&m->S,n,&L->ffn_ln,a->d_model)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.q_proj",pfx); if (load_linear(&m->S,n,&L->q,a->d_model,a->d_model,1)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.k_proj",pfx); if (load_linear(&m->S,n,&L->k,a->d_model,a->d_model,1)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.v_proj",pfx); if (load_linear(&m->S,n,&L->v,a->d_model,a->d_model,1)<0) return -1;
        fuse_linear_qkv(&L->qkv, &L->q, &L->k, &L->v);
        snprintf(n,sizeof(n),"%s.self_attn.out_proj",pfx); if (load_linear(&m->S,n,&L->o,a->d_model,a->d_model,1)<0) return -1;
        snprintf(n,sizeof(n),"%s.fc1",pfx); if (load_linear(&m->S,n,&L->fc1,a->ffn_dim,a->d_model,1)<0) return -1;
        snprintf(n,sizeof(n),"%s.fc2",pfx); if (load_linear(&m->S,n,&L->fc2,a->d_model,a->ffn_dim,1)<0) return -1;
    }
    if (load_layernorm(&m->S, "thinker.audio_tower.ln_post", &a->ln_post, a->d_model) < 0) return -1;
    if (load_linear(&m->S, "thinker.audio_tower.proj1", &a->proj1, a->d_model, a->d_model, 1) < 0) return -1;
    if (load_linear(&m->S, "thinker.audio_tower.proj2", &a->proj2, a->output_dim, a->d_model, 1) < 0) return -1;
    return 0;
}



static int load_audio_tower_gguf(Qwen3Asr *m) {
    AudioTower *a = &m->audio;
    gguf_ctx *g = &m->Mmg;
    /* convs: ggml layout [kw,kh,ic,oc] -> our [oc,ic,kh,kw] */
    const char *conv_names[3] = {"a.conv2d.1","a.conv2d.2","a.conv2d.3"};
    Conv2d *convs[3] = {&a->conv1,&a->conv2,&a->conv3};
    int ds = 480;
    int in_c[3] = {1, ds, ds};
    for (int ci=0;ci<3;ci++){
        char wn[128], bn[128];
        snprintf(wn,sizeof(wn),"%s.weight",conv_names[ci]);
        snprintf(bn,sizeof(bn),"%s.bias",conv_names[ci]);
        const gguf_tensor *wt = gguf_require(g, wn);
        if(!wt) return -1;
        int kw=(int)wt->ne[0], kh=(int)wt->ne[1], ic=(int)wt->ne[2], oc=(int)wt->ne[3];
        Conv2d *C = convs[ci]; C->out=oc; C->in=ic; C->kh=kh; C->kw=kw; C->stride=2; C->pad=1;
        size_t n=(size_t)oc*ic*kh*kw;
        float *tmp=(float*)xmalloc(n*sizeof(float));
        if(gguf_read_f32(g,wt,tmp)!=0){free(tmp);return -1;}
        C->w=(float*)xcalloc(n,sizeof(float));
        /* ggml [kw,kh,ic,oc] contiguous in ne0 fastest -> index = kw + kh*kw + ic*kw*kh + oc*kw*kh*ic */
        for(int o=0;o<oc;o++) for(int i=0;i<ic;i++) for(int khh=0;khh<kh;khh++) for(int kww=0;kww<kw;kww++) {
            size_t gg = (size_t)kww + (size_t)khh*kw + (size_t)i*kw*kh + (size_t)o*kw*kh*ic;
            size_t my = (((size_t)o*ic+i)*kh+khh)*kw+kww;
            C->w[my]=tmp[gg];
        }
        free(tmp);
        C->b=(float*)xcalloc((size_t)oc,sizeof(float));
        const gguf_tensor *bt=gguf_find(g,bn);
        if(bt) gguf_read_f32(g,bt,C->b);
    }
    /* conv_out */
    int freq_after = ((((a->n_mels + 1) / 2 + 1) / 2 + 1) / 2);
    if (load_linear_gguf(g, "a.conv_out", &a->conv_out, 0) < 0) return -1;
    /* position embeddings */
    a->pos_emb = (float*)xcalloc((size_t)a->max_pos*a->d_model,sizeof(float));
    const gguf_tensor *pe = gguf_find(g, "a.position_embd.weight");
    if (pe) {
        /* ggml [d_model, max_pos] */
        float *tmp=(float*)xmalloc((size_t)a->max_pos*a->d_model*sizeof(float));
        if(gguf_read_f32(g,pe,tmp)==0) {
            for(int p=0;p<a->max_pos;p++) for(int d=0;d<a->d_model;d++) a->pos_emb[(size_t)p*a->d_model+d]=tmp[(size_t)d + (size_t)p*a->d_model];
        }
        free(tmp);
    } else {
        float *sinp = build_sinusoids(a->max_pos, a->d_model);
        memcpy(a->pos_emb, sinp, (size_t)a->max_pos*a->d_model*sizeof(float));
        free(sinp);
    }
    /* layers */
    a->layers = (AsrEncLayer *)xcalloc((size_t)a->n_layers, sizeof(AsrEncLayer));
    for (int i=0;i<a->n_layers;i++) {
        char pfx[64]; snprintf(pfx,sizeof(pfx),"a.blk.%d",i);
        char nm[128]; AsrEncLayer *L = &a->layers[i];
        L->n_heads=a->n_heads; L->head_dim=a->head_dim; L->d_model=a->d_model; L->ffn_dim=a->ffn_dim;
        snprintf(nm,sizeof(nm),"%s.ln1",pfx); L->attn_ln.n=a->d_model; L->attn_ln.w=(float*)xcalloc((size_t)a->d_model,sizeof(float)); L->attn_ln.b=(float*)xcalloc((size_t)a->d_model,sizeof(float));
        { char w[128],b[128]; snprintf(w,sizeof(w),"%s.weight",nm); snprintf(b,sizeof(b),"%s.bias",nm); load_f32_gguf(g,w,L->attn_ln.w,a->d_model); const gguf_tensor *bt=gguf_find(g,b); if(bt) gguf_read_f32(g,bt,L->attn_ln.b); }
        snprintf(nm,sizeof(nm),"%s.ln2",pfx); L->ffn_ln.n=a->d_model; L->ffn_ln.w=(float*)xcalloc((size_t)a->d_model,sizeof(float)); L->ffn_ln.b=(float*)xcalloc((size_t)a->d_model,sizeof(float));
        { char w[128],b[128]; snprintf(w,sizeof(w),"%s.weight",nm); snprintf(b,sizeof(b),"%s.bias",nm); load_f32_gguf(g,w,L->ffn_ln.w,a->d_model); const gguf_tensor *bt=gguf_find(g,b); if(bt) gguf_read_f32(g,bt,L->ffn_ln.b); }
        snprintf(nm,sizeof(nm),"%s.attn_q",pfx); if(load_linear_gguf(g,nm,&L->q,1)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.attn_k",pfx); if(load_linear_gguf(g,nm,&L->k,1)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.attn_v",pfx); if(load_linear_gguf(g,nm,&L->v,1)<0) return -1;
        if (fuse_linear_qkv(&L->qkv, &L->q, &L->k, &L->v) == 0) { /* fused */ }
        snprintf(nm,sizeof(nm),"%s.attn_out",pfx); if(load_linear_gguf(g,nm,&L->o,1)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.ffn_up",pfx); if(load_linear_gguf(g,nm,&L->fc1,1)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.ffn_down",pfx); if(load_linear_gguf(g,nm,&L->fc2,1)<0) return -1;
    }
    a->ln_post.n=a->d_model; a->ln_post.w=(float*)xcalloc((size_t)a->d_model,sizeof(float)); a->ln_post.b=(float*)xcalloc((size_t)a->d_model,sizeof(float));
    load_f32_gguf(g,"a.post_ln.weight",a->ln_post.w,a->d_model); { const gguf_tensor *bt=gguf_find(g,"a.post_ln.bias"); if(bt) gguf_read_f32(g,bt,a->ln_post.b); }
    if (load_linear_gguf(g, "mm.a.mlp.1", &a->proj1, 1) < 0) return -1;
    if (load_linear_gguf(g, "mm.a.mlp.2", &a->proj2, 1) < 0) return -1;
    return 0;
}

static int load_text_model_gguf(Qwen3Asr *m) {
    TextModel *t = &m->text;
    gguf_ctx *g = &m->Gguf;
    /* infer from tensors if config not set */
    const gguf_tensor *emb = gguf_require(g, "token_embd.weight");
    const gguf_tensor *q0 = gguf_require(g, "blk.0.attn_q.weight");
    const gguf_tensor *k0 = gguf_require(g, "blk.0.attn_k.weight");
    const gguf_tensor *ff0 = gguf_require(g, "blk.0.ffn_gate.weight");
    if (!emb || !q0 || !k0 || !ff0) return -1;
    if (t->hidden == 0) t->hidden = (int)emb->ne[0];
    if (t->vocab == 0) t->vocab = (int)emb->ne[1];
    if (t->n_heads == 0) t->n_heads = t->hidden / t->head_dim;
    if (t->n_kv_heads == 0) t->n_kv_heads = (int)k0->ne[1] / t->head_dim;
    if (t->inter == 0) t->inter = (int)ff0->ne[1];
    /* count layers */
    if (t->n_layers == 0) {
        for (int i=0;i<512;i++){ char nm[64]; snprintf(nm,sizeof(nm),"blk.%d.attn_q.weight",i); if(!gguf_find(g,nm)) { t->n_layers=i; break; } }
    }
    /* embed */
    t->embed = NULL;
    t->embed_ggml = xmalloc((size_t)emb->nbytes);
    t->embed_ggml_type = emb->type;
    if (gguf_read(g, emb, t->embed_ggml, 0, (int64_t)emb->nbytes) != 0) return -1;
    /* layers */
    t->layers = (TextLayer *)xcalloc((size_t)t->n_layers, sizeof(TextLayer));
    for (int i=0;i<t->n_layers;i++) {
        char pfx[64]; snprintf(pfx,sizeof(pfx),"blk.%d",i);
        char nm[128]; TextLayer *L = &t->layers[i];
        L->n_heads=t->n_heads; L->n_kv_heads=t->n_kv_heads; L->head_dim=t->head_dim; L->hidden=t->hidden; L->inter=t->inter;
        L->ln1.n=t->hidden; L->ln1.w=(float*)xcalloc((size_t)t->hidden,sizeof(float)); L->ln1.b=NULL;
        snprintf(nm,sizeof(nm),"%s.attn_norm.weight",pfx); if(load_f32_gguf(g,nm,L->ln1.w,t->hidden)<0) return -1;
        L->ln2.n=t->hidden; L->ln2.w=(float*)xcalloc((size_t)t->hidden,sizeof(float)); L->ln2.b=NULL;
        snprintf(nm,sizeof(nm),"%s.ffn_norm.weight",pfx); if(load_f32_gguf(g,nm,L->ln2.w,t->hidden)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.attn_q",pfx); if(load_linear_gguf(g,nm,&L->q,0)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.attn_k",pfx); if(load_linear_gguf(g,nm,&L->k,0)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.attn_v",pfx); if(load_linear_gguf(g,nm,&L->v,0)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.attn_output",pfx); if(load_linear_gguf(g,nm,&L->o,0)<0) return -1;
        L->q_norm=(float*)xcalloc((size_t)t->head_dim,sizeof(float));
        snprintf(nm,sizeof(nm),"%s.attn_q_norm.weight",pfx); if(load_f32_gguf(g,nm,L->q_norm,t->head_dim)<0){free(L->q_norm);L->q_norm=NULL;}
        L->k_norm=(float*)xcalloc((size_t)t->head_dim,sizeof(float));
        snprintf(nm,sizeof(nm),"%s.attn_k_norm.weight",pfx); if(load_f32_gguf(g,nm,L->k_norm,t->head_dim)<0){free(L->k_norm);L->k_norm=NULL;}
        snprintf(nm,sizeof(nm),"%s.ffn_gate",pfx); if(load_linear_gguf(g,nm,&L->gate,0)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.ffn_up",pfx); if(load_linear_gguf(g,nm,&L->up,0)<0) return -1;
        snprintf(nm,sizeof(nm),"%s.ffn_down",pfx); if(load_linear_gguf(g,nm,&L->down,0)<0) return -1;
    }
    t->final_norm = (float*)xcalloc((size_t)t->hidden,sizeof(float));
    if (load_f32_gguf(g, "output_norm.weight", t->final_norm, t->hidden) < 0) return -1;
    const gguf_tensor *lh = gguf_find(g, "output.weight");
    if (lh) {
        t->lm_head = NULL;
        t->lm_head_ggml = xmalloc((size_t)lh->nbytes);
        t->lm_head_ggml_type = lh->type;
        if (gguf_read(g, lh, t->lm_head_ggml, 0, (int64_t)lh->nbytes) != 0) return -1;
        t->lm_head_tied = 0;
    } else {
        t->lm_head_ggml = t->embed_ggml; t->lm_head_ggml_type = t->embed_ggml_type; t->lm_head_tied = 1;
    }
    return 0;
}

static int load_text_model(Qwen3Asr *m) {
    TextModel *t = &m->text;
    size_t emb_n = (size_t)t->vocab * t->hidden;
    t->embed = (uint16_t *)xmalloc(emb_n * sizeof(uint16_t));
    st_tensor *et = st_find(&m->S, "thinker.model.embed_tokens.weight");
    if (!et) return -1;
    if (et->dtype == 0) st_read_raw_cap(&m->S, "thinker.model.embed_tokens.weight", t->embed, (int64_t)emb_n*2, 0);
    else {
        float *tmp = (float *)xmalloc(emb_n * sizeof(float));
        if (st_read_f32(&m->S, "thinker.model.embed_tokens.weight", tmp, 0) < 0) { free(tmp); return -1; }
        for (size_t i=0;i<emb_n;i++){ uint32_t u; memcpy(&u,&tmp[i],4); uint32_t lsb=(u>>16)&1; t->embed[i]=(uint16_t)((u+0x7fff+lsb)>>16); }
        free(tmp);
    }
    t->layers = (TextLayer *)xcalloc((size_t)t->n_layers, sizeof(TextLayer));
    for (int i = 0; i < t->n_layers; i++) {
        char pfx[256]; snprintf(pfx, sizeof(pfx), "thinker.model.layers.%d", i);
        char n[320]; TextLayer *L = &t->layers[i];
        L->n_heads=t->n_heads; L->n_kv_heads=t->n_kv_heads; L->head_dim=t->head_dim; L->hidden=t->hidden; L->inter=t->inter;
        snprintf(n,sizeof(n),"%s.input_layernorm.weight",pfx); L->ln1.n=t->hidden; L->ln1.w=(float*)xcalloc((size_t)t->hidden,sizeof(float)); L->ln1.b=NULL; if (st_read_f32(&m->S,n,L->ln1.w,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.post_attention_layernorm.weight",pfx); L->ln2.n=t->hidden; L->ln2.w=(float*)xcalloc((size_t)t->hidden,sizeof(float)); L->ln2.b=NULL; if (st_read_f32(&m->S,n,L->ln2.w,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.q_proj",pfx); if (load_linear(&m->S,n,&L->q,t->n_heads*t->head_dim,t->hidden,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.k_proj",pfx); if (load_linear(&m->S,n,&L->k,t->n_kv_heads*t->head_dim,t->hidden,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.v_proj",pfx); if (load_linear(&m->S,n,&L->v,t->n_kv_heads*t->head_dim,t->hidden,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.o_proj",pfx); if (load_linear(&m->S,n,&L->o,t->hidden,t->n_heads*t->head_dim,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.self_attn.q_norm.weight",pfx); L->q_norm=(float*)xcalloc((size_t)t->head_dim,sizeof(float)); if (st_read_f32(&m->S,n,L->q_norm,0)<0) { free(L->q_norm); L->q_norm=NULL; }
        snprintf(n,sizeof(n),"%s.self_attn.k_norm.weight",pfx); L->k_norm=(float*)xcalloc((size_t)t->head_dim,sizeof(float)); if (st_read_f32(&m->S,n,L->k_norm,0)<0) { free(L->k_norm); L->k_norm=NULL; }
        snprintf(n,sizeof(n),"%s.mlp.gate_proj",pfx); if (load_linear(&m->S,n,&L->gate,t->inter,t->hidden,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.mlp.up_proj",pfx); if (load_linear(&m->S,n,&L->up,t->inter,t->hidden,0)<0) return -1;
        snprintf(n,sizeof(n),"%s.mlp.down_proj",pfx); if (load_linear(&m->S,n,&L->down,t->hidden,t->inter,0)<0) return -1;
    }
    t->final_norm = (float *)xcalloc((size_t)t->hidden, sizeof(float));
    if (st_read_f32(&m->S, "thinker.model.norm.weight", t->final_norm, 0) < 0) return -1;
    st_tensor *lh = st_find(&m->S, "thinker.lm_head.weight");
    if (lh) {
        size_t lhn = (size_t)t->vocab * t->hidden;
        t->lm_head = (uint16_t *)xmalloc(lhn * sizeof(uint16_t));
        if (lh->dtype == 0) st_read_raw_cap(&m->S, "thinker.lm_head.weight", t->lm_head, (int64_t)lhn*2, 0);
        else {
            float *tmp = (float *)xmalloc(lhn * sizeof(float));
            if (st_read_f32(&m->S, "thinker.lm_head.weight", tmp, 0) < 0) { free(tmp); return -1; }
            for (size_t i=0;i<lhn;i++){ uint32_t u; memcpy(&u,&tmp[i],4); uint32_t lsb=(u>>16)&1; t->lm_head[i]=(uint16_t)((u+0x7fff+lsb)>>16); }
            free(tmp);
        }
        t->lm_head_tied = 0;
    } else {
        t->lm_head = t->embed;
        t->lm_head_tied = 1;
    }
    return 0;
}

static int load_model(Qwen3Asr *m) {
    int has_dir = m->model_dir[0] != 0;
    if (has_dir && load_config(m) < 0) return -1;
    if (m->gguf_path[0]) {
        gguf_header_only = 0;
        if (gguf_open(&m->Gguf, m->gguf_path) != 0) {
            fprintf(stderr, "qwen3_asr: gguf_open failed: %s\n", m->Gguf.err);
            return -1;
        }
        m->has_gguf = 1;
    }
    if (m->mmproj_path[0]) {
        gguf_header_only = 0;
        if (gguf_open(&m->Mmg, m->mmproj_path) != 0) {
            fprintf(stderr, "qwen3_asr: mmproj gguf_open failed: %s\n", m->Mmg.err);
            return -1;
        }
        m->has_mmproj = 1;
    }
    if (has_dir) st_init(&m->S, m->model_dir); /* opens safetensors in dir */
    char tok_path[1200];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", m->model_dir);
    FILE *tf = fopen(tok_path, "rb");
    if (!tf) {
        /* Fallback: generate tokenizer.json from vocab.json + merges.txt */
        char vocab_path[1200], merges_path[1200], gen_path[1200];
        snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", m->model_dir);
        snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", m->model_dir);
        snprintf(gen_path, sizeof(gen_path), "/tmp/opencode/qwen3_asr_tokenizer.json");
        FILE *vf=fopen(vocab_path,"rb"), *mf=fopen(merges_path,"rb");
        if (vf && mf) {
            fclose(vf); fclose(mf);
            char cmd[4096];
            snprintf(cmd, sizeof(cmd),
                "python3 -c \"import json,os\n"
                "v=json.load(open('%s'))\n"
                "m=[]\n"
                "for i,l in enumerate(open('%s')):\n"
                " l=l.rstrip('\\n')\n"
                " if i==0 and l.startswith('#'): continue\n"
                " if l: m.append(l)\n"
                "tc=json.load(open('%s/tokenizer_config.json')) if os.path.exists('%s/tokenizer_config.json') else {}\n"
                "at=tc.get('added_tokens_decoder',{})\n"
                "added=[{'id':int(k),'content':v['content'],'single_word':v.get('single_word',False),'lstrip':v.get('lstrip',False),'rstrip':v.get('rstrip',False),'normalized':v.get('normalized',False),'special':v.get('special',False)} for k,v in sorted(at.items(),key=lambda x:int(x[0]))]\n"
                "json.dump({'version':'1.0','model':{'type':'BPE','vocab':v,'merges':m},'added_tokens':added}, open('%s','w'))\n\" >/dev/null 2>&1",
                vocab_path, merges_path, m->model_dir, m->model_dir, gen_path);
            int rc = system(cmd);
            (void)rc;
            tf = fopen(gen_path, "rb");
            if (tf) { fclose(tf); strncpy(tok_path, gen_path, sizeof(tok_path)-1); tf = fopen(tok_path, "rb"); }
        } else {
            if (vf) fclose(vf);
            if (mf) fclose(mf);
        }
    }
    if (tf) { fclose(tf); tok_load(&m->tok, tok_path); m->tok_loaded = 1; }
    if (m->has_mmproj) {
        if (load_audio_tower_gguf(m) < 0) return -1;
    } else {
        if (load_audio_tower(m) < 0) return -1;
    }
    if (m->has_gguf) {
        if (load_text_model_gguf(m) < 0) return -1;
    } else {
        if (load_text_model(m) < 0) return -1;
    }
    m->loaded = 1;
    return 0;
}

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static void layernorm(float *out, const float *x, const float *w, const float *b, int n) {
    float mean = 0.0f;
    for (int i=0;i<n;i++) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int i=0;i<n;i++) { float d=x[i]-mean; var += d*d; }
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i=0;i<n;i++) out[i] = (x[i]-mean)*inv*w[i] + (b?b[i]:0.0f);
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i=0;i<n;i++) ss += x[i]*x[i];
    ss = 1.0f / sqrtf(ss/(float)n + eps);
    for (int i=0;i<n;i++) out[i] = x[i]*ss*w[i];
}

static inline ColiTensorView linear_ggml_view(const Linear *L) {
    ColiTensorView v;
    memset(&v, 0, sizeof(v));
    v.format = COLI_TENSOR_GGML;
    v.data = L->ggml;
    v.rows = L->out;
    v.columns = L->in;
    v.block_columns = (uint32_t)L->ggml_type;
    return v;
}

static void linear_f32(float *out, const float *x, const Linear *L) {
    if (L->ggml) {
        ColiTensorView v = linear_ggml_view(L);
        coli_ggml_matmul(out, &v, x, 1);
        if (L->b) for (int o=0;o<L->out;o++) out[o] += L->b[o];
        return;
    }
    #pragma omp parallel for schedule(static) if(L->out > 64)
    for (int o=0;o<L->out;o++) {
        float acc = L->b ? L->b[o] : 0.0f;
        const uint16_t *rw = L->w + (size_t)o*L->in;
        acc += dot_bf16w_lanes(x, rw, L->in);
        out[o] = acc;
    }
}

/* out[rows,out_dim], x[rows,in_dim], w[out_dim,in_dim] bf16 */
static void matmul_bf16(float *out, const float *x, const uint16_t *w, const float *b, int rows, int in_dim, int out_dim) {
    matmul_bf16w(out, x, w, rows, in_dim, out_dim);
    if (b) {
        #pragma omp parallel for schedule(static) if(rows*out_dim > 256)
        for (int r=0;r<rows;r++) {
            float *orow = out + (size_t)r*out_dim;
            for (int o=0;o<out_dim;o++) orow[o] += b[o];
        }
    }
}

static void matmul_linear(float *out, const float *x, const Linear *L, int rows) {
    if (L->ggml) {
        ColiTensorView v = linear_ggml_view(L);
        coli_ggml_matmul(out, &v, x, rows);
        if (L->b) {
            #pragma omp parallel for schedule(static) if(rows*L->out > 256)
            for (int r=0;r<rows;r++) {
                float *orow = out + (size_t)r*L->out;
                for (int o=0;o<L->out;o++) orow[o] += L->b[o];
            }
        }
        return;
    }
    matmul_bf16(out, x, L->w, L->b, rows, L->in, L->out);
}

static void conv2d_f32(float *out, const float *in, const Conv2d *C, int H, int W) {
    /* in: [C.in,H,W], out: [C.out,H2,W2] */
    int H2 = (H + 2*C->pad - C->kh) / C->stride + 1;
    int W2 = (W + 2*C->pad - C->kw) / C->stride + 1;
    #pragma omp parallel for schedule(static) if(C->out > 1)
    for (int oc=0; oc<C->out; oc++) {
        float *ooc = out + (size_t)oc*H2*W2;
        float bias = C->b ? C->b[oc] : 0.0f;
        for (int i=0;i<H2*W2;i++) ooc[i] = bias;
        for (int ic=0; ic<C->in; ic++) {
            const float *wic = C->w + ((size_t)oc*C->in + ic)*C->kh*C->kw;
            const float *iic = in + (size_t)ic*H*W;
            for (int kh=0; kh<C->kh; kh++) {
                int ih0 = kh - C->pad;
                for (int kw=0; kw<C->kw; kw++) {
                    float ww = wic[(size_t)kh*C->kw + kw];
                    if (ww == 0.0f) continue;
                    int iw0 = kw - C->pad;
                    for (int oh=0; oh<H2; oh++) {
                        int ih = oh*C->stride + ih0;
                        if ((unsigned)ih >= (unsigned)H) continue;
                        float *orow = ooc + (size_t)oh*W2;
                        const float *irow = iic + (size_t)ih*W;
                        for (int ow=0; ow<W2; ow++) {
                            int iw = ow*C->stride + iw0;
                            if ((unsigned)iw < (unsigned)W) orow[ow] += irow[iw] * ww;
                        }
                    }
                }
            }
        }
    }
}

static void apply_gelu(float *x, size_t n) { for (size_t i=0;i<n;i++) x[i]=gelu(x[i]); }

static double now_ms(void);
static int audio_encoder_forward(Qwen3Asr *m, const float *mel, int n_mels, int n_frames, float **out_emb, int *out_n) {
    /* MVP: single-chunk simplified path for short audio. TODO: full chunk/window packing. */
    AudioTower *a = &m->audio;
    if (n_mels != a->n_mels) return -1;
    int chunk_len = a->n_window * 2; /* 100 */
    int pad_frames = ((n_frames + chunk_len - 1) / chunk_len) * chunk_len;
    if (pad_frames <= 0) pad_frames = chunk_len;
    /* Build padded mel [1, n_mels, pad_frames]. Pad with mel floor (not 0). */
    float mel_floor = mel[0];
    for (int i=1; i<n_frames*n_mels; i++) if (mel[i] < mel_floor) mel_floor = mel[i];
    float *padded = (float *)malloc((size_t)n_mels * pad_frames * sizeof(float));
    for (size_t i=0, nn=(size_t)n_mels*pad_frames; i<nn; i++) padded[i] = mel_floor;
    for (int f=0; f<n_frames; f++) for (int mi=0; mi<n_mels; mi++) padded[(size_t)mi*pad_frames + f] = mel[(size_t)f*n_mels + mi];
    int num_chunks = pad_frames / chunk_len;
    int ds = a->conv1.out;
    int freq1 = (n_mels + 2*a->conv1.pad - a->conv1.kh)/a->conv1.stride + 1;
    int time3 = (chunk_len + 2*a->conv1.pad - a->conv1.kh)/a->conv1.stride + 1;
    time3 = (time3 + 2*a->conv2.pad - a->conv2.kh)/a->conv2.stride + 1;
    time3 = (time3 + 2*a->conv3.pad - a->conv3.kh)/a->conv3.stride + 1;
    /* per chunk conv */
    int total_tokens = 0;
    int *tok_per_chunk = (int*)xcalloc((size_t)num_chunks,sizeof(int));
    for (int c=0;c<num_chunks;c++) {
        int valid = n_frames - c*chunk_len; if (valid<0) valid=0; if (valid>chunk_len) valid=chunk_len;
        /* compute _get_feat_extract_output_lengths: chunk_len=n_window*2 */
        int input_lengths_leave = valid % chunk_len; /* 0 when full chunk */
        int feat;
        if (input_lengths_leave == 0) {
            /* full 100-frame chunk: 100->50->25->13 */
            int t1 = (chunk_len - 1) / 2 + 1;
            int t2 = (t1 - 1) / 2 + 1;
            feat = (t2 - 1) / 2 + 1;
        } else {
            feat = (input_lengths_leave - 1) / 2 + 1;
            feat = (feat - 1) / 2 + 1;
            feat = (feat - 1) / 2 + 1;
        }
        if (valid <= 0) feat = 0;
        tok_per_chunk[c]=feat; total_tokens+=feat;
    }
    if (total_tokens <= 0) { free(padded); free(tok_per_chunk); return -1; }
    float *hidden = (float*)xcalloc((size_t)total_tokens * a->d_model, sizeof(float));
    double t_conv0 = now_ms();
    int out_pos = 0;
    for (int c=0;c<num_chunks;c++) {
        /* extract chunk [1,n_mels,chunk_len] */
        float *in = (float*)xcalloc((size_t)n_mels*chunk_len,sizeof(float));
        for (int mi=0;mi<n_mels;mi++) memcpy(in + (size_t)mi*chunk_len, padded + (size_t)mi*pad_frames + c*chunk_len, (size_t)chunk_len*sizeof(float));
        (void)ds; (void)freq1;
        /* compute actual conv sizes stepwise */
        int H=n_mels, W=chunk_len;
        int H1=(H+2*a->conv1.pad-a->conv1.kh)/a->conv1.stride+1; int W1=(W+2*a->conv1.pad-a->conv1.kw)/a->conv1.stride+1;
        int H2f=(H1+2*a->conv2.pad-a->conv2.kh)/a->conv2.stride+1; int W2f=(W1+2*a->conv2.pad-a->conv2.kw)/a->conv2.stride+1;
        int H3=(H2f+2*a->conv3.pad-a->conv3.kh)/a->conv3.stride+1; int W3=(W2f+2*a->conv3.pad-a->conv3.kw)/a->conv3.stride+1;
        float *o1=(float*)xcalloc((size_t)ds*H1*W1,sizeof(float));
        float *o2=(float*)xcalloc((size_t)ds*H2f*W2f,sizeof(float));
        float *o3=(float*)xcalloc((size_t)ds*H3*W3,sizeof(float));
        conv2d_f32(o1,in,&a->conv1,H,W); apply_gelu(o1,(size_t)ds*H1*W1);
        conv2d_f32(o2,o1,&a->conv2,H1,W1); apply_gelu(o2,(size_t)ds*H2f*W2f);
        conv2d_f32(o3,o2,&a->conv3,H2f,W2f); apply_gelu(o3,(size_t)ds*H3*W3);
        /* o3: [ds,H3,W3], permute to [W3, ds*H3], linear to d_model */
        int feat = tok_per_chunk[c];
        for (int t=0;t<feat && t<W3;t++) {
            float *tmp = (float*)xcalloc((size_t)ds*H3,sizeof(float));
            for (int ic=0;ic<ds;ic++) for (int fh=0;fh<H3;fh++) tmp[(size_t)ic*H3+fh] = o3[((size_t)ic*H3+fh)*W3 + t];
            /* add pos emb (per chunk time position, as in transformers: pos[:time_steps] added before selecting valid) */
            float *proj = hidden + (size_t)(out_pos+t)*a->d_model;
            linear_f32(proj, tmp, &a->conv_out);
            for (int d=0; d<a->d_model; d++) proj[d] += a->pos_emb[(size_t)t*a->d_model + d];
            free(tmp);
        }
        out_pos += feat;
        free(in); free(o1); free(o2); free(o3);
    }
    int max_chunk_aftercnn = 0;
    for (int c=0;c<num_chunks;c++) if (tok_per_chunk[c] > max_chunk_aftercnn) max_chunk_aftercnn = tok_per_chunk[c];
    double t_conv1 = now_ms();
    free(padded); free(tok_per_chunk);

    /* Transformer encoder layers */
    int T = total_tokens;
    int D = a->d_model;
    int Hh = a->n_heads;
    int Hd = a->head_dim;
    /* cu_seqlens for n_window_infer windowed attention (matches HF get_audio_cu_seqlens). */
    int n_window_ratio = (a->n_window * 2 > 0) ? (a->n_window_infer / (a->n_window * 2)) : 1;
    if (n_window_ratio < 1) n_window_ratio = 1;
    int window_aftercnn = max_chunk_aftercnn * n_window_ratio;
    if (window_aftercnn <= 0) window_aftercnn = T;
    int nwin = (T + window_aftercnn - 1) / window_aftercnn;
    int *cu = (int*)xmalloc(((size_t)nwin + 1) * sizeof(int));
    cu[0] = 0;
    int cu_n = 0;
    for (int pos=0; pos<T; pos += window_aftercnn) {
        int end = pos + window_aftercnn; if (end > T) end = T;
        cu[++cu_n] = end;
    }
    float *buf = (float*)xcalloc((size_t)T*D,sizeof(float));
    float *q=(float*)xcalloc((size_t)T*D,sizeof(float));
    float *k=(float*)xcalloc((size_t)T*D,sizeof(float));
    float *v=(float*)xcalloc((size_t)T*D,sizeof(float));
    float *qkv_out=(float*)xcalloc((size_t)T*3*D,sizeof(float));
    float *att=(float*)xcalloc((size_t)Hh*T*T,sizeof(float));
    float *tmp=(float*)xcalloc((size_t)T*D,sizeof(float));
    double t_layers0 = now_ms();
    for (int li=0; li<a->n_layers; li++) {
        AsrEncLayer *L = &a->layers[li];
        /* self attn */
        for (int t=0;t<T;t++) layernorm(buf + (size_t)t*D, hidden + (size_t)t*D, L->attn_ln.w, L->attn_ln.b, D);
        if (L->qkv.ggml || L->qkv.w) {
            matmul_linear(qkv_out, buf, &L->qkv, T);
            for (int ti=0;ti<T;ti++) {
                memcpy(q + (size_t)ti*D, qkv_out + (size_t)ti*3*D, (size_t)D*sizeof(float));
                memcpy(k + (size_t)ti*D, qkv_out + (size_t)ti*3*D + D, (size_t)D*sizeof(float));
                memcpy(v + (size_t)ti*D, qkv_out + (size_t)ti*3*D + 2*D, (size_t)D*sizeof(float));
            }
        } else {
            matmul_linear(q, buf, &L->q, T);
            matmul_linear(k, buf, &L->k, T);
            matmul_linear(v, buf, &L->v, T);
        }
        /* windowed attention per cu_seqlens */
        for (int h=0;h<Hh;h++) {
            for (int wi=0; wi<cu_n; wi++) {
                int ws = cu[wi], we = cu[wi+1];
                for (int i=ws;i<we;i++) {
                    float maxv = -1e30f;
                    for (int j=ws;j<we;j++) {
                        float dot=0.0f;
                        for (int d=0;d<Hd;d++) dot += q[(size_t)i*D+h*Hd+d]*k[(size_t)j*D+h*Hd+d];
                        dot /= sqrtf((float)Hd);
                        att[((size_t)h*T+i)*T+j]=dot;
                        if (dot>maxv) maxv=dot;
                    }
                    float sum=0.0f;
                    for (int j=ws;j<we;j++) { att[((size_t)h*T+i)*T+j]=expf(att[((size_t)h*T+i)*T+j]-maxv); sum+=att[((size_t)h*T+i)*T+j]; }
                    if (sum > 0) for (int j=ws;j<we;j++) att[((size_t)h*T+i)*T+j]/=sum;
                }
            }
        }
        memset(tmp,0,(size_t)T*D*sizeof(float));
        for (int h=0;h<Hh;h++) for (int wi=0; wi<cu_n; wi++) {
            int ws = cu[wi], we = cu[wi+1];
            for (int i=ws;i<we;i++) for (int j=ws;j<we;j++) {
                float wgt = att[((size_t)h*T+i)*T+j];
                for (int d=0;d<Hd;d++) tmp[(size_t)i*D+h*Hd+d] += wgt * v[(size_t)j*D+h*Hd+d];
            }
        }
        matmul_linear(buf, tmp, &L->o, T);
        for (size_t i=0;i<(size_t)T*D;i++) hidden[i]+=buf[i];
        /* ffn */
        for (int t=0;t<T;t++) layernorm(buf + (size_t)t*D, hidden + (size_t)t*D, L->ffn_ln.w, L->ffn_ln.b, D);
        float *ff = (float*)xcalloc((size_t)T*a->ffn_dim,sizeof(float));
        matmul_linear(ff, buf, &L->fc1, T);
        for (size_t i=0;i<(size_t)T*a->ffn_dim;i++) ff[i]=gelu(ff[i]);
        matmul_linear(buf, ff, &L->fc2, T);
        for (size_t i=0;i<(size_t)T*D;i++) hidden[i]+=buf[i];
        free(ff);
    }
    double t_layers1 = now_ms();
    /* ln_post */
    for (int t=0;t<T;t++) layernorm(buf + (size_t)t*D, hidden + (size_t)t*D, a->ln_post.w, a->ln_post.b, D);
    memcpy(hidden, buf, (size_t)T*D*sizeof(float));
    /* projector: linear1 + gelu + linear2 */
    float *p1=(float*)xcalloc((size_t)T*a->d_model,sizeof(float));
    matmul_linear(p1, hidden, &a->proj1, T);
    for (size_t i=0;i<(size_t)T*a->d_model;i++) p1[i]=gelu(p1[i]);
    float *p2=(float*)xcalloc((size_t)T*a->output_dim,sizeof(float));
    matmul_linear(p2, p1, &a->proj2, T);
    double t_proj1 = now_ms();
    if (getenv("QWEN3_ASR_DEBUG_TIMING")) fprintf(stderr,"[qwen3_asr] audio_enc_breakdown conv=%.1fms layers=%.1fms proj=%.1fms T=%d\n", t_conv1-t_conv0, t_layers1-t_layers0, t_proj1-t_layers1, T);
    free(p1); free(buf); free(q); free(k); free(v); free(qkv_out); free(att); free(tmp); free(hidden); free(cu);
    *out_emb = p2; *out_n = T;
    return 0;
}

static void rope(float *x, int n_heads, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int h=0;h<n_heads;h++) {
        for (int i=0;i<half;i++) {
            float freq = 1.0f / powf(theta, (float)(2*i) / (float)head_dim);
            float ang = pos * freq;
            float c = cosf(ang), s = sinf(ang);
            float x0 = x[h*head_dim+i], x1 = x[h*head_dim+i+half];
            x[h*head_dim+i] = x0*c - x1*s;
            x[h*head_dim+i+half] = x0*s + x1*c;
        }
    }
}

/* mRoPE: Qwen3-style multimodal RoPE. positions[3] = [t,h,w] (or same for text).
 * mrope_section gives the count of head-dim pairs per axis; we apply each axis
 * to its section, interleaved by 2 in the rotary layout. */
static void mrope(float *x, int n_heads, int head_dim, const int pos[3], float theta, const int mrope_section[4]) {
    int sec[3] = {mrope_section[0], mrope_section[1], mrope_section[2]};
    int total = sec[0]+sec[1]+sec[2];
    if (total <= 0) { /* fallback plain rope */
        rope(x, n_heads, head_dim, pos[0], theta);
        return;
    }
    /* Interleaved mRoPE as in Qwen3-VL: for H/W dims, place at offset+3*j. */
    int pairs = head_dim / 2;
    int *axis = (int*)malloc((size_t)pairs * sizeof(int));
    for (int i=0;i<pairs;i++) axis[i]=0; /* default T */
    for (int dim=1; dim<3; dim++) { /* 1=H, 2=W */
        int length = sec[dim] * 3;
        for (int j=dim; j<length && j<pairs; j+=3) axis[j] = dim;
    }
    int half = head_dim / 2;
    for (int h=0; h<n_heads; h++) {
        for (int i=0; i<pairs; i++) {
            int a = axis[i];
            float freq = 1.0f / powf(theta, (float)(2*i) / (float)head_dim);
            float ang = (float)pos[a] * freq;
            float c = cosf(ang), s = sinf(ang);
            float x0 = x[h*head_dim+i], x1 = x[h*head_dim+i+half];
            x[h*head_dim+i] = x0*c - x1*s;
            x[h*head_dim+i+half] = x0*s + x1*c;
        }
    }
    free(axis);
}

typedef struct {
    float *k; /* [max_seq, n_kv_heads*head_dim] */
    float *v;
    int len;
    int cap;
} KVCache;

static KVCache *kv_alloc(int n_layers, int cap, int kv_dim) {
    KVCache *c = (KVCache*)xcalloc((size_t)n_layers, sizeof(KVCache));
    for (int i=0;i<n_layers;i++) { c[i].cap=cap; c[i].len=0; c[i].k=(float*)xcalloc((size_t)cap*kv_dim,sizeof(float)); c[i].v=(float*)xcalloc((size_t)cap*kv_dim,sizeof(float)); }
    return c;
}
static void kv_free(KVCache *c, int n_layers) { if(!c) return; for(int i=0;i<n_layers;i++){free(c[i].k); free(c[i].v);} free(c); }

static int text_forward(Qwen3Asr *m, const int *tokens, const float *audio_emb, int audio_n, int n_tokens, int start_pos, KVCache *cache, float *logits_out) {
    double t_tf0 = now_ms();
    TextModel *t = &m->text;
    int D=t->hidden, H=t->n_heads, KVH=t->n_kv_heads, HD=t->head_dim, KVdim=KVH*HD;
    int S = n_tokens;
    float *x = (float*)xcalloc((size_t)S*D,sizeof(float));
    /* embeddings + audio injection */
    int ai = 0;
    for (int i=0;i<S;i++) {
        int tok = tokens[i];
        float *dst = x + (size_t)i*D;
        if (tok == m->audio_token_id && audio_emb && ai < audio_n) {
            memcpy(dst, audio_emb + (size_t)ai*D, (size_t)D*sizeof(float));
            ai++;
        } else if (t->embed_ggml) {
            int rb = gq_row_bytes(t->embed_ggml_type, D);
            gq_dequant_row(t->embed_ggml_type, (const uint8_t*)t->embed_ggml + (size_t)tok*rb, dst, D);
        } else {
            const uint16_t *src = t->embed + (size_t)tok*D;
            for (int d=0; d<D; d++) dst[d] = bf16w_to_f32(src[d]);
        }
    }
    const char *dump_layers = getenv("QWEN3_ASR_DUMP_LAYERS");
    if (dump_layers && dump_layers[0]) { FILE *df=fopen(dump_layers,"ab"); if(df){fwrite(x+(size_t)(S-1)*D,sizeof(float),D,df); fclose(df);} }
    float *xb  = (float*)xcalloc((size_t)S*D,sizeof(float));
    float *q   = (float*)xcalloc((size_t)S*H*HD,sizeof(float));
    float *k   = (float*)xcalloc((size_t)S*KVH*HD,sizeof(float));
    float *v   = (float*)xcalloc((size_t)S*KVH*HD,sizeof(float));
    float *xb2 = (float*)xcalloc((size_t)S*D,sizeof(float));
    float *ff1 = (float*)xcalloc((size_t)S*t->inter,sizeof(float));
    float *ff2 = (float*)xcalloc((size_t)S*t->inter,sizeof(float));
    float *att = (float*)xcalloc((size_t)H*(start_pos+S),sizeof(float));
    for (int li=0; li<t->n_layers; li++) {
        TextLayer *L = &t->layers[li];
        const char *dbg_l0 = getenv("QWEN3_ASR_DEBUG_LAYER0");
        int do_l0 = (li==0 && dbg_l0 && dbg_l0[0] && S>1);
        /* batch rmsnorm + QKV projections */
        for (int si=0; si<S; si++) rmsnorm(xb + (size_t)si*D, x + (size_t)si*D, L->ln1.w, D, t->rms_eps);
        if (do_l0) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(xb+(size_t)(S-1)*D,sizeof(float),D,df); fclose(df);} }
        matmul_linear(q, xb, &L->q, S);
        matmul_linear(k, xb, &L->k, S);
        matmul_linear(v, xb, &L->v, S);
        if (do_l0) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(q+(size_t)(S-1)*H*HD,sizeof(float),H*HD,df); fwrite(k+(size_t)(S-1)*KVH*HD,sizeof(float),KVH*HD,df); fwrite(v+(size_t)(S-1)*KVH*HD,sizeof(float),KVH*HD,df); fclose(df);} }
        /* per-token q/k norm, rope, kv store, attention */
        for (int si=0; si<S; si++) {
            int pos = start_pos + si;
            float *qs = q + (size_t)si*H*HD;
            float *ks = k + (size_t)si*KVH*HD;
            float *vs = v + (size_t)si*KVH*HD;
            if (L->q_norm) for(int h=0;h<H;h++){ float ss=0; for(int d=0;d<HD;d++){float vv=qs[h*HD+d]; ss+=vv*vv;} ss=1.0f/sqrtf(ss/HD+1e-6f); for(int d=0;d<HD;d++) qs[h*HD+d]=qs[h*HD+d]*ss*L->q_norm[d]; }
            if (L->k_norm) for(int h=0;h<KVH;h++){ float ss=0; for(int d=0;d<HD;d++){float vv=ks[h*HD+d]; ss+=vv*vv;} ss=1.0f/sqrtf(ss/HD+1e-6f); for(int d=0;d<HD;d++) ks[h*HD+d]=ks[h*HD+d]*ss*L->k_norm[d]; }
            if (do_l0 && si==S-1) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(qs,sizeof(float),H*HD,df); fwrite(ks,sizeof(float),KVH*HD,df); fclose(df);} }
            int mpos[3] = {pos, pos, pos};
            mrope(qs, H, HD, mpos, t->rope_theta, t->mrope_section);
            mrope(ks, KVH, HD, mpos, t->rope_theta, t->mrope_section);
            if (do_l0 && si==S-1) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(qs,sizeof(float),H*HD,df); fwrite(ks,sizeof(float),KVH*HD,df); fclose(df);} }
            KVCache *kc = &cache[li];
            if (pos < kc->cap) {
                memcpy(kc->k + (size_t)pos*KVdim, ks, (size_t)KVdim*sizeof(float));
                memcpy(kc->v + (size_t)pos*KVdim, vs, (size_t)KVdim*sizeof(float));
                if (kc->len <= pos) kc->len = pos+1;
            }
            int kv_len = kc->len;
            int gqa = H / KVH;
            float *out_t = xb2 + (size_t)si*D;
            memset(out_t, 0, (size_t)D*sizeof(float));
            for (int h=0; h<H; h++) {
                int kvh = h / gqa;
                float maxv = -1e30f;
                for (int tpos=0; tpos<kv_len; tpos++) {
                    float dot=0.0f;
                    const float *kk = kc->k + (size_t)tpos*KVdim + kvh*HD;
                    for (int d=0;d<HD;d++) dot += qs[h*HD+d]*kk[d];
                    dot /= sqrtf((float)HD);
                    att[h*kv_len+tpos]=dot; if(dot>maxv) maxv=dot;
                }
                float sum=0.0f;
                for(int tpos=0;tpos<kv_len;tpos++){ att[h*kv_len+tpos]=expf(att[h*kv_len+tpos]-maxv); sum+=att[h*kv_len+tpos]; }
                float inv_sum = sum>0 ? 1.0f/sum : 1.0f;
                for(int tpos=0;tpos<kv_len;tpos++){
                    float wgt = att[h*kv_len+tpos] * inv_sum;
                    const float *vvv = kc->v + (size_t)tpos*KVdim + kvh*HD;
                    for(int d=0;d<HD;d++) out_t[h*HD+d] += wgt*vvv[d];
                }
            }
        }
        if (do_l0) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(xb2+(size_t)(S-1)*D,sizeof(float),D,df); fclose(df);} }
        /* batch output proj + residual */
        matmul_linear(xb, xb2, &L->o, S);
        for (size_t i=0;i<(size_t)S*D;i++) x[i] += xb[i];
        if (do_l0) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(x+(size_t)(S-1)*D,sizeof(float),D,df); fclose(df);} }
        /* batch ffn */
        for (int si=0; si<S; si++) rmsnorm(xb + (size_t)si*D, x + (size_t)si*D, L->ln2.w, D, t->rms_eps);
        if (do_l0) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(xb+(size_t)(S-1)*D,sizeof(float),D,df); fclose(df);} }
        matmul_linear(ff1, xb, &L->gate, S);
        matmul_linear(ff2, xb, &L->up, S);
        if (do_l0) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(ff1+(size_t)(S-1)*t->inter,sizeof(float),t->inter,df); fwrite(ff2+(size_t)(S-1)*t->inter,sizeof(float),t->inter,df); fclose(df);} }
        for (size_t i=0;i<(size_t)S*t->inter;i++) ff1[i] = silu(ff1[i]) * ff2[i];
        matmul_linear(xb, ff1, &L->down, S);
        for (size_t i=0;i<(size_t)S*D;i++) x[i] += xb[i];
        if (do_l0) { FILE *df=fopen(dbg_l0,"ab"); if(df){fwrite(x+(size_t)(S-1)*D,sizeof(float),D,df); fclose(df);} }
        if (dump_layers && dump_layers[0]) { FILE *df=fopen(dump_layers,"ab"); if(df){fwrite(x+(size_t)(S-1)*D,sizeof(float),D,df); fclose(df);} }
    }
    /* logits for last token */
    float *last = x + (size_t)(S-1)*D;
    rmsnorm(xb, last, t->final_norm, D, t->rms_eps);
    const char *dump_lh = getenv("QWEN3_ASR_DUMP_LAST_HIDDEN");
    if (dump_lh && dump_lh[0]) { FILE *df=fopen(dump_lh,"ab"); if(df){fwrite(xb,sizeof(float),D,df); fclose(df);} }
    if (t->lm_head_ggml) {
        ColiTensorView lv; memset(&lv,0,sizeof(lv));
        lv.format = COLI_TENSOR_GGML; lv.data = t->lm_head_ggml;
        lv.rows = t->vocab; lv.columns = D; lv.block_columns = (uint32_t)t->lm_head_ggml_type;
        coli_ggml_matmul(logits_out, &lv, xb, 1);
    } else {
        #pragma omp parallel for schedule(static)
        for (int vix=0; vix<t->vocab; vix++) {
            const uint16_t *rw = t->lm_head + (size_t)vix*D;
            logits_out[vix] = dot_bf16w_lanes(xb, rw, D);
        }
    }
    free(x); free(xb); free(q); free(k); free(v); free(xb2); free(ff1); free(ff2); free(att);
    if (getenv("QWEN3_ASR_DEBUG_TIMING")) fprintf(stderr,"[qwen3_asr] text_forward S=%d start=%d ms=%.1f\n", S, start_pos, now_ms()-t_tf0);
    return 0;
}

static int b64_val(char c){ if(c>='A'&&c<='Z')return c-'A'; if(c>='a'&&c<='z')return c-'a'+26; if(c>='0'&&c<='9')return c-'0'+52; if(c=='+')return 62; if(c=='/')return 63; return -1; }
static unsigned char *b64_decode(const char *s, size_t *out_n){
    size_t n=strlen(s); unsigned char *out=(unsigned char*)xmalloc(n*3/4+4); size_t o=0; int val=0, valb=-8;
    for(size_t i=0;i<n;i++){ unsigned char c=(unsigned char)s[i]; if(c=='=') break; int d=b64_val((char)c); if(d<0) continue; val=(val<<6)+d; valb+=6; if(valb>=0){ out[o++]=(unsigned char)((val>>valb)&0xFF); valb-=8; } }
    *out_n=o; return out;
}

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0+ts.tv_nsec/1e6; }

static int greedy_decode(Qwen3Asr *m, const int *prompt, int n_prompt, const float *audio_emb, int audio_n, char *out, size_t out_cap, int max_new) {
    TextModel *t = &m->text;
    int max_seq = n_prompt + max_new + 8;
    KVCache *cache = kv_alloc(t->n_layers, max_seq, t->n_kv_heads*t->head_dim);
    float *logits = (float*)xmalloc((size_t)t->vocab*sizeof(float));
    int *seq = (int*)xmalloc((size_t)max_seq*sizeof(int));
    memcpy(seq, prompt, (size_t)n_prompt*sizeof(int));
    int n = n_prompt;
    int eos = tok_id_of(&m->tok, "<|im_end|>");
    if (eos < 0) eos = 151645;
    double t0=now_ms();
    text_forward(m, seq, audio_emb, audio_n, n, 0, cache, logits);
    double t_prefill=now_ms();
    int out_len = 0; out[0]=0;
    int steps=0;
    const char *dbg_topk = getenv("QWEN3_ASR_DEBUG_TOPK");
    int dbg_topk_n = dbg_topk ? atoi(dbg_topk) : 0;
    for (int step=0; step<max_new; step++) {
        int best=0; float bestv=logits[0];
        for(int i=1;i<t->vocab;i++) if(logits[i]>bestv){bestv=logits[i]; best=i;}
        if (dbg_topk_n > 0 && step < dbg_topk_n) {
            int kN = 5;
            int topi[5]={0,0,0,0,0}; float topv[5]={-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for(int i=0;i<t->vocab;i++){float v=logits[i]; for(int k2=0;k2<kN;k2++){if(v>topv[k2]){for(int j=kN-1;j>k2;j--){topv[j]=topv[j-1];topi[j]=topi[j-1];} topv[k2]=v; topi[k2]=i; break;}}}
            fprintf(stderr,"[qwen3_asr] topk step=%d", step);
            for(int k2=0;k2<kN;k2++){char d2[64];int dn2=tok_decode(&m->tok,&topi[k2],1,d2,63);d2[dn2>0?dn2:0]=0; fprintf(stderr," %d:%.4f:%s", topi[k2], topv[k2], d2);} 
            fprintf(stderr,"\n"); fflush(stderr);
        }
        if (best == eos) break;
        if (n < max_seq) seq[n++] = best;
        char piece[256];
        int pn = tok_decode(&m->tok, &best, 1, piece, (int)sizeof(piece)-1);
        if (pn > 0) piece[pn]=0; else piece[0]=0;
        if (pn > 0) {
            if (out_len + (size_t)pn + 1 < out_cap) { memcpy(out+out_len, piece, (size_t)pn); out_len += pn; out[out_len]=0; }
        }
        text_forward(m, &best, NULL, 0, 1, n-1, cache, logits);
        steps++;
    }
    double t_decode=now_ms();
    fprintf(stderr, "[qwen3_asr] prefill=%.1fms decode=%.1fms steps=%d prompt=%d audio=%d\n", t_prefill-t0, t_decode-t_prefill, steps, n_prompt, audio_n); fflush(stderr);
    free(seq); free(logits); kv_free(cache, t->n_layers);
    return (int)out_len;
}

static int transcribe_audio(Qwen3Asr *m, const unsigned char *wav, size_t wav_n, char *text_out, size_t cap) {
    if (!m->loaded) return -1;
    double t0=now_ms();
    Q3AsrWav w={0}, r={0};
    int rc = q3asr_read_wav_bytes(wav, wav_n, &w);
    if (rc != 0) { if (cap) text_out[0]=0; return -2; }
    rc = q3asr_resample_16k(&w, &r);
    q3asr_wav_free(&w);
    if (rc != 0) { if (cap) text_out[0]=0; return -3; }
    Q3AsrMel mel={0};
    rc = q3asr_log_mel(r.samples, r.n, 16000, 400, 160, m->audio.n_mels, &mel);
    q3asr_wav_free(&r);
    if (rc != 0) { if (cap) text_out[0]=0; return -4; }
    double t1=now_ms();
    float *audio_emb=NULL; int audio_n=0;
    const char *emb_path = getenv("QWEN3_ASR_AUDIO_EMB");
    if (emb_path && emb_path[0]) {
        FILE *ef = fopen(emb_path, "rb");
        if (ef) {
            fseek(ef, 0, SEEK_END); long sz = ftell(ef); fseek(ef, 0, SEEK_SET);
            int dim = m->audio.output_dim;
            audio_n = (int)(sz / (long)(sizeof(float)*dim));
            audio_emb = (float*)xmalloc((size_t)audio_n*dim*sizeof(float));
            fread(audio_emb, sizeof(float), (size_t)audio_n*dim, ef); fclose(ef);
            rc = 0;
            fprintf(stderr, "[qwen3_asr] loaded audio_emb override %s (%d tokens)\n", emb_path, audio_n);
        } else { rc = audio_encoder_forward(m, mel.mel, mel.n_mels, mel.n_frames, &audio_emb, &audio_n); }
    } else {
        rc = audio_encoder_forward(m, mel.mel, mel.n_mels, mel.n_frames, &audio_emb, &audio_n);
    }
    const char *dump_dir = getenv("QWEN3_ASR_DUMP");
    if (dump_dir && dump_dir[0]) {
        char dp[4096]; FILE *df;
        snprintf(dp,sizeof(dp),"%s/mel.f32",dump_dir); df=fopen(dp,"wb"); if(df){fwrite(mel.mel,sizeof(float),(size_t)mel.n_mels*mel.n_frames,df); fclose(df);}
        snprintf(dp,sizeof(dp),"%s/mel_meta.json",dump_dir); df=fopen(dp,"w"); if(df){fprintf(df,"{\"n_mels\":%d,\"n_frames\":%d}\n",mel.n_mels,mel.n_frames); fclose(df);}
        if (rc==0 && audio_emb) {
            snprintf(dp,sizeof(dp),"%s/audio_emb.f32",dump_dir); df=fopen(dp,"wb"); if(df){fwrite(audio_emb,sizeof(float),(size_t)audio_n*m->audio.output_dim,df); fclose(df);}
            snprintf(dp,sizeof(dp),"%s/audio_emb_meta.json",dump_dir); df=fopen(dp,"w"); if(df){fprintf(df,"{\"n_tokens\":%d,\"dim\":%d}\n",audio_n,m->audio.output_dim); fclose(df);}
        }
    }
    q3asr_mel_free(&mel);
    if (rc != 0) { if (cap) text_out[0]=0; return -5; }
    double t2=now_ms();
    fprintf(stderr, "[qwen3_asr] mel=%.1fms audio_enc=%.1fms audio_tokens=%d\n", t1-t0, t2-t1, audio_n); fflush(stderr);
    /* build prompt - Qwen3-ASR chat template */
    int prompt[4096]; int n=0;
    int t_im_start = tok_id_of(&m->tok, "<|im_start|>");
    int t_im_end = tok_id_of(&m->tok, "<|im_end|>");
    int t_audio_start = tok_id_of(&m->tok, "<|audio_start|>");
    int t_audio_end = tok_id_of(&m->tok, "<|audio_end|>");
    /* Fallback known Qwen3-ASR token IDs */
    if (t_im_start < 0) t_im_start = 151644;
    if (t_im_end < 0) t_im_end = 151645;
    if (t_audio_start < 0) t_audio_start = 151669;
    if (t_audio_end < 0) t_audio_end = 151670;
    /* Use known tokenization for prompt skeleton to avoid tok_encode hang on Qwen tokenizer. */
    /* system\n = [8948,198], \n=[198], user\n=[872,198], assistant\n=[77091,198] */
    prompt[n++]=t_im_start; prompt[n++]=8948; prompt[n++]=198; prompt[n++]=t_im_end;
    prompt[n++]=198;
    prompt[n++]=t_im_start; prompt[n++]=872; prompt[n++]=198;
    prompt[n++]=t_audio_start;
    for(int i=0;i<audio_n && n<4096;i++) prompt[n++]=m->audio_token_id;
    prompt[n++]=t_audio_end;
    prompt[n++]=t_im_end;
    prompt[n++]=198;
    prompt[n++]=t_im_start; prompt[n++]=77091; prompt[n++]=198;
    /* Default prefill: force English transcription, matching HF apply_transcription_request(language='English'). */
    { int prefill[] = {11528, 6364, 151704}; /* "language English<asr_text>" */
      for (int pi=0; pi<3 && n<4096; pi++) prompt[n++]=prefill[pi]; }
    int gen = greedy_decode(m, prompt, n, audio_emb, audio_n, text_out, cap, 256);
    /* Strip Qwen3-ASR control prefix: "language ...<asr_text>" */
    char *asr = strstr(text_out, "<asr_text>");
    if (asr) {
        asr += strlen("<asr_text>");
        memmove(text_out, asr, strlen(asr) + 1);
    }
    /* Trim leading whitespace */
    size_t off = 0; while (text_out[off]==' '||text_out[off]=='\n'||text_out[off]=='\r'||text_out[off]=='\t') off++;
    if (off) memmove(text_out, text_out+off, strlen(text_out+off)+1);
    free(audio_emb);
    return gen >= 0 ? 0 : -6;
}

static int json_extract_string(const char *json, const char *key, char *out, size_t cap) {
    char pat[128]; snprintf(pat, sizeof(pat), "\"%s\":\"", key);
    const char *p = strstr(json, pat); if (!p) return 0; p += strlen(pat);
    size_t o=0; while (*p && *p!='"' && o+1<cap) { if (*p=='\\' && p[1]) { p++; if(*p=='n') out[o++]='\n'; else if(*p=='t') out[o++]='\t'; else if(*p=='r') out[o++]='\r'; else out[o++]=*p; p++; } else out[o++]=*p++; }
    out[o]=0; return 1;
}

static void handle_transcribe(Qwen3Asr *m, FILE *out, const ColiServeCommand *cmd) {
    if (!m->loaded) {
        emit_error(out, cmd->id, "MODEL_NOT_LOADED", "qwen3_asr model not loaded; start with --model DIR --serve");
        return;
    }
    if (!cmd->payload || cmd->payload_bytes == 0) {
        emit_error(out, cmd->id, "BAD_REQUEST", "missing transcription payload");
        return;
    }
    char *json = (char*)xmalloc((size_t)cmd->payload_bytes + 1);
    memcpy(json, cmd->payload, (size_t)cmd->payload_bytes); json[cmd->payload_bytes]=0;
    char *key = strstr(json, "\"audio_b64\"");
    if (!key) { free(json); emit_error(out, cmd->id, "BAD_REQUEST", "missing audio_b64"); return; }
    key = strchr(key + strlen("\"audio_b64\""), ':');
    if (!key) { free(json); emit_error(out, cmd->id, "BAD_REQUEST", "bad audio_b64"); return; }
    key++;
    while (*key == ' ' || *key == '\t' || *key == '\r' || *key == '\n') key++;
    if (*key != '"') { free(json); emit_error(out, cmd->id, "BAD_REQUEST", "bad audio_b64"); return; }
    key++;
    char *end = strchr(key, '"');
    if (!end) { free(json); emit_error(out, cmd->id, "BAD_REQUEST", "bad audio_b64"); return; }
    size_t b64n = (size_t)(end-key);
    char *b64 = (char*)xmalloc(b64n+1); memcpy(b64, key, b64n); b64[b64n]=0;
    size_t wav_n=0; unsigned char *wav = b64_decode(b64, &wav_n); free(b64);
    char text[16384]; text[0]=0;
    int rc = transcribe_audio(m, wav, wav_n, text, sizeof(text));
    free(wav); free(json);
    if (rc != 0) { char msg[128]; snprintf(msg,sizeof(msg),"transcription failed (%d)",rc); emit_error(out, cmd->id, "TRANSCRIBE_FAILED", msg); return; }
    emit_transcript(out, cmd->id, text);
}

static void serve_loop(Qwen3Asr *m) {
    ColiServeWireProfile profile;
    memset(&profile, 0, sizeof(profile));
    profile.max_header_bytes = 4096;
    profile.max_payload_bytes = 64ULL << 20;
    profile.max_extension_bytes = 64ULL << 20;
    profile.allow_extension_bytes = 1;
    profile.allow_prefix_hint = 1;
    for (;;) {
        ColiServeCommand cmd; memset(&cmd,0,sizeof(cmd));
        ColiServeReadResult rr = coli_serve_read_command_alloc(stdin, &profile, &cmd, malloc);
        if (rr == COLI_SERVE_READ_EOF) break;
        if (rr == COLI_SERVE_READ_IGNORED) { coli_serve_command_dispose(&cmd); continue; }
        if (rr != COLI_SERVE_READ_OK) { if (cmd.id[0]) emit_error(stdout, cmd.id, "BAD_REQUEST", "invalid command"); coli_serve_command_dispose(&cmd); continue; }
        if (cmd.kind == COLI_SERVE_COMMAND_TRANSCRIBE) handle_transcribe(m, stdout, &cmd);
        else if (cmd.kind == COLI_SERVE_COMMAND_STOP || cmd.kind == COLI_SERVE_COMMAND_CANCEL) { /* no-op */ }
        else if (cmd.id[0]) emit_error(stdout, cmd.id, "UNSUPPORTED", "command not supported");
        coli_serve_command_dispose(&cmd);
    }
}

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s --model DIR --serve\n", argv0);
}

int main(int argc, char **argv) {
    Qwen3Asr m; memset(&m, 0, sizeof(m));
    int serve = 0;
    const char *snap = getenv("SNAP");
    const char *serve_env = getenv("SERVE");
    if (snap && !m.model_dir[0]) strncpy(m.model_dir, snap, sizeof(m.model_dir)-1);
    if (serve_env && !strcmp(serve_env, "1")) serve = 1;
    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--model") && i+1<argc) { strncpy(m.model_dir, argv[++i], sizeof(m.model_dir)-1); }
        else if (!strcmp(argv[i],"--gguf") && i+1<argc) { strncpy(m.gguf_path, argv[++i], sizeof(m.gguf_path)-1); }
        else if (!strcmp(argv[i],"--mmproj") && i+1<argc) { strncpy(m.mmproj_path, argv[++i], sizeof(m.mmproj_path)-1); }
        else if (!strcmp(argv[i],"--serve")) serve=1;
        else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")) { usage(argv[0]); return 0; }
        else if (argv[i][0] != '-') { /* positional cap or model */ serve = 1; }
    }
    if (!serve) { usage(argv[0]); return 2; }
    if (m.model_dir[0] || m.gguf_path[0]) {
        if (load_model(&m) != 0) { fprintf(stderr, "qwen3_asr: failed to load model\n"); return 1; }
    }
    write_all(stdout, "\x01\x01READY\x01\x01\n", 10);
    write_all(stdout, "CAPS transcribe=1 chat=0\n", 25);
    char stat[128]; snprintf(stat,sizeof(stat),"STAT 0 0.0 0.0 0.0 0 0\n");
    write_all(stdout, stat, strlen(stat));
    fflush(stdout);
    serve_loop(&m);
    return 0;
}
