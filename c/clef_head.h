/* clef_head.h -- Clef (Cloudflare, Apache-2.0): a joint schema head over a
 * decoder's final hidden states, and the rendering of a DECIDE record into
 * that decoder's input. docs/clef.md.
 *
 * Clef is Qwen3.8-27B, post-trained, plus a small transformer that reads the
 * backbone's last hidden states and scores every option of every question in
 * one pass. The specification is the checkpoint's own joint_schema_model.py;
 * each step below names the function it follows:
 *
 *   rendering    encode_record        clef_render(): the chat-shaped prompt,
 *                                     the state, then one FIELD per question
 *                                     with its options as JSON; every piece
 *                                     tokenized on its own, the state cut from
 *                                     the end to fit max_length
 *   head         JointSchemaHead      clef_head_forward(): LayerNorm over the
 *                                     hidden states, span means for questions
 *                                     and options, lexical option vectors from
 *                                     the LM head's rows, two evidence routing
 *                                     layers, four decoder layers over the
 *                                     fields, the prior + gated joint score
 *   answer       systemone            a softmax per question, nothing fitted
 *                                     on top (Clef ships no temperature)
 *
 * The engine owns the backbone: it renders with clef_render() through its own
 * tokenizer, runs the prompt, hands the final-normed hidden state of every
 * position to clef_head_forward(), and reads the logits back in the record's
 * option order. Everything here is f32; the reference runs in bf16.
 *
 * Needs decide_serve.h (the record) and matmul_f32.h (the one GEMM). */
#ifndef COLI_CLEF_HEAD_H
#define COLI_CLEF_HEAD_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "decide_serve.h"
#include "matmul_f32.h"

/* encode_record(max_length=16384), the default systemone() calls it with */
#define CLEF_MAX_LENGTH 16384

/* joint_schema_model.SYSTEM_PROMPT and the fixed pieces of encode_record */
#define CLEF_SYSTEM_PROMPT "Read the complete state and schema. Decide every field jointly. " \
                           "Each answer must be exactly one of that field's allowed options."
#define CLEF_PREFIX "<|im_start|>system\n" CLEF_SYSTEM_PROMPT "<|im_end|>\n<|im_start|>user\nSTATE:\n"
#define CLEF_SUFFIX "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
#define CLEF_NOUL_TRUE  "The proposition is true or the answer is yes."
#define CLEF_NOUL_FALSE "The proposition is false or the answer is no."

/* joint_schema_model.QUESTION_TYPES */
enum { CLEF_NOUL = 0, CLEF_CHOICE = 1, CLEF_SCORE = 2 };

/* ------------------------------------------------------------ rendering */

/* text -> token ids, the engine's tokenizer (added tokens matched, as
 * tokenizer(text, add_special_tokens=False) does). Returns 0 on failure. */
typedef int (*ClefEncodeFn)(void *ctx, const char *text, int **ids, int *count);

typedef struct {
    int type;            /* CLEF_NOUL | CLEF_CHOICE | CLEF_SCORE */
    int qs, qe;          /* the instruction's tokens, [qs, qe) */
    int n;               /* options */
    int *os, *oe;        /* each option's JSON, [os, oe), in the reference's order */
    int *record_index;   /* reference option k is the record's option record_index[k] */
} ClefQuestion;

typedef struct {
    int *ids, n;                     /* the model input */
    ClefQuestion *q; int n_q;        /* in the record's (= the request's) order */
    int state_tokens, state_used;    /* the state's tokens before and after the cut */
} ClefInput;

static void clef_input_free(ClefInput *in)
{
    if (!in) return;
    for (int i = 0; i < in->n_q; i++) {
        free(in->q[i].os); free(in->q[i].oe); free(in->q[i].record_index);
    }
    free(in->q); free(in->ids);
    memset(in, 0, sizeof(*in));
}

typedef struct { int *v; int n, cap, failed; } ClefIds;

static void clef_ids_push(ClefIds *a, const int *v, int n)
{
    if (a->failed || n <= 0) return;
    if (a->n + n > a->cap) {
        int cap = a->cap ? a->cap : 256;
        while (cap < a->n + n) cap *= 2;
        int *grown = (int *)realloc(a->v, (size_t)cap * sizeof(int));
        if (!grown) { a->failed = 1; return; }
        a->v = grown; a->cap = cap;
    }
    memcpy(a->v + a->n, v, (size_t)n * sizeof(int));
    a->n += n;
}

/* _tokens(tokenizer, text): one piece, tokenized on its own and appended */
static int clef_tokens(ClefIds *out, ClefEncodeFn encode, void *ctx, const char *text)
{
    int *ids = NULL, n = 0;
    if (!encode(ctx, text, &ids, &n)) return 0;
    clef_ids_push(out, ids, n);
    free(ids);
    return !out->failed;
}

/* json.dumps(text, ensure_ascii=False) of a string: Python's escapes (\" \\ \b
 * \f \n \r \t, \u00XX for the other control characters, lowercase hex),
 * everything else as it is. */
static void clef_json_string(DecideBuf *b, const char *text)
{
    decide_buf_put(b, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        const char *esc = NULL;
        switch (*p) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        default: break;
        }
        if (esc) decide_buf_put(b, esc, 2);
        else if (*p < 0x20) decide_buf_printf(b, "\\u%04x", *p);
        else decide_buf_put(b, (const char *)p, 1);
    }
    decide_buf_put(b, "\"", 1);
}

/* render({"option_id": id, "description": description}): sort_keys puts the
 * description first; a description of None is left out. */
static char *clef_option_json(const DecideQuestion *q, const DecideOption *o)
{
    DecideBuf b = {0};
    const char *text = o->text;
    int is_json = o->text_json;
    if (q->type == DECIDE_NOUL && !o->has_text) {
        /* question_options(): the defaults, updated by what the caller gave */
        text = !strcmp(o->label, "true") ? CLEF_NOUL_TRUE : CLEF_NOUL_FALSE;
        is_json = 0;
    }
    decide_buf_put(&b, "{", 1);
    if (text) {
        decide_buf_put(&b, "\"description\":", 14);
        if (is_json) decide_buf_put(&b, text, strlen(text));
        else clef_json_string(&b, text);
        decide_buf_put(&b, ",", 1);
    }
    decide_buf_put(&b, "\"option_id\":", 12);
    clef_json_string(&b, o->label);
    decide_buf_put(&b, "}", 1);
    if (b.failed) { free(b.data); return NULL; }
    return b.data;
}

static int clef_cmp_labels(const void *a, const void *b, void *labels)
{
    const DecideOption *o = (const DecideOption *)labels;
    return strcmp(o[*(const int *)a].label, o[*(const int *)b].label);
}

/* A small insertion sort with a context: qsort_r is not portable, and a
 * question has at most 255 options. sorted() on str compares code points,
 * which strcmp on UTF-8 does too. */
static void clef_sort_options(int *order, int n, const DecideOption *options)
{
    for (int i = 1; i < n; i++) {
        int v = order[i], j = i - 1;
        while (j >= 0 && clef_cmp_labels(&order[j], &v, (void *)options) > 0) { order[j + 1] = order[j]; j--; }
        order[j + 1] = v;
    }
}

/* encode_record(tokenizer, record, max_length): the record must be in the raw
 * form (the caller's own values; decide_serve.h). Returns 1, or 0 with `err`
 * starting with the field ("questions: ...") when the record does not fit. */
static int clef_render(const DecideRecord *rec, ClefEncodeFn encode, void *ctx, int max_length,
                       ClefInput *out, char *err, size_t cap)
{
    memset(out, 0, sizeof(*out));
    if (!rec->raw)
        return decide_fail(err, cap, "record: Clef reads the raw form (CAPS decide_record=raw), "
                                     "this one is the default form");
    ClefIds schema = {0}, prefix = {0}, state = {0}, suffix = {0};
    out->q = (ClefQuestion *)calloc((size_t)rec->n_questions, sizeof(ClefQuestion));
    if (!out->q) return decide_fail(err, cap, "out of memory");
    out->n_q = rec->n_questions;
    int ok = clef_tokens(&schema, encode, ctx, "\n\nSCHEMA FIELDS:\n");
    char line[64];
    for (int i = 0; ok && i < rec->n_questions; i++) {
        const DecideQuestion *q = &rec->questions[i];
        ClefQuestion *cq = &out->q[i];
        cq->type = q->type == DECIDE_NOUL ? CLEF_NOUL : q->type == DECIDE_CHOICE ? CLEF_CHOICE : CLEF_SCORE;
        DecideBuf head = {0};
        decide_buf_printf(&head, "\nFIELD %d\nID: ", i + 1);
        decide_buf_put(&head, q->id, strlen(q->id));
        decide_buf_printf(&head, "\nTYPE: %s\nINSTRUCTION: ", decide_type_name(q->type));
        ok = !head.failed && clef_tokens(&schema, encode, ctx, head.data);
        free(head.data);
        if (!ok) break;
        cq->qs = schema.n;
        /* instructions None or "": the question id */
        ok = clef_tokens(&schema, encode, ctx,
                         q->has_instructions && q->instructions[0] ? q->instructions : q->id);
        cq->qe = schema.n;
        ok = ok && clef_tokens(&schema, encode, ctx, "\nALLOWED OPTIONS:\n");
        if (!ok) break;
        cq->n = q->n_options;
        cq->os = (int *)calloc((size_t)cq->n, sizeof(int));
        cq->oe = (int *)calloc((size_t)cq->n, sizeof(int));
        cq->record_index = (int *)calloc((size_t)cq->n, sizeof(int));
        if (!cq->os || !cq->oe || !cq->record_index) { ok = 0; break; }
        /* question_options(): noul true then false (the record has false, true),
         * choice sorted by label, score in level order */
        for (int k = 0; k < cq->n; k++) cq->record_index[k] = k;
        if (q->type == DECIDE_NOUL) { cq->record_index[0] = 1; cq->record_index[1] = 0; }
        else if (q->type == DECIDE_CHOICE) clef_sort_options(cq->record_index, cq->n, q->options);
        for (int k = 0; ok && k < cq->n; k++) {
            snprintf(line, sizeof(line), "OPTION %d: ", k + 1);
            ok = clef_tokens(&schema, encode, ctx, line);
            char *json = ok ? clef_option_json(q, &q->options[cq->record_index[k]]) : NULL;
            cq->os[k] = schema.n;
            ok = json && clef_tokens(&schema, encode, ctx, json);
            cq->oe[k] = schema.n;
            free(json);
            ok = ok && clef_tokens(&schema, encode, ctx, "\n");
            if (ok && cq->oe[k] <= cq->os[k]) ok = 0;
        }
        ok = ok && clef_tokens(&schema, encode, ctx, "END FIELD\n");
        if (ok && cq->qe <= cq->qs) ok = 0;
    }
    ok = ok && clef_tokens(&prefix, encode, ctx, CLEF_PREFIX) &&
         clef_tokens(&suffix, encode, ctx, CLEF_SUFFIX) &&
         clef_tokens(&state, encode, ctx, rec->state);
    if (!ok) {
        free(schema.v); free(prefix.v); free(state.v); free(suffix.v);
        clef_input_free(out);
        return decide_fail(err, cap, "the record could not be tokenized");
    }
    int fixed = prefix.n + schema.n + suffix.n;
    if (fixed > max_length) {
        free(schema.v); free(prefix.v); free(state.v); free(suffix.v);
        clef_input_free(out);
        return decide_fail(err, cap, "questions: the schema requires %d tokens before the state; "
                                     "the maximum is %d", fixed, max_length);
    }
    out->state_tokens = state.n;
    out->state_used = state.n < max_length - fixed ? state.n : max_length - fixed;
    int offset = prefix.n + out->state_used;
    ClefIds all = {0};
    clef_ids_push(&all, prefix.v, prefix.n);
    clef_ids_push(&all, state.v, out->state_used);
    clef_ids_push(&all, schema.v, schema.n);
    clef_ids_push(&all, suffix.v, suffix.n);
    free(schema.v); free(prefix.v); free(state.v); free(suffix.v);
    if (all.failed) { free(all.v); clef_input_free(out); return decide_fail(err, cap, "out of memory"); }
    for (int i = 0; i < out->n_q; i++) {
        ClefQuestion *cq = &out->q[i];
        cq->qs += offset; cq->qe += offset;
        for (int k = 0; k < cq->n; k++) { cq->os[k] += offset; cq->oe[k] += offset; }
    }
    out->ids = all.v;
    out->n = all.n;
    return 1;
}

/* ---------------------------------------------------------------- the head */

typedef struct { float *w, *b; } ClefNorm;                 /* nn.LayerNorm, eps 1e-5 */
typedef struct { float *in_w, *in_b, *out_w, *out_b; } ClefAttn;   /* nn.MultiheadAttention */

typedef struct {               /* EvidenceRoutingLayer */
    ClefNorm query_norm, memory_norm, ff_norm;
    ClefAttn attn;
    float *ff1_w, *ff1_b, *ff2_w, *ff2_b;
} ClefRoute;

typedef struct {               /* nn.TransformerDecoderLayer(norm_first=True, gelu) */
    ClefAttn self_attn, cross_attn;
    ClefNorm norm1, norm2, norm3;
    float *ff1_w, *ff1_b, *ff2_w, *ff2_b;
} ClefDecoder;

typedef struct {
    int hidden, width, routing_layers, layers, heads, feedforward;
    ClefNorm hidden_norm;
    float *memory_proj, *question_proj, *option_question_proj, *global_proj;
    float *option_context_proj, *option_lexical_proj;      /* [width, hidden] */
    float *type_embedding;                                  /* [3, width] */
    ClefRoute *route;
    ClefNorm option_summary_norm, field_norm, option_norm;
    ClefDecoder *dec;
    float *score1_w, *score1_b, *score2_w, *score2_b;       /* [width, 4 width], [1, width] */
    float prior_logit_scale, joint_logit_scale, residual_gate;
} ClefHead;

/* A tensor of the head as f32, `numel` elements, or NULL when it is missing or
 * the wrong size (the caller refuses the checkpoint). */
typedef float *(*ClefLoadFn)(void *ctx, const char *name, int64_t numel);

typedef struct { void *ctx; ClefLoadFn load; int missing; char first[160]; } ClefLoader;

static float *clef_take(ClefLoader *l, const char *name, int64_t numel)
{
    float *p = l->load(l->ctx, name, numel);
    if (!p && !l->missing++) snprintf(l->first, sizeof(l->first), "%s", name);
    return p;
}

static void clef_take_norm(ClefLoader *l, ClefNorm *n, const char *stem, int d)
{
    char name[160];
    snprintf(name, sizeof(name), "%s.weight", stem); n->w = clef_take(l, name, d);
    snprintf(name, sizeof(name), "%s.bias", stem);   n->b = clef_take(l, name, d);
}

static void clef_take_attn(ClefLoader *l, ClefAttn *a, const char *stem, int d)
{
    char name[160];
    snprintf(name, sizeof(name), "%s.in_proj_weight", stem);  a->in_w = clef_take(l, name, (int64_t)3 * d * d);
    snprintf(name, sizeof(name), "%s.in_proj_bias", stem);    a->in_b = clef_take(l, name, (int64_t)3 * d);
    snprintf(name, sizeof(name), "%s.out_proj.weight", stem); a->out_w = clef_take(l, name, (int64_t)d * d);
    snprintf(name, sizeof(name), "%s.out_proj.bias", stem);   a->out_b = clef_take(l, name, d);
}

/* Every tensor of JointSchemaHead's state dict, sized from joint_head_config.json.
 * Returns 1, or 0 with the first missing or misshapen tensor in `err`. */
static int clef_head_load(ClefHead *h, void *ctx, ClefLoadFn load, char *err, size_t cap)
{
    ClefLoader l = {ctx, load, 0, ""};
    const int H = h->hidden, W = h->width, F = h->feedforward;
    char stem[96];
    clef_take_norm(&l, &h->hidden_norm, "hidden_norm", H);
    h->memory_proj = clef_take(&l, "memory_projection.weight", (int64_t)W * H);
    h->question_proj = clef_take(&l, "question_projection.weight", (int64_t)W * H);
    h->option_question_proj = clef_take(&l, "option_question_projection.weight", (int64_t)W * H);
    h->global_proj = clef_take(&l, "global_projection.weight", (int64_t)W * H);
    h->option_context_proj = clef_take(&l, "option_context_projection.weight", (int64_t)W * H);
    h->option_lexical_proj = clef_take(&l, "option_lexical_projection.weight", (int64_t)W * H);
    h->type_embedding = clef_take(&l, "type_embedding.weight", (int64_t)3 * W);
    h->route = (ClefRoute *)calloc((size_t)h->routing_layers, sizeof(ClefRoute));
    h->dec = (ClefDecoder *)calloc((size_t)h->layers, sizeof(ClefDecoder));
    if (!h->route || !h->dec) { snprintf(err, cap, "out of memory"); return 0; }
    for (int i = 0; i < h->routing_layers; i++) {
        ClefRoute *r = &h->route[i];
        char name[160];
        snprintf(stem, sizeof(stem), "evidence_layers.%d.query_norm", i);        clef_take_norm(&l, &r->query_norm, stem, W);
        snprintf(stem, sizeof(stem), "evidence_layers.%d.memory_norm", i);       clef_take_norm(&l, &r->memory_norm, stem, W);
        snprintf(stem, sizeof(stem), "evidence_layers.%d.feedforward_norm", i);  clef_take_norm(&l, &r->ff_norm, stem, W);
        snprintf(stem, sizeof(stem), "evidence_layers.%d.attention", i);         clef_take_attn(&l, &r->attn, stem, W);
        snprintf(name, sizeof(name), "evidence_layers.%d.feedforward.0.weight", i); r->ff1_w = clef_take(&l, name, (int64_t)F * W);
        snprintf(name, sizeof(name), "evidence_layers.%d.feedforward.0.bias", i);   r->ff1_b = clef_take(&l, name, F);
        snprintf(name, sizeof(name), "evidence_layers.%d.feedforward.3.weight", i); r->ff2_w = clef_take(&l, name, (int64_t)W * F);
        snprintf(name, sizeof(name), "evidence_layers.%d.feedforward.3.bias", i);   r->ff2_b = clef_take(&l, name, W);
    }
    clef_take_norm(&l, &h->option_summary_norm, "option_summary_norm", W);
    for (int i = 0; i < h->layers; i++) {
        ClefDecoder *d = &h->dec[i];
        char name[160];
        snprintf(stem, sizeof(stem), "layers.%d.self_attn", i);      clef_take_attn(&l, &d->self_attn, stem, W);
        snprintf(stem, sizeof(stem), "layers.%d.multihead_attn", i); clef_take_attn(&l, &d->cross_attn, stem, W);
        snprintf(stem, sizeof(stem), "layers.%d.norm1", i);          clef_take_norm(&l, &d->norm1, stem, W);
        snprintf(stem, sizeof(stem), "layers.%d.norm2", i);          clef_take_norm(&l, &d->norm2, stem, W);
        snprintf(stem, sizeof(stem), "layers.%d.norm3", i);          clef_take_norm(&l, &d->norm3, stem, W);
        snprintf(name, sizeof(name), "layers.%d.linear1.weight", i); d->ff1_w = clef_take(&l, name, (int64_t)F * W);
        snprintf(name, sizeof(name), "layers.%d.linear1.bias", i);   d->ff1_b = clef_take(&l, name, F);
        snprintf(name, sizeof(name), "layers.%d.linear2.weight", i); d->ff2_w = clef_take(&l, name, (int64_t)W * F);
        snprintf(name, sizeof(name), "layers.%d.linear2.bias", i);   d->ff2_b = clef_take(&l, name, W);
    }
    clef_take_norm(&l, &h->field_norm, "field_norm", W);
    clef_take_norm(&l, &h->option_norm, "option_norm", W);
    h->score1_w = clef_take(&l, "residual_scorer.0.weight", (int64_t)W * 4 * W);
    h->score1_b = clef_take(&l, "residual_scorer.0.bias", W);
    h->score2_w = clef_take(&l, "residual_scorer.3.weight", W);
    h->score2_b = clef_take(&l, "residual_scorer.3.bias", 1);
    float *scalar;
    if ((scalar = clef_take(&l, "prior_logit_scale", 1))) { h->prior_logit_scale = *scalar; free(scalar); }
    if ((scalar = clef_take(&l, "joint_logit_scale", 1))) { h->joint_logit_scale = *scalar; free(scalar); }
    if ((scalar = clef_take(&l, "residual_gate", 1))) { h->residual_gate = *scalar; free(scalar); }
    if (l.missing) {
        snprintf(err, cap, "%d head tensor(s) missing or of the wrong size, the first %s", l.missing, l.first);
        return 0;
    }
    return 1;
}

/* ---- the pieces of the forward ---- */

static void clef_layernorm(float *out, const float *x, const ClefNorm *n, int rows, int d)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *v = x + (int64_t)r * d;
        float *o = out + (int64_t)r * d;
        double mean = 0, var = 0;
        for (int i = 0; i < d; i++) mean += v[i];
        mean /= d;
        for (int i = 0; i < d; i++) { double c = v[i] - mean; var += c * c; }
        var /= d;
        float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int i = 0; i < d; i++) o[i] = (float)((v[i] - mean) * inv) * n->w[i] + n->b[i];
    }
}

static void clef_add_bias(float *y, const float *b, int rows, int d)
{
    for (int r = 0; r < rows; r++) for (int i = 0; i < d; i++) y[(int64_t)r * d + i] += b[i];
}

/* nn.GELU() and activation="gelu": the exact erf form */
static float clef_gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

/* y[rows, out] = x @ W^T + b */
static void clef_linear(float *y, const float *x, const float *w, const float *b, int rows, int in, int out)
{
    matmul(y, x, w, rows, in, out);
    if (b) clef_add_bias(y, b, rows, out);
}

/* nn.MultiheadAttention(batch_first) without a mask: out[nq, d] */
static int clef_attention(const ClefAttn *a, const float *q_in, int nq, const float *kv_in, int nk,
                          int d, int heads, float *out)
{
    float *Q = (float *)malloc((size_t)nq * d * sizeof(float));
    float *K = (float *)malloc((size_t)nk * d * sizeof(float));
    float *V = (float *)malloc((size_t)nk * d * sizeof(float));
    float *C = (float *)malloc((size_t)nq * d * sizeof(float));
    int hd = d / heads, ok = Q && K && V && C;
    if (ok) {
        clef_linear(Q, q_in, a->in_w, a->in_b, nq, d, d);
        clef_linear(K, kv_in, a->in_w + (int64_t)d * d, a->in_b + d, nk, d, d);
        clef_linear(V, kv_in, a->in_w + (int64_t)2 * d * d, a->in_b + 2 * d, nk, d, d);
        const float scale = 1.0f / sqrtf((float)hd);
        #pragma omp parallel
        {
            float *p = (float *)malloc((size_t)nk * sizeof(float));
            #pragma omp for schedule(dynamic, 1)
            for (int t = 0; t < nq * heads; t++) {
                int i = t / heads, h = t % heads;
                float *c = C + (int64_t)i * d + h * hd;
                if (!p) { for (int j = 0; j < hd; j++) c[j] = NAN; continue; }
                const float *qv = Q + (int64_t)i * d + h * hd;
                float mx = -INFINITY;
                for (int k = 0; k < nk; k++) {
                    p[k] = dot_f32_lanes(qv, K + (int64_t)k * d + h * hd, hd) * scale;
                    if (p[k] > mx) mx = p[k];
                }
                double sum = 0;
                for (int k = 0; k < nk; k++) { p[k] = expf(p[k] - mx); sum += p[k]; }
                float inv = (float)(1.0 / sum);
                for (int j = 0; j < hd; j++) c[j] = 0.f;
                for (int k = 0; k < nk; k++) {
                    float w = p[k] * inv;
                    const float *vv = V + (int64_t)k * d + h * hd;
                    for (int j = 0; j < hd; j++) c[j] += w * vv[j];
                }
            }
            free(p);
        }
        clef_linear(out, C, a->out_w, a->out_b, nq, d, d);
    }
    free(Q); free(K); free(V); free(C);
    return ok;
}

/* x += linear2(gelu(linear1(norm(x)))) */
static int clef_feedforward(float *x, const ClefNorm *norm, const float *w1, const float *b1,
                            const float *w2, const float *b2, int rows, int d, int f)
{
    float *n = (float *)malloc((size_t)rows * d * sizeof(float));
    float *u = (float *)malloc((size_t)rows * f * sizeof(float));
    float *y = (float *)malloc((size_t)rows * d * sizeof(float));
    int ok = n && u && y;
    if (ok) {
        clef_layernorm(n, x, norm, rows, d);
        clef_linear(u, n, w1, b1, rows, d, f);
        for (int64_t i = 0; i < (int64_t)rows * f; i++) u[i] = clef_gelu(u[i]);
        clef_linear(y, u, w2, b2, rows, f, d);
        for (int64_t i = 0; i < (int64_t)rows * d; i++) x[i] += y[i];
    }
    free(n); free(u); free(y);
    return ok;
}

static double clef_norm2(const float *v, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return sqrt(s);
}

/* The LM head's row for a token id (the lexical option vectors read the output
 * embedding, not the input one), f32, `hidden` wide. Returns 0 on failure. */
typedef int (*ClefRowFn)(void *ctx, int id, float *out);

/* JointSchemaHead.forward for one record. `hidden` holds the final-normed
 * hidden state of every position, [n, H]; it is overwritten (hidden_norm runs
 * in place). `logits[q]` receives in->q[q].n values in the REFERENCE's option
 * order. Returns 1, or 0 with `err` set. */
static int clef_head_forward(const ClefHead *h, float *hidden, const ClefInput *in,
                             ClefRowFn lm_row, void *row_ctx, double **logits, char *err, size_t cap)
{
    const int H = h->hidden, W = h->width, F = h->feedforward, L = in->n, Q = in->n_q;
    int N = 0;
    for (int q = 0; q < Q; q++) N += in->q[q].n;
    float *memory = (float *)malloc((size_t)L * W * sizeof(float));
    float *qv = (float *)calloc((size_t)Q * H, sizeof(float));      /* question_vectors */
    float *ctxv = (float *)calloc((size_t)N * H, sizeof(float));    /* option context means */
    float *lex = (float *)calloc((size_t)N * H, sizeof(float));     /* lexical option means */
    float *row = (float *)malloc((size_t)H * sizeof(float));
    float *routed = (float *)malloc((size_t)N * W * sizeof(float));
    float *tmp = (float *)malloc((size_t)(N > Q ? N : Q) * W * sizeof(float));
    float *tmp2 = (float *)malloc((size_t)(N > Q ? N : Q) * W * sizeof(float));
    float *mnorm = (float *)malloc((size_t)L * W * sizeof(float));
    float *fields = (float *)malloc((size_t)Q * W * sizeof(float));
    float *base = (float *)malloc((size_t)Q * W * sizeof(float));
    float *summary = (float *)calloc((size_t)Q * W, sizeof(float));
    float *feat = (float *)malloc((size_t)4 * W * sizeof(float));
    float *hid = (float *)malloc((size_t)W * sizeof(float));
    int ok = memory && qv && ctxv && lex && row && routed && tmp && tmp2 && mnorm && fields &&
             base && summary && feat && hid;
    if (!ok) { snprintf(err, cap, "out of memory in the decision head"); goto done; }

    /* normalized_hidden = hidden_norm(hidden_states); memory = memory_projection(...) */
    clef_layernorm(hidden, hidden, &h->hidden_norm, L, H);
    matmul(memory, hidden, h->memory_proj, L, H, W);
    const float *global = hidden + (int64_t)(L - 1) * H;

    /* _mean_span for the questions and the options; the lexical vectors are the
     * mean of the LM head's rows at the option's tokens */
    for (int q = 0, o = 0; q < Q; q++) {
        const ClefQuestion *cq = &in->q[q];
        for (int t = cq->qs; t < cq->qe; t++)
            for (int i = 0; i < H; i++) qv[(int64_t)q * H + i] += hidden[(int64_t)t * H + i];
        for (int i = 0; i < H; i++) qv[(int64_t)q * H + i] /= (float)(cq->qe - cq->qs);
        for (int k = 0; k < cq->n; k++, o++) {
            float *c = ctxv + (int64_t)o * H, *x = lex + (int64_t)o * H;
            for (int t = cq->os[k]; t < cq->oe[k]; t++) {
                for (int i = 0; i < H; i++) c[i] += hidden[(int64_t)t * H + i];
                if (!lm_row(row_ctx, in->ids[t], row)) {
                    ok = 0; snprintf(err, cap, "cannot read the LM head's row %d", in->ids[t]); goto done;
                }
                for (int i = 0; i < H; i++) x[i] += row[i];
            }
            float count = (float)(cq->oe[k] - cq->os[k]);
            for (int i = 0; i < H; i++) { c[i] /= count; x[i] /= count; }
        }
    }

    /* option queries: context + lexical + the question's own vector */
    matmul(routed, ctxv, h->option_context_proj, N, H, W);
    matmul(tmp, lex, h->option_lexical_proj, N, H, W);
    for (int64_t i = 0; i < (int64_t)N * W; i++) routed[i] += tmp[i];
    matmul(base, qv, h->option_question_proj, Q, H, W);
    for (int q = 0, o = 0; q < Q; q++)
        for (int k = 0; k < in->q[q].n; k++, o++)
            for (int i = 0; i < W; i++) routed[(int64_t)o * W + i] += base[(int64_t)q * W + i];

    /* EvidenceRoutingLayer x routing_layers: the options read the whole sequence */
    for (int r = 0; r < h->routing_layers && ok; r++) {
        const ClefRoute *rl = &h->route[r];
        clef_layernorm(tmp, routed, &rl->query_norm, N, W);
        clef_layernorm(mnorm, memory, &rl->memory_norm, L, W);
        ok = clef_attention(&rl->attn, tmp, N, mnorm, L, W, h->heads, tmp2);
        for (int64_t i = 0; ok && i < (int64_t)N * W; i++) routed[i] += tmp2[i];
        ok = ok && clef_feedforward(routed, &rl->ff_norm, rl->ff1_w, rl->ff1_b, rl->ff2_w, rl->ff2_b, N, W, F);
    }
    if (!ok) { snprintf(err, cap, "out of memory in the decision head"); goto done; }

    /* fields: the question, a softmax-weighted summary of its options, the last
     * position, the question type */
    matmul(base, qv, h->question_proj, Q, H, W);
    for (int q = 0, o = 0; q < Q; q++) {
        const ClefQuestion *cq = &in->q[q];
        const float *field = base + (int64_t)q * W;
        double mx = -INFINITY, sum = 0;
        double *w = (double *)malloc((size_t)cq->n * sizeof(double));
        if (!w) { ok = 0; snprintf(err, cap, "out of memory in the decision head"); goto done; }
        for (int k = 0; k < cq->n; k++) {
            w[k] = dot_f32_lanes(routed + (int64_t)(o + k) * W, field, W) / sqrt((double)W);
            if (w[k] > mx) mx = w[k];
        }
        for (int k = 0; k < cq->n; k++) { w[k] = exp(w[k] - mx); sum += w[k]; }
        for (int k = 0; k < cq->n; k++) {
            float wk = (float)(w[k] / sum);
            for (int i = 0; i < W; i++) summary[(int64_t)q * W + i] += wk * routed[(int64_t)(o + k) * W + i];
        }
        free(w);
        o += cq->n;
    }
    clef_layernorm(fields, summary, &h->option_summary_norm, Q, W);
    matmul(hid, global, h->global_proj, 1, H, W);
    for (int q = 0; q < Q; q++)
        for (int i = 0; i < W; i++)
            fields[(int64_t)q * W + i] += base[(int64_t)q * W + i] + hid[i] +
                                          h->type_embedding[(int64_t)in->q[q].type * W + i];

    /* TransformerDecoderLayer x layers (norm_first): self-attention over the
     * fields, cross-attention into the (unnormalized) memory, feed-forward */
    for (int l = 0; l < h->layers && ok; l++) {
        const ClefDecoder *d = &h->dec[l];
        clef_layernorm(tmp, fields, &d->norm1, Q, W);
        ok = clef_attention(&d->self_attn, tmp, Q, tmp, Q, W, h->heads, tmp2);
        for (int64_t i = 0; ok && i < (int64_t)Q * W; i++) fields[i] += tmp2[i];
        if (ok) clef_layernorm(tmp, fields, &d->norm2, Q, W);
        ok = ok && clef_attention(&d->cross_attn, tmp, Q, memory, L, W, h->heads, tmp2);
        for (int64_t i = 0; ok && i < (int64_t)Q * W; i++) fields[i] += tmp2[i];
        ok = ok && clef_feedforward(fields, &d->norm3, d->ff1_w, d->ff1_b, d->ff2_w, d->ff2_b, Q, W, F);
    }
    if (!ok) { snprintf(err, cap, "out of memory in the decision head"); goto done; }
    clef_layernorm(fields, fields, &h->field_norm, Q, W);
    clef_layernorm(routed, routed, &h->option_norm, N, W);     /* options = option_norm(routed) */

    /* the score: prior (lexical vs question + last position) + gated joint
     * (scaled cosine + the residual scorer) */
    const double prior_scale = exp(fmin((double)h->prior_logit_scale, log(100.0)));
    const double joint_scale = exp(fmin((double)h->joint_logit_scale, log(100.0)));
    const double gate = 1.0 / (1.0 + exp(-(double)h->residual_gate));
    for (int q = 0, o = 0; q < Q; q++) {
        const ClefQuestion *cq = &in->q[q];
        const float *field = fields + (int64_t)q * W;
        for (int i = 0; i < H; i++) row[i] = qv[(int64_t)q * H + i] + global[i];
        double anchor_norm = fmax(clef_norm2(row, H), 1e-12);
        double field_norm = fmax(clef_norm2(field, W), 1e-8);
        for (int k = 0; k < cq->n; k++, o++) {
            const float *x = lex + (int64_t)o * H, *opt = routed + (int64_t)o * W;
            double lex_norm = fmax(clef_norm2(x, H), 1e-12), dot = 0;
            for (int i = 0; i < H; i++) dot += (double)x[i] * row[i];
            double prior = prior_scale * dot / (lex_norm * anchor_norm);
            double opt_norm = fmax(clef_norm2(opt, W), 1e-8), fo = 0;
            for (int i = 0; i < W; i++) fo += (double)field[i] * opt[i];
            double cosine = fo / (field_norm * opt_norm);
            for (int i = 0; i < W; i++) {
                feat[i] = field[i];
                feat[W + i] = opt[i];
                feat[2 * W + i] = field[i] * opt[i];
                feat[3 * W + i] = fabsf(field[i] - opt[i]);
            }
            clef_linear(hid, feat, h->score1_w, h->score1_b, 1, 4 * W, W);
            double residual = h->score2_b[0];
            for (int i = 0; i < W; i++) residual += (double)clef_gelu(hid[i]) * h->score2_w[i];
            logits[q][k] = prior + gate * (joint_scale * cosine + residual);
        }
    }
done:
    free(memory); free(qv); free(ctxv); free(lex); free(row); free(routed); free(tmp); free(tmp2);
    free(mnorm); free(fields); free(base); free(summary); free(feat); free(hid);
    return ok;
}

/* A question's answer from its logits (reference order): softmax, then both
 * back in the record's option order. */
static int clef_answer(const ClefQuestion *cq, const double *logits, DecideAnswer *a)
{
    a->n = cq->n;
    a->logits = (double *)calloc((size_t)cq->n, sizeof(double));
    a->probs = (double *)calloc((size_t)cq->n, sizeof(double));
    if (!a->logits || !a->probs) return 0;
    double mx = -INFINITY, sum = 0;
    for (int k = 0; k < cq->n; k++) if (logits[k] > mx) mx = logits[k];
    for (int k = 0; k < cq->n; k++) sum += exp(logits[k] - mx);
    for (int k = 0; k < cq->n; k++) {
        a->logits[cq->record_index[k]] = logits[k];
        a->probs[cq->record_index[k]] = exp(logits[k] - mx) / sum;
    }
    a->temperature = NAN;            /* the head's logits are the answer: no calibration */
    return 1;
}

#endif
