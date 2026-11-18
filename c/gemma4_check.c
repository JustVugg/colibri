/* gemma4_check.c -- read-only Gemma 4 E2B GGUF compatibility validator.
 *
 * This deliberately validates metadata and tensor shapes without loading the
 * 3.2 GiB model weights. It is the first checkpoint before implementing the
 * text-only reference forward path.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"

static int failures;

static void fail(const char *what)
{
    fprintf(stderr, "gemma4-check: %s\n", what);
    failures++;
}

static int i64(gguf_ctx *g, const char *key, int fallback)
{
    int64_t value = gguf_get_int(g, key, fallback);
    return (int)value;
}

static const gguf_tensor *tensor(gguf_ctx *g, const char *name)
{
    const gguf_tensor *t = gguf_find(g, name);
    if (!t) {
        char message[256];
        snprintf(message, sizeof message, "missing tensor %s", name);
        fail(message);
    }
    return t;
}

static void shape(gguf_ctx *g, const char *name, int64_t n0, int64_t n1)
{
    const gguf_tensor *t = tensor(g, name);
    if (!t) return;
    if (t->n_dims != 2 || t->ne[0] != n0 || t->ne[1] != n1) {
        char message[256];
        snprintf(message, sizeof message,
                 "%s has shape [%lld,%lld], expected [%lld,%lld]", name,
                 (long long)t->ne[0], (long long)t->ne[1],
                 (long long)n0, (long long)n1);
        fail(message);
    }
}

static void vector(gguf_ctx *g, const char *name, int64_t n0)
{
    const gguf_tensor *t = tensor(g, name);
    if (!t) return;
    if (t->n_dims != 1 || t->ne[0] != n0) {
        char message[256];
        snprintf(message, sizeof message,
                 "%s has shape [%lld], expected [%lld]", name,
                 (long long)t->ne[0], (long long)n0);
        fail(message);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: gemma4-check <gemma4.gguf>\n");
        return 2;
    }
    gguf_ctx g;
    if (gguf_open(&g, argv[1])) {
        fprintf(stderr, "gemma4-check: %s\n", g.err);
        return 1;
    }
    const gguf_kv *arch = gguf_get(&g, "general.architecture");
    if (!arch || arch->type != GGUF_T_STRING || strcmp(arch->s, "gemma4")) {
        fail("general.architecture is not gemma4");
    }
    int layers = i64(&g, "gemma4.block_count", -1);
    int d = i64(&g, "gemma4.embedding_length", -1);
    int vocab = -1;
    const gguf_kv *tokens = gguf_get(&g, "tokenizer.ggml.tokens");
    if (tokens && tokens->type == GGUF_T_ARRAY) vocab = (int)tokens->n;
    if (layers != 35 || d != 1536 || vocab != 262144)
        fail("expected E2B shape: 35 layers, hidden 1536, vocab 262144");

    shape(&g, "token_embd.weight", d, vocab);
    vector(&g, "output_norm.weight", d);
    shape(&g, "per_layer_token_embd.weight", 35 * 256, vocab);
    shape(&g, "per_layer_model_proj.weight", d, 35 * 256);
    vector(&g, "per_layer_proj_norm.weight", 256);

    int full_layers[] = {4, 9, 14, 19, 24, 29, 34};
    int full = 0;
    for (int layer = 0; layer < layers; layer++) {
        int is_full = 0;
        for (size_t i = 0; i < sizeof full_layers / sizeof full_layers[0]; i++)
            is_full |= layer == full_layers[i];
        int hd = is_full ? 512 : 256;
        int ffn = layer < 15 ? 6144 : 12288;
        char name[128];
        snprintf(name, sizeof name, "blk.%d.attn_q.weight", layer);
        shape(&g, name, d, 8 * hd);
        snprintf(name, sizeof name, "blk.%d.attn_k.weight", layer);
        shape(&g, name, d, hd);
        snprintf(name, sizeof name, "blk.%d.attn_v.weight", layer);
        shape(&g, name, d, hd);
        snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", layer);
        shape(&g, name, d, ffn);
        snprintf(name, sizeof name, "blk.%d.ffn_up.weight", layer);
        shape(&g, name, d, ffn);
        snprintf(name, sizeof name, "blk.%d.ffn_down.weight", layer);
        shape(&g, name, ffn, d);
        snprintf(name, sizeof name, "blk.%d.inp_gate.weight", layer);
        shape(&g, name, d, 256);
        snprintf(name, sizeof name, "blk.%d.proj.weight", layer);
        shape(&g, name, 256, d);
        full += is_full;
    }
    if (full != 7) fail("full-attention layer pattern is not the E2B pattern");

    printf("{\"architecture\":\"gemma4\",\"layers\":%d,\"hidden\":%d,"
           "\"vocab\":%d,\"full_attention_layers\":%d,"
           "\"ple_width\":256,\"status\":\"%s\"}\n",
           layers, d, vocab, full, failures ? "incompatible" : "compatible");
    gguf_close(&g);
    return failures ? 1 : 0;
}
