/* The tokens a model ends a turn with, as it declares them (qwen36.c model_eos_add,
 * serve_eos_ids): config.json's eos_token_id list and generation_config.json's number, a
 * GGUF's tokenizer.ggml.eog_token_ids with its eos_token_id, each once. Flint declares
 * <|end_of_text|>, <|user|>, <|observation|> and <|developer|>; before this the engine
 * stopped on the first alone, and after a tool call it wrote the tool's answer itself. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int has(const int *ids, int n, int id) {
    for (int k = 0; k < n; k++) if (ids[k] == id) return 1;
    return 0;
}

int main(void) {
    int fails = 0;
    /* a checkpoint directory: the list in config.json, one more in generation_config.json */
    const char *dir = "tests/tmp_q36_eos";
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    char path[512];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "wb");
    fputs("{\"hidden_size\": 64, \"num_hidden_layers\": 2, \"vocab_size\": 128, \"rms_norm_eps\": 1e-6,"
          " \"eos_token_id\": [101, 103, 105, 101]}", f);
    fclose(f);
    snprintf(path, sizeof path, "%s/generation_config.json", dir);
    f = fopen(path, "wb");
    fputs("{\"eos_token_id\": 118, \"temperature\": 1.0}", f);
    fclose(f);
    Cfg c; memset(&c, 0, sizeof c);
    load_cfg(&c, dir);
    int ids[Q36_MAX_EOS + 4];
    int n = serve_eos_ids(ids, Q36_MAX_EOS + 4);
    if (n != 4 || !has(ids, n, 101) || !has(ids, n, 103) || !has(ids, n, 105) || !has(ids, n, 118)) {
        printf("FAIL: config.json + generation_config.json gave %d ids:", n);
        for (int k = 0; k < n; k++) printf(" %d", ids[k]);
        printf(" (want 101 103 105 118)\n"); fails++;
    }
    snprintf(path, sizeof path, "%s/config.json", dir); remove(path);
    snprintf(path, sizeof path, "%s/generation_config.json", dir); remove(path);
#ifdef _WIN32
    _rmdir(dir);
#else
    rmdir(dir);
#endif

    /* a GGUF: eog_token_ids (u32) and eos_token_id, the latter already in the list */
    g_n_model_eos = 0;
    uint32_t eog[4] = {150001, 150003, 150005, 150018};
    gguf_kv kv[2]; memset(kv, 0, sizeof kv);
    kv[0].key = "tokenizer.ggml.eog_token_ids"; kv[0].type = GGUF_T_ARRAY; kv[0].atype = GGUF_T_U32;
    kv[0].n = 4; kv[0].arr = eog;
    kv[1].key = "tokenizer.ggml.eos_token_id"; kv[1].type = GGUF_T_U32; kv[1].v.u = 150001;
    gguf_ctx g; memset(&g, 0, sizeof g); g.n_kv = 2; g.kv = kv;
    gguf_model_eos(&g);
    n = serve_eos_ids(ids, Q36_MAX_EOS + 4);
    if (n != 4 || !has(ids, n, 150001) || !has(ids, n, 150003) || !has(ids, n, 150005) || !has(ids, n, 150018)) {
        printf("FAIL: the GGUF's eog_token_ids gave %d ids:", n);
        for (int k = 0; k < n; k++) printf(" %d", ids[k]);
        printf(" (want 150001 150003 150005 150018)\n"); fails++;
    }
    if (!fails) printf("OK test_qwen36_eos: config.json + generation_config.json and a GGUF's eog_token_ids, each id once\n");
    return fails ? 1 : 0;
}
