/* Dense Qwen3.5 is num_experts == 0 on the qwen36 engine. Two invariants of that
 * mode are pinned here against a real one-layer container on disk, loaded by the
 * real model_init_range:
 *   - a dense container holding a router or expert tensor is refused as a shape
 *     conflict. The router bias used to be read into falloc(0), a heap write of
 *     the declared element count (run this under `make test-asan` to see it);
 *   - a well-formed dense container loads and has no expert cache index at all,
 *     instead of a malloc(0) whose NULL result read as OOM on some allocators. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#ifndef _WIN32
#include <sys/wait.h>
#endif
enum { D = 8, V = 4, I = 16 };
typedef struct { const char *name; int64_t n; } tdef;
static const tdef DENSE[] = {
    {"model.embed_tokens.weight", V * D}, {"lm_head.weight", V * D}, {"model.norm.weight", D},
    {"model.layers.0.input_layernorm.weight", D},
    {"model.layers.0.post_attention_layernorm.weight", D},
    {"model.layers.0.self_attn.q_proj.weight", 2 * D * D},
    {"model.layers.0.self_attn.k_proj.weight", D * D},
    {"model.layers.0.self_attn.v_proj.weight", D * D},
    {"model.layers.0.self_attn.o_proj.weight", D * D},
    {"model.layers.0.mlp.gate_proj.weight", I * D},
    {"model.layers.0.mlp.up_proj.weight", I * D},
    {"model.layers.0.mlp.down_proj.weight", D * I},
};
enum { NDENSE = (int)(sizeof(DENSE) / sizeof(DENSE[0])) };
/* Case 0 is the clean container; every other case adds one tensor a dense model
 * must not have. */
static const tdef EXTRA[] = {
    {NULL, 0},
    {"model.layers.0.mlp.gate.e_score_correction_bias", 1},
    {"model.layers.0.mlp.gate.weight", D},
    {"model.layers.0.mlp.experts.0.merged_weight", 1},
    {"model.layers.0.experts.0.gate_proj.weight", 1},
};
enum { NCASES = (int)(sizeof(EXTRA) / sizeof(EXTRA[0])) };
static int write_text(const char *dir, const char *file, const char *text){
    char path[512]; snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    fputs(text, f); return fclose(f);
}
static int write_container(const char *dir, const tdef *extra){
    tdef t[NDENSE + 1]; int n = NDENSE;
    memcpy(t, DENSE, sizeof(DENSE));
    if (extra->name) t[n++] = *extra;
    char hdr[4096]; int len = 0; int64_t off = 0;
    len += snprintf(hdr + len, sizeof(hdr) - len, "{");
    for (int i = 0; i < n; i++, off += t[i - 1].n * 4)
        len += snprintf(hdr + len, sizeof(hdr) - len,
                        "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
                        i ? "," : "", t[i].name, (long long)t[i].n, (long long)off,
                        (long long)(off + t[i].n * 4));
    len += snprintf(hdr + len, sizeof(hdr) - len, "}");
    char path[512]; snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    uint64_t hlen = (uint64_t)len; uint8_t le[8];
    for (int b = 0; b < 8; b++) le[b] = (uint8_t)(hlen >> (8 * b));
    fwrite(le, 1, 8, f); fwrite(hdr, 1, (size_t)len, f);
    float one = 0.01f;
    for (int64_t i = 0; i < off / 4; i++) fwrite(&one, 4, 1, f);
    if (write_text(dir, "config.json",
                   "{\"hidden_size\":8,\"num_hidden_layers\":1,\"vocab_size\":4,"
                   "\"rms_norm_eps\":1e-6}\n")) { fclose(f); return -1; }
    if (write_text(dir, "qwen36_meta.json",
                   "{\"hidden\":8,\"n_layers\":1,\"num_experts\":0,\"topk\":0,"
                   "\"moe_inter\":0,\"shared_inter\":16,\"q_heads\":1,\"kv_heads\":1,"
                   "\"head_dim\":8,\"q_head_dim\":16,\"k_head_dim\":8,\"v_head_dim\":8,"
                   "\"o_in\":8,\"partial_rotary_factor\":0.25,\"attn_output_gate\":true,"
                   "\"has_qk_norm\":false,\"layer_types\":[\"full_attention\"]}\n"))
        { fclose(f); return -1; }
    return fclose(f);
}
static int child_case(const char *dir){
    static Model m;
    model_init_range(&m, dir, 1, 8, 0, 0, 1, 1);
    return m.cache[0].slot_by_expert == NULL ? 0 : 93;
}
static int g_fails = 0;
static void run_case(const char *self, int index){
    char dir[] = "test_qwen36_dense_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); g_fails++; return; }
    char err[600], cmd[1536], log[4096] = {0};
    snprintf(err, sizeof(err), "%s/stderr.txt", dir);
    if (write_container(dir, &EXTRA[index])) { printf("FAIL: case %d fixture\n", index); g_fails++; }
    snprintf(cmd, sizeof(cmd), "\"%s\" --child \"%s\" >/dev/null 2>\"%s\"", self, dir, err);
    int status = system(cmd);
    int got = status >= 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    FILE *f = fopen(err, "rb");
    if (f) { size_t r = fread(log, 1, sizeof(log) - 1, f); log[r] = 0; fclose(f); }
    int refused = strstr(log, "container shape conflict") && strstr(log, EXTRA[index].name ? EXTRA[index].name : "\x01");
    if (index == 0 ? got != 0 : (got != 1 || !refused)) {
        g_fails++;
        printf("FAIL: case %d (%s): exit %d, want %d%s\n--- child stderr ---\n%s\n", index,
               EXTRA[index].name ? EXTRA[index].name : "clean dense", got, index ? 1 : 0,
               index ? " with a shape-conflict refusal" : " and no expert index", log);
    }
    const char *files[] = {"stderr.txt", "model.safetensors", "config.json", "qwen36_meta.json"};
    for (int i = 0; i < 4; i++) { char p[600]; snprintf(p, sizeof(p), "%s/%s", dir, files[i]); remove(p); }
    rmdir(dir);
}
int main(int argc, char **argv){
    if (argc == 3 && !strcmp(argv[1], "--child")) return child_case(argv[2]);
#ifndef _WIN32
    for (int i = 0; i < NCASES; i++) run_case(argv[0], i);
#endif
    if (g_fails) { printf("test_qwen36_dense_container: %d failure(s)\n", g_fails); return 1; }
    printf("OK test_qwen36_dense_container: dense loads without expert state, expert tensors refused\n");
    return 0;
}
