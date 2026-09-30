/* The MoE router correction bias (mlp.gate.e_score_correction_bias) is one float
 * per expert. It used to land in falloc(n_experts) and be filled by st_read_f32,
 * which writes the tensor's DECLARED element count: a container declaring more
 * than n_experts values was a heap write past the buffer (run this under ASan to
 * see it). It must now go through the same cardinality check as every other
 * tensor. Pinned against a real one-layer MoE container loaded by the real
 * model_init_range:
 *   - a bias of exactly n_experts values loads;
 *   - a longer or shorter bias is refused by name, before anything is written. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#ifndef _WIN32
#include <sys/wait.h>
#endif
enum { D = 8, V = 4, E = 4, SI = 16 };
#define BIAS "model.layers.0.mlp.gate.e_score_correction_bias"
typedef struct { const char *name; int64_t n; } tdef;
static const tdef MOE[] = {
    {"model.embed_tokens.weight", V * D}, {"lm_head.weight", V * D}, {"model.norm.weight", D},
    {"model.layers.0.input_layernorm.weight", D},
    {"model.layers.0.post_attention_layernorm.weight", D},
    {"model.layers.0.mlp.gate.weight", E * D},
    {"model.layers.0.self_attn.q_proj.weight", 2 * D * D},
    {"model.layers.0.self_attn.k_proj.weight", D * D},
    {"model.layers.0.self_attn.v_proj.weight", D * D},
    {"model.layers.0.self_attn.o_proj.weight", D * D},
    {"model.layers.0.mlp.shared_expert.gate_proj.weight", SI * D},
    {"model.layers.0.mlp.shared_expert.up_proj.weight", SI * D},
    {"model.layers.0.mlp.shared_expert.down_proj.weight", D * SI},
};
enum { NMOE = (int)(sizeof(MOE) / sizeof(MOE[0])) };
/* bias element count per case; case 0 is the well-formed container */
static const int64_t BIAS_N[] = { E, E * 64, E - 1 };
enum { NCASES = (int)(sizeof(BIAS_N) / sizeof(BIAS_N[0])) };
static int write_text(const char *dir, const char *file, const char *text){
    char path[512]; snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    fputs(text, f); return fclose(f);
}
static int write_container(const char *dir, int64_t bias_n){
    tdef t[NMOE + 1]; int n = NMOE;
    memcpy(t, MOE, sizeof(MOE));
    t[n++] = (tdef){BIAS, bias_n};
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
                   "{\"hidden\":8,\"n_layers\":1,\"num_experts\":4,\"topk\":2,"
                   "\"moe_inter\":16,\"shared_inter\":16,\"q_heads\":1,\"kv_heads\":1,"
                   "\"head_dim\":8,\"q_head_dim\":16,\"k_head_dim\":8,\"v_head_dim\":8,"
                   "\"o_in\":8,\"partial_rotary_factor\":0.25,\"attn_output_gate\":true,"
                   "\"has_qk_norm\":false,\"layer_types\":[\"full_attention\"]}\n"))
        { fclose(f); return -1; }
    return fclose(f);
}
static int child_case(const char *dir){
    static Model m;
    model_init_range(&m, dir, 1, 8, 0, 0, 1, 1);
    return m.L[0].gate_bias ? 0 : 93;
}
static int g_fails = 0;
static void run_case(const char *self, int index){
    char dir[] = "test_qwen36_router_bias_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); g_fails++; return; }
    char err[600], cmd[1536], log[8192] = {0};
    snprintf(err, sizeof(err), "%s/stderr.txt", dir);
    if (write_container(dir, BIAS_N[index])) { printf("FAIL: case %d fixture\n", index); g_fails++; }
    snprintf(cmd, sizeof(cmd), "\"%s\" --child \"%s\" >/dev/null 2>\"%s\"", self, dir, err);
    int status = system(cmd);
    int got = status >= 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    FILE *f = fopen(err, "rb");
    if (f) { size_t r = fread(log, 1, sizeof(log) - 1, f); log[r] = 0; fclose(f); }
    int refused = strstr(log, BIAS) && strstr(log, "refusing");
    if (index == 0 ? got != 0 : (got != 1 || !refused)) {
        g_fails++;
        printf("FAIL: case %d (bias of %lld for %d experts): exit %d, want %d%s\n"
               "--- child stderr ---\n%s\n", index, (long long)BIAS_N[index], E, got,
               index ? 1 : 0, index ? " with a refusal naming the bias" : "", log);
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
    if (g_fails) { printf("test_qwen36_router_bias: %d failure(s)\n", g_fails); return 1; }
    printf("OK test_qwen36_router_bias: n_experts-sized bias loads, any other size refused\n");
    return 0;
}
