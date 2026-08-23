/* Context-size gates for the qwen36 hybrid engine.
 *
 * Regression test for the fixed-size score buffer: attention() used to keep one
 * score per attended key in a stack `float sc[8192]`, filled with
 * `for (t = 0; t <= qpos; t++)`. serve_one() carried the same literal as the
 * Q36_MAXT default, and generate()/tf_nll() checked nothing at all. The two
 * literals agreed only by luck: raising Q36_MAXT moved the guard without moving
 * the buffer, so a longer prompt smashed the stack instead of being refused.
 * Same class of bug as #122/#110 in the MLA path, and fixed the same way -- a
 * per-thread heap buffer sized from max_t, plus one named capacity.
 *
 * The 40-layer hybrid also keeps two kinds of state, and only one of them
 * grows: the 10 Gated Attention layers hold a KV cache, while the 30 Gated
 * DeltaNet layers hold a recurrent state of fixed size and get no KV cache at
 * all (ensure_kv leaves K[i]/V[i] NULL). Mixing those up silently allocates 4x
 * the memory or indexes a NULL layer, so the layout is worth pinning too.
 *
 * No model file: ensure_kv only reads n_layers/is_attn/kv_heads/k_head_dim.
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int fails = 0;

/* Same pattern as test_omp_tune.c / test_stops.c, and for the same reason:
 * compat.h maps setenv() to SetEnvironmentVariableA, which updates the Win32
 * environment block -- but getenv() reads the CRT's own copy and never sees
 * it, so the value under test would silently not arrive. _putenv_s updates
 * the copy getenv() reads. */
static void env_set(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void env_unset(const char *name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static void ck(int cond, const char *what) {
    if (cond) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

/* the shipping Qwen3.6-35B-A3B geometry, from qwen36_meta.json */
#define REAL_LAYERS      40
#define REAL_KVHEADS      2
#define REAL_KHEADDIM   256
#define REAL_ATTN_LAYERS 10          /* i % 4 == 3 */

static void shape_model(Model *m, int n_layers, int kv_heads, int k_head_dim) {
    memset(m, 0, sizeof(*m));
    m->c.n_layers = n_layers;
    m->c.kv_heads = kv_heads;
    m->c.k_head_dim = k_head_dim;
    m->c.is_attn = calloc(n_layers, 1);
    for (int i = 0; i < n_layers; i++) m->c.is_attn[i] = (i % 4 == 3);
}

/* --- the layout every run depends on ------------------------------------- */
static void case_layout(void) {
    Model m; shape_model(&m, REAL_LAYERS, REAL_KVHEADS, REAL_KHEADDIM);
    printf("KV layout at a short context\n");

    m.max_t = 4096;
    ensure_kv(&m);
    ck(m.kv_cap == 4096, "kv_cap follows max_t");

    int with_cache = 0, without = 0;
    for (int i = 0; i < m.c.n_layers; i++) {
        if (m.c.is_attn[i]) { if (m.K[i] && m.V[i]) with_cache++; }
        else if (!m.K[i] && !m.V[i]) without++;
    }
    ck(with_cache == REAL_ATTN_LAYERS, "every attention layer has K and V");
    ck(without == REAL_LAYERS - REAL_ATTN_LAYERS,
       "no DeltaNet layer has a KV cache (recurrent state instead of history)");

    /* 2 kv heads * 256 dims * 4 B * 2 (K and V) * 10 layers */
    ck(2 * (int64_t)REAL_KVHEADS * REAL_KHEADDIM * (int64_t)sizeof(float)
       * REAL_ATTN_LAYERS == 40960, "context costs 40 KB per token");

    /* under ASan this is the actual out-of-bounds gate, not a formality */
    for (int i = 0; i < m.c.n_layers; i++) {
        if (!m.c.is_attn[i]) continue;
        int64_t last = (int64_t)m.c.kv_heads * m.max_t * m.c.k_head_dim - 1;
        m.K[i][last] = 1.f; m.V[i][last] = 2.f;
    }
    ck(1, "last element of every cache is writable");
}

/* --- serve reuses one cache across requests of different lengths ---------- */
static void case_growth(void) {
    Model m; shape_model(&m, REAL_LAYERS, REAL_KVHEADS, REAL_KHEADDIM);
    printf("growth across requests\n");

    m.max_t = 512;  ensure_kv(&m);
    float *first = m.K[3];
    m.max_t = 256;  ensure_kv(&m);
    ck(m.K[3] == first && m.kv_cap == 512,
       "a shorter request leaves the cache alone (never shrinks)");

    m.max_t = 2048; ensure_kv(&m);
    ck(m.kv_cap == 2048, "a longer request grows the cache");
    int64_t last = (int64_t)m.c.kv_heads * m.max_t * m.c.k_head_dim - 1;
    m.K[3][last] = 1.f;    /* the re-allocation must be sized to the NEW max_t */
    ck(1, "the grown cache is writable to its last element");
}

/* --- the ceiling itself --------------------------------------------------- */
static void case_ceiling(void) {
    printf("attention capacity\n");
    /* One capacity, named once, enforced on every path that sets max_t --
     * instead of a stack array in attention() and a second literal in
     * serve_one() that happened to agree. */
    ck(QWEN36_ATTN_MAX_CTX >= 131072,
       "declared capacity covers the model's context length");

    /* The score rows must be sized from max_t and carry one row per thread,
     * so a context past the old 8192 has somewhere to go. */
    Model m; shape_model(&m, 8, REAL_KVHEADS, 16);
    m.max_t = 16384;                       /* twice the old fixed buffer */
    ensure_kv(&m);
    ck(m.attn_sc != NULL && m.attn_sc_thr >= 1, "per-thread score rows allocated");
    for (int t = 0; t < m.attn_sc_thr; t++)
        m.attn_sc[(int64_t)t * m.kv_cap + m.kv_cap - 1] = 1.f;
    ck(1, "every thread's score row holds max_t entries");

    /* Q36_MAXT may lower the ceiling but never raise it past the capacity. */
    env_set("Q36_MAXT", "999999999");
    ck(qwen36_max_ctx() == QWEN36_ATTN_MAX_CTX,
       "Q36_MAXT cannot be raised past the capacity");
    env_set("Q36_MAXT", "4096");
    ck(qwen36_max_ctx() == 4096, "Q36_MAXT can lower the ceiling");
    env_unset("Q36_MAXT");
    ck(qwen36_max_ctx() == QWEN36_DEFAULT_MAX_CTX,
       "unset Q36_MAXT falls back to the conservative default");
}

int main(void) {
    case_layout();
    case_growth();
    case_ceiling();
    if (fails) { printf("FAILED %d\n", fails); return 1; }
    printf("OK test_qwen36_ctx\n");
    return 0;
}
