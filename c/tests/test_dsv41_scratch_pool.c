/* The scratch pool expert_fetch reads GGUF expert matrices into. It replaced a `static __thread`
 * buffer per OpenMP worker that was never freed; libgomp retires and respawns pool threads, so on
 * the real V4.1 GGUF each retired worker left its buffer behind. Checked on raw pthreads that exit
 * the way retired workers do: aligned and writable buffers, reuse, the cap, and thousands of
 * short-lived threads sharing a handful of buffers. (A buffer dropped over the cap is only seen by
 * LeakSanitizer; no fixture reaches expert_fetch's GGUF branch.) */
#define main dsv41_main_unused
#include "../deepseek_v41.c"
#undef main
#include <pthread.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
#define WAVES 250
#define WIDTH 8
#define NEED ((int64_t)6 << 20)

static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
static uint8_t *seen[WAVES * WIDTH];
static int seen_n, bad;

static void *worker(void *arg) {
    (void)arg;
    int64_t n = 0;
    uint8_t *p = scratch_get(NEED, &n);
    if (!p || n < NEED || ((uintptr_t)p % ST_DIRECT_ALIGN)) bad = 1;
    else memset(p, 0x5a, (size_t)NEED);
    pthread_mutex_lock(&mx);
    int known = 0;
    for (int i = 0; i < seen_n; i++) known |= seen[i] == p;
    if (!known && seen_n < WAVES * WIDTH) seen[seen_n++] = p;
    pthread_mutex_unlock(&mx);
    if (p) scratch_put(p, n);
    return NULL;
}

int main(void) {
    int64_t n1 = 0, n2 = 0;
    uint8_t *a = scratch_get(1 << 20, &n1);                  /* aligned, big enough, writable */
    CHECK(a && n1 >= (1 << 20) && ((uintptr_t)a % ST_DIRECT_ALIGN) == 0);
    memset(a, 0xa5, (size_t)n1);
    scratch_put(a, n1);
    uint8_t *b = scratch_get(1 << 19, &n2);                  /* a smaller need gets the same buffer back */
    CHECK(b == a && n2 == n1 && g_scratch_cnt == 0);
    scratch_put(b, n2);
    uint8_t *c = scratch_get(4 << 20, &n2);                  /* too small: traded in, the pool does not keep both */
    CHECK(c && n2 >= (4 << 20) && g_scratch_cnt == 0);
    scratch_put(c, n2);
    uint8_t *e = scratch_get(8 << 20, &n2);                  /* bigger than the pooled one: never handed out short */
    CHECK(e && n2 >= (8 << 20) && g_scratch_cnt == 0);
    compat_aligned_free(e);

    for (int i = 0; i < V41_SCRATCH_POOL + 8; i++) {         /* the cap */
        void *p = NULL;
        CHECK(posix_memalign(&p, ST_DIRECT_ALIGN, (size_t)1 << 16) == 0);
        scratch_put(p, 1 << 16);
    }
    CHECK(g_scratch_cnt == V41_SCRATCH_POOL);
    while (g_scratch_cnt > 0) { int64_t k = 0; compat_aligned_free(scratch_get(1 << 10, &k)); }

    for (int w = 0; w < WAVES; w++) {                        /* thread churn */
        pthread_t t[WIDTH];
        for (int i = 0; i < WIDTH; i++) CHECK(pthread_create(&t[i], NULL, worker, NULL) == 0);
        for (int i = 0; i < WIDTH; i++) CHECK(pthread_join(t[i], NULL) == 0);
    }
    CHECK(!bad && seen_n >= 1 && seen_n <= 2 * WIDTH);       /* bounded by concurrency, not by thread count */
    puts("dsv41 scratch pool: ok");
    return 0;
}
