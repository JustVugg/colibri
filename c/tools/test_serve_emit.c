/* End-to-end test for the qwen36_serve.c socket-emit path WITHOUT a tokenizer.
 *
 * It loads a real (tiny) colibri container, runs generate(), and routes the
 * OpenAI-format output through g_sock_send into a file (instead of a socket).
 * This verifies the generate()+emit_openai_result()+socket-sink path that the
 * HTTP server uses, independent of encode_text (which needs a tokenizer).
 *
 * Build (Linux):  gcc -O2 -I. tools/test_serve_emit.c -o tools/test_serve_emit -lm -fopenmp
 * Run:            SNAP=<tiny_container>  [STREAM=1]  ./tools/test_serve_emit 16 8
 * Output:        serve_emit.out  (OpenAI chat.completion JSON, or SSE chunks + [DONE])
 */
#define QWEN36_NO_MAIN
#include "qwen36.c"
#include <fcntl.h>
#include <unistd.h>

static int g_fd = -1;
static void fake_send(long long fd, const char *buf, int n){
    int f = (int)fd;
    int off = 0;
    while (off < n){
        int r = (int)write(f, buf + off, (size_t)(n - off));
        if (r <= 0) break;
        off += r;
    }
}

int main(int argc, char **argv){
    const char *snap = getenv("SNAP");
    if (!snap){ fprintf(stderr, "set SNAP=<tiny container dir>\n"); return 1; }

    g_fd = open("serve_emit.out", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_fd < 0){ perror("open serve_emit.out"); return 1; }

    int cap  = argc > 1 ? atoi(argv[1]) : 16;
    int bits = argc > 2 ? atoi(argv[2]) : 8;

    double t0 = now_s();
    Model m;
    model_init(&m, snap, cap, bits);
    fprintf(stderr, "[test] model loaded in %.1fs | RSS %.2f GB\n", now_s() - t0, rss_gb());

    g_model      = "qwen3.6-35b-a3b-colibri";
    g_openai     = 1;
    g_stream     = (getenv("STREAM") != NULL) ? 1 : 0;
    g_sock_out   = (long long)g_fd;     /* route emit bytes to the file */
    g_sock_send  = fake_send;
    g_ttft       = -1;
    g_gen_t0     = now_s();
    g_oa_created = (long)time(NULL);
    snprintf(g_oa_id, sizeof g_oa_id, "chatcmpl-test");

    /* prompt token ids (no tokenizer needed; we drive generate directly) */
    int ids[] = {1, 2, 3, 4, 5};
    int np = 5, n_new = 8;
    int *out = (int*)malloc((size_t)(np + n_new) * sizeof(int));

    fprintf(stderr, "[test] generating (stream=%d) ...\n", g_stream);
    generate(&m, ids, np, n_new, out);
    emit_openai_result(out, np, n_new, g_stream);

    free(out);
    close(g_fd);
    fprintf(stderr, "[test] done -> serve_emit.out\n");
    return 0;
}
