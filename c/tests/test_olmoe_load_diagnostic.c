/* Regression for the diagnostic emitted when an original OLMoE checkpoint is
 * passed directly to the engine. Such checkpoints do not contain the merged
 * expert tensors produced by tools/convert_olmoe_merged.py. */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static jmp_buf exit_env;
static int exit_status;
static _Noreturn void olmoe_test_exit(int status);

/* Exercise load_expert_merged itself without terminating the test process. */
#define exit olmoe_test_exit
#define main olmoe_main_unused
#include "../olmoe.c"
#undef main
#undef exit

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

static _Noreturn void olmoe_test_exit(int status) {
    exit_status = status;
    longjmp(exit_env, 1);
}

static int redirect_stderr(const char *path) {
    fflush(stderr);
    int saved = dup(fileno(stderr));
    if (saved < 0 || !freopen(path, "w+", stderr)) {
        fprintf(stdout, "FAIL: could not redirect stderr\n");
        exit(1);
    }
    return saved;
}

static void restore_stderr(int saved, char *buf, size_t bufsz) {
    fflush(stderr);
    rewind(stderr);
    size_t got = fread(buf, 1, bufsz - 1, stderr);
    buf[got] = 0;
    dup2(saved, fileno(stderr));
    close(saved);
}

static int run_load(Model *m, char *message, size_t message_size) {
    const char *path = "tests/tmp_olmoe_load_diagnostic.stderr";
    Slot slot;
    memset(&slot, 0, sizeof(slot));
    int saved = redirect_stderr(path);
    exit_status = -1;
    int trapped = setjmp(exit_env);
    if (!trapped) load_expert_merged(m, 0, 0, &slot);
    restore_stderr(saved, message, message_size);
    remove(path);
    CHECK(trapped, "load_expert_merged returned instead of refusing the checkpoint");
    CHECK(exit_status == 1, "unexpected exit status %d", exit_status);
    return trapped;
}

static void check_conversion_message(const char *message, const char *name) {
    CHECK(strstr(message, name) != NULL, "missing tensor name: %s", message);
    CHECK(strstr(message, "converted merged-int8 checkpoint") != NULL,
          "missing checkpoint requirement: %s", message);
    CHECK(strstr(message, "tools/convert_olmoe_merged.py") != NULL,
          "missing converter path: %s", message);
    CHECK(strstr(message, "--model <source-model>") != NULL,
          "missing converter input argument: %s", message);
    CHECK(strstr(message, "--out <converted-model>") != NULL,
          "missing converter output argument: %s", message);
    CHECK(strstr(message, "use <converted-model> as the model") != NULL,
          "missing instruction to load converted output: %s", message);
    CHECK(strstr(message, "-1 bytes") == NULL,
          "old misleading size diagnostic remains: %s", message);
}

int main(void) {
    const char *weight_name = "model.layers.0.mlp.experts.0.merged_weight";
    const char *scale_name = "model.layers.0.mlp.experts.0.qs";
    char message[2048];
    Model m;
    memset(&m, 0, sizeof(m));
    m.c.inter = 2;
    m.c.hidden = 3;

    /* No tensors: report the conversion requirement at the first missing tensor. */
    if (run_load(&m, message, sizeof(message)))
        check_conversion_message(message, weight_name);

    /* A valid merged weight followed by a missing scale must give the same remedy. */
    st_tensor tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].name = (char *)weight_name;
    tensors[0].nbytes = 18; /* 2 * (inter * hidden) + (hidden * inter) */
    m.S.t = tensors;
    m.S.n = 1;
    if (run_load(&m, message, sizeof(message)))
        check_conversion_message(message, scale_name);

    /* Present-but-malformed tensors remain on the untrusted-container path. */
    tensors[0].nbytes = 17;
    if (run_load(&m, message, sizeof(message))) {
        CHECK(strstr(message, "17 bytes") != NULL, "missing actual bad size: %s", message);
        CHECK(strstr(message, "refusing (untrusted container)") != NULL,
              "security refusal changed: %s", message);
        CHECK(strstr(message, "convert_olmoe_merged.py") == NULL,
              "malformed tensor was mislabeled as unconverted: %s", message);
    }

    tensors[0].nbytes = 18;
    tensors[1].name = (char *)scale_name;
    tensors[1].numel = 6;
    m.S.n = 2;
    if (run_load(&m, message, sizeof(message))) {
        CHECK(strstr(message, "6 elems") != NULL, "missing actual bad scale size: %s", message);
        CHECK(strstr(message, "refusing (untrusted container)") != NULL,
              "scale security refusal changed: %s", message);
        CHECK(strstr(message, "convert_olmoe_merged.py") == NULL,
              "malformed scale was mislabeled as unconverted: %s", message);
    }

    if (failures) {
        fprintf(stderr, "olmoe load diagnostic tests: %d FAILED\n", failures);
        return 1;
    }
    puts("olmoe load diagnostic tests: ok");
    return 0;
}
