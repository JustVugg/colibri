/* Same resident engine, fresh sessions, alternating head modes and exact IDs. */
#define COLI_V4_UNIT_GENERATE_STATS
#define COLI_V4_SKIP_GENERATE_MAIN
#include "../deepseek_v4.c"

typedef struct { int ids[512], count; } HeadBenchTokens;
static int head_bench_token(void *data,int token,float logit,int position,int ordinal) {
    (void)logit; (void)position; (void)ordinal;
    HeadBenchTokens *tokens = data;
    if (tokens->count >= 512) return 1;
    tokens->ids[tokens->count++] = token;
    return 0;
}

int main(int argc,char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL PROMPT_FILE TOKENS ROUNDS MODES\n"
                "modes: 0=CPU, 1=CUDA, v=verify, d=unset/default, t=target-only\n"
                "round 0 is warmup; later rounds reverse mode order on odd rounds\n", argv[0]);
        return 2;
    }
    int maximum = atoi(argv[3]), rounds = atoi(argv[4]);
    const char *modes = argv[5]; size_t mode_count = strlen(modes);
    if (maximum < 1 || maximum > 512 || rounds < 1 || rounds > 10 ||
        !mode_count || mode_count > 8 || strspn(modes, "01vdt") != mode_count) return 2;
    setenv("V4_PREFIX_CKPT", "0", 1);
    setenv("V4_PREFIX_CKPT_DISK", "0", 1);
    setenv("COLI_V4_SAVE_USAGE", "0", 1);
    char error[512] = {0}, *prompt = NULL;
    size_t prompt_length = 0;
    char *user = v4_read_prompt_file(argv[2], error, sizeof(error));
    if (!user || coli_v4_prompt_build(&prompt, &prompt_length, user, NULL, COLI_V4_PROMPT_CHAT)) {
        fprintf(stderr, "prompt: %s\n", error); free(user); return 1;
    }
    free(user);
    ColiV4Engine *engine = NULL;
    ColiV4EngineOpenOptions open_options = {
        .target_model_dir = argv[1], .memory_limit_bytes = 48ULL << 30,
        .context_tokens = 512, .pin_slots_per_layer = -1,
    };
    if (coli_v4_engine_open(&engine, &open_options, error, sizeof(error))) {
        fprintf(stderr, "open: %s\n", error); free(prompt); return 1;
    }
    HeadBenchTokens reference = {0}; int result = 0;
    for (int round = 0; !result && round < rounds; round++) {
        for (size_t item = 0; !result && item < mode_count; item++) {
            char mode = modes[round % 2 ? mode_count - 1 - item : item];
            if (mode == 'd' || mode == 't') unsetenv("DSV4_CUDA_HEAD_BATCH");
            else setenv("DSV4_CUDA_HEAD_BATCH", mode == '0' ? "0" : "1", 1);
            setenv("DSV4_HEAD_VERIFY", mode == 'v' ? "1" : "0", 1);
            /* Restore the cold policy, retaining warmed immutable MTP weights. */
            for (int i = 0; i < V4_DSPARK_BLOCK; i++) g_v4ds_core.accept_ewma[i] = 0.5;
            ColiV4Session *session = NULL;
            ColiV4SessionCreateOptions create = {.max_prompt_tokens=512, .max_new_tokens_cap=maximum};
            ColiV4SessionGenerateOptions generate = {.max_new_tokens=maximum, .no_dspark=mode=='t'};
            ColiV4SessionGenerateStats stats = {0}; HeadBenchTokens tokens = {0};
            double head_before = g_v4_prof_head_s, block_before = g_v4_prof_block_s;
            double started = spec_now();
            fprintf(stderr, "head_bench round=%d mode=%c start\n", round, mode); fflush(stderr);
            result = coli_v4_session_create(&session, engine, &create, error, sizeof(error));
            if (!result) result = coli_v4_session_generate(session, prompt, prompt_length,
                &generate, head_bench_token, &tokens, &stats, error, sizeof(error));
            if (!reference.count) reference = tokens;
            int exact = tokens.count == reference.count &&
                !memcmp(tokens.ids, reference.ids, (size_t)tokens.count * sizeof(int));
            printf("{\"round\":%d,\"warmup\":%s,\"mode\":\"%c\",\"generated\":%d,"
                   "\"prompt\":%d,\"decode_s\":%.9f,\"ttft_s\":%.9f,\"wall_s\":%.9f,"
                   "\"head_s\":%.9f,\"block_s\":%.9f,\"drafted\":%llu,\"accepted\":%llu,"
                   "\"exact\":%s,\"result\":%d,\"ids\":[",
                   round, round ? "false" : "true", mode, stats.generated_tokens, stats.prompt_tokens,
                   stats.decode_sec, stats.time_to_first_token_sec, spec_now()-started,
                   g_v4_prof_head_s-head_before, g_v4_prof_block_s-block_before,
                   (unsigned long long)stats.speculative_drafted, (unsigned long long)stats.speculative_accepted,
                   exact ? "true" : "false", result);
            for (int i = 0; i < tokens.count; i++) printf("%s%d", i ? "," : "", tokens.ids[i]);
            puts("]}"); fflush(stdout);
            coli_v4_session_destroy(session);
            if (!exact) result = -1;
            if (result) fprintf(stderr, "head benchmark failed: %s exact=%d\n", error, exact);
        }
    }
    coli_v4_engine_destroy(engine); free(prompt);
    return result ? 1 : 0;
}
