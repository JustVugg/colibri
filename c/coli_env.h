#ifndef COLI_ENV_H
#define COLI_ENV_H
/* coli_env.h — the single place that knows which environment variables exist.
 *
 * WHY THIS EXISTS
 * ---------------
 * The engines read hundreds of environment variables from getenv() call sites
 * scattered across the sources, and until this header there was no list of
 * them anywhere in the code. (`make check-env` prints the current count; it is
 * the one number here that cannot go stale, because it is computed.)
 * Two consequences, both of which cost real debugging time:
 *
 *   COLI_PREFIL_CHUNK=512 ./colibri ...      <- typo: silently ignored
 *   K3_BITS=8 ./colibri ...                  <- wrong engine: silently ignored
 *
 * Neither prints anything. The run proceeds with the default, produces
 * perfectly plausible output, and the number you write down is wrong. That is
 * the failure mode this header removes: coli_env_check() is called once at
 * startup, compares the process environment against the table below, and says
 * so.
 *
 * This is the approach vLLM converged on (vllm/envs.py): one declaration point,
 * one prefix convention, validation at the boundary. It is worth stating that
 * the exact count is not the problem; the absence of a registry was.
 *
 * WHAT IT DELIBERATELY DOES NOT DO
 * --------------------------------
 * It does not replace the getenv() calls. Rewriting 187 read sites is a
 * mechanical but large change with a real chance of altering a default by
 * accident; it belongs in its own commit, engine by engine, and this table is
 * the precondition for doing it safely. Here the table is checked against the
 * source by `make check-env` (and CI), so it cannot drift in the meantime.
 *
 * KEEPING IT HONEST
 * -----------------
 * Read a new variable anywhere and `make check-env` fails until it is in the
 * table. Remove one and it fails until the row is gone. There is no way to
 * update the code and forget the registry.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The environment block. POSIX declares `environ` but not in any header you are
 * allowed to rely on; Windows spells it _environ; on macOS a program linked
 * against the shared libc must go through _NSGetEnviron(). */
#if defined(_WIN32)
#define COLI_ENVIRON _environ
#elif defined(__APPLE__)
#include <crt_externs.h>
#define COLI_ENVIRON (*_NSGetEnviron())
#else
extern char **environ;
#define COLI_ENVIRON environ
#endif

/* Value shape. Only used for reporting today -- coli_env_dump prints it so a
 * "why did my float get truncated" question answers itself. */
typedef enum { CE_BOOL, CE_INT, CE_FLOAT, CE_STR, CE_PATH } ColiEnvType;

/* Which binaries read the variable. The engines do NOT share a knob set: K3_*
 * is kimi_k3 only, INK_* is inkling only, and a few live in headers everyone
 * includes (route_trace.h, rans.h, omp_tune.h) and so are CE_ALL. */
#define CE_COLIBRI  0x1
#define CE_KIMI     0x2
#define CE_INKLING  0x4
#define CE_OLMOE    0x8
#define CE_DSV4     0x10
#define CE_QWEN     0x20
#define CE_GLM53    0x40
#define CE_QWEN38   0x80
/* CE_ALL is retained for declarations shared by the original four engines.
 * New rows use explicit engine masks derived from their actual call sites. */
#define CE_ALL      (CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE)

#define CE_DEPRECATED 0x1   /* still read; `replacement` names what to use */

typedef struct {
    const char *name;
    unsigned char type;
    unsigned char engines;
    unsigned char flags;
    const char *replacement;   /* non-NULL only for CE_DEPRECATED */
} ColiEnvVar;

/* Sorted by name; check-env asserts the ordering so a merge cannot silently
 * introduce a duplicate. */
static const ColiEnvVar coli_env_table[] = {
    {"ABLATE_OUT",                       CE_PATH  , CE_COLIBRI                                      , 0             , NULL},
    {"ABLATE_SCORE",                     CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"ABSORB",                           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"AUTOPIN",                          CE_INT   , CE_COLIBRI | CE_KIMI                            , 0             , NULL},
    {"CACHE_ROUTE",                      CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CAP",                              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CAP_RAISE",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CHAT",                             CE_STR   , CE_OLMOE                                        , 0             , NULL},
    {"CHAT_TEMPLATE",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CLUSTER_WORKERS",                  CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"CLUSTER_WORKER_PORT",              CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLIBRI_RESIDENT",                 CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_ANS_DIRECT",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_ANS_PACK",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_ANS_PROFILE",                 CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_ANS_SIDECAR",                 CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CORPUS_K",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CORPUS_MINACC",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA",                        CE_INT   , CE_COLIBRI | CE_QWEN                            , 0             , NULL},
    {"COLI_CUDA_ASYNC",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_ATTN",                   CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_ATTN_BATCH",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_ATTN_PREFIX",            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_ATTN_SHARD",             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_DUAL_PROJ",              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_F8_WARP",                CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_MOE_BATCH",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_MOE_BATCH_MIN",          CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_MOE_DOUBLE",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_MTP",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PIPE",                   CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PIPE_SHARD",             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PIPE_S_MIN",             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PROFILE",                CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_RESID",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_ROUTER",                 CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_SHARED_W4A16",           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_SHARED_W4A16_MIN_ROWS",  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_TC_INT4",                CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_TC_MIN_ROWS",            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_TC_W4A16",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_TC_W4A16_MIN",           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_W4_PACKED",              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_DENSE_I8",                    CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_DISKCLASS_WINDOW",            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_DISK_WEIGHTS",                CE_STR   , CE_COLIBRI | CE_DSV4                            , 0             , NULL},
    {"COLI_DRAFT_CORPUS",                CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_DSA_GATHER",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_DSV4_DLL",                    CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_ENV_DUMP",                    CE_BOOL  , CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38, 0             , NULL},
    {"COLI_ENV_STRICT",                  CE_BOOL  , CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38, 0             , NULL},
    {"COLI_EXPERT_STORE",                CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_GEMM_CHUNK",                  CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_GPU",                         CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_GPUS",                        CE_STR   , CE_COLIBRI | CE_QWEN                            , 0             , NULL},
    {"COLI_GPU_FAIL_AFTER",              CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_GROUP_ASYNC",                 CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_K3_CKPT",                     CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"COLI_K3_CKPT_DIR",                 CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"COLI_KEEP_F32",                    CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_KEEP_INT8",                   CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_KV_SHARE",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_LOGIT_DUMP",                  CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_LOGIT_GAP",                   CE_BOOL  , CE_ALL                                          , 0             , NULL},
    {"COLI_METAL",                       CE_INT   , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"COLI_METAL_GEMM_MIN",              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_MOE_EXACT",             CE_BOOL  , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_PREFILL",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_RESSET",                CE_INT   , CE_ALL                                          , 0             , NULL},
    {"COLI_METAL_SPIN",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_UNTRACKED",             CE_INT   , CE_ALL                                          , 0             , NULL},
    {"COLI_MIR_STRIPE",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MMAP",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MODEL",                       CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MODEL_DIRS",                  CE_PATH  , CE_COLIBRI | CE_DSV4                            , 0             , NULL},
    {"COLI_MODEL_MIRROR",                CE_STR   , CE_COLIBRI | CE_DSV4                            , 0             , NULL},
    {"COLI_MTP_GUARD_PCT",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MTP_GUARD_WINDOW",            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_NO_FUSED_PAIR",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_NO_OMP_TUNE",                 CE_STR   , CE_COLIBRI | CE_INKLING | CE_DSV4               , 0             , NULL},
    {"COLI_NUMA",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_OMP_TUNED",                   CE_STR   , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"COLI_PIPE_BLOCK",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_POLICY",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_PREFILL_CHUNK",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_PROMPT",                      CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_RAM_OVERCOMMIT",              CE_INT   , CE_COLIBRI | CE_KIMI                            , 0             , NULL},
    {"COLI_RTOP8",                       CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_SERVE_ALL_STOPS",             CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_SLAB_SHRINK",                 CE_BOOL  , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_SSD_FAST_GBS",                CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_TEMP",                        CE_FLOAT , CE_COLIBRI | CE_KIMI | CE_OLMOE                 , 0             , NULL},
    {"COLI_TIMERS",                      CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"COLI_USAGE",                       CE_STR   , CE_KIMI | CE_OLMOE | CE_QWEN38                 , 0             , NULL},
    {"COLI_USAGE_DECAY",                 CE_FLOAT , CE_KIMI | CE_QWEN38                            , 0             , NULL},
    {"COLI_V4_AUTOPIN",                  CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_DIRECT",                   CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_EXPERT_PREFETCH",          CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_MARKOV_BLOCK",             CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_MARKOV_KEEP",              CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_MARKOV_SPEC",              CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_PREFILL_POOL",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_PREWARM",                  CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_ROWS16",                   CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_SAVE_USAGE",               CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_SHARED_BATCH",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_VK_ATTN",                     CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_DENSE",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_DEV2",                     CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_EXPERTS",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_EXPERTS2",                 CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_QPREP",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_RESERVE2_GB",              CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_RESERVE_GB",               CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_SHADERS",                  CE_STR   , CE_COLIBRI | CE_KIMI | CE_GLM53                 , 0             , NULL},
    {"COLI_VK_SPIN_US",                  CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_VK_TEST_BALLAST",             CE_INT   , CE_ALL                                          , 0             , NULL},
    {"COLI_VULKAN",                      CE_INT   , CE_COLIBRI | CE_GLM53                           , 0             , NULL},
    {"CONF_LIMIT",                       CE_FLOAT , CE_OLMOE | CE_QWEN                              , 0             , NULL},
    {"COUPLE",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COUPLE_D",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COUPLE_K",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CTX",                              CE_INT   , CE_COLIBRI | CE_OLMOE | CE_DSV4                 , 0             , NULL},
    {"CTX_MAX",                          CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"CUDA_DENSE",                       CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CUDA_EXPERT_GB",                   CE_FLOAT , CE_COLIBRI | CE_QWEN                            , 0             , NULL},
    {"CUDA_EXPERT_LOAD_BALANCE",         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CUDA_RAW_EXPERTS",                 CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CUDA_RELEASE_HOST",                CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CUDA_RESERVE_GB",                  CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"DEBUG_LOGITS",                     CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"DIRECT",                           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"DISK_SPLIT",                       CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"DN_DBG",                           CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"DRAFT",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"DROP",                             CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"DSA",                              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"DSA_FORCE",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"DSA_TOPK",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"DSV4_ATTN_PROF",                   CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA",                        CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_ATTN_ASYNC",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_ATTN_DG_AB",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_BATCHED",                CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DEEPGEMM",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DENSE_DG",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DENSE_DUMP",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DEVICE",                 CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DG_AB",                  CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DG_DUMP",                CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DG_PROFILE",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_EXPERT_MIRRORS",         CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_FP4_ROWS",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_CONTIGUOUS",         CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_GROUPED",            CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_MASKED",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_PROF",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_TP2",                CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_PIN_HOST",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_QKV_FUSED",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_REPLICATED_TP2",         CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_TC",                     CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_TP2_TRACE",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_VRAM_RESERVE_MB",        CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_DECODE_PROF",                 CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_HYBRID",                      CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_IDX_VERIFY",                  CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DUMP",                             CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"DUMP_LAYERS",                      CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"ENC_DEBUG",                        CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"EXPERT_BUDGET",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"EXPERT_BUDGET_EXPERIMENTAL",       CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"EXPERT_DROP",                      CE_INT   , CE_OLMOE                                        , 0             , NULL},
    {"EXPERT_WORKER",                    CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"FUSED3",                           CE_STR   , CE_OLMOE                                        , 0             , NULL},
    {"GLM53_BITS",                       CE_INT   , CE_GLM53                                        , 0             , NULL},
    {"GLM53_DUMP_INDEX",                 CE_BOOL  , CE_GLM53                                        , 0             , NULL},
    {"GLM53_EXPERT_GB",                  CE_FLOAT , CE_GLM53                                        , 0             , NULL},
    {"GLM53_MAXT",                       CE_INT   , CE_GLM53                                        , 0             , NULL},
    {"GLM53_PREFILL_CHUNK",              CE_INT   , CE_GLM53                                        , 0             , NULL},
    {"GLM53_VERBOSE",                    CE_BOOL  , CE_GLM53                                        , 0             , NULL},
    {"GLM_SEGMENT_DBITS",                CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"GLM_SEGMENT_EBITS",                CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"GPU_DEV",                          CE_INT   , CE_INKLING                                      , 0             , NULL},
    {"GRAMMAR",                          CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"GRAMMAR_DRAFT",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"HEAT_FILE",                        CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"HOT",                              CE_INT   , CE_OLMOE | CE_QWEN                              , 0             , NULL},
    {"I3_AVX512",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"I3_AVX512_TEST",                   CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"I4S",                              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"I4_ACC512",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"I4_ACC512_TEST",                   CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"IDOT",                             CE_INT   , CE_COLIBRI | CE_INKLING | CE_OLMOE | CE_QWEN    , 0             , NULL},
    {"IDOT_GS",                          CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"INK_DENSE_Q4",                     CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"INK_METAL_MIN_S",                  CE_INT   , CE_INKLING                                      , 0             , NULL},
    {"INK_METAL_SHARED",                 CE_BOOL  , CE_INKLING                                      , 0             , NULL},
    {"INK_PREFIX_LOG",                   CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"INK_SEGMENT_BITS",                 CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"INK_SHARED_BATCH",                 CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"K3_BITS",                          CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_CHAT_IDS",                      CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_CHUNK",                         CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_CUDA",                          CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_DEBUG_OUT",                     CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_DIRECT",                        CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_DIRS",                          CE_PATH  , CE_KIMI                                         , 0             , NULL},
    {"K3_EXPERT_GB",                     CE_FLOAT , CE_KIMI                                         , 0             , NULL},
    {"K3_HEAD_BITS",                     CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_IDOT",                          CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_LAYERS",                        CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_LOAD_THREADS",                  CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_LOGITS",                        CE_PATH  , CE_KIMI                                         , 0             , NULL},
    {"K3_MAXT",                          CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_METAL",                         CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_MLA_BITS",                      CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_MMAP",                          CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_PIPE",                          CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_PREFIX_LOG",                    CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_STAT_EVERY",                    CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_THINK",                         CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_TOPP",                          CE_FLOAT , CE_KIMI                                         , 0             , NULL},
    {"K3_TRACE",                         CE_PATH  , CE_KIMI                                         , 0             , NULL},
    {"K3_VALIDATE_LAYER",                CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_VALIDATE_OUT",                  CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_VALIDATE_TOKEN",                CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_VAL_LOGITS",                    CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_VK",                            CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_VK_FILL_FRAC",                  CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"K3_VK_GB",                         CE_FLOAT , CE_KIMI                                         , 0             , NULL},
    {"K3_VK_UP",                         CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_X0",                            CE_PATH  , CE_KIMI                                         , 0             , NULL},
    {"KV8",                              CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KV8_GS",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVB_FLASH",                        CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVB_FLASH_MB",                     CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVB_TILE_MB",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVSAVE",                           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"KV_SLOTS",                         CE_INT   , CE_COLIBRI | CE_GLM53                           , 0             , NULL},
    {"KV_TQ",                            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KV_TQ_POLAR",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"LOOKA",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"MAX_NEW",                          CE_INT   , CE_OLMOE                                        , 0             , NULL},
    {"MLOCK",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"MODEL",                            CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"MTP",                              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"MTP_DEBUG",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"MTP_PRENORM",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"MTP_SWAP",                         CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"NGEN",                             CE_INT   , CE_COLIBRI | CE_DSV4                            , 0             , NULL},
    {"NOGPU",                            CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"NOPACK",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"NOSTREAM",                         CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"NUCLEUS",                          CE_FLOAT , CE_COLIBRI | CE_OLMOE                           , 0             , NULL},
    {"N_NEW",                            CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"OMP_NUM_THREADS",                  CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"OPENAI",                           CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"PILOT",                            CE_INT   , CE_COLIBRI | CE_OLMOE | CE_QWEN                 , 0             , NULL},
    {"PILOT_EVICT_GUARD",                CE_INT   , CE_COLIBRI | CE_OLMOE                           , 0             , NULL},
    {"PILOT_K",                          CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PILOT_REAL",                       CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PILOT_TWO",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PILOT_WORKERS",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PIN",                              CE_STR   , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"PIN_FILL",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PIN_GB",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"PIN_N",                            CE_INT   , CE_INKLING                                      , 0             , NULL},
    {"PIPE",                             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PIPE_WORKERS",                     CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PLANAR",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"PPL",                              CE_INT   , CE_OLMOE | CE_QWEN | CE_QWEN38                 , 0             , NULL},
    {"PREFETCH",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PROF",                             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"PROMPT",                           CE_STR   , CE_ALL                                          , 0             , NULL},
    {"Q36_EOS",                          CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"Q36_MAXT",                         CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"Q38_EOS",                          CE_INT   , CE_QWEN38                                       , 0             , NULL},
    {"Q38_EXPERT_PARALLEL_READS",        CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_EXPERT_PREFETCH",              CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_MAXT",                         CE_INT   , CE_QWEN38                                       , 0             , NULL},
    {"Q38_NATIVE_BF16",                  CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_NATIVE_FP8",                   CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_PLE_PREFETCH",                 CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_PREFILL_BATCH",                CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_PREFIX_LOG",                   CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_VISION",                       CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"QT_NO_WARMSTART",                  CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"QWEN_DENSE_BATCH",                 CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"QWEN_SHARED_BATCH",                CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"RAM_GB",                           CE_FLOAT , CE_COLIBRI | CE_KIMI | CE_DSV4                  , 0             , NULL},
    {"RANS_AVX512",                      CE_STR   , CE_ALL                                          , 0             , NULL},
    {"RANS_NEON",                        CE_STR   , CE_ALL                                          , 0             , NULL},
    {"RANS_PATH",                        CE_PATH  , CE_ALL                                          , 0             , NULL},
    {"REF",                              CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REF_FORCE",                        CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REPIN",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"REPIN_VERBOSE",                    CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REPLAY",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REP_PEN",                          CE_FLOAT , CE_INKLING                                      , 0             , NULL},
    {"ROUTE_AGREE",                      CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"ROUTE_ALPHA",                      CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"ROUTE_J",                          CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"ROUTE_M",                          CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"ROUTE_P",                          CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"ROUTE_TRACE",                      CE_STR   , CE_ALL | CE_QWEN38                              , 0             , NULL},
    {"RSS_GUARD_GB",                     CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"SCHEMA",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"SCORE",                            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"SCORE_PREFIX",                     CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"SEED",                             CE_INT   , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"SERVE",                            CE_INT   , CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38, 0             , NULL},
    {"SERVE_BATCH",                      CE_INT   , CE_COLIBRI | CE_GLM53                           , 0             , NULL},
    {"SMOOTH",                           CE_FLOAT , CE_OLMOE | CE_QWEN                              , 0             , NULL},
    {"SNAP",                             CE_STR   , CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38, 0             , NULL},
    {"SNAP_MIRROR",                      CE_STR   , CE_COLIBRI | CE_DSV4                            , 0             , NULL},
    {"SPEC",                             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"SPEC_PIN",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"STATS",                            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"TEMP",                             CE_FLOAT , CE_COLIBRI | CE_OLMOE                           , CE_DEPRECATED , "COLI_TEMP"},
    {"TF",                               CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"THINK",                            CE_INT   , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"TOK",                              CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"TOKENS",                           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"TOPK",                             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"TOPP",                             CE_FLOAT , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"URING",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"USAGE_SAVE",                       CE_STR   , CE_ALL | CE_QWEN38                              , 0             , NULL},
    {"V4_DRAFT",                         CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"V4_EXPERT_UNION",                  CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"V4_IDX_BATCH",                     CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_IDX_IDENTITY",                  CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_LOADER_LANES",                  CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"V4_MOE_BANK_FULL",                 CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_MOE_REFILL_GROUP",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_MTP",                           CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"V4_MTP_DRAFT",                     CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"V4_MTP_GB",                        CE_FLOAT , CE_DSV4                                         , 0             , NULL},
    {"V4_MTP_GPU_MIRRORS",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_MTP_MIN",                       CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"V4_MTP_PARTIAL_KEEP",              CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"V4_NGRAM",                         CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"V4_NGRAM_PARTIAL_KEEP",            CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"V4_PREFILL_CHUNK",                 CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"V4_PREFILL_SEGMENT",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_PREFIX_CKPT",                   CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_PREFIX_CKPT_DISK",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_PREFIX_CKPT_MIN",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_PREFIX_CKPT_SLOTS",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_PREFIX_LOG",                    CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"VK_PROF",                          CE_STR   , CE_ALL                                          , 0             , NULL},
    {"WARMUP",                           CE_INT   , CE_OLMOE | CE_QWEN                              , 0             , NULL},
    {"WIDE",                             CE_INT   , CE_OLMOE | CE_QWEN                              , 0             , NULL},
    {"XEXP",                             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
};
#define COLI_ENV_N ((int)(sizeof(coli_env_table) / sizeof(coli_env_table[0])))

static const char *coli_env_type_name(unsigned char t) {
    switch (t) {
    case CE_BOOL:  return "bool";
    case CE_INT:   return "int";
    case CE_FLOAT: return "float";
    case CE_PATH:  return "path";
    default:       return "string";
    }
}

static const ColiEnvVar *coli_env_find(const char *name) {
    int lo = 0, hi = COLI_ENV_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(coli_env_table[mid].name, name);
        if (c < 0) lo = mid + 1;
        else if (c > 0) hi = mid - 1;
        else return &coli_env_table[mid];
    }
    return NULL;
}

/* Levenshtein, capped: only used to say "did you mean" on a name the user
 * already got wrong, so the O(n*m) on two short strings is irrelevant. */
static int coli_env_edit(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b);
    if (la > 63 || lb > 63) return 99;
    int prev[64], cur[64];
    for (int j = 0; j <= lb; j++) prev[j] = j;
    for (int i = 1; i <= la; i++) {
        cur[0] = i;
        for (int j = 1; j <= lb; j++) {
            int del = prev[j] + 1, ins = cur[j - 1] + 1, sub = prev[j - 1] + (a[i - 1] != b[j - 1]);
            int m = del < ins ? del : ins;
            cur[j] = m < sub ? m : sub;
        }
        memcpy(prev, cur, sizeof(int) * (size_t)(lb + 1));
    }
    return prev[lb];
}

/* The nearest known name, or NULL when nothing is close enough to suggest. */
static const char *coli_env_suggest(const char *name) {
    const char *best = NULL;
    int bd = 4;   /* 3 edits is already a stretch for a 10-char name */
    for (int i = 0; i < COLI_ENV_N; i++) {
        int d = coli_env_edit(name, coli_env_table[i].name);
        if (d < bd) { bd = d; best = coli_env_table[i].name; }
    }
    return best;
}

/* Names we own. An unknown FOO=1 in the environment is not our business, but an
 * unknown COLI_FOO=1 almost certainly is. */
static int coli_env_is_ours(const char *n) {
    return !strncmp(n, "COLI_", 5) || !strncmp(n, "K3_", 3) || !strncmp(n, "INK_", 4) ||
           !strncmp(n, "GLM53_", 6) || !strncmp(n, "Q38_", 4);
}

/* Compare the process environment against the table. `self` is the calling
 * engine's CE_* bit, `name` its binary name for the messages.
 *
 * Warns, does not exit: an unrecognised variable has never stopped a run before
 * and turning that into a hard failure would break scripts that set a knob for
 * whichever engine they might launch. COLI_ENV_STRICT=1 makes it fatal for CI
 * and for anyone who wants the guarantee.
 *
 * Returns the number of problems found. */
static int coli_env_check(unsigned char self, const char *name) {
    int bad = 0, strict = getenv("COLI_ENV_STRICT") && atoi(getenv("COLI_ENV_STRICT"));
    char **envp = COLI_ENVIRON;
    for (char **e = envp; e && *e; e++) {
        const char *eq = strchr(*e, '=');
        if (!eq || eq == *e) continue;
        size_t n = (size_t)(eq - *e);
        if (n >= 64) continue;
        char key[64];
        memcpy(key, *e, n);
        key[n] = 0;
        const ColiEnvVar *v = coli_env_find(key);
        if (!v) {
            if (!coli_env_is_ours(key)) continue;   /* not ours, not our problem */
            const char *s = coli_env_suggest(key);
            fprintf(stderr, "[env] unknown variable %s", key);
            if (s) fprintf(stderr, " -- did you mean %s?", s);
            fprintf(stderr, "\n");
            bad++;
            continue;
        }
        if (!(v->engines & self)) {
            fprintf(stderr, "[env] %s is not read by %s (it belongs to %s%s%s%s%s%s%s%s) -- it will have no effect\n",
                    key, name,
                    (v->engines & CE_COLIBRI) ? "colibri " : "", (v->engines & CE_KIMI) ? "kimi_k3 " : "",
                    (v->engines & CE_INKLING) ? "inkling " : "", (v->engines & CE_OLMOE) ? "olmoe " : "",
                    (v->engines & CE_DSV4) ? "deepseek-v4 " : "", (v->engines & CE_QWEN) ? "qwen36 " : "",
                    (v->engines & CE_GLM53) ? "glm53 " : "", (v->engines & CE_QWEN38) ? "qwen38" : "");
            bad++;
            continue;
        }
        if (v->flags & CE_DEPRECATED)
            fprintf(stderr, "[env] %s is deprecated -- use %s\n", key, v->replacement ? v->replacement : "(nothing)");
    }
    if (bad && strict) {
        fprintf(stderr, "[env] COLI_ENV_STRICT=1 and %d problem(s) above\n", bad);
        exit(2);
    }
    return bad;
}

/* Every variable this engine reads, its type, and whether it is set right now.
 * Answers "is my export actually reaching the engine" without a debugger. */
static void coli_env_dump(unsigned char self, const char *name) {
    int n = 0;
    for (int i = 0; i < COLI_ENV_N; i++) {
        const ColiEnvVar *v = &coli_env_table[i];
        if (!(v->engines & self)) continue;
        const char *val = getenv(v->name);
        fprintf(stderr, "  %-32s %-6s %s%s\n", v->name, coli_env_type_name(v->type),
                val ? "= " : "(unset)", val ? val : "");
        n++;
    }
    fprintf(stderr, "[env] %d variables for %s\n", n, name);
}

#endif /* COLI_ENV_H */
