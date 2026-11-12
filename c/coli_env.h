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
#define CE_DSV41    0x100
#define CE_QWENIMAGE 0x200
#define CE_MIMO     0x400
#define CE_LAYA     0x800
#define CE_GLINER   0x1000
/* CE_ALL is retained for declarations shared by the original four engines.
 * New rows use explicit engine masks derived from their actual call sites. */
#define CE_ALL      (CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE)
/* engines == 0: read only by a tool (coli tui, iobench), never by an engine, so
 * no engine warns about it. */

#define CE_DEPRECATED 0x1   /* still read; `replacement` names what to use */
/* set for whichever engine runs, by coli or its gateway (NGEN, SERVE_BATCH, ...) or by the
 * system (Windows' cmd keeps its own prompt in PROMPT): an engine that does not read it says
 * nothing, the user did not set it for that engine */
#define CE_LAUNCHER   0x2

typedef struct {
    const char *name;
    unsigned char type;
    unsigned short engines;
    unsigned char flags;
    const char *replacement;   /* non-NULL only for CE_DEPRECATED */
} ColiEnvVar;

/* Sorted by name; check-env asserts the ordering so a merge cannot silently
 * introduce a duplicate. */
static const ColiEnvVar coli_env_table[] = {
    {"ABLATE_OUT",                       CE_PATH  , CE_COLIBRI                                      , 0             , NULL},
    {"ABLATE_SCORE",                     CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"ABSORB",                           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"ADAPTER", CE_STR, CE_COLIBRI, 0, NULL},
    {"AMX", CE_BOOL, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_KIMI | CE_QWEN | CE_QWEN38 | CE_MIMO, 0, NULL},
    {"AMX_S_MIN", CE_INT, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_KIMI | CE_QWEN | CE_QWEN38 | CE_MIMO, 0, NULL},
    {"AUTOPIN",                          CE_INT   , CE_COLIBRI | CE_KIMI                            , 0             , NULL},
    {"CACHE_ROUTE",                      CE_INT   , CE_COLIBRI | CE_QWEN, 0             , NULL},
    {"CAP",                              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CAP_RAISE",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CHAT",                             CE_STR   , CE_OLMOE                                        , CE_LAUNCHER             , NULL},
    {"CHAT_TEMPLATE",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CLEF_IDS", CE_STR, CE_QWEN, 0, NULL},
    {"CLEF_RECORDS", CE_STR, CE_QWEN, 0, NULL},
    {"CLEF_TOKENIZE", CE_STR, CE_QWEN, 0, NULL},
    {"CLUSTER_SHARED", CE_INT, CE_COLIBRI, 0, NULL},
    {"CLUSTER_WORKERS", CE_STR, CE_COLIBRI | CE_GLM53 | CE_DSV41, CE_LAUNCHER, NULL},
    {"CLUSTER_WORKER_BIND", CE_STR, CE_GLM53, 0, NULL},
    {"CLUSTER_WORKER_CACHE", CE_INT, CE_DSV41, 0, NULL},
    {"CLUSTER_WORKER_PORT", CE_STR, CE_COLIBRI | CE_GLM53 | CE_DSV41, 0, NULL},
    {"COLIBRI_RESIDENT",                 CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_ANS_DIRECT", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_ANS_PACK", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_ANS_PROFILE", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_ANS_SIDECAR", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_API_KEY", CE_STR, 0, 0, NULL},
    {"COLI_CLEF_MAX_LEN", CE_INT, CE_QWEN, 0, NULL},
    {"COLI_CLUSTER_ACT", CE_STR, CE_COLIBRI, 0, NULL},
    {"COLI_CLUSTER_WEIGHTS", CE_INT, CE_GLM53, 0, NULL},
    {"COLI_CORPUS_K",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CORPUS_MINACC",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CPU_CHECK", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CPU_MISSING", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA", CE_INT, CE_COLIBRI | CE_QWEN | CE_QWEN38 | CE_OLMOE | CE_DSV4 | CE_GLM53 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, CE_LAUNCHER, NULL},
    {"COLI_CUDA_ADDNORM", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ASYNC", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN",                   CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_ATTN_BATCH",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_ATTN_DEC", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN_DEC_GQA", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN_DEC_SEL", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN_DEC_TPP", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN_F2_WARPS", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN_GROUPED", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN_PREFIX",            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_ATTN_SEL_TC", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_ATTN_SHARD",             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_BD_FLOAT", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_BD_HCM", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_BD_TILE2", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_BD_TILES", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_CHAIN", CE_INT, CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_LAYA | CE_GLINER | CE_KIMI | CE_DSV41, 0, NULL},
    {"COLI_CUDA_CHAIN_FAULT", CE_INT, CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_CHAIN_FLASH", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_CHAIN_FLASH2", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_CHAIN_GREEDY_CARD", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_LANES", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_LANE_STAGE", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_LAYERS", CE_INT, CE_KIMI, 0, NULL},
    {"COLI_CUDA_CHAIN_LAYERWISE", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_LAYERWISE_ROWS", CE_INT, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_MOE", CE_STR, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_MOE_GROUP", CE_BOOL, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_MTP_CARD", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_MTP_POOL", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_PLE_BF16", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_POOL", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_POOL_CPU_ROWS", CE_INT, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_POOL_EXACT", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_POOL_GB", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_POOL_READ", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_POOL_STAGE", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_POOL_THREADS", CE_INT, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_PROMOTE", CE_INT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_PROMOTE_HITS", CE_INT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_PROMOTE_MIN", CE_INT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_PROMPT_STAGE_MB", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_QKV", CE_STR, CE_OLMOE, 0, NULL},
    {"COLI_CUDA_CHAIN_ROUTER_GATE", CE_STR, CE_QWEN, 0, NULL},
    {"COLI_CUDA_CHAIN_ROWS", CE_INT, CE_OLMOE | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_SERVE_THREADS", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_CHAIN_STAGE_RING", CE_BOOL, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_COPY_K", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_DENSE", CE_INT, CE_COLIBRI | CE_KIMI | CE_DSV4 | CE_GLM53 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_CUDA_DENSE_GB", CE_INT, CE_COLIBRI | CE_KIMI | CE_DSV4 | CE_GLM53 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_CUDA_DNREC_CFG", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_DNREC_ROWS", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_DNREC_SPLIT", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_DUAL_PROJ", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_EXACT_SLOTS", CE_STR, CE_QWEN, 0, NULL},
    {"COLI_CUDA_F32T", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_F8_WARP", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_FLUSH", CE_BOOL, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_G53_ROWS", CE_INT, CE_GLM53, 0, NULL},
    {"COLI_CUDA_GEMV_NR", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_GEMV_SK", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_GEMV_W4", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_GLM_ROWS", CE_INT, CE_COLIBRI, 0, NULL},
    {"COLI_CUDA_GLU", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_GROUP_MOE", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_HCG_SPLIT", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_HCMIX", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_HCM_SPLIT", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_HCM_TILE", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_HOST_POOL", CE_STR, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_HOST_WAIT_FAULT", CE_INT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_HOST_WAIT_S", CE_FLOAT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_CUDA_K3_ROWS", CE_INT, CE_KIMI, 0, NULL},
    {"COLI_CUDA_KV16", CE_STR, CE_QWEN, 0, NULL},
    {"COLI_CUDA_MOE_BATCH",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_MOE_BATCH_MIN",          CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_MOE_DOUBLE",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_CUDA_MOE_GV", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_MOE_HCM", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_MOE_TILE128", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_MTP",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PDL", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_PIPE",                   CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PIPE_SHARD",             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PIPE_S_MIN",             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_PROFILE", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_QSA_TILES", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_RESID",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_ROUTER",                 CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_SHARED_W4A16",           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_SHARED_W4A16_MIN_ROWS",  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_CUDA_TC_F16", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_TC_GEMM", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_TC_INT4", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_TC_MIN_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_TC_W4A16", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_TC_W4A16_MIN", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_TEST_CONV_ROOM_MB", CE_INT, CE_QWEN, 0, NULL},
    {"COLI_CUDA_TIER_STREAM", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_TIER_WARM", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_CUDA_UP_BATCH", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_CUDA_V41_ROWS", CE_INT, CE_DSV41, 0, NULL},
    {"COLI_CUDA_V41_SEG", CE_INT, CE_DSV41, 0, NULL},
    {"COLI_CUDA_W4_PACKED", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_DENSE_BITS", CE_INT, CE_QWEN, CE_LAUNCHER, NULL},
    {"COLI_DENSE_F16", CE_INT, CE_OLMOE, 0, NULL},
    {"COLI_DENSE_I8",                    CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_DENSE_IDOT", CE_BOOL, CE_QWEN, 0, NULL},
    {"COLI_DENSE_INT4", CE_STR, CE_QWEN, 0, NULL},
    {"COLI_DENSE_KEEP_I8", CE_BOOL, CE_QWEN, 0, NULL},
    {"COLI_DISKCLASS_WINDOW",            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_DISK_WEIGHTS",                CE_STR   , CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53, 0             , NULL},
    {"COLI_DRAFT_CORPUS",                CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_DSA_GATHER",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_DSV4_DLL",                    CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_E8_WARP", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_EMBED_MMAP", CE_STR, CE_QWEN, 0, NULL},
    {"COLI_ENV_DUMP", CE_BOOL, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_ENV_STRICT", CE_BOOL, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_EXACT_VERIFY",                CE_BOOL  , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_EXPERT_STORE",                CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_GEMM_CHUNK",                  CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_GLINER_MAX_LEN", CE_INT, CE_GLINER, 0, NULL},
    {"COLI_GPU", CE_STR, CE_COLIBRI | CE_QWEN | CE_QWEN38 | CE_OLMOE | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_GPUS", CE_STR, CE_COLIBRI | CE_QWEN | CE_QWEN38 | CE_OLMOE | CE_LAYA | CE_GLINER, CE_LAUNCHER, NULL},
    {"COLI_GPU_FAIL_AFTER", CE_STR, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_GROUP_ASYNC",                 CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_HEAD_PIN_MIN", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_HITS_LIVE", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_IMG_ACT8", CE_BOOL, CE_QWENIMAGE, 0, NULL},
    {"COLI_IMG_BITS", CE_INT, CE_QWENIMAGE, 0, NULL},
    {"COLI_IMG_LOAD_THREADS", CE_INT, CE_QWENIMAGE, 0, NULL},
    {"COLI_IMG_NATIVE", CE_BOOL, CE_QWENIMAGE, 0, NULL},
    {"COLI_IMG_TE", CE_STR, CE_QWENIMAGE, 0, NULL},
    {"COLI_K3_CKPT",                     CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"COLI_K3_CKPT_DIR",                 CE_STR   , CE_KIMI                                         , 0             , NULL},
    {"COLI_KEEP_F32",                    CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_KEEP_INT8",                   CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"COLI_KV_PREFIX", CE_BOOL, CE_DSV41 | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_MIMO, 0, NULL},
    {"COLI_KV_SHARE",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_LANG", CE_STR, 0, 0, NULL},
    {"COLI_LAYA_HEAD_MAX_LEN", CE_INT, CE_LAYA, 0, NULL},
    {"COLI_LAYA_MAX_LEN", CE_INT, CE_LAYA, 0, NULL},
    {"COLI_LMHEAD_GPU", CE_INT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_LOGIT_DUMP",                  CE_STR   , CE_ALL | CE_COLIBRI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_GLM53 | CE_MIMO, 0             , NULL},
    {"COLI_LOGIT_GAP",                   CE_BOOL  , CE_ALL | CE_COLIBRI | CE_OLMOE, 0             , NULL},
    {"COLI_LOOKUP", CE_INT, CE_COLIBRI | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_LOOKUP_DRAFTS", CE_INT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_LOOKUP_FORCE", CE_STR, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_MAP_EXPERTS", CE_BOOL, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_METAL", CE_INT, CE_COLIBRI | CE_GLM53 | CE_INKLING | CE_OLMOE | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_METAL_GEMM_MIN",              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_MOE_EXACT",             CE_BOOL  , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_PREFILL",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_RESSET",                CE_INT   , CE_ALL                                          , 0             , NULL},
    {"COLI_METAL_SPIN",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_METAL_UNTRACKED",             CE_INT   , CE_ALL                                          , 0             , NULL},
    {"COLI_MIR_STRIPE",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MMAP",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MODEL",                       CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MODEL_DIRS",                  CE_PATH  , CE_COLIBRI | CE_DSV4 | CE_DSV41, 0             , NULL},
    {"COLI_MODEL_MIRROR",                CE_STR   , CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53, 0             , NULL},
    {"COLI_MTP_GUARD_PCT",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_MTP_GUARD_WINDOW",            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_NO_FUSED_PAIR",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_NO_OMP_TUNE", CE_STR, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_NUMA",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_OMP_TUNED", CE_STR, CE_COLIBRI | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_PARK", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_PARK_MIN", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_GLM53, 0, NULL},
    {"COLI_PIN_SLOTS", CE_INT, CE_COLIBRI | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_MIMO, 0, NULL},
    {"COLI_PIPE_BLOCK",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_PLACE", CE_STR, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_POLICY",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_PREFILL_CHUNK",               CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_PREFIX_LOG", CE_BOOL, CE_DSV41 | CE_OLMOE | CE_QWEN | CE_COLIBRI | CE_KIMI | CE_INKLING | CE_DSV4 | CE_QWEN38 | CE_MIMO, 0, NULL},
    {"COLI_PREFIX_STORE", CE_STR, CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_GLM53 | CE_KIMI | CE_MIMO | CE_INKLING | CE_OLMOE, 0, NULL},
    {"COLI_PREFIX_STORE_BLOCK", CE_INT, CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_GLM53 | CE_KIMI | CE_MIMO | CE_INKLING | CE_OLMOE, 0, NULL},
    {"COLI_PREFIX_STORE_MB", CE_INT, CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_GLM53 | CE_KIMI | CE_MIMO | CE_INKLING | CE_OLMOE, 0, NULL},
    {"COLI_PROMPT",                      CE_STR   , CE_ALL | CE_COLIBRI, 0             , NULL},
    {"COLI_PROMPT_BREAK_AFTER", CE_INT, CE_COLIBRI | CE_INKLING | CE_OLMOE | CE_QWEN38 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_PROMPT_SEGMENT", CE_INT, CE_COLIBRI | CE_INKLING | CE_OLMOE | CE_QWEN38 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_PROMPT_SNAPSHOT", CE_STR, CE_QWEN, 0, NULL},
    {"COLI_RAM_OVERCOMMIT",              CE_INT   , CE_COLIBRI | CE_KIMI                            , 0             , NULL},
    {"COLI_RTOP8",                       CE_STR   , CE_ALL                                          , 0             , NULL},
    {"COLI_SERVER", CE_STR, 0, 0, NULL},
    {"COLI_SERVE_ALL_STOPS",             CE_STR   , CE_ALL | CE_COLIBRI | CE_OLMOE, 0             , NULL},
    {"COLI_SLAB_SHRINK",                 CE_BOOL  , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_SPEC_GATE", CE_INT, CE_COLIBRI | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_SSD_FAST_GBS",                CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_TEMP", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_MIMO, CE_LAUNCHER, NULL},
    {"COLI_TIMERS",                      CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"COLI_TRAIN_METAL_MIN", CE_INT, 0, 0, NULL},
    {"COLI_TRUNK_PROBE", CE_BOOL, CE_QWEN, 0, NULL},
    {"COLI_TUI_SIXEL", CE_STR, 0, 0, NULL},
    {"COLI_USAGE", CE_STR, CE_GLM53 | CE_KIMI | CE_OLMOE | CE_QWEN38 | CE_QWEN | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_USAGE_DECAY", CE_FLOAT, CE_COLIBRI | CE_DSV4 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN38 | CE_QWEN | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_USAGE_START", CE_STR, CE_COLIBRI | CE_DSV4 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN38 | CE_QWEN | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_V4_AUTOPIN",                  CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_DIRECT",                   CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_EXPERT_PREFETCH",          CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_EXPERT_SLOTS", CE_INT, CE_DSV4, 0, NULL},
    {"COLI_V4_MARKOV_BLOCK",             CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_MARKOV_KEEP",              CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_MARKOV_SPEC",              CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_PREFILL_POOL",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_PREWARM",                  CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_ROWS16",                   CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_SAVE_USAGE",               CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"COLI_V4_SHARED_BATCH",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"COLI_VK_ATTN",                     CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_ATTN_BLOCK", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_ATTN_BLOCK_FIXED", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_ATTN_DEC", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_ATTN_DEC_GQA", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_ATTN_DEC_TPP", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_ATTN_SLICE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_ATTN_SLICE_WG", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_BDA", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_DEV2", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_VK_CHAIN_DEVKV", CE_STR, CE_OLMOE, 0, NULL},
    {"COLI_VK_CHAIN_FLASH", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_GEMM", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_GEMM_BD", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_GEMM_ROWS", CE_INT, CE_DSV4, 0, NULL},
    {"COLI_VK_CHAIN_GEMV", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_GEMV2", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_GEMV_BD", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_GEMV_LPR", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_GEMV_NR", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_LAYERWISE", CE_STR, CE_QWEN38, 0, NULL},
    {"COLI_VK_CHAIN_LAYERWISE_ROWS", CE_INT, CE_QWEN38, 0, NULL},
    {"COLI_VK_CHAIN_MOE", CE_STR, CE_OLMOE | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_VK_CHAIN_MUX", CE_INT, CE_COLIBRI | CE_QWEN, 0, NULL},
    {"COLI_VK_CHAIN_POOL_CPU_ROWS", CE_INT, CE_QWEN38, 0, NULL},
    {"COLI_VK_CHAIN_PREWARM", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_PROF", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_ROWS_MAX", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_SPIN_US", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_CHAIN_SPLIT", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_COOP", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_COOP_SG", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_COOP_TILE", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_COPY_GRID", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DENSE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DENSE_HOST", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DEV", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DEV2", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DEV2_FAULT", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DEVICE_CAP_MB", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DNCONV_ROWS", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DNREC_ROWS", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DSA_GRID", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DSA_TILED", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_DSV4_PA", CE_BOOL, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_EXPERTS",                  CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_EXPERTS2", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_FREE_RESERVE_GB", CE_FLOAT, CE_DSV41, 0, NULL},
    {"COLI_VK_GEMM_MIN_S", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_GEMM_TILE", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_GEMM_WIDE", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_GGUF_RAW", CE_STR, CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_VK_GRP_WIDE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HOST_POOL_IMPORT", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HOST_VISIBLE_CAP_MB", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HYBRID", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HYBRID_BIG_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HYBRID_EVENTS", CE_STR, CE_OLMOE | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_VK_HYBRID_GATE", CE_STR, CE_OLMOE | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_VK_HYBRID_PCIE", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HYBRID_SCHED", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HYBRID_THREADS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_HYB_GATE_DEBUG", CE_STR, CE_OLMOE | CE_QWEN | CE_QWEN38, 0, NULL},
    {"COLI_VK_IDOT", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_IMPORT", CE_INT, CE_QWEN, 0, NULL},
    {"COLI_VK_KDA_ROWS", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_KV16", CE_STR, CE_OLMOE | CE_QWEN, 0, NULL},
    {"COLI_VK_KV_BLOCK", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO, 0, NULL},
    {"COLI_VK_KV_COLD", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO, 0, NULL},
    {"COLI_VK_KV_DEVICE_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO, 0, NULL},
    {"COLI_VK_KV_FP32", CE_BOOL, CE_DSV41, 0, NULL},
    {"COLI_VK_KV_PIN", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO, 0, NULL},
    /* retired: read once only to say so (v41_host_kvq's one line, the inverted meaning
     * included), so it takes no moved default and no deprecated-name rewrite here */
    {"COLI_VK_KV_QUANT", CE_STR, CE_DSV41, 0, NULL},
    {"COLI_VK_KV_SLICE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO, 0, NULL},
    {"COLI_VK_KV_SPLIT", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO, 0, NULL},
    {"COLI_VK_LEDGER", CE_INT, CE_DSV41, 0, NULL},
    {"COLI_VK_LOADER", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_MLA_DEC", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_MOE_GV", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_MOE_GV_HQ", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_QI_ATTN", CE_STR, CE_QWENIMAGE, 0, NULL},
    {"COLI_VK_QI_RESIDENT", CE_INT, CE_QWENIMAGE, 0, NULL},
    {"COLI_VK_QPREP",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_RESERVE2_GB", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_RESERVE_GB",               CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"COLI_VK_ROUTE_SG", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_SHADERS", CE_STR, CE_COLIBRI | CE_GLM53 | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_SHARED_LIMIT", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_SPIN_US", CE_STR, CE_ALL | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_STAGE", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_STAGED", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_STAGED_FAULT", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_SUBGROUP", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_AMD_OWN", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_BALLAST", CE_INT, CE_ALL | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_CONV_ROOM_MB", CE_INT, CE_QWEN, 0, NULL},
    {"COLI_VK_TEST_EV_NOWAIT", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_GEMM_BENCH", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_GEMM_FMT", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_GEMM_S", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_GEMM_SHAPE", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_HOSTMEM", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_IMPORT_BENCH", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_MATMUL_ONLY", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_NOFILL", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TEST_XB_LARGE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_BALANCE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_COOP", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_COOP_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_EXCLUSIVE", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_FILL", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_GB", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_GEMM_BD", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_GEMM_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_HOST", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_HOST_GB", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_HOST_HEAP_GB", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_MTP", CE_STR, CE_COLIBRI | CE_QWEN38 | CE_DSV41, 0, NULL},
    {"COLI_VK_TIER_POOL_COPY", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_PROMOTE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_QUEUE", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_RATE", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_RESERVE_GB", CE_FLOAT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_STREAM", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_STREAM_HALF", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_STREAM_MIN", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_STREAM_PAR", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_STREAM_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_STREAM_SLOTS", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_SYNC", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_TIER_WARM", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"COLI_VK_UP_IMPORT", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_V41_LAYERWISE", CE_BOOL, CE_DSV41, 0, NULL},
    {"COLI_VK_V41_WOA", CE_BOOL, CE_DSV41, 0, NULL},
    {"COLI_VK_WAIT_FAULT", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_WINDOW_ASK_MB", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_WINDOW_PROBE_MB", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_XB_GEMV", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_XB_GROUPED", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_XB_GROUPED_BD", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_XB_GROUPED_ROWS", CE_INT, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VK_XB_STEP", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"COLI_VOCAB_USAGE", CE_STR, CE_QWEN, 0, NULL},
    {"COLI_VULKAN", CE_INT, CE_COLIBRI | CE_GLM53 | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, CE_LAUNCHER, NULL},
    {"COLI_WORKER_WEIGHT", CE_INT, CE_GLM53, 0, NULL},
    {"COLI_XDNA",                        CE_BOOL  , CE_COLIBRI                                      , 0             , NULL},
    {"CONF_LIMIT",                       CE_FLOAT , CE_OLMOE | CE_QWEN                              , 0             , NULL},
    {"CONSIST", CE_BOOL, CE_COLIBRI, 0, NULL},
    {"CONSIST_NP", CE_INT, CE_COLIBRI, 0, NULL},
    {"CONSIST_TOL", CE_FLOAT, CE_COLIBRI, 0, NULL},
    {"COUPLE",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"COUPLE_D",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"COUPLE_K",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CTX", CE_INT, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_OLMOE | CE_MIMO, 0, NULL},
    {"CTX_MAX",                          CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"CUDA_DENSE",                       CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CUDA_EXPERT_GB",                   CE_FLOAT , CE_COLIBRI | CE_QWEN | CE_QWEN38, CE_LAUNCHER             , NULL},
    {"CUDA_EXPERT_LOAD_BALANCE",         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CUDA_RAW_EXPERTS", CE_INT, CE_COLIBRI | CE_KIMI | CE_OLMOE | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"CUDA_RELEASE_HOST",                CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"CUDA_RESERVE_GB",                  CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"DEBUG_LOGITS",                     CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"DEGRADE_TAU", CE_FLOAT, CE_COLIBRI, 0, NULL},
    {"DEGRADE_ZERO", CE_BOOL, CE_COLIBRI, 0, NULL},
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
    {"DSV4_CUDA_COMPRESSOR", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_DEEPGEMM",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DENSE_DG",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DENSE_DUMP",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DEVICE",                 CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DEVICES", CE_STR, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_DG_AB",                  CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DG_DUMP",                CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_DG_PROFILE",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_EXPERT_MIRRORS",         CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_FP4_ROWS",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_HEAD", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_HEAD_BATCH", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_HEAD_DEVICE", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_INDEXER_HEADS", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_INDEXER_PREP", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_MOE_CONTIGUOUS",         CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_FUSED6", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_MOE_GROUPED",            CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_MASKED",             CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_PROF",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_MOE_TP2",                CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_PEER_COMPUTE", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_PEER_DEVICE", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_PEER_FAULT", CE_STR, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_PEER_MIRRORS", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_PEER_RESERVE_MB", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_PIN_CAP_GB", CE_FLOAT, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_PIN_HOST",               CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_QKV_FUSED",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_REPLICATED_TP2",         CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_RESIDENT_EXPERTS", CE_STR, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_RESIDENT_INDEXER", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_RESIDENT_RESERVE_MB", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_RESIDENT_ROUTE", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_CUDA_TC",                     CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_TP2_TRACE",              CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_VRAM_RESERVE_MB",        CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_CUDA_WO_DECODE", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_DECODE_PROF",                 CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_HEAD_VERIFY", CE_BOOL, CE_DSV4, 0, NULL},
    {"DSV4_HYBRID",                      CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_IDX_VERIFY",                  CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"DSV4_MM_WARP", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_MM_WARP_VERIFY", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_MOE_EM_TB", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_MOE_EXPERT_MAJOR", CE_INT, CE_DSV4, 0, NULL},
    {"DSV4_MOE_EXPERT_MAJOR_VERIFY", CE_INT, CE_DSV4, 0, NULL},
    {"DUMP", CE_STR, CE_QWEN | CE_QWEN38 | CE_COLIBRI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_GLM53 | CE_DSV41, 0, NULL},
    {"DUMP_LAYERS",                      CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"ENC_DEBUG",                        CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"EXPERT_BUDGET",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"EXPERT_BUDGET_EXPERIMENTAL",       CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"EXPERT_DROP",                      CE_INT   , CE_OLMOE                                        , 0             , NULL},
    {"EXPERT_WORKER", CE_STR, CE_COLIBRI | CE_GLM53 | CE_DSV41, 0, NULL},
    {"FUSED3",                           CE_STR   , CE_OLMOE                                        , 0             , NULL},
    {"G53_DIRECT", CE_BOOL, CE_GLM53, 0, NULL},
    {"GGUF_CHECK", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_DSV41 | CE_MIMO, 0, NULL},
    {"GGUF_IGNORE_UNUSED", CE_STR, CE_QWEN, 0, NULL},
    {"GGUF_STRICT", CE_STR, CE_QWEN38, 0, NULL},
    {"GLM53_BITS",                       CE_INT   , CE_GLM53                                        , 0             , NULL},
    {"GLM53_CLUSTER_TIMEOUT", CE_INT, CE_GLM53, 0, NULL},
    {"GLM53_DUMP_INDEX",                 CE_BOOL  , CE_GLM53                                        , 0             , NULL},
    {"GLM53_EXPERT_GB",                  CE_FLOAT , CE_GLM53                                        , 0             , NULL},
    {"GLM53_MAXT",                       CE_INT   , CE_GLM53                                        , 0             , NULL},
    {"GLM53_NUCLEUS", CE_BOOL, CE_GLM53, 0, NULL},
    {"GLM53_PREFILL_CHUNK",              CE_INT   , CE_GLM53                                        , 0             , NULL},
    {"GLM53_REWIND", CE_INT, CE_GLM53, 0, NULL},
    {"GLM53_VERBOSE",                    CE_BOOL  , CE_GLM53                                        , 0             , NULL},
    {"GLM_SEGMENT_DBITS",                CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"GLM_SEGMENT_EBITS",                CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"GPU_DEV",                          CE_INT   , CE_INKLING                                      , 0             , NULL},
    {"GQ_EXACT", CE_STR, CE_COLIBRI | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"GQ_NODECODE", CE_STR, CE_QWEN, 0, NULL},
    {"GRAMMAR",                          CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"GRAMMAR_DRAFT",                    CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"HEAT_FILE",                        CE_STR   , CE_QWEN | CE_QWEN38, 0             , NULL},
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
    {"K3_CUDA",                          CE_INT   , CE_KIMI                                         , CE_LAUNCHER             , NULL},
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
    {"K3_VK_GB",                         CE_FLOAT , CE_KIMI                                         , 0             , NULL},
    {"K3_VK_UP",                         CE_INT   , CE_KIMI                                         , 0             , NULL},
    {"K3_X0",                            CE_PATH  , CE_KIMI                                         , 0             , NULL},
    {"KIMI_DSA_INDEXER", CE_BOOL, CE_KIMI, 0, NULL},
    {"KV16", CE_INT, CE_COLIBRI | CE_OLMOE, 0, NULL},
    {"KV8",                              CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KV8_GS",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVB_FLASH",                        CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVB_FLASH_MB",                     CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVB_TILE_MB",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KVSAVE",                           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"KV_SLOTS", CE_INT, CE_COLIBRI | CE_GLM53 | CE_KIMI | CE_INKLING | CE_OLMOE | CE_DSV4 | CE_QWEN | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, CE_LAUNCHER, NULL},
    {"KV_TQ",                            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"KV_TQ_POLAR",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"LANG", CE_STR, 0, 0, NULL},
    {"LC_ALL", CE_STR, 0, 0, NULL},
    {"LOOKA",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"LORA_UNSAFE", CE_STR, CE_COLIBRI, 0, NULL},
    {"MAX_NEW",                          CE_INT   , CE_OLMOE                                        , CE_LAUNCHER             , NULL},
    {"MIMO_CAP", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_CHUNK", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_CTX", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_DENSE_BITS", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_DIRECT", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_DIRS", CE_PATH, CE_MIMO, 0, NULL},
    {"MIMO_IDOT", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_LAYERS", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_LOGITS", CE_STR, CE_MIMO, 0, NULL},
    {"MIMO_READ_THREADS", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_STATS", CE_INT, CE_MIMO, 0, NULL},
    {"MIMO_TRACE", CE_STR, CE_MIMO, 0, NULL},
    {"MIMO_VISION_DUMP", CE_STR, CE_MIMO, 0, NULL},
    {"MIMO_VK_EXPERTS", CE_INT, CE_MIMO, 0, NULL},
    {"ML4_MMPROJ", CE_STR, CE_COLIBRI, 0, NULL},
    {"ML4_VISION", CE_STR, CE_COLIBRI, 0, NULL},
    {"ML4_VISION_GPU", CE_STR, CE_COLIBRI, 0, NULL},
    {"ML4_VIS_DUMP", CE_STR, CE_COLIBRI, 0, NULL},
    {"MLOCK",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"MODEL",                            CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"MTP",                              CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"MTP_DEBUG",                        CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"MTP_PRENORM",                      CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"MTP_PSTEP", CE_INT, CE_COLIBRI, 0, NULL},
    {"MTP_SWAP",                         CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"NGEN",                             CE_INT   , CE_COLIBRI | CE_DSV4                            , CE_LAUNCHER             , NULL},
    {"NOGPU",                            CE_STR   , CE_INKLING                                      , 0             , NULL},
    {"NOPACK",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"NOSTREAM",                         CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"NUCLEUS",                          CE_FLOAT , CE_COLIBRI | CE_OLMOE                           , 0             , NULL},
    {"N_NEW",                            CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"OMP_NUM_THREADS", CE_STR, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
    {"OPENAI",                           CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"ORACLE_STRICT", CE_BOOL, CE_COLIBRI, 0, NULL},
    {"ORACLE_TF_MAX_MISMATCHES",          CE_INT   , CE_COLIBRI, 0             , NULL},
    {"PAR_LOAD",                         CE_INT   , CE_OLMOE                                        , 0             , NULL},
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
    {"PROMPT",                           CE_STR   , CE_ALL | CE_COLIBRI, CE_LAUNCHER             , NULL},
    {"Q36_DN_GPU", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_EOS",                          CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"Q36_GGUF_ROPE_PERM", CE_INT, CE_QWEN, 0, NULL},
    {"Q36_ILV16", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_IQ3S_CODES", CE_BOOL, CE_QWEN, 0, NULL},
    {"Q36_KV", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_KV_RING", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_KV_RING_SPARE", CE_INT, CE_QWEN, 0, NULL},
    {"Q36_MAXT",                         CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"Q36_MMPROJ", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_MTP", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_MTP_DMOE", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_MTP_DRAFTS", CE_INT, CE_QWEN, 0, NULL},
    {"Q36_MTP_DUMP", CE_INT, CE_QWEN, 0, NULL},
    {"Q36_MTP_FORCE", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_MTP_GPU", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_MTP_PMIN", CE_FLOAT, CE_QWEN, 0, NULL},
    {"Q36_MTP_PSTEP", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_MTP_VOCAB", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_MTP_VOCAB_IDS", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_OFFER_SHEXP", CE_STR, CE_QWEN, 0, NULL},
    {"Q36_VIS_DUMP", CE_STR, CE_QWEN, 0, NULL},
    {"Q38_ACT_COV", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_ACT_DUMP", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_ACT_DUMP_LAYERS", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_ACT_STATS", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_DN_GPU", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_DRAFT_HEAD", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_EOS",                          CE_INT   , CE_QWEN38                                       , 0             , NULL},
    {"Q38_EXPERT_INT2", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_EXPERT_INT4", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_EXPERT_LOWBIT", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_EXPERT_PARALLEL_READS",        CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_EXPERT_PARTIAL", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_EXPERT_PREFETCH",              CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_FP8_KERNEL", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_HEAD_PIN_MARKS", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_ILV16", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_KV", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_MAXT",                         CE_INT   , CE_QWEN38                                       , 0             , NULL},
    {"Q38_MOE_GROUPED", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_MTP", CE_BOOL, CE_QWEN38, 0, NULL},
    {"Q38_MTP_CAP", CE_INT, CE_QWEN38, 0, NULL},
    {"Q38_MTP_CHAIN", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_MTP_DRAFTS", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_MTP_DUMP", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_MTP_FORCE", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_MTP_PSTEP", CE_BOOL, CE_QWEN38, 0, NULL},
    {"Q38_MTP_VOCAB", CE_INT, CE_QWEN38, 0, NULL},
    {"Q38_MTP_WIRING", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_NATIVE_BF16",                  CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_NATIVE_FP8",                   CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_PLE_ADVISE", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_PLE_PREFETCH",                 CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_PREFILL_BATCH",                CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_PREFILL_BATCH_ROWS", CE_INT, CE_QWEN38, 0, NULL},
    {"Q38_PREFILL_WORKSPACE_MIB", CE_INT, CE_QWEN38, 0, NULL},
    {"Q38_PREFIX_LOG",                   CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"Q38_TRUNK_CPU_INT4", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_CPU_INT8", CE_BOOL, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_GPU", CE_BOOL, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_INT4_HEAD", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_MIN_KB", CE_INT, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_SELFTEST", CE_BOOL, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_SKIP", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_STATS", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_TRUNK_STATS_IN", CE_STR, CE_QWEN38, 0, NULL},
    {"Q38_VISION",                       CE_BOOL  , CE_QWEN38                                       , 0             , NULL},
    {"QIV_PROFILE", CE_STR, CE_QWENIMAGE, 0, NULL},
    {"QT_GGUF_ARENA", CE_STR, CE_QWEN | CE_QWEN38, 0, NULL},
    {"QT_HOME", CE_STR, CE_QWEN | CE_QWEN38, 0, NULL},
    {"QT_LAYER_SPLIT", CE_INT, CE_QWEN | CE_QWEN38, 0, NULL},
    {"QT_NO_WARMSTART",                  CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"QT_PREFILL_REPLAN", CE_STR, CE_QWEN, 0, NULL},
    {"QT_PREFILL_REPLAN_MAX", CE_INT, CE_QWEN, 0, NULL},
    {"QT_UPLOAD_SYNC", CE_BOOL, CE_QWEN | CE_QWEN38, 0, NULL},
    {"QWEN36_QPACK", CE_PATH, CE_QWEN, 0, NULL},
    {"QWEN36_QPACK_SLOTS", CE_INT, CE_QWEN, 0, NULL},
    {"QWENIMAGE_ORACLE_TAG", CE_STR, CE_QWENIMAGE, 0, NULL},
    {"QWENIMAGE_PRINT_TOKENS", CE_STR, CE_QWENIMAGE, 0, NULL},
    {"QWENIMAGE_PROF", CE_STR, CE_QWENIMAGE, 0, NULL},
    {"QWEN_DENSE_BATCH",                 CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"QWEN_DENSE_CACHE", CE_STR, CE_QWEN, 0, NULL},
    {"QWEN_EXPERT_ACT", CE_STR, CE_QWEN, 0, NULL},
    {"QWEN_EXPERT_KERNEL", CE_BOOL, CE_QWEN, 0, NULL},
    {"QWEN_SHARED_BATCH",                CE_STR   , CE_QWEN                                         , 0             , NULL},
    {"RAM_GB", CE_FLOAT, CE_COLIBRI | CE_DSV4 | CE_KIMI | CE_OLMOE | CE_QWEN, CE_LAUNCHER, NULL},
    {"RANS_AVX512",                      CE_STR   , CE_ALL                                          , 0             , NULL},
    {"RANS_NEON",                        CE_STR   , CE_ALL                                          , 0             , NULL},
    {"RANS_PATH",                        CE_PATH  , CE_ALL                                          , 0             , NULL},
    {"REF",                              CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REF_FORCE",                        CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REPIN",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"REPIN_VERBOSE",                    CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REPLAY",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"REP_PEN",                          CE_FLOAT , CE_INKLING                                      , 0             , NULL},
    {"ROUTE_AGREE",                      CE_INT   , CE_COLIBRI | CE_QWEN, 0             , NULL},
    {"ROUTE_ALPHA",                      CE_FLOAT , CE_COLIBRI | CE_QWEN, 0             , NULL},
    {"ROUTE_DEBUG", CE_STR, CE_QWEN, 0, NULL},
    {"ROUTE_J",                          CE_INT   , CE_COLIBRI | CE_QWEN, 0             , NULL},
    {"ROUTE_M",                          CE_INT   , CE_COLIBRI | CE_QWEN, 0             , NULL},
    {"ROUTE_P",                          CE_FLOAT , CE_COLIBRI | CE_QWEN, 0             , NULL},
    {"ROUTE_TRACE", CE_STR, CE_ALL | CE_COLIBRI | CE_DSV4 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN38 | CE_QWEN | CE_DSV41 | CE_MIMO, 0, NULL},
    {"RSS_GUARD_GB",                     CE_FLOAT , CE_COLIBRI                                      , 0             , NULL},
    {"SCHEMA",                           CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"SCORE",                            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"SCORE_PREFIX",                     CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"SEED", CE_INT, CE_COLIBRI | CE_DSV41 | CE_INKLING | CE_MIMO, 0, NULL},
    {"SERVE", CE_INT, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, CE_LAUNCHER, NULL},
    {"SERVE_BATCH",                      CE_INT   , CE_COLIBRI | CE_GLM53 | CE_OLMOE, CE_LAUNCHER             , NULL},
    {"SMOOTH",                           CE_FLOAT , CE_OLMOE | CE_QWEN                              , 0             , NULL},
    {"SNAP", CE_STR, CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN | CE_QWEN38 | CE_MIMO | CE_LAYA | CE_GLINER, CE_LAUNCHER, NULL},
    {"SNAP_MIRROR",                      CE_STR   , CE_COLIBRI | CE_DSV4 | CE_DSV41 | CE_GLM53, 0             , NULL},
    {"SPEC",                             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"SPEC_PIN",                         CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"STATS",                            CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"TEMP", CE_FLOAT, CE_COLIBRI | CE_OLMOE | CE_KIMI | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_MIMO | CE_LAYA | CE_GLINER, CE_DEPRECATED, "COLI_TEMP"},
    {"TF",                               CE_STR   , CE_COLIBRI                                      , 0             , NULL},
    {"THINK",                            CE_INT   , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"TOK",                              CE_STR   , CE_QWEN | CE_QWEN38                             , 0             , NULL},
    {"TOKENS",                           CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"TOK_ONLY", CE_STR, CE_QWEN, 0, NULL},
    {"TOPK",                             CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"TOPP",                             CE_FLOAT , CE_COLIBRI | CE_INKLING                         , 0             , NULL},
    {"TRUNK_RESIDENT_LAYERS", CE_INT, CE_COLIBRI, 0, NULL},
    {"URING",                            CE_INT   , CE_COLIBRI                                      , 0             , NULL},
    {"USAGE_SAVE", CE_STR, CE_ALL | CE_COLIBRI | CE_DSV4 | CE_GLM53 | CE_INKLING | CE_KIMI | CE_OLMOE | CE_QWEN38 | CE_QWEN | CE_DSV41 | CE_MIMO, 0, NULL},
    {"V41_DIRECT", CE_BOOL, CE_DSV41, 0, NULL},
    {"V41_DSPARK", CE_BOOL, CE_DSV41, 0, NULL},
    {"V41_DSPARK_GATE", CE_INT, CE_DSV41, 0, NULL},
    {"V41_DSPARK_MAX", CE_INT, CE_DSV41, 0, NULL},
    {"V41_DSPARK_MINACC", CE_INT, CE_DSV41, 0, NULL},
    {"V41_ENGRAM_ADVISE", CE_STR, CE_DSV41, 0, NULL},
    {"V41_ENGRAM_ROWS", CE_INT, CE_DSV41, 0, NULL},
    {"V41_INDEX_OWNER", CE_BOOL, CE_DSV41, 0, NULL},
    {"V41_PREFILL_TAIL", CE_INT, CE_DSV41, 0, NULL},
    {"V41_READ_DEPTH", CE_INT, CE_DSV41, 0, NULL},
    {"V41_SPEC_FORCE", CE_BOOL, CE_DSV41, 0, NULL},
    {"V41_STATE_PROBE", CE_BOOL, CE_DSV41, 0, NULL},
    {"V41_STATS", CE_BOOL, CE_DSV41, 0, NULL},
    {"V41_TRACE", CE_BOOL, CE_DSV41, 0, NULL},
    {"V4_ATTN_BATCH_PROJ", CE_BOOL, CE_DSV4, 0, NULL},
    {"V4_DRAFT",                         CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"V4_EXPERT_UNION",                  CE_BOOL  , CE_DSV4                                         , 0             , NULL},
    {"V4_GPU_RESIDENT_FIRST", CE_BOOL, CE_DSV4, 0, NULL},
    {"V4_GPU_ROUTE_REUSE", CE_BOOL, CE_DSV4, 0, NULL},
    {"V4_IDX_BATCH",                     CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_IDX_FAST", CE_BOOL, CE_DSV4, 0, NULL},
    {"V4_IDX_IDENTITY",                  CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_LOADER_LANES",                  CE_INT   , CE_DSV4                                         , 0             , NULL},
    {"V4_MOE_BANK_FULL",                 CE_STR   , CE_DSV4                                         , 0             , NULL},
    {"V4_MOE_BATCH_PROF", CE_INT, CE_DSV4, 0, NULL},
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
    {"V4_REPLAY_TRACE", CE_PATH, CE_DSV4, 0, NULL},
    {"V4_SERVE_STATS_LOG", CE_BOOL, CE_DSV4, 0, NULL},
    {"VK_PROF", CE_STR, CE_ALL | CE_DSV4 | CE_QWEN | CE_GLM53 | CE_QWEN38 | CE_DSV41 | CE_QWENIMAGE | CE_MIMO | CE_LAYA | CE_GLINER, 0, NULL},
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
/* Set by `coli serve` on the engine it starts and read back by coli, which finds a port's engine
 * by its environment, never by an engine: not an unknown variable. */
static int coli_env_launcher_only(const char *n) {
    return !strcmp(n, "COLI_SERVE_PORT");
}

static int coli_env_is_ours(const char *n) {
    static const char *const prefixes[] = {
        "COLI_", "COLIBRI_", "K3_", "KIMI_", "INK_", "GLM53_",
        "Q38_", "Q36_", "QWEN_", "QWEN36_", "QWENIMAGE_", "QIV_", "DSV4_", "V4_", "V41_", "MIMO_"
    };
    for (unsigned i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i)
        if (!strncmp(n, prefixes[i], strlen(prefixes[i]))) return 1;
    return 0;
}

/* A numeric variable's value is meant for it when it reads as a number: Windows sets TEMP to its
 * temporary folder in every process, which is not the old temperature knob (the engines ignore a
 * TEMP that is not a number too), so it says nothing. */
static int coli_env_meant(const ColiEnvVar *v, const char *value) {
    if (v->type != CE_INT && v->type != CE_FLOAT) return 1;
    char *end;
    strtod(value, &end);
    return end != value && !*end;
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
static int coli_env_check(unsigned short self, const char *name) {
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
            if (!coli_env_is_ours(key) || coli_env_launcher_only(key)) continue;   /* not ours, or coli's own */
            const char *s = coli_env_suggest(key);
            fprintf(stderr, "[env] unknown variable %s", key);
            if (s) fprintf(stderr, " -- did you mean %s?", s);
            fprintf(stderr, "\n");
            bad++;
            continue;
        }
        if (v->engines && !(v->engines & self) && !(v->flags & CE_LAUNCHER)) {
            static const struct { unsigned short bit; const char *name; } owners[] = {
                {CE_COLIBRI, "colibri"}, {CE_KIMI, "kimi_k3"}, {CE_INKLING, "inkling"},
                {CE_OLMOE, "olmoe"}, {CE_DSV4, "deepseek-v4"}, {CE_QWEN, "qwen36"},
                {CE_GLM53, "glm53"}, {CE_QWEN38, "qwen38"}, {CE_DSV41, "deepseek-v41"},
                {CE_QWENIMAGE, "qwenimage"}, {CE_MIMO, "mimo"}, {CE_LAYA, "laya"},
                {CE_GLINER, "gliner_decide"}};
            fprintf(stderr, "[env] %s is not read by %s (it belongs to", key, name);
            for (unsigned i = 0; i < sizeof(owners) / sizeof(owners[0]); i++)
                if (v->engines & owners[i].bit) fprintf(stderr, " %s", owners[i].name);
            fprintf(stderr, ") -- it will have no effect\n");
            bad++;
            continue;
        }
        if ((v->flags & CE_DEPRECATED) && coli_env_meant(v, eq + 1))
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
static void coli_env_dump(unsigned short self, const char *name) {
    const char *enabled = getenv("COLI_ENV_DUMP");
    if (!enabled || !atoi(enabled)) return;
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
