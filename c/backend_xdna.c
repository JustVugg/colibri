/* Optional AMD XDNA2 helper binding -- the C side, which never touches XRT.
 *
 * The split mirrors backend_loader.c: the host resolves an optional native DLL
 * at runtime and links none of its dependencies, so an ordinary Colibri build
 * has no XRT headers, no XRT import library and no XRT DLL import. Everything
 * that needs XRT lives in coli_xdna.dll, which this file loads and may refuse.
 *
 * Refusal is the normal case. A machine with no NPU, no XRT or no helper is
 * ordinary, and every rejection path here ends the same way: the lane is
 * unavailable and the caller keeps doing exactly what it does today.
 *
 * W2-N7-I1 implements the binding only. There is deliberately no device open,
 * no artifact, no weight preparation and no dispatch here yet; the entry-point
 * set is the smallest one that makes the boundary versioned and testable, so
 * later slices extend it rather than inheriting speculative signatures.
 */

#include "backend_xdna.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* posix_memalign / compat_aligned_free. compat.h already maps these onto
 * _aligned_malloc/_aligned_free on Windows and carries the warning that the
 * two MUST be paired -- reusing it avoids a second aligned-allocation
 * convention in the same tree. */
#include "compat.h"

#ifdef _WIN32
#include <windows.h>
#endif

/* Entry points required of a helper at ABI generation 2. Still minimal: exactly
 * the calls the first blocking operation needs, and nothing speculative.
 *
 * Everything crossing this boundary is a fixed-width integer, a size_t, a C
 * string, or a raw pointer with an explicit byte count. No engine type, no XRT
 * type and no C++ type appears in any signature, so the helper cannot be handed
 * -- and cannot acquire -- any knowledge of tensors, experts, layers, routing or
 * models. It executes what it is given. */
typedef unsigned int (*coli_xdna_fn_abi_version)(void);
/* device + runtime + artifact program, for one (M,K,N). */
typedef int          (*coli_xdna_fn_open)(const char *xclbin, const char *insts,
                                          uint32_t m, uint32_t k, uint32_t n);
/* wrap caller-owned prepared BF16 through the qualified userptr path. */
typedef int          (*coli_xdna_fn_wrap_weight)(void *bf16, uint64_t bytes);
/* blocking: submit, wait, copy the artifact-M output out. */
typedef int          (*coli_xdna_fn_execute)(const void *a_bf16, uint64_t a_bytes,
                                             void *c_f32, uint64_t c_bytes);
typedef int          (*coli_xdna_fn_release_weight)(void);
typedef void         (*coli_xdna_fn_shutdown)(void);
typedef const char  *(*coli_xdna_fn_last_error)(void);

static struct {
    ColiXdnaBinding state;
    int             load_attempts;
    char            path[1024];      /* test override; empty = default lookup */
#ifdef _WIN32
    HMODULE         dll;
#endif
    coli_xdna_fn_abi_version    abi_version;
    coli_xdna_fn_open           open;
    coli_xdna_fn_wrap_weight    wrap_weight;
    coli_xdna_fn_execute        execute;
    coli_xdna_fn_release_weight release_weight;
    coli_xdna_fn_shutdown       shutdown;
    coli_xdna_fn_last_error     last_error;
} g_xdna;

/* Drop every callable pointer. Always runs BEFORE the module is released, so a
 * rejected or shut-down helper can never leave a pointer into an unmapped
 * module behind -- the all-or-nothing half of the binding contract. */
static void coli_xdna_clear_entry_points(void){
    g_xdna.abi_version    = NULL;
    g_xdna.open           = NULL;
    g_xdna.wrap_weight    = NULL;
    g_xdna.execute        = NULL;
    g_xdna.release_weight = NULL;
    g_xdna.shutdown       = NULL;
    g_xdna.last_error     = NULL;
}

/* Every helper call goes through one of these. They are the only places a
 * helper pointer is dereferenced, and each refuses when unbound rather than
 * trusting a caller to have checked. */
static int coli_xdna_helper_open_call(const char *xb, const char *ib,
                                      unsigned m, unsigned k, unsigned n){
    if(!g_xdna.open) return -1;
    return g_xdna.open(xb, ib, (uint32_t)m, (uint32_t)k, (uint32_t)n);
}
static int coli_xdna_helper_wrap_call(void *p, uint64_t bytes){
    if(!g_xdna.wrap_weight) return -1;
    return g_xdna.wrap_weight(p, bytes);
}
static int coli_xdna_helper_execute_call(const void *a, uint64_t ab, void *c, uint64_t cb){
    if(!g_xdna.execute) return -1;
    return g_xdna.execute(a, ab, c, cb);
}
static int coli_xdna_helper_release_weight_call(void){
    if(!g_xdna.release_weight) return -1;
    return g_xdna.release_weight();
}
static void coli_xdna_helper_shutdown_call(void){
    if(g_xdna.shutdown) g_xdna.shutdown();
}

const char *coli_xdna_binding_text(ColiXdnaBinding state){
    switch(state){
        case COLI_XDNA_UNPROBED:          return "UNPROBED";
        case COLI_XDNA_AVAILABLE:         return "AVAILABLE";
        case COLI_XDNA_ABSENT:            return "ABSENT";
        case COLI_XDNA_LOAD_FAILED:       return "LOAD_FAILED";
        case COLI_XDNA_ABI_INCOMPATIBLE:  return "ABI_INCOMPATIBLE";
        case COLI_XDNA_SYMBOL_INCOMPLETE: return "SYMBOL_INCOMPLETE";
    }
    return "UNKNOWN";
}

#ifdef _WIN32

/* Where the helper is looked for when no test override is set: beside the
 * running executable, by absolute path. Deliberately NOT a search: no PATH, no
 * current directory, no application-directory fallback. W1 established that an
 * already-loaded module with the same basename wins over a later absolute-path
 * load, so a search order here would be both unsafe and unpredictable. A helper
 * that is not where we look is simply ABSENT. */
static int coli_xdna_default_helper_path(char *out, size_t cap){
    char exe[512];
    DWORD n = GetModuleFileNameA(NULL, exe, (DWORD)sizeof exe);
    if(n == 0 || n >= sizeof exe) return 0;
    char *slash = strrchr(exe, '\\');
    char *fwd   = strrchr(exe, '/');
    if(fwd && (!slash || fwd > slash)) slash = fwd;
    if(!slash) return 0;
    *slash = '\0';
    int written = snprintf(out, cap, "%s\\%s", exe, COLI_XDNA_HELPER_DLL);
    return written > 0 && (size_t)written < cap;
}

static void coli_xdna_probe(void){
    char resolved[512];
    const char *path = g_xdna.path[0] ? g_xdna.path : NULL;

    if(!path){
        if(!coli_xdna_default_helper_path(resolved, sizeof resolved)){
            g_xdna.state = COLI_XDNA_ABSENT;
            return;
        }
        path = resolved;
    }

    /* Absence is not a load failure, and the difference is worth keeping: one
     * means "this build simply has no optional lane here", the other means "a
     * helper is installed but something about it is wrong". */
    if(GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES){
        g_xdna.state = COLI_XDNA_ABSENT;
        return;
    }

    g_xdna.load_attempts++;
    /* LOAD_WITH_ALTERED_SEARCH_PATH lets the helper's own directory satisfy the
     * helper's dependencies (it will eventually need the XRT runtime beside it)
     * without widening the search for the helper itself, which we located by
     * absolute path above. */
    HMODULE dll = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if(!dll){
        g_xdna.state = COLI_XDNA_LOAD_FAILED;
        return;
    }

    /* Resolve into locals: nothing becomes callable until the whole set has
     * validated and the ABI generation matches. */
#if defined(__GNUC__)
    /* GetProcAddress returns FARPROC; casting to the exact exported signature
     * is the standard LoadLibrary idiom, same as backend_loader.c's RESOLVE. */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    coli_xdna_fn_abi_version abi_version =
        (coli_xdna_fn_abi_version)GetProcAddress(dll, "coli_xdna_helper_abi_version");
    coli_xdna_fn_open open_fn =
        (coli_xdna_fn_open)GetProcAddress(dll, "coli_xdna_helper_open");
    coli_xdna_fn_wrap_weight wrap_fn =
        (coli_xdna_fn_wrap_weight)GetProcAddress(dll, "coli_xdna_helper_wrap_weight");
    coli_xdna_fn_execute exec_fn =
        (coli_xdna_fn_execute)GetProcAddress(dll, "coli_xdna_helper_execute");
    coli_xdna_fn_release_weight relw_fn =
        (coli_xdna_fn_release_weight)GetProcAddress(dll, "coli_xdna_helper_release_weight");
    coli_xdna_fn_shutdown shutdown =
        (coli_xdna_fn_shutdown)GetProcAddress(dll, "coli_xdna_helper_shutdown");
    coli_xdna_fn_last_error lasterr_fn =
        (coli_xdna_fn_last_error)GetProcAddress(dll, "coli_xdna_helper_last_error");
#if defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

    /* The version handshake comes first: a helper from a different generation
     * may well export a symbol of the same name with a different meaning, so
     * the missing-symbol check below is only meaningful once generations agree.
     * A helper that cannot even report its generation is incompatible, not
     * incomplete -- there is no version to compare. */
    if(!abi_version || abi_version() != COLI_XDNA_ABI_VERSION){
        FreeLibrary(dll);
        g_xdna.state = COLI_XDNA_ABI_INCOMPATIBLE;
        return;
    }
    /* All-or-nothing: one missing entry point rejects the whole helper. A
     * partially bound helper would report availability and then fail somewhere
     * further in, where the failure is far harder to attribute. */
    if(!open_fn || !wrap_fn || !exec_fn || !relw_fn || !shutdown || !lasterr_fn){
        FreeLibrary(dll);
        g_xdna.state = COLI_XDNA_SYMBOL_INCOMPLETE;
        return;
    }

    g_xdna.dll            = dll;
    g_xdna.abi_version    = abi_version;
    g_xdna.open           = open_fn;
    g_xdna.wrap_weight    = wrap_fn;
    g_xdna.execute        = exec_fn;
    g_xdna.release_weight = relw_fn;
    g_xdna.shutdown       = shutdown;
    g_xdna.last_error     = lasterr_fn;
    g_xdna.state          = COLI_XDNA_AVAILABLE;
}

void coli_xdna_shutdown(void){
    /* Release lane state (which itself calls the helper) BEFORE the module goes
     * away: no helper-owned object may outlive the module that created it, and
     * no host pointer into it may survive the FreeLibrary. */
    coli_xdna_execution_shutdown();
    coli_xdna_clear_entry_points();
    if(g_xdna.dll){
        FreeLibrary(g_xdna.dll);
        g_xdna.dll = NULL;      /* released exactly once; repeat calls are no-ops */
    }
    /* The verdict is deliberately NOT reset to UNPROBED: shutting the lane down
     * is not a reason to re-probe it later in the same process. */
}

#else  /* !_WIN32 */

/* The qualified XDNA lane is Windows/XDNA2-specific (see the N6 architecture
 * freeze). Elsewhere the loader compiles and answers ABSENT, so the rest of the
 * engine needs no platform branches and the contract stays uniform. */
static void coli_xdna_probe(void){
    g_xdna.state = COLI_XDNA_ABSENT;
}

void coli_xdna_shutdown(void){
    coli_xdna_execution_shutdown();
    coli_xdna_clear_entry_points();
}

#endif /* _WIN32 */

ColiXdnaBinding coli_xdna_binding(void){
    /* Sticky: probe at most once per path. Everything after the first call is a
     * branch, which is what keeps a machine with no helper from paying a module
     * load on every candidate operation. */
    if(g_xdna.state == COLI_XDNA_UNPROBED) coli_xdna_probe();
    return g_xdna.state;
}

void coli_xdna_test_set_helper_path(const char *path){
    coli_xdna_shutdown();
    g_xdna.state         = COLI_XDNA_UNPROBED;
    g_xdna.load_attempts = 0;
    if(path && *path){
        snprintf(g_xdna.path, sizeof g_xdna.path, "%s", path);
    } else {
        g_xdna.path[0] = '\0';
    }
}

ColiXdnaBinding coli_xdna_test_state(void){ return g_xdna.state; }
int coli_xdna_test_load_attempts(void){ return g_xdna.load_attempts; }
int coli_xdna_test_entry_points_bound(void){
    return g_xdna.abi_version != NULL && g_xdna.shutdown != NULL;
}

/* ======================================================================
 * SHA-256 (FIPS 180-4)
 *
 * Self-contained on purpose. BCrypt is available to the Windows toolchain, but
 * using it would add -lbcrypt to the host link line for a lane the host does
 * not even link yet, and would fork this file's otherwise portable registry
 * logic along a platform boundary it has no other reason to have. ~90 lines of
 * standard SHA-256, covered by the NIST vectors in the registry test, buys
 * portability and zero new dependencies.
 * ====================================================================== */

static uint32_t coli_sha_ror(uint32_t x, int n){ return (x >> n) | (x << (32 - n)); }

static void coli_sha256_block(uint32_t h[8], const unsigned char *p){
    static const uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
        0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
        0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
        0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
        0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
        0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u };
    uint32_t w[64], a,b,c,d,e,f,g,hh;
    for(int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16)
             | ((uint32_t)p[i*4+2] << 8) | (uint32_t)p[i*4+3];
    for(int i = 16; i < 64; i++){
        uint32_t s0 = coli_sha_ror(w[i-15],7) ^ coli_sha_ror(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = coli_sha_ror(w[i-2],17) ^ coli_sha_ror(w[i-2],19)  ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=h[0]; b=h[1]; c=h[2]; d=h[3]; e=h[4]; f=h[5]; g=h[6]; hh=h[7];
    for(int i = 0; i < 64; i++){
        uint32_t S1 = coli_sha_ror(e,6) ^ coli_sha_ror(e,11) ^ coli_sha_ror(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        uint32_t S0 = coli_sha_ror(a,2) ^ coli_sha_ror(a,13) ^ coli_sha_ror(a,22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

typedef struct { uint32_t h[8]; unsigned char buf[64]; size_t n; uint64_t bits; } ColiSha;

static void coli_sha256_init(ColiSha *s){
    s->h[0]=0x6a09e667u; s->h[1]=0xbb67ae85u; s->h[2]=0x3c6ef372u; s->h[3]=0xa54ff53au;
    s->h[4]=0x510e527fu; s->h[5]=0x9b05688cu; s->h[6]=0x1f83d9abu; s->h[7]=0x5be0cd19u;
    s->n = 0; s->bits = 0;
}
static void coli_sha256_update(ColiSha *s, const unsigned char *p, size_t len){
    s->bits += (uint64_t)len * 8u;
    while(len){
        size_t take = 64 - s->n;
        if(take > len) take = len;
        memcpy(s->buf + s->n, p, take);
        s->n += take; p += take; len -= take;
        if(s->n == 64){ coli_sha256_block(s->h, s->buf); s->n = 0; }
    }
}
static void coli_sha256_final(ColiSha *s, unsigned char out[32]){
    uint64_t bits = s->bits;
    unsigned char pad = 0x80;
    coli_sha256_update(s, &pad, 1);
    s->bits = bits;                       /* padding is not message length */
    unsigned char zero = 0;
    while(s->n != 56){ coli_sha256_update(s, &zero, 1); s->bits = bits; }
    unsigned char len[8];
    for(int i = 0; i < 8; i++) len[i] = (unsigned char)(bits >> (56 - i*8));
    coli_sha256_update(s, len, 8);
    for(int i = 0; i < 8; i++){
        out[i*4]   = (unsigned char)(s->h[i] >> 24);
        out[i*4+1] = (unsigned char)(s->h[i] >> 16);
        out[i*4+2] = (unsigned char)(s->h[i] >> 8);
        out[i*4+3] = (unsigned char)(s->h[i]);
    }
}

void coli_xdna_sha256(const void *data, size_t len, unsigned char out[32]){
    ColiSha s; coli_sha256_init(&s);
    coli_sha256_update(&s, (const unsigned char *)data, len);
    coli_sha256_final(&s, out);
}

int coli_xdna_sha256_file(const char *path, unsigned char out[32]){
    FILE *f = fopen(path, "rb");
    if(!f) return 0;
    ColiSha s; coli_sha256_init(&s);
    unsigned char buf[65536];
    size_t got;
    while((got = fread(buf, 1, sizeof buf, f)) > 0) coli_sha256_update(&s, buf, got);
    int ok = !ferror(f);
    fclose(f);
    if(!ok) return 0;
    coli_sha256_final(&s, out);
    return 1;
}

/* ======================================================================
 * Artifact registry
 * ====================================================================== */

/* Production rows.
 *
 * Only the first qualified family is present: the MoE shared-expert gate/up
 * projection, K=6144 -> N=2048, at the two M buckets the research programme
 * actually qualified. Other shape families were qualified as research artifacts
 * but are deliberately absent -- V1 scope is one unconditional family.
 *
 * Every hash below was recovered from sealed research evidence and recomputed
 * from the artifact bytes; the qualification flags come from a per-artifact
 * eligibility table, not from "it compiled". Filenames are LOGICAL: they are
 * resolved under a caller-supplied root, and no build path appears here.
 *
 * The artifacts themselves are not shipped by this build. A row describing an
 * absent file is the intended state and yields ARTIFACT_UNAVAILABLE. */
static const ColiXdnaArtifact g_xdna_production_rows[] = {
    {
        COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, "F3",
        64, 6144, 2048,
        COLI_XDNA_DT_BF16, COLI_XDNA_DT_BF16, COLI_XDNA_DT_F32,
        COLI_XDNA_TARGET_XDNA2,
        "wa_F3_M64_K6144_N2048.xclbin",
        "59309a70af9fd45ec5ed9467661332e08b9440276cf41afccecf439d34f0a4de",
        "wa_F3_M64_K6144_N2048_insts.bin",
        "bde093ac7bcf2df39159376bb11f6f631be3f5fdf6dac6219ce177bd3c80fc97",
        1, 1, 1, 1
    },
    {
        COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, "F3",
        256, 6144, 2048,
        COLI_XDNA_DT_BF16, COLI_XDNA_DT_BF16, COLI_XDNA_DT_F32,
        COLI_XDNA_TARGET_XDNA2,
        "wa_F3_M256_K6144_N2048.xclbin",
        "e49c34388efea654eb63d006c4024ffe0e0ff763d62c3f769c0362ac3bb17263",
        "wa_F3_M256_K6144_N2048_insts.bin",
        "1a17753ce2516d8ccd7b992b61e1aa7c6b7a9fb655c4c0c0179eff075130d4b1",
        1, 1, 1, 1
    }
};
static const int g_xdna_production_count =
    (int)(sizeof g_xdna_production_rows / sizeof g_xdna_production_rows[0]);

static const ColiXdnaArtifact *g_rows  = g_xdna_production_rows;
/* Initialised to the production count, NOT to 0. Zero here would leave the
 * production table installed but empty, so every production lookup would answer
 * UNKNOWN_FAMILY until a test happened to install a registry -- invisible while
 * the only callers were tests that always install one, and wrong the moment a
 * production caller arrived. I5 is that first production caller. */
static int                     g_nrows =
    (int)(sizeof g_xdna_production_rows / sizeof g_xdna_production_rows[0]);

const char *coli_xdna_static_text(ColiXdnaStatic s){
    switch(s){
        case COLI_XDNA_STATIC_QUALIFIED:                 return "STATIC_ARTIFACT_QUALIFIED";
        case COLI_XDNA_STATIC_HELPER_UNAVAILABLE:        return "HELPER_UNAVAILABLE";
        case COLI_XDNA_STATIC_UNKNOWN_FAMILY:            return "UNKNOWN_FAMILY";
        case COLI_XDNA_STATIC_UNSUPPORTED_SHAPE_OR_BUCKET:return "UNSUPPORTED_SHAPE_OR_BUCKET";
        case COLI_XDNA_STATIC_UNSUPPORTED_FORMAT:        return "UNSUPPORTED_FORMAT";
        case COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE:      return "ARTIFACT_UNAVAILABLE";
        case COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED: return "ARTIFACT_INTEGRITY_FAILED";
        case COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED:      return "ARTIFACT_UNQUALIFIED";
        case COLI_XDNA_STATIC_REGISTRY_INVALID:          return "REGISTRY_INVALID";
    }
    return "UNKNOWN";
}

static int coli_xdna_dtype_ok(ColiXdnaDtype d){
    return d == COLI_XDNA_DT_BF16 || d == COLI_XDNA_DT_F32;
}

static int coli_xdna_sha_text_ok(const char *s){
    if(!s) return 0;
    int i = 0;
    for(; s[i]; i++){
        char c = s[i];
        if(!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 0;
    }
    return i == 64;
}

/* A logical artifact name must stay inside the supplied root. Rejecting the
 * traversal at VALIDATION rather than at open() means a malformed registry can
 * never reach the filesystem at all. */
static int coli_xdna_name_ok(const char *s){
    if(!s || !*s) return 0;
    if(s[0] == '/' || s[0] == '\\') return 0;                 /* rooted */
    if(s[0] && s[1] == ':') return 0;                          /* drive-qualified */
    for(const char *p = s; *p; p++){
        if(p[0] == '.' && p[1] == '.') return 0;               /* any .. segment */
    }
    return 1;
}

static int coli_xdna_same_key(const ColiXdnaArtifact *a, const ColiXdnaArtifact *b){
    return a->family == b->family && a->artifact_m == b->artifact_m
        && a->k == b->k && a->n == b->n
        && a->in_dtype == b->in_dtype && a->weight_dtype == b->weight_dtype
        && a->out_dtype == b->out_dtype;
}

int coli_xdna_registry_validate(void){
    for(int i = 0; i < g_nrows; i++){
        const ColiXdnaArtifact *r = &g_rows[i];
        if(r->family == COLI_XDNA_FAMILY_NONE) return 0;
        if(!r->research_family || !*r->research_family) return 0;
        if(r->artifact_m == 0 || r->k == 0 || r->n == 0) return 0;
        if(!coli_xdna_dtype_ok(r->in_dtype))     return 0;
        if(!coli_xdna_dtype_ok(r->weight_dtype)) return 0;
        if(!coli_xdna_dtype_ok(r->out_dtype))    return 0;
        if(r->target != COLI_XDNA_TARGET_XDNA2)  return 0;
        if(!coli_xdna_name_ok(r->xclbin_name))   return 0;
        if(!coli_xdna_name_ok(r->insts_name))    return 0;
        if(!coli_xdna_sha_text_ok(r->xclbin_sha256)) return 0;
        if(!coli_xdna_sha_text_ok(r->insts_sha256))  return 0;
        /* No silent first-match ambiguity: two rows may never answer one key. */
        for(int j = 0; j < i; j++)
            if(coli_xdna_same_key(r, &g_rows[j])) return 0;
    }
    return 1;
}

const ColiXdnaArtifact *coli_xdna_registry_lookup(const ColiXdnaRequest *q){
    if(!q || q->family == COLI_XDNA_FAMILY_NONE) return NULL;
    for(int i = 0; i < g_nrows; i++){
        const ColiXdnaArtifact *r = &g_rows[i];
        /* The full key. Shape alone is never enough: a different operation with
         * the same M/K/N must not inherit this artifact's qualification. The
         * bucket is matched exactly -- research qualified specific M values, and
         * nothing here extrapolates between them. */
        if(r->family == q->family && r->artifact_m == q->m
           && r->k == q->k && r->n == q->n
           && r->in_dtype == q->in_dtype && r->weight_dtype == q->weight_dtype
           && r->out_dtype == q->out_dtype)
            return r;
    }
    return NULL;
}

static int coli_xdna_join(char *out, size_t cap, const char *root, const char *name){
    int written = snprintf(out, cap, "%s/%s", root, name);
    return written > 0 && (size_t)written < cap;
}

/* Both components must exist and hash exactly. Presence and integrity are
 * separate verdicts because they mean different things operationally: one is
 * "this build does not ship that artifact", the other is "the bytes are not the
 * bytes that were qualified". */
static ColiXdnaStatic coli_xdna_check_file(const char *root, const char *name,
                                           const char *expect_hex){
    char path[2048];
    if(!coli_xdna_join(path, sizeof path, root, name))
        return COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE;
    unsigned char got[32];
    if(!coli_xdna_sha256_file(path, got))
        return COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE;
    static const char *H = "0123456789abcdef";
    char hex[65];
    for(int i = 0; i < 32; i++){ hex[i*2] = H[got[i] >> 4]; hex[i*2+1] = H[got[i] & 15]; }
    hex[64] = '\0';
    if(strcmp(hex, expect_hex) != 0)
        return COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED;
    return COLI_XDNA_STATIC_QUALIFIED;
}

ColiXdnaStatic coli_xdna_artifact_status(const ColiXdnaRequest *q, const char *root){
    if(!coli_xdna_registry_validate()) return COLI_XDNA_STATIC_REGISTRY_INVALID;

    if(!q) return COLI_XDNA_STATIC_UNKNOWN_FAMILY;
    if(q->family == COLI_XDNA_FAMILY_NONE) return COLI_XDNA_STATIC_UNKNOWN_FAMILY;
    if(!coli_xdna_dtype_ok(q->in_dtype) || !coli_xdna_dtype_ok(q->weight_dtype)
       || !coli_xdna_dtype_ok(q->out_dtype))
        return COLI_XDNA_STATIC_UNSUPPORTED_FORMAT;

    const ColiXdnaArtifact *a = coli_xdna_registry_lookup(q);
    if(!a){
        /* Distinguish "we have never qualified this operation" from "we have,
         * but not at this shape or bucket": the second is actionable, the first
         * is not. */
        for(int i = 0; i < g_nrows; i++)
            if(g_rows[i].family == q->family)
                return COLI_XDNA_STATIC_UNSUPPORTED_SHAPE_OR_BUCKET;
        return COLI_XDNA_STATIC_UNKNOWN_FAMILY;
    }

    /* Research qualification is checked BEFORE the filesystem: a row that was
     * never correctness-qualified must decline whether or not its bytes are
     * present and intact. Compiling and dispatching successfully is not
     * evidence of correctness -- N6 measured a design that did both and was
     * numerically wrong. */
    if(!a->runtime_weight_qualified || !a->correctness_qualified
       || !a->userptr_qualified || !a->structural_qualified)
        return COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED;

    if(!root || !*root) return COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE;

    ColiXdnaStatic s = coli_xdna_check_file(root, a->xclbin_name, a->xclbin_sha256);
    if(s != COLI_XDNA_STATIC_QUALIFIED) return s;
    return coli_xdna_check_file(root, a->insts_name, a->insts_sha256);
}

ColiXdnaStatic coli_xdna_static_eligibility(const ColiXdnaRequest *q, const char *root){
    if(coli_xdna_binding() != COLI_XDNA_AVAILABLE)
        return COLI_XDNA_STATIC_HELPER_UNAVAILABLE;
    return coli_xdna_artifact_status(q, root);
}

void coli_xdna_test_set_registry(const ColiXdnaArtifact *rows, int count){
    if(rows && count > 0){ g_rows = rows;                    g_nrows = count; }
    else                 { g_rows = g_xdna_production_rows;  g_nrows = g_xdna_production_count; }
}

/* ======================================================================
 * Prepared host state
 *
 * Engine-owned, derived from the authoritative fmt=4 tensor and never a
 * replacement for it. The helper neither allocates nor frees any of this.
 * ====================================================================== */

struct ColiXdnaPrepared {
    ColiXdnaPrepState state;
    void            *base;      /* 4096-aligned, from posix_memalign */
    size_t           bytes;     /* logical payload: k*n*sizeof(bf16) */
    size_t           capacity;  /* what base can actually hold */
    unsigned         k, n;
    unsigned long long generation;   /* bumped on every successful publication */
};

/* Engine-side host accounting. Deliberately counts LOGICAL payload bytes: the
 * allocator's own bookkeeping overhead is real but not knowable from here, and
 * reporting a number we cannot substantiate would be worse than reporting the
 * one we can. */
static size_t g_xdna_prepared_bytes;
static int    g_xdna_prepared_objects;

/* Defined with the execution lane below, declared here because the prepared
 * buffer owner must be able to tell the lane to let go of memory it is about
 * to free or replace. */
static void coli_xdna_lane_forget_pointer(const void *p);

const char *coli_xdna_prep_text(ColiXdnaPrepState s){
    switch(s){
        case COLI_XDNA_PREP_UNPREPARED: return "UNPREPARED";
        case COLI_XDNA_PREP_PREPARING:  return "PREPARING";
        case COLI_XDNA_PREP_VALID:      return "PREPARED_VALID";
        case COLI_XDNA_PREP_INVALID:    return "PREPARED_INVALID";
    }
    return "UNKNOWN";
}

int coli_xdna_pointer_alignment_ok(const void *p){
    if(!p) return 0;
    return ((uintptr_t)p % (uintptr_t)COLI_XDNA_PREPARED_ALIGNMENT) == 0;
}

ColiXdnaPrepared *coli_xdna_prepared_create(void){
    ColiXdnaPrepared *p = (ColiXdnaPrepared *)calloc(1, sizeof *p);
    if(!p) return NULL;
    p->state = COLI_XDNA_PREP_UNPREPARED;
    g_xdna_prepared_objects++;
    return p;
}

/* Drop capacity only. Validity and allocation lifetime are separate concerns,
 * so they get separate operations. */
void coli_xdna_prepared_free_buffer(ColiXdnaPrepared *p){
    if(!p) return;
    /* Tell the lane before the memory goes away: a helper-owned userptr wrapper
     * borrowing this allocation must never outlive it. */
    coli_xdna_lane_forget_pointer(p->base);
    if(p->base){
        compat_aligned_free(p->base);       /* MUST pair with posix_memalign */
        p->base = NULL;
    }
    if(g_xdna_prepared_bytes >= p->bytes) g_xdna_prepared_bytes -= p->bytes;
    else                                  g_xdna_prepared_bytes = 0;
    p->bytes = 0; p->capacity = 0; p->k = 0; p->n = 0;
    p->state = COLI_XDNA_PREP_UNPREPARED;
}

void coli_xdna_prepared_release(ColiXdnaPrepared **pp){
    if(!pp || !*pp) return;
    coli_xdna_prepared_free_buffer(*pp);
    free(*pp);
    *pp = NULL;                              /* repeated release is a no-op */
    if(g_xdna_prepared_objects > 0) g_xdna_prepared_objects--;
}

/* k*n*2 with checked arithmetic. An unchecked product would wrap and reach the
 * allocator as a small, plausible number, which is the worst possible outcome:
 * a successful allocation far too small for what the caller will write. */
static int coli_xdna_payload_bytes(unsigned k, unsigned n, ColiXdnaDtype dt, size_t *out){
    if(k == 0 || n == 0) return 0;
    if(dt != COLI_XDNA_DT_BF16) return 0;    /* only the qualified prepared dtype */
    size_t kk = (size_t)k, nn = (size_t)n;
    if(kk > SIZE_MAX / nn) return 0;
    size_t elems = kk * nn;
    if(elems > SIZE_MAX / 2u) return 0;
    *out = elems * 2u;                       /* sizeof(bf16) */
    return 1;
}

int coli_xdna_prepare_begin(ColiXdnaPrepared *p, unsigned k, unsigned n,
                            ColiXdnaDtype prepared_dtype){
    if(!p) return 0;
    size_t need = 0;
    if(!coli_xdna_payload_bytes(k, n, prepared_dtype, &need)) return 0;

    /* Reuse retained capacity when it is genuinely large enough. The old bytes
     * stay physically present and are NOT semantically valid: the state is
     * PREPARING until a producer publishes success over them. */
    if(!p->base || p->capacity < need){
        void *fresh = NULL;
        if(p->base){
            compat_aligned_free(p->base);
            p->base = NULL;
            if(g_xdna_prepared_bytes >= p->bytes) g_xdna_prepared_bytes -= p->bytes;
            else                                  g_xdna_prepared_bytes = 0;
            p->bytes = 0; p->capacity = 0;
        }
        if(posix_memalign(&fresh, COLI_XDNA_PREPARED_ALIGNMENT, need) != 0 || !fresh){
            p->state = COLI_XDNA_PREP_UNPREPARED;
            return 0;
        }
        /* The allocator is contracted to align, but the contract is re-checked:
         * an unaligned buffer must never travel further than this line. */
        if(!coli_xdna_pointer_alignment_ok(fresh)){
            compat_aligned_free(fresh);
            p->state = COLI_XDNA_PREP_UNPREPARED;
            return 0;
        }
        p->base = fresh;
        p->capacity = need;
    } else {
        if(g_xdna_prepared_bytes >= p->bytes) g_xdna_prepared_bytes -= p->bytes;
        else                                  g_xdna_prepared_bytes = 0;
    }

    p->bytes = need;                         /* logical payload, never rounded */
    p->k = k; p->n = n;
    g_xdna_prepared_bytes += need;
    p->state = COLI_XDNA_PREP_PREPARING;
    return 1;
}

void *coli_xdna_prepare_dest(ColiXdnaPrepared *p){
    /* Only a producer inside an open cycle may write. A published image is not
     * writable, so it cannot be corrupted while something believes it valid. */
    if(!p || p->state != COLI_XDNA_PREP_PREPARING) return NULL;
    return p->base;
}

int coli_xdna_prepare_publish_success(ColiXdnaPrepared *p){
    if(!p) return 0;
    if(p->state != COLI_XDNA_PREP_PREPARING) return 0;   /* no shortcut to VALID */
    if(!p->base || !coli_xdna_pointer_alignment_ok(p->base)) return 0;
    if(p->bytes == 0 || p->capacity < p->bytes) return 0;
    p->state = COLI_XDNA_PREP_VALID;
    /* A new publication is a new image even when it lands on the same address:
     * retained capacity is reused in place. Anything caching a view of these
     * bytes keys on (pointer, generation), never on the pointer alone. */
    p->generation++;
    return 1;
}

void coli_xdna_prepare_publish_failure(ColiXdnaPrepared *p){
    if(!p) return;
    if(p->state != COLI_XDNA_PREP_PREPARING) return;
    /* Capacity is retained and still accounted; only the contents are revoked. */
    p->state = COLI_XDNA_PREP_INVALID;
}

int coli_xdna_prepared_invalidate(ColiXdnaPrepared *p){
    if(!p) return 0;
    if(p->state != COLI_XDNA_PREP_VALID) return 0;
    p->state = COLI_XDNA_PREP_INVALID;
    /* The image is no longer authoritative, so no device-side view of it may
     * remain current either. */
    coli_xdna_lane_forget_pointer(p->base);
    return 1;
}

ColiXdnaPrepState coli_xdna_prepared_state(const ColiXdnaPrepared *p){
    return p ? p->state : COLI_XDNA_PREP_UNPREPARED;
}
size_t   coli_xdna_prepared_bytes(const ColiXdnaPrepared *p){ return p ? p->bytes : 0; }
unsigned coli_xdna_prepared_k(const ColiXdnaPrepared *p){ return p ? p->k : 0u; }
unsigned coli_xdna_prepared_n(const ColiXdnaPrepared *p){ return p ? p->n : 0u; }
unsigned long long coli_xdna_prepared_generation(const ColiXdnaPrepared *p){
    return p ? p->generation : 0ull;
}

size_t coli_xdna_prepared_total_bytes(void){ return g_xdna_prepared_bytes; }
int    coli_xdna_prepared_live_objects(void){ return g_xdna_prepared_objects; }

const void *coli_xdna_prepared_image(const ColiXdnaPrepared *p){
    /* Only a published image is readable. An unpublished or revoked one returns
     * NULL rather than a pointer to bytes with no authority. */
    if(!p || p->state != COLI_XDNA_PREP_VALID) return NULL;
    return p->base;
}
const void *coli_xdna_prepared_image_unchecked(const ColiXdnaPrepared *p){
    return p ? p->base : NULL;
}

/* ======================================================================
 * fmt=4 grouped int4  ->  BF16 B[K,N]
 *
 * Semantics are taken from the production kernel (quant.h matmul_i4_grouped),
 * not from a specification of it:
 *
 *   rb = (I+1)/2                 bytes per output row
 *   ng = (I+gs-1)/gs             groups per output row
 *   row   = q4    + o*rb         weights are stored [O][I]
 *   scl   = scale + o*ng
 *   byte  = row[i>>1]
 *   nib   = (i&1) ? byte>>4 : byte&0x0F      even i low, odd i high
 *   value = (nib - 8) * scl[i/gs]
 *
 * The destination is BF16 B[K,N] with K=I and N=O, so the write is
 * dst[i*O + o] -- the [O,I] -> [I,O] transform is inherent in the addressing
 * and needs no intermediate matrix. Values are converted one at a time through
 * a float scalar, so no full-sized FP32 image is ever allocated.
 * ====================================================================== */

/* Test-only. Production never sets this; there is no env var or flag for it. */
static int g_xdna_convert_fail_pct;
/* How many conversions have been performed. I1-I3 pinned this at zero because
 * no converter existed; from I4 it counts real work, which is what makes
 * "conversion happened" and "device work happened" separately assertable. */
static int g_xdna_conversions;
void coli_xdna_test_set_convert_fail_pct(int pct){
    g_xdna_convert_fail_pct = (pct > 0 && pct < 100) ? pct : 0;
}

const char *coli_xdna_prep_result_text(ColiXdnaPrepResult r){
    switch(r){
        case COLI_XDNA_PREP_OK:                    return "OK";
        case COLI_XDNA_PREP_ERR_UNSUPPORTED_FORMAT:return "UNSUPPORTED_SOURCE_FORMAT";
        case COLI_XDNA_PREP_ERR_INVALID_SOURCE:    return "INVALID_SOURCE";
        case COLI_XDNA_PREP_ERR_SIZE:              return "SIZE_OVERFLOW";
        case COLI_XDNA_PREP_ERR_ALLOC:             return "ALLOCATION_FAILED";
        case COLI_XDNA_PREP_ERR_FAILED:            return "PREPARATION_FAILED";
        case COLI_XDNA_PREP_ERR_STATE:             return "STATE_ERROR";
    }
    return "UNKNOWN";
}

/* float -> bfloat16, round to nearest even. Truncation would be a different
 * function and a different image; the qualified oracle rounds. */
static unsigned short coli_xdna_f2b(float f){
    unsigned int u;
    memcpy(&u, &f, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

ColiXdnaPrepResult coli_xdna_prepare_from_fmt4(ColiXdnaPrepared *p,
                                               int fmt,
                                               const unsigned char *q4,
                                               const float *scale,
                                               int I, int O, int gs){
    if(!p) return COLI_XDNA_PREP_ERR_STATE;

    /* Everything that can be rejected is rejected BEFORE the cycle opens, so a
     * bad source never leaves an object stranded in PREPARING or INVALID. */
    if(fmt != 4) return COLI_XDNA_PREP_ERR_UNSUPPORTED_FORMAT;
    if(!q4 || !scale) return COLI_XDNA_PREP_ERR_INVALID_SOURCE;
    if(I <= 0 || O <= 0 || gs <= 0) return COLI_XDNA_PREP_ERR_INVALID_SOURCE;

    size_t need = 0;
    if(!coli_xdna_payload_bytes((unsigned)I, (unsigned)O, COLI_XDNA_DT_BF16, &need))
        return COLI_XDNA_PREP_ERR_SIZE;

    if(!coli_xdna_prepare_begin(p, (unsigned)I, (unsigned)O, COLI_XDNA_DT_BF16))
        return COLI_XDNA_PREP_ERR_ALLOC;

    unsigned short *dst = (unsigned short *)coli_xdna_prepare_dest(p);
    if(!dst){                                    /* cannot happen; never leave PREPARING */
        coli_xdna_prepare_publish_failure(p);
        return COLI_XDNA_PREP_ERR_STATE;
    }

    const size_t rb = ((size_t)I + 1u) / 2u;
    const size_t ng = ((size_t)I + (size_t)gs - 1u) / (size_t)gs;
    const int fail_after = g_xdna_convert_fail_pct
                         ? (int)(((long)O * g_xdna_convert_fail_pct) / 100) : -1;

    for(int o = 0; o < O; o++){
        if(fail_after >= 0 && o == fail_after){
            /* Deterministic mid-conversion failure: rows before this one are
             * genuinely converted, the rest keep whatever they held. */
            coli_xdna_prepare_publish_failure(p);
            return COLI_XDNA_PREP_ERR_FAILED;
        }
        const unsigned char *row = q4 + (size_t)o * rb;
        const float *scl = scale + (size_t)o * ng;
        for(int i = 0; i < I; i++){
            unsigned char byte = row[(size_t)i >> 1];
            int nib = (i & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
            dst[(size_t)i * (size_t)O + (size_t)o] =
                coli_xdna_f2b((float)(nib - 8) * scl[(size_t)i / (size_t)gs]);
        }
    }

    if(!coli_xdna_prepare_publish_success(p)){
        coli_xdna_prepare_publish_failure(p);
        return COLI_XDNA_PREP_ERR_STATE;
    }
    g_xdna_conversions++;
    return COLI_XDNA_PREP_OK;
}

/* ======================================================================
 * Full hard eligibility and native execution  (W2-N7-I5)
 *
 * The engine keeps every decision that requires knowing what the operation
 * MEANS: which family it is, which artifact answers it, whether the bytes are
 * trustworthy, how the weight is represented, whether the request is in range.
 * The helper is handed an already-selected, already-verified artifact and an
 * already-prepared buffer, and executes. It is told nothing about tensors,
 * experts, layers, routing or models, and it decides nothing.
 * ====================================================================== */

const char *coli_xdna_hard_text(ColiXdnaHard h){
    switch(h){
        case COLI_XDNA_HARD_ELIGIBLE:                    return "HARD_ELIGIBLE";
        case COLI_XDNA_HARD_FAMILY_UNSUPPORTED:          return "FAMILY_UNSUPPORTED";
        case COLI_XDNA_HARD_M_OUT_OF_RANGE:              return "M_OUT_OF_RANGE";
        case COLI_XDNA_HARD_SHAPE_UNSUPPORTED:           return "SHAPE_UNSUPPORTED";
        case COLI_XDNA_HARD_FORMAT_UNSUPPORTED:          return "FORMAT_UNSUPPORTED";
        case COLI_XDNA_HARD_GROUP_SIZE_UNSUPPORTED:      return "GROUP_SIZE_UNSUPPORTED";
        case COLI_XDNA_HARD_LAYOUT_UNSUPPORTED:          return "LAYOUT_UNSUPPORTED";
        case COLI_XDNA_HARD_ARTIFACT_UNAVAILABLE:        return "ARTIFACT_UNAVAILABLE";
        case COLI_XDNA_HARD_ARTIFACT_INTEGRITY_FAILED:   return "ARTIFACT_INTEGRITY_FAILED";
        case COLI_XDNA_HARD_ARTIFACT_UNQUALIFIED:        return "ARTIFACT_UNQUALIFIED";
        case COLI_XDNA_HARD_REGISTRY_INVALID:            return "REGISTRY_INVALID";
        case COLI_XDNA_HARD_PREPARED_INVALID:            return "PREPARED_INVALID";
        case COLI_XDNA_HARD_ALIGNMENT_INVALID:           return "ALIGNMENT_INVALID";
        case COLI_XDNA_HARD_HELPER_UNAVAILABLE:          return "HELPER_UNAVAILABLE";
        case COLI_XDNA_HARD_HELPER_ABI_INCOMPATIBLE:     return "HELPER_ABI_INCOMPATIBLE";
        case COLI_XDNA_HARD_DEVICE_UNAVAILABLE:          return "DEVICE_UNAVAILABLE";
        case COLI_XDNA_HARD_ARTIFACT_RUNTIME_UNAVAILABLE:return "ARTIFACT_RUNTIME_UNAVAILABLE";
        case COLI_XDNA_HARD_WEIGHT_WRAP_UNAVAILABLE:     return "WEIGHT_WRAP_UNAVAILABLE";
    }
    return "UNKNOWN";
}

const char *coli_xdna_exec_text(ColiXdnaExec e){
    switch(e){
        case COLI_XDNA_EXEC_OK:                    return "OK";
        case COLI_XDNA_EXEC_DECLINED:              return "DECLINED";
        case COLI_XDNA_EXEC_DEVICE_INIT_FAILED:    return "DEVICE_INIT_FAILED";
        case COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED:  return "ARTIFACT_OPEN_FAILED";
        case COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED: return "WEIGHT_PREPARE_FAILED";
        case COLI_XDNA_EXEC_WEIGHT_WRAP_FAILED:    return "WEIGHT_WRAP_FAILED";
        case COLI_XDNA_EXEC_ACTIVATION_FAILED:     return "ACTIVATION_FAILED";
        case COLI_XDNA_EXEC_EXECUTE_FAILED:        return "EXECUTE_FAILED";
        case COLI_XDNA_EXEC_COMPLETION_FAILED:     return "COMPLETION_FAILED";
    }
    return "UNKNOWN";
}

/* ---- execution-lane state ------------------------------------------------
 *
 * Transient per-operation staging (activation, output) and the identity of the
 * artifact currently open in the helper. The prepared WEIGHT is not here: it
 * belongs to the tensor, which outlives any of this. */
static int          g_xdna_force;                 /* internal test control only */
/* Explicit PRODUCT policy (W2-N7-P1), deliberately distinct from g_xdna_force.
 * Set only from a parsed user request; never from discovery, never by default. */
static int          g_xdna_explicit;
static char         g_xdna_root[1024];            /* artifact root; empty = none */
/* Set only by the test seam. When 0, the root is the PRODUCT one derived from
 * the executable directory. Kept as a separate flag rather than inferred from
 * g_xdna_root being empty, because "a test deliberately set no root" and "no
 * test has spoken, use the product default" are different states and the first
 * must not silently acquire the second. */
static int          g_xdna_root_overridden;
static ColiXdnaHard g_xdna_last_hard = COLI_XDNA_HARD_ELIGIBLE;
static ColiXdnaExec g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED;
static int g_xdna_dispatches, g_xdna_completions, g_xdna_artifact_opens;
static int g_xdna_act_preps, g_xdna_padded_ops, g_xdna_fallbacks;

/* ---- bucket accounting ---------------------------------------------------
 *
 * Two buckets are live, so a single padded_ops total no longer identifies what
 * happened: it cannot say which artifact ran, and it weights a 24-row pad the
 * same as a 191-row one. Everything below is indexed by BUCKET SLOT, and the
 * slot mapping is the one place that knows the bucket set.
 *
 * Slots, not artifact_m values, because the accounting must stay O(buckets)
 * rather than sparse-keyed, and because an unknown artifact_m must be a loud
 * miss rather than a silently created bin. */
enum { COLI_XDNA_SLOT_M64 = 0, COLI_XDNA_SLOT_M256 = 1, COLI_XDNA_SLOTS = 2 };

static int coli_xdna_bucket_slot(unsigned artifact_m){
    if(artifact_m == COLI_XDNA_BUCKET_M_SMALL) return COLI_XDNA_SLOT_M64;
    if(artifact_m == COLI_XDNA_BUCKET_M_LARGE) return COLI_XDNA_SLOT_M256;
    return -1;                                  /* not a compiled bucket */
}

static int       g_xdna_b_hard_eligible[COLI_XDNA_SLOTS];
static int       g_xdna_b_dispatches[COLI_XDNA_SLOTS];
static int       g_xdna_b_completions[COLI_XDNA_SLOTS];
static int       g_xdna_b_padded_ops[COLI_XDNA_SLOTS];
static long long g_xdna_b_padded_rows[COLI_XDNA_SLOTS];
static int       g_xdna_b_artifact_opens[COLI_XDNA_SLOTS];
static int       g_xdna_b_switches[COLI_XDNA_SLOTS][COLI_XDNA_SLOTS];
static int       g_xdna_m_above_range_declines;

/* THE BUCKET SELECTOR.
 *
 * The smallest compiled bucket that holds `logical_m`, or 0 if none does.
 * This is SHAPE ELIGIBILITY, not an economic choice: it has no cost model, no
 * device-load input and no preference. A caller may not override it, and it may
 * not round a request up into a bucket that was never built -- there is no
 * M128 artifact, so an S=65 request runs on M256 with 191 padded rows rather
 * than on something that does not exist.
 *
 * Pure: no globals, no device, no registry. The registry lookup that follows
 * still has to find a row for the returned bucket, so a bucket named here but
 * absent from the registry declines at gate 6 exactly as an unknown shape
 * would. */
static unsigned coli_xdna_bucket_for_logical_m(int logical_m){
    if(logical_m < COLI_XDNA_LOGICAL_M_MIN) return 0u;
    if(logical_m <= (int)COLI_XDNA_BUCKET_M_SMALL) return COLI_XDNA_BUCKET_M_SMALL;
    if(logical_m <= (int)COLI_XDNA_BUCKET_M_LARGE) return COLI_XDNA_BUCKET_M_LARGE;
    return 0u;                                  /* above every bucket: decline */
}
static int g_xdna_device_opens, g_xdna_helper_calls, g_xdna_userptr_wraps;
static int g_xdna_wrapper_reuses, g_xdna_wrapper_releases, g_xdna_stale_rejects;
static int g_xdna_output_valid;
static ColiXdnaLaneHealth g_xdna_lane_health = COLI_XDNA_LANE_UNINITIALIZED;

/* Which artifact the helper currently holds open, so a run of operations on the
 * same shape does not reopen the device, the context or the program. This is a
 * RUNTIME object cache, not an identity cache: it can only ever hold what the
 * engine already selected, and a different selection closes it and reopens. */
static struct {
    int      open;
    unsigned m, k, n;
    /* The wrapped weight, identified by (pointer, generation). The pointer
     * alone is NOT an identity: retained capacity is reused in place, so a
     * different image can occupy the same address. */
    const void        *wrapped;
    unsigned long long wrapped_gen;
    unsigned short *act;   size_t act_cap;    /* [artifact_m, K] BF16 staging */
    float          *out;   size_t out_cap;    /* [artifact_m, N] F32 staging  */
} g_lane;
/* Release the helper-owned userptr wrapper and forget its identity.
 *
 * Called from three places, and each one matters: before wrapping a different
 * image, before the engine frees or replaces the memory the wrapper borrows,
 * and at shutdown. The wrapper borrows CALLER memory, so it must never outlive
 * the allocation it points into. */
static void coli_xdna_lane_drop_wrapper(void){
    if(!g_lane.wrapped) return;
    coli_xdna_helper_release_weight_call();
    g_lane.wrapped = NULL;
    g_lane.wrapped_gen = 0;
    g_xdna_wrapper_releases++;
}

/* The engine is about to free or replace prepared memory. If the helper holds a
 * wrapper over exactly that allocation, drop it FIRST. Without this the wrapper
 * would reference a freed region until the next wrap replaced it -- and a
 * subsequent allocation landing on the same address would make the stale
 * wrapper look current. */
static void coli_xdna_lane_forget_pointer(const void *p){
    if(p && g_lane.wrapped == p) coli_xdna_lane_drop_wrapper();
}

/* ---- helper entry points (ABI generation 2) ------------------------------
 *
 * Everything crossing this boundary is a fixed-width integer, a size_t, a raw
 * pointer with an explicit byte count, or a C string. No C++ type, no XRT type,
 * no engine type, and no exception: the helper catches everything and converts
 * it to one of these status codes. */
enum {
    COLI_XDNA_H_OK              =  0,
    COLI_XDNA_H_E_DEVICE        = -1,
    COLI_XDNA_H_E_ARTIFACT      = -2,
    COLI_XDNA_H_E_NOT_OPEN      = -3,
    COLI_XDNA_H_E_WRAP          = -4,
    COLI_XDNA_H_E_SIZE          = -5,
    COLI_XDNA_H_E_DISPATCH      = -6,
    COLI_XDNA_H_E_COMPLETION    = -7,
    COLI_XDNA_H_E_EXCEPTION     = -8
};

int coli_xdna_test_dispatches(void){ return g_xdna_dispatches; }
int coli_xdna_test_completions(void){ return g_xdna_completions; }
int coli_xdna_test_artifact_opens(void){ return g_xdna_artifact_opens; }
int coli_xdna_test_activation_preparations(void){ return g_xdna_act_preps; }
int coli_xdna_test_padded_operations(void){ return g_xdna_padded_ops; }
int coli_xdna_test_fallbacks(void){ return g_xdna_fallbacks; }

/* Per-bucket accessors. An unrecognised artifact M returns 0 -- it is not a
 * bucket this build compiled, so there is nothing to report and nothing may be
 * silently folded into a neighbouring bin. */
int coli_xdna_test_bucket_dispatches(unsigned artifact_m){
    int sl = coli_xdna_bucket_slot(artifact_m); return sl < 0 ? 0 : g_xdna_b_dispatches[sl];
}
int coli_xdna_test_bucket_completions(unsigned artifact_m){
    int sl = coli_xdna_bucket_slot(artifact_m); return sl < 0 ? 0 : g_xdna_b_completions[sl];
}
int coli_xdna_test_bucket_padded_operations(unsigned artifact_m){
    int sl = coli_xdna_bucket_slot(artifact_m); return sl < 0 ? 0 : g_xdna_b_padded_ops[sl];
}
long long coli_xdna_test_bucket_padded_rows(unsigned artifact_m){
    int sl = coli_xdna_bucket_slot(artifact_m); return sl < 0 ? 0 : g_xdna_b_padded_rows[sl];
}
int coli_xdna_test_bucket_artifact_opens(unsigned artifact_m){
    int sl = coli_xdna_bucket_slot(artifact_m); return sl < 0 ? 0 : g_xdna_b_artifact_opens[sl];
}
int coli_xdna_test_bucket_hard_eligible(unsigned artifact_m){
    int sl = coli_xdna_bucket_slot(artifact_m); return sl < 0 ? 0 : g_xdna_b_hard_eligible[sl];
}
int coli_xdna_test_m_above_range_declines(void){ return g_xdna_m_above_range_declines; }
int coli_xdna_test_bucket_switches(unsigned from_m, unsigned to_m){
    int a = coli_xdna_bucket_slot(from_m), b = coli_xdna_bucket_slot(to_m);
    return (a < 0 || b < 0) ? 0 : g_xdna_b_switches[a][b];
}
unsigned coli_xdna_test_bucket_for(int logical_m){
    return coli_xdna_bucket_for_logical_m(logical_m);
}
void coli_xdna_test_reset_bucket_counters(void){
    memset(g_xdna_b_hard_eligible,  0, sizeof g_xdna_b_hard_eligible);
    memset(g_xdna_b_dispatches,     0, sizeof g_xdna_b_dispatches);
    memset(g_xdna_b_completions,    0, sizeof g_xdna_b_completions);
    memset(g_xdna_b_padded_ops,     0, sizeof g_xdna_b_padded_ops);
    memset(g_xdna_b_padded_rows,    0, sizeof g_xdna_b_padded_rows);
    memset(g_xdna_b_artifact_opens, 0, sizeof g_xdna_b_artifact_opens);
    memset(g_xdna_b_switches,       0, sizeof g_xdna_b_switches);
    g_xdna_m_above_range_declines = 0;
}
int coli_xdna_test_wrapper_reuses(void){ return g_xdna_wrapper_reuses; }
int coli_xdna_test_wrapper_releases(void){ return g_xdna_wrapper_releases; }
int coli_xdna_test_stale_wrapper_rejects(void){ return g_xdna_stale_rejects; }
ColiXdnaHard coli_xdna_test_last_hard(void){ return g_xdna_last_hard; }
ColiXdnaExec coli_xdna_test_last_exec(void){ return g_xdna_last_exec; }

void coli_xdna_test_set_force_execution(int on){ g_xdna_force = on ? 1 : 0; }
/* The product artifact root: <exe-dir>\xdna. Same anchor as the helper, same
 * absolute form, same refusal to search. */
int coli_xdna_product_artifact_root(char *out, size_t cap){
#ifdef _WIN32
    char exe[512];
    DWORD n = GetModuleFileNameA(NULL, exe, (DWORD)sizeof exe);
    if(n == 0 || n >= sizeof exe) return 0;
    char *slash = strrchr(exe, '\\');
    char *fwd   = strrchr(exe, '/');
    if(fwd && (!slash || fwd > slash)) slash = fwd;
    if(!slash) return 0;
    *slash = '\0';
    int written = snprintf(out, cap, "%s\\%s", exe, COLI_XDNA_ARTIFACT_DIR);
    return written > 0 && (size_t)written < cap;
#else
    (void)out; (void)cap;
    return 0;
#endif
}

/* The root actually used by execution: a test override when one was set,
 * otherwise the product location. Returns NULL when neither is available,
 * which the artifact status treats as ARTIFACT_UNAVAILABLE -- never as a
 * licence to look somewhere else. */
static const char *coli_xdna_effective_root(char *scratch, size_t cap){
    if(g_xdna_root_overridden) return g_xdna_root[0] ? g_xdna_root : NULL;
    if(coli_xdna_product_artifact_root(scratch, cap)) return scratch;
    return NULL;
}

const char *coli_xdna_provision_text(ColiXdnaProvision p){
    switch(p){
    case COLI_XDNA_PROV_READY:              return "READY";
    case COLI_XDNA_PROV_NOT_REQUESTED:      return "NOT_REQUESTED";
    case COLI_XDNA_PROV_PACKAGE_MISSING:    return "PACKAGE_MISSING";
    case COLI_XDNA_PROV_PACKAGE_INCOMPLETE: return "PACKAGE_INCOMPLETE";
    case COLI_XDNA_PROV_INTEGRITY_FAILED:   return "INTEGRITY_FAILED";
    case COLI_XDNA_PROV_HELPER_UNAVAILABLE: return "HELPER_UNAVAILABLE";
    case COLI_XDNA_PROV_REGISTRY_INVALID:   return "REGISTRY_INVALID";
    }
    return "UNKNOWN";
}

ColiXdnaProvision coli_xdna_provision_status(const char **missing_name){
    if(missing_name) *missing_name = NULL;
    if(!coli_xdna_registry_validate()) return COLI_XDNA_PROV_REGISTRY_INVALID;

    char scratch[1024];
    const char *root = coli_xdna_effective_root(scratch, sizeof scratch);
    if(!root) return COLI_XDNA_PROV_PACKAGE_MISSING;

    /* Every row this build ships, not just the first. A package holding only
     * the small bucket would otherwise report READY and then fail at the first
     * long request -- surfacing at activation is the whole point. */
    int checked = 0;
    for(int i = 0; i < g_nrows; i++){
        const ColiXdnaArtifact *a = &g_rows[i];
        const char *names[2]  = { a->xclbin_name, a->insts_name };
        const char *hashes[2] = { a->xclbin_sha256, a->insts_sha256 };
        for(int k = 0; k < 2; k++){
            ColiXdnaStatic st = coli_xdna_check_file(root, names[k], hashes[k]);
            if(st == COLI_XDNA_STATIC_QUALIFIED){ checked++; continue; }
            if(missing_name) *missing_name = names[k];
            if(st == COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED)
                return COLI_XDNA_PROV_INTEGRITY_FAILED;
            /* Absent file. If NOTHING was verified yet the package as a whole is
             * missing; if some files verified, the package is present but
             * incomplete. Different problems, different fixes. */
            return checked ? COLI_XDNA_PROV_PACKAGE_INCOMPLETE
                           : COLI_XDNA_PROV_PACKAGE_MISSING;
        }
    }

    /* Bytes are good. The helper is the remaining prerequisite, and it is
     * reported separately because "install the package" and "the helper will
     * not load" are not the same instruction. */
    if(coli_xdna_binding() != COLI_XDNA_AVAILABLE)
        return COLI_XDNA_PROV_HELPER_UNAVAILABLE;

    return COLI_XDNA_PROV_READY;
}

void coli_xdna_set_explicit_enabled(int on){ g_xdna_explicit = on ? 1 : 0; }
int  coli_xdna_explicit_enabled(void){ return g_xdna_explicit; }

void coli_xdna_test_set_artifact_root(const char *root){
    g_xdna_root_overridden = 1;
    if(root && *root) snprintf(g_xdna_root, sizeof g_xdna_root, "%s", root);
    else              g_xdna_root[0] = '\0';
}

int coli_xdna_test_device_opens(void){ return g_xdna_device_opens; }
int coli_xdna_test_helper_calls(void){ return g_xdna_helper_calls; }
int coli_xdna_test_userptr_wraps(void){ return g_xdna_userptr_wraps; }
int coli_xdna_test_conversions(void){ return g_xdna_conversions; }

/* ---- the candidate --------------------------------------------------------
 *
 * Gate order is the point of this function, not an accident of it. Everything
 * the engine can answer from its own state is answered first, cheapest and most
 * semantic first, so an operation that could never run costs a handful of
 * integer comparisons and never touches the filesystem, the helper or the
 * device. Nothing later can retroactively excuse an earlier refusal: every gate
 * returns, none of them merely records. */

static ColiXdnaHard coli_xdna_engine_gates(ColiXdnaFamily family,
                                           int fmt, int I, int O, int gs, int planar, int S,
                                           const ColiXdnaArtifact **row_out){
    /*  1  semantic family -- passed in by the call site, never inferred */
    if(family != COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP)
        return COLI_XDNA_HARD_FAMILY_UNSUPPORTED;

    /*  2  logical M within the range some compiled bucket serves.
     *
     *     The upper bound is now the LARGEST bucket, not the only one. Below
     *     the minimum and above the largest are the same verdict to the caller
     *     -- M_OUT_OF_RANGE, decline to the current path -- but the
     *     above-the-range case is counted separately, because it is the new
     *     first-decline boundary and a later slice will want to know how often
     *     real work lands there. */
    if(S < COLI_XDNA_LOGICAL_M_MIN || S > COLI_XDNA_LOGICAL_M_MAX){
        if(S > COLI_XDNA_LOGICAL_M_MAX) g_xdna_m_above_range_declines++;
        return COLI_XDNA_HARD_M_OUT_OF_RANGE;
    }

    /*  3  K / N representable and positive */
    if(I <= 0 || O <= 0) return COLI_XDNA_HARD_SHAPE_UNSUPPORTED;

    /*  4  stored format */
    if(fmt != 4) return COLI_XDNA_HARD_FORMAT_UNSUPPORTED;

    /*  5  group size -- the converter is more permissive than the device */
    if(gs != COLI_XDNA_QUALIFIED_GROUP_SIZE)
        return COLI_XDNA_HARD_GROUP_SIZE_UNSUPPORTED;

    /*  5b in-memory byte layout.
     *
     *     fmt=4 has TWO layouts in current Colibri. The classic PAIR layout
     *     puts elements 2j and 2j+1 in byte j; the K1 PLANAR layout puts
     *     elements k and k+32 in byte k of each 64-element block, and
     *     qt_planarize() rewrites the tensor IN PLACE when the grouped planar
     *     IDOT path is opted into (IDOT_GS=1), for exactly the gs>=64 tensors
     *     this lane qualifies. The converter was qualified against the pair
     *     layout only, so planar bytes would decode to nonsense rather than
     *     fail. Decline instead. */
    if(planar)
        return COLI_XDNA_HARD_LAYOUT_UNSUPPORTED;

    /*  6  a qualified artifact for exactly this family/bucket/shape/dtypes.
     *     The bucket is the artifact's M, which is where logical M is mapped. */
    ColiXdnaRequest q;
    q.family = family;
    q.m = coli_xdna_bucket_for_logical_m(S);      /* smallest bucket holding S */
    if(q.m == 0u) return COLI_XDNA_HARD_M_OUT_OF_RANGE;   /* unreachable: gate 2 */
    q.k = (unsigned)I; q.n = (unsigned)O;
    q.in_dtype = COLI_XDNA_DT_BF16;
    q.weight_dtype = COLI_XDNA_DT_BF16;
    q.out_dtype = COLI_XDNA_DT_F32;

    const ColiXdnaArtifact *row = coli_xdna_registry_lookup(&q);
    if(!row) return COLI_XDNA_HARD_SHAPE_UNSUPPORTED;

    /*  7-9  qualification flags, artifact presence, artifact integrity. All
     *       three live behind coli_xdna_artifact_status, which fails closed.
     *       The sub-reason is carried through rather than collapsed: "this
     *       build does not ship that artifact", "the bytes are not the bytes
     *       that were qualified", and "we never qualified this one" call for
     *       different actions, and an operator who is told only
     *       ARTIFACT_NOT_QUALIFIED cannot tell a missing file from a tampered
     *       one. */
    char root_scratch[1024];
    const char *eff_root = coli_xdna_effective_root(root_scratch, sizeof root_scratch);
    switch(coli_xdna_artifact_status(&q, eff_root)){
        case COLI_XDNA_STATIC_QUALIFIED:
            break;
        case COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE:
            return COLI_XDNA_HARD_ARTIFACT_UNAVAILABLE;
        case COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED:
            return COLI_XDNA_HARD_ARTIFACT_INTEGRITY_FAILED;
        case COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED:
            return COLI_XDNA_HARD_ARTIFACT_UNQUALIFIED;
        case COLI_XDNA_STATIC_REGISTRY_INVALID:
            return COLI_XDNA_HARD_REGISTRY_INVALID;
        default:
            /* Family/shape/format verdicts cannot occur here: the lookup above
             * already matched a row on the full key. */
            return COLI_XDNA_HARD_SHAPE_UNSUPPORTED;
    }

    *row_out = row;
    return COLI_XDNA_HARD_ELIGIBLE;
}

/* Grow a staging buffer to at least `need` bytes. Transient operation state:
 * reused across operations so a steady stream of the same shape allocates once,
 * but carrying no validity and no authority of any kind. */
static int coli_xdna_stage(void **base, size_t *cap, size_t need){
    if(*cap >= need) return 1;
    void *p = malloc(need);
    if(!p) return 0;
    free(*base);
    *base = p; *cap = need;
    return 1;
}

/* The shared execution core. Both modes call this, so their gate order and
 * their failure classification cannot drift apart.
 *
 * It returns a CLASS, never a handled flag: deciding what a failure means for
 * the caller belongs to the mode, not to this function. It writes y only after
 * the helper reports successful completion, and never calls matmul_qt. */
static ColiXdnaExec coli_xdna_attempt(ColiXdnaFamily family,
                                      ColiXdnaPrepared **slot,
                                      int fmt, const unsigned char *q4, const float *scale,
                                      int I, int O, int gs, int planar,
                                      float *y, const float *x, int S)
{
    g_xdna_output_valid = 0;          /* nothing is valid until completion says so */

    if(!slot || !q4 || !scale || !y || !x){
        g_xdna_last_hard = COLI_XDNA_HARD_SHAPE_UNSUPPORTED;
        return (g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED);
    }

    /* A device that would not initialise stays refused for the process. This is
     * the whole of the retry policy, and deliberately not more: the sealed A5
     * taxonomy classifies device-init failure PROCESS-scoped, and re-attempting
     * it on every operation would cost without prospect. */
    if(g_xdna_lane_health == COLI_XDNA_LANE_UNAVAILABLE){
        g_xdna_last_hard = COLI_XDNA_HARD_DEVICE_UNAVAILABLE;
        return (g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED);
    }

    const ColiXdnaArtifact *row = NULL;
    ColiXdnaHard h = coli_xdna_engine_gates(family, fmt, I, O, gs, planar, S, &row);
    if(h != COLI_XDNA_HARD_ELIGIBLE){
        g_xdna_last_hard = h;
        return (g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED);
    }

    /* 10  helper binding. Checked before the weight is prepared: preparing
     *     24 MiB for a lane that cannot execute would be pure waste. The
     *     loader verdict is sticky, so this is a branch, not a module load. */
    {
        ColiXdnaBinding b = coli_xdna_binding();
        if(b != COLI_XDNA_AVAILABLE){
            g_xdna_last_hard = (b == COLI_XDNA_ABI_INCOMPATIBLE
                                || b == COLI_XDNA_SYMBOL_INCOMPLETE)
                             ? COLI_XDNA_HARD_HELPER_ABI_INCOMPATIBLE
                             : COLI_XDNA_HARD_HELPER_UNAVAILABLE;
            return (g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED);
        }
    }

    /* 11  prepared weight. Lazy, and reused while it stays VALID for these
     *     dimensions. No eviction or cache policy here. */
    if(!*slot){
        *slot = coli_xdna_prepared_create();
        if(!*slot){
            g_xdna_last_hard = COLI_XDNA_HARD_PREPARED_INVALID;
            return (g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED);
        }
    }
    ColiXdnaPrepared *prep = *slot;
    if(coli_xdna_prepared_state(prep) != COLI_XDNA_PREP_VALID
       || coli_xdna_prepared_k(prep) != (unsigned)I
       || coli_xdna_prepared_n(prep) != (unsigned)O){
        if(coli_xdna_prepare_from_fmt4(prep, fmt, q4, scale, I, O, gs) != COLI_XDNA_PREP_OK){
            g_xdna_last_hard = COLI_XDNA_HARD_PREPARED_INVALID;
            return (g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED);
        }
    }
    const void *wimg = coli_xdna_prepared_image(prep);   /* NULL unless VALID */
    if(!wimg){
        g_xdna_last_hard = COLI_XDNA_HARD_PREPARED_INVALID;
        return (g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED);
    }

    /* 12  alignment */
    if(!coli_xdna_pointer_alignment_ok(wimg)){
        g_xdna_last_hard = COLI_XDNA_HARD_ALIGNMENT_INVALID;
        return (g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED);
    }

    const unsigned am = row->artifact_m;

    /* 13  device / runtime / artifact runtime object, opened lazily. */
    if(!g_lane.open || g_lane.m != am || g_lane.k != (unsigned)I || g_lane.n != (unsigned)O){
        char xb[2048], ib[2048];
        char rs2[1024];
        const char *r2 = coli_xdna_effective_root(rs2, sizeof rs2);
        if(!r2
           || !coli_xdna_join(xb, sizeof xb, r2, row->xclbin_name)
           || !coli_xdna_join(ib, sizeof ib, r2, row->insts_name)){
            g_xdna_last_hard = COLI_XDNA_HARD_ARTIFACT_RUNTIME_UNAVAILABLE;
            return (g_xdna_last_exec = COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED);
        }
        /* Record the transition BEFORE tearing the old artifact down, while
         * g_lane.m still names the bucket being left. A bucket switch is the
         * expensive case (shutdown + open + forced re-wrap), so it is counted
         * rather than inferred from artifact_opens, which also fires on the
         * very first open and on a K/N change. */
        if(g_lane.open){
            int from = coli_xdna_bucket_slot(g_lane.m), to = coli_xdna_bucket_slot(am);
            if(from >= 0 && to >= 0 && from != to) g_xdna_b_switches[from][to]++;
        }
        if(g_lane.open){ coli_xdna_lane_drop_wrapper(); coli_xdna_helper_shutdown_call(); }
        g_lane.open = 0;
        g_xdna_helper_calls++; g_xdna_device_opens++;
        int rc = coli_xdna_helper_open_call(xb, ib, am, (unsigned)I, (unsigned)O);
        if(rc != COLI_XDNA_H_OK){
            if(rc == COLI_XDNA_H_E_DEVICE){
                g_xdna_lane_health = COLI_XDNA_LANE_UNAVAILABLE;   /* process-scoped */
                g_xdna_last_hard = COLI_XDNA_HARD_DEVICE_UNAVAILABLE;
                return (g_xdna_last_exec = COLI_XDNA_EXEC_DEVICE_INIT_FAILED);
            }
            /* Artifact-runtime scope. The engine already verified these bytes,
             * so this is NOT the same thing as a missing or tampered artifact
             * and must stay distinguishable from both. The lane stays healthy:
             * another shape may well open. */
            g_xdna_last_hard = COLI_XDNA_HARD_ARTIFACT_RUNTIME_UNAVAILABLE;
            return (g_xdna_last_exec = COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED);
        }
        g_lane.open = 1; g_lane.m = am; g_lane.k = (unsigned)I; g_lane.n = (unsigned)O;
        g_lane.wrapped = NULL; g_lane.wrapped_gen = 0;
        g_xdna_lane_health = COLI_XDNA_LANE_HEALTHY;
        g_xdna_artifact_opens++;
        { int sl = coli_xdna_bucket_slot(am); if(sl >= 0) g_xdna_b_artifact_opens[sl]++; }
    }

    /* 14  userptr wrap, keyed on (pointer, generation).
     *
     *     The pointer alone is not an identity. prepare_begin reuses retained
     *     capacity IN PLACE, so re-preparing a different weight yields the same
     *     address with different contents -- and the helper snapshots at wrap
     *     time via sync(BO_TO_DEVICE). Keying on the pointer alone therefore
     *     skipped the re-wrap and left the device computing against a view that
     *     no longer matched the engine image. Measured on real XDNA2 hardware,
     *     not merely reasoned about. */
    {
        unsigned long long gen = coli_xdna_prepared_generation(prep);
        if(g_lane.wrapped == wimg && g_lane.wrapped_gen == gen){
            g_xdna_wrapper_reuses++;
        } else {
            if(g_lane.wrapped == wimg) g_xdna_stale_rejects++;   /* same address, new image */
            if(g_lane.wrapped) coli_xdna_lane_drop_wrapper();
            g_xdna_helper_calls++; g_xdna_userptr_wraps++;
            if(coli_xdna_helper_wrap_call((void *)wimg,
                                          (uint64_t)coli_xdna_prepared_bytes(prep)) != COLI_XDNA_H_OK){
                g_xdna_last_hard = COLI_XDNA_HARD_WEIGHT_WRAP_UNAVAILABLE;
                return (g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_WRAP_FAILED);
            }
            g_lane.wrapped = wimg; g_lane.wrapped_gen = gen;
        }
    }

    g_xdna_last_hard = COLI_XDNA_HARD_ELIGIBLE;   /* FULL_XDNA_HARD_ELIGIBLE */
    { int sl = coli_xdna_bucket_slot(am); if(sl >= 0) g_xdna_b_hard_eligible[sl]++; }

    /* ---- activation staging ------------------------------------------------- */
    size_t abytes = (size_t)am * (size_t)I * 2u;
    size_t obytes = (size_t)am * (size_t)O * 4u;
    if(!coli_xdna_stage((void **)&g_lane.act, &g_lane.act_cap, abytes)
       || !coli_xdna_stage((void **)&g_lane.out, &g_lane.out_cap, obytes)){
        return (g_xdna_last_exec = COLI_XDNA_EXEC_ACTIVATION_FAILED);
    }
    for(size_t i = 0; i < (size_t)S * (size_t)I; i++)
        g_lane.act[i] = coli_xdna_f2b(x[i]);
    if((unsigned)S < am){
        memset(g_lane.act + (size_t)S * (size_t)I, 0,
               ((size_t)am - (size_t)S) * (size_t)I * 2u);
        g_xdna_padded_ops++;
        /* ROWS as well as OPS: with two buckets the op count no longer says how
         * much padding was computed. An S=65 operation on M256 pads 191 rows;
         * an S=63 operation on M64 pads 1. Both are one "padded op". */
        { int sl = coli_xdna_bucket_slot(am);
          if(sl >= 0){ g_xdna_b_padded_ops[sl]++;
                       g_xdna_b_padded_rows[sl] += (long long)(am - (unsigned)S); } }
    }
    g_xdna_act_preps++;

    /* ---- blocking execution ------------------------------------------------- */
    g_xdna_helper_calls++;
    int rc = coli_xdna_helper_execute_call(g_lane.act, (uint64_t)abytes,
                                           g_lane.out, (uint64_t)obytes);
    if(rc != COLI_XDNA_H_OK){
        /* The helper may already have written real bytes into the staging
         * buffer before deciding it had failed. They stay there and are simply
         * never copied: y is untouched, and no property of those bytes can make
         * them valid, because validity is a state and this state is failure. */
        return (g_xdna_last_exec = (rc == COLI_XDNA_H_E_COMPLETION)
                                 ? COLI_XDNA_EXEC_COMPLETION_FAILED
                                 : COLI_XDNA_EXEC_EXECUTE_FAILED);
    }
    g_xdna_dispatches++; g_xdna_completions++;
    { int sl = coli_xdna_bucket_slot(am);
      if(sl >= 0){ g_xdna_b_dispatches[sl]++; g_xdna_b_completions[sl]++; } }

    memcpy(y, g_lane.out, (size_t)S * (size_t)O * 4u);   /* logical rows only */
    g_xdna_output_valid = 1;
    return (g_xdna_last_exec = COLI_XDNA_EXEC_OK);
}

/* AUTO-LIKE mode: the production seam. Any decline or failure returns 0 and the
 * caller runs its current path, which is the exact matmul_qt call that stood
 * there before this lane existed. */
int coli_xdna_try_matmul(ColiXdnaFamily family,
                         ColiXdnaPrepared **slot,
                         int fmt, const unsigned char *q4, const float *scale,
                         int I, int O, int gs, int planar,
                         float *y, const float *x, int S)
{
    /* Gate 0: PERMISSION. Without it this function is inert, which is what keeps
     * the lane from being an automatic scheduler.
     *
     * Two independent sources, and they are not interchangeable:
     *   g_xdna_explicit  the user asked for it (`coli --xdna` -> COLI_XDNA=1)
     *   g_xdna_force     an internal qualification run forced it
     *
     * Neither is discovery. Present hardware, a loadable helper and valid
     * artifacts do not set either one, so a machine that COULD run this lane
     * still does not until someone says so. Permission is also not selection:
     * every hard gate below still decides, and an unqualified operation runs
     * the current path no matter who asked. */
    if(!(g_xdna_explicit || g_xdna_force)){
        /* Clear validity here too. Declining before the core runs still ends an
         * attempt, and a stale 1 left over from an earlier success would claim
         * an XDNA result for an operation the lane never touched. */
        g_xdna_output_valid = 0;
        g_xdna_fallbacks++;
        g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED;
        return 0;
    }
    if(coli_xdna_attempt(family, slot, fmt, q4, scale, I, O, gs, planar, y, x, S)
       == COLI_XDNA_EXEC_OK)
        return 1;
    g_xdna_fallbacks++;
    return 0;
}

/* EXPLICIT mode, internal and test-facing. Returns the class and implies no
 * fallback: a caller here asked for XDNA specifically and gets a classified
 * failure rather than a silent substitution. It does not consult the force
 * control, because asking explicitly IS the request. Kept as a separate entry
 * point rather than a mode flag so that no global state can ever leave the
 * production seam in a no-fallback configuration. */
ColiXdnaExec coli_xdna_test_attempt(ColiXdnaFamily family,
                                    ColiXdnaPrepared **slot,
                                    int fmt, const unsigned char *q4, const float *scale,
                                    int I, int O, int gs, int planar,
                                    float *y, const float *x, int S)
{
    return coli_xdna_attempt(family, slot, fmt, q4, scale, I, O, gs, planar, y, x, S);
}

int coli_xdna_test_output_valid(void){ return g_xdna_output_valid; }

const float *coli_xdna_test_output_staging(size_t *floats){
    if(floats) *floats = g_lane.out_cap / sizeof(float);
    return g_lane.out;
}

ColiXdnaLaneHealth coli_xdna_lane_health(void){ return g_xdna_lane_health; }

const char *coli_xdna_lane_health_text(ColiXdnaLaneHealth h){
    switch(h){
        case COLI_XDNA_LANE_UNINITIALIZED: return "UNINITIALIZED";
        case COLI_XDNA_LANE_HEALTHY:       return "HEALTHY";
        case COLI_XDNA_LANE_UNAVAILABLE:   return "UNAVAILABLE";
    }
    return "UNKNOWN";
}
void coli_xdna_execution_shutdown(void){
    coli_xdna_lane_drop_wrapper();
    if(g_lane.open) coli_xdna_helper_shutdown_call();
    g_lane.open = 0; g_lane.wrapped = NULL; g_lane.wrapped_gen = 0;
    /* Lane health is lane runtime state, so it resets with the rest of it. A
     * fresh attempt after a full teardown is a meaningful thing to allow; a
     * fresh attempt on every operation is not, which is why nothing else
     * clears it. */
    g_xdna_lane_health = COLI_XDNA_LANE_UNINITIALIZED;
    free(g_lane.act); g_lane.act = NULL; g_lane.act_cap = 0;
    free(g_lane.out); g_lane.out = NULL; g_lane.out_cap = 0;
}
