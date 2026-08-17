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
};

/* Engine-side host accounting. Deliberately counts LOGICAL payload bytes: the
 * allocator's own bookkeeping overhead is real but not knowable from here, and
 * reporting a number we cannot substantiate would be worse than reporting the
 * one we can. */
static size_t g_xdna_prepared_bytes;
static int    g_xdna_prepared_objects;

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
    return 1;
}

ColiXdnaPrepState coli_xdna_prepared_state(const ColiXdnaPrepared *p){
    return p ? p->state : COLI_XDNA_PREP_UNPREPARED;
}
size_t   coli_xdna_prepared_bytes(const ColiXdnaPrepared *p){ return p ? p->bytes : 0; }
unsigned coli_xdna_prepared_k(const ColiXdnaPrepared *p){ return p ? p->k : 0u; }
unsigned coli_xdna_prepared_n(const ColiXdnaPrepared *p){ return p ? p->n : 0u; }

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
        case COLI_XDNA_HARD_ARTIFACT_NOT_QUALIFIED:      return "ARTIFACT_NOT_QUALIFIED";
        case COLI_XDNA_HARD_PREPARED_INVALID:            return "PREPARED_INVALID";
        case COLI_XDNA_HARD_ALIGNMENT_INVALID:           return "ALIGNMENT_INVALID";
        case COLI_XDNA_HARD_HELPER_UNAVAILABLE:          return "HELPER_UNAVAILABLE";
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
static char         g_xdna_root[1024];            /* artifact root; empty = none */
static ColiXdnaHard g_xdna_last_hard = COLI_XDNA_HARD_ELIGIBLE;
static ColiXdnaExec g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED;
static int g_xdna_dispatches, g_xdna_completions, g_xdna_artifact_opens;
static int g_xdna_act_preps, g_xdna_padded_ops, g_xdna_fallbacks;
static int g_xdna_device_opens, g_xdna_helper_calls, g_xdna_userptr_wraps;

/* Which artifact the helper currently holds open, so a run of operations on the
 * same shape does not reopen the device, the context or the program. This is a
 * RUNTIME object cache, not an identity cache: it can only ever hold what the
 * engine already selected, and a different selection closes it and reopens. */
static struct {
    int      open;
    unsigned m, k, n;
    /* the wrapped weight, identified by the exact pointer the engine handed us */
    const void *wrapped;
    unsigned short *act;   size_t act_cap;    /* [artifact_m, K] BF16 staging */
    float          *out;   size_t out_cap;    /* [artifact_m, N] F32 staging  */
} g_lane;

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
ColiXdnaHard coli_xdna_test_last_hard(void){ return g_xdna_last_hard; }
ColiXdnaExec coli_xdna_test_last_exec(void){ return g_xdna_last_exec; }

void coli_xdna_test_set_force_execution(int on){ g_xdna_force = on ? 1 : 0; }

void coli_xdna_test_set_artifact_root(const char *root){
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
                                           int fmt, int I, int O, int gs, int S,
                                           const ColiXdnaArtifact **row_out){
    /*  1  semantic family -- passed in by the call site, never inferred */
    if(family != COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP)
        return COLI_XDNA_HARD_FAMILY_UNSUPPORTED;

    /*  2  logical M in the range this slice implements */
    if(S < COLI_XDNA_I5_LOGICAL_M_MIN || S > COLI_XDNA_I5_LOGICAL_M_MAX)
        return COLI_XDNA_HARD_M_OUT_OF_RANGE;

    /*  3  K / N representable and positive */
    if(I <= 0 || O <= 0) return COLI_XDNA_HARD_SHAPE_UNSUPPORTED;

    /*  4  stored format */
    if(fmt != 4) return COLI_XDNA_HARD_FORMAT_UNSUPPORTED;

    /*  5  group size -- the converter is more permissive than the device */
    if(gs != COLI_XDNA_QUALIFIED_GROUP_SIZE)
        return COLI_XDNA_HARD_GROUP_SIZE_UNSUPPORTED;

    /*  6  a qualified artifact for exactly this family/bucket/shape/dtypes.
     *     The bucket is the artifact's M, which is where logical M is mapped. */
    ColiXdnaRequest q;
    q.family = family;
    q.m = (unsigned)COLI_XDNA_I5_LOGICAL_M_MAX;   /* the F3 M64 bucket */
    q.k = (unsigned)I; q.n = (unsigned)O;
    q.in_dtype = COLI_XDNA_DT_BF16;
    q.weight_dtype = COLI_XDNA_DT_BF16;
    q.out_dtype = COLI_XDNA_DT_F32;

    const ColiXdnaArtifact *row = coli_xdna_registry_lookup(&q);
    if(!row) return COLI_XDNA_HARD_SHAPE_UNSUPPORTED;

    /*  7-9  qualification flags, artifact presence, artifact integrity. All
     *       three live behind coli_xdna_artifact_status, which fails closed. */
    if(coli_xdna_artifact_status(&q, g_xdna_root[0] ? g_xdna_root : NULL)
       != COLI_XDNA_STATIC_QUALIFIED)
        return COLI_XDNA_HARD_ARTIFACT_NOT_QUALIFIED;

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

int coli_xdna_try_matmul(ColiXdnaFamily family,
                         ColiXdnaPrepared **slot,
                         int fmt, const unsigned char *q4, const float *scale,
                         int I, int O, int gs,
                         float *y, const float *x, int S)
{
    /* Gate 0: the internal control. Without it this function is inert, which is
     * what keeps I5 from being an automatic scheduler. */
    if(!g_xdna_force){ g_xdna_fallbacks++; g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED; return 0; }

    if(!slot || !q4 || !scale || !y || !x){
        g_xdna_last_hard = COLI_XDNA_HARD_SHAPE_UNSUPPORTED;
        g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED;
        g_xdna_fallbacks++; return 0;
    }

    const ColiXdnaArtifact *row = NULL;
    ColiXdnaHard h = coli_xdna_engine_gates(family, fmt, I, O, gs, S, &row);
    if(h != COLI_XDNA_HARD_ELIGIBLE){
        g_xdna_last_hard = h; g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED;
        g_xdna_fallbacks++; return 0;
    }

    /* 10  helper ABI. Checked before the weight is prepared: preparing 24 MiB
     *     for a lane that cannot execute would be pure waste. */
    if(coli_xdna_binding() != COLI_XDNA_AVAILABLE){
        g_xdna_last_hard = COLI_XDNA_HARD_HELPER_UNAVAILABLE;
        g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED;
        g_xdna_fallbacks++; return 0;
    }

    /* 11  prepared weight. Lazy: prepared on first use of this tensor and
     *     reused while it stays VALID for the same dimensions. There is no
     *     eviction or cache policy here -- that is a later slice's decision. */
    if(!*slot){
        *slot = coli_xdna_prepared_create();
        if(!*slot){
            g_xdna_last_hard = COLI_XDNA_HARD_PREPARED_INVALID;
            g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED;
            g_xdna_fallbacks++; return 0;
        }
    }
    ColiXdnaPrepared *prep = *slot;
    if(coli_xdna_prepared_state(prep) != COLI_XDNA_PREP_VALID
       || coli_xdna_prepared_k(prep) != (unsigned)I
       || coli_xdna_prepared_n(prep) != (unsigned)O){
        if(coli_xdna_prepare_from_fmt4(prep, fmt, q4, scale, I, O, gs) != COLI_XDNA_PREP_OK){
            g_xdna_last_hard = COLI_XDNA_HARD_PREPARED_INVALID;
            g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED;
            g_xdna_fallbacks++; return 0;
        }
    }
    const void *wimg = coli_xdna_prepared_image(prep);   /* NULL unless VALID */
    if(!wimg){
        g_xdna_last_hard = COLI_XDNA_HARD_PREPARED_INVALID;
        g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED;
        g_xdna_fallbacks++; return 0;
    }

    /* 12  alignment. The allocator guarantees it; checking anyway is cheap and
     *     turns a confusing XRT "insufficient free video memory" into a local,
     *     accurate refusal. */
    if(!coli_xdna_pointer_alignment_ok(wimg)){
        g_xdna_last_hard = COLI_XDNA_HARD_ALIGNMENT_INVALID;
        g_xdna_last_exec = COLI_XDNA_EXEC_DECLINED;
        g_xdna_fallbacks++; return 0;
    }

    const unsigned am = row->artifact_m;

    /* 13  device / runtime / artifact runtime object. Opened lazily, only now
     *     that every engine-side gate has passed. */
    if(!g_lane.open || g_lane.m != am || g_lane.k != (unsigned)I || g_lane.n != (unsigned)O){
        char xb[2048], ib[2048];
        if(!coli_xdna_join(xb, sizeof xb, g_xdna_root, row->xclbin_name)
           || !coli_xdna_join(ib, sizeof ib, g_xdna_root, row->insts_name)){
            g_xdna_last_hard = COLI_XDNA_HARD_ARTIFACT_RUNTIME_UNAVAILABLE;
            g_xdna_last_exec = COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED;
            g_xdna_fallbacks++; return 0;
        }
        if(g_lane.open){ coli_xdna_helper_release_weight_call(); coli_xdna_helper_shutdown_call(); }
        g_xdna_helper_calls++; g_xdna_device_opens++;
        int rc = coli_xdna_helper_open_call(xb, ib, am, (unsigned)I, (unsigned)O);
        if(rc != COLI_XDNA_H_OK){
            g_xdna_last_hard = (rc == COLI_XDNA_H_E_DEVICE)
                             ? COLI_XDNA_HARD_DEVICE_UNAVAILABLE
                             : COLI_XDNA_HARD_ARTIFACT_RUNTIME_UNAVAILABLE;
            g_xdna_last_exec = (rc == COLI_XDNA_H_E_DEVICE)
                             ? COLI_XDNA_EXEC_DEVICE_INIT_FAILED
                             : COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED;
            g_xdna_fallbacks++; return 0;
        }
        g_lane.open = 1; g_lane.m = am; g_lane.k = (unsigned)I; g_lane.n = (unsigned)O;
        g_lane.wrapped = NULL;
        g_xdna_artifact_opens++;
    }

    /* 14  userptr wrap of the engine-owned prepared image. The helper wraps
     *     THIS memory; it never allocates a second copy of the weight. */
    if(g_lane.wrapped != wimg){
        g_xdna_helper_calls++; g_xdna_userptr_wraps++;
        if(coli_xdna_helper_wrap_call((void *)wimg,
                                      (uint64_t)coli_xdna_prepared_bytes(prep)) != COLI_XDNA_H_OK){
            g_xdna_last_hard = COLI_XDNA_HARD_WEIGHT_WRAP_UNAVAILABLE;
            g_xdna_last_exec = COLI_XDNA_EXEC_WEIGHT_WRAP_FAILED;
            g_xdna_fallbacks++; return 0;
        }
        g_lane.wrapped = wimg;
    }

    g_xdna_last_hard = COLI_XDNA_HARD_ELIGIBLE;   /* FULL_XDNA_HARD_ELIGIBLE */

    /* ---- activation staging -------------------------------------------------
     * Caller activation is f32 [S, K] row-major. The artifact consumes BF16
     * [artifact_m, K]. The conversion uses the SAME round-to-nearest-even rule
     * as the weight, because both were qualified on that rule together.
     *
     * Rows S..artifact_m-1 are zeroed. Zero is the mathematically neutral row
     * for a pure GEMM: it contributes a zero output row and cannot perturb any
     * logical row, because C = A x B is row-independent. */
    size_t abytes = (size_t)am * (size_t)I * 2u;
    size_t obytes = (size_t)am * (size_t)O * 4u;
    if(!coli_xdna_stage((void **)&g_lane.act, &g_lane.act_cap, abytes)
       || !coli_xdna_stage((void **)&g_lane.out, &g_lane.out_cap, obytes)){
        g_xdna_last_exec = COLI_XDNA_EXEC_ACTIVATION_FAILED;
        g_xdna_fallbacks++; return 0;
    }
    for(size_t i = 0; i < (size_t)S * (size_t)I; i++)
        g_lane.act[i] = coli_xdna_f2b(x[i]);
    if((unsigned)S < am){
        memset(g_lane.act + (size_t)S * (size_t)I, 0,
               ((size_t)am - (size_t)S) * (size_t)I * 2u);
        g_xdna_padded_ops++;
    }
    g_xdna_act_preps++;

    /* ---- blocking execution -------------------------------------------------
     * One operation, one dispatch, one wait. No worker thread, no queue, no
     * second outstanding operation. */
    g_xdna_helper_calls++;
    int rc = coli_xdna_helper_execute_call(g_lane.act, (uint64_t)abytes,
                                           g_lane.out, (uint64_t)obytes);
    if(rc != COLI_XDNA_H_OK){
        g_xdna_last_exec = (rc == COLI_XDNA_H_E_COMPLETION)
                         ? COLI_XDNA_EXEC_COMPLETION_FAILED
                         : COLI_XDNA_EXEC_EXECUTE_FAILED;
        g_xdna_fallbacks++;
        return 0;                       /* y untouched: the caller's path owns it */
    }
    g_xdna_dispatches++; g_xdna_completions++;

    /* Only the logical rows become output. The padded rows are never copied and
     * so can never reach anything downstream, whatever the device wrote there. */
    memcpy(y, g_lane.out, (size_t)S * (size_t)O * 4u);
    g_xdna_last_exec = COLI_XDNA_EXEC_OK;
    return 1;
}

void coli_xdna_execution_shutdown(void){
    if(g_lane.open){
        coli_xdna_helper_release_weight_call();
        coli_xdna_helper_shutdown_call();
    }
    g_lane.open = 0; g_lane.wrapped = NULL;
    free(g_lane.act); g_lane.act = NULL; g_lane.act_cap = 0;
    free(g_lane.out); g_lane.out = NULL; g_lane.out_cap = 0;
}
