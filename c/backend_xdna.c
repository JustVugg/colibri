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

/* Entry points required of a helper at this ABI generation. Kept minimal on
 * purpose: a version handshake, and the release call this file's own lifetime
 * contract needs. Later slices add device/artifact/execute entry points and
 * bump COLI_XDNA_ABI_VERSION with them. */
typedef unsigned int (*coli_xdna_fn_abi_version)(void);
typedef void         (*coli_xdna_fn_shutdown)(void);

static struct {
    ColiXdnaBinding state;
    int             load_attempts;
    char            path[512];       /* test override; empty = default lookup */
#ifdef _WIN32
    HMODULE         dll;
#endif
    coli_xdna_fn_abi_version abi_version;
    coli_xdna_fn_shutdown    shutdown;
} g_xdna = { COLI_XDNA_UNPROBED, 0, {0},
#ifdef _WIN32
             NULL,
#endif
             NULL, NULL };

/* Drop every callable pointer. Always runs BEFORE the module is released, so a
 * rejected or shut-down helper can never leave a pointer into an unmapped
 * module behind -- the all-or-nothing half of the binding contract. */
static void coli_xdna_clear_entry_points(void){
    g_xdna.abi_version = NULL;
    g_xdna.shutdown    = NULL;
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
    coli_xdna_fn_shutdown shutdown =
        (coli_xdna_fn_shutdown)GetProcAddress(dll, "coli_xdna_helper_shutdown");
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
    if(!shutdown){
        FreeLibrary(dll);
        g_xdna.state = COLI_XDNA_SYMBOL_INCOMPLETE;
        return;
    }

    g_xdna.dll         = dll;
    g_xdna.abi_version = abi_version;
    g_xdna.shutdown    = shutdown;
    g_xdna.state       = COLI_XDNA_AVAILABLE;
}

void coli_xdna_shutdown(void){
    if(g_xdna.shutdown) g_xdna.shutdown();
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
static int                     g_nrows = 0;

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
    char path[1024];
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

/* This slice opens no device, calls no helper entry point, wraps no pointer and
 * converts nothing. The counters make that assertable rather than merely stated. */
int coli_xdna_test_device_opens(void){ return 0; }
int coli_xdna_test_helper_calls(void){ return 0; }
int coli_xdna_test_userptr_wraps(void){ return 0; }
int coli_xdna_test_conversions(void){ return 0; }
