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
