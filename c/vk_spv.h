#ifndef COLIBRI_VK_SPV_H
#define COLIBRI_VK_SPV_H

/* Shared shader-path resolution for every engine that brings up the Vulkan
 * backend. This lived twice, byte-identical, as colibri.c's vk_resolve_spv and
 * kimi_k3.c's k3_vk_spv -- so #523's "COLI_VK_SHADERS may be a directory" fix
 * had to be made in both, and the next fix would have to be too. One copy means
 * the engines cannot disagree about where a shader is. */

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#ifdef __linux__
#include <unistd.h>
#endif

/* Resolve the main shader path (#523): COLI_VK_SHADERS may be the qmatmul.spv file
 * itself OR a directory containing it; unset, look alongside the binary
 * (<exedir>/shaders/, the build layout) before the historical CWD-relative
 * fallback, so launching from outside c/ works. The other shaders load as
 * siblings of the returned path (backend derive_*). */
static const char *coli_vk_resolve_spv(char *buf, size_t n) {
    const char *env = getenv("COLI_VK_SHADERS");
    struct stat st;
    if (env && *env) {
        if (!stat(env, &st) && S_ISDIR(st.st_mode)) {
            snprintf(buf, n, "%s/qmatmul.spv", env);
            return buf;
        }
        return env;
    }
#ifdef __linux__
    ssize_t k = readlink("/proc/self/exe", buf, n - 1);
    if (k > 0) {
        buf[k] = 0;
        char *sl = strrchr(buf, '/');
        if (sl && (size_t)(sl + 1 - buf) + sizeof("shaders/qmatmul.spv") <= n) {
            strcpy(sl + 1, "shaders/qmatmul.spv");
            if (!stat(buf, &st)) return buf;
        }
    }
#endif
    return "shaders/qmatmul.spv";
}

#endif /* COLIBRI_VK_SPV_H */
