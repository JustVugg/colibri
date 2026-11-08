#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../compat.h"   /* setenv and unsetenv on Windows too */
#include "../coli_env.h"

int main(void) {
    const unsigned short engines[] = {CE_COLIBRI, CE_KIMI, CE_INKLING, CE_OLMOE,
        CE_DSV4, CE_QWEN, CE_GLM53, CE_QWEN38, CE_DSV41, CE_QWENIMAGE, CE_MIMO, CE_LAYA, CE_GLINER};
    for (unsigned i = 0; i < sizeof(engines) / sizeof(engines[0]); ++i) {
        assert(coli_env_find("COLI_ENV_STRICT")->engines & engines[i]);
        assert(coli_env_find("COLI_ENV_DUMP")->engines & engines[i]);
    }
    assert(coli_env_find("COLI_IMG_BITS")->engines == CE_QWENIMAGE);
    assert(coli_env_find("QWEN36_QPACK")->engines == CE_QWEN);
    assert(coli_env_is_ours("QWENIMAGE_PRF"));
    assert(coli_env_find("V41_DSPARK")->engines == CE_DSV41);
    assert(coli_env_find("V41_READ_DEPTH")->engines == CE_DSV41);
    assert(coli_env_find("Q38_TRUNK_GPU")->engines == CE_QWEN38);
    assert(coli_env_find("COLI_PLACE")->engines == (CE_QWEN | CE_QWEN38));
    assert(coli_env_find("CACHE_ROUTE")->engines & CE_QWEN);
    assert(coli_env_find("SNAP")->engines & CE_DSV41);
    assert(coli_env_find("MIMO_CAP")->engines == CE_MIMO);   /* read through mimo.c's env_int() */
    assert(coli_env_find("SNAP")->engines & CE_MIMO);
    assert(coli_env_is_ours("MIMO_CA"));
    assert(coli_env_find("COLI_LAYA_MAX_LEN")->engines == CE_LAYA);
    assert(coli_env_find("COLI_TUI_SIXEL")->engines == 0);   /* a tool's, no engine warns */
    assert(coli_env_is_ours("V41_DSPAR"));
    assert(!strcmp(coli_env_suggest("V41_DSPAR"), "V41_DSPARK"));
    assert(!coli_env_find("V41_DSPAR"));
    const char *const typos[] = {"COLI_PREFIL_CHUNK", "COLIBRI_ENGINE_SUFFI",
        "K3_BIT", "KIMI_DSA_INDEXE", "INK_BIT", "GLM53_BIT", "Q38_TRUNK_GP",
        "Q36_CTX_MA", "QWEN_EXPERT_KERNL", "DSV4_UNKNOWN", "V4_REPLAY_TRAC", "V41_DSPAR"};
    for (unsigned i = 0; i < sizeof(typos) / sizeof(typos[0]); ++i)
        assert(coli_env_is_ours(typos[i]));
    const char *const unrelated[] = {"EDITOR", "MY_VAR", "CUDA_VISIBLE_DEVICES",
        "OMP_WAIT_POLICY", "QT_QPA_PLATFORM", "VK_ICD_FILENAMES", "QWEN", "V41", "Q38X_FOO"};
    for (unsigned i = 0; i < sizeof(unrelated) / sizeof(unrelated[0]); ++i)
        assert(!coli_env_is_ours(unrelated[i]));
    assert(!strcmp(coli_env_suggest("QWEN_EXPERT_KERNL"), "QWEN_EXPERT_KERNEL"));
    /* what coli serve sets on every engine it starts is no warning on one that does not read it
     * (qwen36 reads neither NGEN nor SERVE_BATCH; COLI_SERVE_PORT is coli's own), and a user's
     * wrong-engine variable still is */
    assert(coli_env_find("NGEN")->flags & CE_LAUNCHER);
    assert(!(coli_env_find("NGEN")->engines & CE_QWEN) && !(coli_env_find("SERVE_BATCH")->engines & CE_QWEN));
    assert(!coli_env_find("COLI_SERVE_PORT") && coli_env_launcher_only("COLI_SERVE_PORT"));
    unsetenv("COLI_ENV_STRICT");
    const int base = coli_env_check(CE_QWEN, "qwen36");   /* whatever the runner's environment holds */
    setenv("NGEN", "512", 1); setenv("SERVE_BATCH", "1", 1); setenv("COLI_SERVE_PORT", "8000", 1);
    assert(coli_env_check(CE_QWEN, "qwen36") == base);
    setenv("K3_BITS", "8", 1);
    assert(!(coli_env_find("K3_BITS")->engines & CE_QWEN));
    assert(coli_env_check(CE_QWEN, "qwen36") == base + 1);
    unsetenv("K3_BITS"); setenv("COLI_SERVE_PORTT", "1", 1);
    assert(coli_env_check(CE_QWEN, "qwen36") == base + 1);
    assert(coli_env_find("PROMPT")->flags & CE_LAUNCHER);   /* cmd's own prompt on Windows */
    /* Windows' TEMP, its temporary folder, is not the deprecated temperature: no line for it */
    assert(coli_env_find("TEMP")->flags & CE_DEPRECATED);
    assert(!coli_env_meant(coli_env_find("TEMP"), "C:\\Users\\me\\AppData\\Local\\Temp"));
    assert(coli_env_meant(coli_env_find("TEMP"), "0.7"));
    puts("test_env_registry: ok");
    return 0;
}
