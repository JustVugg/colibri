#include <assert.h>
#include "../coli_env.h"

int main(void) {
    const unsigned short engines[] = {CE_COLIBRI, CE_KIMI, CE_INKLING, CE_OLMOE,
        CE_DSV4, CE_QWEN, CE_GLM53, CE_QWEN38, CE_DSV41};
    for (unsigned i = 0; i < sizeof(engines) / sizeof(engines[0]); ++i) {
        assert(coli_env_find("COLI_ENV_STRICT")->engines & engines[i]);
        assert(coli_env_find("COLI_ENV_DUMP")->engines & engines[i]);
    }
    assert(coli_env_find("V41_DSPARK")->engines == CE_DSV41);
    assert(coli_env_find("V41_READ_DEPTH")->engines == CE_DSV41);
    assert(coli_env_find("Q38_TRUNK_GPU")->engines == CE_QWEN38);
    assert(coli_env_find("COLI_PLACE")->engines == (CE_QWEN | CE_QWEN38));
    assert(coli_env_find("CACHE_ROUTE")->engines & CE_QWEN);
    assert(coli_env_find("SNAP")->engines & CE_DSV41);
    assert(coli_env_is_ours("V41_DSPAR"));
    assert(!strcmp(coli_env_suggest("V41_DSPAR"), "V41_DSPARK"));
    assert(!coli_env_find("V41_DSPAR"));
    puts("test_env_registry: ok");
    return 0;
}
