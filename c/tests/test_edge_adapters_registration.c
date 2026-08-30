#include "../edge_adapters.h"
#include "../edge_runtime.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    assert(coli_glm_edge_adapter_register() == 0);
    assert(coli_glm53_edge_adapter_register() == 0);
    assert(coli_inkling_edge_adapter_register() == 0);
    assert(coli_kimi_edge_adapter_register() == 0);
    assert(coli_olmoe_edge_adapter_register() == 0);
    assert(coli_qwen36_edge_adapter_register() == 0);
    assert(coli_qwen38_edge_adapter_register() == 0);
    assert(coli_deepseek_v4_edge_adapter_register() == 0);
    assert(coli_edge_adapter_count() == 8);

    static const char *expected[] = {
        "glm", "glm53", "inkling", "kimi", "olmoe", "qwen36", "qwen38",
        "deepseek_v4"
    };
    for (size_t item = 0; item < sizeof(expected) / sizeof(expected[0]); item++)
        assert(coli_edge_adapter_lookup(expected[item]) != NULL);
    puts("every real Edge adapter registers: ok");
    return 0;
}
