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
    for (size_t item = 0; item < sizeof(expected) / sizeof(expected[0]); item++) {
        const ColiEdgeAdapter *adapter =
            coli_edge_adapter_lookup(expected[item]);
        assert(adapter != NULL);
        /* ABI v2: every family exposes the model head both as greedy
         * selection and as full logits; a new adapter that ships only one
         * of the two fails the real edge gate. */
        assert(adapter->select != NULL);
        assert(adapter->logits != NULL);
    }
    puts("every real Edge adapter registers: ok");
    return 0;
}
