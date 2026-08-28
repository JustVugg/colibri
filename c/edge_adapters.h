#ifndef COLIBRI_EDGE_ADAPTERS_H
#define COLIBRI_EDGE_ADAPTERS_H

/* Explicit registration keeps the ordinary Colibri CLI initialization and
 * Windows/MSVC builds independent from the distributed Edge runtime. */

#ifdef __cplusplus
extern "C" {
#endif

int coli_glm_edge_adapter_register(void);
int coli_glm53_edge_adapter_register(void);
int coli_inkling_edge_adapter_register(void);
int coli_kimi_edge_adapter_register(void);
int coli_olmoe_edge_adapter_register(void);
int coli_qwen36_edge_adapter_register(void);
int coli_deepseek_v4_edge_adapter_register(void);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EDGE_ADAPTERS_H */
