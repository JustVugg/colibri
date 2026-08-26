#ifndef COLIBRI_SEGMENT_ADAPTERS_H
#define COLIBRI_SEGMENT_ADAPTERS_H

/*
 * Explicit registration entry points for Colibri's built-in model adapters.
 *
 * Each engine owns its implementation and is normally linked as a separate
 * executable/object.  A Segment consumer links the engines it wants and calls
 * the matching functions before the first adapter lookup.  The ordinary
 * Colibri CLI/server paths do not call these functions, so adding an adapter
 * cannot change standalone inference or initialization order.
 */

#ifdef __cplusplus
extern "C" {
#endif

int coli_glm_segment_adapter_register(void);
int coli_inkling_segment_adapter_register(void);
int coli_kimi_segment_adapter_register(void);
int coli_olmoe_segment_adapter_register(void);
int coli_qwen36_segment_adapter_register(void);
int coli_deepseek_v4_segment_adapter_register(void);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_SEGMENT_ADAPTERS_H */
