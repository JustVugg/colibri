/* Architecture registry + GLM-5.2 descriptor. Model-specific metadata lives
 * here so colibri.c stays model-agnostic at the selection boundary. Adding a
 * model = add a descriptor below and bind its forward-pass hooks. */
#include <string.h>
#include "arch.h"

/* GLM-5.2 (744B MoE): DeepSeek-style compressed-KV attention (MLA), a DSA
 * "lightning indexer", and a native MTP draft head. This is the model shipped
 * with config.json architectures = ["GlmMoeDsaForCausalLM"]. */
static const ModelArch GLM_MOE_DSA = {
    .name          = "GlmMoeDsaForCausalLM",
    .model_type    = "glm_moe_dsa",
    .family        = "GLM-5.2 MoE (MLA + DSA + MTP)",
    .kv_compressed = 1,
    .has_mtp       = 1,
    .has_dsa       = 1,
    /* Official GLM-5.2 chat_template: [gMASK]<sop> once at the head of the conversation,
     * no '\n' after roles, and <think></think> after <|assistant|> DISABLES the think
     * block (nothink). Byte-identical to the template previously hardcoded in colibri.c. */
    .chat_prefix   = "[gMASK]<sop>",
    .chat_turn     = "<|user|>%s<|assistant|>%s",
    .chat_nothink  = "<think></think>",
    .chat_think    = "<think>",
    .chat_eos      = "<|endoftext|>",
    .chat_antiprompt = "<|user|>;<|assistant|>;<|observation|>;<|system|>",
};

/* The registry. New models append here. */
static const ModelArch *const REGISTRY[] = {
    &GLM_MOE_DSA,
};
enum { REGISTRY_N = (int)(sizeof(REGISTRY) / sizeof(REGISTRY[0])) };

const ModelArch *model_arch_select(const char *token){
    if(!token) return NULL;
    for(int i=0;i<REGISTRY_N;i++)
        if(strcmp(REGISTRY[i]->name, token)==0 ||
           (REGISTRY[i]->model_type && strcmp(REGISTRY[i]->model_type, token)==0))
            return REGISTRY[i];
    return NULL;
}

const char *model_arch_supported(void){
    static char buf[256];
    buf[0]=0;
    for(int i=0;i<REGISTRY_N;i++){
        if(i) strncat(buf, ", ", sizeof(buf)-strlen(buf)-1);
        strncat(buf, REGISTRY[i]->name, sizeof(buf)-strlen(buf)-1);
    }
    return buf;
}
