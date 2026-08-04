#ifndef COLIBRI_ARCH_H
#define COLIBRI_ARCH_H
/* Model-architecture interface: the seam that lets a model family be *selected*
 * at load time from config.json's "architectures" field instead of being
 * hardcoded into the binary. A metadata descriptor plus a registry lookup: the
 * engine reads architectures[0] (model_type as fallback), looks it up here, and
 * refuses an unknown model with a clear message rather than mis-reading its
 * weights. The metadata fields are the invariants the per-model forward-pass
 * hooks rely on. */

typedef struct ModelArch {
    const char *name;      /* exact config.json "architectures"[0] token to match */
    const char *model_type;/* config.json "model_type" fallback: HF configs always */
                           /* carry it, but "architectures" can be null (tiny oracle) */
    const char *family;    /* human-readable label for logs                       */
    int kv_compressed;     /* 1 = MLA latent KV (GLM/DeepSeek), 0 = plain MHA/GQA  */
    int has_mtp;           /* 1 = native multi-token-prediction draft head         */
    int has_dsa;           /* 1 = DeepSeek/GLM "lightning indexer" sparse attention */

    /* --- chat template (serve + one-shot run) --- the engine builds each turn as
     * [chat_prefix on the FIRST turn] + chat_turn(user_text, think_block), where
     * think_block is chat_think when THINK=1 else chat_nothink. chat_eos is the token
     * that terminates an assistant turn: it is kept as the serve stop (others are
     * filtered, #401), and its text is the default antiprompt. All const, so a NULL
     * field means "this arch has none" (e.g. Qwen has no BPE prefix). GLM and Qwen
     * fill every field; a future arch that leaves them NULL falls back to raw input. */
    const char *chat_prefix;   /* first-turn BPE prefix  (GLM "[gMASK]<sop>", Qwen "") */
    const char *chat_turn;     /* per-turn printf fmt, exactly two %s: (user, think).
                                * MUST remain a compile-time string literal in this
                                * file's static registry: it drives snprintf, so a
                                * descriptor built from anything else (config, user
                                * input) would be a formatted-output injection point. */
    const char *chat_nothink;  /* think block when thinking is OFF (the default)       */
    const char *chat_think;    /* think block when THINK=1                             */
    const char *chat_eos;      /* assistant-turn terminator token name (serve stop)    */
    const char *chat_antiprompt;/* ';'-sep text markers the decoder stops on as a backup */
                               /* when a marker is emitted as ordinary text, not the eos id */
} ModelArch;

/* Look up a registered architecture by a config.json token, matched against either
 * a descriptor's "architectures" name or its "model_type". Pass architectures[0] if
 * present, else model_type. Returns a static descriptor, or NULL if unsupported. */
const ModelArch *model_arch_select(const char *token);

/* Comma-joined list of supported architecture names, for error messages. */
const char *model_arch_supported(void);

#endif
