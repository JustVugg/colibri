/* stop_ids.h — the end-of-generation token ids of a checkpoint.
 *
 * HF checkpoints declare them in two places and agree on neither: the
 * standalone text models put `eos_token_id` at the top of config.json, the
 * multimodal wrappers put it under `text_config`, and generation_config.json
 * repeats it (as an int or a list) when the repo ships one. An engine that
 * reads only generation_config.json never stops on a converted or hand-copied
 * checkpoint that lacks the file, and produces text until --ngen runs out
 * (#1478). So: generation_config.json first, then config.json at the top
 * level, then config.json's text_config; the caller learns which one answered
 * so it can say so on stderr. Returns the number of ids written to out[]. */
#ifndef COLI_STOP_IDS_H
#define COLI_STOP_IDS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

static int coli_stop_ids_from(jval *obj, int *out, int max) {
    if (!obj || obj->t != J_OBJ) return 0;
    jval *eos = json_get(obj, "eos_token_id");
    int found = 0;
    if (eos && eos->t == J_NUM && found < max) out[found++] = (int)eos->num;
    else if (eos && eos->t == J_ARR)
        for (int i = 0; i < eos->len && found < max; i++)
            if (eos->kids[i]->t == J_NUM) out[found++] = (int)eos->kids[i]->num;
    return found;
}

static jval *coli_stop_ids_parse(const char *dir, const char *file, char **arena) {
    char path[4096]; snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    if (size <= 0 || size > (64l << 20)) { fclose(f); return NULL; }
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) { free(text); fclose(f); return NULL; }
    text[size] = 0; fclose(f);
    jval *root = json_parse(text, arena);
    free(text);
    return root;
}

/* source_out (may be NULL) names where the ids came from: "generation_config.json",
 * "config.json", "config.json (text_config)", or "" when none declared any. */
static int coli_load_stop_ids(const char *dir, int *out, int max, const char **source_out) {
    static const char *files[2] = { "generation_config.json", "config.json" };
    const char *source = "";
    int found = 0;
    for (int fi = 0; fi < 2 && found == 0; fi++) {
        char *arena = NULL;
        jval *root = coli_stop_ids_parse(dir, files[fi], &arena);
        if (root) {
            found = coli_stop_ids_from(root, out, max);
            if (found) source = files[fi];
            else if (fi == 1) {
                found = coli_stop_ids_from(json_get(root, "text_config"), out, max);
                if (found) source = "config.json (text_config)";
            }
        }
        json_free(root); free(arena);
    }
    if (source_out) *source_out = source;
    return found;
}

#endif /* COLI_STOP_IDS_H */
