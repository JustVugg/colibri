/* stop_ids.h (#1478): eos ids come from generation_config.json, else from
 * config.json at the top level, else from config.json's text_config, and the
 * caller learns which; a checkpoint declaring none yields 0 and "". */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#define RMDIR(p) _rmdir(p)
#else
#include <sys/stat.h>
#include <unistd.h>
#define MKDIR(p) mkdir(p, 0755)
#define RMDIR(p) rmdir(p)
#endif
#include "../stop_ids.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
static int put(const char *dir, const char *file, const char *text) {
    char p[512]; snprintf(p, sizeof p, "%s/%s", dir, file);
    FILE *f = fopen(p, "wb"); if (!f) return -1; fputs(text, f); fclose(f); return 0;
}
static void wipe(const char *dir) {
    char p[512];
    snprintf(p, sizeof p, "%s/generation_config.json", dir); remove(p);
    snprintf(p, sizeof p, "%s/config.json", dir); remove(p);
    RMDIR(dir);
}
int main(void) {
    const char *D = "tmp_stop_ids"; int ids[8]; const char *src = NULL;
    wipe(D); CHECK(MKDIR(D) == 0);

    /* generation_config.json wins, list form, and config.json is not consulted */
    CHECK(put(D, "generation_config.json", "{\"eos_token_id\": [154820, 154827, 154829], \"pad_token_id\": 154820}") == 0);
    CHECK(put(D, "config.json", "{\"eos_token_id\": 7}") == 0);
    CHECK(coli_load_stop_ids(D, ids, 8, &src) == 3);
    CHECK(ids[0] == 154820 && ids[1] == 154827 && ids[2] == 154829 && !strcmp(src, "generation_config.json"));

    /* int form, and `max` caps a list */
    CHECK(put(D, "generation_config.json", "{\"eos_token_id\": 42}") == 0);
    CHECK(coli_load_stop_ids(D, ids, 8, &src) == 1 && ids[0] == 42);
    CHECK(put(D, "generation_config.json", "{\"eos_token_id\": [1,2,3,4]}") == 0);
    CHECK(coli_load_stop_ids(D, ids, 2, &src) == 2 && ids[1] == 2);

    /* no generation_config.json: config.json top level */
    { char p[512]; snprintf(p, sizeof p, "%s/generation_config.json", D); remove(p); }
    CHECK(coli_load_stop_ids(D, ids, 8, &src) == 1 && ids[0] == 7 && !strcmp(src, "config.json"));

    /* multimodal wrapper: only text_config declares it (the GLM-5.3-Flash layout) */
    CHECK(put(D, "config.json", "{\"model_type\": \"glm5_next\", \"text_config\": {\"eos_token_id\": [154820, 154827]}}") == 0);
    CHECK(coli_load_stop_ids(D, ids, 8, &src) == 2 && ids[1] == 154827 && !strcmp(src, "config.json (text_config)"));

    /* a generation_config.json without eos does not shadow config.json */
    CHECK(put(D, "generation_config.json", "{\"do_sample\": true}") == 0);
    CHECK(coli_load_stop_ids(D, ids, 8, &src) == 2 && !strcmp(src, "config.json (text_config)"));

    /* nothing anywhere */
    CHECK(put(D, "generation_config.json", "{}") == 0);
    CHECK(put(D, "config.json", "{\"text_config\": {}}") == 0);
    CHECK(coli_load_stop_ids(D, ids, 8, &src) == 0 && !strcmp(src, ""));
    /* and a missing directory is just zero, never a crash */
    CHECK(coli_load_stop_ids("tmp_stop_ids_missing", ids, 8, NULL) == 0);

    wipe(D);
    puts("stop_ids tests: ok");
    return 0;
}
