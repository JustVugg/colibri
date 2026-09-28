#ifndef COLI_GEMMA4_TOOLS_H
#define COLI_GEMMA4_TOOLS_H

#include <stddef.h>

#define COLI_GEMMA4_TOOLS_ERROR_MAX 256
#define COLI_GEMMA4_TOOL_NAME_MAX 128

typedef struct {
    char name[COLI_GEMMA4_TOOL_NAME_MAX];
    size_t arguments_offset;
    size_t arguments_bytes;
} coli_gemma4_tool_call;

int coli_gemma4_render_tools(const char *json, size_t json_bytes,
                             char **rendered, size_t *rendered_bytes,
                             char *error, size_t error_capacity);
int coli_gemma4_parse_tool_calls(const char *generated,
                                 size_t generated_bytes,
                                 coli_gemma4_tool_call *calls,
                                 size_t capacity, size_t *count,
                                 char *error, size_t error_capacity);
int coli_gemma4_render_tool_response(const char *tool_name,
                                     const char *json, size_t json_bytes,
                                     int include_open_token,
                                     char **rendered,
                                     size_t *rendered_bytes,
                                     char *error, size_t error_capacity);

#endif
