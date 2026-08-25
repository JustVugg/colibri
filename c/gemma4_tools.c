#include "gemma4_tools.h"

#include "json.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    char *error;
    size_t error_capacity;
} tool_buffer;

typedef struct {
    const char *key;
    jval *value;
    int original_index;
} tool_field;

static void tool_set_error(tool_buffer *buffer, const char *format, ...) {
    va_list arguments;
    if (!buffer || !buffer->error || !buffer->error_capacity ||
        buffer->error[0]) return;
    va_start(arguments, format);
    vsnprintf(buffer->error, buffer->error_capacity, format, arguments);
    va_end(arguments);
}

static void tool_free_json(jval *value) {
    int index;
    if (!value) return;
    if (value->t == J_OBJ) {
        for (index = 0; index < value->len; ++index) {
            free(value->keys[index]);
            tool_free_json(value->kids[index]);
        }
        free(value->keys);
        free(value->kids);
    } else if (value->t == J_ARR) {
        for (index = 0; index < value->len; ++index)
            tool_free_json(value->kids[index]);
        free(value->kids);
    } else if (value->t == J_STR) {
        free(value->str);
    }
    free(value);
}

typedef struct {
    const unsigned char *cursor;
    const unsigned char *end;
    unsigned depth;
} tool_json_scanner;

static void tool_json_skip_space(tool_json_scanner *scanner) {
    while (scanner->cursor < scanner->end &&
           isspace(*scanner->cursor)) ++scanner->cursor;
}

static int tool_json_scan_value(tool_json_scanner *scanner);

static int tool_json_scan_string(tool_json_scanner *scanner) {
    if (scanner->cursor >= scanner->end || *scanner->cursor++ != '"')
        return 0;
    while (scanner->cursor < scanner->end) {
        unsigned char character = *scanner->cursor++;
        if (character == '"') return 1;
        if (character < 0x20U) return 0;
        if (character == '\\') {
            unsigned escape_index;
            if (scanner->cursor >= scanner->end) return 0;
            character = *scanner->cursor++;
            if (character == 'u') {
                for (escape_index = 0; escape_index < 4; ++escape_index) {
                    if (scanner->cursor >= scanner->end ||
                        !isxdigit(*scanner->cursor++)) return 0;
                }
            } else if (!strchr("\"\\/bfnrt", character)) {
                return 0;
            }
        }
    }
    return 0;
}

static int tool_json_scan_number(tool_json_scanner *scanner) {
    const unsigned char *start = scanner->cursor;
    if (scanner->cursor < scanner->end && *scanner->cursor == '-')
        ++scanner->cursor;
    if (scanner->cursor >= scanner->end) return 0;
    if (*scanner->cursor == '0') {
        ++scanner->cursor;
        if (scanner->cursor < scanner->end &&
            isdigit(*scanner->cursor)) return 0;
    } else {
        if (*scanner->cursor < '1' || *scanner->cursor > '9') return 0;
        while (scanner->cursor < scanner->end &&
               isdigit(*scanner->cursor)) ++scanner->cursor;
    }
    if (scanner->cursor < scanner->end && *scanner->cursor == '.') {
        ++scanner->cursor;
        if (scanner->cursor >= scanner->end ||
            !isdigit(*scanner->cursor)) return 0;
        while (scanner->cursor < scanner->end &&
               isdigit(*scanner->cursor)) ++scanner->cursor;
    }
    if (scanner->cursor < scanner->end &&
        (*scanner->cursor == 'e' || *scanner->cursor == 'E')) {
        ++scanner->cursor;
        if (scanner->cursor < scanner->end &&
            (*scanner->cursor == '+' || *scanner->cursor == '-'))
            ++scanner->cursor;
        if (scanner->cursor >= scanner->end ||
            !isdigit(*scanner->cursor)) return 0;
        while (scanner->cursor < scanner->end &&
               isdigit(*scanner->cursor)) ++scanner->cursor;
    }
    return scanner->cursor > start;
}

static int tool_json_scan_compound(tool_json_scanner *scanner, int object) {
    unsigned char close = object ? '}' : ']';
    ++scanner->cursor;
    if (++scanner->depth > 128) return 0;
    tool_json_skip_space(scanner);
    if (scanner->cursor < scanner->end && *scanner->cursor == close) {
        ++scanner->cursor;
        --scanner->depth;
        return 1;
    }
    for (;;) {
        if (object) {
            if (!tool_json_scan_string(scanner)) return 0;
            tool_json_skip_space(scanner);
            if (scanner->cursor >= scanner->end ||
                *scanner->cursor++ != ':') return 0;
        }
        if (!tool_json_scan_value(scanner)) return 0;
        tool_json_skip_space(scanner);
        if (scanner->cursor >= scanner->end) return 0;
        if (*scanner->cursor == close) {
            ++scanner->cursor;
            --scanner->depth;
            return 1;
        }
        if (*scanner->cursor++ != ',') return 0;
        tool_json_skip_space(scanner);
    }
}

static int tool_json_scan_value(tool_json_scanner *scanner) {
    size_t remaining;
    tool_json_skip_space(scanner);
    if (scanner->cursor >= scanner->end) return 0;
    remaining = (size_t)(scanner->end - scanner->cursor);
    if (*scanner->cursor == '"') return tool_json_scan_string(scanner);
    if (*scanner->cursor == '{') return tool_json_scan_compound(scanner, 1);
    if (*scanner->cursor == '[') return tool_json_scan_compound(scanner, 0);
    if (remaining >= 4 && !memcmp(scanner->cursor, "true", 4)) {
        scanner->cursor += 4;
        return 1;
    }
    if (remaining >= 5 && !memcmp(scanner->cursor, "false", 5)) {
        scanner->cursor += 5;
        return 1;
    }
    if (remaining >= 4 && !memcmp(scanner->cursor, "null", 4)) {
        scanner->cursor += 4;
        return 1;
    }
    return tool_json_scan_number(scanner);
}

static int tool_json_is_complete(const char *input, size_t input_bytes) {
    tool_json_scanner scanner;
    scanner.cursor = (const unsigned char *)input;
    scanner.end = scanner.cursor + input_bytes;
    scanner.depth = 0;
    if (!tool_json_scan_value(&scanner)) return 0;
    tool_json_skip_space(&scanner);
    return scanner.cursor == scanner.end && scanner.depth == 0;
}

static int tool_reserve(tool_buffer *buffer, size_t additional) {
    size_t needed, capacity;
    char *grown;
    if (additional > SIZE_MAX - buffer->length - 1) {
        tool_set_error(buffer, "rendered tool declarations are too large");
        return -1;
    }
    needed = buffer->length + additional + 1;
    if (needed <= buffer->capacity) return 0;
    capacity = buffer->capacity ? buffer->capacity : 256;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    grown = (char *)realloc(buffer->data, capacity);
    if (!grown) {
        tool_set_error(buffer, "out of memory rendering tool declarations");
        return -1;
    }
    buffer->data = grown;
    buffer->capacity = capacity;
    return 0;
}

static int tool_append_bytes(tool_buffer *buffer, const char *text,
                             size_t bytes) {
    if (tool_reserve(buffer, bytes) != 0) return -1;
    if (bytes) memcpy(buffer->data + buffer->length, text, bytes);
    buffer->length += bytes;
    buffer->data[buffer->length] = '\0';
    return 0;
}

static int tool_append(tool_buffer *buffer, const char *text) {
    return tool_append_bytes(buffer, text, strlen(text));
}

static int tool_field_compare(const void *left_pointer,
                              const void *right_pointer) {
    const tool_field *left = (const tool_field *)left_pointer;
    const tool_field *right = (const tool_field *)right_pointer;
    const unsigned char *a = (const unsigned char *)left->key;
    const unsigned char *b = (const unsigned char *)right->key;
    while (*a && *b) {
        int lower_a = tolower(*a), lower_b = tolower(*b);
        if (lower_a != lower_b) return lower_a < lower_b ? -1 : 1;
        ++a;
        ++b;
    }
    if (*a != *b) return *a ? 1 : -1;
    return left->original_index < right->original_index ? -1 :
           left->original_index > right->original_index ? 1 : 0;
}

static int tool_ascii_equal(const char *left, const char *right) {
    while (*left && *right) {
        if (tolower((unsigned char)*left) !=
            tolower((unsigned char)*right)) return 0;
        ++left;
        ++right;
    }
    return *left == *right;
}

static tool_field *tool_sorted_fields(jval *object) {
    tool_field *fields;
    int index;
    if (!object || object->t != J_OBJ || object->len <= 0) return NULL;
    fields = (tool_field *)malloc((size_t)object->len * sizeof(*fields));
    if (!fields) return NULL;
    for (index = 0; index < object->len; ++index) {
        fields[index].key = object->keys[index];
        fields[index].value = object->kids[index];
        fields[index].original_index = index;
    }
    qsort(fields, (size_t)object->len, sizeof(*fields), tool_field_compare);
    return fields;
}

static int tool_append_quoted(tool_buffer *buffer, const char *text) {
    return tool_append(buffer, "<|\"|>") == 0 &&
           tool_append(buffer, text ? text : "") == 0 &&
           tool_append(buffer, "<|\"|>") == 0 ? 0 : -1;
}

static int tool_append_upper_quoted(tool_buffer *buffer, const char *text) {
    size_t index;
    if (tool_append(buffer, "<|\"|>") != 0) return -1;
    for (index = 0; text && text[index]; ++index) {
        char character = (char)toupper((unsigned char)text[index]);
        if (tool_append_bytes(buffer, &character, 1) != 0) return -1;
    }
    return tool_append(buffer, "<|\"|>");
}

static int tool_format_argument(tool_buffer *buffer, jval *value,
                                int quote_keys, unsigned depth);

static int tool_format_sequence_upper(tool_buffer *buffer, jval *value,
                                      unsigned depth) {
    int index;
    if (!value || value->t != J_ARR) return -1;
    if (tool_append(buffer, "[") != 0) return -1;
    for (index = 0; index < value->len; ++index) {
        if (index && tool_append(buffer, ",") != 0) return -1;
        if (value->kids[index]->t == J_STR) {
            if (tool_append_upper_quoted(buffer, value->kids[index]->str) != 0)
                return -1;
        } else if (tool_format_argument(buffer, value->kids[index], 1,
                                        depth + 1) != 0) return -1;
    }
    return tool_append(buffer, "]");
}

static int tool_format_argument(tool_buffer *buffer, jval *value,
                                int quote_keys, unsigned depth) {
    int index;
    char number[64];
    tool_field *fields;
    if (!value || depth > 128) {
        tool_set_error(buffer, "tool schema nesting exceeds 128 levels");
        return -1;
    }
    switch (value->t) {
        case J_NULL:
            return tool_append(buffer, "null");
        case J_BOOL:
            return tool_append(buffer, value->boolean ? "true" : "false");
        case J_NUM:
            snprintf(number, sizeof(number), "%.17g", value->num);
            return tool_append(buffer, number);
        case J_STR:
            return tool_append_quoted(buffer, value->str);
        case J_ARR:
            if (tool_append(buffer, "[") != 0) return -1;
            for (index = 0; index < value->len; ++index) {
                if (index && tool_append(buffer, ",") != 0) return -1;
                if (tool_format_argument(buffer, value->kids[index], quote_keys,
                                         depth + 1) != 0) return -1;
            }
            return tool_append(buffer, "]");
        case J_OBJ:
            fields = tool_sorted_fields(value);
            if (value->len && !fields) {
                tool_set_error(buffer, "out of memory sorting tool schema");
                return -1;
            }
            if (tool_append(buffer, "{") != 0) {
                free(fields);
                return -1;
            }
            for (index = 0; index < value->len; ++index) {
                if ((index && tool_append(buffer, ",") != 0) ||
                    (quote_keys ? tool_append_quoted(buffer, fields[index].key) :
                                  tool_append(buffer, fields[index].key)) != 0 ||
                    tool_append(buffer, ":") != 0 ||
                    tool_format_argument(buffer, fields[index].value,
                                         quote_keys, depth + 1) != 0) {
                    free(fields);
                    return -1;
                }
            }
            free(fields);
            return tool_append(buffer, "}");
    }
    return -1;
}

static int tool_append_required(tool_buffer *buffer, jval *required) {
    int index;
    if (!required || required->t != J_ARR) return -1;
    if (tool_append(buffer, "required:[") != 0) return -1;
    for (index = 0; index < required->len; ++index) {
        if (required->kids[index]->t != J_STR) {
            tool_set_error(buffer, "required entries must be strings");
            return -1;
        }
        if ((index && tool_append(buffer, ",") != 0) ||
            tool_append_quoted(buffer, required->kids[index]->str) != 0)
            return -1;
    }
    return tool_append(buffer, "]");
}

static int tool_is_standard_key(const char *key) {
    return !strcmp(key, "description") || !strcmp(key, "type") ||
           !strcmp(key, "properties") || !strcmp(key, "required") ||
           !strcmp(key, "nullable");
}

static int tool_format_parameters(tool_buffer *buffer, jval *properties,
                                  int filter_keys, unsigned depth);

static int tool_format_array_items(tool_buffer *buffer, jval *items,
                                   unsigned depth) {
    tool_field *fields;
    int index, emitted = 0;
    if (!items || items->t != J_OBJ || items->len == 0) return 0;
    fields = tool_sorted_fields(items);
    if (!fields) {
        tool_set_error(buffer, "out of memory sorting array item schema");
        return -1;
    }
    if (tool_append(buffer, "items:{") != 0) goto failure;
    for (index = 0; index < items->len; ++index) {
        jval *value = fields[index].value;
        if (!value || value->t == J_NULL) continue;
        if (emitted++ && tool_append(buffer, ",") != 0) goto failure;
        if (!strcmp(fields[index].key, "properties")) {
            if (value->t != J_OBJ || tool_append(buffer, "properties:{") != 0 ||
                tool_format_parameters(buffer, value, 0, depth + 1) != 0 ||
                tool_append(buffer, "}") != 0) goto failure;
        } else if (!strcmp(fields[index].key, "required")) {
            if (tool_append_required(buffer, value) != 0) goto failure;
        } else if (!strcmp(fields[index].key, "type")) {
            if (tool_append(buffer, "type:") != 0) goto failure;
            if (value->t == J_STR) {
                if (tool_append_upper_quoted(buffer, value->str) != 0)
                    goto failure;
            } else if (value->t == J_ARR) {
                if (tool_format_sequence_upper(buffer, value, depth + 1) != 0)
                    goto failure;
            } else {
                tool_set_error(buffer, "array item type must be a string or array");
                goto failure;
            }
        } else {
            if (tool_append(buffer, fields[index].key) != 0 ||
                tool_append(buffer, ":") != 0 ||
                tool_format_argument(buffer, value, 1, depth + 1) != 0)
                goto failure;
        }
    }
    free(fields);
    return tool_append(buffer, "}");
failure:
    free(fields);
    return -1;
}

static int tool_format_property(tool_buffer *buffer, jval *value,
                                unsigned depth) {
    jval *type, *description, *enumeration, *items, *nullable;
    jval *properties, *required;
    int has_field = 0;
    if (!value || value->t != J_OBJ) {
        tool_set_error(buffer, "each tool property must be a JSON object");
        return -1;
    }
    type = json_get(value, "type");
    if (!type || type->t != J_STR || !type->str[0]) {
        tool_set_error(buffer, "each tool property needs a string type");
        return -1;
    }
    description = json_get(value, "description");
    enumeration = json_get(value, "enum");
    items = json_get(value, "items");
    nullable = json_get(value, "nullable");
    properties = json_get(value, "properties");
    required = json_get(value, "required");
    if (description && description->t != J_NULL &&
        (description->t != J_STR || description->str[0])) {
        if (description->t != J_STR ||
            tool_append(buffer, "description:") != 0 ||
            tool_append_quoted(buffer, description->str) != 0) {
            tool_set_error(buffer, "tool property description must be a string");
            return -1;
        }
        has_field = 1;
    }
    if (tool_ascii_equal(type->str, "string")) {
        if (enumeration && enumeration->t != J_NULL &&
            (enumeration->t != J_ARR || enumeration->len)) {
            if (enumeration->t != J_ARR ||
                (has_field && tool_append(buffer, ",") != 0) ||
                tool_append(buffer, "enum:") != 0 ||
                tool_format_argument(buffer, enumeration, 1, depth + 1) != 0)
                return -1;
            has_field = 1;
        }
    } else if (tool_ascii_equal(type->str, "array")) {
        if (items && items->t == J_OBJ && items->len) {
            if ((has_field && tool_append(buffer, ",") != 0) ||
                tool_format_array_items(buffer, items, depth + 1) != 0)
                return -1;
            has_field = 1;
        }
    }
    if (nullable && nullable->t == J_BOOL && nullable->boolean) {
        if ((has_field && tool_append(buffer, ",") != 0) ||
            tool_append(buffer, "nullable:true") != 0) return -1;
        has_field = 1;
    }
    if (tool_ascii_equal(type->str, "object")) {
        if (properties && properties->t == J_OBJ) {
            if ((has_field && tool_append(buffer, ",") != 0) ||
                tool_append(buffer, "properties:{") != 0 ||
                tool_format_parameters(buffer, properties, 0, depth + 1) != 0 ||
                tool_append(buffer, "}") != 0) return -1;
            has_field = 1;
        } else {
            if ((has_field && tool_append(buffer, ",") != 0) ||
                tool_append(buffer, "properties:{") != 0 ||
                tool_format_parameters(buffer, value, 1, depth + 1) != 0 ||
                tool_append(buffer, "}") != 0) return -1;
            has_field = 1;
        }
        if (required && required->t == J_ARR && required->len) {
            if ((has_field && tool_append(buffer, ",") != 0) ||
                tool_append_required(buffer, required) != 0) return -1;
            has_field = 1;
        }
    }
    if (has_field && tool_append(buffer, ",") != 0) return -1;
    if (tool_append(buffer, "type:") != 0 ||
        tool_append_upper_quoted(buffer, type->str) != 0 ||
        tool_append(buffer, "}") != 0) return -1;
    return 0;
}

static int tool_format_parameters(tool_buffer *buffer, jval *properties,
                                  int filter_keys, unsigned depth) {
    tool_field *fields;
    int index, emitted = 0;
    if (!properties || properties->t != J_OBJ || depth > 128) return -1;
    fields = tool_sorted_fields(properties);
    if (properties->len && !fields) {
        tool_set_error(buffer, "out of memory sorting tool properties");
        return -1;
    }
    for (index = 0; index < properties->len; ++index) {
        if (filter_keys && tool_is_standard_key(fields[index].key)) continue;
        if (emitted++ && tool_append(buffer, ",") != 0) goto failure;
        if (tool_append(buffer, fields[index].key) != 0 ||
            tool_append(buffer, ":{") != 0 ||
            tool_format_property(buffer, fields[index].value, depth + 1) != 0)
            goto failure;
    }
    free(fields);
    return 0;
failure:
    free(fields);
    return -1;
}

static int tool_format_function(tool_buffer *buffer, jval *tool,
                                int tool_index) {
    jval *kind, *function, *name, *description, *parameters, *response;
    jval *properties, *required, *type;
    if (!tool || tool->t != J_OBJ) goto invalid;
    kind = json_get(tool, "type");
    function = json_get(tool, "function");
    if (!kind || kind->t != J_STR || strcmp(kind->str, "function") ||
        !function || function->t != J_OBJ) goto invalid;
    name = json_get(function, "name");
    description = json_get(function, "description");
    parameters = json_get(function, "parameters");
    if (!name || name->t != J_STR || !name->str[0] ||
        (description && description->t != J_STR) ||
        (parameters && parameters->t != J_OBJ)) goto invalid;
    if (tool_append(buffer, "<|tool>declaration:") != 0 ||
        tool_append(buffer, name->str) != 0 ||
        tool_append(buffer, "{description:") != 0 ||
        tool_append_quoted(buffer, description ? description->str : "") != 0)
        return -1;
    if (parameters && parameters->len) {
        properties = json_get(parameters, "properties");
        required = json_get(parameters, "required");
        type = json_get(parameters, "type");
        if ((properties && properties->t != J_OBJ) ||
            (required && required->t != J_ARR) ||
            !type || type->t != J_STR || !type->str[0]) goto invalid;
        if (tool_append(buffer, ",parameters:{") != 0) return -1;
        if (properties && properties->len) {
            if (tool_append(buffer, "properties:{") != 0 ||
                tool_format_parameters(buffer, properties, 0, 0) != 0 ||
                tool_append(buffer, "},") != 0) return -1;
        }
        if (required && required->len) {
            if (tool_append_required(buffer, required) != 0 ||
                tool_append(buffer, ",") != 0) return -1;
        }
        if (type && type->str[0]) {
            if (tool_append(buffer, "type:") != 0 ||
                tool_append_upper_quoted(buffer, type->str) != 0 ||
                tool_append(buffer, "}") != 0) return -1;
        }
    }
    response = json_get(function, "response");
    if (response) {
        jval *response_description, *response_type;
        if (response->t != J_OBJ) goto invalid;
        response_description = json_get(response, "description");
        response_type = json_get(response, "type");
        if ((response_description && response_description->t != J_STR) ||
            !response_type || response_type->t != J_STR ||
            !tool_ascii_equal(response_type->str, "object")) goto invalid;
        if (tool_append(buffer, ",response:{") != 0) return -1;
        if (response_description && response_description->str[0]) {
            if (tool_append(buffer, "description:") != 0 ||
                tool_append_quoted(buffer, response_description->str) != 0 ||
                tool_append(buffer, ",") != 0) return -1;
        }
        if (tool_append(buffer, "type:") != 0 ||
            tool_append_upper_quoted(buffer, response_type->str) != 0 ||
            tool_append(buffer, "}") != 0) return -1;
    }
    return tool_append(buffer, "}<tool|>");
invalid:
    tool_set_error(buffer, "tool %d is not a valid function declaration",
                   tool_index);
    return -1;
}

int coli_gemma4_render_tools(const char *json, size_t json_bytes,
                             char **rendered, size_t *rendered_bytes,
                             char *error, size_t error_capacity) {
    tool_buffer buffer;
    char *input = NULL, *arena = NULL;
    jval *root = NULL;
    int index, result = -1;
    if (error && error_capacity) error[0] = '\0';
    if (!json || !rendered || !rendered_bytes ||
        (json_bytes == SIZE_MAX)) return -1;
    *rendered = NULL;
    *rendered_bytes = 0;
    memset(&buffer, 0, sizeof(buffer));
    buffer.error = error;
    buffer.error_capacity = error_capacity;
    input = (char *)malloc(json_bytes + 1);
    if (!input) {
        tool_set_error(&buffer, "out of memory reading tool declarations");
        goto cleanup;
    }
    memcpy(input, json, json_bytes);
    input[json_bytes] = '\0';
    if (!tool_json_is_complete(input, json_bytes)) {
        tool_set_error(&buffer, "tools file is not complete JSON");
        goto cleanup;
    }
    root = json_parse(input, &arena);
    if (!root || root->t != J_ARR) {
        tool_set_error(&buffer, "tools file must contain a JSON array");
        goto cleanup;
    }
    for (index = 0; index < root->len; ++index)
        if (tool_format_function(&buffer, root->kids[index], index) != 0)
            goto cleanup;
    if (!buffer.data) {
        buffer.data = (char *)malloc(1);
        if (!buffer.data) {
            tool_set_error(&buffer, "out of memory rendering tool declarations");
            goto cleanup;
        }
        buffer.data[0] = '\0';
    }
    *rendered = buffer.data;
    *rendered_bytes = buffer.length;
    buffer.data = NULL;
    result = 0;
cleanup:
    if (result != 0 && error && error_capacity && !error[0])
        snprintf(error, error_capacity, "cannot render tool declarations");
    free(buffer.data);
    tool_free_json(root);
    free(arena);
    free(input);
    return result;
}

static size_t tool_find_bytes(const char *text, size_t text_bytes,
                              size_t start, const char *needle,
                              size_t needle_bytes) {
    size_t position;
    if (!needle_bytes || start > text_bytes ||
        needle_bytes > text_bytes - start) return SIZE_MAX;
    for (position = start; position <= text_bytes - needle_bytes; ++position)
        if (!memcmp(text + position, needle, needle_bytes)) return position;
    return SIZE_MAX;
}

static int tool_valid_name(const char *name, size_t name_bytes) {
    size_t index;
    if (!name_bytes || name_bytes >= COLI_GEMMA4_TOOL_NAME_MAX) return 0;
    for (index = 0; index < name_bytes; ++index) {
        unsigned char character = (unsigned char)name[index];
        if (!isalnum(character) && character != '_' && character != '-' &&
            character != '.') return 0;
    }
    return 1;
}

int coli_gemma4_parse_tool_calls(const char *generated,
                                 size_t generated_bytes,
                                 coli_gemma4_tool_call *calls,
                                 size_t capacity, size_t *count,
                                 char *error, size_t error_capacity) {
    static const char prefix[] = "<|tool_call>call:";
    static const char close[] = "<tool_call|>";
    size_t cursor = 0, found = 0;
    if (error && error_capacity) error[0] = '\0';
    if (!generated || !calls || !count) return -1;
    *count = 0;
    while (cursor < generated_bytes) {
        size_t begin = tool_find_bytes(
            generated, generated_bytes, cursor, prefix, sizeof(prefix) - 1);
        size_t name_begin, arguments_begin, close_begin, name_bytes;
        if (begin == SIZE_MAX) break;
        name_begin = begin + sizeof(prefix) - 1;
        arguments_begin = tool_find_bytes(
            generated, generated_bytes, name_begin, "{", 1);
        close_begin = tool_find_bytes(
            generated, generated_bytes, name_begin, close, sizeof(close) - 1);
        if (arguments_begin == SIZE_MAX || close_begin == SIZE_MAX ||
            arguments_begin >= close_begin) {
            if (error && error_capacity)
                snprintf(error, error_capacity,
                         "generated tool call is missing arguments or closure");
            return -1;
        }
        name_bytes = arguments_begin - name_begin;
        if (!tool_valid_name(generated + name_begin, name_bytes)) {
            if (error && error_capacity)
                snprintf(error, error_capacity,
                         "generated tool call has an invalid function name");
            return -1;
        }
        if (found >= capacity) {
            if (error && error_capacity)
                snprintf(error, error_capacity,
                         "generated more than %zu tool calls", capacity);
            return -1;
        }
        memcpy(calls[found].name, generated + name_begin, name_bytes);
        calls[found].name[name_bytes] = '\0';
        calls[found].arguments_offset = arguments_begin;
        calls[found].arguments_bytes = close_begin - arguments_begin;
        ++found;
        cursor = close_begin + sizeof(close) - 1;
    }
    if (!found) {
        if (error && error_capacity)
            snprintf(error, error_capacity,
                     "tool-response handoff contained no complete tool call");
        return -1;
    }
    *count = found;
    return 0;
}

int coli_gemma4_render_tool_response(const char *tool_name,
                                     const char *json, size_t json_bytes,
                                     int include_open_token,
                                     char **rendered,
                                     size_t *rendered_bytes,
                                     char *error, size_t error_capacity) {
    tool_buffer buffer;
    char *input = NULL, *arena = NULL;
    jval *root = NULL;
    tool_field *fields = NULL;
    size_t name_bytes;
    int index, result = -1;
    if (error && error_capacity) error[0] = '\0';
    if (!tool_name || !json || !rendered || !rendered_bytes ||
        json_bytes == SIZE_MAX) return -1;
    *rendered = NULL;
    *rendered_bytes = 0;
    memset(&buffer, 0, sizeof(buffer));
    buffer.error = error;
    buffer.error_capacity = error_capacity;
    name_bytes = strlen(tool_name);
    if (!tool_valid_name(tool_name, name_bytes)) {
        tool_set_error(&buffer, "invalid tool response function name");
        goto cleanup;
    }
    input = (char *)malloc(json_bytes + 1);
    if (!input) {
        tool_set_error(&buffer, "out of memory reading tool response");
        goto cleanup;
    }
    memcpy(input, json, json_bytes);
    input[json_bytes] = '\0';
    if (!tool_json_is_complete(input, json_bytes)) {
        tool_set_error(&buffer, "tool response must be one complete JSON value");
        goto cleanup;
    }
    root = json_parse(input, &arena);
    if (!root) {
        tool_set_error(&buffer, "tool response must be a JSON value");
        goto cleanup;
    }
    if ((include_open_token &&
         tool_append(&buffer, "<|tool_response>") != 0) ||
        tool_append(&buffer, "response:") != 0 ||
        tool_append(&buffer, tool_name) != 0 ||
        tool_append(&buffer, "{") != 0) goto cleanup;
    if (root->t == J_OBJ) {
        fields = tool_sorted_fields(root);
        if (root->len && !fields) {
            tool_set_error(&buffer, "out of memory sorting tool response");
            goto cleanup;
        }
        for (index = 0; index < root->len; ++index) {
            if ((index && tool_append(&buffer, ",") != 0) ||
                tool_append(&buffer, fields[index].key) != 0 ||
                tool_append(&buffer, ":") != 0 ||
                tool_format_argument(&buffer, fields[index].value, 0, 0) != 0)
                goto cleanup;
        }
    } else {
        if (tool_append(&buffer, "value:") != 0 ||
            tool_format_argument(&buffer, root, 0, 0) != 0) goto cleanup;
    }
    if (tool_append(&buffer, "}<tool_response|>") != 0) goto cleanup;
    *rendered = buffer.data;
    *rendered_bytes = buffer.length;
    buffer.data = NULL;
    result = 0;
cleanup:
    if (result != 0 && error && error_capacity && !error[0])
        snprintf(error, error_capacity, "cannot render tool response");
    free(fields);
    free(buffer.data);
    tool_free_json(root);
    free(arena);
    free(input);
    return result;
}
