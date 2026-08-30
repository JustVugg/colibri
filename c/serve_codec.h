#ifndef COLI_SERVE_CODEC_H
#define COLI_SERVE_CODEC_H

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"

/* Transport only: parse and emit frames. Admission, queueing, cancellation,
 * KV ownership, scheduling, and generation remain with each family engine. */

#define COLI_SERVE_ID_CAP 64

typedef enum {
    COLI_SERVE_COMMAND_UNKNOWN = 0,
    COLI_SERVE_COMMAND_SUBMIT,
    COLI_SERVE_COMMAND_STOP,
    COLI_SERVE_COMMAND_CANCEL,
    /* IMAGE <id> <bytes> <grid_h> <grid_w>\n<payload>\n, annunciato subito prima
     * del SUBMIT a cui appartiene. Un motore di solo testo che non lo gestisce lo
     * vede come un comando che non sa trattare e lo rifiuta, che e' la risposta
     * giusta: meglio dire di no che accettare una foto e ignorarla. */
    COLI_SERVE_COMMAND_IMAGE,
} ColiServeCommandKind;

typedef enum {
    COLI_SERVE_READ_EOF = -1,
    COLI_SERVE_READ_BAD_FRAME = -2,
    COLI_SERVE_READ_NOMEM = -3,
    COLI_SERVE_READ_BAD_REQUEST = -4,
    COLI_SERVE_READ_IGNORED = 0,
    COLI_SERVE_READ_OK = 1,
} ColiServeReadResult;

typedef struct {
    size_t max_header_bytes;
    uint64_t max_payload_bytes;
    uint64_t max_extension_bytes;
    int max_tokens;
    int require_exact_lf;
    int require_finite_sampling;
    int allow_extension_bytes;
    int allow_prefix_hint;
} ColiServeWireProfile;

typedef struct {
    ColiServeCommandKind kind;
    char id[COLI_SERVE_ID_CAP];
    int slot;
    uint64_t payload_bytes;
    int max_tokens;
    float temperature;
    float top_p;
    uint64_t extension_bytes;
    int prefix_bytes;
    /* IMAGE: la griglia di patch che accompagna il payload. Zero per ogni altro
     * comando. */
    int grid_h, grid_w;
    unsigned char *payload;
} ColiServeCommand;

typedef struct {
    int completion_tokens;
    double tokens_per_second;
    double cache_hit_percent;
    double rss_gb;
    int prompt_tokens;
    int length_limited;
} ColiServeDone;

static inline void coli_serve_command_dispose(ColiServeCommand *command)
{
    if (!command) return;
    free(command->payload);
    memset(command, 0, sizeof(*command));
}

static inline void coli_serve_stdio_init(void)
{
    coli_serve_binary_mode();
    setvbuf(stdin, NULL, _IONBF, 0);
}

static inline unsigned char *coli_serve_command_take_payload(ColiServeCommand *command)
{
    unsigned char *payload = command ? command->payload : NULL;
    if (command) command->payload = NULL;
    return payload;
}

static inline unsigned char *coli_serve_command_extension(ColiServeCommand *command)
{
    if (!command || !command->payload || !command->extension_bytes) return NULL;
    return command->payload + (size_t)command->payload_bytes + 1;
}

static inline int coli_serve_parse_i32(const char *text, int *value)
{
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || !end || *end || parsed < INT32_MIN || parsed > INT32_MAX) return 0;
    *value = (int)parsed;
    return 1;
}

static inline int coli_serve_parse_u64(const char *text, uint64_t *value)
{
    char *end = NULL;
    if (!text[0] || text[0] == '-') return 0;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static inline int coli_serve_parse_f32(const char *text, float *value)
{
    char *end = NULL;
    float parsed = strtof(text, &end);
    if (!end || end == text || *end) return 0;
    *value = parsed;
    return 1;
}

static inline void coli_serve_classify_line(
    const char *line, ColiServeCommand *command)
{
    const char *cursor = line;
    while (*cursor == ' ' || *cursor == '\t') cursor++;
    const char *name = cursor;
    while (*cursor && *cursor != ' ' && *cursor != '\t') cursor++;
    size_t name_size = (size_t)(cursor - name);
    if (name_size == 6 && !memcmp(name, "SUBMIT", 6))
        command->kind = COLI_SERVE_COMMAND_SUBMIT;
    else if (name_size == 4 && !memcmp(name, "STOP", 4))
        command->kind = COLI_SERVE_COMMAND_STOP;
    else if (name_size == 6 && !memcmp(name, "CANCEL", 6))
        command->kind = COLI_SERVE_COMMAND_CANCEL;
    else if (name_size == 5 && !memcmp(name, "IMAGE", 5))
        command->kind = COLI_SERVE_COMMAND_IMAGE;
    else
        return;
    while (*cursor == ' ' || *cursor == '\t') cursor++;
    const char *id = cursor;
    while (*cursor && *cursor != ' ' && *cursor != '\t') cursor++;
    size_t id_size = (size_t)(cursor - id);
    if (id_size && id_size < sizeof(command->id)) {
        memcpy(command->id, id, id_size);
        command->id[id_size] = 0;
    }
}

static inline ColiServeReadResult coli_serve_read_command_alloc(
    FILE *input, const ColiServeWireProfile *profile, ColiServeCommand *command,
    void *(*allocate)(size_t))
{
    char *line;
    size_t count = 0;
    char *fields[10];
    int nfields = 0;

    if (!input || !profile || !command) return COLI_SERVE_READ_BAD_FRAME;
    memset(command, 0, sizeof(*command));
    size_t maximum = profile->max_header_bytes ? profile->max_header_bytes : 4096;
    if (maximum == SIZE_MAX) return COLI_SERVE_READ_NOMEM;
    line = allocate(maximum + 1);
    if (!line) return COLI_SERVE_READ_NOMEM;
    for (;;) {
        int ch = fgetc(input);
        if (ch == EOF) {
            free(line);
            return count ? COLI_SERVE_READ_BAD_FRAME : COLI_SERVE_READ_EOF;
        }
        if (ch == '\n') break;
        if (count == maximum) {
            do ch = fgetc(input); while (ch != EOF && ch != '\n');
            line[count] = 0;
            coli_serve_classify_line(line, command);
            free(line);
            return COLI_SERVE_READ_BAD_REQUEST;
        }
        line[count++] = (char)ch;
    }
    line[count] = 0;
    if (count && line[count - 1] == '\r') line[--count] = 0;
    coli_serve_classify_line(line, command);

    char *cursor = line;
    while (*cursor) {
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (!*cursor) break;
        if (nfields == (int)(sizeof(fields) / sizeof(fields[0]))) {
            free(line);
            return COLI_SERVE_READ_BAD_REQUEST;
        }
        fields[nfields++] = cursor;
        while (*cursor && *cursor != ' ' && *cursor != '\t') cursor++;
        if (*cursor) *cursor++ = 0;
    }
    if (!nfields) {
        free(line);
        return COLI_SERVE_READ_IGNORED;
    }
    if (!strcmp(fields[0], "SUBMIT")) command->kind = COLI_SERVE_COMMAND_SUBMIT;
    else if (!strcmp(fields[0], "STOP")) command->kind = COLI_SERVE_COMMAND_STOP;
    else if (!strcmp(fields[0], "CANCEL")) command->kind = COLI_SERVE_COMMAND_CANCEL;
    else if (!strcmp(fields[0], "IMAGE")) command->kind = COLI_SERVE_COMMAND_IMAGE;
    else {
        free(line);
        return COLI_SERVE_READ_IGNORED;
    }
    if (nfields < 2 || strlen(fields[1]) >= sizeof(command->id)) {
        free(line);
        return COLI_SERVE_READ_BAD_REQUEST;
    }
    memcpy(command->id, fields[1], strlen(fields[1]) + 1);
    if (command->kind == COLI_SERVE_COMMAND_STOP ||
        command->kind == COLI_SERVE_COMMAND_CANCEL) {
        if (nfields != 2) { free(line); return COLI_SERVE_READ_BAD_REQUEST; }
        free(line);
        return COLI_SERVE_READ_OK;
    }
    if (command->kind == COLI_SERVE_COMMAND_IMAGE) {
        /* IMAGE <id> <bytes> <grid_h> <grid_w>\n<payload>\n
         *
         * Il payload va consumato SEMPRE, anche quando il motore non sa cosa
         * farsene: lasciarlo nello stream sposterebbe il frame successivo di
         * qualche megabyte e il gateway leggerebbe pixel come se fossero
         * un'intestazione. Per questo il rifiuto vive nel chiamante, non qui. */
        if (nfields != 5 ||
            !coli_serve_parse_u64(fields[2], &command->payload_bytes) ||
            !coli_serve_parse_i32(fields[3], &command->grid_h) ||
            !coli_serve_parse_i32(fields[4], &command->grid_w) ||
            command->payload_bytes > profile->max_payload_bytes ||
            command->grid_h < 1 || command->grid_w < 1) {
            free(line);
            return COLI_SERVE_READ_BAD_REQUEST;
        }
        free(line);
        command->payload = (unsigned char *)allocate((size_t)command->payload_bytes + 1);
        if (!command->payload) return COLI_SERVE_READ_NOMEM;
        if (command->payload_bytes &&
            fread(command->payload, 1, (size_t)command->payload_bytes, input)
                != (size_t)command->payload_bytes) {
            coli_serve_command_dispose(command);
            return COLI_SERVE_READ_BAD_FRAME;
        }
        command->payload[command->payload_bytes] = 0;
        int terminator = fgetc(input);
        if (terminator != '\n' && !(terminator == '\r' && fgetc(input) == '\n')) {
            coli_serve_command_dispose(command);
            return COLI_SERVE_READ_BAD_FRAME;
        }
        return COLI_SERVE_READ_OK;
    }
    int minimum_fields = 7;
    int maximum_fields = minimum_fields + !!profile->allow_extension_bytes +
                         !!profile->allow_prefix_hint;
    if (nfields < minimum_fields || nfields > maximum_fields ||
        (nfields >= 8 && !profile->allow_extension_bytes) ||
        (nfields >= 9 && (!profile->allow_prefix_hint ||
                          !profile->allow_extension_bytes))) {
        free(line);
        return COLI_SERVE_READ_BAD_REQUEST;
    }
    command->extension_bytes = 0;
    command->prefix_bytes = 0;
    if (!coli_serve_parse_i32(fields[2], &command->slot) ||
        !coli_serve_parse_u64(fields[3], &command->payload_bytes) ||
        !coli_serve_parse_i32(fields[4], &command->max_tokens) ||
        !coli_serve_parse_f32(fields[5], &command->temperature) ||
        !coli_serve_parse_f32(fields[6], &command->top_p) ||
        (nfields >= 8 &&
         !coli_serve_parse_u64(fields[7], &command->extension_bytes)) ||
        (nfields >= 9 && !coli_serve_parse_i32(fields[8], &command->prefix_bytes)) ||
        command->payload_bytes > profile->max_payload_bytes ||
        command->extension_bytes > profile->max_extension_bytes ||
        command->max_tokens < 1 ||
        (profile->max_tokens && command->max_tokens > profile->max_tokens) ||
        (profile->require_finite_sampling &&
         (!isfinite(command->temperature) || !isfinite(command->top_p)))) {
        free(line);
        return COLI_SERVE_READ_BAD_REQUEST;
    }
    free(line);

    if (command->payload_bytes > SIZE_MAX - 1 ||
        command->extension_bytes > SIZE_MAX - 1 - (size_t)command->payload_bytes)
        return COLI_SERVE_READ_NOMEM;
    size_t payload_bytes = (size_t)command->payload_bytes;
    size_t extension_bytes = (size_t)command->extension_bytes;
    command->payload = allocate(payload_bytes + 1 + extension_bytes);
    if (!command->payload) return COLI_SERVE_READ_NOMEM;
    if (fread(command->payload, 1, payload_bytes, input) != payload_bytes ||
        (extension_bytes &&
         fread(command->payload + payload_bytes + 1, 1, extension_bytes, input) !=
             extension_bytes)) {
        coli_serve_command_dispose(command);
        return COLI_SERVE_READ_BAD_FRAME;
    }
    command->payload[payload_bytes] = 0;
    int terminator = fgetc(input);
    if (terminator == EOF || (profile->require_exact_lf && terminator != '\n')) {
        coli_serve_command_dispose(command);
        return COLI_SERVE_READ_BAD_FRAME;
    }
    return COLI_SERVE_READ_OK;
}

static inline ColiServeReadResult coli_serve_read_command(
    FILE *input, const ColiServeWireProfile *profile, ColiServeCommand *command)
{
    return coli_serve_read_command_alloc(input, profile, command, malloc);
}

static inline int coli_serve_write_ready(FILE *output, double rss_gb)
{
    if (fputs("\x01\x01READY\x01\x01\n", output) == EOF ||
        fprintf(output, "STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb) < 0)
        return 0;
    return fflush(output) == 0;
}

static inline int coli_serve_write_accept(FILE *output, const char *id, int prompt_tokens)
{
    return fprintf(output, "ACCEPT %s %d\n", id, prompt_tokens) >= 0 &&
           fflush(output) == 0;
}

static inline int coli_serve_write_data(
    FILE *output, const char *id, const void *data, size_t bytes)
{
    if (fprintf(output, "DATA %s %zu\n", id, bytes) < 0 ||
        (bytes && fwrite(data, 1, bytes, output) != bytes) ||
        fputc('\n', output) == EOF)
        return 0;
    return fflush(output) == 0;
}

/* Engine-authenticated structured tool output. The payload is opaque to this
 * transport; the gateway applies the active family parser. A zero-byte frame
 * declares the sideband authoritative for the request, so ordinary DATA that
 * merely resembles a tool marker is never promoted into a tool call. */
static inline int coli_serve_write_tool(
    FILE *output, const char *id, const void *data, size_t bytes)
{
    if (fprintf(output, "TOOL %s %zu\n", id, bytes) < 0 ||
        (bytes && fwrite(data, 1, bytes, output) != bytes) ||
        fputc('\n', output) == EOF)
        return 0;
    return fflush(output) == 0;
}

static inline int coli_serve_write_error(FILE *output, const char *id, const char *message)
{
    if (fprintf(output, "ERROR %s ", id && *id ? id : "0") < 0) return 0;
    for (const unsigned char *p = (const unsigned char *)(message ? message : ""); *p; p++) {
        unsigned char ch = (*p == '\r' || *p == '\n') ? ' ' : *p;
        if (fputc(ch, output) == EOF) return 0;
    }
    return fputc('\n', output) != EOF && fflush(output) == 0;
}

static inline int coli_serve_write_done(
    FILE *output, const char *id, const ColiServeDone *done)
{
    if (fprintf(output, "DONE %s STAT %d %.3f %.1f %.2f %d %d\n", id,
                done->completion_tokens, done->tokens_per_second,
                done->cache_hit_percent, done->rss_gb, done->prompt_tokens,
                done->length_limited) < 0)
        return 0;
    return fflush(output) == 0;
}

static inline int coli_serve_write_done_i32_suffix(
    FILE *output, const char *id, const ColiServeDone *done,
    const int *suffix, size_t suffix_count)
{
    if (fprintf(output, "DONE %s STAT %d %.3f %.1f %.2f %d %d", id,
                done->completion_tokens, done->tokens_per_second,
                done->cache_hit_percent, done->rss_gb, done->prompt_tokens,
                done->length_limited) < 0)
        return 0;
    for (size_t i = 0; i < suffix_count; i++)
        if (fprintf(output, " %d", suffix[i]) < 0) return 0;
    return fputc('\n', output) != EOF && fflush(output) == 0;
}

static inline int coli_serve_format_done(
    char *output, size_t capacity, const char *id, const ColiServeDone *done)
{
    int count = snprintf(output, capacity,
                         "DONE %s STAT %d %.3f %.1f %.2f %d %d\n", id,
                         done->completion_tokens, done->tokens_per_second,
                         done->cache_hit_percent, done->rss_gb,
                         done->prompt_tokens, done->length_limited);
    return count >= 0 && (size_t)count < capacity ? count : -1;
}

#endif
