/* decide_serve.h -- the engine side of the DECIDE command (docs/systemone.md,
 * "Decision engines"; docs/serve_protocol.md).
 *
 * A decision engine does not generate. It receives one record, the state and
 * the typed questions with their options in order, and answers with one score
 * per option and the probabilities its own calibration makes of them. This
 * header owns what every such engine shares, so the next one (an encoder or a
 * decoder with a head) writes only its model:
 *
 *   - the record, parsed and checked:   decide_record_parse()
 *   - the answer, written as JSON:      decide_answers_json()
 *   - the serve loop over the mux wire: decide_serve(), which reads SUBMIT,
 *     DECIDE, STOP and CANCEL with serve_codec.h, refuses what a decision
 *     engine cannot do, and frames DECISION + DONE.
 *
 * The record (gateway -> engine), one JSON object:
 *
 *   {"state": "<text>", "state_type": "string|object|array|number|boolean",
 *    "questions": [{"id": "q", "type": "choice|score|noul",
 *                   "instructions": "<text>",
 *                   "options": [{"label": "billing", "text": "<text>|null"}, ...]}]}
 *
 * Every text is already text: a JSON state, instructions or criterion was
 * serialized by the gateway with json.dumps(value, ensure_ascii=False), the
 * serialization the reference packages apply to the same values. state_type
 * says what the caller sent, because a model may read a list (a conversation)
 * differently from a document. A noul question always has two options, false
 * then true; a score question has one option per level, level 0 first.
 *
 * The raw form ("record": "raw"), sent to an engine whose CAPS line says
 * decide_record=raw, carries the caller's values instead of the gateway's
 * readings of them, for a model whose reference renders the request itself
 * (Clef): instructions null when none were sent, a noul side the caller did
 * not describe without a "text" key (null when it was described as null), an
 * empty description kept, and JSON values written compactly with sorted keys
 * (json.dumps(value, ensure_ascii=False, separators=(",", ":"),
 * sort_keys=True)) and marked "json": true on an option. The shape and the
 * option order are the default form's.
 *
 * The answer (engine -> gateway), the DECISION payload:
 *
 *   {"answers": [{"id": "q", "logits": [...], "probs": [...],
 *                 "temperature": 1.76, "actions": {"act": 0.98, "escalate": 0.02},
 *                 "tokens": 87, "state_tokens": 40, "state_dropped": 0}],
 *    "input_tokens": 87, "engine_ms": 41.2}
 *
 * `probs` are the engine's final probabilities over the options, in the order
 * the record gave them; `logits` are the raw scores they come from. `actions`
 * is present only for a model with a policy head (Laya's act/escalate). */
#ifndef COLI_DECIDE_SERVE_H
#define COLI_DECIDE_SERVE_H

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "json.h"
#include "serve_codec.h"

enum { DECIDE_CHOICE = 0, DECIDE_SCORE = 1, DECIDE_NOUL = 2 };
enum { DECIDE_STATE_STRING = 0, DECIDE_STATE_OBJECT, DECIDE_STATE_ARRAY, DECIDE_STATE_OTHER };

/* Bounds the gateway enforces too; the engine checks them again because the
 * wire is the trust boundary. */
#define DECIDE_MAX_QUESTIONS 64
#define DECIDE_MAX_OPTIONS 255
#define DECIDE_MAX_ACTIONS 8

typedef struct {
    char *label;
    char *text;          /* NULL when the option has no description */
    int has_text;        /* the option carried a "text" key: in the raw form a noul
                          * side the caller did not describe has none */
    int text_json;       /* raw form: the text is a JSON value, not a string */
} DecideOption;

typedef struct {
    char *id;
    int type;            /* DECIDE_CHOICE | DECIDE_SCORE | DECIDE_NOUL */
    char *instructions;  /* "" when a raw record says null */
    int has_instructions;/* 0: a raw record's caller sent none */
    int n_options;
    DecideOption *options;
} DecideQuestion;

typedef struct {
    char *state;
    int state_type;
    int raw;             /* "record": "raw", the caller's own values (see above) */
    int n_questions;
    DecideQuestion *questions;
    jval *root;          /* owns every string above */
} DecideRecord;

typedef struct {
    int n;                       /* options */
    double *logits, *probs;      /* [n], owned */
    double temperature;          /* NAN when the model has none */
    int n_actions;
    const char *action_names[DECIDE_MAX_ACTIONS];
    double actions[DECIDE_MAX_ACTIONS];
    int tokens, state_tokens, state_dropped;
} DecideAnswer;

static const char *decide_type_name(int type)
{
    return type == DECIDE_CHOICE ? "choice" : type == DECIDE_SCORE ? "score" : "noul";
}

static void decide_record_free(DecideRecord *record)
{
    if (!record) return;
    for (int q = 0; q < record->n_questions; q++) free(record->questions[q].options);
    free(record->questions);
    json_free(record->root);
    memset(record, 0, sizeof(*record));
}

static void decide_answers_free(DecideAnswer *answers, int count)
{
    if (!answers) return;
    for (int i = 0; i < count; i++) { free(answers[i].logits); free(answers[i].probs); }
}

static int decide_fail(char *err, size_t cap, const char *fmt, ...)
{
    if (err && cap) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(err, cap, fmt, args);
        va_end(args);
    }
    return 0;
}

/* Parse and check a record. Returns 1 on success; 0 with a message naming the
 * field otherwise ("questions[2].options: ..."), and the record left empty. */
static int decide_record_parse(const char *json, DecideRecord *record, char *err, size_t cap)
{
    memset(record, 0, sizeof(*record));
    jval *root = json_parse_checked(json);
    if (!root || root->t != J_OBJ) {
        json_free(root);
        return decide_fail(err, cap, "record: not a JSON object");
    }
    record->root = root;
    jval *state = json_get(root, "state"), *kind = json_get(root, "state_type");
    jval *questions = json_get(root, "questions"), *form = json_get(root, "record");
    if (!state || state->t != J_STR) { decide_record_free(record); return decide_fail(err, cap, "state: must be a string"); }
    if (form && (form->t != J_STR || strcmp(form->str, "raw"))) {
        decide_record_free(record);
        return decide_fail(err, cap, "record: unknown form (the one other than the default is \"raw\")");
    }
    record->raw = form != NULL;
    record->state = state->str;
    record->state_type = DECIDE_STATE_STRING;
    if (kind && kind->t == J_STR) {
        if (!strcmp(kind->str, "object")) record->state_type = DECIDE_STATE_OBJECT;
        else if (!strcmp(kind->str, "array")) record->state_type = DECIDE_STATE_ARRAY;
        else if (strcmp(kind->str, "string")) record->state_type = DECIDE_STATE_OTHER;
    }
    if (!questions || questions->t != J_ARR || questions->len < 1 ||
        questions->len > DECIDE_MAX_QUESTIONS) {
        decide_record_free(record);
        return decide_fail(err, cap, "questions: must be an array of 1 to %d questions",
                           DECIDE_MAX_QUESTIONS);
    }
    record->questions = (DecideQuestion *)calloc((size_t)questions->len, sizeof(DecideQuestion));
    if (!record->questions) { decide_record_free(record); return decide_fail(err, cap, "out of memory"); }
    record->n_questions = questions->len;
    for (int q = 0; q < questions->len; q++) {
        jval *entry = questions->kids[q];
        DecideQuestion *out = &record->questions[q];
        jval *id = json_get(entry, "id"), *type = json_get(entry, "type");
        jval *ins = json_get(entry, "instructions"), *opts = json_get(entry, "options");
        if (!entry || entry->t != J_OBJ || !id || id->t != J_STR || !id->str[0]) {
            decide_record_free(record);
            return decide_fail(err, cap, "questions[%d]: needs a non-empty string id", q);
        }
        out->id = id->str;
        if (!type || type->t != J_STR ||
            (strcmp(type->str, "choice") && strcmp(type->str, "score") && strcmp(type->str, "noul"))) {
            decide_record_free(record);
            return decide_fail(err, cap, "questions.%s.type: must be choice, score or noul", out->id);
        }
        out->type = !strcmp(type->str, "choice") ? DECIDE_CHOICE
                  : !strcmp(type->str, "score") ? DECIDE_SCORE : DECIDE_NOUL;
        if (record->raw && ins && ins->t == J_NULL) {
            out->instructions = (char *)"";
        } else if (!ins || ins->t != J_STR) {
            decide_record_free(record);
            return decide_fail(err, cap, "questions.%s.instructions: must be a string%s", out->id,
                               record->raw ? " or null" : "");
        } else {
            out->instructions = ins->str;
            out->has_instructions = 1;
        }
        if (!opts || opts->t != J_ARR || opts->len < 1 || opts->len > DECIDE_MAX_OPTIONS ||
            (out->type == DECIDE_NOUL && opts->len != 2)) {
            decide_record_free(record);
            return decide_fail(err, cap, "questions.%s.options: %s", out->id,
                               out->type == DECIDE_NOUL ? "a noul question has exactly two options, false then true"
                                                        : "must be an array of 1 to 255 options");
        }
        out->options = (DecideOption *)calloc((size_t)opts->len, sizeof(DecideOption));
        if (!out->options) { decide_record_free(record); return decide_fail(err, cap, "out of memory"); }
        out->n_options = opts->len;
        for (int o = 0; o < opts->len; o++) {
            jval *option = opts->kids[o];
            jval *label = json_get(option, "label"), *text = json_get(option, "text");
            jval *is_json = json_get(option, "json");
            if (!option || option->t != J_OBJ || !label || label->t != J_STR ||
                (text && text->t != J_STR && text->t != J_NULL)) {
                decide_record_free(record);
                return decide_fail(err, cap, "questions.%s.options[%d]: needs a string label "
                                   "and a string or null text", out->id, o);
            }
            if (is_json && (is_json->t != J_BOOL || (is_json->boolean && (!text || text->t != J_STR)))) {
                decide_record_free(record);
                return decide_fail(err, cap, "questions.%s.options[%d]: \"json\" marks a string text "
                                   "as a JSON value", out->id, o);
            }
            out->options[o].label = label->str;
            out->options[o].text = text && text->t == J_STR ? text->str : NULL;
            out->options[o].has_text = text != NULL;
            out->options[o].text_json = is_json && is_json->boolean;
        }
    }
    return 1;
}

/* ---- the answer, as JSON -------------------------------------------------- */

typedef struct { char *data; size_t length, capacity; int failed; } DecideBuf;

static void decide_buf_put(DecideBuf *buffer, const char *bytes, size_t count)
{
    if (buffer->failed) return;
    if (buffer->length + count + 1 > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 1024;
        while (capacity < buffer->length + count + 1) capacity *= 2;
        char *grown = (char *)realloc(buffer->data, capacity);
        if (!grown) { buffer->failed = 1; return; }
        buffer->data = grown;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->length, bytes, count);
    buffer->length += count;
    buffer->data[buffer->length] = 0;
}

static void decide_buf_printf(DecideBuf *buffer, const char *fmt, ...)
{
    char small[256];
    va_list args;
    va_start(args, fmt);
    int count = vsnprintf(small, sizeof(small), fmt, args);
    va_end(args);
    if (count < 0) { buffer->failed = 1; return; }
    if ((size_t)count < sizeof(small)) { decide_buf_put(buffer, small, (size_t)count); return; }
    char *large = (char *)malloc((size_t)count + 1);
    if (!large) { buffer->failed = 1; return; }
    va_start(args, fmt);
    vsnprintf(large, (size_t)count + 1, fmt, args);
    va_end(args);
    decide_buf_put(buffer, large, (size_t)count);
    free(large);
}

static void decide_buf_string(DecideBuf *buffer, const char *text)
{
    decide_buf_put(buffer, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == '"' || *p == '\\') { char escaped[2] = {'\\', (char)*p}; decide_buf_put(buffer, escaped, 2); }
        else if (*p < 0x20) decide_buf_printf(buffer, "\\u%04x", *p);
        else decide_buf_put(buffer, (const char *)p, 1);
    }
    decide_buf_put(buffer, "\"", 1);
}

/* JSON has no NaN or infinity; a non-finite number is written as null. */
static void decide_buf_number(DecideBuf *buffer, double value)
{
    if (isfinite(value)) decide_buf_printf(buffer, "%.9g", value);
    else decide_buf_put(buffer, "null", 4);
}

static char *decide_answers_json(const DecideRecord *record, const DecideAnswer *answers,
                                 int input_tokens, double engine_ms, size_t *length)
{
    DecideBuf buffer = {0};
    decide_buf_put(&buffer, "{\"answers\":[", 12);
    for (int q = 0; q < record->n_questions; q++) {
        const DecideAnswer *answer = &answers[q];
        if (q) decide_buf_put(&buffer, ",", 1);
        decide_buf_put(&buffer, "{\"id\":", 6);
        decide_buf_string(&buffer, record->questions[q].id);
        decide_buf_put(&buffer, ",\"logits\":[", 11);
        for (int o = 0; o < answer->n; o++) {
            if (o) decide_buf_put(&buffer, ",", 1);
            decide_buf_number(&buffer, answer->logits[o]);
        }
        decide_buf_put(&buffer, "],\"probs\":[", 11);
        for (int o = 0; o < answer->n; o++) {
            if (o) decide_buf_put(&buffer, ",", 1);
            decide_buf_number(&buffer, answer->probs[o]);
        }
        decide_buf_put(&buffer, "],\"temperature\":", 16);
        decide_buf_number(&buffer, answer->temperature);
        if (answer->n_actions > 0) {
            decide_buf_put(&buffer, ",\"actions\":{", 12);
            for (int a = 0; a < answer->n_actions; a++) {
                if (a) decide_buf_put(&buffer, ",", 1);
                decide_buf_string(&buffer, answer->action_names[a]);
                decide_buf_put(&buffer, ":", 1);
                decide_buf_number(&buffer, answer->actions[a]);
            }
            decide_buf_put(&buffer, "}", 1);
        }
        decide_buf_printf(&buffer, ",\"tokens\":%d,\"state_tokens\":%d,\"state_dropped\":%d}",
                          answer->tokens, answer->state_tokens, answer->state_dropped);
    }
    decide_buf_printf(&buffer, "],\"input_tokens\":%d,\"engine_ms\":%.3f}", input_tokens, engine_ms);
    if (buffer.failed) { free(buffer.data); return NULL; }
    *length = buffer.length;
    return buffer.data;
}

/* ---- the serve loop ------------------------------------------------------- */

/* One record in, one answer per question out. Returns 1 on success. On 0,
 * `err` holds the reason; a message starting with "questions" or "state" is a
 * problem with the request (the gateway answers 422), anything else is the
 * engine's own failure. */
typedef int (*DecideFn)(void *ctx, const DecideRecord *record, DecideAnswer *answers,
                        int *input_tokens, char *err, size_t cap);

static double decide_now_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1e3 + now.tv_nsec / 1e6;
}

/* The answer to a record the engine refuses: "ERROR <id> DECIDE_INVALID <reason>"
 * for the caller's mistake, "ERROR <id> DECIDE_FAILED <reason>" for the engine's. */
static void decide_write_refusal(FILE *output, const char *id, const char *code, const char *reason)
{
    size_t size = strlen(code) + strlen(reason) + 2;
    char *message = (char *)malloc(size);
    if (!message) { coli_serve_write_error(output, id, code); return; }
    snprintf(message, size, "%s %s", code, reason);
    coli_serve_write_error(output, id, message);
    free(message);
}

static int decide_is_request_error(const char *reason)
{
    return !strncmp(reason, "questions", 9) || !strncmp(reason, "state", 5) ||
           !strncmp(reason, "record", 6);
}

/* Serve DECIDE until stdin closes. `caps` is the engine's CAPS line, which must
 * say decide=1 (the gateway sends DECIDE to nothing else) and chat=0 for an
 * engine that cannot generate. `rss_gb` goes on the handshake's STAT line. */
static int decide_serve(FILE *input, FILE *output, DecideFn decide, void *ctx,
                        const char *caps, double rss_gb, int kv_slots)
{
    const ColiServeWireProfile profile = {
        .max_header_bytes = 4096,
        .max_payload_bytes = (uint64_t)64 << 20,
        .max_tokens = 0,
        .require_exact_lf = 0,
        .require_finite_sampling = 0,
    };
    if (!coli_serve_write_ready_caps(output, rss_gb, caps)) return 1;
    for (;;) {
        ColiServeCommand command;
        ColiServeReadResult result = coli_serve_read_command(input, &profile, &command);
        if (result == COLI_SERVE_READ_EOF) return 0;
        if (result == COLI_SERVE_READ_IGNORED) continue;
        if (result == COLI_SERVE_READ_NOMEM || result == COLI_SERVE_READ_BAD_FRAME) {
            coli_serve_write_error(output, command.id, "BAD_FRAME");
            coli_serve_command_dispose(&command);
            return 1;               /* the stream is no longer aligned on a header */
        }
        if (result == COLI_SERVE_READ_BAD_REQUEST) {
            coli_serve_write_error(output, command.id, "BAD_REQUEST");
            coli_serve_command_dispose(&command);
            continue;
        }
        switch (command.kind) {
        case COLI_SERVE_COMMAND_SUBMIT:
        case COLI_SERVE_COMMAND_IMAGE:
            /* Nothing to generate with: say so in the frame the gateway reads. */
            coli_serve_write_error(output, command.id, "NOT_SUPPORTED a decision engine does not generate text");
            break;
        case COLI_SERVE_COMMAND_CANCEL:
            /* A DECIDE runs to its end before the next command is read, so there is
             * never a request in flight to cancel. */
            coli_serve_write_error(output, command.id, "NOT_FOUND");
            break;
        case COLI_SERVE_COMMAND_STOP:
            break;
        case COLI_SERVE_COMMAND_DECIDE: {
            char reason[1024] = "";
            DecideRecord record;
            if (command.slot >= kv_slots) {
                coli_serve_write_error(output, command.id, "BAD_REQUEST");
                break;
            }
            if (!decide_record_parse((const char *)command.payload, &record, reason, sizeof(reason))) {
                decide_write_refusal(output, command.id, "DECIDE_INVALID", reason);
                break;
            }
            DecideAnswer *answers = (DecideAnswer *)calloc((size_t)record.n_questions, sizeof(DecideAnswer));
            int input_tokens = 0;
            double started = decide_now_ms();
            int ok = answers && decide(ctx, &record, answers, &input_tokens, reason, sizeof(reason));
            double elapsed = decide_now_ms() - started;
            if (!ok) {
                if (!answers) snprintf(reason, sizeof(reason), "out of memory");
                decide_write_refusal(output, command.id,
                                     decide_is_request_error(reason) ? "DECIDE_INVALID" : "DECIDE_FAILED",
                                     reason);
            } else {
                size_t length = 0;
                char *json = decide_answers_json(&record, answers, input_tokens, elapsed, &length);
                if (!json) {
                    coli_serve_write_error(output, command.id, "DECIDE_FAILED out of memory");
                } else {
                    ColiServeDone done = {0, elapsed > 0 ? input_tokens / (elapsed / 1e3) : 0.0,
                                          0.0, rss_gb, input_tokens, 0};
                    coli_serve_write_decision(output, command.id, json, length);
                    coli_serve_write_done(output, command.id, &done);
                    free(json);
                }
            }
            decide_answers_free(answers, answers ? record.n_questions : 0);
            free(answers);
            decide_record_free(&record);
            break;
        }
        default:
            coli_serve_write_error(output, command.id, "BAD_REQUEST");
            break;
        }
        coli_serve_command_dispose(&command);
    }
}

#endif
