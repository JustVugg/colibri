/* Freeze qwen36's serve request framing before it adopts serve_codec.h: what the
 * reader makes of each frame the gateway sends, and what is left in the stream after
 * it. No model is loaded: the frames are synthetic bytes. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <assert.h>

static void binary_stream(FILE *stream)
{
#ifdef _WIN32
    assert(_setmode(_fileno(stream), _O_BINARY) != -1);
#else
    (void)stream;
#endif
}

static FILE *bytes_input(const void *data, size_t size)
{
    FILE *stream = tmpfile(); assert(stream); binary_stream(stream);
    assert(fwrite(data, 1, size, stream) == size); rewind(stream); return stream;
}

static FILE *text_input(const char *text) { return bytes_input(text, strlen(text)); }

static FILE *output_stream(void)
{
    FILE *stream = tmpfile(); assert(stream); binary_stream(stream); return stream;
}

/* what the reader wrote, NUL-terminated */
static const char *written(FILE *stream)
{
    static char text[512];
    assert(fflush(stream) == 0); rewind(stream);
    size_t n = fread(text, 1, sizeof text - 1, stream); text[n] = 0;
    return text;
}

static void test_submit_payload_and_fields(void)
{
    static const char payload[] = "<|im_start|>user\n你好\nline two<|im_end|>\n";
    char frame[512];
    int header = snprintf(frame, sizeof frame, "SUBMIT req-7 0 %zu 16 0.7 0.95\n", sizeof payload - 1);
    assert(header > 0);
    memcpy(frame + header, payload, sizeof payload - 1);
    frame[header + sizeof payload - 1] = '\n';
    FILE *in = bytes_input(frame, (size_t)header + sizeof payload), *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 2);
    assert(!strcmp(q.id, "req-7") && q.slot == 0 && q.max_tok == 16);
    assert(q.temp > .69f && q.temp < .71f && q.top_p > .94f && q.top_p < .96f);
    assert(q.plen == (int)sizeof payload - 1 && !memcmp(q.payload, payload, sizeof payload - 1));
    assert(q.payload[q.plen] == 0);
    assert(q.logprobs == 0 && q.pin == 0);
    assert(fgetc(in) == EOF);
    assert(!*written(out));
    free(q.payload); fclose(in); fclose(out);
}

static void test_submit_extension_keys(void)
{
    FILE *in = text_input("SUBMIT x 2 3 8 0 1 logprobs=5 pin=1\nabc\n"), *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 2);
    assert(!strcmp(q.id, "x") && q.slot == 2 && q.logprobs == 5 && q.pin == 1);
    assert(q.plen == 3 && !memcmp(q.payload, "abc", 3));
    assert(fgetc(in) == EOF && !*written(out));
    free(q.payload); fclose(in); fclose(out);
}

/* max_tokens=0 reads the prompt and stops, and only with logprobs (the scorer) */
static void test_zero_max_tokens_needs_logprobs(void)
{
    FILE *in = text_input("SUBMIT s 0 2 0 0 1 logprobs=3\nhi\n"), *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 2 && q.max_tok == 0 && q.logprobs == 3);
    assert(fgetc(in) == EOF && !*written(out));
    free(q.payload); fclose(in); fclose(out);

    in = text_input("SUBMIT t 0 2 0 0 1\nhi\n"); out = output_stream();
    ServeReq r = {0};
    assert(serve_read_req(in, out, &r) == 0 && !r.payload);
    assert(!strcmp(written(out), "ERROR t bad submit header\n"));
    fclose(in); fclose(out);
}

/* an empty prompt is a zero-byte payload and its newline */
static void test_empty_payload(void)
{
    FILE *in = text_input("SUBMIT e 0 0 4 0 1\n\nSTOP e\n"), *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 2 && q.plen == 0 && q.payload && q.payload[0] == 0);
    free(q.payload);
    ServeReq c = {0};
    assert(serve_read_req(in, out, &c) == 5 && !strcmp(c.id, "e"));
    assert(fgetc(in) == EOF && !*written(out));
    fclose(in); fclose(out);
}

static void test_stop_and_cancel(void)
{
    FILE *in = text_input("STOP a\nCANCEL b\n"), *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 5 && !strcmp(q.id, "a"));
    memset(&q, 0, sizeof q);
    assert(serve_read_req(in, out, &q) == 4 && !strcmp(q.id, "b"));
    assert(serve_read_req(in, out, &q) == -1);
    assert(!*written(out));
    fclose(in); fclose(out);
}

/* IMAGE is held for the SUBMIT that follows it; a second one replaces the first */
static void test_image_then_submit(void)
{
    static const unsigned char pixels[8] = {0, 0, 128, 63, 0, 0, 0, 64};
    char frame[256]; size_t n = 0;
    n += (size_t)snprintf(frame + n, sizeof frame - n, "IMAGE v 4 1 1\n");
    memcpy(frame + n, "\x01\x02\x03\x04", 4); n += 4; frame[n++] = '\n';
    n += (size_t)snprintf(frame + n, sizeof frame - n, "IMAGE v 8 2 1\n");
    memcpy(frame + n, pixels, 8); n += 8; frame[n++] = '\n';
    n += (size_t)snprintf(frame + n, sizeof frame - n, "SUBMIT v 0 1 4 0 1\nx\n");
    FILE *in = bytes_input(frame, n), *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 0);
    assert(g_pending_image.present && g_pending_image.bytes == 4 && g_pending_image.grid_h == 1);
    assert(serve_read_req(in, out, &q) == 0);
    assert(g_pending_image.present && g_pending_image.bytes == 8);
    assert(g_pending_image.grid_h == 2 && g_pending_image.grid_w == 1);
    assert(!memcmp(g_pending_image.patches, pixels, 8));
    assert(serve_read_req(in, out, &q) == 2 && !strcmp(q.id, "v") && q.plen == 1);
    assert(fgetc(in) == EOF && !*written(out));
    free(q.payload); q36_pending_image_clear(); fclose(in); fclose(out);
}

/* DECIDE reaches the engine only after it announced decide=1, that is with a Clef head */
static void test_decide_with_clef(void)
{
    static const char record[] = "{\"state\":\"x\",\"questions\":[]}";
    char frame[256];
    int n = snprintf(frame, sizeof frame, "DECIDE d 1 %zu\n%s\n", sizeof record - 1, record);
    g_clef = 1;
    FILE *in = bytes_input(frame, (size_t)n), *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 3);
    assert(!strcmp(q.id, "d") && q.slot == 1 && q.plen == (int)sizeof record - 1);
    assert(!memcmp(q.payload, record, sizeof record - 1) && q.payload[q.plen] == 0);
    assert(fgetc(in) == EOF && !*written(out));
    g_clef = 0;
    free(q.payload); fclose(in); fclose(out);
}

/* frames back to back, and a line the reader does not know between them */
static void test_frames_in_sequence(void)
{
    FILE *in = text_input("SUBMIT a 0 1 4 0 1\nA\nHELLO there\nSUBMIT b 1 2 4 0 1\nBB\nCANCEL a\n");
    FILE *out = output_stream();
    ServeReq q = {0};
    assert(serve_read_req(in, out, &q) == 2 && !strcmp(q.id, "a") && q.plen == 1); free(q.payload);
    memset(&q, 0, sizeof q);
    assert(serve_read_req(in, out, &q) == 0);
    assert(serve_read_req(in, out, &q) == 2 && !strcmp(q.id, "b") && q.slot == 1 && q.plen == 2); free(q.payload);
    memset(&q, 0, sizeof q);
    assert(serve_read_req(in, out, &q) == 4 && !strcmp(q.id, "a"));
    assert(serve_read_req(in, out, &q) == -1 && !*written(out));
    fclose(in); fclose(out);
}

/* The poll a turn makes between layers and tokens: only its own request's STOP and
 * CANCEL count, a CANCEL wins over a STOP, and it reads only while input is ready. */
static FILE *g_poll_in;
static int poll_ready(void)
{
    int c = fgetc(g_poll_in);
    if (c == EOF) return 0;
    ungetc(c, g_poll_in); return 1;
}

static int poll_text(const char *text, const char *id)
{
    g_poll_in = text_input(text);
    int control = serve_control_read(g_poll_in, poll_ready, id);
    assert(fgetc(g_poll_in) == EOF);
    fclose(g_poll_in); g_poll_in = NULL;
    return control;
}

static void test_turn_control_poll(void)
{
    assert(poll_text("", "a") == Q36_CONTROL_NONE);
    assert(poll_text("STOP a\n", "a") == Q36_CONTROL_STOP);
    assert(poll_text("CANCEL a\n", "a") == Q36_CONTROL_CANCEL);
    assert(poll_text("STOP b\nCANCEL b\n", "a") == Q36_CONTROL_NONE);
    assert(poll_text("STOP a\nCANCEL a\n", "a") == Q36_CONTROL_CANCEL);
    assert(poll_text("CANCEL a\nSTOP a\n", "a") == Q36_CONTROL_CANCEL);
    assert(poll_text("CANCEL b\nSTOP a\n", "a") == Q36_CONTROL_STOP);
}

int main(void)
{
    test_submit_payload_and_fields();
    test_submit_extension_keys();
    test_zero_max_tokens_needs_logprobs();
    test_empty_payload();
    test_stop_and_cancel();
    test_image_then_submit();
    test_decide_with_clef();
    test_frames_in_sequence();
    test_turn_control_poll();
    puts("test_qwen36_serve_framing: OK");
    return 0;
}
