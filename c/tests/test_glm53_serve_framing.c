/* glm53's serve request framing: what the reader makes of each frame the gateway sends,
 * what is left in the stream after it, and what the turn's poll answers (frozen before it
 * adopted serve_codec.h), and the malformed frames that now end the session instead of
 * reading the rest of the stream as commands. No model is loaded: the frames are synthetic. */
#define GLM53_NO_MAIN
#include "../glm53.c"

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

static const char *written(FILE *stream)
{
    static char text[512];
    assert(fflush(stream) == 0); rewind(stream);
    size_t n = fread(text, 1, sizeof text - 1, stream); text[n] = 0;
    return text;
}

static void test_submit_payload_and_fields(void)
{
    static const char payload[] = "[gMASK]<sop><|user|>\n你好\nline two<|assistant|>";
    char frame[512];
    int header = snprintf(frame, sizeof frame, "SUBMIT 42 0 %zu 16 0.7 0.95\n", sizeof payload - 1);
    assert(header > 0);
    memcpy(frame + header, payload, sizeof payload - 1);
    frame[header + sizeof payload - 1] = '\n';
    FILE *in = bytes_input(frame, (size_t)header + sizeof payload);
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "SUBMIT"));
    assert(q.id == 42 && q.slot == 0 && q.max_tokens == 16);
    assert(q.temp > .69f && q.temp < .71f && q.top_p > .94f && q.top_p < .96f);
    assert(q.plen == (int)sizeof payload - 1 && !memcmp(q.payload, payload, sizeof payload - 1));
    assert(q.payload[q.plen] == 0 && q.logprobs == 0 && q.pin == 0);
    assert(fgetc(in) == EOF);
    free(q.payload); fclose(in);
}

static void test_submit_extension_keys(void)
{
    FILE *in = text_input("SUBMIT 18446744073709551615 3 3 8 0 1 logprobs=5 pin=1\nabc\n");
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "SUBMIT"));
    assert(q.id == 18446744073709551615ull && q.slot == 3 && q.logprobs == 5 && q.pin == 1);
    assert(q.plen == 3 && !memcmp(q.payload, "abc", 3));
    assert(fgetc(in) == EOF);
    free(q.payload); fclose(in);
}

/* max_tokens=0 reads the prompt and stops, and only with logprobs */
static void test_zero_max_tokens_needs_logprobs(void)
{
    FILE *in = text_input("SUBMIT 7 0 2 0 0 1 logprobs=3\nhi\n");
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "SUBMIT"));
    assert(q.max_tokens == 0 && q.logprobs == 3 && q.plen == 2);
    assert(fgetc(in) == EOF);
    free(q.payload); fclose(in);
}

/* an empty prompt is a zero-byte payload and its newline (serve_one refuses it) */
static void test_empty_payload(void)
{
    FILE *in = text_input("SUBMIT 5 0 0 4 0 1\n\nSTOP 5\n");
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "SUBMIT") && q.plen == 0);
    free(q.payload);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "STOP") && q.id == 5);
    assert(fgetc(in) == EOF);
    fclose(in);
}

static void test_stop_and_cancel(void)
{
    FILE *in = text_input("STOP 1\nCANCEL 2\n");
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "STOP") && q.id == 1);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "CANCEL") && q.id == 2);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 0);
    fclose(in);
}

/* IMAGE is held for the SUBMIT that follows it */
static void test_image_then_submit(void)
{
    static const unsigned char pixels[8] = {0, 0, 128, 63, 0, 0, 0, 64};
    char frame[256]; size_t n = 0;
    n += (size_t)snprintf(frame + n, sizeof frame - n, "IMAGE 9 8 2 1\n");
    memcpy(frame + n, pixels, 8); n += 8; frame[n++] = '\n';
    n += (size_t)snprintf(frame + n, sizeof frame - n, "SUBMIT 9 0 1 4 0 1\nx\n");
    FILE *in = bytes_input(frame, n);
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "IMAGE") && q.id == 9);
    assert(g_pending.patches && g_pending.id == 9 && g_pending.bytes == 8);
    assert(g_pending.grid_h == 2 && g_pending.grid_w == 1 && !memcmp(g_pending.patches, pixels, 8));
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "SUBMIT") && q.id == 9 && q.plen == 1);
    assert(fgetc(in) == EOF);
    free(q.payload); pending_clear(); fclose(in);
}

/* frames back to back, and a line the reader does not know between them */
static void test_frames_in_sequence(void)
{
    FILE *in = text_input("SUBMIT 1 0 1 4 0 1\nA\nHELLO there\nSUBMIT 2 1 2 4 0 1\nBB\nCANCEL 1\n");
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "SUBMIT") && q.id == 1 && q.plen == 1);
    free(q.payload);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && strcmp(verb, "SUBMIT") && strcmp(verb, "CANCEL") &&
           strcmp(verb, "STOP") && strcmp(verb, "IMAGE") && strcmp(verb, "BAD_FRAME"));
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "SUBMIT") && q.id == 2 && q.slot == 1 && q.plen == 2);
    free(q.payload);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "CANCEL") && q.id == 1);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 0);
    fclose(in);
}

/* The poll a turn makes between tokens and prefill chunks */
static FILE *g_poll_in;
static int poll_ready(void)
{
    int c = fgetc(g_poll_in);
    if (c == EOF) return 0;
    ungetc(c, g_poll_in); return 1;
}

static int poll_into(const char *text, unsigned long long id, FILE *out, int *eof)
{
    g_poll_in = text_input(text);
    *eof = 0;
    int control = serve_cancel_read(g_poll_in, out, poll_ready, id, eof);
    assert(fgetc(g_poll_in) == EOF);
    fclose(g_poll_in); g_poll_in = NULL;
    return control;
}

static void test_turn_poll(void)
{
    int eof; FILE *out;
    out = output_stream();
    assert(poll_into("", 7, out, &eof) == SERVE_CTL_NONE && !eof && !*written(out)); fclose(out);
    out = output_stream();
    assert(poll_into("STOP 7\n", 7, out, &eof) == SERVE_CTL_STOP && !*written(out)); fclose(out);
    out = output_stream();
    assert(poll_into("CANCEL 7\n", 7, out, &eof) == SERVE_CTL_CANCEL && !*written(out)); fclose(out);
    out = output_stream();
    assert(poll_into("STOP 7\nCANCEL 7\n", 7, out, &eof) == SERVE_CTL_CANCEL); fclose(out);
    out = output_stream();
    assert(poll_into("CANCEL 7\nSTOP 7\n", 7, out, &eof) == SERVE_CTL_CANCEL); fclose(out);
    /* another request's CANCEL is NOT_FOUND, its STOP is ignored */
    out = output_stream();
    assert(poll_into("CANCEL 8\nSTOP 8\n", 7, out, &eof) == SERVE_CTL_NONE);
    assert(!strcmp(written(out), "ERROR 8 NOT_FOUND\n")); fclose(out);
    /* a SUBMIT while the slot is busy: consumed and refused, its payload is not a CANCEL */
    out = output_stream();
    assert(poll_into("SUBMIT 9 0 8 4 0 1\nCANCEL 7\nSTOP 7\n", 7, out, &eof) == SERVE_CTL_STOP);
    assert(!strcmp(written(out), "ERROR 9 SLOT_BUSY\n")); fclose(out);
    /* an IMAGE is held, without an answer */
    out = output_stream();
    assert(poll_into("IMAGE 9 4 1 1\nabcd\nSTOP 7\n", 7, out, &eof) == SERVE_CTL_STOP && !*written(out));
    assert(g_pending.patches && g_pending.bytes == 4);
    pending_clear(); fclose(out);
}

/* the session after a frame the stream cannot be trusted after: BAD_FRAME, then nothing. A
 * body that broke off comes back without its id (the codec drops the frame whole), so its
 * BAD_FRAME names 0; the session ends either way, and the gateway fails what was in flight. */
static void expect_session_ends(const char *frames, unsigned long long id)
{
    FILE *in = text_input(frames);
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "BAD_FRAME") && q.id == id && !q.payload);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 0);
    g53_input_lost = 0;
    fclose(in);
}

static void test_malformed_frames_end_the_session(void)
{
    expect_session_ends("SUBMIT 1 0 3 4 0 1\nabcXSTOP 1\n", 0);          /* the payload's newline is missing */
    expect_session_ends("SUBMIT 2 0 9 4 0 1\nshort", 0);                  /* the payload is cut short */
    expect_session_ends("SUBMIT 3 0 2 0 0 1\nhi\nSTOP 3\n", 3);          /* max_tokens 0 without logprobs */
    expect_session_ends("SUBMIT 4 0 5 4 0 1 ids=1\n1 2 3\nSTOP 4\n", 4); /* token ids are GLM-5.2's */
    expect_session_ends("SUBMIT 5 0 2 4 0 1 logprobs=x\nhi\nSTOP 5\n", 5);   /* a malformed key, once ignored */
    expect_session_ends("SUBMIT x 0 2 4 0 1\nhi\nSTOP 5\n", 0);          /* the id is not a number */
    expect_session_ends("IMAGE 6 4 0 1\nabcd\nSTOP 6\n", 6);             /* a grid of no rows */
}

/* a header longer than the line is one refused line, not a second command made of its tail */
static void test_overlong_line_is_one_line(void)
{
    char frame[1024]; size_t n = 600 - strlen("STOP 1");
    memset(frame, 'x', n);
    memcpy(frame + n, "STOP 1\nCANCEL 2\n", 16); n += 16;
    FILE *in = bytes_input(frame, n);
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && strcmp(verb, "STOP") && strcmp(verb, "BAD_FRAME"));
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "CANCEL") && q.id == 2);
    assert(fgetc(in) == EOF);
    fclose(in);
}

/* a DECIDE (no decision head here) is consumed and skipped */
static void test_decide_is_skipped(void)
{
    FILE *in = text_input("DECIDE 8 0 2\n{}\nSTOP 8\n");
    ServeReq q; char verb[16];
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && strcmp(verb, "SUBMIT") && !q.payload);
    assert(serve_read_req(in, &q, verb, sizeof verb) == 1 && !strcmp(verb, "STOP") && q.id == 8);
    assert(fgetc(in) == EOF);
    fclose(in);
}

/* mid-turn, a frame that broke off is answered and the input then ends: what follows it is
 * left unread, not taken for the turn's CANCEL */
static void test_turn_poll_after_a_broken_frame(void)
{
    int eof = 0; FILE *out = output_stream();
    g_poll_in = text_input("SUBMIT 9 0 3 4 0 1\nabcXCANCEL 7\n");
    assert(serve_cancel_read(g_poll_in, out, poll_ready, 7, &eof) == SERVE_CTL_NONE && eof);
    assert(fgetc(g_poll_in) == 'C');
    assert(!strcmp(written(out), "ERROR 0 BAD_FRAME\n"));
    fclose(g_poll_in); g_poll_in = NULL;
    g53_input_lost = 0; fclose(out);
}

int main(void)
{
    test_submit_payload_and_fields();
    test_submit_extension_keys();
    test_zero_max_tokens_needs_logprobs();
    test_empty_payload();
    test_stop_and_cancel();
    test_image_then_submit();
    test_frames_in_sequence();
    test_turn_poll();
    test_malformed_frames_end_the_session();
    test_overlong_line_is_one_line();
    test_decide_is_skipped();
    test_turn_poll_after_a_broken_frame();
    puts("test_glm53_serve_framing: OK");
    return 0;
}
