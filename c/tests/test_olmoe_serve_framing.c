/* Freeze OLMoE's existing request framing before moving it behind a shared
 * codec. This includes the production parser and queue, but no model weights. */
#define OLMOE_TESTING 1
#define main coli_olmoe_main_unused
#include "../olmoe.c"
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
    FILE *stream = tmpfile();
    assert(stream);
    binary_stream(stream);
    assert(fwrite(data, 1, size, stream) == size);
    rewind(stream);
    return stream;
}

static size_t read_output(FILE *stream, char *output, size_t capacity)
{
    assert(fflush(stream) == 0);
    rewind(stream);
    return fread(output, 1, capacity, stream);
}

static void reset_queue(void)
{
    for (int i = 0; i < g_qn; i++) free(g_q[i].payload);
    memset(g_q, 0, sizeof(g_q));
    g_qn = 0;
}

static void test_submit_moves_exact_payload_into_queue(void)
{
    static const char frame[] =
        "SUBMIT req-7 0 5 4 0.7 0.95\nab\ncd\n";
    FILE *in = bytes_input(frame, sizeof(frame) - 1);
    FILE *out = tmpfile();
    assert(out);
    binary_stream(out);

    assert(serve_read_cmd(in, out, NULL) == 0);
    assert(g_qn == 1);
    assert(strcmp(g_q[0].id, "req-7") == 0);
    assert(g_q[0].plen == 5);
    assert(memcmp(g_q[0].payload, "ab\ncd", 5) == 0);
    assert(g_q[0].max_tok == 4);
    assert(g_q[0].temp > .69f && g_q[0].top_p > .94f);
    assert(fgetc(in) == EOF);

    char output[64];
    assert(read_output(out, output, sizeof(output)) == 0);
    fclose(in);
    fclose(out);
    reset_queue();
}

static void test_cancel_only_matches_active_request(void)
{
    static const char cancel[] = "CANCEL req-7\n";
    FILE *out = tmpfile();
    assert(out);
    binary_stream(out);

    FILE *match = bytes_input(cancel, sizeof(cancel) - 1);
    assert(serve_read_cmd(match, out, "req-7") == 1);
    fclose(match);

    FILE *other = bytes_input(cancel, sizeof(cancel) - 1);
    assert(serve_read_cmd(other, out, "req-8") == 0);
    fclose(other);
    fclose(out);
}

static void test_errors_are_byte_exact_and_frames_are_consumed(void)
{
    static const char bad[] = "SUBMIT bad 0 -1 4 0.7 0.95\n";
    FILE *bad_in = bytes_input(bad, sizeof(bad) - 1);
    FILE *bad_out = tmpfile();
    assert(bad_out);
    binary_stream(bad_out);
    assert(serve_read_cmd(bad_in, bad_out, NULL) == -1);
    char output[128];
    size_t count = read_output(bad_out, output, sizeof(output));
    static const char bad_want[] = "ERROR bad bad submit header\n";
    assert(count == sizeof(bad_want) - 1);
    assert(memcmp(output, bad_want, count) == 0);
    fclose(bad_in);
    fclose(bad_out);

    g_qn = SRV_QMAX;
    static const char full[] = "SUBMIT full 0 1 4 0.7 0.95\nx\n";
    FILE *full_in = bytes_input(full, sizeof(full) - 1);
    FILE *full_out = tmpfile();
    assert(full_out);
    binary_stream(full_out);
    assert(serve_read_cmd(full_in, full_out, NULL) == 0);
    count = read_output(full_out, output, sizeof(output));
    static const char full_want[] = "ERROR full queue full\n";
    assert(count == sizeof(full_want) - 1);
    assert(memcmp(output, full_want, count) == 0);
    assert(fgetc(full_in) == EOF);
    fclose(full_in);
    fclose(full_out);
    g_qn = 0;
}

static void test_invalid_submit_does_not_parse_its_payload_as_control(void)
{
    static const char *frames[] = {
        "SUBMIT bad 0 13 0 0.7 0.95\nCANCEL live\n\n",
        "SUBMIT bad 0 13 4 0.7 0.95 extra extra\nCANCEL live\n\n",
    };
    for (size_t i = 0; i < sizeof(frames) / sizeof(frames[0]); i++) {
        FILE *input = bytes_input(frames[i], strlen(frames[i]));
        FILE *output = tmpfile();
        assert(output);
        binary_stream(output);

        assert(serve_read_cmd(input, output, "live") == -1);
        assert(fgetc(input) == 'C');
        char bytes[64];
        size_t count = read_output(output, bytes, sizeof(bytes));
        static const char want[] = "ERROR bad bad submit header\n";
        assert(count == sizeof(want) - 1);
        assert(memcmp(bytes, want, count) == 0);
        fclose(input);
        fclose(output);
    }

    char overlong[640];
    int prefix = snprintf(overlong, sizeof(overlong),
                          "SUBMIT bad 0 13 4 0.7 0.95 ");
    assert(prefix > 0);
    memset(overlong + prefix, 'x', 520);
    size_t offset = (size_t)prefix + 520;
    memcpy(overlong + offset, "\nCANCEL live\n\n", 15);
    offset += 15;
    FILE *input = bytes_input(overlong, offset);
    FILE *output = tmpfile();
    assert(output);
    binary_stream(output);
    assert(serve_read_cmd(input, output, "live") == -1);
    assert(fgetc(input) == 'C');
    fclose(input);
    fclose(output);
}

static void test_bad_frame_stops_before_the_next_command(void)
{
    static const char frame[] =
        "SUBMIT broken 0 1 4 0.7 0.95\nxX"
        "SUBMIT next 0 1 4 0.7 0.95\ny\n";
    FILE *input = bytes_input(frame, sizeof(frame) - 1);
    FILE *output = tmpfile();
    assert(output);
    binary_stream(output);

    assert(serve_read_cmd(input, output, NULL) == -1);
    assert(g_qn == 0);
    assert(fgetc(input) == 'S');
    char bytes[16];
    assert(read_output(output, bytes, sizeof(bytes)) == 0);
    fclose(input);
    fclose(output);
}

int main(void)
{
    test_submit_moves_exact_payload_into_queue();
    test_cancel_only_matches_active_request();
    test_errors_are_byte_exact_and_frames_are_consumed();
    test_invalid_submit_does_not_parse_its_payload_as_control();
    test_bad_frame_stops_before_the_next_command();
    puts("olmoe serve framing baseline: ok");
    return 0;
}
