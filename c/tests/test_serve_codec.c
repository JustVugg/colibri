#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../compat.h"
#include "../serve_codec.h"

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

static size_t read_output(FILE *stream, unsigned char *output, size_t capacity)
{
    assert(fflush(stream) == 0);
    rewind(stream);
    return fread(output, 1, capacity, stream);
}

static const ColiServeWireProfile profile = {
    .max_header_bytes = 128,
    .max_payload_bytes = 5,
    .max_tokens = 4,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
};

static int allocation_count;
static int fail_allocation;

static void *fault_alloc(size_t size)
{
    allocation_count++;
    return allocation_count == fail_allocation ? NULL : malloc(size);
}

static void test_submit_and_controls(void)
{
    static const char frame[] = "SUBMIT req-7 0 5 4 0.7 0.95\nab\ncd\n";
    FILE *input = bytes_input(frame, sizeof(frame) - 1);
    ColiServeCommand command;
    assert(coli_serve_read_command(input, &profile, &command) == COLI_SERVE_READ_OK);
    assert(command.kind == COLI_SERVE_COMMAND_SUBMIT);
    assert(strcmp(command.id, "req-7") == 0);
    assert(command.slot == 0 && command.payload_bytes == 5 && command.max_tokens == 4);
    assert(memcmp(command.payload, "ab\ncd", 5) == 0);
    assert(fgetc(input) == EOF);
    coli_serve_command_dispose(&command);
    fclose(input);

    ColiServeWireProfile extended_profile = profile;
    extended_profile.max_extension_bytes = 4;
    extended_profile.allow_extension_bytes = 1;
    extended_profile.allow_prefix_hint = 1;
    static const char extended[] = "SUBMIT req-v4 0 3 4 0 1 4 2\nabcWXYZ\n";
    input = bytes_input(extended, sizeof(extended) - 1);
    assert(coli_serve_read_command(input, &extended_profile, &command) ==
           COLI_SERVE_READ_OK);
    assert(command.payload_bytes == 3 && command.extension_bytes == 4);
    assert(command.prefix_bytes == 2);
    assert(memcmp(command.payload, "abc", 3) == 0 && command.payload[3] == 0);
    assert(memcmp(coli_serve_command_extension(&command), "WXYZ", 4) == 0);
    coli_serve_command_dispose(&command);
    fclose(input);

    static const char subnormal[] = "SUBMIT req-8 0 1 1 1e-40 0.95\nx\n";
    input = bytes_input(subnormal, sizeof(subnormal) - 1);
    assert(coli_serve_read_command(input, &profile, &command) == COLI_SERVE_READ_OK);
    assert(command.temperature > 0.0f && isfinite(command.temperature));
    assert(command.payload_bytes == 1 && command.payload[0] == 'x');
    coli_serve_command_dispose(&command);
    fclose(input);

    static const char stop[] = "STOP req-7\n";
    input = bytes_input(stop, sizeof(stop) - 1);
    assert(coli_serve_read_command(input, &profile, &command) == COLI_SERVE_READ_OK);
    assert(command.kind == COLI_SERVE_COMMAND_STOP);
    coli_serve_command_dispose(&command);
    fclose(input);

    static const char cancel[] = "CANCEL req-7\n";
    input = bytes_input(cancel, sizeof(cancel) - 1);
    assert(coli_serve_read_command(input, &profile, &command) == COLI_SERVE_READ_OK);
    assert(command.kind == COLI_SERVE_COMMAND_CANCEL);
    coli_serve_command_dispose(&command);
    fclose(input);
}

static void test_invalid_headers_and_bodies(void)
{
    static const char *invalid[] = {
        "SUBMIT req 0 6 4 0.7 0.95\n",
        "SUBMIT req 0 5 0 0.7 0.95\n",
        "SUBMIT req 0 5 5 0.7 0.95\n",
        "SUBMIT req 0 5 4 nan 0.95\n",
        "SUBMIT req 0 5 4 0.7 inf\n",
        "SUBMIT req 0 5 4 0.7 0.95 extra\n",
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        FILE *input = bytes_input(invalid[i], strlen(invalid[i]));
        ColiServeCommand command;
        assert(coli_serve_read_command(input, &profile, &command) ==
               COLI_SERVE_READ_BAD_REQUEST);
        fclose(input);
    }

    static const char short_body[] = "SUBMIT req 0 5 4 0.7 0.95\nabc";
    FILE *input = bytes_input(short_body, sizeof(short_body) - 1);
    ColiServeCommand command;
    assert(coli_serve_read_command(input, &profile, &command) ==
           COLI_SERVE_READ_BAD_FRAME);
    fclose(input);

    static const char bad_lf[] = "SUBMIT req 0 5 4 0.7 0.95\nabcdeX";
    input = bytes_input(bad_lf, sizeof(bad_lf) - 1);
    assert(coli_serve_read_command(input, &profile, &command) ==
           COLI_SERVE_READ_BAD_FRAME);
    fclose(input);

    char long_id[COLI_SERVE_ID_CAP + 1];
    memset(long_id, 'x', sizeof(long_id) - 1);
    long_id[sizeof(long_id) - 1] = 0;
    char long_header[256];
    int length = snprintf(long_header, sizeof(long_header),
                          "SUBMIT %s 0 0 1 0.7 0.95\n\n", long_id);
    assert(length > 0 && (size_t)length < sizeof(long_header));
    input = bytes_input(long_header, (size_t)length);
    assert(coli_serve_read_command(input, &profile, &command) ==
           COLI_SERVE_READ_BAD_REQUEST);
    fclose(input);

    static const char unknown[] = "PING req\n";
    input = bytes_input(unknown, sizeof(unknown) - 1);
    assert(coli_serve_read_command(input, &profile, &command) ==
           COLI_SERVE_READ_IGNORED);
    fclose(input);

    char long_line[160];
    memset(long_line, 'x', sizeof(long_line));
    long_line[sizeof(long_line) - 2] = '\n';
    long_line[sizeof(long_line) - 1] = 0;
    input = bytes_input(long_line, sizeof(long_line) - 1);
    assert(coli_serve_read_command(input, &profile, &command) ==
           COLI_SERVE_READ_BAD_REQUEST);
    assert(fgetc(input) == EOF);
    fclose(input);
}

static void test_writer_golden_bytes(void)
{
    FILE *output = tmpfile();
    assert(output);
    binary_stream(output);
    assert(coli_serve_write_ready(output, 1.25));
    assert(coli_serve_write_accept(output, "req-7", 12));
    static const unsigned char data[] = {'A', '\n', 0, 'B'};
    assert(coli_serve_write_data(output, "req-7", data, sizeof(data)));
    assert(coli_serve_write_tool(output, "req-7", NULL, 0));
    assert(coli_serve_write_tool(output, "req-7", data, sizeof(data)));
    assert(coli_serve_write_error(output, "req-7", "bad\r\nframe"));
    ColiServeDone done = {3, 1.25, 50.0, 1.25, 7, 1};
    assert(coli_serve_write_done(output, "req-7", &done));
    char done_line[128];
    assert(coli_serve_format_done(done_line, sizeof(done_line), "req-7", &done) ==
           (int)strlen("DONE req-7 STAT 3 1.250 50.0 1.25 7 1\n"));
    assert(strcmp(done_line, "DONE req-7 STAT 3 1.250 50.0 1.25 7 1\n") == 0);
    int suffix = 5;
    assert(coli_serve_write_done_i32_suffix(output, "req-v4", &done, &suffix, 1));

    static const unsigned char expected[] =
        "\x01\x01READY\x01\x01\nSTAT 0 0.0 0.0 1.25 0 0\n"
        "ACCEPT req-7 12\n"
        "DATA req-7 4\nA\n\0B\n"
        "TOOL req-7 0\n\n"
        "TOOL req-7 4\nA\n\0B\n"
        "ERROR req-7 bad  frame\n"
        "DONE req-7 STAT 3 1.250 50.0 1.25 7 1\n"
        "DONE req-v4 STAT 3 1.250 50.0 1.25 7 1 5\n";
    unsigned char actual[sizeof(expected) + 16];
    size_t count = read_output(output, actual, sizeof(actual));
    assert(count == sizeof(expected) - 1);
    assert(memcmp(actual, expected, count) == 0);
    fclose(output);
}

static void test_allocation_failures_are_fatal_to_the_caller(void)
{
    static const char frame[] = "SUBMIT req 0 1 1 0.7 0.95\nx\n";
    for (int failure = 1; failure <= 2; failure++) {
        FILE *input = bytes_input(frame, sizeof(frame) - 1);
        ColiServeCommand command;
        allocation_count = 0;
        fail_allocation = failure;
        assert(coli_serve_read_command_alloc(input, &profile, &command, fault_alloc) ==
               COLI_SERVE_READ_NOMEM);
        assert(command.payload == NULL);
        if (failure == 1) assert(fgetc(input) == 'S');
        else assert(fgetc(input) == 'x');
        fclose(input);
    }
}

int main(void)
{
    test_submit_and_controls();
    test_invalid_headers_and_bodies();
    test_writer_golden_bytes();
    test_allocation_failures_are_fatal_to_the_caller();
    puts("serve codec tests: ok");
    return 0;
}
