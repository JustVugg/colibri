/* Freeze DeepSeek V4's serving wire before adopting serve_codec.h. */
#define COLI_V4_UNIT_GENERATE_STATS
#define COLI_V4_SKIP_GENERATE_MAIN
#include "../deepseek_v4.c"

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

static size_t read_output(FILE *stream, unsigned char *output, size_t capacity)
{
    assert(fflush(stream) == 0);
    rewind(stream);
    return fread(output, 1, capacity, stream);
}

static void test_submit_shapes_and_controls(void)
{
    static const char base[] = "SUBMIT base 0 5 4 0.7 0.95\nab\ncd\n";
    FILE *input = bytes_input(base, sizeof(base) - 1);
    FILE *output = tmpfile(); assert(output); binary_stream(output);
    V4ServeRequest request = {0};
    assert(v4_serve_read_request(input, output, &request, NULL) == 2);
    assert(strcmp(request.id, "base") == 0);
    assert(request.prompt_bytes == 5 && request.extension_bytes == 0);
    assert(request.prefix_bytes == 0 && memcmp(request.prompt, "ab\ncd", 5) == 0);
    assert(fgetc(input) == EOF);
    free(request.prompt); fclose(input); fclose(output);

    static const char extended[] =
        "SUBMIT ext 0 3 4 0 1 4 2\nabcWXYZ\n";
    input = bytes_input(extended, sizeof(extended) - 1);
    output = tmpfile(); assert(output); binary_stream(output);
    memset(&request, 0, sizeof(request));
    assert(v4_serve_read_request(input, output, &request, NULL) == 2);
    assert(request.prompt_bytes == 3 && request.extension_bytes == 4);
    assert(request.prefix_bytes == 2 && memcmp(request.prompt, "abc", 3) == 0);
    assert(memcmp(request.prompt + request.prompt_bytes + 1, "WXYZ", 4) == 0);
    assert(fgetc(input) == EOF);
    free(request.prompt); fclose(input); fclose(output);

    static const char stop[] = "STOP live\n";
    input = bytes_input(stop, sizeof(stop) - 1);
    output = tmpfile(); assert(output); binary_stream(output);
    assert(v4_serve_read_request(input, output, &request, "live") == 1);
    fclose(input); fclose(output);

    static const char cancel[] = "CANCEL live\n";
    input = bytes_input(cancel, sizeof(cancel) - 1);
    output = tmpfile(); assert(output); binary_stream(output);
    assert(v4_serve_read_request(input, output, &request, "other") == 0);
    fclose(input); fclose(output);
}

static void test_errors_and_writer_bytes(void)
{
    static const char bad[] = "SUBMIT bad 1 0 4 0 1\n\n";
    FILE *input = bytes_input(bad, sizeof(bad) - 1);
    FILE *output = tmpfile(); assert(output); binary_stream(output);
    V4ServeRequest request = {0};
    assert(v4_serve_read_request(input, output, &request, NULL) == -2);
    unsigned char actual[512];
    size_t count = read_output(output, actual, sizeof(actual));
    static const char bad_want[] = "ERROR bad bad submit header\n";
    assert(count == sizeof(bad_want) - 1);
    assert(memcmp(actual, bad_want, count) == 0);
    fclose(input); fclose(output);

    output = tmpfile(); assert(output); binary_stream(output);
    v4_serve_data(output, "req-7", "A\n\0B", 4);
    v4_serve_error(output, "req-7", "bad\r\nframe");
    v4_serve_done(output, "req-7", 3, 1.25, 50.0, 1.25, 7, 1, 5);
    static const unsigned char want[] =
        "DATA req-7 4\nA\n\0B\n"
        "ERROR req-7 bad  frame\n"
        "DONE req-7 STAT 3 1.250 50.0 1.25 7 1 5\n";
    count = read_output(output, actual, sizeof(actual));
    assert(count == sizeof(want) - 1 && memcmp(actual, want, count) == 0);
    fclose(output);
}

static void test_extension_rejection_precedes_accept(void)
{
    V4ServeRequest request = {
        .prompt = "abc",
        .prompt_bytes = 3,
        .max_tokens = 4,
        .extension_bytes = 4,
    };
    snprintf(request.id, sizeof(request.id), "ext");
    request.temperature = 0.0f;
    request.top_p = 1.0f;

    FILE *output = tmpfile(); assert(output); binary_stream(output);
    int saved = dup(fileno(stdout)); assert(saved >= 0);
    assert(fflush(stdout) == 0);
    assert(dup2(fileno(output), fileno(stdout)) >= 0);
    assert(v4_serve_one(NULL, NULL, &request) == 0);
    assert(fflush(stdout) == 0);
    assert(dup2(saved, fileno(stdout)) >= 0); close(saved);

    unsigned char actual[128];
    size_t count = read_output(output, actual, sizeof(actual));
    static const char want[] = "ERROR ext unsupported request extension\n";
    assert(count == sizeof(want) - 1 && memcmp(actual, want, count) == 0);
    fclose(output);
}

int main(void)
{
    test_submit_shapes_and_controls();
    test_errors_and_writer_bytes();
    test_extension_rejection_precedes_accept();
    puts("v4 serve framing baseline: ok");
    return 0;
}
