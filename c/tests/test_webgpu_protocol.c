/* Compatibility gate for native expert workers and WebGPU proxies.
 * Both consume COLIEX01: network-order u32 headers plus raw f32 bytes. */
#define main coli_engine_main_unused
#include "../colibri.c"
#undef main

#include <assert.h>

#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
static void test_shared_wire_header(void)
{
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    uint32_t value;
    float input[2] = {1.0f, -2.0f};
    assert(webgpu_io(sockets[0], (void *)COLI_WEBGPU_MAGIC, 8, 1) == 0);
    value = 1; assert(webgpu_u32(sockets[0], &value, 1) == 0);
    value = 4; assert(webgpu_u32(sockets[0], &value, 1) == 0);
    value = 2; assert(webgpu_u32(sockets[0], &value, 1) == 0);
    value = 3; assert(webgpu_u32(sockets[0], &value, 1) == 0);
    value = 1; assert(webgpu_u32(sockets[0], &value, 1) == 0);
    value = 7; assert(webgpu_u32(sockets[0], &value, 1) == 0);
    value = 1; assert(webgpu_u32(sockets[0], &value, 1) == 0);
    assert(webgpu_io(sockets[0], input, sizeof(input), 1) == 0);

    char magic[8];
    assert(webgpu_io(sockets[1], magic, 8, 0) == 0);
    assert(memcmp(magic, COLI_WEBGPU_MAGIC, 8) == 0);
    value = 0; assert(webgpu_u32(sockets[1], &value, 0) == 0 && value == 1);
    value = 0; assert(webgpu_u32(sockets[1], &value, 0) == 0 && value == 4);
    value = 0; assert(webgpu_u32(sockets[1], &value, 0) == 0 && value == 2);
    value = 0; assert(webgpu_u32(sockets[1], &value, 0) == 0 && value == 3);
    value = 0; assert(webgpu_u32(sockets[1], &value, 0) == 0 && value == 1);
    value = 0; assert(webgpu_u32(sockets[1], &value, 0) == 0 && value == 7);
    value = 0; assert(webgpu_u32(sockets[1], &value, 0) == 0 && value == 1);
    float received[2] = {0};
    assert(webgpu_io(sockets[1], received, sizeof(received), 0) == 0);
    assert(received[0] == 1.0f && received[1] == -2.0f);
    close(sockets[0]); close(sockets[1]);
}
#endif

int main(void)
{
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    test_shared_wire_header();
    puts("webgpu protocol compatibility: ok");
#else
    puts("webgpu protocol compatibility: skipped on Windows");
#endif
    return 0;
}
