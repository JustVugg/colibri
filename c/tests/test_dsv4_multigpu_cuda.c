/* Real backend lifecycle check; pass visible CUDA ordinals as one CSV argument. */
#include <assert.h>
#include <stdio.h>
#include "../backend_cuda_dsv4.h"
#include "../deepseek_v4_gpu_devices.h"

int main(int argc, char **argv) {
    int devices[COLI_V4_GPU_MAX_DEVICES];
    int count = coli_v4_gpu_devices_parse(argc > 1 ? argv[1] : "0,1", devices);
    assert(count >= 2);
    int duplicate[] = {devices[0], devices[0]};
    assert(!dsv4_cuda_init(duplicate, 2));
    assert(dsv4_cuda_init(devices, count));
    assert(!dsv4_cuda_init(devices, count)); /* active contexts survive refusal */
    float input[4] = {1, 2, 3, 4}, output[4] = {0};
    for (int i = 0; i < count; i++) {
        Dsv4CudaActivation *a = dsv4_cuda_activation_create(devices[i], 4);
        assert(a && dsv4_cuda_activation_upload(a, input, 4));
        assert(dsv4_cuda_activation_download(output, a, 4));
        assert(dsv4_cuda_activation_sync(a));
        for (int j = 0; j < 4; j++) assert(output[j] == input[j]);
        assert(dsv4_cuda_kv_ring_append(devices[i], i, input, 0, 1, 4, 4));
        assert(dsv4_cuda_kv_comp_append(devices[i], i, input, 0, 1, 4));
        assert(dsv4_cuda_stream_drain(devices[i]));
        dsv4_cuda_activation_free(a);
    }
    dsv4_cuda_shutdown();
    /* Reopen with a different layer owner: old KV allocations must be gone. */
    int device = devices[count - 1];
    assert(dsv4_cuda_init(&device, 1));
    assert(dsv4_cuda_kv_ring_append(device, 0, input, 0, 1, 4, 4));
    assert(dsv4_cuda_kv_comp_append(device, 0, input, 0, 1, 4));
    assert(dsv4_cuda_stream_drain(device));
    dsv4_cuda_shutdown();
    dsv4_cuda_shutdown();
    puts("test_dsv4_multigpu_cuda: ok");
    return 0;
}
