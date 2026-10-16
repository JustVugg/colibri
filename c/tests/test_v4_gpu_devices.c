#include <assert.h>
#include <stdio.h>
#include "../deepseek_v4_gpu_devices.h"

int main(void) {
    int devices[COLI_V4_GPU_MAX_DEVICES];
    assert(coli_v4_gpu_devices_parse("5, 2,0", devices) == 3);
    assert(devices[0] == 5 && devices[1] == 2 && devices[2] == 0);
    assert(coli_v4_gpu_devices_parse(" 0 ", devices) == 1);
    const char *bad[] = {"", " ", "0,", ",0", "0,,1", "0,0", "-1", "+1",
        "0junk", "1 2", "2147483648", "9999999999999999999999999",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"};
    for (unsigned i = 0; i < sizeof(bad) / sizeof(*bad); i++)
        assert(coli_v4_gpu_devices_parse(bad[i], devices) == -1);
    assert(coli_v4_gpu_devices_parse(NULL, devices) == -1);
    assert(coli_v4_gpu_devices_parse("0", NULL) == -1);
    assert(coli_v4_gpu_devices_parse("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", devices) == 16);
    /* Every layer belongs to exactly one nonempty, contiguous partition. */
    for (int layers = 1; layers <= 64; layers++) {
        for (int count = 1; count <= layers && count <= 16; count++) {
            int visited = 0;
            for (int owner = 0; owner < count; owner++) {
                int first = coli_v4_gpu_layer_begin(owner, count, layers);
                int end = coli_v4_gpu_layer_begin(owner + 1, count, layers);
                assert(first == visited && end > first);
                for (int layer = first; layer < end; layer++) {
                    assert(coli_v4_gpu_layer_owner(layer, count, layers) == owner);
                    visited++;
                }
            }
            assert(visited == layers);
        }
    }
    assert(coli_v4_gpu_layer_owner(-1, 6, 43) == -1);
    assert(coli_v4_gpu_layer_owner(43, 6, 43) == -1);
    assert(coli_v4_gpu_layer_owner(0, 0, 43) == -1);
    assert(coli_v4_gpu_layer_owner(0, 6, 2) == -1);
    puts("test_v4_gpu_devices: ok");
    return 0;
}
