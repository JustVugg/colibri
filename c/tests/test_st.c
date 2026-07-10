#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc020) == -2.5f);
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc100) == -2.5f);
    CHECK(f16_to_f32(0x0001) > 0.0f);
    CHECK(isinf(f16_to_f32(0x7c00)));
    CHECK(st_hash("tensor.weight") == st_hash("tensor.weight"));
    CHECK(st_hash("tensor.weight") != st_hash("tensor.bias"));

    puts("safetensors primitive tests: ok");
    return 0;
}
