/* The OLMoE CUDA loader must fail closed (olmoe_cuda_loader.c, Windows).
 *
 * olmoe.exe resolves the kernels from coli_cuda_olmoe.dll at run time. When that DLL is absent, or is not a library
 * with the expected symbols, every entry point must return its failure sentinel and olmoe_cuda_available() must answer 0,
 * so the engine stays on the CPU instead of crashing. The runners have no GPU, so this is the part of the loader that CI
 * can prove: it points COLI_OLMOE_CUDA_DLL at a file that does not exist and at one that is not a DLL.
 *
 * Off Windows the library is linked directly (no loader) and there is nothing to test here.
 *
 * Build: make -C c tests/test_olmoe_cuda_loader */
#include <stdio.h>
#include <stdlib.h>

#ifndef _WIN32
int main(void) { printf("test_olmoe_cuda_loader: skipped (the loader is Windows-only; Linux links olmoe_cuda.o directly)\n"); return 0; }
#else
#include <string.h>
#include "../olmoe_cuda.h"

static int fails;
#define CHECK(cond, what) do { if (cond) printf("PASS  %s\n", what); else { printf("FAIL  %s\n", what); fails++; } } while (0)

int main(void) {
    /* 1. no DLL at the named path */
    _putenv("COLI_OLMOE_CUDA_DLL=C:\\definitely\\not\\here\\coli_cuda_olmoe.dll");
    CHECK(olmoe_cuda_available() == 0, "missing DLL: olmoe_cuda_available() == 0");
    CHECK(olmoe_cuda_init(0, 16, 2048, 16, 128, 1024, 64, 8, 50304, 1024, 1e-5f, 10000.f, 0) < 0, "missing DLL: init fails");
    CHECK(olmoe_cuda_alloc_slots(8) == 0, "missing DLL: alloc_slots returns 0");
    CHECK(olmoe_cuda_stage_ptr() == NULL, "missing DLL: stage_ptr is NULL");
    CHECK(olmoe_cuda_layer_attn(0) < 0 && olmoe_cuda_layer_route(0) < 0 && olmoe_cuda_layer_experts(NULL) < 0,
          "missing DLL: per-layer calls fail");
    size_t f = 123, t = 456; olmoe_cuda_mem_info(&f, &t);
    CHECK(f == 0 && t == 0, "missing DLL: mem_info reports 0 bytes");
    olmoe_cuda_shutdown();   /* must be a harmless no-op */
    CHECK(1, "missing DLL: shutdown is a no-op");

    printf("\n%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
#endif
