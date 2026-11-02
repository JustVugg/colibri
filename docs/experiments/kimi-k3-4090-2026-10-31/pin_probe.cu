// pin_probe.cu: how much cudaHostAlloc(Mapped|Portable) memory the driver gives in 1 GiB chunks,
// and 17.5 MB host-to-device copies from it (one at a time, waited for, as k3cu_moe_expert does;
// then back to back) against copies from ordinary memory.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
    const int want = argc > 1 ? atoi(argv[1]) : 64;
    const size_t G = (size_t)1 << 30, E = (size_t)17u * 1024 * 1024 + 512 * 1024;   // 17.5 MB
    std::vector<char *> ch;
    double t0 = now();
    for (int i = 0; i < want; i++) {
        void *p = nullptr;
        cudaError_t e = cudaHostAlloc(&p, G, cudaHostAllocMapped | cudaHostAllocPortable);
        if (e != cudaSuccess) { printf("chunk %d: %s\n", i, cudaGetErrorString(e)); cudaGetLastError(); break; }
        memset(p, i, G);
        ch.push_back((char *)p);
    }
    printf("pinned %zu GiB in %.1f s\n", ch.size(), now() - t0);
    size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
    printf("device free %.2f GiB of %.2f\n", fr / 1073741824.0, tot / 1073741824.0);
    char *d = nullptr; cudaMalloc(&d, 8 * E);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    const int n = 400;
    // one at a time, waited for (k3cu_moe_expert's copy)
    t0 = now();
    for (int i = 0; i < n; i++) {
        const char *src = ch[i % ch.size()] + (size_t)(i / ch.size() % 50) * E;
        cudaMemcpyAsync(d + (i % 8) * E, src, E, cudaMemcpyHostToDevice, s);
        cudaEventRecord(ev, s); cudaEventSynchronize(ev);
    }
    double t1 = now() - t0;
    printf("pinned, one at a time: %.1f GB/s (%.3f ms a copy)\n", n * E / t1 / 1e9, t1 / n * 1e3);
    t0 = now();
    for (int i = 0; i < n; i++) {
        const char *src = ch[i % ch.size()] + (size_t)(i / ch.size() % 50) * E;
        cudaMemcpyAsync(d + (i % 8) * E, src, E, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);
    t1 = now() - t0;
    printf("pinned, back to back: %.1f GB/s\n", n * E / t1 / 1e9);
    // ordinary memory
    std::vector<char *> pg;
    for (int i = 0; i < 4; i++) { char *p = (char *)malloc(G); memset(p, i, G); pg.push_back(p); }
    t0 = now();
    for (int i = 0; i < n; i++) {
        const char *src = pg[i % 4] + (size_t)(i / 4 % 50) * E;
        cudaMemcpyAsync(d + (i % 8) * E, src, E, cudaMemcpyHostToDevice, s);
        cudaEventRecord(ev, s); cudaEventSynchronize(ev);
    }
    t1 = now() - t0;
    printf("ordinary, one at a time: %.1f GB/s (%.3f ms a copy)\n", n * E / t1 / 1e9, t1 / n * 1e3);
    for (char *p : ch) cudaFreeHost(p);
    return 0;
}
