// h2d_load.cu: 17.5 MB host-to-device copies from cudaHostAlloc(Mapped|Portable) memory (the
// expert slots' kind), alone, beside unbuffered reads of the model's shards on both drives
// (the expert readers' traffic), and beside CPU threads sweeping RAM (the CPU's experts).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <windows.h>
#include <cuda_runtime.h>

static std::atomic<int> g_stop{0};
static std::atomic<long long> g_read{0}, g_swept{0};
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void reader(std::vector<std::string> files, int id) {
    const size_t B = 17u << 20;
    void *buf = VirtualAlloc(nullptr, B, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    size_t f = id;
    while (!g_stop) {
        HANDLE h = CreateFileA(files[f % files.size()].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        f += 2;
        if (h == INVALID_HANDLE_VALUE) continue;
        DWORD got = 0;
        while (!g_stop && ReadFile(h, buf, (DWORD)B, &got, nullptr) && got) g_read += got;
        CloseHandle(h);
    }
}
static void sweeper(float *p, size_t n) {
    while (!g_stop) {
        float s = 0;
        for (size_t i = 0; i < n; i += 16) s += p[i];
        if (s == 1.234f) printf("!");
        g_swept += (long long)n * 4;
    }
}
static std::vector<std::string> list(const char *dir) {
    std::vector<std::string> v;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((std::string(dir) + "\\*.safetensors").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return v;
    do v.push_back(std::string(dir) + "\\" + fd.cFileName); while (FindNextFileA(h, &fd));
    FindClose(h);
    return v;
}
static double h2d(void *host, void *dev, size_t b, double secs, cudaStream_t s) {
    double t0 = now(); long long moved = 0;
    while (now() - t0 < secs) {
        for (int i = 0; i < 8; i++) cudaMemcpyAsync(dev, (char *)host + (i % 4) * b, b, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
        moved += 8LL * b;
    }
    return moved / 1e9 / (now() - t0);
}
int main() {
    const size_t b = 17500000;
    void *host, *dev; cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (cudaHostAlloc(&host, 4 * b, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { printf("pin failed\n"); return 1; }
    void *host2; cudaMallocHost(&host2, 4 * b);
    cudaMalloc(&dev, b);
    memset(host, 1, 4 * b); memset(host2, 1, 4 * b);
    printf("alone: mapped|portable %.2f GB/s, cudaMallocHost %.2f GB/s\n", h2d(host, dev, b, 3, s), h2d(host2, dev, b, 3, s));
    auto c = list("C:\\AI\\Kimi-K3"), d = list("D:\\AI\\Models\\Kimi-K3");
    printf("%zu + %zu shards\n", c.size(), d.size());
    std::vector<std::thread> th;
    for (int i = 0; i < 2; i++) th.emplace_back(reader, c, i);
    for (int i = 0; i < 2; i++) th.emplace_back(reader, d, i);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    long long r0 = g_read; double t0 = now();
    double g1 = h2d(host, dev, b, 6, s);
    printf("beside the drives' reads (%.1f GB/s): %.2f GB/s\n", (g_read - r0) / 1e9 / (now() - t0), g1);
    size_t n = (size_t)2 << 28;   /* 2 GiB of floats a sweeper... 8 GB total */
    std::vector<float *> bufs;
    for (int i = 0; i < 12; i++) { float *p = (float *)malloc((n / 4) * 4); memset(p, 0, (n / 4) * 4); bufs.push_back(p); th.emplace_back(sweeper, p, n / 4); }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    r0 = g_read; long long s0 = g_swept; t0 = now();
    double g2 = h2d(host, dev, b, 6, s);
    double dt = now() - t0;
    printf("beside the reads (%.1f GB/s) and 12 threads sweeping RAM (%.1f GB/s): %.2f GB/s\n",
           (g_read - r0) / 1e9 / dt, (g_swept - s0) / 1e9 / dt, g2);
    g_stop = 1;
    for (auto &t : th) t.join();
    return 0;
}
