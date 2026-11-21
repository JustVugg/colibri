/* Tiny host-task integration, with a software-only failed launch. No driver
 * fault injection. A CPU watchdog supplies the missing word after 2 seconds if
 * cancellation fails, allowing the queue to drain and the test to fail safely. */
#include "../../backend_cuda.h"
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "../../cuda_chain.h"
#include "sycl_test_args.h"
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES];
    int n = sycl_test_devices(argc, argv, devices, COLI_CUDA_MAX_DEVICES);
    if (n < 0) return 0;
    if (n != 1 || !coli_cuda_init(devices, n)) return 2;
    const auto *t = coli_cuda_chain_ops();
    if (!t || !t->init(devices[0])) return 2;
    auto &q = coli_sycl_queue(devices[0]);
    CcBuf *flag = t->buf(64, CC_UP), *x = t->buf(64, CC_DEV), *y = t->buf(64, CC_DEV);
    float weights[64] = {}, scales[8] = {1,1,1,1,1,1,1,1};
    ColiCudaTensor *matrix = nullptr;
    if (!flag || !x || !y || !coli_cuda_tensor_upload(&matrix, weights, scales, 0, 8, 8, devices[0])) return 2;
    q.wait_and_throw();  // setup complete before the deliberately unanswered host wait
    unsigned *word = static_cast<unsigned *>(t->ptr(flag));
    std::atomic<bool> done{false}, rescued{false}, entered{false};
    std::thread watchdog([&] {
        for (int i = 0; !done && i < 2000; ++i) std::this_thread::sleep_for(1ms);
        if (!done) { rescued = true; __atomic_store_n(word, 7u, __ATOMIC_RELEASE); }
    });
    q.submit([&](sycl::handler &h) { h.host_task([&] { entered = true; }); });
    int ok = t->begin() && t->wait_host(flag, 0, 7) && t->submit(0);
    // Give the host task time to block. Even if runtime scheduling is delayed,
    // the canceled-state-before-dereference branch still has to work.
    std::this_thread::sleep_for(20ms);
    const bool direct = std::getenv("COLI_TEST_CANCEL_STATE") != nullptr;
    int launch = 0;
    if (direct) coli_sycl_host_state(devices[0])->cancel();
    else {
        setenv("COLI_GPU_FAIL_AFTER", "0", 1);
        launch = t->matmul(matrix, x, 0, y, 0, 1);
        unsetenv("COLI_GPU_FAIL_AFTER");
    }
    ok = ok && !launch && t->lost() && coli_sycl_host_state(devices[0])->lost();
    q.wait_and_throw();
    done = true; watchdog.join();
    ok = ok && entered && !rescued && __atomic_load_n(word, __ATOMIC_ACQUIRE) == 0;
    ok = ok && !t->ready() && !t->begin() && !coli_sycl_host_state(devices[0])->wait(nullptr, 1);
    std::printf("%s host wait exits after %s; watchdog=%d, word=%u\n",
                ok ? "OK" : "FAIL", direct ? "shared-state cancellation" : "software launch failure", int(rescued), *word);
    coli_cuda_tensor_free(matrix);
    t->free(flag); t->free(x); t->free(y);
    t->shutdown(); coli_cuda_shutdown();
    return ok ? 0 : 1;
}
