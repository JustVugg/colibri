#include "../coli_sycl_host_sync.h"
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
using namespace std::chrono_literals;
static int failures;
static void check(bool ok, const char *name) {
    std::printf("%s %s\n", ok ? "ok" : "FAIL", name);
    failures += !ok;
}
int main() {
    coli_sycl_host_sync s;
    unsigned word = 3;
    check(s.wait(&word, 2) && s.wait(&word, 3), "ready word uses >=");
    std::atomic<bool> done{false};
    bool answer = false;
    std::thread waiter([&] { answer = s.wait(&word, 5); done = true; });
    std::this_thread::sleep_for(5ms);
    check(!done, "lower value does not release the waiter");
    check(s.signal(&word, 5), "healthy signal stores the answer");
    waiter.join();
    check(answer && word == 5, "delayed writer releases wait");
    word = 0; done = false;
    std::thread canceled([&] { answer = s.wait(&word, 9); done = true; });
    std::this_thread::sleep_for(5ms);
    s.cancel();
    // A broken cancellation must not leave a test waiting forever.
    for (int i = 0; !done && i < 200; ++i) std::this_thread::sleep_for(1ms);
    bool released = done;
    if (!released) __atomic_store_n(&word, 9u, __ATOMIC_RELEASE);
    canceled.join();
    check(released && !answer && s.lost(), "cancellation releases a blocked wait");
    check(!s.wait(nullptr, 1) && !s.signal(nullptr, 1), "canceled tasks never dereference their flag");
    check(word == 0, "cancellation never fabricates host completion");
    auto owner = std::make_shared<coli_sycl_host_sync>();
    auto captured = owner;
    owner->cancel(); owner.reset();
    check(!captured->wait(nullptr, 1), "captured state outlives its owner");
    std::printf("host synchronization: %d failures\n", failures);
    return failures != 0;
}
