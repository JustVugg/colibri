/* Host-only flag synchronization. Never use these atomics from a device kernel.
 * A task captures shared ownership of this state, not the device/chain context.
 * Cancellation and each flag access share a lock: after cancel returns, even a
 * task that has not started cannot touch a flag whose owner is tearing down.
 * No runtime call or sleep is made while holding that lock. */
#ifndef COLI_SYCL_HOST_SYNC_H
#define COLI_SYCL_HOST_SYNC_H
#include <atomic>
#include <mutex>
#include <time.h>
#include <sched.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

class coli_sycl_host_sync {
    std::atomic<bool> canceled_{false};
    std::mutex access_;
public:
    bool lost() const { return canceled_.load(std::memory_order_acquire); }
    void cancel() {
        std::lock_guard<std::mutex> lock(access_);
        canceled_.store(true, std::memory_order_release);
    }
    bool signal(unsigned *flag, unsigned value) {
        std::lock_guard<std::mutex> lock(access_);
        if (lost()) return false;
        __atomic_store_n(flag, value, __ATOMIC_SEQ_CST);
        return true;
    }
    bool wait(const unsigned *flag, unsigned value) {
        unsigned spins = 0;
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(access_);
                if (lost()) return false;  // before touching a possibly retired flag
                if (__atomic_load_n(flag, __ATOMIC_ACQUIRE) >= value) return true;
            }
            if (spins < 1024) {
                ++spins;
#if defined(__x86_64__) || defined(__i386__)
                _mm_pause();
#else
                sched_yield();
#endif
            } else {
                struct timespec ts = {0, 20000};
                nanosleep(&ts, nullptr);
            }
        }
    }
};
#endif
