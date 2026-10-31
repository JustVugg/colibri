"""Exercise the actual resolve function with an ARC mock, no Metal device.

The mock's gpuAddress call starts a second thread that attempts to remove the
slab. It must obtain the registry mutex while the returned buffer stays alive.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent.parent
PRELUDE = r"""
#import <Foundation/Foundation.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <stdint.h>
@protocol MTLBuffer
- (uint64_t)gpuAddress;
@end
struct Slab { const void *base; size_t len; id<MTLBuffer> buf; };
static std::mutex g_slab_mtx;
static std::vector<Slab> g_slabs;
static std::atomic<bool> unlocked{false};
static std::atomic<int> destroyed{0};
@interface MockBuffer : NSObject <MTLBuffer>
@end
@implementation MockBuffer
- (uint64_t)gpuAddress {
    std::thread worker([] {
        if (g_slab_mtx.try_lock()) {
            unlocked=true;
            g_slabs.clear();
            g_slab_mtx.unlock();
        }
    });
    worker.join();
    return 0xabc000;
}
- (void)dealloc { destroyed++; }
@end
"""
TRAILER = r"""
int main() {
    @autoreleasepool {
        g_slabs.push_back({(void*)0x10000,0x1000,[MockBuffer new]});
        uint64_t address=123;
        if (resolve((void*)0xffff,&address)!=nil || address!=123) return 1;
        if (resolve((void*)0x11000,&address)!=nil || address!=123) return 2;
        id<MTLBuffer> result=resolve((void*)0x10080,&address);
        if (!unlocked || !result || address!=0xabc080 || !g_slabs.empty()) return 3;
        if (destroyed!=0) return 4;
        (void)[result gpuAddress];  // keep result alive through the lifetime check
    }
    if (destroyed!=1) return 5;
    return 0;
}
"""

@unittest.skipUnless(sys.platform == "darwin" and shutil.which("clang++"),
                     "requires macOS Objective-C ARC compiler, not a GPU")
class MetalResolveLifetimeTest(unittest.TestCase):
    def test_driver_call_is_outside_lock_and_buffer_is_retained(self):
        source=(HERE/"backend_metal.mm").read_text(encoding="utf-8")
        start=source.index("static id<MTLBuffer> resolve(")
        body=source[start:source.index("\n}",start)+2]
        with tempfile.TemporaryDirectory(prefix="coli-resolve-") as directory:
            root=pathlib.Path(directory);file=root/"resolve.mm";exe=root/"resolve"
            file.write_text(PRELUDE+body+TRAILER)
            subprocess.run(["clang++","-std=c++17","-fobjc-arc","-Wall","-Wextra",
                            "-Werror",str(file),"-framework","Foundation","-o",str(exe)],
                           check=True,capture_output=True,text=True,timeout=60)
            subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=10)

if __name__ == "__main__": unittest.main()
