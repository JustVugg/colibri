"""Real Metal MTLBuffer lifetime and concurrent registry stress, using the PR resolve function.

Requires an actual Apple Metal device. Never creates a large allocation or runs an LLM.
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
#import <Metal/Metal.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
struct Slab { const void *base; size_t len; id<MTLBuffer> buf; };
static std::mutex g_slab_mtx;
static std::vector<Slab> g_slabs;
"""

TRAILER = r"""
int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "SKIP: no Metal device\n"); return 77; }
    void *mem = nullptr;
    if (posix_memalign(&mem, 4096, 4096) != 0) return 1;
    for (size_t i=0;i<4096;i++) ((unsigned char *)mem)[i]=(unsigned char)(i*17u);
    id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:mem length:4096
                                 options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) return 2;
    {
      std::lock_guard<std::mutex> lk(g_slab_mtx);
      g_slabs.push_back({mem,4096,b});
    }
    uint64_t addr=123;
    if (resolve((char *)mem+4096,&addr)!=nil || addr!=123) return 3;
    if (resolve((void *)0x1,&addr)!=nil || addr!=123) return 4;
    id<MTLBuffer> held=resolve((char *)mem+37,&addr);
    if (!held || addr!=(uint64_t)[held gpuAddress]+37) return 5;
    {
      std::lock_guard<std::mutex> lk(g_slab_mtx);
      g_slabs.clear();
    }
    // The retained result must survive removal and support a real Metal command.
    id<MTLBuffer> dst = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[dev newCommandQueue];
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLBlitCommandEncoder> enc=[cb blitCommandEncoder];
    if (!dst || !queue || !cb || !enc) return 6;
    [enc copyFromBuffer:held sourceOffset:0 toBuffer:dst destinationOffset:0 size:4096];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if ([cb status]!=MTLCommandBufferStatusCompleted) return 7;
    if (memcmp([dst contents],mem,4096)!=0) return 8;

    std::atomic<bool> finished{false};
    std::atomic<int> failures{0};
    std::atomic<uint64_t> reads{0};
    std::thread writer([&] {
      @autoreleasepool {
        for (int j=0;j<1500;j++) {
          id<MTLBuffer> cur=[dev newBufferWithBytesNoCopy:mem length:4096
                                      options:MTLResourceStorageModeShared deallocator:nil];
          if (!cur) { failures++; break; }
          { std::lock_guard<std::mutex> lk(g_slab_mtx);
            g_slabs.clear(); g_slabs.push_back({mem,4096,cur}); }
          if ((j&7)==0) std::this_thread::yield();
        }
        { std::lock_guard<std::mutex> lk(g_slab_mtx); g_slabs.clear(); }
        finished=true;
      }
    });
    std::vector<std::thread> readers;
    for(int t=0;t<4;t++) readers.emplace_back([&] {
      @autoreleasepool {
        uint64_t count=0;
        while (!finished || count<1200) {
          uint64_t got=0;
          id<MTLBuffer> found=resolve((char *)mem+43,&got);
          if (found && got!=(uint64_t)[found gpuAddress]+43) failures++;
          count++;
        }
        reads+=count;
      }
    });
    writer.join();
    for(auto &t:readers) t.join();
    if (failures!=0 || reads<4800) return 9;
    // A fictitious base near UINTPTR_MAX proves subtraction-based bounds
    // checking avoids b+len wraparound; no invalid pointer is dereferenced.
    uintptr_t fake=UINTPTR_MAX-32u;
    { std::lock_guard<std::mutex> lk(g_slab_mtx);
      g_slabs.push_back({(void *)fake,128,b}); }
    if (resolve((void *)(fake-1),&addr)!=nil) return 10;
    id<MTLBuffer> edge=resolve((void *)(fake+17u),&addr);
    if (!edge || addr!=(uint64_t)[edge gpuAddress]+17u) return 11;
    { std::lock_guard<std::mutex> lk(g_slab_mtx); g_slabs.clear(); }
    printf("PASS real Metal: blit-after-unregister, 1500 re-registers, %llu queries, high-address bounds\n",
           (unsigned long long)reads.load());
    // Lifetime of externally owned host memory continues through the test.
    // Test process exit reclaims its single page after all GPU work completed.
    return 0;
  }
}
"""

@unittest.skipUnless(sys.platform == "darwin" and shutil.which("clang++"),
                     "requires macOS Objective-C ARC compiler")
class MetalResolveDeviceTest(unittest.TestCase):
    def test_actual_metal_buffer_unregister_and_concurrent_queries(self):
        source=(HERE/"backend_metal.mm").read_text(encoding="utf-8")
        start=source.index("static id<MTLBuffer> resolve(")
        body=source[start:source.index("\n}",start)+2]
        with tempfile.TemporaryDirectory(prefix="coli-device-") as directory:
            root=pathlib.Path(directory)
            code=root/"device.mm"
            exe=root/"device"
            code.write_text(PRELUDE+body+TRAILER)
            subprocess.run(["clang++","-std=c++17","-O1","-fobjc-arc",
                            "-Wall","-Wextra","-Werror",str(code),
                            "-framework","Foundation","-framework","Metal",
                            "-o",str(exe)],check=True,capture_output=True,text=True,timeout=60)
            run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30)
            if run.returncode==77:
                self.skipTest("native Metal device unavailable")
            self.assertEqual(run.returncode,0,run.stdout+"\n"+run.stderr)
            self.assertIn("PASS real Metal:",run.stdout)

if __name__=="__main__":
    unittest.main()
