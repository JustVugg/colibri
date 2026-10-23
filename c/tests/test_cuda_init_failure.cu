/* Two logical devices mapped to one real GPU, with real stream lifetimes.
 * This tests initialization rollback, not physical multi-GPU execution. */
#include "../backend_gpu_compat.h"
#include <cstdio>
static int attempts, live_streams, fail_stream;
static int fail_select=-1, fail_props=-1;
static cudaError_t real_count(int *n) { return cudaGetDeviceCount(n); }
static cudaError_t test_count(int *n) { *n=2;return cudaSuccess; }
static cudaError_t test_select(int device) {
    return device==fail_select ? cudaErrorMemoryAllocation : cudaSetDevice(0);
}
static cudaError_t test_props(cudaDeviceProp *p,int device) {
    return device==fail_props ? cudaErrorMemoryAllocation : cudaGetDeviceProperties(p,0);
}
static cudaError_t test_create(cudaStream_t *s,unsigned flags) {
    if(++attempts==fail_stream) return cudaErrorMemoryAllocation;
    cudaError_t e=cudaStreamCreateWithFlags(s,flags);if(e==cudaSuccess)live_streams++;return e;
}
static cudaError_t test_destroy(cudaStream_t s) {
    cudaError_t e=cudaStreamDestroy(s);if(e==cudaSuccess)live_streams--;return e;
}
#undef cudaGetDeviceCount
#undef cudaSetDevice
#undef cudaGetDeviceProperties
#undef cudaStreamCreateWithFlags
#undef cudaStreamDestroy
#define cudaGetDeviceCount test_count
#define cudaSetDevice test_select
#define cudaGetDeviceProperties test_props
#define cudaStreamCreateWithFlags test_create
#define cudaStreamDestroy test_destroy
#include "../backend_cuda.cu"
int main(void) {
    int count=0;if(real_count(&count)!=cudaSuccess||!count){puts("SKIP: no GPU");return 0;}
    int ids[]={0,1}, duplicate[]={0,0},invalid[]={0,2};
    for(int round=0;round<2;round++){
        attempts=0;
        if(coli_cuda_init(duplicate,2)||attempts||live_streams||coli_cuda_device_count()) return 1;
        if(coli_cuda_init(invalid,2)||attempts||live_streams||coli_cuda_device_count()) return 2;
        for(fail_stream=1;fail_stream<=2;fail_stream++){
            attempts=0;
            if(coli_cuda_init(ids,2)||live_streams||coli_cuda_device_count()) return 3;
        }
        fail_stream=0;
        for(int device=0;device<2;device++){
            g_current_device=-1; /* force the runtime selection fault past the cache */
            fail_select=device;
            if(coli_cuda_init(ids,2)||live_streams||coli_cuda_device_count()) return 11;
            fail_select=-1;
            fail_props=device;
            if(coli_cuda_init(ids,2)||live_streams||coli_cuda_device_count()) return 12;
            fail_props=-1;
        }
        attempts=0;
        if(!coli_cuda_init(ids,2)||live_streams!=2||coli_cuda_device_count()!=2) return 4;
        /* Rejected input must leave an existing backend reachable. */
        if(coli_cuda_init(invalid,2)||live_streams!=2||coli_cuda_device_count()!=2) return 5;
        if(coli_cuda_init(duplicate,2)||live_streams!=2||coli_cuda_device_count()!=2) return 6;
        if(cudaMalloc(&g_ctx[0].x,256)!=cudaSuccess) return 10;
        float *scratch=g_ctx[0].x;
        cudaStream_t original=g_ctx[0].stream;
        int before=attempts;
        if(!coli_cuda_init(ids,2)||attempts!=before||live_streams!=2||g_ctx[0].stream!=original||g_ctx[0].x!=scratch){
            puts("FAIL: repeated init replaced live contexts");return 8;
        }
        int reversed[]={1,0};
        if(coli_cuda_init(ids,1)||coli_cuda_init(reversed,2)||attempts!=before||
           live_streams!=2||g_ctx[0].stream!=original||g_ctx[0].x!=scratch||coli_cuda_device_count()!=2){
            puts("FAIL: device-list change modified active contexts");return 9;
        }
        coli_cuda_shutdown();if(live_streams||coli_cuda_device_count()) return 7;
    }
    puts("CUDA initialization rollback: PASS");return 0;
}
