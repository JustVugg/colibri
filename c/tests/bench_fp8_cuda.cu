/* fmt=8 kernel bench: the original per-(o,s) kernels vs the COLI_CUDA_F8_WARP
 * rework (shared-LUT decode and, where the toolchain compiles it, the
 * cuda_fp8.h cvt decode), on the census expert shapes grouped as 8 experts x
 * S rows, S in {1,4,8,32}. Kernel time only (events around the launches, no
 * H2D/D2H): the question is achieved weight bandwidth against the roofline.
 * GB/s is weight GOODPUT — unique weight bytes per launch over time — so a
 * kernel that re-reads weights per row shows the drop instead of hiding it.
 * pct_peak uses the BENCH_PEAK_GBPS env override only (see device_mem_attrs
 * for why no universal formula exists); -1 when unset. One JSON object on
 * stdout; progress and errors on stderr.
 *
 * Build: make fp8-bench CUDA=1     (or: nvcc -O2 -std=c++17 -arch=native
 *        tests/bench_fp8_cuda.cu -o fp8_bench)
 * The GB10/remote wrapper is tools/run_f8_bench.sh.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

/* No direct <cuda_runtime.h>: backend_gpu_compat.h (via the backend include)
 * supplies the runtime surface for both vendors. */
#include "../backend_cuda.cu"

static float e4m3_ref(uint8_t b){
    int s=b>>7, e=(b>>3)&15, m=b&7;
    if(e==15&&m==7) return NAN;
    float v = e ? ldexpf(1.f+m/8.f, e-7) : ldexpf(m/8.f, -6);
    return s ? -v : v;
}
static uint8_t rnd_e4m3(void){
    uint8_t b=(uint8_t)(rand()&255);
    if((b&0x7F)==0x7F) b&=(uint8_t)~1;
    return b;
}

#define CK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[bench] %s failed: %s\n",#x,cudaGetErrorString(_e)); exit(1);} }while(0)

/* Memory attributes via cudaDeviceGetAttribute, NOT cudaDeviceProp: CUDA 13
 * removed the deprecated memoryClockRate/clockRate members from the struct,
 * while the attribute enums go back to CUDA 11. The attribute's SEMANTICS are
 * device-inconsistent — command clock on HBM parts, but the EFFECTIVE
 * 8533 MT/s rate on GB10, where a 2x DDR factor would double-count — so no
 * universal peak formula exists. The roofline therefore comes ONLY from the
 * BENCH_PEAK_GBPS env override (pct_peak=-1 without it); the raw attributes
 * and the naive 2x-derived candidate are reported so the operator can judge.
 * Failed queries zero the value and clear the sticky error. */
static void device_mem_attrs(int dev,int *khz,int *bus){
    *khz=0;*bus=0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    if(hipDeviceGetAttribute(khz,hipDeviceAttributeMemoryClockRate,dev)!=hipSuccess) *khz=0;
    if(hipDeviceGetAttribute(bus,hipDeviceAttributeMemoryBusWidth,dev)!=hipSuccess) *bus=0;
#else
    if(cudaDeviceGetAttribute(khz,cudaDevAttrMemoryClockRate,dev)!=cudaSuccess) *khz=0;
    if(cudaDeviceGetAttribute(bus,cudaDevAttrGlobalMemoryBusWidth,dev)!=cudaSuccess) *bus=0;
#endif
    (void)cudaGetLastError();
}

static const int E=8, D=2048, I=6144;               /* census: 8 experts, GLM-5.2 dims */

typedef void (*launch_fn)(GroupDesc*,float*,float*,float*,float*,int);

/* Each variant launches ONE kernel (hidden dual or down) so the two matrix
 * orientations are timed separately. */
static void old_hidden(GroupDesc *d,float *g,float *u,float *x,float *y,int S){
    (void)y; grouped_hidden_f8_dual<<<dim3((unsigned)I,(unsigned)S,(unsigned)E),256>>>(g,u,x,d,I,D);
}
static void old_down(GroupDesc *d,float *g,float *u,float *x,float *y,int S){
    (void)u;(void)x; grouped_down_f8<<<dim3((unsigned)D,(unsigned)S,(unsigned)E),256>>>(y,g,d,D,I);
}
static void warp_hidden(GroupDesc *d,float *g,float *u,float *x,float *y,int S){
    (void)y;(void)S; grouped_hidden_f8w_dual<0><<<dim3((unsigned)((I+7)/8),(unsigned)E),256>>>(g,u,x,d,I,D);
}
static void warp_down(GroupDesc *d,float *g,float *u,float *x,float *y,int S){
    (void)u;(void)x;(void)S; grouped_down_f8w<0><<<dim3((unsigned)((D+7)/8),(unsigned)E),256>>>(y,g,d,D,I);
}
#if COLI_F8_HWCVT
static void cvt_hidden(GroupDesc *d,float *g,float *u,float *x,float *y,int S){
    (void)y;(void)S; grouped_hidden_f8w_dual<1><<<dim3((unsigned)((I+7)/8),(unsigned)E),256>>>(g,u,x,d,I,D);
}
static void cvt_down(GroupDesc *d,float *g,float *u,float *x,float *y,int S){
    (void)u;(void)x;(void)S; grouped_down_f8w<1><<<dim3((unsigned)((D+7)/8),(unsigned)E),256>>>(y,g,d,D,I);
}
#endif

int main(void){
    srand(5);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    int khz=0,bus=0; device_mem_attrs(0,&khz,&bus);
    double cand=(khz>0&&bus>0)?2.0*khz*1e3*(bus/8.0)/1e9:0.0;
    double peak=0.0; const char *pe=getenv("BENCH_PEAK_GBPS");
    if(pe&&*pe){ double v=atof(pe); if(v>0) peak=v; }
    fprintf(stderr,"[bench] %s sm_%d%d, mem attrs %d kHz x %d bit "
            "(2x-derived candidate %.1f GB/s), peak %s%.1f GB/s\n",
            prop.name,prop.major,prop.minor,khz,bus,cand,
            peak>0?"(env) ":"(unset) ",peak);

    float lut[256]; for(int i=0;i<256;i++) lut[i]=e4m3_ref((uint8_t)i);
    CK(cudaMemcpyToSymbol(c_e4m3,lut,sizeof(lut)));

    /* device weights + scales, 8 experts */
    const int nbD=D/128,nbI=I/128;
    GroupDesc host[E]; void *dg[E],*du[E],*dd[E],*dgs[E],*dus[E],*dds[E];
    {
        size_t hb=(size_t)I*D, db=(size_t)D*I;
        uint8_t *tmp=(uint8_t*)malloc(hb>db?hb:db);
        float *stmp=(float*)malloc((size_t)nbI*nbD*4);
        for(int c=0;c<E;c++){
            CK(cudaMalloc(&dg[c],hb)); CK(cudaMalloc(&du[c],hb)); CK(cudaMalloc(&dd[c],db));
            CK(cudaMalloc(&dgs[c],(size_t)nbI*nbD*4)); CK(cudaMalloc(&dus[c],(size_t)nbI*nbD*4));
            CK(cudaMalloc(&dds[c],(size_t)nbD*nbI*4));
            for(size_t i=0;i<hb;i++) tmp[i]=rnd_e4m3();
            CK(cudaMemcpy(dg[c],tmp,hb,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(du[c],tmp,hb,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dd[c],tmp,db,cudaMemcpyHostToDevice));
            for(int i=0;i<nbI*nbD;i++) stmp[i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
            CK(cudaMemcpy(dgs[c],stmp,(size_t)nbI*nbD*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dus[c],stmp,(size_t)nbI*nbD*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dds[c],stmp,(size_t)nbD*nbI*4,cudaMemcpyHostToDevice));
        }
        free(tmp); free(stmp);
    }
    const int SMAX=32;
    float *x,*gate,*up,*y; GroupDesc *ddesc;
    CK(cudaMalloc(&x,(size_t)E*SMAX*D*4)); CK(cudaMalloc(&y,(size_t)E*SMAX*D*4));
    CK(cudaMalloc(&gate,(size_t)E*SMAX*I*4)); CK(cudaMalloc(&up,(size_t)E*SMAX*I*4));
    CK(cudaMalloc(&ddesc,sizeof(host)));
    {
        float *hx=(float*)malloc((size_t)E*SMAX*D*4);
        for(size_t i=0;i<(size_t)E*SMAX*D;i++) hx[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        CK(cudaMemcpy(x,hx,(size_t)E*SMAX*D*4,cudaMemcpyHostToDevice)); free(hx);
    }

    struct { const char *path,*kernel; launch_fn fn; double bytes; } cfg[]={
        {"old",     "hidden_dual", old_hidden,  2.0*I*D*E},
        {"old",     "down",        old_down,    1.0*D*I*E},
        {"warp-lut","hidden_dual", warp_hidden, 2.0*I*D*E},
        {"warp-lut","down",        warp_down,   1.0*D*I*E},
#if COLI_F8_HWCVT
        {"warp-cvt","hidden_dual", cvt_hidden,  2.0*I*D*E},
        {"warp-cvt","down",        cvt_down,    1.0*D*I*E},
#endif
    };
    int svals[4]={1,4,8,32};
    printf("{\"device\":\"%s\",\"cc\":\"sm_%d%d\",\"peak_gbps\":%.1f,"
           "\"peak_source\":\"%s\",\"mem_clock_khz\":%d,\"mem_bus_bits\":%d,"
           "\"ddr2x_candidate_gbps\":%.1f,"
           "\"experts\":%d,\"D\":%d,\"I\":%d,\"results\":[",
           prop.name,prop.major,prop.minor,peak,peak>0?"env":"unset",
           khz,bus,cand,E,D,I);
    int first=1;
    for(size_t k=0;k<sizeof(cfg)/sizeof(cfg[0]);k++){
        for(int si=0;si<4;si++){
            int S=svals[si];
            for(int c=0;c<E;c++)
                host[c]={dg[c],du[c],dd[c],(const float*)dgs[c],(const float*)dus[c],
                         (const float*)dds[c],8,8,8,S,c*S,0,0,0};
            CK(cudaMemcpy(ddesc,host,sizeof(host),cudaMemcpyHostToDevice));
            for(int w=0;w<3;w++) cfg[k].fn(ddesc,gate,up,x,y,S);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            const int iters=20;
            CK(cudaEventRecord(e0));
            for(int it=0;it<iters;it++) cfg[k].fn(ddesc,gate,up,x,y,S);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            double gbps=cfg[k].bytes/(ms/1e3)/1e9;
            printf("%s{\"path\":\"%s\",\"kernel\":\"%s\",\"S\":%d,\"ms\":%.4f,"
                   "\"gbps\":%.1f,\"pct_peak\":%.1f}",
                   first?"":",",cfg[k].path,cfg[k].kernel,S,ms,gbps,
                   peak>0?100.0*gbps/peak:-1.0);
            first=0;
            fprintf(stderr,"[bench] %-8s %-11s S=%-2d %.4f ms  %.1f GB/s\n",
                    cfg[k].path,cfg[k].kernel,S,ms,gbps);
        }
    }
    printf("]}\n");
    return 0;
}
