#include "backend_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

typedef struct {
    int device;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    return ctx && cuda_ok(cudaSetDevice(ctx->device), "select device");
}

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        return static_cast<float>(((i & 1) ? (v >> 4) : (v & 15)) - 8);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    quant_matmul<<<hidden_grid,256>>>(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D));
    quant_matmul<<<hidden_grid,256>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D));
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I));
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total};
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    if(!cuda_ok(cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice),
                "expert group descriptors")) return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")) profile=0;
    if(profile) cudaEventRecord(ev[0]);
    if(!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1]);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
    grouped_hidden<<<hg,256>>>(ctx->gate,ctx->x,dev,I,D,0);
    grouped_hidden<<<hg,256>>>(ctx->up,ctx->x,dev,I,D,1);
    silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256>>>(ctx->gate,ctx->up,(size_t)total*I);
    grouped_down<<<og,256>>>(ctx->y,ctx->gate,dev,D,I);
    if(profile) cudaEventRecord(ev[2]);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||
       !cuda_ok(cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost),"expert group output download")) return 0;
    if(profile){
        cudaEventRecord(ev[3]); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}
