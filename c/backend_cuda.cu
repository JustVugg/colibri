#include "backend_cuda.h"

#include "backend_gpu_compat.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <sys/stat.h>
#endif

struct RaggedKVEntry {
    const void *key;
    const float *host_l,*host_r;
    float *latent,*rope;
    int length,capacity,K,R;
};

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int gs;                    /* quant group size; 0 = per-row scales (#334) */
    int ng;                    /* number of scale groups per row = ceil(I/gs) for fmt=4 */
    size_t scale_count;        /* floats in `scales`: O per-row, O*ng grouped */
    int tracked;
    RaggedKVEntry ragged[512];
    int ragged_count;
};

typedef struct {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y,*host_kv; size_t host_x_cap,host_y_cap,host_kv_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *alsc,*arsc; size_t alsc_cap,arsc_cap;   /* KV8: scale per-riga di latent/rope */
    float *asel; size_t asel_cap;                  /* KV8+DSA: lista di selezione (int, riusa reserve) */
    float *apart; size_t apart_cap;                /* KV8 split-T: parziali (cl[K],m,z) per (testa,split) */
    float *pipe_buf[27]; size_t pipe_cap[27];   /* scratch persistenti del resident pipeline */
    uint8_t *kvstage; size_t kvstage_cap,kvstage_off;  /* ring pinned per gli upload async dell'ombra KV */
    cudaStream_t stream;
    cudaEvent_t ev_done; int ev_done_ok;        /* resident-group issue completion (#431 PR-C0) */
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
    int group_pending; size_t group_pending_bytes;   /* async expert-group in flight (Inc.4) */
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
    int ggs,ugs,dgs;      /* per-tensor quant group size; 0 = per-row scales (#334 fmt=4) */
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    (void)cudaGetLastError();   /* consume the sticky error: a failed call must
                                   not poison the next launch's error check */
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

/* fmt=6 (E8/IQ3) geometry, mirroring quant.h. A super-block packs 256 weights
 * into 98 bytes: 64 codebook indices, 8 words of (4x7 signs + 4-bit sub-scale),
 * and one fp16 super-scale. Scales live INSIDE the block, so fmt=6 tensors carry
 * no separate scale array (#452). */
#define COLI_E8_QK      256
#define COLI_E8_SUB      32
#define COLI_E8_BBYTES   98

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2 || fmt == 4) return (size_t)(I + 1) / 2;   /* fmt=4: same packed int4 */
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)(I + 1) / 2;   /* grouped int4: nibbles like fmt 2 */
    if (fmt == 6) return (size_t)(((int64_t)I + COLI_E8_QK - 1) / COLI_E8_QK) * COLI_E8_BBYTES;
    return 0;
}

/* The E8 codebook, uploaded once per device from quant.h's e8_grid so the table
 * has a single source of truth and cannot drift from the CPU decoder. */
__constant__ uint8_t c_e8_grid[256][4];

/* A super-block is 98 bytes, so nothing inside it is guaranteed 4- or 2-byte
 * aligned: assemble the words byte-wise instead of dereferencing. */
__device__ __forceinline__ uint32_t e8_ld_u32(const uint8_t *p){
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
/* Mirrors e8_fp16_to_f32 rather than calling __half2float, so the two decoders
 * cannot disagree on subnormals. */
__device__ __forceinline__ float e8_fp16(const uint8_t *p){
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1F, man=h&0x3FF, bits;
    if (!exp)         bits = man ? (sign|((127u-15u+1u-1u)<<23)|(man<<13)) : sign;
    else if (exp==31) bits = sign|0x7F800000u|(man<<13);
    else              bits = sign|((exp+112u)<<23)|(man<<13);
    float f; memcpy(&f,&bits,4); return f;
}
/* Expand one 32-weight sub-block; mirrors e8_expand_sub in quant.h. */
__device__ __forceinline__ void e8_expand_sub_dev(const uint8_t *blk, int ib, float d, float *out){
    uint32_t word = e8_ld_u32(blk + COLI_E8_QK/4 + ib*4);
    float db = d * (0.5f + (float)((word>>28)&0xF)) * 0.5f;
    const uint8_t *idx = blk + ib*8;
    for (int l=0;l<4;l++){
        uint32_t seven=(word>>(7*l))&0x7F;
        const uint8_t *g0=c_e8_grid[idx[l*2+0]], *g1=c_e8_grid[idx[l*2+1]];
        int par=0;
        for (int j=0;j<8;j++){
            int neg = j<7 ? (int)((seven>>j)&1) : 0;
            if (j<7) par^=neg; else neg=par;        /* odd parity closes the block */
            float mag = (j<4 ? (float)g0[j] : (float)g1[j-4]) * 0.5f;
            out[l*8+j] = neg ? -mag*db : mag*db;
        }
    }
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2 || fmt == 4) {                               /* fmt=4: same nibble layout */
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

/* Scale for output `row`, input element `k`. fmt=4 (grouped int4) stores ng
 * scales per row at scales[row*ng + k/gs]; every other quantized format has
 * one scale per row at scales[row]. Mirrors quant_matmul's fmt==4 branch so the
 * attention absorb kernels apply per-group scales instead of the per-row
 * (fmt=2) semantic that crashed #298's g64 kv_b. */
__device__ static float absorb_scale(const float *wscale, int fmt, int gs, int ng, int row, int k) {
    if (!fmt) return 1.f;
    if (fmt != 4) return wscale[row];
    int g = k / gs; if (g >= ng) g = ng - 1;   /* tail of the last (partial) group */
    return wscale[(size_t)row * ng + g];
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb, int gs, int ng) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    if (fmt == 6) {
        /* E8/IQ3: decode is per 32-weight sub-block, so threads stride over
         * sub-blocks rather than elements -- expanding once per 32 weights
         * instead of redoing the word/parity work for every element. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        int nsub = (I + COLI_E8_SUB - 1) / COLI_E8_SUB;
        for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
            const uint8_t *blk = wrow + (size_t)(sb / (COLI_E8_QK/COLI_E8_SUB)) * COLI_E8_BBYTES;
            float w[COLI_E8_SUB];
            e8_expand_sub_dev(blk, sb % (COLI_E8_QK/COLI_E8_SUB), e8_fp16(blk+96), w);
            int off = sb*COLI_E8_SUB, n = I-off < COLI_E8_SUB ? I-off : COLI_E8_SUB;
            for (int k=0;k<n;k++) sum += xs[off+k]*w[k];
        }
    } else if (fmt == 4) {
        /* Grouped int4: one f32 scale per gs elements along I (ng groups per row).
         * Scale layout: scales[o*ng + g]. Each thread strides through I, applying
         * the appropriate group scale as it crosses group boundaries. This matches
         * the CPU matmul_i4_grouped accumulation exactly. */
        const float *scl = scales + (size_t)o * ng;
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            int g = i / gs;
            if (g >= ng) g = ng - 1;  /* tail elements in the last (partial) group */
            sum += xs[i] * weight_at(weights, fmt, row, i) * scl[g];
        }
    } else {
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * weight_at(weights, fmt, row, i);
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = (fmt && fmt != 4 && fmt != 6) ? partial[0] * scales[o] : partial[0];
}

/* fmt=6 activation rotation, y = Q^T x for Q = D*H/sqrt(n) (#452). One block per
 * row; the power-of-two block is staged in shared memory, capping n at 4096
 * floats -- which covers every block GLM produces (6144 -> 2048+4096, 1536 ->
 * 512+1024). The sign stream is regenerated in-kernel from the same xorshift64*
 * that quant.h's e8_signs uses, so no rotation data is stored or uploaded.
 *
 * Placement note: all routed experts of a layer share one gate/up input, so that
 * rotation belongs to the CALLER (once per layer). This kernel exists for the
 * down projection, whose input is the per-expert silu(gate)*up product and so
 * cannot be shared -- mirroring colibri.c's split at moe(). */
__global__ static void e8_rot_rows_kernel(float *rows, int dim, int off, int n){
    extern __shared__ float sh[];
    __shared__ uint8_t sbits[4096/8];
    if (!threadIdx.x) {
        uint64_t s = 417u + (uint64_t)n;
        for (int i=0;i<(n+7)/8;i++){
            s^=s>>12; s^=s<<25; s^=s>>27;
            sbits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
        }
    }
    __syncthreads();
    float *row = rows + (size_t)blockIdx.x*dim + off;
    for (int i=threadIdx.x;i<n;i+=blockDim.x){
        float v=row[i];
        sh[i] = (sbits[i>>3]>>(i&7)&1) ? -v : v;
    }
    __syncthreads();
    for (int len=1;len<n;len<<=1){
        for (int j=threadIdx.x;j<n/2;j+=blockDim.x){
            int i = (j/len)*(len<<1) + (j%len);
            float u=sh[i], v=sh[i+len];
            sh[i]=u+v; sh[i+len]=u-v;
        }
        __syncthreads();
    }
    float sc=rsqrtf((float)n);
    for (int i=threadIdx.x;i<n;i+=blockDim.x) row[i]=sh[i]*sc;
}

/* Rotate nr rows in place, tiling non-power-of-two dims block-diagonally exactly
 * as e8_rot_rows does. Returns 0 if a block exceeds the shared-memory cap. */
static int e8_rot_rows_dev(float *rows, int nr, int dim, cudaStream_t stream){
    int off = 0;
    while (off < dim) {
        int rem = dim-off, b = rem & (-rem);
        while (b > 4096) b >>= 1;
        e8_rot_rows_kernel<<<(unsigned)nr, 256, (size_t)b*sizeof(float), stream>>>(rows, dim, off, b);
        if (cudaGetLastError() != cudaSuccess) return 0;
        off += b;
    }
    return 1;
}

/* quant_matmul mette le righe S su grid-Y (max 65535): un batch piu' lungo
 * (PREFILL_CHUNK=0 su un prompt >65k) fallirebbe il lancio DOPO che l'attention
 * e' gia' stata pagata, ricadendo in silenzio su CPU — lo stesso tetto che il
 * kernel streaming ha gia' eliminato mettendo S su grid-X. Lancia a fette. */
static void quant_matmul_rows(float *y, const float *x, const void *weights,
                              const float *scales, int fmt, int S, int I, int O,
                              size_t rb, int gs, int ng, cudaStream_t stream) {
    for (int s0 = 0; s0 < S; s0 += 65535) {
        int sn = S - s0 < 65535 ? S - s0 : 65535;
        quant_matmul<<<dim3((unsigned)O,(unsigned)sn),256,0,stream>>>(
            y + (size_t)s0 * O, x + (size_t)s0 * I, weights, scales, fmt, sn, I, O, rb, gs, ng);
    }
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* Four warps share one A tile and compute 16x64 outputs.  This matters for
 * prefill: the first prototype reloaded/converter A once per 16 output cols. */
__global__ static void w4a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both 16x64 projections
 * while sharing the FP32->FP16 conversion of A. */
__global__ static void w4a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
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

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* Fused epilogue: silu(gate)*up lands here instead of a third kernel —
     * the exact silu_mul expression on the exact same inputs, so bit-identical,
     * and the up[] round-trip through global memory disappears. up stays a
     * param so the launch sites keep their signature. */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0]*d.gs[o],u=upv[0]*d.us[o];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

/* fmt=4 grouped-int4 variants (#334): identical structure to the w4 kernels,
 * but the scale varies along the input dimension — sc[o*ng + i/gs], applied
 * per element inside the accumulation (gs is even, so a packed byte never
 * straddles a group). gs<=0 degrades to per-row (ng=1), so mixed fmt2/fmt4
 * groups run correctly through this one kernel family. */
__global__ static void grouped_hidden_g4_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    int ggs=d.ggs>0?d.ggs:D, ugs=d.ugs>0?d.ugs:D;
    const float *gsc=d.gs+(size_t)o*(size_t)((D+ggs-1)/ggs);
    const float *usc=d.us+(size_t)o*(size_t)((D+ugs-1)/ugs);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;float gv=gsc[i/ggs],uv=usc[i/ugs];
        ga+=xs[i]*g0*gv;ua+=xs[i]*u0*uv;
        if(i+1<D){ga+=xs[i+1]*g1*gv;ua+=xs[i+1]*u1*uv;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same epilogue fusion as the w4 dual above (per-group scales already
     * applied inside the accumulation, so silu runs on the raw sums) */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_g4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    int dgs=d.dgs>0?d.dgs:I;
    const float *dsc=d.ds+(size_t)o*(size_t)((I+dgs-1)/dgs);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;float sv=dsc[i/dgs];
        sum+=xs[i]*a*sv;if(i+1<I)sum+=xs[i+1]*z*sv;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}
/* KV8: decodifica un byte e4m3 (cvt hardware da sm_89; PTX/software prima). La scala
 * per-riga viaggia in un array f32 separato e si applica una volta per score/peso. */
__device__ static inline float fp8_e4m3(uint8_t b){
#if COLI_GPU_HAS_FP8
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3)));
#else
    /* Portable e4m3 decode (HIP): identical value grid to the CUDA cvt and to the
     * host LUT in kv_fp8.h — normals (1+M/8)*2^(E-7), denormals M*2^-9, and the
     * NaN code S.1111.111 read back as 0 (the encoder never emits it). */
    int E=(b>>3)&0xF, M=b&7;
    float v = E ? ldexpf(1.f+(float)M*0.125f, E-7) : ldexpf((float)M, -9);
    if(E==15 && M==7) v=0.f;
    return (b&0x80) ? -v : v;
#endif
}

/* Dot di una riga fp8 contro un vettore f32, a carichi VETTORIALI (uint4 = 16
 * byte per istruzione di load). Il profilo ncu del kernel streaming era load-
 * bound (L1 97.5%, SM 19%, DRAM ~0): un load per BYTE. Stessa matematica,
 * stesso ordine di somma per elemento — cambia solo la larghezza del load. */
__device__ static inline float fp8_dot4(const float *qa,uint32_t w){
    return qa[0]*fp8_e4m3((uint8_t)(w&0xffu))
         + qa[1]*fp8_e4m3((uint8_t)((w>>8)&0xffu))
         + qa[2]*fp8_e4m3((uint8_t)((w>>16)&0xffu))
         + qa[3]*fp8_e4m3((uint8_t)((w>>24)&0xffu));
}
__device__ static inline float fp8_row_dot(const float *qa,const uint8_t *p,int n){
    float a=0;
    if(!(n&15) && !((uintptr_t)p&15)){
        const uint4 *v=(const uint4*)p;
        int nq=n>>4;
        for(int i=0;i<nq;i++){
            uint4 w=v[i]; const float *qk=qa+(i<<4);
            a+=fp8_dot4(qk,w.x)+fp8_dot4(qk+4,w.y)+fp8_dot4(qk+8,w.z)+fp8_dot4(qk+12,w.w);
        }
        return a;
    }
    for(int k=0;k<n;k++)a+=qa[k]*fp8_e4m3(p[k]);
    return a;
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale,
                                                int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale,
        int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

/* ---- KV8: gemelli fp8 dei due kernel di assorbimento. latent/rope arrivano come
 * byte e4m3 + scala f32 per riga. Stessa matematica dei gemelli f32: la scala esce
 * dal dot (score = Lsc·Σ q·v + Rsc·Σ qr·v) e per il contesto si fonde nel peso
 * softmax, cosi' i loop interni restano una FMA per byte. */
__global__ static void attention_absorb_kernel8(float *ctx,const float *q,const uint8_t *latent,
        const float *lsc,const uint8_t *rope,const float *rsc,const void *weights,
        const float *wscale,int fmt,int H,int Q,int R,int V,int K,int T,float scale,int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){const uint8_t *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        float a=fp8_row_dot(qa,lt,K);
        float ar=fp8_row_dot(q+(size_t)h*(Q+R)+Q,rt,R);
        scores[t]=(a*lsc[t]+ar*rsc[t])*scale;}
    __syncthreads();
    /* softmax a riduzione ad albero (era seriale su thread 0: 3 passate
     * dipendenti su T<=4096 elementi mentre 255 thread aspettavano) */
    float local=-3.402823466e+38F;for(int t=tid;t<T;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<T;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];
    for(int t=tid;t<T;t+=blockDim.x)scores[t]*=inv*lsc[t];   /* normalizza + scala fusa nel peso */
    __syncthreads();
    /* accumulo del contesto latente: quad di k per thread — load uchar4 COALESCENTI
     * al posto dei load colonna per-byte (stride K) del loop originale. */
    if(!(K&3)&&!((uintptr_t)latent&3)){
        for(int kq=tid;kq<(K>>2);kq+=blockDim.x){int k0=kq<<2;float a0=0,a1=0,a2=0,a3=0;
            for(int t=0;t<T;t++){uint32_t w=*(const uint32_t*)(latent+(size_t)t*K+k0);float sw=scores[t];
                a0+=sw*fp8_e4m3((uint8_t)(w&0xffu));a1+=sw*fp8_e4m3((uint8_t)((w>>8)&0xffu));
                a2+=sw*fp8_e4m3((uint8_t)((w>>16)&0xffu));a3+=sw*fp8_e4m3((uint8_t)((w>>24)&0xffu));}
            cl[k0]=a0;cl[k0+1]=a1;cl[k0+2]=a2;cl[k0+3]=a3;}
    } else
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*fp8_e4m3(latent[(size_t)t*K+k]);cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

__global__ static void attention_absorb_batch_kernel8(float *ctx,const float *q,
        const uint8_t *latent,const float *lsc,const uint8_t *rope,const float *rsc,
        const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale,int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){const uint8_t *lt=latent+(size_t)t*K;
        const uint8_t *rt=rope+(size_t)t*R;float a=fp8_row_dot(qa,lt,K);
        float ar=fp8_row_dot(qs+Q,rt,R);
        scores[t]=(a*lsc[t]+ar*rsc[t])*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];
    for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv*lsc[t];  /* normalizza + scala fusa nel peso */
    __syncthreads();
    /* accumulo del contesto latente: quad di k per thread — load uchar4 COALESCENTI
     * al posto dei load colonna per-byte (stride K) del loop originale. */
    if(!(K&3)&&!((uintptr_t)latent&3)){
        for(int kq=tid;kq<(K>>2);kq+=blockDim.x){int k0=kq<<2;float a0=0,a1=0,a2=0,a3=0;
            for(int t=0;t<nt;t++){uint32_t w=*(const uint32_t*)(latent+(size_t)t*K+k0);float sw=scores[t];
                a0+=sw*fp8_e4m3((uint8_t)(w&0xffu));a1+=sw*fp8_e4m3((uint8_t)((w>>8)&0xffu));
                a2+=sw*fp8_e4m3((uint8_t)((w>>16)&0xffu));a3+=sw*fp8_e4m3((uint8_t)((w>>24)&0xffu));}
            cl[k0]=a0;cl[k0+1]=a1;cl[k0+2]=a2;cl[k0+3]=a3;}
    } else
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*fp8_e4m3(latent[(size_t)t*K+k]);cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}


/* Independent device-resident KV sequence per row. lengths selects the valid
 * prefix; latent/rope point at paged caches updated by the host wrapper. */
__global__ static void attention_absorb_ragged_kernel(float *ctx,const float *q,
        const float *const *latent,const float *const *rope,const int *lengths,
        const void *weights,const float *wscale,int fmt,int S,int H,int Q,int R,
        int V,int K,int T,float scale){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=lengths[s],rbase=h*(Q+V);
    if(s>=S||nt<1||nt>T)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    const float *ls=latent[s],*rs=rope[s];
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          (fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=ls+(size_t)t*K;
        const float *rt=rs+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)a+=scores[t]*ls[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[((size_t)s*H+h)*V+v]=a*(fmt?wscale[row]:1.f);}
}

__global__ static void ragged_kv_append(float *const *latent,float *const *rope,
        const float *packed,const int *old_len,const int *add,const int *offset,int K,int R){
    int s=blockIdx.x,n=add[s],base=offset[s];
    for(int i=threadIdx.x;i<n*(K+R);i+=blockDim.x){
        if(i<n*K)latent[s][(size_t)old_len[s]*K+i]=packed[base+i];
        else rope[s][(size_t)old_len[s]*R+i-n*K]=packed[base+i];
    }
}

/* ---- KV8 long-T: absorb SENZA il tetto di T. Softmax online a tessere (stile
 * flash-attention): niente array scores[T] in shared — per tessera si tiene solo
 * max/somma correnti e si ri-scala l'accumulatore latente cl[K]. Matematicamente
 * e' la stessa softmax (cambia solo l'ordine delle somme fp). smem: ~7 KB fissi. */
#define COLI_ATTN_TILE 256
__global__ static void attention_absorb_stream_kernel8(float *ctx,const float *q,
        const uint8_t *latent,const float *lsc,const uint8_t *rope,const float *rsc,
        const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale,int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*tsc=cl+K,*tls=tsc+COLI_ATTN_TILE,*red=tls+COLI_ATTN_TILE;
    __shared__ float run_m,run_z;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;cl[k]=0;}
    if(!tid){run_m=-3.402823466e+38F;run_z=0;}
    __syncthreads();
    for(int t0=0;t0<nt;t0+=COLI_ATTN_TILE){
        int tn=nt-t0<COLI_ATTN_TILE?nt-t0:COLI_ATTN_TILE;
        for(int j=tid;j<tn;j+=blockDim.x){int t=t0+j;
            const uint8_t *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
            float a=fp8_row_dot(qa,lt,K);
            float ar=fp8_row_dot(qs+Q,rt,R);
            tls[j]=lsc[t];
            tsc[j]=(a*tls[j]+ar*rsc[t])*scale;}
        __syncthreads();
        float local=-3.402823466e+38F;
        for(int j=tid;j<tn;j+=blockDim.x)local=fmaxf(local,tsc[j]);
        red[tid]=local;__syncthreads();
        for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
        float old_m=run_m,new_m=fmaxf(old_m,red[0]);
        float fac=expf(old_m-new_m);                     /* old_m=-inf -> fac=0: cl e' gia' 0 */
        __syncthreads();
        local=0;
        for(int j=tid;j<tn;j+=blockDim.x){float e=expf(tsc[j]-new_m);tsc[j]=e;local+=e;}
        red[tid]=local;__syncthreads();
        for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
        if(!tid){run_z=run_z*fac+red[0];run_m=new_m;}
        /* peso*scala precalcolato una volta per riga (era rifatto per ogni k) */
        for(int j=tid;j<tn;j+=blockDim.x)tsc[j]*=tls[j];
        __syncthreads();
        /* quad di k per thread: load uchar4 coalescenti al posto dei load
         * colonna per-byte (stride K) */
        if(!(K&3)&&!((uintptr_t)latent&3)){
            for(int kq=tid;kq<(K>>2);kq+=blockDim.x){int k0=kq<<2;
                float a0=cl[k0]*fac,a1=cl[k0+1]*fac,a2=cl[k0+2]*fac,a3=cl[k0+3]*fac;
                for(int j=0;j<tn;j++){uint32_t w=*(const uint32_t*)(latent+(size_t)(t0+j)*K+k0);float sw=tsc[j];
                    a0+=sw*fp8_e4m3((uint8_t)(w&0xffu));a1+=sw*fp8_e4m3((uint8_t)((w>>8)&0xffu));
                    a2+=sw*fp8_e4m3((uint8_t)((w>>16)&0xffu));a3+=sw*fp8_e4m3((uint8_t)((w>>24)&0xffu));}
                cl[k0]=a0;cl[k0+1]=a1;cl[k0+2]=a2;cl[k0+3]=a3;}
        } else
        for(int k=tid;k<K;k+=blockDim.x){
            float a=cl[k]*fac;
            for(int j=0;j<tn;j++)a+=tsc[j]*fp8_e4m3(latent[(size_t)(t0+j)*K+k]);
            cl[k]=a;}
        __syncthreads();
    }
    float inv=1.f/run_z;
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*inv*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

/* Decode S=1 a T lungo: 64 blocchi (uno per testa) non riempiono la GPU e il
 * kernel diventa seriale (misurato ~103 ms/layer a 262k). Split-T stile
 * flash-decoding: dim3(H,P) blocchi, ognuno produce un parziale online-softmax
 * (cl[K], m, z) sul suo intervallo di righe; il merge li ricombina in
 * log-sum-exp e proietta il contesto. Stessa softmax, ordine fp diverso. */
__global__ static void attention_absorb_split_kernel8(float *part,const float *q,
        const uint8_t *latent,const float *lsc,const uint8_t *rope,const float *rsc,
        const void *weights,const float *wscale,
        int fmt,int H,int Q,int R,int V,int K,int T,int chunk,float scale,int gs,int ng){
    int h=blockIdx.x,p=blockIdx.y,tid=threadIdx.x,rbase=h*(Q+V);
    int t0=p*chunk, t1=(t0+chunk<T)?t0+chunk:T;
    extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*tsc=cl+K,*tls=tsc+COLI_ATTN_TILE,*red=tls+COLI_ATTN_TILE;
    __shared__ float run_m,run_z;
    float *dst=part+((size_t)h*gridDim.y+p)*(K+2);
    if(t0>=T){ for(int k=tid;k<K;k+=blockDim.x)dst[k]=0;
        if(!tid){dst[K]=-3.402823466e+38F;dst[K+1]=0;} return; }
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;cl[k]=0;}
    if(!tid){run_m=-3.402823466e+38F;run_z=0;}
    __syncthreads();
    for(int b0=t0;b0<t1;b0+=COLI_ATTN_TILE){
        int tn=t1-b0<COLI_ATTN_TILE?t1-b0:COLI_ATTN_TILE;
        for(int j=tid;j<tn;j+=blockDim.x){int t=b0+j;
            const uint8_t *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
            float a=fp8_row_dot(qa,lt,K);
            float ar=fp8_row_dot(q+(size_t)h*(Q+R)+Q,rt,R);
            tls[j]=lsc[t];
            tsc[j]=(a*tls[j]+ar*rsc[t])*scale;}
        __syncthreads();
        float local=-3.402823466e+38F;
        for(int j=tid;j<tn;j+=blockDim.x)local=fmaxf(local,tsc[j]);
        red[tid]=local;__syncthreads();
        for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
        float old_m=run_m,new_m=fmaxf(old_m,red[0]);
        float fac=expf(old_m-new_m);
        __syncthreads();
        local=0;
        for(int j=tid;j<tn;j+=blockDim.x){float e=expf(tsc[j]-new_m);tsc[j]=e;local+=e;}
        red[tid]=local;__syncthreads();
        for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
        if(!tid){run_z=run_z*fac+red[0];run_m=new_m;}
        /* peso*scala precalcolato una volta per riga (era rifatto per ogni k) */
        for(int j=tid;j<tn;j+=blockDim.x)tsc[j]*=tls[j];
        __syncthreads();
        /* quad di k per thread: load uchar4 coalescenti al posto dei load
         * colonna per-byte (stride K) */
        if(!(K&3)&&!((uintptr_t)latent&3)){
            for(int kq=tid;kq<(K>>2);kq+=blockDim.x){int k0=kq<<2;
                float a0=cl[k0]*fac,a1=cl[k0+1]*fac,a2=cl[k0+2]*fac,a3=cl[k0+3]*fac;
                for(int j=0;j<tn;j++){uint32_t w=*(const uint32_t*)(latent+(size_t)(b0+j)*K+k0);float sw=tsc[j];
                    a0+=sw*fp8_e4m3((uint8_t)(w&0xffu));a1+=sw*fp8_e4m3((uint8_t)((w>>8)&0xffu));
                    a2+=sw*fp8_e4m3((uint8_t)((w>>16)&0xffu));a3+=sw*fp8_e4m3((uint8_t)((w>>24)&0xffu));}
                cl[k0]=a0;cl[k0+1]=a1;cl[k0+2]=a2;cl[k0+3]=a3;}
        } else
        for(int k=tid;k<K;k+=blockDim.x){
            float a=cl[k]*fac;
            for(int j=0;j<tn;j++)a+=tsc[j]*fp8_e4m3(latent[(size_t)(b0+j)*K+k]);
            cl[k]=a;}
        __syncthreads();
    }
    for(int k=tid;k<K;k+=blockDim.x)dst[k]=cl[k];
    if(!tid){dst[K]=run_m;dst[K+1]=run_z;}
}

__global__ static void attention_absorb_merge_kernel8(float *ctx,const float *part,
        const void *weights,const float *wscale,int fmt,int H,int Q,int R,int V,int K,int P,int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);
    extern __shared__ float sm[]; float *cl=sm,*red=cl+K;
    float lm=-3.402823466e+38F;
    for(int p=tid;p<P;p+=blockDim.x)lm=fmaxf(lm,part[((size_t)h*P+p)*(K+2)+K]);
    red[tid]=lm;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float M=red[0];__syncthreads();
    float lz=0;
    for(int p=tid;p<P;p+=blockDim.x){const float *pp=part+((size_t)h*P+p)*(K+2);
        lz+=pp[K+1]*expf(pp[K]-M);}
    red[tid]=lz;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];__syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;
        for(int p=0;p<P;p++){const float *pp=part+((size_t)h*P+p)*(K+2);
            a+=pp[k]*expf(pp[K]-M);}
        cl[k]=a*inv;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[(size_t)h*V+v]=a;}
}

/* KV8 + DSA: assorbimento sulle SOLE righe selezionate (lista di posizioni
 * assolute, nt <= index_topk <= 4096: scores stanno in shared come nel gemello
 * denso). E' il percorso che il decode di produzione prende oltre index_topk. */
__global__ static void attention_absorb_sel_kernel8(float *ctx,const float *q,
        const uint8_t *latent,const float *lsc,const uint8_t *rope,const float *rsc,
        const int *sel,const void *weights,const float *wscale,
        int fmt,int H,int Q,int R,int V,int K,int NS,float scale,int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+NS;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int jj=tid;jj<NS;jj+=blockDim.x){int t=sel[jj];
        const uint8_t *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        float a=fp8_row_dot(qa,lt,K);
        float ar=fp8_row_dot(q+(size_t)h*(Q+R)+Q,rt,R);
        scores[jj]=(a*lsc[t]+ar*rsc[t])*scale;}
    __syncthreads();
    /* softmax a riduzione ad albero (era seriale su thread 0 con 255 thread fermi) */
    float local=-3.402823466e+38F;for(int t=tid;t<NS;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<NS;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];
    for(int jj=tid;jj<NS;jj+=blockDim.x)scores[jj]*=inv*lsc[sel[jj]];   /* normalizza + scala fusa */
    __syncthreads();
    if(!(K&3)&&!((uintptr_t)latent&3)){
        for(int kq=tid;kq<(K>>2);kq+=blockDim.x){int k0=kq<<2;float a0=0,a1=0,a2=0,a3=0;
            for(int jj=0;jj<NS;jj++){uint32_t w=*(const uint32_t*)(latent+(size_t)sel[jj]*K+k0);float sw=scores[jj];
                a0+=sw*fp8_e4m3((uint8_t)(w&0xffu));a1+=sw*fp8_e4m3((uint8_t)((w>>8)&0xffu));
                a2+=sw*fp8_e4m3((uint8_t)((w>>16)&0xffu));a3+=sw*fp8_e4m3((uint8_t)((w>>24)&0xffu));}
            cl[k0]=a0;cl[k0+1]=a1;cl[k0+2]=a2;cl[k0+3]=a3;}
    } else
    for(int k=tid;k<K;k+=blockDim.x){float a=0;
        for(int jj=0;jj<NS;jj++)a+=scores[jj]*fp8_e4m3(latent[(size_t)sel[jj]*K+k]);cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

/* Split della SELEZIONE (stile flash-decoding, come il gemello denso split-T):
 * 64 blocchi (uno per testa) non riempiono una GPU da 170 SM — misurato 16.7%%
 * di occupancy e 6.6%% di SM sul percorso DSA di produzione. dim3(H,P): ogni
 * blocco produce un parziale online-softmax (cl[K],m,z) sulla SUA fetta della
 * lista; il merge esistente li ricombina. Stessa softmax, ordine fp diverso. */
__global__ static void attention_absorb_sel_split_kernel8(float *part,const float *q,
        const uint8_t *latent,const float *lsc,const uint8_t *rope,const float *rsc,
        const int *sel,const void *weights,const float *wscale,
        int fmt,int H,int Q,int R,int V,int K,int NS,int chunk,float scale,int gs,int ng){
    int h=blockIdx.x,p=blockIdx.y,tid=threadIdx.x,rbase=h*(Q+V);
    int j0=p*chunk, j1=(j0+chunk<NS)?j0+chunk:NS;
    extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*tsc=cl+K,*tls=tsc+COLI_ATTN_TILE,*red=tls+COLI_ATTN_TILE;
    __shared__ float run_m,run_z;
    float *dst=part+((size_t)h*gridDim.y+p)*(K+2);
    if(j0>=NS){ for(int k=tid;k<K;k+=blockDim.x)dst[k]=0;
        if(!tid){dst[K]=-3.402823466e+38F;dst[K+1]=0;} return; }
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;cl[k]=0;}
    if(!tid){run_m=-3.402823466e+38F;run_z=0;}
    __syncthreads();
    for(int b0=j0;b0<j1;b0+=COLI_ATTN_TILE){
        int tn=j1-b0<COLI_ATTN_TILE?j1-b0:COLI_ATTN_TILE;
        for(int j=tid;j<tn;j+=blockDim.x){int t=sel[b0+j];
            const uint8_t *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
            float a=fp8_row_dot(qa,lt,K);
            float ar=fp8_row_dot(q+(size_t)h*(Q+R)+Q,rt,R);
            tls[j]=lsc[t];
            tsc[j]=(a*tls[j]+ar*rsc[t])*scale;}
        __syncthreads();
        float local=-3.402823466e+38F;
        for(int j=tid;j<tn;j+=blockDim.x)local=fmaxf(local,tsc[j]);
        red[tid]=local;__syncthreads();
        for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
        float old_m=run_m,new_m=fmaxf(old_m,red[0]);
        float fac=expf(old_m-new_m);
        __syncthreads();
        local=0;
        for(int j=tid;j<tn;j+=blockDim.x){float e=expf(tsc[j]-new_m);tsc[j]=e;local+=e;}
        red[tid]=local;__syncthreads();
        for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
        if(!tid){run_z=run_z*fac+red[0];run_m=new_m;}
        for(int j=tid;j<tn;j+=blockDim.x)tsc[j]*=tls[j];
        __syncthreads();
        if(!(K&3)&&!((uintptr_t)latent&3)){
            for(int kq=tid;kq<(K>>2);kq+=blockDim.x){int k0=kq<<2;
                float a0=cl[k0]*fac,a1=cl[k0+1]*fac,a2=cl[k0+2]*fac,a3=cl[k0+3]*fac;
                for(int j=0;j<tn;j++){uint32_t w=*(const uint32_t*)(latent+(size_t)sel[b0+j]*K+k0);float sw=tsc[j];
                    a0+=sw*fp8_e4m3((uint8_t)(w&0xffu));a1+=sw*fp8_e4m3((uint8_t)((w>>8)&0xffu));
                    a2+=sw*fp8_e4m3((uint8_t)((w>>16)&0xffu));a3+=sw*fp8_e4m3((uint8_t)((w>>24)&0xffu));}
                cl[k0]=a0;cl[k0+1]=a1;cl[k0+2]=a2;cl[k0+3]=a3;}
        } else
        for(int k=tid;k<K;k+=blockDim.x){
            float a=cl[k]*fac;
            for(int j=0;j<tn;j++)a+=tsc[j]*fp8_e4m3(latent[(size_t)sel[b0+j]*K+k]);
            cl[k]=a;}
        __syncthreads();
    }
    for(int k=tid;k<K;k+=blockDim.x)dst[k]=cl[k];
    if(!tid){dst[K]=run_m;dst[K+1]=run_z;}
}

/* ==== KV_TQ codec-1 (rotated int4): NATIVE fused attention, no f32 restage =====================
 * Same orthogonality identity the Metal/CPU paths use (proven in test_kv_tq.c): the rotated-int4
 * reconstruction is x_hat = unrotate(c), c_i = Q4LEV[nibble]*std, std = radius/sqrt(n). The
 * randomized-Hadamard rotate/unrotate are ADJOINT orthonormal maps, so WITHOUT dequantizing any
 * row:  score = rotate(q) . c   and   context = unrotate(sum_t w_t c_t). The kernel rotates the
 * absorbed query in shared memory ONCE, dots packed nibbles through the 16-entry codebook,
 * accumulates, and unrotates the context once — the attention_absorb_kernel8 shape at 4 bits.
 *
 * !!! HARDWARE-VALIDATION PENDING: written against the validated Metal/CPU math but NOT yet built
 * on CUDA hardware (fleet box down for the heat wave). Build (sm_89+sm_120) and run the fused
 * f32-reference A/B on the box before relying on this; until then codec-1 falls back to the CPU
 * consumer (now the efficient orthogonality path, not the old 128x-redundant dequant). Scope of
 * this first cut: single-sequence decode, T<=4096 (split-T + batched + device-shadow + DSA-sel
 * variants mirror the kvdev8 family and are the tracked follow-up). */
__constant__ float c_q4lev[16]={ -2.733f,-2.069f,-1.618f,-1.256f,-0.942f,-0.657f,-0.388f,-0.128f,
                                  0.128f, 0.388f, 0.657f, 0.942f, 1.256f, 1.618f, 2.069f, 2.733f };
__device__ static inline uint32_t tq_hash_dev(uint32_t x){
    x+=0x9E3779B9u; x=(x^(x>>16))*0x85EBCA6Bu; x=(x^(x>>13))*0xC2B2AE35u; return x^(x>>16); }
__device__ static inline float tq_sign_dev(int i){ return (tq_hash_dev(0x9E3779B9u^(uint32_t)i)&1u)?-1.f:1.f; }
/* block-parallel normalized FWHT of one power-of-two row resident in shared memory sh[0..n). */
__device__ static inline void tq_fwht_dev(float *sh,int n,int tid,int nth){
    for(int len=1;len<n;len<<=1){
        for(int k=tid;k<(n>>1);k+=nth){ int off=k&(len-1),blk=k/len,j=blk*(len<<1)+off;
            float u=sh[j],v=sh[j+len]; sh[j]=u+v; sh[j+len]=u-v; }
        __syncthreads(); }
    float inv=rsqrtf((float)n); for(int i=tid;i<n;i+=nth) sh[i]*=inv;
    __syncthreads();
}
/* Twin of attention_absorb_kernel8: latent/rope are PACKED NIBBLES (lrb/rrb bytes/row) + per-row
 * radius (lrad/rrad). One block per head; qtr holds the rotated rope query. */
__global__ static void attention_absorb_kernel_tq(float *ctx,const float *q,const uint8_t *latent,
        const float *lrad,const uint8_t *rope,const float *rrad,const void *weights,
        const float *wscale,int fmt,int H,int Q,int R,int V,int K,int T,float scale,int lrb,int rrb,int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T,*qtr=red+256;
    float invsnL=rsqrtf((float)K),invsnR=rsqrtf((float)R);
    /* absorbed latent query qa[k] = sum_d q_d * kv_b[rbase+d][k] */
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    /* rotate the query once: q~ = fwht(sign .* q) — latent (qa in place) + rope (qtr) */
    for(int k=tid;k<K;k+=blockDim.x) qa[k]*=tq_sign_dev(k);
    __syncthreads(); tq_fwht_dev(qa,K,tid,blockDim.x);
    for(int d=tid;d<R;d+=blockDim.x) qtr[d]=q[(size_t)h*(Q+R)+Q+d]*tq_sign_dev(d);
    __syncthreads(); tq_fwht_dev(qtr,R,tid,blockDim.x);
    /* scores: dot rotated query against packed nibbles + per-row radius (std = radius/sqrt(n)) */
    for(int t=tid;t<T;t+=blockDim.x){const uint8_t *lt=latent+(size_t)t*lrb,*rt=rope+(size_t)t*rrb;
        float al=0,ar=0;
        for(int k=0;k<K;k++){int cc=(lt[k>>1]>>((k&1)*4))&0xF;al+=qa[k]*c_q4lev[cc];}
        for(int d=0;d<R;d++){int cc=(rt[d>>1]>>((d&1)*4))&0xF;ar+=qtr[d]*c_q4lev[cc];}
        scores[t]=(al*lrad[t]*invsnL+ar*rrad[t]*invsnR)*scale;}
    __syncthreads();
    /* tree softmax (identical to kernel8) */
    float local=-3.402823466e+38F;for(int t=tid;t<T;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<T;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];
    for(int t=tid;t<T;t+=blockDim.x)scores[t]*=inv*lrad[t]*invsnL;   /* fold w_t*std_t into the weight */
    __syncthreads();
    /* accumulate acc = sum_t w_t*std_t*lev[L[t]] in the rotated basis, then unrotate ONCE */
    for(int k=tid;k<K;k+=blockDim.x){float a=0;
        for(int t=0;t<T;t++){int cc=(latent[(size_t)t*lrb+(k>>1)]>>((k&1)*4))&0xF;a+=scores[t]*c_q4lev[cc];}cl[k]=a;}
    __syncthreads();
    tq_fwht_dev(cl,K,tid,blockDim.x);
    for(int k=tid;k<K;k+=blockDim.x) cl[k]*=tq_sign_dev(k);
    __syncthreads();
    /* output projection ctx[h][v] = sum_k clat[k] * kv_b_value_row */
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

/* Lo scratch cresce monotono col contesto (lb/rb salgono a OGNI token del
 * decode): riallocare alla misura ESATTA significava cudaFree+cudaMalloc per
 * token per device — churn dell'allocatore, sincronizzante, sul percorso caldo.
 * Arrotonda la crescita del 25%. */
static inline size_t grow_bytes(size_t bytes){
    size_t g = bytes + bytes/4;
    return g < bytes ? bytes : g;    /* overflow guard */
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

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

/* Publish quant.h's E8 codebook to every configured device. __constant__ memory
 * is per-device, so this walks the contexts; the engine calls it once after init
 * rather than the backend carrying a second copy of the table that could drift
 * from the CPU decoder's (#452). Safe to call before any fmt=6 upload only. */
extern "C" int coli_cuda_e8_set_grid(const void *grid) {
    if (!grid || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e8_grid, grid, sizeof(c_e8_grid)), "E8 codebook upload"))
            return 0;
    }
    return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    /* #509: the ROCm runtime (comgr, MIOpen, roctracer) reads $TEMP as a temp-dir
     * path. A stray numeric TEMP (the engine's legacy sampling alias) makes comgr's
     * lazy init fail inside the first stream create -- SIGSEGV in the error-unwind
     * on gfx1100, clean hipErrorOutOfMemory on gfx1030. The engine has already
     * parsed g_temp by the time we get here, so a TEMP that is not a real directory
     * is safe to drop before the first ROCm call; a genuine temp-dir is preserved. */
    {
        const char *t = std::getenv("TEMP");
        struct stat st;
        if (t && *t && (stat(t, &st) != 0 || !S_ISDIR(st.st_mode))) unsetenv("TEMP");
    }
#endif
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
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
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
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        if(ctx->alsc)cudaFree(ctx->alsc);if(ctx->arsc)cudaFree(ctx->arsc);
        if(ctx->asel)cudaFree(ctx->asel);if(ctx->apart)cudaFree(ctx->apart);
        for(int b=0;b<27;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_kv) cudaFreeHost(ctx->host_kv);
        if (ctx->kvstage) cudaFreeHost(ctx->kvstage);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x=ctx->host_y=ctx->host_kv=nullptr;ctx->stream=nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->alsc=ctx->arsc=nullptr; ctx->alsc_cap=ctx->arsc_cap=0;
        ctx->asel=nullptr; ctx->asel_cap=0;
        ctx->apart=nullptr; ctx->apart_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=ctx->host_kv_cap=0;
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

/* #653: 1 when the device shares physical memory with the host (Grace-Blackwell /
 * GB10, Jetson, integrated GPUs). On these the expert tier and the RAM cache draw
 * from the same pool, so the RAM budget must account for the tier; on a discrete GPU
 * VRAM is a separate pool and this returns 0. */
extern "C" int coli_cuda_device_integrated(int device) {
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    return prop.integrated ? 1 : 0;
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

/* group size for the NEXT upload on this thread (fmt=4): routed through a
 * thread_local so the widely-wired upload signature (and the Windows DLL ABI)
 * stays untouched. pin_load uploads in parallel, hence thread_local. */
static thread_local int g_upload_gs = 0;
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs);
extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    if (!tensor) return 0;
    if (*tensor) {
        /* Cached device copy: usable even when the caller's host pointers are
         * gone. CUDA_RELEASE_HOST slots null their host pointers after upload,
         * and with the old order (!weights checked first) every later matmul
         * on such a slot failed here — the GPU tier silently never computed
         * for host-released slab experts. */
        ColiCudaTensor *t = *tensor;
        int want_gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device && t->gs == want_gs;
    }
    DeviceContext *ctx = find_ctx(device);
    if (!weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    /* fmt=6 keeps its scales inside each 98-byte block, so it is the one
     * quantized format that legitimately arrives with scales == NULL. */
    if (!rb || (fmt && fmt != 6 && !scales)) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
    t->ng = t->gs ? (I + t->gs - 1) / t->gs : 1;
    t->scale_count = t->gs ? (size_t)O * (size_t)t->ng : (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2||fmt==4){ /* same nibble layout: offset-binary -> signed in place */
        offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt && fmt != 6) {
        if (!cuda_ok(cudaMalloc(&t->scales, t->scale_count * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, t->scale_count * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    if (fmt == 6) t->scale_count = 0;      /* in-block scales: nothing separate to track */
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + ((fmt && fmt != 6) ? t->scale_count * sizeof(float) : 0);
    *tensor = t;
    return 1;
}
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs){
    g_upload_gs = gs>0 ? gs : 0;
    int r = coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device);
    g_upload_gs = 0;
    return r;
}

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || !weights || (tensor->fmt && tensor->fmt != 6 && !scales)) return 0;
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2||tensor->fmt==4){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")) return 0;
    }
    int ng = tensor->ng > 0 ? tensor->ng : 1;
    /* fmt=6 has no scale buffer at all (scales live in-block, scale_count 0), and
     * the fallback below would otherwise copy O floats out of a NULL host pointer. */
    return !tensor->fmt || tensor->fmt==6 || cuda_ok(cudaMemcpy(tensor->scales,scales,
        (tensor->scale_count?tensor->scale_count:(size_t)tensor->O)*sizeof(float),
        cudaMemcpyHostToDevice),"scale refresh");
}

/* Test hook: COLI_GPU_FAIL_AFTER=N makes every GPU COMPUTE entry point report
 * failure after N successful calls (N=0: every call fails), exercising the
 * engine's CPU fallbacks and host-rematerialization end-to-end without real
 * hardware faults. Uploads/queries are not gated. Unset: no effect. */
static long g_gpu_calls;
static int fault_injected(void) {
    const char *fa = std::getenv("COLI_GPU_FAIL_AFTER");
    return fa && g_gpu_calls++ >= std::atol(fa);
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device, int gs) {
    if (fault_injected()) return 0;
    /* fmt=4 carries [O, ceil(I/gs)] scales: without the group size the plain
     * upload truncates the buffer to O floats and quant_matmul divides by
     * gs==0. Callers must come through the gs>0 path (upload_g) or stay on
     * the CPU (#298, #334). */
    if (fmt == 4 && gs <= 0) return 0;
    if (S < 1) return 0;
    if (gs > 0) { if (!coli_cuda_tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs)) return 0; }
    else        { if (!coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0; }
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb, t->gs, t->ng);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (fault_injected()) return 0;
    /* same reason as coli_cuda_matmul: fmt=4 without recorded group info would
     * misread the scales (and divide by gs==0 in the kernel). */
    if (gate && ((gate->fmt == 4 && gate->gs <= 0) ||
                 (up && up->fmt == 4 && up->gs <= 0) ||
                 (down && down->fmt == 4 && down->gs <= 0))) return 0;
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
        gate->fmt,S,D,I,row_bytes(gate->fmt,D),gate->gs,gate->ng);
    quant_matmul<<<hidden_grid,256>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D),up->gs,up->ng);
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    /* fmt=6: the down projection stores W@Q, so its input needs Q^T applied. This
     * one is per-expert (the silu product is not shared), unlike the gate/up input
     * rotation, which the caller does once per layer -- same split as moe(). */
    if (down->fmt == 6 && !e8_rot_rows_dev(ctx->gate, S, I, 0)) return 0;
    quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->gs,down->ng);
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if (fault_injected()) return 0;
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||!COLI_GPU_HAS_WMMA||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    w4a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    if (fault_injected()) return 0;
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1,all_q4=1,any_g4=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")){
        for(int j=0;j<i;j++) cudaEventDestroy(ev[j]); profile=0; break; }   /* (#B8) don't leak the events already created */
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    tc=tc&&all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(all_s4&&COLI_GPU_HAS_WMMA&&ctx->compute_major>=7&&getenv("COLI_CUDA_TC_W4A16")&&
             atoi(getenv("COLI_CUDA_TC_W4A16"))&&
             [&]{ int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
                  for(int c=0;c<count;c++) if(rows[c]>=tc16_min) return 1;
                  return 0; }()){
        /* At least one expert has enough rows for a Tensor Core tile. Groups
         * where EVERY expert is below the threshold (decode: r=1) fall through
         * to the grouped-W4 path below — 3 launches for the whole group instead
         * of 4 per expert (#431: the launch flood measured at ~981 micro-kernels
         * per token came from decode riding this branch's per-expert fallback). */
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                dim3 hg16((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 og16((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                w4a16_gate_up<<<hg16,256,0,ctx->stream>>>(g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul<<<og16,128,0,ctx->stream>>>(y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
            }
            off16+=r;
        }
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{   /* non-dual path has no fused epilogue: silu stays a kernel here */
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_q4&&any_g4){
        /* grouped-int4 (fmt=4) present: per-group scales (#334). fmt=2 members
         * ride along as the ng=1 special case. silu fused in the dual epilogue. */
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        grouped_down_g4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else{
        /* generic path decodes fmt 0/1/2/3 only — a fmt=4 group that slipped the
         * gates above (odd gs) must NOT be silently decoded as int2 (#334). */
        for(int c=0;c<count;c++)
            if(host[c].gf==4||host[c].uf==4||host[c].df==4) return 0;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
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

/* ---- Async expert group (Inc.4): issue/take split of coli_cuda_expert_group ----
 * The measured cost of the sync call at decode is ~0.45 ms/call of HOST-side wait
 * (stream sync + staging), vs ~0.18 ms of actual GPU work — 70% tax, paid ~5x per
 * layer because a token's 8 experts scatter across devices. issue() stages and
 * launches on the device stream and returns immediately; take() syncs and hands
 * back the pinned result rows. One issue may be outstanding per device; moe()
 * takes at each layer end, which also orders the next layer's reuse of the ctx
 * scratch buffers. Small batches only (decode/spec): bigger totals keep the sync
 * path with its TC variants. Numerics are the sync path's small-batch kernels,
 * so greedy output is byte-identical by construction. */
extern "C" int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              const float *x) {
    if (!gates || !ups || !downs || !rows || !x || count < 1 || count > 64) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0,all_s4=1;
    GroupDesc host[64];
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    if(total>8) return 0;                       /* decode-scale only */
    DeviceContext *ctx=find_ctx(device); if(!ctx||ctx->group_pending||!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)) return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue descriptors")||
       !cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue upload")) return 0;
    if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual) grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else {
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(
                ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    } else for(int c=0;c<count;c++){
        int r=rows[c];
        float *g16=ctx->gate+(size_t)host[c].offset*I,*u16=ctx->up+(size_t)host[c].offset*I;
        float *x16=ctx->x+(size_t)host[c].offset*D,*y16=ctx->y+(size_t)host[c].offset*D;
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
            host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
            host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
        silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
        quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
            host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
    }
    if(!cuda_ok(cudaGetLastError(),"expert group issue launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                "expert group issue download")) return 0;
    ctx->group_pending=1; ctx->group_pending_bytes=xb;
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}

extern "C" const float *coli_cuda_expert_group_take(int device) {
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->group_pending) return nullptr;
    ctx->group_pending=0;
    if(!select_ctx(ctx)) return nullptr;
    if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group take")) return nullptr;
    return ctx->host_y;
}


extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!w||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

/* split-T + merge per il decode S=1 oltre il tetto smem: riempie la GPU
 * (H x P blocchi) invece dei soli H del kernel monolitico. */
static int absorb_split_run8(DeviceContext *dc,ColiCudaTensor *w,float *ctx_dev,const float *q_dev,
        const uint8_t *latent_dev,const float *lsc_dev,const uint8_t *rope_dev,const float *rsc_dev,
        int H,int Q,int R,int V,int K,int T,float scale){
    int P=(T+4095)/4096; if(P>64)P=64;
    int chunk=(T+P-1)/P;
    size_t pb=(size_t)H*P*(K+2)*sizeof(float);
    if(!reserve(&dc->apart,&dc->apart_cap,pb))return 0;
    size_t sh1=(size_t)(2*K+2*COLI_ATTN_TILE+256)*sizeof(float);
    attention_absorb_split_kernel8<<<dim3(H,P),256,sh1,dc->stream>>>(dc->apart,q_dev,
        latent_dev,lsc_dev,rope_dev,rsc_dev,w->weights,w->scales,w->fmt,H,Q,R,V,K,T,chunk,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"absorb split launch"))return 0;
    size_t sh2=(size_t)(K+256)*sizeof(float);
    attention_absorb_merge_kernel8<<<H,256,sh2,dc->stream>>>(ctx_dev,dc->apart,
        w->weights,w->scales,w->fmt,H,Q,R,V,K,P,w->gs,w->ng);
    return cuda_ok(cudaGetLastError(),"absorb merge launch");
}

/* ---- KV8: entry point fp8. Stessi contratti dei gemelli f32; latent/rope viaggiano
 * come byte + scala per riga (1/4 del traffico PCIe — a T lunghi e' il collo). */
extern "C" int coli_cuda_attention_absorb8(ColiCudaTensor *w,float *ctx,const float *q,
        const uint8_t *latent,const float *lsc,const uint8_t *rope,const float *rsc,
        int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!ctx||!q||!latent||!lsc||!rope||!rsc||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>(1<<20)||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K,rb=(size_t)T*R;
    size_t sb=(size_t)T*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb)||
       !reserve(&dc->alsc,&dc->alsc_cap,sb)||!reserve(&dc->arsc,&dc->arsc_cap,sb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention fp8 latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->alsc,lsc,sb,cudaMemcpyHostToDevice,dc->stream),"attention latent scale upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention fp8 rope upload")||
       !cuda_ok(cudaMemcpyAsync(dc->arsc,rsc,sb,cudaMemcpyHostToDevice,dc->stream),"attention rope scale upload"))return 0;
    if(T<=4096){
        size_t shared=(size_t)(2*K+T+256)*sizeof(float);
        attention_absorb_kernel8<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,(const uint8_t*)dc->al,
            dc->alsc,(const uint8_t*)dc->ar,dc->arsc,w->weights,w->scales,w->fmt,H,Q,R,V,K,T,scale,w->gs,w->ng);
    } else {                                   /* oltre il tetto smem: split-T + merge */
        if(!absorb_split_run8(dc,w,dc->ac,dc->aq,(const uint8_t*)dc->al,dc->alsc,
            (const uint8_t*)dc->ar,dc->arsc,H,Q,R,V,K,T,scale))return 0;
    }
    if(!cuda_ok(cudaGetLastError(),"attention absorb8 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run8(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const uint8_t *latent,const float *lsc,const uint8_t *rope,
        const float *rsc,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!out||!q||!latent||!lsc||!rope||!rsc||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>(1<<20)||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K,rb=(size_t)T*R;
    size_t sb=(size_t)T*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb)||
       !reserve(&dc->alsc,&dc->alsc_cap,sb)||!reserve(&dc->arsc,&dc->arsc_cap,sb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch fp8 latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->alsc,lsc,sb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent scale upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch fp8 rope upload")||
       !cuda_ok(cudaMemcpyAsync(dc->arsc,rsc,sb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope scale upload"))return 0;
    if(T<=8192){
        size_t shared=(size_t)(2*K+T+256)*sizeof(float);
        attention_absorb_batch_kernel8<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,
            (const uint8_t*)dc->al,dc->alsc,(const uint8_t*)dc->ar,dc->arsc,
            w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    } else {                                   /* oltre il tetto smem: softmax online a tessere */
        size_t shared=(size_t)(2*K+2*COLI_ATTN_TILE+256)*sizeof(float);
        attention_absorb_stream_kernel8<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,
            (const uint8_t*)dc->al,dc->alsc,(const uint8_t*)dc->ar,dc->arsc,
            w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    }
    if(!cuda_ok(cudaGetLastError(),"attention batch8 launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul_rows(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng,dc->stream);
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch8(ColiCudaTensor *w,float *ctx,const float *q,
        const uint8_t *latent,const float *lsc,const uint8_t *rope,const float *rsc,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    return attention_absorb_batch_run8(w,nullptr,ctx,q,latent,lsc,rope,rsc,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch8(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const uint8_t *latent,const float *lsc,const uint8_t *rope,
        const float *rsc,int S,int H,int Q,int R,int V,int K,int T,float scale){
    return attention_absorb_batch_run8(w,proj,out,q,latent,lsc,rope,rsc,S,H,Q,R,V,K,T,scale);
}

/* KV_TQ codec-1 native single-sequence absorb (twin of coli_cuda_attention_absorb8): the packed
 * nibble cache + radii are uploaded per call and consumed by attention_absorb_kernel_tq. lrb/rrb
 * are the codec-1 packed row-byte strides (n/2). T>4096 returns 0 so the caller keeps the CPU
 * path (split-T variant is future work). HARDWARE-VALIDATION PENDING — see the kernel comment. */
extern "C" int coli_cuda_attention_absorb_tq(ColiCudaTensor *w,float *ctx,const float *q,
        const uint8_t *latent,const float *lrad,const uint8_t *rope,const float *rrad,
        int H,int Q,int R,int V,int K,int T,float scale,int lrb,int rrb){
    if(!w||!ctx||!q||!latent||!lrad||!rope||!rrad||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       lrb<1||rrb<1||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*lrb,rb=(size_t)T*rrb;
    size_t sb=(size_t)T*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb)||
       !reserve(&dc->alsc,&dc->alsc_cap,sb)||!reserve(&dc->arsc,&dc->arsc_cap,sb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attn tq q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attn tq latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->alsc,lrad,sb,cudaMemcpyHostToDevice,dc->stream),"attn tq latent radius upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attn tq rope upload")||
       !cuda_ok(cudaMemcpyAsync(dc->arsc,rrad,sb,cudaMemcpyHostToDevice,dc->stream),"attn tq rope radius upload"))return 0;
    size_t shared=(size_t)(2*K+T+256+R)*sizeof(float);
    attention_absorb_kernel_tq<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,(const uint8_t*)dc->al,
        dc->alsc,(const uint8_t*)dc->ar,dc->arsc,w->weights,w->scales,w->fmt,H,Q,R,V,K,T,scale,lrb,rrb,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attn tq launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attn tq context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attn tq synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_project_ragged(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out||!q||!keys||!latent||!rope||!lengths||S<1||S>512||T<1||T>8192||
       H<1||Q<1||R<1||V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);
    if(!select_ctx(dc))return 0;
    float **dl=(float**)std::malloc((size_t)S*sizeof(*dl));
    float **dr=(float**)std::malloc((size_t)S*sizeof(*dr));
    int *old=(int*)std::malloc((size_t)S*sizeof(*old));
    int *add=(int*)std::malloc((size_t)S*sizeof(*add));
    int *off=(int*)std::malloc((size_t)S*sizeof(*off));int packed_n=0;
    if(!dl||!dr||!old||!add||!off){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
    for(int s=0;s<S;s++){
        if(!keys[s]||lengths[s]<1||lengths[s]>T){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
        RaggedKVEntry *e=nullptr;
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){e=&w->ragged[i];break;}
        if(!e){
            if(w->ragged_count>=512){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
            e=&w->ragged[w->ragged_count++];std::memset(e,0,sizeof(*e));e->key=keys[s];
        }
        if(e->K!=K||e->R!=R||e->host_l!=latent[s]||e->host_r!=rope[s]||lengths[s]<e->length){
            if(e->latent)cudaFree(e->latent);if(e->rope)cudaFree(e->rope);
            e->latent=e->rope=nullptr;e->length=e->capacity=0;
            e->K=K;e->R=R;e->host_l=latent[s];e->host_r=rope[s];
        }
        if(lengths[s]>e->capacity){
            int cap=(lengths[s]+63)&~63;float *nl=nullptr,*nr=nullptr;
            if(!cuda_ok(cudaMalloc(&nl,(size_t)cap*K*sizeof(float)),"ragged KV latent page")||
               !cuda_ok(cudaMalloc(&nr,(size_t)cap*R*sizeof(float)),"ragged KV rope page")){
                if(nl)cudaFree(nl);if(nr)cudaFree(nr);std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;
            }
            if(e->length){
                cudaMemcpyAsync(nl,e->latent,(size_t)e->length*K*sizeof(float),cudaMemcpyDeviceToDevice,dc->stream);
                cudaMemcpyAsync(nr,e->rope,(size_t)e->length*R*sizeof(float),cudaMemcpyDeviceToDevice,dc->stream);
            }
            if(e->latent)cudaFree(e->latent);if(e->rope)cudaFree(e->rope);
            e->latent=nl;e->rope=nr;e->capacity=cap;
        }
        dl[s]=e->latent;dr[s]=e->rope;old[s]=e->length;add[s]=lengths[s]-e->length;
        off[s]=packed_n;packed_n+=add[s]*(K+R);
    }
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float);
    size_t cb=(size_t)S*H*V*sizeof(float),ob=(size_t)S*proj->O*sizeof(float);
    size_t pb=(size_t)packed_n*sizeof(float);
    size_t desc=(size_t)S*(2*sizeof(float*)+4*sizeof(int));
    int ok=reserve(&dc->aq,&dc->aq_cap,qb)&&reserve(&dc->ac,&dc->ac_cap,cb)&&
           reserve(&dc->y,&dc->y_cap,ob)&&reserve_bytes(&dc->group_desc,&dc->group_desc_cap,desc)&&
           (!pb||(reserve(&dc->al,&dc->al_cap,pb)&&reserve_pinned(&dc->host_kv,&dc->host_kv_cap,pb)));
    char *db=(char*)dc->group_desc;float **ddl=(float**)db,**ddr=ddl+S;
    int *dn=(int*)(ddr+S),*dold=dn+S,*dadd=dold+S,*doff=dadd+S;
    if(ok&&pb){
        for(int s=0;s<S;s++)if(add[s]){
            float *p=dc->host_kv+off[s];
            std::memcpy(p,latent[s]+(size_t)old[s]*K,(size_t)add[s]*K*sizeof(float));
            std::memcpy(p+(size_t)add[s]*K,rope[s]+(size_t)old[s]*R,(size_t)add[s]*R*sizeof(float));
        }
        ok=cuda_ok(cudaMemcpyAsync(dc->al,dc->host_kv,pb,cudaMemcpyHostToDevice,dc->stream),"ragged KV append upload");
    }
    if(ok)ok=cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"ragged q upload")&&
             cuda_ok(cudaMemcpyAsync(ddl,dl,(size_t)S*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged latent pointers")&&
             cuda_ok(cudaMemcpyAsync(ddr,dr,(size_t)S*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged rope pointers")&&
             cuda_ok(cudaMemcpyAsync(dn,lengths,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged lengths upload")&&
             cuda_ok(cudaMemcpyAsync(dold,old,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged old lengths")&&
             cuda_ok(cudaMemcpyAsync(dadd,add,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append lengths")&&
             cuda_ok(cudaMemcpyAsync(doff,off,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append offsets");
    if(ok&&pb)ragged_kv_append<<<S,256,0,dc->stream>>>(ddl,ddr,dc->al,dold,dadd,doff,K,R);
    if(ok)for(int s=0;s<S;s++){
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){w->ragged[i].length=lengths[s];break;}
    }
    std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);if(!ok)return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_ragged_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,ddl,ddr,
        dn,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    return cuda_ok(cudaGetLastError(),"ragged attention launch")&&
           cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"ragged output download")&&
           cuda_ok(cudaStreamSynchronize(dc->stream),"ragged attention synchronize");
}

/* ---- KV8 long-T: assorbimento con l'ombra fp8 RESIDENTE sul device — sale solo
 * la query, scende solo il contesto. Nessun tetto di T (streaming oltre 4096). */
extern "C" int coli_cuda_attention_absorb_kvdev8(ColiCudaTensor *w,float *ctx,const float *q,
        const uint8_t *latent_dev,const float *lsc_dev,const uint8_t *rope_dev,const float *rsc_dev,
        int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!ctx||!q||!latent_dev||!lsc_dev||!rope_dev||!rsc_dev||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<1||T>(1<<20)||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev8 q upload"))return 0;
    if(T<=4096){
        size_t shared=(size_t)(2*K+T+256)*sizeof(float);
        attention_absorb_kernel8<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,lsc_dev,
            rope_dev,rsc_dev,w->weights,w->scales,w->fmt,H,Q,R,V,K,T,scale,w->gs,w->ng);
    } else {                                   /* oltre il tetto smem: split-T + merge */
        if(!absorb_split_run8(dc,w,dc->ac,dc->aq,latent_dev,lsc_dev,rope_dev,rsc_dev,
            H,Q,R,V,K,T,scale))return 0;
    }
    if(!cuda_ok(cudaGetLastError(),"kvdev8 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev8 ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev8 synchronize"))return 0;
    return 1;
}

/* KV8 + DSA: righe selezionate (indici assoluti nell'ombra device). NS <= 4096:
 * gli score della selezione stanno in shared come nel kernel denso capped. */
extern "C" int coli_cuda_attention_absorb_kvdev8_sel(ColiCudaTensor *w,float *ctx,const float *q,
        const uint8_t *latent_dev,const float *lsc_dev,const uint8_t *rope_dev,const float *rsc_dev,
        const int *sel,int NS,int H,int Q,int R,int V,int K,float scale){
    if(!w||!ctx||!q||!latent_dev||!lsc_dev||!rope_dev||!rsc_dev||!sel||H<1||Q<1||R<1||V<1||
       K<1||K>512||NS<1||NS>4096||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float),sb=(size_t)NS*sizeof(int);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb)||
       !reserve(&dc->asel,&dc->asel_cap,sb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev8 sel q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->asel,sel,sb,cudaMemcpyHostToDevice,dc->stream),"kvdev8 sel list upload"))return 0;
    if(NS>=1024){
        /* H blocchi non riempiono la GPU (ncu: 16.7%% occupancy, 6.6%% SM sul
         * decode DSA di produzione): split della lista in P parziali + merge. */
        int P=(NS+1023)/1024; if(P>64)P=64;
        int chunk=(NS+P-1)/P;
        size_t pb=(size_t)H*P*(K+2)*sizeof(float);
        if(!reserve(&dc->apart,&dc->apart_cap,pb))return 0;
        size_t sh1=(size_t)(2*K+2*COLI_ATTN_TILE+256)*sizeof(float);
        attention_absorb_sel_split_kernel8<<<dim3(H,P),256,sh1,dc->stream>>>(dc->apart,dc->aq,
            latent_dev,lsc_dev,rope_dev,rsc_dev,(const int*)dc->asel,
            w->weights,w->scales,w->fmt,H,Q,R,V,K,NS,chunk,scale,w->gs,w->ng);
        if(!cuda_ok(cudaGetLastError(),"kvdev8 sel split launch"))return 0;
        size_t sh2=(size_t)(K+256)*sizeof(float);
        attention_absorb_merge_kernel8<<<H,256,sh2,dc->stream>>>(dc->ac,dc->apart,
            w->weights,w->scales,w->fmt,H,Q,R,V,K,P,w->gs,w->ng);
    } else {
    size_t shared=(size_t)(2*K+NS+256)*sizeof(float);
    attention_absorb_sel_kernel8<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,lsc_dev,
        rope_dev,rsc_dev,(const int*)dc->asel,w->weights,w->scales,w->fmt,H,Q,R,V,K,NS,scale,w->gs,w->ng);
    }
    if(!cuda_ok(cudaGetLastError(),"kvdev8 sel launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev8 sel ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev8 sel synchronize"))return 0;
    return 1;
}

/* batch + o_proj fusa con l'ombra fp8 residente: per il prefill/verify lungo — sale
 * solo q [S,H,(Q+R)], scende solo out [S,O]. Nessun tetto di T. */
extern "C" int coli_cuda_attention_project_batch_kvdev8(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const uint8_t *latent_dev,const float *lsc_dev,
        const uint8_t *rope_dev,const float *rsc_dev,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    if(!w||!proj||!out||!q||!latent_dev||!lsc_dev||!rope_dev||!rsc_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>(1<<20)||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev8 batch q upload"))return 0;
    if(T<=8192){
        size_t shared=(size_t)(2*K+T+256)*sizeof(float);
        attention_absorb_batch_kernel8<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,
            latent_dev,lsc_dev,rope_dev,rsc_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    } else {
        size_t shared=(size_t)(2*K+2*COLI_ATTN_TILE+256)*sizeof(float);
        attention_absorb_stream_kernel8<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,
            latent_dev,lsc_dev,rope_dev,rsc_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    }
    if(!cuda_ok(cudaGetLastError(),"kvdev8 batch launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul_rows(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng,dc->stream);
    if(!cuda_ok(cudaGetLastError(),"kvdev8 o_proj launch")||
       !cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"kvdev8 out download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev8 batch synchronize"))return 0;
    return 1;
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        int ng = tensor->ng > 0 ? tensor->ng : 1;
        /* Must mirror the upload's accounting exactly: fmt=6 never charged for a
         * scale buffer, and over-subtracting here trips the >= guard below, which
         * silently leaves the tensor's bytes on the device counter forever. */
        size_t bytes = tensor->weight_bytes +
            ((tensor->fmt && tensor->fmt != 6) ? (size_t)tensor->O * ng * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    for(int i=0;i<tensor->ragged_count;i++){
        if(tensor->ragged[i].latent)cudaFree(tensor->ragged[i].latent);
        if(tensor->ragged[i].rope)cudaFree(tensor->ragged[i].rope);
    }
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    int ng = tensor->ng > 0 ? tensor->ng : 1;
    return tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * ng * sizeof(float) : 0);
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=27||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
/* Upload ORDINATO sullo stream dei kernel (ctx->stream). La variante sincrona
 * sopra viaggia sul NULL stream: con sorgente pageable la cudaMemcpy puo'
 * tornare mentre il DMA e' ancora in volo, e i kernel kvdev8 (stream
 * non-blocking) potevano leggere righe/scale stantie del token piu' recente.
 * Qui: staging pinned + cudaMemcpyAsync sullo stesso stream dei kernel — la
 * race sparisce per ordinamento e l'host non si blocca (il decode pagava 4
 * copie bloccanti per layer per token, due da 4 byte). Il ring si riusa dopo
 * una sync di stream (ogni chiamata attention termina con una). */
extern "C" int coli_cuda_pipe_upload_async(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!bytes) return 1;
    if(bytes<=((size_t)16<<20)){
        if(ctx->kvstage_cap<bytes){
            if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"kv stage sync")) return 0;
            if(ctx->kvstage) cudaFreeHost(ctx->kvstage);
            size_t cap=bytes<((size_t)4<<20)?((size_t)4<<20):grow_bytes(bytes);
            ctx->kvstage=nullptr; ctx->kvstage_cap=0; ctx->kvstage_off=0;
            if(cuda_ok(cudaMallocHost((void**)&ctx->kvstage,cap),"kv stage alloc"))
                ctx->kvstage_cap=cap;
            else ctx->kvstage=nullptr;               /* fail-soft: pageable sotto */
        }
        if(ctx->kvstage){
            if(ctx->kvstage_off+bytes>ctx->kvstage_cap){
                if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"kv stage wrap")) return 0;
                ctx->kvstage_off=0;
            }
            uint8_t *st=ctx->kvstage+ctx->kvstage_off;
            memcpy(st,src,bytes); ctx->kvstage_off+=bytes;
            return cuda_ok(cudaMemcpyAsync(dst,st,bytes,cudaMemcpyHostToDevice,ctx->stream),
                           "pipe upload async");
        }
    }
    /* pageable: la copia degrada a staged-sincrona ma RESTA ordinata sullo stream */
    return cuda_ok(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,ctx->stream),
                   "pipe upload async");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
/* ---- device router (#431 PR-A) -------------------------------------------
 * Router for one decode row, entirely on the layer's home device: logits GEMV
 * (E x D, tiny) + sigmoid, bias-augmented top-K selection, route-level TOPP
 * truncation, norm_topk and routed_scale — a float-faithful clone of moe()'s
 * plain routing path (colibri.c FASE A). Selection runs single-thread so the
 * argmax order, tie-breaking (strict >, lowest index wins) and weight math
 * match the CPU reference exactly; only the dot/expf rounding can differ,
 * which is the documented kernel-family divergence class (#100/#163).
 * Results are packed [idx[K] | w[K] | keff] in one scratch buffer and read
 * back with a single tiny D2H. */
__global__ void pipe_router_logits(const float *__restrict__ x,
                                   const float *__restrict__ W,
                                   const float *__restrict__ bias,
                                   int D, float *logit, float *choice){
    int e = blockIdx.x;
    const float *w = W + (size_t)e*D;
    float acc = 0.f;
    for(int i=threadIdx.x; i<D; i+=blockDim.x) acc += x[i]*w[i];
    __shared__ float sh[128];
    sh[threadIdx.x]=acc; __syncthreads();
    for(int s=blockDim.x>>1; s>0; s>>=1){
        if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s];
        __syncthreads();
    }
    if(!threadIdx.x){
        float lg = 1.f/(1.f+expf(-sh[0]));
        logit[e]=lg; choice[e]=lg+bias[e];
    }
}
__global__ void pipe_router_select(const float *__restrict__ logit,
                                   const float *__restrict__ choice, int E,
                                   int Ksel, float topp, int norm_topk,
                                   float routed_scale, char *out){
    if(threadIdx.x||blockIdx.x) return;
    int   *idx = (int*)out;
    float *w   = (float*)(out + Ksel*sizeof(int));
    int   *keff= (int*)(out + Ksel*(sizeof(int)+sizeof(float)));
    for(int kk=0;kk<Ksel;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
            if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
        idx[kk]=best; w[kk]=logit[best];
    }
    int Ke=Ksel;
    if(topp>0.f && topp<1.f){
        for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
            while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
        float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
        float cum=0.f; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
    }
    if(norm_topk){ float sm=0.f; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f;
                   for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
    for(int kk=0;kk<Ke;kk++) w[kk]*=routed_scale;
    *keff=Ke;
}
extern "C" int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host){
    DeviceContext *ctx=find_ctx(device);
    if(!x_dev||!rw_dev||!rb_dev||D<1||E<1||E>4096||Ksel<1||Ksel>64||!select_ctx(ctx)) return 0;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    float *logit=coli_cuda_pipe_scratch(device,22,(size_t)E*sizeof(float));
    float *chc  =coli_cuda_pipe_scratch(device,23,(size_t)E*sizeof(float));
    char  *out  =(char*)coli_cuda_pipe_scratch(device,24,pack);
    if(!logit||!chc||!out) return 0;
    pipe_router_logits<<<E,128>>>(x_dev,(const float*)rw_dev,(const float*)rb_dev,D,logit,chc);
    pipe_router_select<<<1,1>>>(logit,chc,E,Ksel,topp,norm_topk,routed_scale,out);
    if(!cuda_ok(cudaGetLastError(),"pipe router launch")) return 0;
    char buf[64*(sizeof(int)+sizeof(float))+sizeof(int)];
    if(!cuda_ok(cudaMemcpy(buf,out,pack,cudaMemcpyDeviceToHost),"pipe router readback")) return 0;
    memcpy(idx_host,buf,(size_t)Ksel*sizeof(int));
    memcpy(w_host,buf+Ksel*sizeof(int),(size_t)Ksel*sizeof(float));
    memcpy(keff_host,buf+Ksel*(sizeof(int)+sizeof(float)),sizeof(int));
    return 1;
}
/* ---- resident expert-group accumulation (#431 PR-C0) ----------------------
 * Decode-time (S=1) expert groups without the host round-trip: the input row
 * is P2P'd from the layer's home device, the group runs through the grouped-W4
 * kernels on its own stream, the down-projection outputs are weighted and
 * reduced ON DEVICE (fixed expert order), and the device's partial sum is
 * peer-pushed into a per-issue slot on the home device. take() makes the home
 * legacy stream wait on every issue event and reduces the slots in issue order
 * — deterministic, no atomics, no host bytes. The CPU tier overlaps with all
 * of it exactly as before. */
__global__ static void bcast_row(float *dst,const float *src,int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float v=src[i];
        for(int c=0;c<count;c++) dst[(size_t)c*D+i]=v;
    }
}
__global__ static void weighted_sum_rows(float *out,const float *y,const float *w,
                                         int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int c=0;c<count;c++) acc+=w[c]*y[(size_t)c*D+i];   /* fixed order */
        out[i]=acc;
    }
}
__global__ static void sum_slots(float *dst,const float *slots,int n,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int s=0;s<n;s++) acc+=slots[(size_t)s*D+i];        /* issue order */
        dst[i]=acc;
    }
}
extern "C" int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev){
    if(!gates||!ups||!downs||!weights||count<1||count>64||!x_src_dev||!partial_slot_dev) return 0;
    ColiCudaTensor *first=gates[0]; if(!first) return 0;
    int device=first->device,D=first->I,I=first->O;
    GroupDesc host[64];
    int total=0,all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,1,total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total++;
    }
    if(!all_s4) return 0;                       /* resident path: per-row int4 only */
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!ctx->ev_done_ok){
        if(!cuda_ok(cudaEventCreateWithFlags(&ctx->ev_done,cudaEventDisableTiming),
                    "resident group event")) return 0;
        ctx->ev_done_ok=1;
    }
    /* size for the 64-expert cap, not for `count`: reserve() reallocs on growth,
     * and a realloc here could free a buffer the PREVIOUS layer's still-queued
     * async work on this stream reads. Fixed caps make re-issue realloc-free. */
    size_t xb=(size_t)64*D*sizeof(float), ib=(size_t)64*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->ac,&ctx->ac_cap,(size_t)(D+64)*sizeof(float))||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)64*sizeof(GroupDesc)))
        return 0;
    float *w_dev=ctx->ac+D, *partial_local=ctx->ac;
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group desc")||
       !cuda_ok(cudaMemcpyAsync(w_dev,weights,(size_t)count*sizeof(float),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group weights"))
        return 0;
    /* input row: P2P from the home device. The caller guarantees x_src_dev is
     * materialized (the pre-moe nrm download already synced the home stream). */
    if(!cuda_ok(cudaMemcpyPeerAsync(ctx->x,device,x_src_dev,home_device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident group x p2p"))
        return 0;
    bcast_row<<<64,256,0,ctx->stream>>>(ctx->x,ctx->x,count,D);   /* row 0 -> rows 1..count-1 (in-place safe: row 0 rewritten with itself) */
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    dim3 hg((unsigned)I,1,(unsigned)count),og((unsigned)D,1,(unsigned)count);
    grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);  /* silu fused in epilogue */
    grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    weighted_sum_rows<<<48,256,0,ctx->stream>>>(partial_local,ctx->y,w_dev,count,D);
    if(!cuda_ok(cudaMemcpyPeerAsync(partial_slot_dev,home_device,partial_local,device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident partial p2p"))
        return 0;
    if(!cuda_ok(cudaEventRecord(ctx->ev_done,ctx->stream),"resident event record")) return 0;
    return cuda_ok(cudaGetLastError(),"resident group launch");
}
extern "C" int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,
                                           float *slots_dev,float *acc_dev,int D){
    if(n_issued<1||!slots_dev||!acc_dev||D<1) return 0;
    DeviceContext *home=find_ctx(home_device); if(!select_ctx(home)) return 0;
    for(int i=0;i<n_issued;i++){
        DeviceContext *src=find_ctx(devices[i]);
        if(!src||!src->ev_done_ok) return 0;
        if(!cuda_ok(cudaStreamWaitEvent(0,src->ev_done,0),"resident take wait")) return 0;
    }
    sum_slots<<<48,256>>>(acc_dev,slots_dev,n_issued,D);          /* legacy stream: ordered with pipe_* */
    return cuda_ok(cudaGetLastError(),"resident take reduce");
}
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2D(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice),"pipe copy2d");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if (fault_injected()) return 0;
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    dim3 grid((unsigned)t->O,(unsigned)S);
    quant_matmul<<<grid,256>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
        row_bytes(t->fmt,t->I),t->gs,t->ng);
    return cuda_ok(cudaGetLastError(),"pipe gemm");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx_dev,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    if(!w||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,1),256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,1,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"kvdev absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaDeviceSynchronize(),"pipe sync");
}
