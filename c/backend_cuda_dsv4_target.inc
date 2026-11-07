/* A bounded, full-set attention target. Absolute history makes retained-prefix
 * rollback a visibility change: no ring overwrite and no replay of projections.
 * Admission at the engine boundary forbids crossing index_topk. */
struct Dsv4CudaTargetWorkspace {
    int device, capacity, start;
    Dsv4CudaActivation *residual, *middle, *normed, *branch, *hc, *qa, *q, *kv, *context;
    float *comp_kv, *comp_gate;
    int *control, *tokens, *meta;
    float *taps;
    Dsv4CudaGraph *graphs[129];
    bool warmed[129];
    size_t scratch[12];
    unsigned long captures, replays;
    bool mma;
    uint8_t *mma_input;
    float *mma_scale;
};
struct Dsv4CudaTargetLayer {
    Dsv4CudaTargetWeights w;
    float *values, *projected_kv, *projected_gate, *cosines, *sines;
    int64_t *past_positions;
};

/* Same 16-row register layout and 16-byte loads as Naruto's verified
 * cuda_quantized_mma.cuh. Activation scales are per-row, per 128 columns;
 * C1 and speculative rows deliberately use identical tiling/reductions. */
__global__ void target_quantize(uint8_t *out,float *scales,const float *x,int width) {
    int row=blockIdx.y,base=blockIdx.x*128,lane=threadIdx.x;
    float value=x[(long long)row*width+base+lane];
    __shared__ float maximum[128];maximum[lane]=fmaxf(fabsf(value),1e-4f);__syncthreads();
    for(int stride=64;stride;stride>>=1){if(lane<stride)maximum[lane]=fmaxf(maximum[lane],maximum[lane+stride]);__syncthreads();}
    float raw=maximum[0]/448.f;int exponent;frexpf(raw,&exponent);
    float scale=ldexpf(1.f,exponent-1);if(scale<raw)scale*=2.f;
    if(!lane)scales[(long long)row*(width/128)+blockIdx.x]=scale;
    out[(long long)row*width+base+lane]=__nv_cvt_float_to_fp8(value/scale,__NV_SATFINITE,__NV_E4M3);
}
__device__ void target_fp8_pair(uint4 a0,uint4 a1,uint4 b,float *p) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3])
        : "r"(a0.x),"r"(a1.x),"r"(a0.y),"r"(a1.y),"r"(b.x),"r"(b.y));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3])
        : "r"(a0.z),"r"(a1.z),"r"(a0.w),"r"(a1.w),"r"(b.z),"r"(b.w));
#endif
}
template<bool Split> __global__ void target_fp8_mma(const uint8_t *x,const float *xs,
    const uint8_t *w,const uint8_t *ws,float *out,int rows,int O,int I) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,part=lane%4;
    int column=Split?blockIdx.x*8:(blockIdx.x*8+warp)*8;
    int row0=blockIdx.y*16+group,row1=row0+8;
    int span=Split?I/8:I,begin=Split?warp*span:0;
    float accum[4]={0.f,0.f,0.f,0.f};
    for(int k=begin;k<begin+span;k+=64) {
        uint4 a0=make_uint4(0,0,0,0),a1=a0,b=a0;
        if(row0<rows)a0=*(const uint4*)(x+(long long)row0*I+k+part*16);
        if(row1<rows)a1=*(const uint4*)(x+(long long)row1*I+k+part*16);
        if(column+group<O)b=*(const uint4*)(w+(long long)(column+group)*I+k+part*16);
        float p[4]={0.f,0.f,0.f,0.f};target_fp8_pair(a0,a1,b,p);
        float s0=row0<rows?xs[(long long)row0*(I/128)+k/128]:1.f;
        float s1=row1<rows?xs[(long long)row1*(I/128)+k/128]:1.f;
        float weight=column<O?e8(ws[(column/128)*(I/128)+k/128]):1.f;
        accum[0]=fmaf(p[0],s0*weight,accum[0]);accum[1]=fmaf(p[1],s0*weight,accum[1]);
        accum[2]=fmaf(p[2],s1*weight,accum[2]);accum[3]=fmaf(p[3],s1*weight,accum[3]);
    }
    if constexpr(Split) {
        __shared__ float partial[8][32][4];
        for(int i=0;i<4;i++)partial[warp][lane][i]=accum[i];__syncthreads();
        if(warp)return;
        for(int i=0;i<4;i++){float sum=0.f;for(int j=0;j<8;j++)sum=__fadd_rn(sum,partial[j][lane][i]);accum[i]=sum;}
    }
    int o=column+part*2;
    if(row0<rows && o<O){out[(long long)row0*O+o]=accum[0];out[(long long)row0*O+o+1]=accum[1];}
    if(row1<rows && o<O){out[(long long)row1*O+o]=accum[2];out[(long long)row1*O+o+1]=accum[3];}
}
static void target_matmul(Dsv4CudaTargetWorkspace *s,Dsv4CudaTensor *w,const float *x,float *y,int rows,cudaStream_t stream) {
    if (!s->mma) {run_mm_batch(w,x,y,rows,stream);return;}
    target_quantize<<<dim3(w->I/128,rows),128,0,stream>>>(s->mma_input,s->mma_scale,x,w->I);
    if(w->O<=8192 && w->I>=1024)
        target_fp8_mma<true><<<dim3((w->O+7)/8,(rows+15)/16),256,0,stream>>>(s->mma_input,s->mma_scale,(uint8_t*)w->w,w->scale,y,rows,w->O,w->I);
    else target_fp8_mma<false><<<dim3((w->O+63)/64,(rows+15)/16),256,0,stream>>>(s->mma_input,s->mma_scale,(uint8_t*)w->w,w->scale,y,rows,w->O,w->I);
}
static int target_output_projection(Dsv4CudaTargetWorkspace *s,const Dsv4CudaTargetWeights &w,int rows) {
    if (!s->mma && !w.draft) return dsv4_cuda_attention_output_batch(s->context,w.wa,w.wb,w.groups,rows,s->branch);
    auto *c=ctx(s->device);int K=w.wa->I,N=w.wa->O/w.groups,joined=w.wa->O;
    if (!buf((void**)&c->aux1,&c->aux1cap,(size_t)rows*K*4) ||
        !buf((void**)&c->aux2,&c->aux2cap,(size_t)rows*N*4) ||
        !buf((void**)&c->p1,&c->p1cap,(size_t)rows*joined*4)) return 0;
    for (int gidx=0;gidx<w.groups;gidx++) {
        gather_group_rows<<<(rows*K+255)/256,256,0,c->stream>>>(c->aux1,s->context->data,rows,w.groups,K,gidx);
        auto view=*w.wa;view.O=N;view.w=(uint8_t*)w.wa->w+(long long)gidx*N*K;
        view.scale=w.wa->scale+(long long)gidx*(N/128)*(K/128);
        target_matmul(s,&view,c->aux1,c->aux2,rows,c->stream);
        if(w.draft)bf16_round<<<(rows*N+255)/256,256,0,c->stream>>>(c->aux2,rows*N);
        scatter_group_rows<<<(rows*N+255)/256,256,0,c->stream>>>(c->p1,c->aux2,rows,w.groups,N,gidx);
    }
    target_matmul(s,w.wb,c->p1,s->branch->data,rows,c->stream);
    if(w.draft)bf16_round<<<(rows*w.hidden+255)/256,256,0,c->stream>>>(s->branch->data,rows*w.hidden);
    return ok(cudaGetLastError(),"target MMA output projection");
}

__global__ void target_norm(float *x, const float *weight, int width, float eps) {
    x += (long long)blockIdx.x * width;
    __shared__ float inverse;
    if (!threadIdx.x) {
        float sum = 0.f;
        for (int d = 0; d < width; d++) sum = __fadd_rn(sum, __fmul_rn(x[d], x[d]));
        inverse = 1.f / sqrtf(sum / width + eps);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float y = __fmul_rn(x[d], inverse);
        if (weight) y = __fmul_rn(y, weight[d]);
        x[d] = __bfloat162float(__float2bfloat16(y));
    }
}
__global__ void target_rope(float *x, const float *co, const float *si,
    const int *control, int heads, int dim, int rope, int inverse) {
    int t = blockIdx.x / heads, h = blockIdx.x % heads, pairs = rope / 2;
    x += ((long long)t * heads + h) * dim + dim - rope;
    co += (long long)(control[0] + t) * pairs;
    si += (long long)(control[0] + t) * pairs;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        float a = x[2*p], b = x[2*p+1], s = inverse ? -si[p] : si[p];
        x[2*p] = __bfloat162float(__float2bfloat16(__fsub_rn(__fmul_rn(a, co[p]), __fmul_rn(b, s))));
        x[2*p+1] = __bfloat162float(__float2bfloat16(__fadd_rn(__fmul_rn(b, co[p]), __fmul_rn(a, s))));
    }
}
__global__ void target_cache_store(float *history, const float *kv, const int *control, int dim, int nope, int offset) {
    int t = blockIdx.x, base = blockIdx.y * 64, d = base + threadIdx.x;
    if (d >= dim) return;
    const float *x = kv + (long long)t * dim;
    float value = x[d];
    if (d < nope) {
        float maximum = 1e-4f;
        for (int k = base; k < base + 64 && k < nope; k++) maximum = fmaxf(maximum, fabsf(x[k]));
        float raw = maximum / 448.f; int e; frexpf(raw, &e);
        float scale = ldexpf(1.f, e - 1); if (scale < raw) scale *= 2.f;
        value = e4m3_round(fmaxf(-448.f, fminf(448.f, value / scale))) * scale;
    }
    history[(long long)((offset>=0?offset:control[0]) + t) * dim + d] = __bfloat162float(__float2bfloat16(value));
}
__global__ void target_projection_store(float *history, const float *x, const int *control, int width, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width) history[(long long)control[0] * width + i] = x[i];
}
__global__ void target_compress(float *out, const float *kv, const float *gate,
    const float *ape, const float *norm, const float *co, const float *si,
    const int *control, int ratio, int dim, int rope, float eps) {
    int position = control[0] + blockIdx.x;
    if ((position + 1) % ratio) return;
    int overlap = ratio == 4, width = (1 + overlap) * dim;
    int begin = position + 1 - (1 + overlap) * ratio;
    __shared__ float row[512], inverse;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float maximum = -INFINITY;
        for (int j = 0; j < (1 + overlap) * ratio; j++) if (begin + j >= 0) {
            int c = d + (overlap && j >= ratio ? dim : 0);
            float score = gate[(long long)(begin + j) * width + c] + ape[(j % ratio) * width + c];
            maximum = fmaxf(maximum, score);
        }
        float total = 0.f, sum = 0.f;
        for (int j = 0; j < (1 + overlap) * ratio; j++) if (begin + j >= 0) {
            int c = d + (overlap && j >= ratio ? dim : 0);
            float score = gate[(long long)(begin + j) * width + c] + ape[(j % ratio) * width + c];
            float a = expf(score - maximum); total += a;
            sum = __fadd_rn(sum, __fmul_rn(kv[(long long)(begin + j) * width + c], a));
        }
        row[d] = __bfloat162float(__float2bfloat16(sum / total));
    }
    __syncthreads();
    if (!threadIdx.x) {
        float sum = 0.f;
        for (int d = 0; d < dim; d++) sum = __fadd_rn(sum, __fmul_rn(row[d], row[d]));
        inverse = 1.f / sqrtf(sum / dim + eps);
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        row[d] = __bfloat162float(__float2bfloat16(__fmul_rn(__fmul_rn(row[d], inverse), norm[d])));
    __syncthreads();
    int pairs = rope / 2, rp = position + 1 - ratio;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        float a = row[dim - rope + 2*p], b = row[dim - rope + 2*p+1];
        float c = co[(long long)rp * pairs + p], s = si[(long long)rp * pairs + p];
        row[dim - rope + 2*p] = __bfloat162float(__float2bfloat16(__fsub_rn(__fmul_rn(a,c), __fmul_rn(b,s))));
        row[dim - rope + 2*p+1] = __bfloat162float(__float2bfloat16(__fadd_rn(__fmul_rn(b,c), __fmul_rn(a,s))));
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = row[d];
        if (d < dim - rope) {
            int begin64 = d / 64 * 64; float maximum = 1e-4f;
            for (int k = begin64; k < begin64 + 64 && k < dim - rope; k++) maximum = fmaxf(maximum, fabsf(row[k]));
            float raw = maximum / 448.f; int e; frexpf(raw, &e);
            float scale = ldexpf(1.f, e - 1); if (scale < raw) scale *= 2.f;
            value = e4m3_round(fmaxf(-448.f, fminf(448.f, value / scale))) * scale;
        }
        out[(long long)(position / ratio) * dim + d] = __bfloat162float(__float2bfloat16(value));
    }
}
__global__ void target_meta(int *meta, const int *control, int rows, int window, int ratio) {
    int t = threadIdx.x; if (t >= rows) return;
    int end = control[0] + t + 1, begin = max(0, end - window);
    meta[t * 3] = begin; meta[t * 3 + 1] = end - begin; meta[t * 3 + 2] = ratio ? end / ratio : 0;
}
__global__ void target_hash(int *ids, const int64_t *map, const int *tokens, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * 6) ids[i] = (int)map[(long long)tokens[i / 6] * 6 + i % 6];
}
/* One warp per output and four-byte FP4 loads, following Naruto's portable
 * vectorized expert GEMV. No CTA-wide reduction barriers are needed. */
__global__ void target_fp4_vec(const MvDesc *descriptors,int count,int outputs,int inputs) {
    int expert=blockIdx.y,o=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(expert>=count || o>=outputs)return;
    MvDesc d=descriptors[expert];float sum=0.f;
    for(int base=lane*8;base<inputs;base+=256) {
        uint32_t packed=*(const uint32_t*)(d.w+(long long)o*(inputs/2)+base/2);
        float scale=e8(d.scale[(long long)o*(inputs/32)+base/32]);
        float4 a=*(const float4*)(d.x+base),b=*(const float4*)(d.x+base+4);
        float x[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
        #pragma unroll
        for(int i=0;i<8;i++)sum=fmaf(x[i],f4((packed>>(i*4))&15)*scale,sum);
    }
    for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(!lane)d.y[o]=sum;
}

extern "C" Dsv4CudaTargetWorkspace *dsv4_cuda_target_workspace_create(int device, int rows) {
    if (!ctx(device) || rows < 1 || rows > 128 || !ok(cudaSetDevice(device), "target device")) return nullptr;
    auto *s = new Dsv4CudaTargetWorkspace{}; s->device = device; s->capacity = rows;
    const char *mma=getenv("V4_DEVICE_MMA");s->mma=mma && atoi(mma)!=0;
    if (s->mma) {
        cudaDeviceProp prop{};cudaFuncAttributes compiled{};
        if (!ok(cudaGetDeviceProperties(&prop,device),"target MMA capability") || prop.major*10+prop.minor<89 ||
            !ok(cudaFuncGetAttributes(&compiled,target_fp8_mma<false>),"target MMA compilation") || compiled.ptxVersion<89) {
            fprintf(stderr,"[device target] FP8 MMA requires both device and compilation target sm_89 or newer\n");
            delete s;return nullptr;
        }
    }
    Dsv4CudaActivation **slots[] = {&s->residual,&s->middle,&s->normed,&s->branch,&s->hc,&s->qa,&s->q,&s->kv,&s->context};
    const int widths[] = {16384,16384,4096,4096,20,1024,32768,512,32768};
    bool good = true;
    if (s->mma) good=ok(cudaMalloc(&s->mma_input,(size_t)rows*8192),"target MMA activations") && ok(cudaMalloc(&s->mma_scale,(size_t)rows*64*4),"target MMA scales");
    for (int i = 0; i < 9; i++) { *slots[i] = dsv4_cuda_activation_create(device, (long long)rows * widths[i]); good &= *slots[i] != nullptr; }
    good = good && ok(cudaMalloc(&s->control, sizeof(int)), "target position") &&
        ok(cudaMalloc(&s->tokens, rows * sizeof(int)), "target tokens") &&
        ok(cudaMalloc(&s->meta, rows * 3 * sizeof(int)), "target metadata") &&
        ok(cudaMalloc(&s->taps, (size_t)rows * 3 * 16384 * 4), "target taps") &&
        ok(cudaMalloc(&s->comp_kv, (size_t)rows * 1024 * 4), "target compressor values") &&
        ok(cudaMalloc(&s->comp_gate, (size_t)rows * 1024 * 4), "target compressor scores");
    if (!good) { dsv4_cuda_target_workspace_free(s); return nullptr; }
    return s;
}
extern "C" void dsv4_cuda_target_workspace_free(Dsv4CudaTargetWorkspace *s) {
    if (!s) return; cudaSetDevice(s->device);
    for (auto *graph:s->graphs) dsv4_cuda_graph_free(graph);
    if (s->captures || s->replays) fprintf(stderr,"[device target] device=%d captures=%lu replays=%lu\n",s->device,s->captures,s->replays);
    for (auto *a : {s->residual,s->middle,s->normed,s->branch,s->hc,s->qa,s->q,s->kv,s->context}) dsv4_cuda_activation_free(a);
    cudaFree(s->control); cudaFree(s->tokens); cudaFree(s->meta); cudaFree(s->comp_kv); cudaFree(s->comp_gate); cudaFree(s->taps); cudaFree(s->mma_input); cudaFree(s->mma_scale); delete s;
}
extern "C" int dsv4_cuda_target_input(Dsv4CudaTargetWorkspace *s, const float *x, const int *tokens, int start, int rows) {
    if (!s || !x || !tokens || rows < 1 || rows > s->capacity || start < 0 || start>2048-rows) return 0;
    s->start=start;
    auto *c = ctx(s->device); if (!ok(cudaSetDevice(s->device), "target input device")) return 0;
    return ok(cudaMemcpyAsync(s->residual->data,x,(size_t)rows*16384*4,cudaMemcpyHostToDevice,c->stream),"target residual input") &&
        ok(cudaMemcpyAsync(s->tokens,tokens,rows*sizeof(int),cudaMemcpyHostToDevice,c->stream),"target token input") &&
        ok(cudaMemcpyAsync(s->control,&s->start,sizeof(int),cudaMemcpyHostToDevice,c->stream),"target position input");
}
extern "C" int dsv4_cuda_target_transfer(Dsv4CudaTargetWorkspace *dst, Dsv4CudaTargetWorkspace *src, int rows) {
    if (!src || !dst || rows < 1 || rows > src->capacity || rows > dst->capacity) return 0;
    dst->start=src->start;
    auto *a=ctx(src->device), *b=ctx(dst->device);
    if (!ok(cudaSetDevice(src->device),"target source device") || !ok(cudaEventRecord(a->ready,a->stream),"target stage ready") ||
        !ok(cudaSetDevice(dst->device),"target destination device") || !ok(cudaStreamWaitEvent(b->stream,a->ready,0),"target stage wait")) return 0;
    return ok(cudaMemcpyPeerAsync(dst->residual->data,dst->device,src->residual->data,src->device,(size_t)rows*16384*4,b->stream),"target stage residual") &&
        ok(cudaMemcpyPeerAsync(dst->tokens,dst->device,src->tokens,src->device,rows*sizeof(int),b->stream),"target stage tokens") &&
        ok(cudaMemcpyPeerAsync(dst->control,dst->device,src->control,src->device,sizeof(int),b->stream),"target stage position");
}
extern "C" int dsv4_cuda_target_output(Dsv4CudaTargetWorkspace *s, float *x, int rows) {
    return s && x && rows > 0 && rows <= s->capacity &&
        dsv4_cuda_activation_download(x,s->residual,(long long)rows*16384) && dsv4_cuda_activation_sync(s->residual);
}
extern "C" Dsv4CudaTargetLayer *dsv4_cuda_target_layer_create(const Dsv4CudaTargetWeights *w, const float *co, const float *si) {
    if (!w || !co || !si || !w->qa || !w->experts || w->hidden!=4096 || w->rank!=1024 ||
        w->heads!=64 || w->dim!=512 || w->rope!=64 || w->capacity<1 || w->capacity>2048 ||
        (w->ratio!=0 && w->ratio!=4 && w->ratio!=128)) return nullptr;
    int device=w->qa->device;
    Dsv4CudaTensor *required[]={w->qb,w->kv,w->wa,w->wb,w->qnorm,w->kvnorm,w->sink,w->gate,
        w->hc_fn[0],w->hc_fn[1],w->hc_scale[0],w->hc_scale[1],w->hc_base[0],w->hc_base[1],w->norm[0],w->norm[1]};
    for (auto *t:required) if (!t || t->device!=device) return nullptr;
    if (w->experts->device!=device || w->window<1 || w->window>w->capacity || w->groups!=8 ||
        w->qa->O!=w->rank || w->qa->I!=w->hidden || w->qb->O!=w->heads*w->dim || w->qb->I!=w->rank ||
        w->kv->O!=w->dim || w->kv->I!=w->hidden || w->qnorm->fmt!=32 || w->kvnorm->fmt!=32 ||
        w->sink->fmt!=32 || w->qa->fmt!=8 || w->qb->fmt!=8 || w->kv->fmt!=8 ||
        w->wa->fmt!=9 || w->wa->O!=8192 || w->wa->I!=4096 ||
        w->wb->fmt!=8 || w->wb->O!=4096 || w->wb->I!=8192 ||
        (w->draft && (w->capacity<133 || w->ratio))) return nullptr;
    if(w->ratio) {
        int width=(w->ratio==4?2:1)*w->dim;
        Dsv4CudaTensor *comp[]={w->comp_kv,w->comp_gate,w->comp_norm,w->ape};
        for(auto *t:comp)if(!t || t->device!=device)return nullptr;
        if(w->comp_kv->fmt!=16 || w->comp_gate->fmt!=16 || w->comp_norm->fmt!=32 || w->ape->fmt!=32 ||
            w->comp_kv->O!=width || w->comp_gate->O!=width || w->comp_kv->I!=4096 || w->comp_gate->I!=4096)return nullptr;
    }
    if (!ok(cudaSetDevice(device),"target layer device")) return nullptr;
    auto *l=new Dsv4CudaTargetLayer{}; l->w=*w;
    size_t rb=(size_t)w->capacity*(w->rope/2)*4;
    int compressed=w->ratio?w->capacity/w->ratio:0, width=(w->ratio==4?2:1)*w->dim;
    bool good=ok(cudaMalloc(&l->values,(size_t)(w->capacity+compressed)*w->dim*4),"target KV history") &&
        ok(cudaMalloc(&l->cosines,rb),"target cosines") && ok(cudaMalloc(&l->sines,rb),"target sines") &&
        ok(cudaMemcpy(l->cosines,co,rb,cudaMemcpyHostToDevice),"target cosine upload") &&
        ok(cudaMemcpy(l->sines,si,rb,cudaMemcpyHostToDevice),"target sine upload");
    if (good && w->ratio) good=w->comp_kv&&w->comp_gate&&w->ape&&w->comp_norm&&
        ok(cudaMalloc(&l->projected_kv,(size_t)w->capacity*width*4),"target projection history") &&
        ok(cudaMalloc(&l->projected_gate,(size_t)w->capacity*width*4),"target score history");
    if(good && w->draft)good=!w->ratio && ok(cudaMalloc(&l->past_positions,128*sizeof(int64_t)),"draft past positions");
    if (!good) { dsv4_cuda_target_layer_free(l); return nullptr; } return l;
}
extern "C" void dsv4_cuda_target_layer_free(Dsv4CudaTargetLayer *l) {
    if (!l) return; cudaSetDevice(l->w.qa->device);
    cudaFree(l->values); cudaFree(l->projected_kv); cudaFree(l->projected_gate);
    cudaFree(l->cosines); cudaFree(l->sines); cudaFree(l->past_positions); delete l;
}
extern "C" int dsv4_cuda_target_past(Dsv4CudaTargetLayer *l,const float *past,const int64_t *positions) {
    if(!l || !l->w.draft || !past || !positions)return 0;
    auto *c=ctx(l->w.qa->device);
    return ok(cudaSetDevice(c->id),"draft past device") &&
        ok(cudaMemcpyAsync(l->values,past,128*512*4,cudaMemcpyHostToDevice,c->stream),"draft past values") &&
        ok(cudaMemcpyAsync(l->past_positions,positions,128*sizeof(int64_t),cudaMemcpyHostToDevice,c->stream),"draft past positions");
}
extern "C" int dsv4_cuda_target_layer_forward(Dsv4CudaTargetLayer *l, Dsv4CudaTargetWorkspace *s, int n) {
    if (!l || !s || n<1 || n>s->capacity || s->start>l->w.capacity-n || l->w.qa->device!=s->device) return 0;
    const auto &w=l->w; auto *c=ctx(s->device); auto stream=c->stream;
    if (!dsv4_cuda_mhc_pre_norm_batch(s->residual,w.hc_fn[0],w.hc_scale[0],w.hc_base[0],w.norm[0],n,w.hidden,s->hc,s->normed)) return 0;
    target_matmul(s,w.qa,s->normed->data,s->qa->data,n,stream);
    bf16_round<<<(n*w.rank+255)/256,256,0,stream>>>(s->qa->data,n*w.rank);
    target_norm<<<n,256,0,stream>>>(s->qa->data,(float*)w.qnorm->w,w.rank,w.eps);
    target_matmul(s,w.qb,s->qa->data,s->q->data,n,stream);
    bf16_round<<<(n*w.heads*w.dim+255)/256,256,0,stream>>>(s->q->data,n*w.heads*w.dim);
    target_norm<<<n*w.heads,256,0,stream>>>(s->q->data,nullptr,w.dim,w.eps);
    target_rope<<<n*w.heads,64,0,stream>>>(s->q->data,l->cosines,l->sines,s->control,w.heads,w.dim,w.rope,0);
    target_matmul(s,w.kv,s->normed->data,s->kv->data,n,stream);
    bf16_round<<<(n*w.dim+255)/256,256,0,stream>>>(s->kv->data,n*w.dim);
    target_norm<<<n,256,0,stream>>>(s->kv->data,(float*)w.kvnorm->w,w.dim,w.eps);
    target_rope<<<n,64,0,stream>>>(s->kv->data,l->cosines,l->sines,s->control,1,w.dim,w.rope,0);
    target_cache_store<<<dim3(n,w.dim/64),64,0,stream>>>(l->values,s->kv->data,s->control,w.dim,w.dim-w.rope,w.draft?128:-1);
    if (w.ratio) {
        int width=(w.ratio==4?2:1)*w.dim;
        mm_bf16_batch<<<dim3(width,(n+15)/16),256,0,stream>>>((__nv_bfloat16*)w.comp_kv->w,s->normed->data,s->comp_kv,width,w.hidden,n);
        mm_bf16_batch<<<dim3(width,(n+15)/16),256,0,stream>>>((__nv_bfloat16*)w.comp_gate->w,s->normed->data,s->comp_gate,width,w.hidden,n);
        target_projection_store<<<(n*width+255)/256,256,0,stream>>>(l->projected_kv,s->comp_kv,s->control,width,n);
        target_projection_store<<<(n*width+255)/256,256,0,stream>>>(l->projected_gate,s->comp_gate,s->control,width,n);
        target_compress<<<n,256,0,stream>>>(l->values+(long long)w.capacity*w.dim,l->projected_kv,l->projected_gate,
            (float*)w.ape->w,(float*)w.comp_norm->w,l->cosines,l->sines,s->control,w.ratio,w.dim,w.rope,w.eps);
    }
    if(w.draft) {
        if(n>5)return 0;
        dspark_attention_kernel<<<dim3(w.heads,n),256,0,stream>>>(s->context->data,s->q->data,l->values,
            l->past_positions,(float*)w.sink->w,0,128,n,w.heads,w.dim,s->control);
    } else {
        target_meta<<<1,128,0,stream>>>(s->meta,s->control,n,w.window,w.ratio);
        size_t shared=(w.window+(w.ratio?w.capacity/w.ratio:0))*sizeof(float);
        sparse_attn_batch_kernel<<<dim3(w.heads,n),256,shared,stream>>>(s->context->data,s->q->data,l->values,
            (float*)w.sink->w,s->meta,w.capacity,w.heads,w.dim,1.f/sqrtf((float)w.dim));
    }
    target_rope<<<n*w.heads,64,0,stream>>>(s->context->data,l->cosines,l->sines,s->control,w.heads,w.dim,w.rope,1);
    if (!target_output_projection(s,w,n) ||
        !dsv4_cuda_mhc_post_batch(s->branch,s->residual,s->hc,n,w.hidden,s->middle) ||
        !dsv4_cuda_mhc_pre_norm_batch(s->middle,w.hc_fn[1],w.hc_scale[1],w.hc_base[1],w.norm[1],n,w.hidden,s->hc,s->normed)) return 0;
    if (!resident_route_moe(w.experts,w.gate,w.bias,nullptr,w.routed_scale,w.swiglu_limit,
            s->branch->data,s->normed->data,n,true,s->tokens)) return 0;
    return dsv4_cuda_mhc_post_batch(s->branch,s->middle,s->hc,n,w.hidden,s->residual);
}

/* Generic kernels share scratch with the drafter. Growing any allocation
 * invalidates captured addresses, even if cudaMalloc reuses the same pointer. */
static void target_scratch(Dev *c, size_t *sizes) {
    size_t current[]={c->xcap,c->ycap,c->p1cap,c->p2cap,c->p3cap,c->p4cap,
        c->aux1cap,c->aux2cap,c->aux3cap,c->mvdesccap,c->expert_weightscap,c->expert_idcap};
    memcpy(sizes,current,sizeof(current));
}
static void target_graphs_invalidate(Dsv4CudaTargetWorkspace *s) {
    for (auto &g:s->graphs) {dsv4_cuda_graph_free(g);g=nullptr;}
    memset(s->warmed,0,sizeof(s->warmed));
}
extern "C" int dsv4_cuda_target_stage(Dsv4CudaTargetWorkspace *s, Dsv4CudaTargetLayer *const *layers,
    int count, int first_tap, int rows, int graph) {
    if (!s || !layers || count<1 || rows<1 || rows>s->capacity) return 0;
    auto *c=ctx(s->device); size_t scratch[12]; target_scratch(c,scratch);
    if (memcmp(scratch,s->scratch,sizeof(scratch))) {
        target_graphs_invalidate(s);
    }
#ifdef COLI_DSV4_DEEPGEMM
    graph=0; // This capture contract covers the portable kernels only.
#endif
    graph=graph && rows<=16;
    if (graph && s->graphs[rows]) {
        ++s->replays; return dsv4_cuda_graph_launch(s->graphs[rows]);
    }
    bool capture=graph && s->warmed[rows];
    if (capture && !dsv4_cuda_graph_begin(s->device)) return 0;
    int good=1;
    for (int i=0;i<count && good;i++) {
        good=dsv4_cuda_target_layer_forward(layers[i],s,rows);
        int tap=i-first_tap;
        if (good && tap>=0 && tap<3) good=ok(cudaMemcpyAsync(s->taps+(size_t)tap*rows*16384,
            s->residual->data,(size_t)rows*16384*4,cudaMemcpyDeviceToDevice,c->stream),"target feature tap");
    }
    if (capture) {
        auto *g=dsv4_cuda_graph_end(s->device);
        if (!good || !g) {dsv4_cuda_graph_free(g);return 0;}
        s->graphs[rows]=g; ++s->captures;
        good=dsv4_cuda_graph_launch(g);
    }
    target_scratch(c,s->scratch);
    if (memcmp(scratch,s->scratch,sizeof(scratch))) target_graphs_invalidate(s);
    s->warmed[rows]=good;
    return good;
}
extern "C" int dsv4_cuda_target_taps(Dsv4CudaTargetWorkspace *s, float *output, int rows, int first, int count) {
    if (!s || !output || rows<1 || rows>s->capacity || first<0 || count<1 || first+count>3) return 0;
    auto *c=ctx(s->device);
    return ok(cudaSetDevice(s->device),"target tap device") &&
        ok(cudaMemcpyAsync(output+(size_t)first*rows*16384,s->taps+(size_t)first*rows*16384,(size_t)rows*count*16384*4,cudaMemcpyDeviceToHost,c->stream),"target tap output") &&
        ok(cudaStreamSynchronize(c->stream),"target tap sync");
}

struct Dsv4CudaDraftHead {
    Dsv4CudaTensor *head;
    __nv_bfloat16 *markov1,*markov2;
    float *weights,*confidence,*values;
    int *ids,rank;
};
__global__ void draft_markov(float *scores,const __nv_bfloat16 *embedding,
    const __nv_bfloat16 *weights,const int *ids,int item,int anchor,int vocab,int rank) {
    int token=blockIdx.x*blockDim.x+threadIdx.x;if(token>=vocab)return;
    int previous=item?ids[item-1]:anchor;
    float value=scores[token];
    for(int r=0;r<rank;r++) value=__fadd_rn(value,__fmul_rn(
        __bfloat162float(weights[(long long)r*vocab+token]),
        __bfloat162float(embedding[(long long)previous*rank+r])));
    scores[token]=value;
}
__global__ void draft_confidence(float *output,const float *hidden,const float *weights,
    const __nv_bfloat16 *embedding,const int *ids,int rows,int anchor,int width,int rank) {
    int item=blockIdx.x;float score=0.f;
    for(int d=0;d<width;d++)score=__fadd_rn(score,__fmul_rn(weights[d],hidden[(long long)item*width+d]));
    int previous=item?ids[item-1]:anchor;
    for(int r=0;r<rank;r++)score=__fadd_rn(score,__fmul_rn(weights[width+r],__bfloat162float(embedding[(long long)previous*rank+r])));
    output[item]=score>=0.f?1.f/(1.f+expf(-score)):expf(score)/(1.f+expf(score));
}
extern "C" Dsv4CudaDraftHead *dsv4_cuda_draft_head_create(Dsv4CudaTensor *head,
    const uint16_t *markov1,const uint16_t *markov2,const float *confidence,int rank) {
    if(!head || head->fmt!=12 || !markov1 || !markov2 || !confidence || rank!=256 ||
        !ok(cudaSetDevice(head->device),"draft head device"))return nullptr;
    auto *s=new Dsv4CudaDraftHead{};s->head=head;s->rank=rank;
    size_t count=(size_t)head->O*rank;
    auto *transposed=(uint16_t*)malloc(count*2);
    if(!transposed){delete s;return nullptr;}
    for(int t=0;t<head->O;t++)for(int r=0;r<rank;r++)transposed[(size_t)r*head->O+t]=markov2[(size_t)t*rank+r];
    bool good=ok(cudaMalloc(&s->markov1,count*2),"draft embedding") &&
        ok(cudaMalloc(&s->markov2,count*2),"draft Markov weights") &&
        ok(cudaMalloc(&s->weights,(head->I+rank)*4),"draft confidence weights") &&
        ok(cudaMalloc(&s->ids,5*4),"draft ids") && ok(cudaMalloc(&s->confidence,5*4),"draft confidence") &&
        ok(cudaMalloc(&s->values,5*4),"draft maxima") &&
        ok(cudaMemcpy(s->markov1,markov1,count*2,cudaMemcpyHostToDevice),"draft embedding upload") &&
        ok(cudaMemcpy(s->markov2,transposed,count*2,cudaMemcpyHostToDevice),"draft Markov upload") &&
        ok(cudaMemcpy(s->weights,confidence,(head->I+rank)*4,cudaMemcpyHostToDevice),"draft confidence upload");
    free(transposed);if(!good){dsv4_cuda_draft_head_free(s);return nullptr;}return s;
}
extern "C" void dsv4_cuda_draft_head_free(Dsv4CudaDraftHead *s) {
    if(!s)return;cudaSetDevice(s->head->device);
    cudaFree(s->markov1);cudaFree(s->markov2);cudaFree(s->weights);cudaFree(s->ids);
    cudaFree(s->confidence);cudaFree(s->values);delete s;
}
extern "C" int dsv4_cuda_draft_head(Dsv4CudaDraftHead *s,const float *hidden,int rows,int anchor,int *ids,float *confidence) {
    if(!s || !hidden || !ids || !confidence || rows<1 || rows>5 || anchor<0 || anchor>=s->head->O)return 0;
    auto *h=s->head;auto *c=ctx(h->device);
    if(!ok(cudaSetDevice(h->device),"draft head device") ||
        !buf((void**)&c->dx,&c->xcap,(size_t)rows*h->I*4) ||
        !buf((void**)&c->dy,&c->ycap,(size_t)rows*h->O*4) ||
        !ok(cudaMemcpyAsync(c->dx,hidden,(size_t)rows*h->I*4,cudaMemcpyHostToDevice,c->stream),"draft hidden input"))return 0;
    head_scores_batch_exact<<<dim3((h->O+127)/128,(rows+3)/4),128,0,c->stream>>>((__nv_bfloat16*)h->w,c->dx,c->dy,h->O,h->I,rows);
    for(int item=0;item<rows;item++) {
        draft_markov<<<(h->O+127)/128,128,0,c->stream>>>(c->dy+(size_t)item*h->O,s->markov1,s->markov2,s->ids,item,anchor,h->O,s->rank);
        head_first_argmax<<<1,256,0,c->stream>>>(c->dy+(size_t)item*h->O,h->O,s->ids+item,s->values+item);
    }
    draft_confidence<<<rows,1,0,c->stream>>>(s->confidence,c->dx,s->weights,s->markov1,s->ids,rows,anchor,h->I,s->rank);
    return ok(cudaGetLastError(),"draft head launch") &&
        ok(cudaMemcpyAsync(ids,s->ids,rows*4,cudaMemcpyDeviceToHost,c->stream),"draft ids output") &&
        ok(cudaMemcpyAsync(confidence,s->confidence,rows*4,cudaMemcpyDeviceToHost,c->stream),"draft confidence output") &&
        ok(cudaStreamSynchronize(c->stream),"draft head sync");
}
