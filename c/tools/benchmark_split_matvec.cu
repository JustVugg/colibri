// Fused row-wise base/residual decode + GLM-sized W4A32 matvec benchmark.
// Build: nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/bench_split_matvec $0
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
} while(0)

static constexpr int THREADS=256;
static constexpr int SCAN=8192; // next power of two above GLM hidden 6144

__device__ __forceinline__ unsigned bits(const uint8_t *p,unsigned bit,unsigned n){
    if(!n)return 0; uint64_t v=0; memcpy(&v,p+(bit>>3),sizeof(v));
    return (v>>(bit&7))&((1u<<n)-1);
}

__global__ void raw_w4(float *y,const float *x,const uint8_t *w,int I,int O){
    int o=blockIdx.x;if(o>=O)return;const uint8_t *row=w+(size_t)o*((I+1)/2);
    float sum=0;for(int i=threadIdx.x;i<I;i+=blockDim.x){
        uint8_t q=row[i>>1];int v=(i&1)?q>>4:q&15;sum+=x[i]*(float)(v-8);
    }
    __shared__ float partial[THREADS];partial[threadIdx.x]=sum;__syncthreads();
    for(int n=THREADS/2;n;n>>=1){if(threadIdx.x<n)partial[threadIdx.x]+=partial[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[o]=partial[0];
}

template<int BASE_BITS>
__global__ void split_w4(float *y,const float *x,const uint8_t *base,
        const uint8_t *residual,const uint32_t *offsets,int I,int O,
        const uint8_t *width,const uint8_t *symbol){
    int o=blockIdx.x;if(o>=O)return;
    extern __shared__ uint8_t storage[];
    uint16_t *prefix=(uint16_t*)storage;
    float *partial=(float*)(storage+SCAN*sizeof(uint16_t));
    size_t base_row=(size_t)o*((I*BASE_BITS+7)/8);
    for(int i=threadIdx.x;i<SCAN;i+=blockDim.x){
        unsigned group=i<I?bits(base+base_row,i*BASE_BITS,BASE_BITS):0;
        prefix[i]=i<I?width[group]:0;
    }
    __syncthreads();
    for(int stride=1;stride<SCAN;stride<<=1){
        int i=(threadIdx.x+1)*stride*2-1;
        for(;i<SCAN;i+=blockDim.x*stride*2)prefix[i]+=prefix[i-stride];
        __syncthreads();
    }
    if(!threadIdx.x)prefix[SCAN-1]=0;__syncthreads();
    for(int stride=SCAN>>1;stride;stride>>=1){
        int i=(threadIdx.x+1)*stride*2-1;
        for(;i<SCAN;i+=blockDim.x*stride*2){
            uint16_t left=prefix[i-stride];prefix[i-stride]=prefix[i];prefix[i]+=left;
        }
        __syncthreads();
    }
    const uint8_t *rr=residual+offsets[o];float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        unsigned group=bits(base+base_row,i*BASE_BITS,BASE_BITS);
        unsigned local=bits(rr,prefix[i],width[group]);
        int value=(int)symbol[group*16+local]-8;
        sum+=x[i]*(float)value;
    }
    partial[threadIdx.x]=sum;__syncthreads();
    for(int n=THREADS/2;n;n>>=1){if(threadIdx.x<n)partial[threadIdx.x]+=partial[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[o]=partial[0];
}

template<int BASE_BITS>
__global__ void split_w4_segmented(float *y,const float *x,const uint8_t *base,
        const uint8_t *residual,const uint32_t *offsets,int I,int O,
        const uint8_t *width,const uint8_t *symbol){
    int o=blockIdx.x;if(o>=O)return;int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    size_t base_row=(size_t)o*((I*BASE_BITS+7)/8);
    int span=(I+blockDim.x-1)/blockDim.x,begin=threadIdx.x*span,end=min(I,begin+span);
    unsigned count=0;
    for(int i=begin;i<end;i++){
        unsigned group=bits(base+base_row,i*BASE_BITS,BASE_BITS);
        count+=width[group];
    }
    unsigned scan=count;
    for(int d=1;d<32;d<<=1){unsigned v=__shfl_up_sync(0xffffffff,scan,d);
        if(lane>=d)scan+=v;}
    __shared__ unsigned warp_sum[8],warp_prefix[8];
    if(lane==31)warp_sum[warp]=scan;__syncthreads();
    if(warp==0){unsigned v=lane<8?warp_sum[lane]:0,ws=v;
        for(int d=1;d<32;d<<=1){unsigned z=__shfl_up_sync(0xffffffff,ws,d);
            if(lane>=d)ws+=z;}
        if(lane<8)warp_prefix[lane]=ws-v;}
    __syncthreads();
    unsigned bit=warp_prefix[warp]+scan-count;
    const uint8_t *rr=residual+offsets[o];float sum=0;
    for(int i=begin;i<end;i++){
        unsigned group=bits(base+base_row,i*BASE_BITS,BASE_BITS),n=width[group];
        unsigned local=bits(rr,bit,n);bit+=n;
        sum+=x[i]*(float)((int)symbol[group*16+local]-8);
    }
    __shared__ float partial[THREADS];partial[threadIdx.x]=sum;__syncthreads();
    for(int n=THREADS/2;n;n>>=1){if(threadIdx.x<n)partial[threadIdx.x]+=partial[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[o]=partial[0];
}

/* Random-access variant for the dominant 2-bit mode. One residual bit is
 * directly addressable for every weight. Only group 3 needs three more bits;
 * each warp owns a separate exception stream, so no global rank/prefix scan is
 * needed. */
__global__ void split_w4_direct2(float *y,const float *x,const uint8_t *base,
        const uint8_t *low,const uint8_t *extra,const uint32_t *row_offset,
        const uint16_t *warp_offset,int I,int O,const uint8_t *symbol){
    int o=blockIdx.x;if(o>=O)return;int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    size_t base_row=(size_t)o*((I*2+7)/8),low_row=(size_t)o*((I+7)/8);
    const uint8_t *rr=extra+row_offset[o];unsigned ebit=warp_offset[(size_t)o*8+warp];
    float sum=0;
    for(int chunk=warp*32;chunk<I;chunk+=8*32){
        int i=chunk+lane;unsigned group=i<I?bits(base+base_row,i*2,2):0;
        unsigned local=i<I?bits(low+low_row,i,1):0;
        unsigned mask=__ballot_sync(0xffffffff,i<I&&group==3);
        if(i<I&&group==3)local|=bits(rr,ebit+3*__popc(mask&((1u<<lane)-1)),3)<<1;
        if(i<I)sum+=x[i]*(float)((int)symbol[group*16+local]-8);
        ebit+=3*__popc(mask);
    }
    for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
    __shared__ float wp[8];if(!lane)wp[warp]=sum;__syncthreads();
    if(threadIdx.x<8){float v=wp[threadIdx.x];for(int d=4;d;d>>=1)
        v+=__shfl_down_sync(0xff,v,d);if(!threadIdx.x)y[o]=v;}
}

static void put(std::vector<uint8_t>& dst,size_t bit,unsigned n,unsigned value){
    for(unsigned i=0;i<n;i++)if(value&(1u<<i))dst[(bit+i)>>3]|=1u<<((bit+i)&7);
}

template<int BASE_BITS>
static void bench(int I,int O,const std::vector<uint8_t>& q,const std::vector<float>& x,
                  const uint8_t *map_group,const uint8_t *map_local,
                  const std::vector<uint8_t>& widths,const std::vector<uint8_t>& table){
    size_t base_rb=(I*BASE_BITS+7)/8;
    std::vector<uint8_t> base((size_t)O*base_rb);
    std::vector<uint8_t> residual;std::vector<uint32_t> offsets(O+1);
    residual.reserve(q.size()/2);
    for(int o=0;o<O;o++){
        offsets[o]=residual.size();size_t bit=0;std::vector<uint8_t> row((size_t)I/2+8);
        for(int i=0;i<I;i++){
            uint8_t v=q[(size_t)o*I+i],g=map_group[v];
            put(base,(size_t)o*base_rb*8+(size_t)i*BASE_BITS,BASE_BITS,g);
            put(row,bit,widths[g],map_local[v]);bit+=widths[g];
        }
        residual.insert(residual.end(),row.begin(),row.begin()+(bit+7)/8);
    }
    offsets[O]=residual.size();residual.resize(residual.size()+8);

    uint8_t *dq,*db,*dr,*dw,*dt;float *dx,*yr,*ys,*yf;uint32_t *dof;
    CUDA_OK(cudaMalloc(&dq,q.size()/2));CUDA_OK(cudaMalloc(&db,base.size()));
    CUDA_OK(cudaMalloc(&dr,residual.size()));CUDA_OK(cudaMalloc(&dof,offsets.size()*4));
    CUDA_OK(cudaMalloc(&dw,widths.size()));CUDA_OK(cudaMalloc(&dt,table.size()));
    CUDA_OK(cudaMalloc(&dx,x.size()*4));CUDA_OK(cudaMalloc(&yr,O*4));CUDA_OK(cudaMalloc(&ys,O*4));CUDA_OK(cudaMalloc(&yf,O*4));
    std::vector<uint8_t> packed(q.size()/2);
    for(size_t i=0;i<q.size();i+=2)packed[i/2]=q[i]|(q[i+1]<<4);
    CUDA_OK(cudaMemcpy(dq,packed.data(),packed.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db,base.data(),base.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dr,residual.data(),residual.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dof,offsets.data(),offsets.size()*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dw,widths.data(),widths.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dt,table.data(),table.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    size_t smem=SCAN*sizeof(uint16_t)+THREADS*sizeof(float);
    raw_w4<<<O,THREADS>>>(yr,dx,dq,I,O);
    split_w4<BASE_BITS><<<O,THREADS,smem>>>(ys,dx,db,dr,dof,I,O,dw,dt);
    split_w4_segmented<BASE_BITS><<<O,THREADS>>>(yf,dx,db,dr,dof,I,O,dw,dt);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> a(O),b(O),f(O);CUDA_OK(cudaMemcpy(a.data(),yr,O*4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(b.data(),ys,O*4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(f.data(),yf,O*4,cudaMemcpyDeviceToHost));
    float max_abs=0,max_rel=0,fast_abs=0,fast_rel=0;for(int i=0;i<O;i++){float d=fabsf(a[i]-b[i]),z=fabsf(a[i]-f[i]);
        max_abs=fmaxf(max_abs,d);max_rel=fmaxf(max_rel,d/fmaxf(1.f,fabsf(a[i])));
        fast_abs=fmaxf(fast_abs,z);fast_rel=fmaxf(fast_rel,z/fmaxf(1.f,fabsf(a[i])));}

    cudaEvent_t start,stop;CUDA_OK(cudaEventCreate(&start));CUDA_OK(cudaEventCreate(&stop));
    constexpr int R=200;float raw_ms,split_ms,fast_ms,pipe_ms;
    CUDA_OK(cudaEventRecord(start));for(int r=0;r<R;r++)raw_w4<<<O,THREADS>>>(yr,dx,dq,I,O);
    CUDA_OK(cudaEventRecord(stop));CUDA_OK(cudaEventSynchronize(stop));CUDA_OK(cudaEventElapsedTime(&raw_ms,start,stop));
    CUDA_OK(cudaEventRecord(start));for(int r=0;r<R;r++)split_w4<BASE_BITS><<<O,THREADS,smem>>>(ys,dx,db,dr,dof,I,O,dw,dt);
    CUDA_OK(cudaEventRecord(stop));CUDA_OK(cudaEventSynchronize(stop));CUDA_OK(cudaEventElapsedTime(&split_ms,start,stop));
    CUDA_OK(cudaEventRecord(start));for(int r=0;r<R;r++)split_w4_segmented<BASE_BITS><<<O,THREADS>>>(yf,dx,db,dr,dof,I,O,dw,dt);
    CUDA_OK(cudaEventRecord(stop));CUDA_OK(cudaEventSynchronize(stop));CUDA_OK(cudaEventElapsedTime(&fast_ms,start,stop));
    CUDA_OK(cudaHostRegister(residual.data(),residual.size(),cudaHostRegisterDefault));
    CUDA_OK(cudaEventRecord(start));for(int r=0;r<R;r++){CUDA_OK(cudaMemcpyAsync(dr,residual.data(),residual.size(),cudaMemcpyHostToDevice));
        split_w4_segmented<BASE_BITS><<<O,THREADS>>>(yf,dx,db,dr,dof,I,O,dw,dt);}
    CUDA_OK(cudaEventRecord(stop));CUDA_OK(cudaEventSynchronize(stop));CUDA_OK(cudaEventElapsedTime(&pipe_ms,start,stop));
    CUDA_OK(cudaHostUnregister(residual.data()));
    printf("%dx%d base%d: residual %.3f bits/w | scan exact abs %.3g rel %.3g"
        " | segmented abs %.3g rel %.3g | raw %.3f ms | scan %.3f ms (%.2fx)"
        " | segmented %.3f ms (%.2fx) | H2D+segmented %.3f ms (%.2fx)\n",
        O,I,BASE_BITS,(residual.size()-8)*8.0/q.size(),max_abs,max_rel,fast_abs,fast_rel,
        raw_ms/R,split_ms/R,raw_ms/split_ms,fast_ms/R,raw_ms/fast_ms,pipe_ms/R,raw_ms/pipe_ms);
    cudaFree(dq);cudaFree(db);cudaFree(dr);cudaFree(dof);cudaFree(dw);cudaFree(dt);
    cudaFree(dx);cudaFree(yr);cudaFree(ys);cudaFree(yf);
}

static void bench_direct2(int I,int O,const std::vector<uint8_t>& q,
        const std::vector<float>& x,const uint8_t *group,const uint8_t *local,
        const std::vector<uint8_t>& table){
    size_t br=(I*2+7)/8,lr=(I+7)/8;std::vector<uint8_t>base((size_t)O*br),low((size_t)O*lr);
    std::vector<uint8_t>extra;std::vector<uint32_t>row(O+1);std::vector<uint16_t>wo((size_t)O*8);
    for(int o=0;o<O;o++){
        row[o]=extra.size();std::vector<std::vector<uint8_t>> streams(8);
        std::vector<size_t> stream_bits(8);
        for(int warp=0;warp<8;warp++)for(int chunk=warp*32;chunk<I;chunk+=256)
            for(int lane=0;lane<32&&chunk+lane<I;lane++){
                int i=chunk+lane;uint8_t v=q[(size_t)o*I+i],g=group[v],l=local[v];
                put(base,(size_t)o*br*8+(size_t)i*2,2,g);
                put(low,(size_t)o*lr*8+i,1,l&1);
                if(g==3){if(streams[warp].size()<(stream_bits[warp]+10)/8)streams[warp].resize((stream_bits[warp]+10)/8);
                    put(streams[warp],stream_bits[warp],3,l>>1);stream_bits[warp]+=3;}
            }
        size_t bit=0;for(int warp=0;warp<8;warp++){wo[(size_t)o*8+warp]=bit;
            for(size_t j=0;j<stream_bits[warp];j++)if(streams[warp][j>>3]&(1u<<(j&7))){
                if(extra.size()<row[o]+(bit+j+8)/8)extra.resize(row[o]+(bit+j+8)/8);
                extra[row[o]+((bit+j)>>3)]|=1u<<((bit+j)&7);}
            bit+=stream_bits[warp];}
        extra.resize(row[o]+(bit+7)/8);
    }
    row[O]=extra.size();extra.resize(extra.size()+8);
    uint8_t *dq,*db,*dl,*de,*dt;uint32_t*drow;uint16_t*dwo;float*dx,*yr,*yd;
    std::vector<uint8_t>packed(q.size()/2);for(size_t i=0;i<q.size();i+=2)packed[i/2]=q[i]|(q[i+1]<<4);
    CUDA_OK(cudaMalloc(&dq,packed.size()));CUDA_OK(cudaMalloc(&db,base.size()));CUDA_OK(cudaMalloc(&dl,low.size()));
    CUDA_OK(cudaMalloc(&de,extra.size()));CUDA_OK(cudaMalloc(&dt,table.size()));CUDA_OK(cudaMalloc(&drow,row.size()*4));
    CUDA_OK(cudaMalloc(&dwo,wo.size()*2));CUDA_OK(cudaMalloc(&dx,x.size()*4));CUDA_OK(cudaMalloc(&yr,O*4));CUDA_OK(cudaMalloc(&yd,O*4));
    CUDA_OK(cudaMemcpy(dq,packed.data(),packed.size(),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(db,base.data(),base.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dl,low.data(),low.size(),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(de,extra.data(),extra.size(),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dt,table.data(),table.size(),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(drow,row.data(),row.size()*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dwo,wo.data(),wo.size()*2,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    raw_w4<<<O,THREADS>>>(yr,dx,dq,I,O);split_w4_direct2<<<O,THREADS>>>(yd,dx,db,dl,de,drow,dwo,I,O,dt);CUDA_OK(cudaDeviceSynchronize());
    std::vector<float>a(O),b(O);CUDA_OK(cudaMemcpy(a.data(),yr,O*4,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(b.data(),yd,O*4,cudaMemcpyDeviceToHost));
    float ma=0,mr=0;for(int i=0;i<O;i++){float d=fabsf(a[i]-b[i]);ma=fmaxf(ma,d);mr=fmaxf(mr,d/fmaxf(1.f,fabsf(a[i])));}
    cudaEvent_t s,t;cudaEventCreate(&s);cudaEventCreate(&t);constexpr int R=200;float rm,dm,pm;
    cudaEventRecord(s);for(int r=0;r<R;r++)raw_w4<<<O,THREADS>>>(yr,dx,dq,I,O);cudaEventRecord(t);cudaEventSynchronize(t);cudaEventElapsedTime(&rm,s,t);
    cudaEventRecord(s);for(int r=0;r<R;r++)split_w4_direct2<<<O,THREADS>>>(yd,dx,db,dl,de,drow,dwo,I,O,dt);cudaEventRecord(t);cudaEventSynchronize(t);cudaEventElapsedTime(&dm,s,t);
    CUDA_OK(cudaHostRegister(low.data(),low.size(),cudaHostRegisterDefault));CUDA_OK(cudaHostRegister(extra.data(),extra.size(),cudaHostRegisterDefault));
    cudaEventRecord(s);for(int r=0;r<R;r++){cudaMemcpyAsync(dl,low.data(),low.size(),cudaMemcpyHostToDevice);cudaMemcpyAsync(de,extra.data(),extra.size(),cudaMemcpyHostToDevice);
        split_w4_direct2<<<O,THREADS>>>(yd,dx,db,dl,de,drow,dwo,I,O,dt);}cudaEventRecord(t);cudaEventSynchronize(t);cudaEventElapsedTime(&pm,s,t);
    cudaHostUnregister(low.data());cudaHostUnregister(extra.data());
    printf("%dx%d base2-direct: residual %.3f bits/w + metadata %.3f bits/w | abs %.3g rel %.3g | raw %.3f ms | fused %.3f ms (%.2fx) | H2D+fused %.3f ms (%.2fx)\n",
        O,I,(low.size()*8.0+(extra.size()-8)*8.0)/q.size(),(row.size()*4.0+wo.size()*2.0)*8/q.size(),ma,mr,rm/R,dm/R,rm/dm,pm/R,rm/pm);
    cudaFree(dq);cudaFree(db);cudaFree(dl);cudaFree(de);cudaFree(dt);cudaFree(drow);cudaFree(dwo);cudaFree(dx);cudaFree(yr);cudaFree(yd);
}

int main(){
    const double p[16]={0,.0003231,.001374,.005613,.02026,.05401,.1164,.1861,
        .2314,.1861,.1164,.05399,.02025,.005611,.001373,.0003231};
    const uint8_t order[16]={8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0};
    std::mt19937 rng(52);std::discrete_distribution<int> dist(p,p+16);
    std::uniform_real_distribution<float> xf(-1,1);
    for(auto shape: {std::pair<int,int>{6144,2048}, {2048,6144}}){
        int I=shape.first,O=shape.second;std::vector<uint8_t> q((size_t)I*O);
        std::vector<float>x(I);for(auto&v:q)v=dist(rng);for(auto&v:x)v=xf(rng);
        for(int bits_n=1;bits_n<=2;bits_n++){
            std::vector<int> sizes=bits_n==1?std::vector<int>{4,12}:std::vector<int>{1,2,2,11};
            std::vector<uint8_t>w,table(sizes.size()*16);uint8_t group[16]={},local[16]={};int cur=0;
            for(int g=0;g<(int)sizes.size();g++){w.push_back(sizes[g]==1?0:sizes[g]<=2?1:sizes[g]<=4?2:4);
                for(int j=0;j<sizes[g];j++){uint8_t s=order[cur++];group[s]=g;local[s]=j;table[g*16+j]=s;}}
            if(bits_n==1)bench<1>(I,O,q,x,group,local,w,table);else bench<2>(I,O,q,x,group,local,w,table);
            if(bits_n==2)bench_direct2(I,O,q,x,group,local,table);
        }
    }
}
