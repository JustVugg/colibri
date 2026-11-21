/* FP8 expert qualification through the public backend ABI. No new device kernels.
 * Complements the migrated backend suite's FP8 absorb coverage with dense, grouped,
 * issue/take and row invariance, including native Qwen38 D=2560/I=640. The arithmetic
 * host decoder and f32-block/f64-cross-block oracle follow test_fp8_warp_cuda.cu;
 * its 1e-4 relative RMS bound is unchanged. Small/tail shapes always run first. */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL FP8: %s\n", what); std::exit(1); }
}
static uint32_t state = 0xc001d00du;
static uint32_t random_u32() {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static float decode(uint8_t b) {
    int e=(b>>3)&15, m=b&7;
    if(e==15 && m==7) return NAN;
    float v=e ? std::ldexp(1.f+m/8.f,e-7) : std::ldexp(m/8.f,-6);
    return b&128 ? -v : v;
}
static float sample() { return (random_u32()&65535)/32768.f-1.f; }
struct Weight {
    int K,O; std::vector<uint8_t> q; std::vector<float> scale;
    ColiCudaTensor *device=nullptr;
    Weight(int k,int o):K(k),O(o),q((size_t)k*o),scale((size_t)((k+127)/128)*((o+127)/128)) {
        for(auto &b:q) { b=(uint8_t)random_u32(); if((b&127)==127) b&=254; }
        for(auto &s:scale) s=std::ldexp(1.5f+sample()*.5f,-12);
    }
    void upload(int dev) {
        require(coli_cuda_tensor_upload(&device,q.data(),scale.data(),8,K,O,dev),"upload");
    }
    ~Weight() { if(device) coli_cuda_tensor_free(device); }
    Weight(const Weight&)=delete;
    Weight& operator=(const Weight&)=delete;
    void reference(const float *x,float *y) const {
        const int nb=(K+127)/128;
        for(int o=0;o<O;o++) {
            double sum=0;
            for(int b=0;b<nb;b++) {
                float part=0;
                for(int k=b*128;k<std::min(K,(b+1)*128);k++)
                    part+=decode(q[(size_t)o*K+k])*x[k];
                sum+=(double)part*scale[(size_t)(o/128)*nb+b];
            }
            y[o]=(float)sum;
        }
    }
};
static void compare(const char *name,const float *got,const float *want,size_t n) {
    double err=0,ref=0; float maximum=0;
    for(size_t z=0;z<n;z++) {
        require(std::isfinite(got[z])&&std::isfinite(want[z]),"nonfinite result");
        double d=(double)got[z]-want[z];err+=d*d;ref+=(double)want[z]*want[z];
        maximum=std::max(maximum,(float)std::fabs(d));
    }
    double rms=std::sqrt(err/(ref+1e-20));
    std::printf("FP8 %s relative RMS %.9g max_abs %.9g (limit 1e-4)\n",name,rms,maximum);
    require(rms<=1e-4,name);
}
/* the group's second half on the card's finish thread (COLI_SYCL_F8_ASYNC_FINISH): the switch, the host SiLU, a
 * CPU-order kernel and whole 128-blocks, as f8_group_launch decides it */
static bool split_expected(int D,int I) {
    auto is=[](const char *k,const char *v){ const char *e=std::getenv(k); return e && !std::strcmp(e,v); };
    const char *o=std::getenv("COLI_SYCL_F8_CPU_ORDER");
    return is("COLI_SYCL_F8_ASYNC_FINISH","1") && is("COLI_SYCL_F8_HOST_SILU","1") &&
           o && o[0]>='1' && o[0]<='4' && !o[1] && !(D&127) && !(I&127);
}
static void busy_us(int us) {
    const auto end=std::chrono::steady_clock::now()+std::chrono::microseconds(us);
    while(std::chrono::steady_clock::now()<end) {}
}
/* issue/take against the synchronous group in the orders a decode step gives them: the take at once; after the
 * CPU's share (the second half done before the take); after an idle longer than the hand-over's spin (the finish
 * thread asleep when the next group comes) and the next group at once. With the second half on the finish thread,
 * a synchronous group sent while it is out declines (its submissions would land inside the half), and works again
 * after the take; one that comes after the half is done runs, on the same input, and the take still holds the
 * group's bytes. */
static void issue_take_orders(int dev,ColiCudaTensor **dg,ColiCudaTensor **du,ColiCudaTensor **dd,int *rows,
                              const std::vector<float> &x,const std::vector<float> &got,int D,int I) {
    auto one=[&](int before_us,const char *what){
        require(coli_cuda_expert_group_issue(dg,du,dd,rows,2,x.data()),"issue");
        if(before_us>0) busy_us(before_us);
        const float *r=coli_cuda_expert_group_take(dev);
        require(r && !std::memcmp(r,got.data(),got.size()*sizeof(float)),what);
    };
    one(0,"async != sync (the take at once)");
    one(3000,"async != sync (the take after 3 ms on the host)");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    one(0,"async != sync (after an idle of 20 ms)");
    one(0,"async != sync (the next group at once)");
    if(!split_expected(D,I)) return;
    std::vector<float> other(got.size());
    int declined=0;
    for(int k=0;k<20;k++) {
        require(coli_cuda_expert_group_issue(dg,du,dd,rows,2,x.data()),"issue");
        const int took=coli_cuda_expert_group(dg,du,dd,rows,2,other.data(),x.data());
        const float *r=coli_cuda_expert_group_take(dev);
        require(r && !std::memcmp(r,got.data(),got.size()*sizeof(float)),"async != sync (a synchronous group between)");
        if(took) require(!std::memcmp(other.data(),got.data(),got.size()*sizeof(float)),"the group between, after the half");
        declined+=!took;
    }
    require(declined>0,"a synchronous group while the second half is out declines");
    require(coli_cuda_expert_group(dg,du,dd,rows,2,other.data(),x.data()) &&
            !std::memcmp(other.data(),got.data(),got.size()*sizeof(float)),"the synchronous group after the take");
    std::printf("FP8 split D=%d I=%d: issue/take in four orders, a synchronous group between declined %d of 20; bitwise\n",
                D,I,declined);
}
static void shape(int dev,int D,int I) {
    std::printf("FP8 shape START D=%d I=%d\n",D,I);std::fflush(stdout);
    Weight g0(D,I),u0(D,I),d0(I,D),g1(D,I),u1(D,I),d1(I,D);
    Weight *g[2]={&g0,&g1},*u[2]={&u0,&u1},*d[2]={&d0,&d1};
    ColiCudaTensor *dg[2],*du[2],*dd[2]; int rows[2]={1,3}; constexpr int S=4;
    for(int e=0;e<2;e++) {
        g[e]->upload(dev);u[e]->upload(dev);d[e]->upload(dev);
        dg[e]=g[e]->device;du[e]=u[e]->device;dd[e]=d[e]->device;
    }
    std::vector<float>x((size_t)S*D),got((size_t)S*D),want((size_t)S*D),gh(I),uh(I),h(I);
    for(auto &v:x)v=sample();
    int off=0;
    for(int e=0;e<2;e++)for(int row=0;row<rows[e];row++,off++) {
        g[e]->reference(x.data()+(size_t)off*D,gh.data());
        u[e]->reference(x.data()+(size_t)off*D,uh.data());
        for(int i=0;i<I;i++)h[i]=(gh[i]/(1.f+std::exp(-gh[i])))*uh[i];
        d[e]->reference(h.data(),want.data()+(size_t)off*D);
    }
    require(coli_cuda_expert_group(dg,du,dd,rows,2,got.data(),x.data()),"sync group");
    compare("grouped vs CPU",got.data(),want.data(),got.size());
    require(coli_cuda_expert_group_issue(dg,du,dd,rows,2,x.data()),"issue");
    const float *async=coli_cuda_expert_group_take(dev);
    require(async && !std::memcmp(async,got.data(),got.size()*sizeof(float)),"async != sync");
    issue_take_orders(dev,dg,du,dd,rows,x,got,D,I);
    std::vector<float> one(D);int r1=1;off=0;
    for(int e=0;e<2;e++)for(int row=0;row<rows[e];row++,off++) {
        require(coli_cuda_expert_group(dg+e,du+e,dd+e,&r1,1,one.data(),x.data()+(size_t)off*D),"single row");
        require(!std::memcmp(one.data(),got.data()+(size_t)off*D,D*sizeof(float)),"group rows != single rows");
    }
    std::vector<float> dense((size_t)S*I),dense_ref((size_t)S*I);
    require(coli_cuda_matmul(&dg[0],dense.data(),x.data(),nullptr,nullptr,8,S,D,I,dev,0),"dense");
    for(int row=0;row<S;row++)g0.reference(x.data()+(size_t)row*D,dense_ref.data()+(size_t)row*I);
    compare("dense vs CPU",dense.data(),dense_ref.data(),dense.size());
    std::printf("FP8 shape PASS D=%d I=%d; async/sync and row invariance bitwise\n",D,I);
}
int main(int argc,char **argv) {
    int dev[1];int n=sycl_test_devices(argc,argv,dev,1);if(n<0)return 0;
    require(coli_cuda_init(dev,n),"init");
    float lut[256];for(int b=0;b<256;b++)lut[b]=decode((uint8_t)b);
    require(coli_cuda_fp8_set_lut(lut),"publish LUT");
    shape(dev[0],32,8);
    shape(dev[0],128,128);  // full-block CPU-order candidate, tiny before native
    shape(dev[0],128,129);  // odd dense output tail, partial-width down fallback
    shape(dev[0],201,300);
    shape(dev[0],2560,640);
    size_t count=0,bytes=0;coli_cuda_stats(dev[0],&count,&bytes);
    require(count==0 && bytes==0,"tensor accounting leak");
    coli_cuda_shutdown();std::puts("FP8 EXPERT SUITE PASS");return 0;
}
