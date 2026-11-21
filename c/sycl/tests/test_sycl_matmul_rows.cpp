/* Launch-row splitting: tiny forced boundaries first, then optional GLM shape.
 * Two nonzero input columns give an independent, exactly representable oracle.
 * Legacy vs split must also agree bitwise. No new device kernel. */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" int coli_cuda_tensor_gemm_async(ColiCudaTensor *,float *,const float *,int,void *);
static void require(bool ok,const char *why) {
    if(!ok){std::fprintf(stderr,"FAIL MATMUL ROWS: %s\n",why);std::exit(1);}
}
static int quant(int o,int i){return (o*3+i*5)%15-7;}
static float scale(int fmt,int o,int i) {
    if(fmt==0)return .125f;
    int n=fmt==1?o:fmt==4?o+i/64:o/128+i/128;
    return (n%3+1)*.125f;
}
static uint8_t fp8(int q) {
    const uint8_t code[8]={0,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e};
    return code[std::abs(q)]|(q<0?128:0);
}
static void shape(int dev,int fmt,int S,int I,int O,bool large) {
    std::printf("MATMUL ROWS START fmt=%d S=%d I=%d O=%d\n",fmt,S,I,O);std::fflush(stdout);
    int gs=fmt==4?64:0, ng=fmt==4?(I+63)/64:fmt==8?(I+127)/128:1;
    size_t rb=fmt==0?(size_t)I*4:fmt==4?(I+1)/2:I;
    std::vector<float>wf(fmt==0?(size_t)I*O:0);
    std::vector<uint8_t>w(fmt==0?0:rb*O,0x88);
    std::vector<float>sc(fmt==8?(size_t)((O+127)/128)*ng:(size_t)O*ng);
    for(int o=0;o<O;o++)for(int i=0;i<I;i++){
        int q=quant(o,i);
        if(fmt==0)wf[(size_t)o*I+i]=q*scale(fmt,o,i);
        else if(fmt==4){auto &v=w[(size_t)o*rb+i/2];int sh=(i&1)*4;v=(v&~(15<<sh))|((q+8)<<sh);}
        else w[(size_t)o*rb+i]=fmt==8?fp8(q):(uint8_t)(int8_t)q;
    }
    for(size_t n=0;n<sc.size();n++){
        int o=(int)(n/ng),g=(int)(n%ng);
        sc[n]=scale(fmt,fmt==8?o*128:o,fmt==8?g*128:g*64);
    }
    std::vector<float>x((size_t)S*I,0),want((size_t)S*O),got(want.size()),baseline;
    for(int s=0;s<S;s++){
        int a=(s*7)%I,b=(s*13+3)%I;
        float av=(s%7-3)*.25f,bv=(s%5-2)*.5f;
        x[(size_t)s*I+a]+=av;x[(size_t)s*I+b]+=bv;
        for(int o=0;o<O;o++)want[(size_t)s*O+o]=
            av*quant(o,a)*scale(fmt,o,a)+bv*quant(o,b)*scale(fmt,o,b);
    }
    ColiCudaTensor *t=nullptr;
    const void *weights=fmt==0?(const void*)wf.data():(const void*)w.data();
    require(coli_cuda_tensor_upload_g(&t,weights,fmt?sc.data():nullptr,fmt,I,O,dev,gs),"upload");
    float *dx=(float*)coli_cuda_pipe_alloc(dev,x.size()*4),*dy=(float*)coli_cuda_pipe_alloc(dev,got.size()*4);
    require(dx&&dy,"scratch alloc");
    require(coli_cuda_pipe_upload(dev,dx,x.data(),x.size()*4),"input upload");
    for(const char *cap:large?std::vector<const char*>{"auto"}:std::vector<const char*>{"0","3","1","auto"}){
        setenv("COLI_SYCL_MM_ROWS",cap,1);
        for(int api=0;api<3;api++){
            std::fill(got.begin(),got.end(),NAN);
            if(api==0)require(coli_cuda_matmul(&t,got.data(),x.data(),nullptr,nullptr,fmt,S,I,O,dev,gs),"host matmul");
            else {
                require(coli_cuda_pipe_upload(dev,dy,got.data(),got.size()*4),"poison output");
                if(api==1)require(coli_cuda_tensor_gemm_async(t,dy,dx,S,&coli_sycl_queue(dev)),"async gemm");
                else require(coli_cuda_pipe_gemm(t,dy,dx,S),"pipe gemm");
                require(coli_cuda_pipe_download(dev,dy,got.data(),got.size()*4),"download");
            }
            for(size_t n=0;n<got.size();n++)require(std::isfinite(got[n])&&got[n]==want[n],"exact two-column oracle");
            if(baseline.empty())baseline=got;
            require(!std::memcmp(baseline.data(),got.data(),got.size()*4),"row partition/API byte mismatch");
        }
    }
    coli_cuda_pipe_free(dev,dx);coli_cuda_pipe_free(dev,dy);coli_cuda_tensor_free(t);
    std::printf("MATMUL ROWS PASS fmt=%d S=%d I=%d O=%d: exact oracle, host/async/pipe byte identical\n",fmt,S,I,O);std::fflush(stdout);
}
int main(int argc,char **argv) {
    int dev[1];int n=sycl_test_devices(argc,argv,dev,1);if(n<0)return 0;
    // Force the async generic path; this test isolates launch geometry.
    setenv("COLI_SYCL_XMX","0",1);
    require(coli_cuda_init(dev,n),"init");
    float lut[256];for(int b=0;b<256;b++){
        int e=(b>>3)&15,m=b&7;
        float v=e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(m/8.f,-6);
        lut[b]=(e==15&&m==7)?NAN:(b&128?-v:v);
    }
    require(coli_cuda_fp8_set_lut(lut),"FP8 LUT");
    for(int fmt:{0,1,4,8}){
        shape(dev[0],fmt,7,5,3,false);
        shape(dev[0],fmt,17,33,65,false);
    }
    setenv("COLI_CUDA_F8_WARP","0",1);
    shape(dev[0],8,7,5,3,false);
    if(std::getenv("COLI_SYCL_MM_TEST_NATIVE")){
        shape(dev[0],0,338,512,28672,true);
        shape(dev[0],4,338,512,28672,true);
    }
    size_t count=0,bytes=0;coli_cuda_stats(dev[0],&count,&bytes);
    require(!count&&!bytes,"tensor accounting leak");
    coli_cuda_shutdown();std::puts("MATMUL ROWS SUITE PASS");
}
