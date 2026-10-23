/* Device-router kernel oracle (#431 PR-A).
 *
 * Feeds random activations/router weights through pipe_router_logits +
 * pipe_router_select and checks against a CPU reference that replicates
 * moe()'s plain routing path verbatim (sigmoid -> bias-augmented top-K by
 * `choice`, weights from raw `logit`, route-level TOPP truncation, norm_topk,
 * routed_scale). The dot/expf rounding may differ from libm at ~1e-6 rel, so
 * a handful of near-tie index flips across trials is tolerated; the weight
 * math itself must agree to 1e-4 rel on matching selections.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_router_cuda.cu -o tests/test_router_cuda
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#if !defined(__HIPCC__)
#include <cuda_runtime.h>   /* under HIP, backend_gpu_compat.h (via backend_cuda.cu) provides these */
#endif

/* pull in the kernel definitions (same idiom as the CPU tests' #include "../colibri.c") */
#include "../backend_cuda.cu"

static void cpu_ref(const float *x,const float *W,const float *bias,int D,int E,
                    int Ksel,float topp,int norm_topk,float rscale,
                    int *idx,float *w,int *keff){
    float *logit=(float*)malloc(E*sizeof(float)),*choice=(float*)malloc(E*sizeof(float));
    for(int e=0;e<E;e++){ double a=0; const float *r=W+(size_t)e*D;
        for(int i=0;i<D;i++) a+=(double)x[i]*r[i];
        float lg=1.f/(1.f+expf(-(float)a)); logit[e]=lg; choice[e]=lg+bias[e]; }
    for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
            if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
        idx[kk]=best; w[kk]=logit[best]; }
    int Ke=Ksel;
    if(topp>0.f && topp<1.f){
        for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
            while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
        float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
        float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } } }
    if(norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f;
                   for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
    for(int kk=0;kk<Ke;kk++) w[kk]*=rscale;
    *keff=Ke; free(logit); free(choice);
}

int main(void){
    const int D=6144,E=256,K=8,TRIALS=200;
    srand(42);
    float *x,*W,*b; cudaMallocManaged(&x,D*4); cudaMallocManaged(&W,(size_t)E*D*4);
    cudaMallocManaged(&b,E*4);
    float *lg,*ch; char *out;
    cudaMalloc(&lg,E*4); cudaMalloc(&ch,E*4); cudaMalloc(&out,K*8+4);
    int flips=0, bad=0;
    for(int t=0;t<TRIALS;t++){
        float topp = (t%3==1)?0.7f:0.f;
        int   nt   = (t%2);
        float rs   = 1.0f+(t%5)*0.25f;
        for(int i=0;i<D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        for(size_t i=0;i<(size_t)E*D;i++) W[i]=(rand()/(float)RAND_MAX-.5f)*.06f;
        for(int e=0;e<E;e++) b[e]=(rand()/(float)RAND_MAX-.5f)*.02f;
        pipe_router_logits<<<E,128>>>(x,W,b,D,lg,ch);
        pipe_router_select<<<1,1>>>(lg,ch,E,K,topp,nt,rs,out);
        char pack[K*8+4];
        if(cudaMemcpy(pack,out,sizeof(pack),cudaMemcpyDeviceToHost)!=cudaSuccess){
            printf("FAIL cuda\n"); return 1; }
        int gidx[K],gkeff; float gw[K];
        memcpy(gidx,pack,K*4); memcpy(gw,pack+K*4,K*4); memcpy(&gkeff,pack+K*8,4);
        int ridx[K],rkeff; float rw[K];
        cpu_ref(x,W,b,D,E,K,topp,nt,rs,ridx,rw,&rkeff);
        int mism=0; for(int k2=0;k2<K;k2++) if(gidx[k2]!=ridx[k2]) mism++;
        if(mism||gkeff!=rkeff){ flips++; continue; }        /* near-tie flip: counted, tolerated */
        for(int k2=0;k2<gkeff;k2++){
            float ref=rw[k2], d=fabsf(gw[k2]-ref);
            if(d>1e-4f*(fabsf(ref)+1e-6f)+1e-6f){ bad++; break; }
        }
    }
    printf("router oracle: %d trials, %d near-tie flips, %d weight mismatches\n",TRIALS,flips,bad);
    if(flips>4||bad){ printf("FAIL\n"); return 1; }
    printf("OK\n"); return 0;
}
