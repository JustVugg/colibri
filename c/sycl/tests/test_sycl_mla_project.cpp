/* Independent CPU reference for the existing MLA project batch/ragged ABI.
 * Small inputs only: causal rows, ragged growth, shrink and host-pointer
 * replacement exercise the paging lifetime. --host-only needs no GPU. */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int H=2,Q=2,R=2,V=2,K=4,D=5,S=2,T=3;
static constexpr float SCALE=0.5f;
static void reference(float *out,const float *query,const float *latent,
                      const float *rope,int n,const float *w,const float *proj) {
    double ctx[H*V]={};
    for(int h=0;h<H;h++) {
        const float *q=query+h*(Q+R); double qa[K]={},score[T]={},sum=0;
        for(int k=0;k<K;k++) for(int d=0;d<Q;d++)
            qa[k]+=(double)q[d]*w[(h*(Q+V)+d)*K+k];
        for(int t=0;t<n;t++) {
            for(int k=0;k<K;k++)score[t]+=qa[k]*latent[t*K+k];
            for(int d=0;d<R;d++)score[t]+=(double)q[Q+d]*rope[t*R+d];
            score[t]*=SCALE;
        }
        double mx=*std::max_element(score,score+n);
        for(int t=0;t<n;t++){score[t]=std::exp(score[t]-mx);sum+=score[t];}
        for(int v=0;v<V;v++)for(int k=0;k<K;k++) {
            double cl=0;for(int t=0;t<n;t++)cl+=score[t]/sum*latent[t*K+k];
            ctx[h*V+v]+=cl*w[(h*(Q+V)+Q+v)*K+k];
        }
    }
    for(int d=0;d<D;d++) {
        double v=0;for(int i=0;i<H*V;i++)v+=ctx[i]*proj[d*(H*V)+i];
        out[d]=(float)v;
    }
}
static bool equal(const char *name,const float *got,const float *want,int n) {
    for(int i=0;i<n;i++)if(!std::isfinite(got[i])||std::fabs(got[i]-want[i])>1e-4f) {
        std::fprintf(stderr,"%s mismatch %d got %.9g want %.9g\n",name,i,got[i],want[i]);return false;
    }
    return true;
}
int main(int argc,char **argv) {
    float w[H*(Q+V)*K],proj[D*H*V],q[S*H*(Q+R)],lat[2][T*K],rope[2][T*R],altlat[T*K],altrope[T*R];
    for(int i=0;i<(int)(sizeof(w)/sizeof(*w));i++)w[i]=0.15f*std::sin((i+1)*0.19f);
    for(int i=0;i<(int)(sizeof(proj)/sizeof(*proj));i++)proj[i]=0.2f*std::cos((i+1)*0.13f);
    for(int i=0;i<(int)(sizeof(q)/sizeof(*q));i++)q[i]=0.4f*std::sin((i+1)*0.23f);
    for(int s=0;s<S;s++) {
        for(int i=0;i<T*K;i++)lat[s][i]=0.3f*std::sin((i+1+s*3)*0.17f);
        for(int i=0;i<T*R;i++)rope[s][i]=0.2f*std::cos((i+1+s*5)*0.29f);
    }
    for(int i=0;i<T*K;i++)altlat[i]=-lat[0][i]*0.7f;
    for(int i=0;i<T*R;i++)altrope[i]=-rope[0][i]*0.8f;
    /* With one position, attention is independent of q and rope. */
    float a[D],b[D],q2[H*(Q+R)],r2[R];
    reference(a,q,lat[0],rope[0],1,w,proj);
    for(int i=0;i<H*(Q+R);i++)q2[i]=q[i]+1.75f;
    for(int i=0;i<R;i++)r2[i]=rope[0][i]+2.0f;
    reference(b,q2,lat[0],r2,1,w,proj);
    if(!equal("CPU one-position invariant",a,b,D))return 1;
    if(argc==2 && !std::strcmp(argv[1],"--host-only")) {
        std::puts("MLA PROJECT HOST REFERENCE PASS");return 0;
    }
    int devices[1];int ndev=sycl_test_devices(argc,argv,devices,1);
    if(ndev<0)return 0;
    if(ndev!=1||!coli_cuda_init(devices,1))return 2;
    ColiCudaTensor *wt=nullptr,*pt=nullptr;
    bool ok=coli_cuda_tensor_upload_g(&wt,w,nullptr,0,K,H*(Q+V),devices[0],0) &&
            coli_cuda_tensor_upload_g(&pt,proj,nullptr,0,H*V,D,devices[0],0);
    float got[S*D],want[S*D];
    if(ok) {
        ok=coli_cuda_attention_project_batch(wt,pt,got,q,lat[0],rope[0],S,H,Q,R,V,K,T,SCALE);
        for(int s=0;s<S;s++)reference(want+s*D,q+s*H*(Q+R),lat[0],rope[0],T-S+s+1,w,proj);
        ok=ok&&equal("contiguous causal",got,want,S*D);
        if(ok)std::puts("MLA PROJECT BATCH CAUSAL PASS");
    }
    int keys_data[S]={1,2};const void *keys[S]={&keys_data[0],&keys_data[1]};
    const float *ls[S]={lat[0],lat[1]},*rs[S]={rope[0],rope[1]};
    const int steps[3][S]={{1,3},{3,3},{1,2}};
    for(int step=0;step<4&&ok;step++) {
        int lens[S];
        if(step<3){lens[0]=steps[step][0];lens[1]=steps[step][1];}
        else {ls[0]=altlat;rs[0]=altrope;lens[0]=3;lens[1]=2;}
        ok=coli_cuda_attention_project_ragged(wt,pt,got,q,keys,ls,rs,lens,S,H,Q,R,V,K,T,SCALE);
        for(int s=0;s<S;s++)reference(want+s*D,q+s*H*(Q+R),ls[s],rs[s],lens[s],w,proj);
        ok=ok&&equal("ragged cache lifecycle",got,want,S*D);
    }
    if(ok)std::puts("MLA PROJECT RAGGED GROW SHRINK REPLACE PASS");
    if(wt)coli_cuda_tensor_free(wt);if(pt)coli_cuda_tensor_free(pt);coli_cuda_shutdown();
    if(!ok)return 1;
    std::puts("MLA PROJECT SUITE PASS");return 0;
}
