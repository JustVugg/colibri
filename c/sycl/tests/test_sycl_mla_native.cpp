/* Native GLM MLA width, tiny row-count ramp, independent CPU reference.
 * D=4 keeps the output projection small; not a full-model shape test. */
#include "../../backend_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static constexpr int H=64,Q=192,R=64,V=256,K=512,D=4;
static constexpr float SCALE=0.125f;
static void fill(std::vector<float>&v,int mul,int mod,float scale){
    for(size_t i=0;i<v.size();i++)v[i]=((int)((i*(size_t)mul+3)%mod)-mod/2)*scale;
}
static void reference(float*out,const float*q,const float*lat,const float*rope,
                      int n,const float*w,const float*proj){
    double result[D]={};
    for(int h=0;h<H;h++){
        const float *qh=q+(size_t)h*(Q+R);const float *wh=w+(size_t)h*(Q+V)*K;
        std::vector<double> qa(K,0),cl(K,0),sc(n,0);
        for(int d=0;d<Q;d++)for(int k=0;k<K;k++)qa[k]+=(double)qh[d]*wh[(size_t)d*K+k];
        for(int t=0;t<n;t++){
            for(int k=0;k<K;k++)sc[t]+=qa[k]*lat[(size_t)t*K+k];
            for(int j=0;j<R;j++)sc[t]+=(double)qh[Q+j]*rope[(size_t)t*R+j];
            sc[t]*=SCALE;
        }
        double mx=*std::max_element(sc.begin(),sc.end()),z=0;
        for(double &x:sc){x=std::exp(x-mx);z+=x;}
        for(int t=0;t<n;t++)for(int k=0;k<K;k++)cl[k]+=sc[t]/z*lat[(size_t)t*K+k];
        for(int v=0;v<V;v++){
            double x=0;const float *wr=wh+(size_t)(Q+v)*K;
            for(int k=0;k<K;k++)x+=cl[k]*wr[k];
            for(int d=0;d<D;d++)result[d]+=x*proj[(size_t)d*H*V+h*V+v];
        }
    }
    for(int d=0;d<D;d++)out[d]=(float)result[d];
}
static bool compare(const float*got,const float*want,int count){
    double err=0,norm=0,maxabs=0;
    for(int i=0;i<count;i++){
        if(!std::isfinite(got[i]))return false;
        double diff=got[i]-want[i];err+=diff*diff;norm+=(double)want[i]*want[i];maxabs=std::max(maxabs,std::fabs(diff));
    }
    double rms=std::sqrt(err/(norm+1e-20));
    std::printf("native MLA relative RMS %.9g maxabs %.9g reference_norm %.9g\n",rms,maxabs,std::sqrt(norm));
    return norm>1e-12 && (rms<5e-4 || maxabs<2e-5);
}
int main(int argc,char**argv){
    if(argc==2&&!std::strcmp(argv[1],"--host-only")){
        std::vector<float>w((size_t)H*(Q+V)*K),p((size_t)D*H*V),q((size_t)H*(Q+R)),l(K),r(R);
        fill(w,7,29,0.002f);fill(p,3,19,0.004f);fill(q,5,23,0.009f);fill(l,11,31,0.006f);fill(r,7,17,0.01f);
        float a[D],b[D];reference(a,q.data(),l.data(),r.data(),1,w.data(),p.data());
        for(float &v:q)v+=1.0f;for(float &v:r)v-=2.0f;
        reference(b,q.data(),l.data(),r.data(),1,w.data(),p.data());
        if(!compare(a,b,D))return 1;std::puts("NATIVE MLA HOST REFERENCE PASS");return 0;
    }
    if(argc!=5||std::strcmp(argv[1],"--device")||std::strcmp(argv[2],"0")||std::strcmp(argv[3],"--case")){
        std::fprintf(stderr,"usage: %s --device 0 --case 1|4|16|ragged\n",argv[0]);return 2;
    }
    bool ragged=!std::strcmp(argv[4],"ragged");int S=ragged?2:std::atoi(argv[4]);
    if(!ragged&&S!=1&&S!=4&&S!=16)return 2;
    int T=ragged?5:S+4;std::vector<float>w((size_t)H*(Q+V)*K),p((size_t)D*H*V),q((size_t)S*H*(Q+R)),lat((size_t)T*K),rope((size_t)T*R),lat2((size_t)T*K),rope2((size_t)T*R),got((size_t)S*D),want((size_t)S*D);
    fill(w,7,29,0.002f);fill(p,3,19,0.004f);fill(q,5,23,0.009f);fill(lat,11,31,0.006f);fill(rope,7,17,0.01f);fill(lat2,13,31,0.006f);fill(rope2,5,17,0.01f);
    int dev=0;if(!coli_cuda_init(&dev,1))return 2;ColiCudaTensor *wt=nullptr,*pt=nullptr;
    bool ok=coli_cuda_tensor_upload_g(&wt,w.data(),nullptr,0,K,H*(Q+V),dev,0)&&coli_cuda_tensor_upload_g(&pt,p.data(),nullptr,0,H*V,D,dev,0);
    if(ok&&!ragged){
        ok=coli_cuda_attention_project_batch(wt,pt,got.data(),q.data(),lat.data(),rope.data(),S,H,Q,R,V,K,T,SCALE);
        for(int s=0;s<S;s++)reference(want.data()+s*D,q.data()+(size_t)s*H*(Q+R),lat.data(),rope.data(),T-S+s+1,w.data(),p.data());
    } else if(ok){
        int ids[2]={1,2},lens[2]={3,5};const void*keys[2]={&ids[0],&ids[1]};const float*ls[2]={lat.data(),lat2.data()},*rs[2]={rope.data(),rope2.data()};
        ok=coli_cuda_attention_project_ragged(wt,pt,got.data(),q.data(),keys,ls,rs,lens,S,H,Q,R,V,K,T,SCALE);
        for(int s=0;s<S;s++)reference(want.data()+s*D,q.data()+(size_t)s*H*(Q+R),ls[s],rs[s],lens[s],w.data(),p.data());
    }
    ok=ok&&compare(got.data(),want.data(),S*D);
    if(wt)coli_cuda_tensor_free(wt);if(pt)coli_cuda_tensor_free(pt);coli_cuda_shutdown();
    if(!ok)return 1;std::printf("NATIVE MLA %s S=%d H=%d Q=%d R=%d V=%d K=%d D=%d PASS\n",ragged?"RAGGED":"CONTIGUOUS",S,H,Q,R,V,K,D);return 0;
}
