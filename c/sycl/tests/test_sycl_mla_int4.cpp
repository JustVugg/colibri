/* Native GLM MLA geometry and grouped-int4 format; independent sampled CPU
 * reference and finite check of every output. Runs tiny row counts first. */
#include "../../backend_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static constexpr int H=64,Q=192,R=64,V=256,K=512,D=6144,GS=64;
static constexpr float SCALE=0.125f;
static const int SAMPLE[]={0,31,1023,2047,4096,6143};
static void fill_float(std::vector<float>&v,int mul,int mod,float scale){
    for(size_t i=0;i<v.size();i++)v[i]=((int)((i*(size_t)mul+3)%mod)-mod/2)*scale;
}
static void fill_packed(std::vector<uint8_t>&v,uint32_t seed){
    for(auto &x:v){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x=(uint8_t)seed;}
}
static float weight(const std::vector<uint8_t>&w,const std::vector<float>&sc,int I,int row,int col){
    uint8_t packed=w[(size_t)row*(I/2)+col/2];int v=(col&1)?(packed>>4):(packed&15);
    return (v-8)*sc[(size_t)row*(I/GS)+col/GS];
}
static void reference(float*out,const float*q,const float*lat,const float*rope,int n,
                      const std::vector<uint8_t>&w,const std::vector<float>&ws,
                      const std::vector<uint8_t>&p,const std::vector<float>&ps){
    double result[sizeof(SAMPLE)/sizeof(*SAMPLE)]={};
    for(int h=0;h<H;h++){
        const float *qh=q+(size_t)h*(Q+R);std::vector<double>qa(K,0),cl(K,0),score(n,0);
        for(int d=0;d<Q;d++)for(int k=0;k<K;k++)qa[k]+=(double)qh[d]*weight(w,ws,K,h*(Q+V)+d,k);
        for(int t=0;t<n;t++){
            for(int k=0;k<K;k++)score[t]+=qa[k]*lat[(size_t)t*K+k];
            for(int j=0;j<R;j++)score[t]+=(double)qh[Q+j]*rope[(size_t)t*R+j];
            score[t]*=SCALE;
        }
        double mx=*std::max_element(score.begin(),score.end()),z=0;
        for(double &x:score){x=std::exp(x-mx);z+=x;}
        for(int t=0;t<n;t++)for(int k=0;k<K;k++)cl[k]+=score[t]/z*lat[(size_t)t*K+k];
        for(int v=0;v<V;v++){
            double ctx=0;for(int k=0;k<K;k++)ctx+=cl[k]*weight(w,ws,K,h*(Q+V)+Q+v,k);
            int i=h*V+v;
            for(size_t j=0;j<sizeof(SAMPLE)/sizeof(*SAMPLE);j++)result[j]+=ctx*weight(p,ps,H*V,SAMPLE[j],i);
        }
    }
    for(size_t j=0;j<sizeof(SAMPLE)/sizeof(*SAMPLE);j++)out[j]=(float)result[j];
}
static bool compare_samples(const float*got,const float*want,int n){
    double err=0,norm=0,maxabs=0;
    for(int i=0;i<n;i++){
        if(!std::isfinite(got[i]))return false;
        double diff=got[i]-want[i];err+=diff*diff;norm+=(double)want[i]*want[i];maxabs=std::max(maxabs,std::fabs(diff));
    }
    double rms=std::sqrt(err/(norm+1e-20));std::printf("INT4 sampled relative RMS %.9g maxabs %.9g reference_norm %.9g values %d\n",rms,maxabs,std::sqrt(norm),n);
    return norm>1e-12 && (rms<1e-3 || maxabs<1e-5);
}
int main(int argc,char**argv){
    bool host=argc==2&&!std::strcmp(argv[1],"--host-only");
    if(!host&&(argc!=5||std::strcmp(argv[1],"--device")||std::strcmp(argv[2],"0")||std::strcmp(argv[3],"--case"))){
        std::fprintf(stderr,"usage: %s --device 0 --case 1|4|16|64|338|ragged\n",argv[0]);return 2;
    }
    bool ragged=!host&&!std::strcmp(argv[4],"ragged");int S=host?1:(ragged?2:std::atoi(argv[4]));
    if(!host&&!ragged&&S!=1&&S!=4&&S!=16&&S!=64&&S!=338)return 2;
    int T=host?1:(ragged?5:S+4);std::vector<uint8_t>w((size_t)H*(Q+V)*K/2),p((size_t)D*H*V/2);
    std::vector<float>ws((size_t)H*(Q+V)*(K/GS)),ps((size_t)D*(H*V/GS)),q((size_t)S*H*(Q+R)),lat((size_t)T*K),rope((size_t)T*R),lat2((size_t)T*K),rope2((size_t)T*R);
    fill_packed(w,0x12345678u);fill_packed(p,0x23456789u);
    for(size_t i=0;i<ws.size();i++)ws[i]=0.002f+(i%11)*0.00007f;
    for(size_t i=0;i<ps.size();i++)ps[i]=0.0025f+(i%13)*0.00005f;
    fill_float(q,5,23,0.009f);fill_float(lat,11,31,0.006f);fill_float(rope,7,17,0.01f);fill_float(lat2,13,31,0.006f);fill_float(rope2,5,17,0.01f);
    constexpr int NS=sizeof(SAMPLE)/sizeof(*SAMPLE);
    if(host){
        float a[NS],b[NS];reference(a,q.data(),lat.data(),rope.data(),1,w,ws,p,ps);
        for(float &v:q)v+=1.0f;for(float &v:rope)v-=2.0f;
        reference(b,q.data(),lat.data(),rope.data(),1,w,ws,p,ps);
        if(!compare_samples(a,b,NS))return 1;std::puts("NATIVE INT4 MLA HOST REFERENCE PASS");return 0;
    }
    int dev=0;if(!coli_cuda_init(&dev,1))return 2;ColiCudaTensor *wt=nullptr,*pt=nullptr;
    bool ok=coli_cuda_tensor_upload_g(&wt,w.data(),ws.data(),4,K,H*(Q+V),dev,GS)&&coli_cuda_tensor_upload_g(&pt,p.data(),ps.data(),4,H*V,D,dev,GS);
    std::vector<float>got((size_t)S*D);const float*ls[2]={lat.data(),lat2.data()},*rs[2]={rope.data(),rope2.data()};int ids[2]={1,2},lens[2]={3,5};const void*keys[2]={&ids[0],&ids[1]};
    if(ok&&!ragged)ok=coli_cuda_attention_project_batch(wt,pt,got.data(),q.data(),lat.data(),rope.data(),S,H,Q,R,V,K,T,SCALE);
    else if(ok)ok=coli_cuda_attention_project_ragged(wt,pt,got.data(),q.data(),keys,ls,rs,lens,S,H,Q,R,V,K,T,SCALE);
    if(ok)for(float x:got)if(!std::isfinite(x)){ok=false;break;}
    if(!ok){if(wt)coli_cuda_tensor_free(wt);if(pt)coli_cuda_tensor_free(pt);coli_cuda_shutdown();return 1;}
    std::vector<int>rows;if(S<=4){for(int s=0;s<S;s++)rows.push_back(s);}else rows={0,S/2,S-1};
    std::vector<float>a,b;
    for(int s:rows){
        float want[NS];const float *l=ragged?ls[s]:lat.data(),*r=ragged?rs[s]:rope.data();int n=ragged?lens[s]:T-S+s+1;
        reference(want,q.data()+(size_t)s*H*(Q+R),l,r,n,w,ws,p,ps);
        for(int j=0;j<NS;j++){a.push_back(got[(size_t)s*D+SAMPLE[j]]);b.push_back(want[j]);}
    }
    ok=ok&&compare_samples(a.data(),b.data(),(int)a.size());
    if(wt)coli_cuda_tensor_free(wt);if(pt)coli_cuda_tensor_free(pt);coli_cuda_shutdown();
    if(!ok)return 1;std::printf("NATIVE INT4 MLA %s S=%d H=%d Q=%d R=%d V=%d K=%d D=%d GS=%d PASS\n",ragged?"RAGGED":"CONTIGUOUS",S,H,Q,R,V,K,D,GS);return 0;
}
