#include "../backend_cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#ifndef COLI_KV_PAGE_TOKENS
#define COLI_KV_PAGE_TOKENS 64
#endif

int main(){
    int dev=0;if(!coli_cuda_init(&dev,1))return 77;
    constexpr int S=3,H=2,Q=2,R=1,V=2,K=128,D=H*V,O=3;
    constexpr int P=COLI_KV_PAGE_TOKENS,T=P+6,B=1024;
    std::vector<float> w(H*(Q+V)*K),p(O*D),q(S*H*(Q+R));
    for(size_t i=0;i<w.size();i++)w[i]=((int)(i%11)-5)*.07f;
    for(size_t i=0;i<p.size();i++)p[i]=((int)(i%7)-3)*.09f;
    for(size_t i=0;i<q.size();i++)q[i]=((int)(i%13)-6)*.05f;
    ColiCudaTensor *tw=nullptr,*tp=nullptr,*tg=nullptr;
    if(!coli_cuda_tensor_upload(&tw,w.data(),nullptr,0,K,H*(Q+V),dev)||
       !coli_cuda_tensor_upload(&tp,p.data(),nullptr,0,D,O,dev))return 1;
    int n[S]={1,P+1,P+6};std::vector<std::vector<float>> l(S),r(S);
    const float *lp[S],*rp[S];
    const void *keys[S];
    for(int s=0;s<S;s++){
        l[s].resize(B*K);r[s].resize(B*R);
        for(size_t i=0;i<l[s].size();i++)l[s][i]=((int)((i+s*3)%9)-4)*.08f;
        for(size_t i=0;i<r[s].size();i++)r[s][i]=((int)((i+s)%5)-2)*.06f;
        lp[s]=l[s].data();rp[s]=r[s].data();keys[s]=&l[s];
    }
    float got[S*O],ref[S*O],warm[S*O];
    int first[S]={1,P-1,P};
    auto t0=std::chrono::steady_clock::now();
    if(!coli_cuda_attention_project_ragged(tw,tp,warm,q.data(),keys,lp,rp,first,
        S,H,Q,R,V,K,T,.2f))return 2;
    auto t1=std::chrono::steady_clock::now();
    if(!coli_cuda_attention_project_ragged(tw,tp,got,q.data(),keys,lp,rp,n,S,H,Q,R,V,K,T,.2f))return 2;
    auto t2=std::chrono::steady_clock::now();
    for(int s=0;s<S;s++)if(!coli_cuda_attention_project_batch(tw,tp,ref+s*O,
        q.data()+s*H*(Q+R),lp[s],rp[s],1,H,Q,R,V,K,n[s],.2f))return 3;
    double e=0,z=0;for(int i=0;i<S*O;i++){
        double d=got[i]-ref[i];e+=d*d;z+=(double)ref[i]*ref[i];
    }
    double rms=std::sqrt(e/(z+1e-30));

    constexpr int ROWS=H*(Q+V),GS=64,NG=(K+GS-1)/GS;
    std::vector<unsigned char> q4((size_t)ROWS*((K+1)/2));
    std::vector<float> scales((size_t)ROWS*NG);
    for(int row=0;row<ROWS;row++){
        for(int g=0;g<NG;g++)scales[(size_t)row*NG+g]=.01f*(1+row+3*g);
        for(int k=0;k<K;k++){
            int value=(row*3+k*5)%15-7,packed=value+8;
            unsigned char &byte=q4[(size_t)row*((K+1)/2)+(k>>1)];
            if(k&1)byte|=(unsigned char)(packed<<4);else byte=(unsigned char)packed;
        }
    }
    if(!coli_cuda_tensor_upload_g(&tg,q4.data(),scales.data(),4,K,ROWS,dev,GS))return 5;
    if(!coli_cuda_attention_project_ragged(tg,tp,got,q.data(),keys,lp,rp,n,
        S,H,Q,R,V,K,T,.2f))return 6;
    for(int s=0;s<S;s++)if(!coli_cuda_attention_project_batch(tg,tp,ref+s*O,
        q.data()+s*H*(Q+R),lp[s],rp[s],1,H,Q,R,V,K,n[s],.2f))return 7;
    double ge=0,gz=0;for(int i=0;i<S*O;i++){
        double d=got[i]-ref[i];ge+=d*d;gz+=(double)ref[i]*ref[i];
    }
    double grms=std::sqrt(ge/(gz+1e-30));
    int bench_n[S]={B,B,B};
    auto b0=std::chrono::steady_clock::now();
    if(!coli_cuda_attention_project_ragged(tw,tp,warm,q.data(),keys,lp,rp,bench_n,
        S,H,Q,R,V,K,B,.2f))return 2;
    auto b1=std::chrono::steady_clock::now();
    auto t3=std::chrono::steady_clock::now();
    for(int i=0;i<100;i++)if(!coli_cuda_attention_project_ragged(
        tw,tp,warm,q.data(),keys,lp,rp,bench_n,S,H,Q,R,V,K,B,.2f))return 2;
    auto t4=std::chrono::steady_clock::now();
    std::printf("ragged_relative_rms=%.9g ragged_g64_relative_rms=%.9g warm_ms=%.3f cross_page_growth_ms=%.3f grow_1024_ms=%.3f steady_1024_ms=%.3f\n",
        rms,grms,
        std::chrono::duration<double,std::milli>(t1-t0).count(),
        std::chrono::duration<double,std::milli>(t2-t1).count(),
        std::chrono::duration<double,std::milli>(b1-b0).count(),
        std::chrono::duration<double,std::milli>(t4-t3).count()/100.0);
    coli_cuda_tensor_free(tw);coli_cuda_tensor_free(tg);coli_cuda_tensor_free(tp);coli_cuda_shutdown();
    return rms<1e-6&&grms<1e-6?0:4;
}
