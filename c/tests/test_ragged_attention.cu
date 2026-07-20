#include "../backend_cuda.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main(){
    int dev=0;if(!coli_cuda_init(&dev,1))return 77;
    constexpr int S=3,H=2,Q=2,R=1,V=2,K=3,D=H*V,O=3,T=3;
    std::vector<float> w(H*(Q+V)*K),p(O*D),q(S*H*(Q+R));
    for(size_t i=0;i<w.size();i++)w[i]=((int)(i%11)-5)*.07f;
    for(size_t i=0;i<p.size();i++)p[i]=((int)(i%7)-3)*.09f;
    for(size_t i=0;i<q.size();i++)q[i]=((int)(i%13)-6)*.05f;
    ColiCudaTensor *tw=nullptr,*tp=nullptr;
    if(!coli_cuda_tensor_upload(&tw,w.data(),nullptr,0,K,H*(Q+V),dev)||
       !coli_cuda_tensor_upload(&tp,p.data(),nullptr,0,D,O,dev))return 1;
    int n[S]={1,2,3};std::vector<std::vector<float>> l(S),r(S);
    const float *lp[S],*rp[S];
    const void *keys[S];
    for(int s=0;s<S;s++){
        l[s].resize(n[s]*K);r[s].resize(n[s]*R);
        for(size_t i=0;i<l[s].size();i++)l[s][i]=((int)((i+s*3)%9)-4)*.08f;
        for(size_t i=0;i<r[s].size();i++)r[s][i]=((int)((i+s)%5)-2)*.06f;
        lp[s]=l[s].data();rp[s]=r[s].data();keys[s]=&l[s];
    }
    float got[S*O],ref[S*O],warm[S*O];
    int first[S]={1,1,1};
    if(!coli_cuda_attention_project_ragged(tw,tp,warm,q.data(),keys,lp,rp,first,
        S,H,Q,R,V,K,1,.2f))return 2;
    if(!coli_cuda_attention_project_ragged(tw,tp,got,q.data(),keys,lp,rp,n,S,H,Q,R,V,K,T,.2f))return 2;
    for(int s=0;s<S;s++)if(!coli_cuda_attention_project_batch(tw,tp,ref+s*O,
        q.data()+s*H*(Q+R),lp[s],rp[s],1,H,Q,R,V,K,n[s],.2f))return 3;
    double e=0,z=0;for(int i=0;i<S*O;i++){
        double d=got[i]-ref[i];e+=d*d;z+=(double)ref[i]*ref[i];
    }
    double rms=std::sqrt(e/(z+1e-30));std::printf("ragged_relative_rms=%.9g\n",rms);
    coli_cuda_tensor_free(tw);coli_cuda_tensor_free(tp);coli_cuda_shutdown();
    return rms<1e-6?0:4;
}
