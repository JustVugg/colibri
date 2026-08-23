#include "../backend_cuda_dsv4.h"
#include "../dsv4_quant.h"
#include "../dsv4_mhc.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
static float e4(uint8_t b){int s=b>>7,e=(b>>3)&15,m=b&7;float v=!e?std::ldexp((float)m,-9):std::ldexp(1.f+m/8.f,e==15?8:e-7);return s?-v:v;}
static float e8(uint8_t b){return std::ldexp(1.f,b?(int)b-127:-127);}
static float f4(uint8_t q){static float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}
int main(int argc,char**){int devices[6]={0,1,2,3,4,5},dev=0,ndev=argc>1?6:1;if(!dsv4_cuda_init(devices,ndev))return 1;const int O=128,I=128;std::vector<float>x(I),y(O),ref(O);for(int i=0;i<I;i++)x[i]=(i%13-6)*.03125f;
    std::vector<uint8_t>w8(O*I),s8(1);s8[0]=127;for(int i=0;i<O*I;i++){uint8_t v=(uint8_t)((i*17)%240);w8[i]=(v&0x7f)==0x7f?v-1:v;}
    Dsv4CudaTensor*t=nullptr;if(!dsv4_cuda_upload_fp8(&t,w8.data(),s8.data(),O,I,dev)||!dsv4_cuda_matvec(t,y.data(),x.data()))return 2;
    float worst=0;for(int o=0;o<O;o++){float z=0;for(int i=0;i<I;i++)z+=x[i]*e4(w8[o*I+i]);worst=fmaxf(worst,fabsf(z-y[o]));}dsv4_cuda_tensor_free(t);
    std::vector<uint8_t>w4(O*I/2),s4(O*I/32,127);for(int i=0;i<O*I/2;i++)w4[i]=(uint8_t)((i*29)&255);t=nullptr;
    if(!dsv4_cuda_upload_fp4(&t,w4.data(),s4.data(),O,I,dev)||!dsv4_cuda_matvec(t,y.data(),x.data()))return 3;
    for(int o=0;o<O;o++){float z=0;for(int i=0;i<I;i++){uint8_t p=w4[(o*I+i)/2],q=i&1?p>>4:p&15;z+=x[i]*f4(q)*e8(s4[o*(I/32)+i/32]);}worst=fmaxf(worst,fabsf(z-y[o]));}
    dsv4_cuda_tensor_free(t);
    Dsv4CudaTensor *g[2],*u[2],*d[2];float weights[2]={.7f,.3f};std::vector<float>sum(O,0),got(O),gg(O),uu(O),aa(O),part(O);
    for(int e=0;e<2;e++){
        g[e]=u[e]=d[e]=nullptr;
        if(!dsv4_cuda_upload_fp4(&g[e],w4.data(),s4.data(),O,I,dev)||
           !dsv4_cuda_upload_fp4(&u[e],w4.data(),s4.data(),O,I,dev)||
           !dsv4_cuda_upload_fp4(&d[e],w4.data(),s4.data(),O,I,dev))return 4;
        if(!dsv4_cuda_matvec(g[e],gg.data(),x.data())||!dsv4_cuda_matvec(u[e],uu.data(),x.data()))return 5;
        for(int i=0;i<I;i++){float a=dsv4_q_bf16(gg[i]),b=dsv4_q_bf16(uu[i]);aa[i]=dsv4_q_bf16(weights[e]*(a/(1.f+expf(-a)))*b);}
        dsv4_fp8_simulate(aa.data(),I,128);if(!dsv4_cuda_matvec(d[e],part.data(),aa.data()))return 6;
        for(int i=0;i<O;i++)sum[i]+=dsv4_q_bf16(part[i]);
    }
    if(!dsv4_cuda_expert_group(g,u,d,weights,2,1e9f,got.data(),x.data()))return 7;
    for(int i=0;i<O;i++)worst=fmaxf(worst,fabsf(sum[i]-got[i]));
    Dsv4CudaTensor *sg=nullptr,*su=nullptr,*sd=nullptr;std::vector<float>shared(O),combined(O);
    if(!dsv4_cuda_upload_fp8(&sg,w8.data(),s8.data(),O,I,dev)||!dsv4_cuda_upload_fp8(&su,w8.data(),s8.data(),O,I,dev)||
       !dsv4_cuda_upload_fp8(&sd,w8.data(),s8.data(),O,I,dev)||!dsv4_cuda_expert_fp8(sg,su,sd,1e9f,shared.data(),x.data())||
       !dsv4_cuda_moe(g,u,d,weights,2,sg,su,sd,1e9f,combined.data(),x.data()))return 7;
    for(int i=0;i<O;i++){float expect=dsv4_q_bf16(got[i]+dsv4_q_bf16(shared[i]));worst=fmaxf(worst,fabsf(expect-combined[i]));}
    dsv4_cuda_tensor_free(sg);dsv4_cuda_tensor_free(su);dsv4_cuda_tensor_free(sd);
    for(int e=0;e<2;e++){dsv4_cuda_tensor_free(g[e]);dsv4_cuda_tensor_free(u[e]);dsv4_cuda_tensor_free(d[e]);}
    Dsv4CudaActivation *a[6]={};std::vector<float>copy(I);for(int d=0;d<ndev;d++)if(!(a[d]=dsv4_cuda_activation_create(devices[d],I)))return 8;
    if(dsv4_cuda_activation_device(a[0])!=dev||!dsv4_cuda_activation_upload(a[0],x.data(),I))return 8;
    for(int d=1;d<ndev;d++)if(!dsv4_cuda_activation_copy(a[d],a[d-1],I))return 8;
    if(!dsv4_cuda_activation_download(copy.data(),a[ndev-1],I)||!dsv4_cuda_activation_sync(a[ndev-1]))return 8;
    for(int i=0;i<I;i++)worst=fmaxf(worst,fabsf(copy[i]-x[i]));for(int d=0;d<ndev;d++)dsv4_cuda_activation_free(a[d]);
    const int M=4,H=128,N=2*M+M*M,MH=M*H;std::vector<float>res(MH),fn(N*MH),scale={.7f,.8f,.9f},base(N),post(M),comb(M*M),input(H,0),cpuout(MH),gpuinput(H),gpuout(MH),op(H);
    for(int i=0;i<MH;i++)res[i]=(i%19-9)*.01f;for(int i=0;i<N*MH;i++)fn[i]=(i%23-11)*.0002f;for(int i=0;i<N;i++)base[i]=(i%7-3)*.01f;for(int i=0;i<H;i++)op[i]=(i%11-5)*.02f;
    dsv4_mhc_pre(res.data(),fn.data(),scale.data(),base.data(),M,H,1e-6f,1e-4f,1e-6f,2.f,4,post.data(),comb.data(),input.data());dsv4_mhc_post(op.data(),res.data(),post.data(),comb.data(),M,H,cpuout.data());
    Dsv4CudaTensor *tf=nullptr,*ts=nullptr,*tb=nullptr;Dsv4CudaActivation *ar=dsv4_cuda_activation_create(dev,MH),*ast=dsv4_cuda_activation_create(dev,M+M*M+M),*ain=dsv4_cuda_activation_create(dev,H),*ax=dsv4_cuda_activation_create(dev,H),*ao=dsv4_cuda_activation_create(dev,MH);
    if(!ar||!ast||!ain||!ax||!ao||!dsv4_cuda_upload_f32(&tf,fn.data(),N,MH,dev)||!dsv4_cuda_upload_f32(&ts,scale.data(),3,1,dev)||!dsv4_cuda_upload_f32(&tb,base.data(),N,1,dev)||!dsv4_cuda_activation_upload(ar,res.data(),MH)||!dsv4_cuda_activation_upload(ax,op.data(),H)||!dsv4_cuda_mhc_pre(ar,tf,ts,tb,M,H,1e-6f,1e-4f,1e-6f,2.f,4,ast,ain)||!dsv4_cuda_mhc_post(ax,ar,ast,M,H,ao)||!dsv4_cuda_activation_download(gpuinput.data(),ain,H)||!dsv4_cuda_activation_download(gpuout.data(),ao,MH)||!dsv4_cuda_activation_sync(ao))return 9;
    for(int i=0;i<H;i++)worst=fmaxf(worst,fabsf(input[i]-gpuinput[i]));for(int i=0;i<MH;i++)worst=fmaxf(worst,fabsf(cpuout[i]-gpuout[i]));dsv4_cuda_tensor_free(tf);dsv4_cuda_tensor_free(ts);dsv4_cuda_tensor_free(tb);dsv4_cuda_activation_free(ar);dsv4_cuda_activation_free(ast);dsv4_cuda_activation_free(ain);dsv4_cuda_activation_free(ax);dsv4_cuda_activation_free(ao);
    const int heads=2,dim=64,groups=2,rp=4,rope=2*rp;std::vector<float>norm(H,1.f),kvnorm(dim,1.f),sinks(heads,-.25f),nx(H),q(H),k(dim),context(H),attref(H),attgot(H);std::vector<uint16_t>qnorm(H,0x3f80);
    double nss=0;for(float v:x)nss+=(double)v*v;float ninv=1.f/sqrtf((float)(nss/H)+1e-6f);for(int i=0;i<H;i++)nx[i]=dsv4_q_bf16(x[i]*ninv);dsv4_fp8_simulate(nx.data(),H,128);
    Dsv4CudaTensor *tan=nullptr,*tqa=nullptr,*tqn=nullptr,*tqb=nullptr,*tkv=nullptr,*tkvn=nullptr,*tsink=nullptr,*twa=nullptr,*twb=nullptr;
    if(!dsv4_cuda_upload_f32(&tan,norm.data(),H,1,dev)||!dsv4_cuda_upload_fp8(&tqa,w8.data(),s8.data(),H,H,dev)||!dsv4_cuda_upload_bf16(&tqn,qnorm.data(),1,H,dev)||!dsv4_cuda_upload_fp8(&tqb,w8.data(),s8.data(),H,H,dev)||!dsv4_cuda_upload_fp8(&tkv,w8.data(),s8.data(),dim,H,dev)||!dsv4_cuda_upload_f32(&tkvn,kvnorm.data(),dim,1,dev)||!dsv4_cuda_upload_f32(&tsink,sinks.data(),heads,1,dev)||!dsv4_cuda_upload_fp8_bf16(&twa,w8.data(),s8.data(),H,dim,dev)||!dsv4_cuda_upload_fp8(&twb,w8.data(),s8.data(),H,H,dev))return 10;
    if(!dsv4_cuda_qkv(tqa,tqn,tqb,tkv,1e-6f,q.data(),k.data(),nx.data()))return 10;for(int h=0;h<heads;h++){double ss=0;for(int d=0;d<dim;d++)ss+=(double)q[h*dim+d]*q[h*dim+d];float inv=1.f/sqrtf((float)(ss/dim)+1e-6f);for(int d=0;d<dim;d++)q[h*dim+d]=dsv4_q_bf16(q[h*dim+d]*inv);}double kss=0;for(float v:k)kss+=(double)v*v;float kinv=1.f/sqrtf((float)(kss/dim)+1e-6f);for(int d=0;d<dim;d++)k[d]=dsv4_q_bf16(k[d]*kinv);dsv4_fp8_simulate(k.data(),dim-rope,64);
    for(int h=0;h<heads;h++){float score=0;for(int d=0;d<dim;d++)score+=q[h*dim+d]*k[d];score/=sqrtf((float)dim);float mx=fmaxf(score,sinks[h]),a=expf(score-mx),den=a+expf(sinks[h]-mx);for(int d=0;d<dim;d++)context[h*dim+d]=dsv4_q_bf16(a*k[d]/den);}if(!dsv4_cuda_wo(twa,twb,groups,attref.data(),context.data()))return 10;
    Dsv4CudaActivation *ai=dsv4_cuda_activation_create(dev,H),*ay=dsv4_cuda_activation_create(dev,H);if(!ai||!ay||!dsv4_cuda_activation_upload(ai,x.data(),H)||!dsv4_cuda_attention_first(ai,tan,tqa,tqn,tqb,tkv,tkvn,tsink,twa,twb,heads,dim,rope,groups,1e-6f,ay)||!dsv4_cuda_activation_download(attgot.data(),ay,H)||!dsv4_cuda_activation_sync(ay))return 10;for(int i=0;i<H;i++)worst=fmaxf(worst,fabsf(attref[i]-attgot[i]));
    float rc[2*rp],rs[2*rp];for(int p=0;p<rp;p++){rc[p]=1;rs[p]=0;float a=.1f*(p+1);rc[rp+p]=cosf(a);rs[rp+p]=sinf(a);}Dsv4CudaKvCache *kc=dsv4_cuda_kv_create(dev,4,dim,2,rp,rc,rs,rc,rs);if(!kc||!dsv4_cuda_decode_state_set(dev,0,0)||!dsv4_cuda_attention_window(ai,tan,tqa,tqn,tqb,tkv,tkvn,tsink,twa,twb,nullptr,nullptr,nullptr,nullptr,0,heads,dim,rope,groups,0,1e-6f,kc,ay)||!dsv4_cuda_activation_download(attgot.data(),ay,H)||!dsv4_cuda_activation_sync(ay))return 11;for(int i=0;i<H;i++)worst=fmaxf(worst,fabsf(attref[i]-attgot[i]));
    std::vector<float>x2(H),nx2(H),q2(H),k2(dim),ctx2(H),ref2(H);for(int i=0;i<H;i++)x2[i]=x[i]+(i%5-2)*.0078125f;double ss2=0;for(float v:x2)ss2+=(double)v*v;float iv2=1.f/sqrtf((float)(ss2/H)+1e-6f);for(int i=0;i<H;i++)nx2[i]=dsv4_q_bf16(x2[i]*iv2);dsv4_fp8_simulate(nx2.data(),H,128);if(!dsv4_cuda_qkv(tqa,tqn,tqb,tkv,1e-6f,q2.data(),k2.data(),nx2.data()))return 11;
    for(int h=0;h<heads;h++){double ss=0;for(int d=0;d<dim;d++)ss+=(double)q2[h*dim+d]*q2[h*dim+d];float inv=1.f/sqrtf((float)(ss/dim)+1e-6f);for(int d=0;d<dim;d++)q2[h*dim+d]=dsv4_q_bf16(q2[h*dim+d]*inv);for(int p=0;p<rp;p++){int o=h*dim+dim-rope+2*p;float u=q2[o],v=q2[o+1];q2[o]=dsv4_q_bf16(u*rc[rp+p]-v*rs[rp+p]);q2[o+1]=dsv4_q_bf16(u*rs[rp+p]+v*rc[rp+p]);}}double ks2=0;for(float v:k2)ks2+=(double)v*v;float ik2=1.f/sqrtf((float)(ks2/dim)+1e-6f);for(int d=0;d<dim;d++)k2[d]=dsv4_q_bf16(k2[d]*ik2);for(int p=0;p<rp;p++){int o=dim-rope+2*p;float u=k2[o],v=k2[o+1];k2[o]=dsv4_q_bf16(u*rc[rp+p]-v*rs[rp+p]);k2[o+1]=dsv4_q_bf16(u*rs[rp+p]+v*rc[rp+p]);}dsv4_fp8_simulate(k2.data(),dim-rope,64);
    for(int h=0;h<heads;h++){float z0=0,z1=0;for(int d=0;d<dim;d++){z0+=q2[h*dim+d]*k[d];z1+=q2[h*dim+d]*k2[d];}z0/=sqrtf((float)dim);z1/=sqrtf((float)dim);float mx=fmaxf(sinks[h],fmaxf(z0,z1)),a0=expf(z0-mx),a1=expf(z1-mx),den=expf(sinks[h]-mx)+a0+a1;for(int d=0;d<dim;d++)ctx2[h*dim+d]=dsv4_q_bf16((a0*k[d]+a1*k2[d])/den);for(int p=0;p<rp;p++){int o=h*dim+dim-rope+2*p;float u=ctx2[o],v=ctx2[o+1];ctx2[o]=dsv4_q_bf16(u*rc[rp+p]+v*rs[rp+p]);ctx2[o+1]=dsv4_q_bf16(-u*rs[rp+p]+v*rc[rp+p]);}}if(!dsv4_cuda_wo(twa,twb,groups,ref2.data(),ctx2.data())||!dsv4_cuda_activation_upload(ai,x2.data(),H)||!dsv4_cuda_decode_state_set(dev,1,1)||!dsv4_cuda_attention_window(ai,tan,tqa,tqn,tqb,tkv,tkvn,tsink,twa,twb,nullptr,nullptr,nullptr,nullptr,0,heads,dim,rope,groups,1,1e-6f,kc,ay)||!dsv4_cuda_activation_download(attgot.data(),ay,H)||!dsv4_cuda_activation_sync(ay))return 11;for(int i=0;i<H;i++)worst=fmaxf(worst,fabsf(ref2[i]-attgot[i]));dsv4_cuda_kv_free(kc);
    Dsv4CudaTensor *at[]={tan,tqa,tqn,tqb,tkv,tkvn,tsink,twa,twb};for(auto*t:at)dsv4_cuda_tensor_free(t);dsv4_cuda_activation_free(ai);dsv4_cuda_activation_free(ay);
    std::vector<float>rg(256*H),rb(256);for(int i=0;i<256*H;i++)rg[i]=(i%31-15)*.0005f;for(int i=0;i<256;i++)rb[i]=(i%13-6)*.002f;Dsv4CudaTensor *trg=nullptr,*trb=nullptr;Dsv4CudaActivation *rax=dsv4_cuda_activation_create(dev,H);int rid[6],expect_id[6],fixed[6]={7,19,31,63,127,255};float rw[6],expect_w[6];
    std::vector<float>rlog(256);if(!rax||!dsv4_cuda_upload_f32(&trg,rg.data(),256,H,dev)||!dsv4_cuda_upload_f32(&trb,rb.data(),256,1,dev)||!dsv4_cuda_matvec(trg,rlog.data(),x.data())||!dsv4_cuda_activation_upload(rax,x.data(),H)||!dsv4_cuda_route(rax,trg,trb,nullptr,1.5f,rid,rw))return 12;
    std::vector<float>rpv(256);for(int e=0;e<256;e++){float v=rlog[e];rpv[e]=sqrtf(log1pf(expf(-fabsf(v)))+fmaxf(v,0.f));}bool used[256]={};for(int k=0;k<6;k++){int best=-1;float bv=-INFINITY;for(int e=0;e<256;e++)if(!used[e]&&rpv[e]+rb[e]>bv){best=e;bv=rpv[e]+rb[e];}expect_id[k]=best;used[best]=true;}float rsum=0;for(int k=0;k<6;k++)rsum+=rpv[expect_id[k]];for(int k=0;k<6;k++){expect_w[k]=rpv[expect_id[k]]/rsum*1.5f;if(rid[k]!=expect_id[k]){fprintf(stderr,"router mismatch k=%d got=%d expected=%d\n",k,rid[k],expect_id[k]);return 12;}worst=fmaxf(worst,fabsf(rw[k]-expect_w[k]));}
    if(!dsv4_cuda_route(rax,trg,nullptr,fixed,1.5f,rid,rw))return 12;rsum=0;for(int k=0;k<6;k++)rsum+=rpv[fixed[k]];for(int k=0;k<6;k++){if(rid[k]!=fixed[k])return 12;worst=fmaxf(worst,fabsf(rw[k]-rpv[fixed[k]]/rsum*1.5f));}dsv4_cuda_tensor_free(trg);dsv4_cuda_tensor_free(trb);dsv4_cuda_activation_free(rax);
    dsv4_cuda_shutdown();printf("dsv4-cuda worst_abs=%g\n",worst);return worst>1e-3f;
}
