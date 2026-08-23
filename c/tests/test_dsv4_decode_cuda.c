/* test_dsv4_decode_cuda.c â€” decode-path smoke test for the DeepSeek V4 CUDA
 * kernels, intended to be linked against the RUNTIME LOADER
 * (backend_loader_dsv4.c) to validate the Windows MSVC->MinGW ABI boundary,
 * or against the raw nvcc object on Linux. Only BASE-BUILD kernels are used
 * (the DeepGEMM/FlashInfer batched paths are build-option gated and return 0).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

static float fp8_decode(uint8_t b){
    if(!(b & 0x7f)) return 0.0f;
    int s=b&0x80,e=(b>>3)&0xf,m=b&7;
    float v=ldexpf(1.0f+m/8.0f,e-7);
    return s?-v:v;
}

static float e8m0_decode(uint8_t b){ return ldexpf(1.0f,b-127); }

static float fp4_decode(uint8_t q){ static const float a[8]={0,.5f,1,1.5f,2,3,4,6}; return q&8?-a[q&7]:a[q]; }

static int ref_matvec(float *y,const uint8_t*w,const uint8_t*s,int O,int I,const float*x){
    for(int o=0;o<O;o++){double acc=0;for(int i=0;i<I;i++)acc+=fp8_decode(w[(size_t)o*I+i])*e8m0_decode(s[(o/128)*((I+127)/128)+(i/128)])*x[i];y[o]=(float)acc;}
    return 1;
}

int main(void){
    enum { T=4,H=4096,R=1024,K=512,HEADS=64,Q=HEADS*K,MAX=32,ROPE=32,GROUPS=8 };
    int dev=0;
    if(!dsv4_cuda_init(&dev,1)){fprintf(stderr,"init failed (loader/DLL problem?)\n");return 1;}

    /* --- part 1: fp8 matvec against an exact CPU reference ----------------- */
    {
        enum { O=512,I=4096 };
        uint8_t*w=malloc((size_t)O*I),*s=malloc((size_t)(O/128)*(I/128));
        float*x=malloc((size_t)I*sizeof(float)),*got=malloc((size_t)O*sizeof(float)),*ref=malloc((size_t)O*sizeof(float));
        for(int o=0;o<O;o++)for(int i=0;i<I;i++)w[(size_t)o*I+i]=((o*3+i*5)&1)?0xb8:0x38;
        for(int i=0;i<(O/128)*(I/128);i++)s[i]=127;
        for(int i=0;i<I;i++)x[i]=((i%13)-6)/8.0f;
        Dsv4CudaTensor*t=NULL;
        if(!dsv4_cuda_upload_fp8(&t,w,s,O,I,dev)){fprintf(stderr,"fp8 upload failed\n");return 2;}
        if(!dsv4_cuda_matvec(t,got,x)){fprintf(stderr,"matvec failed\n");return 3;}
        ref_matvec(ref,w,s,O,I,x);
        for(int o=0;o<O;o++)if(fabsf(ref[o]-got[o])>.05f*fmaxf(1,fabsf(ref[o]))){fprintf(stderr,"matvec mismatch row=%d ref=%g got=%g\n",o,ref[o],got[o]);return 4;}
        dsv4_cuda_tensor_free(t);free(ref);free(got);free(x);free(s);free(w);
        puts("decode[1/5] matvec vs CPU: OK");
    }

    /* --- part 2: single-token attention_window ------------------------------ */
    {
        enum { WR=8192 };  /* wa output rows */
        uint8_t*qkvw=malloc((size_t)(R+K)*H),*qbs=malloc((size_t)Q*R),*kvw=malloc((size_t)K*H);
        uint8_t*qkvs=malloc((size_t)((R+K+127)/128)*((H+127)/128)),*kv_s=malloc((size_t)((K+127)/128)*((H+127)/128));
        uint8_t*qb_s=malloc((size_t)((Q+127)/128)*((R+127)/128));
        uint8_t*waw=malloc((size_t)WR*H),*waw_s=malloc((size_t)((WR+127)/128)*((H+127)/128));
        uint8_t*wbw=malloc((size_t)H*WR),*wb_s=malloc((size_t)((H+127)/128)*((WR+127)/128));
        float*one=malloc((size_t)H*sizeof(float)),*kvone=malloc((size_t)K*sizeof(float)),*sinkv=calloc((size_t)HEADS,sizeof(float));
        uint16_t*qnone=malloc((size_t)R*sizeof(uint16_t));
        float*rope_cos=malloc((size_t)MAX*ROPE*sizeof(float)),*rope_sin=calloc((size_t)MAX*ROPE,sizeof(float));
        float*x=malloc((size_t)H*sizeof(float)),*got=malloc((size_t)H*sizeof(float));
        if(!qkvw||!qbs||!kvw||!qkvs||!kv_s||!qb_s||!waw||!waw_s||!wbw||!wb_s||!one||!kvone||!sinkv||!qnone||!rope_cos||!rope_sin||!x||!got)return 5;
        memset(qkvw,0,(size_t)(R+K)*H);memset(kvw,0,(size_t)K*H);memset(qbs,0,(size_t)Q*R);
        for(int i=0;i<R+K;i++)qkvw[(size_t)i*H+i%H]=0x38;
        for(int i=0;i<K;i++)kvw[(size_t)i*H+i%H]=0x38;
        for(int i=0;i<Q;i++)qbs[(size_t)i*R+i%R]=0x38;
        memset(qkvs,127,(size_t)((R+K+127)/128)*((H+127)/128));
        memset(kv_s,127,(size_t)((K+127)/128)*((H+127)/128));
        memset(qb_s,127,(size_t)((Q+127)/128)*((R+127)/128));
        memset(waw,0,(size_t)WR*H);
        for(int i=0;i<WR*H;i+=((size_t)WR*H+1))waw[i]=0x38;
        memset(waw_s,127,(size_t)((WR+127)/128)*((H+127)/128));
        memset(wbw,0,(size_t)H*WR);
        for(int i=0;i<H*WR;i+=((size_t)H*WR+1))wbw[i]=0x38;
        memset(wb_s,127,(size_t)((H+127)/128)*((WR+127)/128));
        for(int i=0;i<H;i++)one[i]=1;
        for(int i=0;i<K;i++)kvone[i]=1;
        for(int i=0;i<R;i++)qnone[i]=0x3f80;
        for(int i=0;i<MAX*ROPE;i++)rope_cos[i]=1;
        for(int i=0;i<H;i++)x[i]=(float)((i%7)-3)/8;

        Dsv4CudaTensor *an=NULL,*qa=NULL,*qn=NULL,*qb=NULL,*kv=NULL,*kvn=NULL,*sink=NULL,*wa=NULL,*wb=NULL;
        if(!dsv4_cuda_upload_f32(&an,one,H,1,dev)||!dsv4_cuda_upload_fp8(&qa,qkvw,qkvs,R+K,H,dev)||
           !dsv4_cuda_upload_bf16(&qn,qnone,R,1,dev)||!dsv4_cuda_upload_fp8(&qb,qbs,qb_s,Q,R,dev)||
           !dsv4_cuda_upload_fp8(&kv,kvw,kv_s,K,H,dev)||!dsv4_cuda_upload_f32(&kvn,kvone,K,1,dev)||
           !dsv4_cuda_upload_f32(&sink,sinkv,HEADS,1,dev)||!dsv4_cuda_upload_fp8_bf16(&wa,waw,waw_s,WR,H,dev)||
           !dsv4_cuda_upload_fp8(&wb,wbw,wb_s,H,WR,dev)){fprintf(stderr,"attention weight upload failed\n");return 6;}
        Dsv4CudaKvCache*cache=dsv4_cuda_kv_create(dev,MAX,K,MAX,ROPE,rope_cos,rope_sin,rope_cos,rope_sin);
        Dsv4CudaActivation*in=dsv4_cuda_activation_create(dev,H),*out=dsv4_cuda_activation_create(dev,H);
        if(!cache||!in||!out){fprintf(stderr,"kv/activation alloc failed\n");return 7;}
        if(!dsv4_cuda_activation_upload(in,x,H)){fprintf(stderr,"input upload failed\n");return 8;}
        for(int t=0;t<T;t++){
            if(!dsv4_cuda_decode_state_set(dev,t,t)||
               !dsv4_cuda_attention_window(in,an,qa,qn,qb,kv,kvn,sink,wa,wb,NULL,NULL,NULL,NULL,0,HEADS,K,ROPE*2,GROUPS,t,1e-6f,cache,out)){
                fprintf(stderr,"attention_window failed at token %d\n",t);return 9;
            }
            if(!dsv4_cuda_activation_download(got,out,H)||!dsv4_cuda_activation_sync(out)){fprintf(stderr,"output download failed\n");return 10;}
            for(int i=0;i<H;i++)if(!isfinite(got[i])){fprintf(stderr,"non-finite attention out at %d:%d\n",t,i);return 11;}
        }
        dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);dsv4_cuda_kv_free(cache);
        dsv4_cuda_tensor_free(wb);dsv4_cuda_tensor_free(wa);dsv4_cuda_tensor_free(sink);dsv4_cuda_tensor_free(kvn);
        dsv4_cuda_tensor_free(kv);dsv4_cuda_tensor_free(qb);dsv4_cuda_tensor_free(qn);dsv4_cuda_tensor_free(qa);dsv4_cuda_tensor_free(an);
        free(got);free(x);free(rope_sin);free(rope_cos);free(qnone);free(sinkv);free(kvone);free(one);
        free(wb_s);free(wbw);free(waw_s);free(waw);free(qb_s);free(qbs);free(kv_s);free(kvw);free(qkvs);free(qkvw);
        puts("decode[2/5] attention_window: OK");
    }

    /* --- part 3: expert bank + routed MoE (single token) -------------------- */
    {
        enum { I=2048,E=256 };
        /* The expert BANK stores FP4 (I*H/2 bytes), but the shared experts are
         * uploaded as full FP8 (I*H bytes). Two different buffers. */
        size_t w1=(size_t)I*H/2,s1=(size_t)I*H/32,w2=(size_t)H*I/2,s2=(size_t)H*I/32;
        size_t sf_w=(size_t)I*H,sf_s=(size_t)((I+127)/128)*((H+127)/128),sd_s=(size_t)((H+127)/128)*((I+127)/128);
        Dsv4CudaTensor*sg=NULL,*su=NULL,*sd=NULL,*gate=NULL;
        uint8_t*w=calloc(w1,1),*s=malloc(s1),*dw=calloc(w2,1),*ds=malloc(s2);
        uint8_t*sfw=calloc(sf_w,1),*sfs=malloc(sf_s),*sdfw=calloc(sf_w,1),*sdfs=malloc(sd_s);
        float*gw_router=calloc((size_t)E*H,sizeof(float));
        if(!w||!s||!dw||!ds||!sfw||!sfs||!sdfw||!sdfs||!gw_router)return 12;
        memset(s,127,s1);memset(ds,127,s2);memset(sfs,127,sf_s);memset(sdfs,127,sd_s);
        if(!dsv4_cuda_upload_fp8(&sg,sfw,sfs,I,H,dev)){fprintf(stderr,"sg upload failed\n");return 13;}
        if(!dsv4_cuda_upload_fp8(&su,sfw,sfs,I,H,dev)){fprintf(stderr,"su upload failed\n");return 13;}
        if(!dsv4_cuda_upload_fp8(&sd,sdfw,sdfs,H,I,dev)){fprintf(stderr,"sd upload failed\n");return 13;}
        if(!dsv4_cuda_upload_f32(&gate,gw_router,E,H,dev)){fprintf(stderr,"gate upload failed\n");return 13;}
        Dsv4CudaExpertSet*bank=dsv4_cuda_expert_bank_create(E,H,I,dev,sg,su,sd);
        if(!bank){fprintf(stderr,"expert bank create failed\n");return 14;}
        for(int e=0;e<E;e++){Dsv4CudaTensor*g=NULL,*u=NULL,*d=NULL;
            if(!dsv4_cuda_expert_bank_upload(bank,e,w,s,w,s,dw,ds,&g,&u,&d)){fprintf(stderr,"bank upload %d failed\n",e);return 15;}
            dsv4_cuda_tensor_free(g);dsv4_cuda_tensor_free(u);dsv4_cuda_tensor_free(d);}
        float*x=malloc((size_t)H*sizeof(float));
        Dsv4CudaActivation*in=dsv4_cuda_activation_create(dev,H),*out=dsv4_cuda_activation_create(dev,H);
        float*got=malloc((size_t)H*sizeof(float));
        if(!x||!in||!out||!got)return 16;
        for(int i=0;i<H;i++)x[i]=(float)((i%11)-5)/16;
        if(!dsv4_cuda_activation_upload(in,x,H)){fprintf(stderr,"moe input upload failed\n");return 17;}
        if(!dsv4_cuda_route_moe(in,gate,NULL,0,1.f,bank,7.f,out)){fprintf(stderr,"route_moe failed\n");return 18;}
        if(!dsv4_cuda_activation_download(got,out,H)||!dsv4_cuda_activation_sync(out)){fprintf(stderr,"moe download failed\n");return 19;}
        for(int i=0;i<H;i++)if(!isfinite(got[i])){fprintf(stderr,"non-finite moe out at %d\n",i);return 20;}
        dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);free(got);free(x);
        dsv4_cuda_expert_set_free(bank);dsv4_cuda_tensor_free(gate);dsv4_cuda_tensor_free(sd);dsv4_cuda_tensor_free(su);dsv4_cuda_tensor_free(sg);
        free(gw_router);free(ds);free(dw);free(s);free(w);free(sdfs);free(sdfw);free(sfs);free(sfw);
        puts("decode[3/5] expert bank + route_moe: OK");
    }

    /* --- part 4: mHC pre/post + rmsnorm ------------------------------------- */
    {
        enum { M=4,N=24,parts=8 };
        long long MH=(long long)M*H;
        Dsv4CudaTensor *fn=NULL,*scale=NULL,*base=NULL,*norm=NULL;
        float*fnv=malloc((size_t)N*MH*sizeof(float)),*scv=malloc(3*sizeof(float)),*bsv=malloc((size_t)N*sizeof(float)),*nmv=malloc((size_t)H*sizeof(float));
        float*x=malloc((size_t)MH*sizeof(float)),*got=malloc((size_t)MH*sizeof(float));
        if(!fnv||!scv||!bsv||!nmv||!x||!got)return 21;
        for(int i=0;i<N*MH;i++)fnv[i]=0.01f;scv[0]=1;scv[1]=1;scv[2]=1;for(int i=0;i<N;i++)bsv[i]=0;for(int i=0;i<H;i++)nmv[i]=1;
        for(int i=0;i<MH;i++)x[i]=(float)((i%5)-2)/4;
        if(!dsv4_cuda_upload_f32(&fn,fnv,N,MH,dev)||!dsv4_cuda_upload_f32(&scale,scv,3,1,dev)||
           !dsv4_cuda_upload_f32(&base,bsv,N,1,dev)||!dsv4_cuda_upload_f32(&norm,nmv,H,1,dev)){fprintf(stderr,"mhc weight upload failed\n");return 22;}
        Dsv4CudaActivation*r=dsv4_cuda_activation_create(dev,MH),*state=dsv4_cuda_activation_create(dev,32),*in=dsv4_cuda_activation_create(dev,H),*out=dsv4_cuda_activation_create(dev,MH);
        if(!r||!state||!in||!out)return 23;
        if(!dsv4_cuda_activation_upload(r,x,MH)||!dsv4_cuda_mhc_pre_norm(r,fn,scale,base,norm,M,H,1e-6f,1e-6f,1e-6f,1.f,1,1e-6f,state,in)||
           !dsv4_cuda_mhc_post(in,r,state,M,H,out)||!dsv4_cuda_activation_download(got,out,MH)||!dsv4_cuda_activation_sync(out)){fprintf(stderr,"mhc run failed\n");return 24;}
        for(int i=0;i<MH;i++)if(!isfinite(got[i])){fprintf(stderr,"non-finite mhc out at %d\n",i);return 25;}
        dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);dsv4_cuda_activation_free(state);dsv4_cuda_activation_free(r);
        dsv4_cuda_tensor_free(norm);dsv4_cuda_tensor_free(base);dsv4_cuda_tensor_free(scale);dsv4_cuda_tensor_free(fn);
        free(got);free(x);free(nmv);free(bsv);free(scv);free(fnv);
        puts("decode[4/5] mHC pre/post + rmsnorm: OK");
    }

    /* --- part 5: router gate f32 upload + tensor_device + dsv4_cuda_route --- */
    {
        enum { E=256, DH=64 };
        float*gw=malloc((size_t)E*DH*sizeof(float)),*bv=calloc((size_t)E,sizeof(float));
        float*x=malloc((size_t)DH*sizeof(float));
        if(!gw||!bv||!x)return 26;
        for(int e=0;e<E;e++)gw[(size_t)e*DH+0]=(float)e+1.0f;   /* expert e wins on const input */
        for(int i=0;i<DH;i++)x[i]= i==0?1.0f:0.0f;
        Dsv4CudaTensor*gate=NULL,*bias=NULL;
        if(!dsv4_cuda_upload_f32(&gate,gw,E,DH,dev)||!dsv4_cuda_upload_f32(&bias,bv,E,1,dev)){fprintf(stderr,"router upload failed\n");return 27;}
        if(dsv4_cuda_tensor_device(gate)!=dev){fprintf(stderr,"tensor_device=%d expected %d\n",dsv4_cuda_tensor_device(gate),dev);return 28;}
        if(dsv4_cuda_tensor_bytes(gate)!=(long long)E*DH*sizeof(float)){fprintf(stderr,"tensor_bytes=%lld\n",dsv4_cuda_tensor_bytes(gate));return 29;}
        Dsv4CudaActivation*act=dsv4_cuda_activation_create(dev,DH);
        if(!act||!dsv4_cuda_activation_upload(act,x,DH)){fprintf(stderr,"router activation failed\n");return 30;}
        int ids[6]={0};float wts[6]={0};
        static const int forced[6]={7,8,9,10,11,12};
        if(!dsv4_cuda_route(act,gate,bias,forced,1.0f,ids,wts)){fprintf(stderr,"route failed\n");return 31;}
        for(int k=0;k<6;k++)if(ids[k]!=forced[k]||!isfinite(wts[k])){fprintf(stderr,"route mismatch k=%d id=%d\n",k,ids[k]);return 32;}
        dsv4_cuda_activation_free(act);dsv4_cuda_tensor_free(bias);dsv4_cuda_tensor_free(gate);
        free(x);free(bv);free(gw);
        puts("decode[5/5] router gate upload + tensor_device + route: OK");
    }

    /* --- part 6: fp4 expert matvec against the engine CPU reference -------- */
    {
        enum { O=512, I=4096 };
        /* ExpertStore layout: packed nibbles (even column in low nibble),
         * per-32-column UE8M0 scale, one scale group per row. Same layout the
         * engine's matmul_mxfp4 reference consumes and the mirror attach
         * uploads. */
        size_t pw=(size_t)O*I/2, ps=(size_t)O*(I/32);
        uint8_t*w=malloc(pw),*s=malloc(ps);
        float*x=malloc((size_t)I*sizeof(float)),*got=malloc((size_t)O*sizeof(float)),*ref=malloc((size_t)O*sizeof(float));
        if(!w||!s||!x||!got||!ref)return 33;
        for(size_t i=0;i<pw;i++)w[i]=(uint8_t)(rand()&0xff);
        for(size_t i=0;i<ps;i++)s[i]=(uint8_t)(120+(rand()%8));  /* e8m0 2^-7..1 */
        for(int i=0;i<I;i++)x[i]=((i%17)-8)/8.0f;
        for(int o=0;o<O;o++){
            const uint8_t*wr=w+(size_t)o*(I/2),*sr=s+(size_t)o*(I/32);
            double acc=0;
            for(int g=0;g<I/32;g++){
                float sc=e8m0_decode(sr[g]),ga=0;
                for(int i=g*32;i<g*32+32;i+=2){
                    uint8_t byte=wr[i>>1];
                    ga+=x[i]*fp4_decode(byte&0xF);
                    ga+=x[i+1]*fp4_decode(byte>>4);
                }
                acc+=(double)ga*sc;
            }
            ref[o]=(float)acc;
        }
        Dsv4CudaTensor*t=NULL;
        if(!dsv4_cuda_upload_fp4(&t,w,s,O,I,dev)){fprintf(stderr,"fp4 upload failed\n");return 34;}
        if(!dsv4_cuda_matvec(t,got,x)){fprintf(stderr,"fp4 matvec failed\n");return 35;}
        for(int o=0;o<O;o++)if(fabsf(ref[o]-got[o])>.05f*fmaxf(1,fabsf(ref[o]))){fprintf(stderr,"fp4 matvec mismatch row=%d ref=%g got=%g\n",o,ref[o],got[o]);return 36;}
        dsv4_cuda_tensor_free(t);free(ref);free(got);free(x);free(s);free(w);
        puts("decode[6/6] fp4 expert matvec vs CPU: OK");
    }

    dsv4_cuda_shutdown();
    puts("dsv4 decode path through loader: OK");
    return 0;
}


