/* Includes kernels deliberately: compare their state transitions with the real
 * CPU compressor, including overwriting speculative futures after rollback. */
#include "../backend_cuda_dsv4.cu"
#include "../native_quant.h"
#include <cassert>
#include <cmath>
extern "C" void target_reference(int,int,const float*,const float*,const float*,float*,float*,float*);
static void check(int device,int ratio) {
    constexpr int rows=384,dim=512;
    int width=ratio==4?1024:512;
    std::vector<float> kv(rows*width),gate(rows*width),ape(ratio*width),norm(dim,1.f);
    std::vector<float> co(rows*32),si(rows*32),ref(rows/ratio*dim),actual(ref.size());
    for(int i=0;i<rows*width;i++){kv[i]=sinf(i*.137f)*2.f;gate[i]=cosf(i*.093f)*1.3f;}
    for(int i=0;i<ratio*width;i++)ape[i]=(i%17-8)*.017f;
    assert(cudaSetDevice(device)==cudaSuccess);
    float *dk,*dg,*da,*dn,*dc,*ds,*out;int *pos;
    auto upload=[](float **p,const std::vector<float>&v){assert(cudaMalloc(p,v.size()*4)==cudaSuccess);assert(cudaMemcpy(*p,v.data(),v.size()*4,cudaMemcpyHostToDevice)==cudaSuccess);};
    target_reference(ratio,rows,kv.data(),gate.data(),ape.data(),co.data(),si.data(),ref.data());
    upload(&dk,kv);upload(&dg,gate);upload(&da,ape);upload(&dn,norm);upload(&dc,co);upload(&ds,si);
    assert(cudaMalloc(&out,ref.size()*4)==cudaSuccess);assert(cudaMalloc(&pos,4)==cudaSuccess);
    for(int keep=0;keep<=6;keep++) {
        int start=0;assert(cudaMemcpy(pos,&start,4,cudaMemcpyHostToDevice)==cudaSuccess);
        target_compress<<<rows,256>>>(out,dk,dg,da,dn,dc,ds,pos,ratio,dim,64,1e-6f);
        assert(cudaMemcpy(actual.data(),out,ref.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
        double square=0,base=0;int changed=0;
        for(size_t i=0;i<ref.size();i++){double d=actual[i]-ref[i];square+=d*d;base+=(double)ref[i]*ref[i];changed+=actual[i]!=ref[i];}
        double relative=sqrt(square/base);
        printf("device=%d ratio=%d retained=%d cpu_relative_l2=%.9g changed=%d\n",device,ratio,keep,relative,changed);
        assert(relative<.002);
        // Restore a retained prefix, replace every future projection, then
        // compare suffix execution against both CPU and a fresh GPU replay.
        start=125+keep;
        for(int i=start*width;i<rows*width;i++){kv[i]=cosf(i*.071f+keep);gate[i]=sinf(i*.051f+keep);}
        assert(cudaMemcpy(dk,kv.data(),kv.size()*4,cudaMemcpyHostToDevice)==cudaSuccess);
        assert(cudaMemcpy(dg,gate.data(),gate.size()*4,cudaMemcpyHostToDevice)==cudaSuccess);
        assert(cudaMemcpy(pos,&start,4,cudaMemcpyHostToDevice)==cudaSuccess);
        target_compress<<<rows-start,256>>>(out,dk,dg,da,dn,dc,ds,pos,ratio,dim,64,1e-6f);
        assert(cudaMemcpy(actual.data(),out,actual.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
        std::vector<float> replay(actual.size());start=0;
        assert(cudaMemcpy(pos,&start,4,cudaMemcpyHostToDevice)==cudaSuccess);
        target_compress<<<rows,256>>>(out,dk,dg,da,dn,dc,ds,pos,ratio,dim,64,1e-6f);
        assert(cudaMemcpy(replay.data(),out,replay.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
        assert(!memcmp(actual.data(),replay.data(),actual.size()*4));
        target_reference(ratio,rows,kv.data(),gate.data(),ape.data(),co.data(),si.data(),ref.data());
    }
    for(auto p:{dk,dg,da,dn,dc,ds,out})cudaFree(p);cudaFree(pos);
}
static float fp8_value(uint8_t value) {
    int exponent=(value>>3)&15;float mantissa=(value&7)/8.f;
    float result=exponent?ldexpf(1.f+mantissa,exponent-7):ldexpf(mantissa,-6);
    return value&128?-result:result;
}
static void check_mma(int device,bool split) {
    constexpr int O=256,I=1024,T=49;
    assert(cudaSetDevice(device)==cudaSuccess);
    std::vector<float> input(T*I),actual(T*O),single(T*O),scale(T*I/128);
    std::vector<uint8_t> weights(O*I),scales(O/128*(I/128),127),quantized(T*I);
    for(int i=0;i<T*I;i++)input[i]=ldexpf(sinf(i*.13f),(i/128)%7-3);
    for(size_t i=0;i<scales.size();i++)scales[i]=125+i%5;
    for(int i=0;i<O*I;i++)weights[i]=(0x20+i%40)|(i%3?0:128);
    float *x,*y,*xs;uint8_t *w,*sc,*q;
    assert(cudaMalloc(&x,input.size()*4)==cudaSuccess);assert(cudaMalloc(&y,actual.size()*4)==cudaSuccess);
    assert(cudaMalloc(&w,O*I)==cudaSuccess);assert(cudaMalloc(&sc,scales.size())==cudaSuccess);
    assert(cudaMalloc(&q,input.size())==cudaSuccess);assert(cudaMalloc(&xs,scale.size()*4)==cudaSuccess);
    assert(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(w,weights.data(),O*I,cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(sc,scales.data(),scales.size(),cudaMemcpyHostToDevice)==cudaSuccess);
    target_quantize<<<dim3(I/128,T),128>>>(q,xs,x,I);
    assert(cudaMemcpy(quantized.data(),q,quantized.size(),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(scale.data(),xs,scale.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
    std::vector<float> qdq(T*I);std::vector<uint8_t> encoded(T*I/128);
    for(int t=0;t<T;t++)assert(!coli_fp8_activation_qdq_ref(qdq.data()+t*I,
        encoded.data()+t*(I/128),input.data()+t*I,I,128));
    for(int i=0;i<T*I;i++)assert(qdq[i]==fp8_value(quantized[i])*scale[i/128]);
    if(split)target_fp8_mma<true><<<dim3(O/8,(T+15)/16),256>>>(q,xs,w,sc,y,T,O,I);
    else target_fp8_mma<false><<<dim3(O/64,(T+15)/16),256>>>(q,xs,w,sc,y,T,O,I);
    assert(cudaMemcpy(actual.data(),y,actual.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
    for(int t=0;t<T;t++) {
        if(split)target_fp8_mma<true><<<O/8,256>>>(q+t*I,xs+t*(I/128),w,sc,y+t*O,1,O,I);
        else target_fp8_mma<false><<<O/64,256>>>(q+t*I,xs+t*(I/128),w,sc,y+t*O,1,O,I);
    }
    assert(cudaMemcpy(single.data(),y,single.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(!memcmp(single.data(),actual.data(),actual.size()*4));
    double error=0,base=0;
    for(int t=0;t<T;t++)for(int o=0;o<O;o++) {
        double sum=0;
        for(int i=0;i<I;i++)sum+=(double)fp8_value(quantized[t*I+i])*scale[t*(I/128)+i/128]*fp8_value(weights[o*I+i])*ldexp(1.,(int)scales[(o/128)*(I/128)+i/128]-127);
        double delta=actual[t*O+o]-sum;error+=delta*delta;base+=sum*sum;
    }
    double relative=sqrt(error/base);assert(relative<1e-5);
    printf("device=%d FP8 MMA split=%d scalar/batch exact quantized_oracle_l2=%.9g\n",device,split,relative);
    cudaFree(x);cudaFree(y);cudaFree(w);cudaFree(sc);cudaFree(q);cudaFree(xs);
}
static void check_head(int device) {
    constexpr int V=512,H=128,R=256,T=5;
    std::vector<uint16_t> head(V*H),m1(V*R),m2(V*R);
    std::vector<float> hidden(T*H),weights(H+R),scores(T*V);
    for(int i=0;i<V*H;i++)head[i]=__bfloat16_as_ushort(__float2bfloat16(sinf(i*.07f)*.1f));
    for(int i=0;i<V*R;i++){m1[i]=__bfloat16_as_ushort(__float2bfloat16(sinf(i*.019f)));m2[i]=__bfloat16_as_ushort(__float2bfloat16(cosf(i*.013f)));}
    for(int i=0;i<T*H;i++)hidden[i]=sinf(i*.031f);
    for(int i=0;i<H+R;i++)weights[i]=cosf(i*.09f)*.013f;
    Dsv4CudaTensor *h=nullptr;assert(dsv4_cuda_upload_head_exact(&h,head.data(),V,H,device));
    auto *d=dsv4_cuda_draft_head_create(h,m1.data(),m2.data(),weights.data(),R);assert(d);
    int ids[T];float confidence[T];assert(dsv4_cuda_draft_head(d,hidden.data(),T,17,ids,confidence));
    assert(dsv4_cuda_head_scores_batch_exact(h,hidden.data(),T,scores.data()));
    int previous=17;
    for(int t=0;t<T;t++) {
        int winner=0;float maximum=-INFINITY;
        for(int token=0;token<V;token++) {
            float score=scores[t*V+token];
            for(int r=0;r<R;r++){volatile float product=__bfloat162float(__ushort_as_bfloat16(m2[token*R+r]))*__bfloat162float(__ushort_as_bfloat16(m1[previous*R+r]));score+=product;}
            if(score>maximum){maximum=score;winner=token;}
        }
        assert(ids[t]==winner);float score=0.f;
        for(int i=0;i<H;i++){volatile float product=weights[i]*hidden[t*H+i];score+=product;}
        for(int r=0;r<R;r++){volatile float product=weights[H+r]*__bfloat162float(__ushort_as_bfloat16(m1[previous*R+r]));score+=product;}
        float reference=1.f/(1.f+expf(-score));assert(fabsf(reference-confidence[t])<1e-6f);previous=winner;
    }
    dsv4_cuda_draft_head_free(d);dsv4_cuda_tensor_free(h);
    printf("device=%d Markov sequential ids exact confidence tolerance=1e-6\n",device);
}
static void check_fp4(int device) {
    bool bench=getenv("TARGET_BENCH");int O=bench?2048:128,I=bench?4096:256,count=bench?36:6;
    assert(cudaSetDevice(device)==cudaSuccess);
    int bank=count;
    std::vector<uint8_t> weights((size_t)bank*O*I/2),scales((size_t)bank*O*(I/32),127);
    std::vector<float> input(count*I),old(count*O),fast(count*O);
    for(size_t i=0;i<weights.size();i++)weights[i]=(uint8_t)(i*73+19);
    for(size_t i=0;i<input.size();i++)input[i]=sinf(i*.17f);
    uint8_t *w,*sc;float *x,*y;MvDesc *descriptors;
    assert(cudaMalloc(&w,weights.size())==cudaSuccess);assert(cudaMalloc(&sc,scales.size())==cudaSuccess);
    assert(cudaMalloc(&x,input.size()*4)==cudaSuccess);assert(cudaMalloc(&y,old.size()*4)==cudaSuccess);
    assert(cudaMalloc(&descriptors,count*sizeof(MvDesc))==cudaSuccess);
    assert(cudaMemcpy(w,weights.data(),weights.size(),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(sc,scales.data(),scales.size(),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice)==cudaSuccess);
    std::vector<MvDesc> desc(count);
    for(int i=0;i<count;i++)desc[i]={w+(size_t)(i%bank)*O*I/2,sc+(size_t)(i%bank)*O*(I/32),x+i*I,y+i*O};
    assert(cudaMemcpy(descriptors,desc.data(),count*sizeof(MvDesc),cudaMemcpyHostToDevice)==cudaSuccess);
    cudaEvent_t begin,end;assert(cudaEventCreate(&begin)==cudaSuccess);assert(cudaEventCreate(&end)==cudaSuccess);
    for(int repeat=0;repeat<3;repeat++) {
        mv_fp4_grouped<4><<<dim3((O+3)/4,count),256>>>(descriptors,count,O,I);
        target_fp4_vec<<<dim3((O+7)/8,count),256>>>(descriptors,count,O,I);
    }
    assert(cudaDeviceSynchronize()==cudaSuccess);
    float times[2];
    for(int mode=0;mode<2;mode++) {
        assert(cudaEventRecord(begin)==cudaSuccess);
        for(int repeat=0;repeat<20;repeat++) {
            if(mode)target_fp4_vec<<<dim3((O+7)/8,count),256>>>(descriptors,count,O,I);
            else mv_fp4_grouped<4><<<dim3((O+3)/4,count),256>>>(descriptors,count,O,I);
        }
        assert(cudaEventRecord(end)==cudaSuccess);assert(cudaEventSynchronize(end)==cudaSuccess);
        assert(cudaEventElapsedTime(&times[mode],begin,end)==cudaSuccess);
        assert(cudaMemcpy(mode?fast.data():old.data(),y,old.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess);
    }
    double error=0,base=0;const float values[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
    for(int e=0;e<count;e++)for(int o=0;o<O;o++) {
        double reference=old[e*O+o];
        if(!bench) {
            reference=0;
            for(int i=0;i<I;i++) {
                uint8_t packed=weights[(size_t)(e%bank)*O*I/2+(o*I+i)/2];
                reference+=(double)input[e*I+i]*values[i%2?packed>>4:packed&15];
            }
        }
        double delta=fast[e*O+o]-reference;error+=delta*delta;base+=reference*reference;
    }
    assert(sqrt(error/base)<1e-5);
    printf("device=%d FP4 vector relative_l2=%.9g generic_ms=%.6f vector_ms=%.6f\n",device,sqrt(error/base),times[0]/20,times[1]/20);
    cudaEventDestroy(begin);cudaEventDestroy(end);cudaFree(w);cudaFree(sc);cudaFree(x);cudaFree(y);cudaFree(descriptors);
}
static void check_workspace() {
    setenv("V4_DEVICE_MMA","1",1);
    assert(!dsv4_cuda_target_workspace_create(0,0));assert(!dsv4_cuda_target_workspace_create(0,129));
    auto *a=dsv4_cuda_target_workspace_create(0,6),*b=dsv4_cuda_target_workspace_create(5,6);assert(a && b);
    std::vector<float> x(6*16384),y(x.size());int tokens[6]={1,2,3,4,5,6};
    for(size_t i=0;i<x.size();i++)x[i]=(i%31)*.03125f;
    assert(!dsv4_cuda_target_input(a,x.data(),tokens,-1,1));
    assert(!dsv4_cuda_target_input(a,x.data(),tokens,2043,6));
    assert(!dsv4_cuda_target_input(a,x.data(),tokens,0,7));
    assert(dsv4_cuda_target_input(a,x.data(),tokens,2042,6));
    assert(dsv4_cuda_target_transfer(b,a,6));assert(dsv4_cuda_target_output(b,y.data(),6));
    assert(!memcmp(x.data(),y.data(),x.size()*4));int position=-1,actual[6];
    assert(cudaSetDevice(5)==cudaSuccess);assert(cudaMemcpy(&position,b->control,4,cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(actual,b->tokens,24,cudaMemcpyDeviceToHost)==cudaSuccess);assert(position==2042 && !memcmp(tokens,actual,24));
    dsv4_cuda_target_workspace_free(a);dsv4_cuda_target_workspace_free(b);
    puts("workspace admission, FP8 capability, cross-device data/position/token transfer passed");
}
int main(){int devices[]={0,5};assert(dsv4_cuda_init(devices,2));for(int d:devices){check(d,4);check(d,128);check_mma(d,false);check_mma(d,true);check_head(d);check_fp4(d);}check_workspace();dsv4_cuda_shutdown();}
