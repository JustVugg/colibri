#include "../backend_cuda_dsv4.h"
#include "../native_quant.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    int devices[]={0,5};assert(dsv4_cuda_init(devices,2));
    constexpr int heads=64,hidden=4096;
    std::vector<uint16_t> weights(heads*hidden);
    std::vector<float> input(hidden),expected_w(heads),hw(heads);
    for(size_t i=0;i<weights.size();i++)weights[i]=(0x3a00+i%997)|((i%3)?0:0x8000);
    for(int i=0;i<hidden;i++)input[i]=(i%193-96)*.01234567f;
    for(int device:devices) {
        Dsv4CudaTensor *w=nullptr;
        assert(dsv4_cuda_upload_compressor(&w,weights.data(),heads,hidden,device));
        for(int dim:{32,128,512}) {
            std::vector<float> q(heads*dim),expected(heads*dim),scratch(dim);
            std::vector<uint8_t> scales(dim/32);
            for(size_t i=0;i<q.size();i++)q[i]=coli_bf16_round((int(i*19937%100003)-50001)*.00037f);
            // Zero heads exercise the minimum scale and positive-zero tie.
            for(int i=0;i<dim;i++)q[i]=0.f;
            expected=q;
            bool distinguishes_fma=false;
            for(int h=0;h<heads;h++) {
                float *row=expected.data()+h*dim;
                assert(!coli_hadamard_bf16_ref(row,dim));
                assert(!coli_fp4_activation_qdq_ref(scratch.data(),scales.data(),row,dim,32));
                for(int d=0;d<dim;d++)row[d]=coli_bf16_round(scratch[d]);
                float sum=0,fused=0;
                for(int i=0;i<hidden;i++) {
                    float weight=coli_bf16_decode(weights[h*hidden+i]);
                    volatile float product=weight*input[i];
                    sum+=product;
                    fused=std::fma(weight,input[i],fused);
                }
                distinguishes_fma |= sum!=fused;
                expected_w[h]=sum*(1.f/std::sqrt(float(heads*dim)));
            }
            assert(distinguishes_fma);
            assert(dsv4_cuda_indexer_prepare(w,input.data(),q.data(),hw.data(),dim));
            assert(!memcmp(q.data(),expected.data(),q.size()*4));
            assert(!memcmp(hw.data(),expected_w.data(),heads*4));
            printf("device=%d dim=%d query/head-weights exact\n",device,dim);
        }
        assert(!dsv4_cuda_indexer_prepare(w,input.data(),input.data(),hw.data(),127));
        dsv4_cuda_tensor_free(w);
        constexpr int dim=128;
        for(int tokens:{1,3}) for(int count:{1,7,9,129,513}) {
            std::vector<float> q(tokens*heads*dim),keys(count*dim),hws(tokens*heads),a(tokens*count),b(tokens*count);
            std::vector<int> counts(tokens,count);
            if(tokens>1){counts[0]=0;counts[1]=count/2;}
            for(size_t i=0;i<q.size();i++)q[i]=(int(i%37)-18)*.0137f;
            for(size_t i=0;i<keys.size();i++)keys[i]=(int(i%53)-26)*.0231f;
            for(size_t i=0;i<hws.size();i++)hws[i]=(int(i%19)-9)*.037f;
            setenv("DSV4_CUDA_INDEXER_HEADS","0",1);
            assert(dsv4_cuda_indexer_score_batch(device,q.data(),keys.data(),hws.data(),counts.data(),tokens,heads,dim,count,a.data()));
            unsetenv("DSV4_CUDA_INDEXER_HEADS");
            assert(dsv4_cuda_indexer_score_batch(device,q.data(),keys.data(),hws.data(),counts.data(),tokens,heads,dim,count,b.data()));
            assert(!memcmp(a.data(),b.data(),a.size()*4));
        }
        printf("device=%d score exact: batches 1/3, candidates 1/7/9/129/513, masked counts\n",device);
    }
    dsv4_cuda_shutdown();
}
