#include "../backend_cuda_dsv4.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    int devices[] = {0, 5};
    assert(dsv4_cuda_init(devices, 2));
    constexpr int H = 128, I = 128, E = 256;
    for (int device : devices) {
        Dsv4CudaTensor *g[E] = {}, *u[E] = {}, *d[E] = {};
        std::vector<unsigned char> w(H*I/2), s(H*I/32, 120);
        for (int e = 0; e < E; e++) {
            for (size_t j = 0; j < w.size(); j++) w[j] = (j*29 + e*7) & 255;
            assert(dsv4_cuda_upload_fp4(&g[e],w.data(),s.data(),I,H,device));
            for (auto &v : w) v ^= 0x19;
            assert(dsv4_cuda_upload_fp4(&u[e],w.data(),s.data(),I,H,device));
            for (auto &v : w) v ^= 0x32;
            assert(dsv4_cuda_upload_fp4(&d[e],w.data(),s.data(),H,I,device));
        }
        std::vector<unsigned char> sw(H*I,0x28), ss(1,124);
        Dsv4CudaTensor *sg=nullptr,*su=nullptr,*sd=nullptr,*gate=nullptr,*bias=nullptr;
        assert(dsv4_cuda_upload_fp8(&sg,sw.data(),ss.data(),I,H,device));
        assert(dsv4_cuda_upload_fp8(&su,sw.data(),ss.data(),I,H,device));
        assert(dsv4_cuda_upload_fp8(&sd,sw.data(),ss.data(),H,I,device));
        std::vector<float> router(E*H), bias_values(E), x(H), expected(H), actual(H);
        for (int e=0;e<E;e++) {
            bias_values[e]=(e%11)*.007f;
            for(int h=0;h<H;h++)router[e*H+h]=((e*13+h*7)%31-15)*.003f;
        }
        assert(dsv4_cuda_upload_f32(&gate,router.data(),E,H,device));
        assert(dsv4_cuda_upload_f32(&bias,bias_values.data(),E,1,device));
        std::fill(router.begin(),router.end(),0.f);
        Dsv4CudaTensor *tie_gate=nullptr;
        assert(dsv4_cuda_upload_f32(&tie_gate,router.data(),E,H,device));
        auto *set=dsv4_cuda_expert_set_create(g,u,d,E,sg,su,sd);
        auto *input=dsv4_cuda_activation_create(device,H);
        assert(set && input);
        int fixed[]={255,0,127,6,42,19};
        for(int sample=0;sample<8;sample++) {
            for(int h=0;h<H;h++)x[h]=((h*3+sample*5)%23-11)*.0625f;
            assert(dsv4_cuda_activation_upload(input,x.data(),H));
            int ids[6]; float weights[6];
            const int *forced=sample%2 ? fixed : nullptr;
            auto *test_gate=sample>=4 ? tie_gate : gate;
            auto *test_bias=sample&2 ? nullptr : bias;
            assert(dsv4_cuda_route(input,test_gate,test_bias,forced,1.5f,ids,weights));
            for(int k=1;k<6;k++) for(int j=k;j>0&&ids[j]<ids[j-1];j--) {
                std::swap(ids[j],ids[j-1]); std::swap(weights[j],weights[j-1]);
            }
            Dsv4CudaTensor *gs[6],*us[6],*ds[6];
            for(int k=0;k<6;k++){gs[k]=g[ids[k]];us[k]=u[ids[k]];ds[k]=d[ids[k]];}
            assert(dsv4_cuda_moe(gs,us,ds,weights,6,sg,su,sd,7.f,expected.data(),x.data()));
            assert(dsv4_cuda_resident_route_moe(set,test_gate,test_bias,forced,1.5f,7.f,actual.data(),x.data()));
            bool nonzero=false;
            for(int h=0;h<H;h++) {
                assert(std::isfinite(actual[h]) && actual[h]==expected[h]);
                nonzero |= actual[h]!=0;
            }
            assert(nonzero);
        }
        fixed[0]=256;
        assert(!dsv4_cuda_resident_route_moe(set,gate,bias,fixed,1.5f,7.f,actual.data(),x.data()));
        dsv4_cuda_activation_free(input); dsv4_cuda_expert_set_free(set);
        for(int e=0;e<E;e++){dsv4_cuda_tensor_free(g[e]);dsv4_cuda_tensor_free(u[e]);dsv4_cuda_tensor_free(d[e]);}
        for(auto *t:{sg,su,sd,gate,bias,tie_gate})dsv4_cuda_tensor_free(t);
    }
    dsv4_cuda_shutdown();
    puts("resident route: exact parity on two devices, normal and hash routes");
}
