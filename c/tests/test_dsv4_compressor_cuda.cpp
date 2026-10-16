#include "../backend_cuda_dsv4.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float decode(uint16_t v) {
    uint32_t bits=uint32_t(v)<<16; float x;
    std::memcpy(&x,&bits,sizeof(x)); return x;
}

int main() {
    int devices[]={0,5};
    assert(dsv4_cuda_init(devices,2));
    for(int device:devices) for(int rows:{256,512,1024}) for(int cols:{4096,4103}) {
        std::vector<uint16_t> w(rows*cols), g(rows*cols);
        std::vector<float> x(cols), v(rows), score(rows), expected(rows), gates(rows);
        unsigned seed=12345;
        auto next=[&](){seed=1664525*seed+1013904223;return seed;};
        for(int i=0;i<rows*cols;i++) {
            w[i]=uint16_t((next()%2048)+0x3900) | (next()&0x8000);
            g[i]=uint16_t((next()%2048)+0x3900) | (next()&0x8000);
        }
        for(auto &a:x)a=(int(next()%2000001)-1000000)/100003.f;
        for(int r=0;r<rows;r++) for(int c=0;c<cols;c++) {
            expected[r]=std::fma(decode(w[r*cols+c]),x[c],expected[r]);
            gates[r]=std::fma(decode(g[r*cols+c]),x[c],gates[r]);
        }
        Dsv4CudaTensor *kv=nullptr,*gate=nullptr,*ordinary=nullptr;
        assert(!dsv4_cuda_upload_compressor(&kv,w.data(),rows-1,cols,device));
        assert(!kv);
        assert(dsv4_cuda_upload_compressor(&kv,w.data(),rows,cols,device));
        assert(dsv4_cuda_upload_compressor(&gate,g.data(),rows,cols,device));
        assert(dsv4_cuda_upload_bf16(&ordinary,w.data(),rows,cols,device));
        assert(dsv4_cuda_tensor_bytes(kv)==4LL*rows*cols);
        assert(!dsv4_cuda_compressor_project(kv,ordinary,x.data(),v.data(),score.data()));
        assert(!dsv4_cuda_compressor_project(kv,nullptr,x.data(),v.data(),score.data()));
        assert(dsv4_cuda_compressor_project(kv,gate,x.data(),v.data(),score.data()));
        assert(std::memcmp(v.data(),expected.data(),rows*4)==0);
        assert(std::memcmp(score.data(),gates.data(),rows*4)==0);
        // Decode packing leaves the established prefill matrix untouched.
        std::vector<float> batch(3*cols), a(3*rows), b(3*rows);
        for(int i=0;i<3*cols;i++)batch[i]=x[i%cols]*(i/cols+1);
        assert(dsv4_cuda_matmul_bf16_batch(kv,batch.data(),3,a.data()));
        assert(dsv4_cuda_matmul_bf16_batch(ordinary,batch.data(),3,b.data()));
        assert(a==b);
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<100;i++)assert(dsv4_cuda_compressor_project(kv,gate,x.data(),v.data(),score.data()));
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100;
        printf("device=%d rows=%d cols=%d paired_us=%.2f exact=yes prefill=unchanged\n",device,rows,cols,us);
        dsv4_cuda_tensor_free(kv);dsv4_cuda_tensor_free(gate);dsv4_cuda_tensor_free(ordinary);
    }
    dsv4_cuda_shutdown();
}
