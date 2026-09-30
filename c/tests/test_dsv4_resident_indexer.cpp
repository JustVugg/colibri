#include "../backend_cuda_dsv4.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    int devices[] = {0, 5};
    assert(dsv4_cuda_init(devices, 2));
    constexpr int R = 256, C = 256;
    for (int device : devices) for (int packed : {0, 1}) {
        std::vector<unsigned char> weights(R*C);
        std::vector<float> scales = {.25f, .5f, 1.f, 2.f};
        for (int r=0;r<R;r++) for(int c=0;c<C;c++) {
            int offset=packed ? ((r/8)*C+c)*8+r%8 : r*C+c;
            weights[offset]=(r*11+c*17)%127 | ((r+c)%2 ? 128 : 0);
        }
        Dsv4CudaTensor *tensor=nullptr;
        assert(!dsv4_cuda_upload_fp8_ref(&tensor,weights.data(),scales.data(),R,C-1,packed,device));
        assert(!tensor);
        assert(dsv4_cuda_upload_fp8_ref(&tensor,weights.data(),scales.data(),R,C,packed,device));
        assert(dsv4_cuda_tensor_device(tensor)==device);
        assert(dsv4_cuda_tensor_bytes(tensor)==R*C+4*sizeof(float));
        for(int tokens : {1, 3, 33}) {
            std::vector<float> x(tokens*C), expected(tokens*R), actual(tokens*R);
            for(size_t i=0;i<x.size();i++)x[i]=(int(i*13%37)-18)*.125f;
            assert(dsv4_cuda_fp8_ref_matmul(device,weights.data(),scales.data(),R,C,packed,x.data(),tokens,expected.data()));
            assert(dsv4_cuda_fp8_ref_matmul_resident(tensor,x.data(),tokens,actual.data()));
            bool nonzero=false;
            for(size_t i=0;i<actual.size();i++) {
                assert(std::isfinite(actual[i]) && actual[i]==expected[i]);
                nonzero |= actual[i]!=0;
            }
            assert(nonzero);
            if(tokens==33) {
                std::fill(weights.begin(),weights.end(),0);
                std::fill(scales.begin(),scales.end(),0.f);
                assert(dsv4_cuda_fp8_ref_matmul_resident(tensor,x.data(),tokens,actual.data()));
                assert(actual==expected); // upload owns an immutable device copy
            }
            assert(!dsv4_cuda_matvec(tensor,actual.data(),x.data()));
            assert(!dsv4_cuda_fp8_ref_matmul_resident(tensor,x.data(),1025,actual.data()));
        }
        dsv4_cuda_tensor_free(tensor);
    }
    dsv4_cuda_shutdown();
    puts("resident indexer: exact parity, both layouts, batches 1/3/33, two devices");
}
