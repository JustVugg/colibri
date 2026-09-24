/* Bounded resident-int8 projection microbenchmark; no model files required.
 * Times synchronous host API calls (copies + compute), excluding upload.
 * Compare S one-row calls with one S-row call on identical resident weights.
 * This is not full-model prefill throughput or a CPU/GPU comparison. */
#include "../backend_cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    return values[values.size()/2];
}

static bool measure(int I,int O,int S,int device) {
    std::vector<int8_t> q((size_t)I*O);
    std::vector<float> scale(O), x((size_t)S*I), serial((size_t)S*O), batch(serial.size());
    for(size_t i=0;i<q.size();i++) q[i]=(int)((i*17+i/I*13)%31)-15;
    for(int o=0;o<O;o++) scale[o]=(o%3+1)/128.f;
    for(size_t i=0;i<x.size();i++) x[i]=(int(i%17)-8)/64.f;
    ColiCudaTensor *tensor=nullptr;
    if(!coli_cuda_tensor_upload(&tensor,q.data(),scale.data(),1,I,O,device)) return false;
    auto run = [&](bool batched) {
        float *out=batched?batch.data():serial.data();
        if(batched) return coli_cuda_matmul(&tensor,out,x.data(),nullptr,nullptr,1,S,I,O,device,0)!=0;
        for(int row=0;row<S;row++)
            if(!coli_cuda_matmul(&tensor,out+(size_t)row*O,x.data()+(size_t)row*I,
                                 nullptr,nullptr,1,1,I,O,device,0)) return false;
        return true;
    };
    bool ok=true;
    for(int i=0;i<2 && ok;i++) ok=run(false)&&run(true);
    std::vector<double> times[2], ratios;
    for(int rep=0;rep<9 && ok;rep++) {
        double pair[2]={};
        for(int arm=0;arm<2 && ok;arm++) {
            int mode=(rep+arm)%2;
            auto start=std::chrono::steady_clock::now();
            ok=run(mode!=0);
            pair[mode]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            times[mode].push_back(pair[mode]);
        }
        if(ok) ratios.push_back(pair[0]/pair[1]);
        for(size_t i=0;i<serial.size() && ok;i++)
            ok=std::isfinite(serial[i]) && std::isfinite(batch[i]) &&
               std::fabs(serial[i]-batch[i])<=1e-5f*(1.f+std::fabs(serial[i]));
    }
    // Independent sampled CPU reference, outside timed regions. Full outputs
    // are compared between arms after every pair, not only the final sample.
    double max_error=0;
    for(int row=0;row<S && ok;row+=std::max(1,S/7)) for(int o=0;o<O;o+=std::max(1,O/7)) {
        double reference=0;
        for(int i=0;i<I;i++) reference+=double(x[(size_t)row*I+i])*q[(size_t)o*I+i];
        reference*=scale[o];
        double error=std::fabs(batch[(size_t)row*O+o]-reference);
        max_error=std::max(max_error,error);
        if(error>1e-5*(1+std::fabs(reference))) ok=false;
    }
    coli_cuda_tensor_free(tensor);
    if(!ok){std::fprintf(stderr,"FAIL: resident batch I=%d O=%d S=%d\n",I,O,S);return false;}
    std::printf("{\"input\":%d,\"output\":%d,\"rows\":%d,\"pairs\":9,"
                "\"serial_median_ms\":%.6f,\"batch_median_ms\":%.6f,"
                "\"paired_speedup_median\":%.6f,\"cpu_sample_max_abs_error\":%.9g,"
                "\"serial_ms\":[",I,O,S,median(times[0]),median(times[1]),median(ratios),max_error);
    for(size_t i=0;i<times[0].size();i++) std::printf("%s%.6f",i?",":"",times[0][i]);
    std::printf("],\"batch_ms\":[");
    for(size_t i=0;i<times[1].size();i++) std::printf("%s%.6f",i?",":"",times[1][i]);
    std::puts("]}");
    return true;
}

int main() {
    int device=0;
    if(!coli_cuda_init(&device,1)) return 77;
    bool ok=measure(512,1024,32,device) && measure(2048,2048,32,device) && measure(2048,2048,128,device);
    coli_cuda_shutdown();
    return ok?0:1;
}
