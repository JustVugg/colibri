/* Synchronous peer copies: separate-context refusal, owner validation,
 * pending source writes, both directions, and shutdown/reinitialization. */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void need(bool ok,const char *what) {
    if (!ok) { std::fprintf(stderr,"PEER COPY FAIL: %s\n",what); std::exit(1); }
}
int main(int argc,char **argv) {
    int dev[2], n=sycl_test_devices(argc,argv,dev,2); if(n<0)return 0;
    need(n==2,"two devices required");
    const char *e=std::getenv("COLI_SYCL_PEER_CONTEXT");
    const bool requested=e && !std::strcmp(e,"1");
    constexpr size_t count=262144, bytes=count*sizeof(float);
    for(int cycle=0;cycle<3;cycle++){
        need(coli_cuda_init(dev,n),"init");
        bool shared=coli_sycl_context(dev[0])==coli_sycl_context(dev[1]);
        need(shared==requested,"context selection");
        auto *a=(float*)coli_cuda_pipe_alloc(dev[0],bytes);
        auto *b=(float*)coli_cuda_pipe_alloc(dev[1],bytes);
        need(a&&b,"device allocations");
        std::vector<float> out(count,0);
        /* Leave the producer pending: the peer-copy API must order it. */
        coli_sycl_queue(dev[0]).parallel_for(sycl::range<1>(count),[=](sycl::id<1> i){a[i]=(float)(i[0]%997);});
        if(shared){
            need(coli_cuda_pipe_peer_copy(dev[1],b,dev[0],a,bytes),"device 0 to 1");
            need(coli_cuda_pipe_download(dev[1],b,out.data(),bytes),"download device 1");
            for(size_t i=0;i<count;i++)need(out[i]==(float)(i%997),"forward contents");
            if(cycle==0) for(size_t nb : {size_t(8192),bytes}) for(int dir=0;dir<2;dir++) {
                std::vector<double> times;
                for(int rep=0;rep<5;rep++){
                    auto start=std::chrono::steady_clock::now();
                    need(dir ? coli_cuda_pipe_peer_copy(dev[0],a,dev[1],b,nb)
                             : coli_cuda_pipe_peer_copy(dev[1],b,dev[0],a,nb),"timed peer copy");
                    auto stop=std::chrono::steady_clock::now();
                    times.push_back(std::chrono::duration<double>(stop-start).count());
                }
                std::sort(times.begin(),times.end());
                std::printf("peer-copy %d->%d bytes=%zu median_us=%.3f\n",dev[dir?1:0],dev[dir?0:1],nb,times[2]*1e6);
            }
            std::fill(out.begin(),out.end(),-1.f);
            need(coli_cuda_pipe_upload(dev[1],b,out.data(),bytes),"initialize offset destination");
            need(coli_cuda_pipe_peer_copy(dev[1],b+17,dev[0],a+17,(count-34)*sizeof(float)),"offset pointers");
            need(coli_cuda_pipe_download(dev[1],b,out.data(),bytes),"download offset destination");
            for(size_t i=0;i<count;i++)need(out[i]==(i<17 || i>=count-17 ? -1.f : (float)(i%997)),"offset bounds");
            need(!coli_cuda_pipe_peer_copy(dev[1],b,dev[0],b,bytes),"source owner refused");
            need(!coli_cuda_pipe_peer_copy(dev[0],b,dev[1],a,bytes),"destination owner refused");
            need(!coli_cuda_pipe_peer_copy(dev[1],b,dev[0],out.data(),bytes),"host pointer refused");
            coli_sycl_queue(dev[1]).parallel_for(sycl::range<1>(count),[=](sycl::id<1> i){b[i]=(float)(i[0]%991);});
            need(coli_cuda_pipe_peer_copy(dev[0],a,dev[1],b,bytes),"device 1 to 0");
            need(coli_cuda_pipe_download(dev[0],a,out.data(),bytes),"download device 0");
            for(size_t i=0;i<count;i++)need(out[i]==(float)(i%991),"reverse contents");
        } else {
            need(!coli_cuda_pipe_peer_copy(dev[1],b,dev[0],a,bytes),"separate contexts refused");
            coli_sycl_queue(dev[0]).wait_and_throw();
        }
        coli_cuda_pipe_free(dev[1],b); coli_cuda_pipe_free(dev[0],a); coli_cuda_shutdown();
        std::printf("peer-copy cycle=%d mode=%s PASS\n",cycle,shared?"shared":"separate");
    }
    return 0;
}
