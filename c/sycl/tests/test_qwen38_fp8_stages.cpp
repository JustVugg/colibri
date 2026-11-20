/* Matched Qwen38 CPU/GPU stage diagnostics, through existing public ABI only.
 * Every fixture is validated before any device call. Numerical differences are
 * evidence, not relaxed correctness gates: keep test_sycl_fp8_experts and native
 * NLL qualification separately. See MIGRATION 24 and tests/qwen38_fp8_fixture.h.
 * Output (little endian): Q38GPU1\0, u32 version/count; each case has u32
 * fixture_index,id,D,I,S,pattern, then gate/up/silu_cpu[S*I] and
 * down_cpu/mlp/group/async[S*D]. Rows follow the fixture's order within a case.
 */
#include "../../backend_cuda.h"
#include "../../tests/qwen38_fp8_fixture.h"
#include "sycl_test_args.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

static void need(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
using Floats=std::vector<float>;
using File=std::unique_ptr<FILE,decltype(&fclose)>;
struct Reader {
    File f{nullptr,fclose}; size_t size=0,pos=0;
    explicit Reader(const char *path):f(fopen(path,"rb"),fclose) {
        need(bool(f),"open fixture"); struct stat st{};
        need(fstat(fileno(f.get()),&st)==0 && S_ISREG(st.st_mode) &&
             st.st_size>=16 && st.st_size<=QFS_MAX_BYTES,"bounded regular fixture");
        size=(size_t)st.st_size;
    }
    void read(void *p,size_t n) {
        need(n<=size-pos,"truncated fixture");
        need(fread(p,1,n,f.get())==n,"fixture read"); pos+=n;
    }
    template<class T> std::vector<T> values(size_t n) {
        need(n<=(size-pos)/sizeof(T),"bounded vector length");
        std::vector<T> v(n);read(v.data(),n*sizeof(T));return v;
    }
    Floats floats(size_t n,bool positive=false) {
        auto v=values<float>(n);
        for(float x:v)need(std::isfinite(x)&&(!positive||x>0),"finite fixture float/scale");
        return v;
    }
    void end() { need(pos==size&&fgetc(f.get())==EOF&&!ferror(f.get()),"trailing fixture data"); }
};
struct Weight { std::vector<uint8_t> q; Floats scale; };
struct Case {
    uint32_t fixture,id,D,I,S,pattern;
    std::array<Weight,3> w; Floats x; std::array<Floats,4> v,t;
};
static std::vector<Case> load(const char *path,uint32_t fixture) {
    Reader r(path);char magic[8];uint32_t version,count;
    r.read(magic,8);r.read(&version,4);r.read(&count,4);
    need(!memcmp(magic,QFS_MAGIC,8)&&version==QFS_VERSION&&count>0&&count<=QFS_MAX_CASES,
         "fixture magic/version/count");
    std::vector<Case> cases;cases.reserve(count);
    for(uint32_t id=0;id<count;id++) {
        uint32_t h[5];r.read(h,sizeof(h));
        need(h[0]==id&&h[1]>0&&h[1]<=QFS_MAX_DIM&&h[2]>0&&h[2]<=QFS_MAX_DIM&&
             h[3]>0&&h[3]<=QFS_MAX_ROWS&&h[4]<=QFS_NATIVE_PATTERN,"fixture case header");
        Case c{};c.fixture=fixture;c.id=id;c.D=h[1];c.I=h[2];c.S=h[3];c.pattern=h[4];
        size_t nq=(size_t)c.D*c.I,ns=(size_t)((c.D+127)/128)*((c.I+127)/128);
        size_t nx=(size_t)c.S*c.D,nh=(size_t)c.S*c.I;
        for(auto &w:c.w) {
            w.q=r.values<uint8_t>(nq);
            for(uint8_t b:w.q)need((b&127)!=127,"finite FP8 weights");
            w.scale=r.floats(ns,true);
        }
        c.x=r.floats(nx);
        for(int k=0;k<4;k++)c.v[k]=r.floats(k==3?nx:nh);
        for(int k=0;k<4;k++)c.t[k]=r.floats(k==3?nx:nh);
        cases.push_back(std::move(c));
    }
    r.end();std::printf("FIXTURE VALIDATED index=%u cases=%u bytes=%zu\n",fixture,count,r.size);
    return cases;
}
struct Writer {
    std::string path,part;File f{nullptr,fclose};size_t written=0;
    explicit Writer(const char *p,uint32_t count):path(p),part(path+".part") {
        struct stat st{};errno=0;need(lstat(p,&st)<0&&errno==ENOENT,"output already exists or inaccessible");
        int fd=open(part.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);need(fd>=0,"exclusive output partial");
        f.reset(fdopen(fd,"wb"));if(!f){close(fd);throw std::runtime_error("fdopen output");}
        uint32_t v=1;write("Q38GPU1\0",8);write(&v,4);write(&count,4);
    }
    void write(const void *p,size_t n) {
        need(n<=QFS_MAX_BYTES-written,"output size bound");
        need(fwrite(p,1,n,f.get())==n,"output write");written+=n;
    }
    void publish() {
        need(fflush(f.get())==0&&fsync(fileno(f.get()))==0,"sync completed output");
        need(fclose(f.release())==0,"close completed output");
        need(link(part.c_str(),path.c_str())==0,"publish without replacement");
        need(unlink(part.c_str())==0,"unlink partial");
        auto slash=path.find_last_of('/');std::string dir=slash==std::string::npos?".":path.substr(0,slash);
        if(dir.empty())dir="/";
        int fd=open(dir.c_str(),O_RDONLY|O_DIRECTORY);need(fd>=0,"output directory");
        int rc=fsync(fd),closed=close(fd);need(rc==0&&closed==0,"sync output directory");
    }
};
static void compare(const Case &c,const char *stage,const char *ref,const Floats &a,const Floats &b) {
    need(a.size()==b.size(),"comparison dimensions");double err=0,sum=0,maximum=0;size_t bits=0;
    for(size_t i=0;i<a.size();i++) {
        need(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite GPU/CPU result");
        double d=(double)a[i]-b[i];err+=d*d;sum+=(double)b[i]*b[i];maximum=std::max(maximum,std::fabs(d));
        bits+=memcmp(&a[i],&b[i],sizeof(float))!=0;
    }
    std::printf("STAGE {\"fixture\":%u,\"id\":%u,\"D\":%u,\"I\":%u,\"S\":%u,\"pattern\":%u,"
                "\"stage\":\"%s\",\"reference\":\"%s\",\"relative_rms\":%.17g,\"max_abs\":%.17g,"
                "\"bit_differences\":%zu,\"elements\":%zu}\n",
                c.fixture,c.id,c.D,c.I,c.S,c.pattern,stage,ref,std::sqrt(err/(sum+1e-30)),maximum,bits,a.size());
}
struct Tensors {
    ColiCudaTensor *p[3]={nullptr,nullptr,nullptr};
    ~Tensors(){for(auto t:p)if(t)coli_cuda_tensor_free(t);}
};
struct Pipe {
    int dev;float *p;
    Pipe(int d,size_t n):dev(d),p((float*)coli_cuda_pipe_alloc(d,n*sizeof(float))){need(p,"pipe allocation");}
    ~Pipe(){coli_cuda_pipe_free(dev,p);}
};
static void run_case(int dev,const Case &c,Writer &out) {
    std::printf("GPU_CASE START fixture=%u id=%u D=%u I=%u S=%u pattern=%u\n",c.fixture,c.id,c.D,c.I,c.S,c.pattern);
    std::fflush(stdout);Tensors weights;
    for(int w=0;w<3;w++)need(coli_cuda_tensor_upload(&weights.p[w],c.w[w].q.data(),c.w[w].scale.data(),
              8,w==2?c.I:c.D,w==2?c.D:c.I,dev),"tensor upload");
    size_t nx=(size_t)c.S*c.D,nh=(size_t)c.S*c.I;
    std::array<Floats,7> got;
    for(int k=0;k<7;k++)got[k].resize(k<3?nh:nx,std::numeric_limits<float>::quiet_NaN());
    for(int w=0;w<2;w++)need(coli_cuda_matmul(&weights.p[w],got[w].data(),c.x.data(),nullptr,nullptr,
            8,c.S,c.D,c.I,dev,0),"gate/up matmul");
    {
        Pipe gate(dev,nh),up(dev,nh);
        need(coli_cuda_pipe_upload(dev,gate.p,c.v[0].data(),nh*sizeof(float)),"pipe gate upload");
        need(coli_cuda_pipe_upload(dev,up.p,c.v[1].data(),nh*sizeof(float)),"pipe up upload");
        need(coli_cuda_pipe_silu_mul(dev,gate.p,up.p,nh),"pipe silu mul");
        need(coli_cuda_pipe_download(dev,gate.p,got[2].data(),nh*sizeof(float)),"synchronized pipe download");
    }
    need(coli_cuda_matmul(&weights.p[2],got[3].data(),c.v[2].data(),nullptr,nullptr,
                         8,c.S,c.I,c.D,dev,0),"CPU-hidden down matmul");
    need(coli_cuda_expert_mlp(weights.p[0],weights.p[1],weights.p[2],got[4].data(),c.x.data(),c.S),"full expert MLP");
    int rows=(int)c.S;
    need(coli_cuda_expert_group(weights.p,weights.p+1,weights.p+2,&rows,1,got[5].data(),c.x.data()),"group sync");
    need(coli_cuda_expert_group_issue(weights.p,weights.p+1,weights.p+2,&rows,1,c.x.data()),"group issue");
    const float *async=coli_cuda_expert_group_take(dev);need(async,"group take");
    // take's pointer is borrowed: copy before any subsequent backend call.
    memcpy(got[6].data(),async,nx*sizeof(float));
    need(!memcmp(got[5].data(),got[6].data(),nx*sizeof(float)),"async/sync byte mismatch");
    const char *names[7]={"gate","up","silu_cpu","down_cpu","mlp","group","async"};
    for(int k=0;k<7;k++) {
        int ref=k<3?k:3;compare(c,names[k],"vector",got[k],c.v[ref]);
        // Isolated activation/down use vector inputs, so no table-output claim.
        if(k<2||k>=4)compare(c,names[k],"table",got[k],c.t[ref]);
    }
    uint32_t h[6]={c.fixture,c.id,c.D,c.I,c.S,c.pattern};out.write(h,sizeof(h));
    for(auto &g:got)out.write(g.data(),g.size()*sizeof(float));
    std::printf("GPU_CASE END fixture=%u id=%u finite=1 async_sync_identical=1\n",c.fixture,c.id);std::fflush(stdout);
}
static float decode(uint8_t b) {
    int e=(b>>3)&15,m=b&7;if(e==15&&m==7)return std::numeric_limits<float>::quiet_NaN();
    float v=e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(m/8.f,-6);return b&128?-v:v;
}
int main(int argc,char **argv) {
    bool opened=false;
    try {
        uint32_t endian=1;need(*(uint8_t*)&endian==1&&sizeof(float)==4&&
            std::numeric_limits<float>::is_iec559,"little-endian IEEE binary32");
        std::vector<const char*> paths;std::vector<char*> device_args{argv[0]};
        bool selected_device=false;char default_flag[]="--device",default_device[]="0";
        bool validate=false;const char *output=nullptr;
        for(int i=1;i<argc;i++) {
            if(!strcmp(argv[i],"--fixture")&&i+1<argc)paths.push_back(argv[++i]);
            else if(!strcmp(argv[i],"--output")&&i+1<argc){need(!output,"duplicate output");output=argv[++i];}
            else if(!strcmp(argv[i],"--validate-only"))validate=true;
            else if(!strcmp(argv[i],"--device")&&i+1<argc){
                need(!selected_device,"duplicate device option");selected_device=true;
                char *end=nullptr;errno=0;long d=strtol(argv[i+1],&end,10);
                need(!errno&&end!=argv[i+1]&&!*end&&d>=0&&d<=INT_MAX,"one nonnegative device ordinal");
                device_args.push_back(argv[i]);device_args.push_back(argv[++i]);
            }
            else throw std::runtime_error("usage: --fixture FILE [--fixture FILE] [--device N] (--output NEW_FILE | --validate-only)");
        }
        need(!paths.empty()&&paths.size()<=2,"one or two fixture files required");
        need(validate?!output:output!=nullptr,"select validation-only or output");
        std::vector<Case> cases;
        for(uint32_t i=0;i<paths.size();i++) {
            auto loaded=load(paths[i],i);for(auto &c:loaded)cases.push_back(std::move(c));
        }
        // Tiny matrix shapes before native shapes, preserving source order for ties.
        std::stable_sort(cases.begin(),cases.end(),[](const Case&a,const Case&b){return(size_t)a.D*a.I<(size_t)b.D*b.I;});
        std::printf("ALL FIXTURES VALIDATED cases=%zu; no device initialized\n",cases.size());std::fflush(stdout);
        if(validate)return 0;
        if(!selected_device){device_args.push_back(default_flag);device_args.push_back(default_device);}
        int dev[1];int n=sycl_test_devices((int)device_args.size(),device_args.data(),dev,1);
        need(n==1,"exactly one device");Writer out(output,(uint32_t)cases.size());
        need(coli_cuda_init(dev,n),"device init");opened=true;
        float lut[256];for(int b=0;b<256;b++)lut[b]=decode((uint8_t)b);
        need(coli_cuda_fp8_set_lut(lut),"FP8 lookup table");
        for(const auto &c:cases)run_case(dev[0],c,out);
        size_t count=0,bytes=0;coli_cuda_stats(dev[0],&count,&bytes);need(count==0&&bytes==0,"tensor accounting leak");
        coli_cuda_shutdown();opened=false;out.publish();
        std::printf("GPU STAGE DIAGNOSTICS COMPLETE cases=%zu bytes=%zu; native quality gate unchanged\n",cases.size(),out.written);
        return 0;
    }catch(const std::exception&e){
        std::fprintf(stderr,"FAIL Qwen38 stage diagnostics: %s; device_open=%d\n",e.what(),(int)opened);
        // An exceptional device run is a stop condition. Do not launch cleanup
        // kernels or retry; the guarded process/container exit releases it.
        return 1;
    }
}
