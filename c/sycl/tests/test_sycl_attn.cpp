/* Independent softmax reference for the opt-in XMX attention. --small must be
 * the first run of a new kernel. No test continues after an exception/mismatch.
 * Large tests sample four query rows, but check ALL output padding/finite values.
 * --bench measures the actual chain dispatch (env0 fallback, env1 XMX), batches
 * of eight operations with one completion, after the same CPU reference check.
 * --dnnl runs the same direct cases on oneDNN's fused attention (coli_sycl_dnnl_attn;
 * COLI_SYCL_DNNL_ATTN=2 in a SYCL_DNNL=1 build), the cache 16-byte aligned: the same
 * declines, the profile's two replaced by a misaligned K and a misaligned V, a float32
 * cache declined, the same limits. --bench with COLI_SYCL_DNNL_ATTN=2 times the chain's
 * dispatch through it. */
#include "../coli_sycl.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void need(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
static float sample(unsigned &s) { s = s * 1664525u + 1013904223u; return (s >> 8) / 8388608.f - 1.f; }
static float h16(float x) { return (float)sycl::half(std::fmin(std::fmax(x, -65504.f), 65504.f)); }

struct Case {
    CcAttn p{};
    std::vector<float> x, k, v, g, init;
    std::vector<sycl::half> kh, vh;
    Case(int S, int pos, int HD, int H, int KVH, bool gate, bool zero, int koff = 3) {
        p.S=S; p.pos_base=pos; p.hd=HD; p.H=H; p.KVH=KVH; p.cap=pos+S+5;
        p.q_off=5; p.q_seg=HD+4; p.q_row=H*p.q_seg+8;
        p.g_off=9; p.g_seg=HD+8; p.g_row=H*p.g_seg+12; p.has_gate=gate;
        p.o_off=7; p.o_row=H*HD+12; p.k_off=koff; p.v_off=koff; p.scale=1.f/std::sqrt((float)HD);
        x.resize(p.q_off+(size_t)S*p.q_row+11); g.resize(p.g_off+(size_t)S*p.g_row+11);
        k.resize(p.k_off+(size_t)KVH*p.cap*HD+11); v.resize(k.size());
        init.assign(p.o_off+(size_t)S*p.o_row+11, -1234.5f);
        unsigned seed=39+S+HD;
        for(auto &a:x) a=zero?0.f:sample(seed);
        for(auto &a:g) a=sample(seed)*3.f;
        // Both cache forms contain identical half-representable values.
        for(auto &a:k) a=h16(sample(seed));
        for(auto &a:v) a=h16(sample(seed));
        kh.assign(k.begin(),k.end()); vh.assign(v.begin(),v.end());
    }
    double check(const std::vector<float> &got, bool rounded_q, bool large, double tol) const {
        need(got.size()==init.size(),"output size");
        for(size_t i=0;i<got.size();i++) {
            const size_t j=i>=size_t(p.o_off)?i-p.o_off:SIZE_MAX;
            const bool valid=j/p.o_row<(size_t)p.S && j%p.o_row<(size_t)p.H*p.hd;
            need(valid?std::isfinite(got[i]):got[i]==init[i],"output nonfinite or padding overwritten");
        }
        std::vector<int> rows;
        if(large) rows={0,1,p.S/2,p.S-1}; else for(int r=0;r<p.S;r++) rows.push_back(r);
        double diff=0;
        for(int r:rows) for(int h=0;h<p.H;h++) {
            const int nt=p.pos_base+r+1, kv=h/(p.H/p.KVH);
            std::vector<double> score(nt); double mx=-INFINITY, den=0;
            for(int t=0;t<nt;t++) {
                double dot=0;
                for(int d=0;d<p.hd;d++) {
                    float qv=x[p.q_off+(size_t)r*p.q_row+h*p.q_seg+d];
                    dot+=(rounded_q?h16(qv):qv)*double(k[p.k_off+((size_t)kv*p.cap+t)*p.hd+d]);
                }
                score[t]=dot*p.scale; mx=std::max(mx,score[t]);
            }
            for(auto &s:score) {s=std::exp(s-mx);den+=s;}
            for(int d=0;d<p.hd;d++) {
                double val=0;
                for(int t=0;t<nt;t++) val+=score[t]*v[p.v_off+((size_t)kv*p.cap+t)*p.hd+d];
                val/=den;
                if(p.has_gate) val/=1.+std::exp(-double(g[p.g_off+(size_t)r*p.g_row+h*p.g_seg+d]));
                diff=std::max(diff,std::abs(val-got[p.o_off+(size_t)r*p.o_row+h*p.hd+d]));
            }
        }
        std::printf("  CPU %s Q reference: max_abs %.8g (%s rows), limit %.8g\n",rounded_q?"f16":"f32",diff,large?"four sampled":"all",tol);
        need(diff<tol,"CPU reference mismatch"); return diff;
    }
};

template<class T> static T *upload(sycl::queue &q, const std::vector<T> &v) {
    T *p=sycl::malloc_device<T>(v.size(),q); need(p!=nullptr,"device allocation");
    q.memcpy(p,v.data(),v.size()*sizeof(T)).wait_and_throw(); return p;
}

using AttnEntry = int (*)(const coli_sycl_profile &, sycl::queue &, const coli_sycl_attn &);
static void direct(sycl::queue &q, const coli_sycl_profile &profile, Case &c, bool large,
                   AttnEntry run = coli_sycl_xmx_attn, bool dnnl = false) {
    auto x=upload(q,c.x),k=upload(q,c.k),v=upload(q,c.v),g=upload(q,c.g),out=upload(q,c.init);
    auto kh=upload(q,c.kh),vh=upload(q,c.vh);
    coli_sycl_attn a{x,k,v,0,out,g,c.p,c.p.H/c.p.KVH,0,nullptr,0,0,nullptr};
    // Every unsupported form must decline. Deliberately corrupt one field at a
    // time; output must remain the sentinel before the first valid submission.
    for(int i=0;i<14;i++) {
        auto bad=a; auto pr=profile;
        if(dnnl) {bad.kv_f16=1;bad.kc=kh;bad.vc=vh;}   // so that the field alone decides
        switch(i) {
        case 0: bad.win=1;break; case 1:bad.snk=g;break; case 2:bad.ring=c.p.cap;break;
        case 3:bad.sel=(int*)k;break;case 4:bad.p.sel_row=1;break;
        case 5:bad.p.pos_base=c.p.cap;break;case 6:bad.p.q_off=-1;break;
        case 7:bad.p.q_row=1;break;case 8:bad.G++;break;case 9:bad.p.hd=96;break;
        case 10:bad.q=out;break;case 11:bad.p.has_gate=1;bad.gate=nullptr;break;
        // the profile's fields for XMX; for oneDNN (which does not read them) a K, then a V, not 16-byte aligned
        case 12:if(dnnl)bad.p.k_off++;else pr.has_xmx=0;break;
        case 13:if(dnnl)bad.p.v_off++;else pr.slm_bytes=0;break;
        }
        need(!run(pr,q,bad),"unsupported attention submitted");
    }
    std::vector<float> got(c.init.size()),first;
    q.memcpy(got.data(),out,got.size()*4).wait_and_throw();need(got==c.init,"declined dispatch touched output");
    for(int half=0;half<=1;half++) {
        a.kv_f16=half;a.kc=half?(void*)kh:(void*)k;a.vc=half?(void*)vh:(void*)v;
        if(dnnl&&!half) {   // a float32 cache is not oneDNN's: declined, nothing written
            need(!run(profile,q,a),"oneDNN took a float32 cache"); q.wait_and_throw();
            q.memcpy(got.data(),out,got.size()*4).wait_and_throw();need(got==c.init,"declined dispatch touched output");
            continue;
        }
        need(run(profile,q,a),dnnl?"oneDNN not taken":"XMX not taken"); q.wait_and_throw();
        q.memcpy(got.data(),out,got.size()*4).wait_and_throw();
        std::printf("ATTN direct%s S%d pos%d H%d KV%d hd%d gate%d kv_f16%d\n",dnnl?" (oneDNN)":"",c.p.S,c.p.pos_base,c.p.H,c.p.KVH,c.p.hd,c.p.has_gate,half);
        // Probability rounding plus f32 reduction; original unrounded Q check
        // retained too. No existing chain tolerance is changed by this test.
        c.check(got,true,large,5e-4); c.check(got,false,large,1e-3);
        if(!half) first=got;else if(!dnnl) need(!std::memcmp(first.data(),got.data(),got.size()*4),"cache formats differ in bits");
    }
    for(auto p:{x,k,v,g,out}) sycl::free(p,q);sycl::free(kh,q);sycl::free(vh,q);
}

static void bench(Case &c,int reps) {
    auto *T=cc_ops(); need(T&&T->ready(),"chain not ready");
    auto put=[&](const void *p,size_t bytes) { CcBuf *b=T->buf(bytes,CC_DEV);need(b&&T->begin()&&T->write(b,0,p,bytes)&&T->submit(1),"chain upload");return b;};
    auto x=put(c.x.data(),c.x.size()*4),k=put(c.kh.data(),c.kh.size()*2),v=put(c.vh.data(),c.vh.size()*2),g=put(c.g.data(),c.g.size()*4),o=put(c.init.data(),c.init.size()*4);
    auto frame=[&](int n) {need(T->begin(),"chain begin");for(int i=0;i<n;i++) need(T->attn_h(x,k,v,o,g,nullptr,&c.p),"chain attention");need(T->submit(1),"chain submit");};
    frame(1);
    std::vector<float> got(c.init.size());need(T->read(o,0,got.data(),got.size()*4),"chain read");
    c.check(got,false,true,1e-3);
    /* FNV-1a over the output's bytes: two dispatches that agree bit for bit print the same hash */
    {
        unsigned long long hsh = 1469598103934665603ull;
        const unsigned char *gb = reinterpret_cast<const unsigned char *>(got.data());
        for (size_t i = 0; i < got.size() * 4; i++) { hsh ^= gb[i]; hsh *= 1099511628211ull; }
        std::printf("ATTN HASH S%d pos%d koff%d: %016llx GROUPED_PLAIN=%s GROUPED_VEC=%s\n", c.p.S, c.p.pos_base, c.p.k_off, hsh, std::getenv("COLI_CUDA_ATTN_GROUPED_PLAIN"), std::getenv("COLI_CUDA_ATTN_GROUPED_VEC"));
    }
    frame(2);
    const auto start=std::chrono::steady_clock::now();for(int i=0;i<reps;i++)frame(8);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/(reps*8);
    std::printf("ATTN BENCH S%d pos%d H%d KV%d hd%d: %.6f ms/op, %d x 8 batch, XMX_ATTN=%s\n",c.p.S,c.p.pos_base,c.p.H,c.p.KVH,c.p.hd,ms,reps,std::getenv("COLI_SYCL_XMX_ATTN"));
    for(auto b:{x,k,v,g,o}) T->free(b);
}

int main(int argc,char **argv) {
    int dev=0,reps=5,koff=3;bool small=false,benchmark=false,dnnl=false;
    for(int i=1;i<argc;i++) {
        if(!std::strcmp(argv[i],"--small"))small=true;
        else if(!std::strcmp(argv[i],"--bench"))benchmark=true;
        else if(!std::strcmp(argv[i],"--dnnl"))dnnl=true;
        else if(!std::strcmp(argv[i],"--device")&&i+1<argc)dev=std::atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--reps")&&i+1<argc)reps=std::atoi(argv[++i]);
        else if(!std::strcmp(argv[i],"--koff")&&i+1<argc)koff=std::atoi(argv[++i]);
        else {std::fprintf(stderr,"usage: %s [--device N] [--small|--bench] [--dnnl] [--reps N] [--koff N]\n",argv[0]);return 2;}
    }
    try {
        need(dev>=0&&reps>0&&reps<=100,"invalid args");
        need(coli_cuda_init(&dev,1)&&cc_init(dev),"device/chain init");
        auto &q=coli_sycl_queue(dev);const auto &p=coli_sycl_profile_for(dev);
        if(benchmark) {
            for(int S:{256,2048}) {Case c(S,0,256,16,2,true,false,koff);bench(c,reps);}
        } else if(dnnl) {
            const char *dm=std::getenv("COLI_SYCL_DNNL_ATTN");
            need(dm&&!std::strcmp(dm,"2")&&coli_sycl_dnnl_attn_on(),"--dnnl needs COLI_SYCL_DNNL_ATTN=2 (every call compiled on the caller) in a SYCL_DNNL=1 build");
            // the cache 8 elements in (16 bytes): the least alignment oneDNN's path takes
            for(int hd:{64,128,256}) {
                Case one(1,0,hd,1,1,false,false,8);direct(q,p,one,false,coli_sycl_dnnl_attn,true);
                Case tail(9,27,hd,4,2,true,false,8);direct(q,p,tail,false,coli_sycl_dnnl_attn,true);
                Case zeros(17,33,hd,2,1,false,true,8);direct(q,p,zeros,false,coli_sycl_dnnl_attn,true);
                Case keytail(9,59,hd,4,2,true,false,8);direct(q,p,keytail,false,coli_sycl_dnnl_attn,true);
            }
            if(!small)for(int S:{256,2048}) {Case c(S,S==2048?6144:0,256,16,2,true,false,8);direct(q,p,c,true,coli_sycl_dnnl_attn,true);}
        } else {
            need(coli_sycl_xmx_attn_ok(p,64,1,1,0),"opt-in/profile missing; direct test requires COLI_SYCL_XMX_ATTN=1");
            for(int hd:{64,128,256}) {
                Case one(1,0,hd,1,1,false,false);direct(q,p,one,false);
                Case tail(9,27,hd,4,2,true,false);direct(q,p,tail,false);
                Case zeros(17,33,hd,2,1,false,true);direct(q,p,zeros,false);
                Case keytail(9,59,hd,4,2,true,false);direct(q,p,keytail,false);
            }
            if(!small)for(int S:{256,2048}) {Case c(S,S==2048?6144:0,256,16,2,true,false);direct(q,p,c,true);}
        }
        q.wait_and_throw();std::puts("ATTN PASSED");cc_shutdown();coli_cuda_shutdown();return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"ATTN STOP: %s\n",e.what());return 1;}
}
