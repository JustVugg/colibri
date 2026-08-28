/* fmt=8 (fp8-e4m3) warp-kernel oracle (COLI_CUDA_F8_WARP rework).
 *
 * Complements tests/test_fp8_cuda.cu (which pins the ORIGINAL kernels and the
 * upload API on small tail shapes) with the contracts the warp rework adds:
 *
 *   1. 256-value decode sweep: every byte decoded on-device through each
 *      compiled path (shared-LUT, and the cuda_fp8.h cvt path where built)
 *      against an arithmetic host reference — bit-compare, NaN via isnan,
 *      plus a proof-of-bite pass with one mutated LUT entry.
 *   2. LUT gate: fmt=8 upload refused before coli_cuda_fp8_set_lut.
 *   3. Grouped-vs-CPU on the census shapes (gate/up 6144x2048, down
 *      2048x6144) AND a synthetic tail shape (O,I not %128), relative RMS +
 *      max-ulp logged; dense quant_matmul branch on the census shape too.
 *   4. Old-vs-new: COLI_CUDA_F8_WARP=0 reruns the same group through the
 *      original kernels; both must sit inside the CPU-reference bound (the
 *      two paths are NOT bit-identical — the accumulation convention is the
 *      point of the rework).
 *   5. Mixed-group fallback: fmt=8 + fmt=2 group == per-expert MLP, bitwise.
 *   6. issue/take async parity vs the sync path, bitwise.
 *   7. S-invariance: rows=64 in one call vs 64 rows=1 calls, bitwise (the
 *      s-tile changes concurrency, never the order within a dot).
 *   8. NaN-byte injection: a planted 0x7F reaches the output as NaN exactly
 *      where the CPU reference produces one (no kernel-side scrubbing).
 *   9. Denormal-scale contribution: scale x subnormal-decode landing in the
 *      f32 denormal range must match the CPU bit for bit (catches -ftz/
 *      fast-math contamination of the build, not just kernel bugs).
 *
 * The e4m3 reference decoder is computed arithmetically here, so it
 * cross-checks the LUT rather than assuming it.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_fp8_warp_cuda.cu -o tests/test_fp8_warp
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

/* No direct <cuda_runtime.h>: the backend include below pulls
 * backend_gpu_compat.h, which supplies the runtime surface for BOTH vendors
 * (a direct include breaks the hipcc build the same way test_backend_cuda.cu
 * avoids by including only backend headers). */
#include "../backend_cuda.cu"

#ifdef _WIN32
/* MSVC has no POSIX setenv/unsetenv */
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
static int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif

static float e4m3_ref(uint8_t b){
    int s=b>>7, e=(b>>3)&15, m=b&7;
    if(e==15&&m==7) return NAN;                      /* E4M3-FN: only NaN, no inf */
    float v = e ? ldexpf(1.f+m/8.f, e-7) : ldexpf(m/8.f, -6);
    return s ? -v : v;
}

static uint8_t rnd_e4m3(void){                        /* any byte except the two NaNs */
    uint8_t b=(uint8_t)(rand()&255);
    if((b&0x7F)==0x7F) b&=(uint8_t)~1;
    return b;
}

/* matmul_fp8's accumulation, one output element: float within a 128-block,
 * double across blocks, block scale applied on the block subtotal. */
static void cpu_gemv_f8(const uint8_t *q,const float *sc,int K,int O,
                        const float *x,float *y){
    int nblk=(K+127)/128;
    for(int o=0;o<O;o++){
        const uint8_t *w=q+(size_t)o*K;
        const float *scl=sc+(size_t)(o/128)*nblk;
        double a=0;
        for(int b=0;b*128<K;b++){
            int base=b*128, len=K-base<128?K-base:128; float acc=0;
            for(int i=base;i<base+len;i++) acc+=e4m3_ref(w[i])*x[i];
            a+=(double)acc*scl[b];
        }
        y[o]=(float)a;
    }
}

/* relative RMS + max ulp distance, both logged (the ulp figure is
 * informational: across a sign flip it is not a meaningful distance). */
static int rms_named(const char *name,const float *got,const float *want,
                     size_t n,float limit){
    double err=0,ref=0; long mu=0;
    for(size_t i=0;i<n;i++){
        double d=(double)got[i]-want[i]; err+=d*d; ref+=(double)want[i]*want[i];
        int32_t a,b; memcpy(&a,&got[i],4); memcpy(&b,&want[i],4);
        long u=labs((long)a-(long)b); if(u>mu)mu=u;
    }
    float r=(float)sqrt(err/(ref+1e-20));
    printf("%s: relative RMS %.3g, max ulp %ld\n",name,r,mu);
    if(r>limit){ printf("FAIL %s (limit %.3g)\n",name,limit); return 0; }
    return 1;
}

template<int HW>
__global__ static void f8_sweep_kernel(float *out){
    __shared__ float slut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x) slut[i]=c_e4m3[i];
    __syncthreads();
    for(int i=threadIdx.x;i<256;i+=blockDim.x) out[i]=f8_dec<HW>(slut,(uint8_t)i);
}

static int sweep_check(const char *name,const float *got,const float *want){
    int bad=0;
    for(int i=0;i<256;i++){
        /* std:: qualification on host math: ROCm's __clang_hip_cmath.h hijacks
         * unqualified isnan in host code and its float overload fails. */
        if((i&0x7F)==0x7F){ if(!(std::isnan(got[i])&&std::isnan(want[i]))) bad++; }
        else if(memcmp(&got[i],&want[i],4)) bad++;
    }
    printf("decode sweep [%s]: %d mismatches\n",name,bad);
    return bad;
}

/* One full expert group on the GPU (sync path) + the CPU reference chain. */
static void cpu_expert_chain(const uint8_t *g,const uint8_t *u,const uint8_t *d,
        const float *gs,const float *us,const float *ds,int D,int I,
        const float *x,float *y,float *h){
    float *rg=(float*)malloc((size_t)I*4),*ru=(float*)malloc((size_t)I*4);
    cpu_gemv_f8(g,gs,D,I,x,rg);
    cpu_gemv_f8(u,us,D,I,x,ru);
    for(int o=0;o<I;o++) h[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
    cpu_gemv_f8(d,ds,I,D,h,y);
    free(rg); free(ru);
}

#define NEXP 3
int main(void){
    srand(11);
    /* Pin the warp path explicitly: HIP defaults COLI_CUDA_F8_WARP to 0
     * (unvalidated wave64 seam), and this suite exists to validate exactly
     * that path — on both vendors. The old-kernel leg below sets 0 and
     * restores this. */
    setenv("COLI_CUDA_F8_WARP","1",1);
    int devs[1]={0};
    if(!coli_cuda_init(devs,1)){ printf("FAIL cuda init\n"); return 1; }
    float lut[256]; for(int i=0;i<256;i++) lut[i]=e4m3_ref((uint8_t)i);

    /* ---- LUT gate: refusal before publication ------------------------------ */
    {
        uint8_t w[128]; float sc[1]={1.f}; ColiCudaTensor *t=nullptr;
        for(int i=0;i<128;i++) w[i]=rnd_e4m3();
        if(coli_cuda_tensor_upload(&t,w,sc,8,128,1,0)){ printf("FAIL gate open before LUT\n"); return 1; }
    }
    if(!coli_cuda_fp8_set_lut(lut)){ printf("FAIL set_lut\n"); return 1; }

    /* ---- 256-value decode sweep, every compiled path ----------------------- */
    {
        float *dout,got[256]; cudaMalloc(&dout,256*4);
        f8_sweep_kernel<0><<<1,256>>>(dout);
        cudaMemcpy(got,dout,256*4,cudaMemcpyDeviceToHost);
        if(sweep_check("shared-lut",got,lut)){ printf("FAIL\n"); return 1; }
#if COLI_F8_HWCVT
        f8_sweep_kernel<1><<<1,256>>>(dout);
        cudaMemcpy(got,dout,256*4,cudaMemcpyDeviceToHost);
        if(sweep_check("hw-cvt",got,lut)){ printf("FAIL\n"); return 1; }
#else
        printf("decode sweep [hw-cvt]: not compiled on this toolchain\n");
#endif
        /* proof of bite: a single mutated LUT entry must be caught */
        float bad_lut[256]; memcpy(bad_lut,lut,sizeof(lut));
        bad_lut[0x35]*=2.f;
        cudaMemcpyToSymbol(c_e4m3,bad_lut,sizeof(bad_lut));
        f8_sweep_kernel<0><<<1,256>>>(dout);
        cudaMemcpy(got,dout,256*4,cudaMemcpyDeviceToHost);
        int bit=sweep_check("shared-lut, mutated entry (must bite)",got,lut);
        cudaMemcpyToSymbol(c_e4m3,lut,sizeof(lut));
        cudaFree(dout);
        if(bit!=1){ printf("FAIL sweep has no bite\n"); return 1; }
    }

    /* ---- tail shape (O,I not %128): sync group vs CPU + async parity ------- */
    {
        const int D=201,I=300;                     /* blocks: 128+73 and 2x128+44;
                                                    * D%4!=0 also forces the guarded
                                                    * byte path on full blocks */
        const int nbD=(D+127)/128,nbI=(I+127)/128;
        int rows[NEXP]={1,2,5},total=8;            /* total 8: issue path admits it */
        uint8_t *hg[NEXP],*hu[NEXP],*hd[NEXP]; float *hgs[NEXP],*hus[NEXP],*hds[NEXP];
        ColiCudaTensor *tg[NEXP]={},*tu[NEXP]={},*td[NEXP]={};
        for(int c=0;c<NEXP;c++){
            hg[c]=(uint8_t*)malloc((size_t)I*D); hu[c]=(uint8_t*)malloc((size_t)I*D);
            hd[c]=(uint8_t*)malloc((size_t)D*I);
            hgs[c]=(float*)malloc((size_t)nbI*nbD*4); hus[c]=(float*)malloc((size_t)nbI*nbD*4);
            hds[c]=(float*)malloc((size_t)nbD*nbI*4);
            for(size_t i=0;i<(size_t)I*D;i++){ hg[c][i]=rnd_e4m3(); hu[c][i]=rnd_e4m3(); }
            for(size_t i=0;i<(size_t)D*I;i++) hd[c][i]=rnd_e4m3();
            for(int i=0;i<nbI*nbD;i++){ hgs[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
                                        hus[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12); }
            for(int i=0;i<nbD*nbI;i++) hds[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
            if(!coli_cuda_tensor_upload(&tg[c],hg[c],hgs[c],8,D,I,0)||
               !coli_cuda_tensor_upload(&tu[c],hu[c],hus[c],8,D,I,0)||
               !coli_cuda_tensor_upload(&td[c],hd[c],hds[c],8,I,D,0)){
                printf("FAIL tail upload\n"); return 1; }
        }
        float *x=(float*)malloc((size_t)total*D*4),*ys=(float*)malloc((size_t)total*D*4);
        float *want=(float*)malloc((size_t)total*D*4),*h=(float*)malloc((size_t)I*4);
        for(size_t i=0;i<(size_t)total*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(tg,tu,td,rows,NEXP,ys,x)){ printf("FAIL tail sync group\n"); return 1; }
        int off=0;
        for(int c=0;c<NEXP;c++){ for(int s=0;s<rows[c];s++)
            cpu_expert_chain(hg[c],hu[c],hd[c],hgs[c],hus[c],hds[c],D,I,
                             x+(size_t)(off+s)*D,want+(size_t)(off+s)*D,h); off+=rows[c]; }
        if(!rms_named("tail 201x300 grouped vs CPU",ys,want,(size_t)total*D,1e-4f)) return 1;
        if(!coli_cuda_expert_group_issue(tg,tu,td,rows,NEXP,x)){ printf("FAIL tail issue\n"); return 1; }
        const float *ya=coli_cuda_expert_group_take(0);
        if(!ya){ printf("FAIL tail take\n"); return 1; }
        if(memcmp(ya,ys,(size_t)total*D*4)){ printf("FAIL async != sync (bitwise)\n"); return 1; }
        printf("tail async/take parity: bitwise OK\n");
        for(int c=0;c<NEXP;c++){ coli_cuda_tensor_free(tg[c]);coli_cuda_tensor_free(tu[c]);coli_cuda_tensor_free(td[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        free(x);free(ys);free(want);free(h);
    }

    /* ---- census shapes: grouped (new AND old path) + dense, vs CPU --------- */
    {
        const int D=2048,I=6144;                   /* GLM-5.2 expert geometry */
        const int nbD=D/128,nbI=I/128;
        const int NC=2; int rows[NC]={1,3},total=4;
        uint8_t *hg[NC],*hu[NC],*hd[NC]; float *hgs[NC],*hus[NC],*hds[NC];
        ColiCudaTensor *tg[NC]={},*tu[NC]={},*td[NC]={};
        for(int c=0;c<NC;c++){
            hg[c]=(uint8_t*)malloc((size_t)I*D); hu[c]=(uint8_t*)malloc((size_t)I*D);
            hd[c]=(uint8_t*)malloc((size_t)D*I);
            hgs[c]=(float*)malloc((size_t)nbI*nbD*4); hus[c]=(float*)malloc((size_t)nbI*nbD*4);
            hds[c]=(float*)malloc((size_t)nbD*nbI*4);
            for(size_t i=0;i<(size_t)I*D;i++){ hg[c][i]=rnd_e4m3(); hu[c][i]=rnd_e4m3(); }
            for(size_t i=0;i<(size_t)D*I;i++) hd[c][i]=rnd_e4m3();
            for(int i=0;i<nbI*nbD;i++){ hgs[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
                                        hus[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12); }
            for(int i=0;i<nbD*nbI;i++) hds[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
            if(!coli_cuda_tensor_upload(&tg[c],hg[c],hgs[c],8,D,I,0)||
               !coli_cuda_tensor_upload(&tu[c],hu[c],hus[c],8,D,I,0)||
               !coli_cuda_tensor_upload(&td[c],hd[c],hds[c],8,I,D,0)){
                printf("FAIL census upload\n"); return 1; }
        }
        float *x=(float*)malloc((size_t)total*D*4),*ys=(float*)malloc((size_t)total*D*4);
        float *yold=(float*)malloc((size_t)total*D*4);
        float *want=(float*)malloc((size_t)total*D*4),*h=(float*)malloc((size_t)I*4);
        for(size_t i=0;i<(size_t)total*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(tg,tu,td,rows,NC,ys,x)){ printf("FAIL census sync group\n"); return 1; }
        int off=0;
        for(int c=0;c<NC;c++){ for(int s=0;s<rows[c];s++)
            cpu_expert_chain(hg[c],hu[c],hd[c],hgs[c],hus[c],hds[c],D,I,
                             x+(size_t)(off+s)*D,want+(size_t)(off+s)*D,h); off+=rows[c]; }
        if(!rms_named("census 6144x2048/2048x6144 grouped (warp) vs CPU",ys,want,(size_t)total*D,1e-4f)) return 1;
        /* old kernels, same inputs: still inside the CPU bound (not bitwise —
         * the accumulation convention is what changed) */
        setenv("COLI_CUDA_F8_WARP","0",1);
        if(!coli_cuda_expert_group(tg,tu,td,rows,NC,yold,x)){ printf("FAIL census old-path group\n"); return 1; }
        setenv("COLI_CUDA_F8_WARP","1",1);
        if(!rms_named("census grouped (old kernels) vs CPU",yold,want,(size_t)total*D,1e-4f)) return 1;
        rms_named("census grouped old vs new (informational)",yold,ys,(size_t)total*D,1e9f);

        /* dense branch (quant_matmul fmt=8) on the census gate matrix */
        {
            const int S=2;
            float *dx=(float*)malloc((size_t)S*D*4),*dy=(float*)malloc((size_t)S*I*4);
            float *dwant=(float*)malloc((size_t)S*I*4);
            ColiCudaTensor *t=nullptr;
            for(size_t i=0;i<(size_t)S*D;i++) dx[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
            if(!coli_cuda_matmul(&t,dy,dx,hg[0],hgs[0],8,S,D,I,0,0)){ printf("FAIL census dense matmul\n"); return 1; }
            for(int s=0;s<S;s++) cpu_gemv_f8(hg[0],hgs[0],D,I,dx+(size_t)s*D,dwant+(size_t)s*I);
            if(!rms_named("census dense fmt=8 vs CPU",dy,dwant,(size_t)S*I,1e-4f)) return 1;
            coli_cuda_tensor_free(t);
            free(dx);free(dy);free(dwant);
        }

        /* ---- S-invariance: rows=64 in one call vs 64 single-row calls ------ */
        {
            int r64[1]={64},r1[1]={1};
            float *sx=(float*)malloc((size_t)64*D*4),*y64=(float*)malloc((size_t)64*D*4);
            float *y1=(float*)malloc((size_t)D*4);
            for(size_t i=0;i<(size_t)64*D;i++) sx[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
            if(!coli_cuda_expert_group(&tg[0],&tu[0],&td[0],r64,1,y64,sx)){ printf("FAIL S=64 group\n"); return 1; }
            int sbad=0;
            for(int k=0;k<64;k++){
                if(!coli_cuda_expert_group(&tg[0],&tu[0],&td[0],r1,1,y1,sx+(size_t)k*D)){ printf("FAIL S=1 group\n"); return 1; }
                if(memcmp(y1,y64+(size_t)k*D,(size_t)D*4)) sbad++;
            }
            printf("S-invariance (S=64 vs 64x S=1): %d mismatching rows\n",sbad);
            if(sbad){ printf("FAIL\n"); return 1; }
            free(sx);free(y64);free(y1);
        }
        for(int c=0;c<NC;c++){ coli_cuda_tensor_free(tg[c]);coli_cuda_tensor_free(tu[c]);coli_cuda_tensor_free(td[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        free(x);free(ys);free(yold);free(want);free(h);
    }

    /* ---- grouped accumulation-convention bite ------------------------------
     * Cross-block cancellation staged so the reference-mirroring convention
     * (double across blocks) is load-bearing: block partials x scales are
     * exactly {1.0, 2^-25, -1.0}. A float cross-block accumulator absorbs
     * the 2^-25 into 1.0 and cancels to exactly 0 — the down kernel is
     * compared BITWISE against 2^-25, the hidden dual (through its silu
     * epilogue) against nonzero-and-near silu(2^-25)*1. Kernels launched
     * raw so both grouped kernels are pinned regardless of the toggle. */
    {
        uint8_t hwb[384]={}; float hx[384]={};
        hwb[0]=0x38; hwb[128]=0x38; hwb[256]=0x38;   /* e4m3 1.0 in each block */
        hx[0]=1.f; hx[128]=1.f; hx[256]=1.f;
        float cancel[3]={1.f,ldexpf(1.f,-25),-1.f},one[3]={1.f,0.f,0.f};
        void *dw,*dx,*dcs,*dos,*dy,*dg,*du,*ddesc;
        cudaMalloc(&dw,384); cudaMalloc(&dx,384*4);
        cudaMalloc(&dcs,3*4); cudaMalloc(&dos,3*4);
        cudaMalloc(&dy,4); cudaMalloc(&dg,4); cudaMalloc(&du,4);
        cudaMalloc(&ddesc,sizeof(GroupDesc));
        cudaMemcpy(dw,hwb,384,cudaMemcpyHostToDevice);
        cudaMemcpy(dx,hx,384*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dcs,cancel,12,cudaMemcpyHostToDevice);
        cudaMemcpy(dos,one,12,cudaMemcpyHostToDevice);
        GroupDesc hd_={dw,dw,dw,(const float*)dcs,(const float*)dcs,(const float*)dcs,
                       8,8,8,1,0,0,0,0};
        cudaMemcpy(ddesc,&hd_,sizeof(hd_),cudaMemcpyHostToDevice);
        grouped_down_f8w<0><<<dim3(1,1),256>>>((float*)dy,(const float*)dx,
                                               (const GroupDesc*)ddesc,1,384);
        float got=0,expect=ldexpf(1.f,-25);
        if(cudaDeviceSynchronize()!=cudaSuccess){ printf("FAIL accum-bite launch\n"); return 1; }
        cudaMemcpy(&got,dy,4,cudaMemcpyDeviceToHost);
        if(memcmp(&got,&expect,4)){
            printf("FAIL grouped down accumulation convention: got %a want %a\n",got,expect);
            return 1; }
        GroupDesc hh_={dw,dw,dw,(const float*)dcs,(const float*)dos,(const float*)dcs,
                       8,8,8,1,0,0,0,0};
        cudaMemcpy(ddesc,&hh_,sizeof(hh_),cudaMemcpyHostToDevice);
        grouped_hidden_f8w_dual<0><<<dim3(1,1),256>>>((float*)dg,(float*)du,
                (const float*)dx,(const GroupDesc*)ddesc,1,384);
        if(cudaDeviceSynchronize()!=cudaSuccess){ printf("FAIL accum-bite launch 2\n"); return 1; }
        cudaMemcpy(&got,dg,4,cudaMemcpyDeviceToHost);
        float g25=ldexpf(1.f,-25),hexp=(g25/(1.f+expf(-g25)))*1.f;
        if(got==0.f||fabsf(got-hexp)>1e-3f*hexp){
            printf("FAIL grouped hidden accumulation convention: got %a want ~%a\n",got,hexp);
            return 1; }
        printf("grouped accumulation convention: down bitwise 2^-25, hidden %a\n",got);
        cudaFree(dw);cudaFree(dx);cudaFree(dcs);cudaFree(dos);
        cudaFree(dy);cudaFree(dg);cudaFree(du);cudaFree(ddesc);
    }

    /* ---- vec/byte load-path parity (bitwise) --------------------------------
     * f8w_block selects uchar4/float4 loads by runtime pointer alignment but
     * feeds ONE shared product sequence: the two paths must be bit-identical
     * on the same data. Force the byte path by offsetting x one float into a
     * copy (4-byte aligned, 16-byte misaligned) and compare against the
     * aligned run. */
    {
        const int O=8,K=256;
        uint8_t hq[8*256]; float hxv[256],hs[2*2];
        for(size_t i=0;i<sizeof(hq);i++) hq[i]=rnd_e4m3();
        for(int i=0;i<K;i++) hxv[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        for(int i=0;i<4;i++) hs[i]=ldexpf(1.f+rand()/(float)RAND_MAX,-10);
        void *dq,*dxa,*dxm,*dya,*dym,*ds,*ddesc;
        cudaMalloc(&dq,sizeof(hq)); cudaMalloc(&dxa,K*4); cudaMalloc(&dxm,(K+1)*4);
        cudaMalloc(&dya,O*4); cudaMalloc(&dym,O*4); cudaMalloc(&ds,sizeof(hs));
        cudaMalloc(&ddesc,sizeof(GroupDesc));
        cudaMemcpy(dq,hq,sizeof(hq),cudaMemcpyHostToDevice);
        cudaMemcpy(dxa,hxv,K*4,cudaMemcpyHostToDevice);
        cudaMemcpy((float*)dxm+1,hxv,K*4,cudaMemcpyHostToDevice);
        cudaMemcpy(ds,hs,sizeof(hs),cudaMemcpyHostToDevice);
        GroupDesc pd={dq,dq,dq,(const float*)ds,(const float*)ds,(const float*)ds,
                      8,8,8,1,0,0,0,0};
        cudaMemcpy(ddesc,&pd,sizeof(pd),cudaMemcpyHostToDevice);
        grouped_down_f8w<0><<<dim3(1,1),256>>>((float*)dya,(const float*)dxa,
                                               (const GroupDesc*)ddesc,O,K);
        grouped_down_f8w<0><<<dim3(1,1),256>>>((float*)dym,(const float*)dxm+1,
                                               (const GroupDesc*)ddesc,O,K);
        if(cudaDeviceSynchronize()!=cudaSuccess){ printf("FAIL parity launch\n"); return 1; }
        float ya[O],ym[O];
        cudaMemcpy(ya,dya,O*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(ym,dym,O*4,cudaMemcpyDeviceToHost);
        if(memcmp(ya,ym,sizeof(ya))){ printf("FAIL vec/byte load paths differ bitwise\n"); return 1; }
        printf("vec/byte load-path parity: bitwise OK\n");
        cudaFree(dq);cudaFree(dxa);cudaFree(dxm);cudaFree(dya);cudaFree(dym);
        cudaFree(ds);cudaFree(ddesc);
    }

    /* ---- mixed group (fmt=8 + fmt=2): fallback equals per-expert MLP ------- */
    {
        const int D=256,I=128; const int nbD=D/128,nbI=I/128;
        int rows[2]={2,1},total=3;
        uint8_t *f8g=(uint8_t*)malloc((size_t)I*D),*f8u=(uint8_t*)malloc((size_t)I*D),
                *f8d=(uint8_t*)malloc((size_t)D*I);
        float *f8gs=(float*)malloc((size_t)nbI*nbD*4),*f8us=(float*)malloc((size_t)nbI*nbD*4),
              *f8ds=(float*)malloc((size_t)nbD*nbI*4);
        for(size_t i=0;i<(size_t)I*D;i++){ f8g[i]=rnd_e4m3(); f8u[i]=rnd_e4m3(); }
        for(size_t i=0;i<(size_t)D*I;i++) f8d[i]=rnd_e4m3();
        for(int i=0;i<nbI*nbD;i++){ f8gs[i]=ldexpf(1.f,-10); f8us[i]=ldexpf(1.f,-10); }
        for(int i=0;i<nbD*nbI;i++) f8ds[i]=ldexpf(1.f,-10);
        uint8_t *i4g=(uint8_t*)malloc((size_t)I*((D+1)/2)),*i4u=(uint8_t*)malloc((size_t)I*((D+1)/2)),
                *i4d=(uint8_t*)malloc((size_t)D*((I+1)/2));
        float *i4gs=(float*)malloc((size_t)I*4),*i4us=(float*)malloc((size_t)I*4),
              *i4ds=(float*)malloc((size_t)D*4);
        for(size_t i=0;i<(size_t)I*((D+1)/2);i++){ i4g[i]=(uint8_t)(rand()&255); i4u[i]=(uint8_t)(rand()&255); }
        for(size_t i=0;i<(size_t)D*((I+1)/2);i++) i4d[i]=(uint8_t)(rand()&255);
        for(int i=0;i<I;i++){ i4gs[i]=ldexpf(1.f,-8); i4us[i]=ldexpf(1.f,-8); }
        for(int i=0;i<D;i++) i4ds[i]=ldexpf(1.f,-8);
        ColiCudaTensor *tg[2]={},*tu[2]={},*td[2]={};
        if(!coli_cuda_tensor_upload(&tg[0],f8g,f8gs,8,D,I,0)||
           !coli_cuda_tensor_upload(&tu[0],f8u,f8us,8,D,I,0)||
           !coli_cuda_tensor_upload(&td[0],f8d,f8ds,8,I,D,0)||
           !coli_cuda_tensor_upload(&tg[1],i4g,i4gs,2,D,I,0)||
           !coli_cuda_tensor_upload(&tu[1],i4u,i4us,2,D,I,0)||
           !coli_cuda_tensor_upload(&td[1],i4d,i4ds,2,I,D,0)){
            printf("FAIL mixed upload\n"); return 1; }
        float *x=(float*)malloc((size_t)total*D*4),*yg=(float*)malloc((size_t)total*D*4),
              *ym=(float*)malloc((size_t)total*D*4);
        for(size_t i=0;i<(size_t)total*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(tg,tu,td,rows,2,yg,x)){ printf("FAIL mixed group\n"); return 1; }
        if(!coli_cuda_expert_mlp(tg[0],tu[0],td[0],ym,x,rows[0])||
           !coli_cuda_expert_mlp(tg[1],tu[1],td[1],ym+(size_t)rows[0]*D,x+(size_t)rows[0]*D,rows[1])){
            printf("FAIL mixed per-expert MLP\n"); return 1; }
        if(memcmp(yg,ym,(size_t)total*D*4)){ printf("FAIL mixed group != per-expert MLP (bitwise)\n"); return 1; }
        printf("mixed fmt=8/fmt=2 group falls back to per-expert MLP: bitwise OK\n");
        for(int c=0;c<2;c++){ coli_cuda_tensor_free(tg[c]);coli_cuda_tensor_free(tu[c]);coli_cuda_tensor_free(td[c]); }
        free(f8g);free(f8u);free(f8d);free(f8gs);free(f8us);free(f8ds);
        free(i4g);free(i4u);free(i4d);free(i4gs);free(i4us);free(i4ds);
        free(x);free(yg);free(ym);
    }

    /* ---- NaN-byte injection: planted 0x7F reaches the output as NaN -------- */
    {
        const int D=256,I=128; const int nbD=D/128,nbI=I/128;
        int rows[1]={2},total=2;
        uint8_t *g=(uint8_t*)malloc((size_t)I*D),*u=(uint8_t*)malloc((size_t)I*D),
                *d=(uint8_t*)malloc((size_t)D*I);
        float *gs=(float*)malloc((size_t)nbI*nbD*4),*us=(float*)malloc((size_t)nbI*nbD*4),
              *ds=(float*)malloc((size_t)nbD*nbI*4);
        for(size_t i=0;i<(size_t)I*D;i++){ g[i]=rnd_e4m3(); u[i]=rnd_e4m3(); }
        for(size_t i=0;i<(size_t)D*I;i++) d[i]=rnd_e4m3();
        for(int i=0;i<nbI*nbD;i++){ gs[i]=ldexpf(1.f,-10); us[i]=ldexpf(1.f,-10); }
        for(int i=0;i<nbD*nbI;i++) ds[i]=ldexpf(1.f,-10);
        d[(size_t)3*I+7]=0x7F;                       /* down row 3: output col 3 NaN */
        ColiCudaTensor *tg=nullptr,*tu=nullptr,*td=nullptr;
        if(!coli_cuda_tensor_upload(&tg,g,gs,8,D,I,0)||
           !coli_cuda_tensor_upload(&tu,u,us,8,D,I,0)||
           !coli_cuda_tensor_upload(&td,d,ds,8,I,D,0)){ printf("FAIL nan upload\n"); return 1; }
        float *x=(float*)malloc((size_t)total*D*4),*y=(float*)malloc((size_t)total*D*4),
              *want=(float*)malloc((size_t)total*D*4),*h=(float*)malloc((size_t)I*4);
        for(size_t i=0;i<(size_t)total*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(&tg,&tu,&td,rows,1,y,x)){ printf("FAIL nan group\n"); return 1; }
        for(int s=0;s<total;s++)
            cpu_expert_chain(g,u,d,gs,us,ds,D,I,x+(size_t)s*D,want+(size_t)s*D,h);
        int nbad=0;
        for(size_t i=0;i<(size_t)total*D;i++)
            if(std::isnan(y[i])!=std::isnan(want[i])) nbad++;
        for(int s=0;s<total;s++) if(!std::isnan(y[(size_t)s*D+3])) nbad++;
        printf("NaN injection (0x7F in down row 3): %d policy mismatches\n",nbad);
        if(nbad){ printf("FAIL\n"); return 1; }
        coli_cuda_tensor_free(tg);coli_cuda_tensor_free(tu);coli_cuda_tensor_free(td);
        free(g);free(u);free(d);free(gs);free(us);free(ds);free(x);free(y);free(want);free(h);
    }

    /* ---- denormal-scale contribution (dense path, bit-exact vs CPU) -------- */
    {
        uint8_t w[128]={}; float sc[1]={ldexpf(1.f,-120)};
        float x[128]={},y[1]={-1.f};
        w[0]=0x01; x[0]=1.f;                          /* 2^-9 * 2^-120 = 2^-129: f32 denormal */
        ColiCudaTensor *t=nullptr;
        if(!coli_cuda_matmul(&t,y,x,w,sc,8,1,128,1,0,0)){ printf("FAIL denormal matmul\n"); return 1; }
        float expect=(float)((double)(e4m3_ref(0x01)*1.f)*sc[0]);
        if(expect==0.f||memcmp(&y[0],&expect,4)){
            printf("FAIL denormal contribution: got %a want %a (FTZ contamination?)\n",y[0],expect);
            return 1;
        }
        printf("denormal-scale contribution: bit-exact (%a)\n",y[0]);
        coli_cuda_tensor_free(t);
    }

    /* ---- dense accumulation-convention bite ---------------------------------
     * The dense variant of the grouped bite, through the real entry point
     * (coli_cuda_matmul -> quant_matmul_f8w with the suite's pinned mode):
     * block partials x scales staged as {1.0, 2^-25, -1.0}, result asserted
     * BITWISE == 2^-25. In quant_matmul_f8w the three blocks land on three
     * warps, so a double->float degradation of either the per-warp
     * accumulator or the cross-warp dsum sum absorbs the 2^-25 into 1.0 and
     * cancels to exactly 0. (The denormal test above is single-block and
     * cannot see this; the census checks are RMS-level.) */
    {
        uint8_t w[384]={}; float sc[3]={1.f,ldexpf(1.f,-25),-1.f};
        float x[384]={},y[1]={-1.f};
        w[0]=0x38; w[128]=0x38; w[256]=0x38;          /* e4m3 1.0 in each block */
        x[0]=1.f; x[128]=1.f; x[256]=1.f;
        ColiCudaTensor *t=nullptr;
        if(!coli_cuda_matmul(&t,y,x,w,sc,8,1,384,1,0,0)){ printf("FAIL dense accum-bite matmul\n"); return 1; }
        float expect=ldexpf(1.f,-25);
        if(memcmp(&y[0],&expect,4)){
            printf("FAIL dense accumulation convention: got %a want %a\n",y[0],expect);
            return 1;
        }
        printf("dense accumulation convention: bitwise 2^-25\n");
        coli_cuda_tensor_free(t);
    }

    coli_cuda_shutdown();
    printf("OK\n"); return 0;
}
