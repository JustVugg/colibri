/* fmt=8 (fp8-e4m3) CUDA kernel oracle.
 *
 * Phase 1 feeds random e4m3 bytes + [ceil(O/128), ceil(I/128)] block scales
 * through grouped_hidden_f8_dual / grouped_down_f8 and checks against a CPU
 * reference that replicates matmul_fp8's semantics (LUT decode, one f32 scale
 * per 128x128 block, float accumulate within a block, double across blocks).
 * Dims are chosen so both the row- and column-block axes have a partial tail
 * block. The dual hidden kernel fuses silu into its epilogue (gate[] holds
 * silu(g)*u, up[] is never written), same contract as the w4/g4 duals.
 *
 * The e4m3 reference decoder is computed arithmetically here (sign/exp/mant,
 * OCP E4M3-FN: no infinities, only 0x7F/0xFF are NaN), so it cross-checks the
 * engine-supplied LUT rather than assuming it. NaN bytes are excluded from the
 * random draw: the format's NaN policy is propagation, which a bitwise/relative
 * compare cannot express.
 *
 * Phase 2 goes through the public API: the upload gate must REFUSE fmt=8
 * before coli_cuda_fp8_set_lut publishes the decode table (a kernel against
 * the zero-initialized table would compute silent zeros), then accept it.
 * After that: coli_cuda_matmul (dense quant_matmul branch) vs the reference,
 * coli_cuda_expert_group (sync) vs the reference, and issue/take (async) vs
 * sync, which must match bit for bit.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_fp8_cuda.cu -o tests/test_fp8
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "../backend_cuda.cu"

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

#define COUNT 3
int main(void){
    srand(8);
    const int D=320, I=192;                 /* hidden [I,D]: 2x3 blocks; down [D,I]: 3x2 */
    const int nbD=(D+127)/128, nbI=(I+127)/128;
    const int hs_n=nbI*nbD, ds_n=nbD*nbI;   /* block-scale counts, tails included */
    float lut[256]; for(int i=0;i<256;i++) lut[i]=e4m3_ref((uint8_t)i);

    /* ---- Phase 1: raw kernels vs reference --------------------------------- */
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("[CUDA] device 0: %s, %.1f GB VRAM, sm_%d%d\n",prop.name,
           prop.totalGlobalMem/1e9,prop.major,prop.minor);
    if(cudaMemcpyToSymbol(c_e4m3,lut,sizeof(lut))!=cudaSuccess){ printf("FAIL lut\n"); return 1; }

    int rows[COUNT]={1,2,1}, total=4, bad=0, trials=50;
    for(int t=0;t<trials;t++){
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        GroupDesc host[COUNT]; int off=0;
        for(int c=0;c<COUNT;c++){
            hg[c]=(uint8_t*)malloc((size_t)I*D); hu[c]=(uint8_t*)malloc((size_t)I*D);
            hd[c]=(uint8_t*)malloc((size_t)D*I);
            hgs[c]=(float*)malloc(hs_n*4); hus[c]=(float*)malloc(hs_n*4); hds[c]=(float*)malloc(ds_n*4);
            for(size_t i=0;i<(size_t)I*D;i++){ hg[c][i]=rnd_e4m3(); hu[c][i]=rnd_e4m3(); }
            for(size_t i=0;i<(size_t)D*I;i++) hd[c][i]=rnd_e4m3();
            for(int i=0;i<hs_n;i++){ hgs[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
                                     hus[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12); }
            for(int i=0;i<ds_n;i++) hds[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
            void *dg,*du,*dd,*dgs,*dus,*dds;
            cudaMalloc(&dg,(size_t)I*D); cudaMalloc(&du,(size_t)I*D); cudaMalloc(&dd,(size_t)D*I);
            cudaMalloc(&dgs,hs_n*4); cudaMalloc(&dus,hs_n*4); cudaMalloc(&dds,ds_n*4);
            cudaMemcpy(dg,hg[c],(size_t)I*D,cudaMemcpyHostToDevice);
            cudaMemcpy(du,hu[c],(size_t)I*D,cudaMemcpyHostToDevice);
            cudaMemcpy(dd,hd[c],(size_t)D*I,cudaMemcpyHostToDevice);
            cudaMemcpy(dgs,hgs[c],hs_n*4,cudaMemcpyHostToDevice);
            cudaMemcpy(dus,hus[c],hs_n*4,cudaMemcpyHostToDevice);
            cudaMemcpy(dds,hds[c],ds_n*4,cudaMemcpyHostToDevice);
            host[c]={dg,du,dd,(const float*)dgs,(const float*)dus,(const float*)dds,
                     8,8,8,rows[c],off,0,0,0};
            off+=rows[c];
        }
        float *xs=(float*)malloc((size_t)total*D*4);
        for(size_t i=0;i<(size_t)total*D;i++) xs[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        void *ddesc,*dx,*dgate,*dup,*dy;
        cudaMalloc(&ddesc,sizeof(host)); cudaMalloc(&dx,(size_t)total*D*4);
        cudaMalloc(&dgate,(size_t)total*I*4); cudaMalloc(&dup,(size_t)total*I*4);
        cudaMalloc(&dy,(size_t)total*D*4);
        cudaMemcpy(ddesc,host,sizeof(host),cudaMemcpyHostToDevice);
        cudaMemcpy(dx,xs,(size_t)total*D*4,cudaMemcpyHostToDevice);
        dim3 hgd((unsigned)I,2,COUNT),ogd((unsigned)D,2,COUNT);
        grouped_hidden_f8_dual<<<hgd,256>>>((float*)dgate,(float*)dup,(const float*)dx,
                                            (const GroupDesc*)ddesc,I,D);
        grouped_down_f8<<<ogd,256>>>((float*)dy,(const float*)dgate,(const GroupDesc*)ddesc,D,I);
        if(cudaDeviceSynchronize()!=cudaSuccess){ printf("FAIL cuda\n"); return 1; }
        float *gate=(float*)malloc((size_t)total*I*4),*y=(float*)malloc((size_t)total*D*4);
        cudaMemcpy(gate,dgate,(size_t)total*I*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(y,dy,(size_t)total*D*4,cudaMemcpyDeviceToHost);
        for(int c=0;c<COUNT;c++){
            for(int s=0;s<rows[c];s++){
                float rg[512],ru[512],rh[512],ry[512];
                const float *xr=xs+(size_t)(host[c].offset+s)*D;
                cpu_gemv_f8(hg[c],hgs[c],D,I,xr,rg);
                cpu_gemv_f8(hu[c],hus[c],D,I,xr,ru);
                for(int o=0;o<I;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
                for(int o=0;o<I;o++){
                    float got=gate[(size_t)(host[c].offset+s)*I+o];
                    if(fabsf(got-rh[o])>1e-3f*(fabsf(rh[o])+1e-3f)) bad++;
                }
                cpu_gemv_f8(hd[c],hds[c],I,D,rh,ry);
                for(int o=0;o<D;o++){
                    float got=y[(size_t)(host[c].offset+s)*D+o];
                    if(fabsf(got-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) bad++;
                }
            }
        }
        for(int c=0;c<COUNT;c++){ cudaFree((void*)host[c].g);cudaFree((void*)host[c].u);cudaFree((void*)host[c].d);
            cudaFree((void*)host[c].gs);cudaFree((void*)host[c].us);cudaFree((void*)host[c].ds);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        cudaFree(ddesc);cudaFree(dx);cudaFree(dgate);cudaFree(dup);cudaFree(dy);
        free(xs);free(gate);free(y);
    }
    printf("fp8 oracle: %d trials x %d experts (128-blocks + row/col tails), %d mismatches\n",
           trials,COUNT,bad);
    if(bad){ printf("FAIL\n"); return 1; }

    /* ---- Phase 2: public API — LUT gate, dense matmul, sync group, async ---- */
    {
        int devs[1]={0};
        if(!coli_cuda_init(devs,1)){ printf("FAIL cuda init\n"); return 1; }
        uint8_t *hw=(uint8_t*)malloc((size_t)I*D); float *hsc=(float*)malloc(hs_n*4);
        for(size_t i=0;i<(size_t)I*D;i++) hw[i]=rnd_e4m3();
        for(int i=0;i<hs_n;i++) hsc[i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
        ColiCudaTensor *tt=nullptr;
        /* The gate must hold until the LUT is published: a zero table would
         * make every fmt=8 matmul a silent zero. */
        if(coli_cuda_tensor_upload(&tt,hw,hsc,8,D,I,0)){ printf("FAIL gate open before LUT\n"); return 1; }
        if(!coli_cuda_fp8_set_lut(lut)){ printf("FAIL set_lut\n"); return 1; }
        if(!coli_cuda_tensor_upload(&tt,hw,hsc,8,D,I,0)){ printf("FAIL upload\n"); return 1; }

        int api_bad=0;
        {   /* dense: quant_matmul's fmt=8 branch */
            const int S=2;
            float *x=(float*)malloc((size_t)S*D*4),*y=(float*)malloc((size_t)S*I*4),ry[512];
            for(size_t i=0;i<(size_t)S*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
            if(!coli_cuda_matmul(&tt,y,x,hw,hsc,8,S,D,I,0,0)){ printf("FAIL dense matmul\n"); return 1; }
            for(int s=0;s<S;s++){
                cpu_gemv_f8(hw,hsc,D,I,x+(size_t)s*D,ry);
                for(int o=0;o<I;o++)
                    if(fabsf(y[(size_t)s*I+o]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) api_bad++;
            }
            free(x);free(y);
        }
        int rows2[COUNT]={1,2,1}, total2=4;
        ColiCudaTensor *tg[COUNT]={},*tu[COUNT]={},*td[COUNT]={};
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        for(int c=0;c<COUNT;c++){
            hg[c]=(uint8_t*)malloc((size_t)I*D); hu[c]=(uint8_t*)malloc((size_t)I*D);
            hd[c]=(uint8_t*)malloc((size_t)D*I);
            hgs[c]=(float*)malloc(hs_n*4); hus[c]=(float*)malloc(hs_n*4); hds[c]=(float*)malloc(ds_n*4);
            for(size_t i=0;i<(size_t)I*D;i++){ hg[c][i]=rnd_e4m3(); hu[c][i]=rnd_e4m3(); }
            for(size_t i=0;i<(size_t)D*I;i++) hd[c][i]=rnd_e4m3();
            for(int i=0;i<hs_n;i++){ hgs[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
                                     hus[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12); }
            for(int i=0;i<ds_n;i++) hds[c][i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
            if(!coli_cuda_tensor_upload(&tg[c],hg[c],hgs[c],8,D,I,0)||
               !coli_cuda_tensor_upload(&tu[c],hu[c],hus[c],8,D,I,0)||
               !coli_cuda_tensor_upload(&td[c],hd[c],hds[c],8,I,D,0)){
                printf("FAIL expert upload\n"); return 1; }
        }
        float *x=(float*)malloc((size_t)total2*D*4),*ysync=(float*)malloc((size_t)total2*D*4);
        for(size_t i=0;i<(size_t)total2*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(tg,tu,td,rows2,COUNT,ysync,x)){ printf("FAIL sync group\n"); return 1; }
        if(!coli_cuda_expert_group_issue(tg,tu,td,rows2,COUNT,x)){ printf("FAIL async issue\n"); return 1; }
        const float *yasync=coli_cuda_expert_group_take(0);
        if(!yasync){ printf("FAIL async take\n"); return 1; }
        int off=0;
        for(int c=0;c<COUNT;c++){
            for(int s=0;s<rows2[c];s++){
                float rg[512],ru[512],rh[512],ry[512];
                const float *xr=x+(size_t)(off+s)*D;
                cpu_gemv_f8(hg[c],hgs[c],D,I,xr,rg);
                cpu_gemv_f8(hu[c],hus[c],D,I,xr,ru);
                for(int o=0;o<I;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
                cpu_gemv_f8(hd[c],hds[c],I,D,rh,ry);
                for(int o=0;o<D;o++){
                    size_t z=(size_t)(off+s)*D+o;
                    if(fabsf(ysync[z]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) api_bad++;
                    if(yasync[z]!=ysync[z]) api_bad++;   /* async == sync, bitwise */
                }
            }
            off+=rows2[c];
        }
        printf("fp8 API: LUT gate + dense vs oracle + sync group vs oracle + async vs sync, %d mismatches\n",api_bad);
        coli_cuda_tensor_free(tt);
        for(int c=0;c<COUNT;c++){ coli_cuda_tensor_free(tg[c]);coli_cuda_tensor_free(tu[c]);coli_cuda_tensor_free(td[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        free(x);free(ysync);free(hw);free(hsc);
        if(api_bad){ printf("FAIL\n"); return 1; }

        /* ---- Byte-accounting regression: coli_cuda_tensor_free must undo
         * exactly what upload charged, per scale shape. fmt=8's 128x128 block
         * scale_count ((O+127)/128 blocks, not O rows) is the shape that once
         * made free() over-count and trip the tensor_bytes >= bytes guard,
         * leaving the diagnostic counter stuck non-zero forever; fmt=4 grouped
         * (O*ng row-group scales) is covered alongside since it shares the same
         * free()-side expression. */
        {
            size_t c0,b0,c1,b1;
            coli_cuda_stats(0,&c0,&b0);
            ColiCudaTensor *bt=nullptr;
            uint8_t *bw=(uint8_t*)malloc((size_t)I*D); float *bs=(float*)malloc(hs_n*4);
            for(size_t i=0;i<(size_t)I*D;i++) bw[i]=rnd_e4m3();
            for(int i=0;i<hs_n;i++) bs[i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
            if(!coli_cuda_tensor_upload(&bt,bw,bs,8,D,I,0)){ printf("FAIL byte-check fmt8 upload\n"); return 1; }
            coli_cuda_tensor_free(bt);
            coli_cuda_stats(0,&c1,&b1);
            if(c1!=c0||b1!=b0){
                printf("FAIL byte-check fmt8: count %zu->%zu bytes %zu->%zu (want unchanged)\n",c0,c1,b0,b1);
                return 1;
            }
            free(bw);free(bs);

            const int GI=64,GO=32,GS=32;
            int gng=(GI+GS-1)/GS;
            ColiCudaTensor *gt=nullptr;
            uint8_t *gw=(uint8_t*)malloc((size_t)((GI+1)/2)*GO);
            float *gsc=(float*)malloc((size_t)GO*gng*4);
            for(size_t i=0;i<(size_t)((GI+1)/2)*GO;i++) gw[i]=(uint8_t)rand();
            for(int i=0;i<GO*gng;i++) gsc[i]=ldexpf(1.f+rand()/(float)RAND_MAX,-4);
            coli_cuda_stats(0,&c0,&b0);
            if(!coli_cuda_tensor_upload_g(&gt,gw,gsc,4,GI,GO,0,GS)){ printf("FAIL byte-check fmt4g upload\n"); return 1; }
            coli_cuda_tensor_free(gt);
            coli_cuda_stats(0,&c1,&b1);
            if(c1!=c0||b1!=b0){
                printf("FAIL byte-check fmt4g: count %zu->%zu bytes %zu->%zu (want unchanged)\n",c0,c1,b0,b1);
                return 1;
            }
            free(gw);free(gsc);
            printf("byte accounting: fmt=8 dense + fmt=4 grouped free() restores stats exactly\n");
        }

        /* ---- coli_cuda_tensor_bytes() regression: the live-tensor byte report
         * must equal weight_bytes + scale_count*sizeof(float) exactly (fmt=6
         * excepted: no separate scale buffer, so weight_bytes alone). The old
         * `O * ng` shape over-reported fmt=8 (real footprint is (O+127)/128 * ng
         * block scales, not O*ng) and charged fmt=6 a phantom O*ng*4 scale
         * buffer it never allocates -- this is the same shape bug F4 fixed in
         * tensor_free(), here in the sibling accessor most callers actually use
         * for GPU-tier budget bookkeeping (c/colibri.c). */
        {
            ColiCudaTensor *bt=nullptr;
            uint8_t *bw=(uint8_t*)malloc((size_t)I*D); float *bs=(float*)malloc(hs_n*4);
            for(size_t i=0;i<(size_t)I*D;i++) bw[i]=rnd_e4m3();
            for(int i=0;i<hs_n;i++) bs[i]=ldexpf(1.f+rand()/(float)RAND_MAX,-12);
            if(!coli_cuda_tensor_upload(&bt,bw,bs,8,D,I,0)){ printf("FAIL bytes-check fmt8 upload\n"); return 1; }
            size_t want=bt->weight_bytes+bt->scale_count*sizeof(float);
            size_t got=coli_cuda_tensor_bytes(bt);
            if(got!=want){
                printf("FAIL bytes-check fmt8: got %zu want %zu (weight_bytes %zu + scale_count %zu*4)\n",
                    got,want,bt->weight_bytes,bt->scale_count);
                return 1;
            }
            /* Independent oracle: fmt=8 footprint from quant.h's documented
             * layout -- O*I raw e4m3 bytes + ceil(O/128)*ceil(I/128) f32 block
             * scales (hs_n above) -- no struct fields, so the accessor cannot
             * validate itself against the counts upload happened to store. */
            size_t indep=(size_t)I*D+(size_t)hs_n*sizeof(float);
            if(got!=indep||bt->scale_count!=(size_t)hs_n){
                printf("FAIL bytes-check fmt8 independent: got %zu want %zu (%d*%d + %d*4), scale_count %zu (want %d)\n",
                    got,indep,I,D,hs_n,bt->scale_count,hs_n);
                return 1;
            }
            coli_cuda_tensor_free(bt);
            free(bw);free(bs);

            const int SI=256,SO=32;   /* fmt=6: one exact 256-wide in-block, no tail */
            size_t srb=row_bytes(6,SI);
            ColiCudaTensor *st=nullptr;
            uint8_t *sw=(uint8_t*)malloc(srb*SO);
            for(size_t i=0;i<srb*(size_t)SO;i++) sw[i]=(uint8_t)rand();
            if(!coli_cuda_tensor_upload(&st,sw,nullptr,6,SI,SO,0)){ printf("FAIL bytes-check fmt6 upload\n"); return 1; }
            want=st->weight_bytes;
            got=coli_cuda_tensor_bytes(st);
            if(got!=want||st->scale_count!=0){
                printf("FAIL bytes-check fmt6: got %zu want %zu scale_count %zu (want 0)\n",
                    got,want,st->scale_count);
                return 1;
            }
            coli_cuda_tensor_free(st);
            free(sw);

            const int GI=64,GO=32,GS=32;             /* fmt=4 grouped: unchanged regression */
            int gng=(GI+GS-1)/GS;
            ColiCudaTensor *gt=nullptr;
            uint8_t *gw=(uint8_t*)malloc((size_t)((GI+1)/2)*GO);
            float *gsc=(float*)malloc((size_t)GO*gng*4);
            for(size_t i=0;i<(size_t)((GI+1)/2)*GO;i++) gw[i]=(uint8_t)rand();
            for(int i=0;i<GO*gng;i++) gsc[i]=ldexpf(1.f+rand()/(float)RAND_MAX,-4);
            if(!coli_cuda_tensor_upload_g(&gt,gw,gsc,4,GI,GO,0,GS)){ printf("FAIL bytes-check fmt4g upload\n"); return 1; }
            want=gt->weight_bytes+gt->scale_count*sizeof(float);
            got=coli_cuda_tensor_bytes(gt);
            if(got!=want||gt->scale_count!=(size_t)GO*gng){
                printf("FAIL bytes-check fmt4g: got %zu want %zu scale_count %zu (want %d)\n",
                    got,want,gt->scale_count,GO*gng);
                return 1;
            }
            coli_cuda_tensor_free(gt);
            free(gw);free(gsc);
            printf("tensor_bytes: fmt=8 dense + fmt=6 + fmt=4 grouped report exact footprint\n");
        }
        coli_cuda_shutdown();
    }
    printf("OK\n"); return 0;
}
