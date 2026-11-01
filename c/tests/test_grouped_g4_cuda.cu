/* Grouped-int4 (fmt=4) CUDA kernel oracle (#334).
 *
 * Feeds random offset-binary nibble weights + [O, ng] group scales through
 * grouped_hidden_g4_dual / grouped_down_g4 and checks against a CPU reference
 * that replicates matmul_i4_grouped's semantics (value = nibble - 8, per-group
 * partial dot x scale). Covers gs=64, a non-divisible tail group, and a
 * per-row (gs=0) member riding in the same launch — the fmt=2-compat case.
 * The dual hidden kernel fuses silu into its epilogue, so gate[] holds
 * silu(g)*u and up[] is never written; the oracle checks that contract.
 *
 * The device buffers get the same XOR 0x88 offset->signed conversion the
 * upload path applies, so the kernels are exercised exactly as deployed.
 *
 * Phase 2 goes through the public API instead of raw kernels: upload_g plus
 * coli_cuda_expert_group (sync) and coli_cuda_expert_group_issue/take (the
 * async decode path the VRAM expert tiers ride) against the same reference —
 * the async result must be bit-identical to the sync one. Phase 3 does the same
 * with groups of one format, which take the chain's decode kernels.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_grouped_g4_cuda.cu -o tests/test_grouped_g4
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#if !defined(__HIPCC__)
#include <cuda_runtime.h>   /* under HIP, backend_gpu_compat.h (via backend_cuda.cu) provides these */
#endif

#include "../backend_cuda.cu"

static void cpu_gemv_g4(const uint8_t *q,const float *sc,int K,int O,int gs,
                        const float *x,float *y){
    int rb=(K+1)/2, ng=gs>0?(K+gs-1)/gs:1, egs=gs>0?gs:K;
    for(int o=0;o<O;o++){
        const uint8_t *row=q+(size_t)o*rb; const float *scl=sc+(size_t)o*ng;
        double a=0;
        for(int g=0; g*egs<K; g++){
            int base=g*egs, glen=egs; if(base+glen>K) glen=K-base;
            double p=0;
            for(int i=base;i<base+glen;i++){
                uint8_t v=row[i>>1]; int n=(i&1)?(v>>4):(v&15);
                p+=(double)x[i]*(n-8);
            }
            a+=p*scl[g];
        }
        y[o]=(float)a;
    }
}

int main(void){
    srand(7);
    const int D=200, I=96, gs=64;            /* tail group: 200 % 64 = 8 */
    const int COUNT=3;                       /* expert 0,1: fmt4 gs=64; expert 2: per-row (gs=0) */
    const int rbD=(D+1)/2, rbI=(I+1)/2;
    const int ngD=(D+gs-1)/gs, ngI=(I+gs-1)/gs;
    int trials=50, bad=0;
    for(int t=0;t<trials;t++){
        GroupDesc host[COUNT]; float *xs; cudaMallocManaged(&xs,(size_t)COUNT*D*4);
        float *gate,*up,*y; cudaMallocManaged(&gate,(size_t)COUNT*I*4);
        cudaMallocManaged(&up,(size_t)COUNT*I*4); cudaMallocManaged(&y,(size_t)COUNT*D*4);
        uint8_t *qg[COUNT],*qu[COUNT],*qd[COUNT]; float *sg[COUNT],*su[COUNT],*sd[COUNT];
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        for(int c=0;c<COUNT;c++){
            int cgs = c==2 ? 0 : gs;
            int cngD = cgs? ngD:1, cngI = cgs? ngI:1;
            hg[c]=(uint8_t*)malloc((size_t)I*rbD); hu[c]=(uint8_t*)malloc((size_t)I*rbD);
            hd[c]=(uint8_t*)malloc((size_t)D*rbI);
            hgs[c]=(float*)malloc((size_t)I*cngD*4); hus[c]=(float*)malloc((size_t)I*cngD*4);
            hds[c]=(float*)malloc((size_t)D*cngI*4);
            for(size_t i=0;i<(size_t)I*rbD;i++){ hg[c][i]=rand()&255; hu[c][i]=rand()&255; }
            for(size_t i=0;i<(size_t)D*rbI;i++) hd[c][i]=rand()&255;
            for(size_t i=0;i<(size_t)I*cngD;i++){ hgs[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
                                                  hus[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); }
            for(size_t i=0;i<(size_t)D*cngI;i++) hds[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
            cudaMalloc(&qg[c],(size_t)I*rbD); cudaMalloc(&qu[c],(size_t)I*rbD); cudaMalloc(&qd[c],(size_t)D*rbI);
            cudaMalloc(&sg[c],(size_t)I*cngD*4); cudaMalloc(&su[c],(size_t)I*cngD*4); cudaMalloc(&sd[c],(size_t)D*cngI*4);
            cudaMemcpy(qg[c],hg[c],(size_t)I*rbD,cudaMemcpyHostToDevice);
            cudaMemcpy(qu[c],hu[c],(size_t)I*rbD,cudaMemcpyHostToDevice);
            cudaMemcpy(qd[c],hd[c],(size_t)D*rbI,cudaMemcpyHostToDevice);
            offset_to_signed_s4<<<64,256>>>(qg[c],(size_t)I*rbD);
            offset_to_signed_s4<<<64,256>>>(qu[c],(size_t)I*rbD);
            offset_to_signed_s4<<<64,256>>>(qd[c],(size_t)D*rbI);
            cudaMemcpy(sg[c],hgs[c],(size_t)I*cngD*4,cudaMemcpyHostToDevice);
            cudaMemcpy(su[c],hus[c],(size_t)I*cngD*4,cudaMemcpyHostToDevice);
            cudaMemcpy(sd[c],hds[c],(size_t)D*cngI*4,cudaMemcpyHostToDevice);
            host[c]={qg[c],qu[c],qd[c],sg[c],su[c],sd[c],4,4,4,1,c,cgs,cgs,cgs};
        }
        for(size_t i=0;i<(size_t)COUNT*D;i++) xs[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        GroupDesc *ddesc; cudaMalloc(&ddesc,sizeof(host));
        cudaMemcpy(ddesc,host,sizeof(host),cudaMemcpyHostToDevice);
        dim3 hgd((unsigned)I,1,(unsigned)COUNT),ogd((unsigned)D,1,(unsigned)COUNT);
        grouped_hidden_g4_dual<<<hgd,256>>>(gate,up,xs,ddesc,I,D);
        grouped_down_g4<<<ogd,256>>>(y,gate,ddesc,D,I);
        if(cudaDeviceSynchronize()!=cudaSuccess){ printf("FAIL cuda\n"); return 1; }
        for(int c=0;c<COUNT;c++){
            int cgs=c==2?0:gs;
            float rg[512],ru[512],rh[512],ry[512];
            cpu_gemv_g4(hg[c],hgs[c],D,I,cgs,xs+(size_t)c*D,rg);
            cpu_gemv_g4(hu[c],hus[c],D,I,cgs,xs+(size_t)c*D,ru);
            /* fused epilogue (a03c79e): gate[] = silu(g)*u, up[] untouched */
            for(int o=0;o<I;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
            for(int o=0;o<I;o++)
                if(fabsf(gate[(size_t)c*I+o]-rh[o])>1e-3f*(fabsf(rh[o])+1e-3f)) bad++;
            cpu_gemv_g4(hd[c],hds[c],I,D,cgs,(float*)gate+(size_t)c*I,ry);
            for(int o=0;o<D;o++)
                if(fabsf(y[(size_t)c*D+o]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) bad++;
        }
        for(int c=0;c<COUNT;c++){ cudaFree(qg[c]);cudaFree(qu[c]);cudaFree(qd[c]);
            cudaFree(sg[c]);cudaFree(su[c]);cudaFree(sd[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        cudaFree(ddesc);cudaFree(xs);cudaFree(gate);cudaFree(up);cudaFree(y);
    }
    printf("grouped-g4 oracle: %d trials x %d experts (gs=64 + tail + per-row member), %d mismatches\n",
           trials,COUNT,bad);
    if(bad){ printf("FAIL\n"); return 1; }

    /* ---- Phase 2: public API — upload_g + sync group vs async issue/take ----
     * The async path is what MoE decode (and the VRAM expert tiers) actually
     * call; it must dispatch fmt=4 like the sync path and match it bit for bit. */
    {
        int devs[1]={0};
        if(!coli_cuda_init(devs,1)){ printf("FAIL cuda init\n"); return 1; }
        int rows[COUNT]={1,2,1}, total=4, api_bad=0;
        ColiCudaTensor *tg[COUNT]={},*tu[COUNT]={},*td[COUNT]={};
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        for(int c=0;c<COUNT;c++){
            int cgs = c==2 ? 0 : gs;            /* expert 2 rides along as fmt=2 */
            int cngD = cgs? ngD:1, cngI = cgs? ngI:1, fmt = cgs?4:2;
            hg[c]=(uint8_t*)malloc((size_t)I*rbD); hu[c]=(uint8_t*)malloc((size_t)I*rbD);
            hd[c]=(uint8_t*)malloc((size_t)D*rbI);
            hgs[c]=(float*)malloc((size_t)I*cngD*4); hus[c]=(float*)malloc((size_t)I*cngD*4);
            hds[c]=(float*)malloc((size_t)D*cngI*4);
            for(size_t i=0;i<(size_t)I*rbD;i++){ hg[c][i]=rand()&255; hu[c][i]=rand()&255; }
            for(size_t i=0;i<(size_t)D*rbI;i++) hd[c][i]=rand()&255;
            for(size_t i=0;i<(size_t)I*cngD;i++){ hgs[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
                                                  hus[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); }
            for(size_t i=0;i<(size_t)D*cngI;i++) hds[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
            if(!coli_cuda_tensor_upload_g(&tg[c],hg[c],hgs[c],fmt,D,I,0,cgs)||
               !coli_cuda_tensor_upload_g(&tu[c],hu[c],hus[c],fmt,D,I,0,cgs)||
               !coli_cuda_tensor_upload_g(&td[c],hd[c],hds[c],fmt,I,D,0,cgs)){
                printf("FAIL upload_g\n"); return 1; }
        }
        float *x=(float*)malloc((size_t)total*D*4),*ysync=(float*)malloc((size_t)total*D*4);
        for(size_t i=0;i<(size_t)total*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(tg,tu,td,rows,COUNT,ysync,x)){ printf("FAIL sync group\n"); return 1; }
        if(!coli_cuda_expert_group_issue(tg,tu,td,rows,COUNT,x)){ printf("FAIL async issue\n"); return 1; }
        const float *yasync=coli_cuda_expert_group_take(0);
        if(!yasync){ printf("FAIL async take\n"); return 1; }
        int off=0;
        for(int c=0;c<COUNT;c++){
            int cgs=c==2?0:gs;
            for(int s=0;s<rows[c];s++){
                float rg[512],ru[512],rh[512],ry[512];
                const float *xr=x+(size_t)(off+s)*D;
                cpu_gemv_g4(hg[c],hgs[c],D,I,cgs,xr,rg);
                cpu_gemv_g4(hu[c],hus[c],D,I,cgs,xr,ru);
                for(int o=0;o<I;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
                cpu_gemv_g4(hd[c],hds[c],I,D,cgs,rh,ry);
                for(int o=0;o<D;o++){
                    size_t z=(size_t)(off+s)*D+o;
                    if(fabsf(ysync[z]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) api_bad++;
                    if(yasync[z]!=ysync[z]) api_bad++;   /* async == sync, bitwise */
                }
            }
            off+=rows[c];
        }
        printf("grouped-g4 API: sync vs oracle + async(issue/take) vs sync, %d mismatches\n",api_bad);
        for(int c=0;c<COUNT;c++){ coli_cuda_tensor_free(tg[c]);coli_cuda_tensor_free(tu[c]);coli_cuda_tensor_free(td[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        free(x);free(ysync);
        coli_cuda_shutdown();
        if(api_bad){ printf("FAIL\n"); return 1; }
    }

    /* ---- Phase 3: a decode group whose experts share one format (int4 in groups of 64,
     * then int4 with a scale a row) takes the chain's decode kernels (group_moe_launch,
     * widths multiples of 32): sync against the oracle, async against sync bit for bit ---- */
    for(int fmt=4; fmt>=2; fmt-=2){
        const int D3=256, I3=128, gs3=fmt==4?64:0, rb3=D3/2, rbi3=I3/2;
        const int ng3=gs3?D3/gs3:1, ngi3=gs3?I3/gs3:1;
        int devs[1]={0};
        if(!coli_cuda_init(devs,1)){ printf("FAIL cuda init\n"); return 1; }
        int rows[COUNT]={1,2,1}, total=4, bad3=0;
        /* a prompt chunk's group (more than 8 rows: the tensor cores' sorted GEMMs, f16 or
         * TF32 operands) from the same experts, against the oracle within those roundings */
        int rowsb[COUNT]={20,33,15}, totalb=68, bad4=0;
        ColiCudaTensor *tg[COUNT]={},*tu[COUNT]={},*td[COUNT]={};
        uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
        for(int c=0;c<COUNT;c++){
            hg[c]=(uint8_t*)malloc((size_t)I3*rb3); hu[c]=(uint8_t*)malloc((size_t)I3*rb3); hd[c]=(uint8_t*)malloc((size_t)D3*rbi3);
            hgs[c]=(float*)malloc((size_t)I3*ng3*4); hus[c]=(float*)malloc((size_t)I3*ng3*4); hds[c]=(float*)malloc((size_t)D3*ngi3*4);
            for(size_t i=0;i<(size_t)I3*rb3;i++){ hg[c][i]=rand()&255; hu[c][i]=rand()&255; }
            for(size_t i=0;i<(size_t)D3*rbi3;i++) hd[c][i]=rand()&255;
            for(size_t i=0;i<(size_t)I3*ng3;i++){ hgs[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); hus[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); }
            for(size_t i=0;i<(size_t)D3*ngi3;i++) hds[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
            if(!coli_cuda_tensor_upload_g(&tg[c],hg[c],hgs[c],fmt,D3,I3,0,gs3)||
               !coli_cuda_tensor_upload_g(&tu[c],hu[c],hus[c],fmt,D3,I3,0,gs3)||
               !coli_cuda_tensor_upload_g(&td[c],hd[c],hds[c],fmt,I3,D3,0,gs3)){ printf("FAIL upload_g\n"); return 1; }
        }
        float *x=(float*)malloc((size_t)total*D3*4),*ysync=(float*)malloc((size_t)total*D3*4);
        for(size_t i=0;i<(size_t)total*D3;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        uint64_t m0=g_group_moe_calls;
        if(!coli_cuda_expert_group(tg,tu,td,rows,COUNT,ysync,x)){ printf("FAIL sync group\n"); return 1; }
        if(!coli_cuda_expert_group_issue(tg,tu,td,rows,COUNT,x)){ printf("FAIL async issue\n"); return 1; }
        const float *yasync=coli_cuda_expert_group_take(0);
        if(!yasync){ printf("FAIL async take\n"); return 1; }
        int took=(int)(g_group_moe_calls-m0);
        for(int c=0,off=0;c<COUNT;off+=rows[c],c++)
            for(int s=0;s<rows[c];s++){
                float rg[256],ru[256],rh[256],ry[256];
                const float *xr=x+(size_t)(off+s)*D3;
                cpu_gemv_g4(hg[c],hgs[c],D3,I3,gs3,xr,rg); cpu_gemv_g4(hu[c],hus[c],D3,I3,gs3,xr,ru);
                for(int o=0;o<I3;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
                cpu_gemv_g4(hd[c],hds[c],I3,D3,gs3,rh,ry);
                for(int o=0;o<D3;o++){
                    size_t z=(size_t)(off+s)*D3+o;
                    if(fabsf(ysync[z]-ry[o])>1e-3f*(fabsf(ry[o])+1e-3f)) bad3++;
                    if(yasync[z]!=ysync[z]) bad3++;
                }
            }
        printf("decode group fmt %d on the chain's kernels: %d of 2 groups took them, %d mismatches\n",fmt,took,bad3);
        {
            float *xb=(float*)malloc((size_t)totalb*D3*4),*yb=(float*)malloc((size_t)totalb*D3*4);
            for(size_t i=0;i<(size_t)totalb*D3;i++) xb[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
            uint64_t mb=g_group_moe_calls;
            if(!coli_cuda_expert_group(tg,tu,td,rowsb,COUNT,yb,xb)){ printf("FAIL prompt group\n"); return 1; }
            int tookb=(int)(g_group_moe_calls-mb); double worst=0;
            for(int c=0,off=0;c<COUNT;off+=rowsb[c],c++)
                for(int s=0;s<rowsb[c];s++){
                    float rg[256],ru[256],rh[256],ry[256];
                    const float *xr=xb+(size_t)(off+s)*D3;
                    cpu_gemv_g4(hg[c],hgs[c],D3,I3,gs3,xr,rg); cpu_gemv_g4(hu[c],hus[c],D3,I3,gs3,xr,ru);
                    for(int o=0;o<I3;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
                    cpu_gemv_g4(hd[c],hds[c],I3,D3,gs3,rh,ry);
                    double sc=0; for(int o=0;o<D3;o++) if(fabs(ry[o])>sc) sc=fabs(ry[o]);
                    for(int o=0;o<D3;o++){ double e=fabs(yb[(size_t)(off+s)*D3+o]-ry[o])/(sc>0?sc:1); if(!(e<=worst)) worst=e; }
                }
            if(!(worst<2e-2)) bad4++;
            printf("prompt group fmt %d (%d rows) on the chain's kernels: %d of 1 took them, worst %.2e of the row's largest\n",fmt,totalb,tookb,worst);
            const char *ge4=getenv("COLI_CUDA_GROUP_MOE");
            if(tookb!=1&&!(ge4&&*ge4=='0')) bad4++;
            /* a speculation's verify (pinned): every row the bits it has alone, as a decode step */
            int same=1;
            if(!coli_cuda_expert_group_pinned(tg,tu,td,rowsb,COUNT,yb,xb,1)){ printf("FAIL pinned group\n"); return 1; }
            for(int c=0,off=0;c<COUNT&&same;off+=rowsb[c],c++)
                for(int s=0;s<rowsb[c]&&same;s++){
                    float y1[256]; int one=1;
                    ColiCudaTensor *g1[1]={tg[c]},*u1[1]={tu[c]},*d1[1]={td[c]};
                    if(!coli_cuda_expert_group(g1,u1,d1,&one,1,y1,xb+(size_t)(off+s)*D3)){ same=0; break; }
                    same=!memcmp(y1,yb+(size_t)(off+s)*D3,(size_t)D3*4);
                }
            printf("prompt group fmt %d pinned: every row a decode step's bits: %s\n",fmt,same?"yes":"NO");
            if(!same) bad4++;
            free(xb); free(yb);
        }
        for(int c=0;c<COUNT;c++){ coli_cuda_tensor_free(tg[c]);coli_cuda_tensor_free(tu[c]);coli_cuda_tensor_free(td[c]);
            free(hg[c]);free(hu[c]);free(hd[c]);free(hgs[c]);free(hus[c]);free(hds[c]); }
        free(x);free(ysync);
        coli_cuda_shutdown();
        const char *ge=getenv("COLI_CUDA_GROUP_MOE");
        if(bad3||bad4||(took!=2&&!(ge&&*ge=='0'))){ printf("FAIL\n"); return 1; }
    }
    printf("OK\n"); return 0;
}
