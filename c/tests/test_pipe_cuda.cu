/* Unit tests for the resident-pipeline primitives (Inc.0).
 * Each primitive is checked against the exact CPU math from glm.c.
 * Build: nvcc backend_cuda.cu tests/test_pipe_cuda.cu -o pipe_test && ./pipe_test */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "../backend_cuda.h"

static uint64_t rng=0x9E3779B97F4A7C15ULL;
static float rndf(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
    return (float)((double)(rng>>11)/9007199254740992.0*2.0-1.0); }

static void ref_rmsnorm(float *out,const float *x,const float *w,int D,float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
static void ref_rope(float *v,int pos,int R,float theta){
    int half=R/2; float in[256]; memcpy(in,v,R*sizeof(float));
    for(int j=0;j<half;j++){
        float inv=powf(theta,-2.0f*j/R);
        float ang=pos*inv, cs=cosf(ang), sn=sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        v[j]=a*cs-b*sn; v[half+j]=b*cs+a*sn;
    }
}
static int close_enough(const float *a,const float *b,size_t n,float tol,const char *what){
    float worst=0;
    for(size_t i=0;i<n;i++){
        float d=fabsf(a[i]-b[i]), m=fabsf(b[i])>1.f?fabsf(b[i]):1.f;
        if(d/m>worst) worst=d/m;
    }
    printf("  %-14s max rel %.3e %s\n",what,worst,worst<=tol?"ok":"FAIL");
    return worst<=tol;
}

int main(void){
    int dev0=0;
    if(!coli_cuda_init(&dev0,1)){ puts("pipe tests: skipped (no CUDA device)"); return 0; }
    int ok=1;
    enum { S=7, D=6144, I=2048, R=64, H=4, QH=192 };

    /* rmsnorm */
    {
        float *x=(float*)malloc((size_t)S*D*4), *w=(float*)malloc(D*4);
        float *got=(float*)malloc((size_t)S*D*4), *ref=(float*)malloc((size_t)S*D*4);
        for(size_t i=0;i<(size_t)S*D;i++) x[i]=rndf()*3;
        for(int i=0;i<D;i++) w[i]=1.f+rndf()*0.1f;
        for(int s=0;s<S;s++) ref_rmsnorm(ref+(size_t)s*D,x+(size_t)s*D,w,D,1e-5f);
        float *xd=(float*)coli_cuda_pipe_alloc(0,(size_t)S*D*4);
        float *wd=(float*)coli_cuda_pipe_alloc(0,D*4);
        float *yd=(float*)coli_cuda_pipe_alloc(0,(size_t)S*D*4);
        ok&=coli_cuda_pipe_upload(0,xd,x,(size_t)S*D*4)&&coli_cuda_pipe_upload(0,wd,w,D*4);
        ok&=coli_cuda_pipe_rmsnorm(0,yd,xd,wd,S,D,1e-5f)&&coli_cuda_pipe_sync(0);
        ok&=coli_cuda_pipe_download(0,yd,got,(size_t)S*D*4);
        ok&=close_enough(got,ref,(size_t)S*D,2e-5f,"rmsnorm");
        coli_cuda_pipe_free(0,xd); coli_cuda_pipe_free(0,wd); coli_cuda_pipe_free(0,yd);
        free(x); free(w); free(got); free(ref);
    }
    /* rope on [S,H,QH] query rows, last R dims per head, plus heads=1 k_rot rows */
    {
        size_t n=(size_t)S*H*QH;
        float *v=(float*)malloc(n*4), *ref=(float*)malloc(n*4);
        int pos[S]; for(int s=0;s<S;s++) pos[s]=s*17+3;
        for(size_t i=0;i<n;i++) v[i]=rndf();
        memcpy(ref,v,n*4);
        for(int s=0;s<S;s++) for(int h=0;h<H;h++)
            ref_rope(ref+((size_t)s*H+h)*QH+(QH-R),pos[s],R,10000.f);
        float *vd=(float*)coli_cuda_pipe_alloc(0,n*4);
        int *pd=(int*)coli_cuda_pipe_alloc(0,S*4);
        ok&=coli_cuda_pipe_upload(0,vd,v,n*4)&&coli_cuda_pipe_upload(0,pd,pos,S*4);
        ok&=coli_cuda_pipe_rope(0,vd,pd,S*H,QH,QH-R,R,H,10000.f)&&coli_cuda_pipe_sync(0);
        ok&=coli_cuda_pipe_download(0,vd,v,n*4);
        ok&=close_enough(v,ref,n,3e-4f,"rope");
        coli_cuda_pipe_free(0,vd); coli_cuda_pipe_free(0,pd);
        free(v); free(ref);
    }
    /* silu-mul + residual add */
    {
        size_t n=(size_t)S*I;
        float *g=(float*)malloc(n*4), *u=(float*)malloc(n*4), *ref=(float*)malloc(n*4);
        for(size_t i=0;i<n;i++){ g[i]=rndf()*4; u[i]=rndf()*4; }
        for(size_t i=0;i<n;i++){ float s=g[i]/(1.f+expf(-g[i])); ref[i]=s*u[i]+u[i]; }
        float *gd=(float*)coli_cuda_pipe_alloc(0,n*4), *ud=(float*)coli_cuda_pipe_alloc(0,n*4);
        ok&=coli_cuda_pipe_upload(0,gd,g,n*4)&&coli_cuda_pipe_upload(0,ud,u,n*4);
        ok&=coli_cuda_pipe_silu_mul(0,gd,ud,n);
        ok&=coli_cuda_pipe_add(0,gd,ud,n)&&coli_cuda_pipe_sync(0);
        ok&=coli_cuda_pipe_download(0,gd,g,n*4);
        ok&=close_enough(g,ref,n,2e-5f,"silu+add");
        coli_cuda_pipe_free(0,gd); coli_cuda_pipe_free(0,ud);
        free(g); free(u); free(ref);
    }
    /* fixed-order rows_add */
    {
        float *x=(float*)calloc((size_t)S*D,4), *p=(float*)malloc((size_t)3*D*4), *ref=(float*)calloc((size_t)S*D,4);
        int rows[3]={1,4,6};
        for(size_t i=0;i<(size_t)3*D;i++) p[i]=rndf();
        for(int b=0;b<3;b++) for(int i=0;i<D;i++) ref[(size_t)rows[b]*D+i]+=p[(size_t)b*D+i];
        float *xd=(float*)coli_cuda_pipe_alloc(0,(size_t)S*D*4);
        float *pd=(float*)coli_cuda_pipe_alloc(0,(size_t)3*D*4);
        int *rd=(int*)coli_cuda_pipe_alloc(0,3*4);
        ok&=coli_cuda_pipe_upload(0,xd,x,(size_t)S*D*4)&&coli_cuda_pipe_upload(0,pd,p,(size_t)3*D*4)&&coli_cuda_pipe_upload(0,rd,rows,3*4);
        ok&=coli_cuda_pipe_rows_add(0,xd,pd,rd,3,D)&&coli_cuda_pipe_sync(0);
        ok&=coli_cuda_pipe_download(0,xd,x,(size_t)S*D*4);
        ok&=close_enough(x,ref,(size_t)S*D,1e-6f,"rows_add");
        coli_cuda_pipe_free(0,xd); coli_cuda_pipe_free(0,pd); coli_cuda_pipe_free(0,rd);
        free(x); free(p); free(ref);
    }
    /* device-input gemm vs host-path coli_cuda_matmul on an int4 tensor */
    {
        int O=I, K=D;
        size_t rb=(size_t)(K+1)/2;
        uint8_t *w4=(uint8_t*)malloc((size_t)O*rb);
        float *sc=(float*)malloc(O*4), *x=(float*)malloc((size_t)S*K*4);
        float *ref=(float*)malloc((size_t)S*O*4), *got=(float*)malloc((size_t)S*O*4);
        for(size_t i=0;i<(size_t)O*rb;i++) w4[i]=(uint8_t)(rng=rng*6364136223846793005ULL+1);
        for(int i=0;i<O;i++) sc[i]=0.01f+0.001f*(i%7);
        for(size_t i=0;i<(size_t)S*K;i++) x[i]=rndf();
        ColiCudaTensor *t=NULL;
        ok&=coli_cuda_matmul(&t,ref,x,w4,sc,2,S,K,O,0);   /* host path = reference */
        float *xd=(float*)coli_cuda_pipe_alloc(0,(size_t)S*K*4);
        float *yd=(float*)coli_cuda_pipe_alloc(0,(size_t)S*O*4);
        ok&=coli_cuda_pipe_upload(0,xd,x,(size_t)S*K*4);
        ok&=coli_cuda_pipe_gemm(t,yd,xd,S)&&coli_cuda_pipe_sync(0);
        ok&=coli_cuda_pipe_download(0,yd,got,(size_t)S*O*4);
        ok&=close_enough(got,ref,(size_t)S*O,1e-6f,"gemm(dev)");
        coli_cuda_pipe_free(0,xd); coli_cuda_pipe_free(0,yd);
        coli_cuda_tensor_free(t);
        free(w4); free(sc); free(x); free(ref); free(got);
    }
    coli_cuda_shutdown();
    if(!ok){ puts("pipe tests: FAIL"); return 1; }
    puts("pipe tests: ok");
    return 0;
}
