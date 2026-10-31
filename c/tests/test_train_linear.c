/* Milestone 2: toy frozen int4 linear + trainable LoRA.
 *
 * Validates, against a float64 dense reference computed from the DEQUANTIZED
 * weights (exactly what the int4 kernels encode):
 *   - forward parity (real matmul_i4 kernel + LoRA residual);
 *   - dA / dB / dx gradients (analytic, f32 accumulation vs f64 reference);
 *   - dA / dB spot-checked with central finite differences on the C forward;
 *   - AdamW: 300 steps on a toy regression must cut the loss by >5x, no NaN.
 * Never allocates dW for the frozen base.
 *
 * TRAIN_REF_DUMP=<file>: writes a JSON fixture (inputs + C gradients) that
 * tools/ref_train_linear.py re-checks with PyTorch float64 autograd. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "../train/qlora_ops.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static uint32_t rng_state=0x1234567u;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }
static float fr(void){ return (float)(int32_t)xr()/2147483648.0f; }  /* [-1,1) */

enum { S=5, O=24, I=32, RANK=4 };
static const float ALPHA=8.f;

/* dequantize the int4 QT row-major into d[O*I] (f64): the reference operates on
 * the SAME numbers the kernel uses, so quantization error cancels out. */
static void dequant_f64(const QT *w, double *d){
    int rb=(I+1)/2;
    for(int o=0;o<O;o++) for(int i=0;i<I;i++){
        uint8_t b=w->q4[(int64_t)o*rb+(i>>1)];
        int v=(i&1)?((int)(b>>4)-8):((int)(b&0xF)-8);
        d[(int64_t)o*I+i]=(double)v*(double)w->s[o];
    }
}

/* full f64 reference: forward, MSE loss, dA/dB/dx. scale=ALPHA/RANK */
static double ref_all(const double *Wd, const float *A, const float *B,
                      const float *x, const float *tgt,
                      double *dA, double *dB, double *dx, double *yout){
    double scale=(double)ALPHA/RANK;
    double y[S*O], z[S*RANK], dy[S*O], dz[RANK];
    for(int s=0;s<S;s++){
        for(int r=0;r<RANK;r++){ double a=0; for(int i=0;i<I;i++) a+=(double)A[r*I+i]*x[s*I+i]; z[s*RANK+r]=a; }
        for(int o=0;o<O;o++){ double a=0;
            for(int i=0;i<I;i++) a+=Wd[(int64_t)o*I+i]*x[s*I+i];
            for(int r=0;r<RANK;r++) a+=scale*(double)B[o*RANK+r]*z[s*RANK+r];
            y[s*O+o]=a; }
    }
    double loss=0; int64_t n=S*O;
    for(int64_t i=0;i<n;i++){ double d=y[i]-tgt[i]; loss+=d*d; dy[i]=2.0/n*d; }
    loss/=n;
    if(yout) memcpy(yout,y,sizeof(y));
    if(!dA) return loss;
    memset(dA,0,sizeof(double)*RANK*I); memset(dB,0,sizeof(double)*O*RANK);
    memset(dx,0,sizeof(double)*S*I);
    for(int s=0;s<S;s++){
        for(int o=0;o<O;o++){ double c=scale*dy[s*O+o];
            for(int r=0;r<RANK;r++) dB[o*RANK+r]+=c*z[s*RANK+r]; }
        for(int r=0;r<RANK;r++){ double a=0;
            for(int o=0;o<O;o++) a+=(double)B[o*RANK+r]*dy[s*O+o];
            dz[r]=scale*a; }
        for(int r=0;r<RANK;r++) for(int i=0;i<I;i++){
            dA[r*I+i]+=dz[r]*x[s*I+i];
            dx[s*I+i]+=dz[r]*(double)A[r*I+i]; }
        for(int o=0;o<O;o++){ double c=dy[s*O+o];
            for(int i=0;i<I;i++) dx[s*I+i]+=c*Wd[(int64_t)o*I+i]; }
    }
    return loss;
}

/* C forward + loss only (for finite differences): uses the real kernels */
static float c_loss(const QT *w, LoraTensor *t, const float *x, const float *tgt){
    float y[S*O], z[S*RANK];
    matmul_i4(y,x,w->q4,w->s,S,I,O);
    train_lora_fwd(t,x,y,z,S);
    return train_mse(y,tgt,NULL,(int64_t)S*O);
}

static double relerr(double a, double b){
    double m=fabs(a)>fabs(b)?fabs(a):fabs(b);
    return m<1e-9 ? fabs(a-b) : fabs(a-b)/m;
}

int main(void){
    /* frozen base: random f32 -> real int4 pack (never trainable, no dW anywhere) */
    float *Wf=falloc((int64_t)O*I);
    for(int i=0;i<O*I;i++) Wf[i]=fr();
    QT w; memset(&w,0,sizeof(w));
    qt_alloc(&w,O,I,4); qt_fill(&w,Wf,4);
    CHECK(w.fmt==2);
    double *Wd=malloc(sizeof(double)*O*I); dequant_f64(&w,Wd);

    LoraTensor t={.layer=0,.tgt=LORA_T_O,.O=O,.I=I,.rank=RANK,.scale=ALPHA/RANK};
    t.A=falloc((int64_t)RANK*I); t.B=falloc((int64_t)O*RANK);
    for(int i=0;i<RANK*I;i++) t.A[i]=0.1f*fr();
    for(int i=0;i<O*RANK;i++) t.B[i]=0.1f*fr();

    float x[S*I], tgt[S*O];
    for(int i=0;i<S*I;i++) x[i]=fr();
    for(int i=0;i<S*O;i++) tgt[i]=fr();

    /* --- forward parity: real kernel vs f64 dense on dequantized weights --- */
    float y[S*O], z[S*RANK], dy[S*O];
    matmul_i4(y,x,w.q4,w.s,S,I,O);
    train_lora_fwd(&t,x,y,z,S);
    double yref[S*O];
    ref_all(Wd,t.A,t.B,x,tgt,NULL,NULL,NULL,yref);
    for(int i=0;i<S*O;i++) CHECK(relerr(y[i],yref[i])<1e-4);

    /* --- analytic gradients vs f64 reference --- */
    float loss=train_mse(y,tgt,dy,(int64_t)S*O);
    float *dA=falloc((int64_t)RANK*I), *dB=falloc((int64_t)O*RANK), *dx=falloc((int64_t)S*I);
    memset(dA,0,sizeof(float)*RANK*I); memset(dB,0,sizeof(float)*O*RANK); memset(dx,0,sizeof(float)*S*I);
    train_lora_bwd(&t,x,z,dy,dA,dB,dx,S);
    train_qt_bwd_dx(&w,dy,dx,S);

    double *rdA=malloc(sizeof(double)*RANK*I), *rdB=malloc(sizeof(double)*O*RANK), *rdx=malloc(sizeof(double)*S*I);
    double rloss=ref_all(Wd,t.A,t.B,x,tgt,rdA,rdB,rdx,NULL);
    CHECK(relerr(loss,rloss)<1e-4);
    double emax=0;
    for(int i=0;i<RANK*I;i++){ double e=relerr(dA[i],rdA[i]); if(e>emax)emax=e; }
    for(int i=0;i<O*RANK;i++){ double e=relerr(dB[i],rdB[i]); if(e>emax)emax=e; }
    for(int i=0;i<S*I;i++)    { double e=relerr(dx[i],rdx[i]); if(e>emax)emax=e; }
    fprintf(stderr,"analytic vs f64 reference: max relerr %.2e\n",emax);
    CHECK(emax<5e-4);

    /* --- finite differences on the C forward (central, spot check) --- */
    const float h=1e-3f; double fdmax=0;
    for(int k=0;k<8;k++){
        int idx=(int)(xr()%(RANK*I)); float keep=t.A[idx];
        t.A[idx]=keep+h; float lp=c_loss(&w,&t,x,tgt);
        t.A[idx]=keep-h; float lm=c_loss(&w,&t,x,tgt);
        t.A[idx]=keep;
        double fd=((double)lp-lm)/(2.0*h), e=relerr(fd,rdA[idx]); if(e>fdmax)fdmax=e;
    }
    for(int k=0;k<8;k++){
        int idx=(int)(xr()%(O*RANK)); float keep=t.B[idx];
        t.B[idx]=keep+h; float lp=c_loss(&w,&t,x,tgt);
        t.B[idx]=keep-h; float lm=c_loss(&w,&t,x,tgt);
        t.B[idx]=keep;
        double fd=((double)lp-lm)/(2.0*h), e=relerr(fd,rdB[idx]); if(e>fdmax)fdmax=e;
    }
    fprintf(stderr,"finite differences vs reference: max relerr %.2e\n",fdmax);
    CHECK(fdmax<2e-2);   /* f32 forward noise; central diff, toy magnitudes */

    /* --- optional fixture for the PyTorch float64 cross-check --- */
    const char *dump=getenv("TRAIN_REF_DUMP");
    if(dump){
        FILE *f=fopen(dump,"w"); CHECK(f!=NULL);
        #define ARRF(name,p,n) do{ fprintf(f,"\"%s\":[",name); \
            for(int64_t _i=0;_i<(n);_i++) fprintf(f,"%s%.9g",_i?",":"",(double)(p)[_i]); \
            fprintf(f,"]"); }while(0)
        fprintf(f,"{\"S\":%d,\"O\":%d,\"I\":%d,\"rank\":%d,\"alpha\":%g,",S,O,I,RANK,(double)ALPHA);
        ARRF("Wd",Wd,(int64_t)O*I); fprintf(f,",");
        ARRF("A",t.A,(int64_t)RANK*I); fprintf(f,",");
        ARRF("B",t.B,(int64_t)O*RANK); fprintf(f,",");
        ARRF("x",x,(int64_t)S*I); fprintf(f,",");
        ARRF("tgt",tgt,(int64_t)S*O); fprintf(f,",");
        ARRF("dA",dA,(int64_t)RANK*I); fprintf(f,",");
        ARRF("dB",dB,(int64_t)O*RANK); fprintf(f,",");
        ARRF("dx",dx,(int64_t)S*I);
        fprintf(f,"}\n"); fclose(f);
        fprintf(stderr,"fixture written: %s\n",dump);
        #undef ARRF
    }

    /* --- AdamW overfit: loss must fall by >5x in 300 steps, stay finite --- */
    float *mA=falloc(RANK*I),*vA=falloc(RANK*I),*mB=falloc(O*RANK),*vB=falloc(O*RANK);
    memset(mA,0,sizeof(float)*RANK*I); memset(vA,0,sizeof(float)*RANK*I);
    memset(mB,0,sizeof(float)*O*RANK); memset(vB,0,sizeof(float)*O*RANK);
    AdamW opt=adamw_default(1e-2f);
    float loss0=0,lossN=0;
    for(int step=0;step<300;step++){
        matmul_i4(y,x,w.q4,w.s,S,I,O);
        train_lora_fwd(&t,x,y,z,S);
        float l=train_mse(y,tgt,dy,(int64_t)S*O);
        if(step==0) loss0=l;
        lossN=l;
        CHECK(isfinite(l));
        memset(dA,0,sizeof(float)*RANK*I); memset(dB,0,sizeof(float)*O*RANK);
        train_lora_bwd(&t,x,z,dy,dA,dB,NULL,S);
        adamw_tick(&opt);
        CHECK(adamw_step(&opt,t.A,dA,mA,vA,(int64_t)RANK*I)==0);
        CHECK(adamw_step(&opt,t.B,dB,mB,vB,(int64_t)O*RANK)==0);
    }
    fprintf(stderr,"overfit: loss %.4f -> %.4f (%d steps)\n",loss0,lossN,300);
    CHECK(lossN < loss0/5.f);

    /* --- fmt=4 (grouped int4, real-snapshot layout): train_qt_bwd_dx vs a
     * dequantized f64 reference. gs=16 with I=32 -> 2 groups per row, distinct
     * scales, so a per-row-scale bug cannot pass. --- */
    {
        const int GS=16, NG=(I+GS-1)/GS, RB=(I+1)/2;
        QT g; memset(&g,0,sizeof(g));
        g.fmt=4; g.O=O; g.I=I; g.gs=GS;
        g.q4=malloc((size_t)O*RB); g.s=falloc((int64_t)O*NG);
        for(int i=0;i<O*RB;i++) g.q4[i]=(uint8_t)(xr()&0xFF);
        for(int i=0;i<O*NG;i++) g.s[i]=0.01f+0.005f*(float)(xr()%100);
        float gdy[S*O], gdx[S*I]; double ref[S*I];
        for(int i=0;i<S*O;i++) gdy[i]=fr();
        for(int i=0;i<S*I;i++){ gdx[i]=fr(); ref[i]=gdx[i]; }
        for(int s=0;s<S;s++) for(int o=0;o<O;o++){
            double c=gdy[(int64_t)s*O+o];
            for(int i=0;i<I;i++){
                uint8_t b=g.q4[(int64_t)o*RB+(i>>1)];
                int v=(i&1)?((int)(b>>4)-8):((int)(b&0xF)-8);
                ref[(int64_t)s*I+i]+=c*(double)v*(double)g.s[(int64_t)o*NG+i/GS];
            }
        }
        train_qt_bwd_dx(&g,gdy,gdx,S);
        for(int i=0;i<S*I;i++) CHECK(relerr(gdx[i],ref[i])<1e-4);
        fprintf(stderr,"fmt=4 grouped transpose backward: ok\n");
        free(g.q4); free(g.s);
    }

    puts("toy frozen-int4 + LoRA trainer tests: ok");
    return 0;
}
