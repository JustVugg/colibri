/* Training ops for the QLoRA path (Milestone 2: toy frozen linear + LoRA).
 *
 * Forward (per adapted linear):   y = Q(W) x + s * B(A x)      s = alpha/rank
 * Backward, base frozen — dW is NEVER allocated:
 *   dx += dequant(Q(W))^T dy      (streaming over packed rows, no materialized W)
 *   z   = A x                     (cached by the forward)
 *   dB += s * dy z^T
 *   dz  = s * B^T dy
 *   dA += dz x^T
 *   dx += A^T dz
 * All accumulation in f32 (AGENTS.md §8: start with f32 accumulators).
 *
 * Depends on the QT tensor + LoraTensor types: include AFTER glm.c / lora.h
 * (the toy tests include glm.c wholesale, the same pattern as test_idot.c). */
#ifndef QLORA_OPS_H
#define QLORA_OPS_H

/* dx[S,I] += dequant(Q(W))^T dy[S,O].
 * Column-blocked and OpenMP-parallel: each thread owns a slice of I, so every
 * weight byte is read and dequantized ONCE per call (not once per sample row)
 * and applied to all S rows. Batched callers amortize the stream; S=1 results
 * are unchanged. No dequantized copy of W is ever built (one TQ_BLK-column
 * strip per row lives in a per-thread stack buffer).
 * With COLI_METAL, large-S calls go to the M6 t_tmul kernel (parity-tested in
 * metal-test); threshold via COLI_TRAIN_METAL_MIN (default 16), 0 disables. */
#define TQ_BLK 256
static void train_qt_bwd_dx(const QT *w, const float *dy, float *dx, int S){
    int O=w->O, I=w->I;
    if(w->fmt!=0&&w->fmt!=1&&w->fmt!=2&&w->fmt!=4){
        fprintf(stderr,"train_qt_bwd_dx: fmt=%d not supported\n",w->fmt); exit(1); }
#ifdef COLI_METAL
    static int mmin=-2;
    if(mmin==-2){ const char *e=getenv("COLI_TRAIN_METAL_MIN"); mmin=e?atoi(e):16; if(mmin==0) mmin=-1; }
    if(g_metal_enabled && mmin>0 && S>=mmin && !omp_in_parallel()){
        const void *wp = w->fmt==0?(const void*)w->qf : w->fmt==1?(const void*)w->q8:(const void*)w->q4;
        if(coli_metal_train_tmul(dx,dy,wp,w->s,w->fmt,w->gs,S,I,O)) return;
    }
#endif
    int nb=(I+TQ_BLK-1)/TQ_BLK;
    #pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++){
        int i0=b*TQ_BLK, i1=i0+TQ_BLK<I?i0+TQ_BLK:I, bl=i1-i0;
        float wt[TQ_BLK];
        for(int o=0;o<O;o++){
            int any=0;
            for(int s=0;s<S;s++) if(dy[(int64_t)s*O+o]!=0){ any=1; break; }
            if(!any) continue;
            if(w->fmt==0){
                const float *wr=w->qf+(int64_t)o*I;
                for(int s=0;s<S;s++){ float c=dy[(int64_t)s*O+o]; if(c==0) continue;
                    float *dxs=dx+(int64_t)s*I;
                    for(int i=i0;i<i1;i++) dxs[i]+=c*wr[i]; }
                continue;
            }
            if(w->fmt==1){
                const int8_t *qr=w->q8+(int64_t)o*I; float sc=w->s[o];
                for(int i=i0;i<i1;i++) wt[i-i0]=sc*(float)qr[i];
            } else if(w->fmt==2){
                const uint8_t *qr=w->q4+(int64_t)o*((I+1)/2); float sc=w->s[o];
                for(int i=i0;i<i1;i++){ uint8_t by=qr[i>>1];
                    wt[i-i0]=sc*(float)((int)((i&1)?(by>>4):(by&0xF))-8); }
            } else { /* fmt==4: grouped int4, scale per gs-column group (GLM-5.2 layout) */
                int gs=w->gs, ng=(I+gs-1)/gs;
                const uint8_t *qr=w->q4+(int64_t)o*((I+1)/2);
                const float *scl=w->s+(int64_t)o*ng;
                for(int i=i0;i<i1;i++){ uint8_t by=qr[i>>1];
                    wt[i-i0]=scl[i/gs]*(float)((int)((i&1)?(by>>4):(by&0xF))-8); }
            }
            for(int s=0;s<S;s++){ float c=dy[(int64_t)s*O+o]; if(c==0) continue;
                float *dxs=dx+(int64_t)s*I+i0;
                for(int i=0;i<bl;i++) dxs[i]+=c*wt[i];
            }
        }
    }
}

/* forward with cached activation: z[S,rank] = A x (pre-scale), y[S,O] += s*B z.
 * z is what backward needs — cache it instead of recomputing (toy scale). */
static void train_lora_fwd(const LoraTensor *t, const float *x, float *y, float *z, int S){
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*t->I; float *zs=z+(int64_t)s*t->rank;
        float *ys=y+(int64_t)s*t->O;
        for(int r=0;r<t->rank;r++){ const float *a=t->A+(int64_t)r*t->I;
            float acc=0; for(int i=0;i<t->I;i++) acc+=a[i]*xs[i]; zs[r]=acc; }
        for(int o=0;o<t->O;o++){ const float *b=t->B+(int64_t)o*t->rank;
            float acc=0; for(int r=0;r<t->rank;r++) acc+=b[r]*zs[r]; ys[o]+=t->scale*acc; }
    }
}

/* accumulate dA[rank,I], dB[O,rank] and (optionally, if dx) dx[S,I] */
static void train_lora_bwd(const LoraTensor *t, const float *x, const float *z,
                           const float *dy, float *dA, float *dB, float *dx, int S){
    float dz[LORA_MAX_RANK];
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*t->I, *zs=z+(int64_t)s*t->rank, *dys=dy+(int64_t)s*t->O;
        for(int o=0;o<t->O;o++){ float c=t->scale*dys[o]; float *dbr=dB+(int64_t)o*t->rank;
            for(int r=0;r<t->rank;r++) dbr[r]+=c*zs[r]; }
        for(int r=0;r<t->rank;r++){ float acc=0;
            for(int o=0;o<t->O;o++) acc+=t->B[(int64_t)o*t->rank+r]*dys[o];
            dz[r]=t->scale*acc; }
        for(int r=0;r<t->rank;r++){ float c=dz[r]; float *dar=dA+(int64_t)r*t->I;
            for(int i=0;i<t->I;i++) dar[i]+=c*xs[i]; }
        if(dx){ float *dxs=dx+(int64_t)s*t->I;
            for(int r=0;r<t->rank;r++){ float c=dz[r]; const float *ar=t->A+(int64_t)r*t->I;
                for(int i=0;i<t->I;i++) dxs[i]+=c*ar[i]; } }
    }
}

/* MSE toy loss: loss = mean((y-tgt)^2), dy = 2/(n)*(y-tgt) */
static float train_mse(const float *y, const float *tgt, float *dy, int64_t n){
    double acc=0;
    for(int64_t i=0;i<n;i++){ float d=y[i]-tgt[i]; acc+=(double)d*d; if(dy) dy[i]=2.f/(float)n*d; }
    return (float)(acc/(double)n);
}

/* AdamW, decoupled weight decay; state m/v per parameter, NaN/Inf detected. */
typedef struct { float lr,b1,b2,eps,wd; int64_t t; } AdamW;
static AdamW adamw_default(float lr){ AdamW o={lr,0.9f,0.999f,1e-8f,0.f,0}; return o; }
/* one step over n params; call once per step per tensor with a shared ++t done
 * by adamw_tick. returns 0 ok, -1 if a NaN/Inf gradient was seen (params untouched). */
static void adamw_tick(AdamW *o){ o->t++; }
static int adamw_step(const AdamW *o, float *p, const float *g, float *m, float *v, int64_t n){
    for(int64_t i=0;i<n;i++) if(!isfinite(g[i])) return -1;
    float bc1=1.f-powf(o->b1,(float)o->t), bc2=1.f-powf(o->b2,(float)o->t);
    for(int64_t i=0;i<n;i++){
        m[i]=o->b1*m[i]+(1.f-o->b1)*g[i];
        v[i]=o->b2*v[i]+(1.f-o->b2)*g[i]*g[i];
        float mh=m[i]/bc1, vh=v[i]/bc2;
        p[i]-=o->lr*(mh/(sqrtf(vh)+o->eps)+o->wd*p[i]);
    }
    return 0;
}

#endif /* QLORA_OPS_H */
