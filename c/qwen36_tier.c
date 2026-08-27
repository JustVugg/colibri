/* qwen36_tier.c -- CUDA VRAM expert tier for the qwen36 engine. See header. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "qwen36_tier.h"
#include "backend_cuda.h"

#define QT_MAX_DEV 8
#define QT_QCAP 48            /* upload queue depth (staging ~1.6 MB/entry) */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, queued, planned;
    /* raw RAM pointers (slots are never evicted when cap==n_experts) -- lets
     * warmstart, lookahead and LFRU swaps run without an engine callback */
    const uint8_t *g4,*u4,*d4; const float *gs,*us,*ds;
} QSlot;

static struct {
    int on, nl, ne, D, Ih, topk, ndev;
    int egs; size_t sc_gu, sc_d;   /* expert group size + per-matrix scale counts (gs64) */
    int dev[QT_MAX_DEV];
    size_t budget[QT_MAX_DEV], used[QT_MAX_DEV];
    size_t exp_bytes;                     /* estimated VRAM bytes per expert */
    QSlot *slot;                          /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_t th;
    int th_stop;
    /* upload ring with staging copies */
    struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[QT_QCAP];
    int qh, qt_, qn;
    pthread_cond_t cv;
    /* statistics */
    uint64_t hits[QT_MAX_DEV], miss, uploads, q_full_skips;
    /* issue state of the (single) decode thread */
    int is_cnt[QT_MAX_DEV];
    int is_k[QT_MAX_DEV][32];
    float *is_x;                          /* count*D input replicas per device */
    /* M3 */
    int *fill_order; int fill_cur;        /* warmstart order (heat desc) */
    int issue_open;                       /* guard: no tensor_free while a group is in flight */
    pthread_cond_t cv_take;               /* signals qt_take done + queue space */
    uint64_t tick, swaps, pf_hits, pf_notes;
    uint32_t *heat0;                      /* heat table loaded from HEAT_FILE */
} G;

static QSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

/* Staging: packed int4 (g|u|d) two's-complement -> offset-binary (XOR 0x88,
 * the upload format of backend_cuda fmt=2) + copy the scales (gs|us|ds). */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
                  const float *gs,const float *us,const float *ds){
    size_t mb = (size_t)G.D*G.Ih/2;
    const uint64_t X=0x8888888888888888ull;
    const uint64_t *sg=(const uint64_t*)g4,*su=(const uint64_t*)u4,*sd=(const uint64_t*)d4;
    uint64_t *w0=(uint64_t*)dw,*w1=(uint64_t*)(dw+mb),*w2=(uint64_t*)(dw+2*mb);
    for(size_t i=0;i<mb/8;i++){ w0[i]=sg[i]^X; w1[i]=su[i]^X; w2[i]=sd[i]^X; }
    memcpy(dsc,                 gs, G.sc_gu*sizeof(float));
    memcpy(dsc+G.sc_gu,         us, G.sc_gu*sizeof(float));
    memcpy(dsc+2*G.sc_gu,       ds, G.sc_d *sizeof(float));
}

static void *uploader(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&G.mx);
        while(G.qn==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
        if(G.th_stop && G.qn==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        int layer=G.q[G.qh].layer, eid=G.q[G.qh].eid;
        int vl=G.q[G.qh].v_layer, ve=G.q[G.qh].v_eid;
        uint8_t *w=G.q[G.qh].w; float *sc=G.q[G.qh].s;
        G.qh=(G.qh+1)%QT_QCAP; G.qn--;
        pthread_cond_broadcast(&G.cv_take);          /* queue space available */
        if(ve>=0){
            /* LFRU swap: free the victim only when no group is in flight */
            while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
            QSlot *v=qs(vl,ve);
            ColiCudaTensor *a=v->tg,*b=v->tu,*ct=v->td;
            v->tg=v->tu=v->td=NULL;
            pthread_mutex_unlock(&G.mx);
            if(a)coli_cuda_tensor_free(a); if(b)coli_cuda_tensor_free(b); if(ct)coli_cuda_tensor_free(ct);
        } else pthread_mutex_unlock(&G.mx);

        int dv = G.dev[home(eid)];
        size_t mb=(size_t)G.D*G.Ih/2;
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok;
        if(G.egs){
            ok = coli_cuda_tensor_upload_g(&tg, w,      sc,             4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&tu, w+mb,   sc+G.sc_gu,     4, G.D,  G.Ih, dv, G.egs)
              && coli_cuda_tensor_upload_g(&td, w+2*mb, sc+2*G.sc_gu,   4, G.Ih, G.D,  dv, G.egs);
        } else {
            ok = coli_cuda_tensor_upload(&tg, w,      sc,          2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+mb,   sc+G.Ih,     2, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*mb, sc+2*G.Ih,   2, G.Ih, G.D,  dv);
        }
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        QSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid); G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* device genuinely full: stop trying */ }
        s->queued=0;
        pthread_mutex_unlock(&G.mx);
    }
}

int qt_init(int nl, int ne, int D, int Ih, int cap, int topk, int expert_gs){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(cap != ne){
        fprintf(stderr,"[qtier] cap=%d != n_experts=%d -> tier disabled (needs full RAM residency)\n",cap,ne);
        return 0;
    }
    if(topk>32){ fprintf(stderr,"[qtier] topk>32 unsupported\n"); return 0; }
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk;

    /* devices: COLI_GPUS="0,1" (default: first two visible devices) */
    const char *gl=getenv("COLI_GPUS");
    if (gl && *gl) {
        char buf[128]; snprintf(buf,sizeof buf,"%s",gl);
        for(char *t=strtok(buf,","); t && G.ndev<QT_MAX_DEV; t=strtok(NULL,","))
            G.dev[G.ndev++]=atoi(t);
    } else {
        int available=coli_cuda_available_device_count();
        int want=available<2?available:2;
        for(int i=0;i<want && i<QT_MAX_DEV;i++) G.dev[G.ndev++]=i;
        fprintf(stderr,"[qtier] COLI_GPUS unset: selecting %d visible device(s)\n",G.ndev);
    }
    if(G.ndev<1){ fprintf(stderr,"[qtier] no visible CUDA devices -> CPU path\n"); return 0; }
    if(!coli_cuda_init(G.dev,G.ndev)){ fprintf(stderr,"[qtier] coli_cuda_init failed -> CPU path\n"); return 0; }
    int have=coli_cuda_device_count();
    if(have<G.ndev){ G.ndev=have; }
    if(G.ndev<1){ fprintf(stderr,"[qtier] no CUDA devices -> CPU path\n"); return 0; }

    /* per-device budget: CUDA_EXPERT_GB, or auto = free minus 1 GB headroom.
     * Scale counts follow the container: per-row (expert_gs=0) or grouped
     * (gs64: [O, ceil(I/gs)] per projection). */
    G.egs = expert_gs;
    G.sc_gu = expert_gs ? (size_t)Ih * ((D + expert_gs - 1)/expert_gs) : (size_t)Ih;
    G.sc_d  = expert_gs ? (size_t)D  * ((Ih + expert_gs - 1)/expert_gs) : (size_t)D;
    G.exp_bytes = 3ull*D*Ih/2 + (2*G.sc_gu+G.sc_d)*sizeof(float) + 4096; /* + allocation slack */
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        size_t b = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : (freeb>(1ull<<30) ? freeb-(1ull<<30) : 0);
        G.budget[i]=b;
        fprintf(stderr,"[qtier] dev %d: %.1f GB free, budget %.1f GB (~%zu experts)\n",
                G.dev[i], freeb/1073741824.0, b/1073741824.0, b/G.exp_bytes);
    }
    G.slot=calloc((size_t)nl*ne,sizeof(QSlot));
    G.is_x=malloc((size_t)32*D*sizeof(float));
    if(!G.slot||!G.is_x) return 0;
    /* load learned heat (HEAT_FILE): warmstart order + initial values */
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            if(fread(hdr,4,3,f)==3 && hdr[0]==0x51544831u && hdr[1]==(uint32_t)nl && hdr[2]==(uint32_t)ne){
                G.heat0=malloc((size_t)nl*ne*4);
                if(G.heat0 && fread(G.heat0,4,(size_t)nl*ne,f)==(size_t)nl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++) G.slot[i].heat=G.heat0[i]>>1; /* decay */
                    fprintf(stderr,"[qtier] HEAT_FILE loaded: %s\n",hf);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
    }
    pthread_mutex_init(&G.mx,NULL); pthread_cond_init(&G.cv,NULL); pthread_cond_init(&G.cv_take,NULL);
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0) return 0;
    G.on=1;
    fprintf(stderr,"[qtier] CUDA VRAM expert tier active: %d device(s), %.2f MB/expert\n",
            G.ndev, G.exp_bytes/1048576.0);
    return 1;
}

int qt_ready(void){ return G.on; }

/* Is (layer,eid) currently VRAM-resident? (used to free RAM-side int8 copies) */
int qt_is_resident(int layer,int eid){
    if(!G.on) return 0;
    pthread_mutex_lock(&G.mx);
    int r = qs(layer,eid)->resident;
    pthread_mutex_unlock(&G.mx);
    return r;
}

/* internal, G.mx held: enqueue one upload. victim=-1: plain upload (budget is
 * reserved here); victim>=0: LFRU swap (budget neutral). */
static int enqueue_locked(int layer,int eid,int v_layer,int v_eid,int reserved){
    QSlot *s=qs(layer,eid);
    if(s->resident||s->queued||!s->g4) return 0;
    if(G.qn>=QT_QCAP){ G.q_full_skips++; return 0; }
    int hd=home(eid);
    if(!reserved && v_eid<0 && G.used[hd]+G.exp_bytes>G.budget[hd]) return 0;
    size_t mb=(size_t)G.D*G.Ih/2;
    uint8_t *w=malloc(3*mb); float *sc=malloc((2*G.sc_gu+G.sc_d)*sizeof(float));
    if(!w||!sc){ free(w); free(sc); return 0; }
    if(!reserved && v_eid<0) G.used[hd]+=G.exp_bytes;
    s->queued=1;
    stage(w,sc,s->g4,s->u4,s->d4,s->gs,s->us,s->ds);
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid; G.q[G.qt_].w=w; G.q[G.qt_].s=sc;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%QT_QCAP; G.qn++;
    pthread_cond_signal(&G.cv);
    return 1;
}

void qt_note(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    if(s->heat<0xFFFFFFFFu) s->heat++;
    enqueue_locked(layer,eid,-1,-1,0);
    pthread_mutex_unlock(&G.mx);
}

/* blocking variant for the warmstart (waits for queue space). */
void qt_note_block(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    enqueue_locked(layer,eid,-1,-1,0);
    pthread_mutex_unlock(&G.mx);
}

/* warmstart order -- heat descending (HEAT_FILE) or natural order.
 * Returns 0 once all budgets are full or the list is exhausted. */
static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : ha>hb ? -1 : 0;
}
int qt_fill_next(int *layer,int *eid){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        G.fill_order=malloc(n*sizeof(int));
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
    }
    while((size_t)G.fill_cur<n){
        int gi=G.fill_order[G.fill_cur];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+G.exp_bytes<=G.budget[i]) full=0;
        if(full){ pthread_mutex_unlock(&G.mx); return 0; }
        G.fill_cur++;
        if(s->resident||s->queued) continue;
        if(G.used[hd]+G.exp_bytes>G.budget[hd]) continue;   /* dieses Device voll */
        *layer=l; *eid=e;
        pthread_mutex_unlock(&G.mx);
        return 1;
    }
    pthread_mutex_unlock(&G.mx);
    return 0;
}

/* Plan the whole warmstart set in one pass -- same heat order and budget
 * reservation as qt_fill_next, but without loading. The experts are then
 * loaded by any number of threads and handed over via qt_note_planned. */
int qt_plan_fill(int *layers,int *eids,int max){
    if(!G.on) return 0;
    size_t n=(size_t)G.nl*G.ne;
    int cnt=0;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        G.fill_order=malloc(n*sizeof(int));
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        if(G.heat0){ g_sort_heat=G.heat0; qsort(G.fill_order,n,sizeof(int),cmp_heat_desc); }
        G.fill_cur=0;
    }
    while((size_t)G.fill_cur<n && cnt<max){
        int full=1; for(int i=0;i<G.ndev;i++) if(G.used[i]+G.exp_bytes<=G.budget[i]) full=0;
        if(full) break;
        int gi=G.fill_order[G.fill_cur++];
        int l=gi/G.ne, e=gi%G.ne, hd=home(e);
        QSlot *s=qs(l,e);
        if(s->resident||s->queued||s->planned) continue;
        if(G.used[hd]+G.exp_bytes>G.budget[hd]) continue;
        G.used[hd]+=G.exp_bytes;          /* reserve */
        s->planned=1;
        layers[cnt]=l; eids[cnt]=e; cnt++;
    }
    pthread_mutex_unlock(&G.mx);
    return cnt;
}

/* Thread-safe (callable from multiple loader threads): stage + enqueue one
 * expert reserved by qt_plan_fill; blocks only while the queue is full. */
void qt_note_planned(int layer,int eid,
             const uint8_t *g4,const uint8_t *u4,const uint8_t *d4,
             const float *gs,const float *us,const float *ds){
    if(!G.on || !g4) return;
    QSlot *s=qs(layer,eid);
    pthread_mutex_lock(&G.mx);
    if(!s->g4){ s->g4=g4; s->u4=u4; s->d4=d4; s->gs=gs; s->us=us; s->ds=ds; }
    while(G.qn>=QT_QCAP && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    if(!enqueue_locked(layer,eid,-1,-1,1)){
        /* not enqueueable (e.g. already resident): return the reservation */
        if(s->planned) G.used[home(eid)]-=G.exp_bytes;
    }
    s->planned=0;
    pthread_mutex_unlock(&G.mx);
}

/* waits until the upload queue is drained (end of warmstart). */
void qt_fill_wait(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    while(G.qn>0 && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
}

/* LFRU swap check (every 16 ticks = tokens): per device, coldest resident vs
 * hottest non-resident, with the tier.h hysteresis. */
static void qt_lfru_tick_locked(void){
    if(++G.tick % 16) return;
    size_t n=(size_t)G.nl*G.ne;
    for(int di=0;di<G.ndev;di++){
        int cold=-1, hot=-1; uint32_t ch=0, hh=0;
        for(size_t i=0;i<n;i++){
            QSlot *s=&G.slot[i];
            int e=(int)(i%G.ne);
            if(home(e)!=di) continue;
            if(s->resident && !s->queued){ if(cold<0||s->heat<ch){ cold=(int)i; ch=s->heat; } }
            else if(!s->resident && !s->queued && s->g4){ if(hot<0||s->heat>hh){ hot=(int)i; hh=s->heat; } }
        }
        if(cold<0||hot<0) continue;
        if(hh<=ch+(ch>>2)+4) continue;                    /* hysteresis as in tier.h */
        QSlot *v=&G.slot[cold];
        v->resident=0;                                    /* CPU fallback from now on */
        if(enqueue_locked(hot/G.ne,hot%G.ne,cold/G.ne,cold%G.ne,0)) G.swaps++;
        else v->resident=1;                               /* queue full: revert */
    }
}

uint32_t qt_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on||K>32) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[QT_MAX_DEV][32],*tu[QT_MAX_DEV][32],*td[QT_MAX_DEV][32];
    static int rows[32]={0};
    if(!rows[0]) for(int i=0;i<32;i++) rows[i]=1;
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
    if(layer==0) qt_lfru_tick_locked();
    G.issue_open=1;
    for(int k=0;k<K;k++){
        QSlot *s=qs(layer,eids[k]);
        if(s->resident){
            int di=home(eids[k]); int c=G.is_cnt[di];
            tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
            G.is_k[di][c]=k; G.is_cnt[di]=c+1;
            mask|=1u<<k; G.hits[di]++;
        } else G.miss++;
    }
    pthread_mutex_unlock(&G.mx);

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*8*G.D;               /* per-device input block */
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        if(!coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr)){
            /* issue failed -> hand these k back to the CPU */
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    return mask;
}

void qt_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on) return;
    if(mask) for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        if(!y) continue;
        for(int j=0;j<c;j++){
            float w=val[G.is_k[di][j]];
            const float *row=y+(size_t)j*G.D;
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.is_cnt[di]=0;
    }
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}

void qt_stats(void){
    if(!G.on) return;
    uint64_t hits=0; size_t res=0;
    for(size_t i=0;i<(size_t)G.nl*G.ne;i++) res += G.slot[i].resident;
    fprintf(stderr,"[qtier] resident %zu/%d experts | uploads %llu | miss(CPU) %llu | q_skips %llu\n",
            res, G.nl*G.ne, (unsigned long long)G.uploads,
            (unsigned long long)G.miss, (unsigned long long)G.q_full_skips);
    for(int i=0;i<G.ndev;i++){
        size_t tc=0,tb=0; coli_cuda_stats(G.dev[i],&tc,&tb);
        hits+=G.hits[i];
        fprintf(stderr,"[qtier]   dev %d: hits %llu | %zu tensors, %.2f GB VRAM used (budget %.2f GB)\n",
                G.dev[i], (unsigned long long)G.hits[i], tc, tb/1073741824.0, G.budget[i]/1073741824.0);
    }
    double tot=(double)(hits+G.miss);
    fprintf(stderr,"[qtier] VRAM hit rate: %.1f %% | LFRU swaps %llu\n",
            tot>0? 100.0*hits/tot : 0.0, (unsigned long long)G.swaps);
    { uint64_t calls=0,ex=0,rows=0; double h2d=0,kms=0,d2h=0;
      coli_cuda_group_stats(&calls,&ex,&rows,&h2d,&kms,&d2h);
      if(calls) fprintf(stderr,"[qtier] group_stats: %llu calls, %llu experts | h2d %.0f ms, kernel %.0f ms, d2h %.0f ms\n",
              (unsigned long long)calls,(unsigned long long)ex,h2d,kms,d2h); }
}

void qt_shutdown(void){
    if(!G.on) return;
    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"wb");
        if(f){
            uint32_t hdr[3]={0x51544831u,(uint32_t)G.nl,(uint32_t)G.ne};
            fwrite(hdr,4,3,f);
            for(size_t i=0;i<(size_t)G.nl*G.ne;i++) fwrite(&G.slot[i].heat,4,1,f);
            fclose(f);
            fprintf(stderr,"[qtier] HEAT_FILE saved: %s\n",hf);
        }
    }
    pthread_mutex_lock(&G.mx); G.th_stop=1; pthread_cond_signal(&G.cv); pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    G.on=0;
    coli_cuda_shutdown();
}

#endif /* COLI_CUDA */
