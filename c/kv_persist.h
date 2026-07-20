/* kv_persist.h — .coli_kv on-disk KV cache persistence.
 * Conversations reopen warm across engine restarts: the compressed MLA KV-cache
 * is appended incrementally after every turn, crash-safe (nrec written last).
 * Include after Model/KVState/Cfg are defined; requires now_s() and g_draft. */
#ifndef KV_PERSIST_H
#define KV_PERSIST_H

static int g_kvsave=1;
#define KV_MAGIC  "COLIKV1\0"                    /* v1: righe Lc/Rc f32 */
#define KV_MAGIC2 "COLIKV2\0"                    /* v2 (KV8): righe fp8 e4m3 + scala f32 per riga */
#define KV_MAGIC3 "COLIKV3\0"                    /* v3 (KV_TQ): righe PolarQuant + raggio f32 per riga (h[7]=bits) */
static const char *kv_active_magic(void){ return g_tq?KV_MAGIC3 : g_kv8?KV_MAGIC2 : KV_MAGIC; }

static void kv_hdr(Model *m, int32_t *h, int nrec){
    Cfg *c=&m->c; int nic=0;
    for(int i=0;i<c->n_layers;i++) if(m->Ic && m->Ic[i]) nic++;
    h[0]=c->n_layers; h[1]=c->kv_lora; h[2]=c->qk_rope;
    h[3]=m->has_dsa?c->index_hd:0; h[4]=nic; h[5]=c->vocab; h[6]=nrec;
    h[7]=g_tq?((g_tq_codec<<8)|g_tq_bits):(g_kv8?1:0);   /* format tag: 0=f32, 1=kv8; TQ: codec<<8 | bit width */
}

static int64_t kv_rec_bytes(Model *m){
    Cfg *c=&m->c;
    int64_t rec = 4 + (g_tq ? (int64_t)c->n_layers*(coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec)+coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec)+8)
                            : g_kv8 ? (int64_t)c->n_layers*(c->kv_lora+c->qk_rope+8)
                            : (int64_t)c->n_layers*(c->kv_lora+c->qk_rope)*4);
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i]) rec+=(int64_t)c->index_hd*4;
    return rec;
}

static int kv_disk_open(Model *m){
    KVState *k=m->kv;
    if(k->disk_fp) return 1;
    k->disk_fp=fopen(k->disk_path,"r+b");
    if(k->disk_fp){ char mg[8];                 /* formato del file != formato attivo -> riscrivi */
        if(fread(mg,1,8,k->disk_fp)!=8 || memcmp(mg,kv_active_magic(),8)){
            fclose(k->disk_fp); k->disk_fp=NULL; k->disk_nrec=0; }
    }
    if(!k->disk_fp){
        k->disk_fp=fopen(k->disk_path,"wb");
        if(!k->disk_fp) return 0;
        int32_t h[8]; kv_hdr(m,h,0);
        fwrite(kv_active_magic(),1,8,k->disk_fp); fwrite(h,4,8,k->disk_fp);
        fflush(k->disk_fp);
        fclose(k->disk_fp);
        k->disk_fp=fopen(k->disk_path,"r+b");
        if(!k->disk_fp) return 0;
    }
    return 1;
}

static void kv_disk_truncate(Model *m, int nrec){
    if(!g_kvsave) return;
    KVState *k=m->kv;
    /* Mai scrivere nell'header piu' record di quanti ne esistano fisicamente:
     * un append saltato (OOM dello staging, open fallita) lascia disk_nrec<len,
     * e un truncate al prefix comune dentro quel buco estenderebbe il file su
     * record MAI scritti — il load successivo li leggerebbe come spazzatura. */
    if(nrec>k->disk_nrec) nrec=k->disk_nrec;
    if(k->disk_fp){ fclose(k->disk_fp); k->disk_fp=NULL; }
    FILE *f=fopen(k->disk_path,"r+b");
    if(!f){ k->disk_nrec=0; return; }
    k->disk_nrec=nrec;
    int32_t nr=nrec; fseek(f,8+6*4,SEEK_SET); fwrite(&nr,4,1,f);
    fflush(f); fclose(f);
}

static void kv_disk_reset(Model *m){ kv_disk_truncate(m,0); }

static void kv_disk_append(Model *m, const int *hist, int len){
    KVState *k=m->kv;
    if(!g_kvsave || len<=k->disk_nrec) return;
    Cfg *c=&m->c;
    if(!kv_disk_open(m)) return;
    FILE *f=k->disk_fp;
    int64_t rec = kv_rec_bytes(m);
    if(rec > k->disk_buf_cap){
        uint8_t *nb=realloc(k->disk_buf, rec);
        if(!nb) return;
        k->disk_buf=nb; k->disk_buf_cap=rec;
    }
    fseek(f, 8+8*4 + (int64_t)k->disk_nrec*rec, SEEK_SET);
    for(int p=k->disk_nrec;p<len;p++){
        uint8_t *b=k->disk_buf;
        *(int32_t*)b = hist[p]; b+=4;
        for(int i=0;i<c->n_layers;i++){
            if(g_tq){                      /* v3: byte polari (righe strette) + raggio per riga */
                int lbb=coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec), rbb=coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec);
                memcpy(b, coli_kv_row8(m->Lc8[i],p,lbb), (size_t)lbb); b+=lbb;
                memcpy(b, &m->Lsc[i][p], 4); b+=4;
                memcpy(b, coli_kv_row8(m->Rc8[i],p,rbb), (size_t)rbb); b+=rbb;
                memcpy(b, &m->Rsc[i][p], 4); b+=4;
            } else if(g_kv8){             /* v2: fp8 + scala per riga, stesso staging */
                memcpy(b, coli_kv_row8(m->Lc8[i],p,c->kv_lora), (size_t)c->kv_lora); b+=c->kv_lora;
                memcpy(b, &m->Lsc[i][p], 4); b+=4;
                memcpy(b, coli_kv_row8(m->Rc8[i],p,c->qk_rope), (size_t)c->qk_rope); b+=c->qk_rope;
                memcpy(b, &m->Rsc[i][p], 4); b+=4;
            } else {
                memcpy(b, m->Lc[i]+(int64_t)p*c->kv_lora, (size_t)c->kv_lora*4); b+=c->kv_lora*4;
                memcpy(b, m->Rc[i]+(int64_t)p*c->qk_rope,(size_t)c->qk_rope*4); b+=c->qk_rope*4;
            }
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i]){
            memcpy(b, m->Ic[i]+(int64_t)p*c->index_hd, (size_t)c->index_hd*4); b+=c->index_hd*4;
        }
        fwrite(k->disk_buf, 1, (size_t)rec, f);
    }
    fflush(f);                                   /* dati prima, contatore poi */
#ifdef _WIN32
    _commit(_fileno(f));
#else
    fsync(fileno(f));                            /* i DATI su disco prima che il contatore li
                                                  * dichiari: fflush ferma solo alla page cache */
#endif
    int32_t nr=len; fseek(f,8+6*4,SEEK_SET); fwrite(&nr,4,1,f);
    fflush(f);                                   /* persist the counter too */
#ifdef _WIN32
    _commit(_fileno(f));
#else
    fsync(fileno(f));
#endif
    k->disk_nrec=len;
}
/* Bonifica una riga fp8 letta da disco: l'encoder non emette MAI i codici NaN e4m3
 * (0x7F/0xFF), ma un file corrotto potrebbe — la GPU (__nv_cvt) li decodifica NaN. */
static inline void kv8_sanitize_row(uint8_t *b, int n, float *sc){
    if(!(fabsf(*sc)<3.4e38f)){ memset(b,0,(size_t)n); *sc=1.f; return; }
    for(int i=0;i<n;i++) if((b[i]&0x7F)==0x7F) b[i]=0;
}
/* v3: raggio non finito o negativo -> riga inerte (coli_kvq_dequant_row rende zero). */
static inline void kv_tq_sanitize(float *radius){
    if(!(*radius>=0.f && *radius<3.4e38f)) *radius=0.f;
}

static int kv_disk_load(Model *m, int *hist, int maxctx){
    if(!g_kvsave) return 0;
    KVState *k=m->kv;
    Cfg *c=&m->c;
    FILE *f=fopen(k->disk_path,"rb"); if(!f) return 0;
    char mg[8]; int32_t h[8], w[8]; kv_hdr(m,w,0);
    int dt=-1;                                        /* dtype del FILE: 0=f32 (v1), 1=fp8 (v2), 2=PolarQuant (v3) */
    if(fread(mg,1,8,f)==8){
        if(!memcmp(mg,KV_MAGIC,8)) dt=0; else if(!memcmp(mg,KV_MAGIC2,8)) dt=1;
        else if(!memcmp(mg,KV_MAGIC3,8)) dt=2; }
    if(dt<0 || fread(h,4,8,f)!=8 ||
       h[0]!=w[0]||h[1]!=w[1]||h[2]!=w[2]||h[3]!=w[3]||h[4]!=w[4]||h[5]!=w[5]){
        fprintf(stderr,"[KV] ignoring .coli_kv from a different model or version\n"); fclose(f); return 0; }
    if(dt==1 && !g_kv8){
        fprintf(stderr,"[KV] .coli_kv is fp8 (saved under KV8=1): starting over (set KV8=1 to resume it)\n");
        fclose(f); return 0; }
    if(dt==2 && !g_tq){
        fprintf(stderr,"[KV] .coli_kv is PolarQuant (saved under KV_TQ): starting over (set KV_TQ to resume it)\n");
        fclose(f); return 0; }
    if(dt==2 && g_tq && h[7]!=((g_tq_codec<<8)|g_tq_bits)){   /* h[7] = codec<<8 | bit width */
        fprintf(stderr,"[KV] .coli_kv KV_TQ (codec %d, %d-bit) != current (codec %d, %d-bit): starting over\n",
            (h[7]>>8)&0xFF, h[7]&0xFF, g_tq_codec, g_tq_bits);
        fclose(f); return 0; }
    int nrec=h[6];
    if(nrec<1){ fclose(f); return 0; }
    if(nrec>=maxctx-8-g_draft){
        fprintf(stderr,"[KV] saved conversation (%d tokens) exceeds the context: starting over\n",nrec);
        fclose(f); return 0; }
    double t0=now_s();
    /* v1 f32 sotto KV8/KV_TQ: righe f32 lette in staging e quantizzate al volo */
    float *stage = ((g_kv8||g_tq)&&dt==0) ? falloc(c->kv_lora>c->qk_rope?c->kv_lora:c->qk_rope) : NULL;
    for(int p=0;p<nrec;p++){
        int32_t tk; if(fread(&tk,4,1,f)!=1){ nrec=p; break; } hist[p]=tk;
        for(int i=0;i<c->n_layers;i++){
            if(dt==2){                                /* v3: byte polari + raggio, gia' nel formato in RAM */
                int lbb=coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec), rbb=coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec);
                if(fread(coli_kv_row8(m->Lc8[i],p,lbb), 1, lbb, f)!=(size_t)lbb ||
                   fread(&m->Lsc[i][p], 4, 1, f)!=1 ||
                   fread(coli_kv_row8(m->Rc8[i],p,rbb), 1, rbb, f)!=(size_t)rbb ||
                   fread(&m->Rsc[i][p], 4, 1, f)!=1){ nrec=p; goto out; }
                kv_tq_sanitize(&m->Lsc[i][p]); kv_tq_sanitize(&m->Rsc[i][p]);
            } else if(dt==1){                          /* v2: fp8+scala, gia' nel formato in RAM */
                if(fread(coli_kv_row8(m->Lc8[i],p,c->kv_lora), 1, c->kv_lora, f)!=(size_t)c->kv_lora ||
                   fread(&m->Lsc[i][p], 4, 1, f)!=1 ||
                   fread(coli_kv_row8(m->Rc8[i],p,c->qk_rope), 1, c->qk_rope, f)!=(size_t)c->qk_rope ||
                   fread(&m->Rsc[i][p], 4, 1, f)!=1){ nrec=p; goto out; }
                kv8_sanitize_row(coli_kv_row8(m->Lc8[i],p,c->kv_lora), c->kv_lora, &m->Lsc[i][p]);
                kv8_sanitize_row(coli_kv_row8(m->Rc8[i],p,c->qk_rope), c->qk_rope, &m->Rsc[i][p]);
            } else if(g_tq){                          /* v1 f32 sotto KV_TQ: PolarQuant al volo */
                if(fread(stage, 4, c->kv_lora, f)!=(size_t)c->kv_lora){ nrec=p; goto out; }
                m->Lsc[i][p]=coli_kvq_quant_row(stage, coli_kv_row8(m->Lc8[i],p,coli_kvq_row_bytes(c->kv_lora,g_tq_bits,g_tq_codec)), c->kv_lora, g_tq_bits, g_tq_codec);
                if(fread(stage, 4, c->qk_rope, f)!=(size_t)c->qk_rope){ nrec=p; goto out; }
                m->Rsc[i][p]=coli_kvq_quant_row(stage, coli_kv_row8(m->Rc8[i],p,coli_kvq_row_bytes(c->qk_rope,g_tq_bits,g_tq_codec)), c->qk_rope, g_tq_bits, g_tq_codec);
            } else if(g_kv8){                          /* v1 f32 sotto KV8: fp8 al volo */
                if(fread(stage, 4, c->kv_lora, f)!=(size_t)c->kv_lora){ nrec=p; goto out; }
                m->Lsc[i][p]=coli_kv8_quant_row(stage, coli_kv_row8(m->Lc8[i],p,c->kv_lora), c->kv_lora);
                if(fread(stage, 4, c->qk_rope, f)!=(size_t)c->qk_rope){ nrec=p; goto out; }
                m->Rsc[i][p]=coli_kv8_quant_row(stage, coli_kv_row8(m->Rc8[i],p,c->qk_rope), c->qk_rope);
            } else {
                if(fread(m->Lc[i]+(int64_t)p*c->kv_lora, 4, c->kv_lora, f)!=(size_t)c->kv_lora ||
                   fread(m->Rc[i]+(int64_t)p*c->qk_rope, 4, c->qk_rope, f)!=(size_t)c->qk_rope){ nrec=p; goto out; }
            }
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i])
            if(fread(m->Ic[i]+(int64_t)p*c->index_hd, 4, c->index_hd, f)!=(size_t)c->index_hd){ nrec=p; goto out; }
    }
out:
    fclose(f); free(stage);
    if(nrec>0){
        if(m->has_mtp) m->kv_start[c->n_layers]=-1;
        fprintf(stderr,"[KV] resumed conversation from disk: %d tokens in %.1fs (no re-prefill)\n",
            nrec, now_s()-t0);
        if((g_kv8||g_tq) && dt==0){
            /* upgrade v1->v2/v3: il file f32 non puo' ricevere append quantizzati.
             * disk_nrec=0 e il file resta INTATTO: al primo append il magic lo riscrive. */
            k->disk_nrec=0;
            fprintf(stderr,"[KV] f32 .coli_kv quantized in RAM; will be rewritten as %s at next save\n",
                g_tq?"PolarQuant (v3)":"fp8 (v2)");
            return nrec;
        }
    }
    k->disk_nrec=nrec;
    return nrec;
}

#endif /* KV_PERSIST_H */
