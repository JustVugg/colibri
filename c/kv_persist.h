/* kv_persist.h — .coli_kv on-disk KV cache persistence.
 * Conversations reopen warm across engine restarts: the compressed MLA KV-cache
 * is appended incrementally after every turn, crash-safe (nrec written last).
 * Include after Model/KVState/Cfg are defined; requires now_s() and g_draft. */
#ifndef KV_PERSIST_H
#define KV_PERSIST_H

static int g_kvsave=1;
#define KV_MAGIC "COLIKV1\0"

static void kv_hdr(Model *m, int32_t *h, int nrec){
    Cfg *c=&m->c; int nic=0;
    for(int i=0;i<c->n_layers;i++) if(m->Ic && m->Ic[i]) nic++;
    h[0]=c->n_layers; h[1]=c->kv_lora; h[2]=c->qk_rope;
    h[3]=m->has_dsa?c->index_hd:0; h[4]=nic; h[5]=c->vocab; h[6]=nrec; h[7]=0;
}

static int64_t kv_rec_bytes(Model *m){
    Cfg *c=&m->c;
    int64_t rec = 4 + (int64_t)c->n_layers*(c->kv_lora+c->qk_rope)*4;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i]) rec+=(int64_t)c->index_hd*4;
    return rec;
}

static int kv_disk_open(Model *m){
    KVState *k=m->kv;
    if(k->disk_fp) return 1;
    k->disk_fp=fopen(k->disk_path,"r+b");
    if(!k->disk_fp){
        k->disk_fp=fopen(k->disk_path,"wb");
        if(!k->disk_fp) return 0;
        int32_t h[8]; kv_hdr(m,h,0);
        fwrite(KV_MAGIC,1,8,k->disk_fp); fwrite(h,4,8,k->disk_fp);
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
            memcpy(b, m->Lc[i]+(int64_t)p*c->kv_lora, (size_t)c->kv_lora*4); b+=c->kv_lora*4;
            memcpy(b, m->Rc[i]+(int64_t)p*c->qk_rope,(size_t)c->qk_rope*4); b+=c->qk_rope*4;
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i]){
            memcpy(b, m->Ic[i]+(int64_t)p*c->index_hd, (size_t)c->index_hd*4); b+=c->index_hd*4;
        }
        fwrite(k->disk_buf, 1, (size_t)rec, f);
    }
    fflush(f);
    int32_t nr=len; fseek(f,8+6*4,SEEK_SET); fwrite(&nr,4,1,f);
    fflush(f);
    k->disk_nrec=len;
}

static int kv_disk_load(Model *m, int *hist, int maxctx){
    if(!g_kvsave) return 0;
    KVState *k=m->kv;
    Cfg *c=&m->c;
    FILE *f=fopen(k->disk_path,"rb"); if(!f) return 0;
    char mg[8]; int32_t h[8], w[8]; kv_hdr(m,w,0);
    if(fread(mg,1,8,f)!=8 || memcmp(mg,KV_MAGIC,8) || fread(h,4,8,f)!=8 ||
       h[0]!=w[0]||h[1]!=w[1]||h[2]!=w[2]||h[3]!=w[3]||h[4]!=w[4]||h[5]!=w[5]){
        fprintf(stderr,"[KV] ignoring .coli_kv from a different model or version\n"); fclose(f); return 0; }
    int nrec=h[6];
    if(nrec<1){ fclose(f); return 0; }
    if(nrec>=maxctx-8-g_draft){
        fprintf(stderr,"[KV] saved conversation (%d tokens) exceeds the context: starting over\n",nrec);
        fclose(f); return 0; }
    double t0=now_s();
    for(int p=0;p<nrec;p++){
        int32_t tk; if(fread(&tk,4,1,f)!=1){ nrec=p; break; } hist[p]=tk;
        for(int i=0;i<c->n_layers;i++){
            if(fread(m->Lc[i]+(int64_t)p*c->kv_lora, 4, c->kv_lora, f)!=(size_t)c->kv_lora ||
               fread(m->Rc[i]+(int64_t)p*c->qk_rope, 4, c->qk_rope, f)!=(size_t)c->qk_rope){ nrec=p; goto out; }
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i])
            if(fread(m->Ic[i]+(int64_t)p*c->index_hd, 4, c->index_hd, f)!=(size_t)c->index_hd){ nrec=p; goto out; }
    }
out:
    fclose(f);
    if(nrec>0){
        if(m->has_mtp) m->kv_start[c->n_layers]=-1;
        fprintf(stderr,"[KV] resumed conversation from disk: %d tokens in %.1fs (no re-prefill)\n",
            nrec, now_s()-t0);
    }
    k->disk_nrec=nrec;
    return nrec;
}

#endif /* KV_PERSIST_H */
