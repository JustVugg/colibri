/* Training checkpoint/resume (AGENTS.md §15). A checkpoint directory holds:
 *   adapter.safetensors + adapter.json    (lora_save — loadable by inference)
 *   train_state.bin                       (this file: optimizer + cursor + RNG)
 *
 * train_state.bin layout (little-endian):
 *   int64 magic 'COLITRN1', int64 version=1
 *   int64 step, int64 opt_t
 *   int32 epoch, int32 cursor            (dataset position)
 *   uint64 data_seed, uint64 base_fp
 *   int32 n_tensors, int32 pad
 *   per tensor: int64 nA, nA f32 mA, nA f32 vA, int64 nB, nB f32 mB, nB f32 vB
 *
 * All writes are atomic (tmp -> rename, reusing lora_write_file). A crash mid-
 * save never destroys the previous valid checkpoint. Resume acceptance: 20
 * uninterrupted steps == 10 -> save -> reload -> 10 (tests/test_train_resume). */
#ifndef TRAIN_CHECKPOINT_H
#define TRAIN_CHECKPOINT_H

#define TCKPT_MAGIC 0x434F4C4954524E31ll   /* "COLITRN1" */

typedef struct {
    int64_t step, opt_t;
    int epoch, cursor;
    uint64_t data_seed, base_fp;
} TCkptState;

static int tckpt_save(const char *dir, const LoraAdapter *a,
                      const TCkptState *st, float **m, float **v){
    if(lora_save(dir,a)) return -1;
    size_t cap=64*1024;
    for(int i=0;i<a->n;i++)
        cap += 16 + 16*((size_t)a->t[i].rank*a->t[i].I + (size_t)a->t[i].O*a->t[i].rank);
    unsigned char *buf=malloc(cap); size_t off=0;
    #define W64(v_) do{ int64_t x_=(int64_t)(v_); memcpy(buf+off,&x_,8); off+=8; }while(0)
    #define W32(v_) do{ int32_t x_=(int32_t)(v_); memcpy(buf+off,&x_,4); off+=4; }while(0)
    W64(TCKPT_MAGIC); W64(1);
    W64(st->step); W64(st->opt_t);
    W32(st->epoch); W32(st->cursor);
    W64(st->data_seed); W64(st->base_fp);
    W32(a->n); W32(0);
    for(int i=0;i<a->n;i++){
        int64_t nA=(int64_t)a->t[i].rank*a->t[i].I, nB=(int64_t)a->t[i].O*a->t[i].rank;
        W64(nA); memcpy(buf+off,m[2*i],nA*4);   off+=nA*4;
                 memcpy(buf+off,v[2*i],nA*4);   off+=nA*4;
        W64(nB); memcpy(buf+off,m[2*i+1],nB*4); off+=nB*4;
                 memcpy(buf+off,v[2*i+1],nB*4); off+=nB*4;
    }
    #undef W64
    #undef W32
    char path[2048]; snprintf(path,sizeof(path),"%s/train_state.bin",dir);
    int rc=lora_write_file(path,buf,off);
    free(buf);
    if(rc) fprintf(stderr,"[ckpt] write %s failed\n",path);
    return rc;
}

/* Load optimizer/cursor state matching an adapter ALREADY loaded from `dir`
 * (lora_load validates the adapter itself). m/v must be allocated to the
 * adapter's tensor sizes. Returns 0 ok, -1 on mismatch/corruption. */
static int tckpt_load(const char *dir, const LoraAdapter *a,
                      TCkptState *st, float **m, float **v){
    char path[2048]; snprintf(path,sizeof(path),"%s/train_state.bin",dir);
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[ckpt] no %s\n",path); return -1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char *buf=malloc(sz);
    if(fread(buf,1,sz,f)!=(size_t)sz){ fclose(f); free(buf); return -1; }
    fclose(f);
    size_t off=0;
    #define R64(v_) do{ if(off+8>(size_t)sz) goto bad; memcpy(&v_,buf+off,8); off+=8; }while(0)
    #define R32(v_) do{ if(off+4>(size_t)sz) goto bad; memcpy(&v_,buf+off,4); off+=4; }while(0)
    { int64_t magic,ver; R64(magic); R64(ver);
      if(magic!=TCKPT_MAGIC||ver!=1){ fprintf(stderr,"[ckpt] bad magic/version\n"); goto bad; }
      R64(st->step); R64(st->opt_t);
      int32_t ep,cur; R32(ep); R32(cur); st->epoch=ep; st->cursor=cur;
      R64(st->data_seed); R64(st->base_fp);
      int32_t n,pad; R32(n); R32(pad);
      if(n!=a->n){ fprintf(stderr,"[ckpt] tensor count %d != adapter %d\n",n,a->n); goto bad; }
      for(int i=0;i<a->n;i++){
        int64_t nA=(int64_t)a->t[i].rank*a->t[i].I, nB=(int64_t)a->t[i].O*a->t[i].rank, got;
        R64(got); if(got!=nA){ fprintf(stderr,"[ckpt] tensor %d nA mismatch\n",i); goto bad; }
        if(off+(size_t)nA*8>(size_t)sz) goto bad;
        memcpy(m[2*i],buf+off,nA*4); off+=nA*4;
        memcpy(v[2*i],buf+off,nA*4); off+=nA*4;
        R64(got); if(got!=nB){ fprintf(stderr,"[ckpt] tensor %d nB mismatch\n",i); goto bad; }
        if(off+(size_t)nB*8>(size_t)sz) goto bad;
        memcpy(m[2*i+1],buf+off,nB*4); off+=nB*4;
        memcpy(v[2*i+1],buf+off,nB*4); off+=nB*4;
      }
    }
    #undef R64
    #undef R32
    free(buf); return 0;
bad:
    fprintf(stderr,"[ckpt] %s corrupt/truncated\n",path);
    free(buf); return -1;
}

#endif /* TRAIN_CHECKPOINT_H */
