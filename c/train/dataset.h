/* Pre-tokenized SFT dataset reader (AGENTS.md §14). Format "coli-sft-v1":
 *
 *   <split>.bin   uint32 token ids, all samples concatenated
 *   <split>.msk   uint8 per token: 1 = counts in the loss, 0 = masked (prompt/pad)
 *   <split>.idx   int64 header {magic 0x434F4C49534654n, n_samples},
 *                 then n_samples+1 int64 cumulative token offsets
 *   metadata.json (informational: tokenizer, template, counts — not read here)
 *
 * Produced by tools/prepare_sft.py. Iteration v1 (no packing): one window of
 * seq_len+1 tokens per sample per epoch — long samples truncated, short ones
 * padded at the END with token 0 / mask 0 (causal attention: no real token can
 * attend a pad; pads carry zero loss). Samples with < 2 tokens are skipped at
 * open time. Deterministic seeded Fisher-Yates reshuffle every epoch. */
#ifndef TRAIN_DATASET_H
#define TRAIN_DATASET_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TDS_MAGIC 0x434F4C4953465431ll   /* "COLISFT1" */

typedef struct {
    uint32_t *tok; uint8_t *msk;         /* whole split, resident (token ids are small) */
    int64_t n_tok;
    int64_t *off; int n;                 /* sample offsets, usable sample count */
    int seq_len;
    uint64_t seed; int epoch;
    int *order, cursor;
} TDataset;

static uint64_t tds_rng(uint64_t *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

static void tds_shuffle(TDataset *d){
    uint64_t s=d->seed ^ (0x9E3779B97F4A7C15ull*(uint64_t)(d->epoch+1));
    for(int i=0;i<d->n;i++) d->order[i]=i;
    for(int i=d->n-1;i>0;i--){ int j=(int)(tds_rng(&s)%(uint64_t)(i+1));
        int t=d->order[i]; d->order[i]=d->order[j]; d->order[j]=t; }
}

static void *tds_readfile(const char *dir, const char *split, const char *ext, int64_t *len){
    char p[1024]; snprintf(p,sizeof(p),"%s/%s.%s",dir,split,ext);
    FILE *f=fopen(p,"rb"); if(!f){ fprintf(stderr,"[data] missing %s\n",p); return NULL; }
    fseek(f,0,SEEK_END); *len=ftell(f); fseek(f,0,SEEK_SET);
    void *b=malloc(*len);
    if(fread(b,1,*len,f)!=(size_t)*len){ fclose(f); free(b); fprintf(stderr,"[data] short read %s\n",p); return NULL; }
    fclose(f); return b;
}

/* open one split; returns 0 ok. Deterministic given (dir,split,seq_len,seed). */
static int tds_open(TDataset *d, const char *dir, const char *split, int seq_len, uint64_t seed){
    memset(d,0,sizeof(*d));
    d->seq_len=seq_len; d->seed=seed;
    int64_t bl=0,ml=0,il=0;
    d->tok=(uint32_t*)tds_readfile(dir,split,"bin",&bl); if(!d->tok) return -1;
    d->msk=(uint8_t*) tds_readfile(dir,split,"msk",&ml); if(!d->msk) return -1;
    int64_t *idx=(int64_t*)tds_readfile(dir,split,"idx",&il); if(!idx) return -1;
    d->n_tok=bl/4;
    if(ml!=d->n_tok || il<24 || idx[0]!=TDS_MAGIC){
        fprintf(stderr,"[data] %s/%s: inconsistent files (magic/lengths)\n",dir,split); free(idx); return -1; }
    int ns=(int)idx[1];
    if(il != (int64_t)(2+ns+1)*8 || idx[2+ns]!=d->n_tok){
        fprintf(stderr,"[data] %s/%s: bad index (n=%d)\n",dir,split,ns); free(idx); return -1; }
    /* keep [start,end) pairs of samples with >= 2 tokens; validate monotonicity */
    d->off=malloc((size_t)ns*2*8);
    d->n=0;
    for(int i=0;i<ns;i++){
        int64_t a=idx[2+i], b=idx[2+i+1];
        if(b<a || b>d->n_tok){ fprintf(stderr,"[data] bad offsets sample %d\n",i); free(idx); return -1; }
        if(b-a>=2){ d->off[d->n*2]=a; d->off[d->n*2+1]=b; d->n++; }
    }
    free(idx);
    if(d->n==0){ fprintf(stderr,"[data] %s/%s: no usable samples\n",dir,split); return -1; }
    d->order=malloc((size_t)d->n*4);
    d->epoch=0; d->cursor=0;
    tds_shuffle(d);
    return 0;
}

/* fill one training window: tok_out/msk_out have seq_len+1 entries.
 * Loss applies at positions where msk_out[t+1]==1 (predicting token t+1). */
static void tds_next(TDataset *d, uint32_t *tok_out, uint8_t *msk_out){
    if(d->cursor>=d->n){ d->cursor=0; d->epoch++; tds_shuffle(d); }
    int si=d->order[d->cursor++];
    int64_t a=d->off[si*2], b=d->off[si*2+1];
    int L=d->seq_len+1, n=(int)((b-a)<L?(b-a):L);
    memcpy(tok_out,d->tok+a,(size_t)n*4);
    memcpy(msk_out,d->msk+a,(size_t)n);
    for(int i=n;i<L;i++){ tok_out[i]=0; msk_out[i]=0; }   /* end-pad, no loss */
}

static void tds_free(TDataset *d){ free(d->tok); free(d->msk); free(d->off); free(d->order); }

#endif /* TRAIN_DATASET_H */
