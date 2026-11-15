#ifndef QWEN36_SAMPLE_H
#define QWEN36_SAMPLE_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "argmax_first.h"

/* temperature + top-p sampler. The probabilities and their sum are taken as
 * before, one pass over the vocabulary; the nucleus is then found without
 * sorting all of it (248,320 entries of Qwen3.6: 24 ms a token on the M1).
 * The probabilities go into buckets by their binary exponent; the buckets from
 * the top down until their total passes top_p of the sum, plus one more, hold
 * every token of the nucleus, and only those are sorted, by probability and
 * then token id. The walk over the sorted prefix and the draw are as before,
 * so the token drawn is the same for the same rand() value, unless tokens of
 * exactly equal probability sit where the walk or the draw stops: they are now
 * in token id order, which the full sort left to qsort. With top_p 1 every
 * token is in the nucleus, and when the candidates do not reach the cut
 * (rounding) the whole vocabulary is needed: both take serve_sample_sorted,
 * the full sort as before (ties in token id order there too). */
typedef struct { float p; int id; } SampleProb;
static int sample_prob_desc(const void *a, const void *b){
    const SampleProb *x=a, *y=b;
    if(x->p!=y->p) return (y->p>x->p)-(x->p>y->p);
    return (x->id>y->id)-(x->id<y->id);
}
/* The whole vocabulary sorted. */
static int serve_sample_sorted(const float *lo, int V, float temp, float top_p){
    SampleProb *rank=malloc((size_t)V*sizeof(SampleProb)); float mx=lo[0];
    if(!rank){ fprintf(stderr,"OOM sampling\n"); exit(1); }
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double sum=0;
    for(int i=0;i<V;i++){ float p=expf((lo[i]-mx)/temp); sum+=p; rank[i]=(SampleProb){p,i}; }
    qsort(rank,(size_t)V,sizeof(SampleProb),sample_prob_desc);
    double cut=(top_p>0.f&&top_p<1.f)?top_p*sum:sum, kept=0; int n=0;
    while(n<V&&kept<cut) kept+=rank[n++].p;
    double r=((double)rand()/RAND_MAX)*kept, acc=0; int pick=rank[0].id;
    for(int i=0;i<n;i++){ acc+=rank[i].p; if(acc>=r){ pick=rank[i].id; break; } }
    free(rank); return pick;
}
/* COLI_LOGIT_DUMP=1: each step's top-5 id:logit to stderr, as sample.h's pick_tok writes them (comparing
 * two configurations on the same context: a stored prefix against a fresh read, a backend against another) */
static void serve_logit_dump(const float *lo, int V){
    static int dump = -1;
    if(dump < 0) dump = getenv("COLI_LOGIT_DUMP") ? 1 : 0;
    if(!dump) return;
    int id[5]={-1,-1,-1,-1,-1}; float v[5]={-3e38f,-3e38f,-3e38f,-3e38f,-3e38f};
    for(int t=0;t<V;t++){
        float x=lo[t];
        for(int k=0;k<5;k++) if(x>v[k]){ for(int j=4;j>k;j--){ v[j]=v[j-1]; id[j]=id[j-1]; } v[k]=x; id[k]=t; break; }
    }
    fprintf(stderr,"[LOGITS]");
    for(int k=0;k<5;k++) fprintf(stderr," %d:%.6f",id[k],v[k]);
    fprintf(stderr,"\n");
}
static int serve_sample(const float *lo, int V, float temp, float top_p){
    serve_logit_dump(lo, V);
    if(temp<=0.f) return argmax_first(lo, V);
    if(!(top_p>0.f&&top_p<1.f)) return serve_sample_sorted(lo,V,temp,top_p);   /* every token is in the nucleus */
    float *prob=malloc((size_t)V*sizeof(float)), mx=lo[0];
    if(!prob){ fprintf(stderr,"OOM sampling\n"); exit(1); }
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double sum=0, bucket[256]={0};
    for(int i=0;i<V;i++){ float p=expf((lo[i]-mx)/temp); sum+=p; prob[i]=p; }
    double cut=top_p*sum, kept=0;
    /* bucket 255 - exponent: the largest probabilities first; zero and subnormals in 255 */
    for(int i=0;i<V;i++){ uint32_t bits; memcpy(&bits,&prob[i],4); bucket[255-((bits>>23)&255)]+=prob[i]; }
    int last=0; double total=0;
    while(last<256){ total+=bucket[last]; if(total>=cut) break; last++; }
    if(last<255) last++;                        /* one bucket more: rounding of the bucket sums */
    int cap=0;
    for(int i=0;i<V;i++){ uint32_t bits; memcpy(&bits,&prob[i],4); if(255-(int)((bits>>23)&255)<=last) cap++; }
    SampleProb *rank=malloc((size_t)(cap>0?cap:1)*sizeof(SampleProb));
    if(!rank){ fprintf(stderr,"OOM sampling\n"); exit(1); }
    int m=0;
    for(int i=0;i<V;i++){ uint32_t bits; memcpy(&bits,&prob[i],4); if(255-(int)((bits>>23)&255)<=last) rank[m++]=(SampleProb){prob[i],i}; }
    free(prob);
    qsort(rank,(size_t)m,sizeof(SampleProb),sample_prob_desc);
    int n=0;
    while(n<m&&kept<cut) kept+=rank[n++].p;
    if(kept<cut && m<V){ free(rank); return serve_sample_sorted(lo,V,temp,top_p); }   /* cut not reached */
    double r=((double)rand()/RAND_MAX)*kept, acc=0; int pick=rank[0].id;
    for(int i=0;i<n;i++){ acc+=rank[i].p; if(acc>=r){ pick=rank[i].id; break; } }
    free(rank); return pick;
}

#endif
