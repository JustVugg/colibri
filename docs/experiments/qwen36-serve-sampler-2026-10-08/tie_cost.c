/* What the comparator that also orders ties by token id costs a full sort at top_p 1: the same full sort with
 * the old comparator (ties left to qsort) and the new one, on real logits, alternating. */
#include "qwen36_sample.h"
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int old_desc(const void *a, const void *b){ float pa=((const SampleProb*)a)->p, pb=((const SampleProb*)b)->p; return (pb>pa)-(pa>pb); }
static double one(const float *lo, int V, int (*cmp)(const void*,const void*)){
    SampleProb *rank=malloc((size_t)V*sizeof(SampleProb)); float mx=lo[0];
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double a=now();
    for(int i=0;i<V;i++){ float p=expf((lo[i]-mx)/1.0f); rank[i]=(SampleProb){p,i}; }
    qsort(rank,(size_t)V,sizeof(SampleProb),cmp);
    double t=now()-a; free(rank); return t;
}
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }
int main(int argc,char**argv){
    enum{V=248320,N=60}; float *lo=malloc(V*4); FILE*f=fopen(argv[1],"rb"); if(!f||fread(lo,4,V,f)!=V) return 1; fclose(f);
    double to[N],tn[N]; long eq=0;
    for(int k=0;k<N;k++){ if(k&1){ tn[k]=one(lo,V,sample_prob_desc); to[k]=one(lo,V,old_desc);} else { to[k]=one(lo,V,old_desc); tn[k]=one(lo,V,sample_prob_desc);} }
    qsort(to,N,8,cmpd); qsort(tn,N,8,cmpd);
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    float *p=malloc(V*4); for(int i=0;i<V;i++) p[i]=expf(lo[i]-mx);
    /* how many tokens share their probability with another one */
    SampleProb *r=malloc((size_t)V*sizeof(SampleProb)); for(int i=0;i<V;i++) r[i]=(SampleProb){p[i],i};
    qsort(r,V,sizeof(SampleProb),sample_prob_desc); for(int i=1;i<V;i++) if(r[i].p==r[i-1].p) eq++;
    printf("%s: full sort, old comparator median %.3f ms, new (ties by id) %.3f ms; tokens tied with the previous: %ld of %d\n",argv[1],to[N/2]*1e3,tn[N/2]*1e3,eq,V);
    return 0;
}
