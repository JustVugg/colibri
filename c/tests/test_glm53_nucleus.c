/* CPU-only exact sampling oracle. No weights, service calls or GPU use.
 * Original and candidate sampler bodies are frozen in separate translation units.
 * The candidate is the actual engine function; the original is a separate
 * frozen upstream translation unit. No performance claim is made by this test. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
extern int nucleus_test_rand(void);
#define rand nucleus_test_rand
#define main glm53_engine_main_unused
#include "../glm53.c"
#undef main
#undef rand
static int draw_value, draw_calls;
int nucleus_test_rand(void) { draw_calls++; return draw_value; }
int nucleus_reference(const float*,int,float,float,int,int*);
static int actual_original(const float*x,int n,float t,float p,int draw,int*used) {
    return nucleus_reference(x,n,t,p,draw,used);
}
static int actual_candidate(const float*x,int n,float t,float p,int draw,int*used) {
    g_temp=t;g_topp=p;draw_value=draw;draw_calls=0;g_glm53_nucleus=1;
    int y=sample_token(x,n);*used=draw_calls;return y;
}
#include "../glm53_exact_nucleus.h"


static const float *key;
static int compare(const void *a,const void *b) {
 int x=*(const int*)a,y=*(const int*)b;
 return key[x]<key[y]?1:key[x]>key[y]?-1:0;
}
static uint32_t rng=0x6a09e667;
static uint32_t next(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void *mem(size_t n){void*p=malloc(n);if(!p){fputs("OOM\n",stderr);exit(2);}return p;}
static void check(int yes,const char*msg){if(!yes){fprintf(stderr,"FAIL %s\n",msg);exit(20);}}
static void logits(float*x,int n,int kind) {
 for(int i=0;i<n;i++) {
  int j=(int)(((uint64_t)i*7919u)%n);
  switch(kind) {
   case 0:x[i]=0.f;break;
   case 1:x[i]=j<512?20.f-j*.15f:-40.f;break;
   case 2:x[i]=j<16?20.f:-20.f;break;
   case 3:x[i]=(float)((int)(next()%20000)-10000)/10000.f;break;
   case 4:x[i]=i==n-1?50.f:-50.f;break;
   case 5:x[i]=20.f-j*.002f;break;
   case 6:x[i]=(i&1)?0.f:-0.f;break;
   default:x[i]=i==n-3?0.f:-INFINITY;break;
  }
 }
}
static void normalize(float*p,const float*x,int n,float temp) {
 float top=-INFINITY;for(int i=0;i<n;i++)if(x[i]>top)top=x[i];
 double total=0.;float inv=1.f/temp;
 for(int i=0;i<n;i++){p[i]=expf((x[i]-top)*inv);total+=p[i];}
 for(int i=0;i<n;i++)p[i]=(float)(p[i]/total);
}
static int prefix_check(float*p,int n,float topp) {
 int *rm=mem((n+2)*sizeof(int)),*ref=rm+1,*cm=mem((n+2)*sizeof(int)),*cand=cm+1;
 float *copy=mem((size_t)n*sizeof(float));memcpy(copy,p,(size_t)n*sizeof(float));
 rm[0]=rm[n+1]=cm[0]=cm[n+1]=0x13579;
 for(int i=0;i<n;i++)ref[i]=i;
 key=p;qsort(ref,n,sizeof(int),compare);
 double mass=0.;int keep=n;
 for(int i=0;i<n;i++){mass+=p[ref[i]];if(mass>=topp){keep=i+1;break;}}
 double gotmass=-17.;int gotkeep=-17;
 int fast=g53_nucleus_prefix(p,n,topp,cand,&gotkeep,&gotmass,compare);
 if(fast){check(keep==gotkeep,"exact retained count");check(!memcmp(&mass,&gotmass,sizeof(double)),"bit-exact cumulative mass");check(!memcmp(ref,cand,keep*sizeof(int)),"exact nucleus token order");}
 else check(gotmass==-17. && gotkeep==-17,"declined path preserves output parameters");
 check(!memcmp(copy,p,(size_t)n*sizeof(float)),"probability array unchanged");
 check(rm[0]==0x13579&&rm[n+1]==0x13579&&cm[0]==0x13579&&cm[n+1]==0x13579,"scratch canaries intact");
 free(copy);free(cm);free(rm);return fast;
}

int main(void) {

 int sizes[]={32,1024,1030,4096,154880};float temps[]={.25f,.7f,1.3f};float ps[]={0.f,.1f,.5f,.9f,1.f};int draws[]={0,RAND_MAX/2,RAND_MAX};
 int comparisons=0,prefixes=0,fast=0;
 for(int z=0;z<5;z++) {
  int n=sizes[z];float*x=mem(n*sizeof(float)),*p=mem(n*sizeof(float));
  for(int kind=0;kind<8;kind++) {
   logits(x,n,kind);
   for(int t=0;t<3;t++)for(int k=0;k<5;k++) {
    normalize(p,x,n,temps[t]);fast+=prefix_check(p,n,ps[k]);prefixes++;
    for(int d=0;d<3;d++) {
     int ca=0,cb=0;int a=actual_original(x,n,temps[t],ps[k],draws[d],&ca),b=actual_candidate(x,n,temps[t],ps[k],draws[d],&cb);
     if(a!=b){fprintf(stderr,"MISMATCH n=%d kind=%d temp=%g top_p=%g draw=%d old=%d new=%d\n",n,kind,temps[t],ps[k],draws[d],a,b);return 20;}
     check(ca==1&&cb==1,"exactly one unchanged RNG draw");comparisons++;
    }
   }
  }
  free(p);free(x);
 }
 {int n=2048,*ix=mem(n*sizeof(int)),keep=-1;float*p=mem(n*sizeof(float));double sum=-1;
  for(int i=0;i<n;i++)p[i]=1.f/n;key=p;p[1000]=NAN;
  check(!g53_nucleus_prefix(p,n,.9f,ix,&keep,&sum,compare),"NaN strictly falls back");p[1000]=INFINITY;
  check(!g53_nucleus_prefix(p,n,.9f,ix,&keep,&sum,compare),"infinite probability strictly falls back");p[1000]=-.1f;
  check(!g53_nucleus_prefix(p,n,.9f,ix,&keep,&sum,compare),"negative probability strictly falls back");free(p);free(ix);}
 printf("PASS %d actual-source token/RNG comparisons; %d prefix mass/order fixtures (%d fast, %d strict fallbacks); canaries/nonfinite guards\n",comparisons,prefixes,fast,prefixes-fast);
 /* Greedy mode bypasses sorting and consumes no RNG. */
 { float x[]={-1.f,2.f,2.f,0.f};int ca=0,cb=0;
   int a=actual_original(x,4,0.f,.9f,0,&ca),b=actual_candidate(x,4,0.f,.9f,0,&cb);
   check(a==b&&ca==0&&cb==0,"greedy branch unchanged and consumes no RNG"); }
 return 0;
}
