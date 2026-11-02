/* Reproducible source-only ARM64 NEON vs scalar diagnostic. No LLM loaded. */
#define _POSIX_C_SOURCE 200809L
#include "../sparse_attn.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng = 20261030u;
static uint32_t rand32(void) {
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return (uint32_t)(rng >> 16);
}
static float rnd(float amplitude) {
  return ((float)(rand32() & 65535u)/32768.0f-1.0f)*amplitude;
}
static uint64_t nanos(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (uint64_t)ts.tv_sec*1000000000ULL+(uint64_t)ts.tv_nsec;
}
static volatile float checksum=0;
typedef void (*attn_fn)(float *,const float *,const float *,const int *,
                        int,int,float,float,float *);
static __attribute__((noinline)) double bench(attn_fn fn,float *out,float *q,float *kv,
                             int *idx,int count,int hd,float *score,int repeats) {
  uint64_t start=nanos();
  for (int j=0;j<repeats;j++) {
    // Identical in both arms, but prevents optimizer from hoisting the call.
    idx[0]=(j & 1) ? 0 : 1;
    fn(out,q,kv,idx,count,hd,0.0f,1.0f/sqrtf((float)hd),score);
    checksum+=out[j % hd];
  }
  return (double)(nanos()-start)/repeats/1000.0;
}
static int one_case(int hd,int count,int holes,float amplitude,int repeats) {
  int rows=count+2;
  float *q=malloc(hd*sizeof(float));
  float *kv=malloc((size_t)rows*hd*sizeof(float));
  float *oa=malloc(hd*sizeof(float));
  float *ob=malloc(hd*sizeof(float));
  float *sa=malloc(count*sizeof(float));
  float *sb=malloc(count*sizeof(float));
  int *idx=malloc(count*sizeof(int));
  if (!q||!kv||!oa||!ob||!sa||!sb||!idx) { fprintf(stderr,"OOM\n"); return 1; }
  for(int i=0;i<hd;i++) q[i]=rnd(amplitude);
  for(int i=0;i<rows*hd;i++) kv[i]=rnd(amplitude);
  for(int j=0;j<count;j++) idx[j]=(holes && j%holes==0) ? -1 : (int)(rand32()%rows);
  float scale=1.0f/sqrtf((float)hd);
  coli_sparse_attend_scalar(oa,q,kv,idx,count,hd,0,scale,sa);
  coli_sparse_attend(ob,q,kv,idx,count,hd,0,scale,sb);
  float worst=0,worst_score=0,maxout=0;
  int bad_class=0;
  for(int i=0;i<hd;i++) {
    if (isfinite(oa[i])!=isfinite(ob[i])) bad_class++;
    if (isfinite(oa[i])&&isfinite(ob[i])) {
      float delta=fabsf(oa[i]-ob[i]);
      if(delta>worst) worst=delta;
      if(fabsf(oa[i])>maxout) maxout=fabsf(oa[i]);
    }
  }
  for(int j=0;j<count;j++) {
    if(isfinite(sa[j])!=isfinite(sb[j])) bad_class++;
    if(isfinite(sa[j])&&isfinite(sb[j])) {
      float delta=fabsf(sa[j]-sb[j]);
      if(delta>worst_score) worst_score=delta;
    }
  }
  // Paired, alternating warm measurements; median over six trials.
  double a[6],b[6];
  for(int t=0;t<6;t++) {
    if((t&1)==0) {
      a[t]=bench(coli_sparse_attend_scalar,oa,q,kv,idx,count,hd,sa,repeats);
      b[t]=bench(coli_sparse_attend,ob,q,kv,idx,count,hd,sb,repeats);
    } else {
      b[t]=bench(coli_sparse_attend,ob,q,kv,idx,count,hd,sb,repeats);
      a[t]=bench(coli_sparse_attend_scalar,oa,q,kv,idx,count,hd,sa,repeats);
    }
  }
  for(int i=1;i<6;i++) for(int j=i;j>0&&a[j]<a[j-1];j--) {
    double v=a[j]; a[j]=a[j-1]; a[j-1]=v;
  }
  for(int i=1;i<6;i++) for(int j=i;j>0&&b[j]<b[j-1];j--) {
    double v=b[j]; b[j]=b[j-1]; b[j-1]=v;
  }
  double scalar=(a[2]+a[3])/2,vec=(b[2]+b[3])/2;
  printf("case hd=%d count=%d holes=%d amp=%.2f; max_out_delta=%.8g max_score_delta=%.8g max_out=%.6g class_diff=%d; scalar_us=%.3f vector_us=%.3f speedup=%.3fx\n",
        hd,count,holes,amplitude,worst,worst_score,maxout,bad_class,scalar,vec,scalar/vec);
  free(q);free(kv);free(oa);free(ob);free(sa);free(sb);free(idx);
  return bad_class!=0;
}
int main(void) {
#if defined(COLI_SPARSE_ATTN_NEON)
  puts("compiled_path=NEON");
#else
  puts("compiled_path=OTHER");
#endif
  int failures=0;
  failures+=one_case(512,640,5,1.0f,25);
  failures+=one_case(512,2048,7,1.0f,8);
  failures+=one_case(512,4096,17,1.0f,5);
  failures+=one_case(513,640,0,1.0f,25);
  failures+=one_case(511,640,3,1.0f,25);
  failures+=one_case(64,48,0,1.0f,100);
  failures+=one_case(512,640,5,8.0f,25);
  fprintf(stderr,"checksum=%g\n",checksum);
  return failures != 0;
}
