/* Actual GCC/AVX Qwen38 arithmetic, exported for an independent SYCL reader.
 * Model-free, no device init. Compile with the same NOCUDA flags as
 * test_qwen38_idot. Never replace the native CPU NLL reference with this file.
 */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "qwen38_fp8_fixture.h"
#include <float.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(__AVX2__) || !defined(__FMA__)
#error "The Qwen38 stage exporter requires the actual AVX2/FMA CPU path"
#endif

static void qfs_require(int ok,const char *what) {
    if(!ok){fprintf(stderr,"FAIL Qwen38 fixture: %s (errno=%d)\n",what,errno);exit(1);}
}
static void *qfs_alloc(size_t bytes) {
    void *p=malloc(bytes);qfs_require(p!=NULL,"allocation");return p;
}
static uint32_t qfs_rng;
static uint32_t qfs_random(void) {
    qfs_rng^=qfs_rng<<13;qfs_rng^=qfs_rng>>17;qfs_rng^=qfs_rng<<5;return qfs_rng;
}
static float qfs_sample(void) {return (qfs_random()&65535u)/32768.f-1.f;}
static size_t qfs_written;
static void qfs_write(FILE *f,const void *p,size_t size,size_t count) {
    qfs_require(size==0 || count<=QFS_MAX_BYTES/size,"write size bound");
    size_t bytes=size*count;
    qfs_require(qfs_written<=QFS_MAX_BYTES-bytes,"file size bound");
    qfs_require(fwrite(p,size,count,f)==count,"write");qfs_written+=bytes;
}
static void qfs_environment(void) {
    uint32_t main_csr=_mm_getcsr();
    printf("CPU_ENV main mxcsr=0x%08x rounding=%u ftz=%u daz=%u\n",main_csr,
        (main_csr>>13)&3u,(main_csr>>15)&1u,(main_csr>>6)&1u);
    int cap=omp_get_max_threads(),threads=0;
    qfs_require(cap>=1 && cap<=256,"OpenMP thread bound");
    uint32_t *csr=qfs_alloc((size_t)cap*sizeof(*csr));
    int *bad=calloc((size_t)cap,sizeof(*bad));qfs_require(bad!=NULL,"decoder results");
    #pragma omp parallel
    {
        int tid=omp_get_thread_num();csr[tid]=_mm_getcsr();
        #pragma omp single
        threads=omp_get_num_threads();
        for(int base=0;base<256;base+=8){
            uint8_t codes[8];float got[8];
            for(int i=0;i<8;i++)codes[i]=(uint8_t)(base+i);
            _mm256_storeu_ps(got,e4m3_decode8(codes));
            for(int i=0;i<8;i++){
                float want=e4m3_decode(codes[i]);
                if(isnan(want)){if(!isnan(got[i]))bad[tid]++;}
                else if(!isfinite(got[i])||memcmp(&want,&got[i],sizeof(float)))bad[tid]++;
            }
        }
    }
    int failures=0;
    for(int t=0;t<threads;t++){
        printf("CPU_ENV worker=%d mxcsr=0x%08x rounding=%u ftz=%u daz=%u decoder_mismatches=%d\n",
            t,csr[t],(csr[t]>>13)&3u,(csr[t]>>15)&1u,(csr[t]>>6)&1u,bad[t]);
        if((csr[t]&0xe040u)||bad[t])failures++;
    }
    free(csr);free(bad);
    qfs_require(!(main_csr&0xe040u)&&failures==0,
        "unmodified rounding/FTZ/DAZ and all256 decoder classifications");
    printf("CPU_ENV PASS threads=%d all256 codes including signed zero and NaNs\n",threads);
}
static void qfs_fill(uint8_t *q,float *sc,int K,int O,int pattern) {
    for(int o=0;o<O;o++)for(int k=0;k<K;k++){
        uint8_t b=(uint8_t)qfs_random();if((b&127)==127)b&=254;
        if(pattern==2)b=(k%128)<64?0x38:0xb8;
        q[(size_t)o*K+k]=b;
    }
    size_t n=(size_t)((K+127)/128)*((O+127)/128);
    for(size_t j=0;j<n;j++){
        float mant=1.25f+qfs_sample()*.25f;
        sc[j]=ldexpf(mant,-12+(int)(qfs_random()%4));
    }
}
static void qfs_compare(const char *stage,const float *a,const float *b,size_t n) {
    double err=0,ref=0,maxabs=0;size_t differing=0;
    for(size_t j=0;j<n;j++){
        qfs_require(isfinite(a[j])&&isfinite(b[j]),"nonfinite fixture result");
        double d=(double)a[j]-b[j];err+=d*d;ref+=(double)a[j]*a[j];
        if(fabs(d)>maxabs)maxabs=fabs(d);
        differing+=memcmp(a+j,b+j,sizeof(float))!=0;
    }
    printf("CPU_STAGE %s vector_vs_table relative_rms=%.12g max_abs=%.12g differing=%zu/%zu\n",
        stage,sqrt(err/(ref+1e-30)),maxabs,differing,n);
}
static void qfs_case(FILE *f,uint32_t id,int D,int I,int S,int pattern) {
    qfs_require(D>0&&D<=(int)QFS_MAX_DIM&&I>0&&I<=(int)QFS_MAX_DIM&&
        S>0&&S<=(int)QFS_MAX_ROWS,"case dimensions");
    uint32_t hdr[5]={id,(uint32_t)D,(uint32_t)I,(uint32_t)S,(uint32_t)pattern};
    size_t nq=(size_t)D*I,ns=(size_t)((D+127)/128)*((I+127)/128);
    size_t nx=(size_t)S*D,nh=(size_t)S*I;
    uint8_t *q[3];float *sc[3];
    qfs_rng=0xc001d00du^((uint32_t)D*65537u)^((uint32_t)I*17u);
    for(int w=0;w<3;w++){
        q[w]=qfs_alloc(nq);sc[w]=qfs_alloc(ns*sizeof(float));
        qfs_fill(q[w],sc[w],w==2?I:D,w==2?D:I,pattern);
    }
    float *x=qfs_alloc(nx*sizeof(float));
    for(size_t j=0;j<nx;j++){
        float v=qfs_sample()*1.25f;
        if(pattern==1)v=0.f;
        if(pattern==2)v=(j%(size_t)D)%64==0?0x1p24f:1.f;
        x[j]=v;
    }
    float *v[4],*t[4];
    for(int k=0;k<4;k++){
        size_t n=k==3?nx:nh;v[k]=qfs_alloc(n*sizeof(float));t[k]=qfs_alloc(n*sizeof(float));
    }
    printf("CPU_CASE START id=%u D=%d I=%d S=%d pattern=%d\n",id,D,I,S,pattern);fflush(stdout);
    q38_matmul_fp8_vec(v[0],x,q[0],sc[0],S,D,I);
    q38_matmul_fp8_vec(v[1],x,q[1],sc[1],S,D,I);
    matmul_fp8(t[0],x,q[0],sc[0],S,D,I);
    matmul_fp8(t[1],x,q[1],sc[1],S,D,I);
    for(size_t j=0;j<nh;j++){
        v[2][j]=q38_silu(v[0][j])*v[1][j];
        t[2][j]=q38_silu(t[0][j])*t[1][j];
    }
    q38_matmul_fp8_vec(v[3],v[2],q[2],sc[2],S,I,D);
    matmul_fp8(t[3],t[2],q[2],sc[2],S,I,D);
    const char *names[4]={"gate","up","hidden","down"};
    for(int k=0;k<4;k++)qfs_compare(names[k],v[k],t[k],k==3?nx:nh);
    qfs_write(f,hdr,sizeof(uint32_t),5);
    for(int w=0;w<3;w++){qfs_write(f,q[w],1,nq);qfs_write(f,sc[w],sizeof(float),ns);}
    qfs_write(f,x,sizeof(float),nx);
    for(int k=0;k<4;k++)qfs_write(f,v[k],sizeof(float),k==3?nx:nh);
    for(int k=0;k<4;k++)qfs_write(f,t[k],sizeof(float),k==3?nx:nh);
    for(int w=0;w<3;w++){free(q[w]);free(sc[w]);}
    for(int k=0;k<4;k++){free(v[k]);free(t[k]);}free(x);
    printf("CPU_CASE END id=%u bytes_so_far=%zu\n",id,qfs_written);fflush(stdout);
}
#ifndef QFS_NO_MAIN
int main(int argc,char **argv) {
    qfs_require(argc==2,"usage: export_qwen38_fp8_stages NEW_OUTPUT.bin");
    uint32_t endian=1;
    qfs_require(*(uint8_t*)&endian==1&&sizeof(float)==4&&FLT_RADIX==2&&FLT_MANT_DIG==24,
        "little-endian IEEE binary32 host");
    qfs_environment();
    char part[PATH_MAX],dir[PATH_MAX];
    qfs_require(strlen(argv[1])+6<sizeof(part),"output path bound");
    struct stat st;errno=0;
    qfs_require(lstat(argv[1],&st)<0&&errno==ENOENT,"output must not exist");
    snprintf(part,sizeof(part),"%s.part",argv[1]);
    int fd=open(part,O_WRONLY|O_CREAT|O_EXCL,0600);qfs_require(fd>=0,"exclusive partial file");
    FILE *f=fdopen(fd,"wb");qfs_require(f!=NULL,"fdopen");
    uint32_t version=QFS_VERSION,count=24,id=0;
    qfs_write(f,QFS_MAGIC,1,8);qfs_write(f,&version,4,1);qfs_write(f,&count,4,1);
    const int shapes[4][2]={{128,128},{200,136},{201,300},{2560,640}};
    const int rows[5]={1,2,5,8,9};
    for(int shape=0;shape<4;shape++)for(int s=0;s<5;s++)
        qfs_case(f,id++,shapes[shape][0],shapes[shape][1],rows[s],0);
    qfs_case(f,id++,128,128,1,1);qfs_case(f,id++,128,128,5,1);
    qfs_case(f,id++,256,128,1,2);qfs_case(f,id++,256,128,5,2);
    qfs_require(id==count,"case count");
    qfs_require(fflush(f)==0&&fsync(fd)==0,"flush complete file");
    qfs_require(fclose(f)==0,"close complete file");
    /* link publishes atomically without replacing a racing/existing target. */
    qfs_require(link(part,argv[1])==0,"publish without replacement");
    qfs_require(unlink(part)==0,"remove partial name");
    strcpy(dir,argv[1]);char *slash=strrchr(dir,'/');
    if(!slash)strcpy(dir,".");else if(slash==dir)slash[1]=0;else *slash=0;
    int dfd=open(dir,O_RDONLY|O_DIRECTORY);qfs_require(dfd>=0,"output directory");
    qfs_require(fsync(dfd)==0&&close(dfd)==0,"sync output directory");
    printf("CPU_FIXTURE COMPLETE cases=%u bytes=%zu path=%s\n",count,qfs_written,argv[1]);
    return 0;
}
#endif
