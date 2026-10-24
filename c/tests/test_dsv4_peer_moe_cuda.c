/* The fused decode MoE (dsv4_cuda_moe) with routed experts on a peer GPU.
 *
 * Peer compute (DSV4_CUDA_PEER_COMPUTE, default 1) runs the experts mirrored
 * on the peer ON the peer and writes their down-projection rows into the
 * primary's reduce buffer; =0 makes the primary read them in place over P2P.
 * Both are claimed bit-identical to computing every expert on the primary:
 * checked here for every on-peer mask of 6 experts.
 *
 * Then a peer failure is injected (DSV4_CUDA_PEER_FAULT, at the join, the
 * way an asynchronous fault on the peer surfaces). While the peer still
 * answers, the failing call itself must return the all-on-primary bytes (its
 * experts are recomputed on the primary from the same mirrors); with ":lost"
 * the call fails cleanly, so the engine can fall back to host copies. Either
 * way the peer is latched off once, with one log line: dsv4_cuda_peer_healthy
 * turns 0, peer compute and peer access report 0, tensors on the peer are
 * refused (the MoE, the expert group and the matvec the engine's CPU
 * fallback could reach), and an all-on-primary call right after gives the
 * reference bytes (no stale error left behind).
 *
 * ":sticky" injects the kind of error a real fault on the peer leaves (a
 * launch failure, sticky for the process's CUDA: the F18 probe showed a
 * __trap() on the peer takes device 0's context with it). The backend then
 * latches CUDA off for the whole process: the failing call is refused,
 * dsv4_cuda_lost turns 1, and every call after it, all-on-primary ones too,
 * is refused without touching CUDA, so the engine takes its host paths.
 *
 * The backend reads those variables once per process, so the test re-runs
 * itself once per scenario ("--child <name>"). Needs two devices with peer
 * access both ways (skips otherwise): DSV4_CUDA_DEVICE and
 * DSV4_CUDA_PEER_DEVICE, 0 and 1 by default. Tiny: under 3 MiB of tensors
 * per device beyond the CUDA context. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
static void env_set(const char *name, const char *value){_putenv_s(name, value ? value : "");}
#else
static void env_set(const char *name, const char *value){if(value)setenv(name, value, 1);else unsetenv(name);}
#endif

enum { H=512, I=256, E=6, SKIP=77 };
static uint64_t g_rng;
static uint32_t rnd(void){g_rng=g_rng*6364136223846793005ull+1442695040888963407ull;return (uint32_t)(g_rng>>33);}
static void fill(uint8_t *p,size_t n,int lo,int span){for(size_t i=0;i<n;i++)p[i]=(uint8_t)(span?lo+(int)(rnd()%(unsigned)span):(int)rnd());}
/* e4m3 codes without the two NaNs */
static void fill_fp8(uint8_t *p,size_t n){for(size_t i=0;i<n;i++){uint8_t v=(uint8_t)rnd();p[i]=(v&0x7f)==0x7f?v^1:v;}}

static Dsv4CudaTensor *fp4(const uint8_t *w,const uint8_t *s,int O,int In,int dev){Dsv4CudaTensor*t=NULL;return dsv4_cuda_upload_fp4(&t,w,s,O,In,dev)?t:NULL;}
static Dsv4CudaTensor *fp8(int O,int In,int dev){
    size_t wn=(size_t)O*In,sn=(size_t)((O+127)/128)*((In+127)/128);uint8_t*w=malloc(wn),*s=malloc(sn);Dsv4CudaTensor*t=NULL;
    if(w&&s){fill_fp8(w,wn);fill(s,sn,118,4);if(!dsv4_cuda_upload_fp8(&t,w,s,O,In,dev))t=NULL;}free(w);free(s);return t;}

static int child(const char *mode){
    const char*a=getenv("DSV4_CUDA_DEVICE"),*b=getenv("DSV4_CUDA_PEER_DEVICE");int devs[2]={a?atoi(a):0,b?atoi(b):1};
    if(devs[0]==devs[1]||!dsv4_cuda_init(devs,2)){printf("SKIP: devices %d and %d did not initialise\n",devs[0],devs[1]);return SKIP;}
    int p=devs[0],q=devs[1];
    if(!dsv4_cuda_peer_access(p,q)){printf("SKIP: no peer access between devices %d and %d\n",p,q);dsv4_cuda_shutdown();return SKIP;}
    int compute=dsv4_cuda_peer_compute(p,q);
    if(!dsv4_cuda_peer_healthy(q)||(!strcmp(mode,"inplace")&&compute)){fprintf(stderr,"%s: fresh peer: healthy %d, compute %d\n",mode,dsv4_cuda_peer_healthy(q),compute);return 2;}
    /* the same expert weights on both devices: P on the primary, Q on the peer */
    g_rng=0x5eed;
    Dsv4CudaTensor*gP[E],*uP[E],*dP[E],*gQ[E],*uQ[E],*dQ[E];
    size_t w1=(size_t)I*H/2,s1=(size_t)I*H/32,w2=(size_t)H*I/2,s2=(size_t)H*I/32;
    uint8_t*gw=malloc(w1),*gs=malloc(s1),*uw=malloc(w1),*us=malloc(s1),*dw=malloc(w2),*ds=malloc(s2);if(!gw||!gs||!uw||!us||!dw||!ds)return 3;
    for(int e=0;e<E;e++){
        fill(gw,w1,0,0);fill(gs,s1,119,4);fill(uw,w1,0,0);fill(us,s1,119,4);fill(dw,w2,0,0);fill(ds,s2,120,4);
        if(!(gP[e]=fp4(gw,gs,I,H,p))||!(uP[e]=fp4(uw,us,I,H,p))||!(dP[e]=fp4(dw,ds,H,I,p))||
           !(gQ[e]=fp4(gw,gs,I,H,q))||!(uQ[e]=fp4(uw,us,I,H,q))||!(dQ[e]=fp4(dw,ds,H,I,q)))return 3;}
    free(gw);free(gs);free(uw);free(us);free(dw);free(ds);
    Dsv4CudaTensor*sg=fp8(I,H,p),*su=fp8(I,H,p),*sd=fp8(H,I,p);if(!sg||!su||!sd)return 3;
    float x[H],wt[E]={.31f,.22f,.17f,.13f,.10f,.07f},ref[H],got[H];for(int i=0;i<H;i++)x[i]=(float)((int)(rnd()%2001)-1000)/1000.f;
    if(!dsv4_cuda_moe(gP,uP,dP,wt,E,sg,su,sd,10.f,ref,x)){fprintf(stderr,"%s: all-on-primary MoE failed\n",mode);return 4;}
    float peak=0;for(int i=0;i<H;i++){if(!isfinite(ref[i]))return 4;peak=fmaxf(peak,fabsf(ref[i]));}if(peak<=0)return 4;
    Dsv4CudaTensor*g[E],*u[E],*d[E];
#define MASK(m) for(int e=0;e<E;e++){int on=((m)>>e)&1;g[e]=on?gQ[e]:gP[e];u[e]=on?uQ[e]:uP[e];d[e]=on?dQ[e]:dP[e];}
#define SAME() (memcmp(got,ref,sizeof ref)==0)
    if(!strcmp(mode,"compute")||!strcmp(mode,"inplace")){
        int bad=0;for(int m=0;m<(1<<E);m++){MASK(m);memset(got,0,sizeof got);
            if(!dsv4_cuda_moe(g,u,d,wt,E,sg,su,sd,10.f,got,x)||!SAME()){fprintf(stderr,"%s: on-peer mask 0x%02x differs from all-on-primary\n",mode,m);bad++;}}
        if(bad||!dsv4_cuda_peer_healthy(q))return 5;
        printf("%s (peer compute %s): all %d on-peer masks bit-identical to all-on-primary\n",mode,compute?"on":"off",1<<E);
    }else{
        /* soft: the 3rd peer pass fails, the peer still answers; lost: the 1st one, the peer counts as lost;
         * sticky: the 1st one, with a sticky error */
        int soft=!strcmp(mode,"soft"),stk=!strcmp(mode,"sticky");
        if(!compute){printf("SKIP: peer compute is off for devices %d and %d (DSV4_CUDA_BATCHED=0, or different SM versions)\n",p,q);dsv4_cuda_shutdown();return SKIP;}
        if(soft){for(int m=1;m<=2;m++){MASK(m);if(!dsv4_cuda_moe(g,u,d,wt,E,sg,su,sd,10.f,got,x)||!SAME()){fprintf(stderr,"soft: pass %d before the fault differs\n",m);return 6;}}}
        int fm=soft?0x2d:0x01;MASK(fm);memset(got,0,sizeof got);
        int r=dsv4_cuda_moe(g,u,d,wt,E,sg,su,sd,10.f,got,x);
        if(soft&&(!r||!SAME())){fprintf(stderr,"soft: the failing call returned %d, output %s all-on-primary\n",r,SAME()?"equal to":"differs from");return 6;}
        if(!soft&&r){fprintf(stderr,"%s: the failing call claimed success\n",mode);return 6;}
        if(dsv4_cuda_lost()!=stk){fprintf(stderr,"%s: dsv4_cuda_lost() is %d after the fault (want %d)\n",mode,dsv4_cuda_lost(),stk);return 7;}
        if(dsv4_cuda_peer_healthy(q)||dsv4_cuda_peer_compute(p,q)||dsv4_cuda_peer_access(p,q)){
            fprintf(stderr,"%s: after the fault: healthy %d, compute %d, access %d (want 0 0 0)\n",mode,dsv4_cuda_peer_healthy(q),dsv4_cuda_peer_compute(p,q),dsv4_cuda_peer_access(p,q));return 7;}
        MASK(0x05);if(dsv4_cuda_moe(g,u,d,wt,E,sg,su,sd,10.f,got,x)){fprintf(stderr,"%s: a latched peer's tensors were used\n",mode);return 8;}
        if(dsv4_cuda_expert_group(gQ,uQ,dQ,wt,E,10.f,got,x)){fprintf(stderr,"%s: the expert group ran on the latched peer\n",mode);return 8;}
        if(dsv4_cuda_matvec_grouped(gQ[0],got,x,1)){fprintf(stderr,"%s: a matvec ran on the latched peer\n",mode);return 8;}
        if(stk){
            if(dsv4_cuda_moe(gP,uP,dP,wt,E,sg,su,sd,10.f,got,x)||dsv4_cuda_expert_group(gP,uP,dP,wt,E,10.f,got,x)||
               dsv4_cuda_matvec_grouped(gP[0],got,x,1)||dsv4_cuda_mem_free_mb(p)>=0){fprintf(stderr,"sticky: a call on the primary ran after CUDA was lost\n");return 9;}
            printf("sticky: failing call refused; CUDA lost for the process, every later call refused (host paths)\n");
            dsv4_cuda_shutdown();return 0;
        }
        memset(got,0,sizeof got);if(!dsv4_cuda_moe(gP,uP,dP,wt,E,sg,su,sd,10.f,got,x)||!SAME()){fprintf(stderr,"%s: all-on-primary after the latch differs\n",mode);return 9;}
        if(!dsv4_cuda_expert_group(gP,uP,dP,wt,E,10.f,got,x)){fprintf(stderr,"%s: the expert group on the primary failed after the latch\n",mode);return 9;}
        if(!dsv4_cuda_matvec_grouped(gP[0],got,x,1)){fprintf(stderr,"%s: a matvec on the primary failed after the latch\n",mode);return 9;}
        printf("%s: failing call %s; peer reported unhealthy and refused, primary unaffected\n",mode,soft?"bit-identical (recomputed on the primary)":"refused (host-data fallback)");
    }
    for(int e=0;e<E;e++){dsv4_cuda_tensor_free(gP[e]);dsv4_cuda_tensor_free(uP[e]);dsv4_cuda_tensor_free(dP[e]);dsv4_cuda_tensor_free(gQ[e]);dsv4_cuda_tensor_free(uQ[e]);dsv4_cuda_tensor_free(dQ[e]);}
    dsv4_cuda_tensor_free(sg);dsv4_cuda_tensor_free(su);dsv4_cuda_tensor_free(sd);dsv4_cuda_shutdown();return 0;
}

int main(int argc,char **argv){
    if(argc>2&&!strcmp(argv[1],"--child"))return child(argv[2]);
    static const struct { const char *mode,*compute,*fault; int latch; } runs[]={
        {"compute",NULL,NULL,0},{"inplace","0",NULL,0},{"soft",NULL,"3",1},{"lost",NULL,"1:lost",1},{"sticky",NULL,"1:sticky",1}};
    /* the context only loads the kernels this test launches */
    if(!getenv("CUDA_MODULE_LOADING"))env_set("CUDA_MODULE_LOADING","LAZY");
    for(size_t k=0;k<sizeof runs/sizeof*runs;k++){
        env_set("DSV4_CUDA_PEER_COMPUTE",runs[k].compute);env_set("DSV4_CUDA_PEER_FAULT",runs[k].fault);
        char cmd[4096],out[1<<14];if(snprintf(cmd,sizeof cmd,"\"%s\" --child %s 2>&1",argv[0],runs[k].mode)>=(int)sizeof cmd)return 1;
        FILE*f=popen(cmd,"r");if(!f)return 1;size_t n=fread(out,1,sizeof out-1,f);out[n]=0;int st=pclose(f);
#ifndef _WIN32
        if(st!=-1&&(st&0x7f)==0)st=(st>>8)&0xff;
#endif
        if(st==SKIP){const char*why=strstr(out,"SKIP");printf("  %s: %s",runs[k].mode,why?why:"SKIP\n");if(k==0){puts("dsv4 peer MoE: SKIP");return 0;}continue;}
        int latches=0;for(const char*s=out;(s=strstr(s,"latched off"))!=NULL;s++)latches++;
        if(st||latches!=runs[k].latch){fprintf(stderr,"%s: status %d, %d latch line(s) (want %d):\n%s",runs[k].mode,st,latches,runs[k].latch,out);return 1;}
        for(char*line=out,*nl;*line;line=nl?nl+1:line+strlen(line)){nl=strchr(line,'\n');if(nl)*nl=0;if(strstr(line,"latched off")||strstr(line,"bit-identical")||strstr(line,"refused")||strstr(line,"lost for this process"))printf("  %s\n",line);}
    }
    puts("dsv4 peer MoE: OK");return 0;
}
