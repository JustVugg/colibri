/* The generic (non-DeepGEMM) kernel variants must agree bit for bit.
 *
 * dsv4_cuda_route_moe_ids_batch, the generic batched prefill MoE, has two
 * routed-expert kernels: one fp4 matvec per route (DSV4_MOE_EXPERT_MAJOR=0)
 * and the expert-major default, DSV4_MOE_EM_TB routes per pass. run_mm_batch
 * (dsv4_cuda_matmul_batch, and the shared experts of the MoE above) has the
 * block kernel and the warp-per-row one (DSV4_MM_WARP, on fp8 shapes whose I
 * is a multiple of 256: auto = I <= 1024, 1 = all of them, 0 = never). Each
 * pair is claimed bit-identical.
 *
 * The backend reads those variables once per process, so this test re-runs
 * itself once per setting ("--child") and compares every child's output
 * hashes with the default run's; the _VERIFY settings add the backend's own
 * raw fp32 comparison of both kernels. Each child also checks the outputs
 * against the per-token decode entry points (dsv4_cuda_moe, dsv4_cuda_matvec),
 * so variants that agree on garbage still fail. Those agree exactly today,
 * but the check allows rounding: decode kernels are free to change.
 *
 * Small on purpose (about 450 MiB of device memory per child, the CUDA
 * context included): a 9-expert bank (~115 MiB) at the H=4096 / I=2048 the
 * entry point requires, plus ~70 MiB of dense weights. Chunk and token counts
 * include 1 and tails that are not multiples of any TB or tile. DeepGEMM
 * builds have neither kernel pair, so the test skips there. */
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

enum { H=4096, I=2048, E=9, K=6, MAXT=128 };
#define COUNT(a) ((int)(sizeof(a)/sizeof(*(a))))
static const int chunks[]={1,2,3,5,8,13,31,64,127,128};           /* MoE chunk sizes */
static const struct { int O,In,fmt; } shapes[]={{32768,1024,8},{32768,1024,9},{1000,1024,8},{384,512,9},{520,2048,8}};
static const int tokens[]={1,5,8,13,16,31,128};                    /* dense batch sizes */

static uint64_t g_rng;
static uint32_t rnd(void){g_rng=g_rng*6364136223846793005ull+1442695040888963407ull;return (uint32_t)(g_rng>>33);}
static void seed(uint64_t s){g_rng=s*0x9e3779b97f4a7c15ull+1;rnd();}
static uint64_t fnv(const void *p,size_t n){const uint8_t*b=p;uint64_t h=1469598103934665603ull;for(size_t i=0;i<n;i++){h^=b[i];h*=1099511628211ull;}return h;}
static void fill_bytes(uint8_t *p,size_t n){for(size_t i=0;i<n;i++)p[i]=(uint8_t)rnd();}
static void fill_scales(uint8_t *p,size_t n,int lo,int span){for(size_t i=0;i<n;i++)p[i]=(uint8_t)(lo+(int)(rnd()%(unsigned)span));}
/* e4m3 codes without the two NaNs (0x7f, 0xff); -0 (0x80) stays in. */
static void fill_fp8(uint8_t *p,size_t n){for(size_t i=0;i<n;i++){uint8_t v=(uint8_t)rnd();p[i]=(v&0x7f)==0x7f?v^1:v;}}
static void fill_x(float *x,size_t n){for(size_t i=0;i<n;i++){uint32_t r=rnd();x[i]=r%97==0?-0.f:(float)((int)(r%2001)-1000)/1000.f;}}

static Dsv4CudaTensor *fp8_tensor(int O,int In,int fmt){
    size_t wn=(size_t)O*In,sn=(size_t)((O+127)/128)*((In+127)/128);uint8_t*w=malloc(wn),*s=malloc(sn);Dsv4CudaTensor*t=NULL;
    if(!w||!s){free(w);free(s);return NULL;}
    fill_fp8(w,wn);fill_scales(s,sn,In>=4096?114:116,5);
    int ok=fmt==9?dsv4_cuda_upload_fp8_bf16(&t,w,s,O,In,0):dsv4_cuda_upload_fp8(&t,w,s,O,In,0);free(s);free(w);return ok?t:NULL;
}

/* |got-ref| over one row must stay within rel of the row's largest |ref|. */
static int close_rows(const float *got,const float *ref,int n,float rel,const char *what,int a,int b){
    float worst=0,peak=0;for(int i=0;i<n;i++){peak=fmaxf(peak,fabsf(ref[i]));worst=fmaxf(worst,fabsf(got[i]-ref[i]));}
    for(int i=0;i<n;i++)if(!isfinite(got[i])){fprintf(stderr,"%s %d/%d: non-finite output at %d\n",what,a,b,i);return 0;}
    if(peak<=0||worst>rel*peak){fprintf(stderr,"%s %d/%d: max |diff| %g against peak %g\n",what,a,b,worst,peak);return 0;}
    return 1;
}

/* Hash lines go out one write each (fflush): the parent reads the child's
 * stdout and stderr through one pipe and must not see them interleaved. */
static int child(void){
    int dev=0;if(!dsv4_cuda_init(&dev,1))return 1;
    if(dsv4_cuda_expert_bank_slots(E)!=E){fprintf(stderr,"generic bank must size to the expert count: %d slots for %d\n",dsv4_cuda_expert_bank_slots(E),E);return 2;}
    /* ---- generic batched MoE over host routes ---- */
    seed(1);
    Dsv4CudaTensor*sg=fp8_tensor(I,H,8),*su=fp8_tensor(I,H,8),*sd=fp8_tensor(H,I,8);if(!sg||!su||!sd)return 3;
    Dsv4CudaExpertSet*bank=dsv4_cuda_expert_bank_create(E,H,I,dev,sg,su,sd);if(!bank)return 3;
    Dsv4CudaTensor*g[E],*u[E],*d[E];
    size_t w1=(size_t)I*H/2,s1=(size_t)I*H/32,w2=(size_t)H*I/2,s2=(size_t)H*I/32;
    uint8_t*gw=malloc(w1),*gs=malloc(s1),*uw=malloc(w1),*us=malloc(s1),*dw=malloc(w2),*ds=malloc(s2);if(!gw||!gs||!uw||!us||!dw||!ds)return 4;
    for(int e=0;e<E;e++){
        fill_bytes(gw,w1);fill_bytes(uw,w1);fill_bytes(dw,w2);fill_scales(gs,s1,118,7);fill_scales(us,s1,118,7);fill_scales(ds,s2,118,7);
        g[e]=u[e]=d[e]=NULL;if(!dsv4_cuda_expert_bank_upload(bank,e,gw,gs,uw,us,dw,ds,&g[e],&u[e],&d[e]))return 5;}
    free(ds);free(dw);free(us);free(uw);free(gs);free(gw);
    float*x=malloc((size_t)MAXT*H*sizeof(float)),*got=malloc((size_t)MAXT*H*sizeof(float)),*ref=malloc((size_t)H*sizeof(float)),*wt=malloc((size_t)MAXT*K*sizeof(float));
    int*ids=malloc((size_t)MAXT*K*sizeof(int));
    Dsv4CudaActivation*in=dsv4_cuda_activation_create(dev,(long long)MAXT*H),*out=dsv4_cuda_activation_create(dev,(long long)MAXT*H);
    if(!x||!got||!ref||!wt||!ids||!in||!out)return 6;
    const float limit=7.f;
    for(int c=0;c<COUNT(chunks);c++){int T=chunks[c];seed(100+c);fill_x(x,(size_t)T*H);
        for(int t=0;t<T;t++)for(int k=0;k<K;k++){
            int e;
            if(T==127)e=E-1-k;      /* every token picks the same six: 127 routes per expert */
            else for(;;){int r=(int)(rnd()%(E+4));e=r<E?r:r-E;int dup=0;for(int j=0;j<k;j++)dup|=ids[t*K+j]==e;if(!dup)break;}
            ids[t*K+k]=e;wt[t*K+k]=(float)(50+rnd()%1000)/2000.f;}
        if(!dsv4_cuda_activation_upload(in,x,(long long)T*H)||!dsv4_cuda_route_moe_ids_batch(in,ids,wt,T,bank,limit,out)||
           !dsv4_cuda_activation_download(got,out,(long long)T*H)||!dsv4_cuda_activation_sync(out)){fprintf(stderr,"moe ids batch failed, count=%d\n",T);return 7;}
        printf("moe count=%d %016llx\n",T,(unsigned long long)fnv(got,(size_t)T*H*sizeof(float)));fflush(stdout);
        int probe[3]={0,T/2,T-1};
        for(int p=0;p<3;p++){int t=probe[p];Dsv4CudaTensor*pg[K],*pu[K],*pd[K];
            for(int k=0;k<K;k++){pg[k]=g[ids[t*K+k]];pu[k]=u[ids[t*K+k]];pd[k]=d[ids[t*K+k]];}
            if(!dsv4_cuda_moe(pg,pu,pd,wt+t*K,K,sg,su,sd,limit,ref,x+(size_t)t*H)){fprintf(stderr,"decode moe failed\n");return 8;}
            if(!close_rows(got+(size_t)t*H,ref,H,1e-2f,"moe vs decode, count/token",T,t))return 9;}
    }
    dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);
    for(int e=0;e<E;e++){dsv4_cuda_tensor_free(g[e]);dsv4_cuda_tensor_free(u[e]);dsv4_cuda_tensor_free(d[e]);}
    dsv4_cuda_expert_set_free(bank);dsv4_cuda_tensor_free(sd);dsv4_cuda_tensor_free(su);dsv4_cuda_tensor_free(sg);
    free(ids);free(wt);free(ref);free(got);free(x);
    /* ---- dense batch: warp kernel by default for I <= 1024 ---- */
    for(int s=0;s<COUNT(shapes);s++){int O=shapes[s].O,In=shapes[s].In;seed(200+s);
        Dsv4CudaTensor*w=fp8_tensor(O,In,shapes[s].fmt);
        float*dx=malloc((size_t)MAXT*In*sizeof(float)),*dy=malloc((size_t)MAXT*O*sizeof(float)),*dr=malloc((size_t)O*sizeof(float));
        Dsv4CudaActivation*din=dsv4_cuda_activation_create(dev,(long long)MAXT*In),*dout=dsv4_cuda_activation_create(dev,(long long)MAXT*O);
        if(!w||!dx||!dy||!dr||!din||!dout)return 10;
        for(int j=0;j<COUNT(tokens);j++){int T=tokens[j];fill_x(dx,(size_t)T*In);
            if(!dsv4_cuda_activation_upload(din,dx,(long long)T*In)||!dsv4_cuda_matmul_batch(w,din,T,dout)||
               !dsv4_cuda_activation_download(dy,dout,(long long)T*O)||!dsv4_cuda_activation_sync(dout)){fprintf(stderr,"matmul batch failed\n");return 11;}
            printf("dense %dx%d fmt=%d T=%d %016llx\n",O,In,shapes[s].fmt,T,(unsigned long long)fnv(dy,(size_t)T*O*sizeof(float)));fflush(stdout);
            int probe[2]={0,T-1};
            for(int p=0;p<2;p++){int t=probe[p];if(!dsv4_cuda_matvec(w,dr,dx+(size_t)t*In)){fprintf(stderr,"matvec failed\n");return 12;}
                if(!close_rows(dy+(size_t)t*O,dr,O,1e-4f,"dense vs matvec, T/token",T,t))return 13;}
        }
        dsv4_cuda_activation_free(dout);dsv4_cuda_activation_free(din);dsv4_cuda_tensor_free(w);free(dr);free(dy);free(dx);
    }
    dsv4_cuda_shutdown();return 0;
}

/* Every variable a setting below touches; each child starts with all unset.
 * The _VERIFY settings also run both kernels of a pair inside the backend and
 * compare their raw fp32 outputs (before the bf16 roundings that can hide a
 * difference in the MoE result); the backend prints a report when it counts a
 * mismatch, and any non-zero count fails the test. */
static const char *const knobs[]={"DSV4_MOE_EXPERT_MAJOR","DSV4_MOE_EM_TB","DSV4_MOE_EXPERT_MAJOR_VERIFY","DSV4_MM_WARP","DSV4_MM_WARP_VERIFY"};
static const char *const settings[][4]={
    {NULL},                                       /* defaults: expert-major TB=4, MM_WARP auto (the reference) */
    {"DSV4_MOE_EXPERT_MAJOR","0"},                /* one matvec per route */
    {"DSV4_MOE_EXPERT_MAJOR","1"},
    {"DSV4_MOE_EM_TB","2"},{"DSV4_MOE_EM_TB","3"},{"DSV4_MOE_EM_TB","4"},{"DSV4_MOE_EM_TB","6"},
    {"DSV4_MOE_EXPERT_MAJOR_VERIFY","1","DSV4_MOE_EM_TB","2"},{"DSV4_MOE_EXPERT_MAJOR_VERIFY","1","DSV4_MOE_EM_TB","3"},
    {"DSV4_MOE_EXPERT_MAJOR_VERIFY","1","DSV4_MOE_EM_TB","4"},{"DSV4_MOE_EXPERT_MAJOR_VERIFY","1","DSV4_MOE_EM_TB","6"},
    {"DSV4_MM_WARP","0"},                         /* block kernel everywhere */
    {"DSV4_MM_WARP","1"},                         /* warp kernel on every fp8 shape, the MoE's shared experts included */
    {"DSV4_MM_WARP","1","DSV4_MM_WARP_VERIFY","1"},
};

/* Runs one child under `set` and keeps only its hash lines in `hashes`:
 * 1 = ran clean, 0 = the child failed, -1 = the backend counted mismatches. */
static int run_child(const char *self,const char *const *set,char *hashes,size_t cap,char *label,size_t lcap){
    static char all[1<<17];size_t used=0;label[0]=0;
    for(int i=0;i<COUNT(knobs);i++)env_set(knobs[i],NULL);
    for(int i=0;i<4&&set[i];i+=2){env_set(set[i],set[i+1]);size_t l=strlen(label);snprintf(label+l,lcap-l,"%s%s=%s",l?" ":"",set[i],set[i+1]);}
    if(!label[0])snprintf(label,lcap,"defaults");
    char cmd[4096];if(snprintf(cmd,sizeof(cmd),"\"%s\" --child 2>&1",self)>=(int)sizeof(cmd))return 0;
    FILE*p=popen(cmd,"r");if(!p)return 0;
    size_t n=fread(all,1,sizeof(all)-1,p);all[n]=0;int status=pclose(p);
    if(status!=0){fprintf(stderr,"child (%s) failed with status %d:\n%s",label,status,all);return 0;}
    hashes[0]=0;
    for(char*line=all,*next;*line;line=next){
        char*nl=strchr(line,'\n');next=nl?nl+1:line+strlen(line);if(nl)*nl=0;
        if(!strncmp(line,"moe ",4)||!strncmp(line,"dense ",6)){size_t l=strlen(line);if(used+l+2>cap)return 0;memcpy(hashes+used,line,l);used+=l;hashes[used++]='\n';hashes[used]=0;continue;}
        if(!strstr(line,"verify calls="))continue;
        /* "fc1_mismatch=N fc2_mismatch=N" (MoE) or "mismatches=N" (dense) */
        for(const char*m=line;(m=strstr(m,"mismatch"))!=NULL;m+=8){const char*q=m+8;if(!strncmp(q,"es",2))q+=2;
            if(*q=='='&&strtoull(q+1,NULL,10)){fprintf(stderr,"%s: the backend's own check found mismatching bits:\n  %s\n",label,line);return -1;}}
    }
    return 1;
}

int main(int argc,char **argv){
    if(argc>1&&!strcmp(argv[1],"--child"))return child();
    if(strcmp(dsv4_cuda_backend_name(),"generic")){printf("dsv4 generic bitwise: SKIP (%s build has no generic kernel pairs)\n",dsv4_cuda_backend_name());return 0;}
    enum { CAP=1<<16 };static char ref[CAP],got[CAP];char label[256];
    int r=run_child(argv[0],settings[0],ref,CAP,label,sizeof(label));if(r<=0)return r<0?2:1;
    int lines=0,want=COUNT(chunks)+COUNT(shapes)*COUNT(tokens);for(const char*c=ref;*c;c++)lines+=*c=='\n';
    if(lines!=want){fprintf(stderr,"defaults: %d output hashes, expected %d\n",lines,want);return 1;}
    for(int s=1;s<COUNT(settings);s++){
        if((r=run_child(argv[0],settings[s],got,CAP,label,sizeof(label)))<=0)return r<0?2:1;
        if(strcmp(ref,got)){
            const char*a=ref,*b=got;while(*a&&*a==*b){a++;b++;}while(a>ref&&a[-1]!='\n'){a--;b--;}
            fprintf(stderr,"%s is not bit-identical to the defaults:\n  defaults: %.*s\n  %s: %.*s\n",label,
                    (int)strcspn(a,"\n"),a,label,(int)strcspn(b,"\n"),b);
            return 2;}
        printf("  %s: all %d outputs bit-identical\n",label,lines);
    }
    puts("dsv4 generic bitwise: OK");return 0;
}
