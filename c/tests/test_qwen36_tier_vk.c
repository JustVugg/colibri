/* Qwen3.6 Vulkan expert tier gate: drives qwen36_tier.c through the shared Vulkan
 * backend with synthetic experts (no model file) and checks the accumulated
 * routed-expert output against a CPU reference. Compiled twice: this file for
 * packed int4 experts, and test_qwen36_tier_vk_int8.c (which #defines
 * TIER_VK_INT8 1 before including this one) for raw int8 experts -- the Vulkan
 * backend cannot be re-initialised in-process, so the int8 scenario runs as a
 * separate executable rather than a second qt_init() here. Built into every
 * `make check`; without VK=1 both compile to a skip, and with VK=1 but no
 * usable device they skip at runtime (exit 0) so CPU hosts and CI stay green. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifndef TIER_VK_INT8
#define TIER_VK_INT8 0
#endif
#ifndef COLI_VULKAN
int main(void){ puts("skip: built without VK=1"); return 0; }
#else
#include "../qwen36_tier.h"
#include "../backend_vulkan.h"

static int fail(const char *m){ fprintf(stderr,"qwen36 tier vk test failed: %s\n",m); return 1; }
static uint32_t g_rng=0x9E3779B9u;
static uint32_t rnd(void){ g_rng=g_rng*1664525u+1013904223u; return g_rng>>8; }
static float frand(void){ return (float)(rnd()%2001)/1000.f-1.f; }   /* [-1,1] */

/* synthetic geometry (Qwen3.6-35B-A3B is D=2048, I=768; tiny fixture is 64/32) */
enum { NL=2, NE=8, D=64, IH=32, TOPK=2 };
typedef struct { uint8_t *g4,*u4,*d4; float *gs,*us,*ds; } Exp;
static Exp E[NL][NE];

/* int4: packed two's-complement (LOW nibble = even column), per-row f32 scales,
 * the container layout qwen36.c hands the tier; the tier XORs to offset-binary.
 * int8 (TIER_VK_INT8): raw signed bytes, one per element, same per-row scales. */
static void make_expert(Exp *e){
#if TIER_VK_INT8
    size_t mb=(size_t)D*IH;
#else
    size_t mb=(size_t)D*IH/2;
#endif
    e->g4=malloc(mb); e->u4=malloc(mb); e->d4=malloc(mb);
    e->gs=malloc(IH*sizeof(float)); e->us=malloc(IH*sizeof(float)); e->ds=malloc(D*sizeof(float));
    for(size_t i=0;i<mb;i++){ e->g4[i]=(uint8_t)rnd(); e->u4[i]=(uint8_t)rnd(); e->d4[i]=(uint8_t)rnd(); }
    for(int o=0;o<IH;o++){ e->gs[o]=0.01f+0.02f*(float)(rnd()%100)/100.f; e->us[o]=0.01f+0.02f*(float)(rnd()%100)/100.f; }
    for(int o=0;o<D;o++) e->ds[o]=0.01f+0.02f*(float)(rnd()%100)/100.f;
}
#if TIER_VK_INT8
static float deq(const uint8_t *row,int i){ return (float)((const int8_t*)row)[i]; }
/* y[O] = (x[I] . W[O,I]) * s[O]; int8 row stride is I bytes (no packing) */
static void gemv(float *y,const float *x,const uint8_t *w,const float *s,int I,int O){
    for(int o=0;o<O;o++){ const uint8_t *row=w+(size_t)o*I; double a=0;
        for(int i=0;i<I;i++) a+=x[i]*deq(row,i); y[o]=(float)a*s[o]; }
}
#else
static float deq(const uint8_t *row,int i){ int nib=(i&1)?(row[i>>1]>>4):(row[i>>1]&15); return (float)((nib&8)?nib-16:nib); }
/* y[O] = (x[I] . W[O,I]) * s[O] */
static void gemv(float *y,const float *x,const uint8_t *w,const float *s,int I,int O){
    for(int o=0;o<O;o++){ const uint8_t *row=w+(size_t)o*((I+1)/2); double a=0;
        for(int i=0;i<I;i++) a+=x[i]*deq(row,i); y[o]=(float)a*s[o]; }
}
#endif
static void expert_ref(float *y,const float *x,const Exp *e){
    float g[IH],u[IH];
    gemv(g,x,e->g4,e->gs,D,IH); gemv(u,x,e->u4,e->us,D,IH);
    for(int i=0;i<IH;i++){ float v=g[i]; g[i]=(v/(1.f+expf(-v)))*u[i]; }
    gemv(y,g,e->d4,e->ds,IH,D);
}
static void note_all(void){
    for(int l=0;l<NL;l++) for(int e=0;e<NE;e++)
        qt_note_block(l,e,E[l][e].g4,E[l][e].u4,E[l][e].d4,E[l][e].gs,E[l][e].us,E[l][e].ds);
    qt_fill_wait();
}
static int count_resident(int layer){ int n=0; for(int e=0;e<NE;e++) n+=qt_is_resident(layer,e); return n; }

/* One tier init for the whole run: the backend is not designed to be torn down and
 * brought up again inside one process (arenas outlive coli_vk_shutdown). The budget
 * admits exactly two experts:
 *   int4: per-expert bytes = 3*D*IH/2 + (2*IH+D)*4 + 4096 = 7680; VK_EXPERT_GB=0.00002
 *         (21474 bytes) admits two, not three.
 *   int8: per-expert bytes = 3*D*IH   + (2*IH+D)*4 + 4096 = 10752; VK_EXPERT_GB=0.000025
 *         (26843 bytes) admits two, not three.
 * The natural warmstart order fills layer 0, eids 0 and 1, so those two are the
 * resident pair and eid 2 is a guaranteed miss in both modes. */

/* resident experts, served from VRAM, reproduce the CPU reference path */
static int resident_experts_match_cpu_path(void){
    if(!qt_is_resident(0,0)||!qt_is_resident(0,1)) return fail("warmstart should have placed layer 0 eids 0 and 1");
    float x[D]; for(int i=0;i<D;i++) x[i]=frand();
    int eids[TOPK]={0,1}; float val[TOPK]={0.7f,0.3f};
    uint32_t mask=qt_issue(0,eids,TOPK,x);
    if(mask!=3u) return fail("both resident experts should be served by the GPU");
    float out[D]={0}, ref[D]={0}, y[D];
    qt_take(mask,val,TOPK,out);
    for(int k=0;k<TOPK;k++){ expert_ref(y,x,&E[0][eids[k]]); for(int d=0;d<D;d++) ref[d]+=val[k]*y[d]; }
    double maxrel=0;
    for(int d=0;d<D;d++){ double den=fabs(ref[d])>1e-3?fabs(ref[d]):1e-3; double r=fabs(out[d]-ref[d])/den; if(r>maxrel) maxrel=r; }
    printf("resident_experts_match_cpu_path: maxrel %.3e\n",maxrel);
    if(maxrel>2e-3) return fail("GPU expert output diverges from the CPU reference");
    return 0;
}

/* a non-resident expert returns no mask bit so the engine computes it on the CPU */
static int misses_fall_back_to_cpu(void){
    if(qt_is_resident(0,2)) return fail("eid 2 should not fit the two-expert budget");
    float x[D]; for(int i=0;i<D;i++) x[i]=frand();
    int eids[1]={2}; float val[1]={1.f}; float out[D]={0};
    uint32_t mask=qt_issue(0,eids,1,x);
    qt_take(mask,val,1,out);
    if(mask!=0) return fail("non-resident expert must not be claimed by the GPU");
    for(int d=0;d<D;d++) if(out[d]!=0.f) return fail("GPU wrote output for a miss");
    puts("misses_fall_back_to_cpu: ok");
    return 0;
}

/* heat the misses hard, past the CUDA tier's 16-tick swap check and its 25%+4
 * hysteresis; on Vulkan nothing may move. */
static int residency_frozen_after_warmstart(void){
    float x[D]; for(int i=0;i<D;i++) x[i]=frand();
    float val[32]; for(int i=0;i<32;i++) val[i]=1.f;
    float out[D]={0};
    int before[NL][NE]; for(int l=0;l<NL;l++) for(int e=0;e<NE;e++) before[l][e]=qt_is_resident(l,e);
    for(int t=0;t<64;t++){
        for(int l=0;l<NL;l++){
            int hot[NE]; int n=0; for(int e=0;e<NE;e++) if(!before[l][e]) hot[n++]=e;
            for(int k=0;k<n;k++) qt_note(l,hot[k],E[l][hot[k]].g4,E[l][hot[k]].u4,E[l][hot[k]].d4,E[l][hot[k]].gs,E[l][hot[k]].us,E[l][hot[k]].ds);
            uint32_t m=qt_issue(l,hot,n,x); qt_take(m,val,n,out);
            if(m) return fail("a non-resident expert was served from VRAM");
        }
    }
    qt_fill_wait();
    for(int l=0;l<NL;l++) for(int e=0;e<NE;e++)
        if(qt_is_resident(l,e)!=before[l][e]) return fail("residency changed after warmstart (Vulkan tier must fill once)");
    puts("residency_frozen_after_warmstart: ok");
    return 0;
}

int main(void){
    fprintf(stderr,"qwen36 tier vk test: %s experts\n", TIER_VK_INT8 ? "int8" : "int4");
    for(int l=0;l<NL;l++) for(int e=0;e<NE;e++) make_expert(&E[l][e]);
    setenv("COLI_VULKAN","1",1);
#if TIER_VK_INT8
    setenv("VK_EXPERT_GB","0.000025",1);
#else
    setenv("VK_EXPERT_GB","0.00002",1);
#endif
    unsetenv("HEAT_FILE"); unsetenv("QT_NO_WARMSTART");
    /* per-row scales (expert_gs=0); expert_is_int4 picks packed int4 vs raw int8 */
#if TIER_VK_INT8
    if(!qt_init(NL,NE,D,IH,NE,TOPK,0,0)){ puts("skip: no usable Vulkan device"); return 0; }
#else
    if(!qt_init(NL,NE,D,IH,NE,TOPK,0,1)){ puts("skip: no usable Vulkan device"); return 0; }
#endif
    note_all();
    int res=count_resident(0)+count_resident(1);
    if(res!=2) { fprintf(stderr,"resident=%d\n",res); return fail("budget should admit exactly two experts"); }
    int r=resident_experts_match_cpu_path();
    if(!r) r=misses_fall_back_to_cpu();
    if(!r) r=residency_frozen_after_warmstart();
    qt_shutdown();
    return r;
}
#endif
