/* cu_tier.h -- the routed-expert tier on a CUDA device: vk_tier.h's calls, prefixed cut_.
 *
 * cu_tier.c compiles vk_tier.c a second time over the CUDA backend (cu_xb.cu, through
 * cu_tier_ops.h): the same tier, its lines tagged "[CUDA] tier". An engine integrates it as
 * vk_tier.h says, with cut_ for vkt_ and its CUDA device open (coli_cuda_init) in place of
 * the Vulkan one. The declarations below are vk_tier.h's (cu_tier.c checks them against the
 * definitions); without COLI_CUDA they are stubs and the tier is off. The tier reads the
 * same COLI_VK_TIER* settings. Not on CUDA yet: the dense chain's expert table, a second
 * device, streaming and the low-bit formats (fmt 200-203). */
#ifndef COLI_CU_TIER_H
#define COLI_CU_TIER_H
#include "vk_tier.h"   /* the types: VktConfig, VktExpertSrc, VktHostExpert */

#ifdef COLI_CUDA
int  cut_wanted(void);
int  cut_init(const VktConfig *cfg, uint32_t *const *heat);
int  cut_ready(void);
void cut_shutdown(void);
int  cut_plan(int *layers, int *eids, int max);
void cut_warm_report(const char *engine, int n, double seconds);
int  cut_put(int layer, int eid, const VktExpertSrc *src);
void cut_put_done(void);
void cut_note(int layer, int eid, const VktExpertSrc *src);
int  cut_issue(int layer, const float *x, int S, int K, const int *idx, uint8_t *taken);
int  cut_issue_w(int layer, const float *x, int S, int K, const int *idx, const float *w, uint8_t *taken);
int  cut_wants(int layer, int eid);
int  cut_join(const float **rows);
int cut_join_sum(const float **rows, const float *w, const float **sum);
int  cut_resident(int layer, int eid);
void cut_begin_forward(void);
void cut_report(const char *scope, unsigned long long ram_hits, unsigned long long disk_loads);
size_t cut_expert_bytes(int hidden, int inter, VktFmt gate_up, VktFmt down);
int  cut_step_rows(int S, int block);
int  cut_stream_prefetch(int layer, int S);
int  cut_devices(void);
int  cut_layers(void);
int  cut_ram_first(int layer, int eid);
void cut_ram_gave(void);
unsigned cut_epoch(void);
int  cut_dev_table(int layer, uint32_t *entries);
void cut_dev_count(int layer, int S, int K, const int *idx);
void cut_settle(void);
int  cut_dev_changes(unsigned since, int *slots, int max);
int  cut_dev_table_hybrid(int layer, uint32_t *entries, VktHostExpert *hx);
int  cut_dev_entry(int layer, int eid, uint32_t *ent, uint32_t *ent_hybrid, VktHostExpert *hx, uint8_t *host);
int  cut_host_pairs(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                    float limit, int nt, float *gu, float *hh, float *out);
int  cut_host_pairs_w(const VktHostExpert *const *hx, const float *const *xr, int np, int D, int I,
                      float limit, int nt, float *gu, float *hh, float *out, const float *w);
int  cut_host_pool_on(void);
size_t cut_stage_room(void);
int  cut_dev_host(int layer, uint8_t *host);
int  cut_score(int layer, uint64_t *score);
#else
static inline int cut_host_pool_on(void){return 0;}
static inline size_t cut_stage_room(void){return 0;}
static inline int cut_dev_host(int l,uint8_t *h){(void)l;(void)h;return -1;}
static inline int cut_score(int l,uint64_t *s){(void)l;(void)s;return -1;}
static inline int cut_dev_table_hybrid(int l,uint32_t *e,VktHostExpert *h){(void)l;(void)e;(void)h;return -1;}
static inline int cut_dev_entry(int l,int e,uint32_t *t,uint32_t *th,VktHostExpert *h,uint8_t *o){(void)l;(void)e;(void)t;(void)th;(void)h;(void)o;return -1;}
static inline int cut_host_pairs(const VktHostExpert *const *h,const float *const *x,int n,int D,int I,float l,int t,float *g,float *u,float *o){(void)h;(void)x;(void)n;(void)D;(void)I;(void)l;(void)t;(void)g;(void)u;(void)o;return 0;}
static inline int cut_dev_changes(unsigned s,int *c,int m){(void)s;(void)c;(void)m;return -1;}
static inline int  cut_wanted(void){return 0;}
static inline int  cut_init(const VktConfig *c, uint32_t *const *h){(void)c;(void)h;return 0;}
static inline int  cut_ready(void){return 0;}
static inline void cut_shutdown(void){}
static inline int  cut_plan(int *l,int *e,int m){(void)l;(void)e;(void)m;return 0;}
static inline void cut_warm_report(const char *e,int n,double s){(void)e;(void)n;(void)s;}
static inline int  cut_put(int l,int e,const VktExpertSrc *s){(void)l;(void)e;(void)s;return 0;}
static inline void cut_put_done(void){}
static inline void cut_note(int l,int e,const VktExpertSrc *s){(void)l;(void)e;(void)s;}
static inline int  cut_issue(int l,const float *x,int S,int K,const int *i,uint8_t *t){(void)l;(void)x;(void)S;(void)K;(void)i;(void)t;return 0;}
static inline int  cut_issue_w(int l,const float *x,int S,int K,const int *i,const float *w,uint8_t *t){(void)l;(void)x;(void)S;(void)K;(void)i;(void)w;(void)t;return 0;}
static inline int  cut_wants(int l,int e){(void)l;(void)e;return 0;}
static inline int  cut_join(const float **r){(void)r;return 0;}
static inline int  cut_join_sum(const float **r, const float *w, const float **s){(void)r;(void)w;*s=0;return 0;}
static inline int  cut_resident(int l,int e){(void)l;(void)e;return 0;}
static inline void cut_begin_forward(void){}
static inline void cut_report(const char *s,unsigned long long r,unsigned long long d){(void)s;(void)r;(void)d;}
static inline size_t cut_expert_bytes(int h,int i,VktFmt a,VktFmt b){(void)h;(void)i;(void)a;(void)b;return 0;}
static inline int  cut_step_rows(int S,int b){return S<b?S:b;}
static inline int  cut_stream_prefetch(int l,int S){(void)l;(void)S;return 0;}
static inline int  cut_devices(void){return 0;}
static inline int  cut_layers(void){return 0;}
static inline int  cut_ram_first(int l,int e){(void)l;(void)e;return 0;}
static inline void cut_ram_gave(void){}
static inline unsigned cut_epoch(void){return 0;}
static inline int  cut_dev_table(int l,uint32_t *e){(void)l;(void)e;return -1;}
static inline void cut_dev_count(int l,int S,int K,const int *i){(void)l;(void)S;(void)K;(void)i;}
static inline void cut_settle(void){}
#endif
#endif
