/* LoRA adapter runtime: load/save/apply small trainable adapters on top of the
 * frozen quantized base (y = Q(W)x + scale*B(Ax), scale = alpha/rank).
 *
 * Format "colibri-lora-v1": a directory with
 *   adapter.safetensors  F32 tensors  <target>.lora_A.weight [rank,I]
 *                                     <target>.lora_B.weight [O,rank]
 *   adapter.json         metadata     {format, base_model, base_fingerprint,
 *                                      rank, alpha, dtype, targets[], tensor_name_map_version}
 * <target> uses the checkpoint tensor names Colibri already loads, e.g.
 * "model.layers.3.self_attn.o_proj" -> deterministic mapping onto Layer fields.
 *
 * Adapters are NEVER merged into base weights and never modify base files.
 * A base-fingerprint mismatch is rejected unless the caller passes unsafe=1.
 * Reader/writer reuse st.h + json.h; no new dependencies. */
#ifndef LORA_H
#define LORA_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include "st.h"
#include "json.h"

#define LORA_FORMAT   "colibri-lora-v1"
#define LORA_MAX_RANK 256
#define LORA_MAX_LAYERS 65536

/* v1 target allowlist: resident dense attention projections only (AGENTS.md §4).
 * Routed experts / router / embeddings / lm_head are deliberately absent. */
enum { LORA_T_O = 0, LORA_T_QA, LORA_T_QB, LORA_T_KVA, LORA_T_KVB, LORA_NTGT };
static const char *lora_tgt_suffix[LORA_NTGT] = {
    "self_attn.o_proj", "self_attn.q_a_proj", "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj",
};

typedef struct {
    int layer, tgt;          /* transformer layer index + LORA_T_* */
    int O, I, rank;
    float scale;             /* alpha/rank, folded once at load */
    float *A, *B;            /* A [rank,I], B [O,rank], row-major f32 */
} LoraTensor;

typedef struct {
    int n;                   /* adapted (layer,target) pairs */
    LoraTensor *t;
    int rank; float alpha;
    char base_model[128];
    uint64_t base_fp;
    int *lut, lut_layers;    /* lut[layer*LORA_NTGT+tgt] = idx+1, 0 = none */
} LoraAdapter;

/* FNV-1a over the architecture ints that must match between base and adapter.
 * Callers feed the same fields in the same order (see lora_fp_cfg users). */
static uint64_t lora_fp_mix(uint64_t h, int64_t v){
    for(int i=0;i<8;i++){ h^=(v>>(8*i))&0xff; h*=1099511628211ULL; }
    return h;
}
#define LORA_FP_SEED 1469598103934665603ULL

static const LoraTensor *lora_find(const LoraAdapter *a, int layer, int tgt){
    if(!a || layer<0 || layer>=a->lut_layers) return NULL;
    int idx=a->lut[layer*LORA_NTGT+tgt];
    return idx ? &a->t[idx-1] : NULL;
}

/* y[S,O] += scale * B(A x): two skinny matmuls, rank<=LORA_MAX_RANK.
 * f32 accumulation; zero B (or zero A) leaves y bitwise unchanged. */
static void lora_apply(const LoraTensor *t, const float *x, float *y, int S){
    #pragma omp parallel for schedule(static) if(S>4)
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*t->I; float *ys=y+(int64_t)s*t->O;
        float z[LORA_MAX_RANK];
        for(int r=0;r<t->rank;r++){
            const float *a=t->A+(int64_t)r*t->I;
            float acc=0; for(int i=0;i<t->I;i++) acc+=a[i]*xs[i];
            z[r]=t->scale*acc;
        }
        for(int o=0;o<t->O;o++){
            const float *b=t->B+(int64_t)o*t->rank;
            float acc=0; for(int r=0;r<t->rank;r++) acc+=b[r]*z[r];
            ys[o]+=acc;
        }
    }
}

/* Decimal layer index. sscanf %d is not usable here: a digit string that does
 * not fit in int is undefined, and on this libc 4294967296 wraps to 0 and
 * would load the tensor onto layer 0. Stop at the layer table instead. */
static int lora_parse_layer(const char *s, int *n_out, const char **rest){
    if(*s=='+') s++;                 /* "model.layers.+3" is the same layer as 3 */
    if(*s<'0' || *s>'9') return -1;
    int n=0;
    while(*s>='0' && *s<='9'){
        int d=*s-'0';
        if(n > (LORA_MAX_LAYERS - d) / 10) return -1;
        n=n*10+d;
        s++;
    }
    if(n>=LORA_MAX_LAYERS) return -1;
    *n_out=n; *rest=s;
    return 0;
}

/* parse "model.layers.<n>.<suffix>" -> layer/tgt; -1 if not in the allowlist */
static int lora_parse_target(const char *name, int *layer, int *tgt){
    static const char prefix[]="model.layers.";
    if(strncmp(name,prefix,sizeof(prefix)-1)) return -1;
    int n; const char *rest;
    if(lora_parse_layer(name+(sizeof(prefix)-1),&n,&rest) || *rest!='.') return -1;
    rest++;
    for(int k=0;k<LORA_NTGT;k++)
        if(!strcmp(rest,lora_tgt_suffix[k])){ *layer=n; *tgt=k; return 0; }
    return -1;
}

static void lora_free(LoraAdapter *a){
    if(!a) return;
    for(int i=0;i<a->n;i++){ free(a->t[i].A); free(a->t[i].B); }
    free(a->t); free(a->lut); free(a);
}

/* Load an adapter directory. expect_fp: base fingerprint the engine computed
 * from its config; mismatch (or format/rank problems) -> NULL + stderr reason.
 * unsafe=1 downgrades the fingerprint mismatch to a warning. */
static LoraAdapter *lora_load(const char *dir, uint64_t expect_fp, int unsafe){
    char path[2048]; snprintf(path,sizeof(path),"%s/adapter.json",dir);
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[LORA] %s: no adapter.json\n",dir); return NULL; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    if(sz<=0 || sz>(1<<20)){ fprintf(stderr,"[LORA] adapter.json size %ld out of range\n",sz); fclose(f); return NULL; }
    char *txt=malloc(sz+1);
    if(fread(txt,1,sz,f)!=(size_t)sz){ fprintf(stderr,"[LORA] adapter.json read failed\n"); fclose(f); free(txt); return NULL; }
    fclose(f); txt[sz]=0;
    char *arena=NULL; jval *md=json_parse(txt,&arena);
    free(txt);
    if(!md){ fprintf(stderr,"[LORA] adapter.json parse error\n"); return NULL; }

    jval *jfmt=json_get(md,"format"), *jrank=json_get(md,"rank"), *jalpha=json_get(md,"alpha");
    jval *jfp=json_get(md,"base_fingerprint"), *jbm=json_get(md,"base_model"), *jtg=json_get(md,"targets");
    if(!jfmt||jfmt->t!=J_STR||strcmp(jfmt->str,LORA_FORMAT)){
        fprintf(stderr,"[LORA] unsupported format (want %s)\n",LORA_FORMAT); return NULL; }
    if(!jrank||jrank->t!=J_NUM||!jalpha||jalpha->t!=J_NUM||!jtg||jtg->t!=J_ARR||jtg->len<1){
        fprintf(stderr,"[LORA] adapter.json: missing rank/alpha/targets\n"); return NULL; }
    /* Cast only after the range check: (int)4.5 is 4, which then passes
     * [1, LORA_MAX_RANK] and sizes the A/B reads for the wrong rank. */
    double rd=jrank->num;
    if(!isfinite(rd) || rd<1.0 || rd>(double)LORA_MAX_RANK || rd!=floor(rd)){
        fprintf(stderr,"[LORA] rank must be an integer in [1,%d]\n",LORA_MAX_RANK); return NULL; }
    int rank=(int)rd;
    uint64_t fp=0;
    if(jfp&&jfp->t==J_STR) fp=strtoull(jfp->str,NULL,16);
    if(fp!=expect_fp){
        fprintf(stderr,"[LORA] base fingerprint mismatch: adapter %016llx vs model %016llx\n",
                (unsigned long long)fp,(unsigned long long)expect_fp);
        if(!unsafe){ fprintf(stderr,"[LORA] refusing to load (LORA_UNSAFE=1 to override)\n"); return NULL; }
        fprintf(stderr,"[LORA] LORA_UNSAFE=1: loading anyway\n");
    }

    /* (float) of a non-finite alpha is inf, and scale = alpha/rank then poisons
     * every adapted layer. A finite alpha, including 0, is a real scale. */
    double ad=jalpha->num;
    float alpha=(float)ad;
    if(!isfinite(ad) || !isfinite(alpha)){
        fprintf(stderr,"[LORA] alpha is not finite\n"); return NULL; }
    LoraAdapter *a=calloc(1,sizeof(*a));
    a->rank=rank; a->alpha=alpha; a->base_fp=fp;
    if(jbm&&jbm->t==J_STR) snprintf(a->base_model,sizeof(a->base_model),"%s",jbm->str);
    a->n=jtg->len;
    a->t=calloc(a->n,sizeof(LoraTensor));

    shards S; memset(&S,0,sizeof(S));
    st_init(&S,dir);   /* closed on every way out: on Windows an open shard keeps a later save from replacing it */
    int maxl=0;
    for(int i=0;i<a->n;i++){
        if(jtg->kids[i]->t!=J_STR){ fprintf(stderr,"[LORA] targets[%d] not a string\n",i); st_destroy(&S); lora_free(a); return NULL; }
        const char *tn=jtg->kids[i]->str;
        LoraTensor *t=&a->t[i];
        if(lora_parse_target(tn,&t->layer,&t->tgt)){
            fprintf(stderr,"[LORA] target %s not in v1 allowlist\n",tn); st_destroy(&S); lora_free(a); return NULL; }
        char an[512],bn[512];
        snprintf(an,sizeof(an),"%s.lora_A.weight",tn);
        snprintf(bn,sizeof(bn),"%s.lora_B.weight",tn);
        int64_t na=st_numel(&S,an), nb=st_numel(&S,bn);
        if(na<=0||nb<=0){ fprintf(stderr,"[LORA] missing tensors for %s\n",tn); st_destroy(&S); lora_free(a); return NULL; }
        if(na%rank||nb%rank){ fprintf(stderr,"[LORA] %s: numel not divisible by rank %d\n",tn,rank); st_destroy(&S); lora_free(a); return NULL; }
        t->rank=rank; t->I=(int)(na/rank); t->O=(int)(nb/rank);
        t->scale=a->alpha/(float)rank;
        t->A=malloc((size_t)na*4); t->B=malloc((size_t)nb*4);
        st_read_f32(&S,an,t->A,0);
        st_read_f32(&S,bn,t->B,0);
        if(t->layer+1>maxl) maxl=t->layer+1;
        for(int j=0;j<i;j++) if(a->t[j].layer==t->layer&&a->t[j].tgt==t->tgt){
            fprintf(stderr,"[LORA] duplicate target %s\n",tn); st_destroy(&S); lora_free(a); return NULL; }
    }
    a->lut_layers=maxl;
    a->lut=calloc((size_t)maxl*LORA_NTGT,sizeof(int));
    for(int i=0;i<a->n;i++) a->lut[a->t[i].layer*LORA_NTGT+a->t[i].tgt]=i+1;
    st_destroy(&S);   /* the tensors are copies now */
    return a;
}

/* --- writer: minimal safetensors (F32 only) + adapter.json, atomic rename --- */
static int lora_write_file(const char *path, const void *buf, size_t n){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"wb"); if(!f) return -1;
    if(fwrite(buf,1,n,f)!=n){ fclose(f); remove(tmp); return -1; }
    if(fflush(f)||fclose(f)){ remove(tmp); return -1; }
    return rename(tmp,path);
}

/* mkdir -p: create every missing component (adapter-out paths are often nested
 * like adapters/run1 — a single mkdir fails with ENOENT on the parent).
 * Windows' CRT _mkdir() takes the path only — no mode argument — and accepts
 * either separator, so backslash paths split correctly too. */
#ifdef _WIN32
#include <direct.h>
#define LORA_MKDIR(p) _mkdir(p)
#else
#define LORA_MKDIR(p) mkdir((p),0755)
#endif
static void lora_mkpath(const char *dir){
    char p[2048]; snprintf(p,sizeof(p),"%s",dir);
    for(char *c=p+1;*c;c++) if(*c=='/'||*c=='\\'){ char sep=*c; *c=0; LORA_MKDIR(p); *c=sep; }
    LORA_MKDIR(p);
}

static int lora_save(const char *dir, const LoraAdapter *a){
    lora_mkpath(dir);
    /* safetensors header: json object name->{dtype,shape,data_offsets} */
    size_t hcap=4096+(size_t)a->n*512, hn=0; char *hdr=malloc(hcap);
    int64_t off=0; hn+=snprintf(hdr+hn,hcap-hn,"{");
    for(int i=0;i<a->n;i++){
        const LoraTensor *t=&a->t[i];
        int64_t na=(int64_t)t->rank*t->I*4, nb=(int64_t)t->O*t->rank*4;
        hn+=snprintf(hdr+hn,hcap-hn,
            "%s\"model.layers.%d.%s.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]},",
            i?",":"",t->layer,lora_tgt_suffix[t->tgt],t->rank,t->I,(long long)off,(long long)(off+na));
        off+=na;
        hn+=snprintf(hdr+hn,hcap-hn,
            "\"model.layers.%d.%s.lora_B.weight\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%lld,%lld]}",
            t->layer,lora_tgt_suffix[t->tgt],t->O,t->rank,(long long)off,(long long)(off+nb));
        off+=nb;
    }
    hn+=snprintf(hdr+hn,hcap-hn,"}");
    size_t total=8+hn+(size_t)off;
    unsigned char *buf=malloc(total);
    uint64_t hlen=hn; memcpy(buf,&hlen,8); memcpy(buf+8,hdr,hn);
    unsigned char *p=buf+8+hn;
    for(int i=0;i<a->n;i++){
        const LoraTensor *t=&a->t[i];
        size_t na=(size_t)t->rank*t->I*4, nb=(size_t)t->O*t->rank*4;
        memcpy(p,t->A,na); p+=na; memcpy(p,t->B,nb); p+=nb;
    }
    char path[2048]; snprintf(path,sizeof(path),"%s/adapter.safetensors",dir);
    int rc=lora_write_file(path,buf,total);
    free(buf); free(hdr);
    if(rc){ fprintf(stderr,"[LORA] write %s failed\n",path); return -1; }

    size_t mcap=1024+(size_t)a->n*300, mn=0; char *mj=malloc(mcap);
    mn+=snprintf(mj+mn,mcap-mn,
        "{\"format\":\"%s\",\"base_model\":\"%s\",\"base_fingerprint\":\"%016llx\","
        "\"rank\":%d,\"alpha\":%g,\"dtype\":\"F32\",\"tensor_name_map_version\":1,\"targets\":[",
        LORA_FORMAT,a->base_model,(unsigned long long)a->base_fp,a->rank,(double)a->alpha);
    for(int i=0;i<a->n;i++)
        mn+=snprintf(mj+mn,mcap-mn,"%s\"model.layers.%d.%s\"",i?",":"",a->t[i].layer,lora_tgt_suffix[a->t[i].tgt]);
    mn+=snprintf(mj+mn,mcap-mn,"]}");
    snprintf(path,sizeof(path),"%s/adapter.json",dir);
    rc=lora_write_file(path,mj,mn);
    free(mj);
    if(rc){ fprintf(stderr,"[LORA] write %s failed\n",path); return -1; }
    return 0;
}

#endif /* LORA_H */
