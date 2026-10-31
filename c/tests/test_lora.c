/* LoRA adapter runtime tests: apply math vs naive dense reference, safetensors
 * round trip, zero-adapter identity, fingerprint mismatch rejection. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../lora.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static unsigned rng_state = 42;
static float frand(void){          /* deterministic LCG in [-1,1) */
    rng_state = rng_state*1664525u + 1013904223u;
    return (float)(int32_t)rng_state / 2147483648.0f;
}

/* y[S,O] += alpha/rank * B(A x) computed the obvious dense way */
static void naive_apply(int S,int O,int I,int rank,float scale,
                        const float *A,const float *B,const float *x,float *y){
    for(int s=0;s<S;s++) for(int o=0;o<O;o++){
        double acc=0;
        for(int r=0;r<rank;r++){
            double z=0; for(int i=0;i<I;i++) z+=(double)A[r*I+i]*x[s*I+i];
            acc+=(double)B[o*rank+r]*z;
        }
        y[(int64_t)s*O+o]+=(float)(scale*acc);
    }
}

int main(void){
    const int S=3, O=16, I=24, rank=4;
    const float alpha=8.f;

    LoraTensor t={.layer=2,.tgt=LORA_T_O,.O=O,.I=I,.rank=rank,.scale=alpha/rank};
    t.A=malloc(rank*I*4); t.B=malloc(O*rank*4);
    float *x=malloc(S*I*4), *y=malloc(S*O*4), *yref=malloc(S*O*4);
    for(int i=0;i<rank*I;i++) t.A[i]=frand();
    for(int i=0;i<O*rank;i++) t.B[i]=frand();
    for(int i=0;i<S*I;i++) x[i]=frand();
    for(int i=0;i<S*O;i++) y[i]=yref[i]=frand();

    /* 1) apply matches the naive dense reference */
    lora_apply(&t,x,y,S);
    naive_apply(S,O,I,rank,t.scale,t.A,t.B,x,yref);
    for(int i=0;i<S*O;i++) CHECK(fabsf(y[i]-yref[i])<1e-4f);

    /* 2) zero B -> bitwise identity */
    float *y0=malloc(S*O*4); memcpy(y0,yref,S*O*4);
    memset(t.B,0,O*rank*4);
    lora_apply(&t,x,yref,S);
    CHECK(memcmp(y0,yref,S*O*4)==0);
    for(int i=0;i<O*rank;i++) t.B[i]=frand();   /* restore nonzero for the round trip */

    /* 3) save -> load round trip preserves tensors and metadata */
    LoraAdapter save={0};
    save.n=1; save.t=&t; save.rank=rank; save.alpha=alpha; save.base_fp=0xabcdef12345678ULL;
    snprintf(save.base_model,sizeof(save.base_model),"tiny-test");
    const char *dir=getenv("TMPDIR")?getenv("TMPDIR"):"/tmp";
    char adir[1024]; snprintf(adir,sizeof(adir),"%s/coli_lora_test_%d",dir,(int)getpid());
    CHECK(lora_save(adir,&save)==0);

    LoraAdapter *ld=lora_load(adir,0xabcdef12345678ULL,0);
    CHECK(ld!=NULL);
    CHECK(ld->n==1 && ld->rank==rank && fabsf(ld->alpha-alpha)<1e-6f);
    CHECK(strcmp(ld->base_model,"tiny-test")==0);
    const LoraTensor *lt=lora_find(ld,2,LORA_T_O);
    CHECK(lt!=NULL && lt->O==O && lt->I==I && lt->rank==rank);
    CHECK(memcmp(lt->A,t.A,rank*I*4)==0);
    CHECK(memcmp(lt->B,t.B,O*rank*4)==0);
    CHECK(lora_find(ld,0,LORA_T_O)==NULL);      /* un-adapted slots stay empty */
    CHECK(lora_find(ld,2,LORA_T_QB)==NULL);
    lora_free(ld);

    /* 4) fingerprint mismatch: rejected strict, accepted with unsafe=1 */
    CHECK(lora_load(adir,0xdeadbeefULL,0)==NULL);
    LoraAdapter *lu=lora_load(adir,0xdeadbeefULL,1);
    CHECK(lu!=NULL); lora_free(lu);

    /* 4b) rank 4.5 must not truncate to 4 and size the I/O for the wrong rank.
     * 4.0 is an integer-valued float and still loads. */
    {
        char jpath[1100]; snprintf(jpath,sizeof(jpath),"%s/adapter.json",adir);
        FILE *jf=fopen(jpath,"rb"); CHECK(jf);
        fseek(jf,0,SEEK_END); long jn=ftell(jf); fseek(jf,0,SEEK_SET);
        char *js=malloc((size_t)jn+1);
        CHECK(js && fread(js,1,(size_t)jn,jf)==(size_t)jn);
        fclose(jf); js[jn]=0;
        char *at=strstr(js,"\"rank\":4,");
        CHECK(at);
        jf=fopen(jpath,"wb"); CHECK(jf);
        CHECK(fwrite(js,1,(size_t)(at-js),jf)==(size_t)(at-js));
        fputs("\"rank\":4.5,",jf);
        fputs(at+9,jf);
        CHECK(fclose(jf)==0);
        CHECK(lora_load(adir,0xabcdef12345678ULL,0)==NULL);
        jf=fopen(jpath,"wb"); CHECK(jf);
        CHECK(fwrite(js,1,(size_t)(at-js),jf)==(size_t)(at-js));
        fputs("\"rank\":4.0,",jf);
        fputs(at+9,jf);
        CHECK(fclose(jf)==0);
        LoraAdapter *lr=lora_load(adir,0xabcdef12345678ULL,0);
        CHECK(lr!=NULL && lr->rank==4);
        lora_free(lr);
        free(js);
    }

    /* 4c) a non-finite alpha becomes scale inf and is applied to every layer. */
    {
        char jpath[1100]; snprintf(jpath,sizeof(jpath),"%s/adapter.json",adir);
        FILE *jf=fopen(jpath,"rb"); CHECK(jf);
        fseek(jf,0,SEEK_END); long jn=ftell(jf); fseek(jf,0,SEEK_SET);
        char *js=malloc((size_t)jn+1);
        CHECK(js && fread(js,1,(size_t)jn,jf)==(size_t)jn);
        fclose(jf); js[jn]=0;
        char *at=strstr(js,"\"alpha\":8,");
        CHECK(at);
        jf=fopen(jpath,"wb"); CHECK(jf);
        CHECK(fwrite(js,1,(size_t)(at-js),jf)==(size_t)(at-js));
        fputs("\"alpha\":1e9999,",jf);
        fputs(at+10,jf);
        CHECK(fclose(jf)==0);
        CHECK(lora_load(adir,0xabcdef12345678ULL,0)==NULL);
        free(js);
    }

    /* 5) a target's layer index sizes and indexes the lookup table: a negative
     * index, or one no model has, is refused instead of written out of bounds */
    const int bad_layers[2]={-1,1<<20};
    for(int k=0;k<2;k++){
        t.layer=bad_layers[k];
        CHECK(lora_save(adir,&save)==0);
        CHECK(lora_load(adir,0xabcdef12345678ULL,0)==NULL);
    }
    t.layer=2;

    /* 6) a digit string that does not fit in int must not wrap onto a real layer.
     * sscanf %d of 4294967296 is 0 here, which is a legal index. */
    {
        int layer=-1, tgt=-1;
        CHECK(lora_parse_target("model.layers.4294967296.self_attn.o_proj",&layer,&tgt)!=0);
        CHECK(lora_parse_target("model.layers.4294967297.self_attn.o_proj",&layer,&tgt)!=0);
        CHECK(lora_parse_target("model.layers.65536.self_attn.o_proj",&layer,&tgt)!=0);
        CHECK(lora_parse_target("model.layers.65535.self_attn.o_proj",&layer,&tgt)==0);
        CHECK(layer==65535 && tgt==LORA_T_O);
        CHECK(lora_parse_target("model.layers.+3.self_attn.o_proj",&layer,&tgt)==0);
        CHECK(layer==3);
        CHECK(lora_parse_target("model.layers.03.self_attn.o_proj",&layer,&tgt)==0);
        CHECK(layer==3);
    }

    /* cleanup */
    char cmd[1200]; snprintf(cmd,sizeof(cmd),"rm -rf %s",adir); system(cmd);

    free(t.A); free(t.B); free(x); free(y); free(yref); free(y0);
    puts("lora adapter tests: ok");
    return 0;
}
