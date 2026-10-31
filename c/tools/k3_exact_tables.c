/* k3_exact_tables.c -- k3_exact.h's tables and its proof on every float.
 *
 * For each of the 2^32 floats: k3x_expf_fast against exp_f (exp_f.h: expf's bits) and
 * k3x_tanhf_fast against the C library's tanhf. Where the fast part gives an answer it has to
 * be the reference's bits (a single other answer fails the run); where it gives none, the
 * float and the reference's bits go in the table. Writes k3_exact_tables.h (the path given,
 * default ./k3_exact_tables.h); with --check, compares the tables with that file instead.
 *
 * Built by the Makefile's k3-exact-tables rule (MinGW, -ffp-contract=off): exp_f is the x87
 * sequence only there, and the tables hold MinGW's tanhf. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../exp_f.h"
#include "../k3_exact.h"

typedef struct { uint32_t in, out; } Pair;
typedef struct { Pair *p; size_t n, cap; } List;
static void add(List *l, uint32_t in, uint32_t out){
    if(l->n==l->cap){ l->cap=l->cap?l->cap*2:1024; l->p=realloc(l->p,l->cap*sizeof(Pair)); if(!l->p){ fprintf(stderr,"OOM\n"); exit(1); } }
    l->p[l->n].in=in; l->p[l->n].out=out; l->n++;
}
static int by_in(const void *a, const void *b){
    uint32_t x=((const Pair*)a)->in, y=((const Pair*)b)->in; return x<y?-1:x>y;
}
static void merge(List *dst, List *src){ for(size_t i=0;i<src->n;i++) add(dst,src->p[i].in,src->p[i].out); free(src->p); }

static void emit(FILE *f, const char *name, const List *l){
    fprintf(f,"#define K3X_%s_N %zu\n",name,l->n);
    fprintf(f,"static const uint32_t k3x_%s_in[K3X_%s_N] = {",name,name);
    for(size_t i=0;i<l->n;i++) fprintf(f,"%s0x%08x",i==0?"\n    ":i%8?",":",\n    ",l->p[i].in);
    fprintf(f,"\n};\nstatic const uint32_t k3x_%s_out[K3X_%s_N] = {",name,name);
    for(size_t i=0;i<l->n;i++) fprintf(f,"%s0x%08x",i==0?"\n    ":i%8?",":",\n    ",l->p[i].out);
    fprintf(f,"\n};\n");
}

int main(int argc, char **argv){
    const char *path="k3_exact_tables.h"; int check=0;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--check")) check=1; else path=argv[i]; }
#if !EXP_F_X87
    fprintf(stderr,"k3_exact_tables: exp_f is not MinGW's x87 sequence in this build: the tables hold MinGW's bits\n");
    return 2;
#endif
    int nt=1;
#ifdef _OPENMP
    nt=omp_get_max_threads();
#endif
    List *ex=calloc((size_t)nt,sizeof(List)), *th=calloc((size_t)nt,sizeof(List));
    unsigned long long bad_e=0, bad_t=0, sure_e=0, sure_t=0;
    uint32_t first_e=0, first_t=0;
    #pragma omp parallel for schedule(dynamic,1) reduction(+:bad_e,bad_t,sure_e,sure_t)
    for(int64_t blk=0;blk<(1<<16);blk++){
        int me=0;
#ifdef _OPENMP
        me=omp_get_thread_num();
#endif
        for(uint32_t lo=0;lo<(1u<<16);lo++){
            uint32_t b=(uint32_t)blk<<16|lo;
            float x=k3x_bitsf(b), f;
            float re=exp_f(x);
            if(k3x_expf_fast(x,&f)){
                sure_e++;
                if(k3x_fbits(f)!=k3x_fbits(re)){
                    #pragma omp critical
                    { if(!first_e) first_e=b; }
                    bad_e++;
                }
            } else add(&ex[me],b,k3x_fbits(re));
            float rt=tanhf(x);
            if(k3x_tanhf_fast(x,&f)){
                sure_t++;
                if(k3x_fbits(f)!=k3x_fbits(rt)){
                    #pragma omp critical
                    { if(!first_t) first_t=b; }
                    bad_t++;
                }
            } else add(&th[me],b,k3x_fbits(rt));
        }
    }
    List e={0}, t={0};
    for(int i=0;i<nt;i++){ merge(&e,&ex[i]); merge(&t,&th[i]); }
    qsort(e.p,e.n,sizeof(Pair),by_in); qsort(t.p,t.n,sizeof(Pair),by_in);
    printf("exp:  %llu floats sure, %llu differ from exp_f, %zu in the table\n",sure_e,bad_e,e.n);
    printf("tanh: %llu floats sure, %llu differ from tanhf, %zu in the table\n",sure_t,bad_t,t.n);
    if(bad_e){ float x=k3x_bitsf(first_e), f; k3x_expf_fast(x,&f);
        printf("  an exp miss: x=0x%08x (%a): fast %a, exp_f %a\n",first_e,x,f,exp_f(x)); }
    if(bad_t){ float x=k3x_bitsf(first_t), f; k3x_tanhf_fast(x,&f);
        printf("  a tanh miss: x=0x%08x (%a): fast %a, tanhf %a\n",first_t,x,f,tanhf(x)); }
    if(bad_e||bad_t) return 1;
    char tmp[1100]; snprintf(tmp,sizeof tmp,"%s.tmp",path);
    FILE *m=fopen(tmp,"wb");
    if(!m){ fprintf(stderr,"%s: cannot write\n",tmp); return 1; }
    fprintf(m,"/* k3_exact_tables.h -- written by tools/k3_exact_tables.c (make k3-exact-tables): the floats\n"
              " * whose exp_f or MinGW tanhf k3_exact.h's double arithmetic leaves undecided, with the\n"
              " * reference's bits; every other float's were checked equal. Do not edit. */\n"
              "#ifndef K3_EXACT_TABLES_H\n#define K3_EXACT_TABLES_H\n#include <stdint.h>\n");
    emit(m,"EXP",&e); emit(m,"TANH",&t);
    fprintf(m,"#endif\n");
    if(fclose(m)){ fprintf(stderr,"%s: cannot write\n",tmp); return 1; }
    if(check){
        int same=0;
        FILE *a=fopen(path,"rb"), *b=fopen(tmp,"rb");
        if(a&&b){
            same=1;
            for(;;){   /* a checkout's CRLF lines are the same tables */
                int x, y;
                do x=fgetc(a); while(x=='\r');
                do y=fgetc(b); while(y=='\r');
                if(x!=y){ same=0; break; }
                if(x==EOF) break;
            }
        }
        if(a) fclose(a);
        if(b) fclose(b);
        remove(tmp);
        printf("%s: %s\n",path,same?"the same tables":"DIFFERENT tables");
        return same?0:1;
    }
    remove(path);
    if(rename(tmp,path)){ fprintf(stderr,"%s: cannot write\n",path); return 1; }
    printf("wrote %s\n",path);
    return 0;
}
