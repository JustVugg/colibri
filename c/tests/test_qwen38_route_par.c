/* Q38_ROUTE_PAR's one-pass top-k (q38_topk_onepass) against the K-pass pick every MoE path
 * used before it, through q38_route_probs: on random router rows (ties, NaN terms, every
 * term NaN, fewer finite terms than K, K up to the engine's maximum) both give the same
 * ranks, softmax values and gates, to the bit, with norm_topk on and off. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"

static unsigned g_s=20261121u;
static unsigned rnd(void){ g_s=g_s*1664525u+1013904223u; return g_s>>8; }

int main(void){
    enum { E_MAX=600, T=50000 };
    long bad=0,tied=0,nan_rows=0,fell=0;
    float a[E_MAX],b[E_MAX];
    for(int t=0;t<T;t++){
        const int E=1+(int)(rnd()%(t%100==0?E_MAX:64)),K=1+(int)(rnd()%Q38_MAX_TOPK),kind=(int)(rnd()%5);
        const int norm=(int)(rnd()&1);
        for(int e=0;e<E;e++){
            unsigned r=rnd();
            a[e]=kind==0?(float)(r%4):                       /* four levels: many ties */
                 kind==1&&r%5==0?NAN:                         /* some NaN logits */
                 kind==2?NAN:                                 /* every logit NaN */
                 kind==3&&r%3==0?-INFINITY:                   /* softmax zeros: fewer candidates */
                 (float)(r%100000)/10000.f-5.f;
            b[e]=a[e];
        }
        int ia[Q38_MAX_TOPK],ib[Q38_MAX_TOPK];float ga[Q38_MAX_TOPK],gb[Q38_MAX_TOPK];
        q38_route_probs(a,E,K,0,norm,0,ia,ga);
        q38_route_probs(b,E,K,0,norm,1,ib,gb);
        bad+=memcmp(ia,ib,(size_t)K*sizeof(int))||memcmp(ga,gb,(size_t)K*sizeof(float))||memcmp(a,b,(size_t)E*sizeof(float));
        tied+=kind==0;nan_rows+=kind==1||kind==2;
        for(int z=0;z<K;z++)if(!(a[ia[z]]>-1.f)||a[ia[z]]==0.f){fell++;break;}
    }
    printf("test_qwen38_route_par: %d rows (%ld with ties, %ld with NaN, %ld with fallback or zero ranks), %ld differ\n",
           T,tied,nan_rows,fell,bad);
    if(bad||!tied||!nan_rows||!fell){printf("test_qwen38_route_par: FAIL\n");return 1;}
    printf("test_qwen38_route_par: ok\n");
    return 0;
}
