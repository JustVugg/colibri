/* Microbenchmark: old duplicate-prefix router scan vs marked-score selection.
 *
 * This measures only top-K expert selection. The input is prepared before each
 * timed call, and both arms use the same finite score arrays and fallback path.
 * The full run uses 11 seeds and 2,000 repetitions per cell. Each cell also
 * checks that both selectors return the same ordered expert ids before timing.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdint.h>

#define N_REPEAT 2000
#define N_SEEDS 11

static int select_old(const float *choice, int E, int K, int *idx, int layer){
    for(int kk=0;kk<K;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){
            int taken=0;
            for(int j=0;j<kk;j++) if(idx[j]==e){ taken=1; break; }
            if(!taken && choice[e]>bv){ bv=choice[e]; best=e; }
        }
        idx[kk]=router_best_or_fallback(best,kk,E,layer);
    }
    return idx[0];
}

static uint32_t rng_state;
static void seed_rng(uint32_t seed){ rng_state=seed; }
static float next_score(void){
    rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5;
    return (float)(rng_state>>8)*(1.f/16777216.f);
}

static void fill_scores(float *choice, int E, int shape){
    for(int e=0;e<E;e++) choice[e]=next_score();
    if(shape==1){
        for(int e=0;e<E;e++) choice[e]*=0.02f;
        choice[0]=1.f;
        if(E>4) choice[E/4]=0.8f;
    } else if(shape==2){
        for(int e=0;e<E;e++) choice[e]=(float)(e/4)*0.01f;
    }
}

static void sort_doubles(double *v, int n){
    for(int i=1;i<n;i++){
        double x=v[i]; int j=i-1;
        while(j>=0 && v[j]>x){ v[j+1]=v[j]; j--; }
        v[j+1]=x;
    }
}

static double bench_old(const float *choice, int E, int K, int *idx){
    static double samples[N_REPEAT];
    volatile int sink=0;
    for(int r=0;r<N_REPEAT;r++){
        double t0=now_s();
        sink^=select_old(choice,E,K,idx,0);
        samples[r]=(now_s()-t0)*1e9;
    }
    if(sink==INT32_MIN) fprintf(stderr,"unexpected benchmark sink\n");
    sort_doubles(samples,N_REPEAT);
    return samples[N_REPEAT/2];
}

static double bench_new(const float *choice, int E, int K, int *idx, float *work){
    static double samples[N_REPEAT];
    volatile int sink=0;
    for(int r=0;r<N_REPEAT;r++){
        memcpy(work,choice,(size_t)E*sizeof(float));
        double t0=now_s();
        router_select_topk(work,E,K,idx,0);
        sink^=idx[0];
        samples[r]=(now_s()-t0)*1e9;
    }
    if(sink==INT32_MIN) fprintf(stderr,"unexpected benchmark sink\n");
    sort_doubles(samples,N_REPEAT);
    return samples[N_REPEAT/2];
}

static int verify_selection(const float *choice, int E, int K, int layer){
    int old_idx[32], new_idx[32]; float work[256];
    select_old(choice,E,K,old_idx,layer);
    memcpy(work,choice,(size_t)E*sizeof(float));
    router_select_topk(work,E,K,new_idx,layer);
    for(int kk=0;kk<K;kk++) if(old_idx[kk]!=new_idx[kk]){
        fprintf(stderr,"selection mismatch: E=%d K=%d slot=%d old=%d new=%d\n",
                E,K,kk,old_idx[kk],new_idx[kk]);
        return 0;
    }
    return 1;
}

int main(void){
    const int cases[][2]={{64,8},{256,8},{256,32}};
    const char *shapes[]={"random","peaked","plateau"};
    printf("bench_router_select: old duplicate scan vs marked-score scan (median of %d seeds x %d reps)\n",
           N_SEEDS,N_REPEAT);
    printf("%-9s %4s %3s %14s %14s %9s\n","shape","E","K","old ns/call","new ns/call","speedup");
    printf("---------------------------------------------------------------------\n");
    float choice[256], work[256]; int idx[32];
    double old_seed[N_SEEDS], new_seed[N_SEEDS];
    for(int shape=0;shape<3;shape++) for(size_t ci=0;ci<sizeof(cases)/sizeof(cases[0]);ci++){
        int E=cases[ci][0], K=cases[ci][1];
        for(int seed=0;seed<N_SEEDS;seed++){
            seed_rng(0xA5A5A5A5u+(uint32_t)(seed*0x9E3779B9u));
            fill_scores(choice,E,shape);
            if(!verify_selection(choice,E,K,0)) return 2;
            old_seed[seed]=bench_old(choice,E,K,idx);
            new_seed[seed]=bench_new(choice,E,K,idx,work);
        }
        sort_doubles(old_seed,N_SEEDS); sort_doubles(new_seed,N_SEEDS);
        double old=old_seed[N_SEEDS/2], now=new_seed[N_SEEDS/2];
        printf("%-9s %4d %3d %14.0f %14.0f %8.2fx\n",shapes[shape],E,K,old,now,old/now);
    }
    puts("bench_router_select: done (lower ns is better; speedup = old/new)");
    return 0;
}
