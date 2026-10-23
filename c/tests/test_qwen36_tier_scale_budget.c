/* Separate scale allocations can cross different allocator size classes. */
#include <stdio.h>
#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"
int main(void) {
    /* D, Ih, group size (0 = int8), bytes/expert, allowance, admitted experts.
     * Constants include three weight and three separate scale allocations. */
    const struct { int D,Ih,gs; size_t bytes,budget; int planned; } cases[]={
        {64,2048,0,458752,917503,1},
        {2048,64,0,450560,901119,1},
        {64,3000,0,655360,1310720,2},
        {64,64,0,49152,98304,2},
        {96,2048,64,385024,770047,1},
    };
    setenv("COLI_CUDA","1",1); setenv("COLI_GPUS","0",1);
    setenv("QT_NO_WARMSTART","1",1); setenv("HEAT_FILE","",1);
    for(size_t i=0;i<sizeof cases/sizeof *cases;i++) {
        char gb[64];snprintf(gb,sizeof gb,"%.17g",cases[i].budget/1073741824.0);
        setenv("CUDA_EXPERT_GB",gb,1);
        if(!qt_init(1,3,cases[i].D,cases[i].Ih,3,1,cases[i].gs,cases[i].gs!=0)) return 1;
        int layers[3],eids[3],n=qt_plan_fill(layers,eids,3);
        int ok=G.exp_bytes==cases[i].bytes && n==cases[i].planned &&
               G.used[0]==cases[i].bytes*n && G.used[0]<=G.budget[0];
        if(!ok) fprintf(stderr,"case %zu: bytes=%zu expected=%zu planned=%d expected=%d\n",
                        i,G.exp_bytes,cases[i].bytes,n,cases[i].planned);
        qt_shutdown();
        /* Accommodate dev's retained host storage until teardown cleanup lands. */
        free(G.slot);G.slot=NULL;free(G.is_x);G.is_x=NULL;
        free(G.fill_order);G.fill_order=NULL;free(G.heat0);G.heat0=NULL;
        if(!ok) return 1;
    }
    puts("tier scale budget: ok");return 0;
}
