/* Owned tensors must be released before backend shutdown, including standalone
 * dense projections. A pending expert group must finish before its weights go. */
#include <math.h>
#include <stdio.h>
#include "../compat.h"
#define coli_cuda_tensor_free fixture_free
#define coli_cuda_shutdown fixture_shutdown
#define coli_cuda_expert_group_take fixture_take
#include "qwen36_fake_cuda.h"
#undef coli_cuda_tensor_free
#undef coli_cuda_shutdown
#undef coli_cuda_expert_group_take
static int freed, stops, pending, failures;
static ColiCudaTensor *borrowed[3];
#define CHECK(c) do { if(!(c)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c);failures++;} } while(0)
void coli_cuda_tensor_free(ColiCudaTensor *t) {
    if(!t) return;
    for(int i=0;i<3;i++) CHECK(!pending || t!=borrowed[i]);
    freed++; fixture_free(t);
}
void coli_cuda_shutdown(void) {
    CHECK(!pending); CHECK(freed==fake_uploads); stops++;
}
const float *coli_cuda_expert_group_take(int device) { pending=0; return NULL; }
static int issue(int device,int count,const float *x) { pending=1; return 1; }
#include "../qwen36_tier.c"
int main(void) {
    enum { D=64, I=32 };
    int8_t q[D*D]; float sc[D], x[D], y[D];
    memset(q,1,sizeof q);
    for(int i=0;i<D;i++){sc[i]=1.f;x[i]=1.f;}
    fake_dense_compute=1;
    for(int round=1;round<=2;round++) {
        memset(q,round,sizeof q);
        CHECK(qt_dnproj_init(0,q,sc,D,D,0));
        CHECK(qt_dense_init(q,sc,D,D,0)>=0);
        CHECK(qt_dnproj_matmul(0,y,x,D,D));
        CHECK(y[0]==D*round);
        qt_shutdown();
        CHECK(freed==fake_uploads); CHECK(!G_dnp[0].t && !G_dnp[0].on);
        CHECK(!qt_dnproj_matmul(0,y,x,D,D)); CHECK(qt_dense_count()==0);
        int before=freed; qt_shutdown(); CHECK(freed==before); CHECK(stops==0);
    }
    setenv("COLI_CUDA","1",1); setenv("COLI_GPUS","0",1);
    setenv("COLI_PLACE","experts=0,lmhead=0,dnproj=0",1);
    setenv("QT_NO_WARMSTART","1",1); setenv("HEAT_FILE","",1);
    setenv("CUDA_EXPERT_GB","0.0625",1);
    for(int round=0;round<2;round++) {
        CHECK(qt_init(1,2,D,I,2,1,0,1));
        CHECK(qt_dnproj_init(0,q,sc,D,D,0));
        CHECK(qt_lmhead_init(q,sc,D,D));
        CHECK(qt_dense_init(q,sc,D,D,0)>=0);
        uint8_t g[D*I/2]={0}, u[D*I/2]={0}, d[D*I/2]={0};
        qt_note(0,0,g,u,d,sc,sc,sc); qt_fill_wait();
        CHECK(qt_is_resident(0,0));
        borrowed[0]=qs(0,0)->tg; borrowed[1]=qs(0,0)->tu; borrowed[2]=qs(0,0)->td;
        fake_issue_hook=issue;
        int eid=0; CHECK(qt_issue(0,&eid,1,x)==1);
        qt_shutdown();
        CHECK(!pending); CHECK(freed==fake_uploads); CHECK(!G.on);
        CHECK(!G.slot && !G.is_x && !G.fill_order && !G.heat0);
        CHECK(!G_lmh.t && !G_lmh.on && !G_lmh.dev_ok);
        CHECK(!G_dnp[0].t && !G_dnp[0].on);
        int before=freed; qt_shutdown(); CHECK(freed==before);
    }
    printf("tier release: %s (%d uploads, %d frees)\n",failures?"FAIL":"ok",fake_uploads,freed);
    return failures!=0;
}
