/* layer_cuda_shard_kvb() (colibri.c, COLI_CUDA_ATTN_SHARD) -- the multi-device kv_b
 * head-shard uploader. Its rb/weights/scale arithmetic is written for exactly fmt=1/2/3/4
 * (per-row byte strides, per-row or per-group scales); before the guard this file proves,
 * an fmt=8 (fp8-e4m3-b128) kv_b was ADMITTED and only failed safe BY ACCIDENT: fmt=8
 * keeps its raw e4m3 bytes in q8 (q4 stays NULL, per the QT struct comment), so the
 * function selected a NULL weight pointer with an int2 row stride and
 * coli_cuda_tensor_upload_g's !weights check happened to reject the upload before
 * anything dereferenced it -- silent, unnamed, and one refactor away from a misread
 * (fmt=8's per-128x128-BLOCK scale array would ALSO have been sliced with per-row
 * geometry). This probe pins the explicit refusal that replaced the accident:
 *   (1) an fmt=8 kv_b shard attempt refuses BY NAME on stderr, BEFORE any pointer/stride
 *       use, and the message says what serves fmt=8 instead (the absorb path on the
 *       layer home device);
 *   (2) no shard state is minted (n_kv_b_shard==0, kv_b_shard[] untouched,
 *       kv_b.cuda_eligible unchanged);
 *   (3) the notice is bounded (once per process per fmt, never per layer);
 *   (4) other un-shardable fmts (fmt=6 here) refuse by name too, with their own message;
 *   (5) an allowlisted fmt (fmt=1) is NOT refused -- with no device context initialized
 *       its upload fails silently and no shard is minted, but no refusal line appears,
 *       so the guard is format-targeted, not a blanket gate.
 * PROOF-OF-BITE: built against the pre-guard colibri.c, (1)/(3)/(4) fail -- the fmt=8
 * call slid past the format check into the accidental-safe upload rejection with no
 * message at all. Needs -DCOLI_CUDA (CUDA=1) to compile the function; a CPU-only build
 * SKIPs loudly instead of pretending to cover it. No GPU work is performed: every path
 * exercised here returns before any device context exists, so this runs on a CUDA build
 * host even without a card.
 * Portable stderr capture: freopen/dup2, same seam as tests/test_kvb_notice.c. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef COLI_CUDA
int main(void){
    printf("test_shard_kvb_refuse: SKIP (built without COLI_CUDA -- build with CUDA=1 on a CUDA host to exercise layer_cuda_shard_kvb)\n");
    return 0;
}
#else

#include <unistd.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static int redirect_stderr(const char *path){
    fflush(stderr);
    int saved = dup(fileno(stderr));
    if(saved<0){ printf("FAIL: dup(stderr) failed -- capture seam unusable, aborting\n"); exit(1); }
    if(!freopen(path, "w+", stderr)){
        printf("FAIL: freopen(%s) failed -- capture seam unusable, aborting\n", path);
        exit(1);
    }
    return saved;
}
static void restore_stderr(int saved, char *buf, size_t bufsz){
    fflush(stderr);
    long n = ftell(stderr);
    if(n<0) n=0;
    rewind(stderr);
    size_t want = (size_t)n < bufsz-1 ? (size_t)n : bufsz-1;
    size_t got = fread(buf, 1, want, stderr);
    buf[got] = 0;
    fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
}

static int count_sub(const char *hay, const char *needle){
    int n=0; const char *p=hay;
    while((p=strstr(p,needle))){ n++; p++; }
    return n;
}

/* kv_b-shaped fixture at H=4, Q=8, V=8 -> O=H*(Q+V)=64; I=200 (nblkI=2, partial column
 * tail -- the exact scale geometry the shard's per-row slicing would have misread). */
enum { H=4, Q=8, V=8, O=H*(Q+V), I=200 };

static void mk_layer_fmt8(Layer *l, int8_t *q8, float *s){
    memset(l,0,sizeof *l);
    for(int64_t i=0;i<(int64_t)O*I;i++) q8[i]=(int8_t)(i*37+11);
    int64_t nblk=fp8_nblk(O)*fp8_nblk(I);
    for(int64_t i=0;i<nblk;i++) s[i]=0.01f+0.001f*(float)i;
    l->kv_b.fmt=8; l->kv_b.O=O; l->kv_b.I=I; l->kv_b.gs=0;
    l->kv_b.q8=q8; l->kv_b.s=s;          /* q4 stays NULL: the fmt=8 convention */
    l->kv_b.cuda_eligible=1; l->kv_b.cuda_device=0;
}

static void run_shard(Layer *l, char *buf, size_t bufsz){
    int saved = redirect_stderr("tests/tmp_shard_kvb_refuse.stderr");
    layer_cuda_shard_kvb(l,H,Q,V);
    restore_stderr(saved, buf, bufsz);
    remove("tests/tmp_shard_kvb_refuse.stderr");
}

int main(void){
    /* pretend a 2-device dense-CUDA setup so the function's early gate passes; no
     * device context is ever initialized, and none is needed (see file header). */
    g_cuda_enabled=1; g_cuda_dense=1; g_cuda_ndev=2;
    g_cuda_devices[0]=0; g_cuda_devices[1]=0;

    static int8_t q8[(int64_t)O*I];
    static float s[((O+127)/128)*((I+127)/128)];
    static float srow[O];   /* per-row scales for the allowlisted fmt=1 case */
    char err[4096];
    Layer l;

    /* (1)+(2): fmt=8 refuses by name, names the absorb path, mints no shard state */
    mk_layer_fmt8(&l,q8,s);
    run_shard(&l,err,sizeof err);
    CHECK(strstr(err,"layer_cuda_shard_kvb")!=NULL);
    CHECK(strstr(err,"refus")!=NULL);
    CHECK(strstr(err,"fmt=8")!=NULL);
    CHECK(strstr(err,"absorb path")!=NULL);        /* says what serves fmt=8 instead */
    CHECK(l.n_kv_b_shard==0);
    CHECK(l.kv_b_shard[0]==NULL && l.kv_b_shard[1]==NULL);
    CHECK(l.kv_b.cuda_eligible==1);                /* shard bookkeeping never ran */

    /* (3): bounded -- a second fmt=8 layer (any of the other 60) adds no second line */
    mk_layer_fmt8(&l,q8,s);
    run_shard(&l,err,sizeof err);
    CHECK(count_sub(err,"layer_cuda_shard_kvb")==0);
    CHECK(l.n_kv_b_shard==0);

    /* (4): fmt=6 (E8/IQ3, single 4-byte scale tag) refuses by name with its own message */
    memset(&l,0,sizeof l);
    l.kv_b.fmt=6; l.kv_b.O=O; l.kv_b.I=I; l.kv_b.gs=0;
    l.kv_b.q4=(uint8_t*)q8; l.kv_b.s=s;   /* non-NULL on purpose: only the guard saves it */
    run_shard(&l,err,sizeof err);
    CHECK(strstr(err,"layer_cuda_shard_kvb")!=NULL);
    CHECK(strstr(err,"refus")!=NULL);
    CHECK(strstr(err,"fmt=6")!=NULL);
    CHECK(l.n_kv_b_shard==0);

    /* (5): allowlisted fmt=1 is NOT refused -- upload fails silently (no device
     * context), no shard minted, but no refusal line either */
    memset(&l,0,sizeof l);
    for(int i=0;i<O;i++) srow[i]=0.01f;
    l.kv_b.fmt=1; l.kv_b.O=O; l.kv_b.I=I; l.kv_b.gs=0;
    l.kv_b.q8=q8; l.kv_b.s=srow;
    run_shard(&l,err,sizeof err);
    CHECK(strstr(err,"refus")==NULL);
    CHECK(l.n_kv_b_shard==0);

    if(fails){ printf("layer_cuda_shard_kvb refusal probe: %d FAILED\n", fails); return 1; }
    printf("layer_cuda_shard_kvb refusal probe: ok\n");
    return 0;
}
#endif /* COLI_CUDA */
