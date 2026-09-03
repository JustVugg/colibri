/* wq-v0-class (e8x4g64: int8 spine + grouped-int4 g64 experts) END-TO-END
 * loader harness -- the C-side half of the mint->load regression case.
 *
 * tests/test_e8x4g64_mint_load.py drives it: builds a synthetic FP8 GLM-shaped
 * checkpoint (tools/glm_fp8_emit.py, the same fixture helper the fp8
 * passthrough e2e uses), mints it with the REAL tools/convert_fp8_to_int4.py
 * CLI at the e8x4g64 recipe (--ebits 8 --xbits 4 --io-bits 8 --group-size 64),
 * then invokes this binary against the real output directory. Sibling of
 * tests/test_fp8_e2e_loader.c (fmt=8 passthrough), same division of labor:
 * the Python side proves the TOOL ran for real; this side proves the REAL
 * st_init/qt_from_disk (the identical functions every model load uses)
 * resolve every minted tensor to the format the recipe promises.
 *
 * Usage: test_e8x4g64_loader <container_dir> [name O I wantfmt wantgs]...
 * For every (name,O,I,wantfmt,wantgs) 5-tuple, calls the REAL qt_from_disk and
 * asserts:
 *   (a) the resolved fmt equals wantfmt (1 = int8 per-row spine, 4 = grouped
 *       int4 experts) and, for fmt=4, the derived group size equals wantgs --
 *       both resolved by qt_resolve_fmt's byte arithmetic on the tool's actual
 *       output, neither mocked nor stamped;
 *   (b) the weight and scale buffers are non-NULL;
 *   (c) every dequantized value is finite -- catches a scale-layout or
 *       packing bug a pure byte-count check wouldn't.
 *
 * The D-2 duplicate-name refusal is exercised through this same binary: the
 * Python driver runs it a second time against a copy of the container holding
 * a duplicated shard, and st_init below then refuses (exit 1, naming both
 * shards) before any tuple is checked -- the positive half of the D-2 pair,
 * on a tool-produced container. The clean run doubles as the negative
 * control: a legitimate mint loads with no refusal. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv){
    if(argc < 2){ fprintf(stderr,"usage: %s <dir> [name O I wantfmt wantgs]...\n", argv[0]); return 2; }
    if((argc-2) % 5 != 0){
        fprintf(stderr,"args after <dir> must come in (name,O,I,wantfmt,wantgs) 5-tuples (got %d)\n", argc-2);
        return 2;
    }
    const char *dir = argv[1];
    int ntensors = (argc-2)/5;
    if(ntensors == 0){ fprintf(stderr,"no tuples given -- nothing to check\n"); return 2; }

    static Model gm; memset(&gm,0,sizeof gm);
    st_init(&gm.S, dir);   /* D-2 duplicate-name detection lives in here (st.h) */

    int fails = 0;
    for(int i=0;i<ntensors;i++){
        const char *name = argv[2+i*5];
        int O       = atoi(argv[3+i*5]);
        int I       = atoi(argv[4+i*5]);
        int wantfmt = atoi(argv[5+i*5]);
        int wantgs  = atoi(argv[6+i*5]);

        QT t; memset(&t,0,sizeof t);
        qt_from_disk(&gm, name, O, I, 8, 0, &t);   /* THE REAL LOADER PATH -- not mocked */

        if(t.fmt != wantfmt){
            printf("FAIL %s: fmt=%d, expected %d\n", name, t.fmt, wantfmt);
            fails++; continue;
        }
        if(wantfmt == 4 && t.gs != wantgs){
            printf("FAIL %s: fmt=4 resolved with gs=%d, expected g%d\n", name, t.gs, wantgs);
            fails++; continue;
        }
        const void *w = t.fmt==1 ? (const void*)t.q8 : (const void*)t.q4;
        if(!w || !t.s){
            printf("FAIL %s: fmt=%d but weights=%p s=%p (expected both non-NULL)\n",
                   name, t.fmt, w, (void*)t.s);
            fails++; continue;
        }
        /* (c) all-finite dequant, per format's own scale geometry */
        int64_t bad = -1;
        if(t.fmt == 1){
            for(int64_t o=0; o<O && bad<0; o++){ float s=t.s[o];
                for(int64_t ii=0; ii<I; ii++){
                    float v=(float)t.q8[o*(int64_t)I+ii]*s;
                    if(!isfinite(v)){ bad=o*(int64_t)I+ii; break; } } }
        } else if(t.fmt == 4){
            int ng=(I+t.gs-1)/t.gs;
            for(int64_t o=0; o<O && bad<0; o++){ const float *scl=t.s+o*ng;
                const uint8_t *row=t.q4+o*(int64_t)((I+1)/2);
                for(int64_t ii=0; ii<I; ii++){
                    uint8_t b=row[ii>>1]; int nib=(ii&1)?(b>>4):(b&0xF);
                    float v=((int)nib-8)*scl[ii/t.gs];
                    if(!isfinite(v)){ bad=o*(int64_t)I+ii; break; } } }
        } else {
            printf("FAIL %s: this harness only knows the e8x4g64 formats (1, 4); "
                   "wantfmt=%d is a driver bug\n", name, wantfmt);
            fails++; continue;
        }
        if(bad >= 0){
            printf("FAIL %s: non-finite dequantized value at flat index %lld\n", name, (long long)bad);
            fails++; continue;
        }
        printf("ok %s: fmt=%d%s O=%d I=%d, loaded through the real loader, all-finite\n",
               name, t.fmt, t.fmt==4?" (g64)":"", O, I);
    }
    if(fails){ printf("e8x4g64 mint->load: %d/%d tensor(s) FAILED\n", fails, ntensors); return 1; }
    printf("e8x4g64 mint->load: ok (%d tensor(s))\n", ntensors);
    return 0;
}
