/* RoPE inv_freq precompute (byte-identical guard).
 *
 * rope_interleave() used to recompute inv=powf(theta,-2j/qk_rope) for every
 * position inside its thread-local cache refresh; it now precomputes inv[j]
 * once (keyed on theta/qk) and forms ang=pos*inv[j] per position. The rotation
 * is byte-sensitive: cs/sn feed q/k dot products, so any bit drift changes the
 * sampled token. This test drives the REAL rope_interleave (via the test_topp.c
 * include-colibri.c pattern) and compares its rotated output, BIT-FOR-BIT,
 * against an independent reference that reimplements the OLD powf-per-position
 * formula. Sweeps the qk_rope sizes and rope_theta values GLM/Kimi actually use.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <math.h>
#include <stdint.h>

#define CHECK(cond) do { if(!(cond)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#cond); return 1; } } while(0)

static int biteq(float a, float b){ uint32_t x,y; memcpy(&x,&a,4); memcpy(&y,&b,4); return x==y; }

/* Reference: the pre-optimisation rotation, powf recomputed every call. */
static void rope_ref(float *v, int pos, int qk, float theta){
    int half=qk/2; float in[256]; memcpy(in,v,qk*sizeof(float));
    for(int j=0;j<half;j++){
        float inv=powf(theta,-2.0f*j/qk), ang=pos*inv;
        float cs=cosf(ang), sn=sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
}

int main(void){
    const int qks[]   = {64,128,96,256};
    const float ths[] = {10000.f, 1000000.f, 500000.f};
    long checked=0;
    for(int qi=0; qi<4; qi++) for(int ti=0; ti<3; ti++){
        int qk=qks[qi]; float th=ths[ti];
        Cfg c; memset(&c,0,sizeof c); c.qk_rope=qk; c.theta=th;
        for(int pos=0; pos<200000; pos += (pos<4096 ? 1 : 37)){
            float base[256], vr[256], vn[256];
            /* deterministic, varied input (no rand: reproducible across platforms) */
            for(int i=0;i<qk;i++) base[i] = sinf((float)(i*7 + pos)) * 1.5f - 0.3f;
            memcpy(vr,base,qk*sizeof(float)); memcpy(vn,base,qk*sizeof(float));
            rope_ref(vr,pos,qk,th);
            rope_interleave(vn,pos,&c);          /* the REAL shipping function */
            for(int i=0;i<qk;i++){
                checked++;
                CHECK(biteq(vr[i],vn[i]));
            }
        }
    }
    printf("test_rope_invfreq: %ld rotated floats bit-identical to reference\n",checked);
    return 0;
}
