#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../backend_metal.h"

static float ref(float g, float u, float limit) {
    float gv = fminf(g, limit);
    float uv = fmaxf(-limit, fminf(u, limit));
    return (gv / (1.0f + expf(-gv))) * uv;
}

int main(void) {
    const float limit = 10.0f;
    float g[] = {-20.0f,-10.0001f,-10.0f,-9.9999f,-1.0f,-0.0f,0.0f,1.0f,9.9999f,10.0f,10.0001f,20.0f,
                 3.25f,-7.75f,12.5f,0.125f};
    float u[] = {-20.0f,-10.0001f,-10.0f,-9.9999f,-1.0f,-0.0f,0.0f,1.0f,9.9999f,10.0f,10.0001f,20.0f,
                 -12.5f,7.75f,3.25f,-0.125f};
    enum { N = sizeof(g)/sizeof(g[0]) };
    float expect[N];
    for (int i=0;i<N;i++) expect[i]=ref(g[i],u[i],limit);

    if (!coli_metal_init() || !coli_metal_available()) {
        fprintf(stderr,"SKIP: Metal unavailable\n");
        return 77;
    }
    if (!coli_metal_silu_mul_clamped(g,u,N,limit)) {
        fprintf(stderr,"FAIL: Metal clamped SwiGLU call failed\n");
        return 2;
    }

    float max_abs=0.0f, max_rel=0.0f;
    int bad=0;
    for (int i=0;i<N;i++) {
        float ae=fabsf(g[i]-expect[i]);
        float re=ae/fmaxf(1.0f,fabsf(expect[i]));
        if (ae>max_abs) max_abs=ae;
        if (re>max_rel) max_rel=re;
        if (!(ae <= 2e-5f || re <= 2e-5f)) {
            fprintf(stderr,"mismatch[%d]: metal=%.9g cpu=%.9g abs=%.3g rel=%.3g\n",i,g[i],expect[i],ae,re);
            bad++;
        }
    }
    printf("GLM53 clamped SwiGLU oracle: n=%d limit=%.3f max_abs=%.9g max_rel=%.9g\n",N,limit,max_abs,max_rel);
    coli_metal_shutdown();
    if (bad) { fprintf(stderr,"FAIL: %d mismatches\n",bad); return 1; }
    puts("GLM53 clamped SwiGLU oracle: OK");
    return 0;
}
