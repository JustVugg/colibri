/* Experimental model-specific V4PG v1 predictor sidecar. Never used by routing. */
#ifndef V4_NP_GATES_H
#define V4_NP_GATES_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef struct { uint16_t *gate; float *intercept; } V4NpGate;
typedef struct { int layers, experts, hidden, depths; V4NpGate *entries; } V4NpGates;
static void v4_np_gates_free(V4NpGates *g) {
    if (g->entries) for (int i=0;i<g->layers*g->depths;i++) {
        free(g->entries[i].gate); free(g->entries[i].intercept);
    }
    free(g->entries); memset(g,0,sizeof(*g));
}
static int v4_np_gates_u32(FILE *f, uint32_t *v) {
    unsigned char b[4]; if (fread(b,1,4,f)!=4) return -1;
    *v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24); return 0;
}
/* Transactional: any malformed entry rejects the entire file. Max payload 1 GiB. */
static int v4_np_gates_load(V4NpGates *out,const char *path,int layers,int experts,int hidden) {
    V4NpGates g={0}; FILE *f=fopen(path,"rb"); char magic[4]; uint32_t h[5],p; size_t bytes=24;
    if (!f) return -1;
    if (fread(magic,1,4,f)!=4 || memcmp(magic,"V4PG",4)) goto bad;
    for (int i=0;i<5;i++) if (v4_np_gates_u32(f,&h[i])) goto bad;
    if (h[0]!=1 || h[1]!=(uint32_t)layers || h[2]!=(uint32_t)experts || h[3]!=(uint32_t)hidden ||
        layers<1 || layers>128 || experts<1 || experts>256 || hidden<1 || hidden>16384 || h[4]<1 || h[4]>2) goto bad;
    g.layers=layers;g.experts=experts;g.hidden=hidden;g.depths=(int)h[4];
    g.entries=calloc((size_t)layers*g.depths,sizeof(*g.entries)); if(!g.entries) goto bad;
    for(int i=0;i<layers*g.depths;i++) {
        if(v4_np_gates_u32(f,&p) || p>1) goto bad;
        bytes+=4; if(!p) continue;
        /* A depth-d predictor cannot target a layer preceding its input. */
        if(i%layers < i/layers+1) goto bad;
        size_t count=(size_t)experts*hidden, payload=count*2+(size_t)experts*4;
        if(payload>((size_t)1<<30)-bytes) goto bad; bytes+=payload;
        V4NpGate *e=&g.entries[i]; e->gate=malloc(count*2);e->intercept=malloc((size_t)experts*4);
        if(!e->gate || !e->intercept) goto bad;
        if(fread(e->gate,2,count,f)!=count) goto bad;
        for(size_t j=0;j<count;j++) {
            unsigned char *b=(unsigned char *)e->gate+2*j;
            uint16_t v=(uint16_t)(b[0]|((uint16_t)b[1]<<8));
            if((v&0x7f80)==0x7f80) goto bad; e->gate[j]=v;
        }
        for(int j=0;j<experts;j++) {uint32_t v;if(v4_np_gates_u32(f,&v)) goto bad;
            memcpy(&e->intercept[j],&v,4);if(!isfinite(e->intercept[j])) goto bad;}
    }
    if(fgetc(f)!=EOF || ferror(f)) goto bad;
    fclose(f);v4_np_gates_free(out);*out=g;return 0;
bad: fclose(f);v4_np_gates_free(&g);return -1;
}
static const V4NpGate *v4_np_gates_get(const V4NpGates *g,int layer,int depth) {
    if(!g->entries || layer<0 || layer>=g->layers || depth<1 || depth>g->depths) return NULL;
    const V4NpGate *e=&g->entries[(depth-1)*g->layers+layer];return e->gate?e:NULL;
}
/* Raw dumped x is already the exporter's input; intercept precedes softplus.
 * _ex (V4_NP_ONLINE): add (n floats, or NULL) is added to the logits after the
 * intercept, and z_out (n floats, or NULL) receives the logits the guess used. */
static int v4_np_gates_route_ex(const V4NpGate *e,float *weights,int *ids,const float *x,
                           float *scratch,const float *bias,int n,int d,int k,float scale,int vector,
                           const float *add,float *z_out) {
    __attribute__((aligned(32))) float logits[256]; if(n<1 || n>256) return -1;
#ifdef V4_NP_ROUTE_AVX2
    if(vector) {
        int r = 0;
        for (; r + 3 < n; r += 4) {
            const uint16_t *row0 = e->gate + (size_t)r * d;
            const uint16_t *row1 = e->gate + (size_t)(r + 1) * d;
            const uint16_t *row2 = e->gate + (size_t)(r + 2) * d;
            const uint16_t *row3 = e->gate + (size_t)(r + 3) * d;
            __m256 a0_0 = _mm256_setzero_ps(), a0_1 = a0_0;
            __m256 a1_0 = a0_0, a1_1 = a0_0;
            __m256 a2_0 = a0_0, a2_1 = a0_0;
            __m256 a3_0 = a0_0, a3_1 = a0_0;
            int c = 0;
            for (; c + 31 < d; c += 32) {
                __m256 vx0 = _mm256_loadu_ps(x + c);
                a0_0 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c), vx0, a0_0);
                a1_0 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c), vx0, a1_0);
                a2_0 = _mm256_fmadd_ps(v4_np_bf16x8(row2 + c), vx0, a2_0);
                a3_0 = _mm256_fmadd_ps(v4_np_bf16x8(row3 + c), vx0, a3_0);

                __m256 vx1 = _mm256_loadu_ps(x + c + 8);
                a0_1 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c + 8), vx1, a0_1);
                a1_1 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c + 8), vx1, a1_1);
                a2_1 = _mm256_fmadd_ps(v4_np_bf16x8(row2 + c + 8), vx1, a2_1);
                a3_1 = _mm256_fmadd_ps(v4_np_bf16x8(row3 + c + 8), vx1, a3_1);

                __m256 vx2 = _mm256_loadu_ps(x + c + 16);
                a0_0 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c + 16), vx2, a0_0);
                a1_0 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c + 16), vx2, a1_0);
                a2_0 = _mm256_fmadd_ps(v4_np_bf16x8(row2 + c + 16), vx2, a2_0);
                a3_0 = _mm256_fmadd_ps(v4_np_bf16x8(row3 + c + 16), vx2, a3_0);

                __m256 vx3 = _mm256_loadu_ps(x + c + 24);
                a0_1 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c + 24), vx3, a0_1);
                a1_1 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c + 24), vx3, a1_1);
                a2_1 = _mm256_fmadd_ps(v4_np_bf16x8(row2 + c + 24), vx3, a2_1);
                a3_1 = _mm256_fmadd_ps(v4_np_bf16x8(row3 + c + 24), vx3, a3_1);
            }
            float sum0 = v4_np_hsum8(_mm256_add_ps(a0_0, a0_1));
            float sum1 = v4_np_hsum8(_mm256_add_ps(a1_0, a1_1));
            float sum2 = v4_np_hsum8(_mm256_add_ps(a2_0, a2_1));
            float sum3 = v4_np_hsum8(_mm256_add_ps(a3_0, a3_1));
            for (; c < d; c++) {
                sum0 += v4_np_bf16(row0[c]) * x[c];
                sum1 += v4_np_bf16(row1[c]) * x[c];
                sum2 += v4_np_bf16(row2[c]) * x[c];
                sum3 += v4_np_bf16(row3[c]) * x[c];
            }
            logits[r] = sum0;
            logits[r + 1] = sum1;
            logits[r + 2] = sum2;
            logits[r + 3] = sum3;
        }
        for (; r + 1 < n; r += 2) {
            const uint16_t *row0 = e->gate + (size_t)r * d;
            const uint16_t *row1 = e->gate + (size_t)(r + 1) * d;
            __m256 a0_0 = _mm256_setzero_ps(), a0_1 = a0_0, a0_2 = a0_0, a0_3 = a0_0;
            __m256 a1_0 = a0_0, a1_1 = a0_0, a1_2 = a0_0, a1_3 = a0_0;
            int c = 0;
            for (; c + 31 < d; c += 32) {
                __m256 vx0 = _mm256_loadu_ps(x + c);
                __m256 vx1 = _mm256_loadu_ps(x + c + 8);
                __m256 vx2 = _mm256_loadu_ps(x + c + 16);
                __m256 vx3 = _mm256_loadu_ps(x + c + 24);
                a0_0 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c), vx0, a0_0);
                a1_0 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c), vx0, a1_0);
                a0_1 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c + 8), vx1, a0_1);
                a1_1 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c + 8), vx1, a1_1);
                a0_2 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c + 16), vx2, a0_2);
                a1_2 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c + 16), vx2, a1_2);
                a0_3 = _mm256_fmadd_ps(v4_np_bf16x8(row0 + c + 24), vx3, a0_3);
                a1_3 = _mm256_fmadd_ps(v4_np_bf16x8(row1 + c + 24), vx3, a1_3);
            }
            float sum0 = v4_np_hsum8(_mm256_add_ps(_mm256_add_ps(a0_0, a0_1), _mm256_add_ps(a0_2, a0_3)));
            float sum1 = v4_np_hsum8(_mm256_add_ps(_mm256_add_ps(a1_0, a1_1), _mm256_add_ps(a1_2, a1_3)));
            for (; c < d; c++) {
                sum0 += v4_np_bf16(row0[c]) * x[c];
                sum1 += v4_np_bf16(row1[c]) * x[c];
            }
            logits[r] = sum0;
            logits[r + 1] = sum1;
        }
        for (; r < n; r++) {
            const uint16_t *row = e->gate + (size_t)r * d;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            int c = 0;
            for (; c + 31 < d; c += 32) {
                a0 = _mm256_fmadd_ps(v4_np_bf16x8(row + c), _mm256_loadu_ps(x + c), a0);
                a1 = _mm256_fmadd_ps(v4_np_bf16x8(row + c + 8), _mm256_loadu_ps(x + c + 8), a1);
                a2 = _mm256_fmadd_ps(v4_np_bf16x8(row + c + 16), _mm256_loadu_ps(x + c + 16), a2);
                a3 = _mm256_fmadd_ps(v4_np_bf16x8(row + c + 24), _mm256_loadu_ps(x + c + 24), a3);
            }
            float sum = v4_np_hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
            for (; c < d; c++) sum += v4_np_bf16(row[c]) * x[c];
            logits[r] = sum;
        }
    }
    else
#endif
    { for(int r=0;r<n;r++) {float sum=0;for(int c=0;c<d;c++) {uint32_t b=(uint32_t)e->gate[(size_t)r*d+c]<<16;float w;memcpy(&w,&b,4);sum+=w*x[c];}logits[r]=sum;} }
    (void)scratch;(void)vector;
    for(int r=0;r<n;r++) {logits[r]+=e->intercept[r];if(!isfinite(logits[r])) return -1;}
    if(add) for(int r=0;r<n;r++) {logits[r]+=add[r];if(!isfinite(logits[r])) return -1;}
    if(z_out) memcpy(z_out,logits,(size_t)n*sizeof(*z_out));
    return coli_v4_route_bf16_logits(weights,ids,x,e->gate,bias,NULL,n,d,k,scale,logits);
}
static int v4_np_gates_route(const V4NpGate *e,float *weights,int *ids,const float *x,
                           float *scratch,const float *bias,int n,int d,int k,float scale,int vector) {
    return v4_np_gates_route_ex(e,weights,ids,x,scratch,bias,n,d,k,scale,vector,NULL,NULL);
}
#endif
