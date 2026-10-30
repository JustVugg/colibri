/* Original upstream sampler, frozen from c0f785c9 for bit-exact regression. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static float g_temp, g_topp;
static int draw_value, draw_calls;
static int fixed_rand(void) { draw_calls++; return draw_value; }
#define rand fixed_rand
static int argmax(const float*x,int n) { int k=0;for(int i=1;i<n;i++)if(x[i]>x[k])k=i;return k; }
static const float *g_sort_key = NULL;
static int compare_by_probability(const void *a, const void *b) {
    const int x = *(const int *)a, y = *(const int *)b;
    if (g_sort_key[x] < g_sort_key[y]) return 1;
    if (g_sort_key[x] > g_sort_key[y]) return -1;
    return 0;
}

/* Temperatura piu' nucleus. Con temperatura nulla e' argmax e basta. */
static int sample_token(const float *logits, int vocab) {
    if (g_temp <= 1e-4f) return argmax(logits, vocab);
    static float *probability = NULL;
    static int *order = NULL;
    static int room = 0;
    if (room < vocab) {
        probability = realloc(probability, (size_t)vocab * sizeof(float));
        order = realloc(order, (size_t)vocab * sizeof(int));
        room = vocab;
        if (!probability || !order) { fprintf(stderr, "OOM in sampling\n"); exit(1); }
    }
    float top = -INFINITY;
    for (int i = 0; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    const float inverse = 1.0f / g_temp;
    for (int i = 0; i < vocab; i++) {
        probability[i] = expf((logits[i] - top) * inverse);
        total += probability[i];
        order[i] = i;
    }
    for (int i = 0; i < vocab; i++) probability[i] = (float)(probability[i] / total);

    /* nucleus: si ordina per probabilita' e si taglia dove la somma supera
     * top_p. Con vocab 154880 l'ordinamento pieno costa, ma succede una volta
     * per token e il token costa incomparabilmente di piu'. */
    g_sort_key = probability;
    qsort(order, (size_t)vocab, sizeof(int), compare_by_probability);
    double cumulative = 0.0;
    int keep = vocab;
    for (int i = 0; i < vocab; i++) {
        cumulative += probability[order[i]];
        if (cumulative >= g_topp) { keep = i + 1; break; }
    }
    double pick = ((double)rand() / ((double)RAND_MAX + 1.0)) * cumulative;
    double walk = 0.0;
    for (int i = 0; i < keep; i++) {
        walk += probability[order[i]];
        if (walk >= pick) return order[i];
    }
    return order[0];
}


int nucleus_reference(const float*x,int n,float t,float p,int draw,int*used) {
    g_temp=t;g_topp=p;draw_value=draw;draw_calls=0;
    int y=sample_token(x,n);*used=draw_calls;return y;
}
