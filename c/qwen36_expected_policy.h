/* Decode expert-reuse estimates. For every expert: how likely it is
 * to be chosen when its layer runs, and how likely it is to be chosen again
 * right after being chosen, both as averages that halve every 16 decode
 * steps. The retained expert whose next use is expected farthest away is the
 * victim, and a miss replaces it only if its own next use is expected sooner.
 * Metadata only: no payload, I/O or model numerics. Every call holds the
 * owning QwenGlobalCache mutex. */
#ifndef QWEN36_EXPECTED_POLICY_H
#define QWEN36_EXPECTED_POLICY_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t keys, layers, experts, topk, bytes;
    double alpha, beta, prior;
    double *probability, *repeat_numerator, *repeat_denominator;
    unsigned char *last_selected, *current_selected;
    int decode, group_active;
    size_t current_layer;
} QwenExpectedPolicy;

static void qep_free(QwenExpectedPolicy *p) {
    if (!p) return;
    free(p->probability); free(p->repeat_numerator); free(p->repeat_denominator);
    free(p->last_selected); free(p->current_selected); free(p);
}
static QwenExpectedPolicy *qep_new(size_t keys,size_t layers,size_t experts,size_t topk) {
    if (!keys || !layers || !experts || !topk || topk>experts ||
        layers>SIZE_MAX/experts || layers*experts!=keys ||
        keys>SIZE_MAX/(3*sizeof(double)+1)) return NULL;
    size_t bytes=keys*(3*sizeof(double)+1);
    if (bytes>SIZE_MAX-experts || bytes+experts>SIZE_MAX-sizeof(QwenExpectedPolicy)) return NULL;
    QwenExpectedPolicy *p=calloc(1,sizeof(*p)); if (!p) return NULL;
    p->keys=keys; p->layers=layers; p->experts=experts; p->topk=topk;
    p->bytes=sizeof(*p)+bytes+experts;
    /* 1 - 2^(-1/16) as a hexadecimal literal, so that every compiler gets these bits. */
    p->alpha=0x1.5b505d5b6f267p-5; p->beta=1.0-p->alpha;
    p->prior=(double)topk/(double)experts;
    p->probability=malloc(keys*sizeof(double));
    p->repeat_numerator=malloc(keys*sizeof(double));
    p->repeat_denominator=malloc(keys*sizeof(double));
    p->last_selected=calloc(keys,1); p->current_selected=calloc(experts,1);
    if (!p->probability || !p->repeat_numerator || !p->repeat_denominator ||
        !p->last_selected || !p->current_selected) { qep_free(p); return NULL; }
    for (size_t k=0;k<keys;k++) {
        p->probability[k]=p->prior;
        p->repeat_numerator[k]=p->prior*p->prior;
        p->repeat_denominator[k]=p->prior;
    }
    return p;
}
/* A decode group: the top-k of one layer. It is only noted here; the
 * estimates change when the whole group has been acquired (qep_end). Any
 * other group (prefill, or not one layer's top-k) keeps the frequency policy. */
static void qep_begin(QwenExpectedPolicy *p,const int *keys,unsigned n) {
    if (!p) return;
    p->group_active=0;
    if (!p->decode || n!=p->topk || !n || keys[0]<0 || (size_t)keys[0]>=p->keys) return;
    size_t layer=(size_t)keys[0]/p->experts;
    memset(p->current_selected,0,p->experts);
    for (unsigned i=0;i<n;i++) {
        if (keys[i]<0 || (size_t)keys[i]>=p->keys || (size_t)keys[i]/p->experts!=layer) return;
        size_t e=(size_t)keys[i]%p->experts;
        if (p->current_selected[e]) return;
        p->current_selected[e]=1;
    }
    p->current_layer=layer; p->group_active=1;
}
/* Layer steps until the next use of key: the steps to its layer, plus a full
 * round of layers for each time it is expected to be skipped there. */
static double qep_expected_time(const QwenExpectedPolicy *p,size_t key) {
#if defined(__clang__)
#pragma clang fp contract(off)  /* no fused multiply-adds: the same choices on every compiler */
#endif
    size_t layer=key/p->experts;
    size_t distance=(layer+p->layers-p->current_layer)%p->layers;
    if (!distance) distance=p->layers;
    double probability=p->probability[key], next=probability;
    int selected=layer==p->current_layer ? p->current_selected[key%p->experts] : p->last_selected[key];
    if (selected && p->repeat_denominator[key]>0) {
        next=p->repeat_numerator[key]/p->repeat_denominator[key];
        if (next<0) next=0;
        if (next>1) next=1;
    }
    if (probability<=0) return next==1 ? (double)distance : INFINITY;
    return (double)distance+(double)p->layers*(1.0-next)/probability;
}
static void qep_end(QwenExpectedPolicy *p,int observed) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (!p || !p->group_active) return;
    if (observed) {
        size_t begin=p->current_layer*p->experts;
        for (size_t e=0;e<p->experts;e++) {
            size_t k=begin+e;
            p->probability[k]*=p->beta;
            p->repeat_numerator[k]*=p->beta;
            p->repeat_denominator[k]*=p->beta;
        }
        for (size_t e=0;e<p->experts;e++) if (p->current_selected[e]) p->probability[begin+e]+=p->alpha;
        for (size_t e=0;e<p->experts;e++) if (p->last_selected[begin+e]) {
            p->repeat_denominator[begin+e]+=p->alpha;
            if (p->current_selected[e]) p->repeat_numerator[begin+e]+=p->alpha;
        }
        memcpy(p->last_selected+begin,p->current_selected,p->experts);
    }
    p->group_active=0;
}
#endif
