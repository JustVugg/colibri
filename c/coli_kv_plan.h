/* coli_kv_plan.h -- the KV cache planned once, for the whole context (COLI_KV_PLAN, on).
 *
 * A chain engine's KV cache used to grow with the conversation: the host's rows reallocated and
 * copied as a request passed their end, the device's mirror sized to what a step reached (and an
 * eighth more), allocated again and uploaded whole when it grew, its room taken from the expert
 * tier at that moment. Each growth asked the device for a large buffer and the staging to fill it
 * when its memory was the fullest, mid-conversation: a refused allocation stopped the chain (#2793),
 * and the upload made one submission long. Planned, the host's cache is allocated for the whole
 * context at its first use (rows a conversation never reaches are pages never touched), the
 * device's mirror for the whole context at its first forward (whole where the plan's room holds
 * it, else the split, vk_kvsplit.h, decided then), and the expert tier keeps that room from the
 * start. Nothing grows, shrinks or uploads again while a conversation runs. COLI_KV_PLAN=0 keeps
 * the growth. */
#ifndef COLI_KV_PLAN_H
#define COLI_KV_PLAN_H
#include <stdlib.h>
static inline int coli_kv_plan(void) {   /* read at each call: the tests switch it between cases */
    const char *e = getenv("COLI_KV_PLAN");
    return !(e && *e == '0');
}
/* The host's cache at the whole context: planned, and `bytes` at most half of the RAM free now (avail, bytes;
 * 0 when unknown: allowed). Windows commits what is allocated, touched or not: a cache past the commit room
 * would fail its allocation where the growth would have lasted, so past half of it the growth stays. */
static inline int coli_kv_plan_host(double bytes, double avail) {
    return coli_kv_plan() && (avail <= 0 || bytes <= avail / 2);
}
#endif
