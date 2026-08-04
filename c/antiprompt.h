#ifndef COLIBRI_ANTIPROMPT_H
#define COLIBRI_ANTIPROMPT_H
/* Text-level stop sequences ("antiprompts").
 *
 * is_stop() (sample.h) matches token IDs. A heavily-quantized model can emit a
 * role delimiter like "<|user|>" as ORDINARY text tokens ('<','|','user','|','>')
 * instead of the atomic control-token id, so the id stop never fires and the model
 * runs on, hallucinating the next turn. Worse, serve mode deliberately drops the
 * role-marker ids from the stop set (sample.h #401, to protect <tool_call> blocks
 * from int4 argmax noise), so even the atomic token wouldn't stop there.
 *
 * This matches the decoded TEXT instead. It is noise-proof in a way an id stop is
 * not: the full literal "<|user|>" never appears inside a legitimate <tool_call>,
 * so it can't misfire the way a single stop-token id can. Output is streamed with a
 * short hold-back (the longest needle minus one byte) so a delimiter split across
 * tokens is caught and the delimiter itself is never printed.
 *
 * Config: COLI_ANTIPROMPT, ';'-separated needles. Empty / "off" / "0" disables. */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define AP_MAX_NEEDLES 8
#define AP_MAX_NEEDLE  64
typedef struct {
    char needle[AP_MAX_NEEDLES][AP_MAX_NEEDLE];
    int  nlen[AP_MAX_NEEDLES];
    int  n;                 /* number of needles (0 = disabled, pure passthrough) */
    int  maxlen;            /* longest needle: hold-back size is maxlen-1 */
    char pend[512];         /* unflushed tail; may hold a partial needle prefix */
    int  pend_n;
    int  hit;               /* a needle completed on this stream */
} Antiprompt;

/* Configure from COLI_ANTIPROMPT, or `defaults` (same ';'-separated format) when the
 * env var is unset. Empty/"off"/"0" disables (passthrough). */
static void ap_config(Antiprompt *a, const char *defaults){
    memset(a,0,sizeof(*a));
    const char *env=getenv("COLI_ANTIPROMPT");
    if(env && (!*env || !strcmp(env,"off") || !strcmp(env,"0"))) return;   /* disabled */
    const char *spec = env ? env : defaults;
    if(!spec) return;
    const char *p=spec;
    while(*p && a->n<AP_MAX_NEEDLES){
        const char *semi=strchr(p,';');
        size_t L = semi ? (size_t)(semi-p) : strlen(p);
        if(L>0 && L<AP_MAX_NEEDLE){
            memcpy(a->needle[a->n],p,L); a->needle[a->n][L]=0;
            a->nlen[a->n]=(int)L;
            if((int)L>a->maxlen) a->maxlen=(int)L;
            a->n++;
        }
        if(!semi) break;
        p=semi+1;
    }
}

/* Earliest needle occurrence in pend[0..pend_n): returns its start index, or -1. */
static int ap_scan(const Antiprompt *a){
    int best=-1;
    for(int i=0;i<a->n;i++){
        const char *h=strstr(a->pend,a->needle[i]);
        if(h){ int p=(int)(h-a->pend); if(best<0||p<best) best=p; }
    }
    return best;
}

/* Feed one token's decoded text. Writes the now-SAFE prefix to `sink` (may be NULL,
 * e.g. in tests). On a needle match sets a->hit and writes only the bytes BEFORE the
 * delimiter (delimiter and everything after are dropped). Returns 1 if it just hit. */
static int ap_push(Antiprompt *a, const char *text, int tn, FILE *sink){
    if(a->n==0){                                   /* disabled: straight passthrough */
        if(sink && tn>0) fwrite(text,1,(size_t)tn,sink);
        return 0;
    }
    if(a->hit) return 0;                            /* already stopped: swallow the rest */
    /* Append; keep pend bounded. pend never exceeds (maxlen-1)+tn, and tn<=~63 for a
     * single detokenized token, so 512 is ample; guard defensively regardless. */
    if(tn>0){
        if(a->pend_n+tn > (int)sizeof(a->pend)-1){   /* overflow guard: flush oldest */
            int drop=a->pend_n+tn-((int)sizeof(a->pend)-1);
            if(drop>a->pend_n) drop=a->pend_n;
            if(sink) fwrite(a->pend,1,(size_t)drop,sink);
            memmove(a->pend,a->pend+drop,(size_t)(a->pend_n-drop));
            a->pend_n-=drop;
        }
        memcpy(a->pend+a->pend_n,text,(size_t)tn); a->pend_n+=tn;
    }
    a->pend[a->pend_n]=0;

    int at=ap_scan(a);
    if(at>=0){                                     /* matched: emit prefix, drop the rest */
        if(sink && at>0) fwrite(a->pend,1,(size_t)at,sink);
        a->pend_n=0; a->pend[0]=0; a->hit=1;
        return 1;
    }
    /* No match: safe to flush all but the last (maxlen-1) bytes, which could still be
     * the start of a needle completed by the next token. */
    int keep=a->maxlen-1; if(keep<0) keep=0; if(keep>a->pend_n) keep=a->pend_n;
    int flush=a->pend_n-keep;
    if(flush>0){
        if(sink) fwrite(a->pend,1,(size_t)flush,sink);
        memmove(a->pend,a->pend+flush,(size_t)keep);
        a->pend_n=keep; a->pend[keep]=0;
    }
    return 0;
}

/* End of turn with no match: emit the held tail (legit output, not a delimiter). */
static void ap_flush(Antiprompt *a, FILE *sink){
    if(a->n==0 || a->hit){ a->pend_n=0; return; }
    if(sink && a->pend_n>0) fwrite(a->pend,1,(size_t)a->pend_n,sink);
    a->pend_n=0; a->pend[0]=0;
}

#endif
