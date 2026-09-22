/* Validation for the GLM reference file; no model or Python dependency. */
#ifndef COLI_ORACLE_H
#define COLI_ORACLE_H
#include <errno.h>
#include <math.h>
#include "json.h"

typedef struct {
    int *prompt, *full, *tf;
    int np, nfull;
} OracleRef;

static void oracle_ref_free(OracleRef *r) {
    free(r->prompt); free(r->full); free(r->tf);
    memset(r, 0, sizeof(*r));
}

static int *oracle_read_ids(jval *root, const char *key, int vocab, int *n) {
    jval *a=json_get(root,key);
    *n=0;
    if (!a || a->t!=J_ARR || a->len<1) {
        fprintf(stderr,"[ORACLE] %s must be a nonempty token array\n",key);
        return NULL;
    }
    int *ids=malloc((size_t)a->len*sizeof(*ids));
    if (!ids) { fprintf(stderr,"[ORACLE] out of memory reading %s\n",key); return NULL; }
    for (int i=0; i<a->len; i++) {
        jval *v=a->kids[i];
        if (v->t!=J_NUM || !isfinite(v->num) || v->num<0 || v->num>=vocab ||
            v->num!=floor(v->num)) {
            fprintf(stderr,"[ORACLE] %s[%d] must be an integer token in [0,%d)\n",key,i,vocab);
            free(ids); return NULL;
        }
        ids[i]=(int)v->num;
    }
    *n=a->len;
    return ids;
}

static int oracle_ref_parse(const char *text, int vocab, int teacher_forcing, OracleRef *r) {
    memset(r,0,sizeof(*r));
    jval *root=json_parse_checked(text);
    if (!root || root->t!=J_OBJ || vocab<1) {
        fprintf(stderr,"[ORACLE] invalid reference JSON or vocabulary\n");
        json_free(root); return 0;
    }
    /* Duplicate keys could otherwise silently select a stale prediction array. */
    const char *keys[]={"prompt_ids","full_ids","tf_pred"};
    for (int k=0; k<3; k++) {
        int count=0;
        for (int i=0; i<root->len; i++) if (!strcmp(root->keys[i],keys[k])) count++;
        if (count>1) {
            fprintf(stderr,"[ORACLE] duplicate reference field %s\n",keys[k]);
            goto fail;
        }
    }
    r->prompt=oracle_read_ids(root,"prompt_ids",vocab,&r->np);
    r->full=oracle_read_ids(root,"full_ids",vocab,&r->nfull);
    if (!r->prompt || !r->full) goto fail;
    if (r->nfull<r->np || (!teacher_forcing && r->nfull==r->np) ||
        memcmp(r->prompt,r->full,(size_t)r->np*sizeof(int))) {
        fprintf(stderr,"[ORACLE] full_ids must start with prompt_ids and include the compared tokens\n");
        goto fail;
    }
    if (teacher_forcing) {
        int ntf=0;
        r->tf=oracle_read_ids(root,"tf_pred",vocab,&ntf);
        if (!r->tf) goto fail;
        if (ntf!=r->nfull) {
            fprintf(stderr,"[ORACLE] tf_pred length %d != full_ids length %d\n",ntf,r->nfull);
            goto fail;
        }
    }
    json_free(root); return 1;
fail:
    json_free(root); oracle_ref_free(r); return 0;
}

static int oracle_logits_finite(const float *lo, int vocab) {
    for (int i=0; i<vocab; i++) if (!isfinite(lo[i])) return 0;
    return 1;
}

/* Exact by default. A configured budget must leave at least one required match. */
static int oracle_tf_allowance(const char *text, int total, int *allowed) {
    *allowed=0;
    if (total<1) return 0;
    if (!text) return 1;
    if (*text<'0' || *text>'9') return 0;
    char *end;
    errno=0;
    long value=strtol(text,&end,10);
    if (errno==ERANGE || *end || value<0 || value>=total) return 0;
    *allowed=(int)value;
    return 1;
}
#endif
