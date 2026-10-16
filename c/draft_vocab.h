/* draft_vocab.h -- the tokens a draft head scores.
 *
 * Scoring the whole vocabulary spends most of the time on tokens a conversation never uses.
 * The list has the `top` tokens with the highest counts, every token with a count of at
 * least DRAFT_VOCAB_NEAR_NUM/DRAFT_VOCAB_NEAR_DEN of the count of the last of those, and
 * every token of the current conversation.
 *
 * Every token read as part of a prompt and every token picked adds 1 to the count of that
 * token. All counts halve after DRAFT_VOCAB_HALVE_EVERY counted tokens, so the list follows
 * the topic. */
#ifndef COLI_DRAFT_VOCAB_H
#define COLI_DRAFT_VOCAB_H
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "own_file.h"    /* the counts file sits in a model directory that may come from a download */

/* The size of the list when Q38_MTP_VOCAB is unset. On replays of recorded drafts, 1,000 tokens
 * lost speed on short chats and 16,000 gained. */
#define DRAFT_VOCAB_DEFAULT_TOP 16000

/* A token whose count is at least NUM/DEN of the count of the last listed one is almost as
 * frequent, so the list includes it instead of ending exactly at the N-th token. */
#define DRAFT_VOCAB_NEAR_NUM 3
#define DRAFT_VOCAB_NEAR_DEN 4

/* Counts halve after this many counted tokens, the period of VKT_DECAY_TOKENS in vk_tier.c. */
#define DRAFT_VOCAB_HALVE_EVERY 1024

typedef enum {
    DRAFT_VOCAB_LOADED,
    DRAFT_VOCAB_NO_FILE,       /* nothing at the path: the first save creates the file */
    DRAFT_VOCAB_NOT_COUNTS,    /* the path cannot be read as a counts file */
    DRAFT_VOCAB_OTHER_SIZE,    /* counts for another vocabulary size */
    DRAFT_VOCAB_DAMAGED        /* the counts end at a record that cannot be read */
} DraftVocabLoad;

typedef enum {
    DRAFT_VOCAB_SAVED,
    DRAFT_VOCAB_LEFT_ALONE,    /* the file at the path is not counts that the load read to the end */
    DRAFT_VOCAB_NOT_SAVED      /* the write failed */
} DraftVocabSave;

/* One request: the tokens passed to draft_vocab_count_token since draft_vocab_begin. */
typedef struct {
    uint64_t picks, picks_inside, list_tokens, appended, rebuilds;
} DraftVocabRun;

typedef struct {
    int vocab, top;
    float *count;                         /* per token id; floats, so halving is exact and never rounds a count to 0 */
    uint8_t *in_list, *in_conversation;   /* per token id */
    int *list, list_n;                    /* ascending ids from the last rebuild, then the appended ones */
    int counted;                          /* tokens counted since the last halving */
    char *path;                           /* the counts file, allocated; set by draft_vocab_load */
    DraftVocabLoad found;                 /* what draft_vocab_load found at the path */
    int records;                          /* the count records it read */
    DraftVocabRun run;
} DraftVocab;

/* The bits of a count. Counts are not negative, and the bits of non-negative floats
 * order like the floats do. */
static uint32_t draft_vocab_bits(float count) {
    uint32_t bits;
    memcpy(&bits,&count,sizeof bits);
    return bits;
}

/* The n-th largest of the `vocab` counts (1 <= n <= vocab), found by its bytes from the
 * top. *above gets how many counts are larger than it. */
static float draft_vocab_nth_largest(const float *count,int vocab,int n,int *above) {
    uint64_t nth=0;   /* the bytes found so far, the rest 0 */
    int left=n;       /* the rank of the n-th count among those that share the bytes found so far */
    *above=0;
    for(int shift=24;shift>=0;shift-=8){
        int bytes[256]={0};
        for(int i=0;i<vocab;i++){
            uint32_t bits=draft_vocab_bits(count[i]);
            if(((uint64_t)bits>>(shift+8))==(nth>>(shift+8)))bytes[(bits>>shift)&255]++;
        }
        int byte=255;
        while(byte>0&&bytes[byte]<left){left-=bytes[byte];*above+=bytes[byte];byte--;}
        nth|=(uint64_t)byte<<shift;
    }
    uint32_t bits=(uint32_t)nth;
    float value;
    memcpy(&value,&bits,sizeof value);
    return value;
}

static int draft_vocab_near_top(double count,double nth) {
    return DRAFT_VOCAB_NEAR_DEN*count>=DRAFT_VOCAB_NEAR_NUM*nth;
}

static void draft_vocab_rebuild(DraftVocab *v) {
    int above;
    double nth=draft_vocab_nth_largest(v->count,v->vocab,v->top,&above);
    int zero_slots=nth>0?0:v->top-above;   /* fewer than `top` counted tokens: the lowest ids fill the list */
    memset(v->in_list,0,(size_t)v->vocab);
    v->list_n=0;
    for(int id=0;id<v->vocab;id++){
        double c=v->count[id];
        int in=0;
        if(c>0)in=draft_vocab_near_top(c,nth);
        else if(zero_slots>0){in=1;zero_slots--;}
        if(in||v->in_conversation[id]){v->in_list[id]=1;v->list[v->list_n++]=id;}
    }
}

static void draft_vocab_free(DraftVocab *v) {
    if(!v)return;
    free(v->count);free(v->in_list);free(v->in_conversation);free(v->list);free(v->path);
    free(v);
}

/* A list for `vocab` tokens that scores the `top` most used; NULL when out of memory. */
static DraftVocab *draft_vocab_new(int vocab,int top) {
    DraftVocab *v=(DraftVocab*)calloc(1,sizeof *v);
    if(!v)return NULL;
    v->vocab=vocab;v->top=top<vocab?top:vocab;
    v->count=(float*)calloc((size_t)vocab,sizeof *v->count);
    v->in_list=(uint8_t*)calloc((size_t)vocab,1);
    v->in_conversation=(uint8_t*)calloc((size_t)vocab,1);
    v->list=(int*)malloc((size_t)vocab*sizeof *v->list);
    if(!v->count||!v->in_list||!v->in_conversation||!v->list){draft_vocab_free(v);return NULL;}
    draft_vocab_rebuild(v);
    return v;
}

typedef enum { DRAFT_VOCAB_COUNTED, DRAFT_VOCAB_HALVED } DraftVocabCount;

/* Adds 1 to the count of `tok`, a valid id. DRAFT_VOCAB_HALVED when that completed a period
 * and halved every count. */
static DraftVocabCount draft_vocab_count(DraftVocab *v,int tok) {
    v->count[tok]+=1.f;
    if(++v->counted<DRAFT_VOCAB_HALVE_EVERY)return DRAFT_VOCAB_COUNTED;
    for(int i=0;i<v->vocab;i++)v->count[i]*=0.5f;
    v->counted=0;
    return DRAFT_VOCAB_HALVED;
}

/* An id outside the vocabulary would index the arrays of v out of range. */
static void draft_vocab_check_token(const DraftVocab *v,int tok) {
    if(tok>=0&&tok<v->vocab)return;
    fprintf(stderr,"draft vocabulary: token id %d is outside 0..%d\n",tok,v->vocab-1);
    exit(1);
}

/* A request starts: its prompt of n tokens is the conversation. The state of the first
 * `cached_prompt_tokens` tokens came from a cache, and they were counted when they were
 * first read. The other tokens are counted now, and the list is rebuilt. */
static void draft_vocab_begin(DraftVocab *v,const int *prompt,int n,int cached_prompt_tokens) {
    memset(v->in_conversation,0,(size_t)v->vocab);
    for(int i=0;i<n;i++){draft_vocab_check_token(v,prompt[i]);v->in_conversation[prompt[i]]=1;}
    for(int i=cached_prompt_tokens;i<n;i++)draft_vocab_count(v,prompt[i]);
    draft_vocab_rebuild(v);
    memset(&v->run,0,sizeof v->run);
}

/* The request picked the token `tok`: it is counted and added to the list if missing. */
static void draft_vocab_count_token(DraftVocab *v,int tok) {
    DraftVocabRun *r=&v->run;
    draft_vocab_check_token(v,tok);
    r->picks++;r->list_tokens+=(uint64_t)v->list_n;
    if(v->in_list[tok])r->picks_inside++;
    else{v->in_list[tok]=1;v->list[v->list_n++]=tok;r->appended++;}
    v->in_conversation[tok]=1;
    if(draft_vocab_count(v,tok)==DRAFT_VOCAB_HALVED){draft_vocab_rebuild(v);r->rebuilds++;}
}

/* The numbers of the report line of a request: the size of the list averaged over its picks,
 * the share of its picks that were in the list, and the ids appended and the rebuilds. */
typedef struct {
    double average_list_tokens, inside_percent;
    uint64_t appended, rebuilds;
} DraftVocabSummary;

static DraftVocabSummary draft_vocab_summary(const DraftVocab *v) {
    const DraftVocabRun *r=&v->run;
    DraftVocabSummary summary={0,0,r->appended,r->rebuilds};
    if(r->picks){
        summary.average_list_tokens=(double)r->list_tokens/r->picks;
        summary.inside_percent=100.0*r->picks_inside/r->picks;
    }
    return summary;
}

/* The sum of the counts and how many tokens have one. */
static void draft_vocab_totals(const DraftVocab *v,double *tokens,int *distinct) {
    *tokens=0;*distinct=0;
    for(int id=0;id<v->vocab;id++)if(v->count[id]>0){*tokens+=v->count[id];(*distinct)++;}
}

/* The counts file is text. Its first line is "-1 <vocab>", then one line "<id> <count>" per
 * token with a count above 0. The counts of a file for another vocabulary size are not read. */
static DraftVocabLoad draft_vocab_read(DraftVocab *v) {
    FILE *f=coli_own_fopen(v->path,"rb");
    if(!f)return errno==ENOENT?DRAFT_VOCAB_NO_FILE:DRAFT_VOCAB_NOT_COUNTS;
    int id,size;
    if(fscanf(f,"%d %d",&id,&size)!=2||id!=-1){fclose(f);return DRAFT_VOCAB_NOT_COUNTS;}
    if(size!=v->vocab){fclose(f);return DRAFT_VOCAB_OTHER_SIZE;}
    float count;
    int scanned;
    while((scanned=fscanf(f,"%d %f",&id,&count))==2){
        v->records++;
        if(id>=0&&id<v->vocab&&count>0&&isfinite(count))v->count[id]=count;
    }
    int damaged=scanned!=EOF||ferror(f);
    fclose(f);
    draft_vocab_rebuild(v);
    return damaged?DRAFT_VOCAB_DAMAGED:DRAFT_VOCAB_LOADED;
}

/* `path`, an allocated string, becomes the counts file of v: v frees it. v records what it
 * found there, and draft_vocab_save replaces the file only if that was nothing or counts that
 * the load read to the end. */
static DraftVocabLoad draft_vocab_load(DraftVocab *v,char *path) {
    free(v->path);
    v->path=path;
    v->records=0;
    v->found=draft_vocab_read(v);
    return v->found;
}

/* The counts file of v, replaced as a whole, so a failed write leaves the old one. */
static DraftVocabSave draft_vocab_save(const DraftVocab *v) {
    if(v->found!=DRAFT_VOCAB_NO_FILE&&v->found!=DRAFT_VOCAB_LOADED)return DRAFT_VOCAB_LEFT_ALONE;
    FILE *f=coli_own_fopen_tmp(v->path);
    if(!f)return DRAFT_VOCAB_NOT_SAVED;
    fprintf(f,"-1 %d\n",v->vocab);
    for(int id=0;id<v->vocab;id++)if(v->count[id]>0)fprintf(f,"%d %.9g\n",id,v->count[id]);
    int failed=ferror(f);
    if(fclose(f)!=0||failed)return DRAFT_VOCAB_NOT_SAVED;
    return coli_own_replace(v->path)==0?DRAFT_VOCAB_SAVED:DRAFT_VOCAB_NOT_SAVED;
}

#endif /* COLI_DRAFT_VOCAB_H */
