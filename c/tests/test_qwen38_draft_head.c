/* The MTP head's draft from a list of tokens (q38_mtp_best_token, draft_vocab.h), with no model.
 *
 * A reduced list never changes the score of a token it lists, and the head scores the whole
 * vocabulary when the list cannot be used:
 *  - the score of a listed token is the byte-for-byte score the whole lm_head gives it;
 *  - the draft is the best listed token, the lower id on a tie, and the best of the whole
 *    vocabulary when the list is all of it or when there is no list;
 *  - a Q38_MTP_DUMP record is the row, the token and the logits of the whole vocabulary;
 *  - the head gets a list only with the MTP head on, Q38_MTP_VOCAB above 0, no Q38_MTP_DUMP
 *    (which writes the logits of the whole vocabulary), no int4 copy of the lm_head
 *    (Q38_DRAFT_HEAD=int4), int8 rows of the lm_head on the host and a counts file that is not the
 *    expert history, however either path is spelled;
 *  - with the dense chain on, the device scores the whole vocabulary and the list is not read;
 *  - the counts file is written at the end of a run and read by the next one. It is not written
 *    under USAGE_SAVE=0 and when it cannot be read as counts. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

static int fails;
static void ck(int ok,const char *what){ if(ok){printf("  ok   %s\n",what);return;} printf("  FAIL %s\n",what); fails++; }

static unsigned g_seed=4242;
static float rnd(void){ g_seed=g_seed*1103515245u+12345u; return ((g_seed>>8)&0xFFFF)/32768.f-1.f; }
static uint16_t f32_to_bf16(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

enum { V=300, I=128 };

/* 1 when q38_vocab_decide gives a model with the MTP head `mtp` and the lm_head int8 rows `q8` a list;
 * the model directory is /nonexistent */
static int decides_list(int mtp,int8_t *q8){
    Model d; memset(&d,0,sizeof d);
    d.c.vocab=V; d.mtp=mtp; d.lm_head.q8=q8;
    q38_vocab_decide(&d,"/nonexistent");
    int listed=d.draft_vocab!=NULL;
    draft_vocab_free(d.draft_vocab);
    return listed;
}

static void write_text(const char *path,const char *text){ FILE *f=fopen(path,"w"); fputs(text,f); fclose(f); }

/* 1 when the file contains exactly `text` */
static int file_has(const char *path,const char *text){
    char got[128]="";
    FILE *f=fopen(path,"r");
    if(!f)return 0;
    size_t n=fread(got,1,sizeof got-1,f);
    fclose(f);
    got[n]=0;
    return !strcmp(got,text);
}

/* the list contains exactly `ids` in this order */
static void set_list(DraftVocab *v,const int *ids,int n){
    memset(v->in_list,0,(size_t)v->vocab);
    for(int j=0;j<n;j++){ v->list[j]=ids[j]; v->in_list[ids[j]]=1; }
    v->list_n=n;
}

int main(void){
    Model m; memset(&m,0,sizeof m);
    m.c.vocab=V; m.c.hidden=I;
    q38_weight_reserve(&m.lm_head,Q38_WEIGHT_BF16,V,I);
    uint16_t *rows=(uint16_t*)m.lm_head.data;
    for(int64_t k=0;k<(int64_t)V*I;k++)rows[k]=f32_to_bf16(rnd());
    q38_trunk_quantize(&m.lm_head,&m.lm_head.q8,&m.lm_head.q8sc);
    float x[I],logit[V];
    for(int k=0;k<I;k++)x[k]=rnd()*3.f;
    q38_weight_matmul(logit,x,&m.lm_head,1,I,V);
    int whole=q38_argmax(logit,V);

    printf("the score of a listed token\n");
    {
        int ids[]={290,3,150,77,0,299};
        float score[6];
        q38_head_scores(score,x,&m.lm_head,ids,6);
        int same=1;
        for(int j=0;j<6;j++)same&=!memcmp(&score[j],&logit[ids[j]],sizeof(float));
        ck(same,"equals the whole lm_head's score of that token, byte for byte");
    }

    printf("the Q38_MTP_DUMP record\n");
    {
        FILE *dump=fopen("test_qwen38_draft_head.tmp","wb+");
        ck(dump!=NULL,"opened the dump file");
        g_q38_mtp_dump=dump;
        q38_mtp_dump_logits(&m,x,(Q38MtpDumpRow){11,22});
        g_q38_mtp_dump=NULL;
        ck(ftell(dump)==(long)(2*sizeof(int32_t)+V*sizeof(float)),"the record is the row, the token and every logit");
        rewind(dump);
        int32_t got[2]; float back[V];
        ck(fread(got,sizeof got,1,dump)==1&&fread(back,sizeof(float),V,dump)==V&&
           got[0]==11&&got[1]==22&&!memcmp(back,logit,sizeof back),"the record has the row, the token and the logits");
        fclose(dump); remove("test_qwen38_draft_head.tmp");
    }

    printf("the draft\n");
    ck(q38_mtp_best_token(&m,x)==whole,"without a list: the best of the whole vocabulary");
    {
        DraftVocab *v=draft_vocab_new(V,V);
        m.draft_vocab=v;
        ck(v->list_n==V,"a list as long as the vocabulary has every token");
        ck(q38_mtp_best_token(&m,x)==whole,"a list of every token: the best of the whole vocabulary");
        draft_vocab_free(v);

        v=draft_vocab_new(V,40); m.draft_vocab=v;
        ck(v->list_n==40,"a list of 40 tokens");
        int best=v->list[0];
        for(int j=1;j<v->list_n;j++)if(logit[v->list[j]]>logit[best])best=v->list[j];
        ck(q38_mtp_best_token(&m,x)==best,"a short list: the best listed token");

        int others[]={(whole+1)%V,(whole+2)%V,(whole+3)%V,(whole+4)%V};   /* the overall best is not listed */
        int best_of_others=others[0];
        for(int j=1;j<4;j++)if(logit[others[j]]>logit[best_of_others])best_of_others=others[j];
        set_list(v,others,4);
        ck(q38_mtp_best_token(&m,x)==best_of_others,"a list without the overall best gives the best listed token");
#ifdef COLI_VULKAN
        g_vk_chain=1;
        ck(q38_mtp_best_token(&m,x)==whole,"with the dense chain on: the best of the whole vocabulary, the list not read");
        g_vk_chain=0;
#endif

        for(int tied=7;tied<=9;tied+=2){   /* rows 7 and 9 become copies of the best row: a tie at the top score */
            memmove(m.lm_head.q8+(int64_t)tied*I,m.lm_head.q8+(int64_t)whole*I,I);
            m.lm_head.q8sc[tied]=m.lm_head.q8sc[whole];
        }
        int low=-1;
        for(int id=0;id<V;id++)if(id!=7&&id!=9&&(low<0||logit[id]<logit[low]))low=id;
        ck(logit[low]<logit[whole],"a third token scores below the tied pair");
        struct { const char *what; int ids[3]; int n; } ties[]={
            {"the lower id last",{9,7},2},
            {"the lower id first",{7,9},2},
            {"the lower id last, a lower score after them",{9,7,low},3},
            {"the lower id first, a lower score after them",{7,9,low},3},
            {"the lower id last, a lower score before them",{low,9,7},3},
            {"the lower id first, a lower score before them",{low,7,9},3},
        };
        for(size_t k=0;k<sizeof ties/sizeof *ties;k++){
            set_list(v,ties[k].ids,ties[k].n);
            char what[128];
            snprintf(what,sizeof what,"a tie goes to the lower id: %s",ties[k].what);
            ck(q38_mtp_best_token(&m,x)==7,what);
        }
        draft_vocab_free(v); m.draft_vocab=NULL;
    }

#ifdef IDOT_R16
    printf("the rows interleaved by 16, as the engine keeps them on the CPU\n");
    {
        enum { V16=320 };   /* whole groups of 16 rows */
        Model r; memset(&r,0,sizeof r);
        r.c.vocab=V16; r.c.hidden=I;
        q38_weight_reserve(&r.lm_head,Q38_WEIGHT_BF16,V16,I);
        uint16_t *bf=(uint16_t*)r.lm_head.data;
        for(int64_t k=0;k<(int64_t)V16*I;k++)bf[k]=f32_to_bf16(rnd());
        q38_trunk_quantize(&r.lm_head,&r.lm_head.q8,&r.lm_head.q8sc);
        int8_t *packed=(int8_t*)malloc((size_t)V16*I);
        i8r16_pack(packed,r.lm_head.q8,V16,I);
        memcpy(r.lm_head.q8,packed,(size_t)V16*I);
        free(packed);
        r.lm_head.q8r16=1;
        float lo[V16];
        q38_weight_matmul(lo,x,&r.lm_head,1,I,V16);
        int top=q38_argmax(lo,V16);

        int ids[]={top,319,1,16,17,200};
        float score[6];
        q38_head_scores(score,x,&r.lm_head,ids,6);
        int same=1;
        for(int j=0;j<6;j++)same&=!memcmp(&score[j],&lo[ids[j]],sizeof(float));
        ck(same,"the score of a listed token equals the whole lm_head's score, byte for byte");

        DraftVocab *v=draft_vocab_new(V16,8);
        int listed[]={(top+5)%V16,top,(top+40)%V16,(top+100)%V16};
        set_list(v,listed,4);
        r.draft_vocab=v;
        ck(q38_mtp_best_token(&r,x)==top,"a list with the best token of the whole vocabulary drafts that token");
        draft_vocab_free(v);
    }
#endif

    printf("the decision\n");
    {
        unsetenv("Q38_MTP_VOCAB");
        unsetenv("COLI_USAGE");
        unsetenv("COLI_VOCAB_USAGE");
        ck(decides_list(1,m.lm_head.q8),"a list when the head has int8 rows on the host");
        ck(!decides_list(1,NULL),"no list without int8 rows on the host");
        ck(!decides_list(0,m.lm_head.q8),"no list without the MTP head");
        {
            Model d; memset(&d,0,sizeof d);
            uint8_t copy;
            d.c.vocab=V; d.mtp=1; d.lm_head.q8=m.lm_head.q8; d.draft_codes=&copy;
            q38_vocab_decide(&d,"/nonexistent");
            ck(!d.draft_vocab,"no list beside the int4 copy of the lm_head (Q38_DRAFT_HEAD=int4)");
            draft_vocab_free(d.draft_vocab);
        }
        setenv("Q38_MTP_VOCAB","0",1);
        ck(!decides_list(1,m.lm_head.q8),"no list for Q38_MTP_VOCAB=0");
        unsetenv("Q38_MTP_VOCAB");
        FILE *dump=fopen("test_qwen38_draft_head.tmp","wb+");
        ck(dump!=NULL,"opened the dump file");
        g_q38_mtp_dump=dump;
        ck(!decides_list(1,m.lm_head.q8),"no list under Q38_MTP_DUMP");
        g_q38_mtp_dump=NULL;
        fclose(dump); remove("test_qwen38_draft_head.tmp");
    }

    printf("the counts file\n");
    {
        const char *path="test_qwen38_draft_head_counts.tmp";
        setenv("COLI_VOCAB_USAGE",path,1);
        remove(path);
        Model a; memset(&a,0,sizeof a);
        a.c.vocab=V;
        a.draft_vocab=draft_vocab_new(V,10);
        q38_vocab_init("/nonexistent",&a);
        ck(a.draft_vocab!=NULL,"a missing counts file leaves the list on");

        const int prompt[]={5,6,7};
        draft_vocab_begin(a.draft_vocab,prompt,3,1);   /* token 5 came from the cache */

        unsetenv("USAGE_SAVE");
        q38_vocab_save(&a);
        Model b; memset(&b,0,sizeof b);
        b.c.vocab=V;
        b.draft_vocab=draft_vocab_new(V,10);
        q38_vocab_init("/nonexistent",&b);
        ck(b.draft_vocab->count[6]==1&&b.draft_vocab->count[7]==1,"the next run reads the counts the run wrote");

        remove(path);
        setenv("USAGE_SAVE","0",1);
        q38_vocab_save(&a);
        FILE *none=fopen(path,"rb");
        ck(none==NULL,"USAGE_SAVE=0 writes no counts file");
        if(none)fclose(none);
        unsetenv("USAGE_SAVE");

        draft_vocab_free(a.draft_vocab); draft_vocab_free(b.draft_vocab);
        remove(path);
        unsetenv("COLI_VOCAB_USAGE");
    }

    printf("a file at the counts path that the run did not write\n");
    {
        const char *path="test_qwen38_draft_head_foreign.tmp";
        setenv("COLI_VOCAB_USAGE",path,1);
        unsetenv("USAGE_SAVE");

        write_text(path,"not a counts file\n");
        Model a; memset(&a,0,sizeof a);
        a.c.vocab=V;
        a.draft_vocab=draft_vocab_new(V,10);
        q38_vocab_init("/nonexistent",&a);
        ck(a.draft_vocab!=NULL,"a file that is not counts leaves the list on");
        a.draft_vocab->count[5]=3;
        q38_vocab_save(&a);
        ck(file_has(path,"not a counts file\n"),"the end of the run leaves that file as it was");
        draft_vocab_free(a.draft_vocab);

        write_text(path,"the expert history\n");
        setenv("COLI_USAGE",path,1);
        ck(!decides_list(1,m.lm_head.q8),"no list when its counts file is the expert history");
        char spelled[64];
        snprintf(spelled,sizeof spelled,"./%s",path);
        setenv("COLI_VOCAB_USAGE",spelled,1);
        ck(!decides_list(1,m.lm_head.q8),"no list under another spelling of that file");
        remove(path);
        ck(!decides_list(1,m.lm_head.q8),"no list for a file that does not exist yet");
        setenv("COLI_USAGE","/nonexistent/.coli_vocab",1);
        unsetenv("COLI_VOCAB_USAGE");
        ck(!decides_list(1,m.lm_head.q8),"no list when the expert history is the default counts file");
        unsetenv("COLI_USAGE");

        remove(path);
        unsetenv("COLI_VOCAB_USAGE");
    }

    q38_weight_free(&m.lm_head);
    if(fails){ printf("test_qwen38_draft_head: %d failure(s)\n",fails); return 1; }
    printf("OK test_qwen38_draft_head: the draft from a list of tokens\n");
    return 0;
}
