/* Qwen3.8-Flash-Next text inference in dependency-free C.
 *
 * The language model is 12 repetitions of three Gated DeltaNet layers and one
 * Qwen Sparse Attention (QSA) layer.  Four Gated-Residual streams replace the
 * usual scalar residual; layer 2 injects pageable hashed n-gram embeddings
 * (PLE), and every layer has a 512-way sparse MoE plus a shared expert.
 *
 * Resident tensors are loaded from the official multimodal checkpoint's text
 * namespace.  Vision and MTP tensors are indexed but never read.  Official
 * block-FP8 experts remain native in a bounded per-layer LRU; the 51B-parameter
 * PLE table remains on disk and only its sixteen 160-byte rows/token are read.
 *
 * Runtime: SNAP=<dir>, optional Q38_MAXT, Q38_EXPERT_PREFETCH=0 and
 * Q38_EXPERT_PARALLEL_READS=0 A/B controls, and argv
 * `qwen38 <cache/layer> <reserved> [ref.json]`.  The second positional remains
 * reserved so Colibri's common cache-aware launcher ABI stays stable.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Hard context ceiling: the model's max_position_embeddings. Every buffer that
 * scales with position (KV cache, attention score row) is allocated from max_t,
 * so this is a policy limit, not a buffer limit -- but it is ONE limit, named
 * once. It used to be the literal 8192 in two unrelated places: the size of a
 * stack array in attention() and the default of Q38_MAXT in serve_one(). They
 * agreed by luck, and raising Q38_MAXT moved the guard without moving the
 * buffer, so a longer prompt overran the stack instead of being refused.
 * Context costs 54 KB/token in KV/index state (12 QSA layers, f32) -- 128k is 6.75 GiB
 * -- which is why Q38_MAXT still defaults far below this. */
#define QWEN38_ATTN_MAX_CTX 262144
#define QWEN38_DEFAULT_MAX_CTX 8192

/* Effective ceiling: Q38_MAXT if set and sane, the conservative default
 * otherwise; never above the hard limit. */
static int qwen38_max_ctx(void) {
    const char *e = getenv("Q38_MAXT");
    if (!e || !*e) return QWEN38_DEFAULT_MAX_CTX;
    char *end = NULL;
    errno = 0;
    long v = strtol(e, &end, 10);
    if (errno || end == e || *end || v < 1) return QWEN38_DEFAULT_MAX_CTX;
    return v > QWEN38_ATTN_MAX_CTX ? QWEN38_ATTN_MAX_CTX : (int)v;
}
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "st.h"
#include "json.h"   /* tokenizer.json parsing (reuse minimal parser) */
#include "tok_unicode.h"
#include "tok_unicode_o200k.h"
#include "qwen38_nfc.h"
/* quant.h owns Colibri's authoritative E4M3 table.  This engine retains the
 * Qwen sibling's established dense kernels, so rename quant.h's two generic
 * helpers while including it; every format-specific symbol keeps its shared
 * name and implementation. */
#define matmul coli_quant_matmul
#define matmul_q coli_quant_matmul_q
#include "quant.h"
#undef matmul
#undef matmul_q
#include "route_trace.h"             /* shared ROUTE_TRACE + .coli_usage contract */
#include "serve_codec.h"             /* shared SUBMIT/STOP/CANCEL framing */
/* ---------- tokenizer (optional, for human-readable output) ---------- */
static char **g_tok = NULL;   /* id -> piece string (strdup'd) */
static int    g_tok_n = 0;

static int hexnib(char c){
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return 0;
}

/* ===== text -> ids : BPE encoder (mirrors HF/Qwen tokenizer.json) =====
 * Builds piece->id (reverse vocab) + pair->rank (merges) maps, plus the
 * GPT-2 byte-to-unicode mapping. Encode = special-token split + GPT-2 regex
 * pre-tokenize + per-piece ByteLevel map + BPE merges. */
typedef struct { char **keys; int *vals; uint8_t *used; size_t cap; } SMap;
static unsigned shash(const char *s){ unsigned h=2166136261u; while(*s){ h^=(unsigned char)*s++; h*=16777619u; } return h; }
static int smap_init(SMap *m,size_t cap){
    if(!m||cap<2||(cap&(cap-1))||cap>SIZE_MAX/sizeof(char*)||
       cap>SIZE_MAX/sizeof(int))return -1;
    memset(m,0,sizeof(*m));m->cap=cap;
    m->keys=(char**)calloc(cap,sizeof(char*));
    m->vals=(int*)malloc(cap*sizeof(int));
    m->used=(uint8_t*)calloc(cap,sizeof(uint8_t));
    if(!m->keys||!m->vals||!m->used){free(m->keys);free(m->vals);free(m->used);memset(m,0,sizeof(*m));return -1;}
    return 0;
}
static int smap_init_entries(SMap *m,size_t entries){
    if(entries>SIZE_MAX/2u)return -1;
    size_t need=entries*2u,cap=2u;
    while(cap<need){if(cap>SIZE_MAX/2u)return -1;cap*=2u;}
    return smap_init(m,cap);
}
static int smap_put(SMap *m,const char *k,int v){
    if(!m||!m->cap||!k)return -1;
    size_t h=(size_t)shash(k)&(m->cap-1u);
    for(size_t probes=0;probes<m->cap;probes++,h=(h+1u)&(m->cap-1u)){
        if(!m->used[h]){m->used[h]=1;m->keys[h]=(char*)k;m->vals[h]=v;return 0;}
        if(m->keys[h]&&!strcmp(m->keys[h],k)){m->vals[h]=v;return 0;}
    }
    return -1;
}
static int smap_get(const SMap *m,const char *k){
    if(!m||!m->cap||!k)return -1;
    size_t h=(size_t)shash(k)&(m->cap-1u);
    for(size_t probes=0;probes<m->cap&&m->used[h];probes++,h=(h+1u)&(m->cap-1u))
        if(m->keys[h]&&!strcmp(m->keys[h],k))return m->vals[h];
    return -1;
}

static SMap  g_rev;                 /* piece string -> id (encode) */
static SMap  g_merge;               /* "a\x1F b" pair -> rank (encode) */
static char  byte_sym_utf8[256][8]; /* byte -> UTF-8 of mapped codepoint */
static short g_unmap[512];          /* mapped codepoint -> original byte (-1 = unused) */
static int   g_nspecial = 0;
static char **g_sp_str = NULL; static int *g_sp_id = NULL; static int *g_sp_len = NULL;
static int g_tok_nfc = 0;

static void free_tokenizer(void){
    for(int i=0;i<g_tok_n;i++) free(g_tok ? g_tok[i] : NULL);
    free(g_tok); g_tok=NULL; g_tok_n=0;
    free(g_rev.keys); free(g_rev.vals); free(g_rev.used); memset(&g_rev,0,sizeof g_rev);
    if(g_merge.keys) for(size_t i=0;i<g_merge.cap;i++)
        if(g_merge.used && g_merge.used[i]) free(g_merge.keys[i]);
    free(g_merge.keys); free(g_merge.vals); free(g_merge.used); memset(&g_merge,0,sizeof g_merge);
    for(int i=0;i<g_nspecial;i++) free(g_sp_str ? g_sp_str[i] : NULL);
    free(g_sp_str); free(g_sp_id); free(g_sp_len);
    g_sp_str=NULL; g_sp_id=NULL; g_sp_len=NULL; g_nspecial=0; g_tok_nfc=0;
}

static const char *jstr(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_STR)?v->str:NULL; }
static double jnum(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_NUM)?v->num:0; }

enum { U_W=0, U_L=1, U_M=2, U_N=3, U_P=4, U_O=5 };
static int uclass(unsigned cp){
    if(is_S(cp))return U_W;
    if(is_L(cp))return U_L;
    if(is_X(cp))return U_M; /* uni_X minus uni_L is exactly Unicode Mark. */
    if(is_N(cp))return U_N;
    return U_P;             /* every remaining category is regex punctuation. */
}
static int utf8_decode(const char *s,int i,int n,int *adv){
    if(!s||i<0||i>=n){if(adv)*adv=0;return 0xfffd;}
    unsigned char c=(unsigned char)s[i]; int cp,a;
    if(c<0x80){cp=c;a=1;}
    else if((c>>5)==6){cp=c&0x1F;a=2;}
    else if((c>>4)==14){cp=c&0x0F;a=3;}
    else if((c>>3)==30){cp=c&0x07;a=4;}
    else {cp=c;a=1;}
    for(int k=1;k<a;k++){
        if(i+k>=n||((unsigned char)s[i+k]&0xC0)!=0x80){if(adv)*adv=1;return c;}
        cp=(cp<<6)|((unsigned char)s[i+k]&0x3F);
    }
    if(adv)*adv=a; return cp;
}
static int utf8_adv(const char *s,int i,int n){ int a; utf8_decode(s,i,n,&a); return a?a:1; }

static void build_byte_sym(void){
    for(int i=0;i<512;i++) g_unmap[i]=-1;
    int bs[256]; for(int b=0;b<256;b++) bs[b]=0;
    for(int b=33;b<=126;b++) bs[b]=1;
    for(int b=161;b<=172;b++) bs[b]=1;
    for(int b=174;b<=255;b++) bs[b]=1;
    int cn=0;
    for(int b=0;b<256;b++){
        int cp = bs[b]?b:(256+cn); if(!bs[b]) cn++;
        int k=0; unsigned c=(unsigned)cp;
        if(c<0x80) byte_sym_utf8[b][k++]=(char)c;
        else if(c<0x800){ byte_sym_utf8[b][k++]=0xC0|(c>>6); byte_sym_utf8[b][k++]=0x80|(c&0x3F); }
        else { byte_sym_utf8[b][k++]=0xE0|(c>>12); byte_sym_utf8[b][k++]=0x80|((c>>6)&0x3F); byte_sym_utf8[b][k++]=0x80|(c&0x3F); }
        byte_sym_utf8[b][k]=0;
        g_unmap[cp]=(short)b;   /* reverse: mapped codepoint -> original byte */
    }
}
static void q38_encode_oom(void){fprintf(stderr,"[enc] out of memory\n");exit(1);}
static void *q38_encode_malloc(size_t bytes){
    void *result=malloc(bytes);if(!result)q38_encode_oom();return result;
}
static void *q38_encode_realloc(void *allocation,size_t bytes){
    void *result=realloc(allocation,bytes);if(!result)q38_encode_oom();return result;
}
static void push_id(int **ids,int *n,int *cap,int v){
    if(!ids||!n||!cap||*n<0||*cap<1||*n>*cap){
        fprintf(stderr,"[enc] invalid token buffer state\n");exit(1);
    }
    if(*n==*cap){
        if(*cap>INT_MAX/2||(size_t)(*cap*2)>SIZE_MAX/sizeof(int))q38_encode_oom();
        *cap*=2;*ids=(int*)q38_encode_realloc(*ids,(size_t)*cap*sizeof(int));
    }
    (*ids)[(*n)++]=v;
}

static int try_special(const char *s,int i,int n,int *id_out){
    int best_len=0,best_id=-1;
    for(int k=0;k<g_nspecial;k++){
        int L=g_sp_len[k]; if(L<=0||i+L>n) continue;
        if(memcmp(s+i,g_sp_str[k],L)==0){ if(L>best_len){best_len=L;best_id=g_sp_id[k];} }
    }
    *id_out=best_id; return best_len;
}
/* Pre-tokenize splitter, mirrors the HF/Qwen regex alternation:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
 *   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * Returns the byte index just past the piece starting at i. */
static int pretok_end(const char *s,int i,int n){
    if (s[i]=='\''){
        const char *cands[]={"ll","ve","re","s","t","m","d"}; int clen[]={2,2,2,1,1,1,1};
        int best=0;
        for(int c=0;c<7;c++){ int L=clen[c]; if(i+1+L>n) continue; int ok=1; for(int k=0;k<L;k++){ char a=(char)tolower((unsigned char)s[i+1+k]); if(a!=cands[c][k]){ok=0;break;} } if(ok&&L>best)best=L; }
        if(best>0) return i+1+best;
    }
    int adv; unsigned c0=utf8_decode(s,i,n,&adv);
    { /* rule2: optional non-(cr/lf/letter/number) prefix then letter/mark run */
        int k=i; unsigned c=c0; int prefix=0;
        if(k<n && c!='\r'&&c!='\n'&&uclass(c)!=U_L&&uclass(c)!=U_N&&k+adv<n){
            int a2; unsigned c1=utf8_decode(s,k+adv,n,&a2);
            if(uclass(c1)==U_L||uclass(c1)==U_M){ prefix=1; k+=adv; }
        }
        if(prefix || uclass(c)==U_L || uclass(c)==U_M){
            while(k<n){ int a; unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)==U_L||uclass(cc)==U_M) k+=a; else break; }
            return k;
        }
    }
    if(uclass(c0)==U_N) return i+adv;
    { /* rule4: optional space + punctuation run (+ trailing newlines) */
        int k=i;
        if(s[i]==' '&&i+1<n){ int a1; unsigned c1=utf8_decode(s,i+1,n,&a1); if(uclass(c1)!=U_W&&uclass(c1)!=U_L&&uclass(c1)!=U_N&&c1!='\r'&&c1!='\n'){ k=i+1; while(k<n){int a;unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)!=U_W&&uclass(cc)!=U_L&&uclass(cc)!=U_N&&cc!='\r'&&cc!='\n')k+=a; else break;} while(k<n&&(s[k]=='\r'||s[k]=='\n'))k++; return k; } }
        if(uclass(c0)!=U_W&&uclass(c0)!=U_L&&uclass(c0)!=U_N&&c0!='\r'&&c0!='\n'){ int k2=i; while(k2<n){int a;unsigned cc=utf8_decode(s,k2,n,&a); if(uclass(cc)!=U_W&&uclass(cc)!=U_L&&uclass(cc)!=U_N&&cc!='\r'&&cc!='\n')k2+=a; else break;} while(k2<n&&(s[k2]=='\r'||s[k2]=='\n'))k2++; return k2; }
    }
    if(uclass(c0)==U_W){
        int k=i,last=i,last_newline_end=-1,count=0;
        while(k<n){
            int a;unsigned cc=utf8_decode(s,k,n,&a);if(uclass(cc)!=U_W)break;
            last=k;if(cc=='\r'||cc=='\n')last_newline_end=k+a;k+=a;count++;
        }
        /* `\s*[\r\n]+` ends at the final newline, leaving any following
         * indentation for the next regex match.  Without a newline,
         * `\s+(?!\S)` leaves the final whitespace character available as the
         * optional prefix of the following word/punctuation match. */
        if(last_newline_end>=0)return last_newline_end;
        if(k<n&&count>1)return last;
        return k;
    }
    return i+adv;
}
#ifdef QWEN38_TEST_TOKENIZER
static uint64_t g_bpe_pair_checks;
#define Q38_BPE_PAIR_CHECK() (g_bpe_pair_checks++)
#else
#define Q38_BPE_PAIR_CHECK() ((void)0)
#endif

static int bpe_pair_rank(const char *left,const char *right){
    size_t left_length=strlen(left),right_length=strlen(right);
    if(right_length>SIZE_MAX-2u||left_length>SIZE_MAX-right_length-2u)
        q38_encode_oom();
    size_t key_length=left_length+right_length+2u;
    char *key=(char*)q38_encode_malloc(key_length);
    memcpy(key,left,left_length);key[left_length]=0x1f;
    memcpy(key+left_length+1u,right,right_length+1u);
    Q38_BPE_PAIR_CHECK();int rank=smap_get(&g_merge,key);free(key);return rank;
}

static char *bpe_join(const char *left,const char *right){
    size_t left_length=strlen(left),right_length=strlen(right);
    if(right_length>SIZE_MAX-1u||left_length>SIZE_MAX-right_length-1u)
        q38_encode_oom();
    char *joined=(char*)q38_encode_malloc(left_length+right_length+1u);
    memcpy(joined,left,left_length);memcpy(joined+left_length,right,right_length+1u);
    return joined;
}

static void bpe_piece(const char *piece,int len,int **ids,int *n,int *cap){
    if(len<=0)return;
    int sc=0,scap=16;
    char **syms=(char**)q38_encode_malloc((size_t)scap*sizeof(char*));
    for(int b=0;b<len;b++){
        const char *sym=byte_sym_utf8[(unsigned char)piece[b]];
        size_t sl=strlen(sym);char *d=(char*)q38_encode_malloc(sl+1u);
        memcpy(d,sym,sl+1u);
        if(sc==scap){
            if(scap>INT_MAX/2||(size_t)(scap*2)>SIZE_MAX/sizeof(char*))q38_encode_oom();
            scap*=2;syms=(char**)q38_encode_realloc(
                syms,(size_t)scap*sizeof(char*));
        }
        syms[sc++]=d;
    }
    while(sc>1){
        int best=-1;
        for(int k=0;k<sc-1;k++){
            int rank=bpe_pair_rank(syms[k],syms[k+1]);
            if(rank>=0&&(best<0||rank<best))best=rank;
        }
        if(best<0)break;
        /* GPT-2/Qwen BPE merges every non-overlapping occurrence of the
         * selected pair before ranks are recomputed. Besides matching the
         * reference algorithm, this keeps a long run of rank-zero pairs from
         * rescanning almost the entire request once per merge. */
        int read=0,write=0;
        while(read<sc){
            if(read+1<sc&&bpe_pair_rank(syms[read],syms[read+1])==best){
                char *joined=bpe_join(syms[read],syms[read+1]);
                free(syms[read]);free(syms[read+1]);syms[write++]=joined;read+=2;
            }else syms[write++]=syms[read++];
        }
        sc=write;
    }
    for(int k=0;k<sc;k++){ int id=smap_get(&g_rev,syms[k]); if(id<0) id=0; push_id(ids,n,cap,id); free(syms[k]); }
    free(syms);
}
static int next_special(const char *text,int from,int length){
    for(int at=from;at<length;at++){
        int unused; if(try_special(text,at,length,&unused)>0)return at;
    }
    return length;
}

static void encode_ordinary(const char *text,int length,
                            int **ids,int *count,int *capacity){
    for(int at=0;at<length;){
        int end=pretok_end(text,at,length);
        if(end<=at)end=at+utf8_adv(text,at,length);
        if(end>length)end=length;
        bpe_piece(text+at,end-at,ids,count,capacity);
        at=end;
    }
}

static void encode_text_n(const char *text,size_t text_length,
                          int **out_ids,int *out_n){
    if(!out_ids||!out_n||(text_length&&!text)||text_length>(size_t)INT_MAX){
        fprintf(stderr,"[enc] invalid or oversized input\n");exit(1);
    }
    int cap=1024,n=0;int *ids=(int*)malloc((size_t)cap*sizeof(int));
    if(!ids){fprintf(stderr,"[enc] out of memory\n");exit(1);}
    int length=(int)text_length;
    for(int at=0;at<length;){
        int sid,L=try_special(text,at,length,&sid);
        if(L>0){push_id(&ids,&n,&cap,sid);at+=L;continue;}
        /* Added tokens have normalized=false in the official tokenizer. Match
         * them against the original bytes, then normalize only the ordinary
         * span before regex pre-tokenization. */
        int limit=next_special(text,at+1,length);
        char *normalized=NULL;size_t normalized_length=0;
        if(g_tok_nfc){
            if(q38_nfc_normalize(text+at,(size_t)(limit-at),
                                 &normalized,&normalized_length)||
               normalized_length>(size_t)INT_MAX){
                free(normalized);free(ids);fprintf(stderr,"[enc] NFC normalization failed\n");exit(1);
            }
        }else{
            normalized=(char*)malloc((size_t)(limit-at)+1u);
            if(!normalized){free(ids);fprintf(stderr,"[enc] out of memory\n");exit(1);}
            memcpy(normalized,text+at,(size_t)(limit-at));
            normalized_length=(size_t)(limit-at);normalized[normalized_length]='\0';
        }
        encode_ordinary(normalized,(int)normalized_length,&ids,&n,&cap);
        free(normalized);at=limit;
    }
    *out_ids=ids;*out_n=n;
}

static void encode_text(const char *text,int **out_ids,int *out_n){
    encode_text_n(text,strlen(text),out_ids,out_n);
}

/* Load Qwen tokenizer.json and build the id/piece and BPE merge tables. Every
 * post-publication error clears the complete global state, and callers receive
 * an explicit status instead of guessing success from a partially filled
 * pointer table. */
static int load_tokenizer(const char *path){
    FILE *file=NULL;char *buffer=NULL,*arena=NULL;jval *root=NULL;
    int result=-1;
    free_tokenizer();
    file=fopen(path,"rb");
    if(!file){fprintf(stderr,"[tok] cannot open %s\n",path);goto cleanup;}
    if(fseek(file,0,SEEK_END)){fprintf(stderr,"[tok] cannot seek %s\n",path);goto cleanup;}
    long file_size=ftell(file);
    if(file_size<0||file_size>(256L<<20)||fseek(file,0,SEEK_SET)){
        fprintf(stderr,"[tok] invalid or oversized tokenizer %s\n",path);goto cleanup;
    }
    buffer=(char*)malloc((size_t)file_size+1u);
    if(!buffer||fread(buffer,1,(size_t)file_size,file)!=(size_t)file_size){
        fprintf(stderr,"[tok] cannot read %s\n",path);goto cleanup;
    }
    buffer[file_size]='\0';root=json_parse(buffer,&arena);
    if(!root||root->t!=J_OBJ){
        fprintf(stderr,"[tok] invalid tokenizer JSON in %s\n",path);goto cleanup;
    }
    jval *model=json_get(root,"model");if(!model)model=root;
    if(model->t!=J_OBJ){
        fprintf(stderr,"[tok] invalid model object in %s\n",path);goto cleanup;
    }
    jval *vocab=json_get(model,"vocab");if(!vocab)vocab=json_get(model,"tokens");
    if(!vocab||(vocab->t!=J_OBJ&&vocab->t!=J_ARR)||vocab->len<1){
        fprintf(stderr,"[tok] no valid model.vocab/tokens in %s\n",path);goto cleanup;
    }
    int max_id=0;
    if(vocab->t==J_OBJ){
        for(int index=0;index<vocab->len;index++){
            jval *value=vocab->kids[index];
            if(!value||value->t!=J_NUM||!isfinite(value->num)||value->num<0||
               value->num>10000000||floor(value->num)!=value->num){
                fprintf(stderr,"[tok] invalid vocabulary id in %s\n",path);goto cleanup;
            }
            int id=(int)value->num;if(id>max_id)max_id=id;
        }
    }else{
        if(vocab->len>10000001){
            fprintf(stderr,"[tok] vocabulary is too large in %s\n",path);goto cleanup;
        }
        max_id=vocab->len-1;
    }
    g_tok=(char**)calloc((size_t)max_id+1u,sizeof(char*));g_tok_n=max_id+1;
    if(!g_tok){fprintf(stderr,"[tok] OOM loading %s\n",path);goto cleanup;}
    if(vocab->t==J_OBJ){
        for(int index=0;index<vocab->len;index++){
            int id=(int)vocab->kids[index]->num;
            if(g_tok[id]){
                fprintf(stderr,"[tok] duplicate vocabulary id in %s\n",path);goto cleanup;
            }
            g_tok[id]=strdup(vocab->keys[index]);
            if(!g_tok[id]){fprintf(stderr,"[tok] OOM loading %s\n",path);goto cleanup;}
        }
    }else{
        for(int index=0;index<vocab->len;index++){
            jval *value=vocab->kids[index];
            if(!value||value->t!=J_STR){
                fprintf(stderr,"[tok] invalid vocabulary token in %s\n",path);goto cleanup;
            }
            g_tok[index]=strdup(value->str);
            if(!g_tok[index]){fprintf(stderr,"[tok] OOM loading %s\n",path);goto cleanup;}
        }
    }

    jval *normalizer=json_get(root,"normalizer");
    if(normalizer&&normalizer->t!=J_NULL){
        const char *kind=normalizer->t==J_OBJ?jstr(normalizer,"type"):NULL;
        if(!kind||strcmp(kind,"NFC")){
            fprintf(stderr,"[tok] unsupported tokenizer normalizer in %s\n",path);goto cleanup;
        }
        g_tok_nfc=1;
    }
    if(smap_init_entries(&g_rev,(size_t)vocab->len)){
        fprintf(stderr,"[tok] OOM sizing vocabulary map for %s\n",path);goto cleanup;
    }
    for(int id=0;id<g_tok_n;id++)if(g_tok[id]&&smap_put(&g_rev,g_tok[id],id)){
        fprintf(stderr,"[tok] vocabulary map is full in %s\n",path);goto cleanup;
    }

    jval *merges=json_get(model,"merges");
    if(merges&&merges->t!=J_ARR){
        fprintf(stderr,"[tok] invalid merge list in %s\n",path);goto cleanup;
    }
    size_t merge_entries=merges?(size_t)merges->len:0u;
    if(smap_init_entries(&g_merge,merge_entries)){
        fprintf(stderr,"[tok] OOM sizing merge map for %s\n",path);goto cleanup;
    }
    if(merges)for(int rank=0;rank<merges->len;rank++){
        jval *entry=merges->kids[rank];
        const char *encoded=entry&&entry->t==J_STR?entry->str:NULL;
        const char *space=encoded?strchr(encoded,' '):NULL;
        if(!space||space==encoded||!space[1]){
            fprintf(stderr,"[tok] invalid merge entry in %s\n",path);goto cleanup;
        }
        size_t left_length=(size_t)(space-encoded),right_length=strlen(space+1);
        if(right_length>SIZE_MAX-2u||left_length>SIZE_MAX-right_length-2u){
            fprintf(stderr,"[tok] oversized merge entry in %s\n",path);goto cleanup;
        }
        size_t key_length=left_length+right_length+2u;
        char *key=(char*)malloc(key_length);
        if(!key){fprintf(stderr,"[tok] OOM loading merges from %s\n",path);goto cleanup;}
        memcpy(key,encoded,left_length);key[left_length]=0x1f;
        memcpy(key+left_length+1u,space+1,right_length+1u);
        if(smap_get(&g_merge,key)>=0){free(key);continue;} /* first rank wins */
        if(smap_put(&g_merge,key,rank)){
            free(key);fprintf(stderr,"[tok] merge map is full in %s\n",path);goto cleanup;
        }
    }

    jval *added=json_get(root,"added_tokens");
    if(added&&added->t!=J_ARR){
        fprintf(stderr,"[tok] invalid added-token list in %s\n",path);goto cleanup;
    }
    if(added){
        int added_max=g_tok_n-1;
        for(int index=0;index<added->len;index++){
            jval *token=added->kids[index];
            jval *id_value=token&&token->t==J_OBJ?json_get(token,"id"):NULL;
            const char *content=token&&token->t==J_OBJ?jstr(token,"content"):NULL;
            if(!id_value||id_value->t!=J_NUM||!isfinite(id_value->num)||
               id_value->num<0||id_value->num>10000000||
               floor(id_value->num)!=id_value->num||!content||
               strlen(content)>(size_t)INT_MAX){
                fprintf(stderr,"[tok] invalid added token in %s\n",path);goto cleanup;
            }
            if((int)id_value->num>added_max)added_max=(int)id_value->num;
        }
        if(added_max>=g_tok_n){
            char **grown=(char**)realloc(
                g_tok,((size_t)added_max+1u)*sizeof(char*));
            if(!grown){
                fprintf(stderr,"[tok] OOM loading added tokens from %s\n",path);goto cleanup;
            }
            memset(grown+g_tok_n,0,(size_t)(added_max+1-g_tok_n)*sizeof(char*));
            g_tok=grown;g_tok_n=added_max+1;
        }
        g_nspecial=added->len;
        if(g_nspecial&&((size_t)g_nspecial>SIZE_MAX/sizeof(char*)||
                         (size_t)g_nspecial>SIZE_MAX/sizeof(int))){
            fprintf(stderr,"[tok] too many added tokens in %s\n",path);goto cleanup;
        }
        g_sp_str=(char**)calloc((size_t)g_nspecial,sizeof(char*));
        g_sp_id=(int*)malloc((size_t)g_nspecial*sizeof(int));
        g_sp_len=(int*)malloc((size_t)g_nspecial*sizeof(int));
        if(g_nspecial&&(!g_sp_str||!g_sp_id||!g_sp_len)){
            fprintf(stderr,"[tok] OOM loading added tokens from %s\n",path);goto cleanup;
        }
        for(int index=0;index<added->len;index++){
            jval *token=added->kids[index];const char *content=jstr(token,"content");
            g_sp_str[index]=strdup(content);g_sp_id[index]=(int)jnum(token,"id");
            if(!g_sp_str[index]){
                fprintf(stderr,"[tok] OOM loading added tokens from %s\n",path);goto cleanup;
            }
            g_sp_len[index]=(int)strlen(g_sp_str[index]);
            if(g_tok[g_sp_id[index]]&&strcmp(g_tok[g_sp_id[index]],content)){
                fprintf(stderr,"[tok] added token id conflicts with vocabulary in %s\n",
                        path);goto cleanup;
            }else if(!g_tok[g_sp_id[index]]){
                g_tok[g_sp_id[index]]=strdup(content);
                if(!g_tok[g_sp_id[index]]){
                    fprintf(stderr,"[tok] OOM loading added tokens from %s\n",path);goto cleanup;
                }
            }
        }
    }
    build_byte_sym();result=0;
    fprintf(stderr,"[tok] loaded %d base pieces + %d added tokens (max id %d) from %s\n",
            vocab->len,g_nspecial,g_tok_n-1,path);
cleanup:
    if(file)fclose(file);json_free(root);free(buffer);free(arena);
    if(result)free_tokenizer();
    return result;
}

/* Decode token ids to text using g_tok, writing to stdout. Handles Qwen's
 * byte-representation markers (Ġ=space, Ċ=newline, ▁=space) and <0xXX> byte
 * fallback. Only active when a tokenizer was loaded. */
/* ---- streaming / incremental decode support ---- */
static int    g_stream = 0;            /* 1 = emit tokens as they are generated */
static unsigned char g_sbuf[16];       /* carries a partial UTF-8 char across tokens */
static int    g_sbn = 0;

/* ---- OpenAI-compatible output + timing ---- */
static int    g_openai = 0;            /* 1 = emit OpenAI Chat Completions format (SSE/JSON) */
static double g_gen_t0 = 0;            /* generate() start (monotonic seconds) */
static double g_ttft   = -1;           /* time to first token (s); -1 = unset */
static long   g_oa_created = 0;        /* unix timestamp for OpenAI "created" */
static char   g_oa_id[64];             /* OpenAI-style id, e.g. chatcmpl-... */
static char   g_model[128] = "qwen3.8-flash-next-colibri";
static double now_s(void);   /* forward decl; defined later near model code */
static int q38_completion_nonce(double monotonic_seconds){
    if(!isfinite(monotonic_seconds)||monotonic_seconds<0.0)return 0;
    double within_ten=fmod(monotonic_seconds,10.0);
    return (int)(within_ten*1000.0)%10000;
}
/* Decode throughput is reciprocal TPOT: prefill produces token one, so N
 * completion tokens contain exactly N-1 measured decode intervals. */
static double q38_decode_rate(int generated,double decode_seconds){
    return generated>1&&isfinite(decode_seconds)&&decode_seconds>1e-6?
           (double)(generated-1)/decode_seconds:0.0;
}

/* Output sink for server mode: when g_sock_out >= 0, SSE/JSON bytes are routed
 * to the live socket via g_sock_send instead of stdout. Lets qwen38_serve.c
 * reuse all emit logic without any change to the CLI path. */
static long long g_sock_out = -1;
static void (*g_sock_send)(long long fd, const char *buf, int n) = NULL;
static char g_q38_usage[2100];        /* <snap>/.coli_usage, or COLI_USAGE */

/* JSON-escape a byte string into out (no surrounding quotes). Returns length,
 * or -1 without emitting a partial string when the destination is too small. */
static int json_escape(const unsigned char *s, int n, char *out, int outsz){
    if(n<0||outsz<1||!out||(n&& !s))return -1;
    int o = 0;
    for (int i=0;i<n;i++){
        unsigned char c = s[i];
        int need=c<0x20?((c=='\n'||c=='\r'||c=='\t'||c=='\b'||c=='\f')?2:6):
                 ((c=='"'||c=='\\')?2:1);
        if(o>outsz-1-need)return -1;
        if(c=='"'||c=='\\'){out[o++]='\\';out[o++]=(char)c;}
        else if(c=='\n'){out[o++]='\\';out[o++]='n';}
        else if(c=='\r'){out[o++]='\\';out[o++]='r';}
        else if(c=='\t'){out[o++]='\\';out[o++]='t';}
        else if(c=='\b'){out[o++]='\\';out[o++]='b';}
        else if(c=='\f'){out[o++]='\\';out[o++]='f';}
        else if(c<0x20){snprintf(out+o,(size_t)(outsz-o),"\\u%04x",c);o+=6;}
        else out[o++]=(char)c;
    }
    out[o] = 0;
    return o;
}

static char *json_escape_alloc(const unsigned char *s,size_t n){
    if(n>(SIZE_MAX-1u)/6u||n>((size_t)INT_MAX-1u)/6u)return NULL;
    size_t cap=n*6u+1u;char *out=(char*)malloc(cap);
    if(!out||json_escape(s,(int)n,out,(int)cap)<0){free(out);return NULL;}
    return out;
}

static char *q38_format_alloc(const char *format,...){
    va_list args,copy;va_start(args,format);va_copy(copy,args);
    int needed=vsnprintf(NULL,0,format,args);va_end(args);
    if(needed<0||(size_t)needed==SIZE_MAX){va_end(copy);return NULL;}
    char *out=(char*)malloc((size_t)needed+1u);
    if(!out||vsnprintf(out,(size_t)needed+1u,format,copy)!=needed){free(out);out=NULL;}
    va_end(copy);return out;
}

static int utf8_emit(const unsigned char *bytes,int count,
                     unsigned char *out,int outcap,int *outn){
    if(count<0||*outn<0||*outn>outcap||count>outcap-*outn)return -1;
    if(count)memcpy(out+*outn,bytes,(size_t)count);*outn+=count;return 0;
}

/* Incrementally validate decoded token bytes as UTF-8. Hugging Face's
 * ByteLevel decoder uses replacement semantics for arbitrary byte tokens, so
 * malformed maximal subparts become U+FFFD instead of leaking invalid bytes
 * into JSON/SSE. A valid scalar may remain split across token boundaries. */
static int utf8_drain(unsigned char *buf,int *bn,const unsigned char *b,int n,
                      unsigned char *out,int outcap,int *outn){
    if(!buf||!bn||*bn<0||*bn>3||n<0||(n&&!b)||outcap<0||(outcap&&!out)||!outn)return -1;
    static const unsigned char replacement[3]={0xef,0xbf,0xbd};*outn=0;
    for(int k=0;k<n;k++){
        unsigned char current=b[k];int pending=1;
        while(pending){
            if(!*bn){
                if(current<0x80){if(utf8_emit(&current,1,out,outcap,outn))return -1;}
                else if((current>=0xc2&&current<=0xdf)||
                        (current>=0xe0&&current<=0xef)||
                        (current>=0xf0&&current<=0xf4))buf[(*bn)++]=current;
                else if(utf8_emit(replacement,3,out,outcap,outn))return -1;
                pending=0;continue;
            }
            unsigned char lead=buf[0];int position=*bn;
            int continuation=current>=0x80&&current<=0xbf;
            if(position==1){
                if(lead==0xe0)continuation=current>=0xa0&&current<=0xbf;
                else if(lead==0xed)continuation=current>=0x80&&current<=0x9f;
                else if(lead==0xf0)continuation=current>=0x90&&current<=0xbf;
                else if(lead==0xf4)continuation=current>=0x80&&current<=0x8f;
            }
            if(!continuation){
                if(utf8_emit(replacement,3,out,outcap,outn))return -1;
                *bn=0;continue; /* reprocess current as a new leading byte */
            }
            buf[(*bn)++]=current;pending=0;
            int need=lead<=0xdf?2:lead<=0xef?3:4;
            if(*bn==need){
                if(utf8_emit(buf,need,out,outcap,outn))return -1;*bn=0;
            }
        }
    }
    return *outn;
}

static int utf8_finish(unsigned char *buf,int *bn,
                       unsigned char *out,int outcap,int *outn){
    static const unsigned char replacement[3]={0xef,0xbf,0xbd};
    if(!buf||!bn||*bn<0||*bn>3||!outn)return -1;*outn=0;
    if(*bn&&utf8_emit(replacement,3,out,outcap,outn))return -1;
    *bn=0;return *outn;
}

/* Emit one Server-Sent-Event chunk (OpenAI streaming uses `data: <json>` lines). */
static void sse_chunk(const char *json){
    char hdr[8]; int hl = snprintf(hdr, sizeof hdr, "data: ");
    if (g_sock_out >= 0 && g_sock_send){
        g_sock_send(g_sock_out, hdr, hl);
        g_sock_send(g_sock_out, json, (int)strlen(json));
        g_sock_send(g_sock_out, "\n\n", 2);
    } else {
        fwrite(hdr, 1, (size_t)hl, stdout);
        fwrite(json, 1, (size_t)strlen(json), stdout);
        fwrite("\n\n", 1, 2, stdout);
        fflush(stdout);
    }
}

/* Decode a single token id into its raw (unmapped) bytes.
 * The vocab stores byte-level BPE pieces: each piece is UTF-8 of the
 * GPT-2 byte_to_unicode-mapped codepoints. We reverse that mapping so the
 * output is the original text bytes (correct for CJK / non-ASCII too).
 * <0xXX> byte-fallback tokens emit the raw byte directly. */
static int decode_id_to_bytes(int id,unsigned char *out,size_t outcap){
    size_t outn=0;
    if(!g_tok||id<0||id>=g_tok_n||!g_tok[id])return 0;
    const unsigned char *pc = (const unsigned char*)g_tok[id];
    /* byte-fallback token: <0xXX> -> raw byte */
    if (strlen((const char *)pc)==6 && pc[0]=='<' && pc[1]=='0' &&
        pc[2]=='x' && pc[5]=='>'){
        if(!outcap)return -1;out[outn++]=(unsigned char)(hexnib((char)pc[3])*16+hexnib((char)pc[4]));
        return (int)outn;
    }
    int i = 0;
    while (pc[i]){
        int cp, extra;
        if (pc[i] < 0x80){ cp = pc[i]; extra = 0; }
        else if ((pc[i] & 0xE0) == 0xC0){ cp = pc[i] & 0x1F; extra = 1; }
        else if ((pc[i] & 0xF0) == 0xE0){ cp = pc[i] & 0x0F; extra = 2; }
        else if ((pc[i] & 0xF8) == 0xF0){ cp = pc[i] & 0x07; extra = 3; }
        else { i++; continue; }                 /* stray lead byte, skip */
        int ok = 1;
        for (int e=0; e<extra; e++){ if (!pc[i+1+e]){ ok=0; break; } cp = (cp<<6) | (pc[i+1+e] & 0x3F); }
        i += 1 + extra;
        if (!ok) continue;
        if(outn>=outcap)return -1;
        if(cp==0x2581)out[outn++]=' '; /* SentencePiece space marker (kept safe) */
        else if(cp<512&&g_unmap[cp]>=0)out[outn++]=(unsigned char)g_unmap[cp];
        else out[outn++]=(unsigned char)cp;
    }
    return outn>(size_t)INT_MAX?-1:(int)outn;
}

static int decode_id_alloc(int id,unsigned char **out,int *outn){
    if(!out||!outn)return -1;*out=NULL;*outn=0;
    if(!g_tok||id<0||id>=g_tok_n||!g_tok[id])return 0;
    size_t cap=strlen(g_tok[id])+1u;if(cap>(size_t)INT_MAX)return -1;
    unsigned char *bytes=(unsigned char*)malloc(cap);if(!bytes)return -1;
    int n=decode_id_to_bytes(id,bytes,cap);if(n<0){free(bytes);return -1;}
    *out=bytes;*outn=n;return 0;
}

static int q38_bytes_append(unsigned char **out,size_t *length,size_t *capacity,
                            const unsigned char *bytes,size_t count){
    if(count>SIZE_MAX-*length)return -1;size_t need=*length+count;
    if(need>*capacity){
        size_t grown=*capacity?*capacity:256u;
        while(grown<need){if(grown>SIZE_MAX/2u){grown=need;break;}grown*=2u;}
        unsigned char *next=(unsigned char*)realloc(*out,grown);if(!next)return -1;
        *out=next;*capacity=grown;
    }
    if(count)memcpy(*out+*length,bytes,count);*length=need;return 0;
}

/* Decode a range without imposing a response-size ceiling. The returned byte
 * length remains authoritative even when the text contains an embedded NUL. */
static int decode_range_alloc(const int *arr,int from,int to,
                              unsigned char **out,size_t *outn){
    if(!arr||from<0||to<from||!out||!outn)return -1;
    unsigned char carry[4];int carry_n=0;unsigned char *all=NULL;size_t length=0,capacity=0;
    for(int index=from;index<to;index++){
        unsigned char *token=NULL;int token_n=0;
        if(decode_id_alloc(arr[index],&token,&token_n)){free(all);return -1;}
        if(token_n>(INT_MAX-3)/3){free(token);free(all);return -1;}
        int chunk_cap=token_n*3+3;
        unsigned char *chunk=(unsigned char*)malloc((size_t)chunk_cap);int chunk_n=0;
        if(!chunk||utf8_drain(carry,&carry_n,token,token_n,chunk,chunk_cap,&chunk_n)<0||
           q38_bytes_append(&all,&length,&capacity,chunk,(size_t)chunk_n)){
            free(chunk);free(token);free(all);return -1;
        }
        free(chunk);free(token);
    }
    unsigned char tail[3];int tail_n=0;
    if(utf8_finish(carry,&carry_n,tail,sizeof tail,&tail_n)||
       q38_bytes_append(&all,&length,&capacity,tail,(size_t)tail_n)||
       q38_bytes_append(&all,&length,&capacity,(const unsigned char*)"",1u)){
        free(all);return -1;
    }
    length--;*out=all;*outn=length;return 0;
}

/* Append bytes to a buffer and flush any complete UTF-8 codepoints; any
 * trailing partial sequence is left in the buffer for the next call. */
static void out_bytes(FILE *stream,unsigned char *buf,int *bn,
                      const unsigned char *b,int n){
    if(n>(INT_MAX-3)/3){fprintf(stderr,"[decode] token is too large\n");exit(1);}
    int cap=n*3+3,outn=0;unsigned char *decoded=(unsigned char*)malloc((size_t)cap);
    if(!decoded||utf8_drain(buf,bn,b,n,decoded,cap,&outn)<0){
        free(decoded);fprintf(stderr,"[decode] invalid output capacity\n");exit(1);
    }
    if(outn)fwrite(decoded,1,(size_t)outn,stream);free(decoded);
}

static void print_decoded(FILE *stream,const int *arr,int from,int to){
    unsigned char *text=NULL;size_t length=0;
    if(decode_range_alloc(arr,from,to,&text,&length)){
        fprintf(stderr,"[decode] out of memory\n");exit(1);
    }
    if(length)fwrite(text,1,length,stream);free(text);
}

static void openai_content_chunk(const unsigned char *bytes,size_t length){
    char *content=json_escape_alloc(bytes,length);
    char *model=json_escape_alloc((const unsigned char*)g_model,strlen(g_model));
    if(!content||!model){free(content);free(model);fprintf(stderr,"[openai] out of memory formatting response\n");exit(1);}
    char *json=q38_format_alloc(
        "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
        g_oa_id,g_oa_created,model,content);
    free(content);free(model);
    if(!json){fprintf(stderr,"[openai] out of memory formatting response\n");exit(1);}
    sse_chunk(json);free(json);
}

/* Streaming variants: emit one token at a time. In OpenAI mode each token is
 * one SSE `chat.completion.chunk` (delta.content = decoded text for this token,
 * carrying partial UTF-8 across tokens so CJK never splits mid-codepoint).
 * Otherwise emit raw readable text, flushing complete UTF-8 codepoints. */
static void stream_token(int id){
    if (g_openai){
        if (g_ttft < 0) g_ttft = now_s() - g_gen_t0;   /* TTFT on first token */
        if (!g_tok){
            char token[32];int length=snprintf(token,sizeof token,"%d",id);
            openai_content_chunk((const unsigned char*)token,(size_t)length);return;
        }
        unsigned char *token=NULL;int token_n=0;
        if(decode_id_alloc(id,&token,&token_n)){fprintf(stderr,"[decode] out of memory\n");exit(1);}
        if(token_n>(INT_MAX-3)/3){free(token);fprintf(stderr,"[decode] token is too large\n");exit(1);}
        int chunk_cap=token_n*3+3;
        unsigned char *chunk=(unsigned char*)malloc((size_t)chunk_cap);int chunk_n=0;
        if(!chunk||utf8_drain(g_sbuf,&g_sbn,token,token_n,chunk,chunk_cap,&chunk_n)<0){
            free(chunk);free(token);fprintf(stderr,"[decode] invalid output capacity\n");exit(1);
        }
        if(chunk_n>0)openai_content_chunk(chunk,(size_t)chunk_n);
        free(chunk);free(token);
        return;
    }
    /* default raw-text streaming */
    if (!g_tok){ printf("%d ", id); fflush(stdout); return; }
    unsigned char *token=NULL;int token_n=0;
    if(decode_id_alloc(id,&token,&token_n)){fprintf(stderr,"[decode] out of memory\n");exit(1);}
    out_bytes(stdout,g_sbuf,&g_sbn,token,token_n);free(token);
    fflush(stdout);   /* make streaming visible immediately even when piped */
}
static void stream_flush(void){
    unsigned char tail[3];int tail_n=0;
    if(utf8_finish(g_sbuf,&g_sbn,tail,sizeof tail,&tail_n)<0){
        fprintf(stderr,"[decode] invalid UTF-8 finalization\n");exit(1);
    }
    if(tail_n)fwrite(tail,1,(size_t)tail_n,stdout);
}

/* Emit the final OpenAI Chat Completions response for a finished generation.
 * Streaming: flushes any trailing partial UTF-8 as a last content chunk, then
 * sends the termination chunk (finish_reason + usage + timings) and "data: [DONE]".
 * Non-streaming: sends a single chat.completion JSON object.
 * When g_sock_out >= 0 the bytes go to the live socket; otherwise to stdout. */
static void emit_openai_result(const int *out, int np, int n_new, int stream){
    double total = now_s() - g_gen_t0;
    if (g_ttft < 0) g_ttft = total;   /* non-streaming: all tokens arrive at once */
    double gen_t = total - g_ttft;
    double tps = q38_decode_rate(n_new,gen_t);
    if (stream){
        unsigned char tail[3];int tail_n=0;
        if(utf8_finish(g_sbuf,&g_sbn,tail,sizeof tail,&tail_n)<0){
            fprintf(stderr,"[decode] invalid UTF-8 finalization\n");exit(1);
        }
        if(tail_n)openai_content_chunk(tail,(size_t)tail_n);
        char *model=json_escape_alloc((const unsigned char*)g_model,strlen(g_model));
        char *jb=model?q38_format_alloc(
          "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
          "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"
          "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
          "\"timings\":{\"ttft_s\":%.3f,\"tokens_per_sec\":%.3f,\"total_s\":%.3f}}",
          g_oa_id,g_oa_created,model,np,n_new,np+n_new,g_ttft,tps,total):NULL;
        free(model);if(!jb){fprintf(stderr,"[openai] out of memory formatting response\n");exit(1);}
        sse_chunk(jb);free(jb);
        char done[16]; int dl = snprintf(done, sizeof done, "data: [DONE]\n\n");
        if (g_sock_out >= 0 && g_sock_send) g_sock_send(g_sock_out, done, dl);
        else { fwrite(done, 1, (size_t)dl, stdout); fflush(stdout); }
    } else {
        unsigned char *text=NULL;size_t text_length=0;
        if(decode_range_alloc(out,np,np+n_new,&text,&text_length)){
            fprintf(stderr,"[decode] out of memory\n");exit(1);
        }
        char *content=json_escape_alloc(text,text_length);
        char *model=json_escape_alloc((const unsigned char*)g_model,strlen(g_model));
        free(text);
        char *buf=content&&model?q38_format_alloc(
          "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\","
          "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
          "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
          "\"timings\":{\"ttft_s\":%.3f,\"tokens_per_sec\":%.3f,\"total_s\":%.3f}}\n",
          g_oa_id,g_oa_created,model,content,np,n_new,np+n_new,g_ttft,tps,total):NULL;
        free(content);free(model);
        if(!buf){fprintf(stderr,"[openai] out of memory formatting response\n");exit(1);}
        size_t length=strlen(buf);
        if(g_sock_out>=0&&g_sock_send)g_sock_send(g_sock_out,buf,(int)length);
        else{fwrite(buf,1,length,stdout);fflush(stdout);}free(buf);
    }
}

/* ---------- native text inference core ---------- */
#include "qwen38_core.h"
#ifdef COLI_SEGMENT_ADAPTER
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"
#endif
#ifdef COLI_EDGE_ADAPTER
#include "edge_runtime.h"
#include "edge_adapters.h"
#include "edge_adapter_internal.h"
#endif

/* Qwen3.8 routes every transformer layer and has no MTP row. Keep the
 * process-global route_trace owner in the CLI/serve model only; segment
 * adapters can coexist in one process and intentionally leave telemetry
 * detached, just like the other range-native engines. */
static void q38_telemetry_init(const char *snap, const Model *m) {
    rt_init("qwen38", m->c.layers, m->c.experts);
    rt_drop_row(m->c.layers);
    const char *up = getenv("COLI_USAGE");
    if (up && *up) snprintf(g_q38_usage, sizeof g_q38_usage, "%s", up);
    else snprintf(g_q38_usage, sizeof g_q38_usage, "%s/.coli_usage", snap);
    int64_t h = rt_load(g_q38_usage);
    if (h > 0)
        fprintf(stderr, "[USAGE] expert history: %lld selections (%s)\n",
                (long long)h, g_q38_usage);
}

static int q38_context_total(const Cfg *c, int prompt_tokens, int new_tokens,
                             const char *operation) {
    int limit = c->max_positions < QWEN38_ATTN_MAX_CTX
              ? c->max_positions : QWEN38_ATTN_MAX_CTX;
    if (prompt_tokens < 1 || new_tokens < 1 || prompt_tokens > limit ||
        new_tokens > limit - prompt_tokens) {
        fprintf(stderr, "[ctx] %s requires prompt=%d + new=%d tokens, capacity=%d\n",
                operation, prompt_tokens, new_tokens, limit);
        exit(1);
    }
    return prompt_tokens + new_tokens;
}

static void q38_validate_ids(const Cfg *c, const int *ids, int count,
                             const char *label) {
    if (!ids || count < 1) {
        fprintf(stderr, "%s must contain at least one token\n", label);
        exit(1);
    }
    for (int i = 0; i < count; i++) {
        if (ids[i] < 0 || ids[i] >= c->vocab) {
            fprintf(stderr, "%s[%d]=%d outside vocabulary [0,%d)\n",
                    label, i, ids[i], c->vocab);
            exit(1);
        }
    }
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    /* Same ceiling serve_one() enforces. Past max_position_embeddings the RoPE
     * positions leave the range the model was trained on, so this is a
     * correctness limit, not just a memory one. */
    m->max_t = q38_context_total(c, np, n_new, "generation");
    q38_validate_ids(c, prompt, np, "prompt_ids");
    reset_recurrent(m);
    ensure_kv(m);
    m->kv_len = 0;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        if (s == 0 && g_ttft < 0) g_ttft = now_s() - g_gen_t0;   /* record TTFT */
        if (g_stream) { stream_token(best); fflush(stdout); }
        if (s == n_new - 1) {
            if (g_capture_last_logit) {
                g_last_logit = malloc((size_t)c->vocab * sizeof(float));
                if (!g_last_logit) {
                    fprintf(stderr, "OOM capturing Qwen3.8 oracle logits\n"); exit(1);
                }
                memcpy(g_last_logit, logit, (size_t)c->vocab * sizeof(float));
            }
            free(logit); out[len++] = best; break;
        }
        free(logit); out[len++] = best;
        int one = best;
        logit = step(m, &one, 1, len - 1);
    }
}

static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    if (nfull <= np) {
        fprintf(stderr, "PPL requires full_ids to extend prompt_ids\n");
        exit(1);
    }
    m->max_t = q38_context_total(c, np, nfull - np, "PPL");
    q38_validate_ids(c, full, nfull, "full_ids");
    reset_recurrent(m);
    ensure_kv(m);
    m->kv_len = 0;
    double nll = 0; int scored = 0;
    float *logit = step(m, full, np, 0);
    for (int i = np; i < nfull; i++) {
        float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[full[i]] - mx - log(Z));
        scored++;
        free(logit); logit = NULL;
        if (i == nfull - 1) break;
        logit = step(m, &full[i], 1, i);
    }
    if (logit) free(logit);
    *nll_out = nll / scored;
    return scored;
}

static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { fprintf(stderr, "ref.json: missing array \"%s\"\n", key); exit(1); }
    if (a->len < 1 || (uint64_t)a->len > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "ref.json: invalid array length for \"%s\"\n", key); exit(1);
    }
    int *r = malloc((size_t)a->len * sizeof(int));
    if (!r) { fprintf(stderr, "ref.json: OOM reading \"%s\"\n", key); exit(1); }
    for (int i = 0; i < a->len; i++) {
        jval *v = a->kids[i];
        if (!v || v->t != J_NUM || !isfinite(v->num) || v->num < 0 ||
            v->num > INT_MAX || floor(v->num) != v->num) {
            fprintf(stderr, "ref.json: %s[%d] must be a non-negative integer\n", key, i);
            exit(1);
        }
        r[i] = (int)v->num;
    }
    *n_out = a->len; return r;
}

static jval *read_reference_logits(jval *root,int vocab){
    jval *schema=json_get(root,"schema_version");
    if(!schema||schema->t!=J_NUM||!isfinite(schema->num)||
       floor(schema->num)!=schema->num||
       (schema->num!=1.0&&schema->num!=2.0)){
        fprintf(stderr,"ref.json: schema_version must be the supported integer 1 or 2\n");exit(1);
    }
    int requires_logits=schema->num==2.0;
    jval *values=json_get(root,"final_logits");
    if(!values){
        if(requires_logits){
            fprintf(stderr,"ref.json: schema version 2 requires final_logits\n");exit(1);
        }
        return NULL;
    }
    if(values->t!=J_ARR||values->len!=vocab){
        fprintf(stderr,"ref.json: final_logits must contain exactly %d numbers\n",vocab);exit(1);
    }
    for(int i=0;i<values->len;i++){
        jval *value=values->kids[i];
        if(!value||value->t!=J_NUM||!isfinite(value->num)||
           value->num>FLT_MAX||value->num<-FLT_MAX){
            fprintf(stderr,"ref.json: final_logits[%d] must be a finite float\n",i);exit(1);
        }
    }
    return values;
}

static int compare_reference_logits(const float *actual,const jval *expected){
    if(!actual||!expected||expected->t!=J_ARR)return 0;
    double dot=0.0,actual_ss=0.0,expected_ss=0.0,max_abs=0.0,max_ref=0.0;
    for(int i=0;i<expected->len;i++){
        double a=actual[i],b=expected->kids[i]->num;
        if(!isfinite(a)){
            fprintf(stderr,"Numeric oracle: non-finite C logit at %d\n",i);return 0;
        }
        dot+=a*b;actual_ss+=a*a;expected_ss+=b*b;
        double delta=fabs(a-b);if(delta>max_abs)max_abs=delta;
        double magnitude=fabs(b);if(magnitude>max_ref)max_ref=magnitude;
    }
    double cosine=actual_ss>0.0&&expected_ss>0.0?
                  dot/sqrt(actual_ss*expected_ss):(actual_ss==expected_ss?1.0:0.0);
    /* The C path accumulates in FP32 while the released PyTorch oracle carries
     * BF16 activations. Bound both pointwise drift and vector direction. */
    double max_allowed=fmax(0.01,0.02*max_ref);
    int match=cosine>=0.9999&&max_abs<=max_allowed;
    printf("Numeric oracle: cosine=%.8f max_abs=%.6g (limit %.6g) %s\n",
           cosine,max_abs,max_allowed,match?"PASS":"FAIL");
    return match;
}

#if !defined(QWEN38_NO_MAIN) || defined(QWEN38_TEST_SERVE)

/* ===================== coli serve mode (SERVE=1) ===================== *
 * Implements the colibri gateway wire protocol so `coli chat` / `coli web` /
 * `coli serve` can drive this engine. Without it the engine is unreachable:
 * users run `coli chat`, not the binary directly.
 * Protocol (matches kimi_k3.c / inkling.c, the other non-GLM engines):
 *   engine:  \x01\x01READY\x01\x01\n
 *            STAT 0 0.00 0.0 <rss>\n
 *   gateway: SUBMIT <id> <slot> <plen> <max_tok> <temp> <top_p>\n <payload bytes>\n
 *   Qwen3.8 accepts slot 0 only; its single hybrid context is reused when
 *   the next prompt begins with the exact token sequence it already covers.
 *   engine:  ACCEPT <id> <np>\n
 *            DATA <id> <n>\n <bytes>\n     (repeated per decoded chunk)
 *            DONE <id> STAT <gen> <tps> <hit%> <rss> <np> <limited>\n
 *   gateway: CANCEL <id>  (abort current turn)
 * Windows: stdout/stdin must go binary BEFORE the READY sentinel or the CRT
 * rewrites the trailing \n as \r\n and the gateway never matches it -> the
 * session hangs forever (#748). compat.h's coli_serve_binary_mode (#749)
 * carries that fix for every engine; see its comment for the full story. */

typedef struct {
    char id[64];
    int slot;
    int max_tok;
    float temp, top_p;
    char *payload;
    int plen;
} ServeReq;
static const ColiServeWireProfile q38_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 24,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
};

/* Returns 2=SUBMIT, 1=STOP(active), 3=CANCEL(active), 0=ignored, -1=EOF/fatal.
 * A SUBMIT encountered during a synchronous turn is consumed and rejected so
 * its payload cannot desynchronize the following control frame. */
static int serve_read_req(FILE *in,FILE *out,ServeReq *q,const char *active_id){
    ColiServeCommand command;
    ColiServeReadResult result=coli_serve_read_command(in,&q38_wire,&command);
    if(result==COLI_SERVE_READ_EOF||result==COLI_SERVE_READ_BAD_FRAME)return -1;
    if(result==COLI_SERVE_READ_NOMEM){coli_serve_write_error(out,command.id,"out of memory");return -1;}
    if(result==COLI_SERVE_READ_BAD_REQUEST){
        if(command.kind==COLI_SERVE_COMMAND_SUBMIT)
            coli_serve_write_error(out,command.id,"bad submit header");
        coli_serve_command_dispose(&command);return 0;
    }
    if(result!=COLI_SERVE_READ_OK)return 0;
    if(command.kind==COLI_SERVE_COMMAND_STOP||command.kind==COLI_SERVE_COMMAND_CANCEL){
        int active=active_id&&!strcmp(command.id,active_id);
        int control=active?(command.kind==COLI_SERVE_COMMAND_STOP?1:3):0;
        coli_serve_command_dispose(&command);return control;
    }
    if(command.kind!=COLI_SERVE_COMMAND_SUBMIT){coli_serve_command_dispose(&command);return 0;}
    if(active_id){
        coli_serve_write_error(out,command.id,"engine busy");
        coli_serve_command_dispose(&command);return 0;
    }
    /* Qwen3.8 owns one recurrent/KV state.  The gateway still sends the
     * common cache-slot field, so reject every slot except that sole slot
     * after the codec has consumed the complete frame. */
    if(command.slot!=0){
        coli_serve_write_error(out,command.id,"invalid cache slot");
        coli_serve_command_dispose(&command);return 0;
    }
    if(!q){coli_serve_command_dispose(&command);return 0;}
    snprintf(q->id,sizeof(q->id),"%s",command.id);
    q->slot=command.slot;
    q->max_tok=command.max_tokens;q->temp=command.temperature;q->top_p=command.top_p;
    q->payload=(char*)coli_serve_command_take_payload(&command);q->plen=(int)command.payload_bytes;
    coli_serve_command_dispose(&command);return 2;
}

static void serve_data(const char *id, const char *p, int n){
    if(n<=0) return;
    printf("DATA %s %d\n",id,n);
    fwrite(p,1,(size_t)n,stdout); fputc('\n',stdout); fflush(stdout);
}

/* temperature + top-p sampler (ported from kimi_k3.c; vocab ~250k -> qsort O(V log V) per token) */
typedef struct { float p; int id; } SampleProb;
static int sample_prob_desc(const void *a, const void *b){
    float pa=((const SampleProb*)a)->p, pb=((const SampleProb*)b)->p;
    return (pb>pa)-(pa>pb);
}
static int serve_sample(const float *lo, int V, float temp, float top_p){
    if(temp<=0.f){ int b=0; for(int i=1;i<V;i++) if(lo[i]>lo[b]) b=i; return b; }
    SampleProb *rank=malloc((size_t)V*sizeof(SampleProb)); float mx=lo[0];
    if(!rank){ fprintf(stderr,"OOM sampling\n"); exit(1); }
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double sum=0;
    for(int i=0;i<V;i++){ float p=expf((lo[i]-mx)/temp); sum+=p; rank[i]=(SampleProb){p,i}; }
    qsort(rank,(size_t)V,sizeof(SampleProb),sample_prob_desc);
    double cut=(top_p>0.f&&top_p<1.f)?top_p*sum:sum, kept=0; int n=0;
    while(n<V&&kept<cut) kept+=rank[n++].p;
    double r=((double)rand()/RAND_MAX)*kept, acc=0; int pick=rank[0].id;
    for(int i=0;i<n;i++){ acc+=rank[i].p; if(acc>=r){ pick=rank[i].id; break; } }
    free(rank); return pick;
}

/* Chat turns end on <|im_end|>, base completions on <|endoftext|>. Resolve
 * both ids from the tokenizer's added_tokens: Qwen3.8's 248320-token vocab
 * puts them at 248044+, so the old hardcoded 151645 (the 151k-vocab Qwen id)
 * silently never matched and every serve turn ran into max_tok. Q38_EOS
 * still overrides for experiments. */
static int serve_eos_ids(int *ids,int cap,int config_eos,int vocab){
    int n=0;
    const char *override=getenv("Q38_EOS");
    if(override){
        char *end=NULL;errno=0;long value=strtol(override,&end,10);
        if(errno||end==override||*end||value<0||value>=vocab){
            fprintf(stderr,"Q38_EOS must be an integer in [0,%d)\n",vocab);exit(1);
        }
        ids[n++]=(int)value;return n;
    }
    for(int k=0;k<g_nspecial && n<cap;k++)
        if(!strcmp(g_sp_str[k],"<|im_end|>")||!strcmp(g_sp_str[k],"<|endoftext|>"))
            ids[n++]=g_sp_id[k];
    if(!n && config_eos>=0) ids[n++]=config_eos;
    return n;
}

/* ---------- one-slot hybrid prompt prefix -----------------------------
 *
 * Qwen3.8 has two different kinds of state.  DeltaNet and PLE carry a
 * recurrent state which generation mutates in place; QSA carries position-
 * indexed K/V/index rows which are append-only.  A prompt cache therefore
 * keeps a bounded snapshot of only the recurrent state and leaves the QSA
 * rows in the model's existing buffers.  Generation writes rows strictly
 * after the cached prompt, so those rows remain valid for the next turn.
 *
 * There is deliberately one process-local cache: the Qwen serve ABI exposes
 * one slot and the serve loop owns one Model.  This is not a second KV cache
 * and cannot grow with the number of requests.  The snapshot is published
 * only after the complete prompt has been evaluated; a cancelled/stopped
 * decode consequently cannot damage the prompt state that follows it. */
typedef struct {
    Model *owner;
    int valid;
    int layers;
    int vocab;
    size_t rec_cells;
    size_t conv_cells;
    size_t ple_cells;
    int *ids;
    int id_cap;
    int len;
    float *logits;
    float **dn_rec;
    float **dn_conv;
    float *ple_conv;
    int64_t ple_history[2];
    int ple_history_len;
} Q38PrefixCache;

static Q38PrefixCache g_q38_prefix;

static int q38_size_mul(size_t left,size_t right,size_t *out){
    if(!out||(right&&left>SIZE_MAX/right))return 0;
    *out=left*right;return 1;
}

static int q38_prefix_geometry(const Model *m,size_t *rec_cells,
                               size_t *conv_cells,size_t *ple_cells){
    if(!m||!rec_cells||!conv_cells||!ple_cells||m->c.layers<1||
       m->c.layers>Q38_MAX_LAYERS||m->c.vocab<1||!m->c.is_attn)return 0;
    const Cfg *c=&m->c;size_t rec=0,conv=0,ple=0,cells=0;
    if(c->dn_vheads<=0||c->dn_kdim<=0||c->dn_vdim<=0||
       c->dn_conv_dim<=0||c->dn_convk<2)return 0;
    if(!q38_size_mul((size_t)c->dn_vheads,(size_t)c->dn_kdim,&cells)||
       !q38_size_mul(cells,(size_t)c->dn_vdim,&cells))return 0;
    rec=cells;
    if(!q38_size_mul((size_t)c->dn_conv_dim,(size_t)(c->dn_convk-1),&cells))return 0;
    conv=cells;
    if(c->ple_layer>=0&&c->ple_layer<c->layers){
        if(c->hc_width<=0||c->ple_convk<2||c->ngram_size<=0||
           !q38_size_mul((size_t)c->hc_width,(size_t)(c->ple_convk-1),&cells)||
           !q38_size_mul(cells,(size_t)c->ngram_size,&ple))return 0;
        if(!m->PLE_conv_state||!m->ple_history)return 0;
    }
    *rec_cells=rec;*conv_cells=conv;*ple_cells=ple;return 1;
}

static void q38_prefix_cache_dispose(Q38PrefixCache *cache){
    if(!cache)return;
    if(cache->dn_rec)for(int i=0;i<cache->layers;i++)free(cache->dn_rec[i]);
    if(cache->dn_conv)for(int i=0;i<cache->layers;i++)free(cache->dn_conv[i]);
    free(cache->dn_rec);free(cache->dn_conv);free(cache->ple_conv);
    free(cache->ids);free(cache->logits);memset(cache,0,sizeof(*cache));
}

static void q38_prefix_cache_invalidate(void){
    g_q38_prefix.valid=0;g_q38_prefix.len=0;
}

static int q38_prefix_cache_layout(Model *m){
    if(!m)return 0;
    size_t rec=0,conv=0,ple=0;
    if(!q38_prefix_geometry(m,&rec,&conv,&ple))return 0;
    if(g_q38_prefix.owner!=m||g_q38_prefix.layers!=m->c.layers||
       g_q38_prefix.vocab!=m->c.vocab||g_q38_prefix.rec_cells!=rec||
       g_q38_prefix.conv_cells!=conv||g_q38_prefix.ple_cells!=ple){
        q38_prefix_cache_dispose(&g_q38_prefix);g_q38_prefix.owner=m;
        g_q38_prefix.layers=m->c.layers;g_q38_prefix.vocab=m->c.vocab;
        g_q38_prefix.rec_cells=rec;g_q38_prefix.conv_cells=conv;
        g_q38_prefix.ple_cells=ple;
    }
    if(g_q38_prefix.dn_rec&&g_q38_prefix.dn_conv&&g_q38_prefix.logits&&
       (!ple||g_q38_prefix.ple_conv))return 1;
    size_t logits_bytes=0;
    if(!q38_size_mul((size_t)m->c.vocab,sizeof(float),&logits_bytes))return 0;
    g_q38_prefix.dn_rec=(float**)calloc((size_t)m->c.layers,sizeof(float*));
    g_q38_prefix.dn_conv=(float**)calloc((size_t)m->c.layers,sizeof(float*));
    g_q38_prefix.logits=(float*)malloc(logits_bytes);
    if(!g_q38_prefix.dn_rec||!g_q38_prefix.dn_conv||!g_q38_prefix.logits){
        q38_prefix_cache_dispose(&g_q38_prefix);return 0;
    }
    for(int i=0;i<m->c.layers;i++)if(!m->c.is_attn[i]){
        if(!m->DN_rec[i]||!m->DN_conv[i]||
           (g_q38_prefix.rec_cells>SIZE_MAX/sizeof(float))||
           (g_q38_prefix.conv_cells>SIZE_MAX/sizeof(float)))
            {q38_prefix_cache_dispose(&g_q38_prefix);return 0;}
        g_q38_prefix.dn_rec[i]=(float*)malloc(g_q38_prefix.rec_cells*sizeof(float));
        g_q38_prefix.dn_conv[i]=(float*)malloc(g_q38_prefix.conv_cells*sizeof(float));
        if(!g_q38_prefix.dn_rec[i]||!g_q38_prefix.dn_conv[i]){
            q38_prefix_cache_dispose(&g_q38_prefix);return 0;
        }
    }
    if(ple){
        if(g_q38_prefix.ple_cells>SIZE_MAX/sizeof(float)){
            q38_prefix_cache_dispose(&g_q38_prefix);return 0;
        }
        g_q38_prefix.ple_conv=(float*)malloc(g_q38_prefix.ple_cells*sizeof(float));
        if(!g_q38_prefix.ple_conv){q38_prefix_cache_dispose(&g_q38_prefix);return 0;}
    }
    return 1;
}

static int q38_prefix_ids_reserve(int len){
    if(len<1||(size_t)len>SIZE_MAX/sizeof(int))return 0;
    if(g_q38_prefix.ids&&len<=g_q38_prefix.id_cap)return 1;
    size_t bytes=(size_t)len*sizeof(int);int *ids=(int*)malloc(bytes);
    if(!ids)return 0;
    if(g_q38_prefix.ids&&g_q38_prefix.len>0)
        memcpy(ids,g_q38_prefix.ids,(size_t)g_q38_prefix.len*sizeof(int));
    free(g_q38_prefix.ids);g_q38_prefix.ids=ids;g_q38_prefix.id_cap=len;return 1;
}

static void q38_prefix_copy_state(Model *m,int to_cache){
    for(int i=0;i<m->c.layers;i++)if(!m->c.is_attn[i]){
        float *rec=to_cache?g_q38_prefix.dn_rec[i]:m->DN_rec[i];
        float *conv=to_cache?g_q38_prefix.dn_conv[i]:m->DN_conv[i];
        const float *src_rec=to_cache?m->DN_rec[i]:g_q38_prefix.dn_rec[i];
        const float *src_conv=to_cache?m->DN_conv[i]:g_q38_prefix.dn_conv[i];
        memcpy(rec,src_rec,g_q38_prefix.rec_cells*sizeof(float));
        memcpy(conv,src_conv,g_q38_prefix.conv_cells*sizeof(float));
    }
    if(g_q38_prefix.ple_cells){
        if(to_cache){
            memcpy(g_q38_prefix.ple_conv,m->PLE_conv_state,
                   g_q38_prefix.ple_cells*sizeof(float));
            g_q38_prefix.ple_history_len=m->ple_history_len;
            memcpy(g_q38_prefix.ple_history,m->ple_history,sizeof(g_q38_prefix.ple_history));
        }else{
            memcpy(m->PLE_conv_state,g_q38_prefix.ple_conv,
                   g_q38_prefix.ple_cells*sizeof(float));
            m->ple_history_len=g_q38_prefix.ple_history_len;
            memcpy(m->ple_history,g_q38_prefix.ple_history,sizeof(g_q38_prefix.ple_history));
        }
    }
}

static int q38_prefix_cache_save(Model *m,const int *ids,int len,const float *logits){
    if(!m||!ids||len<1||!logits||!q38_prefix_cache_layout(m)||
       !q38_prefix_ids_reserve(len))return 0;
    memcpy(g_q38_prefix.ids,ids,(size_t)len*sizeof(int));
    memcpy(g_q38_prefix.logits,logits,(size_t)m->c.vocab*sizeof(float));
    q38_prefix_copy_state(m,1);g_q38_prefix.len=len;g_q38_prefix.valid=1;return 1;
}

/* Restore only when the complete cached prompt is an exact prefix.  A shorter
 * request would require rewinding append-only QSA rows and is therefore a
 * safe miss; equality is useful because the saved prompt logits let the
 * caller enter decode without feeding the final prompt token twice. */
static int q38_prefix_restore(Model *m,const int *ids,int len){
    if(!m||!ids||len<1||!g_q38_prefix.valid||g_q38_prefix.owner!=m||
       g_q38_prefix.len<1||g_q38_prefix.len>len||
       memcmp(g_q38_prefix.ids,ids,(size_t)g_q38_prefix.len*sizeof(int)))return 0;
    q38_prefix_copy_state(m,0);m->kv_len=g_q38_prefix.len;return g_q38_prefix.len;
}

static const float *q38_prefix_cached_logits(Model *m){
    return m&&g_q38_prefix.valid&&g_q38_prefix.owner==m?g_q38_prefix.logits:NULL;
}

static void q38_prefix_cache_release(Model *m){
    if(!m||g_q38_prefix.owner!=m)return;
    q38_prefix_cache_dispose(&g_q38_prefix);
}

/* Allocate the complete served QSA bank once, before READY.  Prefix reuse
 * relies on append-only position rows; one planned allocation both preserves
 * those rows and avoids a growth peak containing the old and new banks at the
 * same time.  An initialized serve model is therefore never grown in place. */
static int q38_serve_ensure_kv(Model *m,int capacity){
    if(!m||capacity<1)return 0;
    if(m->kv_cap>0)return m->K&&m->kv_cap>=capacity;
    m->max_t=capacity;ensure_kv(m);
    return m->K&&m->kv_cap>=capacity;
}

static double q38_cache_hit_percent(uint64_t hits,uint64_t misses){
    double total=(double)hits+(double)misses;
    return total?100.0*(double)hits/total:0.0;
}

static int q38_format_prof(char *out,size_t capacity,double wall_s,int prompt_tokens,
                            int completion_tokens,const Q38Timers *timers){
    double expert_read=timers->seconds[Q38_TM_EXPERT_READ];
    double fp8_expand=timers->seconds[Q38_TM_FP8_EXPAND];
    double expert_compute=timers->seconds[Q38_TM_ROUTED_EXPERT]+
                          timers->seconds[Q38_TM_SHARED_EXPERT];
    double attention=timers->seconds[Q38_TM_DELTANET]+
                     timers->seconds[Q38_TM_QSA_INDEX]+
                     timers->seconds[Q38_TM_QSA_ATTENTION];
    int count=snprintf(out,capacity,
        "PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n",
        wall_s,prompt_tokens,completion_tokens,expert_read,
        expert_read+fp8_expand,expert_compute,attention,
        timers->seconds[Q38_TM_LM_HEAD],
        (unsigned long long)timers->forwards);
    return count>=0&&(size_t)count<capacity?count:-1;
}

static int serve_one(Model *m, ServeReq *q){
    int *ids=NULL, np=0;
    encode_text_n(q->payload,(size_t)q->plen,&ids,&np); /* byte-counted prompt; qwen38 adds no BOS */
    int max_ctx=m->kv_cap;
    if(np<1 || np>max_ctx || q->max_tok<1 || q->max_tok>max_ctx-np){
        printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",q->id,np,q->max_tok,max_ctx);
        fflush(stdout); free(ids); return 0;
    }
    printf("ACCEPT %s %d\n",q->id,np); fflush(stdout);
    double request_started=now_s();
    uint64_t hits_before=m->hits, misses_before=m->miss;
    Q38Timers timers_before=m->timers;
    int reuse=q38_prefix_restore(m,ids,np);
    float *lo=NULL;
    if(reuse==np){
        const float *cached=q38_prefix_cached_logits(m);
        if(!cached){
            q38_prefix_cache_invalidate();reset_recurrent(m);m->kv_len=0;
            lo=step(m,ids,np,0);reuse=0;
        }else{
            lo=falloc(m->c.vocab);
            memcpy(lo,cached,(size_t)m->c.vocab*sizeof(float));
        }
    }else if(reuse>0){
        lo=step(m,ids+reuse,np-reuse,reuse);
    }else{
        /* A miss invalidates the snapshot before reset so no later request can
         * pair its recurrent state with the old token record. */
        q38_prefix_cache_invalidate();reset_recurrent(m);m->kv_len=0;
        lo=step(m,ids,np,0);
    }
    if(reuse!=np&&!q38_prefix_cache_save(m,ids,np,lo)&&getenv("Q38_PREFIX_LOG"))
        fprintf(stderr,"[qwen38 prefix] cache disabled for request %s (state snapshot unavailable)\n",q->id);
    if(getenv("Q38_PREFIX_LOG"))
        fprintf(stderr,"[qwen38 prefix] request=%s reused=%d/%d\n",q->id,reuse,np);
    int gen=0, limited=1, cancelled=0, stopped=0, input_eof=0;
    int eos_ids[4];int n_eos=serve_eos_ids(eos_ids,4,m->c.eos_id,m->c.vocab);
    double first_token_at=0.0,last_token_at=0.0;
    unsigned char sbuf[16]; int sbn=0;
    for(int s=0;s<q->max_tok;s++){
        while(!input_eof&&coli_stdin_readable()){
            int control=serve_read_req(stdin,stdout,NULL,q->id);
            if(control<0){input_eof=1;break;}
            if(control==1)stopped=1;
            if(control==3)cancelled=1;
        }
        if(cancelled||stopped)break;
        int tk = serve_sample(lo, m->c.vocab, q->temp, q->top_p);
        free(lo); lo=NULL;
        int is_eos=0; for(int e=0;e<n_eos;e++) if(tk==eos_ids[e]) is_eos=1;
        if(is_eos){ limited=0; break; }
        double token_at=now_s();
        if(!gen)first_token_at=token_at;
        last_token_at=token_at;
        unsigned char *token=NULL;int token_n=0;
        if(decode_id_alloc(tk,&token,&token_n)){fprintf(stderr,"[decode] out of memory\n");exit(1);}
        if(token_n>(INT_MAX-3)/3){free(token);fprintf(stderr,"[decode] token is too large\n");exit(1);}
        int chunk_cap=token_n*3+3;
        unsigned char *chunk=(unsigned char*)malloc((size_t)chunk_cap);int chunk_n=0;
        if(!chunk||utf8_drain(sbuf,&sbn,token,token_n,chunk,chunk_cap,&chunk_n)<0){
            free(chunk);free(token);fprintf(stderr,"[decode] invalid output capacity\n");exit(1);
        }
        if(chunk_n>0)serve_data(q->id,(char*)chunk,chunk_n);
        free(chunk);free(token);
        gen++;
        while(!input_eof&&coli_stdin_readable()){
            int control=serve_read_req(stdin,stdout,NULL,q->id);
            if(control<0){input_eof=1;break;}
            if(control==1)stopped=1;
            if(control==3)cancelled=1;
        }
        if(cancelled||stopped){limited=0;break;}
        /* The next logits are not needed after the final requested token.
         * The published prompt snapshot, rather than the live decode state,
         * is restored on the next request, so stepping here would only run a
         * full discarded decode pass. */
        if(s == q->max_tok - 1) break;
        lo = step(m, &tk, 1, np+s);
    }
    free(lo); free(ids);
    if(cancelled){coli_serve_write_error(stdout,q->id,"CANCELLED");return input_eof?-1:0;}
    if(stopped)limited=0;
    unsigned char tail[3];int tail_n=0;
    if(utf8_finish(sbuf,&sbn,tail,sizeof tail,&tail_n)<0){
        fprintf(stderr,"[decode] invalid UTF-8 finalization\n");exit(1);
    }
    if(tail_n)serve_data(q->id,(char*)tail,tail_n);
    double wall_s=now_s()-request_started;
    double decode_s=gen>1?last_token_at-first_token_at:0.0;
    uint64_t hits=m->hits-hits_before,misses=m->miss-misses_before;
    Q38Timers timers=q38_tm_delta(&m->timers,&timers_before);
    ColiServeDone done={gen,q38_decode_rate(gen,decode_s),
                        q38_cache_hit_percent(hits,misses),rss_gb(),np,limited};
    coli_serve_write_done(stdout,q->id,&done);
    /* Stable dashboard contract.  Disk service and synchronous miss wait
     * intentionally overlap; all finer Qwen phases remain available through
     * COLI_TIMERS=1 without widening the shared frame. */
    char profile[256];int profile_bytes=q38_format_prof(profile,sizeof profile,wall_s,np,gen,&timers);
    if(profile_bytes>0)fwrite(profile,1,(size_t)profile_bytes,stdout);
    else fprintf(stderr,"[qwen38] internal error: PROF frame overflow\n");
    fflush(stdout);
    q38_tm_report_bank(&timers,"request");
    return input_eof?-1:0;
}

static void serve_loop(Model *m){
    coli_serve_binary_mode();
    setvbuf(stdin,NULL,_IONBF,0);
    int max_ctx=qwen38_max_ctx();
    if(max_ctx>m->c.max_positions)max_ctx=m->c.max_positions;
    if(!q38_serve_ensure_kv(m,max_ctx)){
        fprintf(stderr,"[serve] unable to allocate QSA state\n");return;
    }
    fputs("\x01\x01READY\x01\x01\n",stdout);
    printf("STAT 0 0.00 0.0 %.2f\n",rss_gb());
    fflush(stdout);
    for(;;){
        ServeReq q={0}; int r;
        do r=serve_read_req(stdin,stdout,&q,NULL); while(r!=2&&r>=0);
        if(r<0){q38_prefix_cache_release(m);return;}
        if(r==2){
            int status=serve_one(m,&q);free(q.payload);
            if(status<0){q38_prefix_cache_release(m);return;}
        }
    }
}

static int q38_reference_mode(const char *path,int serve_mode){
    if(serve_mode||!path)return 0;
    size_t length=strlen(path);
    return length>=5&&!strcmp(path+length-5,".json");
}

#ifndef QWEN38_TEST_SERVE
int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    if (getenv("OPENAI")) g_openai = 1;                       /* OpenAI-compatible output */
    const char *mv = getenv("MODEL");
    if (mv && *mv) snprintf(g_model, sizeof g_model, "%s", mv);
    int cap   = argc > 1 ? atoi(argv[1]) : 1;
    int bits  = argc > 2 ? atoi(argv[2]) : 8;
    /* cap < 1 leaves every layer cache empty, so expert_get finds no slot to
     * evict and waits for a publish that can never come. The old lru=0 fallback
     * turned that into a heap OOB instead; neither is a failure mode to ship. */
    if (cap < 1) { fprintf(stderr, "cache/layer must be >= 1 (got %d)\n", cap); return 1; }
    if (bits < 2 || bits > 8) { fprintf(stderr, "reserved bits argument must be 2..8 (got %d)\n", bits); return 1; }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";
    int serve_mode=getenv("SERVE")&&getenv("SERVE")[0]=='1';

    fprintf(stderr, "== Qwen3.8-Flash-Next native text engine | cache=%d/layer | CPU ==\n", cap);

    int is_ref=q38_reference_mode(refpath,serve_mode);

    int *prompt=NULL, *full=NULL, *out=NULL;
    int np=0, nfull=0, n_new=0;
    char *buf=NULL, *arena=NULL; jval *ref_root=NULL,*ref_logits=NULL;
    /* serve mode gets its prompts over the wire: skip the argv prompt file
     * entirely, or the default "ref.json" kills the engine before serve_loop
     * is ever reached — which is exactly how `coli` launches it (SERVE=1, no
     * prompt argument). */
    /* Load before model_init so malformed/missing tokenizer state cannot spend
     * minutes materializing a model that serve/text mode must then refuse. A
     * numeric reference run deliberately remains tokenizer-optional. */
    int tokenizer_status=-1;
    {
        const char *tokpath = getenv("TOK");
        if (tokpath && *tokpath) tokenizer_status=load_tokenizer(tokpath);
        else if (argc > 4 && argv[4] && *argv[4])
            tokenizer_status=load_tokenizer(argv[4]);
        else {
            char tpb[2048];snprintf(tpb,sizeof tpb,"%s/tokenizer.json",snap);
            tokenizer_status=load_tokenizer(tpb);
        }
    }
    if(tokenizer_status&&(serve_mode||!is_ref)){
        fprintf(stderr,serve_mode?
                "[serve] a complete tokenizer.json is required (put it in SNAP or set TOK)\n":
                "[enc] a complete tokenizer.json is required (put it in SNAP or set TOK)\n");
        free_tokenizer();return 1;
    }

    if (serve_mode) {
        /* no argv prompt to load */
    } else if (is_ref) {
        FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
        if (fseek(f,0,SEEK_END)) { perror(refpath); fclose(f); return 1; }
        long n=ftell(f);
        if (n<0 || n>(256L<<20) || fseek(f,0,SEEK_SET)) {
            fprintf(stderr, "%s: invalid or oversized reference file\n", refpath);
            fclose(f); return 1;
        }
        buf=malloc((size_t)n+1);
        if (!buf || fread(buf,1,(size_t)n,f)!=(size_t)n) {
            fprintf(stderr, "cannot read %s\n", refpath); free(buf); fclose(f); return 1;
        }
        buf[n]=0; fclose(f);
        ref_root = json_parse(buf, &arena);
        if (!ref_root || ref_root->t != J_OBJ) {
            fprintf(stderr, "%s: expected a JSON object\n", refpath); json_free(ref_root); free(buf); free(arena); return 1;
        }
        prompt = read_int_array(ref_root,"prompt_ids",&np);
        full   = read_int_array(ref_root,"full_ids",&nfull);
        if (nfull<=np) {
            fprintf(stderr, "%s: full_ids must extend prompt_ids\n", refpath); return 1;
        }
        for (int i=0;i<np;i++) if (prompt[i]!=full[i]) {
            fprintf(stderr, "%s: full_ids does not begin with prompt_ids at token %d\n",
                    refpath,i); return 1;
        }
        n_new  = nfull - np;
    } else {
        /* text-prompt mode: read file as raw text, encode in C */
        FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
        if (fseek(f,0,SEEK_END)) { perror(refpath); fclose(f); return 1; }
        long n=ftell(f);
        if (n<0 || n>(256L<<20) || fseek(f,0,SEEK_SET)) {
            fprintf(stderr, "%s: invalid or oversized prompt file\n", refpath);
            fclose(f); return 1;
        }
        char *txt=malloc((size_t)n+1);
        if (!txt || fread(txt,1,(size_t)n,f)!=(size_t)n) {
            fprintf(stderr, "cannot read %s\n", refpath); free(txt); fclose(f); return 1;
        }
        txt[n]=0; fclose(f);
        if (!g_tok) { fprintf(stderr, "[enc] no tokenizer loaded; cannot encode text. Put tokenizer.json in SNAP or set TOK.\n"); free(txt); return 1; }
        encode_text_n(txt,(size_t)n,&prompt,&np);
        free(txt);
        n_new = 64;
        const char *nnew_env=getenv("N_NEW");
        if (nnew_env && *nnew_env) {
            char *end=NULL; errno=0; long value=strtol(nnew_env,&end,10);
            if (errno || end==nnew_env || *end || value<1 || value>INT_MAX) {
                fprintf(stderr, "N_NEW must be an integer in [1,%d]\n",INT_MAX); return 1;
            }
            n_new=(int)value;
        }
        fprintf(stderr, "[enc] prompt tokens: %d | generating %d new tokens\n", np, n_new);
        if (getenv("ENC_DEBUG") && np <= 300) { fprintf(stderr, "[enc] prompt ids: "); for (int i=0;i<np;i++) fprintf(stderr, "%d ", prompt[i]); fprintf(stderr, "\n"); }
    }

    Model m; model_init(&m, snap, cap, bits);
    if(is_ref)ref_logits=read_reference_logits(ref_root,m.c.vocab);
    g_capture_last_logit=ref_logits!=NULL||getenv("DUMP")!=NULL;
    q38_telemetry_init(snap, &m);
    fprintf(stderr, "resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());

    /* coli serve mode: speak the gateway wire protocol instead of argv
     * generation. AFTER the tier init: serve sessions ride the VRAM experts
     * exactly like argv runs, and serve_loop never returns. */
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        if (!g_tok) { fprintf(stderr, "[serve] tokenizer.json required (put in SNAP or set TOK)\n");
            q38_model_free(&m); rt_destroy(); return 1; }
        serve_loop(&m);
        rt_save(g_q38_usage, 0);
        q38_prefix_cache_release(&m);
        q38_model_free(&m); free_tokenizer(); rt_destroy();
        return 0;
    }

    if (is_ref && getenv("PPL") && atoi(getenv("PPL")) == 1) {
        double nll; double t = now_s();
        int scored = tf_nll(&m, full, nfull, np, &nll);
        double dt = now_s() - t;
        double tot = m.hits + m.miss;
        printf("TF-NLL: %.4f nats/token over %d tokens | ppl = %.2f\n", nll, scored, exp(nll));
        printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss);
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
        rt_save(g_q38_usage,0);
        q38_model_free(&m); free(prompt); free(full); json_free(ref_root);
        free(buf); free(arena); free_tokenizer(); rt_destroy(); return 0;
    }

    q38_validate_ids(&m.c,prompt,np,"prompt_ids");
    if(is_ref) q38_validate_ids(&m.c,full,nfull,"full_ids");
    int total_tokens=q38_context_total(&m.c,np,n_new,"generation");
    out = malloc((size_t)total_tokens * sizeof(int));
    if(!out){fprintf(stderr,"OOM allocating generation token buffer\n");
        q38_model_free(&m); free(prompt); free(full); json_free(ref_root);
        free(buf); free(arena); free_tokenizer(); rt_destroy(); return 1;}
    /* timing + OpenAI id setup (before generation) */
    g_ttft = -1; g_gen_t0 = now_s();
    if (g_openai){
        g_oa_created = (long)time(NULL);
        snprintf(g_oa_id,sizeof g_oa_id,"chatcmpl-%ld%04d",g_oa_created,
                 q38_completion_nonce(now_s()));
    }
    /* streaming text: emit tokens as they are produced (text mode + tokenizer only) */
    if (!is_ref && g_tok && !getenv("NOSTREAM")) {
        g_stream = 1; g_sbn = 0;
        if (g_openai){
            char *model=json_escape_alloc((const unsigned char*)g_model,strlen(g_model));
            char *jb=model?q38_format_alloc(
              "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
              "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}",
              g_oa_id,g_oa_created,model):NULL;
            free(model);if(!jb){fprintf(stderr,"[openai] out of memory formatting response\n");exit(1);}
            sse_chunk(jb);free(jb);
        } else {
            fprintf(stderr, "Generated (%d new tokens):\nText : ", n_new); fflush(stderr);
        }
    }
    double t = now_s();
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    /* DUMP=<path>: write last-token logits (raw float32, vocab) for a torch-free
     * cosine comparison against tools/_ref_dn.py --dump. */
    if (g_last_logit && getenv("DUMP")) {
        const char *dp = getenv("DUMP");
        FILE *df = fopen(dp && *dp ? dp : "qwen38_logits.f32", "wb");
        if (df) { fwrite(g_last_logit, sizeof(float), (size_t)m.c.vocab, df); fclose(df);
                  fprintf(stderr, "[dump] wrote %d logits -> %s\n", m.c.vocab, dp && *dp ? dp : "qwen38_logits.f32"); }
        else fprintf(stderr, "[dump] cannot open %s\n", dp ? dp : "qwen38_logits.f32");
    }

    int ref_match = 0,ref_numeric_match=1;
    if (is_ref) {
        int match = 0;
        printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
        printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
        if (g_tok) { printf("Text      : "); print_decoded(stdout,out,np,nfull); printf("\n"); }
        printf("\nMatching tokens: %d/%d\n", match, n_new);
        if(ref_logits)ref_numeric_match=compare_reference_logits(g_last_logit,ref_logits);
        ref_match = match;
    } else {
    if (g_openai) {
        emit_openai_result(out, np, n_new, g_stream);
    } else if (g_stream) {
        stream_flush(); fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "\nGenerated (%d new tokens):\n", n_new);
        if (g_tok) { fprintf(stderr,"Text      : "); print_decoded(stderr,out,np,np+n_new); fprintf(stderr,"\n"); }
        else { fprintf(stderr, "Ids       : "); for (int i=np;i<np+n_new;i++) fprintf(stderr, "%d ", out[i]); fprintf(stderr, "\n"); }
    }
    }
    double tot = m.hits + m.miss;
    if (g_ttft >= 0) fprintf(stderr, "TTFT: %.2f s (time to first token)\n", g_ttft);
    tm_report(&m);
    fprintf(stderr, "\nPEAK RSS: %.2f GB\n", rss_gb());
    fprintf(stderr, "Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    fprintf(stderr, "Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    rt_save(g_q38_usage, 0);
    free(g_last_logit); g_last_logit=NULL;
    q38_model_free(&m); free(out); free(prompt); free(full); json_free(ref_root);
    free(buf); free(arena); free_tokenizer(); rt_destroy();
    /* Oracle mode is a gate, not a report: a mismatch must fail the caller.
     * inkling.c does the same (`return (match == ngen) ? 0 : 1;`) and its CI
     * job relies on it — without this, tools/make_qwen38_oracle.py could be
     * wired into a workflow that stays green through any regression. */
    if (is_ref) return ref_match == n_new && ref_numeric_match ? 0 : 1;
    return 0;
}
#endif /* QWEN38_TEST_SERVE */
#endif /* !QWEN38_NO_MAIN || QWEN38_TEST_SERVE */

static int q38_u64_mul(uint64_t left,uint64_t right,uint64_t *out){
    if(!out||(right&&left>UINT64_MAX/right))return -1;*out=left*right;return 0;
}
static int q38_u64_add(uint64_t *total,uint64_t value){
    if(!total||*total>UINT64_MAX-value)return -1;*total+=value;return 0;
}

static int q38_segment_tensor_storage_valid(const st_tensor *tensor){
    uint64_t expected_bytes;
    if(!tensor||tensor->dtype<0||tensor->dtype>6||tensor->numel<1||
       tensor->nbytes<1||q38_u64_mul((uint64_t)tensor->numel,
                                    (uint64_t)st_dtype_esz(tensor->dtype),
                                    &expected_bytes)||
       expected_bytes>INT64_MAX)return 0;
    return tensor->nbytes==(int64_t)expected_bytes;
}

enum {
    Q38_EXPERT_FP8_BLOCK = 1u << 0,
    Q38_EXPERT_FP8_EXPANDED = 1u << 1,
    Q38_EXPERT_BF16 = 1u << 2,
    Q38_EXPERT_F16 = 1u << 3,
    Q38_EXPERT_F32 = 1u << 4
};

/* Return the bytes retained by one cache matrix after loading.  The source
 * dtype alone is not enough: native FP8 retains its F32 block scales, native
 * BF16 retains two-byte elements, and both opt-out modes retain expanded F32.
 * Keeping this calculation beside the loader's representation choices makes
 * Segment's memory contract describe the allocation it actually performs. */
static int q38_segment_matrix_bytes(Model *m,const char *weight_name,
                                    int rows,int cols,uint64_t *bytes,
                                    unsigned *numeric_kinds){
    uint64_t elements;
    if(!m||!weight_name||!bytes||!numeric_kinds||rows<1||cols<1||
       q38_u64_mul((uint64_t)rows,(uint64_t)cols,&elements)||
       elements>INT64_MAX)return -1;
    st_tensor *weight=st_find(&m->S,weight_name);
    if(!weight||!q38_segment_tensor_storage_valid(weight)||
       (weight->dtype>2&&weight->dtype!=4)||weight->rank!=2||
       weight->shape[0]!=rows||weight->shape[1]!=cols||
       weight->numel!=(int64_t)elements)return -1;
    if(weight->dtype==4){
        char scale_name[340];
        int length=snprintf(scale_name,sizeof scale_name,"%s_scale_inv",weight_name);
        int64_t block_rows=fp8_nblk(rows),block_cols=fp8_nblk(cols);
        st_tensor *scale=length>0&&(size_t)length<sizeof scale_name?
                         st_find(&m->S,scale_name):NULL;
        if(!scale||scale->dtype<0||scale->dtype>2||scale->rank!=2||
           scale->shape[0]!=block_rows||scale->shape[1]!=block_cols||
           scale->numel!=block_rows*block_cols||
           !q38_segment_tensor_storage_valid(scale))return -1;
        if(m->native_fp8){
            uint64_t scale_bytes;
            if(q38_u64_mul((uint64_t)scale->numel,sizeof(float),&scale_bytes)||
               q38_u64_add(&elements,scale_bytes))return -1;
            *bytes=elements;*numeric_kinds|=Q38_EXPERT_FP8_BLOCK;
        }else{
            if(q38_u64_mul(elements,sizeof(float),bytes))return -1;
            *numeric_kinds|=Q38_EXPERT_FP8_EXPANDED;
        }
        return 0;
    }
    if(weight->dtype<0||weight->dtype>2)return -1;
    uint64_t element_bytes=weight->dtype==0&&m->native_bf16?
                           sizeof(uint16_t):sizeof(float);
    if(q38_u64_mul(elements,element_bytes,bytes))return -1;
    *numeric_kinds|=weight->dtype==0?Q38_EXPERT_BF16:
                    weight->dtype==1?Q38_EXPERT_F16:Q38_EXPERT_F32;
    return 0;
}

static int q38_segment_fused_matrix_bytes(Model *m,const st_tensor *tensor,
                                          int64_t elements,uint64_t *bytes,
                                          unsigned *numeric_kinds){
    if(!m||!bytes||!numeric_kinds||!tensor||tensor->dtype<0||
       tensor->dtype>2||elements<1||elements>tensor->numel||
       !q38_segment_tensor_storage_valid(tensor))return -1;
    uint64_t element_bytes=tensor->dtype==0&&m->native_bf16?
                           sizeof(uint16_t):sizeof(float);
    if(q38_u64_mul((uint64_t)elements,element_bytes,bytes))return -1;
    *numeric_kinds|=tensor->dtype==0?Q38_EXPERT_BF16:
                    tensor->dtype==1?Q38_EXPERT_F16:Q38_EXPERT_F32;
    return 0;
}

/* Sum the maximum possible resident expert slot for every layer in the loaded
 * range.  Individual-expert containers are allowed to be heterogeneous, so a
 * layer is priced at its largest expert rather than assuming expert zero is
 * representative.  Fused containers have one dtype/geometry for every expert. */
static int q38_segment_expert_layout(Model *m,uint32_t begin,uint32_t end,
                                     uint64_t *bytes_per_capacity,
                                     uint64_t *fixed_scale_bytes,
                                     unsigned *numeric_kinds){
    if(!m||!bytes_per_capacity||!fixed_scale_bytes||!numeric_kinds||begin>=end||
       end>(uint32_t)m->c.layers)return -1;
    Cfg *c=&m->c;uint64_t range_bytes=0,range_scales=0;unsigned kinds=0;
    for(uint32_t layer=begin;layer<end;layer++){
        char name[320];q38_name(m,name,sizeof name,(int)layer,
                                "mlp.experts.gate_up_proj");
        st_tensor *gate_up=st_find(&m->S,name);uint64_t layer_bytes=0;
        if(gate_up){
            if(gate_up->rank!=3||gate_up->shape[0]!=c->experts||
               gate_up->shape[1]!=(int64_t)2*c->inter||
               gate_up->shape[2]!=c->hidden)return -1;
            uint64_t part;
            if(q38_segment_fused_matrix_bytes(m,gate_up,
                 (int64_t)2*c->inter*c->hidden,&part,&kinds)||
               q38_u64_add(&layer_bytes,part))return -1;
            q38_name(m,name,sizeof name,(int)layer,"mlp.experts.down_proj");
            st_tensor *down=st_find(&m->S,name);
            if(!down||down->rank!=3||down->shape[0]!=c->experts||
               down->shape[1]!=c->hidden||down->shape[2]!=c->inter||
               q38_segment_fused_matrix_bytes(m,down,
                    (int64_t)c->hidden*c->inter,&part,&kinds)||
               q38_u64_add(&layer_bytes,part))return -1;
        }else{
            const char *projection[3]={"gate_proj","up_proj","down_proj"};
            int rows[3]={c->inter,c->inter,c->hidden};
            int cols[3]={c->hidden,c->hidden,c->inter};
            int all_native_fp8=m->native_fp8;
            for(int expert=0;expert<c->experts;expert++){
                uint64_t expert_bytes=0;
                for(int projection_index=0;projection_index<3;projection_index++){
                    char suffix[192];
                    int length=snprintf(suffix,sizeof suffix,
                        "mlp.experts.%d.%s.weight",expert,
                        projection[projection_index]);
                    if(length<0||(size_t)length>=sizeof suffix)return -1;
                    q38_name(m,name,sizeof name,(int)layer,suffix);
                    st_tensor *weight=st_find(&m->S,name);
                    if(!weight||weight->dtype!=4)all_native_fp8=0;
                    uint64_t part;
                    if(q38_segment_matrix_bytes(m,name,rows[projection_index],
                         cols[projection_index],&part,&kinds)||
                       q38_u64_add(&expert_bytes,part))return -1;
                }
                if(expert_bytes>layer_bytes)layer_bytes=expert_bytes;
            }
            if(all_native_fp8){
                uint64_t scale_count=(uint64_t)fp8_nblk(c->inter)*
                                     (uint64_t)fp8_nblk(c->hidden);
                uint64_t per_slot_scales,layer_scale_bank;
                if(q38_u64_mul(3u,scale_count,&per_slot_scales)||
                   q38_u64_mul(per_slot_scales,sizeof(float),&per_slot_scales)||
                   layer_bytes<per_slot_scales||
                   q38_u64_mul(per_slot_scales,(uint64_t)c->experts,
                               &layer_scale_bank)||
                   q38_u64_add(&range_scales,layer_scale_bank))return -1;
                /* Eligible checkpoints borrow normalized scales from one
                 * per-layer bank, so only raw FP8 payload belongs to a slot.
                 * Charging the full bank even for a repacked checkpoint that
                 * later falls back is conservative and keeps tight Segment
                 * memory limits safe. */
                layer_bytes-=per_slot_scales;
            }
        }
        if(!layer_bytes||q38_u64_add(&range_bytes,layer_bytes))return -1;
    }
    *bytes_per_capacity=range_bytes;*fixed_scale_bytes=range_scales;
    *numeric_kinds=kinds;return 0;
}

static int q38_segment_cache_capacity(uint64_t bytes_per_capacity,
                                      uint64_t fixed_scale_bytes,int experts,
                                      uint64_t memory_limit,int *capacity){
    if(!bytes_per_capacity||experts<1||!capacity)return -1;
    if(!memory_limit){*capacity=1;return 0;}
    if(memory_limit<=fixed_scale_bytes)return -1;
    uint64_t slots=(memory_limit-fixed_scale_bytes)/bytes_per_capacity;
    if(!slots)return -1;
    if(slots>(uint64_t)experts)slots=(uint64_t)experts;
    *capacity=(int)slots;return 0;
}

static void q38_segment_numeric_class(char *out,size_t capacity,
                                      unsigned numeric_kinds){
    const char *known=NULL;
    switch(numeric_kinds){
        case Q38_EXPERT_FP8_BLOCK:
            known="qwen38/fp8-block-f32dot-f64blocksum/cpu-v1";break;
        case Q38_EXPERT_FP8_EXPANDED:
            known="qwen38/fp8-expanded-f32dot/cpu-v1";break;
        case Q38_EXPERT_BF16:
            known="qwen38/bf16-values-f32dot/cpu-v1";break;
        case Q38_EXPERT_F16:
            known="qwen38/f16-expanded-f32dot/cpu-v1";break;
        case Q38_EXPERT_F32:
            known="qwen38/f32dot/cpu-v1";break;
        default:
            /* The bitset is part of this opaque compatibility identifier. In
             * particular, FP8_BLOCK and FP8_EXPANDED are different bits, so a
             * heterogeneous range cannot silently change its reduction rules
             * while continuing to claim snapshot/numeric compatibility. */
            snprintf(out,capacity,"qwen38/mixed-expert-arithmetic-%02x/cpu-v1",
                     numeric_kinds);return;
    }
    snprintf(out,capacity,"%s",known);
}

static int q38_segment_cache_resize(Model *m,int capacity){
    if(!m||capacity<1)return -1;
    for(int layer=m->range_begin;layer<m->range_end;layer++){
        LCache *cache=&m->cache[layer];
        if(cache->n)return -1;
        Slot *slots=(Slot*)calloc((size_t)capacity,sizeof(*slots));
        if(!slots)return -1;
        free(cache->slots);cache->slots=slots;cache->cap=capacity;
    }
    return 0;
}

static int q38_segment_session_state_bytes(const Cfg *c,uint32_t begin,
                                            uint32_t end,uint32_t context,
                                            uint64_t *out){
    if(!c||!out||begin>=end||end>(uint32_t)c->layers||!context)return -1;
    uint64_t total=0,cells,bytes;
    for(uint32_t layer=begin;layer<end;layer++){
        if(c->is_attn[layer]){
            if(q38_u64_mul((uint64_t)c->kv_heads,context,&cells)||
               q38_u64_mul(cells,(uint64_t)c->head_dim,&cells)||
               q38_u64_mul(cells,2u*sizeof(float),&bytes)||
               q38_u64_add(&total,bytes)||
               q38_u64_mul(context,(uint64_t)c->idx_dim,&cells)||
               q38_u64_mul(cells,sizeof(float),&bytes)||q38_u64_add(&total,bytes))return -1;
        }else{
            if(q38_u64_mul((uint64_t)c->dn_vheads,(uint64_t)c->dn_kdim,&cells)||
               q38_u64_mul(cells,(uint64_t)c->dn_vdim,&cells)||
               q38_u64_mul(cells,sizeof(float),&bytes)||q38_u64_add(&total,bytes)||
               q38_u64_mul((uint64_t)c->dn_conv_dim,(uint64_t)(c->dn_convk-1),&cells)||
               q38_u64_mul(cells,sizeof(float),&bytes)||q38_u64_add(&total,bytes))return -1;
        }
    }
    if(c->ple_layer>=0&&(uint32_t)c->ple_layer>=begin&&(uint32_t)c->ple_layer<end){
        if(q38_u64_mul((uint64_t)c->hc_width,(uint64_t)(c->ple_convk-1),&cells)||
           q38_u64_mul(cells,(uint64_t)c->ngram_size,&cells)||
           q38_u64_mul(cells,sizeof(float),&bytes)||q38_u64_add(&total,bytes)||
           q38_u64_add(&total,2u*sizeof(int64_t)+sizeof(int)))return -1;
    }
    *out=total;return 0;
}

#ifdef COLI_SEGMENT_ADAPTER
/* ---------- engine-owned Segment adapter ------------------------------ */

typedef struct {
    Model model;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
} Qwen38SegmentEngine;

typedef struct {
    Qwen38SegmentEngine *engine;
    float **K, **V, **IK, **DN_rec, **DN_conv;
    int64_t ple_history[2];
    int ple_history_len;
    float *PLE_conv_state;
    uint32_t context_tokens, position;
} Qwen38SegmentSession;

static void qwen38_segment_session_state_free(Qwen38SegmentSession *s) {
    if (!s) return;
    Cfg *c=&s->engine->model.c;
    for (uint32_t i=s->engine->layer_begin;i<s->engine->layer_end;i++) {
        free(s->K ? s->K[i] : NULL); free(s->V ? s->V[i] : NULL); free(s->IK ? s->IK[i] : NULL);
        free(s->DN_rec ? s->DN_rec[i] : NULL); free(s->DN_conv ? s->DN_conv[i] : NULL);
    }
    free(s->K); free(s->V); free(s->IK); free(s->DN_rec); free(s->DN_conv);
    (void)c;
    free(s->PLE_conv_state);
}

static int qwen38_segment_engine_open(
    void **engine_impl, ColiSegmentCapabilities *capabilities,
    const ColiSegmentEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options || !options->model_dir)
        return coli_segment_adapter_error(error,error_size,"invalid Qwen3.8 Segment open");
    *engine_impl=NULL;
    if (options->backend_mask && (options->backend_mask & ~COLI_SEGMENT_CAP_CPU))
        return coli_segment_adapter_error(error,error_size,"Qwen3.8 Segment currently supports CPU");
    Cfg probe; q38_load_cfg(&probe,options->model_dir); q38_validate_cfg(&probe);
    int layers=probe.layers, maxctx=probe.max_positions;
    uint32_t end=options->layer_end ? options->layer_end : (uint32_t)layers;
    if (options->layer_begin>=end || end>(uint32_t)layers){free(probe.is_attn);
        return coli_segment_adapter_error(error,error_size,"Qwen3.8 Segment range exceeds model");
    }
    if (!options->context_tokens || options->context_tokens>(uint32_t)maxctx || options->context_tokens>INT_MAX){free(probe.is_attn);
        return coli_segment_adapter_error(error,error_size,"Qwen3.8 Segment context exceeds model limit");
    }
    free(probe.is_attn);
    Qwen38SegmentEngine *e=(Qwen38SegmentEngine*)calloc(1,sizeof(*e));
    if(!e) return coli_segment_adapter_error(error,error_size,"out of memory opening Qwen3.8 Segment");
    e->layer_begin=options->layer_begin; e->layer_end=end; e->context_tokens=options->context_tokens;
    if(pthread_mutex_init(&e->run_lock,NULL)){free(e);return coli_segment_adapter_error(error,error_size,"cannot initialize Qwen3.8 Segment lock");}
    /* Open with a single empty cache slot so tensor metadata and the effective
     * native-storage switches are available before honoring the caller's
     * budget. No expert is loaded during model_init_range, so resizing here
     * cannot discard data or perturb LRU state. */
    model_init_range(&e->model,options->model_dir,1,8,
                     (int)e->layer_begin,(int)e->layer_end,0,0);
    uint64_t bytes_per_capacity=0,fixed_scale_bytes=0;
    unsigned numeric_kinds=0;int cap=1;
    if(q38_segment_expert_layout(&e->model,e->layer_begin,e->layer_end,
                                 &bytes_per_capacity,&fixed_scale_bytes,
                                 &numeric_kinds)){
        q38_model_free(&e->model);pthread_mutex_destroy(&e->run_lock);free(e);
        return coli_segment_adapter_error(error,error_size,
            "Qwen3.8 Segment checkpoint has an unsupported expert layout");
    }
    if(q38_segment_cache_capacity(bytes_per_capacity,fixed_scale_bytes,
                                  e->model.c.experts,
                                  options->memory_limit_bytes,&cap)){
        q38_model_free(&e->model);pthread_mutex_destroy(&e->run_lock);free(e);
        return coli_segment_adapter_error(error,error_size,
            "Qwen3.8 Segment memory limit cannot hold one expert per loaded layer");
    }
    if(cap!=1&&q38_segment_cache_resize(&e->model,cap)){
        q38_model_free(&e->model);pthread_mutex_destroy(&e->run_lock);free(e);
        return coli_segment_adapter_error(error,error_size,
            "out of memory sizing Qwen3.8 Segment expert cache");
    }
    e->model.max_t=(int)e->context_tokens; e->model.kv_cap=(int)e->context_tokens;
    memset(capabilities,0,sizeof(*capabilities)); capabilities->struct_size=sizeof(*capabilities);
    capabilities->abi_version=COLI_SEGMENT_ABI_VERSION;
    capabilities->flags=COLI_SEGMENT_CAP_TOKEN_IDS|COLI_SEGMENT_CAP_SNAPSHOT|COLI_SEGMENT_CAP_RANGE_NATIVE|
                        COLI_SEGMENT_CAP_MULTI_SESSION|COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,sizeof(capabilities->engine_id),"qwen38");
    coli_segment_capability_string(capabilities->state_schema,sizeof(capabilities->state_schema),"qwen38/hyper-kv-deltanet-ple-f32-v1");
    q38_segment_numeric_class(capabilities->numeric_class,
                              sizeof(capabilities->numeric_class),numeric_kinds);
    capabilities->state_dtype=COLI_SEGMENT_DTYPE_F32; capabilities->state_width=(uint32_t)e->model.c.hc_width;
    capabilities->max_batch_rows=128; capabilities->max_context_tokens=(uint32_t)maxctx; capabilities->num_layers=(uint32_t)layers;
    *engine_impl=e; return 0;
}

static void qwen38_segment_engine_destroy(void *impl) {
    Qwen38SegmentEngine *e=(Qwen38SegmentEngine*)impl; if(!e)return;
    q38_model_free(&e->model); pthread_mutex_destroy(&e->run_lock); free(e);
}

static int qwen38_segment_session_create(void *impl,void **out,const ColiSegmentSessionOptions *options,char *error,size_t error_size) {
    Qwen38SegmentEngine *e=(Qwen38SegmentEngine*)impl;
    if(!e||!out||!options)
        return coli_segment_adapter_error(error,error_size,"invalid Qwen3.8 Segment session context");
    *out=NULL;
    if(!options->context_tokens||options->context_tokens>e->context_tokens)
        return coli_segment_adapter_error(error,error_size,"invalid Qwen3.8 Segment session context");
    uint64_t state_bytes=0;
    if(q38_segment_session_state_bytes(&e->model.c,e->layer_begin,e->layer_end,
                                       options->context_tokens,&state_bytes))
        return coli_segment_adapter_error(error,error_size,"Qwen3.8 session state size overflow");
    if(options->memory_limit_bytes&&state_bytes>options->memory_limit_bytes)
        return coli_segment_adapter_error(error,error_size,"Qwen3.8 session state exceeds memory limit");
    Qwen38SegmentSession *s=(Qwen38SegmentSession*)calloc(1,sizeof(*s)); if(!s)return coli_segment_adapter_error(error,error_size,"out of memory creating Qwen3.8 session");
    s->engine=e;s->context_tokens=options->context_tokens;
    int n=e->model.c.layers; s->K=(float**)calloc((size_t)n,sizeof(float*));s->V=(float**)calloc((size_t)n,sizeof(float*));s->IK=(float**)calloc((size_t)n,sizeof(float*));
    s->DN_rec=(float**)calloc((size_t)n,sizeof(float*));s->DN_conv=(float**)calloc((size_t)n,sizeof(float*));
    if(!s->K||!s->V||!s->IK||!s->DN_rec||!s->DN_conv)goto oom;
    Cfg *c=&e->model.c;
    for(uint32_t i=e->layer_begin;i<e->layer_end;i++){
        size_t cells,bytes;
        if(c->is_attn[i]){
            if(coli_segment_size_mul((size_t)c->kv_heads,s->context_tokens,&cells)||coli_segment_size_mul(cells,(size_t)c->head_dim,&cells))goto oom;
            if(coli_segment_size_mul(cells,sizeof(float),&bytes))goto oom;
            s->K[i]=(float*)calloc(cells,sizeof(float));s->V[i]=(float*)calloc(cells,sizeof(float));
            if(coli_segment_size_mul((size_t)s->context_tokens,(size_t)c->idx_dim,&cells)||coli_segment_size_mul(cells,sizeof(float),&bytes))goto oom;s->IK[i]=(float*)calloc(cells,sizeof(float));
            if(!s->K[i]||!s->V[i]||!s->IK[i])goto oom;
        }else{
            if(coli_segment_size_mul((size_t)c->dn_vheads,(size_t)c->dn_kdim,&cells)||coli_segment_size_mul(cells,(size_t)c->dn_vdim,&cells))goto oom;s->DN_rec[i]=(float*)calloc(cells,sizeof(float));
            if(coli_segment_size_mul(cells,sizeof(float),&bytes))goto oom;
            if(coli_segment_size_mul((size_t)c->dn_conv_dim,(size_t)c->dn_convk-1,&cells)||coli_segment_size_mul(cells,sizeof(float),&bytes))goto oom;s->DN_conv[i]=(float*)calloc(cells,sizeof(float));
            if(!s->DN_rec[i]||!s->DN_conv[i])goto oom;
        }
    }
    if(c->ple_layer>=0&&(uint32_t)c->ple_layer>=e->layer_begin&&(uint32_t)c->ple_layer<e->layer_end){
        size_t cells,bytes;if(coli_segment_size_mul((size_t)c->hc_width,(size_t)c->ple_convk-1,&cells)||coli_segment_size_mul(cells,(size_t)c->ngram_size,&cells)||coli_segment_size_mul(cells,sizeof(float),&bytes))goto oom;
        s->PLE_conv_state=(float*)calloc(cells,sizeof(float));if(!s->PLE_conv_state)goto oom;
    }
    *out=s;return 0;
oom:qwen38_segment_session_state_free(s);free(s);return coli_segment_adapter_error(error,error_size,"out of memory allocating Qwen3.8 state");
}

static void qwen38_segment_session_destroy(void *impl){Qwen38SegmentSession *s=(Qwen38SegmentSession*)impl;if(!s)return;qwen38_segment_session_state_free(s);free(s);}

static int qwen38_segment_session_run(void *impl,const ColiSegmentRunRequest *r,char *error,size_t error_size){
    Qwen38SegmentSession *s=(Qwen38SegmentSession*)impl; Qwen38SegmentEngine *e=s?s->engine:NULL; Cfg *c=e?&e->model.c:NULL;
    if(!s||!r||r->position!=s->position)return coli_segment_adapter_error(error,error_size,"Qwen3.8 Segment requires contiguous positions");
    if(!r->rows||r->rows>128||r->rows>s->context_tokens||
       r->position>s->context_tokens-r->rows||!r->input||!r->output||
       !r->token_ids||r->token_count!=r->rows)
        return coli_segment_adapter_error(error,error_size,"invalid Qwen3.8 Segment run request");
    size_t cells,bytes;if(coli_segment_size_mul((size_t)r->rows,(size_t)c->hc_width,&cells)||coli_segment_size_mul(cells,sizeof(float),&bytes)||r->input_bytes!=bytes||r->output_bytes!=bytes)
        return coli_segment_adapter_error(error,error_size,"Qwen3.8 Segment activation size mismatch");
    for(uint32_t i=0;i<r->rows;i++)if(r->token_ids[i]<0||r->token_ids[i]>=c->vocab)return coli_segment_adapter_error(error,error_size,"Qwen3.8 Segment token id outside vocabulary");
    if(r->should_cancel&&r->should_cancel(r->cancel_user_data))return coli_segment_adapter_error(error,error_size,"Qwen3.8 Segment run cancelled");
    if(r->output!=r->input)memcpy(r->output,r->input,bytes);
    pthread_mutex_lock(&e->run_lock); Model *m=&e->model;
    m->K=s->K;m->V=s->V;m->IK=s->IK;m->DN_rec=s->DN_rec;m->DN_conv=s->DN_conv;m->ple_history=s->ple_history;m->ple_history_len=s->ple_history_len;m->PLE_conv_state=s->PLE_conv_state;m->kv_cap=(int)s->context_tokens;m->kv_len=(int)s->position;
    q38_layers_forward_range(m,(float*)r->output,r->token_ids,(int)r->rows,(int)r->position,(int)e->layer_begin,(int)e->layer_end);
    s->ple_history_len=m->ple_history_len;m->K=NULL;m->V=NULL;m->IK=NULL;m->DN_rec=NULL;m->DN_conv=NULL;m->ple_history=NULL;m->PLE_conv_state=NULL;m->kv_len=0;
    pthread_mutex_unlock(&e->run_lock);s->position+=r->rows;return 0;
}

static int qwen38_segment_spans(Qwen38SegmentSession *s,uint32_t position,ColiSegmentStateSpan **out,size_t *count,char *error,size_t error_size){
    Cfg *c=&s->engine->model.c;
    if(position>s->context_tokens)return coli_segment_adapter_error(error,error_size,"Qwen3.8 snapshot position exceeds context");
    size_t cap=0;
    for(uint32_t i=s->engine->layer_begin;i<s->engine->layer_end;i++){
        size_t add=2;
        if(c->is_attn[i]){
            /* One indexer span plus one K and one V span for every KV head. */
            if((uint64_t)c->kv_heads>(SIZE_MAX-1)/2)
                return coli_segment_adapter_error(error,error_size,"Qwen3.8 state span count overflow");
            add=2*(size_t)c->kv_heads+1;
        }
        if(add>SIZE_MAX-cap){return coli_segment_adapter_error(error,error_size,"Qwen3.8 state span count overflow");}
        cap+=add;
    }
    if(c->ple_layer>=0&&(uint32_t)c->ple_layer>=s->engine->layer_begin&&(uint32_t)c->ple_layer<s->engine->layer_end){if(cap>SIZE_MAX-3)return coli_segment_adapter_error(error,error_size,"Qwen3.8 state span count overflow");cap+=3;}
    if(cap>SIZE_MAX/sizeof(ColiSegmentStateSpan))return coli_segment_adapter_error(error,error_size,"Qwen3.8 state span count overflow");
    ColiSegmentStateSpan *v=(ColiSegmentStateSpan*)calloc(cap,sizeof(*v));if(cap&&!v)return coli_segment_adapter_error(error,error_size,"out of memory describing Qwen3.8 state");size_t n=0;
    for(uint32_t i=s->engine->layer_begin;i<s->engine->layer_end;i++)if(c->is_attn[i]){
        size_t cells,rb,stride;
        if(coli_segment_size_mul((size_t)position,(size_t)c->head_dim,&cells)||coli_segment_size_mul(cells,sizeof(float),&rb)||coli_segment_size_mul((size_t)s->context_tokens,(size_t)c->head_dim,&stride)){free(v);return coli_segment_adapter_error(error,error_size,"Qwen3.8 attention state size overflow");}
        for(int h=0;h<c->kv_heads;h++){v[n++]=(ColiSegmentStateSpan){s->K[i]+(size_t)h*stride,rb};v[n++]=(ColiSegmentStateSpan){s->V[i]+(size_t)h*stride,rb};}
        if(coli_segment_size_mul((size_t)position,(size_t)c->idx_dim,&cells)||coli_segment_size_mul(cells,sizeof(float),&cells)){free(v);return coli_segment_adapter_error(error,error_size,"Qwen3.8 indexer state size overflow");} v[n++]=(ColiSegmentStateSpan){s->IK[i],cells};
    }else{
        size_t cells;
        if(coli_segment_size_mul((size_t)c->dn_vheads,(size_t)c->dn_kdim,&cells)||coli_segment_size_mul(cells,(size_t)c->dn_vdim,&cells)||coli_segment_size_mul(cells,sizeof(float),&cells)){free(v);return coli_segment_adapter_error(error,error_size,"Qwen3.8 recurrent state size overflow");} v[n++]=(ColiSegmentStateSpan){s->DN_rec[i],cells};
        if(coli_segment_size_mul((size_t)c->dn_conv_dim,(size_t)c->dn_convk-1,&cells)||coli_segment_size_mul(cells,sizeof(float),&cells)){free(v);return coli_segment_adapter_error(error,error_size,"Qwen3.8 convolution state size overflow");} v[n++]=(ColiSegmentStateSpan){s->DN_conv[i],cells};
    }
    if(c->ple_layer>=0&&(uint32_t)c->ple_layer>=s->engine->layer_begin&&(uint32_t)c->ple_layer<s->engine->layer_end){v[n++]=(ColiSegmentStateSpan){&s->ple_history_len,sizeof(s->ple_history_len)};v[n++]=(ColiSegmentStateSpan){s->ple_history,sizeof(s->ple_history)};size_t cells;if(coli_segment_size_mul((size_t)c->hc_width,(size_t)c->ple_convk-1,&cells)||coli_segment_size_mul(cells,(size_t)c->ngram_size,&cells)||coli_segment_size_mul(cells,sizeof(float),&cells)){free(v);return coli_segment_adapter_error(error,error_size,"Qwen3.8 PLE state size overflow");}v[n++]=(ColiSegmentStateSpan){s->PLE_conv_state,cells};}
    if(n!=cap){free(v);return coli_segment_adapter_error(error,error_size,"Qwen3.8 state span topology mismatch");}
    *out=v;*count=n;return 0;
}

static int qwen38_segment_session_snapshot(void *impl,ColiSegmentWriteFn wf,void *wd,char *error,size_t es){Qwen38SegmentSession *s=(Qwen38SegmentSession*)impl;ColiSegmentStateSpan *sp=NULL;size_t n=0,bytes=0;if(!s||qwen38_segment_spans(s,s?s->position:0,&sp,&n,error,es)||coli_segment_spans_size(sp,n,&bytes)){free(sp);return coli_segment_adapter_error(error,es,"Qwen3.8 snapshot size overflow");}ColiSegmentSnapshotHeader h;coli_segment_snapshot_header_init(&h,"qwen38",s->engine->layer_begin,s->engine->layer_end,s->context_tokens,s->position,bytes,coli_segment_spans_hash(sp,n));int rc=coli_segment_stream_write(wf,wd,&h,sizeof(h),error,es);if(!rc)rc=coli_segment_spans_write(sp,n,wf,wd,error,es);free(sp);return rc;}

static int qwen38_segment_payload_valid(
    const ColiSegmentStateSpan *spans,size_t count,
    const unsigned char *payload,size_t payload_bytes,
    void *user_data,char *error,size_t error_size){
    Qwen38SegmentSession *s=(Qwen38SegmentSession*)user_data;
    const unsigned char *cursor=payload; size_t consumed=0;
    int found_len=0,found_history=0;
    int history_len=0; int64_t history[2]={0,0};
    for(size_t i=0;i<count;i++){
        if(spans[i].size>payload_bytes-consumed)
            return coli_segment_adapter_error(error,error_size,"Qwen3.8 snapshot payload layout is invalid");
        if(spans[i].data==&s->ple_history_len&&spans[i].size==sizeof(history_len)){
            memcpy(&history_len,cursor,sizeof(history_len));found_len=1;
        }else if(spans[i].data==s->ple_history&&spans[i].size==sizeof(history)){
            memcpy(history,cursor,sizeof(history));found_history=1;
        }
        cursor+=spans[i].size;consumed+=spans[i].size;
    }
    if(!s->PLE_conv_state)return 0;
    Cfg *c=&s->engine->model.c;
    if(!found_len||!found_history||history_len<0||history_len>2||
       (history_len>0&&(history[0]<0||history[0]>=c->vocab))||
       (history_len>1&&(history[1]<0||history[1]>=c->vocab)))
        return coli_segment_adapter_error(error,error_size,"Qwen3.8 snapshot PLE history is invalid");
    return 0;
}

static int qwen38_segment_session_restore(void *impl,ColiSegmentReadFn rf,void *rd,char *error,size_t es){
    Qwen38SegmentSession *s=(Qwen38SegmentSession*)impl; ColiSegmentSnapshotHeader h;
    if(!s||coli_segment_stream_read(rf,rd,&h,sizeof(h),error,es))return -1;
    if(h.position>s->context_tokens||h.layer_begin!=s->engine->layer_begin||
       h.layer_end!=s->engine->layer_end||h.context_tokens!=s->context_tokens)
        return coli_segment_adapter_error(error,es,"Qwen3.8 snapshot identity is incompatible");
    ColiSegmentStateSpan *sp=NULL; size_t n=0,bytes=0;
    if(qwen38_segment_spans(s,h.position,&sp,&n,error,es)||
       coli_segment_spans_size(sp,n,&bytes)||
       coli_segment_snapshot_header_valid(&h,"qwen38",s->engine->layer_begin,
                                          s->engine->layer_end,s->context_tokens,
                                          bytes,error,es)){
        free(sp);return -1;
    }
    int rc=coli_segment_spans_restore_checked(
        sp,n,h.payload_hash,rf,rd,qwen38_segment_payload_valid,s,error,es);
    free(sp);if(rc)return rc;
    s->position=h.position;return 0;
}

static const ColiSegmentAdapter qwen38_segment_adapter={sizeof(ColiSegmentAdapter),COLI_SEGMENT_ABI_VERSION,"qwen38",qwen38_segment_engine_open,qwen38_segment_engine_destroy,qwen38_segment_session_create,qwen38_segment_session_destroy,qwen38_segment_session_run,qwen38_segment_session_snapshot,qwen38_segment_session_restore,{0}};
int coli_qwen38_segment_adapter_register(void){return coli_segment_adapter_register(&qwen38_segment_adapter);}
#endif /* COLI_SEGMENT_ADAPTER */

#ifdef COLI_EDGE_ADAPTER
/* ---------- engine-owned model Edge adapter --------------------------- */

typedef struct {
    Model model;
} Qwen38EdgeEngine;

/* Edge owns only the tokenizer and boundary tensors.  Do not call
 * q38_model_free(): a boundary-only model deliberately has no Layer/cache
 * arrays, and keeping this destructor explicit makes it impossible for a
 * later core change to make the Edge process load or free transformer state. */
static void qwen38_edge_engine_destroy(void *engine_impl) {
    Qwen38EdgeEngine *engine=(Qwen38EdgeEngine*)engine_impl;
    if(!engine)return;
    Model *model=&engine->model;
    q38_weight_free(&model->embed);
    q38_weight_free(&model->lm_head);
    free(model->final_gr.norm);
    q38_weight_free(&model->final_gr.down);
    q38_weight_free(&model->final_gr.up);
    free(model->c.is_attn);
    st_destroy(&model->S);
    free_tokenizer();
    free(engine);
}

static int qwen38_edge_engine_open(
    void **engine_impl,ColiEdgeCapabilities *capabilities,
    const ColiEdgeEngineOptions *options,char *error,size_t error_size) {
    if(!engine_impl||!capabilities||!options||!options->model_dir)
        return coli_edge_adapter_error(error,error_size,
                                       "invalid Qwen3.8 Edge open");
    *engine_impl=NULL;
    if(options->backend_mask&&(options->backend_mask&~COLI_EDGE_CAP_CPU))
        return coli_edge_adapter_error(error,error_size,
                                       "Qwen3.8 Edge supports CPU only");
    /* The production tokenizer is process-global.  Segment-only engines do
     * not touch it, so a distributed process may host every Segment adapter
     * plus one Qwen3.8 Edge engine without cross-wiring vocabularies. */
    if(g_tok)
        return coli_edge_adapter_error(error,error_size,
                                       "a Qwen3.8 tokenizer is already active");
    Qwen38EdgeEngine *engine=(Qwen38EdgeEngine*)calloc(1,sizeof(*engine));
    if(!engine)
        return coli_edge_adapter_error(error,error_size,
                                       "out of memory opening Qwen3.8 Edge");
    Model *model=&engine->model;
    model->native_bf16=q38_env_bool("Q38_NATIVE_BF16",1);
    model->native_fp8=q38_env_bool("Q38_NATIVE_FP8",1);
    q38_load_cfg(&model->c,options->model_dir);
    q38_validate_cfg(&model->c);
    st_init(&model->S,options->model_dir);
    if(st_has(&model->S,"model.language_model.embed_tokens.weight"))
        snprintf(model->prefix,sizeof model->prefix,"model.language_model");
    else if(st_has(&model->S,"model.embed_tokens.weight"))
        snprintf(model->prefix,sizeof model->prefix,"model");
    else {
        qwen38_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error,error_size,
            "Qwen3.8 checkpoint has no text embedding");
    }

    Cfg *config=&model->c;char name[320];
    snprintf(name,sizeof name,"%s.embed_tokens.weight",model->prefix);
    model->embed=q38_load_weight(model,name,config->vocab,config->hidden);
    model->lm_head=q38_load_weight(model,"lm_head.weight",
                                   config->vocab,config->hidden);
    q38_load_gr(model,&model->final_gr,-1,NULL,0);

    char tokenizer_path[4096];
    int path_length=snprintf(tokenizer_path,sizeof tokenizer_path,
                             "%s/tokenizer.json",options->model_dir);
    if(path_length<0||(size_t)path_length>=sizeof tokenizer_path||
       load_tokenizer(tokenizer_path)) {
        qwen38_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error,error_size,
                                       "cannot load Qwen3.8 tokenizer");
    }

    uint64_t resident=model->resident_weight_bytes;
    uint64_t norm_bytes=(uint64_t)config->hc_width*sizeof(float);
    if(resident>UINT64_MAX-norm_bytes) {
        qwen38_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error,error_size,
                                       "Qwen3.8 Edge resident size overflows");
    }
    resident+=norm_bytes;
    if(options->memory_limit_bytes&&resident>options->memory_limit_bytes) {
        qwen38_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error,error_size,
                                       "Qwen3.8 Edge exceeds memory limit");
    }

    /* Numeric compatibility includes the actual routed-expert representation
     * even though Edge does not load those tensors.  This must be byte-for-byte
     * identical to the full Segment range or the runtime refuses the pairing. */
    uint64_t slot_bytes=0,fixed_scale_bytes=0;unsigned numeric_kinds=0;
    if(q38_segment_expert_layout(model,0,(uint32_t)config->layers,
                                 &slot_bytes,&fixed_scale_bytes,
                                 &numeric_kinds)) {
        qwen38_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error,error_size,
            "Qwen3.8 Edge checkpoint has an unsupported expert layout");
    }
    (void)slot_bytes;(void)fixed_scale_bytes;

    memset(capabilities,0,sizeof(*capabilities));
    capabilities->struct_size=sizeof(*capabilities);
    capabilities->abi_version=COLI_EDGE_ABI_VERSION;
    capabilities->flags=COLI_EDGE_CAP_TOKENIZE|COLI_EDGE_CAP_DETOKENIZE|
                        COLI_EDGE_CAP_GREEDY|COLI_EDGE_CAP_CPU;
    coli_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id),"qwen38");
    coli_edge_capability_string(
        capabilities->state_schema,sizeof(capabilities->state_schema),
        "qwen38/hyper-kv-deltanet-ple-f32-v1");
    q38_segment_numeric_class(capabilities->numeric_class,
                              sizeof(capabilities->numeric_class),numeric_kinds);
    coli_edge_capability_string(
        capabilities->tokenizer_class,sizeof(capabilities->tokenizer_class),
        "qwen38/hf-byte-bpe-nfc-v1");
    capabilities->state_dtype=COLI_EDGE_DTYPE_F32;
    capabilities->state_width=(uint32_t)config->hc_width;
    capabilities->vocab_size=(uint32_t)config->vocab;
    capabilities->max_batch_rows=128;
    capabilities->max_context_tokens=(uint32_t)config->max_positions;
    capabilities->num_layers=(uint32_t)config->layers;
    capabilities->bos_token_id=-1;
    capabilities->eos_token_id=config->eos_id;
    capabilities->resident_bytes=resident;
    *engine_impl=engine;
    return 0;
}

static int qwen38_edge_tokenize(
    void *engine_impl,const char *text,size_t text_bytes,
    int32_t *token_ids,size_t token_capacity,size_t *token_count,
    char *error,size_t error_size) {
    (void)engine_impl;
    if(!text||!token_count||text_bytes>INT_MAX)
        return coli_edge_adapter_error(error,error_size,
                                       "invalid Qwen3.8 tokenizer input");
    int *ids=NULL,count=0;
    encode_text_n(text,text_bytes,&ids,&count);
    if(count<0||(token_ids&&token_capacity<(size_t)count)) {
        free(ids);
        return coli_edge_adapter_error(error,error_size,
                                       "Qwen3.8 token output buffer is too small");
    }
    *token_count=(size_t)count;
    if(token_ids)
        for(int item=0;item<count;item++)token_ids[item]=ids[item];
    free(ids);
    return 0;
}

static int qwen38_edge_detokenize(
    void *engine_impl,const int32_t *token_ids,size_t token_count,
    char *text,size_t text_capacity,size_t *text_bytes,
    char *error,size_t error_size) {
    Qwen38EdgeEngine *engine=(Qwen38EdgeEngine*)engine_impl;
    if(!engine||!token_ids||!token_count||!text_bytes||token_count>INT_MAX||
       token_count>SIZE_MAX/sizeof(int))
        return coli_edge_adapter_error(error,error_size,
                                       "invalid Qwen3.8 detokenizer input");
    int *ids=(int*)malloc(token_count*sizeof(*ids));
    if(!ids)
        return coli_edge_adapter_error(error,error_size,
            "out of memory detokenizing Qwen3.8 tokens");
    for(size_t item=0;item<token_count;item++){
        int token=token_ids[item];
        if(token<0||token>=engine->model.c.vocab||token>=g_tok_n||
           !g_tok||!g_tok[token]){
            free(ids);
            return coli_edge_adapter_error(error,error_size,
                                           "Qwen3.8 token ID is not decodable");
        }
        ids[item]=token;
    }
    unsigned char *decoded=NULL;size_t decoded_bytes=0;
    int result=decode_range_alloc(ids,0,(int)token_count,
                                  &decoded,&decoded_bytes);
    free(ids);
    if(result) {
        free(decoded);
        return coli_edge_adapter_error(error,error_size,
            "cannot detokenize Qwen3.8 tokens");
    }
    *text_bytes=decoded_bytes;
    if(text&&text_capacity<decoded_bytes+1u) {
        free(decoded);
        return coli_edge_adapter_error(error,error_size,
                                       "Qwen3.8 text output buffer is too small");
    }
    if(text)memcpy(text,decoded,decoded_bytes+1u);
    free(decoded);
    return 0;
}

static int qwen38_edge_embed(void *engine_impl,
                             const ColiEdgeEmbedRequest *request,
                             char *error,size_t error_size) {
    Qwen38EdgeEngine *engine=(Qwen38EdgeEngine*)engine_impl;
    Cfg *config=&engine->model.c;
    float *output=(float*)request->output;
    float *row=(float*)malloc((size_t)config->hidden*sizeof(*row));
    if(!row)
        return coli_edge_adapter_error(error,error_size,
            "out of memory embedding Qwen3.8 tokens");
    for(uint32_t index=0;index<request->rows;index++) {
        if(request->should_cancel&&
           request->should_cancel(request->cancel_user_data)) {
            free(row);
            return coli_edge_adapter_error(error,error_size,
                                           "Qwen3.8 Edge embedding cancelled");
        }
        int token=request->token_ids[index];
        if(token<0||token>=config->vocab) {
            free(row);
            return coli_edge_adapter_error(error,error_size,
                                           "Qwen3.8 token ID is out of range");
        }
        q38_weight_row(&engine->model.embed,token,row);
        float *state=output+(size_t)index*config->hc_width;
        for(int copy=0;copy<config->hc_count;copy++)
            memcpy(state+(size_t)copy*config->hidden,row,
                   (size_t)config->hidden*sizeof(*row));
    }
    free(row);
    return 0;
}

static int qwen38_edge_select(void *engine_impl,
                              const ColiEdgeSelectRequest *request,
                              char *error,size_t error_size) {
    Qwen38EdgeEngine *engine=(Qwen38EdgeEngine*)engine_impl;
    Cfg *config=&engine->model.c;
    float *hidden=(float*)malloc((size_t)config->hidden*sizeof(*hidden));
    float *logits=(float*)malloc((size_t)config->vocab*sizeof(*logits));
    if(!hidden||!logits) {
        free(logits);free(hidden);
        return coli_edge_adapter_error(error,error_size,
                                       "out of memory running Qwen3.8 head");
    }
    const float *input=(const float*)request->input;
    for(uint32_t row=0;row<request->rows;row++) {
        if(request->should_cancel&&
           request->should_cancel(request->cancel_user_data)) {
            free(logits);free(hidden);
            return coli_edge_adapter_error(error,error_size,
                                           "Qwen3.8 Edge selection cancelled");
        }
        q38_gr_read(&engine->model,&engine->model.final_gr,
                    input+(size_t)row*config->hc_width,1,hidden,NULL);
        q38_weight_matmul(logits,hidden,&engine->model.lm_head,
                          1,config->hidden,config->vocab);
        if(coli_edge_argmax(logits,(uint32_t)config->vocab,
                            &request->token_ids[row],
                            request->scores?&request->scores[row]:NULL)) {
            free(logits);free(hidden);
            return coli_edge_adapter_error(error,error_size,
                                           "Qwen3.8 Edge head failed");
        }
    }
    free(logits);free(hidden);
    return 0;
}

static const ColiEdgeAdapter qwen38_edge_adapter={
    sizeof(ColiEdgeAdapter),COLI_EDGE_ABI_VERSION,"qwen38",
    qwen38_edge_engine_open,qwen38_edge_engine_destroy,
    qwen38_edge_tokenize,qwen38_edge_detokenize,
    qwen38_edge_embed,qwen38_edge_select,{0}
};

int coli_qwen38_edge_adapter_register(void) {
    return coli_edge_adapter_register(&qwen38_edge_adapter);
}
#endif /* COLI_EDGE_ADAPTER */
