/* Qwen3.6-35B-A3B inference engine in pure C, Phase 2: Gated Attention + Gated
 * DeltaNet (recurrent linear attention) + streaming MoE.
 *
 * The full model is a hybrid: 10 x (3 x Gated DeltaNet -> MoE, 1 x Gated
 * Attention -> MoE). Phase 1 implemented ONLY the 25% attention layers and
 * treated the DeltaNet layers as identity; Phase 2 implements BOTH:
 *   - Gated Attention (GQA, per-head q/k RMSNorm, partial RoPE, output gate).
 *   - Gated DeltaNet: causal depthwise conv1d + recurrent gated-delta-rule with a
 *     carried conv ring + state S[h]=[kdim,vdim], then per-head Gated RMSNorm.
 * Every layer (attention or DeltaNet) carries its own MoE/MLP block.
 *
 * DENSE (embed, attn/dn q/k/v/o & projections, q/k norms, RMSNorm, router gate,
 * shared expert, lm_head, final norm) resident in RAM (float32). Expert weights
 * read from disk on-demand via pread + posix_fadvise(DONTNEED), cached LRU
 * per-layer, with a PILOT prefetch thread -- the same mechanism that fits
 * GLM-5.2 in 15 GB.
 *
 * Env vars (inherited from olmoe.c): PILOT, HOT, WARMUP, WIDE, SMOOTH, CONF_LIMIT.
 * Plus: SNAP=<dir>, and argv: qwen36 <cache/layer> <ebits> [ref.json] [PPL=1].
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Hard context ceiling: the model's max_position_embeddings. Every buffer that
 * scales with position (KV cache, attention score row) is allocated from max_t,
 * so this is a policy limit, not a buffer limit -- but it is ONE limit, named
 * once. It used to be the literal 8192 in two unrelated places: the size of a
 * stack array in attention() and the default of Q36_MAXT in serve_one(). They
 * agreed by luck, and raising Q36_MAXT moved the guard without moving the
 * buffer, so a longer prompt overran the stack instead of being refused.
 * Context costs 40 KB/token in KV (10 attention layers, f32) -- 128k is 5.0 GiB
 * -- which is why Q36_MAXT still defaults far below this. */
#define QWEN36_ATTN_MAX_CTX 262144
#define QWEN36_DEFAULT_MAX_CTX 8192
/* A speculative verify's rows: the picked token and up to five prompt-lookup drafts.
 * A rejection after row k restores the state the verify copied after that row. */
#define Q36_SPEC_ROWS 6
#define Q36_SPEC_SNAPS (Q36_SPEC_ROWS - 1)
static int g_q36_rowwise;   /* set for a speculative verify: its rows must get a decode step's bits (q36_spec_step) */

/* Effective ceiling: Q36_MAXT if set and sane, the conservative default
 * otherwise; never above the hard limit. */
/* A Clef checkpoint raises the default to its own input budget, 16384 (clef_head.h):
 * the KV rows are allocated as a request needs them, so the ceiling costs nothing
 * until a record that long arrives (2 GiB of f32 rows on the 27B at 16384). */
static int g_q36_default_ctx = QWEN36_DEFAULT_MAX_CTX;
static int qwen36_max_ctx(void) {
    const char *e = getenv("Q36_MAXT");
    int v = (e && *e) ? atoi(e) : g_q36_default_ctx;
    if (v < 1) v = g_q36_default_ctx;
    return v > QWEN36_ATTN_MAX_CTX ? QWEN36_ATTN_MAX_CTX : v;
}
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "cpu_check.h"   /* the processor runs what this build assumes, or it says so (#1979) */
#include "serve_poll.h"       /* CANCEL a meta' turno (#1332) */
#include "cli_args.h"
#include "st.h"
#include "omp_tune.h"
#include "kv_prefix.h"
#include "spec_draft.h" /* prompt-lookup drafts and the gate that decides when they pay (COLI_LOOKUP=1) */
#include "pin_pool.h"   /* riuso del prefisso tra turni (shared) */
#include "decode_batch.h" /* ColiSubmit + coli_submit_ext: le chiavi key=value di SUBMIT */
#include "json.h"   /* tokenizer.json parsing (reuse minimal parser) */
#include "tok_unicode.h"        /* is_S/is_L/is_N: the exact pre-tokenizer classes (Clef) */
#include "qwen38_nfc.h"         /* NFC, the tokenizer.json normalizer (Clef) */
#include "qwen36_tier.h"   /* optional CUDA VRAM expert tier */
#include "vk_tier.h"       /* optional Vulkan routed-expert tier (COLI_VULKAN=1; stubs without VK=1) */
#include "route_trace.h"   /* the tier's expert history (.coli_usage), kept only while the tier is on */
#include "expert_ffn.h"    /* routed experts: planar int4 kernel + layer runner */
#include "idot.h"          /* integer dot kernels for the dense trunk (COLI_DENSE_IDOT, COLI_DENSE_BITS) */
#include "qwen38_vision.h" /* the ViT: the same tower in Qwen3.5/3.6/3.8 (#1757) */
#include "qwen36_qpack.h"  /* optional MLX-affine qpack routed-expert path */
#include "affine_quant.h"  /* MLX affine dense triples in model.safetensors */
#ifdef COLI_METAL
#include "backend_metal.h" /* coli_metal_init: affine pipelines for the store */
#endif
#ifdef COLI_VULKAN
#include "backend_vulkan.h" /* COLI_VULKAN=1: the resident dense trunk on a Vulkan device */
static int g_vk_ready = 0;
static int g_vk_dense = 1;  /* COLI_VK_DENSE=0: the dense trunk stays on the CPU, the expert tier alone uses the device */
static int g_vk_import = 0; /* COLI_VK_IMPORT: the device reads the int8, f16 and f32 rows where the host keeps them */
#endif
/* The dense rows' host allocation. With COLI_VULKAN set (a VK=1 build), page-aligned and
 * a whole number of pages, so a device that shares the CPU's RAM can read them in place
 * (coli_vk_tensor_import) instead of holding a second copy; otherwise malloc. The
 * values, and what the CPU computes from them, are the same either way. */
#define Q36_PAGE 4096
static int q36_waligned(void){
#ifdef COLI_VULKAN
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VULKAN"); on = e && atoi(e) != 0; }
    return on;
#else
    return 0;
#endif
}
static void *q36_walloc(size_t bytes){
    if (q36_waligned()) {
        void *p = NULL;
        size_t sz = (bytes + Q36_PAGE - 1) / Q36_PAGE * Q36_PAGE;
        return posix_memalign(&p, Q36_PAGE, sz ? sz : Q36_PAGE) ? NULL : p;
    }
    return malloc(bytes);
}
static void q36_wfree(void *p){
    /* Windows' posix_memalign is _aligned_malloc. Every q/h allocation uses
     * the same cached choice as q36_walloc, including an int8 copy discarded
     * after int4 packing, so its matching release must follow that choice. */
    if (q36_waligned()) compat_aligned_free(p);
    else free(p);
}
#ifdef COLI_SEGMENT_ADAPTER
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"
#endif
#ifdef COLI_EDGE_ADAPTER
#include "edge_runtime.h"
#include "edge_adapters.h"
#include "edge_adapter_internal.h"
#include <limits.h>
#endif

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <dlfcn.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

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
typedef struct { char **keys; int *vals; int *used; int cap; } SMap;
static unsigned shash(const char *s){ unsigned h=2166136261u; while(*s){ h^=(unsigned char)*s++; h*=16777619u; } return h; }
static void smap_init(SMap *m,int cap){ m->cap=cap; m->keys=calloc((size_t)cap,sizeof(char*)); m->vals=malloc((size_t)cap*sizeof(int)); m->used=calloc((size_t)cap,sizeof(int)); }
static void smap_put(SMap *m,const char *k,int v){ if(!k)return; unsigned h=shash(k)&(m->cap-1); while(m->used[h]){ if(m->keys[h]&&strcmp(m->keys[h],k)==0){m->vals[h]=v;return;} h=(h+1)&(m->cap-1);} m->used[h]=1; m->keys[h]=(char*)k; m->vals[h]=v; }
static int smap_get(SMap *m,const char *k){ if(!m||!m->cap||!k)return -1; unsigned h=shash(k)&(m->cap-1); while(m->used[h]){ if(m->keys[h]&&strcmp(m->keys[h],k)==0)return m->vals[h]; h=(h+1)&(m->cap-1);} return -1; }

static SMap  g_rev;                 /* piece string -> id (encode) */
static SMap  g_merge;               /* "a\x1F b" pair -> rank (encode) */
static char  byte_sym_utf8[256][8]; /* byte -> UTF-8 of mapped codepoint */
static short g_unmap[512];          /* mapped codepoint -> original byte (-1 = unused) */
static int   g_nspecial = 0;
static char **g_sp_str = NULL; static int *g_sp_id = NULL; static int *g_sp_len = NULL;

static const char *jstr(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_STR)?v->str:NULL; }
static double jnum(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_NUM)?v->num:0; }

enum { U_W=0, U_L=1, U_M=2, U_N=3, U_P=4, U_O=5 };
/* Clef (clef_head.h) renders every request into the prompt its reference
 * tokenizes, so its token ids must be the reference's to the last one: the full
 * Unicode classes and the NFC normalizer. The regex is the one its tokenizer.json
 * and transformers' Qwen2Tokenizer apply, `[^\r\n\p{L}\p{N}]?\p{L}+` and
 * ` ?[^\s\p{L}\p{N}]+[\r\n]*`: a combining mark is not part of a letter run there
 * (measured: "x" U+0303 U+0304 "y" is three pieces), so it is classed with
 * punctuation and U_M never comes out. The ranges below are the original
 * approximation, kept for every other checkpoint so their ids do not move. */
static int g_tok_exact = 0;
#ifndef QWEN36_NO_MAIN
static int g_clef = 0;        /* a Clef decision head is loaded (DECIDE answered) */
#endif
static int uclass(unsigned cp){
    if (g_tok_exact) {
        if (is_S(cp)) return U_W;
        if (is_L(cp)) return U_L;
        if (is_N(cp)) return U_N;
        return U_P;
    }
    if (cp==0x20||cp==0x09||cp==0x0A||cp==0x0D||cp==0x0B||cp==0x0C) return U_W;
    if (cp==0x00A0||cp==0x2000||cp==0x2001||cp==0x2002||cp==0x2003||cp==0x2004||cp==0x2005||cp==0x2006||cp==0x2007||cp==0x2008||cp==0x2009||cp==0x200A||cp==0x2028||cp==0x2029||cp==0x202F||cp==0x205F||cp==0x3000||cp==0xFEFF) return U_W;
    if (cp>=0x30&&cp<=0x39) return U_N;
    if (cp>=0xFF10&&cp<=0xFF19) return U_N;
    if (cp>=0x0660&&cp<=0x0669) return U_N;
    if ((cp>=0x41&&cp<=0x5A)||(cp>=0x61&&cp<=0x7A)) return U_L;
    if (cp>=0x00C0&&cp<=0x024F) return U_L;
    if (cp>=0x0400&&cp<=0x04FF) return U_L;
    if (cp>=0x0600&&cp<=0x06FF) return U_L;
    if (cp>=0x1F00&&cp<=0x1FFF) return U_L;
    if (cp>=0x3040&&cp<=0x30FF) return U_L;
    if (cp>=0x3400&&cp<=0x4DBF) return U_L;
    if (cp>=0x4E00&&cp<=0x9FFF) return U_L;
    if (cp>=0xAC00&&cp<=0xD7A3) return U_L;
    if (cp>=0x300&&cp<=0x36F) return U_M;
    if (cp>=0x1AB0&&cp<=0x1AFF) return U_M;
    if (cp>=0x1DC0&&cp<=0x1DFF) return U_M;
    if (cp>=0x20D0&&cp<=0x20FF) return U_M;
    if (cp>=0xFE20&&cp<=0xFE2F) return U_M;
    if (cp>=0x21&&cp<=0x2F) return U_P;
    if (cp>=0x3A&&cp<=0x40) return U_P;
    if (cp>=0x5B&&cp<=0x60) return U_P;
    if (cp>=0x7B&&cp<=0x7E) return U_P;
    if (cp>=0x3000&&cp<=0x303F) return U_P;
    if (cp>=0xFF01&&cp<=0xFF0F) return U_P;
    if (cp>=0xFF1A&&cp<=0xFF20) return U_P;
    if (cp>=0xFF3B&&cp<=0xFF40) return U_P;
    if (cp>=0xFF5B&&cp<=0xFF65) return U_P;
    if (cp>=0x2010&&cp<=0x2027) return U_P;
    if (cp>=0x2030&&cp<=0x205E) return U_P;
    return U_O;
}
/* A serving payload is byte-counted and may end in a truncated multibyte
 * sequence. Treat that byte as one invalid unit without reading past it: this
 * is qwen38's utf8_decode, which the two engines share a pre-tokenizer with. */
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
static void push_id(int **ids,int *n,int *cap,int v){
    if(*n==*cap){
        *cap*=2;
        int *tmp=realloc(*ids,*cap*sizeof(int));
        if(!tmp){ fprintf(stderr,"qwen36: OOM reallocating token id buffer (%d entries)\n",*cap); exit(1); }
        *ids=tmp;
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
        if(k<n && c!='\r'&&c!='\n'&&uclass(c)!=U_L&&uclass(c)!=U_N){
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
        /* The three whitespace rules, in the regex's order. `last` is where the
         * final whitespace char of the run starts, `nl_end` where the last CR/LF
         * inside the run ends. */
        int k=i,last=i,nl_end=-1;
        while(k<n){int a;unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)!=U_W) break; last=k; k+=a; if(cc=='\r'||cc=='\n') nl_end=k;}
        /* rule5: \s*[\r\n]+ -- greedy up to the LAST newline; spaces after it
         * belong to the next piece ("\n  x" is "\n" then " " then " x"). */
        if(nl_end>0) return nl_end;
        /* rule6: \s+(?!\S) -- a run followed by a non-space keeps its last char
         * for the next piece, which then takes it as " x" or " ." (HF: "  <" is
         * " " then " <", not "  " then "<"). A single char cannot back off and
         * falls to rule7, \s+, which takes it whole. */
        if(k<n && last>i) return last;
        return k;
    }
    return i+adv;
}
static void bpe_piece(const char *piece,int len,int **ids,int *n,int *cap){
    if(len<=0) return;
    int sc=0,scap=16; char **syms=malloc(scap*sizeof(char*));
    for(int b=0;b<len;b++){
        const char *sym=byte_sym_utf8[(unsigned char)piece[b]];
        int sl=(int)strlen(sym); char *d=malloc(sl+1); memcpy(d,sym,sl); d[sl]=0;
        if(sc==scap){
            scap*=2;
            char **tmp=realloc(syms,scap*sizeof(char*));
            if(!tmp){ fprintf(stderr,"qwen36: OOM reallocating BPE symbol buffer (%d entries)\n",scap); exit(1); }
            syms=tmp;
        }
        syms[sc++]=d;
    }
    while(sc>1){
        int best=-1,besti=-1;
        for(int k=0;k<sc-1;k++){
            const char *a=syms[k],*b=syms[k+1];
            size_t kl=(size_t)strlen(a)+1+(size_t)strlen(b)+1;
            char *key=malloc(kl); snprintf(key,kl,"%s\x1F%s",a,b);
            int r=smap_get(&g_merge,key); free(key);
            if(r>=0 && (best<0||r<best)){best=r;besti=k;}
        }
        if(besti<0) break;
        char *m=malloc(strlen(syms[besti])+strlen(syms[besti+1])+1);
        strcpy(m,syms[besti]); strcat(m,syms[besti+1]);
        free(syms[besti]); free(syms[besti+1]); syms[besti]=m;
        for(int k=besti+1;k<sc-1;k++) syms[k]=syms[k+1]; sc--;
    }
    for(int k=0;k<sc;k++){ int id=smap_get(&g_rev,syms[k]); if(id<0) id=0; push_id(ids,n,cap,id); free(syms[k]); }
    free(syms);
}
/* The next added token at or after i, or n when there is none. HF splits the
 * added tokens out FIRST and pre-tokenizes only the ordinary text between them.
 * Looking for the special at the start of each piece is not the same thing: the
 * punctuation rule ` ?[^\s\p{L}\p{M}\p{N}]+` swallows the `<|` of `<|im_end|>`
 * together with the `.` before it, so the special was never at a piece start and
 * got encoded as text (#1653: `X.<|im_end|>` was 7 tokens instead of 3, and
 * every chat turn ending in punctuation paid +4). qwen38.c already splits this
 * way; this is the same shape. */
static int next_special(const char *s,int i,int n){
    for(int k=i;k<n;k++){ int sid; if(try_special(s,k,n,&sid)>0) return k; }
    return n;
}
static void encode_text(const char *text,int **out_ids,int *out_n){
    int cap=1024,n=0; int *ids=malloc(cap*sizeof(int));
    int tlen=(int)strlen(text); int i=0;
    while(i<tlen){
        int sid; int L=try_special(text,i,tlen,&sid);
        if(L>0){ push_id(&ids,&n,&cap,sid); i+=L; continue; }
        /* Ordinary text runs to the next added token, and the pre-tokenizer
         * sees that boundary as the end of its input, exactly as HF's does. */
        int end=next_special(text,i+1,tlen);
        if(g_tok_exact){
            /* the added tokens are matched on the original bytes (normalized=false),
             * the ordinary span is NFC-normalized before the regex (qwen38.c) */
            char *norm=NULL; size_t nl=0;
            if(q38_nfc_normalize(text+i,(size_t)(end-i),&norm,&nl)||nl>(size_t)INT_MAX){
                free(norm); fprintf(stderr,"[enc] NFC normalization failed\n"); exit(1);
            }
            for(int k=0;k<(int)nl;){
                int j=pretok_end(norm,k,(int)nl); if(j<=k) j=k+utf8_adv(norm,k,(int)nl);
                if(j>(int)nl) j=(int)nl;
                bpe_piece(norm+k,j-k,&ids,&n,&cap);
                k=j;
            }
            free(norm); i=end; continue;
        }
        while(i<end){
            int j=pretok_end(text,i,end); if(j<=i) j=i+utf8_adv(text,i,end);
            if(j>end) j=end;
            bpe_piece(text+i,j-i,&ids,&n,&cap);
            i=j;
        }
    }
    *out_ids=ids; *out_n=n;
}

/* Load Qwen tokenizer.json and build an id->piece table from model.vocab
 * and added_tokens. Merges are irrelevant for decoding. */
static void load_tokenizer(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[tok] cannot open %s\n", path); return; }
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1);
    if (fread(buf,1,(size_t)n,f) != (size_t)n) { /* ignore short read */ }
    buf[n] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    jval *model = json_get(root, "model"); if (!model) model = root;
    jval *vocab = json_get(model, "vocab");
    if (!vocab) vocab = json_get(model, "tokens");
    if (!vocab) { fprintf(stderr, "[tok] no model.vocab/tokens in %s\n", path); free(buf); return; }
    jval *adds = json_get(root, "added_tokens");

    int mx = 0;
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>mx)mx=id; }
    } else {
        mx = vocab->len - 1;
    }
    if (adds && adds->t==J_ARR){
        for (int k=0;k<adds->len;k++){
            jval *t = adds->kids[k];
            int id = (int)jnum(t,"id");
            if (id > mx) mx = id;
        }
    }

    g_tok = calloc((size_t)mx+1, sizeof(char*));
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>=0 && id<=mx) g_tok[id]=strdup(vocab->keys[i]); }
    } else {
        for (int i=0;i<vocab->len;i++){ if(vocab->kids[i] && vocab->kids[i]->t==J_STR) g_tok[i]=strdup(vocab->kids[i]->str); }
    }
    if (adds && adds->t==J_ARR){
        for (int k=0;k<adds->len;k++){
            jval *t = adds->kids[k];
            const char *c = jstr(t,"content");
            int id = (int)jnum(t,"id");
            /* Only the non-special ones: <think>, </think>, <tool_call>,
             * <tool_response> are text the gateway parses. Special tokens
             * (<|im_start|>, <|endoftext|>, ...) keep decoding to nothing,
             * as reference decoding does with skip_special_tokens. */
            jval *sp = json_get(t,"special");
            if (sp && sp->t==J_BOOL && sp->boolean) continue;
            if (c && id>=0 && id<=mx && !g_tok[id])
                g_tok[id]=strdup(c);
        }
    }
    g_tok_n = mx+1;

    /* ---- encoder tables (text -> ids) ---- */
    smap_init(&g_rev, 1<<19);
    for (int i=0;i<g_tok_n;i++) if (g_tok[i]) smap_put(&g_rev, g_tok[i], i);

    smap_init(&g_merge, 1<<19);
    jval *merges = json_get(model, "merges");
    if (merges && merges->t==J_ARR){
        for (int r=0;r<merges->len;r++){
            /* Two on-disk spellings for one merge table: legacy tokenizer.json
             * writes "a b" strings, tokenizers >= 0.20 (transformers 4.45+,
             * the Qwen3.6 checkpoints included) writes ["a","b"] pairs.  The
             * string-only reader SILENTLY indexed zero merges from the pair
             * form, and encode_text degraded to one token per byte-symbol --
             * 24 tokens for a 24-char prompt, real-model run -- because
             * bpe_piece treats an empty merge table as "nothing to merge",
             * not as an error. */
            jval *mk = merges->kids[r];
            const char *a, *b;
            int la, lb;
            if (mk && mk->t==J_STR && mk->str){
                const char *sp = strchr(mk->str, ' '); if(!sp) continue;
                a = mk->str; la = (int)(sp - mk->str);
                b = sp + 1;  lb = (int)strlen(b);
            } else if (mk && mk->t==J_ARR && mk->len==2 &&
                       mk->kids[0] && mk->kids[0]->t==J_STR && mk->kids[0]->str &&
                       mk->kids[1] && mk->kids[1]->t==J_STR && mk->kids[1]->str){
                a = mk->kids[0]->str; la = (int)strlen(a);
                b = mk->kids[1]->str; lb = (int)strlen(b);
            } else continue;
            char *key=malloc(la+1+lb+1);
            memcpy(key,a,la); key[la]=0x1F; memcpy(key+la+1,b,lb); key[la+1+lb]=0;
            smap_put(&g_merge, key, r);
        }
    }
    if (adds && adds->t==J_ARR && g_nspecial==0){
        g_nspecial = adds->len;
        g_sp_str = malloc(g_nspecial*sizeof(char*));
        g_sp_id   = malloc(g_nspecial*sizeof(int));
        g_sp_len  = malloc(g_nspecial*sizeof(int));
        for (int k=0;k<adds->len;k++){
            jval *t = adds->kids[k];
            const char *c = jstr(t,"content");
            g_sp_str[k] = c?strdup(c):strdup("");
            g_sp_id[k]  = (int)jnum(t,"id");
            g_sp_len[k] = (int)strlen(g_sp_str[k]);
        }
    }
    build_byte_sym();

    fprintf(stderr, "[tok] loaded %d pieces (max id %d) from %s\n", vocab->len, mx, path);
    free(buf);
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
static const char *g_model = "qwen3.6-35b-a3b-colibri";
static double now_s(void);   /* forward decl; defined later near model code */

/* Output sink for server mode: when g_sock_out >= 0, SSE/JSON bytes are routed
 * to the live socket via g_sock_send instead of stdout. Lets qwen36_serve.c
 * reuse all emit logic without any change to the CLI path. */
static long long g_sock_out = -1;
static void (*g_sock_send)(long long fd, const char *buf, int n) = NULL;

/* JSON-escape a byte string into out (no surrounding quotes). Returns length. */
static int json_escape(const unsigned char *s, int n, char *out, int outsz){
    int o = 0;
    for (int i=0;i<n;i++){
        unsigned char c = s[i];
        if (c == '"'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='"'; } }
        else if (c == '\\'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='\\'; } }
        else if (c == '\n'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='n'; } }
        else if (c == '\r'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='r'; } }
        else if (c == '\t'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='t'; } }
        else if (c == '\b'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='b'; } }
        else if (c == '\f'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='f'; } }
        else if (c < 0x20){ if(o+6<outsz){ sprintf(out+o, "\\u%04x", c); o+=6; } }
        else { if(o+1<outsz) out[o++] = (char)c; }
    }
    if (o < outsz) out[o] = 0;
    return o;
}

/* Append b[0..n) into buf (*bn), extract as many LEADING complete UTF-8
 * codepoints as possible into out[0..*outn) (max 255). Trailing partial
 * sequence stays in buf. Returns bytes written to out. */
static int utf8_drain(unsigned char *buf, int *bn, const unsigned char *b, int n, unsigned char *out, int *outn){
    *outn = 0;
    for (int k=0;k<n;k++){ if (*bn < 16) buf[(*bn)++] = b[k]; }
    int j = 0;
    while (j < *bn){
        unsigned char lead = buf[j]; int need;
        if (lead < 0x80) need = 1;
        else if ((lead & 0xE0) == 0xC0) need = 2;
        else if ((lead & 0xF0) == 0xE0) need = 3;
        else if ((lead & 0xF8) == 0xF0) need = 4;
        else { memmove(buf+j, buf+j+1, *bn-j-1); (*bn)--; continue; }
        if (j+need > *bn) break;
        if (*outn + need <= 255){ for (int x=0;x<need;x++) out[(*outn)++] = buf[j+x]; }
        memmove(buf+j, buf+j+need, *bn-j-need);
        *bn -= need;
    }
    return *outn;
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
static void decode_id_to_bytes(int id, unsigned char *out, int *outn){
    *outn = 0;
    if (!g_tok || id<0 || id>=g_tok_n || !g_tok[id]) return;
    const unsigned char *pc = (const unsigned char*)g_tok[id];
    /* byte-fallback token: <0xXX> -> raw byte */
    if (pc[0]=='<' && pc[1]=='0' && pc[2]=='x' && pc[5]=='>'){
        out[(*outn)++] = (unsigned char)(hexnib((char)pc[3])*16 + hexnib((char)pc[4]));
        return;
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
        if (cp == 0x2581) out[(*outn)++] = ' ';             /* SentencePiece space marker (kept safe) */
        else if (cp < 512 && g_unmap[cp] >= 0) out[(*outn)++] = (unsigned char)g_unmap[cp]; /* reverse byte_to_unicode */
        else out[(*outn)++] = (unsigned char)cp;
        if (*outn >= 255) break;
    }
}

/* Decode a range of token ids into a NUL-terminated text buffer (non-streaming). */
static int decode_range(const int *arr, int from, int to, char *ob, int obsz){
    unsigned char sb[16]; int sbn = 0; int o = 0;
    for (int i=from;i<to;i++){
        unsigned char tmp[256]; int tn = 0; decode_id_to_bytes(arr[i], tmp, &tn);
        unsigned char chunk[256]; int cn = 0; utf8_drain(sb, &sbn, tmp, tn, chunk, &cn);
        for (int k=0;k<cn && o<obsz-1;k++) ob[o++] = (char)chunk[k];
    }
    for (int k=0;k<sbn && o<obsz-1;k++) ob[o++] = (char)sb[k];   /* flush any trailing partial */
    if (o < obsz) ob[o] = 0;
    return o;
}

/* Append bytes to a buffer and flush any complete UTF-8 codepoints; any
 * trailing partial sequence is left in the buffer for the next call. */
static void out_bytes(unsigned char *buf, int *bn, const unsigned char *b, int n){
    for (int k=0; k<n; k++){
        if (*bn < 16) buf[(*bn)++] = b[k];
        int j = 0;
        while (j < *bn){
            unsigned char lead = buf[j]; int need;
            if (lead < 0x80) need = 1;
            else if ((lead & 0xE0) == 0xC0) need = 2;
            else if ((lead & 0xF0) == 0xE0) need = 3;
            else if ((lead & 0xF8) == 0xF0) need = 4;
            else { putchar(buf[j]); memmove(buf+j, buf+j+1, *bn-j-1); (*bn)--; continue; }
            if (j+need > *bn) break;
            fwrite(buf+j, 1, (size_t)need, stdout);
            memmove(buf+j, buf+j+need, *bn-j-need);
            *bn -= need;
        }
    }
}

static void print_decoded(const int *arr, int from, int to){
    unsigned char buf[16]; int bn = 0;
    for (int i=from;i<to;i++){
        unsigned char tmp[256]; int tn = 0;
        decode_id_to_bytes(arr[i], tmp, &tn);
        out_bytes(buf, &bn, tmp, tn);
    }
    if (bn) fwrite(buf, 1, (size_t)bn, stdout);
}

/* Streaming variants: emit one token at a time. In OpenAI mode each token is
 * one SSE `chat.completion.chunk` (delta.content = decoded text for this token,
 * carrying partial UTF-8 across tokens so CJK never splits mid-codepoint).
 * Otherwise emit raw readable text, flushing complete UTF-8 codepoints. */
static void stream_token(int id){
    if (g_openai){
        if (g_ttft < 0) g_ttft = now_s() - g_gen_t0;   /* TTFT on first token */
        if (!g_tok){
            char jb[512];
            snprintf(jb, sizeof jb,
              "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
              "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%d\"},\"finish_reason\":null}]}",
              g_oa_id, g_oa_created, g_model, id);
            sse_chunk(jb); return;
        }
        unsigned char tmp[256]; int tn = 0;
        decode_id_to_bytes(id, tmp, &tn);
        unsigned char chunk[256]; int cn = 0;
        utf8_drain(g_sbuf, &g_sbn, tmp, tn, chunk, &cn);
        if (cn > 0){
            char esc[1024]; json_escape(chunk, cn, esc, sizeof esc);
            char jb[2048];
            snprintf(jb, sizeof jb,
              "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
              "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
              g_oa_id, g_oa_created, g_model, esc);
            sse_chunk(jb);
        }
        return;
    }
    /* default raw-text streaming */
    if (!g_tok){ printf("%d ", id); fflush(stdout); return; }
    unsigned char tmp[256]; int tn = 0;
    decode_id_to_bytes(id, tmp, &tn);
    out_bytes(g_sbuf, &g_sbn, tmp, tn);
    fflush(stdout);   /* make streaming visible immediately even when piped */
}
static void stream_flush(void){ if (g_sbn){ fwrite(g_sbuf, 1, (size_t)g_sbn, stdout); g_sbn = 0; } }

/* Emit the final OpenAI Chat Completions response for a finished generation.
 * Streaming: flushes any trailing partial UTF-8 as a last content chunk, then
 * sends the termination chunk (finish_reason + usage + timings) and "data: [DONE]".
 * Non-streaming: sends a single chat.completion JSON object.
 * When g_sock_out >= 0 the bytes go to the live socket; otherwise to stdout. */
static void emit_openai_result(const int *out, int np, int n_new, int stream){
    double total = now_s() - g_gen_t0;
    if (g_ttft < 0) g_ttft = total;   /* non-streaming: all tokens arrive at once */
    double gen_t = total - g_ttft;
    double tps = (gen_t > 1e-6 && n_new > 1) ? n_new / gen_t : (total > 0 ? n_new / total : 0.0);
    if (stream){
        if (g_sbn > 0){
            unsigned char chunk[16]; int cn = 0;
            for (int k=0;k<g_sbn;k++) chunk[cn++] = g_sbuf[k]; g_sbn = 0;
            if (cn > 0){
                char esc[256]; json_escape(chunk, cn, esc, sizeof esc);
                char jb[768];
                snprintf(jb, sizeof jb,
                  "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
                  "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
                  g_oa_id, g_oa_created, g_model, esc);
                sse_chunk(jb);
            }
        }
        char jb[700];
        snprintf(jb, sizeof jb,
          "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
          "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"
          "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
          "\"timings\":{\"ttft_s\":%.3f,\"tokens_per_sec\":%.3f,\"total_s\":%.3f}}",
          g_oa_id, g_oa_created, g_model, np, n_new, np+n_new, g_ttft, tps, total);
        sse_chunk(jb);
        char done[16]; int dl = snprintf(done, sizeof done, "data: [DONE]\n\n");
        if (g_sock_out >= 0 && g_sock_send) g_sock_send(g_sock_out, done, dl);
        else { fwrite(done, 1, (size_t)dl, stdout); fflush(stdout); }
    } else {
        char text[1<<16]; decode_range(out, np, np+n_new, text, sizeof text);
        char esc[1<<16]; json_escape((const unsigned char*)text, (int)strlen(text), esc, sizeof esc);
        char buf[1<<20];
        int bl = snprintf(buf, sizeof buf,
          "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\","
          "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
          "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
          "\"timings\":{\"ttft_s\":%.3f,\"tokens_per_sec\":%.3f,\"total_s\":%.3f}}\n",
          g_oa_id, g_oa_created, g_model, esc, np, n_new, np+n_new, g_ttft, tps, total);
        if (g_sock_out >= 0 && g_sock_send) g_sock_send(g_sock_out, buf, bl);
        else { fwrite(buf, 1, (size_t)bl, stdout); fflush(stdout); }
    }
}

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_active;
    int q_heads, kv_heads, head_dim;        /* k/v head dim == attention head dim */
    int q_head_dim;                         /* q per-head total = head_dim*2 when attn_output_gate */
    int k_head_dim, v_head_dim, o_in;       /* o_in = q_heads*head_dim (o_proj input) */
    int rope_dim, rotary_dim;               /* rotary_dim = actual rotated dims (head_dim*partial_rotary_factor) */
    int n_experts, topk, inter, shared_inter, vocab;
    int n_group, topk_group;
    float theta, eps, partial_rotary_factor;
    int norm_topk, has_qk_norm, has_bias, attn_output_gate;
    /* RMSNorm weight dialect for the tensors rmsnorm_row touches (input/post
     * layernorms, q/k norms, final norm).  1 = HF Qwen3.6 zero-centered
     * storage, apply (1 + w) -- the convention every converted snapshot uses
     * and the engine has always assumed.  0 = full-gamma storage (MLX-derived
     * containers: mlx-lm materialises the +1 into the weights at conversion),
     * unshifted back to zero-centered at LOAD so the forward pass stays one
     * convention.  The DeltaNet gated norm is full-gamma in BOTH dialects and
     * is untouched by this flag. */
    int zero_centered_norms;
    uint8_t *is_attn;   /* [n_layers] 1 if Gated Attention layer, 0 if DeltaNet */
    /* Gated DeltaNet (linear_attention) dims, read from qwen36_meta.json. */
    int dn_vheads, dn_kheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim;
    int expert_gs;      /* expert scale group size along input dim; 0 = per-row */
    /* Mixed expert layout (convert_qwen36.py --down-bits): gate/up stay int4
     * (ebits, expert_gs), down_proj is int8 with its own group size. One slab
     * per expert, [gate int4 packed | up int4 packed | down int8], 2*inter*hidden
     * bytes -- told apart from int4 (1.5x) and int8 (3x) by size, like today. */
    int expert_down_bits, expert_down_gs;
    /* Vision (#1757). The tower is the same in Qwen3.5, 3.6 and 3.8 -- only its
     * output width follows the text model -- so the qwen38 one is reused. 0 depth
     * = text only (a container converted without it, or a text-only checkpoint). */
    int vis_depth, vis_hidden, vis_heads, vis_inter, vis_patch, vis_merge;
    int vis_temporal, vis_in_ch, vis_out_hidden, vis_num_pos;
    int image_token;
    /* Interleaved M-RoPE: frequency j rotates by the height position when
     * j % 3 == 1 and j < 3*sec[1], by the width one when j % 3 == 2 and
     * j < 3*sec[2], else by the temporal one. Text has all three equal. */
    int mrope_section[3];
} Cfg;

/* ---- Dense int8: a dense matrix that is quantized to int8 during load
 * (load_tq, below matmul_d) instead of loaded as f32 and quantized in a
 * separate pass afterward -- so the f32 staging buffer for THIS matrix alone
 * is what's briefly resident, not every dense matrix in the model at once.
 * `w` is the f32 copy: kept (and `q`/`sc` left NULL) when COLI_DENSE_I8=0,
 * the reference/parity path; freed once `q`/`sc` are populated otherwise
 * (COLI_KEEP_F32=1 keeps it alongside them, for debugging). matmul_d
 * dispatches on q!=NULL directly -- no pointer-keyed scan. */
/* q/sc: int8 rows with one scale per row (the classic copy, what the VRAM
 * tier uploads). q4/sg: the same matrix as int4 planar blocks of 64 with one
 * scale per group (COLI_DENSE_BITS=4), the layout the K1b grouped kernel
 * reads; ng = I/64 groups per row.
 * vk/vk_off: the Vulkan device copy (COLI_VULKAN=1) of whichever of q4, q or w
 * matmul_d reads, uploaded at the first matmul_d and kept; vk_off = the upload
 * failed once, this matrix stays on the CPU. Both stay zero without VK=1. */
/* h: the matrix as f16 (COLI_DENSE_BITS=16), the only copy then. */
/* vk_name, vk_tag, vk_quant: what load_tq read it with, to read it back from disk
 * (q36_dho_reload); vk_fmt, vk_gone: with the dense weights on the device only
 * (COLI_VK_DENSE_HOST), the device copy's format and 1 while no host copy is kept. */
typedef struct { const float *w; int8_t *q; float *sc; int I, O; uint8_t *q4; float *sg; int ng;
                 void *vk; int vk_off; uint16_t *h;
                 char *vk_name; const char *vk_tag; int vk_quant, vk_fmt, vk_gone, vk_imported; } QW;
static void qw_free(QW *w) {
#ifdef COLI_VULKAN
    if (w->vk) coli_vk_tensor_free((ColiVkTensor *)w->vk);
    w->vk = NULL; w->vk_off = 0; w->vk_imported = 0;
    free(w->vk_name); w->vk_name = NULL; w->vk_gone = 0;
#endif
    free((void*)w->w); q36_wfree(w->q); free(w->sc); free(w->q4); free(w->sg); q36_wfree(w->h);
    w->w = NULL; w->q = NULL; w->sc = NULL; w->q4 = NULL; w->sg = NULL; w->ng = 0; w->h = NULL;
}

/* ---------- per-layer dense weights ---------- */
typedef struct {
    float *in_ln, *post_ln, *qn, *kn, *gate_bias;
    QW q, k, v, o, gate;
    QW sh_g, sh_u, sh_d; float *sh_gate;   /* shared expert (dense, int8-during-load) + shared_expert_gate */
    /* Gated DeltaNet (linear_attention) dense weights (f16->f32 via st_read_f32). */
    QW dn_qkv, dn_z; float *dn_b, *dn_a;   /* in_proj_qkv/z (int8-during-load), b/a (not dense-matmul'd) */
    float *dn_conv;                        /* conv1d.weight [conv_dim, convk] (groups=conv_dim) */
    float *dn_dtbias, *dn_alog;            /* dt_bias[vh], A_log[vh] */
    float *dn_norm;                        /* RMSNormGated weight [vdim] */
    QW dn_out;                             /* out_proj [hidden, value_dim] */
    /* VRAM copies the tier placed (qt_dense handle + 1, 0 = stays on the CPU):
     * the DeltaNet out_proj, the attention q/k/v/o and the shared expert's
     * three matrices. Offered per layer as "dnout", "attnproj", "shexp";
     * see trunk_offer_dense / trunk_place_dense. */
    int qth_dnout, qth_q, qth_k, qth_v, qth_o, qth_shg, qth_shu, qth_shd;
} Layer;

/* ---------- LRU expert cache (int8 weights + per-row float scales) ---------- */
/* pw: the expert as expert_ffn.h wants it (planar int4, gate|up|down), the
 * only weight copy a slot holds when the shared kernel is active; g/u/d and
 * g4/u4/d4 are then NULL. */
typedef struct { int eid; int pinned; int is_int4; int8_t *g, *u, *d; uint8_t *g4, *u4, *d4; uint8_t *pw; float *gs, *us, *ds; uint64_t used;
                 unsigned busy; /* callers computing from this slot (expert_hold): never evicted */ } Slot;
typedef struct {
    Slot *slots;
    int *slot_by_expert;                  /* expert id -> resident slot, -1 if absent */
    int n, cap;
} LCache;

/* CACHE_ROUTE telemetry (docs/CACHE_ROUTE.md): the lever changes which experts
 * run, so it carries its own meters. Only touched when the lever is on. */
typedef struct {
    uint64_t slots, swaps, swaps_vram;    /* chosen slots; not in the true top-K; of those, VRAM-resident */
    uint64_t agree_hit, agree_tot;        /* |chosen ∩ true top-K| summed, K summed */
    double kl_sum; uint64_t kl_n;         /* mean KL(true top-K mass || chosen mass) */
} RouteStats;

/* One conversation's state for a multiplexed serve (KV_SLOTS>1, serve_mux below):
 * what a forward reads and writes that belongs to one token sequence. The Model holds
 * the conversation being prefilled; the others wait here, and q36_seq_swap trades the
 * two sets. A multiplexed decode step parks them all and reads each row's from its
 * Q36Row. mpos and rope_delta are an image turn's rope positions: the decode rows of
 * that turn sit past them. */
typedef struct {
    float **K, **V, **DN_rec, **DN_conv;
    int kv_len;
    kv_prefix kvp;
    int *mpos, mpos_len, rope_delta;
} Q36Seq;
typedef struct { Q36Seq *seq; int pos; } Q36Row;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    float *embed, *final_norm;
    uint16_t *embed_h;      /* COLI_DENSE_BITS=16: the table in f16, embed is NULL */
    QW lm_head;
    Layer *L;
    LCache *cache;          /* [n_layers] */
    int *active_of;         /* [n_layers] original->active idx (Phase 2: identity for all layers) */
    float **DN_rec;         /* [n_layers] recurrent state S[h]=[kdim,vdim] for DeltaNet layers (NULL for attn) */
    float **DN_conv;        /* [n_layers] conv ring [conv_dim, convk-1] for DeltaNet layers (NULL for attn) */
    /* DeltaNet layers on the GPU (Q36_DN_GPU=1, qt_dn_gpu_*): the host arrays
     * above stay canonical; per layer, dn_dev_fresh says the device holds the
     * newest state (nothing to upload before a GPU step), dn_host_stale says
     * the device advanced past the host copy (download before any CPU use:
     * a CPU step, a snapshot, a reset that must not be undone). */
    uint8_t *dn_dev_fresh, *dn_host_stale; int dn_dev;
    uint64_t clock, hits, miss;
    RouteStats route;          /* CACHE_ROUTE / ROUTE_AGREE meters */
    /* Telemetria per la dashboard (Brain/Profile): tempo di lettura esperti
     * accumulato dall'avvio, e bitmap degli esperti toccati nel turno. */
    double t_disk;
    uint8_t **ehit;
    float **K, **V; int kv_len, max_t, kv_cap;
    /* What the current attention and DeltaNet state was built from, so a
     * turn that resends the transcript prefills only the new tail. The ids
     * are recorded where they are fed, never derived from a counter: see
     * kv_prefix.h for why that distinction is the whole safety argument. */
    kv_prefix kvp;
    float *attn_sc;            /* [attn_sc_thr * kv_cap] score rows, one per thread */
    int attn_sc_thr;
    double dense_load_s;
    uint32_t *freq;
    int freq_token_count, hot_pinned, hot_n, warmup_tokens, token_count;
    float *momentum_logits;
    float pilot_smooth, pilot_conf_limit;
    uint8_t *is_pinned;
    uint8_t *is_queued;
    uint8_t *seen;             /* prefill-collected experts (COLIBRI_RESIDENT) */
    int resident_mode;         /* 0 off; 1 pin this-prompt experts (CPU no-evict -> GPU resident) */
    int resident_collecting;   /* prefill in progress, collecting routed experts */
    int vk_hist;               /* the Vulkan tier is on: routes counted into route_trace.h's history */
    int first_step;            /* the first step() call is the prefill */
    /* Vision, for the turn that carries an image. vis_map maps an ABSOLUTE
     * position to a row of vis_rows or -1; mpos holds the (t, h, w) rope
     * positions of the prompt, and past it every axis is pos + rope_delta
     * (HF's mrope_position_deltas). All of it is dropped after the turn. */
    Q38Vision vis; int vis_ready;
    float *vis_rows; int vis_rows_n;
    int *vis_map, vis_map_len;
    int *mpos, mpos_len, rope_delta;
#ifdef COLI_VULKAN
    void *vkchain;             /* the dense chain's device state (qwen36_chain.h), NULL until it runs */
    void *vkchain2;            /* its layers on COLI_VK_DEV2's device, after the primary's (qwen36_chain.h) */
#endif
    /* A speculative verify (prompt lookup, q36_spec_step) copies the DeltaNet state
     * after each of its first snap_rows rows, row r into slot r (0 = no copy), so a
     * draft rejected after row r+1 rolls back by swapping slot r in. snap_slots: the
     * slots allocated, the first time a verify that deep runs. */
    int snap_rows, snap_slots;
    float **snap_rec[Q36_SPEC_SNAPS], **snap_conv[Q36_SPEC_SNAPS];
    /* A multiplexed decode step (q36_step_rows): row s is the token at
     * mux_rows[s].pos of mux_rows[s].seq, so the attention and DeltaNet read and
     * write that conversation's state; NULL in every other forward. */
    const Q36Row *mux_rows;
} Model;

static pthread_mutex_t g_pilot_mx = PTHREAD_MUTEX_INITIALIZER;
static struct { int l, e; } pilot_q[4096];
static volatile unsigned pilot_r = 0, pilot_w = 0;
static Model *pilot_m = NULL;
static int g_pilot = 0;
static int g_wide  = 1;
/* CACHE_ROUTE family, same names and defaults as the GLM engine (docs/CACHE_ROUTE.md). */
static int   g_cache_route = 0;
static int   g_route_j     = 2;
static int   g_route_m     = 12;
static float g_route_p     = 0.f;
static float g_route_alpha = 1.f;
static int   g_route_agree = 0;

static void pilot_prefetch(Model *m, int lnext, const float *x, int S);
static void *pilot_worker(void *arg);
static void ensure_pilot_worker_started(Model *m);
static void slot_ensure_allocated(Model *m, Slot *s);
/* The cap main() must hand to qt_init.
 *
 * model_init_range resolves the cap<=0 "auto" sentinel into its own local copy
 * and writes the result to every layer's cache; main()'s variable keeps the
 * sentinel. qt_init refuses the VRAM expert tier for any cap != n_experts
 * outside fp8-stream mode (qwen36_tier.c:536), so passing the unresolved 0
 * switched the tier off under COLI_CUDA=1 even when auto-sizing had picked
 * every expert. Found by review on #1747, not by a run: the guard sits behind
 * COLI_CUDA and the CPU build links the inline stub.
 *
 * Pure on purpose, same convention as qwen36_cap_for_ram below: no Model
 * pointer, no globals, no I/O, so test_qwen36_cap_precedence.c can pin it
 * without a container. */
static int qwen36_resolved_cap(int cap, const LCache *cache, int n_layers) {
    if (cap > 0 || !cache || n_layers < 1) return cap;
    return cache[0].cap;
}

static int qwen36_cap_for_ram(double resident_gb, double avail_gb, double ram_gb_override,
                               int hidden, int inter, int n_experts, int n_active_layers,
                               int is_int4, double *slot_gb_out, double *budget_gb_out);

#ifdef COLI_CACHE_INDEX_TEST
static uint64_t g_slot_index_probes;
#endif

/* Runtime callers hold g_pilot_mx.  Validate both sides so stale bookkeeping
 * can only become a cache miss, never a wrong-expert hit. */
static Slot *slot_indexed(Model *m, int layer, int eid) {
    if (layer < 0 || layer >= m->c.n_layers || eid < 0 ||
        eid >= m->c.n_experts) return NULL;
    LCache *lc = &m->cache[layer];
    if (!lc->slot_by_expert) return NULL;
#ifdef COLI_CACHE_INDEX_TEST
    g_slot_index_probes++;
#endif
    int i = lc->slot_by_expert[eid];
    if (i < 0 || i >= lc->n || lc->slots[i].eid != eid) return NULL;
    return &lc->slots[i];
}

static void cache_unindex(Model *m, int layer, Slot *s) {
    LCache *lc = &m->cache[layer];
    int eid = s->eid, i = (int)(s - lc->slots);
    if (lc->slot_by_expert && eid >= 0 && eid < m->c.n_experts &&
        lc->slot_by_expert[eid] == i)
        lc->slot_by_expert[eid] = -1;
}

static void cache_hide(Model *m, int layer, Slot *s) {
    cache_unindex(m, layer, s);
    s->eid = -1;
}

static void cache_publish(Model *m, int layer, Slot *s, int eid) {
    LCache *lc = &m->cache[layer];
    cache_unindex(m, layer, s);
    s->eid = eid;
    if (lc->slot_by_expert && eid >= 0 && eid < m->c.n_experts)
        lc->slot_by_expert[eid] = (int)(s - lc->slots);
}

/* A slot someone is computing from, as colibri.c's ESlot.in_flight: taken
 * under g_pilot_mx at lookup (expert_hold), dropped without it when the
 * compute is done. A release without its hold would wrap the count and
 * leave the slot unevictable for good, so it stops here instead. */
static void slot_hold(Slot *s)    { __atomic_add_fetch(&s->busy, 1, __ATOMIC_ACQ_REL); }
static void slot_release(Slot *s) {
    if (__atomic_fetch_sub(&s->busy, 1, __ATOMIC_ACQ_REL) == 0) {
        fprintf(stderr, "qwen36: expert slot released more often than held\n"); abort();
    }
}
static int  slot_busy(const Slot *s) { return __atomic_load_n(&s->busy, __ATOMIC_ACQUIRE) != 0; }

/* The slot a full layer cache gives up, caller holds g_pilot_mx: the least
 * recently used one that is neither being loaded (eid < 0) nor computed from
 * (busy), unpinned unless allow_pinned. -1 when there is none. The demand
 * path and the PILOT worker both pick here, so neither can evict an expert a
 * moe run is still reading. One whose expert the Vulkan tier holds goes first
 * (vkt_ram_first: RAM and VRAM do not keep the same experts when RAM is short). */
static int slot_victim(const LCache *lc, int layer, int allow_pinned) {
    int lru = -1, dev = -1;
    for (int i = 0; i < lc->n; i++) {
        const Slot *s = &lc->slots[i];
        if (s->eid < 0 || slot_busy(s) || (s->pinned && !allow_pinned)) continue;
        if (vkt_ram_first(layer, s->eid)) { if (dev < 0 || s->used < lc->slots[dev].used) dev = i; continue; }
        if (lru < 0 || s->used < lc->slots[lru].used) lru = i;
    }
    return dev >= 0 ? dev : lru;
}

static void ensure_pilot_worker_started(Model *m) {
    if (!pilot_m) {
        pilot_m = m;
        pthread_t t;
        if (pthread_create(&t, NULL, pilot_worker, NULL) != 0) {
            fprintf(stderr, "Error: Failed to create pilot prefetch worker thread\n");
            exit(1);
        }
        pthread_detach(t);
    }
}

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
/* same wrapper convention as colibri.c/olmoe.c/inkling.c/kimi_k3.c/deepseek_v4.c */
static double mem_available_gb(void) { return compat_mem_available_gb(); }

/* ---- M-PROF (R2): per-phase wall-clock accumulators, COLI_TIMERS=1 ---- */
static int g_timers = -1;
static double g_tm_dec[6], g_tm_pre[6];   /* 0=deltanet 1=attention 2=moe_total 3=shared 4=router 5=lm_head */
static long g_tm_dec_tokens = 0, g_tm_pre_tokens = 0;
static double tm_now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3 + ts.tv_nsec/1e6; }
static int tm_on(void){ if(g_timers<0){ const char *e=getenv("COLI_TIMERS"); g_timers = (e && *e=='1'); } return g_timers; }
double g_qt_iss=0, g_qt_cpu=0, g_qt_tak=0;   /* QTIER-Phasen (Decode) */
double g_dn_sub[4];                           /* DN: proj, conv+split, l2n+rec, norm+out */
double g_tm_step=0;                           /* step() total (decode) */
static double g_xf_load=0, g_xf_run=0;        /* expert_ffn path: expert fetch (misses) vs compute, decode */
static double g_tm_win_moe=0; static int g_tm_win_n=0;
static void tm_add(int S, int idx, double ms){
    if(S==1){
        g_tm_dec[idx]+=ms;
        if(idx==2) g_tm_win_moe+=ms;
        if(idx==5 && ++g_tm_win_n==32){
            if(tm_on()) fprintf(stderr,"[timers] window: moe %.0f ms/token (last 32)\n", g_tm_win_moe/32.0);
            g_tm_win_moe=0; g_tm_win_n=0;
        }
    } else g_tm_pre[idx]+=ms;
}
static void tm_report(void){
    if(!tm_on()) return;
    static const char *nm[6]={"deltanet","attention","moe_total","(shared)","(router)","lm_head"};
    fprintf(stderr,"[timers] decode: %ld tokens  (shared/router are subsets of moe_total)\n", g_tm_dec_tokens);
    double sum=0;
    for(int i=0;i<6;i++){
        fprintf(stderr,"[timers]   %-10s %9.1f ms  %8.2f ms/token\n",
                nm[i], g_tm_dec[i], g_tm_dec_tokens? g_tm_dec[i]/g_tm_dec_tokens:0.0);
        if(i!=3&&i!=4) sum+=g_tm_dec[i];
    }
    fprintf(stderr,"[timers]   %-10s %9.1f ms  %8.2f ms/token\n","TOTAL",sum,g_tm_dec_tokens?sum/g_tm_dec_tokens:0.0);
    if(g_tm_step>0)
        fprintf(stderr,"[timers]   step() total: %.1f ms/token (outside the phases: %.1f)\n",
            g_tm_step/g_tm_dec_tokens,
            (g_tm_step-(g_tm_dec[0]+g_tm_dec[1]+g_tm_dec[2]+g_tm_dec[5]))/g_tm_dec_tokens);
    if(g_dn_sub[0]+g_dn_sub[1]+g_dn_sub[2]+g_dn_sub[3]>0)
        fprintf(stderr,"[timers]   dn-sub: proj %.1f | conv %.1f | l2n+rec %.1f | norm+out %.1f ms/token\n",
            g_dn_sub[0]/g_tm_dec_tokens,g_dn_sub[1]/g_tm_dec_tokens,g_dn_sub[2]/g_tm_dec_tokens,g_dn_sub[3]/g_tm_dec_tokens);
    if(g_xf_load+g_xf_run>0)
        fprintf(stderr,"[timers]   expert kernel: fetch %.2f | compute %.2f ms/token\n",
                g_xf_load/g_tm_dec_tokens, g_xf_run/g_tm_dec_tokens);
    if(g_qt_iss+g_qt_cpu+g_qt_tak>0)
        fprintf(stderr,"[timers]   qtier: issue %.2f | cpu-miss %.2f | take %.2f ms/token\n",
                g_qt_iss/g_tm_dec_tokens, g_qt_cpu/g_tm_dec_tokens, g_qt_tak/g_tm_dec_tokens);
    fprintf(stderr,"[timers] prefill: %ld tokens  dn=%.0f attn=%.0f moe=%.0f(sh=%.0f rt=%.0f) head=%.0f ms\n",
            g_tm_pre_tokens,g_tm_pre[0],g_tm_pre[1],g_tm_pre[2],g_tm_pre[3],g_tm_pre[4],g_tm_pre[5]);
}
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

#include "matmul_f32.h"   /* y[S,O] = x[S,I] @ W^T, W [O,I] f32 row-major */

/* y[1,O] = x[1,I] @ W^T with W quantized: q[O,I] int8 + scale per row.
 * matmul_q lives in qgemv.h so tests/test_qgemv.c can link the exact kernel
 * the engine runs (qwen36.c has a main() and cannot itself be linked into a
 * test binary). */
#include "qgemv.h"

/* Multi-row dense-int8 prefill kernel.  matmul_q() above is deliberately kept
 * as the S=1 decode implementation: its four AVX accumulators stay in
 * registers and its reduction order is covered by the token-exact oracle.
 *
 * For prompt rows, process two independent activations per weight decode.  A
 * two-row tile leaves enough AVX2 registers for both sets of four accumulators
 * plus the converted weights; a four-row tile spills on the x86-64-v3 target.
 * Each row uses the exact same FMA streams and reduction tree as matmul_q(), so
 * batching changes neither a float bit nor the decode path. */
static void matmul_q_batch(float *y, const float *x, const int8_t *q,
                           const float *scale, int S, int I, int O) {
#if defined(__AVX2__) && defined(__FMA__)
    /* Every shipping Qwen3.6 dense input width is 32-aligned.  Keep unusual
     * checkpoint shapes on the literal historical kernel instead of trying
     * to make two interleaved scalar tails depend on compiler contraction. */
    if (I & 31) {
        for (int s = 0; s < S; s++)
            matmul_q(y+(int64_t)s*O, x+(int64_t)s*I, q, scale, I, O);
        return;
    }
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        int row = 0;
        for (; row + 1 < S; row += 2) {
            const float *x0 = x + (int64_t)row * I;
            const float *x1 = x0 + I;
            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
            __m256 a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
            __m256 a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 32 <= I; i += 32) {
                __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
                __m128i b1 = _mm_loadu_si128((const __m128i*)(w + i + 16));
                __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0));
                __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8)));
                __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1));
                __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8)));
                a00 = _mm256_fmadd_ps(_mm256_loadu_ps(x0+i),    w0, a00);
                a01 = _mm256_fmadd_ps(_mm256_loadu_ps(x0+i+8),  w1, a01);
                a02 = _mm256_fmadd_ps(_mm256_loadu_ps(x0+i+16), w2, a02);
                a03 = _mm256_fmadd_ps(_mm256_loadu_ps(x0+i+24), w3, a03);
                a10 = _mm256_fmadd_ps(_mm256_loadu_ps(x1+i),    w0, a10);
                a11 = _mm256_fmadd_ps(_mm256_loadu_ps(x1+i+8),  w1, a11);
                a12 = _mm256_fmadd_ps(_mm256_loadu_ps(x1+i+16), w2, a12);
                a13 = _mm256_fmadd_ps(_mm256_loadu_ps(x1+i+24), w3, a13);
            }
            a00 = _mm256_add_ps(_mm256_add_ps(a00,a01), _mm256_add_ps(a02,a03));
            a10 = _mm256_add_ps(_mm256_add_ps(a10,a11), _mm256_add_ps(a12,a13));
            __m128 s0 = _mm_add_ps(_mm256_castps256_ps128(a00), _mm256_extractf128_ps(a00,1));
            __m128 s1 = _mm_add_ps(_mm256_castps256_ps128(a10), _mm256_extractf128_ps(a10,1));
            s0 = _mm_add_ps(s0, _mm_movehl_ps(s0,s0));
            s1 = _mm_add_ps(s1, _mm_movehl_ps(s1,s1));
            s0 = _mm_add_ss(s0, _mm_shuffle_ps(s0,s0,1));
            s1 = _mm_add_ss(s1, _mm_shuffle_ps(s1,s1,1));
            y[(int64_t)row*O + o] = _mm_cvtss_f32(s0) * scale[o];
            y[(int64_t)(row+1)*O + o] = _mm_cvtss_f32(s1) * scale[o];
        }
        if (row < S) {
            const float *xs = x + (int64_t)row * I;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 32 <= I; i += 32) {
                __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
                __m128i b1 = _mm_loadu_si128((const __m128i*)(w + i + 16));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i),    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i+16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i+24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8))), a3);
            }
            a0 = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
            __m128 ss = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
            ss = _mm_add_ps(ss, _mm_movehl_ps(ss,ss));
            ss = _mm_add_ss(ss, _mm_shuffle_ps(ss,ss,1));
            y[(int64_t)row*O + o] = _mm_cvtss_f32(ss) * scale[o];
        }
    }
#else
    /* Other ISAs retain the established implementation until they have an
     * independently exact multi-row kernel. */
    for (int s = 0; s < S; s++)
        matmul_q(y+(int64_t)s*O, x+(int64_t)s*I, q, scale, I, O);
#endif
}

/* Group-scaled int8 GEMV: one f32 scale per `gs` input elements per row
 * (gs64 expert containers). Row layout of `scale`: [O][I/gs] row-major.
 * matmul_q_gs lives in gsgemv.h so tests/test_gsgemv.c can link the exact
 * kernel the engine runs. */
static int g_expert_gs = 0;   /* set from qwen36_meta.json (expert_gs) at load */
#include "gsgemv.h"

static int g_expert_mixed = 0;   /* mixed layout on disk (int4 gate/up, int8 down) */
static int g_expert_down_gs = 0; /* down_proj's group size in the mixed layout (0 = per row) */
/* 1 = expert container packs int4 (tier fmt=4); 0 = int8 per-row (tier fmt=1).
 * Same signal main's nbytes probe and tier_warmstart receive; the decode path
 * needs it to offer int8 experts (#1391): on an int8 container e->g4 is NULL. */
static int g_expert_is_int4 = 1;

/* Shared expert kernel (expert_ffn.h): routed experts stay planar int4 in
 * RAM and a layer runs as (expert, row-chunk) items. On by default for an
 * int4 gs=64 container whose widths are multiples of 64, off under the CUDA
 * expert tier (it uploads the pair-layout int4 and computes misses from the
 * int8 copy) and with QWEN_EXPERT_KERNEL=0, which keeps the historical
 * unpack-to-int8 path for A/Bs. Decided once from the container itself. */
static int container_layer_is_int4(Model *m, int layer);
/* The routed experts take the same integer path as the dense trunk
 * (expert_ffn.h mode 1: activation to int8 once per row, dpbusd against the
 * planar nibbles). Measured on the 35B: expert compute 22.7 to 15.9
 * ms/token for +0.1% perplexity. QWEN_EXPERT_ACT=f32 restores mode 0, f32
 * activations and the bit-identical contract with the pair kernels. */
static int xf_act_mode(void){ static int v=-1; if(v<0){ const char *e=getenv("QWEN_EXPERT_ACT"); v=(e&&!strcmp(e,"f32"))?0:1; } return v; }
static int xf_mode(Model *m) {
    static int v = -1;
    if (v >= 0) return v;
    const char *e = getenv("QWEN_EXPERT_KERNEL");
    int on = !(e && *e == '0');
#ifdef COLI_CUDA
    { const char *cu = getenv("COLI_CUDA"); if (cu && *cu == '1') on = 0; }
#endif
    Cfg *c = &m->c;
    if (c->expert_gs != XF_BLOCK || !xf_layout_ok(c->hidden) || !xf_layout_ok(c->inter)) on = 0;
    if (on) {
        int probe = -1;
        for (int l = 0; l < c->n_layers && probe < 0; l++) {
            char nm[256];
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.0.merged_weight", m->active_of[l]);
            if (st_find(&m->S, nm)) probe = container_layer_is_int4(m, m->active_of[l]);
        }
        if (probe != 1) on = 0;
    }
    v = on;
    if (v) fprintf(stderr, "[qwen36] expert kernel: planar int4 in RAM (expert_ffn.h), QWEN_EXPERT_KERNEL=0 restores int8 unpack\n");
    return v;
}

/* The single offer decision the decode path makes for a routed expert: offer
 * whichever format the container actually packed, exactly what tier_warmstart
 * does (int4 → the packed g4/u4/d4, int8 → the live RAM weights, tier fmt=1
 * since #1334). moe()'s resident offer, the pilot-prefetch lookahead, and the
 * #1391 test all call THIS function, so the gate can't drift between them.
 * Read-only: never frees, never rewrites -- ownership stays in warmstart
 * (#1341). */
static void tier_offer_slot(int layer, int eid, const Slot *s) {
    if (s->g4)
        qt_note(layer, eid, s->g4, s->u4, s->d4, s->gs, s->us, s->ds);
    else if (!g_expert_is_int4 && s->g)
        qt_note(layer, eid, (const uint8_t *)s->g, (const uint8_t *)s->u,
                (const uint8_t *)s->d, s->gs, s->us, s->ds);
}
/* Expert-GEMV dispatch: per-row scales (classic) or grouped (gs64 container). */
static void matmul_qe(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    if (g_expert_gs) matmul_q_gs(y, x, q, scale, I, O, g_expert_gs);
    else matmul_q(y, x, q, scale, I, O);
}
/* down_proj: in the mixed layout it carries its own scale layout (int8, per
 * row or expert_down_gs), everywhere else it is matmul_qe. */
static void matmul_qd(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    if (!g_expert_mixed) { matmul_qe(y, x, q, scale, I, O); return; }
    if (g_expert_down_gs) matmul_q_gs(y, x, q, scale, I, O, g_expert_down_gs);
    else matmul_q(y, x, q, scale, I, O);
}

/* ---- Dense int8: per-row quantized copies of the large f32 matrices.
 * matmul_d dispatches directly off QW.q (no pointer-keyed scan -- see QW,
 * above Layer); COLI_DENSE_I8=0 falls back to f32 (QW.w, reference path for
 * parity tests). ~4x less memory traffic. */
/* ---- Dense trunk, integer dot products ------------------------------------
 *
 * Every dense GEMV of a token (DeltaNet projections and out_proj, attention
 * q/k/v/o, the shared expert, lm_head) used to multiply int8 weights by f32
 * activations: each weight byte converted to f32 and fed to an FMA, eight
 * weights per instruction. Measured on lm_head (248320 x 2048 int8, 508 MB)
 * that runs at 29 GB/s on a 16-core AVX-512 host whose memory bus does 80:
 * the kernel, not the bus, was the limit, and the dense part of a token is
 * 1.9 GB of int8 on the 35B, three times the routed experts.
 *
 * The activation is now quantized to int8 once per call (one scale,
 * amax/127, the qrow_i8 contract the expert IDOT already uses) and the dot
 * is integer: 32 weights per instruction on AVX2 (maddubs), 64 on AVX-512
 * VNNI, exact int32 sums scaled once per output. Not bit-identical to the
 * f32 path (the activation is rounded): measured on the 35B, +1.0%
 * perplexity on 4 x 512 tokens, lm_head 12.6 to 10.2 ms/token, decode
 * 6.71 to 7.35 tok/s alone and 8.23 with the experts' int8 activations.
 * COLI_DENSE_IDOT=0 restores the f32-activation kernel.
 *
 * COLI_DENSE_BITS=4 additionally stores the dense matrices as int4 in blocks
 * of 64 with one scale per block, the K1b planar layout, halving the bytes
 * the token reads; lm_head alone goes from 508 to 254 MB. It implies the
 * integer dot (that layout has no f32 kernel). Same gate: measured. */
static int dense_idot_on(void){ static int v=-1; if(v<0){ const char *e=getenv("COLI_DENSE_IDOT"); v=!(e&&*e=='0'); } return v; }
/* COLI_DENSE_BITS: 8 (default) int8 rows, 4 the int4 planar copy per
 * COLI_DENSE_INT4, 16 the container's own f16 values (no quantization: twice
 * the RAM of int8, the precision the checkpoint was converted at). */
static int dense_bits(void){ static int v=-1; if(v<0){ const char *e=getenv("COLI_DENSE_BITS"); int b=e?atoi(e):8; v=b==4?4:b==16?16:8; } return v; }

/* f32 rows -> int4 in blocks of 64 with one f32 scale per block, packed as the
 * K1b planar layout (unsigned nibbles v+8, block b: lo nibbles = elements
 * b*64..b*64+31, hi = b*64+32..b*64+63). The quantizer is the symmetric
 * absmax/7 the expert containers use. */
static void pack_int4_g64_planar(const float *w, uint8_t *q4, float *sg, int O, int I){
    int rb = I / 2, ng = I / 64;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        uint8_t *row = q4 + (int64_t)o * rb;
        float *sr = sg + (int64_t)o * ng;
        for (int g = 0; g < ng; g++) {
            const float *blk = wr + g * 64;
            float amax = 0.f;
            for (int k = 0; k < 64; k++) { float a = fabsf(blk[k]); if (a > amax) amax = a; }
            /* absmax/7 is where the search starts; the scale is then refined
             * by least squares against the rounded codes (s = w.q / q.q) and
             * the candidate with the smallest squared error wins. Three
             * rounds: measured on the 35B this recovers a third of the
             * perplexity absmax alone loses at 4 bits. */
            float best_s = amax / 7.f; if (best_s < 1e-8f) best_s = 1e-8f;
            int best_q[64]; double best_err = 1e30;
            float s = best_s;
            for (int round = 0; round < 4; round++) {
                int q[64]; double err = 0, wq = 0, qq = 0;
                float inv = 1.f / s;
                for (int k = 0; k < 64; k++) {
                    int v = (int)lrintf(blk[k] * inv); if (v > 7) v = 7; if (v < -8) v = -8;
                    q[k] = v; double d = (double)blk[k] - (double)v * s; err += d * d;
                    wq += (double)blk[k] * v; qq += (double)v * v;
                }
                if (err < best_err) { best_err = err; best_s = s; memcpy(best_q, q, sizeof q); }
                if (qq <= 0) break;
                float ns = (float)(wq / qq);          /* least-squares scale for these codes */
                if (ns <= 0.f || ns == s) break;
                s = ns;
            }
            sr[g] = best_s;
            uint8_t *dst = row + g * 32;
            for (int k = 0; k < 32; k++)
                dst[k] = (uint8_t)((best_q[k] + 8) | ((best_q[k + 32] + 8) << 4));
        }
    }
}
#ifdef COLI_QWEN_BATCH_TEST
static uint64_t g_qwen_matmul_d_calls;
#endif
static int dense_i8_on(void){ static int v=-1; if(v<0){ const char *e=getenv("COLI_DENSE_I8"); v=!(e&&*e=='0'); } return v; }
/* COLI_KV_PREFIX=0: never reuse a previous turn's state. Kept as an escape
 * hatch and as the B arm of the A/B that shows reuse changes nothing but
 * the time. */
static int kv_prefix_off(void){ const char *e=getenv("COLI_KV_PREFIX"); return e && *e=='0'; }
static int dense_batch_on(void){ const char *e=getenv("QWEN_DENSE_BATCH"); return !(e&&*e=='0'); }
/* COLI_DENSE_INT4 names which components take the int4 copy when
 * COLI_DENSE_BITS=4: a comma list of lmhead, dnproj, dnout, attn, shexp,
 * router; unset means all of them. The parts of the trunk pay 4 bits
 * differently (measured: the attention projections and the DeltaNet input
 * projections cost the most perplexity, lm_head the least per byte saved),
 * so the default is chosen per component from the numbers, not for the
 * whole trunk at once. */
static int dense_int4_wanted(const char *tag){
    if (dense_bits() != 4 || !tag) return 0;
    const char *e = getenv("COLI_DENSE_INT4");
    if (!e || !*e) return 1;
    size_t n = strlen(tag);
    for (const char *p = e; *p; ) {
        while (*p == ',' || *p == ' ') p++;
        const char *q = p; while (*q && *q != ',' && *q != ' ') q++;
        if ((size_t)(q - p) == n && !strncmp(p, tag, n)) return 1;
        p = q;
    }
    return 0;
}
/* Per-row max-abs / round-clamp int8 quantization of an in-memory f32 matrix
 * W [O][I] row-major -- same math load_tq runs during streamed load, factored
 * out so it can also run on a caller-owned buffer directly (tests that
 * synthesize weights in memory, without a shard file to load from). Does not
 * touch out->w -- the caller sets that (or leaves it, e.g. load_tq frees it
 * right after). `tag` selects the int4 planar copy per dense_int4_wanted
 * (NULL/COLI_DENSE_BITS!=4: skipped, out->q4 stays NULL). */
/* Whether a matrix that got its int4 copy keeps the int8 one too. Only the CUDA
 * placer reads it (it uploads int8 rows); on the CPU matmul_d takes the int4 copy
 * whenever there is one, so keeping both doubled the trunk for nothing: measured on
 * Qwen3.8-27B with COLI_DENSE_BITS=4, 42.4 GB resident against 29.1 in int8 (#1757).
 * COLI_DENSE_KEEP_I8=1 keeps both. */
static int dense_keep_i8(void){
    static int v = -1;
    if (v < 0) {
        v = getenv("COLI_DENSE_KEEP_I8") && getenv("COLI_DENSE_KEEP_I8")[0] == '1';
#ifdef COLI_CUDA
        { const char *cu = getenv("COLI_CUDA"); if (cu && *cu == '1') v = 1; }
#endif
    }
    return v;
}
static void qw_quantize(const float *W, int I, int O, const char *tag, QW *out) {
    int8_t *q = q36_walloc((size_t)O*I); float *sc = malloc((size_t)O*sizeof(float));
    if (!q || !sc) { fprintf(stderr, "OOM qw_quantize\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *r = W + (int64_t)o*I; float am = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am/127.f : 1.f; sc[o] = s; float inv = 1.f/s;
        int8_t *d = q + (int64_t)o*I;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(r[i]*inv); if (v>127) v=127; if (v<-127) v=-127; d[i] = (int8_t)v; }
    }
    out->q = q; out->sc = sc; out->I = I; out->O = O;
    out->q4 = NULL; out->sg = NULL; out->ng = 0;
    if (dense_int4_wanted(tag) && I % 64 == 0) {
        uint8_t *q4 = malloc((size_t)O*(I/2)); float *sg = malloc((size_t)O*(I/64)*sizeof(float));
        if (q4 && sg) {
            pack_int4_g64_planar(W, q4, sg, O, I);
            out->q4 = q4; out->sg = sg; out->ng = I/64;
            if (!dense_keep_i8()) { q36_wfree(out->q); free(out->sc); out->q = NULL; out->sc = NULL; }
        } else { free(q4); free(sg); }
    }
}
#ifdef COLI_VULKAN
/* COLI_VULKAN=1: a resident dense matrix answers from the Vulkan device, with
 * the weights the CPU would read, in the order matmul_d picks them: the int4
 * copy when there is one, else the int8 rows (fmt 1, one scale per row), else
 * the f32 matrix (fmt 10, COLI_DENSE_I8=0: the reference path). The planar
 * int4 blocks are not the shader's layout (fmt 4: low nibble = even column),
 * so they are repacked once at upload, nibble for nibble: the same codes v+8
 * and the same per-64 scales, nothing requantized. For the quantized copies
 * only the activation differs from the CPU's default: it stays f32 on the
 * device, where the integer dot rounds it to int8 (COLI_DENSE_IDOT=0 is the
 * CPU arm with the same arithmetic); the f32 matrix differs from the CPU only
 * in the order of the sums. The device copy lives in w->vk from the first
 * call; a failed upload sets vk_off and the matrix stays on the CPU.
 * The backend has one command buffer: never from a parallel region. */
static void vk_q4_planar_to_fmt4(const QW *w, uint8_t *dst) {
    const int I = w->I, rb = I / 2, ng = I / 64;
    for (int o = 0; o < w->O; o++) {
        const uint8_t *src = w->q4 + (size_t)o * rb;
        uint8_t *row = dst + (size_t)o * rb;
        for (int g = 0; g < ng; g++)
            for (int p = 0; p < 32; p++) {
                int lo = 2 * p, hi = 2 * p + 1;   /* columns g*64+lo, g*64+hi */
                uint8_t a = lo < 32 ? (src[g * 32 + lo] & 15) : (src[g * 32 + lo - 32] >> 4);
                uint8_t b = hi < 32 ? (src[g * 32 + hi] & 15) : (src[g * 32 + hi - 32] >> 4);
                row[g * 32 + p] = (uint8_t)(a | (b << 4));
            }
    }
}
static unsigned g_vk_placed[4];   /* uploads by format: int8 rows, int4, f32, f16 */
static int g_q36_vk_dev;          /* the device vk_qw_tensor uploads a new matrix to: 1 for the
                                   * chain's layers on COLI_VK_DEV2's (qwen36_chain.h) */
/* The format a matrix goes to the device in: the copy matmul_d reads (int4, int8 rows,
 * f16 with COLI_DENSE_BITS=16, else f32), its rows and its scales. */
static int vk_qw_fmt(const QW *w, const void **wq, const float **sc) {
    if (w->vk_gone) { *wq = NULL; *sc = NULL; return w->vk_fmt; }
    if (w->q4) { *wq = w->q4; *sc = w->sg; return 4; }
    if (w->q) { *wq = w->q; *sc = w->sc; return 1; }
    if (w->h) { *wq = w->h; *sc = NULL; return 14; }
    *wq = w->w; *sc = NULL;   /* fmt 10: no scales */
    return 10;
}
static void q36_dho_reload(QW *w);
/* w's device copy, uploaded on the first call (the int4 copy repacked); NULL when it
 * cannot be (vk_off then keeps the matrix on the CPU). The dense chain reads the same
 * tensor, so the per-matrix path and the chain never hold a matrix twice. With
 * COLI_VK_IMPORT and host copies retained (or an integrated GPU), the int8 and f16
 * rows are not copied: the device reads q36_walloc's pages in place, and a matrix
 * the import refuses is copied as before. */
static ColiVkTensor *vk_qw_tensor(const QW *w) {
    if (w->vk_off) return NULL;
    QW *mw = (QW *)w;   /* vk is a cache in a matrix the forward pass treats as read-only */
    ColiVkTensor **t = (ColiVkTensor **)&mw->vk;
    if (*t) return *t;
    if (w->vk_gone) return NULL;
    int I = w->I, O = w->O;
    const void *wq; const float *sc;
    int fmt = vk_qw_fmt(w, &wq, &sc), gs = fmt == 4 ? 64 : 0;
    if (!wq) return NULL;
    int ok;
    /* On an integrated device these are already one shared allocation. Retain
     * it instead of moving the same physical bytes into a smaller logical
     * device-local heap. q36_dho_drop keeps imported pages alive. */
    if (g_q36_vk_dev) {   /* the second device: a copy there (no import, no per-matrix path) */
        uint8_t *packed = w->q4 ? malloc((size_t)O * (I / 2)) : NULL;
        if (packed) vk_q4_planar_to_fmt4(w, packed);
        ok = (!w->q4 || packed) && coli_vk_tensor_ensure2(t, w->q4 ? (const void *)packed : wq, sc, fmt, I, O, gs);
        free(packed);
        if (!ok) { mw->vk_off = 1; return NULL; }
        g_vk_placed[fmt == 1 ? 0 : fmt == 4 ? 1 : fmt == 14 ? 3 : 2]++;
        return *t;
    }
    if (g_vk_import && (coli_vk_device_integrated() || !coli_vk_dense_device_only()) &&
        fmt != 4 && w->w != wq) {   /* q36_walloc'd rows (f32 rows are falloc's) */
        size_t rb = fmt == 1 ? (size_t)I : (size_t)I * 2;
        size_t alloc = (rb * (size_t)O + Q36_PAGE - 1) / Q36_PAGE * Q36_PAGE;
        if (coli_vk_tensor_import(t, wq, alloc, sc, fmt, I, O, gs)) {
            mw->vk_imported = 1;
            g_vk_placed[fmt == 1 ? 0 : 3]++;
            return *t;
        }
    }
    if (w->q4) {
        uint8_t *packed = malloc((size_t)O * (I / 2));
        if (packed) vk_q4_planar_to_fmt4(w, packed);
        ok = packed && coli_vk_tensor_ensure(t, packed, sc, fmt, I, O, gs);
        free(packed);
    } else ok = coli_vk_tensor_ensure(t, wq, sc, fmt, I, O, gs);
    if (!ok) { mw->vk_off = 1; return NULL; }
    g_vk_placed[fmt == 1 ? 0 : fmt == 4 ? 1 : fmt == 14 ? 3 : 2]++;
    return *t;
}
static int vk_dense_matmul(float *y, const float *x, const QW *w, int S, int I, int O) {
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    if (w->vk_off || w->I != I || w->O != O || S < 1 || S > 65535) return 0;
    const void *wq; const float *sc;
    int fmt = vk_qw_fmt(w, &wq, &sc), gs = fmt == 4 ? 64 : 0;
    if (!wq && !w->vk_gone) return 0;
    ColiVkTensor *t = vk_qw_tensor(w);
    if (!t || coli_vk_tensor_dev(t)) return 0;   /* a matrix of the second device's layers: only its chain reads it */
    /* a verify's rows (g_q36_rowwise) one at a time: a decode step's GEMV, not the batch's GEMM */
    if (g_q36_rowwise && S > 1) {
        for (int s = 0; s < S; s++)
            if (!coli_vk_matmul(&t, y + (int64_t)s * O, x + (int64_t)s * I, wq, sc, fmt, 1, I, O, gs)) return 0;
        return 1;
    }
    return coli_vk_matmul(&t, y, x, wq, sc, fmt, S, I, O, gs);
}
/* One line at the end of a run or a serve turn: how many matmuls the device
 * really answered, so a test can tell a used path from an initialised one. */
/* The tier's line and, while it keeps one, its history: at every run and turn end. */
static char g_vk_usage[2100];   /* where the tier's history lives (vk_tier_start) */
static void vk_tier_turn(Model *m, const char *scope) {
    if (!m->vk_hist) return;
    vkt_report(scope, m->hits, m->miss);
    rt_save(g_vk_usage, 1);
}
static void vk_report(void) {
    if (!g_vk_ready) return;
    size_t bytes = 0, tensors = 0;
    coli_vk_mem_info(&bytes, &tensors);
    char more[96] = "";
    size_t imp = coli_vk_imported_bytes();
    if (g_vk_placed[3] || imp)
        snprintf(more, sizeof more, ", f16 %u; %.1f MiB read in place", g_vk_placed[3], imp / 1048576.0);
    fprintf(stderr, "[VK] qwen36: %llu matmuls on the GPU (%zu matrices resident, %.1f MiB; placed int8 %u, int4 %u, f32 %u%s)\n",
            coli_vk_matmul_calls(), tensors, bytes / 1048576.0,
            g_vk_placed[0], g_vk_placed[1], g_vk_placed[2], more);
}
#endif
/* f32 -> f16, round to nearest even: exact for a value that came from f16 (the
 * container), which is the only use (COLI_DENSE_BITS=16). */
static uint16_t f32_to_f16_bits(float f){
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u, mant = x & 0x7fffffu;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = 1u << (shift - 1), rest = mant & ((1u << shift) - 1), v = mant >> shift;
        if (rest > half || (rest == half && (v & 1))) v++;
        return (uint16_t)(sign | v);
    }
    uint32_t v = ((uint32_t)exp << 10) | (mant >> 13), rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (v & 1))) v++;
    return (uint16_t)(sign | v);
}

/* y[S,O] = x[S,I] @ W^T with W in f16 (COLI_DENSE_BITS=16). A thread converts
 * 16 rows of W to f32 at a time and runs every row of x against them, four
 * outputs per pass over x, so W is read once per call and x stays in cache. */
#define Q36_H16_TILE 16
static inline void dot4_f32_lanes(const float *x, const float *w, int I, float *out){
    float a0[16] = {0}, a1[16] = {0}, a2[16] = {0}, a3[16] = {0};
    const float *w1 = w + I, *w2 = w + 2 * (int64_t)I, *w3 = w + 3 * (int64_t)I;
    int i = 0;
    for (; i + 16 <= I; i += 16)
        for (int j = 0; j < 16; j++) {
            float xv = x[i + j];
            a0[j] = MATMUL_F32_MADD(a0[j], xv, w[i + j]);
            a1[j] = MATMUL_F32_MADD(a1[j], xv, w1[i + j]);
            a2[j] = MATMUL_F32_MADD(a2[j], xv, w2[i + j]);
            a3[j] = MATMUL_F32_MADD(a3[j], xv, w3[i + j]);
        }
    for (int j = 0; i + j < I; j++) {
        float xv = x[i + j];
        a0[j] = MATMUL_F32_MADD(a0[j], xv, w[i + j]);
        a1[j] = MATMUL_F32_MADD(a1[j], xv, w1[i + j]);
        a2[j] = MATMUL_F32_MADD(a2[j], xv, w2[i + j]);
        a3[j] = MATMUL_F32_MADD(a3[j], xv, w3[i + j]);
    }
    for (int h = 8; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) { a0[j] += a0[j + h]; a1[j] += a1[j + h]; a2[j] += a2[j + h]; a3[j] += a3[j + h]; }
    out[0] = a0[0]; out[1] = a1[0]; out[2] = a2[0]; out[3] = a3[0];
}
static void matmul_h(float *y, const float *x, const uint16_t *W, int S, int I, int O){
    int tiles = (O + Q36_H16_TILE - 1) / Q36_H16_TILE;
    #pragma omp parallel
    {
        float *wt = malloc((size_t)Q36_H16_TILE * I * sizeof(float));
        if (!wt) { fprintf(stderr, "OOM in the f16 dense GEMM\n"); exit(1); }
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < tiles; t++) {
            int o0 = t * Q36_H16_TILE, n = O - o0 < Q36_H16_TILE ? O - o0 : Q36_H16_TILE;
            f16_to_f32_bulk(W + (int64_t)o0 * I, wt, (int64_t)n * I);
            for (int s = 0; s < S; s++) {
                const float *xs = x + (int64_t)s * I;
                float *ys = y + (int64_t)s * O + o0;
                int r = 0;
                for (; r + 4 <= n; r += 4) dot4_f32_lanes(xs, wt + (int64_t)r * I, I, ys + r);
                for (; r < n; r++) ys[r] = dot_f32_lanes(xs, wt + (int64_t)r * I, I);
            }
        }
        free(wt);
    }
}

static void matmul_d(float *y, const float *x, const QW *w, int S, int I, int O){
#ifdef COLI_QWEN_BATCH_TEST
    g_qwen_matmul_d_calls++;
#endif
#ifdef COLI_VULKAN
    if (g_vk_ready && g_vk_dense && (w->q || w->q4 || w->w || w->h || w->vk_gone) && vk_dense_matmul(y, x, w, S, I, O)) return;
    /* the CPU needs a matrix the device holds alone (a lost device): read it back */
    if (w->vk_gone) q36_dho_reload((QW *)w);
#endif
    if (w->h) { matmul_h(y, x, w->h, S, I, O); return; }
    if (w->q || w->q4) {
        if (w->q4 || dense_idot_on()) {
            /* integer dot: the activation rows to int8 once, then the K1b
             * grouped kernel (int4 planar) or the per-row int8 kernel */
            int ng = I / 64;
            int8_t *xq = malloc((size_t)S * I);
            float *sx = malloc((size_t)S * sizeof(float));
            int32_t *xsg = w->q4 ? malloc((size_t)S * ng * sizeof(int32_t)) : NULL;
            if (xq && sx && (!w->q4 || xsg)) {
                for (int s = 0; s < S; s++)
                    sx[s] = dense_act_i8(x + (int64_t)s * I, I, xq + (int64_t)s * I, xsg ? xsg + (int64_t)s * ng : NULL);
                if (w->q4) matmul_i4p_grouped_idot(y, xq, sx, xsg, w->q4, w->sg, S, I, O, 64);
                else       matmul_q_idot(y, xq, sx, w->q, w->sc, S, I, O);
                free(xq); free(sx); free(xsg);
                return;
            }
            free(xq); free(sx); free(xsg);      /* out of memory: the f32 path below */
            if (!w->q) { fprintf(stderr, "OOM in the int4 dense GEMV (no int8 copy to fall back on)\n"); exit(1); }
        }
        if (S > 1 && dense_batch_on())
            matmul_q_batch(y, x, w->q, w->sc, S, I, O);
        else
            for (int s = 0; s < S; s++) matmul_q(y+(int64_t)s*O, x+(int64_t)s*I, w->q, w->sc, I, O);
        return;
    }
    matmul(y, x, w->w, S, I, O);
}
/* A dense matrix the tier placed in VRAM (handle+1 kept in the Layer, 0 = CPU):
 * a device matmul, or 0 and the caller runs matmul_d as before. The
 * tier turns a failing handle off itself, so the fallback is permanent. */
static inline int qtd_batch(int hp1, float *y, const float *x, int S, int I, int O){
    return hp1 > 0 && qt_dense_matmul_batch(hp1 - 1, y, x, S, I, O);
}
static inline int qtd(int hp1, float *y, const float *x, int I, int O){
    return hp1 > 0 && qt_dense_matmul(hp1 - 1, y, x, I, O);
}
/* Bytes of w's dense-i8 copy (int8 rows + per-row scales), 0 when there is
 * none (COLI_DENSE_I8=0): nothing to offer, the CPU path stands. */
static size_t qdw_bytes(const QW *w){
    return w->q ? (size_t)w->I * w->O + (size_t)w->O * sizeof(float) : 0;
}
/* Upload w's dense-i8 copy to `dev`; handle+1, or 0 when it stays on the CPU. */
static int qdw_place(const QW *w, int dev){
    if (dev == QT_PLACE_CPU || !w->q) return 0;
    int h = qt_dense_init(w->q, w->sc, w->I, w->O, dev);
    return h >= 0 ? h + 1 : 0;
}

/* rmsnorm over a row of length D (in-place capable: out may == x).
 * Qwen3_5MoeRMSNorm: out = (x * rsqrt(mean(x^2)+eps)) * (1.0 + weight). */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * (1.0f + w[i]);
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* softplus(z) = log(1+exp(z)), stable for large z (HF GatedDeltaNet g_rule). */
static float softplus_f(float z) { return z > 20.f ? z : log1pf(expf(z)); }

/* ---------- loading ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || n>(256L<<20)){ fprintf(stderr,"%s: config.json missing or larger than 256 MB\n",path); exit(1); }
    char *buf = malloc((size_t)n+1); if(!buf){ fprintf(stderr,"OOM reading %s\n",path); exit(1); }
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)req_num(r,"hidden_size");
    c->n_layers  = (int)req_num(r,"num_hidden_layers");
    c->vocab     = (int)req_num(r,"vocab_size");
    c->eps       = (float)req_num(r,"rms_norm_eps");
    jval *th = json_get(r,"rope_theta"); c->theta = th ? (float)th->num : 10000.f;
    free(buf); free(arena);
    /* Phase-1 defaults; overridden by qwen36_meta.json in load_meta.
     * Clamped so a missing meta file can never produce a divide-by-zero. */
    c->q_heads = (c->hidden >= 16) ? (c->hidden / 16) : 1;
    if (c->q_heads < 1) c->q_heads = 1;
    c->kv_heads = c->q_heads / 8; if (c->kv_heads < 1) c->kv_heads = 1;
    c->head_dim = c->hidden / c->q_heads; if (c->head_dim < 1) c->head_dim = 1;
    c->k_head_dim = c->head_dim; c->v_head_dim = c->head_dim;
    c->q_head_dim = c->head_dim * 2;        /* includes attn_output_gate */
    c->o_in = c->q_heads * c->head_dim;
    c->rotary_dim = (c->head_dim >= 4) ? (c->head_dim / 4) : 2;
    if (c->rotary_dim % 2 != 0) c->rotary_dim += 1;
    c->rope_dim = c->head_dim;
    c->partial_rotary_factor = 0.25f;
    c->n_experts = 256; c->topk = 8; c->inter = 512; c->shared_inter = 512;
    c->n_group = 1; c->topk_group = 1; c->norm_topk = 1; c->has_qk_norm = 1; c->has_bias = 0;
    c->attn_output_gate = 1; c->n_active = 0; c->zero_centered_norms = 1;
    c->vis_depth = 0; c->image_token = -1;
    c->mrope_section[0] = 11; c->mrope_section[1] = 11; c->mrope_section[2] = 10;
    if (c->n_layers <= 0 || c->n_layers > 512) { fprintf(stderr, "load_cfg: n_layers=%d out of range 1..512\n", c->n_layers); exit(1); }
    c->is_attn = calloc((size_t)c->n_layers, sizeof(uint8_t));
    for (int i = 0; i < c->n_layers; i++) c->is_attn[i] = (i % 4 == 3) ? 1 : 0;
}

/* Read qwen36_meta.json (emitted FLAT by convert_qwen36.py) to override the
 * Phase-1 defaults with the real model dimensions. The converter derives the
 * head dims from the actual weight shapes, so these are authoritative. Falls
 * back silently to the i%4==3 pattern and defaults if the file is absent. */

/* Every dimension the forward pass indexes with, checked once against the
 * buffers that actually exist. Both config.json and qwen36_meta.json ship
 * INSIDE the container, so a mismatched or hostile pair is a supply-chain
 * input, not a programmer error -- and the repo just spent six advisories
 * removing this bug class (unvalidated config -> heap OOB). Pattern follows
 * kimi_k3.c: one guarded expression per dimension, exit with a clear message.
 *
 * The fixed-size buffers below are the reason the ceilings are what they are;
 * raising one means raising the buffer with it:
 *   moe()      uint8_t keep[1024]        -> n_experts <= 1024
 *   moe()      int idx[256], val[256]    -> topk      <= 256
 *   deltanet() float kvl[512], dl[512]   -> dn_vdim   <= 512   (OpenMP region)
 */
#define CFG_NEED(cond, ...) do { if (!(cond)) {         fprintf(stderr, "[cfg] "); fprintf(stderr, __VA_ARGS__);         fprintf(stderr, " -- refusing\n"); exit(1); } } while (0)

static void validate_cfg(const Cfg *c, int n_layers_from_config) {
    CFG_NEED(c->n_layers > 0 && c->n_layers <= 512,
             "n_layers %d out of range 1..512", c->n_layers);
    /* A layer count that disagrees between the two files is a broken container:
     * is_attn was sized from config.json before meta could override n_layers. */
    CFG_NEED(c->n_layers == n_layers_from_config,
             "config.json says %d layers, qwen36_meta.json says %d",
             n_layers_from_config, c->n_layers);
    CFG_NEED(c->hidden > 0 && c->hidden <= 65536, "hidden %d out of range", c->hidden);
    CFG_NEED(c->vocab > 0, "vocab %d must be positive", c->vocab);
    if (c->n_experts == 0) {
        /* A dense checkpoint of the same family (Qwen3.5 / Qwen3.8 27B, #1757): the
         * layer's MLP is loaded as the shared expert, ungated, and nothing is routed. */
        CFG_NEED(c->topk == 0, "a dense model (num_experts 0) routes nothing, but topk is %d",
                 c->topk);
        CFG_NEED(c->shared_inter > 0, "dense MLP width %d must be positive", c->shared_inter);
    } else {
        CFG_NEED(c->n_experts > 0 && c->n_experts <= 1024,
                 "num_experts %d out of range 1..1024 (keep[] in moe())", c->n_experts);
        CFG_NEED(c->topk > 0 && c->topk <= 256,
                 "topk %d out of range 1..256 (idx[]/val[] in moe())", c->topk);
        CFG_NEED(c->topk <= c->n_experts, "topk %d exceeds num_experts %d",
                 c->topk, c->n_experts);
        /* shared_inter 0: no shared expert (Qwen3 MoE, qwen3_moe); Qwen3.5/3.6 carry one. */
        CFG_NEED(c->inter > 0 && c->shared_inter >= 0 && c->shared_inter <= 1 << 20,
                 "moe_inter %d must be positive and shared_inter %d in 0..1048576",
                 c->inter, c->shared_inter);
    }
    if (c->vis_depth) {
        CFG_NEED(c->vis_depth > 0 && c->vis_depth <= 64, "vision depth %d out of range", c->vis_depth);
        CFG_NEED(c->vis_hidden > 0 && c->vis_hidden <= 8192 && c->vis_heads > 0 &&
                 c->vis_hidden % c->vis_heads == 0, "vision hidden %d / heads %d", c->vis_hidden, c->vis_heads);
        CFG_NEED(c->vis_inter > 0 && c->vis_inter <= 65536, "vision inter %d", c->vis_inter);
        CFG_NEED(c->vis_patch > 0 && c->vis_patch <= 64 && c->vis_merge > 0 && c->vis_merge <= 8 &&
                 c->vis_temporal > 0 && c->vis_temporal <= 8 && c->vis_in_ch > 0 && c->vis_in_ch <= 8,
                 "vision patch %d merge %d temporal %d channels %d", c->vis_patch, c->vis_merge,
                 c->vis_temporal, c->vis_in_ch);
        CFG_NEED(c->vis_num_pos > 0 && c->vis_num_pos <= 65536, "vision positions %d", c->vis_num_pos);
        CFG_NEED(c->vis_out_hidden == c->hidden, "vision tower writes %d wide rows into a %d-wide model",
                 c->vis_out_hidden, c->hidden);
        CFG_NEED(c->image_token >= 0 && c->image_token < c->vocab, "image_token_id %d outside the vocabulary",
                 c->image_token);
        CFG_NEED(c->mrope_section[0] >= 0 && c->mrope_section[1] >= 0 && c->mrope_section[2] >= 0 &&
                 2 * (c->mrope_section[0] + c->mrope_section[1] + c->mrope_section[2]) == c->rotary_dim,
                 "mrope_section %d+%d+%d does not cover rotary_dim %d", c->mrope_section[0],
                 c->mrope_section[1], c->mrope_section[2], c->rotary_dim);
    }
    CFG_NEED(c->q_heads > 0 && c->kv_heads > 0 && c->head_dim > 0,
             "attention dims q_heads=%d kv_heads=%d head_dim=%d must be positive",
             c->q_heads, c->kv_heads, c->head_dim);
    CFG_NEED(c->q_heads % c->kv_heads == 0,
             "q_heads %d is not a multiple of kv_heads %d (GQA grouping)",
             c->q_heads, c->kv_heads);
    CFG_NEED(c->k_head_dim > 0 && c->v_head_dim > 0 && c->q_head_dim > 0,
             "per-head dims must be positive");
    /* DeltaNet: every one of these indexes a buffer or divides. */
    if (c->n_active < c->n_layers) {          /* at least one DeltaNet layer */
        CFG_NEED(c->dn_vheads > 0 && c->dn_kheads > 0,
                 "dn_vheads %d / dn_kheads %d must be positive (rep = vh / vk)",
                 c->dn_vheads, c->dn_kheads);
        CFG_NEED(c->dn_vheads % c->dn_kheads == 0,
                 "dn_vheads %d is not a multiple of dn_kheads %d",
                 c->dn_vheads, c->dn_kheads);
        CFG_NEED(c->dn_kdim > 0, "dn_kdim %d must be positive", c->dn_kdim);
        CFG_NEED(c->dn_vdim > 0 && c->dn_vdim <= 512,
                 "dn_vdim %d out of range 1..512 (kvl[]/dl[] in deltanet())",
                 c->dn_vdim);
        CFG_NEED(c->dn_convk >= 2, "dn_convk %d must be >= 2 (conv ring is convk-1)",
                 c->dn_convk);
        CFG_NEED(c->dn_conv_dim ==
                 2 * c->dn_kheads * c->dn_kdim + c->dn_vheads * c->dn_vdim,
                 "dn_conv_dim %d != 2*kheads*kdim + vheads*vdim (%d)",
                 c->dn_conv_dim,
                 2 * c->dn_kheads * c->dn_kdim + c->dn_vheads * c->dn_vdim);
    }
}

static void load_meta(Cfg *c, const char *snap) {
    /* is_attn was sized by load_cfg from config.json and is NOT resized here, so
     * every write below is bounded by this count, not by whatever the meta says. */
    const int is_attn_len = c->n_layers;
    char path[2048]; snprintf(path, sizeof(path), "%s/qwen36_meta.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) { printf("[meta] %s not found; using i%%4==3 + defaults\n", path); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc((size_t)n+1); if(!buf){fclose(f);return;}
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ free(buf); fclose(f); return; } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    if (r && r->t == J_OBJ) {
        jval *v;
        #define G(name,field) if((v=json_get(r,name))&&v->t==J_NUM) c->field=(int)v->num
        G("hidden", hidden); G("n_layers", n_layers); G("n_active", n_active);
        G("q_heads", q_heads); G("kv_heads", kv_heads); G("head_dim", head_dim);
        G("q_head_dim", q_head_dim); G("k_head_dim", k_head_dim); G("v_head_dim", v_head_dim);
        G("o_in", o_in); G("rope_dim", rope_dim); G("qk_rope_head_dim", rope_dim);
        G("expert_gs", expert_gs);
        G("expert_down_bits", expert_down_bits); G("expert_down_gs", expert_down_gs);
        G("num_experts", n_experts); G("topk", topk);
        G("moe_inter", inter); G("shared_inter", shared_inter);
        G("n_group", n_group); G("topk_group", topk_group);
        G("dn_vheads", dn_vheads); G("dn_kheads", dn_kheads); G("dn_kdim", dn_kdim);
        G("dn_vdim", dn_vdim); G("dn_convk", dn_convk); G("dn_conv_dim", dn_conv_dim);
        #undef G
        /* validate_cfg makes exactly this check, and used to be the only one --
         * but it runs AFTER load_meta returns, one line too late to stop the
         * is_attn writes below. A container whose config.json said 4 layers and
         * whose meta said 8 therefore wrote past a 4-byte allocation before
         * anything refused it: ASan heap-buffer-overflow at the layer_types
         * loop, reached from an ordinary model directory. Refuse here, while
         * is_attn is still the only thing that has been sized. */
        CFG_NEED(c->n_layers == is_attn_len,
                 "config.json says %d layers, qwen36_meta.json says %d",
                 is_attn_len, c->n_layers);
        if((v=json_get(r,"partial_rotary_factor"))&&v->t==J_NUM) c->partial_rotary_factor=(float)v->num;
        if((v=json_get(r,"rope_theta"))&&v->t==J_NUM) c->theta=(float)v->num;
        if((v=json_get(r,"rms_eps"))&&v->t==J_NUM) c->eps=(float)v->num;
        if((v=json_get(r,"attn_output_gate"))&&v->t==J_BOOL) c->attn_output_gate=v->boolean;
        if((v=json_get(r,"norm_topk_prob"))&&v->t==J_BOOL) c->norm_topk=v->boolean;
        if((v=json_get(r,"zero_centered_norms"))&&v->t==J_BOOL) c->zero_centered_norms=v->boolean;
        if((v=json_get(r,"has_bias"))&&v->t==J_BOOL) c->has_bias=v->boolean;
        if((v=json_get(r,"has_qk_norm"))&&v->t==J_BOOL) c->has_qk_norm=v->boolean;
        if((v=json_get(r,"image_token_id"))&&v->t==J_NUM) c->image_token=(int)v->num;
        if((v=json_get(r,"mrope_section"))&&v->t==J_ARR&&v->len==3)
            for(int k=0;k<3;k++) if(v->kids[k]->t==J_NUM) c->mrope_section[k]=(int)v->kids[k]->num;
        {   /* the tower, written by the converter only when it copied the weights */
            jval *vv = json_get(r,"vision");
            if (vv && vv->t==J_OBJ) {
                jval *w;
                #define GV(name,field) if((w=json_get(vv,name))&&w->t==J_NUM) c->field=(int)w->num
                GV("depth",vis_depth); GV("hidden",vis_hidden); GV("heads",vis_heads);
                GV("inter",vis_inter); GV("patch",vis_patch); GV("merge",vis_merge);
                GV("temporal",vis_temporal); GV("in_ch",vis_in_ch);
                GV("out_hidden",vis_out_hidden); GV("num_pos",vis_num_pos);
                #undef GV
            }
        }
        /* derive rotary_dim from head_dim * partial_rotary_factor (HF formula) */
        if (c->partial_rotary_factor > 0.f)
            c->rotary_dim = (int)(c->head_dim * c->partial_rotary_factor + 0.5f);
        else
            c->rotary_dim = c->head_dim;
        if (c->rotary_dim < 2) c->rotary_dim = 2;
        if (c->rotary_dim % 2 != 0) c->rotary_dim += 1;
        if (c->rotary_dim > c->head_dim) c->rotary_dim = c->head_dim;
        /* rebuild is_attn from explicit layer_types if present */
        jval *lt = json_get(r,"layer_types");
        if (lt && lt->t==J_ARR) {
            for (int i=0;i<c->n_layers;i++) c->is_attn[i]=0;
            c->n_active=0;
            for (int i=0;i<lt->len && i<c->n_layers;i++){
                const char *s = (lt->kids[i]->t==J_STR)? lt->kids[i]->str : "";
                if (s && strcmp(s,"full_attention")==0) { c->is_attn[i]=1; c->n_active++; }
            }
        }
    }
    free(buf); free(arena);
    fprintf(stderr, "[meta] loaded: q_heads=%d kv_heads=%d head_dim=%d q_head_dim=%d k_head_dim=%d v_head_dim=%d "
           "o_in=%d rotary_dim=%d n_experts=%d topk=%d inter=%d shared_inter=%d n_group=%d topk_group=%d "
           "attn_output_gate=%d n_active=%d\n",
           c->q_heads, c->kv_heads, c->head_dim, c->q_head_dim, c->k_head_dim, c->v_head_dim,
           c->o_in, c->rotary_dim, c->n_experts, c->topk, c->inter, c->shared_inter,
           c->n_group, c->topk_group, c->attn_output_gate, c->n_active);
    if (c->dn_vheads > 0)
        fprintf(stderr, "[meta] DeltaNet: vheads=%d kheads=%d kdim=%d vdim=%d convk=%d conv_dim=%d\n",
               c->dn_vheads, c->dn_kheads, c->dn_kdim, c->dn_vdim, c->dn_convk, c->dn_conv_dim);
}

/* Multimodal Qwen3.5/3.6 checkpoints (a Swiftlet qpack container's
 * model.safetensors included) store the text stack under a `language_model.`
 * prefix; the engine and the converter both speak unprefixed names.  Resolve
 * the plain name first so converted snapshots are untouched, then the
 * prefixed one; when neither exists return the CANONICAL name so the caller's
 * refusal names the tensor the engine actually wanted. */
#define QW_DENSE_NAME_MAX 288   /* 256-byte call-site buffers + the prefix */
static const char *dense_resolve(Model *m, const char *name,
                                 char *buf, size_t cap) {
    if (st_has(&m->S, name)) return name;
    snprintf(buf, cap, "language_model.%s", name);
    if (st_has(&m->S, buf)) return buf;
    return name;
}
static int dense_has(Model *m, const char *name) {
    char rn[QW_DENSE_NAME_MAX];
    return st_has(&m->S, dense_resolve(m, name, rn, sizeof rn));
}

/* Dense tensor stored as an MLX affine triple (packed U32 `.weight` + sibling
 * `.scales`/`.biases`, the layout Swiftlet writes into a qpack container's
 * model.safetensors): expand to f32 rows through the checked affine contract.
 * Every dimension is anchored to `want`, the CONFIG-implied element count the
 * forward pass will index with -- the same discipline as load_t_n below, so a
 * container whose packed geometry disagrees with config.json is a refusal,
 * never a plausible heap OOB.  bits (Q4/Q8) and the group size are DERIVED
 * from the shapes (logical input over packed words, logical input over scale
 * groups) rather than parsed from config quantization overrides: the file's
 * own byte layout is what the expansion must agree with, and the affine
 * validator re-checks the derived geometry against every buffer length. */
static float *load_t_affine(Model *m, const char *wname, int64_t want) {
    st_tensor *w = st_find(&m->S, wname);
    if (!w) { fprintf(stderr, "missing %s\n", wname); exit(1); }
    size_t wlen = strlen(wname);
    char sname[QW_DENSE_NAME_MAX], bname[QW_DENSE_NAME_MAX];
    if (wlen < 7 || strcmp(wname + wlen - 7, ".weight") != 0 ||
        wlen - 7 + sizeof(".scales") > sizeof(sname)) {
        fprintf(stderr, "%s: U32 tensor is not a `.weight` with room for "
                "`.scales`/`.biases` siblings -- refusing\n", wname); exit(1);
    }
    snprintf(sname, sizeof(sname), "%.*s.scales", (int)(wlen - 7), wname);
    snprintf(bname, sizeof(bname), "%.*s.biases", (int)(wlen - 7), wname);
    st_tensor *s = st_find(&m->S, sname), *b = st_find(&m->S, bname);
    if (!s || !b) {
        fprintf(stderr, "%s: affine `.scales`/`.biases` siblings are missing "
                "-- refusing\n", wname); exit(1);
    }
    ColiAffineScalarFormat sf;
    switch (s->dtype) {
        case 0: sf = COLI_AFFINE_SCALAR_BF16; break;
        case 1: sf = COLI_AFFINE_SCALAR_F16; break;
        case 2: sf = COLI_AFFINE_SCALAR_F32; break;
        default:
            fprintf(stderr, "%s: scales dtype %s is not BF16/F16/F32 -- "
                    "refusing\n", sname, st_dtype_name(s->dtype)); exit(1);
    }
    if (b->dtype != s->dtype || b->numel != s->numel) {
        fprintf(stderr, "%s: biases dtype/numel disagree with scales -- "
                "refusing\n", bname); exit(1);
    }
    int64_t packed = w->rank >= 2 ? w->shape[w->rank - 1] : 0;
    int64_t rows = packed > 0 ? w->numel / packed : 0;
    if (want <= 0 || rows <= 0 || w->numel != rows * packed ||
        want % rows != 0) {
        fprintf(stderr, "%s: packed shape does not divide the config-implied "
                "%lld elements -- refusing\n", wname, (long long)want); exit(1);
    }
    int64_t in = want / rows;
    int64_t per_word = packed > 0 && in % packed == 0 ? in / packed : 0;
    if (per_word != 8 && per_word != 4) {
        fprintf(stderr, "%s: %lld packed words for %lld logical columns is "
                "neither Q4 nor Q8 -- refusing\n",
                wname, (long long)packed, (long long)in); exit(1);
    }
    int64_t groups = s->numel % rows == 0 ? s->numel / rows : 0;
    int64_t gs = groups > 0 && in % groups == 0 ? in / groups : 0;
    if (gs <= 0) {
        fprintf(stderr, "%s: %lld scale groups do not tile %lld logical "
                "columns -- refusing\n",
                sname, (long long)groups, (long long)in); exit(1);
    }
    void *wraw = malloc((size_t)w->nbytes);
    void *sraw = malloc((size_t)s->nbytes);
    void *braw = malloc((size_t)b->nbytes);
    if (!wraw || !sraw || !braw) { fprintf(stderr, "OOM reading %s\n", wname); exit(1); }
    st_read_raw(&m->S, wname, wraw, 1);
    st_read_raw(&m->S, sname, sraw, 1);
    st_read_raw(&m->S, bname, braw, 1);
    ColiAffineQuantizedView view = {
        wraw, sraw, braw,
        (size_t)w->nbytes, (size_t)s->nbytes, (size_t)b->nbytes,
        (size_t)rows, (size_t)in, (size_t)gs,
        per_word == 8 ? COLI_AFFINE_MLX_Q4 : COLI_AFFINE_MLX_Q8, sf
    };
    float *p = falloc(want);
    ColiAffineStatus status = coli_affine_dequant_ref(&view, p);
    if (status != COLI_AFFINE_OK) {
        fprintf(stderr, "%s: affine expansion refused (%s)\n",
                wname, coli_affine_status_string(status)); exit(1);
    }
    free(wraw); free(sraw); free(braw);
    return p;
}

/* `want` is the element count the forward pass will index with. The container
 * is a file, not an invariant: this used to allocate whatever st_numel reported
 * while every read afterwards used CONFIG dims, so a short tensor was a plain
 * heap OOB read (embed is indexed as m->embed + ids[s]*D). The expert path
 * already refuses a wrong size; this is the same discipline for the dense set. */
static float *load_t_n(Model *m, const char *name, int64_t want) {
    char rn[QW_DENSE_NAME_MAX];
    const char *nm = dense_resolve(m, name, rn, sizeof rn);
    st_tensor *t = st_find(&m->S, nm);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (t->dtype == 7) return load_t_affine(m, nm, want);
    int64_t n = t->numel;
    if (want > 0 && n != want) {
        fprintf(stderr, "%s: %lld elements, config implies %lld -- refusing\n",
                nm, (long long)n, (long long)want); exit(1);
    }
    float *p = falloc(n);
    st_read_f32(&m->S, nm, p, 0);
    return p;
}

/* Dense matrix load, quantized to int8 (+ int4 planar per `tag`, see
 * dense_int4_wanted) DURING loading rather than in a separate pass over the
 * whole model afterward (see QW, above Layer): reads `name` (I*O elements,
 * same size discipline as load_t_n), and when `quantize` && COLI_DENSE_I8 is
 * on, quantizes it via qw_quantize and frees the f32 staging buffer right
 * away -- so at most one dense matrix's f32 copy is ever resident at a time,
 * not the whole model's. `quantize` is false for loaders that never ran
 * through the old post-hoc qdw_register pass either (the Segment/Edge
 * adapters build partial or auxiliary models straight off
 * model_init_range/load_t_n, never main()'s dense-i8 block) -- passing it
 * through keeps their f32-only behavior exactly as it was; `tag` is unused
 * on that path. */
static void load_tq(Model *m, const char *name, int I, int O, int quantize, const char *tag, QW *out) {
    float *p = load_t_n(m, name, (int64_t)I * O);
    out->w = p; out->q = NULL; out->sc = NULL; out->I = I; out->O = O;
    out->q4 = NULL; out->sg = NULL; out->ng = 0;
    out->vk = NULL; out->vk_off = 0; out->h = NULL;
    if (!quantize || !dense_i8_on()) return;
    if (dense_bits() == 16) {
        uint16_t *h = q36_walloc((size_t)I * O * sizeof(uint16_t));
        if (!h) { fprintf(stderr, "OOM keeping %s in f16\n", name); exit(1); }
        #pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < (int64_t)I * O; k++) h[k] = f32_to_f16_bits(p[k]);
        out->h = h;
        free(p); out->w = NULL;
        return;
    }
    qw_quantize(p, I, O, tag, out);
    if (getenv("COLI_KEEP_F32")) out->w = p; else { free(p); out->w = NULL; }
}
/* load_tq, and under COLI_VULKAN a note of what it read (name, tag, whether it was
 * quantized) for q36_dho_reload. Every resident dense matrix of the model loads here. */
static void load_tq_noted(Model *m, const char *name, int I, int O, int quantize, const char *tag, QW *out) {
    load_tq(m, name, I, O, quantize, tag, out);
#ifdef COLI_VULKAN
    out->vk_name = strdup(name); out->vk_tag = tag; out->vk_quant = quantize;
    out->vk_fmt = 0; out->vk_gone = 0;
    if (!out->vk_name) { fprintf(stderr, "OOM dense matrix name\n"); exit(1); }
#endif
}
#ifdef COLI_VULKAN
/* ---- dense weights on the device only (COLI_VK_DENSE_HOST) ------------------------
 * With the dense part on the device (the chain, or COLI_VK_DENSE), every resident
 * dense matrix goes up at start and its host copy (int8 rows, the int4 copy, f32 or f16) is
 * given back. The CPU then never multiplies by it, except after a lost device:
 * matmul_d first reads it back here, through load_tq with what it was loaded with
 * (the same quantization, so the same bytes), and keeps it from there on. What keeps
 * its host copy: the embedding (its rows are gathered on the CPU), the DeltaNet's
 * dn_a/dn_b and the shared expert's gate vector (the CPU's DeltaNet and shared expert
 * read them, and the chain uploads its own copies), norms, the vision tower, a matrix
 * the device refused, or rows it already imported in place. */
static Model *g_q36_dho_model;
static pthread_mutex_t g_q36_dho_mx = PTHREAD_MUTEX_INITIALIZER;
static size_t q36_dho_host_bytes(const QW *w) {
    size_t b = 0;
    if (w->q4) b += (size_t)w->O * (w->I / 2) + (size_t)w->O * (w->I / 64) * sizeof(float);
    if (w->q) b += (size_t)w->O * w->I + (size_t)w->O * sizeof(float);
    if (w->w) b += (size_t)w->O * w->I * sizeof(float);
    if (w->h) b += (size_t)w->O * w->I * sizeof(uint16_t);
    return b;
}
static void q36_dho_reload(QW *w) {
    pthread_mutex_lock(&g_q36_dho_mx);
    if (w->vk_gone) {
        Model *m = g_q36_dho_model;
        if (!m || !w->vk_name) { fprintf(stderr, "[VK] qwen36: a dense matrix the device held alone cannot be read back\n"); exit(1); }
        QW t; memset(&t, 0, sizeof t);
        load_tq(m, w->vk_name, w->I, w->O, w->vk_quant, w->vk_tag, &t);
        w->w = t.w; w->q = t.q; w->sc = t.sc; w->q4 = t.q4; w->sg = t.sg; w->ng = t.ng; w->h = t.h;
        w->vk_gone = 0;
        coli_vk_dense_host_reloaded(q36_dho_host_bytes(w));
    }
    pthread_mutex_unlock(&g_q36_dho_mx);
}
static void q36_dho_drop(QW *w, size_t *bytes, int *n) {
    if (!w->vk_name || w->vk_gone || w->vk_off || w->vk_imported || !(w->q || w->q4 || w->w || w->h)) return;
    const void *wq; const float *sc;
    int fmt = vk_qw_fmt(w, &wq, &sc);
    if (!vk_qw_tensor(w) || w->vk_imported) return;   /* refused or still read in place: keep its host pages */
    size_t b = q36_dho_host_bytes(w);
    free((void *)w->w); q36_wfree(w->q); free(w->sc); free(w->q4); free(w->sg); q36_wfree(w->h);
    w->h = NULL; w->w = NULL; w->q = NULL; w->sc = NULL; w->q4 = NULL; w->sg = NULL;
    w->vk_fmt = fmt; w->vk_gone = 1;
    coli_vk_dense_host_dropped(b);
    *bytes += b; (*n)++;
}
static void q36_dho_count(QW *w, size_t *bytes, int *n) {
    if (w->vk_name && !w->vk_off && !w->vk_imported && (w->q || w->q4 || w->w || w->h)) { *bytes += q36_dho_host_bytes(w); (*n)++; }
}
/* Every resident dense matrix: the head, then each layer's. */
static void q36_dho_each(Model *m, void (*f)(QW *, size_t *, int *), size_t *bytes, int *n) {
    f(&m->lm_head, bytes, n);
    for (int i = 0; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i];
        QW *ws[] = {&l->q, &l->k, &l->v, &l->o, &l->gate, &l->sh_g, &l->sh_u, &l->sh_d,
                    &l->dn_qkv, &l->dn_z, &l->dn_out};
        for (size_t k = 0; k < sizeof ws / sizeof ws[0]; k++) f(ws[k], bytes, n);
    }
}
#endif

/* ---------- vision (#1757) ----------
 * The weights are the checkpoint's own model.visual.*, copied by the converter,
 * read as f32 like every other small tensor here. */
static void q36_vis_linear(Model *m, Q38Linear *l, const char *stem, int out, int in) {
    char nm[256];
    snprintf(nm, sizeof nm, "model.visual.%s.weight", stem);
    l->w = load_t_n(m, nm, (int64_t)out * in);
    snprintf(nm, sizeof nm, "model.visual.%s.bias", stem);
    l->b = st_has(&m->S, nm) ? load_t_n(m, nm, out) : NULL;
    l->out = out; l->in = in;
}
static void q36_vis_norm(Model *m, Q38Norm *n, const char *stem, int width) {
    char nm[256];
    snprintf(nm, sizeof nm, "model.visual.%s.weight", stem); n->w = load_t_n(m, nm, width);
    snprintf(nm, sizeof nm, "model.visual.%s.bias", stem);   n->b = load_t_n(m, nm, width);
}
static void q36_load_vision(Model *m) {
    Cfg *c = &m->c;
    if (!c->vis_depth) return;
    if (!st_has(&m->S, "model.visual.pos_embed.weight")) {
        fprintf(stderr, "[qwen36] qwen36_meta.json describes a vision tower but the container has none; text only\n");
        c->vis_depth = 0; return;
    }
    Q38Vision *v = &m->vis;
    memset(v, 0, sizeof *v);
    v->depth = c->vis_depth; v->hidden = c->vis_hidden; v->heads = c->vis_heads;
    v->head_dim = c->vis_hidden / c->vis_heads; v->inter = c->vis_inter;
    v->patch = c->vis_patch; v->merge = c->vis_merge; v->temporal = c->vis_temporal;
    v->in_ch = c->vis_in_ch; v->out_hidden = c->vis_out_hidden;
    v->num_pos = c->vis_num_pos; v->side = (int)(sqrt((double)c->vis_num_pos) + 0.5);
    v->eps = 1e-6f;
    if (v->side * v->side != v->num_pos) {
        fprintf(stderr, "[qwen36] vision num_position_embeddings %d is not a square grid -- refusing\n", v->num_pos);
        exit(1);
    }
    q36_vis_linear(m, &v->patch_embed, "patch_embed.proj", v->hidden, v->in_ch * v->temporal * v->patch * v->patch);
    v->pos_embed = load_t_n(m, "model.visual.pos_embed.weight", (int64_t)v->num_pos * v->hidden);
    v->blocks = calloc((size_t)v->depth, sizeof(Q38VBlock));
    if (!v->blocks) { fprintf(stderr, "OOM vision blocks\n"); exit(1); }
    for (int i = 0; i < v->depth; i++) {
        char stem[96];
        snprintf(stem, sizeof stem, "blocks.%d.norm1", i);          q36_vis_norm(m, &v->blocks[i].norm1, stem, v->hidden);
        snprintf(stem, sizeof stem, "blocks.%d.norm2", i);          q36_vis_norm(m, &v->blocks[i].norm2, stem, v->hidden);
        snprintf(stem, sizeof stem, "blocks.%d.attn.qkv", i);       q36_vis_linear(m, &v->blocks[i].qkv, stem, 3 * v->hidden, v->hidden);
        snprintf(stem, sizeof stem, "blocks.%d.attn.proj", i);      q36_vis_linear(m, &v->blocks[i].proj, stem, v->hidden, v->hidden);
        snprintf(stem, sizeof stem, "blocks.%d.mlp.linear_fc1", i); q36_vis_linear(m, &v->blocks[i].fc1, stem, v->inter, v->hidden);
        snprintf(stem, sizeof stem, "blocks.%d.mlp.linear_fc2", i); q36_vis_linear(m, &v->blocks[i].fc2, stem, v->hidden, v->inter);
    }
    int wide = v->hidden * v->merge * v->merge;
    q36_vis_norm(m, &v->merger_norm, "merger.norm", v->hidden);
    q36_vis_linear(m, &v->merger_fc1, "merger.linear_fc1", wide, wide);
    q36_vis_linear(m, &v->merger_fc2, "merger.linear_fc2", v->out_hidden, wide);
    m->vis_ready = 1;
    fprintf(stderr, "[qwen36] vision tower: %d blocks, hidden %d, %d heads, patch %d, merge %d\n",
            v->depth, v->hidden, v->heads, v->patch, v->merge);
}

static void q36_vision_detach(Model *m) {
    free(m->vis_rows); free(m->vis_map); free(m->mpos);
    m->vis_rows = NULL; m->vis_map = NULL; m->mpos = NULL;
    m->vis_rows_n = m->vis_map_len = m->mpos_len = m->rope_delta = 0;
}

/* Run the tower on one image and lay its rows and rope positions over the prompt
 * `ids` (absolute positions 0..n-1). One image per prompt: its placeholders must
 * be one contiguous run of exactly as many tokens as the merged grid gives.
 * Positions follow HF Qwen3_5Model.get_rope_index: text counts up on all three
 * axes; the image's token (row, col) sits at (start, start+row, start+col); the
 * text after it resumes at start + max(rows, cols); past the prompt every axis
 * is pos + rope_delta, rope_delta = max position + 1 - n. */
static int q36_vision_attach(Model *m, const float *patches, int grid_h, int grid_w,
                             const int *ids, int n) {
    Cfg *c = &m->c;
    if (!m->vis_ready || n <= 0) return -1;
    if (grid_h <= 0 || grid_w <= 0 || grid_h % c->vis_merge || grid_w % c->vis_merge) return -1;
    int lh = grid_h / c->vis_merge, lw = grid_w / c->vis_merge, tokens = lh * lw;
    int first = -1, slots = 0;
    for (int i = 0; i < n; i++) if (ids[i] == c->image_token) { if (first < 0) first = i; slots++; }
    if (slots != tokens || first < 0) {
        fprintf(stderr, "[qwen36] prompt has %d image placeholders but the grid gives %d tokens\n", slots, tokens);
        return -1;
    }
    for (int i = first; i < first + tokens; i++)
        if (ids[i] != c->image_token) {
            fprintf(stderr, "[qwen36] the image placeholders are not one contiguous run (one image per prompt)\n");
            return -1;
        }
    q36_vision_detach(m);
    m->vis_rows = calloc((size_t)tokens * c->hidden, sizeof(float));
    m->vis_map = malloc((size_t)n * sizeof(int));
    m->mpos = malloc((size_t)n * 3 * sizeof(int));
    if (!m->vis_rows || !m->vis_map || !m->mpos) { q36_vision_detach(m); return -1; }
    if (q38_vision_forward(&m->vis, patches, grid_h, grid_w, m->vis_rows) != tokens) { q36_vision_detach(m); return -1; }
    int cur = 0, maxpos = -1;
    for (int i = 0; i < n; i++) {
        int *pp = m->mpos + (size_t)i * 3;
        if (i >= first && i < first + tokens) {
            int k = i - first, start = cur;
            m->vis_map[i] = k;
            pp[0] = start; pp[1] = start + k / lw; pp[2] = start + k % lw;
            if (k == tokens - 1) cur = start + (lh > lw ? lh : lw);
        } else {
            m->vis_map[i] = -1;
            pp[0] = pp[1] = pp[2] = cur++;
        }
        for (int a = 0; a < 3; a++) if (pp[a] > maxpos) maxpos = pp[a];
    }
    m->vis_rows_n = tokens; m->vis_map_len = n; m->mpos_len = n;
    m->rope_delta = maxpos + 1 - n;
    return tokens;
}

/* Loader for the RMSNorm weights rmsnorm_row applies as (1 + w).  HF Qwen3.6
 * stores them zero-centered and the forward pass is written for that; an
 * MLX-derived container stores FULL gamma (mlx-lm materialises the +1 at
 * conversion -- measured on the production qpack container: input_layernorm
 * mean 1.03, q_norm mean 1.33, where the zero-centered forms centre on 0).
 * Feeding full gamma through (1 + w) doubles every normalised activation and
 * the model degenerates to noise, so the shift is undone HERE, at load, and
 * the forward pass keeps exactly one convention.  The DeltaNet gated norm is
 * full gamma in both dialects (its forward multiplies plain w) and must NOT
 * come through this loader. */
static float *load_norm_n(Model *m, const char *name, int64_t want) {
    float *w = load_t_n(m, name, want);
    if (!m->c.zero_centered_norms)
        for (int64_t i = 0; i < want; i++) w[i] -= 1.0f;
    return w;
}

static void model_init_range(Model *m, const char *snap, int cap, int bits,
                             int layer_begin, int layer_end,
                             int load_boundaries, int allocate_state) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    int n_layers_from_config = m->c.n_layers;
    load_meta(&m->c, snap);
    validate_cfg(&m->c, n_layers_from_config);
    /* load_cfg and validate_cfg both guard n_layers, but the compiler can't see
     * across function boundaries, so re-assert here to silence -Walloc-size-larger-than. */
    if (m->c.n_layers <= 0) { fprintf(stderr, "model_init: n_layers=%d invalid\n", m->c.n_layers); exit(1); }
    if (m->c.rotary_dim > m->c.head_dim || m->c.rotary_dim % 2 != 0) {
        fprintf(stderr, "rotary_dim %d invalid for head_dim %d\n", m->c.rotary_dim, m->c.head_dim); exit(1);
    }
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    if (layer_end == 0) layer_end = c->n_layers;
    if (layer_begin < 0 || layer_end > c->n_layers ||
        layer_begin >= layer_end) {
        fprintf(stderr, "invalid Qwen3.6 layer range [%d,%d) for %d layers\n",
                layer_begin, layer_end, c->n_layers);
        exit(1);
    }
    double t0 = now_s();
    /* Quantize during load only for the full-model path (main()'s static Model,
     * load_boundaries=1): the Segment/Edge adapters build partial or auxiliary
     * models straight off this same loop and never ran the old post-hoc
     * qdw_register pass either, so gating on load_boundaries keeps their
     * f32-only numerics exactly as they were. */
    int quantize_dense = load_boundaries && dense_i8_on();
    int qcount = 0; double qfreed = 0;
    if (load_boundaries) {
        m->embed = load_t_n(m, "model.embed_tokens.weight", (int64_t)c->vocab * c->hidden);
#ifndef COLI_VULKAN
        if (quantize_dense && dense_bits() == 16) {   /* 16-bit trunk: the table too, 2.5 GB less on the 27B */
            int64_t n = (int64_t)c->vocab * c->hidden;
            m->embed_h = malloc((size_t)n * sizeof(uint16_t));
            if (!m->embed_h) { fprintf(stderr, "OOM keeping the embeddings in f16\n"); exit(1); }
            #pragma omp parallel for schedule(static)
            for (int64_t k = 0; k < n; k++) m->embed_h[k] = f32_to_f16_bits(m->embed[k]);
            free(m->embed); m->embed = NULL;
        }
#endif
        load_tq_noted(m, "lm_head.weight", c->hidden, c->vocab, quantize_dense, "lmhead", &m->lm_head);
        if (m->lm_head.q || m->lm_head.q4 || m->lm_head.h) { qcount++; qfreed += (double)c->hidden * c->vocab * sizeof(float); }
        m->final_norm = load_norm_n(m, "model.norm.weight", c->hidden);
        q36_load_vision(m);
    }
    m->L = calloc((size_t)c->n_layers, sizeof(Layer));
    /* Phase 2: the converter stores EVERY layer (Gated-Attention + Gated DeltaNet)
     * under its OWN original index model.layers.{i}. So active_of is the identity
     * map; experts and dense weights are read from model.layers.{i} for all i. */
    m->active_of = malloc((size_t)c->n_layers * sizeof(int));
    for (int i = 0; i < c->n_layers; i++) m->active_of[i] = i;
    char nm[256];
    int q_out = c->q_heads * c->q_head_dim, kv_out = c->kv_heads * c->k_head_dim;
    #define QCOUNT(field) do { if ((field).q || (field).q4 || (field).h) { qcount++; qfreed += (double)(field).I * (field).O * sizeof(float); } } while (0)
    for (int i = layer_begin; i < layer_end; i++) {
        int ai = m->active_of[i];        /* == i for Phase 2 */
        Layer *l = &m->L[i];
        /* input/post layernorms exist for every layer; they go through
         * load_norm_n (rmsnorm_row weights), the router does not. */
        #define LD(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,ai); l->field = load_norm_n(m,nm,(want))
        LD(in_ln,  "input_layernorm.weight", c->hidden);
        LD(post_ln,"post_attention_layernorm.weight", c->hidden);
        #undef LD
        if (c->n_experts) {
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.weight", ai);
            load_tq_noted(m, nm, c->hidden, c->n_experts, quantize_dense, "router", &l->gate);
            QCOUNT(l->gate);
        } else l->gate = (QW){0};
        /* q/k norms are per-head [head_dim]; only on attention layers, load if present */
        if (c->has_qk_norm) {
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_norm.weight", ai);
            l->qn = dense_has(m, nm) ? load_norm_n(m, nm, c->head_dim) : NULL;
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_norm.weight", ai);
            l->kn = dense_has(m, nm) ? load_norm_n(m, nm, c->head_dim) : NULL;
        } else { l->qn = NULL; l->kn = NULL; }
        /* router correction bias (optional) */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.e_score_correction_bias", ai);
        if (c->n_experts && dense_has(m, nm)) { l->gate_bias = load_t_n(m, nm, c->n_experts); }
        else l->gate_bias = NULL;
        /* shared expert (dense, int8-during-load). A dense model's MLP is the same
         * SwiGLU under mlp.{gate,up,down}_proj, with no gate in front of it. */
        const char *shp = c->n_experts ? "mlp.shared_expert." : "mlp.";
        if (c->shared_inter > 0) {   /* Qwen3 MoE has none: its sh_* stay empty */
        snprintf(nm,sizeof(nm),"model.layers.%d.%sgate_proj.weight", ai, shp);
        load_tq_noted(m, nm, c->hidden, c->shared_inter, quantize_dense, "shexp", &l->sh_g); QCOUNT(l->sh_g);
        snprintf(nm,sizeof(nm),"model.layers.%d.%sup_proj.weight", ai, shp);
        load_tq_noted(m, nm, c->hidden, c->shared_inter, quantize_dense, "shexp", &l->sh_u); QCOUNT(l->sh_u);
        snprintf(nm,sizeof(nm),"model.layers.%d.%sdown_proj.weight", ai, shp);
        load_tq_noted(m, nm, c->shared_inter, c->hidden, quantize_dense, "shexp", &l->sh_d); QCOUNT(l->sh_d);
        }
        /* shared_expert_gate: Linear(hidden -> 1), sigmoid-gated shared expert */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert_gate.weight", ai);
        l->sh_gate = c->n_experts && c->shared_inter > 0 && dense_has(m, nm) ? load_t_n(m, nm, c->hidden) : NULL;
        if (c->is_attn[i]) {
            /* Gated Attention (full_attention) layer, dense projections int8-during-load */
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_proj.weight", ai);
            load_tq_noted(m, nm, c->hidden, q_out, quantize_dense, "attn", &l->q); QCOUNT(l->q);
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_proj.weight", ai);
            load_tq_noted(m, nm, c->hidden, kv_out, quantize_dense, "attn", &l->k); QCOUNT(l->k);
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.v_proj.weight", ai);
            load_tq_noted(m, nm, c->hidden, kv_out, quantize_dense, "attn", &l->v); QCOUNT(l->v);
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.o_proj.weight", ai);
            load_tq_noted(m, nm, c->o_in, c->hidden, quantize_dense, "attn", &l->o); QCOUNT(l->o);
            l->dn_qkv=l->dn_z=(QW){0}; l->dn_b=l->dn_a=l->dn_conv=NULL;
            l->dn_dtbias=l->dn_alog=l->dn_norm=NULL; l->dn_out=(QW){0};
        } else {
            /* Gated DeltaNet (linear_attention) layer */
            l->q=l->k=l->v=l->o=(QW){0};
            #define LD4(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn." suffix,ai); l->field = load_t_n(m,nm,(want))
            int64_t vdim_tot = (int64_t)c->dn_vheads * c->dn_vdim;
            snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn.in_proj_qkv.weight", ai);
            load_tq_noted(m, nm, c->hidden, c->dn_conv_dim, quantize_dense, "dnproj", &l->dn_qkv); QCOUNT(l->dn_qkv);
            snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn.in_proj_z.weight", ai);
            load_tq_noted(m, nm, c->hidden, (int)vdim_tot, quantize_dense, "dnproj", &l->dn_z); QCOUNT(l->dn_z);
            LD4(dn_b,   "in_proj_b.weight",   (int64_t)c->dn_vheads * c->hidden);
            LD4(dn_a,   "in_proj_a.weight",   (int64_t)c->dn_vheads * c->hidden);
            LD4(dn_conv,"conv1d.weight",      (int64_t)c->dn_conv_dim * c->dn_convk);
            LD4(dn_dtbias, "dt_bias",         c->dn_vheads);
            LD4(dn_alog,"A_log",              c->dn_vheads);
            LD4(dn_norm, "norm.weight",       c->dn_vdim);
            #undef LD4
            snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn.out_proj.weight", ai);
            load_tq_noted(m, nm, (int)vdim_tot, c->hidden, quantize_dense, "dnout", &l->dn_out); QCOUNT(l->dn_out);
        }
    }
    #undef QCOUNT
    if (quantize_dense)
        fprintf(stderr, "[dense-i8] %d matrices %s during load, %.1f GB f32 freed\n", qcount,
                dense_bits() == 16 ? "kept in f16" : "quantized", qfreed/1073741824.0);
    if (cap <= 0 && c->n_experts <= 0) {
        /* Dense checkpoint (#1757): nothing is ever routed, so the per-layer
         * expert cache is never touched by moe()/expert_get(). Sizing it
         * from RAM would be meaningless -- n_experts==0 has nothing to clamp
         * qwen36_cap_for_ram's derived value against, so an unclamped RAM
         * budget could otherwise calloc an absurd slot count for a cache
         * that will sit empty. cap=1 is a harmless placeholder. */
        cap = 1;
    } else if (cap <= 0) {
        /* cap<=0 sentinel: derive from host RAM, same "0 = auto" convention as
         * colibri.c/olmoe.c. rss_gb() here reflects all dense weights resident
         * (this loop just finished) but not yet DN_rec/DN_conv/m->cache
         * themselves (allocated below) -- a known, small underestimate left
         * inside the 12% margin rather than reordering allocation here.
         * xf_mode(m) is resolvable at this point (st_init/active_of already
         * ran) but is memoized process-globally, not per-Model* -- a
         * pre-existing, unrelated bug if this process ever builds two models
         * with different container formats (out of scope here). */
        double resident = rss_gb();
        double avail = mem_available_gb();
        const char *ram_env = getenv("RAM_GB");
        double ram_override = ram_env ? atof(ram_env) : 0.0;
        int n_active = layer_end - layer_begin;
        double slot_gb = 0.0, budget_gb = 0.0;
        cap = qwen36_cap_for_ram(resident, avail, ram_override,
                                  c->hidden, c->inter, c->n_experts, n_active,
                                  xf_mode(m), &slot_gb, &budget_gb);
        fprintf(stderr, "[qwen36] cache auto-sized: %d slots/layer of %d experts "
                        "(%.1f GB budget via %s, %.1f GB dense resident, %.0f MB/slot)\n",
                cap, c->n_experts, budget_gb,
                ram_override > 0.0 ? "RAM_GB" : "88% of available RAM",
                resident, slot_gb * 1000.0);
    }
    m->cache = calloc((size_t)c->n_layers, sizeof(LCache));
    for (int i = layer_begin; i < layer_end; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc((size_t)cap, sizeof(Slot));
        m->cache[i].slot_by_expert = malloc((size_t)c->n_experts * sizeof(int));
        if (!m->cache[i].slot_by_expert) { fprintf(stderr,"OOM expert cache index\n"); exit(1); }
        for (int e = 0; e < c->n_experts; e++) m->cache[i].slot_by_expert[e] = -1;
    }
    /* per-layer DeltaNet recurrent + conv state (only for linear_attention layers) */
    m->DN_rec = calloc((size_t)c->n_layers, sizeof(float*));
    m->DN_conv = calloc((size_t)c->n_layers, sizeof(float*));
    m->dn_dev_fresh = calloc((size_t)c->n_layers, 1); m->dn_host_stale = calloc((size_t)c->n_layers, 1); m->dn_dev = 0;
    for (int i = layer_begin; allocate_state && i < layer_end; i++) {
        if (c->is_attn[i]) { m->DN_rec[i] = NULL; m->DN_conv[i] = NULL; continue; }
        if (c->dn_vheads <= 0) { fprintf(stderr, "layer %d is DeltaNet but dn dims missing from meta\n", i); exit(1); }
        m->DN_rec[i]  = calloc((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, sizeof(float));
        m->DN_conv[i] = calloc((size_t)c->dn_conv_dim * (c->dn_convk - 1), sizeof(float));
    }
    m->freq = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint32_t));
    m->hot_pinned = 0; m->freq_token_count = 0;
    m->hot_n         = getenv("HOT")    ? atoi(getenv("HOT"))    : 0;
    m->warmup_tokens = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    m->token_count = 0;
    m->momentum_logits = calloc((size_t)c->n_layers * c->n_experts, sizeof(float));
    float sv = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    if (sv < 0.f) sv = 0.f; if (sv > 0.95f) sv = 0.95f;
    m->pilot_smooth = sv;
    m->is_pinned = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->seen = calloc((size_t)c->n_layers * c->n_experts, 1);
    m->resident_mode = getenv("COLIBRI_RESIDENT") ? atoi(getenv("COLIBRI_RESIDENT")) : 0;
    m->resident_collecting = 0;
    m->first_step = 1;
    m->is_queued = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    float cl = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;
    if (cl < 0.1f) cl = 0.1f; if (cl > 1.0f) cl = 1.0f;
    m->pilot_conf_limit = cl;
    m->dense_load_s = now_s() - t0;
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    model_init_range(m, snap, cap, bits, 0, 0, 1, 1);
}

/* scale counts per expert matrix: per-row (gs=0) or grouped along input dim */
static int64_t scale_count_gu(const Cfg *c){ return c->expert_gs ? (int64_t)c->inter * ((c->hidden + c->expert_gs - 1) / c->expert_gs) : c->inter; }
static int down_gs_of(const Cfg *c){ return c->expert_down_bits ? c->expert_down_gs : c->expert_gs; }
static int64_t scale_count_d (const Cfg *c){ int gs = down_gs_of(c); return gs ? (int64_t)c->hidden * ((c->inter + gs - 1) / gs) : c->hidden; }

/* Pure: no Model pointer, no globals, no I/O, same testability contract as
 * coli_resolve_cap / k3_cap_for_ram. cap<=0 sentinel ("auto") resolves to
 * this. Mirrors olmoe.c's cap<=0 budget block (resident + avail*0.88, same
 * margin as colibri.c's cap_for_ram), extended with the int4/xf_mode slot-size split that
 * slot_ensure_allocated() above uses (half the int8 bytes) and with
 * n_active_layers instead of always c->n_layers, for the Segment/Edge
 * partial-model layer-range builds. Approximates the scale-float block the
 * same way olmoe.c does (ignores expert_gs grouping precision) -- this is a
 * budget estimate under a safety margin, not exact accounting. */
static int qwen36_cap_for_ram(double resident_gb, double avail_gb, double ram_gb_override,
                               int hidden, int inter, int n_experts, int n_active_layers,
                               int is_int4, double *slot_gb_out, double *budget_gb_out) {
    double budget = ram_gb_override > 0.0 ? ram_gb_override : resident_gb + avail_gb * 0.88;
    if (budget_gb_out) *budget_gb_out = budget;
    double room = budget - resident_gb;
    double per_expert = (double)hidden * inter * (is_int4 ? 1.5 : 3.0);
    double slot_gb = (per_expert + (double)(inter * 2 + hidden) * sizeof(float)) / 1e9;
    if (slot_gb_out) *slot_gb_out = slot_gb;
    int layers = n_active_layers < 1 ? 1 : n_active_layers;
    int derived = room > 0.0 && slot_gb > 0.0 ? (int)(room / slot_gb / (double)layers) : 0;
    if (derived < 1) derived = 1;
    if (n_experts > 0 && derived > n_experts) derived = n_experts;
    return derived;
}

static void slot_ensure_allocated(Model *m, Slot *s) {
    if (s->g || s->pw) return;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden;
    int64_t nd = (int64_t)c->hidden * c->inter;
    if (xf_mode(m)) {
        /* half the bytes of the int8 block: the int4 stays packed */
        s->pw = malloc((size_t)(ng + ng + nd) / 2);
        if (!s->pw) { fprintf(stderr, "Error: OOM allocating slot weights\n"); exit(1); }
        float *s_block = falloc(2*scale_count_gu(c) + scale_count_d(c));
        s->gs = s_block; s->us = s_block + scale_count_gu(c); s->ds = s_block + 2*scale_count_gu(c);
        s->pinned = 0; s->is_int4 = 1; s->g = s->u = s->d = NULL; s->g4 = s->u4 = s->d4 = NULL;
        return;
    }
    int8_t *w_block = malloc(ng + ng + nd);
    if (!w_block) { fprintf(stderr, "Error: OOM allocating slot weights\n"); exit(1); }
    s->g = w_block;
    s->u = w_block + ng;
    s->d = w_block + ng + ng;
    float *s_block = falloc(2*scale_count_gu(c) + scale_count_d(c));
    s->gs = s_block;
    s->us = s_block + scale_count_gu(c);
    s->ds = s_block + 2*scale_count_gu(c);
    s->pinned = 0;
    s->is_int4 = 0;
    s->g4 = s->u4 = s->d4 = NULL;   /* packed int4 (allocated on int4 load if GPU int4 active) */
}


/* Unpack packed signed-int4 experts to int8. Two nibbles per byte; LOW nibble is
 * element 2k, HIGH is 2k+1, each signed 4-bit two's complement. Must stay in step
 * with pack_int4 in c/tools/convert_qwen36.py.
 *
 * This is the hottest thing on this engine's CPU decode path. Every expert cache
 * miss unpacks a whole expert -- 3*inter*hidden = 6.29M values on
 * Qwen3.6-35B-A3B -- and a fit of decode time against miss count put ~60% of
 * decode inside it.
 *
 * The loop this replaces was indexed by ELEMENT, so it reloaded raw[i>>1], did an
 * i&1 select and took a branch for every value. Walking BYTES and sign-extending
 * by shifting removes the branch. Vectorising then needs an INTERLEAVING store,
 * which is why no compiler gets there from the scalar form: the two nibble
 * streams are consecutive in the output, so it needs vst2q on NEON or
 * unpacklo/unpackhi on AVX2 rather than a strided store. Checking the
 * disassembly after the branchless rewrite confirmed zero vector registers.
 *
 * Sign extension is branchless in both paths. In vectors the signed value of a
 * nibble n is (n ^ 8) - 8; the scalar tail gets the same result more cheaply by
 * casting the nibble into the top four bits and arithmetic-shifting back down.
 *
 * Integer throughout, so unlike a float reduction this is bit-exact by
 * construction rather than by tolerance -- verified identical to the original
 * branching form over all 256 byte values, at every length around a vector
 * boundary, and on a full-size 6.29M-value random expert, on AVX2 and NEON. */
static void unpack_int4_to_int8(int8_t *out, const uint8_t *raw, int64_t n)
{
    int64_t nb = n / 2, b = 0;                  /* n is 3*inter*hidden, always even */
#if defined(__AVX2__)
    const __m128i m4 = _mm_set1_epi8(0x0F), e8 = _mm_set1_epi8(8);
    for (; b + 16 <= nb; b += 16) {
        __m128i by = _mm_loadu_si128((const __m128i *)(raw + b));
        __m128i lo = _mm_and_si128(by, m4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);   /* 16-bit shift: mask after */
        lo = _mm_sub_epi8(_mm_xor_si128(lo, e8), e8);
        hi = _mm_sub_epi8(_mm_xor_si128(hi, e8), e8);
        _mm_storeu_si128((__m128i *)(out + 2 * b),      _mm_unpacklo_epi8(lo, hi));
        _mm_storeu_si128((__m128i *)(out + 2 * b + 16), _mm_unpackhi_epi8(lo, hi));
    }
#elif defined(__ARM_NEON)
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t e8 = vdupq_n_s8(8);
    for (; b + 16 <= nb; b += 16) {
        uint8x16_t by = vld1q_u8(raw + b);
        int8x16x2_t z;
        z.val[0] = vsubq_s8(veorq_s8(vreinterpretq_s8_u8(vandq_u8(by, m4)), e8), e8);
        z.val[1] = vsubq_s8(veorq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), e8), e8);
        vst2q_s8(out + 2 * b, z);               /* the interleaved store is the trick */
    }
#endif
    for (; b < nb; b++) {                       /* tail, and the whole loop if scalar */
        uint8_t byte = raw[b];
        out[2 * b]     = (int8_t)(byte << 4) >> 4;
        out[2 * b + 1] = (int8_t)(byte & 0xF0) >> 4;
    }
}

static void load_expert_merged(Model *m, int layer, int eid, Slot *s) {
    char nm[256], qsnm[256];
    int la = m->active_of[layer];   /* container stores experts under active index */
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.merged_weight", la, eid);
    snprintf(qsnm, sizeof(qsnm), "model.layers.%d.mlp.experts.%d.qs", la, eid);
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_s = 2*scale_count_gu(cc) + scale_count_d(cc);
    st_tensor *tw = st_find(&m->S, nm), *ts = st_find(&m->S, qsnm);
    int64_t want_mixed = ng + nd;   /* gate|up packed int4 (ng bytes) + down int8 (nd bytes) */
    if (!tw || (tw->nbytes != want_w && tw->nbytes != want_w / 2 && tw->nbytes != want_mixed)) {
        fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld (int8), %lld (int4) or %lld (int4 gate/up + int8 down)\n",
                nm, (long long)(tw ? tw->nbytes : -1), (long long)want_w, (long long)(want_w / 2), (long long)want_mixed); exit(1); }
    if (!ts || ts->numel != want_s) {
        fprintf(stderr, "%s: scale array is %lld elems — expected %lld (refusing)\n",
                qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1); }
    /* int4 detection by ON-DISK SIZE (robust against a mislabeled meta.ebits, e.g. the
       i8 container whose meta says ebits=4 but stores int8).  True int4 packed uint8 is
       exactly N/2 bytes (N = 3*inter*hidden, always even).  Unpack in-place to int8 so the
       rest of the MoE path (matmul_q) is unchanged.  Nibble convention (must match
       c/tools/convert_qwen36.py pack_int4): LOW nibble = element 2k, HIGH nibble = 2k+1;
       each nibble is signed 4-bit (sign-extend if bit3 set). */
    if (tw->nbytes == want_mixed) {
        /* mixed layout: unpack gate|up (2*ng int4 elements in ng bytes) into the
         * slot's g|u block, copy down's int8 rows behind them. No packed copy is
         * kept: the tier does not take this layout yet (main refuses it). */
        static int noted_m = 0;
        if (!noted_m) { fprintf(stderr, "[qwen36] mixed expert layout detected (int4 gate/up, int8 down) — unpacking gate/up to int8 in slot\n"); noted_m = 1; }
        uint8_t *raw = (uint8_t *)malloc((size_t)want_mixed);
        if (!raw) { fprintf(stderr, "OOM reading mixed expert %s\n", nm); exit(1); }
        st_read_raw(&m->S, nm, raw, 1);
        unpack_int4_to_int8(s->g, raw, ng + ng);          /* 2*ng elements from ng bytes */
        memcpy(s->d, raw + ng, (size_t)nd);
        free(raw);
        s->is_int4 = 0;
        free(s->g4); free(s->u4); free(s->d4); s->g4 = s->u4 = s->d4 = NULL;
        st_read_f32(&m->S, qsnm, s->gs, 0);
        return;
    }
    if (tw->nbytes == want_w / 2) {
        static int noted = 0;
        if (!noted) { fprintf(stderr, "[qwen36] int4 packed weights detected — %s\n", s->pw ? "kept int4, repacked planar for expert_ffn.h" : "unpacking to int8 in slot"); noted = 1; }
        uint8_t *raw = (uint8_t *)malloc((size_t)(want_w / 2));
        if (!raw) { fprintf(stderr, "OOM reading int4 expert %s\n", nm); exit(1); }
        st_read_raw(&m->S, nm, raw, 1);
        if (s->pw) {
            /* shared kernel: pairs -> planar, never int8 */
            int64_t gp = ng / 2;
            xf_repack_pairs_signed(s->pw,          raw,          cc->inter,  cc->hidden);
            xf_repack_pairs_signed(s->pw + gp,     raw + gp,     cc->inter,  cc->hidden);
            xf_repack_pairs_signed(s->pw + 2 * gp, raw + 2 * gp, cc->hidden, cc->inter);
            s->is_int4 = 1;
            free(raw);
            st_read_f32(&m->S, qsnm, s->gs, 0);
            return;
        }
        unpack_int4_to_int8(s->g, raw, want_w);
        s->is_int4 = 1;
        /* Free any previous occupant first (LRU slot reuse). */
        free(s->g4); free(s->u4); free(s->d4); s->g4 = s->u4 = s->d4 = NULL;
        /* Keep the packed int4 bytes alongside the unpacked int8 copy only when
         * the CUDA tier is actually running: they are its upload source, and
         * they let slot_ensure_int8() rematerialize an evicted expert whose
         * int8 copy the warmstart freed. Without the tier nothing ever reads
         * them, and keeping them would add ~50% to expert-cache RSS on the
         * recommended gs64 container -- so qt_ready() gates the allocation.
         * Under CUDA=0 that is an inline `return 0` and this costs nothing. */
        if (qt_ready()) {
            int64_t gp = ng / 2, up = ng / 2, dp = nd / 2;   /* gate/up/down packed sizes */
            s->g4 = (uint8_t *)malloc((size_t)gp);
            s->u4 = (uint8_t *)malloc((size_t)up);
            s->d4 = (uint8_t *)malloc((size_t)dp);
            if (!s->g4 || !s->u4 || !s->d4) { fprintf(stderr, "OOM int4-packed %s\n", nm); exit(1); }
            memcpy(s->g4, raw,           (size_t)gp);
            memcpy(s->u4, raw + gp,      (size_t)up);
            memcpy(s->d4, raw + gp + up, (size_t)dp);
        }
        free(raw);
    } else {
        s->is_int4 = 0;
        free(s->g4); free(s->u4); free(s->d4); s->g4 = s->u4 = s->d4 = NULL;
        st_read_raw(&m->S, nm, s->g, 1);
    }
    st_read_f32(&m->S, qsnm, s->gs, 0);
}

/* Robust int4 detection by on-disk size of one expert tensor (ignores a possibly
 * mislabeled meta.ebits — cf. load_expert_merged).  Returns 1 if the container
 * stores true int4 packed weights, 0 otherwise.  Used to pick the Vulkan
 * pipeline at init time. */
static int container_layer_is_int4(Model *m, int layer) {
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    char nm[256];
    snprintf(nm, sizeof(nm),
             "model.layers.%d.mlp.experts.0.merged_weight", layer);
    st_tensor *tw = st_find(&m->S, nm);
    if (!tw) return 0;
    return (tw->nbytes == want_w / 2) ? 1 : 0;
}

/* Rematerialize a slot's int8 block from its packed int4 copy on demand
 * (~0.5 ms, no container access). Needed after the warmstart freed the int8
 * copies of VRAM-resident experts and one of them got LFRU-evicted. */
static void slot_ensure_int8(Model *m, Slot *s) {
    if (s->g || !s->g4) return;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden, nd = (int64_t)c->hidden * c->inter;
    int8_t *w = malloc((size_t)(ng + ng + nd));
    if (!w) { fprintf(stderr, "OOM slot_ensure_int8\n"); exit(1); }
    /* #1271's unpack_int4_to_int8 (branchless, AVX2/NEON/scalar -- see its
     * definition above load_expert_merged, which already uses it for the
     * container's int4 read) instead of this function's own separate scalar
     * copy of the same nibble-unpack math. g4/u4/d4 are three independently
     * malloc'd packed buffers here (unlike load_expert_merged's single
     * contiguous `raw`), so one call per segment. */
    unpack_int4_to_int8(w,           s->g4, ng);
    unpack_int4_to_int8(w + ng,      s->u4, ng);
    unpack_int4_to_int8(w + ng + ng, s->d4, nd);
    s->g = w; s->u = w + ng; s->d = w + ng + ng;
}

/* Experts the VRAM tier evicted since the last token (LFRU and re-plan swaps)
 * lose their VRAM copy, and on an int4 container their int8 copy went at the
 * warmstart; the first CPU miss then pays slot_ensure_int8 (a malloc and a
 * 3 MB unpack) inside the decode step. Rebuilding them here, in parallel and
 * before the layers run, keeps that out of the miss path: measured on the
 * 3070, 2,000 re-plan victims cost ~7 ms/token over the next 300 tokens on
 * the miss path and nothing here. */
static void expert_get(Model *m, int layer, int eid, Slot **out);
static void tier_rebuild_evicted(Model *m) {
    if (!qt_ready()) return;
    int ls[512], es[512];
    int n = qt_evicted_take(ls, es, 512);
    if (n <= 0) return;
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++) {
        Slot *e; expert_get(m, ls[i], es[i], &e);
        slot_ensure_int8(m, e);
    }
}

/* QT_PREFILL_REPLAN=1: after each prefill layer's routing, hand the tier that
 * layer's counts over the prompt rows (qt_replan), so residents this prompt
 * never routes to make way for the ones it routes to most while the rest of
 * the prefill still computes. Measured on the 3070 (docs/qwen36-cuda-tier.md):
 * +13..37 points decode hit rate over a heat file from other prompts, at equal
 * budget, output unchanged (placement never changes routing). */
static int prefill_replan_on(void) {
    static int on = -1;
    if (on < 0) { const char *p = getenv("QT_PREFILL_REPLAN"); on = p && *p == '1'; }
    return on;
}
static int prefill_replan_cap(void) {
    static int cap = -1;
    if (cap < 0) { const char *p = getenv("QT_PREFILL_REPLAN_MAX"); cap = p ? atoi(p) : 24; if (cap < 0) cap = 0; }
    return cap;
}

/* Segna l'esperto instradato per la bitmap HITS della dashboard. Vive qui,
 * fuori dalla regione QWEN36_NO_MAIN: expert_get la chiama anche nel build
 * del segment adapter, dove il resto della telemetria serve non esiste. */
static pthread_mutex_t g_ehit_mx = PTHREAD_MUTEX_INITIALIZER;
static void ehit_mark(Model *m, int layer, int eid){
    const Cfg *c=&m->c;
    /* The first touch can come from a parallel region (qwen36: the tier
     * warmstart's omp loop calls expert_get from twelve threads at once):
     * one thread published m->ehit while it was still filling the rows and
     * a sibling dereferenced m->ehit[layer] == NULL -- SIGSEGV in about one
     * run in twelve on a CUDA warmstart. Build the table privately, publish
     * it once under a lock (double-checked), and read it with acquire order. */
    uint8_t **ehit=__atomic_load_n(&m->ehit,__ATOMIC_ACQUIRE);
    if(!ehit){
        pthread_mutex_lock(&g_ehit_mx);
        ehit=m->ehit;
        if(!ehit){
            ehit=calloc((size_t)c->n_layers,sizeof(uint8_t*));
            for(int i=0;i<c->n_layers;i++) ehit[i]=calloc((size_t)c->n_experts,1);
            __atomic_store_n(&m->ehit,ehit,__ATOMIC_RELEASE);
        }
        pthread_mutex_unlock(&g_ehit_mx);
    }
    if(layer>=0&&layer<c->n_layers&&eid>=0&&eid<c->n_experts) ehit[layer][eid]=1;
}
/* hold=1 marks the slot busy before the lock drops, so nothing can evict it
 * between this lookup and the caller's slot_release. */
static void expert_fetch(Model *m, int layer, int eid, Slot **out, int hold) {
    ehit_mark(m, layer, eid);   /* tocca solo m->ehit[layer][eid] */
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_pilot_mx);
    Slot *hit = slot_indexed(m, layer, eid);
    if (hit) {
        m->hits++; hit->used = ++m->clock; *out = hit;
        if (hold) slot_hold(hit);
        pthread_mutex_unlock(&g_pilot_mx); return;
    }
    m->miss++;
    Cfg *c = &m->c; Slot *s;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        /* LRU eviction: an unpinned slot first, else the oldest pinned one;
         * never one being loaded or computed from. */
        int lru = slot_victim(lc, layer, 0);
        if (lru < 0) lru = slot_victim(lc, layer, 1);
        while (lru < 0) {
            /* Every slot is held or being loaded. Only this thread holds (the
             * PILOT worker never does) and a moe run holds fewer slots than
             * cap, so with nothing loading a hold leaked: say so, don't hang. */
            int loading = 0;
            for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid < 0) loading = 1;
            if (!loading) { fprintf(stderr, "qwen36: layer %d: all %d expert slots held, none loading\n", layer, lc->n); exit(1); }
            /* EVERY slot is in flight: each buffer is owned by an unlocked pread
             * in the pilot worker (or a demand load) that will publish into it.
             * The old last resort (lru=0) stole such a slot mid-load — two writers
             * racing the same slab, then whichever published last decided the
             * expert id the resident bytes answered to. Wait for a publish instead
             * and rescan; in-flight always drains because a load either finishes
             * or the process is already dead in the water.
             *
             * Taken verbatim from olmoe.c, which this cache derives from and
             * where this exact fallback was deleted for exactly this reason.
             * Reachable whenever cap is smaller than the number of candidates a
             * layer has in flight — PILOT queues up to 128 per layer — i.e. on
             * any small-RAM box, and it corrupts silently rather than crashing. */
            pthread_mutex_unlock(&g_pilot_mx);
            sleep_ms(1);
            pthread_mutex_lock(&g_pilot_mx);
            lru = slot_victim(lc, layer, 1);
        }
        s = &lc->slots[lru]; s->pinned = 0;
        if (vkt_ram_first(layer, s->eid)) vkt_ram_gave();
    }
    cache_hide(m, layer, s); s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);
    double t_read0 = now_s();
    load_expert_merged(m, layer, eid, s);
    double t_read = now_s() - t_read0;
    pthread_mutex_lock(&g_pilot_mx);
    m->t_disk += t_read;        /* sotto lock: qui arrivano anche i thread del PILOT */
    cache_publish(m, layer, s, eid); s->pinned = m->is_pinned[layer * c->n_experts + eid]; s->used = ++m->clock;
    if (hold) slot_hold(s);
    *out = s; pthread_mutex_unlock(&g_pilot_mx);
}
static void expert_get(Model *m, int layer, int eid, Slot **out) { expert_fetch(m, layer, eid, out, 0); }
/* expert_get for a caller that computes from the slot after the lock drops:
 * it stays resident until slot_release. */
static Slot *expert_hold(Model *m, int layer, int eid) { Slot *s; expert_fetch(m, layer, eid, &s, 1); return s; }

static void pin_hot_experts(Model *m) {
    Cfg *c = &m->c;
    if (m->hot_n <= 0 || m->hot_pinned) return;
    m->hot_pinned = 1;
    int is_dynamic = (m->hot_n >= 100);
    double thresh = is_dynamic ? (double)m->hot_n / 1000.0 : 0.0;
    int pinned_total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        uint32_t *freq_l = m->freq + (int64_t)l * c->n_experts;
        uint64_t layer_total = 0;
        for (int e = 0; e < c->n_experts; e++) layer_total += freq_l[e];
        if (layer_total == 0) continue;
        int max_pin = m->cache[l].cap - 8; if (max_pin < 4) max_pin = 4;
        int hn = is_dynamic ? max_pin : (m->hot_n < c->n_experts ? m->hot_n : c->n_experts);
        if (hn > 256) hn = 256;
        int hot_eids[256], actual_hn = 0;
        for (int k = 0; k < hn; k++) {
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < c->n_experts; e++) {
                int already = 0;
                for (int j = 0; j < k; j++) if (hot_eids[j] == e) { already = 1; break; }
                if (!already && freq_l[e] > bv) { bv = freq_l[e]; best = e; }
            }
            if (best < 0 || bv == 0) break;
            if (is_dynamic && bv < thresh * layer_total) break;
            hot_eids[k] = best; actual_hn++;
        }
        for (int k = 0; k < actual_hn; k++) {
            int eid = hot_eids[k];
            m->is_pinned[l * c->n_experts + eid] = 1;
            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            Slot *resident = slot_indexed(m, l, eid);
            if (resident) { resident->pinned = 1; found = 1; }
            pthread_mutex_unlock(&g_pilot_mx);
            if (!found && g_pilot > 0) {
                ensure_pilot_worker_started(m);
                unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                int gidx = l * c->n_experts + eid;
                pthread_mutex_lock(&g_pilot_mx);
                int already = m->is_queued[gidx];
                if (!already && w - r < 4096) {
                    pilot_q[w & 4095].l = l; pilot_q[w & 4095].e = eid; m->is_queued[gidx] = 1;
                    __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
                }
                pthread_mutex_unlock(&g_pilot_mx);
            }
            pinned_total++;
        }
    }
    fprintf(stderr, "[HOT] Pinned %d experts (top-%d/layer) after %d warmup tokens\n", pinned_total, m->hot_n, m->freq_token_count);
}

/* COLIBRI_RESIDENT: after prefill (mode 1) and continuously through decode (mode 2),
 * pin every expert this prompt routed to, so the CPU LRU never evicts their RAM
 * slots. `quiet` suppresses the log line when nothing new was pinned
 * (used for the per-token mid-decode calls). A per-layer pin budget = cap prevents
 * pinning more experts than fit in the cache (which would deadlock the LRU). */
static int apply_resident(Model *m, int quiet) {
    Cfg *c = &m->c;
    int newly = 0, over = 0;
    for (int l = 0; l < c->n_layers; l++) {
        uint8_t *row = m->seen + (int64_t)l * c->n_experts;
        int cap = m->cache[l].cap;
        int already = 0;
        for (int e = 0; e < c->n_experts; e++) if (m->is_pinned[l * c->n_experts + e]) already++;
        int budget = cap - already;                 /* free pin slots in this layer */
        int seen = 0;
        for (int e = 0; e < c->n_experts; e++) {
            if (!row[e]) continue;
            seen++;
            if (m->is_pinned[l * c->n_experts + e]) continue;   /* already pinned */
            if (budget <= 0) { over++; continue; }             /* layer full, skip */
            m->is_pinned[l * c->n_experts + e] = 1;
            newly++; budget--;
        }
        if (seen > cap) over += seen - cap;
        LCache *lc = &m->cache[l];
        for (int i = 0; i < lc->n; i++)
            if (lc->slots[i].eid >= 0 && row[lc->slots[i].eid])
                lc->slots[i].pinned = 1;
    }
    if (!quiet || newly > 0)
        fprintf(stderr, "[RESIDENT] Pinned %d new experts (CPU no-evict -> GPU resident)%s\n",
                newly, over > 0 ? " | WARN: exceed per-layer cap, raise cap for full coverage" : "");
    return newly;
}

/* ---------- RoPE: applied to the FIRST rope_dim dims of each head (Qwen3 partial rope) ---------- */
static void rope_head_partial(float *x, int pos, int rope_dim, int head_dim, float theta) {
    int h = rope_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f * j / rope_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* The same rotation with a position per axis (interleaved M-RoPE, see Cfg).
 * With the three positions equal it is rope_head_partial, operation for
 * operation. */
static void rope_head_mrope(float *x, const int pos3[3], const int sec[3], int rope_dim, float theta) {
    int h = rope_dim / 2;
    for (int j = 0; j < h; j++) {
        int axis = (j % 3 == 1 && j < 3 * sec[1]) ? 1 : (j % 3 == 2 && j < 3 * sec[2]) ? 2 : 0;
        float inv = powf(theta, -2.0f * j / rope_dim);
        float ang = pos3[axis] * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}
static inline void mrope_at(const Model *m, int p, int pos3[3]) {
    if (m->mpos && p >= 0 && p < m->mpos_len) {
        pos3[0] = m->mpos[(size_t)p * 3]; pos3[1] = m->mpos[(size_t)p * 3 + 1]; pos3[2] = m->mpos[(size_t)p * 3 + 2];
    } else pos3[0] = pos3[1] = pos3[2] = p + m->rope_delta;
}

/* Gated Attention (GQA) matching HF Qwen3_5MoeAttention:
 *  - q_proj outputs query(head_dim) ++ attn_output_gate(head_dim); k/v are head_dim.
 *  - per-head q/k RMSNorm (weight [head_dim], 1.0+weight).
 *  - partial RoPE on the first rotary_dim dims of each head (text: mRoPE == standard).
 *  - scale = head_dim^-0.5; GQA repeat_kv.
 *  - attn_out = attn_out * sigmoid(gate), then o_proj (input dim = q_heads*head_dim). */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, D = c->hidden;
    int kvd = c->k_head_dim;
    int qdim = c->q_head_dim;                  /* per-head q total (query+gate) */
    int q_out = H * qdim;                      /* q_proj output dim */
    int kv_out = KV * kvd;                     /* k/v_proj output dim */
    int q_per_kv = H / KV;
    int rotary = c->rotary_dim;
    /* HF always chunks q_proj output into query(head_dim) ++ gate(head_dim),
     * regardless of the attn_output_gate config flag -- so split whenever the
     * q per-head dim exceeds the (k/v) head dim. */
    int gate_dim = (qdim > hd) ? (qdim - hd) : 0;
    float *q = falloc((int64_t)S*q_out);
    float *k = falloc((int64_t)S*kv_out);
    float *vv= falloc((int64_t)S*kv_out);
    /* The projections the tier placed answer from VRAM for the whole batch,
     * with one backend call per matrix; unavailable handles use CPU matmul. */
    if (!qtd_batch(l->qth_q, q, x, S, D, q_out))   matmul_d(q, x, &l->q, S, D, q_out);
    if (!qtd_batch(l->qth_k, k, x, S, D, kv_out))  matmul_d(k, x, &l->k, S, D, kv_out);
    if (!qtd_batch(l->qth_v, vv, x, S, D, kv_out)) matmul_d(vv, x, &l->v, S, D, kv_out);
    /* split q into query (first hd) and gate (next gate_dim), both per head */
    float *query = falloc((int64_t)S*H*hd);
    float *gate  = gate_dim ? falloc((int64_t)S*H*gate_dim) : NULL;   /* qwen3_moe: no gate */
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            const float *qs = q + (int64_t)s*q_out + hh*qdim;
            memcpy(query + ((int64_t)s*H + hh)*hd, qs, hd*sizeof(float));
            if (gate_dim) memcpy(gate + ((int64_t)s*H + hh)*gate_dim, qs + hd, gate_dim*sizeof(float));
        }
    }
    const Q36Row *mr = m->mux_rows;   /* a multiplexed step: each row's own conversation and position */
    for (int s = 0; s < S; s++) {
        int pos = mr ? mr[s].pos : pos_base + s;
        /* the rope of the row's conversation: an image turn's decode rows sit at
         * pos + rope_delta on every axis (mrope_at past the prompt) */
        int mrope = mr ? (mr[s].seq->mpos || mr[s].seq->rope_delta) : (m->mpos || m->rope_delta);
        int p3[3];
        if (mrope) { if (mr) p3[0] = p3[1] = p3[2] = pos + mr[s].seq->rope_delta; else mrope_at(m, pos, p3); }
        for (int hh = 0; hh < H; hh++) {
            float *qh = query + ((int64_t)s*H + hh)*hd;
            if (l->qn) rmsnorm_row(qh, qh, l->qn, hd, c->eps);
            if (mrope) rope_head_mrope(qh, p3, c->mrope_section, rotary, c->theta);
            else rope_head_partial(qh, pos, rotary, hd, c->theta);
        }
        for (int kvh = 0; kvh < KV; kvh++) {
            float *kh = k + (int64_t)s*KV*kvd + kvh*kvd;
            if (l->kn) rmsnorm_row(kh, kh, l->kn, kvd, c->eps);
            if (mrope) rope_head_mrope(kh, p3, c->mrope_section, rotary, c->theta);
            else rope_head_partial(kh, pos, rotary, kvd, c->theta);
        }
    }
    for (int s = 0; s < S; s++) for (int kvh = 0; kvh < KV; kvh++) {
        int t = mr ? mr[s].pos : pos_base + s;
        float *Kl = mr ? mr[s].seq->K[layer] : m->K[layer], *Vl = mr ? mr[s].seq->V[layer] : m->V[layer];
        memcpy(Kl + ((int64_t)kvh*m->max_t + t)*kvd, k + (int64_t)s*KV*kvd + kvh*kvd, kvd*sizeof(float));
        memcpy(Vl + ((int64_t)kvh*m->max_t + t)*kvd, vv + (int64_t)s*KV*kvd + kvh*kvd, kvd*sizeof(float));
    }
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*H*hd);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int kvh = hh / q_per_kv;
            int qpos = mr ? mr[s].pos : pos_base + s;
            const float *Kl = mr ? mr[s].seq->K[layer] : m->K[layer], *Vl = mr ? mr[s].seq->V[layer] : m->V[layer];
            const float *qv = query + ((int64_t)s*H + hh)*hd;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = m->attn_sc + (int64_t)tid * m->kv_cap;
            for (int t = 0; t <= qpos; t++) {
                const float *kv = Kl + ((int64_t)kvh*m->max_t + t)*kvd;
                float acc = dot_f32_lanes(qv, kv, kvd);
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + ((int64_t)s*H + hh)*hd;
            for (int dd = 0; dd < kvd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = Vl + ((int64_t)kvh*m->max_t + t)*kvd;
                float a = sc[t]; for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    /* apply attn_output_gate: attn_out *= sigmoid(gate). Without a gate (Qwen3 MoE)
     * the output passes as it is: a missing gate is not sigmoid(0) = 0.5. */
    float *ag = falloc((int64_t)S*H*hd);
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) for (int dd = 0; dd < hd; dd++) {
        int o = ((int64_t)s*H + hh)*hd + dd;
        ag[o] = gate_dim ? ctx[o] * (1.f / (1.f + expf(-gate[o]))) : ctx[o];
    }
    if (!qtd_batch(l->qth_o, out, ag, S, H*hd, D)) matmul_d(out, ag, &l->o, S, H*hd, D);
    free(q); free(k); free(vv); free(query); free(gate); free(ctx); free(ag);
}

/* Batch the CPU shared expert across prompt rows.  The three resident matrices
 * are otherwise traversed S times even though every row uses the same weights.
 * Decode (S=1) stays on the scalar matmul_d -> matmul_q path above.  Long
 * prompts are chunked so the additional activations consume at most 32 MiB;
 * QWEN_SHARED_BATCH=0 is an exact scalar A/B switch, while a positive value
 * sets a smaller row cap. */
static int qwen_shared_batch_rows(int S, int D, int I) {
    if (S <= 1) return 1;
    const char *env = getenv("QWEN_SHARED_BATCH");
    if (env) {
        int requested = atoi(env);
        if (requested <= 0) return 1;
        if (requested < S) S = requested;
    }
    int64_t row_bytes = ((int64_t)2*I + D) * (int64_t)sizeof(float);
    int64_t bounded = (32LL << 20) / (row_bytes > 0 ? row_bytes : 1);
    if (bounded < 1) bounded = 1;
    if (S > bounded) S = (int)bounded;
    return S;
}

static void qwen_shared_experts_cpu(Model *m, Layer *l, const float *x, int S,
                                    float *out, float *g, float *u, float *hh) {
    Cfg *c=&m->c; int D=c->hidden, I=c->shared_inter;
    if (I <= 0) return;   /* no shared expert (qwen3_moe) */
    int B=qwen_shared_batch_rows(S,D,I);
    double _ts=tm_now();
    if (B == 1) {
        for (int s=0;s<S;s++) {
            const float *xs=x+(int64_t)s*D;
            if(!qtd(l->qth_shg,g,xs,D,I)) matmul_d(g,xs,&l->sh_g,1,D,I);
            if(!qtd(l->qth_shu,u,xs,D,I)) matmul_d(u,xs,&l->sh_u,1,D,I);
            for(int i=0;i<I;i++){float sv=g[i];g[i]=(sv/(1.f+expf(-sv)))*u[i];}
            if(!qtd(l->qth_shd,hh,g,I,D)) matmul_d(hh,g,&l->sh_d,1,I,D);
            float sgate=1.f;
            if(l->sh_gate){float sg=0.f;for(int i=0;i<D;i++)sg+=xs[i]*l->sh_gate[i];sgate=1.f/(1.f+expf(-sg));}
            float *os=out+(int64_t)s*D;
            for(int d=0;d<D;d++)os[d]+=sgate*hh[d];
        }
    } else {
        float *bg=falloc((int64_t)2*B*I), *bu=bg+(int64_t)B*I;
        float *bh=falloc((int64_t)B*D);
        for(int base=0;base<S;base+=B){
            int rows=S-base<B?S-base:B;
            matmul_d(bg,x+(int64_t)base*D,&l->sh_g,rows,D,I);
            matmul_d(bu,x+(int64_t)base*D,&l->sh_u,rows,D,I);
            for(int64_t q=0;q<(int64_t)rows*I;q++){float sv=bg[q];bg[q]=(sv/(1.f+expf(-sv)))*bu[q];}
            matmul_d(bh,bg,&l->sh_d,rows,I,D);
            for(int s=0;s<rows;s++){
                const float *xs=x+(int64_t)(base+s)*D;
                float sgate=1.f;
                if(l->sh_gate){float sg=0.f;for(int i=0;i<D;i++)sg+=xs[i]*l->sh_gate[i];sgate=1.f/(1.f+expf(-sg));}
                float *os=out+(int64_t)(base+s)*D;const float *hs=bh+(int64_t)s*D;
                for(int d=0;d<D;d++)os[d]+=sgate*hs[d];
            }
        }
        free(bg);free(bh);
    }
    tm_add(S,3,tm_now()-_ts);
}

/* MoE: grouped top-k routing (+ optional router bias) + shared expert.
 * Mirrors HF Qwen3 MoE: softmax(gate), optional group-limited top-k, normalized
 * weights, sum routed experts, then add the un-gated shared expert. */
/* One MoE layer through expert_ffn.h. The experts a run holds must all be
 * resident at once: each is held (expert_hold) until the kernel is done with
 * it, so neither the run's next load nor the PILOT worker can evict it, and
 * the batch is cut to what the layer cache can hold:
 * the whole prompt chunk when cap covers S*K slots, one token when it covers
 * K, one (token, expert) pair otherwise (cap=1 in CI evicts on every routed
 * expert). out starts at zero (moe) and every cut adds into it through the
 * one xf_moe_add call below, val*expert in rank order, so the three cuts
 * produce the same bits. A pair run used to go through a zeroed buffer and a
 * second add: two roundings where the fused multiply-add of an FMA build
 * does one, and cap=1 moved the logits in their last bits. */
static void moe_xf_run(Model *m, int layer, const float *x, int S, float *out, const int *idx, const float *val) {
    Cfg *c = &m->c; int D = c->hidden, K = c->topk, F = c->inter;
    int cap = m->cache[layer].cap;
    int64_t gp = (int64_t)F * D / 2;
    int per = cap >= S * K ? S : 1;           /* tokens per run */
    int kper = cap >= K ? K : 1;              /* experts per run */
    int n = per * kper;
    XfExpert *ex = malloc(sizeof(XfExpert) * (size_t)n);
    const XfExpert **exp = malloc(sizeof(XfExpert *) * (size_t)n);
    int *ridx = malloc(sizeof(int) * (size_t)n); float *rval = falloc(n);
    Slot **held = malloc(sizeof(Slot *) * (size_t)n);
    void *scratch = malloc(xf_moe_scratch_bytes(per, kper, D, F));
    if (!ex || !exp || !ridx || !held || !scratch) { fprintf(stderr, "OOM moe_xf_run\n"); exit(1); }
    int timed = tm_on() && S == 1;
    for (int s0 = 0; s0 < S; s0 += per) {
        for (int k0 = 0; k0 < K; k0 += kper) {
            double t0 = timed ? tm_now() : 0;
            for (int s = 0; s < per; s++) for (int k = 0; k < kper; k++) {
                int src = (s0 + s) * K + (k0 + k), dst = s * kper + k;
                ridx[dst] = idx[src]; rval[dst] = val[src]; exp[dst] = NULL; held[dst] = NULL;
                if (idx[src] < 0) continue;
                Slot *e = held[dst] = expert_hold(m, layer, idx[src]);
                ex[dst].g4 = e->pw; ex[dst].u4 = e->pw + gp; ex[dst].d4 = e->pw + 2 * gp;
                ex[dst].gs = e->gs; ex[dst].us = e->us; ex[dst].ds = e->ds;
                exp[dst] = &ex[dst];
            }
            double t1 = timed ? tm_now() : 0;
            xf_moe_add(out + (int64_t)s0 * D, x + (int64_t)s0 * D, per, kper, D, F, ridx, rval, exp, xf_act_mode(), scratch);
            if (timed) { double t2 = tm_now(); g_xf_load += t1 - t0; g_xf_run += t2 - t1; }
            for (int i = 0; i < n; i++) if (held[i]) slot_release(held[i]);
        }
    }
    free(ex); free(exp); free(ridx); free(rval); free(held); free(scratch);
}

/* ---------- CACHE_ROUTE: residency-aware top-K fill (docs/CACHE_ROUTE.md) ----------
 * The GLM engine's max-rank lever (arXiv:2412.00099) with one more level: an
 * expert already in the VRAM tier outranks one that is only in the RAM cache,
 * which outranks one on disk. The ranking is the same softmax mass the plain
 * path uses, restricted to `keep`. Selection inside the top-`win` window: the
 * true top-J first, then VRAM-resident experts in rank order, then RAM-
 * resident ones, then the true ranking. lvl(ctx, e) answers 2 / 1 / 0 and is
 * asked only for ranked candidates past J. val[] carries the raw mass, with
 * substitutes scaled by alpha; moe() renormalises afterwards exactly as it
 * does for the plain top-K. Callable without a Model so
 * tests/test_qwen36_cache_route.c can pin the selection against a residency
 * table. */
#define ROUTE_RANK_MAX 256
static void route_select(const float *pr, const uint8_t *keep, int E, int K,
                         int J, int M, float P, float alpha,
                         int (*lvl)(void *ctx, int e), void *ctx,
                         int *idx, float *val, RouteStats *st) {
    if (K > ROUTE_RANK_MAX) K = ROUTE_RANK_MAX;
    if (J < 0) J = 0; if (J > K) J = K;
    int cap = (P > 0.f && P < 1.f) ? (M > 4*K ? M : 4*K) : (M > K ? M : K);
    if (cap > E) cap = E;
    if (cap > ROUTE_RANK_MAX) cap = ROUTE_RANK_MAX;
    int rank[ROUTE_RANK_MAX]; float rw[ROUTE_RANK_MAX]; int8_t rl[ROUTE_RANK_MAX];
    int n = 0;
    for (int r = 0; r < cap; r++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            if (keep && !keep[e]) continue;
            int taken = 0; for (int j = 0; j < n; j++) if (rank[j] == e) { taken = 1; break; }
            if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
        }
        if (best < 0) break;
        rank[n] = best; rw[n] = bv; n++;
    }
    int Kt = K < n ? K : n;             /* the true top-K is rank[0..Kt) */
    int win = n;
    if (P > 0.f && P < 1.f) {           /* cumulative-mass window: grow past K until P of the ranked mass */
        float tot = 1e-20f; for (int r = 0; r < n; r++) tot += rw[r] > 0 ? rw[r] : 0;
        float cum = 0; win = Kt;
        for (int r = 0; r < n; r++) { cum += rw[r] > 0 ? rw[r] : 0; win = r + 1; if (cum >= P * tot) break; }
        if (win < Kt) win = Kt;
    }
    for (int r = 0; r < n; r++) rl[r] = (r >= J && r < win) ? (int8_t)lvl(ctx, rank[r]) : 0;
    int chosen = 0, pos[ROUTE_RANK_MAX]; uint8_t used[ROUTE_RANK_MAX] = {0};
    for (int r = 0; r < J && r < n && chosen < K; r++) { pos[chosen++] = r; used[r] = 1; }
    for (int level = 2; level >= 1; level--)
        for (int r = J; r < win && chosen < K; r++)
            if (!used[r] && rl[r] == level) { pos[chosen++] = r; used[r] = 1; }
    for (int r = 0; r < n && chosen < K; r++)
        if (!used[r]) { pos[chosen++] = r; used[r] = 1; }
    for (int kk = 0; kk < chosen; kk++) {
        int r = pos[kk]; idx[kk] = rank[r]; val[kk] = rw[r];
        if (r >= Kt && alpha > 0.f && alpha < 1.f) val[kk] *= alpha;
    }
    for (int kk = chosen; kk < K; kk++) { idx[kk] = -1; val[kk] = 0.f; }   /* fewer eligible than K: keep mask */
    if (!st) return;
    st->slots += (uint64_t)chosen; st->agree_tot += (uint64_t)chosen;
    float tsum = 1e-20f, csum = 1e-20f;
    for (int t = 0; t < Kt; t++) tsum += rw[t] > 0 ? rw[t] : 0;
    for (int kk = 0; kk < chosen; kk++) {
        csum += val[kk] > 0 ? val[kk] : 0;
        if (pos[kk] < Kt) st->agree_hit++;
        else { st->swaps++; if (rl[pos[kk]] == 2) st->swaps_vram++; }
    }
    double kl = 0;                      /* KL(true top-K mass || chosen mass), as the GLM meter */
    for (int t = 0; t < Kt; t++) {
        double pt = (rw[t] > 0 ? rw[t] : 0) / tsum; if (pt <= 0) continue;
        double pc = 1e-12;
        for (int kk = 0; kk < chosen; kk++) if (pos[kk] == t) { pc = (val[kk] > 0 ? val[kk] : 0) / csum; break; }
        kl += pt * log(pt / pc);
    }
    st->kl_sum += kl; st->kl_n++;
}

/* Residency levels for route_select: 2 = in the VRAM tier, 1 = in this
 * layer's RAM cache (pinned or LRU), 0 = would be read from disk. */
typedef struct { Model *m; int layer; } RouteCtx;
static int route_level(void *vctx, int e) {
    RouteCtx *rc = (RouteCtx *)vctx;
    if (qt_is_resident(rc->layer, e) || vkt_resident(rc->layer, e)) return 2;
    pthread_mutex_lock(&g_pilot_mx);
    Slot *s = slot_indexed(rc->m, rc->layer, e);
    pthread_mutex_unlock(&g_pilot_mx);
    return s ? 1 : 0;
}

static void route_footer(FILE *f, const Model *m) {
    if (g_cache_route && m->route.slots)
        fprintf(f, "CACHE_ROUTE J=%d M=%d P=%.2f alpha=%.2f | swap %.1f%% (%llu/%llu, %llu to VRAM)\n",
                g_route_j, g_route_m, g_route_p, g_route_alpha,
                100.0*m->route.swaps/m->route.slots, (unsigned long long)m->route.swaps,
                (unsigned long long)m->route.slots, (unsigned long long)m->route.swaps_vram);
    if (m->route.agree_tot)
        fprintf(f, "route_agree %.1f%% | route_kl %.4f\n", 100.0*m->route.agree_hit/m->route.agree_tot,
                m->route.kl_n ? m->route.kl_sum/(double)m->route.kl_n : 0.0);
}

/* ---- the Vulkan routed-expert tier (vk_tier.c) ---------------------------------
 * One MoE layer with the tier on, per block of rows: the block's resident experts go
 * to the device as one batch (vkt_issue) and the CPU computes the other (row, rank)
 * pairs into output rows of their own -- the shared kernel's in cache-sized cuts,
 * the int8 slots' one at a time -- and the shared expert into a buffer of its own
 * while the batch runs. Then every rank of every row joins `out` in rank order,
 * the device's and the CPU's alike, and the shared expert after them, as moe()
 * adds them; so the order of the sum never depends on which experts were resident.
 * Every expert the CPU computed passes its RAM bytes to the tier (vkt_note), which
 * may promote it. */
#define QWEN_VK_ROWS 64
static VktExpertSrc vk_slot_src(Model *m, const Slot *e) {
    if (e->pw) {
        int64_t gp = (int64_t)m->c.inter * m->c.hidden / 2;
        return (VktExpertSrc){e->pw, e->pw + gp, e->pw + 2 * gp, e->gs, e->us, e->ds};
    }
    return (VktExpertSrc){e->g, e->u, e->d, e->gs, e->us, e->ds};
}
/* The shared kernel's CPU pairs (want[i] set), into ctb[i]: moe_xf_run's cuts, the
 * held slots released after each. */
static void moe_vk_xf_cpu(Model *m, int layer, const float *x, int S, const int *idx,
                          const uint8_t *want, float *ctb) {
    Cfg *c = &m->c; int D = c->hidden, K = c->topk, F = c->inter;
    int cap = m->cache[layer].cap;
    int64_t gp = (int64_t)F * D / 2;
    int per = cap >= S * K ? S : 1, kper = cap >= K ? K : 1, n = per * kper;
    XfExpert *ex = malloc(sizeof(XfExpert) * (size_t)n);
    const XfExpert **exp = malloc(sizeof(XfExpert *) * (size_t)n);
    int *ridx = malloc(sizeof(int) * (size_t)n); Slot **held = malloc(sizeof(Slot *) * (size_t)n);
    float *lctb = falloc((int64_t)n * D);
    void *scratch = malloc(xf_moe_scratch_bytes(per, kper, D, F));
    if (!ex || !exp || !ridx || !held || !scratch) { fprintf(stderr, "OOM moe_vk_xf_cpu\n"); exit(1); }
    for (int s0 = 0; s0 < S; s0 += per)
        for (int k0 = 0; k0 < K; k0 += kper) {
            int any = 0;
            for (int s = 0; s < per; s++) for (int k = 0; k < kper; k++) {
                int src = (s0 + s) * K + (k0 + k), dst = s * kper + k;
                ridx[dst] = -1; exp[dst] = NULL; held[dst] = NULL;
                if (idx[src] < 0 || !want[src]) continue;
                Slot *e = held[dst] = expert_hold(m, layer, idx[src]);
                ex[dst] = (XfExpert){e->pw, e->pw + gp, e->pw + 2 * gp, e->gs, e->us, e->ds};
                exp[dst] = &ex[dst]; ridx[dst] = idx[src]; any = 1;
            }
            if (any) xf_moe_compute(lctb, x + (int64_t)s0 * D, per, kper, D, F, ridx, exp, xf_act_mode(), scratch);
            for (int s = 0; s < per; s++) for (int k = 0; k < kper; k++) {
                int src = (s0 + s) * K + (k0 + k), dst = s * kper + k;
                if (!held[dst]) continue;
                memcpy(ctb + (int64_t)src * D, lctb + (int64_t)dst * D, (size_t)D * sizeof(float));
                VktExpertSrc vs = vk_slot_src(m, held[dst]); vkt_note(layer, idx[src], &vs);
                slot_release(held[dst]);
            }
        }
    free(ex); free(exp); free(ridx); free(held); free(lctb); free(scratch);
}
/* The int8 slots' CPU pairs (want[i] set), into ctb[i], one at a time. */
static void moe_vk_i8_cpu(Model *m, int layer, const float *x, int S, const int *idx,
                          const uint8_t *want, float *ctb, float *g, float *u) {
    Cfg *c = &m->c; int D = c->hidden, K = c->topk, I = c->inter;
    for (int i = 0; i < S * K; i++) {
        if (idx[i] < 0 || !want[i]) continue;
        const float *xs = x + (int64_t)(i / K) * D;
        Slot *e = expert_hold(m, layer, idx[i]);   /* the PILOT worker may not evict it mid-matmul */
        slot_ensure_int8(m, e);
        matmul_qe(g, xs, e->g, e->gs, D, I);
        matmul_qe(u, xs, e->u, e->us, D, I);
        for (int j = 0; j < I; j++) { float gv = g[j]; g[j] = (gv / (1.f + expf(-gv))) * u[j]; }
        matmul_qd(ctb + (int64_t)i * D, g, e->d, e->ds, I, D);
        VktExpertSrc vs = vk_slot_src(m, e); vkt_note(layer, idx[i], &vs);
        slot_release(e);
    }
}
static void moe_vk_run(Model *m, Layer *l, int layer, const float *x, int S, float *out,
                       const int *idx, const float *val, int routed_only) {
    Cfg *c = &m->c; int D = c->hidden, K = c->topk, I = c->inter, SI = routed_only ? 0 : c->shared_inter;
    int xf = xf_mode(m), B = vkt_step_rows(S, QWEN_VK_ROWS);   /* a whole prompt chunk when the tier streams */
    float *ctb = falloc((int64_t)B * K * D), *shb = SI > 0 ? falloc((int64_t)B * D) : NULL;
    float *g = falloc(I > SI ? I : SI), *u = falloc(I > SI ? I : SI), *hh = falloc(D);
    uint8_t *taken = malloc((size_t)B * K), *want = malloc((size_t)B * K);
    const float **dev = malloc(sizeof(*dev) * (size_t)B * K);
    if (!taken || !want || !dev) { fprintf(stderr, "OOM moe_vk_run\n"); exit(1); }
    for (int s0 = 0; s0 < S; s0 += B) {
        int rows = S - s0 < B ? S - s0 : B, n = rows * K;
        const float *xb = x + (int64_t)s0 * D; const int *ib = idx + (int64_t)s0 * K;
        const float *vb = val + (int64_t)s0 * K;
        int ndev = vkt_issue(layer, xb, rows, K, ib, taken);
        for (int i = 0; i < n; i++) { want[i] = !taken[i]; if (taken[i]) ehit_mark(m, layer, ib[i]); }
        if (xf) moe_vk_xf_cpu(m, layer, xb, rows, ib, want, ctb);
        else    moe_vk_i8_cpu(m, layer, xb, rows, ib, want, ctb, g, u);
        if (shb) { memset(shb, 0, (size_t)rows * D * sizeof(float)); qwen_shared_experts_cpu(m, l, xb, rows, shb, g, u, hh); }
        const float *dsum = NULL;   /* a step wholly on the device: its routed rows summed there */
        if (ndev && !vkt_join_sum(dev, vb, &dsum)) {   /* the batch failed (the tier stops): those pairs here */
            if (xf) moe_vk_xf_cpu(m, layer, xb, rows, ib, taken, ctb);
            else    moe_vk_i8_cpu(m, layer, xb, rows, ib, taken, ctb, g, u);
            memset(taken, 0, (size_t)n);
        }
        /* rows apart (a prompt step reads tens of MB here); each row in rank order */
        #pragma omp parallel for schedule(static) if (rows >= 64)
        for (int s = 0; s < rows; s++) {
            float *os = out + (int64_t)(s0 + s) * D;
            if (dsum) {   /* the device's sum is this loop's for a zeroed row */
                const float *ds = dsum + (int64_t)s * D;
                for (int d = 0; d < D; d++) os[d] += ds[d];
                if (shb) { const float *sp = shb + (int64_t)s * D; for (int d = 0; d < D; d++) os[d] += sp[d]; }
                continue;
            }
            for (int k = 0; k < K; k++) {
                int i = s * K + k;
                if (ib[i] < 0) continue;
                const float *cp = taken[i] ? dev[i] : ctb + (int64_t)i * D;
                float w = vb[i];
                for (int d = 0; d < D; d++) os[d] += w * cp[d];
            }
            if (shb) { const float *sp = shb + (int64_t)s * D; for (int d = 0; d < D; d++) os[d] += sp[d]; }
        }
    }
    free(ctb); free(shb); free(g); free(u); free(hh); free(taken); free(want); free(dev);
}

/* moe_ex: logits_in (the router's raw rows, S x E) when the caller already has them,
 * else NULL and the router runs here; routed_only = 1 leaves the shared expert out,
 * so `out` gets the routed experts' rank-order sum alone (the Vulkan dense chain runs
 * the router and the shared expert on the device and adds them there). moe() is
 * moe_ex(..., NULL, 0): the same operations in the same order as before. */
static void moe_ex(Model *m, Layer *l, int layer, float *x, int S, float *out,
                   const float *logits_in, int routed_only) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    if (E == 0) {
        memset(out, 0, (int64_t)S*D*sizeof(float));
        if (routed_only) return;
        float *g = falloc(c->shared_inter), *u = falloc(c->shared_inter), *hh = falloc(D);
        qwen_shared_experts_cpu(m, l, x, S, out, g, u, hh);
        free(g); free(u); free(hh);
        (void)layer;
        return;
    }
    float *logits = falloc((int64_t)S*E);
    double _tr = tm_now();
    if (logits_in) memcpy(logits, logits_in, (size_t)S*E*sizeof(float));
    else matmul_d(logits, x, &l->gate, S, D, E);
    tm_add(S, 4, tm_now()-_tr);
    if (c->has_bias && l->gate_bias) {
        for (int s = 0; s < S; s++) { float *pr = logits + (int64_t)s*E; for (int e = 0; e < E; e++) pr[e] += l->gate_bias[e]; }
    }
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    float *sh = falloc(I), *shu = falloc(I), *shd = falloc(D);  /* shared expert scratch */
    /* When a qpack container owns the routed experts, they run through
     * the MLX-affine path (Metal on Apple, CPU reference elsewhere);
     * neither the CUDA tier nor the expert_ffn kernel is consulted.
     * Routing, shared expert, and the combine stay on the CPU below. */
    int use_qq = qq_active();
    int use_qt = !use_qq && qt_ready();
    /* the Vulkan tier takes the routed experts of every format it was given
     * (moe_vk_run): the routing is collected first, as for the shared kernel */
    int use_vk = !use_qq && !use_qt && vkt_ready();
    int use_xf = !use_qq && !use_qt && !use_vk && xf_mode(m);
    /* prefill re-plan: this layer's routing counts over the prompt rows */
    uint32_t *rp_cnt = (use_qt && S > 1 && prefill_replan_on()) ? calloc((size_t)E, sizeof(uint32_t)) : NULL;
    int *xidx = use_xf || use_vk ? malloc(sizeof(int) * (size_t)S * K) : NULL;
    float *xval = use_xf || use_vk ? falloc((int64_t)S * K) : NULL;
    /* A prompt's routing in parallel (the shared kernel's and the Vulkan tier's, which
     * collect the routing first): softmax, group-limited top-k and renormalisation per
     * token are independent, each exactly as the loop below; the momentum logits cross
     * tokens but never feed the routing, so they go first in token order, and the
     * bookkeeping (collected experts, frequencies, the tier's history) after, in token
     * order. Not with cache routing or the agreement meter, whose state crosses tokens.
     * 1011 tokens of Qwen3.6: about 4 ms a layer on one core. */
    int par_route = (use_xf || use_vk) && S >= 64 && !g_cache_route && !g_route_agree && E <= 1024 && K <= 256 &&
                    !(c->n_group > 1 && c->n_group > E);
    if (par_route) {
        if (m->momentum_logits && m->pilot_smooth > 0.f) {
            float *ema = m->momentum_logits + (int64_t)layer * E;
            for (int s = 0; s < S; s++) {
                const float *pr = logits + (int64_t)s*E;
                int is_zero = 1; for (int e = 0; e < E; e++) if (ema[e] != 0.f) { is_zero = 0; break; }
                if (is_zero) { for (int e = 0; e < E; e++) ema[e] = pr[e]; }
                else { for (int e = 0; e < E; e++) ema[e] = (1.f - m->pilot_smooth)*pr[e] + m->pilot_smooth*ema[e]; }
            }
        }
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < S; s++) {
            float *pr = logits + (int64_t)s*E;
            softmax_row(pr, E);
            uint8_t keep[1024]; int Ec = E;
            if (c->n_group > 1 && c->n_group <= Ec) {
                int per = E / c->n_group;
                float gs[1024];
                for (int gi = 0; gi < c->n_group; gi++) {
                    float b1 = -1e30f, b2 = -1e30f;
                    for (int e = gi*per; e < gi*per+per; e++) { float v = pr[e]; if (v > b1) { b2=b1; b1=v; } else if (v > b2) b2=v; }
                    gs[gi] = b1 + b2;
                }
                uint8_t gkeep[1024] = {0};
                for (int kk = 0; kk < c->topk_group; kk++) {
                    int bg = -1; float bv = -1e30f;
                    for (int gi = 0; gi < c->n_group; gi++) { if (!gkeep[gi] && gs[gi] > bv) { bv = gs[gi]; bg = gi; } }
                    if (bg < 0) break; gkeep[bg] = 1;
                }
                for (int e = 0; e < Ec; e++) keep[e] = 0;
                for (int gi = 0; gi < c->n_group; gi++) if (gkeep[gi]) for (int e = gi*per; e < gi*per+per; e++) keep[e] = 1;
            } else {
                for (int e = 0; e < Ec; e++) keep[e] = 1;
            }
            int *idx = xidx + (int64_t)s*K; float *val = xval + (int64_t)s*K;
            for (int kk = 0; kk < K; kk++) {
                int best = -1; float bv = -1e30f;
                for (int e = 0; e < E; e++) {
                    if (!keep[e]) continue;
                    int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
                    if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
                }
                idx[kk] = best; val[kk] = bv;
            }
            for (int kk = 0; kk < K; kk++) if (idx[kk] < 0) { idx[kk] = -1 - kk; val[kk] = 0.f; }   /* degraded: below */
            float sm=0; for (int kk=0;kk<K;kk++) sm+=val[kk]; if (sm>0) for (int kk=0;kk<K;kk++) val[kk]/=sm;
        }
        for (int s = 0; s < S; s++) {
            int *idx = xidx + (int64_t)s*K;
            for (int kk = 0; kk < K; kk++) {   /* the serial loop's degradation, its warning once */
                if (idx[kk] >= 0) continue;
                static int warned_par;
                if (!warned_par) {
                    warned_par = 1;
                    fprintf(stderr, "[router] non-finite logits at layer %d, or fewer than top-k "
                                    "experts eligible: selection degraded\n", layer);
                }
                idx[kk] = kk;
            }
            if (m->resident_collecting)
                for (int kk = 0; kk < K; kk++) m->seen[(int64_t)layer * E + idx[kk]] = 1;
            if (!m->hot_pinned && m->freq) {
                uint32_t *freq_l = m->freq + (int64_t)layer * E;
                for (int kk = 0; kk < K; kk++) freq_l[idx[kk]]++;
            }
            if (use_vk && m->vk_hist) rt_count(layer, idx, K);
        }
    }
    for (int s = par_route ? S : 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        if (m->momentum_logits && m->pilot_smooth > 0.f) {
            float *ema = m->momentum_logits + (int64_t)layer * E;
            int is_zero = 1; for (int e = 0; e < E; e++) if (ema[e] != 0.f) { is_zero = 0; break; }
            if (is_zero) { for (int e = 0; e < E; e++) ema[e] = pr[e]; }
            else { for (int e = 0; e < E; e++) ema[e] = (1.f - m->pilot_smooth)*pr[e] + m->pilot_smooth*ema[e]; }
        }
        softmax_row(pr, E);
        /* group-limited top-k selection */
        uint8_t keep[1024]; int Ec = E < 1024 ? E : 1024;
        if (c->n_group > 1 && c->n_group <= Ec) {
            int per = E / c->n_group;
            float gs[1024];
            for (int gi = 0; gi < c->n_group; gi++) {
                float b1 = -1e30f, b2 = -1e30f;
                for (int e = gi*per; e < gi*per+per; e++) { float v = pr[e]; if (v > b1) { b2=b1; b1=v; } else if (v > b2) b2=v; }
                gs[gi] = b1 + b2;
            }
            uint8_t gkeep[1024] = {0};
            for (int kk = 0; kk < c->topk_group; kk++) {
                int bg = -1; float bv = -1e30f;
                for (int gi = 0; gi < c->n_group; gi++) { if (!gkeep[gi] && gs[gi] > bv) { bv = gs[gi]; bg = gi; } }
                if (bg < 0) break; gkeep[bg] = 1;
            }
            for (int e = 0; e < Ec; e++) keep[e] = 0;
            for (int gi = 0; gi < c->n_group; gi++) if (gkeep[gi]) for (int e = gi*per; e < gi*per+per; e++) keep[e] = 1;
        } else {
            for (int e = 0; e < Ec; e++) keep[e] = 1;
        }
        int idx[256]; float val[256];
        if (g_cache_route) {
            RouteCtx rc = { m, layer };
            route_select(pr, keep, E, K, g_route_j, g_route_m, g_route_p, g_route_alpha,
                         route_level, &rc, idx, val, &m->route);
        } else {
            for (int kk = 0; kk < K; kk++) {
                int best = -1; float bv = -1e30f;
                for (int e = 0; e < E; e++) {
                    if (!keep[e]) continue;
                    int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
                    if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
                }
                idx[kk] = best; val[kk] = bv;
            }
            if (g_route_agree) {            /* plain routing: full agreement by construction */
                m->route.agree_hit += (uint64_t)K; m->route.agree_tot += (uint64_t)K; m->route.kl_n++;
            }
        }
        /* SEC: an all-NaN router row (a corrupt tile, an fp overflow) leaves best at
         * -1 above -- NaN > bv is false for every expert -- and route_select pads
         * with -1 when fewer than K experts rank. Every consumer below takes the id
         * as an index and a file offset: expert_get() went looking for experts.-1.
         * Same degradation as rt_router_pick in route_trace.h, which this engine
         * does not include: the slot's own index, in range because topk <=
         * n_experts is a config check, at weight 0 so the slot adds nothing. */
        for (int kk = 0; kk < K; kk++) {
            if (idx[kk] >= 0) continue;
            static int warned;
            if (!warned) {
                warned = 1;
                fprintf(stderr, "[router] non-finite logits at layer %d, or fewer than top-k "
                                "experts eligible: selection degraded\n", layer);
            }
            idx[kk] = kk; val[kk] = 0.f;
        }
        if (m->resident_collecting) {
            for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) m->seen[(int64_t)layer * E + idx[kk]] = 1;
        }
        /* HF renormalizes the top-k router weights unconditionally */
        { float sm=0; for (int kk=0;kk<K;kk++) sm+=val[kk]; if (sm>0) for (int kk=0;kk<K;kk++) val[kk]/=sm; }
        if (!m->hot_pinned && m->freq) {
            uint32_t *freq_l = m->freq + (int64_t)layer * E;
            for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) freq_l[idx[kk]]++;
        }
        if (rp_cnt) for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) rp_cnt[idx[kk]]++;
        const float *xs = x + (int64_t)s*D;
        if (use_qq) {
            /* A failed expert is fatal: once the container owns the routed
             * experts there is no other weight source to fall back to, and a
             * silently skipped expert would just be a wrong answer. */
            for (int kk = 0; kk < K; kk++) {
                if (idx[kk] < 0) continue;
                if (!qq_expert_forward(layer, idx[kk], xs, val[kk],
                                       out + (int64_t)s*D)) {
                    fprintf(stderr, "qpack expert layer %d expert %d failed"
                            " -- refusing\n", layer, idx[kk]);
                    exit(1);
                }
            }
        } else if (use_xf || use_vk) {
            for (int kk = 0; kk < K; kk++) { xidx[s*K+kk] = idx[kk]; xval[s*K+kk] = val[kk]; }
            if (use_vk && m->vk_hist) rt_count(layer, idx, K);   /* the tier's history (.coli_usage) */
        } else if (use_qt) {
            /* CUDA expert tier: run the resident experts as async groups on
             * all devices, compute the misses on the CPU (overlapped), then
             * collect the GPU results. */
            for (int kk = 0; kk < K; kk++) {
                Slot *e; expert_get(m, layer, idx[kk], &e);
                /* Offer whichever format the container actually packed. The old
                 * gate `if (e->g4)` never fired on an int8 container (g4 is
                 * NULL there), so the tier stayed at 0 uploads for the life of
                 * the process (#1391). tier_offer_slot is the shared decision:
                 * same pointer choice tier_warmstart makes, one place. */
                tier_offer_slot(layer, idx[kk], e);
            }
            double _q0 = tm_now();
            uint32_t qmask = qt_issue(layer, idx, K, xs);
            double _q1 = tm_now();
            for (int kk = 0; kk < K; kk++) {
                if (qmask & (1u<<kk)) continue;
                Slot *e = expert_hold(m, layer, idx[kk]);
                slot_ensure_int8(m, e);
                matmul_qe(g, xs, e->g, e->gs, D, I);
                matmul_qe(u, xs, e->u, e->us, D, I);
                for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                matmul_qd(hh, g, e->d, e->ds, I, D);
                slot_release(e);
                float w = val[kk]; float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += w * hh[d];
            }
            /* Compute the shared expert NOW so it overlaps with the GPU
             * groups; the common block below is skipped. */
            if (c->shared_inter > 0) {
                double _ts2 = tm_now();
                int Ish = c->shared_inter;
                if (!qtd(l->qth_shg, sh, xs, D, Ish))  matmul_d(sh, xs, &l->sh_g, 1, D, Ish);
                if (!qtd(l->qth_shu, shu, xs, D, Ish)) matmul_d(shu, xs, &l->sh_u, 1, D, Ish);
                for (int i = 0; i < Ish; i++) { float sv = sh[i]; sh[i] = (sv / (1.f + expf(-sv))) * shu[i]; }
                if (!qtd(l->qth_shd, shd, sh, Ish, D)) matmul_d(shd, sh, &l->sh_d, 1, Ish, D);
                float sgate = 1.f;
                if (l->sh_gate) {
                    float sg = 0.f; const float *wg = l->sh_gate;
                    for (int i = 0; i < D; i++) sg += xs[i] * wg[i];
                    sgate = 1.f / (1.f + expf(-sg));
                }
                float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += sgate * shd[d];
                tm_add(S, 3, tm_now()-_ts2);
            }
            double _q2 = tm_now();
            if(!qt_take(qmask, val, K, out + (int64_t)s*D)){
                fprintf(stderr,"qwen36: CUDA expert collection failed at layer %d; stopping inference\n",layer);
                exit(1);
            }
            if (tm_on() && S==1) {
                extern double g_qt_iss, g_qt_cpu, g_qt_tak;
                g_qt_iss += _q1-_q0; g_qt_cpu += _q2-_q1; g_qt_tak += tm_now()-_q2;
            }
        } else {
            for (int kk = 0; kk < K; kk++) {
                Slot *e = expert_hold(m, layer, idx[kk]);   /* the PILOT worker may not evict it mid-matmul */
                slot_ensure_int8(m, e);
                matmul_qe(g, xs, e->g, e->gs, D, I);
                matmul_qe(u, xs, e->u, e->us, D, I);
                for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                matmul_qd(hh, g, e->d, e->ds, I, D);
                slot_release(e);
                float w = val[kk];
                float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += w * hh[d];
            }
        }
    }
    if (rp_cnt) { qt_replan(layer, rp_cnt, prefill_replan_cap()); free(rp_cnt); }   /* this layer's swaps upload while the next layers compute */
    if (use_xf) { moe_xf_run(m, layer, x, S, out, xidx, xval); free(xidx); free(xval); }
    if (use_vk) { moe_vk_run(m, l, layer, x, S, out, xidx, xval, routed_only); free(xidx); free(xval); }
    /* The CUDA tier keeps its per-token shared block above because it overlaps
     * resident GPU experts, and the Vulkan tier computes it inside moe_vk_run while
     * its batch runs.  CPU prefill instead traverses each shared matrix once per
     * bounded chunk. */
    if (!use_qt && !use_vk && !routed_only) qwen_shared_experts_cpu(m,l,x,S,out,sh,shu,shd);
    free(logits); free(g); free(u); free(hh); free(sh); free(shu); free(shd);
}
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    moe_ex(m, l, layer, x, S, out, NULL, 0);
}

/* Gated DeltaNet (linear_attention) forward — recurrent gated-delta-rule.
 * Mirrors HF Qwen3_5MoeGatedDeltaNet with a carried causal-conv ring + recurrent
 * state S[h]=[kdim,vdim]. The conv ring and S persist in m->DN_conv/rec[layer]
 * across step() calls (prefill chunk -> decode tokens). Math validated
 * torch-free against the prefill (zero-padded conv) path in tools/_ref_dn_stream.py.
 *
 * Per token: qkv=x@qkv^T; z=x@z^T; b=x@b^T; a=x@a^T; beta=sigmoid(b);
 *   g=-exp(A_log)*softplus(a+dt_bias);
 *   conv_out[c]=silu(sum_{kk} w[kk]*ring[kk] + w[convk-1]*qkv[c]); advance ring;
 *   split conv_out -> q_in/k_in/v_in; repeat_interleave q,k by rep; l2norm
 *   (q scaled by 1/sqrt(kdim)); recurrence S[h]*=exp(g); kv=k@S; delta=(v-kv)*beta;
 *   S+=k (x) delta; out=q@S; per-head Gated RMSNorm (plain weight) -> out_proj. */
/* Bound both the host result block and device input/output staging. */
static int dnproj_batch_rows(int S, int H, int O) {
    int64_t rows = (32LL << 20) / (((int64_t)H + O) * sizeof(float));
    if (rows < 1) rows = 1;
    if (rows > 256) rows = 256;
    return S < rows ? S : (int)rows;
}

/* Host/device state hand-over for a DeltaNet layer that runs on the GPU. The
 * host arrays are canonical: the device copy is a cache that is fresh (holds
 * what the host holds) or ahead (host_stale: the device advanced). */
static void dn_gpu_push(Model *m, int layer) {
    if (!m->dn_dev_fresh[layer]) {
        if (qt_dn_gpu_set_state(layer, m->DN_conv[layer], m->DN_rec[layer])) m->dn_dev_fresh[layer] = 1;
    }
}
static void dn_gpu_pull(Model *m, int layer) {
    if (m->dn_host_stale && m->dn_host_stale[layer]) {
        if (qt_dn_gpu_get_state(layer, m->DN_conv[layer], m->DN_rec[layer])) m->dn_host_stale[layer] = 0;
        else fprintf(stderr, "[dn] layer %d: could not read the GPU state back; the CPU continues from a stale copy\n", layer);
    }
}
static void dn_gpu_pull_all(Model *m) {
    if (!m->dn_dev) return;
    for (int i = 0; i < m->c.n_layers; i++) if (!m->c.is_attn[i]) dn_gpu_pull(m, i);
}
/* the host state was rewritten (reset, restore): the device copy is old */
static void dn_gpu_invalidate(Model *m) {
    if (!m->dn_dev_fresh) return;
    memset(m->dn_dev_fresh, 0, (size_t)m->c.n_layers);
    memset(m->dn_host_stale, 0, (size_t)m->c.n_layers);
}
static int dn_gpu_env_on(void) {
    static int on = -1;
    if (on < 0) { const char *p = getenv("Q36_DN_GPU"); on = p && *p == '1'; }
    return on;
}

static void deltanet(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    (void)pos_base;
    Cfg *c = &m->c;
    int vh = c->dn_vheads, vk = c->dn_kheads, kdim = c->dn_kdim, vdim = c->dn_vdim;
    int convk = c->dn_convk, conv_dim = c->dn_conv_dim;
    int rep = vh / vk;
    int key_dim_tot = vk * kdim;
    int value_dim = vh * vdim;
    float scale = 1.f / sqrtf((float)kdim);
    int H = c->hidden;

    /* The projections have no recurrent dependency: the input ones (qkv, z, b,
     * a) run once for a bounded block of rows, the conv and the recurrence then
     * consume the block in order, and the out_proj runs once over the block's
     * normed outputs. Every kernel computes each row on its own, so a block
     * gives the bits of one call per token; what it changes is the device
     * path, which sees one S-row call (a GEMM) instead of S one-row GEMVs.
     * The CUDA tier projects qkv ++ z into one interleaved block (qkvz). */
    int proj_dim = conv_dim + value_dim;
    int B = dnproj_batch_rows(S, H, proj_dim);
    float *qkvz = falloc((int64_t)B * proj_dim);
    float *qkvb = falloc((int64_t)B * conv_dim);
    float *zb   = falloc((int64_t)B * value_dim);
    float *bb   = falloc((int64_t)B * vh);
    float *ab   = falloc((int64_t)B * vh);
    float *outrb = falloc((int64_t)B * value_dim);
    float *beta= falloc(vh);
    float *gg  = falloc(vh);
    float *conv_out = falloc(conv_dim);
    float *q = falloc(vh * kdim);
    float *k = falloc(vh * kdim);
    float *outv = falloc(value_dim);
    float *kv = falloc(vdim);
    float *delta = falloc(vdim);

    /* [vh*kdim*vdim] and [conv_dim*(convk-1)]; a multiplexed step parks every
     * conversation and takes each row's below */
    float *rec = m->DN_rec ? m->DN_rec[layer] : NULL;
    float *ring = m->DN_conv ? m->DN_conv[layer] : NULL;
    FILE *dbg = layer == 0 && getenv("DN_DBG") ? fopen(getenv("DN_DBG"), "wb") : NULL;

    /* Decode token with the layer on the GPU: gates on the CPU (two tiny
     * matmuls), everything else -- in_proj, conv, recurrence, gated norm,
     * out_proj -- in one device chain, host in, host out. The state stays on
     * the card; the host copy is refreshed only when something on the CPU
     * asks for it (dn_gpu_pull). */
    if (S == 1 && m->dn_dev && qt_dn_gpu_ready(layer)) {
        extern double g_dn_sub[4];
        double _g0 = tm_now();
        matmul(bb, x, l->dn_b, 1, H, vh);
        matmul(ab, x, l->dn_a, 1, H, vh);
        for (int h = 0; h < vh; h++) {
            beta[h] = 1.f / (1.f + expf(-bb[h]));
            gg[h] = expf(-expf(l->dn_alog[h]) * softplus_f(ab[h] + l->dn_dtbias[h]));   /* egh */
        }
        dn_gpu_push(m, layer);
        int ok = m->dn_dev_fresh[layer] && qt_dn_gpu_step(layer, x, out, gg, beta);
        if (ok) {
            m->dn_host_stale[layer] = 1;
            if (tm_on()) g_dn_sub[0] += tm_now() - _g0;
            if (dbg) fclose(dbg);
            free(qkvz); free(qkvb); free(zb); free(bb); free(ab); free(outrb);
            free(beta); free(gg);
            free(conv_out); free(q); free(k); free(outv); free(kv); free(delta);
            return;
        }
        /* the tier turned the layer off: continue on the CPU from the state
         * the card still holds (a failed step may or may not have advanced it) */
        dn_gpu_pull(m, layer);
    } else if (m->dn_dev) {
        dn_gpu_pull(m, layer);              /* prefill or a CPU-only layer: the host must be current */
    }

    for (int base = 0; base < S; base += B) {
        int rows = S - base < B ? S - base : B;
        const float *xb = x + (int64_t)base * H;
        extern double g_dn_sub[4];
        double _d0 = tm_now();
        int gpu_block = qt_dnproj_ready(layer) && qt_dnproj_matmul_batch(layer, qkvz, xb, rows, H, proj_dim);
        if (!gpu_block) {
            matmul_d(qkvb, xb, &l->dn_qkv, rows, H, conv_dim);
            matmul_d(zb,   xb, &l->dn_z,   rows, H, value_dim);
        }
        matmul(bb, xb, l->dn_b, rows, H, vh);
        matmul(ab, xb, l->dn_a, rows, H, vh);
        if (tm_on() && S==1){ double t=tm_now(); g_dn_sub[0]+=t-_d0; _d0=t; }
        for (int r = 0; r < rows; r++) {
        if (m->mux_rows) { rec = m->mux_rows[base + r].seq->DN_rec[layer]; ring = m->mux_rows[base + r].seq->DN_conv[layer]; }
        const float *qkv = gpu_block ? qkvz + (int64_t)r * proj_dim : qkvb + (int64_t)r * conv_dim;
        const float *z = gpu_block ? qkv + conv_dim : zb + (int64_t)r * value_dim;
        const float *b = bb + (int64_t)r * vh, *a = ab + (int64_t)r * vh;
        float *outr = outrb + (int64_t)r * value_dim;
        for (int h = 0; h < vh; h++) {
            beta[h] = 1.f / (1.f + expf(-b[h]));
            gg[h] = -expf(l->dn_alog[h]) * softplus_f(a[h] + l->dn_dtbias[h]);
        }
        /* causal depthwise conv1d (groups=conv_dim, kernel=convk) with carried ring
         * (serial: ~33k FLOP, an OpenMP fork/join would cost more) */
        for (int cc = 0; cc < conv_dim; cc++) {
            const float *w = l->dn_conv + (int64_t)cc * convk;
            const float *rg = ring + (int64_t)cc * (convk - 1);
            float acc = 0.f;
            for (int kk = 0; kk < convk - 1; kk++) acc += w[kk] * rg[kk];
            acc += w[convk - 1] * qkv[cc];
            conv_out[cc] = acc / (1.f + expf(-acc));   /* silu */
        }
        /* advance ring: drop oldest, append current token's qkv */
        for (int cc = 0; cc < conv_dim; cc++) {
            float *rg = ring + (int64_t)cc * (convk - 1);
            for (int kk = 0; kk < convk - 2; kk++) rg[kk] = rg[kk + 1];
            rg[convk - 2] = qkv[cc];
        }
        if (tm_on() && S==1){ double t=tm_now(); g_dn_sub[1]+=t-_d0; _d0=t; }
        /* split into query/key (key_dim_tot each) + value (value_dim) */
        const float *q_in = conv_out;
        const float *k_in = conv_out + key_dim_tot;
        const float *v_in = conv_out + 2 * key_dim_tot;
        /* repeat_interleave q/k by rep along head dim (vk heads -> vh heads).
         * HF semantics (torch repeat_interleave): each key head is repeated
         * `rep` consecutive times, so VALUE head h takes KEY head (h / rep).
         * This is NOT h % vk. Verified against _ref_dn.py L245-247. */
        for (int h = 0; h < vh; h++) {
            int vk_idx = h / rep;
            memcpy(q + (int64_t)h * kdim, q_in + (int64_t)vk_idx * kdim, kdim * sizeof(float));
            memcpy(k + (int64_t)h * kdim, k_in + (int64_t)vk_idx * kdim, kdim * sizeof(float));
        }
        /* per-head l2norm (+ scale q by 1/sqrt(kdim)); eps 1e-6 inside sqrt (HF default) */
        for (int oh = 0; oh < vh; oh++) {
            float *qh = q + (int64_t)oh * kdim;
            double sq = 1e-6; for (int d = 0; d < kdim; d++) sq += (double)qh[d] * qh[d];
            double nq = sqrt(sq);
            for (int d = 0; d < kdim; d++) qh[d] = (float)((double)qh[d] / nq * scale);
            float *kh = k + (int64_t)oh * kdim;
            double sk = 1e-6; for (int d = 0; d < kdim; d++) sk += (double)kh[d] * kh[d];
            double nk = sqrt(sk);
            for (int d = 0; d < kdim; d++) kh[d] = (float)((double)kh[d] / nk);
        }
        /* recurrent gated delta rule over the value heads (heads are
         * independent -> parallel; kv/delta thread-local) */
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < vh; h++) {
            float kvl[512], dl[512];   /* vdim <= 512 */
            float *Sh = rec + (int64_t)h * kdim * vdim;
            float egh = expf(gg[h]);
            for (int t = 0; t < kdim * vdim; t++) Sh[t] *= egh;
            const float *kd = k + (int64_t)h * kdim;
            const float *vd = v_in + (int64_t)h * vdim;
            /* kv = kd @ Sh  (length vdim) */
            for (int vv = 0; vv < vdim; vv++) kvl[vv] = 0.f;
            for (int kk = 0; kk < kdim; kk++) {
                float kkd = kd[kk]; const float *Sr = Sh + (int64_t)kk * vdim;
                for (int vv = 0; vv < vdim; vv++) kvl[vv] += kkd * Sr[vv];
            }
            /* delta = (v - kv) * beta */
            for (int vv = 0; vv < vdim; vv++) dl[vv] = (vd[vv] - kvl[vv]) * beta[h];
            /* Sh += outer(kd, delta) */
            for (int kk = 0; kk < kdim; kk++) {
                float kkd = kd[kk]; float *Sr = Sh + (int64_t)kk * vdim;
                for (int vv = 0; vv < vdim; vv++) Sr[vv] += kkd * dl[vv];
            }
            /* out = qd @ Sh */
            const float *qd = q + (int64_t)h * kdim;
            float *ov = outv + (int64_t)h * vdim;
            for (int vv = 0; vv < vdim; vv++) ov[vv] = 0.f;
            for (int kk = 0; kk < kdim; kk++) {
                float qkd = qd[kk]; const float *Sr = Sh + (int64_t)kk * vdim;
                for (int vv = 0; vv < vdim; vv++) ov[vv] += qkd * Sr[vv];
            }
        }
        if (tm_on() && S==1){ double t=tm_now(); g_dn_sub[2]+=t-_d0; _d0=t; }
        /* a verify: the state this row leaves, for the rollback of a draft rejected
         * after it (q36_spec_rollback) */
        if (base + r < m->snap_rows) {
            memcpy(m->snap_rec[base + r][layer], rec, (size_t)vh * kdim * vdim * sizeof(float));
            memcpy(m->snap_conv[base + r][layer], ring, (size_t)conv_dim * (convk - 1) * sizeof(float));
        }
        /* per-head Gated RMSNorm (plain weight, r=1/sqrt(mean+eps)) then silu(z) gate; the
         * out_proj runs over the block below.
         * HF Qwen3_5MoeRMSNormGated: out = (o*r)*weight * silu(z) = (o*r)*weight * z/(1+e^-z).
         * NB: it is silu (z in numerator), NOT sigmoid. */
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < vh; h++) {
            const float *o = outv + (int64_t)h * vdim;
            const float *zr = z + (int64_t)h * vdim;
            const float *w = l->dn_norm;
            double ms = 0; for (int d = 0; d < vdim; d++) ms += (double)o[d] * o[d];
            float r = 1.f / sqrtf((float)(ms / vdim) + c->eps);
            for (int d = 0; d < vdim; d++) {
                float val = o[d] * r * w[d];
                outr[(int64_t)h * vdim + d] = val * zr[d] / (1.f + expf(-zr[d]));
            }
        }
        if (dbg && base + r == 0) {   /* DN_DBG: the first token of layer 0 */
            fwrite(conv_out, sizeof(float), conv_dim, dbg);
            fwrite(q, sizeof(float), (int64_t)vh * kdim, dbg);
            fwrite(outv, sizeof(float), value_dim, dbg);
            fwrite(z, sizeof(float), value_dim, dbg);
            fwrite(outr, sizeof(float), value_dim, dbg);
        }
        }
        float *ob = out + (int64_t)base * H;
        if (rows == 1 ? !qtd(l->qth_dnout, ob, outrb, value_dim, H)
                      : !qtd_batch(l->qth_dnout, ob, outrb, rows, value_dim, H))
            matmul_d(ob, outrb, &l->dn_out, rows, value_dim, H);
        if (tm_on() && S==1){ g_dn_sub[3]+=tm_now()-_d0; }
        if (dbg && base == 0) {       /* ... its out_proj row and gates, then done */
            fwrite(ob, sizeof(float), H, dbg);
            fwrite(bb, sizeof(float), vh, dbg);
            fwrite(ab, sizeof(float), vh, dbg);
            for (int h = 0; h < vh; h++) beta[h] = 1.f / (1.f + expf(-bb[h]));
            fwrite(beta, sizeof(float), vh, dbg);
            for (int h = 0; h < vh; h++) gg[h] = -expf(l->dn_alog[h]) * softplus_f(ab[h] + l->dn_dtbias[h]);
            fwrite(gg, sizeof(float), vh, dbg);
            fclose(dbg); dbg = NULL;
        }
    }
    if (m->dn_dev_fresh) m->dn_dev_fresh[layer] = 0;   /* the host advanced: the device copy is old */
    free(qkvz); free(qkvb); free(zb); free(bb); free(ab); free(outrb);
    free(beta); free(gg);
    free(conv_out); free(q); free(k); free(outv); free(kv); free(delta);
}

/* The rest of the dense trunk, offered to the placer by name and layer with
 * the bytes of the dense-i8 copies (docs/qwen36-cuda-tier.md, "Placement"):
 *   dnout    -- the DeltaNet out_proj, one matrix per DeltaNet layer
 *   attnproj -- q, k, v, o of every attention layer, offered as one item
 *   shexp    -- gate, up, down of the shared expert, every layer
 * Measured on ds (CPU, 8 threads): of the 37.5 ms a DeltaNet layer stack
 * costs per decoded token, 23.4 are the input projections (already placeable
 * as "dnproj"), 8.3 the out_proj and norm, 3.3 the convolution and 2.4 the
 * recurrence -- the matmuls are the cost, not the recurrence, so this is
 * where the trunk goes. Offer order after lmhead and dnproj: dnout, attnproj,
 * shexp, each in layer order, so a partial placement is whole layers. A
 * component is offered only when every matrix of it has a dense-i8 copy. */
static void trunk_offer_dense(Model *m){
    Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) continue;
        size_t b = qdw_bytes(&m->L[i].dn_out);
        if (b) qt_trunk_offer("dnout", i, b);
    }
    for (int i = 0; i < c->n_layers; i++) {
        if (!c->is_attn[i]) continue;
        Layer *l = &m->L[i];
        size_t bq = qdw_bytes(&l->q), bk = qdw_bytes(&l->k), bv = qdw_bytes(&l->v), bo = qdw_bytes(&l->o);
        if (bq && bk && bv && bo) qt_trunk_offer("attnproj", i, bq + bk + bv + bo);
    }
    /* The shared expert is offered only on request (Q36_OFFER_SHEXP=1). On the
     * card it is 120 synchronous small GEMVs per token that sit between
     * qt_issue and qt_take, where on the CPU it hides behind the expert group:
     * measured on the 35B, 3070, shared 3.7-4.0 ms/token on the CPU against
     * 6.3 on the same card and 11-12.7 with layers on a slower second card
     * (docs/qwen36-cuda-tier.md). A hand-written COLI_PLACE naming shexp is
     * still obeyed when the offer is made. */
    { const char *so = getenv("Q36_OFFER_SHEXP");
      if (so && *so == '1')
        for (int i = 0; i < c->n_layers; i++) {
            Layer *l = &m->L[i];
            size_t bg = qdw_bytes(&l->sh_g), bu = qdw_bytes(&l->sh_u), bd = qdw_bytes(&l->sh_d);
            if (bg && bu && bd) qt_trunk_offer("shexp", i, bg + bu + bd);
        } }
}
/* After qt_init decided: upload what was placed, keep the handles in the
 * Layer. Every matrix falls back on its own, so a failed upload costs one
 * GEMV on the CPU, never the component. Returns the number of matrices
 * placed; `vram_bytes` gets their size. */
static int trunk_place_dense(Model *m, double *vram_bytes){
    Cfg *c = &m->c; int placed = 0; double vram = 0;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (!c->is_attn[i]) {
            l->qth_dnout = qdw_place(&l->dn_out, qt_place_of("dnout", i));
            if (l->qth_dnout) { placed++; vram += (double)qdw_bytes(&l->dn_out); }
        } else {
            int dev = qt_place_of("attnproj", i);
            l->qth_q = qdw_place(&l->q, dev); l->qth_k = qdw_place(&l->k, dev);
            l->qth_v = qdw_place(&l->v, dev); l->qth_o = qdw_place(&l->o, dev);
            if (l->qth_q) { placed++; vram += (double)qdw_bytes(&l->q); }
            if (l->qth_k) { placed++; vram += (double)qdw_bytes(&l->k); }
            if (l->qth_v) { placed++; vram += (double)qdw_bytes(&l->v); }
            if (l->qth_o) { placed++; vram += (double)qdw_bytes(&l->o); }
        }
        int dev = qt_place_of("shexp", i);
        l->qth_shg = qdw_place(&l->sh_g, dev); l->qth_shu = qdw_place(&l->sh_u, dev); l->qth_shd = qdw_place(&l->sh_d, dev);
        if (l->qth_shg) { placed++; vram += (double)qdw_bytes(&l->sh_g); }
        if (l->qth_shu) { placed++; vram += (double)qdw_bytes(&l->sh_u); }
        if (l->qth_shd) { placed++; vram += (double)qdw_bytes(&l->sh_d); }
    }
    if (vram_bytes) *vram_bytes = vram;
    return placed;
}

/* Measured, not assumed. The placer prices a trunk component by the bytes it
 * saves on the CPU's memory bus, which presumes the GPU answers a GEMV faster
 * than the CPU does. Four Tesla M10 (sm_50) said otherwise: every placed
 * component ran slower there, lm_head 68.8 ms against 41.7 on the CPU, and
 * decode fell from 3.56 to 2.68 tok/s (#1652). So before any trunk upload,
 * one DeltaNet input projection (the most numerous placed matrix) is timed
 * both ways on the device that would host it, ten GEMVs each, best of three
 * rounds after a warm-up, and the trunk goes to VRAM only if the GPU wins.
 * Only the automatic placement is questioned: a hand-written COLI_PLACE
 * stands. COLI_TRUNK_PROBE=0 skips the probe and trusts the placer. The
 * probe's copy stays resident (one projection, ~25 MB on the 35B). */
static int trunk_probe_gpu_wins(Model *m){
    const char *e = getenv("COLI_TRUNK_PROBE");
    if (e && *e == '0') return 1;
    if (!qt_place_is_auto()) return 1;
    Cfg *c = &m->c;
    QW *w = NULL; int dev = QT_PLACE_CPU;
    for (int i = 0; i < c->n_layers && !w; i++) {
        if (c->is_attn[i]) continue;
        if (m->L[i].dn_qkv.q) { w = &m->L[i].dn_qkv; dev = qt_place_of("dnproj", i); }
    }
    if (!w) return 1;                                /* dense-i8 off: nothing will be placed */
    if (dev == QT_PLACE_CPU) dev = qt_place_of("lmhead", 0);
    if (dev == QT_PLACE_CPU) return 1;               /* nothing placed: nothing to measure */
    int I = w->I, O = w->O;
    int h = qt_dense_init(w->q, w->sc, I, O, dev);
    if (h < 0) return 1;                             /* cannot measure: the placer's word stands */
    float *x = malloc((size_t)I * sizeof(float)), *y = malloc((size_t)O * sizeof(float));
    if (!x || !y) { free(x); free(y); return 1; }
    for (int i = 0; i < I; i++) x[i] = sinf(0.37f * (float)i);
    double gpu = 1e30, cpu = 1e30;
    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < 3; k++) if (!qt_dense_matmul(h, y, x, I, O)) { free(x); free(y); return 1; }
        double t0 = now_s();
        for (int k = 0; k < 10; k++) if (!qt_dense_matmul(h, y, x, I, O)) { free(x); free(y); return 1; }
        double tg = (now_s() - t0) / 10;
        for (int k = 0; k < 3; k++) matmul_q(y, x, w->q, w->sc, I, O);
        t0 = now_s();
        for (int k = 0; k < 10; k++) matmul_q(y, x, w->q, w->sc, I, O);
        double tc = (now_s() - t0) / 10;
        if (tg < gpu) gpu = tg;
        if (tc < cpu) cpu = tc;
    }
    free(x); free(y);
    int wins = gpu < cpu;
    fprintf(stderr, "[place] probe: one [%d x %d] int8 GEMV takes %.3f ms on CUDA dev %d, %.3f ms on the CPU -> trunk %s\n",
            O, I, gpu * 1e3, dev, cpu * 1e3, wins ? "to VRAM" : "stays on the CPU");
    return wins;
}

static void layers_forward_range(Model *m, float *x, int S, int pos_base,
                                 int layer_begin, int layer_end,
                                 int allow_prefetch, FILE *lf) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = layer_begin; i < layer_end; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        double _t0 = tm_now();
        if (c->is_attn[i]) {
            attention(m, l, i, nrm, S, pos_base, tmp);
            tm_add(S, 1, tm_now()-_t0);
        } else {
            deltanet(m, l, i, nrm, S, pos_base, tmp);
            tm_add(S, 0, tm_now()-_t0);
        }
        if (lf) fwrite(tmp + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* sublayer output */
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* post-deltanet residual */
        if (allow_prefetch && g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        _t0 = tm_now();
        moe(m, l, i, nrm, S, tmp);
        tm_add(S, 2, tm_now()-_t0);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);
        if (allow_prefetch && g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (allow_prefetch && g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
    }
    free(nrm); free(tmp);
}

/* Fotografia dello stato dopo un prefill "pinnato" (SUBMIT pin=1).
 *
 * A cosa serve. Punteggiare un menu chiuso significa mandare N volte
 * "prompt + opzione_i". Il prefisso condiviso e sempre lo stesso, ma
 * kv_prefix.h e tutto-o-niente: dopo l'opzione 1 lo stato tenuto e
 * "prompt + opzione_1", che NON e un prefisso di "prompt + opzione_2", e si
 * ricomincia da capo. Misurato su qwen36: 0,90 s quando il riuso prende,
 * 40 s quando non prende.
 *
 * Perche una fotografia e non il riuso parziale. Tenere il prefisso comune e
 * ricalcolare dalla divergenza e corretto per l'attenzione, che guarda solo
 * indietro, ma NON per gli strati ricorrenti: lo stato di una ricorrenza non
 * si riavvolge a una posizione arbitraria. qwen36 ha 30 layer lineari su 40.
 * Una fotografia invece funziona ovunque, perche riporta lo stato a un punto
 * in cui c'e davvero stato.
 *
 * Cosa contiene. Le righe KV delle posizioni del prompt non si copiano: non
 * sono state toccate, basta riportare indietro kv_len. Si copiano lo stato
 * ricorrente (DN_rec + DN_conv, ~63 MB su questo modello) e il vettore di
 * logit finale, cosi il predittore del primo token fresco esiste e la lettura
 * del prefill copre anche quello. */
static ColiPinPool g_pins;              /* piu scatti annidati, vedi pin_pool.h */
static const float *g_pin_logit = NULL; /* logit dello scatto rimesso */
static int    g_pin_use_logit = 0;   /* 1 quando questa richiesta e ripartita dalla fotografia */

/* Lo stato che questo motore deve fotografare oltre alle righe K/V: la
 * ricorrenza e la finestra di convoluzione di ogni strato DeltaNet. Sono
 * decine di MB per scatto, quindi COLI_PIN_SLOTS conta davvero qui. */
typedef struct { float **rec, **conv; int n_layers; } Q36PinState;

/* Stato della lettura del prefill: dichiarato qui perche step() lo consulta e
 * step() viene prima del codice di servizio che lo accende. Solo il servizio
 * lo accende e solo il servizio definisce serve_echo: senza main non esiste. */
#ifndef QWEN36_NO_MAIN
static int   g_echo_k  = 0;      /* 0 = spento */
static const char *g_echo_id = NULL;
static void serve_echo(const char *id, int pos, int token, const float *lo, int V, int k);
#endif
/* Clef's decision head reads the final-normed hidden state of EVERY position of
 * the prompt (clef_head.h). When set, step() writes them here, [S, hidden]; the
 * DECIDE path sets it around its one prefill and nothing else ever does. */
static float *g_hidden_sink = NULL;

#ifdef COLI_VULKAN
#include "qwen36_chain.h"  /* COLI_VK_CHAIN: every layer's dense chain on the device */
#endif

static void q36_embed_row(Model *m, int id, int pos, float *row) {
    int D = m->c.hidden;
    int vrow = (m->vis_map && pos < m->vis_map_len) ? m->vis_map[pos] : -1;
    if (vrow >= 0 && vrow < m->vis_rows_n)
        memcpy(row, m->vis_rows + (int64_t)vrow*D, D*sizeof(float));
    else if (m->embed_h)
        f16_to_f32_bulk(m->embed_h + (int64_t)id*D, row, D);
    else
        memcpy(row, m->embed + (int64_t)id*D, D*sizeof(float));
}

/* The forward: ids[0..S) at pos_base through every layer, the final norm and
 * lm_head on the last `nlogits` rows (one for a step, every row of a speculative
 * verify), each row's logits computed as a decode step computes them. */
static float *step_ex(Model *m, const int *ids, int S, int pos_base, int nlogits) {
    Cfg *c = &m->c; int D = c->hidden;
    if (m->resident_mode && m->first_step) m->resident_collecting = 1;
    /* Per-layer residual dump (last token) for torch-free cosine debugging.
     * Set DUMP_LAYERS=<path> to write n_layers * D raw float32 rows. */
    FILE *lf = NULL; const char *lfn = getenv("DUMP_LAYERS");
    if (lfn) { lf = fopen(lfn, "wb"); if (!lf) fprintf(stderr, "DUMP_LAYERS: cannot open %s\n", lfn); }
    if (g_pilot && m->token_count > 0) {
        pthread_mutex_lock(&g_pilot_mx);
        memset(m->is_queued, 0, (size_t)c->n_layers * c->n_experts);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) {
        /* The gather indexes embed by token id, so an id outside the vocabulary
         * reads off the end. Ids reach here from the tokenizer, from a serve
         * request and from the engine's own sampler -- three sources, one of
         * which is remote, and none of them checked until now. */
        if (ids[s] < 0 || ids[s] >= c->vocab) {
            fprintf(stderr, "token id %d out of range 0..%d -- refusing\n",
                    ids[s], c->vocab - 1);
            exit(1);
        }
        q36_embed_row(m, ids[s], pos_base + s, x + (int64_t)s*D);
    }
    tier_rebuild_evicted(m);
    int chain_n = 0;   /* layers the chain ran (a partial chain: the CPU runs the rest) */
#ifdef COLI_VULKAN
    /* COLI_VK_CHAIN: the layers, the final norm and lm_head on the device; x comes
     * back only when the prefill read-out below needs every row. A partial chain
     * brings back every row's residual after its layers: the CPU continues from there. */
    float *chain_logit = NULL;
    if (g_vk_chain) {
        int echo = 0, rows_only = 0;
#ifndef QWEN36_NO_MAIN
        echo = g_echo_k > 0 && g_echo_id && S > 1;
#endif
        if (g_hidden_sink) echo = 1;   /* the head reads every row */
        chain_logit = falloc((int64_t)nlogits * c->vocab);
        chain_n = q36c_forward(m, x, S, pos_base, lf, echo, nlogits, chain_logit, &rows_only);
        if (!chain_n) {
            free(chain_logit); chain_logit = NULL;
            q36c_cpu_step(m, pos_base);
            /* Earlier chunks may have returned their final rows for Clef or
             * prompt logprobs. A failed forward replays every row on the CPU. */
            for (int s = 0; s < S; s++)
                q36_embed_row(m, ids[s], pos_base + s, x + (int64_t)s*D);
        } else if (rows_only) { free(chain_logit); chain_logit = NULL; }
    }
    if (!chain_logit)
#endif
    layers_forward_range(m, x, S, pos_base, chain_n, c->n_layers, 1, lf);
    if (g_hidden_sink) {
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < S; s++)
            rmsnorm_row(g_hidden_sink + (int64_t)s*D, x + (int64_t)s*D, m->final_norm, D, c->eps);
    }
    /* Recorded HERE, where the tokens actually entered the state, rather than
     * derived from the caller's bookkeeping: the invariant that fed[0..len-1]
     * are the ids the state was built from is the whole safety argument. */
    kv_prefix_record(&m->kvp, ids, pos_base, S);
    /* the placeholders' ids do not say which picture they held */
    if (m->vis_map && m->vis_rows_n > 0) kv_prefix_taint(&m->kvp);
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens) pin_hot_experts(m);
    m->kv_len = pos_base + S;
    /* Lettura del prefill: una passata di lm_head per posizione, pagata SOLO
     * dalle richieste che hanno chiesto il canale. La posizione p predice il
     * token p+1, quindi si copre l'intero blocco fresco tranne il suo primo
     * token, il cui predittore sta nello stato precedente. Per questo il
     * chiamante arretra di uno il riuso del prefisso quando la lettura e
     * accesa: cosi il primo token dell'opzione ricade sempre qui dentro. */
#ifndef QWEN36_NO_MAIN
    if (g_echo_k > 0 && g_echo_id && S > 0) {
        float *erow = falloc(D), *elog = falloc(c->vocab);
        /* Il primo token fresco e predetto dallo stato PRECEDENTE, che dopo un
         * riavvolgimento e proprio quello fotografato: senza questi logit
         * l'opzione perderebbe il suo primo token, che spesso e l'unico. */
        if (g_pin_use_logit && g_pin_logit)
            serve_echo(g_echo_id, pos_base, ids[0], g_pin_logit, c->vocab, g_echo_k);
        for (int p = 0; p + 1 < S; p++) {
            rmsnorm_row(erow, x + (int64_t)p*D, m->final_norm, D, c->eps);
            if (!qt_lmhead_matmul(elog, erow, D, c->vocab))
                matmul_d(elog, erow, &m->lm_head, 1, D, c->vocab);
            serve_echo(g_echo_id, pos_base + p + 1, ids[p+1], elog, c->vocab, g_echo_k);
        }
        free(erow); free(elog);
    }
#endif
    float *last = falloc(D);
    float *logit;
#ifdef COLI_VULKAN
    if (chain_logit) logit = chain_logit;   /* the chain ran the final norm and lm_head */
    else
#endif
    logit = falloc((int64_t)nlogits * c->vocab);
    double _th = tm_now();
#ifdef COLI_VULKAN
    if (!chain_logit)
#endif
    for (int r = 0; r < nlogits; r++) {
        float *lr = logit + (int64_t)r * c->vocab;
        rmsnorm_row(last, x + (int64_t)(S - nlogits + r)*D, m->final_norm, D, c->eps);
        if (!qt_lmhead_matmul(lr, last, D, c->vocab))
            matmul_d(lr, last, &m->lm_head, 1, D, c->vocab);
    }
    if (tm_on()) { tm_add(S, 5, tm_now()-_th); if (S==1) g_tm_dec_tokens++; else g_tm_pre_tokens += S; }
    free(x); free(last);
    if (lf) fclose(lf);
    if (m->resident_collecting) {
        int prefill_end = m->first_step;
        apply_resident(m, prefill_end ? 0 : 1);   /* always report after prefill; quiet mid-decode */
        if (prefill_end) {
            m->first_step = 0;
            if (m->resident_mode < 2) m->resident_collecting = 0;  /* mode 1: stop after prefill */
            /* mode 2: keep collecting through decode for incremental pin */
        }
    }
    return logit;
}

static float *step(Model *m, const int *ids, int S, int pos_base) {
    return step_ex(m, ids, S, pos_base, 1);
}

/* ---- several conversations at once (KV_SLOTS>1, serve_mux) --------------------
 * Each conversation owns a Q36Seq: its K and V rows, its DeltaNet state, its record
 * of the tokens its rows hold, an image turn's rope positions. The Model holds the
 * conversation a prefill runs on (q36_seq_swap trades it for a parked one); a decode
 * step parks them all and runs one forward over a row of each (q36_step_rows). */
static void q36_seq_swap(Model *m, Q36Seq *q) {
#define Q36_SEQ_SWAP(T,a,b) do { T t_ = (a); (a) = (b); (b) = t_; } while (0)
    Q36_SEQ_SWAP(float **, m->K, q->K); Q36_SEQ_SWAP(float **, m->V, q->V);
    Q36_SEQ_SWAP(float **, m->DN_rec, q->DN_rec); Q36_SEQ_SWAP(float **, m->DN_conv, q->DN_conv);
    Q36_SEQ_SWAP(int, m->kv_len, q->kv_len); Q36_SEQ_SWAP(kv_prefix, m->kvp, q->kvp);
    Q36_SEQ_SWAP(int *, m->mpos, q->mpos); Q36_SEQ_SWAP(int, m->mpos_len, q->mpos_len);
    Q36_SEQ_SWAP(int, m->rope_delta, q->rope_delta);
#undef Q36_SEQ_SWAP
}

/* A conversation's state of its own, at the Model's KV capacity (ensure_kv sized it
 * for the whole context before the serve began): 0 when out of memory. */
static void q36_seq_free(Model *m, Q36Seq *q) {
    for (int i = 0; i < m->c.n_layers; i++) {
        if (q->K) free(q->K[i]); if (q->V) free(q->V[i]);
        if (q->DN_rec) free(q->DN_rec[i]); if (q->DN_conv) free(q->DN_conv[i]);
    }
    free(q->K); free(q->V); free(q->DN_rec); free(q->DN_conv);
    kv_prefix_free(&q->kvp); free(q->mpos);
    memset(q, 0, sizeof *q);
}
static int q36_seq_alloc(Model *m, Q36Seq *q) {
    Cfg *c = &m->c; memset(q, 0, sizeof *q);
    int n = c->n_layers;
    q->K = calloc((size_t)n, sizeof(float *)); q->V = calloc((size_t)n, sizeof(float *));
    q->DN_rec = calloc((size_t)n, sizeof(float *)); q->DN_conv = calloc((size_t)n, sizeof(float *));
    int ok = q->K && q->V && q->DN_rec && q->DN_conv && kv_prefix_alloc(&q->kvp, m->kv_cap);
    for (int i = 0; ok && i < n; i++) {
        if (c->is_attn[i]) {
            size_t kv = (size_t)c->kv_heads * m->kv_cap * c->k_head_dim;
            q->K[i] = malloc(kv * sizeof(float)); q->V[i] = malloc(kv * sizeof(float));
            ok = q->K[i] && q->V[i];
        } else if (m->DN_rec[i]) {
            q->DN_rec[i] = calloc((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, sizeof(float));
            q->DN_conv[i] = calloc((size_t)c->dn_conv_dim * (c->dn_convk - 1), sizeof(float));
            ok = q->DN_rec[i] && q->DN_conv[i];
        }
    }
    if (!ok) { q36_seq_free(m, q); return 0; }
    return 1;
}

/* One decode step of several conversations: row s is the token ids[s] at
 * rows[s].pos of the conversation rows[s].seq, every conversation parked. The
 * projections, the routed experts and lm_head run once over the S rows; the
 * attention and DeltaNet read and write each row's own conversation (m->mux_rows).
 * The CPU kernels give a row the same bits whatever S is, so each conversation gets
 * the logits it would alone. Decode rows are never an image's. S rows of logits. */
static float *q36_step_rows(Model *m, const Q36Row *rows, const int *ids, int S) {
    Cfg *c = &m->c; int D = c->hidden;
    if (g_pilot && m->token_count > 0) {
        pthread_mutex_lock(&g_pilot_mx);
        memset(m->is_queued, 0, (size_t)c->n_layers * c->n_experts);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) {
        if (ids[s] < 0 || ids[s] >= c->vocab) {
            fprintf(stderr, "token id %d out of range 0..%d -- refusing\n", ids[s], c->vocab - 1);
            exit(1);
        }
        if (m->embed_h) f16_to_f32_bulk(m->embed_h + (int64_t)ids[s]*D, x + (int64_t)s*D, D);
        else memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    }
    tier_rebuild_evicted(m);
    m->mux_rows = rows;
    layers_forward_range(m, x, S, 0, 0, c->n_layers, 1, NULL);
    m->mux_rows = NULL;
    for (int s = 0; s < S; s++) {
        Q36Seq *q = rows[s].seq;
        kv_prefix_record(&q->kvp, ids + s, rows[s].pos, 1);
        q->kv_len = rows[s].pos + 1;
    }
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens) pin_hot_experts(m);
    float *last = falloc((int64_t)S*D), *logit = falloc((int64_t)S * c->vocab);
    double _th = tm_now();
    for (int s = 0; s < S; s++) rmsnorm_row(last + (int64_t)s*D, x + (int64_t)s*D, m->final_norm, D, c->eps);
    /* the head once for every row; the CUDA tier's copy answers a row at a time */
    int on_card = 1;
    for (int s = 0; s < S && on_card; s++)
        on_card = qt_lmhead_matmul(logit + (int64_t)s * c->vocab, last + (int64_t)s*D, D, c->vocab);
    if (!on_card) matmul_d(logit, last, &m->lm_head, S, D, c->vocab);
    if (tm_on()) { tm_add(S, 5, tm_now()-_th); g_tm_dec_tokens += S; }
    free(x); free(last);
    return logit;
}

static void pilot_realload(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    pthread_mutex_lock(&g_pilot_mx);
    if (!m->is_queued[layer * c->n_experts + eid]) { pthread_mutex_unlock(&g_pilot_mx); return; }
    if (slot_indexed(m, layer, eid)) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
    Slot *s;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        int lru = slot_victim(lc, layer, 0);
        if (lru < 0) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
        s = &lc->slots[lru]; s->pinned = 0;
        if (vkt_ram_first(layer, s->eid)) vkt_ram_gave();
    }
    cache_hide(m, layer, s); s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);
    double t_read0 = now_s();
    load_expert_merged(m, layer, eid, s);
    double t_read = now_s() - t_read0;
    pthread_mutex_lock(&g_pilot_mx);
    m->t_disk += t_read;        /* sotto lock: qui arrivano anche i thread del PILOT */
    cache_publish(m, layer, s, eid); s->pinned = m->is_pinned[layer*c->n_experts+eid]; s->used = ++m->clock;
    m->is_queued[layer*c->n_experts+eid] = 0; pthread_mutex_unlock(&g_pilot_mx);
}

static void *pilot_worker(void *arg) {
    (void)arg;
    while (1) {
        unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
        unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
        if (r == w) { sleep_ms(1); continue; }
        int layer = pilot_q[r & 4095].l, eid = pilot_q[r & 4095].e;
        pilot_realload(pilot_m, layer, eid);
        __atomic_store_n(&pilot_r, r + 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void pilot_prefetch(Model *m, int lnext, const float *x, int S) {
    if (lnext < 0 || lnext >= m->c.n_layers || m->c.n_experts == 0) return;
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts;
    ensure_pilot_worker_started(m);
    float *logits = falloc((int64_t)S * E);
    Layer *l = &m->L[lnext];
    float *nrm_x = falloc((int64_t)S * D);
    for (int s = 0; s < S; s++) rmsnorm_row(nrm_x + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
    matmul_d(logits, nrm_x, &l->gate, S, D, E);   /* int8 copy (f32 may be freed) */
    free(nrm_x);
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        float *blended = pr;
        float *ema = m->momentum_logits + (int64_t)lnext*E;
        if (m->pilot_smooth > 0.f) {
            blended = falloc(E); int is_zero = 1;
            for (int e = 0; e < E; e++) if (ema[e] != 0.f) { is_zero = 0; break; }
            if (is_zero) { for (int e = 0; e < E; e++) { ema[e] = pr[e]; blended[e] = pr[e]; } }
            else { for (int e = 0; e < E; e++) { blended[e] = (1.f-m->pilot_smooth)*pr[e] + m->pilot_smooth*ema[e]; ema[e] = blended[e]; } }
        }
        int cand = 0; int idx[128];
        float max_logit = -1e30f; for (int e = 0; e < E; e++) if (blended[e] > max_logit) max_logit = blended[e];
        float *exps = falloc(E); float sum_exps = 0.f;
        for (int e = 0; e < E; e++) { exps[e] = expf(blended[e] - max_logit); sum_exps += exps[e]; }
        float cum_sum = 0.f; int min_cand = c->topk; int max_cand = c->topk * g_wide;
        if (max_cand < min_cand) max_cand = min_cand; if (max_cand > 128) max_cand = 128; if (max_cand > E) max_cand = E;
        for (int kk = 0; kk < max_cand; kk++) {
            int best = -1; float bv = -1.f;
            for (int e = 0; e < E; e++) { int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;} if (!taken && exps[e] > bv) { bv = exps[e]; best = e; } }
            if (best < 0) break;
            idx[kk] = best; cum_sum += bv; cand++;
            if (cum_sum >= m->pilot_conf_limit * sum_exps && cand >= min_cand) break;
        }
        free(exps);
        if (blended != pr) free(blended);
        for (int a = 0; a < cand-1; a++) for (int b = a+1; b < cand; b++)
            if (idx[b] >= 0 && (idx[a] < 0 || idx[a] > idx[b])) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
        for (int kk = 0; kk < cand; kk++) {
            int eid = idx[kk]; if (eid < 0) continue;
            int found = 0, fz = -1; pthread_mutex_lock(&g_pilot_mx); LCache *lc = &m->cache[lnext];
            Slot *resident = slot_indexed(m, lnext, eid);
            if (resident) { found = 1; fz = (int)(resident - lc->slots); }
            pthread_mutex_unlock(&g_pilot_mx);
            /* Lookahead: RAM-resident layer-L+1 candidates go to VRAM asynchronously */
            if (found && fz >= 0 && qt_ready()) {
                Slot *ps = &lc->slots[fz];
                /* Same int8 container case as the resident offer in moe():
                 * on an int8 container ps->g4 is NULL and the prefetch offer
                 * never fired (#1391). tier_offer_slot handles both formats. */
                tier_offer_slot(lnext, eid, ps);
            }
            if (!found && vkt_ram_first(lnext, eid)) found = 1;   /* the device serves it: nothing to read */
            if (!found) {
                int gidx = lnext*E + eid;
                pthread_mutex_lock(&g_pilot_mx); int already_queued = m->is_queued[gidx];
                if (!already_queued) m->is_queued[gidx] = 1;
                pthread_mutex_unlock(&g_pilot_mx);
                if (!already_queued) {
                    unsigned w2 = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                    unsigned r2 = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                    if (w2 - r2 < 4096) { pilot_q[w2 & 4095].l = lnext; pilot_q[w2 & 4095].e = eid; __atomic_store_n(&pilot_w, w2+1, __ATOMIC_RELEASE); }
                    else { pthread_mutex_lock(&g_pilot_mx); m->is_queued[gidx] = 0; pthread_mutex_unlock(&g_pilot_mx); }
                }
            }
        }
    }
    free(logits);
}

/* When DUMP=<path> is set, generate() copies the last-token logits here so main()
 * can write them to <path> (raw float32, length = vocab). Lets a torch-free
 * cosine comparison against tools/_ref_dn.py's numpy logits validate the port. */
static float *g_last_logit = NULL;

/* Zero the DeltaNet recurrent state so a new request doesn't inherit the
 * previous conversation's hidden state. Must be called at the start of every
 * generation (the CLI runs once, so this is also correct there). */
static void q36_pin_state_free(void *v){
    Q36PinState *st = (Q36PinState *)v;
    if (!st) return;
    for (int i = 0; i < st->n_layers; i++){
        if (st->rec)  free(st->rec[i]);
        if (st->conv) free(st->conv[i]);
    }
    free(st->rec); free(st->conv); free(st);
}

static void pin_drop(void){
    coli_pin_pool_clear(&g_pins, q36_pin_state_free);
    g_pin_logit = NULL; g_pin_use_logit = 0;
}

static Q36PinState *q36_pin_state_save(Model *m, Q36PinState *reuse){
    Cfg *c = &m->c;
#ifdef COLI_VULKAN
    q36c_sync_host(m);   /* the dense chain keeps the newest state on the device */
#endif
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim;
    size_t nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    Q36PinState *st = reuse;
    if (st && st->n_layers != c->n_layers) { q36_pin_state_free(st); st = NULL; }
    if (!st){
        st = (Q36PinState *)calloc(1, sizeof(*st));
        if (!st) return NULL;
        st->n_layers = c->n_layers;
        st->rec  = (float**)calloc((size_t)c->n_layers, sizeof(float*));
        st->conv = (float**)calloc((size_t)c->n_layers, sizeof(float*));
        if (!st->rec || !st->conv) { q36_pin_state_free(st); return NULL; }
        for (int i = 0; i < c->n_layers; i++){
            if (c->is_attn[i]) continue;
            st->rec[i]  = (float*)malloc(nr * sizeof(float));
            st->conv[i] = (float*)malloc(nc * sizeof(float));
            if (!st->rec[i] || !st->conv[i]) { q36_pin_state_free(st); return NULL; }
        }
    }
    dn_gpu_pull_all(m);   /* the card may be ahead of the host copy */
    for (int i = 0; i < c->n_layers; i++){
        if (c->is_attn[i]) continue;
        if (m->DN_rec[i]  && st->rec[i])  memcpy(st->rec[i],  m->DN_rec[i],  nr * sizeof(float));
        if (m->DN_conv[i] && st->conv[i]) memcpy(st->conv[i], m->DN_conv[i], nc * sizeof(float));
    }
    return st;
}

static void pin_save(Model *m, const int *ids, int n, const float *logit){
    Cfg *c = &m->c;
    coli_pin_pool_init(&g_pins, c->vocab);
    ColiPin *k = coli_pin_store(&g_pins, ids, n, logit);
    if (!k) return;                       /* ottimizzazione, mai un errore */
    Q36PinState *st = q36_pin_state_save(m, (Q36PinState *)k->state);
    if (!st) { k->len = 0; return; }      /* senza stato lo scatto sarebbe una bugia */
    k->state = st;
    fprintf(stderr, "[PIN] scatto a %d token\n", n); fflush(stderr);
}

/* Se la fotografia e un prefisso del nuovo prompt, la si rimette e si riparte
 * da li. Restituisce quanti token sono gia fatti, 0 se non si applica. */
static int pin_restore(Model *m, const int *ids, int n){
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim;
    size_t nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    g_pin_logit = NULL;
    /* Il piu profondo degli scatti che sia un prefisso stretto di questo
     * prompt. Gli strati di attenzione non stanno nella fotografia: le loro
     * righe K/V restano dove sono, e se i banchi sono stati ributtati quelle
     * righe non descrivono piu niente -- solo il registro del prefisso lo sa
     * (kv_prefix_holds). Uno scatto orfano si butta e si riprova col
     * precedente, invece di rinunciare e rifare tutto da zero. */
    int s = coli_pin_best(&g_pins, ids, n);
    while (s >= 0) {
        ColiPin *k = &g_pins.slot[s];
        Q36PinState *st = (Q36PinState *)k->state;
        if (st && kv_prefix_holds(&m->kvp, k->ids, k->len)) {
            for (int i = 0; i < c->n_layers; i++){
                if (c->is_attn[i]) continue;
                if (m->DN_rec[i]  && st->rec[i])  memcpy(m->DN_rec[i],  st->rec[i],  nr * sizeof(float));
                if (m->DN_conv[i] && st->conv[i]) memcpy(m->DN_conv[i], st->conv[i], nc * sizeof(float));
            }
#ifdef COLI_VULKAN
            q36c_host_wrote(m, 0);   /* the device's copy goes up again before the next chain step */
#endif
            dn_gpu_invalidate(m);   /* restored on the host: the card's copy is from another prompt */
            m->kv_len = k->len;
            kv_prefix_clear(&m->kvp);
            kv_prefix_record(&m->kvp, k->ids, 0, k->len);
            g_pin_logit = k->logit;
            coli_pin_touch(&g_pins, s);
            return k->len;
        }
        k->len = 0;
        s = coli_pin_best(&g_pins, ids, n);
    }
    return 0;
}

static void reset_recurrent(Model *m){
    Cfg *c = &m->c;
    /* Paired with the record on purpose: whoever zeroes the state must also
     * forget what it was built from, or the two disagree in favour of the one
     * nobody can check. */
    kv_prefix_clear(&m->kvp);
    for (int i = 0; i < c->n_layers; i++){
        if (c->is_attn[i]) continue;
        if (m->DN_rec[i])  memset(m->DN_rec[i],  0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        if (m->DN_conv[i]) memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
#ifdef COLI_VULKAN
    q36c_host_wrote(m, 1);   /* zeros: the dense chain fills its copy with zeros */
#endif
    dn_gpu_invalidate(m);   /* zero on the host is the truth now; the card re-loads it before its next step */
}

/* Allocate (once) or reuse the KV cache across requests. Grows only when a
 * longer context is needed; never shrinks. Frees the previous buffers on
 * growth so the server doesn't leak KV memory across requests. */
static void ensure_kv(Model *m){
    Cfg *c = &m->c;
    if (m->kv_cap >= m->max_t && m->K) {
        /* max_t is the ROW STRIDE of the KV cache, not just a capacity: a row
         * lives at (head*max_t + position)*head_dim. Callers set it per request
         * from prompt+max_tok, so a shorter request used to shrink the stride
         * while the allocation stayed the same size -- harmless only as long as
         * every turn rewrote every row from position 0. Reuse reads rows an
         * earlier turn wrote, so the stride has to stay the one they were
         * written with: the allocation's, which is kv_cap. */
        m->max_t = m->kv_cap;
        return;
    }
    /* Growth COPIES the rows instead of discarding them. A chat resends a
     * longer transcript every turn, so this reallocation lands on exactly the
     * turn that wants to reuse the previous one's state: freeing the rows here
     * would make kv_prefix_reuse miss in the one case it exists for. The copy
     * is a memcpy at DRAM speed; the prefill it saves is seconds of expert
     * reads on a streaming engine. The row stride IS max_t, so the copy must be
     * per head -- one flat memcpy would land every head but the first at the
     * wrong offset, and that reads as a plausible answer from another
     * conversation rather than as a crash. */
    float **oldK = m->K, **oldV = m->V;
    int old_stride = m->kv_cap, keep = m->K ? m->kvp.len : 0;
    if (keep > old_stride) keep = old_stride;
    if (keep > m->max_t)   keep = m->max_t;
    m->K = calloc((size_t)c->n_layers, sizeof(float*)); m->V = calloc((size_t)c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++){
        if (c->is_attn[i]){
            int64_t kvd = c->k_head_dim;
            m->K[i] = falloc((int64_t)c->kv_heads * m->max_t * kvd);
            m->V[i] = falloc((int64_t)c->kv_heads * m->max_t * kvd);
            if (keep > 0 && oldK && oldK[i] && oldV[i])
                for (int h = 0; h < c->kv_heads; h++){
                    memcpy(m->K[i] + (int64_t)h*m->max_t*kvd,
                           oldK[i] + (int64_t)h*old_stride*kvd, (size_t)keep*kvd*sizeof(float));
                    memcpy(m->V[i] + (int64_t)h*m->max_t*kvd,
                           oldV[i] + (int64_t)h*old_stride*kvd, (size_t)keep*kvd*sizeof(float));
                }
        } else { m->K[i] = NULL; m->V[i] = NULL; }
    }
    if (oldK){
        for (int i = 0; i < c->n_layers; i++){ if (oldK[i]) free(oldK[i]); if (oldV[i]) free(oldV[i]); }
        free(oldK); free(oldV);
    }
    /* Attention scores: one row per thread, indexed by absolute position, so
     * each row must hold max_t entries. Sized here rather than in attention()
     * because it grows with the context exactly like the KV cache does, and
     * because a per-call allocation would run 10x per token. */
    free(m->attn_sc);
    m->attn_sc_thr = 1;
#ifdef _OPENMP
    m->attn_sc_thr = omp_get_max_threads();
    if (m->attn_sc_thr < 1) m->attn_sc_thr = 1;
#endif
    m->attn_sc = falloc((int64_t)m->attn_sc_thr * m->max_t);
    m->kv_cap = m->max_t;
    /* The record describes those same positions, so it survives with them. If
     * its own allocation fails, reuse simply stops: this is an optimisation and
     * must never be the reason a turn fails. */
    if (keep > 0) {
        if (!kv_prefix_grow(&m->kvp, m->max_t, keep)) { kv_prefix_clear(&m->kvp); m->kv_len = 0; }
        else if (m->kv_len > keep) m->kv_len = keep;
    } else if (!kv_prefix_alloc(&m->kvp, m->max_t)) {
        kv_prefix_clear(&m->kvp);
    }
}

/* ---- speculative decoding with prompt lookup (COLI_LOOKUP=1) ----------------
 * The caller asks for the logits that follow the token it just picked
 * (q36_spec_step). When the recent tokens repeat an n-gram of the context, the
 * tokens that followed it there are proposed (spec_draft.h, up to
 * COLI_LOOKUP_DRAFTS of them), and one forward over the token and its k drafts
 * (the verify, S = k+1 rows) returns the first row's logits as step() would and
 * keeps the others. The caller's next picks settle the drafts one at a time: a
 * pick equal to the next draft is answered with that draft's row and no
 * forward; the first pick that differs undoes the rows after the ones that
 * stood. The DeltaNet state and conv rings go back to the copy the verify took
 * after its last standing row (deltanet() copies them after each of the first
 * snap_rows rows, row r into slot r; the rollback swaps pointers), kv_len and
 * the kv_prefix record go back to it, and the attention rows past it are a
 * stale tail the next forward overwrites.
 *
 * Every verify row is computed as a decode step computes it: the CPU kernels
 * give a row the same bits whatever S is, a Vulkan matrix runs a verify's rows
 * one at a time (g_q36_rowwise, vk_dense_matmul and the chain), and lm_head
 * reads each row on its own (step_ex). So the logits the caller sees, and its
 * picks, greedy or sampled, are those of plain decoding. Drafting stays off
 * where that does not hold: the CUDA tier (its float order follows residency),
 * CACHE_ROUTE (it routes by residency) and a qpack container.
 *
 * Whether a proposal is drafted is the gate's call (spec_draft.h): the measured
 * acceptance by draft position and the measured cost of a verify by its rows;
 * COLI_SPEC_GATE=0 drafts every proposal in full (tests). */
typedef struct {
    int lookup, lookup_max;      /* COLI_LOOKUP: on; COLI_LOOKUP_DRAFTS: drafts per verify, 1..5 */
    int force, force_row;        /* COLI_LOOKUP_FORCE (tests): the oracle's tokens as the proposal */
    int *hist, hist_n, hist_cap; /* the tokens fed so far and the one about to be: what lookup searches */
    SpecGate gate;
    int ahead_n, ahead_i, ahead_pos;   /* a verify's rows at ahead_pos.., ahead_i the next to hand out */
    int ids[Q36_SPEC_ROWS];
    float *ahead_logit;
    uint64_t drafts, accepted, verifies, forwards, tokens;
    int ended_ahead;             /* the generation ended with a verify's rows unconsumed (undone) */
} Q36Spec;
static const int *g_q36_oracle; static int g_q36_oracle_n;   /* ref.json's full ids (COLI_LOOKUP_FORCE) */

static int q36_spec_force_mode(const char *v, int *row) {
    *row = 0;
    if (!v || !*v) return 0;
    if (!strcmp(v, "accept")) return 'a';
    if (!strcmp(v, "mixed")) return 'm';
    if (!strcmp(v, "cycle")) return 'c';
    if (!strncmp(v, "row", 3) && v[3] >= '1' && v[3] <= '0' + Q36_SPEC_SNAPS && !v[4]) { *row = v[3] - '0'; return 'w'; }
    fprintf(stderr, "COLI_LOOKUP_FORCE must be accept, mixed, cycle or row1..row%d\n", Q36_SPEC_SNAPS);
    exit(1);
}

static void q36_spec_begin(Model *m, Q36Spec *sp, const int *prompt, int np) {
    memset(sp, 0, sizeof *sp);
    const char *e = getenv("COLI_LOOKUP");
    /* on by default, always gated (docs/speculative.md): COLI_LOOKUP=0 turns it off */
    int asked = e && *e == '1';
    sp->lookup = !(e && *e == '0');
    e = getenv("COLI_LOOKUP_DRAFTS");
    sp->lookup_max = e && *e ? atoi(e) : Q36_SPEC_SNAPS;
    if (sp->lookup_max < 1 || sp->lookup_max > Q36_SPEC_SNAPS) {
        fprintf(stderr, "COLI_LOOKUP_DRAFTS must be an integer in 1..%d\n", Q36_SPEC_SNAPS); exit(1);
    }
    sp->force = q36_spec_force_mode(getenv("COLI_LOOKUP_FORCE"), &sp->force_row);
    e = getenv("COLI_SPEC_GATE");
    spec_gate_init(&sp->gate, e && *e == '0');
    if (sp->lookup && (qt_ready() || g_cache_route || qq_active() || m->dn_dev)) {
        static int said;
        if (!said && asked) {   /* the default stays quiet: nothing was asked for */
            said = 1;
            fprintf(stderr, "[qwen36] COLI_LOOKUP=1: no drafts under %s (its results depend on what is "
                            "resident, so a verify would not reproduce plain decoding)\n",
                    qt_ready() || m->dn_dev ? "the CUDA tier" : g_cache_route ? "CACHE_ROUTE" : "a qpack container");
        }
        sp->lookup = 0;
    }
    if (sp->lookup && prompt && np > 0) {
        sp->hist_cap = np + 256;
        sp->hist = (int *)malloc((size_t)sp->hist_cap * sizeof(int));
        if (sp->hist) { memcpy(sp->hist, prompt, (size_t)np * sizeof(int)); sp->hist_n = np; }
        else { sp->lookup = 0; sp->hist_cap = 0; }
    }
}

static void q36_spec_free(Q36Spec *sp) { free(sp->hist); sp->hist = NULL; sp->hist_n = sp->hist_cap = 0; }

/* Copy slots for a verify of up to rows+1 rows, allocated the first time they are
 * needed. 0 = out of memory (the caller takes a plain step). */
static int q36_spec_alloc(Model *m, int rows) {
    Cfg *c = &m->c;
    if (rows > Q36_SPEC_SNAPS) rows = Q36_SPEC_SNAPS;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int sl = m->snap_slots; sl < rows; sl++) {
        if (!m->snap_rec[sl] && !(m->snap_rec[sl] = (float **)calloc((size_t)c->n_layers, sizeof(float *)))) return 0;
        if (!m->snap_conv[sl] && !(m->snap_conv[sl] = (float **)calloc((size_t)c->n_layers, sizeof(float *)))) return 0;
        for (int i = 0; i < c->n_layers; i++) {
            if (c->is_attn[i]) continue;
            if (!m->snap_rec[sl][i] && !(m->snap_rec[sl][i] = (float *)malloc(nr * sizeof(float)))) return 0;
            if (!m->snap_conv[sl][i] && !(m->snap_conv[sl][i] = (float *)malloc(nc * sizeof(float)))) return 0;
        }
        m->snap_slots = sl + 1;
    }
    return 1;
}

/* Back to the state after the verify's first `keep` rows, `len` positions fed. */
static void q36_spec_rollback(Model *m, int len, int keep) {
    Cfg *c = &m->c; int slot = keep - 1;
#ifdef COLI_VULKAN
    q36c_rollback(m, slot, len);   /* the dense chain's own copies: its device buffers swap too */
#endif
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) continue;
        float *t = m->DN_rec[i]; m->DN_rec[i] = m->snap_rec[slot][i]; m->snap_rec[slot][i] = t;
        t = m->DN_conv[i]; m->DN_conv[i] = m->snap_conv[slot][i]; m->snap_conv[slot][i] = t;
    }
    m->kv_len = len;
    if (m->kvp.len > len) m->kvp.len = len;
}

static void q36_spec_settle(Model *m, Q36Spec *sp, int keep, int rejected) {
    if (keep < sp->ahead_n) q36_spec_rollback(m, sp->ahead_pos + keep, keep);
    spec_gate_result(&sp->gate, SPEC_SRC_LOOKUP, keep - 1 + (rejected ? 1 : 0), keep - 1);
    free(sp->ahead_logit); sp->ahead_logit = NULL; sp->ahead_n = sp->ahead_i = 0;
}

/* The oracle's token at draft row j (1-based) of a verify at pos, or a wrong one at
 * the row the forced mode rejects; -1 past the oracle. */
static int q36_spec_forced(const Q36Spec *sp, int k, int j, int pos, int V) {
    int truth = pos + j < g_q36_oracle_n ? g_q36_oracle[pos + j] : -1;
    if (truth < 0) return -1;
    int v = (int)sp->verifies, wrong = sp->force == 'm' ? (v & 1) : sp->force == 'w' ? sp->force_row :
                                       sp->force == 'c' ? v % (k + 1) + 1 : 0;
    return j == wrong ? (truth + 1) % V : truth;
}

static void q36_spec_hist_push(Q36Spec *sp, int tok, int pos) {
    if (!sp->lookup || pos < 0) return;
    if (pos >= sp->hist_cap) {
        int cap = sp->hist_cap ? sp->hist_cap : 256; while (cap <= pos) cap *= 2;
        int *h = (int *)realloc(sp->hist, (size_t)cap * sizeof(int));
        if (!h) { sp->lookup = 0; return; }
        sp->hist = h; sp->hist_cap = cap;
    }
    if (pos > sp->hist_n) { sp->lookup = 0; return; }   /* a gap: the caller's history is not ours */
    sp->hist[pos] = tok; sp->hist_n = pos + 1;
}

/* The logits that follow `tok`, fed at `pos`, exactly as step(m,&tok,1,pos) gives
 * them; `more` is how many tokens the caller may still want after `tok`. The
 * returned buffer is the caller's (it may hold more rows past the first vocab). */
static float *q36_spec_step(Model *m, Q36Spec *sp, int tok, int pos, int more) {
    Cfg *c = &m->c; int V = c->vocab;
    sp->tokens++;
    q36_spec_hist_push(sp, tok, pos);
    if (sp->ahead_n) {
        int i = sp->ahead_i;
        if (tok == sp->ids[i] && pos == sp->ahead_pos + i) {
            float *logit = falloc(V);
            memcpy(logit, sp->ahead_logit + (int64_t)(i - 1) * V, (size_t)V * sizeof(float));
            sp->accepted++;
            if (++sp->ahead_i == sp->ahead_n) q36_spec_settle(m, sp, sp->ahead_n, 0);
            return logit;
        }
        q36_spec_settle(m, sp, i, 1);
    }
    int room = more - 1;
    if (room > sp->lookup_max) room = sp->lookup_max;
    if (pos + room >= m->kv_cap) room = m->kv_cap - 1 - pos;
    int d[Q36_SPEC_ROWS], k = 0;
    if (sp->lookup && room > 0) {
        int n = 0;
        if (sp->force)
            while (n < room) { int t = q36_spec_forced(sp, room, n + 1, pos, V); if (t < 0) break; d[n++] = t; }
        else n = spec_lookup(sp->hist, sp->hist_n, 2, 4, room, d);
        if (n > 0) k = spec_gate_pick(&sp->gate, SPEC_SRC_LOOKUP, n, NULL);
        if (k > 0 && !q36_spec_alloc(m, k)) k = 0;
    }
    if (!k) {
        sp->forwards++;
        double t0 = now_s();
        float *logit = step(m, &tok, 1, pos);
        spec_gate_forward(&sp->gate, 1, now_s() - t0);
        return logit;
    }
    int S = k + 1;
    sp->ids[0] = tok; memcpy(sp->ids + 1, d, (size_t)k * sizeof(int));
    m->snap_rows = k; g_q36_rowwise = 1;
    double t0 = now_s();
    float *logit = step_ex(m, sp->ids, S, pos, S);
    spec_gate_forward(&sp->gate, S, now_s() - t0);
    m->snap_rows = 0; g_q36_rowwise = 0;
    sp->forwards++; sp->drafts += (uint64_t)k; sp->verifies++;
    sp->ahead_n = S; sp->ahead_i = 1; sp->ahead_pos = pos;
    sp->ahead_logit = falloc((int64_t)k * V);
    memcpy(sp->ahead_logit, logit + V, (size_t)k * V * sizeof(float));
    return logit;
}

/* End of a generation: verify rows the caller never consumed are undone, so the
 * state is the one plain decoding leaves after the same tokens. */
static void q36_spec_end(Model *m, Q36Spec *sp) {
    if (sp->ahead_n) { sp->ended_ahead = 1; q36_spec_settle(m, sp, sp->ahead_i, 0); }
}

static void q36_spec_report(const Q36Spec *sp, const char *scope) {
    if (!sp->lookup && !sp->verifies) return;
    if (!sp->gate.off) {
        char g[384];
        spec_gate_describe(&sp->gate, SPEC_SRC_LOOKUP, g, sizeof g);
        fprintf(stderr, "[qwen36 spec gate] %s: %s\n", scope, g);
    }
    fprintf(stderr, "[qwen36 lookup] %s: %.2f tokens/forward (%llu forwards per %llu tokens) | "
                    "acceptance %.1f%% (%llu/%llu drafts in %llu verifies) | gate %s, %llu declined, %llu probes%s\n",
            scope, sp->forwards ? (double)sp->tokens / sp->forwards : 0.0,
            (unsigned long long)sp->forwards, (unsigned long long)sp->tokens,
            sp->drafts ? 100.0 * sp->accepted / sp->drafts : 0.0,
            (unsigned long long)sp->accepted, (unsigned long long)sp->drafts, (unsigned long long)sp->verifies,
            sp->gate.off ? "off" : "on", (unsigned long long)sp->gate.declined[SPEC_SRC_LOOKUP],
            (unsigned long long)sp->gate.probes[SPEC_SRC_LOOKUP], sp->ended_ahead ? " | ended mid-verify" : "");
}

static Q36Spec g_q36_run_spec;   /* generate()'s, reported by main() */

static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    /* Same ceiling serve_one() enforces. Past max_position_embeddings the RoPE
     * positions leave the range the model was trained on, so this is a
     * correctness limit, not just a memory one. */
    if (np + n_new > QWEN36_ATTN_MAX_CTX) {
        fprintf(stderr, "[ctx] prompt %d + %d new exceeds the %d-token ceiling\n",
                np, n_new, QWEN36_ATTN_MAX_CTX);
        exit(1);
    }
    m->max_t = np + n_new;
    reset_recurrent(m);
    ensure_kv(m);
    m->kv_len = 0;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);
    qt_stats_mark();   /* [qtier] stats also report the hit rate from here on */
    int len = np;
    Q36Spec *sp = &g_q36_run_spec;
    q36_spec_begin(m, sp, prompt, np);   /* COLI_LOOKUP=1: drafts, every token still the argmax below */
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        if (s == 0 && g_ttft < 0) g_ttft = now_s() - g_gen_t0;   /* record TTFT */
        if (g_stream) { stream_token(best); fflush(stdout); }
        if (s == n_new - 1) {
            if (getenv("DUMP")) {
                g_last_logit = malloc((size_t)c->vocab * sizeof(float));
                memcpy(g_last_logit, logit, (size_t)c->vocab * sizeof(float));
            }
            free(logit); out[len++] = best; break;
        }
        free(logit); out[len++] = best;
        { extern double g_tm_step; double _s0 = tm_now();
          logit = q36_spec_step(m, sp, best, len - 1, n_new - 1 - s);
          if (tm_on()) g_tm_step += tm_now()-_s0; }
    }
    q36_spec_end(m, sp);
}

static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    if (nfull > QWEN36_ATTN_MAX_CTX) {
        fprintf(stderr, "[ctx] %d tokens exceed the %d-token ceiling\n",
                nfull, QWEN36_ATTN_MAX_CTX);
        exit(1);
    }
    m->max_t = nfull;
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
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

#ifndef QWEN36_NO_MAIN

/* ===================== coli serve mode (SERVE=1) ===================== *
 * Implements the colibri gateway wire protocol so `coli chat` / `coli web` /
 * `coli serve` can drive this engine. Without it the engine is unreachable:
 * users run `coli chat`, not the binary directly.
 * Protocol (matches kimi_k3.c / inkling.c, the other non-GLM engines):
 *   engine:  \x01\x01READY\x01\x01\n
 *            STAT 0 0.00 0.0 <rss>\n
 *   gateway: SUBMIT <id> <slot> <plen> <max_tok> <temp> <top_p>\n <payload bytes>\n
 *   engine:  ACCEPT <id> <np>\n
 *            DATA <id> <n>\n <bytes>\n     (repeated per decoded chunk)
 *            DONE <id> STAT <gen> <tps> <hit%> <rss> <np> <limited>\n
 *   gateway: CANCEL <id>  (abort current turn)
 * Windows: stdout/stdin must go binary BEFORE the READY sentinel or the CRT
 * rewrites the trailing \n as \r\n and the gateway never matches it -> the
 * session hangs forever (#748). compat.h's coli_serve_binary_mode (#749)
 * carries that fix for every engine; see its comment for the full story. */

typedef struct { char id[64]; int slot, max_tok; float temp, top_p; char *payload; int plen;
                 int logprobs;   /* SUBMIT logprobs=k: 0 = canale spento (opt-in) */
                 int pin;        /* SUBMIT pin=1: fotografa lo stato dopo il prefill */
               } ServeReq;

/* An image waiting for the SUBMIT that uses it (IMAGE <id> <bytes> <grid_h>
 * <grid_w>, then the float32 patches and a newline -- the frame serve_codec.h
 * defines and the gateway already sends to qwen38). A second one before the
 * SUBMIT replaces the first, and says so. */
static struct { unsigned char *patches; unsigned long long bytes; int grid_h, grid_w, present; } g_pending_image;
static void q36_pending_image_clear(void){
    free(g_pending_image.patches);
    memset(&g_pending_image, 0, sizeof g_pending_image);
}

static int serve_read_req(ServeReq *q){
    char line[512], cmd[16], id[64];
    if(!fgets(line,sizeof(line),stdin)) return -1;
    if(sscanf(line,"%15s %63s",cmd,id)<2) return 0;
    if(!strcmp(cmd,"IMAGE")){
        /* The payload is consumed whatever happens next: left in the stream it
         * would be read as the following frame's header. A header that does not
         * say how long the payload is cannot be skipped, so it ends the session. */
        unsigned long long bytes; int gh, gw;
        if(sscanf(line,"%*s %*s %llu %d %d",&bytes,&gh,&gw)!=3 || bytes>(1ull<<30) || gh<1 || gw<1){
            printf("ERROR %s bad image header\n",id); fflush(stdout); return -1;
        }
        unsigned char *buf = malloc((size_t)bytes + 1);
        if(!buf){ printf("ERROR %s out of memory\n",id); fflush(stdout); return -1; }
        if(bytes && fread(buf,1,(size_t)bytes,stdin)!=(size_t)bytes){ free(buf); return -1; }
        int t = fgetc(stdin); if(t=='\r') t = fgetc(stdin);
        if(t!='\n'){ free(buf); return -1; }
        if(g_pending_image.present) fprintf(stderr,"[qwen36] a second image arrived before its SUBMIT; dropping the first\n");
        q36_pending_image_clear();
        g_pending_image.patches = buf; g_pending_image.bytes = bytes;
        g_pending_image.grid_h = gh; g_pending_image.grid_w = gw; g_pending_image.present = 1;
        return 0;
    }
    if(g_clef && !strcmp(cmd,"DECIDE")){
        /* DECIDE <id> <slot> <bytes>, then the record and a newline (Clef only:
         * an engine without the head never announced decide=1). As for IMAGE, a
         * header that does not say how long the payload is ends the session. */
        int dslot; unsigned long long bytes;
        if(sscanf(line,"%*s %*s %d %llu",&dslot,&bytes)!=2 || dslot<0 || bytes>(64ull<<20)){
            printf("ERROR %s BAD_FRAME\n",id); fflush(stdout); return -1;
        }
        char *payload=malloc((size_t)bytes+1);
        if(!payload){ printf("ERROR %s DECIDE_FAILED out of memory\n",id); fflush(stdout); return -1; }
        if(bytes && fread(payload,1,(size_t)bytes,stdin)!=(size_t)bytes){ free(payload); return -1; }
        int t = fgetc(stdin); if(t=='\r') t = fgetc(stdin);
        if(t!='\n'){ free(payload); return -1; }
        payload[bytes]=0;
        snprintf(q->id,sizeof(q->id),"%s",id);
        q->payload=payload; q->plen=(int)bytes; q->slot=dslot;
        return 3;
    }
    /* 4: the request it names ends (serve_mux; a lone serve reads them mid-turn) */
    if(!strcmp(cmd,"CANCEL")||!strcmp(cmd,"STOP")){ snprintf(q->id,sizeof(q->id),"%s",id); return 4; }
    if(strcmp(cmd,"SUBMIT")) return 0;
    int slot, plen, max_tok; float temp, top_p;
    if(sscanf(line,"%*s %*s %d %d %d %f %f",&slot,&plen,&max_tok,&temp,&top_p)!=5 ||
       plen<0||plen>(1<<24)||max_tok<0){   /* 0 = modalita jev, vedi sotto */
        printf("ERROR %s bad submit header\n",id); fflush(stdout); return 0;
    }
    q->slot = slot;
    /* Le chiavi key=value stanno dopo i campi fissi. Si riusa il parser
     * condiviso di decode_batch.h invece di scriverne un secondo: e lo stesso
     * namespace che colibri.c gia accetta, quindi i due motori non divergono.
     * Un client vecchio non manda nessuna chiave e finisce a logprobs=0. */
    q->logprobs = 0;
    q->pin = 0;
    {
        /* Stessa tecnica di coli_submit_parse: %n da l'offset dopo i campi
         * numerici, e la chiave deve essere separata da spazio (altrimenti
         * "1logprobs=5" sarebbe un campo malformato, non un opt-in). Qui i
         * campi fissi sono 6 e non 7, perche questo header non ha gbytes. */
        int base = 0;
        int sd, sp_, smt; float st, stp;
        if (sscanf(line, "%*s %*s %d %d %d %f %f%n", &sd, &sp_, &smt, &st, &stp, &base) == 5 &&
            base > 0 && (line[base] == ' ' || line[base] == '\t')) {
            /* coli_submit_ext rifiuta qualunque byte non-spazio dopo il valore,
             * e fgets lascia il '\n' in fondo: va tolto, o "logprobs=5\n" e
             * un valore con spazzatura attaccata. Si lavora su una copia per
             * non toccare la riga che il chiamante ha gia parsato. */
            char ext_buf[256];
            snprintf(ext_buf, sizeof ext_buf, "%s", line + base);
            for (char *e = ext_buf; *e; e++) if (*e == '\n' || *e == '\r') { *e = 0; break; }
            const char *tail = ext_buf;
            while (*tail == ' ' || *tail == '\t') tail++;
            if (*tail) {
                ColiSubmit ext; memset(&ext, 0, sizeof ext);
                if (coli_submit_ext(ext_buf, &ext)) { q->logprobs = ext.logprobs; q->pin = ext.pin; }
                else { printf("ERROR %s bad submit extension\n", id); fflush(stdout); return 0; }
            }
        }
    }
    /* La convalida di max_tok si chiude QUI, non sopra: 0 e legittimo solo in
     * modalita jev, e se logprobs c'e si sa solo dopo aver letto le chiavi. */
    if(max_tok < 1 && !(max_tok == 0 && q->logprobs > 0)){
        printf("ERROR %s bad submit header\n",id); fflush(stdout); return 0;
    }
    char *payload=malloc((size_t)plen+1);
    if(!payload){ printf("ERROR %s out of memory\n",id); fflush(stdout); return 0; }
    if(fread(payload,1,(size_t)plen,stdin)!=(size_t)plen){ free(payload); return -1; }
    (void)fgetc(stdin); payload[plen]=0;
    snprintf(q->id,sizeof(q->id),"%s",id);
    q->max_tok=max_tok; q->temp=temp; q->top_p=top_p;
    q->payload=payload; q->plen=plen;
    return 2;
}

/* Coda numerica di un token, formato IDENTICO a logprob_tail di colibri.c:
 * " <lp> <k> [tid tlp]*k", normalizzata in log-softmax sul vocabolario intero.
 * Due motori che scrivono lo stesso canale devono scrivere lo stesso formato,
 * altrimenti il client ne ha due da conoscere. lo==NULL da " nan 0". */

/* Un token -> un frame, con la coda numerica. Si usa SOLO quando il client ha
 * chiesto logprobs=k: quel client punteggia, non mostra testo, quindi il buffer
 * UTF-8 (che accorpa i byte di piu token in un DATA) qui sarebbe di intralcio.
 * Senza logprobs il percorso resta quello di prima, byte per byte. */
static void serve_data_lp(const char *id, const char *p, int n, const char *tail){
    printf("DATA %s %d%s\n",id,n,tail);
    if(n>0) fwrite(p,1,(size_t)n,stdout);
    fputc('\n',stdout); fflush(stdout);
}

/* Lettura del prefill (canale logprobs, opt-in). I logit alla posizione p
 * predicono il token p+1, quindi scorrendo le posizioni fresche si ottiene il
 * logprob di OGNI token del blocco, non solo dei primi k di una classifica:
 * e cio che serve a punteggiare un'opzione che non e fra le piu probabili, e
 * a punteggiarne una di piu token sommando i suoi pezzi.
 * Vive in variabili globali e non nella firma di step() perche step() e la
 * matematica: il protocollo non deve entrarci. */
static void serve_echo(const char *id, int pos, int token, const float *lo, int V, int k){
    char tail[1024]; coli_logprob_tail(tail,sizeof tail,lo,V,token,k);
    unsigned char b[256]; int n=0; decode_id_to_bytes(token,b,&n);
    printf("ECHO %s %d %d%s\n",id,n,pos,tail);
    if(n>0) fwrite(b,1,(size_t)n,stdout);
    fputc('\n',stdout); fflush(stdout);
}

static void serve_data(const char *id, const char *p, int n){
    if(n<=0) return;
    printf("DATA %s %d\n",id,n);
    fwrite(p,1,(size_t)n,stdout); fputc('\n',stdout); fflush(stdout);
}

#include "qwen36_sample.h"

/* Chat turns end on <|im_end|>, base completions on <|endoftext|>. Resolve
 * both ids from the tokenizer's added_tokens: Qwen3.6's 248320-token vocab
 * puts them at 248044+, so the old hardcoded 151645 (the 151k-vocab Qwen id)
 * silently never matched and every serve turn ran into max_tok. Q36_EOS
 * still overrides for experiments. */
static int serve_eos_ids(int *ids, int cap){
    int n=0;
    if(getenv("Q36_EOS")){ ids[n++]=atoi(getenv("Q36_EOS")); return n; }
    for(int k=0;k<g_nspecial && n<cap;k++)
        if(!strcmp(g_sp_str[k],"<|im_end|>")||!strcmp(g_sp_str[k],"<|endoftext|>"))
            ids[n++]=g_sp_id[k];
    if(!n) ids[n++]=151645;   /* tokenizer without added_tokens: old default */
    return n;
}

/* Un CANCEL per la richiesta in corso, visto SENZA bloccare (#1332).
 *
 * Prima serve_read_req era l'unico posto che leggeva un CANCEL, e viene
 * chiamata solo fra una richiesta e l'altra: quando il comando arrivava, il
 * turno che doveva fermare era gia' finito. Il gateway intanto manda CANCEL e
 * aspetta l'ack tenendo l'ammissione dello scheduler, quindi un client che si
 * disconnette non liberava niente.
 *
 * Ritorna 1 se e' arrivato un CANCEL/STOP per `id`. Le righe che non
 * riconosciamo si scartano, com'e' sempre stato: la regola di compatibilita'
 * del protocollo vale in tutte e due le direzioni. Un SUBMIT non puo'
 * legalmente arrivare mentre l'unico slot e' occupato; se arriva viene
 * ignorato qui e sara' la lettura normale a rifiutarlo.
 *
 * Non legge MAI se stdin non e' pronto: una getline bloccante qui fermerebbe la
 * generazione in attesa di un comando che potrebbe non arrivare mai. */
static int serve_cancel_pending(const char *id){
    int cancelled = 0;
    while(coli_serve_stdin_ready()){
        char line[512], cmd[16], who[64];
        if(!fgets(line,sizeof(line),stdin)) break;
        if(sscanf(line,"%15s %63s",cmd,who)<2) continue;
        if((!strcmp(cmd,"CANCEL")||!strcmp(cmd,"STOP")) && !strcmp(who,id)) cancelled = 1;
    }
    return cancelled;
}

/* --- Dashboard: Brain e Profile ------------------------------------------
 * Stesse quattro righe di colibri.c/glm53.c, stessi byte: EMAP dopo READY,
 * HITS e PROF prima di DONE. Qui ogni layer e' MoE, quindi la griglia e'
 * n_layers x n_experts. Il tier VRAM non espone una query di residenza:
 * la griglia distingue solo disco (0) e cache RAM (1). Le fasi del Profile
 * vengono dai timer che il motore ha gia' (deltanet, attention, moe, head),
 * ora accumulati sempre e riportati a schermo solo con COLI_TIMERS=1; il
 * disco e' misurato attorno a load_expert_merged su entrambi i percorsi.
 * L'attesa asincrona resta 0 per costruzione. */
static void dash_hex(const uint8_t *bytes,int n,char *hex){
    for(int b=0;b<n;b++){ hex[2*b]="0123456789abcdef"[bytes[b]>>4]; hex[2*b+1]="0123456789abcdef"[bytes[b]&15]; }
    hex[2*n]=0;
}
static void emap_emit(Model *m){
    const Cfg *c=&m->c; const int rows=c->n_layers, cols=c->n_experts;
    /* A dense model has no expert grid. "EMAP 64 0 " would be a three-field line,
     * and the gateway stops its dispatcher on a line it cannot parse (#1757). */
    if(cols==0) return;
    uint8_t *cells=calloc((size_t)rows*cols,1);
    pthread_mutex_lock(&g_pilot_mx);
    for(int i=0;i<rows;i++) for(int e=0;e<cols;e++)
        cells[(size_t)i*cols+e]=(uint8_t)((vkt_resident(i,e)?2:slot_indexed(m,i,e)?1:0)<<6);   /* 2 = on the Vulkan device */
    pthread_mutex_unlock(&g_pilot_mx);
    char *hex=malloc((size_t)rows*cols*2+1); dash_hex(cells,rows*cols,hex);
    printf("EMAP %d %d %s\n",rows,cols,hex); fflush(stdout); free(hex); free(cells);
}
static void hits_emit(Model *m){
    const Cfg *c=&m->c; const int rows=c->n_layers, cols=c->n_experts, nb=(rows*cols+7)/8;
    if(cols==0) return;   /* see emap_emit */
    if(!m->ehit) ehit_mark(m,-1,-1);
    uint8_t *bm=calloc((size_t)nb,1); int bit=0;
    for(int i=0;i<rows;i++) for(int e=0;e<cols;e++,bit++)
        if(m->ehit[i][e]){ bm[bit>>3]|=(uint8_t)(1<<(bit&7)); m->ehit[i][e]=0; }
    char *hex=malloc((size_t)nb*2+1); dash_hex(bm,nb,hex);
    printf("HITS %d %d %s\n",rows,cols,hex); fflush(stdout); free(hex); free(bm);
}
static double tm_sum(int idx){ return (g_tm_dec[idx]+g_tm_pre[idx])/1e3; }   /* ms -> s */

/* The generation budget a request gets. max_tokens is a CEILING, not a
 * target (#260/#382, the rule GLM and DeepSeek V4 already apply): the prompt
 * must fit with room for one token (none for a read-only logprobs request,
 * docs/systemone.md), and the budget is then clamped to what the context can hold.
 * Returns the budget, or -1 when the PROMPT does not fit. Refusing when
 * prompt + budget exceeded the context (#1641) turned the gateway's default
 * output budget -- 8192 here, the whole default context -- into a 400 on
 * every message of `coli chat` and on every request without max_tokens. */
static int qwen36_serve_budget(int np, int max_tok, int max_ctx, int read_only){
    if (np < 1) return -1;
    int room = max_ctx - np;
    if (read_only) return room < 0 ? -1 : (max_tok < room ? max_tok : room);
    if (room < 1) return -1;
    return max_tok > room ? room : max_tok;
}

/* The counters when a request's decoding began: PROF and DONE report its share. */
typedef struct { double s_disk, s_attn, s_moe, s_head, t0; } Q36ReqClock;

/* A request's prompt into the conversation the Model holds: its tokens and image,
 * the budget, ACCEPT, the prefix reuse and the pins, the prefill and its read-out.
 * 1 with the prompt's ids and the logits after it; 0 when the request ended here,
 * its ERROR written. serve_one and serve_mux start every request here. */
static int q36_serve_start(Model *m, ServeReq *q, int **ids_out, int *np_out, float **lo_out,
                           Q36ReqClock *clk){
    int *ids=NULL, np=0;
    encode_text(q->payload, &ids, &np);          /* payload is raw prompt text; qwen36 adds no BOS */
    if(g_pending_image.present){
        Cfg *vc = &m->c;
        if(!m->vis_ready){
            printf("ERROR %s this engine has no vision tower; images are not supported\n",q->id);
            fflush(stdout); q36_pending_image_clear(); free(ids); return 0;
        }
        unsigned long long want = (unsigned long long)g_pending_image.grid_h*g_pending_image.grid_w*
            vc->vis_in_ch*vc->vis_temporal*vc->vis_patch*vc->vis_patch*sizeof(float);
        if(g_pending_image.bytes!=want){
            printf("ERROR %s BAD_IMAGE bytes=%llu expected=%llu\n",q->id,g_pending_image.bytes,want);
            fflush(stdout); q36_pending_image_clear(); free(ids); return 0;
        }
        if(q36_vision_attach(m,(const float*)g_pending_image.patches,g_pending_image.grid_h,
                             g_pending_image.grid_w,ids,np)<0){
            printf("ERROR %s BAD_IMAGE the prompt and the grid disagree\n",q->id);
            fflush(stdout); q36_pending_image_clear(); free(ids); return 0;
        }
        q36_pending_image_clear();
    }
    int max_ctx = qwen36_max_ctx();
    int budget = qwen36_serve_budget(np, q->max_tok, max_ctx, q->logprobs > 0);
    if(budget < 0){
        printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",q->id,np,q->max_tok,max_ctx);
        fflush(stdout); free(ids); return 0;
    }
    if(budget < q->max_tok){
        fprintf(stderr,"[serve] max_tokens %d clamped to %d (context %d - prompt %d); raise Q36_MAXT for longer answers\n",
                q->max_tok, budget, max_ctx, np);
        q->max_tok = budget;
    }
    printf("ACCEPT %s %d\n",q->id,np); fflush(stdout);
    m->max_t = np + q->max_tok;
    /* Grow the cache BEFORE deciding, so the decision sees the state that will
     * actually be there: ensure_kv preserves both the rows and the record. */
    ensure_kv(m);
    /* A chat client resends the whole transcript every turn. If this prompt
     * begins with the ids the current state was built from, that state already
     * IS the state at those positions: prefill only the tail. Either the reused
     * positions are token-identical or nothing is reused -- there is no partial
     * case, because nothing here can rewind a state. COLI_KV_PREFIX=0 turns it
     * off for an A/B; COLI_PREFIX_LOG=1 reports the decision and its reason,
     * because "it did not get faster" is otherwise indistinguishable from
     * "reuse is not wired up". */
    int reuse = kv_prefix_off() ? 0 : kv_prefix_reuse(&m->kvp, ids, np);
    if (getenv("COLI_PREFIX_LOG")) {
        if (reuse)
            fprintf(stderr, "[PREFIX] reusing %d of %d prompt tokens (%.0f%%)\n",
                    reuse, np, 100.0 * reuse / np);
        else
            fprintf(stderr, "[PREFIX] no reuse: held=%d cap=%d prompt=%d%s%s\n",
                    m->kvp.len, m->kvp.cap, np, m->kvp.tainted ? " tainted" : "",
                    (m->kvp.len > 0 && m->kvp.len < np) ? " (diverged)" : "");
        fflush(stderr);
    }
    /* Con la lettura accesa il riuso arretra di un token: il predittore del
     * primo token fresco deve ricadere nel blocco che ricalcoliamo, altrimenti
     * quel token resta senza logprob ed e proprio quello che al chiamante
     * serve (il primo token dell'opzione). Costa una posizione. */
    /* La fotografia si prova SEMPRE, non solo quando il riuso vivo fallisce.
     * Altrimenti la prima opzione (che trova ancora lo stato del prompt e
     * quindi passa dal riuso normale) resterebbe senza i logit salvati, e il
     * suo primo token senza logprob: proprio il token che serve. Rimetterla
     * quando lo stato e gia quello costa una memcpy, non un prefill. */
    g_pin_use_logit = 0;
    {
        int pinned = pin_restore(m, ids, np);
        if (pinned) { reuse = pinned; g_pin_use_logit = 1; }
    }
    g_echo_k = q->logprobs; g_echo_id = q->id;
    if (!reuse) { reset_recurrent(m); m->kv_len = 0; }
    /* Per-REQUEST state, not per-process: without this the server keeps the
     * first request's prefill flag and expert-collection set forever, so
     * COLIBRI_RESIDENT=1 collects on request #1 and never again, and the
     * router EMA carries one conversation's history into the next. */
    m->first_step = 1;
    if (m->seen) memset(m->seen, 0, (size_t)m->c.n_layers * m->c.n_experts);
    if (m->momentum_logits)
        memset(m->momentum_logits, 0,
               (size_t)m->c.n_layers * m->c.n_experts * sizeof(float));
    /* `reuse` is the ABSOLUTE position of the first fresh token: attention and
     * the KV rows are position-indexed, so this has to be the real offset. */
    float *lo = step(m, ids + reuse, np - reuse, reuse);
    qt_stats_mark();
    if (q->pin) pin_save(m, ids, np, lo);
    clk->s_disk=m->t_disk; clk->s_attn=tm_sum(0)+tm_sum(1); clk->s_moe=tm_sum(2); clk->s_head=tm_sum(5);
    clk->t0=now_s();
    g_echo_k = 0; g_echo_id = NULL;   /* la lettura riguarda il prefill, non la decodifica */
    *ids_out=ids; *np_out=np; *lo_out=lo;
    return 1;
}

static void serve_one(Model *m, ServeReq *q){
    int *ids=NULL, np=0; float *lo=NULL; Q36ReqClock clk;
    if(!q36_serve_start(m,q,&ids,&np,&lo,&clk)) return;
    int gen=0, limited=1, forwards=1;   /* il prefill e' il primo forward */
    const double s_disk=clk.s_disk, s_attn=clk.s_attn, s_moe=clk.s_moe, s_head=clk.s_head;
    int eos_ids[4]; int n_eos=serve_eos_ids(eos_ids,4);
    double t0=clk.t0;
    unsigned char sbuf[16]; int sbn=0;
    Q36Spec spec; q36_spec_begin(m, &spec, ids, np);   /* COLI_LOOKUP=1: drafts, every token still sampled from the exact logits */
    for(int s=0;s<q->max_tok;s++){
        int tk = serve_sample(lo, m->c.vocab, q->temp, q->top_p);
        /* La coda si calcola PRIMA della free: dopo, lo non c'e piu. */
        char lptail[1024]; lptail[0]=0;
        if(q->logprobs>0) coli_logprob_tail(lptail,sizeof lptail,lo,m->c.vocab,tk,q->logprobs);
        free(lo); lo=NULL;
        int is_eos=0; for(int e=0;e<n_eos;e++) if(tk==eos_ids[e]) is_eos=1;
        if(is_eos){ limited=0; break; }
        unsigned char tmp[256]; int tn=0; decode_id_to_bytes(tk, tmp, &tn);
        if(q->logprobs>0){
            serve_data_lp(q->id,(char*)tmp,tn,lptail);
        } else {
            unsigned char chunk[256]; int cn=0; utf8_drain(sbuf,&sbn,tmp,tn,chunk,&cn);
            if(cn>0) serve_data(q->id,(char*)chunk,cn);
        }
        gen++;
        /* #1332: una guardata a stdin per token. Il costo e' una select con
         * timeout zero; il guadagno e' che il gateway smette di aspettare un
         * turno che nessuno vuole piu'. */
        if(serve_cancel_pending(q->id)){
            free(lo); lo=NULL;
            q36_spec_end(m, &spec); q36_spec_free(&spec);   /* the state of the tokens fed, drafts undone */
            if(sbn>0) serve_data(q->id,(char*)sbuf,sbn);
            printf("ERROR %s CANCELLED\n",q->id); fflush(stdout);
            free(ids);
            return;
        }
        /* The next logits are not needed after the final requested token.
         * serve_one() resets the recurrent/KV state for every request, so
         * stepping here would only run a full discarded decode pass. */
        if(s == q->max_tok - 1) break;
        lo = q36_spec_step(m, &spec, tk, np+s, q->max_tok-1-s);
    }
    q36_spec_end(m, &spec);
    forwards += (int)spec.forwards;
    if(sbn>0) serve_data(q->id,(char*)sbuf,sbn);   /* flush trailing partial UTF-8 */
    free(lo); free(ids);
    double dt=now_s()-t0;
    hits_emit(m);
    {
        double disk=m->t_disk-s_disk, moe=tm_sum(2)-s_moe;
        /* microsecond resolution: a tiny-fixture turn on a fast runner is under
         * a millisecond, and at %.3f every phase (and the wall) printed 0.000,
         * which the dashboard tests read as "not measured" (dev CI, 2026-09-14) */
        printf("PROF %.6f %d %d %.6f %.6f %.6f %.6f %.6f %llu\n", dt, np, gen,
               disk, 0.0, moe>disk?moe-disk:0.0, tm_sum(0)+tm_sum(1)-s_attn, tm_sum(5)-s_head,
               (unsigned long long)forwards);   /* contati, non dedotti: l'ultimo token non ne fa uno */
        fflush(stdout);
    }
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n",q->id,gen,
           dt>0?gen/dt:0.0,0.0,rss_gb(),np,limited);
    fflush(stdout);
    {char scope[96]; snprintf(scope, sizeof scope, "turn %s", q->id); q36_spec_report(&spec, scope);}
    q36_spec_free(&spec);
#ifdef COLI_VULKAN
    vk_report();   /* stderr: the wire protocol on stdout is untouched */
    q36c_report(m);
    vk_tier_turn(m, "turn");
#endif
}

/* ======================= Clef: DECIDE on the same engine ======================= *
 * Cloudflare's Clef (docs/clef.md) is Qwen3.8-27B post-trained with a joint schema
 * head: a container converted from it carries joint_head_config.json and
 * joint_head.safetensors next to its shards (tools/convert_qwen36.py copies them).
 * With them the engine still chats, and also answers DECIDE: the record is
 * rendered the way the reference renders it (clef_head.h), the backbone reads it
 * in one prefill, and the head scores every option from the final-normed hidden
 * state of every position. Without the two files none of this runs: g_clef stays
 * 0, CAPS says what it always said, and a DECIDE frame is not even parsed. */
#include "clef_head.h"

static ClefHead g_clef_head;
static int g_clef_max_len = CLEF_MAX_LENGTH;

static int q36_has_clef_head(const char *snap){
    char a[2048], b[2048];
    snprintf(a, sizeof a, "%s/joint_head_config.json", snap);
    snprintf(b, sizeof b, "%s/joint_head.safetensors", snap);
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    int both = fa && fb;
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    return both;
}

/* A tensor of joint_head.safetensors (st_init indexed it with the shards) as f32. */
static float *q36_clef_tensor(void *ctx, const char *name, int64_t numel){
    Model *m = (Model *)ctx;
    st_tensor *t = st_find(&m->S, name);
    if (!t || t->numel != numel || t->dtype > 2) return NULL;   /* BF16, F16, F32 */
    float *p = (float *)malloc((size_t)numel * sizeof(float));
    if (!p) return NULL;
    st_read_f32(&m->S, name, p, 0);
    return p;
}

/* The LM head's row for a token, at the precision on disk: the lexical option
 * vectors average them, and the dense copy the engine multiplies with may be
 * int8 or int4. */
static int q36_clef_lm_row(void *ctx, int id, float *out){
    Model *m = (Model *)ctx;
    int D = m->c.hidden;
    if (id < 0 || id >= m->c.vocab) return 0;
    if (m->lm_head.w) { memcpy(out, m->lm_head.w + (int64_t)id * D, (size_t)D * sizeof(float)); return 1; }
    if (m->lm_head.h) { f16_to_f32_bulk(m->lm_head.h + (int64_t)id * D, out, D); return 1; }
    char rn[QW_DENSE_NAME_MAX];
    st_tensor *t = st_find(&m->S, dense_resolve(m, "lm_head.weight", rn, sizeof rn));
    if (!t || t->numel != (int64_t)m->c.vocab * D || t->dtype > 2) return 0;
    int esz = st_dtype_esz(t->dtype);
    uint16_t *raw = (uint16_t *)malloc((size_t)D * esz);
    if (!raw) return 0;
    st_read_range_raw_cap(&m->S, t->fd, t->off + (int64_t)id * D * esz, (int64_t)D * esz,
                          raw, (int64_t)D * esz, 0, "clef lm_head row");
    if (t->dtype == 0) bf16_to_f32_bulk(raw, out, D);
    else if (t->dtype == 1) f16_to_f32_bulk(raw, out, D);
    else memcpy(out, raw, (size_t)D * sizeof(float));
    free(raw);
    return 1;
}

static int q36_clef_encode(void *ctx, const char *text, int **ids, int *count){
    (void)ctx;
    encode_text(text, ids, count);
    return 1;
}

/* joint_head_config.json -> the head's shape, checked against the backbone, then
 * every tensor. A broken head is a refusal at load, never a chat model that
 * quietly cannot decide. */
static void q36_clef_load(Model *m, const char *snap){
    char path[2048];
    snprintf(path, sizeof path, "%s/joint_head_config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > (1L << 20)) { fprintf(stderr, "[clef] %s: empty or larger than 1 MB\n", path); exit(1); }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "[clef] cannot read %s\n", path); exit(1); }
    buf[n] = 0; fclose(f);
    jval *cfg = json_parse_checked(buf);
    free(buf);
    if (!cfg || cfg->t != J_OBJ) { fprintf(stderr, "[clef] %s is not a JSON object\n", path); exit(1); }
    ClefHead *h = &g_clef_head;
    memset(h, 0, sizeof(*h));
    #define CLEF_KEY(key, field, lo, hi) do { jval *v = json_get(cfg, key); \
        if (!v || v->t != J_NUM || v->num != floor(v->num) || v->num < (lo) || v->num > (hi)) { \
            fprintf(stderr, "[clef] %s: \"%s\" missing or outside %d..%d -- refusing\n", path, key, lo, hi); exit(1); } \
        h->field = (int)v->num; } while (0)
    CLEF_KEY("hidden_size", hidden, 1, 65536);
    CLEF_KEY("width", width, 1, 16384);
    CLEF_KEY("routing_layers", routing_layers, 0, 64);
    CLEF_KEY("layers", layers, 0, 64);
    CLEF_KEY("heads", heads, 1, 256);
    CLEF_KEY("feedforward", feedforward, 1, 65536);
    #undef CLEF_KEY
    json_free(cfg);
    if (h->hidden != m->c.hidden || h->width % h->heads) {
        fprintf(stderr, "[clef] head hidden_size %d / width %d / heads %d do not fit a %d-wide backbone -- refusing\n",
                h->hidden, h->width, h->heads, m->c.hidden);
        exit(1);
    }
    char err[256];
    if (!clef_head_load(h, m, q36_clef_tensor, err, sizeof err)) {
        fprintf(stderr, "[clef] joint_head.safetensors: %s -- refusing\n", err);
        exit(1);
    }
    const char *e = getenv("COLI_CLEF_MAX_LEN");
    if (e && *e) {
        g_clef_max_len = atoi(e);
        if (g_clef_max_len < 1) { fprintf(stderr, "[clef] COLI_CLEF_MAX_LEN must be positive\n"); exit(1); }
    }
    g_clef = 1;
    if (g_clef_max_len > g_q36_default_ctx) g_q36_default_ctx = g_clef_max_len;
    fprintf(stderr, "[clef] decision head: %d routing + %d decoder layers, width %d, %d heads; "
            "a record reads up to %d tokens (the engine's context is %d)\n",
            h->routing_layers, h->layers, h->width, h->heads, g_clef_max_len, qwen36_max_ctx());
}

/* One record: rendered, one prefill that keeps every position's hidden state,
 * the head, a softmax per question. `keep` (the --records test mode) leaves the
 * rendered input in *last for the caller to print. */
static int q36_clef_decide(Model *m, const DecideRecord *rec, DecideAnswer *answers, int *input_tokens,
                           ClefInput *last, char *err, size_t cap){
    Cfg *c = &m->c;
    ClefInput in;
    if (!clef_render(rec, q36_clef_encode, NULL, g_clef_max_len, &in, err, cap)) return 0;
    int max_ctx = qwen36_max_ctx();
    if (in.n > max_ctx) {
        int n = in.n;
        clef_input_free(&in);
        return decide_fail(err, cap, "record: %d tokens do not fit this engine's context of %d, "
                           "set by Q36_MAXT (Clef itself reads up to %d; unset, the context is that)",
                           n, max_ctx, g_clef_max_len);
    }
    float *hidden = (float *)malloc((size_t)in.n * c->hidden * sizeof(float));
    double **logits = (double **)calloc((size_t)in.n_q, sizeof(double *));
    int ok = hidden && logits;
    for (int q = 0; ok && q < in.n_q; q++) ok = (logits[q] = (double *)calloc((size_t)in.q[q].n, sizeof(double))) != NULL;
    if (!ok) snprintf(err, cap, "out of memory for %d positions", in.n);
    if (ok) {
        /* a decision reads from position 0, with no picture and no reuse: the
         * attention rows and the recurrence it leaves behind are recorded in kvp
         * by step(), so the next chat turn or snapshot checks them as usual */
        q36_vision_detach(m);
        m->max_t = in.n;
        ensure_kv(m);
        reset_recurrent(m);
        m->kv_len = 0;
        m->first_step = 1;
        if (m->seen) memset(m->seen, 0, (size_t)c->n_layers * c->n_experts);
        if (m->momentum_logits) memset(m->momentum_logits, 0, (size_t)c->n_layers * c->n_experts * sizeof(float));
        g_hidden_sink = hidden;
#ifdef COLI_VULKAN
        double tb = decide_now_ms();
#endif
        float *lo = step(m, in.ids, in.n, 0);
        g_hidden_sink = NULL;
        free(lo);
#ifdef COLI_VULKAN
        double th = decide_now_ms();
#endif
        ok = clef_head_forward(&g_clef_head, hidden, &in, q36_clef_lm_row, m, logits, err, cap);
#ifdef COLI_VULKAN
        /* where a decision's time goes, for the device's measurements (docs/vulkan.md) */
        fprintf(stderr, "[clef] %d tokens: backbone %.1f ms, head %.1f ms\n", in.n, th - tb, decide_now_ms() - th);
#endif
    }
    for (int q = 0; ok && q < in.n_q; q++) {
        ok = clef_answer(&in.q[q], logits[q], &answers[q]);
        answers[q].tokens = in.n;
        answers[q].state_tokens = in.state_tokens;
        answers[q].state_dropped = in.state_tokens - in.state_used;
        if (!ok) snprintf(err, cap, "out of memory");
    }
    if (logits) for (int q = 0; q < in.n_q; q++) free(logits[q]);
    free(logits); free(hidden);
    *input_tokens = in.n;
    if (ok && last) *last = in;
    else clef_input_free(&in);
    return ok;
}

/* DECIDE <id>: DECISION + DONE, or ERROR <id> DECIDE_INVALID|DECIDE_FAILED. */
static void clef_serve_one(Model *m, ServeReq *q){
    char reason[1024] = "";
    DecideRecord rec;
    if (!decide_record_parse(q->payload, &rec, reason, sizeof reason)) {
        decide_write_refusal(stdout, q->id, "DECIDE_INVALID", reason);
        return;
    }
    DecideAnswer *answers = (DecideAnswer *)calloc((size_t)rec.n_questions, sizeof(DecideAnswer));
    int tokens = 0;
    double started = decide_now_ms();
    int ok = answers && q36_clef_decide(m, &rec, answers, &tokens, NULL, reason, sizeof reason);
    double elapsed = decide_now_ms() - started;
    if (!ok) {
        if (!answers) snprintf(reason, sizeof reason, "out of memory");
        decide_write_refusal(stdout, q->id, decide_is_request_error(reason) ? "DECIDE_INVALID" : "DECIDE_FAILED",
                             reason);
    } else {
        size_t length = 0;
        char *json = decide_answers_json(&rec, answers, tokens, elapsed, &length);
        if (!json) coli_serve_write_error(stdout, q->id, "DECIDE_FAILED out of memory");
        else {
            ColiServeDone done = {0, elapsed > 0 ? tokens / (elapsed / 1e3) : 0.0, 0.0, rss_gb(), tokens, 0};
            coli_serve_write_decision(stdout, q->id, json, length);
            coli_serve_write_done(stdout, q->id, &done);
            free(json);
        }
    }
    fprintf(stderr, "[clef] DECIDE %s: %d question(s), %d tokens, %.1f ms\n", q->id, rec.n_questions, tokens, elapsed);
#ifdef COLI_VULKAN
    vk_report();   /* stderr: the device's share of the decision */
    q36c_report(m);
#endif
    decide_answers_free(answers, answers ? rec.n_questions : 0);
    free(answers);
    decide_record_free(&rec);
}

static char *q36_read_file(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = n >= 0 ? (char *)malloc((size_t)n + 1) : NULL;
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = 0; fclose(f);
    return buf;
}

/* Test modes (tests/test_clef_tiny.py), the path the serve loop takes minus the
 * wire: CLEF_TOKENIZE=<json array of strings> prints their ids; CLEF_RECORDS=
 * <json array of {"payload": record}> prints one DECISION object per record, and
 * with CLEF_IDS=1 the rendered input (ids, spans, option order) beside it. */
static int q36_clef_test_modes(Model *m){
    const char *tok = getenv("CLEF_TOKENIZE"), *recs = getenv("CLEF_RECORDS");
    if (tok) {
        char *text = q36_read_file(tok);
        jval *root = text ? json_parse_checked(text) : NULL;
        if (!root || root->t != J_ARR) { fprintf(stderr, "%s: expected a JSON array of strings\n", tok); return 1; }
        printf("[");
        for (int i = 0; i < root->len; i++) {
            if (root->kids[i]->t != J_STR) { fprintf(stderr, "%s: entry %d is not a string\n", tok, i); return 1; }
            int *ids = NULL, n = 0;
            encode_text(root->kids[i]->str, &ids, &n);
            printf("%s[", i ? "," : "");
            for (int t = 0; t < n; t++) printf("%s%d", t ? "," : "", ids[t]);
            printf("]");
            free(ids);
        }
        printf("]\n");
        json_free(root); free(text);
        return 0;
    }
    if (!g_clef) { fprintf(stderr, "CLEF_RECORDS: this checkpoint has no decision head\n"); return 1; }
    int dump = getenv("CLEF_IDS") && getenv("CLEF_IDS")[0] == '1';
    char *text = q36_read_file(recs);
    jval *root = text ? json_parse_checked(text) : NULL;
    if (!root || root->t != J_ARR) { fprintf(stderr, "%s: expected a JSON array of records\n", recs); return 1; }
    for (int i = 0; i < root->len; i++) {
        jval *payload = json_get(root->kids[i], "payload");
        char reason[1024] = "";
        DecideRecord rec;
        DecideBuf out = {0};
        if (!payload || payload->t != J_STR || !decide_record_parse(payload->str, &rec, reason, sizeof reason)) {
            if (!payload || payload->t != J_STR) snprintf(reason, sizeof reason, "record %d has no string payload", i);
            decide_buf_put(&out, "{\"error\":", 9); decide_buf_string(&out, reason); decide_buf_put(&out, "}", 1);
            printf("%s\n", out.data); fflush(stdout); free(out.data);
            continue;
        }
        DecideAnswer *answers = (DecideAnswer *)calloc((size_t)rec.n_questions, sizeof(DecideAnswer));
        ClefInput in; memset(&in, 0, sizeof in);
        int tokens = 0;
        double t0 = decide_now_ms();
        int ok = answers && q36_clef_decide(m, &rec, answers, &tokens, &in, reason, sizeof reason);
        double ms = decide_now_ms() - t0;
        if (!ok) {
            decide_buf_put(&out, "{\"error\":", 9); decide_buf_string(&out, reason); decide_buf_put(&out, "}", 1);
        } else {
            size_t len = 0;
            char *json = decide_answers_json(&rec, answers, tokens, ms, &len);
            if (json && len > 1) {
                decide_buf_put(&out, json, len - 1);            /* reopen the object */
                if (dump) {
                    decide_buf_put(&out, ",\"input_ids\":[", 14);
                    for (int t = 0; t < in.n; t++) decide_buf_printf(&out, "%s%d", t ? "," : "", in.ids[t]);
                    decide_buf_printf(&out, "],\"state_tokens\":%d,\"state_used\":%d,\"spans\":[",
                                      in.state_tokens, in.state_used);
                    for (int q = 0; q < in.n_q; q++) {
                        const ClefQuestion *cq = &in.q[q];
                        decide_buf_printf(&out, "%s{\"type\":%d,\"question\":[%d,%d],\"options\":[",
                                          q ? "," : "", cq->type, cq->qs, cq->qe);
                        for (int k = 0; k < cq->n; k++) decide_buf_printf(&out, "%s[%d,%d]", k ? "," : "", cq->os[k], cq->oe[k]);
                        decide_buf_put(&out, "],\"order\":[", 11);
                        for (int k = 0; k < cq->n; k++) decide_buf_printf(&out, "%s%d", k ? "," : "", cq->record_index[k]);
                        decide_buf_put(&out, "]}", 2);
                    }
                    decide_buf_put(&out, "]", 1);
                }
                decide_buf_put(&out, "}", 1);
            }
            free(json);
        }
        printf("%s\n", out.data ? out.data : "{\"error\":\"out of memory\"}");
        fflush(stdout);
        free(out.data);
#ifdef COLI_VULKAN
        vk_report();
        q36c_report(m);
#endif
        clef_input_free(&in);
        decide_answers_free(answers, answers ? rec.n_questions : 0);
        free(answers);
        decide_record_free(&rec);
    }
    json_free(root); free(text);
    return 0;
}

/* ---- several conversations at once (KV_SLOTS>1) -------------------------------
 * The gateway's cache slots, each a conversation with a state of its own (Q36Seq).
 * A SUBMIT on a free slot starts its request at once through q36_serve_start, on
 * that slot's state: its prefix reuse and pins work as a lone serve's. Then every
 * step picks the next token of each active request and runs one forward over a row
 * of each (q36_step_rows): the matrices and the experts are read once for all of
 * them. A request's frames are a lone request's; they interleave by id. Nothing
 * drafts (speculation follows one conversation). STOP and CANCEL end a request as
 * serve_one ends it, with ERROR CANCELLED. A Clef DECIDE runs at once on its slot. */
static int g_q36_mux_slots = 1;
static Q36Seq *g_q36_mux_seq;   /* [slots]: the conversations the Model does not hold */
static int g_q36_mux_cur;       /* the slot the Model holds, -1 when every one is parked */

typedef struct {
    ServeReq q;
    int active, cancel, limited, forwards;
    int *ids, np, gen;
    float *lo;                  /* the logits the next pick reads */
    Q36ReqClock clk;
    unsigned char sbuf[16]; int sbn;
} Q36MuxReq;

static void q36_mux_bind(Model *m, int slot){
    if (g_q36_mux_cur == slot) return;
    if (g_q36_mux_cur >= 0) q36_seq_swap(m, &g_q36_mux_seq[g_q36_mux_cur]);   /* the held one back */
    if (slot >= 0) q36_seq_swap(m, &g_q36_mux_seq[slot]);
    g_q36_mux_cur = slot;
}

/* The whole context's KV for every slot, before READY: slot 0 is the Model's own. */
static int q36_mux_alloc(Model *m){
    int n = g_q36_mux_slots;
    m->max_t = qwen36_max_ctx(); ensure_kv(m);
    g_q36_mux_seq = calloc((size_t)n, sizeof *g_q36_mux_seq);
    if (!g_q36_mux_seq || !m->K) return 0;
    for (int i = 1; i < n; i++) if (!q36_seq_alloc(m, &g_q36_mux_seq[i])) return 0;
    g_q36_mux_cur = 0;
    return 1;
}

/* A request's end, as serve_one ends one. */
static void q36_mux_finish(Model *m, Q36MuxReq *r){
    free(r->lo); r->lo = NULL; free(r->ids); r->ids = NULL; r->active = 0;
    if (g_q36_mux_cur != r->q.slot) q36_mux_bind(m, r->q.slot);
    /* an image turn's rope positions go with the turn, as q36_vision_detach drops them */
    free(m->mpos); m->mpos = NULL; m->mpos_len = m->rope_delta = 0;
    if (r->sbn > 0) serve_data(r->q.id, (char *)r->sbuf, r->sbn);   /* the trailing partial UTF-8 */
    if (r->cancel) { printf("ERROR %s CANCELLED\n", r->q.id); fflush(stdout); return; }
    double dt = now_s() - r->clk.t0;
    hits_emit(m);
    {
        double disk = m->t_disk - r->clk.s_disk, moe = tm_sum(2) - r->clk.s_moe;
        printf("PROF %.6f %d %d %.6f %.6f %.6f %.6f %.6f %llu\n", dt, r->np, r->gen,
               disk, 0.0, moe > disk ? moe - disk : 0.0, tm_sum(0) + tm_sum(1) - r->clk.s_attn,
               tm_sum(5) - r->clk.s_head, (unsigned long long)r->forwards);
        fflush(stdout);
    }
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", r->q.id, r->gen,
           dt > 0 ? r->gen / dt : 0.0, 0.0, rss_gb(), r->np, r->limited);
    fflush(stdout);
}

/* The next token of an active request, as serve_one's loop picks and sends it: 1
 * with the token when the request goes on, 0 when it ended. */
static int q36_mux_pick(Model *m, Q36MuxReq *r, const int *eos_ids, int n_eos, int *tk_out){
    if (r->cancel || r->gen >= r->q.max_tok) { q36_mux_finish(m, r); return 0; }
    int tk = serve_sample(r->lo, m->c.vocab, r->q.temp, r->q.top_p);
    char lptail[1024]; lptail[0] = 0;
    if (r->q.logprobs > 0) coli_logprob_tail(lptail, sizeof lptail, r->lo, m->c.vocab, tk, r->q.logprobs);
    free(r->lo); r->lo = NULL;
    for (int e = 0; e < n_eos; e++) if (tk == eos_ids[e]) { r->limited = 0; q36_mux_finish(m, r); return 0; }
    unsigned char tmp[256]; int tn = 0; decode_id_to_bytes(tk, tmp, &tn);
    if (r->q.logprobs > 0) serve_data_lp(r->q.id, (char *)tmp, tn, lptail);
    else {
        unsigned char chunk[256]; int cn = 0; utf8_drain(r->sbuf, &r->sbn, tmp, tn, chunk, &cn);
        if (cn > 0) serve_data(r->q.id, (char *)chunk, cn);
    }
    r->gen++;
    /* the next logits are not needed after the last requested token (serve_one) */
    if (r->gen >= r->q.max_tok) { q36_mux_finish(m, r); return 0; }
    *tk_out = tk; return 1;
}

static void serve_mux(Model *m){
    int n = g_q36_mux_slots, V = m->c.vocab, input_eof = 0;
    Q36MuxReq *rq = calloc((size_t)n, sizeof *rq);
    Q36Row *rows = malloc((size_t)n * sizeof *rows);
    int *tok = malloc((size_t)n * sizeof(int)), *who = malloc((size_t)n * sizeof(int));
    if (!rq || !rows || !tok || !who) { fprintf(stderr, "[serve] out of memory\n"); exit(1); }
    int eos_ids[4]; int n_eos = serve_eos_ids(eos_ids, 4);
    unsigned long long steps = 0, nrows = 0;
    fprintf(stderr, "[qwen36] serving %d conversations at once (KV_SLOTS)\n", n);
    for (;;) {
        int active = 0; for (int i = 0; i < n; i++) active += rq[i].active;
        /* idle: wait for a command; decoding: take one only when one is there */
        if (!input_eof && (!active || coli_serve_stdin_ready())) {
            ServeReq q = {0};
            int r = serve_read_req(&q);
            if (r < 0) input_eof = 1;
            else if (r == 4) {
                for (int i = 0; i < n; i++) if (rq[i].active && !strcmp(rq[i].q.id, q.id)) rq[i].cancel = 1;
            } else if (r == 2 || r == 3) {
                int bad = q.slot < 0 || q.slot >= n;
                if (bad || rq[q.slot].active) {
                    printf("ERROR %s %s\n", q.id, bad ? "invalid cache slot" : "SLOT_BUSY"); fflush(stdout);
                } else if (r == 3) {
                    q36_mux_bind(m, q.slot); clef_serve_one(m, &q);
                } else {
                    Q36MuxReq *t = &rq[q.slot];
                    memset(t, 0, sizeof *t); t->q = q; q.payload = NULL;
                    q36_mux_bind(m, t->q.slot);
                    int ok = q36_serve_start(m, &t->q, &t->ids, &t->np, &t->lo, &t->clk);
                    /* the image's rows are the prompt's: no decode row reads them; its rope
                     * positions stay with the conversation until the turn ends */
                    free(m->vis_rows); free(m->vis_map); m->vis_rows = NULL; m->vis_map = NULL;
                    m->vis_rows_n = m->vis_map_len = 0;
                    if (ok) { t->active = 1; t->limited = 1; t->forwards = 1; }
                    else { free(m->mpos); m->mpos = NULL; m->mpos_len = m->rope_delta = 0; }
                    free(t->q.payload); t->q.payload = NULL;
                    if (ok) emap_emit(m);
                }
                free(q.payload);
            }
        }
        active = 0; for (int i = 0; i < n; i++) active += rq[i].active;
        if (!active) { if (input_eof) break; continue; }
        int S = 0, ended = 0;
        for (int i = 0; i < n; i++) if (rq[i].active) {
            int tk;
            if (!q36_mux_pick(m, &rq[i], eos_ids, n_eos, &tk)) { ended = 1; continue; }
            rows[S] = (Q36Row){&g_q36_mux_seq[i], rq[i].np + rq[i].gen - 1}; tok[S] = tk; who[S] = i; S++;
        }
        if (S) {
            /* every conversation parked: the rows read theirs from g_q36_mux_seq */
            q36_mux_bind(m, -1);
            float *lo = q36_step_rows(m, rows, tok, S);
            steps++; nrows += (unsigned long long)S;
            for (int s = 0; s < S; s++) {
                Q36MuxReq *t = &rq[who[s]];
                t->lo = falloc(V); memcpy(t->lo, lo + (int64_t)s * V, (size_t)V * sizeof(float));
                t->forwards++;
            }
            free(lo);
        }
        if (ended) {
#ifdef COLI_VULKAN
            vk_report();
            vk_tier_turn(m, "turn");
#endif
            emap_emit(m);
        }
    }
    fprintf(stderr, "[qwen36] KV_SLOTS=%d: %llu decode steps, %llu rows (%.2f a step)\n", n, steps, nrows,
            steps ? (double)nrows / (double)steps : 0.0);
    q36_mux_bind(m, 0);
    free(rq); free(rows); free(tok); free(who);
}

static void serve_loop(Model *m){
    coli_serve_binary_mode();
    setvbuf(stdin,NULL,_IONBF,0);
    if(g_q36_mux_slots>1 && !q36_mux_alloc(m)){
        fprintf(stderr,"[serve] unable to allocate the state of %d conversations (KV_SLOTS)\n",g_q36_mux_slots); return;
    }
    fputs("\x01\x01READY\x01\x01\n",stdout);
    /* fra READY e STAT: il gateway lo legge nella stretta di mano, quindi sa che
     * modalita' serve prima della prima richiesta (docs/serve_protocol.md) */
    /* Clef answers DECIDE too, and still chats: decide=1 without chat=0. Its
     * records come in the raw form (decide_record=raw, decide_serve.h). */
    printf("CAPS vision=%d%s\n",m->vis_ready?1:0,g_clef?" decide=1 decide_record=raw":"");
    printf("STAT 0 0.00 0.0 %.2f\n",rss_gb());
    fflush(stdout);
    emap_emit(m);          /* dopo READY e STAT: il boot reader legge STAT dopo il sentinel */
    fflush(stdout);
    if(g_q36_mux_slots>1){ serve_mux(m); return; }
    for(;;){
        ServeReq q={0}; int r;
        do r=serve_read_req(&q); while(r==0||r==4);
        if(r<0) return;
        /* Resend the grid after EVERY turn, not only after READY: at boot the
         * expert cache is empty by definition, and that cold snapshot stayed
         * the only one the dashboard ever saw -- all grey, RAM 0, everything
         * on disk, forever. HITS was already per turn, which is why the white
         * "routed now" flash worked while the residency colour never moved.
         * inkling.c, kimi_k3.c, qwen38.c, deepseek_v41.c and colibri.c
         * already do this. */
        if(r==2){ serve_one(m,&q); q36_vision_detach(m); free(q.payload); emap_emit(m); }
        if(r==3){ clef_serve_one(m,&q); free(q.payload); }
    }
}

/* Warmstart body, lifted out of main so a test can drive it against an
 * in-memory model without a container (the QT_NO_WARMSTART check stays at the
 * call site). expert_is_int4 is what main probed from the on-disk expert size.
 */
static void tier_warmstart(Model *m, int expert_is_int4) {
    /* Plan the set (heat order), then load+stage IN PARALLEL. The
     * load path is thread-safe: expert_get locks the layer cache
     * (g_pilot_mx), st_read_raw uses pread; entries are unique. */
    double t0 = now_s();
    int cap_total = m->c.n_layers * m->c.n_experts;
    int *wpl = malloc((size_t)cap_total*sizeof(int));
    int *wpe = malloc((size_t)cap_total*sizeof(int));
    int wn = qt_plan_fill(wpl, wpe, cap_total);
    /* Load ALL experts into RAM, not just the planned (VRAM) set:
     * otherwise the first touch of a CPU-fallback expert triggers a
     * ~12 ms container read in the middle of decode (measured: 139
     * ms/token on a single-GPU run). Planned ones also go to VRAM. */
    uint8_t *planned = calloc((size_t)cap_total, 1);
    for (int i = 0; i < wn; i++) planned[wpl[i]*m->c.n_experts + wpe[i]] = 1;
    int keep8 = getenv("COLI_KEEP_INT8") != NULL;
    #pragma omp parallel for schedule(dynamic, 16)
    for (int gi = 0; gi < cap_total; gi++) {
        int l = gi / m->c.n_experts, eidw = gi % m->c.n_experts;
        Slot *e; expert_get(m, l, eidw, &e);
        /* int4: i puntatori impacchettati; int8: i pesi stessi. Prima
         * qui si esigeva e->g4, che su un container int8 e' NULL: la
         * promozione non partiva mai e il budget restava riservato a
         * vuoto (#1331). */
        const uint8_t *wg = expert_is_int4 ? e->g4 : (const uint8_t *)e->g;
        const uint8_t *wu = expert_is_int4 ? e->u4 : (const uint8_t *)e->u;
        const uint8_t *wd = expert_is_int4 ? e->d4 : (const uint8_t *)e->d;
        if (planned[gi]) {
            /* Reported even when wg is NULL: the tier reserved budget for
             * this expert in qt_plan_fill, and a planned expert that is never
             * reported keeps that reservation forever. With no weights the
             * tier hands the bytes back instead of uploading. */
            qt_note_planned(l, eidw, wg, wu, wd, e->gs, e->us, e->ds);
            if (!wg) continue;
            /* The staging copy is done. On an int4 container the int8 copy
             * can go RIGHT AWAY so it never shows up in peak RSS: g4/u4/d4
             * stay as the source of truth, and slot_ensure_int8() rebuilds
             * the int8 block on an LFRU eviction with no container access.
             * An int8 container has no second copy -- e->g4 is NULL -- so
             * freeing e->g there both dangles the pointer qt_note_planned
             * just stored (any later stage() reads freed memory) and leaves
             * slot_ensure_int8() unable to bring the expert back: it returns
             * early without g4, and the CPU fallback in the decode loop then
             * dereferences NULL. Keep it (#1341). COLI_KEEP_INT8 keeps its
             * meaning for int4.
             * The condition is ownership, not format: do not free what was
             * just handed over. int4 handed g4 (wg != e->g), so the int8
             * copy is spare; int8 handed e->g itself, so it stays. A future
             * format that also aliases e->g is then correct without anyone
             * remembering to extend a format check here. */
            if (!keep8 && e->g && wg != (const uint8_t *)e->g) { free(e->g); e->g = e->u = e->d = NULL; }
        }
    }
    qt_fill_wait();
    free(wpl); free(wpe); free(planned);
    /* The parenthesis used to read "int8 only for non-residents", which was
     * true only while the int8 copy of every resident was freed. Since #1341
     * that free is int4-only: on an int8 container every resident keeps its
     * weights, so the RSS saving the old line implied does not exist there.
     * Say which container this is instead of promising a saving the reader
     * will not see. */
    fprintf(stderr, "[qtier] warmstart (parallel): all %d experts in RAM (%s), %d in VRAM -- %.1f s\n",
            cap_total,
            expert_is_int4 ? "int8 copy dropped for residents, kept for non-residents"
                           : "int8 container: all experts keep their weights in RAM",
            wn, now_s()-t0);
}

#ifdef COLI_VULKAN
/* ---- the Vulkan routed-expert tier: start ----------------------------------------
 * After the weights and the device: how the experts sit in the slots (the shared
 * kernel's planar int4, or the int8 copy of an int4 / int8 container, gate/up and
 * down each with their own scales), the dense matrices still to come to the device,
 * the RAM the expert cache may still take; the history this engine keeps for the
 * tier (route_trace.h's .coli_usage, beside the snapshot or COLI_USAGE), and from it
 * a warm start, read through temporary slots in parallel. */
static size_t vk_qw_bytes(const QW *w) {
    return w->q4 ? (size_t)w->O * (w->I / 2) + (size_t)w->O * (w->I / 64) * sizeof(float)
         : w->q  ? (size_t)w->O * w->I + (size_t)w->O * sizeof(float)
         : w->w  ? (size_t)w->O * w->I * sizeof(float) : 0;
}
/* What still goes to the device after the tier starts: not a matrix placed already (the
 * chain's setup at start, or the dense weights on the device only) nor one that stays
 * on the CPU (a partial chain's CPU layers and head: vk_off). */
static size_t vk_qw_still(const QW *w) { return w->vk || w->vk_off ? 0 : vk_qw_bytes(w); }
static size_t vk_dense_bytes(Model *m) {
    if (!g_vk_dense && !g_vk_chain) return 0;
    if (coli_vk_dense_device_only()) return 0;   /* placed already (q36_dho_start): the free memory the tier reads counts them */
    size_t b = vk_qw_still(&m->lm_head);
    for (int i = 0; i < m->c.n_layers; i++) {
        Layer *L = &m->L[i];
        const QW *w[] = {&L->q, &L->k, &L->v, &L->o, &L->gate, &L->sh_g, &L->sh_u, &L->sh_d,
                         &L->dn_qkv, &L->dn_z, &L->dn_out};
        for (size_t k = 0; k < sizeof w / sizeof *w; k++) b += vk_qw_still(w[k]);
    }
    return b;
}
/* Is the expert in this layer's RAM cache now (the tier's balance asks)? */
static int vk_in_ram(void *ctx, int layer, int e) {
    Model *m = ctx;
    pthread_mutex_lock(&g_pilot_mx);
    int r = slot_indexed(m, layer, e) != NULL;
    pthread_mutex_unlock(&g_pilot_mx);
    return r;
}
/* After the device, the chain, a qpack container and the CUDA tier are decided, before
 * the expert tier sizes its budget: the dense matrices go up now, so the tier sees them
 * placed. The CUDA tier and a qpack container keep the host copies (the dense part
 * stays theirs or the CPU's). */
/* With the fit of a partial chain (q36c_start) only the N layers' matrices (and the head's
 * when it goes up) count: the CPU's are vk_off. The chain's setup then drops each layer's
 * host copies once the layer is on the device (q36c_place, then q36_dho_placed). */
static void q36_dho_placed(void) {
    if (coli_vk_device_integrated() && coli_vk_imported_bytes())
        fprintf(stderr, "[VK] qwen36: dense rows shared with the integrated GPU, %.1f MiB read in place; "
                        "the imported pages stay alive as the only weight copy\n", coli_vk_imported_bytes() / 1048576.0);
    coli_vk_dense_host_placed("qwen36", "the embedding (its rows are gathered on the CPU), the DeltaNet's a/b rows and "
                              "the shared expert's gate vector, norms, the vision tower, any imported pages");
}
static void q36_dho_start(Model *m) {
    if (!g_vk_ready) return;
    size_t bytes = 0; int n = 0;
    q36_dho_each(m, q36_dho_count, &bytes, &n);
    if (!coli_vk_dense_host_decide("qwen36", (g_vk_chain || g_vk_dense) && n > 0 && !qt_ready() && !qq_active(), bytes))
        return;
    g_q36_dho_model = m;
    g_vk_dense = 1;   /* the steps the chain declines run their matrices on the device too: the CPU has none */
    if (g_q36c_fit_on && g_vk_chain) return;   /* the chain's setup drops them layer by layer (q36c_place) */
    bytes = 0; n = 0;
    q36_dho_each(m, q36_dho_drop, &bytes, &n);
    q36_dho_placed();
}

/* The tier's streaming (a big prompt chunk's cold experts on the device): an expert's
 * bytes as the CPU path would have them, its RAM cache or the disk, held until the tier
 * has copied them. */
static int vk_load(void *ctx, int layer, int e, VktExpertSrc *src, void **h) {
    Model *m = ctx;
    Slot *s = expert_hold(m, layer, e);
    if (!s) return 0;
    if (!xf_mode(m)) slot_ensure_int8(m, s);
    *src = vk_slot_src(m, s); *h = s;
    return 1;
}
static void vk_release(void *ctx, void *h) { (void)ctx; slot_release((Slot *)h); }
static void vk_tier_start(Model *m, const char *snap, int cap, int expert_is_int4, int expert_mixed) {
    Cfg *c = &m->c;
    if (!g_vk_ready || c->n_experts == 0 || qq_active()) return;
    if (qt_ready()) { fprintf(stderr, "[VK] tier qwen36: the CUDA expert tier is on and wins; the Vulkan tier stays off\n"); return; }
    const char *on = getenv("COLI_VK_TIER");
    if (on && *on == '0') return;
    int gs = c->expert_gs, dgs = down_gs_of(c);
    VktFmt gu, dn;
    if (xf_mode(m)) { gu = (VktFmt){VKT_SRC_I4U_PLANAR64, 64}; dn = gu; }
    else if (expert_mixed) {   /* int4 gate/up unpacked to int8 in the slot, int8 down */
        gu = (VktFmt){gs ? VKT_SRC_I8_AS_I4_GS : VKT_SRC_I8_AS_I4_ROW, gs};
        dn = (VktFmt){dgs ? VKT_SRC_I8_GS : VKT_SRC_I8_ROW, dgs};
    } else if (expert_is_int4) {
        gu = (VktFmt){gs ? VKT_SRC_I8_AS_I4_GS : VKT_SRC_I8_AS_I4_ROW, gs}; dn = gu;
    } else {
        gu = (VktFmt){gs ? VKT_SRC_I8_GS : VKT_SRC_I8_ROW, gs}; dn = gu;
    }
    int64_t ng = (int64_t)c->inter * c->hidden;
    size_t slot = (size_t)(xf_mode(m) ? (2 * ng + ng) / 2 : 3 * ng) +
                  (size_t)(2 * scale_count_gu(c) + scale_count_d(c)) * sizeof(float);
    VktConfig vc = {.engine = "qwen36", .layers = c->n_layers, .experts = c->n_experts,
                    .hidden = c->hidden, .inter = c->inter, .topk = c->topk,
                    .gate_up = gu, .down = dn, .act = VKT_ACT_SWIGLU,
                    .max_rows = QWEN_VK_ROWS * c->topk,
                    .ram_reserve = slot * (size_t)cap * (size_t)c->n_layers,
                    .dense_bytes = vk_dense_bytes(m),
                    .in_ram = vk_in_ram, .ram_ctx = m,
                    .load = vk_load, .release = vk_release, .load_ctx = m};
    rt_init("qwen36", c->n_layers, c->n_experts);
    rt_drop_row(c->n_layers);   /* no MTP row */
    const char *up = getenv("COLI_USAGE");
    if (up && *up) snprintf(g_vk_usage, sizeof g_vk_usage, "%s", up);
    else snprintf(g_vk_usage, sizeof g_vk_usage, "%s/.coli_usage", snap);
    int64_t h = rt_load(g_vk_usage);
    if (h > 0) fprintf(stderr, "[USAGE] expert history: %lld selections (%s)\n", (long long)h, g_vk_usage);
    atexit(coli_vk_shutdown);   /* before vkt_init, which makes the expert batch's pipelines and can still refuse (no room): the device goes at exit either way, after the tier's teardown */
    if (!vkt_init(&vc, rt_counts_all())) { rt_destroy(); return; }
    m->vk_hist = 1;
    atexit(vkt_shutdown);
    int all = c->n_layers * c->n_experts;
    int *pl = malloc((size_t)all * sizeof(int)), *pe = malloc((size_t)all * sizeof(int));
    const char *warm = getenv("COLI_VK_TIER_WARM");   /* 0: no warm start, the tier fills as experts pass by */
    int n = pl && pe && !(warm && *warm == '0') ? vkt_plan(pl, pe, all) : 0;
    if (n > 0) {
        double t0 = now_s();
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < n; i++) {
            Slot tmp; memset(&tmp, 0, sizeof tmp);
            slot_ensure_allocated(m, &tmp);
            load_expert_merged(m, pl[i], pe[i], &tmp);
            VktExpertSrc src = vk_slot_src(m, &tmp);
            vkt_put(pl[i], pe[i], &src);
            free(tmp.pw); free(tmp.g); free(tmp.gs); free(tmp.g4); free(tmp.u4); free(tmp.d4);
        }
        vkt_put_done();
        fprintf(stderr, "[VK] tier qwen36: warm start, %d experts from the history in %.1fs\n", n, now_s() - t0);
    }
    free(pl); free(pe);
}
#endif

int main(int argc, char **argv) {
    /* Physical-core team sizing, as colibri/inkling/kimi_k3/olmoe/deepseek-v41
     * do. Without it this engine takes one thread per logical CPU, which on an
     * SMT host doubles the team for no arithmetic and pays a barrier per tiny
     * per-expert region (#718 measured +2.3x from the sizing alone on a
     * 16C/32T part). OMP_NUM_THREADS wins, COLI_NO_OMP_TUNE=1 disables. */
    coli_omp_tune_threads("qwen36");
    const char *snap = getenv("SNAP");
    if (!snap) { coli_print_launcher_help("Qwen3.6", "SNAP=<model directory> ./qwen36 ..."); return 1; }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    if (g_wide < 1) g_wide = 1; if (g_wide > 4) g_wide = 4;
    g_cache_route = getenv("CACHE_ROUTE") ? atoi(getenv("CACHE_ROUTE")) : 0;         /* prefer resident experts (VRAM tier, then RAM cache) inside the top-M window; changes which experts run: docs/CACHE_ROUTE.md */
    g_route_j     = getenv("ROUTE_J")     ? atoi(getenv("ROUTE_J"))     : 2;         /* true top-J always taken, resident or not, under CACHE_ROUTE=1 */
    g_route_m     = getenv("ROUTE_M")     ? atoi(getenv("ROUTE_M"))     : 12;        /* rank window inside which a resident expert may replace an unresident one */
    g_route_p     = getenv("ROUTE_P")     ? (float)atof(getenv("ROUTE_P"))     : 0.f; /* cumulative-mass window for CACHE_ROUTE (0 = fixed M) */
    g_route_alpha = getenv("ROUTE_ALPHA") ? (float)atof(getenv("ROUTE_ALPHA")) : 1.f; /* scale substituted experts' gate mass before renorm (1 = off) */
    g_route_agree = getenv("ROUTE_AGREE") ? atoi(getenv("ROUTE_AGREE")) : g_cache_route; /* overlap% + KL vs the true top-K in the footer; auto-on under CACHE_ROUTE=1 */
    if (g_cache_route)
        fprintf(stderr, "[qwen36] CACHE_ROUTE=1 J=%d M=%d P=%.2f alpha=%.2f: VRAM tier > RAM cache > disk inside the top-M window (lossy: A/B it)\n",
                g_route_j, g_route_m, g_route_p, g_route_alpha);
    if (getenv("OPENAI")) g_openai = 1;                       /* OpenAI-compatible output */
    const char *mv = getenv("MODEL"); if (mv && *mv) g_model = mv;
    int hot_n = getenv("HOT") ? atoi(getenv("HOT")) : 0;
    int cap   = argc > 1 ? coli_arg_int(argv[1], "cache/layer") : 0;
    int bits  = argc > 2 ? coli_arg_int(argv[2], "expert bits") : 4;
    /* cap<=0 (explicit 0, or bare omission above) is the "auto-size from host
     * RAM" sentinel resolved later in model_init_range, once dense weights are
     * resident and xf_mode is known -- see qwen36_cap_for_ram(). The invariant
     * that a cap<1 must never reach the per-layer calloc (expert_get would
     * then find no slot to evict and wait for a publish that can never come;
     * the old lru=0 fallback turned that into a heap OOB instead) still holds,
     * just enforced there instead of only here for the auto path. */
    if (cap < 0) { fprintf(stderr, "cache/layer must be >= 0 (0 = auto; got %d)\n", cap); return 1; }
    if (bits < 2 || bits > 8) { fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits); return 1; }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf   = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;

    /* #1376: every capacity knob announced itself here except the one that
     * refuses requests. The context ceiling surfaced only in the
     * CONTEXT_EXCEEDED line, i.e. after a request had already failed. */
    char cap_disp[16];
    if (cap > 0) snprintf(cap_disp, sizeof cap_disp, "%d", cap);
    else snprintf(cap_disp, sizeof cap_disp, "auto");
    fprintf(stderr, "== qwen36 Phase-2 engine | cache=%s/layer bits=%d ctx=%d pilot=%d wide=%d hot=%d smooth=%.2f conf=%.2f ==\n",
           cap_disp, bits, qwen36_max_ctx(), g_pilot, g_wide, hot_n, smooth, conf);


    int is_ref = 0;
    jval *ref_image = NULL;      /* {"grid_h", "grid_w", "patches": [...]}: one image for the oracle */
    int rplen = (int)strlen(refpath);
    if (rplen>=5 && strcmp(refpath+rplen-5, ".json")==0) is_ref = 1;

    int *prompt=NULL, *full=NULL, *out=NULL;
    int np=0, nfull=0, n_new=0;
    char *buf=NULL, *arena=NULL;
    /* serve mode gets its prompts over the wire: skip the argv prompt file
     * entirely, or the default "ref.json" kills the engine before serve_loop
     * is ever reached — which is exactly how `coli` launches it (SERVE=1, no
     * prompt argument). */
    int serve_mode = getenv("SERVE") && getenv("SERVE")[0]=='1';
    if (serve_mode) {   /* KV_SLOTS: how many conversations the serve decodes at once */
        const char *ks = getenv("KV_SLOTS");
        if (ks && *ks) {
            char *end = NULL; long v = strtol(ks, &end, 10);
            if (end == ks || *end || v < 1 || v > 16) { fprintf(stderr, "KV_SLOTS must be between 1 and 16\n"); return 2; }
            g_q36_mux_slots = (int)v;
        }
        if (g_q36_mux_slots > 1)
            fprintf(stderr, "[qwen36] KV_SLOTS=%d: nothing drafts (speculation follows one conversation)\n", g_q36_mux_slots);
    }
    if (getenv("CLEF_RECORDS") || getenv("CLEF_TOKENIZE")) serve_mode = 1;   /* Clef's test modes: no prompt */

    /* load tokenizer early so text-prompt mode can encode before model_init */
    {
        const char *tokpath = getenv("TOK");
        if (tokpath && *tokpath) load_tokenizer(tokpath);
        else if (argc > 4 && argv[4] && *argv[4]) load_tokenizer(argv[4]);
        else { char tpb[2048]; snprintf(tpb,sizeof tpb,"%s/tokenizer.json",snap); load_tokenizer(tpb); }
        /* Clef: its requests are rendered into prompts, and the ids must be the
         * reference tokenizer's (see g_tok_exact) */
        if (q36_has_clef_head(snap)) g_tok_exact = 1;
    }

    if (serve_mode) {
        /* no argv prompt to load */
    } else if (is_ref) {
        FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        buf=malloc(n+1); if (fread(buf,1,n,f)!=(size_t)n) {} buf[n]=0; fclose(f);
        jval *ref = json_parse(buf, &arena);
        ref_image = ref ? json_get(ref, "image") : NULL;
        prompt = read_int_array(ref,"prompt_ids",&np);
        full   = read_int_array(ref,"full_ids",&nfull);
        n_new  = nfull - np;
    } else {
        /* text-prompt mode: read file as raw text, encode in C */
        FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        char *txt=malloc(n+1); if (fread(txt,1,n,f)!=(size_t)n) {} txt[n]=0; fclose(f);
        if (!g_tok) { fprintf(stderr, "[enc] no tokenizer loaded; cannot encode text. Put tokenizer.json in SNAP or set TOK.\n"); free(txt); return 1; }
        encode_text(txt, &prompt, &np);
        free(txt);
        n_new = getenv("N_NEW") ? atoi(getenv("N_NEW")) : 64;
        if (n_new < 1) n_new = 1;
        fprintf(stderr, "[enc] prompt tokens: %d | generating %d new tokens\n", np, n_new);
        if (getenv("ENC_DEBUG") && np <= 300) { fprintf(stderr, "[enc] prompt ids: "); for (int i=0;i<np;i++) fprintf(stderr, "%d ", prompt[i]); fprintf(stderr, "\n"); }
    }

    /* static, not a stack local: the PILOT prefetch worker is detached and
     * loops forever, and it keeps this address in the global pilot_m. A stack
     * Model dies when main returns while that thread is still dereferencing
     * it -- ASan: stack-use-after-return, READ of size 8, in a worker thread,
     * with the run's tokens already correct (#1262). Static storage outlives
     * every thread, so the pointer the worker holds stays valid. */
    static Model m; model_init(&m, snap, cap, bits);
    if (g_tok_exact) q36_clef_load(&m, snap);
    g_expert_gs = m.c.expert_gs;
    if (g_expert_gs) fprintf(stderr, "[qwen36] group-scaled experts: gs=%d\n", g_expert_gs);
    fprintf(stderr, "resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());
#ifdef COLI_VULKAN
    /* After the weights: the dense matrices upload at their first matmul_d.
     * No device (or COLI_VULKAN unset) leaves g_vk_ready 0, the CPU path. */
    g_vk_ready = coli_vk_init_env_tier("qwen36", vkt_wanted() && m.c.n_experts > 0 && !qq_active());
    g_vk_dense = coli_vk_dense();   /* COLI_VK_DENSE; unset, off on a device sharing the CPU's RAM with the tier on */
    /* COLI_VK_CHAIN: every layer's dense chain on the device (qwen36_chain.h); the
     * CUDA expert tier and a qpack container keep it off, as they keep the tier off */
    /* measured on a Radeon 780M: decode and prefill both faster (docs/vulkan.md) */
    if (g_vk_ready && !qq_active())
        g_vk_chain = coli_vk_chain_decide("qwen36", vkt_wanted() && m.c.n_experts > 0, COLI_VK_CHAIN_ON);
    if (g_vk_chain && g_q36_mux_slots > 1) {
        g_vk_chain = 0;
        fprintf(stderr, "[VK] qwen36: KV_SLOTS=%d: the dense chain is off (it keeps one conversation's state on the device); "
                        "the expert tier runs every conversation's experts\n", g_q36_mux_slots);
    }
    if (g_vk_ready) {
        /* COLI_VK_IMPORT: 1 the device reads the dense rows in place, 0 it copies them.
         * Unset: in place for a model without routed experts (Qwen3.8-27B, Clef) on a
         * device that shares the CPU's RAM, whose trunk is the whole model and would be
         * held twice (54 GB in f16: past a 61 GB box; measured on a Radeon 780M, a
         * prompt's GEMM within 4% and a decode GEMV within 9% of the copy's); a copy
         * elsewhere, as before. */
        const char *e = getenv("COLI_VK_IMPORT");
        int want = e && *e ? atoi(e) != 0 : m.c.n_experts == 0 && coli_vk_device_shares_ram();
        g_vk_import = want && coli_vk_import_alignment() > 0 && coli_vk_import_alignment() <= Q36_PAGE;
        if (want || (e && *e))
            fprintf(stderr, "[VK] qwen36: int8 and f16 dense rows %s (%s)\n", g_vk_import ? "read in place, not copied" : "copied",
                    !want ? "COLI_VK_IMPORT=0" : g_vk_import ? (e && *e ? "COLI_VK_IMPORT=1" :
                    "a dense model on a device sharing the CPU's RAM; COLI_VK_IMPORT=0 copies them")
                    : "this device cannot import host memory");
    }
    /* the chain's pipelines come up with its fit (q36c_start, below): nothing of the
     * chain is on the device before it has decided how many layers fit */
#endif
    if (ref_image && ref_image->t == J_OBJ) {
        jval *gh = json_get(ref_image, "grid_h"), *gw = json_get(ref_image, "grid_w");
        jval *pv = json_get(ref_image, "patches");
        if (!gh || !gw || !pv || gh->t != J_NUM || gw->t != J_NUM || pv->t != J_ARR) {
            fprintf(stderr, "ref.json image needs grid_h, grid_w and patches\n"); return 1;
        }
        float *px = malloc((size_t)pv->len * sizeof(float));
        if (!px) { fprintf(stderr, "OOM image patches\n"); return 1; }
        for (int i = 0; i < pv->len; i++) px[i] = pv->kids[i]->t == J_NUM ? (float)pv->kids[i]->num : 0.f;
        Cfg *vc = &m.c;
        long long want = (long long)gh->num * (long long)gw->num * vc->vis_in_ch * vc->vis_temporal * vc->vis_patch * vc->vis_patch;
        if (!m.vis_ready || pv->len != want ||
            q36_vision_attach(&m, px, (int)gh->num, (int)gw->num, prompt, np) < 0) {
            fprintf(stderr, "the ref.json image does not fit this model (%d values, %lld expected, tower %s)\n",
                    pv->len, want, m.vis_ready ? "loaded" : "absent");
            return 1;
        }
        free(px);
        fprintf(stderr, "[qwen36] oracle image: %dx%d patches, %d tokens, rope delta %d\n",
                (int)gh->num, (int)gw->num, m.vis_rows_n, m.rope_delta);
    }
    /* dense matrices are quantized to int8 (+ int4 planar per COLI_DENSE_BITS/
     * COLI_DENSE_INT4, see dense_int4_wanted) during model_init above
     * (COLI_DENSE_I8=0 disables it) -- see load_tq/QW; model_init_range
     * already logged the count and freed bytes. */

    /* QWEN36_QPACK=<dir> hands the routed experts to a Swiftlet qpack
     * container through the MLX-affine path (Metal on Apple builds, CPU
     * reference elsewhere).  A mismatched or unsupported container is a
     * refusal, not a silent fallback to the snapshot experts. */
    const char *qq_dir = getenv("QWEN36_QPACK");
    if (qq_dir && *qq_dir) {
        char qq_err[256];
#ifdef COLI_METAL
        /* Compile the Metal affine pipelines BEFORE the store opens.  The
         * store dispatches through coli_metal_matmul_affine_slot, which is
         * gated on coli_metal_affine_available() -- i.e. on pipelines that
         * only coli_metal_init() compiles.  The parity gates call it in
         * their own main; the ENGINE never did, so a METAL=1 build ran every
         * real-model routed projection on the CPU affine reference while
         * looking identical apart from qq_counts (measured: metal=0
         * cpu=37440 on the first 35B smoke).  Not a refusal when
         * unavailable: the CPU reference is the documented oracle arm, but
         * it must be CHOSEN (QWEN36_QPACK_SYNC / qq_force_cpu) or loudly
         * reported, never a silent default. */
        if (coli_metal_init())
            fprintf(stderr, "[qpack] Metal affine pipelines ready\n");
        else
            fprintf(stderr, "[qpack] Metal unavailable -- routed experts"
                    " fall back to the CPU affine reference\n");
#endif
        /* QWEN36_QPACK_SLOTS=<n> bounds the refillable expert slot pool
         * (0/unset = the default in qwen36_qpack.c, clamped to the container
         * total either way). */
        const char *qq_slots_env = getenv("QWEN36_QPACK_SLOTS");
        if (qq_slots_env && *qq_slots_env)
            qq_set_slot_count(atoi(qq_slots_env));
        if (!qq_open(qq_dir, m.c.n_layers, m.c.n_experts, m.c.hidden,
                     m.c.inter, qq_err, sizeof(qq_err))) {
            fprintf(stderr, "[qpack] %s: %s\n", qq_dir, qq_err);
            return 1;
        }
        atexit(qq_close);
        {
            int qq_slots = 0;
            qq_slot_stats(&qq_slots, NULL, NULL);
            fprintf(stderr, "[qpack] routed experts <- MLX affine container"
                    " %s (%d bounded slots)\n", qq_dir, qq_slots);
        }
    }

    /* Optional CUDA VRAM expert tier (COLI_CUDA=1): hot experts live in
     * DEVICE_LOCAL memory across the configured GPUs, misses fall back to the
     * CPU int8 path. See qwen36_tier.h. Skipped when the qpack container owns
     * the routed experts: the tier would warmstart-load the snapshot experts
     * moe() no longer computes. */
    /* Formato degli esperti dalla TAGLIA SU DISCO del primo, non da meta.ebits:
     * esiste un container i8 il cui meta dichiara ebits=4 (stesso motivo per cui
     * il loader piu' sopra guarda nbytes). Il tier ne ha bisogno prima di
     * riservare qualunque budget: e' int4 impacchettato che va in VRAM come
     * fmt=4, int8 come fmt=1. Sbagliare qui era #1331 -- budget riservato,
     * planned=1, e zero promozioni per tutta la vita del processo. */
    int expert_is_int4 = 1, expert_mixed = 0;
    {
        char probe[256];
        snprintf(probe, sizeof(probe),
                 "model.layers.%d.mlp.experts.0.merged_weight", m.active_of[0]);
        st_tensor *pt = st_find(&m.S, probe);
        int64_t want = 2*(int64_t)m.c.inter*m.c.hidden + (int64_t)m.c.hidden*m.c.inter;
        if (pt && pt->nbytes == want) expert_is_int4 = 0;   /* int8: un byte per elemento */
        /* mixed (convert_qwen36.py --down-bits): int4 gate/up + int8 down = 2/3 of int8 */
        if (pt && pt->nbytes == want * 2 / 3) { expert_is_int4 = 0; expert_mixed = 1; }
    }
    /* Una riga, sempre: e' l'unico modo di verificare il probe dall'esterno
     * (CI sul container tiny int8, #1331) senza una scheda. */
    fprintf(stderr, "[qwen36] expert format on disk: %s\n",
            m.c.n_experts == 0 ? "none (dense model: every layer's MLP is resident)"
            : expert_mixed ? "int4 gate/up + int8 down (mixed; CPU path, no VRAM tier yet)"
                         : expert_is_int4 ? "int4 packed (tier fmt=4)" : "int8 (tier fmt=1)");
    g_expert_is_int4 = expert_is_int4;
    g_expert_mixed = expert_mixed; g_expert_down_gs = expert_mixed ? m.c.expert_down_gs : 0;
    if (expert_mixed && m.c.expert_down_bits == 0)
        fprintf(stderr, "[qwen36] mixed layout on disk but qwen36_meta.json has no expert_down_bits -- down scales assumed per row\n");
    /* Offer the dense trunk to the placer before the tier decides its budget:
     * sizes only, from the same dense-i8 entries the uploads below will use.
     * No entry (dense-i8 off) means nothing to offer, and the CPU path stands.
     * A qpack container owns routed execution outright, so it must not seed
     * CUDA placement state for a tier that is deliberately skipped below. */
    if (!qq_active()) {
        int O_qkv = m.c.dn_conv_dim, O_z = m.c.dn_vheads * m.c.dn_vdim;
        if (m.lm_head.q)
            qt_trunk_offer("lmhead", 0, (size_t)m.lm_head.I * m.lm_head.O + (size_t)m.lm_head.O * sizeof(float));
        for (int i = 0; i < m.c.n_layers; i++) {
            if (m.c.is_attn[i]) continue;
            if (m.L[i].dn_qkv.q && m.L[i].dn_z.q)
                qt_trunk_offer("dnproj", i, (size_t)(O_qkv + O_z) * m.c.hidden + (size_t)(O_qkv + O_z) * sizeof(float));
        }
        trunk_offer_dense(&m);   /* dnout, attnproj, shexp: the rest of the per-token dense work */
    }
    if (expert_mixed && getenv("COLI_CUDA") && getenv("COLI_CUDA")[0] == '1')
        fprintf(stderr, "[qwen36] COLI_CUDA=1 ignored: the VRAM expert tier does not take the mixed layout yet (one format per expert)\n");
    /* The sentinel never leaves main() unresolved: see qwen36_resolved_cap. */
    cap = qwen36_resolved_cap(cap, m.cache, m.c.n_layers);
    if (!qq_active() && !expert_mixed && m.c.n_experts > 0 &&
        qt_init(m.c.n_layers, m.c.n_experts, m.c.hidden, m.c.inter, cap, m.c.topk,
                m.c.expert_gs, expert_is_int4)) {
        fprintf(stderr, "[gpu] MoE experts -> CUDA VRAM tier\n");
        atexit(qt_shutdown);
        /* The placer predicted; measure before uploading a byte of trunk. */
        if (!trunk_probe_gpu_wins(&m)) qt_trunk_withdraw("measured slower than the CPU");
        /* R4 role split: park the dense-i8 lm_head on COLI_LMHEAD_GPU. The
         * QW struct on m.lm_head holds the int8 rows + per-row scales the
         * CPU path uses; the GPU applies the identical semantics. */
        if (m.lm_head.q)
            qt_lmhead_init(m.lm_head.q, m.lm_head.sc, m.lm_head.I, m.lm_head.O);
        /* R4 step 2: DeltaNet input projections, per layer, wherever
         * COLI_PLACE puts them. qkv and z are both [O_x, hidden] int8 with
         * per-row scales, so fusing them is a concatenation along O -- two
         * memcpys, no requantization, and the GPU sees one GEMV per layer
         * instead of two. The host copy is freed right after the upload. */
        {
            int placed = 0, O_qkv = m.c.dn_conv_dim, O_z = m.c.dn_vheads * m.c.dn_vdim;
            int Of = O_qkv + O_z, Hd = m.c.hidden;
            double vram = 0;
            for (int i = 0; i < m.c.n_layers; i++) {
                if (m.c.is_attn[i]) continue;
                int dev = qt_place_of("dnproj", i);
                if (dev == QT_PLACE_CPU) continue;
                const int8_t *q1 = m.L[i].dn_qkv.q, *q2 = m.L[i].dn_z.q;
                const float *s1 = m.L[i].dn_qkv.sc, *s2 = m.L[i].dn_z.sc;
                if (!q1 || !q2) continue;      /* dense-i8 off: CPU path stands */
                int8_t *qf = malloc((size_t)Of * Hd);
                float  *sf = malloc((size_t)Of * sizeof(float));
                if (!qf || !sf) { free(qf); free(sf); break; }
                memcpy(qf, q1, (size_t)O_qkv * Hd);
                memcpy(qf + (size_t)O_qkv * Hd, q2, (size_t)O_z * Hd);
                memcpy(sf, s1, (size_t)O_qkv * sizeof(float));
                memcpy(sf + O_qkv, s2, (size_t)O_z * sizeof(float));
                if (qt_dnproj_init(i, qf, sf, Hd, Of, dev)) {
                    placed++; vram += (double)Of * Hd;
                }
                free(qf); free(sf);
            }
            if (placed)
                fprintf(stderr, "[dnp] %d DeltaNet-Projektionen auf GPU (%.2f GB VRAM)\n",
                        placed, vram / 1073741824.0);
        }
        /* The rest of the trunk, wherever the placer put it: out_proj,
         * attention projections, shared expert. Handles live in the Layer. */
        {
            double vram = 0;
            int placed = trunk_place_dense(&m, &vram);
            if (placed)
                fprintf(stderr, "[dense] %d trunk matrices on GPU (dnout/attnproj/shexp, %.2f GB VRAM)\n",
                        placed, vram / 1073741824.0);
        }
        /* Q36_DN_GPU=1: where the in_proj and the out_proj of a DeltaNet layer
         * both sit on one card, the conv ring, the recurrence and the gated
         * norm go there too, and a decode token runs the layer end to end on
         * the device (qt_dn_gpu_step). Measured motivation: with the trunk in
         * VRAM the CPU still spent ~8 of 39 ms/token on these steps -- not
         * arithmetic, host round trips, thirty per token. */
        if (dn_gpu_env_on()) {
            int n = 0; double vram = 0;
            for (int i = 0; i < m.c.n_layers; i++) {
                if (m.c.is_attn[i] || !m.L[i].qth_dnout || !qt_dnproj_ready(i)) continue;
                if (qt_dn_gpu_init(i, m.c.dn_vheads, m.c.dn_kheads, m.c.dn_kdim, m.c.dn_vdim, m.c.dn_conv_dim, m.c.dn_convk,
                                   m.c.hidden, m.L[i].dn_conv, m.L[i].dn_norm, m.c.eps, m.L[i].qth_dnout)) {
                    n++; vram += (double)m.c.dn_vheads * m.c.dn_kdim * m.c.dn_vdim * 4 + (double)m.c.dn_conv_dim * (m.c.dn_convk - 1) * 4;
                }
            }
            m.dn_dev = n > 0;
            if (m.dn_dev && g_q36_mux_slots > 1) {   /* the card holds one conversation's DeltaNet state */
                m.dn_dev = 0;
                fprintf(stderr, "[dn] KV_SLOTS=%d: DeltaNet on the CPU\n", g_q36_mux_slots);
            }
            if (n) fprintf(stderr, "[dn] %d DeltaNet layers run on the GPU end to end (conv, recurrence, gated norm; %.0f MB of state in VRAM)\n",
                           n, vram / 1048576.0);
            else fprintf(stderr, "[dn] Q36_DN_GPU=1 but no layer has both projections on one card; the CPU path stands\n");
        }
        /* Warmstart: fill the VRAM budget BEFORE the first token (heat order
         * when HEAT_FILE exists, natural order otherwise), loading all RAM
         * slots along the way. */
        const char *nws = getenv("QT_NO_WARMSTART");
        if (!(nws && *nws=='1')) tier_warmstart(&m, expert_is_int4);
    }

#ifdef COLI_VULKAN
    q36c_start(&m);      /* COLI_VK_CHAIN: how many layers the chain places (vkc_fit), before any upload */
    if (g_vk_chain && !g_q36c_fit_on && !vkc_init()) g_vk_chain = 0;   /* no fit (PILOT, a geometry, ...): as before */
    q36_dho_start(&m);   /* COLI_VK_DENSE_HOST: the dense matrices on the device only, before the tier sizes its budget */
    if (g_q36c_fit_on && g_vk_chain) {   /* the chain's layers on the device now, so the tier sizes after them */
        q36c_place(&m);
        if (coli_vk_dense_device_only()) {
            coli_vk_dense_host_layers(g_q36c_fit.n, g_q36c_fit.L);
            q36_dho_placed();
        }
    }
    vk_tier_start(&m, snap, cap, expert_is_int4, expert_mixed);   /* COLI_VULKAN=1: hot routed experts on the device */
    if (g_vk_ready && !vkt_ready() && !g_vk_dense) g_vk_dense = coli_vk_dense_decide("qwen36", 0, 1);   /* no tier after all */
    if (g_vk_chain && qt_ready()) {   /* the CUDA tier keeps its priority */
        fprintf(stderr, "[VK] qwen36: dense chain off, the CUDA expert tier is on\n");
        g_vk_chain = 0;
    }
    if (g_vk_chain && g_pilot) fprintf(stderr, "[VK] qwen36: PILOT prefetch reads the residual on the host: the dense chain stays off\n");
    if (g_vk_chain || g_q36c_fit_on) atexit(vkc_shutdown_all);   /* registered after the tier's: runs before the device goes */
#endif

    /* coli serve mode: speak the gateway wire protocol instead of argv
     * generation. AFTER the tier init: serve sessions ride the VRAM experts
     * exactly like argv runs, and serve_loop never returns. */
    if (getenv("CLEF_RECORDS") || getenv("CLEF_TOKENIZE")) {
        if (!g_tok) { fprintf(stderr, "[clef] tokenizer.json required\n"); return 1; }
        return q36_clef_test_modes(&m);
    }
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        if (!g_tok) { fprintf(stderr, "[serve] tokenizer.json required (put in SNAP or set TOK)\n"); return 1; }
        serve_loop(&m);
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
        route_footer(stdout, &m);
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
#ifdef COLI_VULKAN
        vk_report();
        q36c_report(&m);
        vk_tier_turn(&m, "run");
#endif
        free(buf); free(arena); return 0;
    }

    out = malloc((np + n_new) * sizeof(int));
    /* timing + OpenAI id setup (before generation) */
    g_ttft = -1; g_gen_t0 = now_s();
    if (g_openai){
        g_oa_created = (long)time(NULL);
        snprintf(g_oa_id, sizeof g_oa_id, "chatcmpl-%ld%04d", g_oa_created, (int)(now_s()*1000) % 10000);
    }
    /* streaming text: emit tokens as they are produced (text mode + tokenizer only) */
    if (!is_ref && g_tok && !getenv("NOSTREAM")) {
        g_stream = 1; g_sbn = 0;
        if (g_openai){
            char jb[320];
            snprintf(jb, sizeof jb,
              "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
              "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}",
              g_oa_id, g_oa_created, g_model);
            sse_chunk(jb);
        } else {
            fprintf(stderr, "Generated (%d new tokens):\nText : ", n_new); fflush(stderr);
        }
    }
    double t = now_s();
    if (is_ref) { g_q36_oracle = full; g_q36_oracle_n = nfull; }   /* COLI_LOOKUP_FORCE */
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    /* DUMP=<path>: write last-token logits (raw float32, vocab) for a torch-free
     * cosine comparison against tools/_ref_dn.py --dump. */
    if (g_last_logit) {
        const char *dp = getenv("DUMP");
        FILE *df = fopen(dp && *dp ? dp : "qwen36_logits.f32", "wb");
        if (df) { fwrite(g_last_logit, sizeof(float), (size_t)m.c.vocab, df); fclose(df);
                  fprintf(stderr, "[dump] wrote %d logits -> %s\n", m.c.vocab, dp && *dp ? dp : "qwen36_logits.f32"); }
        else fprintf(stderr, "[dump] cannot open %s\n", dp ? dp : "qwen36_logits.f32");
    }

    int ref_match = 0;
    if (is_ref) {
        int match = 0;
        printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
        printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
        if (g_tok) { printf("Text      : "); print_decoded(out, np, nfull); printf("\n"); }
        printf("\nMatching tokens: %d/%d\n", match, n_new);
        ref_match = match;
    } else {
    if (g_openai) {
        emit_openai_result(out, np, n_new, g_stream);
    } else if (g_stream) {
        stream_flush(); fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "\nGenerated (%d new tokens):\n", n_new);
        if (g_tok) { fprintf(stderr, "Text      : "); print_decoded(out, np, np+n_new); fprintf(stderr, "\n"); }
        else { fprintf(stderr, "Ids       : "); for (int i=np;i<np+n_new;i++) fprintf(stderr, "%d ", out[i]); fprintf(stderr, "\n"); }
    }
    }
    double tot = m.hits + m.miss;
    if (g_ttft >= 0) fprintf(stderr, "TTFT: %.2f s (time to first token)\n", g_ttft);
    q36_spec_report(&g_q36_run_spec, "run");
    q36_spec_free(&g_q36_run_spec);
    tm_report();
    qt_stats();
    /* qpack dispatch evidence, mirroring what the parity gates assert: which
     * path computed the routed projections (coli_metal_matmul_affine_slot
     * returns 0 for CPU fallback by contract, and qq_counts is where that
     * becomes visible) and how hard the bounded slot pool worked.  A run
     * whose experts silently fell back to the CPU reference shows cpu != 0
     * here instead of hiding it. */
    if (qq_active()) {
        uint64_t qmp = 0, qcp = 0, qev = 0, qfl = 0;
        int qsl = 0;
        qq_counts(&qmp, &qcp);
        qq_slot_stats(&qsl, &qev, &qfl);
        fprintf(stderr, "[qpack] projections: metal=%llu cpu=%llu"
                " | slots=%d fills=%llu evictions=%llu\n",
                (unsigned long long)qmp, (unsigned long long)qcp,
                qsl, (unsigned long long)qfl, (unsigned long long)qev);
    }
    fprintf(stderr, "\nPEAK RSS: %.2f GB\n", rss_gb());
    fprintf(stderr, "Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    route_footer(stderr, &m);
    fprintf(stderr, "Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
#ifdef COLI_VULKAN
    vk_report();
    q36c_report(&m);
    vk_tier_turn(&m, "run");
#endif
    free(buf); free(arena);
    /* Oracle mode is a gate, not a report: a mismatch must fail the caller.
     * inkling.c does the same (`return (match == ngen) ? 0 : 1;`) and its CI
     * job relies on it — without this, tools/make_qwen36_oracle.py could be
     * wired into a workflow that stays green through any regression. */
    if (is_ref) return ref_match == n_new ? 0 : 1;
    return 0;
}
#endif /* QWEN36_NO_MAIN */

#ifdef COLI_SEGMENT_ADAPTER
/* ---------- engine-owned Segment adapter ------------------------------ */

typedef struct {
    Model model;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
} Qwen36SegmentEngine;

typedef struct {
    Qwen36SegmentEngine *engine;
    float **K, **V, **DN_rec, **DN_conv;
    uint32_t context_tokens, position;
} Qwen36SegmentSession;

static void qwen36_segment_layer_free(Layer *layer) {
    free(layer->in_ln); free(layer->post_ln);
    qw_free(&layer->q); qw_free(&layer->k); qw_free(&layer->v); qw_free(&layer->o);
    free(layer->qn); free(layer->kn); qw_free(&layer->gate); free(layer->gate_bias);
    qw_free(&layer->sh_g); qw_free(&layer->sh_u); qw_free(&layer->sh_d); free(layer->sh_gate);
    qw_free(&layer->dn_qkv); qw_free(&layer->dn_z); free(layer->dn_b); free(layer->dn_a);
    free(layer->dn_conv); free(layer->dn_dtbias); free(layer->dn_alog);
    free(layer->dn_norm); qw_free(&layer->dn_out);
}

static void qwen36_segment_model_destroy(Qwen36SegmentEngine *engine) {
    if (!engine) return;
    Model *model = &engine->model;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        qwen36_segment_layer_free(&model->L[layer]);
        LCache *cache = &model->cache[layer];
        for (int slot = 0; slot < cache->n; slot++) {
            free(cache->slots[slot].g);
            free(cache->slots[slot].pw);
            free(cache->slots[slot].gs);
            free(cache->slots[slot].g4);
            free(cache->slots[slot].u4);
            free(cache->slots[slot].d4);
        }
        free(cache->slot_by_expert); free(cache->slots);
    }
    free(model->attn_sc);
    free(model->seen); free(model->is_queued); free(model->is_pinned);
    free(model->momentum_logits); free(model->freq);
    free(model->DN_conv); free(model->DN_rec); free(model->dn_dev_fresh); free(model->dn_host_stale);
    free(model->cache); free(model->active_of); free(model->L);
    free(model->c.is_attn);
    st_destroy(&model->S);
}

static int qwen36_segment_engine_open(
    void **engine_impl, ColiSegmentCapabilities *capabilities,
    const ColiSegmentEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid Qwen3.6 Segment open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_SEGMENT_CAP_CPU))
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.6 Segment currently supports CPU");
    if (options->context_tokens > QWEN36_ATTN_MAX_CTX)
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.6 Segment context exceeds model limit");
    Cfg config;
    memset(&config, 0, sizeof(config));
    load_cfg(&config, options->model_dir);
    int configured_layers = config.n_layers;
    load_meta(&config, options->model_dir);
    validate_cfg(&config, configured_layers);
    if (options->layer_end > (uint32_t)config.n_layers) {
        free(config.is_attn);
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.6 Segment range exceeds model");
    }
    int range_layers = (int)(options->layer_end - options->layer_begin);
    int cap = 0; /* sentinel: model_init_range derives it from host RAM,
                  * honoring memory_limit_bytes==0's own doc comment ("uses
                  * the adapter's ordinary automatic budget") */
    if (options->memory_limit_bytes) {
        uint64_t weights = (uint64_t)config.hidden * config.inter * 3u;
        uint64_t per_slot = weights +
            (uint64_t)(config.inter * 2 + config.hidden) * sizeof(float);
        uint64_t slots = per_slot && range_layers > 0
            ? options->memory_limit_bytes / per_slot / (uint64_t)range_layers
            : 0;
        cap = slots > (uint64_t)config.n_experts ? config.n_experts : (int)slots;
        if (cap < 1) cap = 1;
    }
    Qwen36SegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine) {
        free(config.is_attn);
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory opening Qwen3.6 Segment");
    }
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;
    if (pthread_mutex_init(&engine->run_lock, NULL)) {
        free(config.is_attn); free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "cannot initialize Qwen3.6 Segment lock");
    }
    free(config.is_attn);
    model_init_range(&engine->model, options->model_dir, cap, 8,
                     (int)options->layer_begin, (int)options->layer_end, 0, 0);
    engine->model.quant_bits = container_layer_is_int4(
        &engine->model, (int)options->layer_begin) ? 4 : 8;
    free(engine->model.DN_rec); free(engine->model.DN_conv); free(engine->model.dn_dev_fresh); free(engine->model.dn_host_stale);
    engine->model.DN_rec = NULL; engine->model.DN_conv = NULL;
    engine->model.dn_dev_fresh = NULL; engine->model.dn_host_stale = NULL;   /* a segment run re-reads them (deltanet) and teardown frees again */
    engine->model.max_t = (int)options->context_tokens;
    engine->model.kv_cap = (int)options->context_tokens;
    engine->model.attn_sc_thr = 1;
#ifdef _OPENMP
    engine->model.attn_sc_thr = omp_get_max_threads();
    if (engine->model.attn_sc_thr < 1) engine->model.attn_sc_thr = 1;
#endif
    size_t scratch_cells;
    if (coli_segment_size_mul((size_t)engine->model.attn_sc_thr,
                              options->context_tokens, &scratch_cells)) {
        qwen36_segment_model_destroy(engine);
        pthread_mutex_destroy(&engine->run_lock); free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.6 attention scratch overflows");
    }
    engine->model.attn_sc = calloc(scratch_cells, sizeof(float));
    if (!engine->model.attn_sc) {
        qwen36_segment_model_destroy(engine);
        pthread_mutex_destroy(&engine->run_lock); free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory for Qwen3.6 attention");
    }
    engine->model.resident_mode = 0;

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_SNAPSHOT |
                          COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION |
                          COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "qwen36");
    coli_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "qwen36/kv-deltanet-conv-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "qwen36/f32-int%d/cpu-v1", engine->model.quant_bits);
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = (uint32_t)engine->model.c.hidden;
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = QWEN36_ATTN_MAX_CTX;
    capabilities->num_layers = (uint32_t)engine->model.c.n_layers;
    *engine_impl = engine;
    return 0;
}

static void qwen36_segment_engine_destroy(void *engine_impl) {
    Qwen36SegmentEngine *engine = (Qwen36SegmentEngine *)engine_impl;
    if (!engine) return;
    qwen36_segment_model_destroy(engine);
    pthread_mutex_destroy(&engine->run_lock);
    free(engine);
}

static int qwen36_segment_session_create(
    void *engine_impl, void **session_impl,
    const ColiSegmentSessionOptions *options, char *error, size_t error_size) {
    Qwen36SegmentEngine *engine = (Qwen36SegmentEngine *)engine_impl;
    if (!engine || !session_impl || !options)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid Qwen3.6 Segment session");
    *session_impl = NULL;
    Qwen36SegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory creating Qwen3.6 session");
    session->engine = engine;
    session->context_tokens = options->context_tokens;
    int layers = engine->model.c.n_layers;
    session->K = calloc((size_t)layers, sizeof(*session->K));
    session->V = calloc((size_t)layers, sizeof(*session->V));
    session->DN_rec = calloc((size_t)layers, sizeof(*session->DN_rec));
    session->DN_conv = calloc((size_t)layers, sizeof(*session->DN_conv));
    if (!session->K || !session->V || !session->DN_rec || !session->DN_conv)
        goto oom;
    Cfg *config = &engine->model.c;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        size_t cells;
        if (config->is_attn[layer]) {
            if (coli_segment_size_mul((size_t)config->kv_heads,
                                      options->context_tokens, &cells) ||
                coli_segment_size_mul(cells, (size_t)config->k_head_dim,
                                      &cells)) goto oom;
            session->K[layer] = calloc(cells, sizeof(float));
            session->V[layer] = calloc(cells, sizeof(float));
            if (!session->K[layer] || !session->V[layer]) goto oom;
        } else {
            if (coli_segment_size_mul((size_t)config->dn_vheads,
                                      (size_t)config->dn_kdim, &cells) ||
                coli_segment_size_mul(cells, (size_t)config->dn_vdim,
                                      &cells)) goto oom;
            session->DN_rec[layer] = calloc(cells, sizeof(float));
            if (coli_segment_size_mul((size_t)config->dn_conv_dim,
                                      (size_t)(config->dn_convk - 1),
                                      &cells)) goto oom;
            session->DN_conv[layer] = calloc(cells, sizeof(float));
            if (!session->DN_rec[layer] || !session->DN_conv[layer]) goto oom;
        }
    }
    *session_impl = session;
    return 0;

oom:
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        free(session->K ? session->K[layer] : NULL);
        free(session->V ? session->V[layer] : NULL);
        free(session->DN_rec ? session->DN_rec[layer] : NULL);
        free(session->DN_conv ? session->DN_conv[layer] : NULL);
    }
    free(session->K); free(session->V);
    free(session->DN_rec); free(session->DN_conv); free(session);
    return coli_segment_adapter_error(error, error_size,
                                       "out of memory allocating Qwen3.6 state");
}

static void qwen36_segment_session_destroy(void *session_impl) {
    Qwen36SegmentSession *session = (Qwen36SegmentSession *)session_impl;
    if (!session) return;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++) {
        free(session->K[layer]); free(session->V[layer]);
        free(session->DN_rec[layer]); free(session->DN_conv[layer]);
    }
    free(session->K); free(session->V);
    free(session->DN_rec); free(session->DN_conv); free(session);
}

static int qwen36_segment_session_run(void *session_impl,
                                      const ColiSegmentRunRequest *request,
                                      char *error, size_t error_size) {
    Qwen36SegmentSession *session = (Qwen36SegmentSession *)session_impl;
    if (!session || !request || request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "Qwen3.6 Segment requires contiguous positions");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.6 Segment run cancelled");
    Qwen36SegmentEngine *engine = session->engine;
    if (request->output != request->input)
        memcpy(request->output, request->input, request->input_bytes);
    pthread_mutex_lock(&engine->run_lock);
    Model *model = &engine->model;
    model->K = session->K; model->V = session->V;
    model->DN_rec = session->DN_rec; model->DN_conv = session->DN_conv;
    model->max_t = (int)session->context_tokens;
    model->kv_len = (int)session->position;
    layers_forward_range(model, (float *)request->output, (int)request->rows,
                         (int)request->position, (int)engine->layer_begin,
                         (int)engine->layer_end, 0, NULL);
    model->K = NULL; model->V = NULL;
    model->DN_rec = NULL; model->DN_conv = NULL;
    model->kv_len = 0;
    pthread_mutex_unlock(&engine->run_lock);
    session->position += request->rows;
    return 0;
}

static int qwen36_segment_spans(
    Qwen36SegmentSession *session, uint32_t position,
    ColiSegmentStateSpan **spans_output, size_t *count_output,
    char *error, size_t error_size) {
    Cfg *config = &session->engine->model.c;
    size_t capacity = (size_t)(session->engine->layer_end -
                               session->engine->layer_begin) *
                      (size_t)(2 * config->kv_heads + 2);
    ColiSegmentStateSpan *spans = capacity
        ? calloc(capacity, sizeof(*spans)) : NULL;
    if (capacity && !spans)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory describing Qwen3.6 state");
    size_t count = 0;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++) {
        if (config->is_attn[layer]) {
            size_t row_bytes = (size_t)position * config->k_head_dim *
                               sizeof(float);
            size_t stride = (size_t)session->context_tokens *
                            config->k_head_dim;
            for (int kv = 0; kv < 2; kv++) {
                float *state = kv ? session->V[layer] : session->K[layer];
                for (int head = 0; head < config->kv_heads; head++)
                    spans[count++] = (ColiSegmentStateSpan){
                        state + head * stride, row_bytes};
            }
        } else {
            size_t rec_cells, conv_cells;
            if (coli_segment_size_mul((size_t)config->dn_vheads,
                                      (size_t)config->dn_kdim, &rec_cells) ||
                coli_segment_size_mul(rec_cells, (size_t)config->dn_vdim,
                                      &rec_cells) ||
                coli_segment_size_mul((size_t)config->dn_conv_dim,
                                      (size_t)(config->dn_convk - 1),
                                      &conv_cells)) {
                free(spans);
                return coli_segment_adapter_error(error, error_size,
                                                   "Qwen3.6 state size overflow");
            }
            spans[count++] = (ColiSegmentStateSpan){
                session->DN_rec[layer], rec_cells * sizeof(float)};
            spans[count++] = (ColiSegmentStateSpan){
                session->DN_conv[layer], conv_cells * sizeof(float)};
        }
    }
    *spans_output = spans; *count_output = count;
    return 0;
}

static int qwen36_segment_session_snapshot(
    void *session_impl, ColiSegmentWriteFn write_fn, void *write_user_data,
    char *error, size_t error_size) {
    Qwen36SegmentSession *session = (Qwen36SegmentSession *)session_impl;
    ColiSegmentStateSpan *spans = NULL; size_t count = 0, payload_bytes;
    if (!session || qwen36_segment_spans(session, session->position, &spans,
                                         &count, error, error_size) ||
        coli_segment_spans_size(spans, count, &payload_bytes)) {
        free(spans);
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.6 snapshot size overflow");
    }
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(
        &header, "qwen36", session->engine->layer_begin,
        session->engine->layer_end, session->context_tokens, session->position,
        payload_bytes, coli_segment_spans_hash(spans, count));
    int result = coli_segment_stream_write(
        write_fn, write_user_data, &header, sizeof(header), error, error_size);
    if (!result)
        result = coli_segment_spans_write(spans, count, write_fn,
                                          write_user_data, error, error_size);
    free(spans);
    return result;
}

static int qwen36_segment_session_restore(
    void *session_impl, ColiSegmentReadFn read_fn, void *read_user_data,
    char *error, size_t error_size) {
    Qwen36SegmentSession *session = (Qwen36SegmentSession *)session_impl;
    ColiSegmentSnapshotHeader header;
    if (!session || coli_segment_stream_read(read_fn, read_user_data, &header,
                                             sizeof(header), error, error_size))
        return -1;
    ColiSegmentStateSpan *spans = NULL; size_t count = 0, payload_bytes;
    if (qwen36_segment_spans(session, header.position, &spans, &count,
                             error, error_size) ||
        coli_segment_spans_size(spans, count, &payload_bytes) ||
        coli_segment_snapshot_header_valid(
            &header, "qwen36", session->engine->layer_begin,
            session->engine->layer_end, session->context_tokens, payload_bytes,
            error, error_size)) {
        free(spans); return -1;
    }
    int result = coli_segment_spans_restore(
        spans, count, header.payload_hash, read_fn, read_user_data,
        error, error_size);
    free(spans);
    if (!result) session->position = header.position;
    return result;
}

static const ColiSegmentAdapter qwen36_segment_adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION, "qwen36",
    qwen36_segment_engine_open, qwen36_segment_engine_destroy,
    qwen36_segment_session_create, qwen36_segment_session_destroy,
    qwen36_segment_session_run, qwen36_segment_session_snapshot,
    qwen36_segment_session_restore, {0}
};

int coli_qwen36_segment_adapter_register(void) {
    return coli_segment_adapter_register(&qwen36_segment_adapter);
}
#endif /* COLI_SEGMENT_ADAPTER */

#ifdef COLI_EDGE_ADAPTER
/* ---------- engine-owned model Edge adapter --------------------------- */

typedef struct {
    Model model;
} Qwen36EdgeEngine;

static void qwen36_edge_tokenizer_destroy(void) {
    for (int item = 0; item < g_tok_n; item++) free(g_tok[item]);
    free(g_tok); g_tok = NULL; g_tok_n = 0;
    for (int slot = 0; slot < g_merge.cap; slot++)
        if (g_merge.used && g_merge.used[slot]) free(g_merge.keys[slot]);
    free(g_rev.keys); free(g_rev.vals); free(g_rev.used);
    free(g_merge.keys); free(g_merge.vals); free(g_merge.used);
    memset(&g_rev, 0, sizeof(g_rev)); memset(&g_merge, 0, sizeof(g_merge));
    for (int item = 0; item < g_nspecial; item++) free(g_sp_str[item]);
    free(g_sp_str); free(g_sp_id); free(g_sp_len);
    g_sp_str = NULL; g_sp_id = NULL; g_sp_len = NULL; g_nspecial = 0;
}

static void qwen36_edge_engine_destroy(void *engine_impl) {
    Qwen36EdgeEngine *engine = (Qwen36EdgeEngine *)engine_impl;
    if (!engine) return;
    free(engine->model.embed);
    qw_free(&engine->model.lm_head);
    free(engine->model.final_norm);
    free(engine->model.c.is_attn);
    st_destroy(&engine->model.S);
    qwen36_edge_tokenizer_destroy();
    free(engine);
}

static int qwen36_edge_engine_open(
    void **engine_impl, ColiEdgeCapabilities *capabilities,
    const ColiEdgeEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return coli_edge_adapter_error(error, error_size,
                                       "invalid Qwen3.6 Edge open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_EDGE_CAP_CPU))
        return coli_edge_adapter_error(error, error_size,
                                       "Qwen3.6 Edge supports CPU only");
    /* qwen36.c's production tokenizer is process-global. The Edge runtime
     * makes that limitation explicit instead of silently cross-wiring two
     * model vocabularies in one process. Lumabri hosts one active model per
     * chatter process; a future tokenizer refactor can lift this restriction. */
    if (g_tok)
        return coli_edge_adapter_error(error, error_size,
                                       "a Qwen3.6 tokenizer is already active");
    Qwen36EdgeEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory opening Qwen3.6 Edge");
    load_cfg(&engine->model.c, options->model_dir);
    int config_layers = engine->model.c.n_layers;
    load_meta(&engine->model.c, options->model_dir);
    validate_cfg(&engine->model.c, config_layers);
    st_init(&engine->model.S, options->model_dir);
    Cfg *config = &engine->model.c;
    engine->model.embed = load_t_n(
        &engine->model, "model.embed_tokens.weight",
        (int64_t)config->vocab * config->hidden);
    /* quantize=0: this engine never ran the old post-hoc qdw_register pass
     * either (only main()'s static Model did), so lm_head stays f32-only here,
     * exactly as before. */
    load_tq(&engine->model, "lm_head.weight", config->hidden, config->vocab, 0, "lmhead", &engine->model.lm_head);
    engine->model.final_norm = load_norm_n(
        &engine->model, "model.norm.weight", config->hidden);
    engine->model.quant_bits = container_layer_is_int4(&engine->model, 0) ? 4 : 8;
    char tokenizer_path[4096];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             options->model_dir);
    load_tokenizer(tokenizer_path);
    if (!g_tok) {
        qwen36_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error, error_size,
                                       "cannot load Qwen3.6 tokenizer");
    }
    uint64_t cells = (uint64_t)config->vocab * config->hidden;
    uint64_t resident = (2u * cells + (uint64_t)config->hidden) * sizeof(float);
    if (options->memory_limit_bytes && resident > options->memory_limit_bytes) {
        qwen36_edge_engine_destroy(engine);
        return coli_edge_adapter_error(error, error_size,
                                       "Qwen3.6 Edge exceeds memory limit");
    }

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_EDGE_ABI_VERSION;
    capabilities->flags = COLI_EDGE_CAP_TOKENIZE |
                          COLI_EDGE_CAP_DETOKENIZE |
                          COLI_EDGE_CAP_GREEDY | COLI_EDGE_CAP_LOGITS |
                          COLI_EDGE_CAP_CPU;
    coli_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id), "qwen36");
    coli_edge_capability_string(capabilities->state_schema,
                                sizeof(capabilities->state_schema),
                                "qwen36/kv-deltanet-conv-f32-v1");
    snprintf(capabilities->numeric_class,
             sizeof(capabilities->numeric_class),
             "qwen36/f32-int%d/cpu-v1", engine->model.quant_bits);
    coli_edge_capability_string(capabilities->tokenizer_class,
                                sizeof(capabilities->tokenizer_class),
                                "qwen36/hf-byte-bpe-v1");
    capabilities->state_dtype = COLI_EDGE_DTYPE_F32;
    capabilities->state_width = (uint32_t)config->hidden;
    capabilities->vocab_size = (uint32_t)config->vocab;
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = QWEN36_ATTN_MAX_CTX;
    capabilities->num_layers = (uint32_t)config->n_layers;
    capabilities->bos_token_id = -1;
    capabilities->eos_token_id = -1;
    capabilities->resident_bytes = resident;
    *engine_impl = engine;
    return 0;
}

static int qwen36_edge_tokenize(
    void *engine_impl, const char *text, size_t text_bytes,
    int32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_size) {
    (void)engine_impl;
    if (!text || !token_count || text_bytes > INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "invalid Qwen3.6 tokenizer input");
    char *copy = malloc(text_bytes + 1u);
    if (!copy)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory tokenizing Qwen3.6 text");
    memcpy(copy, text, text_bytes); copy[text_bytes] = '\0';
    int *ids = NULL, count = 0;
    encode_text(copy, &ids, &count);
    free(copy);
    if (count < 0 || (token_ids && token_capacity < (size_t)count)) {
        free(ids);
        return coli_edge_adapter_error(error, error_size,
                                       "Qwen3.6 token output buffer is too small");
    }
    *token_count = (size_t)count;
    if (token_ids)
        for (int item = 0; item < count; item++) token_ids[item] = ids[item];
    free(ids);
    return 0;
}

static int qwen36_edge_detokenize(
    void *engine_impl, const int32_t *token_ids, size_t token_count,
    char *text, size_t text_capacity, size_t *text_bytes,
    char *error, size_t error_size) {
    (void)engine_impl;
    if (!token_ids || !token_count || !text_bytes || token_count > INT_MAX ||
        token_count > (SIZE_MAX - 1u) / 255u ||
        token_count > ((size_t)INT_MAX - 1u) / 255u)
        return coli_edge_adapter_error(error, error_size,
                                       "invalid Qwen3.6 detokenizer input");
    int *ids = malloc(token_count * sizeof(*ids));
    size_t capacity = token_count * 255u + 1u;
    char *temporary = malloc(capacity);
    if (!ids || !temporary) {
        free(temporary); free(ids);
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory detokenizing Qwen3.6 tokens");
    }
    for (size_t item = 0; item < token_count; item++) ids[item] = token_ids[item];
    int count = decode_range(ids, 0, (int)token_count,
                             temporary, (int)capacity);
    free(ids);
    *text_bytes = (size_t)count;
    if (text && text_capacity < (size_t)count + 1u) {
        free(temporary);
        return coli_edge_adapter_error(error, error_size,
                                       "Qwen3.6 text output buffer is too small");
    }
    if (text) memcpy(text, temporary, (size_t)count + 1u);
    free(temporary);
    return 0;
}

static int qwen36_edge_embed(void *engine_impl,
                             const ColiEdgeEmbedRequest *request,
                             char *error, size_t error_size) {
    Qwen36EdgeEngine *engine = (Qwen36EdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        int token = request->token_ids[row];
        if (token < 0 || token >= config->vocab)
            return coli_edge_adapter_error(error, error_size,
                                           "Qwen3.6 token ID is out of range");
        memcpy(output + (size_t)row * config->hidden,
               engine->model.embed + (size_t)token * config->hidden,
               (size_t)config->hidden * sizeof(float));
    }
    return 0;
}

static int qwen36_edge_select(void *engine_impl,
                              const ColiEdgeSelectRequest *request,
                              char *error, size_t error_size) {
    Qwen36EdgeEngine *engine = (Qwen36EdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    float *logits = falloc(config->vocab);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "Qwen3.6 Edge selection cancelled");
        }
        rmsnorm_row(normalized, input + (size_t)row * config->hidden,
                    engine->model.final_norm, config->hidden, config->eps);
        matmul_d(logits, normalized, &engine->model.lm_head,
                 1, config->hidden, config->vocab);
        if (coli_edge_argmax(logits, (uint32_t)config->vocab,
                            &request->token_ids[row],
                            request->scores ? &request->scores[row] : NULL)) {
            free(logits); free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "Qwen3.6 Edge head failed");
        }
    }
    free(logits); free(normalized);
    return 0;
}

static int qwen36_edge_logits(void *engine_impl,
                              const ColiEdgeLogitsRequest *request,
                              char *error, size_t error_size) {
    Qwen36EdgeEngine *engine = (Qwen36EdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(normalized);
            return coli_edge_adapter_error(error, error_size,
                                           "Qwen3.6 Edge logits cancelled");
        }
        rmsnorm_row(normalized, input + (size_t)row * config->hidden,
                    engine->model.final_norm, config->hidden, config->eps);
        matmul_d(request->logits + (size_t)row * config->vocab,
                 normalized, &engine->model.lm_head,
                 1, config->hidden, config->vocab);
    }
    free(normalized);
    return 0;
}

static const ColiEdgeAdapter qwen36_edge_adapter = {
    sizeof(ColiEdgeAdapter), COLI_EDGE_ABI_VERSION, "qwen36",
    qwen36_edge_engine_open, qwen36_edge_engine_destroy,
    qwen36_edge_tokenize, qwen36_edge_detokenize,
    qwen36_edge_embed, qwen36_edge_select, qwen36_edge_logits, {0}
};

int coli_qwen36_edge_adapter_register(void) {
    return coli_edge_adapter_register(&qwen36_edge_adapter);
}
#endif /* COLI_EDGE_ADAPTER */
