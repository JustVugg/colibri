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

/* Effective ceiling: Q36_MAXT if set and sane, the conservative default
 * otherwise; never above the hard limit. */
static int qwen36_max_ctx(void) {
    const char *e = getenv("Q36_MAXT");
    int v = (e && *e) ? atoi(e) : QWEN36_DEFAULT_MAX_CTX;
    if (v < 1) v = QWEN36_DEFAULT_MAX_CTX;
    return v > QWEN36_ATTN_MAX_CTX ? QWEN36_ATTN_MAX_CTX : v;
}
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "st.h"
#include "json.h"   /* tokenizer.json parsing (reuse minimal parser) */
#include "qwen36_tier.h"   /* optional transparent Vulkan compute backend for MoE experts */
#ifdef COLI_SEGMENT_ADAPTER
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"
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
static void smap_init(SMap *m,int cap){ m->cap=cap; m->keys=calloc(cap,sizeof(char*)); m->vals=malloc(cap*sizeof(int)); m->used=calloc(cap,sizeof(int)); }
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
static int uclass(unsigned cp){
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
static int utf8_decode(const char *s,int i,int n,int *adv){
    unsigned char c=(unsigned char)s[i]; int cp,a;
    if(c<0x80){cp=c;a=1;}
    else if((c>>5)==6){cp=c&0x1F;a=2;}
    else if((c>>4)==14){cp=c&0x0F;a=3;}
    else if((c>>3)==30){cp=c&0x07;a=4;}
    else {cp=c;a=1;}
    for(int k=1;k<a;k++){ if(i+k<n && ((unsigned char)s[i+k]&0xC0)==0x80) cp=(cp<<6)|((unsigned char)s[i+k]&0x3F); }
    if(adv)*adv=a; return cp;
}
static int utf8_adv(const char *s,int i){ int a; utf8_decode(s,i,0x7fffffff,&a); return a; }

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
static void push_id(int **ids,int *n,int *cap,int v){ if(*n==*cap){*cap*=2; *ids=realloc(*ids,*cap*sizeof(int));} (*ids)[(*n)++]=v; }

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
    if(uclass(c0)==U_W){ int k=i; while(k<n){int a;unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)==U_W)k+=a; else break;} return k; }
    return i+adv;
}
static void bpe_piece(const char *piece,int len,int **ids,int *n,int *cap){
    if(len<=0) return;
    int sc=0,scap=16; char **syms=malloc(scap*sizeof(char*));
    for(int b=0;b<len;b++){
        const char *sym=byte_sym_utf8[(unsigned char)piece[b]];
        int sl=(int)strlen(sym); char *d=malloc(sl+1); memcpy(d,sym,sl); d[sl]=0;
        if(sc==scap){scap*=2; syms=realloc(syms,scap*sizeof(char*));} syms[sc++]=d;
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
static void encode_text(const char *text,int **out_ids,int *out_n){
    int cap=1024,n=0; int *ids=malloc(cap*sizeof(int));
    int tlen=(int)strlen(text); int i=0;
    while(i<tlen){
        int sid; int L=try_special(text,i,tlen,&sid);
        if(L>0){ push_id(&ids,&n,&cap,sid); i+=L; continue; }
        int j=pretok_end(text,i,tlen); if(j<=i) j=i+utf8_adv(text,i);
        bpe_piece(text+i,j-i,&ids,&n,&cap);
        i=j;
    }
    *out_ids=ids; *out_n=n;
}

/* Load Qwen tokenizer.json and build an id->piece table. Only needs the
 * "model.vocab" map (piece string -> id); merges are irrelevant for decoding. */
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
    int mx = 0;
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>mx)mx=id; }
    } else {
        mx = vocab->len - 1;
    }
    g_tok = calloc((size_t)mx+1, sizeof(char*));
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>=0 && id<=mx) g_tok[id]=strdup(vocab->keys[i]); }
    } else {
        for (int i=0;i<vocab->len;i++){ if(vocab->kids[i] && vocab->kids[i]->t==J_STR) g_tok[i]=strdup(vocab->kids[i]->str); }
    }
    g_tok_n = mx+1;

    /* ---- encoder tables (text -> ids) ---- */
    smap_init(&g_rev, 1<<19);
    for (int i=0;i<g_tok_n;i++) if (g_tok[i]) smap_put(&g_rev, g_tok[i], i);

    smap_init(&g_merge, 1<<19);
    jval *merges = json_get(model, "merges");
    if (merges && merges->t==J_ARR){
        for (int r=0;r<merges->len;r++){
            const char *e = merges->kids[r]->str; if(!e) continue;
            const char *sp = strchr(e, ' '); if(!sp) continue;
            int la=(int)(sp-e), lb=(int)strlen(sp+1);
            char *key=malloc(la+1+lb+1);
            memcpy(key,e,la); key[la]=0x1F; memcpy(key+la+1,sp+1,lb); key[la+1+lb]=0;
            smap_put(&g_merge, key, r);
        }
    }
    jval *adds = json_get(root, "added_tokens");
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
        if (c == '"'){ if(o+2<outsz){ out[o++]='"'; out[o++]='"'; } }
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

/* Append b[0..n) into buf/*bn, extract as many LEADING complete UTF-8
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
    if (!g_tok || id<0 || id>=g_tok_n) return;
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
            char jb[160];
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
            char jb[1200];
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
                char jb[400];
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
    uint8_t *is_attn;   /* [n_layers] 1 if Gated Attention layer, 0 if DeltaNet */
    /* Gated DeltaNet (linear_attention) dims, read from qwen36_meta.json. */
    int dn_vheads, dn_kheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim;
    int expert_gs;      /* expert scale group size along input dim; 0 = per-row */
} Cfg;

/* ---------- per-layer dense weights ---------- */
typedef struct {
    float *in_ln, *post_ln, *q, *k, *v, *o, *qn, *kn, *gate, *gate_bias;
    float *sh_g, *sh_u, *sh_d, *sh_gate;   /* shared expert (dense f32) + shared_expert_gate */
    /* Gated DeltaNet (linear_attention) dense weights (f16->f32 via st_read_f32). */
    float *dn_qkv, *dn_z, *dn_b, *dn_a;    /* in_proj_qkv/z/b/a */
    float *dn_conv;                        /* conv1d.weight [conv_dim, convk] (groups=conv_dim) */
    float *dn_dtbias, *dn_alog;            /* dt_bias[vh], A_log[vh] */
    float *dn_norm;                        /* RMSNormGated weight [vdim] */
    float *dn_out;                         /* out_proj [hidden, value_dim] */
} Layer;

/* ---------- LRU expert cache (int8 weights + per-row float scales) ---------- */
typedef struct { int eid; int pinned; int is_int4; int8_t *g, *u, *d; uint8_t *g4, *u4, *d4; float *gs, *us, *ds; uint64_t used; } Slot;
typedef struct {
    Slot *slots;
    int *slot_by_expert;                  /* expert id -> resident slot, -1 if absent */
    int n, cap;
} LCache;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;          /* [n_layers] */
    int *active_of;         /* [n_layers] original->active idx (Phase 2: identity for all layers) */
    float **DN_rec;         /* [n_layers] recurrent state S[h]=[kdim,vdim] for DeltaNet layers (NULL for attn) */
    float **DN_conv;        /* [n_layers] conv ring [conv_dim, convk-1] for DeltaNet layers (NULL for attn) */
    uint64_t clock, hits, miss;
    float **K, **V; int kv_len, max_t, kv_cap;
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
    int first_step;            /* the first step() call is the prefill */
} Model;

static pthread_mutex_t g_pilot_mx = PTHREAD_MUTEX_INITIALIZER;
static struct { int l, e; } pilot_q[4096];
static volatile unsigned pilot_r = 0, pilot_w = 0;
static Model *pilot_m = NULL;
static int g_pilot = 0;
static int g_wide  = 1;

static void pilot_prefetch(Model *m, int lnext, const float *x, int S);
static void *pilot_worker(void *arg);
static void ensure_pilot_worker_started(Model *m);
static void slot_ensure_allocated(Model *m, Slot *s);

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

/* ---- M-PROF (R2): per-phase wall-clock accumulators, COLI_TIMERS=1 ---- */
static int g_timers = -1;
static double g_tm_dec[6], g_tm_pre[6];   /* 0=deltanet 1=attention 2=moe_total 3=shared 4=router 5=lm_head */
static long g_tm_dec_tokens = 0, g_tm_pre_tokens = 0;
static double tm_now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3 + ts.tv_nsec/1e6; }
static int tm_on(void){ if(g_timers<0){ const char *e=getenv("COLI_TIMERS"); g_timers = (e && *e=='1'); } return g_timers; }
double g_qt_iss=0, g_qt_cpu=0, g_qt_tak=0;   /* QTIER-Phasen (Decode) */
double g_dn_sub[4];                           /* DN: proj, conv+split, l2n+rec, norm+out */
double g_tm_step=0;                           /* step() total (decode) */
static double g_tm_win_moe=0; static int g_tm_win_n=0;
static void tm_add(int S, int idx, double ms){
    if(S==1){
        g_tm_dec[idx]+=ms;
        if(idx==2) g_tm_win_moe+=ms;
        if(idx==5 && ++g_tm_win_n==32){
            fprintf(stderr,"[timers] window: moe %.0f ms/token (last 32)\n", g_tm_win_moe/32.0);
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
    if(g_qt_iss+g_qt_cpu+g_qt_tak>0)
        fprintf(stderr,"[timers]   qtier: issue %.2f | cpu-miss %.2f | take %.2f ms/token\n",
                g_qt_iss/g_tm_dec_tokens, g_qt_cpu/g_tm_dec_tokens, g_qt_tak/g_tm_dec_tokens);
    fprintf(stderr,"[timers] prefill: %ld tokens  dn=%.0f attn=%.0f moe=%.0f(sh=%.0f rt=%.0f) head=%.0f ms\n",
            g_tm_pre_tokens,g_tm_pre[0],g_tm_pre[1],g_tm_pre[2],g_tm_pre[3],g_tm_pre[4],g_tm_pre[5]);
}
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T,  W is [O,I] row-major */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[1,O] = x[1,I] @ W^T with W quantized: q[O,I] int8 + scale per row. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#endif
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    /* IDOT is opt-in, not default-on: this path quantizes the ACTIVATIONS to
     * Q8_0 per 16-element block, which the scalar path does not, so the two are
     * not numerically equivalent. olmoe shipped it default-on and it cost
     * token-exactness end to end (#1044, fixed in af48fe8 by making it opt-in);
     * qwen36 inherited the same default from the same family of kernels. The
     * tiny-oracle gate would not have caught it -- that job runs on x86. */
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && atoi(e)); }
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I / 16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    /* Hand-vectorized int8->f32 GEMV (gcc does not auto-vectorize the
     * convert+accumulate chain). 32 weights per iteration, FMA accumulate. */
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 32 <= I; i += 32) {
            __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
            __m128i b1 = _mm_loadu_si128((const __m128i*)(w + i + 16));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8))), a3);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
        s = _mm_add_ps(s, _mm_movehl_ps(s,s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
        float acc = _mm_cvtss_f32(s);
        for (; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#else
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#endif
}

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
 * (gs64 expert containers). Row layout of `scale`: [O][I/gs] row-major. */
static int g_expert_gs = 0;   /* set from qwen36_meta.json (expert_gs) at load */
static void matmul_q_gs(float *y, const float *x, const int8_t *q, const float *scale,
                        int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
#if defined(__AVX2__) && defined(__FMA__)
    if ((gs & 31) == 0) {
        #pragma omp parallel for schedule(static) if(O >= 256)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            const float *sc = scale + (int64_t)o * ng;
            float acc = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 16 <= end; i += 16) {
                    __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
                }
                a0 = _mm256_add_ps(a0, a1);
                __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
                s = _mm_add_ps(s, _mm_movehl_ps(s,s));
                s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
                acc += _mm_cvtss_f32(s) * sc[gi];
            }
            y[o] = acc;
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        const float *sc = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int gi = 0; gi < ng; gi++) {
            int base = gi * gs, end = base + gs; if (end > I) end = I;
            float part = 0.f;
            for (int i = base; i < end; i++) part += x[i] * (float)w[i];
            acc += part * sc[gi];
        }
        y[o] = acc;
    }
}
/* Expert-GEMV dispatch: per-row scales (classic) or grouped (gs64 container). */
static void matmul_qe(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    if (g_expert_gs) matmul_q_gs(y, x, q, scale, I, O, g_expert_gs);
    else matmul_q(y, x, q, scale, I, O);
}

/* ---- Dense int8: per-row quantized copies of the large f32 matrices.
 * matmul_d dispatches via pointer lookup to matmul_q; COLI_DENSE_I8=0 falls
 * back to f32 (reference path for parity tests). ~4x less memory traffic. */
#define QDW_MAX 1024
static struct { const float *w; int8_t *q; float *sc; int I, O; } g_qdw[QDW_MAX];
static int g_qdw_n = 0;
#ifdef COLI_QWEN_BATCH_TEST
static uint64_t g_qwen_matmul_d_calls;
#endif
static int dense_i8_on(void){ static int v=-1; if(v<0){ const char *e=getenv("COLI_DENSE_I8"); v=!(e&&*e=='0'); } return v; }
static int dense_batch_on(void){ const char *e=getenv("QWEN_DENSE_BATCH"); return !(e&&*e=='0'); }
static void qdw_register(const float *W, int I, int O){
    if (!W || !dense_i8_on() || g_qdw_n >= QDW_MAX) return;
    int8_t *q = malloc((size_t)O*I); float *sc = malloc((size_t)O*sizeof(float));
    if (!q || !sc) { free(q); free(sc); return; }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *r = W + (int64_t)o*I; float am = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am/127.f : 1.f; sc[o] = s; float inv = 1.f/s;
        int8_t *d = q + (int64_t)o*I;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(r[i]*inv); if (v>127) v=127; if (v<-127) v=-127; d[i] = (int8_t)v; }
    }
    g_qdw[g_qdw_n].w=W; g_qdw[g_qdw_n].q=q; g_qdw[g_qdw_n].sc=sc; g_qdw[g_qdw_n].I=I; g_qdw[g_qdw_n].O=O; g_qdw_n++;
}
static void matmul_d(float *y, const float *x, const float *W, int S, int I, int O){
#ifdef COLI_QWEN_BATCH_TEST
    g_qwen_matmul_d_calls++;
#endif
    for (int i = 0; i < g_qdw_n; i++) if (g_qdw[i].w == W && g_qdw[i].I == I) {
        if (S > 1 && dense_batch_on())
            matmul_q_batch(y, x, g_qdw[i].q, g_qdw[i].sc, S, I, O);
        else
            for (int s = 0; s < S; s++) matmul_q(y+(int64_t)s*O, x+(int64_t)s*I, g_qdw[i].q, g_qdw[i].sc, I, O);
        return;
    }
    matmul(y, x, W, S, I, O);
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
    c->attn_output_gate = 1; c->n_active = 0;
    c->is_attn = calloc(c->n_layers, sizeof(uint8_t));
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
    CFG_NEED(c->n_experts > 0 && c->n_experts <= 1024,
             "num_experts %d out of range 1..1024 (keep[] in moe())", c->n_experts);
    CFG_NEED(c->topk > 0 && c->topk <= 256,
             "topk %d out of range 1..256 (idx[]/val[] in moe())", c->topk);
    CFG_NEED(c->topk <= c->n_experts, "topk %d exceeds num_experts %d",
             c->topk, c->n_experts);
    CFG_NEED(c->inter > 0 && c->shared_inter > 0,
             "moe_inter %d / shared_inter %d must be positive", c->inter, c->shared_inter);
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
        G("num_experts", n_experts); G("topk", topk);
        G("moe_inter", inter); G("shared_inter", shared_inter);
        G("n_group", n_group); G("topk_group", topk_group);
        G("dn_vheads", dn_vheads); G("dn_kheads", dn_kheads); G("dn_kdim", dn_kdim);
        G("dn_vdim", dn_vdim); G("dn_convk", dn_convk); G("dn_conv_dim", dn_conv_dim);
        #undef G
        if((v=json_get(r,"partial_rotary_factor"))&&v->t==J_NUM) c->partial_rotary_factor=(float)v->num;
        if((v=json_get(r,"rope_theta"))&&v->t==J_NUM) c->theta=(float)v->num;
        if((v=json_get(r,"rms_eps"))&&v->t==J_NUM) c->eps=(float)v->num;
        if((v=json_get(r,"attn_output_gate"))&&v->t==J_BOOL) c->attn_output_gate=v->boolean;
        if((v=json_get(r,"norm_topk_prob"))&&v->t==J_BOOL) c->norm_topk=v->boolean;
        if((v=json_get(r,"has_bias"))&&v->t==J_BOOL) c->has_bias=v->boolean;
        if((v=json_get(r,"has_qk_norm"))&&v->t==J_BOOL) c->has_qk_norm=v->boolean;
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

/* `want` is the element count the forward pass will index with. The container
 * is a file, not an invariant: this used to allocate whatever st_numel reported
 * while every read afterwards used CONFIG dims, so a short tensor was a plain
 * heap OOB read (embed is indexed as m->embed + ids[s]*D). The expert path
 * already refuses a wrong size; this is the same discipline for the dense set. */
static float *load_t_n(Model *m, const char *name, int64_t want) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (want > 0 && n != want) {
        fprintf(stderr, "%s: %lld elements, config implies %lld -- refusing\n",
                name, (long long)n, (long long)want); exit(1);
    }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
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
    if (load_boundaries) {
        m->embed      = load_t_n(m, "model.embed_tokens.weight", (int64_t)c->vocab * c->hidden);
        m->lm_head    = load_t_n(m, "lm_head.weight", (int64_t)c->vocab * c->hidden);
        m->final_norm = load_t_n(m, "model.norm.weight", c->hidden);
    }
    m->L = calloc(c->n_layers, sizeof(Layer));
    /* Phase 2: the converter stores EVERY layer (Gated-Attention + Gated DeltaNet)
     * under its OWN original index model.layers.{i}. So active_of is the identity
     * map; experts and dense weights are read from model.layers.{i} for all i. */
    m->active_of = malloc((size_t)c->n_layers * sizeof(int));
    for (int i = 0; i < c->n_layers; i++) m->active_of[i] = i;
    char nm[256];
    for (int i = layer_begin; i < layer_end; i++) {
        int ai = m->active_of[i];        /* == i for Phase 2 */
        Layer *l = &m->L[i];
        /* input/post layernorms + MoE exist for every layer */
        #define LD(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,ai); l->field = load_t_n(m,nm,(want))
        LD(in_ln,  "input_layernorm.weight", c->hidden);
        LD(post_ln,"post_attention_layernorm.weight", c->hidden);
        LD(gate, "mlp.gate.weight", (int64_t)c->n_experts * c->hidden);
        #undef LD
        /* q/k norms are per-head [head_dim]; only on attention layers, load if present */
        if (c->has_qk_norm) {
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_norm.weight", ai);
            l->qn = st_has(&m->S, nm) ? load_t_n(m, nm, c->head_dim) : NULL;
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_norm.weight", ai);
            l->kn = st_has(&m->S, nm) ? load_t_n(m, nm, c->head_dim) : NULL;
        } else { l->qn = NULL; l->kn = NULL; }
        /* router correction bias (optional) */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.e_score_correction_bias", ai);
        if (st_has(&m->S, nm)) { l->gate_bias = falloc(c->n_experts); st_read_f32(&m->S, nm, l->gate_bias, 0); }
        else l->gate_bias = NULL;
        /* shared expert (dense f32) */
        #define LD2(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert." suffix,ai); l->field = load_t_n(m,nm,(want))
        LD2(sh_g, "gate_proj.weight", (int64_t)c->shared_inter * c->hidden);
        LD2(sh_u, "up_proj.weight",   (int64_t)c->shared_inter * c->hidden);
        LD2(sh_d, "down_proj.weight", (int64_t)c->hidden * c->shared_inter);
        #undef LD2
        /* shared_expert_gate: Linear(hidden -> 1), sigmoid-gated shared expert */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert_gate.weight", ai);
        l->sh_gate = st_has(&m->S, nm) ? load_t_n(m, nm, c->hidden) : NULL;
        if (c->is_attn[i]) {
            /* Gated Attention (full_attention) layer */
            #define LD3(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d.self_attn." suffix,ai); l->field = load_t_n(m,nm,(want))
            LD3(q, "q_proj.weight", (int64_t)c->q_heads * c->q_head_dim * c->hidden);
            LD3(k, "k_proj.weight", (int64_t)c->kv_heads * c->k_head_dim * c->hidden);
            LD3(v, "v_proj.weight", (int64_t)c->kv_heads * c->v_head_dim * c->hidden);
            LD3(o, "o_proj.weight", (int64_t)c->hidden * c->o_in);
            #undef LD3
            l->dn_qkv=l->dn_z=l->dn_b=l->dn_a=l->dn_conv=NULL;
            l->dn_dtbias=l->dn_alog=l->dn_norm=l->dn_out=NULL;
        } else {
            /* Gated DeltaNet (linear_attention) layer */
            l->q=l->k=l->v=l->o=NULL;
            #define LD4(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn." suffix,ai); l->field = load_t_n(m,nm,(want))
            int64_t vdim_tot = (int64_t)c->dn_vheads * c->dn_vdim;
            LD4(dn_qkv, "in_proj_qkv.weight", (int64_t)c->dn_conv_dim * c->hidden);
            LD4(dn_z,   "in_proj_z.weight",   vdim_tot * c->hidden);
            LD4(dn_b,   "in_proj_b.weight",   (int64_t)c->dn_vheads * c->hidden);
            LD4(dn_a,   "in_proj_a.weight",   (int64_t)c->dn_vheads * c->hidden);
            LD4(dn_conv,"conv1d.weight",      (int64_t)c->dn_conv_dim * c->dn_convk);
            LD4(dn_dtbias, "dt_bias",         c->dn_vheads);
            LD4(dn_alog,"A_log",              c->dn_vheads);
            LD4(dn_norm, "norm.weight",       c->dn_vdim);
            LD4(dn_out, "out_proj.weight",    (int64_t)c->hidden * vdim_tot);
            #undef LD4
        }
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = layer_begin; i < layer_end; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
        m->cache[i].slot_by_expert = malloc((size_t)c->n_experts * sizeof(int));
        if (!m->cache[i].slot_by_expert) { fprintf(stderr,"OOM expert cache index\n"); exit(1); }
        for (int e = 0; e < c->n_experts; e++) m->cache[i].slot_by_expert[e] = -1;
    }
    /* per-layer DeltaNet recurrent + conv state (only for linear_attention layers) */
    m->DN_rec = calloc(c->n_layers, sizeof(float*));
    m->DN_conv = calloc(c->n_layers, sizeof(float*));
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
static int64_t scale_count_d (const Cfg *c){ return c->expert_gs ? (int64_t)c->hidden * ((c->inter  + c->expert_gs - 1) / c->expert_gs) : c->hidden; }

static void slot_ensure_allocated(Model *m, Slot *s) {
    if (s->g) return;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden;
    int64_t nd = (int64_t)c->hidden * c->inter;
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
    if (!tw || (tw->nbytes != want_w && tw->nbytes != want_w / 2)) {
        fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld (int8) or %lld (int4)\n",
                nm, (long long)(tw ? tw->nbytes : -1), (long long)want_w, (long long)(want_w / 2)); exit(1); }
    if (!ts || ts->numel != want_s) {
        fprintf(stderr, "%s: scale array is %lld elems — expected %lld (refusing)\n",
                qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1); }
    /* int4 detection by ON-DISK SIZE (robust against a mislabeled meta.ebits, e.g. the
       i8 container whose meta says ebits=4 but stores int8).  True int4 packed uint8 is
       exactly N/2 bytes (N = 3*inter*hidden, always even).  Unpack in-place to int8 so the
       rest of the MoE path (matmul_q) is unchanged.  Nibble convention (must match
       c/tools/convert_qwen36.py pack_int4): LOW nibble = element 2k, HIGH nibble = 2k+1;
       each nibble is signed 4-bit (sign-extend if bit3 set). */
    if (tw->nbytes == want_w / 2) {
        static int noted = 0;
        if (!noted) { fprintf(stderr, "[qwen36] int4 packed weights detected — unpacking to int8 in slot\n"); noted = 1; }
        uint8_t *raw = (uint8_t *)malloc((size_t)(want_w / 2));
        if (!raw) { fprintf(stderr, "OOM reading int4 expert %s\n", nm); exit(1); }
        st_read_raw(&m->S, nm, raw, 1);
        for (int64_t i = 0; i < want_w; i++) {
            uint8_t byte = raw[i >> 1];
            int8_t v = (int8_t)((i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF));
            if (v & 8) v -= 16;                 /* sign-extend signed 4-bit */
            s->g[i] = v;
        }
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
    const uint8_t *src4[3] = { s->g4, s->u4, s->d4 };
    int64_t lens[3] = { ng, ng, nd };
    int8_t *dst = w;
    for (int t = 0; t < 3; t++) {
        const uint8_t *p = src4[t];
        for (int64_t i = 0; i < lens[t]; i += 2) {
            uint8_t b = p[i >> 1];
            int8_t lo = (int8_t)(b & 0xF); if (lo & 8) lo -= 16;
            int8_t hi = (int8_t)((b >> 4) & 0xF); if (hi & 8) hi -= 16;
            dst[i] = lo; dst[i + 1] = hi;
        }
        dst += lens[t];
    }
    s->g = w; s->u = w + ng; s->d = w + ng + ng;
}

static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_pilot_mx);
    Slot *hit = slot_indexed(m, layer, eid);
    if (hit) {
        m->hits++; hit->used = ++m->clock; *out = hit;
        pthread_mutex_unlock(&g_pilot_mx); return;
    }
    m->miss++;
    Cfg *c = &m->c; Slot *s;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        /* LRU eviction — skip pinned and in-flight (eid==-1) slots */
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0) {
            /* All slots are pinned or in-flight; find the oldest non-in-flight
             * slot (may be pinned, but never one currently being loaded). */
            for (int i = 0; i < lc->n; i++) { if (lc->slots[i].eid < 0) continue; if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i; }
        }
        while (lru < 0) {
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
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        s = &lc->slots[lru]; s->pinned = 0;
    }
    cache_hide(m, layer, s); s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);
    load_expert_merged(m, layer, eid, s);
    pthread_mutex_lock(&g_pilot_mx);
    cache_publish(m, layer, s, eid); s->pinned = m->is_pinned[layer * c->n_experts + eid]; s->used = ++m->clock;
    *out = s; pthread_mutex_unlock(&g_pilot_mx);
}

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
    matmul_d(q, x, l->q, S, D, q_out);
    matmul_d(k, x, l->k, S, D, kv_out);
    matmul_d(vv, x, l->v, S, D, kv_out);
    /* split q into query (first hd) and gate (next gate_dim), both per head */
    float *query = falloc((int64_t)S*H*hd);
    float *gate  = falloc((int64_t)S*H*gate_dim);
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            const float *qs = q + (int64_t)s*q_out + hh*qdim;
            memcpy(query + ((int64_t)s*H + hh)*hd, qs, hd*sizeof(float));
            if (gate_dim) memcpy(gate + ((int64_t)s*H + hh)*gate_dim, qs + hd, gate_dim*sizeof(float));
        }
    }
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            float *qh = query + ((int64_t)s*H + hh)*hd;
            if (l->qn) rmsnorm_row(qh, qh, l->qn, hd, c->eps);
            rope_head_partial(qh, pos_base + s, rotary, hd, c->theta);
        }
        for (int kvh = 0; kvh < KV; kvh++) {
            float *kh = k + (int64_t)s*KV*kvd + kvh*kvd;
            if (l->kn) rmsnorm_row(kh, kh, l->kn, kvd, c->eps);
            rope_head_partial(kh, pos_base + s, rotary, kvd, c->theta);
        }
    }
    for (int s = 0; s < S; s++) for (int kvh = 0; kvh < KV; kvh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)kvh*m->max_t + t)*kvd, k + (int64_t)s*KV*kvd + kvh*kvd, kvd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)kvh*m->max_t + t)*kvd, vv + (int64_t)s*KV*kvd + kvh*kvd, kvd*sizeof(float));
    }
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*H*hd);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int kvh = hh / q_per_kv;
            int qpos = pos_base + s;
            const float *qv = query + ((int64_t)s*H + hh)*hd;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = m->attn_sc + (int64_t)tid * m->kv_cap;
            for (int t = 0; t <= qpos; t++) {
                const float *kv = m->K[layer] + ((int64_t)kvh*m->max_t + t)*kvd;
                float acc = 0; for (int dd = 0; dd < kvd; dd++) acc += qv[dd]*kv[dd];
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + ((int64_t)s*H + hh)*hd;
            for (int dd = 0; dd < kvd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = m->V[layer] + ((int64_t)kvh*m->max_t + t)*kvd;
                float a = sc[t]; for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    /* apply attn_output_gate: attn_out *= sigmoid(gate) */
    float *ag = falloc((int64_t)S*H*hd);
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) for (int dd = 0; dd < hd; dd++) {
        int o = ((int64_t)s*H + hh)*hd + dd;
        float g = gate_dim ? gate[o] : 0.f;
        ag[o] = ctx[o] * (1.f / (1.f + expf(-g)));
    }
    matmul_d(out, ag, l->o, S, H*hd, D);
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
    int B=qwen_shared_batch_rows(S,D,I);
    double _ts=tm_on()?tm_now():0.0;
    if (B == 1) {
        for (int s=0;s<S;s++) {
            const float *xs=x+(int64_t)s*D;
            matmul_d(g,xs,l->sh_g,1,D,I);
            matmul_d(u,xs,l->sh_u,1,D,I);
            for(int i=0;i<I;i++){float sv=g[i];g[i]=(sv/(1.f+expf(-sv)))*u[i];}
            matmul_d(hh,g,l->sh_d,1,I,D);
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
            matmul_d(bg,x+(int64_t)base*D,l->sh_g,rows,D,I);
            matmul_d(bu,x+(int64_t)base*D,l->sh_u,rows,D,I);
            for(int64_t q=0;q<(int64_t)rows*I;q++){float sv=bg[q];bg[q]=(sv/(1.f+expf(-sv)))*bu[q];}
            matmul_d(bh,bg,l->sh_d,rows,I,D);
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
    if(tm_on())tm_add(S,3,tm_now()-_ts);
}

/* MoE: grouped top-k routing (+ optional router bias) + shared expert.
 * Mirrors HF Qwen3 MoE: softmax(gate), optional group-limited top-k, normalized
 * weights, sum routed experts, then add the un-gated shared expert. */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    double _tr = tm_on() ? tm_now() : 0.0;
    matmul_d(logits, x, l->gate, S, D, E);
    if (tm_on()) tm_add(S, 4, tm_now()-_tr);
    if (c->has_bias && l->gate_bias) {
        for (int s = 0; s < S; s++) { float *pr = logits + (int64_t)s*E; for (int e = 0; e < E; e++) pr[e] += l->gate_bias[e]; }
    }
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    float *sh = falloc(I), *shu = falloc(I), *shd = falloc(D);  /* shared expert scratch */
    int use_qt = qt_ready();
    for (int s = 0; s < S; s++) {
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
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                if (!keep[e]) continue;
                int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
                if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
            }
            idx[kk] = best; val[kk] = bv;
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
        const float *xs = x + (int64_t)s*D;
        if (use_qt) {
            /* CUDA expert tier: run the resident experts as async groups on
             * all devices, compute the misses on the CPU (overlapped), then
             * collect the GPU results. */
            for (int kk = 0; kk < K; kk++) {
                Slot *e; expert_get(m, layer, idx[kk], &e);
                if (e->g4) qt_note(layer, idx[kk], e->g4, e->u4, e->d4, e->gs, e->us, e->ds);
            }
            double _q0 = tm_on()? tm_now():0;
            uint32_t qmask = qt_issue(layer, idx, K, xs);
            double _q1 = tm_on()? tm_now():0;
            for (int kk = 0; kk < K; kk++) {
                if (qmask & (1u<<kk)) continue;
                Slot *e; expert_get(m, layer, idx[kk], &e);
                slot_ensure_int8(m, e);
                matmul_qe(g, xs, e->g, e->gs, D, I);
                matmul_qe(u, xs, e->u, e->us, D, I);
                for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                matmul_qe(hh, g, e->d, e->ds, I, D);
                float w = val[kk]; float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += w * hh[d];
            }
            /* Compute the shared expert NOW so it overlaps with the GPU
             * groups; the common block below is skipped. */
            {
                double _ts2 = tm_on() ? tm_now() : 0.0;
                int Ish = c->shared_inter;
                matmul_d(sh, xs, l->sh_g, 1, D, Ish);
                matmul_d(shu, xs, l->sh_u, 1, D, Ish);
                for (int i = 0; i < Ish; i++) { float sv = sh[i]; sh[i] = (sv / (1.f + expf(-sv))) * shu[i]; }
                matmul_d(shd, sh, l->sh_d, 1, Ish, D);
                float sgate = 1.f;
                if (l->sh_gate) {
                    float sg = 0.f; const float *wg = l->sh_gate;
                    for (int i = 0; i < D; i++) sg += xs[i] * wg[i];
                    sgate = 1.f / (1.f + expf(-sg));
                }
                float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += sgate * shd[d];
                if (tm_on()) tm_add(S, 3, tm_now()-_ts2);
            }
            double _q2 = tm_on()? tm_now():0;
            qt_take(qmask, val, K, out + (int64_t)s*D);
            if (tm_on() && S==1) {
                extern double g_qt_iss, g_qt_cpu, g_qt_tak;
                g_qt_iss += _q1-_q0; g_qt_cpu += _q2-_q1; g_qt_tak += tm_now()-_q2;
            }
        } else {
            for (int kk = 0; kk < K; kk++) {
                Slot *e; expert_get(m, layer, idx[kk], &e);
                slot_ensure_int8(m, e);
                matmul_qe(g, xs, e->g, e->gs, D, I);
                matmul_qe(u, xs, e->u, e->us, D, I);
                for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                matmul_qe(hh, g, e->d, e->ds, I, D);
                float w = val[kk];
                float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += w * hh[d];
            }
        }
    }
    /* The CUDA tier keeps its per-token shared block above because it overlaps
     * resident GPU experts.  CPU prefill instead traverses each shared matrix
     * once per bounded chunk. */
    if (!use_qt) qwen_shared_experts_cpu(m,l,x,S,out,sh,shu,shd);
    free(logits); free(g); free(u); free(hh); free(sh); free(shu); free(shd);
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

    float *qkv = falloc(conv_dim);
    float *z   = falloc(value_dim);
    float *b   = falloc(vh);
    float *a   = falloc(vh);
    float *beta= falloc(vh);
    float *gg  = falloc(vh);
    float *conv_out = falloc(conv_dim);
    float *q = falloc(vh * kdim);
    float *k = falloc(vh * kdim);
    float *outv = falloc(value_dim);
    float *outr = falloc(value_dim);
    float *kv = falloc(vdim);
    float *delta = falloc(vdim);

    float *rec = m->DN_rec[layer];      /* [vh*kdim*vdim] */
    float *ring = m->DN_conv[layer];    /* [conv_dim*(convk-1)] */

    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s * H;
        extern double g_dn_sub[4];
        double _d0 = tm_on()? tm_now():0;
        /* projections (single-token matmuls) */
        matmul_d(qkv, xs, l->dn_qkv, 1, H, conv_dim);
        matmul_d(z,   xs, l->dn_z,   1, H, value_dim);
        matmul(b,   xs, l->dn_b,   1, H, vh);
        matmul(a,   xs, l->dn_a,   1, H, vh);
        if (tm_on() && S==1){ double t=tm_now(); g_dn_sub[0]+=t-_d0; _d0=t; }
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
        /* per-head Gated RMSNorm (plain weight, r=1/sqrt(mean+eps)) then silu(z) gate, then out_proj.
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
        matmul_d(out + (int64_t)s * H, outr, l->dn_out, 1, value_dim, H);
        if (tm_on() && S==1){ g_dn_sub[3]+=tm_now()-_d0; }
        if (layer == 0 && s == 0 && getenv("DN_DBG")) {
            FILE *dbg = fopen(getenv("DN_DBG"), "wb");
            if (dbg) {
                fwrite(conv_out, sizeof(float), conv_dim, dbg);
                fwrite(q, sizeof(float), (int64_t)vh * kdim, dbg);
                fwrite(outv, sizeof(float), value_dim, dbg);
                fwrite(z, sizeof(float), value_dim, dbg);
                fwrite(outr, sizeof(float), value_dim, dbg);
                fwrite(out + (int64_t)s * H, sizeof(float), H, dbg);
                fwrite(b, sizeof(float), vh, dbg);
                fwrite(a, sizeof(float), vh, dbg);
                fwrite(beta, sizeof(float), vh, dbg);
                fwrite(gg, sizeof(float), vh, dbg);
                fclose(dbg);
            }
        }
    }
    free(qkv); free(z); free(b); free(a); free(beta); free(gg);
    free(conv_out); free(q); free(k); free(outv); free(outr); free(kv); free(delta);
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
        double _t0 = tm_on() ? tm_now() : 0.0;
        if (c->is_attn[i]) {
            attention(m, l, i, nrm, S, pos_base, tmp);
            if (tm_on()) tm_add(S, 1, tm_now()-_t0);
        } else {
            deltanet(m, l, i, nrm, S, pos_base, tmp);
            if (tm_on()) tm_add(S, 0, tm_now()-_t0);
        }
        if (lf) fwrite(tmp + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* sublayer output */
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* post-deltanet residual */
        if (allow_prefetch && g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        _t0 = tm_on() ? tm_now() : 0.0;
        moe(m, l, i, nrm, S, tmp);
        if (tm_on()) tm_add(S, 2, tm_now()-_t0);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);
        if (allow_prefetch && g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (allow_prefetch && g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
    }
    free(nrm); free(tmp);
}

static float *step(Model *m, const int *ids, int S, int pos_base) {
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
        memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    }
    layers_forward_range(m, x, S, pos_base, 0, c->n_layers, 1, lf);
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens) pin_hot_experts(m);
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    double _th = tm_on() ? tm_now() : 0.0;
    matmul_d(logit, last, m->lm_head, 1, D, c->vocab);
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

static void pilot_realload(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    pthread_mutex_lock(&g_pilot_mx);
    if (!m->is_queued[layer * c->n_experts + eid]) { pthread_mutex_unlock(&g_pilot_mx); return; }
    if (slot_indexed(m, layer, eid)) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
    Slot *s;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        int lru = -1;
        for (int i = 0; i < lc->n; i++) { if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue; if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i; }
        if (lru < 0) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
        s = &lc->slots[lru]; s->pinned = 0;
    }
    cache_hide(m, layer, s); s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);
    load_expert_merged(m, layer, eid, s);
    pthread_mutex_lock(&g_pilot_mx);
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
    if (lnext < 0 || lnext >= m->c.n_layers) return;
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts;
    ensure_pilot_worker_started(m);
    float *logits = falloc((int64_t)S * E);
    Layer *l = &m->L[lnext];
    float *nrm_x = falloc((int64_t)S * D);
    for (int s = 0; s < S; s++) rmsnorm_row(nrm_x + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
    matmul_d(logits, nrm_x, l->gate, S, D, E);   /* int8 copy (f32 may be freed) */
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
                if (ps->g4) qt_note(lnext, eid, ps->g4, ps->u4, ps->d4, ps->gs, ps->us, ps->ds);
            }
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
static void reset_recurrent(Model *m){
    Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++){
        if (c->is_attn[i]) continue;
        if (m->DN_rec[i])  memset(m->DN_rec[i],  0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        if (m->DN_conv[i]) memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
}

/* Allocate (once) or reuse the KV cache across requests. Grows only when a
 * longer context is needed; never shrinks. Frees the previous buffers on
 * growth so the server doesn't leak KV memory across requests. */
static void ensure_kv(Model *m){
    Cfg *c = &m->c;
    if (m->kv_cap >= m->max_t && m->K) return;
    if (m->K){
        for (int i = 0; i < c->n_layers; i++){ if (m->K[i]) free(m->K[i]); if (m->V[i]) free(m->V[i]); }
        free(m->K); free(m->V); m->K = NULL; m->V = NULL;
    }
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++){
        if (c->is_attn[i]){
            m->K[i] = falloc((int64_t)c->kv_heads * m->max_t * c->k_head_dim);
            m->V[i] = falloc((int64_t)c->kv_heads * m->max_t * c->k_head_dim);
        } else { m->K[i] = NULL; m->V[i] = NULL; }
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
}

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
    int len = np;
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
        int one = best;
        { extern double g_tm_step; double _s0 = tm_on()? tm_now():0;
          logit = step(m, &one, 1, len - 1);
          if (tm_on()) g_tm_step += tm_now()-_s0; }
    }
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

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } ServeReq;

static int serve_read_req(ServeReq *q){
    char line[512], cmd[16], id[64];
    if(!fgets(line,sizeof(line),stdin)) return -1;
    if(sscanf(line,"%15s %63s",cmd,id)<2) return 0;
    if(!strcmp(cmd,"CANCEL")||!strcmp(cmd,"STOP")) return 0;
    if(strcmp(cmd,"SUBMIT")) return 0;
    int slot, plen, max_tok; float temp, top_p;
    if(sscanf(line,"%*s %*s %d %d %d %f %f",&slot,&plen,&max_tok,&temp,&top_p)!=5 ||
       plen<0||plen>(1<<24)||max_tok<1){
        printf("ERROR %s bad submit header\n",id); fflush(stdout); return 0;
    }
    (void)slot;
    char *payload=malloc((size_t)plen+1);
    if(!payload){ printf("ERROR %s out of memory\n",id); fflush(stdout); return 0; }
    if(fread(payload,1,(size_t)plen,stdin)!=(size_t)plen){ free(payload); return -1; }
    (void)fgetc(stdin); payload[plen]=0;
    snprintf(q->id,sizeof(q->id),"%s",id);
    q->max_tok=max_tok; q->temp=temp; q->top_p=top_p;
    q->payload=payload; q->plen=plen;
    return 2;
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

static void serve_one(Model *m, ServeReq *q){
    int *ids=NULL, np=0;
    encode_text(q->payload, &ids, &np);          /* payload is raw prompt text; qwen36 adds no BOS */
    int max_ctx = qwen36_max_ctx();
    if(np<1 || np+q->max_tok>max_ctx){
        printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",q->id,np,q->max_tok,max_ctx);
        fflush(stdout); free(ids); return;
    }
    printf("ACCEPT %s %d\n",q->id,np); fflush(stdout);
    m->max_t = np + q->max_tok;
    reset_recurrent(m); ensure_kv(m); m->kv_len = 0;
    /* Per-REQUEST state, not per-process: without this the server keeps the
     * first request's prefill flag and expert-collection set forever, so
     * COLIBRI_RESIDENT=1 collects on request #1 and never again, and the
     * router EMA carries one conversation's history into the next. */
    m->first_step = 1;
    if (m->seen) memset(m->seen, 0, (size_t)m->c.n_layers * m->c.n_experts);
    if (m->momentum_logits)
        memset(m->momentum_logits, 0,
               (size_t)m->c.n_layers * m->c.n_experts * sizeof(float));
    float *lo = step(m, ids, np, 0);
    int gen=0, limited=1;
    int eos_ids[4]; int n_eos=serve_eos_ids(eos_ids,4);
    double t0=now_s();
    unsigned char sbuf[16]; int sbn=0;
    for(int s=0;s<q->max_tok;s++){
        int tk = serve_sample(lo, m->c.vocab, q->temp, q->top_p);
        free(lo); lo=NULL;
        int is_eos=0; for(int e=0;e<n_eos;e++) if(tk==eos_ids[e]) is_eos=1;
        if(is_eos){ limited=0; break; }
        unsigned char tmp[256]; int tn=0; decode_id_to_bytes(tk, tmp, &tn);
        unsigned char chunk[256]; int cn=0; utf8_drain(sbuf,&sbn,tmp,tn,chunk,&cn);
        if(cn>0) serve_data(q->id,(char*)chunk,cn);
        gen++;
        /* The next logits are not needed after the final requested token.
         * serve_one() resets the recurrent/KV state for every request, so
         * stepping here would only run a full discarded decode pass. */
        if(s == q->max_tok - 1) break;
        lo = step(m, &tk, 1, np+s);
    }
    if(sbn>0) serve_data(q->id,(char*)sbuf,sbn);   /* flush trailing partial UTF-8 */
    free(lo); free(ids);
    double dt=now_s()-t0;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n",q->id,gen,
           dt>0?gen/dt:0.0,0.0,rss_gb(),np,limited);
    fflush(stdout);
}

static void serve_loop(Model *m){
    coli_serve_binary_mode();
    setvbuf(stdin,NULL,_IONBF,0);
    fputs("\x01\x01READY\x01\x01\n",stdout);
    printf("STAT 0 0.00 0.0 %.2f\n",rss_gb());
    fflush(stdout);
    for(;;){
        ServeReq q={0}; int r;
        do r=serve_read_req(&q); while(r==0);
        if(r<0) return;
        if(r==2){ serve_one(m,&q); free(q.payload); }
    }
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { coli_print_launcher_help("Qwen3.6"); return 1; }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    if (g_wide < 1) g_wide = 1; if (g_wide > 4) g_wide = 4;
    if (getenv("OPENAI")) g_openai = 1;                       /* OpenAI-compatible output */
    const char *mv = getenv("MODEL"); if (mv && *mv) g_model = mv;
    int hot_n = getenv("HOT") ? atoi(getenv("HOT")) : 0;
    int cap   = argc > 1 ? atoi(argv[1]) : 16;
    int bits  = argc > 2 ? atoi(argv[2]) : 4;
    /* cap < 1 leaves every layer cache empty, so expert_get finds no slot to
     * evict and waits for a publish that can never come. The old lru=0 fallback
     * turned that into a heap OOB instead; neither is a failure mode to ship. */
    if (cap < 1) { fprintf(stderr, "cache/layer must be >= 1 (got %d)\n", cap); return 1; }
    if (bits < 2 || bits > 8) { fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits); return 1; }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf   = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;

    fprintf(stderr, "== qwen36 Phase-2 engine | cache=%d/layer bits=%d pilot=%d wide=%d hot=%d smooth=%.2f conf=%.2f ==\n",
           cap, bits, g_pilot, g_wide, hot_n, smooth, conf);


    int is_ref = 0;
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

    /* load tokenizer early so text-prompt mode can encode before model_init */
    {
        const char *tokpath = getenv("TOK");
        if (tokpath && *tokpath) load_tokenizer(tokpath);
        else if (argc > 4 && argv[4] && *argv[4]) load_tokenizer(argv[4]);
        else { char tpb[2048]; snprintf(tpb,sizeof tpb,"%s/tokenizer.json",snap); load_tokenizer(tpb); }
    }

    if (serve_mode) {
        /* no argv prompt to load */
    } else if (is_ref) {
        FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        buf=malloc(n+1); if (fread(buf,1,n,f)!=(size_t)n) {} buf[n]=0; fclose(f);
        jval *ref = json_parse(buf, &arena);
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

    Model m; model_init(&m, snap, cap, bits);
    g_expert_gs = m.c.expert_gs;
    if (g_expert_gs) fprintf(stderr, "[qwen36] group-scaled experts: gs=%d\n", g_expert_gs);
    fprintf(stderr, "resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());
    /* quantize the large dense matrices to int8 (COLI_DENSE_I8=0 disables) */
    if (dense_i8_on()) {
        double tq = now_s();
        Cfg *qc = &m.c; int D2 = qc->hidden;
        int q_out = qc->q_heads * qc->q_head_dim, kv_out = qc->kv_heads * qc->k_head_dim;
        for (int i = 0; i < qc->n_layers; i++) {
            Layer *l = &m.L[i];
            qdw_register(l->q, D2, q_out); qdw_register(l->k, D2, kv_out);
            qdw_register(l->v, D2, kv_out); qdw_register(l->o, qc->o_in, D2);
            qdw_register(l->gate, D2, qc->n_experts);
            qdw_register(l->sh_g, D2, qc->shared_inter); qdw_register(l->sh_u, D2, qc->shared_inter);
            qdw_register(l->sh_d, qc->shared_inter, D2);
            qdw_register(l->dn_qkv, D2, qc->dn_conv_dim);
            qdw_register(l->dn_z, D2, qc->dn_vheads * qc->dn_vdim);
            qdw_register(l->dn_out, qc->dn_vheads * qc->dn_vdim, D2);
        }
        qdw_register(m.lm_head, D2, qc->vocab);
        /* Free the f32 originals -- the pointers only serve as lookup keys in
         * matmul_d from here on (never dereferenced again).
         * COLI_KEEP_F32=1 keeps them (debug). */
        double freed = 0;
        if (!getenv("COLI_KEEP_F32")) {
            for (int i = 0; i < g_qdw_n; i++) {
                freed += (double)g_qdw[i].I * g_qdw[i].O * sizeof(float);
                free((void*)g_qdw[i].w);
            }
        }
        fprintf(stderr, "[dense-i8] %d matrices quantized in %.1f s, %.1f GB f32 freed\n",
                g_qdw_n, now_s()-tq, freed/1073741824.0);
    }

    /* Optional CUDA VRAM expert tier (COLI_CUDA=1): hot experts live in
     * DEVICE_LOCAL memory across the configured GPUs, misses fall back to the
     * CPU int8 path. See qwen36_tier.h. */
    if (qt_init(m.c.n_layers, m.c.n_experts, m.c.hidden, m.c.inter, cap, m.c.topk, m.c.expert_gs)) {
        fprintf(stderr, "[gpu] MoE experts -> CUDA VRAM tier\n");
        atexit(qt_shutdown);
        /* Warmstart: fill the VRAM budget BEFORE the first token (heat order
         * when HEAT_FILE exists, natural order otherwise), loading all RAM
         * slots along the way. */
        const char *nws = getenv("QT_NO_WARMSTART");
        if (!(nws && *nws=='1')) {
            /* Plan the set (heat order), then load+stage IN PARALLEL. The
             * load path is thread-safe: expert_get locks the layer cache
             * (g_pilot_mx), st_read_raw uses pread; entries are unique. */
            double t0 = now_s();
            int cap_total = m.c.n_layers * m.c.n_experts;
            int *wpl = malloc((size_t)cap_total*sizeof(int));
            int *wpe = malloc((size_t)cap_total*sizeof(int));
            int wn = qt_plan_fill(wpl, wpe, cap_total);
            /* Load ALL experts into RAM, not just the planned (VRAM) set:
             * otherwise the first touch of a CPU-fallback expert triggers a
             * ~12 ms container read in the middle of decode (measured: 139
             * ms/token on a single-GPU run). Planned ones also go to VRAM. */
            uint8_t *planned = calloc((size_t)cap_total, 1);
            for (int i = 0; i < wn; i++) planned[wpl[i]*m.c.n_experts + wpe[i]] = 1;
            int keep8 = getenv("COLI_KEEP_INT8") != NULL;
            #pragma omp parallel for schedule(dynamic, 16)
            for (int gi = 0; gi < cap_total; gi++) {
                int l = gi / m.c.n_experts, eidw = gi % m.c.n_experts;
                Slot *e; expert_get(&m, l, eidw, &e);
                if (planned[gi] && e->g4) {
                    qt_note_planned(l, eidw, e->g4, e->u4, e->d4, e->gs, e->us, e->ds);
                    /* The staging copy is done; free the int8 copy RIGHT AWAY
                     * so it never shows up in peak RSS. On LFRU eviction
                     * slot_ensure_int8() rematerializes from g4 (no container
                     * access). */
                    if (!keep8 && e->g) { free(e->g); e->g = e->u = e->d = NULL; }
                }
            }
            qt_fill_wait();
            free(wpl); free(wpe); free(planned);
            fprintf(stderr, "[qtier] warmstart (parallel): all %d experts in RAM (int8 only for non-residents), %d in VRAM -- %.1f s\n",
                    cap_total, wn, now_s()-t0);
        }
    }

    /* coli serve mode: speak the gateway wire protocol instead of argv
     * generation. AFTER the tier init: serve sessions ride the VRAM experts
     * exactly like argv runs, and serve_loop never returns. */
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
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
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
    tm_report();
    qt_stats();
    fprintf(stderr, "\nPEAK RSS: %.2f GB\n", rss_gb());
    fprintf(stderr, "Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    fprintf(stderr, "Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
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
    free(layer->q); free(layer->k); free(layer->v); free(layer->o);
    free(layer->qn); free(layer->kn); free(layer->gate); free(layer->gate_bias);
    free(layer->sh_g); free(layer->sh_u); free(layer->sh_d); free(layer->sh_gate);
    free(layer->dn_qkv); free(layer->dn_z); free(layer->dn_b); free(layer->dn_a);
    free(layer->dn_conv); free(layer->dn_dtbias); free(layer->dn_alog);
    free(layer->dn_norm); free(layer->dn_out);
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
    free(model->DN_conv); free(model->DN_rec);
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
    int cap = 16;
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
    free(engine->model.DN_rec); free(engine->model.DN_conv);
    engine->model.DN_rec = NULL; engine->model.DN_conv = NULL;
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
