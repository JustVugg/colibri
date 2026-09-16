/* qwen3.c — dense Qwen3 (Qwen3ForCausalLM) engine.
 *
 * Standard dense transformer: pre-norm RMSNorm (plain weight, no +1),
 * GQA attention with per-head q/k RMSNorm and FULL RoPE (theta 1e6),
 * SwiGLU MLP, untied lm_head. No MoE, no linear-attention layers -- the
 * Qwen3.6 hybrid is c/qwen36.c, Flash-Next is c/qwen38.c.
 *
 * Container (see tools/convert_qwen3_dense.py): directory of safetensors
 * shards + config.json + tokenizer.json + qwen3_meta.json, read lazily via
 * st.h. Two on-disk formats per large matrix, picked by the converter:
 *   f16 passthrough  -> loaded to f32, plain matmul
 *   int4-gs64        -> U8 [O, ceil(I/2)] nibble+8 + F32 [O, ng] .qs scales,
 *                        run through quant.h matmul_i4_grouped as-is
 * CLI ABI matches the family: SNAP=<dir> ./qwen3 <cap> <bits> [ref.json];
 * cap/bits are accepted for launcher compatibility and ignored (the container
 * format decides everything). ref.json oracle mode exits non-zero on any
 * token mismatch.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#include <omp.h>

#include <sys/resource.h>
#include <unistd.h>

#include "serve_poll.h"       /* CANCEL mid-turn (#1332) */
#include "cli_args.h"
#include "st.h"
#include "json.h"   /* tokenizer.json parsing (reuse minimal parser) */
#include "quant.h"  /* matmul_i4_grouped (int4-gs64 containers) */
#include <limits.h>

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <dlfcn.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

/* Hard ceiling on context. Past max_position_embeddings (the per-model soft
 * ceiling, read from config/meta) the RoPE positions leave the trained range. */
#define QWEN3_HARD_MAX_CTX 262144

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
static const char *g_model = "qwen3-8b-colibri";
static double now_s(void);   /* forward decl; defined later near model code */

/* Output sink for server mode: when g_sock_out >= 0, SSE/JSON bytes are routed
 * to the live socket via g_sock_send instead of stdout. Lets the serve loop
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


/* ================= model ================= */

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab;
    int q_heads, kv_heads, head_dim;   /* head_dim = per-head q/k/v dim */
    int o_in;                          /* o_proj input dim = q_heads*head_dim */
    int inter;                         /* SwiGLU intermediate */
    float theta, eps;
    int gs;                            /* int4 group size on disk; 0 = no int4 */
    int max_pos;                       /* max_position_embeddings (soft ctx ceiling) */
} Cfg;

/* A large matmul weight, in either on-disk format the converter emits.
 * Exactly one of f / (q4,sc) is set; wm_mm dispatches to the right kernel. */
typedef struct {
    float *f;            /* f32 (f16/bf16 container, loaded via st_read_f32) */
    uint8_t *q4;         /* int4 packed [O, ceil(I/2)], nibble+8 */
    float *sc;           /* group scales [O, ceil(I/gs)] row-major */
    int I, O, gs;
} Wm;

/* ---------- per-layer weights ---------- */
typedef struct {
    float *in_ln, *post_ln;            /* RMSNorm weights [hidden] */
    float *qn, *kn;                    /* per-head q/k RMSNorm [head_dim] */
    Wm q, k, v, o;                     /* attention projections */
    Wm gate_w, up_w, down_w;           /* SwiGLU MLP */
} Layer;

typedef struct {
    Cfg c;
    shards S;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    float **K, **V; int kv_len, max_t, kv_cap;
    float *attn_sc;                    /* [attn_sc_thr * max_t] score rows */
    int attn_sc_thr;
    double load_s;
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }

static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }

static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T for f32 weights comes from quant.h (matmul), which
 * already has the AVX2/NEON paths; int4 containers use matmul_i4_grouped. */

/* Wm dispatch: int4 containers run quant.h's matmul_i4_grouped as-is. */
static void wm_mm(float *y, const float *x, const Wm *w, int S) {
    if (w->q4) matmul_i4_grouped(y, x, w->q4, w->sc, S, w->I, w->O, w->gs);
    else matmul(y, x, w->f, S, w->I, w->O);
}

/* rmsnorm over a row of length D (in-place capable: out may == x).
 * Qwen3RMSNorm: out = (x * rsqrt(mean(x^2)+eps)) * weight -- PLAIN weight,
 * unlike Qwen3.6's (1.0+weight) variant. */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- loading ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}

/* Every dimension the forward pass indexes with, checked once. config.json and
 * qwen3_meta.json ship inside the container, so a mismatched or hostile pair is
 * a supply-chain input, not a programmer error -- same discipline as qwen36.c
 * (one guarded expression per dimension, exit with a clear message). */
#define CFG_NEED(cond, ...) do { if (!(cond)) {         fprintf(stderr, "[cfg] "); fprintf(stderr, __VA_ARGS__);         fprintf(stderr, " -- refusing\n"); exit(1); } } while (0)

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
    c->inter     = (int)req_num(r,"intermediate_size");
    c->q_heads   = (int)req_num(r,"num_attention_heads");
    c->kv_heads  = (int)req_num(r,"num_key_value_heads");
    jval *hd = json_get(r,"head_dim");
    c->head_dim  = hd ? (int)hd->num : c->hidden / (c->q_heads > 0 ? c->q_heads : 1);
    /* transformers <5 writes rope_theta at the top level, >=5 nests it under
     * rope_parameters. Defaulting the base would produce plausible-looking but
     * wrong tokens, so take whichever the config has and leave it at 0 (caught
     * by validate_cfg) when it has neither and no meta supplies it. */
    jval *th = json_get(r,"rope_theta");
    if (!th || th->t != J_NUM) {
        jval *rp = json_get(r,"rope_parameters");
        th = (rp && rp->t == J_OBJ) ? json_get(rp,"rope_theta") : NULL;
    }
    c->theta = (th && th->t == J_NUM) ? (float)th->num : 0.f;
    jval *mp = json_get(r,"max_position_embeddings");
    c->max_pos = mp ? (int)mp->num : 40960;
    /* defaults, overridden by qwen3_meta.json */
    c->o_in = c->q_heads * c->head_dim;
    c->gs = 0;
    free(buf); free(arena);
}

static void load_meta(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/qwen3_meta.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) return;   /* meta optional: config.json has the dims */
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || n>(256L<<20)){ fclose(f); return; }
    char *buf = malloc((size_t)n+1); if(!buf){fclose(f);return;}
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ free(buf); fclose(f); return; } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    if (r && r->t == J_OBJ) {
        jval *v;
        #define G(name,field) if((v=json_get(r,name))&&v->t==J_NUM) c->field=(int)v->num
        G("hidden", hidden); G("n_layers", n_layers); G("vocab", vocab);
        G("q_heads", q_heads); G("kv_heads", kv_heads); G("head_dim", head_dim);
        G("o_in", o_in); G("inter", inter); G("gs", gs);
        G("max_position_embeddings", max_pos);
        #undef G
        if((v=json_get(r,"rope_theta"))&&v->t==J_NUM) c->theta=(float)v->num;
        if((v=json_get(r,"rms_eps"))&&v->t==J_NUM) c->eps=(float)v->num;
    }
    free(buf); free(arena);
}

static void validate_cfg(const Cfg *c) {
    CFG_NEED(c->n_layers > 0 && c->n_layers <= 512,
             "n_layers %d out of range 1..512", c->n_layers);
    CFG_NEED(c->hidden > 0 && c->hidden <= 65536, "hidden %d out of range", c->hidden);
    CFG_NEED(c->vocab > 0, "vocab %d must be positive", c->vocab);
    CFG_NEED(c->inter > 0, "intermediate_size %d must be positive", c->inter);
    CFG_NEED(c->q_heads > 0 && c->kv_heads > 0 && c->head_dim > 0,
             "attention dims q_heads=%d kv_heads=%d head_dim=%d must be positive",
             c->q_heads, c->kv_heads, c->head_dim);
    CFG_NEED(c->q_heads % c->kv_heads == 0,
             "q_heads %d is not a multiple of kv_heads %d (GQA grouping)",
             c->q_heads, c->kv_heads);
    CFG_NEED(c->head_dim % 2 == 0, "head_dim %d must be even (RoPE pairs)", c->head_dim);
    CFG_NEED(c->o_in > 0, "o_in %d must be positive", c->o_in);
    CFG_NEED(c->theta > 0, "rope_theta missing from both config.json and qwen3_meta.json");
    CFG_NEED(c->max_pos > 0 && c->max_pos <= QWEN3_HARD_MAX_CTX,
             "max_position_embeddings %d out of range 1..%d",
             c->max_pos, QWEN3_HARD_MAX_CTX);
    if (c->gs) CFG_NEED(c->gs > 0, "gs %d must be 0 or positive", c->gs);
}

/* `want` is the element count the forward pass will index with; refuse a
 * container whose tensor disagrees with the config (qwen36.c discipline). */
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

/* Load a matmul weight [O, I] in either container format.
 * int4-gs64 is detected by on-disk SHAPE (dtype U8/I8, nbytes == O*ceil(I/2),
 * plus a matching .qs scale tensor), not by meta flags -- the same rule
 * qwen36.c applies to experts. */
static void load_wm(Model *m, Wm *w, const char *name, int O, int I) {
    char qs_name[512];
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(1); }
    w->I = I; w->O = O; w->gs = 0; w->f = NULL; w->q4 = NULL; w->sc = NULL;
    int64_t q4_bytes = (int64_t)O * ((I + 1) / 2);
    snprintf(qs_name, sizeof(qs_name), "%s.qs", name);
    if (t->dtype == 3 && t->nbytes == q4_bytes && st_has(&m->S, qs_name)) {
        int gs = m->c.gs > 0 ? m->c.gs : 64;
        int64_t ng = (int64_t)O * ((I + gs - 1) / gs);
        if (st_numel(&m->S, qs_name) != ng) {
            fprintf(stderr, "%s: .qs has %lld elements, int4-gs%d implies %lld -- refusing\n",
                    qs_name, (long long)st_numel(&m->S, qs_name), gs, (long long)ng);
            exit(1);
        }
        w->q4 = malloc((size_t)q4_bytes);
        if (!w->q4) { fprintf(stderr, "OOM %s\n", name); exit(1); }
        st_read_raw(&m->S, name, w->q4, 0);
        w->sc = falloc(ng);
        st_read_f32(&m->S, qs_name, w->sc, 0);
        w->gs = gs;
        return;
    }
    if (t->dtype != 0 && t->dtype != 1 && t->dtype != 2) {
        fprintf(stderr, "%s: dtype %d is neither float nor a matching int4 layout -- refusing\n",
                name, t->dtype); exit(1);
    }
    w->f = load_t_n(m, name, (int64_t)O * I);
}

static void model_init(Model *m, const char *snap) {
    memset(m, 0, sizeof(*m));
    load_cfg(&m->c, snap);
    load_meta(&m->c, snap);
    /* Q3_MAXT is how `coli --ctx` reaches this engine (family_registry's
     * context_env). It only ever LOWERS the ceiling -- raising it past
     * max_position_embeddings would push RoPE outside the trained range --
     * and it runs after load_meta so the container still has the last word
     * on what the model itself supports. */
    const char *maxt = getenv("Q3_MAXT");
    if (maxt && *maxt) {
        int v = atoi(maxt);
        if (v < 1) { fprintf(stderr, "Q3_MAXT=%s must be a positive token count\n", maxt); exit(1); }
        if (v < m->c.max_pos) m->c.max_pos = v;
    }
    validate_cfg(&m->c);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    m->embed      = load_t_n(m, "model.embed_tokens.weight", (int64_t)c->vocab * c->hidden);
    m->lm_head    = st_has(&m->S, "lm_head.weight")
                  ? load_t_n(m, "lm_head.weight", (int64_t)c->vocab * c->hidden)
                  : m->embed;    /* tied embeddings */
    m->final_norm = load_t_n(m, "model.norm.weight", c->hidden);
    m->L = calloc((size_t)c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t_n(m,nm,(want))
        LD(in_ln,  "input_layernorm.weight", c->hidden);
        LD(post_ln,"post_attention_layernorm.weight", c->hidden);
        LD(qn, "self_attn.q_norm.weight", c->head_dim);
        LD(kn, "self_attn.k_norm.weight", c->head_dim);
        #undef LD
        #define LW(field, suffix, O, I) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); load_wm(m, &l->field, nm, (O), (I))
        LW(q,   "self_attn.q_proj.weight", c->q_heads * c->head_dim, c->hidden);
        LW(k,   "self_attn.k_proj.weight", c->kv_heads * c->head_dim, c->hidden);
        LW(v,   "self_attn.v_proj.weight", c->kv_heads * c->head_dim, c->hidden);
        LW(o,   "self_attn.o_proj.weight", c->hidden, c->o_in);
        LW(gate_w, "mlp.gate_proj.weight", c->inter, c->hidden);
        LW(up_w,   "mlp.up_proj.weight",   c->inter, c->hidden);
        LW(down_w, "mlp.down_proj.weight", c->hidden, c->inter);
        #undef LW
    }
    m->load_s = now_s() - t0;
}

/* ---------- forward ---------- */

/* RoPE over the first rope_dim dims of each head (== the whole head for dense
 * Qwen3: full rotary, unlike Qwen3.6's partial 0.25). */
static void rope_head(float *x, int pos, int rope_dim, float theta) {
    int h = rope_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f * j / rope_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* GQA attention matching HF Qwen3Attention: per-head q/k RMSNorm, full RoPE,
 * scale = head_dim^-0.5, causal, repeat_kv. */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    int q_out = H * hd, kv_out = KV * hd, q_per_kv = H / KV;
    float *q = falloc((int64_t)S*q_out);
    float *k = falloc((int64_t)S*kv_out);
    float *vv= falloc((int64_t)S*kv_out);
    wm_mm(q,  x, &l->q, S);
    wm_mm(k,  x, &l->k, S);
    wm_mm(vv, x, &l->v, S);
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            float *qh = q + ((int64_t)s*H + hh)*hd;
            if (l->qn) rmsnorm_row(qh, qh, l->qn, hd, c->eps);
            rope_head(qh, pos_base + s, hd, c->theta);
        }
        for (int kvh = 0; kvh < KV; kvh++) {
            float *kh = k + ((int64_t)s*KV + kvh)*hd;
            if (l->kn) rmsnorm_row(kh, kh, l->kn, hd, c->eps);
            rope_head(kh, pos_base + s, hd, c->theta);
        }
    }
    for (int s = 0; s < S; s++) for (int kvh = 0; kvh < KV; kvh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)kvh*m->max_t + t)*hd, k + ((int64_t)s*KV + kvh)*hd, hd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)kvh*m->max_t + t)*hd, vv + ((int64_t)s*KV + kvh)*hd, hd*sizeof(float));
    }
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*H*hd);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int kvh = hh / q_per_kv;
            int qpos = pos_base + s;
            const float *qv = q + ((int64_t)s*H + hh)*hd;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = m->attn_sc + (int64_t)tid * m->kv_cap;
            for (int t = 0; t <= qpos; t++) {
                const float *kv = m->K[layer] + ((int64_t)kvh*m->max_t + t)*hd;
                float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kv[dd];
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + ((int64_t)s*H + hh)*hd;
            for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = m->V[layer] + ((int64_t)kvh*m->max_t + t)*hd;
                float a = sc[t]; for (int dd = 0; dd < hd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    wm_mm(out, ctx, &l->o, S);
    free(q); free(k); free(vv); free(ctx);
}

/* SwiGLU MLP: down(silu(gate(x)) * up(x)) */
static void mlp(Model *m, Layer *l, float *x, int S, float *out) {
    Cfg *c = &m->c; int I = c->inter;
    float *g = falloc((int64_t)S*I);
    float *u = falloc((int64_t)S*I);
    wm_mm(g, x, &l->gate_w, S);
    wm_mm(u, x, &l->up_w, S);
    for (int64_t j = 0; j < (int64_t)S*I; j++) {
        float sv = g[j];
        g[j] = (sv / (1.f + expf(-sv))) * u[j];
    }
    wm_mm(out, g, &l->down_w, S);
    free(g); free(u);
}

static void layers_forward(Model *m, float *x, int S, int pos_base) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        mlp(m, l, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    free(nrm); free(tmp);
}

static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) {
        /* the embed gather indexes by token id: an id outside the vocab reads
         * off the end (qwen36.c discipline: refuse, all three input sources) */
        if (ids[s] < 0 || ids[s] >= c->vocab) {
            fprintf(stderr, "token id %d out of range 0..%d -- refusing\n",
                    ids[s], c->vocab - 1);
            exit(1);
        }
        memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    }
    layers_forward(m, x, S, pos_base);
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(last);
    return logit;
}

/* Allocate (once) or reuse the KV cache across requests; grows only when a
 * longer context is needed, never shrinks (qwen36.c ensure_kv). */
static void ensure_kv(Model *m){
    Cfg *c = &m->c;
    if (m->kv_cap >= m->max_t && m->K) return;
    if (m->K){
        for (int i = 0; i < c->n_layers; i++){ if (m->K[i]) free(m->K[i]); if (m->V[i]) free(m->V[i]); }
        free(m->K); free(m->V); m->K = NULL; m->V = NULL;
    }
    m->K = calloc((size_t)c->n_layers, sizeof(float*)); m->V = calloc((size_t)c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++){
        m->K[i] = falloc((int64_t)c->kv_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->kv_heads * m->max_t * c->head_dim);
    }
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
    if (np + n_new > c->max_pos) {
        fprintf(stderr, "[ctx] prompt %d + %d new exceeds the %d-token ceiling\n",
                np, n_new, c->max_pos);
        exit(1);
    }
    m->max_t = np + n_new;
    ensure_kv(m);
    m->kv_len = 0;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        if (s == 0 && g_ttft < 0) g_ttft = now_s() - g_gen_t0;
        if (g_stream) { stream_token(best); fflush(stdout); }
        free(logit); out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1);
    }
}

static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    if (nfull > c->max_pos) {
        fprintf(stderr, "[ctx] %d tokens exceed the %d-token ceiling\n", nfull, c->max_pos);
        exit(1);
    }
    m->max_t = nfull;
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

#ifndef QWEN3_NO_MAIN

/* ===================== coli serve mode (SERVE=1) ===================== *
 * Same gateway wire protocol as qwen36.c / kimi_k3.c / inkling.c:
 *   engine:  \x01\x01READY\x01\x01\n
 *            STAT 0 0.00 0.0 <rss>\n
 *   gateway: SUBMIT <id> <slot> <plen> <max_tok> <temp> <top_p>\n <payload bytes>\n
 *   engine:  ACCEPT <id> <np>\n
 *            DATA <id> <n>\n <bytes>\n     (repeated per decoded chunk)
 *            PROF ...                       (optional, dashboard phases)
 *            DONE <id> STAT <gen> <tps> <hit%> <rss> <np> <limited>\n
 *   gateway: CANCEL <id>  (abort current turn, #1332)
 * No EMAP/HITS lines: those describe an expert grid and this engine is dense. */

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

/* temperature + top-p sampler (same as qwen36.c) */
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
 * both ids from the tokenizer's added_tokens (qwen36.c: the hardcoded 151645
 * silently never matched on larger vocabs). Q3_EOS overrides. */
static int serve_eos_ids(int *ids, int cap){
    int n=0;
    if(getenv("Q3_EOS")){ ids[n++]=atoi(getenv("Q3_EOS")); return n; }
    for(int k=0;k<g_nspecial && n<cap;k++)
        if(!strcmp(g_sp_str[k],"<|im_end|>")||!strcmp(g_sp_str[k],"<|endoftext|>"))
            ids[n++]=g_sp_id[k];
    if(!n) ids[n++]=151645;   /* tokenizer without added_tokens: Qwen3 default */
    return n;
}

/* Non-blocking CANCEL check (#1332): never reads when stdin is not ready. */
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

static void serve_one(Model *m, ServeReq *q){
    int *ids=NULL, np=0;
    encode_text(q->payload, &ids, &np);          /* payload is raw prompt text; no BOS */
    if(np<1 || np+q->max_tok > m->c.max_pos){
        printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",
               q->id,np,q->max_tok,m->c.max_pos);
        fflush(stdout); free(ids); return;
    }
    printf("ACCEPT %s %d\n",q->id,np); fflush(stdout);
    m->max_t = np + q->max_tok;
    ensure_kv(m); m->kv_len = 0;
    float *lo = step(m, ids, np, 0);
    int gen=0, limited=1, forwards=1;   /* the prefill is the first forward */
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
        if(serve_cancel_pending(q->id)){
            free(lo); lo=NULL;
            if(sbn>0) serve_data(q->id,(char*)sbuf,sbn);
            printf("ERROR %s CANCELLED\n",q->id); fflush(stdout);
            free(ids);
            return;
        }
        /* next logits are not needed after the final requested token */
        if(s == q->max_tok - 1) break;
        lo = step(m, &tk, 1, np+s); forwards++;
    }
    if(sbn>0) serve_data(q->id,(char*)sbuf,sbn);   /* flush trailing partial UTF-8 */
    free(lo); free(ids);
    double dt=now_s()-t0;
    /* PROF (dashboard phases): dense engine -- expert phases are zero by
     * construction, attention/head not yet instrumented. */
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", dt, np, gen,
           0.0, 0.0, 0.0, 0.0, 0.0, (unsigned long long)forwards);
    fflush(stdout);
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
    if (!snap) { coli_print_launcher_help("Qwen3"); return 1; }
    if (getenv("OPENAI")) g_openai = 1;
    const char *mv = getenv("MODEL"); if (mv && *mv) g_model = mv;
    /* cap/bits exist for launcher ABI compatibility only: a dense model has
     * no expert cache and the container format fixes the precision. */
    int cap   = argc > 1 ? coli_arg_int(argv[1], "cache/layer") : 0;
    int bits  = argc > 2 ? coli_arg_int(argv[2], "bits") : 0;
    const char *refpath = argc > 3 ? argv[3] : "ref.json";
    (void)cap; (void)bits;

    fprintf(stderr, "== qwen3 dense engine | cap/bits args accepted and ignored (container decides) ==\n");

    int is_ref = 0;
    int rplen = (int)strlen(refpath);
    if (rplen>=5 && strcmp(refpath+rplen-5, ".json")==0) is_ref = 1;

    int *prompt=NULL, *full=NULL, *out=NULL;
    int np=0, nfull=0, n_new=0;
    char *buf=NULL, *arena=NULL;
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
    }

    static Model m; model_init(&m, snap);
    fprintf(stderr, "resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.load_s, rss_gb());

    if (serve_mode) {
        if (!g_tok) { fprintf(stderr, "[serve] tokenizer.json required (put in SNAP or set TOK)\n"); return 1; }
        serve_loop(&m);
        return 0;
    }

    if (is_ref && getenv("PPL") && atoi(getenv("PPL")) == 1) {
        double nll; double t = now_s();
        int scored = tf_nll(&m, full, nfull, np, &nll);
        double dt = now_s() - t;
        printf("TF-NLL: %.4f nats/token over %d tokens | ppl = %.2f\n", nll, scored, exp(nll));
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
        free(buf); free(arena); return 0;
    }

    out = malloc((np + n_new) * sizeof(int));
    g_ttft = -1; g_gen_t0 = now_s();
    if (g_openai){
        g_oa_created = (long)time(NULL);
        snprintf(g_oa_id, sizeof g_oa_id, "chatcmpl-%ld%04d", g_oa_created, (int)(now_s()*1000) % 10000);
    }
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
    if (g_ttft >= 0) fprintf(stderr, "TTFT: %.2f s (time to first token)\n", g_ttft);
    fprintf(stderr, "\nPEAK RSS: %.2f GB\n", rss_gb());
    fprintf(stderr, "Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    free(buf); free(arena);
    /* Oracle mode is a gate, not a report (qwen36.c/inkling.c discipline). */
    if (is_ref) return ref_match == n_new ? 0 : 1;
    return 0;
}
#endif /* QWEN3_NO_MAIN */
