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
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "st.h"
#include "json.h"   /* tokenizer.json parsing (reuse minimal parser) */
#include "vulkan_gemv.h"   /* optional transparent Vulkan compute backend for MoE experts */

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
typedef struct { Slot *slots; int n, cap; } LCache;

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
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
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
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
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

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    load_meta(&m->c, snap);
    if (m->c.rotary_dim > m->c.head_dim || m->c.rotary_dim % 2 != 0) {
        fprintf(stderr, "rotary_dim %d invalid for head_dim %d\n", m->c.rotary_dim, m->c.head_dim); exit(1);
    }
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    m->embed      = load_t(m, "model.embed_tokens.weight");
    m->lm_head    = load_t(m, "lm_head.weight");
    m->final_norm = load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    /* Phase 2: the converter stores EVERY layer (Gated-Attention + Gated DeltaNet)
     * under its OWN original index model.layers.{i}. So active_of is the identity
     * map; experts and dense weights are read from model.layers.{i} for all i. */
    m->active_of = malloc((size_t)c->n_layers * sizeof(int));
    for (int i = 0; i < c->n_layers; i++) m->active_of[i] = i;
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        int ai = m->active_of[i];        /* == i for Phase 2 */
        Layer *l = &m->L[i];
        /* input/post layernorms + MoE exist for every layer */
        #define LD(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,ai); l->field = load_t(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LD(gate, "mlp.gate.weight");
        #undef LD
        /* q/k norms are per-head [head_dim]; only on attention layers, load if present */
        if (c->has_qk_norm) {
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_norm.weight", ai);
            l->qn = st_has(&m->S, nm) ? load_t(m, nm) : NULL;
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_norm.weight", ai);
            l->kn = st_has(&m->S, nm) ? load_t(m, nm) : NULL;
        } else { l->qn = NULL; l->kn = NULL; }
        /* router correction bias (optional) */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.e_score_correction_bias", ai);
        if (st_has(&m->S, nm)) { l->gate_bias = falloc(c->n_experts); st_read_f32(&m->S, nm, l->gate_bias, 0); }
        else l->gate_bias = NULL;
        /* shared expert (dense f32) */
        #define LD2(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert." suffix,ai); l->field = load_t(m,nm)
        LD2(sh_g, "gate_proj.weight"); LD2(sh_u, "up_proj.weight"); LD2(sh_d, "down_proj.weight");
        #undef LD2
        /* shared_expert_gate: Linear(hidden -> 1), sigmoid-gated shared expert */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert_gate.weight", ai);
        l->sh_gate = st_has(&m->S, nm) ? load_t(m, nm) : NULL;
        if (c->is_attn[i]) {
            /* Gated Attention (full_attention) layer */
            #define LD3(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d.self_attn." suffix,ai); l->field = load_t(m,nm)
            LD3(q, "q_proj.weight"); LD3(k, "k_proj.weight");
            LD3(v, "v_proj.weight"); LD3(o, "o_proj.weight");
            #undef LD3
            l->dn_qkv=l->dn_z=l->dn_b=l->dn_a=l->dn_conv=NULL;
            l->dn_dtbias=l->dn_alog=l->dn_norm=l->dn_out=NULL;
        } else {
            /* Gated DeltaNet (linear_attention) layer */
            l->q=l->k=l->v=l->o=NULL;
            #define LD4(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn." suffix,ai); l->field = load_t(m,nm)
            LD4(dn_qkv, "in_proj_qkv.weight"); LD4(dn_z, "in_proj_z.weight");
            LD4(dn_b,   "in_proj_b.weight");    LD4(dn_a, "in_proj_a.weight");
            LD4(dn_conv,"conv1d.weight");       LD4(dn_dtbias, "dt_bias");
            LD4(dn_alog,"A_log");               LD4(dn_norm, "norm.weight");
            LD4(dn_out, "out_proj.weight");
            #undef LD4
        }
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
    }
    /* per-layer DeltaNet recurrent + conv state (only for linear_attention layers) */
    m->DN_rec = calloc(c->n_layers, sizeof(float*));
    m->DN_conv = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
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
    float *s_block = falloc(c->inter + c->inter + c->hidden);
    s->gs = s_block;
    s->us = s_block + c->inter;
    s->ds = s_block + c->inter + c->inter;
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
    int64_t want_s = (int64_t)cc->inter + cc->inter + cc->hidden;
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
        /* Keep the packed bytes for the int4 GPU shader. The unpacked int8 copy
         * above is retained for the CPU fallback / int8-GPU path. Only allocate
         * when a GPU int4 backend is actually active, to avoid doubling expert
         * memory on CPU-only or int8-GPU runs. Free any previous occupant first
         * (LRU slot reuse). */
        free(s->g4); free(s->u4); free(s->d4); s->g4 = s->u4 = s->d4 = NULL;
        /* Always keep the packed int4 bytes for the int4 GPU shader, regardless
         * of whether the Vulkan backend is currently initialised.  Gating this
         * on vg_use_int4() created a load/initialisation-order race: experts can
         * be loaded (and the GPU slot cache stamped "valid") before vg_init() has
         * run, leaving g4 empty and the GPU reading all-zero weights while the
         * CPU int8 fallback (s->g) stayed correct.  The unpacked int8 copy in s->g
         * remains the CPU fallback. */
        {
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
static int container_is_int4(Model *m) {
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    char nm[256];
    snprintf(nm, sizeof(nm), "model.layers.0.mlp.experts.0.merged_weight");
    st_tensor *tw = st_find(&m->S, nm);
    if (!tw) return 0;
    return (tw->nbytes == want_w / 2) ? 1 : 0;
}

static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_pilot_mx);
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i];
        pthread_mutex_unlock(&g_pilot_mx); return;
    }
    m->miss++;
    Cfg *c = &m->c; Slot *s;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0) {
            for (int i = 0; i < lc->n; i++) { if (lc->slots[i].eid < 0) continue; if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i; }
        }
        if (lru < 0) lru = 0;
        s = &lc->slots[lru]; s->pinned = 0;
    }
    s->eid = -1; s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);
    load_expert_merged(m, layer, eid, s);
    pthread_mutex_lock(&g_pilot_mx);
    s->eid = eid; s->pinned = m->is_pinned[layer * c->n_experts + eid]; s->used = ++m->clock;
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
            LCache *lc = &m->cache[l];
            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) { lc->slots[i].pinned = 1; found = 1; break; }
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

/* COLIBRI_RESIDENT: after prefill, pin every expert this prompt actually routed to.
 * CPU LRU then never evicts them, so vg_expert_ensure_int4 keeps hitting its valid
 * slot in g_w -> the expert stays resident in GPU memory for the whole decode. */
/* COLIBRI_RESIDENT: after prefill (mode 1) and continuously through decode (mode 2),
 * pin every expert this prompt routed to. CPU LRU then never evicts them, so
 * vg_expert_ensure_int4 keeps hitting its valid slot in g_w -> expert stays resident
 * in GPU memory. `quiet` suppresses the log line when nothing new was pinned
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
    matmul(q, x, l->q, S, D, q_out);
    matmul(k, x, l->k, S, D, kv_out);
    matmul(vv, x, l->v, S, D, kv_out);
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
            float sc[8192];
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
    matmul(out, ag, l->o, S, H*hd, D);
    free(q); free(k); free(vv); free(query); free(gate); free(ctx); free(ag);
}

/* MoE: grouped top-k routing (+ optional router bias) + shared expert.
 * Mirrors HF Qwen3 MoE: softmax(gate), optional group-limited top-k, normalized
 * weights, sum routed experts, then add the un-gated shared expert. */
enum { GPU_BACKEND_CPU = 0, GPU_BACKEND_VULKAN = 1 };
static int g_gpu_backend = GPU_BACKEND_CPU;   /* chosen backend (set by gpu_probe) */
static int g_gpu_count   = 0;                 /* # compute-capable devices */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    matmul(logits, x, l->gate, S, D, E);
    if (c->has_bias && l->gate_bias) {
        for (int s = 0; s < S; s++) { float *pr = logits + (int64_t)s*E; for (int e = 0; e < E; e++) pr[e] += l->gate_bias[e]; }
    }
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    float *sh = falloc(I), *shu = falloc(I), *shd = falloc(D);  /* shared expert scratch */
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
        if (g_gpu_backend == GPU_BACKEND_VULKAN && vg_ready()) {
            /* GPU path: route on CPU, run all K experts' GEMVs in 2 dispatches.
             * The shared expert (below) stays on CPU. */
            int handles[256];
            for (int kk = 0; kk < K; kk++) {
                Slot *e; expert_get(m, layer, idx[kk], &e);
                int li = (int)(e - m->cache[layer].slots);   /* mirror CPU LRU slot index */
                if (e->g4) {
                    /* int4 packed weights -> robust Shader-only GPU path (no int8/dot-product) */
                    vg_expert_ensure_int4(layer, li, idx[kk], e->g4, e->u4, e->d4, e->gs, e->us, e->ds);
                } else {
                    vg_expert_ensure(layer, li, idx[kk], e->g, e->u, e->d, e->gs, e->us, e->ds);
                }
                handles[kk] = layer * m->cache[layer].cap + li;
            }
            vg_moe_run(layer, K, handles, val, xs, out + (int64_t)s*D);
            if (getenv("MOE_DBG") && layer == 0 && s == 0) {
                for (int dk = 0; dk < K; dk++) {
                    Slot *de; expert_get(m, layer, idx[dk], &de);
                    fprintf(stderr,"[MOE_DBG] kk%d eid=%d g4=%s g0=%d gs0=%.4f u4=%s u0=%d us0=%.4f d4=%s d0=%d ds0=%.4f\n",
                        dk, idx[dk],
                        de->g4?"Y":"N", (int)de->g[0], de->gs[0],
                        de->u4?"Y":"N", (int)de->u[0], de->us[0],
                        de->d4?"Y":"N", (int)de->d[0], de->ds[0]);
                }
                float *oref = falloc((int64_t)D);
                float *gg = falloc((int64_t)I), *uu = falloc((int64_t)I), *hh2 = falloc((int64_t)D);
                for (int kk = 0; kk < K; kk++) {
                    Slot *e; expert_get(m, layer, idx[kk], &e);
                    matmul_q(gg, xs, e->g, e->gs, D, I);
                    matmul_q(uu, xs, e->u, e->us, D, I);
                    for (int i = 0; i < I; i++) { float gv = gg[i]; gg[i] = (gv / (1.f + expf(-gv))) * uu[i]; }
                    matmul_q(hh2, gg, e->d, e->ds, I, D);
                    for (int d = 0; d < D; d++) oref[d] += val[kk] * hh2[d];
                }
                double dot=0,n1=0,n2=0; const float *oa = out + (int64_t)s*D;
                for (int d=0;d<D;d++){ dot+=oref[d]*oa[d]; n1+=oref[d]*oref[d]; n2+=oa[d]*oa[d]; }
                double cos = (n1>0&&n2>0)? dot/sqrt(n1*n2) : 0;
                double gnorm=0, cnorm=0; for(int d=0;d<D;d++){gnorm+=oa[d]*oa[d]; cnorm+=oref[d]*oref[d];}
                fprintf(stderr,"[MOE_DBG] L0 T0 GPU vs CPU MoE cos=%.5f |GPU|=%g |CPU|=%g\n", cos, sqrt(gnorm), sqrt(cnorm));
                free(oref); free(gg); free(uu); free(hh2);
            }
        } else {
            for (int kk = 0; kk < K; kk++) {
                Slot *e; expert_get(m, layer, idx[kk], &e);
                matmul_q(g, xs, e->g, e->gs, D, I);
                matmul_q(u, xs, e->u, e->us, D, I);
                for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                matmul_q(hh, g, e->d, e->ds, I, D);
                float w = val[kk];
                float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += w * hh[d];
            }
        }
        /* shared expert (SwiGLU), sigmoid-gated by shared_expert_gate */
        int Ish = c->shared_inter;
        matmul(sh, xs, l->sh_g, 1, D, Ish);
        matmul(shu, xs, l->sh_u, 1, D, Ish);
        for (int i = 0; i < Ish; i++) { float sv = sh[i]; sh[i] = (sv / (1.f + expf(-sv))) * shu[i]; }
        matmul(shd, sh, l->sh_d, 1, Ish, D);
        float sgate = 1.f;
        if (l->sh_gate) {
            float sg = 0.f; const float *wg = l->sh_gate;
            for (int i = 0; i < D; i++) sg += xs[i] * wg[i];
            sgate = 1.f / (1.f + expf(-sg));
        }
        float *os = out + (int64_t)s*D;
        for (int d = 0; d < D; d++) os[d] += sgate * shd[d];
    }
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
        /* projections (single-token matmuls) */
        matmul(qkv, xs, l->dn_qkv, 1, H, conv_dim);
        matmul(z,   xs, l->dn_z,   1, H, value_dim);
        matmul(b,   xs, l->dn_b,   1, H, vh);
        matmul(a,   xs, l->dn_a,   1, H, vh);
        for (int h = 0; h < vh; h++) {
            beta[h] = 1.f / (1.f + expf(-b[h]));
            gg[h] = -expf(l->dn_alog[h]) * softplus_f(a[h] + l->dn_dtbias[h]);
        }
        /* causal depthwise conv1d (groups=conv_dim, kernel=convk) with carried ring */
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
        /* recurrent gated delta rule over the value heads */
        for (int h = 0; h < vh; h++) {
            float *Sh = rec + (int64_t)h * kdim * vdim;
            float egh = expf(gg[h]);
            for (int t = 0; t < kdim * vdim; t++) Sh[t] *= egh;
            const float *kd = k + (int64_t)h * kdim;
            const float *vd = v_in + (int64_t)h * vdim;
            /* kv = kd @ Sh  (length vdim) */
            for (int vv = 0; vv < vdim; vv++) {
                float acc = 0.f;
                for (int kk = 0; kk < kdim; kk++) acc += kd[kk] * Sh[(int64_t)kk * vdim + vv];
                kv[vv] = acc;
            }
            /* delta = (v - kv) * beta */
            for (int vv = 0; vv < vdim; vv++) delta[vv] = (vd[vv] - kv[vv]) * beta[h];
            /* Sh += outer(kd, delta) */
            for (int kk = 0; kk < kdim; kk++) {
                float kkd = kd[kk];
                for (int vv = 0; vv < vdim; vv++) Sh[(int64_t)kk * vdim + vv] += kkd * delta[vv];
            }
            /* out = qd @ Sh */
            const float *qd = q + (int64_t)h * kdim;
            for (int vv = 0; vv < vdim; vv++) {
                float acc = 0.f;
                for (int kk = 0; kk < kdim; kk++) acc += qd[kk] * Sh[(int64_t)kk * vdim + vv];
                outv[(int64_t)h * vdim + vv] = acc;
            }
        }
        /* per-head Gated RMSNorm (plain weight, r=1/sqrt(mean+eps)) then silu(z) gate, then out_proj.
         * HF Qwen3_5MoeRMSNormGated: out = (o*r)*weight * silu(z) = (o*r)*weight * z/(1+e^-z).
         * NB: it is silu (z in numerator), NOT sigmoid. */
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
        matmul(out + (int64_t)s * H, outr, l->dn_out, 1, value_dim, H);
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
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        if (c->is_attn[i]) {
            attention(m, l, i, nrm, S, pos_base, tmp);
        } else {
            deltanet(m, l, i, nrm, S, pos_base, tmp);
        }
        if (lf) fwrite(tmp + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* sublayer output */
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* post-deltanet residual */
        if (g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        moe(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);
        if (g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
    }
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens) pin_hot_experts(m);
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(nrm); free(tmp); free(last);
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
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
    Slot *s;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        int lru = -1;
        for (int i = 0; i < lc->n; i++) { if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue; if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i; }
        if (lru < 0) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
        s = &lc->slots[lru]; s->pinned = 0;
    }
    s->eid = -1; s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);
    load_expert_merged(m, layer, eid, s);
    pthread_mutex_lock(&g_pilot_mx);
    s->eid = eid; s->pinned = m->is_pinned[layer*c->n_experts+eid]; s->used = ++m->clock;
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
    matmul(logits, nrm_x, l->gate, S, D, E);
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
            int found = 0; pthread_mutex_lock(&g_pilot_mx); LCache *lc = &m->cache[lnext];
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == eid) { found = 1; break; }
            pthread_mutex_unlock(&g_pilot_mx);
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
    m->kv_cap = m->max_t;
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
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
        int one = best; logit = step(m, &one, 1, len - 1);
    }
}

static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
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

/* ===================== optional GPU backend probe =====================
 * Phase-1 of GPU support: at startup, transparently detect whether a
 * Vulkan compute-capable GPU (integrated or discrete) exists. If so the
 * engine will later use it; if not (no loader / no device) it silently
 * falls back to the CPU path. The user never configures anything.
 *
 * This probe does NOT link Vulkan at compile time. It dynamically loads
 * the platform Vulkan loader (vulkan-1.dll / libvulkan.so) via
 * LoadLibrary/dlopen and resolves the few symbols needed to enumerate
 * devices, so the default binary stays zero-dependency. Actual compute
 * (matmul_q on GPU) is a later phase; this step only reports availability. */
/* (g_gpu_backend / g_gpu_count / GPU_BACKEND_* are declared above moe().) */

/* --- hand-declared minimal Vulkan pieces (avoids needing vulkan.h) --- */
#define VK_RESULT_SUCCESS 0
#define VK_QUEUE_COMPUTE_BIT 0x00000020u
#define VK_ST_APPLICATION_INFO 0
#define VK_ST_INSTANCE_CREATE_INFO 1
#define VK_QF_OFF_FLAGS 0    /* queueFlags is the FIRST field of VkQueueFamilyProperties (it has NO sType/pNext) */
#define VK_QF_STRIDE 32      /* sizeof(VkQueueFamilyProperties)=4(queueFlags)+4(queueCount)+12(minImgGran)+4(tsBits)+8(minAlloc)=32 */

typedef void* VkInstance;       /* opaque handle */
typedef void* VkPhysicalDevice; /* opaque handle (pointer-sized everywhere) */
typedef struct { uint32_t sType; const void *pNext; const char *pApplicationName; uint32_t applicationVersion; const char *pEngineName; uint32_t engineVersion; uint32_t apiVersion; } VkAppInfoMini;
typedef struct { uint32_t sType; const void *pNext; const void *pApplicationInfo; uint32_t enabledLayerCount; const char* const *ppEnabledLayerNames; uint32_t enabledExtensionCount; const char* const *ppEnabledExtensionNames; } VkInstCIMini;
typedef int  (*PFN_vkCreateInstance)(const VkInstCIMini*, const void*, VkInstance*);
typedef int  (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, void*);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void*);

static int gpu_probe(void) {
    g_gpu_backend = GPU_BACKEND_CPU;
    g_gpu_count = 0;

    /* 0) optional hard disable for debugging (COLIBRI_GPU=0 -> CPU only) */
    {
        const char *ge = getenv("COLIBRI_GPU");
        if (ge && *ge == '0') { fprintf(stderr, "[gpu] COLIBRI_GPU=0 -> CPU backend forced\n"); return 0; }
    }

    /* 1) locate the Vulkan loader dynamically (no compile-time link) */
#if defined(_WIN32)
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) { fprintf(stderr, "[gpu] no vulkan-1.dll present -> CPU backend\n"); return 0; }
    #define VK_SYM(n) ((void*)GetProcAddress(lib, n))
#else
    void *lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!lib) lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!lib) { fprintf(stderr, "[gpu] no libvulkan present -> CPU backend\n"); return 0; }
    #define VK_SYM(n) dlsym(lib, n)
#endif
    PFN_vkCreateInstance  f_create = (PFN_vkCreateInstance)VK_SYM("vkCreateInstance");
    PFN_vkEnumeratePhysicalDevices f_enum = (PFN_vkEnumeratePhysicalDevices)VK_SYM("vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties f_qf = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)VK_SYM("vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkDestroyInstance f_destroy = (PFN_vkDestroyInstance)VK_SYM("vkDestroyInstance");
    #undef VK_SYM
    if (!f_create || !f_enum || !f_qf || !f_destroy) { fprintf(stderr, "[gpu] vulkan loader missing required symbols -> CPU backend\n"); return 0; }

    /* 2) create a bare instance (no layers/extensions needed for probing) */
    VkAppInfoMini ai; memset(&ai,0,sizeof ai); ai.sType = VK_ST_APPLICATION_INFO; ai.apiVersion = (1u<<22);
    VkInstCIMini ci; memset(&ci,0,sizeof ci); ci.sType = VK_ST_INSTANCE_CREATE_INFO; ci.pApplicationInfo = &ai;
    VkInstance inst = NULL;
    if (f_create(&ci, NULL, &inst) != VK_RESULT_SUCCESS || !inst) { fprintf(stderr, "[gpu] vkCreateInstance failed -> CPU backend\n"); return 0; }

    /* 3) enumerate physical devices */
    uint32_t ndev = 0;
    f_enum(inst, &ndev, NULL);
    if (ndev == 0) { fprintf(stderr, "[gpu] vulkan instance ok but 0 devices -> CPU backend\n"); f_destroy(inst, NULL); return 0; }
    VkPhysicalDevice *devs = (VkPhysicalDevice*)calloc(ndev, sizeof(VkPhysicalDevice));
    f_enum(inst, &ndev, devs);

    /* 4) count devices exposing a compute-capable queue family */
    int compute_devs = 0;
    for (uint32_t i = 0; i < ndev; i++) {
        uint32_t nf = 0;
        f_qf(devs[i], &nf, NULL);
        if (nf == 0) continue;
        char *qf = (char*)calloc(nf, VK_QF_STRIDE);
        f_qf(devs[i], &nf, qf);
        for (uint32_t j = 0; j < nf; j++) {
            uint32_t flags = *(uint32_t*)(qf + (size_t)j*VK_QF_STRIDE + VK_QF_OFF_FLAGS);
            if (flags & VK_QUEUE_COMPUTE_BIT) { compute_devs++; break; }
        }
        free(qf);
    }
    free(devs);
    f_destroy(inst, NULL);

    if (compute_devs > 0) {
        g_gpu_backend = GPU_BACKEND_VULKAN;
        g_gpu_count = compute_devs;
        fprintf(stderr, "[gpu] detected %d Vulkan compute device(s) -> GPU backend ENABLED\n", compute_devs);
    } else {
        fprintf(stderr, "[gpu] %u device(s) found but none compute-capable -> CPU backend\n", ndev);
    }
    return g_gpu_backend;
}

#ifndef QWEN36_NO_MAIN
int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    if (g_wide < 1) g_wide = 1; if (g_wide > 4) g_wide = 4;
    if (getenv("OPENAI")) g_openai = 1;                       /* OpenAI-compatible output */
    const char *mv = getenv("MODEL"); if (mv && *mv) g_model = mv;
    int hot_n = getenv("HOT") ? atoi(getenv("HOT")) : 0;
    int cap   = argc > 1 ? atoi(argv[1]) : 16;
    int bits  = argc > 2 ? atoi(argv[2]) : 4;
    if (bits < 2 || bits > 8) { fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits); return 1; }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf   = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;

    fprintf(stderr, "== qwen36 Phase-2 engine | cache=%d/layer bits=%d pilot=%d wide=%d hot=%d smooth=%.2f conf=%.2f ==\n",
           cap, bits, g_pilot, g_wide, hot_n, smooth, conf);

    gpu_probe();   /* transparent: use GPU if a Vulkan compute device exists, else CPU */

    int is_ref = 0;
    int rplen = (int)strlen(refpath);
    if (rplen>=5 && strcmp(refpath+rplen-5, ".json")==0) is_ref = 1;

    int *prompt=NULL, *full=NULL, *out=NULL;
    int np=0, nfull=0, n_new=0;
    char *buf=NULL, *arena=NULL;

    /* load tokenizer early so text-prompt mode can encode before model_init */
    {
        const char *tokpath = getenv("TOK");
        if (tokpath && *tokpath) load_tokenizer(tokpath);
        else if (argc > 4 && argv[4] && *argv[4]) load_tokenizer(argv[4]);
        else { char tpb[2048]; snprintf(tpb,sizeof tpb,"%s/tokenizer.json",snap); load_tokenizer(tpb); }
    }

    if (is_ref) {
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
    fprintf(stderr, "resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());

    /* optional transparent Vulkan compute backend for the routed-expert GEMVs.
     * Only attempted when gpu_probe() found a compute-capable device; if vg_init
     * fails for any reason we silently fall back to the CPU MoE path. */
    if (g_gpu_backend == GPU_BACKEND_VULKAN) {
        vg_cfg vc;
        vc.n_layers = m.c.n_layers;
        vc.hidden   = m.c.hidden;
        vc.inter    = m.c.inter;
        vc.cap      = cap;
        vc.topk     = m.c.topk;
        vc.weight_bits = container_is_int4(&m) ? 4 : 8;  /* by on-disk size, not meta.ebits */
        if (vg_init(&vc) == 0) {
            fprintf(stderr, "[gpu] MoE expert GEMVs -> Vulkan compute device\n");
            atexit(vg_shutdown);
        } else {
            fprintf(stderr, "[gpu] vg_init failed -> keeping CPU MoE path\n");
            g_gpu_backend = GPU_BACKEND_CPU;
        }
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

    if (is_ref) {
        int match = 0;
        printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
        printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
        if (g_tok) { printf("Text      : "); print_decoded(out, np, nfull); printf("\n"); }
        printf("\nMatching tokens: %d/%d\n", match, n_new);
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
    fprintf(stderr, "\nPEAK RSS: %.2f GB\n", rss_gb());
    fprintf(stderr, "Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    fprintf(stderr, "Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    free(buf); free(arena);
    return 0;
}
#endif /* QWEN36_NO_MAIN */
