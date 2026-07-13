/* grammar.h — draft grammaticale (#48): GBNF (sottoinsieme) valutata a livello di BYTE.
 *
 * Idea: nei workload a output vincolato (JSON/NDJSON, function calling, estrazione
 * strutturata) una frazione dei token e' DETERMINISTICA data la grammatica: parentesi,
 * virgolette, nomi delle chiavi, separatori, valori enum. Quegli span sono draft
 * gratuiti ad acceptance ~1: nessuna testa, nessuna lookup table — la verifica
 * batch-union li conferma e paga UN forward per piu' token. E si aggancia anche dove
 * la testa MTP int4 non parte (#8).
 *
 * La grammatica non vincola MAI il campionamento: propone solo draft, che la verifica
 * accetta o rifiuta come qualunque altro draft. Grammatica sbagliata o fuori sync =>
 * draft rifiutati, output IDENTICO. E' un acceleratore puro, mai un filtro.
 *
 * Sottoinsieme GBNF (stile llama.cpp), valutato sui BYTE:
 *   root ::= obj+                          # la regola di partenza si chiama "root"
 *   obj  ::= "{" pair ("," pair)* "}" "\n"
 *   str  ::= "\"" [^"\\]* "\""
 * Supportato: letterali "..." (escape \" \\ \n \r \t \xHH), classi [a-z0-9-] anche
 * negate [^...], riferimenti a regole, gruppi (...), postfissi ? * +, commenti #,
 * alternate con |, epsilon come "". Le regole possono estendersi su piu' righe: una
 * nuova regola inizia dove un identificatore e' seguito da "::=".
 * NON supportato: ripetizioni {m,n}, range unicode nelle classi (le classi lavorano
 * sui byte; per l'UTF-8 multibyte usare i letterali, che passano i byte grezzi).
 * Ricorsione sinistra: intercettata dal tetto di profondita' -> il walker si spegne
 * (alive=0) e la generazione prosegue senza draft. Mai un blocco, mai un crash.
 *
 * Il walker e' un PDA con INSIEME di stack (come llama.cpp): ogni stack in forma
 * normale ha in cima un simbolo terminale (classe di byte) oppure e' vuoto (parse
 * completabile qui). gr_forced() estende il prefisso finche' esiste UN SOLO byte
 * legale e il parse non e' terminabile: quel prefisso e' il draft forzato.
 */
#ifndef COLI_GRAMMAR_H
#define COLI_GRAMMAR_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GR_MAX_RULES  1024
#define GR_MAX_STACKS 64      /* ambiguita' massima seguita in parallelo */
#define GR_MAX_DEPTH  64      /* profondita' massima di uno stack del PDA */

typedef struct { uint8_t bits[32]; } GrCls;              /* insieme di byte ammessi */
enum { GR_CLS = 0, GR_REF = 1 };
typedef struct { uint8_t t; int16_t ref; GrCls c; } GrSym;
typedef struct { GrSym *s; int n, cap; } GrAlt;          /* una sequenza di simboli */
typedef struct { GrAlt *a; int n, cap; char name[64]; } GrRule;
typedef struct { GrRule r[GR_MAX_RULES]; int n; int root; char err[160]; } Grammar;

/* frame = posizione dentro un alternate: (regola, alternate, simbolo) */
typedef struct { int16_t r, a, s; } GrFrame;
typedef struct { GrFrame f[GR_MAX_DEPTH]; int16_t n; } GrStack;
typedef struct { Grammar *G; GrStack st[GR_MAX_STACKS]; int n; int alive; } GrState;

/* ---------- costruzione ---------- */

static int gr__alt_new(Grammar *G, int ri){
    GrRule *R=&G->r[ri];
    if(R->n==R->cap){ int nc=R->cap?R->cap*2:4;
        GrAlt *na=(GrAlt*)realloc(R->a,(size_t)nc*sizeof(GrAlt)); if(!na) return -1;
        R->a=na; R->cap=nc; }
    memset(&R->a[R->n],0,sizeof(GrAlt));
    return R->n++;
}
static int gr__push(Grammar *G, int ri, int ai, const GrSym *sy){
    GrAlt *A=&G->r[ri].a[ai];
    if(A->n==A->cap){ int nc=A->cap?A->cap*2:8;
        GrSym *ns=(GrSym*)realloc(A->s,(size_t)nc*sizeof(GrSym)); if(!ns) return -1;
        A->s=ns; A->cap=nc; }
    A->s[A->n++]=*sy; return 0;
}
static int gr__rule(Grammar *G, const char *name, int len){
    if(len>63) len=63;
    for(int i=0;i<G->n;i++)
        if((int)strlen(G->r[i].name)==len && !memcmp(G->r[i].name,name,(size_t)len)) return i;
    if(G->n>=GR_MAX_RULES) return -1;
    GrRule *R=&G->r[G->n]; memset(R,0,sizeof *R);
    memcpy(R->name,name,(size_t)len);
    return G->n++;
}
static int gr__anon(Grammar *G){                          /* regola sintetica ($n non collide: '$' non e' un identificatore */
    if(G->n>=GR_MAX_RULES) return -1;
    GrRule *R=&G->r[G->n]; memset(R,0,sizeof *R);
    snprintf(R->name,sizeof R->name,"$%d",G->n);
    return G->n++;
}

/* ---------- parser GBNF ---------- */

static const char* gr__ws(const char *p){
    for(;;){
        while(*p==' '||*p=='\t'||*p=='\r'||*p=='\n') p++;
        if(*p=='#'){ while(*p && *p!='\n') p++; continue; }
        return p;
    }
}
static int gr__idch(char c){
    return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-';
}
static int gr__idlen(const char *p){ int n=0; while(gr__idch(p[n])) n++; return n; }
static int gr__hex(char c){
    if(c>='0'&&c<='9') return c-'0';
    if(c>='a'&&c<='f') return c-'a'+10;
    if(c>='A'&&c<='F') return c-'A'+10;
    return -1;
}
static int gr__esc(const char **pp){                      /* dopo la barra: byte 0-255 o -1 */
    const char *p=*pp; int c=-1;
    switch(*p){
        case 'n': c='\n'; break;  case 'r': c='\r'; break;  case 't': c='\t'; break;
        case '"': c='"';  break;  case '\\':c='\\'; break;
        case '[': c='[';  break;  case ']': c=']';  break;
        case '-': c='-';  break;  case '^': c='^';  break;
        case 'x': { int h=gr__hex(p[1]), l=gr__hex(p[2]);
                    if(h>=0&&l>=0){ c=h*16+l; p+=2; } break; }
        default: return -1;
    }
    if(c<0) return -1;
    *pp=p+1; return c;
}
static int gr__lit(Grammar *G, int ri, int ai, const char **pp){
    const char *p=*pp+1;
    while(*p && *p!='"'){
        int b;
        if(*p=='\\'){ p++; b=gr__esc(&p);
            if(b<0){ snprintf(G->err,sizeof G->err,"invalid escape in literal"); return -1; } }
        else b=(unsigned char)*p++;
        GrSym s; memset(&s,0,sizeof s); s.t=GR_CLS; s.c.bits[b>>3]|=(uint8_t)(1u<<(b&7));
        if(gr__push(G,ri,ai,&s)){ snprintf(G->err,sizeof G->err,"out of memory"); return -1; }
    }
    if(*p!='"'){ snprintf(G->err,sizeof G->err,"unterminated literal"); return -1; }
    *pp=p+1; return 0;
}
static int gr__cls(Grammar *G, int ri, int ai, const char **pp){
    const char *p=*pp+1; int neg=0;
    GrSym s; memset(&s,0,sizeof s); s.t=GR_CLS;
    if(*p=='^'){ neg=1; p++; }
    while(*p && *p!=']'){
        int lo, hi;
        if(*p=='\\'){ p++; lo=gr__esc(&p);
            if(lo<0){ snprintf(G->err,sizeof G->err,"invalid escape in character class"); return -1; } }
        else lo=(unsigned char)*p++;
        hi=lo;
        if(*p=='-' && p[1] && p[1]!=']'){
            p++;
            if(*p=='\\'){ p++; hi=gr__esc(&p);
                if(hi<0){ snprintf(G->err,sizeof G->err,"invalid escape in character class"); return -1; } }
            else hi=(unsigned char)*p++;
        }
        if(hi<lo){ int t=lo; lo=hi; hi=t; }
        for(int b=lo;b<=hi;b++) s.c.bits[b>>3]|=(uint8_t)(1u<<(b&7));
    }
    if(*p!=']'){ snprintf(G->err,sizeof G->err,"unterminated character class"); return -1; }
    if(neg) for(int i=0;i<32;i++) s.c.bits[i]=(uint8_t)~s.c.bits[i];
    *pp=p+1;
    if(gr__push(G,ri,ai,&s)){ snprintf(G->err,sizeof G->err,"out of memory"); return -1; }
    return 0;
}
/* postfisso ? * + sull'ITEM appena letto (simboli [n0, n) dell'alternate corrente).
 * L'item diventa una regola anonima I; poi:  ?  ->  R ::= I | ""
 *                                            *  ->  R ::= I R | ""
 *                                            +  ->  R ::= I R | I           */
static int gr__postfix(Grammar *G, int ri, int ai, int n0, char op){
    int k=G->r[ri].a[ai].n-n0;
    if(k<=0) return 0;                                    /* postfisso su "" : no-op */
    int ii=gr__anon(G); if(ii<0) goto full;
    int ia=gr__alt_new(G,ii); if(ia<0) goto full;
    for(int j=0;j<k;j++) if(gr__push(G,ii,ia,&G->r[ri].a[ai].s[n0+j])) goto full;
    G->r[ri].a[ai].n=n0;
    int rr=gr__anon(G); if(rr<0) goto full;
    GrSym I; memset(&I,0,sizeof I); I.t=GR_REF; I.ref=(int16_t)ii;
    GrSym R; memset(&R,0,sizeof R); R.t=GR_REF; R.ref=(int16_t)rr;
    int a0=gr__alt_new(G,rr); if(a0<0) goto full;
    if(gr__push(G,rr,a0,&I)) goto full;
    if(op=='*'||op=='+') if(gr__push(G,rr,a0,&R)) goto full;
    int a1=gr__alt_new(G,rr); if(a1<0) goto full;         /* "" per ? e *, I per + */
    if(op=='+') if(gr__push(G,rr,a1,&I)) goto full;
    if(gr__push(G,ri,ai,&R)) goto full;                   /* l'item nell'alternate diventa R */
    return 0;
full:
    snprintf(G->err,sizeof G->err,"grammar is too large");
    return -1;
}
static int gr__alts(Grammar *G, int ri, const char **pp, int depth, int in_group){
    if(depth>32){ snprintf(G->err,sizeof G->err,"groups are nested too deeply"); return -1; }
    const char *p=*pp;
    int ai=gr__alt_new(G,ri);
    if(ai<0){ snprintf(G->err,sizeof G->err,"out of memory"); return -1; }
    for(;;){
        p=gr__ws(p);
        if(!*p){
            if(in_group){ snprintf(G->err,sizeof G->err,"missing ')'"); return -1; }
            break;
        }
        if(*p==')'){
            if(!in_group){ snprintf(G->err,sizeof G->err,"unexpected ')'"); return -1; }
            break;
        }
        if(*p=='|'){
            p++;
            ai=gr__alt_new(G,ri);
            if(ai<0){ snprintf(G->err,sizeof G->err,"out of memory"); return -1; }
            continue;
        }
        int n0=G->r[ri].a[ai].n;
        if(*p=='"'){
            if(gr__lit(G,ri,ai,&p)) return -1;
        } else if(*p=='['){
            if(gr__cls(G,ri,ai,&p)) return -1;
        } else if(*p=='('){
            p++;
            int gi=gr__anon(G);
            if(gi<0){ snprintf(G->err,sizeof G->err,"grammar is too large"); return -1; }
            if(gr__alts(G,gi,&p,depth+1,1)) return -1;
            p=gr__ws(p);
            if(*p!=')'){ snprintf(G->err,sizeof G->err,"missing ')'"); return -1; }
            p++;
            GrSym s; memset(&s,0,sizeof s); s.t=GR_REF; s.ref=(int16_t)gi;
            if(gr__push(G,ri,ai,&s)){ snprintf(G->err,sizeof G->err,"out of memory"); return -1; }
        } else if(gr__idch(*p)){
            int nl=gr__idlen(p);
            const char *after=gr__ws(p+nl);
            if(!in_group && !strncmp(after,"::=",3)) break;   /* inizia la prossima regola */
            int ref=gr__rule(G,p,nl);
            if(ref<0){ snprintf(G->err,sizeof G->err,"too many rules"); return -1; }
            p+=nl;
            GrSym s; memset(&s,0,sizeof s); s.t=GR_REF; s.ref=(int16_t)ref;
            if(gr__push(G,ri,ai,&s)){ snprintf(G->err,sizeof G->err,"out of memory"); return -1; }
        } else {
            snprintf(G->err,sizeof G->err,"unexpected character '%c'",*p); return -1;
        }
        p=gr__ws(p);
        if(*p=='?'||*p=='*'||*p=='+'){ if(gr__postfix(G,ri,ai,n0,*p)) return -1; p++; }
    }
    *pp=p;
    return 0;
}
/* parse del testo GBNF. 0 = ok; -1 = errore (messaggio in G->err). */
static int gr_parse(Grammar *G, const char *src){
    memset(G,0,sizeof *G); G->root=-1;
    const char *p=src;
    for(;;){
        p=gr__ws(p);
        if(!*p) break;
        int nl=gr__idlen(p);
        if(nl<=0){ snprintf(G->err,sizeof G->err,"expected a rule, found '%c'",*p); return -1; }
        const char *name=p;
        const char *q=gr__ws(p+nl);
        if(strncmp(q,"::=",3)){ snprintf(G->err,sizeof G->err,"expected '::=' after '%.*s'",nl,name); return -1; }
        p=q+3;
        int ri=gr__rule(G,name,nl);
        if(ri<0){ snprintf(G->err,sizeof G->err,"too many rules"); return -1; }
        if(G->r[ri].n>0){ snprintf(G->err,sizeof G->err,"duplicate rule '%.*s'",nl,name); return -1; }
        if(gr__alts(G,ri,&p,0,0)) return -1;
    }
    for(int i=0;i<G->n;i++){
        if(!strcmp(G->r[i].name,"root")) G->root=i;
        if(G->r[i].n==0){ snprintf(G->err,sizeof G->err,"rule '%s' is used but never defined",G->r[i].name); return -1; }
    }
    if(G->root<0){ snprintf(G->err,sizeof G->err,"missing 'root' rule"); return -1; }
    return 0;
}
static void gr_free(Grammar *G){
    for(int i=0;i<G->n;i++){
        for(int a=0;a<G->r[i].n;a++) free(G->r[i].a[a].s);
        free(G->r[i].a);
    }
    G->n=0;
}

/* ---------- walker (PDA a insieme di stack) ---------- */

static int gr__set_add(GrState *S, const GrStack *k){
    for(int i=0;i<S->n;i++)
        if(S->st[i].n==k->n && !memcmp(S->st[i].f,k->f,(size_t)k->n*sizeof(GrFrame))) return 1;
    if(S->n>=GR_MAX_STACKS) return 0;                     /* troppa ambiguita': fail-safe */
    S->st[S->n++]=*k; return 1;
}
/* porta lo stack in forma normale (cima = terminale, o stack vuoto = parse completo),
 * diramando sugli alternate delle regole referenziate. 0 = overflow (fail-safe). */
static int gr__normalize(Grammar *G, GrStack *k, GrState *out, int depth){
    for(;;){
        if(k->n==0) return gr__set_add(out,k);
        GrFrame *t=&k->f[k->n-1];
        GrAlt *A=&G->r[t->r].a[t->a];
        if(t->s>=A->n){ k->n--; continue; }               /* alternate esaurito: pop */
        GrSym *sy=&A->s[t->s];
        if(sy->t==GR_CLS) return gr__set_add(out,k);
        if(depth>=GR_MAX_DEPTH) return 0;                 /* ricorsione sinistra / epsilon-ciclo */
        t->s++;                                           /* il chiamante riprende OLTRE il ref */
        GrRule *C=&G->r[sy->ref];
        for(int a=0;a<C->n;a++){
            if(k->n>=GR_MAX_DEPTH) return 0;
            GrStack cp=*k;
            cp.f[cp.n].r=sy->ref; cp.f[cp.n].a=(int16_t)a; cp.f[cp.n].s=0; cp.n++;
            if(!gr__normalize(G,&cp,out,depth+1)) return 0;
        }
        return 1;
    }
}
static void gr_state_init(GrState *S, Grammar *G){
    S->G=G; S->n=0; S->alive=1;
    GrRule *R=&G->r[G->root];
    for(int a=0;a<R->n;a++){
        GrStack k; k.n=1;
        k.f[0].r=(int16_t)G->root; k.f[0].a=(int16_t)a; k.f[0].s=0;
        if(!gr__normalize(G,&k,S,0)){ S->alive=0; return; }
    }
    if(S->n==0) S->alive=0;
}
/* avanza di un byte. 1 = consumato; 0 = byte non ammesso (stato INVARIATO);
 * -1 = walker spento (overflow: da qui in poi niente piu' draft). */
static int gr_accept(GrState *S, unsigned char b){
    if(!S->alive) return -1;
    GrState out; out.G=S->G; out.n=0; out.alive=1;
    for(int i=0;i<S->n;i++){
        GrStack *k=&S->st[i];
        if(k->n==0) continue;                             /* parse gia' completo: non consuma */
        GrFrame *t=&k->f[k->n-1];
        GrSym *sy=&S->G->r[t->r].a[t->a].s[t->s];
        if(!(sy->c.bits[b>>3]&(1u<<(b&7)))) continue;
        GrStack cp=*k; cp.f[cp.n-1].s++;
        if(!gr__normalize(S->G,&cp,&out,0)){ S->alive=0; return -1; }
    }
    if(out.n==0) return 0;
    S->n=out.n;
    memcpy(S->st,out.st,(size_t)out.n*sizeof(GrStack));
    return 1;
}
/* insieme dei byte ammessi adesso (bitmap 256). Ritorna il conteggio;
 * *can_end = 1 se il parse puo' terminare qui (quindi il modello puo' emettere EOS). */
static int gr_admissible(const GrState *S, unsigned char mask[32], int *can_end){
    memset(mask,0,32); int end=0;
    for(int i=0;i<S->n;i++){
        const GrStack *k=&S->st[i];
        if(k->n==0){ end=1; continue; }
        const GrFrame *t=&k->f[k->n-1];
        const GrSym *sy=&S->G->r[t->r].a[t->a].s[t->s];
        for(int j=0;j<32;j++) mask[j]|=sy->c.bits[j];
    }
    int cnt=0;
    for(int j=0;j<32;j++){ unsigned v=mask[j]; while(v){ cnt+=v&1; v>>=1; } }
    if(can_end)*can_end=end;
    return cnt;
}
/* prefisso FORZATO: si estende finche' c'e' UN SOLO byte legale e il parse non e'
 * terminabile (li' il modello potrebbe fermarsi). Non muta S. Ritorna i byte scritti. */
static int gr_forced(const GrState *S, char *out, int max){
    if(!S->alive||S->n==0) return 0;
    GrState cp=*S;
    int n=0;
    while(n<max){
        unsigned char m[32]; int end;
        int c=gr_admissible(&cp,m,&end);
        if(c!=1||end) break;
        int b=0; while(b<256 && !(m[b>>3]&(1u<<(b&7)))) b++;
        if(b>=256 || gr_accept(&cp,(unsigned char)b)!=1) break;
        out[n++]=(char)b;
    }
    return n;
}

#endif /* COLI_GRAMMAR_H */
