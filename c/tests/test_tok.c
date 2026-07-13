/* Validazione del tokenizer C contro l'oracolo HF.
 * build da c/: gcc -O2 tests/test_tok.c -o tok_test
 * uso:  ./tok_test <tokenizer.json>   (legge righe "TEXT\tID,ID,.." da stdin) */
#define _GNU_SOURCE
#include "../tok.h"

int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: %s tokenizer.json < cases\n",argv[0]); return 1; }
    Tok T;
    tok_load(&T, argv[1]);
    fprintf(stderr,"loaded: vocab_ids=%d specials=%d\n", T.n_ids, T.nsp);
    char *line=NULL; size_t cap=0; ssize_t nr;
    int pass=0, tot=0, dpass=0;
    while((nr=getline(&line,&cap,stdin))>=0){
        if(nr>0 && line[nr-1]=='\n'){ line[--nr]=0; }
        if(nr==0) continue;
        char *tab=strchr(line,'\t'); if(!tab) continue;
        *tab=0; const char *text=line; const char *idstr=tab+1;
        /* il testo puo' contenere \n e \t codificati come \\n \\t */
        char tbuf[4096]; int tn=0;
        for(const char *q=text; *q && tn<4095; q++){
            if(q[0]=='\\' && q[1]=='n'){ tbuf[tn++]='\n'; q++; }
            else if(q[0]=='\\' && q[1]=='t'){ tbuf[tn++]='\t'; q++; }
            else if(q[0]=='\\' && q[1]=='r'){ tbuf[tn++]='\r'; q++; }
            else if(q[0]=='\\' && q[1]=='\\'){ tbuf[tn++]='\\'; q++; }
            else tbuf[tn++]=*q;
        }
        tbuf[tn]=0;
        int exp[4096], ne=0;
        for(const char *q=idstr; *q; ){ while(*q==','||*q==' ')q++; if(!*q)break; exp[ne++]=atoi(q); while(*q&&*q!=',')q++; }
        int got[4096]; int ng=tok_encode(&T,tbuf,tn,got,4096);
        int ok = (ng==ne); for(int i=0;i<ng&&ok;i++) ok = (got[i]==exp[i]);
        tot++; if(ok) pass++;
        /* round-trip decode */
        char dec[8192]; int dn=tok_decode(&T,got,ng,dec,8191);
        int drt = (dn==tn) && !memcmp(dec,tbuf,tn);
        if(drt) dpass++;
        if(!ok || !drt){
            fprintf(stderr,"MISMATCH text=%s\n  exp(%d):",text,ne); for(int i=0;i<ne;i++)fprintf(stderr," %d",exp[i]);
            fprintf(stderr,"\n  got(%d):",ng); for(int i=0;i<ng;i++)fprintf(stderr," %d",got[i]);
            fprintf(stderr,"\n  decode_ok=%d\n", drt);
        }
    }
    printf("ENCODE: %d/%d  DECODE(round-trip): %d/%d\n", pass,tot, dpass,tot);
    return pass==tot ? 0 : 2;
}
