#include <stdio.h>
#include <string.h>

#include "../grammar.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* alimenta il walker con una stringa; ritorna quanti byte sono stati consumati */
static int feed(GrState *S, const char *s){
    int n=0;
    while(s[n]){ if(gr_accept(S,(unsigned char)s[n])!=1) break; n++; }
    return n;
}

int main(void){
    static Grammar G;
    GrState S;
    char buf[512];

    /* letterale: tutto il prefisso e' forzato, poi il parse termina */
    CHECK(gr_parse(&G,"root ::= \"{\\\"id\\\":\"")==0);
    gr_state_init(&S,&G);
    CHECK(S.alive);
    CHECK(gr_forced(&S,buf,sizeof buf)==6);
    CHECK(!memcmp(buf,"{\"id\":",6));
    CHECK(feed(&S,"{\"id\":")==6);
    unsigned char m[32]; int end;
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);        /* parse completo: solo EOS */
    gr_free(&G);

    /* alternate: il forcing si ferma alla diramazione e riparte dopo */
    CHECK(gr_parse(&G,"root ::= \"a\" (\"b\" | \"c\") \"d\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='a');
    CHECK(feed(&S,"ab")==2);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='d');
    gr_free(&G);

    /* enum stile #48: forzata la virgoletta, poi dal primo byte l'intero valore */
    CHECK(gr_parse(&G,
        "root ::= \"\\\"\" val \"\\\"\"\n"
        "val  ::= \"no_fit\" | \"partial_fit\" | \"good_fit\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='\"');
    CHECK(feed(&S,"\"n")==2);
    CHECK(gr_forced(&S,buf,sizeof buf)==6 && !memcmp(buf,"o_fit\"",6));
    gr_free(&G);

    /* classi con range, star: niente forcing dove la grammatica dirama */
    CHECK(gr_parse(&G,"root ::= \"a\" [0-9]* \"b\"")==0);
    gr_state_init(&S,&G);
    CHECK(feed(&S,"a")==1);
    CHECK(gr_admissible(&S,m,&end)==11 && end==0);       /* 0-9 oppure b */
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    CHECK(feed(&S,"42b")==3);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* plus su gruppo: il forcing si ferma dove il parse e' terminabile */
    CHECK(gr_parse(&G,"root ::= (\"x\" \"\\n\")+")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==2 && buf[0]=='x' && buf[1]=='\n');
    CHECK(feed(&S,"x\n")==2);
    CHECK(gr_admissible(&S,m,&end)==1 && end==1);        /* puo' chiudere o aprire una riga */
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    gr_free(&G);

    /* postfisso su letterale multi-byte: ripete l'INTERO letterale */
    CHECK(gr_parse(&G,"root ::= \"ab\"+ \"c\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==2 && !memcmp(buf,"ab",2));
    CHECK(feed(&S,"ab")==2);
    CHECK(gr_admissible(&S,m,&end)==2 && end==0);        /* 'a' (ripete) o 'c' (chiude) */
    CHECK(feed(&S,"abc")==3);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* classe negata: l'unione con la chiusura copre tutti i byte -> nessun forcing */
    CHECK(gr_parse(&G,"root ::= \"\\\"\" [^\"]* \"\\\"\"")==0);
    gr_state_init(&S,&G);
    CHECK(feed(&S,"\"")==1);
    CHECK(gr_admissible(&S,m,&end)==256 && end==0);
    CHECK(feed(&S,"ciao \\ mondo\"")==13);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* desync: byte non ammesso NON muta lo stato */
    CHECK(gr_parse(&G,"root ::= \"ab\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_accept(&S,'x')==0);
    CHECK(gr_accept(&S,'a')==1 && gr_accept(&S,'b')==1);
    gr_free(&G);

    /* escape esadecimale e commenti; regole su piu' righe */
    CHECK(gr_parse(&G,
        "# grammatica di prova\n"
        "root ::= \"\\x41\"   # una A\n"
        "         [\\x30-\\x32]\n")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='A');
    CHECK(feed(&S,"A1")==2);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* errori di parse: regola indefinita, root mancante, ')' fuori posto */
    CHECK(gr_parse(&G,"root ::= foo")!=0);
    CHECK(gr_parse(&G,"a ::= \"x\"")!=0);
    CHECK(gr_parse(&G,"root ::= \"x\" )")!=0);

    /* ricorsione sinistra: parse ok, walker si spegne senza bloccare (fail-safe) */
    CHECK(gr_parse(&G,"root ::= root \"a\" | \"b\"")==0);
    gr_state_init(&S,&G);
    CHECK(!S.alive);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    gr_free(&G);

    /* grammatica NDJSON realistica (il workload di #48): span forzati lunghi */
    CHECK(gr_parse(&G,
        "root ::= riga+\n"
        "riga ::= \"{\\\"id\\\":\\\"\" chiave \"\\\",\\\"fit_category\\\":\\\"\" cat \"\\\"}\" \"\\n\"\n"
        "chiave ::= [a-z0-9-]+\n"
        "cat  ::= \"no_fit\" | \"partial_fit\" | \"good_fit\"\n")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==7 && !memcmp(buf,"{\"id\":\"",7));
    CHECK(feed(&S,"{\"id\":\"ocds-123\"")==16);
    int nf=gr_forced(&S,buf,sizeof buf);
    buf[nf]=0;
    CHECK(nf==17 && !strcmp(buf,",\"fit_category\":\""));
    CHECK(feed(&S,",\"fit_category\":\"p")==18);
    nf=gr_forced(&S,buf,sizeof buf); buf[nf]=0;
    CHECK(nf==13 && !strcmp(buf,"artial_fit\"}\n"));
    CHECK(feed(&S,"artial_fit\"}\n")==13);
    /* a fine riga il parse e' terminabile (riga+): niente forcing — il modello puo'
     * fermarsi qui. Il forcing riparte appena il modello apre la riga successiva. */
    CHECK(gr_admissible(&S,m,&end)==1 && end==1);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    CHECK(feed(&S,"{")==1);
    CHECK(gr_forced(&S,buf,sizeof buf)==6 && !memcmp(buf,"\"id\":\"",6));
    gr_free(&G);

    puts("test_grammar: ok");
    return 0;
}
