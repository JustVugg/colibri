/* Regressione #369: un solo logit NaN o +Inf faceva emettere a dist_build/dist_sample
 * il token 0 PER SEMPRE, in silenzio (il fallback `g_pbuf[i]>0` e' falso su NaN ovunque).
 * Ora la distribuzione degenere ripiega sull'argmax dei logit FINITI e avvisa una volta.
 *
 * Il test verifica: (a) con logit sani il campionamento resta corretto; (b) con NaN/+Inf
 * iniettato il token scelto e' l'argmax dei FINITI (mai 0 per default), su ogni posizione
 * del NaN inclusa lo[0]; (c) nessun NaN sopravvive in g_pbuf. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <stdio.h>

static uint32_t rs=0x1234abcd; static uint32_t xr(){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return rs;}

static int pbuf_has_nan(int V){ for(int i=0;i<V;i++) if(!isfinite(g_pbuf[i])) return 1; return 0; }

int main(void){
    int fail=0, V=2000;
    float *lo=malloc(V*sizeof(float));
    g_temp=0.7f; g_nuc=0.90f;                     /* il path serve di default */

    /* (a) logit sani: il campionato deve avere prob > 0 e nessun NaN nel buffer */
    for(int i=0;i<V;i++) lo[i]=(float)((int)(xr()%2000)-1000)/100.f;
    int known=1337; lo[known]=50.f;               /* picco netto */
    dist_build(lo,V);
    if(pbuf_has_nan(V)){ printf("  FAIL: NaN in g_pbuf con logit sani\n"); fail=1; }
    if(g_pbuf[known]<=0.f){ printf("  FAIL: il picco ha prob 0\n"); fail=1; }
    if(!fail) printf("  logit sani: distribuzione valida, picco vivo            ok\n");

    /* (b) NaN/+Inf iniettato in varie posizioni; il finito-argmax deve vincere */
    float bad[3]; bad[0]=NAN; bad[1]=INFINITY; bad[2]=-INFINITY;
    const char *bn[3]={"NaN","+Inf","-Inf"};
    for(int b=0;b<3;b++){
        for(int pos=0;pos<3;pos++){               /* lo[0], meta', ultimo */
            for(int i=0;i<V;i++) lo[i]=(float)((int)(xr()%400)-200)/100.f;
            int amax=777; lo[amax]=9.0f;           /* massimo FINITO atteso */
            int at = pos==0?0 : pos==1?V/2 : V-1;
            if(at==amax) amax=amax+1;              /* non sovrapporre */
            lo[amax]=9.0f;
            lo[at]=bad[b];                         /* veleno */
            dist_build(lo,V);
            if(pbuf_has_nan(V)){ printf("  FAIL: NaN sopravvive (%s @ %d)\n",bn[b],at); fail=1; continue; }
            /* con -Inf il fallback puo' non scattare (max finito resta), ma il buffer
             * deve restare valido e sommare ~1: con +Inf/NaN scatta il delta su amax */
            int picked=-1; float pv=-1;
            for(int i=0;i<V;i++) if(g_pbuf[i]>pv){pv=g_pbuf[i];picked=i;}
            if(b<2 && picked!=amax){                /* NaN e +Inf: delta esatto su amax */
                printf("  FAIL: %s @ %d -> picked %d, atteso argmax finito %d\n",bn[b],at,picked,amax); fail=1;
            }
        }
    }
    if(!fail) printf("  NaN/+Inf iniettato: argmax dei finiti vince, mai 0/NaN   ok\n");

    /* (c) caso estremo: TUTTI non finiti -> non deve crashare, buffer valido */
    for(int i=0;i<V;i++) lo[i]=NAN;
    dist_build(lo,V);
    if(pbuf_has_nan(V)){ printf("  FAIL: tutti-NaN lascia NaN nel buffer\n"); fail=1; }
    else printf("  tutti non-finiti: nessun crash, buffer valido               ok\n");

    printf(fail?"test_sample_nan: FAIL\n":"test_sample_nan: ok\n");
    return fail;
}
