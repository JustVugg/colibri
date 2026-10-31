/* Real engine state, no model, network, GPU or external fixture required. */
#define GLM53_NO_MAIN
#include "../glm53.c"
static int checks;
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
    checks++;
}
static void mark(GSession *s, float value) {
    for (int i=0;i<32;i++) s->layer[1].kda_state[i]=value+i*0.125f;
    for (int i=0;i<48;i++) s->layer[1].kda_window[i]=-value-i*0.0625f;
}
int main(void) {
    omp_set_num_threads(1);
    GModel m={0}; Cfg *c=&m.c;
    c->n_layers=2; c->kv_lora=8; c->index_hd=4; c->vocab=256;
    c->kda_heads=2; c->kda_hd=4; c->kda_proj=8; c->conv_k=2; c->is_full[0]=1;
    GSession s={0}; s.cap=256; s.filled=80;
    s.layer=calloc(2,sizeof(*s.layer)); check(s.layer!=NULL,"layers allocated");
    s.layer[0].latent=calloc(256*8,sizeof(float));
    s.layer[0].ikeys=calloc(256*4,sizeof(float));
    s.layer[0].igates=calloc(256*4,sizeof(float));
    s.layer[1].kda_state=calloc(32,sizeof(float));
    s.layer[1].kda_window=calloc(48,sizeof(float));
    check(s.layer[0].latent && s.layer[0].ikeys && s.layer[0].igates &&
          s.layer[1].kda_state && s.layer[1].kda_window,"state allocated");
    int ids[256],branch[256]; uint64_t keys[256]; float logits[256];
    for(int i=0;i<256;i++){ids[i]=i;keys[i]=i;logits[i]=i*0.003f;}
    KVSlot slot={0};slot.session=&s;mark(&s,80);
    check(slot_pin_save(&m,&slot,ids,80,logits),"checkpoint at 80");
    s.filled=120;mark(&s,120);slot_remember(&slot,ids,keys,120);
    check(slot_pin_restore(&m,&slot,ids,140,120)==80 && s.filled==80,
          "negative control: old restore rewinds valid state");
    s.filled=120;mark(&s,120);
    float state[32],window[48];
    memcpy(state,s.layer[1].kda_state,sizeof state);
    memcpy(window,s.layer[1].kda_window,sizeof window);
    check(slot_pin_for_request(&m,&slot,ids,140,120,120,0)==0 && s.filled==120,
          "plain extension does not rewind");
    check(!memcmp(state,s.layer[1].kda_state,sizeof state) &&
          !memcmp(window,s.layer[1].kda_window,sizeof window),"live state is bit exact");
    check(slot_pin_for_request(&m,&slot,ids,140,120,120,1)==80 && s.filled==80,
          "predictor requests still restore the pin");
    s.filled=120;mark(&s,120);memcpy(branch,ids,sizeof ids);branch[100]=2000;
    check(slot_pin_for_request(&m,&slot,branch,140,100,0,0)==80,
          "diverged suffix restores a valid common checkpoint");
    s.filled=120;mark(&s,120);branch[0]=2001;
    check(slot_pin_for_request(&m,&slot,branch,140,0,0,0)==0 && s.filled==120,
          "unrelated prefix cannot restore or modify state");
    slot_pin_drop(&m,&slot);free(slot.tokens);free(slot.keys);
    free(s.layer[0].latent);free(s.layer[0].ikeys);free(s.layer[0].igates);
    free(s.layer[1].kda_state);free(s.layer[1].kda_window);free(s.layer);
    printf("glm53 live-prefix policy: %d checks passed\n",checks);return 0;
}
