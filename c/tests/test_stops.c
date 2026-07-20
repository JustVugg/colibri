/* Stop-token arming: the engine must not depend on a converted checkpoint's
 * config.json being complete.
 *
 * Why this test exists (#298, #307): GLM-5.2 declares THREE eos ids
 * (<|endoftext|>, <|user|>, <|observation|>). A conversion tool that rewrites
 * config.json with a reduced eos_token_id list leaves the engine stopping on
 * fewer tokens than the model actually emits -- and the ones it misses get
 * DETOKENIZED AND PRINTED INTO THE CHAT as literal text, with generation
 * continuing past the real end of the turn. @woolcoxm hit exactly this on a
 * g64 checkpoint and confirmed it by comparing token ids.
 *
 * Two independent defenses, one test each:
 *   1. eos_token_id is unioned from generation_config.json (HuggingFace's
 *      authority for generation) as well as config.json.
 *   2. every added-token the TOKENIZER marks "special":true is a stop, whatever
 *      the configs say. Those are control tokens (<|user|>, <sop>, ...) and are
 *      never legitimate content. <think>/<tool_call> are "special":false and
 *      must NOT be swept up -- they are real output.
 *
 * Defense 2 is what makes this robust against checkpoints we don't control:
 * even with BOTH configs mutilated, a control token cannot leak into a reply. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static const char *TOKJSON =
"{\"model\":{\"vocab\":{\"a\":0,\"b\":1,\"c\":2},\"merges\":[[\"a\",\"b\"]]},"
" \"added_tokens\":["
"   {\"id\":100,\"content\":\"<|endoftext|>\",\"special\":true},"
"   {\"id\":101,\"content\":\"<|user|>\",\"special\":true},"
"   {\"id\":102,\"content\":\"<|observation|>\",\"special\":true},"
"   {\"id\":103,\"content\":\"<|assistant|>\",\"special\":true},"
"   {\"id\":104,\"content\":\"<sop>\",\"special\":true},"
"   {\"id\":110,\"content\":\"<think>\",\"special\":false},"
"   {\"id\":111,\"content\":\"<tool_call>\",\"special\":false}"
"]}";

/* minimal config.json that survives load_cfg's range validation */
static void write_cfg(const char *dir, const char *fname, const char *eos_json){
    char p[512]; snprintf(p,sizeof(p),"%s/%s",dir,fname);
    FILE *f=fopen(p,"w"); if(!f){ perror(p); exit(1); }
    fprintf(f,"{\"hidden_size\":64,\"num_hidden_layers\":2,\"num_attention_heads\":4,"
              "\"n_routed_experts\":8,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,"
              "\"intermediate_size\":64,\"first_k_dense_replace\":1,\"q_lora_rank\":0,"
              "\"kv_lora_rank\":16,\"qk_nope_head_dim\":8,\"qk_rope_head_dim\":8,"
              "\"v_head_dim\":8,\"n_shared_experts\":1,\"vocab_size\":200,"
              "\"n_group\":1,\"topk_group\":1,\"rope_theta\":10000.0");
    if(eos_json) fprintf(f,",\"eos_token_id\":%s",eos_json);
    fprintf(f,"}\n"); fclose(f);
}
static void write_tok(const char *dir){
    char p[512]; snprintf(p,sizeof(p),"%s/tokenizer.json",dir);
    FILE *f=fopen(p,"w"); if(!f){ perror(p); exit(1); }
    fputs(TOKJSON,f); fclose(f);
}
static void rm_file(const char *dir, const char *fname){
    char p[512]; snprintf(p,sizeof(p),"%s/%s",dir,fname); remove(p);
}

static int expect(const char *what, int id, int want){
    int got=is_stop(id);
    if(got!=want){ fprintf(stderr,"  FAIL %s: token %d stop=%d, expected %d\n",what,id,got,want); return 1; }
    return 0;
}

int main(void){
    int fail=0;
    /* Relative to the CWD, like test_compat_direct's TMPF — NOT "/tmp/...".
     * These binaries are built by MinGW into native Windows .exe files, which
     * resolve Windows paths: "/tmp" is not one, so mkdtemp there fails ENOENT
     * and the whole `make check` goes red on the windows job (and only there). */
    char dir[]="test_stops_XXXXXX";
    if(!mkdtemp(dir)){ perror("mkdtemp"); return 1; }
    write_tok(dir);
    char tkp[512]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",dir);
    Tok T; tok_load(&T,tkp);
    printf("test_stops: stop arming vs incomplete checkpoint metadata\n");

    /* 1. the tokenizer's "special" flag must survive loading, and must NOT
     *    swallow <think>/<tool_call>, which are real content. */
    if(!T.id_special[101]){ fprintf(stderr,"  FAIL <|user|> not marked special\n"); fail=1; }
    if(!T.id_special[104]){ fprintf(stderr,"  FAIL <sop> not marked special\n"); fail=1; }
    if(T.id_special[110]){ fprintf(stderr,"  FAIL <think> wrongly marked special\n"); fail=1; }
    if(T.id_special[111]){ fprintf(stderr,"  FAIL <tool_call> wrongly marked special\n"); fail=1; }
    if(!fail) printf("  tokenizer: special flag parsed, <think>/<tool_call> excluded   ok\n");

    /* 2. config.json mutilated to one eos, generation_config.json intact:
     *    the union must recover all three. This is @woolcoxm's case. */
    { Cfg c; memset(&c,0,sizeof c);
      write_cfg(dir,"config.json","[100]");
      write_cfg(dir,"generation_config.json","[100,101,102]");
      load_cfg(&c,dir);
      if(c.n_stop!=3){ fprintf(stderr,"  FAIL union: expected 3 stops, got %d\n",c.n_stop); fail=1; }
      else printf("  config eos=[100] + generation_config=[100,101,102] -> 3 stops   ok\n"); }

    /* 3. no generation_config.json at all: must not crash, config alone wins */
    { Cfg c; memset(&c,0,sizeof c);
      write_cfg(dir,"config.json","[100,101,102]");
      rm_file(dir,"generation_config.json");
      load_cfg(&c,dir);
      if(c.n_stop!=3){ fprintf(stderr,"  FAIL no-genconfig: expected 3, got %d\n",c.n_stop); fail=1; }
      else printf("  generation_config.json absent -> config alone, no crash   ok\n"); }

    /* 4. BOTH configs mutilated: the tokenizer's special set must still stop the
     *    turn on every control token, and must still leave <think> alone. */
    { Cfg c; memset(&c,0,sizeof c);
      write_cfg(dir,"config.json","[100]");
      write_cfg(dir,"generation_config.json","[100]");
      load_cfg(&c,dir);
      stops_arm_tok(&c,-1,&T);
      fail|=expect("endoftext (config)",   100,1);
      fail|=expect("<|user|> (tokenizer)", 101,1);
      fail|=expect("<|observation|>",      102,1);
      fail|=expect("<|assistant|>",        103,1);
      fail|=expect("<sop>",                104,1);
      fail|=expect("<think> must NOT stop",110,0);
      fail|=expect("<tool_call> must NOT stop",111,0);
      fail|=expect("plain token must NOT stop",2,0);
      if(!fail) printf("  both configs mutilated -> tokenizer still stops all 5 control tokens   ok\n"); }

    /* 5. T=NULL (validation/oracle path): config stops only, unchanged behaviour */
    { Cfg c; memset(&c,0,sizeof c);
      write_cfg(dir,"config.json","[100]");
      rm_file(dir,"generation_config.json");
      load_cfg(&c,dir);
      stops_arm_tok(&c,-1,NULL);
      fail|=expect("T=NULL: config stop",100,1);
      fail|=expect("T=NULL: no tokenizer sweep",101,0);
      if(!fail) printf("  T=NULL -> config stops only (validation path untouched)   ok\n"); }

    rm_file(dir,"config.json"); rm_file(dir,"generation_config.json"); rm_file(dir,"tokenizer.json");
    rmdir(dir);
    if(fail){ printf("test_stops: FAIL\n"); return 1; }
    printf("test_stops: ok\n");
    return 0;
}
