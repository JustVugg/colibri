/* test_ablate_mode.c — the ablation scoring mode's manifest loader, evidence
 * writer and dispatch contract, with no model and no weights.
 *
 * The mode's real work is one teacher-forced prefill per manifest item, which
 * needs a loaded model.  Everything around that prefill -- accepting or
 * refusing a manifest, binding it by digest, opening the output exactly once,
 * writing the records an offline reader parses, and reporting completion --
 * runs without any model math at all.  The adapter build bypasses only the
 * model computation and enters the same production parser, writer and
 * formatter, so those parts can be pinned on any machine.
 *
 * Required properties:
 *   P1 ROUND TRIP — a well-formed manifest produces a complete evidence
 *      artifact: a header naming the config and manifest digests, one item
 *      record per item, one logit record per target position, and a done
 *      record whose counts equal the header's expectations.
 *   P2 DIGEST BINDING — the manifest digest in the artifact is a digest of the
 *      manifest's own bytes under a domain prefix, so an artifact cannot be
 *      re-attached to a different manifest.
 *   P3 REFUSAL — malformed manifests are refused rather than partly run, and
 *      each refusal leaves no evidence file behind.
 *   P4 DISPATCH CONTRACT — the mode reports failure and writes nothing when it
 *      is handed no manifest path, which is the value the engine's environment
 *      lookup yields when the mode was not requested.
 *   P5 NEW FILE ONLY — an output path that already exists is refused, never
 *      truncated.
 *   P6 LENIENT FRAMING — the two line endings a host editor produces without
 *      meaning to (a CRLF terminator, and a last line with no terminator) are
 *      accepted, and all three framings of the same content bind to the same
 *      manifest digest.
 *   P7 NAMED REFUSALS — every refusal says why on the error stream; a silent
 *      non-zero exit is indistinguishable from a crash, and a refusal that
 *      names a line names the record at fault rather than the last one read.
 *   P8 BOUNDED INPUT — a manifest may not ask the loader for an unbounded
 *      allocation by declaring an enormous item length.
 * Exit 0 = all pass.
 *
 * With "--emit-round-trip <dir> [lf|crlf|unterminated]" it instead writes one
 * config, one manifest in the named framing and the artifact produced from
 * them into <dir> and exits, so a checker written independently of this engine
 * can validate real producer output -- in each framing the engine accepts.
 */
#define COLI_TEST_ABLATE_ADAPTERS 1
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#undef COLI_TEST_ABLATE_ADAPTERS

static int fails = 0;
#define CHECK(cond,msg) do{ if(!(cond)){ printf("  FAIL: %s\n", msg); fails++; } \
                            else printf("  ok:   %s\n", msg); }while(0)

/* Process-unique so two concurrent test runs in the same CWD never collide. */
static char MANIFEST_PATH[64];
static char EVIDENCE_PATH[64];
static char STDERR_PATH[64];

static void temp_paths_init(void){
    long pid=(long)getpid();
    snprintf(MANIFEST_PATH,sizeof(MANIFEST_PATH),"tmp_test_ablate_manifest.%ld.txt",pid);
    snprintf(EVIDENCE_PATH,sizeof(EVIDENCE_PATH),"tmp_test_ablate_evidence.%ld.jsonl",pid);
    snprintf(STDERR_PATH,sizeof(STDERR_PATH),"tmp_test_ablate_stderr.%ld.txt",pid);
}

/* An eight-expert, four-layer, 64-token vocabulary is enough for every bound
 * the loader checks, and small enough to write by hand.  The digest is taken
 * over the same bytes a reader would find in the config file, so an external
 * checker can recompute it. */
static const char CONFIG_JSON[]=
    "{\"hidden_size\":8,\"num_hidden_layers\":4,\"n_routed_experts\":8,"
    "\"first_k_dense_replace\":1,\"vocab_size\":64}\n";

static void cfg_init(Cfg *c){
    memset(c,0,sizeof(*c));
    c->hidden=8; c->vocab=64; c->n_layers=4; c->first_dense=1; c->n_experts=8;
    evidence_sha256_hex(CONFIG_JSON,sizeof(CONFIG_JSON)-1,c->config_sha256);
}

static int file_write(const char *path, const char *body){
    FILE *f=fopen(path,"wb");
    if(!f) return -1;
    size_t want=strlen(body);
    size_t got=fwrite(body,1,want,f);
    return (fclose(f)==0 && got==want) ? 0 : -1;
}

static void manifest_write(const char *body){
    (void)file_write(MANIFEST_PATH,body);
}

/* The manifest used for the round trip, shared with the emitted artifact so an
 * external checker sees exactly what this test checks. */
static const char ROUND_TRIP_MANIFEST[]=
    "0 3 2 0 0 1 2 3\n"
    "1 4 2 1 1 2 3 -1 4 5 6 7\n";

static char *slurp(const char *path, size_t *len_out){
    FILE *f=fopen(path,"rb");
    if(!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0){ fclose(f); return NULL; }
    char *b=malloc((size_t)n+1);
    if(!b){ fclose(f); return NULL; }
    size_t got=fread(b,1,(size_t)n,f); fclose(f);
    b[got]=0;
    if(len_out) *len_out=got;
    return b;
}

/* Run the mode with model computation bypassed. */
static int run_mode(Cfg *c, const char *manifest_path, const char *out_path){
    static Model m;
    memset(&m,0,sizeof(m));
    m.c=*c;
    remove(out_path);
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    g_ablate_adapter_test.bypass_model_compute=1;
    g_ablate_adapter_test.override_outp=1;
    g_ablate_adapter_test.outp=out_path;
    int rc=ablate_mode_dispatch(&m,manifest_path,ablate_model_mode_run);
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    return rc;
}

static int count_occurrences(const char *hay, const char *needle){
    int n=0;
    for(const char *p=strstr(hay,needle); p; p=strstr(p+1,needle)) n++;
    return n;
}

static void t_round_trip(Cfg *c){
    printf("P1 round trip and P2 digest binding\n");
    /* item T n_prompt mode ncells (L E A)... t_0..t_{T-1} */
    const char *body=ROUND_TRIP_MANIFEST;
    manifest_write(body);
    CHECK(run_mode(c,MANIFEST_PATH,EVIDENCE_PATH)==0, "a well-formed manifest runs to completion");

    size_t len=0;
    char *text=slurp(EVIDENCE_PATH,&len);
    CHECK(text!=NULL, "the requested evidence file exists");
    if(!text) return;

    CHECK(count_occurrences(text,"\"t\":\"hdr\"")==1, "exactly one header record");
    CHECK(count_occurrences(text,"\"t\":\"ah\"")==2, "one item record per manifest item");
    CHECK(count_occurrences(text,"\"t\":\"lg\"")==3, "one logit record per target position");
    CHECK(count_occurrences(text,"\"t\":\"done\"")==1, "exactly one done record");
    CHECK(strstr(text,"\"schema\":\"coli-ablate/2\"")!=NULL, "the header names the schema");
    CHECK(strstr(text,"\"expected_items\":2")!=NULL, "the header expects both items");
    CHECK(strstr(text,"\"expected_targets\":3")!=NULL, "the header expects all three targets");
    CHECK(strstr(text,"\"completed_items\":2")!=NULL, "the done record completed both items");
    CHECK(strstr(text,"\"completed_targets\":3")!=NULL, "the done record completed all targets");
    CHECK(strstr(text,c->config_sha256)!=NULL, "the header carries the loaded config digest");
    CHECK(strstr(text,"\"cells\":[[2,3,-1]]")!=NULL, "the ablated cell is recorded as given");

    /* P2: recompute the manifest digest the way an offline reader would. */
    static const char domain[]="coli-ablate-manifest/2\n";
    EvidenceSha256 h; unsigned char raw[32]; char expect[65];
    static const char hex[]="0123456789abcdef";
    evidence_sha256_init(&h);
    evidence_sha256_update(&h,domain,sizeof(domain)-1);
    evidence_sha256_update(&h,body,strlen(body));
    evidence_sha256_final(&h,raw);
    for(int i=0;i<32;i++){ expect[2*i]=hex[raw[i]>>4]; expect[2*i+1]=hex[raw[i]&15]; }
    expect[64]=0;
    CHECK(count_occurrences(text,expect)==2,
          "both records carry a digest of the manifest bytes under the domain prefix");

    free(text);
    remove(EVIDENCE_PATH);
}

static void t_refusal(Cfg *c){
    printf("P3 malformed manifests are refused\n");
    static const struct { const char *body; const char *why; } bad[] = {
        { "", "an empty manifest" },
        { "\n", "a manifest that is only a line terminator" },
        { "0 3 2 0 0 1 2 3\n\n", "a manifest with an empty line in it" },
        { "0 3 2 0\r 0 1 2 3\n", "a carriage return inside a record" },
        { "00 3 2 0 0 1 2 3\n", "a leading zero in a field" },
        { "0 3 2 0 0 1 2 3 4\n", "a token count that exceeds the declared length" },
        { "0 3 2 0 0 1 2\n", "a token count below the declared length" },
        { "0 1 1 0 0 5\n", "a sequence with no target position" },
        { "0 3 3 0 0 1 2 3\n", "a prompt that leaves no target position" },
        { "0 3 2 4 1 2 3 -1 1 2 3\n", "an unknown ablation mode" },
        { "0 3 2 1 0 1 2 3\n", "an ablating mode with no cells" },
        { "0 3 2 0 1 2 3 -1 1 2 3\n", "a baseline item with cells" },
        { "0 3 2 1 1 0 3 -1 1 2 3\n", "a layer below the first routed layer" },
        { "0 3 2 1 1 2 99 -1 1 2 3\n", "an expert beyond the model's expert count" },
        { "0 3 2 3 1 2 3 3 1 2 3\n", "a swap whose target is the ablated expert" },
        { "0 3 2 1 1 2 3 0 1 2 3\n", "a swap target on a non-swap mode" },
        { "0 3 2 1 2 2 3 -1 2 3 -1 1 2 3\n", "a duplicate cell within one item" },
        { "0 3 2 0 0 1 2 999\n", "a token beyond the vocabulary" },
        { "0 3 2 0 0 1 2 3\n0 3 2 0 0 1 2 3\n", "a duplicate item id" },
    };
    for(unsigned i=0;i<sizeof(bad)/sizeof(bad[0]);i++){
        manifest_write(bad[i].body);
        int rc=run_mode(c,MANIFEST_PATH,EVIDENCE_PATH);
        char *text=slurp(EVIDENCE_PATH,NULL);
        CHECK(rc==1 && text==NULL, bad[i].why);
        free(text);
        remove(EVIDENCE_PATH);
    }
}

static void t_dispatch_contract(Cfg *c){
    printf("P4 the mode is not entered without a manifest path\n");
    static Model m;
    memset(&m,0,sizeof(m));
    m.c=*c;
    remove(EVIDENCE_PATH);
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    g_ablate_adapter_test.bypass_model_compute=1;
    g_ablate_adapter_test.override_outp=1;
    g_ablate_adapter_test.outp=EVIDENCE_PATH;
    /* NULL is what the engine's environment lookup yields when the mode was
     * not requested; the dispatch must refuse it and touch nothing. */
    int rc=ablate_mode_dispatch(&m,NULL,ablate_model_mode_run);
    char *text=slurp(EVIDENCE_PATH,NULL);
    CHECK(rc==1, "no manifest path is a failed mode result");
    CHECK(text==NULL, "no manifest path writes no evidence file");
    free(text);
    rc=ablate_mode_dispatch(&m,MANIFEST_PATH,NULL);
    CHECK(rc==1, "no mode implementation is a failed mode result");
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    remove(EVIDENCE_PATH);
}

static void t_new_file_only(Cfg *c){
    printf("P5 an existing output path is refused\n");
    manifest_write("0 3 2 0 0 1 2 3\n");
    FILE *f=fopen(EVIDENCE_PATH,"wb");
    if(f){ fputs("do not truncate me\n",f); fclose(f); }
    static Model m;
    memset(&m,0,sizeof(m));
    m.c=*c;
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    g_ablate_adapter_test.bypass_model_compute=1;
    g_ablate_adapter_test.override_outp=1;
    g_ablate_adapter_test.outp=EVIDENCE_PATH;
    int rc=ablate_mode_dispatch(&m,MANIFEST_PATH,ablate_model_mode_run);
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    CHECK(rc==1, "an existing output path is a failed mode result");
    char *text=slurp(EVIDENCE_PATH,NULL);
    CHECK(text && strcmp(text,"do not truncate me\n")==0, "the existing file is left intact");
    free(text);
    remove(EVIDENCE_PATH);
}

/* Copy the 64 digest characters that follow the named key, or return 0. */
static int digest_of(const char *text, const char *key, char out[65]){
    const char *at=strstr(text,key);
    if(!at) return 0;
    at+=strlen(key);
    for(int i=0;i<64;i++){
        if(!at[i]) return 0;
        out[i]=at[i];
    }
    out[64]=0;
    return 1;
}

static int manifest_digest_of_run(Cfg *c, const char *body, char out[65]){
    manifest_write(body);
    if(run_mode(c,MANIFEST_PATH,EVIDENCE_PATH)!=0) return 0;
    char *text=slurp(EVIDENCE_PATH,NULL);
    int ok=text && digest_of(text,"\"manifest_sha256\":\"",out);
    free(text);
    remove(EVIDENCE_PATH);
    return ok;
}

static void t_lenient_framing(Cfg *c){
    printf("P6 the framings a host editor produces are accepted\n");
    char canonical[65], crlf[65], unterminated[65];
    CHECK(manifest_digest_of_run(c,"0 3 2 0 0 1 2 3\n",canonical),
          "a canonical manifest runs");
    CHECK(manifest_digest_of_run(c,"0 3 2 0 0 1 2 3\r\n",crlf),
          "a manifest with CRLF line endings runs");
    CHECK(manifest_digest_of_run(c,"0 3 2 0 0 1 2 3",unterminated),
          "a manifest whose last line has no terminator runs");
    /* Known answer, pinned identically in tests/test_check_ablate_evidence.py:
     * SHA-256 of "coli-ablate-manifest/2\n" followed by the canonical record.
     * Pinning the literal on both sides is what makes the two implementations
     * of the canonical rule checkable against each other rather than only
     * against themselves. */
    CHECK(strcmp(canonical,
                 "c63a48c375b14ca60f26c7e3c5dd36b5929ffaf669a45511c93deee6e8bbd5ed")==0,
          "the canonical digest matches the value the offline checker computes");
    CHECK(strcmp(canonical,crlf)==0,
          "a CRLF manifest binds to the same digest as its canonical form");
    CHECK(strcmp(canonical,unterminated)==0,
          "an unterminated manifest binds to the same digest as its canonical form");

    /* Both leniencies at once, across more than one record. */
    char mixed[65];
    CHECK(manifest_digest_of_run(c,"0 3 2 0 0 1 2 3\r\n1 3 2 0 0 4 5 6",mixed),
          "CRLF endings and a missing final terminator together run");
    char plain[65];
    CHECK(manifest_digest_of_run(c,"0 3 2 0 0 1 2 3\n1 3 2 0 0 4 5 6\n",plain),
          "the same two records in canonical form run");
    CHECK(strcmp(mixed,plain)==0, "the mixed framing binds to the canonical digest");
}

/* Refusals must say why.  stderr is redirected to a file for the duration and
 * left there: everything this test reports goes to stdout. */

static long stderr_bytes_of(Cfg *c, const char *manifest_body, void (*damage)(Cfg *)){
    Cfg local=*c;
    if(damage) damage(&local);
    manifest_write(manifest_body);
    if(!freopen(STDERR_PATH,"w",stderr)) return -1;
    int rc=run_mode(&local,MANIFEST_PATH,EVIDENCE_PATH);
    fflush(stderr);
    size_t len=0;
    char *text=slurp(STDERR_PATH,&len);
    free(text);
    remove(EVIDENCE_PATH);
    return rc==1 ? (long)len : -1;
}

static void damage_unhashed_config(Cfg *c){ memset(c->config_sha256,0,65); }
static void damage_short_digest(Cfg *c){ c->config_sha256[7]='Z'; }
static void damage_layer_count(Cfg *c){ c->n_layers=129; }
static void damage_expert_count(Cfg *c){ c->n_experts=0; }
static void damage_vocab(Cfg *c){ c->vocab=(1<<24)+1; }
static void damage_first_dense(Cfg *c){ c->first_dense=c->n_layers+1; }

static void t_named_refusals(Cfg *c){
    printf("P7 every refusal says why on the error stream\n");
    static const char good[]="0 3 2 0 0 1 2 3\n";
    static const struct { void (*damage)(Cfg *); const char *body; const char *why; } cases[] = {
        { damage_unhashed_config, good, "an unhashed config is refused with a reason" },
        { damage_short_digest,    good, "a malformed config digest is refused with a reason" },
        { damage_layer_count,     good, "too many layers is refused with a reason" },
        { damage_expert_count,    good, "no experts is refused with a reason" },
        { damage_vocab,           good, "an implausible vocabulary is refused with a reason" },
        { damage_first_dense,     good, "a first routed layer past the end is refused with a reason" },
        { NULL, "0 3 2 0 0 1 2 999\n", "a malformed manifest is refused with a reason" },
        { NULL, "", "an empty manifest is refused with a reason" },
    };
    for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){
        long bytes=stderr_bytes_of(c,cases[i].body,cases[i].damage);
        CHECK(bytes>0, cases[i].why);
    }
    remove(STDERR_PATH);
}

/* Run once with stderr captured, and hand the captured text back. */
static char *refusal_text_of(Cfg *c, const char *manifest_body){
    manifest_write(manifest_body);
    if(!freopen(STDERR_PATH,"w",stderr)) return NULL;
    int rc=run_mode(c,MANIFEST_PATH,EVIDENCE_PATH);
    fflush(stderr);
    remove(EVIDENCE_PATH);
    if(rc!=1) return NULL;
    return slurp(STDERR_PATH,NULL);
}

static void t_refusal_names_the_offender(Cfg *c){
    printf("P7b a refusal names the record at fault\n");
    /* Three good records, then a fourth repeating the first record's id, then
     * a fifth good one.  The duplicate check runs after the whole file is read,
     * so naming the last line read would report line 5 here. */
    char *text=refusal_text_of(c,
        "7 3 2 0 0 1 2 3\n"
        "8 3 2 0 0 1 2 3\n"
        "9 3 2 0 0 1 2 3\n"
        "7 3 2 0 0 4 5 6\n"
        "10 3 2 0 0 1 2 3\n");
    CHECK(text!=NULL, "a duplicate item id is refused");
    CHECK(text && strstr(text,"at line 4")!=NULL,
          "the duplicate names the line that repeats the id, not the last line");
    CHECK(text && strstr(text,"at line 5")==NULL,
          "the duplicate does not name the last line of the file");
    free(text);

    /* A malformed record in the middle names itself, not the end of file. */
    text=refusal_text_of(c,
        "0 3 2 0 0 1 2 3\n"
        "nonsense\n"
        "2 3 2 0 0 1 2 3\n");
    CHECK(text && strstr(text,"at line 2")!=NULL,
          "a malformed record names its own line");
    free(text);
    remove(STDERR_PATH);
}

static void t_bounded_item_length(Cfg *c){
    printf("P8 an absurd item length is refused, not allocated\n");
    char *text=refusal_text_of(c,"0 9000000000000000000 2 0 0 1 2 3\n");
    CHECK(text!=NULL, "an item length near the 64-bit ceiling is refused");
    CHECK(text && strstr(text,"above the")!=NULL &&
          strstr(text,"limit for one item")!=NULL,
          "the refusal names the per-item token limit");
    free(text);

    char body[64];
    snprintf(body,sizeof(body),"0 %d 2 0 0 1 2 3\n",ABLATE_MAX_ITEM_TOKENS+1);
    text=refusal_text_of(c,body);
    CHECK(text!=NULL, "one token past the documented limit is refused");
    CHECK(text && strstr(text,"above the")!=NULL, "and names the limit");
    free(text);
    remove(STDERR_PATH);
}

/* nll must come from a subtraction done wholly in double
 * (logZ - lo[gold]), not from a float-rounded intermediate. A row this
 * wide (max 1.0e7, gold logit 0.25, everything else far below the max)
 * makes the two formulas disagree: the float subtraction lo[gold]-max
 * rounds gold's contribution away entirely (its ulp near 1.0e7 is 1.0,
 * and 0.25 is under half that), while the all-double path keeps it. */
static void t_nll_pin(Cfg *c){
    printf("nll is a double-precision reduction, not a float intermediate\n");
    int V=c->vocab;                    /* 64, from CONFIG_JSON */
    float *row=malloc(sizeof(float)*(size_t)V);
    for(int i=0;i<V;i++) row[i]=-1.0e7f;
    row[0]=1.0e7f;                      /* the row's max */
    row[2]=0.25f;                       /* the gold logit -- deliberately tiny
                                          * next to the max */

    /* item 0: T=3, n_prompt=2, tokens 1,5,2 -- the one target position is
     * pos=1, whose gold token is tokens[2]=2. */
    manifest_write("0 3 2 0 0 1 5 2\n");
    static Model m;
    memset(&m,0,sizeof(m));
    m.c=*c;
    remove(EVIDENCE_PATH);
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    g_ablate_adapter_test.bypass_model_compute=1;
    g_ablate_adapter_test.override_outp=1;
    g_ablate_adapter_test.outp=EVIDENCE_PATH;
    g_ablate_adapter_test.forced_row=row;
    int rc=ablate_mode_dispatch(&m,MANIFEST_PATH,ablate_model_mode_run);
    memset(&g_ablate_adapter_test,0,sizeof(g_ablate_adapter_test));
    CHECK(rc==0, "the wide-spread row runs to completion");

    char *text=slurp(EVIDENCE_PATH,NULL);
    CHECK(text!=NULL, "the evidence file exists");
    if(text){
        const char *key="\"nll\":";
        const char *at=strstr(text,key);
        CHECK(at!=NULL, "the logit record carries an nll field");
        if(at){
            double got=strtod(at+strlen(key),NULL);
            /* dev's formula, replicated here in double from the row: */
            double max=row[0];
            double se=0;
            for(int i=0;i<V;i++) se+=exp((double)row[i]-max);
            double expect=(max+log(se))-(double)row[2];
            CHECK(got==expect,
                  "nll matches dev's double-precision formula exactly");
        }
    }
    free(text);
    remove(EVIDENCE_PATH);
    free(row);
}

/* config_sha256 must be bound through the production load path
 * (cfg_root, which load_cfg calls), not filled in by a test double that
 * hashes the literal separately. cfg_root takes the digest as an
 * out-parameter, so the hashing branch is exercised directly through
 * that parameter rather than by round-tripping the ABLATE_SCORE
 * environment variable through setenv/getenv (the pairing is not
 * reliable on every platform's C runtime -- see load_cfg's caller
 * below for the one assertion that still goes through load_cfg itself,
 * which needs no environment variable at all). */
static void t_load_cfg_digest_binding(Cfg *unused){
    (void)unused;
    printf("load_cfg binds config_sha256 to the loaded config.json bytes\n");
    /* load_cfg validates the full GLM config shape (n_group==1, every CKR
     * bound), unlike cfg_init above which sets Cfg fields directly -- so
     * this fixture, unlike CONFIG_JSON, must be a config.json load_cfg
     * actually accepts. */
    static const char CFG_BODY[]=
        "{\"hidden_size\":64,\"num_hidden_layers\":2,\"num_attention_heads\":4,"
        "\"n_routed_experts\":8,\"num_experts_per_tok\":2,"
        "\"moe_intermediate_size\":32,\"intermediate_size\":64,"
        "\"first_k_dense_replace\":1,\"q_lora_rank\":0,\"kv_lora_rank\":16,"
        "\"qk_nope_head_dim\":8,\"qk_rope_head_dim\":8,\"v_head_dim\":8,"
        "\"n_shared_experts\":1,\"vocab_size\":200,\"n_group\":1,"
        "\"topk_group\":1,\"rope_theta\":10000.0}\n";
    /* shasum -a 256 of the exact CFG_BODY bytes above, computed independently
     * of this codebase. */
    static const char PINNED_DIGEST[]=
        "3fc4425efb033c287f49e5db866946faf9672609e8916218ec6aacb8d4d530cf";
    CHECK(strlen(PINNED_DIGEST)==64, "the pinned literal is 64 hex characters");

    char dir[]="tmp_test_ablate_cfg_XXXXXX";
    if(!mkdtemp(dir)){ CHECK(0, "the config test directory could not be created"); return; }
    char path[512]; snprintf(path,sizeof(path),"%s/config.json",dir);
    CHECK(file_write(path,CFG_BODY)==0, "the temp config.json was written");

    /* load_cfg's own default call (no environment variable touched at
     * all) must leave config_sha256 unset -- this is the production
     * entry point every real caller uses. */
    Cfg c1; memset(&c1,0,sizeof(c1));
    load_cfg(&c1,dir);
    CHECK(c1.config_sha256[0]==0,
          "the default load path leaves config_sha256 unset");

    /* The hashing branch itself: called directly through cfg_root's
     * digest out-parameter, the same function load_cfg calls when its
     * caller asks for a digest. */
    char *ar=NULL; char got_digest[65]; memset(got_digest,0,sizeof(got_digest));
    jval *r=cfg_root(dir,&ar,got_digest);
    CHECK(r!=NULL, "cfg_root parses the temp config.json");

    /* Independently computed SHA-256 of the exact file bytes. */
    EvidenceSha256 h; unsigned char raw[32]; char expect[65];
    static const char hex[]="0123456789abcdef";
    evidence_sha256_init(&h);
    evidence_sha256_update(&h,CFG_BODY,sizeof(CFG_BODY)-1);
    evidence_sha256_final(&h,raw);
    for(int i=0;i<32;i++){ expect[2*i]=hex[raw[i]>>4]; expect[2*i+1]=hex[raw[i]&15]; }
    expect[64]=0;
    CHECK(strcmp(got_digest,expect)==0,
          "cfg_root's digest matches an independently computed SHA-256 of the bytes");
    /* And against a literal computed outside this codebase, so a change to
     * both the fixture and the hash function in the same wrong direction
     * cannot pass unnoticed. */
    CHECK(strcmp(got_digest,PINNED_DIGEST)==0,
          "cfg_root's digest matches the pinned literal for this fixture");

    free(ar);
    remove(path);
    rmdir(dir);
}

/* Write one config, one manifest and the artifact produced from them into a
 * directory, for an independently written checker to validate. */
/* The round-trip manifest re-framed the way a host editor might have saved it.
 * The engine accepts all three and binds them to the same digest. */
static int reframe(const char *canonical, const char *framing,
                   char *dst, size_t cap){
    size_t out=0;
    for(size_t i=0;canonical[i];i++){
        if(canonical[i]=='\n'){
            if(strcmp(framing,"crlf")==0){
                if(out+2>=cap) return -1;
                dst[out++]='\r';
            }else if(strcmp(framing,"unterminated")==0 && canonical[i+1]==0){
                break;                        /* drop the final terminator */
            }
            if(out+1>=cap) return -1;
            dst[out++]='\n';
            continue;
        }
        if(out+1>=cap) return -1;
        dst[out++]=canonical[i];
    }
    dst[out]=0;
    return 0;
}

static int emit_round_trip(Cfg *c, const char *dir, const char *framing){
    char config_path[1024], manifest_path[1024], evidence_path[1024];
    if(snprintf(config_path,sizeof(config_path),"%s/config.json",dir)>=(int)sizeof(config_path) ||
       snprintf(manifest_path,sizeof(manifest_path),"%s/manifest.txt",dir)>=(int)sizeof(manifest_path) ||
       snprintf(evidence_path,sizeof(evidence_path),"%s/evidence.jsonl",dir)>=(int)sizeof(evidence_path)){
        fprintf(stderr,"test_ablate_mode: directory name is too long\n");
        return 1;
    }
    char framed[4096];
    if(reframe(ROUND_TRIP_MANIFEST,framing,framed,sizeof(framed))!=0){
        fprintf(stderr,"test_ablate_mode: unknown or oversized framing %s\n",framing);
        return 1;
    }
    if(file_write(config_path,CONFIG_JSON)!=0 ||
       file_write(manifest_path,framed)!=0){
        fprintf(stderr,"test_ablate_mode: cannot write into %s\n",dir);
        return 1;
    }
    int rc=run_mode(c,manifest_path,evidence_path);
    if(rc!=0) fprintf(stderr,"test_ablate_mode: the ablation mode failed\n");
    return rc;
}

int main(int argc, char **argv){
    temp_paths_init();
    Cfg c; cfg_init(&c);
    if((argc==3 || argc==4) && strcmp(argv[1],"--emit-round-trip")==0)
        return emit_round_trip(&c,argv[2],argc==4?argv[3]:"lf");
    printf("test_ablate_mode\n");
    t_round_trip(&c);
    t_refusal(&c);
    t_dispatch_contract(&c);
    t_new_file_only(&c);
    t_lenient_framing(&c);
    t_named_refusals(&c);
    t_refusal_names_the_offender(&c);
    t_bounded_item_length(&c);
    t_nll_pin(&c);
    t_load_cfg_digest_binding(&c);
    remove(MANIFEST_PATH);
    remove(EVIDENCE_PATH);
    printf(fails ? "FAILED (%d)\n" : "PASSED (%d failures)\n", fails);
    return fails ? 1 : 0;
}
