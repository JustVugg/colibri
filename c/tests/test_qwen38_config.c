/* Qwen4-Exp configuration fidelity and hostile-geometry refusal. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

static const char *directory = "test_qwen38_config_tmp";
static const char *config_hc_count = "4";

static int write_config(const char *hidden_act,const char *gate,
                        const char *attention_bias,const char *tie,
                        const char *rope_type,const char *norm_topk,
                        const char *partial,const char *ple_layer,
                        const char *key_dim){
    char path[256];snprintf(path,sizeof path,"%s/config.json",directory);
    FILE *file=fopen(path,"wb");if(!file)return -1;
    char gate_field[96]="";
    if(gate)snprintf(gate_field,sizeof gate_field,"\"output_gate_type\":%s,",gate);
    int written=fprintf(file,
        "{\"model_type\":\"qwen4_exp_text\","
        "\"hidden_size\":8,\"num_hidden_layers\":2,\"vocab_size\":16,"
        "\"max_position_embeddings\":128,\"eos_token_id\":2,"
        "\"hidden_act\":%s,%s"
        "\"attention_bias\":%s,\"tie_word_embeddings\":%s,"
        "\"rms_norm_eps\":0.000001,"
        "\"rope_parameters\":{\"rope_type\":%s,\"rope_theta\":10000,"
        "\"partial_rotary_factor\":%s},"
        "\"hc_count\":%s,\"hc_lowrank\":4,"
        "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"head_dim\":6,"
        "\"indexer_n_heads\":1,\"indexer_kv_heads\":1,"
        "\"indexer_head_dim\":4,\"indexer_budget\":2,"
        "\"indexer_compress_ratio\":1,"
        "\"num_experts\":2,\"num_experts_per_tok\":1,"
        "\"moe_intermediate_size\":4,\"shared_expert_intermediate_size\":4,"
        "\"norm_topk_prob\":%s,"
        "\"linear_num_key_heads\":1,\"linear_num_value_heads\":1,"
        "\"linear_key_head_dim\":%s,\"linear_value_head_dim\":2,"
        "\"linear_conv_kernel_dim\":2,"
        "\"ple_embed_dim\":8,\"ple_conv_kernel_size\":2,"
        "\"ngram_size\":3,\"heads_per_ngram\":1,\"split_ngram_parts\":1,"
        "\"ple_layer_ids\":[%s],"
        "\"layer_types\":[\"linear_attention\",\"qwen_sparse_attention\"]}",
        hidden_act,gate_field,attention_bias,tie,rope_type,partial,
        config_hc_count,norm_topk,key_dim,ple_layer);
    return written<0||fclose(file)!=0?-1:0;
}

static int valid_config(void){
    return write_config("\"silu\"","\"sigmoid\"","false","false",
                        "\"default\"","true","0.49","1","2");
}

#ifndef _WIN32
static int expect_failure(const char *hidden_act,const char *gate,
                          const char *attention_bias,const char *tie,
                          const char *rope_type,const char *norm_topk,
                          const char *partial,const char *ple_layer,
                          const char *key_dim){
    CHECK(write_config(hidden_act,gate,attention_bias,tie,rope_type,norm_topk,
                       partial,ple_layer,key_dim)==0);
    pid_t child=fork();CHECK(child>=0);
    if(!child){
        FILE *sink=fopen("/dev/null","wb");if(sink)dup2(fileno(sink),2);
        Cfg config;q38_load_cfg(&config,directory);q38_validate_cfg(&config);
        free(config.is_attn);_exit(0);
    }
    int status=0;CHECK(waitpid(child,&status,0)==child);
    CHECK(WIFEXITED(status)&&WEXITSTATUS(status)!=0);return 0;
}

static int expect_reference_failure(const char *document){
    pid_t child=fork();CHECK(child>=0);
    if(!child){
        FILE *sink=fopen("/dev/null","wb");if(sink)dup2(fileno(sink),2);
        char *arena=NULL;jval *root=json_parse(document,&arena);
        (void)read_reference_logits(root,1);
        json_free(root);free(arena);_exit(0);
    }
    int status=0;CHECK(waitpid(child,&status,0)==child);
    CHECK(WIFEXITED(status)&&WEXITSTATUS(status)!=0);return 0;
}
#endif

int main(void){
#ifdef _WIN32
    CHECK(_mkdir(directory)==0||errno==EEXIST);
#else
    CHECK(mkdir(directory,0700)==0||errno==EEXIST);
#endif
    CHECK(valid_config()==0);
    Cfg config;q38_load_cfg(&config,directory);q38_validate_cfg(&config);
    CHECK(config.rotary_dim==2);free(config.is_attn);
    char *arena=NULL;jval *reference=json_parse(
        "{\"schema_version\":2,\"final_logits\":[0.0]}",&arena);
    CHECK(read_reference_logits(reference,1)!=NULL);
    json_free(reference);free(arena);

#ifndef _WIN32
    CHECK(expect_failure("\"gelu\"","\"sigmoid\"","false","false","\"default\"","true","0.49","1","2")==0);
    CHECK(expect_failure("\"silu\"","\"silu\"","false","false","\"default\"","true","0.49","1","2")==0);
    CHECK(expect_failure("\"silu\"",NULL,"false","false","\"default\"","true","0.49","1","2")==0);
    CHECK(expect_failure("\"silu\"","\"sigmoid\"","true","false","\"default\"","true","0.49","1","2")==0);
    CHECK(expect_failure("\"silu\"","\"sigmoid\"","false","true","\"default\"","true","0.49","1","2")==0);
    CHECK(expect_failure("\"silu\"","\"sigmoid\"","false","false","\"linear\"","true","0.49","1","2")==0);
    CHECK(expect_failure("\"silu\"","\"sigmoid\"","\"false\"","false","\"default\"","true","0.49","1","2")==0);
    CHECK(expect_failure("\"silu\"","\"sigmoid\"","false","false","\"default\"","\"true\"","0.49","1","2")==0);
    CHECK(write_config("\"silu\"","\"sigmoid\"","false","false",
                       "\"default\"","true","0.49","2","2")==0);
    q38_load_cfg(&config,directory);q38_validate_cfg(&config);
    CHECK(config.ple_layer==1&&config.is_attn[1]);free(config.is_attn);
    CHECK(expect_failure("\"silu\"","\"sigmoid\"","false","false","\"default\"","true","0.49","1","2147483647")==0);
    config_hc_count="1";
    CHECK(expect_failure("\"silu\"","\"sigmoid\"","false","false","\"default\"","true","0.49","1","2")==0);
    config_hc_count="4";
    CHECK(expect_reference_failure("{}")==0);
    CHECK(expect_reference_failure("{\"schema_version\":\"2\",\"final_logits\":[0]}")==0);
    CHECK(expect_reference_failure("{\"schema_version\":1.5,\"final_logits\":[0]}")==0);
    CHECK(expect_reference_failure("{\"schema_version\":3,\"final_logits\":[0]}")==0);
    CHECK(expect_reference_failure("{\"schema_version\":2}")==0);
#else
    puts("test_qwen38_config: hostile refusal subtests skipped on Windows");
#endif
    char path[256];snprintf(path,sizeof path,"%s/config.json",directory);CHECK(remove(path)==0);
#ifdef _WIN32
    CHECK(_rmdir(directory)==0);
#else
    CHECK(rmdir(directory)==0);
#endif
    puts("test_qwen38_config: strict variants, truncating RoPE, overflow refusal: ok");
    return 0;
}
