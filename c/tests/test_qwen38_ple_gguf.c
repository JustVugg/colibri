/* GGUF PLE loading: an unscaled table is read unchanged; an external scale
 * must be refused by name, since q38_ple_row does not apply one. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#ifndef _WIN32
#include <sys/wait.h>
#endif

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

static int load_ple(void){
    Model model={0};Layer layer={0};
    model.c.hidden=2;model.c.hc_width=2;model.c.ple_dim=2;model.c.ple_convk=2;
    model.c.ngram_size=3;model.c.ngram_heads=2;model.c.ngram_head_dim=2;
    snprintf(model.prefix,sizeof model.prefix,"model");
    q38_load_ple(&model,&layer);
    CHECK(model.ple_weight_scale==1.f);
    CHECK(model.ple_gg==gguf_find(&g_q38gg,"per_layer_token_embd.weight"));
    CHECK(model.ple_part_count==1&&model.ple_part_start[1]==5);
    CHECK(model.ple_multipliers[2]==5&&model.ple_head_vocab[1]==3&&model.ple_head_offset[1]==2);
    float row[2];q38_ple_row(&model,0,row);
    CHECK(row[0]==1.f&&row[1]==-2.f);
    q38_ple_row(&model,4,row);CHECK(row[0]==9.f&&row[1]==-10.f);
    q38_weight_free(&layer.ple_key);q38_weight_free(&layer.ple_value);
    free(layer.ple_norm_key);free(layer.ple_norm_query);free(layer.ple_norm_conv);free(layer.ple_conv);
    return 0;
}

#ifndef _WIN32
static int expect_scale_refusal(void){
    int pipefd[2];CHECK(pipe(pipefd)==0);
    pid_t child=fork();CHECK(child>=0);
    if(!child){
        close(pipefd[0]);
        if(dup2(pipefd[1],2)<0)_exit(2);
        close(pipefd[1]);
        _exit(load_ple());
    }
    close(pipefd[1]);
    char error[1024];size_t used=0;ssize_t n;
    while(used<sizeof error-1&&(n=read(pipefd[0],error+used,sizeof error-1-used))>0)used+=(size_t)n;
    error[used]=0;close(pipefd[0]);
    int status=0;CHECK(waitpid(child,&status,0)==child);
    if(!WIFEXITED(status)||WEXITSTATUS(status)!=1||
       !strstr(error,"per_layer_token_embd.scale")||!strstr(error,"refusing")){
        fprintf(stderr,"expected per_layer_token_embd.scale refusal, status=%d, stderr: %s\n",status,error);
        return 1;
    }
    return 0;
}
#endif

int main(void){
    /* A synthetic GGUF index with real file-backed tensor reads. Only the
     * optional scale's index entry changes between the control and refusal. */
    gguf_tensor tensors[]={
        {.name="blk.0.ple_key.weight",.ne={2,2,1,1}},
        {.name="blk.0.ple_value.weight",.ne={2,2,1,1}},
        {.name="blk.0.ple_norm_key.weight",.ne={2,1,1,1}},
        {.name="blk.0.ple_norm_query.weight",.ne={2,1,1,1}},
        {.name="blk.0.ple_norm_conv.weight",.ne={2,1,1,1}},
        {.name="blk.0.ple_conv1d.weight",.ne={2,2,1,1}},
        {.name="per_layer_token_embd.weight",.ne={2,5,1,1}},
        {.name="per_layer_token_embd.scale",.ne={1,1,1,1}},
    };
    int64_t multipliers[]={1,3,5},vocab[]={2,3},offsets[]={0,2};
    gguf_kv metadata[]={
        {.key="qwen4exp.ple.layer_multipliers",.type=GGUF_T_ARRAY,.atype=GGUF_T_I64,.n=3,.arr=multipliers},
        {.key="qwen4exp.ple.head_vocab_sizes",.type=GGUF_T_ARRAY,.atype=GGUF_T_I64,.n=2,.arr=vocab},
        {.key="qwen4exp.ple.head_offsets",.type=GGUF_T_ARRAY,.atype=GGUF_T_I64,.n=2,.arr=offsets},
    };
    const char *path="test_qwen38_ple_gguf_tmp";
    FILE *file=fopen(path,"w+b");CHECK(file);
    int fd=fileno(file),index[32];
    for(int i=0;i<32;i++)index[i]=-1;
    g_q38gg=(gguf_ctx){.n_files=1,.fds=&fd,.n_kv=3,.kv=metadata,
                       .n_tensors=7,.t=tensors,.idx=index,.idx_cap=32};
    for(int i=0;i<8;i++){
        gguf_tensor *t=&tensors[i];t->type=GQ_F32;t->n_dims=2;
        t->off=(uint64_t)ftell(file);t->nbytes=t->ne[0]*t->ne[1]*4;
        for(int j=0;j<t->ne[0]*t->ne[1];j++){
            float value=i==6?(j%2?-(float)(j+1):(float)(j+1)):1.f;
            CHECK(fwrite(&value,sizeof value,1,file)==1);
        }
        if(i==7)continue;
        int slot=(int)(gguf_hash(t->name)%32);
        while(index[slot]>=0)slot=(slot+1)%32;
        index[slot]=i;
    }
    CHECK(fflush(file)==0);g_q38_gguf=1;
    int failed=load_ple();
#ifndef _WIN32
    /* Even an identity scale is unsupported: refuse its presence, without
     * guessing whether the table's type or the scale's value makes it safe. */
    int slot=(int)(gguf_hash(tensors[7].name)%32);
    while(index[slot]>=0)slot=(slot+1)%32;
    index[slot]=7;g_q38gg.n_tensors=8;
    failed|=expect_scale_refusal();
#else
    puts("test_qwen38_ple_gguf: scale refusal subtest skipped on Windows");
#endif
    g_q38_gguf=0;memset(&g_q38gg,0,sizeof g_q38gg);
    CHECK(fclose(file)==0);CHECK(remove(path)==0);
    if(failed)return 1;
    puts("test_qwen38_ple_gguf: unscaled PLE rows and named scale refusal: ok");
    return 0;
}
