/* GLM config must not select more routed experts than the model contains. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#ifndef _WIN32
#include <sys/wait.h>
#endif

typedef struct { int experts, topk, accept; } cfg_case;
static const cfg_case CASES[]={{1,1,1},{8,2,1},{1,2,0},{2,64,0}};

static int write_cfg(const char *dir,const cfg_case *tc){
    char path[512]; snprintf(path,sizeof(path),"%s/config.json",dir);
    FILE *f=fopen(path,"wb"); if(!f) return -1;
    fprintf(f,"{\"hidden_size\":64,\"num_hidden_layers\":2,\"num_attention_heads\":4,"
              "\"n_routed_experts\":%d,\"num_experts_per_tok\":%d,"
              "\"moe_intermediate_size\":32,\"intermediate_size\":64,"
              "\"first_k_dense_replace\":1,\"q_lora_rank\":0,\"kv_lora_rank\":16,"
              "\"qk_nope_head_dim\":8,\"qk_rope_head_dim\":8,\"v_head_dim\":8,"
              "\"n_shared_experts\":1,\"vocab_size\":200,\"n_group\":1,"
              "\"topk_group\":1,\"rope_theta\":10000.0}\n",tc->experts,tc->topk);
    return fclose(f);
}

static int child_case(int index,const char *dir){
    if(index<0||index>=(int)(sizeof(CASES)/sizeof(CASES[0]))) return 90;
    Cfg c; memset(&c,0,sizeof(c)); load_cfg(&c,dir);
    if(!CASES[index].accept) return 91;
    return c.n_experts==CASES[index].experts&&c.topk==CASES[index].topk?0:92;
}

static int child_status(const char *self,int index,const char *dir){
    char cmd[1536];
#ifdef _WIN32
    snprintf(cmd,sizeof(cmd),"call \"%s\" --child %d \"%s\" 2>NUL",self,index,dir);
#else
    snprintf(cmd,sizeof(cmd),"\"%s\" --child %d \"%s\" 2>/dev/null",self,index,dir);
#endif
    int status=system(cmd);
#ifdef _WIN32
    return status;
#else
    return status>=0&&WIFEXITED(status)?WEXITSTATUS(status):-1;
#endif
}

int main(int argc,char **argv){
    if(argc==4&&!strcmp(argv[1],"--child")) return child_case(atoi(argv[2]),argv[3]);
    for(int i=0;i<(int)(sizeof(CASES)/sizeof(CASES[0]));i++){
        char dir[]="test_cfg_topk_XXXXXX"; if(!mkdtemp(dir)){perror("mkdtemp");return 1;}
        if(write_cfg(dir,&CASES[i])) return 1;
        int got=child_status(argv[0],i,dir),want=CASES[i].accept?0:1;
        char path[512]; snprintf(path,sizeof(path),"%s/config.json",dir); remove(path); rmdir(dir);
        if(got!=want){fprintf(stderr,"case %d: E=%d K=%d exit=%d, expected %d\n",i,CASES[i].experts,CASES[i].topk,got,want);return 1;}
    }
    puts("config top-k bounds: ok"); return 0;
}
