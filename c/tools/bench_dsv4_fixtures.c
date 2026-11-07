/* Diagnostic C1 fixture runner; production engine, no numerical changes. */
#define COLI_V4_UNIT_GENERATE_STATS
#define COLI_V4_SKIP_GENERATE_MAIN
#include "../deepseek_v4.c"
#ifdef COLI_V4_GPU_TIER
#include "../backend_cuda_dsv4.h"
#endif

typedef struct { int ids[512], count; } Tokens;
static int record(void *data, int token, float logit, int position, int ordinal) {
    (void)logit; (void)position; (void)ordinal;
    Tokens *t = data;
    if (t->count == 512) return 1;
    t->ids[t->count++] = token;
    return 0;
}
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    int maximum = atoi(argv[3]), rounds = atoi(argv[4]);
    const char *modes = argv[5];
    if (maximum < 1 || maximum > 512 || rounds < 1 || rounds > 4 ||
        !*modes || strlen(modes) > 3 || strspn(modes,"td") != strlen(modes)) return 2;
    FILE *f = fopen(argv[2],"rb");
    if (!f) return 2;
    char data[65536]; size_t bytes = fread(data,1,sizeof(data)-1,f); fclose(f);
    data[bytes] = 0;
    char *prompts[6]; size_t lengths[6]; int count=0;
    for (size_t offset=0; offset<bytes;) {
        if (count==6) return 2;
        prompts[count]=data+offset; lengths[count]=strlen(data+offset);
        if (!lengths[count]) return 2;
        offset += lengths[count++]+1;
    }
    if (count!=6) return 2;
    setenv("V4_PREFIX_CKPT","0",1); setenv("V4_PREFIX_CKPT_DISK","0",1);
    setenv("COLI_V4_SAVE_USAGE","0",1); unsetenv("DSV4_CUDA_HEAD_BATCH");
    char error[512]={0}; ColiV4Engine *engine=NULL;
    ColiV4EngineOpenOptions options={.target_model_dir=argv[1],
        .memory_limit_bytes=48ULL<<30,.context_tokens=512,.pin_slots_per_layer=-1};
    if (coli_v4_engine_open(&engine,&options,error,sizeof(error))) {
        fprintf(stderr,"open: %s\n",error); return 1;
    }
    Tokens refs[2][6]={0}; int result=0;
    for (int round=0; !result && round<rounds; round++) {
        for (size_t m=0; !result && m<strlen(modes); m++) {
            char mode=modes[round%2 ? strlen(modes)-1-m : m];
            double batch_start=spec_now(); int total=0;
            for (int i=0; !result && i<count; i++) {
                for (int k=0;k<V4_DSPARK_BLOCK;k++) g_v4ds_core.accept_ewma[k]=0.5;
                ColiV4Session *session=NULL; Tokens tokens={0};
                ColiV4SessionCreateOptions create={.max_prompt_tokens=512,.max_new_tokens_cap=maximum};
                ColiV4SessionGenerateOptions generate={.max_new_tokens=maximum,.no_dspark=mode=='t'};
                ColiV4SessionGenerateStats stats={0};
                double start=spec_now();
                fprintf(stderr,"fixture round=%d mode=%c prompt=%d start\n",round,mode,i); fflush(stderr);
                result=coli_v4_session_create(&session,engine,&create,error,sizeof(error));
#ifdef COLI_V4_GPU_TIER
                int profile=getenv("BENCH_PROFILE") && round==1 && mode=='d' && i==0;
                if(profile)dsv4_cuda_profiler_start();
#endif
                if (!result) result=coli_v4_session_generate(session,prompts[i],lengths[i],&generate,record,&tokens,&stats,error,sizeof(error));
#ifdef COLI_V4_GPU_TIER
                if(profile)dsv4_cuda_profiler_stop();
#endif
                double wall=spec_now()-start;
                Tokens *ref=&refs[mode=='t' ? 0 : 1][i];
                if (!ref->count) *ref=tokens;
                int exact=tokens.count==ref->count && !memcmp(tokens.ids,ref->ids,(size_t)tokens.count*sizeof(int));
                int target_exact=tokens.count==refs[0][i].count && !memcmp(tokens.ids,refs[0][i].ids,(size_t)tokens.count*sizeof(int));
                printf("{\"round\":%d,\"mode\":\"%c\",\"fixture\":%d,\"prompt\":%d,\"generated\":%d,\"decode_s\":%.9f,\"ttft_s\":%.9f,\"wall_s\":%.9f,\"drafted\":%llu,\"accepted\":%llu,\"eos\":%d,\"exact\":%s,\"exact_to_target\":%s,\"result\":%d,\"ids\":[",round,mode,i,stats.prompt_tokens,stats.generated_tokens,stats.decode_sec,stats.time_to_first_token_sec,wall,(unsigned long long)stats.speculative_drafted,(unsigned long long)stats.speculative_accepted,stats.eos_stopped,exact?"true":"false",target_exact?"true":"false",result);
                for(int k=0;k<tokens.count;k++) printf("%s%d",k?",":"",tokens.ids[k]);
                puts("]}"); fflush(stdout); total+=stats.generated_tokens;
                coli_v4_session_destroy(session);
                if (!exact || (refs[0][i].count && !target_exact) || tokens.count!=maximum || stats.eos_stopped) result=-1;
                if(result) fprintf(stderr,"failed: %s exact=%d\n",error,exact);
            }
            fprintf(stderr,"fixture_summary round=%d mode=%c tokens=%d wall=%.9f result=%d\n",round,mode,total,spec_now()-batch_start,result);
        }
    }
    coli_v4_engine_destroy(engine); return result?1:0;
}
