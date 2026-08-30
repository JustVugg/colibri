/* Qwen3.8 native resident/expert weight dispatch.  Besides arithmetic parity,
 * this writes a real per-expert block-FP8 safetensors container and exercises
 * the production loader and capacity-one LRU in both native and expanded
 * modes. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#define COLI_SEGMENT_ADAPTER
#include <pthread.h>
#include "../qwen38.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

typedef struct {
    char name[192];
    const char *dtype;
    int rows,cols;
    int64_t begin,end;
    unsigned char raw[24];
    size_t raw_bytes;
    float scale;
    int expert,projection,is_scale;
} FixtureTensor;

static uint16_t to_bf16(float value){
    union { float f; uint32_t u; } bits={value};return (uint16_t)(bits.u>>16);
}

/* Independent formulation of quant.h's documented native-FP8 reduction:
 * F32 accumulation inside each 128-column block and F64 accumulation after
 * applying each block scale.  Do not call matmul_fp8 here—the dispatch test
 * must remain capable of detecting a defect in that production primitive. */
static void reference_fp8_matmul(float *out,const float *input,
                                 const uint8_t *weight,const float *scales,
                                 int sequences,int inputs,int outputs){
    int64_t input_blocks=fp8_nblk(inputs);
    for(int output=0;output<outputs;output++){
        for(int sequence=0;sequence<sequences;sequence++){
            double total=0.0;
            for(int64_t block=0;block<input_blocks;block++){
                int first=(int)(block*FP8_BLOCK);
                int last=first+FP8_BLOCK<inputs?first+FP8_BLOCK:inputs;
                float subtotal=0.0f;
                for(int input_index=first;input_index<last;input_index++)
                    subtotal+=e4m3_decode(weight[(int64_t)output*inputs+input_index])*
                              input[(int64_t)sequence*inputs+input_index];
                total+=(double)subtotal*
                       (double)scales[(output/FP8_BLOCK)*input_blocks+block];
            }
            out[(int64_t)sequence*outputs+output]=(float)total;
        }
    }
}

static int add_fixture_tensor(FixtureTensor *tensors,int *count,int64_t *offset,
                              const char *name,const char *dtype,int rows,int cols,
                              int expert,int projection,int is_scale){
    if(*count>=12)return -1;
    FixtureTensor *tensor=&tensors[(*count)++];memset(tensor,0,sizeof(*tensor));
    int length=snprintf(tensor->name,sizeof tensor->name,"%s",name);
    if(length<0||(size_t)length>=sizeof tensor->name)return -1;
    tensor->dtype=dtype;tensor->rows=rows;tensor->cols=cols;
    tensor->begin=*offset;tensor->expert=expert;tensor->projection=projection;
    tensor->is_scale=is_scale;
    if(is_scale){
        tensor->scale=0.125f*(float)(1+expert*3+projection);
        *offset+=!strcmp(dtype,"F32")?(int64_t)sizeof(float):
                                      (int64_t)sizeof(uint16_t);
    }else if(!strcmp(dtype,"BF16")){
        tensor->raw_bytes=(size_t)rows*cols*sizeof(uint16_t);
        for(int index=0;index<rows*cols;index++){
            uint16_t value=to_bf16(0.0625f*(float)(1+expert*7+projection*3+index));
            memcpy(tensor->raw+(size_t)index*sizeof(value),&value,sizeof(value));
        }
        *offset+=(int64_t)tensor->raw_bytes;
    }else{
        tensor->raw_bytes=(size_t)rows*cols;
        for(int index=0;index<rows*cols;index++)
            tensor->raw[index]=(unsigned char)(0x20+expert*0x18+projection*7+index);
        *offset+=(int64_t)tensor->raw_bytes;
    }
    tensor->end=*offset;return 0;
}

static int write_fp8_fixture(const char *directory,int mixed,int adjacent){
    FixtureTensor tensors[12];int count=0;int64_t offset=0;
    const char *projection[3]={"gate_proj","up_proj","down_proj"};
    for(int expert=0;expert<2;expert++)for(int kind=0;kind<3;kind++){
        int rows=kind==2?2:3,cols=kind==2?3:2;char name[192];
        int length=snprintf(name,sizeof name,
            "model.layers.0.mlp.experts.%d.%s.weight",expert,projection[kind]);
        const char *dtype=mixed&&expert==1&&kind==0?"BF16":"F8_E4M3";
        if(length<0||(size_t)length>=sizeof name||
           add_fixture_tensor(tensors,&count,&offset,name,dtype,rows,cols,
                              expert,kind,0))return -1;
        if(!strcmp(dtype,"BF16"))continue;
        length=snprintf(name,sizeof name,
            "model.layers.0.mlp.experts.%d.%s.weight_scale_inv",expert,
            projection[kind]);
        if(length<0||(size_t)length>=sizeof name||
           add_fixture_tensor(tensors,&count,&offset,name,adjacent?"BF16":"F32",1,1,
                              expert,kind,1))return -1;
    }

    int order[12];for(int index=0;index<count;index++)order[index]=index;
    if(adjacent){
        /* Match the official packing invariant while deliberately keeping the
         * JSON entries in construction order: compact gate/up sidecars, compact
         * down sidecars, then gate+up payload pairs.  The production loader
         * must use offsets and numeric expert ids, not header order. */
        for(int left=1;left<count;left++){
            int value=order[left],right=left-1;
            FixtureTensor *v=&tensors[value];
            int vcat=v->is_scale?(v->projection<2?0:1):2;
            int vkey=vcat*100+v->expert*3+v->projection;
            while(right>=0){
                FixtureTensor *r=&tensors[order[right]];
                int rcat=r->is_scale?(r->projection<2?0:1):2;
                int rkey=rcat*100+r->expert*3+r->projection;
                if(rkey<=vkey)break;
                order[right+1]=order[right];right--;
            }
            order[right+1]=value;
        }
        offset=0;
        for(int physical=0;physical<count;physical++){
            FixtureTensor *tensor=&tensors[order[physical]];
            int64_t bytes=tensor->is_scale?
                (!strcmp(tensor->dtype,"F32")?(int64_t)sizeof(float):
                                                (int64_t)sizeof(uint16_t)):
                (int64_t)tensor->raw_bytes;
            tensor->begin=offset;tensor->end=offset+bytes;offset+=bytes;
        }
    }

    char header[8192];size_t used=0;header[used++]='{';
    for(int index=0;index<count;index++){
        FixtureTensor *tensor=&tensors[index];
        int length=snprintf(header+used,sizeof header-used,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],"
            "\"data_offsets\":[%lld,%lld]}",index?",":"",tensor->name,
            tensor->dtype,tensor->rows,tensor->cols,(long long)tensor->begin,
            (long long)tensor->end);
        if(length<0||(size_t)length>=sizeof header-used)return -1;
        used+=(size_t)length;
    }
    if(used+1>=sizeof header)return -1;header[used++]='}';

    char path[512];int path_length=snprintf(path,sizeof path,
                                           "%s/model.safetensors",directory);
    if(path_length<0||(size_t)path_length>=sizeof path)return -1;
    FILE *file=fopen(path,"wb");if(!file)return -1;
    uint64_t header_length=used;
    int failed=fwrite(&header_length,sizeof header_length,1,file)!=1||
               fwrite(header,1,used,file)!=used;
    for(int physical=0;physical<count&&!failed;physical++){
        FixtureTensor *tensor=&tensors[order[physical]];
        if(tensor->is_scale&&!strcmp(tensor->dtype,"F32"))
            failed=fwrite(&tensor->scale,sizeof tensor->scale,1,file)!=1;
        else if(tensor->is_scale){
            uint16_t scale=to_bf16(tensor->scale);
            failed=fwrite(&scale,sizeof scale,1,file)!=1;
        }
        else
            failed=fwrite(tensor->raw,1,tensor->raw_bytes,file)!=tensor->raw_bytes;
    }
    if(fclose(file))failed=1;return failed?-1:0;
}

static int init_fixture_model(Model *model,const char *directory,int native_fp8){
    memset(model,0,sizeof(*model));model->c.hidden=2;model->c.inter=3;
    model->c.experts=2;model->c.topk=2;model->c.shared_inter=2;
    model->c.layers=1;model->range_begin=0;model->range_end=1;
    model->native_fp8=native_fp8;model->native_bf16=1;model->prefill_batch=1;
    snprintf(model->prefix,sizeof model->prefix,"model");
    st_init(&model->S,directory);
    model->cache=(LCache*)calloc(1,sizeof(*model->cache));
    if(!model->cache){st_destroy(&model->S);return -1;}
    LCache *cache=&model->cache[0];cache->cap=1;
    cache->slots=(Slot*)calloc(1,sizeof(*cache->slots));
    cache->by_expert=(int*)malloc(2*sizeof(*cache->by_expert));
    if(!cache->slots||!cache->by_expert){
        free(cache->slots);free(cache->by_expert);free(model->cache);
        st_destroy(&model->S);memset(model,0,sizeof(*model));return -1;
    }
    cache->by_expert[0]=cache->by_expert[1]=-1;return 0;
}

static void destroy_fixture_model(Model *model){
    if(model->cache){
        for(int slot=0;slot<model->cache[0].n;slot++){
            q38_weight_free(&model->cache[0].slots[slot].gate);
            q38_weight_free(&model->cache[0].slots[slot].up);
            q38_weight_free(&model->cache[0].slots[slot].down);
            free(model->cache[0].slots[slot].fp8_slab);
        }
        free(model->cache[0].slots);free(model->cache[0].by_expert);
    }
    if(model->expert_scales)free(model->expert_scales[0].values);
    free(model->expert_scales);free(model->cache);st_destroy(&model->S);
    memset(model,0,sizeof(*model));
}

static int init_fixture_moe_layer(Model *model){
    static const float router[4]={1.0f,-1.0f,-1.0f,1.0f};
    static const float shared_gate[4]={0.25f,-0.5f,0.75f,0.125f};
    static const float shared_up[4]={-0.25f,0.5f,0.375f,0.625f};
    static const float shared_down[4]={0.5f,-0.125f,0.25f,0.75f};
    static const float gate[2]={0.2f,-0.35f};
    Layer *layer=(Layer*)calloc(1,sizeof(*layer));
    if(!layer)return -1;
    q38_weight_reserve(&layer->router,Q38_WEIGHT_F32,2,2);
    q38_weight_reserve(&layer->sh_g,Q38_WEIGHT_F32,2,2);
    q38_weight_reserve(&layer->sh_u,Q38_WEIGHT_F32,2,2);
    q38_weight_reserve(&layer->sh_d,Q38_WEIGHT_F32,2,2);
    memcpy(layer->router.data,router,sizeof router);
    memcpy(layer->sh_g.data,shared_gate,sizeof shared_gate);
    memcpy(layer->sh_u.data,shared_up,sizeof shared_up);
    memcpy(layer->sh_d.data,shared_down,sizeof shared_down);
    layer->sh_gate=(float*)malloc(sizeof gate);
    if(!layer->sh_gate){
        q38_weight_free(&layer->router);q38_weight_free(&layer->sh_g);
        q38_weight_free(&layer->sh_u);q38_weight_free(&layer->sh_d);
        free(layer);return -1;
    }
    memcpy(layer->sh_gate,gate,sizeof gate);model->L=layer;
    return 0;
}

static void destroy_fixture_moe_layer(Model *model){
    if(!model->L)return;
    q38_weight_free(&model->L[0].router);q38_weight_free(&model->L[0].sh_g);
    q38_weight_free(&model->L[0].sh_u);q38_weight_free(&model->L[0].sh_d);
    free(model->L[0].sh_gate);free(model->L);model->L=NULL;
}

static uint8_t fixture_fp8_byte(int expert,int projection,int index){
    return (uint8_t)(0x20+expert*0x18+projection*7+index);
}

static float fixture_fp8_scale(int expert,int projection){
    return 0.125f*(float)(1+expert*3+projection);
}

static int verify_loaded_fp8_weight(const Q38Weight *weight,int expert,
                                    int projection,int native_fp8){
    int rows=projection==2?2:3,cols=projection==2?3:2;
    if(!weight||weight->rows!=rows||weight->cols!=cols||
       weight->kind!=(native_fp8?Q38_WEIGHT_FP8:Q38_WEIGHT_F32))return -1;
    float scale=fixture_fp8_scale(expert,projection);
    for(int index=0;index<rows*cols;index++){
        uint8_t raw=fixture_fp8_byte(expert,projection,index);
        if(native_fp8){
            if(((const uint8_t*)weight->data)[index]!=raw||
               weight->scale_count!=1||weight->scales[0]!=scale)return -1;
        }else if(((const float*)weight->data)[index]!=e4m3_decode(raw)*scale)
            return -1;
    }
    float input[3]={0.75f,-0.5f,0.25f},got[3]={0},want[3]={0};
    for(int row=0;row<rows;row++){
        float subtotal=0.0f;
        for(int col=0;col<cols;col++){
            float value=e4m3_decode(fixture_fp8_byte(
                expert,projection,row*cols+col));
            if(!native_fp8)value*=scale;
            subtotal+=value*input[col];
        }
        want[row]=native_fp8?(float)((double)subtotal*(double)scale):subtotal;
    }
    q38_weight_matmul(got,input,weight,1,cols,rows);
    return memcmp(got,want,(size_t)rows*sizeof(float))?-1:0;
}

static int verify_loaded_fp8_slot(const Slot *slot,int expert,int native_fp8){
    return verify_loaded_fp8_weight(&slot->gate,expert,0,native_fp8)||
           verify_loaded_fp8_weight(&slot->up,expert,1,native_fp8)||
           verify_loaded_fp8_weight(&slot->down,expert,2,native_fp8)?-1:0;
}

static int check_fixture_mode(const char *directory,int native_fp8,int expect_fast){
    Model model;if(init_fixture_model(&model,directory,native_fp8))return -1;
    int result=-1;
    uint64_t bytes_per_capacity=0,fixed_scale_bytes=0;unsigned numeric_kinds=0;
    if(q38_segment_expert_layout(&model,0,1,&bytes_per_capacity,
                                 &fixed_scale_bytes,&numeric_kinds))
        goto cleanup;
    char numeric_class[96];
    q38_segment_numeric_class(numeric_class,sizeof numeric_class,numeric_kinds);
    if(native_fp8){
        if(bytes_per_capacity!=18||fixed_scale_bytes!=24||
           numeric_kinds!=Q38_EXPERT_FP8_BLOCK||
           strcmp(numeric_class,
                  "qwen38/fp8-block-f32dot-f64blocksum/cpu-v1"))goto cleanup;
    }else if(bytes_per_capacity!=72||fixed_scale_bytes||
              numeric_kinds!=Q38_EXPERT_FP8_EXPANDED||
              strcmp(numeric_class,
                     "qwen38/fp8-expanded-f32dot/cpu-v1"))goto cleanup;
    int capacity=0;
    if(q38_segment_cache_capacity(bytes_per_capacity,fixed_scale_bytes,2,
                                  fixed_scale_bytes+bytes_per_capacity*2,
                                  &capacity)||capacity!=2||
       !q38_segment_cache_capacity(bytes_per_capacity,fixed_scale_bytes,2,
                                   fixed_scale_bytes+bytes_per_capacity-1,
                                   &capacity))goto cleanup;
    if(q38_segment_cache_resize(&model,2)||model.cache[0].cap!=2||
       q38_segment_cache_resize(&model,1)||model.cache[0].cap!=1)goto cleanup;

    Slot *first=q38_expert_get(&model,0,0);if(!first)goto cleanup;
    Q38WeightKind expected=native_fp8?Q38_WEIGHT_FP8:Q38_WEIGHT_F32;
    if(first->gate.kind!=expected||first->up.kind!=expected||
       first->down.kind!=expected||model.miss!=1||model.hits!=0||
       verify_loaded_fp8_slot(first,0,native_fp8))goto cleanup;
    void *slab=first->fp8_slab;
    if(expect_fast){
        if(!native_fp8||!slab||first->gate.data!=slab||
           first->up.data!=(unsigned char*)slab+6||
           first->down.data!=(unsigned char*)slab+12||
           model.expert_weight_reads!=2||model.expert_scale_reads!=2||
           model.expert_pair_reads!=1||model.expert_scale_bytes!=24)
            goto cleanup;
    }else if(model.expert_pair_reads||slab)goto cleanup;
    float first_value=native_fp8?e4m3_decode(((uint8_t*)first->gate.data)[0])*
                                  first->gate.scales[0]:
                                 ((float*)first->gate.data)[0];
    if(q38_expert_get(&model,0,0)!=first||model.miss!=1||model.hits!=1)goto cleanup;
    Slot *second=q38_expert_get(&model,0,1);
    if(second!=first||model.cache[0].by_expert[0]!=-1||
       model.cache[0].by_expert[1]!=0||model.miss!=2||model.hits!=1||
       verify_loaded_fp8_slot(second,1,native_fp8))goto cleanup;
    if(expect_fast&&(second->fp8_slab!=slab||model.expert_weight_reads!=4||
                     model.expert_scale_reads!=2||model.expert_pair_reads!=2))
        goto cleanup;
    float second_value=native_fp8?e4m3_decode(((uint8_t*)second->gate.data)[0])*
                                   second->gate.scales[0]:
                                  ((float*)second->gate.data)[0];
    if(first_value==second_value)goto cleanup;
    if(q38_expert_get(&model,0,0)!=first||model.cache[0].by_expert[0]!=0||
       model.cache[0].by_expert[1]!=-1||model.miss!=3)goto cleanup;
    if(expect_fast&&(first->fp8_slab!=slab||model.expert_weight_reads!=6||
                     model.expert_scale_reads!=2||model.expert_pair_reads!=3))
        goto cleanup;
    result=0;
cleanup:
    destroy_fixture_model(&model);return result;
}

static int check_mixed_layout(const char *directory,int native_fp8,
                              uint64_t expected_bytes,unsigned expected_kinds,
                              const char *expected_class){
    Model model;if(init_fixture_model(&model,directory,native_fp8))return -1;
    uint64_t bytes=0,fixed=0;unsigned kinds=0;char numeric_class[96];int result=-1;
    if(!q38_segment_expert_layout(&model,0,1,&bytes,&fixed,&kinds)){
        q38_segment_numeric_class(numeric_class,sizeof numeric_class,kinds);
        if(bytes==expected_bytes&&!fixed&&kinds==expected_kinds&&
           !strcmp(numeric_class,expected_class)){
            Slot *slot=q38_expert_get(&model,0,0);
            if(slot&&!verify_loaded_fp8_slot(slot,0,native_fp8)&&
               q38_expert_get(&model,0,1)==slot&&
               slot->gate.kind==Q38_WEIGHT_BF16&&
               slot->up.kind==(native_fp8?Q38_WEIGHT_FP8:Q38_WEIGHT_F32)&&
               slot->down.kind==(native_fp8?Q38_WEIGHT_FP8:Q38_WEIGHT_F32)&&
               model.cache[0].by_expert[0]==-1&&
               model.cache[0].by_expert[1]==0){
                int bf16_ok=1;
                const uint16_t *gate=(const uint16_t*)slot->gate.data;
                for(int index=0;index<6;index++)
                    if(gate[index]!=to_bf16(0.0625f*(float)(8+index)))bf16_ok=0;
                if(bf16_ok&&!verify_loaded_fp8_weight(&slot->up,1,1,native_fp8)&&
                   !verify_loaded_fp8_weight(&slot->down,1,2,native_fp8)&&
                   q38_expert_get(&model,0,0)==slot&&
                   slot->gate.kind==(native_fp8?Q38_WEIGHT_FP8:Q38_WEIGHT_F32)&&
                   model.cache[0].by_expert[0]==0&&
                   model.cache[0].by_expert[1]==-1)result=0;
            }
        }
    }
    destroy_fixture_model(&model);return result;
}

static int check_parallel_batch(const char *directory){
    Model model;if(init_fixture_model(&model,directory,1))return -1;
    int result=-1;model.expert_parallel_reads=1;
    int first_ids[2]={1,0};Slot *first[2]={0};
    if(q38_expert_get_batch(&model,0,first_ids,2,first)||model.cache[0].n||
       model.miss||model.hits)goto cleanup;
    if(q38_segment_cache_resize(&model,2))goto cleanup;
    if(!q38_expert_get_batch(&model,0,first_ids,2,first)||!first[0]||!first[1]||
       first[0]==first[1]||first[0]->eid!=1||first[1]->eid!=0||
       model.miss!=2||model.hits||model.expert_parallel_batches!=1||
       model.expert_weight_reads!=4||model.expert_scale_reads!=2||
       model.expert_pair_reads!=2||verify_loaded_fp8_slot(first[0],1,1)||
       verify_loaded_fp8_slot(first[1],0,1))goto cleanup;
    int second_ids[2]={0,1};Slot *second[2]={0};
    if(!q38_expert_get_batch(&model,0,second_ids,2,second)||
       second[0]!=first[1]||second[1]!=first[0]||model.miss!=2||
       model.hits!=2||model.expert_parallel_batches!=1||
       model.expert_weight_reads!=4)goto cleanup;
    result=0;
cleanup:
    destroy_fixture_model(&model);return result;
}

static int check_malformed_scale_metadata(const char *directory){
    Model model;if(init_fixture_model(&model,directory,1))return -1;
    char name[320];
    q38_name(&model,name,sizeof name,0,
             "mlp.experts.0.gate_proj.weight_scale_inv");
    st_tensor *scale=st_find(&model.S,name);int result=-1;
    if(scale){
        int64_t original_bytes=scale->nbytes;
        scale->nbytes++;
        uint64_t bytes=0,fixed=0;unsigned kinds=0;
        if(q38_segment_expert_layout(&model,0,1,&bytes,&fixed,&kinds)&&
           !q38_prepare_expert_scale_bank(&model,0))result=0;
        scale->nbytes=original_bytes;
    }
    destroy_fixture_model(&model);return result;
}

static int check_moe_prefill_parity(const char *directory,int native_fp8,
                                    int cache_capacity){
    static const float input[6]={0.5f,-0.25f,0.5f,-0.25f,0.5f,-0.25f};
    Model prefill={0},decode={0};float batched[6],serial[6];int result=-1;
    if(init_fixture_model(&prefill,directory,native_fp8) ||
       init_fixture_model(&decode,directory,native_fp8) ||
       init_fixture_moe_layer(&prefill) || init_fixture_moe_layer(&decode))
        goto cleanup;
    prefill.c.norm_topk=1;decode.c.norm_topk=1;
    if(cache_capacity!=1&&q38_segment_cache_resize(&prefill,cache_capacity))
        goto cleanup;
    if(cache_capacity!=1&&q38_segment_cache_resize(&decode,cache_capacity))
        goto cleanup;
    prefill.expert_parallel_reads=1;
    q38_moe(&prefill,prefill.L,0,input,3,batched);
    for(int row=0;row<3;row++)
        q38_moe_decode(&decode,decode.L,0,input+(int64_t)row*2,1,
                       serial+(int64_t)row*2);
    if(memcmp(batched,serial,sizeof batched)){
        fprintf(stderr,"prefill parity mismatch native=%d cap=%d:\n",native_fp8,cache_capacity);
        for(int i=0;i<6;i++)fprintf(stderr," %d %.9g %.9g\n",i,batched[i],serial[i]);
        goto cleanup;
    }
    /* Every row routes to both experts.  The grouped prefill loads each expert
     * once, even with a one-slot cache; serial decode must reload both experts
     * for every row.  Native FP8 uses two physical payload ranges per load. */
    uint64_t reads_per_expert=native_fp8?2:3;
    uint64_t expected_decode_misses=cache_capacity>=2?2:6;
    uint64_t expected_decode_reads=expected_decode_misses*reads_per_expert;
    if(prefill.miss!=2||decode.miss!=expected_decode_misses||
       prefill.expert_weight_reads!=2*reads_per_expert||
       decode.expert_weight_reads!=expected_decode_reads){
        fprintf(stderr,"prefill metrics mismatch native=%d cap=%d: miss %llu/%llu reads %llu/%llu\n",
                native_fp8,cache_capacity,(unsigned long long)prefill.miss,
                (unsigned long long)decode.miss,(unsigned long long)prefill.expert_weight_reads,
                (unsigned long long)decode.expert_weight_reads);
        goto cleanup;
    }
    if(cache_capacity>=2&&
       (prefill.hits!=0||decode.hits!=4||
        prefill.expert_parallel_batches!=(native_fp8?1u:0u))){
        fprintf(stderr,"prefill cache metrics mismatch native=%d cap=%d: hits %llu/%llu batches %llu\n",
                native_fp8,cache_capacity,(unsigned long long)prefill.hits,
                (unsigned long long)decode.hits,(unsigned long long)prefill.expert_parallel_batches);
        goto cleanup;
    }
    result=0;
cleanup:
    destroy_fixture_moe_layer(&prefill);destroy_fixture_moe_layer(&decode);
    destroy_fixture_model(&prefill);destroy_fixture_model(&decode);return result;
}

static int check_mixed_moe_prefill_parity(const char *directory){
    static const float input[6]={0.5f,-0.25f,0.5f,-0.25f,0.5f,-0.25f};
    Model prefill={0},decode={0};float batched[6],serial[6];int result=-1;
    if(init_fixture_model(&prefill,directory,1)||
       init_fixture_model(&decode,directory,1)||
       init_fixture_moe_layer(&prefill)||init_fixture_moe_layer(&decode))
        goto cleanup;
    prefill.c.norm_topk=1;decode.c.norm_topk=1;
    q38_moe(&prefill,prefill.L,0,input,3,batched);
    for(int row=0;row<3;row++)
        q38_moe_decode(&decode,decode.L,0,input+(int64_t)row*2,1,
                       serial+(int64_t)row*2);
    if(!memcmp(batched,serial,sizeof batched)&&prefill.miss==2&&decode.miss==6)
        result=0;
cleanup:
    destroy_fixture_moe_layer(&prefill);destroy_fixture_moe_layer(&decode);
    destroy_fixture_model(&prefill);destroy_fixture_model(&decode);return result;
}

static void fill_test_weight(Q38Weight *weight,int rows,int columns,int salt){
    q38_weight_reserve(weight,Q38_WEIGHT_F32,rows,columns);
    float *values=(float*)weight->data;
    for(int index=0;index<rows*columns;index++)
        values[index]=(float)(((index+1)*(salt+3))%17-8)/16.f;
}

/* Projection batching must not alter the causal DeltaNet recurrence. Compare
 * a three-row chunk with three one-row calls, including both retained states. */
static int check_deltanet_prefill_parity(void){
    enum { ROWS=3,H=2,VH=1,KH=1,KD=2,VD=2,V=VH*VD,CD=2*KH*KD+V,CK=2 };
    Cfg config={0};
    config.hidden=H;config.dn_vheads=VH;config.dn_kheads=KH;
    config.dn_kdim=KD;config.dn_vdim=VD;config.dn_conv_dim=CD;
    config.dn_convk=CK;config.eps=1e-6f;
    Layer layer={0};
    fill_test_weight(&layer.dn_qkv,CD,H,1);
    fill_test_weight(&layer.dn_z,V,H,2);
    fill_test_weight(&layer.dn_b,VH,H,3);
    fill_test_weight(&layer.dn_a,VH,H,4);
    fill_test_weight(&layer.dn_out,H,V,5);
    layer.dn_conv=(float*)malloc((size_t)CD*CK*sizeof(float));
    layer.dn_dtbias=(float*)malloc(VH*sizeof(float));
    layer.dn_alog=(float*)malloc(VH*sizeof(float));
    layer.dn_norm=(float*)malloc(VD*sizeof(float));
    if(!layer.dn_conv||!layer.dn_dtbias||!layer.dn_alog||!layer.dn_norm)
        goto fail;
    for(int index=0;index<CD*CK;index++)
        layer.dn_conv[index]=(float)((index%7)-3)/16.f;
    layer.dn_dtbias[0]=0.125f;layer.dn_alog[0]=-0.75f;
    layer.dn_norm[0]=0.875f;layer.dn_norm[1]=1.125f;

    float recurrence_a[VH*KD*VD]={0},recurrence_b[VH*KD*VD]={0};
    float convolution_a[CD*(CK-1)]={0},convolution_b[CD*(CK-1)]={0};
    float *recurrence_rows_a[1]={recurrence_a},*recurrence_rows_b[1]={recurrence_b};
    float *convolution_rows_a[1]={convolution_a},*convolution_rows_b[1]={convolution_b};
    Model batched={0},serial={0};
    batched.c=config;serial.c=config;
    batched.prefill_batch=1;
    batched.DN_rec=recurrence_rows_a;serial.DN_rec=recurrence_rows_b;
    batched.DN_conv=convolution_rows_a;serial.DN_conv=convolution_rows_b;
    float input[ROWS*H]={0.5f,-0.25f,-0.75f,0.625f,0.125f,0.875f};
    float output_a[ROWS*H],output_b[ROWS*H];
    q38_deltanet(&batched,&layer,0,input,ROWS,output_a);
    for(int row=0;row<ROWS;row++)
        q38_deltanet(&serial,&layer,0,input+(int64_t)row*H,1,
                     output_b+(int64_t)row*H);
    if(memcmp(output_a,output_b,sizeof output_a)||
       memcmp(recurrence_a,recurrence_b,sizeof recurrence_a)||
       memcmp(convolution_a,convolution_b,sizeof convolution_a))goto fail;

    q38_weight_free(&layer.dn_qkv);q38_weight_free(&layer.dn_z);
    q38_weight_free(&layer.dn_b);q38_weight_free(&layer.dn_a);
    q38_weight_free(&layer.dn_out);free(layer.dn_conv);free(layer.dn_dtbias);
    free(layer.dn_alog);free(layer.dn_norm);return 0;
fail:
    q38_weight_free(&layer.dn_qkv);q38_weight_free(&layer.dn_z);
    q38_weight_free(&layer.dn_b);q38_weight_free(&layer.dn_a);
    q38_weight_free(&layer.dn_out);free(layer.dn_conv);free(layer.dn_dtbias);
    free(layer.dn_alog);free(layer.dn_norm);return -1;
}

static int check_segment_failure_outputs(void){
    Model partial={0};partial.c.layers=1;q38_model_free(&partial);
    char error[128];void *output=(void*)(uintptr_t)1;
    ColiSegmentCapabilities capabilities;
    ColiSegmentEngineOptions engine_options={
        .struct_size=sizeof(engine_options),.model_dir="unused",
        .context_tokens=1,.backend_mask=COLI_SEGMENT_CAP_CUDA,
    };
    if(!qwen38_segment_engine_open(&output,&capabilities,&engine_options,
                                   error,sizeof error)||output)return -1;

    Qwen38SegmentEngine engine={0};engine.context_tokens=1;
    ColiSegmentSessionOptions session_options={
        .struct_size=sizeof(session_options),.context_tokens=2,
    };
    output=(void*)(uintptr_t)1;
    if(!qwen38_segment_session_create(&engine,&output,&session_options,
                                      error,sizeof error)||output)return -1;
    return 0;
}

int main(void){
    enum { S=2, I=257, O=129 };
    Q38Weight fp8={0};q38_weight_reserve(&fp8,Q38_WEIGHT_FP8,O,I);
    CHECK(fp8.scale_count==6);CHECK(q38_weight_bytes(&fp8)==(uint64_t)O*I+6*sizeof(float));
    uint8_t *raw=(uint8_t*)fp8.data;
    for(int64_t i=0;i<fp8.elements;i++)raw[i]=(uint8_t)((i*17+13)%0x7f);
    for(int64_t block=0;block<fp8.scale_count;block++)
        fp8.scales[block]=(float)(block+1)*0.25f*(block&1?-1.0f:1.0f);
    float x[S*I];for(int i=0;i<S*I;i++)x[i]=(float)((i%19)-9)/7.f;
    float want[S*O],got[S*O];
    reference_fp8_matmul(want,x,raw,fp8.scales,S,I,O);
    q38_weight_matmul(got,x,&fp8,S,I,O);
    CHECK(!memcmp(want,got,sizeof want));
    void *same=fp8.data;q38_weight_reserve(&fp8,Q38_WEIGHT_FP8,O,I);
    CHECK(fp8.data==same);
    q38_weight_free(&fp8);CHECK(fp8.kind==Q38_WEIGHT_NONE&&!fp8.data&&!fp8.scales);

    enum { BI=9, BO=4, BS=3 };
    Q38Weight bf16={0};q38_weight_reserve(&bf16,Q38_WEIGHT_BF16,BO,BI);
    uint16_t *half=(uint16_t*)bf16.data;float full[BO*BI];
    for(int i=0;i<BO*BI;i++){
        float value=(float)((i%11)-5)*0.125f;
        half[i]=to_bf16(value);full[i]=bf16_to_f32(half[i]);
    }
    float bx[BS*BI],bwant[BS*BO],bgot[BS*BO];
    for(int i=0;i<BS*BI;i++)bx[i]=(float)((i%7)-3)*0.25f;
    q38_matmul(bwant,bx,full,BS,BI,BO);
    q38_weight_matmul(bgot,bx,&bf16,BS,BI,BO);
    CHECK(!memcmp(bwant,bgot,sizeof bwant));
    float row[BI];q38_weight_row(&bf16,2,row);
    CHECK(!memcmp(row,full+2*BI,sizeof row));
    CHECK(q38_weight_bytes(&bf16)==(uint64_t)BO*BI*sizeof(uint16_t));
    q38_weight_free(&bf16);

    char directory[]="test_qwen38_native_XXXXXX";CHECK(mkdtemp(directory)!=NULL);
    CHECK(write_fp8_fixture(directory,0,0)==0);
    CHECK(check_fixture_mode(directory,1,0)==0);
    CHECK(check_fixture_mode(directory,0,0)==0);
    char path[512];int path_length=snprintf(path,sizeof path,
                                           "%s/model.safetensors",directory);
    CHECK(path_length>0&&(size_t)path_length<sizeof path);
    CHECK(remove(path)==0);CHECK(rmdir(directory)==0);

    char mixed_directory[]="test_qwen38_mixed_XXXXXX";
    CHECK(mkdtemp(mixed_directory)!=NULL);
    CHECK(write_fp8_fixture(mixed_directory,1,0)==0);
    CHECK(check_mixed_layout(mixed_directory,1,32,
          Q38_EXPERT_FP8_BLOCK|Q38_EXPERT_BF16,
          "qwen38/mixed-expert-arithmetic-05/cpu-v1")==0);
    CHECK(check_mixed_layout(mixed_directory,0,72,
          Q38_EXPERT_FP8_EXPANDED|Q38_EXPERT_BF16,
          "qwen38/mixed-expert-arithmetic-06/cpu-v1")==0);
    CHECK(check_mixed_moe_prefill_parity(mixed_directory)==0);
    path_length=snprintf(path,sizeof path,"%s/model.safetensors",mixed_directory);
    CHECK(path_length>0&&(size_t)path_length<sizeof path);
    CHECK(remove(path)==0);CHECK(rmdir(mixed_directory)==0);

    char adjacent_directory[]="test_qwen38_adjacent_XXXXXX";
    CHECK(mkdtemp(adjacent_directory)!=NULL);
    CHECK(write_fp8_fixture(adjacent_directory,0,1)==0);
    CHECK(check_fixture_mode(adjacent_directory,1,1)==0);
    CHECK(check_fixture_mode(adjacent_directory,0,0)==0);
    CHECK(check_parallel_batch(adjacent_directory)==0);
    CHECK(check_malformed_scale_metadata(adjacent_directory)==0);
    CHECK(check_moe_prefill_parity(adjacent_directory,1,1)==0);
    CHECK(check_moe_prefill_parity(adjacent_directory,1,2)==0);
    CHECK(check_moe_prefill_parity(adjacent_directory,0,1)==0);
    CHECK(check_deltanet_prefill_parity()==0);
    path_length=snprintf(path,sizeof path,"%s/model.safetensors",adjacent_directory);
    CHECK(path_length>0&&(size_t)path_length<sizeof path);
    CHECK(remove(path)==0);CHECK(rmdir(adjacent_directory)==0);

    CHECK(check_segment_failure_outputs()==0);

    puts("qwen38 native weights: FP8 oracle, coalesced slabs/scales, parallel demand sets, fallbacks, adapter failures: ok");
    return 0;
}
