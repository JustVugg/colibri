/* K3_MMAP: prepared U8 weights and F32 scales must produce exactly the same
 * CPU math as the existing heap-copy loader while remaining file-backed. */
#define main kimi_k3_main_unused
#include "../kimi_k3.c"
#undef main

#include <stdio.h>

static int failures;
#define CHECK(cond, ...) do { if(!(cond)){ \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while(0)

static void write_fixture(const char *dir){
    char path[512]; snprintf(path,sizeof(path),"%s/model.safetensors",dir);
    const char *hdr=
        "{\"w8\":{\"dtype\":\"U8\",\"shape\":[2,4],\"data_offsets\":[0,8]},"
        "\"w8.qs\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]},"
        "\"w4\":{\"dtype\":\"U8\",\"shape\":[2,32],\"data_offsets\":[16,80]},"
        "\"w4.qs\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[80,88]}}";
    const int8_t q8[8]={-7,3,9,12,5,-4,2,11};
    const float s8[2]={0.25f,1.5f};
    uint8_t q4[64]; float s4[2]={0.125f,0.75f};
    for(int i=0;i<64;i++) q4[i]=(uint8_t)((i*13+7)&255);
    size_t raw_hlen=strlen(hdr);
    size_t pad=(8-((8+raw_hlen)%8))%8;
    uint64_t hlen=(uint64_t)(raw_hlen+pad);
    FILE *f=fopen(path,"wb");
    fwrite(&hlen,8,1,f); fwrite(hdr,1,raw_hlen,f);
    for(size_t i=0;i<pad;i++) fputc(' ',f);
    fwrite(q8,1,sizeof(q8),f); fwrite(s8,1,sizeof(s8),f);
    fwrite(q4,1,sizeof(q4),f); fwrite(s4,1,sizeof(s4),f); fclose(f);
}

static void close_fixture_shards(shards *S){
    st_mirror_reset(S);
    for(int i=0;i<S->nfd;i++){
        if(S->dfds[i]>=0&&S->dfds[i]!=S->fds[i]) close(S->dfds[i]);
        if(S->fds[i]>=0) close(S->fds[i]);
        S->dfds[i]=S->fds[i]=-1;
    }
}

static void compare_weight(Model *m, const char *name, int O, int I, int bits){
    W copied={0}, mapped={0}; float x[64], yc[2]={0}, ym[2]={0};
    float ac[64]={0}, am[64]={0};
    for(int i=0;i<I;i++) x[i]=(float)(i%9-4)*0.125f;

    g_k3_mmap=0; g_bits_env=0; w_load(m,&copied,name,O,I,bits);
    g_k3_mmap=1; w_load(m,&mapped,name,O,I,bits);
    CHECK(mapped.mapped,"%s did not use mapped storage",name);
    CHECK(!copied.mapped,"%s changed the default loader",name);
    CHECK(mapped.s==(float*)mapped.scale_map.data,"%s scales are not mapped",name);
    if(mapped.fmt==1) CHECK(mapped.q8==(int8_t*)mapped.data_map.data,"%s int8 data are not mapped",name);
    if(mapped.fmt==4) CHECK(mapped.q4==(uint8_t*)mapped.data_map.data,"%s int4 data are not mapped",name);

    w_matmul(yc,x,&copied,1); w_matmul(ym,x,&mapped,1);
    CHECK(!memcmp(yc,ym,sizeof(yc)),"%s matmul output differs",name);
    w_addrow(&copied,1,0.75f,ac); w_addrow(&mapped,1,0.75f,am);
    CHECK(!memcmp(ac,am,(size_t)I*sizeof(float)),"%s row expansion differs",name);
    float dc=w_rowdot(&copied,1,x), dm=w_rowdot(&mapped,1,x);
    CHECK(!memcmp(&dc,&dm,sizeof(dc)),"%s row dot differs",name);

    w_release_host(&mapped); w_release_host(&copied);
}

int main(void){
    char dir[]="test_k3_mmap_XXXXXX"; CHECK(mkdtemp(dir)!=NULL,"mkdtemp failed");
    write_fixture(dir);
    Model m; memset(&m,0,sizeof(m)); st_init(&m.S,dir);

    compare_weight(&m,"w8",2,4,8);
    compare_weight(&m,"w4",2,64,4);
    CHECK(k3_mmap_backend_allowed(1,0,0),"CPU-only mapped mode refused");
    CHECK(!k3_mmap_backend_allowed(1,1,0),"mapped Vulkan combination accepted");
    CHECK(!k3_mmap_backend_allowed(1,0,1),"mapped CUDA combination accepted");
    CHECK(k3_mmap_backend_allowed(0,1,1),"default loader changed GPU policy");

    close_fixture_shards(&m.S);
    char path[512]; snprintf(path,sizeof(path),"%s/model.safetensors",dir);
    CHECK(remove(path)==0,"fixture file cleanup failed");
#ifdef _WIN32
    CHECK(_rmdir(dir)==0,"fixture directory cleanup failed");
#else
    CHECK(rmdir(dir)==0,"fixture directory cleanup failed");
#endif
    if(failures){ fprintf(stderr,"k3 mmap: %d failure(s)\n",failures); return 1; }
    puts("k3 mmap: mapped and copied CPU paths are exact"); return 0;
}
