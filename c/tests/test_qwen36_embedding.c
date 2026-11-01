/* Native embedding row/ownership/error contracts. Synthetic files only. */
#define _GNU_SOURCE
#include <assert.h>
#include <limits.h>
#include "../st.h"
static int observed_map(int,int64_t,size_t,compat_ro_map *,const void **);
static void observed_unmap(compat_ro_map *);
#define QEM_MAP_VIEW observed_map
#define QEM_UNMAP_VIEW observed_unmap
#include "../qwen36_embedding.h"
static int checks,map_calls,live_maps,inject;
#define CHECK(c) do {checks++;if(!(c)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);exit(1);}}while(0)
static int observed_map(int fd,int64_t off,size_t len,compat_ro_map *map,const void **data) {
    map_calls++;if(inject==1){errno=ENOMEM;return -1;}
    int status=qem_map_view(fd,off,len,map,data);if(!status)live_maps++;
    if(!status && inject==2){errno=ENOMEM;return -1;}return status;
}
static void observed_unmap(compat_ro_map *map) {
    if(map && map->base)live_maps--;compat_unmap_readonly(map);
}
typedef struct {FILE *file;st_tensor t;shards s;} Fixture;
static void fixture(Fixture *f,int dtype,int I,int O,unsigned salt) {
    memset(f,0,sizeof(*f));f->file=tmpfile();CHECK(f->file!=NULL);
    unsigned char prefix[37]={0};CHECK(fwrite(prefix,1,sizeof prefix,f->file)==sizeof prefix);
    uint32_t rng=0x71379u;
    static const uint32_t special[]={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc12345,0x7f812345};
    for(size_t i=0;i<(size_t)I*O;i++) {
        if(dtype==2){rng=rng*1664525u+1013904223u;uint32_t v=i<sizeof special/sizeof special[0]?special[i]:rng;CHECK(fwrite(&v,4,1,f->file)==1);}
        else{uint16_t v=(uint16_t)(i+salt);CHECK(fwrite(&v,2,1,f->file)==1);}
    }
    CHECK(!fflush(f->file));f->t.name="embedding";f->t.fd=fileno(f->file);f->t.off=sizeof prefix;
    f->t.rank=2;f->t.shape[0]=O;f->t.shape[1]=I;f->t.dtype=dtype;
    f->t.numel=(int64_t)I*O;f->t.nbytes=f->t.numel*(dtype==2?4:2);f->s.t=&f->t;f->s.n=1;
}
static void conversion_and_lifetime(int dtype) {
    enum{I=257,O=256};Fixture f;fixture(&f,dtype,I,O,0);
    float *expected=malloc((size_t)I*O*4);CHECK(expected!=NULL);
    st_read_f32(&f.s,"embedding",expected,0);
    QwenEmbedding a={0},b={0};CHECK(qem_open(&a,&f.s,"embedding",I,O));CHECK(qem_open(&b,&f.s,"embedding",I,O));
    CHECK(live_maps==2 && a.data!=b.data && a.nbytes==(size_t)f.t.nbytes);
    CHECK(a.map.len>=a.nbytes && a.map.len-a.nbytes<65536);
    int before=map_calls;QwenEmbedding saved=a;
    CHECK(!qem_open(&a,&f.s,"embedding",I,O) && map_calls==before && !memcmp(&a,&saved,sizeof a));
    // A live view owns its mapping, not a borrowed shards/tensor/fd identity.
    CHECK(!fclose(f.file));memset(&f,0,sizeof f);
    float row[I+2];row[0]=123.f;row[I+1]=-456.f;
    for(int id=O-1;id>=0;id--){
        CHECK(qem_row(&a,id,row+1,I));CHECK(!memcmp(row+1,expected+(size_t)id*I,I*4));
        CHECK(row[0]==123.f && row[I+1]==-456.f);
    }
    qem_free(&a);qem_free(&a);CHECK(qem_empty(&a) && live_maps==1);
    CHECK(qem_row(&b,O-1,row+1,I) && !memcmp(row+1,expected+(O-1)*I,I*4));
    float unchanged[I+2];memcpy(unchanged,row,sizeof row);
    CHECK(!qem_row(&b,-1,row+1,I));CHECK(!qem_row(&b,O,row+1,I));
    CHECK(!qem_row(&b,0,row+1,I-1));CHECK(!qem_row(&a,0,row+1,I));
    CHECK(!memcmp(row,unchanged,sizeof row));
    CHECK(!qem_row(&b,0,(float *)(uintptr_t)b.data,I)); // no write into immutable view
    for(int bad=0;bad<6;bad++){
        QwenEmbedding invalid=b;
        if(bad==0)invalid.I=INT_MAX;
        if(bad==1)invalid.nbytes--;
        if(bad==2)invalid.row_bytes=0;
        if(bad==3)invalid.dtype=3;
        if(bad==4)invalid.map.len=SIZE_MAX;
        if(bad==5)invalid.data=(const unsigned char *)((uintptr_t)b.map.base-1);
        CHECK(!qem_row(&invalid,0,row+1,I));CHECK(!memcmp(row,unchanged,sizeof row));
    }
    qem_free(&b);CHECK(qem_empty(&b) && !live_maps);free(expected);
    printf("%s rows: %s match st.h's reader; two mappings of one tensor outlive its file and are released one by one\n",
           dtype==2?"F32":dtype?"F16":"BF16",dtype==2?"special values and random bit patterns":"all 65536 bit patterns");
}
static void failures(void) {
    Fixture tiny;fixture(&tiny,1,1,1,17);QwenEmbedding view={0};
    CHECK(qem_open(&view,&tiny.s,"embedding",1,1));
    CHECK(!qem_row(&view,0,(float *)view.map.base,1)); /* aligned prefix padding */
    CHECK(!qem_row(&view,0,(float *)((unsigned char *)view.map.base+view.map.len-4),1)); /* page tail */
    qem_free(&view);fclose(tiny.file);CHECK(!live_maps);
    Fixture f;fixture(&f,1,65,7,23);QwenEmbedding out={0};
    for(int fault=1;fault<=2;fault++){
        inject=fault;CHECK(!qem_open(&out,&f.s,"embedding",65,7));CHECK(qem_empty(&out) && !live_maps);
    }
    inject=0;st_tensor original=f.t;int before=map_calls;
    for(int which=0;which<10;which++){
        f.t=original;
        if(which==0)f.t.dtype=3;
        if(which==1)f.t.rank=1;
        if(which==2)f.t.shape[0]++;
        if(which==3)f.t.shape[1]++;
        if(which==4)f.t.numel--;
        if(which==5)f.t.nbytes--;
        if(which==6)f.t.off=-1;
        if(which==7)f.t.off=INT64_MAX;
        if(which==8)f.t.fd=-1;
        if(which==9)f.t.off=original.off+original.nbytes;
        CHECK(!qem_open(&out,&f.s,"embedding",65,7));CHECK(qem_empty(&out) && !live_maps && map_calls==before);
    }
    f.t=original;CHECK(!qem_open(&out,&f.s,"missing",65,7));
    CHECK(!qem_open(&out,&f.s,"embedding",0,7));CHECK(!qem_open(&out,&f.s,"embedding",65,-1));
    f.t.shape[0]=f.t.shape[1]=INT_MAX;f.t.numel=(int64_t)INT_MAX*INT_MAX;f.t.nbytes=f.t.numel*2;
    CHECK(!qem_open(&out,&f.s,"embedding",INT_MAX,INT_MAX));CHECK(qem_empty(&out) && map_calls==before);
    f.t=original;QwenEmbedding dirty={0};dirty.I=1;
    CHECK(!qem_open(&dirty,&f.s,"embedding",65,7) && dirty.I==1 && map_calls==before);
    CHECK(!ftruncate(f.t.fd,(off_t)(f.t.off+f.t.nbytes-1)));
    CHECK(!qem_open(&out,&f.s,"embedding",65,7) && qem_empty(&out) && map_calls==before);
    fclose(f.file);CHECK(!qem_open(&out,&f.s,"embedding",65,7) && qem_empty(&out));
    CHECK(!live_maps);
}
int main(void) {
    for(int dtype=0;dtype<3;dtype++)conversion_and_lifetime(dtype);
    failures();printf("native embedding: %d checks, 0 failures, live mappings=%d\n",checks,live_maps);return 0;
}
