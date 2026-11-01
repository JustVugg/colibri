/* qwen36's embedding read in place: the checkpoint's embed_tokens tensor is
 * mapped read-only, and a token's row is converted to f32 only when the forward
 * reads it, with st.h's bf16/f16 helpers: the same values as the f32 table the
 * engine loads otherwise. On the 35B that table is 248,320 x 2048 floats, 2 GB
 * of RAM; the mapped pages cost nothing until read, and the system can drop
 * them again. The checkpoint must not change while it is mapped. */
#ifndef COLI_QWEN36_EMBEDDING_H
#define COLI_QWEN36_EMBEDDING_H
#include "st.h"
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <sys/stat.h>

typedef struct {
    compat_ro_map map;
    const unsigned char *data;
    size_t nbytes,row_bytes;
    int I,O,dtype; /* safetensors BF16=0, F16=1, F32=2 */
} QwenEmbedding;

/* A private read-only mapping of the file, never an anonymous copy; on Windows
 * st.h's read-only file view, its length rounded up to whole pages. */
static inline int qem_map_view(int fd,int64_t off,size_t len,compat_ro_map *map,const void **data) {
#ifdef _WIN32
    int status=compat_map_readonly(fd,off,len,map,data);
    if(!status) {
        SYSTEM_INFO si;GetSystemInfo(&si);size_t page=si.dwPageSize?si.dwPageSize:4096;
        if(map->len>SIZE_MAX-(page-1)){compat_unmap_readonly(map);*data=NULL;errno=EOVERFLOW;return -1;}
        map->len=((map->len+page-1)/page)*page;
    }
    return status;
#else
    if(!map || !data || fd<0 || off<0 || !len){errno=EINVAL;return -1;}
    memset(map,0,sizeof(*map));*data=NULL;
    long page=sysconf(_SC_PAGESIZE);if(page<=0){errno=EINVAL;return -1;}
    uint64_t aligned=(uint64_t)off-(uint64_t)off%(uint64_t)page;
    size_t delta=(size_t)((uint64_t)off-aligned);
    if(len>SIZE_MAX-delta || (int64_t)(off_t)aligned!=(int64_t)aligned){errno=EOVERFLOW;return -1;}
    size_t extent=len+delta;
    if(extent>SIZE_MAX-((size_t)page-1)){errno=EOVERFLOW;return -1;}
    extent=((extent+(size_t)page-1)/(size_t)page)*(size_t)page;
    void *base=mmap(NULL,extent,PROT_READ,MAP_PRIVATE,fd,(off_t)aligned);
    if(base==MAP_FAILED)return -1;
    map->base=base;map->len=extent;*data=(const unsigned char *)base+delta;return 0;
#endif
}
#ifndef QEM_MAP_VIEW
#define QEM_MAP_VIEW qem_map_view
#endif
#ifndef QEM_UNMAP_VIEW
#define QEM_UNMAP_VIEW compat_unmap_readonly
#endif

static inline int qem_empty(const QwenEmbedding *e) {
    if(!e || e->map.base || e->map.len || e->data || e->nbytes || e->row_bytes || e->I || e->O || e->dtype)return 0;
#ifdef _WIN32
    if(e->map.mapping)return 0;
#endif
    return 1;
}
static inline void qem_free(QwenEmbedding *e) {
    if(!e)return;QEM_UNMAP_VIEW(&e->map);memset(e,0,sizeof(*e));
}
/* Every shape and extent check, and the mapping, come before *out is written:
 * on failure the caller's empty descriptor is unchanged (and the engine loads
 * the f32 table instead). */
static inline int qem_open(QwenEmbedding *out,shards *s,const char *name,int I,int O) {
    if(!qem_empty(out) || !s || !name || I<=0 || O<=0){errno=EINVAL;return 0;}
    const st_tensor *t=st_find(s,name);
    if(!t || t->dtype<0 || t->dtype>2 || t->rank!=2 || t->shape[0]!=O || t->shape[1]!=I || t->fd<0 || t->off<0){errno=EINVAL;return 0;}
    size_t esz=t->dtype==2?4:2;
    uint64_t count=(uint64_t)(unsigned)I*(uint64_t)(unsigned)O;
    if(count>SIZE_MAX/esz || count>INT64_MAX/esz || t->numel!=(int64_t)count || t->nbytes!=(int64_t)(count*esz) || t->off>INT64_MAX-t->nbytes){errno=EINVAL;return 0;}
    struct stat sb;
    if(fstat(t->fd,&sb))return 0;
    if(sb.st_size<0 || (uint64_t)t->off>(uint64_t)sb.st_size || (uint64_t)t->nbytes>(uint64_t)sb.st_size-(uint64_t)t->off){errno=EINVAL;return 0;}
    QwenEmbedding tmp={0};const void *data=NULL;
    if(QEM_MAP_VIEW(t->fd,t->off,(size_t)t->nbytes,&tmp.map,&data)!=0 || !data){
        QEM_UNMAP_VIEW(&tmp.map);return 0;
    }
    tmp.data=data;tmp.nbytes=(size_t)t->nbytes;tmp.row_bytes=(size_t)I*esz;
    tmp.I=I;tmp.O=O;tmp.dtype=t->dtype;*out=tmp;return 1;
}
/* One row into out (at least I floats, not inside the mapping), checked before
 * anything is written. memcpy reads unaligned rows and keeps f32 bits as they
 * are (signed zeros, NaNs); bf16 and f16 are converted exactly. */
static inline int qem_row(const QwenEmbedding *e,int token,float *out,size_t out_count) {
    if(!e || !out || !e->data || !e->map.base || e->I<=0 || e->O<=0 ||
       e->dtype<0 || e->dtype>2 || token<0 || token>=e->O || out_count<(size_t)e->I)return 0;
    size_t esz=e->dtype==2?4:2;
    if((size_t)e->I>SIZE_MAX/4 || e->row_bytes!=(size_t)e->I*esz ||
       (size_t)e->O>SIZE_MAX/e->row_bytes || e->nbytes!=(size_t)e->O*e->row_bytes)return 0;
    uintptr_t base=(uintptr_t)e->map.base,src=(uintptr_t)e->data,dst=(uintptr_t)out;
    if(e->map.len>UINTPTR_MAX-base || src<base || src-base>e->map.len ||
       e->nbytes>e->map.len-(src-base))return 0;
    size_t output_bytes=(size_t)e->I*4;
    if(output_bytes>UINTPTR_MAX-dst || (dst<base+e->map.len && base<dst+output_bytes))return 0;
    const unsigned char *row=e->data+(size_t)token*e->row_bytes;
    if(e->dtype==2)memcpy(out,row,output_bytes);
    else for(int i=0;i<e->I;i++) {
        uint16_t value;memcpy(&value,row+(size_t)i*2,2);
        out[i]=e->dtype==1?f16_to_f32(value):bf16_to_f32(value);
    }
    return 1;
}
#endif
