/* Tiny offline conversion fixtures only. No model, inference or GPU.
 * The converter repacks each record in place, one 32-byte block at a time,
 * so the fixture also covers expert_ffn.h's in-place repack. */
#define _GNU_SOURCE
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dirent.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef _WIN32
#include <signal.h>
#include <sys/wait.h>
/* fstat as the stamp sees it, with the device number or the inode changed at
 * will: a restart can renumber a volume, and the stamp must not change; a new
 * inode is another file, and it must. */
static dev_t stamp_device_flip;
static ino_t stamp_inode_flip;
static int stamp_fstat(int fd,struct stat *info) {
    int result=fstat(fd,info);
    if(!result){info->st_dev^=stamp_device_flip;info->st_ino^=stamp_inode_flip;}
    return result;
}
#define fstat stamp_fstat
#endif
#include "../qwen36_expert_source.h"
#ifndef _WIN32
#undef fstat
#endif

#ifndef _WIN32
static int observed_read(int,void *,size_t,uint64_t,char *,size_t);
static size_t observed_write(const void *,size_t,size_t,FILE *);
static int observed_sync(int);
static int observed_close(FILE *);
static int observed_rename(const char *,const char *);
static void *observed_malloc(size_t);
static void observed_free(void *);
/* Source helpers remain real; only converter calls get deterministic faults. */
#define qes_read_at observed_read
#define fwrite observed_write
#define fsync observed_sync
#define fclose observed_close
#define rename observed_rename
#define malloc observed_malloc
#define free observed_free
#include "../qwen36_expert_convert.h"
#undef qes_read_at
#undef fwrite
#undef fsync
#undef fclose
#undef rename
#undef malloc
#undef free

static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);exit(1);}}while(0)
enum {LAYERS=2,EXPERTS=2,HIDDEN=128,INTER=64,KEYS=LAYERS*EXPERTS,
      PACKED=3*HIDDEN*INTER/2,SCALES=3*HIDDEN*INTER/64,
      RECORD=PACKED+SCALES*4,SOURCE_HEADER=4096};
static const QefGeometry geometry={LAYERS,EXPERTS,HIDDEN,INTER,64};
/* +0, -0, -1, smallest negative normal, -2.5: accepted without changing bits. */
static const uint32_t finite_scale_bits[]={
    UINT32_C(0x00000000),UINT32_C(0x80000000),UINT32_C(0xbf800000),
    UINT32_C(0x80800000),UINT32_C(0xc0200000)
};
static char error_text[512];
typedef struct {
    shards source;
    st_tensor tensor[2*KEYS+1];
    char names[2*KEYS+1][160];
    char directory[128],path[3][256],output[256];
    uint8_t weights[KEYS][PACKED];
    float scales[KEYS][SCALES];
} Fixture;

enum {FAULT_NONE,FAULT_READ,FAULT_MUTATE,FAULT_WRITE,FAULT_SYNC,FAULT_CLOSE,FAULT_RENAME,FAULT_SIGNAL};
static int fault,read_countdown,write_countdown,mutate_fd;
static off_t mutate_offset;
static unsigned allocations,live_allocations;
static size_t largest_allocation;
static int allocation_failure=-1;
static void *tracked[2];

static void *observed_malloc(size_t n) {
    allocations++;
    if(allocation_failure==0)return NULL;
    if(allocation_failure>0)allocation_failure--;
    void *p=malloc(n);
    if(p) {
        CHECK(live_allocations<2);
        for(int i=0;i<2;i++)if(!tracked[i]){tracked[i]=p;break;}
        live_allocations++;
        if(n>largest_allocation)largest_allocation=n;
    }
    return p;
}
static void observed_free(void *p) {
    if(p) {
        int found=0;
        for(int i=0;i<2;i++)if(tracked[i]==p){tracked[i]=NULL;found=1;break;}
        CHECK(found && live_allocations);live_allocations--;
    }
    free(p);
}
static void write_at(int fd,const void *bytes,size_t n,off_t offset) {
    const unsigned char *p=bytes;
    while(n) {
        ssize_t wrote=pwrite(fd,p,n,offset);
        if(wrote<0 && errno==EINTR)continue;
        CHECK(wrote>0);p+=wrote;n-=(size_t)wrote;offset+=wrote;
    }
}
static int observed_read(int fd,void *out,size_t n,uint64_t off,char *error,size_t cap) {
    if(fault==FAULT_READ && --read_countdown==0) {
        fault=FAULT_NONE;return qes_error(error,cap,"injected source read failure");
    }
    int ok=qes_read_at(fd,out,n,off,error,cap);
    if(ok && fault==FAULT_MUTATE && --read_countdown==0) {
        struct stat before;unsigned char byte;
        CHECK(fstat(mutate_fd,&before)==0);
        CHECK(pread(mutate_fd,&byte,1,mutate_offset)==1);
        byte^=0x35;write_at(mutate_fd,&byte,1,mutate_offset);
        /* Make the metadata delta deterministic even on a coarse filesystem.
         * The changed byte can be outside all fingerprint samples. */
        struct timespec times[2]={{0,UTIME_OMIT},{before.st_mtime+1,0}};
        CHECK(futimens(mutate_fd,times)==0);
        fault=FAULT_NONE;
    }
    return ok;
}
static size_t observed_write(const void *p,size_t size,size_t count,FILE *out) {
    if(fault==FAULT_WRITE && --write_countdown==0) {
        fault=FAULT_NONE;errno=ENOSPC;return 0;
    }
    if(fault==FAULT_SIGNAL && --write_countdown==0)raise(SIGTERM);
    return fwrite(p,size,count,out);
}
static int observed_sync(int fd) {
    if(fault==FAULT_SYNC){fault=FAULT_NONE;errno=EIO;return -1;}
    return fsync(fd);
}
static int observed_close(FILE *out) {
    int result=fclose(out);
    if(fault==FAULT_CLOSE){fault=FAULT_NONE;errno=EIO;return EOF;}
    return result;
}
static int observed_rename(const char *from,const char *to) {
    if(fault==FAULT_RENAME){fault=FAULT_NONE;errno=EACCES;return -1;}
    return rename(from,to);
}

static void fixture_header(Fixture *f,int which) {
    char json[SOURCE_HEADER];memset(json,' ',sizeof json);
    size_t at=0;json[at++]='{';int first=1;
    for(int i=0;i<f->source.n;i++) {
        st_tensor *t=&f->tensor[i];if(t->fd!=f->source.fds[which])continue;
        int n=snprintf(json+at,sizeof json-at,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
            first?"":",",t->name,t->dtype==2?"F32":"U8",(long long)t->numel,
            (long long)(t->off-8-SOURCE_HEADER),(long long)(t->off-8-SOURCE_HEADER+t->nbytes));
        CHECK(n>0 && (size_t)n<sizeof json-at);at+=(size_t)n;first=0;
    }
    CHECK(at+1<sizeof json);json[at++]='}';memset(json+at,' ',sizeof json-at);
    const uint64_t header_bytes=sizeof json;
    write_at(f->source.fds[which],&header_bytes,sizeof header_bytes,0);
    write_at(f->source.fds[which],json,sizeof json,8);
    struct stat state;CHECK(fstat(f->source.fds[which],&state)==0);
    f->source.sizes[which]=state.st_size;
}
static void fixture_tensor(Fixture *f,int index,int source_fd,int dtype,
                            size_t elements,off_t offset) {
    st_tensor *t=&f->tensor[index];t->name=f->names[index];t->fd=source_fd;
    t->dtype=dtype;t->numel=(int64_t)elements;t->nbytes=(int64_t)(elements*(dtype==2?4:1));
    t->rank=1;t->shape[0]=(int64_t)elements;t->off=offset;
}
static void fixture_open(Fixture *f) {
    memset(f,0,sizeof *f);
    snprintf(f->directory,sizeof f->directory,"/tmp/colibri-qec-XXXXXX");
    CHECK(mkdtemp(f->directory));
    snprintf(f->output,sizeof f->output,"%s/experts.qc",f->directory);
    f->source.nfd=3;f->source.n=2*KEYS+1;f->source.t=f->tensor;
    const char *names[]={"weights.safetensors","scales.safetensors","dense.safetensors"};
    for(int i=0;i<3;i++) {
        snprintf(f->path[i],sizeof f->path[i],"%s/%s",f->directory,names[i]);
        f->source.paths[i]=f->path[i];
        f->source.fds[i]=open(f->path[i],O_RDWR|O_CREAT|O_EXCL,0600);
        CHECK(f->source.fds[i]>=0);f->source.dfds[i]=-2;
        for(int r=0;r<ST_MAX_MIR;r++)f->source.mfds[r][i]=-1;
    }
    for(unsigned key=0;key<KEYS;key++) {
        unsigned layer=key/EXPERTS,expert=key%EXPERTS;
        snprintf(f->names[2*key],sizeof f->names[0],
                 "model.layers.%u.mlp.experts.%u.merged_weight",layer,expert);
        snprintf(f->names[2*key+1],sizeof f->names[0],
                 "model.layers.%u.mlp.experts.%u.qs",layer,expert);
        /* Reverse the physical records in both shards. The output order must
         * follow numeric (layer, expert), independent of checkpoint offsets. */
        fixture_tensor(f,2*(int)key,f->source.fds[0],3,PACKED,
                       8+SOURCE_HEADER+(KEYS-1-key)*PACKED);
        fixture_tensor(f,2*(int)key+1,f->source.fds[1],2,SCALES,
                       8+SOURCE_HEADER+(KEYS-1-key)*SCALES*4);
        for(size_t j=0;j<PACKED;j++)f->weights[key][j]=(uint8_t)(j*37+(j>>4)*11+key*67);
        for(size_t j=0;j<SCALES;j++)f->scales[key][j]=ldexpf((float)(1+j%31),(int)((j+key)%13)-8);
        f->scales[key][0]=FLT_MIN;
        memcpy(f->scales[key]+1,finite_scale_bits,sizeof finite_scale_bits);
        write_at(f->source.fds[0],f->weights[key],PACKED,f->tensor[2*key].off);
        write_at(f->source.fds[1],f->scales[key],SCALES*4,f->tensor[2*key+1].off);
    }
    strcpy(f->names[2*KEYS],"model.dense.unrelated");
    fixture_tensor(f,2*KEYS,f->source.fds[2],3,16,8+SOURCE_HEADER);
    static const unsigned char dense[16]={1,2,3,4,5,6};
    write_at(f->source.fds[2],dense,sizeof dense,8+SOURCE_HEADER);
    for(int i=0;i<3;i++)fixture_header(f,i);
}
static void no_temporary(Fixture *f) {
    DIR *dir=opendir(f->directory);CHECK(dir);struct dirent *entry;
    while((entry=readdir(dir)))CHECK(!strstr(entry->d_name,".tmp."));
    CHECK(closedir(dir)==0);CHECK(live_allocations==0);
}
static uint8_t *read_file(const char *path,size_t *bytes) {
    int fd=open(path,O_RDONLY);struct stat s;CHECK(fd>=0 && fstat(fd,&s)==0 && s.st_size>=0);
    *bytes=(size_t)s.st_size;uint8_t *result=malloc(*bytes?*bytes:1);CHECK(result);
    CHECK(qes_read_at(fd,result,*bytes,0,error_text,sizeof error_text));CHECK(close(fd)==0);return result;
}
static void preserved(Fixture *f,const uint8_t *before,size_t bytes) {
    size_t got;uint8_t *after=read_file(f->output,&got);
    CHECK(got==bytes && !memcmp(before,after,bytes));free(after);no_temporary(f);
}
/* Independent scalar reference, not the repack implementation under test. */
static void reference_matrix(uint8_t *out,const uint8_t *pairs,int rows,int columns) {
    for(int row=0;row<rows;row++)for(int block=0;block<columns/64;block++)
        for(int k=0;k<32;k++) {
            size_t base=(size_t)row*columns/2+(size_t)block*32;
            unsigned lo=(pairs[base+k/2]>>((k&1)*4))&15;
            unsigned hi=(pairs[base+16+k/2]>>((k&1)*4))&15;
            out[base+k]=(uint8_t)((lo^8)|((hi^8)<<4));
        }
}
static void verify_cache(Fixture *f) {
    uint64_t fp=qes_fingerprint(&f->source,&geometry,error_text,sizeof error_text);CHECK(fp);
    QefFile file={0};CHECK(qef_open(&file,f->output,&geometry,fp,error_text,sizeof error_text));
    CHECK(file.h.record_bytes==RECORD && file.h.count==KEYS);
    uint8_t actual[RECORD],expected[RECORD];size_t matrix=PACKED/3;
    for(unsigned key=0;key<KEYS;key++) {
        CHECK(qef_read(&file,key/EXPERTS,key%EXPERTS,actual,sizeof actual,error_text,sizeof error_text));
        reference_matrix(expected,f->weights[key],INTER,HIDDEN);
        reference_matrix(expected+matrix,f->weights[key]+matrix,INTER,HIDDEN);
        reference_matrix(expected+2*matrix,f->weights[key]+2*matrix,HIDDEN,INTER);
        memcpy(expected+PACKED,f->scales[key],SCALES*4);
        CHECK(!memcmp(actual,expected,sizeof expected));
        CHECK(!memcmp(actual+PACKED+4,finite_scale_bits,sizeof finite_scale_bits));
        uint8_t pad[16384];size_t n=(size_t)(file.h.stride-file.h.record_bytes);
        CHECK(qes_read_at(file.fd,pad,n,file.h.data_offset+(uint64_t)key*file.h.stride+RECORD,error_text,sizeof error_text));
        for(size_t i=0;i<n;i++)CHECK(pad[i]==0);
    }
    uint8_t header_pad[16384];size_t n=(size_t)file.h.data_offset-sizeof(QefHeader);
    CHECK(qes_read_at(file.fd,header_pad,n,sizeof(QefHeader),error_text,sizeof error_text));
    for(size_t i=0;i<n;i++)CHECK(header_pad[i]==0);
    qef_close(&file);no_temporary(f);
}
static void test_success(Fixture *f) {
    allocations=0;largest_allocation=0;
    CHECK(qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
    CHECK(allocations==2 && largest_allocation==RECORD && live_allocations==0);
    verify_cache(f);
    size_t old_bytes;uint8_t *old=read_file(f->output,&old_bytes);
    int old_fd=open(f->output,O_RDONLY);struct stat old_stat,new_stat;
    CHECK(old_fd>=0 && fstat(old_fd,&old_stat)==0);
    f->weights[0][0]^=0x91;
    write_at(f->source.fds[0],f->weights[0],PACKED,f->tensor[0].off);
    CHECK(qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
    CHECK(stat(f->output,&new_stat)==0 && old_stat.st_ino!=new_stat.st_ino);
    uint8_t *held=malloc(old_bytes);CHECK(held);
    CHECK(qes_read_at(old_fd,held,old_bytes,0,error_text,sizeof error_text));
    CHECK(!memcmp(old,held,old_bytes));free(held);free(old);CHECK(close(old_fd)==0);
    verify_cache(f);
}
static void test_failures(Fixture *f) {
    size_t bytes;uint8_t *old=read_file(f->output,&bytes);
    QefGeometry bad=geometry;bad.group=32;
    CHECK(!qec_convert(&f->source,&bad,f->output,error_text,sizeof error_text));preserved(f,old,bytes);
    bad=geometry;bad.hidden=UINT32_MAX-63;
    CHECK(!qec_convert(&f->source,&bad,f->output,error_text,sizeof error_text));preserved(f,old,bytes);
    int original_dtype=f->tensor[3].dtype;f->tensor[3].dtype=1;
    CHECK(!qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
    f->tensor[3].dtype=original_dtype;preserved(f,old,bytes);
    const float bad_scales[]={INFINITY,-INFINITY,NAN};
    for(size_t i=0;i<sizeof bad_scales/sizeof bad_scales[0];i++) {
        write_at(f->source.fds[1],bad_scales+i,4,f->tensor[5].off+10*4);
        CHECK(!qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
        CHECK(strstr(error_text,"invalid scale"));preserved(f,old,bytes);
        write_at(f->source.fds[1],&f->scales[2][10],4,f->tensor[5].off+10*4);
    }
    for(int i=0;i<2;i++) {
        allocation_failure=i;
        CHECK(!qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
        allocation_failure=-1;preserved(f,old,bytes);
    }
    const int failures[]={FAULT_READ,FAULT_WRITE,FAULT_SYNC,FAULT_CLOSE,FAULT_RENAME};
    for(size_t i=0;i<sizeof failures/sizeof failures[0];i++) {
        fault=failures[i];read_countdown=3;write_countdown=4;
        CHECK(!qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
        CHECK(fault==FAULT_NONE && error_text[0]);preserved(f,old,bytes);
    }
    /* First mutate an unsampled expert byte; then a nonexpert shard, which
     * only the full source-fd snapshot (not qes_fingerprint) covers. */
    for(int which=0;which<2;which++) {
        mutate_fd=f->source.fds[which?2:0];
        mutate_offset=which?f->tensor[2*KEYS].off+3:f->tensor[0].off+512;
        unsigned char saved;CHECK(pread(mutate_fd,&saved,1,mutate_offset)==1);
        fault=FAULT_MUTATE;read_countdown=1;
        CHECK(!qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
        CHECK(fault==FAULT_NONE && strstr(error_text,"changed"));preserved(f,old,bytes);
        write_at(mutate_fd,&saved,1,mutate_offset);
    }
    free(old);
}
/* Scales that are a bfloat16 / 7, or the all-zero group's 1e-12 / 7: layout 2,
 * 2 bytes a scale on disk, read back as the same F32 bits. Then the fixture's
 * own scales again, which have no code: layout 1. */
static void test_codes(Fixture *f) {
    float saved[KEYS][SCALES];
    memcpy(saved,f->scales,sizeof saved);
    for(unsigned key=0;key<KEYS;key++) {
        for(size_t j=0;j<SCALES;j++)
            f->scales[key][j]=j%17==3?qef_eps_scale():qef_code_scale((uint16_t)(0x3c00+(key*37+j*5)%0x0800));
        write_at(f->source.fds[1],f->scales[key],SCALES*4,f->tensor[2*key+1].off);
    }
    allocations=0;largest_allocation=0;
    CHECK(qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
    CHECK(allocations==2 && largest_allocation==RECORD && live_allocations==0);
    uint64_t fp=qes_fingerprint(&f->source,&geometry,error_text,sizeof error_text);CHECK(fp);
    QefFile file={0};CHECK(qef_open(&file,f->output,&geometry,fp,error_text,sizeof error_text));
    CHECK(file.h.layout==QEF_LAYOUT_CODES && file.h.record_bytes==PACKED+SCALES*2);
    uint8_t actual[RECORD],expected[RECORD];size_t matrix=PACKED/3;
    for(unsigned key=0;key<KEYS;key++) {
        CHECK(qef_read(&file,key/EXPERTS,key%EXPERTS,actual,sizeof actual,error_text,sizeof error_text));
        reference_matrix(expected,f->weights[key],INTER,HIDDEN);
        reference_matrix(expected+matrix,f->weights[key]+matrix,INTER,HIDDEN);
        reference_matrix(expected+2*matrix,f->weights[key]+2*matrix,HIDDEN,INTER);
        memcpy(expected+PACKED,f->scales[key],SCALES*4);
        CHECK(!memcmp(actual,expected,sizeof expected));
    }
    qef_close(&file);no_temporary(f);
    memcpy(f->scales,saved,sizeof saved);
    for(unsigned key=0;key<KEYS;key++)write_at(f->source.fds[1],f->scales[key],SCALES*4,f->tensor[2*key+1].off);
    CHECK(qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
    verify_cache(f);
}
static void test_aliases(Fixture *f) {
    for(int i=0;i<3;i++) {
        size_t bytes;uint8_t *old=read_file(f->path[i],&bytes);
        CHECK(!qec_convert(&f->source,&geometry,f->path[i],error_text,sizeof error_text));
        size_t after_bytes;uint8_t *after=read_file(f->path[i],&after_bytes);
        CHECK(bytes==after_bytes && !memcmp(old,after,bytes));free(old);free(after);
    }
    char alias[256];snprintf(alias,sizeof alias,"%s/alias.qc",f->directory);
    CHECK(link(f->path[0],alias)==0);
    CHECK(!qec_convert(&f->source,&geometry,alias,error_text,sizeof error_text));CHECK(unlink(alias)==0);
    CHECK(symlink(f->path[1],alias)==0);
    CHECK(!qec_convert(&f->source,&geometry,alias,error_text,sizeof error_text));
    struct stat s;CHECK(lstat(alias,&s)==0 && S_ISLNK(s.st_mode));CHECK(unlink(alias)==0);
    CHECK(symlink("absent-source",alias)==0);
    CHECK(!qec_convert(&f->source,&geometry,alias,error_text,sizeof error_text));CHECK(unlink(alias)==0);
    /* An otherwise unused registered replica is still a checkpoint file. */
    int mirror=open(alias,O_CREAT|O_EXCL|O_RDWR,0600);CHECK(mirror>=0);
    const char marker[]="mirror bytes";write_at(mirror,marker,sizeof marker,0);
    f->source.nrep=1;f->source.mfds[0][0]=mirror;f->source.nmirror[0]=1;
    CHECK(!qec_convert(&f->source,&geometry,alias,error_text,sizeof error_text));
    char after[sizeof marker];CHECK(pread(mirror,after,sizeof after,0)==sizeof after);
    CHECK(!memcmp(marker,after,sizeof marker));
    f->source.nrep=0;f->source.mfds[0][0]=-1;f->source.nmirror[0]=0;
    CHECK(close(mirror)==0 && unlink(alias)==0);no_temporary(f);
}
/* A restart can give the volume another device number: the stamp of a shard stays the
 * same, while another file (inode), a resized one or a touched one get another stamp. */
/* qes_fingerprint itself on the tiny checkpoint's real files: the same stamp
 * with the volume renumbered, another with a new inode. */
static void test_stamp_fingerprint(Fixture *f) {
    uint64_t base=qes_fingerprint(&f->source,&geometry,error_text,sizeof error_text);CHECK(base);
    stamp_device_flip=(dev_t)0x5a5a;
    CHECK(qes_fingerprint(&f->source,&geometry,error_text,sizeof error_text)==base);
    stamp_device_flip=0;stamp_inode_flip=(ino_t)1;
    CHECK(qes_fingerprint(&f->source,&geometry,error_text,sizeof error_text)!=base);
    stamp_inode_flip=0;
    CHECK(qes_fingerprint(&f->source,&geometry,error_text,sizeof error_text)==base);
}
/* A conversion ended by SIGTERM in the middle of its writes removes its
 * temporary and ends by that signal; the file already there stays as it was,
 * with the mode a plain file gets under the umask. */
static void test_signal(Fixture *f) {
    CHECK(qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text));
    struct stat state;CHECK(stat(f->output,&state)==0);
    mode_t mask=umask(0);umask(mask);CHECK((state.st_mode&0777)==(0666&~mask));
    size_t bytes;uint8_t *before=read_file(f->output,&bytes);
    fflush(NULL);
    pid_t child=fork();CHECK(child>=0);
    if(child==0) {
        fault=FAULT_SIGNAL;write_countdown=3;
        qec_convert(&f->source,&geometry,f->output,error_text,sizeof error_text);
        _exit(0);
    }
    int status;CHECK(waitpid(child,&status,0)==child);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status)==SIGTERM);
    preserved(f,before,bytes);free(before);
}
static void test_stamp_device(void) {
    struct stat info;CHECK(stat(".",&info)==0);
    uint64_t a[6],b[6];qes_stat_stamp(&info,a);
    struct stat other=info;other.st_dev^=1;qes_stat_stamp(&other,b);CHECK(!memcmp(a,b,sizeof a));
    other=info;other.st_ino^=1;qes_stat_stamp(&other,b);CHECK(memcmp(a,b,sizeof a));
    other=info;other.st_size+=1;qes_stat_stamp(&other,b);CHECK(memcmp(a,b,sizeof a));
    other=info;other.st_mtime+=1;qes_stat_stamp(&other,b);CHECK(memcmp(a,b,sizeof a));
}
int main(void) {
    test_stamp_device();
    Fixture f;fixture_open(&f);
    test_stamp_fingerprint(&f);
    char fixed_temp[280];snprintf(fixed_temp,sizeof fixed_temp,"%s.tmp",f.output);
    int sentinel=open(fixed_temp,O_CREAT|O_EXCL|O_RDWR,0600);CHECK(sentinel>=0);
    const char marker[]="preexisting .tmp must survive";write_at(sentinel,marker,sizeof marker,0);
    test_success(&f);test_codes(&f);test_failures(&f);test_aliases(&f);test_signal(&f);
    /* Directly constructed tensor inventories without nfd still protect and
     * snapshot the descriptors they actually read. */
    f.source.nfd=0;
    CHECK(qec_convert(&f.source,&geometry,f.output,error_text,sizeof error_text));
    verify_cache(&f);f.source.nfd=3;
    char after[sizeof marker];CHECK(pread(sentinel,after,sizeof after,0)==sizeof after);
    CHECK(!memcmp(marker,after,sizeof marker));CHECK(close(sentinel)==0 && unlink(fixed_temp)==0);
    for(int i=0;i<3;i++){CHECK(close(f.source.fds[i])==0);CHECK(unlink(f.path[i])==0);}
    CHECK(unlink(f.output)==0 && rmdir(f.directory)==0);
    printf("qwen36 expert conversion: %u checks passed\n",checks);return 0;
}
#else
int main(void) {puts("qwen36 expert conversion fixtures require POSIX");return 0;}
#endif
