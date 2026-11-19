/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* qwen38_wex.h -- Qwen3.8's routed experts as MXFP4 records: the AMD Quark checkpoint
 * (amd/Qwen3.8-Flash-Next-Quark-MXFP4) read from a WEX1 container written by
 * tools/repack_q38_mxfp4_wex.py, from a WNR3 container built from it (tools/build_nr3.py),
 * or straight from the Quark shards.
 *
 * Included once by qwen38_core.h after the int4 sidecar's attach. It fills the same
 * Q38Int4Experts index with x->mx = 1, so the RAM LRU, the parallel batch loader, the
 * RAM plan and the MTP split (the head keeps the snapshot's FP8 experts) take its records
 * unchanged. The trunk, the PLE, the shared expert and the MTP head stay the snapshot's.
 *
 * A record is the WEX1 slab, the Quark tensors' bytes unchanged: the E8M0 scales of w1, w2,
 * w3, then their FP4 E2M1 nibbles (w1 = gate_proj, w2 = down_proj, w3 = up_proj; scales
 * [rows, cols/32] and nibbles [rows, cols/2] bytes, low nibble = even column), computed by
 * quant.h's matmul_mxfp4 with f32 activations.
 *
 *   COLI_WEX_PATH=<file>                 the container (WEX1, or WNR3 = WEX3 v3 flag 32); reads
 *                                        are unbuffered and 4096-aligned. A WNR3 record is read
 *                                        into its slab's tail and decoded in place: by the helper
 *                                        kaira_wex_cuda.dll on the GPU (a load wave in one call,
 *                                        feature 32; a single miss through feature 16), else on
 *                                        the CPU (~33 MB/s: a test path)
 *   COLI_WEX_EXTRA_MIRRORS=<dir>[;<dir>]  up to three dirs holding a byte-identical copy under
 *                                        the same file name (a copy that differs is ignored)
 *   COLI_WEX_WEIGHTS=<w0>,<w1>[,...]      read split over primary + mirrors, default equal; an
 *                                        expert always goes to the same drive (hash of layer, expert)
 *   COLI_WEX_STRIPE=1                     with a mirror: a miss read alone (a decode step's lone miss)
 *                                        reads its first part from its hashed drive and the rest from
 *                                        the next one at the same time, cut by COLI_WEX_WEIGHTS on a
 *                                        4096-byte page; default 0 (the A/B switch)
 *   COLI_WEX_ROOT=<64 hex>                pins a WEX4 container's root (tools/wex4_verify prints it):
 *                                        another root, or a container that is not WEX4, is refused
 *   COLI_WEX_VERIFY=1                     WEX4: each record's stored bytes are hashed against the
 *                                        index at their first read (whole and striped reads apart);
 *                                        a mismatch stops the engine. Default 0: no per-record cost
 *   Q38_MXFP4_DIR=<dir>                 the Quark shards themselves, no container (six reads per
 *                                        expert, through the page cache: a check, not the fast path)
 * Either one set is a promise: anything that does not describe this model's experts is
 * refused, never a quiet fall back to FP8. Under COLI_CUDA=1 the CUDA tier takes the
 * decoded records as fmt=7 (qt_init_mxfp4) the way it takes FP8 ones: copied from the RAM
 * slots by qt_note and by the warmstart. COLI_WEX_GPU_DIRECT (deepseek_v4's decode straight
 * into VRAM) has no qwen38 path yet: the tier has no device-buffer fill entry (plan P5). The
 * Vulkan tier has no MXFP4 source; it stays off. */
#ifndef COLI_QWEN38_WEX_H
#define COLI_QWEN38_WEX_H
#include "wex_format.h"
#include "wex_decode.h"
/* One translation unit with qwen38.c: the many Makefile rules that build qwen38.c (the engine
 * and its tests) need no new object for the helper loader and the WNR2/WNR3 CPU decoder.
 * The segment/edge builds link wex_decode.o themselves (SEGMENT_ALL_OBJS): including it there
 * too would define every coli_wex_* symbol twice. */
#ifndef COLI_SEGMENT_ADAPTER
#include "wex_decode.c"
#endif

/* gate, up, down (the Slot's order) -> the record's matrix w1, w3, w2 */
static const int q38_mx4_matrix[3]={0,2,1};

static int64_t q38_mx4_rows(const Cfg *c,int k){ return k<2?c->inter:c->hidden; }
static int64_t q38_mx4_cols(const Cfg *c,int k){ return k<2?c->hidden:c->inter; }

/* The record's layout for this geometry: where each matrix's scales and nibbles start
 * (indexed by WEX matrix w1, w2, w3), and the slab's raw and 4096-aligned bytes. */
static int64_t q38_mx4_layout(const Cfg *c,int64_t scale_off[3],int64_t weight_off[3],int64_t *raw) {
    static const int slot_of[3]={0,2,1};   /* w1 = gate, w2 = down, w3 = up */
    int64_t sb[3],wb[3],at=0;
    for(int w=0;w<3;w++){
        int k=slot_of[w];int64_t rows=q38_mx4_rows(c,k),cols=q38_mx4_cols(c,k);
        sb[w]=rows*((cols+31)/32);wb[w]=rows*((cols+1)/2);
    }
    for(int w=0;w<3;w++){ if(scale_off)scale_off[w]=at; at+=sb[w]; }
    for(int w=0;w<3;w++){ if(weight_off)weight_off[w]=at; at+=wb[w]; }
    if(raw)*raw=at;
    return (at+WEX_ALIGN-1)/WEX_ALIGN*WEX_ALIGN;
}
static int64_t q38_mx4_record_bytes(const Cfg *c){ return q38_mx4_layout(c,NULL,NULL,NULL); }

static const char *q38_mx4_env(const char *name){ const char *v=getenv(name); return v&&*v?v:NULL; }
/* 1 when the run asked for MXFP4 experts (the preload gate sizes the cache floor by them) */
static int q38_mx4_wanted(void){ return q38_mx4_env("COLI_WEX_PATH")||q38_mx4_env("Q38_MXFP4_DIR"); }

static void q38_mx4_refuse(const char *what,const char *why) {
    fprintf(stderr,"[qwen38] MXFP4 experts %s: %s -- refusing (unset COLI_WEX_PATH / Q38_MXFP4_DIR "
                   "for the snapshot's own experts)\n",what,why);
    exit(1);
}

static int q38_wex_open(const char *path) {
#ifdef _WIN32
    HANDLE h=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
                         FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED,NULL);
    if(h==INVALID_HANDLE_VALUE)return -1;
    int fd=_open_osfhandle((intptr_t)h,_O_RDONLY|_O_BINARY);
    if(fd<0)CloseHandle(h);
    return fd;
#elif defined(O_DIRECT)
    int fd=open(path,O_RDONLY|O_DIRECT);
    return fd>=0?fd:open(path,O_RDONLY);   /* a filesystem without O_DIRECT reads through the cache */
#elif defined(__APPLE__)
    return compat_open_direct(path);
#else
    return open(path,O_RDONLY);
#endif
}
static int64_t q38_wex_fsize(int fd) {
#ifdef _WIN32
    return (int64_t)compat_fsize(fd);
#else
    struct stat sb; return fstat(fd,&sb)?-1:(int64_t)sb.st_size;
#endif
}
/* One aligned read of exactly n bytes; 0 on success. */
static int q38_wex_read(int fd,void *buf,int64_t n,int64_t off) {
    return pread(fd,buf,(size_t)n,(off_t)off)==(ssize_t)n?0:-1;
}
/* n bytes at off: the first cut from fd0, the rest from fd1 (a byte-identical mirror), both reads
 * in flight together. 0 on success, else the half that failed first (1: fd0's, 2: fd1's), with
 * errno = EIO and, on Windows, the read's GetLastError in compat_pread_lasterr. */
static int q38_wex_read2(int fd0,int fd1,uint8_t *buf,int64_t n,int64_t cut,int64_t off) {
#ifdef _WIN32
    int fd[2]={fd0,fd1},pending[2]={0,0},bad=0;
    int64_t at[2]={off,off+cut},len[2]={cut,n-cut};
    OVERLAPPED ov[2];HANDLE h[2],ev[2]={NULL,NULL};
    DWORD err=0;
#define Q38_READ2_FAIL(i,code) do{if(!bad){bad=(i)+1;err=(code);}}while(0)
    for(int i=0;i<2;i++){
        intptr_t osfh=_get_osfhandle(fd[i]);
        /* this call's own events, closed below: a thread-local pair would leak per OpenMP thread */
        ev[i]=CreateEventA(NULL,TRUE,FALSE,NULL);
        if(!ev[i]){Q38_READ2_FAIL(i,GetLastError());continue;}
        if(osfh==-1||osfh==-2||len[i]>0x7FFFFFFF){Q38_READ2_FAIL(i,ERROR_INVALID_PARAMETER);continue;}
        h[i]=(HANDLE)osfh;
        memset(&ov[i],0,sizeof ov[i]);
        ov[i].Offset=(DWORD)((uint64_t)at[i]&0xFFFFFFFFu);ov[i].OffsetHigh=(DWORD)((uint64_t)at[i]>>32);
        ov[i].hEvent=ev[i];
        DWORD rd=0;
        if(ReadFile(h[i],buf+(i?cut:0),(DWORD)len[i],&rd,&ov[i])){if(rd!=(DWORD)len[i])Q38_READ2_FAIL(i,ERROR_HANDLE_EOF);}
        else if(GetLastError()==ERROR_IO_PENDING)pending[i]=1;
        else Q38_READ2_FAIL(i,GetLastError());
    }
    for(int i=0;i<2;i++){   /* every read in flight is waited for: buf must not be released under it */
        DWORD rd=0;
        if(pending[i]){
            if(!GetOverlappedResult(h[i],&ov[i],&rd,TRUE))Q38_READ2_FAIL(i,GetLastError());
            else if(rd!=(DWORD)len[i])Q38_READ2_FAIL(i,ERROR_HANDLE_EOF);
        }
        if(ev[i])CloseHandle(ev[i]);
    }
#undef Q38_READ2_FAIL
    if(bad){compat_pread_lasterr=err;errno=EIO;}
    return bad;
#else
    /* shortcut: one read after the other (no overlap), upgrade with a second thread or io_uring if a Linux A/B wants it */
    if(q38_wex_read(fd0,buf,cut,off))return 1;
    return q38_wex_read(fd1,buf+cut,n-cut,off+cut)?2:0;
#endif
}
/* The stripe's cut for a record of n bytes whose hashed drive is r and second drive s: the two
 * drives' COLI_WEX_WEIGHTS shares, on a 4096-byte page; 0 when the record is one page. */
static int64_t q38_wex_stripe_cut(const int *wex_cut,int r,int s,int64_t n) {
    double wr=wex_cut[r]-(r?wex_cut[r-1]:0),ws=wex_cut[s]-(s?wex_cut[s-1]:0);
    int64_t pages=n/WEX_ALIGN;
    if(n%WEX_ALIGN||pages<2||!(wr+ws>0))return 0;
    int64_t first=(int64_t)((double)pages*wr/(wr+ws)+0.5);
    if(first<1)first=1;
    if(first>pages-1)first=pages-1;
    return first*WEX_ALIGN;
}

#include "wex4.h"
/* WEX4 container (wex4.h): the header page and the index are validated, COLI_WEX_ROOT pins the
 * root, and the caller gets what a WEX1 / WNR3 container gives it: the WexEntry table and a
 * header page with the legacy fields (for WNR3 also the WNMD descriptor of the model blob, whose
 * SHA-256 is checked here). The page keeps the root at byte 4064, so equal pages <=> same
 * container. Record bytes and offsets are those of the old container. */
static int q38_wex4_manifest(int fd,const Cfg *c,unsigned char *page,unsigned char **table,
                             size_t *table_bytes,int64_t file_bytes) {
    Wex4Header h;Wex4View v;Wex4Record rec;Wex4Blob blob;
    void *index=NULL,*t=NULL,*model=NULL;
    uint64_t span,slab=0;uint32_t r,flags=0,model_blob=0;
    const char *pin=getenv("COLI_WEX_ROOT");
    int bad=1;
    if(wex4_parse_header(page,(uint64_t)file_bytes,&h)||h.layers!=(uint32_t)c->layers||h.experts!=(uint32_t)c->experts)return -1;
    if(pin&&*pin){
        char hex[65];
        for(r=0;r<32u;r++)snprintf(hex+r*2u,3,"%02x",h.root[r]);
        if(strcasecmp(pin,hex)){fprintf(stderr,"[qwen38] WEX4 root %s is not the pinned COLI_WEX_ROOT\n",hex);return -1;}
    }
    span=wex4_align((uint64_t)h.layers*h.experts*sizeof(WexEntry));
    /* the record hashes sit right behind the table (span bytes), 32 per slot (COLI_WEX_VERIFY) */
    uint64_t shas=wex4_align((uint64_t)h.layers*h.experts*32u);
    if(posix_memalign(&index,WEX_ALIGN,(size_t)wex4_align(h.index_bytes)+WEX_ALIGN)||posix_memalign(&t,WEX_ALIGN,(size_t)(span+shas)))goto done;
    if(q38_wex_read(fd,index,(int64_t)wex4_align(h.index_bytes),(int64_t)h.index_offset)||
       wex4_open(&h,(const unsigned char*)index,(size_t)wex4_align(h.index_bytes),&v))goto done;
    memset(t,0,(size_t)(span+shas));
    for(r=0;r<h.records;r++){
        wex4_get_record(&v,r,&rec);
        if(!(rec.flags&WEX4_REC_EXPERT))continue;
        size_t slot=(size_t)(rec.owner>>16)*h.experts+(rec.owner&0xffffu);
        memcpy((unsigned char*)t+span+slot*32u,rec.sha,32u);
        if(wex4_expert_entry(&v,r,&((WexEntry*)t)[slot]))goto done;
        if(wex4_align(rec.decoded)>slab)slab=wex4_align(rec.decoded);
        if(rec.flags&WEX4_REC_VALUE_EXACT)flags|=WEX_HEADER_CANONICAL_ZERO;
        if(rec.codec==WEX4_CODEC_WNR3){
            flags|=WEX_HEADER_NEURAL_RANS3;
            if(model_blob&&model_blob!=rec.blob)goto done;   /* one registered model */
            model_blob=rec.blob;
        }
    }
    for(r=0;(flags&WEX_HEADER_CANONICAL_ZERO)&&r<h.records;r++){   /* canonical zero is per container here */
        wex4_get_record(&v,r,&rec);
        if(rec.codec==WEX4_CODEC_WNR3&&!(rec.flags&WEX4_REC_VALUE_EXACT))goto done;
    }
    memset(&blob,0,sizeof blob);
    if(model_blob){
        unsigned char digest[32];uint64_t at;
        wex4_get_blob(&v,model_blob-1u,&blob);
        if(posix_memalign(&model,WEX_ALIGN,(size_t)wex4_align(blob.bytes))||
           q38_wex_read(fd,model,(int64_t)wex4_align(blob.bytes),(int64_t)blob.offset))goto done;
        wex4_sha256((const unsigned char*)model,(size_t)blob.bytes,digest);
        if(memcmp(digest,blob.sha,32u))goto done;
        for(at=blob.bytes;at<wex4_align(blob.bytes);at++)if(((const unsigned char*)model)[at])goto done;
    }
    {
        WexHeader legacy={model_blob?WEX_MAGIC_V3:WEX_MAGIC_V1,model_blob?3u:1u,h.layers,h.experts,slab,
                          model_blob?blob.offset+wex4_align(blob.bytes):WEX_ALIGN,flags,0u};
        memset(page,0,WEX_ALIGN);
        memcpy(page,&legacy,sizeof legacy);
        if(model_blob){   /* where q38_wex_model looks for the model */
            uint32_t words[6]={WEX_NR2_DESCRIPTOR_MAGIC,(uint32_t)blob.bytes,(uint32_t)blob.offset,(uint32_t)(blob.offset>>32),blob.id,0u};
            memcpy(page+WEX_NR2_DESCRIPTOR_OFFSET,words,sizeof words);
        }
        uint32_t marker=WEX4_MAGIC;
        memcpy(page+56u,&marker,4u);   /* tells the caller the hashes are behind the table */
        memcpy(page+WEX4_HEADER_HASHED,h.root,32u);
    }
    fprintf(stderr,"[qwen38] WEX4 container (hashed header and index)\n");
    *table=(unsigned char*)t;*table_bytes=(size_t)h.layers*h.experts*sizeof(WexEntry);t=NULL;bad=0;
done:
    if(index)compat_aligned_free(index);
    if(model)compat_aligned_free(model);
    if(t)compat_aligned_free(t);
    return bad?-1:0;
}

/* The header page and the table of one container (aligned buffers, the caller frees
 * both with compat_aligned_free); 0 when they are this model's WEX1. *wex4 (when not NULL)
 * is 1 for a WEX4 file, whose page comes back rewritten in the WEX1/WNR3 shape. */
static int q38_wex_manifest(int fd,const Cfg *c,unsigned char **page,unsigned char **table,
                            size_t *table_bytes,int64_t *file_bytes,int *wex4) {
    size_t bytes=0,span=0;void *p=NULL,*t=NULL;
    *page=*table=NULL;*file_bytes=q38_wex_fsize(fd);
    if(wex4)*wex4=0;
    if(*file_bytes<WEX_ALIGN||posix_memalign(&p,WEX_ALIGN,WEX_ALIGN))return -1;
    *page=(unsigned char*)p;
    if(q38_wex_read(fd,p,WEX_ALIGN,0))return -1;
    if(wex_read_u32_le((const unsigned char*)p)==WEX4_MAGIC){
        if(wex4)*wex4=1;
        return q38_wex4_manifest(fd,c,(unsigned char*)p,table,table_bytes,*file_bytes);
    }
    if(q38_mx4_env("COLI_WEX_ROOT")){   /* a pinned root refuses a legacy file: it has no root (W06) */
        fprintf(stderr,"[qwen38] a container that is not WEX4 is not the pinned COLI_WEX_ROOT\n");
        return -1;
    }
    const WexHeader *h=(const WexHeader*)p;
    int wex1=h->magic==WEX_MAGIC_V1&&h->version==1u;
    int wnr3=h->magic==WEX_MAGIC_V3&&h->version==3u&&(h->flags&WEX_HEADER_NEURAL_RANS3);
    if(!(wex1||wnr3)||wex_validate_header(h,(uint64_t)*file_bytes,c->layers,c->experts,&bytes,&span))return -1;
    if(posix_memalign(&t,WEX_ALIGN,span))return -1;
    *table=(unsigned char*)t;*table_bytes=bytes;
    /* trust boundary: every entry's offsets, sizes and codec against the header and the file */
    return q38_wex_read(fd,t,(int64_t)span,(int64_t)h->table_offset)||
           wex_validate_table(h,(const WexEntry*)t,(uint64_t)*file_bytes)?-1:0;
}

/* WNR3: the container's WNM2 model, registered with the CPU decoder and the helper. */
static void q38_wex_model(int fd,const unsigned char *page,const char *path) {
    uint64_t at=0,aligned=0;uint32_t bytes=0,id=0,got=0;void *blob=NULL;
    int bad=wex_nr2_model_location((const WexHeader*)page,page,&at,&bytes,&id)||wex_aligned_size(bytes,&aligned)||
            posix_memalign(&blob,WEX_ALIGN,(size_t)aligned)||q38_wex_read(fd,blob,(int64_t)aligned,(int64_t)at)||
            coli_wex_register_model((const unsigned char*)blob,bytes,&got)||got!=id;
    if(blob)compat_aligned_free(blob);
    if(bad)q38_mx4_refuse(path,"its WNM2 model does not load");
}

/* One entry is this model's MXFP4 record: raw (codec 0) or, in a WNR3 container, codec 5
 * (wex_validate_table has checked its place in the file and its size against its codec). */
static int q38_wex_entry_ok(const WexEntry *e,const Cfg *c,const int64_t scale_off[3],
                            const int64_t weight_off[3],int64_t raw,int64_t record,int nr3) {
    uint64_t r=0,a=0;
    int raw_ok=!e->flags&&e->total_bytes==(uint32_t)record;
    int nr3_ok=nr3&&e->flags==WEX_ENTRY_NEURAL_RANS3&&e->total_bytes<(uint32_t)record;
    if(!(raw_ok||nr3_ok)||wex_entry_raw_sizes(e,&r,&a)||(int64_t)r!=raw||(int64_t)a!=record)return 0;
    static const int slot_of[3]={0,2,1};
    for(int w=0;w<3;w++){
        int k=slot_of[w];
        if(e->weight_shapes[w][0]!=q38_mx4_rows(c,k)||e->weight_shapes[w][1]*2!=q38_mx4_cols(c,k)||
           e->scale_offsets[w]!=(uint32_t)scale_off[w]||e->weight_offsets[w]!=(uint32_t)weight_off[w])return 0;
    }
    return 1;
}

static void q38_mx4_attach_wex(Model *m,Q38Int4Experts *x,const char *path) {
    Cfg *c=&m->c;
    int fd=q38_wex_open(path);
    if(fd<0)q38_mx4_refuse(path,"cannot be opened");
    unsigned char *page,*table;size_t table_bytes;int64_t file_bytes;
    if(q38_wex_manifest(fd,c,&page,&table,&table_bytes,&file_bytes,&x->wex4))
        q38_mx4_refuse(path,"is not a WEX1 or WNR3 container of this model's layers and experts "
                            "(WEX2, palette and WNR/WNR2 records are not read here)");
    const WexHeader *h=(const WexHeader*)page;
    x->wex_flags=h->flags;x->nr3=!!(h->flags&WEX_HEADER_NEURAL_RANS3);
    if(x->nr3){
        q38_wex_model(fd,page,path);
        if(coli_wex_container_refused(h->flags,"qwen38"))q38_mx4_refuse(path,"canonical-zero container refused");
        if(!coli_wex_cuda_nr3_available())
            fprintf(stderr,"[qwen38] WNR3 container without kaira_wex_cuda.dll (helper feature 16) next to the "
                           "engine: every miss decodes on the CPU (~33 MB/s, a test path)\n");
    }
    const WexEntry *entries=(const WexEntry*)table;
    for(int layer=m->range_begin;layer<m->range_end;layer++)
        for(int expert=0;expert<c->experts;expert++)
            if(!q38_wex_entry_ok(&entries[(int64_t)layer*c->experts+expert],c,x->mx_scale_off,x->mx_weight_off,
                                 x->mx_raw,x->record_bytes,x->nr3)){
                char why[200];
                snprintf(why,sizeof why,"layer %d expert %d is missing or is not one MXFP4 record of %lld bytes "
                         "(codec 0, or codec 5 in a WNR3 container)",layer,expert,(long long)x->mx_raw);
                q38_mx4_refuse(path,why);
            }
    x->wex_fd[0]=fd;x->wex_n=1;
    {   /* COLI_WEX_VERIFY=1: each record's stored bytes are hashed at their first read (per layout) */
        uint32_t marker;const char *verify=q38_mx4_env("COLI_WEX_VERIFY");
        memcpy(&marker,page+56u,4u);
        if(verify&&atoi(verify)){
            if(marker==WEX4_MAGIC){
                x->wex_seen=(unsigned char*)calloc((size_t)c->layers*c->experts,1u);
                if(!x->wex_seen){fprintf(stderr,"OOM WEX4 first-read marks\n");exit(1);}
                x->wex_sha=table+wex4_align(table_bytes);
            }
            fprintf(stderr,"[qwen38] WEX4 first-read SHA-256 of every record: %s\n",
                    x->wex_sha?"on (COLI_WEX_VERIFY=1)":"off (not a WEX4 container: no record hashes)");
        }
    }
    /* mirrors: the same file name in each dir, header page and table byte-identical */
    const char *list=q38_mx4_env("COLI_WEX_EXTRA_MIRRORS");
    const char *base=path+strlen(path);
    while(base>path&&base[-1]!='/'&&base[-1]!='\\')base--;
    char dirs[4096];snprintf(dirs,sizeof dirs,"%s",list?list:"");
    for(char *dir=dirs,*next;dir&&*dir&&x->wex_n<4;dir=next){
        char *sep=dir;while(*sep&&*sep!=';'&&*sep!=',')sep++;
        next=*sep?sep+1:NULL;*sep=0;
        while(*dir==' ')dir++;
        if(!*dir)continue;
        char mirror[8192];snprintf(mirror,sizeof mirror,"%s/%s",dir,base);
        int mfd=q38_wex_open(mirror);
        if(mfd<0){fprintf(stderr,"[qwen38] WEX mirror %s cannot be opened; ignored\n",mirror);continue;}
        unsigned char *mpage,*mtable;size_t mbytes;int64_t msize;
        int same=!q38_wex_manifest(mfd,c,&mpage,&mtable,&mbytes,&msize,NULL)&&msize==file_bytes&&
                 mbytes==table_bytes&&!memcmp(mpage,page,WEX_ALIGN)&&!memcmp(mtable,table,table_bytes);
        compat_aligned_free(mpage);compat_aligned_free(mtable);
        if(!same){fprintf(stderr,"[qwen38] WEX mirror %s differs from %s; ignored\n",mirror,path);close(mfd);continue;}
        x->wex_fd[x->wex_n++]=mfd;
        fprintf(stderr,"[qwen38] WEX mirror active: %s\n",mirror);
    }
    compat_aligned_free(page);
    x->wex_entry=(WexEntry*)table;   /* kept: the reads and the decoder need each entry */
    /* the read split: drive r takes the hash bytes below cut[r] */
    double w[4]={1,1,1,1},sum=0;int custom=0;
    const char *weights=q38_mx4_env("COLI_WEX_WEIGHTS");
    if(weights){
        char b[128];snprintf(b,sizeof b,"%s",weights);int k=0,bad=0;
        for(char *t=strtok(b,", ");t;t=strtok(NULL,", ")){
            double v=atof(t);
            if(!(v>0)||k>=4){bad=1;break;}
            w[k++]=v;
        }
        custom=!bad&&k==x->wex_n;
        if(!custom)fprintf(stderr,"[qwen38] COLI_WEX_WEIGHTS '%s' ignored (want %d positive weights)\n",
                           weights,x->wex_n);
    }
    if(!custom)for(int r=0;r<4;r++)w[r]=1;
    for(int r=0;r<x->wex_n;r++)sum+=w[r];
    double acc=0;
    for(int r=0;r<x->wex_n;r++){ acc+=w[r]; x->wex_cut[r]=(int)(acc/sum*256.0+0.5); }
    x->wex_cut[x->wex_n-1]=256;
    const char *stripe=q38_mx4_env("COLI_WEX_STRIPE");
    x->wex_stripe=x->wex_n>1&&stripe&&stripe[0]=='1';
    fprintf(stderr,"[qwen38] MXFP4 experts from %s %s: %lld-byte records, direct reads, %d drive(s)",
            x->wex4?(x->nr3?"WEX4 (WNR3 records)":"WEX4"):x->nr3?"WNR3":"WEX1",path,(long long)x->record_bytes,x->wex_n);
    for(int r=0;r<x->wex_n&&x->wex_n>1;r++)fprintf(stderr,"%s%.0f%%",r?" ":" split ",100.0*w[r]/sum);
    if(x->wex_stripe)fprintf(stderr,", a lone miss striped over two drives (COLI_WEX_STRIPE=1)");
    fprintf(stderr,"\n");
}

/* Quark tensor `name` of one matrix; NULL unless dtype and shape are this model's. */
static st_tensor *q38_mx4_quark_tensor(Model *m,Q38Int4Experts *x,int layer,int expert,int k,int scale) {
    static const char *proj[3]={"gate_proj","up_proj","down_proj"};
    char suffix[192],name[320];
    snprintf(suffix,sizeof suffix,"mlp.experts.%d.%s.weight%s",expert,proj[k],scale?"_scale":"");
    q38_name(m,name,sizeof name,layer,suffix);
    st_tensor *t=st_find(&x->S,name);
    int64_t rows=q38_mx4_rows(&m->c,k),cols=q38_mx4_cols(&m->c,k);
    int64_t width=scale?(cols+31)/32:(cols+1)/2;
    /* trust boundary: U8 nibbles, U8 or F8_E8M0 scales, exactly this geometry */
    if(!t||(t->dtype!=3&&!(scale&&t->dtype==5))||t->rank!=2||t->shape[0]!=rows||t->shape[1]!=width||
       t->nbytes!=rows*width)return NULL;
    return t;
}

static void q38_mx4_attach_quark(Model *m,Q38Int4Experts *x,const char *dir) {
    st_init(&x->S,dir);
    for(int layer=m->range_begin;layer<m->range_end;layer++)
        for(int expert=0;expert<m->c.experts;expert++)
            for(int part=0;part<6;part++)
                if(!q38_mx4_quark_tensor(m,x,layer,expert,part%3,part>=3)){
                    char why[200];
                    snprintf(why,sizeof why,"layer %d expert %d has no U8 Quark MXFP4 %s of this geometry "
                             "(gate/up/down_proj.weight + weight_scale)",layer,expert,part>=3?"scale":"weight");
                    q38_mx4_refuse(dir,why);
                }
    fprintf(stderr,"[qwen38] MXFP4 experts from the Quark shards in %s: %lld-byte records "
                   "(no container: six reads per expert)\n",dir,(long long)x->record_bytes);
}

/* COLI_WEX_PATH or Q38_MXFP4_DIR: the routed experts of the model's layers as MXFP4 records. */
static void q38_expert_mx4_attach(Model *m) {
    const char *path=q38_mx4_env("COLI_WEX_PATH"),*dir=q38_mx4_env("Q38_MXFP4_DIR");
    if(!path&&!dir)return;
    if(path&&dir)q38_mx4_refuse(path,"COLI_WEX_PATH and Q38_MXFP4_DIR are both set; pick one");
    Q38Int4Experts *x=(Q38Int4Experts*)calloc(1,sizeof(*x));
    if(!x){fprintf(stderr,"OOM MXFP4 expert index\n");exit(1);}
    x->mx=1;
    x->record_bytes=q38_mx4_layout(&m->c,x->mx_scale_off,x->mx_weight_off,&x->mx_raw);
    if(x->record_bytes>UINT32_MAX)q38_mx4_refuse(path?path:dir,"an expert of this geometry exceeds a WEX record");
    if(path)q38_mx4_attach_wex(m,x,path); else q38_mx4_attach_quark(m,x,dir);
    const char *direct=q38_mx4_env("COLI_WEX_GPU_DIRECT");
    if(path&&direct&&strcmp(direct,"0"))
        fprintf(stderr,"[qwen38] COLI_WEX_GPU_DIRECT=%s: qwen38 has no direct-to-VRAM fill yet; records decode "
                       "into RAM slots and the CUDA tier copies them from there\n",direct);
    m->x4=x;
}

/* Point a slot's three matrices into one record. */
static void q38_mx4_bind(Model *m,Slot *slot,uint8_t *record) {
    const Q38Int4Experts *x=m->x4;
    Q38Weight *dst[3]={&slot->gate,&slot->up,&slot->down};
    for(int k=0;k<3;k++){
        Q38Weight *weight=dst[k];int w=q38_mx4_matrix[k];q38_weight_free(weight);
        weight->rows=(int)q38_mx4_rows(&m->c,k);weight->cols=(int)q38_mx4_cols(&m->c,k);
        weight->data=record+x->mx_weight_off[w];weight->e8=record+x->mx_scale_off[w];
        weight->elements=(int64_t)weight->rows*weight->cols;
        weight->scale_count=(int64_t)weight->rows*((weight->cols+31)/32);
        weight->kind=Q38_WEIGHT_MXFP4;
    }
}

/* The slot's 4096-aligned record inside its slab (an over-allocation, so the plain free()
 * of every slot teardown stays right). */
static uint8_t *q38_mx4_record(const Slot *slot) {
    return (uint8_t*)(((uintptr_t)slot->int4_slab+WEX_ALIGN-1)&~(uintptr_t)(WEX_ALIGN-1));
}
static const WexEntry *q38_mx4_entry(const Model *m,int layer,int expert) {
    return &m->x4->wex_entry[(int64_t)layer*m->c.experts+expert];
}
/* A WNR3 record read into its slab's tail becomes the raw slab in place (helper or CPU). */
static void q38_mx4_decode(Model *m,int layer,int expert,uint8_t *record) {
    if(coli_wex_decode_record(q38_mx4_entry(m,layer,expert),record,(size_t)m->x4->record_bytes)){
        fprintf(stderr,"[qwen38] WNR3 record of layer %d expert %d does not decode (corrupt container)\n",layer,expert);
        exit(1);
    }
}

/* One miss into the slot's own slab. Safe from the parallel batch loader: every job owns a
 * distinct slot, and a read is positional. defer = 1 leaves a WNR3 record undecoded in the
 * slab's tail for the load wave's single helper call (q38_mx4_decode_wave). alone = 1: no other
 * miss reads beside this one, so COLI_WEX_STRIPE may give it a second drive. */
static void q38_mx4_load(Model *m,int layer,int expert,Slot *slot,int defer,int alone) {
    Q38Int4Experts *x=m->x4;
    if(!slot->int4_slab||slot->int4_slab_bytes!=x->record_bytes){
        q38_slab_unpin(slot);
        void *replacement=realloc(slot->int4_slab,(size_t)x->record_bytes+WEX_ALIGN);
        if(!replacement){fprintf(stderr,"OOM MXFP4 expert record\n");exit(1);}
        slot->int4_slab=replacement;slot->int4_slab_bytes=x->record_bytes;
    }
    uint8_t *record=q38_mx4_record(slot);
    q38_mx4_bind(m,slot,record);
    if(x->wex_n){
        int64_t at=(int64_t)layer*m->c.experts+expert;
        const WexEntry *e=q38_mx4_entry(m,layer,expert);
        size_t tail=wex_record_read_offset(e);
        uint32_t h=(uint32_t)at*2654435761u;int hv=(int)((h>>16)&255),r=0;
        while(hv>=x->wex_cut[r])r++;
        int s=(r+1)%x->wex_n;
        int64_t cut=alone&&x->wex_stripe?q38_wex_stripe_cut(x->wex_cut,r,s,(int64_t)e->total_bytes):0;
        int bad=cut?q38_wex_read2(x->wex_fd[r],x->wex_fd[s],record+tail,(int64_t)e->total_bytes,cut,(int64_t)e->file_offset):
                    q38_wex_read(x->wex_fd[r],record+tail,e->total_bytes,(int64_t)e->file_offset)?1:0;
        if(bad){
            int drive=bad==2?s:r;int64_t where=(int64_t)e->file_offset+(bad==2?cut:0);   /* the half that failed */
            unsigned long code=0;
#ifdef _WIN32
            code=(unsigned long)compat_pread_lasterr;
#endif
            fprintf(stderr,"[qwen38] WEX read of layer %d expert %d failed (drive %d, offset %lld%s): %s (Windows error %lu)\n",
                    layer,expert,drive,(long long)where,cut?", striped":"",strerror(errno),code);exit(1);
        }
        unsigned char layout=cut?2u:1u;   /* a striped read takes its tail from another drive */
        if(x->wex_sha&&!(__atomic_load_n(&x->wex_seen[at],__ATOMIC_RELAXED)&layout)){
            unsigned char digest[32];
            wex4_sha256(record+tail,e->total_bytes,digest);
            if(memcmp(digest,x->wex_sha+(size_t)at*32u,32u)){
                fprintf(stderr,"[qwen38] WEX4 record of layer %d expert %d (drive %d%s) does not match its SHA-256 "
                               "(COLI_WEX_VERIFY=1): corrupt container\n",layer,expert,r,cut?", striped":"");
                exit(1);
            }
            __atomic_fetch_or(&x->wex_seen[at],layout,__ATOMIC_RELAXED);
        }
        if(!e->flags)return;
        if(!wex_nr3_owner_ok(e,record+tail,(uint32_t)layer,(uint32_t)expert,x->wex_flags)){
            fprintf(stderr,"[qwen38] WNR3 record at layer %d expert %d belongs to another expert (corrupt container)\n",
                    layer,expert);exit(1);
        }
        if(!defer)q38_mx4_decode(m,layer,expert,record);
        return;
    }
    for(int k=0;k<3;k++)
        for(int scale=0;scale<2;scale++){
            st_tensor *t=q38_mx4_quark_tensor(m,x,layer,expert,k,scale);
            int64_t off=scale?x->mx_scale_off[q38_mx4_matrix[k]]:x->mx_weight_off[q38_mx4_matrix[k]];
            if(!t){fprintf(stderr,"Quark MXFP4 tensor of layer %d expert %d changed after attach\n",layer,expert);exit(1);}
            st_read_range_raw_cap(&x->S,t->fd,t->off,t->nbytes,record+off,x->mx_raw-off,1,
                                  "pread Qwen3.8 Quark MXFP4 expert");
        }
}

/* 1 when a load wave of n misses reads every WNR3 record first and decodes them all in one
 * helper call (feature 32) */
static int q38_mx4_wave(const Model *m,int n) {
    return n>1&&m->x4&&m->x4->mx&&m->x4->nr3&&coli_wex_cuda_nr3_available()&&coli_wex_cuda_batch_host_available();
}

/* The wave's decode: each record sits in its own slot's tail, and the helper copies it to the
 * device before it copies the slab back over it (one lane's streams, in order). A record the
 * helper did not decode may have a half-written slab: it is read and decoded again alone.
 * shortcut: device 0 and pageable slabs (the helper blocks while it queues); pin the slots
 * (coli_wex_cuda_host_register) or pick the tier's card when the speed A/B shows it matters. */
static void q38_mx4_decode_wave(Model *m,int layer,int n,const int *experts,Slot *const *slots) {
    static int said;
    if(!__atomic_exchange_n(&said,1,__ATOMIC_RELAXED))
        fprintf(stderr,"[qwen38] WNR3: each load wave's records decode on the GPU in one helper call (feature 32)\n");
    const WexEntry **entries=(const WexEntry**)calloc((size_t)n,sizeof(*entries));
    const unsigned char **records=(const unsigned char**)calloc((size_t)n,sizeof(*records));
    unsigned char **dst=(unsigned char**)calloc((size_t)n,sizeof(*dst));
    int *status=(int*)calloc((size_t)n,sizeof(int)),*job=(int*)calloc((size_t)n,sizeof(int)),k=0;
    if(!entries||!records||!dst||!status||!job){fprintf(stderr,"OOM WNR3 decode wave\n");exit(1);}
    for(int i=0;i<n;i++){
        const WexEntry *e=q38_mx4_entry(m,layer,experts[i]);
        if(!e->flags)continue;
        uint8_t *record=q38_mx4_record(slots[i]);
        entries[k]=e;records[k]=record+wex_record_read_offset(e);dst[k]=record;job[k++]=i;
    }
    if(k&&coli_wex_cuda_decode_batch_to_host(k,entries,records,0,dst,status))
        for(int j=0;j<k;j++)
            if(status[j])q38_mx4_load(m,layer,experts[job[j]],slots[job[j]],0,0);
    free(entries);free(records);free(dst);free(status);free(job);
}

static void q38_mx4_free(Q38Int4Experts *x) {
    for(int r=0;r<x->wex_n;r++)close(x->wex_fd[r]);
    if(x->wex_entry)compat_aligned_free(x->wex_entry);
    free(x->wex_seen);
}
#endif /* COLI_QWEN38_WEX_H */
