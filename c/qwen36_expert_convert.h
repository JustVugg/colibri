/* Offline expert cache writer. No inference, quantization or model-sized
 * allocation: one expert record is read, repacked in place, then written.
 * Conversion is POSIX-only; including this header remains valid on Windows. */
#ifndef QWEN36_EXPERT_CONVERT_H
#define QWEN36_EXPERT_CONVERT_H

#include <stdarg.h>
#include "qwen36_expert_source.h"
#include "expert_ffn.h"
#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static int qec_error(char *error,size_t cap,const char *fmt,...) {
    if(error && cap) {
        va_list args;va_start(args,fmt);vsnprintf(error,cap,fmt,args);va_end(args);
    }
    return 0;
}

#ifndef _WIN32
/* Capture all indexed primary shards, including ones without an expert,
 * tensor descriptors supplied directly, and registered read replicas.
 * Metadata comparisons include nanoseconds and ctime, so a writer cannot
 * conceal an ordinary in-place update by restoring the old mtime. */
#define QEC_MAX_SOURCES (ST_MAX_SHARDS*(1+ST_MAX_MIR))
typedef struct {
    int count,primary_count;
    int fd[QEC_MAX_SOURCES];
    struct stat state[QEC_MAX_SOURCES];
} QecSnapshot;

static int qec_same_stat(const struct stat *a,const struct stat *b) {
    if(a->st_dev!=b->st_dev || a->st_ino!=b->st_ino ||
       a->st_size!=b->st_size || a->st_mode!=b->st_mode ||
       a->st_mtime!=b->st_mtime || a->st_ctime!=b->st_ctime)return 0;
#ifdef __APPLE__
    return a->st_mtimespec.tv_nsec==b->st_mtimespec.tv_nsec &&
           a->st_ctimespec.tv_nsec==b->st_ctimespec.tv_nsec;
#else
    return a->st_mtim.tv_nsec==b->st_mtim.tv_nsec &&
           a->st_ctim.tv_nsec==b->st_ctim.tv_nsec;
#endif
}

static int qec_snapshot_add(QecSnapshot *saved,int fd,char *error,size_t cap) {
    for(int i=0;i<saved->count;i++)if(saved->fd[i]==fd)return 1;
    if(saved->count==QEC_MAX_SOURCES)return qec_error(error,cap,"too many source descriptors");
    if(fd<0)return qec_error(error,cap,"invalid source shard descriptor");
    struct stat *state=&saved->state[saved->count];
    if(fstat(fd,state))return qec_error(error,cap,"cannot stat source descriptor %d: %s",fd,strerror(errno));
    if(!S_ISREG(state->st_mode))return qec_error(error,cap,"source descriptor %d is not a regular file",fd);
    saved->fd[saved->count++]=fd;
    return 1;
}

static int qec_snapshot(shards *S,QecSnapshot *saved,char *error,size_t cap) {
    if(!S || S->nfd<0 || S->nfd>ST_MAX_SHARDS || S->n<=0 || !S->t ||
       S->nrep<0 || S->nrep>ST_MAX_MIR)
        return qec_error(error,cap,"invalid source shard inventory");
    saved->count=0;saved->primary_count=S->nfd;
    for(int i=0;i<S->nfd;i++)if(!qec_snapshot_add(saved,S->fds[i],error,cap))return 0;
    for(int i=0;i<S->n;i++)if(!qec_snapshot_add(saved,S->t[i].fd,error,cap))return 0;
    for(int r=0;r<S->nrep;r++)for(int i=0;i<S->nfd;i++)
        if(S->mfds[r][i]>=0 && !qec_snapshot_add(saved,S->mfds[r][i],error,cap))return 0;
    return 1;
}

static int qec_unchanged(shards *S,const QecSnapshot *saved,char *error,size_t cap) {
    if(S->nfd!=saved->primary_count)return qec_error(error,cap,"source shard inventory changed during conversion");
    for(int i=0;i<saved->count;i++) {
        struct stat now;
        if(fstat(saved->fd[i],&now) ||
           !qec_same_stat(&saved->state[i],&now))
            return qec_error(error,cap,"source shard %d changed during conversion",i);
    }
    return 1;
}

/* Check immediately before publication as well as before any conversion.
 * Reject symlink outputs, including dangling ones; never follow one while
 * deciding whether a checkpoint is a safe destination. */
static int qec_output_safe(const char *output,const QecSnapshot *saved,
                           char *error,size_t cap) {
    struct stat target;
    if(lstat(output,&target)) {
        if(errno==ENOENT)return 1;
        return qec_error(error,cap,"cannot inspect output %s: %s",output,strerror(errno));
    }
    if(S_ISLNK(target.st_mode))return qec_error(error,cap,"refusing symlink output %s",output);
    if(!S_ISREG(target.st_mode))return qec_error(error,cap,"output is not a regular file: %s",output);
    for(int i=0;i<saved->count;i++)
        if(target.st_dev==saved->state[i].st_dev && target.st_ino==saved->state[i].st_ino)
            return qec_error(error,cap,"output aliases source shard %d: %s",i,output);
    return 1;
}

/* A conversion stopped by a signal (Ctrl-C, a closed terminal, kill, a file size
 * limit) removes its temporary, 17 GB on the 35B model, then ends as the signal
 * would have ended it. Only signals left at their default action are taken:
 * a program with its own handlers keeps them. SIGKILL and a stopped machine
 * leave the temporary, <output>.tmp.XXXXXX, which can be deleted. */
static const int qec_signals[]={SIGINT,SIGTERM,SIGHUP,SIGXFSZ};
static struct sigaction qec_previous[sizeof qec_signals/sizeof qec_signals[0]];
static int qec_taken[sizeof qec_signals/sizeof qec_signals[0]];
static char qec_signal_path[4096];
static volatile sig_atomic_t qec_signal_armed;

static void qec_on_signal(int number) {
    if(qec_signal_armed)unlink(qec_signal_path);
    for(size_t i=0;i<sizeof qec_signals/sizeof qec_signals[0];i++)
        if(qec_signals[i]==number)sigaction(number,&qec_previous[i],NULL);
    raise(number);
}

static void qec_arm(const char *temporary) {
    size_t length=strlen(temporary);
    if(length>=sizeof qec_signal_path)return;
    memcpy(qec_signal_path,temporary,length+1);
    qec_signal_armed=1;
    for(size_t i=0;i<sizeof qec_signals/sizeof qec_signals[0];i++) {
        struct sigaction now,take;
        qec_taken[i]=0;
        if(sigaction(qec_signals[i],NULL,&now) || now.sa_handler!=SIG_DFL)continue;
        memset(&take,0,sizeof take);take.sa_handler=qec_on_signal;sigemptyset(&take.sa_mask);
        qec_taken[i]=!sigaction(qec_signals[i],&take,&qec_previous[i]);
    }
}

static void qec_disarm(void) {
    for(size_t i=0;i<sizeof qec_signals/sizeof qec_signals[0];i++)
        if(qec_taken[i]){sigaction(qec_signals[i],&qec_previous[i],NULL);qec_taken[i]=0;}
    qec_signal_armed=0;
}

/* fsync, and on macOS F_FULLFSYNC after it: there fsync leaves the data in the
 * drive's cache, and the file is renamed into place right after. Best effort
 * where the filesystem has no F_FULLFSYNC. */
static int qec_sync(int fd) {
    if(fsync(fd))return -1;
#ifdef F_FULLFSYNC
    (void)fcntl(fd,F_FULLFSYNC);
#endif
    return 0;
}

/* The rename made durable: the directory synced too. Best effort, the file is
 * in place already. */
static void qec_sync_directory(const char *output) {
    char directory[4096];
    const char *slash=strrchr(output,'/');
    size_t length=slash?(size_t)(slash-output):0;
    if(length>=sizeof directory)return;
    if(slash){memcpy(directory,output,length);directory[length]=0;}
    else strcpy(directory,".");
    if(!directory[0])strcpy(directory,"/");
    int fd=open(directory,O_RDONLY);
    if(fd>=0){(void)fsync(fd);close(fd);}
}

static int qec_write(FILE *out,const void *data,size_t bytes,uint64_t *at,
                     char *error,size_t cap) {
    if(fwrite(data,1,bytes,out)!=bytes)
        return qec_error(error,cap,"cannot write expert cache: %s",strerror(errno));
    *at+=(uint64_t)bytes;
    return 1;
}

static int qec_pad(FILE *out,uint64_t *at,uint64_t end,char *error,size_t cap) {
    static const unsigned char zero[16384]={0};
    if(*at>end)return qec_error(error,cap,"expert cache layout overlaps");
    while(*at<end) {
        size_t n=end-*at>sizeof zero?sizeof zero:(size_t)(end-*at);
        if(!qec_write(out,zero,n,at,error,cap))return 0;
    }
    return 1;
}

/* 1 when every scale of every expert has a layout-2 code: one pass over the
 * scales alone (about 2 GB of the 35B checkpoint) into the caller's buffer,
 * stopping at the first that has none. */
static int qec_codes_fit(shards *S,const QefGeometry *geometry,const QefHeader *h,float *scales,int *fits,
                         char *error,size_t cap) {
    *fits=1;
    for(uint32_t layer=0;*fits && layer<geometry->layers;layer++)
        for(uint32_t expert=0;*fits && expert<geometry->experts;expert++) {
            const st_tensor *weights=NULL,*sc=NULL;
            if(!qes_sources(S,geometry,layer,expert,&weights,&sc,error,cap) ||
               !qes_read_at(sc->fd,scales,(size_t)(h->scale_count*sizeof(float)),(uint64_t)sc->off,error,cap))
                return 0;
            uint16_t code;
            for(uint64_t k=0;*fits && k<h->scale_count;k++)*fits=qef_scale_code(scales[k],&code);
        }
    return 1;
}

static int qec_convert(shards *S,const QefGeometry *geometry,const char *output,
                       char *error,size_t cap) {
    QefHeader header;
    QecSnapshot saved;
    uint8_t *record=NULL;
    char *temporary=NULL;
    FILE *out=NULL;
    int fd=-1,ok=0,temporary_owned=0;
    uint64_t at=0,fingerprint;
    if(error && cap)error[0]=0;
    if(!geometry || !output || !*output)return qec_error(error,cap,"missing geometry or expert cache output");
    if(geometry->hidden>INT_MAX || geometry->inter>INT_MAX)
        return qec_error(error,cap,"expert matrix dimensions exceed the repack interface");
    /* Validate dimensions before traversing the source inventory. */
    if(!qef_layout(geometry,1,&header,error,cap) ||
       !qec_snapshot(S,&saved,error,cap) ||
       !qec_output_safe(output,&saved,error,cap))return 0;
    fingerprint=qes_fingerprint(S,geometry,error,cap);
    if(!fingerprint)return 0;
    /* Weights and F32 scales, the same bytes in both layouts: the scales are
     * read as F32 first, and the pass below uses the same buffer. */
    if(qef_payload_bytes(&header)>SIZE_MAX)return qec_error(error,cap,"expert record exceeds address space");
    record=(uint8_t *)malloc((size_t)qef_payload_bytes(&header));
    if(!record)return qec_error(error,cap,"cannot allocate one expert record");
    int codes=0;
    if(!qec_codes_fit(S,geometry,&header,(float *)(void *)(record+(size_t)header.packed_bytes),&codes,error,cap) ||
       !qef_layout_as(geometry,fingerprint,codes?QEF_LAYOUT_CODES:QEF_LAYOUT,&header,error,cap))goto done;
    size_t output_length=strlen(output);
    static const char suffix[]=".tmp.XXXXXX";
    if(output_length>SIZE_MAX-sizeof suffix) {
        qec_error(error,cap,"expert cache output path is too long");goto done;
    }
    temporary=(char *)malloc(output_length+sizeof suffix);
    if(!temporary){qec_error(error,cap,"cannot allocate output path");goto done;}
    memcpy(temporary,output,output_length);
    memcpy(temporary+output_length,suffix,sizeof suffix);
    /* mkstemp uses O_CREAT|O_EXCL in the destination's own directory. */
    fd=mkstemp(temporary);
    if(fd<0){qec_error(error,cap,"cannot create temporary expert cache: %s",strerror(errno));goto done;}
    temporary_owned=1;
    qec_arm(temporary);
    /* mkstemp creates it 0600: the mode a plain file gets under this umask. */
    mode_t mask=umask(0);umask(mask);
    if(fchmod(fd,0666&~mask)){qec_error(error,cap,"cannot set the expert cache's mode: %s",strerror(errno));goto done;}
    out=fdopen(fd,"wb");
    if(!out){qec_error(error,cap,"cannot open temporary expert cache stream: %s",strerror(errno));goto done;}
    fd=-1; /* stream owns the descriptor from here */
    if(!qec_write(out,&header,sizeof header,&at,error,cap) ||
       !qec_pad(out,&at,header.data_offset,error,cap))goto done;
    size_t matrix_bytes=(size_t)header.packed_bytes/3;
    for(uint32_t layer=0;layer<geometry->layers;layer++) {
        for(uint32_t expert=0;expert<geometry->experts;expert++) {
            const st_tensor *weights=NULL,*scales=NULL;
            if(!qes_sources(S,geometry,layer,expert,&weights,&scales,error,cap) ||
               !qes_read_at(weights->fd,record,(size_t)header.packed_bytes,
                            (uint64_t)weights->off,error,cap) ||
               !qes_read_at(scales->fd,record+(size_t)header.packed_bytes,
                            (size_t)(header.scale_count*sizeof(float)),
                            (uint64_t)scales->off,error,cap))goto done;
            float *scale_values=(float *)(void *)(record+(size_t)header.packed_bytes);
            for(uint64_t k=0;k<header.scale_count;k++) {
                /* Preserve every finite F32 encoding, including signed zero
                 * and negative values, exactly as the source loader does. */
                if(!isfinite(scale_values[k])) {
                    qec_error(error,cap,"invalid scale at layer %u expert %u index %llu",
                              layer,expert,(unsigned long long)k);goto done;
                }
            }
            if(codes) {
                /* Scales to codes in place, from the first: code k is written over
                 * scale k / 2, which is read already. */
                uint16_t *code=(uint16_t *)(void *)scale_values;
                for(uint64_t k=0;k<header.scale_count;k++)
                    if(!qef_scale_code(scale_values[k],&code[k])) {
                        qec_error(error,cap,"scale without a code at layer %u expert %u index %llu",
                                  layer,expert,(unsigned long long)k);goto done;
                    }
            }
            uint8_t *gate=record,*up=record+matrix_bytes,*down=record+2*matrix_bytes;
            xf_repack_pairs_signed(gate,gate,(int)geometry->inter,(int)geometry->hidden);
            xf_repack_pairs_signed(up,up,(int)geometry->inter,(int)geometry->hidden);
            xf_repack_pairs_signed(down,down,(int)geometry->hidden,(int)geometry->inter);
            uint64_t end=at+header.stride;
            if(!qec_write(out,record,(size_t)header.record_bytes,&at,error,cap) ||
               !qec_pad(out,&at,end,error,cap))goto done;
        }
    }
    if(at!=header.bytes){qec_error(error,cap,"expert cache output size does not match layout");goto done;}
    if(fflush(out)){qec_error(error,cap,"cannot flush expert cache: %s",strerror(errno));goto done;}
    if(qec_sync(fileno(out))){qec_error(error,cap,"cannot sync expert cache: %s",strerror(errno));goto done;}
    if(fclose(out)) {
        out=NULL;qec_error(error,cap,"cannot close expert cache: %s",strerror(errno));goto done;
    }
    out=NULL;
    /* Re-evaluate source identity after all reads and the potentially slow
     * disk flush. Keep the prior output intact on any source change. */
    uint64_t final_fingerprint=qes_fingerprint(S,geometry,error,cap);
    if(!final_fingerprint)goto done;
    if(final_fingerprint!=fingerprint) {
        qec_error(error,cap,"source fingerprint changed during conversion");goto done;
    }
    if(!qec_unchanged(S,&saved,error,cap) ||
       !qec_output_safe(output,&saved,error,cap))goto done;
    if(rename(temporary,output)) {
        qec_error(error,cap,"cannot publish expert cache %s: %s",output,strerror(errno));goto done;
    }
    temporary_owned=0;
    qec_sync_directory(output);
    ok=1;
done:
    if(out)fclose(out);
    if(fd>=0)close(fd);
    if(temporary_owned)unlink(temporary);
    qec_disarm();
    free(temporary);
    free(record);
    return ok;
}
#else
static int qec_convert(shards *S,const QefGeometry *geometry,const char *output,
                       char *error,size_t cap) {
    (void)S;(void)geometry;(void)output;
    return qec_error(error,cap,"expert cache conversion requires POSIX");
}
#endif
#endif
