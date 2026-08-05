/* COLI_PIPE_BLOCK: the pipe pool's condvar waiter must be observably
 * equivalent to the sched_yield spin it replaces — same bytes land in the
 * same ws[] slots, and no interleaving loses a wakeup (the worker RELEASE-
 * stores ready[] BEFORE taking mx to broadcast; the waiter re-checks under
 * the lock, so a flag set between its fast-path check and the wait cannot
 * be missed). Both waiters are exercised against the same on-disk fixture,
 * alternating parked waits (wait issued before the load finishes) with
 * fast-path waits (load already done), across enough generations to cycle
 * the pool's gen-tagged cursor.
 *
 * Also pins the PIPE_WORKERS => PIPE implication table: fires ONLY when
 * PIPE is unset in the env AND the platform default left the pipe off AND
 * PIPE_WORKERS parses positive (PIPE_WORKERS=0/empty/negative must NOT
 * silently enable a clamped 1-worker pipe). */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }

enum { NE=8, LAYER=1 };            /* experts 0..NE-1 on one MoE layer */
/* per-expert file image: [gate 12][up 12][down 12][gate.qs 12][up.qs 12][down.qs 16] */
enum { WB=12, QS_G=12, QS_U=12, QS_D=16, ESZ=3*WB+QS_G+QS_U+QS_D };

static unsigned char wbyte(int e,int j){ return (unsigned char)(e*31+j+1); }
static float scale(int e,int i){ return (float)(e*8+i)+0.5f; }

#define TMPF "test_pipe_block.tmp"

static int write_fixture(void){
    FILE *w=fopen(TMPF,"wb"); if(!w) return fail("create temp");
    for(int e=0;e<NE;e++){
        unsigned char img[ESZ];
        for(int j=0;j<3*WB;j++) img[j]=wbyte(e,j);
        float sc[(QS_G+QS_U+QS_D)/4];
        for(int i=0;i<(int)(sizeof(sc)/sizeof(sc[0]));i++) sc[i]=scale(e,i);
        memcpy(img+3*WB,sc,sizeof(sc));
        if(fwrite(img,1,ESZ,w)!=ESZ){ fclose(w); return fail("expert fixture write"); }
    }
    fclose(w);
    return 0;
}

static int build_fixture(Model *m,int fd){
    m->c.hidden=4; m->c.moe_inter=3; m->ebits=8;
    m->S.n=NE*6; m->S.cap=NE*6; m->S.t=calloc(NE*6,sizeof(st_tensor));
    if(!m->S.t) return fail("tensor metadata allocation");
    const char *proj[3]={"gate_proj","up_proj","down_proj"};
    int sbytes[3]={QS_G,QS_U,QS_D};
    for(int e=0;e<NE;e++){
        int64_t wo=(int64_t)e*ESZ, so=wo+3*WB;
        for(int k=0;k<3;k++){
            char name[300];
            snprintf(name,sizeof(name),"model.layers.%d.mlp.experts.%d.%s.weight",LAYER,e,proj[k]);
            m->S.t[e*6+k]=(st_tensor){.name=strdup(name),.fd=fd,.off=wo,
                .nbytes=WB,.dtype=3 /* U8 */,.numel=WB}; wo+=WB;
            size_t n=strlen(name); memcpy(name+n,".qs",4);
            m->S.t[e*6+3+k]=(st_tensor){.name=strdup(name),.fd=fd,.off=so,
                .nbytes=sbytes[k],.dtype=2 /* F32 */,.numel=sbytes[k]/4}; so+=sbytes[k];
        }
    }
    return 0;
}

static int check_slot(ESlot *s,int e){
    if(s->eid!=e || s->g.fmt!=1 || s->u.fmt!=1 || s->d.fmt!=1){
        fprintf(stderr,"  slot: eid=%d (want %d) fmt g/u/d=%d/%d/%d (want 1/1/1)\n",
                s->eid,e,s->g.fmt,s->u.fmt,s->d.fmt);
        return 1;
    }
    const unsigned char *g=(const unsigned char*)s->g.q8,
                        *u=(const unsigned char*)s->u.q8,
                        *d=(const unsigned char*)s->d.q8;   /* q8 is int8_t; compare raw bytes */
    for(int j=0;j<WB;j++)
        if(g[j]!=wbyte(e,j) || u[j]!=wbyte(e,WB+j) || d[j]!=wbyte(e,2*WB+j)){
            fprintf(stderr,"  slot e=%d weight byte %d: g=%d/%d u=%d/%d d=%d/%d (got/want)\n",e,j,
                    g[j],wbyte(e,j),u[j],wbyte(e,WB+j),d[j],wbyte(e,2*WB+j));
            return 1;
        }
    for(int i=0;i<3;i++)
        if(s->g.s[i]!=scale(e,i) || s->u.s[i]!=scale(e,3+i)){
            fprintf(stderr,"  slot e=%d scale %d: g=%g/%g u=%g/%g (got/want)\n",e,i,
                    (double)s->g.s[i],(double)scale(e,i),(double)s->u.s[i],(double)scale(e,3+i));
            return 1;
        }
    for(int i=0;i<4;i++)
        if(s->d.s[i]!=scale(e,6+i)){
            fprintf(stderr,"  slot e=%d scale %d: d=%g/%g (got/want)\n",e,i,
                    (double)s->d.s[i],(double)scale(e,6+i));
            return 1;
        }
    return 0;
}

static int run_generations(Model *m,int block,int gens){
    g_pipe_block=block;
    for(int gen=0;gen<gens;gen++){
        int eids[NE];
        for(int q=0;q<NE;q++) eids[q]=(gen*3+q)%NE;   /* deterministic shuffle across gens */
        pipe_dispatch(m,LAYER,eids,NE);
        if(gen%4==0) usleep(300);                     /* let loads finish → fast-path wait */
        for(int i=0;i<NE;i++){
            /* odd gens wait on the LAST-dispatched slot first: with jobs this
             * small, in-order waits mostly find ready already set — reverse
             * order is what actually parks the waiter on the condvar. */
            int q=(gen&1)?NE-1-i:i;
            pipe_wait(q);
            if(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire))
                return fail(block?"blocking wait returned before ready":"spin wait returned before ready");
            if(check_slot(&m->ws[q],eids[q])) return fail(block?"slot contents (block)":"slot contents (spin)");
        }
    }
    return 0;
}

static int test_implication_table(void){
    struct { const char *pipe_env,*pw_env; int pipe_now,want; } T[]={
        {NULL,"4",0,1},   /* pool sized, pipe off, PIPE unset → imply */
        {NULL,"16",0,1},
        {NULL,"0",0,0},   /* PIPE_WORKERS=0 must NOT enable a clamped pipe */
        {NULL,"",0,0},
        {NULL,"-3",0,0},
        {"0","4",0,0},    /* explicit PIPE=0 always wins */
        {"1","4",1,0},    /* explicit PIPE=1: nothing to imply */
        {NULL,"4",1,0},   /* platform default already ON (win32) */
        {NULL,NULL,0,0},
    };
    for(size_t i=0;i<sizeof(T)/sizeof(T[0]);i++)
        if(pipe_workers_imply_pipe(T[i].pipe_env,T[i].pw_env,T[i].pipe_now)!=T[i].want){
            fprintf(stderr,"FAIL: implication row %zu (PIPE=%s PIPE_WORKERS=%s pipe_now=%d)\n",
                    i,T[i].pipe_env?T[i].pipe_env:"<unset>",T[i].pw_env?T[i].pw_env:"<unset>",T[i].pipe_now);
            return 1;
        }
    return 0;
}

int main(void){
    if(test_implication_table()) return 1;

    /* Relative to the CWD, like test_compat_direct's TMPF — NOT "/tmp/...":
     * the windows job builds native .exe files and "/tmp" is not a Windows
     * path. fwrite then reopen read-only: Windows compat has pread, not pwrite. */
    if(write_fixture()) return 1;
    int fd=open(TMPF,COMPAT_O_RDONLY);
    if(fd<0) return fail("open temp");

    static Model m;                                   /* zeroed: buffered pread path, no mmap/cuda */
    if(build_fixture(&m,fd)){ close(fd); remove(TMPF); return 1; }

    g_pipe=1; g_pipe_nw=4;
    pipe_init(&m);

    /* spin waiter first (control), then the condvar waiter under the same
     * dispatch pattern; 200 generations each cycles the gen-tagged cursor
     * and alternates parked/fast-path waits. */
    if(run_generations(&m,0,200) || run_generations(&m,1,200)){ close(fd); remove(TMPF); return 1; }

    for(int q=0;q<NE;q++){ compat_aligned_free(m.ws[q].slab); free(m.ws[q].fslab); }
    for(int i=0;i<m.S.n;i++) free(m.S.t[i].name);
    free(m.S.t);
    close(fd);
    remove(TMPF);
    puts("test_pipe_block: ok");
    return 0;
}
