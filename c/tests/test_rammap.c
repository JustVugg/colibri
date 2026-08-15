#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <sys/vfs.h>
#endif
#include <unistd.h>

#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fail(const char *what){ fprintf(stderr,"FAIL: %s\n",what); return 1; }

static int test_mixed_tier_accounting(void){
    ColiRamTier ram=tiers_ram_account(300.0,2,1,1,50);
    if(ram.experts!=3 || ram.bytes!=250.0)
        return fail("mixed anonymous/VRAM/RAMMAP tier accounting");
    return 0;
}

#ifdef __linux__
static int expert_greedy_token(ESlot *slot,const float x[4],float out[4]){
    float gate[3],up[3],hidden[3];
    matmul_qt(gate,x,&slot->g,1); matmul_qt(up,x,&slot->u,1);
    for(int i=0;i<3;i++) hidden[i]=siluf(gate[i])*up[i];
    matmul_qt(out,hidden,&slot->d,1);
    int token=0; for(int i=1;i<4;i++) if(out[i]>out[token]) token=i;
    return token;
}

static int make_regular_temp(char *path,size_t cap){
    const char *roots[]={"/var/tmp","/tmp","."};
    for(size_t i=0;i<sizeof(roots)/sizeof(roots[0]);i++){
        snprintf(path,cap,"%s/coli-rammap-XXXXXX",roots[i]);
        int fd=mkstemp(path); if(fd<0) continue;
        struct statfs fs;
        if(!fstatfs(fd,&fs) && (unsigned long)fs.f_type!=(unsigned long)TMPFS_MAGIC) return fd;
        close(fd); unlink(path);
    }
    path[0]=0; return -1;
}

static void add_expert(Model *m,int expert,int tfd,int mixed_fd){
    static const char *proj[3]={"gate_proj","up_proj","down_proj"};
    static const int wb[3]={12,12,12}, sb[3]={12,12,16};
    int64_t base=(int64_t)expert*76,wo=base,so=base+36;
    for(int k=0;k<3;k++){
        char name[300];
        snprintf(name,sizeof(name),"model.layers.0.mlp.experts.%d.%s.weight",expert,proj[k]);
        st_tensor *w=&m->S.t[m->S.n++];
        *w=(st_tensor){strdup(name),tfd,wo,wb[k],3,wb[k]}; wo+=wb[k];
        size_t n=strlen(name); memcpy(name+n,".qs",4);
        st_tensor *q=&m->S.t[m->S.n++];
        int fd=(expert==1 && k==2)?mixed_fd:tfd;
        *q=(st_tensor){strdup(name),fd,so,sb[k],2,sb[k]/4}; so+=sb[k];
    }
}

static int64_t add_quant_expert(Model *m,int fd,long fs_magic,int fmt,
                                int hidden,int moe_inter,int64_t base){
    static const char *proj[3]={"gate_proj","up_proj","down_proj"};
    int OO[3]={moe_inter,moe_inter,hidden},II[3]={hidden,hidden,moe_inter};
    memset(m,0,sizeof(*m)); m->c.hidden=hidden; m->c.moe_inter=moe_inter;
    m->S.cap=6; m->S.t=calloc(6,sizeof(st_tensor));
    m->S.fds[0]=fd; m->S.fs_magic[0]=fs_magic; m->S.is_tmpfs[0]=1; m->S.nfd=1;
    int64_t off=base,total=0;
    for(int k=0;k<3;k++){
        int O=OO[k],I=II[k];
        int64_t wb=fmt==6 ? (int64_t)O*e8_rowbytes(I) : (int64_t)O*I;
        int64_t sb=fmt==6 ? 4 : fp8_nblk(O)*fp8_nblk(I)*4;
        char name[300];
        snprintf(name,sizeof(name),"model.layers.0.mlp.experts.0.%s.weight",proj[k]);
        m->S.t[m->S.n++]=(st_tensor){strdup(name),fd,off,wb,3,wb}; off+=wb;
        size_t n=strlen(name); memcpy(name+n,".qs",4);
        m->S.t[m->S.n++]=(st_tensor){strdup(name),fd,off,sb,2,sb/4}; off+=sb;
        total+=wb+sb;
    }
    return total;
}

static void free_quant_expert(Model *m){
    for(int i=0;i<m->S.n;i++) free(m->S.t[i].name);
    free(m->S.t);
}

static int check_direct_quant_format(int fd,long fs_magic,int fmt,int hidden,int moe_inter,
                                     int64_t base,int64_t expected_total){
    Model m; int64_t fixture_total=add_quant_expert(&m,fd,fs_magic,fmt,hidden,moe_inter,base);
    ESlot slot={0}; int64_t got=rammap_bind_one(&m,0,0,&slot);
    QT *q[3]={&slot.g,&slot.u,&slot.d};
    int OO[3]={moe_inter,moe_inter,hidden},II[3]={hidden,hidden,moe_inter};
    int bad=fixture_total!=expected_total || got!=expected_total ||
            slot.eid!=0 || slot.backing!=ESLOT_BACKING_RAMMAP;
    for(int k=0;k<3;k++){
        int64_t want_weight=fmt==6 ? (int64_t)OO[k]*e8_rowbytes(II[k])
                                   : (int64_t)OO[k]*II[k];
        int64_t want_scale=fmt==6 ? 4 : fp8_nblk(OO[k])*fp8_nblk(II[k])*4;
        if(q[k]->fmt!=fmt || q[k]->O!=OO[k] || q[k]->I!=II[k] || q[k]->gs!=0 ||
           qt_scale_bytes(q[k])!=want_scale || qt_bytes(q[k])-want_scale!=want_weight ||
           (fmt==6 ? q[k]->q4==NULL : q[k]->q8==NULL)) bad=1;
    }
    free_quant_expert(&m);
    return bad ? fail(fmt==6 ? "direct E8 binding / geometry" :
                               "direct FP8 binding / geometry") : 0;
}
#endif

int main(void){
    if(test_mixed_tier_accounting()) return 1;
#ifndef __linux__
    puts("test_rammap: skipped (Linux only)"); return 0;
#else
    uint8_t owned_byte=0;
    ESlot no_host={0}, released_owned={.backing=ESLOT_BACKING_OWNED};
    ESlot owned={.slab=&owned_byte,.backing=ESLOT_BACKING_OWNED};
    ESlot rammap={.backing=ESLOT_BACKING_RAMMAP};
    ESlot mmap_slot={.backing=ESLOT_BACKING_MMAP};
    if(expert_host_ready(&no_host) || expert_host_ready(&released_owned) ||
       !expert_host_ready(&owned) || !expert_host_ready(&rammap) ||
       !expert_host_ready(&mmap_slot))
        return fail("expert host-ready backing classification");
    QT grouped={.fmt=4,.O=2,.I=129,.gs=64};
    QT int3={.fmt=5,.O=2,.I=65};
    QT e8={.fmt=6,.O=2,.I=257};
    if(qt_scale_bytes(&grouped)!=24 || qt_scale_bytes(&int3)!=16 ||
       qt_bytes(&int3)-qt_scale_bytes(&int3)!=96 ||
       qt_scale_bytes(&e8)!=4 || qt_bytes(&e8)-qt_scale_bytes(&e8)!=392)
        return fail("grouped/int3/E8 scale and weight byte geometry");
    ProfPhysicalWire unavailable=prof_physical_wire(-1);
    ProfPhysicalWire measured_zero=prof_physical_wire(0);
    ProfPhysicalWire measured_bytes=prof_physical_wire(4096);
    if(unavailable.valid || unavailable.bytes!=0 ||
       !measured_zero.valid || measured_zero.bytes!=0 ||
       !measured_bytes.valid || measured_bytes.bytes!=4096)
        return fail("physical-read wire validity distinguishes unavailable from zero");
    uint64_t read_bytes;
    if(prof_physical_read_bytes(&read_bytes)){
        ProfBase sample={.physical_read_bytes=read_bytes,.physical_read_valid=1};
        if(prof_physical_read_delta(&sample)<0)
            return fail("monotonic /proc/self/io read_bytes sampling");
    }
    char tpath[]="/dev/shm/coli-rammap-XXXXXX";
    int tfd=mkstemp(tpath);
    if(tfd<0){ printf("test_rammap: skipped (/dev/shm: %s)\n",strerror(errno)); return 0; }
    struct statfs tfs;
    if(fstatfs(tfd,&tfs) || (unsigned long)tfs.f_type!=(unsigned long)TMPFS_MAGIC){
        close(tfd); unlink(tpath); puts("test_rammap: skipped (/dev/shm is not tmpfs)"); return 0;
    }
    shards probe={0}; int probe_fd=st_open_fd(&probe,tpath);
    if(!st_fd_is_tmpfs(&probe,probe_fd) || probe.dfds[0]!=-1)
        return fail("st_open_fd tmpfs detection / O_DIRECT suppression");
    close(probe_fd); free(probe.paths[0]);
    char rpath[256]; int rfd=make_regular_temp(rpath,sizeof(rpath));
    int untracked_mixed=0;
    if(rfd<0){
        char fallback[]="/dev/shm/coli-rammap-mixed-XXXXXX";
        rfd=mkstemp(fallback); if(rfd<0){ close(tfd); unlink(tpath); return fail("mixed fixture"); }
        snprintf(rpath,sizeof(rpath),"%s",fallback);
        untracked_mixed=1; /* still proves all six descriptors must be verified */
    }
    unsigned char data[152]={0};
    for(int expert=0;expert<2;expert++){
        int base=expert*76;
        for(int i=0;i<36;i++) data[base+i]=(unsigned char)(1+(expert*17+i)%31);
        float scales[10]; for(int i=0;i<10;i++) scales[i]=0.01f*(float)(i+1);
        memcpy(data+base+36,scales,sizeof(scales));
    }
    if(pwrite(tfd,data,sizeof(data),0)!=(ssize_t)sizeof(data) ||
       pwrite(rfd,data,sizeof(data),0)!=(ssize_t)sizeof(data)){
        close(tfd); close(rfd); unlink(tpath); unlink(rpath); return fail("fixture write");
    }
    if(ftruncate(tfd,1<<20)) return fail("extended quantized fixture");
    if(check_direct_quant_format(tfd,(long)tfs.f_type,6,384,256,4096,137996)) return 1;
    if(check_direct_quant_format(tfd,(long)tfs.f_type,8,384,256,262144,294984)) return 1;

    /* At I=98 the E8 weight/tag geometry collides with FP8 (and, for O=1,
     * int8). Direct mapping must decline an unstamped ambiguous expert and
     * leave the ordinary slab/SSD loader in charge, never guess a decoder. */
    Model collision; add_quant_expert(&collision,tfd,(long)tfs.f_type,6,98,64,600000);
    ESlot collision_slot={0};
    if(rammap_bind_one(&collision,0,0,&collision_slot)!=0 ||
       collision_slot.backing==ESLOT_BACKING_RAMMAP)
        return fail("ambiguous E8/FP8 direct binding fails closed");
    free_quant_expert(&collision);

    Model m={0}; m.c.n_layers=1; m.c.n_experts=2; m.c.hidden=4; m.c.moe_inter=3;
    m.c.first_dense=0; m.ebits=8; m.L=calloc(1,sizeof(Layer)); m.L[0].sparse=1;
    m.S.cap=12; m.S.t=calloc(12,sizeof(st_tensor));
    m.S.fds[0]=tfd; m.S.fs_magic[0]=(long)tfs.f_type; m.S.is_tmpfs[0]=1; m.S.nfd=1;
    if(!untracked_mixed){
        struct statfs rfs; if(fstatfs(rfd,&rfs)){ return fail("regular fstatfs"); }
        m.S.fds[1]=rfd; m.S.fs_magic[1]=(long)rfs.f_type;
        m.S.is_tmpfs[1]=((unsigned long)rfs.f_type==(unsigned long)TMPFS_MAGIC); m.S.nfd=2;
    }
    add_expert(&m,0,tfd,tfd); add_expert(&m,1,tfd,rfd);
    m.pin=calloc(2,sizeof(ESlot*)); m.npin=calloc(2,sizeof(int));
    m.ecache=calloc(2,sizeof(ESlot*)); m.ecn=calloc(2,sizeof(int));

    g_mmap=0; g_rammap=1; g_ram_prefault=0;
    rammap_build(&m);
    if(m.rammap_experts!=1 || m.rammap_bytes!=76 || !rammap_slot(&m,0,0) || rammap_slot(&m,0,1))
        return fail("full-six-tensor tmpfs eligibility / mixed fallback");
    if(!rammap_modes_conflict(1,1) || rammap_modes_conflict(1,0) || rammap_modes_conflict(0,1))
        return fail("COLI_MMAP/COLI_RAMMAP conflict");

    ESlot staged={0}; atomic_store_explicit(&g_prof_io,0,memory_order_relaxed);
    if(expert_load_impl(&m,0,0,&staged,1,0) ||
       atomic_load_explicit(&g_prof_io,memory_order_relaxed)!=0)
        return fail("tmpfs slab path has zero SSD-backed expert requests");
    ESlot *mapped=rammap_slot(&m,0,0); float x[4]={.25f,-.5f,.75f,1.f};
    QT *direct_qt[3]={&mapped->g,&mapped->u,&mapped->d};
    QT *slab_qt[3]={&staged.g,&staged.u,&staged.d};
    float in3[3]={.2f,-.4f,.6f};
    for(int k=0;k<3;k++){
        float direct_y[4]={0},slab_y[4]={0};
        const float *input=k<2?x:in3; int outputs=k<2?3:4;
        matmul_qt(direct_y,input,direct_qt[k],1); matmul_qt(slab_y,input,slab_qt[k],1);
        if(direct_qt[k]->fmt!=slab_qt[k]->fmt || direct_qt[k]->gs!=slab_qt[k]->gs ||
           memcmp(direct_y,slab_y,(size_t)outputs*sizeof(float)))
            return fail("all direct and tmpfs-slab projections are equivalent");
    }
    compat_aligned_free(staged.slab); free(staged.fslab);

    char state_root[]="/tmp/coli-rammap-state-XXXXXX";
    if(!mkdtemp(state_root)) return fail("state fixture");
    char state_leaf[320]; snprintf(state_leaf,sizeof(state_leaf),"%s/node/engine",state_root);
    struct stat state_st;
    if(state_dir_prepare(state_leaf) || stat(state_leaf,&state_st) || !S_ISDIR(state_st.st_mode))
        return fail("explicit state directory creation");
    char state_too_long[2048]; memset(state_too_long,'x',sizeof(state_too_long)-1); state_too_long[0]='/';
    state_too_long[sizeof(state_too_long)-1]=0;
    if(state_dir_prepare(state_too_long)==0 || errno!=ENAMETOOLONG)
        return fail("state directory suffix length guard");

    char profile[]="/tmp/coli-rammap-profile-XXXXXX"; int pfd=mkstemp(profile);
    if(pfd<0 || dprintf(pfd,"0 0 100\n0 1 50\n")<0){ return fail("pin profile"); }
    close(pfd); pin_load(&m,profile,0.001,1); unlink(profile);
    if(m.npin[0]!=1 || m.pin[0][0].eid!=1) return fail("PIN excludes direct experts");
    if(expert_resident_slot(&m,0,1,0)!=&m.pin[0][0]) return fail("hybrid SSD fallback residency");

    /* The same expert bytes are available on both fixtures. Exercise three
     * complete paths and compare a synthetic greedy token: all-SSD slabs,
     * full tmpfs RAM-map, and five tmpfs tensors + one SSD fallback tensor. */
    st_tensor *expert1[6]; int expert1_fd[6],nexpert1=0;
    for(int i=0;i<m.S.n;i++) if(strstr(m.S.t[i].name,"experts.1.")){
        expert1[nexpert1]=&m.S.t[i]; expert1_fd[nexpert1]=m.S.t[i].fd; nexpert1++;
    }
    if(nexpert1!=6) return fail("expert1 tensor fixture");
    ESlot ssd={0},full={0},mixed={0};
    for(int i=0;i<6;i++) expert1[i]->fd=rfd;
    atomic_store_explicit(&g_prof_io,0,memory_order_relaxed);
    if(expert_load_impl(&m,0,1,&ssd,1,0)) return fail("SSD expert load");
    if(!untracked_mixed && atomic_load_explicit(&g_prof_io,memory_order_relaxed)!=76)
        return fail("all-SSD requested expert byte telemetry");
    for(int i=0;i<6;i++) expert1[i]->fd=tfd;
    if(rammap_bind_one(&m,0,1,&full)!=76) return fail("full direct expert binding");
    for(int i=0;i<6;i++) expert1[i]->fd=expert1_fd[i];
    atomic_store_explicit(&g_prof_io,0,memory_order_relaxed);
    if(expert_load_impl(&m,0,1,&mixed,1,0)) return fail("hybrid expert load");
    if(!untracked_mixed && atomic_load_explicit(&g_prof_io,memory_order_relaxed)!=16)
        return fail("descriptor-classified requested expert byte telemetry");
    float y_ssd[4],y_full[4],y_mixed[4];
    int tok_ssd=expert_greedy_token(&ssd,x,y_ssd);
    int tok_full=expert_greedy_token(&full,x,y_full);
    int tok_mixed=expert_greedy_token(&mixed,x,y_mixed);
    if(tok_ssd!=tok_full || tok_ssd!=tok_mixed ||
       memcmp(y_ssd,y_full,sizeof(y_ssd)) || memcmp(y_ssd,y_mixed,sizeof(y_ssd)))
        return fail("SSD/full-RAM/hybrid greedy token identity");
    compat_aligned_free(ssd.slab); free(ssd.fslab);
    compat_aligned_free(mixed.slab); free(mixed.fslab);

    m.pin[0]=realloc(m.pin[0],2*sizeof(ESlot)); memset(&m.pin[0][1],0,sizeof(ESlot));
    m.pin[0][1].eid=0; m.pin[0][1].backing=ESLOT_BACKING_OWNED; m.npin[0]=2;
    m.ecache[0]=calloc(1,sizeof(ESlot)); m.ecache[0][0].eid=0; m.ecache[0][0].used=123; m.ecn[0]=1;
    atomic_store_explicit(&g_prof_io,777,memory_order_relaxed); m.rammap_calls=0;
    ESlot *direct=rammap_slot(&m,0,0);
    if(!expert_slot_is_pinned(&m,0,&m.pin[0][0]) ||
       expert_slot_is_pinned(&m,0,direct) ||
       expert_slot_is_pinned(&m,0,&m.ecache[0][0]))
        return fail("pin tier classification uses slot identity");
    if(expert_resident_slot(&m,0,0,1)!=direct || m.rammap_calls!=1 ||
       m.ecache[0][0].used!=123 || atomic_load_explicit(&g_prof_io,memory_order_relaxed)!=777)
        return fail("direct-map precedence / LRU and I/O exclusion / telemetry");
    if(!expert_is_resident(&m,0,0) || m.rammap_calls!=1) return fail("residency probe telemetry isolation");

    compat_aligned_free(m.pin[0][0].slab); free(m.pin[0][0].fslab); free(m.pin[0]);
    free(m.ecache[0]); free(m.pin); free(m.npin); free(m.ecache); free(m.ecn);
    free(m.rammap); free(m.L);
    for(int i=0;i<m.S.n;i++) free(m.S.t[i].name); free(m.S.t);
    for(int i=0;i<g_nmaps;i++) munmap(g_maps[i].base,g_maps[i].len);
    g_nmaps=0; close(tfd); close(rfd); unlink(tpath); unlink(rpath);
    char state_node[320]; snprintf(state_node,sizeof(state_node),"%s/node",state_root);
    rmdir(state_leaf); rmdir(state_node); rmdir(state_root);
    puts("test_rammap: ok"); return 0;
#endif
}
