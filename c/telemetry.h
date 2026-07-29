/* telemetry.h — dashboard protocol lines, stats/usage persistence, hardware probe.
 * Include after Model/Cfg/QT/ESlot/shards and st.h are defined; requires
 * qt_bytes(), now_s(), rss_gb(), edisk_s(), and the g_cuda_* globals (ifdef). */
#ifndef TELEMETRY_H
#define TELEMETRY_H

/* PR #377: forward decl — rammap_slot() is defined in colibri.c below the point
 * this header is #included, but emap_emit/tiers_emit consult it for the tmpfs
 * direct tier. telemetry.h is included after Model/ESlot are defined. */
static ESlot *rammap_slot(Model *m, int layer, int eid);

static int64_t tbytes(int O,int I,int bits){
    if(bits>=16) return (int64_t)O*I*4;
    if(bits>=5)  return (int64_t)O*I + (int64_t)O*4;
    return (int64_t)O*((I+1)/2) + (int64_t)O*4;
}

static int64_t expert_bytes_probe(Model *m, int ebits){
    Cfg *c=&m->c; int64_t eb=0; char nm[256];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.gate_proj.weight",c->first_dense);
    if(st_nbytes(&m->S,nm)>0){
        const char *suf[3]={"gate_proj","up_proj","down_proj"};
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight",c->first_dense,suf[k]);
            eb+=st_nbytes(&m->S,nm);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight.qs",c->first_dense,suf[k]);
            int64_t q=st_nbytes(&m->S,nm); if(q>0) eb+=q;
        }
    }
    if(eb<=0) eb = tbytes(c->moe_inter,c->hidden,ebits)*2 + tbytes(c->hidden,c->moe_inter,ebits);
    return eb;
}

/* BRAIN MAP: per-turn expert hit bitmap for the dashboard. */
static uint8_t **g_ehit;
static void ehit_mark(Model *m, int layer, int eid){
    if(!g_ehit){ Cfg *c=&m->c;
        g_ehit=calloc(c->n_layers+1,sizeof(uint8_t*));
        for(int i=0;i<=c->n_layers;i++) g_ehit[i]=calloc(c->n_experts,1);
    }
    g_ehit[layer][eid]=1;
}

/* CPU model + cores + RAM (GB); empty/zero where unavailable. */
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif
static void hw_probe(char *cpu, size_t cn, int *cores, double *ram_total, double *ram_avail){
    cpu[0]=0;
#ifdef _WIN32
#if defined(__x86_64__) || defined(__i386__)
    { unsigned int r[12]={0}; unsigned int *w=r;
      for(unsigned int f=0x80000002u; f<=0x80000004u; f++,w+=4)
          __get_cpuid(f,&w[0],&w[1],&w[2],&w[3]);
      char *b=(char*)r; b[47]=0; while(*b==' ')b++;
      snprintf(cpu,cn,"%s",b); }
#endif
#elif defined(__APPLE__)
    { size_t sl=cn; if(sysctlbyname("machdep.cpu.brand_string",cpu,&sl,NULL,0)) cpu[0]=0; }
#else
    FILE *ci=fopen("/proc/cpuinfo","r");
    if(ci){ char ln[256];
        while(fgets(ln,sizeof(ln),ci)) if(!strncmp(ln,"model name",10)){
            char *p=strchr(ln,':'); if(p){ p++; while(*p==' ')p++;
            int n=(int)strlen(p); if(n>0&&p[n-1]=='\n')p[--n]=0;
            snprintf(cpu,cn,"%s",p); } break; }
        fclose(ci); }
#endif
    *cores=0;
#ifdef _WIN32
    { SYSTEM_INFO si; GetSystemInfo(&si); *cores=(int)si.dwNumberOfProcessors; }
#elif defined(_SC_NPROCESSORS_ONLN)
    *cores=(int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    *ram_total=*ram_avail=0;
#ifdef _WIN32
    compat_meminfo(ram_total,ram_avail);
#elif defined(__APPLE__)
    { uint64_t ms=0; size_t sl=sizeof(ms);
      if(!sysctlbyname("hw.memsize",&ms,&sl,NULL,0)) *ram_total=(double)ms/1e9;
      int64_t pgsz=0; sl=sizeof(pgsz);
      if(sysctlbyname("hw.pagesize",&pgsz,&sl,NULL,0)||pgsz<=0) pgsz=16384;
      vm_statistics64_data_t vs; mach_msg_type_number_t nc=HOST_VM_INFO64_COUNT;
      if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vs,&nc)==KERN_SUCCESS)
          /* macOS analogue of Linux MemAvailable: free + inactive + purgeable */
          *ram_avail=(double)(vs.free_count+vs.inactive_count+vs.purgeable_count)*(double)pgsz/1e9; }
#else
    FILE *mi=fopen("/proc/meminfo","r");
    if(mi){ char ln[256]; double mt=0,ma=0;
        while(fgets(ln,sizeof(ln),mi)){
            if(sscanf(ln,"MemTotal: %lf",&mt)==1) *ram_total=mt/1e6;
            if(sscanf(ln,"MemAvailable: %lf",&ma)==1) *ram_avail=ma/1e6;
        } fclose(mi); }
#endif
}

/* Per-device CUDA placement.  Keep the documented GPUS line for older
 * dashboards, then publish an integer-byte, versioned record for control-plane
 * consumers that need to distinguish model tensors from card-wide VRAM use.
 *
 * GPUDETAIL v1 record (eight fields per device):
 *   <cuda_ordinal> <identity-or-dash> <total_bytes> <free_bytes>
 *   <model_bytes> <expert_bytes> <nonexpert_bytes> <expert_count>
 *
 * The CUDA backend currently exposes ordinals but not PCI/UUID identity, so
 * identity is "-".  The reserved token makes adding a backend identity query
 * wire-compatible later. */
static void gpus_emit(Model *m){
    int ndev=0,valid_count=0;
#ifdef COLI_CUDA
    if(g_cuda_enabled) ndev=g_cuda_ndev;
    uint64_t expert_bytes[COLI_CUDA_MAX_DEVICES]={0};
    int expert_count[COLI_CUDA_MAX_DEVICES]={0};
    size_t free_bytes[COLI_CUDA_MAX_DEVICES]={0};
    size_t total_bytes[COLI_CUDA_MAX_DEVICES]={0};
    size_t model_bytes[COLI_CUDA_MAX_DEVICES]={0};
    unsigned char detail_valid[COLI_CUDA_MAX_DEVICES]={0};
    if(ndev){
        Cfg *c=&m->c;
        for(int li=0;li<=c->n_layers;li++) for(int z=0;z<m->npin[li];z++){
            ESlot *s=&m->pin[li][z];
            if(!s->g.cuda && !s->u.cuda && !s->d.cuda) continue;
            int device=s->g.cuda?s->g.cuda_device:
                       s->u.cuda?s->u.cuda_device:s->d.cuda_device;
            int di=-1;
            for(int i=0;i<ndev;i++) if(g_cuda_devices[i]==device){ di=i; break; }
            if(di<0) continue;
            expert_count[di]++;
            expert_bytes[di]+=(uint64_t)coli_cuda_tensor_bytes(s->g.cuda)
                             +(uint64_t)coli_cuda_tensor_bytes(s->u.cuda)
                             +(uint64_t)coli_cuda_tensor_bytes(s->d.cuda);
        }
        for(int i=0;i<ndev;i++){
            size_t tensor_count=0;
            int memory_ok=coli_cuda_mem_info(
                g_cuda_devices[i],&free_bytes[i],&total_bytes[i]);
            coli_cuda_stats(g_cuda_devices[i],&tensor_count,&model_bytes[i]);
            (void)tensor_count;
            if(memory_ok
                    && total_bytes[i]>0
                    && free_bytes[i]<=total_bytes[i]
                    && model_bytes[i]<=total_bytes[i]
                    && expert_bytes[i]<=(uint64_t)model_bytes[i]){
                detail_valid[i]=1;
                valid_count++;
            }
        }
    }
#else
    (void)m;
#endif
    /* GPUS has implicit ordinal positions, so a partial sample cannot be
     * represented safely. Fail that legacy advisory closed as an empty set. */
    int legacy_count=valid_count==ndev?ndev:0;
    printf("GPUS %d",legacy_count);
#ifdef COLI_CUDA
    if(legacy_count) for(int i=0;i<ndev;i++){
        double used_gb=(double)(total_bytes[i]-free_bytes[i])/1e9;
        printf(" %.3f %.3f %d",used_gb,
            (double)total_bytes[i]/1e9,expert_count[i]);
    }
#endif
    putchar('\n');
    /* GPUDETAIL carries explicit ordinals, so valid devices remain useful
     * while an omitted device makes control-plane completeness checks stale. */
    printf("GPUDETAIL 1 %d",valid_count);
#ifdef COLI_CUDA
    for(int i=0;i<ndev;i++) if(detail_valid[i]){
        uint64_t nonexpert_bytes=(uint64_t)model_bytes[i]-expert_bytes[i];
        printf(" %d - %llu %llu %llu %llu %llu %d",
            g_cuda_devices[i],
            (unsigned long long)total_bytes[i],
            (unsigned long long)free_bytes[i],
            (unsigned long long)model_bytes[i],
            (unsigned long long)expert_bytes[i],
            (unsigned long long)nonexpert_bytes,
            expert_count[i]);
    }
#endif
    putchar('\n');
    fflush(stdout);
}

static void hwinfo_emit(Model *m){
    Cfg *c=&m->c; (void)c;
    char cpu[256]; int cores; double ram_total,ram_avail;
    hw_probe(cpu,sizeof(cpu),&cores,&ram_total,&ram_avail);
    int ngpu=0; double vram_total=0;
    char gpu_name[128]="";
#ifdef COLI_CUDA
    ngpu=g_cuda_ndev; vram_total=m->gpu_expert_bytes/1e9;
    for(int i=0;i<g_cuda_ndev;i++){
        size_t fr=0,to=0; coli_cuda_mem_info(g_cuda_devices[i],&fr,&to);
        if(!i) vram_total=(double)to*g_cuda_ndev/1e9;
    }
    if(g_cuda_ndev>0)
        snprintf(gpu_name,sizeof(gpu_name),"CUDA device x%d",g_cuda_ndev);
#endif
    printf("HWINFO %d %.1f %.1f %d %.1f %s|%s\n",
        cores,ram_total,ram_avail,ngpu,vram_total,cpu,gpu_name);
    fflush(stdout);
    gpus_emit(m);
}

static void tiers_emit(Model *m){
    Cfg *c=&m->c; int nsp=0;
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    int total=(nsp+(m->has_mtp?1:0))*c->n_experts;
    int pinned=0,lru=0;
    for(int i=0;i<=c->n_layers;i++){ pinned+=m->npin?m->npin[i]:0; lru+=m->ecn?m->ecn[i]:0; }
    int vram=0; double vram_gb=0;
#ifdef COLI_CUDA
    vram=m->gpu_expert_count; vram_gb=m->gpu_expert_bytes/1e9;
#endif
    int anon_ram=pinned-vram+lru; if(anon_ram<0) anon_ram=0;
    int ram=anon_ram+m->rammap_experts;
    int disk=total-vram-ram; if(disk<0) disk=0;
    double eb=(double)expert_bytes_probe(m,m->ebits);
    double ram_gb=anon_ram*eb/1e9+m->rammap_bytes/1e9;
    printf("TIERS %d %d %d %.2f %.2f\n",vram,ram,disk,vram_gb,ram_gb);
    fflush(stdout);
}

static void emap_emit(Model *m){
    Cfg *c=&m->c;
    int rows=0;
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) rows++;
    int has_mtp = m->has_mtp && m->eusage[c->n_layers];
    if(has_mtp) rows++;
    int cols=c->n_experts;
    char *hex=malloc((size_t)rows*cols*2+1); int w=0;
    for(int i=0;i<=c->n_layers;i++){
        int is_row = (i<c->n_layers && m->L[i].sparse) || (i==c->n_layers && has_mtp);
        if(!is_row) continue;
        for(int e=0;e<cols;e++){
            int tier=rammap_slot(m,i,e)?1:0;
            ESlot *P=m->pin[i];
            for(int z=0;!tier && z<m->npin[i];z++) if(P[z].eid==e){
#ifdef COLI_CUDA
                tier = P[z].g.cuda?2:1;
#else
                tier = 1;
#endif
                break; }
            if(!tier && m->ecache && m->ecache[i])
                for(int z=0;z<m->ecn[i];z++) if(m->ecache[i][z].eid==e){ tier=1; break; }
            uint32_t u = m->eusage[i]?m->eusage[i][e]:0;
            int heat=0; while(u){ heat++; u>>=1; } if(heat>63) heat=63;
            int b=(tier<<6)|heat;
            hex[w++]="0123456789abcdef"[b>>4]; hex[w++]="0123456789abcdef"[b&15];
        }
    }
    hex[w]=0;
    printf("EMAP %d %d %s\n",rows,cols,hex); fflush(stdout); free(hex);
}

static void hits_emit(Model *m){
    Cfg *c=&m->c; if(!g_ehit) return;
    int rows=0;
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) rows++;
    int has_mtp = m->has_mtp && m->eusage[c->n_layers];
    if(has_mtp) rows++;
    int cols=c->n_experts, nb=(rows*cols+7)/8;
    uint8_t *bm=calloc(nb,1); int bit=0;
    for(int i=0;i<=c->n_layers;i++){
        int is_row = (i<c->n_layers && m->L[i].sparse) || (i==c->n_layers && has_mtp);
        if(!is_row) continue;
        for(int e=0;e<cols;e++,bit++)
            if(g_ehit[i][e]){ bm[bit>>3]|=1<<(bit&7); g_ehit[i][e]=0; }
    }
    char *hex=malloc((size_t)nb*2+1); int w=0;
    for(int b=0;b<nb;b++){ hex[w++]="0123456789abcdef"[bm[b]>>4]; hex[w++]="0123456789abcdef"[bm[b]&15]; }
    hex[w]=0;
    printf("HITS %d %d %s\n",rows,cols,hex); fflush(stdout); free(hex); free(bm);
}

/* The history format lives in route_trace.h so every engine writes the same bytes;
 * these keep the Model-shaped call sites unchanged. */
static void stats_dump_q(Model *m, const char *path, int quiet){ (void)m; rt_save(path,quiet); }
static void stats_dump(Model *m, const char *path){ stats_dump_q(m,path,0); }

static char g_usage_path[2100]="";
static int64_t usage_load(Model *m, const char *path){ (void)m; return rt_load(path); }
static void usage_save(Model *m){ if(g_usage_path[0]) stats_dump_q(m,g_usage_path,1); }

#endif /* TELEMETRY_H */
