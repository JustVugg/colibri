/* telemetry.h — dashboard protocol lines, stats/usage persistence, hardware probe.
 * Include after Model/Cfg/QT/ESlot/shards and st.h are defined; requires
 * qt_bytes(), now_s(), rss_gb(), edisk_s(), and the g_cuda_* globals (ifdef). */
#ifndef TELEMETRY_H
#define TELEMETRY_H

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
#else
    FILE *mi=fopen("/proc/meminfo","r");
    if(mi){ char ln[256]; double mt=0,ma=0;
        while(fgets(ln,sizeof(ln),mi)){
            if(sscanf(ln,"MemTotal: %lf",&mt)==1) *ram_total=mt/1e6;
            if(sscanf(ln,"MemAvailable: %lf",&ma)==1) *ram_avail=ma/1e6;
        } fclose(mi); }
#endif
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
    int ram=pinned-vram+lru; if(ram<0) ram=0;
    int disk=total-vram-ram; if(disk<0) disk=0;
    double eb=(double)expert_bytes_probe(m,m->ebits);
    printf("TIERS %d %d %d %.2f %.2f\n",vram,ram,disk,vram_gb,ram*eb/1e9);
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
            int tier=0;
            ESlot *P=m->pin[i];
            for(int z=0;z<m->npin[i];z++) if(P[z].eid==e){
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

static void stats_dump_q(Model *m, const char *path, int quiet){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"w"); if(!f){ if(!quiet) perror(tmp); return; }
    Cfg *c=&m->c; int64_t tot=0, nz=0;
    for(int i=0;i<=c->n_layers;i++){ if(!m->eusage[i]) continue;
        for(int e=0;e<c->n_experts;e++) if(m->eusage[i][e]){ fprintf(f,"%d %d %u\n",i,e,m->eusage[i][e]); tot+=m->eusage[i][e]; nz++; } }
    fclose(f); rename(tmp,path);
    if(!quiet) fprintf(stderr,"[STATS] %lld selections across %lld distinct experts -> %s\n",(long long)tot,(long long)nz,path);
}
static void stats_dump(Model *m, const char *path){ stats_dump_q(m,path,0); }

static char g_usage_path[2100]="";
static int64_t usage_load(Model *m, const char *path){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int l,e; uint32_t cnt; int64_t tot=0;
    while(fscanf(f,"%d %d %u",&l,&e,&cnt)==3)
        if(l>=0&&l<=c->n_layers&&e>=0&&e<c->n_experts&&m->eusage[l]){ m->eusage[l][e]+=cnt; tot+=cnt; }
    fclose(f); return tot;
}
static void usage_save(Model *m){ if(g_usage_path[0]) stats_dump_q(m,g_usage_path,1); }

#endif /* TELEMETRY_H */
