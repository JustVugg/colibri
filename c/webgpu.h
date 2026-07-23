/* WebGPU proxy client. The Python bridge terminates WebSocket frames and
 * forwards this compact binary batch to browser workers. */
#if !defined(_WIN32)
#define COLI_WEBGPU_MAGIC "COLIEX01"

static int webgpu_io(int fd, void *buf, size_t n, int write_mode){
    char *p=(char*)buf;
    while(n){
        ssize_t r=write_mode?send(fd,p,n,0):recv(fd,p,n,MSG_WAITALL);
        if(r<=0){ if(r<0&&errno==EINTR) continue; return -1; }
        p+=r; n-=(size_t)r;
    }
    return 0;
}
static int webgpu_u32(int fd, uint32_t *v, int write_mode){
    uint32_t x=write_mode?htonl(*v):0;
    if(webgpu_io(fd,write_mode?(void*)&x:(void*)v,sizeof(x),write_mode)) return -1;
    if(!write_mode) *v=ntohl(*v);
    return 0;
}
static int webgpu_connect(const char *spec, WebGPUWorker *out){
    char copy[256]; strncpy(copy,spec,sizeof(copy)-1); copy[sizeof(copy)-1]=0;
    char *colon=strrchr(copy,':'); if(!colon||colon==copy||!colon[1]) return -1;
    *colon=0; int port=atoi(colon+1); if(port<1||port>65535) return -1;
    char portbuf[16]; snprintf(portbuf,sizeof(portbuf),"%d",port);
    struct addrinfo hint={0},*ai=NULL; hint.ai_socktype=SOCK_STREAM;
    if(getaddrinfo(copy,portbuf,&hint,&ai)!=0) return -1;
    int fd=-1;
    for(struct addrinfo *p=ai;p;p=p->ai_next){
        fd=socket(p->ai_family,p->ai_socktype,p->ai_protocol);
        if(fd<0) continue;
        if(connect(fd,p->ai_addr,p->ai_addrlen)==0) break;
        close(fd); fd=-1;
    }
    freeaddrinfo(ai); if(fd<0) return -1;
    out->fd=fd; out->port=port; strncpy(out->host,copy,sizeof(out->host)-1); out->host[sizeof(out->host)-1]=0;
    return 0;
}
static void webgpu_close(void){
    if(g_webgpu_enabled && g_webgpu_worker.fd>=0) close(g_webgpu_worker.fd);
    g_webgpu_enabled=0;
}
static void webgpu_init(void){
    const char *list=getenv("WEBGPU_WORKERS"); if(!list||!*list) return;
    if(strchr(list,',')){ fprintf(stderr,"[WEBGPU] one proxy endpoint is supported in this slice\n"); exit(1); }
    if(webgpu_connect(list,&g_webgpu_worker)){ fprintf(stderr,"[WEBGPU] proxy %s is unreachable\n",list); exit(1); }
    g_webgpu_enabled=1;
    fprintf(stderr,"[WEBGPU] coordinator connected to proxy %s\n",list);
}

typedef struct { int eid,nr; int *rows; float *weights,*inputs; } WebGPUItem;
static void webgpu_item_free(WebGPUItem *item){ free(item->rows); free(item->weights); free(item->inputs); memset(item,0,sizeof(*item)); }
static int webgpu_item(const int *idxs,const float *ws,const int *keff,int K,int S,int eid,
                       WebGPUItem *item,int D,const float *x){
    item->eid=eid; item->nr=0;
    for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++) if(idxs[(int64_t)s*K+k]==eid){ item->nr++; break; }
    if(!item->nr) return 0;
    item->rows=malloc((size_t)item->nr*sizeof(int)); item->weights=malloc((size_t)item->nr*sizeof(float));
    item->inputs=falloc((int64_t)item->nr*D); int r=0;
    for(int s=0;s<S;s++) for(int k=0;k<keff[s];k++) if(idxs[(int64_t)s*K+k]==eid){
        item->rows[r]=s; item->weights[r]=ws[(int64_t)s*K+k];
        memcpy(item->inputs+(int64_t)r*D,x+(int64_t)s*D,(size_t)D*sizeof(float)); r++; break;
    }
    return 1;
}
static void webgpu_moe_batch(Model *m,int layer,float *x,int S,float *out,
                             const int *idxs,const float *ws,const int *keff,int K,
                             const int *uniq,int base,int nb){
    WebGPUWorker *w=&g_webgpu_worker; WebGPUItem items[64]; memset(items,0,sizeof(items)); int n=0,D=m->c.hidden;
    for(int j=0;j<nb;j++) if(n<64 && webgpu_item(idxs,ws,keff,K,S,uniq[base+j],&items[n],D,x)) n++;
    if(!n) return;
    uint32_t v; char magic[8];
    if(webgpu_io(w->fd,(void*)COLI_WEBGPU_MAGIC,8,1)) goto fail;
    v=1; if(webgpu_u32(w->fd,&v,1)) goto fail;
    v=(uint32_t)layer; if(webgpu_u32(w->fd,&v,1)) goto fail;
    v=(uint32_t)D; if(webgpu_u32(w->fd,&v,1)) goto fail;
    v=(uint32_t)m->c.moe_inter; if(webgpu_u32(w->fd,&v,1)) goto fail;
    v=(uint32_t)n; if(webgpu_u32(w->fd,&v,1)) goto fail;
    for(int j=0;j<n;j++){
        v=(uint32_t)items[j].eid; if(webgpu_u32(w->fd,&v,1)) goto fail;
        v=(uint32_t)items[j].nr; if(webgpu_u32(w->fd,&v,1)) goto fail;
        if(webgpu_io(w->fd,items[j].inputs,(size_t)items[j].nr*D*sizeof(float),1)) goto fail;
    }
    if(webgpu_io(w->fd,magic,8,0)||memcmp(magic,COLI_WEBGPU_MAGIC,8)) goto fail;
    if(webgpu_u32(w->fd,&v,0)||v!=1) goto fail;
    if(webgpu_u32(w->fd,&v,0)||v!=0) goto fail;
    if(webgpu_u32(w->fd,&v,0)||v!=(uint32_t)n) goto fail;
    for(int j=0;j<n;j++){
        uint32_t eid,nr;
        if(webgpu_u32(w->fd,&eid,0)||webgpu_u32(w->fd,&nr,0)||eid!=(uint32_t)items[j].eid||nr!=(uint32_t)items[j].nr) goto fail;
        float *y=falloc((int64_t)nr*D);
        if(webgpu_io(w->fd,y,(size_t)nr*D*sizeof(float),0)){ free(y); goto fail; }
        for(uint32_t r=0;r<nr;r++){ float *dst=out+(int64_t)items[j].rows[r]*D; float wt=items[j].weights[r];
            for(int d=0;d<D;d++) dst[d]+=wt*y[(int64_t)r*D+d]; }
        free(y);
    }
    for(int j=0;j<n;j++) webgpu_item_free(&items[j]);
    return;
fail:
    for(int j=0;j<n;j++) webgpu_item_free(&items[j]);
    fprintf(stderr,"[WEBGPU] proxy %s:%d failed during layer %d batch\n",w->host,w->port,layer); exit(1);
}
#endif
