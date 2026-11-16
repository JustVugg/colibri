/* The optional CUDA helper for WEX records (docs/wex.md): kaira_wex_cuda.dll on
 * Windows, kaira_wex_cuda.so elsewhere, built by `make wex-cuda-helper`. The
 * engines never link it: wex_decode.c opens it when COLI_WEX_CUDA_LIB names it.
 * Pooled workspaces with private CUDA streams keep cold-load reconstruction
 * separate from inference streams. A failed decode never changes the caller's
 * compressed record, and never writes the caller's destination.
 *
 * ABI 1 (wex_cuda_helper_abi) is unchanged: wex_cuda_decode_to_host.
 * wex_cuda_helper_features() lists optional additions, resolved by name:
 *   WEX_CUDA_FEATURE_DEVICE   wex_cuda_decode_to_device: reconstruct straight
 *                             into caller-owned device buffers (no GPU->RAM trip)
 *   WEX_CUDA_FEATURE_REGISTER wex_cuda_host_register / _unregister for the
 *                             caller's staging buffers
 *   WEX_CUDA_FEATURE_NR2      wex_cuda_register_model; then wex_cuda_decode_to_host
 *                             and wex_cuda_decode_to_device also accept WNR2
 *                             records (entry->flags == WEX_ENTRY_NEURAL_RANS2)
 *   WEX_CUDA_FEATURE_BATCH    wex_cuda_decode_batch_to_device: a layer's WNR2 misses
 *                             in one call, pipelined over lanes, one host wait
 *   WEX_CUDA_FEATURE_NR3      the three decode entry points also accept WNR3
 *                             records (entry->flags == WEX_ENTRY_NEURAL_RANS3, same
 *                             model registry); a record failing its integrity check
 *                             (hash, ceil, canonical zero) returns -1
 *   WEX_CUDA_FEATURE_BATCH_HOST wex_cuda_decode_batch_to_host: the batch pipeline,
 *                             each WNR2/WNR3 record decoded into a caller host buffer
 *                             (canonical raw slab), for stores that fill RAM slots
 */
#include "../wex_cuda.cuh"
#ifdef _WIN32
#define WEX_CUDA_EXPORT extern "C" __declspec(dllexport)
#else
#define WEX_CUDA_EXPORT extern "C" __attribute__((visibility("default")))
#endif
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

struct WexCudaWorkspace {
    unsigned char *host_raw=nullptr,*device_record=nullptr,*device_raw=nullptr;
    uint32_t *host_offsets=nullptr,*device_offsets=nullptr,*host_error=nullptr,*device_error=nullptr,*lut=nullptr;
    cudaStream_t stream=nullptr;
    cudaStream_t aux=nullptr;              /* WNR2: kernels, while `stream` uploads */
    cudaEvent_t head=nullptr,data=nullptr; /* WNR2: record head / whole record on the device */
    cudaEvent_t consumed=nullptr;          /* batch: the kernels no longer read device_record */
    size_t capacity=0,tiles=0,lut_bytes=0;
    int device=-1;
    ~WexCudaWorkspace() { clear(); }
    void clear() {
        if(aux) { cudaStreamSynchronize(aux); cudaStreamDestroy(aux); }
        if(head) cudaEventDestroy(head);
        if(data) cudaEventDestroy(data);
        if(consumed) cudaEventDestroy(consumed);
        aux=nullptr; head=data=consumed=nullptr;
        if(stream) cudaStreamSynchronize(stream);
        if(device_record) cudaFree(device_record);
        if(device_raw) cudaFree(device_raw);
        if(device_offsets) cudaFree(device_offsets);
        if(device_error) cudaFree(device_error);
        if(lut) cudaFree(lut);
        if(host_raw) cudaFreeHost(host_raw);
        if(host_offsets) cudaFreeHost(host_offsets);
        if(host_error) cudaFreeHost(host_error);
        if(stream) cudaStreamDestroy(stream);
        host_raw=device_record=device_raw=nullptr;
        host_offsets=device_offsets=host_error=device_error=lut=nullptr;
        stream=nullptr; capacity=tiles=lut_bytes=0; device=-1;
    }
    /* with_host=false (batch lanes): no page-locked host_raw, they never copy back */
    bool ensure(size_t bytes,size_t tile_count,size_t lookup_bytes,bool with_host=true) {
        int current=-1;
        if(cudaGetDevice(&current)!=cudaSuccess) return false;
        if(device==current && capacity>=bytes && tiles>=tile_count && lut_bytes>=lookup_bytes &&
           (host_raw || !with_host)) return true;
        if(device>=0 && device!=current) { int keep=current; cudaSetDevice(device); clear(); cudaSetDevice(keep); }
        else clear();
        size_t free_bytes,total_bytes;
        if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess ||
           free_bytes<bytes*2u+tile_count*4u+lookup_bytes+8u*1024u*1024u) return false;
#define ALLOC_OK(call) do { if((call)!=cudaSuccess) { clear(); return false; } } while(0)
        ALLOC_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        if(with_host) ALLOC_OK(cudaMallocHost((void**)&host_raw,bytes));
        ALLOC_OK(cudaMallocHost((void**)&host_offsets,tile_count*4u));
        ALLOC_OK(cudaMallocHost((void**)&host_error,4u));
        ALLOC_OK(cudaMalloc((void**)&device_record,bytes));
        ALLOC_OK(cudaMalloc((void**)&device_raw,bytes));
        ALLOC_OK(cudaMalloc((void**)&device_offsets,tile_count*4u));
        ALLOC_OK(cudaMalloc((void**)&device_error,4u));
        ALLOC_OK(cudaMalloc((void**)&lut,lookup_bytes));
#undef ALLOC_OK
        capacity=bytes; tiles=tile_count; lut_bytes=lookup_bytes; device=current;
        return true;
    }
    /* After ensure(): the second stream and two events of the WNR2 path. */
    bool ensure_nr2() {
        if(aux) return true;
        if(cudaStreamCreateWithFlags(&aux,cudaStreamNonBlocking)!=cudaSuccess) { aux=nullptr; return false; }
        if(cudaEventCreateWithFlags(&head,cudaEventDisableTiming)!=cudaSuccess ||
           cudaEventCreateWithFlags(&data,cudaEventDisableTiming)!=cudaSuccess ||
           cudaEventCreateWithFlags(&consumed,cudaEventDisableTiming)!=cudaSuccess) {
            if(head) cudaEventDestroy(head);
            if(data) cudaEventDestroy(data);
            cudaStreamDestroy(aux); aux=nullptr; head=data=consumed=nullptr;
            return false;
        }
        return true;
    }
};

struct WexCudaWaitOnExit {
    cudaStream_t stream;
    bool completed=false;
    ~WexCudaWaitOnExit() { if(!completed) cudaStreamSynchronize(stream); }
};

/* ---- workspace pools ------------------------------------------------------
 * A call takes a workspace from a pool for its duration and gives it back, so
 * the helper holds as many as there were calls in flight at once, however many
 * threads ever called it. Workspaces kept per thread leaked: the engines' reader
 * teams retire and respawn threads (libgomp, a reader pool per load), and a
 * thread-exit destructor cannot free CUDA memory on Windows (cudaFree there
 * fails with cudaErrorDevicesUnavailable), so every exited thread kept its
 * ~2x record bytes of device memory and page-locked host memory until the
 * device ran out and every later record went to the CPU. A pool and its idle
 * workspaces are never destroyed (CUDA cannot be called while the process
 * unloads). take() prefers a workspace last used on the current device. */
template<class T> struct WexCudaPool {
    std::mutex mutex;
    std::vector<T*> idle;
    T *take() {
        int current=-1;
        if(cudaGetDevice(&current)!=cudaSuccess) { (void)cudaGetLastError(); current=-1; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if(!idle.empty()) {
                size_t pick=idle.size()-1;
                for(size_t i=idle.size();i-->0;) if(idle[i]->device()==current) { pick=i; break; }
                T *w=idle[pick];
                idle.erase(idle.begin()+(std::ptrdiff_t)pick);
                return w;
            }
        }
        return new(std::nothrow) T();
    }
    void give(T *w) {
        if(!w) return;
        std::lock_guard<std::mutex> lock(mutex);
        try { idle.push_back(w); } catch(...) { delete w; }   /* no room in the list: free it here */
    }
};
template<class T> struct WexCudaLease {
    WexCudaPool<T> &pool;
    T *w;
    explicit WexCudaLease(WexCudaPool<T> &p) : pool(p), w(p.take()) {}
    ~WexCudaLease() { pool.give(w); }
    WexCudaLease(const WexCudaLease &)=delete;
    WexCudaLease &operator=(const WexCudaLease &)=delete;
};
/* a single-record workspace (the to-host and to-device entry points) */
struct WexCudaSingle {
    WexCudaWorkspace workspace;
    int device() const { return workspace.device; }
};

/* ---- WNR2 model registry --------------------------------------------------
 * Process-global: a private host copy of every registered WNM2 blob, keyed by
 * its id (FNV-1a), and its device tables, built on first use per device and
 * kept for the process lifetime (never freed, so a found pointer stays valid). */
#define WEX_CUDA_NR2_MAX_MODELS 16
#define WEX_CUDA_NR2_MAX_DEVICES 64
struct WexCudaNr2Model {
    uint32_t id;
    unsigned char blob[WEX_NR2_MODEL_BYTES];
    unsigned char *tables[WEX_CUDA_NR2_MAX_DEVICES];
};
static std::mutex wex_cuda_nr2_mutex;
static WexCudaNr2Model *wex_cuda_nr2_models[WEX_CUDA_NR2_MAX_MODELS];
static int wex_cuda_nr2_count;

static WexCudaNr2Model *wex_cuda_nr2_find(uint32_t id) {   /* caller holds the mutex */
    for(int i=0;i<wex_cuda_nr2_count;i++) if(wex_cuda_nr2_models[i]->id==id) return wex_cuda_nr2_models[i];
    return nullptr;
}

/* 0 and the device tables of model `id` on the current device `device`,
 * building them on `stream` the first time; 1 if the model is not registered
 * or CUDA fails. */
static int wex_cuda_nr2_tables(uint32_t id,int device,cudaStream_t stream,WexCudaNr2Tables *tables) {
    std::lock_guard<std::mutex> lock(wex_cuda_nr2_mutex);
    WexCudaNr2Model *model=wex_cuda_nr2_find(id);
    if(!model || device<0 || device>=WEX_CUDA_NR2_MAX_DEVICES) return 1;
    const int buckets=WEX_CUDA_NR2_LOOKUP==0;
    if(!model->tables[device]) {
        unsigned char *base=nullptr;
        if(cudaMalloc((void**)&base,wex_cuda_nr2_table_bytes(buckets))!=cudaSuccess) { (void)cudaGetLastError(); return 1; }
        WexCudaNr2Tables built;
        if(cudaMemcpyAsync(base,model->blob,WEX_NR2_MODEL_BYTES,cudaMemcpyHostToDevice,stream)!=cudaSuccess ||
           wex_cuda_build_nr2_tables(base,buckets,stream,&built)!=cudaSuccess ||
           cudaStreamSynchronize(stream)!=cudaSuccess) {
            (void)cudaGetLastError(); cudaStreamSynchronize(stream); cudaFree(base); (void)cudaGetLastError();
            return 1;
        }
        model->tables[device]=base;
    }
    wex_cuda_nr2_table_view(model->tables[device],buckets,tables);
    return 0;
}

static int wex_cuda_nr2_registered(uint32_t id) {
    std::lock_guard<std::mutex> lock(wex_cuda_nr2_mutex);
    return wex_cuda_nr2_find(id)!=nullptr;
}

/* Register a WNM2 model blob (validated with wex_nr2_model_open; the helper
 * keeps its own copy). 0 when registered, or already registered with the same
 * bytes; -1 for an invalid blob or another blob with the same id; 1 when the
 * table (16 models) is full or memory is short. */
WEX_CUDA_EXPORT int wex_cuda_register_model(const unsigned char *blob,size_t bytes) {
    WexNr2Model probe;
    if(wex_nr2_model_open(blob,bytes,&probe)) return -1;
    std::lock_guard<std::mutex> lock(wex_cuda_nr2_mutex);
    if(WexCudaNr2Model *known=wex_cuda_nr2_find(probe.id))
        return std::memcmp(known->blob,blob,WEX_NR2_MODEL_BYTES) ? -1 : 0;
    if(wex_cuda_nr2_count>=WEX_CUDA_NR2_MAX_MODELS) return 1;
    WexCudaNr2Model *model=new(std::nothrow) WexCudaNr2Model();
    if(!model) return 1;
    model->id=probe.id;
    std::memcpy(model->blob,blob,WEX_NR2_MODEL_BYTES);
    wex_cuda_nr2_models[wex_cuda_nr2_count++]=model;
    return 0;
}

/* Host validation of a WNR2 record (entry->total_bytes at `record`, host
 * memory): -1 if invalid, 1 if its model is not registered, else 0. */
static int wex_cuda_nr2_prepare(const WexEntry *entry,const unsigned char *record,WexNr2Record &view,
                                WexCudaNr2Plan &plan) {
    if(wex_cuda_plan_nr2(entry,record,entry->total_bytes,&view,&plan)) return -1;
    return wex_cuda_nr2_registered(plan.model_id) ? 0 : 1;
}

/* Once per process: a WNR3 record was rejected by its integrity checks. */
static void wex_cuda_nr3_report(uint32_t error) {
    static std::atomic<int> reported{0};
    if((error & (WEX_CUDA_ERR_HASH|WEX_CUDA_ERR_CEIL|WEX_CUDA_ERR_CZ)) && !reported.exchange(1))
        std::fprintf(stderr,"[wex] WNR3 record failed its integrity check (error 0x%x); rejected\n",error);
}

/* Prepared record (host memory) -> workspace.device_raw on the current device.
 * `stream` only copies: the stream offsets, then the record in two pieces,
 * header + lengths + scale payload first (host-to-device copies share one
 * copy engine in submission order, so the small ones go first). `aux` runs
 * the kernels: kernel A as soon as the head is on the device, overlapping the
 * weight payload upload, then kernel B once all of it arrived. Then, with
 * `commit`, the commit kernel copies to the caller's device buffers only if no
 * stream failed; without it, device_raw goes back to host_raw (private).
 * (Measured alternatives were slower on Windows/WDDM: kernel B on `stream`
 * behind the upload, or the offsets uploaded on `aux`; every switch between
 * the copy engine and kernels costs ~15-35 us there.)
 * Returns 0, 1 (CUDA/resources unavailable) or -1 (a stream failed on the
 * device); every copy and kernel has completed when it returns. */
static int wex_cuda_nr2_decode(const WexEntry *entry,const unsigned char *record,const WexNr2Record &view,
                               const WexCudaNr2Plan &plan,WexCudaWorkspace &workspace,const WexCudaCommit *commit) {
    uint64_t raw,aligned;
    if(wex_entry_raw_sizes(entry,&raw,&aligned)) return -1;
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess) { (void)cudaGetLastError(); return 1; }
    if(!workspace.ensure((size_t)aligned,(size_t)plan.streams+1u,4u) || !workspace.ensure_nr2()) {
        (void)cudaGetLastError(); return 1;
    }
    WexCudaNr2Tables tables;
    if(wex_cuda_nr2_tables(plan.model_id,device,workspace.stream,&tables)) return 1;
    cudaStream_t copy=workspace.stream,work=workspace.aux;
    WexCudaWaitOnExit copy_guard{copy},work_guard{work};
#define RUN_OK(call) do { if((call)!=cudaSuccess) { (void)cudaGetLastError(); return 1; } } while(0)
    wex_cuda_nr2_offsets(&view,&plan,workspace.host_offsets);
    RUN_OK(cudaMemcpyAsync(workspace.device_offsets,workspace.host_offsets,((size_t)plan.streams+1u)*4u,
                           cudaMemcpyHostToDevice,copy));
    RUN_OK(cudaMemcpyAsync(workspace.device_record,record,plan.weight_payload,cudaMemcpyHostToDevice,copy));
    RUN_OK(cudaEventRecord(workspace.head,copy));
    RUN_OK(cudaMemcpyAsync(workspace.device_record+plan.weight_payload,record+plan.weight_payload,
                           entry->total_bytes-plan.weight_payload,cudaMemcpyHostToDevice,copy));
    RUN_OK(cudaEventRecord(workspace.data,copy));
    RUN_OK(cudaStreamWaitEvent(work,workspace.head,0));
    RUN_OK(cudaMemsetAsync(workspace.device_error,0,4u,work));
    RUN_OK(wex_cuda_launch_nr2_scales(&plan,&tables,workspace.device_record,workspace.device_offsets,
                                      workspace.device_raw,workspace.device_error,work));
    RUN_OK(cudaStreamWaitEvent(work,workspace.data,0));
    RUN_OK(wex_cuda_launch_nr2_weights<WEX_CUDA_NR2_LOOKUP>(&plan,&tables,workspace.device_record,
           workspace.device_offsets,workspace.device_raw,workspace.device_error,work));
    if(commit) RUN_OK(wex_cuda_launch_commit(workspace.device_raw,commit,workspace.device_error,work));
    else if(plan.weight_base==entry->scale_bytes)
        RUN_OK(cudaMemcpyAsync(workspace.host_raw,workspace.device_raw,(size_t)raw,cudaMemcpyDeviceToHost,work));
    else {
        RUN_OK(cudaMemcpyAsync(workspace.host_raw,workspace.device_raw,entry->scale_bytes,cudaMemcpyDeviceToHost,work));
        RUN_OK(cudaMemcpyAsync(workspace.host_raw+entry->scale_bytes,workspace.device_raw+plan.weight_base,
                               entry->weight_bytes,cudaMemcpyDeviceToHost,work));
    }
    RUN_OK(cudaMemcpyAsync(workspace.host_error,workspace.device_error,4u,cudaMemcpyDeviceToHost,work));
    RUN_OK(cudaStreamSynchronize(work));
    RUN_OK(cudaStreamSynchronize(copy));   /* idle already: `work` waited for its last event */
#undef RUN_OK
    copy_guard.completed=work_guard.completed=true;
    wex_cuda_nr3_report(*workspace.host_error);
    return *workspace.host_error ? -1 : 0;
}

/* the single-record pools: to host (with page-locked host_raw) and to device */
static WexCudaPool<WexCudaSingle> &wex_cuda_host_pool() { static auto *pool=new WexCudaPool<WexCudaSingle>(); return *pool; }
static WexCudaPool<WexCudaSingle> &wex_cuda_device_pool() { static auto *pool=new WexCudaPool<WexCudaSingle>(); return *pool; }

static int wex_cuda_nr2_to_host(const WexEntry *entry,unsigned char *slab,size_t slab_capacity) {
    uint64_t raw,aligned;
    if(wex_entry_raw_sizes(entry,&raw,&aligned) || aligned>slab_capacity) return -1;
    size_t offset=wex_record_read_offset(entry);
    if(!offset || offset>slab_capacity || entry->total_bytes>slab_capacity-offset) return -1;
    WexNr2Record view;
    WexCudaNr2Plan plan;
    int status=wex_cuda_nr2_prepare(entry,slab+offset,view,plan);
    if(status) return status;
    WexCudaLease<WexCudaSingle> lease(wex_cuda_host_pool());
    if(!lease.w) return 1;
    /* the record is fully uploaded before the decoded slab comes back */
    status=wex_cuda_nr2_decode(entry,slab+offset,view,plan,lease.w->workspace,nullptr);
    if(status) return status;
    std::memcpy(slab,lease.w->workspace.host_raw,(size_t)raw);
    return 0;
}

WEX_CUDA_EXPORT unsigned int wex_cuda_helper_abi(void) { return 1u; }

WEX_CUDA_EXPORT int wex_cuda_decode_to_host(const WexEntry *entry,
    unsigned char *slab,size_t slab_capacity) {
    if(entry && slab && wex_entry_nr_model(entry)) return wex_cuda_nr2_to_host(entry,slab,slab_capacity);
    uint64_t raw,aligned;
    WexNeuralRans view;
    if(!entry || !slab || entry->flags!=WEX_ENTRY_NEURAL_RANS ||
       wex_entry_raw_sizes(entry,&raw,&aligned) || aligned>slab_capacity || entry->scale_bytes%16u ||
       entry->total_bytes>=aligned) return -1;
    size_t offset=(size_t)(aligned-entry->total_bytes);
    if(wex_nr_parse(slab+offset,entry->total_bytes,entry->weight_bytes,&view) ||
       wex_unpack_scale_tiles(view.scales,view.scale_encoded,nullptr,entry->scale_bytes)) return -1;
    WexCudaLease<WexCudaSingle> lease(wex_cuda_host_pool());
    if(!lease.w) return 1;
    WexCudaWorkspace &workspace=lease.w->workspace;
    size_t tiles=((size_t)entry->scale_bytes+255u)/256u;
    size_t lut_bytes=(size_t)view.classes*(view.scale_only?1u:17u)*WEX_CUDA_NR_LUT_SLOTS*4u;
    if(!workspace.ensure((size_t)aligned,tiles,lut_bytes)) return 1;
    if(wex_cuda_neural_tile_offsets(&view,entry->scale_bytes,workspace.host_offsets,tiles)) return -1;
    WexCudaWaitOnExit guard{workspace.stream};
#define RUN_OK(call) do { if((call)!=cudaSuccess) return 1; } while(0)
    /* The caller retains its slab until the stream is synchronized below.
     * CUDA can consume an already-registered cache slab directly, or stage a
     * pageable slab itself. Do not pin it here: the engine owns its lifetime.
     * Device validation and D2H still finish in private scratch before commit. */
    RUN_OK(cudaMemcpyAsync(workspace.device_record,slab+offset,entry->total_bytes,cudaMemcpyHostToDevice,workspace.stream));
    RUN_OK(cudaMemcpyAsync(workspace.device_offsets,workspace.host_offsets,tiles*4u,cudaMemcpyHostToDevice,workspace.stream));
    RUN_OK(wex_cuda_launch_neural_rans(entry,&view,workspace.device_record,workspace.device_offsets,
                                      workspace.device_raw,workspace.lut,workspace.device_error,workspace.stream));
    RUN_OK(cudaMemcpyAsync(workspace.host_error,workspace.device_error,4u,cudaMemcpyDeviceToHost,workspace.stream));
    RUN_OK(cudaMemcpyAsync(workspace.host_raw,workspace.device_raw,(size_t)raw,cudaMemcpyDeviceToHost,workspace.stream));
    RUN_OK(cudaStreamSynchronize(workspace.stream)); guard.completed=true;
#undef RUN_OK
    if(*workspace.host_error) return -1;
    std::memcpy(slab,workspace.host_raw,(size_t)raw);
    return 0;
}

#define WEX_CUDA_FEATURE_DEVICE   1u
#define WEX_CUDA_FEATURE_REGISTER 2u
#define WEX_CUDA_FEATURE_NR2      4u
#define WEX_CUDA_FEATURE_BATCH    8u
#define WEX_CUDA_FEATURE_NR3      16u
#define WEX_CUDA_FEATURE_BATCH_HOST 32u
WEX_CUDA_EXPORT unsigned int wex_cuda_helper_features(void) {
    return WEX_CUDA_FEATURE_DEVICE | WEX_CUDA_FEATURE_REGISTER | WEX_CUDA_FEATURE_NR2 | WEX_CUDA_FEATURE_BATCH |
           WEX_CUDA_FEATURE_NR3 | WEX_CUDA_FEATURE_BATCH_HOST;
}

static int wex_cuda_nr2_to_device(const WexEntry *entry,const unsigned char *record,int device,
    unsigned char *const *dst_scale,unsigned char *const *dst_weight) {
    uint64_t raw,aligned;
    if(wex_entry_raw_sizes(entry,&raw,&aligned) || entry->total_bytes>=aligned) return -1;
    for(int m=0;m<3;m++) if(!dst_scale[m] || !dst_weight[m]) return -1;
    WexNr2Record view;
    WexCudaNr2Plan plan;
    int status=wex_cuda_nr2_prepare(entry,record,view,plan);
    if(status) return status;
    if(cudaSetDevice(device)!=cudaSuccess) { (void)cudaGetLastError(); return 1; }
    /* the commit kernel writes the destination only when every stream decoded */
    WexCudaCommit commit;
    for(int m=0;m<3;m++) {
        commit.destination[m]=dst_scale[m];
        commit.source[m]=plan.scale_offset[m];
        commit.bytes[m]=(uint32_t)entry->scale_shapes[m][0]*entry->scale_shapes[m][1];
        commit.destination[3+m]=dst_weight[m];
        commit.source[3+m]=plan.weight_offset[m];
        commit.bytes[3+m]=(uint32_t)entry->weight_shapes[m][0]*entry->weight_shapes[m][1];
    }
    WexCudaLease<WexCudaSingle> lease(wex_cuda_device_pool());
    if(!lease.w) return 1;
    return wex_cuda_nr2_decode(entry,record,view,plan,lease.w->workspace,&commit);
}

/* Reconstruct a Neural-rANS record straight into device memory that the caller
 * owns (for example the inference backend's expert mirror): matrix m's scales
 * go to dst_scale[m] and its weights to dst_weight[m], in the canonical WEX
 * layout of that matrix. The compressed record is read from host memory
 * (entry->total_bytes, page-locked for full speed). Pointers from another CUDA
 * runtime of the same process are valid: both use the device's primary context.
 * The destination is written only after the decode reported no error, and the
 * call returns after every copy completed, so the caller may use the buffers
 * on any stream. Returns 0, 1 when CUDA/resources are unavailable, -1 when the
 * input is invalid (the destination is then untouched). WNR2 records (with
 * WEX_CUDA_FEATURE_NR2) need their model registered first, else 1; their
 * destination is written by a device-side commit that runs only if every
 * stream decoded (so a CUDA failure reported after that commit, returning 1,
 * leaves correct data there, never partial or corrupt data). */
WEX_CUDA_EXPORT int wex_cuda_decode_to_device(const WexEntry *entry,
    const unsigned char *record,int device,unsigned char *const *dst_scale,unsigned char *const *dst_weight) {
    if(entry && record && dst_scale && dst_weight && wex_entry_nr_model(entry))
        return wex_cuda_nr2_to_device(entry,record,device,dst_scale,dst_weight);
    uint64_t raw,aligned;
    WexNeuralRans view;
    if(!entry || !record || !dst_scale || !dst_weight || entry->flags!=WEX_ENTRY_NEURAL_RANS ||
       wex_entry_raw_sizes(entry,&raw,&aligned) || entry->scale_bytes%16u || entry->total_bytes>=aligned) return -1;
    for(int m=0;m<3;m++) if(!dst_scale[m] || !dst_weight[m]) return -1;
    if(wex_nr_parse(record,entry->total_bytes,entry->weight_bytes,&view) ||
       wex_unpack_scale_tiles(view.scales,view.scale_encoded,nullptr,entry->scale_bytes)) return -1;
    if(cudaSetDevice(device)!=cudaSuccess) { (void)cudaGetLastError(); return 1; }
    WexCudaLease<WexCudaSingle> lease(wex_cuda_device_pool());
    if(!lease.w) return 1;
    WexCudaWorkspace &workspace=lease.w->workspace;
    size_t tiles=((size_t)entry->scale_bytes+255u)/256u;
    size_t lut_bytes=(size_t)view.classes*(view.scale_only?1u:17u)*WEX_CUDA_NR_LUT_SLOTS*4u;
    if(!workspace.ensure((size_t)aligned,tiles,lut_bytes)) return 1;
    if(wex_cuda_neural_tile_offsets(&view,entry->scale_bytes,workspace.host_offsets,tiles)) return -1;
    WexCudaWaitOnExit guard{workspace.stream};
#define RUN_OK(call) do { if((call)!=cudaSuccess) { (void)cudaGetLastError(); return 1; } } while(0)
    RUN_OK(cudaMemcpyAsync(workspace.device_record,record,entry->total_bytes,cudaMemcpyHostToDevice,workspace.stream));
    RUN_OK(cudaMemcpyAsync(workspace.device_offsets,workspace.host_offsets,tiles*4u,cudaMemcpyHostToDevice,workspace.stream));
    RUN_OK(wex_cuda_launch_neural_rans(entry,&view,workspace.device_record,workspace.device_offsets,
                                      workspace.device_raw,workspace.lut,workspace.device_error,workspace.stream));
    RUN_OK(cudaMemcpyAsync(workspace.host_error,workspace.device_error,4u,cudaMemcpyDeviceToHost,workspace.stream));
    RUN_OK(cudaStreamSynchronize(workspace.stream));
    if(*workspace.host_error) { guard.completed=true; return -1; }
    for(int m=0;m<3;m++) {
        size_t scale_bytes=(size_t)entry->scale_shapes[m][0]*entry->scale_shapes[m][1];
        size_t weight_bytes=(size_t)entry->weight_shapes[m][0]*entry->weight_shapes[m][1];
        RUN_OK(cudaMemcpyAsync(dst_scale[m],workspace.device_raw+entry->scale_offsets[m],scale_bytes,
                               cudaMemcpyDeviceToDevice,workspace.stream));
        RUN_OK(cudaMemcpyAsync(dst_weight[m],workspace.device_raw+entry->weight_offsets[m],weight_bytes,
                               cudaMemcpyDeviceToDevice,workspace.stream));
    }
    RUN_OK(cudaStreamSynchronize(workspace.stream)); guard.completed=true;
#undef RUN_OK
    return 0;
}

/* device_raw (plan layout) -> the canonical raw slab at host dst, on `work` */
static cudaError_t wex_cuda_copy_back(const WexEntry *entry,const WexCudaNr2Plan &plan,const unsigned char *device_raw,
                                      unsigned char *dst,cudaStream_t work) {
    if(plan.weight_base==entry->scale_bytes)
        return cudaMemcpyAsync(dst,device_raw,(size_t)entry->scale_bytes+entry->weight_bytes,cudaMemcpyDeviceToHost,work);
    cudaError_t e=cudaMemcpyAsync(dst,device_raw,entry->scale_bytes,cudaMemcpyDeviceToHost,work);
    return e!=cudaSuccess ? e : cudaMemcpyAsync(dst+entry->scale_bytes,device_raw+plan.weight_base,entry->weight_bytes,
                                                cudaMemcpyDeviceToHost,work);
}

/* ---- batched WNR2 device decode (WEX_CUDA_FEATURE_BATCH) -------------------
 * One call for a layer's misses instead of one per expert. Records spread over
 * up to WEX_CUDA_BATCH_LANES lanes, each a device workspace with its own copy
 * stream (`stream`) and kernel stream (`aux`); every copy and kernel is queued
 * first and the host waits once at the end, so uploads overlap other lanes'
 * kernels. A lane's next record waits for event `consumed` (the lane's
 * previous kernels no longer read device_record/device_offsets). Per-record
 * offsets and error words live in page-locked scratch that outlives the call's
 * queued copies. Measured on an RTX 5090 (WNR2 records of DeepSeek V4): 0.28-0.30
 * ms per 11 MiB record at 6-8 records, against ~1.0 ms per single call. */
#define WEX_CUDA_BATCH_LANES 8
struct WexCudaBatchHost {
    uint32_t *offsets=nullptr,*errors=nullptr;
    size_t offset_cap=0,error_cap=0;
    ~WexCudaBatchHost() { if(offsets) cudaFreeHost(offsets); if(errors) cudaFreeHost(errors); }
    static bool grow(uint32_t *&p,size_t &cap,size_t want) {
        if(want<=cap) return true;
        if(p) cudaFreeHost(p);   /* the previous call synchronized every copy that read it */
        p=nullptr; cap=0;
        if(cudaMallocHost((void**)&p,want*4u)!=cudaSuccess) { (void)cudaGetLastError(); p=nullptr; return false; }
        cap=want;
        return true;
    }
};
/* one batch call's lanes and page-locked scratch, from a pool like the single workspaces */
struct WexCudaBatchSet {
    WexCudaWorkspace lanes[WEX_CUDA_BATCH_LANES];
    WexCudaBatchHost host;
    int device() const { return lanes[0].device; }
};
static WexCudaPool<WexCudaBatchSet> &wex_cuda_batch_pool() { static auto *pool=new WexCudaPool<WexCudaBatchSet>(); return *pool; }

/* n WNR2 records (host memory, page-locked for full speed) to their caller-
 * owned device buffers: record i's matrix m goes to dst_scale[3*i+m] and
 * dst_weight[3*i+m]. status[i] on return: 0 decoded and written; -1 invalid
 * record or destination (untouched); 1 CUDA/resources unavailable or model not
 * registered (the destination then holds correct data or is untouched, never a
 * partial write: the device-side commit runs only after a clean decode).
 * Every copy and kernel has completed when the call returns. Returns the
 * number of records whose status is not 0.
 * With dst_host (and no dst_scale/dst_weight): record i's raw slab goes to the
 * host buffer dst_host[i] instead, copied back only once its error word is 0. */
static int wex_cuda_batch(int n,const WexEntry *const *entries,const unsigned char *const *records,int device,
    unsigned char *const *dst_scale,unsigned char *const *dst_weight,unsigned char *const *dst_host,int *status) {
    if(n<=0) return 0;
    if(!entries || !records || !(dst_host || (dst_scale && dst_weight)) || !status) return n;
    const int PENDING=2;
    std::vector<WexNr2Record> views((size_t)n);
    std::vector<WexCudaNr2Plan> plans((size_t)n);
    std::vector<size_t> offset_at((size_t)n,0);
    size_t offsets=0,max_aligned=0,max_streams=0;
    int pending=0;
    for(int i=0;i<n;i++) {
        status[i]=-1;
        const WexEntry *entry=entries[i];
        uint64_t raw,aligned;
        if(!entry || !records[i] || !wex_entry_nr_model(entry) ||
           wex_entry_raw_sizes(entry,&raw,&aligned) || entry->total_bytes>=aligned) continue;
        bool targets=dst_host ? dst_host[i]!=nullptr : true;
        if(!dst_host) for(int m=0;m<3;m++) if(!dst_scale[3*i+m] || !dst_weight[3*i+m]) targets=false;
        if(!targets) continue;
        int prepared=wex_cuda_nr2_prepare(entry,records[i],views[(size_t)i],plans[(size_t)i]);
        if(prepared) { status[i]=prepared; continue; }
        status[i]=PENDING; pending++;
        offset_at[(size_t)i]=offsets;
        offsets+=(size_t)plans[(size_t)i].streams+1u;
        if((size_t)aligned>max_aligned) max_aligned=(size_t)aligned;
        if((size_t)plans[(size_t)i].streams+1u>max_streams) max_streams=(size_t)plans[(size_t)i].streams+1u;
    }
    auto finish=[&](int value) { for(int i=0;i<n;i++) if(status[i]==PENDING) status[i]=value; };
    auto failures=[&] { int bad=0; for(int i=0;i<n;i++) bad+=status[i]!=0; return bad; };
    if(!pending) return failures();
    if(cudaSetDevice(device)!=cudaSuccess) { (void)cudaGetLastError(); finish(1); return failures(); }
    WexCudaLease<WexCudaBatchSet> lease(wex_cuda_batch_pool());
    if(!lease.w) { finish(1); return failures(); }
    WexCudaBatchHost &host=lease.w->host;
    WexCudaWorkspace *lane=lease.w->lanes;
    const int lanes=pending<WEX_CUDA_BATCH_LANES ? pending : WEX_CUDA_BATCH_LANES;
    if(!WexCudaBatchHost::grow(host.offsets,host.offset_cap,offsets) ||
       !WexCudaBatchHost::grow(host.errors,host.error_cap,(size_t)n)) { finish(1); return failures(); }
    for(int l=0;l<lanes;l++)
        if(!lane[l].ensure(max_aligned,max_streams,4u,false) || !lane[l].ensure_nr2()) {
            (void)cudaGetLastError(); finish(1); return failures();
        }
    /* record i's upload and kernels on lane w (and, for device destinations, its
     * commit), then its error word; 0 = queued or refused before queueing (status
     * set), -1 = a CUDA call failed: the lane's order is unknown */
    auto queue=[&](int i,WexCudaWorkspace &w)->int {
        const WexEntry *entry=entries[i];
        const WexCudaNr2Plan &plan=plans[(size_t)i];
        WexCudaNr2Tables tables;
        if(wex_cuda_nr2_tables(plan.model_id,device,w.stream,&tables)) { status[i]=1; return 0; }
        uint32_t *host_offsets=host.offsets+offset_at[(size_t)i];
        wex_cuda_nr2_offsets(&views[(size_t)i],&plan,host_offsets);
        WexCudaCommit commit;
        for(int m=0;m<3 && !dst_host;m++) {
            commit.destination[m]=dst_scale[3*i+m];
            commit.source[m]=plan.scale_offset[m];
            commit.bytes[m]=(uint32_t)entry->scale_shapes[m][0]*entry->scale_shapes[m][1];
            commit.destination[3+m]=dst_weight[3*i+m];
            commit.source[3+m]=plan.weight_offset[m];
            commit.bytes[3+m]=(uint32_t)entry->weight_shapes[m][0]*entry->weight_shapes[m][1];
        }
        cudaStream_t copy=w.stream,work=w.aux;
        const unsigned char *record=records[i];
        if(cudaStreamWaitEvent(copy,w.consumed,0)!=cudaSuccess ||
           cudaMemcpyAsync(w.device_offsets,host_offsets,((size_t)plan.streams+1u)*4u,cudaMemcpyHostToDevice,copy)!=cudaSuccess ||
           cudaMemcpyAsync(w.device_record,record,plan.weight_payload,cudaMemcpyHostToDevice,copy)!=cudaSuccess ||
           cudaEventRecord(w.head,copy)!=cudaSuccess ||
           cudaMemcpyAsync(w.device_record+plan.weight_payload,record+plan.weight_payload,
                           entry->total_bytes-plan.weight_payload,cudaMemcpyHostToDevice,copy)!=cudaSuccess ||
           cudaEventRecord(w.data,copy)!=cudaSuccess ||
           cudaStreamWaitEvent(work,w.head,0)!=cudaSuccess ||
           cudaMemsetAsync(w.device_error,0,4u,work)!=cudaSuccess ||
           wex_cuda_launch_nr2_scales(&plan,&tables,w.device_record,w.device_offsets,w.device_raw,w.device_error,work)!=cudaSuccess ||
           cudaStreamWaitEvent(work,w.data,0)!=cudaSuccess ||
           wex_cuda_launch_nr2_weights<WEX_CUDA_NR2_LOOKUP>(&plan,&tables,w.device_record,w.device_offsets,
                                                            w.device_raw,w.device_error,work)!=cudaSuccess ||
           cudaEventRecord(w.consumed,work)!=cudaSuccess ||
           (!dst_host && wex_cuda_launch_commit(w.device_raw,&commit,w.device_error,work)!=cudaSuccess) ||
           cudaMemcpyAsync(host.errors+i,w.device_error,4u,cudaMemcpyDeviceToHost,work)!=cudaSuccess) {
            (void)cudaGetLastError();
            return -1;
        }
        return 0;
    };
    auto drain=[&](int count)->bool {   /* every queued copy and kernel of lanes 0..count-1 */
        bool clean=true;
        for(int l=0;l<count;l++) {
            if(cudaStreamSynchronize(lane[l].aux)!=cudaSuccess) clean=false;
            if(cudaStreamSynchronize(lane[l].stream)!=cudaSuccess) clean=false;
        }
        if(!clean) (void)cudaGetLastError();
        return clean;
    };
    /* a queued call that fails leaves the lane's order unknown: stop queueing,
     * wait for what is queued, and report it and the rest as unavailable */
    int broken_at=n;
    bool synced=true;
    if(!dst_host) {
        /* device destinations: the commit kernel writes them only after a clean
         * decode, so every record is queued at once and the host waits once */
        for(int i=0,k=0;i<n && broken_at==n;i++)
            if(status[i]==PENDING && queue(i,lane[k++%lanes])) broken_at=i;
        synced=drain(lanes);
        for(int i=0;i<n;i++)
            if(status[i]==PENDING) {
                status[i]=i>=broken_at || !synced ? 1 : host.errors[i] ? -1 : 0;
                if(status[i]==-1) wex_cuda_nr3_report(host.errors[i]);
            }
        return failures();
    }
    /* host destinations: no device-side step can gate a copy to host memory, so
     * the records go in rounds of one per lane: their decodes, one host wait for
     * their error words, then a copy back of only the records that decoded clean,
     * and a second wait. A refused record leaves dst[i] untouched, as every other
     * entry point does (WNR3-SPEC 4.7). Each copy back starts after its record's
     * whole upload and kernels, so dst[i] may overlap record i itself. */
    int round[WEX_CUDA_BATCH_LANES];
    for(int i=0;i<n && broken_at==n && synced;) {
        int used=0;
        for(;i<n && used<lanes;i++) {
            if(status[i]!=PENDING) continue;
            if(queue(i,lane[used])) { broken_at=i; break; }
            if(status[i]==PENDING) round[used++]=i;
        }
        if(!(synced=drain(used))) break;
        for(int r=0;r<used;r++) {
            const int j=round[r];
            if(host.errors[j]) { status[j]=-1; wex_cuda_nr3_report(host.errors[j]); continue; }
            if(wex_cuda_copy_back(entries[j],plans[(size_t)j],lane[r].device_raw,dst_host[j],lane[r].aux)!=cudaSuccess) {
                (void)cudaGetLastError();
                status[j]=1;   /* dst[j] may hold part of the copy: CUDA failed, the record did not */
                continue;
            }
            status[j]=-2;      /* copied back once the lanes drain */
        }
        bool copied=drain(used);
        for(int r=0;r<used;r++) if(status[round[r]]==-2) status[round[r]]=copied ? 0 : 1;
        synced=copied;
    }
    finish(1);   /* not queued: a CUDA failure stopped the call first */
    return failures();
}

WEX_CUDA_EXPORT int wex_cuda_decode_batch_to_device(int n,const WexEntry *const *entries,
    const unsigned char *const *records,int device,unsigned char *const *dst_scale,
    unsigned char *const *dst_weight,int *status) {
    if(!dst_scale || !dst_weight) return n>0 ? n : 0;
    return wex_cuda_batch(n,entries,records,device,dst_scale,dst_weight,nullptr,status);
}

/* WEX_CUDA_FEATURE_BATCH_HOST: n WNR2/WNR3 records (host memory, page-locked for
 * full speed) decoded on the GPU into caller host buffers: dst[i] receives
 * record i's raw slab (scale_bytes + weight_bytes, canonical WEX layout, as
 * wex_cuda_decode_to_host leaves at the slab start); dst[i] must not overlap any
 * OTHER record, but may overlap record i itself (a store that decodes in place,
 * the record in its slab's tail): the copy back starts only after the host saw
 * record i's error word, so after its whole upload and kernels -- keep that order
 * in any rework. dst[i] is best page-locked (wex_cuda_host_register).
 * status[i]: 0 decoded and written; -1 invalid record or a failed integrity
 * check, dst[i] untouched; 1 CUDA/resources unavailable or model not registered:
 * dst[i] untouched, except after a CUDA failure during the copy back itself,
 * which may leave part of a correct slab there (so a caller treats a slot whose
 * status is not 0 as empty). Every copy has completed when the call returns.
 * Returns the number of records whose status is not 0. */
WEX_CUDA_EXPORT int wex_cuda_decode_batch_to_host(int n,const WexEntry *const *entries,
    const unsigned char *const *records,int device,unsigned char *const *dst,int *status) {
    if(!dst) return n>0 ? n : 0;
    return wex_cuda_batch(n,entries,records,device,nullptr,nullptr,dst,status);
}

/* Page-lock a caller staging buffer for this helper's uploads (portable). */
WEX_CUDA_EXPORT int wex_cuda_host_register(void *pointer,size_t bytes) {
    if(!pointer || !bytes) return -1;
    if(cudaHostRegister(pointer,bytes,cudaHostRegisterPortable)!=cudaSuccess) { (void)cudaGetLastError(); return 1; }
    return 0;
}
WEX_CUDA_EXPORT int wex_cuda_host_unregister(void *pointer) {
    if(!pointer) return -1;
    if(cudaHostUnregister(pointer)!=cudaSuccess) { (void)cudaGetLastError(); return 1; }
    return 0;
}
