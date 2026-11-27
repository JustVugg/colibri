/* coli_sycl_profile.cpp — the SYCL side of coli_sycl.h's sections 1 and 2: reading a device
 * into the profile (the decisions are coli_sycl_profile_rules.cpp's), the GPU list the engine's
 * ordinals index, the per-device context and queues, and the scratch arena. Host code only: no
 * kernel is defined here, so the unit compiles without -fsycl's device pass doing any work,
 * and nothing touches the runtime before the first call of section 2 (COLI_SYCL_LIST=1 is
 * answered at that call: every GPU's profile on stdout, exit 0). */
#include "coli_sycl.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

/* ---- 1. reading a device ----------------------------------------------------------------- */

/* The architecture's row. Every intel_gpu_* value of 2025.3's device_architecture.def is named
 * (the aliases share their target's value and need no case). A GPU whose IP version the runtime's
 * table does not know answers architecture::unknown (no exception; its value has 0x99 in the top
 * byte, so it is not in the Intel GPU family's 0x00 range), and a later runtime may answer a value
 * this switch has no case for: both are OTHER (the generic profile) on an Intel device, which
 * read_props decides by the vendor id, never by the name; anything else is NONE. */
static coli_sycl_arch arch_of(syclex::architecture a) {
    using A = syclex::architecture;
    switch (a) {
    case A::intel_gpu_pvc: case A::intel_gpu_pvc_vg: return COLI_SYCL_ARCH_PVC;
    case A::intel_gpu_bmg_g21: case A::intel_gpu_bmg_g31: return COLI_SYCL_ARCH_BMG;
    case A::intel_gpu_lnl_m: return COLI_SYCL_ARCH_LNL;
    case A::intel_gpu_ptl_h: case A::intel_gpu_ptl_u: return COLI_SYCL_ARCH_PTL;
    case A::intel_gpu_wcl: return COLI_SYCL_ARCH_WCL;
    case A::intel_gpu_arl_h: return COLI_SYCL_ARCH_ARL;
    case A::intel_gpu_mtl_u: case A::intel_gpu_mtl_h: return COLI_SYCL_ARCH_MTL;   /* arl_u, arl_s, mtl_s alias mtl_u */
    case A::intel_gpu_acm_g10: case A::intel_gpu_acm_g11: case A::intel_gpu_acm_g12: return COLI_SYCL_ARCH_DG2;
    case A::intel_gpu_tgllp: case A::intel_gpu_rkl: case A::intel_gpu_adl_s: case A::intel_gpu_adl_p:
    case A::intel_gpu_adl_n: case A::intel_gpu_dg1: return COLI_SYCL_ARCH_XELP;
    case A::intel_gpu_bdw: case A::intel_gpu_skl: case A::intel_gpu_kbl: case A::intel_gpu_cfl:
    case A::intel_gpu_apl: case A::intel_gpu_glk: case A::intel_gpu_whl: case A::intel_gpu_aml:
    case A::intel_gpu_cml: case A::intel_gpu_icllp: case A::intel_gpu_ehl: return COLI_SYCL_ARCH_GEN9;
    case A::unknown: return COLI_SYCL_ARCH_OTHER;   /* a GPU the 2025.3 runtime cannot name (is_gpu held by the caller) */
    default: return COLI_SYCL_ARCH_NONE;            /* a value with no row: read_props promotes an Intel device to OTHER */
    }
}

static coli_sycl_dtype dtype_of(syclex::matrix::matrix_type t) {
    using M = syclex::matrix::matrix_type;
    switch (t) {
    case M::fp16: return COLI_SYCL_DT_F16;   case M::bf16: return COLI_SYCL_DT_BF16;
    case M::sint8: return COLI_SYCL_DT_INT8; case M::tf32: return COLI_SYCL_DT_TF32;
    case M::fp32: return COLI_SYCL_DT_F32;   case M::sint32: return COLI_SYCL_DT_S32;
    default: return COLI_SYCL_DT_OTHER;
    }
}

/* one query of a device (or platform), its zero when the device or its backend declines (OpenCL
 * has no PCI address, say) */
template <typename Desc, typename Obj, typename T>
static T query(const Obj &o, T fallback) {
    try { return o.template get_info<Desc>(); } catch (const sycl::exception &) { return fallback; }
}

static coli_sycl_props read_props(const sycl::device &d) {
    coli_sycl_props pr;
    memset(&pr, 0, sizeof pr);
    if (!d.is_gpu()) return pr;   /* arch NONE: a CPU or accelerator device is never profiled */
    try {
        pr.arch = arch_of(d.get_info<syclex::info::device::architecture>());
    } catch (const sycl::exception &) {
        pr.arch = COLI_SYCL_ARCH_OTHER;   /* the query itself refused: the generic profile */
    }
    /* a value with no row on an Intel GPU (a part newer than the switch): the generic profile, by
     * the vendor id (0x8086) */
    if (pr.arch == COLI_SYCL_ARCH_NONE && query<sycl::info::device::vendor_id>(d, (uint32_t)0) == 0x8086u) pr.arch = COLI_SYCL_ARCH_OTHER;
    snprintf(pr.device_name, sizeof pr.device_name, "%s", query<sycl::info::device::name>(d, std::string("")).c_str());
    if (d.has(sycl::aspect::ext_intel_device_id)) pr.pci_device_id = query<sycl::ext::intel::info::device::device_id>(d, (uint32_t)0);
    if (d.has(sycl::aspect::ext_intel_pci_address))
        snprintf(pr.pci_address, sizeof pr.pci_address, "%s", query<sycl::ext::intel::info::device::pci_address>(d, std::string("")).c_str());
    {
        std::vector<size_t> sizes = query<sycl::info::device::sub_group_sizes>(d, std::vector<size_t>());
        std::sort(sizes.begin(), sizes.end());
        for (size_t s : sizes) if (pr.sg_count < 8) pr.sg_sizes[pr.sg_count++] = (int)s;
    }
    if (d.has(sycl::aspect::ext_intel_gpu_eu_count)) pr.eu_count = (int)query<sycl::ext::intel::info::device::gpu_eu_count>(d, (uint32_t)0);
    pr.max_compute_units = (int)query<sycl::info::device::max_compute_units>(d, (uint32_t)0);
    if (d.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu))
        pr.hw_threads_per_eu = (int)query<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>(d, (uint32_t)0);
    pr.slm_bytes = (size_t)query<sycl::info::device::local_mem_size>(d, (uint64_t)0);
    pr.max_wg_size = query<sycl::info::device::max_work_group_size>(d, (size_t)0);
    pr.global_mem_bytes = (size_t)query<sycl::info::device::global_mem_size>(d, (uint64_t)0);
    /* "integrated": SYCL 2020 deprecated host_unified_memory without a replacement for the
     * question (the usm_* aspects it points at are true on every Intel GPU); the runtime still
     * answers it from the driver, so it stays, with the warning off */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    pr.host_unified_memory = query<sycl::info::device::host_unified_memory>(d, false) ? 1 : 0;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    try {   /* the matrix engines' shapes: what the kernels may take is decided by the rules */
        std::vector<syclex::matrix::combination> cs = d.get_info<syclex::info::device::matrix_combinations>();
        std::vector<coli_sycl_combo> plain;
        for (const auto &c : cs) {
            coli_sycl_combo e;
            e.max_m = c.max_msize; e.max_n = c.max_nsize; e.max_k = c.max_ksize;
            e.m = c.msize; e.n = c.nsize; e.k = c.ksize;
            e.a = dtype_of(c.atype); e.b = dtype_of(c.btype); e.c = dtype_of(c.ctype);
            plain.push_back(e);
        }
        pr.xmx_shapes = coli_sycl_xmx_shape_flags(plain.data(), (int)plain.size());
    } catch (const sycl::exception &) {
        pr.xmx_shapes = 0;
    }
    return pr;
}

coli_sycl_profile coli_sycl_profile_detect(const sycl::device &d) {
    return coli_sycl_profile_derive(read_props(d), coli_sycl_env_read(), stderr);
}

/* ---- 2. the GPU list and the per-device state --------------------------------------------- */

/* The GPUs this process may use, enumerated once: every GPU device of every platform, the
 * Level Zero view only when it exists (the OpenCL platform lists the same cards again), in PCI
 * order where every device has an address (else the runtime's order). Enumeration faults
 * (no runtime, no driver) leave the list empty and are logged once. */
static const std::vector<sycl::device> &gpu_list(void) {
    static std::vector<sycl::device> list;
    static bool done = false;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (done) return list;
    done = true;
    /* the free-memory query (ext_intel_free_memory, what the engine's budgets read through
     * coli_sycl_mem_info) exists only when Level Zero's sysman is on before the driver
     * initialises: this is the process's first runtime call, so it is set here, never over a
     * value the environment chose (Dockerfile.sycl sets it for the image as well) */
    setenv("ZES_ENABLE_SYSMAN", "1", 0);
    try {
        std::vector<sycl::device> all = sycl::device::get_devices(sycl::info::device_type::gpu);
        bool have_l0 = false;
        for (const auto &d : all) if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) have_l0 = true;
        std::vector<std::pair<std::string, sycl::device>> keyed;
        bool every_address = true;
        for (const auto &d : all) {
            if (have_l0 && d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
            std::string addr = d.has(sycl::aspect::ext_intel_pci_address)
                               ? query<sycl::ext::intel::info::device::pci_address>(d, std::string("")) : std::string("");
            if (addr.empty()) every_address = false;
            keyed.emplace_back(addr, d);
        }
        if (every_address)
            std::stable_sort(keyed.begin(), keyed.end(),
                             [](const std::pair<std::string, sycl::device> &a, const std::pair<std::string, sycl::device> &b) { return a.first < b.first; });
        for (auto &k : keyed) list.push_back(k.second);
    } catch (const sycl::exception &e) {
        std::fprintf(stderr, "[SYCL] device discovery: %s\n", e.what());
        list.clear();
    }
    return list;
}

/* COLI_SYCL_LIST=1: the first-contact check. Answered at the first entry point that would touch
 * a device, so a run with it set never opens one. */
static void list_and_exit_if_asked(void) {
    const char *v = getenv("COLI_SYCL_LIST");
    if (!v || strcmp(v, "1")) return;
    coli_sycl_print_profiles();
    fflush(stdout);
    exit(0);
}

extern "C" void coli_sycl_print_profiles(void) {
    const std::vector<sycl::device> &list = gpu_list();
    printf("%zu GPU device%s\n", list.size(), list.size() == 1 ? "" : "s");
    /* an empty list is the usual first-contact failure, so it says what it means: nothing
     * was opened, no kernel ran, and the fix is outside the engine (the container's /dev/dri
     * and render group, or the driver) */
    if (list.empty())
        printf("  no SYCL GPU device: the runtime sees no Level Zero / OpenCL GPU (is /dev/dri passed to the\n"
               "  container and the user in the render group? sycl-ls says the same in the vendor's words)\n");
    for (size_t i = 0; i < list.size(); i++) {
        const sycl::device &d = list[i];
        coli_sycl_profile p = coli_sycl_profile_detect(d);
        coli_sycl_profile_print(stdout, (int)i, p);
        printf("  platform %s, driver %s\n",
               query<sycl::info::platform::name>(d.get_platform(), std::string("?")).c_str(),
               query<sycl::info::device::driver_version>(d, std::string("?")).c_str());
    }
}

int coli_sycl_available_device_count(void) {
    list_and_exit_if_asked();
    return (int)gpu_list().size();
}

/* The architecture listed GPU `ordinal` would have once open (its profile's, with the environment's
 * overrides), read without opening it: COLI_SYCL_ARCH_NONE past the list. For a decision taken before
 * coli_sycl_open (coli_cuda_tuned_default asked by an engine before its first init). Quiet: nothing is
 * said here that the open does not say again. */
coli_sycl_profile coli_sycl_listed_profile(int ordinal) {
    const std::vector<sycl::device> &l = gpu_list();
    if (ordinal < 0 || ordinal >= (int)l.size()) return {};
    return coli_sycl_profile_derive(read_props(l[(size_t)ordinal]), coli_sycl_env_read(), nullptr);
}
coli_sycl_arch coli_sycl_listed_arch(int ordinal) {
    return coli_sycl_listed_profile(ordinal).arch;
}

/* An open device: its context, the two in-order queues (compute, the chain's copies), the
 * profile, the arena's slots and the blocks the arena retired, built in that order so a failed
 * queue leaves nothing behind. The async handler reports and sets `lost`: an error the runtime
 * delivers this way (a command whose execution failed, handed to the handler at a wait_and_throw
 * or throw_asynchronous, never thrown to the caller) is what coli_sycl_lost answers, and the
 * units read it as they read a thrown exception (the chain's live / cc_try, the backend's
 * fault_injected). The object is made with new and never moved, so the handler holds its
 * address. */
#define COLI_SYCL_SLOTS 4
#define COLI_SYCL_RETIRED_MAX 8   /* retired blocks held before a grow must wait for the queues */
static sycl::async_handler async_handler(std::shared_ptr<coli_sycl_host_sync> state) {
    return [state](sycl::exception_list el) {
        for (auto &e : el) {
            try { std::rethrow_exception(e); }
            catch (const sycl::exception &ex) { std::fprintf(stderr, "[SYCL] async error (the device is lost): %s\n", ex.what()); }
            catch (...) { std::fprintf(stderr, "[SYCL] async error (the device is lost)\n"); }
        }
        state->cancel();
    };
}
struct Dev {
    std::shared_ptr<coli_sycl_host_sync> loss;  // constructed before handlers/queues
    sycl::device dev;
    sycl::context ctx;
    sycl::queue q, cq;
    coli_sycl_profile prof;
    struct { void *p; size_t bytes; } slot[COLI_SYCL_SLOTS];
    void *retired[COLI_SYCL_RETIRED_MAX]; int n_retired; size_t retired_bytes;
    int free_query;          /* the device answers ext_intel_free_memory (Level Zero with ZES_ENABLE_SYSMAN=1) */
    Dev(const sycl::device &d, const sycl::property_list &props,
        const sycl::context *shared = nullptr, std::shared_ptr<coli_sycl_host_sync> shared_loss = {})
        : loss(shared_loss ? shared_loss : std::make_shared<coli_sycl_host_sync>()), dev(d),
          ctx(shared ? *shared : sycl::context(d, handler())), q(ctx, d, handler(), props), cq(ctx, d, handler(), props) {
        memset(&prof, 0, sizeof prof);
        memset(slot, 0, sizeof slot);
        memset(retired, 0, sizeof retired); n_retired = 0; retired_bytes = 0;
        free_query = 0;
    }
    sycl::async_handler handler(void) {
        return async_handler(loss);
    }
};
/* the retired blocks, freed: the caller saw the queues pass whatever read them */
static void reap(Dev &dv) {
    for (int i = 0; i < dv.n_retired; i++) { try { sycl::free(dv.retired[i], dv.ctx); } catch (const sycl::exception &) {} }
    dv.n_retired = 0; dv.retired_bytes = 0;
}
/* a block out of use at the next frame's end: into the list; a full list (that many grows without
 * a frame end is not a shape this engine has) waits for the queues and frees first, logged */
static void retire(Dev &dv, int device, void *p, size_t bytes) {
    if (dv.n_retired == COLI_SYCL_RETIRED_MAX) {
        std::fprintf(stderr, "[SYCL] device %d: %d retired blocks without a frame end: waiting for the queues\n", device, dv.n_retired);
        try { dv.q.wait(); dv.cq.wait(); } catch (const sycl::exception &) {}
        reap(dv);
    }
    dv.retired[dv.n_retired++] = p; dv.retired_bytes += bytes;
}
/* Raw pointers, deleted by coli_sycl_close only: a process that exits with devices open leaves
 * them, since destroying queues and contexts during static destruction, after the runtime has
 * begun its own shutdown, is a known way to hang at exit. */
static Dev *g_dev[COLI_CUDA_MAX_DEVICES];                    /* by the engine's ordinal */
static int g_open[COLI_CUDA_MAX_DEVICES], g_nopen = 0;      /* the list coli_sycl_open took */
static bool g_peer_open = false;
static std::mutex g_mutex;                                   /* open / close / the arena */

static void fprof_enable(int level);   /* section 5, below */

static const coli_sycl_profile &empty_profile(void) {
    static coli_sycl_profile p;
    static bool made = false;
    if (!made) {
        memset(&p, 0, sizeof p);
        strcpy(p.name, "none");
        strcpy(p.device_name, "no device open");
        p.sg_default = 16;
        made = true;
    }
    return p;
}

int coli_sycl_open(const int *devices, int count) {
    list_and_exit_if_asked();
    const std::vector<sycl::device> &list = gpu_list();
    const int available = (int)list.size();
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!available) { std::fprintf(stderr, "[SYCL] no GPU device (COLI_SYCL_LIST=1 prints what the runtime sees)\n"); return 0; }
    /* the whole list is checked before anything is made, as coli_cuda_init does */
    for (int i = 0; i < count; i++) {
        if (devices[i] < 0 || devices[i] >= available) {
            std::fprintf(stderr, "[SYCL] invalid device %d (available: 0..%d)\n", devices[i], available - 1);
            return 0;
        }
        if (devices[i] >= COLI_CUDA_MAX_DEVICES) {   /* the per-device table is indexed by the ordinal */
            std::fprintf(stderr, "[SYCL] device %d: ordinals above %d are not supported\n", devices[i], COLI_CUDA_MAX_DEVICES - 1);
            return 0;
        }
        for (int j = 0; j < i; j++) if (devices[j] == devices[i]) {
            std::fprintf(stderr, "[SYCL] duplicate device %d\n", devices[i]);
            return 0;
        }
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_nopen) {   /* the same list keeps everything; another needs a close first */
        bool same = g_nopen == count;
        for (int i = 0; same && i < count; i++) {
            bool found = false;
            for (int j = 0; j < g_nopen; j++) if (g_open[j] == devices[i]) found = true;
            same = found;
        }
        if (!same) std::fprintf(stderr, "[SYCL] device list change requires shutdown first\n");
        return same;
    }
    const char *prof_env = getenv("COLI_SYCL_PROFILING");
    const char *fp_env = getenv("COLI_SYCL_FRAME_PROF");   /* section 5: per-command device times, needs profiling queues */
    const int fprof = fp_env && *fp_env && strcmp(fp_env, "0") ? (atoi(fp_env) > 1 ? 2 : 1) : 0;
    const bool profiling = (prof_env && !strcmp(prof_env, "1")) || fprof > 0;   /* event timestamps (cu_xb's elapsed times): off by default */
    if (fprof) fprof_enable(fprof);
    /* Experimental B60-only peer context. Require every directed pair and a
     * common Level Zero platform. A refusal leaves independent contexts. */
    std::unique_ptr<sycl::context> shared;
    std::shared_ptr<coli_sycl_host_sync> shared_loss;
    const char *peer_env = getenv("COLI_SYCL_PEER_CONTEXT");
    if (peer_env && !strcmp(peer_env, "1") && count >= 2 && count <= 4) {
        std::vector<sycl::device> peer;
        std::vector<std::pair<int,int>> enabled;
        try {
            bool capable = true;
            for (int i = 0; i < count; i++) {
                peer.push_back(list[devices[i]]);
                coli_sycl_profile p = coli_sycl_profile_detect(peer.back());
                if (p.arch != COLI_SYCL_ARCH_BMG || p.pci_device_id != 0xe211 ||
                    peer.back().get_platform().get_backend() != sycl::backend::ext_oneapi_level_zero ||
                    (i && peer.back().get_platform() != peer[0].get_platform())) capable = false;
            }
            for (int i = 0; capable && i < count; i++)
                for (int j = 0; capable && j < count; j++)
                    if (i != j && !peer[i].ext_oneapi_can_access_peer(peer[j])) capable = false;
            if (capable) {
                enabled.reserve((size_t)count * (count - 1));
                for (int i = 0; i < count; i++)
                    for (int j = 0; j < count; j++) if (i != j) {
                        peer[i].ext_oneapi_enable_peer_access(peer[j]); enabled.emplace_back(i,j);
                    }
                shared_loss = std::make_shared<coli_sycl_host_sync>();
                shared = std::make_unique<sycl::context>(peer, async_handler(shared_loss));
                g_peer_open = true;
                std::fprintf(stderr, "[SYCL] experimental B60 peer context enabled for %d devices\n", count);
            } else std::fprintf(stderr, "[SYCL] peer context unavailable; independent contexts retained\n");
        } catch (const std::exception &e) {
            std::fprintf(stderr, "[SYCL] peer context unavailable: %s; independent contexts retained\n", e.what());
            shared.reset(); shared_loss.reset();
            for (auto p : enabled) try { peer[p.first].ext_oneapi_disable_peer_access(peer[p.second]); } catch (const sycl::exception &) {}
        }
    }
    for (int i = 0; i < count; i++) {
        const int ordinal = devices[i];
        try {
            sycl::property_list props = profiling
                ? sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}
                : sycl::property_list{sycl::property::queue::in_order()};
            auto dv = std::make_unique<Dev>(list[ordinal], props, shared.get(), shared_loss);
            dv->prof = coli_sycl_profile_detect(dv->dev);
            dv->free_query = dv->dev.has(sycl::aspect::ext_intel_free_memory) ? 1 : 0;
            g_dev[ordinal] = dv.release();
            g_open[g_nopen++] = ordinal;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "[SYCL] device %d: cannot open: %s\n", ordinal, e.what());
            for (int j = 0; j < g_nopen; j++) { delete g_dev[g_open[j]]; g_dev[g_open[j]] = nullptr; }
            g_nopen = 0;
            shared.reset();
            if (g_peer_open) {
                for (int a = 0; a < count; a++) for (int b = 0; b < count; b++) if (a != b)
                    try { auto d=list[devices[a]]; d.ext_oneapi_disable_peer_access(list[devices[b]]); } catch (const sycl::exception &) {}
                g_peer_open = false;
            }
            return 0;
        }
        const coli_sycl_profile &p = g_dev[ordinal]->prof;
        std::fprintf(stderr, "[SYCL] device %d: %s, %.1f GB, profile %s, %d EUs, SLM %zu KiB, xmx %s, free memory %s%s\n",
                     ordinal, p.device_name, p.global_mem_bytes / 1e9, p.name, p.eu_count, p.slm_bytes / 1024,
                     p.has_xmx ? "on" : "off",
                     g_dev[ordinal]->free_query ? "from the device" : "estimated from the backend's books (no free_memory query: ZES_ENABLE_SYSMAN)",
                     profiling ? (fprof ? ", profiling queues, frame profile (COLI_SYCL_FRAME_PROF)" : ", profiling queues") : "");
    }
    return 1;
}

void coli_sycl_close(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    /* Peer work can access another device's allocation: drain all queues
     * before releasing the first device's memory. */
    for (int j = 0; j < g_nopen; j++) {
        Dev *dv = g_dev[g_open[j]];
        if (dv) try { dv->q.wait(); dv->cq.wait(); } catch (const sycl::exception &) {}
    }
    for (int j = 0; j < g_nopen; j++) {
        Dev *dv = g_dev[g_open[j]];
        if (!dv) continue;
        try {
            dv->q.wait(); dv->cq.wait();
            for (auto &s : dv->slot) if (s.p) { sycl::free(s.p, dv->ctx); s.p = nullptr; s.bytes = 0; }
            reap(*dv);
        } catch (const sycl::exception &e) {
            std::fprintf(stderr, "[SYCL] device %d: close: %s\n", g_open[j], e.what());
        }
        delete dv;
        g_dev[g_open[j]] = nullptr;
    }
    if (g_peer_open) {
        const auto &list = gpu_list();
        for (int a = 0; a < g_nopen; a++) for (int b = 0; b < g_nopen; b++) if (a != b)
            try { auto d=list[g_open[a]]; d.ext_oneapi_disable_peer_access(list[g_open[b]]); } catch (const sycl::exception &) {}
        g_peer_open = false;
    }
    g_nopen = 0;
}

int coli_sycl_is_open(int device) {
    return device >= 0 && device < COLI_CUDA_MAX_DEVICES && g_dev[device] != nullptr;
}
static Dev &dev_of(int device, const char *what) {
    if (!coli_sycl_is_open(device)) {
        std::fprintf(stderr, "[SYCL] %s: device %d is not open\n", what, device);
        abort();
    }
    return *g_dev[device];
}
sycl::device &coli_sycl_device(int device) { return dev_of(device, "device").dev; }
sycl::context &coli_sycl_context(int device) { return dev_of(device, "context").ctx; }
sycl::queue &coli_sycl_queue(int device) { return dev_of(device, "queue").q; }
sycl::queue &coli_sycl_copy_queue(int device) { return dev_of(device, "copy queue").cq; }
int coli_sycl_queue_device(const sycl::queue &q) {
    for (int j = 0; j < g_nopen; j++) {
        Dev *dv = g_dev[g_open[j]];
        if (dv && (dv->q == q || dv->cq == q)) return g_open[j];
    }
    return -1;
}
const coli_sycl_profile &coli_sycl_profile_for(int device) {
    return coli_sycl_is_open(device) ? g_dev[device]->prof : empty_profile();
}

/* The arena: one device allocation a slot, grown (never shrunk) to the largest request with
 * headroom (half again), 256-byte aligned, under the mutex (the engine's worker and its uploader
 * may ask at once). A grow never waits for the queues: the old block, which a kernel of an open
 * frame may still read, is retired and freed by coli_sycl_scratch_reap once the chain has seen
 * the queue pass the frames (cc_submit with a wait, cc_finish) or at close. A wait here could sit
 * behind a cc_wait_host kernel whose host side comes after the frame (the device-MoE forward is
 * one frame, and a hybrid layer's host part is served after the next layer is recorded): the
 * deadlock the chain's own attention scratch avoids the same way. A grow after a slot's first
 * allocation is logged, as an allocation's worth of time in the token loop. */
void *coli_sycl_scratch(int device, size_t bytes, int slot) {
    if (!coli_sycl_is_open(device) || slot < 0 || slot >= COLI_SYCL_SLOTS || !bytes) return nullptr;
    std::lock_guard<std::mutex> lock(g_mutex);
    Dev &dv = *g_dev[device];
    auto &s = dv.slot[slot];
    if (s.p && s.bytes >= bytes) return s.p;
    const size_t want = (bytes + bytes / 2 + 255) & ~(size_t)255;
    try {
        if (s.p) {
            std::fprintf(stderr, "[SYCL] device %d: scratch slot %d grows %zu -> %zu bytes (the old block is retired until the frame ends)\n",
                         device, slot, s.bytes, want);
            retire(dv, device, s.p, s.bytes);
            s.p = nullptr; s.bytes = 0;
        }
        void *p = sycl::aligned_alloc_device(256, want, dv.dev, dv.ctx);
        if (!p) return nullptr;
        s.p = p; s.bytes = want;
    } catch (const sycl::exception &e) {
        std::fprintf(stderr, "[SYCL] device %d: scratch slot %d: %s\n", device, slot, e.what());
        return nullptr;
    }
    return s.p;
}
size_t coli_sycl_scratch_bytes(int device, int slot) {
    if (!coli_sycl_is_open(device) || slot < 0 || slot >= COLI_SYCL_SLOTS) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_dev[device]->slot[slot].bytes;
}
void coli_sycl_scratch_reap(int device) {
    if (!coli_sycl_is_open(device)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    reap(*g_dev[device]);
}
int coli_sycl_retire(const sycl::queue &q, void *p, size_t bytes) {
    if (!p) return 1;
    /* the device's own queues only: those are what the chain's frame waits (the reap's trigger)
     * cover; a block of another queue on the same context (cu_xb's own, a test's) could still be
     * read when the chain's queue has passed its frames, so its owner waits on it and frees */
    const int device = coli_sycl_queue_device(q);
    if (device < 0) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    retire(*g_dev[device], device, p, bytes);
    return 1;
}

/* ---- the lost flag and the memory figure ------------------------------------------------- */

int coli_sycl_lost(int device) {
    if (!coli_sycl_is_open(device)) return 0;
    Dev &dv = *g_dev[device];
    if (!dv.loss->lost()) {   /* what the runtime collected since the last wait goes to the handler, which sets the flag */
        try { dv.q.throw_asynchronous(); dv.cq.throw_asynchronous(); }
        catch (const sycl::exception &e) { std::fprintf(stderr, "[SYCL] device %d: %s\n", device, e.what()); dv.loss->cancel(); }
    }
    return dv.loss->lost() ? 1 : 0;
}

std::shared_ptr<coli_sycl_host_sync> coli_sycl_host_state(int device) {
    return dev_of(device, "host state").loss;
}
void coli_sycl_mark_lost(int device) {
    if (coli_sycl_is_open(device)) g_dev[device]->loss->cancel();
}

/* the units' books of device memory, for the free figure where the device has no query: each
 * registers its function once (coli_cuda_init, cc_init, cu_open); read without the mutex, since
 * a ledger function reads its own unit's counters */
#define COLI_SYCL_LEDGERS 8
static coli_sycl_ledger_fn g_ledger[COLI_SYCL_LEDGERS];
static int g_nledger;
void coli_sycl_ledger_add(coli_sycl_ledger_fn fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (int i = 0; i < g_nledger; i++) if (g_ledger[i] == fn) return;
    if (g_nledger < COLI_SYCL_LEDGERS) g_ledger[g_nledger++] = fn;
}
int coli_sycl_mem_info(int device, size_t *free_bytes, size_t *total_bytes, int *queried) {
    if (!coli_sycl_is_open(device)) return 0;
    Dev &dv = *g_dev[device];
    size_t total = 0, fr = 0;
    try {
        total = (size_t)dv.dev.get_info<sycl::info::device::global_mem_size>();
        if (dv.free_query) fr = (size_t)dv.dev.get_info<sycl::ext::intel::info::device::free_memory>();
    } catch (const sycl::exception &e) {
        std::fprintf(stderr, "[SYCL] device %d: memory info: %s\n", device, e.what());
        return 0;
    }
    if (!dv.free_query) {   /* the total less what this backend holds: the arena, then every unit's books */
        size_t booked = 0;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            for (auto &s : dv.slot) booked += s.bytes;
            booked += dv.retired_bytes;
        }
        for (int i = 0; i < g_nledger; i++) booked += g_ledger[i](device);
        fr = total > booked ? total - booked : 0;
    }
    if (free_bytes) *free_bytes = fr;
    if (total_bytes) *total_bytes = total;
    if (queried) *queried = dv.free_query;
    return 1;
}

/* ---- 5. frame profiling (coli_sycl.h section 5) ------------------------------------------ */
int coli_sycl_fprof_flag = 0;
namespace {
struct FpEv { sycl::event ev; std::string kind; };
struct FpKind { unsigned long long n = 0, ns = 0, untimed = 0; };
struct FpBucket {   /* the frames of one command count */
    unsigned long long frames = 0, cmds = 0, timed = 0, untimed = 0, gaps = 0, gap_ns = 0, busy_ns = 0, span_ns = 0;
    unsigned long long gap_hist[4] = {0, 0, 0, 0};   /* gaps under 5 us, 5-20, 20-100, over 100 */
    double host_ms = 0, submit_ms = 0, wait_ms = 0;
    std::map<std::string, FpKind> kinds;
};
struct FpState {
    std::mutex mu;
    std::vector<FpEv> pending;
    std::map<unsigned, FpBucket> buckets;
    int level = 0, dumped = 0;
    unsigned long long frames = 0, wrapped = 0;
};
/* on the heap and never deleted: sycl::events must not be destroyed in static destruction */
FpState *g_fp = nullptr;
}
static void fprof_enable(int level) {
    if (!g_fp) g_fp = new FpState;
    g_fp->level = level;
    coli_sycl_fprof_flag = 1;
}
void coli_sycl_fprof_push(const sycl::event &e) {
    if (!g_fp) return;
    std::lock_guard<std::mutex> lock(g_fp->mu);
    g_fp->pending.push_back(FpEv{e, std::string()});
}
void coli_sycl_fprof_name(const char *kind) {
    if (!g_fp || !kind) return;
    std::lock_guard<std::mutex> lock(g_fp->mu);
    for (size_t i = g_fp->pending.size(); i-- > 0 && g_fp->pending[i].kind.empty(); ) g_fp->pending[i].kind = kind;
}
void coli_sycl_fprof_frame(double host_ms, double submit_ms, double wait_ms) {
    if (!g_fp) return;
    std::vector<FpEv> evs;
    { std::lock_guard<std::mutex> lock(g_fp->mu); evs.swap(g_fp->pending); }
    if (evs.empty()) return;
    struct T { unsigned long long s, e; int ok; };
    std::vector<T> t(evs.size());
    unsigned long long first = 0, last = 0, busy = 0, gap = 0, ngap = 0, ntimed = 0, prev_end = 0, hist[4] = {0, 0, 0, 0};
    bool wrapped = false;
    for (size_t i = 0; i < evs.size(); i++) {
        t[i] = T{0, 0, 0};
        try {   /* a host task's or a barrier's event may have no timestamps: untimed, counted apart */
            unsigned long long s = evs[i].ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            unsigned long long e = evs[i].ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (s && e >= s) t[i] = T{s, e, 1};
        } catch (const sycl::exception &) {} catch (const std::exception &) {}
        if (!t[i].ok) continue;
        if (!ntimed) first = t[i].s;
        else if (t[i].s + 1000000000ull < prev_end) wrapped = true;   /* the device timer wrapped inside this frame: dropped */
        else if (t[i].s > prev_end) {
            unsigned long long g = t[i].s - prev_end;
            gap += g; ngap++; hist[g < 5000 ? 0 : g < 20000 ? 1 : g < 100000 ? 2 : 3]++;
        }
        if (t[i].e > last) last = t[i].e;
        if (t[i].e > prev_end) prev_end = t[i].e;
        busy += t[i].e - t[i].s; ntimed++;
    }
    g_fp->frames++;
    if (wrapped || (ntimed && last - first > 60000000000ull)) { g_fp->wrapped++; return; }
    FpBucket &b = g_fp->buckets[(unsigned)evs.size()];
    b.frames++; b.cmds += evs.size(); b.timed += ntimed; b.untimed += evs.size() - ntimed;
    b.gaps += ngap; b.gap_ns += gap; b.busy_ns += busy; b.span_ns += ntimed ? last - first : 0;
    for (int k = 0; k < 4; k++) b.gap_hist[k] += hist[k];
    b.host_ms += host_ms; b.submit_ms += submit_ms; b.wait_ms += wait_ms;
    for (size_t i = 0; i < evs.size(); i++) {
        FpKind &k = b.kinds[evs[i].kind.empty() ? std::string("(unnamed)") : evs[i].kind];
        k.n++;
        if (t[i].ok) k.ns += t[i].e - t[i].s; else k.untimed++;
    }
    if (g_fp->level >= 2 && !g_fp->dumped && b.frames == 8) {   /* one frame command by command: the eighth of its bucket */
        g_fp->dumped = 1;
        std::fprintf(stderr, "[SYCL] frame profile: frame %llu (%zu commands) command by command: index, kind, start (us from the "
                             "frame's first), duration (us), idle before it (us)\n", g_fp->frames, evs.size());
        unsigned long long pe = 0;
        for (size_t i = 0; i < evs.size(); i++) {
            if (!t[i].ok) { std::fprintf(stderr, "[SYCL]   %4zu %-34s untimed\n", i, evs[i].kind.c_str()); continue; }
            double gp = pe && t[i].s > pe ? (double)(t[i].s - pe) / 1e3 : 0.0;
            std::fprintf(stderr, "[SYCL]   %4zu %-34s %10.1f %8.1f %8.1f\n", i, evs[i].kind.c_str(),
                         (double)(t[i].s - first) / 1e3, (double)(t[i].e - t[i].s) / 1e3, gp);
            if (t[i].e > pe) pe = t[i].e;
        }
    }
}
void coli_sycl_fprof_report(void) {
    if (!g_fp) return;
    static int done = 0;
    if (done) return;
    done = 1;
    std::fprintf(stderr, "[SYCL] frame profile (COLI_SYCL_FRAME_PROF): %llu frames waited for, %llu dropped (timer wrap); by commands a frame:\n",
                 g_fp->frames, g_fp->wrapped);
    const FpBucket *modal = nullptr; unsigned mkey = 0;
    for (auto &kv : g_fp->buckets) if (!modal || kv.second.frames > modal->frames) { modal = &kv.second; mkey = kv.first; }
    for (auto &kv : g_fp->buckets) {
        const FpBucket &b = kv.second; const double f = (double)b.frames;
        std::fprintf(stderr, "[SYCL]   %u commands x %llu frames: host %.2f ms a frame (submitting %.2f, waiting %.2f); device span %.2f ms, "
                             "busy %.2f ms (%.0f %% of the span), idle between commands %.2f ms over %.0f gaps (mean %.1f us; <5 us %.0f, "
                             "5-20 %.0f, 20-100 %.0f, >100 %.0f); %.1f commands untimed\n",
                     kv.first, b.frames, b.host_ms / f, b.submit_ms / f, b.wait_ms / f, b.span_ns / f / 1e6, b.busy_ns / f / 1e6,
                     b.span_ns ? 100.0 * (double)b.busy_ns / (double)b.span_ns : 0.0, b.gap_ns / f / 1e6, b.gaps / f,
                     b.gaps ? (double)b.gap_ns / (double)b.gaps / 1e3 : 0.0, b.gap_hist[0] / f, b.gap_hist[1] / f, b.gap_hist[2] / f,
                     b.gap_hist[3] / f, b.untimed / f);
    }
    if (!modal) return;
    std::vector<std::pair<std::string, FpKind>> rows(modal->kinds.begin(), modal->kinds.end());
    std::sort(rows.begin(), rows.end(), [](const std::pair<std::string, FpKind> &a, const std::pair<std::string, FpKind> &b) { return a.second.ns > b.second.ns; });
    const double f = (double)modal->frames, busy = (double)modal->busy_ns;
    std::fprintf(stderr, "[SYCL]   the %u-command frames, per frame: %-34s %9s %10s %9s %7s\n", mkey, "kind", "launches", "device ms", "us each", "% busy");
    for (auto &r : rows)
        std::fprintf(stderr, "[SYCL]     %-46s %9.1f %10.3f %9.1f %7.1f%s\n", r.first.c_str(), r.second.n / f, r.second.ns / f / 1e6,
                     r.second.n > r.second.untimed ? (double)r.second.ns / (double)(r.second.n - r.second.untimed) / 1e3 : 0.0,
                     busy > 0 ? 100.0 * (double)r.second.ns / busy : 0.0, r.second.untimed ? " (untimed)" : "");
}
