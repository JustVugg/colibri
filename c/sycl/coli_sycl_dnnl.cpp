/* coli_sycl_dnnl.cpp — a prompt chunk's attention through oneDNN's fused attention (the Graph API's
 * scaled dot-product attention). Optional twice over: compiled only in a SYCL_DNNL=1 build (which
 * defines COLI_SYCL_DNNL and links the toolkit's libdnnl; the runtime image's redist carries it),
 * taken with COLI_SYCL_DNNL_ATTN=1 or 2, or unset where every card is a Max (the measured default, head
 * size 256). Elsewhere coli_sycl_stubs.cpp's weak definitions decline.
 *
 * The chain's plain causal prompt chunk (cuda_chain.dp.cpp, cc_attn_t): Q float32 in its padded
 * rows, the head-major float16 cache (K[(kvh*cap + t)*hd + d] from k_off), O float32, the output
 * gate. One operation: Q -> float16 [S][H][hd] (a small kernel); the partition (MatMul, scale, the
 * bottom-right causal mask from GenIndex/Add/Subtract/GreaterEqual/Select, SoftMax, MatMul; the KV
 * heads broadcast over their query heads, oneDNN's GQA form) reading K and V straight from the
 * cache; O float16 -> o float32, times sigmoid(gate) where the layer has one, as cc_attn_gk does.
 *
 * Measured on a Max 1100 (campaign DNNL1, DNNL2): oneDNN 3.9.1 runs it as its micro-kernel attention
 * (sdpa ocl:micro:reusable), 0.11 x the grouped kernel's time at 2 048 rows. One partition per exact
 * (rows, keys) shape: in 3.9.1 a partition declared for the cache's capacity, with the length as the
 * mask's run-time scalar, gives wrong results. The first compile in a process takes about 0.55 s,
 * later shapes about 0.5 ms (the kernel is reused); a device keeps the last 64. K and V must start
 * 16-byte aligned (a start 6 bytes in gives wrong results without an error); others decline. The
 * fused kernel needs the matrix engines: a device without them declines (oneDNN could pick an unfused
 * partition there, the whole score matrix in scratch memory). A head layout (H, KV heads, hd) oneDNN
 * refuses once is not asked again. The scale goes as float16: exact for hd 64 and 256.
 *   COLI_SYCL_DNNL_ATTN=1  oneDNN's engine and that first compile are made on a thread of its own:
 *                          the chunks that come meanwhile keep the chain's kernels (a short prompt
 *                          in a fresh process never waits for them); 2: on the caller, every
 *                          eligible chunk from the first (the tests, the quality gates).
 * No host wait inside a frame (one there can hang behind a host task the chain serves later): grown
 * buffers go to coli_sycl_retire, oneDNN's own frees wait for their events here, never on the host.
 * Only the queue's submissions may throw out of coli_sycl_dnnl_attn (a device failure, which cc_try
 * reports); anything else declines, and turns the path off for the process where retrying is useless.
 * Not bit-identical to cc_attn_gk: float16 Q, P and O, oneDNN's own order. */
#include "coli_sycl.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef COLI_SYCL_DNNL
#include "backend_cuda.h"   /* coli_cuda_tuned_default */
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {
namespace dg = dnnl::graph;
using LT = dg::logical_tensor;
using DT = LT::data_type;
using Dims = LT::dims;
using Clock = std::chrono::steady_clock;

/* oneDNN's allocator. A free with an event (the memory may still be in use) waits in g_pending for
 * that event, reaped at the next call, never by a host wait; at release all are freed */
struct Pending { void *p; sycl::context ctx; sycl::event ev; };
std::mutex g_free_mx;
std::vector<Pending> *g_pending = new std::vector<Pending>;   // never destroyed, as g_devs
void *dn_alloc(size_t size, size_t align, const void *dev, const void *ctx) {
    try {
        return sycl::aligned_alloc_device(align > 64 ? align : 64, size, *static_cast<const sycl::device *>(dev),
                                          *static_cast<const sycl::context *>(ctx));
    } catch (const std::exception &) { return nullptr; }
}
void dn_free(void *ptr, const void *, const void *ctx, void *event) {
    if (!ptr) return;
    const sycl::context &c = *static_cast<const sycl::context *>(ctx);
    if (!event) { try { sycl::free(ptr, c); } catch (const std::exception &) {} return; }
    std::lock_guard<std::mutex> lk(g_free_mx);
    g_pending->push_back({ptr, c, *static_cast<sycl::event *>(event)});
}
void reap_pending(bool all) {   // all: at release, after the queue's wait
    std::lock_guard<std::mutex> lk(g_free_mx);
    for (size_t i = 0; i < g_pending->size();) {
        Pending &r = (*g_pending)[i];
        bool done = all;
        if (!done) {
            try { done = r.ev.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete; }
            catch (const std::exception &) { done = false; }
        }
        if (!done) { i++; continue; }
        try { sycl::free(r.p, r.ctx); } catch (const std::exception &) {}
        (*g_pending)[i] = g_pending->back();
        g_pending->pop_back();
    }
}

/* COLI_SYCL_DNNL_ATTN: 0 off, 1 the first compile in the background, 2 on the caller; unset, the value measured
 * on the devices' architecture (coli_cuda_tuned_default: 1 where every opened card is a Max, which it says on
 * stderr; COLI_PLATFORM_DEFAULTS=0 keeps 0), and then for head size 256 only: the one measured end to end
 * (Qwen3.6-35B-A3B, campaign DNNL3); 64 and 128 pass the device tests and are taken with the switch set. Asked at
 * the first prompt chunk, after the devices are open. */
struct Mode { int m; bool dflt; };
const Mode &mode_info() {
    static const Mode mi = [] {
        const char *e = std::getenv("COLI_SYCL_DNNL_ATTN");
        if (e && *e) return Mode{(!std::strcmp(e, "1") || !std::strcmp(e, "2")) ? *e - '0' : 0, false};
        const int t = coli_cuda_tuned_default("COLI_SYCL_DNNL_ATTN", 0);
        return Mode{t == 1 || t == 2 ? t : 0, true};
    }();
    return mi;
}
int mode() { return mode_info().m; }
std::atomic<int> g_off{0};   // turned off for the process (a oneDNN failure that retrying would repeat)
void turn_off(const char *why) {
    if (!g_off.exchange(1)) std::fprintf(stderr, "[SYCL] prompt attention (oneDNN): %s; the chain's kernels from now on\n", why);
}

struct Key {
    int S, L, H, KVH, hd, cap;
    bool operator<(const Key &o) const {
        return std::tie(S, L, H, KVH, hd, cap) < std::tie(o.S, o.L, o.H, o.KVH, o.hd, o.cap);
    }
};
using Family = std::tuple<int, int, int>;   // H, KV heads, hd
struct Part {
    dg::compiled_partition cp;
    LT query, key, value, out, scale_lt, ninf_lt, kvlen, qlen;
};
constexpr size_t kKeep = 64;   // compiled partitions a device keeps
struct Dev {
    sycl::queue q;             // a copy of the chain's queue: the handle stays valid while held
    dnnl::engine eng;          // made by ensure_engine: in mode 1 on warm_th, off the request's path
    dnnl::stream strm;
    bool has_eng = false;
    sycl::half *q16 = nullptr, *o16 = nullptr, *scale = nullptr;
    float *ninf = nullptr;
    size_t n16 = 0;            // elements in q16 and o16
    float scale_now = NAN;     // the value *scale holds
    std::list<std::pair<Key, std::shared_ptr<Part>>> lru;   // most recent first
    std::map<Key, std::list<std::pair<Key, std::shared_ptr<Part>>>::iterator> at;
    std::set<Family> refused;  // head layouts oneDNN refused: not asked again
    std::atomic<int> ready{0}; // 0 nothing compiled yet, 1 compiling on warm_th, 2 a partition compiled
    std::thread warm_th;
    std::mutex mx;             // the caches and buffers; warm_th takes it only to store its result
    explicit Dev(const sycl::queue &qq) : q(qq) {}
};
std::mutex g_mx;
std::vector<Dev *> *g_devs = new std::vector<Dev *>;   // never destroyed: no static destructor after the runtime's

Dev *dev_for(sycl::queue &q) {   // the queue's record, no oneDNN object yet; may throw bad_alloc
    std::lock_guard<std::mutex> lk(g_mx);
    for (Dev *d : *g_devs) if (d->q == q) return d;
    std::unique_ptr<Dev> d(new Dev(q));
    g_devs->push_back(d.get());
    return d.release();
}
/* oneDNN's engine and stream on the queue's device and context: some milliseconds, so in mode 1
 * warm_th makes them with the first compile (the 21-token prompt's first token was 5 ms later when
 * the request's path made them: campaign DNNL3). The caller holds d->mx, or is warm_th before it
 * publishes ready = 2 under d->mx. May throw. */
void ensure_engine(Dev *d) {
    if (d->has_eng) return;
    d->eng = dg::sycl_interop::make_engine_with_allocator(d->q.get_device(), d->q.get_context(),
                                                         dg::sycl_interop::make_allocator(dn_alloc, dn_free));
    d->strm = dnnl::sycl_interop::make_stream(d->eng, d->q);
    d->has_eng = true;
}

// one partition: Q rows S (dense [S][H][hd]), K/V of L keys in the head-major cache of capacity cap
std::shared_ptr<Part> make_part(Dev *d, const Key &k) {
    const int rep = k.H / k.KVH;
    size_t id = 0;
    const Dims q_sz = {1, k.KVH, rep, k.S, k.hd}, kv_sz = {1, k.KVH, 1, k.L, k.hd}, sc_sz = {1, k.KVH, rep, k.S, k.L};
    const Dims one = {1};
    const Dims q_st = {(int64_t)k.S * k.H * k.hd, (int64_t)rep * k.hd, k.hd, (int64_t)k.H * k.hd, 1};
    const Dims kv_st = {(int64_t)k.KVH * k.cap * k.hd, (int64_t)k.cap * k.hd, (int64_t)k.cap * k.hd, k.hd, 1};
    const auto strided = LT::layout_type::strided;
    auto p = std::make_shared<Part>();
    p->query = LT(id++, DT::f16, q_sz, q_st); p->key = LT(id++, DT::f16, kv_sz, kv_st);
    p->value = LT(id++, DT::f16, kv_sz, kv_st); p->out = LT(id++, DT::f16, q_sz, q_st);
    LT score(id++, DT::f32, sc_sz, strided), scaled(id++, DT::f32, sc_sz, strided);
    p->scale_lt = LT(id++, DT::f16, one, strided); p->ninf_lt = LT(id++, DT::f32, one, strided);
    LT row_idx(id++, DT::s32, sc_sz, strided), col_idx(id++, DT::s32, sc_sz, strided);
    p->kvlen = LT(id++, DT::s32, 0, strided, LT::property_type::host_scalar);
    p->qlen = LT(id++, DT::s32, 0, strided, LT::property_type::host_scalar);
    LT row_add(id++, DT::s32, sc_sz, strided), row_sub(id++, DT::s32, sc_sz, strided);
    LT keep(id++, DT::boolean, sc_sz, strided), masked(id++, DT::f32, sc_sz, strided);
    LT probs(id++, DT::f16, sc_sz, strided);
    dg::op bmm1(id++, dg::op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(dg::op::attr::transpose_b, true);
    bmm1.add_inputs({p->query, p->key}); bmm1.add_outputs({score});
    dg::op mul(id++, dg::op::kind::Multiply, "scale");
    mul.add_inputs({score, p->scale_lt}); mul.add_outputs({scaled});
    dg::op gir(id++, dg::op::kind::GenIndex, "row_index");
    gir.set_attr<int64_t>(dg::op::attr::axis, -2);
    gir.add_inputs({scaled}); gir.add_outputs({row_idx});
    dg::op add(id++, dg::op::kind::Add, "row_plus_kv");
    add.add_inputs({row_idx, p->kvlen}); add.add_outputs({row_add});
    dg::op sub(id++, dg::op::kind::Subtract, "minus_q");
    sub.add_inputs({row_add, p->qlen}); sub.add_outputs({row_sub});
    dg::op gic(id++, dg::op::kind::GenIndex, "col_index");
    gic.set_attr<int64_t>(dg::op::attr::axis, -1);
    gic.add_inputs({scaled}); gic.add_outputs({col_idx});
    dg::op ge(id++, dg::op::kind::GreaterEqual, "keep");
    ge.add_inputs({row_sub, col_idx}); ge.add_outputs({keep});
    dg::op sel(id++, dg::op::kind::Select, "mask");
    sel.add_inputs({keep, scaled, p->ninf_lt}); sel.add_outputs({masked});
    dg::op sm(id++, dg::op::kind::SoftMax, "softmax");
    sm.set_attr<int64_t>(dg::op::attr::axis, -1);
    sm.set_attr<std::string>(dg::op::attr::mode, "inf_as_zero");
    sm.add_inputs({masked}); sm.add_outputs({probs});
    dg::op bmm2(id++, dg::op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, p->value}); bmm2.add_outputs({p->out});
    dg::graph gr(dnnl::engine::kind::gpu);
    for (dg::op *o : {&bmm1, &mul, &gir, &add, &sub, &gic, &ge, &sel, &sm, &bmm2}) gr.add_op(*o);
    gr.finalize();
    std::vector<dg::partition> parts = gr.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) return nullptr;
    p->cp = parts[0].compile({p->query, p->key, p->scale_lt, p->kvlen, p->qlen, p->ninf_lt, p->value}, {p->out}, d->eng);
    return p;
}

// compile k, or nullptr with the reason; never throws
std::shared_ptr<Part> try_make(Dev *d, const Key &k, std::string &why) {
    why = "not one supported partition";
    try { return make_part(d, k); }
    catch (const std::exception &e) { why = e.what(); }   // dnnl::error is one
    return nullptr;
}
Family family(const Key &k) { return Family(k.H, k.KVH, k.hd); }
void refuse(Dev *d, const Key &k, const std::string &why) {   // d->mx held
    d->refused.insert(family(k));
    std::fprintf(stderr, "[SYCL] prompt attention (oneDNN): H%d KV%d hd%d refused at S%d L%d (%s); those layers keep the chain's kernels\n",
                 k.H, k.KVH, k.hd, k.S, k.L, why.c_str());
}
void say_ready(double ms, const char *where) {
    const dnnl_version_t *ver = dnnl_version();
    std::fprintf(stderr, "[SYCL] prompt attention: oneDNN %d.%d.%d fused attention, float16 Q/P/O; first partition compiled %s in %.0f ms\n",
                 ver->major, ver->minor, ver->patch, where, ms);
}
// keep p for k (d->mx held); may throw bad_alloc before changing anything
void keep(Dev *d, const Key &k, const std::shared_ptr<Part> &p) {
    d->lru.emplace_front(k, p);
    try { d->at[k] = d->lru.begin(); }
    catch (...) { d->lru.pop_front(); throw; }
    if (d->lru.size() > kKeep) { d->at.erase(d->lru.back().first); d->lru.pop_back(); }
}
// the engine and the first compile of a process, on warm_th (mode 1); never throws
void warm_up(Dev *d, Key k) {
    const auto t0 = Clock::now();
    std::string why;
    std::shared_ptr<Part> p;
    bool eng = true;
    try { ensure_engine(d); }   // nobody else touches the engine before ready = 2
    catch (const std::exception &e) { eng = false; why = e.what(); }
    if (eng) p = try_make(d, k, why);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(d->mx);
    try {
        if (p) { keep(d, k, p); say_ready(ms, "in the background"); d->ready.store(2); return; }
        if (!eng) turn_off((std::string("no engine (") + why + ")").c_str());
        else refuse(d, k, why);
    } catch (const std::exception &) { turn_off("out of host memory"); }
    d->ready.store(0);
}
// the partition for k on the caller (d->mx held); nullptr when oneDNN refuses it; may throw bad_alloc
std::shared_ptr<Part> part_for(Dev *d, const Key &k) {
    auto it = d->at.find(k);
    if (it != d->at.end()) { d->lru.splice(d->lru.begin(), d->lru, it->second); return it->second->second; }
    const auto t0 = Clock::now();
    std::string why;
    std::shared_ptr<Part> p = try_make(d, k, why);
    if (!p) { refuse(d, k, why); return nullptr; }
    keep(d, k, p);
    if (d->ready.load() != 2) {
        say_ready(std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), "on the caller");
        d->ready.store(2);
    }
    return p;
}

/* a block that queued work may still read: retired to the frame's end (coli_sycl_retire); a queue
 * the chain's frames do not cover (a test's own) is waited on instead, outside any frame */
void give_back(Dev *d, void *p, size_t bytes) {
    if (!p) return;
    if (!coli_sycl_retire(d->q, p, bytes)) { d->q.wait_and_throw(); sycl::free(p, d->q); }
}
// float16 buffers for Q and O of n elements, and the two scalars; 0 when the device refuses
int room(Dev *d, size_t n) {
    if (!d->scale) d->scale = sycl::aligned_alloc_device<sycl::half>(64, 32, d->q);
    if (!d->ninf) {
        d->ninf = sycl::aligned_alloc_device<float>(64, 16, d->q);
        if (d->ninf) d->q.fill(d->ninf, -INFINITY, 1);
    }
    if (!d->scale || !d->ninf) return 0;
    if (n <= d->n16) return 1;
    const size_t want = n > 2 * d->n16 ? n : 2 * d->n16;
    sycl::half *nq = sycl::aligned_alloc_device<sycl::half>(64, want, d->q), *no = sycl::aligned_alloc_device<sycl::half>(64, want, d->q);
    if (!nq || !no) {   // nothing queued uses the new blocks yet
        if (nq) sycl::free(nq, d->q);
        if (no) sycl::free(no, d->q);
        return 0;
    }
    give_back(d, d->q16, d->n16 * sizeof(sycl::half));
    give_back(d, d->o16, d->n16 * sizeof(sycl::half));
    d->q16 = nq; d->o16 = no; d->n16 = want;
    return 1;
}
}  // namespace

int coli_sycl_dnnl_attn_on(void) { return mode() != 0 && !g_off.load(); }

int coli_sycl_dnnl_attn(const coli_sycl_profile &pr, sycl::queue &q, const coli_sycl_attn &a) {
    const auto &v = a.p;
    if (!coli_sycl_dnnl_attn_on() || !pr.has_xmx || a.kv_f16 != 1 || !(v.hd == 64 || v.hd == 128 || v.hd == 256) ||
        a.sel || v.sel_row > 0 || a.win || a.snk || a.ring || !a.q || !a.kc || !a.vc || !a.o || a.q == a.o || a.gate == a.o ||
        v.S < 1 || v.H < 1 || v.KVH < 1 || v.H % v.KVH || v.H / v.KVH != a.G ||
        v.pos_base < 0 || (int64_t)v.pos_base + v.S > v.cap || v.q_off < 0 || v.o_off < 0 ||
        v.k_off < 0 || v.v_off < 0 || v.q_seg < v.hd || (int64_t)v.q_row < (int64_t)(v.H - 1) * v.q_seg + v.hd ||
        (int64_t)v.o_row < (int64_t)v.H * v.hd || !std::isfinite(v.scale) ||
        (v.has_gate && (!a.gate || v.g_off < 0 || v.g_seg < v.hd || (int64_t)v.g_row < (int64_t)(v.H - 1) * v.g_seg + v.hd))) return 0;
    /* K and V must start 16-byte aligned: oneDNN 3.9.1 reads a cache that starts 6 bytes into its
     * buffer wrongly, without an error (campaign DNNL3: max error 0.38-1.76); 16-byte starts were
     * right in every case measured. The engines' caches start at whole layers. */
    if (((uintptr_t)a.kc + (uintptr_t)v.k_off * 2) % 16 || ((uintptr_t)a.vc + (uintptr_t)v.v_off * 2) % 16) return 0;
    if (mode_info().dflt && v.hd != 256) return 0;   // the measured default: the head size measured end to end
    reap_pending(false);
    Dev *d = nullptr;
    try { d = dev_for(q); }
    catch (const std::exception &e) { turn_off(e.what()); return 0; }
    std::lock_guard<std::mutex> lk(d->mx);
    const Key k{v.S, v.pos_base + v.S, v.H, v.KVH, v.hd, v.cap};
    if (d->refused.count(family(k))) return 0;
    if (mode() == 1 && d->ready.load() != 2) {
        /* the process's first compile (about 0.55 s) runs on warm_th; these rows keep the chain's
         * kernels meanwhile. A refused layout lets the next eligible one start it again. */
        if (d->ready.load() == 0) {
            if (d->warm_th.joinable()) d->warm_th.join();   // a refused start's thread, finished
            try { d->warm_th = std::thread(warm_up, d, k); d->ready.store(1); }   // warm_up needs d->mx: it stores after we return
            catch (const std::exception &e) { turn_off(e.what()); }
        }
        return 0;
    }
    try { ensure_engine(d); }   // mode 2's first call (mode 1's warm_th made it already); not a queue submission
    catch (const std::exception &e) { turn_off((std::string("no engine (") + e.what() + ")").c_str()); return 0; }
    const size_t n = (size_t)v.S * v.H * v.hd;
    std::shared_ptr<Part> p;
    try {
        if (!room(d, n)) return 0;
        p = part_for(d, k);
    } catch (const sycl::exception &) {
        throw;   // the queue (a retire's fallback wait): the device's failure, for cc_try
    } catch (const std::exception &e) {
        turn_off(e.what());
        return 0;
    }
    if (!p) return 0;
    if (!(d->scale_now == v.scale)) { coli_sycl_fprof_ev(d->q.fill(d->scale, sycl::half(v.scale), 1)); d->scale_now = v.scale; }
    const int H = v.H, hd = v.hd, has_gate = v.has_gate;
    const size_t n8 = n / 8, q_off = v.q_off, q_row = v.q_row, q_seg = v.q_seg, o_off = v.o_off, o_row = v.o_row;
    const size_t g_off = v.g_off, g_row = v.g_row, g_seg = v.g_seg;
    const float *qp = a.q, *gp = a.gate;
    float *op = a.o;
    sycl::half *q16 = d->q16, *o16 = d->o16;
    coli_sycl_fprof_ev(q.parallel_for(sycl::range<1>(n8), [=](sycl::id<1> i) {
        const size_t e = i[0] * 8, dd = e % hd, sh = e / hd, h = sh % H, s = sh / H;
        const float *src = qp + q_off + s * q_row + h * q_seg + dd;
        sycl::vec<sycl::half, 8> w;
        for (int j = 0; j < 8; j++) w[j] = sycl::half(sycl::fmin(sycl::fmax(src[j], -65504.f), 65504.f));
        *reinterpret_cast<sycl::vec<sycl::half, 8> *>(q16 + e) = w;
    }));
    int32_t kv_len = v.pos_base + v.S, q_len = v.S;
    const sycl::half *kp = static_cast<const sycl::half *>(a.kc) + v.k_off, *vp = static_cast<const sycl::half *>(a.vc) + v.v_off;
    try {
        std::vector<dg::tensor> ins = {dg::tensor(p->query, d->eng, q16), dg::tensor(p->key, d->eng, const_cast<sycl::half *>(kp)),
                                       dg::tensor(p->scale_lt, d->eng, d->scale), dg::tensor::make_scalar_tensor(p->kvlen, &kv_len),
                                       dg::tensor::make_scalar_tensor(p->qlen, &q_len), dg::tensor(p->ninf_lt, d->eng, d->ninf),
                                       dg::tensor(p->value, d->eng, const_cast<sycl::half *>(vp))};
        std::vector<dg::tensor> outs = {dg::tensor(p->out, d->eng, o16)};
        coli_sycl_fprof_ev(dg::sycl_interop::execute(p->cp, d->strm, ins, outs));
    } catch (const sycl::exception &) {
        throw;   // the device's failure, for cc_try
    } catch (const std::exception &e) {
        /* nothing of the output is written yet (only Q's float16 copy): decline, and leave the
         * path off for the process rather than call oneDNN again */
        std::string why = std::string("execution failed (") + e.what() + ")";
        turn_off(why.c_str());
        return 0;
    }
    coli_sycl_fprof_ev(q.parallel_for(sycl::range<1>(n8), [=](sycl::id<1> i) {
        const size_t e = i[0] * 8, dd = e % hd, sh = e / hd, h = sh % H, s = sh / H;
        const sycl::vec<sycl::half, 8> w = *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(o16 + e);
        float *dst = op + o_off + s * o_row + h * hd + dd;
        if (has_gate) {
            const float *gt = gp + g_off + s * g_row + h * g_seg + dd;
            for (int j = 0; j < 8; j++) dst[j] = (float)w[j] * (1.f / (1.f + sycl::native::exp(-gt[j])));
        } else {
            for (int j = 0; j < 8; j++) dst[j] = (float)w[j];
        }
    }));
    return 1;
}

/* the chain's shutdown, after its wait on the queue (not called for a lost context: nothing to wait for) */
void coli_sycl_dnnl_release(sycl::queue &q) {
    std::lock_guard<std::mutex> lk(g_mx);
    for (size_t i = 0; i < g_devs->size(); i++) {
        Dev *d = (*g_devs)[i];
        if (!(d->q == q)) continue;
        if (d->warm_th.joinable()) d->warm_th.join();   // not under d->mx: warm_up takes it to finish
        {
            std::lock_guard<std::mutex> dl(d->mx);
            try { d->q.wait_and_throw(); } catch (const std::exception &) {}
            for (void *p : {(void *)d->q16, (void *)d->o16, (void *)d->scale, (void *)d->ninf})
                if (p) { try { sycl::free(p, d->q); } catch (const std::exception &) {} }
            d->lru.clear(); d->at.clear(); d->refused.clear();
        }
        reap_pending(true);
        delete d;
        g_devs->erase(g_devs->begin() + (long)i);
        return;
    }
}
#endif  // COLI_SYCL_DNNL
