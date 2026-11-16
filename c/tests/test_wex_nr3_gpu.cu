/* The CUDA helper's WNR3 decoder (feature 16) against the CPU reference
 * (wex_neural_rans.h). Not part of `make test-c`: it needs an NVIDIA GPU and the
 * helper (`make wex-cuda-test` builds both and runs it on the synthetic cases).
 *
 * 1. synthetic experts, CZ off/on, chunk pairs 512:256 .. 16384:16384: every
 *    entry point (to host, to device, batch to device, batch to host) equals the
 *    CPU decode, which equals the source / Z(source);
 * 2. crafted records only the WNR3 checks reject (a CZ record carrying -0, a
 *    ceil block at coded scale 0, a WNR3 record under codec 4, a scale stream
 *    only its hash rejects): GPU and CPU both reject, and every entry point
 *    leaves its destination untouched;
 * 3. per records file (RECORDS.bin: the model blob, then entries, decoded slabs
 *    and stored records of real experts): every record decoded by both batch
 *    APIs equals the stored slab, then MUTATIONS mutated records (single bit
 *    flips anywhere in the record, header byte edits) decoded by the CPU and by
 *    the two GPU batch APIs: the same status, the same bytes when both accept,
 *    the destination untouched when both refuse;
 * 4. thread churn: 64 rounds of 4 fresh threads through every entry point, all decoded, and the
 *    device memory the helper holds does not grow with the threads that called it.
 * usage: test_wex_nr3_gpu HELPER [MUTATIONS=120] [RECORDS.bin ...]
 * Every mutation keeps a copy of its record and its CPU decode in RAM: on 103
 * real DeepSeek V4.1 records, 300 mutations needed about 17 GB and 120 ran in 16 GB. */
#ifdef _WIN32
#include <windows.h>
#define helper_open(path) ((void *)LoadLibraryA(path))
#define helper_symbol(h, name) ((void *)GetProcAddress((HMODULE)(h), name))
#else
#include <dlfcn.h>
#define helper_open(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define helper_symbol(h, name) dlsym(h, name)
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include "../wex_format.h"

typedef int (*RegFn)(const unsigned char *, size_t);
typedef int (*HostFn)(const WexEntry *, unsigned char *, size_t);
typedef int (*DevFn)(const WexEntry *, const unsigned char *, int, unsigned char *const *, unsigned char *const *);
typedef int (*BatchFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                       unsigned char *const *, unsigned char *const *, int *);
typedef int (*BatchHostFn)(int, const WexEntry *const *, const unsigned char *const *, int,
                           unsigned char *const *, int *);
typedef unsigned int (*FeatFn)(void);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

static RegFn reg; static HostFn host_fn; static DevFn dev; static BatchFn batch; static BatchHostFn batch_host;

struct Rec { WexEntry e; std::vector<unsigned char> expected, record; };

/* device destinations for up to 16 records of the largest geometry */
struct Arena {
    unsigned char *base = nullptr; size_t per = 0; std::vector<unsigned char *> ds, dw; size_t ms[3], mw[3];
    void init(const std::vector<Rec> &recs) {
        for (int m = 0; m < 3; m++) { ms[m] = mw[m] = 0; }
        for (auto &r : recs) for (int m = 0; m < 3; m++) {
            ms[m] = std::max(ms[m], (size_t)r.e.scale_shapes[m][0] * r.e.scale_shapes[m][1]);
            mw[m] = std::max(mw[m], (size_t)r.e.weight_shapes[m][0] * r.e.weight_shapes[m][1]);
        }
        per = 0; for (int m = 0; m < 3; m++) per += ms[m] + mw[m];
        if (base) cudaFree(base);
        CK(cudaMalloc(&base, per * 16));
        ds.assign(48, nullptr); dw.assign(48, nullptr);
        for (int b = 0; b < 16; b++) {
            unsigned char *p = base + per * b;
            for (int m = 0; m < 3; m++) { ds[3 * b + m] = p; p += ms[m]; }
            for (int m = 0; m < 3; m++) { dw[3 * b + m] = p; p += mw[m]; }
        }
    }
    /* slot -> canonical slab (scales w1..w3, weights w1..w3) */
    std::vector<unsigned char> fetch(int slot, const WexEntry &e) {
        std::vector<unsigned char> out((size_t)e.scale_bytes + e.weight_bytes);
        for (int m = 0; m < 3; m++) {
            CK(cudaMemcpy(out.data() + e.scale_offsets[m], ds[3 * slot + m], (size_t)e.scale_shapes[m][0] * e.scale_shapes[m][1], cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(out.data() + e.weight_offsets[m], dw[3 * slot + m], (size_t)e.weight_shapes[m][0] * e.weight_shapes[m][1], cudaMemcpyDeviceToHost));
        }
        return out;
    }
    bool untouched(int slot) {
        std::vector<unsigned char> got(per);
        CK(cudaMemcpy(got.data(), base + per * slot, per, cudaMemcpyDeviceToHost));
        for (unsigned char c : got) if (c != 0xA5) return false;
        return true;
    }
};

/* CPU reference: status and slab */
static int cpu_decode(const Rec &r, const WexNr2Model &model, std::vector<unsigned char> &out) {
    uint64_t raw, aligned;
    CHECK(!wex_entry_raw_sizes(&r.e, &raw, &aligned));
    std::vector<unsigned char> slab((size_t)aligned, 0x5a);
    size_t at = wex_record_read_offset(&r.e);
    if (!at || r.record.size() != r.e.total_bytes) return -1;
    memcpy(slab.data() + at, r.record.data(), r.record.size());
    int status = wex_decode_nr2_record(&r.e, slab.data(), slab.size(), &model);
    out.assign(slab.begin(), slab.begin() + (size_t)raw);
    return status;
}

/* GPU batch decode of up to 16 records into arena slots 0..n-1 */
static void gpu_batch(Arena &a, const std::vector<const Rec *> &rs, std::vector<int> &st,
                      std::vector<unsigned char *> &pinned) {
    int n = (int)rs.size();
    std::vector<const WexEntry *> es(n); std::vector<const unsigned char *> ps(n);
    st.assign(n, 7);
    for (int i = 0; i < n; i++) {
        if (pinned[i]) cudaFreeHost(pinned[i]);
        CK(cudaMallocHost((void **)&pinned[i], rs[i]->record.size()));
        memcpy(pinned[i], rs[i]->record.data(), rs[i]->record.size());
        es[i] = &rs[i]->e; ps[i] = pinned[i];
    }
    CK(cudaMemset(a.base, 0xA5, a.per * 16));
    batch(n, es.data(), ps.data(), 0, a.ds.data(), a.dw.data(), st.data());
}

/* a batch-to-host destination the helper did not write (gpu_batch_host fills it with 0xA5) */
static bool host_untouched(const std::vector<unsigned char> &v) {
    for (unsigned char c : v) if (c != 0xA5) return false;
    return true;
}

/* GPU batch decode of n records into host buffers (feature 32); out[i] = raw slab */
static void gpu_batch_host(const std::vector<const Rec *> &rs, const std::vector<unsigned char *> &pinned,
                           std::vector<int> &st, std::vector<std::vector<unsigned char>> &out) {
    int n = (int)rs.size();
    std::vector<const WexEntry *> es(n); std::vector<const unsigned char *> ps(n); std::vector<unsigned char *> ds(n);
    st.assign(n, 7); out.resize(n);
    for (int i = 0; i < n; i++) {
        out[i].assign((size_t)rs[i]->e.scale_bytes + rs[i]->e.weight_bytes, 0xA5);
        es[i] = &rs[i]->e; ps[i] = pinned[i]; ds[i] = out[i].data();
    }
    int bad = batch_host(n, es.data(), ps.data(), 0, ds.data(), st.data());
    CHECK(bad == (int)std::count_if(st.begin(), st.end(), [](int v) { return v != 0; }));
}

/* ---- synthetic experts (same generator as tests/test_wex_format.c) ---- */
static uint32_t rnd(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static WexEntry synth_entry(uint32_t rows, uint32_t cols_bytes) {
    WexEntry e; memset(&e, 0, sizeof(e));
    uint32_t sc = rows * (cols_bytes / 16u), wb = rows * cols_bytes;
    e.scale_bytes = 3u * sc; e.weight_bytes = 3u * wb;
    for (int m = 0; m < 3; m++) {
        e.scale_shapes[m][0] = e.weight_shapes[m][0] = (uint16_t)rows;
        e.scale_shapes[m][1] = (uint16_t)(cols_bytes / 16u); e.weight_shapes[m][1] = (uint16_t)cols_bytes;
        e.scale_offsets[m] = m * sc; e.weight_offsets[m] = e.scale_bytes + m * wb;
    }
    return e;
}
static void synth_fill(const WexEntry &e, unsigned char *raw, uint32_t seed) {
    static const unsigned char general[16] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7};
    static const unsigned char even[8] = {0, 0, 2, 2, 4, 5, 6, 7};
    uint32_t s = seed;
    for (int m = 0; m < 3; m++) {
        uint32_t blocks = (uint32_t)e.scale_shapes[m][0] * e.scale_shapes[m][1];
        unsigned char *sc = raw + e.scale_offsets[m], *w = raw + e.weight_offsets[m];
        for (uint32_t b = 0; b < blocks; b++) {
            uint32_t kind = rnd(&s) % 8u;
            unsigned char *q = w + (size_t)b * 16u;
            sc[b] = (unsigned char)(118u + rnd(&s) % 5u);
            if (kind == 4u && b % 37u == 0) sc[b] = (unsigned char)(b & 1u ? 255u : 254u);
            if (kind == 6u && b % 41u == 0) sc[b] = 0;   /* scale 0 on a non-even block is legal */
            for (uint32_t k = 0; k < 32u; k++) {
                uint32_t r = rnd(&s);
                uint32_t a = kind < 4u ? general[r & 15u] : kind < 6u ? even[r & 7u] : kind == 6u ? (r & 7u) % 5u : 0u;
                uint32_t x = a | ((r >> 8) & 8u);
                if (k < 2u && kind < 6u) x = (x & 8u) | 7u;
                if (k & 1u) q[k / 2u] = (unsigned char)(q[k / 2u] | x << 4); else q[k / 2u] = (unsigned char)x;
            }
        }
    }
}
static void views(const WexEntry &e, unsigned char *raw, unsigned char *s[3], unsigned char *w[3]) {
    for (int m = 0; m < 3; m++) { s[m] = raw + e.scale_offsets[m]; w[m] = raw + e.weight_offsets[m]; }
}

static long g_compared = 0, g_rejected_both = 0;

static void compare_all_paths(Arena &a, const Rec &r, const WexNr2Model &model, int expect_status) {
    std::vector<unsigned char> cpu;
    int cs = cpu_decode(r, model, cpu);
    CHECK(cs == expect_status);
    if (!cs) CHECK(cpu == r.expected);
    /* batch */
    std::vector<int> st; std::vector<unsigned char *> pin(1, nullptr);
    gpu_batch(a, {&r}, st, pin);
    CHECK(st[0] == cs);
    if (!cs) CHECK(a.fetch(0, r.e) == r.expected); else CHECK(a.untouched(0));
    /* batch to host: dst untouched on failure */
    { std::vector<int> hs; std::vector<std::vector<unsigned char>> ho;
      gpu_batch_host({&r}, pin, hs, ho);
      CHECK(hs[0] == cs);
      if (!cs) CHECK(ho[0] == r.expected); else CHECK(host_untouched(ho[0])); }
    /* single to device */
    unsigned char *s3[3] = {a.ds[0], a.ds[1], a.ds[2]}, *w3[3] = {a.dw[0], a.dw[1], a.dw[2]};
    CK(cudaMemset(a.base, 0xA5, a.per));
    CHECK(dev(&r.e, pin[0], 0, s3, w3) == cs);
    if (!cs) CHECK(a.fetch(0, r.e) == r.expected); else CHECK(a.untouched(0));
    /* to host: the record at the slab end, slab untouched on failure */
    uint64_t raw, aligned;
    CHECK(!wex_entry_raw_sizes(&r.e, &raw, &aligned));
    std::vector<unsigned char> slab((size_t)aligned, 0x5a);
    size_t at = wex_record_read_offset(&r.e);
    memcpy(slab.data() + at, r.record.data(), r.record.size());
    CHECK(host_fn(&r.e, slab.data(), slab.size()) == cs);
    if (!cs) CHECK(!memcmp(slab.data(), r.expected.data(), (size_t)raw)); else CHECK(slab[0] == 0x5a);
    cudaFreeHost(pin[0]);
    g_compared++;
    if (cs) g_rejected_both++;
}

static void synthetic(Arena &a) {
    static const uint32_t wchunk[4] = {512u, 2048u, 4096u, 16384u}, schunk[4] = {256u, 512u, 2048u, 16384u};
    WexEntry e = synth_entry(256u, 1024u);   /* 3 x 256 x 2048 FP4, 16384 blocks per matrix */
    WexNr2Layout layout; wex_nr2_layout(&e, &layout);
    size_t raw = (size_t)e.scale_bytes + e.weight_bytes;
    uint64_t r64, aligned; CHECK(!wex_entry_raw_sizes(&e, &r64, &aligned));
    std::vector<unsigned char> src(raw), coded(raw), out(4 * raw + 65536);
    unsigned char *s[3], *w[3];
    for (int cz = 0; cz < 2; cz++) {
        std::vector<uint64_t> wc(3 * 567 * 16, 0), sc(3 * 256, 0);
        for (uint32_t seed = 1; seed <= 3u; seed++) {
            synth_fill(e, src.data(), seed * 6007u);
            coded = src;
            if (cz) wex_nr3_canonical_zero(coded.data() + e.scale_bytes, e.weight_bytes);
            views(e, coded.data(), s, w);
            wex_nr3_ceil_forward(&layout, s, w, NULL);
            wex_nr3_count(&layout, (const unsigned char *const *)s, (const unsigned char *const *)w, wc.data(), sc.data());
        }
        std::vector<unsigned char> blob(WEX_NR2_MODEL_BYTES);
        wex_nr2_model_build(wc.data(), sc.data(), blob.data());
        WexNr2Model model;
        CHECK(!wex_nr2_model_open(blob.data(), blob.size(), &model) && reg(blob.data(), blob.size()) == 0);
        synth_fill(e, src.data(), 4242u);
        coded = src;
        if (cz) wex_nr3_canonical_zero(coded.data() + e.scale_bytes, e.weight_bytes);
        views(e, coded.data(), s, w);
        uint32_t flags = wex_nr3_ceil_forward(&layout, s, w, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
        std::vector<unsigned char> expected(src);
        if (cz) wex_nr3_canonical_zero(expected.data() + e.scale_bytes, e.weight_bytes);
        std::vector<Rec> recs;
        for (int c = 0; c < 4; c++) {
            size_t written = 0;
            std::fill(out.begin(), out.end(), 0);
            CHECK(!wex_nr3_encode(&model, &layout, (const unsigned char *const *)s, (const unsigned char *const *)w,
                                  wchunk[c], schunk[c], flags, 0x00270078u, out.data(), out.size(), &written));
            Rec r; r.e = e; r.e.flags = WEX_ENTRY_NEURAL_RANS3;
            r.e.total_bytes = (uint32_t)((written + 4095u) & ~(size_t)4095u);
            CHECK(r.e.total_bytes < aligned);
            r.record.assign(out.begin(), out.begin() + r.e.total_bytes);
            r.expected = expected;
            recs.push_back(r);
        }
        a.init(recs);
        for (auto &r : recs) compare_all_paths(a, r, model, 0);
        /* crafted: WNR3 record under codec 4 */
        { Rec r = recs[1]; r.e.flags = WEX_ENTRY_NEURAL_RANS2; compare_all_paths(a, r, model, -1); }
        {   /* crafted: scale stream 0 re-encoded with one scale moved where srel stays clamped, under
             * its original x0 (weights decode unchanged): only the scale-stream hash rejects it */
            const Rec &base = recs[1];
            WexNr2Record v;
            CHECK(!wex_nr3_parse(base.record.data(), base.record.size(), &layout, model.id, &v));
            uint32_t count = std::min(layout.scale_count[0], v.scale_chunk), len0 = wex_nr_u16(v.lengths);
            uint32_t ident[4] = {model.id, v.owner, 0u, v.word24}, x0 = wex_nr3_stream_x0(ident, s[0], count), k;
            std::vector<unsigned char> data(s[0], s[0] + count), tmp(2u * WEX_NR2_MAX_CHUNK + 8u);
            for (k = 0; k < count; k++) {
                int d = (int)data[k] - (int)v.median[0];
                if ((d >= 3 && data[k] < 255u) || d <= -4) break;   /* +1 keeps srel clamped */
            }
            CHECK(k < count);
            data[k]++;
            unsigned char *p = tmp.data() + tmp.size();
            uint32_t state = x0;
            for (uint32_t j = count; j-- > 0;) CHECK(!wex_nr2_put(&state, &p, tmp.data(), wex_nr2_scale_cdf(&model, 0), data[j]));
            CHECK(!wex_nr2_flush(state, &p, tmp.data()));
            uint32_t nl = (uint32_t)(tmp.data() + tmp.size() - p);
            uint32_t used = v.metadata_bytes + v.scale_payload_bytes + v.weight_payload_bytes;
            Rec r = base;
            r.e.total_bytes = (uint32_t)(((size_t)used - len0 + nl + 4095u) & ~(size_t)4095u);
            r.record.assign(r.e.total_bytes, 0);
            memcpy(r.record.data(), base.record.data(), v.metadata_bytes);
            r.record[WEX_NR2_HEADER_BYTES] = (unsigned char)nl; r.record[WEX_NR2_HEADER_BYTES + 1u] = (unsigned char)(nl >> 8);
            uint32_t sp = v.scale_payload_bytes - len0 + nl;
            memcpy(r.record.data() + 8, &sp, 4);
            memcpy(r.record.data() + v.metadata_bytes, p, nl);
            memcpy(r.record.data() + v.metadata_bytes + nl, base.record.data() + v.metadata_bytes + len0,
                   used - v.metadata_bytes - len0);
            CHECK(r.e.total_bytes < aligned && !wex_nr3_parse(r.record.data(), r.record.size(), &layout, model.id, &v));
            compare_all_paths(a, r, model, -1);
        }
        /* crafted: CZ flag on data carrying -0 (non-CZ coded data) / ceil block at coded scale 0 */
        std::vector<unsigned char> c2(src);
        views(e, c2.data(), s, w);
        uint32_t f2 = wex_nr3_ceil_forward(&layout, s, w, NULL);
        for (int k = 0; k < 2; k++) {
            unsigned char anchor[3];
            uint32_t cf = f2, b = 0;
            if (k == 0) cf |= WEX_NR3_FLAG_CZ;
            else {
                for (b = 0; b < layout.scale_count[2] && wex_nr3_block_max(w[2] + (size_t)b * 16u, NULL) != 5u; b++) {}
                CHECK(b < layout.scale_count[2]);
                s[2][b] = 0;
            }
            for (int m = 0; m < 3; m++) anchor[m] = wex_nr3_anchor(s[m], layout.scale_count[m]);
            size_t written = 0;
            std::fill(out.begin(), out.end(), 0);
            CHECK(!wex_nrx_encode(&model, &layout, (const unsigned char *const *)s, (const unsigned char *const *)w,
                                  2048u, 512u, anchor, WEX_NR3_MAGIC, cf, 7u, out.data(), out.size(), &written));
            Rec r; r.e = e; r.e.flags = WEX_ENTRY_NEURAL_RANS3;
            r.e.total_bytes = (uint32_t)((written + 4095u) & ~(size_t)4095u);
            if (r.e.total_bytes >= aligned) continue;   /* the misfit model may not compress -0 data */
            r.record.assign(out.begin(), out.begin() + r.e.total_bytes);
            r.expected = src;
            compare_all_paths(a, r, model, -1);
        }
        printf("synthetic cz=%d: 4 chunk pairs x 4 entry points equal to the CPU decode, crafted records rejected, "
               "destinations untouched\n", cz);
    }
}

/* 4. thread churn, as the engines' reader teams do (an OpenMP team whose threads a runtime retires
 * and respawns, a reader pool per load): ROUNDS rounds of THREADS fresh threads, each decoding one
 * record through every entry point and then exiting. Every decode must succeed and equal the CPU
 * decode, and the device memory the helper holds afterwards must not grow with the number of
 * threads that ever called it: workspaces belong to the calls in flight, not to the threads. */
static void thread_churn() {
    WexEntry e = synth_entry(256u, 1024u);
    WexNr2Layout layout; wex_nr2_layout(&e, &layout);
    size_t raw = (size_t)e.scale_bytes + e.weight_bytes;
    uint64_t r64, aligned; CHECK(!wex_entry_raw_sizes(&e, &r64, &aligned));
    std::vector<unsigned char> src(raw), out(4 * raw + 65536);
    unsigned char *s[3], *w[3];
    std::vector<uint64_t> wc(3 * 567 * 16, 0), sc(3 * 256, 0);
    synth_fill(e, src.data(), 777u);
    views(e, src.data(), s, w);
    uint32_t flags = wex_nr3_ceil_forward(&layout, s, w, NULL);
    wex_nr3_count(&layout, (const unsigned char *const *)s, (const unsigned char *const *)w, wc.data(), sc.data());
    std::vector<unsigned char> blob(WEX_NR2_MODEL_BYTES);
    wex_nr2_model_build(wc.data(), sc.data(), blob.data());
    WexNr2Model model;
    CHECK(!wex_nr2_model_open(blob.data(), blob.size(), &model) && reg(blob.data(), blob.size()) == 0);
    synth_fill(e, src.data(), 777u);
    std::vector<unsigned char> coded(src);
    views(e, coded.data(), s, w);
    wex_nr3_ceil_forward(&layout, s, w, NULL);
    size_t written = 0;
    CHECK(!wex_nr3_encode(&model, &layout, (const unsigned char *const *)s, (const unsigned char *const *)w,
                          2048u, 512u, flags, 0x00010002u, out.data(), out.size(), &written));
    Rec r; r.e = e; r.e.flags = WEX_ENTRY_NEURAL_RANS3;
    r.e.total_bytes = (uint32_t)((written + 4095u) & ~(size_t)4095u);
    CHECK(r.e.total_bytes < aligned);
    r.record.assign(out.begin(), out.begin() + r.e.total_bytes);
    r.expected = src;
    std::vector<unsigned char> cpu;
    CHECK(cpu_decode(r, model, cpu) == 0 && cpu == r.expected);
    const size_t at = wex_record_read_offset(&r.e);
    unsigned char *dev_out = nullptr;
    const int THREADS = 4, ROUNDS = 64;
    CK(cudaMalloc(&dev_out, raw * THREADS));
    std::vector<int> bad(THREADS);
    auto round = [&] {
        std::vector<std::thread> team;
        for (int t = 0; t < THREADS; t++) team.emplace_back([&, t] {
            int fails = 0;
            std::vector<unsigned char> slab((size_t)aligned, 0x5a);
            memcpy(slab.data() + at, r.record.data(), r.record.size());
            if (host_fn(&r.e, slab.data(), slab.size()) || memcmp(slab.data(), r.expected.data(), raw)) fails++;
            unsigned char *ds[3], *dw[3];
            views(r.e, dev_out + raw * t, ds, dw);
            if (dev(&r.e, r.record.data(), 0, ds, dw)) fails++;
            const WexEntry *es[1] = {&r.e};
            const unsigned char *ps[1] = {r.record.data()};
            std::vector<unsigned char> host(raw, 0xA5);
            unsigned char *hd[1] = {host.data()};
            int st[1] = {7};
            if (batch_host(1, es, ps, 0, hd, st) || st[0] || host != r.expected) fails++;
            if (batch(1, es, ps, 0, ds, dw, st) || st[0]) fails++;
            bad[(size_t)t] = fails;
        });
        for (auto &t : team) t.join();
        for (int t = 0; t < THREADS; t++) CHECK(bad[(size_t)t] == 0);
        std::vector<unsigned char> back(raw);
        for (int t = 0; t < THREADS; t++) {
            CK(cudaMemcpy(back.data(), dev_out + raw * t, raw, cudaMemcpyDeviceToHost));
            CHECK(back == r.expected);
        }
    };
    round();   /* the first round's workspaces and the model's device tables */
    size_t free_before, free_after, total;
    CK(cudaMemGetInfo(&free_before, &total));
    for (int k = 1; k < ROUNDS; k++) round();
    CK(cudaMemGetInfo(&free_after, &total));
    CK(cudaFree(dev_out));
    const double grew = free_before > free_after ? (double)(free_before - free_after) / 1048576.0 : 0.0;
    /* one round's workspaces (THREADS x a host, a device and a batch workspace of this record) are
     * about 26 MiB; kept per thread, the other ROUNDS-1 rounds' threads held 1.6 GiB more */
    printf("thread churn: %d rounds x %d fresh threads x 4 entry points decoded, device memory held grew %.0f MiB\n",
           ROUNDS, THREADS, grew);
    CHECK(grew <= 64.0);
}

static void bin_test(Arena &a, const char *path, int mutations) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    uint32_t mb, n;
    CHECK(fread(&mb, 4, 1, f) == 1);
    std::vector<unsigned char> blob(mb);
    CHECK(fread(blob.data(), 1, mb, f) == mb && fread(&n, 4, 1, f) == 1);
    WexNr2Model model;
    CHECK(!wex_nr2_model_open(blob.data(), mb, &model) && reg(blob.data(), mb) == 0);
    std::vector<Rec> recs(n);
    for (auto &r : recs) {
        uint32_t rb;
        CHECK(fread(&r.e, sizeof(WexEntry), 1, f) == 1 && fread(&rb, 4, 1, f) == 1);
        r.expected.resize(rb); CHECK(fread(r.expected.data(), 1, rb, f) == rb);
        r.record.resize(r.e.total_bytes); CHECK(fread(r.record.data(), 1, r.e.total_bytes, f) == r.e.total_bytes);
    }
    fclose(f);
    a.init(recs);
    std::vector<unsigned char *> pin(16, nullptr);
    std::vector<int> st;
    for (size_t i = 0; i < recs.size(); i += 8) {   /* clean records */
        std::vector<const Rec *> rs;
        for (size_t k = i; k < std::min(recs.size(), i + 8); k++) rs.push_back(&recs[k]);
        gpu_batch(a, rs, st, pin);
        for (size_t k = 0; k < rs.size(); k++) { CHECK(st[k] == 0); CHECK(a.fetch((int)k, rs[k]->e) == rs[k]->expected); }
        std::vector<std::vector<unsigned char>> ho;
        gpu_batch_host(rs, pin, st, ho);
        for (size_t k = 0; k < rs.size(); k++) { CHECK(st[k] == 0); CHECK(ho[k] == rs[k]->expected); }
    }
    /* mutations: CPU (8 threads) and GPU batch of 8 */
    std::mt19937 rng(20261110u);
    std::vector<Rec> muts((size_t)mutations);
    for (int i = 0; i < mutations; i++) {
        Rec &m = muts[(size_t)i];
        m = recs[rng() % recs.size()];
        uint32_t kind = rng() % 8u;
        size_t at;
        if (kind < 6u) {   /* single bit anywhere in the used bytes */
            WexNr2Layout layout; WexNr2Record view;
            wex_nr2_layout(&m.e, &layout);
            CHECK(!wex_nr3_parse(m.record.data(), m.record.size(), &layout, 0, &view));
            size_t used = view.metadata_bytes + view.scale_payload_bytes + view.weight_payload_bytes;
            at = rng() % used;
            m.record[at] ^= (unsigned char)(1u << (rng() % 8u));
        } else if (kind == 6u) { at = 24u + rng() % 8u; m.record[at] ^= (unsigned char)(1u << (rng() % 8u)); }   /* anchors, flags, owner */
        else { at = 32u + rng() % 64u; m.record[at] ^= 1u; }   /* lengths table */
    }
    std::vector<int> cpu_status((size_t)mutations);
    std::vector<std::vector<unsigned char>> cpu_out((size_t)mutations);
    {
        std::vector<std::thread> pool;
        for (int t = 0; t < 8; t++) pool.emplace_back([&, t] {
            for (int i = t; i < mutations; i += 8) cpu_status[(size_t)i] = cpu_decode(muts[(size_t)i], model, cpu_out[(size_t)i]);
        });
        for (auto &t : pool) t.join();
    }
    long same = 0, accepted = 0;
    for (int i = 0; i < mutations; i += 8) {
        std::vector<const Rec *> rs;
        for (int k = i; k < std::min(mutations, i + 8); k++) rs.push_back(&muts[(size_t)k]);
        gpu_batch(a, rs, st, pin);
        for (size_t k = 0; k < rs.size(); k++) {
            int c = cpu_status[(size_t)i + k];
            if (st[k] != c) { fprintf(stderr, "mutation %d: cpu %d gpu %d\n", i + (int)k, c, st[k]); exit(1); }
            if (!c) { CHECK(a.fetch((int)k, rs[k]->e) == cpu_out[(size_t)i + k]); accepted++; }
            else CHECK(a.untouched((int)k));
            same++;
        }
        std::vector<std::vector<unsigned char>> ho;
        gpu_batch_host(rs, pin, st, ho);
        for (size_t k = 0; k < rs.size(); k++) {
            int c = cpu_status[(size_t)i + k];
            if (st[k] != c) { fprintf(stderr, "mutation %d (to host): cpu %d gpu %d\n", i + (int)k, c, st[k]); exit(1); }
            if (!c) CHECK(ho[k] == cpu_out[(size_t)i + k]); else CHECK(host_untouched(ho[k]));
        }
    }
    for (auto p : pin) if (p) cudaFreeHost(p);
    printf("{\"bin\":\"%s\",\"records\":%u,\"clean_ok\":%u,\"mutations\":%d,\"cpu_gpu_same_status\":%ld,\"accepted\":%ld}\n",
           path, n, n, mutations, same, accepted);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) return 2;
    void *h = helper_open(argv[1]);
    CHECK(h);
    FeatFn abi = (FeatFn)helper_symbol(h, "wex_cuda_helper_abi");
    FeatFn feat = (FeatFn)helper_symbol(h, "wex_cuda_helper_features");
    reg = (RegFn)helper_symbol(h, "wex_cuda_register_model");
    host_fn = (HostFn)helper_symbol(h, "wex_cuda_decode_to_host");
    dev = (DevFn)helper_symbol(h, "wex_cuda_decode_to_device");
    batch = (BatchFn)helper_symbol(h, "wex_cuda_decode_batch_to_device");
    batch_host = (BatchHostFn)helper_symbol(h, "wex_cuda_decode_batch_to_host");
    CHECK(abi && abi() == 1u && feat && feat() == 63u && reg && host_fn && dev && batch && batch_host);
    printf("helper %s: ABI 1, features 63\n", argv[1]);
    CK(cudaSetDevice(0));
    Arena a;
    synthetic(a);
    thread_churn();
    int mutations = argc > 2 ? atoi(argv[2]) : 120;
    for (int i = 3; i < argc; i++) bin_test(a, argv[i], mutations);
    printf("compared %ld (rejected by both %ld)\nALL OK\n", g_compared, g_rejected_both);
    return 0;
}
