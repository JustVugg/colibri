/* kolibri1_capture.h -- TEST-ONLY capture hook for kolibri1.c. Not a
 * shared header: it is included by kolibri1.c alone, after k1_die / k1_malloc, and everything in
 * it is gated by environment variables (a NULL/0 state costs one predictable branch per layer).
 *
 *   The residual stream is native FP64 (residual policy kolibri1-residual-fp64-c1), so the capture is L<ll>.x_out.f64 [T][H] (8 bytes per element, never
 *                      narrowed to fp32); ENGINE.json has schema kolibri1-engine-capture-2 and records residual_dtype / residual_policy.
 *   K1_CAPTURE=<dir>   per layer ll: L<ll>.x_out.f64 [T][H] (was .f32), L<ll>.z.f32 [T][E], L<ll>.sel.i16 [T][K],
 *                      L<ll>.w.f32 [T][K], appended in ascending position order whatever the prefill
 *                      chunking.  Integrity contract: the files are written in chunks of 128 positions;
 *                      after a chunk's bytes are flushed and fsync'ed one line
 *                      {file,pos0,n,offset,bytes,sha256} is appended to chunks.jsonl (and fsync'ed).
 *                      At a clean end the engine writes ENGINE.json (atomically, fsync + rename) with
 *                      what it knows (inputs, flags, counters, errors = 0).  The engine NEVER writes
 *                      DONE.json: that needs the comparator's verdict on the chunks and the streamed
 *                      logits (an external comparator).  DONE.json and ENGINE.json are deleted
 *                      at the start of every run in the directory; chunks.jsonl is truncated.
 *   K1_LOGITS_FIFO=<path>  frames  "K1LG" u32 pos0 u32 n (<= 128) u32 V, then f32[n][V], produced by the
 *                      engine's own final norm + head on slices that never cross a multiple of 128.
 *                      Blocking writes (a FIFO or a plain file) give back-pressure.
 *   K1_TEST_KILL_AT=<pos>  failure injection: SIGKILL the process in the middle of the capture of the
 *                      block that contains position <pos> (after half of the layers were written).
 *   K1_REPLAY_IDS=<schedule>  (test only; needs K1_CAPTURE and K1_FEED) IDs-only routing replay: at every layer and position
 *                      the engine EXECUTES the expert IDs of the schedule (the other implementation's selection, in its rank order)
 *                      and recomputes the weights as sigmoid(its own z[forced id]): no bias, no renormalisation, its own arithmetic
 *                      and accumulation path.  Its OWN natural selection (what it would have chosen) is recorded separately in
 *                      nat_sel.i16 [L][T][K] int16 (sha256 in ENGINE.json); the capture's sel/w files hold what was executed.
 *                      The schedule is a binary file (see k1_replay_open); a malformed, duplicate, out-of-range, truncated,
 *                      checksum-failing or mismatched (input, layers, positions, widths) schedule is a fatal error before any layer runs.
 *   K1_TEST_MUT_NATSEL=<layer>:<pos>  (control only; needs K1_REPLAY_IDS) corrupts the RECORDED natural selection of that row (its
 *                      6th pick is replaced by an expert outside the natural top-7) so that the comparator's natural-selection checks
 *                      must fire although the executed IDs, and so every continuous tensor, are unchanged.
 * With K1_REPLAY_IDS unset nothing here changes the computation: one predictable branch per layer.
 * No tensor beyond the ones listed above is captured (plus the replay sidecar nat_sel.i16: 12 B per token-layer, only under replay). */
#ifndef KOLIBRI1_CAPTURE_H
#define KOLIBRI1_CAPTURE_H

#include <fcntl.h>
#include <signal.h>

/* ---------------------------------------------------------------- sha256 ---- */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t nb; } K1Sha;

static const uint32_t K1_SHA_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
#define K1_ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void k1_sha_block(K1Sha *s, const uint8_t *p) {
    uint32_t w[64], a, b, c, d, e, f, g, h;
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = K1_ROR(w[i - 15], 7) ^ K1_ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = K1_ROR(w[i - 2], 17) ^ K1_ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = s->h[0]; b = s->h[1]; c = s->h[2]; d = s->h[3]; e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = K1_ROR(e, 6) ^ K1_ROR(e, 11) ^ K1_ROR(e, 25), ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K1_SHA_K[i] + w[i];
        uint32_t S0 = K1_ROR(a, 2) ^ K1_ROR(a, 13) ^ K1_ROR(a, 22), mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}
static void k1_sha_init(K1Sha *s) {
    static const uint32_t i0[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    memcpy(s->h, i0, sizeof(i0)); s->len = 0; s->nb = 0;
}
static void k1_sha_update(K1Sha *s, const void *data, size_t n) {
    const uint8_t *p = data;
    s->len += n;
    while (n) {
        size_t t = 64 - s->nb; if (t > n) t = n;
        memcpy(s->buf + s->nb, p, t); s->nb += t; p += t; n -= t;
        if (s->nb == 64) { k1_sha_block(s, s->buf); s->nb = 0; }
    }
}
static void k1_sha_final_hex(K1Sha *s, char hex[65]) {
    uint64_t bits = s->len * 8;
    uint8_t pad = 0x80;
    k1_sha_update(s, &pad, 1);
    pad = 0;
    while (s->nb != 56) k1_sha_update(s, &pad, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
    k1_sha_update(s, lb, 8);
    for (int i = 0; i < 8; i++) snprintf(hex + 8 * i, 9, "%08x", s->h[i]);
}
static void k1_sha_hex(const void *data, size_t n, char hex[65]) {
    K1Sha s; k1_sha_init(&s); k1_sha_update(&s, data, n); k1_sha_final_hex(&s, hex);
}
/* 0 on success; hex = "" when the file cannot be read */
static int k1_sha_file(const char *path, char hex[65]) {
    hex[0] = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    K1Sha s; k1_sha_init(&s);
    uint8_t *buf = k1_malloc(1 << 20, "sha buffer");
    size_t r;
    while ((r = fread(buf, 1, 1 << 20, f)) > 0) k1_sha_update(&s, buf, r);
    free(buf); fclose(f);
    k1_sha_final_hex(&s, hex);
    return 0;
}

/* --------------------------------------------------------------- capture ---- */
#define K1_CAP_CHUNK 128
enum { K1_CF_X = 0, K1_CF_Z = 1, K1_CF_SEL = 2, K1_CF_W = 3, K1_CF_N = 4 };

typedef struct {
    FILE *f;
    char name[48];
    size_t rb;                 /* row bytes */
    uint8_t *buf;              /* the chunk being filled */
    int64_t next;              /* next position expected */
    int64_t offset;            /* bytes already written */
    long nchunks;
} K1CapFile;

typedef struct {
    int on;
    char dir[3000];
    int T, L, H, E, K;
    K1CapFile *files;          /* [L][K1_CF_N] */
    FILE *chunks;
    long nlines;
    int kill_at;
    /* logits stream */
    int fifo_fd;
    long frames;
    int64_t fifo_next;
    int dirty_err;
} K1Cap;

static K1Cap k1cap;            /* zero = off */

static void k1_cap_fsync_file(FILE *f) {
    if (fflush(f) != 0) k1_die("capture: flush failed (%s)", strerror(errno));
    if (fsync(fileno(f)) != 0) k1_die("capture: fsync failed (%s)", strerror(errno));
}

static void k1_cap_unlink(const char *name) {
    char p[3200]; snprintf(p, sizeof(p), "%s/%s", k1cap.dir, name);
    unlink(p);
}

/* The completion markers of a previous run in K1_CAPTURE are removed FIRST THING in main(), before the feed,
 * the checkpoint or any other fallible step is touched, so that a rerun which fails early can never leave an older success verifiable. */
static void k1_cap_invalidate_early(void) {
    const char *dir = getenv("K1_CAPTURE");
    if (!dir || !*dir) return;
    static const char *names[] = { "DONE.json", "ENGINE.json", "DONE.json.tmp", "ENGINE.json.tmp", "nat_sel.i16", "nat_sel.i16.tmp" };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        char p[3300];
        if (snprintf(p, sizeof(p), "%s/%s", dir, names[i]) >= (int)sizeof(p)) k1_die("K1_CAPTURE path is too long");
        if (unlink(p) != 0 && errno != ENOENT) k1_die("cannot invalidate %s (%s)", p, strerror(errno));
    }
}

static void k1_cap_open(const char *dir, int T, int L, int H, int E, int K) {
    memset(&k1cap, 0, sizeof(k1cap));
    if (strlen(dir) > sizeof(k1cap.dir) - 1) k1_die("K1_CAPTURE path is too long");
    snprintf(k1cap.dir, sizeof(k1cap.dir), "%s", dir);
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) k1_die("K1_CAPTURE=%s: %s", dir, strerror(errno));
    k1cap.T = T; k1cap.L = L; k1cap.H = H; k1cap.E = E; k1cap.K = K;
    k1cap.fifo_fd = -1;
    k1_cap_unlink("DONE.json"); k1_cap_unlink("ENGINE.json"); k1_cap_unlink("DONE.json.tmp"); k1_cap_unlink("ENGINE.json.tmp");
    k1_cap_unlink("nat_sel.i16"); k1_cap_unlink("nat_sel.i16.tmp");
    char p[3200];
    snprintf(p, sizeof(p), "%s/chunks.jsonl", dir);
    k1cap.chunks = fopen(p, "wb");
    if (!k1cap.chunks) k1_die("cannot create %s (%s)", p, strerror(errno));
    k1cap.files = k1_calloc((size_t)L * K1_CF_N, sizeof(K1CapFile), "capture files");
    static const char *suffix[K1_CF_N] = { "x_out.f64", "z.f32", "sel.i16", "w.f32" };   /* FP64 residual capture */
    const size_t rbytes[K1_CF_N] = { (size_t)H * 8, (size_t)E * 4, (size_t)K * 2, (size_t)K * 4 };
    for (int l = 0; l < L; l++)
        for (int k = 0; k < K1_CF_N; k++) {
            K1CapFile *cf = &k1cap.files[l * K1_CF_N + k];
            snprintf(cf->name, sizeof(cf->name), "L%02d.%s", l, suffix[k]);
            snprintf(p, sizeof(p), "%s/%s", dir, cf->name);
            cf->f = fopen(p, "wb");
            if (!cf->f) k1_die("cannot create %s (%s)", p, strerror(errno));
            cf->rb = rbytes[k];
            cf->buf = k1_malloc(cf->rb * K1_CAP_CHUNK, "capture chunk buffer");
        }
    const char *ka = getenv("K1_TEST_KILL_AT");
    k1cap.kill_at = ka && *ka ? atoi(ka) : -1;
    k1cap.on = 1;
}

static void k1_cap_flush_chunk(K1CapFile *cf, int64_t pos0, int nrows) {
    size_t bytes = (size_t)nrows * cf->rb;
    if (fwrite(cf->buf, 1, bytes, cf->f) != bytes) k1_die("capture: short write to %s", cf->name);
    k1_cap_fsync_file(cf->f);
    char hex[65];
    k1_sha_hex(cf->buf, bytes, hex);
    fprintf(k1cap.chunks, "{\"file\":\"%s\",\"pos0\":%lld,\"n\":%d,\"offset\":%lld,\"bytes\":%zu,\"sha256\":\"%s\"}\n",
            cf->name, (long long)pos0, nrows, (long long)cf->offset, bytes, hex);
    k1_cap_fsync_file(k1cap.chunks);
    cf->offset += (int64_t)bytes; cf->nchunks++; k1cap.nlines++;
}

/* rows p0..p0+n-1 of one file; `data` is n rows of cf->rb bytes */
static void k1_cap_put(K1CapFile *cf, int64_t p0, int n, const void *data) {
    if (p0 != cf->next) k1_die("capture: %s got position %lld, expected %lld (positions must ascend)", cf->name, (long long)p0, (long long)cf->next);
    const uint8_t *src = data;
    for (int r = 0; r < n; r++) {
        int64_t pos = p0 + r;
        int slot = (int)(pos % K1_CAP_CHUNK);
        memcpy(cf->buf + (size_t)slot * cf->rb, src + (size_t)r * cf->rb, cf->rb);
        cf->next = pos + 1;
        if (slot == K1_CAP_CHUNK - 1) k1_cap_flush_chunk(cf, pos - (K1_CAP_CHUNK - 1), K1_CAP_CHUNK);
        else if (cf->next == k1cap.T) k1_cap_flush_chunk(cf, pos - slot, slot + 1);
    }
}

static void k1_cap_check_finite_d(const double *x, size_t n, const char *what, int l, int64_t p0) {
    for (size_t i = 0; i < n; i++)
        if (!isfinite(x[i])) k1_die("capture: non-finite %s at layer %d, position %lld, element %zu", what, l, (long long)p0, i);
}

static void k1_cap_check_finite(const float *x, size_t n, const char *what, int l, int64_t p0) {
    for (size_t i = 0; i < n; i++)
        if (!isfinite(x[i])) k1_die("capture: non-finite %s at layer %d, position %lld, element %zu", what, l, (long long)p0, i);
}

/* after layer l of the block p0..p0+n-1: the layer's completed stream, the router logits, the
 * top-K ids and weights.  Pointers are valid for this call only. */
static void k1_cap_layer(int l, int p0, int n, const double *xout, const float *z, const int *ids, const float *w) {
    K1Cap *C = &k1cap;
    if (l >= C->L) return;
    if (p0 + n > C->T) k1_die("capture: block [%d,+%d) exceeds the declared input length %d", p0, n, C->T);
    k1_cap_check_finite_d(xout, (size_t)n * C->H, "x_out", l, p0);
    k1_cap_check_finite(z, (size_t)n * C->E, "router logits", l, p0);
    k1_cap_check_finite(w, (size_t)n * C->K, "router weights", l, p0);
    int16_t *s16 = k1_malloc((size_t)n * C->K * sizeof(int16_t), "sel16");
    for (int i = 0; i < n * C->K; i++) {
        if (ids[i] < 0 || ids[i] >= C->E || ids[i] > 32767) k1_die("capture: router id %d out of range at layer %d", ids[i], l);
        s16[i] = (int16_t)ids[i];
    }
    k1_cap_put(&C->files[l * K1_CF_N + K1_CF_X], p0, n, xout);
    k1_cap_put(&C->files[l * K1_CF_N + K1_CF_Z], p0, n, z);
    k1_cap_put(&C->files[l * K1_CF_N + K1_CF_SEL], p0, n, s16);
    k1_cap_put(&C->files[l * K1_CF_N + K1_CF_W], p0, n, w);
    free(s16);
    if (C->kill_at >= 0 && p0 <= C->kill_at && C->kill_at < p0 + n && l == C->L / 2) {
        fprintf(stderr, "[kolibri1] K1_TEST_KILL_AT=%d: SIGKILL after layer %d of block [%d,+%d)\n", C->kill_at, l, p0, n);
        fflush(stderr);
        raise(SIGKILL);
    }
}

/* ---------------------------------------------------------- logits stream ---- */
static void k1_cap_fifo_open(const char *path) {
    k1cap.fifo_fd = open(path, O_WRONLY);
    if (k1cap.fifo_fd < 0) k1_die("K1_LOGITS_FIFO=%s: %s", path, strerror(errno));
}
static void k1_cap_write_all(int fd, const void *p, size_t n) {
    const uint8_t *b = p;
    while (n) {
        ssize_t r = write(fd, b, n);
        if (r < 0) { if (errno == EINTR) continue; k1_die("logits stream: write failed (%s)", strerror(errno)); }
        b += r; n -= (size_t)r;
    }
}
static void k1_cap_frame(int64_t pos0, int n, int V, const float *lg) {
    if (pos0 != k1cap.fifo_next) k1_die("logits stream: frame at position %lld, expected %lld", (long long)pos0, (long long)k1cap.fifo_next);
    if (n < 1 || n > K1_CAP_CHUNK) k1_die("logits stream: bad frame length %d", n);
    uint32_t hd[4]; memcpy(hd, "K1LG", 4); hd[1] = (uint32_t)pos0; hd[2] = (uint32_t)n; hd[3] = (uint32_t)V;
    k1_cap_write_all(k1cap.fifo_fd, hd, sizeof(hd));
    k1_cap_write_all(k1cap.fifo_fd, lg, (size_t)n * V * sizeof(float));
    k1cap.fifo_next = pos0 + n; k1cap.frames++;
}

/* clean end: every file must hold all T positions.  Closes the stream, writes ENGINE.json.
 * `extra` is a ready-made list of JSON members (no braces, may be empty). */
static void k1_cap_finish(const char *extra) {
    K1Cap *C = &k1cap;
    if (!C->on) return;
    for (int i = 0; i < C->L * K1_CF_N; i++) {
        K1CapFile *cf = &C->files[i];
        if (cf->next != C->T) k1_die("capture incomplete: %s holds %lld of %d positions", cf->name, (long long)cf->next, C->T);
        k1_cap_fsync_file(cf->f); fclose(cf->f); cf->f = NULL;
    }
    k1_cap_fsync_file(C->chunks); fclose(C->chunks); C->chunks = NULL;
    if (C->fifo_fd >= 0) {
        if (C->fifo_next != C->T) k1_die("logits stream incomplete: %lld of %d positions", (long long)C->fifo_next, C->T);
        close(C->fifo_fd); C->fifo_fd = -1;
    }
    char tmp[3200], fin[3200];
    snprintf(tmp, sizeof(tmp), "%s/ENGINE.json.tmp", C->dir);
    snprintf(fin, sizeof(fin), "%s/ENGINE.json", C->dir);
    FILE *f = fopen(tmp, "wb");
    if (!f) k1_die("cannot create %s", tmp);
    fprintf(f, "{\"schema\":\"kolibri1-engine-capture-2\",\"residual_dtype\":\"fp64\",\"positions\":%d,\"layers\":%d,\"hidden\":%d,\"experts\":%d,\"topk\":%d,"
               "\"chunk_positions\":%d,\"chunk_lines\":%ld,\"logit_frames\":%ld,\"errors\":0%s%s}\n",
            C->T, C->L, C->H, C->E, C->K, K1_CAP_CHUNK, C->nlines, C->frames, extra && *extra ? "," : "", extra ? extra : "");
    k1_cap_fsync_file(f); fclose(f);
    if (rename(tmp, fin) != 0) k1_die("cannot rename %s (%s)", tmp, strerror(errno));
    C->on = 0;
}

/* --------------------------------------------------------- routing replay ---- */
/* Schedule file (little endian): "K1RS" u32 version=1, u32 direction (1 = the engine executes the reference's IDs), u32 T, u32 L, u32 K, u32 E,
 * u8 feed_sha256[32], u8 ckpt_identity_sha256[32], u8 binding_sha256[32] (the comparator's hash of the source-capture binding record,
 * only recorded here), int16 ids[L][T][K], u8 sha256_of_everything_before[32]. */
#define K1_RP_HDR (4 + 6 * 4 + 96)
typedef struct {
    int on;
    int T, L, K, E;
    int16_t *ids;              /* [L][T][K] the schedule */
    int16_t *nat;              /* [L][T][K] the engine's own natural selection (recorded) */
    uint8_t *seen;             /* [L][T] */
    char sched_sha[65], feed_sha[65], ckpt_sha[65], bind_sha[65];
    long rows, differ;         /* rows executed; rows whose forced SET differs from the natural one */
    int mut_l, mut_p;          /* K1_TEST_MUT_NATSEL, -1 = off */
    char nat_sha[65];
} K1Replay;
static K1Replay k1rp;

static void k1_hex32(const uint8_t *b, char hex[65]) { for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", b[i]); }

static void k1_replay_open(const char *path, int T, int L, int K, int E, const char *feed_hex) {
    memset(&k1rp, 0, sizeof(k1rp));
    k1rp.mut_l = k1rp.mut_p = -1;
    FILE *f = fopen(path, "rb");
    if (!f) k1_die("K1_REPLAY_IDS=%s: %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) k1_die("K1_REPLAY_IDS=%s: cannot seek", path);
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) k1_die("K1_REPLAY_IDS=%s: cannot size", path);
    if (sz < K1_RP_HDR + 32) k1_die("K1_REPLAY_IDS=%s: %ld bytes is shorter than the header and checksum (truncated schedule)", path, sz);
    uint8_t *b = k1_malloc((size_t)sz, "replay schedule");
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) k1_die("K1_REPLAY_IDS=%s: short read", path);
    fclose(f);
    if (memcmp(b, "K1RS", 4)) k1_die("K1_REPLAY_IDS=%s: bad magic (not a replay schedule)", path);
    uint32_t h[6]; memcpy(h, b + 4, sizeof(h));
    if (h[0] != 1) k1_die("K1_REPLAY_IDS: schedule version %u is not supported", h[0]);
    if (h[1] != 1) k1_die("K1_REPLAY_IDS: schedule direction %u is not 'the engine executes the reference's IDs' (1)", h[1]);
    if ((int)h[2] != T) k1_die("K1_REPLAY_IDS: schedule is for %u positions, this run feeds %d", h[2], T);
    if ((int)h[3] != L) k1_die("K1_REPLAY_IDS: schedule is for %u layers, this run executes %d", h[3], L);
    if ((int)h[4] != K) k1_die("K1_REPLAY_IDS: schedule top-k %u != the model's %d", h[4], K);
    if ((int)h[5] != E) k1_die("K1_REPLAY_IDS: schedule expert count %u != the model's %d", h[5], E);
    size_t nid = (size_t)L * (size_t)T * (size_t)K;
    if ((size_t)sz != (size_t)K1_RP_HDR + nid * 2 + 32) k1_die("K1_REPLAY_IDS: file is %ld bytes, expected %zu (truncated or extended schedule)", sz, (size_t)K1_RP_HDR + nid * 2 + 32);
    char have[65], want[65];
    k1_sha_hex(b, (size_t)sz - 32, have); k1_hex32(b + sz - 32, want);
    if (strcmp(have, want)) k1_die("K1_REPLAY_IDS: schedule checksum %s != recorded %s (corrupted schedule)", have, want);
    k1_hex32(b + 28, k1rp.feed_sha); k1_hex32(b + 60, k1rp.ckpt_sha); k1_hex32(b + 92, k1rp.bind_sha);
    if (!feed_hex || strcmp(feed_hex, k1rp.feed_sha)) k1_die("K1_REPLAY_IDS: schedule is bound to feed %s, this run feeds %s", k1rp.feed_sha, feed_hex ? feed_hex : "(none)");
    k1rp.ids = k1_malloc(nid * 2, "replay ids");
    memcpy(k1rp.ids, b + K1_RP_HDR, nid * 2);
    snprintf(k1rp.sched_sha, sizeof(k1rp.sched_sha), "%s", want);
    free(b);
    for (size_t r = 0; r < (size_t)L * T; r++) {
        const int16_t *row = k1rp.ids + r * (size_t)K;
        for (int i = 0; i < K; i++) {
            if (row[i] < 0 || row[i] >= E) k1_die("K1_REPLAY_IDS: layer %zu position %zu: id %d is outside [0, %d)", r / (size_t)T, r % (size_t)T, row[i], E);
            for (int j = 0; j < i; j++)
                if (row[j] == row[i]) k1_die("K1_REPLAY_IDS: layer %zu position %zu: duplicate id %d", r / (size_t)T, r % (size_t)T, row[i]);
        }
    }
    k1rp.T = T; k1rp.L = L; k1rp.K = K; k1rp.E = E;
    k1rp.nat = k1_calloc(nid, 2, "replay natural selections");
    k1rp.seen = k1_calloc((size_t)L * T, 1, "replay seen");
    const char *m = getenv("K1_TEST_MUT_NATSEL");
    if (m && *m) {
        int ml, mp;
        if (sscanf(m, "%d:%d", &ml, &mp) != 2 || ml < 0 || ml >= L || mp < 0 || mp >= T) k1_die("K1_TEST_MUT_NATSEL=%s: expected <layer>:<pos> inside the run", m);
        k1rp.mut_l = ml; k1rp.mut_p = mp;
        fprintf(stderr, "[kolibri1] ******** K1_TEST_MUT_NATSEL=%s: the RECORDED natural selection of that row is corrupted (control) ********\n", m);
    }
    k1rp.on = 1;
    fprintf(stderr, "[kolibri1] K1_REPLAY_IDS: executing the schedule's expert IDs (%s), weights = sigmoid(own z[forced ids]); natural selections recorded separately\n", k1rp.sched_sha);
}

/* clean end: every (layer, position) was recorded exactly once.  Writes nat_sel.i16 (tmp + fsync + rename) and fills `out` with the
 * "replay" member of ENGINE.json.  `dir` is the capture directory. */
static void k1_replay_finish(const char *dir, char *out, size_t cap) {
    K1Replay *R = &k1rp;
    for (size_t i = 0; i < (size_t)R->L * R->T; i++)
        if (R->seen[i] != 1) k1_die("replay: layer %zu position %zu was recorded %d times (expected once)", i / (size_t)R->T, i % (size_t)R->T, R->seen[i]);
    size_t nb = (size_t)R->L * R->T * R->K * 2;
    k1_sha_hex(R->nat, nb, R->nat_sha);
    char tmp[3300], fin[3300];
    snprintf(tmp, sizeof(tmp), "%s/nat_sel.i16.tmp", dir); snprintf(fin, sizeof(fin), "%s/nat_sel.i16", dir);
    FILE *f = fopen(tmp, "wb");
    if (!f) k1_die("cannot create %s", tmp);
    if (fwrite(R->nat, 1, nb, f) != nb) k1_die("short write to %s", tmp);
    k1_cap_fsync_file(f); fclose(f);
    if (rename(tmp, fin) != 0) k1_die("cannot rename %s (%s)", tmp, strerror(errno));
    snprintf(out, cap, "\"replay\":{\"schedule_sha256\":\"%s\",\"feed_sha256\":\"%s\",\"ckpt_identity_sha256\":\"%s\",\"binding_sha256\":\"%s\","
             "\"positions\":%d,\"layers\":%d,\"topk\":%d,\"experts\":%d,\"rows_executed\":%ld,\"rows_forced_set_differs_from_natural\":%ld,"
             "\"nat_sel_sha256\":\"%s\",\"nat_sel_bytes\":%zu,\"test_mut_natsel\":%s}",
             R->sched_sha, R->feed_sha, R->ckpt_sha, R->bind_sha, R->T, R->L, R->K, R->E, R->rows, R->differ, R->nat_sha, nb,
             R->mut_l >= 0 ? "true" : "false");
}

#endif /* KOLIBRI1_CAPTURE_H */
