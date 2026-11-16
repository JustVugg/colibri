/* wex4_verify -- one read-only pass over a WEX4 container (qwen38 or V4.1), then the root to pin.
 *
 *   tools/wex4_verify FILE [THREADS]
 *
 * Checks what the engines check at open (wex4.h: header page hash, index hash and every index
 * rule), then every blob and every record: its stored bytes against the SHA-256 the index holds
 * (blob padding must be zero). Header page, blobs, records and index tile the file, and the root
 * (the header page's hash) covers the index, so a pass means every byte of the file is the one
 * the root names. Prints "COLI_WEX_ROOT=<root>" for the engine's environment; exit 0 only then
 * (1: some blob or record does not match, 2: the file does not open as WEX4).
 * Reads are unbuffered (no page-cache churn), 8 MiB at a time, on THREADS threads (default 8).
 * The file is opened read-only and never written. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../wex4.h"
#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
typedef HANDLE W4vFile;
#define W4V_BAD INVALID_HANDLE_VALUE
static W4vFile w4v_open(const char *path) {
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
}
static void w4v_close(W4vFile f) { CloseHandle(f); }
static uint64_t w4v_size(W4vFile f) { LARGE_INTEGER n; return GetFileSizeEx(f, &n) ? (uint64_t)n.QuadPart : 0; }
/* n bytes at off (both 4096-aligned): 0 on success; this call's own event, so threads may share f */
static int w4v_read(W4vFile f, void *buf, uint32_t n, uint64_t off) {
    OVERLAPPED ov;
    DWORD got = 0;
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)off; ov.OffsetHigh = (DWORD)(off >> 32);
    ov.hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (!ov.hEvent) return -1;
    BOOL ok = ReadFile(f, buf, n, &got, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(f, &ov, &got, TRUE);
    CloseHandle(ov.hEvent);
    return ok && got == n ? 0 : -1;
}
static void *w4v_alloc(size_t n) { return _aligned_malloc(n, WEX4_PAGE); }
static void w4v_free(void *p) { _aligned_free(p); }
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
typedef int W4vFile;
#define W4V_BAD (-1)
static W4vFile w4v_open(const char *path) { return open(path, O_RDONLY); }
static void w4v_close(W4vFile f) { close(f); }
static uint64_t w4v_size(W4vFile f) { struct stat s; return fstat(f, &s) ? 0 : (uint64_t)s.st_size; }
static int w4v_read(W4vFile f, void *buf, uint32_t n, uint64_t off) { return pread(f, buf, n, (off_t)off) == (ssize_t)n ? 0 : -1; }
static void *w4v_alloc(size_t n) { void *p = NULL; return posix_memalign(&p, WEX4_PAGE, n) ? NULL : p; }
static void w4v_free(void *p) { free(p); }
#endif

#define W4V_CHUNK (8u << 20)

/* SHA-256 of `bytes` at `off` (a multiple of 4096 is read; the bytes past `bytes` must be zero).
 * 0 = matches want, 1 = differs, -1 = read failed. */
static int w4v_item(W4vFile f, unsigned char *buf, uint64_t off, uint64_t bytes, const unsigned char *want) {
    Wex4Sha s;
    unsigned char digest[32];
    uint64_t done = 0, span = wex4_align(bytes);
    int pad_ok = 1;
    wex4_sha_init(&s);
    while (done < span) {
        uint32_t n = span - done < W4V_CHUNK ? (uint32_t)(span - done) : W4V_CHUNK, i;
        if (w4v_read(f, buf, n, off + done)) return -1;
        uint32_t data = bytes - done < n ? (uint32_t)(bytes - done) : n;
        wex4_sha_update(&s, buf, data);
        for (i = data; i < n; i++) pad_ok &= !buf[i];
        done += n;
    }
    wex4_sha_final(&s, digest);
    return memcmp(digest, want, 32u) || !pad_ok ? 1 : 0;
}

/* The whole check; root gets the root's 64 hex digits once the file opens. Returns the exit code. */
static int wex4_verify_file(const char *path, int threads, char root[65], FILE *log) {
    W4vFile f = w4v_open(path);
    unsigned char *page = NULL, *index = NULL;
    Wex4Header h;
    Wex4View v;
    int status = 2, bad = 0;
    uint64_t checked = 0, size;
    root[0] = 0;
    if (f == W4V_BAD) { fprintf(log, "%s: cannot be opened\n", path); return 2; }
    size = w4v_size(f);
    page = (unsigned char *)w4v_alloc(WEX4_PAGE);
    if (!page || size < WEX4_PAGE || w4v_read(f, page, WEX4_PAGE, 0)) { fprintf(log, "%s: no header page\n", path); goto done; }
    int rc = wex4_parse_header(page, size, &h);
    if (rc) { fprintf(log, "%s: not a valid WEX4 header page (wex4.h error %d)\n", path, rc); goto done; }
    for (int i = 0; i < 32; i++) snprintf(root + i * 2, 3, "%02x", h.root[i]);
    index = (unsigned char *)w4v_alloc((size_t)wex4_align(h.index_bytes) + WEX4_PAGE);
    if (!index || (h.index_bytes && w4v_read(f, index, (uint32_t)wex4_align(h.index_bytes), h.index_offset)) ||
        (rc = wex4_open(&h, index, (size_t)wex4_align(h.index_bytes), &v))) {
        fprintf(log, "%s: the index does not open (wex4.h error %d)\n", path, rc);
        root[0] = 0;
        goto done;
    }
    fprintf(log, "%s: WEX4 %u.%u, %u blobs, %u records, %u tensors, grid %u x %u, %llu bytes\n", path, h.major, h.minor,
            h.blobs, h.records, h.tensors, h.layers, h.experts, (unsigned long long)size);
    if (threads < 1) threads = 1;
    time_t w0 = time(NULL);
    int64_t items = (int64_t)h.blobs + h.records;
#pragma omp parallel num_threads(threads) reduction(+ : bad, checked)
    {
        unsigned char *buf = (unsigned char *)w4v_alloc(W4V_CHUNK);
#pragma omp for schedule(dynamic, 1)
        for (int64_t i = 0; i < items; i++) {
            uint64_t off, bytes;
            const unsigned char *want;
            if (i < (int64_t)h.blobs) {
                Wex4Blob b; wex4_get_blob(&v, (uint32_t)i, &b); off = b.offset; bytes = b.bytes; want = b.sha;
            } else {
                Wex4Record r; wex4_get_record(&v, (uint32_t)(i - h.blobs), &r); off = r.offset; bytes = r.stored; want = r.sha;
            }
            int got = buf ? w4v_item(f, buf, off, bytes, want) : -1;
            if (got) {
                bad++;
#pragma omp critical
                fprintf(log, "  %s %lld at byte %llu (%llu bytes): %s\n", i < (int64_t)h.blobs ? "blob" : "record",
                        (long long)(i < (int64_t)h.blobs ? i : i - h.blobs), (unsigned long long)off,
                        (unsigned long long)bytes, got < 0 ? "read failed" : "SHA-256 or padding does not match the index");
            } else checked += wex4_align(bytes);
        }
        w4v_free(buf);
    }
    double wall = difftime(time(NULL), w0);
    if (bad) {
        fprintf(log, "FAILED: %d of %lld blobs/records do not match; do not pin this file\n", bad, (long long)items);
        status = 1;
    } else {
        fprintf(log, "every blob and record matches the index: %.2f GB checked in about %.0f s (%d threads)\n",
                checked / 1e9, wall, threads);
        status = 0;
    }
done:
    w4v_free(index);
    w4v_free(page);
    w4v_close(f);
    return status;
}

#ifndef WEX4_VERIFY_NO_MAIN
int main(int argc, char **argv) {
    char root[65];
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s FILE [THREADS]   (read-only; prints COLI_WEX_ROOT=<root> when every record matches)\n", argv[0]);
        return 2;
    }
    int status = wex4_verify_file(argv[1], argc > 2 ? atoi(argv[2]) : 8, root, stdout);
    if (!status) printf("COLI_WEX_ROOT=%s\n", root);
    else if (root[0]) printf("root %s (NOT verified: do not pin)\n", root);
    return status;
}
#endif
