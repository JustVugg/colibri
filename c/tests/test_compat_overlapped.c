/* test_compat_overlapped.c -- on Windows, the descriptor several threads read at once.
 *
 * A handle opened without FILE_FLAG_OVERLAPPED is a synchronous file object: the I/O
 * manager holds its lock for the whole of each request, so the reads of a team of
 * threads go through it one at a time. compat_open_overlapped() gives st.h a second
 * descriptor of a shard for the readers that find the first one taken; pread() waits
 * for its own read on it.
 *
 * Pinned here: the kernel's own word on that handle (no synchronous-mode bit), and that
 * pread() means on it what it means on any descriptor -- the bytes under contention,
 * the short read at end of file -- with fstat, lseek(SEEK_END), compat_fsize, the
 * read-only mapping and the readahead hint still working. The descriptors that are
 * not overlapped (open()'s, the unbuffered twin) are read by a team here too: pread()
 * changed under them; and so is one that is overlapped and unbuffered, where every
 * read really waits for the drive. Who reads through which descriptor is st.h's and is in
 * test_st_shared_read.c. How much the overlap is worth is a measurement: c/iobench.c. */
#include <stdio.h>

#ifndef _WIN32
int main(void){ puts("compat overlapped tests: skipped (POSIX pread already overlaps)"); return 0; }
#else

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include "../compat.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
#define TMPF "test_overlapped.tmp"
#define FSZ ((8u << 20) + 123u)          /* not a sector multiple: the last direct block is short */
#define NTHREADS 8
#define NREADS 300

static uint8_t pat(uint32_t i){ return (uint8_t)((i * 2654435761u) >> 24); }
static int same(const uint8_t *b, uint32_t off, uint32_t n){
    for(uint32_t i = 0; i < n; i++) if(b[i] != pat(off + i)) return 0;
    return 1;
}

/* FILE_MODE_INFORMATION, as the I/O manager keeps it for the file object. */
#define MODE_SYNCHRONOUS 0x30u           /* FILE_SYNCHRONOUS_IO_ALERT | FILE_SYNCHRONOUS_IO_NONALERT */
static int handle_mode(int fd, unsigned *mode){
    typedef LONG (WINAPI *QueryFn)(HANDLE, void *, void *, ULONG, int);
    QueryFn query = (QueryFn)(void *)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryInformationFile");
    struct { void *status; ULONG_PTR information; } iosb;
    ULONG m = 0;
    if(!query || query((HANDLE)_get_osfhandle(fd), &iosb, &m, sizeof m, 16 /* FileModeInformation */) < 0) return 0;
    *mode = (unsigned)m;
    return 1;
}

typedef struct { int fd, direct, tid, bad; } Job;

static unsigned __stdcall reader(void *arg){
    Job *j = (Job *)arg;
    uint8_t *buf = (uint8_t *)_aligned_malloc(1u << 20, 4096);
    uint32_t s = 2463534242u * (uint32_t)(j->tid + 1);
    if(!buf){ j->bad = 1; return 0; }
    for(int i = 0; i < NREADS && !j->bad; i++){
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        uint32_t off = s % FSZ, len = 1 + (s >> 8) % 300000u;
        if(j->direct){ off &= ~4095u; len = (len + 4095u) & ~4095u; }
        uint32_t want = FSZ - off < len ? FSZ - off : len;      /* a read that reaches the end is short */
        ssize_t r = pread(j->fd, buf, len, (off_t)off);
        if(r != (ssize_t)want || !same(buf, off, want)) j->bad = 1;
    }
    _aligned_free(buf);
    return 0;
}

static int read_together(int fd, int direct){
    Job job[NTHREADS]; HANDLE th[NTHREADS];
    for(int t = 0; t < NTHREADS; t++){
        job[t] = (Job){fd, direct, t, 0};
        th[t] = (HANDLE)_beginthreadex(NULL, 0, reader, &job[t], 0, NULL);
        if(!th[t]) return 0;
    }
    WaitForMultipleObjects(NTHREADS, th, TRUE, INFINITE);
    int ok = 1;
    for(int t = 0; t < NTHREADS; t++){ CloseHandle(th[t]); if(job[t].bad) ok = 0; }
    return ok;
}

int main(void){
    uint8_t *all = (uint8_t *)malloc(FSZ);
    CHECK(all);
    for(uint32_t i = 0; i < FSZ; i++) all[i] = pat(i);
    FILE *w = fopen(TMPF, "wb");
    CHECK(w && fwrite(all, 1, FSZ, w) == FSZ);
    fclose(w);
    free(all);

    int fd = compat_open_overlapped(TMPF), dfd = compat_open_direct(TMPF);
    CHECK(fd >= 0 && dfd >= 0);

    /* the point of the descriptor: it is not a synchronous file object */
    unsigned mode = ~0u;
    CHECK(handle_mode(fd, &mode) && !(mode & MODE_SYNCHRONOUS));
    int plain = open(TMPF, COMPAT_O_RDONLY);                    /* and the probe can tell: the CRT's own is */
    CHECK(plain >= 0 && handle_mode(plain, &mode) && (mode & MODE_SYNCHRONOUS));

    /* every thread gets its own bytes back, on the overlapped descriptor and on the others */
    CHECK(read_together(fd, 0));
    CHECK(read_together(plain, 0));
    CHECK(read_together(dfd, 1));

    /* and where every read is still in flight while the next ones start: overlapped and unbuffered,
     * each read waits for the drive. A pread that waited on anything but its own event would hand
     * back a read that is not finished. */
    HANDLE oh = CreateFileA(TMPF, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE, NULL,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED|FILE_FLAG_NO_BUFFERING, NULL);
    int ofd = oh == INVALID_HANDLE_VALUE ? -1 : _open_osfhandle((intptr_t)oh, _O_RDONLY|_O_BINARY);
    CHECK(ofd >= 0);
    CHECK(read_together(ofd, 1));

    /* end of file, as POSIX pread reports it */
    uint8_t *buf = (uint8_t *)_aligned_malloc(1u << 16, 4096);
    CHECK(buf);
    CHECK(pread(fd, buf, 100, (off_t)FSZ - 40) == 40 && same(buf, FSZ - 40, 40));
    CHECK(pread(fd, buf, 100, (off_t)FSZ) == 0);
    CHECK(pread(fd, buf, 100, (off_t)FSZ + 4096) == 0);
    CHECK(pread(dfd, buf, 4096, (off_t)(8u << 20)) == 123 && same(buf, 8u << 20, 123));
    CHECK(pread(dfd, buf, 4096, (off_t)(8u << 20) + 4096) == 0);
    CHECK(pread(dfd, buf, 4096, 1000) < 0);                     /* unaligned: refused, never wrong bytes */
    CHECK(pread(ofd, buf, 4096, (off_t)(8u << 20)) == 123 && same(buf, 8u << 20, 123));
    CHECK(pread(ofd, buf, 4096, (off_t)(8u << 20) + 4096) == 0);
    CHECK(pread(ofd, buf, 4096, 1000) < 0);

    /* what else st.h and the engines may ask of a descriptor they read */
    struct stat sb;
    CHECK(fstat(fd, &sb) == 0 && (int64_t)sb.st_size == (int64_t)FSZ);
    CHECK((int64_t)lseek(fd, 0, SEEK_END) == (int64_t)FSZ);
    CHECK(compat_fsize(fd) == (off_t)FSZ && compat_fsize(dfd) == (off_t)FSZ);
    compat_ro_map map; const void *view = NULL;
    CHECK(compat_map_readonly(fd, 4097, 70000, &map, &view) == 0 && same((const uint8_t *)view, 4097, 70000));
    compat_unmap_readonly(&map);
    CHECK(posix_fadvise(fd, 0, (off_t)FSZ, POSIX_FADV_WILLNEED) == 0);
    CHECK(pread(fd, buf, 1u << 16, 12345) == (ssize_t)(1u << 16) && same(buf, 12345, 1u << 16));

    /* no file, no second descriptor: the caller keeps the one it has */
    CHECK(compat_open_overlapped("no_such_shard.tmp") < 0);

    _aligned_free(buf);
    close(ofd); close(plain); close(dfd); close(fd);
    remove(TMPF);
    puts("compat overlapped tests: ok");
    return 0;
}
#endif
