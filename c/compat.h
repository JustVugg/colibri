/* compat.h — shim di portabilita' per piattaforme non-Linux (oggi: macOS / Apple Silicon,
 * Windows 11 x86-64 via MinGW-w64).
 * Su Linux questo header e' un NO-OP totale: nessun simbolo definito o ridefinito,
 * zero impatto sul percorso x86 esistente.
 * Regola: ogni differenza di piattaforma vive QUI; i .c restano puliti. */
#ifndef COMPAT_H
#define COMPAT_H

#ifdef __APPLE__
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

/* --- posix_fadvise: assente su macOS ---
 * WILLNEED -> F_RDADVISE (readahead esplicito: stessa semantica).
 * DONTNEED -> no-op: XNU non espone un drop mirato per-range; la sua unified
 *             buffer cache si autoregola sotto pressione. Il motore usa DONTNEED
 *             solo come consiglio, quindi ignorarlo e' corretto (e su una macchina
 *             con molta RAM tenere le pagine e' proprio cio' che si vuole). */
#ifndef POSIX_FADV_NORMAL
#define POSIX_FADV_NORMAL      0
#define POSIX_FADV_RANDOM      1
#define POSIX_FADV_SEQUENTIAL  2
#define POSIX_FADV_WILLNEED    3
#define POSIX_FADV_DONTNEED    4
#define POSIX_FADV_NOREUSE     5
#endif
static inline int compat_fadvise(int fd, off_t off, off_t len, int advice){
    if(advice==POSIX_FADV_WILLNEED){
        struct radvisory ra;
        ra.ra_offset = off;
        ra.ra_count  = (int)(len>0x7FFFFFFF ? 0x7FFFFFFF : len);
        return fcntl(fd, F_RDADVISE, &ra)<0 ? -1 : 0;
    }
    return 0;
}
#define posix_fadvise compat_fadvise

/* --- O_DIRECT: assente su macOS ---
 * L'equivalente e' F_NOCACHE sul fd (bypass della unified buffer cache).
 * compat_open_direct() apre il fd "gemello" senza cache, come il twin O_DIRECT
 * di st.h. Le pread allineate a 4K del chiamante restano valide: F_NOCACHE non
 * impone vincoli di allineamento. */
static inline int compat_open_direct(const char *path){
    int fd = open(path, O_RDONLY);
    if(fd>=0) fcntl(fd, F_NOCACHE, 1);
    return fd;
}
#endif /* __APPLE__ */

/* ===================================================================
 * Windows 11 x86-64 (MinGW-w64 / MSYS2)
 * ===================================================================
 * pread         -> compat_pread  (ReadFile + OVERLAPPED su raw handle:
 *                                  thread-safe, 64-bit offset, no CRT
 *                                  text-mode translation — NEVER use
 *                                  _read/_lseeki64 which are racy AND
 *                                  corrupt 0x0A bytes in binary files).
 * posix_fadvise -> no-op (advisory only; macOS already no-ops DONTNEED).
 * mlock         -> compat_mlock  (VirtualLock + crescita working set).
 * posix_memalign->_aligned_malloc(free must be compat_aligned_free).
 * rename        -> compat_rename (MoveFileEx MOVEFILE_REPLACE_EXISTING;
 *                                  CRT rename fails EEXIST if dest exists,
 *                                  breaking stats atomic-write every turn).
 * meminfo       -> compat_meminfo (GlobalMemoryStatusEx: ullTotalPhys,
 *                                  ullAvailPhys — approx MemAvailable).
 * getpid        -> _getpid
 * =================================================================== */
#ifdef _WIN32

/* Belt-and-braces: 64-bit off_t mandatory — model is 370 GB, every pread
 * region can exceed 2 GB. 32-bit off_t silently wraps >4 GB offsets into the
 * first 4 GB → reads wrong weight bytes → silent token corruption. */
#if !defined(_FILE_OFFSET_BITS) || _FILE_OFFSET_BITS < 64
#error "_FILE_OFFSET_BITS=64 required on Windows (add -D_FILE_OFFSET_BITS=64 to CFLAGS)"
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#include <malloc.h>
#include <fcntl.h>
#include <errno.h>

/* --- O_BINARY: belt-and-braces vs CRT text-mode (0x0A byte corruption) --- */
#ifndef O_BINARY
#define O_BINARY 0x8000
#endif
/* All open() calls for model data must use binary mode.  The compat_pread
 * wrapper already bypasses CRT via ReadFile on the raw OS handle, so this
 * is defense-in-depth: if anyone adds a future CRT-based read path, O_BINARY
 * prevents 0x0A bytes from being silently translated to \r\n. */
#define COMPAT_O_RDONLY (O_RDONLY | O_BINARY)

/* --- posix_fadvise: Windows has no direct equivalent. Semantics:
 *      WILLNEED  -> warm the OS page cache so a later synchronous pread finds the
 *                   pages resident. Implemented as an overlapped background ReadFile
 *                   into a throwaway scratch buffer (fire-and-forget readahead). Called
 *                   from the dedicated PILOT I/O thread / next-block readahead in moe(),
 *                   NEVER inline on the hot path (the existing comment at glm.c:2847
 *                   measures inline fadvise submit at ~0.5ms x 169k calls = +92s/48tok).
 *                   Each call owns its OVERLAPPED + scratch buffer -> thread-safe.
 *      DONTNEED  -> no-op: Windows' standby-list trimming self-regulates under pressure,
 *                   and on a low-RAM host keeping the pages is what we want for reuse.
 *                   Matches macOS (compat.h:16-19) which no-ops DONTNEED for the same
 *                   reason. The engine only ever uses DONTNEED as an advisory. */
#ifndef POSIX_FADV_NORMAL
#define POSIX_FADV_NORMAL      0
#define POSIX_FADV_RANDOM      1
#define POSIX_FADV_SEQUENTIAL  2
#define POSIX_FADV_WILLNEED    3
#define POSIX_FADV_DONTNEED    4
#define POSIX_FADV_NOREUSE     5
#endif
static inline int compat_fadvise(int fd, off_t off, off_t len, int advice){
    if(advice!=POSIX_FADV_WILLNEED || len<=0) return 0;
    intptr_t osfh=_get_osfhandle(fd);
    if(osfh==-1 || osfh==-2) return 0;
    HANDLE h=(HANDLE)osfh;
    /* Cap the readahead window: reading a whole 19MB expert per hint is fine on the
     * PILOT thread, but a pathological huge len would spike transient memory. */
    size_t rdlen = (len>(off_t)(64*1024*1024)) ? (size_t)(64*1024*1024) : (size_t)len;
    char *buf=(char*)_aligned_malloc(rdlen, 4096);
    if(!buf) return -1;
    OVERLAPPED ov={0};
    ov.Offset     = (DWORD)( (off_t)off        & 0xFFFFFFFFULL);
    ov.OffsetHigh = (DWORD)(((off_t)off >> 32) & 0xFFFFFFFFULL);
    /* Issue an overlapped read. With a non-OVERLAPPED-opened handle ReadFile still
     * accepts lpOverlapped (it carries the 64-bit offset) and blocks until the read
     * completes — but crucially it populates the standby page cache for this region,
     * so the later synchronous pread on the same offsets faults from RAM not disk. */
    DWORD got=0;
    ReadFile(h, buf, (DWORD)rdlen, &got, &ov);
    _aligned_free(buf);
    return 0;
}
#define posix_fadvise compat_fadvise

/* --- pread -> ReadFile + OVERLAPPED su raw OS handle ---
 * Thread-safe (no shared seek position). Gestisce offset >4 GB e chunking
 * per letture >2 GB (anche se i tensori individuali sono nell'ordine dei
 * MB-centinaia di MB, il wrapper e' robusto per ogni taglia). */
static inline ssize_t compat_pread(int fd, void *buf, size_t n, off_t off){
    intptr_t osfh = _get_osfhandle(fd);
    if(osfh == -1 || osfh == -2){ errno = EBADF; return -1; }
    HANDLE h = (HANDLE)osfh;
    size_t total = 0;
    while(total < n){
        size_t chunk = n - total;
        DWORD chunk32 = (chunk > 0x7FFFFFFF) ? 0x7FFFFFFF : (DWORD)chunk;
        OVERLAPPED ov = {0};
        ov.Offset     = (DWORD)( (off + (off_t)total)        & 0xFFFFFFFFULL);
        ov.OffsetHigh = (DWORD)(((off + (off_t)total) >> 32) & 0xFFFFFFFFULL);
        DWORD rd = 0;
        if(!ReadFile(h, (char*)buf + total, chunk32, &rd, &ov)){
            DWORD err = GetLastError();
            if(err == ERROR_HANDLE_EOF) break;  /* past EOF → return bytes read (0 if none, matching POSIX pread) */
            if(err == ERROR_INVALID_HANDLE || err == ERROR_INVALID_FUNCTION) errno = EBADF;
            else errno = EIO;
            return -1;
        }
        total += rd;
        if(rd == 0 || rd < chunk32) break;  /* EOF or partial (file truncated) */
    }
    return (ssize_t)total;
}
#define pread(fd,buf,n,off) compat_pread(fd,buf,n,off)

/* --- mlock -> VirtualLock con crescita del working set ---
 * VirtualLock fallisce oltre il working set MINIMO del processo (default ~qualche
 * centinaio di KB): prima si allarga il working set di len + margine, poi si blocca.
 * Best effort come mlock su Linux: -1 su fallimento, il chiamante decide (pin_wire
 * lo tratta come non-fatale). SeIncreaseWorkingSetPrivilege e' concesso agli utenti
 * standard di default. */
static inline int compat_mlock(const void *addr, size_t len){
    HANDLE p = GetCurrentProcess();
    SIZE_T mn = 0, mx = 0;
    if(GetProcessWorkingSetSize(p, &mn, &mx)){
        SIZE_T need = len + (SIZE_T)(1u<<20);
        SetProcessWorkingSetSize(p, mn + need, mx + need);   /* best effort */
    }
    return VirtualLock((LPVOID)addr, len) ? 0 : -1;
}
static inline int compat_munlock(const void *addr, size_t len){
    return VirtualUnlock((LPVOID)addr, len) ? 0 : -1;
}

/* --- posix_memalign -> _aligned_malloc ---
 * ATTN: memoria allocata con _aligned_malloc DEVE essere liberata con
 * _aligned_free, NON con free(). Vedi compat_aligned_free sotto.
 * Audit: l'unico sito che libera memoria aligned e' free(s->slab) in
 * glm.c:892 (cambiato in compat_aligned_free). s->fslab usa falloc()
 * (malloc semplice) -> il suo free() resta plain. */
#ifndef ENOMEM
#define ENOMEM 12
#endif
static inline int compat_posix_memalign(void **memptr, size_t alignment, size_t size){
    if(alignment < sizeof(void*)) alignment = sizeof(void*);
    *memptr = _aligned_malloc(size, alignment);
    return *memptr ? 0 : ENOMEM;
}
#define posix_memalign(memptr,alignment,size) compat_posix_memalign(memptr,alignment,size)

/* matching free per memoria aligned di _aligned_malloc */
#define compat_aligned_free _aligned_free

/* --- meminfo: GlobalMemoryStatusEx ---
 * ullAvailPhys ~ MemAvailable di Linux (include standby/free/zero pages —
 * pagine recuperabili senza swap). Guida il cap automatico della cache
 * expert: se sbagliato, la cache e' mis-sized → swap thrash o OOM. */
static inline void compat_meminfo(double *total_gb, double *avail_gb){
    MEMORYSTATUSEX msx = {0};
    msx.dwLength = sizeof(msx);
    if(GlobalMemoryStatusEx(&msx)){
        *total_gb = (double)msx.ullTotalPhys / 1e9;
        *avail_gb = (double)msx.ullAvailPhys  / 1e9;
    } else {
        *total_gb = 0; *avail_gb = 0;
    }
}

/* --- rename -> MoveFileEx (CRT rename EEXIST se destinazione esiste) ---
 * stats_dump_q chiama rename(tmp, path) OGNI turno di serve: dopo il primo
 * write il file esiste gia', e CRT rename fallisce silenziosamente,
 * affamando la pipeline REPIN/heat/PIN del suo segnale persistente. */
static inline int compat_rename(const char *old, const char *new){
    return MoveFileExA(old, new, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
}
#define rename(old,new) compat_rename(old,new)

/* --- getpid -> _getpid --- */
#define getpid() _getpid()

/* --- rss_gb: getrusage -> GetProcessMemoryInfo ---
 * ru_maxrss in KB (come Linux): rss_gb() divide per 1e6 → GB corretti. */
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
struct rusage { long ru_maxrss; };
#define RUSAGE_SELF 0
static inline int getrusage(int who, struct rusage *r){
    (void)who;
    PROCESS_MEMORY_COUNTERS_EX pmc = {0};
    pmc.cb = sizeof(pmc);
    if(GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))){
        r->ru_maxrss = (long)(pmc.PeakWorkingSetSize / 1024);  /* ru_maxrss = peak, not current */
        return 0;
    }
    r->ru_maxrss = 0; return -1;
}

/* --- getline -> compat_getline (fgets + realloc) --- */
#include <sys/types.h>  /* ssize_t */
static inline ssize_t compat_getline(char **lineptr, size_t *n, FILE *stream){
    if(!lineptr || !n || !stream){ errno = EINVAL; return -1; }
    if(!*lineptr || !*n){ *n = 128; free(*lineptr); *lineptr = malloc(*n); if(!*lineptr) return -1; }
    size_t pos = 0; int c;
    while((c = fgetc(stream)) != EOF){
        if(pos + 1 >= *n){ size_t nn = *n * 2; char *np = realloc(*lineptr, nn); if(!np) return -1; *lineptr = np; *n = nn; }
        (*lineptr)[pos++] = (char)c;
        if(c == '\n') break;
    }
    if(pos == 0) return -1;
    (*lineptr)[pos] = '\0';
    return (ssize_t)pos;
}
#define getline(lineptr,n,stream) compat_getline(lineptr,n,stream)

/* --- O_DIRECT -> FILE_FLAG_NO_BUFFERING ---
 * Apre il fd "gemello" senza cache del file system, come il twin O_DIRECT di
 * st.h su Linux e F_NOCACHE su macOS. Stesso contratto: offset, lunghezza e
 * buffer del chiamante devono essere allineati a 4K (gli slab expert usano
 * posix_memalign(4096) e il percorso DIRECT=1 del motore allinea gia' offset
 * e len); richieste non allineate falliscono con -1, mai dati corrotti.
 * Il fd si usa con la normale pread() (compat_pread -> ReadFile+OVERLAPPED). */
static inline int compat_open_direct(const char *path){
    HANDLE h = CreateFileA(path, GENERIC_READ,
                           FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
    if(h == INVALID_HANDLE_VALUE) return -1;
    int fd = _open_osfhandle((intptr_t)h, _O_RDONLY|_O_BINARY);
    if(fd < 0){ CloseHandle(h); return -1; }
    return fd;
}

/* --- dimensione file da fd: GetFileSizeEx ---
 * La lseek(SEEK_END) del CRT ritorna -1 sui fd NO_BUFFERING (misurato su
 * UCRT): la dimensione si chiede direttamente al kernel. Funziona su
 * qualsiasi fd (buffered o direct). -1 su errore. */
static inline off_t compat_fsize(int fd){
    intptr_t osfh = _get_osfhandle(fd);
    if(osfh == -1 || osfh == -2) return -1;
    LARGE_INTEGER li;
    if(!GetFileSizeEx((HANDLE)osfh, &li)) return -1;
    return (off_t)li.QuadPart;
}

/* --- setenv -> SetEnvironmentVariableA (POSIX setenv assente su Windows) --- */
static inline int compat_setenv(const char *name, const char *value, int overwrite){
    if(!overwrite && getenv(name)) return 0;
    return SetEnvironmentVariableA(name, value) ? 0 : -1;
}
#define setenv(name,value,overwrite) compat_setenv(name,value,overwrite)

#endif /* _WIN32 */

/* --- compat_aligned_free su piattaforme diverse da Windows ---
 * Su Linux/macOS, posix_memalign usa free() normale. */
#ifndef compat_aligned_free
#define compat_aligned_free free
#endif

/* --- COMPAT_O_RDONLY: O_RDONLY con O_BINARY su Windows, O_RDONLY puro altrove --- */
#ifndef COMPAT_O_RDONLY
#define COMPAT_O_RDONLY O_RDONLY
#endif

#endif /* COMPAT_H */
