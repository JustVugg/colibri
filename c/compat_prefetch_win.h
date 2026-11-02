/* Bounded, advisory file readahead for Windows. Included by compat.h after
 * windows.h. No model pointers or CRT descriptors escape into a worker. */
#ifndef COLI_COMPAT_PREFETCH_WIN_H
#define COLI_COMPAT_PREFETCH_WIN_H

#define COMPAT_PREFETCH_JOBS 4
#define COMPAT_PREFETCH_BYTES (64u * 1024u * 1024u)

/* Keep the Windows 7 build target usable: the hint is optional on systems
 * without PrefetchVirtualMemory. This is its Windows 8 range ABI. */
typedef struct { PVOID VirtualAddress; SIZE_T NumberOfBytes; } CompatPrefetchRange;
typedef BOOL (WINAPI *CompatPrefetchFn)(HANDLE, ULONG_PTR, CompatPrefetchRange *, ULONG);
typedef struct {
    HANDLE file;
    HMODULE module;
    unsigned long long offset;
    SIZE_T bytes;
} CompatPrefetchJob;
static volatile LONG compat_prefetch_pending;

static VOID CALLBACK compat_prefetch_run(PTP_CALLBACK_INSTANCE instance, PVOID context) {
    CompatPrefetchJob *job = (CompatPrefetchJob *)context;
    /* An SDK DLL may be released as soon as its caller returns. The worker
     * owns a module reference until the thread pool has left this callback. */
    if (job->module) FreeLibraryWhenCallbackReturns(instance, job->module);
    CallbackMayRunLong(instance);
    CompatPrefetchFn prefetch = (CompatPrefetchFn)(void *)GetProcAddress(
        GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    LARGE_INTEGER size;
    if (prefetch && GetFileSizeEx(job->file, &size) && size.QuadPart > 0 &&
        job->offset < (unsigned long long)size.QuadPart) {
        unsigned long long left = (unsigned long long)size.QuadPart - job->offset;
        SIZE_T bytes = left < job->bytes ? (SIZE_T)left : job->bytes;
        SYSTEM_INFO info; GetSystemInfo(&info);
        unsigned long long base = job->offset - job->offset % info.dwAllocationGranularity;
        SIZE_T delta = (SIZE_T)(job->offset - base);
        HANDLE mapping = CreateFileMappingW(job->file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (mapping) {
            void *view = MapViewOfFile(mapping, FILE_MAP_READ, (DWORD)(base >> 32),
                                      (DWORD)base, delta + bytes);
            if (view) {
                CompatPrefetchRange range = {(char *)view + delta, bytes};
                prefetch(GetCurrentProcess(), 1, &range, 0);
                UnmapViewOfFile(view);
            }
            CloseHandle(mapping);
        }
    }
    CloseHandle(job->file);
    free(job);
    InterlockedDecrement(&compat_prefetch_pending);
}

static inline int compat_prefetch_file(int fd, off_t off, off_t len) {
    if (fd < 0 || off < 0 || len <= 0) return 0;
    /* Drop excess hints instead of queueing stale experts or blocking inference.
     * The worker maps at most 64 MiB and allocates no read-sized scratch copy. */
    LONG n = InterlockedCompareExchange(&compat_prefetch_pending, 0, 0);
    for (;;) {
        if (n >= COMPAT_PREFETCH_JOBS) return 0;
        LONG was = InterlockedCompareExchange(&compat_prefetch_pending, n + 1, n);
        if (was == n) break;
        n = was;
    }
    intptr_t osfh = _get_osfhandle(fd);
    HANDLE file = NULL;
    HMODULE module = NULL;
    CompatPrefetchJob *job = NULL;
    if (osfh == -1 || osfh == -2 ||
        !DuplicateHandle(GetCurrentProcess(), (HANDLE)osfh, GetCurrentProcess(),
                         &file, 0, FALSE, DUPLICATE_SAME_ACCESS)) goto fail;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                           (LPCSTR)(void *)compat_prefetch_run, &module)) goto fail;
    /* Executables cannot be unloaded; only a DLL needs deferred release. */
    if (module == GetModuleHandleA(NULL)) module = NULL;
    job = (CompatPrefetchJob *)malloc(sizeof(*job));
    if (!job) goto fail;
    *job = (CompatPrefetchJob){file, module, (unsigned long long)off,
        len > (off_t)COMPAT_PREFETCH_BYTES ? COMPAT_PREFETCH_BYTES : (SIZE_T)len};
    if (TrySubmitThreadpoolCallback(compat_prefetch_run, job, NULL)) return 0;
fail:
    if (file) CloseHandle(file);
    if (module) FreeLibrary(module);
    free(job);
    InterlockedDecrement(&compat_prefetch_pending);
    return 0;                       /* advice is never required for correctness */
}
#endif
