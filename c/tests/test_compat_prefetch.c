/* Windows readahead must never read weights on the caller, grow an unbounded
 * queue, or follow a recycled CRT descriptor. Real mappings and thread-pool
 * workers are used; a gate makes scheduling and resource failures deterministic. */
#include <stdio.h>
#ifndef _WIN32
int main(void) { puts("compat prefetch: skipped (native POSIX advice)"); return 0; }
#else
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

static HANDLE gate;
static volatile LONG submitted, completed, inspected, reads, bad;
static int reject_submit, reject_duplicate, missing_api;
static SIZE_T expected_bytes;
static unsigned char expected_value;
typedef struct { PVOID VirtualAddress; SIZE_T NumberOfBytes; } ProbeRange;
typedef BOOL (WINAPI *PrefetchFn)(HANDLE, ULONG_PTR, ProbeRange *, ULONG);

static BOOL WINAPI probe_prefetch(HANDLE process, ULONG_PTR count, ProbeRange *ranges, ULONG flags) {
    if (count != 1 || flags || ranges[0].NumberOfBytes != expected_bytes ||
        *(unsigned char *)ranges[0].VirtualAddress != expected_value)
        InterlockedIncrement(&bad);
    InterlockedIncrement(&inspected);
    PrefetchFn fn = (PrefetchFn)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    return fn ? fn(process, count, ranges, flags) : TRUE;
}
static FARPROC WINAPI probe_get_proc(HMODULE module, LPCSTR name) {
    if (!strcmp(name, "PrefetchVirtualMemory"))
        return missing_api ? NULL : (FARPROC)(void *)probe_prefetch;
    return GetProcAddress(module, name);
}
typedef struct { PTP_SIMPLE_CALLBACK fn; PVOID arg; } ProbeWork;
static VOID CALLBACK probe_worker(PTP_CALLBACK_INSTANCE instance, PVOID arg) {
    ProbeWork work = *(ProbeWork *)arg; free(arg);
    if (WaitForSingleObject(gate, 10000) != WAIT_OBJECT_0) InterlockedIncrement(&bad);
    work.fn(instance, work.arg);
    InterlockedIncrement(&completed);
}
static BOOL WINAPI probe_submit(PTP_SIMPLE_CALLBACK fn, PVOID arg, PTP_CALLBACK_ENVIRON env) {
    if (reject_submit) return FALSE;
    ProbeWork *work = malloc(sizeof(*work));
    if (!work) return FALSE;
    *work = (ProbeWork){fn, arg};
    if (!TrySubmitThreadpoolCallback(probe_worker, work, env)) { free(work); return FALSE; }
    InterlockedIncrement(&submitted);
    return TRUE;
}
static BOOL WINAPI probe_duplicate(HANDLE a, HANDLE b, HANDLE c, LPHANDLE d, DWORD e, BOOL f, DWORD g) {
    return !reject_duplicate && DuplicateHandle(a, b, c, d, e, f, g);
}
static BOOL WINAPI probe_read(HANDLE a, LPVOID b, DWORD c, LPDWORD d, LPOVERLAPPED e) {
    InterlockedIncrement(&reads);
    return ReadFile(a, b, c, d, e);
}
#define GetProcAddress probe_get_proc
#define TrySubmitThreadpoolCallback probe_submit
#define DuplicateHandle probe_duplicate
#define ReadFile probe_read
#include "../compat.h"
#undef GetProcAddress
#undef TrySubmitThreadpoolCallback
#undef DuplicateHandle
#undef ReadFile

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
static int drain(void) {
    SetEvent(gate);
    ULONGLONG end = GetTickCount64() + 10000;
    while (InterlockedCompareExchange(&completed, 0, 0) != submitted && GetTickCount64() < end) Sleep(1);
    if (completed != submitted || compat_prefetch_pending || bad)
        fprintf(stderr, "drain: submitted=%ld completed=%ld pending=%ld inspected=%ld bad=%ld\n",
                submitted, completed, compat_prefetch_pending, inspected, bad);
    return completed == submitted && !compat_prefetch_pending && !bad;
}
int main(void) {
    gate = CreateEventW(NULL, TRUE, FALSE, NULL); CHECK(gate);
    const char *path = "test_prefetch.tmp";
    int fd = _open(path, _O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
    CHECK(fd >= 0);
    unsigned char data[4096]; memset(data, 0x5a, sizeof data);
    CHECK(_write(fd, data, sizeof data) == sizeof data);
    expected_value = 0x5a; expected_bytes = 4001;
    for (int i = 0; i < 100; i++) CHECK(!posix_fadvise(fd, 95, 4001, POSIX_FADV_WILLNEED));
    CHECK(submitted == COMPAT_PREFETCH_JOBS);
    CHECK(compat_prefetch_pending == COMPAT_PREFETCH_JOBS);
    CHECK(reads == 0);                  /* no synchronous pre-read on the caller */
    CHECK(_lseeki64(fd, 0, SEEK_CUR) == sizeof data);
    int old = fd; _close(fd);
    fd = _open("test_prefetch_reused.tmp", _O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
    CHECK(fd == old);                   /* queued work must retain the original file */
    memset(data, 0x33, sizeof data); CHECK(_write(fd, data, sizeof data) == sizeof data);
    CHECK(drain()); CHECK(inspected == COMPAT_PREFETCH_JOBS);
    CHECK(remove(path) == 0);           /* no duplicate handle or mapping left open */

    DWORD handles_before, handles_after;
    CHECK(GetProcessHandleCount(GetCurrentProcess(), &handles_before));
    reject_submit = 1;
    for (int i = 0; i < 100; i++) CHECK(!posix_fadvise(fd, 0, 4096, POSIX_FADV_WILLNEED));
    reject_submit = 0; reject_duplicate = 1;
    for (int i = 0; i < 100; i++) CHECK(!posix_fadvise(fd, 0, 4096, POSIX_FADV_WILLNEED));
    reject_duplicate = 0;
    CHECK(!compat_prefetch_pending && submitted == COMPAT_PREFETCH_JOBS);
    CHECK(GetProcessHandleCount(GetCurrentProcess(), &handles_after));
    CHECK(handles_after <= handles_before);

    /* >4 GiB, unaligned offset and a hint crossing EOF. The sparse file avoids
     * allocating its hole on disk; only the final page is written. */
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    DWORD got;
    CHECK(DeviceIoControl(h, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &got, NULL));
    off_t offset = ((off_t)1 << 32) + 8192 + 19;
    CHECK(_lseeki64(fd, offset, SEEK_SET) == offset);
    CHECK(_write(fd, data, sizeof data) == sizeof data);
    expected_value = 0x33; expected_bytes = sizeof data;
    CHECK(!posix_fadvise(fd, offset, 65536, POSIX_FADV_WILLNEED)); CHECK(drain());
    CHECK(inspected == COMPAT_PREFETCH_JOBS + 1);

    /* Capped window, missing API and empty/invalid ranges are only hints. */
    expected_bytes = COMPAT_PREFETCH_BYTES; expected_value = 0x33;
    CHECK(!posix_fadvise(fd, 0, offset, POSIX_FADV_WILLNEED)); CHECK(drain());
    LONG before = inspected;
    missing_api = 1;
    CHECK(!posix_fadvise(fd, 0, 4096, POSIX_FADV_WILLNEED)); CHECK(drain());
    missing_api = 0;
    CHECK(!posix_fadvise(fd, offset + 65536, 4096, POSIX_FADV_WILLNEED)); CHECK(drain());
    CHECK(inspected == before);
    LONG jobs = submitted;
    CHECK(!posix_fadvise(fd, 0, 4096, POSIX_FADV_DONTNEED));
    CHECK(!posix_fadvise(fd, -1, 4096, POSIX_FADV_WILLNEED));
    CHECK(!posix_fadvise(-1, 0, 4096, POSIX_FADV_WILLNEED));
    CHECK(!posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED));
    CHECK(submitted == jobs && !reads);
    _close(fd);
    /* The helper lives beside this executable, independent of the test CWD. */
    char library[MAX_PATH];
    DWORD length = GetModuleFileNameA(NULL, library, sizeof library);
    CHECK(length && length < sizeof library);
    char *slash = strrchr(library, '\\'); CHECK(slash);
    CHECK((size_t)(slash - library) + sizeof("\\compat_prefetch_module.dll") < sizeof library);
    strcpy(slash + 1, "compat_prefetch_module.dll");
    HANDLE entered = CreateEventW(NULL, TRUE, FALSE, NULL); CHECK(entered);
    ResetEvent(gate);
    HMODULE module = LoadLibraryA(library); CHECK(module);
    typedef int (*StartPrefetch)(const char *, HANDLE, HANDLE);
    StartPrefetch start = (StartPrefetch)(void *)GetProcAddress(module, "start_prefetch");
    CHECK(start && start("test_prefetch_reused.tmp", gate, entered));
    CHECK(WaitForSingleObject(entered, 10000) == WAIT_OBJECT_0);
    CHECK(FreeLibrary(module));
    CHECK(GetModuleHandleA("compat_prefetch_module.dll"));
    SetEvent(gate);
    ULONGLONG end = GetTickCount64() + 10000;
    while (GetModuleHandleA("compat_prefetch_module.dll") && GetTickCount64() < end) Sleep(1);
    CHECK(!GetModuleHandleA("compat_prefetch_module.dll"));
    CloseHandle(entered);
    CHECK(!remove("test_prefetch_reused.tmp"));
    CloseHandle(gate);
    puts("compat prefetch: PASS (async, bounded, fd reuse, >4 GiB, EOF, failures, DLL lifetime)");
    return 0;
}
#endif
