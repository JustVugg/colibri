/* Loaded and released while an advisory worker is running: an SDK caller
 * must not be able to unload the callback's code before it returns. */
#ifdef _WIN32
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
static HANDLE release_gate, entered_gate;
static FARPROC WINAPI delayed_get_proc(HMODULE module, LPCSTR name) {
    if (!strcmp(name, "PrefetchVirtualMemory")) {
        SetEvent(entered_gate);
        if (WaitForSingleObject(release_gate, 10000) != WAIT_OBJECT_0) return NULL;
    }
    return GetProcAddress(module, name);
}
#define GetProcAddress delayed_get_proc
#include "../compat.h"
#undef GetProcAddress
__declspec(dllexport) int start_prefetch(const char *path, HANDLE release, HANDLE entered) {
    release_gate = release; entered_gate = entered;
    int fd = _open(path, _O_RDONLY | _O_BINARY);
    if (fd < 0) return 0;
    compat_fadvise(fd, 0, 4096, POSIX_FADV_WILLNEED);
    _close(fd);
    return compat_prefetch_pending > 0;
}
#endif
