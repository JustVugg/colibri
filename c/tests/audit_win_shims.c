/* audit: verify compat_pread >4GB offset + compat_rename replace-existing (Windows) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../compat.h"
#include <winioctl.h>

int main(void){
    /* --- rename second-write test --- */
    FILE *f = fopen("aud_tmp1", "wb"); fputs("v1", f); fclose(f);
    if(rename("aud_tmp1", "aud_dst")){ printf("rename #1 FAIL\n"); return 1; }
    f = fopen("aud_tmp2", "wb"); fputs("v2", f); fclose(f);
    if(rename("aud_tmp2", "aud_dst")){ printf("rename #2 (dest exists) FAIL <- CRT EEXIST bug\n"); return 1; }
    char buf[8] = {0};
    f = fopen("aud_dst", "rb"); fread(buf, 1, 2, f); fclose(f);
    remove("aud_dst");
    if(strcmp(buf, "v2")){ printf("rename content FAIL (%s)\n", buf); return 1; }
    printf("rename replace-existing: ok\n");

    /* --- pread >4GB offset test (sparse file) --- */
    const char *p = "aud_big.bin";
    int fd = open(p, O_RDWR | O_CREAT | O_BINARY, 0644);
    if(fd < 0){ printf("open FAIL\n"); return 1; }
    /* mark sparse so 5GB costs ~nothing on NTFS */
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    DWORD br; DeviceIoControl(h, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &br, NULL);
    off_t off = (off_t)5 * 1024 * 1024 * 1024;  /* 5 GiB */
    const char magic[] = "COLIBRI64";
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(off & 0xFFFFFFFFULL); ov.OffsetHigh = (DWORD)(off >> 32);
    if(!WriteFile(h, magic, sizeof(magic), &br, &ov)){ printf("write@5GB FAIL\n"); close(fd); remove(p); return 1; }
    char rd[sizeof(magic)] = {0};
    ssize_t n = pread(fd, rd, sizeof(magic), off);
    close(fd); remove(p);
    if(n != sizeof(magic) || memcmp(rd, magic, sizeof(magic))){
        printf("pread >4GB FAIL (n=%lld data=%s) <- OVERLAPPED 64-bit fill broken\n", (long long)n, rd);
        return 1;
    }
    printf("pread >4GB offset: ok\n");
    return 0;
}
