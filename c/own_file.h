/* own_file.h -- opening the files the engine keeps beside a model.
 *
 * The engines write their own state into the model's directory: the KV cache
 * (.coli_kv), the expert history (.coli_usage), prefix checkpoints (.coli_ckpt/),
 * pinning tables. That directory usually comes from a download, so anything at
 * those names may have been planted by whoever packed the model. A symlink
 * named .coli_kv made the engine write its KV cache, whose bytes follow the
 * weights, to wherever the link pointed.
 *
 * coli_own_fopen() opens such a file only when the name is a regular file of its
 * own or does not exist yet (then it creates one): it never follows a symlink in
 * the last component, and refuses a FIFO, a device or a directory. "w" modes
 * truncate after those checks, never before. The modes are fopen's: "rb", "r+b",
 * "wb", "w", "w+b". A refused name fails like a failed fopen (NULL, errno set:
 * ELOOP for a link, EINVAL for anything else that is not a regular file).
 *
 * coli_own_is_link() says whether a path's last component is a symlink (or, on
 * Windows, any reparse point), for directories the engine creates and writes in.
 *
 * Windows has no O_NOFOLLOW: there the name is checked for a reparse point just
 * before the open. Creating a symlink there needs a privilege or developer
 * mode, and archive tools leave them out unless asked. */
#ifndef OWN_FILE_H
#define OWN_FILE_H

#include <errno.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static inline int coli_own_is_link(const char *path){
    DWORD a = GetFileAttributesA(path);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_REPARSE_POINT);
}

static inline FILE *coli_own_fopen(const char *path, const char *mode){
    if(mode[0] != 'r' && mode[0] != 'w'){ errno = EINVAL; return NULL; }   /* the modes the POSIX side takes */
    DWORD a = GetFileAttributesA(path);
    if(a != INVALID_FILE_ATTRIBUTES){
        if(a & FILE_ATTRIBUTE_REPARSE_POINT){ errno = ELOOP; return NULL; }
        if(a & FILE_ATTRIBUTE_DIRECTORY){ errno = EINVAL; return NULL; }
    }
    return fopen(path, mode);
}
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

static inline int coli_own_is_link(const char *path){
    struct stat st;
    return lstat(path, &st) == 0 && S_ISLNK(st.st_mode);
}

static inline FILE *coli_own_fopen(const char *path, const char *mode){
    int rw = strchr(mode, '+') != NULL, fl;
    if(mode[0] == 'r')      fl = rw ? O_RDWR : O_RDONLY;
    else if(mode[0] == 'w') fl = (rw ? O_RDWR : O_WRONLY) | O_CREAT;
    else { errno = EINVAL; return NULL; }
#ifdef O_CLOEXEC
    fl |= O_CLOEXEC;
#endif
    /* O_NONBLOCK so a planted FIFO cannot hang the open; cleared once the
     * descriptor is known to be a regular file. */
    int fd = open(path, fl | O_NOFOLLOW | O_NONBLOCK, 0666);
    if(fd < 0) return NULL;
    struct stat st;
    if(fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)){ close(fd); errno = EINVAL; return NULL; }
    int cur = fcntl(fd, F_GETFL);
    if(cur != -1) fcntl(fd, F_SETFL, cur & ~O_NONBLOCK);
    if(mode[0] == 'w' && ftruncate(fd, 0) != 0){ int e = errno; close(fd); errno = e; return NULL; }
    FILE *f = fdopen(fd, mode);
    if(!f){ int e = errno; close(fd); errno = e; }
    return f;
}
#endif

#endif /* OWN_FILE_H */
