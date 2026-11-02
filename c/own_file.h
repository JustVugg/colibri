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
 * coli_own_fopen_tmp() and coli_own_replace() write a state file as a whole: the
 * first opens "<path>.tmp" for writing, the second renames it over <path> once
 * the caller has written and closed it. A program reading <path> gets the old
 * file or the new one, never half of it. A link or another file under the temp
 * name is removed and the open is tried once more.
 *
 * coli_own_same_path() says whether two paths are one file, also for a file that
 * does not exist yet.
 *
 * Windows has no O_NOFOLLOW: there the name is checked for a reparse point just
 * before the open. Creating a symlink there needs a privilege or developer
 * mode, and archive tools leave them out unless asked. */
#ifndef OWN_FILE_H
#define OWN_FILE_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compat.h"      /* rename() replaces an existing file on Windows */

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

/* The absolute path of `path` with links resolved, in allocated memory. A file that does not
 * exist yet is resolved through its directory. NULL when that fails too. */
static inline char *coli_own_resolve(const char *path){
#ifdef _WIN32
    return _fullpath(NULL, path, 0);
#else
    char *full = realpath(path, NULL);
    if(full || errno != ENOENT) return full;
    const char *slash = strrchr(path, '/'), *name = slash ? slash + 1 : path;
    char *dir = slash ? strndup(path, slash == path ? 1 : (size_t)(slash - path)) : strdup(".");
    char *real = dir ? realpath(dir, NULL) : NULL;
    free(dir);
    if(!real) return NULL;
    char *joined = (char *)malloc(strlen(real) + strlen(name) + 2);
    if(joined) sprintf(joined, "%s/%s", real, name);
    free(real);
    return joined;
#endif
}

/* 1 when both paths are one file. Paths that cannot be resolved are compared as written. */
static inline int coli_own_same_path(const char *a, const char *b){
    char *ra = coli_own_resolve(a), *rb = coli_own_resolve(b);
    int same = ra && rb ? strcmp(ra, rb) == 0 : strcmp(a, b) == 0;
    free(ra);
    free(rb);
    return same;
}

/* "<path>.tmp" in allocated memory; NULL with errno set when there is none. */
static inline char *coli_own_tmp_name(const char *path){
    size_t n = strlen(path);
    char *tmp = (char *)malloc(n + sizeof ".tmp");
    if(!tmp){ errno = ENOMEM; return NULL; }
    memcpy(tmp, path, n);
    memcpy(tmp + n, ".tmp", sizeof ".tmp");
    return tmp;
}

/* Opens "<path>.tmp" with fopen's mode "w". NULL with errno set on failure. */
static inline FILE *coli_own_fopen_tmp(const char *path){
    char *tmp = coli_own_tmp_name(path);
    if(!tmp) return NULL;
    FILE *f = coli_own_fopen(tmp, "w");
    if(!f && (errno == ELOOP || errno == EINVAL) && remove(tmp) == 0) f = coli_own_fopen(tmp, "w");
    int e = errno;
    free(tmp);
    errno = e;
    return f;
}

/* Renames the closed "<path>.tmp" over <path>. 0 on success, -1 with errno set on failure. */
static inline int coli_own_replace(const char *path){
    char *tmp = coli_own_tmp_name(path);
    if(!tmp) return -1;
    int rc = rename(tmp, path);
    int e = errno;
    free(tmp);
    errno = e;
    return rc;
}

#endif /* OWN_FILE_H */
