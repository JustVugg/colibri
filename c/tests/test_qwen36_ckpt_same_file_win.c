/* Unit test for Windows GetFileInformationByHandle same-file detection
 * in qwen36_cuda_checkpoint.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "../st.h"
#include "../qwen36_cuda_checkpoint.h"

static int checks = 0;
static int failures = 0;

static void ck(int cond, const char *what) {
    checks++;
    if (!cond) {
        failures++;
        fprintf(stderr, "FAIL: %s\n", what);
    }
}

#ifdef _WIN32

static int write_exact_file(const char *path, const char *data) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    size_t n = strlen(data);
    int ok = fwrite(data, 1, n, f) == n;
    return fclose(f) == 0 && ok;
}

static int set_fixed_times(const char *path, const FILETIME *ft) {
    HANDLE h = CreateFileA(path, FILE_WRITE_ATTRIBUTES,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    BOOL ok = SetFileTime(h, ft, ft, ft);
    CloseHandle(h);
    return ok != 0;
}

int main(void) {
    char tmp_root[MAX_PATH];
    DWORD len = GetTempPathA((DWORD)sizeof tmp_root, tmp_root);
    ck(len > 0 && len < sizeof tmp_root, "GetTempPathA succeeded");
    if (!len || len >= sizeof tmp_root) return 1;

    char dir[512];
    snprintf(dir, sizeof dir, "%scolibri_q36_same_file_%lu", tmp_root, (unsigned long)GetCurrentProcessId());
    RemoveDirectoryA(dir);
    ck(CreateDirectoryA(dir, NULL) != 0, "create temp test directory");

    char path_a[1024], path_b[1024], path_link[1024], path_cfg[1024];
    snprintf(path_a, sizeof path_a, "%s/model-00000.safetensors", dir);
    snprintf(path_b, sizeof path_b, "%s/other-00000.safetensors", dir);
    snprintf(path_link, sizeof path_link, "%s/hardlink-00000.safetensors", dir);
    snprintf(path_cfg, sizeof path_cfg, "%s/config.json", dir);

    DeleteFileA(path_a);
    DeleteFileA(path_b);
    DeleteFileA(path_link);
    DeleteFileA(path_cfg);

    /* Create two distinct files with identical content, size, and timestamps. */
    ck(write_exact_file(path_a, "abc"), "write file A");
    ck(write_exact_file(path_b, "abc"), "write file B");
    ck(write_exact_file(path_cfg, "{}"), "write config.json");

    /* 2025-01-01 00:00:00 UTC in 100ns ticks since 1601-01-01. */
    const uint64_t ticks = 133801632000000000ULL;
    FILETIME ft;
    ft.dwLowDateTime = (DWORD)(ticks & 0xFFFFFFFFu);
    ft.dwHighDateTime = (DWORD)(ticks >> 32);
    ck(set_fixed_times(path_a, &ft), "set fixed timestamps on file A");
    ck(set_fixed_times(path_b, &ft), "set fixed timestamps on file B");

    BY_HANDLE_FILE_INFORMATION info_a = {0}, info_b = {0}, info_dir = {0};
    ck(q36_checkpoint_path_info(path_a, &info_a), "path_info succeeds on file A");
    ck(q36_checkpoint_path_info(path_b, &info_b), "path_info succeeds on file B");
    ck(!q36_checkpoint_path_info(dir, &info_dir), "path_info rejects directory");

    /* Confirm sizes and timestamps truly match, so only volume+index distinguishes them. */
    ck(info_a.nFileSizeHigh == info_b.nFileSizeHigh && info_a.nFileSizeLow == info_b.nFileSizeLow,
       "distinct files have identical sizes");
    ck(info_a.ftLastWriteTime.dwHighDateTime == info_b.ftLastWriteTime.dwHighDateTime &&
       info_a.ftLastWriteTime.dwLowDateTime == info_b.ftLastWriteTime.dwLowDateTime &&
       info_a.ftCreationTime.dwHighDateTime == info_b.ftCreationTime.dwHighDateTime &&
       info_a.ftCreationTime.dwLowDateTime == info_b.ftCreationTime.dwLowDateTime,
       "distinct files have identical creation and write timestamps");
    ck(!q36_checkpoint_same_win_info(&info_a, &info_b),
       "two distinct files with identical size and timestamps are NOT treated as the same file");

    /* Two separate open handles to the same file ARE treated as the same file. */
    int fd1 = open(path_a, COMPAT_O_RDONLY);
    int fd2 = open(path_a, COMPAT_O_RDONLY);
    int fd_b = open(path_b, COMPAT_O_RDONLY);
    ck(fd1 >= 0 && fd2 >= 0 && fd_b >= 0, "open handles to file A (x2) and file B");

    BY_HANDLE_FILE_INFORMATION info_fd1 = {0}, info_fd2 = {0}, info_fdb = {0};
    ck(q36_checkpoint_fd_info(fd1, &info_fd1), "fd_info succeeds on fd1");
    ck(q36_checkpoint_fd_info(fd2, &info_fd2), "fd_info succeeds on fd2");
    ck(q36_checkpoint_fd_info(fd_b, &info_fdb), "fd_info succeeds on fd_b");
    ck(!q36_checkpoint_fd_info(-1, &info_dir), "fd_info rejects invalid fd");

    ck(q36_checkpoint_same_win_info(&info_fd1, &info_fd2),
       "two handles to the same file ARE treated as the same file");
    ck(q36_checkpoint_same_win_info(&info_fd1, &info_a),
       "handle info matches path info for the same file");
    ck(!q36_checkpoint_same_win_info(&info_fd1, &info_fdb),
       "handles to distinct files with identical size and timestamps are NOT the same file");

    /* Hardlink to file A must be treated as the same file as file A. */
    if (CreateHardLinkA(path_link, path_a, NULL)) {
        BY_HANDLE_FILE_INFORMATION info_link = {0};
        ck(q36_checkpoint_path_info(path_link, &info_link), "path_info succeeds on hardlink");
        ck(q36_checkpoint_same_win_info(&info_a, &info_link),
           "hardlink and original file ARE treated as the same file");
        ck(q36_checkpoint_same_win_info(&info_fd1, &info_link),
           "open handle and hardlink to the same file ARE treated as the same file");
        ck(!q36_checkpoint_same_win_info(&info_b, &info_link),
           "hardlink to file A is NOT treated as the same file as distinct file B");
    } else {
        fprintf(stderr, "NOTE: CreateHardLinkA unsupported on temp volume (err=%lu), skipping hardlink check\n",
                (unsigned long)GetLastError());
    }

    /* End-to-end q36_checkpoint_matches check:
     * fd1 points to path_a -> must PASS.
     * fd_b points to distinct file B with identical bytes/size/timestamps while path claims path_a -> must FAIL. */
    const Q36CheckpointFile files[] = {
        {"model-00000.safetensors", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"config.json", 2, "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"},
    };
    const Q36Checkpoint ckpt = {dir, files, 1, 2};
    shards s_same = {.nfd = 1, .fds = {fd1}, .paths = {path_a}, .sizes = {3}};
    shards s_diff = {.nfd = 1, .fds = {fd_b}, .paths = {path_a}, .sizes = {3}};

    ck(q36_checkpoint_matches(&s_same, &ckpt),
       "q36_checkpoint_matches accepts fd opened on the named checkpoint file");
    ck(!q36_checkpoint_matches(&s_diff, &ckpt),
       "q36_checkpoint_matches rejects fd opened on a distinct file with identical bytes, size, and timestamps");

    if (fd1 >= 0) close(fd1);
    if (fd2 >= 0) close(fd2);
    if (fd_b >= 0) close(fd_b);
    DeleteFileA(path_link);
    DeleteFileA(path_a);
    DeleteFileA(path_b);
    DeleteFileA(path_cfg);
    RemoveDirectoryA(dir);

    printf("test_qwen36_ckpt_same_file_win: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}

#else

int main(void) {
    printf("test_qwen36_ckpt_same_file_win: skipped (non-Windows)\n");
    return 0;
}

#endif
