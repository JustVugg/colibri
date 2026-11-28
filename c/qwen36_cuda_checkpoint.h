/* Converted Qwen3.6 checkpoint qualified for CUDA profiles.
 * Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64, revision c619aa594ad1e70af82168fb6b4878427896e21c. */
#ifndef QWEN36_CUDA_CHECKPOINT_H
#define QWEN36_CUDA_CHECKPOINT_H

#include "evidence_digest.h"
#include <inttypes.h>
#include <time.h>
#ifdef _WIN32
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/vfs.h>
#elif defined(__APPLE__)
#include <sys/mount.h>
#endif

typedef struct {
    const char *name;
    int64_t bytes;
    const char *sha256;
} Q36CheckpointFile;

enum { Q36_CHECKPOINT_SHARDS = 41 };
static const Q36CheckpointFile q36_checkpoint_files[] = {
    {"model-00000.safetensors", 527831896, "96c003f9a1e2d0b9ee8211ea70c3f0ae3e22a8056b5e5b6c74cec672498a3350"},
    {"model-00001.safetensors", 527831896, "3015f861eb6410d5be254677406e9fd6c6a4112e367216207ed0608d1672ff4f"},
    {"model-00002.safetensors", 527831896, "e43270ea0473a38a2d519ea672f95c4dc86cf6a06958e98afaac06c1714070d5"},
    {"model-00003.safetensors", 514921584, "5503548569908203386969198610b45d75bbb2ae894833d3dfadf61f84ebdfc7"},
    {"model-00004.safetensors", 527831896, "5730b65eb434d4988f2a67f9d85688c1ba5749e74f95852b33901ee00b41b3e6"},
    {"model-00005.safetensors", 527831896, "4a2f9aaaffbdd2f6590e1ec98f2fbd381752e414b6044af6a23557d9ce622bc4"},
    {"model-00006.safetensors", 527831896, "a4d6da971c8627a4fb6b4929869bae60d9407879794583737c38f7e421244117"},
    {"model-00007.safetensors", 514921584, "fc64d0d18127e855e4193c48941e109e3596a2abca1b5f840b351a284a485f2c"},
    {"model-00008.safetensors", 527831896, "5fc1ad29276bc97f866f97b537674853bbba948991a671c3df88ef00daa011d0"},
    {"model-00009.safetensors", 527831896, "a54d8bad1e2e7fab6fd09b9c68fd6a616a8322433e33c023cedc2f6d61d05eda"},
    {"model-00010.safetensors", 527832424, "36fd1512ab7bef053c8960a2326ee4948a417400fb6c88ada5c672a3cf701a67"},
    {"model-00011.safetensors", 514922112, "fb4e06f18f3373a937444ecf72f340af2d184162ec3911adced68fd45fdc5326"},
    {"model-00012.safetensors", 527832424, "c9c88522e672b305fbf237e72583fd0967445c6b00d5af1d9d1a08a495f6f1db"},
    {"model-00013.safetensors", 527832424, "80e6dc06289531eaf98a7f98b14f290b068e05eae45670a8055b8cb8cd1331af"},
    {"model-00014.safetensors", 527832424, "9cc49310e4a11420124626e3b3b2654cb0d5e0a65d28780d4ffb222d76c15718"},
    {"model-00015.safetensors", 514922112, "8c6a2af8e8ace8e3b363c5399bdb6a39b40b78a20c8ff3d071b43ce92a60f4a8"},
    {"model-00016.safetensors", 527832424, "b44be5832ecb09b45be7831fddea95709cc9ff676749c26d82ea8ed0bed85627"},
    {"model-00017.safetensors", 527832424, "2871020915c72fe774f6fdfbcb1aa566b0680613b0a80cacf6ececef0adfeb44"},
    {"model-00018.safetensors", 527832424, "85bf77269c27687a590058c8965dea55928edc012f403f3c96343b7bc023b302"},
    {"model-00019.safetensors", 514922112, "f0b3353c1ac570833ce4050f6b44201fdda22d1582d24567b888292d28243e87"},
    {"model-00020.safetensors", 527832424, "39c9309115cd1b469314d2ef40ae4ff54c378c88d008c32009976a7c9ae0c66d"},
    {"model-00021.safetensors", 527832424, "930a367959cc36a84c74e0a9c35288a77a4c358139e86b456d251ce70b05f44c"},
    {"model-00022.safetensors", 527832424, "d1bc8932eab6a9dfa2415c07a9aff0002389a234ea4c968c861b1fd257b021c6"},
    {"model-00023.safetensors", 514922112, "3a39595d18c27dbf1394c3171dc043b7f2c2065d16fcb5e979e305b5f2125991"},
    {"model-00024.safetensors", 527832424, "2a1fb7365f6578022442b6d0fee07dd2c9697faf9a944d51bc170b2cbd52b06f"},
    {"model-00025.safetensors", 527832424, "4665ab8badf23317866b938f2fb5f8f86518ee99ae4b3fbc3bd61b5826adf71f"},
    {"model-00026.safetensors", 527832424, "c4d0804f358ee2393c8357ab25e9a6a365c9cee4a7050047678bd48a535a84f8"},
    {"model-00027.safetensors", 514922112, "35203a57b60e783aea8083a5f47443dbacd833fabd86a1a2af355d2d1838752a"},
    {"model-00028.safetensors", 527832424, "a0fe8c94723c3a8d10f69ddc67d15da1282f707beba6ac383f62c0b91cbe1864"},
    {"model-00029.safetensors", 527832424, "b9a2468b56aac02f6ef90d7e18303038c305598220c8d0d09d3afdfab2e571fc"},
    {"model-00030.safetensors", 527832424, "33e9e46d1a005e31e00f06a95d4bd108ead8723a3d11877b6f1fe8cc94fe3f4f"},
    {"model-00031.safetensors", 514922112, "2ef86ddb9c2c8ed4428304e21960f26c7c9e002ca430ad25b66c0e93bf5faa1c"},
    {"model-00032.safetensors", 527832424, "14caf9cdda5eb5ae755355e595435529bc8152e5e8f6bc28ed53f081004f99e6"},
    {"model-00033.safetensors", 527832424, "2f8415c2f03659aadcaac1554b15fbdd3e84f533fca7daf735c3d3e74bc16140"},
    {"model-00034.safetensors", 527832424, "664706807a61f4f926ac9aa37dc9597f90fb2555d5d5d8c1511e2b97e9da2859"},
    {"model-00035.safetensors", 514922112, "fba2d5d1354c79036d81d6eacdbe82a90ee4a207dccd2ffdd5b191691d2314e9"},
    {"model-00036.safetensors", 527832424, "acc9d8df72095d8bbec9ccd3c29e07b07bf064813767dcb482c963e62ea067e5"},
    {"model-00037.safetensors", 527832424, "05c3010ea712109b2fb01cfabf2c93fed825bf9e0d0ac95f6b528af6b0509666"},
    {"model-00038.safetensors", 527832424, "049206b4a32d35883b3f24a1db58fcc4bdd6c21246580090dd6b7d88ea68e060"},
    {"model-00039.safetensors", 514922112, "b516d6817f490658c9c178c3cc3ca833708d78253526ed51883fa4fff8cb7662"},
    {"model-globals.safetensors", 2034241832, "b00b11ff7de0072b5a3e08c04b7d8bbfe065ae2d1a299de6e0a2ae6d9896bb23"},
    {"config.json", 2081, "7606939241b64e82308369b308af8abe85d24e70846704c41ed4e0b59687a674"},
    {"tokenizer.json", 12807982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"},
    {"qwen36_meta.json", 1702, "cb2d7d4054865c7d2d6f6da3c902b51d9942e5f19b53e646b23909cdd4ce0fc4"},
};

typedef struct {
    const char *snap;
    const Q36CheckpointFile *files;
    int nshards, nfiles;
} Q36Checkpoint;

enum { Q36_CHECKPOINT_NFILES = sizeof q36_checkpoint_files / sizeof *q36_checkpoint_files };
#define Q36_CHECKPOINT_CACHE ".colibri-qwen36-cuda-checkpoint-v1"
#define Q36_CHECKPOINT_CACHE_HEADER "qwen36 checkpoint v1\n"

static Q36Checkpoint q36_qualified_checkpoint(const char *snap) {
    return (Q36Checkpoint){snap, q36_checkpoint_files, Q36_CHECKPOINT_SHARDS, Q36_CHECKPOINT_NFILES};
}

static long q36_checkpoint_mtime_ns(const struct stat *st) {
#if defined(__APPLE__)
    return st->st_mtimespec.tv_nsec;
#elif defined(__linux__)
    return st->st_mtim.tv_nsec;
#else
    (void)st;
    return 0;
#endif
}

static long q36_checkpoint_ctime_ns(const struct stat *st) {
#if defined(__APPLE__)
    return st->st_ctimespec.tv_nsec;
#elif defined(__linux__)
    return st->st_ctim.tv_nsec;
#else
    (void)st;
    return 0;
#endif
}

#ifdef _WIN32
static int q36_checkpoint_fd_info(int fd, BY_HANDLE_FILE_INFORMATION *out) {
    intptr_t osf = _get_osfhandle(fd);
    if (osf == -1) return 0;
    return GetFileInformationByHandle((HANDLE)osf, out) &&
        !(out->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
}

static int q36_checkpoint_path_info(const char *path, BY_HANDLE_FILE_INFORMATION *out) {
    HANDLE h = CreateFileA(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    BOOL ok = GetFileInformationByHandle(h, out);
    CloseHandle(h);
    return ok && !(out->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
}

static int q36_checkpoint_same_win_info(const BY_HANDLE_FILE_INFORMATION *a,
                                        const BY_HANDLE_FILE_INFORMATION *b) {
    return a->dwVolumeSerialNumber == b->dwVolumeSerialNumber &&
        a->nFileIndexHigh == b->nFileIndexHigh && a->nFileIndexLow == b->nFileIndexLow &&
        a->nFileSizeHigh == b->nFileSizeHigh && a->nFileSizeLow == b->nFileSizeLow &&
        a->ftLastWriteTime.dwHighDateTime == b->ftLastWriteTime.dwHighDateTime &&
        a->ftLastWriteTime.dwLowDateTime == b->ftLastWriteTime.dwLowDateTime &&
        a->ftCreationTime.dwHighDateTime == b->ftCreationTime.dwHighDateTime &&
        a->ftCreationTime.dwLowDateTime == b->ftCreationTime.dwLowDateTime;
}
#else
static int q36_checkpoint_same_file(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino && a->st_size == b->st_size &&
        a->st_mtime == b->st_mtime && q36_checkpoint_mtime_ns(a) == q36_checkpoint_mtime_ns(b) &&
        a->st_ctime == b->st_ctime && q36_checkpoint_ctime_ns(a) == q36_checkpoint_ctime_ns(b);
}
#endif

/* Only local filesystems with stable inode numbers and subsecond change times
 * may reuse a digest. The same content check works on any other filesystem. */
static int q36_checkpoint_cache_fs(int fd) {
#if defined(__linux__)
    struct statfs fs;
    if (fstatfs(fd, &fs) != 0) return 0;
    switch ((unsigned)fs.f_type) {
        case 0xEF53u:     /* ext2/3/4 */
        case 0x58465342u: /* XFS */
        case 0x9123683Eu: /* Btrfs */
        case 0x01021994u: /* tmpfs */
        case 0x794c7630u: /* overlayfs */
        case 0x2fc12fc1u: /* ZFS */
            return 1;
    }
#elif defined(__APPLE__)
    struct statfs fs;
    return fstatfs(fd, &fs) == 0 && !strcmp(fs.f_fstypename, "apfs");
#else
    (void)fd;
#endif
    return 0;
}

static int q36_checkpoint_path(char *dst, size_t cap, const char *snap, const char *name) {
    int n = snprintf(dst, cap, "%s/%s", snap, name);
    return n >= 0 && (size_t)n < cap;
}

static int q36_checkpoint_open(const char *path) {
#if defined(__linux__) || defined(__APPLE__)
    return open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
#else
    return open(path, COMPAT_O_RDONLY);
#endif
}

static int q36_checkpoint_digest(int fd, const Q36CheckpointFile *file) {
    EvidenceSha256 state;
    unsigned char chunk[64 * 1024];
    unsigned char raw[32];
    static const char hex[] = "0123456789abcdef";
    evidence_sha256_init(&state);
    for (int64_t pos = 0; pos < file->bytes;) {
        int64_t n = file->bytes - pos;
        if (n > (int64_t)sizeof chunk) n = sizeof chunk;
        if (st_pread_exact(fd, chunk, n, pos) != 0) return 0;
        evidence_sha256_update(&state, chunk, (size_t)n);
        pos += n;
    }
    evidence_sha256_final(&state, raw);
    for (size_t i = 0; i < sizeof raw; i++)
        if (file->sha256[2 * i] != hex[raw[i] >> 4] || file->sha256[2 * i + 1] != hex[raw[i] & 15])
            return 0;
    return 1;
}

enum { Q36_CHECKPOINT_LINE = 320 };
typedef char Q36CheckpointCache[Q36_CHECKPOINT_NFILES][Q36_CHECKPOINT_LINE];

static void q36_checkpoint_cache_read(const char *path, Q36CheckpointCache lines, int nfiles) {
#if defined(__linux__) || defined(__APPLE__)
    int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    struct stat st;
    if (fd < 0) return;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != geteuid() ||
        (st.st_mode & 022) || !q36_checkpoint_cache_fs(fd)) { close(fd); return; }
    FILE *f = fdopen(fd, "r");
    if (!f) { close(fd); return; }
    char header[sizeof Q36_CHECKPOINT_CACHE_HEADER];
    if (fgets(header, sizeof header, f) && !strcmp(header, Q36_CHECKPOINT_CACHE_HEADER)) {
        for (int i = 0; i < nfiles; i++) {
            if (!fgets(lines[i], Q36_CHECKPOINT_LINE, f) || !strchr(lines[i], '\n')) {
                memset(lines, 0, (size_t)nfiles * Q36_CHECKPOINT_LINE);
                break;
            }
        }
    }
    fclose(f);
#else
    (void)path; (void)lines; (void)nfiles;
#endif
}

static void q36_checkpoint_cache_write(const char *path, Q36CheckpointCache lines, int nfiles) {
#if defined(__linux__) || defined(__APPLE__)
    char temp[2100];
    int n = snprintf(temp, sizeof temp, "%s.XXXXXX", path);
    if (n < 0 || (size_t)n >= sizeof temp) return;
    int fd = mkstemp(temp);
    if (fd < 0) return;
    FILE *f = fdopen(fd, "w");
    if (!f) { close(fd); unlink(temp); return; }
    int ok = fputs(Q36_CHECKPOINT_CACHE_HEADER, f) >= 0;
    for (int i = 0; ok && i < nfiles; i++) ok = fputs(lines[i], f) >= 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok || rename(temp, path) != 0) unlink(temp);
#else
    (void)path; (void)lines; (void)nfiles;
#endif
}

static int q36_checkpoint_stamp(char line[Q36_CHECKPOINT_LINE], const Q36CheckpointFile *file,
                                int fd, const struct stat *st) {
    /* Wait past the filesystem clock tick before making a stamp reusable.
     * A write within the same tick can have the same timestamps. */
    time_t now = time(NULL);
    if (!st->st_ino || st->st_ctime >= now - 1 || st->st_mtime >= now - 1 || !q36_checkpoint_cache_fs(fd))
        return 0;
    int n = snprintf(line, Q36_CHECKPOINT_LINE,
                     "%s %s %" PRId64 " %ju %ju %jd %ld %jd %ld\n",
                     file->name, file->sha256, file->bytes, (uintmax_t)st->st_dev, (uintmax_t)st->st_ino,
                     (intmax_t)st->st_mtime, q36_checkpoint_mtime_ns(st),
                     (intmax_t)st->st_ctime, q36_checkpoint_ctime_ns(st));
    return n >= 0 && n < Q36_CHECKPOINT_LINE;
}

static int q36_checkpoint_matches(const shards *s, const Q36Checkpoint *checkpoint) {
    if (!checkpoint || !checkpoint->snap || s->nrep || checkpoint->nshards <= 0 || checkpoint->nshards != s->nfd ||
        checkpoint->nfiles <= checkpoint->nshards || checkpoint->nfiles > Q36_CHECKPOINT_NFILES)
        return 0;
    const int nshards = checkpoint->nshards, nfiles = checkpoint->nfiles;
    struct stat before[Q36_CHECKPOINT_NFILES];
    int fds[Q36_CHECKPOINT_NFILES];
    for (int i = nshards; i < nfiles; i++) fds[i] = -1;
    char path[2048], cache[2048];
    int ok = 0;
#ifdef _WIN32
    BY_HANDLE_FILE_INFORMATION win_before[Q36_CHECKPOINT_NFILES];
#endif
    for (int i = 0; i < nfiles; i++) {
        const Q36CheckpointFile *file = &checkpoint->files[i];
        if (!q36_checkpoint_path(path, sizeof path, checkpoint->snap, file->name)) goto done;
        if (i < nshards) {
            if (!s->paths[i] || strcmp(s->paths[i], path) || s->sizes[i] != file->bytes) goto done;
            fds[i] = s->fds[i];
        } else {
            fds[i] = q36_checkpoint_open(path);
        }
        if (fds[i] < 0 || fstat(fds[i], &before[i]) != 0 || !S_ISREG(before[i].st_mode) ||
            before[i].st_size != file->bytes) goto done;
#ifdef _WIN32
        if (!q36_checkpoint_fd_info(fds[i], &win_before[i])) goto done;
#endif
    }

    Q36CheckpointCache old = {{0}}, fresh;
    const int cache_path = q36_checkpoint_path(cache, sizeof cache, checkpoint->snap, Q36_CHECKPOINT_CACHE);
    if (cache_path) q36_checkpoint_cache_read(cache, old, nfiles);
    int changed = 0, cacheable = 0, announced = 0;
    /* Reject different configurations before reading any of the large shards. */
    for (int j = 0; j < nfiles; j++) {
        const int i = (nshards + j) % nfiles;
        const Q36CheckpointFile *file = &checkpoint->files[i];
        int stamp = q36_checkpoint_stamp(fresh[i], file, fds[i], &before[i]);
        if (stamp && !strcmp(old[i], fresh[i])) { cacheable = 1; continue; }
        if (i < nshards && !announced) {
            fprintf(stderr, "[chain] qwen36: verifying the CUDA profile's checkpoint weights\n");
            announced = 1;
        }
        if (!q36_checkpoint_digest(fds[i], file)) goto done;
        if (!stamp) stamp = q36_checkpoint_stamp(fresh[i], file, fds[i], &before[i]);
        if (!stamp) strcpy(fresh[i], "-\n");
        cacheable |= stamp;
        changed |= strcmp(old[i], fresh[i]) != 0;
    }
    for (int i = 0; i < nfiles; i++) {
        if (!q36_checkpoint_path(path, sizeof path, checkpoint->snap, checkpoint->files[i].name)) goto done;
#ifdef _WIN32
        BY_HANDLE_FILE_INFORMATION opened, named;
        if (!q36_checkpoint_fd_info(fds[i], &opened) || !q36_checkpoint_same_win_info(&win_before[i], &opened) ||
            !q36_checkpoint_path_info(path, &named) || !q36_checkpoint_same_win_info(&opened, &named)) goto done;
#else
        struct stat opened, named;
        if (fstat(fds[i], &opened) != 0 || !q36_checkpoint_same_file(&before[i], &opened) ||
            stat(path, &named) != 0 || !q36_checkpoint_same_file(&opened, &named)) goto done;
#endif
    }
    if (cache_path && cacheable && changed) q36_checkpoint_cache_write(cache, fresh, nfiles);
    ok = 1;
done:
    for (int i = nshards; i < nfiles; i++) if (fds[i] >= 0) close(fds[i]);
    return ok;
}

#endif
