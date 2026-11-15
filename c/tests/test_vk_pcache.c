/* The compute pipelines' cache on disk (vk_pcache.h, #2777) on whatever device the loader
 * gives (Lavapipe in CI): a first start writes the device's file, the next one reads it,
 * a file of another device or a cut one never reaches the driver and is written over, and
 * COLI_VK_PIPELINE_CACHE=0 keeps no file. The backend is included to see its cache table. */
#include "../backend_vulkan.c"
#include <assert.h>

static long file_len(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    long n = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : -1;
    fclose(f);
    return n;
}
static void header(const char *p, uint32_t h[4]) {
    FILE *f = fopen(p, "rb");
    assert(f && fread(h, 4, 4, f) == 4);
    fclose(f);
}
static void put(const char *p, const void *data, size_t n) {
    FILE *f = fopen(p, "wb");
    assert(f && fwrite(data, 1, n, f) == n && fclose(f) == 0);
}
/* a start and its teardown: the cache of the primary device, read from the file or not */
static int start(const char *spv, char *path, size_t n, VkPhysicalDeviceProperties *pp) {
    assert(coli_vk_init(spv));
    int i = pc_find(G.dev);
    assert(i >= 0);
    snprintf(path, n, "%s", coli_vk_pcache_path(G.dev));
    vkGetPhysicalDeviceProperties(G.phys, pp);
    int had = g_pc[i].had;
    /* the start made its pipelines through the cache, so the teardown writes it */
    assert(!path[0] || g_pc[i].dirty || file_len(path) >= 32);
    coli_vk_shutdown();
    for (int k = 0; k < PC_MAXDEV; k++) assert(!g_pc[k].dev);   /* closed with the device */
    return had;
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    const char *td = getenv("TMPDIR");
    char dir[900];
    snprintf(dir, sizeof dir, "%s/coli-vkpc-XXXXXX", td && *td ? td : "/tmp");
    assert(mkdtemp(dir));
    char sub[1100];
    snprintf(sub, sizeof sub, "%s/new", dir);   /* a directory not there yet: made */
    setenv("COLI_VK_PIPELINE_CACHE", sub, 1);
    char path[1024];
    VkPhysicalDeviceProperties pp;

    /* a first start: no file, then the device's at its teardown */
    assert(!start(spv, path, sizeof path, &pp));
    assert(!strncmp(path, sub, strlen(sub)) && strstr(path, ".bin"));
    long n = file_len(path);
    uint32_t h[4];
    assert(n >= 32);
    header(path, h);
    assert(h[0] >= 32 && h[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE && h[2] == pp.vendorID && h[3] == pp.deviceID);
    /* the next start reads it */
    assert(start(spv, path, sizeof path, &pp));

    /* another vendor's file under this name: not handed to the driver, written over */
    uint8_t *data = malloc((size_t)n);
    FILE *f = fopen(path, "rb");
    assert(data && f && fread(data, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    uint32_t other = pp.vendorID ^ 0x5a5au;
    memcpy(data + 8, &other, 4);
    put(path, data, (size_t)n);
    assert(!start(spv, path, sizeof path, &pp));
    header(path, h);
    assert(h[2] == pp.vendorID);
    /* another driver's cache UUID: the same */
    memcpy(data + 8, &pp.vendorID, 4);
    data[16] ^= 0xff;
    put(path, data, (size_t)n);
    assert(!start(spv, path, sizeof path, &pp));
    /* a file cut short of its header: the same */
    put(path, data, 20);
    assert(!start(spv, path, sizeof path, &pp));
    assert(file_len(path) >= 32);
    assert(start(spv, path, sizeof path, &pp));
    free(data);

    /* a named build: the watcher's thread comes and goes around it, at once here */
    assert(coli_vk_init(spv));
    g_pc_watch_s = 0;
    for (int k = 0; k < 3; k++) {
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = G.shader, .pName = "main"},
            .layout = G.plyt};
        VkPipeline p = VK_NULL_HANDLE;
        assert(coli_vk_pipe_create(G.dev, &ci, &p, k ? "qmatmul.spv" : NULL) == VK_SUCCESS && p);
        vkDestroyPipeline(G.dev, p, NULL);
    }
    g_pc_watch_s = 10;
    coli_vk_shutdown();

    /* no temporary file is left beside it */
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.%ld.tmp", path, (long)getpid());
    assert(file_len(tmp) < 0);

    /* COLI_VK_PIPELINE_CACHE=0: a cache for the process alone, no file */
    setenv("COLI_VK_PIPELINE_CACHE", "0", 1);
    assert(!start(spv, path, sizeof path, &pp));
    assert(!path[0]);

    /* unset: under $XDG_CACHE_HOME/colibri/vk-pipelines */
    unsetenv("COLI_VK_PIPELINE_CACHE");
    setenv("XDG_CACHE_HOME", dir, 1);
    assert(!start(spv, path, sizeof path, &pp));
    snprintf(tmp, sizeof tmp, "%s/colibri/vk-pipelines/", dir);
    assert(!strncmp(path, tmp, strlen(tmp)));

    char cmd[1200];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
    if (system(cmd) != 0) fprintf(stderr, "test_vk_pcache: could not remove %s\n", dir);
    puts("test_vk_pcache: ok");
    return 0;
}
