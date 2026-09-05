/* La guardia RSS deve contare la memoria che il processo POSSIEDE, non quella
 * che il kernel puo' riprendersi da solo.
 *
 * rss_guard() sfratta esperti quando la misura supera il budget. Leggendo
 * VmRSS, che include le pagine di file mappate, con COLI_MAP_EXPERTS=1 (#1325)
 * la misura sale di gigabyte senza che un byte in piu' sia stato sottratto al
 * sistema: la guardia si metterebbe a sfrattare per liberare memoria che non
 * stava occupando.
 *
 * Il test mappa un file vero e pretende le DUE cose che rendono sicura la
 * correzione: la misura non deve seguire la mappatura, e deve invece seguire
 * una allocazione anonima -- se non facesse la seconda, avremmo reso la guardia
 * cieca invece che precisa, che sarebbe peggio del difetto. */
#include <stdio.h>

/* Il contratto verificato qui e' quello di /proc/self/status, che esiste solo
 * su Linux: la guardia RSS su altre piattaforme usa gia' un'altra strada
 * (rss_gb). Non c'e' niente da portare -- portarlo vorrebbe dire inventare un
 * equivalente di RssAnon che il codice sotto test non usa. Su tutto il resto
 * il test si dichiara saltato invece di fallire, cosi' `make check` resta
 * verde ovunque e nessuno lo scopre da un rosso in CI. */
#if !defined(__linux__)
int main(void) { printf("test_rss_anon: non-Linux, skip\n"); return 0; }
#else

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* La stessa lettura che fa current_rss_gb in colibri.c: RssAnon con VmRSS come
 * ripiego. Duplicata qui e non inclusa perche' colibri.c e' un'unita' enorme e
 * questo test deve restare veloce; il contratto verificato e' identico. */
static double rss_anon_gb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    unsigned long long anon = 0, vm = 0;
    int ha = 0, hv = 0;
    while (fgets(line, sizeof line, f)) {
        if (!ha && !strncmp(line, "RssAnon:", 8)) { if (sscanf(line+8, "%llu", &anon)==1) ha=1; }
        else if (!hv && !strncmp(line, "VmRSS:", 6)) { if (sscanf(line+6, "%llu", &vm)==1) hv=1; }
        if (ha && hv) break;
    }
    fclose(f);
    if (!ha && !hv) return -1;
    return (double)(ha ? anon : vm) / (1024.0 * 1024.0);
}

static double vmrss_gb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; unsigned long long kb = 0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "VmRSS:", 6)) { sscanf(line+6, "%llu", &kb); break; }
    fclose(f);
    return (double)kb / (1024.0 * 1024.0);
}

int main(void) {
    if (rss_anon_gb() < 0) { printf("test_rss_anon: /proc non leggibile, salto\n"); return 0; }

    const size_t MAPPED = 256u << 20;          /* 256 MB: ben oltre il rumore */
    char path[] = "/tmp/coli_rss_anon_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); return 1; }
    if (ftruncate(fd, (off_t)MAPPED) != 0) { perror("ftruncate"); return 1; }

    double anon_before = rss_anon_gb(), vm_before = vmrss_gb();

    /* ---- 1. una mappatura di file toccata NON deve muovere la misura ---- */
    char *m = mmap(NULL, MAPPED, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    volatile char sink = 0;
    for (size_t i = 0; i < MAPPED; i += 4096) sink ^= m[i];   /* falla residente */
    (void)sink;

    double anon_mapped = rss_anon_gb(), vm_mapped = vmrss_gb();

    check(vm_mapped - vm_before > 0.15,
          "il test non ha reso residente la mappatura: senza questo non prova nulla");
    check(anon_mapped - anon_before < 0.05,
          "la misura segue le pagine di file mappate: la guardia sfratterebbe "
          "esperti per liberare memoria che il kernel puo' recuperare da solo");

    munmap(m, MAPPED);
    close(fd); unlink(path);

    /* ---- 2. una allocazione ANONIMA deve invece muoverla ---- */
    double anon_pre = rss_anon_gb();
    const size_t ANON = 256u << 20;
    char *a = malloc(ANON);
    if (!a) { printf("  FAIL: malloc\n"); return 1; }
    /* volatile e letta dopo: senza questo il compilatore elimina la scrittura
     * (la memoria non viene mai riletta prima della free) e le pagine non
     * vengono mai toccate davvero -- misurato: RssAnon fermo a +4 kB su 256 MB
     * allocati. Un test che alloca senza toccare non misura niente. */
    volatile char *va = (volatile char *)a;
    for (size_t i = 0; i < ANON; i += 4096) va[i] = (char)(i & 0x7f);
    volatile char keep = 0;
    for (size_t i = 0; i < ANON; i += (4096 * 64)) keep ^= va[i];
    (void)keep;
    double anon_post = rss_anon_gb();
    check(anon_post - anon_pre > 0.15,
          "la misura NON segue la memoria anonima: la guardia sarebbe cieca, "
          "che e' peggio del difetto che stiamo correggendo");
    free(a);

    if (fails) { printf("test_rss_anon: %d fallimenti\n", fails); return 1; }
    printf("test_rss_anon: ok\n");
    return 0;
}

#endif /* __linux__ */
