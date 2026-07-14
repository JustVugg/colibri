/* test_compat_direct.c — O_DIRECT equivalente su Windows: FILE_FLAG_NO_BUFFERING.
 * compat_open_direct(path) deve dare un fd che bypassa la cache del file system,
 * leggibile con pread() a offset/len/buffer allineati a 4K (stesso contratto di
 * O_DIRECT su Linux: richieste non allineate falliscono, mai dati corrotti). */
#include <stdio.h>

#ifndef _WIN32
int main(void){ puts("compat direct tests: skipped (POSIX has native O_DIRECT)"); return 0; }
#else

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <io.h>
#include "../compat.h"

static int fail(const char *m){ fprintf(stderr,"compat direct test failed: %s\n",m); return 1; }

#define FSZ (1u<<20)
#define TMPF "test_direct.tmp"

int main(void){
    FILE *w=fopen(TMPF,"wb"); if(!w) return fail("create temp");
    uint8_t *pat=malloc(FSZ);
    for(uint32_t i=0;i<FSZ;i++) pat[i]=(uint8_t)(i*2246822519u>>24);
    if(fwrite(pat,1,FSZ,w)!=FSZ){ fclose(w); return fail("short write"); }
    fclose(w);

    int dfd = compat_open_direct(TMPF);
    if(dfd<0) return fail("compat_open_direct returned -1");

    /* lettura allineata 4K (offset, len, buffer): deve restituire i byte esatti */
    void *buf=NULL;
    if(posix_memalign(&buf,4096,64*1024)!=0) return fail("alloc aligned");
    if(pread(dfd, buf, 64*1024, 4096)!=64*1024) return fail("aligned pread size");
    if(memcmp(buf, pat+4096, 64*1024)!=0) return fail("aligned pread data mismatch");

    /* richiesta non allineata: deve fallire (-1), MAI restituire dati sbagliati */
    ssize_t r = pread(dfd, buf, 64*1024, 1000);
    if(r>0 && memcmp(buf, pat+1000, (size_t)r)!=0) return fail("misaligned read returned wrong data");

    /* dimensione file: lseek(SEEK_END) FALLISCE sui fd NO_BUFFERING (misurato:
     * ritorna -1); compat_fsize deve funzionare su entrambi i tipi di fd */
    if(compat_fsize(dfd)!=(off_t)FSZ) return fail("compat_fsize on direct fd");
    int bfd = open(TMPF, COMPAT_O_RDONLY);
    if(compat_fsize(bfd)!=(off_t)FSZ) return fail("compat_fsize on buffered fd");
    close(bfd);

    /* fd inesistente */
    if(compat_open_direct("no_such_file.tmp")>=0) return fail("open missing file must fail");
    if(compat_fsize(-1)>=0) return fail("compat_fsize on bad fd must be negative");

    close(dfd);
    compat_aligned_free(buf); free(pat); remove(TMPF);
    puts("compat direct tests: ok");
    return 0;
}
#endif /* _WIN32 */
