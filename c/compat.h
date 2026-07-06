/* compat.h — shim di portabilita' per piattaforme non-Linux (oggi: macOS / Apple Silicon).
 * Su Linux questo header e' un NO-OP totale: nessun simbolo definito o ridefinito,
 * zero impatto sul percorso x86 esistente.
 * Regola: ogni differenza di piattaforma vive QUI; i .c restano puliti. */
#ifndef COMPAT_H
#define COMPAT_H

#ifdef __APPLE__
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

/* --- posix_fadvise: assente su macOS ---
 * WILLNEED -> F_RDADVISE (readahead esplicito: stessa semantica).
 * DONTNEED -> no-op: XNU non espone un drop mirato per-range; la sua unified
 *             buffer cache si autoregola sotto pressione. Il motore usa DONTNEED
 *             solo come consiglio, quindi ignorarlo e' corretto (e su una macchina
 *             con molta RAM tenere le pagine e' proprio cio' che si vuole). */
#ifndef POSIX_FADV_NORMAL
#define POSIX_FADV_NORMAL      0
#define POSIX_FADV_RANDOM      1
#define POSIX_FADV_SEQUENTIAL  2
#define POSIX_FADV_WILLNEED    3
#define POSIX_FADV_DONTNEED    4
#define POSIX_FADV_NOREUSE     5
#endif
static inline int compat_fadvise(int fd, off_t off, off_t len, int advice){
    if(advice==POSIX_FADV_WILLNEED){
        struct radvisory ra;
        ra.ra_offset = off;
        ra.ra_count  = (int)(len>0x7FFFFFFF ? 0x7FFFFFFF : len);
        return fcntl(fd, F_RDADVISE, &ra)<0 ? -1 : 0;
    }
    return 0;
}
#define posix_fadvise compat_fadvise

/* --- O_DIRECT: assente su macOS ---
 * L'equivalente e' F_NOCACHE sul fd (bypass della unified buffer cache).
 * compat_open_direct() apre il fd "gemello" senza cache, come il twin O_DIRECT
 * di st.h. Le pread allineate a 4K del chiamante restano valide: F_NOCACHE non
 * impone vincoli di allineamento. */
static inline int compat_open_direct(const char *path){
    int fd = open(path, O_RDONLY);
    if(fd>=0) fcntl(fd, F_NOCACHE, 1);
    return fd;
}
#endif /* __APPLE__ */

#endif /* COMPAT_H */
