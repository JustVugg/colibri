/* Microbench: banda in lettura RANDOM con blocchi tipo-expert (~19 MB int4).
 * Misura cio' che fa il motore davvero: N thread che leggono in parallelo
 * (expert_load sotto omp parallel for), buffered oppure O_DIRECT.
 * uso: ./iobench <file_grande> [blocco_MB] [n_letture] [threads] [direct 0/1]
 * build: gcc -O2 -fopenmp iobench.c -o iobench */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include "compat.h"
#ifdef _OPENMP
#include <omp.h>
#endif
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s file [blkMB] [n] [threads] [direct 0/1]\n",argv[0]);return 1;}
    long blk=(argc>2?atol(argv[2]):19)*1024*1024;
    int n=argc>3?atoi(argv[3]):64;
    int nth=argc>4?atoi(argv[4]):8;
    int direct=argc>5?atoi(argv[5]):1;
#ifdef O_DIRECT
    int fd=open(argv[1],O_RDONLY|(direct?O_DIRECT:0));
    if(fd<0 && direct){ fprintf(stderr,"O_DIRECT is unavailable (%s); using buffered I/O\n",strerror(errno));
        direct=0; fd=open(argv[1],O_RDONLY); }
#else
    int fd=open(argv[1],O_RDONLY);                 /* macOS: F_NOCACHE ~ O_DIRECT */
#ifdef __APPLE__
    if(direct && fd>=0) fcntl(fd,F_NOCACHE,1);
#else
    if(direct){ fprintf(stderr,"O_DIRECT is unavailable; using buffered I/O\n"); direct=0; }
#endif
#endif
    if(fd<0){perror("open");return 1;}
    off_t sz=lseek(fd,0,SEEK_END);
    if(sz<blk*2){fprintf(stderr,"file is too small\n");return 1;}
    /* offset random pre-generati (stessi per ogni configurazione: srand fisso).
     * 30 bit di rand combinati: su Windows RAND_MAX=32767 e un singolo rand()*4096
     * copre solo i primi 134 MB del file (tutti in page cache = misura falsa). */
    off_t *offs=malloc(n*sizeof(off_t)); srand(1234);
    for(int i=0;i<n;i++){ off_t r30=((off_t)rand()<<15)|rand(); off_t o=(r30*4096)%(sz-blk); offs[i]=o&~4095L; }
    double t0=now(); int64_t tot=0;   /* long e' 32-bit su Windows (LLP64): >2GB andava in overflow */
    #pragma omp parallel num_threads(nth) reduction(+:tot)
    {
        void *buf; if(posix_memalign(&buf,4096,blk)){perror("memalign");exit(1);}
        #pragma omp for schedule(dynamic,1)
        for(int i=0;i<n;i++){
            ssize_t r=pread(fd,buf,blk,offs[i]);
            if(r<0) perror("pread"); else tot+=r;
        }
        compat_aligned_free(buf);   /* su Windows posix_memalign=_aligned_malloc: free() corrompe l'heap */
    }
    double dt=now()-t0;
    printf("%s x%d threads: %d reads x %ldMB = %.1f GB in %.2fs -> %.2f GB/s (%.1f effective ms/block)\n",
        direct?"O_DIRECT":"buffered", nth, n, blk/1024/1024, tot/1e9, dt, tot/1e9/dt, dt/n*1000);
    close(fd); free(offs); return 0;
}
