#ifndef COLIBRI_IO_POLICY_H
#define COLIBRI_IO_POLICY_H

#include <stdlib.h>

static int io_env_threads(const char *name,int fallback){
    const char *value=getenv(name);
    if(!value||!*value) return fallback;
    int threads=atoi(value); return threads>0?threads:fallback;
}

static int expert_io_threads(int misses,int foreground){
    if(misses<1) return 1;
    int legacy=io_env_threads("IO_THREADS",0);
    if(legacy) return legacy<misses?legacy:misses;
    int fallback=foreground?(misses<8?misses:8):misses;
    int threads=io_env_threads(foreground?"IO_OVERLAP_THREADS":"IO_IDLE_THREADS",fallback);
    return threads<misses?threads:misses;
}

#endif
