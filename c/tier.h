#ifndef COLIBRI_TIER_H
#define COLIBRI_TIER_H

#include <stdint.h>

/* Pick one RAM/VRAM hot-store slot to replace from recent routing heat.
 * The fixed margin handles tiny samples; the 25% margin prevents ping-pong. */
static int tier_pick_swap(const uint32_t *heat, int nexpert,
                          const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat || !pinned || npin<1 || nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++) if(heat[pinned[z]]<heat[pinned[cold]]) cold=z;
    int hot=-1; uint32_t fh=0;
    for(int e=0;e<nexpert;e++){
        int resident=0;
        for(int z=0;z<npin;z++) if(pinned[z]==e){ resident=1; break; }
        if(!resident && heat[e]>fh){ fh=heat[e]; hot=e; }
    }
    if(hot<0) return 0;
    uint32_t fc=heat[pinned[cold]];
    if(fh<=fc+(fc>>2)+4) return 0;
    *slot=cold; *eid=hot; *gain=(long)fh-(long)fc;
    return 1;
}

static void tier_decay(uint32_t *heat, int nexpert){
    for(int e=0;e<nexpert;e++) heat[e]>>=1;
}

#endif
