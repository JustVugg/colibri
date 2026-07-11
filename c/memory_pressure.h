#ifndef COLIBRI_MEMORY_PRESSURE_H
#define COLIBRI_MEMORY_PRESSURE_H

static inline double coli_safe_pin_bytes(double requested, double expert_budget,
                                         double max_fraction){
    if(requested<=0 || expert_budget<=0 || max_fraction<=0) return 0;
    if(max_fraction>1) max_fraction=1;
    double limit=expert_budget*max_fraction;
    return requested<limit ? requested : limit;
}

static inline int coli_memory_pressure(double available, double reserve,
                                       double swap_free, double swap_free_boot,
                                       double swap_tolerance){
    if(available>=0 && reserve>0 && available<reserve) return 1;
    return swap_free>=0 && swap_free_boot>=0 && swap_free+swap_tolerance<swap_free_boot;
}

#endif
