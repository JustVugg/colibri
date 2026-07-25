/* Memory budget manager for the trainer (AGENTS.md §11).
 *
 * One place owns the --ram budget. Mandatory categories are reserved first, in
 * order; only the REMAINDER feeds the expert cache. If the mandatory set alone
 * exceeds the budget, planning fails loudly before any training starts — the
 * trainer must never discover mid-step that it cannot fit.
 *
 * Measurement is macOS-honest: phys_footprint (compressed memory counted, what
 * Activity Monitor shows) rather than plain RSS, peak via ru_maxrss, and swap
 * growth via vm.swapusage deltas — a run that leans on swap is a failed run
 * even if it "works". Linux fallback: /proc/self/statm. */
#ifndef TRAIN_BUDGET_H
#define TRAIN_BUDGET_H
#include <stdint.h>
#include <stdio.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

typedef struct {
    int64_t total;           /* --ram budget (bytes) */
    /* reserved first, in this order (§11 dynamic budget rule) */
    int64_t dense;           /* resident base tensors (model_init resident_bytes) */
    int64_t adapter_state;   /* params + grads + AdamW m/v (4x param bytes, f32) */
    int64_t train_scratch;   /* activation stash + backward transients */
    int64_t ckpt_reserve;    /* per-layer x_in checkpoints + routing ids */
    int64_t os_margin;       /* safety floor, default 8 GB */
    /* derived */
    int64_t expert_cache;    /* remainder — the ONLY elastic category */
    int64_t swap_base;       /* vm.swapusage at plan time, for delta reporting */
} TBudget;

/* current physical footprint of this process, bytes (0 if unavailable) */
static int64_t tb_footprint(void){
#ifdef __APPLE__
    task_vm_info_data_t vi; mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&vi,&n)==KERN_SUCCESS)
        return (int64_t)vi.phys_footprint;
    return 0;
#else
    FILE *f=fopen("/proc/self/statm","r"); if(!f) return 0;
    long tot=0,res=0; int rc=fscanf(f,"%ld %ld",&tot,&res); fclose(f);
    return rc==2 ? (int64_t)res*4096 : 0;
#endif
}

static int64_t tb_peak(void){
    struct rusage ru; getrusage(RUSAGE_SELF,&ru);
#ifdef __APPLE__
    return (int64_t)ru.ru_maxrss;            /* bytes on macOS */
#else
    return (int64_t)ru.ru_maxrss*1024;       /* KiB on Linux */
#endif
}

static int64_t tb_swap_used(void){
#ifdef __APPLE__
    struct xsw_usage sw; size_t len=sizeof(sw);
    if(sysctlbyname("vm.swapusage",&sw,&len,NULL,0)==0) return (int64_t)sw.xsu_used;
#endif
    return 0;
}

/* Plan the budget. Returns 0 and fills expert_cache, or -1 (with a report on
 * stderr) if the mandatory categories don't fit under total. */
static int tbudget_plan(TBudget *b){
    if(b->os_margin<=0) b->os_margin=8ll<<30;
    int64_t mandatory = b->dense + b->adapter_state + b->train_scratch
                      + b->ckpt_reserve + b->os_margin;
    b->swap_base = tb_swap_used();
    if(mandatory > b->total){
        fprintf(stderr,"[budget] does not fit: mandatory %.2f GB > budget %.2f GB\n"
                       "[budget]   dense %.2f | adapter+opt %.2f | scratch %.2f | ckpt %.2f | OS %.2f\n",
                mandatory/1073741824.0, b->total/1073741824.0,
                b->dense/1073741824.0, b->adapter_state/1073741824.0,
                b->train_scratch/1073741824.0, b->ckpt_reserve/1073741824.0,
                b->os_margin/1073741824.0);
        return -1;
    }
    b->expert_cache = b->total - mandatory;
    return 0;
}

/* one status line, cheap enough for every-step logging */
static void tbudget_log(const TBudget *b, FILE *out){
    int64_t fp=tb_footprint(), pk=tb_peak(), sw=tb_swap_used()-b->swap_base;
    fprintf(out,
        "[mem] footprint %.2f GB | peak %.2f GB | budget %.2f GB "
        "(dense %.2f, adapter %.2f, scratch %.2f, ckpt %.2f, expert-cache %.2f, OS %.2f) | swap-delta %+.1f MB%s\n",
        fp/1073741824.0, pk/1073741824.0, b->total/1073741824.0,
        b->dense/1073741824.0, b->adapter_state/1073741824.0,
        b->train_scratch/1073741824.0, b->ckpt_reserve/1073741824.0,
        b->expert_cache/1073741824.0, b->os_margin/1073741824.0,
        sw/1048576.0, sw>(64ll<<20) ? "  ** SWAPPING — run is invalid **" : "");
}

/* hard gate for the step loop: footprint above ceiling -> caller must abort */
static int tbudget_violated(const TBudget *b, int64_t ceiling){
    int64_t fp=tb_footprint();
    if(ceiling>0 && fp>ceiling) return 1;
    if(tb_swap_used()-b->swap_base > (256ll<<20)) return 2;   /* sustained swap */
    return 0;
}

#endif /* TRAIN_BUDGET_H */
