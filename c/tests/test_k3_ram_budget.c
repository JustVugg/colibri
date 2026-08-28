/* test_k3_ram_budget.c -- #855: Kimi K3 sized its expert cache from
 * K3_EXPERT_GB alone, with nothing subtracted for the resident set, no
 * MemAvailable, no reserve, and no guard. On the reported 256 GB box that
 * projected a 251.4 GB peak: the cache is empty when the init line prints and
 * fills as the session runs, so the first request answers and a later one dies.
 *
 * k3_cap_for_ram() is pure by design -- no globals, no model, no I/O -- so the
 * arithmetic can be checked against the reported numbers without the
 * checkpoint. The cases below use @brad-evony's figures verbatim:
 *
 *     [K3] init done in 2540.5s | 93 layers | expert cache 136/layer
 *          (17.5 MB/slot) | RSS 35.2 GB
 *     set K3_EXPERT_GB=220 ; python coli web --ram 242 ...
 */
#define main coli_k3_main_unused
#include "../kimi_k3.c"
#undef main

#include <stdio.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* The reported machine, exactly. */
enum { NMOE = 93, N_EXPERTS = 384 };
static const double SLOT_GB   = 17.5e-3;   /* 17.5 MB per slot */
static const double RESIDENT  = 35.2;      /* RSS at the init line */
static const double KV_GB     = 3.0;       /* projected, K3_MAXT default */
static const double RESERVE   = 2.5 + 1.2 + KV_GB;

/* What v1.5.0 did: K3_EXPERT_GB straight into a cap, nothing subtracted. */
static int cap_v150(double egb){
    int cap = (int)((egb*1e9)/(SLOT_GB*1e9*(double)NMOE));
    if(cap < 1) cap = 1;
    if(cap > N_EXPERTS) cap = N_EXPERTS;
    return cap;
}
static double peak_gb(int cap){ return RESIDENT + RESERVE + (double)cap*SLOT_GB*NMOE; }
static double peak_ckpt_gb(int cap, double ckpt){ return peak_gb(cap) + ckpt; }

int main(void){
    /* ---- the reported failure, reproduced ---------------------------------- */
    int old = cap_v150(220.0);
    printf("v1.5.0: K3_EXPERT_GB=220 -> cap %d/layer, projected peak %.1f GB\n",
        old, peak_gb(old));
    CHECK(old == 135 || old == 136);        /* the reported 136, modulo rounding */
    CHECK(peak_gb(old) > 250.0);            /* on a 256 GB box */

    /* ---- the same request, now clamped ------------------------------------- */
    /* --ram 242 is the whole-process ceiling the user actually passed. */
    double left = 0;
    int cap = k3_cap_for_ram(242.0, RESIDENT, RESERVE, SLOT_GB, NMOE,
                             old, N_EXPERTS, &left);
    printf("fixed : --ram 242 -> %.1f GB for experts -> cap %d/layer, peak %.1f GB\n",
        left, cap, peak_gb(cap));
    CHECK(cap > 0);
    CHECK(cap < old);                       /* the request was lowered */
    CHECK(peak_gb(cap) <= 242.0 + 0.1);     /* and the peak now fits the budget */
    /* It must not overshoot in the other direction either: a cache so small it
     * defeats the point would be a different bug, not a fix. */
    CHECK(cap > old/2);

    /* ---- opt-in recurrent checkpoints are part of the same ceiling ------- */
    /* Full K3 geometry: 69 KDA rows, 24 heads x 256 recurrent state and three
     * 6144x4 conv windows. This is the ~434 MiB/photo stated by COLI_K3_CKPT.
     * The planner runs before photos are allocated, so it must reserve every
     * enabled slot or RAM_GB will be exceeded later in the conversation. */
    Model cm={0}; cm.c.n_layers=69; cm.c.kda_heads=24; cm.c.kda_hd=256;
    cm.c.kda_proj=6144; cm.c.conv_k=4;
    cm.L=calloc((size_t)cm.c.n_layers,sizeof(Layer));
    for(int i=0;i<cm.c.n_layers;i++) cm.L[i].kda=1;
    double one_ckpt=k3_ckpt_reserve_gb(&cm,1);
    double four_ckpt=k3_ckpt_reserve_gb(&cm,4);
    printf("        COLI_K3_CKPT=1 -> %.1f MiB; 4 slots -> %.2f GB reserved\n",
           one_ckpt*1e9/1048576.0,four_ckpt);
    CHECK(k3_ckpt_reserve_gb(&cm,0)==0.0);
    CHECK(one_ckpt>0.42&&one_ckpt<0.47);
    CHECK(four_ckpt>1.7&&four_ckpt<1.9);
    /* disk-parked photos (COLI_K3_CKPT_DIR): one bounce buffer in RAM no
     * matter how many slots -- the low-RAM machine must not be charged N. */
    g_k3_ckpt_dir="/anywhere";
    CHECK(k3_ckpt_reserve_gb(&cm,4)==one_ckpt);
    g_k3_ckpt_dir=NULL;
    int with_ckpt=k3_cap_for_ram(242.0,RESIDENT,RESERVE+four_ckpt,
                                 SLOT_GB,NMOE,old,N_EXPERTS,&left);
    printf("        --ram 242 + 4 checkpoints -> cap %d/layer, peak %.1f GB\n",
           with_ckpt,peak_ckpt_gb(with_ckpt,four_ckpt));
    CHECK(with_ckpt<cap);
    CHECK(peak_ckpt_gb(with_ckpt,four_ckpt)<=242.0+0.1);
    double one_slot_peak=RESIDENT+RESERVE+four_ckpt+SLOT_GB*NMOE;
    CHECK(k3_cap_for_ram(one_slot_peak-0.01,RESIDENT,RESERVE+four_ckpt,
                          SLOT_GB,NMOE,old,N_EXPERTS,NULL)==0);
    free(cm.L);

    /* ---- the flag has to mean something ------------------------------------ */
    /* Two budgets, everything else equal: a bigger --ram must buy more cache.
     * Before this change --ram was not read at all, so both would be 136. */
    int at180 = k3_cap_for_ram(180.0, RESIDENT, RESERVE, SLOT_GB, NMOE, old, N_EXPERTS, NULL);
    int at242 = k3_cap_for_ram(242.0, RESIDENT, RESERVE, SLOT_GB, NMOE, old, N_EXPERTS, NULL);
    printf("        --ram 180 -> cap %d   |   --ram 242 -> cap %d\n", at180, at242);
    CHECK(at180 < at242);
    CHECK(peak_gb(at180) <= 180.0 + 0.1);

    /* ---- it must only ever LOWER ------------------------------------------- */
    /* A user who asked for a small cache on a huge machine keeps it: this is a
     * clamp, not an autotuner. Budget deliberately absurd. */
    int small = k3_cap_for_ram(4096.0, RESIDENT, RESERVE, SLOT_GB, NMOE, 4, N_EXPERTS, NULL);
    printf("        K3_EXPERT_GB small (cap 4) on a 4 TB budget -> cap %d\n", small);
    CHECK(small == 4);

    /* ---- no budget left ---------------------------------------------------- */
    /* Resident alone over the budget: zero, so the caller can refuse rather than
     * being handed a comfortable-looking 1 it never asked about. */
    int none = k3_cap_for_ram(30.0, RESIDENT, RESERVE, SLOT_GB, NMOE, old, N_EXPERTS, &left);
    printf("        --ram 30 with %.1f GB resident -> %.1f GB for experts, cap %d\n",
        RESIDENT, left, none);
    CHECK(none == 0);
    CHECK(left < 0);

    /* ---- the cap never exceeds the expert count ---------------------------- */
    int huge = k3_cap_for_ram(100000.0, RESIDENT, RESERVE, SLOT_GB, NMOE,
                              N_EXPERTS*4, N_EXPERTS, NULL);
    CHECK(huge <= N_EXPERTS);

    /* ---- degenerate inputs do not produce a negative or wild cap ----------- */
    CHECK(k3_cap_for_ram(242.0, RESIDENT, RESERVE, 0.0, NMOE, 7, N_EXPERTS, NULL) == 7);
    CHECK(k3_cap_for_ram(242.0, RESIDENT, RESERVE, SLOT_GB, 0, 7, N_EXPERTS, NULL) >= 0);

    printf("test_k3_ram_budget: %s\n", fails?"FAILED":"ok");
    return fails?1:0;
}
