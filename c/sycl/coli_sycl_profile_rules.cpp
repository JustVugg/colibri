/* coli_sycl_profile_rules.cpp — the decisions a profile makes from what a device reports
 * (coli_sycl_profile.h), with no SYCL in the unit: coli_sycl_profile.cpp hands it the values it
 * read, tests/test_sycl_profile.cpp hands it fake sets. Every rule is written once here; the
 * reading side never decides anything. */
#include "coli_sycl_profile.h"
#include <cstdlib>
#include <cstring>

/* ---- names ---------------------------------------------------------------------------- */

static const struct { coli_sycl_arch arch; const char *name; } arch_names[] = {
    { COLI_SYCL_ARCH_PVC, "pvc" },   { COLI_SYCL_ARCH_BMG, "bmg" },   { COLI_SYCL_ARCH_LNL, "lnl" },
    { COLI_SYCL_ARCH_PTL, "ptl" },   { COLI_SYCL_ARCH_ARL, "arl" },   { COLI_SYCL_ARCH_MTL, "mtl" },
    { COLI_SYCL_ARCH_DG2, "dg2" },   { COLI_SYCL_ARCH_WCL, "wcl" },   { COLI_SYCL_ARCH_XELP, "xelp" },
    { COLI_SYCL_ARCH_GEN9, "gen9" }, { COLI_SYCL_ARCH_OTHER, "generic" }, { COLI_SYCL_ARCH_NONE, "none" },
};

const char *coli_sycl_arch_name(coli_sycl_arch a) {
    for (const auto &r : arch_names) if (r.arch == a) return r.name;
    return "none";
}
coli_sycl_arch coli_sycl_arch_from_name(const char *name) {
    if (!name) return COLI_SYCL_ARCH_NONE;
    for (const auto &r : arch_names) if (r.arch != COLI_SYCL_ARCH_NONE && !strcmp(r.name, name)) return r.arch;
    return COLI_SYCL_ARCH_NONE;
}
const char *coli_sycl_gen_name(coli_sycl_gen g) {
    switch (g) {
    case COLI_SYCL_GEN_XE: return "xe";     case COLI_SYCL_GEN_PVC: return "pvc";
    case COLI_SYCL_GEN_XE2: return "xe2";   case COLI_SYCL_GEN_XE3: return "xe3";
    case COLI_SYCL_GEN_LEGACY: return "legacy";
    default: return "unknown";
    }
}
coli_sycl_gen coli_sycl_gen_of(coli_sycl_arch a) {
    switch (a) {
    case COLI_SYCL_ARCH_PVC: return COLI_SYCL_GEN_PVC;
    case COLI_SYCL_ARCH_BMG: case COLI_SYCL_ARCH_LNL: return COLI_SYCL_GEN_XE2;
    case COLI_SYCL_ARCH_PTL: case COLI_SYCL_ARCH_WCL: return COLI_SYCL_GEN_XE3;
    case COLI_SYCL_ARCH_ARL: case COLI_SYCL_ARCH_MTL: case COLI_SYCL_ARCH_DG2: return COLI_SYCL_GEN_XE;
    case COLI_SYCL_ARCH_XELP: case COLI_SYCL_ARCH_GEN9: return COLI_SYCL_GEN_LEGACY;
    default: return COLI_SYCL_GEN_UNKNOWN;
    }
}

/* ---- the matrix engines' shapes ------------------------------------------------------- */

/* whether an entry takes an m x n x k tile. The runtime encodes each dimension on its own: a
 * fixed size (msize, nsize, ksize) or a range (max_msize, max_nsize, max_ksize; the other zero),
 * and Intel's rows mix them: {max_msize 8, nsize 16, ksize 32} is M up to 8 with N and K fixed
 * (the f16 / bf16 8x16x16, s8 8x16x32 and tf32 8x16x8 rows of pvc and the Xe2 parts), the
 * compound f16 16x16x16 a row of three fixed sizes, dg2's 8xK rows the same with N fixed at 8.
 * So an entry is taken dimension by dimension: a fixed one exactly, a ranged one up to its
 * maximum, a dimension with neither never. */
static bool dim_takes(size_t max, size_t fixed, size_t want) { return fixed ? fixed == want : max >= want; }
static bool combo_takes(const coli_sycl_combo &c, size_t m, size_t n, size_t k) {
    return dim_takes(c.max_m, c.m, m) && dim_takes(c.max_n, c.n, n) && dim_takes(c.max_k, c.k, k);
}
unsigned coli_sycl_xmx_shape_flags(const coli_sycl_combo *c, int n) {
    unsigned f = 0;
    for (int i = 0; i < n; i++) {
        const coli_sycl_combo &e = c[i];
        if (e.a != e.b) continue;
        if (e.a == COLI_SYCL_DT_F16 && e.c == COLI_SYCL_DT_F32) {
            if (combo_takes(e, 8, 16, 16)) f |= COLI_SYCL_XMX_F16;
            if (combo_takes(e, 16, 16, 16)) f |= COLI_SYCL_XMX_F16_16;
        } else if (e.a == COLI_SYCL_DT_BF16 && e.c == COLI_SYCL_DT_F32) {
            if (combo_takes(e, 8, 16, 16)) f |= COLI_SYCL_XMX_BF16;
        } else if (e.a == COLI_SYCL_DT_INT8 && e.c == COLI_SYCL_DT_S32) {
            if (combo_takes(e, 8, 16, 32)) f |= COLI_SYCL_XMX_INT8;
        } else if (e.a == COLI_SYCL_DT_TF32 && e.c == COLI_SYCL_DT_F32) {
            if (combo_takes(e, 8, 16, 8)) f |= COLI_SYCL_XMX_TF32;
        }
    }
    return f;
}
bool coli_sycl_xmx_supported(const coli_sycl_profile &p, coli_sycl_dtype dt) {
    if (!p.has_xmx) return false;
    switch (dt) {
    case COLI_SYCL_DT_F16: return (p.xmx_shapes & COLI_SYCL_XMX_F16) != 0;
    case COLI_SYCL_DT_BF16: return (p.xmx_shapes & COLI_SYCL_XMX_BF16) != 0;
    case COLI_SYCL_DT_INT8: return (p.xmx_shapes & COLI_SYCL_XMX_INT8) != 0;
    case COLI_SYCL_DT_TF32: return (p.xmx_shapes & COLI_SYCL_XMX_TF32) != 0;
    default: return false;
    }
}

/* ---- the derivation ------------------------------------------------------------------- */

coli_sycl_env coli_sycl_env_read(void) {
    coli_sycl_env e;
    e.sg = getenv("COLI_SYCL_SG");
    e.xmx = getenv("COLI_SYCL_XMX");
    e.profile = getenv("COLI_SYCL_PROFILE");
    return e;
}

/* ---- the defaults measured on an architecture (coli_sycl_profile.h) --------------------- */

/* Each row: measured on a Data Center GPU Max 1100 (2026-11-18/19; c/sycl/README.md,
 * "Defaults measured on a Data Center GPU Max"), the texts identical to the switch off in every
 * run. A row is added only with such a measurement on the architecture it names. The first five
 * rows together are the accepted Qwen3.6 state: first token of a 32K prompt 119.9 -> 26 s, eight
 * conversations 2.63 x one on a card. */
static const struct { coli_sycl_arch arch; const char *knob; int value; } tuned_rows[] = {
    /* the prompt's plain rows through the grouped attention kernel from 32 rows (a kv head's keys and values read once
     * for its query heads) */
    { COLI_SYCL_ARCH_PVC, "COLI_CUDA_ATTN_GROUPED_PLAIN", 32 },
    /* the grouped kernel with eight-element key loads and value pairs */
    { COLI_SYCL_ARCH_PVC, "COLI_CUDA_ATTN_GROUPED_VEC", 2 },
    /* the prompt's DeltaNet recurrence (cc_dnrec_split_k) at sub-group 16 */
    { COLI_SYCL_ARCH_PVC, "COLI_CUDA_DNREC_PROMPT_SG", 16 },
    /* the multi-row int8 GEMV's x rows unpadded for blocks of four rows past 32 KiB */
    { COLI_SYCL_ARCH_PVC, "COLI_CUDA_GEMV_XPAD", 2 },
    /* several conversations' DeltaNet convolutions, recurrences, attention and K/V rows as one launch a layer
     * (1 | 2 | 4 | 8) */
    { COLI_SYCL_ARCH_PVC, "COLI_CUDA_MUX_BATCH", 15 },
    /* no start-up timing of the card against the CPU: an engine started while the card is busy otherwise ran on the
     * CPU with another text in 4 of 8 pairs of starts; alone, the same text and rate */
    { COLI_SYCL_ARCH_PVC, "COLI_TRUNK_PROBE", 0 },
    /* Qwen3.8's FP8 hybrid as qualified and measured here (qwen38.c applies its table where this says 1): the trunk
     * on the CPU, the CPU's FP8 order and SiLU on the card (it has FP64), the rank-order sum -- greedy NLL 0.3030 /
     * human 2.6323 -- with the CPU-order kernels 4 and a prompt chunk's resident experts on the card. Unlike the rows
     * above this one changes the bits against the code's default (towards the CPU's: the default, the trunk on the
     * card, failed that gate at 0.3288); it is not measured anywhere else, so nowhere else does it apply. */
    { COLI_SYCL_ARCH_PVC, "Q38_SYCL_HYBRID", 1 },
    /* a start's load, Qwen3.6-35B-A3B on one card (h4-load attempts 2-5, the texts and both losses identical): the
     * experts in one device arena a card, staged through page-locked buffers and repacked outside the tier's lock
     * (arena 2), left in the page cache once read (a restart takes them from memory: 16.7 GB of reclaimable cache on
     * the measured host), the int8 trunk read from canonical.qc where one is beside the model or named by
     * QWEN_DENSE_CACHE (made once by qwen36-canonical; without one the trunk is converted as before): expert warm
     * start 9.5-11.5 s -> 1.25 s on a restart (4.0 s on the first start after the cache is empty), trunk phase
     * 4.8 -> 1.1 s. The arena's slot bookkeeping under the chain's promotions is the fixed one (8ffb54b3). */
    { COLI_SYCL_ARCH_PVC, "COLI_CUDA_EXPERT_ARENA", 2 },
    { COLI_SYCL_ARCH_PVC, "COLI_EXPERT_PAGECACHE", 1 },
    { COLI_SYCL_ARCH_PVC, "COLI_CUDA_DENSE_CACHE", 1 },
    /* a prompt chunk's attention through oneDNN's fused attention, in a SYCL_DNNL=1 build only (coli_sycl_dnnl.cpp;
     * other builds never read this row), mode 1 (the first compile off the request's path): Qwen3.6-35B-A3B first
     * token of a 32K prompt 25.99 -> 11.51 s, 16K 9.07 -> 5.89 s, 3K, 21 tokens and decode unchanged (campaign DNNL3).
     * Unlike the rows above the attention's bits change (float16 Q, P and O, oneDNN's order), but the texts were
     * identical to the switch off in every measured class, and the prompt-path loss was 2.759443 against the grouped
     * kernel's 2.761657 (limit 2.775465). Taken from this row for head size 256 only (the one measured end to end). */
    { COLI_SYCL_ARCH_PVC, "COLI_SYCL_DNNL_ATTN", 1 },
};
int coli_sycl_tuned_default(const coli_sycl_profile &p, const char *platform_defaults, const char *knob, int generic) {
    if (!knob || (platform_defaults && !strcmp(platform_defaults, "0"))) return generic;
    for (const auto &r : tuned_rows)
        if (r.arch == p.arch && !strcmp(r.knob, knob)) return r.value;
    return generic;
}

static void say(FILE *log, const char *fmt, const char *a, const char *b) {
    if (log) { fputs("[SYCL] ", log); fprintf(log, fmt, a, b); fputc('\n', log); }
}
static int offered(const coli_sycl_props &pr, int sg) {
    for (int i = 0; i < pr.sg_count; i++) if (pr.sg_sizes[i] == sg) return 1;
    return 0;
}
/* the sub-group of the warp-32-style kernels for an architecture: 32 on pvc (the native
 * SIMD32 of Xe-HPC, and the width the kernels were written at), 16 everywhere else (the
 * native width of Xe2 and Xe-HPG); each only where the device offers it. The kernels exist at
 * 16 and 32 only, so a device that offers neither (no Intel GPU since gen9; a CPU device's 8)
 * is told what will run, 16, which it may refuse at the launch, rather than a size the
 * dispatch has no instance of. */
static int default_sg(const coli_sycl_props &pr, coli_sycl_arch arch, FILE *log) {
    const int want = arch == COLI_SYCL_ARCH_PVC ? 32 : 16;
    if (offered(pr, want)) return want;
    if (offered(pr, 16)) return 16;
    if (offered(pr, 32)) return 32;
    if (pr.sg_count) say(log, "device %s offers neither sub-group 16 nor 32: the kernels run at 16, which it may refuse%s", pr.device_name, "");
    return 16;
}

coli_sycl_profile coli_sycl_profile_derive(const coli_sycl_props &pr, const coli_sycl_env &env, FILE *log) {
    coli_sycl_profile p;
    memset(&p, 0, sizeof p);
    /* the measured values, with a stated default where a query left its zero */
    p.arch = pr.arch;
    snprintf(p.device_name, sizeof p.device_name, "%s", pr.device_name[0] ? pr.device_name : "unknown device");
    snprintf(p.pci_address, sizeof p.pci_address, "%s", pr.pci_address);
    p.pci_device_id = pr.pci_device_id;
    p.sg_count = pr.sg_count > 8 ? 8 : pr.sg_count;
    for (int i = 0; i < p.sg_count; i++) p.sg_sizes[i] = pr.sg_sizes[i];
    p.eu_count = pr.eu_count > 0 ? pr.eu_count : pr.max_compute_units > 0 ? pr.max_compute_units : 1;
    p.hw_threads_per_eu = pr.hw_threads_per_eu > 0 ? pr.hw_threads_per_eu : 8;   /* pvc, xe2, dg2: 8 (xe-lp 7) */
    p.slm_bytes = pr.slm_bytes ? pr.slm_bytes : 64 * 1024;   /* the smallest any Xe part has */
    p.max_wg_size = pr.max_wg_size ? pr.max_wg_size : 256;    /* the kernels' block; 1024 on every known part */
    p.global_mem_bytes = pr.global_mem_bytes;
    p.is_integrated = pr.host_unified_memory ? 1 : 0;
    p.xmx_shapes = pr.xmx_shapes;
    if (pr.sg_count == 0) say(log, "device %s lists no sub-group sizes: assuming 16%s", p.device_name, "");
    if (!pr.slm_bytes) say(log, "device %s reports no local memory size: assuming 64 KiB%s", p.device_name, "");

    /* COLI_SYCL_PROFILE=<arch>[-sg16|-sg32]: that architecture's decisions on this device */
    int forced_sg = 0;
    if (env.profile && *env.profile) {
        char base[32];
        strncpy(base, env.profile, sizeof base - 1); base[sizeof base - 1] = 0;
        char *dash = strstr(base, "-sg");
        if (dash) { forced_sg = atoi(dash + 3); *dash = 0; }
        coli_sycl_arch a = coli_sycl_arch_from_name(base);
        if (a == COLI_SYCL_ARCH_NONE) {
            say(log, "COLI_SYCL_PROFILE=%s: no such profile (pvc, bmg, lnl, ptl, arl, mtl, dg2, wcl, xelp, gen9, generic), keeping %s",
                env.profile, coli_sycl_arch_name(p.arch));
            forced_sg = 0;
        } else {
            if (a != p.arch) say(log, "COLI_SYCL_PROFILE=%s: taking its decisions on a %s device", env.profile, coli_sycl_arch_name(p.arch));
            p.arch = a;
        }
    }
    p.gen = coli_sycl_gen_of(p.arch);

    /* the sub-group: the architecture's default, then the profile suffix, then COLI_SYCL_SG */
    p.sg_default = default_sg(pr, p.arch, log);
    if (forced_sg) {
        if ((forced_sg == 16 || forced_sg == 32) && offered(pr, forced_sg)) p.sg_default = forced_sg;
        else say(log, "COLI_SYCL_PROFILE=%s: the device does not offer that sub-group size, keeping the default%s", env.profile, "");
    }
    if (env.sg && *env.sg) {
        int sg = atoi(env.sg);
        if ((sg == 16 || sg == 32) && offered(pr, sg)) p.sg_default = sg;
        else say(log, "COLI_SYCL_SG=%s: not 16 or 32, or not offered by %s: ignored", env.sg, p.device_name);
    }

    /* the matrix engines: the f16 8x16x16 shape, the one every kernel of the family runs on, is
     * listed (so "xmx on" means the engines run, not that some shape exists), and sub-group 16
     * (which joint_matrix runs at on every Intel part) is offered; the environment may turn them
     * off, or on where only the sub-group rule said no */
    p.has_xmx = (p.xmx_shapes & COLI_SYCL_XMX_F16) != 0 && offered(pr, 16);
    if (env.xmx && *env.xmx) {
        if (!strcmp(env.xmx, "0")) p.has_xmx = 0;
        else if (!strcmp(env.xmx, "1")) {
            if (p.xmx_shapes & COLI_SYCL_XMX_F16) p.has_xmx = 1;
            else say(log, "COLI_SYCL_XMX=1: %s lists no f16 8x16x16 shape (the kernels' tile), the matrix engines stay off%s", p.device_name, "");
        } else say(log, "COLI_SYCL_XMX=%s: not 0 or 1, ignored%s", env.xmx, "");
    }

    snprintf(p.name, sizeof p.name, "%s-sg%d", coli_sycl_arch_name(p.arch), p.sg_default);
    return p;
}

void coli_sycl_profile_print(FILE *out, int ordinal, const coli_sycl_profile &p) {
    fprintf(out, "device %d: %s\n", ordinal, p.device_name);
    fprintf(out, "  profile %s (arch %s, gen %s)", p.name, coli_sycl_arch_name(p.arch), coli_sycl_gen_name(p.gen));
    if (p.pci_address[0]) fprintf(out, ", pci %s", p.pci_address);
    if (p.pci_device_id) fprintf(out, " id 0x%04x", p.pci_device_id);
    fprintf(out, "%s\n", p.is_integrated ? ", integrated" : "");
    fprintf(out, "  sub-groups:");
    for (int i = 0; i < p.sg_count; i++) fprintf(out, " %d", p.sg_sizes[i]);
    fprintf(out, " (kernels at %d)\n", p.sg_default);
    fprintf(out, "  %d EUs x %d threads, SLM %zu KiB, work-group %zu, memory %.1f GB\n",
            p.eu_count, p.hw_threads_per_eu, p.slm_bytes / 1024, p.max_wg_size, p.global_mem_bytes / 1e9);
    fprintf(out, "  xmx %s:", p.has_xmx ? "on" : "off");
    if (!p.xmx_shapes) fprintf(out, " no shape the kernels take");
    if (p.xmx_shapes & COLI_SYCL_XMX_F16) fprintf(out, " f16-8x16x16");
    if (p.xmx_shapes & COLI_SYCL_XMX_F16_16) fprintf(out, " f16-16x16x16");
    if (p.xmx_shapes & COLI_SYCL_XMX_BF16) fprintf(out, " bf16-8x16x16");
    if (p.xmx_shapes & COLI_SYCL_XMX_INT8) fprintf(out, " s8-8x16x32");
    if (p.xmx_shapes & COLI_SYCL_XMX_TF32) fprintf(out, " tf32-8x16x8");
    fputc('\n', out);
}
