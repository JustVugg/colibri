/* coli_sycl_profile.h — the device profile of the SYCL backend, and how it is derived.
 *
 * Split out of coli_sycl.h so the derivation (the decisions a profile makes from what the device
 * reports: the sub-group the warp-32-style kernels run at, whether the matrix engines are used,
 * the profile's name, the environment overrides) is plain C++ with nothing of SYCL in it.
 * coli_sycl_profile.cpp reads a sycl::device into a coli_sycl_props and hands the values over;
 * tests/test_sycl_profile.cpp hands over fake sets (pvc, bmg, lnl, an unknown part) and checks
 * the decisions in a process that has no SYCL runtime in it (coli_sycl_profile_rules.cpp is the
 * implementation, compiled by any C++17 compiler). Nothing here touches a device. */
#ifndef COLI_SYCL_PROFILE_H
#define COLI_SYCL_PROFILE_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

/* the architecture, from ext::oneapi::experimental::info::device::architecture; a value this
 * header has no row for is COLI_SYCL_ARCH_OTHER (an Intel GPU of an unknown generation: the
 * runtime's own architecture::unknown for a part its table lacks, or a newer value on a device
 * whose vendor id is Intel's) or COLI_SYCL_ARCH_NONE (not an Intel GPU: a CPU device, another
 * vendor). Every intel_gpu_* value of oneAPI 2025.3's device_architecture.def has a row
 * (coli_sycl_profile.cpp maps them). */
enum coli_sycl_arch {
    COLI_SYCL_ARCH_NONE = 0,
    COLI_SYCL_ARCH_PVC,      /* intel_gpu_pvc, intel_gpu_pvc_vg (Data Center GPU Max) */
    COLI_SYCL_ARCH_BMG,      /* intel_gpu_bmg_g21, intel_gpu_bmg_g31 (Arc B-series, Arc Pro B) */
    COLI_SYCL_ARCH_LNL,      /* intel_gpu_lnl_m (Lunar Lake) */
    COLI_SYCL_ARCH_PTL,      /* intel_gpu_ptl_h, intel_gpu_ptl_u (Panther Lake) */
    COLI_SYCL_ARCH_ARL,      /* intel_gpu_arl_h (Arrow Lake H; arl_u / arl_s are mtl_u aliases) */
    COLI_SYCL_ARCH_MTL,      /* intel_gpu_mtl_h, intel_gpu_mtl_u (Meteor Lake, Arrow Lake U/S) */
    COLI_SYCL_ARCH_DG2,      /* intel_gpu_acm_g10 / g11 / g12 (Alchemist, "dg2") */
    COLI_SYCL_ARCH_WCL,      /* intel_gpu_wcl (Wildcat Lake, Xe3) */
    COLI_SYCL_ARCH_XELP,     /* intel_gpu_tgllp, rkl, adl_s / p / n, dg1 (Xe-LP, gen 12) */
    COLI_SYCL_ARCH_GEN9,     /* intel_gpu_bdw .. cml, icllp, ehl (gen 8, 9 and 11) */
    COLI_SYCL_ARCH_OTHER     /* an Intel GPU this header does not know: the generic profile */
};
/* the generation, which is what the kernels' choices key on */
enum coli_sycl_gen {
    COLI_SYCL_GEN_UNKNOWN = 0,
    COLI_SYCL_GEN_XE,        /* Xe-HPG / Xe-LPG: dg2, mtl, arl */
    COLI_SYCL_GEN_PVC,       /* Xe-HPC */
    COLI_SYCL_GEN_XE2,       /* bmg, lnl */
    COLI_SYCL_GEN_XE3,       /* ptl, wcl; treated as xe2 by every decision until measured */
    COLI_SYCL_GEN_LEGACY     /* gen 8 .. 12 LP (no matrix engines): the generic decisions */
};
/* the XMX shapes a joint_matrix kernel may take, from matrix_combinations (sub-group 16 on every
 * known part). The runtime lists M as a range up to 8 with N and K fixed for the 8xNxK rows of
 * pvc and the Xe2 parts (f16 / bf16 8x16x16, s8 8x16x32, tf32 8x16x8), and the compound f16
 * 16x16x16 as a fixed row (the spec also lists bf16 1x64x16 and 32x64x16, which no kernel here
 * takes). */
enum {
    COLI_SYCL_XMX_F16  = 1u << 0,   /* f16 x f16 -> f32, 8x16x16 */
    COLI_SYCL_XMX_BF16 = 1u << 1,   /* bf16 x bf16 -> f32, 8x16x16 */
    COLI_SYCL_XMX_INT8 = 1u << 2,   /* s8 x s8 -> s32, 8x16x32 */
    COLI_SYCL_XMX_TF32 = 1u << 3,   /* tf32 x tf32 -> f32, 8x16x8 */
    COLI_SYCL_XMX_F16_16 = 1u << 4  /* the compound f16 16x16x16 (pvc, bmg): the WMMA tile shape */
};
/* the operand type a caller asks coli_sycl_xmx_supported about; the accumulator types after
 * them only name a matrix_combinations entry's C and D (coli_sycl_combo) */
enum coli_sycl_dtype {
    COLI_SYCL_DT_F16 = 0, COLI_SYCL_DT_BF16, COLI_SYCL_DT_INT8, COLI_SYCL_DT_TF32,
    COLI_SYCL_DT_F32, COLI_SYCL_DT_S32, COLI_SYCL_DT_OTHER
};

struct coli_sycl_profile {
    coli_sycl_arch arch;
    coli_sycl_gen gen;
    char name[32];             /* the profile's name: pvc-sg32, bmg-sg16, lnl-sg16, generic-sg16 ... (COLI_SYCL_PROFILE) */
    char device_name[128];     /* info::device::name, for the [CUDA] device line */
    uint32_t pci_device_id;    /* ext::intel::info::device::device_id (0x0bda the Max 1100, 0xe20b the B580 ...) */
    char pci_address[32];      /* ext::intel::info::device::pci_address, the ordering key */
    int sg_default;            /* 16 or 32: the sub-group the warp-32-style kernels launch at */
    int sg_sizes[8];           /* info::device::sub_group_sizes, ascending */
    int sg_count;
    int eu_count;              /* ext::intel::info::device::gpu_eu_count (max_compute_units elsewhere) */
    int hw_threads_per_eu;     /* gpu_hw_threads_per_eu (8 on pvc): eu_count * this = resident sub-groups */
    size_t slm_bytes;          /* info::device::local_mem_size, the device's word: 128 KiB on pvc and the Arc B-series, 64 KiB on the Xe2 iGPUs (the test's fixtures); never assumed, 64 KiB only when the query fails */
    size_t max_wg_size;        /* info::device::max_work_group_size (1024) */
    size_t global_mem_bytes;   /* info::device::global_mem_size */
    int has_xmx;               /* matrix_combinations lists the f16 8x16x16 shape (the kernels' tile), sub-group 16 is offered, and COLI_SYCL_XMX is not 0: the engines run */
    unsigned xmx_shapes;       /* COLI_SYCL_XMX_* */
    int is_integrated;         /* info::device::host_unified_memory: lnl, ptl, arl, mtl */
};

/* ---- what detection reads, before any decision ------------------------------------------- */

/* One matrix_combinations entry in plain terms: each dimension is fixed (m, n or k; its max_ 0)
 * or ranged (max_m, max_n or max_k; its fixed size 0), as the runtime encodes them (Intel's rows
 * are {max_m 8, n 16, k 16 | 32 | 8}: M up to 8, N and K fixed); the operand types a and b, the
 * accumulator c. */
struct coli_sycl_combo {
    size_t max_m, max_n, max_k, m, n, k;
    coli_sycl_dtype a, b, c;
};
/* the COLI_SYCL_XMX_* flags the entries give: a shape flag is set when an entry of the right
 * types takes the kernels' tile (8x16x16 f16 / bf16, 8x16x32 s8, 8x16x8 tf32, 16x16x16 f16 for
 * F16_16); dg2's 8x8 tiles at sub-group 8 set nothing, which is right: the kernels are SG16 */
unsigned coli_sycl_xmx_shape_flags(const coli_sycl_combo *c, int n);

/* The values detection reads (coli_sycl_profile.cpp, one try/catch a query: a query that fails
 * leaves its zero, and the derivation fills a stated default). A test fills one by hand. */
struct coli_sycl_props {
    coli_sycl_arch arch;       /* from the architecture query (OTHER: an Intel GPU with no row, NONE: no Intel GPU) */
    char device_name[128];
    uint32_t pci_device_id;
    char pci_address[32];
    int sg_sizes[8];           /* ascending, at most 8 kept */
    int sg_count;
    int eu_count;              /* gpu_eu_count, 0 when the device has no ext_intel_gpu_eu_count aspect */
    int max_compute_units;     /* info::device::max_compute_units, the fallback for eu_count */
    int hw_threads_per_eu;     /* gpu_hw_threads_per_eu, 0 when absent */
    size_t slm_bytes;
    size_t max_wg_size;
    size_t global_mem_bytes;
    int host_unified_memory;
    unsigned xmx_shapes;       /* coli_sycl_xmx_shape_flags over matrix_combinations (0: none, or no query) */
};

/* the environment's say: NULL where a variable is unset (coli_sycl_env_read takes them from the
 * process; a test passes literals) */
struct coli_sycl_env {
    const char *sg;            /* COLI_SYCL_SG=16|32: the sub-group, where the device offers it */
    const char *xmx;           /* COLI_SYCL_XMX=0|1: the matrix engines off, or on where a shape is listed */
    const char *profile;       /* COLI_SYCL_PROFILE=<name>[-sg16|-sg32]: the named profile's decisions on this device */
};
coli_sycl_env coli_sycl_env_read(void);

/* The default an architecture's measurements give a switch the environment leaves unset: the value
 * measured best on that architecture where there is a measurement, else `generic` (the code's own
 * default). Only choices that leave the output bits as they are sit in the table -- a kernel path,
 * a launch shape, a sub-group, a placement pin -- each measured on the architecture it names with
 * the texts identical to the switch off. `platform_defaults` is COLI_PLATFORM_DEFAULTS (NULL when
 * unset; a test passes literals): "0" returns `generic` for every switch. The callers read the
 * switch's own variable first: an explicit value always wins over this. Rows
 * (coli_sycl_profile_rules.cpp): on pvc (Data Center GPU Max) COLI_CUDA_ATTN_GROUPED_PLAIN 32,
 * COLI_CUDA_ATTN_GROUPED_VEC 2, COLI_CUDA_DNREC_PROMPT_SG 16, COLI_CUDA_GEMV_XPAD 2,
 * COLI_CUDA_MUX_BATCH 15, COLI_TRUNK_PROBE 0. */
int coli_sycl_tuned_default(const coli_sycl_profile &p, const char *platform_defaults, const char *knob, int generic);

/* The derivation: the profile of a device described by `pr`, with `env` applied. Never fails:
 * an unknown architecture takes the generic decisions from the values alone, a value the query
 * left at zero takes a stated default, an override the device cannot honour is reported on `log`
 * (stderr in the backend; NULL keeps it quiet) and dropped. The rules:
 *   sg_default   32 on pvc, 16 everywhere else, each only where offered; else 16 when offered,
 *                else 32 when offered, else 16 (the kernels' only instances), said on `log`;
 *   has_xmx      the f16 8x16x16 flag is set (the kernels' tile) and sub-group 16 is offered
 *                (joint_matrix runs at 16); COLI_SYCL_XMX=0 clears it, =1 sets it only when
 *                that shape is listed;
 *   name         "<arch>-sg<N>": pvc-sg32, bmg-sg16, generic-sg16 ...;
 *   COLI_SYCL_PROFILE=<arch> takes that architecture's decisions (and its arch / gen tags, which
 *                is what the kernels key on) over the measured values; an unknown name is logged
 *                and ignored; a -sgN suffix sets the sub-group, COLI_SYCL_SG wins over it. */
coli_sycl_profile coli_sycl_profile_derive(const coli_sycl_props &pr, const coli_sycl_env &env, FILE *log);

/* names: the architecture's short name (pvc, bmg, ..., generic, none), its parse (the base of
 * COLI_SYCL_PROFILE; NONE for a name with no row), the generation's, and the generation an
 * architecture belongs to */
const char *coli_sycl_arch_name(coli_sycl_arch a);
coli_sycl_arch coli_sycl_arch_from_name(const char *name);
const char *coli_sycl_gen_name(coli_sycl_gen g);
coli_sycl_gen coli_sycl_gen_of(coli_sycl_arch a);
static inline int coli_sycl_sg_offered(const coli_sycl_profile &p, int sg) {
    for (int i = 0; i < p.sg_count; i++) if (p.sg_sizes[i] == sg) return 1;
    return 0;
}
/* whether the profile's matrix engines take operands of `dt`: has_xmx and the shape's flag */
bool coli_sycl_xmx_supported(const coli_sycl_profile &p, coli_sycl_dtype dt);
/* one device's profile, a few lines on `out` (COLI_SYCL_LIST=1's format): "device <ordinal>: ..." */
void coli_sycl_profile_print(FILE *out, int ordinal, const coli_sycl_profile &p);

#endif /* COLI_SYCL_PROFILE_H */
