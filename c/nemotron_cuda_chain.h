/* nemotron_cuda_chain.h -- End-to-end GPU execution chain + VRAM LRU expert
 * tiering for Nemotron-H / Nemotron-3.5-Super-VL-120B-A12B.
 *
 * Included by nemotron.c after NemoModel, NemoState, and nemo_get_expert.
 * Loads coli_cuda_nemotron.dll at runtime on Windows (or links directly when
 * compiled with -DCOLI_CUDA on Linux/macOS). Falls back cleanly to CPU if no
 * CUDA DLL or GPU is available.
 */
#ifndef COLIBRI_NEMOTRON_CUDA_CHAIN_H
#define COLIBRI_NEMOTRON_CUDA_CHAIN_H

#include "backend_cuda_nemotron.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef struct {
    int fmt, O, I;
    void *d_data;
    uint8_t *d_scales;
    float scale_scalar;
} NemoCuMat;

typedef struct {
    int expert_id;
    uint64_t stamp;
    void *d_up_data, *d_down_data;
    uint8_t *d_up_sc, *d_down_sc;
    float up_scale2, down_scale2;
    int fmt;
} NemoCuExpertSlot;

typedef struct {
    int cap;
    uint64_t clock;
    int exp_in, moe_inter;
    int fmt;
    size_t up_wbytes, up_sbytes, dn_wbytes, dn_sbytes;
    uint8_t *d_arena; /* single contiguous VRAM slab for all cap slots */
    NemoCuExpertSlot *slots;
    int *lut; /* [n_experts] -> slot idx or -1 */
} NemoCuExpertCache;

typedef struct {
    int type;
    float *d_norm_w;
    /* Mamba-2 */
    NemoCuMat in_proj, out_proj;
    float *d_conv_w, *d_conv_b, *d_dt_bias, *d_A_neg_exp, *d_D, *d_mamba_norm;
    /* Attention */
    NemoCuMat q_proj, k_proj, v_proj, o_proj;
    /* MoE */
    float *d_gate_w;
    NemoCuMat fc1_lat, fc2_lat;
    NemoCuMat sh_up, sh_down;
    NemoCuExpertCache vcache;
    /* MLP */
    NemoCuMat mlp_up, mlp_down;
} NemoCuLayer;

typedef struct {
    const ColiNemoCudaOps *ops;
    void *dl_handle;
    int active;
    int max_ctx;

    float *d_norm_f;
    NemoCuMat lm_head;
    NemoCuLayer *layers;

    /* GPU state */
    float **d_k_cache, **d_v_cache;
    float **d_conv_state, **d_ssm_state;

    /* GPU scratch */
    float *d_x, *d_xn, *d_res, *d_logits;
    float *d_mb_proj, *d_mb_xbc, *d_mb_y;
    float *d_q, *d_k, *d_v, *d_ao;
    float *d_lat_in, *d_lat_out, *d_exp_h, *d_sh_h, *d_sh_out;
    float *d_rlogits;

    uint64_t vram_hits, vram_misses;
} NemoCuda;

static size_t nemo_mat_wbytes(const NemoMat *m) {
    if (m->fmt == NEMO_FMT_F32)   return (size_t)m->O * m->I * 4;
    if (m->fmt == NEMO_FMT_BF16)  return (size_t)m->O * m->I * 2;
    if (m->fmt == NEMO_FMT_FP8)   return (size_t)m->O * m->I;
    return (size_t)m->O * (m->I / 2);
}

static size_t nemo_mat_sbytes(const NemoMat *m) {
    return (m->fmt == NEMO_FMT_NVFP4) ? (size_t)m->O * (m->I / 16) : 0;
}

static int nemo_cu_upload_mat(const ColiNemoCudaOps *ops, NemoCuMat *dst, const NemoMat *src) {
    memset(dst, 0, sizeof(*dst));
    dst->fmt = src->fmt; dst->O = src->O; dst->I = src->I;
    dst->scale_scalar = src->scale_scalar;
    size_t wb = nemo_mat_wbytes(src);
    dst->d_data = ops->upload(src->data, wb);
    if (!dst->d_data) return -1;
    size_t sb = nemo_mat_sbytes(src);
    if (sb > 0) {
        dst->d_scales = (uint8_t *)ops->upload(src->scales, sb);
        if (!dst->d_scales) return -1;
    }
    return 0;
}

static const ColiNemoCudaOps *nemo_cuda_load_ops(void **handle_out) {
    *handle_out = NULL;
#if defined(COLI_CUDA) && !defined(_WIN32)
    return coli_cuda_nemotron_ops();
#elif defined(_WIN32)
    const char *env = getenv("COLI_NEMOTRON_CUDA_DLL");
    HMODULE h = NULL;
    if (env && env[0]) h = LoadLibraryA(env);
    if (!h) {
        /* Try next to the executable first */
        char exe[MAX_PATH];
        DWORD n = GetModuleFileNameA(NULL, exe, MAX_PATH);
        if (n > 0 && n < MAX_PATH) {
            char *sl = strrchr(exe, '\\');
            if (!sl) sl = strrchr(exe, '/');
            if (sl) {
                snprintf(sl + 1, (size_t)(MAX_PATH - (sl + 1 - exe)), "coli_cuda_nemotron.dll");
                h = LoadLibraryA(exe);
            }
        }
    }
    if (!h) h = LoadLibraryA("coli_cuda_nemotron.dll");
    if (!h) return NULL;
    typedef const ColiNemoCudaOps *(*ops_fn)(void);
    ops_fn fn = (ops_fn)(void *)GetProcAddress(h, "coli_cuda_nemotron_ops");
    if (!fn) { FreeLibrary(h); return NULL; }
    *handle_out = (void *)h;
    return fn();
#else
    const char *env = getenv("COLI_NEMOTRON_CUDA_DLL");
    void *h = dlopen(env && env[0] ? env : "./libcoli_cuda_nemotron.so", RTLD_NOW);
    if (!h) return NULL;
    typedef const ColiNemoCudaOps *(*ops_fn)(void);
    ops_fn fn = (ops_fn)dlsym(h, "coli_cuda_nemotron_ops");
    if (!fn) { dlclose(h); return NULL; }
    *handle_out = h;
    return fn();
#endif
}

static int nemo_cuda_init(NemoCuda *cu, NemoModel *m, int max_ctx, int dev, int vram_slots_override) {
    memset(cu, 0, sizeof(*cu));
    cu->ops = nemo_cuda_load_ops(&cu->dl_handle);
    if (!cu->ops) return -1;
    if (cu->ops->init(dev) != 0) return -1;
    const ColiNemoCudaOps *ops = cu->ops;
    const NemoCfg *c = &m->c;
    cu->max_ctx = max_ctx;

    size_t free0 = 0, total0 = 0;
    ops->mem_info(&free0, &total0);

    /* Upload dense trunk */
    cu->d_norm_f = (float *)ops->upload(m->norm_f, (size_t)c->hidden * sizeof(float));
    if (!cu->d_norm_f || nemo_cu_upload_mat(ops, &cu->lm_head, &m->lm_head) != 0) return -1;

    cu->layers = (NemoCuLayer *)xcalloc((size_t)c->n_layers, sizeof(NemoCuLayer), "cu_layers");
    for (int i = 0; i < c->n_layers; i++) {
        const NemoLayer *L = &m->layers[i];
        NemoCuLayer *CL = &cu->layers[i];
        CL->type = L->type;
        CL->d_norm_w = (float *)ops->upload(L->norm_w, (size_t)c->hidden * sizeof(float));
        if (!CL->d_norm_w) return -1;
        if (L->type == NEMO_BLK_MAMBA) {
            if (nemo_cu_upload_mat(ops, &CL->in_proj, &L->in_proj) != 0) return -1;
            if (nemo_cu_upload_mat(ops, &CL->out_proj, &L->out_proj) != 0) return -1;
            CL->d_conv_w = (float *)ops->upload(L->conv_w, (size_t)c->mamba_conv_dim * c->conv_kernel * sizeof(float));
            if (L->conv_b) CL->d_conv_b = (float *)ops->upload(L->conv_b, (size_t)c->mamba_conv_dim * sizeof(float));
            CL->d_dt_bias   = (float *)ops->upload(L->dt_bias, (size_t)c->mamba_heads * sizeof(float));
            CL->d_A_neg_exp = (float *)ops->upload(L->A_neg_exp, (size_t)c->mamba_heads * sizeof(float));
            CL->d_D         = (float *)ops->upload(L->D, (size_t)c->mamba_heads * sizeof(float));
            CL->d_mamba_norm = (float *)ops->upload(L->mamba_norm, (size_t)c->mamba_d_inner * sizeof(float));
        } else if (L->type == NEMO_BLK_ATTN) {
            if (nemo_cu_upload_mat(ops, &CL->q_proj, &L->q_proj) != 0) return -1;
            if (nemo_cu_upload_mat(ops, &CL->k_proj, &L->k_proj) != 0) return -1;
            if (nemo_cu_upload_mat(ops, &CL->v_proj, &L->v_proj) != 0) return -1;
            if (nemo_cu_upload_mat(ops, &CL->o_proj, &L->o_proj) != 0) return -1;
        } else if (L->type == NEMO_BLK_MOE) {
            CL->d_gate_w = (float *)ops->upload(L->gate_w, (size_t)c->n_experts * c->hidden * sizeof(float));
            if (L->has_latent) {
                if (nemo_cu_upload_mat(ops, &CL->fc1_lat, &L->fc1_lat) != 0) return -1;
                if (nemo_cu_upload_mat(ops, &CL->fc2_lat, &L->fc2_lat) != 0) return -1;
            }
            if (L->has_shared) {
                if (nemo_cu_upload_mat(ops, &CL->sh_up, &L->sh_up) != 0) return -1;
                if (nemo_cu_upload_mat(ops, &CL->sh_down, &L->sh_down) != 0) return -1;
            }
        } else {
            if (nemo_cu_upload_mat(ops, &CL->mlp_up, &L->mlp_up) != 0) return -1;
            if (nemo_cu_upload_mat(ops, &CL->mlp_down, &L->mlp_down) != 0) return -1;
        }
    }

    /* Allocate GPU state */
    int kv_dim = c->n_kv_heads * c->head_dim;
    if (c->n_attn > 0) {
        cu->d_k_cache = (float **)xcalloc((size_t)c->n_attn, sizeof(float *), "cu_k");
        cu->d_v_cache = (float **)xcalloc((size_t)c->n_attn, sizeof(float *), "cu_v");
        for (int a = 0; a < c->n_attn; a++) {
            cu->d_k_cache[a] = (float *)ops->alloc((size_t)max_ctx * kv_dim * sizeof(float));
            cu->d_v_cache[a] = (float *)ops->alloc((size_t)max_ctx * kv_dim * sizeof(float));
            ops->memset_zero(cu->d_k_cache[a], (size_t)max_ctx * kv_dim * sizeof(float));
            ops->memset_zero(cu->d_v_cache[a], (size_t)max_ctx * kv_dim * sizeof(float));
        }
    }
    if (c->n_mamba > 0) {
        cu->d_conv_state = (float **)xcalloc((size_t)c->n_mamba, sizeof(float *), "cu_cs");
        cu->d_ssm_state  = (float **)xcalloc((size_t)c->n_mamba, sizeof(float *), "cu_ss");
        size_t cs = (size_t)c->mamba_conv_dim * c->conv_kernel * sizeof(float);
        size_t ss = (size_t)c->mamba_heads * c->mamba_head_dim * c->ssm_state * sizeof(float);
        for (int mb = 0; mb < c->n_mamba; mb++) {
            cu->d_conv_state[mb] = (float *)ops->alloc(cs);
            cu->d_ssm_state[mb]  = (float *)ops->alloc(ss);
            ops->memset_zero(cu->d_conv_state[mb], cs);
            ops->memset_zero(cu->d_ssm_state[mb], ss);
        }
    }

    /* Allocate GPU scratch */
    int exp_in = c->moe_latent > 0 ? c->moe_latent : c->hidden;
    int proj_out = c->mamba_d_inner + c->mamba_conv_dim + c->mamba_heads;
    cu->d_x      = (float *)ops->alloc((size_t)c->hidden * sizeof(float));
    cu->d_xn     = (float *)ops->alloc((size_t)c->hidden * sizeof(float));
    cu->d_res    = (float *)ops->alloc((size_t)c->hidden * sizeof(float));
    cu->d_logits = (float *)ops->alloc((size_t)c->vocab * sizeof(float));
    if (c->n_mamba > 0) {
        cu->d_mb_proj = (float *)ops->alloc((size_t)proj_out * sizeof(float));
        cu->d_mb_xbc  = (float *)ops->alloc((size_t)c->mamba_conv_dim * sizeof(float));
        cu->d_mb_y    = (float *)ops->alloc((size_t)c->mamba_d_inner * sizeof(float));
    }
    if (c->n_attn > 0) {
        cu->d_q  = (float *)ops->alloc((size_t)c->n_heads * c->head_dim * sizeof(float));
        cu->d_k  = (float *)ops->alloc((size_t)kv_dim * sizeof(float));
        cu->d_v  = (float *)ops->alloc((size_t)kv_dim * sizeof(float));
        cu->d_ao = (float *)ops->alloc((size_t)c->n_heads * c->head_dim * sizeof(float));
    }
    if (c->n_moe > 0) {
        cu->d_lat_in  = (float *)ops->alloc((size_t)exp_in * sizeof(float));
        cu->d_lat_out = (float *)ops->alloc((size_t)exp_in * sizeof(float));
        cu->d_exp_h   = (float *)ops->alloc((size_t)c->moe_inter * sizeof(float));
        cu->d_rlogits = (float *)ops->alloc((size_t)c->n_experts * sizeof(float));
        if (c->shared_inter > 0) {
            cu->d_sh_h   = (float *)ops->alloc((size_t)c->shared_inter * sizeof(float));
            cu->d_sh_out = (float *)ops->alloc((size_t)c->hidden * sizeof(float));
        }
    } else if (c->n_mlp > 0) {
        cu->d_sh_h = (float *)ops->alloc((size_t)c->inter * sizeof(float));
    }

    /* Size and pre-allocate per-layer VRAM expert LRU cache */
    int vram_ecap = 0;
    if (c->n_moe > 0) {
        /* Probe expert 0 of first MoE layer to get exact format & byte size */
        int first_moe = -1;
        for (int i = 0; i < c->n_layers; i++) if (m->layers[i].type == NEMO_BLK_MOE) { first_moe = i; break; }
        NemoExpertSlot *probe = nemo_get_expert(m, first_moe, 0);
        int efmt = probe->up.fmt;
        size_t up_wb = nemo_mat_wbytes(&probe->up), up_sb = nemo_mat_sbytes(&probe->up);
        size_t dn_wb = nemo_mat_wbytes(&probe->down), dn_sb = nemo_mat_sbytes(&probe->down);

        size_t free1 = 0, total1 = 0;
        ops->mem_info(&free1, &total1);
        /* Leave comfortable headroom for WDDM / desktop / CUDA runtime */
        size_t headroom = (total1 >= (size_t)16 * 1024 * 1024 * 1024ULL)
                        ? (size_t)3072 * 1024 * 1024ULL
                        : ((total1 >= (size_t)8 * 1024 * 1024 * 1024ULL)
                           ? (size_t)1536 * 1024 * 1024ULL
                           : (size_t)256 * 1024 * 1024ULL);
        size_t avail = free1 > headroom ? (free1 - headroom) : 0;
        size_t align = 256;
        size_t up_wa = (up_wb + align - 1) & ~(align - 1);
        size_t up_sa = (up_sb + align - 1) & ~(align - 1);
        size_t dn_wa = (dn_wb + align - 1) & ~(align - 1);
        size_t dn_sa = (dn_sb + align - 1) & ~(align - 1);
        size_t slot_stride = up_wa + up_sa + dn_wa + dn_sa;
        vram_ecap = (int)(avail / ((size_t)c->n_moe * slot_stride));
        if (vram_slots_override > 0) vram_ecap = vram_slots_override;
        if (vram_ecap < c->top_k) vram_ecap = c->top_k;
        if (vram_ecap > c->n_experts) vram_ecap = c->n_experts;

        for (int i = 0; i < c->n_layers; i++) {
            if (m->layers[i].type != NEMO_BLK_MOE) continue;
            NemoCuExpertCache *vc = &cu->layers[i].vcache;
            vc->cap = vram_ecap;
            vc->exp_in = exp_in;
            vc->moe_inter = c->moe_inter;
            vc->fmt = efmt;
            vc->up_wbytes = up_wb; vc->up_sbytes = up_sb;
            vc->dn_wbytes = dn_wb; vc->dn_sbytes = dn_sb;
            vc->d_arena = (uint8_t *)ops->alloc((size_t)vram_ecap * slot_stride);
            if (!vc->d_arena) return -1;
            vc->slots = (NemoCuExpertSlot *)xcalloc((size_t)vram_ecap, sizeof(NemoCuExpertSlot), "vslots");
            vc->lut = (int *)xmalloc((size_t)c->n_experts * sizeof(int), "vlut");
            for (int e = 0; e < c->n_experts; e++) vc->lut[e] = -1;
            for (int s = 0; s < vram_ecap; s++) {
                uint8_t *base = vc->d_arena + (size_t)s * slot_stride;
                vc->slots[s].expert_id = -1;
                vc->slots[s].fmt = efmt;
                vc->slots[s].d_up_data   = base;
                vc->slots[s].d_up_sc     = up_sb > 0 ? (base + up_wa) : NULL;
                vc->slots[s].d_down_data = base + up_wa + up_sa;
                vc->slots[s].d_down_sc   = dn_sb > 0 ? (base + up_wa + up_sa + dn_wa) : NULL;
            }
        }
    }

    ops->sync();
    /* Free CPU copies of dense GPU-resident weights (~7 GB RAM on 120B) */
    if (m->lm_head.data && m->lm_head.data != m->emb.data) nemo_mat_free(&m->lm_head);
    for (int i = 0; i < c->n_layers; i++) {
        NemoLayer *L = &m->layers[i];
        if (L->type == NEMO_BLK_MAMBA) {
            nemo_mat_free(&L->in_proj); nemo_mat_free(&L->out_proj);
        } else if (L->type == NEMO_BLK_ATTN) {
            nemo_mat_free(&L->q_proj); nemo_mat_free(&L->k_proj);
            nemo_mat_free(&L->v_proj); nemo_mat_free(&L->o_proj);
        } else if (L->type == NEMO_BLK_MOE) {
            if (L->has_latent) { nemo_mat_free(&L->fc1_lat); nemo_mat_free(&L->fc2_lat); }
            if (L->has_shared) { nemo_mat_free(&L->sh_up); nemo_mat_free(&L->sh_down); }
        } else {
            nemo_mat_free(&L->mlp_up); nemo_mat_free(&L->mlp_down);
        }
    }
    size_t free2 = 0, total2 = 0;
    ops->mem_info(&free2, &total2);
    cu->active = 1;
    fprintf(stderr, "[nemotron-cuda] GPU ready: used %.2f / %.2f GB VRAM | rss=%.2f GB | VRAM expert tier=%d/%d per MoE layer (%d total slots)\n",
            (double)(total0 - free2) / (1024.0 * 1024.0 * 1024.0),
            (double)total0 / (1024.0 * 1024.0 * 1024.0),
            rss_gb(),
            vram_ecap, c->n_experts, vram_ecap * c->n_moe);
    return 0;
}

static void nemo_cuda_state_clear(NemoCuda *cu, const NemoCfg *c) {
    if (!cu || !cu->active) return;
    const ColiNemoCudaOps *ops = cu->ops;
    size_t cs = (size_t)c->mamba_conv_dim * c->conv_kernel * sizeof(float);
    size_t ss = (size_t)c->mamba_heads * c->mamba_head_dim * c->ssm_state * sizeof(float);
    for (int mb = 0; mb < c->n_mamba; mb++) {
        ops->memset_zero(cu->d_conv_state[mb], cs);
        ops->memset_zero(cu->d_ssm_state[mb], ss);
    }
}

static NemoCuExpertSlot *nemo_cuda_get_expert(NemoCuda *cu, NemoModel *m, int li, int exp_id) {
    NemoCuExpertCache *vc = &cu->layers[li].vcache;
    int s = vc->lut[exp_id];
    if (s >= 0) {
        vc->slots[s].stamp = ++vc->clock;
        cu->vram_hits++;
        return &vc->slots[s];
    }
    cu->vram_misses++;
    int victim = 0;
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < vc->cap; i++) {
        if (vc->slots[i].expert_id < 0) { victim = i; break; }
        if (vc->slots[i].stamp < best) { best = vc->slots[i].stamp; victim = i; }
    }
    NemoCuExpertSlot *vsl = &vc->slots[victim];
    if (vsl->expert_id >= 0) vc->lut[vsl->expert_id] = -1;

    /* Fetch from RAM LRU cache (or safetensors on miss) and stream H2D into preallocated VRAM slot */
    NemoExpertSlot *hsl = nemo_get_expert(m, li, exp_id);
    const ColiNemoCudaOps *ops = cu->ops;
    ops->h2d(vsl->d_up_data, hsl->up.data, vc->up_wbytes);
    if (vc->up_sbytes > 0) ops->h2d(vsl->d_up_sc, hsl->up.scales, vc->up_sbytes);
    ops->h2d(vsl->d_down_data, hsl->down.data, vc->dn_wbytes);
    if (vc->dn_sbytes > 0) ops->h2d(vsl->d_down_sc, hsl->down.scales, vc->dn_sbytes);
    vsl->up_scale2   = hsl->up.scale_scalar;
    vsl->down_scale2 = hsl->down.scale_scalar;
    vsl->expert_id   = exp_id;
    vsl->stamp       = ++vc->clock;
    vc->lut[exp_id]  = victim;
    return vsl;
}

static void nemo_cuda_forward_one(NemoCuda *cu, NemoModel *m, NemoState *st,
                                  int tok, const float *emb_in, int want_logits) {
    const ColiNemoCudaOps *ops = cu->ops;
    const NemoCfg *c = &m->c;
    int D = c->hidden;
    if (st->pos >= cu->max_ctx) {
        fprintf(stderr, "[nemotron-cuda] context overflow (%d >= %d)\n", st->pos, cu->max_ctx);
        exit(1);
    }
    if (emb_in) memcpy(m->x, emb_in, (size_t)D * sizeof(float));
    else nemo_embed_token(m, tok, m->x);
    ops->h2d(cu->d_x, m->x, (size_t)D * sizeof(float));

    for (int li = 0; li < c->n_layers; li++) {
        const NemoLayer *L = &m->layers[li];
        const NemoCuLayer *CL = &cu->layers[li];
        ops->rmsnorm(cu->d_xn, cu->d_x, CL->d_norm_w, D, c->rms_eps);

        if (CL->type == NEMO_BLK_MAMBA) {
            int mi = c->mamba_idx[li];
            int D_in = c->mamba_d_inner, C_dim = c->mamba_conv_dim;
            ops->matvec(cu->d_mb_proj, CL->in_proj.d_data, CL->in_proj.d_scales,
                        CL->in_proj.scale_scalar, CL->in_proj.fmt,
                        CL->in_proj.O, CL->in_proj.I, cu->d_xn, 0, 0.0f);
            ops->mamba_conv_silu(cu->d_mb_xbc, cu->d_conv_state[mi], cu->d_mb_proj + D_in,
                                 CL->d_conv_w, CL->d_conv_b, C_dim, c->conv_kernel);
            ops->mamba_ssm_step(cu->d_mb_y, cu->d_ssm_state[mi], cu->d_mb_xbc,
                                cu->d_mb_proj + D_in + C_dim, CL->d_dt_bias,
                                CL->d_A_neg_exp, CL->d_D,
                                c->mamba_heads, c->mamba_head_dim, c->ssm_state, c->n_groups,
                                c->dt_min, c->dt_max);
            ops->mamba_gated_rmsnorm(cu->d_mb_y, cu->d_mb_proj, CL->d_mamba_norm,
                                     D_in, c->n_groups, c->rms_eps);
            ops->matvec(cu->d_res, CL->out_proj.d_data, CL->out_proj.d_scales,
                        CL->out_proj.scale_scalar, CL->out_proj.fmt,
                        CL->out_proj.O, CL->out_proj.I, cu->d_mb_y, 0, 0.0f);
        } else if (CL->type == NEMO_BLK_ATTN) {
            int ai = c->attn_idx[li];
            ops->matvec(cu->d_q, CL->q_proj.d_data, CL->q_proj.d_scales, CL->q_proj.scale_scalar,
                        CL->q_proj.fmt, CL->q_proj.O, CL->q_proj.I, cu->d_xn, 0, 0.0f);
            ops->matvec(cu->d_k, CL->k_proj.d_data, CL->k_proj.d_scales, CL->k_proj.scale_scalar,
                        CL->k_proj.fmt, CL->k_proj.O, CL->k_proj.I, cu->d_xn, 0, 0.0f);
            ops->matvec(cu->d_v, CL->v_proj.d_data, CL->v_proj.d_scales, CL->v_proj.scale_scalar,
                        CL->v_proj.fmt, CL->v_proj.O, CL->v_proj.I, cu->d_xn, 0, 0.0f);
            ops->attn_nope_gqa(cu->d_ao, cu->d_k_cache[ai], cu->d_v_cache[ai],
                               cu->d_q, cu->d_k, cu->d_v,
                               c->n_heads, c->n_kv_heads, c->head_dim, st->pos);
            ops->matvec(cu->d_res, CL->o_proj.d_data, CL->o_proj.d_scales, CL->o_proj.scale_scalar,
                        CL->o_proj.fmt, CL->o_proj.O, CL->o_proj.I, cu->d_ao, 0, 0.0f);
        } else if (CL->type == NEMO_BLK_MOE) {
            int E = c->n_experts, K = c->top_k;
            int exp_in = L->has_latent ? c->moe_latent : D;

            /* 1. Router matvec on GPU -> small D2H of E floats (2 KB for E=512) */
            ops->matvec(cu->d_rlogits, CL->d_gate_w, NULL, 1.0f, NEMO_FMT_F32, E, D, cu->d_xn, 0, 0.0f);
            /* Kick off latent projection + shared expert while router D2H completes */
            if (L->has_latent) {
                ops->matvec(cu->d_lat_in, CL->fc1_lat.d_data, CL->fc1_lat.d_scales,
                            CL->fc1_lat.scale_scalar, CL->fc1_lat.fmt,
                            CL->fc1_lat.O, CL->fc1_lat.I, cu->d_xn, 0, 0.0f);
            }
            if (L->has_shared) {
                ops->matvec(cu->d_sh_h, CL->sh_up.d_data, CL->sh_up.d_scales,
                            CL->sh_up.scale_scalar, CL->sh_up.fmt,
                            CL->sh_up.O, CL->sh_up.I, cu->d_xn, 1, 0.0f);
                ops->matvec(cu->d_sh_out, CL->sh_down.d_data, CL->sh_down.d_scales,
                            CL->sh_down.scale_scalar, CL->sh_down.fmt,
                            CL->sh_down.O, CL->sh_down.I, cu->d_sh_h, 0, 0.0f);
            }
            ops->d2h(m->router_logits, cu->d_rlogits, (size_t)E * sizeof(float));

            /* Exact CPU top-k routing (matches CPU path bit-for-bit) */
            float scores[1024], choice[1024];
            for (int e = 0; e < E; e++) {
                float sc = 1.0f / (1.0f + expf(-m->router_logits[e]));
                scores[e] = sc;
                choice[e] = sc + (L->gate_bias ? L->gate_bias[e] : 0.0f);
            }
            if (c->n_group > 1 && c->topk_group < c->n_group) {
                int epg = E / c->n_group;
                float gs[64];
                for (int g = 0; g < c->n_group; g++) {
                    float m1 = -1e30f, m2 = -1e30f;
                    for (int j = 0; j < epg; j++) {
                        float v = choice[g * epg + j];
                        if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) m2 = v;
                    }
                    gs[g] = m1 + m2;
                }
                int keep[64] = {0};
                for (int k = 0; k < c->topk_group; k++) {
                    int bi = -1; float bv = -1e30f;
                    for (int g = 0; g < c->n_group; g++)
                        if (!keep[g] && gs[g] > bv) { bv = gs[g]; bi = g; }
                    if (bi >= 0) keep[bi] = 1;
                }
                for (int g = 0; g < c->n_group; g++)
                    if (!keep[g]) for (int j = 0; j < epg; j++) choice[g * epg + j] = -1e30f;
            }
            int top_idx[NEMO_MAX_TOPK];
            float top_w[NEMO_MAX_TOPK];
            int used[1024] = {0};
            float wsum = 0.0f;
            for (int k = 0; k < K; k++) {
                int bi = 0; float bv = -1e30f;
                for (int e = 0; e < E; e++)
                    if (!used[e] && choice[e] > bv) { bv = choice[e]; bi = e; }
                used[bi] = 1;
                top_idx[k] = bi;
                top_w[k] = scores[bi];
                wsum += scores[bi];
            }
            if (c->norm_topk_prob) {
                float inv = 1.0f / (wsum + 1e-20f);
                for (int k = 0; k < K; k++) top_w[k] *= inv;
            }
            for (int k = 0; k < K; k++) top_w[k] *= c->routed_scale;

            /* 2. Routed experts in VRAM (LRU tier) */
            const float *d_ein = L->has_latent ? cu->d_lat_in : cu->d_xn;
            float *d_eout = L->has_latent ? cu->d_lat_out : cu->d_res;
            ops->memset_zero(d_eout, (size_t)exp_in * sizeof(float));
            for (int k = 0; k < K; k++) {
                NemoCuExpertSlot *vsl = nemo_cuda_get_expert(cu, m, li, top_idx[k]);
                ops->matvec(cu->d_exp_h, vsl->d_up_data, vsl->d_up_sc, vsl->up_scale2,
                            vsl->fmt, c->moe_inter, exp_in, d_ein, 1, 0.0f);
                ops->matvec(d_eout, vsl->d_down_data, vsl->d_down_sc, vsl->down_scale2,
                            vsl->fmt, exp_in, c->moe_inter, cu->d_exp_h, 0, top_w[k]);
            }
            if (L->has_latent) {
                ops->matvec(cu->d_res, CL->fc2_lat.d_data, CL->fc2_lat.d_scales,
                            CL->fc2_lat.scale_scalar, CL->fc2_lat.fmt,
                            CL->fc2_lat.O, CL->fc2_lat.I, cu->d_lat_out, 0, 0.0f);
            }
            if (L->has_shared) ops->add_inplace(cu->d_res, cu->d_sh_out, D);
        } else {
            ops->matvec(cu->d_sh_h, CL->mlp_up.d_data, CL->mlp_up.d_scales,
                        CL->mlp_up.scale_scalar, CL->mlp_up.fmt,
                        CL->mlp_up.O, CL->mlp_up.I, cu->d_xn, 1, 0.0f);
            ops->matvec(cu->d_res, CL->mlp_down.d_data, CL->mlp_down.d_scales,
                        CL->mlp_down.scale_scalar, CL->mlp_down.fmt,
                        CL->mlp_down.O, CL->mlp_down.I, cu->d_sh_h, 0, 0.0f);
        }

        ops->add_inplace(cu->d_x, cu->d_res, D);
    }

    st->pos++;
    if (want_logits) {
        ops->rmsnorm(cu->d_xn, cu->d_x, cu->d_norm_f, D, c->rms_eps);
        ops->matvec(cu->d_logits, cu->lm_head.d_data, cu->lm_head.d_scales,
                    cu->lm_head.scale_scalar, cu->lm_head.fmt,
                    cu->lm_head.O, cu->lm_head.I, cu->d_xn, 0, 0.0f);
        ops->d2h(m->logits, cu->d_logits, (size_t)c->vocab * sizeof(float));
    }
}

#endif
