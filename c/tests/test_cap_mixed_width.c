/* test_cap_mixed_width.c -- #856: a mixed-width container must not price every
 * expert row at the container's widest width.
 *
 * GLM-5.2 ships as int4 routed experts with an int8 MTP head. #793 made
 * expert_bytes_probe() return the WIDEST width in the container, which is right
 * for the shared ws[64] working set and wrong for the per-layer LRU caches --
 * each of which holds only its own row's experts. cap_for_ram() divided by it,
 * so the cache halved: 154 -> 77 slots per row on the reporter's box, with the
 * 5,852 experts that left RAM reappearing one-for-one on disk.
 *
 * Two things have to hold, and this exercises both against a REAL container
 * written to disk and indexed by st_init -- fabricated widths, not fabricated
 * arithmetic:
 *
 *   1. the budget divisor is the SUM of each row's real width
 *   2. a slot that once held a wide expert SHRINKS when it is reused for a
 *      narrow row -- otherwise the wide slab migrates through the ws[]<->LRU
 *      swap into narrow rows and (1) becomes a lie the allocator disproves
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* Small enough to write, structured exactly like the real thing: sparse layers
 * 1..4 are int4 (half a byte per weight + one f32 scale per row), layer 5 is the
 * MTP head at int8 (one byte per weight). Ratio 2:1 on the weights, which is the
 * ratio that produced the exact halving in the field. */
/* Big enough that the shrink's 25% relative threshold dominates its 64 KB floor:
 * a toy-sized expert is smaller than the anti-churn floor, so nothing would ever
 * shrink and the test would pass by never exercising the path. */
enum { N_LAYERS = 5, FIRST_DENSE = 1, N_EXPERTS = 2, O = 384, I = 768 };

static const char *SUF[3] = { "gate_proj", "up_proj", "down_proj" };

typedef struct { char name[128]; int64_t nbytes; const char *dtype; } Ent;
static Ent ents[512]; static int nent;

static void add(const char *fmt, int64_t nbytes, const char *dtype, int l, int e, const char *suf){
    Ent *t=&ents[nent++];
    snprintf(t->name,sizeof t->name,fmt,l,e,suf);
    t->nbytes=nbytes; t->dtype=dtype;
}

/* gate/up are [moe_inter, hidden]; down is [hidden, moe_inter] -- transposed, so
 * its row count (and therefore its per-row scale array) differs. Getting this
 * wrong makes the loader reject the container, which is itself a useful signal
 * that the fixture is shaped like a real one.
 * int4: rows*(cols+1)/2 weight bytes + rows f32 scales.  int8: rows*cols + rows f32.
 * Exactly the arithmetic expert_bytes_layer() sums out of the headers. */
static void add_expert_dims(int l, int e, int int8, int inter, int hid){
    for(int k=0;k<3;k++){
        int rows = (k==2) ? hid : inter, cols = (k==2) ? inter : hid;   /* k==2 is down_proj */
        add("model.layers.%d.mlp.experts.%d.%s.weight",
            int8 ? (int64_t)rows*cols : (int64_t)rows*((cols+1)/2), "U8", l, e, SUF[k]);
        add("model.layers.%d.mlp.experts.%d.%s.weight.qs",
            (int64_t)rows*4, "F32", l, e, SUF[k]);
    }
}
static void add_expert(int l, int e, int int8){ add_expert_dims(l,e,int8,O,I); }

/* Flush whatever add_expert queued into a real single-shard container. st_init
 * refuses data_offsets past EOF, so the payload has to exist -- no sparse
 * shortcut: ftruncate is free on ext4 and is NOT free on the NTFS runner. */
static int write_ents(const char *dir){
    char path[256]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    char *hdr=malloc(1<<20); int hn=0; int64_t off=0;
    hn+=sprintf(hdr+hn,"{");
    for(int i=0;i<nent;i++){
        hn+=sprintf(hdr+hn,"%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
            i?",":"", ents[i].name, ents[i].dtype,
            (long long)ents[i].nbytes/(ents[i].dtype[0]=='F'?4:1),
            (long long)off,(long long)(off+ents[i].nbytes));
        off+=ents[i].nbytes;
    }
    hn+=sprintf(hdr+hn,"}");
    while(hn%8) hdr[hn++]=' ';

    FILE *f=fopen(path,"wb"); if(!f){ free(hdr); return 0; }
    uint64_t hl=(uint64_t)hn; fwrite(&hl,8,1,f); fwrite(hdr,1,(size_t)hn,f);
    void *zero=calloc(1,65536);
    for(int64_t left=off;left>0;){ size_t n=left>65536?65536:(size_t)left; fwrite(zero,1,n,f); left-=(int64_t)n; }
    free(zero); free(hdr); fclose(f);
    return 1;
}

static int write_container(const char *dir){
    nent=0;
    for(int l=FIRST_DENSE;l<N_LAYERS;l++)
        for(int e=0;e<N_EXPERTS;e++) add_expert(l,e,0);          /* routed rows: int4 */
    for(int e=0;e<N_EXPERTS;e++) add_expert(N_LAYERS,e,1);       /* MTP row:     int8 */
    return write_ents(dir);
}

/* GLM-5.2's real per-expert widths, from its config.json: hidden 6144,
 * moe_intermediate 2048, first_k_dense_replace 3, 78 layers, MTP head at int8.
 * Only expert 0 of one routed layer and of the MTP layer are written, because
 * expert_bytes_layer() probes exactly those; the other 74 routed rows fall back
 * to tbytes(), which yields the identical width. 54 MB, and it is the only way
 * to make the probe return the int8 head -- which is the whole defect. */
static int write_glm_probe_container(const char *dir){
    nent=0;
    add_expert_dims(3,0,0,2048,6144);      /* routed row, int4  -> 18,915,328 B */
    add_expert_dims(78,0,1,2048,6144);     /* MTP head,   int8  -> 37,789,696 B */
    return write_ents(dir);
}

int main(void){
    const char *dir="tests/tmp_cap_mixed";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    if(!write_container(dir)){ printf("FAIL: could not write the fixture container\n"); return 1; }

    Model m; memset(&m,0,sizeof m);
    Cfg *c=&m.c;
    c->n_layers=N_LAYERS; c->first_dense=FIRST_DENSE; c->n_experts=N_EXPERTS;
    c->hidden=I; c->moe_inter=O;
    m.ebits=4; m.has_mtp=1;
    m.L=calloc(N_LAYERS+1,sizeof(Layer));
    for(int i=FIRST_DENSE;i<N_LAYERS;i++) m.L[i].sparse=1;
    st_init(&m.S,dir);        /* void: it exits on a malformed container */
    if(!st_find(&m.S,"model.layers.1.mlp.experts.0.gate_proj.weight")){
        printf("FAIL: st_init did not index the fixture\n"); return 1; }

    /* ---- A. GLM-5.2's REAL geometry, no container ------------------------- */
    /* expert_bytes_row falls back to the tbytes() formula when a header is absent,
     * so the published config alone reproduces the shipped widths -- and this is
     * the only coverage that fallback branch has. Numbers from GLM-5.2's
     * config.json: 78 layers, first_k_dense_replace 3, hidden 6144,
     * moe_intermediate 2048, num_nextn_predict_layers 1. */
    {
        const char *gdir="tests/tmp_cap_glm";
#ifdef _WIN32
        mkdir(gdir);
#else
        mkdir(gdir,0755);
#endif
        if(!write_glm_probe_container(gdir)){ printf("FAIL: GLM fixture\n"); return 1; }
        Model g; memset(&g,0,sizeof g);
        g.c.n_layers=78; g.c.first_dense=3; g.c.n_experts=256;
        g.c.hidden=6144; g.c.moe_inter=2048; g.ebits=4; g.has_mtp=1;
        g.L=calloc(79,sizeof(Layer));
        for(int i=3;i<78;i++) g.L[i].sparse=1;                  /* 75 MoE rows */
        st_init(&g.S,gdir);
        int grows=0; for(int i=0;i<78;i++) if(g.L[i].sparse) grows++;
        int64_t gnarrow=expert_bytes_row(&g,3,4);               /* from the header */
        int64_t gwide  =expert_bytes_row(&g,78,4);              /* from the header */
        int64_t gfall  =expert_bytes_row(&g,40,4);              /* no header: tbytes fallback */
        int64_t gprobe =expert_bytes_probe(&g,4);
        printf("A. GLM-5.2: %d MoE rows + MTP = %d | int4 %lld B | int8 MTP %lld B | probe %lld B\n",
            grows,grows+1,(long long)gnarrow,(long long)gwide,(long long)gprobe);
        CHECK(grows==75);                       /* 76 rows, exactly what the report showed */
        CHECK(gnarrow==18915328);               /* the 18.9 MB int4 expert the docs quote */
        CHECK(gwide  ==37789696);               /* the 37.79 MB int8 MTP measured on the A6000 */
        CHECK(gfall  ==gnarrow);                /* headerless rows agree with the headers */
        CHECK(gprobe ==gwide);                  /* the probe is the container maximum */

        /* Cost of ADDING the MTP row -- the whole of #856. Measured twice the same
         * way, so nothing here duplicates cap_for_ram's slack formula. */
        g_mem_avail_boot=512.0;
        g.has_mtp=0; g.resident_bytes=0; g.ecap=1<<20; cap_for_ram(&g,242.0,4,4096);
        int cap_no=g.ecap;
        g.has_mtp=1; g.resident_bytes=0; g.ecap=1<<20; cap_for_ram(&g,242.0,4,4096);
        int cap_yes=g.ecap;
        printf("A. cap %d -> %d on adding one MTP row to 75 (%.1f%% lost)\n",
            cap_no,cap_yes,100.0*(1.0-(double)cap_yes/(double)cap_no));
        CHECK(cap_no>0 && cap_yes>0);
        /* One row in 76 costs ~2.6%. v1.5.0 charged ~51% -- the halving. */
        CHECK((double)cap_yes/(double)cap_no > 0.95);
        /* And the reporter's shape: from the SAME budget, the old divisor yields
         * half the slots. 11,704 experts in RAM became 5,852; 154/row became 77. */
        double d_new=(double)grows*(double)gnarrow+(double)gwide;
        double d_150=(double)(grows+2)*(double)gprobe;
        CHECK(fabs(d_150/d_new-2.0)<0.01);
        CHECK((int)(154.0*d_new/d_150)==77);
        free(g.L);
        { char pth[256]; snprintf(pth,sizeof pth,"%s/model.safetensors",gdir);
          remove(pth); rmdir(gdir); }
    }

    /* ---- the widths the container really holds ---------------------------- */
    int64_t narrow = expert_bytes_row(&m,FIRST_DENSE,m.ebits);
    int64_t wide   = expert_bytes_row(&m,N_LAYERS,m.ebits);
    int64_t probe  = expert_bytes_probe(&m,m.ebits);
    printf("  routed row  %lld B\n  MTP row     %lld B\n  probe(max)  %lld B\n",
        (long long)narrow,(long long)wide,(long long)probe);
    CHECK(narrow > 0);
    CHECK(wide > narrow);                    /* int8 head is wider than int4 rows */
    CHECK(probe == wide);                    /* the probe is the container maximum */

    /* ---- (1) the divisor is the sum of the REAL widths --------------------- */
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m.L[i].sparse) nsp++;
    double row_b = expert_cache_row_bytes(&m,m.ebits);
    double want  = (double)nsp*(double)narrow + (double)wide;
    double v150  = (double)(nsp+2)*(double)probe;    /* what v1.5.0 divided by */
    printf("  divisor now %.0f B  (v1.5.0 used %.0f B, %.2fx)\n", row_b, v150, v150/row_b);
    CHECK(fabs(row_b-want) < 1.0);
    CHECK(v150 > row_b);                     /* v1.5.0 over-charged, hence the halving */

    /* The cap is avail/divisor, so the ratio of divisors IS the ratio of caps.
     * Four int4 rows charged as int8 against one genuinely int8 row lands just
     * under 2x -- the scales are the same size at both widths, so the weights
     * double and the container does not quite. That is the halving the reporter
     * measured (154 -> 77) at this fixture's scale. */
    CHECK(v150 > 1.8*row_b);

    /* expert_cache_bytes_per_slot feeds the autopin LRU reserve; it must agree
     * with the cap's divisor or pinning re-introduces the same shortfall. */
    CHECK(fabs(expert_cache_bytes_per_slot(&m,m.ebits)-row_b) < 1.0);

    /* ---- end to end: the cap cap_for_ram() really sets ---------------------- */
    /* Discriminating, and it duplicates none of cap_for_ram's slack formula: run the
     * SAME budget with and without the MTP row. Adding one int8 row to four int4 rows
     * must cost (4u+2u)/4u = 1.5x of the cap. v1.5.0 charged (4+2)*2u against 4u --
     * 3x -- because the MTP row both widened every other row AND counted twice. */
    g_mem_avail_boot=64.0;
    m.has_mtp=0; m.resident_bytes=0; m.ecap=1<<20; cap_for_ram(&m,8.0,m.ebits,128);
    int cap_routed_only=m.ecap;
    m.has_mtp=1; m.resident_bytes=0; m.ecap=1<<20; cap_for_ram(&m,8.0,m.ebits,128);
    int cap_with_mtp=m.ecap;
    double cost=(double)cap_routed_only/(double)cap_with_mtp;
    printf("  cap %d (routed only) -> %d (+MTP row) = %.2fx\n",
        cap_routed_only,cap_with_mtp,cost);
    CHECK(cap_routed_only>0 && cap_with_mtp>0);
    CHECK(cost > 1.3 && cost < 1.7);      /* ~1.5x, the honest cost of one wide row */
    CHECK(cost < 2.5);                    /* v1.5.0 landed near 3x here */

    /* ---- (2) a reused slot must shrink back down --------------------------- */
    /* This is what makes (1) true rather than optimistic: the LRU promotion in
     * moe() swaps ws[] slots into per-layer caches, so without a shrink the wide
     * slab migrates into a narrow row and stays there. */
    ESlot s; memset(&s,0,sizeof s);
    CHECK(expert_load(&m,N_LAYERS,0,&s,0,0)==0);      /* wide row first */
    int64_t cap_after_wide = s.slab_cap;
    CHECK(cap_after_wide >= wide);
    CHECK(expert_load(&m,FIRST_DENSE,0,&s,0,0)==0);   /* now the same slot, narrow row */
    int64_t cap_after_narrow = s.slab_cap;
    printf("  slab_cap after MTP %lld B -> after routed %lld B\n",
        (long long)cap_after_wide,(long long)cap_after_narrow);
    CHECK(cap_after_narrow < cap_after_wide);         /* it came down */
    CHECK(cap_after_narrow >= narrow);                /* and still fits what it holds */

    /* Negative control: the shrink must never make a slot too small for its own
     * expert. Reload the wide row into the shrunken slot and require it to grow. */
    CHECK(expert_load(&m,N_LAYERS,1,&s,0,0)==0);
    CHECK(s.slab_cap >= wide);

    /* ---- D. the migration itself, as moe() actually performs it ------------ */
    /* colibri.c:4803 is  ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;  -- the cache
     * slot's OLD contents travel back into ws[]. So a wide slab reaches a narrow row
     * on the THIRD step, not the first, which is why a two-load test would miss it. */
    {
        ESlot ws, mtp_cache, main_cache;
        memset(&ws,0,sizeof ws); memset(&mtp_cache,0,sizeof mtp_cache);
        memset(&main_cache,0,sizeof main_cache);
        ESlot tmp;
        /* 1: MTP miss loads wide into ws, promoted into an empty MTP cache slot */
        CHECK(expert_load(&m,N_LAYERS,0,&ws,0,0)==0);
        tmp=mtp_cache; mtp_cache=ws; ws=tmp;
        /* 2: another MTP miss; the promotion now EVICTS, handing the wide slab back */
        CHECK(expert_load(&m,N_LAYERS,1,&ws,0,0)==0);
        tmp=mtp_cache; mtp_cache=ws; ws=tmp;
        CHECK(ws.slab_cap >= wide);            /* ws is carrying a wide slab now */
        /* 3: a MAIN row's miss reuses that slot -- the bleed, if nothing shrinks */
        CHECK(expert_load(&m,FIRST_DENSE,1,&ws,0,0)==0);
        tmp=main_cache; main_cache=ws; ws=tmp;
        printf("D. slab that reached the int4 row: %lld B (a wide one is %lld B)\n",
            (long long)main_cache.slab_cap,(long long)wide);
        CHECK(main_cache.slab_cap < wide);     /* the narrow row is not paying int8 */
        CHECK(main_cache.slab_cap >= narrow);  /* and still holds what it holds */
    }

    /* ---- E. the diagnostic switch actually gates the shrink ---------------- */
    /* g_slab_shrink is set from COLI_SLAB_SHRINK in main(), which this test does
     * not run -- so it is set directly here. Verifying the switch by exporting
     * the variable would have proved nothing, and very nearly did. */
    {
        ESlot s2; memset(&s2,0,sizeof s2);
        g_slab_shrink = 0;
        CHECK(expert_load(&m,N_LAYERS,0,&s2,0,0)==0);          /* wide */
        int64_t wide_cap = s2.slab_cap;
        CHECK(expert_load(&m,FIRST_DENSE,0,&s2,0,0)==0);       /* then narrow */
        printf("E. shrink OFF: %lld B -> %lld B (must not move)\n",
            (long long)wide_cap,(long long)s2.slab_cap);
        CHECK(s2.slab_cap == wide_cap);        /* grow-only: the wide slab stays */
        g_slab_shrink = 1;
        CHECK(expert_load(&m,FIRST_DENSE,1,&s2,0,0)==0);       /* narrow again, shrink on */
        printf("E. shrink ON : %lld B -> %lld B (must come down)\n",
            (long long)wide_cap,(long long)s2.slab_cap);
        CHECK(s2.slab_cap < wide_cap);
    }

    /* ---- C. arena slices must never be shrunk ------------------------------ */
    /* pin_arena_bind hands slots interior pointers into ONE per-layer allocation
     * (#419). Freeing one would corrupt the heap, and they are already per-layer
     * width so they have nothing to shrink. Prove the guard, not the intention. */
    {
        size_t stride=(size_t)wide+8192;
        uint8_t *arena=NULL;
        CHECK(posix_memalign((void**)&arena,4096,stride)==0);
        float *farena=calloc(1<<16,sizeof(float));
        ESlot a; memset(&a,0,sizeof a);
        a.slab=arena; a.aslab=arena; a.slab_cap=(int64_t)stride;
        a.fslab=farena; a.afslab=farena; a.fslab_cap=1<<16;
        CHECK(expert_load(&m,FIRST_DENSE,0,&a,0,0)==0);   /* narrow expert in a wide arena slice */
        printf("C. arena slot: slab %s, cap %lld (was %lld)\n",
            a.slab==arena?"untouched":"MOVED", (long long)a.slab_cap,(long long)stride);
        CHECK(a.slab==arena);                  /* not freed, not reallocated */
        CHECK(a.slab_cap==(int64_t)stride);    /* and its capacity was left alone */
        CHECK(a.fslab==farena);
        free(farena); compat_aligned_free(arena);
    }

    /* 5 MB of zero payload: leave the tree as we found it. */
    { char path[256]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
      remove(path); rmdir(dir); }

    printf("test_cap_mixed_width: %s\n", fails?"FAILED":"ok");
    return fails?1:0;
}
