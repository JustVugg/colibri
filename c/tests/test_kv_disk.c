/* .coli_kv round-trip: v1 (f32) and v2 (KV8 fp8+scale) records, the v1->v2
 * upgrade path (quantize-on-load; the v1 file survives until the first save rewrites it
 * as v2), the v2-under-f32 reject, and the format-mismatch self-heal in
 * kv_disk_append (magic differs -> file rewritten from record 0). No model
 * file needed: the disk paths only read c->n_layers/kv_lora/qk_rope/vocab. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#define PATH "tests/.coli_kv_test.tmp"
static int fails=0;
#define CHECK(cond, ...) do{ if(!(cond)){ fails++; \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fputc('\n',stderr);} }while(0)

static float fill(int i,int p,int j){ return sinf(i*7.f+p*1.3f+j*0.37f)*(1.f+p); }

int main(void){
    static Model m;
    m.c.n_layers=2; m.c.kv_lora=8; m.c.qk_rope=4; m.c.vocab=256;
    m.kv=calloc(1,sizeof(KVState));
    snprintf(m.kv->disk_path,sizeof(m.kv->disk_path),"%s",PATH);
    remove(PATH); g_kvsave=1; g_draft=0;
    int NP=5, hist[5]={11,22,33,44,55}, hist2[16]={0};

    /* ---- v1: f32 append + load must round-trip exactly ---- */
    g_kv8=0; kv_alloc(&m,16);
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){
        for(int j=0;j<m.c.kv_lora;j++) m.Lc[i][(int64_t)p*m.c.kv_lora+j]=fill(i,p,j);
        for(int j=0;j<m.c.qk_rope;j++) m.Rc[i][(int64_t)p*m.c.qk_rope+j]=fill(i+9,p,j);
    }
    kv_disk_append(&m,hist,NP);
    CHECK(m.kv->disk_nrec==NP, "v1 append nrec=%d", m.kv->disk_nrec);
    kv_alloc(&m,16); m.kv->disk_nrec=0;             /* cache azzerata, si ricarica da disco */
    CHECK(kv_disk_load(&m,hist2,16)==NP, "v1 load");
    for(int p=0;p<NP;p++) CHECK(hist2[p]==hist[p], "v1 hist[%d]", p);
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++)
        for(int j=0;j<m.c.kv_lora;j++)
            CHECK(m.Lc[i][(int64_t)p*m.c.kv_lora+j]==fill(i,p,j), "v1 Lc[%d] p%d j%d", i,p,j);

    /* ---- v1 file + KV8: quantize-on-load; il file v1 resta INTATTO (un crash
     * prima del primo save non deve perdere la conversazione) ---- */
    if(m.kv->disk_fp){ fclose(m.kv->disk_fp); m.kv->disk_fp=NULL; }  /* "riavvio" del processo */
    g_kv8=1; kv_alloc(&m,16); m.kv->disk_nrec=0;
    CHECK(kv_disk_load(&m,hist2,16)==NP, "v1->kv8 load");
    CHECK(m.kv->disk_nrec==0, "v1->kv8 leaves disk_nrec=0 for the full rewrite");
    { FILE *f=fopen(PATH,"rb"); char mg[8]={0};
      CHECK(f && fread(mg,1,8,f)==8 && !memcmp(mg,KV_MAGIC,8),
            "v1 file must survive the upgrade load untouched");
      if(f) fclose(f); }
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){
        float sc=m.Lsc[i][p];
        for(int j=0;j<m.c.kv_lora;j++){
            float want=fill(i,p,j), got=coli_fp8_lut[m.Lc8[i][(int64_t)p*m.c.kv_lora+j]]*sc;
            CHECK(fabsf(got-want)<=fabsf(want)/16.f+sc*0x1p-10f+1e-7f,
                  "v1->kv8 Lc[%d] p%d j%d: %g vs %g", i,p,j,got,want);
        }
    }

    /* ---- v2: fp8 append + load must round-trip BYTE-identical ---- */
    kv_disk_append(&m,hist,NP);
    CHECK(m.kv->disk_nrec==NP, "v2 append nrec=%d", m.kv->disk_nrec);
    { FILE *f=fopen(PATH,"rb"); char mg[8]={0};
      CHECK(f && fread(mg,1,8,f)==8 && !memcmp(mg,KV_MAGIC2,8), "v2 magic on disk");
      if(f) fclose(f); }
    uint8_t keepL[2][5*8]; float keepS[2][5];
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){
        memcpy(keepL[i]+(size_t)p*m.c.kv_lora, m.Lc8[i]+(int64_t)p*m.c.kv_lora, m.c.kv_lora);
        keepS[i][p]=m.Lsc[i][p];
    }
    kv_alloc(&m,16); m.kv->disk_nrec=0;
    CHECK(kv_disk_load(&m,hist2,16)==NP, "v2 load");
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){
        CHECK(!memcmp(m.Lc8[i]+(int64_t)p*m.c.kv_lora, keepL[i]+(size_t)p*m.c.kv_lora, m.c.kv_lora),
              "v2 Lc8 bytes layer %d pos %d", i,p);
        CHECK(m.Lsc[i][p]==keepS[i][p], "v2 Lsc layer %d pos %d", i,p);
    }

    /* ---- v2 sotto f32: si riparte da zero, il file NON va creduto ---- */
    if(m.kv->disk_fp){ fclose(m.kv->disk_fp); m.kv->disk_fp=NULL; }  /* "riavvio" del processo */
    g_kv8=0; kv_alloc(&m,16); m.kv->disk_nrec=0;
    CHECK(kv_disk_load(&m,hist2,16)==0, "v2 under f32 must be rejected");

    /* ---- append con formato diverso dal file: riscrittura dal record 0 ---- */
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){
        for(int j=0;j<m.c.kv_lora;j++) m.Lc[i][(int64_t)p*m.c.kv_lora+j]=fill(i,p,j);
        for(int j=0;j<m.c.qk_rope;j++) m.Rc[i][(int64_t)p*m.c.qk_rope+j]=fill(i+9,p,j);
    }
    if(m.kv->disk_fp){ fclose(m.kv->disk_fp); m.kv->disk_fp=NULL; }  /* "riavvio" del processo */
    m.kv->disk_nrec=3;                              /* stantio: la magic v2 lo invalida */
    kv_disk_append(&m,hist,NP);
    { FILE *f=fopen(PATH,"rb"); char mg[8]={0};
      CHECK(f && fread(mg,1,8,f)==8 && !memcmp(mg,KV_MAGIC,8), "self-heal rewrote v1 magic");
      if(f) fclose(f); }
    kv_alloc(&m,16); m.kv->disk_nrec=0;
    CHECK(kv_disk_load(&m,hist2,16)==NP, "self-healed v1 load");
    for(int p=0;p<NP;p++) CHECK(hist2[p]==hist[p], "self-heal hist[%d]", p);

    /* ---- v3: KV_TQ PolarQuant append + load must round-trip BYTE-identical, plus
     * the reject paths (v3 under f32; v3 under a different bit width). Same tiny
     * synthetic config: kv_lora=8, qk_rope=4 (both powers of two, as TQ requires). */
    if(m.kv->disk_fp){ fclose(m.kv->disk_fp); m.kv->disk_fp=NULL; }
    remove(PATH);
    g_kv8=0; g_tq=1; g_tq_bits=4;
    kv_alloc(&m,16); m.kv->disk_nrec=0;
    int lbb=coli_tq_row_bytes(m.c.kv_lora,g_tq_bits), rbb=coli_tq_row_bytes(m.c.qk_rope,g_tq_bits);
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){       /* populate via the producer's PolarQuant */
        float lr[8], rr[4];
        for(int j=0;j<m.c.kv_lora;j++) lr[j]=fill(i,p,j);
        for(int j=0;j<m.c.qk_rope;j++) rr[j]=fill(i+9,p,j);
        m.Lsc[i][p]=coli_tq_quant_row(lr, coli_kv_row8(m.Lc8[i],p,lbb), m.c.kv_lora, g_tq_bits);
        m.Rsc[i][p]=coli_tq_quant_row(rr, coli_kv_row8(m.Rc8[i],p,rbb), m.c.qk_rope, g_tq_bits);
    }
    kv_disk_append(&m,hist,NP);
    CHECK(m.kv->disk_nrec==NP, "v3 append nrec=%d", m.kv->disk_nrec);
    { FILE *f=fopen(PATH,"rb"); char mg[8]={0};
      CHECK(f && fread(mg,1,8,f)==8 && !memcmp(mg,KV_MAGIC3,8), "v3 magic on disk");
      if(f) fclose(f); }
    uint8_t keepL3[2][5*8], keepR3[2][5*4]; float keepLs3[2][5], keepRs3[2][5];
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){
        memcpy(keepL3[i]+(size_t)p*lbb, coli_kv_row8(m.Lc8[i],p,lbb), lbb);
        memcpy(keepR3[i]+(size_t)p*rbb, coli_kv_row8(m.Rc8[i],p,rbb), rbb);
        keepLs3[i][p]=m.Lsc[i][p]; keepRs3[i][p]=m.Rsc[i][p];
    }
    kv_alloc(&m,16); m.kv->disk_nrec=0;
    CHECK(kv_disk_load(&m,hist2,16)==NP, "v3 load");
    for(int i=0;i<m.c.n_layers;i++) for(int p=0;p<NP;p++){
        CHECK(!memcmp(coli_kv_row8(m.Lc8[i],p,lbb), keepL3[i]+(size_t)p*lbb, lbb), "v3 Lc8 bytes layer %d pos %d", i,p);
        CHECK(!memcmp(coli_kv_row8(m.Rc8[i],p,rbb), keepR3[i]+(size_t)p*rbb, rbb), "v3 Rc8 bytes layer %d pos %d", i,p);
        CHECK(m.Lsc[i][p]==keepLs3[i][p], "v3 Lsc layer %d pos %d", i,p);
        CHECK(m.Rsc[i][p]==keepRs3[i][p], "v3 Rsc layer %d pos %d", i,p);
    }
    /* v3 under f32: reject (the file must not be believed) */
    if(m.kv->disk_fp){ fclose(m.kv->disk_fp); m.kv->disk_fp=NULL; }
    g_tq=0; g_kv8=0; kv_alloc(&m,16); m.kv->disk_nrec=0;
    CHECK(kv_disk_load(&m,hist2,16)==0, "v3 under f32 must be rejected");
    /* v3 under a DIFFERENT bit width: reject (can't decode 4-bit angles as 3-bit) */
    if(m.kv->disk_fp){ fclose(m.kv->disk_fp); m.kv->disk_fp=NULL; }
    g_tq=1; g_tq_bits=3; kv_alloc(&m,16); m.kv->disk_nrec=0;
    CHECK(kv_disk_load(&m,hist2,16)==0, "v3 under different bit width must be rejected");
    g_tq=0;

    remove(PATH);
    if(fails){ fprintf(stderr,"%d failure(s)\n",fails); return 1; }
    printf("OK kv_disk v1/v2 round-trip + upgrade + reject + self-heal\n");
    return 0;
}
