/* Link the engine's COLI_V4_UNIT_ROUTE_BF16 object. Writes only argv[1] (or a temp file). */
#include "../v4_np_route.h"
#include "../v4_np_gates.h"
#include <assert.h>
static void u32(FILE *f,uint32_t v) {unsigned char b[4]={(unsigned char)v,(unsigned char)(v>>8),(unsigned char)(v>>16),(unsigned char)(v>>24)};assert(fwrite(b,1,4,f)==4);}
static void fixture(const char *path,int fault) {
    FILE *f=fopen(path,"wb");assert(f);fwrite("V4PG",1,4,f);
    u32(f,fault==1?2:1);u32(f,3);u32(f,2);u32(f,2);u32(f,2);
    for(int depth=1;depth<=2;depth++) for(int layer=0;layer<3;layer++) {
        int present=layer>=depth;
        u32(f,fault==2&&layer==0?2:(uint32_t)present);
        if(!present) continue;
        uint16_t gate[4]={0x3f80,0,0,0x3f80}; /* identity */
        if(fault==3&&depth==2) gate[0]=0x7f80;
        for(int j=0;j<4;j++){fputc(gate[j]&255,f);fputc(gate[j]>>8,f);}
        if(fault==6&&depth==2){fclose(f);return;} /* truncated intercept */
        u32(f,0);u32(f,fault==4&&depth==2?0x7fc00000:0x40000000); /* c=[0,2] */
    }
    if(fault==5) fputc(0,f);
    fclose(f);
}
int main(int argc,char **argv) {
    const char *path = argc >= 2 ? argv[1] : "v4_np_gates_fixture.tmp";
    V4NpGates g={0};fixture(path,0);
    assert(!v4_np_gates_load(&g,path,3,2,2));
    assert(!v4_np_gates_get(&g,0,1));assert(!v4_np_gates_get(&g,1,2));
    const V4NpGate *e=v4_np_gates_get(&g,2,2);assert(e);
    float x[2]={1,0},w[2],expected_w[2],bias[2]={0,0},scratch[2];int ids[2],expected[2];
    float logits[2]={1,2};
    assert(!coli_v4_route_bf16_logits(expected_w,expected,x,e->gate,bias,NULL,2,2,2,1,logits));
    assert(expected[0]==1);
    assert(!v4_np_gates_route(e,w,ids,x,scratch,bias,2,2,2,1,0));
    assert(!memcmp(w,expected_w,sizeof(w))&&!memcmp(ids,expected,sizeof(ids)));
    /* Bias affects selection AFTER activation, not the learned intercept. */
    bias[0]=10;
    assert(!v4_np_gates_route(e,w,ids,x,scratch,bias,2,2,2,1,1));assert(ids[0]==0);
    /* Exercise AVX2 body and odd tail, using exact products plus intercept. */
    uint16_t wide[66];float wx[33],wi[2]={0,2};
    for(int c=0;c<33;c++){wide[c]=0x3f80;wide[33+c]=0x3f00;wx[c]=0.125f;}
    V4NpGate wide_gate={wide,wi};bias[0]=0;
    assert(!v4_np_gates_route(&wide_gate,w,ids,wx,NULL,bias,2,33,2,1,0));
    assert(!v4_np_gates_route(&wide_gate,expected_w,expected,wx,NULL,bias,2,33,2,1,1));
    assert(!memcmp(w,expected_w,sizeof(w))&&!memcmp(ids,expected,sizeof(ids)));
    /* Exercise _ex with additive online offset and z_out capture. */
    float add[2]={5.0f, -5.0f}, z_out[2]={0,0};
    assert(!v4_np_gates_route_ex(e,w,ids,x,scratch,bias,2,2,2,1,0,add,z_out));
    assert(ids[0]==0 && fabsf(z_out[0]-6.0f)<1e-6f && fabsf(z_out[1]+3.0f)<1e-6f);
    for(int fault=1;fault<=6;fault++) {fixture(path,fault);assert(v4_np_gates_load(&g,path,3,2,2));assert(v4_np_gates_get(&g,2,2)==e);}
    fixture(path,0);assert(v4_np_gates_load(&g,path,4,2,2));
    FILE *f=fopen(path,"wb");assert(f);fwrite("V4PG",1,4,f);fclose(f);
    assert(v4_np_gates_load(&g,path,3,2,2));
    remove(path);v4_np_gates_free(&g);puts("v4 np gates: ok");return 0;
}
