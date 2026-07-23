/* e4m3 KV8: the encoder and the decode LUT must agree exactly. The whole
 * fp8 code space is tiny (256 codes), so the round-trip is tested EXHAUSTIVELY:
 * encode(decode(b)) == b for every representable byte. On top of that:
 * round-to-nearest-even at mantissa midpoints, ±448 saturation, the per-row
 * amax quantizer's error bound, and the coli_kv_row8 arithmetic twin. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../kv_fp8.h"
#include "../decode_batch.h"

static int fails=0;
#define CHECK(cond, ...) do{ if(!(cond)){ fails++; \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fputc('\n',stderr);} }while(0)

int main(void){
    coli_fp8_lut_init();

    /* exhaustive round-trip on every code except the NaN patterns (0x7F/0xFF,
     * decoded as 0 by design, never produced by the encoder) */
    for(int b=0;b<256;b++){
        if((b&0x7F)==0x7F) continue;             /* NaN codes: decoded as 0, never produced */
        uint8_t e=coli_fp8_enc(coli_fp8_lut[b]);
        CHECK(e==(uint8_t)b, "roundtrip byte 0x%02x -> %g -> 0x%02x", b, coli_fp8_lut[b], e);
    }

    /* saturation and specials */
    CHECK(coli_fp8_enc(448.f)==0x7E, "448 is the max normal");
    CHECK(coli_fp8_enc(-448.f)==0xFE, "-448");
    CHECK(coli_fp8_enc(1000.f)==0x7E, "overflow saturates");
    CHECK(coli_fp8_enc(-1e30f)==0xFE, "negative overflow saturates");
    CHECK(coli_fp8_enc(INFINITY)==0x7E, "+inf saturates");
    CHECK(coli_fp8_enc(NAN)==0x00, "NaN stored as inert 0");
    CHECK(coli_fp8_enc(0.f)==0x00 && coli_fp8_enc(-0.f)==0x80, "signed zeros preserved");

    /* round-to-nearest-even at midpoints: steps of 2 in [16,32) (E=11).
     * 17 ties between 16 (mant 0, even) and 18 (mant 1, odd) -> 16;
     * 19 ties between 18 (mant 1) and 20 (mant 2, even) -> 20. */
    CHECK(coli_fp8_lut[coli_fp8_enc(17.f)]==16.f, "RNE tie 17 -> 16, got %g", coli_fp8_lut[coli_fp8_enc(17.f)]);
    CHECK(coli_fp8_lut[coli_fp8_enc(19.f)]==20.f, "RNE tie 19 -> 20, got %g", coli_fp8_lut[coli_fp8_enc(19.f)]);
    CHECK(coli_fp8_lut[coli_fp8_enc(17.1f)]==18.f, "17.1 -> 18");
    /* denormal boundary: half of the smallest denormal (2^-10) ties to 0 (even) */
    CHECK(coli_fp8_enc(0x1p-10f)==0x00, "2^-10 ties to even 0");
    CHECK(coli_fp8_enc(0x1.8p-10f)==0x01, "0.75*2^-9 rounds up to the first denormal");
    /* mantissa overflow across the exponent boundary: just below 2^-6 */
    CHECK(coli_fp8_lut[coli_fp8_enc(0x1.fcp-7f)]==0x1p-6f, "denormal grid rounds up into the first normal");
    /* binade-boundary tie WITH mantissa carry (k==16 -> k=8,E++): 15.5 lies exactly
     * between 15 (M=7, odd) and 16 (next binade, M=0, even) -> RNE must carry to 16.
     * The exhaustive roundtrip never triggers this (grid points encode carry-free). */
    CHECK(coli_fp8_lut[coli_fp8_enc(15.5f)]==16.f, "RNE tie 15.5 -> 16 (mantissa carry), got %g",
          coli_fp8_lut[coli_fp8_enc(15.5f)]);
    CHECK(coli_fp8_lut[coli_fp8_enc(15.9f)]==16.f, "15.9 rounds up across the binade");
    CHECK(coli_fp8_lut[coli_fp8_enc(0x1.dp3f)]==14.f, "14.5 ties down to even 14");

    /* per-row quantizer: amax maps exactly to ±448*scale, error bounded by
     * |x|/16 (3 mantissa bits) + half a denormal step (scale*2^-10) */
    enum { N=576 };
    float row[N]; uint8_t q[N]; float deq[N];
    for(int i=0;i<N;i++) row[i]=sinf(0.7f*i)*expf(-((i%37)*0.21f))*(i%2?1.f:-1.f)*3.7f;
    row[123]=-9.25f;                              /* the amax, negative on purpose */
    float scale=coli_kv8_quant_row(row,q,N);
    CHECK(fabsf(scale-9.25f/448.f)<1e-9f, "scale = amax/448 (got %g)", scale);
    coli_kv8_dequant_row(q,scale,deq,N);
    CHECK(deq[123]==-9.25f || fabsf(deq[123]+9.25f)<=9.25f*1e-6f, "amax survives the trip (got %g)", deq[123]);
    for(int i=0;i<N;i++){
        float err=fabsf(deq[i]-row[i]), bound=fabsf(row[i])/16.f+scale*0x1p-10f+1e-7f;
        CHECK(err<=bound, "elem %d: |%g - %g| = %g > bound %g", i, deq[i], row[i], err, bound);
    }

    /* all-zero row: bytes 0, scale 1 (no fabricated 1/0) */
    float zrow[8]={0}; uint8_t zq[8];
    CHECK(coli_kv8_quant_row(zrow,zq,8)==1.f, "zero row keeps scale 1");
    for(int i=0;i<8;i++) CHECK(zq[i]==0, "zero row byte %d", i);
    /* subnormal amax: 448/amax would overflow to +inf and saturate every nonzero
     * element to ±448 — the guard must treat the row as zero instead */
    float tiny[4]={1e-38f,-1e-38f,0,1e-40f}; uint8_t tq[4];
    CHECK(coli_kv8_quant_row(tiny,tq,4)==1.f, "subnormal row keeps scale 1");
    for(int i=0;i<4;i++) CHECK(tq[i]==0, "subnormal row byte %d", i);

    /* row accessor twin: same arithmetic as coli_kv_row, byte-typed */
    uint8_t buf[7*5];
    CHECK(coli_kv_row8(buf,4,7)==buf+28, "coli_kv_row8 arithmetic");
    CHECK(coli_kv_row8(buf,0,7)==buf, "coli_kv_row8 row 0");

    if(fails){ fprintf(stderr,"%d failure(s)\n",fails); return 1; }
    printf("OK kv_fp8 e4m3 (exhaustive roundtrip + RNE + row quant)\n");
    return 0;
}
