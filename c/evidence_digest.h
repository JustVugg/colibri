/* evidence_digest.h — SHA-256 over the exact bytes an evidence-producing mode
 * consumed.  Header-only: all functions are static — include from the engine.
 *
 * Diagnostic modes of the engine write artifacts that a separate offline tool
 * checks.  For that check to mean anything, the artifact has to name the inputs
 * it was produced from in a way the tool can recompute independently: the exact
 * config.json bytes that were loaded, and the exact manifest bytes that were
 * run.  A digest over those byte ranges does that, and a short self-contained
 * implementation keeps it from becoming a build dependency of the whole engine
 * on a crypto library it otherwise never needs.  This is an integrity aid for
 * reproducibility, not a security boundary. */
#ifndef EVIDENCE_DIGEST_H
#define EVIDENCE_DIGEST_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} EvidenceSha256;

static uint32_t evidence_rotr32(uint32_t x, unsigned n){
    return (x>>n)|(x<<(32-n));
}

static void evidence_sha256_block(EvidenceSha256 *s, const unsigned char *p){
    static const uint32_t k[64]={
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
    };
    uint32_t w[64];
    for(int i=0;i<16;i++)
        w[i]=((uint32_t)p[4*i]<<24)|((uint32_t)p[4*i+1]<<16)|
             ((uint32_t)p[4*i+2]<<8)|(uint32_t)p[4*i+3];
    for(int i=16;i<64;i++){
        uint32_t s0=evidence_rotr32(w[i-15],7)^evidence_rotr32(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=evidence_rotr32(w[i-2],17)^evidence_rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3];
    uint32_t e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for(int i=0;i<64;i++){
        uint32_t S1=evidence_rotr32(e,6)^evidence_rotr32(e,11)^evidence_rotr32(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+k[i]+w[i];
        uint32_t S0=evidence_rotr32(a,2)^evidence_rotr32(a,13)^evidence_rotr32(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c), t2=S0+maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

static void evidence_sha256_init(EvidenceSha256 *s){
    *s=(EvidenceSha256){{
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u,
    },0,{0},0};
}

/* Streaming update: a caller can hash a file line by line as it validates it,
 * so the digest covers exactly the bytes that were accepted. */
static void evidence_sha256_update(EvidenceSha256 *s, const void *data, size_t len){
    const unsigned char *p=(const unsigned char *)data;
    s->bits+=(uint64_t)len*8u;
    while(len){
        size_t take=64-s->used; if(take>len) take=len;
        memcpy(s->block+s->used,p,take); s->used+=take; p+=take; len-=take;
        if(s->used==64){ evidence_sha256_block(s,s->block); s->used=0; }
    }
}

static void evidence_sha256_final(EvidenceSha256 *s, unsigned char out[32]){
    uint64_t bits=s->bits;
    s->block[s->used++]=0x80;
    if(s->used>56){
        memset(s->block+s->used,0,64-s->used);
        evidence_sha256_block(s,s->block); s->used=0;
    }
    memset(s->block+s->used,0,56-s->used);
    for(int i=0;i<8;i++) s->block[63-i]=(unsigned char)(bits>>(8*i));
    evidence_sha256_block(s,s->block);
    for(int i=0;i<8;i++){
        out[4*i]=(unsigned char)(s->h[i]>>24);
        out[4*i+1]=(unsigned char)(s->h[i]>>16);
        out[4*i+2]=(unsigned char)(s->h[i]>>8);
        out[4*i+3]=(unsigned char)s->h[i];
    }
}

/* One-shot digest as the 64 lowercase hex characters an artifact carries,
 * NUL-terminated so it can be written straight into a text record. */
static void evidence_sha256_hex(const void *data, size_t len, char out[65]){
    static const char hex[]="0123456789abcdef";
    EvidenceSha256 s; unsigned char raw[32];
    evidence_sha256_init(&s); evidence_sha256_update(&s,data,len);
    evidence_sha256_final(&s,raw);
    for(int i=0;i<32;i++){ out[2*i]=hex[raw[i]>>4]; out[2*i+1]=hex[raw[i]&15]; }
    out[64]=0;
}

#endif /* EVIDENCE_DIGEST_H */
