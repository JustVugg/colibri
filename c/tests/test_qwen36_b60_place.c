/* The B60 chain placement must preserve every expert's existing home and
 * fall back to the ordinary placer when a precondition is not met. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int failures;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); failures++; } } while (0)

int main(void) {
    const size_t mib = (size_t)1024 * 1024;
    size_t capacity[2] = { 6 * 1024 * mib, 6 * 1024 * mib };
    memset(&G, 0, sizeof G);
    G.ndev=2; G.dev[0]=3; G.dev[1]=7;
    G.D=2048; G.Ih=512; G.wfmt=4; G.egs=64; G.exp_bytes=mib;
    G_stream=G_trunk_only=G_home_layer=0; G_offer_n=0; G_auto_on=0;
    unsetenv("COLI_CUDA_CHAIN");
    qt_trunk_offer("lmhead", 0, mib);
    for(int l=0;l<40;l++){
        if(l%4==3) qt_trunk_offer("attnproj", l, mib);
        else { qt_trunk_offer("dnproj", l, mib); qt_trunk_offer("dnout", l, mib); }
        qt_trunk_offer("shexp", l, mib);
    }

    CHECK(!auto_place_b60_qwen(40,256,8,capacity,0));
    CHECK(G_offer[0].dev==QT_PLACE_CPU && !G_auto_on);
    size_t saved=capacity[0]; capacity[0]=5120*mib+110*mib;
    CHECK(!auto_place_b60_qwen(40,256,8,capacity,1)); /* one MiB short for the trunk */
    CHECK(G_offer[0].dev==QT_PLACE_CPU && !G_auto_on);
    capacity[0]=saved; capacity[1]=5119*mib;
    CHECK(!auto_place_b60_qwen(40,256,8,capacity,1)); /* second card cannot hold its experts */
    capacity[1]=saved;
    G_stream=1; CHECK(!auto_place_b60_qwen(40,256,8,capacity,1)); G_stream=0;
    G_home_layer=1; CHECK(!auto_place_b60_qwen(40,256,8,capacity,1)); G_home_layer=0;
    setenv("COLI_CUDA_CHAIN","0",1);
    CHECK(!auto_place_b60_qwen(40,256,8,capacity,1));
    unsetenv("COLI_CUDA_CHAIN");
    G_offer_n--; CHECK(!auto_place_b60_qwen(40,256,8,capacity,1)); G_offer_n++;
    CHECK(auto_place_b60_qwen(40,256,8,capacity,1));
    CHECK(G_auto_on && G_auto_lmh==3 && G_trunk_bytes[0]==111*mib && G_trunk_bytes[1]==0);
    for(int o=0;o<G_offer_n;o++) CHECK(G_offer[o].dev==3);
    for(int l=0;l<40;l++) if(l%4!=3) CHECK(G_auto_dnp[l]==3);
    printf("B60 Qwen placement: %s\n", failures ? "FAIL" : "PASS");
    return failures != 0;
}
