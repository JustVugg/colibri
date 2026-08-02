#include <math.h>
#include <stdio.h>
#include "../dsv4_quant.h"

static int near(float a,float b){ return fabsf(a-b)<1e-6f; }
int main(void){
    if(!near(dsv4_e4m3(0x00),0)||!near(dsv4_e4m3(0x01),0.001953125f)||
       !near(dsv4_e4m3(0x38),1)||!near(dsv4_e4m3(0x7e),448)||
       !near(dsv4_e4m3(0xfe),-448)||!isnan(dsv4_e4m3(0x7f))) return 1;
    if(!near(dsv4_e8m0(126),.5f)||!near(dsv4_e8m0(127),1)||
       !near(dsv4_e8m0(128),2)||!isnan(dsv4_e8m0(255))) return 2;
    { float x[3]={1,2,-1},y[2]; uint8_t w[6]={0x38,0x40,0x38,0x40,0x38,0xb8};
      uint8_t s[1]={127}; dsv4_fp8_matvec(y,x,w,s,2,3);
      if(!near(y[0],4)||!near(y[1],5)) return 3; }
    puts("dsv4-quant: OK"); return 0;
}
