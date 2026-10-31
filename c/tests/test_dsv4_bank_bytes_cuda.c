/* The sizes the engine's default DSV4_CUDA_VRAM_RESERVE_MB is built from must
 * be what the backend really allocates: dsv4_cuda_expert_bank_slot_bytes
 * against the free-VRAM drop of a real (small, 16-slot) bank, and the decode
 * mirror size (gate + up + down fp4 tensors, 12.75 MiB at the V4 geometry)
 * against dsv4_cuda_tensor_bytes. ~0.2 GiB of VRAM. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

int main(void){
    enum { H=4096, I=2048, N=16 };
    int dev=0;if(!dsv4_cuda_init(&dev,1))return 1;
    int dg=strcmp(dsv4_cuda_backend_name(),"generic")!=0;
    long long fc1_s=2LL*I*(H/32),fc2_s=(long long)H*(I/32);
    long long core=(long long)I*H+fc1_s+(long long)H*(I/2)+fc2_s+(dg?fc1_s+fc2_s:0);
    long long slot=dsv4_cuda_expert_bank_slot_bytes(H,I);
    /* the weights and scales exactly, plus the 3 pointer-table entries */
    if(slot<core||slot>core+64){fprintf(stderr,"slot bytes %lld, want %lld (+table)\n",slot,core);return 2;}
    if(core!=(dg?14155776LL:13369344LL)){fprintf(stderr,"slot core %lld is not %s MiB\n",core,dg?"13.5":"12.75");return 3;}
    if(dsv4_cuda_expert_bank_slots(132)!=(dg?256:132)||dsv4_cuda_expert_bank_slots(0)!=0||dsv4_cuda_expert_bank_slot_bytes(0,I)!=0)return 4;
    long long free0=dsv4_cuda_mem_free_mb(dev);
    Dsv4CudaExpertSet*bank=dsv4_cuda_expert_bank_create(N,H,I,dev,NULL,NULL,NULL);if(!bank)return 5;
    long long free1=dsv4_cuda_mem_free_mb(dev),want_mb=(N*slot)>>20,took=free0-free1;
    /* five cudaMallocs rounded up to the allocator's 2 MiB pages */
    if(free0<0||free1<0||took<want_mb||took>want_mb+12){fprintf(stderr,"bank of %d took %lld MiB, slot bytes say %lld\n",N,took,want_mb);return 6;}
    dsv4_cuda_expert_set_free(bank);
    size_t wn=(size_t)I*H/2,sn=(size_t)I*H/32;uint8_t*w=calloc(wn,1),*s=calloc(sn,1);if(!w||!s)return 7;
    Dsv4CudaTensor*g=NULL,*u=NULL,*d=NULL;
    if(!dsv4_cuda_upload_fp4(&g,w,s,I,H,dev)||!dsv4_cuda_upload_fp4(&u,w,s,I,H,dev)||!dsv4_cuda_upload_fp4(&d,w,s,H,I,dev))return 8;
    long long mirror=dsv4_cuda_tensor_bytes(g)+dsv4_cuda_tensor_bytes(u)+dsv4_cuda_tensor_bytes(d);
    if(mirror!=13369344LL){fprintf(stderr,"decode mirror %lld bytes, want 12.75 MiB\n",mirror);return 9;}
    dsv4_cuda_tensor_free(d);dsv4_cuda_tensor_free(u);dsv4_cuda_tensor_free(g);free(s);free(w);dsv4_cuda_shutdown();
    printf("dsv4 bank bytes: slot %lld, %d-slot bank took %lld MiB, mirror %lld: OK\n",slot,N,took,mirror);return 0;
}
