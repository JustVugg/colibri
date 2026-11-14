/* Host pinning under a record that moves inside its slab (V4_SLAB_SHIFT): a
 * refill page-locks the range it copies from, and the slab's next record
 * starts at another byte, so its range runs across the end of the one locked
 * before. Both the refill and a first upload from such a range must go
 * through and land the bytes they were given (HIP refused the copy: invalid
 * argument). ~10 MiB of VRAM. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

enum { O=2048, I=2048, WN=O*I/2, SN=O*I/32, REC=WN+SN, SLAB=4*REC };

/* a tensor filled from pageable memory holding the same record: the reference */
static int same(Dsv4CudaTensor*t,const uint8_t*rec,const float*x,int dev){
    static float y[O],ref[O];uint8_t*copy=malloc(REC);Dsv4CudaTensor*r=NULL;if(!copy)return 0;
    memcpy(copy,rec,REC);
    int ok=dsv4_cuda_upload_fp4(&r,copy+SN,copy,O,I,dev)&&dsv4_cuda_matvec(r,ref,x)&&dsv4_cuda_matvec(t,y,x)&&!memcmp(y,ref,sizeof y);
    dsv4_cuda_tensor_free(r);free(copy);return ok;
}

int main(void){
    int dev=0;if(!dsv4_cuda_init(&dev,1))return 1;
    uint8_t*slab=aligned_alloc(4096,SLAB);if(!slab)return 2;
    uint32_t z=0x9e3779b9u;for(size_t i=0;i<SLAB;i++){z=z*1664525u+1013904223u;slab[i]=(uint8_t)(z>>24);}
    static float x[I];for(int i=0;i<I;i++)x[i]=(float)(i%13-6)*.03125f;
    /* a record: its scales, then its weights; three starts in one slab */
    const uint8_t*a=slab,*b=slab+5*4096+64,*c=slab+11*4096+192;
    Dsv4CudaTensor*t=NULL,*u=NULL;
    if(!dsv4_cuda_upload_fp4(&t,a+SN,a,O,I,dev))return 3;
    if(!dsv4_cuda_tensor_refill_fp4(t,a+SN,a,O,I,1)||!same(t,a,x,dev)){fprintf(stderr,"refill from the slab's first record failed\n");return 4;}
    if(!dsv4_cuda_tensor_refill_fp4(t,b+SN,b,O,I,1)||!same(t,b,x,dev)){fprintf(stderr,"refill from a record across the end of a pinned range failed\n");return 5;}
    if(!dsv4_cuda_upload_fp4(&u,c+SN,c,O,I,dev)||!same(u,c,x,dev)){fprintf(stderr,"first upload from a record across the end of a pinned range failed\n");return 6;}
    if(!dsv4_cuda_tensor_refill_fp4(t,a+SN,a,O,I,1)||!same(t,a,x,dev)){fprintf(stderr,"refill from the first record again failed\n");return 7;}
    dsv4_cuda_tensor_free(u);dsv4_cuda_tensor_free(t);dsv4_cuda_shutdown();free(slab);
    printf("dsv4 pin overlap: refills and a first upload across a pinned range: OK\n");return 0;
}
