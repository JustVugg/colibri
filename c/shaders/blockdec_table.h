/* blockdec_table.h: where shaders/blockdec.glsl finds its tables, in 32-bit words of the
 * table buffer the backend builds at device start from ggml_grids.h (the bytes the CPU
 * decoders read, little-endian words; 8-byte grid rows are two words). Included by the
 * shaders and by backend_vulkan.c. In a buffer, not as constants in the shader: a driver
 * that keeps a constant array per invocation (llvmpipe) copied all 35 KB of them at every
 * invocation of every format. */
#ifndef BLOCKDEC_TABLE_H
#define BLOCKDEC_TABLE_H
#define BDT_BLK     0      /* 43: elements per block of ggml type t (0: not a block type here) */
#define BDT_BYTES   43     /* 43: bytes per block */
#define BDT_KSIGNS  86     /* 128: ksigns_iq2xs */
#define BDT_IQ4NL   214    /* 16: kvalues_iq4nl, as int */
#define BDT_MXFP4   230    /* 16: kvalues_mxfp4, as int */
#define BDT_IQ2XXS  246    /* 256 rows of 8 bytes */
#define BDT_IQ2XS   758    /* 512 rows of 8 bytes */
#define BDT_IQ2S    1782   /* 1024 rows of 8 bytes */
#define BDT_IQ3XXS  3830   /* 256 rows of 4 bytes */
#define BDT_IQ3S    4086   /* 512 rows of 4 bytes */
#define BDT_IQ1S    4598   /* 2048 rows of 8 signed bytes */
#define BDT_VQ2     8694   /* 1024: vq2's codebooks, A then B, 256 rows of 8 signed bytes each (the model's,
                             set by coli_vk_set_vq2_books; zeros until then) */
#define BDT_WORDS   9718
#endif
