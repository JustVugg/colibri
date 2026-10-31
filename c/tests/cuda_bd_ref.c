/* The CPU's GGUF decoders (ggml_quants.h) for tests/test_cuda_bd.cu, which nvcc compiles as
 * C++ (the header's designated initializers are C's): the integer form's unpack and the
 * activation's quantization, the float dequantization. */
#include "../ggml_quants.h"
int ref_unpack(int t, const uint8_t *row, int n, int8_t *w8, float *sw, float *mw) { return gq_unpack_row(t, row, n, w8, sw, mw); }
void ref_quant(const float *x, int n, int8_t *xq, float *dx, int32_t *xs) { gq_quant_act_ref(x, n, xq, dx, xs); }
int ref_dequant(int t, const uint8_t *row, float *out, int n) { return gq_dequant_row(t, row, out, n); }
long ref_row_bytes(int t, int n) { return (long)gq_row_bytes(t, n); }
int ref_block_bytes(int t) { return GQ_TYPES[t].bytes; }
int ref_block_elems(int t) { return GQ_TYPES[t].blck; }
const char *ref_name(int t) { return gq_type_name(t); }
