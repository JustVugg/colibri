#pragma once
#include <cstddef>
#include <hip/hip_runtime.h>

typedef int   cublasStatus_t;
typedef int   cublasOperation_t;
typedef int   cublasComputeType_t;
typedef int   cublasLtMatmulMatrixScale_t;
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;
typedef struct { char _opaque[256]; } cublasLtMatmulHeuristicResult_t;

#define CUBLAS_STATUS_SUCCESS                    0
#define CUBLAS_STATUS_NOT_SUPPORTED              15
#define CUBLAS_OP_N                              0
#define CUBLAS_OP_T                              1
#define CUBLAS_COMPUTE_32F                       0
#define CUDA_R_32F                               0
#define CUDA_R_8F_E4M3                           1
#define CUBLASLT_MATMUL_DESC_TRANSA              0
#define CUBLASLT_MATMUL_DESC_TRANSB              1
#define CUBLASLT_MATMUL_DESC_A_SCALE_MODE        2
#define CUBLASLT_MATMUL_DESC_B_SCALE_MODE        3
#define CUBLASLT_MATMUL_DESC_A_SCALE_POINTER     4
#define CUBLASLT_MATMUL_DESC_B_SCALE_POINTER     5
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES 0
#define CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 0

static inline cublasStatus_t cublasLtCreate(cublasLtHandle_t* h){if(h)*h=(void*)1;return 0;}
static inline cublasStatus_t cublasLtDestroy(cublasLtHandle_t){return 0;}
static inline cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* p,int,int){if(p)*p=(void*)1;return 0;}
static inline cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t){return 0;}
static inline cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t,int,const void*,size_t){return 0;}
static inline cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* p,int,unsigned long long,unsigned long long,long long){if(p)*p=(void*)1;return 0;}
static inline cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t){return 0;}
static inline cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* p){if(p)*p=(void*)1;return 0;}
static inline cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t){return 0;}
static inline cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t,int,const void*,size_t){return 0;}
static inline cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t,cublasLtMatmulDesc_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatrixLayout_t,cublasLtMatmulPreference_t,int,cublasLtMatmulHeuristicResult_t*,int* found){if(found)*found=0;return 0;}
static inline cublasStatus_t cublasLtMatmul(cublasLtHandle_t,cublasLtMatmulDesc_t,const void*,const void*,cublasLtMatrixLayout_t,const void*,cublasLtMatrixLayout_t,const void*,const void*,cublasLtMatrixLayout_t,void*,cublasLtMatrixLayout_t,const void*,void*,size_t,hipStream_t){return 0;}
