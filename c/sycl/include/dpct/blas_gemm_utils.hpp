//==---- blas_gemm_utils.hpp ----------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// This file contains the implementation of GEneral Matrix Multiplication by
// using oneDNN. More datatype combinations and the epilogue can be supported by
// oneDNN, which is not available in blas_utils.hpp using oneMKL.
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BLAS_GEMM_UTILS_HPP__
#define __DPCT_BLAS_GEMM_UTILS_HPP__

#include "compat_service.hpp"
#include "blas_utils.hpp"
#include "dnnl_utils.hpp"

namespace dpct::blas_gemm::experimental {
enum class order_t : std::uint8_t {
  col,
  row,
  col32,
  col4_4r2_8c,
  col32_2r_4r4
};
} // namespace dpct::blas_gemm::experimental

#include "detail/blas_gemm_utils_detail.hpp"

namespace dpct::blas_gemm::experimental {
enum class pointer_mode_t {
  host,
  device,
  device_vector,
  alpha_device_vector_beta_zero,
  alpha_device_vector_beta_host
};
enum class epilogue_t {
  nop = 1,
  relu,
  bias,
  gelu,
  gelu_bias,
  gelu_aux,
  gelu_aux_bias,
  dgelu,
  bgradb
};

class descriptor;
using descriptor_ptr = descriptor *;
class matrix_layout_t;
using matrix_layout_ptr = matrix_layout_t *;
class matmul_desc_t;
using matmul_desc_ptr = matmul_desc_t *;
class transform_desc_t;
using transform_desc_ptr = transform_desc_t *;

class descriptor {
public:
  descriptor() {}
  void init(sycl::queue *q_ptr) {
    _engine = ::dnnl::sycl_interop::make_engine(q_ptr->get_device(),
                                                q_ptr->get_context());
    _engine_stream = ::dnnl::sycl_interop::make_stream(_engine, *q_ptr);
  }
  ::dnnl::engine get_engine() const noexcept { return _engine; }
  ::dnnl::stream get_engine_stream() const noexcept { return _engine_stream; };

private:
  ::dnnl::engine _engine;
  ::dnnl::stream _engine_stream;
};

class matrix_layout_t {
public:
  enum class attribute {
    type,
    order,
    rows,
    cols,
    ld,
    batch_count,
    strided_batch_offset
  };

  matrix_layout_t(library_data_t type, std::uint64_t rows, std::uint64_t cols,
                  std::int64_t ld)
      : _type(type), _rows(rows), _cols(cols), _ld(ld) {}

  void set_attribute(attribute attr, const void *mem) {
    get_set_attr<true>(attr, const_cast<void *>(mem));
  }
  void get_attribute(attribute attr, void *mem) {
    get_set_attr<false>(attr, mem);
  }

private:
  template <bool is_set> void get_set_attr(attribute attr, void *mem) {
#define CASE(tag)                                                              \
  case attribute::tag:                                                         \
    if constexpr (is_set) {                                                    \
      _##tag = *static_cast<decltype(_##tag) *>(mem);                          \
    } else {                                                                   \
      *static_cast<decltype(_##tag) *>(mem) = _##tag;                          \
    }                                                                          \
    break;
    switch (attr) {
      CASE(type)
      CASE(order)
      CASE(rows)
      CASE(cols)
      CASE(ld)
      CASE(batch_count)
      CASE(strided_batch_offset)
    }
#undef CASE
  }

  library_data_t _type;
  order_t _order = order_t::col;
  std::uint64_t _rows;
  std::uint64_t _cols;
  std::int64_t _ld;
  std::uint64_t _batch_count = 1;
  std::uint64_t _strided_batch_offset = 0;

  friend sycl::event matmul(descriptor_ptr handle, matmul_desc_ptr computeDesc,
                            const void *alpha, const void *a,
                            matrix_layout_ptr a_desc, const void *b,
                            matrix_layout_ptr b_desc, const void *beta,
                            const void *c, matrix_layout_ptr c_desc, void *d,
                            matrix_layout_ptr d_desc,
                            ::dpct::cs::queue_ptr q_ptr);
  friend sycl::event
  matrix_transform(transform_desc_ptr transform_desc, const void *alpha,
                   const void *a, matrix_layout_ptr a_desc, const void *beta,
                   const void *b, matrix_layout_ptr b_desc, void *c,
                   matrix_layout_ptr c_desc, ::dpct::cs::queue_ptr q_ptr);
};

class matmul_desc_t {
public:
  enum class attribute {
    compute_type,
    scale_type,
    bias_data_type,
    bias_pointer,
    pointer_mode,
    trans_a,
    trans_b,
    trans_c,
    epilogue,
    epilogue_aux_ld,
    epilogue_aux_pointer,
    epilogue_aux_data_type,
    a_scale_pointer,
    b_scale_pointer,
    d_scale_pointer,
    absmax_d_pointer,
    unsupport
  };

  matmul_desc_t(compute_type compute_type, library_data_t scale_type)
      : _compute_type(compute_type), _scale_type(scale_type) {}

  void set_attribute(attribute attr, const void *mem) {
    if (attr != attribute::unsupport)
      get_set_attr<true>(attr, const_cast<void *>(mem));
  }
  void get_attribute(attribute attr, void *mem) {
    if (attr != attribute::unsupport)
      get_set_attr<false>(attr, mem);
  }

private:
  template <bool is_set> void get_set_attr(attribute attr, void *mem) {
#define CASE(tag)                                                              \
  case attribute::tag:                                                         \
    if constexpr (is_set) {                                                    \
      _##tag = *static_cast<decltype(_##tag) *>(mem);                          \
    } else {                                                                   \
      *static_cast<decltype(_##tag) *>(mem) = _##tag;                          \
    }                                                                          \
    break;
    switch (attr) {
      CASE(compute_type)
      CASE(scale_type)
      CASE(bias_data_type)
      CASE(bias_pointer)
      CASE(pointer_mode)
      CASE(trans_a)
      CASE(trans_b)
      CASE(trans_c)
      CASE(epilogue)
      CASE(a_scale_pointer)
      CASE(b_scale_pointer)
      CASE(d_scale_pointer)
      CASE(absmax_d_pointer)
      CASE(epilogue_aux_ld)
      CASE(epilogue_aux_pointer)
      CASE(epilogue_aux_data_type)
    default:
      break;
    }
#undef CASE
  }

  compute_type _compute_type;
  library_data_t _scale_type;
  library_data_t _bias_data_type = library_data_t::real_float;
  pointer_mode_t _pointer_mode = pointer_mode_t::host;
  oneapi::mkl::transpose _trans_a = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose _trans_b = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose _trans_c = oneapi::mkl::transpose::nontrans;
  epilogue_t _epilogue = epilogue_t::nop;
  library_data_t _epilogue_aux_data_type = library_data_t::real_float;
  size_t _epilogue_aux_ld = 0;
  void *_a_scale_pointer = nullptr;
  void *_b_scale_pointer = nullptr;
  void *_d_scale_pointer = nullptr;
  void *_absmax_d_pointer = nullptr;
  void *_bias_pointer = nullptr;
  void *_epilogue_aux_pointer = nullptr;

  friend sycl::event matmul(descriptor_ptr handle, matmul_desc_ptr computeDesc,
                            const void *alpha, const void *a,
                            matrix_layout_ptr a_desc, const void *b,
                            matrix_layout_ptr b_desc, const void *beta,
                            const void *c, matrix_layout_ptr c_desc, void *d,
                            matrix_layout_ptr d_desc,
                            ::dpct::cs::queue_ptr q_ptr);
};

/// This function does the following operations:
/// (1) D_temp = epilogue(alpha * scale_a * op_a(A) * scale_b * op_b(B) + beta * C)
/// (2) Amax = absmax(D_temp) when matmul_desc_t::attribute::absmax_d_pointer is specified
/// (3) D = scale_d * D_temp
///   "op_a" is specified by the matmul_desc_t::attribute::trans_a
///   (default is nontrans)
///   "op_b" is specified by the matmul_desc_t::attribute::trans_b
///   (default is nontrans)
///   "scale_a" is specified by the matmul_desc_t::attribute::a_scale_pointer
///   (default is nullptr, which behaves as 1.0)
///   "scale_b" is specified by the matmul_desc_t::attribute::b_scale_pointer
///   (default is nullptr, which behaves as 1.0)
///   "scale_d" is specified by the matmul_desc_t::attribute::d_scale_pointer
///   (default is nullptr, which behaves as 1.0)
///   "alpha" can be a scalar value or a vector, which is specified by the
///   matmul_desc_tattribute::::pointer_mode
///   "epilogue" is specified by the matmul_desc_t::attribute::epilogue
/// Currently, this function only supports the following type combinations:
///   scale_type==int32 && a_type==int8 && b_type==int8 && c_type==int32;
///   scale_type==float && a_type==int8 && b_type==int8 && c_type==int8;
///   scale_type==float && a_type==int8 && b_type==int8 && c_type==int32;
///   scale_type==float && a_type==float && b_type==float && c_type==float.
/// Currently, this function only supports beta==0 or beta==1.
/// Currently, this function only supports the relu, bias, gelu, gelu_bias,
/// gelu_aux, gelu_aux_bias, dgelu and bgradb epilogue.
/// NOTE: Non-col-major matrix will be converted to col-major matrix before.
/// TODO: Impl row-major matmul without layout conversion.
/// multiplication and converted back after multiplication.
/// \param [in] handle A handle containing context info.
/// \param [in] compute_desc Describe the computation.
/// \param [in] alpha Scaling factor alpha.
/// \param [in] a Input matrix A.
/// \param [in] a_desc Describe the matrix A.
/// \param [in] b Input matrix B.
/// \param [in] b_desc Describe the matrix B.
/// \param [in] beta Scaling factor beta.
/// \param [in] c Input matrix C.
/// \param [in] c_desc Describe the matrix C.
/// \param [out] d Output matrix D.
/// \param [in] d_desc Describe the matrix D.
/// \param [in] q_ptr The queue where the routine should be executed.
inline sycl::event matmul(descriptor_ptr handle, matmul_desc_ptr compute_desc,
                          const void *alpha, const void *a,
                          matrix_layout_ptr a_desc, const void *b,
                          matrix_layout_ptr b_desc, const void *beta,
                          const void *c, matrix_layout_ptr c_desc, void *d,
                          matrix_layout_ptr d_desc,
                          ::dpct::cs::queue_ptr q_ptr) {
  const size_t m = compute_desc->_trans_a == oneapi::mkl::transpose::nontrans
                       ? a_desc->_rows
                       : a_desc->_cols;
  const size_t n = d_desc->_cols;
  const size_t k = compute_desc->_trans_b == oneapi::mkl::transpose::nontrans
                       ? b_desc->_rows
                       : b_desc->_cols;
  const library_data_t a_type = a_desc->_type;
  const library_data_t b_type = b_desc->_type;
  const library_data_t c_type = c_desc->_type;
  const library_data_t d_type = d_desc->_type;
  const library_data_t scale_type = compute_desc->_scale_type;

  if (!q_ptr)
    q_ptr = &::dpct::cs::get_default_queue();
  handle->init(q_ptr);
  bool vector_alpha = false;
  bool device_alpha = false;
  if (compute_desc->_pointer_mode == pointer_mode_t::device_vector ||
      compute_desc->_pointer_mode ==
          pointer_mode_t::alpha_device_vector_beta_zero ||
      compute_desc->_pointer_mode ==
          pointer_mode_t::alpha_device_vector_beta_host) {
    vector_alpha = true;
    device_alpha = true;
  } else if (compute_desc->_pointer_mode == pointer_mode_t::device) {
    device_alpha = true;
  }

  bool beta_is_zero = true;
  if (beta != nullptr) {
    int beta_value = detail::type_dispatch<detail::get_beta_value_impl>(
        compute_desc->_scale_type, beta, q_ptr);
    if (beta_value != 0) {
      beta_is_zero = false;
      if (beta_value != 1)
        throw std::runtime_error(
            "dpct::blas_gemm::experimental::matmul() does "
            "not support non-zero and non-one beta currently.");
    }
  }

  if (compute_desc->_epilogue != epilogue_t::nop &&
      compute_desc->_epilogue != epilogue_t::relu &&
      compute_desc->_epilogue != epilogue_t::bias &&
      compute_desc->_epilogue != epilogue_t::gelu &&
      compute_desc->_epilogue != epilogue_t::gelu_bias &&
      compute_desc->_epilogue != epilogue_t::gelu_aux &&
      compute_desc->_epilogue != epilogue_t::gelu_aux_bias &&
      compute_desc->_epilogue != epilogue_t::dgelu &&
      compute_desc->_epilogue != epilogue_t::bgradb) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matmul() only "
        "supports relu, bias, gelu, gelu_bias, gelu_aux, "
        "gelu_aux_bias, dgelu and bgradb epilogue currently.");
  }

  if (!(compute_desc->_scale_type == library_data_t::real_int32 &&
        a_desc->_type == library_data_t::real_int8 &&
        b_desc->_type == library_data_t::real_int8 &&
        c_desc->_type == library_data_t::real_int32) &&
      !(compute_desc->_scale_type == library_data_t::real_float &&
        a_desc->_type == library_data_t::real_int8 &&
        b_desc->_type == library_data_t::real_int8 &&
        c_desc->_type == library_data_t::real_int8) &&
      !(compute_desc->_scale_type == library_data_t::real_float &&
        a_desc->_type == library_data_t::real_int8 &&
        b_desc->_type == library_data_t::real_int8 &&
        c_desc->_type == library_data_t::real_int32) &&
      !(compute_desc->_scale_type == library_data_t::real_float &&
        a_desc->_type == library_data_t::real_float &&
        b_desc->_type == library_data_t::real_float &&
        c_desc->_type == library_data_t::real_float)) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matmul() only supports data type "
        "combinataions:\n  scale_type==int32 && a_type==int8 && b_type==int8"
        "&& c_type==int32,\n  scale_type==float && a_type==int8 && "
        "b_type==int8 && c_type==int8,\n  scale_type==float && a_type==int8"
        "&& b_type==int8 && c_type==int32 or\n  scale_type==float && "
        "a_type==float"
        "&& b_type==float && c_type==float.");
  }

  if (a_desc->_batch_count != b_desc->_batch_count ||
      a_desc->_batch_count != c_desc->_batch_count ||
      a_desc->_batch_count != d_desc->_batch_count) {
    throw std::runtime_error("dpct::blas_gemm::experimental::matmul() does not "
                             "support different batch count.");
  }

#ifdef DPCT_USM_LEVEL_NONE
  if (a_desc->_batch_count != 1) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matmul() only supports "
        "batch mode with USM enabled.");
  }
#endif

  size_t a_elm_size = dpct::detail::library_data_size[static_cast<unsigned int>(
                          a_desc->_type)] /
                      8;
  size_t b_elm_size = dpct::detail::library_data_size[static_cast<unsigned int>(
                          b_desc->_type)] /
                      8;
  size_t c_elm_size = dpct::detail::library_data_size[static_cast<unsigned int>(
                          c_desc->_type)] /
                      8;
  size_t d_elm_size = dpct::detail::library_data_size[static_cast<unsigned int>(
                          d_desc->_type)] /
                      8;

  static const auto matmul_single =
      [](descriptor_ptr handle, matmul_desc_ptr compute_desc, const void *alpha,
         const void *a, matrix_layout_ptr a_desc, const void *b,
         matrix_layout_ptr b_desc, const void *beta, const void *c,
         matrix_layout_ptr c_desc, void *d, matrix_layout_ptr d_desc,
         ::dpct::cs::queue_ptr q_ptr, const size_t m, const size_t n,
         const size_t k, const library_data_t a_type,
         const library_data_t b_type, const library_data_t c_type,
         const library_data_t d_type, const library_data_t scale_type,
         bool vector_alpha, bool device_alpha, bool beta_is_zero,
         size_t a_elm_size, size_t b_elm_size, size_t c_elm_size,
         size_t d_elm_size) -> sycl::event {
    // For non-col_major matrix, convert it to col_major.
    const void *new_a = a;
    const void *new_b = b;
    const void *new_c = c;
    void *new_d = d;
    bool new_b_allocated = false;
    bool new_c_allocated = false;
    bool new_d_allocated = false;
    size_t new_lda = a_desc->_ld, new_ldb = b_desc->_ld, new_ldc = c_desc->_ld,
           new_ldd = d_desc->_ld;
    std::vector<sycl::event> transform_events;

    if (a_desc->_order != order_t::col)
      new_lda = a_desc->_rows;
    new_a = ::dpct::cs::malloc(a_elm_size * a_desc->_cols * new_lda, *q_ptr);
    sycl::event e_init;
    if (a_desc->_order != order_t::col)
      e_init = detail::type_dispatch<detail::matrix_transform_impl>(
          a_desc->_type, q_ptr, a_desc->_rows, a_desc->_cols, a_desc->_ld,
          a_desc->_order, (const std::int8_t *)a, new_lda, order_t::col,
          (std::int8_t *)new_a, std::vector<sycl::event>{});
    else
      e_init = ::dpct::cs::memcpy(
          *q_ptr, (void *)new_a, a, a_elm_size * a_desc->_cols * new_lda,
          ::dpct::cs::memcpy_direction::device_to_device);

    // alpha = alpha * scale_a * scale_b
    sycl::event e_scale_new_a = detail::scale_new_a(
        q_ptr, m, k, (void *)new_a, a_type, alpha, scale_type, vector_alpha,
        device_alpha, compute_desc->_a_scale_pointer,
        compute_desc->_b_scale_pointer, {e_init});

    transform_events.push_back(e_scale_new_a);

    if (b_desc->_order != order_t::col) {
      new_ldb = b_desc->_rows;
      new_b = ::dpct::cs::malloc(b_elm_size * b_desc->_cols * new_ldb, *q_ptr);
      new_b_allocated = true;
      sycl::event e = detail::type_dispatch<detail::matrix_transform_impl>(
          b_desc->_type, q_ptr, b_desc->_rows, b_desc->_cols, b_desc->_ld,
          b_desc->_order, b, new_ldb, order_t::col, const_cast<void *>(new_b),
          std::vector<sycl::event>{});
      transform_events.push_back(e);
    }

    if (!beta_is_zero && c_desc->_order != order_t::col) {
      new_ldc = c_desc->_rows;
      new_c = ::dpct::cs::malloc(c_elm_size * c_desc->_cols * new_ldc, *q_ptr);
      new_c_allocated = true;
      sycl::event e = detail::type_dispatch<detail::matrix_transform_impl>(
          c_desc->_type, q_ptr, c_desc->_rows, c_desc->_cols, c_desc->_ld,
          c_desc->_order, c, new_ldc, order_t::col, const_cast<void *>(new_c),
          std::vector<sycl::event>{});
      transform_events.push_back(e);
    }

    if (d_desc->_order != order_t::col) {
      new_ldd = d_desc->_rows;
      new_d = ::dpct::cs::malloc(d_elm_size * d_desc->_cols * new_ldd, *q_ptr);
      new_d_allocated = true;
    }

    // start to call oneDNN matmul primitive
    // a,d are col_major, b is row_major
    const ::dnnl::memory::dim M = m;
    const ::dnnl::memory::dim N = n;
    const ::dnnl::memory::dim K = k;

    ::dnnl::memory::dims src_dims = {M, K};
    ::dnnl::memory::dims weights_dims = {K, N};
    ::dnnl::memory::dims bias_dims = {M, N};
    ::dnnl::memory::dims dst_dims = {M, N};

    const ::dnnl::memory::dims src_strides =
        compute_desc->_trans_a == oneapi::mkl::transpose::nontrans
            ? ::dnnl::memory::dims{1, static_cast<long>(new_lda)}
            : ::dnnl::memory::dims{static_cast<long>(new_lda), 1};
    const ::dnnl::memory::dims weights_strides =
        compute_desc->_trans_b == oneapi::mkl::transpose::nontrans
            ? ::dnnl::memory::dims{1, static_cast<long>(new_ldb)}
            : ::dnnl::memory::dims{static_cast<long>(new_ldb), 1};
    const ::dnnl::memory::dims bias_strides =
        ::dnnl::memory::dims{1, static_cast<long>(new_ldc)};
    const ::dnnl::memory::dims dst_strides =
        ::dnnl::memory::dims{1, static_cast<long>(new_ldd)};

    auto src_md = ::dnnl::memory::desc(
        src_dims, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(a_type),
        src_strides);
    auto weights_md = ::dnnl::memory::desc(
        weights_dims, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(b_type),
        weights_strides);
    auto bias_md = ::dnnl::memory::desc(
        bias_dims, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(c_type),
        bias_strides);
    auto dst_md = ::dnnl::memory::desc(
        dst_dims, dpct::dnnl::memory_desc_ext::to_dnnl_data_type(d_type),
        dst_strides);

    auto *src_mem =
        new ::dnnl::memory(src_md, handle->get_engine(), DNNL_MEMORY_NONE);
    auto *weights_mem =
        new ::dnnl::memory(weights_md, handle->get_engine(), DNNL_MEMORY_NONE);
    auto *bias_mem =
        new ::dnnl::memory(bias_md, handle->get_engine(), DNNL_MEMORY_NONE);
    auto *dst_mem =
        new ::dnnl::memory(dst_md, handle->get_engine(), DNNL_MEMORY_NONE);

#ifdef DPCT_USM_LEVEL_NONE
    detail::type_dispatch<detail::set_buffer_impl>(a_type, src_mem, new_a);
    detail::type_dispatch<detail::set_buffer_impl>(b_type, weights_mem, new_b);
    if (!beta_is_zero)
      detail::type_dispatch<detail::set_buffer_impl>(c_type, bias_mem, new_c);
    detail::type_dispatch<detail::set_buffer_impl>(d_type, dst_mem, new_d);
#else
    src_mem->set_data_handle(const_cast<void *>(new_a));
    weights_mem->set_data_handle(const_cast<void *>(new_b));
    if (!beta_is_zero)
      bias_mem->set_data_handle(const_cast<void *>(new_c));
    dst_mem->set_data_handle(new_d);
#endif

    std::unordered_map<int, ::dnnl::memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, *src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, *weights_mem});
    matmul_args.insert({DNNL_ARG_DST, *dst_mem});
    ::dnnl::primitive_attr matmul_attr;

    ::dnnl::post_ops matmul_ops;
    if (!beta_is_zero) {
      matmul_ops.append_binary(::dnnl::algorithm::binary_add, bias_md);
      matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(matmul_ops.len() - 1) |
                              DNNL_ARG_SRC_1,
                          *bias_mem});
    }

    ::dnnl::memory *po_bias_mem = nullptr;
    auto po_bias_md =
        ::dnnl::memory::desc(::dnnl::memory::dims{M, 1},
                             dpct::dnnl::memory_desc_ext::to_dnnl_data_type(
                                 compute_desc->_bias_data_type),
                             ::dnnl::memory::dims{1, M});
    if (compute_desc->_epilogue == epilogue_t::bias ||
        compute_desc->_epilogue == epilogue_t::gelu_bias ||
        compute_desc->_epilogue == epilogue_t::gelu_aux_bias) {
      po_bias_mem = new ::dnnl::memory(po_bias_md, handle->get_engine(),
                                       DNNL_MEMORY_NONE);
#ifdef DPCT_USM_LEVEL_NONE
      detail::type_dispatch<detail::set_buffer_impl>(
          compute_desc->_bias_data_type, po_bias_mem,
          compute_desc->_bias_pointer);
#else
      po_bias_mem->set_data_handle(compute_desc->_bias_pointer);
#endif
    }

    ::dnnl::memory *po_bias_bgradb_mem = nullptr;
    auto po_bias_bgradb_md = ::dnnl::memory::desc(
        compute_desc->_trans_b == oneapi::mkl::transpose::nontrans
            ? ::dnnl::memory::dims{N, 1}
            : ::dnnl::memory::dims{1, N},
        dpct::dnnl::memory_desc_ext::to_dnnl_data_type(
            compute_desc->_bias_data_type),
        compute_desc->_trans_b == oneapi::mkl::transpose::nontrans
            ? ::dnnl::memory::dims{1, N}
            : ::dnnl::memory::dims{N, 1});
    if (compute_desc->_epilogue == epilogue_t::bgradb) {
      po_bias_bgradb_mem = new ::dnnl::memory(
          po_bias_bgradb_md, handle->get_engine(), DNNL_MEMORY_NONE);
#ifdef DPCT_USM_LEVEL_NONE
      detail::type_dispatch<detail::set_buffer_impl>(
          compute_desc->_bias_data_type, po_bias_bgradb_mem,
          compute_desc->_bias_pointer);
#else
      po_bias_bgradb_mem->set_data_handle(compute_desc->_bias_pointer);
#endif
    }

    ::dnnl::memory *po_aux_mem = nullptr;
    auto po_aux_md = ::dnnl::memory::desc(
        ::dnnl::memory::dims{M, N},
        dpct::dnnl::memory_desc_ext::to_dnnl_data_type(
            compute_desc->_epilogue_aux_data_type),
        ::dnnl::memory::dims{
            1, static_cast<long>(compute_desc->_epilogue_aux_ld)});
    if (compute_desc->_epilogue == epilogue_t::dgelu) {
      po_aux_mem =
          new ::dnnl::memory(po_aux_md, handle->get_engine(), DNNL_MEMORY_NONE);
#ifdef DPCT_USM_LEVEL_NONE
      detail::type_dispatch<detail::set_buffer_impl>(
          compute_desc->_epilogue_aux_data_type, po_aux_mem,
          compute_desc->_epilogue_aux_pointer);
#else
      po_aux_mem->set_data_handle(compute_desc->_epilogue_aux_pointer);
#endif
    }

    switch (compute_desc->_epilogue) {
    case epilogue_t::relu:
      matmul_ops.append_eltwise(::dnnl::algorithm::eltwise_relu, 0.f, 0.f);
      break;
    case epilogue_t::bias:
    case epilogue_t::gelu_bias: {
      matmul_ops.append_binary(::dnnl::algorithm::binary_add, po_bias_md);
      matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(matmul_ops.len() - 1) |
                              DNNL_ARG_SRC_1,
                          *po_bias_mem});
      if (compute_desc->_epilogue == epilogue_t::gelu_bias)
        matmul_ops.append_eltwise(::dnnl::algorithm::eltwise_gelu_tanh, 0.f,
                                  0.f);
      break;
    }
    case epilogue_t::gelu:
      matmul_ops.append_eltwise(::dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
      break;
    default:
      break;
    }

    matmul_attr.set_post_ops(matmul_ops);

    auto matmul_pd = ::dnnl::matmul::primitive_desc(
        handle->get_engine(), src_md, weights_md, dst_md, matmul_attr);
    auto matmul_prim = ::dnnl::matmul(matmul_pd);
    sycl::event matmul_prim_event =
        ::dnnl::sycl_interop::execute(matmul_prim, handle->get_engine_stream(),
                                      matmul_args, transform_events);

    // post-op implemented by separate primitives
    sycl::event post_op_prim_event;
    if (compute_desc->_epilogue == epilogue_t::gelu_aux ||
        compute_desc->_epilogue == epilogue_t::gelu_aux_bias) {
      sycl::event prev_event = matmul_prim_event;
      if (compute_desc->_epilogue == epilogue_t::gelu_aux_bias) {
        auto po_bias_pd = ::dnnl::binary::primitive_desc(
            handle->get_engine(), ::dnnl::algorithm::binary_add, dst_md,
            po_bias_md, dst_md);
        auto po_bias_prim = ::dnnl::binary(po_bias_pd);
        std::unordered_map<int, ::dnnl::memory> po_bias_args;
        po_bias_args.insert({DNNL_ARG_SRC_0, *dst_mem});
        po_bias_args.insert({DNNL_ARG_SRC_1, *po_bias_mem});
        po_bias_args.insert({DNNL_ARG_DST, *dst_mem});
        sycl::event prev_event = ::dnnl::sycl_interop::execute(
            po_bias_prim, handle->get_engine_stream(), po_bias_args,
            {matmul_prim_event});
      }
      size_t size_of_element =
          dpct::detail::library_data_size[static_cast<unsigned int>(
              d_desc->_type)] /
          8;
      sycl::event copy_e = dpct::blas::matrix_mem_copy_async(
          compute_desc->_epilogue_aux_pointer, new_d,
          compute_desc->_epilogue_aux_ld, d_desc->_ld, m, n, size_of_element,
          ::dpct::cs::memcpy_direction::automatic, *q_ptr, {prev_event});

      auto gelu_pd = ::dnnl::eltwise_forward::primitive_desc(
          handle->get_engine(), ::dnnl::prop_kind::forward_training,
          ::dnnl::algorithm::eltwise_gelu_tanh, dst_md, dst_md);
      auto gelu_prim = ::dnnl::eltwise_forward(gelu_pd);
      std::unordered_map<int, ::dnnl::memory> gelu_args;
      gelu_args.insert({DNNL_ARG_SRC, *dst_mem});
      gelu_args.insert({DNNL_ARG_DST, *dst_mem});
      post_op_prim_event = ::dnnl::sycl_interop::execute(
          gelu_prim, handle->get_engine_stream(), gelu_args, {copy_e});
    } else if (compute_desc->_epilogue == epilogue_t::dgelu) {
      auto gelu_pd = ::dnnl::eltwise_forward::primitive_desc(
          handle->get_engine(), ::dnnl::prop_kind::forward_training,
          ::dnnl::algorithm::eltwise_gelu_tanh, po_aux_md, po_aux_md);
      auto dgelu_pd = ::dnnl::eltwise_backward::primitive_desc(
          handle->get_engine(), ::dnnl::algorithm::eltwise_gelu_tanh, dst_md,
          dst_md, po_aux_md, gelu_pd);
      auto dgelu_prim = ::dnnl::eltwise_backward(dgelu_pd);
      std::unordered_map<int, ::dnnl::memory> dgelu_args;
      dgelu_args.insert({DNNL_ARG_SRC, *po_aux_mem});
      dgelu_args.insert({DNNL_ARG_DIFF_DST, *dst_mem});
      dgelu_args.insert({DNNL_ARG_DIFF_SRC, *dst_mem});
      post_op_prim_event =
          ::dnnl::sycl_interop::execute(dgelu_prim, handle->get_engine_stream(),
                                        dgelu_args, {matmul_prim_event});
    } else if (compute_desc->_epilogue == epilogue_t::bgradb) {
      auto reduction_pd = ::dnnl::reduction::primitive_desc(
          handle->get_engine(), ::dnnl::algorithm::reduction_sum, weights_md,
          po_bias_bgradb_md, 0.f, 0.f);
      auto reduction_prim = ::dnnl::reduction(reduction_pd);
      std::unordered_map<int, ::dnnl::memory> reduction_args;
      reduction_args.insert({DNNL_ARG_SRC, *weights_mem});
      reduction_args.insert({DNNL_ARG_DST, *po_bias_bgradb_mem});
      post_op_prim_event = ::dnnl::sycl_interop::execute(
          reduction_prim, handle->get_engine_stream(), reduction_args,
          {matmul_prim_event});
    }

#ifndef DPCT_USM_LEVEL_NONE
    // WA to avoid incorrect result on CPU with USM
    if (q_ptr->get_device().is_cpu()) {
      matmul_prim_event.wait();
      post_op_prim_event.wait();
    }
#endif
    // end of calling oneDNN

    sycl::event absmax_d_event;
    if (auto absmax_ptr = compute_desc->_absmax_d_pointer) {
      absmax_d_event = detail::type_dispatch<detail::absmax_impl>(
          d_desc->_type, absmax_ptr, new_d, new_ldd, d_desc->_rows,
          d_desc->_cols, q_ptr,
          std::vector<sycl::event>{matmul_prim_event, post_op_prim_event});
    }

    sycl::event scale_d_event;
    if (auto d_scale_ptr = compute_desc->_d_scale_pointer) {
      scale_d_event = detail::type_dispatch<detail::scale_d_impl>(
          d_desc->_type, d_scale_ptr, new_d, new_ldd, d_desc->_rows,
          d_desc->_cols, q_ptr, compute_desc->_scale_type,
          std::vector<sycl::event>{matmul_prim_event, absmax_d_event,
                                   post_op_prim_event});
    }

    sycl::event transform_d_event;
    if (d_desc->_order != order_t::col) {
      transform_d_event = detail::type_dispatch<detail::matrix_transform_impl>(
          d_desc->_type, q_ptr, d_desc->_rows, d_desc->_cols, new_ldd,
          order_t::col, new_d, d_desc->_ld, d_desc->_order, d,
          std::vector<sycl::event>{matmul_prim_event, absmax_d_event,
                                   post_op_prim_event});
    }

    sycl::event free_event = q_ptr->submit([&](sycl::handler &cgh) {
      cgh.depends_on({transform_d_event, matmul_prim_event, absmax_d_event,
                      post_op_prim_event});
      cgh.host_task([=] {
        delete src_mem;
        delete weights_mem;
        delete bias_mem;
        delete dst_mem;
        if (po_bias_mem)
          delete po_bias_mem;
        if (po_bias_bgradb_mem)
          delete po_bias_bgradb_mem;
        if (po_aux_mem)
          delete po_aux_mem;
        ::dpct::cs::free((void *)new_a, *q_ptr);
        if (new_b_allocated)
          ::dpct::cs::free((void *)new_b, *q_ptr);
        if (new_c_allocated)
          ::dpct::cs::free((void *)new_c, *q_ptr);
        if (new_d_allocated)
          ::dpct::cs::free((void *)new_d, *q_ptr);
      });
    });
    return free_event;
  };

  std::vector<sycl::event> events;
  const std::byte *offsetted_a = static_cast<const std::byte *>(a);
  const std::byte *offsetted_b = static_cast<const std::byte *>(b);
  const std::byte *offsetted_c = static_cast<const std::byte *>(c);
  std::byte *offsetted_d = static_cast<std::byte *>(d);
  for (std::uint64_t i = 0; i < a_desc->_batch_count; i++) {
    sycl::event e = matmul_single(
        handle, compute_desc, alpha, offsetted_a, a_desc, offsetted_b, b_desc,
        beta, offsetted_c, c_desc, offsetted_d, d_desc, q_ptr, m, n, k, a_type,
        b_type, c_type, d_type, scale_type, vector_alpha, device_alpha,
        beta_is_zero, a_elm_size, b_elm_size, c_elm_size, d_elm_size);
    events.push_back(e);

    offsetted_a += a_elm_size * a_desc->_strided_batch_offset;
    offsetted_b += b_elm_size * b_desc->_strided_batch_offset;
    offsetted_c += c_elm_size * c_desc->_strided_batch_offset;
    offsetted_d += d_elm_size * d_desc->_strided_batch_offset;
  }

  return q_ptr->single_task(events, [] {});
}

class transform_desc_t {
public:
  enum class attribute { scale_type, pointer_mode, trans_a, trans_b };

  transform_desc_t(library_data_t scale_type) : _scale_type(scale_type) {}
  void set_attribute(attribute attr, const void *mem) {
    get_set_attr<true>(attr, const_cast<void *>(mem));
  }
  void get_attribute(attribute attr, void *mem) {
    get_set_attr<false>(attr, mem);
  }

private:
  template <bool is_set> void get_set_attr(attribute attr, void *mem) {
#define CASE(tag)                                                              \
  case attribute::tag:                                                         \
    if constexpr (is_set) {                                                    \
      _##tag = *static_cast<decltype(_##tag) *>(mem);                          \
    } else {                                                                   \
      *static_cast<decltype(_##tag) *>(mem) = _##tag;                          \
    }                                                                          \
    break;
    switch (attr) {
      CASE(scale_type)
      CASE(pointer_mode)
      CASE(trans_a)
      CASE(trans_b)
    }
#undef CASE
  }

  library_data_t _scale_type;
  pointer_mode_t _pointer_mode = pointer_mode_t::host;
  oneapi::mkl::transpose _trans_a = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose _trans_b = oneapi::mkl::transpose::nontrans;

  friend sycl::event
  matrix_transform(transform_desc_ptr transform_desc, const void *alpha,
                   const void *a, matrix_layout_ptr a_desc, const void *beta,
                   const void *b, matrix_layout_ptr b_desc, void *c,
                   matrix_layout_ptr c_desc, ::dpct::cs::queue_ptr q_ptr);
};

/// This function does operation:
/// C = alpha*transformation(A) + beta*transformation(B).
/// The "transformation" includes matrix transpose and layout conversion.
/// Currently it only supports alpha==1 && beta==0.
/// \param [in] transform_desc Describe the transformation.
/// \param [in] alpha Scaling factor alpha.
/// \param [in] a Input matrix A.
/// \param [in] a_desc Describe the matrix A.
/// \param [in] beta Scaling factor beta.
/// \param [in] b Input matrix B.
/// \param [in] b_desc Describe the matrix B.
/// \param [out] c Output matrix C.
/// \param [in] c_desc Describe the matrix C.
/// \param [in] q_ptr The queue where the routine should be executed.
inline sycl::event matrix_transform(transform_desc_ptr transform_desc,
                                    const void *alpha, const void *a,
                                    matrix_layout_ptr a_desc, const void *beta,
                                    const void *b, matrix_layout_ptr b_desc,
                                    void *c, matrix_layout_ptr c_desc,
                                    ::dpct::cs::queue_ptr q_ptr) {
  if (!q_ptr)
    q_ptr = &::dpct::cs::get_default_queue();

  if (transform_desc->_pointer_mode != pointer_mode_t::host) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() "
        "only supports pointer_mode_t::host as pointer_mode currently.");
  }
  if (transform_desc->_scale_type != library_data_t::real_float) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() "
        "only supports library_data_t::real_float as scale_type currently.");
  }

  if (alpha != nullptr) {
    if (1.0f != *reinterpret_cast<const float *>(alpha))
      throw std::runtime_error(
          "dpct::blas_gemm::experimental::matrix_transform() does not "
          "support non-one alpha currently.");
  }

  if (beta != nullptr) {
    if (0.0f != *reinterpret_cast<const float *>(beta))
      throw std::runtime_error(
          "dpct::blas_gemm::experimental::matrix_transform() does not "
          "support non-zero beta currently.");
  }

  if (b != nullptr) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() does not "
        "support matrix B currently.");
  }

  if ((a_desc->_type != library_data_t::real_int8 ||
       c_desc->_type != library_data_t::real_int8) &&
      (a_desc->_type != library_data_t::real_int32 ||
       c_desc->_type != library_data_t::real_int32)) {
    throw std::runtime_error(
        "dpct::blas_gemm::experimental::matrix_transform() only supports "
        "combinations of data types: a_type==real_int8&&c_type==real_int8, "
        "a_type==real_int32&&c_type==real_int32.");
  }

  return detail::type_dispatch<detail::matrix_transform_impl>(
      a_desc->_type, q_ptr, a_desc->_rows, a_desc->_cols, a_desc->_ld,
      a_desc->_order, a, c_desc->_ld, c_desc->_order, c,
      std::vector<sycl::event>{});
}
} // namespace dpct::blas_gemm::experimental
#endif // __DPCT_BLAS_GEMM_UTILS_HPP__
