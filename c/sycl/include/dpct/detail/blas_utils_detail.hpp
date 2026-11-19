//==---- blas_utils_detail.hpp---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BLAS_UTILS_DETAIL_HPP__
#define __DPCT_BLAS_UTILS_DETAIL_HPP__

#include <thread>
#include <utility>
#include <vector>

namespace dpct {
namespace blas {
namespace detail {
template <typename target_t, typename source_t> class parameter_wrapper_base_t {
public:
  parameter_wrapper_base_t(sycl::queue q, source_t *source, size_t ele_num)
      : _source_attribute(::dpct::cs::detail::get_pointer_attribute(q, source)),
        _q(q), _source(source), _ele_num(ele_num),
        _target(construct_member_variable_target()) {}

  ~parameter_wrapper_base_t() {
    if (_need_free) {
      _q.submit([&](sycl::handler &cgh) {
        cgh.host_task([t = _target, q = _q] { ::dpct::cs::free(t, q); });
      });
    }
  }

protected:
  ::dpct::cs::detail::pointer_access_attribute _source_attribute;
  sycl::queue _q;
  source_t *_source = nullptr;
  size_t _ele_num;
  bool _need_free = true;
  target_t *_target;

private:
  target_t *construct_member_variable_target() {
    if constexpr (std::is_same_v<target_t, source_t>) {
      if (_source_attribute ==
          ::dpct::cs::detail::pointer_access_attribute::host_only)
        return (target_t *)::dpct::cs::malloc(sizeof(target_t) * _ele_num, _q);
#ifdef DPCT_USM_LEVEL_NONE
      auto alloc = dpct::detail::mem_mgr::instance().translate_ptr(_source);
      size_t offset = (byte_t *)_source - alloc.alloc_ptr;
      if (offset)
        return (target_t *)::dpct::cs::malloc(sizeof(target_t) * _ele_num, _q);
#endif
      // If (data type is same && it is device pointer && (USM || buffer offset
      // is 0)), it can be used directly.
      _need_free = false;
      return _source;
    } else {
      return (target_t *)::dpct::cs::malloc(sizeof(target_t) * _ele_num, _q);
    }
  }
};

inline library_data_t compute_type_to_library_data_t(compute_type ct) {
  switch (ct) {
  case compute_type::f16:
  case compute_type::f16_standard:
    return library_data_t::real_half;
  case compute_type::f32:
  case compute_type::f32_standard:
  case compute_type::f32_fast_bf16:
  case compute_type::f32_fast_tf32:
    return library_data_t::real_float;
  case compute_type::f64:
  case compute_type::f64_standard:
    return library_data_t::real_double;
  case compute_type::i32:
  case compute_type::i32_standard:
    return library_data_t::real_int32;
  default:
    throw std::runtime_error("conversion is not supported.");
  }
}
} // namespace detail
} // namespace blas

namespace detail {

inline void mem_free(sycl::queue *exec_queue,
                     std::vector<void *> pointers_array, sycl::event e) {
  e.wait();
  for (auto p : pointers_array)
    sycl::free(p, *exec_queue);
}

inline int stride_for(int num_elems, int mem_align_in_elems) {
  return ((num_elems - 1) / mem_align_in_elems + 1) * mem_align_in_elems;
}

#ifndef DPCT_USM_LEVEL_NONE
template <typename T> class working_memory {
  T *_input_ptr;
  T *_temp_ptr;
  bool _is_sycl_malloced = false;
  bool _is_scalar_value = false;
  sycl::queue _q;
  sycl::event _e;

public:
  working_memory(size_t size, sycl::queue q) : _q(q) {
    _is_scalar_value = false;
    _temp_ptr = (T *)sycl::malloc_device(size, q);
  }
  working_memory(T *result_ptr, sycl::queue q) : _input_ptr(result_ptr), _q(q) {
    _is_scalar_value = true;
    _is_sycl_malloced = sycl::get_pointer_type(_input_ptr, _q.get_context()) !=
                        sycl::usm::alloc::unknown;
    if (!_is_sycl_malloced)
      _temp_ptr = sycl::malloc_shared<T>(1, _q);
  }
  auto get_ptr() {
    if (_is_scalar_value && _is_sycl_malloced)
      return _input_ptr;
    return _temp_ptr;
  }
  void set_event(sycl::event e) { _e = e; }
  ~working_memory() {
    if (_is_scalar_value) {
      if (!_is_sycl_malloced) {
        _q.memcpy(_input_ptr, _temp_ptr, sizeof(T)).wait();
        sycl::free(_temp_ptr, _q);
      }
    } else {
      std::vector<void *> ptrs{_temp_ptr};
      ::dpct::cs::enqueue_free(ptrs, {_e});
    }
  }
};
#endif

template <typename Tx, typename Tr>
inline void nrm2_impl(sycl::queue &q, std::int64_t n, const void *x,
                      std::int64_t incx, void *result) {
#ifdef DPCT_USM_LEVEL_NONE
  auto x_buffer = dpct::get_buffer<Tx>(x);
  auto r_buffer =
      sycl::buffer<Tr, 1>(reinterpret_cast<Tr *>(result), sycl::range<1>(1));
  if (dpct::is_device_ptr(result))
    r_buffer = dpct::get_buffer<Tr>(result);
  oneapi::mkl::blas::column_major::nrm2(q, n, x_buffer, incx, r_buffer);
#else
  working_memory<Tr> res_mem(reinterpret_cast<Tr *>(result), q);
  oneapi::mkl::blas::column_major::nrm2(q, n, reinterpret_cast<const Tx *>(x),
                                        incx, res_mem.get_ptr());
#endif
}

template <bool is_conjugate, class Txy, class Tr>
inline void dotuc_impl(sycl::queue &q, std::int64_t n, const Txy *x,
                       std::int64_t incx, const Txy *y, std::int64_t incy,
                       Tr *result) {
#ifdef DPCT_USM_LEVEL_NONE
  auto x_buffer = dpct::get_buffer<Txy>(x);
  auto y_buffer = dpct::get_buffer<Txy>(y);
  auto r_buffer = sycl::buffer<Tr, 1>((Tr *)result, sycl::range<1>(1));
  if (dpct::is_device_ptr(result))
    r_buffer = dpct::get_buffer<Tr>(result);
  if constexpr (std::is_same_v<Txy, std::complex<float>> ||
                std::is_same_v<Txy, std::complex<double>>) {
    if constexpr (is_conjugate)
      oneapi::mkl::blas::column_major::dotc(q, n, x_buffer, incx, y_buffer,
                                            incy, r_buffer);
    else
      oneapi::mkl::blas::column_major::dotu(q, n, x_buffer, incx, y_buffer,
                                            incy, r_buffer);
  } else
    oneapi::mkl::blas::column_major::dot(q, n, x_buffer, incx, y_buffer, incy,
                                         r_buffer);
#else
  working_memory<Tr> res_mem(result, q);
  if constexpr (std::is_same_v<Txy, std::complex<float>> ||
                std::is_same_v<Txy, std::complex<double>>) {
    if constexpr (is_conjugate)
      oneapi::mkl::blas::column_major::dotc(q, n, x, incx, y, incy,
                                            res_mem.get_ptr());
    else
      oneapi::mkl::blas::column_major::dotu(q, n, x, incx, y, incy,
                                            res_mem.get_ptr());
  } else
    oneapi::mkl::blas::column_major::dot(q, n, x, incx, y, incy,
                                         res_mem.get_ptr());
#endif
}

template <bool is_conjugate>
inline void dotuc(sycl::queue &q, std::int64_t n, const void *x,
                  library_data_t x_type, std::int64_t incx, const void *y,
                  library_data_t y_type, std::int64_t incy, void *result,
                  library_data_t result_type) {
  std::uint64_t key =
      detail::get_type_combination_id(x_type, y_type, result_type);
  switch (key) {
  case detail::get_type_combination_id(library_data_t::real_float,
                                       library_data_t::real_float,
                                       library_data_t::real_float): {
    detail::dotuc_impl<is_conjugate>(q, n, reinterpret_cast<const float *>(x),
                                     incx, reinterpret_cast<const float *>(y),
                                     incy, reinterpret_cast<float *>(result));
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_double,
                                       library_data_t::real_double,
                                       library_data_t::real_double): {
    detail::dotuc_impl<is_conjugate>(q, n, reinterpret_cast<const double *>(x),
                                     incx, reinterpret_cast<const double *>(y),
                                     incy, reinterpret_cast<double *>(result));
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float,
                                       library_data_t::complex_float,
                                       library_data_t::complex_float): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const std::complex<float> *>(x), incx,
        reinterpret_cast<const std::complex<float> *>(y), incy,
        reinterpret_cast<std::complex<float> *>(result));
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double,
                                       library_data_t::complex_double,
                                       library_data_t::complex_double): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const std::complex<double> *>(x), incx,
        reinterpret_cast<const std::complex<double> *>(y), incy,
        reinterpret_cast<std::complex<double> *>(result));
    break;
  }
#ifdef __INTEL_MKL__
  case detail::get_type_combination_id(library_data_t::real_half,
                                       library_data_t::real_half,
                                       library_data_t::real_half): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const sycl::half *>(x), incx,
        reinterpret_cast<const sycl::half *>(y), incy,
        reinterpret_cast<sycl::half *>(result));
    break;
  }
#endif
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

template <class Tx, class Te>
inline void scal_impl(sycl::queue &q, std::int64_t n, const void *alpha,
                      void *x, std::int64_t incx) {
  Te alpha_val = get_value(reinterpret_cast<const Te *>(alpha), q);
  auto data_x = get_memory<Tx>(x);
  oneapi::mkl::blas::column_major::scal(q, n, alpha_val, data_x, incx);
}

template <class Txy, class Te>
inline void axpy_impl(sycl::queue &q, std::int64_t n, const void *alpha,
                      const void *x, std::int64_t incx, void *y,
                      std::int64_t incy) {
  Te alpha_val = get_value(reinterpret_cast<const Te *>(alpha), q);
  auto data_x = get_memory<const Txy>(x);
  auto data_y = get_memory<Txy>(y);
  oneapi::mkl::blas::column_major::axpy(q, n, alpha_val, data_x, incx, data_y,
                                        incy);
}

template <class Txy, class Tc, class Ts>
inline void rot_impl(sycl::queue &q, std::int64_t n, void *x, std::int64_t incx,
                     void *y, std::int64_t incy, const void *c, const void *s) {
  Tc c_value = get_value(reinterpret_cast<const Tc *>(c), q);
  Ts s_value = get_value(reinterpret_cast<const Ts *>(s), q);
  auto data_x = get_memory<Txy>(x);
  auto data_y = get_memory<Txy>(y);
  oneapi::mkl::blas::column_major::rot(q, n, data_x, incx, data_y, incy,
                                       c_value, s_value);
}

template <class Tx, class Ty, class Tparam>
inline void rotm_impl(sycl::queue &q, std::int64_t n, void *x, int64_t incx,
                      void *y, int64_t incy, const void *param) {
  auto data_x = get_memory<Tx>(x);
  auto data_y = get_memory<Ty>(y);
  auto data_param = get_memory<Tparam>(const_cast<void *>(param));
  oneapi::mkl::blas::column_major::rotm(q, n, data_x, incx, data_y, incy,
                                        data_param);
}

template <class Tx, class Ty>
inline void copy_impl(sycl::queue &q, std::int64_t n, const void *x,
                      std::int64_t incx, void *y, std::int64_t incy) {
  auto data_x = get_memory<const Tx>(x);
  auto data_y = get_memory<Ty>(y);
  oneapi::mkl::blas::column_major::copy(q, n, data_x, incx, data_y, incy);
}

template <class Tx, class Ty>
inline void swap_impl(sycl::queue &q, std::int64_t n, void *x,
                      std::int64_t incx, void *y, std::int64_t incy) {
  auto data_x = get_memory<Tx>(x);
  auto data_y = get_memory<Ty>(y);
  oneapi::mkl::blas::column_major::swap(q, n, data_x, incx, data_y, incy);
}

template <class Tx, class Tres>
inline void asum_impl(sycl::queue &q, std::int64_t n, const void *x,
                      std::int64_t incx, void *res) {
  auto data_x = get_memory<Tx>(x);
  auto data_res = get_memory<Tres>(res);
  oneapi::mkl::blas::column_major::asum(q, n, data_x, incx, data_res);
}

template <class T>
inline void iamax_impl(sycl::queue &q, std::int64_t n, const void *x,
                       std::int64_t incx, std::int64_t *res) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  auto data_x = get_memory<T>(x);
  auto data_res = get_memory<std::int64_t>(res);
  oneapi::mkl::blas::column_major::iamax(q, n, data_x, incx, data_res,
                                         oneapi::mkl::index_base::one);
#endif
}

template <class T>
inline void iamin_impl(sycl::queue &q, std::int64_t n, const void *x,
                       std::int64_t incx, std::int64_t *res) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  auto data_x = get_memory<T>(x);
  auto data_res = get_memory<std::int64_t>(res);
  oneapi::mkl::blas::column_major::iamin(q, n, data_x, incx, data_res,
                                         oneapi::mkl::index_base::one);
#endif
}

#ifdef __INTEL_MKL__
template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                      oneapi::mkl::transpose b_trans, int m, int n, int k,
                      const void *alpha, const void *a, int lda, const void *b,
                      int ldb, const void *beta, void *c, int ldc,
                      oneapi::mkl::blas::compute_mode cm) {
  Ts alpha_value = get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = get_value(reinterpret_cast<const Ts *>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k,
                                        alpha_value, data_a, lda, data_b, ldb,
                                        beta_value, data_c, ldc, cm);
}
#endif

#ifdef __INTEL_MKL__
#define DPCT_COMPUTE_MODE_PARAM , oneapi::mkl::blas::compute_mode cm
#define DPCT_COMPUTE_MODE_ARG , cm
#else
#define DPCT_COMPUTE_MODE_PARAM
#define DPCT_COMPUTE_MODE_ARG
#endif

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_batch_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                            oneapi::mkl::transpose b_trans, int m, int n, int k,
                            const void *alpha, const void **a, int lda,
                            const void **b, int ldb, const void *beta, void **c,
                            int ldc, int batch_size DPCT_COMPUTE_MODE_PARAM) {
  struct matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts value_info[2];
    std::int64_t size_info[3];
    std::int64_t ld_info[3];
    std::int64_t groupsize_info;
  };

  Ts alpha_value = get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = get_value(reinterpret_cast<const Ts *>(beta), q);

  matrix_info_t *matrix_info =
      (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
  matrix_info->transpose_info[0] = a_trans;
  matrix_info->transpose_info[1] = b_trans;
  matrix_info->value_info[0] = alpha_value;
  matrix_info->value_info[1] = beta_value;
  matrix_info->size_info[0] = m;
  matrix_info->size_info[1] = n;
  matrix_info->size_info[2] = k;
  matrix_info->ld_info[0] = lda;
  matrix_info->ld_info[1] = ldb;
  matrix_info->ld_info[2] = ldc;
  matrix_info->groupsize_info = batch_size;

  sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
      q, matrix_info->transpose_info, matrix_info->transpose_info + 1,
      matrix_info->size_info, matrix_info->size_info + 1,
      matrix_info->size_info + 2, matrix_info->value_info,
      reinterpret_cast<const Ta **>(a), matrix_info->ld_info,
      reinterpret_cast<const Tb **>(b), matrix_info->ld_info + 1,
      matrix_info->value_info + 1, reinterpret_cast<Tc **>(c),
      matrix_info->ld_info + 2, 1,
      &(matrix_info->groupsize_info)DPCT_COMPUTE_MODE_ARG);

  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); });
  });
}

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_batch_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                            oneapi::mkl::transpose b_trans, int m, int n, int k,
                            const void *alpha, const void *a, int lda,
                            long long int stride_a, const void *b, int ldb,
                            long long int stride_b, const void *beta, void *c,
                            int ldc, long long int stride_c,
                            int batch_size DPCT_COMPUTE_MODE_PARAM) {
  Ts alpha_value = get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = get_value(reinterpret_cast<const Ts *>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm_batch(
      q, a_trans, b_trans, m, n, k, alpha_value, data_a, lda, stride_a, data_b,
      ldb, stride_b, beta_value, data_c, ldc, stride_c,
      batch_size DPCT_COMPUTE_MODE_ARG);
}

template <bool is_hermitian, class T>
inline void syherk_impl(sycl::queue &q, oneapi::mkl::uplo uplo,
                        oneapi::mkl::transpose trans, int n, int k,
                        const void *alpha, const void *a, int lda,
                        const void *beta, void *c,
                        int ldc DPCT_COMPUTE_MODE_PARAM) {
  auto data_a = get_memory<const T>(a);
  auto data_c = get_memory<T>(c);
  if constexpr (is_hermitian) {
    auto alpha_value =
        get_value(reinterpret_cast<const typename T::value_type *>(alpha), q);
    auto beta_value =
        get_value(reinterpret_cast<const typename T::value_type *>(beta), q);
    oneapi::mkl::blas::column_major::herk(q, uplo, trans, n, k, alpha_value,
                                          data_a, lda, beta_value, data_c,
                                          ldc DPCT_COMPUTE_MODE_ARG);
  } else {
    T alpha_value = get_value(reinterpret_cast<const T *>(alpha), q);
    T beta_value = get_value(reinterpret_cast<const T *>(beta), q);
    oneapi::mkl::blas::column_major::syrk(q, uplo, trans, n, k, alpha_value,
                                          data_a, lda, beta_value, data_c,
                                          ldc DPCT_COMPUTE_MODE_ARG);
  }
}

template <bool is_hermitian, class T, class Tbeta>
inline void rk_impl(sycl::queue &q, oneapi::mkl::uplo uplo,
                    oneapi::mkl::transpose trans, int n, int k, const T *alpha,
                    const T *a, int lda, const T *b, int ldb, const Tbeta *beta,
                    T *c, int ldc DPCT_COMPUTE_MODE_PARAM) {
  // For symmetric matrix, this function performs: C = alpha*OP(A)*(OP(B))^T +
  // beta*C For Hermitian matrix, this function performs: C =
  // alpha*OP(A)*(OP(B))^H + beta*C The gemmt() function performs: C =
  // alpha*OPA(A)*OPB(B) + beta*C So the OPB need be updated before we call
  // gemmt().
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  using Ts = typename ::dpct::detail::lib_data_traits_t<Tbeta>;
  Ty alpha_value = get_value(reinterpret_cast<const Ty *>(alpha), q);
  Ts beta_value = get_value(reinterpret_cast<const Ts *>(beta), q);
  oneapi::mkl::transpose trans_A = trans, trans_B = trans;
  int origin_b_rows = trans == oneapi::mkl::transpose::nontrans ? n : k;
  int origin_b_cols = trans == oneapi::mkl::transpose::nontrans ? k : n;

  if ((is_hermitian && trans == oneapi::mkl::transpose::trans) ||
      (!is_hermitian && !std::is_floating_point_v<Ty> &&
       trans == oneapi::mkl::transpose::conjtrans)) {
    // In this case, OPB need be a conjugate operation,
    // but only notrans, conjtrans and trans are available.
    // So we need do a conjtrans operation first, then do a trans operation.
    trans_B = oneapi::mkl::transpose::trans;
    auto data_a = get_memory<const Ty>(a);
    auto data_c = get_memory<Ty>(c);
#ifdef DPCT_USM_LEVEL_NONE
    auto new_B_buffer =
        sycl::buffer<Ty, 1>(sycl::range<1>(origin_b_rows * origin_b_cols));
    auto from_buffer = dpct::get_buffer<Ty>(b);
    oneapi::mkl::blas::column_major::omatcopy_batch(
        q, oneapi::mkl::transpose::conjtrans, origin_b_rows, origin_b_cols,
        Ts(1.0), from_buffer, ldb, origin_b_rows * ldb, new_B_buffer,
        origin_b_cols, origin_b_rows * origin_b_cols, 1);
    oneapi::mkl::blas::column_major::gemmt(
        q, uplo, trans_A, trans_B, n, k, alpha_value, data_a, lda, new_B_buffer,
        origin_b_cols, beta_value, data_c, ldc DPCT_COMPUTE_MODE_ARG);
#else
    working_memory<T> new_B(origin_b_rows * origin_b_cols * sizeof(T), q);
    oneapi::mkl::blas::column_major::omatcopy_batch(
        q, oneapi::mkl::transpose::conjtrans, origin_b_rows, origin_b_cols,
        Ts(1.0), reinterpret_cast<const Ty *>(b), ldb, origin_b_rows * ldb,
        reinterpret_cast<Ty *>(new_B.get_ptr()), origin_b_cols,
        origin_b_rows * origin_b_cols, 1);
    sycl::event e = oneapi::mkl::blas::column_major::gemmt(
        q, uplo, trans_A, trans_B, n, k, alpha_value, data_a, lda,
        reinterpret_cast<Ty *>(new_B.get_ptr()), origin_b_cols, beta_value,
        data_c, ldc DPCT_COMPUTE_MODE_ARG);
    new_B.set_event(e);
#endif
  } else {
    if constexpr (is_hermitian) {
      trans_B = trans == oneapi::mkl::transpose::nontrans
                    ? oneapi::mkl::transpose::conjtrans
                    : oneapi::mkl::transpose::nontrans;
    } else {
      trans_B = trans == oneapi::mkl::transpose::nontrans
                    ? oneapi::mkl::transpose::trans
                    : oneapi::mkl::transpose::nontrans;
    }
    auto data_a = get_memory<const Ty>(a);
    auto data_b = get_memory<const Ty>(b);
    auto data_c = get_memory<Ty>(c);
    oneapi::mkl::blas::column_major::gemmt(
        q, uplo, trans_A, trans_B, n, k, alpha_value, data_a, lda, data_b, ldb,
        beta_value, data_c, ldc DPCT_COMPUTE_MODE_ARG);
  }
}

template <class Ta, class Tb, class Ts>
inline void
trsm_batch_impl(sycl::queue &q, oneapi::mkl::side left_right,
                oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                oneapi::mkl::diag unit_diag, int m, int n, const void *alpha,
                const void **a, int lda, void **b, int ldb,
                int batch_size DPCT_COMPUTE_MODE_PARAM) {
  struct matrix_info_t {
    matrix_info_t(oneapi::mkl::side side_info, oneapi::mkl::uplo uplo_info,
                  oneapi::mkl::transpose transpose_info,
                  oneapi::mkl::diag diag_info, Ts value_info, std::int64_t m,
                  std::int64_t n, std::int64_t lda, std::int64_t ldb,
                  std::int64_t groupsize_info)
        : side_info(side_info), uplo_info(uplo_info),
          transpose_info(transpose_info), diag_info(diag_info),
          value_info(value_info), groupsize_info(groupsize_info) {
      size_info[0] = m;
      size_info[1] = n;
      ld_info[0] = lda;
      ld_info[1] = ldb;
    }
    oneapi::mkl::side side_info;
    oneapi::mkl::uplo uplo_info;
    oneapi::mkl::transpose transpose_info;
    oneapi::mkl::diag diag_info;
    Ts value_info;
    std::int64_t size_info[2];
    std::int64_t ld_info[2];
    std::int64_t groupsize_info;
  };

  Ts alpha_value = get_value(reinterpret_cast<const Ts *>(alpha), q);

  matrix_info_t *matrix_info =
      new matrix_info_t(left_right, upper_lower, trans, unit_diag, alpha_value,
                        m, n, lda, ldb, batch_size);

  sycl::event e = oneapi::mkl::blas::column_major::trsm_batch(
      q, &(matrix_info->side_info), &(matrix_info->uplo_info),
      &(matrix_info->transpose_info), &(matrix_info->diag_info),
      matrix_info->size_info, matrix_info->size_info + 1,
      &(matrix_info->value_info), reinterpret_cast<const Ta **>(a),
      matrix_info->ld_info, reinterpret_cast<Tb **>(b),
      matrix_info->ld_info + 1, 1,
      &(matrix_info->groupsize_info)DPCT_COMPUTE_MODE_ARG);

  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete matrix_info; });
  });
}

template <typename T>
inline void getrfnp_batch_wrapper(sycl::queue &exec_queue, int n, T *a[],
                                  int lda, int *info, int batch_size) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  // Set the info array value to 0
  ::dpct::cs::fill<unsigned char>(exec_queue, info, 0,
                                  sizeof(int) * batch_size);
  std::int64_t stride_a = n * lda;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<Ty>(
          exec_queue, n, n, lda, stride_a, batch_size);

  Ty *a_strided_mem =
      (Ty *)::dpct::cs::malloc(stride_a * batch_size * sizeof(Ty), exec_queue);
  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  ::dpct::cs::memcpy(::dpct::cs::get_default_queue(), host_a, a,
                     batch_size * sizeof(T *))
      .wait();
  for (std::int64_t i = 0; i < batch_size; ++i)
    ::dpct::cs::memcpy(::dpct::cs::get_default_queue(),
                       a_strided_mem + i * stride_a, host_a[i],
                       n * lda * sizeof(T))
        .wait();

#ifdef DPCT_USM_LEVEL_NONE
  {
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    auto a_buffer = get_buffer<Ty>(a_strided_mem);
    oneapi::mkl::lapack::getrfnp_batch(exec_queue, n, n, a_buffer, lda,
                                       stride_a, batch_size, scratchpad,
                                       scratchpad_size);
  }
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(::dpct::cs::memcpy(
        exec_queue, host_a[i], a_strided_mem + i * stride_a,
        n * lda * sizeof(T), ::dpct::cs::memcpy_direction::automatic));
#else
  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  sycl::event e = oneapi::mkl::lapack::getrfnp_batch(
      exec_queue, n, n, a_strided_mem, lda, stride_a, batch_size, scratchpad,
      scratchpad_size);
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(::dpct::cs::memcpy(
        exec_queue, host_a[i], a_strided_mem + i * stride_a,
        n * lda * sizeof(T), ::dpct::cs::memcpy_direction::automatic, {e}));

  std::vector<void *> ptrs{scratchpad, a_strided_mem};
  ::dpct::cs::enqueue_free(ptrs, events, exec_queue);
#endif

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.host_task([=] { std::free(host_a); });
  });
#endif
}
} // namespace detail
} // namespace dpct

#endif // __DPCT_BLAS_UTILS_DETAIL_HPP__
