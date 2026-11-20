//==---- lapack_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LAPACK_UTILS_HPP__
#define __DPCT_LAPACK_UTILS_HPP__

#include "compat_service.hpp"
#include "lib_common_utils.hpp"

#include "detail/lapack_utils_detail.hpp"

namespace dpct {
namespace lapack {
/// Computes all eigenvalues and, optionally, eigenvectors of a real generalized
/// symmetric definite eigenproblem using a divide and conquer method.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] itype Must be 1 or 2 or 3. Specifies the problem type to be solved.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrices A and B.
/// \param [in,out] a The symmetric matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b The symmetric matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [out] w Eigenvalues.
/// \param [in] scratchpad Scratchpad memory to be used by the routine
/// for storing intermediate results.
/// \param [in] scratchpad_size Size of scratchpad memory as a number of
/// floating point elements of type T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T>
inline int sygvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                 oneapi::mkl::uplo uplo, int n, T *a, int lda, T *b, int ldb,
                 T *w, T *scratchpad, int scratchpad_size, int *info) {
#ifdef DPCT_USM_LEVEL_NONE
  auto info_buf = get_buffer<int>(info);
  auto a_buffer = get_buffer<T>(a);
  auto b_buffer = get_buffer<T>(b);
  auto w_buffer = get_buffer<T>(w);
  auto scratchpad_buffer = get_buffer<T>(scratchpad);
  int info_val = 0;
  int ret_val = 0;
  try {
    oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a_buffer, lda,
                               b_buffer, ldb, w_buffer, scratchpad_buffer,
                               scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: sygvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    info_val = static_cast<int>(e.info());
    ret_val = 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    ret_val = 1;
  }
  queue.submit([&, info_val](sycl::handler &cgh) {
    auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<::dpct::cs::kernel_name<class sygvd_set_info, T>>(
        [=]() { info_acc[0] = info_val; });
  });
  return ret_val;
#else
  try {
    oneapi::mkl::lapack::sygvd(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                               scratchpad, scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: sygvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    int info_val = static_cast<int>(e.info());
    queue.memcpy(info, &info_val, sizeof(int)).wait();
    return 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, sizeof(int)).wait();
    return 1;
  }
  queue.memset(info, 0, sizeof(int));
  return 0;
#endif
}
/// Computes all the eigenvalues, and optionally, the eigenvectors of a complex
/// generalized Hermitian positive-definite eigenproblem using a divide and
/// conquer method.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] itype Must be 1 or 2 or 3. Specifies the problem type to be solved.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrices A and B.
/// \param [in,out] a The Hermitian matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b The Hermitian matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [in] w Eigenvalues.
/// \param [in] scratchpad Scratchpad memory to be used by the routine
/// for storing intermediate results.
/// \param [in] scratchpad_size Size of scratchpad memory as a number of
/// floating point elements of type T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T, typename Tw>
inline int hegvd(sycl::queue &queue, std::int64_t itype, oneapi::mkl::job jobz,
                 oneapi::mkl::uplo uplo, int n, T *a, int lda, T *b, int ldb,
                 Tw *w, T *scratchpad, int scratchpad_size, int *info) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
#ifdef DPCT_USM_LEVEL_NONE
  auto info_buf = get_buffer<int>(info);
  auto a_buffer = get_buffer<Ty>(a);
  auto b_buffer = get_buffer<Ty>(b);
  auto w_buffer = get_buffer<Tw>(w);
  auto scratchpad_buffer = get_buffer<Ty>(scratchpad);
  int info_val = 0;
  int ret_val = 0;
  try {
    oneapi::mkl::lapack::hegvd(queue, itype, jobz, uplo, n, a_buffer, lda,
                               b_buffer, ldb, w_buffer, scratchpad_buffer,
                               scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: hegvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    info_val = static_cast<int>(e.info());
    ret_val = 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    ret_val = 1;
  }
  queue.submit([&, info_val](sycl::handler &cgh) {
    auto info_acc = info_buf.get_access<sycl::access_mode::write>(cgh);
    cgh.single_task<::dpct::cs::kernel_name<class hegvd_set_info, T>>(
        [=]() { info_acc[0] = info_val; });
  });
  return ret_val;
#else
  try {
    oneapi::mkl::lapack::hegvd(queue, itype, jobz, uplo, n, (Ty *)a, lda, (Ty *)b,
                               ldb, w, (Ty *)scratchpad, scratchpad_size);
  } catch (oneapi::mkl::lapack::exception const& e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: hegvd"
              << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl;
    int info_val = static_cast<int>(e.info());
    queue.memcpy(info, &info_val, sizeof(int)).wait();
    return 1;
  } catch (sycl::exception const& e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    queue.memset(info, 0, sizeof(int)).wait();
    return 1;
  }
  queue.memset(info, 0, sizeof(int));
  return 0;
#endif
}
/// Computes the Cholesky factorizations of a batch of symmetric (or Hermitian,
/// for complex data) positive-definite matrices.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in,out] a Array of pointers to matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
/// \param [in] group_size The batch size.
template <typename T>
inline int potrf_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, int n,
                       T *a[], int lda, int *info, int group_size) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  struct matrix_info_t {
    oneapi::mkl::uplo uplo_info;
    std::int64_t n_info;
    std::int64_t lda_info;
    std::int64_t group_size_info;
  };
  matrix_info_t *matrix_info =
      (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
  matrix_info->uplo_info = uplo;
  matrix_info->n_info = n;
  matrix_info->lda_info = lda;
  matrix_info->group_size_info = group_size;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::potrf_batch_scratchpad_size<Ty>(
          queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
          &(matrix_info->lda_info), 1, &(matrix_info->group_size_info));
  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, queue);
  int has_execption = 0;

  static const std::vector<sycl::event> empty_events{};
  static const std::string api_name = "oneapi::mkl::lapack::potrf_batch";
  if (info)
    queue.memset(info, 0, group_size * sizeof(int));
  sycl::event e = ::dpct::detail::catch_batch_error_f<sycl::event>(
      &has_execption, api_name, queue, nullptr, info,
      matrix_info->group_size_info, oneapi::mkl::lapack::potrf_batch, queue,
      &(matrix_info->uplo_info), &(matrix_info->n_info), (Ty **)a,
      &(matrix_info->lda_info), (std::int64_t)1,
      &(matrix_info->group_size_info), (Ty *)scratchpad,
      (std::int64_t)scratchpad_size, empty_events);

  queue.submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() mutable {
      ::dpct::detail::catch_batch_error(
          nullptr, api_name, queue, nullptr, info, matrix_info->group_size_info,
          [](sycl::event _e) {
            _e.wait_and_throw();
            return 0;
          },
          e);
      std::free(matrix_info);
      sycl::free(scratchpad, queue);
    });
  });
  return has_execption;
#endif
}
/// Solves a batch of systems of linear equations with a Cholesky-factored
/// symmetric (Hermitian) positive-definite coefficient matrices.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] queue Device queue where calculations will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] nrhs The number of right-hand sides.
/// \param [in,out] a Array of pointers to matrix A.
/// \param [in] lda The leading dimension of matrix A.
/// \param [in,out] b Array of pointers to matrix B.
/// \param [in] ldb The leading dimension of matrix B.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
/// \param [in] group_size The batch size.
template <typename T>
inline int potrs_batch(sycl::queue &queue, oneapi::mkl::uplo uplo, int n,
                       int nrhs, T *a[], int lda, T *b[], int ldb, int *info,
                       int group_size) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  struct matrix_info_t {
    oneapi::mkl::uplo uplo_info;
    std::int64_t n_info;
    std::int64_t nrhs_info;
    std::int64_t lda_info;
    std::int64_t ldb_info;
    std::int64_t group_size_info;
  };
  matrix_info_t *matrix_info =
      (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
  matrix_info->uplo_info = uplo;
  matrix_info->n_info = n;
  matrix_info->nrhs_info = nrhs;
  matrix_info->lda_info = lda;
  matrix_info->ldb_info = ldb;
  matrix_info->group_size_info = group_size;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::potrs_batch_scratchpad_size<Ty>(
          queue, &(matrix_info->uplo_info), &(matrix_info->n_info),
          &(matrix_info->nrhs_info), &(matrix_info->lda_info),
          &(matrix_info->ldb_info), 1, &(matrix_info->group_size_info));
  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, queue);
  int has_execption = 0;

  if (info)
    queue.memset(info, 0, group_size * sizeof(int));

  static const std::vector<sycl::event> empty_events{};
  static const std::string api_name = "oneapi::mkl::lapack::potrs_batch";
  sycl::event e = ::dpct::detail::catch_batch_error_f<sycl::event>(
      &has_execption, api_name, queue, nullptr, info,
      matrix_info->group_size_info, oneapi::mkl::lapack::potrs_batch, queue,
      &(matrix_info->uplo_info), &(matrix_info->n_info),
      &(matrix_info->nrhs_info), (const Ty *const *)a, &(matrix_info->lda_info),
      (Ty **)b, &(matrix_info->ldb_info), (std::int64_t)1,
      &(matrix_info->group_size_info), (Ty *)scratchpad,
      (std::int64_t)scratchpad_size, empty_events);

  queue.submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() mutable {
      ::dpct::detail::catch_batch_error(
          nullptr, api_name, queue, nullptr, info, matrix_info->group_size_info,
          [](sycl::event _e) {
            _e.wait_and_throw();
            return 0;
          },
          e);
      std::free(matrix_info);
      sycl::free(scratchpad, queue);
    });
  });
  return has_execption;
#endif
}

/// Computes the size of workspace memory of getrf function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The workspace size in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int getrf_scratchpad_size(sycl::queue &q, std::int64_t m, std::int64_t n,
                                 library_data_t a_type, std::int64_t lda,
                                 std::size_t *device_ws_size,
                                 std::size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::getrf_scratchpad_size_impl>(
      q, a_type, nullptr, "getrf_scratchpad_size", q, m, n, a_type, lda,
      device_ws_size_tmp);
  *device_ws_size = detail::element_number_to_byte(device_ws_size_tmp, a_type);
  return ret;
}

/// Computes the LU factorization of a general m-by-n matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. Overwritten by L and U. The unit
/// diagonal elements of L are not stored.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] ipiv The pivot indices. If \p ipiv is nullptr, non-pivoting
/// LU factorization is computed.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int getrf(sycl::queue &q, std::int64_t m, std::int64_t n,
                 library_data_t a_type, void *a, std::int64_t lda,
                 std::int64_t *ipiv, void *device_ws,
                 std::size_t device_ws_size, int *info) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  std::size_t device_ws_size_in_element_number =
      detail::byte_to_element_number(device_ws_size, a_type);
  if (ipiv == nullptr) {
    return detail::lapack_shim<detail::getrfnp_impl>(
        q, a_type, info, "getrfnp_batch", q, m, n, a_type, a, lda, ipiv,
        device_ws, device_ws_size_in_element_number, info);
  }
  return detail::lapack_shim<detail::getrf_impl>(
      q, a_type, info, "getrf", q, m, n, a_type, a, lda, ipiv, device_ws,
      device_ws_size_in_element_number, info);
#endif
}

/// Solves a system of linear equations with a LU-factored square coefficient
/// matrix, with multiple right-hand sides.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] trans Indicates the form of the linear equation.
/// \param [in] n The order of the matrix A and the number of rows in matrix B.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] a The input matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] ipiv The pivot indices.
/// \param [in] b_type The data type of the matrix B.
/// \param [in, out] b The matrix B, whose columns are the right-hand sides
/// for the systems of equations.
/// \param [in] ldb The leading dimension of the matrix B.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int getrs(sycl::queue &q, oneapi::mkl::transpose trans, std::int64_t n,
                 std::int64_t nrhs, library_data_t a_type, void *a,
                 std::int64_t lda, std::int64_t *ipiv, library_data_t b_type,
                 void *b, std::int64_t ldb, int *info) {
  return detail::lapack_shim<detail::getrs_impl>(
      q, a_type, info, "getrs_scratchpad_size/getrs", q, trans, n, nrhs, a_type,
      a, lda, ipiv, b_type, b, ldb, info);
}

/// Computes the size of workspace memory of geqrf function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The device workspace size in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int geqrf_scratchpad_size(sycl::queue &q, std::int64_t m, std::int64_t n,
                                 library_data_t a_type, std::int64_t lda,
                                 std::size_t *device_ws_size,
                                 std::size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::geqrf_scratchpad_size_impl>(
      q, a_type, nullptr, "geqrf_scratchpad_size", q, m, n, a_type, lda,
      device_ws_size_tmp);
  *device_ws_size = detail::element_number_to_byte(device_ws_size_tmp, a_type);
  return ret;
}

/// Computes the QR factorization of a general m-by-n matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. Overwritten by the factorization data.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] tau_type The data type of the array tau.
/// \param [in] tau The array contains scalars that define elementary reflectors
/// for the matrix Q in its decomposition in a product of elementary reflectors.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int geqrf(sycl::queue &q, std::int64_t m, std::int64_t n,
                 library_data_t a_type, void *a, std::int64_t lda,
                 library_data_t tau_type, void *tau, void *device_ws,
                 std::size_t device_ws_size, int *info) {
  std::size_t device_ws_size_in_element_number =
      detail::byte_to_element_number(device_ws_size, a_type);
  return detail::lapack_shim<detail::geqrf_impl>(
      q, a_type, info, "geqrf", q, m, n, a_type, a, lda, tau_type, tau,
      device_ws, device_ws_size_in_element_number, info);
}

/// Computes the size of workspace memory of gesvd function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobu Must be 'A' (representing jobsvd::vectors), 'S'
/// (representing jobsvd::somevec), 'O' (representing jobsvd::vectorsina) or 'N'
/// (representing jobsvd::novec).
/// \param [in] jobvt Must be 'A' (representing jobsvd::vectors), 'S'
/// (representing jobsvd::somevec), 'O' (representing jobsvd::vectorsina) or 'N'
/// (representing jobsvd::novec).
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] u_type The data type of the matrix U.
/// \param [in] ldu The leading dimension of the matrix U.
/// \param [in] vt_type The data type of the matrix VT.
/// \param [in] ldvt The leading dimension of the matrix VT.
/// \param [out] device_ws_size The device workspace size in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int gesvd_scratchpad_size(sycl::queue &q, signed char jobu,
                                 signed char jobvt, std::int64_t m,
                                 std::int64_t n, library_data_t a_type,
                                 std::int64_t lda, library_data_t u_type,
                                 std::int64_t ldu, library_data_t vt_type,
                                 std::int64_t ldvt, std::size_t *device_ws_size,
                                 std::size_t *host_ws_size = nullptr) {
  oneapi::mkl::jobsvd jobu_enum = detail::char2jobsvd(jobu);
  oneapi::mkl::jobsvd jobvt_enum = detail::char2jobsvd(jobvt);
  if (host_ws_size)
    *host_ws_size = 0;
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::gesvd_scratchpad_size_impl>(
      q, a_type, nullptr, "gesvd_scratchpad_size", q, jobu_enum, jobvt_enum, m,
      n, a_type, lda, u_type, ldu, vt_type, ldvt, device_ws_size_tmp);
  *device_ws_size = detail::element_number_to_byte(device_ws_size_tmp, a_type);
  return ret;
}

/// Computes the size of workspace memory of gesvd function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::vec or job::novec
/// \param [in] all_vec Only have effects when \param jobz is job::vec.If the
/// value is zero, all m columns of U are returned in the matrix U, otherwise
/// the first min( \param m, \param n ) columns of U (the left singular vectors)
/// are returned in the matrix U.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] u_type The data type of the matrix U.
/// \param [in] ldu The leading dimension of the matrix U.
/// \param [in] vt_type The data type of the matrix VT.
/// \param [in] ldvt The leading dimension of the matrix VT.
/// \param [out] device_ws_size The device workspace size as a number of
/// elements of type \param a_type.
/// \param [out] host_ws_size The host workspace size as a number of elements
/// of type \param a_type. Currently the value is always zero.
inline int gesvd_scratchpad_size(sycl::queue &q, oneapi::mkl::job jobz,
                                 std::int64_t all_vec, std::int64_t m,
                                 std::int64_t n, library_data_t a_type,
                                 std::int64_t lda, library_data_t u_type,
                                 std::int64_t ldu, library_data_t vt_type,
                                 std::int64_t ldvt, int *device_ws_size,
                                 std::size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  oneapi::mkl::jobsvd jobu;
  oneapi::mkl::jobsvd jobvt;
  if (jobz == oneapi::mkl::job::vec) {
    if (all_vec) {
      jobu = jobvt = oneapi::mkl::jobsvd::somevec;
    } else {
      jobu = jobvt = oneapi::mkl::jobsvd::vectors;
    }
  } else if (jobz == oneapi::mkl::job::novec) {
    jobu = jobvt = oneapi::mkl::jobsvd::novec;
  } else {
    throw std::runtime_error("the job type is unsupported");
  }
  std::size_t device_ws_size_64;
  int ret = detail::lapack_shim<detail::gesvd_scratchpad_size_impl>(
      q, a_type, nullptr, "gesvd_scratchpad_size", q, jobu, jobvt, m, n, a_type,
      lda, u_type, ldu, vt_type, ldvt, device_ws_size_64);
  *device_ws_size = device_ws_size_64;
  return ret;
}

/// Computes the size of workspace memory of gesvd function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobu Must be 'A' (representing jobsvd::vectors), 'S'
/// (representing jobsvd::somevec), 'O' (representing jobsvd::vectorsina) or 'N'
/// (representing jobsvd::novec).
/// \param [in] jobvt Must be 'A' (representing jobsvd::vectors), 'S'
/// (representing jobsvd::somevec), 'O' (representing jobsvd::vectorsina) or 'N'
/// (representing jobsvd::novec).
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A and it will be overwritten according
/// to \p jobu and \p jobvt.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] s_type The data type of the matrix S.
/// \param [out] s The output matrix S.
/// \param [in] u_type The data type of the matrix U.
/// \param [out] u The output matrix U.
/// \param [in] ldu The leading dimension of the matrix U.
/// \param [in] vt_type The data type of the matrix VT.
/// \param [out] vt The output matrix VT.
/// \param [in] ldvt The leading dimension of the matrix VT.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int gesvd(sycl::queue &q, signed char jobu, signed char jobvt,
                 std::int64_t m, std::int64_t n, library_data_t a_type, void *a,
                 std::int64_t lda, library_data_t s_type, void *s,
                 library_data_t u_type, void *u, std::int64_t ldu,
                 library_data_t vt_type, void *vt, std::int64_t ldvt,
                 void *device_ws, std::size_t device_ws_size, int *info) {
  oneapi::mkl::jobsvd jobu_enum = detail::char2jobsvd(jobu);
  oneapi::mkl::jobsvd jobvt_enum = detail::char2jobsvd(jobvt);
  std::size_t device_ws_size_in_element_number =
      detail::byte_to_element_number(device_ws_size, a_type);
  return detail::lapack_shim<detail::gesvd_impl>(
      q, a_type, info, "gesvd", q, jobu_enum, jobvt_enum, m, n, a_type, a, lda,
      s_type, s, u_type, u, ldu, vt_type, vt, ldvt, device_ws,
      device_ws_size_in_element_number, info);
}

/// Computes the size of workspace memory of gesvd function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::vec or job::novec.
/// \param [in] all_vec Only have effects when \param jobz is job::vec.If the
/// value is zero, all m columns of U are returned in the matrix U, otherwise
/// the first min( \param m, \param n ) columns of U (the left singular vectors)
/// are returned in the matrix U.
/// \param [in] m The number of rows in the matrix A.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A and it will be overwritten according
/// to \p jobu and \p jobvt.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] s_type The data type of the matrix S.
/// \param [out] s The output matrix S.
/// \param [in] u_type The data type of the matrix U.
/// \param [out] u The output matrix U.
/// \param [in] ldu The leading dimension of the matrix U.
/// \param [in] vt_type The data type of the matrix VT.
/// \param [out] vt The output matrix VT.
/// \param [in] ldvt The leading dimension of the matrix VT.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The device workspace size as a number of
/// elements of type \param a_type.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int gesvd(sycl::queue &q, oneapi::mkl::job jobz, std::int64_t all_vec,
                 std::int64_t m, std::int64_t n, library_data_t a_type, void *a,
                 std::int64_t lda, library_data_t s_type, void *s,
                 library_data_t u_type, void *u, std::int64_t ldu,
                 library_data_t vt_type, void *vt, std::int64_t ldvt,
                 void *device_ws, std::size_t device_ws_size, int *info) {
  oneapi::mkl::jobsvd jobu;
  oneapi::mkl::jobsvd jobvt;
  if (jobz == oneapi::mkl::job::vec) {
    if (all_vec) {
      jobu = jobvt = oneapi::mkl::jobsvd::somevec;
    } else {
      jobu = jobvt = oneapi::mkl::jobsvd::vectors;
    }
  } else if (jobz == oneapi::mkl::job::novec) {
    jobu = jobvt = oneapi::mkl::jobsvd::novec;
  } else {
    throw std::runtime_error("the job type is unsupported");
  }

  detail::lapack_shim<detail::gesvd_conj_impl>(
      q, a_type, info, "gesvd", q, jobu, jobvt, m, n, a_type, a, lda, s_type, s,
      u_type, u, ldu, vt_type, vt, ldvt, device_ws, device_ws_size, info);
  return 0;
}

/// Computes the size of workspace memory of potrf function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The device workspace size in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int potrf_scratchpad_size(sycl::queue &q, oneapi::mkl::uplo uplo,
                                 std::int64_t n, library_data_t a_type,
                                 std::int64_t lda, std::size_t *device_ws_size,
                                 std::size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::potrf_scratchpad_size_impl>(
      q, a_type, nullptr, "potrf_scratchpad_size", q, uplo, n, a_type, lda,
      device_ws_size_tmp);
  *device_ws_size = detail::element_number_to_byte(device_ws_size_tmp, a_type);
  return ret;
}

/// Computes the Cholesky factorization of a symmetric (Hermitian)
/// positive-definite matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The number of columns in the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. Overwritten by the Cholesky factor U
/// or L, as specified by \p uplo.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int potrf(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                 library_data_t a_type, void *a, std::int64_t lda,
                 void *device_ws, std::size_t device_ws_size, int *info) {
  std::size_t device_ws_size_in_element_number =
      detail::byte_to_element_number(device_ws_size, a_type);
  return detail::lapack_shim<detail::potrf_impl>(
      q, a_type, info, "potrf", q, uplo, n, a_type, a, lda, device_ws,
      device_ws_size_in_element_number, info);
}

/// Solves a system of linear equations with a Cholesky-factored symmetric
/// (Hermitian) positive-definite coefficient matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A and the number of rows in matrix B.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. Overwritten by the Cholesky factor U
/// or L, as specified by \p uplo.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] b_type The data type of the matrix B.
/// \param [in, out] b The matrix B, whose columns are the right-hand sides
/// for the systems of equations.
/// \param [in] ldb The leading dimension of the matrix B.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int potrs(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                 std::int64_t nrhs, library_data_t a_type, void *a,
                 std::int64_t lda, library_data_t b_type, void *b,
                 std::int64_t ldb, int *info) {
  return detail::lapack_shim<detail::potrs_impl>(
      q, a_type, info, "potrs_scratchpad_size/potrs", q, uplo, n, nrhs, a_type,
      a, lda, b_type, b, ldb, info);
}

/// Computes the size of workspace memory of syevx/heevx function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] range Must be rangev::all, rangev::values or uplo::indices.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] vl If range == rangev::values, the lower bound of the interval
/// to be searched for eigenvalues
/// \param [in] vu If range == rangev::values, the upper bound of the interval
/// to be searched for eigenvalues
/// \param [in] il If range == rangev::indices, the indices of the smallest
/// eigenvalue to be returned.
/// \param [in] iu If range == rangev::indices, the indices of the largest
/// eigenvalue to be returned.
/// \param [in] w_type The data type of the eigenvalues.
/// \param [out] device_ws_size The device workspace size in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int syheevx_scratchpad_size(sycl::queue &q, oneapi::mkl::job jobz,
                                   oneapi::mkl::rangev range,
                                   oneapi::mkl::uplo uplo, std::int64_t n,
                                   library_data_t a_type, std::int64_t lda,
                                   void *vl, void *vu, std::int64_t il,
                                   std::int64_t iu, library_data_t w_type,
                                   std::size_t *device_ws_size,
                                   std::size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::syheevx_scratchpad_size_impl>(
      q, a_type, nullptr, "syevx_scratchpad_size/heevx_scratchpad_size", q,
      compz_jobz, range, uplo, n, lda, vl, vu, il, iu,
      device_ws_size_tmp);
  *device_ws_size = detail::element_number_to_byte(device_ws_size_tmp, a_type);
  return ret;
}

/// Computes selected eigenvalues and, optionally, eigenvectors of a
/// symmetric/Hermitian matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] range Must be rangev::all, rangev::values or uplo::indices.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. On exit, the lower or upper triangle is
/// overwritten.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] vl If range == rangev::values, the lower bound of the interval
/// to be searched for eigenvalues
/// \param [in] vu If range == rangev::values, the upper bound of the interval
/// to be searched for eigenvalues
/// \param [in] il If range == rangev::indices, the indices of the smallest
/// eigenvalue to be returned.
/// \param [in] iu If range == rangev::indices, the indices of the largest
/// eigenvalue to be returned.
/// \param [out] m The total number of eigenvalues found.
/// \param [in] w_type The data type of the eigenvalues.
/// \param [out] w The eigenvalues of the matrix A in ascending order.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int syheevx(sycl::queue &q, oneapi::mkl::job jobz,
                   oneapi::mkl::rangev range, oneapi::mkl::uplo uplo,
                   std::int64_t n, library_data_t a_type, void *a,
                   std::int64_t lda, void *vl, void *vu, std::int64_t il,
                   std::int64_t iu, std::int64_t *m, library_data_t w_type,
                   void *w, void *device_ws, std::size_t device_ws_size,
                   int *info) {
  std::size_t device_ws_size_in_element_number =
      detail::byte_to_element_number(device_ws_size, a_type);
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  int ret = detail::lapack_shim<detail::syheevx_impl>(
      q, a_type, info, "syevx/heevx", q, compz_jobz, range, uplo, n, a_type, a,
      lda, vl, vu, il, iu, m, w_type, w, device_ws,
      device_ws_size_in_element_number, info);
  q.wait();
  return ret;
}

/// Computes the size of workspace memory of syevx/heevx function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] range Must be rangev::all, rangev::values or uplo::indices.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] vl If range == rangev::values, the lower bound of the interval
/// to be searched for eigenvalues
/// \param [in] vu If range == rangev::values, the upper bound of the interval
/// to be searched for eigenvalues
/// \param [in] il If range == rangev::indices, the indices of the smallest
/// eigenvalue to be returned.
/// \param [in] iu If range == rangev::indices, the indices of the largest
/// eigenvalue to be returned.
/// \param [out] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
template <typename T, typename ValueT>
inline int syheevx_scratchpad_size(sycl::queue &q, oneapi::mkl::job jobz,
                                   oneapi::mkl::rangev range,
                                   oneapi::mkl::uplo uplo, int n, int lda,
                                   ValueT vl, ValueT vu, int il, int iu,
                                   int *device_ws_size) {
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::syheevx_scratchpad_size_impl>(
      q, detail::get_library_data_t_from_type<T>(), nullptr,
      "syevx_scratchpad_size/heevx_scratchpad_size", q, compz_jobz, range, uplo,
      n, lda, &vl, &vu, il, iu, device_ws_size_tmp);
  *device_ws_size = (int)device_ws_size_tmp;
  return ret;
}

/// Computes selected eigenvalues and, optionally, eigenvectors of a
/// symmetric/Hermitian matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] range Must be rangev::all, rangev::values or uplo::indices.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in, out] a The input matrix A. On exit, the lower or upper triangle is
/// overwritten.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] vl If range == rangev::values, the lower bound of the interval
/// to be searched for eigenvalues
/// \param [in] vu If range == rangev::values, the upper bound of the interval
/// to be searched for eigenvalues
/// \param [in] il If range == rangev::indices, the indices of the smallest
/// eigenvalue to be returned.
/// \param [in] iu If range == rangev::indices, the indices of the largest
/// eigenvalue to be returned.
/// \param [out] m The total number of eigenvalues found.
/// \param [out] w The eigenvalues of the matrix A in ascending order.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T, typename ValueT>
inline int syheevx(sycl::queue &q, oneapi::mkl::job jobz,
                   oneapi::mkl::rangev range, oneapi::mkl::uplo uplo, int n,
                   T *a, int lda, ValueT vl, ValueT vu, int il, int iu, int *m,
                   ValueT *w, T *device_ws, int device_ws_size, int *info) {
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  std::int64_t m64;
  int ret = detail::lapack_shim<detail::syheevx_impl>(
      q, detail::get_library_data_t_from_type<T>(), info, "syevx/heevx", q,
      compz_jobz, range, uplo, n, detail::get_library_data_t_from_type<T>(), a,
      lda, &vl, &vu, il, iu, &m64,
      detail::get_library_data_t_from_type<ValueT>(), w, device_ws,
      device_ws_size, info);
  q.wait();
  *m = (int)m64;
  return ret;
}

/// Computes the size of workspace memory of sygvx/hegvx function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] itype Must be 1, 2 or 3.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] range Must be rangev::all, rangev::values or uplo::indices.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] ldb The leading dimension of the matrix B.
/// \param [in] vl If range == rangev::values, the lower bound of the interval
/// to be searched for eigenvalues
/// \param [in] vu If range == rangev::values, the upper bound of the interval
/// to be searched for eigenvalues
/// \param [in] il If range == rangev::indices, the indices of the smallest
/// eigenvalue to be returned.
/// \param [in] iu If range == rangev::indices, the indices of the largest
/// eigenvalue to be returned.
/// \param [in] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
template <typename T, typename ValueT>
inline int
syhegvx_scratchpad_size(sycl::queue &q, int itype, oneapi::mkl::job jobz,
                        oneapi::mkl::rangev range, oneapi::mkl::uplo uplo,
                        int n, int lda, int ldb, ValueT vl, ValueT vu, int il,
                        int iu, int *device_ws_size) {
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::syhegvx_scratchpad_size_impl>(
      q, detail::get_library_data_t_from_type<T>(), nullptr,
      "sygvx_scratchpad_size/hegvx_scratchpad_size", q, itype, compz_jobz,
      range, uplo, n, lda, ldb, &vl, &vu, il, iu, device_ws_size_tmp);
  *device_ws_size = (int)device_ws_size_tmp;
  return ret;
}

/// Computes selected eigenvalues and, optionally, eigenvectors of a real
/// generalized symmetric/Hermitian definite eigenproblem.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] itype Must be 1, 2 or 3.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] range Must be rangev::all, rangev::values or uplo::indices.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in, out] a The input matrix A. On exit, the lower or upper triangle is
/// overwritten.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in, out] b The input matrix B.
/// \param [in] ldb The leading dimension of the matrix B.
/// \param [in] vl If range == rangev::values, the lower bound of the interval
/// to be searched for eigenvalues
/// \param [in] vu If range == rangev::values, the upper bound of the interval
/// to be searched for eigenvalues
/// \param [in] il If range == rangev::indices, the indices of the smallest
/// eigenvalue to be returned.
/// \param [in] iu If range == rangev::indices, the indices of the largest
/// eigenvalue to be returned.
/// \param [out] m The total number of eigenvalues found.
/// \param [out] w The eigenvalues of the matrix A in ascending order.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T, typename ValueT>
inline int syhegvx(sycl::queue &q, int itype, oneapi::mkl::job jobz,
                   oneapi::mkl::rangev range, oneapi::mkl::uplo uplo, int n,
                   T *a, int lda, T *b, int ldb, ValueT vl, ValueT vu, int il,
                   int iu, int *m, ValueT *w, T *device_ws, int device_ws_size,
                   int *info) {
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  std::int64_t m64;
  int ret = detail::lapack_shim<detail::syhegvx_impl>(
      q, detail::get_library_data_t_from_type<T>(), info, "sygvx/hegvx", q,
      itype, compz_jobz, range, uplo, n, a, lda, b, ldb, &vl, &vu, il, iu, &m64,
      w, device_ws, device_ws_size, info);
  q.wait();
  *m = (int)m64;
  return ret;
}

/// Computes the size of workspace memory of sygvd/hegvd function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] itype Must be 1, 2 or 3.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] ldb The leading dimension of the matrix B.
/// \param [in] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
template <typename T>
inline int syhegvd_scratchpad_size(sycl::queue &q, int itype,
                                   oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, int n, int lda,
                                   int ldb, int *device_ws_size) {
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::syhegvd_scratchpad_size_impl>(
      q, detail::get_library_data_t_from_type<T>(), nullptr,
      "sygvd_scratchpad_size/hegvd_scratchpad_size", q, itype, jobz, uplo, n,
      lda, ldb, device_ws_size_tmp);
  *device_ws_size = (int)device_ws_size_tmp;
  return ret;
}

/// Computes all eigenvalues and, optionally, eigenvectors of a real generalized
/// symmetric/Hermitian definite eigenproblem using a divide and conquer method.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] itype Must be 1, 2 or 3.
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in, out] a The input matrix A. On exit, it is overwritten by eigenvectors.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in, out] b The input matrix B.
/// \param [in] ldb The leading dimension of the matrix B.
/// \param [out] w The eigenvalues of the matrix A in ascending order.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T, typename ValueT>
inline int syhegvd(sycl::queue &q, int itype, oneapi::mkl::job jobz,
                   oneapi::mkl::uplo uplo, int n, T *a, int lda, T *b, int ldb,
                   ValueT *w, T *device_ws, int device_ws_size, int *info) {
  return detail::lapack_shim<detail::syhegvd_impl>(
      q, detail::get_library_data_t_from_type<T>(), info, "sygvd/hegvd", q,
      itype, jobz, uplo, n, a, lda, b, ldb, w, device_ws, device_ws_size, info);
}

/// Computes the size of workspace memory of syev/heev function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
template <typename T>
inline int syheev_scratchpad_size(sycl::queue &q, oneapi::mkl::job jobz,
                                  oneapi::mkl::uplo uplo, int n, int lda,
                                  int *device_ws_size) {
  std::size_t device_ws_size_tmp;
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  int ret = detail::lapack_shim<detail::syheev_scratchpad_size_impl>(
      q, detail::get_library_data_t_from_type<T>(), nullptr,
      "syev_scratchpad_size/heev_scratchpad_size", q, compz_jobz, uplo, n, lda,
      device_ws_size_tmp);
  *device_ws_size = (int)device_ws_size_tmp;
  return ret;
}

/// Computes all eigenvalues and, optionally, eigenvectors of a real symmetric
/// or Hermitian matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in, out] a The input matrix A. On exit, it is overwritten by
/// eigenvectors.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] w The eigenvalues of the matrix A in ascending order.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T, typename ValueT>
inline int syheev(sycl::queue &q, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                  int n, T *a, int lda, ValueT *w, T *device_ws,
                  int device_ws_size, int *info) {
  oneapi::mkl::compz compz_jobz = detail::job2compz(jobz);
  return detail::lapack_shim<detail::syheev_impl>(
      q, detail::get_library_data_t_from_type<T>(), info, "syev/heev", q,
      compz_jobz, uplo, n, a, lda, w, device_ws, device_ws_size, info);
}

/// Computes the size of workspace memory of syevd/heevd function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] w_type The data type of the eigenvalues.
/// \param [out] device_ws_size The device workspace in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int syheevd_scratchpad_size(sycl::queue &q, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n,
                                   library_data_t a_type, std::int64_t lda,
                                   library_data_t w_type,
                                   std::size_t *device_ws_size,
                                   std::size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::syheevd_scratchpad_size_impl>(
      q, a_type, nullptr, "syevd_scratchpad_size/heevd_scratchpad_size", q,
      jobz, uplo, n, a_type, lda, device_ws_size_tmp);
  *device_ws_size = detail::element_number_to_byte(device_ws_size_tmp, a_type);
  return ret;
}

/// Computes all eigenvalues and, optionally, all eigenvectors of a real
/// symmetric or Hermitian matrix using divide and conquer algorithm.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. On exit, it is overwritten by
/// eigenvectors.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] w_type The data type of the eigenvalues.
/// \param [out] w The eigenvalues of the matrix A in ascending order.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int syheevd(sycl::queue &q, oneapi::mkl::job jobz,
                   oneapi::mkl::uplo uplo, std::int64_t n,
                   library_data_t a_type, void *a, std::int64_t lda,
                   library_data_t w_type, void *w, void *device_ws,
                   std::size_t device_ws_size, int *info) {
  std::size_t device_ws_size_in_element_number =
      detail::byte_to_element_number(device_ws_size, a_type);
  return detail::lapack_shim<detail::syheevd_impl>(
      q, a_type, info, "syevd/heevd", q, jobz, uplo, n, a_type, a, lda, w,
      device_ws, device_ws_size_in_element_number, info);
}

/// Computes the size of workspace memory of syevd/heevd function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] w_type The data type of the eigenvalues.
/// \param [out] device_ws_size The device workspace size as a number of
/// elements of type \tparam T.
template <typename T>
inline int syheevd_scratchpad_size(sycl::queue &q, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t lda, int *device_ws_size) {
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::syheevd_scratchpad_size_impl>(
      q, detail::get_library_data_t_from_type<T>(), nullptr,
      "syevd_scratchpad_size/heevd_scratchpad_size", q, jobz, uplo, n,
      detail::get_library_data_t_from_type<T>(), lda, device_ws_size_tmp);
  *device_ws_size = (int)device_ws_size_tmp;
  return ret;
}

/// Computes all eigenvalues and, optionally, all eigenvectors of a real
/// symmetric or Hermitian matrix using divide and conquer algorithm.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] jobz Must be job::novec or job::vec.
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. On exit, it is overwritten by
/// eigenvectors.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] w_type The data type of the eigenvalues.
/// \param [out] w The eigenvalues of the matrix A in ascending order.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size as a number of
/// elements of type \tparam T.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
template <typename T, typename ValueT>
inline int syheevd(sycl::queue &q, oneapi::mkl::job jobz,
                   oneapi::mkl::uplo uplo, std::int64_t n, T *a,
                   std::int64_t lda, ValueT *w, T *device_ws,
                   int device_ws_size, int *info) {
  return detail::lapack_shim<detail::syheevd_impl>(
      q, detail::get_library_data_t_from_type<T>(), info, "syevd/heevd", q,
      jobz, uplo, n, detail::get_library_data_t_from_type<T>(), a, lda, w,
      device_ws, device_ws_size, info);
}

/// Computes the size of workspace memory of trtri function.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] diag Must be diag::nonunit or diag::unit.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [out] device_ws_size The device workspace in bytes.
/// \param [out] host_ws_size The host workspace size in bytes. Currently the
/// value is always zero.
inline int trtri_scratchpad_size(sycl::queue &q, oneapi::mkl::uplo uplo,
                                 oneapi::mkl::diag diag, std::int64_t n,
                                 library_data_t a_type, std::int64_t lda,
                                 std::size_t *device_ws_size,
                                 std::size_t *host_ws_size = nullptr) {
  if (host_ws_size)
    *host_ws_size = 0;
  std::size_t device_ws_size_tmp;
  int ret = detail::lapack_shim<detail::trtri_scratchpad_size_impl>(
      q, a_type, nullptr, "trtri_scratchpad_size", q, uplo, diag, n, a_type,
      lda, device_ws_size_tmp);
  *device_ws_size = detail::element_number_to_byte(device_ws_size_tmp, a_type);
  return ret;
}

/// Computes the inverse of a triangular matrix.
/// \return Returns 0 if no synchronous exception, otherwise returns 1.
/// \param [in] q Device queue where computation will be performed. It must
/// have the in_order property when using the USM mode (DPCT_USM_LEVEL_NONE is
/// not defined).
/// \param [in] uplo Must be uplo::upper or uplo::lower.
/// \param [in] diag Must be diag::nonunit or diag::unit.
/// \param [in] n The order of the matrix A.
/// \param [in] a_type The data type of the matrix A.
/// \param [in, out] a The input matrix A. On exit, it is overwritten by
/// the inverse matrix of A.
/// \param [in] lda The leading dimension of the matrix A.
/// \param [in] device_ws The workspace.
/// \param [in] device_ws_size The workspace size in bytes.
/// \param [out] info If lapack synchronous exception is caught, the value
/// returned from info() method of the exception is set to \p info.
inline int trtri(sycl::queue &q, oneapi::mkl::uplo uplo, oneapi::mkl::diag diag,
                 std::int64_t n, library_data_t a_type, void *a,
                 std::int64_t lda, void *device_ws, std::size_t device_ws_size,
                 int *info) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  std::size_t device_ws_size_in_element_number =
      detail::byte_to_element_number(device_ws_size, a_type);
  return detail::lapack_shim<detail::trtri_impl>(
      q, a_type, info, "trtri", q, uplo, diag, n, a_type, a, lda, device_ws,
      device_ws_size_in_element_number, info);
#endif
}
} // namespace lapack
} // namespace dpct

#endif // __DPCT_LAPACK_UTILS_HPP__
