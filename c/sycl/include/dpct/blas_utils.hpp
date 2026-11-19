//==---- blas_utils.hpp----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BLAS_UTILS_HPP__
#define __DPCT_BLAS_UTILS_HPP__

#include "compat_service.hpp"
#include "lib_common_utils.hpp"

#include "detail/blas_utils_detail.hpp"

namespace dpct {
namespace blas {

/// Parameter input/output properties are:
/// in: input parameter
/// out: output parameter
/// in_out: input and output parameter
enum class parameter_inout_prop { in, out, in_out };

/// parameter_wrapper_t is a class to wrap the parameter to fit the oneMKL
/// interface.
/// E.g.,
/// \code
/// void foo(sycl::queue q, int *res) {
///   // Declare the wrapper.
///   parameter_wrapper_t<std::int64_t, int, parameter_inout_prop::out>
///       res_wrapper(q, res);
///   // parameter_wrapper_t::get_ptr() is passed as a parameter. The result is
///   // saved into the wrapper.
///   oneapi::mkl::...(q, ..., res_wrapper.get_ptr());
///   // The action to copy the result in the wrapper to the original variable
///   // is submitted when destructing the wrapper automatically.
/// }
/// \endcode
/// \tparam target_t Data type to fit oneMKL parameter data type. E.g.,
/// get_ptr() returns \tparam target_t data type and can be used as oneMKL
/// parameter with USM memory configuration.
/// \tparam source_t Data type of the original parameter.
/// \tparam inout_prop The input/output property of the parameter.
template <typename target_t, typename source_t, parameter_inout_prop inout_prop>
class parameter_wrapper_t
    : public detail::parameter_wrapper_base_t<target_t, source_t> {
  static_assert(inout_prop == parameter_inout_prop::out &&
                "Only parameter_inout_prop::out is supported if "
                "target_t and source_t are not same.");
  using base_t = detail::parameter_wrapper_base_t<target_t, source_t>;
  using base_t::_q;
  using base_t::_source;
  using base_t::_source_attribute;
  using base_t::_target;

public:
  /// Constructor. Malloc the wrapper memory.
  /// \param q The queue used for internal malloc and memcpy
  /// \param source The original parameter
  parameter_wrapper_t(sycl::queue q, source_t *source) : base_t(q, source, 1) {}
  /// Destructor. Copy back content from wrapper memory to original memory
  ~parameter_wrapper_t() {
#ifdef DPCT_USM_LEVEL_NONE
    if (_source_attribute ==
        ::dpct::cs::detail::pointer_access_attribute::device_only) {
      _q.submit([&](sycl::handler &cgh) {
        auto from_acc = dpct::get_buffer<target_t>(_target)
                            .template get_access<sycl::access_mode::read>(cgh);
        auto to_acc = dpct::get_buffer<source_t>(_source)
                          .template get_access<sycl::access_mode::write>(cgh);
        cgh.single_task<::dpct::cs::kernel_name<
            class parameter_wrapper_copyback, target_t, source_t>>(
            [=]() { to_acc[0] = static_cast<source_t>(from_acc[0]); });
      });
    } else {
      source_t temp =
          static_cast<source_t>(dpct::get_host_ptr<target_t>(_target)[0]);
      *_source = temp;
    }
#else
    if (_source_attribute ==
        ::dpct::cs::detail::pointer_access_attribute::device_only) {
      _q.template single_task<::dpct::cs::kernel_name<
          class parameter_wrapper_copyback, target_t, source_t>>(
          [t = _target, s = _source]() { *s = static_cast<source_t>(*t); });
    } else {
      target_t temp;
      _q.memcpy(&temp, _target, sizeof(target_t)).wait();
      *_source = static_cast<source_t>(temp);
    }
#endif
  }
  /// Get the target memory of the wrapper to represent the wrapped variable
  target_t *get_ptr() { return _target; }
};

/// parameter_wrapper_t is a class to wrap the parameter to fit the oneMKL
/// interface.
/// E.g.,
/// \code
/// void foo(sycl::queue q, int *params, int ele_num) {
///   // Declare the wrapper.
///   parameter_wrapper_t<float, parameter_inout_prop::in_out> params_wrapper(
///       q, params, ele_num);
///   // parameter_wrapper_t::get_ptr() is passed as a parameter. The result is
///   // saved into the wrapper.
///   oneapi::mkl::...(q, ..., params_wrapper.get_ptr());
///   // The action to copy the result in the wrapper to the original variable
///   // is submitted when destructing the wrapper automatically.
/// }
/// \endcode
/// \tparam target_t Data type to fit oneMKL parameter data type. E.g.,
/// get_ptr() returns \tparam target_t data type and can be used as oneMKL
/// parameter with USM memory configuration.
/// \tparam inout_prop The input/output property of the parameter.
template <typename target_t, parameter_inout_prop inout_prop>
class parameter_wrapper_t<target_t, target_t, inout_prop>
    : public detail::parameter_wrapper_base_t<target_t, target_t> {
  using base_t = detail::parameter_wrapper_base_t<target_t, target_t>;
  using base_t::_ele_num;
  using base_t::_need_free;
  using base_t::_q;
  using base_t::_source;
  using base_t::_source_attribute;
  using base_t::_target;

public:
  /// Constructor. Malloc the wrapper memory.
  /// \param q The queue used for internal malloc and memcpy
  /// \param source The original parameter
  /// \param ele_num Element number in \p source
  parameter_wrapper_t(sycl::queue q, target_t *source, size_t ele_num = 1)
      : base_t(q, source, ele_num) {
    if constexpr (inout_prop != parameter_inout_prop::out) {
      if (_need_free) {
        ::dpct::cs::memcpy(_q, _target, _source, sizeof(target_t) * _ele_num,
                           ::dpct::cs::memcpy_direction::automatic);
      }
    }
  }

  /// Constructor. Only vaild for parameter_inout_prop::in.
  /// \param q The queue used for internal malloc and memcpy
  /// \param source The original parameter
  /// \param ele_num Element number in \p source
  template <parameter_inout_prop prop = inout_prop>
  parameter_wrapper_t(
      sycl::queue q, const target_t *source, size_t ele_num = 1,
      typename std::enable_if<prop == parameter_inout_prop::in>::type * = 0)
      : parameter_wrapper_t(q, const_cast<target_t *>(source), ele_num) {}

  /// Destructor. Copy back content from wrapper memory to original memory
  ~parameter_wrapper_t() {
    if constexpr (inout_prop != parameter_inout_prop::in) {
      if (_need_free) {
        sycl::event e = ::dpct::cs::memcpy(
            _q, _source, _target, sizeof(target_t) * _ele_num,
            ::dpct::cs::memcpy_direction::automatic);
        (void)e;
        if (_source_attribute ==
            ::dpct::cs::detail::pointer_access_attribute::host_only)
          e.wait();
      }
    }
  }
  /// Get the target memory of the wrapper to represent the wrapped variable
  target_t *get_ptr() { return _target; }
};

using wrapper_int_to_int64_out =
    parameter_wrapper_t<std::int64_t, int, parameter_inout_prop::out>;
using wrapper_int64_out =
    parameter_wrapper_t<std::int64_t, std::int64_t, parameter_inout_prop::out>;
using wrapper_float_out =
    parameter_wrapper_t<float, float, parameter_inout_prop::out>;
using wrapper_double_out =
    parameter_wrapper_t<double, double, parameter_inout_prop::out>;
using wrapper_float2_out =
    parameter_wrapper_t<sycl::float2, sycl::float2, parameter_inout_prop::out>;
using wrapper_double2_out = parameter_wrapper_t<sycl::double2, sycl::double2,
                                                parameter_inout_prop::out>;
using wrapper_float_inout =
    parameter_wrapper_t<float, float, parameter_inout_prop::in_out>;
using wrapper_double_inout =
    parameter_wrapper_t<double, double, parameter_inout_prop::in_out>;
using wrapper_float2_inout = parameter_wrapper_t<sycl::float2, sycl::float2,
                                                 parameter_inout_prop::in_out>;
using wrapper_double2_inout = parameter_wrapper_t<sycl::double2, sycl::double2,
                                                  parameter_inout_prop::in_out>;
using wrapper_float_in =
    parameter_wrapper_t<float, float, parameter_inout_prop::in>;
using wrapper_double_in =
    parameter_wrapper_t<double, double, parameter_inout_prop::in>;

/// Copy matrix data synchronously. The default leading dimension is column.
/// \param [out] to_ptr A pointer points to the destination location.
/// \param [in] from_ptr A pointer points to the source location.
/// \param [in] to_ld The leading dimension the destination matrix.
/// \param [in] from_ld The leading dimension the source matrix.
/// \param [in] rows The number of rows of the source matrix.
/// \param [in] cols The number of columns of the source matrix.
/// \param [in] elem_size The element size in bytes.
/// \param [in] direction The direction of the data copy.
/// \param [in] queue The queue where the routine should be executed.
/// \param [in] deps A list of events to wait for before starting copy.
inline void
matrix_mem_copy_sync(void *to_ptr, const void *from_ptr, std::int64_t to_ld,
                     std::int64_t from_ld, std::int64_t rows, std::int64_t cols,
                     std::int64_t elem_size,
                     ::dpct::cs::memcpy_direction direction =
                         ::dpct::cs::memcpy_direction::automatic,
                     sycl::queue &queue = ::dpct::cs::get_default_queue(),
                     const std::vector<sycl::event> &deps = {}) {
  if (to_ptr == from_ptr && to_ld == from_ld)
    return;
  if (to_ld == from_ld) {
    size_t copy_size = elem_size * ((cols - 1) * (size_t)to_ld + rows);
    ::dpct::cs::memcpy(queue, to_ptr, from_ptr, copy_size, direction, deps)
        .wait();
  } else {
    sycl::event::wait(::dpct::cs::memcpy(
        queue, to_ptr, from_ptr, elem_size * to_ld, elem_size * from_ld,
        elem_size * rows, cols, direction, deps));
  }
}
/// Copy matrix data asynchronously. The default leading dimension is column.
/// \return Output event to wait on to ensure copy is complete.
/// \param [out] to_ptr A pointer points to the destination location.
/// \param [in] from_ptr A pointer points to the source location.
/// \param [in] to_ld The leading dimension the destination matrix.
/// \param [in] from_ld The leading dimension the source matrix.
/// \param [in] rows The number of rows of the source matrix.
/// \param [in] cols The number of columns of the source matrix.
/// \param [in] elem_size The element size in bytes.
/// \param [in] direction The direction of the data copy.
/// \param [in] queue The queue where the routine should be executed.
/// \param [in] deps A list of events to wait for before starting copy.
inline sycl::event
matrix_mem_copy_async(void *to_ptr, const void *from_ptr, std::int64_t to_ld,
                      std::int64_t from_ld, std::int64_t rows,
                      std::int64_t cols, std::int64_t elem_size,
                      ::dpct::cs::memcpy_direction direction =
                          ::dpct::cs::memcpy_direction::automatic,
                      sycl::queue &queue = ::dpct::cs::get_default_queue(),
                      const std::vector<sycl::event> &deps = {}) {
  if (to_ptr == from_ptr && to_ld == from_ld)
    return sycl::event();
  if (to_ld == from_ld) {
    size_t copy_size = elem_size * ((cols - 1) * (size_t)to_ld + rows);
    return ::dpct::cs::memcpy(queue, to_ptr, from_ptr, copy_size, direction,
                              deps);
  }
  auto events = ::dpct::cs::memcpy(queue, to_ptr, from_ptr, elem_size * to_ld,
                                   elem_size * from_ld, elem_size * rows, cols,
                                   direction, deps);
  return queue.single_task(events, [] {});
}

/// Copy matrix data. The default leading dimension is column.
/// \param [out] to_ptr A pointer points to the destination location.
/// \param [in] from_ptr A pointer points to the source location.
/// \param [in] to_ld The leading dimension the destination matrix.
/// \param [in] from_ld The leading dimension the source matrix.
/// \param [in] rows The number of rows of the source matrix.
/// \param [in] cols The number of columns of the source matrix.
/// \param [in] elem_size The element size in bytes.
/// \param [in] direction The direction of the data copy.
/// \param [in] queue The queue where the routine should be executed.
/// \param [in] async If this argument is true, the return of the function
/// does NOT guarantee the copy is completed.
inline void
matrix_mem_copy(void *to_ptr, const void *from_ptr, std::int64_t to_ld,
                std::int64_t from_ld, std::int64_t rows, std::int64_t cols,
                std::int64_t elem_size,
                ::dpct::cs::memcpy_direction direction =
                    ::dpct::cs::memcpy_direction::automatic,
                sycl::queue &queue = ::dpct::cs::get_default_queue(),
                bool async = false) {
  if (async)
    matrix_mem_copy_async(to_ptr, from_ptr, to_ld, from_ld, rows, cols,
                          elem_size, direction, queue);
  else
    matrix_mem_copy_sync(to_ptr, from_ptr, to_ld, from_ld, rows, cols,
                         elem_size, direction, queue);
}

enum class math_mode : int {
  mm_default,
  mm_tf32,
};

using ::dpct::compute_type;

class descriptor {
public:
  void set_queue(::dpct::cs::queue_ptr q_ptr) noexcept { _queue_ptr = q_ptr; }
  sycl::queue &get_queue() noexcept { return *_queue_ptr; }
  void set_math_mode(math_mode mm) noexcept { _mm = mm; }
  math_mode get_math_mode() const noexcept { return _mm; }
  static inline void set_saved_queue(::dpct::cs::queue_ptr q_ptr) noexcept {
    _saved_queue_ptr = q_ptr;
  }
  static inline sycl::queue &get_saved_queue() noexcept {
    return *_saved_queue_ptr;
  }

private:
  ::dpct::cs::queue_ptr _queue_ptr = &::dpct::cs::get_default_queue();
  math_mode _mm = math_mode::mm_default;
  static inline ::dpct::cs::queue_ptr _saved_queue_ptr =
      &::dpct::cs::get_default_queue();
};

using descriptor_ptr = descriptor *;
} // namespace blas

/// Get the value of \p s.
/// Copy the data to host synchronously, then return the data.
/// \param [in] p The pointer points the data.
/// \param [in] q The queue where the memory copy should be executed.
template <typename T> inline T get_value(const T *s, sycl::queue &q) {
  return detail::get_value(s, q);
}
template <typename T>
inline std::complex<T> get_value(const sycl::vec<T, 2> *s, sycl::queue &q) {
  return detail::get_value(s, q);
}

inline oneapi::mkl::transpose get_transpose(int t) {
  if (t == 0) {
    return oneapi::mkl::transpose::nontrans;
  } else if (t == 1) {
    return oneapi::mkl::transpose::trans;
  } else {
    return oneapi::mkl::transpose::conjtrans;
  }
}

/// Computes the LU factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in, out] a Array of pointers to matrices. These matrices will be
/// overwritten by lower triangulars with unit diagonal elements and upper
/// triangulars.
/// \param [in] lda The leading dimension of the matrices.
/// \param [out] ipiv An array stores the pivot indices. If \p ipiv is nullptr,
/// non-pivoting LU factorization is computed.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrf_batch_wrapper(sycl::queue &exec_queue, int n, T *a[], int lda,
                                int *ipiv, int *info, int batch_size) {
  if (ipiv == nullptr) {
    detail::getrfnp_batch_wrapper(exec_queue, n, a, lda, info, batch_size);
    return;
  }
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  // Set the info array value to 0
  ::dpct::cs::fill<unsigned char>(exec_queue, info, 0,
                                  sizeof(int) * batch_size);
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_a = n * lda;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getrf_batch_scratchpad_size<Ty>(
          exec_queue, n, n, lda, stride_a, stride_ipiv, batch_size);

  T *a_buffer_ptr;
  a_buffer_ptr = (T *)::dpct::cs::malloc(stride_a * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  for (std::int64_t i = 0; i < batch_size; ++i)
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));

  {
    sycl::buffer<std::int64_t, 1> ipiv_buf(
        sycl::range<1>(batch_size * stride_ipiv));
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    oneapi::mkl::lapack::getrf_batch(exec_queue, n, n, a_buffer, lda, stride_a,
                                     ipiv_buf, stride_ipiv, batch_size,
                                     scratchpad, scratchpad_size);

    auto to_buffer = get_buffer<int>(ipiv);
    exec_queue.submit([&](sycl::handler &cgh) {
      auto from_acc = ipiv_buf.get_access<sycl::access_mode::read>(cgh);
      auto to_acc = to_buffer.get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<
          ::dpct::cs::kernel_name<class getrf_device_int64_to_int, T>>(
          sycl::range<2>(batch_size, n), [=](sycl::id<2> id) {
            to_acc[id.get(0) * n + id.get(1)] =
                static_cast<int>(from_acc[id.get(0) * stride_ipiv + id.get(1)]);
          });
    });
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(::dpct::cs::memcpy(
        exec_queue, host_a[i], a_buffer_ptr + i * stride_a, n * lda * sizeof(T),
        ::dpct::cs::memcpy_direction::automatic));

  std::vector<void *> ptrs{host_a};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptrs, events);
  mem_free_thread.detach();
#else
  std::int64_t *m_int64 = new std::int64_t;
  std::int64_t *n_int64 = new std::int64_t;
  std::int64_t *lda_int64 = new std::int64_t;
  std::int64_t *group_sizes = new std::int64_t;
  *m_int64 = n;
  *n_int64 = n;
  *lda_int64 = lda;
  *group_sizes = batch_size;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getrf_batch_scratchpad_size<Ty>(
          exec_queue, m_int64, n_int64, lda_int64, 1, group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);
  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *)).wait();
  for (std::int64_t i = 0; i < batch_size; ++i)
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;

  oneapi::mkl::lapack::getrf_batch(exec_queue, m_int64, n_int64,
                                   (Ty **)a_shared, lda_int64, ipiv_int64_ptr,
                                   1, group_sizes, scratchpad, scratchpad_size);

  sycl::event e = exec_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<
        ::dpct::cs::kernel_name<class getrf_device_int64_to_int, T>>(
        sycl::range<1>(batch_size * n), [=](sycl::id<1> idx) {
          ipiv[idx] = static_cast<int>(ipiv_int64[idx]);
        });
  });

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] {
      delete m_int64;
      delete n_int64;
      delete lda_int64;
      delete group_sizes;
    });
  });
  std::vector<void *> ptrs{scratchpad, ipiv_int64, ipiv_int64_ptr, a_shared};
  ::dpct::cs::enqueue_free(ptrs, {e}, exec_queue);
#endif
}

/// Solves a system of linear equations with a batch of LU-factored square
/// coefficient matrices, with multiple right-hand sides.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] trans Indicates the form of the linear equations.
/// \param [in] n The order of the matrices.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [in] ipiv An array stores the pivots.
/// \param [in, out] b Array of pointers to matrices, whose columns are
/// the right-hand sides for the systems of equations.
/// \param [in] ldb The leading dimension of the matrices in \p b.
/// \param [out] info A value stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrs_batch_wrapper(sycl::queue &exec_queue,
                                oneapi::mkl::transpose trans, int n, int nrhs,
                                const T *a[], int lda, const int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  // Set the info value to 0
  *info = 0;
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_a = n * lda;
  std::int64_t stride_b = nrhs * ldb;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getrs_batch_scratchpad_size<Ty>(
          exec_queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b,
          batch_size);

  T *a_buffer_ptr, *b_buffer_ptr;
  a_buffer_ptr = (T *)::dpct::cs::malloc(stride_a * batch_size * sizeof(T));
  b_buffer_ptr = (T *)::dpct::cs::malloc(stride_b * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  T **host_b = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));
  for (std::int64_t i = 0; i < batch_size; ++i) {
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));
    dpct_memcpy(b_buffer_ptr + i * stride_b, host_b[i], nrhs * ldb * sizeof(T));
  }

  {
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    sycl::buffer<std::int64_t, 1> ipiv_buf(
        sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<sycl::access_mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<
          ::dpct::cs::kernel_name<class getrs_device_int64_to_int, T>>(
          sycl::range<2>(batch_size, n), [=](sycl::id<2> id) {
            to_acc[id.get(0) * stride_ipiv + id.get(1)] =
                static_cast<std::int64_t>(from_acc[id.get(0) * n + id.get(1)]);
          });
    });

    oneapi::mkl::lapack::getrs_batch(exec_queue, trans, n, nrhs, a_buffer, lda,
                                     stride_a, ipiv_buf, stride_ipiv, b_buffer,
                                     ldb, stride_b, batch_size, scratchpad,
                                     scratchpad_size);
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(::dpct::cs::memcpy(
        exec_queue, host_b[i], b_buffer_ptr + i * stride_b,
        nrhs * ldb * sizeof(T), ::dpct::cs::memcpy_direction::automatic));
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptrs, events);
  mem_free_thread.detach();
#else
  std::int64_t *n_int64 = new std::int64_t;
  std::int64_t *nrhs_int64 = new std::int64_t;
  std::int64_t *lda_int64 = new std::int64_t;
  std::int64_t *ldb_int64 = new std::int64_t;
  std::int64_t *group_sizes = new std::int64_t;
  oneapi::mkl::transpose *trans_array = new oneapi::mkl::transpose;
  *n_int64 = n;
  *nrhs_int64 = nrhs;
  *lda_int64 = lda;
  *ldb_int64 = ldb;
  *group_sizes = batch_size;
  *trans_array = trans;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getrs_batch_scratchpad_size<Ty>(
          exec_queue, trans_array, n_int64, nrhs_int64, lda_int64, ldb_int64, 1,
          group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);
  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **b_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(b_shared, b, batch_size * sizeof(T *));

  exec_queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<
            ::dpct::cs::kernel_name<class getrs_device_int64_to_int, T>>(
            sycl::range<1>(batch_size * n), [=](sycl::id<1> idx) {
              ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
            });
      })
      .wait();

  for (std::int64_t i = 0; i < batch_size; ++i)
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;

  sycl::event e = oneapi::mkl::lapack::getrs_batch(
      exec_queue, trans_array, n_int64, nrhs_int64, (Ty **)a_shared, lda_int64,
      ipiv_int64_ptr, (Ty **)b_shared, ldb_int64, 1, group_sizes, scratchpad,
      scratchpad_size);

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] {
      delete n_int64;
      delete nrhs_int64;
      delete lda_int64;
      delete ldb_int64;
      delete group_sizes;
      delete trans_array;
    });
  });
  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64, a_shared,
                           b_shared};
  ::dpct::cs::enqueue_free(ptrs, {e}, exec_queue);
#endif
}

/// Computes the inverses of a batch of LU-factored matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [in] ipiv An array stores the pivots.
/// \param [out] b Array of pointers to inverse matrices.
/// \param [in] ldb The leading dimension of the matrices in \p b.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getri_batch_wrapper(sycl::queue &exec_queue, int n, const T *a[],
                                int lda, int *ipiv, T *b[], int ldb, int *info,
                                int batch_size) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  // Set the info array value to 0
  ::dpct::cs::fill<unsigned char>(exec_queue, info, 0,
                                  sizeof(int) * batch_size);
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_b = n * ldb;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getri_batch_scratchpad_size<Ty>(
          exec_queue, n, ldb, stride_b, stride_ipiv, batch_size);

  T *b_buffer_ptr;
  b_buffer_ptr = (T *)::dpct::cs::malloc(stride_b * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  T **host_b = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));

  for (std::int64_t i = 0; i < batch_size; ++i) {
    // Need to create a copy of input matrices "a" to keep them unchanged.
    // Matrices "b" (copy of matrices "a") will be used as input and output
    // parameter in oneapi::mkl::lapack::getri_batch call.
    blas::matrix_mem_copy(b_buffer_ptr + i * stride_b, host_a[i], ldb, lda, n,
                          n, sizeof(T), dpct::device_to_device, exec_queue);
  }

  {
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    sycl::buffer<std::int64_t, 1> ipiv_buf(
        sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<sycl::access_mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<
          ::dpct::cs::kernel_name<class getri_device_int64_to_int, T>>(
          sycl::range<2>(batch_size, n), [=](sycl::id<2> id) {
            to_acc[id.get(0) * stride_ipiv + id.get(1)] =
                static_cast<std::int64_t>(from_acc[id.get(0) * n + id.get(1)]);
          });
    });

    oneapi::mkl::lapack::getri_batch(exec_queue, n, b_buffer, ldb, stride_b,
                                     ipiv_buf, stride_ipiv, batch_size,
                                     scratchpad, scratchpad_size);
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(::dpct::cs::memcpy(
        exec_queue, host_b[i], b_buffer_ptr + i * stride_b, n * ldb * sizeof(T),
        ::dpct::cs::memcpy_direction::automatic));
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptrs, events);
  mem_free_thread.detach();
#else
  std::int64_t *n_int64 = new std::int64_t;
  std::int64_t *ldb_int64 = new std::int64_t;
  std::int64_t *group_sizes = new std::int64_t;
  *n_int64 = n;
  *ldb_int64 = ldb;
  *group_sizes = batch_size;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getri_batch_scratchpad_size<Ty>(
          exec_queue, n_int64, ldb_int64, 1, group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<
        ::dpct::cs::kernel_name<class getri_device_int64_to_int, T>>(
        sycl::range<1>(batch_size * n), [=](sycl::id<1> idx) {
          ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
        });
  });

  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **b_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(b_shared, b, batch_size * sizeof(T *)).wait();
  for (std::int64_t i = 0; i < batch_size; ++i) {
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;
    // Need to create a copy of input matrices "a" to keep them unchanged.
    // Matrices "b" (copy of matrices "a") will be used as input and output
    // parameter in oneapi::mkl::lapack::getri_batch call.
    blas::matrix_mem_copy(b_shared[i], a_shared[i], ldb, lda, n, n, sizeof(T),
                          ::dpct::cs::memcpy_direction::device_to_device,
                          exec_queue);
  }

  sycl::event e = oneapi::mkl::lapack::getri_batch(
      exec_queue, n_int64, (Ty **)b_shared, ldb_int64, ipiv_int64_ptr, 1,
      group_sizes, scratchpad, scratchpad_size);

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] {
      delete n_int64;
      delete ldb_int64;
      delete group_sizes;
    });
  });
  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64, a_shared,
                           b_shared};
  ::dpct::cs::enqueue_free(ptrs, {e}, exec_queue);
#endif
}

/// Computes the QR factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] m The number of rows in the matrices.
/// \param [in] n The number of columns in the matrices.
/// \param [in, out] a Array of pointers to matrices. These
/// matrices will be overwritten by the factorization data.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [out] tau An array stores the scalars.
/// \param [out] info A value stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void geqrf_batch_wrapper(sycl::queue exec_queue, int m, int n, T *a[],
                                int lda, T *tau[], int *info, int batch_size) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  // Set the info value to 0
  *info = 0;
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_a = n * lda;
  std::int64_t stride_tau = (std::max)(1, (std::min)(m, n));
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Ty>(
          exec_queue, m, n, lda, stride_a, stride_tau, batch_size);

  T *a_buffer_ptr, *tau_buffer_ptr;
  a_buffer_ptr = (T *)::dpct::cs::malloc(stride_a * batch_size * sizeof(T));
  tau_buffer_ptr = (T *)::dpct::cs::malloc(stride_tau * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  T **host_tau = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_tau, tau, batch_size * sizeof(T *));

  for (std::int64_t i = 0; i < batch_size; ++i)
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));
  {
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    auto tau_buffer = get_buffer<Ty>(tau_buffer_ptr);
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    oneapi::mkl::lapack::geqrf_batch(exec_queue, m, n, a_buffer, lda, stride_a,
                                     tau_buffer, stride_tau, batch_size,
                                     scratchpad, scratchpad_size);
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events_a;
  std::vector<sycl::event> events_tau;
  for (std::int64_t i = 0; i < batch_size; ++i) {
    events_a.push_back(::dpct::cs::memcpy(
        exec_queue, host_a[i], a_buffer_ptr + i * stride_a, n * lda * sizeof(T),
        ::dpct::cs::memcpy_direction::automatic));
    events_tau.push_back(::dpct::cs::memcpy(
        exec_queue, host_tau[i], tau_buffer_ptr + i * stride_tau,
        (std::max)(1, (std::min)(m, n)) * sizeof(T),
        ::dpct::cs::memcpy_direction::automatic));
  }
  std::vector<void *> ptr_a{host_a};
  std::vector<void *> ptr_tau{host_tau};
  std::thread mem_free_thread_a(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptr_a, events_a);
  std::thread mem_free_thread_tau(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptr_tau, events_tau);
  mem_free_thread_a.detach();
  mem_free_thread_tau.detach();
#else
  std::int64_t *m_int64 = new std::int64_t;
  std::int64_t *n_int64 = new std::int64_t;
  std::int64_t *lda_int64 = new std::int64_t;
  std::int64_t *group_sizes = new std::int64_t;
  *m_int64 = n;
  *n_int64 = n;
  *lda_int64 = lda;
  *group_sizes = batch_size;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Ty>(
          exec_queue, m_int64, n_int64, lda_int64, 1, group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **tau_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(tau_shared, tau, batch_size * sizeof(T *)).wait();

  sycl::event e = oneapi::mkl::lapack::geqrf_batch(
      exec_queue, m_int64, n_int64, (Ty **)a_shared, lda_int64,
      (Ty **)tau_shared, 1, group_sizes, scratchpad, scratchpad_size);

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] {
      delete m_int64;
      delete n_int64;
      delete lda_int64;
      delete group_sizes;
    });
  });
  std::vector<void *> ptrs{scratchpad, a_shared, tau_shared};
  ::dpct::cs::enqueue_free(ptrs, {e}, exec_queue);
#endif
}

namespace blas {
#ifdef __INTEL_MKL__
template <typename T>
inline oneapi::mkl::blas::compute_mode
deduce_compute_mode(std::optional<compute_type> ct, math_mode mm) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  if (ct) {
    switch (ct.value()) {
    case compute_type::f16_standard:
    case compute_type::f32_standard:
    case compute_type::f64_standard:
    case compute_type::i32_standard:
      return oneapi::mkl::blas::compute_mode::standard;
    case compute_type::f32:
      if constexpr (std::is_same_v<Ty, std::complex<float>> ||
                    std::is_same_v<Ty, std::complex<double>>)
        return oneapi::mkl::blas::compute_mode::complex_3m;
      break;
    case compute_type::f32_fast_bf16:
      return oneapi::mkl::blas::compute_mode::float_to_bf16;
    case compute_type::f32_fast_tf32:
      return oneapi::mkl::blas::compute_mode::float_to_tf32;
    default:
      [[fallthrough]];
    }
  }
  if (mm == math_mode::mm_tf32)
    return oneapi::mkl::blas::compute_mode::float_to_tf32;
  return oneapi::mkl::blas::compute_mode::unset;
}

inline oneapi::mkl::blas::compute_mode
deduce_compute_mode(std::optional<compute_type> ct, math_mode mm,
                    bool is_complex) {
  if (is_complex)
    return deduce_compute_mode<std::complex<float>>(ct, mm);
  return deduce_compute_mode<float>(ct, mm);
}
#endif

/// Computes matrix-matrix product with general matrices.
/// \param [in] desc_ptr Descriptor.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] ct Compute type.
inline void gemm(descriptor_ptr desc_ptr, oneapi::mkl::transpose a_trans,
                 oneapi::mkl::transpose b_trans, std::int64_t m, std::int64_t n,
                 std::int64_t k, const void *alpha, const void *a,
                 library_data_t a_type, std::int64_t lda, const void *b,
                 library_data_t b_type, std::int64_t ldb, const void *beta,
                 void *c, library_data_t c_type, std::int64_t ldc,
                 std::variant<compute_type, library_data_t> ct) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  sycl::queue q = desc_ptr->get_queue();
  oneapi::mkl::blas::compute_mode cm = oneapi::mkl::blas::compute_mode::unset;
  library_data_t scaling_type;
  if (auto ct_p = std::get_if<compute_type>(&ct)) {
    cm = deduce_compute_mode(*ct_p, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
    scaling_type = detail::compute_type_to_library_data_t(*ct_p);
  } else {
    cm = deduce_compute_mode(std::nullopt, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
    scaling_type = std::get<library_data_t>(ct);
  }

  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (scaling_type == library_data_t::real_double &&
             c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key = dpct::detail::get_type_combination_id(
      a_type, b_type, c_type, scaling_type);
  switch (key) {
  case dpct::detail::get_type_combination_id(
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_impl<float, float, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double, library_data_t::real_double): {
    dpct::detail::gemm_impl<double, double, double, double>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float, library_data_t::complex_float): {
    dpct::detail::gemm_impl<std::complex<float>, std::complex<float>,
                            std::complex<float>, std::complex<float>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double, library_data_t::complex_double): {
    dpct::detail::gemm_impl<std::complex<double>, std::complex<double>,
                            std::complex<double>, std::complex<double>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_half): {
    dpct::detail::gemm_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float,
                            float>(q, a_trans, b_trans, m, n, k, alpha, a, lda,
                                   b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_impl<sycl::half, sycl::half, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value =
        dpct::get_value(reinterpret_cast<const float *>(beta), q);
    sycl::half alpha_half(alpha_value);
    sycl::half beta_half(beta_value);
    dpct::detail::gemm_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
        q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half,
        c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_impl<std::int8_t, std::int8_t, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_bfloat16, library_data_t::real_float): {
    dpct::detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                            oneapi::mkl::bfloat16, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_int32, library_data_t::real_int32): {
    float alpha_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
    float beta_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
    dpct::detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, float>(
        q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float,
        c, ldc, cm);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#endif
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] desc_ptr Descriptor.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
/// \param [in] scaling_type Data type of the scaling factors.
/// \param [in] ct Compute type.
inline void gemm_batch(descriptor_ptr desc_ptr, oneapi::mkl::transpose a_trans,
                       oneapi::mkl::transpose b_trans, std::int64_t m,
                       std::int64_t n, std::int64_t k, const void *alpha,
                       const void *a[], library_data_t a_type, std::int64_t lda,
                       const void *b[], library_data_t b_type, std::int64_t ldb,
                       const void *beta, void *c[], library_data_t c_type,
                       std::int64_t ldc, std::int64_t batch_size,
                       std::variant<compute_type, library_data_t> ct) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  sycl::queue q = desc_ptr->get_queue();
#ifdef __INTEL_MKL__
  oneapi::mkl::blas::compute_mode cm = oneapi::mkl::blas::compute_mode::unset;
#endif
  library_data_t scaling_type;
  if (auto ct_p = std::get_if<compute_type>(&ct)) {
#ifdef __INTEL_MKL__
    cm = deduce_compute_mode(*ct_p, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
#endif
    scaling_type = detail::compute_type_to_library_data_t(*ct_p);
  } else {
#ifdef __INTEL_MKL__
    cm = deduce_compute_mode(std::nullopt, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
#endif
    scaling_type = std::get<library_data_t>(ct);
  }

  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (scaling_type == library_data_t::real_double &&
             c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key = dpct::detail::get_type_combination_id(
      a_type, b_type, c_type, scaling_type);
  switch (key) {
  case dpct::detail::get_type_combination_id(
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<float, float, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double, library_data_t::real_double): {
    dpct::detail::gemm_batch_impl<double, double, double, double>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float, library_data_t::complex_float): {
    dpct::detail::gemm_batch_impl<std::complex<float>, std::complex<float>,
                                  std::complex<float>, std::complex<float>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double, library_data_t::complex_double): {
    dpct::detail::gemm_batch_impl<std::complex<double>, std::complex<double>,
                                  std::complex<double>, std::complex<double>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_half): {
    dpct::detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                  sycl::half>(q, a_trans, b_trans, m, n, k,
                                              alpha, a, lda, b, ldb, beta, c,
                                              ldc, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
#ifdef __INTEL_MKL__
  case dpct::detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_bfloat16, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                                  oneapi::mkl::bfloat16, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                                  float, float>(q, a_trans, b_trans, m, n, k,
                                                alpha, a, lda, b, ldb, beta, c,
                                                ldc, batch_size, cm);
    break;
  }
#endif
  case dpct::detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_int32, library_data_t::real_int32): {
    float alpha_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
    float beta_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
    dpct::detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                  float>(
        q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float,
        c, ldc, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value =
        dpct::get_value(reinterpret_cast<const float *>(beta), q);
    sycl::half alpha_half(alpha_value);
    sycl::half beta_half(beta_value);
    dpct::detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                  sycl::half>(
        q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half,
        c, ldc, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#endif
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] desc_ptr Descriptor.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] stride_a Stride between the different A matrices.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] stride_b Stride between the different B matrices.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] stride_c Stride between the different C matrices.
/// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
/// \param [in] ct Compute type.
inline void gemm_batch(descriptor_ptr desc_ptr, oneapi::mkl::transpose a_trans,
                       oneapi::mkl::transpose b_trans, std::int64_t m,
                       std::int64_t n, std::int64_t k, const void *alpha,
                       const void *a, library_data_t a_type, std::int64_t lda,
                       long long int stride_a, const void *b,
                       library_data_t b_type, std::int64_t ldb,
                       long long int stride_b, const void *beta, void *c,
                       library_data_t c_type, std::int64_t ldc,
                       long long int stride_c, std::int64_t batch_size,
                       std::variant<compute_type, library_data_t> ct) {
  sycl::queue q = desc_ptr->get_queue();
#ifdef __INTEL_MKL__
  oneapi::mkl::blas::compute_mode cm = oneapi::mkl::blas::compute_mode::unset;
#endif
  library_data_t scaling_type;
  if (auto ct_p = std::get_if<compute_type>(&ct)) {
#ifdef __INTEL_MKL__
    cm = deduce_compute_mode(*ct_p, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
#endif
    scaling_type = detail::compute_type_to_library_data_t(*ct_p);
  } else {
#ifdef __INTEL_MKL__
    cm = deduce_compute_mode(std::nullopt, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
#endif
    scaling_type = std::get<library_data_t>(ct);
  }

  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (scaling_type == library_data_t::real_double &&
             c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key = dpct::detail::get_type_combination_id(
      a_type, b_type, c_type, scaling_type);
  switch (key) {
  case dpct::detail::get_type_combination_id(
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<float, float, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double, library_data_t::real_double): {
    dpct::detail::gemm_batch_impl<double, double, double, double>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float, library_data_t::complex_float): {
    dpct::detail::gemm_batch_impl<std::complex<float>, std::complex<float>,
                                  std::complex<float>, std::complex<float>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double, library_data_t::complex_double): {
    dpct::detail::gemm_batch_impl<std::complex<double>, std::complex<double>,
                                  std::complex<double>, std::complex<double>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_half): {
    dpct::detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                  sycl::half>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
#ifdef __INTEL_MKL__
  case dpct::detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_bfloat16, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                                  oneapi::mkl::bfloat16, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size, cm);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                                  float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size, cm);
    break;
  }
#endif
  case dpct::detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_int32, library_data_t::real_int32): {
    float alpha_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
    float beta_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
    dpct::detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                  float>(
        q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, stride_a, b, ldb,
        stride_b, &beta_float, c, ldc, stride_c,
        batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_float, library_data_t::real_float): {
    dpct::detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value =
        dpct::get_value(reinterpret_cast<const float *>(beta), q);
    sycl::half alpha_half(alpha_value);
    sycl::half beta_half(beta_value);
    dpct::detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                  sycl::half>(
        q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb,
        stride_b, &beta_half, c, ldc, stride_c, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Performs a symmetric/hermitian rank-k update.
/// \tparam is_hermitian True means current matrix is hermitian.
/// \param [in] desc_ptr Descriptor.
/// \param [in] uplo Specifies whether matrix c is upper or lower triangular.
/// \param [in] trans Specifies op(a), the transposition operation applied to
/// matrix a.
/// \param [in] n Number of rows and columns of matrix c.
/// \param [in] k Number of columns of matrix op(a).
/// \param [in] alpha Scaling factor for the rank-k update.
/// \param [in] a Input matrix a.
/// \param [in] a_type Data type of the matrix a.
/// \param [in] lda Leading dimension of the matrix a.
/// \param [in] beta Scaling factor for the rank-k update.
/// \param [in, out] c Input/Output matrix c.
/// \param [in] c_type Data type of the matrix c.
/// \param [in] ldc Leading dimension of the matrix c.
/// \param [in] ct Compute type.
template <bool is_hermitian>
inline void syherk(descriptor_ptr desc_ptr, oneapi::mkl::uplo uplo,
                   oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                   const void *alpha, const void *a, library_data_t a_type,
                   std::int64_t lda, const void *beta, void *c,
                   library_data_t c_type, std::int64_t ldc,
                   std::variant<compute_type, library_data_t> ct) {
  sycl::queue q = desc_ptr->get_queue();
#ifdef __INTEL_MKL__
  oneapi::mkl::blas::compute_mode cm = oneapi::mkl::blas::compute_mode::unset;
  if (auto ct_p = std::get_if<compute_type>(&ct)) {
    cm = deduce_compute_mode(*ct_p, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
  } else {
    cm = deduce_compute_mode(std::nullopt, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
  }
#endif
  std::uint64_t key = dpct::detail::get_type_combination_id(a_type, c_type);
  if (!is_hermitian &&
      dpct::detail::get_type_combination_id(
          library_data_t::real_float, library_data_t::real_float) == key) {
    dpct::detail::syherk_impl<false, float>(q, uplo, trans, n, k, alpha, a, lda,
                                            beta, c, ldc DPCT_COMPUTE_MODE_ARG);
  } else if (!is_hermitian && dpct::detail::get_type_combination_id(
                                  library_data_t::real_double,
                                  library_data_t::real_double) == key) {
    dpct::detail::syherk_impl<false, double>(q, uplo, trans, n, k, alpha, a,
                                             lda, beta, c,
                                             ldc DPCT_COMPUTE_MODE_ARG);
  } else if (dpct::detail::get_type_combination_id(
                 library_data_t::complex_float,
                 library_data_t::complex_float) == key) {
    dpct::detail::syherk_impl<is_hermitian, std::complex<float>>(
        q, uplo, trans, n, k, alpha, a, lda, beta, c,
        ldc DPCT_COMPUTE_MODE_ARG);
  } else if (dpct::detail::get_type_combination_id(
                 library_data_t::complex_double,
                 library_data_t::complex_double) == key) {
    dpct::detail::syherk_impl<is_hermitian, std::complex<double>>(
        q, uplo, trans, n, k, alpha, a, lda, beta, c,
        ldc DPCT_COMPUTE_MODE_ARG);
  } else {
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// This routines perform a special rank-k update of a symmetric matrix C by
/// general matrices A and B.
/// \param [in] desc_ptr Descriptor.
/// \param [in] uplo Specifies whether C's data is stored in its upper or lower triangle.
/// \param [in] trans Specifies the operation to apply.
/// \param [in] n The number of rows and columns in C.
/// \param [in] k The inner dimension of matrix multiplications.
/// \param [in] alpha Scaling factor for the rank-k update.
/// \param [in] a Input matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for the rank-k update.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] ldc Leading dimension of C.
template <class T>
inline void syrk(descriptor_ptr desc_ptr, oneapi::mkl::uplo uplo,
                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                 const T *alpha, const T *a, std::int64_t lda, const T *b,
                 std::int64_t ldb, const T *beta, T *c, std::int64_t ldc) {
  sycl::queue q = desc_ptr->get_queue();
#ifdef __INTEL_MKL__
  auto cm = deduce_compute_mode<T>(std::nullopt, desc_ptr->get_math_mode());
#endif
  dpct::detail::rk_impl<false, T, T>(q, uplo, trans, n, k, alpha, a, lda, b,
                                     ldb, beta, c, ldc DPCT_COMPUTE_MODE_ARG);
}

/// This routines perform a special rank-k update of a Hermitian matrix C by
/// general matrices A and B.
/// \param [in] desc_ptr Descriptor.
/// \param [in] uplo Specifies whether C's data is stored in its upper or lower triangle.
/// \param [in] trans Specifies the operation to apply.
/// \param [in] n The number of rows and columns in C.
/// \param [in] k The inner dimension of matrix multiplications.
/// \param [in] alpha Scaling factor for the rank-k update.
/// \param [in] a Input matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for the rank-k update.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] ldc Leading dimension of C.
template <class T, class Tbeta>
inline void herk(descriptor_ptr desc_ptr, oneapi::mkl::uplo uplo,
                 oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                 const T *alpha, const T *a, std::int64_t lda, const T *b,
                 std::int64_t ldb, const Tbeta *beta, T *c, std::int64_t ldc) {
  sycl::queue q = desc_ptr->get_queue();
#ifdef __INTEL_MKL__
  auto cm = deduce_compute_mode<T>(std::nullopt, desc_ptr->get_math_mode());
#endif
  dpct::detail::rk_impl<true, T, Tbeta>(q, uplo, trans, n, k, alpha, a, lda, b,
                                        ldb, beta, c,
                                        ldc DPCT_COMPUTE_MODE_ARG);
}

/// This routine performs a group of trsm operations. Each trsm solves an
/// equation of the form op(A) * X = alpha * B or X * op(A) = alpha * B.
/// \param [in] desc_ptr Descriptor.
/// \param [in] left_right Specifies A multiplies X on the left or on the right.
/// \param [in] upper_lower Specifies A is upper or lower triangular.
/// \param [in] trans Specifies the operation applied to A.
/// \param [in] unit_diag Specifies whether A is unit triangular.
/// \param [in] m Number of rows of the B matrices.
/// \param [in] n Number of columns of the B matrices.
/// \param [in] alpha Scaling factor for the solutions.
/// \param [in] a Input matrices A.
/// \param [in] a_type Data type of the matrices A.
/// \param [in] lda Leading dimension of the matrices A.
/// \param [in, out] b Input and output matrices B.
/// \param [in] b_type Data type of the matrices B.
/// \param [in] ldb Leading dimension of the matrices B.
/// \param [in] batch_size Specifies the number of trsm operations to perform.
/// \param [in] ct Compute type.
inline void trsm_batch(descriptor_ptr desc_ptr, oneapi::mkl::side left_right,
                       oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, std::int64_t m,
                       std::int64_t n, const void *alpha, const void **a,
                       library_data_t a_type, std::int64_t lda, void **b,
                       library_data_t b_type, std::int64_t ldb,
                       std::int64_t batch_size,
                       std::variant<compute_type, library_data_t> ct) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  sycl::queue q = desc_ptr->get_queue();
#ifdef __INTEL_MKL__
  oneapi::mkl::blas::compute_mode cm = oneapi::mkl::blas::compute_mode::unset;
  if (auto ct_p = std::get_if<compute_type>(&ct))
    cm = deduce_compute_mode(*ct_p, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
  else
    cm = deduce_compute_mode(std::nullopt, desc_ptr->get_math_mode(),
                             a_type == library_data_t::complex_float ||
                                 a_type == library_data_t::complex_double);
#endif
  std::uint64_t key = dpct::detail::get_type_combination_id(a_type, b_type);
  switch (key) {
  case dpct::detail::get_type_combination_id(library_data_t::real_float,
                                             library_data_t::real_float): {
    dpct::detail::trsm_batch_impl<float, float, float>(
        q, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(library_data_t::real_double,
                                             library_data_t::real_double): {
    dpct::detail::trsm_batch_impl<double, double, double>(
        q, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                             library_data_t::complex_float): {
    dpct::detail::trsm_batch_impl<std::complex<float>, std::complex<float>,
                                  std::complex<float>>(
        q, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  case dpct::detail::get_type_combination_id(library_data_t::complex_double,
                                             library_data_t::complex_double): {
    dpct::detail::trsm_batch_impl<std::complex<double>, std::complex<double>,
                                  std::complex<double>>(
        q, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, batch_size DPCT_COMPUTE_MODE_ARG);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#endif
}

/// Computes a triangular matrix-general matrix product.
/// \param [in] desc_ptr Descriptor.
/// \param [in] left_right Specifies A is on the left or right side of the
/// multiplication.
/// \param [in] upper_lower Specifies A is upper or lower triangular.
/// \param [in] trans Specifies the operation applied to A.
/// \param [in] unit_diag Specifies whether A is unit triangular.
/// \param [in] m Number of rows of B.
/// \param [in] n Number of columns of B.
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrices A.
/// \param [in] lda Leading dimension of the matrices A.
/// \param [in] b Input matrices B.
/// \param [in] ldb Leading dimension of the matrices B.
/// \param [out] c Output matrices C.
/// \param [in] ldc Leading dimension of the matrices C.
template <class T>
inline void trmm(descriptor_ptr desc_ptr, oneapi::mkl::side left_right,
                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                 const T *alpha, const T *a, std::int64_t lda, const T *b,
                 std::int64_t ldb, T *c, std::int64_t ldc) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  sycl::queue q = desc_ptr->get_queue();
#ifdef __INTEL_MKL__
  auto cm = deduce_compute_mode<Ty>(std::nullopt, desc_ptr->get_math_mode());
#endif
  auto alpha_val = dpct::get_value(alpha, q);
  if (b != c) {
    dpct::blas::matrix_mem_copy(c, b, ldc, ldb, m, n, sizeof(Ty),
                                ::dpct::cs::memcpy_direction::device_to_device,
                                q);
  }
  auto data_a = dpct::detail::get_memory<const Ty>(a);
  auto data_c = dpct::detail::get_memory<Ty>(c);
  oneapi::mkl::blas::column_major::trmm(q, left_right, upper_lower, trans,
                                        unit_diag, m, n, alpha_val, data_a, lda,
                                        data_c, ldc DPCT_COMPUTE_MODE_ARG);
}

/// Computes the Euclidean norm of a vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
inline void nrm2(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                 library_data_t x_type, std::int64_t incx, void *result,
                 library_data_t result_type) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key =
      ::dpct::detail::get_type_combination_id(x_type, result_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float,
                                               library_data_t::real_float): {
    ::dpct::detail::nrm2_impl<float, float>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double,
                                               library_data_t::real_double): {
    ::dpct::detail::nrm2_impl<double, double>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                               library_data_t::real_float): {
    ::dpct::detail::nrm2_impl<std::complex<float>, float>(q, n, x, incx,
                                                          result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_double,
                                               library_data_t::real_double): {
    ::dpct::detail::nrm2_impl<std::complex<double>, double>(q, n, x, incx,
                                                            result);
    break;
  }
#ifdef __INTEL_MKL__
  case ::dpct::detail::get_type_combination_id(library_data_t::real_half,
                                               library_data_t::real_half): {
    ::dpct::detail::nrm2_impl<sycl::half, sycl::half>(q, n, x, incx, result);
    break;
  }
#endif
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes the dot product of two vectors.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in] y Input vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
inline void dot(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                library_data_t x_type, std::int64_t incx, const void *y,
                library_data_t y_type, std::int64_t incy, void *result,
                library_data_t result_type) {
  sycl::queue q = desc_ptr->get_queue();
  ::dpct::detail::dotuc<false>(q, n, x, x_type, incx, y, y_type, incy, result,
                               result_type);
}

/// Computes the dot product of two vectors, conjugating the first vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in] y Input vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
inline void dotc(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                 library_data_t x_type, std::int64_t incx, const void *y,
                 library_data_t y_type, std::int64_t incy, void *result,
                 library_data_t result_type) {
  sycl::queue q = desc_ptr->get_queue();
  ::dpct::detail::dotuc<true>(q, n, x, x_type, incx, y, y_type, incy, result,
                              result_type);
}

/// Computes the product of a vector by a scalar.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] alpha The scale factor alpha.
/// \param [in] alpha_type The data type of alpha.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
inline void scal(descriptor_ptr desc_ptr, std::int64_t n, const void *alpha,
                 library_data_t alpha_type, void *x, library_data_t x_type,
                 std::int64_t incx) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key = ::dpct::detail::get_type_combination_id(x_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float): {
    ::dpct::detail::scal_impl<float, float>(q, n, alpha, x, incx);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double): {
    ::dpct::detail::scal_impl<double, double>(q, n, alpha, x, incx);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float): {
    ::dpct::detail::scal_impl<std::complex<float>, std::complex<float>>(
        q, n, alpha, x, incx);
    break;
  }
  case ::dpct::detail::get_type_combination_id(
      library_data_t::complex_double): {
    ::dpct::detail::scal_impl<std::complex<double>, std::complex<double>>(
        q, n, alpha, x, incx);
    break;
  }
#ifdef __INTEL_MKL__
  case ::dpct::detail::get_type_combination_id(library_data_t::real_half): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    sycl::half alaph_half(alpha_value);
    ::dpct::detail::scal_impl<sycl::half, sycl::half>(q, n, &alaph_half, x,
                                                      incx);
    break;
  }
#endif
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a vector-scalar product and adds the result to a vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] alpha The scale factor alpha.
/// \param [in] alpha_type The data type of alpha.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
inline void axpy(descriptor_ptr desc_ptr, std::int64_t n, const void *alpha,
                 library_data_t alpha_type, const void *x,
                 library_data_t x_type, std::int64_t incx, void *y,
                 library_data_t y_type, std::int64_t incy) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key =
      ::dpct::detail::get_type_combination_id(x_type, alpha_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float,
                                               library_data_t::real_float): {
    ::dpct::detail::axpy_impl<float, float>(q, n, alpha, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double,
                                               library_data_t::real_double): {
    ::dpct::detail::axpy_impl<double, double>(q, n, alpha, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                               library_data_t::complex_float): {
    ::dpct::detail::axpy_impl<std::complex<float>, std::complex<float>>(
        q, n, alpha, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double): {
    ::dpct::detail::axpy_impl<std::complex<double>, std::complex<double>>(
        q, n, alpha, x, incx, y, incy);
    break;
  }
#ifdef __INTEL_MKL__
  case ::dpct::detail::get_type_combination_id(library_data_t::real_half,
                                               library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    sycl::half alaph_half(alpha_value);
    ::dpct::detail::axpy_impl<sycl::half, sycl::half>(q, n, &alaph_half, x,
                                                      incx, y, incy);
    break;
  }
#endif
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Performs rotation of points in the plane.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [in] c Scaling factor.
/// \param [in] s Scaling factor.
/// \param [in] cs_type Data type of the scaling factors.
inline void rot(descriptor_ptr desc_ptr, std::int64_t n, void *x,
                library_data_t x_type, std::int64_t incx, void *y,
                library_data_t y_type, std::int64_t incy, const void *c,
                const void *s, library_data_t cs_type) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key = ::dpct::detail::get_type_combination_id(x_type, cs_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float,
                                               library_data_t::real_float): {
    ::dpct::detail::rot_impl<float, float, float>(q, n, x, incx, y, incy, c, s);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double,
                                               library_data_t::real_double): {
    ::dpct::detail::rot_impl<double, double, double>(q, n, x, incx, y, incy, c,
                                                     s);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                               library_data_t::real_float): {
    ::dpct::detail::rot_impl<std::complex<float>, float, float>(q, n, x, incx,
                                                                y, incy, c, s);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_double,
                                               library_data_t::real_double): {
    ::dpct::detail::rot_impl<std::complex<double>, double, double>(
        q, n, x, incx, y, incy, c, s);
    break;
  }
#ifdef __INTEL_MKL__
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                               library_data_t::complex_float): {
    ::dpct::detail::rot_impl<std::complex<float>, float, std::complex<float>>(
        q, n, x, incx, y, incy, c, s);
    break;
  }
  case ::dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double): {
    ::dpct::detail::rot_impl<std::complex<double>, double,
                             std::complex<double>>(q, n, x, incx, y, incy, c,
                                                   s);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_half,
                                               library_data_t::real_half): {
    ::dpct::detail::rot_impl<sycl::half, sycl::half, sycl::half>(q, n, x, incx,
                                                                 y, incy, c, s);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_bfloat16,
                                               library_data_t::real_bfloat16): {
    ::dpct::detail::rot_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                             oneapi::mkl::bfloat16>(q, n, x, incx, y, incy, c,
                                                    s);
    break;
  }
#endif
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Performs modified Givens rotation of points in the plane.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [in] param Array of 5 parameters.
/// \param [in] param_type Data type of \p param.
inline void rotm(descriptor_ptr desc_ptr, std::int64_t n, void *x,
                 library_data_t x_type, int64_t incx, void *y,
                 library_data_t y_type, int64_t incy, const void *param,
                 library_data_t param_type) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key =
      ::dpct::detail::get_type_combination_id(x_type, param_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float,
                                               library_data_t::real_float,
                                               library_data_t::real_float): {
    ::dpct::detail::rotm_impl<float, float, float>(q, n, x, incx, y, incy,
                                                   param);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double,
                                               library_data_t::real_double,
                                               library_data_t::real_double): {
    ::dpct::detail::rotm_impl<double, double, double>(q, n, x, incx, y, incy,
                                                      param);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Copies a vector to another vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] y Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
inline void copy(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                 library_data_t x_type, std::int64_t incx, void *y,
                 library_data_t y_type, std::int64_t incy) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key = ::dpct::detail::get_type_combination_id(x_type, y_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float,
                                               library_data_t::real_float): {
    ::dpct::detail::copy_impl<float, float>(q, n, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double,
                                               library_data_t::real_double): {
    ::dpct::detail::copy_impl<double, double>(q, n, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                               library_data_t::complex_float): {
    ::dpct::detail::copy_impl<std::complex<float>, std::complex<float>>(
        q, n, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double): {
    ::dpct::detail::copy_impl<std::complex<double>, std::complex<double>>(
        q, n, x, incx, y, incy);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Swaps a vector with another vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
inline void swap(descriptor_ptr desc_ptr, std::int64_t n, void *x,
                 library_data_t x_type, std::int64_t incx, void *y,
                 library_data_t y_type, std::int64_t incy) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key = ::dpct::detail::get_type_combination_id(x_type, y_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float,
                                               library_data_t::real_float): {
    ::dpct::detail::swap_impl<float, float>(q, n, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double,
                                               library_data_t::real_double): {
    ::dpct::detail::swap_impl<double, double>(q, n, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                               library_data_t::complex_float): {
    ::dpct::detail::swap_impl<std::complex<float>, std::complex<float>>(
        q, n, x, incx, y, incy);
    break;
  }
  case ::dpct::detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double): {
    ::dpct::detail::swap_impl<std::complex<double>, std::complex<double>>(
        q, n, x, incx, y, incy);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes the sum of magnitudes of the vector elements.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The scalar result.
/// \param [in] result_type Data type of \p result.
inline void asum(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                 library_data_t x_type, std::int64_t incx, void *result,
                 library_data_t result_type) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key =
      ::dpct::detail::get_type_combination_id(x_type, result_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float,
                                               library_data_t::real_float): {
    ::dpct::detail::asum_impl<float, float>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double,
                                               library_data_t::real_double): {
    ::dpct::detail::asum_impl<double, double>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float,
                                               library_data_t::real_float): {
    ::dpct::detail::asum_impl<std::complex<float>, float>(q, n, x, incx,
                                                          result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_double,
                                               library_data_t::real_double): {
    ::dpct::detail::asum_impl<std::complex<double>, double>(q, n, x, incx,
                                                            result);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Finds the index of the element with the largest absolute value in a vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The index of the maximal element.
inline void iamax(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                  library_data_t x_type, std::int64_t incx,
                  std::int64_t *result) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key = ::dpct::detail::get_type_combination_id(x_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float): {
    ::dpct::detail::iamax_impl<float>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double): {
    ::dpct::detail::iamax_impl<double>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float): {
    ::dpct::detail::iamax_impl<std::complex<float>>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(
      library_data_t::complex_double): {
    ::dpct::detail::iamax_impl<std::complex<double>>(q, n, x, incx, result);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Finds the index of the element with the smallest absolute value in a vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The index of the minimum element.
inline void iamin(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                  library_data_t x_type, std::int64_t incx,
                  std::int64_t *result) {
  sycl::queue q = desc_ptr->get_queue();
  std::uint64_t key = ::dpct::detail::get_type_combination_id(x_type);
  switch (key) {
  case ::dpct::detail::get_type_combination_id(library_data_t::real_float): {
    ::dpct::detail::iamin_impl<float>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::real_double): {
    ::dpct::detail::iamin_impl<double>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(library_data_t::complex_float): {
    ::dpct::detail::iamin_impl<std::complex<float>>(q, n, x, incx, result);
    break;
  }
  case ::dpct::detail::get_type_combination_id(
      library_data_t::complex_double): {
    ::dpct::detail::iamin_impl<std::complex<double>>(q, n, x, incx, result);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Finds the index of the element with the largest absolute value in a vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The index of the maximal element.
inline void iamax(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                  library_data_t x_type, std::int64_t incx, int *result) {
  dpct::blas::wrapper_int_to_int64_out wrapper(desc_ptr->get_queue(), result);
  iamax(desc_ptr, n, x, x_type, incx, wrapper.get_ptr());
}

/// Finds the index of the element with the smallest absolute value in a vector.
/// \param [in] desc_ptr Descriptor.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The index of the minimum element.
inline void iamin(descriptor_ptr desc_ptr, std::int64_t n, const void *x,
                  library_data_t x_type, std::int64_t incx, int *result) {
  dpct::blas::wrapper_int_to_int64_out wrapper(desc_ptr->get_queue(), result);
  iamin(desc_ptr, n, x, x_type, incx, wrapper.get_ptr());
}

/// Finds the least squares solutions for a batch of overdetermined linear
/// systems. Uses the QR factorization to solve a grouped batch of linear
/// systems with full rank matrices.
/// \param [in] desc_ptr Descriptor.
/// \param [in] trans Operation applied to \p a.
/// \param [in] m The number of rows of \p a.
/// \param [in] n The number of columns of \p a.
/// \param [in] nrhs The number of columns of \p b.
/// \param [in, out] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of \p a.
/// \param [in, out] b Array of pointers to matrices.
/// \param [in] ldb The leading dimension of \p b.
/// \param [out] info Set to 0 if no error.
/// \param [out] dev_info Optional. If it is not NULL : dev_info[i]==0 means the
/// i-th problem is successful; dev_info[i]!=0 means dev_info[i] is the first
/// zero diagonal element of the i-th \p a .
/// \param [in] batch_size The size of the batch.
template <typename T>
inline sycl::event
gels_batch_wrapper(descriptor_ptr desc_ptr, oneapi::mkl::transpose trans, int m,
                   int n, int nrhs, T *const a[], int lda, T *const b[],
                   int ldb, int *info, int *dev_info, int batch_size) {
#if defined(DPCT_USM_LEVEL_NONE) || !defined(__INTEL_MKL__)
#if defined(DPCT_USM_LEVEL_NONE)
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#endif
#else
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  sycl::queue exec_queue = desc_ptr->get_queue();
  struct matrix_info_t {
    oneapi::mkl::transpose trans_info;
    std::int64_t m_info;
    std::int64_t n_info;
    std::int64_t nrhs_info;
    std::int64_t lda_info;
    std::int64_t ldb_info;
    std::int64_t group_size_info;
  };
  matrix_info_t *matrix_info =
      (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
  matrix_info->trans_info = trans;
  matrix_info->m_info = m;
  matrix_info->n_info = n;
  matrix_info->nrhs_info = nrhs;
  matrix_info->lda_info = lda;
  matrix_info->ldb_info = ldb;
  matrix_info->group_size_info = batch_size;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::gels_batch_scratchpad_size<Ty>(
          exec_queue, &(matrix_info->trans_info), &(matrix_info->m_info),
          &(matrix_info->n_info), &(matrix_info->nrhs_info),
          &(matrix_info->lda_info), &(matrix_info->ldb_info), 1,
          &(matrix_info->group_size_info));
  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);

  *info = 0;
  if (dev_info)
    exec_queue.memset(dev_info, 0, batch_size * sizeof(int));
  static const std::vector<sycl::event> empty_events{};
  static const std::string api_name = "oneapi::mkl::lapack::gels_batch";
  sycl::event e = ::dpct::detail::catch_batch_error_f<sycl::event>(
      nullptr, api_name, exec_queue, info, dev_info, batch_size,
      oneapi::mkl::lapack::gels_batch, exec_queue, &(matrix_info->trans_info),
      &(matrix_info->m_info), &(matrix_info->n_info), &(matrix_info->nrhs_info),
      (Ty **)a, &(matrix_info->lda_info), (Ty **)b, &(matrix_info->ldb_info),
      (std::int64_t)1, &(matrix_info->group_size_info), (Ty *)scratchpad,
      (std::int64_t)scratchpad_size, empty_events);

  return exec_queue.submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() mutable {
      ::dpct::detail::catch_batch_error(
          nullptr, api_name, exec_queue, info, dev_info,
          matrix_info->group_size_info,
          [](sycl::event _e) {
            _e.wait_and_throw();
            return 0;
          },
          e);
      std::free(matrix_info);
      sycl::free(scratchpad, exec_queue);
    });
  });
#endif
}
} // namespace blas

/// Computes matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] scaling_type Data type of the scaling factors.
[[deprecated("Please use dpct::blas::gemm(...) instead.")]] inline void
gemm(sycl::queue &q, oneapi::mkl::transpose a_trans,
     oneapi::mkl::transpose b_trans, int m, int n, int k, const void *alpha,
     const void *a, library_data_t a_type, int lda, const void *b,
     library_data_t b_type, int ldb, const void *beta, void *c,
     library_data_t c_type, int ldc, library_data_t scaling_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::gemm(&desc, a_trans, b_trans, m, n, k, alpha, a, a_type, lda, b, b_type,
             ldb, beta, c, c_type, ldc, scaling_type);
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
/// \param [in] scaling_type Data type of the scaling factors.
[[deprecated("Please use dpct::blas::gemm_batch(...) instead.")]] inline void
gemm_batch(sycl::queue &q, oneapi::mkl::transpose a_trans,
           oneapi::mkl::transpose b_trans, int m, int n, int k,
           const void *alpha, const void *a[], library_data_t a_type, int lda,
           const void *b[], library_data_t b_type, int ldb, const void *beta,
           void *c[], library_data_t c_type, int ldc, int batch_size,
           library_data_t scaling_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::gemm_batch(&desc, a_trans, b_trans, m, n, k, alpha, a, a_type, lda, b,
                   b_type, ldb, beta, c, c_type, ldc, batch_size, scaling_type);
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] stride_a Stride between the different A matrices.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] stride_b Stride between the different B matrices.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] stride_c Stride between the different C matrices.
/// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
/// \param [in] scaling_type Data type of the scaling factors.
[[deprecated("Please use dpct::blas::gemm_batch(...) instead.")]] inline void
gemm_batch(sycl::queue &q, oneapi::mkl::transpose a_trans,
           oneapi::mkl::transpose b_trans, int m, int n, int k,
           const void *alpha, const void *a, library_data_t a_type, int lda,
           long long int stride_a, const void *b, library_data_t b_type,
           int ldb, long long int stride_b, const void *beta, void *c,
           library_data_t c_type, int ldc, long long int stride_c,
           int batch_size, library_data_t scaling_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::gemm_batch(&desc, a_trans, b_trans, m, n, k, alpha, a, a_type, lda,
                   stride_a, b, b_type, ldb, stride_b, beta, c, c_type, ldc,
                   stride_c, batch_size, scaling_type);
}

/// This routines perform a special rank-k update of a symmetric matrix C by
/// general matrices A and B.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] uplo Specifies whether C's data is stored in its upper or lower triangle.
/// \param [in] trans Specifies the operation to apply.
/// \param [in] n The number of rows and columns in C.
/// \param [in] k The inner dimension of matrix multiplications.
/// \param [in] alpha Scaling factor for the rank-k update.
/// \param [in] a Input matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for the rank-k update.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] ldc Leading dimension of C.
template <class T>
[[deprecated("Please use dpct::blas::syrk(...) instead.")]] inline void
syrk(sycl::queue &q, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
     int n, int k, const T *alpha, const T *a, int lda, const T *b, int ldb,
     const T *beta, T *c, int ldc) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::syrk<T>(&desc, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// This routines perform a special rank-k update of a Hermitian matrix C by
/// general matrices A and B.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] uplo Specifies whether C's data is stored in its upper or lower triangle.
/// \param [in] trans Specifies the operation to apply.
/// \param [in] n The number of rows and columns in C.
/// \param [in] k The inner dimension of matrix multiplications.
/// \param [in] alpha Scaling factor for the rank-k update.
/// \param [in] a Input matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for the rank-k update.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] ldc Leading dimension of C.
template <class T, class Tbeta>
[[deprecated("Please use dpct::blas::herk(...) instead.")]] inline void
herk(sycl::queue &q, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
     int n, int k, const T *alpha, const T *a, int lda, const T *b, int ldb,
     const Tbeta *beta, T *c, int ldc) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::herk(&desc, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// This routine performs a group of trsm operations. Each trsm solves an
/// equation of the form op(A) * X = alpha * B or X * op(A) = alpha * B.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] left_right Specifies A multiplies X on the left or on the right.
/// \param [in] upper_lower Specifies A is upper or lower triangular.
/// \param [in] trans Specifies the operation applied to A.
/// \param [in] unit_diag Specifies whether A is unit triangular.
/// \param [in] m Number of rows of the B matrices.
/// \param [in] n Number of columns of the B matrices.
/// \param [in] alpha Scaling factor for the solutions.
/// \param [in] a Input matrices A.
/// \param [in] a_type Data type of the matrices A.
/// \param [in] lda Leading dimension of the matrices A.
/// \param [in, out] b Input and output matrices B.
/// \param [in] b_type Data type of the matrices B.
/// \param [in] ldb Leading dimension of the matrices B.
/// \param [in] batch_size Specifies the number of trsm operations to perform.
/// \param [in] scaling_type Data type of the scaling factors.
[[deprecated("Please use dpct::blas::trsm_batch(...) instead.")]] inline void
trsm_batch(sycl::queue &q, oneapi::mkl::side left_right,
           oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           oneapi::mkl::diag unit_diag, int m, int n, const void *alpha,
           const void **a, library_data_t a_type, int lda, void **b,
           library_data_t b_type, int ldb, int batch_size,
           library_data_t scaling_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::trsm_batch(&desc, left_right, upper_lower, trans, unit_diag, m, n,
                   alpha, a, a_type, lda, b, b_type, ldb, batch_size,
                   scaling_type);
}

/// Computes a triangular matrix-general matrix product.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] left_right Specifies A is on the left or right side of the
/// multiplication.
/// \param [in] upper_lower Specifies A is upper or lower triangular.
/// \param [in] trans Specifies the operation applied to A.
/// \param [in] unit_diag Specifies whether A is unit triangular.
/// \param [in] m Number of rows of B.
/// \param [in] n Number of columns of B.
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrices A.
/// \param [in] lda Leading dimension of the matrices A.
/// \param [in] b Input matrices B.
/// \param [in] ldb Leading dimension of the matrices B.
/// \param [out] c Output matrices C.
/// \param [in] ldc Leading dimension of the matrices C.
template <class T>
[[deprecated("Please use dpct::blas::trmm(...) instead.")]] inline void
trmm(sycl::queue &q, oneapi::mkl::side left_right,
     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
     oneapi::mkl::diag unit_diag, int m, int n, const T *alpha, const T *a,
     int lda, const T *b, int ldb, T *c, int ldc) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::trmm<T>(&desc, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                a, lda, b, ldb, c, ldc);
}

/// Computes the Euclidean norm of a vector.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
[[deprecated("Please use dpct::blas::nrm2(...) instead.")]] inline void
nrm2(sycl::queue &q, int n, const void *x, library_data_t x_type, int incx,
     void *result, library_data_t result_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::nrm2(&desc, n, x, x_type, incx, result, result_type);
}

/// Computes the dot product of two vectors.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in] y Input vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
[[deprecated("Please use dpct::blas::dot(...) instead.")]] inline void
dot(sycl::queue &q, int n, const void *x, library_data_t x_type, int incx,
    const void *y, library_data_t y_type, int incy, void *result,
    library_data_t result_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::dot(&desc, n, x, x_type, incx, y, y_type, incy, result, result_type);
}

/// Computes the dot product of two vectors, conjugating the first vector.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in] y Input vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
[[deprecated("Please use dpct::blas::dotc(...) instead.")]] inline void
dotc(sycl::queue &q, int n, const void *x, library_data_t x_type, int incx,
     const void *y, library_data_t y_type, int incy, void *result,
     library_data_t result_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::dotc(&desc, n, x, x_type, incx, y, y_type, incy, result, result_type);
}

/// Computes the product of a vector by a scalar.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] alpha The scale factor alpha.
/// \param [in] alpha_type The data type of alpha.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
[[deprecated("Please use dpct::blas::scal(...) instead.")]] inline void
scal(sycl::queue &q, int n, const void *alpha, library_data_t alpha_type,
     void *x, library_data_t x_type, int incx) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::scal(&desc, n, alpha, alpha_type, x, x_type, incx);
}

/// Computes a vector-scalar product and adds the result to a vector.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] alpha The scale factor alpha.
/// \param [in] alpha_type The data type of alpha.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
[[deprecated("Please use dpct::blas::axpy(...) instead.")]] inline void
axpy(sycl::queue &q, int n, const void *alpha, library_data_t alpha_type,
     const void *x, library_data_t x_type, int incx, void *y,
     library_data_t y_type, int incy) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::axpy(&desc, n, alpha, alpha_type, x, x_type, incx, y, y_type, incy);
}

/// Performs rotation of points in the plane.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [in] c Scaling factor.
/// \param [in] s Scaling factor.
/// \param [in] cs_type Data type of the scaling factors.
[[deprecated("Please use dpct::blas::rot(...) instead.")]] inline void
rot(sycl::queue &q, int n, void *x, library_data_t x_type, int incx, void *y,
    library_data_t y_type, int incy, const void *c, const void *s,
    library_data_t cs_type) {
  blas::descriptor desc;
  desc.set_queue(&q);
  blas::rot(&desc, n, x, x_type, incx, y, y_type, incy, c, s, cs_type);
}
} // namespace dpct
#undef DPCT_COMPUTE_MODE_ARG
#undef DPCT_COMPUTE_MODE_PARAM
#endif // __DPCT_BLAS_UTILS_HPP__
