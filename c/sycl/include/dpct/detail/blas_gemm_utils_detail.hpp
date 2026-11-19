//==---- blas_gemm_utils_detail.hpp---------------------*- C++ -*-----------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BLAS_GEMM_UTILS_DETAIL_HPP__
#define __DPCT_BLAS_GEMM_UTILS_DETAIL_HPP__

namespace dpct::blas_gemm::experimental::detail {
/// Sacling each row of matrix D with the corresponding element of vector alpha.
template <class T, class Tscale>
sycl::event scale_new_a_impl(::dpct::cs::queue_ptr q_ptr, int rows, int cols,
                             void *a_ptr, const void *alpha_ptr,
                             bool vector_alpha, bool device_alpha,
                             const void *a_scale_ptr, const void *b_scale_ptr,
                             const std::vector<sycl::event> &dependencies) {
  std::vector<sycl::event> deps = dependencies;
  T *a = (T *)a_ptr;
  Tscale *alpha = (Tscale *)alpha_ptr;
  Tscale *a_scale = (Tscale *)a_scale_ptr;
  Tscale *b_scale = (Tscale *)b_scale_ptr;

  if (!device_alpha) {
    Tscale *alpha_host = alpha;
    alpha = (Tscale *)::dpct::cs::malloc(sizeof(Tscale), *q_ptr);
    deps.push_back(
        ::dpct::cs::memcpy(*q_ptr, alpha, alpha_host, sizeof(Tscale)));
  }
  if (!a_scale_ptr) {
    a_scale = (Tscale *)::dpct::cs::malloc(sizeof(Tscale), *q_ptr);
    deps.push_back(::dpct::cs::fill<Tscale>(*q_ptr, a_scale, 1.0, 1));
  }
  if (!b_scale_ptr) {
    b_scale = (Tscale *)::dpct::cs::malloc(sizeof(Tscale), *q_ptr);
    deps.push_back(::dpct::cs::fill<Tscale>(*q_ptr, b_scale, 1.0, 1));
  }

  sycl::event e = q_ptr->submit([&](sycl::handler &cgh) {
    cgh.depends_on(deps);
#ifdef DPCT_USM_LEVEL_NONE
    access_wrapper<T *> a_acc(a, cgh);
    access_wrapper<Tscale *> alpha_acc(alpha, cgh);
    access_wrapper<Tscale *> a_scale_acc(a_scale, cgh);
    access_wrapper<Tscale *> b_scale_acc(b_scale, cgh);
#endif
    cgh.parallel_for<
        ::dpct::cs::kernel_name<class scale_with_alpha, T, Tscale>>(
        sycl::range<2>(rows, cols), [=](sycl::id<2> index) {
#ifdef DPCT_USM_LEVEL_NONE
          T *a_data = a_acc.get_raw_pointer();
          Tscale *alpha_data = alpha_acc.get_raw_pointer();
          Tscale *a_scale_data = a_scale_acc.get_raw_pointer();
          Tscale *b_scale_data = b_scale_acc.get_raw_pointer();
#else
          T *a_data = a;
          const Tscale *alpha_data = alpha;
          const Tscale *a_scale_data = a_scale;
          const Tscale *b_scale_data = b_scale;
#endif
          size_t row_idx = index.get(0);
          size_t col_idx = index.get(1);
          size_t idx = rows * col_idx + row_idx;

          Tscale ab_scale = a_scale_data[0] * b_scale_data[0];

          if (vector_alpha)
            a_data[idx] = a_data[idx] * alpha_data[row_idx] * ab_scale;
          else
            a_data[idx] = a_data[idx] * alpha_data[0] * ab_scale;
        });
  });
  return q_ptr->submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] {
      if (!device_alpha)
        ::dpct::cs::free(alpha, *q_ptr);
      if (!a_scale_ptr)
        ::dpct::cs::free(a_scale, *q_ptr);
      if (!b_scale_ptr)
        ::dpct::cs::free(b_scale, *q_ptr);
    });
  });
}

// a is col major without padding
inline sycl::event scale_new_a(::dpct::cs::queue_ptr q_ptr, int rows, int cols,
                               void *a, library_data_t a_type,
                               const void *alpha, library_data_t scale_type,
                               bool vector_alpha, bool device_alpha,
                               const void *a_scale, const void *b_scale,
                               const std::vector<sycl::event> &deps) {
  std::uint64_t key = dpct::detail::get_type_combination_id(a_type, scale_type);
  sycl::event e;
  switch (key) {
#define __SCALE_NEW_A_IMPL_CASE(A_TYPE_ENUM, SCALE_TYPE_ENUM, A_TYPE,          \
                                SCALE_TYPE)                                    \
  case dpct::detail::get_type_combination_id(                                  \
      library_data_t::A_TYPE_ENUM, library_data_t::SCALE_TYPE_ENUM): {         \
    e = scale_new_a_impl<A_TYPE, SCALE_TYPE>(q_ptr, rows, cols, a, alpha,      \
                                             vector_alpha, device_alpha,       \
                                             a_scale, b_scale, deps);          \
    break;                                                                     \
  }
    __SCALE_NEW_A_IMPL_CASE(real_int8, real_float, std::int8_t, float)
    __SCALE_NEW_A_IMPL_CASE(real_int32, real_float, int, float)
    __SCALE_NEW_A_IMPL_CASE(real_int8, real_int32, std::int8_t, int)
    __SCALE_NEW_A_IMPL_CASE(real_float, real_float, float, float)
#undef __SCALE_NEW_A_IMPL_CASE
  default:
    throw std::runtime_error("dpct::blas_gemm::experimental::detail::scale_new_"
                             "a_impl() does not support the data "
                             "type combination currently.");
  }
  return e;
}

/// Get a linear idx map for a 2D point (row_idx, col_idx) between src_order and
/// dst_order.
inline std::tuple<size_t, size_t>
get_linear_idx_map(size_t rows, size_t cols, size_t src_ld, order_t src_order,
                   size_t dst_ld, order_t dst_order, size_t row_idx,
                   size_t col_idx) {
#define COMBINE(from, to)                                                      \
  static_cast<std::uint16_t>(from) << 8 | static_cast<std::uint8_t>(to)

  size_t from_linear_idx, to_linear_idx;
  switch (COMBINE(src_order, dst_order)) {
  case COMBINE(order_t::col, order_t::row): {
    from_linear_idx = src_ld * col_idx + row_idx;
    to_linear_idx = dst_ld * row_idx + col_idx;
    break;
  }
  case COMBINE(order_t::row, order_t::col): {
    from_linear_idx = src_ld * row_idx + col_idx;
    to_linear_idx = dst_ld * col_idx + row_idx;
    break;
  }
  case COMBINE(order_t::col, order_t::col32): {
    from_linear_idx = src_ld * col_idx + row_idx;
    to_linear_idx = dst_ld * (col_idx / 32) + 32 * row_idx + col_idx % 32;
    break;
  }
  case COMBINE(order_t::col32, order_t::col): {
    from_linear_idx = src_ld * (col_idx / 32) + 32 * row_idx + col_idx % 32;
    to_linear_idx = dst_ld * col_idx + row_idx;
    break;
  }
  case COMBINE(order_t::col, order_t::col4_4r2_8c): {
    from_linear_idx = src_ld * col_idx + row_idx;

    size_t from_row_in_row8_col32 = row_idx % 8;
    size_t from_col_in_row8_col32 = col_idx % 32;

    size_t to_row_in_row8_col32 =
        4 * (from_row_in_row8_col32 % 2) + from_col_in_row8_col32 / 8;
    size_t to_col_in_row8_col32 = 16 * ((from_col_in_row8_col32 / 4) % 2) +
                                  4 * (from_row_in_row8_col32 / 2) +
                                  from_col_in_row8_col32 % 4;
    size_t to_linear_idx_in_row8_col32 =
        to_row_in_row8_col32 * 32 + to_col_in_row8_col32;

    to_linear_idx = dst_ld * (col_idx / 32) + (row_idx / 8) * (32 * 8) +
                    to_linear_idx_in_row8_col32;
    break;
  }
  case COMBINE(order_t::col4_4r2_8c, order_t::col): {
    to_linear_idx = dst_ld * col_idx + row_idx;

    size_t to_row_in_row8_col32 = row_idx % 8;
    size_t to_col_in_row8_col32 = col_idx % 32;

    size_t from_row_in_row8_col32 =
        4 * (to_row_in_row8_col32 % 2) + to_col_in_row8_col32 / 8;
    size_t from_col_in_row8_col32 = 16 * ((to_col_in_row8_col32 / 4) % 2) +
                                    4 * (to_row_in_row8_col32 / 2) +
                                    to_col_in_row8_col32 % 4;
    size_t from_linear_idx_in_row8_col32 =
        from_row_in_row8_col32 * 32 + from_col_in_row8_col32;

    from_linear_idx = src_ld * (col_idx / 32) + (row_idx / 8) * (32 * 8) +
                      from_linear_idx_in_row8_col32;
    break;
  }
  case COMBINE(order_t::col, order_t::col32_2r_4r4): {
    from_linear_idx = src_ld * col_idx + row_idx;

    size_t from_row_in_row32_col32 = row_idx % 32;
    size_t from_col_in_row32_col32 = col_idx % 32;

    size_t to_row_in_row32_col32 = 8 * ((from_row_in_row32_col32 % 8) / 2) +
                                   (from_row_in_row32_col32 / 8) * 2 +
                                   from_row_in_row32_col32 % 2;
    size_t to_col_in_row32_col32 = from_col_in_row32_col32;
    size_t to_linear_idx_in_row32_col32 =
        to_row_in_row32_col32 * 32 + to_col_in_row32_col32;

    to_linear_idx = dst_ld * (col_idx / 32) + (row_idx / 32) * (32 * 32) +
                    to_linear_idx_in_row32_col32;
    break;
  }
  case COMBINE(order_t::col32_2r_4r4, order_t::col): {
    to_linear_idx = dst_ld * col_idx + row_idx;

    size_t to_row_in_row32_col32 = row_idx % 32;
    size_t to_col_in_row32_col32 = col_idx % 32;

    size_t from_row_in_row32_col32 = 8 * ((to_row_in_row32_col32 % 8) / 2) +
                                     (to_row_in_row32_col32 / 8) * 2 +
                                     to_row_in_row32_col32 % 2;
    size_t from_col_in_row32_col32 = to_col_in_row32_col32;
    size_t from_linear_idx_in_row32_col32 =
        from_row_in_row32_col32 * 32 + from_col_in_row32_col32;

    from_linear_idx = src_ld * (col_idx / 32) + (row_idx / 32) * (32 * 32) +
                      from_linear_idx_in_row32_col32;
    break;
  }
  }
#undef COMBINE
  return std::make_tuple(from_linear_idx, to_linear_idx);
}

template <template <typename> typename functor_t, typename... args_t>
inline auto type_dispatch(library_data_t type, args_t &&...args) {
  switch (type) {
  case library_data_t::real_float:
    return functor_t<float>()(std::forward<args_t>(args)...);
  case library_data_t::real_int8:
    return functor_t<std::int8_t>()(std::forward<args_t>(args)...);
  case library_data_t::real_int32:
    return functor_t<int>()(std::forward<args_t>(args)...);
  default:
    throw std::runtime_error("the data type is unsupported");
  }
}

template <typename T> struct matrix_transform_impl {
  sycl::event operator()(::dpct::cs::queue_ptr q_ptr, size_t rows, size_t cols,
                         size_t a_ld, order_t a_order, const void *a,
                         size_t c_ld, order_t c_order, void *c,
                         std::vector<sycl::event> deps) {
    if ((a_order != order_t::col && c_order != order_t::col) ||
        (a_order == order_t::col && c_order == order_t::col)) {
      throw std::runtime_error("dpct::blas_gemm::experimental::detail::matrix_"
                               "transform_impl() does not "
                               "support the order combination currently.");
    }

    return q_ptr->submit([&](sycl::handler &cgh) {
      cgh.depends_on(deps);
#ifdef DPCT_USM_LEVEL_NONE
      access_wrapper<const T *> a_acc(a, cgh);
      access_wrapper<T *> c_acc(c, cgh);
#endif
      cgh.parallel_for<
          ::dpct::cs::kernel_name<class matrix_transform_col_to_row, T>>(
          sycl::range<2>(a_ld, cols), [=](sycl::id<2> index) {
#ifdef DPCT_USM_LEVEL_NONE
            auto a_data = a_acc.get_raw_pointer();
            auto c_data = c_acc.get_raw_pointer();
#else
            auto a_data = (const T *)a;
            auto c_data = (T *)c;
#endif
            size_t row_idx = index.get(0);
            size_t col_idx = index.get(1);
            if (row_idx < rows) {
              size_t from_linear_idx, to_linear_idx;
              std::tie(from_linear_idx, to_linear_idx) = get_linear_idx_map(
                  rows, cols, a_ld, a_order, c_ld, c_order, row_idx, col_idx);
              c_data[to_linear_idx] = a_data[from_linear_idx];
            }
          });
    });
  }
};

#ifdef DPCT_USM_LEVEL_NONE
template <typename T> struct scale_d_impl {
  sycl::event operator()(const void *d_scale_ptr, void *d, size_t ld,
                         size_t rows, size_t cols, ::dpct::cs::queue_ptr q_ptr,
                         dpct::library_data_t scale_type,
                         std::vector<sycl::event> deps) {
    if (scale_type == dpct::library_data_t::real_float)
      return q_ptr->submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        access_wrapper<const float *> d_scale_acc(
            static_cast<const float *>(d_scale_ptr), cgh);
        access_wrapper<T *> d_acc(d, cgh);
        cgh.parallel_for<::dpct::cs::kernel_name<class scale_d_float, T>>(
            sycl::range<2>(ld, cols), [=](sycl::id<2> idx) {
              float scale_factor = d_scale_acc.get_raw_pointer()[0];
              auto d_data = d_acc.get_raw_pointer();
              size_t row_idx = idx.get(0);
              size_t col_idx = idx.get(1);
              if (row_idx < rows) {
                size_t linear_idx = row_idx + ld * col_idx;
                d_data[linear_idx] = d_data[linear_idx] * scale_factor;
              }
            });
      });
    else {
      // int type
      return q_ptr->submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
        access_wrapper<const int *> d_scale_acc(
            static_cast<const int *>(d_scale_ptr), cgh);
        access_wrapper<T *> d_acc(d, cgh);
        cgh.parallel_for<::dpct::cs::kernel_name<class scale_d_int, T>>(
            sycl::range<2>(ld, cols), [=](sycl::id<2> idx) {
              float scale_factor =
                  static_cast<float>(d_scale_acc.get_raw_pointer()[0]);
              auto d_data = d_acc.get_raw_pointer();
              size_t row_idx = idx.get(0);
              size_t col_idx = idx.get(1);
              if (row_idx < rows) {
                size_t linear_idx = row_idx + ld * col_idx;
                d_data[linear_idx] = d_data[linear_idx] * scale_factor;
              }
            });
      });
    }
  }
};

template <typename T> struct set_buffer_impl {
  void operator()(::dnnl::memory *dnnl_memory, const void *ptr) {
    auto buf = get_buffer<T>(ptr);
    ::dnnl::sycl_interop::set_buffer(*dnnl_memory, buf);
  }
};
#else
template <typename T> struct scale_d_impl {
  sycl::event operator()(const void *d_scale_ptr, void *d, size_t ld,
                         size_t rows, size_t cols, ::dpct::cs::queue_ptr q_ptr,
                         dpct::library_data_t scale_type,
                         std::vector<sycl::event> deps) {
    return q_ptr->submit([&](sycl::handler &cgh) {
      cgh.depends_on(deps);
      cgh.parallel_for<::dpct::cs::kernel_name<class scale_d, T>>(
          sycl::range<2>(ld, cols), [=](sycl::id<2> idx) {
            float scale_factor;
            if (scale_type == dpct::library_data_t::real_float)
              scale_factor = static_cast<const float *>(d_scale_ptr)[0];
            else {
              // int type
              scale_factor =
                  static_cast<float>(static_cast<const int *>(d_scale_ptr)[0]);
            }
            auto d_data = (T *)d;
            size_t row_idx = idx.get(0);
            size_t col_idx = idx.get(1);
            if (row_idx < rows) {
              size_t linear_idx = row_idx + ld * col_idx;
              d_data[linear_idx] = d_data[linear_idx] * scale_factor;
            }
          });
    });
  }
};
#endif

template <typename T> struct get_beta_value_impl {
  int operator()(const void *beta, ::dpct::cs::queue_ptr q_ptr) {
    T beta_host;
    ::dpct::cs::memcpy(*q_ptr, &beta_host, beta, sizeof(T),
                       ::dpct::cs::memcpy_direction::automatic)
        .wait();
    T zero = T(0);
    T one = T(1);
    if (beta_host == zero)
      return 0;
    else if (beta_host == one)
      return 1;
    return -1;
  }
};

template <typename T> struct abs_max_op {
  auto operator()(const T &lhs, const T &rhs) {
    T abs_lhs = lhs >= 0 ? lhs : -lhs;
    T abs_rhs = rhs >= 0 ? rhs : -rhs;
    return (abs_lhs < abs_rhs) ? abs_rhs : abs_lhs;
  }
};

template <typename T> struct absmax_impl {
  sycl::event operator()(void *absmax_ptr, const void *new_d, size_t ld,
                         size_t rows, size_t cols, ::dpct::cs::queue_ptr q_ptr,
                         std::vector<sycl::event> deps) {
    return q_ptr->submit([&](sycl::handler &cgh) {
#ifdef DPCT_USM_LEVEL_NONE
      auto absmax_reduction = sycl::reduction(
          get_buffer<T>(absmax_ptr), cgh, T(0), abs_max_op<T>(),
          {sycl::property::reduction::initialize_to_identity()});
      access_wrapper<const T *> new_d_acc(new_d, cgh);
#else
      auto absmax_reduction = sycl::reduction(
          (T *)(absmax_ptr), T(0), abs_max_op<T>(),
          {sycl::property::reduction::initialize_to_identity()});
#endif
      cgh.depends_on(deps);
      cgh.parallel_for<::dpct::cs::kernel_name<class absmax_reduction, T>>(
          sycl::range<2>(ld, cols), absmax_reduction,
          [=](sycl::id<2> idx, auto &absmax) {
#ifdef DPCT_USM_LEVEL_NONE
            auto new_d_data = new_d_acc.get_raw_pointer();
#else
            auto new_d_data = (const T *)new_d;
#endif
            size_t row_idx = idx.get(0);
            size_t col_idx = idx.get(1);
            if (row_idx < rows) {
              size_t linear_idx = row_idx + ld * col_idx;
              absmax.combine(new_d_data[linear_idx]);
            }
          });
    });
  }
};
} // namespace dpct::blas_gemm::experimental::detail

#endif // __DPCT_BLAS_GEMM_UTILS_DETAIL_HPP__
