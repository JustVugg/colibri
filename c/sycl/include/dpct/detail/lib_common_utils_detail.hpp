//==---- lib_common_utils_detail.hpp --------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LIB_COMMON_UTILS_DETAIL_HPP__
#define __DPCT_LIB_COMMON_UTILS_DETAIL_HPP__

namespace dpct::detail {
template <typename T> struct lib_data_traits { using type = T; };
template <typename T> struct lib_data_traits<sycl::vec<T, 2>> {
  using type = std::complex<T>;
};
template <class T> using lib_data_traits_t = typename lib_data_traits<T>::type;

template <typename T> inline auto get_memory(const void *x) {
  T *new_x = reinterpret_cast<T *>(const_cast<void *>(x));
#ifdef DPCT_USM_LEVEL_NONE
  return dpct::get_buffer<std::remove_cv_t<T>>(new_x);
#else
  return new_x;
#endif
}

template <typename T>
inline typename ::dpct::detail::lib_data_traits_t<T> get_value(const T *s,
                                                               sycl::queue &q) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  Ty s_h;
  if (::dpct::cs::detail::get_pointer_attribute(q, s) ==
      ::dpct::cs::detail::pointer_access_attribute::device_only)
    ::dpct::cs::memcpy(q, (void *)&s_h, (void *)s, sizeof(T),
                       ::dpct::cs::memcpy_direction::device_to_host)
        .wait();
  else
    s_h = *reinterpret_cast<const Ty *>(s);
  return s_h;
}

template <typename ArgT>
inline constexpr std::uint64_t get_type_combination_id(ArgT Val) {
  static_assert((unsigned char)library_data_t::library_data_t_size <=
                    std::numeric_limits<unsigned char>::max() &&
                "library_data_t size exceeds limit.");
  static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
  return (std::uint64_t)Val;
}

template <typename FirstT, typename... RestT>
inline constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                       RestT... RestVal) {
  static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                    std::numeric_limits<unsigned char>::max() &&
                "library_data_t size exceeds limit.");
  static_assert(sizeof...(RestT) <= 8 && "Too many parameters");
  static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
  return get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
}

inline constexpr std::size_t library_data_size[] = {
    8 * sizeof(float),                    // real_float
    8 * sizeof(std::complex<float>),      // complex_float
    8 * sizeof(double),                   // real_double
    8 * sizeof(std::complex<double>),     // complex_double
    8 * sizeof(sycl::half),               // real_half
    8 * sizeof(std::complex<sycl::half>), // complex_half
    16,                                   // real_bfloat16
    16 * 2,                               // complex_bfloat16
    4,                                    // real_int4
    4 * 2,                                // complex_int4
    4,                                    // real_uint4
    4 * 2,                                // complex_uint4
    8,                                    // real_int8
    8 * 2,                                // complex_int8
    8,                                    // real_uint8
    8 * 2,                                // complex_uint8
    16,                                   // real_int16
    16 * 2,                               // complex_int16
    16,                                   // real_uint16
    16 * 2,                               // complex_uint16
    32,                                   // real_int32
    32 * 2,                               // complex_int32
    32,                                   // real_uint32
    32 * 2,                               // complex_uint32
    64,                                   // real_int64
    64 * 2,                               // complex_int64
    64,                                   // real_uint64
    64 * 2,                               // complex_uint64
    8,                                    // real_int8_4
    8,                                    // real_int8_32
    8                                     // real_uint8_4
};

template <class func_t, typename... args_t>
std::invoke_result_t<func_t, args_t...>
catch_batch_error(int *has_execption, const std::string &api_names,
                  sycl::queue exec_queue, int *info, int *dev_info,
                  int batch_size, func_t &&f, args_t &&...args) {
  try {
    return f(std::forward<args_t>(args)...);
  } catch (oneapi::mkl::lapack::batch_error const &be) {
    std::cerr << "Unexpected exception caught during call to API(s): "
              << api_names << std::endl
              << "reason: " << be.what() << std::endl
              << "number: " << be.info() << std::endl;
    if (be.info() < 0 && info)
      *info = be.info();
    int i = 0;
    auto &ids = be.ids();
    std::vector<int> info_vec(batch_size, 0);
    for (auto const &e : be.exceptions()) {
      try {
        std::rethrow_exception(e);
      } catch (oneapi::mkl::lapack::exception &e) {
        std::cerr << "Exception in problem: " << ids[i] << std::endl
                  << "reason: " << e.what() << std::endl
                  << "info: " << e.info() << std::endl;
        info_vec[ids[i]] = e.info();
        i++;
      }
    }
    if (dev_info)
      exec_queue.memcpy(dev_info, info_vec.data(), batch_size * sizeof(int))
          .wait();
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    if (dev_info)
      exec_queue.memset(dev_info, 0, batch_size * sizeof(int)).wait();
  }
  if (has_execption)
    *has_execption = 1;
  return {};
}

template <typename ret_t, typename... args_t>
ret_t catch_batch_error_f(int *has_execption, const std::string &api_names,
                          sycl::queue exec_queue, int *info, int *dev_info,
                          int batch_size, ret_t (*f)(args_t...),
                          args_t &&...args) {
  return catch_batch_error(has_execption, api_names, exec_queue, info, dev_info,
                           batch_size, f, args...);
}
} // namespace dpct::detail

#endif // __DPCT_LIB_COMMON_UTILS_DETAIL_HPP__
