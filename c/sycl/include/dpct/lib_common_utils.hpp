//==---- lib_common_utils.hpp ---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LIB_COMMON_UTILS_HPP__
#define __DPCT_LIB_COMMON_UTILS_HPP__

#include "compat_service.hpp"

#if defined(__has_include) && __has_include(<oneapi/math.hpp>)
#include <oneapi/math.hpp>
#include <oneapi/mkl/namespace_alias.hpp>
#elif defined(__has_include) && __has_include(<oneapi/mkl.hpp>)
#include <oneapi/mkl.hpp>
#elif defined(__has_include)
#error "SYCLomatic runtime requires oneMath/oneMKL support"
#else
#error "SYCLomatic runtime requires __has_include support"
#endif

namespace dpct {

enum class version_field : int { major, minor, update, patch };

/// Returns the requested field of Intel(R) oneAPI Math Kernel Library version.
/// \param field The version information field (major, minor, update or patch).
/// \param result The result value.
inline void mkl_get_version(version_field field, int *result) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  MKLVersion version;
  mkl_get_version(&version);
  if (version_field::major == field) {
    *result = version.MajorVersion;
  } else if (version_field::minor == field) {
    *result = version.MinorVersion;
  } else if (version_field::update == field) {
    *result = version.UpdateVersion;
  } else if (version_field::patch == field) {
    *result = 0;
  } else {
    throw std::runtime_error("unknown field");
  }
#endif
}

enum class library_data_t : unsigned char {
  real_float = 0,
  complex_float,
  real_double,
  complex_double,
  real_half,
  complex_half,
  real_bfloat16,
  complex_bfloat16,
  real_int4,
  complex_int4,
  real_uint4,
  complex_uint4,
  real_int8,
  complex_int8,
  real_uint8,
  complex_uint8,
  real_int16,
  complex_int16,
  real_uint16,
  complex_uint16,
  real_int32,
  complex_int32,
  real_uint32,
  complex_uint32,
  real_int64,
  complex_int64,
  real_uint64,
  complex_uint64,
  real_int8_4,
  real_int8_32,
  real_uint8_4,
  real_f8_e4m3,
  real_f8_e5m2,
  library_data_t_size
};

enum class compute_type : int {
  f16,
  f16_standard,
  f32,
  f32_standard,
  f32_fast_bf16,
  f32_fast_tf32,
  f64,
  f64_standard,
  i32,
  i32_standard,
};

#ifdef DPCT_USM_LEVEL_NONE
/// Cast a "rvalue reference to a temporary object" to an "lvalue reference to
/// that temporary object".
/// CAUTION:
/// The returned lvalue reference is available only before the last step in
/// evaluating the full-expression that contains this function call.
/// \param [in] temporary_object The rvalue reference to a temporary object.
/// \returns The lvalue reference to that temporary object.
template <typename T>
inline typename std::enable_if_t<std::is_rvalue_reference_v<T &&>, T &>
rvalue_ref_to_lvalue_ref(T &&temporary_object) {
  return temporary_object;
}
#endif
} // namespace dpct

#include "detail/lib_common_utils_detail.hpp"

#endif // __DPCT_LIB_COMMON_UTILS_HPP__
