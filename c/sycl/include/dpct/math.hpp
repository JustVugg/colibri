//==---- math.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MATH_HPP__
#define __DPCT_MATH_HPP__

#include <climits>
#include <limits>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "detail/math_detail.hpp"

namespace dpct {

/// Compute fast_length for variable-length array
/// \param [in] a The array
/// \param [in] len Length of the array
/// \returns The computed fast_length
inline float fast_length(const float *a, int len) {
  switch (len) {
  case 1:
    return a[0];
  case 2:
    return sycl::fast_length(sycl::float2(a[0], a[1]));
  case 3:
    return sycl::fast_length(sycl::float3(a[0], a[1], a[2]));
  case 4:
    return sycl::fast_length(sycl::float4(a[0], a[1], a[2], a[3]));
  case 0:
    return 0;
  default:
    float f = 0;
    for (int i = 0; i < len; ++i)
      f += a[i] * a[i];
    return sycl::sqrt(f);
  }
}

/// Calculate the square root of the input array.
/// \param [in] a The array pointer
/// \param [in] len Length of the array
/// \returns The square root
template <typename T> inline T length(const T *a, const int len) {
  switch (len) {
  case 1:
    return a[0];
  case 2:
    return sycl::length(sycl::vec<T, 2>(a[0], a[1]));
  case 3:
    return sycl::length(sycl::vec<T, 3>(a[0], a[1], a[2]));
  case 4:
    return sycl::length(sycl::vec<T, 4>(a[0], a[1], a[2], a[3]));
  default:
    T ret = 0;
    for (int i = 0; i < len; ++i)
      ret += a[i] * a[i];
    return sycl::sqrt(ret);
  }
}

/// Returns min(max(val, min_val), max_val)
/// \param [in] val The input value
/// \param [in] min_val The minimum value
/// \param [in] max_val The maximum value
/// \returns the value between min_val and max_val
template <typename T> inline T clamp(T val, T min_val, T max_val) {
  return sycl::clamp(val, min_val, max_val);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline sycl::ext::oneapi::bfloat16 clamp(sycl::ext::oneapi::bfloat16 val,
                                         sycl::ext::oneapi::bfloat16 min_val,
                                         sycl::ext::oneapi::bfloat16 max_val) {
  if (val < min_val)
    return min_val;
  if (val > max_val)
    return max_val;
  return val;
}
template <>
inline sycl::vec<sycl::ext::oneapi::bfloat16, 2>
clamp(sycl::vec<sycl::ext::oneapi::bfloat16, 2> val,
      sycl::vec<sycl::ext::oneapi::bfloat16, 2> min_val,
      sycl::vec<sycl::ext::oneapi::bfloat16, 2> max_val) {
  return {clamp(val[0], min_val[0], max_val[0]),
          clamp(val[1], min_val[1], max_val[1])};
}
#endif
template <typename T>
inline sycl::marray<T, 2> clamp(sycl::marray<T, 2> val,
                                sycl::marray<T, 2> min_val,
                                sycl::marray<T, 2> max_val) {
  return {clamp(val[0], min_val[0], max_val[0]),
          clamp(val[1], min_val[1], max_val[1])};
}

/// Performs comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, T, T>, bool>, bool>
compare(const T a, const T b, const BinaryOperation binary_op) {
  return binary_op(a, b);
}
template <typename T>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<std::not_equal_to<>, T, T>, bool>, bool>
compare(const T a, const T b, const std::not_equal_to<> binary_op) {
  return !detail::isnan(a) && !detail::isnan(b) && binary_op(a, b);
}

/// Performs unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, T, T>, bool>, bool>
unordered_compare(const T a, const T b, const BinaryOperation binary_op) {
  return detail::isnan(a) || detail::isnan(b) || binary_op(a, b);
}

/// Performs 2 element comparison and return true if both results are true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, bool>
compare_both(const T a, const T b, const BinaryOperation binary_op) {
  return compare(a[0], b[0], binary_op) && compare(a[1], b[1], binary_op);
}

/// Performs 2 element unordered comparison and return true if both results are
/// true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, bool>
unordered_compare_both(const T a, const T b, const BinaryOperation binary_op) {
  return unordered_compare(a[0], b[0], binary_op) &&
         unordered_compare(a[1], b[1], binary_op);
}

/// Performs 2 element comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, T>
compare(const T a, const T b, const BinaryOperation binary_op) {
  return {compare(a[0], b[0], binary_op), compare(a[1], b[1], binary_op)};
}

/// Performs 2 elements comparison, compare result of each element is 0 (false)
/// or 0xffff (true), returns an unsigned int by composing compare result of two
/// elements.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline unsigned compare_mask(const sycl::vec<T, 2> a, const sycl::vec<T, 2> b,
                             const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-compare(a[0], b[0], binary_op),
                             -compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}
template <typename T, class BinaryOperation>
inline unsigned compare_mask(const sycl::marray<T, 2> a,
                             const sycl::marray<T, 2> b,
                             const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-compare(a[0], b[0], binary_op),
                             -compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}

/// Performs 2 element unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, T>
unordered_compare(const T a, const T b, const BinaryOperation binary_op) {
  return {unordered_compare(a[0], b[0], binary_op),
          unordered_compare(a[1], b[1], binary_op)};
}

/// Performs 2 elements unordered comparison, compare result of each element is
/// 0 (false) or 0xffff (true), returns an unsigned int by composing compare
/// result of two elements.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline unsigned unordered_compare_mask(const sycl::vec<T, 2> a,
                                       const sycl::vec<T, 2> b,
                                       const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-unordered_compare(a[0], b[0], binary_op),
                             -unordered_compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}
template <typename T, class BinaryOperation>
inline unsigned unordered_compare_mask(const sycl::marray<T, 2> a,
                                       const sycl::marray<T, 2> b,
                                       const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-unordered_compare(a[0], b[0], binary_op),
                             -unordered_compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}

/// Bitfield-extract.
///
/// \tparam T The type of \param source value, must be an integer.
/// \param source The source value to extracting.
/// \param bit_start The position to start extracting.
/// \param num_bits The number of bits to extracting.
template <typename T>
inline std::enable_if_t<std::is_unsigned_v<T>, T>
bfe(const T source, const uint32_t bit_start, const uint32_t num_bits) {
  const T mask = (T{1} << num_bits) - 1;
  return (source >> bit_start) & mask;
}

/// Bitfield-extract with boundary checking.
///
/// Extract bit field from \param source and return the zero or sign-extended
/// result. Source \param bit_start gives the bit field starting bit position,
/// and source \param num_bits gives the bit field length in bits.
///
/// The result is padded with the sign bit of the extracted field. If the start
/// position is beyond the msb of the input, the result was filled with the
/// replicated sign bit of the extracted field.
///
/// \tparam T The type of \param source value, must be an integer.
/// \param source The source value to extracting.
/// \param bit_start The position to start extracting.
/// \param num_bits The number of bits to extracting.
template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T>
bfe_safe(const T source, const uint32_t bit_start, const uint32_t num_bits) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                std::is_same_v<T, int32_t>) {
    int32_t res{};
    asm volatile("bfe.s32 %0, %1, %2, %3;"
                 : "=r"(res)
                 : "r"((int32_t)source), "r"(bit_start), "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, uint8_t> ||
                       std::is_same_v<T, uint16_t> ||
                       std::is_same_v<T, uint32_t>) {
    uint32_t res{};
    asm volatile("bfe.u32 %0, %1, %2, %3;"
                 : "=r"(res)
                 : "r"((uint32_t)source), "r"(bit_start), "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    T res{};
    asm volatile("bfe.s64 %0, %1, %2, %3;"
                 : "=l"(res)
                 : "l"(source), "r"(bit_start), "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    T res{};
    asm volatile("bfe.u64 %0, %1, %2, %3;"
                 : "=l"(res)
                 : "l"(source), "r"(bit_start), "r"(num_bits));
    return res;
  }
#endif
  const uint32_t bit_width = CHAR_BIT * sizeof(T);
  const uint32_t pos = (std::min)(bit_start, bit_width);
  const uint32_t len = (std::min)(pos + num_bits, bit_width) - pos;
  if constexpr (std::is_signed_v<T>) {
    const T mask = (T{1} << len) - 1;

    // Find the sign-bit, the result is padded with the sign bit of the
    // extracted field.
    //
    // sign_bit = len == 0 ? 0 : source[min(pos + len - 1, bit_width - 1)]
    const uint32_t sign_bit_pos = (std::min)(pos + len - 1, bit_width - 1);
    const T sign_bit = len != 0 && ((source >> sign_bit_pos) & 1);
    const T sign_bit_padding = (-sign_bit & ~mask);
    return ((source >> pos) & mask) | sign_bit_padding;
  } else {
    return dpct::bfe(source, pos, len);
  }
}

/// Bitfield-insert.
///
/// \tparam T The type of \param x and \param y , must be an unsigned integer.
/// \param x The source of the bitfield.
/// \param y The source where bitfield is inserted.
/// \param bit_start The position to start insertion.
/// \param num_bits The number of bits to insertion.
template <typename T>
inline std::enable_if_t<std::is_unsigned_v<T>, T>
bfi(const T x, const T y, const uint32_t bit_start, const uint32_t num_bits) {
  constexpr unsigned bit_width = CHAR_BIT * sizeof(T);

  // if bit_start > bit_width || len == 0, should return y.
  const uint32_t ignore_bfi = bit_start > bit_width || num_bits == 0;
  T extract_bitfield_mask = (~(T{0}) >> (bit_width - num_bits)) << bit_start;
  T clean_bitfield_mask = ~extract_bitfield_mask;
  return (y & (-ignore_bfi | clean_bitfield_mask)) |
         (~-ignore_bfi & ((x << bit_start) & extract_bitfield_mask));
}

/// Bitfield-insert with boundary checking.
///
/// Align and insert a bit field from \param x into \param y . Source \param
/// bit_start gives the starting bit position for the insertion, and source
/// \param num_bits gives the bit field length in bits.
///
/// \tparam T The type of \param x and \param y , must be an unsigned integer.
/// \param x The source of the bitfield.
/// \param y The source where bitfield is inserted.
/// \param bit_start The position to start insertion.
/// \param num_bits The number of bits to insertion.
template <typename T>
inline std::enable_if_t<std::is_unsigned_v<T>, T>
bfi_safe(const T x, const T y, const uint32_t bit_start,
         const uint32_t num_bits) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                std::is_same_v<T, uint32_t>) {
    uint32_t res{};
    asm volatile("bfi.b32 %0, %1, %2, %3, %4;"
                 : "=r"(res)
                 : "r"((uint32_t)x), "r"((uint32_t)y), "r"(bit_start),
                   "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    uint64_t res{};
    asm volatile("bfi.b64 %0, %1, %2, %3, %4;"
                 : "=l"(res)
                 : "l"(x), "l"(y), "r"(bit_start), "r"(num_bits));
    return res;
  }
#endif
  constexpr unsigned bit_width = CHAR_BIT * sizeof(T);
  const uint32_t pos = (std::min)(bit_start, bit_width);
  const uint32_t len = (std::min)(pos + num_bits, bit_width) - pos;
  return dpct::bfi(x, y, pos, len);
}

/// Determine whether 2 element value is NaN.
/// \param [in] a The input value
/// \returns the comparison result
template <typename T>
inline std::enable_if_t<T::size() == 2, T> isnan(const T a) {
  return {detail::isnan(a[0]), detail::isnan(a[1])};
}

/// Emulated function for __funnelshift_l
inline unsigned int funnelshift_l(unsigned int low, unsigned int high,
                                  unsigned int shift) {
  return (sycl::upsample(high, low) << (shift & 31U)) >> 32;
}

/// Emulated function for __funnelshift_lc
inline unsigned int funnelshift_lc(unsigned int low, unsigned int high,
                                   unsigned int shift) {
  return (sycl::upsample(high, low) << sycl::min(shift, 32U)) >> 32;
}

/// Emulated function for __funnelshift_r
inline unsigned int funnelshift_r(unsigned int low, unsigned int high,
                                  unsigned int shift) {
  return (sycl::upsample(high, low) >> (shift & 31U)) & 0xFFFFFFFF;
}

/// Emulated function for __funnelshift_rc
inline unsigned int funnelshift_rc(unsigned int low, unsigned int high,
                                   unsigned int shift) {
  return (sycl::upsample(high, low) >> sycl::min(shift, 32U)) & 0xFFFFFFFF;
}

/// cbrt function wrapper.
template <typename T> inline T cbrt(T val) { return sycl::cbrt((T)val); }

template <typename T1, typename T2>
std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                 std::common_type_t<T1, T2>>
min(T1 a, T2 b) {
  using common_t = std::common_type_t<T1, T2>;
  return sycl::min(static_cast<common_t>(a), static_cast<common_t>(b));
}
template <typename T1, typename T2>
std::enable_if_t<std::is_floating_point_v<T1> && std::is_floating_point_v<T2>,
                 std::common_type_t<T1, T2>>
min(T1 a, T2 b) {
  using common_t = std::common_type_t<T1, T2>;
  return sycl::fmin(static_cast<common_t>(a), static_cast<common_t>(b));
}
template <typename T1, typename T2>
std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                 std::common_type_t<T1, T2>>
max(T1 a, T2 b) {
  using common_t = std::common_type_t<T1, T2>;
  return sycl::max(static_cast<common_t>(a), static_cast<common_t>(b));
}
template <typename T1, typename T2>
std::enable_if_t<std::is_floating_point_v<T1> && std::is_floating_point_v<T2>,
                 std::common_type_t<T1, T2>>
max(T1 a, T2 b) {
  using common_t = std::common_type_t<T1, T2>;
  return sycl::fmax(static_cast<common_t>(a), static_cast<common_t>(b));
}

// pow functions overload.
inline float pow(const float a, const int b) { return sycl::pown(a, b); }
inline double pow(const double a, const int b) { return sycl::pown(a, b); }
inline float pow(const float a, const float b) { return sycl::pow(a, b); }
inline double pow(const double a, const double b) { return sycl::pow(a, b); }
template <typename T, typename U>
inline typename std::enable_if_t<std::is_floating_point_v<T>, T>
pow(const T a, const U b) {
  return sycl::pow(a, static_cast<T>(b));
}
template <typename T, typename U>
inline typename std::enable_if_t<!std::is_floating_point_v<T>, double>
pow(const T a, const U b) {
  return sycl::pow(static_cast<double>(a), static_cast<double>(b));
}

/// Performs relu saturation.
/// \param [in] a The input value
/// \returns the relu saturation result
template <typename T> inline T relu(T a) {
  T zero{};
  if constexpr (detail::is_floating_point<T>)
    return !detail::isnan(a) && a < zero ? zero : a;
  else
    return a < zero ? zero : a;
}
template <typename T, int N>
inline sycl::vec<T, N> relu(const sycl::vec<T, N> a) {
  sycl::vec<T, N> ret;
  for (int i = 0; i < N; ++i)
    ret[i] = relu(a[i]);
  return ret;
}
template <class T> inline sycl::marray<T, 2> relu(const sycl::marray<T, 2> a) {
  return {relu(a[0]), relu(a[1])};
}

/// Performs complex number multiply addition.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns the operation result
template <typename T>
inline sycl::vec<T, 2> complex_mul_add(const sycl::vec<T, 2> a,
                                       const sycl::vec<T, 2> b,
                                       const sycl::vec<T, 2> c) {
  return sycl::vec<T, 2>{a[0] * b[0] - a[1] * b[1] + c[0],
                         a[0] * b[1] + a[1] * b[0] + c[1]};
}
template <typename T>
inline sycl::marray<T, 2> complex_mul_add(const sycl::marray<T, 2> a,
                                          const sycl::marray<T, 2> b,
                                          const sycl::marray<T, 2> c) {
  return sycl::marray<T, 2>{a[0] * b[0] - a[1] * b[1] + c[0],
                            a[0] * b[1] + a[1] * b[0] + c[1]};
}

/// Performs 2 elements comparison and returns the bigger one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the bigger value
template <typename T> inline T fmax_nan(const T a, const T b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmax(a, b);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline sycl::ext::oneapi::bfloat16
fmax_nan(const sycl::ext::oneapi::bfloat16 a,
         const sycl::ext::oneapi::bfloat16 b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmax(float(a), float(b));
}
#endif
template <typename T>
inline sycl::vec<T, 2> fmax_nan(const sycl::vec<T, 2> a,
                                const sycl::vec<T, 2> b) {
  return {fmax_nan(a[0], b[0]), fmax_nan(a[1], b[1])};
}
template <typename T>
inline sycl::marray<T, 2> fmax_nan(const sycl::marray<T, 2> a,
                                   const sycl::marray<T, 2> b) {
  return {fmax_nan(a[0], b[0]), fmax_nan(a[1], b[1])};
}

/// Performs 2 elements comparison and returns the smaller one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the smaller value
template <typename T> inline T fmin_nan(const T a, const T b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmin(a, b);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline sycl::ext::oneapi::bfloat16
fmin_nan(const sycl::ext::oneapi::bfloat16 a,
         const sycl::ext::oneapi::bfloat16 b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmin(float(a), float(b));
}
#endif
template <typename T>
inline sycl::vec<T, 2> fmin_nan(const sycl::vec<T, 2> a,
                                const sycl::vec<T, 2> b) {
  return {fmin_nan(a[0], b[0]), fmin_nan(a[1], b[1])};
}
template <typename T>
inline sycl::marray<T, 2> fmin_nan(const sycl::marray<T, 2> a,
                                   const sycl::marray<T, 2> b) {
  return {fmin_nan(a[0], b[0]), fmin_nan(a[1], b[1])};
}

/// A sycl::abs wrapper functors.
struct abs {
  template <typename T> auto operator()(const T x) const {
    return sycl::abs(x);
  }
};

/// A sycl::abs_diff wrapper functors.
struct abs_diff {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::abs_diff(x, y);
  }
};

/// A sycl::add_sat wrapper functors.
struct add_sat {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::add_sat(x, y);
  }
};

/// A sycl::rhadd wrapper functors.
struct rhadd {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::rhadd(x, y);
  }
};

/// A sycl::hadd wrapper functors.
struct hadd {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::hadd(x, y);
  }
};

/// A sycl::max wrapper functors.
struct maximum {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::max(x, y);
  }
  template <typename T>
  auto operator()(const T x, const T y, bool *pred) const {
    return (x >= y) ? ((*pred = true), x) : ((*pred = false), y);
  }
};

/// A sycl::min wrapper functors.
struct minimum {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::min(x, y);
  }
  template <typename T>
  auto operator()(const T x, const T y, bool *pred) const {
    return (x <= y) ? ((*pred = true), x) : ((*pred = false), y);
  }
};

/// A sycl::sub_sat wrapper functors.
struct sub_sat {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::sub_sat(x, y);
  }
};

/// Compute vectorized binary operation value for two values, with each value
/// treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] BinaryOperation The binary operation class
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op The operation to do with the two values
/// \param [in] need_relu Whether the result need relu saturation
/// \returns The vectorized binary operation value of the two values
template <typename VecT, class BinaryOperation>
inline unsigned vectorized_binary(unsigned a, unsigned b,
                                  const BinaryOperation binary_op,
                                  bool need_relu = false) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.as<VecT>();
  auto v3 = v1.as<VecT>();
  auto v4 =
      detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
  if (need_relu)
    v4 = relu(v4);
  v0 = v4.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Compute vectorized binary operation value with pred for two values, with
/// each value treated as a 2 \p T type elements vector type.
///
/// \tparam [in] VecT The type of the vector
/// \tparam [in] BinaryOperation The binary operation class
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op The operation with pred to do with the two values
/// \param [out] pred_hi The pred pointer that pass into high halfword operation
/// \param [out] pred_lo The pred pointer that pass into low halfword operation
/// \returns The vectorized binary operation value of the two values
template <typename VecT, typename BinaryOperation>
inline unsigned vectorized_binary_with_pred(unsigned a, unsigned b,
                                            const BinaryOperation binary_op,
                                            bool *pred_hi, bool *pred_lo) {
  auto v1 = sycl::vec<unsigned, 1>(a).as<VecT>();
  auto v2 = sycl::vec<unsigned, 1>(b).as<VecT>();
  VecT ret;
  ret[0] = binary_op(v1[0], v2[0], pred_lo);
  ret[1] = binary_op(v1[1], v2[1], pred_hi);
  return ret.template as<sycl::vec<unsigned, 1>>();
}

/// Compute vectorized isgreater for two values, with each value treated as a
/// vector type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <typename S, typename T> inline T vectorized_isgreater(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = v2 > v3;
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized max for two values, with each value treated as a vector
/// type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized max of the two values
template <typename S, typename T> inline T vectorized_max(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::max(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized min for two values, with each value treated as a vector
/// type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized min of the two values
template <typename S, typename T> inline T vectorized_min(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::min(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized unary operation for a value, with the value treated as a
/// vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] UnaryOperation The unary operation class
/// \param [in] a The input value
/// \returns The vectorized unary operation value of the input value
template <typename VecT, class UnaryOperation>
inline unsigned vectorized_unary(unsigned a, const UnaryOperation unary_op) {
  sycl::vec<unsigned, 1> v0{a};
  auto v1 = v0.as<VecT>();
  auto v2 = unary_op(v1);
  v0 = v2.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Compute vectorized absolute difference for two values without modulo
/// overflow, with each value treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized absolute difference of the two values
template <typename VecT>
inline unsigned vectorized_sum_abs_diff(unsigned a, unsigned b) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  // Need convert element type to wider signed type to avoid overflow.
  auto v2 = v0.as<VecT>().template convert<int>();
  auto v3 = v1.as<VecT>().template convert<int>();
  auto v4 = sycl::abs_diff(v2, v3);
  unsigned sum = 0;
  for (size_t i = 0; i < v4.size(); ++i) {
    sum += v4[i];
  }
  return sum;
}

/// Compute two vectorized binary operation value with pred for three values,
/// with each value treated as a 2 \p T type elements vector type.
///
/// \tparam [in] VecT The type of the vector
/// \tparam [in] BinaryOperation1 The first binary operation class
/// \tparam [in] BinaryOperation2 The second binary operation class
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] binary_op1 The first operation to do with the first two values
/// \param [in] binary_op2 The second operation to do with the third values
/// \param [in] need_relu Whether the result need relu saturation
/// \returns The two vectorized binary operation value of the three values
template <typename VecT, typename BinaryOperation1, typename BinaryOperation2>
inline unsigned vectorized_ternary(unsigned a, unsigned b, unsigned c,
                                   const BinaryOperation1 binary_op1,
                                   const BinaryOperation2 binary_op2,
                                   bool need_relu = false) {
  const auto v1 = sycl::vec<unsigned, 1>(a).as<VecT>();
  const auto v2 = sycl::vec<unsigned, 1>(b).as<VecT>();
  const auto v3 = sycl::vec<unsigned, 1>(c).as<VecT>();
  auto v4 =
      detail::vectorized_binary<VecT, BinaryOperation1>()(v1, v2, binary_op1);
  v4 = detail::vectorized_binary<VecT, BinaryOperation2>()(v4, v3, binary_op2);
  if (need_relu)
    v4 = relu(v4);
  return v4.template as<sycl::vec<unsigned, 1>>();
}

/// Two-way dot product-accumulate. Calculate and return interger_vector2(
/// \param a) dot product interger_vector2(low16_bit( \param b))  + \param c
///
/// \tparam [in] T1 The type of first value.
/// \tparam [in] T2 The type of second value.
/// \param [in] a The first value.
/// \param [in] b The second value.
/// \param [in] c The third value. It has type uint32_t if both T1 and T1 are
/// uint32_t else has type int32_t.
/// \return Two-way 16-bit to 8-bit dot product which is accumulated in 32-bit
/// result.
template <typename T1, typename T2, typename T3>
inline auto dp2a_lo(T1 a, T2 b, T3 c) {
  detail::dot_product_acc_t<T1, T2> res = c;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  res = __dp2a_lo(a, b, c);
#else
  auto va = ::dpct::detail::extract_and_sign_or_zero_extend2(a);
  auto vb = ::dpct::detail::extract_and_sign_or_zero_extend4(b);
  res += va[0] * vb[0];
  res += va[1] * vb[1];
#endif
  return res;
}

/// Two-way dot product-accumulate. Calculate and return interger_vector2(
/// \param a) dot product interger_vector2(high_16bit( \param b)) + \param c
///
/// \tparam [in] T1 The type of first value.
/// \tparam [in] T2 The type of second value.
/// \param [in] a The first value.
/// \param [in] b The second value.
/// \param [in] c The third value. It has type uint32_t if both T1 and T1 are
/// uint32_t else has type int32_t.
/// \return Two-way 16-bit to 8-bit dot product which is accumulated in 32-bit
/// result.
template <typename T1, typename T2, typename T3>
inline auto dp2a_hi(T1 a, T2 b, T3 c) {
  detail::dot_product_acc_t<T1, T2> res = c;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  res = __dp2a_hi(a, b, c);
#else
  auto va = ::dpct::detail::extract_and_sign_or_zero_extend2(a);
  auto vb = ::dpct::detail::extract_and_sign_or_zero_extend4(b);
  res += va[0] * vb[2];
  res += va[1] * vb[3];
#endif
  return res;
}

/// Four-way byte dot product-accumulate. Calculate and return interger_vector4(
/// \param a) dot product interger_vector4( \param b)  + \param c
///
/// \tparam [in] T1 The type of first value.
/// \tparam [in] T2 The type of second value.
/// \param [in] a The first value.
/// \param [in] b The second value.
/// \param [in] c The third value. It has type uint32_t if both T1 and T1 are
/// uint32_t else has type int32_t.
/// \return Four-way byte dot product which is accumulated in 32-bit result.
template <typename T1, typename T2, typename T3>
inline auto dp4a(T1 a, T2 b, T3 c) {
  detail::dot_product_acc_t<T1, T2> res = c;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  res = __dp4a(a, b, c);
#else
  auto va = ::dpct::detail::extract_and_sign_or_zero_extend4(a);
  auto vb = ::dpct::detail::extract_and_sign_or_zero_extend4(b);
  res += va[0] * vb[0];
  res += va[1] * vb[1];
  res += va[2] * vb[2];
  res += va[3] * vb[3];
#endif
  return res;
}

/// Extend \p a and \p b to 33 bit and add them.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, std::plus());
}

/// Extend Inputs to 33 bit, add \p a, \p b, then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend addition of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, std::plus(), second_op);
}

/// Extend \p a and \p b to 33 bit and add them with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, std::plus());
}

/// Extend Inputs to 33 bit, add \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend addition of \p a, \p b with saturation and \p second_op
/// with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, std::plus(), second_op);
}

/// Extend \p a and \p b to 33 bit and minus them.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, std::minus());
}

/// Extend Inputs to 33 bit, minus \p a, \p b, then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend subtraction of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, std::minus(), second_op);
}

/// Extend \p a and \p b to 33 bit and minus them with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, std::minus());
}

/// Extend Inputs to 33 bit, minus \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend subtraction of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, std::minus(), second_op);
}

/// Extend \p a and \p b to 33 bit and do abs_diff.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, abs_diff());
}

/// Extend Inputs to 33 bit, abs_diff \p a, \p b, then do \p second_op with \p
/// c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend abs_diff of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, abs_diff(), second_op);
}

/// Extend \p a and \p b to 33 bit and do abs_diff with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, abs_diff());
}

/// Extend Inputs to 33 bit, abs_diff \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend abs_diff of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff_sat(AT a, BT b, CT c,
                                         BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, abs_diff(), second_op);
}

/// Extend \p a and \p b to 33 bit and return smaller one.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The smaller one of the two extended values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, minimum());
}

/// Extend Inputs to 33 bit, find the smaller one in \p a, \p b, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The smaller one of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, minimum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return smaller one with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The smaller one of the two extended values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, minimum());
}

/// Extend Inputs to 33 bit, find the smaller one in \p a, \p b with saturation,
/// then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The smaller one of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, minimum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return bigger one.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The bigger one of the two extended values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, maximum());
}

/// Extend Inputs to 33 bit, find the bigger one in \p a, \p b, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The bigger one of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, maximum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return bigger one with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The bigger one of the two extended values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, maximum());
}

/// Extend Inputs to 33 bit, find the bigger one in \p a, \p b with saturation,
/// then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The bigger one of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, maximum(), second_op);
}

/// Extend \p a and \p b to 33 bit and compare input values using specified
/// comparison \p cmp .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values.
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_compare(AT a, BT b, BinaryOperation cmp) {
  return detail::extend_binary<unsigned, false>(a, b, cmp);
}

/// Extend Inputs to 33 bit, and compare input values using specified comparison
/// \p cmp , then do \p second_op with \p c .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \tparam [in] SecondBinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] cmp The comparsion operator
/// \param [in] second_op The operation to do with the third value
/// \returns The comparison result of the two extended values. and \p second_op
/// with \p c .
template <typename AT, typename BT, typename BinaryOperation,
          typename SecondBinaryOperation>
inline constexpr unsigned extend_compare(AT a, BT b, unsigned c,
                                         BinaryOperation cmp,
                                         SecondBinaryOperation second_op) {
  return detail::extend_binary<unsigned, false>(a, b, c, cmp, second_op);
}

/// Extend \p a and \p b to 33 bit and return a << clamp(b, 0, 32).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns a << clamp(b, 0, 32)
template <typename RetT, typename T>
inline constexpr RetT extend_shl_clamp(T a, uint32_t b) {
  return detail::extend_binary<RetT, false>(a, sycl::clamp(b, 0u, 32u),
                                            detail::shift_left());
}

/// Extend Inputs to 33 bit, and return second_op(a << clamp(b, 0, 32), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(a << clamp(b, 0, 32), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shl_clamp(T a, uint32_t b, uint32_t c,
                                       BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, sycl::clamp(b, 0u, 32u), c,
                                            detail::shift_left(), second_op);
}

/// Extend \p a and \p b to 33 bit and return sat(a << clamp(b, 0, 32)).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns sat(a << clamp(b, 0, 32))
template <typename RetT, typename T>
inline constexpr RetT extend_shl_sat_clamp(T a, uint32_t b) {
  return detail::extend_binary<RetT, true>(a, sycl::clamp(b, 0u, 32u),
                                           detail::shift_left());
}

/// Extend Inputs to 33 bit, and return second_op(sat(a << clamp(b, 0, 32)), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(sat(a << clamp(b, 0, 32)), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shl_sat_clamp(T a, uint32_t b, uint32_t c,
                                           BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, sycl::clamp(b, 0u, 32u), c,
                                           detail::shift_left(), second_op);
}

/// Extend \p a and \p b to 33 bit and return a << (b & 0x1F).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns a << (b & 0x1F)
template <typename RetT, typename T>
inline constexpr RetT extend_shl_wrap(T a, uint32_t b) {
  return detail::extend_binary<RetT, false>(a, b & 0x1F, detail::shift_left());
}

/// Extend Inputs to 33 bit, and return second_op(a << (b & 0x1F), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(a << (b & 0x1F), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shl_wrap(T a, uint32_t b, uint32_t c,
                                      BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b & 0x1F, c,
                                            detail::shift_left(), second_op);
}

/// Extend \p a and \p b to 33 bit and return sat(a << (b & 0x1F)).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns sat(a << (b & 0x1F))
template <typename RetT, typename T>
inline constexpr RetT extend_shl_sat_wrap(T a, uint32_t b) {
  return detail::extend_binary<RetT, true>(a, b & 0x1F, detail::shift_left());
}

/// Extend Inputs to 33 bit, and return second_op(sat(a << (b & 0x1F)), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(sat(a << (b & 0x1F)), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shl_sat_wrap(T a, uint32_t b, uint32_t c,
                                          BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b & 0x1F, c, detail::shift_left(),
                                           second_op);
}

/// Extend \p a and \p b to 33 bit and return a >> clamp(b, 0, 32).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns a >> clamp(b, 0, 32)
template <typename RetT, typename T>
inline constexpr RetT extend_shr_clamp(T a, uint32_t b) {
  return detail::extend_binary<RetT, false>(a, sycl::clamp(b, 0u, 32u),
                                            detail::shift_right());
}

/// Extend Inputs to 33 bit, and return second_op(a >> clamp(b, 0, 32), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(a >> clamp(b, 0, 32), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shr_clamp(T a, uint32_t b, uint32_t c,
                                       BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, sycl::clamp(b, 0u, 32u), c,
                                            detail::shift_right(), second_op);
}

/// Extend \p a and \p b to 33 bit and return sat(a >> clamp(b, 0, 32)).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns sat(a >> clamp(b, 0, 32))
template <typename RetT, typename T>
inline constexpr RetT extend_shr_sat_clamp(T a, uint32_t b) {
  return detail::extend_binary<RetT, true>(a, sycl::clamp(b, 0u, 32u),
                                           detail::shift_right());
}

/// Extend Inputs to 33 bit, and return second_op(sat(a >> clamp(b, 0, 32)), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(sat(a >> clamp(b, 0, 32)), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shr_sat_clamp(T a, uint32_t b, uint32_t c,
                                           BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, sycl::clamp(b, 0u, 32u), c,
                                           detail::shift_right(), second_op);
}

/// Extend \p a and \p b to 33 bit and return a >> (b & 0x1F).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns a >> (b & 0x1F)
template <typename RetT, typename T>
inline constexpr RetT extend_shr_wrap(T a, uint32_t b) {
  return detail::extend_binary<RetT, false>(a, b & 0x1F, detail::shift_right());
}

/// Extend Inputs to 33 bit, and return second_op(a >> (b & 0x1F), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(a >> (b & 0x1F), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shr_wrap(T a, uint32_t b, uint32_t c,
                                      BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b & 0x1F, c,
                                            detail::shift_right(), second_op);
}

/// Extend \p a and \p b to 33 bit and return sat(a >> (b & 0x1F)).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \returns sat(a >> (b & 0x1F))
template <typename RetT, typename T>
inline constexpr RetT extend_shr_sat_wrap(T a, uint32_t b) {
  return detail::extend_binary<RetT, true>(a, b & 0x1F, detail::shift_right());
}

/// Extend Inputs to 33 bit, and return second_op(sat(a >> (b & 0x1F)), c).
/// \param [in] a The source value
/// \param [in] b The offset to shift
/// \param [in] c The value to merge
/// \param [in] second_op The operation to do with the third value
/// \returns second_op(sat(a >> (b & 0x1F)), c)
template <typename RetT, typename T, typename BinaryOperation>
inline constexpr RetT extend_shr_sat_wrap(T a, uint32_t b, uint32_t c,
                                          BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b & 0x1F, c,
                                           detail::shift_right(), second_op);
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a
/// 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized addition of the two
/// values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, std::plus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 2 elements vector type and extend each element to 17 bit. Then add each
/// half of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized subtraction of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, std::minus());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized abs_diff of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, abs_diff());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized minimum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, minimum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, maximum());
}

/// Extend \p a and \p b to 33 bit and vectorized compare input values using
///
/// specified comparison \p cmp .
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values.
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare2(AT a, BT b, BinaryOperation cmp) {
  return detail::extend_vbinary2<unsigned, false, false>(a, b, 0, cmp);
}

/// Extend Inputs to 33 bit, and vectorized compare input values using specified
/// comparison \p cmp , then add the result with \p c .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values, and add the
/// result with \p c .
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare2_add(AT a, BT b, unsigned c,
                                               BinaryOperation cmp) {
  return detail::extend_vbinary2<unsigned, false, true>(a, b, c, cmp);
}

/// Compute vectorized average of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c,
                                                     detail::average());
}

/// Compute vectorized average of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend average maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, detail::average());
}

/// Compute vectorized average of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, detail::average());
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a
/// 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized addition of the two
/// values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, std::plus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 4 elements vector type and extend each element to 9 bit. Then add each
/// half of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized subtraction of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, std::minus());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized abs_diff of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, abs_diff());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized minimum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, minimum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, maximum());
}

/// Extend \p a and \p b to 33 bit and vectorized compare input values using
///
/// specified comparison \p cmp .
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values.
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare4(AT a, BT b, BinaryOperation cmp) {
  return detail::extend_vbinary4<unsigned, false, false>(a, b, 0, cmp);
}

/// Extend Inputs to 33 bit, and vectorized compare input values using specified
/// comparison \p cmp , then add the result with \p c .
///
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the compare operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] cmp The comparsion operator
/// \returns The comparison result of the two extended values, and add the
/// result with \p c .
template <typename AT, typename BT, typename BinaryOperation>
inline constexpr unsigned extend_vcompare4_add(AT a, BT b, unsigned c,
                                               BinaryOperation cmp) {
  return detail::extend_vbinary4<unsigned, false, true>(a, b, c, cmp);
}

/// Compute vectorized average of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c,
                                                     detail::average());
}

/// Compute vectorized average of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized average of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, detail::average());
}

/// Compute vectorized average of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized average of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vavrg4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, detail::average());
}

namespace experimental {
namespace matrix {
namespace syclex = sycl::ext::oneapi::experimental;
struct row_major
    : public std::integral_constant<syclex::matrix::layout,
                                    syclex::matrix::layout::row_major> {};
struct col_major
    : public std::integral_constant<syclex::matrix::layout,
                                    syclex::matrix::layout::col_major> {};
struct a : public std::integral_constant<syclex::matrix::use,
                                         syclex::matrix::use::a> {};
struct b : public std::integral_constant<syclex::matrix::use,
                                         syclex::matrix::use::b> {};
struct accumulator
    : public std::integral_constant<syclex::matrix::use,
                                    syclex::matrix::use::accumulator> {};

template <class use, int m, int n, int k> struct matrix_size_traits;
template <int m, int n, int k> struct matrix_size_traits<a, m, n, k> {
  static constexpr int rows = m;
  static constexpr int cols = k;
};

template <int m, int n, int k> struct matrix_size_traits<b, m, n, k> {
  static constexpr int rows = k;
  static constexpr int cols = n;
};

template <int m, int n, int k> struct matrix_size_traits<accumulator, m, n, k> {
  static constexpr int rows = m;
  static constexpr int cols = n;
};

// A class that wraps the syclex::matrix::joint_matrix class and provides
// copy constructor and assignment operator.
template <typename use, int m, int n, int k, typename T,
          typename layout = std::integral_constant<
              syclex::matrix::layout, syclex::matrix::layout::dynamic>>
class joint_matrix {
  using joint_matrix_type = syclex::matrix::joint_matrix<
      sycl::sub_group, T, use::value, matrix_size_traits<use, m, n, k>::rows,
      matrix_size_traits<use, m, n, k>::cols, layout::value>;

  static inline decltype(auto) get_wi_data(joint_matrix_type &matrix) {
    return sycl::ext::oneapi::detail::get_wi_data(
        sycl::ext::oneapi::this_work_item::get_sub_group(), matrix);
  }

public:
  joint_matrix()
      : matrix(), x(matrix), num_elements(get_wi_data(matrix).length()) {}
  joint_matrix(joint_matrix &other)
      : x(matrix), num_elements(get_wi_data(matrix).length()) {
    syclex::matrix::joint_matrix_copy(
        sycl::ext::oneapi::this_work_item::get_sub_group(), other.get(),
        matrix);
  }
  joint_matrix &operator=(joint_matrix &other) {
    if (this != &other) {
      syclex::matrix::joint_matrix_copy(
          sycl::ext::oneapi::this_work_item::get_sub_group(), other.get(),
          matrix);
    }
    return *this;
  }

  joint_matrix_type &get() { return matrix; }

  const joint_matrix_type &get() const { return matrix; }

  class matrix_accessor {
    friend joint_matrix;
    joint_matrix_type &matrix;
    matrix_accessor(joint_matrix_type &matrix) : matrix(matrix) {}

  public:
    decltype(auto) operator[](unsigned I) { return get_wi_data(matrix)[I]; }
    decltype(auto) operator[](unsigned I) const {
      return get_wi_data(matrix)[I];
    }
  };

private:
  joint_matrix_type matrix;

public:
  matrix_accessor x;
  const size_t num_elements;
};

/// Collectively loads 1 8x8 b16 (128 bytes) matrix from local memory to private
/// memory per sub-group. Requires the sub-group size of kernel calling this
/// function to be 32.
/// 'mat' specifies the matrix index to be loaded. The first '(mat + 1) * 8'
/// work items of sub-group contain the starting address of their respective
/// matrix row in 'addr'.
/// After distributing addresses to other work items, each of the 32 work items
/// load 32-bits (2 packed 16-bit data) into 'm' for a total of 128 bytes.
/// 'trans' specifies to perform a transposed/non-transposed load by each work
/// item like below
/// Row Major: Each row of the matrix is loaded by a group of 4 work items(wi)
/// row-0: wi0 wi0 wi1 wi1 ... wi3 wi3
/// row-1: wi4 wi4 wi5 wi5 ... wi7 wi7
/// ...
/// row-6: wi24 wi24 wi25 wi25 ... wi27 wi27
/// row-7: wi28 wi28 wi29 wi29 ... wi31 wi31
/// Col Major: Each col of the matrix is loaded by a group of 4 work items(wi)
/// row-0: wi0 wi4 wi8 ... wi28
/// row-1: wi0 wi4 wi8 ... wi28
/// ...
/// row-6: wi3 wi7 wi11 ... wi31
/// row-7: wi3 wi7 wi11 ... wi31
/// \tparam [in] T Type of result variable (currently only supports 16-bit type)
/// \param [in] addr The starting address of corresponding matrix row for a work
/// item in local memory
/// \param [in] m The private memory to store the matrix. It points to 2 b16
/// type elements.
/// \param [in] trans Indicates whether the matrix to be loaded transposed
/// \param [in] mat The matrix index to be loaded
template <typename T>
void ldmatrix(uintptr_t addr, T *m, bool trans = false, unsigned mat = 0) {
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  int lane = sg.get_local_linear_id();

  int lane_group8_row = lane / 8;
  int lane_group8_col = lane % 8;

  if (!trans) {
    // calculate the source lane
    int src_lane = 2 * lane_group8_row;
    if (lane_group8_col >= 4)
      src_lane += 1;

    // Broadcast the address from the source lane
    auto recv_addr_uintp =
        dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

    // Cast the received address from uintptr_t to the type of 'm'
    auto recv_addr = reinterpret_cast<T *>(recv_addr_uintp);

    // Non-transposed load
    *m = recv_addr[lane_group8_col % 4];
  } else {
    // calculate the source lane
    int src_lane = (lane % 4) * 2;

    // Broadcast the address from the source lane
    auto recv_addr_uintp_1 =
        dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
    auto recv_addr_uintp_2 =
        dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

    // Cast the received address from uintptr_t to 'half *'
    auto recv_addr_1 = reinterpret_cast<sycl::half *>(recv_addr_uintp_1);
    auto recv_addr_2 = reinterpret_cast<sycl::half *>(recv_addr_uintp_2);

    // Transposed load
    int index = lane / 4;
    sycl::half val0 = recv_addr_1[index];
    sycl::half val1 = recv_addr_2[index];

    // Combine the two 16-bits into one 32-bit value
    sycl::half2 val = sycl::half2(val0, val1);
    *m = *reinterpret_cast<T *>(&val);
  }
}

/// Collectively loads 2 8x8 b16 (256 bytes) matrix from local memory to private
/// memory per sub-group. Requires the sub-group size of kernel calling this
/// function to be 32.
/// The first 16 work items of sub-group contain the starting address of their
/// respective matrix row in 'addr'.
/// After distributing addresses to other work items, each of the 32 work items
/// load 64-bits (32-bits per matrix) into 'm1' & 'm2' for a total of 256 bytes.
/// 'trans' specifies to perform a transposed/non-transposed load by each work
/// item like below
/// Row Major: Each row of the matrices is loaded by a group of 4 work items(wi)
/// row-0: wi0 wi0 wi1 wi1 ... wi3 wi3
/// row-1: wi4 wi4 wi5 wi5 ... wi7 wi7
/// ...
/// row-6: wi24 wi24 wi25 wi25 ... wi27 wi27
/// row-7: wi28 wi28 wi29 wi29 ... wi31 wi31
/// Col Major: Each col of the matrices is loaded by a group of 4 work items(wi)
/// row-0: wi0 wi4 wi8 ... wi28
/// row-1: wi0 wi4 wi8 ... wi28
/// ...
/// row-6: wi3 wi7 wi11 ... wi31
/// row-7: wi3 wi7 wi11 ... wi31
/// \tparam [in] T Type of result variable (currently only supports 16-bit type)
/// \param [in] addr The starting address of corresponding matrix row for a work
/// item in local memory
/// \param [in] m1 The private memory to store the data of 1st matrix. It points
/// to 2 b16 type elements.
/// \param [in] m2 The private memory to store the data of 2nd matrix. It points
/// to 2 b16 type elements.
/// \param [in] trans Indicates whether the matrix to be loaded transposed
template <typename T>
void ldmatrix(uintptr_t addr, T *m1, T *m2, bool trans = false) {
  // Load 1st matrix
  ldmatrix(addr, m1, trans, 0);
  // Load 2nd matrix
  ldmatrix(addr, m2, trans, 1);
}

/// Collectively loads 4 8x8 b16 (512 bytes) matrix from local memory to private
/// memory per sub-group. Requires the sub-group size of kernel calling this
/// function to be 32.
/// Each work item of sub-group contains the starting address of their
/// respective matrix row in 'addr'.
/// After distributing addresses to other work items, each of the 32 work items
/// load 128-bits (32-bits per matrix) into 'm1', 'm2', 'm3' & 'm4' for a total
/// of 512 bytes.
/// 'trans' specifies to perform a transposed/non-transposed load by each work
/// item like below
/// Row Major: Each row of the matrices is loaded by a group of 4 work items(wi)
/// row-0: wi0 wi0 wi1 wi1 ... wi3 wi3
/// row-1: wi4 wi4 wi5 wi5 ... wi7 wi7
/// ...
/// row-6: wi24 wi24 wi25 wi25 ... wi27 wi27
/// row-7: wi28 wi28 wi29 wi29 ... wi31 wi31
/// Col Major: Each col of the matrices is loaded by a group of 4 work items(wi)
/// row-0: wi0 wi4 wi8 ... wi28
/// row-1: wi0 wi4 wi8 ... wi28
/// ...
/// row-6: wi3 wi7 wi11 ... wi31
/// row-7: wi3 wi7 wi11 ... wi31
/// \tparam [in] T Type of result variable (currently only supports 16-bit type)
/// \param [in] addr The starting address of corresponding matrix row for a work
/// item in local memory
/// \param [in] m1 The private memory to store the data of 1st matrix. It points
/// to 2 b16 type elements.
/// \param [in] m2 The private memory to store the data of 2nd matrix. It points
/// to 2 b16 type elements.
/// \param [in] m3 The private memory to store the data of 3rd matrix. It points
/// to 2 b16 type elements.
/// \param [in] m4 The private memory to store the data of 4th matrix. It points
/// to 2 b16 type elements.
/// \param [in] trans Indicates whether the matrix to be loaded transposed
template <typename T>
void ldmatrix(uintptr_t addr, T *m1, T *m2, T *m3, T *m4, bool trans = false) {
  // Load 1st matrix
  ldmatrix(addr, m1, trans, 0);
  // Load 2nd matrix
  ldmatrix(addr, m2, trans, 1);
  // Load 3rd matrix
  ldmatrix(addr, m3, trans, 2);
  // Load 4th matrix
  ldmatrix(addr, m4, trans, 3);
}

/// Collectively stores 1 8x8 b16 (128 bytes) matrix from private memory to
/// local memory per sub-group.
/// Requires the sub-group size of kernel calling this function to be 32.
/// 'mat' specifies the matrix index to be stored. The first '(mat + 1) * 8'
/// work items of sub-group contain the starting address of their respective
/// matrix row in 'addr'.
/// After distributing addresses to other work items, each of the 32 work items
/// store 32-bits (2 packed 16-bit data) into 'm' for a total of 128 bytes.
/// 'trans' specifies to perform a transposed/non-transposed store by each work
/// item like below
/// Row Major: Each row of the matrix is stored by a group of 4 work items(wi)
/// row-0: wi0 wi0 wi1 wi1 ... wi3 wi3
/// row-1: wi4 wi4 wi5 wi5 ... wi7 wi7
/// ...
/// row-6: wi24 wi24 wi25 wi25 ... wi27 wi27
/// row-7: wi28 wi28 wi29 wi29 ... wi31 wi31
/// Col Major: Each col of the matrix is stored by a group of 4 work items(wi)
/// row-0: wi0 wi4 wi8 ... wi28
/// row-1: wi0 wi4 wi8 ... wi28
/// ...
/// row-6: wi3 wi7 wi11 ... wi31
/// row-7: wi3 wi7 wi11 ... wi31
/// \tparam [in] T Type of result variable (currently only supports 16-bit type)
/// \param [in] addr The starting address of corresponding matrix row for a work
/// item in local memory
/// \param [in] m The local memory to store the matrix. It points to 2 b16
/// type elements.
/// \param [in] trans Indicates whether the matrix to be stored transposed
/// \param [in] mat The matrix index to be stored
template <typename T>
void stmatrix(uintptr_t addr, T m, bool trans = false, unsigned mat = 0) {
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  int lane = sg.get_local_linear_id();

  int lane_group8_row = lane / 8;
  int lane_group8_col = lane % 8;

  if (!trans) {
    // calculate the source lane
    int src_lane = 2 * lane_group8_row;
    if (lane_group8_col >= 4)
      src_lane += 1;

    // Broadcast the address from the source lane
    auto recv_addr_uintp =
        dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

    // Cast the received address from uintptr_t to the type of 'm'
    auto recv_addr = reinterpret_cast<T *>(recv_addr_uintp);

    // Non-transposed store
    recv_addr[lane_group8_col % 4] = m;
  } else {
    // calculate the source lane
    int src_lane = (lane % 4) * 2;

    // Broadcast the address from the source lane
    auto recv_addr_uintp_1 =
        dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
    auto recv_addr_uintp_2 =
        dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

    // Cast the received address from uintptr_t to 'half *'
    auto recv_addr_1 = reinterpret_cast<sycl::half *>(recv_addr_uintp_1);
    auto recv_addr_2 = reinterpret_cast<sycl::half *>(recv_addr_uintp_2);

    // Split the 32-bit value of 'm' into two 16-bits
    sycl::half *val = reinterpret_cast<sycl::half *>(&m);

    // Transposed store
    int index = lane / 4;
    recv_addr_1[index] = val[0];
    recv_addr_2[index] = val[1];
  }
}

/// Collectively stores 2 8x8 b16 (256 bytes) matrix from private memory to
/// local memory per sub-group.
/// Requires the sub-group size of kernel calling this function to be 32.
/// The first 16 work items of sub-group contain the starting address of their
/// respective matrix row in 'addr'.
/// After distributing addresses to other work items, each of the 32 work items
/// store 64-bits (32-bits per matrix) into 'm1' & 'm2' for a total of 256
/// bytes.
/// 'trans' specifies to perform a transposed/non-transposed store by each work
/// item like below
/// Row Major: Each row of the matrices is stored by a group of 4 work items(wi)
/// row-0: wi0 wi0 wi1 wi1 ... wi3 wi3
/// row-1: wi4 wi4 wi5 wi5 ... wi7 wi7
/// ...
/// row-6: wi24 wi24 wi25 wi25 ... wi27 wi27
/// row-7: wi28 wi28 wi29 wi29 ... wi31 wi31
/// Col Major: Each col of the matrices is stored by a group of 4 work items(wi)
/// row-0: wi0 wi4 wi8 ... wi28
/// row-1: wi0 wi4 wi8 ... wi28
/// ...
/// row-6: wi3 wi7 wi11 ... wi31
/// row-7: wi3 wi7 wi11 ... wi31
/// \tparam [in] T Type of result variable (currently only supports 16-bit type)
/// \param [in] addr The starting address of corresponding matrix row for a work
/// item in local memory
/// \param [in] m1 The local memory to store the data of 1st matrix. It points
/// to 2 b16 type elements.
/// \param [in] m2 The local memory to store the data of 2nd matrix. It points
/// to 2 b16 type elements.
/// \param [in] trans Indicates whether the matrix to be stored transposed
template <typename T>
void stmatrix(uintptr_t addr, T m1, T m2, bool trans = false) {
  // Store 1st matrix
  stmatrix(addr, m1, trans, 0);
  // Store 2nd matrix
  stmatrix(addr, m2, trans, 1);
}

/// Collectively stores 4 8x8 b16 (512 bytes) matrix from private memory to
/// local memory per sub-group.
/// Requires the sub-group size of kernel calling this function to be 32.
/// Each work item of sub-group contains the starting address of their
/// respective matrix row in 'addr'.
/// After distributing addresses to other work items, each of the 32 work items
/// store 128-bits (32-bits per matrix) into 'm1', 'm2', 'm3' & 'm4' for a total
/// of 512 bytes.
/// 'trans' specifies to perform a transposed/non-transposed store by each work
/// item like below
/// Row Major: Each row of the matrices is stored by a group of 4 work items(wi)
/// row-0: wi0 wi0 wi1 wi1 ... wi3 wi3
/// row-1: wi4 wi4 wi5 wi5 ... wi7 wi7
/// ...
/// row-6: wi24 wi24 wi25 wi25 ... wi27 wi27
/// row-7: wi28 wi28 wi29 wi29 ... wi31 wi31
/// Col Major: Each col of the matrices is stored by a group of 4 work items(wi)
/// row-0: wi0 wi4 wi8 ... wi28
/// row-1: wi0 wi4 wi8 ... wi28
/// ...
/// row-6: wi3 wi7 wi11 ... wi31
/// row-7: wi3 wi7 wi11 ... wi31
/// \tparam [in] T Type of result variable (currently only supports 16-bit type)
/// \param [in] addr The starting address of corresponding matrix row for a work
/// item in local memory
/// \param [in] m1 The local memory to store the data of 1st matrix. It points
/// to 2 b16 type elements.
/// \param [in] m2 The local memory to store the data of 2nd matrix. It points
/// to 2 b16 type elements.
/// \param [in] m3 The local memory to store the data of 3rd matrix. It points
/// to 2 b16 type elements.
/// \param [in] m4 The local memory to store the data of 4th matrix. It points
/// to 2 b16 type elements.
/// \param [in] trans Indicates whether the matrix to be stored transposed
template <typename T>
void stmatrix(uintptr_t addr, T m1, T m2, T m3, T m4, bool trans = false) {
  // Store 1st matrix
  stmatrix(addr, m1, trans, 0);
  // Store 2nd matrix
  stmatrix(addr, m2, trans, 1);
  // Store 3rd matrix
  stmatrix(addr, m3, trans, 2);
  // Store 4th matrix
  stmatrix(addr, m4, trans, 3);
}

/// A helper struct that defines the pack type for the input matrix fragments
/// of mma() function based on the type of input matrix fragments.
/// The MMAType struct is specialized for different types of input matrices.
/// Currently, the specialization for f16, bf16 and s8 types is defined below.
/// \tparam [in] T The type of the input matrix fragments
template <typename T> struct MMAType {
  using PackType = uint32_t;
};

/// Each work item of a sub-group (limited to size 32) calling this function
/// calculates a subset fragment for the output matrix D using MAD operation on
/// A, B & C matrix fragments (D = A * B + C).
/// Current supported shapes & types:
/// - m8n8k4 (f32.f16.f16.f32)
/// - m8n8k16 (s32.s8.s8.s32)
/// - m16n8k8 (f32.f16.f16.f32 & f32.bf16.bf16.f32)
/// - m16n8k16 (f32.f16.f16.f32 & s32.s8.s8.s32)
/// - m16n8k32 (s32.s8.s8.s32)
/// Here, m, n & k define the shapes of A, B & C matrices respectively
/// (A = [m x k], B = [k x n], C = [m x n]).
/// \tparam [in] M The rows of A, C & D matrices
/// \tparam [in] N The columns of B, C, D matrices
/// \tparam [in] K The columns & rows of A & B matrices respectively
/// \tparam [in] ABType The type of the input matrix (A & B) fragment
/// \tparam [in] CDType The type of the output matrix (C & D) fragment
/// \param [out] d_mat_frag The fragment of the output matrix D to store the
/// result of A * B + C
/// \param [in] a_mat_frag The fragment of the input matrix A to be multiplied
/// with B matrix fragment
/// \param [in] b_mat_frag The fragment of the input matrix B to be multiplied
/// with A matrix fragment
/// \param [in] c_mat_frag The fragment of the input matrix C to be added with
/// the result of A * B fragments
template <int M, int N, int K, typename ABType, typename CDType>
void mma(volatile void **d_mat_frag, void *a_mat_frag, void *b_mat_frag,
         void *c_mat_frag) {
  auto d = reinterpret_cast<volatile CDType **>(d_mat_frag);
  auto a = reinterpret_cast<typename MMAType<ABType>::PackType *>(a_mat_frag);
  auto b = reinterpret_cast<typename MMAType<ABType>::PackType *>(b_mat_frag);
  auto c = reinterpret_cast<CDType *>(c_mat_frag);

  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  int lane = sg.get_local_linear_id();

  static_assert((M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
                    (M == 16 && N == 8 && K == 8) ||
                    (M == 16 && N == 8 && K == 16) ||
                    (M == 16 && N == 8 && K == 32),
                "Unsupported MMA shape!");

  short row_load_offset = 4 * (lane >> 2);
  short col_load_offset = 8 * (lane % 4);

  if constexpr (M == 8 && N == 8 && K == 4) {
    if constexpr (std::is_floating_point_v<CDType>) {
      col_load_offset = row_load_offset % 16;

      // Init D matrix with fragments of C matrix
      *d[0] = c[0];
      *d[1] = c[1];
      *d[2] = c[2];
      *d[3] = c[3];
      *d[4] = c[4];
      *d[5] = c[5];
      *d[6] = c[6];
      *d[7] = c[7];

      // Calculate the row and col offset indices to iterate through the row &
      // col fragments of A & B matrices
      int r_ind = (lane % 2) ? 1 : 0;
      int c_ind = ((lane % 4) / 2) ? 2 : 0;

      // Each sub-group is responsible for computing a fragment size of 8*8
      // elements of matrix D for each of 4 MMA computations.
      // Each work item computes 8 elements of matrix D by gathering
      // their corresponding col & row matrix fragments of length k (4)
      // from A & B matrices respectively using below mapping logic:
      // row0 = (i % 4) if (lane < 16) else (i % 4) + 4
      // col0 = (lane % 4)
      // As each row & col fragment of A & B matrices is distributed across
      // 4 work items, each iteration of below loop loads a partial fragment of
      // matrix A (row) and matrix B (col) using the row & col offsets.
      typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

      for (int i = 0; i < 4; i++) {
        // Load partial fragment from col0 of matrix A ({a0, a1})
        recv_a[0] = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
        // Load partial fragment from col0 of matrix A ({a2, a3})
        recv_a[1] = dpct::select_from_sub_group(sg, a[1], row_load_offset + i);

        // Load partial fragment from row0 of matrix B ({b0, b1})
        recv_b[0] = dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
        // Load partial fragment from row0 of matrix B ({b2, b3})
        recv_b[1] = dpct::select_from_sub_group(sg, b[1], col_load_offset + i);

        auto ra = reinterpret_cast<ABType *>(recv_a);
        auto rb = reinterpret_cast<ABType *>(recv_b);

        // Each work item calculates a partial product of A & B matrix fragments
        // and adds it to the corresponding D matrix fragment
        // (for even work item indices)
        // d0 += col0{ a0 } * row0{ b0 }
        // d1 += col0{ a0 } * row0{ b1 }
        // d2 += col1{ a2 } * row0{ b0 }
        // d3 += col1{ a2 } * row0{ b1 }
        // (for odd work item indices)
        // d0 += col0{ a1 } * row0{ b2 }
        // d1 += col0{ a1 } * row0{ b3 }
        // d2 += col1{ a3 } * row0{ b2 }
        // d3 += col1{ a3 } * row0{ b3 }
        *d[0] += static_cast<float>(ra[r_ind]) * static_cast<float>(rb[c_ind]);
        *d[1] +=
            static_cast<float>(ra[r_ind]) * static_cast<float>(rb[c_ind + 1]);
        *d[2] +=
            static_cast<float>(ra[r_ind + 2]) * static_cast<float>(rb[c_ind]);
        *d[3] += static_cast<float>(ra[r_ind + 2]) *
                 static_cast<float>(rb[c_ind + 1]);

        // Load partial fragment from row1 of matrix B ({b0, b1})
        recv_b[0] =
            dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 16);
        // Load partial fragment from row1 of matrix B ({b2, b3})
        recv_b[1] =
            dpct::select_from_sub_group(sg, b[1], col_load_offset + i + 16);

        // (for even work item indices)
        // d0 += col0{ a0 } * row1{ b0 }
        // d1 += col0{ a0 } * row1{ b1 }
        // d2 += col1{ a2 } * row1{ b0 }
        // d3 += col1{ a2 } * row1{ b1 }
        // (for odd work item indices)
        // d0 += col0{ a1 } * row1{ b2 }
        // d1 += col0{ a1 } * row1{ b3 }
        // d2 += col1{ a3 } * row1{ b2 }
        // d3 += col1{ a3 } * row1{ b3 }
        *d[4] += static_cast<float>(ra[r_ind]) * static_cast<float>(rb[c_ind]);
        *d[5] +=
            static_cast<float>(ra[r_ind]) * static_cast<float>(rb[c_ind + 1]);
        *d[6] +=
            static_cast<float>(ra[r_ind + 2]) * static_cast<float>(rb[c_ind]);
        *d[7] += static_cast<float>(ra[r_ind + 2]) *
                 static_cast<float>(rb[c_ind + 1]);
      }
    }
  } else if constexpr (M == 8 && N == 8 && K == 16) {
    if constexpr (std::is_integral_v<ABType>) {
      // Init D matrix with fragments of C matrix
      *d[0] = c[0];
      *d[1] = c[1];

      // Each sub-group is responsible for computing a fragment size of 16*8
      // elements of matrix D.
      // Each work item computes 2 elements of matrix D by gathering
      // their corresponding row & col matrix fragments of length k (16)
      // from A & B matrices respectively using below mapping logic:
      // row0 = ((lane % 4) * 4) + i
      // col0 = (lane >> 2)
      // As each row & col fragment of A & B matrices is distributed across
      // 4 work items, each iteration of below loop loads a partial fragment of
      // matrix A (row) and matrix B (col) using the row & col offsets.
      for (int i = 0; i < 4; i++) {
        typename MMAType<ABType>::PackType recv_a, recv_b[2];

        // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
        recv_a = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
        // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
        recv_b[0] = dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
        // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
        recv_b[1] =
            dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

        auto a = reinterpret_cast<ABType *>(&recv_a);
        auto b = reinterpret_cast<ABType *>(recv_b);

        // Each work item calculates a partial product of A & B matrix fragments
        // and adds it to the corresponding D matrix fragment
        // d0 += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 }
        // d1 += row0{ a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 }
        // d2 += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 }
        // d3 += row0{ a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 }
        for (int j = 0; j < 4; j++) {
          *d[0] += a[j] * b[j];
          *d[1] += a[j] * b[j + 4];
        }
      }
    }
  } else if constexpr (M == 16 && N == 8 && K == 8) {
    if constexpr (std::is_floating_point_v<CDType>) {
      // Init D matrix fragment with C matrix fragment
      *d[0] = c[0];
      *d[1] = c[1];
      *d[2] = c[2];
      *d[3] = c[3];

      // Each sub-group is responsible for computing a fragment size of 16*8
      // elements of matrix D.
      // Each work item computes 4 elements of matrix D by gathering
      // their corresponding row & col matrix fragments of length k (8)
      // from A & B matrices respectively using below mapping logic:
      // row0 = (lane >> 2) & row1 = (lane >> 2) + 8
      // col0 = (lane % 4) * 2 + (i & 0x1)
      // As each row & col fragment of A & B matrices is distributed across
      // 4 work items, each iteration of below loop loads a partial fragment of
      // matrix A (row) and matrix B (col) using the row & col offsets.
      for (int i = 0; i < 4; i++) {
        typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

        // Load partial fragment from row0 of matrix A ({a0, a1})
        recv_a[0] = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
        // Load partial fragment from row1 of matrix A ({a2, a3})
        recv_a[1] = dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
        // Load partial fragment from col0 of matrix B ({b0, b1})
        recv_b[0] = dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
        // Load partial fragment from col1 of matrix B ({b0, b1})
        recv_b[1] =
            dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

        auto ra = reinterpret_cast<ABType *>(recv_a);
        auto rb = reinterpret_cast<ABType *>(recv_b);

        // Each work item calculates a partial product of A & B matrix fragments
        // and adds it to the corresponding D matrix fragment
        // d0 += row0{ a0, a1 } * col0{ b0, b1 }
        // d1 += row0{ a0, a1 } * col1{ b0, b1 }
        // d2 += row1{ a2, a3 } * col0{ b0, b1 }
        // d3 += row1{ a2, a3 } * col1{ b0, b1 }
        for (int j = 0; j < 2; j++) {
          *d[0] += static_cast<float>(ra[j]) * static_cast<float>(rb[j]);
          *d[1] += static_cast<float>(ra[j]) * static_cast<float>(rb[j + 2]);
          *d[2] += static_cast<float>(ra[j + 2]) * static_cast<float>(rb[j]);
          *d[3] +=
              static_cast<float>(ra[j + 2]) * static_cast<float>(rb[j + 2]);
        }
      }
    }
  } else if constexpr (M == 16 && N == 8 && K == 16) {
    if constexpr (std::is_floating_point_v<CDType>) {
      // Init D matrix fragment with C matrix fragment
      *d[0] = c[0];
      *d[1] = c[1];
      *d[2] = c[2];
      *d[3] = c[3];

      // Each sub-group is responsible for computing a fragment size of 16*8
      // elements of matrix D.
      // Each work item computes 4 elements of matrix D by gathering
      // their corresponding row & col matrix fragments of length k (8)
      // from A & B matrices respectively using below mapping logic:
      // row0 = (lane >> 2)    & row1 = (lane >> 2) + 8
      // col0 = (lane % 4) * 2 & col1 = (lane % 4) * 2 + 1
      // As each row & col fragment of A & B matrices is distributed across
      // 4 work items, each iteration of below loop loads a partial fragment of
      // matrix A (row) and matrix B (col) using the row & col offsets.
      for (int i = 0; i < 4; i++) {
        typename MMAType<ABType>::PackType recv_a[4], recv_b[4];

        // Load partial fragment from row0 of matrix A ({a0, a1})
        recv_a[0] = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
        // Load partial fragment from row0 of matrix A ({a2, a3})
        recv_a[1] = dpct::select_from_sub_group(sg, a[2], row_load_offset + i);
        // Load partial fragment from row1 of matrix A ({a0, a1})
        recv_a[2] = dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
        // Load partial fragment from row1 of matrix A ({a2, a3})
        recv_a[3] = dpct::select_from_sub_group(sg, a[3], row_load_offset + i);

        // Load partial fragment from col0 of matrix B ({b0, b1})
        recv_b[0] = dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
        // Load partial fragment from col0 of matrix B ({b2, b3})
        recv_b[1] = dpct::select_from_sub_group(sg, b[1], col_load_offset + i);
        // Load partial fragment from col1 of matrix B ({b0, b1})
        recv_b[2] =
            dpct::select_from_sub_group(sg, b[0], col_load_offset + 4 + i);
        // Load partial fragment from col1 of matrix B ({b2, b3})
        recv_b[3] =
            dpct::select_from_sub_group(sg, b[1], col_load_offset + 4 + i);

        auto ra = reinterpret_cast<ABType *>(recv_a);
        auto rb = reinterpret_cast<ABType *>(recv_b);

        // Each work item calculates a partial product of A & B matrix fragments
        // and adds it to the corresponding D matrix fragment
        // d0 += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 }
        // d1 += row0{ a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 }
        // d2 += row1{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 }
        // d3 += row1{ a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 }
        for (int j = 0; j < 4; j++) {
          *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
          *d[1] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
          *d[2] += static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
          *d[3] +=
              static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j + 4]);
        }
      }
    } else if constexpr (std::is_integral_v<ABType>) {
      // Init D matrix with fragments of C matrix
      *d[0] = c[0];
      *d[1] = c[1];
      *d[2] = c[2];
      *d[3] = c[3];

      // Each sub-group is responsible for computing a fragment size of 16*8
      // elements of matrix D.
      // Each work item computes 4 elements of matrix D by gathering
      // their corresponding row & col matrix fragments of length k (8)
      // from A & B matrices respectively using below mapping logic:
      // row0 = (lane >> 2)    & row1 = (lane >> 2) + 8
      // col0 = (lane % 4) * 2 & col1 = (lane % 4) * 2 + 1
      // As each row & col fragment of A & B matrices is distributed across
      // 4 work items, each iteration of below loop loads a partial fragment of
      // matrix A (row) and matrix B (col) using the row & col offsets.
      for (int i = 0; i < 4; i++) {
        typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

        // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
        recv_a[0] = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
        // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
        recv_a[1] = dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
        // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
        recv_b[0] = dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
        // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
        recv_b[1] =
            dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

        auto ra = reinterpret_cast<ABType *>(recv_a);
        auto rb = reinterpret_cast<ABType *>(recv_b);

        // Each work item calculates a partial product of A & B matrix fragments
        // and adds it to the corresponding D matrix fragment
        // d0 += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 }
        // d1 += row0{ a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 }
        // d2 += row1{ a4, a5, a6, a7 } * col0{ b0, b1, b2, b3 }
        // d3 += row1{ a4, a5, a6, a7 } * col1{ b4, b5, b6, b7 }
        for (int i = 0; i < 4; i++) {
          *d[0] += ra[i] * rb[i];
          *d[1] += ra[i] * rb[i + 4];
          *d[2] += ra[i + 4] * rb[i];
          *d[3] += ra[i + 4] * rb[i + 4];
        }
      }
    }
  } else if constexpr (M == 16 && N == 8 && K == 32) {
    if constexpr (std::is_integral_v<ABType>) {
      // Init D matrix with fragments of C matrix
      *d[0] = c[0];
      *d[1] = c[1];
      *d[2] = c[2];
      *d[3] = c[3];

      // Each sub-group is responsible for computing a fragment size of 16*8
      // elements of matrix D.
      // Each work item computes 4 elements of matrix D by gathering
      // their corresponding row & col matrix fragments of length k (32)
      // from A & B matrices respectively using below mapping logic:
      // row0 = (lane >> 2)    & row1 = (lane >> 2) + 8
      // col0 = ((lane % 4) * 4) + (i & 0x3) & col1 = ((lane % 4) * 4) + (i &
      // 0x3) As each row & col fragment of A & B matrices is distributed across
      // 4 work items, each iteration of below loop loads a partial fragment of
      // matrix A (row) and matrix B (col) using the row & col offsets.
      for (int i = 0; i < 4; i++) {
        typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

        // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
        recv_a[0] = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
        // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
        recv_a[1] = dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
        // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
        recv_b[0] = dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
        // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
        recv_b[1] =
            dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

        auto a = reinterpret_cast<ABType *>(recv_a);
        auto b = reinterpret_cast<ABType *>(recv_b);

        // Each work item calculates a partial product of A & B matrix fragments
        // and adds it to the corresponding D matrix fragment
        // d0 += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 }
        // d1 += row0{ a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 }
        // d2 += row1{ a4, a5, a6, a7 } * col0{ b0, b1, b2, b3 }
        // d3 += row1{ a4, a5, a6, a7 } * col1{ b0, b1, b2, b3 }
        for (int j = 0; j < 4; j++) {
          *d[0] += a[j] * b[j];
          *d[1] += a[j] * b[j + 4];
          *d[2] += a[j + 4] * b[j];
          *d[3] += a[j + 4] * b[j + 4];
        }
      }

      for (int i = 0; i < 4; i++) {
        typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

        // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
        recv_a[0] = dpct::select_from_sub_group(sg, a[2], row_load_offset + i);
        // Load partial fragment from row1 of matrix A ({a12, a13, a14, a15})
        recv_a[1] = dpct::select_from_sub_group(sg, a[3], row_load_offset + i);
        // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
        recv_b[0] = dpct::select_from_sub_group(sg, b[1], col_load_offset + i);
        // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
        recv_b[1] =
            dpct::select_from_sub_group(sg, b[1], col_load_offset + i + 4);

        auto a = reinterpret_cast<ABType *>(recv_a);
        auto b = reinterpret_cast<ABType *>(recv_b);

        // Each work item calculates a partial product of A & B matrix fragments
        // and adds it to the corresponding D matrix fragment
        // d0 += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 }
        // d1 += row0{ a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 }
        // d2 += row1{ a12, a13, a14, a15 } * col0{ b4, b5, b6, b7 }
        // d3 += row1{ a12, a13, a14, a15 } * col1{ b4, b5, b6, b7 }
        for (int j = 0; j < 4; j++) {
          *d[0] += a[j] * b[j];
          *d[1] += a[j] * b[j + 4];
          *d[2] += a[j + 4] * b[j];
          *d[3] += a[j + 4] * b[j + 4];
        }
      }
    }
  }
}

} // namespace matrix
} // namespace experimental

} // namespace dpct

#endif // __DPCT_MATH_HPP__
