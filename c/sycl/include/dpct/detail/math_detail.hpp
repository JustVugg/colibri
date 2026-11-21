//==---- math_internal.hpp -----------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===---------------------------------------------------------------------===//

#ifndef __DPCT_DETAIL_MATH_DETAIL_HPP__
#define __DPCT_DETAIL_MATH_DETAIL_HPP__

namespace dpct {
// forward declar
template <typename T> inline T clamp(T val, T min_val, T max_val);

namespace detail {

template <typename VecT, class BinaryOperation, class = void>
class vectorized_binary {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    VecT v4;
    for (size_t i = 0; i < v4.size(); ++i) {
      v4[i] = binary_op(a[i], b[i]);
    }
    return v4;
  }
};
template <typename VecT, class BinaryOperation>
class vectorized_binary<
    VecT, BinaryOperation,
    std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>> {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    return binary_op(a, b).template as<VecT>();
  }
};

template <typename T> inline bool isnan(const T a) { return sycl::isnan(a); }
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
inline bool isnan(const sycl::ext::oneapi::bfloat16 a) {
  return sycl::ext::oneapi::experimental::isnan(a);
}
#endif

template <typename T>
constexpr bool is_floating_point =
    std::disjunction_v<std::is_floating_point<T>, std::is_same<T, sycl::half>
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
                       ,
                       std::is_same<T, sycl::ext::oneapi::bfloat16>
#endif
                       >;

struct shift_left {
  template <typename T>
  auto operator()(const T x, const uint32_t offset) const {
    return x << offset;
  }
};

struct shift_right {
  template <typename T>
  auto operator()(const T x, const uint32_t offset) const {
    return x >> offset;
  }
};

struct average {
  template <typename T> auto operator()(const T x, const T y) const {
    return (x + y + (x + y >= 0)) >> 1;
  }
};

/// Extend the 'val' to 'bit' size, zero extend for unsigned int and signed
/// extend for signed int.
template <typename T> inline auto zero_or_signed_extent(T val, unsigned bit) {
  if constexpr (std::is_signed_v<T>) {
    if constexpr (std::is_same_v<T, int32_t>) {
      assert(bit < 64 &&
             "When extend int32 value, bit must be smaller than 64.");
      return int64_t(val) << (64 - bit) >> (64 - bit);
    } else if constexpr (std::is_same_v<T, int16_t>) {
      assert(bit < 32 &&
             "When extend int16 value, bit must be smaller than 32.");
      return int32_t(val) << (32 - bit) >> (32 - bit);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      assert(bit < 16 &&
             "When extend int8 value, bit must be smaller than 16.");
      return int16_t(val) << (16 - bit) >> (16 - bit);
    } else {
      assert(bit < 64 && "Cannot extend int64 value.");
      return val;
    }
  } else
    return val;
}

template <typename RetT, bool NeedSat, typename AT, typename BT,
          typename BinaryOperation>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<RetT> && sizeof(AT) == 4 && sizeof(BT) == 4 &&
        sizeof(RetT) == 4,
    RetT>
extend_binary(AT a, BT b, BinaryOperation binary_op) {
  int64_t extend_a = zero_or_signed_extent(a, 33);
  int64_t extend_b = zero_or_signed_extent(b, 33);
  int64_t ret = binary_op(extend_a, extend_b);
  if constexpr (NeedSat)
    return dpct::clamp<int64_t>(ret, std::numeric_limits<RetT>::min(),
                                std::numeric_limits<RetT>::max());
  return ret;
}

template <typename RetT, bool NeedSat, typename AT, typename BT, typename CT,
          typename BinaryOperation1, typename BinaryOperation2>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<CT> && std::is_integral_v<RetT> && sizeof(AT) == 4 &&
        sizeof(BT) == 4 && sizeof(CT) == 4 && sizeof(RetT) == 4,
    RetT>
extend_binary(AT a, BT b, CT c, BinaryOperation1 binary_op,
              BinaryOperation2 second_op) {
  int64_t extend_a = zero_or_signed_extent(a, 33);
  int64_t extend_b = zero_or_signed_extent(b, 33);
  int64_t extend_temp =
      zero_or_signed_extent(binary_op(extend_a, extend_b), 34);
  if constexpr (NeedSat)
    extend_temp =
        dpct::clamp<int64_t>(extend_temp, std::numeric_limits<RetT>::min(),
                             std::numeric_limits<RetT>::max());
  int64_t extend_c = zero_or_signed_extent(c, 33);
  return second_op(extend_temp, extend_c);
}

template <typename T> sycl::vec<int32_t, 2> extractAndExtend2(T a) {
  sycl::vec<int32_t, 2> ret;
  sycl::vec<T, 1> va{a};
  if constexpr (std::is_signed_v<T>) {
    auto v = va.template as<sycl::vec<int16_t, 2>>();
    ret[0] = zero_or_signed_extent(v[0], 17);
    ret[1] = zero_or_signed_extent(v[1], 17);
  } else {
    auto v = va.template as<sycl::vec<uint16_t, 2>>();
    ret[0] = zero_or_signed_extent(v[0], 17);
    ret[1] = zero_or_signed_extent(v[1], 17);
  }
  return ret;
}

template <typename T> sycl::vec<int16_t, 4> extractAndExtend4(T a) {
  sycl::vec<int16_t, 4> ret;
  sycl::vec<T, 1> va{a};
  if constexpr (std::is_signed_v<T>) {
    auto v = va.template as<sycl::vec<int8_t, 4>>();
    ret[0] = zero_or_signed_extent(v[0], 9);
    ret[1] = zero_or_signed_extent(v[1], 9);
    ret[2] = zero_or_signed_extent(v[2], 9);
    ret[3] = zero_or_signed_extent(v[3], 9);
  } else {
    auto v = va.template as<sycl::vec<uint8_t, 4>>();
    ret[0] = zero_or_signed_extent(v[0], 9);
    ret[1] = zero_or_signed_extent(v[1], 9);
    ret[2] = zero_or_signed_extent(v[2], 9);
    ret[3] = zero_or_signed_extent(v[3], 9);
  }
  return ret;
}

template <typename RetT, bool NeedSat, bool NeedAdd, typename AT, typename BT,
          typename BinaryOperation>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<RetT> && sizeof(AT) == 4 && sizeof(BT) == 4 &&
        sizeof(RetT) == 4,
    RetT>
extend_vbinary2(AT a, BT b, RetT c, BinaryOperation binary_op) {
  sycl::vec<int32_t, 2> extend_a = extractAndExtend2(a);
  sycl::vec<int32_t, 2> extend_b = extractAndExtend2(b);
  sycl::vec<int32_t, 2> temp{binary_op(extend_a[0], extend_b[0]),
                             binary_op(extend_a[1], extend_b[1])};
  if constexpr (NeedSat) {
    int32_t min_val = 0, max_val = 0;
    if constexpr (std::is_signed_v<RetT>) {
      min_val = std::numeric_limits<int16_t>::min();
      max_val = std::numeric_limits<int16_t>::max();
    } else {
      min_val = std::numeric_limits<uint16_t>::min();
      max_val = std::numeric_limits<uint16_t>::max();
    }
    temp = dpct::clamp(temp, {min_val, min_val}, {max_val, max_val});
  }
  if constexpr (NeedAdd) {
    return temp[0] + temp[1] + c;
  }
  if constexpr (std::is_signed_v<RetT>) {
    return sycl::vec<int16_t, 2>{temp[0], temp[1]}.as<sycl::vec<RetT, 1>>();
  } else {
    return sycl::vec<uint16_t, 2>{temp[0], temp[1]}.as<sycl::vec<RetT, 1>>();
  }
}

template <typename RetT, bool NeedSat, bool NeedAdd, typename AT, typename BT,
          typename BinaryOperation>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<RetT> && sizeof(AT) == 4 && sizeof(BT) == 4 &&
        sizeof(RetT) == 4,
    RetT>
extend_vbinary4(AT a, BT b, RetT c, BinaryOperation binary_op) {
  sycl::vec<int16_t, 4> extend_a = extractAndExtend4(a);
  sycl::vec<int16_t, 4> extend_b = extractAndExtend4(b);
  sycl::vec<int16_t, 4> temp{
      binary_op(extend_a[0], extend_b[0]), binary_op(extend_a[1], extend_b[1]),
      binary_op(extend_a[2], extend_b[2]), binary_op(extend_a[3], extend_b[3])};
  if constexpr (NeedSat) {
    int16_t min_val = 0, max_val = 0;
    if constexpr (std::is_signed_v<RetT>) {
      min_val = std::numeric_limits<int8_t>::min();
      max_val = std::numeric_limits<int8_t>::max();
    } else {
      min_val = std::numeric_limits<uint8_t>::min();
      max_val = std::numeric_limits<uint8_t>::max();
    }
    temp = dpct::clamp(temp, {min_val, min_val, min_val, min_val},
                       {max_val, max_val, max_val, max_val});
  }
  if constexpr (NeedAdd) {
    return temp[0] + temp[1] + temp[2] + temp[3] + c;
  }
  if constexpr (std::is_signed_v<RetT>) {
    return sycl::vec<int8_t, 4>{temp[0], temp[1], temp[2], temp[3]}
        .as<sycl::vec<RetT, 1>>();
  } else {
    return sycl::vec<uint8_t, 4>{temp[0], temp[1], temp[2], temp[3]}
        .as<sycl::vec<RetT, 1>>();
  }
}

template <typename T1, typename T2>
using dot_product_acc_t =
    std::conditional_t<std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
                       uint32_t, int32_t>;

template <typename T> sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
  return sycl::vec<T, 1>(val)
      .template as<sycl::vec<
          std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>, 4>>()
      .template convert<T>();
}

template <typename T> sycl::vec<T, 2> extract_and_sign_or_zero_extend2(T val) {
  return sycl::vec<T, 1>(val)
      .template as<sycl::vec<
          std::conditional_t<std::is_signed_v<T>, int16_t, uint16_t>, 2>>()
      .template convert<T>();
}

} // namespace detail
} // namespace dpct
#endif //!__DPCT_DETAIL_MATH_DETAIL_HPP__
