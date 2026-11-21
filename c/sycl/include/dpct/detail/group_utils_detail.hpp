//==---- group_utils_detail.hpp ---------------------*- C++----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===---------------------------------------------------------------------===//
#ifndef __DPCT_DETAIL_GROUP_UTILS_DETAIL_HPP__
#define __DPCT_DETAIL_GROUP_UTILS_DETAIL_HPP__

#include <iterator>
#include <stdexcept>
#include <sycl/sycl.hpp>

namespace dpct {

namespace group {
namespace detail {

typedef uint16_t digit_counter_type;
typedef uint32_t packed_counter_type;

template <int N, int CURRENT_VAL = N, int COUNT = 0> struct log2 {
  enum { VALUE = log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT> struct log2<N, 0, COUNT> {
  enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

template <int RADIX_BITS, bool DESCENDING = false> class radix_rank {
public:
  static size_t get_local_memory_size(size_t group_threads) {
    return group_threads * PADDED_COUNTER_LANES * sizeof(packed_counter_type);
  }

  radix_rank(uint8_t *local_memory) : _local_memory(local_memory) {}

  template <typename Item, typename KT, int VALUES_PER_THREAD>
  __dpct_inline__ void
  rank_keys(const Item &item, KT (&keys)[VALUES_PER_THREAD],
            int (&ranks)[VALUES_PER_THREAD], int current_bit, int num_bits) {

    digit_counter_type thread_prefixes[VALUES_PER_THREAD];
    digit_counter_type *digit_counters[VALUES_PER_THREAD];
    digit_counter_type *buffer =
        reinterpret_cast<digit_counter_type *>(_local_memory);

    reset_local_memory(item);

    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      uint32_t digit = ::dpct::bfe(keys[i], current_bit, num_bits);
      uint32_t sub_counter = digit >> LOG_COUNTER_LANES;
      uint32_t counter_lane = digit & (COUNTER_LANES - 1);

      if (DESCENDING) {
        sub_counter = PACKING_RATIO - 1 - sub_counter;
        counter_lane = COUNTER_LANES - 1 - counter_lane;
      }

      digit_counters[i] =
          &buffer[counter_lane * item.get_local_range().size() * PACKING_RATIO +
                  item.get_local_linear_id() * PACKING_RATIO + sub_counter];
      thread_prefixes[i] = *digit_counters[i];
      *digit_counters[i] = thread_prefixes[i] + 1;
    }

    item.barrier(sycl::access::fence_space::local_space);

    scan_counters(item);

    item.barrier(sycl::access::fence_space::local_space);

    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
      ranks[i] = thread_prefixes[i] + *digit_counters[i];
    }
  }

private:
  template <typename Item>
  __dpct_inline__ void reset_local_memory(const Item &item) {
    packed_counter_type *ptr =
        reinterpret_cast<packed_counter_type *>(_local_memory);

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      ptr[i * item.get_local_range().size() + item.get_local_linear_id()] = 0;
    }
  }

  template <typename Item>
  __dpct_inline__ packed_counter_type upsweep(const Item &item) {
    packed_counter_type sum = 0;
    packed_counter_type *ptr =
        reinterpret_cast<packed_counter_type *>(_local_memory);

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; i++) {
      cached_segment[i] =
          ptr[item.get_local_linear_id() * PADDED_COUNTER_LANES + i];
    }

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      sum += cached_segment[i];
    }

    return sum;
  }

  template <typename Item>
  __dpct_inline__ void exclusive_downsweep(const Item &item,
                                           packed_counter_type raking_partial) {
    packed_counter_type *ptr =
        reinterpret_cast<packed_counter_type *>(_local_memory);
    packed_counter_type sum = raking_partial;

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      packed_counter_type value = cached_segment[i];
      cached_segment[i] = sum;
      sum += value;
    }

#pragma unroll
    for (int i = 0; i < PADDED_COUNTER_LANES; ++i) {
      ptr[item.get_local_linear_id() * PADDED_COUNTER_LANES + i] =
          cached_segment[i];
    }
  }

  struct prefix_callback {
    __dpct_inline__ packed_counter_type
    operator()(packed_counter_type block_aggregate) {
      packed_counter_type block_prefix = 0;

#pragma unroll
      for (int packed = 1; packed < PACKING_RATIO; packed++) {
        block_prefix += block_aggregate
                        << (sizeof(digit_counter_type) * 8 * packed);
      }

      return block_prefix;
    }
  };

  template <typename Item>
  __dpct_inline__ void scan_counters(const Item &item) {
    packed_counter_type raking_partial = upsweep(item);

    prefix_callback callback;
    packed_counter_type exclusive_partial = exclusive_scan(
        item, raking_partial, sycl::ext::oneapi::plus<packed_counter_type>(),
        callback);

    exclusive_downsweep(item, exclusive_partial);
  }

private:
  static constexpr int PACKING_RATIO =
      sizeof(packed_counter_type) / sizeof(digit_counter_type);
  static constexpr int LOG_PACKING_RATIO = log2<PACKING_RATIO>::VALUE;
  static constexpr int LOG_COUNTER_LANES = RADIX_BITS - LOG_PACKING_RATIO;
  static constexpr int COUNTER_LANES = 1 << LOG_COUNTER_LANES;
  static constexpr int PADDED_COUNTER_LANES = COUNTER_LANES + 1;

  packed_counter_type cached_segment[PADDED_COUNTER_LANES];
  uint8_t *_local_memory;
};

template <typename T, typename U> struct base_traits {

  static __dpct_inline__ U twiddle_in(U key) {
    throw std::runtime_error("Not implemented");
  }
  static __dpct_inline__ U twiddle_out(U key) {
    throw std::runtime_error("Not implemented");
  }
};

template <typename U> struct base_traits<uint32_t, U> {
  static __dpct_inline__ U twiddle_in(U key) { return key; }
  static __dpct_inline__ U twiddle_out(U key) { return key; }
};

template <typename U> struct base_traits<int, U> {
  static constexpr U HIGH_BIT = U(1) << ((sizeof(U) * 8) - 1);
  static __dpct_inline__ U twiddle_in(U key) { return key ^ HIGH_BIT; }
  static __dpct_inline__ U twiddle_out(U key) { return key ^ HIGH_BIT; }
};

template <typename U> struct base_traits<float, U> {
  static constexpr U HIGH_BIT = U(1) << ((sizeof(U) * 8) - 1);
  static __dpct_inline__ U twiddle_in(U key) {
    U mask = (key & HIGH_BIT) ? U(-1) : HIGH_BIT;
    return key ^ mask;
  }
  static __dpct_inline__ U twiddle_out(U key) {
    U mask = (key & HIGH_BIT) ? HIGH_BIT : U(-1);
    return key ^ mask;
  }
};

template <typename U> struct base_traits<sycl::half, U> {
  static constexpr U HIGH_BIT = U(1) << ((sizeof(U) * 8) - 1);
  static __dpct_inline__ U twiddle_in(U key) {
    U mask = (key & HIGH_BIT) ? U(-1) : HIGH_BIT;
    return key ^ mask;
  }
  static __dpct_inline__ U twiddle_out(U key) {
    U mask = (key & HIGH_BIT) ? HIGH_BIT : U(-1);
    return key ^ mask;
  }
};

template <typename T> struct traits : base_traits<T, T> {};
template <> struct traits<uint32_t> : base_traits<uint32_t, uint32_t> {};
template <> struct traits<int> : base_traits<int, uint32_t> {};
template <> struct traits<float> : base_traits<float, uint32_t> {};
template <> struct traits<sycl::half> : base_traits<sycl::half, uint16_t> {};

template <int N> struct power_of_two {
  enum { VALUE = ((N & (N - 1)) == 0) };
};

__dpct_inline__ uint32_t shr_add(uint32_t x, uint32_t shift, uint32_t addend) {
  return (x >> shift) + addend;
}

} // namespace detail
} // namespace group
} // namespace dpct
#endif // !__DPCT_DETAIL_GROUP_UTILS_DETAIL_HPP__
