//==---- dpcpp_extensions.h ------------------*- C++ -*---------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------===//

#ifndef __DPCT_DPCPP_EXTENSIONS_H__
#define __DPCT_DPCPP_EXTENSIONS_H__

#include <stdexcept>

#ifdef SYCL_EXT_ONEAPI_USER_DEFINED_REDUCTIONS
#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>
#endif

#include "functional.h"

namespace dpct {
namespace group {
namespace detail {

template <typename... _Args>
constexpr auto __reduce_over_group(_Args... __args) {
  return sycl::reduce_over_group(__args...);
}

template <typename... _Args> constexpr auto __group_broadcast(_Args... __args) {
  return sycl::group_broadcast(__args...);
}

template <typename... _Args>
constexpr auto __exclusive_scan_over_group(_Args... __args) {
  return sycl::exclusive_scan_over_group(__args...);
}

template <typename... _Args>
constexpr auto __inclusive_scan_over_group(_Args... __args) {
  return sycl::inclusive_scan_over_group(__args...);
}

} // end namespace detail

/// Perform an exclusive scan over the values of inputs from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param inputs Pointer to the input data for the scan operation.
/// \param outputs Pointer to the location where scan results will be stored.
/// \param init initial value of the scan result.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan.
template <typename Item, typename T, class BinaryOperation,
          int VALUES_PER_THREAD>
__dpct_inline__ void
exclusive_scan(const Item &item, T (&inputs)[VALUES_PER_THREAD],
               T (&outputs)[VALUES_PER_THREAD], T init,
               BinaryOperation binary_op) {
  T result = inputs[0];

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    result = binary_op(result, inputs[i]);
  }

  T exclusive_result =
      detail::__exclusive_scan_over_group(item.get_group(), result, binary_op);

  T input = inputs[0];
  if (item.get_local_linear_id() == 0) {
    outputs[0] = init;
  } else {
    outputs[0] = exclusive_result;
  }

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    T output = binary_op(input, outputs[i - 1]);
    input = inputs[i];
    outputs[i] = output;
  }
}

/// Perform an exclusive scan over the values of input from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Input data for the scan operation.
/// \param init initial value of the scan result.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param group_aggregate group-wide aggregate of all inputs
/// in the work-items of the group. \returns exclusive scan of the first i
/// work-items where item is the i-th work item.
template <typename Item, typename T, class BinaryOperation>
__dpct_inline__ T
exclusive_scan(const Item &item, T input, T init, BinaryOperation binary_op,
               T &group_aggregate) {
  T output = detail::__exclusive_scan_over_group(item.get_group(), input, init,
                                                 binary_op);
  if (item.get_local_linear_id() == item.get_local_range().size() - 1) {
    group_aggregate = binary_op(output, input);
  }

  group_aggregate = detail::__group_broadcast(
      item.get_group(), group_aggregate, item.get_local_range().size() - 1);
  return output;
}

/// Perform an exclusive scan over the values of input from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Input data for the scan operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param prefix_callback_op functor invoked by the first
/// work-item in the group that returns the
///        initial value in the resulting scan of the work-items in the group.
/// \returns exclusive scan of the input elements assigned to work-items in the
/// group.
template <typename Item, typename T, class BinaryOperation,
          class GroupPrefixCallbackOperation>
__dpct_inline__ T
exclusive_scan(const Item &item, T input, BinaryOperation binary_op,
               GroupPrefixCallbackOperation &prefix_callback_op) {
  T group_aggregate;

  T output =
      detail::__exclusive_scan_over_group(item.get_group(), input, binary_op);
  if (item.get_local_linear_id() == item.get_local_range().size() - 1) {
    group_aggregate = binary_op(output, input);
  }

  group_aggregate = detail::__group_broadcast(
      item.get_group(), group_aggregate, item.get_local_range().size() - 1);

  T group_prefix = prefix_callback_op(group_aggregate);
  if (item.get_local_linear_id() == 0) {
    output = group_prefix;
  } else {
    output = binary_op(group_prefix, output);
  }

  return output;
}

/// Perform a reduction of the data elements assigned to all threads in the
/// group.
///
/// \param item A work-item in a group.
/// \param inputs Pointer to the input data for the reduce operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \returns value of the reduction using binary_op
template <typename Item, typename T, class BinaryOperation,
          int VALUES_PER_THREAD>
__dpct_inline__ T
reduce(Item item, T (&inputs)[VALUES_PER_THREAD], BinaryOperation binary_op) {
  T result = inputs[0];

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; i++) {
    result = binary_op(result, inputs[i]);
  }
  return detail::__reduce_over_group(item.get_group(), result, binary_op);
}

/// Perform a reduction on a limited number of the work items in a subgroup
///
/// \param item A work-item in a group.
/// \param value value per work item which is to be reduced
/// \param items_to_reduce num work items at the start of the subgroup to reduce
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \returns value of the reduction using binary_op
template <typename Item, typename T, class BinaryOperation>
__dpct_inline__
typename ::std::enable_if_t<sycl::has_known_identity_v<BinaryOperation, T>, T>
reduce_over_partial_group(const Item &item, const T &value,
                          const ::std::uint16_t &items_to_reduce,
                          BinaryOperation binary_op) {
  T value_temp = (item.get_local_linear_id() < items_to_reduce)
                     ? value
                     : sycl::known_identity_v<BinaryOperation, T>;
  return detail::__reduce_over_group(item.get_sub_group(), value_temp,
                                     binary_op);
}

/// Perform an inclusive scan over the values of inputs from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param inputs Pointer to the input data for the scan operation.
/// \param outputs Pointer to the location where scan results will be stored.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \returns inclusive scan of the input elements assigned to
/// work-items in the group.
template <typename Item, typename T, class BinaryOperation,
          int VALUES_PER_THREAD>
__dpct_inline__ void
inclusive_scan(const Item &item, T (&inputs)[VALUES_PER_THREAD],
               T (&outputs)[VALUES_PER_THREAD], BinaryOperation binary_op) {
  T result = inputs[0];

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    result = binary_op(result, inputs[i]);
  }

  T exclusive_result =
      detail::__exclusive_scan_over_group(item.get_group(), result, binary_op);

  if (item.get_local_linear_id() == 0) {
    outputs[0] = inputs[0];
  } else {
    outputs[0] = binary_op(inputs[0], exclusive_result);
  }

#pragma unroll
  for (int i = 1; i < VALUES_PER_THREAD; ++i) {
    outputs[i] = binary_op(inputs[i], outputs[i - 1]);
  }
}

/// Perform an inclusive scan over the values of inputs from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Pointer to the input data for the scan operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param group_aggregate group-wide aggregate of all inputs
/// in the work-items of the group. \returns inclusive scan of the input
/// elements assigned to work-items in the group.
template <typename Item, typename T, class BinaryOperation>
__dpct_inline__ T inclusive_scan(const Item &item, T input,
                                                BinaryOperation binary_op,
                                                T &group_aggregate) {
  T output =
      detail::__inclusive_scan_over_group(item.get_group(), input, binary_op);
  if (item.get_local_linear_id() == item.get_local_range().size() - 1) {
    group_aggregate = output;
  }

  group_aggregate = detail::__group_broadcast(
      item.get_group(), group_aggregate, item.get_local_range().size() - 1);
  return output;
}

/// Perform an inclusive scan over the values of input from all work-items in
/// the group using the operator binary_op, which must be one of the SYCL 2020
/// group algorithms library function objects.
///
/// \param item A work-item in a group.
/// \param input Input data for the scan operation.
/// \param binary_op functor that implements the binary operation used to
/// perform the scan. \param prefix_callback_op functor invoked by the first
/// work-item in the group that returns the
///        initial value in the resulting scan of the work-items in the group.
/// \returns inclusive scan of the input elements assigned to work-items in the
/// group.
template <typename Item, typename T, class BinaryOperation,
          class GroupPrefixCallbackOperation>
__dpct_inline__ T
inclusive_scan(const Item &item, T input, BinaryOperation binary_op,
               GroupPrefixCallbackOperation &prefix_callback_op) {
  T group_aggregate;

  T output = inclusive_scan(item, input, binary_op, group_aggregate);
  T group_prefix = prefix_callback_op(group_aggregate);

  return binary_op(group_prefix, output);
}

} // namespace group

namespace device {

namespace detail {

template <typename... _Args> constexpr auto __joint_reduce(_Args... __args) {
  return sycl::joint_reduce(__args...);
}

} // namespace detail

/// Perform a reduce on each of the segments specified within data stored on
/// the device.
///
/// \param queue Command queue used to access device used for reduction
/// \param inputs Pointer to the data elements on the device to be reduced
/// \param outputs Pointer to the storage where the reduced value for each
/// segment will be stored \param segment_count number of segments to be reduced
/// \param begin_offsets Pointer to the set of indices that are the first
/// element in each segment \param end_offsets Pointer to the set of indices
/// that are one past the last element in each segment \param binary_op functor
/// that implements the binary operation used to perform the scan. \param init
/// initial value of the reduction for each segment.
template <int GROUP_SIZE, typename T, typename OffsetT, class BinaryOperation>
void segmented_reduce(sycl::queue queue, T *inputs, T *outputs,
                      size_t segment_count, OffsetT *begin_offsets,
                      OffsetT *end_offsets, BinaryOperation binary_op, T init) {

  sycl::range<1> global_size(segment_count * GROUP_SIZE);
  sycl::range<1> local_size(GROUP_SIZE);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          OffsetT segment_begin = begin_offsets[item.get_group_linear_id()];
          OffsetT segment_end = end_offsets[item.get_group_linear_id()];
          if (segment_begin == segment_end) {
            if (item.get_local_linear_id() == 0) {
              outputs[item.get_group_linear_id()] = init;
            }
            return;
          }

          sycl::multi_ptr<T, sycl::access::address_space::global_space>
              input_ptr = inputs;
          T group_aggregate = detail::__joint_reduce(
              item.get_group(), input_ptr + segment_begin,
              input_ptr + segment_end, init, binary_op);

          if (item.get_local_linear_id() == 0) {
            outputs[item.get_group_linear_id()] = group_aggregate;
          }
        });
  });
}


#ifdef SYCL_EXT_ONEAPI_USER_DEFINED_REDUCTIONS

namespace experimental {
namespace detail {
template <typename _Tp, typename... _Ts> struct __is_any {
  constexpr static bool value = std::disjunction_v<
      std::is_same<std::remove_cv_t<_Tp>, std::remove_cv_t<_Ts>>...>;
};

template <typename _Tp, typename _Bp> struct __in_native_op_list {
  constexpr static bool value =
      __is_any<_Bp, sycl::plus<_Tp>, sycl::bit_or<_Tp>, sycl::bit_xor<_Tp>,
               sycl::bit_and<_Tp>, sycl::maximum<_Tp>, sycl::minimum<_Tp>,
               sycl::multiplies<_Tp>>::value;
};

template <typename _Tp, typename _Bp> struct __is_native_op {
  constexpr static bool value = __in_native_op_list<_Tp, _Bp>::value ||
                                __in_native_op_list<void, _Bp>::value;
};

} // namespace detail

/// Perform a reduce on each of the segments specified within data stored on
/// the device. Compared with dpct::device::segmented_reduce, this experimental
/// feature support user define reductions.
///
/// \param queue Command queue used to access device used for reduction
/// \param inputs Pointer to the data elements on the device to be reduced
/// \param outputs Pointer to the storage where the reduced value for each
/// segment will be stored \param segment_count number of segments to be reduced
/// \param begin_offsets Pointer to the set of indices that are the first
/// element in each segment \param end_offsets Pointer to the set of indices
/// that are one past the last element in each segment \param binary_op functor
/// that implements the binary operation used to perform the scan. \param init
/// initial value of the reduction for each segment.
template <int GROUP_SIZE, typename T, typename OffsetT, class BinaryOperation>
void segmented_reduce(sycl::queue queue, T *inputs, T *outputs,
                      size_t segment_count, OffsetT *begin_offsets,
                      OffsetT *end_offsets, BinaryOperation binary_op, T init) {

  sycl::range<1> global_size(segment_count * GROUP_SIZE);
  sycl::range<1> local_size(GROUP_SIZE);

  if constexpr (!detail::__is_native_op<T, BinaryOperation>::value) {
    queue.submit([&](sycl::handler &cgh) {
      size_t temp_memory_size = GROUP_SIZE * sizeof(T);
      auto scratch = sycl::local_accessor<std::byte, 1>(temp_memory_size, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(global_size, local_size),
          [=](sycl::nd_item<1> item) {
            OffsetT segment_begin = begin_offsets[item.get_group_linear_id()];
            OffsetT segment_end = end_offsets[item.get_group_linear_id()];
            if (segment_begin == segment_end) {
              if (item.get_local_linear_id() == 0) {
                outputs[item.get_group_linear_id()] = init;
              }
              return;
            }
            // Create a handle that associates the group with an allocation it
            // can use
            auto handle =
                sycl::ext::oneapi::experimental::group_with_scratchpad(
                    item.get_group(),
                    sycl::span(&scratch[0], temp_memory_size));
            T group_aggregate = sycl::ext::oneapi::experimental::joint_reduce(
                handle, inputs + segment_begin, inputs + segment_end, init,
                binary_op);
            if (item.get_local_linear_id() == 0) {
              outputs[item.get_group_linear_id()] = group_aggregate;
            }
          });
    });
  } else {
    dpct::device::segmented_reduce<GROUP_SIZE>(queue, inputs, outputs,
                                               segment_count, begin_offsets,
                                               end_offsets, binary_op, init);
  }
}
} // namespace experimental

#endif // SYCL_EXT_ONEAPI_USER_DEFINED_REDUCTIONS


} // namespace device
} // namespace dpct

#endif
