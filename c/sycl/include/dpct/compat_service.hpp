//==---- compat_service.hpp -----------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
// This file contains some compatibility service APIs which are used by all
// library utility helper functions. The compatibility service APIs are based on
// either dpct implementation or syclcompat implementation, which is controlled
// by the macro USE_DPCT_HELPER.
//===----------------------------------------------------------------------===//

#ifndef __DPCT_COMPAT_SERVICE_HPP__
#define __DPCT_COMPAT_SERVICE_HPP__

namespace dpct {
namespace cs {
#if USE_DPCT_HELPER
namespace ns = ::dpct;
using memcpy_direction = ::dpct::memcpy_direction;
template <class... Args> using kernel_name = dpct_kernel_name<Args...>;
using byte_t = ::dpct::byte_t;
#else
namespace ns = ::syclcompat;
using memcpy_direction = ::syclcompat::detail::memcpy_direction;
template <class... Args> using kernel_name = syclcompat_kernel_name<Args...>;
#ifndef __dpct_inline__
#define __dpct_inline__ __syclcompat_inline__
#endif
using byte_t = ::syclcompat::byte_t;
#endif

using ns::get_current_device;
using ns::get_default_context;
using ns::queue_ptr;

namespace detail {
using ns::detail::get_pointer_attribute;
using ns::detail::pointer_access_attribute;
} // namespace detail

inline sycl::queue &get_default_queue() {
#if USE_DPCT_HELPER
  return ::dpct::get_default_queue();
#else
  return *::syclcompat::detail::dev_mgr::instance()
              .current_device()
              .default_queue();
#endif
}

inline sycl::event
memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
       memcpy_direction direction = memcpy_direction::automatic,
       const std::vector<sycl::event> &dep_events = {}) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction,
                                     dep_events);
#else
  return ::syclcompat::detail::memcpy(q, to_ptr, from_ptr, size, dep_events);
#endif
}

inline std::vector<sycl::event>
memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t to_pitch,
       size_t from_pitch, size_t x, size_t y,
       memcpy_direction direction = memcpy_direction::automatic,
       const std::vector<sycl::event> &dep_events = {}) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_memcpy(
      q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
      sycl::range<3>(from_pitch, y, 1), sycl::id<3>(0, 0, 0),
      sycl::id<3>(0, 0, 0), sycl::range<3>(x, y, 1), direction, dep_events);
#else
  return ::syclcompat::detail::memcpy(
      q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
      sycl::range<3>(from_pitch, y, 1), sycl::id<3>(0, 0, 0),
      sycl::id<3>(0, 0, 0), sycl::range<3>(x, y, 1), dep_events);
#endif
}

inline void *malloc(size_t size, sycl::queue q = get_default_queue()) {
#if USE_DPCT_HELPER
  return ::dpct::dpct_malloc(size, q);
#else
  return ::syclcompat::malloc(size, q);
#endif
}

template <typename valueT>
inline sycl::event fill(sycl::queue &q, void *dev_ptr, valueT value,
                        size_t size) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_memset<valueT>(q, dev_ptr, value, size);
#else
  return ::syclcompat::detail::fill<valueT>(q, dev_ptr, value, size);
#endif
}

inline void free(void *to_ptr, sycl::queue q) {
#if USE_DPCT_HELPER
  return ::dpct::detail::dpct_free(to_ptr, q);
#else
  return ::syclcompat::free(to_ptr, q);
#endif
}

inline sycl::event enqueue_free(const std::vector<void *> &pointers,
                                const std::vector<sycl::event> &events,
                                sycl::queue q = get_default_queue()) {
#if USE_DPCT_HELPER
  return ::dpct::detail::async_dpct_free(pointers, events, q);
#else
  return ::syclcompat::enqueue_free(pointers, events, q);
#endif
}

} // namespace cs
} // namespace dpct

#endif // __DPCT_COMPAT_SERVICE_HPP__
