//==---- memory.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MEMORY_HPP__
#define __DPCT_MEMORY_HPP__

#include "device.hpp"
#include <sycl/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <map>
#include <utility>
#include <thread>
#include <type_traits>

#if defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif

namespace dpct {
enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};
enum memory_region {
  global = 0, // device global memory
  constant,   // device constant memory
  local,      // device local memory
  shared,     // memory which can be accessed by host and device
};

typedef uint8_t byte_t;

/// Buffer type to be used in Memory Management runtime.
typedef sycl::buffer<byte_t> buffer_t;

class image_matrix;

/// dpct accessor used as device function parameter.
template <class T, memory_region Memory, size_t Dimension> class accessor;
static inline void dpct_free(void *ptr, sycl::queue &q);
template <typename T> static sycl::buffer<T> get_buffer(const void *ptr);
static buffer_t get_buffer(const void *ptr);

/// Pitched 2D/3D memory data.
class pitched_data {
public:
  pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
  pitched_data(void *data, size_t pitch, size_t x, size_t y)
      : _data(data), _pitch(pitch), _x(x), _y(y) {}

  void *get_data_ptr() { return _data; }
  void set_data_ptr(const void *data) { _data = const_cast<void *>(data); }

  size_t get_pitch() { return _pitch; }
  void set_pitch(size_t pitch) { _pitch = pitch; }

  size_t get_x() { return _x; }
  void set_x(size_t x) { _x = x; };

  size_t get_y() { return _y; }
  void set_y(size_t y) { _y = y; }

private:
  void *_data;
  size_t _pitch, _x, _y;
};

namespace experimental {
class image_mem_wrapper;
} // namespace experimental

/// Memory copy parameters for 2D/3D memory data.
struct memcpy_parameter {
  struct data_wrapper {
    pitched_data pitched{};
    sycl::id<3> pos{};
    size_t pos_x_in_bytes{0};
    int dev_id{0};
#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
    experimental::image_mem_wrapper *image_bindless{nullptr};
#endif
    image_matrix *image{nullptr};
  };
  data_wrapper from{};
  data_wrapper to{};
  sycl::range<3> size{};
  size_t size_x_in_bytes{0};
  memcpy_direction direction{memcpy_direction::automatic};
};

} // namespace dpct

#include "detail/memory_detail.hpp"
namespace dpct {

#ifdef DPCT_USM_LEVEL_NONE
/// Check if the pointer \p ptr represents device pointer or not.
///
/// \param ptr The pointer to be checked.
/// \returns true if \p ptr is a device pointer.
template<class T>
static inline bool is_device_ptr(T ptr) {
  if constexpr (std::is_pointer<T>::value) {
    return detail::mem_mgr::instance().is_device_ptr(ptr);
  }
  return false;
}
/// Get the base address of the allocated memory for the given pointer \p ptr.
template <class T> static inline void *get_base_addr(T ptr) {
  if constexpr (std::is_pointer<T>::value) {
    return detail::mem_mgr::instance().get_base_addr(ptr);
  }
  return nullptr;
}
#endif

/// Get the buffer and the offset of a piece of memory pointed to by \p ptr.
///
/// \param ptr Pointer to a piece of memory.
/// If NULL is passed as an argument, an exception will be thrown.
/// \returns a pair containing both the buffer and the offset.
static std::pair<buffer_t, size_t> get_buffer_and_offset(const void *ptr) {
  if (ptr) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    size_t offset = (byte_t *)ptr - alloc.alloc_ptr;
    return std::make_pair(alloc.buffer, offset);
  } else {
    throw std::runtime_error(
        "NULL pointer argument in get_buffer_and_offset function is invalid");
  }
}

/// Get the data pointed from \p ptr as a 1D buffer reinterpreted as type T.
template <typename T> static sycl::buffer<T> get_buffer(const void *ptr) {
  if (!ptr)
    return sycl::buffer<T>(sycl::range<1>(0));
  auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
  return alloc.buffer.reinterpret<T>(
      sycl::range<1>(alloc.size / sizeof(T)));
}

/// Get the buffer of a piece of memory pointed to by \p ptr.
///
/// \param ptr Pointer to a piece of memory.
/// \returns the buffer.
static buffer_t get_buffer(const void *ptr) {
  return detail::mem_mgr::instance().translate_ptr(ptr).buffer;
}

/// A wrapper class contains an accessor and an offset.
template <typename PtrT,
          sycl::access_mode accessMode = sycl::access_mode::read_write>
class access_wrapper {
  sycl::accessor<byte_t, 1, accessMode> accessor;
  size_t offset;

public:
  /// Construct the accessor wrapper for memory pointed by \p ptr.
  ///
  /// \param ptr Pointer to memory.
  /// \param cgh The command group handler.
  access_wrapper(const void *ptr, sycl::handler &cgh)
      : accessor(get_buffer(ptr).get_access<accessMode>(cgh)), offset(0) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    offset = (byte_t *)ptr - alloc.alloc_ptr;
  }
  template <typename U = PtrT>
  access_wrapper(
      PtrT ptr, sycl::handler &cgh,
      typename std::enable_if_t<!std::is_same_v<
          std::remove_cv_t<std::remove_reference_t<U>>, void *>> * = 0)
      : access_wrapper((const void *)ptr, cgh) {}

  /// Get the device pointer.
  ///
  /// \returns a device pointer with offset.
  PtrT get_raw_pointer() const { return (PtrT)(&accessor[0] + offset); }
};

/// Get the accessor for memory pointed by \p ptr.
///
/// \param ptr Pointer to memory.
/// If NULL is passed as an argument, an exception will be thrown.
/// \param cgh The command group handler.
/// \returns an accessor.
template <typename T,
          sycl::access_mode accessMode = sycl::access_mode::read_write>
static auto get_access(const T *ptr, sycl::handler &cgh) {
  if (ptr) {
    auto alloc = detail::mem_mgr::instance().translate_ptr(ptr);
    if constexpr (std::is_same_v<std::remove_reference_t<T>, void>)
      return alloc.buffer.template get_access<accessMode>(cgh);
    else
      return alloc.buffer
          .template reinterpret<T>(sycl::range<1>(alloc.size / sizeof(T)))
          .template get_access<accessMode>(cgh);
  } else {
    throw std::runtime_error(
        "NULL pointer argument in get_access function is invalid");
  }
}

/// Allocate memory block on the device.
/// \param num_bytes Number of bytes to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
template <typename T>
static inline void *dpct_malloc(T num_bytes,
                                sycl::queue &q = get_default_queue()) {
  return detail::dpct_malloc(static_cast<size_t>(num_bytes), q);
}

/// Get the host pointer from a buffer that is mapped to virtual pointer ptr.
/// \param ptr Virtual Pointer mapped to device buffer
/// \returns A host pointer
template <typename T> static inline T *get_host_ptr(const void *ptr) {
  auto BufferOffset = get_buffer_and_offset(ptr);
  auto host_ptr =
      BufferOffset.first.get_host_access()
          .get_pointer();
  return (T *)(host_ptr + BufferOffset.second);
}

/// Allocate memory block for 3D array on the device.
/// \param size Size of the memory block, in bytes.
/// \param q Queue to execute the allocate task.
/// \returns A pitched_data object which stores the memory info.
static inline pitched_data
dpct_malloc(sycl::range<3> size, sycl::queue &q = get_default_queue()) {
  pitched_data pitch(nullptr, 0, size.get(0), size.get(1));
  size_t pitch_size;
  pitch.set_data_ptr(detail::dpct_malloc(pitch_size, size.get(0), size.get(1),
                                         size.get(2), q));
  pitch.set_pitch(pitch_size);
  return pitch;
}

/// Allocate memory block for 2D array on the device.
/// \param [out] pitch Aligned size of x in bytes.
/// \param x Range in dim x.
/// \param y Range in dim y.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y,
                                sycl::queue &q = get_default_queue()) {
  return detail::dpct_malloc(pitch, x, y, 1, q);
}

/// free
/// \param ptr Point to free.
/// \param q Queue to execute the free task.
/// \returns no return value.
static inline void dpct_free(void *ptr,
                             sycl::queue &q = get_default_queue()) {
#ifndef DPCT_USM_LEVEL_NONE
  dpct::get_device(dpct::get_device_id(q.get_device())).queues_wait_and_throw();
#endif
  detail::dpct_free(ptr, q);
}

/// Free the device memory pointed by a batch of pointers in \p pointers which
/// are related to \p q after \p events completed.
///
/// \param pointers The pointers point to the device memory requested to be freed.
/// \param events The events to be waited.
/// \param q The sycl::queue the memory relates to.
inline void async_dpct_free(const std::vector<void *> &pointers,
                            const std::vector<sycl::event> &events,
                            sycl::queue &q = get_default_queue()) {
  detail::async_dpct_free(pointers, events, q);
}

/// Synchronously copies \p size bytes from the address specified by \p from_ptr
/// to the address specified by \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device,
/// \a device_to_host, \a device_to_device or \a automatic. The function will
/// return after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                        memcpy_direction direction = automatic,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction).wait();
}

/// Synchronously copies \p size bytes from the address specified by \p from_ptr
/// on device specified by \p from_dev_id to the address specified by \p to_ptr
/// on device specified by \p to_dev_id. The function will return after the copy
/// is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_dev_id Destination device ID.
/// \param from_ptr Pointer to source memory address.
/// \param from_dev_id Source device ID.
/// \param size Number of bytes to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void dpct_memcpy(void *to_ptr, int to_dev_id, const void *from_ptr,
                        int from_dev_id, size_t size,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, to_dev_id, from_ptr, from_dev_id, size).wait();
}

/// Asynchronously copies \p size bytes from the address specified by \p
/// from_ptr to the address specified by \p to_ptr. The value of \p direction is
/// used to set the copy direction, it can be \a host_to_host, \a
/// host_to_device, \a device_to_host, \a device_to_device or \a automatic. The
/// return of the function does NOT guarantee the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void async_dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                              memcpy_direction direction = automatic,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
}

/// Asynchronously copies \p size bytes from the address specified by \p
/// from_ptr on device specified by \p from_dev_id to the address specified by
/// \p to_ptr on device specified by \p to_dev_id. The return of the function
/// does NOT guarantee the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_dev_id Destination device ID.
/// \param from_ptr Pointer to source memory address.
/// \param from_dev_id Source device ID.
/// \param size Number of bytes to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void async_dpct_memcpy(void *to_ptr, int to_dev_id, const void *from_ptr,
                              int from_dev_id, size_t size,
                              sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, to_dev_id, from_ptr, from_dev_id, size);
}

/// Synchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device, \a
/// device_to_host, \a device_to_device or \a automatic. The function will
/// return after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void dpct_memcpy(void *to_ptr, size_t to_pitch,
                               const void *from_ptr, size_t from_pitch,
                               size_t x, size_t y,
                               memcpy_direction direction = automatic,
                               sycl::queue &q = dpct::get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch,
                                            from_pitch, x, y, direction));
}

/// Asynchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// \p from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The value of \p direction is used to
/// set the copy direction, it can be \a host_to_host, \a host_to_device, \a
/// device_to_host, \a device_to_device or \a automatic. The return of the
/// function does NOT guarantee the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void
async_dpct_memcpy(void *to_ptr, size_t to_pitch, const void *from_ptr,
                  size_t from_pitch, size_t x, size_t y,
                  memcpy_direction direction = automatic,
                  sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                      direction);
}

/// Synchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specified by \p size.
/// The value of \p direction is used to set the copy direction, it can be \a
/// host_to_host, \a host_to_device, \a device_to_host, \a device_to_device or
/// \a automatic. The function will return after the copy is completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void dpct_memcpy(pitched_data to, sycl::id<3> to_pos,
                               pitched_data from, sycl::id<3> from_pos,
                               sycl::range<3> size,
                               memcpy_direction direction = automatic,
                               sycl::queue &q = dpct::get_default_queue()) {
  sycl::event::wait(
      detail::dpct_memcpy(q, to, to_pos, from, from_pos, size, direction));
}

/// Asynchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specified by \p size.
/// The value of \p direction is used to set the copy direction, it can be \a
/// host_to_host, \a host_to_device, \a device_to_host, \a device_to_device or
/// \a automatic. The return of the function does NOT guarantee the copy is
/// completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param direction Direction of the copy.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void
async_dpct_memcpy(pitched_data to, sycl::id<3> to_pos, pitched_data from,
                  sycl::id<3> from_pos, sycl::range<3> size,
                  memcpy_direction direction = automatic,
                  sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, to, to_pos, from, from_pos, size, direction);
}

/// Synchronously copies 2D/3D memory data specified by \p param . The function
/// will return after the copy is completed.
///
/// \param param Memory copy parameters.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void dpct_memcpy(const memcpy_parameter &param,
                               sycl::queue &q = dpct::get_default_queue()) {
  sycl::event::wait(detail::dpct_memcpy(q, param));
}

/// Asynchronously copies 2D/3D memory data specified by \p param . The return
/// of the function does NOT guarantee the copy is completed.
///
/// \param param Memory copy parameters.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void async_dpct_memcpy(const memcpy_parameter &param,
                                     sycl::queue &q = get_default_queue()) {
  detail::dpct_memcpy(q, param);
}

/**
 * @brief Sets 1 byte data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q synchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void dpct_memset(void *dev_ptr, int value, size_t size,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, dev_ptr, value, size).wait();
}

/**
 * @brief Sets 2 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q synchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void dpct_memset_d16(void *dev_ptr, unsigned short value, size_t size,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size).wait();
}
/**
 * @brief Sets 4 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q synchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void dpct_memset_d32(void *dev_ptr, unsigned int value, size_t size,
                        sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size).wait();
}

/**
 * @brief Sets 1 byte data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q asynchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void async_dpct_memset(void *dev_ptr, int value, size_t size,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, dev_ptr, value, size);
}
/**
 * @brief Sets 2 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q asynchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void async_dpct_memset_d16(void *dev_ptr, unsigned short value, size_t size,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size);
}
/**
 * @brief Sets 4 bytes data \p value to the first \p size elements starting from
 * \p dev_ptr in \p q asynchronously.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @param [in] q The queue in which the operation is done.
 */
static void async_dpct_memset_d32(void *dev_ptr, unsigned int value, size_t size,
                              sycl::queue &q = dpct::get_default_queue()) {
  detail::dpct_memset(q, dev_ptr, value, size);
}

/**
 * @brief Sets 1 byte data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * synchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset(void *ptr, size_t pitch, int val, size_t x,
                               size_t y,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset<unsigned char>(q, ptr, pitch, val, x, y));
}
/**
 * @brief Sets 2 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * synchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset_d16(void *ptr, size_t pitch, unsigned short val, size_t x,
                               size_t y,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset(q, ptr, pitch, val, x, y));
}
/**
 * @brief Sets 4 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * synchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset_d32(void *ptr, size_t pitch, unsigned int val, size_t x,
                               size_t y,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset(q, ptr, pitch, val, x, y));
}

/**
 * @brief Sets 1 byte data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * asynchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset(void *ptr, size_t pitch, int val, size_t x,
                                     size_t y,
                                     sycl::queue &q = get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, ptr, pitch, val, x, y);
}

/**
 * @brief Sets 2 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * asynchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset_d16(void *ptr, size_t pitch,
                                         unsigned short val, size_t x, size_t y,
                                         sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, ptr, pitch, val, x, y);
}

/**
 * @brief Sets 4 bytes data \p val to the pitched 2D memory region pointed by \p ptr in \p q
 * asynchronously.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset_d32(void *ptr, size_t pitch,
                                         unsigned int val, size_t x, size_t y,
                                         sycl::queue &q = get_default_queue()) {
  detail::dpct_memset(q, ptr, pitch, val, x, y);
}

/**
 * @brief Sets 1 byte data \p value to the 3D memory region pointed by \p data in \p q
 * synchronously.
 * @param [in] data Pointer to the pitched device memory region.
 * @param [in] value The value to be set.
 * @param [in] size 3D memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void dpct_memset(pitched_data pitch, int val,
                               sycl::range<3> size,
                               sycl::queue &q = get_default_queue()) {
  sycl::event::wait(detail::dpct_memset<unsigned char>(q, pitch, val, size));
}

/**
 * @brief Sets 1 byte data \p value to the 3D memory region pointed by \p data in \p q
 * asynchronously.
 * @param [in] data Pointer to the pitched device memory region.
 * @param [in] value The value to be set.
 * @param [in] size 3D memory region by number of elements.
 * @param [in] q The queue in which the operation is done.
 */
static inline void async_dpct_memset(pitched_data pitch, int val,
                                     sycl::range<3> size,
                                     sycl::queue &q = get_default_queue()) {
  detail::dpct_memset<unsigned char>(q, pitch, val, size);
}

namespace experimental {
typedef sycl::ext::oneapi::experimental::physical_mem *physical_mem_ptr;

struct mem_location {
  int id = 0;
  // Location type. Value 1 means device location, and thus, id is a device id.
  int type = 1;
};

struct mem_prop {
  mem_location location;
  int type = 1; // Memory type. Value 1 means default device memory.
};

struct mem_access_desc {
  sycl::ext::oneapi::experimental::address_access_mode flags;
  mem_location location;
};
} // namespace experimental

template <class T, memory_region Memory> class accessor<T, Memory, 3> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<3>;
  accessor(pointer_t data, const sycl::range<3> &in_range)
      : _data(data), _range(in_range) {}
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const sycl::range<3> &in_range)
      : accessor(
            acc.template get_multi_ptr<sycl::access::decorated::no>().get(),
            in_range) {}
  accessor<T, Memory, 2> operator[](size_t index) const {
    sycl::range<2> sub(_range.get(1), _range.get(2));
    return accessor<T, Memory, 2>(_data + index * sub.size(), sub);
  }

  pointer_t get_ptr() const { return _data; }

private:
  pointer_t _data;
  sycl::range<3> _range;
};
template <class T, memory_region Memory> class accessor<T, Memory, 2> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<2>;
  accessor(pointer_t data, const sycl::range<2> &in_range)
      : _data(data), _range(in_range) {}

  template <memory_region M = Memory>
  [[deprecated]] accessor(
      typename std::enable_if<M != local, const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  [[deprecated]] accessor(const accessor_t &acc, const sycl::range<2> &in_range)
      : accessor(const_cast<pointer_t>(
                     acc.template get_multi_ptr<sycl::access::decorated::no>()
                         .get()),
                 in_range) {}

  pointer_t operator[](size_t index) const {
    return _data + _range.get(1) * index;
  }

  pointer_t get_ptr() const { return _data; }

private:
  pointer_t _data;
  sycl::range<2> _range;
};

template <class T, size_t Dimension>
using global_memory = detail::device_memory<T, global, Dimension>;
template <class T, size_t Dimension>
using constant_memory = detail::device_memory<T, constant, Dimension>;
template <class T, size_t Dimension>
using shared_memory = detail::device_memory<T, shared, Dimension>;

// dpct::deprecated:: is for functionality that was introduced for compatibility
// purpose, but relies on deprecated C++ features, which are either removed or
// will be removed in the future standards.
// Direct use of deprecated functionality in this namespace should be avoided.
namespace deprecated {

template <typename T>
using usm_host_allocator = detail::deprecated::usm_allocator<T, sycl::usm::alloc::host>;

template <typename T>
using usm_device_allocator = detail::deprecated::usm_allocator<T, sycl::usm::alloc::shared>;
} // namespace deprecated

class pointer_attributes {
public:
  enum class type {
    memory_type,
    device_pointer,
    host_pointer,
    is_managed,
    device_id,
    unsupported
  };

  void init(const void *ptr,
              sycl::queue &q = dpct::get_default_queue()) {
#ifdef DPCT_USM_LEVEL_NONE
    throw std::runtime_error(
          "dpct::pointer_attributes: only works for USM pointer.");
#else
    memory_type = sycl::get_pointer_type(ptr, q.get_context());
    if (memory_type == sycl::usm::alloc::unknown) {
      device_id = -1;
      return;
    }
    device_pointer = (memory_type !=
                        sycl::usm::alloc::unknown) ? ptr : nullptr;
    host_pointer = (memory_type !=
                        sycl::usm::alloc::unknown) &&
                   (memory_type != sycl::usm::alloc::device) ? ptr : nullptr;
    sycl::device device_obj = sycl::get_pointer_device(ptr, q.get_context());
    device_id = dpct::dev_mgr::instance().get_device_id(device_obj);
#endif
  }

  // Query pointer propreties listed in attributes and store the results in data array
  static void get(unsigned int numAttributes, type *attributes,
                  void **data, device_ptr ptr) {
    pointer_attributes sycl_attributes;

    sycl_attributes.init(ptr);

    for (int i = 0; i < numAttributes; i++) {
      switch (attributes[i]) {
      case type::memory_type:
        *static_cast<int *>(data[i]) =
            static_cast<int>(sycl_attributes.get_memory_type());
        break;
      case type::device_pointer:
        *(reinterpret_cast<void **>(data[i])) =
            const_cast<void *>(sycl_attributes.get_device_pointer());
        break;
      case type::host_pointer:
        *(reinterpret_cast<void **>(data[i])) =
            const_cast<void *>(sycl_attributes.get_host_pointer());
        break;
      case type::is_managed:
        *static_cast<unsigned int *>(data[i]) =
            sycl_attributes.is_memory_shared();
        break;
      case type::device_id:
        *static_cast<unsigned int *>(data[i]) = sycl_attributes.get_device_id();
        break;
      default:
        data[i] = nullptr;
        break;
      }
    }
  }

  sycl::usm::alloc get_memory_type() {
    return memory_type;
  }

  const void *get_device_pointer() {
    return device_pointer;
  }

  const void *get_host_pointer() {
    return host_pointer;
  }

  bool is_memory_shared() {
    return memory_type == sycl::usm::alloc::shared;
  }

  unsigned int get_device_id() {
    return device_id;
  }

private:
  sycl::usm::alloc memory_type = sycl::usm::alloc::unknown;
  const void *device_pointer = nullptr;
  const void *host_pointer = nullptr;
  unsigned int device_id = -1;
};

} // namespace dpct
#endif // __DPCT_MEMORY_HPP__
