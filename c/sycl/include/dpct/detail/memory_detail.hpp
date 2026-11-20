//==---- memory_internal.hpp ---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===---------------------------------------------------------------------===//

#ifndef __DPCT_DETAIL_MEMORY_DETAIL_HPP__
#define __DPCT_DETAIL_MEMORY_DETAIL_HPP__

namespace dpct {

#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
namespace experimental {
namespace detail {
static sycl::event dpct_memcpy(const image_mem_wrapper *src,
                               const sycl::id<3> &src_id,
                               const size_t src_x_offest_byte,
                               pitched_data &dest, const sycl::id<3> &dest_id,
                               const size_t dest_x_offest_byte,
                               const sycl::range<3> &size,
                               const size_t copy_x_size_byte, sycl::queue q);
static sycl::event
dpct_memcpy(const pitched_data src, const sycl::id<3> &src_id,
            const size_t src_x_offest_byte, image_mem_wrapper *dest,
            const sycl::id<3> &dest_id, const size_t dest_x_offest_byte,
            const sycl::range<3> &size, const size_t copy_x_size_byte,
            sycl::queue q);
static sycl::event
dpct_memcpy(const image_mem_wrapper *src, const sycl::id<3> &src_id,
            const size_t src_x_offest_byte, image_mem_wrapper *dest,
            const sycl::id<3> &dest_id, const size_t dest_x_offest_byte,
            const sycl::range<3> &size, const size_t copy_x_size_byte,
            sycl::queue q);
} // namespace detail
} // namespace experimental

#endif // !SYCL_EXT_ONEAPI_BINDLESS_IMAGES

class image_matrix;

namespace detail {
static pitched_data to_pitched_data(image_matrix *image);

class mem_mgr {
  mem_mgr() {
    // Reserved address space, no real memory allocation happens here.
#if defined(__linux__)
    mapped_address_space =
        (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#elif defined(_WIN64)
    mapped_address_space = (byte_t *)VirtualAlloc(
        NULL,               // NULL specified as the base address parameter
        mapped_region_size, // Size of allocation
        MEM_RESERVE,        // Allocate reserved pages
        PAGE_NOACCESS);     // Protection = no access
#else
#error "Only support Windows and Linux."
#endif
    next_free = mapped_address_space;
  };

public:
  using buffer_id_t = int;

  struct allocation {
    buffer_t buffer;
    byte_t *alloc_ptr;
    size_t size;
  };

  ~mem_mgr() {
#if defined(__linux__)
    munmap(mapped_address_space, mapped_region_size);
#elif defined(_WIN64)
    VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#else
#error "Only support Windows and Linux."
#endif
  };

  mem_mgr(const mem_mgr &) = delete;
  mem_mgr &operator=(const mem_mgr &) = delete;
  mem_mgr(mem_mgr &&) = delete;
  mem_mgr &operator=(mem_mgr &&) = delete;

  /// Allocate
  void *mem_alloc(size_t size) {
    if (!size)
      return nullptr;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (next_free + size > mapped_address_space + mapped_region_size) {
      throw std::runtime_error(
          "dpct_malloc: out of memory for virtual memory pool");
    }
    // Allocation
    sycl::range<1> r(size);
    buffer_t buf(r);
    allocation A{buf, next_free, size};
    // Map allocation to device pointer
    void *result = next_free;
    m_map.emplace(next_free + size, A);
    // Update pointer to the next free space.
    next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

    return result;
  }

  /// Deallocate
  void mem_free(const void *ptr) {
    if (!ptr)
      return;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    m_map.erase(it);
  }

  /// map: device pointer -> allocation(buffer, alloc_ptr, size)
  allocation translate_ptr(const void *ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = get_map_iterator(ptr);
    return it->second;
  }

  void *get_base_addr(const void *ptr) {
    allocation alloc = translate_ptr(ptr);
    return alloc.alloc_ptr;
  }

  /// Check if the pointer represents device pointer or not.
  bool is_device_ptr(const void *ptr) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return (mapped_address_space <= ptr) &&
           (ptr < mapped_address_space + mapped_region_size);
  }

  /// Returns the instance of memory manager singleton.
  static mem_mgr &instance() {
    static mem_mgr m;
    return m;
  }

private:
  std::map<byte_t *, allocation> m_map;
  mutable std::mutex m_mutex;
  byte_t *mapped_address_space;
  byte_t *next_free;
  const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
  const size_t alignment = 256;
  /// This padding may be defined to some positive value to debug
  /// out of bound accesses.
  const size_t extra_padding = 0;

  std::map<byte_t *, allocation>::iterator get_map_iterator(const void *ptr) {
    auto it = m_map.upper_bound((byte_t *)ptr);
    if (it == m_map.end()) {
      // Not a virtual pointer.
      throw std::runtime_error("can not get buffer from non-virtual pointer");
    }
    const allocation &alloc = it->second;
    if (ptr < alloc.alloc_ptr) {
      // Out of bound.
      // This may happen if there's a gap between allocations due to alignment
      // or extra padding and pointer points to this gap.
      throw std::runtime_error("invalid virtual pointer");
    }
    return it;
  }
};

template <class T, memory_region Memory, size_t Dimension> class accessor;
template <memory_region Memory, class T = byte_t> class memory_traits {
public:
  static constexpr sycl::access::target target = sycl::access::target::device;
  static constexpr sycl::access_mode mode = (Memory == constant)
                                                ? sycl::access_mode::read
                                                : sycl::access_mode::read_write;
  static constexpr size_t type_size = sizeof(T);
  using value_t = typename std::remove_cv<T>::type;
  template <size_t Dimension = 1>
  using accessor_t = typename std::conditional<
      Memory == local, sycl::local_accessor<value_t, Dimension>,
      sycl::accessor<T, Dimension, mode, target>>::type;
  using pointer_t = T *;
};

static inline void *dpct_malloc(size_t size, sycl::queue &q) {
#ifdef DPCT_USM_LEVEL_NONE
  return mem_mgr::instance().mem_alloc(size * sizeof(byte_t));
#else
  return sycl::malloc_device(size, q.get_device(), q.get_context());
#endif // DPCT_USM_LEVEL_NONE
}

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                sycl::queue &q) {
  pitch = PITCH_DEFAULT_ALIGN(x);
  return dpct_malloc(pitch * y * z, q);
}

/**
 * @brief Sets \p value to the first \p size elements starting from \p dev_ptr
 * in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] dev_ptr Pointer to the virtual device memory address.
 * @param [in] value The value to be set.
 * @param [in] size Number of elements to be set to the value.
 * @return An event representing the memset operation.
 */
template <typename valueT>
static inline sycl::event dpct_memset(sycl::queue &q, void *dev_ptr,
                                      valueT value, size_t size) {
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = mem_mgr::instance();
  assert(mm.is_device_ptr(dev_ptr));
  auto alloc = mm.translate_ptr(dev_ptr);
  size_t offset = (valueT *)dev_ptr - (valueT *)alloc.alloc_ptr;

  return q.submit([&](sycl::handler &cgh) {
    auto r = sycl::range<1>(size);
    auto o = sycl::id<1>(offset);
    auto new_buffer = alloc.buffer.reinterpret<valueT>(
        sycl::range<1>(alloc.size / sizeof(valueT)));
    sycl::accessor<valueT, 1, sycl::access_mode::write,
                   sycl::access::target::device>
        acc(new_buffer, cgh, r, o);
    cgh.fill(acc, value);
  });
#else
  return q.fill(dev_ptr, value, size);
#endif // DPCT_USM_LEVEL_NONE
}

/**
 * @brief Sets \p value to the 3D memory region pointed by \p data in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] data Pointer to the pitched device memory region.
 * @param [in] value The value to be set.
 * @param [in] size 3D memory region by number of elements.
 * @return An event list representing the memset operations.
 */
template <typename valueT>
static inline std::vector<sycl::event>
dpct_memset(sycl::queue &q, dpct::pitched_data data, valueT value,
            sycl::range<3> size) {
  std::vector<sycl::event> event_list;
  size_t slice = data.get_pitch() * data.get_y();
  unsigned char *data_surface = (unsigned char *)data.get_data_ptr();
  for (size_t z = 0; z < size.get(2); ++z) {
    unsigned char *data_ptr = data_surface;
    for (size_t y = 0; y < size.get(1); ++y) {
      event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
      data_ptr += data.get_pitch();
    }
    data_surface += slice;
  }
  return event_list;
}

/**
 * @brief Sets \p val to the pitched 2D memory region pointed by \p ptr in \p q.
 * @tparam valueT The type of the element to be set.
 * @param [in] q The queue in which the operation is done.
 * @param [in] ptr Pointer to the virtual device memory.
 * @param [in] pitch The pitch size by number of elements, including padding.
 * @param [in] val The value to be set.
 * @param [in] x The width of memory region by number of elements.
 * @param [in] y The height of memory region by number of elements.
 * @return An event list representing the memset operations.
 */
template <typename valueT>
static inline std::vector<sycl::event> dpct_memset(sycl::queue &q, void *ptr,
                                                   size_t pitch, valueT val,
                                                   size_t x, size_t y) {
  return dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                     sycl::range<3>(x, y, 1));
}

enum class pointer_access_attribute {
  host_only = 0,
  device_only,
  host_device,
  end
};

static pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                      const void *ptr) {
#ifdef DPCT_USM_LEVEL_NONE
  return mem_mgr::instance().is_device_ptr(ptr)
             ? pointer_access_attribute::device_only
             : pointer_access_attribute::host_only;
#else
  switch (sycl::get_pointer_type(ptr, q.get_context())) {
  case sycl::usm::alloc::unknown:
    return pointer_access_attribute::host_only;
  case sycl::usm::alloc::device:
    return pointer_access_attribute::device_only;
  case sycl::usm::alloc::shared:
  case sycl::usm::alloc::host:
    return pointer_access_attribute::host_device;
  }
#endif
}

static memcpy_direction deduce_memcpy_direction(sycl::queue &q, void *to_ptr,
                                                const void *from_ptr,
                                                memcpy_direction dir) {
  switch (dir) {
  case memcpy_direction::host_to_host:
  case memcpy_direction::host_to_device:
  case memcpy_direction::device_to_host:
  case memcpy_direction::device_to_device:
    return dir;
  case memcpy_direction::automatic: {
    // table[to_attribute][from_attribute]
    static const memcpy_direction
        direction_table[static_cast<unsigned>(pointer_access_attribute::end)]
                       [static_cast<unsigned>(pointer_access_attribute::end)] =
                           {{memcpy_direction::host_to_host,
                             memcpy_direction::device_to_host,
                             memcpy_direction::host_to_host},
                            {memcpy_direction::host_to_device,
                             memcpy_direction::device_to_device,
                             memcpy_direction::device_to_device},
                            {memcpy_direction::host_to_host,
                             memcpy_direction::device_to_device,
                             memcpy_direction::device_to_device}};
    return direction_table[static_cast<unsigned>(get_pointer_attribute(
        q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
  }
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
}

static sycl::event
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
  if (!size)
    return sycl::event{};
#ifdef DPCT_USM_LEVEL_NONE
  auto &mm = mem_mgr::instance();
  auto real_direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);

  switch (real_direction) {
  case host_to_host:
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      cgh.host_task([=] { std::memcpy(to_ptr, from_ptr, size); });
    });
  case host_to_device: {
    auto alloc = mm.translate_ptr(to_ptr);
    size_t offset = (byte_t *)to_ptr - alloc.alloc_ptr;
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto r = sycl::range<1>(size);
      auto o = sycl::id<1>(offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::write,
                     sycl::access::target::device>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(from_ptr, acc);
    });
  }
  case device_to_host: {
    auto alloc = mm.translate_ptr(from_ptr);
    size_t offset = (byte_t *)from_ptr - alloc.alloc_ptr;
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto r = sycl::range<1>(size);
      auto o = sycl::id<1>(offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::read,
                     sycl::access::target::device>
          acc(alloc.buffer, cgh, r, o);
      cgh.copy(acc, to_ptr);
    });
  }
  case device_to_device: {
    auto to_alloc = mm.translate_ptr(to_ptr);
    auto from_alloc = mm.translate_ptr(from_ptr);
    size_t to_offset = (byte_t *)to_ptr - to_alloc.alloc_ptr;
    size_t from_offset = (byte_t *)from_ptr - from_alloc.alloc_ptr;
    return q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto r = sycl::range<1>(size);
      auto to_o = sycl::id<1>(to_offset);
      auto from_o = sycl::id<1>(from_offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::write,
                     sycl::access::target::device>
          to_acc(to_alloc.buffer, cgh, r, to_o);
      sycl::accessor<byte_t, 1, sycl::access_mode::read,
                     sycl::access::target::device>
          from_acc(from_alloc.buffer, cgh, r, from_o);
      cgh.copy(from_acc, to_acc);
    });
  }
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
#else
  return q.memcpy(to_ptr, from_ptr, size, dep_events);
#endif // DPCT_USM_LEVEL_NONE
}

// Get actual copy range and make sure it will not exceed range.
static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                    size_t pitch) {
  return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
}

static inline size_t get_offset(sycl::id<3> id, size_t slice, size_t pitch) {
  return slice * id.get(2) + pitch * id.get(1) + id.get(0);
}

// RAII for host pointer
class host_buffer {
  void *_buf;
  size_t _size;
  sycl::queue &_q;
  const std::vector<sycl::event> &_deps; // free operation depends

public:
  host_buffer(size_t size, sycl::queue &q, const std::vector<sycl::event> &deps)
      : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
  void *get_ptr() const { return _buf; }
  size_t get_size() const { return _size; }
  ~host_buffer() {
    if (_buf) {
      _q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::free(buf); });
      });
    }
  }
};

static sycl::event dpct_memcpy(sycl::queue &q, void *to_ptr, int to_dev_id,
                               const void *from_ptr, int from_dev_id,
                               size_t size) {
  if (to_dev_id == from_dev_id)
    return dpct_memcpy(q, to_ptr, from_ptr, size,
                       memcpy_direction::device_to_device);
  // Now, different device have different context, and memcpy API cannot copy
  // data between different context. So we need use host buffer to copy the data
  // between devices.
  std::vector<sycl::event> event_list;
  host_buffer buf(size, q, event_list);
  auto copy_events = dpct_memcpy(q, buf.get_ptr(), from_ptr, size,
                                 memcpy_direction::device_to_host);
  event_list.push_back(dpct::detail::dpct_memcpy(
      q, to_ptr, buf.get_ptr(), size, memcpy_direction::host_to_device,
      {copy_events}));
  return event_list[0];
}

static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            sycl::range<3> to_range, sycl::range<3> from_range,
            sycl::id<3> to_id, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events);

static inline void copy_to_device_via_host_buffer(
    sycl::queue &q, const sycl::range<3> &size, unsigned char *to_surface,
    const sycl::range<3> &to_range, const unsigned char *from_surface,
    const sycl::range<3> &from_range, std::vector<sycl::event> &event_list,
    memcpy_direction direction,
    const std::vector<sycl::event> &dep_events = {}) {
  assert(direction == device_to_host || direction == host_to_host);
  size_t to_slice = to_range.get(1) * to_range.get(0);
  host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                  event_list);
  std::vector<sycl::event> host_events;
  size_t size_slice = size.get(1) * size.get(0);
  if (to_slice == size_slice) {
    // Copy host data to a temp host buffer with the shape of target.
    host_events =
        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, direction,
                    dep_events);
  } else {
    // Copy host data to a temp host buffer with the shape of target.
    host_events =
        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, direction,
                    // If has padding data, not sure whether it is useless. So
                    // fill temp buffer with it.
                    std::vector<sycl::event>{dpct_memcpy(
                        q, buf.get_ptr(), to_surface, buf.get_size(),
                        device_to_host, dep_events)});
  }
  // Copy from temp host buffer to device with only one submit.
  event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(), buf.get_size(),
                                   host_to_device, host_events));
}

/// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
/// and \p from_range to another specified by \p to_ptr and \p to_range.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
            sycl::range<3> to_range, sycl::range<3> from_range,
            sycl::id<3> to_id, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction,
            const std::vector<sycl::event> &dep_events = {}) {
  std::vector<sycl::event> event_list;

  size_t to_slice = to_range.get(1) * to_range.get(0),
         from_slice = from_range.get(1) * from_range.get(0);
  unsigned char *to_surface =
      (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
  const unsigned char *from_surface =
      (const unsigned char *)from_ptr +
      get_offset(from_id, from_slice, from_range.get(0));

  if (to_slice == from_slice && to_slice == size.get(1) * size.get(0)) {
    return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                        direction, dep_events)};
  }
  direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
  size_t size_slice = size.get(1) * size.get(0);
  switch (direction) {
  case host_to_host:
    for (size_t z = 0; z < size.get(2); ++z) {
      unsigned char *to_ptr = to_surface;
      const unsigned char *from_ptr = from_surface;
      if (to_range.get(0) == from_range.get(0) &&
          to_range.get(0) == size.get(0)) {
        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                         direction, dep_events));
      } else {
        for (size_t y = 0; y < size.get(1); ++y) {
          event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.get(0),
                                           direction, dep_events));
          to_ptr += to_range.get(0);
          from_ptr += from_range.get(0);
        }
      }
      to_surface += to_slice;
      from_surface += from_slice;
    }
    break;
  case host_to_device: {
    copy_to_device_via_host_buffer(q, size, to_surface, to_range, from_surface,
                                   from_range, event_list, host_to_host,
                                   dep_events);
    break;
  }
  case device_to_host: {
    host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                    event_list);
    // Copy from host temp buffer to host target with reshaping.
    event_list = dpct_memcpy(
        q, to_surface, buf.get_ptr(), to_range, from_range,
        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
        // Copy from device to temp host buffer with only one submit.
        std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                             buf.get_size(), device_to_host,
                                             dep_events)});
    break;
  }
  case device_to_device:
#ifdef DPCT_USM_LEVEL_NONE
  {
    auto &mm = mem_mgr::instance();
    auto to_alloc = mm.translate_ptr(to_surface);
    auto from_alloc = mm.translate_ptr(from_surface);
    size_t to_offset = (byte_t *)to_surface - to_alloc.alloc_ptr;
    size_t from_offset = (byte_t *)from_surface - from_alloc.alloc_ptr;
    event_list.push_back(q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      auto to_o = sycl::id<1>(to_offset);
      auto from_o = sycl::id<1>(from_offset);
      sycl::accessor<byte_t, 1, sycl::access_mode::write,
                     sycl::access::target::device>
          to_acc(to_alloc.buffer, cgh,
                 get_copy_range(size, to_slice, to_range.get(0)), to_o);
      sycl::accessor<byte_t, 1, sycl::access_mode::read,
                     sycl::access::target::device>
          from_acc(from_alloc.buffer, cgh,
                   get_copy_range(size, from_slice, from_range.get(0)), from_o);
      cgh.parallel_for<class dpct_memcpy_3d_detail_usmnone>(
          size, [=](sycl::id<3> id) {
            to_acc[get_offset(id, to_slice, to_range.get(0))] =
                from_acc[get_offset(id, from_slice, from_range.get(0))];
          });
    }));
  }
#else
    event_list.push_back(q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      cgh.parallel_for<class dpct_memcpy_3d_detail>(size, [=](sycl::id<3> id) {
        to_surface[get_offset(id, to_slice, to_range.get(0))] =
            from_surface[get_offset(id, from_slice, from_range.get(0))];
      });
    }));
#endif
  break;
  default:
    throw std::runtime_error("dpct_memcpy: invalid direction value");
  }
  return event_list;
}

/// memcpy 2D/3D matrix specified by pitched_data.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
            pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                     sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                     sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id,
                     from_id, size, direction);
}

/// memcpy 2D/3D matrix between different devices.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id, int to_dev_id,
            pitched_data from, sycl::id<3> from_id, int from_dev_id,
            sycl::range<3> size) {
  if (to_dev_id == from_dev_id)
    return dpct_memcpy(q, to, to_id, from, from_id, size,
                       memcpy_direction::device_to_device);
  std::vector<sycl::event> event_list;
  const auto to_range = sycl::range<3>(to.get_pitch(), to.get_y(), 1),
             from_range = sycl::range<3>(from.get_pitch(), from.get_y(), 1);
  const size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
  unsigned char *to_surface = (unsigned char *)to.get_data_ptr() +
                              get_offset(to_id, to_slice, to_range.get(0));
  const unsigned char *from_surface =
      (const unsigned char *)from.get_data_ptr() +
      get_offset(from_id, from_slice, from_range.get(0));
  copy_to_device_via_host_buffer(q, size, to_surface, to_range, from_surface,
                                 from_range, event_list, device_to_host);
  return event_list;
}

/// memcpy 2D matrix with pitch.
static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t to_pitch,
            size_t from_pitch, size_t x, size_t y,
            memcpy_direction direction = automatic) {
  return dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                     sycl::range<3>(from_pitch, y, 1), sycl::id<3>(0, 0, 0),
                     sycl::id<3>(0, 0, 0), sycl::range<3>(x, y, 1), direction);
}

static inline std::vector<sycl::event>
dpct_memcpy(sycl::queue &q, const memcpy_parameter &param) {
  auto to = param.to.pitched;
  auto from = param.from.pitched;
#ifdef SYCL_EXT_ONEAPI_BINDLESS_IMAGES
  if (param.to.image_bindless != nullptr &&
      param.from.image_bindless != nullptr) {
    return {experimental::detail::dpct_memcpy(
        param.from.image_bindless, param.from.pos, param.from.pos_x_in_bytes,
        param.to.image_bindless, param.to.pos, param.to.pos_x_in_bytes,
        param.size, param.size_x_in_bytes, q)};
  } else if (param.to.image_bindless != nullptr) {
    return {experimental::detail::dpct_memcpy(
        from, param.from.pos, param.from.pos_x_in_bytes,
        param.to.image_bindless, param.to.pos, param.to.pos_x_in_bytes,
        param.size, param.size_x_in_bytes, q)};
  } else if (param.from.image_bindless != nullptr) {
    return {experimental::detail::dpct_memcpy(
        param.from.image_bindless, param.from.pos, param.from.pos_x_in_bytes,
        to, param.to.pos, param.to.pos_x_in_bytes, param.size,
        param.size_x_in_bytes, q)};
  }
#endif
  auto size = param.size;
  auto to_pos = param.to.pos;
  auto from_pos = param.from.pos;
  // If the src and dest are not bindless image, the x can be set to XInByte.
  if (param.size_x_in_bytes != 0) {
    size[0] = param.size_x_in_bytes;
  }
  if (param.to.pos_x_in_bytes != 0) {
    to_pos[0] = param.to.pos_x_in_bytes;
  }
  if (param.from.pos_x_in_bytes != 0) {
    from_pos[0] = param.from.pos_x_in_bytes;
  }
  if (param.to.image != nullptr) {
    to = to_pitched_data(param.to.image);
  }
  if (param.from.image != nullptr) {
    from = to_pitched_data(param.from.image);
  }
  if (deduce_memcpy_direction(q, to.get_data_ptr(), from.get_data_ptr(),
                              param.direction) == device_to_device)
    return dpct_memcpy(q, to, to_pos, param.to.dev_id, from, from_pos,
                       param.from.dev_id, size);
  return dpct_memcpy(q, to, to_pos, from, from_pos, size, param.direction);
}

namespace deprecated {

template <typename T, sycl::usm::alloc AllocKind> class usm_allocator {
private:
  using Alloc = sycl::usm_allocator<T, AllocKind>;
  Alloc _impl;

public:
  using value_type = typename std::allocator_traits<Alloc>::value_type;
  using pointer = typename std::allocator_traits<Alloc>::pointer;
  using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
  using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
  using const_void_pointer =
      typename std::allocator_traits<Alloc>::const_void_pointer;
  using reference = typename std::allocator_traits<Alloc>::value_type &;
  using const_reference =
      const typename std::allocator_traits<Alloc>::value_type &;
  using difference_type =
      typename std::allocator_traits<Alloc>::difference_type;
  using size_type = typename std::allocator_traits<Alloc>::size_type;
  using propagate_on_container_copy_assignment = typename std::allocator_traits<
      Alloc>::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment = typename std::allocator_traits<
      Alloc>::propagate_on_container_move_assignment;
  using propagate_on_container_swap =
      typename std::allocator_traits<Alloc>::propagate_on_container_swap;
  using is_always_equal =
      typename std::allocator_traits<Alloc>::is_always_equal;

  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind> other;
  };

  usm_allocator() : _impl(dpct::get_default_queue()) {}
  usm_allocator(sycl::queue &q) : _impl(q) {}
  ~usm_allocator() {}
  usm_allocator(const usm_allocator &other) : _impl(other._impl) {}
  usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
  pointer address(reference r) { return &r; }
  const_pointer address(const_reference r) { return &r; }
  pointer allocate(size_type cnt, const_void_pointer hint = nullptr) {
    return std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
  }
  void deallocate(pointer p, size_type cnt) {
    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
  }
  size_type max_size() const {
    return std::allocator_traits<Alloc>::max_size(_impl);
  }
  bool operator==(const usm_allocator &other) const {
    return _impl == other._impl;
  }
  bool operator!=(const usm_allocator &other) const {
    return _impl != other._impl;
  }
};

} // namespace deprecated

inline void dpct_free(void *ptr, const sycl::queue &q) {
  if (ptr) {
#ifdef DPCT_USM_LEVEL_NONE
    detail::mem_mgr::instance().mem_free(ptr);
#else
    sycl::free(ptr, q.get_context());
#endif // DPCT_USM_LEVEL_NONE
  }
}

inline sycl::event async_dpct_free(const std::vector<void *> &pointers,
                                   const std::vector<sycl::event> &events,
                                   sycl::queue q) {
  return q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.host_task([=] {
      for (auto p : pointers)
        if (p) {
          detail::dpct_free(p, q);
        }
    });
  });
}

/// Device variable with address space of shared, global or constant.
template <class T, memory_region Memory, size_t Dimension> class device_memory {
public:
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<Dimension>;
  using value_t = typename detail::memory_traits<Memory, T>::value_t;
  using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

  device_memory() : device_memory(sycl::range<Dimension>(1)) {}

  /// Constructor of 1-D array with initializer list
  device_memory(const sycl::range<Dimension> &in_range,
                std::initializer_list<value_t> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range.size());
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    std::memcpy(_host_ptr, init_list.begin(), init_list.size() * sizeof(T));
  }

  /// Constructor of 2-D array with initializer list
  template <size_t D = Dimension>
  device_memory(
      const typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
      std::initializer_list<std::initializer_list<value_t>> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range[0]);
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    auto tmp_data = _host_ptr;
    for (auto sub_list : init_list) {
      assert(sub_list.size() <= in_range[1]);
      std::memcpy(tmp_data, sub_list.begin(), sub_list.size() * sizeof(T));
      tmp_data += in_range[1];
    }
  }

  /// Constructor with range
  device_memory(const sycl::range<Dimension> &range_in)
      : _size(range_in.size() * sizeof(T)), _range(range_in), _reference(false),
        _host_ptr(nullptr), _device_ptrs(get_ptrs_size(), nullptr) {
    static_assert((Memory == global) || (Memory == constant) ||
                      (Memory == shared),
                  "device memory region should be global, constant or shared");
    // Make sure that singleton class mem_mgr and dev_mgr will destruct later
    // than this.
    detail::mem_mgr::instance();
    dev_mgr::instance();
  }

  /// Constructor with range
  template <class... Args>
  device_memory(Args... Arguments)
      : device_memory(sycl::range<Dimension>(Arguments...)) {}

  ~device_memory() {
    if (!_reference) {
      for (unsigned i = 0; i < _device_ptrs.size(); ++i) {
        if (auto ptr = _device_ptrs[i])
          dpct::dpct_free(ptr, get_device(i).default_queue());
      }
    }
    if (_host_ptr)
      std::free(_host_ptr);
  }

  /// Allocate memory with default queue, and init memory if has initial value.
  void init() { init(dpct::get_default_queue()); }
  /// Allocate memory with specified queue, and init memory if has initial
  /// value.
  void init(sycl::queue &q) { (void)get_ptr_impl(q); }

  /// The variable is assigned to a device pointer.
#ifndef DPCT_USM_LEVLE_NONE
  [[deprecated]]
#endif
  void
  assign(value_t *src, size_t size) {
    this->~device_memory();
    new (this) device_memory(src, size);
  }

  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used.
  value_t *get_ptr() { return get_ptr(get_default_queue()); }
  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used.
  value_t *get_ptr(sycl::queue &q) { return get_ptr_impl(q); }

  /// Get the device memory object size in bytes.
  size_t get_size() { return _size; }

  template <size_t D = Dimension>
  typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
    auto ptr = get_ptr();
#ifdef DPCT_USM_LEVEL_NONE
    return dpct::get_buffer<typename std::enable_if<D == 1, T>::type>(ptr)
        .get_host_access(sycl::read_write)[index];
#else
    return ptr[index];
#endif // DPCT_USM_LEVEL_NONE
  }

#ifdef DPCT_USM_LEVEL_NONE
  /// Get sycl::accessor for the device memory object when usm is not used.
  accessor_t get_access(sycl::handler &cgh) {
    return get_buffer(_device_ptrs.front())
        .template reinterpret<T, Dimension>(_range)
        .template get_access<detail::memory_traits<Memory, T>::mode,
                             detail::memory_traits<Memory, T>::target>(cgh);
  }
#else
  /// Get dpct::accessor with dimension info for the device memory object
  /// when usm is used and dimension is greater than 1.
  template <size_t D = Dimension>
  typename std::enable_if<D != 1, dpct_accessor_t>::type
  get_access(sycl::handler &cgh) {
    return dpct_accessor_t((T *)_device_ptrs.front(), _range);
  }
#endif // DPCT_USM_LEVEL_NONE

private:
  device_memory(value_t *memory_ptr, size_t size)
      : _size(size), _range(size / sizeof(T)), _reference(true),
        _device_ptrs(get_ptrs_size(), memory_ptr) {}

  value_t *allocate_device(sycl::queue &q) {
    _q = q;
#ifndef DPCT_USM_LEVEL_NONE
    if (Memory == shared) {
      return (value_t *)sycl::malloc_shared(_size, q.get_device(),
                                            q.get_context());
    }
#ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
    if (Memory == constant) {
      return (value_t *)sycl::malloc_device(
          _size, q.get_device(), q.get_context(),
          sycl::ext::oneapi::property::usm::device_read_only());
    }
#endif
#endif
    return (value_t *)detail::dpct_malloc(_size, q);
  }

  value_t *get_ptr_impl(sycl::queue &q) {
#ifdef DPCT_USM_LEVEL_NONE
    auto &ptr = _device_ptrs.front();
#else
    auto &ptr = _device_ptrs[get_device_id(q.get_device())];
#endif
    if (ptr || !_size)
      return ptr;
    ptr = allocate_device(q);
    if (_host_ptr)
      detail::dpct_memcpy(q, ptr, _host_ptr, _size, host_to_device);
    return ptr;
  }

  static size_t get_ptrs_size() {
#ifdef DPCT_USM_LEVEL_NONE
    return 1;
#else
    return device_count();
#endif
  }

  size_t _size;
  sycl::range<Dimension> _range;
  bool _reference;
  value_t *_host_ptr;
  std::vector<value_t *> _device_ptrs;
  sycl::queue _q;
};
template <class T, memory_region Memory>
class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
public:
  using base = device_memory<T, Memory, 1>;
  using value_t = typename base::value_t;
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<0>;

  /// Constructor with initial value.
  device_memory(const value_t &val) : base(sycl::range<1>(1), {val}) {}

  /// Default constructor
  device_memory() : base(1) {}

#ifdef DPCT_USM_LEVEL_NONE
  /// Get sycl::accessor for the device memory object when usm is not used.
  accessor_t get_access(sycl::handler &cgh) {
    auto buf = get_buffer(base::get_ptr())
                   .template reinterpret<T, 1>(sycl::range<1>(1));
    return accessor_t(buf, cgh);
  }
#endif // DPCT_USM_LEVEL_NONE
};

} // namespace detail
} // namespace dpct
#endif //! __DPCT_DETAIL_MEMORY_DETAIL_HPP__
