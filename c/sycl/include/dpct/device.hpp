//==---- device.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_DEVICE_HPP__
#define __DPCT_DEVICE_HPP__

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stack>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif
#if defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace dpct {
namespace detail {
static void get_version(const sycl::device &dev, int &major, int &minor) {
  // Version string has the following format:
  // a. OpenCL<space><major.minor><space><vendor-specific-information>
  // b. <major.minor>
  std::string ver;
  ver = dev.get_info<sycl::info::device::version>();
  std::string::size_type i = 0;
  while (i < ver.size()) {
    if (isdigit(ver[i]))
      break;
    i++;
  }
  if (i < ver.size())
    major = std::stoi(&(ver[i]));
  else
    major = 0;
  while (i < ver.size()) {
    if (ver[i] == '.')
      break;
    i++;
  }
  i++;
  if (i < ver.size())
    minor = std::stoi(&(ver[i]));
  else
    minor = 0;
}
} // namespace detail

/// SYCL default exception handler
inline auto exception_handler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                << e.what() << std::endl
                << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
  }
};

typedef sycl::event *event_ptr;

typedef sycl::queue *queue_ptr;

typedef char *device_ptr;

using queue_callback = std::function<void (queue_ptr, int, void*)>;

/// Destroy \p event pointed memory.
///
/// \param event Pointer to the sycl::event address.
static void destroy_event(event_ptr event) {
    delete event;
}

class device_info {
public:
  // get interface
  const char *get_name() const { return _name; }
  char *get_name() { return _name; }
  template <typename WorkItemSizesTy = sycl::range<3>,
            std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                 std::is_same_v<WorkItemSizesTy, int *>,
                             int> = 0>
  auto get_max_work_item_sizes() const {
    if constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
      return sycl::range<3>(_max_work_item_sizes_i[0],
                            _max_work_item_sizes_i[1],
                            _max_work_item_sizes_i[2]);
    else {
      return _max_work_item_sizes_i;
    }  
  }
  template <typename WorkItemSizesTy = sycl::range<3>,
            std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                 std::is_same_v<WorkItemSizesTy, int *>,
                             int> = 0>
  auto get_max_work_item_sizes() {
    if constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
      return sycl::range<3>(_max_work_item_sizes_i[0],
                            _max_work_item_sizes_i[1],
                            _max_work_item_sizes_i[2]);
    else {
      return _max_work_item_sizes_i;
    }  
  }
  bool get_host_unified_memory() const { return _host_unified_memory; }
  int get_major_version() const { return _major; }
  int get_minor_version() const { return _minor; }
  int get_integrated() const { return _integrated; }
  int get_max_clock_frequency() const { return _frequency; }
  int get_max_compute_units() const { return _max_compute_units; }
  int get_max_work_group_size() const { return _max_work_group_size; }
  int get_max_sub_group_size() const { return _max_sub_group_size; }
  int get_max_work_items_per_compute_unit() const {
    return _max_work_items_per_compute_unit;
  }
  int get_max_register_size_per_work_group() const {
    return _max_register_size_per_work_group;
  }
  template <typename NDRangeSizeTy = size_t *,
            std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                 std::is_same_v<NDRangeSizeTy, int *>,
                             int> = 0>
  auto get_max_nd_range_size() const {
    if constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
      return _max_nd_range_size;
    else
      return _max_nd_range_size_i;
  }
  template <typename NDRangeSizeTy = size_t *,
            std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                 std::is_same_v<NDRangeSizeTy, int *>,
                             int> = 0>
  auto get_max_nd_range_size() {
    if constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
      return _max_nd_range_size;
    else
      return _max_nd_range_size_i;
  }
  size_t get_global_mem_size() const { return _global_mem_size; }
  size_t get_local_mem_size() const { return _local_mem_size; }
  /// Returns the maximum clock rate of device's global memory in kHz. If
  /// compiler does not support this API then returns default value 3200000 kHz.
  unsigned int get_memory_clock_rate() const { return _memory_clock_rate; }
  /// Returns the maximum bus width between device and memory in bits. If
  /// compiler does not support this API then returns default value 64 bits.
  unsigned int get_memory_bus_width() const { return _memory_bus_width; }
  uint32_t get_device_id() const { return _device_id; }
  std::array<unsigned char, 16> get_uuid() const { return _uuid; }
  /// Returns global memory cache size in bytes.
  unsigned int get_global_mem_cache_size() const {
    return _global_mem_cache_size;
  }
  int get_image1d_max() const { return _image1d_max; }
  auto get_image2d_max() const { return _image2d_max; }
  auto get_image2d_max() { return _image2d_max; }
  auto get_image3d_max() const { return _image3d_max; }
  auto get_image3d_max() { return _image3d_max; }

  // set interface
  void set_name(const char* name) {
    size_t length = strlen(name);
    if (length < 256) {
      std::memcpy(_name, name, length + 1);
    } else {
      std::memcpy(_name, name, 255);
      _name[255] = '\0';
    }
  }
  void set_max_work_item_sizes(const sycl::range<3> max_work_item_sizes) {
    for (int i = 0; i < 3; ++i)
      _max_work_item_sizes_i[i] = max_work_item_sizes[i];
  }
  void set_host_unified_memory(bool host_unified_memory) {
    _host_unified_memory = host_unified_memory;
  }
  void set_major_version(int major) { _major = major; }
  void set_minor_version(int minor) { _minor = minor; }
  void set_integrated(int integrated) { _integrated = integrated; }
  void set_max_clock_frequency(int frequency) { _frequency = frequency; }
  void set_max_compute_units(int max_compute_units) {
    _max_compute_units = max_compute_units;
  }
  void set_global_mem_size(size_t global_mem_size) {
    _global_mem_size = global_mem_size;
  }
  void set_local_mem_size(size_t local_mem_size) {
    _local_mem_size = local_mem_size;
  }
  void set_max_work_group_size(int max_work_group_size) {
    _max_work_group_size = max_work_group_size;
  }
  void set_max_sub_group_size(int max_sub_group_size) {
    _max_sub_group_size = max_sub_group_size;
  }
  void
  set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit) {
    _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
  }
  void set_max_nd_range_size(int max_nd_range_size[]) {
    for (int i = 0; i < 3; i++) {
      _max_nd_range_size[i] = max_nd_range_size[i];
      _max_nd_range_size_i[i] = max_nd_range_size[i];
    }
  }
  void set_max_nd_range_size(sycl::id<3> max_nd_range_size) {
    for (int i = 0; i < 3; i++) {
      _max_nd_range_size[i] = max_nd_range_size[i];
      _max_nd_range_size_i[i] = max_nd_range_size[i];
    }
  }
  void set_memory_clock_rate(unsigned int memory_clock_rate) {
    _memory_clock_rate = memory_clock_rate;
  }
  void set_memory_bus_width(unsigned int memory_bus_width) {
    _memory_bus_width = memory_bus_width;
  }
  void
  set_max_register_size_per_work_group(int max_register_size_per_work_group) {
    _max_register_size_per_work_group = max_register_size_per_work_group;
  }
  void set_device_id(uint32_t device_id) {
    _device_id = device_id;
  }
  void set_uuid(std::array<unsigned char, 16> uuid) {
    _uuid = std::move(uuid);
  }
  void set_global_mem_cache_size(unsigned int global_mem_cache_size) {
    _global_mem_cache_size = global_mem_cache_size;
  }
  void set_image1d_max(size_t image_max_buffer_size) {
    _image1d_max = image_max_buffer_size;
  }

private:
  char _name[256];
  int _max_work_item_sizes_i[3];
  bool _host_unified_memory = false;
  int _major;
  int _minor;
  int _integrated = 0;
  int _frequency;
  // Set estimated value 3200000 kHz as default value.
  unsigned int _memory_clock_rate = 3200000;
  // Set estimated value 64 bits as default value.
  unsigned int _memory_bus_width = 64;
  unsigned int _global_mem_cache_size;
  int _max_compute_units;
  int _max_work_group_size;
  int _max_sub_group_size;
  int _max_work_items_per_compute_unit;
  int _max_register_size_per_work_group;
  size_t _global_mem_size;
  size_t _local_mem_size;
  size_t _max_nd_range_size[3];
  int _max_nd_range_size_i[3];
  uint32_t _device_id;
  std::array<unsigned char, 16> _uuid;
  int _image1d_max;
  int _image2d_max[2];
  int _image3d_max[3];
};

static int get_major_version(const sycl::device &dev) {
  int major, minor;
  detail::get_version(dev, major, minor);
  return major;
}

static int get_minor_version(const sycl::device &dev) {
  int major, minor;
  detail::get_version(dev, major, minor);
  return minor;
}

static void get_device_info(device_info &out, const sycl::device &dev) {
  device_info prop;
  prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

  int major, minor;
  detail::get_version(dev, major, minor);
  prop.set_major_version(major);
  prop.set_minor_version(minor);

  prop.set_max_work_item_sizes(
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
      // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
      // is an enum class element
      dev.get_info<sycl::info::device::max_work_item_sizes>());
#else
      // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
      // an int
      dev.get_info<sycl::info::device::max_work_item_sizes<3>>());
#endif
  prop.set_host_unified_memory(dev.has(sycl::aspect::usm_host_allocations));

  prop.set_max_clock_frequency(
      dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

#ifdef SYCL_EXT_ONEAPI_NUM_COMPUTE_UNITS
  prop.set_max_compute_units(
      dev.get_info<sycl::ext::oneapi::info::device::num_compute_units>());
#else
  prop.set_max_compute_units(
      dev.get_info<sycl::info::device::max_compute_units>());
#endif
  prop.set_max_work_group_size(
      dev.get_info<sycl::info::device::max_work_group_size>());
  prop.set_global_mem_size(dev.get_info<sycl::info::device::global_mem_size>());
  prop.set_local_mem_size(dev.get_info<sycl::info::device::local_mem_size>());

#if (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
  if (dev.has(sycl::aspect::ext_intel_memory_clock_rate)) {
    unsigned int tmp =
        dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
    if (tmp != 0)
      prop.set_memory_clock_rate(1000 * tmp);
  }
  if (dev.has(sycl::aspect::ext_intel_memory_bus_width)) {
    prop.set_memory_bus_width(
        dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
  }
  if (dev.has(sycl::aspect::ext_intel_device_id)) {
    prop.set_device_id(
        dev.get_info<sycl::ext::intel::info::device::device_id>());
  }
  if (dev.has(sycl::aspect::ext_intel_device_info_uuid)) {
    prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
  }
#elif defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
memory_bus_width are not supported by the compiler used. \
Use 3200000 kHz as memory_clock_rate default value. \
Use 64 bits as memory_bus_width default value.")
#else
#warning "get_device_info: querying memory_clock_rate and \
memory_bus_width are not supported by the compiler used. \
Use 3200000 kHz as memory_clock_rate default value. \
Use 64 bits as memory_bus_width default value."
#endif

  size_t max_sub_group_size = 1;
  std::vector<size_t> sub_group_sizes =
      dev.get_info<sycl::info::device::sub_group_sizes>();

  for (const auto &sub_group_size : sub_group_sizes) {
    if (max_sub_group_size < sub_group_size)
      max_sub_group_size = sub_group_size;
  }

  prop.set_max_sub_group_size(max_sub_group_size);

  prop.set_max_work_items_per_compute_unit(
      dev.get_info<sycl::info::device::max_work_group_size>());
#ifdef SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY
  prop.set_max_nd_range_size(
      dev.get_info<
          sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>());
#else
#if defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying the maximum number \
of work groups is not supported.")
#else
#warning "get_device_info: querying the maximum number of \
work groups is not supported."
#endif
  int max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
  prop.set_max_nd_range_size(max_nd_range_size);
#endif

#ifdef SYCL_EXT_CODEPLAY_MAX_REGISTERS_PER_WORK_GROUP_QUERY
  if (dev.get_backend() == sycl::backend::ext_oneapi_cuda)
    prop.set_max_register_size_per_work_group(
        dev.get_info<sycl::ext::codeplay::experimental::info::device::
                         max_registers_per_work_group>());
  else
#endif
    // Estimates max register size per work group, feel free to update the value
    // according to device properties.
    prop.set_max_register_size_per_work_group(65536);

  prop.set_global_mem_cache_size(
      dev.get_info<sycl::info::device::global_mem_cache_size>());

  prop.set_image1d_max(
      dev.get_info<sycl::info::device::image_max_buffer_size>());
  prop.get_image2d_max()[0] =
      dev.get_info<sycl::info::device::image2d_max_width>();
  prop.get_image2d_max()[1] =
      dev.get_info<sycl::info::device::image2d_max_height>();
  prop.get_image3d_max()[0] =
      dev.get_info<sycl::info::device::image3d_max_width>();
  prop.get_image3d_max()[1] =
      dev.get_info<sycl::info::device::image3d_max_height>();
  prop.get_image3d_max()[2] =
      dev.get_info<sycl::info::device::image3d_max_depth>();
  out = prop;
}

/// Util function to check whether a device supports some kinds of sycl::aspect.
inline void
has_capability_or_fail(const sycl::device &dev,
                       const std::initializer_list<sycl::aspect> &props) {
  for (const auto &it : props) {
    if (dev.has(it))
      continue;
    switch (it) {
    case sycl::aspect::fp64:
      throw std::runtime_error("'double' is not supported in '" +
                               dev.get_info<sycl::info::device::name>() +
                               "' device");
      break;
    case sycl::aspect::fp16:
      throw std::runtime_error("'half' is not supported in '" +
                               dev.get_info<sycl::info::device::name>() +
                               "' device");
      break;
    default:
#define __SYCL_ASPECT(ASPECT, ID)                                              \
  case sycl::aspect::ASPECT:                                                   \
    return #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
      auto getAspectNameStr = [](sycl::aspect AspectNum) -> std::string {
        switch (AspectNum) {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
        default:
          return "unknown aspect";
        }
      };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
      throw std::runtime_error(
          "'" + getAspectNameStr(it) + "' is not supported in '" +
          dev.get_info<sycl::info::device::name>() + "' device");
    }
    break;
  }
}

/// dpct device extension
class device_ext : public sycl::device {
  typedef std::mutex mutex_type;

public:
  device_ext() : sycl::device(), _ctx(*this) {}
  ~device_ext() {
    std::lock_guard<mutex_type> lock(m_mutex);
    clear_queues();
  }
  device_ext(const sycl::device &base) : sycl::device(base), _ctx(*this) {
    std::lock_guard<mutex_type> lock(m_mutex);
    init_queues();
  }

  int is_native_atomic_supported() { return 0; }
  int get_major_version() const {
    return dpct::get_major_version(*this);
  }

  int get_minor_version() const {
    return dpct::get_minor_version(*this);
  }

  int get_max_compute_units() const {
    return get_device_info().get_max_compute_units();
  }

  /// Return the maximum clock frequency of this device in KHz.
  int get_max_clock_frequency() const {
    return get_device_info().get_max_clock_frequency();
  }

  int get_integrated() const { return get_device_info().get_integrated(); }

  int get_max_sub_group_size() const {
    return get_device_info().get_max_sub_group_size();
  }

  int get_max_register_size_per_work_group() const {
    return get_device_info().get_max_register_size_per_work_group();
  }

  int get_max_work_group_size() const {
    return get_device_info().get_max_work_group_size();
  }

  int get_mem_base_addr_align() const {
    return get_info<sycl::info::device::mem_base_addr_align>();
  }

  int get_mem_base_addr_align_in_bytes() const {
    return get_info<sycl::info::device::mem_base_addr_align>() / 8;
  }

  size_t get_global_mem_size() const {
    return get_device_info().get_global_mem_size();
  }

  size_t get_local_mem_size() const {
    return get_device_info().get_local_mem_size();
  }

  int get_max_pitch() const { return INT_MAX; }

  /// Get the number of bytes of free and total memory on the SYCL device.
  /// \param [out] free_memory The number of bytes of free memory on the SYCL device.
  /// \param [out] total_memory The number of bytes of total memory on the SYCL device.
  void get_memory_info(size_t &free_memory, size_t &total_memory) {
#if (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    if (!has(sycl::aspect::ext_intel_free_memory)) {
      std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
      free_memory = 0;
    } else {
      free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
    }
#else
    std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
    free_memory = 0;
#if defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#else
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
    total_memory = get_device_info().get_global_mem_size();
  }

  void get_device_info(device_info &out) const {
    dpct::get_device_info(out, *this);
  }

  device_info get_device_info() const {
    device_info prop;
    dpct::get_device_info(prop, *this);
    return prop;
  }

  void reset() {
    std::lock_guard<mutex_type> lock(m_mutex);
    clear_queues();
    init_queues();
  }

  sycl::queue &in_order_queue() { return *_q_in_order; }

  sycl::queue &out_of_order_queue() { return *_q_out_of_order; }

  sycl::queue &default_queue() {
#ifdef DPCT_USM_LEVEL_NONE
    return out_of_order_queue();
#else
    return in_order_queue();
#endif // DPCT_USM_LEVEL_NONE
  }

  void queues_wait_and_throw() {
    std::unique_lock<mutex_type> lock(m_mutex);
    std::vector<std::shared_ptr<sycl::queue>> current_queues(
        _queues);
    lock.unlock();
    for (const auto &q : current_queues) {
      q->wait_and_throw();
    }
    // Guard the destruct of current_queues to make sure the ref count is safe.
    lock.lock();
  }

  std::vector<sycl::event> get_in_order_queues_last_events() {
    std::unique_lock<mutex_type> lock(m_mutex);
    std::vector<sycl::event> last_events;
    std::vector<std::shared_ptr<sycl::queue>> current_queues(_queues);
    lock.unlock();
    for (const auto &q : current_queues) {
      if (q->is_in_order()) {
        auto last_event = q->ext_oneapi_get_last_event();
        [&](auto &&_e) {
          if constexpr (std::is_same_v<
                            std::remove_reference_t<decltype(last_event)>,
                            sycl::event>)
            last_events.push_back(_e);
          else if (_e.has_value())
            last_events.push_back(_e.value());
        }(last_event);
      }
    }
    // Guard the destruct of current_queues to make sure the ref count is safe.
    lock.lock();
    return last_events;
  }

  sycl::queue *create_queue(bool enable_exception_handler = false) {
#ifdef DPCT_USM_LEVEL_NONE
    return create_out_of_order_queue(enable_exception_handler);
#else
    return create_in_order_queue(enable_exception_handler);
#endif // DPCT_USM_LEVEL_NONE
  }

  sycl::queue *create_in_order_queue(bool enable_exception_handler = false) {
    std::lock_guard<mutex_type> lock(m_mutex);
    return create_queue_impl(enable_exception_handler,
                             sycl::property::queue::in_order());
  }

  sycl::queue *create_out_of_order_queue(bool enable_exception_handler = false) {
    std::lock_guard<mutex_type> lock(m_mutex);
    return create_queue_impl(enable_exception_handler);
  }

  void destroy_queue(sycl::queue *&queue) {
    std::lock_guard<mutex_type> lock(m_mutex);
    _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                  [=](const std::shared_ptr<sycl::queue> &q) -> bool {
                                    return q.get() == queue;
                                  }),
                   _queues.end());
    queue = nullptr;
  }
  [[deprecated("set_saved_queue for device_ext is deprecated, please use "
               "dpct::blas::descriptor::set_saved_queue instead")]] void
  set_saved_queue(sycl::queue *q) {
    std::lock_guard<mutex_type> lock(m_mutex);
    _saved_queue = q;
  }
  [[deprecated(
      "get_saved_queue for device_ext is deprecated, please use "
      "dpct::blas::descriptor::get_saved_queue instead")]] sycl::queue *
  get_saved_queue() const {
    std::lock_guard<mutex_type> lock(m_mutex);
    return _saved_queue;
  }
  sycl::context get_context() const { return _ctx; }

  void
  has_capability_or_fail(const std::initializer_list<sycl::aspect> &props) {
    ::dpct::has_capability_or_fail(*this, props);
  }

private:
  void clear_queues() {
    _queues.clear();
    _q_in_order = _q_out_of_order = _saved_queue = nullptr;
  }

  void init_queues() {
    _q_in_order = create_queue_impl(true, sycl::property::queue::in_order());
    _q_out_of_order = create_queue_impl(true);
    _saved_queue = &default_queue();
  }

  /// Caller should acquire resource \p m_mutex before calling this function.
  template <class... Properties>
  sycl::queue *create_queue_impl(bool enable_exception_handler,
                                 Properties... properties) {
    sycl::async_handler eh = {};
    if (enable_exception_handler) {
      eh = exception_handler;
    }
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, eh,
        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
            sycl::property::queue::enable_profiling(),
#endif
            properties...)));

    return _queues.back().get();
  }

  void get_version(int &major, int &minor) const {
    detail::get_version(*this, major, minor);
  }
  sycl::queue *_q_in_order, *_q_out_of_order;
  sycl::queue *_saved_queue;
  sycl::context _ctx;
  std::vector<std::shared_ptr<sycl::queue>> _queues;
  mutable mutex_type m_mutex;
};

static inline unsigned int get_tid() {
#if defined(__linux__)
  return syscall(SYS_gettid);
#elif defined(_WIN64)
  return GetCurrentThreadId();
#else
#error "Only support Windows and Linux."
#endif
}

/// device manager
class dev_mgr {
public:
  device_ext &current_device() { return get_device(current_device_id()); }
  device_ext &cpu_device() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (_cpu_device == -1) {
      throw std::runtime_error("no valid cpu device");
    } else {
      return *_devs[_cpu_device];
    }
  }
  device_ext &get_device(unsigned int id) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    check_id(id);
    return *_devs[id];
  }

  unsigned int current_device_id() const {
    if (_dev_stack.empty())
      return DEFAULT_DEVICE_ID;
    return _dev_stack.top();
  }

  /// Select device with a device ID.
  /// \param [in] id The id of the device which can
  /// be obtained through get_device_id(const sycl::device).
  void select_device(unsigned int id) {
    /// Replace the top of the stack with the given device id
    if (_dev_stack.empty()) {
      push_device(id);
    } else {
      check_id(id);
      _dev_stack.top() = id;
    }
  }

  unsigned int device_count() { return _devs.size(); }

  unsigned int get_device_id(const sycl::device &dev) {
    unsigned int id = 0;
    for (auto dev_item : _devs) {
      if (*dev_item == dev) {
        return id;
      }
      id++;
    }
    throw std::runtime_error(
        "The device[" + dev.get_info<sycl::info::device::name>() +
        "] is filtered out by dpct::dev_mgr::filter/dpct::filter_device in "
        "current device "
        "list!");
  }

  /// List all the devices with its id in dev_mgr.
  void list_devices() const {
    for (size_t i = 0; i < _devs.size(); ++i) {
      std::cout << "" << i << ": "
                << _devs[i]->get_info<sycl::info::device::name>() << std::endl;
    }
  }

  /// Filter out devices; only keep the device whose name contains one of the
  /// subname in \p dev_subnames.
  /// May break device id mapping and change current device. It's better to be
  /// called before other DPCT/SYCL APIs.
  void filter(const std::vector<std::string> &dev_subnames) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto iter = _devs.begin();
    while (iter != _devs.end()) {
      std::string dev_name = (*iter)->get_info<sycl::info::device::name>();
      bool matched = false;
      for (const auto &name : dev_subnames) {
        if (dev_name.find(name) != std::string::npos) {
          matched = true;
          break;
        }
      }
      if (matched)
        ++iter;
      else
        iter = _devs.erase(iter);
    }
    _cpu_device = -1;
    for (unsigned i = 0; i < _devs.size(); ++i) {
      if (_devs[i]->is_cpu()) {
        _cpu_device = i;
        break;
      }
    }
    /// Clear the device stack for all thread here. But we don't have access to
    /// all threads current implementation.
#ifdef DPCT_HELPER_VERBOSE
    list_devices();
#endif
  }

  template <class DeviceSelector>
  std::enable_if_t<
      std::is_invocable_r_v<int, DeviceSelector, const sycl::device &>>
  select_device(const DeviceSelector &selector = sycl::gpu_selector_v) {
    sycl::device selected_device = sycl::device(selector);
    unsigned int selected_device_id = get_device_id(selected_device);
    select_device(selected_device_id);
  }

  /// Update the device stack for the current thread id
  void push_device(unsigned int id) {
    check_id(id);
    _dev_stack.push(id);
  }

  /// Remove the device from top of the stack if it exist
  unsigned int pop_device() {
    if (_dev_stack.empty())
      throw std::runtime_error("can't pop an empty dpct device stack");

    auto id = _dev_stack.top();
    _dev_stack.pop();
    return id;
  }

  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

private:
  mutable std::recursive_mutex m_mutex;
  dev_mgr() {
    sycl::device default_device =
        sycl::device(sycl::default_selector_v);
    _devs.push_back(std::make_shared<device_ext>(default_device));

    std::vector<sycl::device> sycl_all_devs =
        sycl::device::get_devices(sycl::info::device_type::all);
    // Collect other devices except for the default device.
    if (default_device.is_cpu())
      _cpu_device = 0;
    for (auto &dev : sycl_all_devs) {
      if (dev == default_device) {
        continue;
      }
      _devs.push_back(std::make_shared<device_ext>(dev));
      if (_cpu_device == -1 && dev.is_cpu()) {
        _cpu_device = _devs.size() - 1;
      }
    }
#ifdef DPCT_HELPER_VERBOSE
    list_devices();
#endif
  }
  void check_id(unsigned int id) const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }
  std::vector<std::shared_ptr<device_ext>> _devs;
  /// stack of devices resulting from CUDA context change;
  static inline thread_local std::stack<unsigned int> _dev_stack;
  /// DEFAULT_DEVICE_ID is used, if current_device_id() finds an empty
  /// _dev_stack, which means the default device should be used for the current
  /// thread.
  const unsigned int DEFAULT_DEVICE_ID = 0;
  int _cpu_device = -1;
};

/// Util function to get the default queue of current selected device depends on
/// the USM config. Return the default out-of-ordered queue when USM-none is
/// enabled, otherwise return the default in-ordered queue.
static inline sycl::queue &get_default_queue() {
  return dev_mgr::instance().current_device().default_queue();
}

/// Util function to get the default in-ordered queue of current device in
/// dpct device manager.
static inline sycl::queue &get_in_order_queue() {
  return dev_mgr::instance().current_device().in_order_queue();
}

/// Util function to get the default out-of-ordered queue of current device in
/// dpct device manager.
static inline sycl::queue &get_out_of_order_queue() {
  return dev_mgr::instance().current_device().out_of_order_queue();
}

/// Util function to get the id of current device in
/// dpct device manager.
static inline unsigned int get_current_device_id() {
  return dev_mgr::instance().current_device_id();
}

/// Util function to get the current device.
static inline device_ext &get_current_device() {
  return dev_mgr::instance().current_device();
}

/// Util function to get a device by id.
static inline device_ext &get_device(unsigned int id) {
  return dev_mgr::instance().get_device(id);
}

/// Util function to get the context of the default queue of current
/// device in dpct device manager.
static inline sycl::context get_default_context() {
  return dpct::get_current_device().get_context();
}

/// Util function to get a CPU device.
static inline device_ext &cpu_device() {
  return dev_mgr::instance().cpu_device();
}

static inline unsigned int select_device(unsigned int id) {
  dev_mgr::instance().select_device(id);
  return id;
}

template <class DeviceSelector>
static inline std::enable_if_t<
    std::is_invocable_r_v<int, DeviceSelector, const sycl::device &>>
select_device(const DeviceSelector &selector = sycl::gpu_selector_v) {
  dev_mgr::instance().select_device(selector);
}

/// Filter out devices; only keep the device whose name contains one of the
/// subname in \p dev_subnames.
/// May break device id mapping and change current device. It's better to be
/// called before other DPCT/SYCL APIs.
static inline void filter_device(const std::vector<std::string> &dev_subnames) {
  dev_mgr::instance().filter(dev_subnames);
}

/// List all the devices with its id in dev_mgr.
static inline void list_devices() { dev_mgr::instance().list_devices(); }

static inline unsigned int get_device_id(const sycl::device &dev){
  return dev_mgr::instance().get_device_id(dev);
}

static inline unsigned int get_cpu_device_id() {
  return get_device_id(cpu_device());
}

/// Util function to do implicit sync among queues of the same device then
/// insert a synchronize barrier in the queue. For USM, If the queue is the
/// default in-order queue, try to sync with all queues available in the current
/// device before inserting a barrier. For USM-none, If the queue is the default
/// out-of-order queue, try to sync with all queues available in the current
/// device before inserting a barrier, else try to sync in the current queue
/// before inserting a barrier.
/// \param [out] event_ptr The memory to store the event.
/// \param [in] queue The queue specified to do synchronization.
inline void sync_barrier(sycl::event *event_ptr,
                         sycl::queue *queue = &get_default_queue()) {
  if (*queue == get_default_queue()) {
    // Wait all the kernel tasks in all the queues of current device completed.
    dpct::get_current_device().queues_wait_and_throw();
  }

#ifdef DPCT_USM_LEVEL_NONE
  if (*queue != get_default_queue()) {
    // For out-of-ordered queue, wait all the kernel tasks in \p queue
    // completed.
    queue->wait();
  }
#endif

#ifdef DPCT_PROFILING_ENABLED
  *event_ptr = queue->ext_oneapi_submit_barrier();
#else
  *event_ptr = queue->single_task([=]() {});
#endif
}

static inline unsigned int push_device_for_curr_thread(unsigned int id) {
  dev_mgr::instance().push_device(id);
  return id;
}

static inline unsigned int pop_device_for_curr_thread(void) {
  return dev_mgr::instance().pop_device();
}

static inline unsigned int device_count() {
  return dev_mgr::instance().device_count();
}
} // namespace dpct

#endif // __DPCT_DEVICE_HPP__