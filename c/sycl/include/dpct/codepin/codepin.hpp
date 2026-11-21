//==---- codepin.hpp -------------------------*- C++ -*------------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===-------------------------------------------------------------------===//
#ifndef __DPCT_CODEPIN_HPP__
#define __DPCT_CODEPIN_HPP__

#include "serialization/basic.hpp"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>
#include <stdlib.h>

// Random seed for data sampling.
#ifndef CODEPIN_RAND_SEED
#define CODEPIN_RAND_SEED 0
#endif

// Data size threshold to trigger data sampling.
// array/pointer size larger than the threshold will be sampled.
#ifndef CODEPIN_SAMPLING_THRESHOLD
#define CODEPIN_SAMPLING_THRESHOLD 20
#endif

// Sampling percent, interval: [0, 100]
// 0: No data will be logged.  
// 100: all data will be logged.
#ifndef CODEPIN_SAMPLING_PERCENT
#define CODEPIN_SAMPLING_PERCENT 1
#endif

#define CODEPIN_TO_STR(x) CODEPIN_STR(x)
#define CODEPIN_STR(x) #x
#pragma message(                                                               \
    "CodePin data sampling feature is enabled for data dump. As follow list 3 configs for data sampling:")
#pragma message("CODEPIN_RAND_SEED: " CODEPIN_TO_STR(CODEPIN_RAND_SEED))
#pragma message("CODEPIN_SAMPLING_THRESHOLD: " CODEPIN_TO_STR(CODEPIN_SAMPLING_THRESHOLD))
#pragma message("CODEPIN_SAMPLING_PERCENT: " CODEPIN_TO_STR(CODEPIN_SAMPLING_PERCENT))
#pragma message(                                                               \
    "Define the macros in the build command to change sampling configs. Also refer to codepin.hpp for definitions and default value of the macros.")

namespace dpct {
namespace experimental {
namespace codepin {

inline static std::map<std::string, int> api_index;
inline static std::map<std::string, event_t> event_map;
inline static bool rand_seed_setup = false;
namespace detail {

inline static std::unordered_set<void *> ptr_unique;

class logger {
public:
  logger(const std::string &dump_file)
      : opf(dump_file), json_ss(opf), arr(json_ss) {
    auto top_obj = arr.object();
    top_obj.key("CodePin Random Seed");
    top_obj.value(CODEPIN_RAND_SEED);
    top_obj.key("CodePin Sampling Threshold");
    top_obj.value(CODEPIN_SAMPLING_THRESHOLD);
    top_obj.key("CodePin Sampling Percent");
    top_obj.value(CODEPIN_SAMPLING_PERCENT);
  }
  ~logger() {}

  detail::json_stringstream &get_stringstream() {
    return this->json_ss;
  }

  template <class... Args>
  void print_CP(const std::string &cp_id, std::string device_name,
                size_t free_byte, size_t total_byte, float elapse_time,
                queue_t queue, Args... args) {
    ptr_unique.clear();
    auto obj = arr.object();
    obj.key("ID");
    obj.value(cp_id);
    obj.key("Device Name");
    obj.value(device_name);
    obj.key("Device ID");
#ifdef __NVCC__
    int device_id;
    cudaGetDevice(&device_id);
    obj.value(device_id);
#else
    obj.value((int)dpct::get_current_device_id());
#endif
    obj.key("Stream Address");
    obj.value((void *)queue);
    obj.key("Free Device Memory");
    obj.value(free_byte);
    obj.key("Total Device Memory");
    obj.value(total_byte);
    obj.key("Elapse Time(ms)");
    obj.value(elapse_time);
    obj.key("CheckPoint");
    auto cp_obj =
        obj.value<detail::json_stringstream::json_obj>();
    print_args(cp_obj, queue, 0, args...);
  }

  void print_args(json_stringstream::json_obj &obj, queue_t queue,
                  int index = 0) {}

  template <class First, class... RestArgs>
  void print_args(json_stringstream::json_obj &obj, queue_t queue, int index,
                  std::string_view arg_name, First &arg, RestArgs... args) {
    obj.key(arg_name);
    {
      auto type_obj =
          obj.value<detail::json_stringstream::json_obj>();
      detail::data_ser<First>::print_type_name(type_obj);
      type_obj.key("Address");
      print_address(type_obj, arg);
      type_obj.key("Index");
      type_obj.value(index);
      type_obj.key("Data");
      detail::data_ser<First>::dump(json_ss, arg, queue);
    }
    print_args(obj, queue, index + 1, args...);
  }
  template <class ArgT>
  void print_address(json_stringstream::json_obj &obj, ArgT arg) {
    if constexpr (std::is_pointer<ArgT>::value) {
      obj.value((void *)arg);
    } else {
      obj.value((void *)&arg);
    }
  }

private:
  std::ofstream opf;
  detail::json_stringstream json_ss;
  detail::json_stringstream::json_array arr;
};

#ifdef __NVCC__
inline std::string data_file_prefix = "CodePin_CUDA_";
#else
inline std::string data_file_prefix = "CodePin_SYCL_";
#endif

inline std::string get_data_file_name(std::string_view data_file_prefix) {
  std::time_t now_time = std::time(nullptr);
  std::stringstream strs;
  strs << data_file_prefix
       << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S")
       << ".json";
  return strs.str();
}

inline logger log(get_data_file_name(data_file_prefix));

inline std::map<void *, uint32_t> &get_ptr_size_map() {
  static std::map<void *, uint32_t> ptr_size_map;
  return ptr_size_map;
}

inline uint32_t get_ptr_size_in_bytes(void *ptr) {
  const std::map<void *, uint32_t> &ptr_size_map = get_ptr_size_map();
  const auto &it = ptr_size_map.find(ptr);
  return (it != ptr_size_map.end()) ? it->second : 0;
}

template <class T>
class data_ser<T*, void> {
public:
  static void dump(detail::json_stringstream &ss, T* value,
                   queue_t queue) {
    using PointeeType = std::remove_cv_t<std::remove_pointer_t<T>>;
    PointeeType *non_const_value = const_cast<PointeeType *>(value);
    if (ptr_unique.find(non_const_value) != ptr_unique.end()) {
      return;
    }
    ptr_unique.insert(non_const_value);
    int size = get_ptr_size_in_bytes(non_const_value);
    size = size == 0 ? 1 : size / sizeof(*value);
    PointeeType *dump_addr = non_const_value;
    bool is_dev = is_dev_ptr(non_const_value);
    if (is_dev) {
      PointeeType *h_data = new PointeeType[size];
#ifdef __NVCC__
      cudaMemcpyAsync(h_data, value, size * sizeof(PointeeType),
                      cudaMemcpyDeviceToHost, queue);
      cudaStreamSynchronize(queue);
#else
      queue->memcpy((void *)h_data, (void *)value, size * sizeof(PointeeType))
          .wait();
#endif
      dump_addr = h_data;
    }
    auto arr = ss.array();
    for (int i = 0; i < size; ++i) {
      if (size > CODEPIN_SAMPLING_THRESHOLD && i != 0) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (r > (float)CODEPIN_SAMPLING_PERCENT/(float)100)
          continue;
      }
      auto obj = arr.object();
      detail::data_ser<PointeeType>::print_type_name(obj);
      obj.key("Data");
      detail::data_ser<PointeeType>::dump(
          ss, *(dump_addr + i), queue);
    }
    if(is_dev)
      delete[] dump_addr;
  }
  static void print_type_name(
      detail::json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value("Pointer");
  }
};

template <class T>
class data_ser<T, typename std::enable_if<std::is_array<T>::value>::type> {
public:
  static void dump(detail::json_stringstream &ss, T value,
                   queue_t queue) {
    auto arr = ss.array();
    size_t size = sizeof(T) / sizeof(value[0]);
    for (size_t i = 0; i < size; ++i) {
      if (size > CODEPIN_SAMPLING_THRESHOLD && i != 0) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (r > (float)CODEPIN_SAMPLING_PERCENT/(float)100)
          continue;
      }
      auto obj = arr.object();
      detail::data_ser<
          std::remove_extent_t<T>>::print_type_name(obj);
      obj.key("Data");
      detail::data_ser<std::remove_extent_t<T>>::dump(
          ss, value[i], queue);
    }
  }
  static void print_type_name(
      detail::json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value("Array");
  }
};

template <class... Args>
void gen_log_API_CP(const std::string &cp_id, std::string device_name,
                    size_t free_byte, size_t total_byte, float elapse_time,
                    queue_t queue, Args... args) {
  if (!rand_seed_setup) {
    srand(CODEPIN_RAND_SEED);
    rand_seed_setup = true;
    std::cout << "CodePin data sampling is enabled for data dump. As follow list 3 "
                 "configs for data sampling:"
              << std::endl;
    std::cout << "CODEPIN_RAND_SEED: " << CODEPIN_RAND_SEED << std::endl;
    std::cout << "CODEPIN_SAMPLING_THRESHOLD: " << CODEPIN_SAMPLING_THRESHOLD
              << std::endl;
    std::cout << "CODEPIN_SAMPLING_PERCENT: " << CODEPIN_SAMPLING_PERCENT
              << std::endl;
  }
  log.print_CP(cp_id, device_name, free_byte, total_byte, elapse_time, queue,
               args...);
}
} // namespace detail

#ifdef __NVCC__
inline void synchronize(cudaStream_t stream) { cudaStreamSynchronize(stream); }
#else
inline void synchronize(sycl::queue *q) { q->wait(); }
#endif

/// Generate API check point prolog.
/// \param cp_id The UID of the function call.
/// \param queue The sycl queue to synchronize the command execution.
/// \param args The var name string and variable value pair list.
template <class... Args>
void gen_prolog_API_CP(const std::string &cp_id,
                       queue_t queue, Args&&... args) {
  synchronize(queue);
  std::string prolog_tag = cp_id + ":" + "prolog";
  if (api_index.find(cp_id) == api_index.end()) {
    api_index[cp_id] = 0;
  } else {
    api_index[cp_id]++;
  }
  std::string event_id =
      cp_id + ":" + std::to_string(api_index[cp_id]);
  size_t free_byte, total_byte;
#ifdef __NVCC__
  int device;
  cudaGetDevice(&device);  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  std::string device_name(deviceProp.name);
  cudaMemGetInfo(&free_byte, &total_byte);
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, queue);
  event_map[event_id] = event;
#else
  dpct::get_current_device().get_memory_info(free_byte, total_byte);
  std::string device_name = dpct::get_current_device().get_info<sycl::info::device::name>();
#ifdef DPCT_PROFILING_ENABLED
  sycl::event event = queue->ext_oneapi_submit_barrier();
  event_map[event_id] = event;
#endif //DPCT_PROFILING_ENABLED
#endif

  detail::gen_log_API_CP(prolog_tag, device_name, free_byte, total_byte, 0.0f,
                         queue, args...);
}

/// Generate API check point epilog.
/// \param cp_id The UID of the function call.
/// \param stream The sycl queue to synchronize the command execution.
/// \param args The var name string and variable value pair list.
template <class... Args>
void gen_epilog_API_CP(const std::string &cp_id,
                       queue_t queue, Args&&... args) {
  synchronize(queue);
  std::string epilog_tag = cp_id + ":" + "epilog";
  std::string event_id =
      cp_id + ":" + std::to_string(api_index[cp_id]);
  size_t free_byte, total_byte;
  float kernel_elapsed_time = 0.0f;
#ifdef __NVCC__
  int device;
  cudaGetDevice(&device);  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  std::string device_name(deviceProp.name);
  cudaMemGetInfo(&free_byte, &total_byte);
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, queue);
  auto pre_event = event_map[event_id];
  event_map.erase(event_id);
  cudaEventSynchronize(event);
  cudaEventElapsedTime(&kernel_elapsed_time, pre_event, event);
#else
#ifdef DPCT_PROFILING_ENABLED
  sycl::event event = queue->ext_oneapi_submit_barrier();
  auto pre_event = event_map[event_id];
  event_map.erase(event_id);
  event.wait_and_throw();
  kernel_elapsed_time =
      (event.get_profiling_info<sycl::info::event_profiling::command_end>() -
       pre_event
           .get_profiling_info<sycl::info::event_profiling::command_start>()) /
      1000000.0f;
#endif //DPCT_PROFILING_ENABLED
  std::string device_name = dpct::get_current_device().get_info<sycl::info::device::name>();
  dpct::get_current_device().get_memory_info(free_byte, total_byte);
#endif
  detail::gen_log_API_CP(epilog_tag, device_name, free_byte, total_byte,
                         kernel_elapsed_time, queue, args...);
}

inline std::map<void *, uint32_t> &get_ptr_size_map() {
  return detail::get_ptr_size_map();
}
} // namespace codepin
} // namespace experimental
} // namespace dpct

namespace dpctexp = dpct::experimental;


#endif // End of __DPCT_CODEPIN_HPP__
