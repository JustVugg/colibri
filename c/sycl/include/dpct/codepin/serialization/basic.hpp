//==----------- basic.hpp ----------------------------*-C++-*-------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===--------------------------------------------------------------------===//
#ifndef __DPCT_CODEPIN_SER_BASIC_HPP__
#define __DPCT_CODEPIN_SER_BASIC_HPP__

#if defined(__linux__)
#include <cxxabi.h>
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#ifdef __NVCC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif

namespace dpct {
namespace experimental {
namespace codepin {

#ifdef __NVCC__
typedef cudaStream_t queue_t;
#else
typedef dpct::queue_ptr queue_t;
#endif

#ifdef __NVCC__
typedef cudaEvent_t event_t;
#else
typedef sycl::event event_t;
#endif

namespace detail {

inline bool is_dev_ptr(void *p) {
#ifdef __NVCC__
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, p);
  if (attr.type == cudaMemoryTypeDevice)
    return true;
  return false;
#else
  dpct::pointer_attributes attributes;
  attributes.init(p);
  if (attributes.get_device_pointer() != nullptr)
    return true;
  return false;
#endif
}

class json_stringstream {
  public:
  json_stringstream(std::ofstream &ofst) : os(ofst) {
    if (!ofst.is_open()) {
      throw std::runtime_error("Error while openning file: ");
    }
  }

private:
  std::string indent;
  const size_t tab_length = 2;
  std::ofstream &os;

#if defined(__linux__)
  const std::string newline = "\n";
#elif defined(_WIN64)
  const std::string newline = "\r\n";
#else
#error Only support windows and Linux.
#endif


public:
  class json_obj {
    bool isFirst = true;
    json_stringstream &js;
  private:
    friend class json_stringstream;
    json_obj(json_stringstream &json_ss) : js(json_ss) {
      js.os << "{" << js.newline;
      js.indent.append(js.tab_length, ' ');
      js.os << js.indent;
    }

  public:
    template<class T> T value();

    void key(std::string_view key) {
      if (!isFirst){
        js.os << "," << js.newline << js.indent;
      } else {
        isFirst = false;
      }
      js.os << "\"" << key << "\": ";
    };

    void value(std::string_view value) { js.os << "\"" << value << "\""; };
    void value(float value) { js.os << "\"" << value << "\""; };
    void value(size_t value) { js.os << "\"" << value << "\""; };
    void value(int value) { js.os << "\"" << value << "\""; };
    void value(void *value) { js.os << "\"" << value << "\""; };
    ~json_obj() {
      js.indent.resize(js.indent.size() - js.tab_length);
      js.os << js.newline;
      js.os << js.indent;
      js.os << "}";
    }
  };

  class json_array {
    bool isFirst = true;
    json_stringstream &js;
  public:
    json_array(json_stringstream &json_ss) : js(json_ss) {
      if(!(js.os))
        return;
      js.os << "[" << js.newline;
      js.indent.append(js.tab_length, ' ');
      js.os << js.indent;
    }

    json_obj object() {
      if(!isFirst){
        js.os << "," << js.newline << js.indent;
      } else {
        isFirst = false;
      }
      return json_obj(js);      
    }

    template<class MemberT>
    void member(const MemberT &t) {
      if(!isFirst){
        js.os << "," << js.newline << js.indent;
      } else {
        isFirst = false;
      }
      js.os << t;    
    }

    ~json_array() {
      js.indent.resize(js.indent.size() - js.tab_length);
      js.os << js.newline;
      js.os << js.indent;
      js.os << "]";
    }
  };

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<const char *, std::decay_t<T>>>>
  json_stringstream &operator<<(T &&value) {
    os << std::forward<T>(value);
    return *this;
  }

  json_obj object(){
    return json_obj(*this);
  }
  json_array array(){
    return json_array(*this);
  }
};

template <>
inline json_stringstream::json_obj
json_stringstream::json_obj::value<json_stringstream::json_obj>() {
  return js.object();
}

template <typename T> inline std::string demangle_name() {
  std::string ret_str = "";
#if defined(__linux__)
  int s;
  auto mangle_name = typeid(T).name();
  auto demangle_name = abi::__cxa_demangle(mangle_name, NULL, NULL, &s);
  if (s != 0) {
    ret_str = "CODEPIN:ERROR:0: Unable to demangle symbol " +
              std::string(mangle_name) + ".";
  } else {
    ret_str = demangle_name;
    std::free(demangle_name);
  }
#else
  ret_str = typeid(T).name();
#endif
  return ret_str;
}

#ifdef __NVCC__
template <> inline std::string demangle_name<__half>() { return "fp16"; }
template <> inline std::string demangle_name<__nv_bfloat16>() { return "bf16"; }
#else
template <> inline std::string demangle_name<sycl::half>() { return "fp16"; }
template <> inline std::string demangle_name<sycl::ext::oneapi::bfloat16>() {
  return "bf16";
}
#endif

template <class T, class T2 = void> class data_ser {

public:
  static void dump(json_stringstream &ss, T value, queue_t queue) {
    auto obj = ss.object();
    obj.key("Data");
    obj.value("CODEPIN:ERROR:1: Unable to find the corresponding serialization "
              "function.");
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<T>()));
  }
};

template <class T>
class data_ser<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
  static void dump(json_stringstream &ss, const T &value, queue_t queue) {
    auto arr = ss.array();
    arr.member<T>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<T>()));
  }
};

#ifdef __NVCC__
template <> class data_ser<__half> {
public:
  static void dump(json_stringstream &ss, const __half &value, queue_t queue) {
    float f = __half2float(value);
    auto arr = ss.array();
    arr.member<float>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<__half>()));
  }
};
template <> class data_ser<__nv_bfloat16> {
public:
  static void dump(json_stringstream &ss, const __nv_bfloat16 &value,
                   queue_t queue) {
    float f = __bfloat162float(value);
    auto arr = ss.array();
    arr.member<float>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<__nv_bfloat16>()));
  }
};
#else
template <typename T>
class data_ser<T,
               typename std::enable_if<
                   std::is_same<T, sycl::half>::value ||
                   std::is_same<T, sycl::ext::oneapi::bfloat16>::value>::type> {
public:
  static void dump(json_stringstream &ss, const T &value, queue_t queue) {
    auto arr = ss.array();
    arr.member<T>(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj) {
    obj.key("Type");
    obj.value(std::string(demangle_name<T>()));
  }

};
#endif

#ifdef __NVCC__
template <> class data_ser<int3> {
public:
  static void dump(json_stringstream &ss, const int3 &value,
                   queue_t queue) {
    auto arr = ss.array();
    {
      auto obj_x = arr.object();
      obj_x.key("x");
      auto value_x =
          obj_x
              .value<json_stringstream::json_obj>();
      data_ser<int>::print_type_name(value_x);
      value_x.key("Data");
      data_ser<int>::dump(ss, value.x, queue);
    }
    {
      auto obj_y = arr.object();
      obj_y.key("y");
      auto value_y =
          obj_y
              .value<json_stringstream::json_obj>();
      data_ser<int>::print_type_name(value_y);
      value_y.key("Data");
      data_ser<int>::dump(ss, value.y, queue);
    }
    {
      auto obj_z = arr.object();
      obj_z.key("z");
      auto value_z =
          obj_z
              .value<json_stringstream::json_obj>();
      data_ser<int>::print_type_name(value_z);
      value_z.key("Data");
      data_ser<int>::dump(ss, value.z, queue);
    }
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("int3");
  }
};

template <> class data_ser<float3> {
public:
  static void dump(json_stringstream &ss, const float3 &value,
                   queue_t queue) {
    auto arr = ss.array();
    {
      auto obj_x = arr.object();
      obj_x.key("x");
      auto value_x =
          obj_x
              .value<json_stringstream::json_obj>();
      data_ser<float>::print_type_name(value_x);
      value_x.key("Data");
      data_ser<float>::dump(ss, value.x, queue);
    }
    {
      auto obj_y = arr.object();
      obj_y.key("y");
      auto value_y =
          obj_y
              .value<json_stringstream::json_obj>();
      data_ser<float>::print_type_name(value_y);
      value_y.key("Data");
      data_ser<float>::dump(ss, value.y, queue);
    }
    {
      auto obj_z = arr.object();
      obj_z.key("z");
      auto value_z =
          obj_z
              .value<json_stringstream::json_obj>();
      data_ser<float>::print_type_name(value_z);
      value_z.key("Data");
      data_ser<float>::dump(ss, value.z, queue);
    }
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("float3");
  }
};

#else
template <> class data_ser<sycl::int3> {
public:
  static void dump(json_stringstream &ss, const sycl::int3 &value,
                   queue_t queue) {
    auto arr = ss.array();
    {
      auto obj_x = arr.object();
      obj_x.key("x");
      auto value_x =
          obj_x
              .value<json_stringstream::json_obj>();
      data_ser<int>::print_type_name(value_x);
      value_x.key("Data");
      data_ser<int>::dump(ss, value.x(), queue);
    }
    {
      auto obj_y = arr.object();
      obj_y.key("y");
      auto value_y =
          obj_y
              .value<json_stringstream::json_obj>();
      data_ser<int>::print_type_name(value_y);
      value_y.key("Data");
      data_ser<int>::dump(ss, value.y(), queue);
    }
    {
      auto obj_z = arr.object();
      obj_z.key("z");
      auto value_z =
          obj_z
              .value<json_stringstream::json_obj>();
      data_ser<int>::print_type_name(value_z);
      value_z.key("Data");
      data_ser<int>::dump(ss, value.z(), queue);
    }
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("sycl::int3");
  }
};

template <> class data_ser<sycl::float3> {
public:
  static void dump(json_stringstream &ss, const sycl::float3 &value,
                   queue_t queue) {
    auto arr = ss.array();
    {
      auto obj_x = arr.object();
      obj_x.key("x");
      auto value_x =
          obj_x
              .value<json_stringstream::json_obj>();
      data_ser<float>::print_type_name(value_x);
      value_x.key("Data");
      data_ser<float>::dump(ss, value.x(), queue);
    }
    {
      auto obj_y = arr.object();
      obj_y.key("y");
      auto value_y =
          obj_y
              .value<json_stringstream::json_obj>();
      data_ser<float>::print_type_name(value_y);
      value_y.key("Data");
      data_ser<float>::dump(ss, value.y(), queue);
    }
    {
      auto obj_z = arr.object();
      obj_z.key("z");
      auto value_z =
          obj_z
              .value<json_stringstream::json_obj>();
      data_ser<float>::print_type_name(value_z);
      value_z.key("Data");
      data_ser<float>::dump(ss, value.z(), queue);
    }
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("sycl::float3");
  }
};
#endif

template <> class data_ser<char *> {
public:
  static void dump(json_stringstream &ss, const char *value,
                   queue_t queue) {
    auto obj = ss.object();
    obj.key("Data");
    const char *dump_addr = value;
    bool is_dev = is_dev_ptr((void*)value);
    if (is_dev) {
      const char *h_data = new char[strlen(value)];
#ifdef __NVCC__
      cudaMemcpyAsync((void *)h_data, (void *)value,
                      strlen(value) * sizeof(char), cudaMemcpyDeviceToHost,
                      queue);
      cudaStreamSynchronize(queue);
#else
      queue->memcpy((void *)h_data, (void *)value, strlen(value) * sizeof(char))
          .wait();
#endif
      dump_addr = h_data;    
    }
    obj.value(std::string(dump_addr));
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("char *");
  }
};

template <> class data_ser<std::string> {
public:
  static void dump(json_stringstream &ss, const std::string &value,
                   queue_t queue) {
    auto obj = ss.object();
    obj.key("Data");
    obj.value(value);
  }
  static void print_type_name(json_stringstream::json_obj &obj){
    obj.key("Type");
    obj.value("std::string");
  }
};

} // namespace detail
} // namespace codepin
} // namespace experimental
} // namespace dpct

#endif
