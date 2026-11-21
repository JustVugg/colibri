//==---- dnnl_utils_detail.hpp----------------------------*- C++ -*---------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_DNNL_UTILS_DETAIL_HPP__
#define __DPCT_DNNL_UTILS_DETAIL_HPP__

namespace dpct::dnnl::detail {

typedef std::string primitive_cache_key_type;
typedef std::list<primitive_cache_key_type> usage_list_type;
struct primitive_cache_value_type {
  ::dnnl::primitive *_primitive;
  std::unordered_map<int, ::dnnl::memory> *_args;
  usage_list_type::iterator _usage_it;
  std::function<void(::dnnl::primitive *)> _destructor;
  sycl::event _e;
  sycl::queue _q;
  primitive_cache_value_type(
      ::dnnl::primitive *primitive,
      std::unordered_map<int, ::dnnl::memory> *args,
      usage_list_type::iterator usage_it,
      std::function<void(::dnnl::primitive *)> destructor, sycl::event e,
      sycl::queue q)
      : _primitive(primitive), _args(args), _usage_it(usage_it),
        _destructor(destructor), _e(e), _q(q) {}
};
struct primitive_and_args {
  ::dnnl::primitive *primitive;
  std::unordered_map<int, ::dnnl::memory> *args;
};
typedef std::unordered_map<primitive_cache_key_type,
                           std::shared_ptr<primitive_cache_value_type>>
    cache_map_type;

// The primitive cache uses LRU replacement policy, and the default cache
// capacity is 1024.
class primitive_cache {
  int _capacity = 1024;
  usage_list_type usage;
  cache_map_type cache_map;
  void touch(cache_map_type::iterator it, sycl::event e = {},
             bool update_event = false) {
    if (it->second->_usage_it != usage.begin()) {
      const primitive_cache_key_type &key = it->first;
      usage.erase(it->second->_usage_it);
      usage.push_front(key);
      it->second->_usage_it = usage.begin();
    }
    if (update_event) {
      it->second->_e = e;
    }
  }

public:
  std::shared_ptr<primitive_cache_value_type>
  get(const primitive_cache_key_type &key) {
    auto it = cache_map.find(key);
    if (it == cache_map.end()) {
      return nullptr;
    }
    touch(it);
    return it->second;
  }
  void put(const primitive_cache_key_type &key, ::dnnl::primitive *value,
           std::unordered_map<int, ::dnnl::memory> *args,
           std::function<void(::dnnl::primitive *)> destructor, sycl::event e,
           sycl::queue *q) {
    auto it = cache_map.find(key);
    if (it != cache_map.end()) {
      touch(it, e, true);
    } else {
      if (cache_map.size() == _capacity) {
        auto v = *(cache_map.find(usage.back())->second);
        v._q.submit([=](sycl::handler &cgh) {
          cgh.depends_on(v._e);
          cgh.host_task([=] {
            delete v._args;
            v._destructor(v._primitive);
          });
        });
        cache_map.erase(usage.back());
        usage.pop_back();
      }
      usage.push_front(key);
      cache_map[key] = std::make_shared<primitive_cache_value_type>(
          value, args, usage.begin(), destructor, e, *q);
    }
  }
};

} // namespace dpct::dnnl::detail

#endif // __DPCT_DNNL_UTILS_DETAIL_HPP__
