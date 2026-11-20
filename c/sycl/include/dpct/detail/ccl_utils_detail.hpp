//==---- ccl_utils_detail.hpp----------------------------*- C++ -*----------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_CCL_UTILS_DETAIL_HPP__
#define __DPCT_CCL_UTILS_DETAIL_HPP__

namespace dpct::ccl::detail {

/// Get stored kvs with specified kvs address.
inline std::shared_ptr<oneapi::ccl::kvs> &
get_kvs(const oneapi::ccl::kvs::address_type &addr) {
  struct hash {
    std::size_t operator()(const oneapi::ccl::kvs::address_type &in) const {
      return std::hash<std::string_view>()(
          std::string_view(in.data(), in.size()));
    }
  };
  static std::unordered_map<oneapi::ccl::kvs::address_type,
                            std::shared_ptr<oneapi::ccl::kvs>, hash>
      kvs_map;
  return kvs_map[addr];
}

/// Help class to init ccl environment.
class ccl_init_helper {
public:
  ccl_init_helper() { oneapi::ccl::init(); }
};

} // namespace dpct::ccl::detail

#endif // __DPCT_CCL_UTILS_DETAIL_HPP__
