//==---- iterator_adaptor.h -----------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_ITERATOR_ADAPTOR_H__
#define __DPCT_ITERATOR_ADAPTOR_H__

#include <boost/iterator_adaptors.hpp>

namespace dpct {

typedef boost::use_default use_default;

typedef boost::iterator_core_access iterator_core_access;

template <class Derived, class Base, class Value = use_default,
          class Traversal = use_default, class System = use_default,
          class Reference = use_default, class Difference = use_default>
class iterator_adaptor
    : public boost::iterator_adaptor<Derived, Base, Value, Traversal, Reference,
                                     Difference> {
protected:
  typedef boost::iterator_adaptor<Derived, Base, Value, Traversal, Reference,
                                  Difference>
      base_t;

  // Note: System tag is not used to dispatch to any specific backend, this is
  //       controlled explicitly by the user with execution policies. System tag
  //       is kept for potential future extension by specialization based on
  //       this part of the type definition.

public:
  // It is the user's responsibility to ensure extra data is available on the
  // device if they want to use this iterator with device execution policies
  using is_passed_directly = ::std::true_type;
  using iterator_category = ::std::random_access_iterator_tag;

  iterator_adaptor() {}

  explicit iterator_adaptor(Base const &iter) : base_t(iter) {}

  typename base_t::reference
  operator[](typename base_t::difference_type n) const {
    return *(this->derived() + n);
  }
};

} // end namespace dpct

#endif
