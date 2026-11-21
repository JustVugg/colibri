//==---- vector.h ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_VECTOR_H__
#define __DPCT_VECTOR_H__

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/memory>

#include "memory.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace dpct {

namespace internal {
template <typename Iter, typename Void = void> // for non-iterators
struct is_iterator : std::false_type {};

template <typename Iter> // For iterators
struct is_iterator<
    Iter,
    typename std::enable_if<
        !std::is_void<typename Iter::iterator_category>::value, void>::type>
    : std::true_type {};

template <typename T> // For pointers
struct is_iterator<T *> : std::true_type {};
} // end namespace internal

#ifndef DPCT_USM_LEVEL_NONE

// device_allocator_traits is a helper struct which makes use of custom
//  allocator constructor routines when they are specified serially on the host,
//  while enabling oneDPL pstl accelleration when such custom constructors are
//  not provided.
template <typename _Allocator> struct device_allocator_traits {

  // taken from libc++
  template <class, class _Alloc, class... _Args>
  struct __has_construct_impl : ::std::false_type {};

  template <class _Alloc, class... _Args>
  struct __has_construct_impl<decltype((void)std::declval<_Alloc>().construct(
                                  std::declval<_Args>()...)),
                              _Alloc, _Args...> : ::std::true_type {};

  template <class _Alloc, class... _Args>
  struct __has_construct : __has_construct_impl<void, _Alloc, _Args...> {};

  template <class _Alloc, class _Pointer, class = void>
  struct __has_destroy : ::std::false_type {};

  template <class _Alloc, class _Pointer>
  struct __has_destroy<_Alloc, _Pointer,
                       decltype((void)std::declval<_Alloc>().destroy(
                           std::declval<_Pointer>()))> : ::std::true_type {};
  // end of taken from libc++

  template <typename T, typename Size>
  static void uninitialized_value_construct_n(_Allocator alloc, T *p, Size n) {
    assert(p != nullptr && "value constructing null data");
    if constexpr (__has_construct<_Allocator, T *>::value) {
      for (Size i = 0; i < n; i++) {
        ::std::allocator_traits<_Allocator>::construct(alloc, p + i);
      }
    } else {
      ::std::uninitialized_value_construct_n(
          oneapi::dpl::execution::make_device_policy(
              ::dpct::cs::get_default_queue()),
          p, n);
    }
  }

  template <typename T, typename Size, typename Value>
  static void uninitialized_fill_n(_Allocator alloc, T *first, Size n,
                                   const Value &value) {
    assert(first != nullptr && "filling null data");
    if constexpr (__has_construct<_Allocator, T *, const Value &>::value) {
      for (Size i = 0; i < n; i++) {
        ::std::allocator_traits<_Allocator>::construct(alloc, first + i, value);
      }
    } else {
      ::std::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(
                                      ::dpct::cs::get_default_queue()),
                                  first, n, value);
    }
  }

  template <typename Iter1, typename Size, typename T>
  static void __uninitialized_custom_copy_n(_Allocator alloc, Iter1 first,
                                            Size n, T *d_first) {
    for (Size i = 0; i < n; i++) {
      ::std::allocator_traits<_Allocator>::construct(alloc, d_first + i,
                                                     *(first + i));
    }
  }

  template <typename Iter1, typename Size, typename T>
  static void uninitialized_device_copy_n(_Allocator alloc, Iter1 first, Size n,
                                          T *d_first) {
    assert(d_first != nullptr && "copying into null data");
    if constexpr (__has_construct<_Allocator, T *,
                                  typename ::std::iterator_traits<
                                      Iter1>::value_type>::value) {
      __uninitialized_custom_copy_n(alloc, first, n, d_first);
    } else {
      ::std::uninitialized_copy_n(oneapi::dpl::execution::make_device_policy(
                                      ::dpct::cs::get_default_queue()),
                                  first, n, d_first);
    }
  }

  template <typename Iter1, typename Size, typename T>
  static void uninitialized_host_copy_n(_Allocator alloc, Iter1 first, Size n,
                                        T *d_first) {
    assert(d_first != nullptr && "copying into null data");
    if constexpr (__has_construct<_Allocator, T *,
                                  typename ::std::iterator_traits<
                                      Iter1>::value_type>::value) {
      __uninitialized_custom_copy_n(alloc, first, n, d_first);
    } else {
      ::std::uninitialized_copy_n(first, n, d_first);
    }
  }

  template <typename T, typename Size>
  static void destroy_n(_Allocator alloc, T *p, Size n) {
    assert(p != nullptr && "destroying null data");
    if constexpr (__has_destroy<_Allocator, T *>::value) {
      for (Size i = 0; i < n; i++) {
        ::std::allocator_traits<_Allocator>::destroy(alloc, p + i);
      }
    } else {
      ::std::destroy_n(oneapi::dpl::execution::make_device_policy(
                           ::dpct::cs::get_default_queue()),
                       p, n);
    }
  }
};

template <typename T,
          typename Allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>>
class device_vector {
public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = device_pointer<T>;
  using const_pointer = device_pointer<const T>;
  using difference_type =
      typename ::std::iterator_traits<iterator>::difference_type;
  using size_type = ::std::size_t;
  using allocator_type = Allocator;
  using alloc_traits = ::std::allocator_traits<Allocator>;

private:
  Allocator _alloc;
  size_type _size;
  size_type _capacity;
  T *_storage;

  size_type _min_capacity() const { return size_type(1); }

  void _set_capacity_and_alloc() {
    _capacity = (::std::max)(_size * 2, _min_capacity());
    _storage = alloc_traits::allocate(_alloc, _capacity);
  }

  void _construct(size_type n, size_type start_idx = 0) {
    if (n > 0) {
      device_allocator_traits<Allocator>::uninitialized_value_construct_n(
          _alloc, _storage + start_idx, n);
    }
  }

  void _construct(size_type n, const T &value, size_type start_idx = 0) {
    if (n > 0) {
      device_allocator_traits<Allocator>::uninitialized_fill_n(
          _alloc, _storage + start_idx, n, value);
    }
  }

  template <typename Iter>
  void _construct_iter(Iter first, size_type n, size_type start_idx = 0) {
    if (n > 0) {
      device_allocator_traits<Allocator>::uninitialized_device_copy_n(
          _alloc, first, n, _storage + start_idx);
    }
  }

  template <typename Iter>
  void _construct_iter_host(Iter first, size_type n, size_type start_idx = 0) {
    if (n > 0) {
      device_allocator_traits<Allocator>::uninitialized_host_copy_n(
          _alloc, first, n, _storage + start_idx);
    }
  }

  void _destroy(size_type n, size_type start_idx = 0) {
    if (n > 0) {
      device_allocator_traits<Allocator>::destroy_n(_alloc,
                                                    _storage + start_idx, n);
    }
  }

  void _assign_elements(const device_vector &other) {
    if (other.size() <= _size) {
      // if incoming elements fit within existing elements, copy then destroy
      // excess
      ::std::copy(oneapi::dpl::execution::make_device_policy(
                      ::dpct::cs::get_default_queue()),
                  other.begin(), other.end(), begin());
      resize(other.size());
    } else if (other.size() < _capacity) {
      // if incoming elements don't fit within existing elements but do fit
      // within total capacity
      // copy elements that fit, then use uninitialized copy to ge the rest
      // and adjust size
      std::copy_n(oneapi::dpl::execution::make_device_policy(
                      ::dpct::cs::get_default_queue()),
                  other.begin(), _size, begin());
      device_allocator_traits<Allocator>::uninitialized_device_copy_n(
          _alloc, other.begin() + _size, other.size() - _size,
          _storage + _size);
      _size = other.size();
    } else {
      // If incoming elements exceed current capacity, destroy all existing
      // elements, then allocate incoming vectors capacity, then use
      // uninitialized copy
      clear();
      reserve(other.capacity());
      device_allocator_traits<Allocator>::uninitialized_device_copy_n(
          _alloc, other.begin(), other.size(), _storage);
      _size = other.size();
    }
  }

public:
  template <typename OtherA> operator ::std::vector<T, OtherA>() const {
    auto __tmp = ::std::vector<T, OtherA>(this->size());
    ::std::copy(oneapi::dpl::execution::make_device_policy(
                    ::dpct::cs::get_default_queue()),
                this->begin(), this->end(), __tmp.begin());
    return __tmp;
  }

  device_vector(
      const Allocator &alloc = Allocator(::dpct::cs::get_default_queue()))
      : _alloc(alloc), _size(0), _capacity(_min_capacity()) {
    _set_capacity_and_alloc();
  }

  ~device_vector() /*= default*/ {
    clear();
    alloc_traits::deallocate(_alloc, _storage, _capacity);
  }

  explicit device_vector(size_type n, const Allocator &alloc = Allocator(
                                          ::dpct::cs::get_default_queue()))
      : _alloc(alloc), _size(n) {
    _set_capacity_and_alloc();
    _construct(n);
  }

  explicit device_vector(
      size_type n, const T &value,
      const Allocator &alloc = Allocator(::dpct::cs::get_default_queue()))
      : _alloc(alloc), _size(n) {
    _set_capacity_and_alloc();
    _construct(n, value);
  }

  device_vector(device_vector &&other)
      : _alloc(std::move(other._alloc)), _size(other.size()),
        _capacity(other.capacity()), _storage(other._storage) {
    other._size = 0;
    other._capacity = 0;
    other._storage = nullptr;
  }

  device_vector(device_vector &&other, const Allocator &alloc)
      : _alloc(alloc), _size(other.size()), _capacity(other.capacity()) {
    _storage = alloc_traits::allocate(_alloc, _capacity);
    _construct_iter(other.begin(), _size); // ok to parallelize
    other._size = 0;
    other._capacity = 0;
    other._storage = nullptr;
  }

  template <typename InputIterator>
  device_vector(
      InputIterator first,
      typename ::std::enable_if_t<
          dpct::internal::is_iterator<InputIterator>::value, InputIterator>
          last,
      const Allocator &alloc = Allocator(::dpct::cs::get_default_queue()))
      : _alloc(alloc) {
    _size = ::std::distance(first, last);
    _set_capacity_and_alloc();
    // unsafe to parallelize on device as we dont know if InputIterator is
    // valid oneDPL input type
    _construct_iter_host(first, _size);
  }

  device_vector(const device_vector &other, const Allocator &alloc)
      : _alloc(alloc) {
    _size = other.size();
    _capacity = other.capacity();
    _storage = alloc_traits::allocate(_alloc, _capacity);
    _construct_iter(other.begin(), _size);
  }

  device_vector(const device_vector &other)
      : device_vector(
            other,
            alloc_traits::select_on_container_copy_construction(other._alloc)) {
  }

  template <typename OtherAllocator>
  device_vector(
      const device_vector<T, OtherAllocator> &other,
      const Allocator &alloc = Allocator(::dpct::cs::get_default_queue()))
      : _alloc(alloc) {
    _size = other.size();
    _capacity = other.capacity();
    _storage = alloc_traits::allocate(_alloc, _capacity);
    _construct_iter(other.begin(), _size);
  }

  template <typename OtherAllocator>
  device_vector(const ::std::vector<T, OtherAllocator> &v)
      : device_vector(v.begin(), v.end()) {}

  template <typename OtherAllocator>
  device_vector(const ::std::vector<T, OtherAllocator> &v,
                const Allocator &alloc)
      : device_vector(v.begin(), v.end(), alloc) {}

  template <typename OtherAllocator>
  device_vector &operator=(const ::std::vector<T, OtherAllocator> &v) {
    resize(v.size());
    ::std::copy(oneapi::dpl::execution::make_device_policy(
                    ::dpct::cs::get_default_queue()),
                v.begin(), v.end(), begin());
    return *this;
  }

  device_vector &operator=(const device_vector &other) {
    // Copy assignment operator:
    if constexpr (alloc_traits::propagate_on_container_copy_assignment::value) {
      clear();
      alloc_traits::deallocate(_alloc, _storage, _capacity);
      _capacity = 0;
      _alloc = other._alloc;
    }
    _assign_elements(other);
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    // Move assignment operator:
    if constexpr (alloc_traits::propagate_on_container_move_assignment::value) {
      clear();
      alloc_traits::deallocate(_alloc, _storage, _capacity);
      _alloc = ::std::move(other._alloc);
      _storage = ::std::move(other._storage);
      _capacity = ::std::move(other._capacity);
      _size = ::std::move(other._size);
    } else {
      _assign_elements(other);
      // destroy and deallocate other vector
      other.clear();
      alloc_traits::deallocate(other._alloc, other._storage, other._capacity);
    }
    other._size = 0;
    other._capacity = 0;
    other._storage = nullptr;
    return *this;
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(_storage, 0); }
  iterator end() { return device_iterator<T>(_storage, size()); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(_storage, 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(_storage, size()); }
  const_iterator cend() const { return end(); }
  T *real_begin() { return _storage; }
  const T *real_begin() const { return _storage; }
  void swap(device_vector &v) {
    if constexpr (::std::allocator_traits<
                      Allocator>::propagate_on_container_swap::value) {
      ::std::swap(_alloc, v._alloc);
      ::std::swap(_size, v._size);
      ::std::swap(_capacity, v._capacity);
      ::std::swap(_storage, v._storage);
    } else {
      // swap all elements up to the minimum size between the two vectors
      size_type min_size = (::std::min)(size(), v.size());
      auto zip = oneapi::dpl::make_zip_iterator(begin(), v.begin());
      ::std::for_each(oneapi::dpl::execution::make_device_policy(
                          ::dpct::cs::get_default_queue()),
                      zip, zip + min_size, [](auto zip_ele) {
                        std::swap(::std::get<0>(zip_ele),
                                  ::std::get<1>(zip_ele));
                      });
      // then copy the elements beyond the end of the smaller list, and resize
      if (size() > v.size()) {
        v.reserve(capacity());
        device_allocator_traits<Allocator>::uninitialized_device_copy_n(
            _alloc, begin() + min_size, size() - min_size,
            v._storage + min_size);
        v._size = size();
        _destroy(size() - min_size, min_size);
        _size = min_size;
      } else if (size() < v.size()) {
        reserve(v.capacity());
        device_allocator_traits<Allocator>::uninitialized_device_copy_n(
            _alloc, v.begin() + min_size, v.size() - min_size,
            _storage + min_size);
        _size = v.size();
        v._destroy(v.size() - min_size, min_size);
        v._size = min_size;
      }
    }
  }
  reference operator[](size_type n) { return _storage[n]; }
  const_reference operator[](size_type n) const { return _storage[n]; }
  void reserve(size_type n) {
    if (n > capacity()) {
      // allocate buffer for new size
      auto tmp = alloc_traits::allocate(_alloc, n);
      // copy content (old buffer to new buffer)
      if (capacity() > 0) {
        device_allocator_traits<Allocator>::uninitialized_device_copy_n(
            _alloc, begin(), _size, tmp);
        alloc_traits::deallocate(_alloc, _storage, _capacity);
      }
      _storage = tmp;
      _capacity = n;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    if (new_size > size()) {
      _construct(new_size - size(), x, size());
    } else {
      _destroy(_size - new_size, new_size);
    }
    _size = new_size;
  }
  size_type max_size(void) const {
    return ::std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const { return _capacity; }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return pointer(_storage); }
  const_pointer data(void) const { return const_pointer(_storage); }
  void shrink_to_fit(void) {
    if (_size != capacity() && capacity() > _min_capacity()) {
      size_type tmp_capacity = (::std::max)(_size, _min_capacity());
      auto tmp = alloc_traits::allocate(_alloc, tmp_capacity);
      if (_size > 0) {
        ::std::copy(oneapi::dpl::execution::make_device_policy(
                        ::dpct::cs::get_default_queue()),
                    begin(), end(), tmp);
      }
      alloc_traits::deallocate(_alloc, _storage, _capacity);
      _storage = tmp;
      _capacity = tmp_capacity;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    if (_size > 0) {
      ::std::fill(oneapi::dpl::execution::make_device_policy(
                      ::dpct::cs::get_default_queue()),
                  begin(), begin() + n, x);
    }
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename ::std::enable_if<internal::is_iterator<InputIterator>::value,
                                   InputIterator>::type last) {
    auto n = ::std::distance(first, last);
    resize(n);
    if (_size > 0) {
      // unsafe to call on device as we don't know the InputIterator type
      ::std::copy(first, last, begin());
    }
  }
  void clear(void) {
    _destroy(_size);
    _size = 0;
  }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0) {
      _destroy(1, _size - 1);
      --_size;
    }
  }
  iterator erase(iterator first, iterator last) {
    auto n = ::std::distance(first, last);
    if (last == end()) {
      _destroy(n, _size - n);
      _size = _size - n;
      return end();
    }
    auto m = ::std::distance(last, end());
    if (m <= 0) {
      return end();
    } else if (n >= m) {
      ::std::copy(oneapi::dpl::execution::make_device_policy(
                      ::dpct::cs::get_default_queue()),
                  last, last + m, first);
    } else {
      auto tmp = alloc_traits::allocate(_alloc, m);

      device_allocator_traits<Allocator>::uninitialized_device_copy_n(
          _alloc, last, m, tmp);

      std::copy(oneapi::dpl::execution::make_device_policy(
                    ::dpct::cs::get_default_queue()),
                tmp, tmp + m, first);
      device_allocator_traits<Allocator>::destroy_n(_alloc, tmp, m);
      alloc_traits::deallocate(_alloc, tmp, m);
    }
    // now destroy the remaining elements
    _destroy(n, size() - n);
    _size -= n;
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = ::std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      reserve(size() + n);
      device_allocator_traits<Allocator>::uninitialized_fill_n(
          _alloc, _storage + size(), n, x);
      _size += n;
    } else {
      auto i_n = ::std::distance(begin(), position);
      // allocate temporary storage
      auto m = ::std::distance(position, end());
      // will throw if position is not inside active vector
      auto tmp = alloc_traits::allocate(_alloc, m);
      // copy remainder
      device_allocator_traits<Allocator>::uninitialized_device_copy_n(
          _alloc, position, m, tmp);
      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();
      ::std::fill(oneapi::dpl::execution::make_device_policy(
                      ::dpct::cs::get_default_queue()),
                  position, position + n, x);
      ::std::copy(oneapi::dpl::execution::make_device_policy(
                      ::dpct::cs::get_default_queue()),
                  tmp, tmp + m, position + n);
      device_allocator_traits<Allocator>::destroy_n(_alloc, tmp, m);
      alloc_traits::deallocate(_alloc, tmp, m);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename ::std::enable_if<internal::is_iterator<InputIterator>::value,
                                   InputIterator>::type last) {
    auto n = ::std::distance(first, last);
    if (position == end()) {
      reserve(size() + n);
      // unsafe to call on device as we dont know the InputIterator type
      ::std::uninitialized_copy(first, last, end());
      _size += n;
    } else {
      auto m = ::std::distance(position, end());
      // will throw if position is not inside active vector
      auto tmp = alloc_traits::allocate(_alloc, m);

      device_allocator_traits<Allocator>::uninitialized_device_copy_n(
          _alloc, position, m, tmp);

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();
      // unsafe to call on device as we dont know the InputIterator type
      ::std::copy(first, last, position);
      ::std::copy(oneapi::dpl::execution::make_device_policy(
                      ::dpct::cs::get_default_queue()),
                  tmp, tmp + m, position + n);
      device_allocator_traits<Allocator>::destroy_n(_alloc, tmp, m);
      alloc_traits::deallocate(_alloc, tmp, m);
    }
  }
  Allocator get_allocator() const { return _alloc; }
};

#else

template <typename T, typename Allocator = detail::__buffer_allocator<T>>
class device_vector {
  static_assert(
      std::is_same<Allocator, detail::__buffer_allocator<T>>::value,
      "device_vector doesn't support custom allocator when USM is not used.");

public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = device_pointer<T>;
  using const_pointer = device_pointer<const T>;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

private:
  using Buffer = sycl::buffer<T, 1>;
  using Range = sycl::range<1>;
  // Using mem_mgr to handle memory allocation
  void *_storage;
  size_type _size;

  size_type _min_capacity() const { return size_type(1); }

  void *alloc_store(size_type num_bytes) {
    return detail::mem_mgr::instance().mem_alloc(num_bytes);
  }

public:
  template <typename OtherA> operator std::vector<T, OtherA>() const {
    auto __tmp = std::vector<T, OtherA>(this->size());
    std::copy(oneapi::dpl::execution::dpcpp_default, this->begin(), this->end(),
              __tmp.begin());
    return __tmp;
  }
  device_vector()
      : _storage(alloc_store(_min_capacity() * sizeof(T))), _size(0) {}
  ~device_vector() = default;
  explicit device_vector(size_type n) : device_vector(n, T()) {}
  explicit device_vector(size_type n, const T &value)
      : _storage(alloc_store((std::max)(n, _min_capacity()) * sizeof(T))),
        _size(n) {
    auto buf = get_buffer();
    std::fill(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(buf),
              oneapi::dpl::begin(buf) + n, T(value));
  }
  device_vector(const device_vector &other)
      : _storage(other._storage), _size(other.size()) {}
  device_vector(device_vector &&other)
      : _storage(std::move(other._storage)), _size(other.size()) {}

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !std::is_pointer<InputIterator>::value &&
                        std::is_same<typename std::iterator_traits<
                                         InputIterator>::iterator_category,
                                     std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(
                  ::dpct::cs::get_default_queue()),
              first, last, dst);
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<std::is_pointer<InputIterator>::value,
                                        InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    Buffer tmp_buf(first, last);
    auto start = oneapi::dpl::begin(tmp_buf);
    auto end = oneapi::dpl::end(tmp_buf);
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(
                  ::dpct::cs::get_default_queue()),
              start, end, dst);
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !std::is_same<typename std::iterator_traits<
                                          InputIterator>::iterator_category,
                                      std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    std::vector<T> tmp(first, last);
    Buffer tmp_buf(tmp);
    auto start = oneapi::dpl::begin(tmp_buf);
    auto end = oneapi::dpl::end(tmp_buf);
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(
                  ::dpct::cs::get_default_queue()),
              start, end, dst);
  }

  template <typename OtherAllocator>
  device_vector(const device_vector<T, OtherAllocator> &v)
      : _storage(alloc_store(v.size() * sizeof(T))), _size(v.size()) {
    auto buf = get_buffer();
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(
                  ::dpct::cs::get_default_queue()),
              v.real_begin(), v.real_begin() + v.size(), dst);
  }

  template <typename OtherAllocator>
  device_vector(std::vector<T, OtherAllocator> &v)
      : _storage(alloc_store(v.size() * sizeof(T))), _size(v.size()) {
    std::copy(oneapi::dpl::execution::dpcpp_default, v.begin(), v.end(),
              oneapi::dpl::begin(get_buffer()));
  }

  device_vector &operator=(const device_vector &other) {
    // Copy assignment operator:
    _size = other.size();
    void *tmp = alloc_store(_size * sizeof(T));
    auto tmp_buf =
        detail::mem_mgr::instance()
            .translate_ptr(tmp)
            .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
    std::copy(oneapi::dpl::execution::dpcpp_default,
              oneapi::dpl::begin(other.get_buffer()),
              oneapi::dpl::end(other.get_buffer()),
              oneapi::dpl::begin(tmp_buf));
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = tmp;
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    // Move assignment operator:
    _size = other.size();
    this->_storage = std::move(other._storage);
    return *this;
  }
  template <typename OtherAllocator>
  device_vector &operator=(const std::vector<T, OtherAllocator> &v) {
    Buffer data(v.begin(), v.end());
    _size = v.size();
    void *tmp = alloc_store(_size * sizeof(T));
    auto tmp_buf =
        detail::mem_mgr::instance()
            .translate_ptr(tmp)
            .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
    std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(data),
              oneapi::dpl::end(data), oneapi::dpl::begin(tmp_buf));
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = tmp;

    return *this;
  }
  Buffer get_buffer() const {
    return detail::mem_mgr::instance()
        .translate_ptr(_storage)
        .buffer.template reinterpret<T, 1>(sycl::range<1>(capacity()));
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(get_buffer(), 0); }
  iterator end() { return device_iterator<T>(get_buffer(), _size); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(get_buffer(), 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(get_buffer(), _size); }
  const_iterator cend() const { return end(); }
  T *real_begin() {
    // This code returns a pointer to a data within sycl buffer accessor which
    // is leaving scope. This relies on undefined
    // behavior and may not provide a valid pointer to data inside that buffer.
    return (detail::mem_mgr::instance()
                .translate_ptr(_storage)
                .buffer.get_host_access())
        .get_pointer();
  }
  const T *real_begin() const {
    // This code returns a pointer to a data within sycl buffer accessor which
    // is leaving scope. This relies on undefined
    // behavior and may not provide a valid pointer to data inside that buffer.
    return const_cast<device_vector *>(this)->real_begin();
  }
  void swap(device_vector &v) {
    void *temp = v._storage;
    v._storage = this->_storage;
    this->_storage = temp;
    std::swap(_size, v._size);
  }
  reference operator[](size_type n) { return *(begin() + n); }
  const_reference operator[](size_type n) const { return *(begin() + n); }
  void reserve(size_type n) {
    if (n > capacity()) {
      // create new buffer (allocate for new size)
      void *a = alloc_store(n * sizeof(T));

      // copy content (old buffer to new buffer)
      if (_storage != nullptr) {
        auto tmp = detail::mem_mgr::instance()
                       .translate_ptr(a)
                       .buffer.template reinterpret<T, 1>(sycl::range<1>(n));
        auto src_buf = get_buffer();
        std::copy(oneapi::dpl::execution::dpcpp_default,
                  oneapi::dpl::begin(src_buf), oneapi::dpl::end(src_buf),
                  oneapi::dpl::begin(tmp));

        // deallocate old memory
        detail::mem_mgr::instance().mem_free(_storage);
      }
      _storage = a;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    if (_size < new_size) {
      auto src_buf = get_buffer();
      std::fill(oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::begin(src_buf) + _size,
                oneapi::dpl::begin(src_buf) + new_size, x);
    }
    _size = new_size;
  }
  size_type max_size(void) const {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const {
    return _storage != nullptr ? detail::mem_mgr::instance()
                                         .translate_ptr(_storage)
                                         .buffer.size() /
                                     sizeof(T)
                               : 0;
  }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return pointer(reinterpret_cast<T *>(_storage)); }
  const_pointer data(void) const {
    return const_pointer(reinterpret_cast<const T *>(_storage));
  }
  void shrink_to_fit(void) {
    if (_size != capacity()) {
      void *a = alloc_store(_size * sizeof(T));
      auto tmp = detail::mem_mgr::instance()
                     .translate_ptr(a)
                     .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
      std::copy(oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::begin(get_buffer()),
                oneapi::dpl::begin(get_buffer()) + _size,
                oneapi::dpl::begin(tmp));
      detail::mem_mgr::instance().mem_free(_storage);
      _storage = a;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    std::fill(oneapi::dpl::execution::dpcpp_default, begin(), begin() + n, x);
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    resize(n);
    if (internal::is_iterator<InputIterator>::value &&
        !std::is_pointer<InputIterator>::value)
      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, begin());
    else {
      Buffer tmp(first, last);
      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), begin());
    }
  }
  void clear(void) {
    _size = 0;
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = nullptr;
  }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
  iterator erase(iterator first, iterator last) {
    auto n = std::distance(first, last);
    if (last == end()) {
      _size = _size - n;
      return end();
    }
    Buffer tmp{Range(std::distance(last, end()))};
    // copy remainder to temporary buffer.
    std::copy(oneapi::dpl::execution::dpcpp_default, last, end(),
              oneapi::dpl::begin(tmp));
    // override (erase) subsequence in storage.
    std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
              oneapi::dpl::end(tmp), first);
    resize(_size - n);
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      resize(size() + n);
      std::fill(oneapi::dpl::execution::dpcpp_default, end() - n, end(), x);
    } else {
      auto i_n = std::distance(begin(), position);
      // allocate temporary storage
      Buffer tmp{Range(std::distance(position, end()))};
      // copy remainder
      std::copy(oneapi::dpl::execution::dpcpp_default, position, end(),
                oneapi::dpl::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::fill(oneapi::dpl::execution::dpcpp_default, position, position + n,
                x);

      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), position + n);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    if (position == end()) {
      resize(size() + n);
      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, end());
    } else {
      Buffer tmp{Range(std::distance(position, end()))};

      std::copy(oneapi::dpl::execution::dpcpp_default, position, end(),
                oneapi::dpl::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, position);
      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), position + n);
    }
  }
};

#endif

} // end namespace dpct

#endif
