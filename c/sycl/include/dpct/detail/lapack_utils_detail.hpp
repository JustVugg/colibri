//==---- lapack_utils_detail.hpp ------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LAPACK_UTILS_DETAIL_HPP__
#define __DPCT_LAPACK_UTILS_DETAIL_HPP__

namespace dpct::lapack::detail {
template <template <typename> typename functor_t, typename... args_t>
inline int lapack_shim(sycl::queue &q, library_data_t a_type, int *info,
                       std::string const &lapack_api_name, args_t &&...args) {
  auto handle_lapack_exception = [&](const oneapi::mkl::lapack::exception &e) {
    std::cerr << "Unexpected exception caught during call to LAPACK API: "
              << lapack_api_name << std::endl
              << "reason: " << e.what() << std::endl
              << "info: " << e.info() << std::endl
              << "detail: " << e.detail() << std::endl;
    if (e.info() < std::numeric_limits<int>::min() ||
        e.info() > std::numeric_limits<int>::max()) {
      throw std::runtime_error("e.info() exceeds the limit of int type");
    }
    int info_val = static_cast<int>(e.info());
    if (info)
      ::dpct::cs::memcpy(q, info, &info_val, sizeof(int),
                         ::dpct::cs::memcpy_direction::host_to_device)
          .wait();
    return 1;
  };
  try {
    switch (a_type) {
    case library_data_t::real_float: {
      functor_t<float>()(std::forward<args_t>(args)...);
      break;
    }
    case library_data_t::real_double: {
      functor_t<double>()(std::forward<args_t>(args)...);
      break;
    }
    case library_data_t::complex_float: {
      functor_t<std::complex<float>>()(std::forward<args_t>(args)...);
      break;
    }
    case library_data_t::complex_double: {
      functor_t<std::complex<double>>()(std::forward<args_t>(args)...);
      break;
    }
    default:
      throw std::runtime_error("the data type is unsupported");
    }
  } catch (oneapi::mkl::lapack::batch_error const &be) {
    try {
      std::rethrow_exception(be.exceptions()[0]);
    } catch (oneapi::mkl::lapack::exception &e) {
      return handle_lapack_exception(e);
    }
  } catch (oneapi::mkl::lapack::exception const &e) {
    return handle_lapack_exception(e);
  } catch (sycl::exception const &e) {
    std::cerr << "Caught synchronous SYCL exception:" << std::endl
              << "reason: " << e.what() << std::endl;
    if (info)
      ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int)).wait();
    return 1;
  }
  return 0;
}

template <typename T> class working_memory {
public:
  working_memory(std::size_t element_number, const sycl::queue &q) : _q(q) {
    _ptr = ::dpct::cs::malloc(element_number * sizeof(T), _q);
  }
  auto get_memory() { return dpct::detail::get_memory<T>(_ptr); }
  auto get_ptr() { return _ptr; }
  void set_event(sycl::event e) { _e = e; }
  ~working_memory() {
    if (_ptr) {
      ::dpct::cs::enqueue_free({_ptr}, {_e}, _q);
    }
  }

private:
  void *_ptr = nullptr;
  sycl::event _e;
  sycl::queue _q;
};

std::size_t byte_to_element_number(std::size_t size_in_byte,
                                   dpct::library_data_t element_type) {
  auto dv = std::lldiv(
      size_in_byte,
      dpct::detail::library_data_size[static_cast<unsigned int>(element_type)] /
          8);
  if (dv.rem) {
    throw std::runtime_error(
        "size_in_byte is not divisible by the size of element (in bytes)");
  }
  return dv.quot;
}
std::size_t element_number_to_byte(std::size_t size_in_element,
                                   dpct::library_data_t element_type) {
  auto dv = std::lldiv(
      dpct::detail::library_data_size[static_cast<unsigned int>(element_type)],
      8);
  if (dv.rem) {
    throw std::runtime_error(
        "the size of element (in bits) is not divisible by 8");
  }
  return size_in_element * dv.quot;
}

inline oneapi::mkl::jobsvd char2jobsvd(signed char job) {
  switch (job) {
  case 'A':
    return oneapi::mkl::jobsvd::vectors;
  case 'S':
    return oneapi::mkl::jobsvd::somevec;
  case 'O':
    return oneapi::mkl::jobsvd::vectorsina;
  case 'N':
    return oneapi::mkl::jobsvd::novec;
  default:
    throw std::runtime_error("the job type is unsupported");
  }
}

template <typename T> struct getrf_scratchpad_size_impl {
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, std::int64_t lda,
                  std::size_t &device_ws_size) {
    device_ws_size =
        oneapi::mkl::lapack::getrf_scratchpad_size<T>(q, m, n, lda);
  }
};

template <typename T> struct getrf_impl {
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  std::int64_t *ipiv, void *device_ws,
                  std::size_t device_ws_size, int *info) {
    auto ipiv_data = dpct::detail::get_memory<std::int64_t>(ipiv);
    auto a_data = dpct::detail::get_memory<T>(a);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    oneapi::mkl::lapack::getrf(q, m, n, a_data, lda, ipiv_data, device_ws_data,
                               device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
  }
};

template <typename T> struct getrs_impl {
  void operator()(sycl::queue &q, oneapi::mkl::transpose trans, std::int64_t n,
                  std::int64_t nrhs, library_data_t a_type, void *a,
                  std::int64_t lda, std::int64_t *ipiv, library_data_t b_type,
                  void *b, std::int64_t ldb, int *info) {
    auto ipiv_data = dpct::detail::get_memory<std::int64_t>(ipiv);
    std::int64_t device_ws_size = oneapi::mkl::lapack::getrs_scratchpad_size<T>(
        q, trans, n, nrhs, lda, ldb);
    working_memory<T> device_ws(device_ws_size, q);
    auto device_ws_data = device_ws.get_memory();
    auto a_data = dpct::detail::get_memory<T>(a);
    auto b_data = dpct::detail::get_memory<T>(b);
    oneapi::mkl::lapack::getrs(q, trans, n, nrhs, a_data, lda, ipiv_data,
                               b_data, ldb, device_ws_data, device_ws_size);
    sycl::event e = ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
    device_ws.set_event(e);
  }
};

template <typename T> struct geqrf_scratchpad_size_impl {
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, std::int64_t lda,
                  std::size_t &device_ws_size) {
    device_ws_size =
        oneapi::mkl::lapack::geqrf_scratchpad_size<T>(q, m, n, lda);
  }
};

template <typename T> struct geqrf_impl {
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  library_data_t tau_type, void *tau, void *device_ws,
                  std::size_t device_ws_size, int *info) {
    auto a_data = dpct::detail::get_memory<T>(a);
    auto tau_data = dpct::detail::get_memory<T>(tau);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    oneapi::mkl::lapack::geqrf(q, m, n, a_data, lda, tau_data, device_ws_data,
                               device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
  }
};

template <typename T> struct getrfnp_impl {
  void operator()(sycl::queue &q, std::int64_t m, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  std::int64_t *ipiv, void *device_ws,
                  std::size_t device_ws_size, int *info) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    std::int64_t a_stride = m * lda;
    auto a_data = dpct::detail::get_memory<T>(a);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    oneapi::mkl::lapack::getrfnp_batch(q, m, n, a_data, lda, a_stride, 1,
                                       device_ws_data, device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
#endif
  }
};

template <typename T> struct gesvd_scratchpad_size_impl {
  void operator()(sycl::queue &q, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                  library_data_t a_type, std::int64_t lda,
                  library_data_t u_type, std::int64_t ldu,
                  library_data_t vt_type, std::int64_t ldvt,
                  std::size_t &device_ws_size) {
    device_ws_size = oneapi::mkl::lapack::gesvd_scratchpad_size<T>(
        q, jobu, jobvt, m, n, lda, ldu, ldvt);
  }
};

template <typename T> struct ElementType { using value_tpye = T; };
template <typename T> struct ElementType<std::complex<T>> {
  using value_tpye = T;
};
template <typename T> struct gesvd_impl {
  void operator()(sycl::queue &q, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  library_data_t s_type, void *s, library_data_t u_type,
                  void *u, std::int64_t ldu, library_data_t vt_type, void *vt,
                  std::int64_t ldvt, void *device_ws,
                  std::size_t device_ws_size, int *info) {
    auto a_data = dpct::detail::get_memory<T>(a);
    auto s_data =
        dpct::detail::get_memory<typename ElementType<T>::value_tpye>(s);
    auto u_data = dpct::detail::get_memory<T>(u);
    auto vt_data = dpct::detail::get_memory<T>(vt);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    oneapi::mkl::lapack::gesvd(q, jobu, jobvt, m, n, a_data, lda, s_data,
                               u_data, ldu, vt_data, ldvt, device_ws_data,
                               device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
  }
};
template <typename T> struct gesvd_conj_impl : public gesvd_impl<T> {
  void operator()(sycl::queue &q, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  library_data_t s_type, void *s, library_data_t u_type,
                  void *u, std::int64_t ldu, library_data_t vt_type, void *vt,
                  std::int64_t ldvt, void *device_ws,
                  std::size_t device_ws_size, int *info) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    using base = gesvd_impl<T>;
    base::operator()(q, jobu, jobvt, m, n, a_type, a, lda, s_type, s, u_type, u,
                     ldu, vt_type, vt, ldvt, device_ws, device_ws_size, info);
    auto vt_data = dpct::detail::get_memory<T>(vt);
    oneapi::mkl::blas::row_major::imatcopy(q, oneapi::mkl::transpose::conjtrans,
                                           n, n, T(1.0f), vt_data, ldvt, ldvt);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
#endif
  }
};

template <typename T> struct potrf_scratchpad_size_impl {
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                  library_data_t a_type, std::int64_t lda,
                  std::size_t &device_ws_size) {
    device_ws_size =
        oneapi::mkl::lapack::potrf_scratchpad_size<T>(q, uplo, n, lda);
  }
};

template <typename T> struct potrf_impl {
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                  library_data_t a_type, void *a, std::int64_t lda,
                  void *device_ws, std::size_t device_ws_size, int *info) {
    auto a_data = dpct::detail::get_memory<T>(a);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    oneapi::mkl::lapack::potrf(q, uplo, n, a_data, lda, device_ws_data,
                               device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
  }
};

template <typename T> struct potrs_impl {
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::int64_t nrhs, library_data_t a_type, void *a,
                  std::int64_t lda, library_data_t b_type, void *b,
                  std::int64_t ldb, int *info) {
    std::int64_t device_ws_size = oneapi::mkl::lapack::potrs_scratchpad_size<T>(
        q, uplo, n, nrhs, lda, ldb);
    working_memory<T> device_ws(device_ws_size, q);
    auto device_ws_data = device_ws.get_memory();
    auto a_data = dpct::detail::get_memory<T>(a);
    auto b_data = dpct::detail::get_memory<T>(b);
    oneapi::mkl::lapack::potrs(q, uplo, n, nrhs, a_data, lda, b_data, ldb,
                               device_ws_data, device_ws_size);
    sycl::event e = ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
    device_ws.set_event(e);
  }
};

template <typename T> struct value_type_trait { using value_type = T; };
template <typename T> struct value_type_trait<std::complex<T>> {
  using value_type = T;
};

template <typename T> auto lamch_s() {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  if constexpr (std::is_same_v<T, float>) {
    return slamch("S");
  } else if constexpr (std::is_same_v<T, double>) {
    return dlamch("S");
  }
  throw std::runtime_error("the type is unsupported");
#endif
}

#define DISPATCH_FLOAT_FOR_SCRATCHPAD_SIZE(FUNC, ...)                          \
  do {                                                                         \
    if constexpr (std::is_floating_point_v<T>) {                               \
      device_ws_size = oneapi::mkl::lapack::sy##FUNC(__VA_ARGS__);             \
    } else {                                                                   \
      device_ws_size = oneapi::mkl::lapack::he##FUNC(__VA_ARGS__);             \
    }                                                                          \
  } while (0)

#define DISPATCH_FLOAT_FOR_CALCULATION(FUNC, ...)                              \
  do {                                                                         \
    if constexpr (std::is_floating_point_v<T>) {                               \
      oneapi::mkl::lapack::sy##FUNC(__VA_ARGS__);                              \
    } else {                                                                   \
      oneapi::mkl::lapack::he##FUNC(__VA_ARGS__);                              \
    }                                                                          \
  } while (0)

template <typename T> struct syheevx_scratchpad_size_impl {
  void operator()(sycl::queue &q, oneapi::mkl::compz jobz,
                  oneapi::mkl::rangev range, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::int64_t lda, void *vl, void *vu,
                  std::int64_t il, std::int64_t iu,
                  std::size_t &device_ws_size) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    using value_t = typename value_type_trait<T>::value_type;
    auto vl_value = *reinterpret_cast<value_t *>(vl);
    auto vu_value = *reinterpret_cast<value_t *>(vu);
    auto abstol = 2 * lamch_s<value_t>();
    DISPATCH_FLOAT_FOR_SCRATCHPAD_SIZE(evx_scratchpad_size<T>, q, jobz, range,
                                       uplo, n, lda, vl_value, vu_value, il, iu,
                                       abstol, lda);
#endif
  }
};

template <typename T> constexpr library_data_t get_library_data_t_from_type() {
  if constexpr (std::is_same_v<T, float>) {
    return library_data_t::real_float;
  } else if constexpr (std::is_same_v<T, double>) {
    return library_data_t::real_double;
  } else if constexpr (std::is_same_v<T, sycl::float2> ||
                       std::is_same_v<T, std::complex<float>>) {
    return library_data_t::complex_float;
  } else if constexpr (std::is_same_v<T, sycl::double2> ||
                       std::is_same_v<T, std::complex<double>>) {
    return library_data_t::complex_double;
  }
  throw std::runtime_error("the type is unsupported");
}

template <typename T> struct syheevx_impl {
  void operator()(sycl::queue &q, oneapi::mkl::compz jobz,
                  oneapi::mkl::rangev range, oneapi::mkl::uplo uplo,
                  std::int64_t n, library_data_t a_type, void *a,
                  std::int64_t lda, void *vl, void *vu, std::int64_t il,
                  std::int64_t iu, std::int64_t *m, library_data_t w_type,
                  void *w, void *device_ws, std::size_t device_ws_size,
                  int *info) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    using value_t = typename value_type_trait<T>::value_type;
    working_memory<T> z(n * lda, q);
    working_memory<std::int64_t> m_device(1, q);
    auto z_data = z.get_memory();
    auto m_device_data = m_device.get_memory();
    auto a_data = dpct::detail::get_memory<T>(a);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    auto vl_value = *reinterpret_cast<value_t *>(vl);
    auto vu_value = *reinterpret_cast<value_t *>(vu);
    auto w_data = dpct::detail::get_memory<value_t>(w);
    auto abstol = 2 * lamch_s<value_t>();
    DISPATCH_FLOAT_FOR_CALCULATION(evx, q, jobz, range, uplo, n, a_data, lda,
                                   vl_value, vu_value, il, iu, abstol,
                                   m_device_data, w_data, z_data, lda,
                                   device_ws_data, device_ws_size);
    ::dpct::cs::memcpy(q, a, z.get_ptr(), n * lda * sizeof(T),
                       ::dpct::cs::memcpy_direction::device_to_device);
    ::dpct::cs::memcpy(q, m, m_device.get_ptr(), sizeof(std::int64_t),
                       ::dpct::cs::memcpy_direction::device_to_host);
    sycl::event e = ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
    z.set_event(e);
    m_device.set_event(e);
#endif
  }
};

template <typename T> struct syhegvx_scratchpad_size_impl {
  void operator()(sycl::queue &q, std::int64_t itype, oneapi::mkl::compz jobz,
                  oneapi::mkl::rangev range, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::int64_t lda, std::int64_t ldb, void *vl,
                  void *vu, std::int64_t il, std::int64_t iu,
                  std::size_t &device_ws_size) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    using value_t = typename value_type_trait<T>::value_type;
    auto vl_value = *reinterpret_cast<value_t *>(vl);
    auto vu_value = *reinterpret_cast<value_t *>(vu);
    auto abstol = 2 * lamch_s<value_t>();
    DISPATCH_FLOAT_FOR_SCRATCHPAD_SIZE(gvx_scratchpad_size<T>, q, itype, jobz,
                                       range, uplo, n, lda, ldb, vl_value,
                                       vu_value, il, iu, abstol, lda);
#endif
  }
};

template <typename T> struct syhegvx_impl {
  void operator()(sycl::queue &q, std::int64_t itype, oneapi::mkl::compz jobz,
                  oneapi::mkl::rangev range, oneapi::mkl::uplo uplo,
                  std::int64_t n, void *a, std::int64_t lda, void *b,
                  std::int64_t ldb, void *vl, void *vu, std::int64_t il,
                  std::int64_t iu, std::int64_t *m, void *w, void *device_ws,
                  std::size_t device_ws_size, int *info) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    using value_t = typename value_type_trait<T>::value_type;
    working_memory<T> z(n * lda, q);
    working_memory<std::int64_t> m_device(1, q);
    auto z_data = z.get_memory();
    auto m_device_data = m_device.get_memory();
    auto a_data = dpct::detail::get_memory<T>(a);
    auto b_data = dpct::detail::get_memory<T>(b);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    auto vl_value = *reinterpret_cast<value_t *>(vl);
    auto vu_value = *reinterpret_cast<value_t *>(vu);
    auto w_data = dpct::detail::get_memory<value_t>(w);
    auto abstol = 2 * lamch_s<value_t>();
    DISPATCH_FLOAT_FOR_CALCULATION(gvx, q, itype, jobz, range, uplo, n, a_data,
                                   lda, b_data, ldb, vl_value, vu_value, il, iu,
                                   abstol, m_device_data, w_data, z_data, lda,
                                   device_ws_data, device_ws_size);
    ::dpct::cs::memcpy(q, a, z.get_ptr(), n * lda * sizeof(T),
                       ::dpct::cs::memcpy_direction::device_to_device);
    ::dpct::cs::memcpy(q, m, m_device.get_ptr(), sizeof(std::int64_t),
                       ::dpct::cs::memcpy_direction::device_to_host);
    sycl::event e = ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
    z.set_event(e);
    m_device.set_event(e);
#endif
  }
};

template <typename T> struct syhegvd_scratchpad_size_impl {
  void operator()(sycl::queue &q, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                  std::int64_t ldb, std::size_t &device_ws_size) {
    DISPATCH_FLOAT_FOR_SCRATCHPAD_SIZE(gvd_scratchpad_size<T>, q, itype, jobz,
                                       uplo, n, lda, ldb);
  }
};

template <typename T> struct syhegvd_impl {
  void operator()(sycl::queue &q, std::int64_t itype, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, void *a,
                  std::int64_t lda, void *b, std::int64_t ldb, void *w,
                  void *device_ws, std::size_t device_ws_size, int *info) {
    using value_t = typename value_type_trait<T>::value_type;
    auto a_data = dpct::detail::get_memory<T>(a);
    auto b_data = dpct::detail::get_memory<T>(b);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    auto w_data = dpct::detail::get_memory<value_t>(w);
    DISPATCH_FLOAT_FOR_CALCULATION(gvd, q, itype, jobz, uplo, n, a_data, lda,
                                   b_data, ldb, w_data, device_ws_data,
                                   device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
  }
};

oneapi::mkl::compz job2compz(const oneapi::mkl::job &job) {
  oneapi::mkl::compz ret;
  if (job == oneapi::mkl::job::novec) {
    ret = oneapi::mkl::compz::novectors;
  } else if (job == oneapi::mkl::job::vec) {
    ret = oneapi::mkl::compz::vectors;
  } else {
    throw std::runtime_error("the job type is unsupported");
  }
  return ret;
}

template <typename T> struct syheev_scratchpad_size_impl {
  void operator()(sycl::queue &q, oneapi::mkl::compz jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                  std::size_t &device_ws_size) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    DISPATCH_FLOAT_FOR_SCRATCHPAD_SIZE(ev_scratchpad_size<T>, q, jobz, uplo, n,
                                       lda);
#endif
  }
};

template <typename T> struct syheev_impl {
  void operator()(sycl::queue &q, oneapi::mkl::compz jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, void *a,
                  std::int64_t lda, void *w, void *device_ws,
                  std::size_t device_ws_size, int *info) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    using value_t = typename value_type_trait<T>::value_type;
    auto a_data = dpct::detail::get_memory<T>(a);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    auto w_data = dpct::detail::get_memory<value_t>(w);
    DISPATCH_FLOAT_FOR_CALCULATION(ev, q, jobz, uplo, n, a_data, lda, w_data,
                                   device_ws_data, device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
#endif
  }
};

template <typename T> struct syheevd_scratchpad_size_impl {
  void operator()(sycl::queue &q, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                  std::int64_t n, library_data_t a_type, std::int64_t lda,
                  std::size_t &device_ws_size) {
    DISPATCH_FLOAT_FOR_SCRATCHPAD_SIZE(evd_scratchpad_size<T>, q, jobz, uplo, n,
                                       lda);
  }
};

template <typename T> struct syheevd_impl {
  void operator()(sycl::queue &q, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                  std::int64_t n, library_data_t a_type, void *a,
                  std::int64_t lda, void *w, void *device_ws,
                  std::size_t device_ws_size, int *info) {
    using value_t = typename value_type_trait<T>::value_type;
    auto a_data = dpct::detail::get_memory<T>(a);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    auto w_data = dpct::detail::get_memory<value_t>(w);
    DISPATCH_FLOAT_FOR_CALCULATION(evd, q, jobz, uplo, n, a_data, lda, w_data,
                                   device_ws_data, device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
  }
};

#undef DISPATCH_FLOAT_FOR_SCRATCHPAD_SIZE
#undef DISPATCH_FLOAT_FOR_CALCULATION

template <typename T> struct trtri_scratchpad_size_impl {
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo,
                  oneapi::mkl::diag diag, std::int64_t n, library_data_t a_type,
                  std::int64_t lda, std::size_t &device_ws_size) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    device_ws_size =
        oneapi::mkl::lapack::trtri_scratchpad_size<T>(q, uplo, diag, n, lda);
#endif
  }
};

template <typename T> struct trtri_impl {
  void operator()(sycl::queue &q, oneapi::mkl::uplo uplo,
                  oneapi::mkl::diag diag, std::int64_t n, library_data_t a_type,
                  void *a, std::int64_t lda, void *device_ws,
                  std::size_t device_ws_size, int *info) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    auto a_data = dpct::detail::get_memory<T>(a);
    auto device_ws_data = dpct::detail::get_memory<T>(device_ws);
    oneapi::mkl::lapack::trtri(q, uplo, diag, n, a_data, lda, device_ws_data,
                               device_ws_size);
    ::dpct::cs::fill<unsigned char>(q, info, 0, sizeof(int));
#endif
  }
};
} // namespace dpct::lapack::detail

#endif // __DPCT_LAPACK_UTILS_DETAIL_HPP__
