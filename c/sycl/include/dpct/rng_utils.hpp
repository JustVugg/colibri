//==---- rng_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_RNG_UTILS_HPP__
#define __DPCT_RNG_UTILS_HPP__

#include "compat_service.hpp"
#include "lib_common_utils.hpp"

#if defined(__has_include) && __has_include(<oneapi/math/rng/device.hpp>)
#include <oneapi/math/rng/device.hpp>
#elif defined(__has_include) && __has_include(<oneapi/mkl/rng/device.hpp>)
#include <oneapi/mkl/rng/device.hpp>
#elif defined(__has_include)
#error "SYCLomatic runtime requires oneMath/oneMKL support"
#else
#error "SYCLomatic runtime requires __has_include support"
#endif

namespace dpct::rng {
enum class random_mode {
  best,
  legacy,
  optimal,
};

enum class random_engine_type {
  philox4x32x10,
  mrg32k3a,
  mt2203,
  mt19937,
  sobol,
  mcg59
};
} // namespace dpct::rng

#include "detail/rng_utils_detail.hpp"

namespace dpct::rng {
namespace device {
/// The random number generator on device.
/// \tparam engine_t The device random number generator engine. It can only be
/// oneapi::mkl::rng::device::mrg32k3a<1> or
/// oneapi::mkl::rng::device::mrg32k3a<4> or
/// oneapi::mkl::rng::device::philox4x32x10<1> or
/// oneapi::mkl::rng::device::philox4x32x10<4> or "
/// oneapi::mkl::rng::device::mcg59<1>.
template <typename engine_t> class rng_generator {
  static_assert(
      std::disjunction_v<
          std::is_same<engine_t, oneapi::mkl::rng::device::mrg32k3a<1>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::mrg32k3a<4>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::philox4x32x10<1>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::philox4x32x10<4>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::mcg59<1>>>,
      "engine_t can only be oneapi::mkl::rng::device::mrg32k3a<1> or "
      "oneapi::mkl::rng::device::mrg32k3a<4> or "
      "oneapi::mkl::rng::device::philox4x32x10<1> or "
      "oneapi::mkl::rng::device::philox4x32x10<4> or "
      "oneapi::mkl::rng::device::mcg59<1>.");
  static constexpr bool _is_engine_vec_size_one = std::disjunction_v<
      std::is_same<engine_t, oneapi::mkl::rng::device::mrg32k3a<1>>,
      std::is_same<engine_t, oneapi::mkl::rng::device::philox4x32x10<1>>,
      std::is_same<engine_t, oneapi::mkl::rng::device::mcg59<1>>>;
  static constexpr std::uint64_t default_seed = 0;
  oneapi::mkl::rng::device::bits<std::uint32_t> _distr_bits;
  oneapi::mkl::rng::device::uniform_bits<std::uint32_t> _distr_uniform_bits;
  oneapi::mkl::rng::device::gaussian<float> _distr_gaussian_float;
  oneapi::mkl::rng::device::gaussian<double> _distr_gaussian_double;
  oneapi::mkl::rng::device::lognormal<float> _distr_lognormal_float;
  oneapi::mkl::rng::device::lognormal<double> _distr_lognormal_double;
  oneapi::mkl::rng::device::poisson<std::uint32_t> _distr_poisson;
  oneapi::mkl::rng::device::uniform<float> _distr_uniform_float;
  oneapi::mkl::rng::device::uniform<double> _distr_uniform_double;
  engine_t _engine;

public:
  /// Default constructor of rng_generator
  rng_generator() { _engine = engine_t(default_seed); }
  /// Constructor of rng_generator if engine type is not mcg59
  /// \param [in] seed The seed to initialize the engine state.
  /// \param [in] num_to_skip Set the number of elements need to be skipped.
  /// The number is calculated as: num_to_skip[0] + num_to_skip[1] * 2^64 +
  /// num_to_skip[2] * 2^128 + ... + num_to_skip[n-1] * 2^(64*(n-1))
  template <typename T = engine_t,
            typename std::enable_if<!std::is_same_v<
                T, oneapi::mkl::rng::device::mcg59<1>>>::type * = nullptr>
  rng_generator(std::uint64_t seed,
                std::initializer_list<std::uint64_t> num_to_skip) {
    _engine = engine_t(seed, num_to_skip);
  }
  /// Constructor of rng_generator if engine type is mcg59
  /// \param [in] seed The seed to initialize the engine state.
  /// \param [in] num_to_skip Set the number of elements need to be skipped.
  template <typename T = engine_t,
            typename std::enable_if<std::is_same_v<
                T, oneapi::mkl::rng::device::mcg59<1>>>::type * = nullptr>
  rng_generator(std::uint64_t seed, std::uint64_t num_to_skip) {
    _engine = engine_t(seed, num_to_skip);
  }

  /// Generate random number(s) obeys distribution \tparam distr_t.
  /// \tparam T The distribution of the random number. It can only be
  /// oneapi::mkl::rng::device::bits<std::uint32_t>,
  /// oneapi::mkl::rng::device::uniform_bits<std::uint32_t>,
  /// oneapi::mkl::rng::device::gaussian<float>,
  /// oneapi::mkl::rng::device::gaussian<double>,
  /// oneapi::mkl::rng::device::lognormal<float>,
  /// oneapi::mkl::rng::device::lognormal<double>,
  /// oneapi::mkl::rng::device::poisson<std::uint32_t>,
  /// oneapi::mkl::rng::device::uniform<float> or
  /// oneapi::mkl::rng::device::uniform<double>
  /// \tparam vec_size The length of the return vector. It can only be 1, 2
  /// or 4.
  /// \param distr_params The parameter(s) for lognormal or poisson
  /// distribution.
  /// \return The vector of the random number(s).
  template <typename distr_t, int vec_size, class... distr_params_t>
  auto generate(distr_params_t... distr_params) {
    static_assert(vec_size == 1 || vec_size == 2 || vec_size == 4,
                  "vec_size is not supported.");
    static_assert(
        std::disjunction_v<
            std::is_same<distr_t,
                         oneapi::mkl::rng::device::bits<std::uint32_t>>,
            std::is_same<distr_t,
                         oneapi::mkl::rng::device::uniform_bits<std::uint32_t>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::gaussian<float>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::gaussian<double>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::lognormal<float>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::lognormal<double>>,
            std::is_same<distr_t,
                         oneapi::mkl::rng::device::poisson<std::uint32_t>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::uniform<float>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::uniform<double>>>,
        "distribution is not supported.");
#ifndef __INTEL_MKL__
    static_assert(
        vec_size == 4 || _is_engine_vec_size_one,
        "When using the oneMKL Interfaces Project, this function only support "
        "vec_size == 4 or _is_engine_vec_size_one is true.");
#endif

    if constexpr (std::is_same_v<
                      distr_t, oneapi::mkl::rng::device::bits<std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_bits);
    }
    if constexpr (std::is_same_v<
                      distr_t,
                      oneapi::mkl::rng::device::uniform_bits<std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_uniform_bits);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::gaussian<float>>) {
      return generate_vec<vec_size>(_distr_gaussian_float);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::gaussian<double>>) {
      return generate_vec<vec_size>(_distr_gaussian_double);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::lognormal<float>>) {
      return generate_vec<vec_size>(_distr_lognormal_float, distr_params...,
                                    0.0f, 1.0f);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::lognormal<double>>) {
      return generate_vec<vec_size>(_distr_lognormal_double, distr_params...,
                                    0.0, 1.0);
    }
    if constexpr (std::is_same_v<distr_t, oneapi::mkl::rng::device::poisson<
                                              std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_poisson, distr_params...);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::uniform<float>>) {
      return generate_vec<vec_size>(_distr_uniform_float);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::uniform<double>>) {
      return generate_vec<vec_size>(_distr_uniform_double);
    }
  }

  /// Get the random number generator engine.
  /// \return The reference of the internal random number generator engine.
  engine_t &get_engine() { return _engine; }

private:
  template <typename distr_t> auto generate_single(distr_t &distr) {
    if constexpr (_is_engine_vec_size_one) {
      return oneapi::mkl::rng::device::generate(distr, _engine);
    }
#ifdef __INTEL_MKL__
    else {
      return oneapi::mkl::rng::device::generate_single(distr, _engine);
    }
#endif
  }

  template <int vec_size, typename distr_t, class... distr_params_t>
  auto generate_vec(distr_t &distr, distr_params_t... distr_params) {
    if constexpr (sizeof...(distr_params_t)) {
      typename distr_t::param_type pt(distr_params...);
      distr.param(pt);
    }
    if constexpr (vec_size == 4) {
      if constexpr (_is_engine_vec_size_one) {
        sycl::vec<typename distr_t::result_type, 4> res;
        res.x() = oneapi::mkl::rng::device::generate(distr, _engine);
        res.y() = oneapi::mkl::rng::device::generate(distr, _engine);
        res.z() = oneapi::mkl::rng::device::generate(distr, _engine);
        res.w() = oneapi::mkl::rng::device::generate(distr, _engine);
        return res;
      } else {
        return oneapi::mkl::rng::device::generate(distr, _engine);
      }
    } else if constexpr (vec_size == 1) {
      return generate_single(distr);
    } else if constexpr (vec_size == 2) {
      sycl::vec<typename distr_t::result_type, 2> res;
      res.x() = generate_single(distr);
      res.y() = generate_single(distr);
      return res;
    }
  }
};
} // namespace device

typedef std::shared_ptr<host::detail::rng_generator_base> host_rng_ptr;

/// Create a host random number generator.
/// \tparam work_on_cpu Whether the work is offloaded to CPU.
/// \param type The random engine type.
/// \param q The queue where the generator should be executed.
/// \return The pointer of random number generator.
inline host_rng_ptr
create_host_rng(const random_engine_type type,
                sycl::queue &q = ::dpct::cs::get_default_queue()) {
  switch (type) {
  case random_engine_type::philox4x32x10:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::philox4x32x10>>(q);
  case random_engine_type::mrg32k3a:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mrg32k3a>>(q);
#ifndef __INTEL_MKL__
    throw std::runtime_error(host::detail::OneMKLNotSupport);
#else
  case random_engine_type::mt2203:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mt2203>>(q);
  case random_engine_type::mt19937:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mt19937>>(q);
  case random_engine_type::sobol:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::sobol>>(q);
  case random_engine_type::mcg59:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mcg59>>(q);
#endif
  }
}
} // namespace dpct::rng

template <class engine_t>
struct sycl::is_device_copyable<dpct::rng::device::rng_generator<engine_t>>
    : std::true_type {};

#endif // __DPCT_RNG_UTILS_HPP__
