//==---- rng_utils_detail.hpp --------------------------*- C++ -*-----------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_RNG_UTILS_DETAIL_HPP__
#define __DPCT_RNG_UTILS_DETAIL_HPP__

namespace dpct::rng::host::detail {
static const std::string OneMKLNotSupport =
    "The oneAPI Math Kernel Library (oneMKL) Interfaces Project does not "
    "support this API.";
class rng_generator_base {
public:
  /// Set the seed of host rng_generator.
  /// \param seed The engine seed.
  virtual void set_seed(const std::uint64_t seed) = 0;

  /// Set the dimensions of host rng_generator.
  /// \param dimensions The engine dimensions.
  virtual void set_dimensions(const std::uint32_t dimensions) = 0;

  /// Set the queue of host rng_generator.
  /// \param queue The engine queue.
  virtual void set_queue(sycl::queue *queue) = 0;

  /// Set the mode of host rng_generator.
  /// \param mode The engine mode.
  virtual void set_mode(const random_mode mode) = 0;

  /// Generate unsigned int random number(s) with 'uniform_bits' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform_bits(unsigned int *output,
                                            std::int64_t n) = 0;

  /// Generate unsigned long long random number(s) with 'uniform_bits'
  /// distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform_bits(unsigned long long *output,
                                            std::int64_t n) = 0;

  /// Generate float random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  virtual inline void generate_lognormal(float *output, std::int64_t n, float m,
                                         float s) = 0;

  /// Generate double random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  virtual inline void generate_lognormal(double *output, std::int64_t n,
                                         double m, double s) = 0;

  /// Generate float random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  virtual inline void generate_gaussian(float *output, std::int64_t n,
                                        float mean, float stddev) = 0;

  /// Generate double random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  virtual inline void generate_gaussian(double *output, std::int64_t n,
                                        double mean, double stddev) = 0;

  /// Generate unsigned int random number(s) with 'poisson' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param lambda Lambda for the Poisson distribution.
  virtual inline void generate_poisson(unsigned int *output, std::int64_t n,
                                       double lambda) = 0;

  /// Generate float random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform(float *output, std::int64_t n) = 0;

  /// Generate double random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform(double *output, std::int64_t n) = 0;

  /// Skip ahead several random number(s).
  /// \param num_to_skip The number of random numbers to be skipped.
  virtual void skip_ahead(const std::uint64_t num_to_skip) = 0;

  /// Set the direction numbers of host rng_generator. Only Sobol engine
  /// supports this method.
  /// \param direction_numbers The engine direction numbers.
  virtual void set_direction_numbers(
      const std::vector<std::uint32_t> &direction_numbers) = 0;

  /// Set the engine index of host rng_generator. Only MT2203 engine
  /// supports this method.
  /// \param engine_idx The engine index.
  virtual void set_engine_idx(std::uint32_t engine_idx) = 0;

protected:
  /// Construct the host rng_generator.
  /// \param queue The queue where the generator should be executed.
  rng_generator_base(sycl::queue *queue) : _queue(queue) {}

  sycl::queue *_queue = nullptr;
  std::uint64_t _seed{0};
  std::uint32_t _dimensions{1};
  random_mode _mode{random_mode::best};
  std::vector<std::uint32_t> _direction_numbers;
  std::uint32_t _engine_idx{0};
};

/// The random number generator on host.
template <typename engine_t = oneapi::mkl::rng::philox4x32x10>
class rng_generator : public rng_generator_base {
public:
  /// Constructor of rng_generator.
  /// \param q The queue where the generator should be executed.
  rng_generator(sycl::queue &q = ::dpct::cs::get_default_queue())
      : rng_generator_base(&q),
        _engine(create_engine(&q, _seed, _dimensions, _mode)) {}

  /// Set the seed of host rng_generator.
  /// \param seed The engine seed.
  void set_seed(const std::uint64_t seed) {
    if (seed == _seed)
      return;
    _seed = seed;
    _engine = create_engine(_queue, _seed, _dimensions, _mode);
  }

  /// Set the dimensions of host rng_generator.
  /// \param dimensions The engine dimensions.
  void set_dimensions(const std::uint32_t dimensions) {
    if (dimensions == _dimensions)
      return;
    _dimensions = dimensions;
    _engine = create_engine(_queue, _seed, _dimensions, _mode);
  }

  /// Set the queue of host rng_generator.
  /// \param queue The engine queue.
  void set_queue(sycl::queue *queue) {
    if (queue == _queue)
      return;
    _queue = queue;
    _engine = create_engine(_queue, _seed, _dimensions, _mode);
  }

  /// Set the mode of host rng_generator.
  /// \param mode The engine mode.
  void set_mode(const random_mode mode) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(OneMKLNotSupport);
#else
    if constexpr (!std::is_same_v<engine_t, oneapi::mkl::rng::mrg32k3a>) {
      throw std::runtime_error("Only mrg32k3a engine support this method.");
    }
    if (mode == _mode)
      return;
    _mode = mode;
    _engine = create_engine(_queue, _seed, _dimensions, _mode);
#endif
  }

  /// Set the direction numbers of Sobol host rng_generator.
  /// \param direction_numbers The user-defined direction numbers.
  void
  set_direction_numbers(const std::vector<std::uint32_t> &direction_numbers) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(OneMKLNotSupport);
#else
    if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::sobol>) {
      if (direction_numbers == _direction_numbers)
        return;
      _direction_numbers = direction_numbers;
      _engine =
          create_engine(_queue, _seed, _dimensions, _mode, _direction_numbers);
    } else {
      throw std::runtime_error("Only Sobol engine supports this method.");
    }
#endif
  }

  /// Set the engine index of MT2203 host rng_generator.
  /// \param engine_idx The user-defined engine index.
  void set_engine_idx(std::uint32_t engine_idx) {
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
    if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::mt2203>) {
      if (engine_idx == _engine_idx)
        return;
      _engine_idx = engine_idx;
      _engine = create_engine(_queue, _seed, _dimensions, _mode, std::nullopt,
                              _engine_idx);
    } else {
      throw std::runtime_error("Only MT2203 engine supports this method.");
    }
#endif
  }

  /// Generate unsigned int random number(s) with 'uniform_bits' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform_bits(unsigned int *output, std::int64_t n) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(OneMKLNotSupport);
#else
    static_assert(sizeof(unsigned int) == sizeof(std::uint32_t));
    generate<oneapi::mkl::rng::uniform_bits<std::uint32_t>>(
        (std::uint32_t *)output, n);
#endif
  }

  /// Generate unsigned long long random number(s) with 'uniform_bits'
  /// distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform_bits(unsigned long long *output,
                                    std::int64_t n) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(OneMKLNotSupport);
#else
    static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));
    generate<oneapi::mkl::rng::uniform_bits<std::uint64_t>>(
        (std::uint64_t *)output, n);
#endif
  }

  /// Generate float random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  inline void generate_lognormal(float *output, std::int64_t n, float m,
                                 float s) {
    generate<oneapi::mkl::rng::lognormal<float>>(output, n, m, s);
  }

  /// Generate double random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  inline void generate_lognormal(double *output, std::int64_t n, double m,
                                 double s) {
    generate<oneapi::mkl::rng::lognormal<double>>(output, n, m, s);
  }

  /// Generate float random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  inline void generate_gaussian(float *output, std::int64_t n, float mean,
                                float stddev) {
    generate<oneapi::mkl::rng::gaussian<float>>(output, n, mean, stddev);
  }

  /// Generate double random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  inline void generate_gaussian(double *output, std::int64_t n, double mean,
                                double stddev) {
    generate<oneapi::mkl::rng::gaussian<double>>(output, n, mean, stddev);
  }

  /// Generate unsigned int random number(s) with 'poisson' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param lambda Lambda for the Poisson distribution.
  inline void generate_poisson(unsigned int *output, std::int64_t n,
                               double lambda) {
    generate<oneapi::mkl::rng::poisson<unsigned int>>(output, n, lambda);
  }

  /// Generate float random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform(float *output, std::int64_t n) {
    generate<oneapi::mkl::rng::uniform<float>>(output, n);
  }

  /// Generate double random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform(double *output, std::int64_t n) {
    generate<oneapi::mkl::rng::uniform<double>>(output, n);
  }

  /// Skip ahead several random number(s).
  /// \param num_to_skip The number of random numbers to be skipped.
  void skip_ahead(const std::uint64_t num_to_skip) {
#ifndef __INTEL_MKL__
    oneapi::mkl::rng::skip_ahead(_engine, num_to_skip);
#else
    if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::mt2203>)
      throw std::runtime_error("no skip_ahead method of mt2203 engine.");
    else
      oneapi::mkl::rng::skip_ahead(_engine, num_to_skip);
#endif
  }

private:
  static inline engine_t
  create_engine(sycl::queue *queue, const std::uint64_t seed,
                const std::uint32_t dimensions, const random_mode mode,
                std::optional<std::vector<std::uint32_t>> direction_numbers =
                    std::nullopt,
                std::optional<std::uint32_t> engine_idx = std::nullopt) {
#ifdef __INTEL_MKL__
    if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::mrg32k3a>) {
      // oneapi::mkl::rng::mrg32k3a_mode is only supported for GPU device. For
      // other devices, this argument will be ignored.
      if (queue->get_device().is_gpu()) {
#if __cplusplus < 202002L
        // before C++20
        switch (mode) {
        case random_mode::best:
          return engine_t(*queue, seed,
                          oneapi::mkl::rng::mrg32k3a_mode::custom{81920});
        case random_mode::legacy:
          return engine_t(*queue, seed,
                          oneapi::mkl::rng::mrg32k3a_mode::custom{4096});
        case random_mode::optimal:
          return engine_t(*queue, seed,
                          oneapi::mkl::rng::mrg32k3a_mode::optimal_v);
        }
#else
        // since C++20
#if defined(_MSC_VER) && !defined(__clang__)
#pragma message("warning: oneapi::mkl::rng::mrg32k3a_mode::custom cannot be "  \
                "initialized with a value since C++20. Ignore the parameter "  \
                "`mode`.")
#else
#warning "oneapi::mkl::rng::mrg32k3a_mode::custom cannot be initialized with " \
         "a value since C++20. Ignore the parameter `mode`."
#endif
        return engine_t(*queue, dimensions);
#endif
      }
    } else if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::mt2203>) {
      if (engine_idx.has_value()) {
        return engine_t(*queue, seed, engine_idx.value());
      }
    } else if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::sobol>) {
      if (direction_numbers.has_value()) {
        return engine_t(*queue, direction_numbers.value());
      }
      return engine_t(*queue, dimensions);
    }
#endif
    return engine_t(*queue, seed);
  }

  template <typename distr_t, typename buffer_t, class... distr_params_t>
  void generate(buffer_t *output, const std::int64_t n,
                const distr_params_t... distr_params) {
    auto output_buf = ::dpct::detail::get_memory<buffer_t>(output);
    oneapi::mkl::rng::generate(distr_t(distr_params...), _engine, n,
                               output_buf);
  }
  engine_t _engine{};
};
} // namespace dpct::rng::host::detail

#endif // __DPCT_RNG_UTILS_DETAIL_HPP__
