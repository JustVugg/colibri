//==---- sparse_utils_detail.hpp --------------------------*- C++ -*--------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_SPARSE_UTILS_DETAIL_HPP__
#define __DPCT_SPARSE_UTILS_DETAIL_HPP__

namespace dpct::sparse {
namespace detail {
using csrgemm_args_info =
    std::tuple<oneapi::mkl::transpose, oneapi::mkl::transpose, int, int, int,
               const std::shared_ptr<matrix_info>, const void *, const int *,
               const int *, const std::shared_ptr<matrix_info>, const void *,
               const int *, const int *, const std::shared_ptr<matrix_info>,
               const int *>;
struct csrgemm_args_info_hash {
  size_t operator()(const csrgemm_args_info &args) const {
    std::stringstream ss;
    ss << (char)std::get<0>(args) << ":";
    ss << (char)std::get<1>(args) << ":";
    ss << std::get<2>(args) << ":";
    ss << std::get<3>(args) << ":";
    ss << std::get<4>(args) << ":";
    ss << std::get<5>(args).get() << ":";
    ss << std::get<6>(args) << ":";
    ss << std::get<7>(args) << ":";
    ss << std::get<8>(args) << ":";
    ss << std::get<9>(args).get() << ":";
    ss << std::get<10>(args) << ":";
    ss << std::get<11>(args) << ":";
    ss << std::get<12>(args) << ":";
    ss << std::get<13>(args).get() << ":";
    ss << std::get<14>(args) << ":";
    return std::hash<std::string>{}(ss.str());
  }
};

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
template <typename handle_t> class handle_manager {
public:
  handle_manager() {}
  handle_manager(const handle_manager &other) = delete;
  handle_manager operator=(handle_manager other) = delete;
  ~handle_manager() {
    if (!_q || !_h)
      return;
    release();
  }
  void init(sycl::queue *q) {
    _q = q;
    _h = new handle_t;
    _init_func(*_q, _h);
  }
  sycl::event release() {
    if (!_q || !_h)
      return sycl::event();
    sycl::event e = _rel_func(*_q, _h, _deps);
    sycl::event ret = _q->submit([&](sycl::handler &cgh) {
      cgh.depends_on(e);
      cgh.host_task([_hh = _h] { delete _hh; });
    });
    _h = nullptr;
    _q = nullptr;
    return ret;
  }
  handle_t &get_handle() { return *_h; }
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  bool is_empty() { return !_h; }

protected:
  sycl::queue *_q = nullptr;

private:
  using init_func_t = std::function<void(sycl::queue &, handle_t *)>;
  using rel_func_t = std::function<sycl::event(
      sycl::queue &, handle_t *, const std::vector<sycl::event> &dependencies)>;
  handle_t *_h = nullptr;
  inline static init_func_t _init_func = nullptr;
  inline static rel_func_t _rel_func = nullptr;
  std::vector<sycl::event> _deps;
};
template <>
inline handle_manager<oneapi::mkl::sparse::matrix_handle_t>::init_func_t
    handle_manager<oneapi::mkl::sparse::matrix_handle_t>::_init_func =
        [](sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t *p_desc) {
          oneapi::mkl::sparse::init_matrix_handle(p_desc);
        };
template <>
inline handle_manager<oneapi::mkl::sparse::matrix_handle_t>::rel_func_t
    handle_manager<oneapi::mkl::sparse::matrix_handle_t>::_rel_func =
        oneapi::mkl::sparse::release_matrix_handle;

template <>
inline handle_manager<oneapi::mkl::sparse::omatadd_descr_t>::init_func_t
    handle_manager<oneapi::mkl::sparse::omatadd_descr_t>::_init_func =
        oneapi::mkl::sparse::init_omatadd_descr;
template <>
inline handle_manager<oneapi::mkl::sparse::omatadd_descr_t>::rel_func_t
    handle_manager<oneapi::mkl::sparse::omatadd_descr_t>::_rel_func =
        [](sycl::queue &queue, oneapi::mkl::sparse::omatadd_descr_t *p_desc,
           const std::vector<sycl::event> &dependencies) -> sycl::event {
  return oneapi::mkl::sparse::release_omatadd_descr(queue, *p_desc,
                                                    dependencies);
};

template <>
inline handle_manager<oneapi::mkl::sparse::matmat_descr_t>::init_func_t
    handle_manager<oneapi::mkl::sparse::matmat_descr_t>::_init_func =
        [](sycl::queue &queue, oneapi::mkl::sparse::matmat_descr_t *p_desc) {
          oneapi::mkl::sparse::init_matmat_descr(p_desc);
        };
template <>
inline handle_manager<oneapi::mkl::sparse::matmat_descr_t>::rel_func_t
    handle_manager<oneapi::mkl::sparse::matmat_descr_t>::_rel_func =
        [](sycl::queue &queue, oneapi::mkl::sparse::matmat_descr_t *p_desc,
           const std::vector<sycl::event> &dependencies) -> sycl::event {
  return queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([=] { oneapi::mkl::sparse::release_matmat_descr(p_desc); });
  });
};
class matrix_handle_manager
    : public handle_manager<oneapi::mkl::sparse::matrix_handle_t> {
public:
  using handle_manager<oneapi::mkl::sparse::matrix_handle_t>::handle_manager;
  template <typename Ty>
  void set_matrix_data(const int rows, const int cols,
                       oneapi::mkl::index_base base, const void *row_ptr,
                       const void *col_ind, const void *val) {
#ifdef DPCT_USM_LEVEL_NONE
    _row_ptr_buf = dpct::detail::get_memory<int>((int *)row_ptr);
    _col_ind_buf = dpct::detail::get_memory<int>((int *)col_ind);
    _val_buf = dpct::detail::get_memory<Ty>((Ty *)val);
    oneapi::mkl::sparse::set_csr_data(*_q, get_handle(), rows, cols, base,
                                      _row_ptr_buf, _col_ind_buf,
                                      std::get<sycl::buffer<Ty>>(_val_buf));
#else
    oneapi::mkl::sparse::set_csr_data(*_q, get_handle(), rows, cols, base,
                                      (int *)row_ptr, (int *)col_ind,
                                      (Ty *)val);
#endif
  }

private:
#ifdef DPCT_USM_LEVEL_NONE
  static constexpr size_t _max_data_variable_size =
      (std::max)({sizeof(sycl::buffer<float>), sizeof(sycl::buffer<double>),
                  sizeof(sycl::buffer<std::complex<float>>),
                  sizeof(sycl::buffer<std::complex<double>>)});
  using value_buf_t =
      std::variant<std::array<std::byte, _max_data_variable_size>,
                   sycl::buffer<float>, sycl::buffer<double>,
                   sycl::buffer<std::complex<float>>,
                   sycl::buffer<std::complex<double>>>;
  sycl::buffer<int> _row_ptr_buf = sycl::buffer<int>(0);
  sycl::buffer<int> _col_ind_buf = sycl::buffer<int>(0);
  value_buf_t _val_buf;
#endif
};

#ifdef DPCT_USM_LEVEL_NONE
#define SPARSE_CALL(CALL, HANDLE) CALL;
#else
#define SPARSE_CALL(CALL, HANDLE)                                              \
  sycl::event e = CALL;                                                        \
  HANDLE->add_dependency(e);
#endif

template <typename T> struct optimize_csrsv_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose trans, int row_col,
                  const std::shared_ptr<matrix_info> info, const void *val,
                  const int *row_ptr, const int *col_ind,
                  std::shared_ptr<optimize_info> optimize_info) {
    using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
    auto temp_row_ptr = dpct::detail::get_memory<int>(row_ptr);
    auto temp_col_ind = dpct::detail::get_memory<int>(col_ind);
    auto temp_val = dpct::detail::get_memory<Ty>(val);
#ifdef DPCT_USM_LEVEL_NONE
    optimize_info->_row_ptr_buf = temp_row_ptr;
    optimize_info->_col_ind_buf = temp_col_ind;
    optimize_info->_val_buf = temp_val;
    auto &data_row_ptr = optimize_info->_row_ptr_buf;
    auto &data_col_ind = optimize_info->_col_ind_buf;
    auto &data_val = std::get<sycl::buffer<Ty>>(optimize_info->_val_buf);
#else
    auto data_row_ptr = temp_row_ptr;
    auto data_col_ind = temp_col_ind;
    auto data_val = temp_val;
#endif
    oneapi::mkl::sparse::set_csr_data(queue, optimize_info->get_matrix_handle(),
                                      row_col, row_col, info->get_index_base(),
                                      data_row_ptr, data_col_ind, data_val);
    if (info->get_matrix_type() != matrix_info::matrix_type::tr)
      throw std::runtime_error("dpct::sparse::optimize_csrsv_impl()(): "
                               "oneapi::mkl::sparse::optimize_trsv "
                               "only accept triangular matrix.");
    SPARSE_CALL(oneapi::mkl::sparse::optimize_trsv(
                    queue, info->get_uplo(), trans, info->get_diag(),
                    optimize_info->get_matrix_handle()),
                optimize_info);
  }
};
template <typename T> struct csrsv_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose trans, int row_col,
                  const void *alpha, const std::shared_ptr<matrix_info> info,
                  const void *val, const int *row_ptr, const int *col_ind,
                  std::shared_ptr<optimize_info> optimize_info, const void *x,
                  void *y) {
    using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
    auto alpha_value =
        dpct::detail::get_value(static_cast<const Ty *>(alpha), queue);
    auto data_x = dpct::detail::get_memory<Ty>(x);
    auto data_y = dpct::detail::get_memory<Ty>(y);
    SPARSE_CALL(oneapi::mkl::sparse::trsv(queue, info->get_uplo(), trans,
                                          info->get_diag(), alpha_value,
                                          optimize_info->get_matrix_handle(),
                                          data_x, data_y),
                optimize_info);
  }
};

template <typename T> struct optimize_csrsm_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, int row_col, int nrhs,
                  const std::shared_ptr<matrix_info> info, const void *val,
                  const int *row_ptr, const int *col_ind,
                  std::shared_ptr<optimize_info> optimize_info) {
    using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
    auto temp_row_ptr = dpct::detail::get_memory<int>(row_ptr);
    auto temp_col_ind = dpct::detail::get_memory<int>(col_ind);
    auto temp_val = dpct::detail::get_memory<Ty>(val);
#ifdef DPCT_USM_LEVEL_NONE
    optimize_info->_row_ptr_buf = temp_row_ptr;
    optimize_info->_col_ind_buf = temp_col_ind;
    optimize_info->_val_buf = temp_val;
    auto &data_row_ptr = optimize_info->_row_ptr_buf;
    auto &data_col_ind = optimize_info->_col_ind_buf;
    auto &data_val = std::get<sycl::buffer<Ty>>(optimize_info->_val_buf);
#else
    auto data_row_ptr = temp_row_ptr;
    auto data_col_ind = temp_col_ind;
    auto data_val = temp_val;
#endif
    oneapi::mkl::sparse::set_csr_data(queue, optimize_info->get_matrix_handle(),
                                      row_col, row_col, info->get_index_base(),
                                      data_row_ptr, data_col_ind, data_val);
    if (info->get_matrix_type() != matrix_info::matrix_type::tr)
      throw std::runtime_error("dpct::sparse::optimize_csrsv_impl()(): "
                               "oneapi::mkl::sparse::optimize_trsm "
                               "only accept triangular matrix.");
    SPARSE_CALL(oneapi::mkl::sparse::optimize_trsm(
                    queue,
                    transb == oneapi::mkl::transpose::nontrans
                        ? oneapi::mkl::layout::col_major
                        : oneapi::mkl::layout::row_major,
                    info->get_uplo(), transa, info->get_diag(),
                    optimize_info->get_matrix_handle(), nrhs),
                optimize_info);
  }
};
template <typename T> struct csrsm_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, int row_col, int nrhs,
                  const void *alpha, const std::shared_ptr<matrix_info> info,
                  const void *val, const int *row_ptr, const int *col_ind,
                  void *b, int ldb,
                  std::shared_ptr<optimize_info> optimize_info) {
    using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
    auto alpha_value =
        dpct::detail::get_value(static_cast<const Ty *>(alpha), queue);
    auto data_b = dpct::detail::get_memory<Ty>(b);

    int x_size =
        ldb * (transb == oneapi::mkl::transpose::nontrans ? nrhs : row_col);
    Ty *x = (Ty *)::dpct::cs::malloc(sizeof(Ty) * x_size, queue);

    auto data_x = dpct::detail::get_memory<Ty>(x);

    sycl::event e1;
#ifndef DPCT_USM_LEVEL_NONE
    e1 =
#endif
        oneapi::mkl::sparse::trsm(
            queue,
            transb == oneapi::mkl::transpose::nontrans
                ? oneapi::mkl::layout::col_major
                : oneapi::mkl::layout::row_major,
            transa, oneapi::mkl::transpose::nontrans, info->get_uplo(),
            info->get_diag(), alpha_value, optimize_info->get_matrix_handle(),
            data_b, nrhs, ldb, data_x, ldb);

    sycl::event e2 =
        ::dpct::cs::memcpy(queue, b, x, sizeof(Ty) * x_size,
                           ::dpct::cs::memcpy_direction::automatic, {e1});

    sycl::event e3 = ::dpct::cs::enqueue_free({x}, {e2}, queue);
    optimize_info->add_dependency(e3);
  }
};

template <typename T> struct spmv_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose trans,
                  const void *alpha, sparse_matrix_desc_t a,
                  std::shared_ptr<dense_vector_desc> x, const void *beta,
                  std::shared_ptr<dense_vector_desc> y) {
    auto alpha_value =
        dpct::detail::get_value(reinterpret_cast<const T *>(alpha), queue);
    auto beta_value =
        dpct::detail::get_value(reinterpret_cast<const T *>(beta), queue);
    auto data_x = dpct::detail::get_memory<T>(x->get_value());
    auto data_y = dpct::detail::get_memory<T>(y->get_value());
    if (a->get_diag().has_value() && a->get_uplo().has_value()) {
      oneapi::mkl::sparse::optimize_trmv(queue, a->get_uplo().value(), trans,
                                         a->get_diag().value(),
                                         a->get_matrix_handle());
      SPARSE_CALL(oneapi::mkl::sparse::trmv(queue, a->get_uplo().value(), trans,
                                            a->get_diag().value(), alpha_value,
                                            a->get_matrix_handle(), data_x,
                                            beta_value, data_y),
                  a);
    } else {
      oneapi::mkl::sparse::optimize_gemv(queue, trans, a->get_matrix_handle());
      SPARSE_CALL(oneapi::mkl::sparse::gemv(queue, trans, alpha_value,
                                            a->get_matrix_handle(), data_x,
                                            beta_value, data_y),
                  a);
    }
  }
};

template <typename T> struct spmm_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose trans_a,
                  oneapi::mkl::transpose trans_b, const void *alpha,
                  sparse_matrix_desc_t a, std::shared_ptr<dense_matrix_desc> b,
                  const void *beta, std::shared_ptr<dense_matrix_desc> c) {
    auto alpha_value =
        dpct::detail::get_value(reinterpret_cast<const T *>(alpha), queue);
    auto beta_value =
        dpct::detail::get_value(reinterpret_cast<const T *>(beta), queue);
    auto data_b = dpct::detail::get_memory<T>(b->get_value());
    auto data_c = dpct::detail::get_memory<T>(c->get_value());
    SPARSE_CALL(
        oneapi::mkl::sparse::gemm(queue, b->get_layout(), trans_a, trans_b,
                                  alpha_value, a->get_matrix_handle(), data_b,
                                  b->get_col_num(), b->get_leading_dim(),
                                  beta_value, data_c, c->get_leading_dim()),
        a);
  }
};

template <typename T> struct spsv_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::uplo uplo,
                  oneapi::mkl::diag diag, oneapi::mkl::transpose trans_a,
                  const void *alpha, sparse_matrix_desc_t a,
                  std::shared_ptr<dense_vector_desc> x,
                  std::shared_ptr<dense_vector_desc> y) {
    auto alpha_value =
        dpct::detail::get_value(reinterpret_cast<const T *>(alpha), queue);
    auto data_x = dpct::detail::get_memory<T>(x->get_value());
    auto data_y = dpct::detail::get_memory<T>(y->get_value());
    SPARSE_CALL(oneapi::mkl::sparse::trsv(queue, uplo, trans_a, diag,
                                          alpha_value, a->get_matrix_handle(),
                                          data_x, data_y),
                a);
  }
};

template <typename T> struct spsm_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose trans_a,
                  oneapi::mkl::transpose trans_b, oneapi::mkl::uplo uplo,
                  oneapi::mkl::diag diag, const void *alpha,
                  sparse_matrix_desc_t a, std::shared_ptr<dense_matrix_desc> b,
                  std::shared_ptr<dense_matrix_desc> c) {
    auto alpha_value =
        dpct::detail::get_value(reinterpret_cast<const T *>(alpha), queue);
    auto data_b = dpct::detail::get_memory<T>(b->get_value());
    auto data_c = dpct::detail::get_memory<T>(c->get_value());
    SPARSE_CALL(oneapi::mkl::sparse::trsm(
                    queue, b->get_layout(), trans_a, trans_b, uplo, diag,
                    alpha_value, a->get_matrix_handle(), data_b,
                    c->get_col_num(), b->get_leading_dim(), data_c,
                    c->get_leading_dim()),
                a);
  }
};
#undef SPARSE_CALL

template <typename T, bool is_host_memory, typename host_memory_t = void>
struct temp_memory {
  static_assert(!is_host_memory || !std::is_same_v<host_memory_t, void>,
                "host_memory_t cannot be void when the input parameter ptr "
                "points to host memory");
  temp_memory(sycl::queue queue, void *ptr)
      : _queue(queue)
#ifdef DPCT_USM_LEVEL_NONE
        ,
        _buffer(is_host_memory ? sycl::buffer<T, 1>(sycl::range<1>(1))
                               : sycl::buffer<T, 1>(dpct::get_buffer<T>(ptr)))
#endif
  {
    if constexpr (is_host_memory) {
      _original_host_ptr = static_cast<host_memory_t *>(ptr);
#ifdef DPCT_USM_LEVEL_NONE
      auto _buffer_acc = _buffer.get_host_access(sycl::write_only);
      _buffer_acc[0] = static_cast<T>(*_original_host_ptr);
#else
      _memory_ptr = sycl::malloc_host<T>(1, _queue);
      *_memory_ptr = static_cast<T>(*_original_host_ptr);
#endif
    } else {
#ifndef DPCT_USM_LEVEL_NONE
      _memory_ptr = static_cast<T *>(ptr);
#endif
    }
  }

  ~temp_memory() {
    if constexpr (is_host_memory) {
#ifdef DPCT_USM_LEVEL_NONE
      auto _buffer_acc = _buffer.get_host_access(sycl::read_only);
      *_original_host_ptr = static_cast<host_memory_t>(_buffer_acc[0]);
#else
      _queue.wait();
      *_original_host_ptr = *_memory_ptr;
      sycl::free(_memory_ptr, _queue);
#endif
    }
  }
  auto get_memory_ptr() {
#ifdef DPCT_USM_LEVEL_NONE
    return &_buffer;
#else
    return _memory_ptr;
#endif
  }

private:
  sycl::queue _queue;
  host_memory_t *_original_host_ptr = nullptr;
#ifdef DPCT_USM_LEVEL_NONE
  sycl::buffer<T, 1> _buffer;
#else
  T *_memory_ptr;
#endif
};

template <typename T> struct csr2csc_impl {
  void operator()(sycl::queue queue, int m, int n, int nnz,
                  const void *from_val, const int *from_row_ptr,
                  const int *from_col_ind, void *to_val, int *to_col_ptr,
                  int *to_row_ind, conversion_scope range,
                  oneapi::mkl::index_base base) {
    using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
    oneapi::mkl::sparse::matrix_handle_t from_handle = nullptr;
    oneapi::mkl::sparse::matrix_handle_t to_handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&from_handle);
    oneapi::mkl::sparse::init_matrix_handle(&to_handle);
    auto data_from_row_ptr = dpct::detail::get_memory<int>(from_row_ptr);
    auto data_from_col_ind = dpct::detail::get_memory<int>(from_col_ind);
    auto data_from_val = dpct::detail::get_memory<Ty>(from_val);
    auto data_to_col_ptr = dpct::detail::get_memory<int>(to_col_ptr);
    auto data_to_row_ind = dpct::detail::get_memory<int>(to_row_ind);
    void *new_to_value = to_val;
    if (range == conversion_scope::index) {
      new_to_value =
          ::dpct::cs::malloc(sizeof(Ty) * nnz, ::dpct::cs::get_default_queue());
    }
    auto data_to_val = dpct::detail::get_memory<Ty>(new_to_value);
    oneapi::mkl::sparse::set_csr_data(queue, from_handle, m, n, base,
                                      data_from_row_ptr, data_from_col_ind,
                                      data_from_val);
    oneapi::mkl::sparse::set_csr_data(queue, to_handle, n, m, base,
                                      data_to_col_ptr, data_to_row_ind,
                                      data_to_val);
    sycl::event e1 = oneapi::mkl::sparse::omatcopy(
        queue, oneapi::mkl::transpose::trans, from_handle, to_handle);
    sycl::event e2 =
        oneapi::mkl::sparse::release_matrix_handle(queue, &from_handle, {e1});
    sycl::event e3 =
        oneapi::mkl::sparse::release_matrix_handle(queue, &to_handle, {e1});
    if (range == conversion_scope::index) {
      ::dpct::cs::enqueue_free({new_to_value}, {e2, e3}, queue);
    }
  }
};
#endif

template <template <typename> typename functor_t, typename... args_t>
inline void spblas_shim(library_data_t type, args_t &&...args) {
  switch (type) {
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
    throw std::runtime_error("The data type is not supported.");
  }
}

template <typename T> struct csrmv_impl {
  void operator()(sycl::queue &queue, oneapi::mkl::transpose trans,
                  int num_rows, int num_cols, const void *alpha,
                  const std::shared_ptr<matrix_info> info, const void *val,
                  const int *row_ptr, const int *col_ind, const void *x,
                  const void *beta, void *y) {
#ifndef __INTEL_MKL__
    throw std::runtime_error(
        "The oneAPI Math Kernel Library (oneMKL) Interfaces "
        "Project does not support this API.");
#else
    using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
    auto alpha_value =
        dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
    auto beta_value =
        dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

    oneapi::mkl::sparse::matrix_handle_t *sparse_matrix_handle =
        new oneapi::mkl::sparse::matrix_handle_t;
    oneapi::mkl::sparse::init_matrix_handle(sparse_matrix_handle);
    auto data_row_ptr = dpct::detail::get_memory<int>(row_ptr);
    auto data_col_ind = dpct::detail::get_memory<int>(col_ind);
    auto data_val = dpct::detail::get_memory<Ty>(val);
    oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, num_rows,
                                      num_cols, info->get_index_base(),
                                      data_row_ptr, data_col_ind, data_val);

    auto data_x = dpct::detail::get_memory<Ty>(x);
    auto data_y = dpct::detail::get_memory<Ty>(y);
    switch (info->get_matrix_type()) {
    case matrix_info::matrix_type::ge: {
      oneapi::mkl::sparse::optimize_gemv(queue, trans, *sparse_matrix_handle);
      oneapi::mkl::sparse::gemv(queue, trans, alpha_value,
                                *sparse_matrix_handle, data_x, beta_value,
                                data_y);
      break;
    }
    case matrix_info::matrix_type::sy: {
      oneapi::mkl::sparse::symv(queue, info->get_uplo(), alpha_value,
                                *sparse_matrix_handle, data_x, beta_value,
                                data_y);
      break;
    }
    case matrix_info::matrix_type::tr: {
      oneapi::mkl::sparse::optimize_trmv(queue, info->get_uplo(), trans,
                                         info->get_diag(),
                                         *sparse_matrix_handle);
      oneapi::mkl::sparse::trmv(
          queue, info->get_uplo(), trans, info->get_diag(), alpha_value,
          *sparse_matrix_handle, data_x, beta_value, data_y);
      break;
    }
    default:
      throw std::runtime_error(
          "the spmv does not support matrix_info::matrix_type::he");
    }

    sycl::event e =
        oneapi::mkl::sparse::release_matrix_handle(queue, sparse_matrix_handle);
    queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(e);
      cgh.host_task([=] { delete sparse_matrix_handle; });
    });
#endif
  }
};
} // namespace detail
} // namespace dpct::sparse

#endif // __DPCT_SPARSE_UTILS_DETAIL_HPP__
