//==---- sparse_utils.hpp -------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_SPARSE_UTILS_HPP__
#define __DPCT_SPARSE_UTILS_HPP__

#include "compat_service.hpp"
#include "lib_common_utils.hpp"

namespace dpct::sparse {
class matrix_info {
public:
  /// Matrix types are:
  /// ge: General matrix
  /// sy: Symmetric matrix
  /// he: Hermitian matrix
  /// tr: Triangular matrix
  enum class matrix_type : int { ge = 0, sy, he, tr };

  auto get_matrix_type() const { return _matrix_type; }
  auto get_diag() const { return _diag; }
  auto get_uplo() const { return _uplo; }
  auto get_index_base() const { return _index_base; }
  void set_matrix_type(matrix_type mt) { _matrix_type = mt; }
  void set_diag(oneapi::mkl::diag d) { _diag = d; }
  void set_uplo(oneapi::mkl::uplo u) { _uplo = u; }
  void set_index_base(oneapi::mkl::index_base ib) { _index_base = ib; }

private:
  matrix_type _matrix_type = matrix_type::ge;
  oneapi::mkl::diag _diag = oneapi::mkl::diag::nonunit;
  oneapi::mkl::uplo _uplo = oneapi::mkl::uplo::upper;
  oneapi::mkl::index_base _index_base = oneapi::mkl::index_base::zero;
};

/// Sparse matrix data format
enum matrix_format : int { csr = 1, coo = 2 };

/// Sparse matrix attribute
enum matrix_attribute : int { uplo = 0, diag };

enum class conversion_scope : int { index = 0, index_and_value };

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
// Forward declaration
namespace detail {
template <typename T> struct optimize_csrsv_impl;
template <typename T> struct optimize_csrsm_impl;
}

/// Saving the optimization information for solving a system of linear
/// equations.
class optimize_info {
public:
  /// Constructor
  optimize_info() { oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle); }
  /// Destructor
  ~optimize_info() {
    oneapi::mkl::sparse::release_matrix_handle(::dpct::cs::get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
  }
  /// Add dependency for the destructor.
  /// \param [in] e The event which the destructor depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }
#ifdef DPCT_USM_LEVEL_NONE
  template <typename T> friend struct detail::optimize_csrsv_impl;
  template <typename T> friend struct detail::optimize_csrsm_impl;
#endif

private:
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
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

/// Structure for describe a sparse matrix
class sparse_matrix_desc {
public:
  /// Constructor
  /// \param [out] desc The descriptor to be created
  /// \param [in] row_num Number of rows of the sparse matrix.
  /// \param [in] col_num Number of colums of the sparse matrix.
  /// \param [in] nnz Non-zero elements in the sparse matrix.
  /// \param [in] row_ptr An array of length \p row_num + 1. If the \p row_ptr is
  /// NULL, the sparse_matrix_desc will allocate internal memory for it. This
  /// internal memory can be gotten from get_shadow_row_ptr().
  /// \param [in] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [in] value An array containing the non-zero elements of the sparse matrix.
  /// \param [in] row_ptr_type Data type of the \p row_ptr .
  /// \param [in] col_ind_type Data type of the \p col_ind .
  /// \param [in] base Indicates how input arrays are indexed.
  /// \param [in] value_type Data type of the \p value .
  /// \param [in] data_format The matrix data format.
  sparse_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                     std::int64_t nnz, void *row_ptr, void *col_ind,
                     void *value, library_data_t row_ptr_type,
                     library_data_t col_ind_type, oneapi::mkl::index_base base,
                     library_data_t value_type, matrix_format data_format)
      : _row_num(row_num), _col_num(col_num), _nnz(nnz), _row_ptr(row_ptr),
        _col_ind(col_ind), _value(value), _row_ptr_type(row_ptr_type),
        _col_ind_type(col_ind_type), _base(base), _value_type(value_type),
        _data_format(data_format) {
    if (_data_format != matrix_format::csr &&
        _data_format != matrix_format::coo) {
      throw std::runtime_error("the sparse matrix data format is unsupported");
    }
    oneapi::mkl::sparse::init_matrix_handle(&_matrix_handle);
    set_data();
  }
  /// Destructor
  ~sparse_matrix_desc() {
    oneapi::mkl::sparse::release_matrix_handle(::dpct::cs::get_default_queue(),
                                               &_matrix_handle, _deps)
        .wait();
  }

  /// Add dependency for the destroy method.
  /// \param [in] e The event which the destroy method depends on.
  void add_dependency(sycl::event e) { _deps.push_back(e); }
  /// Get the internal saved matrix handle.
  /// \return Returns the matrix handle.
  oneapi::mkl::sparse::matrix_handle_t get_matrix_handle() const noexcept {
    return _matrix_handle;
  }
  /// Get the values saved in the descriptor
  /// \param [out] row_num Number of rows of the sparse matrix.
  /// \param [out] col_num Number of colums of the sparse matrix.
  /// \param [out] nnz Non-zero elements in the sparse matrix.
  /// \param [out] row_ptr An array of length \p row_num + 1.
  /// \param [out] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [out] value An array containing the non-zero elements of the sparse matrix.
  /// \param [out] row_ptr_type Data type of the \p row_ptr .
  /// \param [out] col_ind_type Data type of the \p col_ind .
  /// \param [out] base Indicates how input arrays are indexed.
  /// \param [out] value_type Data type of the \p value .
  void get_desc(int64_t *row_num, int64_t *col_num, int64_t *nnz,
                void **row_ptr, void **col_ind, void **value,
                library_data_t *row_ptr_type, library_data_t *col_ind_type,
                oneapi::mkl::index_base *base,
                library_data_t *value_type) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
    *row_ptr = _row_ptr;
    *col_ind = _col_ind;
    *value = _value;
    *row_ptr_type = _row_ptr_type;
    *col_ind_type = _col_ind_type;
    *base = _base;
    *value_type = _value_type;
  }
  /// Get the sparse matrix data format of this descriptor
  /// \param [out] format The matrix data format result
  void get_format(matrix_format *data_format) const noexcept {
    *data_format = _data_format;
  }
  /// Get the index base of this descriptor
  /// \param [out] base The index base result
  void get_base(oneapi::mkl::index_base *base) const noexcept { *base = _base; }
  /// Get the value pointer of this descriptor
  /// \param [out] value The value pointer result
  void get_value(void **value) const noexcept { *value = _value; }
  /// Set the value pointer of this descriptor
  /// \param [in] value The input value pointer
  void set_value(void *value) {
    if (!value) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_value(): The value "
          "pointer is NULL.");
    }
    if (_value && (_value != value)) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_value(): "
          "The _value pointer is not NULL. It cannot be reset.");
    }
    _value = value;
    set_data();
  }
  /// Get the size of the sparse matrix
  /// \param [out] row_num Number of rows of the sparse matrix.
  /// \param [out] col_num Number of colums of the sparse matrix.
  /// \param [out] nnz Non-zero elements in the sparse matrix.
  void get_size(int64_t *row_num, int64_t *col_num,
                int64_t *nnz) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *nnz = _nnz;
  }
  /// Set the sparse matrix attribute
  /// \param [in] attribute The attribute type
  /// \param [in] data The attribute value
  /// \param [in] data_size The data size of the attribute value
  void set_attribute(matrix_attribute attribute, const void *data,
                     size_t data_size) {
    if (attribute == matrix_attribute::diag) {
      const oneapi::mkl::diag *diag_ptr =
          reinterpret_cast<const oneapi::mkl::diag *>(data);
      if (*diag_ptr == oneapi::mkl::diag::unit) {
        _diag = oneapi::mkl::diag::unit;
      } else if (*diag_ptr == oneapi::mkl::diag::nonunit) {
        _diag = oneapi::mkl::diag::nonunit;
      } else {
        throw std::runtime_error("unsupported diag value");
      }
    } else if (attribute == matrix_attribute::uplo) {
      const oneapi::mkl::uplo *uplo_ptr =
          reinterpret_cast<const oneapi::mkl::uplo *>(data);
      if (*uplo_ptr == oneapi::mkl::uplo::upper) {
        _uplo = oneapi::mkl::uplo::upper;
      } else if (*uplo_ptr == oneapi::mkl::uplo::lower) {
        _uplo = oneapi::mkl::uplo::lower;
      } else {
        throw std::runtime_error("unsupported uplo value");
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  /// Get the sparse matrix attribute
  /// \param [out] attribute The attribute type
  /// \param [out] data The attribute value
  /// \param [out] data_size The data size of the attribute value
  void get_attribute(matrix_attribute attribute, void *data,
                     size_t data_size) const {
    if (attribute == matrix_attribute::diag) {
      oneapi::mkl::diag *diag_ptr = reinterpret_cast<oneapi::mkl::diag *>(data);
      if (_diag.has_value()) {
        *diag_ptr = _diag.value();
      } else {
        *diag_ptr = oneapi::mkl::diag::nonunit;
      }
    } else if (attribute == matrix_attribute::uplo) {
      oneapi::mkl::uplo *uplo_ptr = reinterpret_cast<oneapi::mkl::uplo *>(data);
      if (_uplo.has_value()) {
        *uplo_ptr = _uplo.value();
      } else {
        *uplo_ptr = oneapi::mkl::uplo::lower;
      }
    } else {
      throw std::runtime_error("unsupported attribute");
    }
  }
  /// Set the pointers for describing the sparse matrix
  /// \param [in] row_ptr An array of length \p row_num + 1.
  /// \param [in] col_ind An array containing the column indices in index-based
  /// numbering.
  /// \param [in] value An array containing the non-zero elements of the sparse matrix.
  void set_pointers(void *row_ptr, void *col_ind, void *value) {
    if (!row_ptr) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_pointers(): The "
          "row_ptr pointer is NULL.");
    }
    if (!col_ind) {
      throw std::runtime_error(
          "dpct::sparse::sparse_matrix_desc::set_pointers(): The "
          "col_ind pointer is NULL.");
    }
    if (_row_ptr && (_row_ptr != row_ptr)) {
      throw std::runtime_error("dpct::sparse::sparse_matrix_desc::set_pointers("
                               "): The _row_ptr pointer is "
                               "not NULL. It cannot be reset.");
    }
    if (_col_ind && (_col_ind != col_ind)) {
      throw std::runtime_error("dpct::sparse::sparse_matrix_desc::set_pointers("
                               "): The _col_ind pointer is "
                               "not NULL. It cannot be reset.");
    }
    _row_ptr = row_ptr;
    _col_ind = col_ind;

    // The descriptor will be updated in the set_value function
    set_value(value);
  }

  /// Get the diag attribute
  /// \return diag value
  std::optional<oneapi::mkl::diag> get_diag() const noexcept { return _diag; }
  /// Get the uplo attribute
  /// \return uplo value
  std::optional<oneapi::mkl::uplo> get_uplo() const noexcept { return _uplo; }
  /// Set the number of non-zero elements
  /// \param nnz [in] The number of non-zero elements.
  void set_nnz(std::int64_t nnz) noexcept { _nnz = nnz; }
  /// Get the type of the value pointer.
  /// \return The type of the value pointer.
  library_data_t get_value_type() const noexcept { return _value_type; }
  /// Get the row_ptr.
  /// \return The row_ptr.
  void *get_row_ptr() const noexcept { return _row_ptr; }
  /// If the internal _row_ptr is NULL, the sparse_matrix_desc will allocate
  /// internal memory for it in the constructor. The internal memory can be gotten
  /// from this interface.
  /// \return The shadow row_ptr.
  void *get_shadow_row_ptr() const noexcept { return _shadow_row_ptr.get(); }
  /// Get the type of the col_ind pointer.
  /// \return The type of the col_ind pointer.
  library_data_t get_col_ind_type() const noexcept { return _col_ind_type; }
  /// Get the row_num.
  /// \return The row_num.
  std::int64_t get_row_num() const noexcept { return _row_num; }

private:
  inline static const std::function<void(void *)> _shadow_row_ptr_deleter =
      [](void *ptr) { ::dpct::cs::free(ptr, ::dpct::cs::get_default_queue()); };
  template <typename index_t, typename value_t> void set_data() {
    void *row_ptr = nullptr;
    if (_shadow_row_ptr) {
      row_ptr = _shadow_row_ptr.get();
    } else if (_row_ptr) {
      row_ptr = _row_ptr;
    } else {
      row_ptr = ::dpct::cs::malloc(sizeof(index_t) * (_row_num + 1),
                                   ::dpct::cs::get_default_queue());
      _shadow_row_ptr.reset(row_ptr);
    }
#ifdef DPCT_USM_LEVEL_NONE
    using data_index_t = sycl::buffer<index_t>;
    using data_value_t = sycl::buffer<value_t>;
#else
    using data_index_t = index_t *;
    using data_value_t = value_t *;
#endif
    _data_row_ptr = dpct::detail::get_memory<index_t>(row_ptr);
    _data_col_ind = dpct::detail::get_memory<index_t>(_col_ind);
    _data_value = dpct::detail::get_memory<value_t>(_value);
    if (_data_format == matrix_format::csr)
      oneapi::mkl::sparse::set_csr_data(
          ::dpct::cs::get_default_queue(), _matrix_handle, _row_num, _col_num,
          _base, std::get<data_index_t>(_data_row_ptr),
          std::get<data_index_t>(_data_col_ind),
          std::get<data_value_t>(_data_value));
    else
      oneapi::mkl::sparse::set_coo_data(
          ::dpct::cs::get_default_queue(), _matrix_handle, _row_num, _col_num,
          _nnz, _base, std::get<data_index_t>(_data_row_ptr),
          std::get<data_index_t>(_data_col_ind),
          std::get<data_value_t>(_data_value));
    ::dpct::cs::get_default_queue().wait();
  }

  void set_data() {
    std::uint64_t key = dpct::detail::get_type_combination_id(
        _row_ptr_type, _col_ind_type, _value_type);
    switch (key) {
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::real_float): {
      set_data<std::int32_t, float>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::real_double): {
      set_data<std::int32_t, double>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int32,
                                               library_data_t::real_int32,
                                               library_data_t::complex_float): {
      set_data<std::int32_t, std::complex<float>>();
      break;
    }
    case dpct::detail::get_type_combination_id(
        library_data_t::real_int32, library_data_t::real_int32,
        library_data_t::complex_double): {
      set_data<std::int32_t, std::complex<double>>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::real_float): {
      set_data<std::int64_t, float>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::real_double): {
      set_data<std::int64_t, double>();
      break;
    }
    case dpct::detail::get_type_combination_id(library_data_t::real_int64,
                                               library_data_t::real_int64,
                                               library_data_t::complex_float): {
      set_data<std::int64_t, std::complex<float>>();
      break;
    }
    case dpct::detail::get_type_combination_id(
        library_data_t::real_int64, library_data_t::real_int64,
        library_data_t::complex_double): {
      set_data<std::int64_t, std::complex<double>>();
      break;
    }
    default:
      throw std::runtime_error("the combination of data type is unsupported");
    }
  }

  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _nnz;
  void *_row_ptr;
  void *_col_ind;
  void *_value;
  library_data_t _row_ptr_type;
  library_data_t _col_ind_type;
  oneapi::mkl::index_base _base;
  library_data_t _value_type;
  oneapi::mkl::sparse::matrix_handle_t _matrix_handle = nullptr;
  std::vector<sycl::event> _deps;
  matrix_format _data_format;
  std::optional<oneapi::mkl::uplo> _uplo;
  std::optional<oneapi::mkl::diag> _diag;
  std::unique_ptr<void, std::function<void(void *)>> _shadow_row_ptr =
      std::unique_ptr<void, std::function<void(void *)>>(
          nullptr, _shadow_row_ptr_deleter);

  static constexpr size_t _max_data_variable_size = (std::max)(
      {sizeof(sycl::buffer<std::int32_t>), sizeof(sycl::buffer<std::int64_t>),
       sizeof(sycl::buffer<float>), sizeof(sycl::buffer<double>),
       sizeof(sycl::buffer<std::complex<float>>),
       sizeof(sycl::buffer<std::complex<double>>), sizeof(void *)});
  using index_variant_t =
      std::variant<std::array<std::byte, _max_data_variable_size>,
                   sycl::buffer<std::int32_t>, sycl::buffer<std::int64_t>,
                   std::int32_t *, std::int64_t *>;
  using value_variant_t =
      std::variant<std::array<std::byte, _max_data_variable_size>,
                   sycl::buffer<float>, sycl::buffer<double>,
                   sycl::buffer<std::complex<float>>,
                   sycl::buffer<std::complex<double>>, float *, double *,
                   std::complex<float> *, std::complex<double> *>;
  index_variant_t _data_row_ptr;
  index_variant_t _data_col_ind;
  value_variant_t _data_value;
};
#endif

class sparse_matrix_desc;
using sparse_matrix_desc_t = std::shared_ptr<sparse_matrix_desc>;

/// Structure for describe a dense vector
class dense_vector_desc {
public:
  dense_vector_desc(std::int64_t ele_num, void *value,
                    library_data_t value_type)
      : _ele_num(ele_num), _value(value), _value_type(value_type) {}
  void get_desc(std::int64_t *ele_num, const void **value,
                library_data_t *value_type) const noexcept {
    *ele_num = _ele_num;
    *value = _value;
    *value_type = _value_type;
  }
  void get_desc(std::int64_t *ele_num, void **value,
                library_data_t *value_type) const noexcept {
    get_desc(ele_num, const_cast<const void **>(value), value_type);
  }
  void *get_value() const noexcept { return _value; }
  void set_value(void *value) { _value = value; }
  library_data_t get_value_type() const noexcept { return _value_type; }
  std::int64_t get_ele_num() const noexcept { return _ele_num; }

private:
  std::int64_t _ele_num;
  void *_value;
  library_data_t _value_type;
};

/// Structure for describe a dense matrix
class dense_matrix_desc {
public:
  dense_matrix_desc(std::int64_t row_num, std::int64_t col_num,
                    std::int64_t leading_dim, void *value,
                    library_data_t value_type, oneapi::mkl::layout layout)
      : _row_num(row_num), _col_num(col_num), _leading_dim(leading_dim),
        _value(value), _value_type(value_type), _layout(layout) {}
  void get_desc(std::int64_t *row_num, std::int64_t *col_num,
                std::int64_t *leading_dim, void **value,
                library_data_t *value_type,
                oneapi::mkl::layout *layout) const noexcept {
    *row_num = _row_num;
    *col_num = _col_num;
    *leading_dim = _leading_dim;
    *value = _value;
    *value_type = _value_type;
    *layout = _layout;
  }
  void *get_value() const noexcept { return _value; }
  void set_value(void *value) { _value = value; }
  std::int64_t get_col_num() const noexcept { return _col_num; }
  std::int64_t get_leading_dim() const noexcept { return _leading_dim; }
  oneapi::mkl::layout get_layout() const noexcept { return _layout; }

private:
  std::int64_t _row_num;
  std::int64_t _col_num;
  std::int64_t _leading_dim;
  void *_value;
  library_data_t _value_type;
  oneapi::mkl::layout _layout;
};
} // namespace dpct::sparse

#include "detail/sparse_utils_detail.hpp"

namespace dpct::sparse {
class descriptor {
public:
  descriptor() {}
  void set_queue(::dpct::cs::queue_ptr q_ptr) noexcept { _queue_ptr = q_ptr; }
  sycl::queue &get_queue() const noexcept { return *_queue_ptr; }

private:
#ifdef __INTEL_MKL__
  struct matmat_info {
    detail::matrix_handle_manager matrix_handle_a;
    detail::matrix_handle_manager matrix_handle_b;
    detail::matrix_handle_manager matrix_handle_c;
    detail::handle_manager<oneapi::mkl::sparse::matmat_descr_t> matmat_desc;
    bool is_empty() { return matmat_desc.is_empty(); }
    void init(sycl::queue *q_ptr) {
      matmat_desc.init(q_ptr);
      matrix_handle_a.init(q_ptr);
      matrix_handle_b.init(q_ptr);
      matrix_handle_c.init(q_ptr);
    }
  };
  std::unordered_map<detail::csrgemm_args_info, matmat_info,
                     detail::csrgemm_args_info_hash> &
  get_csrgemm_info_map() {
    return _csrgemm_info_map;
  }

  std::unordered_map<detail::csrgemm_args_info, matmat_info,
                     detail::csrgemm_args_info_hash>
      _csrgemm_info_map;

  template <typename T>
  friend void
  csrgemm_nnz(descriptor *desc, oneapi::mkl::transpose trans_a,
              oneapi::mkl::transpose trans_b, int m, int n, int k,
              const std::shared_ptr<matrix_info> info_a, int nnz_a,
              const T *val_a, const int *row_ptr_a, const int *col_ind_a,
              const std::shared_ptr<matrix_info> info_b, int nnz_b,
              const T *val_b, const int *row_ptr_b, const int *col_ind_b,
              const std::shared_ptr<matrix_info> info_c, int *row_ptr_c,
              int *nnz_host_ptr);
  template <typename T>
  friend void csrgemm(descriptor *desc, oneapi::mkl::transpose trans_a,
                      oneapi::mkl::transpose trans_b, int m, int n, int k,
                      const std::shared_ptr<matrix_info> info_a, const T *val_a,
                      const int *row_ptr_a, const int *col_ind_a,
                      const std::shared_ptr<matrix_info> info_b, const T *val_b,
                      const int *row_ptr_b, const int *col_ind_b,
                      const std::shared_ptr<matrix_info> info_c, T *val_c,
                      const int *row_ptr_c, int *col_ind_c);
#endif
  ::dpct::cs::queue_ptr _queue_ptr = &::dpct::cs::get_default_queue();
};

using descriptor_ptr = descriptor *;

/// Computes a CSR format sparse matrix-dense vector product.
/// y = alpha * op(A) * x + beta * y
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] num_rows Number of rows of the matrix A.
/// \param [in] num_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] x Data of the vector x.
/// \param [in] beta Scaling factor for the vector x.
/// \param [in, out] y Data of the vector y.
template <typename T>
void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans, int num_rows,
           int num_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *x, const T *beta,
           T *y) {
  detail::csrmv_impl<T>()(queue, trans, num_rows, num_cols, alpha, info, val,
                          row_ptr, col_ind, x, beta, y);
}

/// Computes a CSR format sparse matrix-dense vector product. y = A * x
///
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] values An array containing the non-zero elements of the matrix.
/// \param [in] row_offsets An array of length \p num_rows + 1.
/// \param [in] column_indices An array containing the column indices in
/// index-based numbering.
/// \param [in] vector_x Data of the vector x.
/// \param [in, out] vector_y Data of the vector y.
/// \param [in] num_rows Number of rows of the matrix A.
/// \param [in] num_cols Number of columns of the matrix A.
template <typename T>
void csrmv(sycl::queue &queue, const T *values, const int *row_offsets,
           const int *column_indices, const T *vector_x, T *vector_y,
           int num_rows, int num_cols) {
  T alpha{1}, beta{0};
  auto matrix_info = std::make_shared<dpct::sparse::matrix_info>();
  matrix_info->set_index_base(oneapi::mkl::index_base::zero);
  matrix_info->set_matrix_type(dpct::sparse::matrix_info::matrix_type::ge);
  detail::csrmv_impl<T>()(queue, oneapi::mkl::transpose::nontrans, num_rows,
                          num_cols, &alpha, matrix_info, values, row_offsets,
                          column_indices, vector_x, &beta, vector_y);
}

/// Computes a CSR format sparse matrix-dense vector product.
/// y = alpha * op(A) * x + beta * y
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] num_rows Number of rows of the matrix A.
/// \param [in] num_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] alpha_type Data type of \p alpha .
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] val_type Data type of \p val .
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] x Data of the vector x.
/// \param [in] x_type Data type of \p x .
/// \param [in] beta Scaling factor for the vector x.
/// \param [in] beta_type Data type of \p beta .
/// \param [in, out] y Data of the vector y.
/// \param [in] y_type Data type of \p y .
inline void csrmv(sycl::queue &queue, oneapi::mkl::transpose trans,
                  int num_rows, int num_cols, const void *alpha,
                  library_data_t alpha_type,
                  const std::shared_ptr<matrix_info> info, const void *val,
                  library_data_t val_type, const int *row_ptr,
                  const int *col_ind, const void *x, library_data_t x_type,
                  const void *beta, library_data_t beta_type, void *y,
                  library_data_t y_type) {
  detail::spblas_shim<detail::csrmv_impl>(val_type, queue, trans, num_rows,
                                          num_cols, alpha, info, val, row_ptr,
                                          col_ind, x, beta, y);
}

/// Computes a CSR format sparse matrix-dense matrix product.
/// C = alpha * op(A) * op(B) + beta * C
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a The operation applied to the matrix A.
/// \param [in] trans_b The operation applied to the matrix B.
/// \param [in] sparse_rows Number of rows of the matrix A.
/// \param [in] dense_cols Number of columns of the matrix B or C.
/// \param [in] sparse_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] b Data of the matrix B.
/// \param [in] ldb Leading dimension of the matrix B.
/// \param [in] beta Scaling factor for the matrix B.
/// \param [in, out] c Data of the matrix C.
/// \param [in] ldc Leading dimension of the matrix C.
template <typename T>
void csrmm(sycl::queue &queue, oneapi::mkl::transpose trans_a,
           oneapi::mkl::transpose trans_b, int sparse_rows, int dense_cols,
           int sparse_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *b, int ldb,
           const T *beta, T *c, int ldc) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
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
  oneapi::mkl::sparse::set_csr_data(queue, *sparse_matrix_handle, sparse_rows,
                                    sparse_cols, info->get_index_base(),
                                    data_row_ptr, data_col_ind, data_val);

  auto data_b = dpct::detail::get_memory<Ty>(b);
  auto data_c = dpct::detail::get_memory<Ty>(c);
  sycl::event gemm_event;
  switch (info->get_matrix_type()) {
  case matrix_info::matrix_type::ge: {
#ifndef DPCT_USM_LEVEL_NONE
    gemm_event =
#endif
        oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::col_major,
                                  trans_a, trans_b, alpha_value,
                                  *sparse_matrix_handle, data_b, dense_cols,
                                  ldb, beta_value, data_c, ldc);
    break;
  }
  default:
    throw std::runtime_error(
        "the csrmm does not support matrix_info::matrix_type::sy, "
        "matrix_info::matrix_type::tr and matrix_info::matrix_type::he");
  }
#ifdef DPCT_USM_LEVEL_NONE
  queue.wait();
#endif
  sycl::event e = oneapi::mkl::sparse::release_matrix_handle(
      queue, sparse_matrix_handle, {gemm_event});
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete sparse_matrix_handle; });
  });
#endif
}

/// Computes a CSR format sparse matrix-dense matrix product.
/// C = alpha * op(A) * B + beta * C
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the matrix A.
/// \param [in] sparse_rows Number of rows of the matrix A.
/// \param [in] dense_cols Number of columns of the matrix op(B) or C.
/// \param [in] sparse_cols Number of columns of the matrix A.
/// \param [in] alpha Scaling factor for the matrix A.
/// \param [in] info Matrix info of the matrix A.
/// \param [in] val An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in] b Data of the matrix B.
/// \param [in] ldb Leading dimension of the matrix B.
/// \param [in] beta Scaling factor for the matrix B.
/// \param [in, out] c Data of the matrix C.
/// \param [in] ldc Leading dimension of the matrix C.
template <typename T>
void csrmm(sycl::queue &queue, oneapi::mkl::transpose trans, int sparse_rows,
           int dense_cols, int sparse_cols, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, const T *b, int ldb,
           const T *beta, T *c, int ldc) {
  csrmm<T>(queue, trans, oneapi::mkl::transpose::nontrans, sparse_rows,
           dense_cols, sparse_cols, alpha, info, val, row_ptr, col_ind, b, ldb,
           beta, c, ldc);
}

#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
/// Performs internal optimizations for solving a system of linear equations for
/// a CSR format sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans The operation applied to the sparse matrix.
/// \param [in] row_col Number of rows of the sparse matrix.
/// \param [in] info Matrix info of the sparse matrix.
/// \param [in] val An array containing the non-zero elements of the sparse matrix.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [out] optimize_info The result of the optimizations.
template <typename T>
void optimize_csrsv(sycl::queue &queue, oneapi::mkl::transpose trans,
                    int row_col, const std::shared_ptr<matrix_info> info,
                    const T *val, const int *row_ptr, const int *col_ind,
                    std::shared_ptr<optimize_info> optimize_info) {
  detail::optimize_csrsv_impl<T>()(queue, trans, row_col, info, val, row_ptr,
                                   col_ind, optimize_info);
}

inline void optimize_csrsv(sycl::queue &queue, oneapi::mkl::transpose trans,
                           int row_col, const std::shared_ptr<matrix_info> info,
                           const void *val, library_data_t val_type,
                           const int *row_ptr, const int *col_ind,
                           std::shared_ptr<optimize_info> optimize_info) {
  detail::spblas_shim<detail::optimize_csrsv_impl>(
      val_type, queue, trans, row_col, info, val, row_ptr, col_ind,
      optimize_info);
}

template <typename T>
void csrsv(sycl::queue &queue, oneapi::mkl::transpose trans, int row_col,
           const T *alpha, const std::shared_ptr<matrix_info> info,
           const T *val, const int *row_ptr, const int *col_ind,
           std::shared_ptr<optimize_info> optimize_info, const T *x, T *y) {
  detail::csrsv_impl<T>()(queue, trans, row_col, alpha, info, val, row_ptr,
                          col_ind, optimize_info, x, y);
}

inline void csrsv(sycl::queue &queue, oneapi::mkl::transpose trans, int row_col,
                  const void *alpha, library_data_t alpha_type,
                  const std::shared_ptr<matrix_info> info, const void *val,
                  library_data_t val_type, const int *row_ptr,
                  const int *col_ind,
                  std::shared_ptr<optimize_info> optimize_info, const void *x,
                  library_data_t x_type, void *y, library_data_t y_type) {
  detail::spblas_shim<detail::csrsv_impl>(val_type, queue, trans, row_col,
                                          alpha, info, val, row_ptr, col_ind,
                                          optimize_info, x, y);
}

/// Performs internal optimizations for dpct::sparse::csrsm by analyzing
/// the provided matrix structure and operation parameters. The matrix A must be
/// a triangular sparse matrix with the CSR format.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] transa The operation applied to A.
/// \param [in] transb The operation applied to B and X.
/// \param [in] row_col Number of rows and columns of A.
/// \param [in] nrhs Number of columns op_b(B).
/// \param [in] info Matrix info of A.
/// \param [in] val An array containing the non-zero elements of A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [out] optimize_info The result of the optimizations.
template <typename T>
void optimize_csrsm(sycl::queue &queue, oneapi::mkl::transpose transa,
                    oneapi::mkl::transpose transb, int row_col, int nrhs,
                    const std::shared_ptr<matrix_info> info, const T *val,
                    const int *row_ptr, const int *col_ind,
                    std::shared_ptr<optimize_info> optimize_info) {
  detail::optimize_csrsm_impl<T>()(queue, transa, transb, row_col, nrhs, info,
                                   val, row_ptr, col_ind, optimize_info);
}

/// Solves the sparse triangular system op_a(A) * op_b(X) = alpha * op_b(B).
/// A is a sparse triangular matrix with the CSR format of size \p row_col
/// by \p row_col .
/// B is a dense matrix of size \p row_col by \p nrhs ( \p transb is nontrans)
/// or \p nrhs by \p row_col ( \p transb isn't nontrans).
/// X is the solution dense matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] transa The operation applied to A.
/// \param [in] transb The operation applied to B and X.
/// \param [in] row_col Number of rows and columns of A.
/// \param [in] nrhs Number of columns op_b(B).
/// \param [in] alpha Specifies the scalar.
/// \param [in] info Matrix info of A.
/// \param [in] val An array containing the non-zero elements of A.
/// \param [in] row_ptr An array of length \p num_rows + 1.
/// \param [in] col_ind An array containing the column indices in index-based
/// numbering.
/// \param [in, out] b The RHS matrix. It will be overwritten by the X.
/// \param [in] ldb The leading dimension of B and X.
/// \param [in] optimize_info The result of the optimizations.
template <typename T>
void csrsm(sycl::queue &queue, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, int row_col, int nrhs, const T *alpha,
           const std::shared_ptr<matrix_info> info, const T *val,
           const int *row_ptr, const int *col_ind, T *b, int ldb,
           std::shared_ptr<optimize_info> optimize_info) {
  detail::csrsm_impl<T>()(queue, transa, transb, row_col, nrhs, alpha, info,
                          val, row_ptr, col_ind, b, ldb, optimize_info);
}

/// Computes a sparse matrix-dense vector product: y = alpha * op(a) * x + beta * y.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans Specifies operation on input matrix.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] x Specifies the dense vector x.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] y Specifies the dense vector y.
/// \param [in] data_type Specifies the data type of \param a, \param x and \param y .
inline void spmv(sycl::queue queue, oneapi::mkl::transpose trans,
                 const void *alpha, sparse_matrix_desc_t a,
                 std::shared_ptr<dense_vector_desc> x, const void *beta,
                 std::shared_ptr<dense_vector_desc> y,
                 library_data_t data_type) {
  detail::spblas_shim<detail::spmv_impl>(data_type, queue, trans, alpha, a, x,
                                         beta, y);
}

/// Computes a sparse matrix-dense matrix product: c = alpha * op(a) * op(b) + beta * c.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the dense matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the dense matrix c.
/// \param [in] data_type Specifies the data type of \param a, \param b and \param c .
inline void spmm(sycl::queue queue, oneapi::mkl::transpose trans_a,
                 oneapi::mkl::transpose trans_b, const void *alpha,
                 sparse_matrix_desc_t a, std::shared_ptr<dense_matrix_desc> b,
                 const void *beta, std::shared_ptr<dense_matrix_desc> c,
                 library_data_t data_type) {
  if (b->get_layout() != c->get_layout())
    throw std::runtime_error("the layout of b and c are different");
  detail::spblas_shim<detail::spmm_impl>(data_type, queue, trans_a, trans_b,
                                         alpha, a, b, beta, c);
}

/// Do initial estimation of work and load balancing of computing a sparse
/// matrix-sparse matrix product.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the sparse matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the sparse matrix c.
/// \param [in] matmat_descr Describes the sparse matrix-sparse matrix operation
/// to be executed.
/// \param [in, out] size_temp_buffer Specifies the size of workspace.
/// \param [in] temp_buffer Specifies the memory of the workspace.
inline void
spgemm_work_estimation(sycl::queue queue, oneapi::mkl::transpose trans_a,
                       oneapi::mkl::transpose trans_b, const void *alpha,
                       sparse_matrix_desc_t a, sparse_matrix_desc_t b,
                       const void *beta, sparse_matrix_desc_t c,
                       oneapi::mkl::sparse::matmat_descr_t matmat_descr,
                       size_t *size_temp_buffer, void *temp_buffer) {
  if (temp_buffer) {
    detail::temp_memory<std::int64_t, true, size_t> size_memory(
        queue, size_temp_buffer);
    detail::temp_memory<std::uint8_t, false> work_memory(queue, temp_buffer);
    oneapi::mkl::sparse::matmat(
        queue, a->get_matrix_handle(), b->get_matrix_handle(),
        c->get_matrix_handle(),
        oneapi::mkl::sparse::matmat_request::work_estimation, matmat_descr,
        size_memory.get_memory_ptr(), work_memory.get_memory_ptr()
#ifndef DPCT_USM_LEVEL_NONE
        , {}
#endif
    );
  } else {
    oneapi::mkl::sparse::set_matmat_data(
        matmat_descr, oneapi::mkl::sparse::matrix_view_descr::general, trans_a,
        oneapi::mkl::sparse::matrix_view_descr::general, trans_b,
        oneapi::mkl::sparse::matrix_view_descr::general);
    detail::temp_memory<std::int64_t, true, size_t> size_memory(
        queue, size_temp_buffer);
    oneapi::mkl::sparse::matmat(
        queue, a->get_matrix_handle(), b->get_matrix_handle(),
        c->get_matrix_handle(),
        oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size,
        matmat_descr, size_memory.get_memory_ptr(), nullptr
#ifndef DPCT_USM_LEVEL_NONE
        , {}
#endif
    );
  }
}

/// Do internal products for computing the C matrix of computing a sparse
/// matrix-sparse matrix product.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the sparse matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the sparse matrix c.
/// \param [in] matmat_descr Describes the sparse matrix-sparse matrix operation
/// to be executed.
/// \param [in, out] size_temp_buffer Specifies the size of workspace.
/// \param [in] temp_buffer Specifies the memory of the workspace.
inline void spgemm_compute(sycl::queue queue, oneapi::mkl::transpose trans_a,
                           oneapi::mkl::transpose trans_b, const void *alpha,
                           sparse_matrix_desc_t a, sparse_matrix_desc_t b,
                           const void *beta, sparse_matrix_desc_t c,
                           oneapi::mkl::sparse::matmat_descr_t matmat_descr,
                           size_t *size_temp_buffer, void *temp_buffer) {
  if (temp_buffer) {
    std::int64_t nnz_value = 0;
    {
      detail::temp_memory<std::int64_t, true, size_t> size_memory(
          queue, size_temp_buffer);
      detail::temp_memory<std::uint8_t, false> work_memory(queue, temp_buffer);
      detail::temp_memory<std::int64_t, true, std::int64_t> nnz_memory(
          queue, &nnz_value);
      oneapi::mkl::sparse::matmat(
          queue, a->get_matrix_handle(), b->get_matrix_handle(),
          c->get_matrix_handle(), oneapi::mkl::sparse::matmat_request::compute,
          matmat_descr, size_memory.get_memory_ptr(),
          work_memory.get_memory_ptr()
#ifndef DPCT_USM_LEVEL_NONE
          , {}
#endif
      );
      oneapi::mkl::sparse::matmat(
          queue, a->get_matrix_handle(), b->get_matrix_handle(),
          c->get_matrix_handle(), oneapi::mkl::sparse::matmat_request::get_nnz,
          matmat_descr, nnz_memory.get_memory_ptr(), nullptr
#ifndef DPCT_USM_LEVEL_NONE
          , {}
#endif
      );
    }
    c->set_nnz(nnz_value);
  } else {
    detail::temp_memory<std::int64_t, true, size_t> size_memory(
        queue, size_temp_buffer);
    oneapi::mkl::sparse::matmat(
        queue, a->get_matrix_handle(), b->get_matrix_handle(),
        c->get_matrix_handle(),
        oneapi::mkl::sparse::matmat_request::get_compute_buf_size, matmat_descr,
        size_memory.get_memory_ptr(), nullptr
#ifndef DPCT_USM_LEVEL_NONE
        , {}
#endif
    );
  }
}

/// Do any remaining internal products and accumulation and transfer into final
/// C matrix arrays of computing a sparse matrix-sparse matrix product.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the sparse matrix b.
/// \param [in] beta Specifies the scalar beta.
/// \param [in, out] c Specifies the sparse matrix c.
/// \param [in] matmat_descr Describes the sparse matrix-sparse matrix operation
/// to be executed.
inline void spgemm_finalize(sycl::queue queue, oneapi::mkl::transpose trans_a,
                            oneapi::mkl::transpose trans_b, const void *alpha,
                            sparse_matrix_desc_t a, sparse_matrix_desc_t b,
                            const void *beta, sparse_matrix_desc_t c,
                            oneapi::mkl::sparse::matmat_descr_t matmat_descr) {
  oneapi::mkl::sparse::matmat(queue, a->get_matrix_handle(),
                              b->get_matrix_handle(), c->get_matrix_handle(),
                              oneapi::mkl::sparse::matmat_request::finalize,
                              matmat_descr, nullptr, nullptr
#ifdef DPCT_USM_LEVEL_NONE
  );
#else
  , {}).wait();
#endif
  if (c->get_shadow_row_ptr()) {
    switch (c->get_col_ind_type()) {
    case library_data_t::real_int32: {
      ::dpct::cs::memcpy(::dpct::cs::get_default_queue(), c->get_row_ptr(),
                         c->get_shadow_row_ptr(),
                         sizeof(std::int32_t) * (c->get_row_num() + 1))
          .wait();
      break;
    }
    case library_data_t::real_int64: {
      ::dpct::cs::memcpy(::dpct::cs::get_default_queue(), c->get_row_ptr(),
                         c->get_shadow_row_ptr(),
                         sizeof(std::int64_t) * (c->get_row_num() + 1))
          .wait();
      break;
    }
    default:
      throw std::runtime_error("dpct::sparse::spgemm_finalize(): The data type "
                               "of the col_ind in matrix c is unsupported.");
    }
  }
}

/// Performs internal optimizations for spsv by analyzing the provided matrix
/// structure and operation parameters.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] a Specifies the sparse matrix a.
inline void spsv_optimize(sycl::queue queue, oneapi::mkl::transpose trans_a,
                          sparse_matrix_desc_t a) {
  if (!a->get_uplo() || !a->get_diag()) {
    throw std::runtime_error(
        "dpct::sparse::spsv_optimize(): oneapi::mkl::sparse::optimize_trsv "
        "needs uplo and diag attributes to be specified.");
  }
  oneapi::mkl::sparse::optimize_trsv(
      queue, a->get_uplo().value(), oneapi::mkl::transpose::nontrans,
      a->get_diag().value(), a->get_matrix_handle());
}

/// Solves a system of linear equations for a sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] x Specifies the dense vector x.
/// \param [in, out] y Specifies the dense vector y.
/// \param [in] data_type Specifies the data type of \param a, \param x and
/// \param y .
inline void spsv(sycl::queue queue, oneapi::mkl::transpose trans_a,
                 const void *alpha, sparse_matrix_desc_t a,
                 std::shared_ptr<dense_vector_desc> x,
                 std::shared_ptr<dense_vector_desc> y,
                 library_data_t data_type) {
  if (!a->get_uplo() || !a->get_diag()) {
    throw std::runtime_error(
        "dpct::sparse::spsv(): oneapi::mkl::sparse::trsv needs uplo and diag "
        "attributes to be specified.");
  }
  oneapi::mkl::uplo uplo = a->get_uplo().value();
  oneapi::mkl::diag diag = a->get_diag().value();
  detail::spblas_shim<detail::spsv_impl>(a->get_value_type(), queue, uplo, diag,
                                         trans_a, alpha, a, x, y);
}

/// Performs internal optimizations for spsm by analyzing the provided matrix
/// structure and operation parameters.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] a Specifies the dense matrix b.
/// \param [in] a Specifies the dense matrix c.
inline void spsm_optimize(sycl::queue queue, oneapi::mkl::transpose trans_a,
                          sparse_matrix_desc_t a,
                          std::shared_ptr<dense_matrix_desc> b,
                          std::shared_ptr<dense_matrix_desc> c) {
  if (!a->get_uplo() || !a->get_diag()) {
    throw std::runtime_error(
        "dpct::sparse::spsm_optimize(): oneapi::mkl::sparse::optimize_trsm "
        "needs uplo and diag attributes to be specified.");
  }
  oneapi::mkl::sparse::optimize_trsm(
      queue, b->get_layout(), a->get_uplo().value(), trans_a,
      a->get_diag().value(), a->get_matrix_handle(), c->get_col_num());
}

/// Solves a system of linear equations with multiple right-hand sides (RHS)
/// for a triangular sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] trans_a Specifies operation on input matrix a.
/// \param [in] trans_b Specifies operation on input matrix b.
/// \param [in] alpha Specifies the scalar alpha.
/// \param [in] a Specifies the sparse matrix a.
/// \param [in] b Specifies the dense matrix b.
/// \param [in, out] c Specifies the dense matrix c.
/// \param [in] data_type Specifies the data type of \param a, \param b and
/// \param c .
inline void spsm(sycl::queue queue, oneapi::mkl::transpose trans_a,
                 oneapi::mkl::transpose trans_b, const void *alpha,
                 sparse_matrix_desc_t a, std::shared_ptr<dense_matrix_desc> b,
                 std::shared_ptr<dense_matrix_desc> c,
                 library_data_t data_type) {
  if (!a->get_uplo() || !a->get_diag()) {
    throw std::runtime_error(
        "dpct::sparse::spsm(): oneapi::mkl::sparse::trsm needs uplo and diag "
        "attributes to be specified.");
  }
  oneapi::mkl::uplo uplo = a->get_uplo().value();
  oneapi::mkl::diag diag = a->get_diag().value();
  detail::spblas_shim<detail::spsm_impl>(data_type, queue, trans_a, trans_b,
                                         uplo, diag, alpha, a, b, c);
}

/// Convert a CSR sparse matrix to a CSC sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] m Number of rows of the matrix.
/// \param [in] n Number of columns of the matrix.
/// \param [in] nnz Number of non-zero elements.
/// \param [in] from_val An array containing the non-zero elements of the input
/// matrix.
/// \param [in] from_row_ptr An array of length \p m + 1.
/// \param [in] from_col_ind An array containing the column indices in
/// index-based numbering.
/// \param [out] to_val An array containing the non-zero elements of the output
/// matrix.
/// \param [out] to_col_ptr An array of length \p n + 1.
/// \param [out] to_row_ind An array containing the row indices in index-based
/// numbering.
/// \param [in] range Specifies the conversion scope.
/// \param [in] base Specifies the index base.
template <typename T>
inline void csr2csc(sycl::queue queue, int m, int n, int nnz, const T *from_val,
                    const int *from_row_ptr, const int *from_col_ind, T *to_val,
                    int *to_col_ptr, int *to_row_ind, conversion_scope range,
                    oneapi::mkl::index_base base) {
  detail::csr2csc_impl<T>()(queue, m, n, nnz, from_val, from_row_ptr,
                            from_col_ind, to_val, to_col_ptr, to_row_ind, range,
                            base);
}

/// Convert a CSR sparse matrix to a CSC sparse matrix.
/// \param [in] queue The queue where the routine should be executed. It must
/// have the in_order property when using the USM mode.
/// \param [in] m Number of rows of the matrix.
/// \param [in] n Number of columns of the matrix.
/// \param [in] nnz Number of non-zero elements.
/// \param [in] from_val An array containing the non-zero elements of the input
/// matrix.
/// \param [in] from_row_ptr An array of length \p m + 1.
/// \param [in] from_col_ind An array containing the column indices in
/// index-based numbering.
/// \param [out] to_val An array containing the non-zero elements of the output
/// matrix.
/// \param [out] to_col_ptr An array of length \p n + 1.
/// \param [out] to_row_ind An array containing the row indices in index-based
/// numbering.
/// \param [in] value_type Data type of \p from_val and \p to_val .
/// \param [in] range Specifies the conversion scope.
/// \param [in] base Specifies the index base.
inline void csr2csc(sycl::queue queue, int m, int n, int nnz,
                    const void *from_val, const int *from_row_ptr,
                    const int *from_col_ind, void *to_val, int *to_col_ptr,
                    int *to_row_ind, library_data_t value_type,
                    conversion_scope range, oneapi::mkl::index_base base) {
  detail::spblas_shim<detail::csr2csc_impl>(
      value_type, queue, m, n, nnz, from_val, from_row_ptr, from_col_ind,
      to_val, to_col_ptr, to_row_ind, range, base);
}

/// Calculate the non-zero elements number of the result of a
/// sparse matrix (CSR format)-sparse matrix (CSR format) product:
/// C = op(A) * op(B)
/// \param [in] desc The descriptor of this calculation.
/// \param [in] trans_a The operation applied to the matrix A.
/// \param [in] trans_b The operation applied to the matrix B.
/// \param [in] m The rows number of op(A) and C.
/// \param [in] n The columns number of op(B) and C.
/// \param [in] k The columns number of op(A) and rows number of op(B).
/// \param [in] info_a Matrix info of the matrix A.
/// \param [in] nnz_a Non-zero elements number of matrix A.
/// \param [in] val_a An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr_a An array of length row number + 1.
/// \param [in] col_ind_a An array containing the column indices in index-based
/// numbering.
/// \param [in] info_b Matrix info of the matrix B.
/// \param [in] nnz_b Non-zero elements number of matrix B.
/// \param [in] val_b An array containing the non-zero elements of the matrix B.
/// \param [in] row_ptr_b An array of length row number + 1.
/// \param [in] col_ind_b An array containing the column indices in index-based
/// numbering.
/// \param [in] info_c Matrix info of the matrix C.
/// \param [in] row_ptr_c An array of length row number + 1.
/// \param [out] nnz_host_ptr Non-zero elements number of matrix C.
template <typename T>
void csrgemm_nnz(descriptor_ptr desc, oneapi::mkl::transpose trans_a,
                 oneapi::mkl::transpose trans_b, int m, int n, int k,
                 const std::shared_ptr<matrix_info> info_a, int nnz_a,
                 const T *val_a, const int *row_ptr_a, const int *col_ind_a,
                 const std::shared_ptr<matrix_info> info_b, int nnz_b,
                 const T *val_b, const int *row_ptr_b, const int *col_ind_b,
                 const std::shared_ptr<matrix_info> info_c, int *row_ptr_c,
                 int *nnz_host_ptr) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  sycl::queue &queue = desc->get_queue();
  detail::csrgemm_args_info args(trans_a, trans_b, m, n, k, info_a, val_a,
                                 row_ptr_a, col_ind_a, info_b, val_b, row_ptr_b,
                                 col_ind_b, info_c, row_ptr_c);
  auto &info = desc->get_csrgemm_info_map()[args];

  if (info.is_empty()) {
    info.init(&queue);

    int rows_c = m;
    int cols_c = n;
    int rows_a = (trans_a == oneapi::mkl::transpose::nontrans) ? m : k;
    int cols_a = (trans_a == oneapi::mkl::transpose::nontrans) ? k : m;
    int rows_b = (trans_b == oneapi::mkl::transpose::nontrans) ? k : n;
    int cols_b = (trans_b == oneapi::mkl::transpose::nontrans) ? n : k;

    info.matrix_handle_a.set_matrix_data<Ty>(
        rows_a, cols_a, info_a->get_index_base(), row_ptr_a, col_ind_a, val_a);
    info.matrix_handle_b.set_matrix_data<Ty>(
        rows_b, cols_b, info_b->get_index_base(), row_ptr_b, col_ind_b, val_b);
    // In the future, oneMKL will allow nullptr to be passed in for row_ptr_c in
    // the initial calls before matmat. But currently, it needs an array of
    // length row_number + 1.
    info.matrix_handle_c.set_matrix_data<Ty>(
        rows_c, cols_c, info_c->get_index_base(), row_ptr_c, nullptr, nullptr);

    oneapi::mkl::sparse::set_matmat_data(
        info.matmat_desc.get_handle(),
        oneapi::mkl::sparse::matrix_view_descr::general, trans_a,
        oneapi::mkl::sparse::matrix_view_descr::general, trans_b,
        oneapi::mkl::sparse::matrix_view_descr::general);
  }

#ifdef DPCT_USM_LEVEL_NONE
#define __MATMAT(STEP, NNZ_C)                                                  \
  oneapi::mkl::sparse::matmat(queue, info.matrix_handle_a.get_handle(),        \
                              info.matrix_handle_b.get_handle(),               \
                              info.matrix_handle_c.get_handle(), STEP,         \
                              info.matmat_desc.get_handle(), NNZ_C, nullptr)
#else
#define __MATMAT(STEP, NNZ_C)                                                  \
  oneapi::mkl::sparse::matmat(                                                 \
      queue, info.matrix_handle_a.get_handle(),                                \
      info.matrix_handle_b.get_handle(), info.matrix_handle_c.get_handle(),    \
      STEP, info.matmat_desc.get_handle(), NNZ_C, nullptr, {})
#endif

  __MATMAT(oneapi::mkl::sparse::matmat_request::work_estimation, nullptr);
  queue.wait();

  __MATMAT(oneapi::mkl::sparse::matmat_request::compute, nullptr);

  int nnz_c_int = 0;
#ifdef DPCT_USM_LEVEL_NONE
  sycl::buffer<std::int64_t, 1> nnz_buf_c(1);
  __MATMAT(oneapi::mkl::sparse::matmat_request::get_nnz, &nnz_buf_c);
  nnz_c_int = nnz_buf_c.get_host_access(sycl::read_only)[0];
#else
  std::int64_t *nnz_c = sycl::malloc_host<std::int64_t>(1, queue);
  __MATMAT(oneapi::mkl::sparse::matmat_request::get_nnz, nnz_c);
  queue.wait();
  nnz_c_int = *nnz_c;
#endif

  if (nnz_host_ptr) {
    *nnz_host_ptr = nnz_c_int;
  }
}

/// Computes a sparse matrix (CSR format)-sparse matrix (CSR format) product:
/// C = op(A) * op(B)
/// \param [in] desc The descriptor of this calculation.
/// \param [in] trans_a The operation applied to the matrix A.
/// \param [in] trans_b The operation applied to the matrix B.
/// \param [in] m The rows number of op(A) and C.
/// \param [in] n The columns number of op(B) and C.
/// \param [in] k The columns number of op(A) and rows number of op(B).
/// \param [in] info_a Matrix info of the matrix A.
/// \param [in] val_a An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr_a An array of length row number + 1.
/// \param [in] col_ind_a An array containing the column indices in index-based
/// numbering.
/// \param [in] info_b Matrix info of the matrix B.
/// \param [in] val_b An array containing the non-zero elements of the matrix B.
/// \param [in] row_ptr_b An array of length row number + 1.
/// \param [in] col_ind_b An array containing the column indices in index-based
/// numbering.
/// \param [in] info_c Matrix info of the matrix C.
/// \param [out] val_c An array containing the non-zero elements of the matrix
/// C.
/// \param [in] row_ptr_c An array of length row number + 1.
/// \param [out] col_ind_c An array containing the column indices in index-based
/// numbering.
template <typename T>
void csrgemm(descriptor_ptr desc, oneapi::mkl::transpose trans_a,
             oneapi::mkl::transpose trans_b, int m, int n, int k,
             const std::shared_ptr<matrix_info> info_a, const T *val_a,
             const int *row_ptr_a, const int *col_ind_a,
             const std::shared_ptr<matrix_info> info_b, const T *val_b,
             const int *row_ptr_b, const int *col_ind_b,
             const std::shared_ptr<matrix_info> info_c, T *val_c,
             const int *row_ptr_c, int *col_ind_c) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  sycl::queue &queue = desc->get_queue();
  detail::csrgemm_args_info args(trans_a, trans_b, m, n, k, info_a, val_a,
                                 row_ptr_a, col_ind_a, info_b, val_b, row_ptr_b,
                                 col_ind_b, info_c, row_ptr_c);
  auto &info = desc->get_csrgemm_info_map()[args];
  if (info.is_empty()) {
    throw std::runtime_error("csrgemm_nnz is not invoked previously.");
  }

  int rows_c = m;
  int cols_c = n;
  info.matrix_handle_c.set_matrix_data<Ty>(
      rows_c, cols_c, info_c->get_index_base(), row_ptr_c, col_ind_c, val_c);
  sycl::event e;
#ifndef DPCT_USM_LEVEL_NONE
  e =
#endif
      __MATMAT(oneapi::mkl::sparse::matmat_request::finalize, nullptr);
#undef __MATMAT

  info.matrix_handle_a.add_dependency(e);
  info.matrix_handle_b.add_dependency(e);
  info.matrix_handle_c.add_dependency(e);
  info.matmat_desc.add_dependency(e);
  desc->get_csrgemm_info_map().erase(args);
}

/// Contains internal information for csrgemm2 by analyzing the provided
/// matrix structure and operation parameters.
class csrgemm2_info {
  detail::matrix_handle_manager matrix_handle_a;
  detail::matrix_handle_manager matrix_handle_b;
  detail::matrix_handle_manager matrix_handle_axb;
  detail::matrix_handle_manager matrix_handle_d;
  detail::matrix_handle_manager matrix_handle_c;
  detail::handle_manager<oneapi::mkl::sparse::matmat_descr_t> matmat_desc;
  detail::handle_manager<oneapi::mkl::sparse::omatadd_descr_t> omatadd_desc;
  void *row_ptr_axb = nullptr;
  void *col_ind_axb = nullptr;
  void *val_axb = nullptr;
#ifdef DPCT_USM_LEVEL_NONE
  sycl::buffer<std::int64_t, 1> *temp_buffer_1_size;
  sycl::buffer<std::int64_t, 1> *temp_buffer_2_size;
  sycl::buffer<std::uint8_t, 1> *temp_buffer_1;
  sycl::buffer<std::uint8_t, 1> *temp_buffer_2;
#else
  std::int64_t *temp_buffer_1_size;
  std::int64_t *temp_buffer_2_size;
  std::uint8_t *temp_buffer_1;
  std::uint8_t *temp_buffer_2;
#endif
  enum matrix_c_datatype_t {
    mcd_float,
    mcd_double,
    mcd_float2,
    mcd_double2,
  };
  template <typename T> void set_matrix_c_datatype() {
    if constexpr (std::is_same_v<T, float>) {
      matrix_c_datatype = matrix_c_datatype_t::mcd_float;
    } else if constexpr (std::is_same_v<T, double>) {
      matrix_c_datatype = matrix_c_datatype_t::mcd_double;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
      matrix_c_datatype = matrix_c_datatype_t::mcd_float2;
    } else {
      matrix_c_datatype = matrix_c_datatype_t::mcd_double2;
    }
  }
  matrix_c_datatype_t matrix_c_datatype;
  void init(sycl::queue *q_ptr) {
    matrix_handle_a.init(q_ptr);
    matrix_handle_b.init(q_ptr);
    matrix_handle_axb.init(q_ptr);
    matrix_handle_d.init(q_ptr);
    matrix_handle_c.init(q_ptr);
    matmat_desc.init(q_ptr);
    omatadd_desc.init(q_ptr);
  }
  sycl::event release(sycl::queue q, sycl::event dep) {
    matrix_handle_a.add_dependency(dep);
    matrix_handle_b.add_dependency(dep);
    matrix_handle_axb.add_dependency(dep);
    matrix_handle_d.add_dependency(dep);
    matrix_handle_c.add_dependency(dep);
    matmat_desc.add_dependency(dep);
    omatadd_desc.add_dependency(dep);
    std::vector<sycl::event> events;
    events.push_back(matrix_handle_a.release());
    events.push_back(matrix_handle_b.release());
    events.push_back(matrix_handle_axb.release());
    events.push_back(matrix_handle_d.release());
    events.push_back(matrix_handle_c.release());
    events.push_back(matmat_desc.release());
    events.push_back(omatadd_desc.release());
    return q.single_task(events, [] {});
  }
  template <typename T>
  friend void csrgemm2_get_buffer_size(
      descriptor_ptr desc, int m, int n, int k, const T *alpha,
      const std::shared_ptr<matrix_info> info_a, int nnz_a,
      const int *row_ptr_a, const int *col_ind_a,
      const std::shared_ptr<matrix_info> info_b, int nnz_b,
      const int *row_ptr_b, const int *col_ind_b, const T *beta,
      const std::shared_ptr<matrix_info> info_d, int nnz_d,
      const int *row_ptr_d, const int *col_ind_d,
      std::shared_ptr<csrgemm2_info> info, size_t *buffer_size_in_bytes);
  friend void csrgemm2_nnz(descriptor_ptr desc, int m, int n, int k,
                           const std::shared_ptr<matrix_info> info_a, int nnz_a,
                           const int *row_ptr_a, const int *col_ind_a,
                           const std::shared_ptr<matrix_info> info_b, int nnz_b,
                           const int *row_ptr_b, const int *col_ind_b,
                           const std::shared_ptr<matrix_info> info_d, int nnz_d,
                           const int *row_ptr_d, const int *col_ind_d,
                           const std::shared_ptr<matrix_info> info_c,
                           int *row_ptr_c, int *nnz_ptr,
                           std::shared_ptr<csrgemm2_info> info, void *buffer);
  template <typename T>
  friend void
  csrgemm2(descriptor_ptr desc, int m, int n, int k, const T *alpha,
           const std::shared_ptr<matrix_info> info_a, int nnz_a, const T *val_a,
           const int *row_ptr_a, const int *col_ind_a,
           const std::shared_ptr<matrix_info> info_b, int nnz_b, const T *val_b,
           const int *row_ptr_b, const int *col_ind_b, const T *beta,
           const std::shared_ptr<matrix_info> info_d, int nnz_d, const T *val_d,
           const int *row_ptr_d, const int *col_ind_d,
           const std::shared_ptr<matrix_info> info_c, T *val_c,
           const int *row_ptr_c, int *col_ind_c,
           std::shared_ptr<csrgemm2_info> info, void *buffer);
};

/// Calculate the required workspace size of the following operation:
/// C = alpha * A * B + beta * D
/// \param [in] desc The descriptor of this calculation.
/// \param [in] m The rows number of A, D and C.
/// \param [in] n The columns number of B, D and C.
/// \param [in] k The columns number of A and rows number of B.
/// \param [in] alpha Scaling factor.
/// \param [in] info_a Matrix info of the matrix A.
/// \param [in] nnz_a Non-zero elements number of matrix A.
/// \param [in] row_ptr_a An array of length \p m + 1.
/// \param [in] col_ind_a An array containing the column indices in index-based
/// numbering.
/// \param [in] info_b Matrix info of the matrix B.
/// \param [in] nnz_b Non-zero elements number of matrix B.
/// \param [in] row_ptr_b An array of length \p k + 1.
/// \param [in] col_ind_b An array containing the column indices in index-based
/// numbering.
/// \param [in] beta Scaling factor.
/// \param [in] info_d Matrix info of the matrix D.
/// \param [in] nnz_d Non-zero elements number of matrix D.
/// \param [in] row_ptr_d An array of length \p m + 1.
/// \param [in] col_ind_d An array containing the column indices in index-based
/// numbering.
/// \param [in, out] info The information of csrgemm2 operation.
/// \param [out] buffer_size_in_bytes Workspace memory size in bytes.
template <typename T>
void csrgemm2_get_buffer_size(
    descriptor_ptr desc, int m, int n, int k, const T *alpha,
    const std::shared_ptr<matrix_info> info_a, int nnz_a, const int *row_ptr_a,
    const int *col_ind_a, const std::shared_ptr<matrix_info> info_b, int nnz_b,
    const int *row_ptr_b, const int *col_ind_b, const T *beta,
    const std::shared_ptr<matrix_info> info_d, int nnz_d, const int *row_ptr_d,
    const int *col_ind_d, std::shared_ptr<csrgemm2_info> info,
    size_t *buffer_size_in_bytes) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  sycl::queue &queue = desc->get_queue();
  info->set_matrix_c_datatype<Ty>();

  info->row_ptr_axb = (int *)::dpct::cs::malloc((m + 1) * sizeof(int), queue);
  info->init(&queue);

  info->matrix_handle_a.set_matrix_data<Ty>(m, k, info_a->get_index_base(),
                                            row_ptr_a, col_ind_a, nullptr);
  info->matrix_handle_b.set_matrix_data<Ty>(k, n, info_b->get_index_base(),
                                            row_ptr_b, col_ind_b, nullptr);
  info->matrix_handle_axb.set_matrix_data<Ty>(
      m, n, oneapi::mkl::index_base::zero, info->row_ptr_axb, nullptr, nullptr);
  info->matrix_handle_d.set_matrix_data<Ty>(m, n, info_d->get_index_base(),
                                            row_ptr_d, col_ind_d, nullptr);
  info->matrix_handle_c.set_matrix_data<Ty>(m, n, oneapi::mkl::index_base::zero,
                                            nullptr, nullptr, nullptr);

  oneapi::mkl::sparse::set_matmat_data(
      info->matmat_desc.get_handle(),
      oneapi::mkl::sparse::matrix_view_descr::general,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::sparse::matrix_view_descr::general,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::sparse::matrix_view_descr::general);

#ifdef DPCT_USM_LEVEL_NONE
#define __MATMAT(STEP, PTR1, PTR2)                                             \
  oneapi::mkl::sparse::matmat(queue, info->matrix_handle_a.get_handle(),       \
                              info->matrix_handle_b.get_handle(),              \
                              info->matrix_handle_axb.get_handle(),            \
                              oneapi::mkl::sparse::matmat_request::STEP,       \
                              info->matmat_desc.get_handle(), PTR1, PTR2)
#else
#define __MATMAT(STEP, PTR1, PTR2)                                             \
  oneapi::mkl::sparse::matmat(queue, info->matrix_handle_a.get_handle(),       \
                              info->matrix_handle_b.get_handle(),              \
                              info->matrix_handle_axb.get_handle(),            \
                              oneapi::mkl::sparse::matmat_request::STEP,       \
                              info->matmat_desc.get_handle(), PTR1, PTR2, {})
#endif

  std::int64_t nnz_axb = 0;
#ifdef DPCT_USM_LEVEL_NONE
  info->temp_buffer_1_size = new sycl::buffer<std::int64_t, 1>(1);
  __MATMAT(get_work_estimation_buf_size, info->temp_buffer_1_size, nullptr);
  info->temp_buffer_1 = new sycl::buffer<std::uint8_t, 1>(
      info->temp_buffer_1_size->get_host_access(sycl::read_only)[0]);
  __MATMAT(work_estimation, info->temp_buffer_1_size, info->temp_buffer_1);
  info->temp_buffer_2_size = new sycl::buffer<std::int64_t, 1>(1);
  __MATMAT(get_compute_structure_buf_size, info->temp_buffer_2_size, nullptr);
  info->temp_buffer_2 = new sycl::buffer<std::uint8_t, 1>(
      info->temp_buffer_2_size->get_host_access(sycl::read_only)[0]);
  __MATMAT(compute_structure, info->temp_buffer_2_size, info->temp_buffer_2);
  sycl::buffer<std::int64_t, 1> nnz_axb_buf(1);
  __MATMAT(get_nnz, &nnz_axb_buf, nullptr);
  nnz_axb = nnz_axb_buf.get_host_access(sycl::read_only)[0];
#else
  info->temp_buffer_1_size = sycl::malloc_host<std::int64_t>(1, queue);
  __MATMAT(get_work_estimation_buf_size, info->temp_buffer_1_size, nullptr);
  queue.wait();
  info->temp_buffer_1 =
      sycl::malloc_device<std::uint8_t>(info->temp_buffer_1_size[0], queue);
  __MATMAT(work_estimation, info->temp_buffer_1_size, info->temp_buffer_1);
  info->temp_buffer_2_size = sycl::malloc_host<std::int64_t>(1, queue);
  __MATMAT(get_compute_structure_buf_size, info->temp_buffer_2_size, nullptr);
  queue.wait();
  info->temp_buffer_2 =
      sycl::malloc_device<std::uint8_t>(info->temp_buffer_2_size[0], queue);
  __MATMAT(compute_structure, info->temp_buffer_2_size, info->temp_buffer_2);
  std::int64_t *nnz_axb_ptr = sycl::malloc_host<std::int64_t>(1, queue);
  __MATMAT(get_nnz, nnz_axb_ptr, nullptr);
  queue.wait();
  nnz_axb = *nnz_axb_ptr;
  sycl::free(nnz_axb_ptr, queue);
#endif

  info->col_ind_axb = (int *)::dpct::cs::malloc(nnz_axb * sizeof(int), queue);
  info->val_axb = (Ty *)::dpct::cs::malloc(nnz_axb * sizeof(Ty), queue);
  info->matrix_handle_axb.set_matrix_data<Ty>(
      m, n, oneapi::mkl::index_base::zero, info->row_ptr_axb, info->col_ind_axb,
      info->val_axb);

  __MATMAT(finalize_structure, nullptr, nullptr);

  std::int64_t ws_size = 0;
  oneapi::mkl::sparse::omatadd_buffer_size(
      queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      info->matrix_handle_axb.get_handle(), info->matrix_handle_d.get_handle(),
      info->matrix_handle_c.get_handle(),
      oneapi::mkl::sparse::omatadd_alg::default_alg,
      info->omatadd_desc.get_handle(), ws_size);
  *buffer_size_in_bytes = ws_size;
}

/// Calculate the non-zero elements number of the matrix C in following operation:
/// C = alpha * A * B + beta * D
/// \param [in] desc The descriptor of this calculation.
/// \param [in] m The rows number of A, D and C.
/// \param [in] n The columns number of B, D and C.
/// \param [in] k The columns number of A and rows number of B.
/// \param [in] info_a Matrix info of the matrix A.
/// \param [in] nnz_a Non-zero elements number of matrix A.
/// \param [in] row_ptr_a An array of length \p m + 1.
/// \param [in] col_ind_a An array containing the column indices in index-based
/// numbering.
/// \param [in] info_b Matrix info of the matrix B.
/// \param [in] nnz_b Non-zero elements number of matrix B.
/// \param [in] row_ptr_b An array of length \p k + 1.
/// \param [in] col_ind_b An array containing the column indices in index-based
/// numbering.
/// \param [in] info_d Matrix info of the matrix D.
/// \param [in] nnz_d Non-zero elements number of matrix D.
/// \param [in] row_ptr_d An array of length \p m + 1.
/// \param [in] col_ind_d An array containing the column indices in index-based
/// numbering.
/// \param [in] info_c Matrix info of the matrix C.
/// \param [in] row_ptr_c An array of length \p m + 1.
/// \param [out] nnz_ptr Non-zero elements number of matrix C.
/// \param [in] info The information of csrgemm2 operation.
/// \param [in] buffer Workspace memory.
inline void csrgemm2_nnz(descriptor_ptr desc, int m, int n, int k,
                         const std::shared_ptr<matrix_info> info_a, int nnz_a,
                         const int *row_ptr_a, const int *col_ind_a,
                         const std::shared_ptr<matrix_info> info_b, int nnz_b,
                         const int *row_ptr_b, const int *col_ind_b,
                         const std::shared_ptr<matrix_info> info_d, int nnz_d,
                         const int *row_ptr_d, const int *col_ind_d,
                         const std::shared_ptr<matrix_info> info_c,
                         int *row_ptr_c, int *nnz_ptr,
                         std::shared_ptr<csrgemm2_info> info, void *buffer) {
  sycl::queue &queue = desc->get_queue();
  if (info->matrix_c_datatype ==
      csrgemm2_info::matrix_c_datatype_t::mcd_float) {
    info->matrix_handle_c.set_matrix_data<float>(m, n, info_c->get_index_base(),
                                                 nullptr, nullptr, nullptr);
  } else if (info->matrix_c_datatype ==
             csrgemm2_info::matrix_c_datatype_t::mcd_double) {
    info->matrix_handle_c.set_matrix_data<double>(
        m, n, info_c->get_index_base(), nullptr, nullptr, nullptr);
  } else if (info->matrix_c_datatype ==
             csrgemm2_info::matrix_c_datatype_t::mcd_float2) {
    info->matrix_handle_c.set_matrix_data<std::complex<float>>(
        m, n, info_c->get_index_base(), nullptr, nullptr, nullptr);
  } else {
    info->matrix_handle_c.set_matrix_data<std::complex<double>>(
        m, n, info_c->get_index_base(), nullptr, nullptr, nullptr);
  }
  auto data_buffer = dpct::detail::get_memory<std::uint8_t>(buffer);
  oneapi::mkl::sparse::omatadd_analyze(
      queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      info->matrix_handle_axb.get_handle(), info->matrix_handle_d.get_handle(),
      info->matrix_handle_c.get_handle(),
      oneapi::mkl::sparse::omatadd_alg::default_alg,
      info->omatadd_desc.get_handle(),
#ifdef DPCT_USM_LEVEL_NONE
      &data_buffer
#else
      data_buffer
#endif
  );
  std::int64_t nnz_c = 0;
  oneapi::mkl::sparse::omatadd_get_nnz(
      queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      info->matrix_handle_axb.get_handle(), info->matrix_handle_d.get_handle(),
      info->matrix_handle_c.get_handle(),
      oneapi::mkl::sparse::omatadd_alg::default_alg,
      info->omatadd_desc.get_handle(), nnz_c);

  assert((nnz_c >= INT_MIN && nnz_c <= INT_MAX) && "nnz_c is out of range.");
  int nnz_c_int = nnz_c;
  if (nnz_ptr)
    ::dpct::cs::memcpy(queue, nnz_ptr, &nnz_c_int, sizeof(int)).wait();
  int row_ptr_c_0 =
      (info_c->get_index_base() == oneapi::mkl::index_base::zero) ? 0 : 1;
  nnz_c_int += row_ptr_c_0;
  ::dpct::cs::memcpy(queue, row_ptr_c + m, &nnz_c_int, sizeof(int));
  ::dpct::cs::memcpy(queue, row_ptr_c, &row_ptr_c_0, sizeof(int)).wait();
}

/// Computes the matrix C in the following operation:
/// C = alpha * A * B + beta * D
/// \param [in] desc The descriptor of this calculation.
/// \param [in] m The rows number of A, D and C.
/// \param [in] n The columns number of B, D and C.
/// \param [in] k The columns number of A and rows number of B.
/// \param [in] alpha Scaling factor.
/// \param [in] info_a Matrix info of the matrix A.
/// \param [in] nnz_a Non-zero elements number of matrix A.
/// \param [in] val_a An array containing the non-zero elements of the matrix A.
/// \param [in] row_ptr_a An array of length \p m + 1.
/// \param [in] col_ind_a An array containing the column indices in index-based
/// numbering.
/// \param [in] info_b Matrix info of the matrix B.
/// \param [in] nnz_b Non-zero elements number of matrix B.
/// \param [in] val_b An array containing the non-zero elements of the matrix B.
/// \param [in] row_ptr_b An array of length \p k + 1.
/// \param [in] col_ind_b An array containing the column indices in index-based
/// numbering.
/// \param [in] beta Scaling factor.
/// \param [in] info_d Matrix info of the matrix D.
/// \param [in] nnz_d Non-zero elements number of matrix D.
/// \param [in] val_d An array containing the non-zero elements of the matrix D.
/// \param [in] row_ptr_d An array of length \p m + 1.
/// \param [in] col_ind_d An array containing the column indices in index-based
/// numbering.
/// \param [in] info_c Matrix info of the matrix C.
/// \param [out] val_c An array containing the non-zero elements of the matrix
/// C.
/// \param [in] row_ptr_c An array of length \p m + 1.
/// \param [out] col_ind_c An array containing the column indices in index-based
/// numbering.
/// \param [in] info The information of csrgemm2 operation.
/// \param [in] buffer Workspace memory.
template <typename T>
void csrgemm2(descriptor_ptr desc, int m, int n, int k, const T *alpha,
              const std::shared_ptr<matrix_info> info_a, int nnz_a,
              const T *val_a, const int *row_ptr_a, const int *col_ind_a,
              const std::shared_ptr<matrix_info> info_b, int nnz_b,
              const T *val_b, const int *row_ptr_b, const int *col_ind_b,
              const T *beta, const std::shared_ptr<matrix_info> info_d,
              int nnz_d, const T *val_d, const int *row_ptr_d,
              const int *col_ind_d, const std::shared_ptr<matrix_info> info_c,
              T *val_c, const int *row_ptr_c, int *col_ind_c,
              std::shared_ptr<csrgemm2_info> info, void *buffer) {
  using Ty = typename ::dpct::detail::lib_data_traits_t<T>;
  sycl::queue &queue = desc->get_queue();
  auto alpha_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(alpha), queue);
  auto beta_value =
      dpct::detail::get_value(reinterpret_cast<const Ty *>(beta), queue);

  info->matrix_handle_a.set_matrix_data<Ty>(m, k, info_a->get_index_base(),
                                            row_ptr_a, col_ind_a, val_a);
  info->matrix_handle_b.set_matrix_data<Ty>(k, n, info_b->get_index_base(),
                                            row_ptr_b, col_ind_b, val_b);
  __MATMAT(compute, nullptr, nullptr);
  __MATMAT(finalize, nullptr, nullptr);

  info->matrix_handle_d.set_matrix_data<Ty>(m, n, info_d->get_index_base(),
                                            row_ptr_d, col_ind_d, val_d);
  info->matrix_handle_c.set_matrix_data<Ty>(m, n, info_c->get_index_base(),
                                            row_ptr_c, col_ind_c, val_c);

  sycl::event e;
#ifndef DPCT_USM_LEVEL_NONE
  e =
#endif
      oneapi::mkl::sparse::omatadd(
          queue, oneapi::mkl::transpose::nontrans,
          oneapi::mkl::transpose::nontrans, alpha_value,
          info->matrix_handle_axb.get_handle(), beta_value,
          info->matrix_handle_d.get_handle(),
          info->matrix_handle_c.get_handle(),
          oneapi::mkl::sparse::omatadd_alg::default_alg,
          info->omatadd_desc.get_handle());

  std::vector<sycl::event> events;
  events.push_back(e);
  events.push_back(info->release(queue, e));
  queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.host_task([_p1 = info->row_ptr_axb, _p2 = info->col_ind_axb,
                   _p3 = info->val_axb, _p4 = info->temp_buffer_1_size,
                   _p5 = info->temp_buffer_1, _p6 = info->temp_buffer_2_size,
                   _p7 = info->temp_buffer_2, _q = queue] {
      ::dpct::cs::free(_p1, _q);
      ::dpct::cs::free(_p2, _q);
      ::dpct::cs::free(_p3, _q);
#ifdef DPCT_USM_LEVEL_NONE
      delete _p4;
      delete _p5;
      delete _p6;
      delete _p7;
#else
      sycl::free(_p4, _q);
      sycl::free(_p5, _q);
      sycl::free(_p6, _q);
      sycl::free(_p7, _q);
#endif
    });
  });
}

#endif
} // namespace dpct::sparse

#endif // __DPCT_SPARSE_UTILS_HPP__
