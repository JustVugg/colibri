//==--------------- shmem_utils.hpp ---------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_SHMEM_UTILS_HPP__
#define __DPCT_SHMEM_UTILS_HPP__

namespace dpct::shmemx {

enum flags { RUNTIME_MPI = 1 << 1, RUNTIME_OPENSHMEM = 1 << 2 };

/// Initialize Intel SHMEM library based on exisiting communicator
/// \param [in] runtime_flags specify the underlying communicator like MPI/
/// OpenSHMEM to initialize iSHMEM for launcher agnostic bootstrapping
/// \param [in] attr Additional attributes for initializing the iSHMEM library.
void init_attr(unsigned runtime_flags, ishmemx_attr_t *attr) {
  if (runtime_flags == 0) {
    // if no runtime flags are present, initialize iSHMEM normally
    ishmem_init();
  } else {
    unsigned ishmem_runtime_flags = 0;

    if (runtime_flags & RUNTIME_MPI)
      ishmem_runtime_flags |= ISHMEMX_RUNTIME_MPI;
    if (runtime_flags & RUNTIME_OPENSHMEM)
      ishmem_runtime_flags |= ISHMEMX_RUNTIME_OPENSHMEM;

    attr->runtime = static_cast<ishmemx_runtime_type_t>(ishmem_runtime_flags);
    ishmemx_init_attr(attr);
  }
}

/// Update signal address with signal using signal operation
/// \param [out] sig_addr Symmetric address of the signal data object to be
/// updated on the remote PE.
/// \param [in] signal Unsigned 64-bit value that is used for updating the
/// remote sig_addr signal data object.
/// \param [in] sig_op Operator used to update signal data object
/// \param [in] pe Processing element
void signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  if (sig_op == ISHMEM_SIGNAL_SET) {
    ishmemx_signal_set(sig_addr, signal, pe);
  } else if (sig_op == ISHMEM_SIGNAL_ADD) {
    ishmemx_signal_add(sig_addr, signal, pe);
  } else {
    throw std::runtime_error("Unsupported signal operator!");
  }
}

} // namespace dpct::shmemx

#endif // __DPCT_SHMEM_UTILS_HPP__
