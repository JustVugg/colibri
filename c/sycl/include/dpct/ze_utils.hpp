//==---- ze_utils.hpp ---------------------------------*- C++
//-*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_ZE_UTILS_HPP__
#define __DPCT_ZE_UTILS_HPP__

#ifdef ONEAPI_BACKEND_LEVEL_ZERO_EXT
#if defined(__linux__)
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
namespace dpct {
namespace experimental {

struct ipc_mem_handle_ext_t {
  pid_t pid;
  ze_ipc_mem_handle_t handle;
};

namespace detail {

#ifndef _SYS_pidfd_open
#define _SYS_pidfd_open 434 // syscall number for pidfd_open
#endif

#ifndef _SYS_pidfd_getfd
#define _SYS_pidfd_getfd 438 // syscall number for pidfd_getfd
#endif

inline int get_fd_of_peer_process(ipc_mem_handle_ext_t ext_handle) {
  int pidfd = syscall(_SYS_pidfd_open, ext_handle.pid,
                      0); // obtain a file descriptor that refers to a
                          // process(requires kernel 5.6+).
  if (pidfd < 0)
    return -1;
  return syscall(_SYS_pidfd_getfd, pidfd, *(int *)ext_handle.handle.data,
                 0); // obtain a duplicate of another process's file
                     // descriptor(requires kernel 5.6+).
}

} // namespace detail

/// Creates an IPC memory handle for the specified allocation.
/// \param [in] ptr Pointer to the device memory allocation
/// \param [out] ext_handle_ptr IPC memory handle extension
inline ze_result_t get_mem_ipc_handle(const void *ptr,
                                      ipc_mem_handle_ext_t *ext_handle_ptr) {
  ext_handle_ptr->pid = getpid();
  return zeMemGetIpcHandle(
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          dpct::get_current_device().get_context()),
      ptr, &ext_handle_ptr->handle);
}

/// Opens an IPC memory handle to retrieve a device pointer.
/// \param [in] ext_handle IPC memory handle extension
/// \param [out] pptr Pointer to device allocation in this process
inline ze_result_t open_mem_ipc_handle(ipc_mem_handle_ext_t ext_handle,
                                       void **pptr) {
  int fd = detail::get_fd_of_peer_process(ext_handle);
  if (fd < 0)
    throw std::runtime_error("Cannot get file descriptor of peer process.");
  *((int *)ext_handle.handle.data) = fd;

  return zeMemOpenIpcHandle(
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          dpct::get_current_device().get_context()),
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          (sycl::device)dpct::get_current_device()),
      ext_handle.handle, 0u, pptr);
}

} // namespace experimental
} // namespace dpct

#endif // __linux__
#endif // ONEAPI_BACKEND_LEVEL_ZERO_EXT

#endif // ! __DPCT_ZE_UTILS_HPP__
