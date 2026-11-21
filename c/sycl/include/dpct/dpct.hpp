//==---- dpct.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

// clang-format off
/*
There are 3 macros that can be used to control the behavior of the helper
function files.
+========================+=========================+===========================+
|                        | Helper function files   | Helper function files     |
| Macro Name             | Used in dpct migrated   | Used in direct            |
|                        | code                    | programming               |
+========================+=========================+===========================+
| DPCT_USM_LEVEL_NONE    | USM by default.         | USM by default.           |
| (Use USM or SYCL       | Use SYCL Buffer by      | Define it explicitly to   |
| buffer)                | running dpct with       | use the SYCL buffer.      |
|                        | "--use-level=none".     |                           |
+------------------------+-------------------------+---------------------------+
| DPCT_PROFILING_ENABLED | Enabled heuristically   | Disable by default.       |
| (Enable SYCL queue     | depends on the input    | Define it explicitly to   |
| profiling)             | code.                   | enable.                   |
|                        | Enable explicitly by    |                           |
|                        | running dpct with       |                           |
|                        | "--enable-profiling".   |                           |
|                        | Disable by removing the |                           |
|                        | macro definition        |                           |
|                        | manually.               |                           |
+------------------------+-------------------------+---------------------------+
| DPCT_HELPER_VERBOSE    | Disable by default.     | Disable by default.       |
| (Verbose option)       | Define it explicitly to | Define it explicitly to   |
|                        | enable.                 | enable.                   |
+========================+=========================+===========================+
*/
// clang-format on

#ifndef __DPCT_HPP__
#define __DPCT_HPP__

#include <sycl/sycl.hpp>
#include <iostream>
#include <limits.h>
#include <math.h>

template <class... Args> class dpct_kernel_name;
template <int Arg> class dpct_kernel_scalar;

#include "atomic.hpp"
#include "device.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "math.hpp"
#include "memory.hpp"
#include "util.hpp"

#include "bindless_images.hpp"
#include "graph.hpp"
#include "ze_utils.hpp"
#define USE_DPCT_HELPER 1

#if defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#else
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#else
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_COMPATIBILITY_TEMP (900)

namespace dpct {
enum error_code { success = 0, default_error = 999 };
/// A dummy function introduced to assist auto migration.
/// The migration tool user should replace it with a real error-handling function.
/// SYCL reports errors using exceptions and does not use error codes.
inline const char *get_error_string_dummy(int ec) {
  (void)ec;
  return "<FIXME: Placeholder>"; // Return the error string for the error code
                                 // ec.
}
} // namespace dpct

#define DPCT_CHECK_ERROR(expr)                                                 \
  [&]() {                                                                      \
    try {                                                                      \
      expr;                                                                    \
      return dpct::success;                                                    \
    } catch (std::exception const &e) {                                        \
      std::cerr << e.what() << std::endl;                                      \
      return dpct::default_error;                                              \
    }                                                                          \
  }()

#define DPCT_PI_F (3.14159274101257f)
#define DPCT_PI (3.141592653589793115998)

#endif // __DPCT_HPP__
