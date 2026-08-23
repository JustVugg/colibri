// glibc_c23_math_compat.h — glibc >= 2.41 C++17 noexcept clash for CUDA host passes.
//
// glibc >= 2.41 (Debian trixie, Ubuntu 26.04, ...) declares the C23 math
// additions (rsqrt/rsqrtf, sinpi/cospi, ...) via __THROW, which expands to
// noexcept(true) in C++. CUDA's crt/math_functions.h declares the same
// functions WITHOUT an exception specification, and since C++17 noexcept is
// part of the function type, nvcc's host pass fails with:
//
//   mathcalls.h: error: exception specification is incompatible with that of
//   previous function "rsqrt" (declared at ... crt/math_functions.h)
//
// This header is force-included into the HOST pass (NVCCFLAGS
// -Xcompiler=-include, ... on Linux). It strips glibc's __THROW so the C
// declarations carry no exception specification, matching CUDA's. The device
// pass is unaffected: -Xcompiler flags never reach nvcc's device frontend,
// and it includes CUDA's math_functions.h directly.
//
// Deliberately Linux-only: Windows builds use MSVC cl.exe, which never sees
// glibc headers.

#ifndef COLI_GLIBC_C23_MATH_COMPAT_H
#define COLI_GLIBC_C23_MATH_COMPAT_H

#if defined(__cplusplus) && defined(__linux__)
#include <features.h> /* defines __GLIBC__ on glibc; musl ships one too */
#if defined(__GLIBC__) && defined(__THROW)
#undef __THROW
#define __THROW
#endif
#endif /* __cplusplus && __linux__ */

#endif /* COLI_GLIBC_C23_MATH_COMPAT_H */
