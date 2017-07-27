#pragma once

#if defined(__APPLE__) && (__clang_major__ < 8)

#include <emmintrin.h>

// This version of apple clang has a bug that _cvtsh_ss is not defined, see
// https://reviews.llvm.org/D16177
static __inline float
    __attribute__((__always_inline__, __nodebug__, __target__("f16c")))
_cvtsh_ss(unsigned short a)
{
  __v8hi v = {(short)a, 0, 0, 0, 0, 0, 0, 0};
  __v4sf r = __builtin_ia32_vcvtph2ps(v);
  return r[0];
}

#endif
