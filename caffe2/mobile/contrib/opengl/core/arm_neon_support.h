
#pragma once

#include "caffe2/core/common.h"

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#if !defined(__ARM_NEON__)
#define __ARM_NEON__ 1
#endif
#if CAFFE2_IOS
#include "arm_neon.h"
#elif CAFFE2_ANDROID
#include "caffe2/mobile/contrib/opengl/android/arm_neon_support.h"
#endif
#endif
