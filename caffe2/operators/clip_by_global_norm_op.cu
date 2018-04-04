/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *
 * Copyright (c) 2018, NVIDIA CORPORATION, All rights reserved
 * Distributed under 2-clause BSD license; see accompanying LICENSE file
 *
 **/

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/clip_by_global_norm_op.h"

namespace caffe2 {

namespace {
__global__ void ClipRatioKernel_ClipByGlobalNormOp(
    const float* sum,
    float* scaled_clip_ratio,
    float scale,
    float clip_norm) {

  if (threadIdx.x == 0) {
    auto global_norm = sqrt(*sum) * scale;
    auto clip_ratio = clip_norm / fmax(global_norm, clip_norm);
    *scaled_clip_ratio = clip_ratio * scale;
  }
}
}

template <>
bool ClipByGlobalNormOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if ((Input(0).IsType<float16>()) &&
        (this->output_type_ == "float16" || this->output_type_ == "input_type")) {
    return DoRunWithType<float16, float16>();
  } else if ((Input(0).IsType<float16>()) && (this->output_type_ == "float")) {
    return DoRunWithType<float16, float>();
  } else {
    LOG(FATAL) << "Unsupported input/output types";
  }
  return true;
}

template <>
void ClipByGlobalNormOp<CUDAContext>::ClipRatio(
    const float* sum, float* scaled_clip_ratio) {

  ClipRatioKernel_ClipByGlobalNormOp<<<1, 1, 0, context_.cuda_stream()>>>(
    sum, scaled_clip_ratio, scale_, clip_norm_);
}

REGISTER_CUDA_OPERATOR(ClipByGlobalNorm, ClipByGlobalNormOp<CUDAContext>);
}  // namespace caffe2
