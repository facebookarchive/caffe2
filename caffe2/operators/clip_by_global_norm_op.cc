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

#include "caffe2/operators/clip_by_global_norm_op.h"

namespace caffe2 {

template <>
bool ClipByGlobalNormOp<CPUContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float, float>();
  } else {
    LOG(FATAL) << "Unsupported input type";
  }
  return true;
}

template <>
void ClipByGlobalNormOp<CPUContext>::ClipRatio(
    const float* sum, float* scaled_clip_ratio) {

  auto global_norm = std::sqrt(*sum) * scale_;
  auto clip_ratio = clip_norm_ / std::max(global_norm, clip_norm_);
  *scaled_clip_ratio = clip_ratio * scale_;
}

REGISTER_CPU_OPERATOR(ClipByGlobalNorm, ClipByGlobalNormOp<CPUContext>);

OPERATOR_SCHEMA(ClipByGlobalNorm)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .AllowOneToOneInplace()
    .SetDoc(R"DOC(
ClipByGlobalNorm operator clips every element in input tensors so that
the global norm of all the input tensors becomes at most clip_norm.
It also multiplies every input element by scale before calculating
the global norm or clipping the values.
)DOC")
    .Arg("clip_norm", "Upper bound on the global norm of input tensors")
    .Arg("scale", "Scalar used for multiplication")
    .Arg("output_type", "Datatype of output tensors."
          "Options are float, float16, input_type."
          "The latter makes the output datatype the same as inputs'")
    .Input(
        0,
        "inputs",
        "Input tensors")
    .Output(
        0,
        "outputs",
        "Output tensors");

}  // namespace caffe2
