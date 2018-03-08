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

#include "caffe2/operators/reduce_sum_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ReduceSum, ReduceSumOp<CPUContext>);

OPERATOR_SCHEMA(ReduceSum)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
      Computes the sum of the input tensor's element along the provided axes.
      The resulted tensor has the same rank as the input if keepdims equal 1.
      If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
    )DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension or not, default 1 sum keep reduced dimension.")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(ReduceSum);

} // namespace caffe2
