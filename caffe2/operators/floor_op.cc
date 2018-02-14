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

#include "caffe2/operators/floor_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Floor, FloorOp<float, CPUContext>);

OPERATOR_SCHEMA(Floor)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor function, y = floor(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(Floor);

}  // namespace caffe2
