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

#ifndef CAFFE2_OPERATORS_MINMAX_OPS_H_
#define CAFFE2_OPERATORS_MINMAX_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

template <class Context>
class MeanOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MeanOp)

  template <typename T>
  bool DoRunWithType() {
    auto& input0 = Input(0);
    auto* output = Output(0);

    output->ResizeLike(input0);
    output->CopyFrom(input0, &context_);

    if (InputSize() == 1) {
      return true;
    }

    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      if (output->dims() != Input(i).dims()) {
        CAFFE_THROW(
            "Check failed: output->dims() == Input(i).dims().",
            "Description: Input #",
            i,
            ", input dimension:",
            Input(i).dims(),
            " should match output dimension: ",
            output->dims());
      }
    }

    T* output_data = output->template mutable_data<T>();
    for (int i = 1; i < InputSize(); ++i) {
      math::Add(
          output->size(),
          output_data,
          Input(i).template data<T>(),
          output_data,
          &context_);
    }

    math::Scale(
        output->size(),
        1.0f / InputSize(),
        output_data,
        output_data,
        &context_);

    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      CAFFE_THROW(
          "Mean operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).meta().name());
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MINMAX_OPS_H_
