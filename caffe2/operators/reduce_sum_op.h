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

#ifndef CAFFE2_OPERATORS_REDUCE_SUM_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_SUM_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

template <class Context>
class ReduceSumOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceSumOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    axes_ = OperatorBase::GetRepeatedArgument<int>("axes");
    keepdims_ = OperatorBase::GetSingleArgument<int>("keepdims", 1);
  }

  template <typename T>
  bool DoRunWithType() {
    auto& input = Input(0);

    vector<TIndex> dims = input.dims();
    int ndim = input.ndim();

    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      std::sort(axes_.begin(), axes_.end());
      CAFFE_ENFORCE(axes_.front() >= 0, "Axes ids must be non-negative.");
      CAFFE_ENFORCE(
          axes_.back() < ndim,
          "Axes ids must be smaller than the dimensions of input.");
    }

    auto* output = Output(0);
    output->ResizeLike(input);
    output->CopyFrom(input, &context_);
    buffer_.ResizeLike(input);

    for (int id = axes_.size() - 1; id >= 0; id--) {
      int reduced_axis = axes_[id];
      int reduced_dim = dims[reduced_axis];
      int front = size_to_dim_(reduced_axis, dims);
      int back = size_from_dim_(reduced_axis + 1, dims);

      T* output_data = output->template mutable_data<T>();
      T* temp_buf = buffer_.template mutable_data<T>();

      for (int i = 0; i < front; i++) {
        for (int k = 0; k < back; k++) {
          T sum = 0;
          for (int j = 0; j < reduced_dim; j++) {
            sum += output_data[i * reduced_dim * back + j * back + k];
          }
          temp_buf[i * back + k] = sum;
        }
      }

      output->swap(buffer_);

      if (keepdims_) {
        dims[reduced_axis] = 1;
      } else {
        dims.erase(dims.begin() + reduced_axis);
      }
    }

    output->Resize(dims);

    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      CAFFE_THROW(
          "ReduceSum operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).meta().name());
    }
  }

 private:
  std::vector<int> axes_;
  int keepdims_;
  Tensor<Context> buffer_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_SUM_OPS_H_
