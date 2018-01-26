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

#ifndef CAFFE2_OPERATORS_CLIP_BY_GLOBAL_NORM_OP_H_
#define CAFFE2_OPERATORS_CLIP_BY_GLOBAL_NORM_OP_H_

#include <limits>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ClipByGlobalNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ClipByGlobalNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    clip_norm_ = OperatorBase::GetSingleArgument<float>("clip_norm", -1.0);
    CAFFE_ENFORCE_GE(clip_norm_, 0);
    scale_ = OperatorBase::GetSingleArgument<float>("scale", 1.0);
    output_type_ = OperatorBase::GetSingleArgument<string>("output_type", "input_type");
    CAFFE_ENFORCE(
        (output_type_ == "float") ||
        (output_type_ == "float16") ||
        (output_type_ == "input_type"));

    CAFFE_ENFORCE_GT(InputSize(), 0);
    CAFFE_ENFORCE_GT(OutputSize(), 0);
    CAFFE_ENFORCE_EQ(InputSize(), OutputSize());
  }

  bool RunOnDevice() override;

  template<typename IN_TYPE, typename OUT_TYPE>
  bool DoRunWithType() {
    for (auto i = 1; i < InputSize(); i++) {
      CAFFE_ENFORCE(Input(i).template IsType<IN_TYPE>());
    }
    // Store the SumSqr of each tensor in acc
    Tensor<Context> acc;
    acc.Resize(InputSize());
    auto acc_data = acc.template mutable_data<float>();
    for (auto i = 0; i < InputSize(); ++i) {
      auto& X = Input(i);
      math::SumSqr<IN_TYPE, Context, float>(
        X.size(),
        X.template data<IN_TYPE>(),
        &acc_data[i],
        &context_,
        &scratch_);
    }

    Tensor<Context> sum;
    sum.Resize(1);
    auto sum_data = sum.template mutable_data<float>();
    math::Sum<float, Context>(
        InputSize(), acc_data, sum_data, &context_, &scratch_);
    Tensor<Context> scaled_clip_ratio;
    scaled_clip_ratio.Resize(1);
    ClipRatio(sum_data, scaled_clip_ratio.template mutable_data<float>());

    for (auto i = 0; i < InputSize(); ++i) {
      auto& X = Input(i);
      auto* Y = Output(i);
      Y->ResizeLike(X);
      math::Scale<IN_TYPE, Context, OUT_TYPE>(
        X.size(),
        scaled_clip_ratio.template data<float>(),
        X.template data<IN_TYPE>(),
        Y->template mutable_data<OUT_TYPE>(),
        &context_);
    }

    return true;
  }

 private:
  void ClipRatio(const float* sum, float* scaled_clip_ratio);

 protected:
  float clip_norm_;
  float scale_;
  string output_type_;
  Tensor<Context> scratch_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CLIP_BY_GLOBAL_NORM_OP_H_
