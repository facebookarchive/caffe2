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

#include "caffe2/operators/reduce_ops.h"

namespace caffe2 {

vector<TIndex> ConvertFromInputIndex(TIndex index, vector<TIndex>& dims) {
  TIndex ndim = dims.size();
  vector<TIndex> nd_idx(ndim);

  for (TIndex i = ndim - 1; i >= 0 && index > 0; i--) {
    nd_idx[i] = index % dims[i];
    index /= dims[i];
  }
  return nd_idx;
}

TIndex ConvertToOutputIndex(
    const vector<int>& axes,
    const vector<TIndex>& nd_idx,
    vector<TIndex>& dims) {
  TIndex index = 0;
  TIndex mul = 1;

  for (TIndex i = dims.size() - 1, j = axes.size() - 1; i >= 0; i--) {
    if (j >= 0 && axes[j] == i) {
      j--;
    } else {
      index += nd_idx[i] * mul;
      mul *= dims[i];
    }
  }
  return index;
}

template <typename T, class Context>
void ComputeSum(
    const T* X_data,
    const TIndex X_size,
    vector<TIndex>& dims,
    T* Y_data,
    vector<int>& axes,
    int keepdims) {
  for (TIndex X_idx = 0; X_idx < X_size; X_idx++) {
    vector<TIndex> nd_idx = ConvertFromInputIndex(X_idx, dims);
    TIndex Y_idx = ConvertToOutputIndex(axes, nd_idx, dims);
    Y_data[Y_idx] += X_data[X_idx];
  }

  for (TIndex id = axes.size() - 1; id >= 0; id--) {
    TIndex reduced_axis = axes[id];
    if (keepdims) {
      dims[reduced_axis] = 1;
    } else {
      dims.erase(dims.begin() + reduced_axis);
    }
  }
}

template <typename T, class Context>
bool ReduceSumOp<T, Context>::Compute(
    const T* X_data,
    const TIndex X_size,
    vector<TIndex>& dims,
    T* Y_data,
    vector<int>& axes,
    int keepdims) {
  math::Set<T, Context>(X_size, 0.f, Y_data, &context_);
  ComputeSum<T, Context>(X_data, X_size, dims, Y_data, axes, keepdims);
  Output(0)->Resize(dims);
  return true;
}

template <typename T, class Context>
bool ReduceMeanOp<T, Context>::Compute(
    const T* X_data,
    const TIndex X_size,
    vector<TIndex>& dims,
    T* Y_data,
    vector<int>& axes,
    int keepdims) {
  math::Set<T, Context>(X_size, 0.f, Y_data, &context_);
  ComputeSum<T, Context>(X_data, X_size, dims, Y_data, axes, keepdims);
  Output(0)->Resize(dims);
  TIndex Y_size = Output(0)->size();
  math::Scale(
      Y_size, static_cast<T>(Y_size) / X_size, Y_data, Y_data, &context_);

  return true;
}

REGISTER_CPU_OPERATOR(ReduceSum, ReduceSumOp<float, CPUContext>);

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
        "Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(ReduceSum);

REGISTER_CPU_OPERATOR(ReduceMean, ReduceMeanOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceMean)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
      Computes the mean of the input tensor's element along the provided axes.
      The resulted tensor has the same rank as the input if keepdims equal 1.
      If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
    )DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(ReduceMean);

} // namespace caffe2
