#ifndef CAFFE2_OPERATORS_RELU_OP_H_
#define CAFFE2_OPERATORS_RELU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "glog/logging.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class TanhOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(TanhOp);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice();

 protected:
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(TanhOp);
};

template <typename dtype, class DeviceContext>
class TanhGradientOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(TanhGradientOp);
  USE_OPERATOR_BASE_FUNCTIONS;

  bool RunOnDevice();

 protected:
  // Input: X, dY; Output: dX
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  DISABLE_COPY_AND_ASSIGN(TanhGradientOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_TANH_OP_H_
