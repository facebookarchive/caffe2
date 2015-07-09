#include "caffe2/operators/tanh_op.h"

namespace caffe2 {

float tanh(float x) {
	if (x >= 0) {
		float enx = exp(-2.0*x);
		return (1 - enx)/(1 + enx);
	} else {
		float epx = exp(2.0*x);
		return (epx - 1)/(epx + 1);
	}
}

template <>
bool TanhOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  const float* Xdata = X.data();
  float* Ydata = Y->mutable_data();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = tanh(Xdata[i]);
  }
  return true;
}

template <>
bool TanhGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(Y.size(), 0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ReshapeLike(Y);
  const float* Ydata = Y.data();
  const float* dYdata = dY.data();
  float* dXdata = dX->mutable_data();
  for (int i = 0; i < dX.size(); ++i) {
    dXdata[i] = dYdata[i]*(1 - Ydata[i]*Ydata[i]);
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Tanh, TanhOp<float, CPUContext>)
REGISTER_CPU_OPERATOR(TanhGradient, TanhGradientOp<float, CPUContext>)
}  // namespace
}  // namespace caffe2
