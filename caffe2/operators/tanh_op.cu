#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/tanh_op.h"

namespace caffe2 {
namespace {
__device__ float tanh(float x) {
  if (x >= 0) {
    float enx = exp(-2.0*x);
    return (1 - enx)/(1 + enx);
  } else {
    float epx = exp(2.0*x);
    return (epx - 1)/(epx + 1);
  }
}

template <typename dtype>
__global__ void TanhKernel(const int N, const dtype* X, dtype* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = tanh(X[i]);
  }
}

template <typename dtype>
__global__ void TanhGradientKernel(const int N, const dtype* Y, const dtype* dY,
                              dtype* dX) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i]*(1 - Y[i]*Y[i]);
  }
}
}  // namespace

template <>
bool TanhOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  TanhKernel<<<CAFFE_GET_BLOCKS(X.size()), CAFFE_CUDA_NUM_THREADS,
               0, device_context_.cuda_stream()>>>(
      X.size(), X.data(), Y->mutable_data());
  return true;
}

template <>
bool TanhGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(Y.size(), 0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ReshapeLike(Y);
  TanhGradientKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS,
                       0, device_context_.cuda_stream()>>>(
      Y.size(), Y.data(), dY.data(), dX->mutable_data());
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(Tanh, TanhOp<float, CUDAContext>)
REGISTER_CUDA_OPERATOR(TanhGradient, TanhGradientOp<float, CUDAContext>)
}  // namespace
}  // namespace caffe2
