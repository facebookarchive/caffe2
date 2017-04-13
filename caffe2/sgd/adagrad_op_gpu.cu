#include "adagrad_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

__global__ void AdagradUpdate(
    int N,
    const float* g,
    const float* h,
    float* ng,
    float* nh,
    float epsilon,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = nh[i] = h[i] + gi * gi;
    ng[i] = lr[0] * gi / (std::sqrt(hi) + epsilon);
  }
}

template <>
void adagrad_update<CUDAContext>(
    int N,
    const float* g,
    const float* h,
    float* ng,
    float* nh,
    float epsilon,
    const float* lr,
    CUDAContext* context) {
  AdagradUpdate<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, g, h, ng, nh, epsilon, lr);
}

__global__ void AdagradCompute(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = nh[i] = h[i] + gi * gi;
    nw[i] = w[i] + lr[0] * gi / (std::sqrt(hi) + epsilon);
  }
}

template <>
void adagrad_compute<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    const float* lr,
    CUDAContext* context) {
  AdagradCompute<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, w, g, h, nw, nh, epsilon, lr);
}

namespace {
REGISTER_CUDA_OPERATOR(Adagrad, AdagradOp<float, CUDAContext>);
}
}
