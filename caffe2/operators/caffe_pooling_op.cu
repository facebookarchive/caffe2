#include <cfloat>

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/pool_op.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

/***
  * Note: CUDA kernels are minor changes from those at:
  * https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu
  * Originally licensed under BSD
  **/
template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
  }
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    const int* const mask_slice = mask + offset;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += top_diff_slice[ph * pooled_width + pw];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
};

class CaffePoolOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  CaffePoolOp(const OperatorDef& operator_def, Workspace* ws)
    : ConvPoolOpBase<CUDAContext>(operator_def, ws) {
  }
  ~CaffePoolOp() {
  }

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override;

  // Input: X
  // Output: Y, mask
};

class CaffePoolGradientOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  CaffePoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
    : ConvPoolOpBase<CUDAContext>(operator_def, ws) {
  }
  ~CaffePoolGradientOp() {
  }

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override;

  // Input: X, dY, mask
  // Output: dX
};

template <typename T>
bool CaffePoolOp::DoRunWithType() {
	auto& X = Input(0);
  auto* Y = Output(0);
  auto* mask = Output(1);

  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, X.dim32(1));
  int output_size = Y->size();
  mask->Resize(output_size);

  MaxPoolForward<T><<<CAFFE_GET_BLOCKS(output_size),
                      CAFFE_CUDA_NUM_THREADS,
                      0,
                      context_.cuda_stream()>>>(
      output_size, X.data<T>(), X.dim32(0), X.dim32(1), X.dim32(2), X.dim32(3),
      Y->dim32(2), Y->dim32(3), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, Y->mutable_data<T>(), mask->mutable_data<int>());
  return true;
      
}

bool CaffePoolOp::RunOnDevice() {
  auto& X = Input(0);

  if (X.IsType<float>()) {
    return DoRunWithType<float>();
  } else if (X.IsType<float16>()) {
    //return DoRunWithType<float16>();
    return false;
  } else {
    return false;
  }
}

template <typename T>
bool CaffePoolGradientOp::DoRunWithType() {
  auto& X  = Input(0);
  auto& dY = Input(1);
  auto& mask = Input(2);
  auto* dX = Output(0);

  dX->ResizeLike(X);
  ConvPoolOpBase<CUDAContext>::ComputePads(X.dim32(2), X.dim32(3));

  MaxPoolBackward<T><<<CAFFE_GET_BLOCKS(X.size()),
                       CAFFE_CUDA_NUM_THREADS,
                       0,
                       context_.cuda_stream()>>>(
      X.size(), dY.data<T>(), mask.data<int>(), X.dim32(0), X.dim32(1), X.dim32(2), X.dim32(3),
      dY.dim32(2), dY.dim32(3), kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_t_, pad_l_, dX->template mutable_data<T>());
  return true;
}

bool CaffePoolGradientOp::RunOnDevice() {
  auto& X = Input(0);

  if (X.IsType<float>()) {
    return DoRunWithType<float>();
  } else if (X.IsType<float16>()) {
    // return DoRunWithType<float16>();
    return false;
  } else {
    return false;
  }
}

namespace {
REGISTER_CUDA_OPERATOR(CaffeMaxPool, CaffePoolOp);
REGISTER_CUDA_OPERATOR(CaffeMaxPoolGradient, CaffePoolGradientOp);

class GetCaffePoolGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CaffeMaxPoolGradient",
        "",
        vector<string>{I(0), GO(0), O(1)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(CaffeMaxPool, GetCaffePoolGradient);

};

}; // namespace caffe2
