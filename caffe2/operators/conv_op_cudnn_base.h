#pragma once

#include <sstream>

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Earlier in the days Caffe sets the default cudnn workspace to 8MB. We bump
// it up to 64MB in Caffe2, as this enables the use of Winograd in many cases,
// something very beneficial to more recent CNN models.
static constexpr size_t kCONV_CUDNN_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024;

namespace {
template <typename ArrayOfcudnnConvolutionAlgoPerf_t>
inline void LogCuDNNPerfStats(
    const ArrayOfcudnnConvolutionAlgoPerf_t& perf_stat,
    int returned_algo_count) {
  VLOG(1) << "Perf result: (algo: stat, time, memory)";
  for (int i = 0; i < returned_algo_count; ++i) {
    const auto& stat = perf_stat[i];
    VLOG(1) << stat.algo << ": " << stat.status << " " << stat.time << " "
            << stat.memory;
  }
}

// Easier indexing into force_algo_ vector
enum {
  ALGO_FWD = 0,
  ALGO_WGRAD = 1,
  ALGO_DGRAD = 2
} algoIndex_t;

}  // namespace

class CudnnConvOpBase : public ConvPoolOpBase<CUDAContext> {
 public:
  CudnnConvOpBase(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        cudnn_ws_nbytes_limit_(OperatorBase::GetSingleArgument<size_t>(
            "ws_nbytes_limit",
            kCONV_CUDNN_WORKSPACE_LIMIT_BYTES)),
        exhaustive_search_(
            OperatorBase::GetSingleArgument<int>("exhaustive_search", 0)),
        deterministic_(
            OperatorBase::GetSingleArgument<int>("deterministic", 0)),
        cudnn_state_(OperatorBase::GetSingleArgument<int>("cudnn_state", 0)),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)),
        force_algo_(OperatorBase::GetRepeatedArgument<int>("force_algo", vector<int>{-1,-1,-1})) {
    CHECK(!deterministic_ || !exhaustive_search_);
    CAFFE_ENFORCE(group_ > 0);
    CAFFE_ENFORCE(!deterministic_ || !exhaustive_search_);
    for (int i = 0; i < kernel_.size(); ++i) {
      OPERATOR_NEEDS_FEATURE(
          pads_[i] == pads_[kernel_.size() + i],
          "The current padding scheme leads to unequal padding on the left "
          "and right, which is not supported by cudnn.");
    }
    // dilated convolution supported by some algorithms in cuDNN v6
#if !(CUDNN_VERSION_MIN(6,0,0))
    OPERATOR_NEEDS_FEATURE(
        dilation_h() == 1 && dilation_w() == 1,
        "The cudnn convolution does not support dilation yet.");
#endif

    bool individual_force_algo = OperatorBase::HasArgument("force_algo_fwd") ||
                                 OperatorBase::HasArgument("force_algo_dgrad") ||
                                 OperatorBase::HasArgument("force_algo_wgrad");
    if (OperatorBase::HasArgument("force_algo")) {
      CAFFE_ENFORCE(!individual_force_algo,
                   "Cannot specify both force_algo and any of",
                   "force_algo_fwd, force_algo_dgrad, force_algo_wgrad");
    } else {
      force_algo_ = std::vector<int>{-1,-1,-1};
      force_algo_[ALGO_FWD] =
          OperatorBase::GetSingleArgument<int>("force_algo_fwd", -1);
      force_algo_[ALGO_DGRAD] =
          OperatorBase::GetSingleArgument<int>("force_algo_dgrad", -1);
      force_algo_[ALGO_WGRAD] =
          OperatorBase::GetSingleArgument<int>("force_algo_wgrad", -1);
    }

    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
    CUDNN_ENFORCE(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CudnnConvOpBase() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
    CUDNN_ENFORCE(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

 protected:

  // wrapper to set up convolution descriptors
  template <typename MATH>
  void SetConvolutionDescriptor(
      int size,
      cudnnConvolutionDescriptor_t conv_desc_) {
#if CUDNN_VERSION_MIN(6,0,0)
    if (size == 2) {
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    } else {
      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          dilation_.data(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    }
#else
    if (size == 2) {
      CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(
          conv_desc_,
          pad_t(),
          pad_l(),
          stride_h(),
          stride_w(),
          1,
          1,
          CUDNN_CROSS_CORRELATION));
    } else {
      vector<int> ones(dilation_.size(), 1);
      CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(
          conv_desc_,
          kernel_.size(),
          pads_.data(),
          stride_.data(),
          ones.data(),
          CUDNN_CROSS_CORRELATION,
          cudnnTypeWrapper<MATH>::type));
    }
#endif
  }

  // Helper to extract dimension data from a tensor
  struct DimensionParam {
    int N, C, H, W, D;
  };

  DimensionParam GetDimensionsFromTensorDims(
      const int ndim,
      const vector<TIndex>& dims,
      StorageOrder order) {

    DimensionParam param;

    switch (order) {
      case StorageOrder::NHWC:
        param.N = dims.at(0);
        param.H = dims.at(1);
        param.W = ndim > 3 ? dims.at(2) : 1;
        param.D = ndim > 4 ? dims.at(3) : 1;
        param.C = dims.at(ndim - 1);
        break;
      case StorageOrder::NCHW:
        param.N = dims.at(0);
        param.C = dims.at(1);
        param.H = dims.at(2);
        param.W = ndim > 3 ? dims.at(3) : 1;
        param.D = ndim > 4 ? dims.at(4) : 1;
        break;
      default:
        LOG(FATAL) << "Unknown storage order: " << order;
    }

    return param;
  }

  // A helper function to set up the tensor Nd desriptor, depending on the order
  // the group and the type given.
  template <typename T>
  void SetTensorNdDescriptorWithGroup(
      int size,
      cudnnTensorDescriptor_t desc_,
      int N,
      int C,
      int H,
      int W,
      int D) {
    switch (order_) {
      case StorageOrder::NHWC:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              desc_,
              cudnnTypeWrapper<T>::type,
              N,
              C / group_,
              H,
              W,
              H * W * C,
              1,
              W * C,
              C));
        } else {
          C = C / group_;
          vector<int> dims = {N, H, W, D, C};
          vector<int> strides = {H * W * D * C, W * D * C, D * C, C, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              desc_,
              cudnnTypeWrapper<T>::type,
              size > 3 ? size : 4,
              dims.data(),
              strides.data()));
        }
        break;
      case StorageOrder::NCHW:
        if (size == 4) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(
              desc_,
              cudnnTypeWrapper<T>::type,
              N,
              C / group_,
              H,
              W,
              C * H * W,
              H * W,
              W,
              1));
        } else {
          C = C / group_;
          vector<int> dims = {N, C, H, W, D};
          vector<int> strides = {C * H * W * D, H * W * D, W * D, D, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              desc_,
              cudnnTypeWrapper<T>::type,
              size > 3 ? size : 4,
              dims.data(),
              strides.data()));
        }
        break;
      default:
        LOG(FATAL) << "Unknown storage order: " << order_;
    }
  }

  vector<TIndex> cudnn_input_dims_;
  vector<TIndex> cudnn_filter_dims_;

  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t top_desc_;
  // top desc for bias add in case we do group convolution
  cudnnTensorDescriptor_t top_desc_for_bias_;
  cudnnConvolutionDescriptor_t conv_desc_;
  const size_t cudnn_ws_nbytes_limit_;
  size_t cudnn_ws_nbytes_;
  bool exhaustive_search_;
  bool deterministic_;
  size_t cudnn_state_;
  int no_bias_;
  bool bias_;
  vector<int> force_algo_; // stored as FWD, dFILTER, dDATA
};

}  // namespace caffe2
