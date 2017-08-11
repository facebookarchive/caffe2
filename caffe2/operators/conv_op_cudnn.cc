#include "caffe2/operators/conv_op_cudnn_base.h"

namespace caffe2 {

// Manually specified number of algorithms implemented in CuDNN.
// This does not have any performance implications, as we will always find the
// fastest algorithm; setting them to the right number of algorithms will enable
// us to best report the statistics when doing an exhaustive search, though.
static constexpr size_t kNUM_CUDNN_FWD_ALGS = 8;

class CudnnConvOp final : public CudnnConvOpBase {
 public:
  CudnnConvOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws)  {}

  ~CudnnConvOp() {}

  template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionFwdAlgo_t algo_;
  AlgorithmsCache<cudnnConvolutionFwdAlgo_t> algo_cache_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

class CudnnConvGradientOp final : public CudnnConvOpBase {
 public:
  CudnnConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
  }

  ~CudnnConvGradientOp() {}

  template <typename T_X, typename T_DY, typename T_W, typename T_B,
            typename MATH,
            typename T_DX, typename T_DW, typename T_DB>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t> filter_algo_cache_;
  AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t> data_algo_cache_;
  bool no_bias_;
  // input: X, W, dY
  // output: dW, db, and optionally dX
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

template <typename T_X, typename T_W, typename T_B, typename MATH, typename T_Y>
bool CudnnConvOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto* Y = Output(0);

  // Figure out the output shape
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);
  const int M = filter.dim32(0);
  ConvPoolOpBase<CUDAContext>::SetOutputSize(X, Y, M);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  switch (order_) {
    case StorageOrder::NHWC:
      N = X.dim32(0);
      H = X.dim32(1);
      W = X.ndim() > 3 ? X.dim32(2) : 1;
      D = X.ndim() > 4 ? X.dim32(3) : 1;
      C = X.dim32(X.ndim() - 1);
      H_out = Y->dim32(1);
      W_out = Y->ndim() > 3 ? Y->dim32(2) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(3) : 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C / group_);
      group_offset_X = C / group_;
      group_offset_Y = M / group_;
      break;
    case StorageOrder::NCHW:
      N = X.dim32(0);
      C = X.dim32(1);
      H = X.dim32(2);
      W = X.ndim() > 3 ? X.dim32(3) : 1;
      D = X.ndim() > 4 ? X.dim32(4) : 1;
      H_out = Y->dim32(2);
      W_out = Y->ndim() > 3 ? Y->dim32(3) : 1;
      D_out = Y->ndim() > 4 ? Y->dim32(4) : 1;
      CAFFE_ENFORCE_EQ(filter.dim32(1), C / group_);
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
      }
      group_offset_X = C / group_ * H * W * D;
      group_offset_Y = M / group_ * H_out * W_out * D_out;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

  CAFFE_ENFORCE(
      C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  int group_offset_filter = filter.size() / group_;

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      SetTensorNdDescriptorWithGroup<T_X>(
          X.ndim(), bottom_desc_, N, C, H, W, D);
    }
    if (filter_changed) {
      cudnn_filter_dims_ = filter.dims();
      if (kernel_.size() == 2) {
        CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            M / group_,
            C / group_,
            kernel_h(),
            kernel_w()));
      } else {
        vector<int> dims(filter.dims().begin(), filter.dims().end());
        dims[0] /= group_;
        order_ == StorageOrder::NCHW ? dims[1] /= group_
                                     : dims[filter.ndim() - 1] /= group_;
        dims[filter.ndim() - 1] /= group_;
        CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            dims.size(),
            dims.data()));
      }
      if (InputSize() == 3) {
        if (kernel_.size() == 2) {
          CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
              bias_desc_,
              GetCudnnTensorFormat(order_),
              cudnnTypeWrapper<T_B>::type,
              1,
              M,
              1,
              1));
        } else {
          std::vector<int> bias_dims(X.ndim(), 1);
          bias_dims[1] = M;
          std::vector<int> strides = {M, 1, 1, 1, 1, 1};
          CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
              bias_desc_,
              cudnnTypeWrapper<T_B>::type,
              X.ndim() > 3 ? X.ndim() : 4,
              bias_dims.data(),
              strides.data()));
        }
      }
    }
    // Set the output
    SetTensorNdDescriptorWithGroup<T_Y>(
        X.ndim(), top_desc_, N, M, H_out, W_out, D_out);
    // Set the output with descriptor useful for bias addition in one run.
    if (kernel_.size() == 2) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<T_B>::type,
          N,
          M,
          H_out,
          W_out));
    } else {
      vector<int> dims = {N, M, H_out, W_out, D_out};
      vector<int> strides = {M * H_out * W_out * D_out,
                             H_out * W_out * D_out,
                             H_out * D_out,
                             D_out,
                             1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          top_desc_for_bias_,
          cudnnTypeWrapper<T_B>::type,
          X.ndim() > 3 ? X.ndim() : 4,
          dims.data(),
          strides.data()));
    }
    // Set the convolution descriptor
#if CUDNN_VERSION_MIN(6,0,0)
    if (kernel_.size() == 2) {
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
    if (kernel_.size() == 2) {
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
    if (force_algo_[ALGO_FWD] >= 0) {
      algo_ = (cudnnConvolutionFwdAlgo_t)force_algo_[ALGO_FWD];
    } else if (deterministic_) {
      algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else if (exhaustive_search_) {
      algo_ = algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
        VLOG(1) << "CUDNN Convolution: doing exhaustive search.";
        // When we do an exhaustive search, we will ignore the workspace size
        // limit and simply go for the fastest algorithm. If you happen to run
        // out of memory later, you will be on your own...
        int returned_algo_count;
        std::array<cudnnConvolutionFwdAlgoPerf_t, kNUM_CUDNN_FWD_ALGS>
            perf_stat;

        // no need to clean up workspace,
        cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
          // Actually run the search.
          CUDNN_ENFORCE(cudnnFindConvolutionForwardAlgorithmEx(
              state->cudnn_handle(),
              bottom_desc_,
              X.template data<T_X>(),
              filter_desc_,
              filter.template data<T_W>(),
              conv_desc_,
              top_desc_,
              Y->template mutable_data<T_Y>(),
              kNUM_CUDNN_FWD_ALGS,
              &returned_algo_count,
              perf_stat.data(),
              state->workspace().get(cudnn_ws_nbytes_limit_),
              cudnn_ws_nbytes_limit_));
        });
        LogCuDNNPerfStats(perf_stat, returned_algo_count);
        return perf_stat[0].algo;
      });
    } else {
      // Get the convolution algorithm based on the workspace limit.
      CUDNN_ENFORCE(cudnnGetConvolutionForwardAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          filter_desc_,
          conv_desc_,
          top_desc_,
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &algo_));
    }
    CUDNN_ENFORCE(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        bottom_desc_,
        filter_desc_,
        conv_desc_,
        top_desc_,
        algo_,
        &cudnn_ws_nbytes_));
    VLOG(1) << "CuDNN algorithm: " << algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  // Now, actually run the computation.
  // Filter
  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnConvolutionForward(
          state->cudnn_handle(),
          cudnnTypeWrapper<T_X>::kOne(),
          bottom_desc_,
          X.template data<T_X>() + i * group_offset_X,
          filter_desc_,
          filter.template data<T_W>() + i * group_offset_filter,
          conv_desc_,
          algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T_Y>::kZero(),
          top_desc_,
          Y->template mutable_data<T_Y>() + i * group_offset_Y));
    });
  }
  // Bias
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);

    CAFFE_ENFORCE_EQ(bias.ndim(), 1);
    CAFFE_ENFORCE_EQ(bias.dim32(0), M);

    CUDNN_ENFORCE(cudnnAddTensor(
        cudnn_wrapper_.inline_cudnn_handle(),
        cudnnTypeWrapper<T_B>::kOne(),
        bias_desc_,
        bias.template data<T_B>(),
        cudnnTypeWrapper<T_Y>::kOne(),
        top_desc_for_bias_,
        Y->template mutable_data<T_Y>()));
  }
  // Done.
  return true;
}

bool CudnnConvOp::RunOnDevice() {

  if (Input(0).IsType<float>()) {
    return DoRunWithType<float,      // X
                         float,      // W
                         float,      // B
                         float,      // Math
                         float>();   // Y
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16,      // X
                         float16,      // W
                         float16,      // B
                         float,      // Math
                         float16>();   // Y
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 are supported by "
               << "cudnn convolution, but input " << debug_def().input(0)
               << " has [" << Input(0).meta().name() << "]";
  }
  return true;
}


namespace {

REGISTER_CUDNN_OPERATOR(Conv, CudnnConvOp);

}


}  // namespace caffe2
