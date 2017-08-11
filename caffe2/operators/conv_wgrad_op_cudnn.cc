#include "caffe2/operators/conv_op_cudnn_base.h"

namespace caffe2 {

// Manually specified number of algorithms implemented in CuDNN.
// This does not have any performance implications, as we will always find the
// fastest algorithm; setting them to the right number of algorithms will enable
// us to best report the statistics when doing an exhaustive search, though.
static constexpr size_t kNUM_CUDNN_BWD_FILTER_ALGS = 6;

class CudnnConvFilterGradientOp final : public CudnnConvOpBase {
 public:
  CudnnConvFilterGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws) {}
  ~CudnnConvFilterGradientOp() {}

  template <typename T_X, typename T_DY, typename T_W, typename MATH, typename T_DW>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionBwdFilterAlgo_t algo_;
  AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t> algo_cache_;

  // in/out: {X, dY, W} -> {dW}
  INPUT_TAGS(INPUT, OUTPUT_GRAD, FILTER);
  OUTPUT_TAGS(FILTER_GRAD);
};

template <typename T_X, typename T_DY, typename T_W, typename MATH, typename T_DW>
bool CudnnConvFilterGradientOp::DoRunWithType() {
	auto& X = Input(INPUT);
  auto& dY = Input(OUTPUT_GRAD);
  auto& filter = Input(FILTER);
  auto* dfilter = Output(FILTER_GRAD);
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);

  const int M = filter.dim32(0);
  int N = 0, C = 0, H = 0, W = 0, D = 0, H_out = 0, W_out = 0, D_out = 0;
  int group_offset_X = 0, group_offset_Y = 0;

  DimensionParam in = GetDimensionsFromTensorDims(
      X.ndim(),
      X.dims(),
      order_);

  DimensionParam out = GetDimensionsFromTensorDims(
      dY.ndim(),
      dY.dims(),
      order_);

  N = in.N;
  C = in.C;
  H = in.H;
  W = in.W;
  D = in.D;

  H_out = out.H;
  W_out = out.W;
  D_out = out.D;

  switch (order_) {
    case StorageOrder::NHWC:
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
      }
      CAFFE_ENFORCE_EQ(filter.dim32(filter.ndim() - 1), C / group_);
      group_offset_X = C / group_;
      group_offset_Y = M / group_;
      break;
    case StorageOrder::NCHW:
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
  if (kernel_.size() == 1) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H});
  } else if (kernel_.size() == 2) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H, W});
  } else if (kernel_.size() == 3) {
    ConvPoolOpBase<CUDAContext>::ComputePads({H, W, D});
  } else {
    CAFFE_THROW("Unsupported kernel size:", kernel_.size());
  }
  dfilter->ResizeLike(filter);

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
        CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
            filter_desc_,
            cudnnTypeWrapper<T_W>::type,
            GetCudnnTensorFormat(order_),
            dims.size(),
            dims.data()));
      }
    }
    // Set the output
    SetTensorNdDescriptorWithGroup<T_DY>(
        X.ndim(), top_desc_, N, M, H_out, W_out, D_out);
    // Set the convolution descriptor
    SetConvolutionDescriptor<MATH>(
        kernel_.size(),
        conv_desc_);

    // Set the workspace
    size_t ws_size;

    if (force_algo_[ALGO_WGRAD] >= 0) {
      algo_ = (cudnnConvolutionBwdFilterAlgo_t)force_algo_[ALGO_WGRAD];
    } else if (deterministic_) {
      algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else if (exhaustive_search_) {
      algo_ =
          algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
            VLOG(1) << "CUDNN Convolution bwd: doing filter exhaustive search.";
            // When we do an exhaustive search, we will ignore the workspace
            // size
            // limit and simply go for the fastest algorithm. If you happen to
            // run
            // out of memory later, you will be on your own...
            int returned_algo_count;
            // We clean up the current workspace memory so that the forward
            // algorithm is free to allocate memory.
            // Actually run the search.
            std::array<
                cudnnConvolutionBwdFilterAlgoPerf_t,
                kNUM_CUDNN_BWD_FILTER_ALGS>
                filter_perf_stat;

            cudnn_wrapper_.with_cudnn_state(
                cudnn_state_, [&](CuDNNState* state) {
                  CUDNN_ENFORCE(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                      state->cudnn_handle(),
                      bottom_desc_,
                      X.template data<T_X>(),
                      top_desc_,
                      dY.template data<T_DY>(),
                      conv_desc_,
                      filter_desc_,
                      dfilter->template mutable_data<T_DW>(),
                      kNUM_CUDNN_BWD_FILTER_ALGS,
                      &returned_algo_count,
                      filter_perf_stat.data(),
                      state->workspace().get(cudnn_ws_nbytes_limit_),
                      cudnn_ws_nbytes_limit_));
                });
            LogCuDNNPerfStats(filter_perf_stat, returned_algo_count);
            return filter_perf_stat[0].algo;
          });

    } else {
      // choose backward algorithm for filter
      CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          bottom_desc_,
          top_desc_,
          conv_desc_,
          filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &algo_));
    }
    // get workspace for backwards filter algorithm
    CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        bottom_desc_,
        top_desc_,
        conv_desc_,
        filter_desc_,
        algo_,
        &ws_size));
    cudnn_ws_nbytes_ = ws_size;

    VLOG(1) << "CuDNN bwd filter algorithm: " << algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(
          state->cudnn_handle(),
          cudnnTypeWrapper<T_X>::kOne(),
          bottom_desc_,
          X.template data<T_X>() + i * group_offset_X,
          top_desc_,
          dY.template data<T_DY>() + i * group_offset_Y,
          conv_desc_,
          algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T_DW>::kZero(),
          filter_desc_,
          dfilter->template mutable_data<T_DW>() + i * group_offset_filter));
    });
  }
  return true;
}

bool CudnnConvFilterGradientOp::RunOnDevice() {
	if (Input(0).IsType<float>()) {
		return DoRunWithType<float,    // X
                         float,    // dY
                         float,    // W
                         float,    // Math
                         float>(); // dW
	} else if (Input(0).IsType<float16>()) {
		return DoRunWithType<float16,    // X
                         float16,    // dY
                         float16,    // W
                         float,      // Mat
                         float16>(); // dW
	} else {
    LOG(FATAL) << "Only float (32bit) and float16 are supported by "
               << "cudnn wgrad convolution, but input " << def().input(0) << " has ["
               << Input(0).meta().name() << "]";
	}
	return false;
}

namespace {

OPERATOR_SCHEMA(ConvFilterGradient)
  .NumInputs(3)
  .NumOutputs(1);

REGISTER_CUDNN_OPERATOR(ConvFilterGradient, CudnnConvFilterGradientOp);

}

}  // namespace caffe2
