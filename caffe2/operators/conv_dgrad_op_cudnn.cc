#include "caffe2/operators/conv_op_cudnn_base.h"

namespace caffe2 {

static constexpr size_t kNUM_CUDNN_BWD_DATA_ALGS = 6;

class CudnnConvDataGradientOp final : public CudnnConvOpBase {
 public:
  CudnnConvDataGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws) {}
  ~CudnnConvDataGradientOp() {}

  template <typename T_X, typename T_W, typename T_DY, typename MATH, typename T_DX>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  cudnnConvolutionBwdDataAlgo_t algo_;
  AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t> algo_cache_;

  // in/out: {X, W, dY} -> {dX}
  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};


template <typename T_X, typename T_W, typename T_DY,
          typename MATH,
          typename T_DX>
bool CudnnConvDataGradientOp::DoRunWithType() {
  auto& X = Input(INPUT);
  auto& filter = Input(FILTER);
  auto& dY = Input(OUTPUT_GRAD);
  auto* dX = Output(INPUT_GRAD);

  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  CAFFE_ENFORCE(filter.ndim() >= 3 && filter.ndim() <= 5);

  dX->ResizeLike(X);

  // get data pointers for every tensor
  const T_X* X_data = X.template data<T_X>();
  const T_W* filter_data = filter.template data<T_X>();
  const T_DY* dY_data = dY.template data<T_DY>();
  T_DX* dX_data = dX->template mutable_data<T_DX>();

  const int M = filter.dim32(0);
  int N = 0, C = 0;
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
      group_offset_X = C / group_ * in.H * in.W * in.D;
      group_offset_Y = M / group_ * out.H * out.W * out.D;
      break;
    default:
      LOG(FATAL) << "Unknown storage order: " << order_;
  }

  CAFFE_ENFORCE(
      in.C % group_ == 0,
      "If you set group, the number of input channels should be divisible "
      "by group.");
  CAFFE_ENFORCE(
      M % group_ == 0,
      "If you set group, the number of output channels should be divisible "
      "by group.");

  int group_offset_filter = filter.size() / group_;
  if (kernel_.size() == 1) {
    ConvPoolOpBase<CUDAContext>::ComputePads({in.H});
  } else if (kernel_.size() == 2) {
    ConvPoolOpBase<CUDAContext>::ComputePads({in.H, in.W});
  } else if (kernel_.size() == 3) {
    ConvPoolOpBase<CUDAContext>::ComputePads({in.H, in.W, in.D});
  } else {
    CAFFE_THROW("Unsupported kernel size:", kernel_.size());
  }

  // Set up the cudnn algorithms & workspace if necessary
  bool input_changed = (X.dims() != cudnn_input_dims_);
  bool filter_changed = (filter.dims() != cudnn_filter_dims_);
  if (input_changed || filter_changed) {
    VLOG(1) << "Changing the cudnn descriptor configurations.";
    if (input_changed) {
      cudnn_input_dims_ = X.dims();
      SetTensorNdDescriptorWithGroup<T_X>(
          X.ndim(), bottom_desc_, N, C, in.H, in.W, in.D);
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
    SetTensorNdDescriptorWithGroup<T_DX>(
        X.ndim(), top_desc_, in.N, M, out.H, out.W, out.D);

    // Set the convolution descriptor
    SetConvolutionDescriptor<MATH>(
          kernel_.size(),
          conv_desc_);

    // Set the workspace
    size_t ws_size;

    // Pick dX algo if needed
    if (force_algo_[ALGO_DGRAD] >= 0) {
      algo_ = (cudnnConvolutionBwdDataAlgo_t)force_algo_[ALGO_DGRAD];
    } else if (deterministic_) {
      algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else if (exhaustive_search_) {
      algo_ =
          algo_cache_.getAlgorithm(X.dims(), filter.dims(), [&]() {
            VLOG(1) << "CUDNN Convolution bwd: doing data exhaustive search.";
            int returned_algo_count;

            std::array<
                cudnnConvolutionBwdDataAlgoPerf_t,
                kNUM_CUDNN_BWD_DATA_ALGS>
                data_perf_stat;
            cudnn_wrapper_.with_cudnn_state(
                cudnn_state_, [&](CuDNNState* state) {
                  CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithmEx(
                      state->cudnn_handle(),
                      filter_desc_,
                      filter_data,
                      top_desc_,
                      dY_data,
                      conv_desc_,
                      bottom_desc_,
                      dX_data,
                      kNUM_CUDNN_BWD_DATA_ALGS,
                      &returned_algo_count,
                      data_perf_stat.data(),
                      state->workspace().get(cudnn_ws_nbytes_limit_),
                      cudnn_ws_nbytes_limit_));
                });

            LogCuDNNPerfStats(data_perf_stat, returned_algo_count);
            return data_perf_stat[0].algo;
          });
    } else {
      CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm(
          cudnn_wrapper_.inline_cudnn_handle(),
          filter_desc_,
          top_desc_,
          conv_desc_,
          bottom_desc_,
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          cudnn_ws_nbytes_limit_,
          &algo_));
    }

    // get workspace for backwards data algorithm
    CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        filter_desc_,
        top_desc_,
        conv_desc_,
        bottom_desc_,
        algo_,
        &ws_size));
    cudnn_ws_nbytes_ = ws_size;

    VLOG(1) << "CuDNN bwd data algorithm: " << algo_;
    VLOG(1) << "CuDNN workspace size: " << cudnn_ws_nbytes_;
  }

  for (int i = 0; i < group_; ++i) {
    cudnn_wrapper_.with_cudnn_state(cudnn_state_, [&](CuDNNState* state) {
      // Compute the gradient w.r.t. the input.
      auto* dX = Output(INPUT_GRAD);
      dX->ResizeLike(X);
      CUDNN_ENFORCE(cudnnConvolutionBackwardData(
          state->cudnn_handle(),
          cudnnTypeWrapper<T_W>::kOne(),
          filter_desc_,
          filter_data + i * group_offset_filter,
          top_desc_,
          dY_data + i * group_offset_Y,
          conv_desc_,
          algo_,
          state->workspace().get(cudnn_ws_nbytes_),
          cudnn_ws_nbytes_,
          cudnnTypeWrapper<T_DX>::kZero(),
          bottom_desc_,
          dX_data + i * group_offset_X));
    });
  }
  return true;
}

// TODO(Yangqing): a lot of the function contents are very similar. Consider
// consolidating them.
bool CudnnConvDataGradientOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float,    //  X
                         float,    //  W
                         float,    // dY
                         float,    // Math
                         float>(); // dX
  }
  else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16,    //  X
                         float16,    //  W
                         float16,    // dY
                         float,      // Math
                         float16>(); // dX
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 are supported by "
               << "cudnn dgrad convolution, but input " << def().input(0) << " has ["
               << Input(0).meta().name() << "]";
  }
  return true;
}

namespace {

OPERATOR_SCHEMA(ConvDataGradient)
  .NumInputs(3)
  .NumOutputs(1);

REGISTER_CUDNN_OPERATOR(ConvDataGradient, CudnnConvDataGradientOp);

}

}  // namespace caffe2
