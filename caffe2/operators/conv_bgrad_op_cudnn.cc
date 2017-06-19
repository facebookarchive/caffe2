#include "caffe2/operators/conv_op_cudnn_base.h"

namespace caffe2 {

class CudnnConvBiasGradientOp final : public CudnnConvOpBase {
 public:
  CudnnConvBiasGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : CudnnConvOpBase(operator_def, ws) {}
  ~CudnnConvBiasGradientOp() {}

  template <typename T_DY, typename MATH, typename T_DB>
  bool DoRunWithType();

  bool RunOnDevice() override;

 private:
  // in/out: {X, dY, W, B} -> {db}
  INPUT_TAGS(OUTPUT_GRAD);
  OUTPUT_TAGS(BIAS_GRAD);
};

template <typename T_DY, typename MATH, typename T_DB>
bool CudnnConvBiasGradientOp::DoRunWithType() {
	auto& dY = Input(OUTPUT_GRAD);
	auto* dbias = Output(BIAS_GRAD);

  int N = 0, C_out = 0, H_out = 0, W_out = 0, D_out = 0;

  DimensionParam params = GetDimensionsFromTensorDims(
      dY.ndim(), dY.dims(), order_);

  N = params.N;
  C_out = params.C;
  H_out = params.H;
  W_out = params.W;
  D_out = params.D;

  if (kernel_.size() == 2) {
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        bias_desc_,
        GetCudnnTensorFormat(order_),
        cudnnTypeWrapper<T_DB>::type,
        1,
        C_out,
        1,
        1));
  } else {
    std::vector<int> bias_dims(dY.ndim(), 1);
    bias_dims[1] = C_out;
    std::vector<int> strides = {C_out, 1, 1, 1, 1, 1};
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        bias_desc_,
        cudnnTypeWrapper<T_DB>::type,
        dY.ndim() > 3 ? dY.ndim() : 4,
        bias_dims.data(),
        strides.data()));
  }

  // Set the output with descriptor useful for bias addition in one run.
  if (kernel_.size() == 2) {
    CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
        top_desc_for_bias_,
        GetCudnnTensorFormat(order_),
        cudnnTypeWrapper<T_DB>::type,
        N,
        C_out,
        H_out,
        W_out));
  } else {
    vector<int> dims = {N, C_out, H_out, W_out, D_out};
    vector<int> strides = {C_out * H_out * W_out * D_out,
                           H_out * W_out * D_out,
                           H_out * D_out,
                           D_out,
                           1};
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        top_desc_for_bias_,
        cudnnTypeWrapper<T_DB>::type,
        dY.ndim() > 3 ? dY.ndim() : 4,
        dims.data(),
        strides.data()));
  }
  // Done.

  dbias->Resize(C_out);

  CUDNN_ENFORCE(cudnnConvolutionBackwardBias(
      cudnn_wrapper_.inline_cudnn_handle(),
      cudnnTypeWrapper<T_DY>::kOne(),
      top_desc_for_bias_,
      dY.template data<T_DY>(),
      cudnnTypeWrapper<T_DB>::kZero(),
      bias_desc_,
      dbias->template mutable_data<T_DB>()));

   return true;
}


bool CudnnConvBiasGradientOp::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float,    // dY
                         float,    // Math
                         float>(); // db
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16,    // dY
                         float,      // Math
                         float16>(); // db
  } else {
    LOG(FATAL) << "Only float (32bit) and float16 are supported by "
               << "cudnn bgrad convolution, but input " << def().input(0) << " has ["
               << Input(0).meta().name() << "]";
	}
  return false;
};

namespace {

OPERATOR_SCHEMA(ConvBiasGradient)
  .NumInputs(1)
  .NumOutputs(1);

REGISTER_CUDNN_OPERATOR(ConvBiasGradient, CudnnConvBiasGradientOp);

}

}  // namespace cafffe2
