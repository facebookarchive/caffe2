#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_SPATIAL_BN_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_SPATIAL_BN_OP_H_

#include "caffe2/core/operator.h"
#include "context.h"
#include "kernels/spatial_batch_norm_impl.h"
#include "kernels/utils.h"

namespace caffe2 {
namespace {

template <typename T, typename Context> // Either float or cl_half
class SpatialBNOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SpatialBNOp(const OperatorDef& def, Workspace* ws)
  : Operator<Context>(def, ws),
    is_test_(OperatorBase::GetSingleArgument<int>("is_test", 1)),
    order_(StringToStorageOrder(
    OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
      CAFFE_ENFORCE(is_test_, "BN only supported at inference on opencl engine");
      CAFFE_ENFORCE_EQ(order_, StorageOrder::NCHW, "Only NCHW supported on opencl engine");
    }

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& scale = OperatorBase::Inputs()[SCALE]->template Get<Tensor<CPUContext>>();
    const auto& bias =  OperatorBase::Inputs()[BIAS]->template Get<Tensor<CPUContext>>();
    const auto& mean =  OperatorBase::Inputs()[EST_MEAN]->template Get<Tensor<CPUContext>>();
    const auto& var =   OperatorBase::Inputs()[EST_VAR]->template Get<Tensor<CPUContext>>();
    if (!scaleBuffer_) { // Assume none have been copied over.
      scaleBuffer_ = caffe2::make_unique<TensorCL>(scale.dims());
      context_.template CoercedCopy<T>(scale, *scaleBuffer_);
    }
    if (!biasBuffer_) {
      biasBuffer_ = caffe2::make_unique<TensorCL>(bias.dims());
      context_.template CoercedCopy<T>(bias, *biasBuffer_);
    }
    if (!meanBuffer_) {
      meanBuffer_ = caffe2::make_unique<TensorCL>(mean.dims());
      context_.template CoercedCopy<T>(mean, *meanBuffer_);
    }
    if (!varBuffer_) {
      varBuffer_ = caffe2::make_unique<TensorCL>(var.dims());
      context_.template CoercedCopy<T>(var, *varBuffer_);
    }
    auto* Y = Output(0);
    Y->ResizeLike(X);

    if (!spatialBNKernel_) {
      if (std::is_same<T, float>::value) {
        kernel_args_.emplace_back(("REAL"), "float");
      } else if (std::is_same<T, cl_half>::value) {
        kernel_args_.emplace_back(("REAL"), "half");
      }
      std::string arg_list = BuildArgumentList(kernel_args_);
      spatialBNKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kSpatialBN, arg_list));
    }
    auto& ctx = context_.GetSingleton();
    cl::Event event;
    OPENCL_CHECK(spatialBNKernel_->setArg(0, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(spatialBNKernel_->setArg(1, X.size()));
    OPENCL_CHECK(spatialBNKernel_->setArg(2,
      *(cl::Buffer*)scaleBuffer_->template mutable_data<T>()));
    OPENCL_CHECK(spatialBNKernel_->setArg(3,
      *(cl::Buffer*)biasBuffer_->template mutable_data<T>()));
    OPENCL_CHECK(spatialBNKernel_->setArg(4,
      *(cl::Buffer*)meanBuffer_->template mutable_data<T>()));
    OPENCL_CHECK(spatialBNKernel_->setArg(5,
     *(cl::Buffer*)varBuffer_->template mutable_data<T>()));
    OPENCL_CHECK(spatialBNKernel_->setArg(6, *(cl::Buffer*)Y->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *spatialBNKernel_,
          cl::NullRange,
          cl::NDRange(X.dim32(3) * X.dim32(2), X.dim32(1)), // HW, C
          cl::NullRange,
          NULL,
          &event));
    return true;
  }
 private:
  std::unique_ptr<cl::Kernel> spatialBNKernel_;
  std::vector<std::pair<std::string, std::string>> kernel_args_;
  INPUT_TAGS(INPUT, SCALE, BIAS, EST_MEAN, EST_VAR);
  std::unique_ptr<TensorCL> scaleBuffer_;
  std::unique_ptr<TensorCL> biasBuffer_;
  std::unique_ptr<TensorCL> meanBuffer_;
  std::unique_ptr<TensorCL> varBuffer_;
  bool is_test_;
  StorageOrder order_;
};

} // namespace
} // namespace caffe2

#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_SPATIAL_BN_OP_H_

