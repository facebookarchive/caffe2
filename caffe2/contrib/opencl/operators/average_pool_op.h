#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_AVERAGE_POOL_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_AVERAGE_POOL_OP_H_

#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "context.h"
#include "kernels/average_pool_impl.h"

namespace caffe2 {
namespace {

template <typename T, typename Context> // Either float or cl_half
class AveragePoolOp final : public ConvPoolOpBase<Context> {
 public:
  AveragePoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
  }
  USE_CONV_POOL_BASE_FUNCTIONS(Context);

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    ConvPoolOpBase<Context>::SetOutputSize(X, Y, X.dim32(1));
    if (!averagePoolKernel_) {
      if (std::is_same<T, float>::value) {
        kernel_args_.emplace_back(("REAL"), "float");
      } else if (std::is_same<T, cl_half>::value) {
        kernel_args_.emplace_back(("REAL"), "half");
      }
      // Work in float4
      kernel_args_.emplace_back(("KERNEL"), to_string(kernel_[0]));
      kernel_args_.emplace_back(("WIDTH"), to_string(X.dim32(3)));
      kernel_args_.emplace_back(("HEIGHT"), to_string(X.dim32(2)));
      kernel_args_.emplace_back(("OUT_WIDTH"), to_string(Y->dim32(3)));
      kernel_args_.emplace_back(("OUT_HEIGHT"), to_string(Y->dim32(2)));
      kernel_args_.emplace_back(("STRIDE"), to_string(stride_[0]));
      std::string arg_list = BuildArgumentList(kernel_args_);
      averagePoolKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kAveragePool, arg_list));
    }
    auto& ctx = context_.GetSingleton();
    cl::Event event;
    OPENCL_CHECK(averagePoolKernel_->setArg(0, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(averagePoolKernel_->setArg(1, *(cl::Buffer*)Y->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *averagePoolKernel_,
          cl::NullRange,
          cl::NDRange(Y->dim32(3), Y->dim32(2), Y->dim32(1)),
          cl::NullRange,
          NULL,
          &event));
    return true;
  }
 private:
  std::unique_ptr<cl::Kernel> averagePoolKernel_;
  std::vector<std::pair<std::string, std::string>> kernel_args_;
};

} // namespace
} // namespace caffe2

#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_AVERAGE_POOL_OP_H_



