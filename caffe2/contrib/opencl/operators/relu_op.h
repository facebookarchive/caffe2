#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_RELU_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_RELU_OP_H_

#include "caffe2/core/operator.h"
#include "context.h"
#include "kernels/relu_impl.h"
#include "kernels/utils.h"

namespace caffe2 {
namespace {

template <typename T, typename Context> // Either float or cl_half
class ReluOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(ReluOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
		Y->ResizeLike(X);
    if (!reluKernel_) {
      if (std::is_same<T, float>::value) {
        kernel_args_.emplace_back(("REAL4"), "float4");
      } else if (std::is_same<T, cl_half>::value) {
        kernel_args_.emplace_back(("REAL4"), "half4");
      }
      std::string arg_list = BuildArgumentList(kernel_args_);
      reluKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kRelu, arg_list));
    }
    auto& ctx = context_.GetSingleton();
    cl::Event event;
    OPENCL_CHECK(reluKernel_->setArg(0, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(reluKernel_->setArg(1, *(cl::Buffer*)Y->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *reluKernel_,
          cl::NullRange,
          cl::NDRange(X.size() / 4),
          cl::NullRange,
          NULL,
          &event));
    return true;
  }
 private:
  std::unique_ptr<cl::Kernel> reluKernel_;
  std::vector<std::pair<std::string, std::string>> kernel_args_;
};

} // namespace
} // namespace caffe2

#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_RELU_OP_H_
