#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_ADD_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_ADD_OP_H_

#include "caffe2/core/operator.h"
#include "context.h"
#include "kernels/add_impl.h"
#include "kernels/utils.h"

namespace caffe2 {
namespace {

template <typename T, typename Context> // Either float or cl_half
class AddOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(AddOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& Y = Input(1);
    CAFFE_ENFORCE_EQ(X.size(), Y.size());
    auto* Z = Output(0);
		Z->ResizeLike(X);
    if (!addKernel_) {
      if (std::is_same<T, float>::value) {
        kernel_args_.emplace_back(("REAL4"), "float4");
      } else if (std::is_same<T, cl_half>::value) {
        kernel_args_.emplace_back(("REAL4"), "half4");
      }
      std::string arg_list = BuildArgumentList(kernel_args_);
      addKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kAdd, arg_list));
    }
    auto& ctx = context_.GetSingleton();
    cl::Event event;
    OPENCL_CHECK(addKernel_->setArg(0, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(addKernel_->setArg(1, *(cl::Buffer*)Y.template data<T>()));
    OPENCL_CHECK(addKernel_->setArg(2, *(cl::Buffer*)Z->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *addKernel_,
          cl::NullRange,
          cl::NDRange(X.size() / 4),
          cl::NullRange,
          NULL,
          &event));
    return true;
  }
 private:
  std::unique_ptr<cl::Kernel> addKernel_;
  std::vector<std::pair<std::string, std::string>> kernel_args_;
};

} // namespace
} // namespace caffe2

#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_ADD_OP_H_

