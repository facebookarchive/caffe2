#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_CONCAT_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_CONCAT_OP_H_

#include "caffe2/core/operator.h"
#include "context.h"
#include "kernels/concat_impl.h"

namespace caffe2 {
namespace {

template <typename T, typename Context> // Either float or cl_half
class ConcatOp final : public Operator<Context> {
 public:
  ConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
     CAFFE_ENFORCE_EQ(order_, StorageOrder::NCHW);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& Y = Input(1);
    CAFFE_ENFORCE(!(X.size() % 4));
    CAFFE_ENFORCE(!(Y.size() % 4));
    auto* Z = Output(0);
    CAFFE_ENFORCE(X.dim32(0) == 1);
    CAFFE_ENFORCE(Y.dim32(0) == 1);
    CAFFE_ENFORCE_EQ(Y.dim32(2), X.dim32(2));
    CAFFE_ENFORCE_EQ(Y.dim32(3), X.dim32(3));
    const auto Yc = Y.dim32(1); // NCHW
    const auto Xc = X.dim32(1);
		Z->Resize(1, Yc + Xc, Y.dim32(2), Y.dim32(3));
    if (!concatKernel_) {
      if (std::is_same<T, float>::value) {
        kernel_args_.emplace_back(("REAL4"), "float4");
      } else if (std::is_same<T, cl_half>::value) {
        kernel_args_.emplace_back(("REAL4"), "half4");
      }
      // Work in float4
      kernel_args_.emplace_back(("A_SIZE4"), to_string(X.size() >> 2));
      std::string arg_list = BuildArgumentList(kernel_args_);
      concatKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kConcat, arg_list));
    }
    auto& ctx = context_.GetSingleton();
    cl::Event event;
    OPENCL_CHECK(concatKernel_->setArg(0, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(concatKernel_->setArg(1, *(cl::Buffer*)Y.template data<T>()));
    OPENCL_CHECK(concatKernel_->setArg(2, *(cl::Buffer*)Z->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *concatKernel_,
          cl::NullRange,
          cl::NDRange((X.size() + Y.size()) >> 2),
          cl::NullRange,
          NULL,
          &event));
    return true;
  }
 private:
  std::unique_ptr<cl::Kernel> concatKernel_;
  std::vector<std::pair<std::string, std::string>> kernel_args_;
  StorageOrder order_;
};

} // namespace
} // namespace caffe2

#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_CONCAT_OP_H_


