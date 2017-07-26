#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_CHANNEL_SHUFFLE_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_CHANNEL_SHUFFLE_OP_H_

#include "caffe2/core/operator.h"
#include "context.h"
#include "kernels/channel_shuffle_impl.h"
#include "kernels/utils.h"

namespace caffe2 {
namespace {

template <typename T, typename Context> // Either float or cl_half
class ChannelShuffleOp final : public Operator<Context> {
 public:
  ChannelShuffleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        group_(OperatorBase::template GetSingleArgument<int>("group", 1)) {
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
		Y->ResizeLike(X);
		
    if (!channelShuffleKernel_) {
      if (std::is_same<T, float>::value) {
        kernel_args_.emplace_back(("REAL"), "float");
      } else if (std::is_same<T, cl_half>::value) {
        kernel_args_.emplace_back(("REAL"), "half");
      }
      kernel_args_.emplace_back(("SPATIAL"), to_string(X.dim32(2) * X.dim32(3)));
      kernel_args_.emplace_back(("CHANNEL"), to_string(X.dim32(1)));
      kernel_args_.emplace_back(("GROUP"), to_string(group_));
      std::string arg_list = BuildArgumentList(kernel_args_);
      channelShuffleKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kChannelShuffle, arg_list));
    }
    auto& ctx = context_.GetSingleton();
    cl::Event event;
    OPENCL_CHECK(channelShuffleKernel_->setArg(0, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(channelShuffleKernel_->setArg(1, *(cl::Buffer*)Y->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *channelShuffleKernel_,
          cl::NullRange,
          cl::NDRange(X.size()),
          cl::NullRange,
          NULL,
          &event));
    return true;
  }
 private:
  std::unique_ptr<cl::Kernel> channelShuffleKernel_;
  std::vector<std::pair<std::string, std::string>> kernel_args_;
  int group_;
};

} // namespace
} // namespace caffe2

#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_CHANNEL_SHUFFLE_OP_H_

