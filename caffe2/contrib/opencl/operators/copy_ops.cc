#include "caffe2/core/operator.h"

#include "kernels/utils.h"
#include "context.h"
#include "operator.h"

namespace caffe2 {
namespace {

class CopyToOpenCLHalfOp final : public Operator<OpenCLContext> {
 public:
  CopyToOpenCLHalfOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {
		kernel_ = context_.BuildKernel(kFloatToHalf);
  }

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->Get<Tensor<CPUContext>>();
    auto* Y = Output(0);
    if (!tmpY_) {
      tmpY_ = caffe2::make_unique<Tensor<OpenCLContext>>(X.dims());
    }
    tmpY_->Resize(X.dims());
    Y->Resize(X.dims());

    context_.Copy<float>(X, *tmpY_);

    OPENCL_CHECK(kernel_.setArg(0, *(cl::Buffer*)tmpY_->data<float>()));
    OPENCL_CHECK(kernel_.setArg(1, *(cl::Buffer*)Y->mutable_data<cl_half>()));
		cl::Event event;
    auto& ctx = context_.GetSingleton();
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
					kernel_,
					cl::NullRange,
					cl::NDRange(X.size()),
					cl::NullRange,
					NULL,
					&event));

    return true;
  }
 private:
  std::unique_ptr<Tensor<OpenCLContext>> tmpY_;
  cl::Kernel kernel_;
};

REGISTER_CPU_OPERATOR(CopyToOpenCLHalf, CopyToOpenCLHalfOp);
OPERATOR_SCHEMA(CopyToOpenCLHalf).NumInputs(1, 2).NumOutputs(1);

class CopyFromOpenCLHalfOp final : public Operator<OpenCLContext> {
 public:
  CopyFromOpenCLHalfOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {
		kernel_ = context_.BuildKernel(kHalfToFloat);
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    if (!tmpX_) {
      tmpX_ = caffe2::make_unique<Tensor<OpenCLContext>>(X.dims());
    }
    tmpX_->Resize(X.dims());

    auto* Y = Outputs()[0]->GetMutable<Tensor<CPUContext>>();
    Y->Resize(X.dims());

    OPENCL_CHECK(kernel_.setArg(0, *(cl::Buffer*)X.data<cl_half>()));
    OPENCL_CHECK(kernel_.setArg(1, *(cl::Buffer*)tmpX_->mutable_data<float>()));

		cl::Event event;
    auto& ctx = context_.GetSingleton();
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
					kernel_,
					cl::NullRange,
					cl::NDRange(X.size()),
					cl::NullRange,
					NULL,
					&event));

    ctx.queue.finish();

    context_.Copy<float>(*tmpX_, *Y);

    return true;
  }
 private:
  std::unique_ptr<Tensor<OpenCLContext>> tmpX_;
  cl::Kernel kernel_;
};

REGISTER_CPU_OPERATOR(CopyFromOpenCLHalf, CopyFromOpenCLHalfOp);
OPERATOR_SCHEMA(CopyFromOpenCLHalf).NumInputs(1).NumOutputs(1);

class CopyToOpenCLOp final : public Operator<OpenCLContext> {
 public:
  CopyToOpenCLOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->Get<Tensor<CPUContext>>();
    auto* Y = Output(0);
    Y->Resize(X.dims());
    context_.Copy<float>(X, *Y);
    return true;
  }
};

REGISTER_CPU_OPERATOR(CopyToOpenCL, CopyToOpenCLOp);
OPERATOR_SCHEMA(CopyToOpenCL).NumInputs(1, 2).NumOutputs(1);

class CopyFromOpenCLOp final : public Operator<OpenCLContext> {
 public:
  CopyFromOpenCLOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<OpenCLContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Outputs()[0]->GetMutable<Tensor<CPUContext>>();
    Y->Resize(X.dims());
    auto& ctx = context_.GetSingleton();
    ctx.queue.finish();
    context_.Copy<float>(X, *Y);
    return true;
  }
};

REGISTER_CPU_OPERATOR(CopyFromOpenCL, CopyFromOpenCLOp);
OPERATOR_SCHEMA(CopyFromOpenCL).NumInputs(1).NumOutputs(1);

} // namespace
} // namespace caffe2
