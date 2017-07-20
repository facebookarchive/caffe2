#include "conv_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

template<>
void ConvOp<cl_half>::TypedCopy(const Tensor<CPUContext>& src, Tensor<OpenCLContext>& dst) {
  auto tmpBuffer = caffe2::make_unique<TensorCL>(src.dims());
  context_.Copy<float>(src, *tmpBuffer);
  if (!toHalfKernel_) {
    toHalfKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kFloatToHalf));
  }
  OPENCL_CHECK(toHalfKernel_->setArg(0, *(cl::Buffer*)tmpBuffer->data<float>()));
  OPENCL_CHECK(toHalfKernel_->setArg(1, *(cl::Buffer*)dst.mutable_data<cl_half>()));

  auto& ctx = context_.GetSingleton();
  cl::Event event;
  OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
        *toHalfKernel_,
        cl::NullRange,
        cl::NDRange(src.size()),
        cl::NullRange,
        NULL,
        &event));
  event.wait();
}

template<>
void ConvOp<float>::TypedCopy(const Tensor<CPUContext>& src, Tensor<OpenCLContext>& dst) {
  dst.mutable_data<float>();
  context_.Copy<float>(src, dst);
}

REGISTER_OPENCL_OPERATOR(Conv, ConvOp<float>);
REGISTER_OPENCL_OPERATOR(ConvHalf, ConvOp<cl_half>);

} // namespace
} // namespace caffe2
