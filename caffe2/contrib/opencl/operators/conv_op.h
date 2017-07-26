#ifndef CAFFE2_CONTRIB_OPENCL_OPERATORS_CONV_OP_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATORS_CONV_OP_H_

#include "caffe2/operators/conv_pool_op_base.h"

#include "context.h"
#include "kernels/conv_impl.h"
#include "kernels/utils.h"

namespace caffe2 {
namespace {

template <typename T> // Either float or cl_half
class ConvOp final : public ConvPoolOpBase<OpenCLContext> {
 public:
  ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<OpenCLContext>(operator_def, ws),
        use_MEC(OperatorBase::template GetSingleArgument<bool>("use_MEC", false)) {
    // TODO(bwasti): This operation isn't quite complete, padding/stride(?)/bias missing
    CAFFE_ENFORCE(kernel_[0] == kernel_[1], "OpenCL currently only supports square kernels");

    // Set up stuff
    if (std::is_same<T, float>::value) {
      kernel_args_.emplace_back(("REAL"), "float");
      kernel_args_.emplace_back(("REAL4"), "float4");
      kernel_args_.emplace_back(("READ_IMAGE"), "read_imagef");
      filter_image_format_ = cl::ImageFormat(CL_RGBA, CL_FLOAT);
    } else if (std::is_same<T, cl_half>::value) {
      kernel_args_.emplace_back(("REAL"), "half");
      kernel_args_.emplace_back(("REAL4"), "half4");
      kernel_args_.emplace_back(("READ_IMAGE"), "read_imageh");
      filter_image_format_ = cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT);
    }

    kernel_args_.emplace_back(("KERNEL"), to_string(kernel_[0]));
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(0);
    auto& filter = Inputs()[FILTER]->template Get<Tensor<CPUContext>>();
    if (InputSize() > 2) {
      auto& bias = Inputs()[BIAS]->template Get<Tensor<CPUContext>>();
    }
    auto* Y = Output(0);
    CAFFE_ENFORCE_EQ(X.dims().size(), 4);
    const int N = X.dim(0), H_in = X.dim(2), W_in = X.dim(3), C_in = X.dim(1);
    CAFFE_ENFORCE_EQ(filter.ndim(), 4);
    CAFFE_ENFORCE_EQ(filter.dim(1) * group_, C_in);
    CAFFE_ENFORCE_EQ(filter.dim(2), kernel_[0]);
    CAFFE_ENFORCE_EQ(filter.dim(3), kernel_[1]);
    const int C_out = filter.dim(0);
    ConvPoolOpBase<OpenCLContext>::SetOutputSize(X, Y, filter.dim32(0));
    const int H_out = Y->dim(2);
    const int W_out = Y->dim(3);
    if (kernel_[0] == kernel_[1] && kernel_[0] == 1) {
      CAFFE_ENFORCE(!(C_out % 8));
      CAFFE_ENFORCE(!(C_in % 8));
      return Run1x1GConv(N, C_out, C_in, H_out, H_in, W_out, W_in, group_);
    } else if (kernel_[0] == kernel_[1] &&
            stride_[0] == stride_[1] &&
            kernel_[0] == 3 &&
            group_ == C_in) {
      return Run3x3DWConv(N, C_out, C_in, H_out, H_in, W_out, W_in, group_, stride_[0]);
    }
    CAFFE_THROW("No known implementation of this Conv in NCHW");
    return false;
  }
  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Input(0);
    auto& filter = Inputs()[FILTER]->template Get<Tensor<CPUContext>>();
		if (InputSize() > 2) {
			auto& bias = Inputs()[BIAS]->template Get<Tensor<CPUContext>>();
		}
    auto* Y = Output(0);

    // Dimensionality setup
    CAFFE_ENFORCE_EQ(X.dims().size(), 4);
    const int N = X.dim(0), H_in = X.dim(1), W_in = X.dim(2), C_in = X.dim(3);
    CAFFE_ENFORCE_EQ(filter.ndim(), 4);
    CAFFE_ENFORCE_EQ(filter.dim(3), C_in);
    CAFFE_ENFORCE_EQ(filter.dim(1), kernel_[0]);
    CAFFE_ENFORCE_EQ(filter.dim(2), kernel_[1]);
    const int C_out = filter.dim(0);
    ConvPoolOpBase<OpenCLContext>::SetOutputSize(X, Y, filter.dim32(0));
    const int H_out = Y->dim(1);
    const int W_out = Y->dim(2);
    if (C_in >= 128 || use_MEC) {
      return RunWithMECConv(N, C_out, C_in, H_out, H_in, W_out, W_in);
    } else {
      return RunWithDirectConv(N, C_out, C_in, H_out, H_in, W_out, W_in);
    }
  }

 private:
  cl::Image* filterBufferImage_;
  std::unique_ptr<TensorCL> filterBuffer_;
  std::unique_ptr<TensorCL> mecBuffer_;
  // Lowered convolution.
  std::unique_ptr<cl::Kernel> loweringKernel_;
  std::unique_ptr<cl::Kernel> gemmKernel_;
  // 1x1 Grouped conv
  std::unique_ptr<cl::Kernel> gemm1x1Kernel_;
  // 3x3 DW Conv
  std::unique_ptr<cl::Kernel> dw3x3Kernel_;
  // Direct convolution.
  std::unique_ptr<cl::Kernel> directKernel_;

  std::unique_ptr<cl::Kernel> toHalfKernel_;
  std::vector<std::pair<std::string, std::string>> kernel_args_;
  cl::ImageFormat filter_image_format_;
  bool use_MEC = false;
  INPUT_TAGS(INPUT, FILTER, BIAS);

  bool Run3x3DWConv(const int N,
      const int C_out, const int C_in,
      const int H_out, const int H_in,
      const int W_out, const int W_in,
      const int G, const int s) {
    CAFFE_ENFORCE_EQ(kernel_[0], 3);
    const auto& X = Input(0);
    auto& filter = Inputs()[FILTER]->template Get<Tensor<CPUContext>>();
		if (InputSize() > 2) {
			auto& bias = Inputs()[BIAS]->template Get<Tensor<CPUContext>>();
		}
    auto* Y = Output(0);
    CAFFE_ENFORCE_EQ(X.dim32(1), C_out);
    CAFFE_ENFORCE_EQ(X.dim32(1), group_);

    if (!dw3x3Kernel_) {
      kernel_args_.emplace_back(("STRIDE"),  to_string(s));
      std::string arg_list = BuildArgumentList(kernel_args_);
      dw3x3Kernel_ = make_unique<cl::Kernel>(context_.BuildKernel(k3x3DW, arg_list));
    }

    auto& ctx = context_.GetSingleton();
    cl::Event event;
    if (!filterBuffer_ || filterBuffer_->size() != filter.size()) {
      filterBuffer_ = caffe2::make_unique<TensorCL>(filter.dims());
      context_.CoercedCopy<T>(filter, *filterBuffer_);
    }

    OPENCL_CHECK(dw3x3Kernel_->setArg(0, *(cl::Buffer*)filterBuffer_->template data<T>()));
    OPENCL_CHECK(dw3x3Kernel_->setArg(1, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(dw3x3Kernel_->setArg(2, H_in));
    OPENCL_CHECK(dw3x3Kernel_->setArg(3, H_out));
    OPENCL_CHECK(dw3x3Kernel_->setArg(4, W_in));
    OPENCL_CHECK(dw3x3Kernel_->setArg(5, W_out));
    OPENCL_CHECK(dw3x3Kernel_->setArg(6, *(cl::Buffer*)Y->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *dw3x3Kernel_,
          cl::NullRange,
          cl::NDRange((W_out + 3) >> 2, (H_out + 3) >> 2, C_out), // This order should be tuned
          cl::NullRange,
          NULL,
          &event));
    return true;

  }
  
  bool Run1x1GConv(const int N,
      const int C_out, const int C_in,
      const int H_out, const int H_in,
      const int W_out, const int W_in,
      const int G) {
    const auto& X = Input(0);
		auto& filter = Inputs()[FILTER]->template Get<Tensor<CPUContext>>();
		if (InputSize() > 2) {
			auto& bias = Inputs()[BIAS]->template Get<Tensor<CPUContext>>();
		}
    auto* Y = Output(0);

    // We compile the kernels on the first run to get channel info embedded
    if (!gemm1x1Kernel_) {
      kernel_args_.emplace_back(("IN_CHANNEL_DIV_G"),  to_string(C_in / G));
      kernel_args_.emplace_back(("OUT_CHANNEL_DIV_G"), to_string(C_out / G));
      kernel_args_.emplace_back(("FILTER_DIV_G"), to_string(filter.size() / G));
      std::string arg_list = BuildArgumentList(kernel_args_);
      gemm1x1Kernel_ = make_unique<cl::Kernel>(context_.BuildKernel(k1x1Gemm, arg_list));
    }

    auto& ctx = context_.GetSingleton();
    cl::Event event;
    if (!filterBuffer_ || filterBuffer_->size() != filter.size()) {
      filterBuffer_ = caffe2::make_unique<TensorCL>(filter.dims());
      context_.CoercedCopy<T>(filter, *filterBuffer_);
    }

    OPENCL_CHECK(gemm1x1Kernel_->setArg(0, *(cl::Buffer*)filterBuffer_->template data<T>()));
    OPENCL_CHECK(gemm1x1Kernel_->setArg(1, C_in / G));
    OPENCL_CHECK(gemm1x1Kernel_->setArg(2, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(gemm1x1Kernel_->setArg(3, *(cl::Buffer*)Y->template mutable_data<T>()));
    OPENCL_CHECK(gemm1x1Kernel_->setArg(4, H_out * W_out)); // LDC
    OPENCL_CHECK(gemm1x1Kernel_->setArg(5, C_out / G)); // M
    OPENCL_CHECK(gemm1x1Kernel_->setArg(6, H_out * W_out)); // N
    OPENCL_CHECK(gemm1x1Kernel_->setArg(7, C_in / G)); // K
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *gemm1x1Kernel_,
          cl::NullRange,
          cl::NDRange((H_out * W_out) >> 2, (C_out / G) >> 3, G),
          cl::NullRange,
          NULL,
          &event));
    return true;
  }

  bool RunWithMECConv(const int N,
      const int C_out, const int C_in,
      const int H_out, const int H_in,
      const int W_out, const int W_in) {
    const auto& X = Input(0);
    auto& filter = Inputs()[FILTER]->template Get<Tensor<CPUContext>>();
		if (InputSize() > 2) {
			auto& bias = Inputs()[BIAS]->template Get<Tensor<CPUContext>>();
		}
    auto* Y = Output(0);

    // We compile the kernels on the first run to get channel info embedded
    if (!loweringKernel_ || !gemmKernel_) {
      kernel_args_.emplace_back(("IN_CHANNEL"),  to_string(C_in));
      kernel_args_.emplace_back(("OUT_CHANNEL"), to_string(C_out));
      std::string arg_list = BuildArgumentList(kernel_args_);
      loweringKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kMECLowering, arg_list));
      gemmKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kMECGemm, arg_list));
    }

    auto& ctx = context_.GetSingleton();
    cl::Event event;

    // We transform and cache the filter. This is super slow...
    if (!filterBuffer_ || filterBuffer_->size() != filter.size()) {
      filterBuffer_ = caffe2::make_unique<TensorCL>(filter.dims());
      TensorCPU transposedFilter;
      transposedFilter.Resize(filter.dims());
      const int filterH_ = C_in * kernel_[1] * kernel_[0];
      const int filterW_ = C_out;
      // TODO(bwasti): get NT gemm (perf win there and here)
      // We currently store in the wrong way
      for (auto i = 0; i < filterH_; ++i) {
        for (auto j = 0; j < filterW_; ++j) {
          transposedFilter.template mutable_data<float>()[i * filterW_ + j] = 
            filter.template data<float>()[j * filterH_ + i];
        }
      }
      context_.CoercedCopy<T>(transposedFilter, *filterBuffer_);
      // Boilerplate code for copying things to image2d
      cl_int error;
      filterBufferImage_ = new cl::Image2D(ctx.context, CL_MEM_READ_WRITE,
          filter_image_format_, filterW_/4, filterH_, 0, NULL, &error);
      OPENCL_CHECK(error);
      cl::size_t<3> origin;
      cl::size_t<3> region;
      origin[0] = 0; origin[1] = 0; origin[2] = 0;
      region[0] = filterW_/4;
      region[1] = filterH_;
      region[2] = 1;
      size_t row_pitch;
      size_t slice_pitch;
      OPENCL_CHECK(ctx.queue.enqueueCopyBufferToImage(*(cl::Buffer*)filterBuffer_->template data<T>(),
          *filterBufferImage_,
          0,
          origin,
          region,
          NULL,
          &event));
      event.wait();
    }

    if (!mecBuffer_ || mecBuffer_->size() != W_out * H_in * kernel_[1] * C_in) {
      mecBuffer_ = caffe2::make_unique<TensorCL>(std::vector<int>{W_out, H_in, kernel_[1], C_in});
    }

    // Lowering step.
    OPENCL_CHECK(loweringKernel_->setArg(0, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(loweringKernel_->setArg(1, H_in));
    OPENCL_CHECK(loweringKernel_->setArg(2, H_out));
    OPENCL_CHECK(loweringKernel_->setArg(3, W_in));
    OPENCL_CHECK(loweringKernel_->setArg(4, W_out));
    OPENCL_CHECK(loweringKernel_->setArg(5, *(cl::Buffer*)mecBuffer_->template mutable_data<T>()));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *loweringKernel_,
          cl::NullRange,
          cl::NDRange(W_out, H_in, C_in >> 2),
          cl::NullRange,
          NULL,
          &event));

    // GEMM step.
    OPENCL_CHECK(gemmKernel_->setArg(0, *(cl::Buffer*)mecBuffer_->template data<T>()));
    OPENCL_CHECK(gemmKernel_->setArg(1, H_in * kernel_[0] * stride_[0] * C_in)); //lda
    OPENCL_CHECK(gemmKernel_->setArg(2, *(cl::Image2D*)filterBufferImage_));
    OPENCL_CHECK(gemmKernel_->setArg(3, *(cl::Buffer*)Y->template mutable_data<T>())); // output
    OPENCL_CHECK(gemmKernel_->setArg(4, C_out)); // ldc
    OPENCL_CHECK(gemmKernel_->setArg(5, W_out)); // M
    OPENCL_CHECK(gemmKernel_->setArg(6, C_out)); // N
    OPENCL_CHECK(gemmKernel_->setArg(7, kernel_[0] * kernel_[1] * C_in)); // K
    OPENCL_CHECK(gemmKernel_->setArg(8, H_out)); // iters TODO remove

    size_t maxWorkGroupSize;
    OPENCL_CHECK(gemmKernel_->getWorkGroupInfo(ctx.device, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSize));
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *gemmKernel_,
          cl::NullRange,
          cl::NDRange(C_out >> 2, W_out >> 3, H_out),
          cl::NullRange,
          NULL,
          &event));
    return true;
  }

  bool RunWithDirectConv(const int N,
      const int C_out, const int C_in,
      const int H_out, const int H_in,
      const int W_out, const int W_in) {
    const auto& X = Input(0);
    auto& filter = Inputs()[FILTER]->template Get<Tensor<CPUContext>>();
		if (InputSize() > 2) {
			auto& bias = Inputs()[BIAS]->template Get<Tensor<CPUContext>>();
		}
    auto* Y = Output(0);

    if (!directKernel_) {
      kernel_args_.emplace_back(("IN_CHANNEL"),  to_string(C_in));
      kernel_args_.emplace_back(("OUT_CHANNEL"), to_string(C_out));
      std::string arg_list = BuildArgumentList(kernel_args_);
      directKernel_ = make_unique<cl::Kernel>(context_.BuildKernel(kDirectConv, arg_list));
    }

    auto& ctx = context_.GetSingleton();
    cl::Event event;

    if (!filterBuffer_ || filterBuffer_->size() != filter.size()) {
      filterBuffer_ = caffe2::make_unique<TensorCL>(filter.dims());
      context_.CoercedCopy<T>(filter, *filterBuffer_);
      const int filterW_ = kernel_[0] * kernel_[1] * C_in;
      const int filterH_ = C_out;
      cl_int error;
      if (std::is_same<T, float>::value) {
        filterBufferImage_ = new cl::Image2D(ctx.context, CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_RGBA, CL_FLOAT), filterW_/4, filterH_, 0, NULL, &error);
      } else if (std::is_same<T, cl_half>::value) {
        filterBufferImage_ = new cl::Image2D(ctx.context, CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_RGBA, CL_HALF_FLOAT), filterW_/4, filterH_, 0, NULL, &error);
      }
      OPENCL_CHECK(error);
      cl::size_t<3> origin;
      cl::size_t<3> region;
      origin[0] = 0; origin[1] = 0; origin[2] = 0;
      region[0] = filterW_/4;
      region[1] = filterH_;
      region[2] = 1;
      size_t row_pitch;
      size_t slice_pitch;
      OPENCL_CHECK(ctx.queue.enqueueCopyBufferToImage(*(cl::Buffer*)filterBuffer_->template data<T>(),
          *filterBufferImage_,
          0,
          origin,
          region,
          NULL,
          &event));
      event.wait();
    }

    OPENCL_CHECK(directKernel_->setArg(0, *(cl::Image2D*)filterBufferImage_));
    OPENCL_CHECK(directKernel_->setArg(1, *(cl::Buffer*)X.template data<T>()));
    OPENCL_CHECK(directKernel_->setArg(2, H_in));
    OPENCL_CHECK(directKernel_->setArg(3, H_out));
    OPENCL_CHECK(directKernel_->setArg(4, W_in));
    OPENCL_CHECK(directKernel_->setArg(5, W_out));
    OPENCL_CHECK(directKernel_->setArg(6, *(cl::Buffer*)Y->template mutable_data<T>()));
    const int W_group_size = 8;//std::min(W_out >> 3, 16);
    const int H_group_size = 16;//std::min(H_out >> 2, 16);
    const int C_group_size = 4;//std::min(C_out, 4);
    OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
          *directKernel_,
          cl::NullRange,
          cl::NDRange(W_out >> 2, C_out >> 2, H_out), // This order should be tuned
          cl::NullRange,
          //cl::NDRange(W_group_size, C_group_size, H_group_size),
          NULL,
          &event));
    return true;
  }
};

} // namespace
} // namespace caffe2
#endif // CAFFE2_CONTRIB_OPENCL_OPERATORS_CONV_OP_H_
