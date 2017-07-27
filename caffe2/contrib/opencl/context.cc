#include "context.h"

#include "caffe2/core/context.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(Tensor<OpenCLContext>);

OpenCLContextSingleton::OpenCLContextSingleton() {
  const auto platform_id = 0;
  const auto device_id = 0;

  auto platforms = std::vector<cl::Platform>();
  OPENCL_CHECK(cl::Platform::get(&platforms));
  if (platforms.size() == 0 || platform_id >= platforms.size()) {
    CAFFE_THROW("Cannot find platform for OpenCL.");
  }
  platform = platforms[platform_id];

  devices = std::vector<cl::Device>();
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.size() == 0 || device_id >= devices.size()) {
    CAFFE_THROW("Cannot find OpenCL compatible device.");
  }
  device = devices[device_id];

  context  = cl::Context({device});
  queue = cl::CommandQueue(context, device);
}

OpenCLContextSingleton& OpenCLContextSingleton::getInstance() {
  static OpenCLContextSingleton* instance;
  if (instance == nullptr) {
    instance = new OpenCLContextSingleton();
  }
  return *instance;
}

std::pair<void*, MemoryDeleter> OpenCLContext::New(size_t nbytes) {
  auto& ctx = GetSingleton();
  cl_int err = 0;

  cl::Buffer* buffer = new cl::Buffer(ctx.context, CL_MEM_READ_WRITE,
      nbytes, nullptr, &err);
  OPENCL_CHECK(err);
  return {(void *)buffer, Delete};
}

template <>
void OpenCLContext::CopyBytes<OpenCLContext, CPUContext>(size_t nbytes, const void *src, void *dst) {
  auto& ctx = GetSingleton();
  OPENCL_CHECK(cl::copy(ctx.queue, *((cl::Buffer*)src), static_cast<char*>(dst), static_cast<char*>(dst) + nbytes));
  OPENCL_CHECK(cl::finish());
}

template <>
void OpenCLContext::CopyBytes<CPUContext, OpenCLContext>(size_t nbytes, const void *src, void *dst) {
  auto& ctx = GetSingleton();
  //OPENCL_CHECK(cl::copy(ctx.queue,
  //  static_cast<const char*>(src),
  //  static_cast<const char*>(src) + nbytes,
  //  *((cl::Buffer*)(dst))
  //));
	cl::Event event;
  OPENCL_CHECK(cl::flush());
	OPENCL_CHECK(
			ctx.queue.enqueueWriteBuffer(
				*((cl::Buffer*)(dst)),
				CL_TRUE,
				0,
				nbytes,
				src,
				nullptr,
				&event));
	OPENCL_CHECK(event.wait());
}

void OpenCLContext::Delete(void *ptr) {
  delete (cl::Buffer *)ptr;
}

struct OpenCLContextSingleton& OpenCLContext::GetSingleton() {
  return OpenCLContextSingleton::getInstance();
}

cl::Kernel OpenCLContext::BuildKernel(const char* src, std::string additional_options, const char* fn_name) {
  auto& ctx = GetSingleton();

  cl::Program::Sources source(1,
      std::make_pair(src, strlen(src)));

  cl_int err = CL_SUCCESS;
  cl::Program p = cl::Program(ctx.context, source, &err);
  OPENCL_CHECK(err);

  std::string options = "-cl-std=CL1.1 -cl-fast-relaxed-math -cl-single-precision-constant";
  options += additional_options;

  // TODO support more than one device
  // this will involve checking a compiler exists on each device
  vector<cl::Device> devices_{ctx.device};
  err = p.build(devices_, options.c_str());
  cl_build_status build_status = p.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(ctx.device);
  if (err != CL_SUCCESS || build_status != CL_BUILD_SUCCESS) {
    auto str = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device);
    LOG(ERROR) << "Error code: " << err << " Build status: " << build_status;
    CAFFE_THROW(str);
  }

  auto kernel = cl::Kernel(p, fn_name, &err);
  OPENCL_CHECK(err);
  return kernel;
}

std::string BuildArgumentList(std::vector<std::pair<std::string, std::string>> args) {
  std::string out = " "; // There may be args before this
  for (auto arg : args) {
    out += "-D " + arg.first + "=" + arg.second + " ";
  }
  return out;
}

template<>
void OpenCLContext::CoercedCopy<cl_half>(const Tensor<CPUContext>& src, Tensor<OpenCLContext>& dst) {
  auto tmpBuffer = caffe2::make_unique<TensorCL>(src.dims());
  Copy<float>(src, *tmpBuffer);

  auto& ctx = GetSingleton();
  if (!ctx.toHalfKernel_) {
    ctx.toHalfKernel_ = make_unique<cl::Kernel>(OpenCLContext::BuildKernel(kFloatToHalf));
  }
  OPENCL_CHECK(ctx.toHalfKernel_->setArg(0, *(cl::Buffer*)tmpBuffer->data<float>()));
  OPENCL_CHECK(ctx.toHalfKernel_->setArg(1, *(cl::Buffer*)dst.mutable_data<cl_half>()));

  cl::Event event;
  OPENCL_CHECK(ctx.queue.enqueueNDRangeKernel(
        *ctx.toHalfKernel_,
        cl::NullRange,
        cl::NDRange(src.size()),
        cl::NullRange,
        NULL,
        &event));
  event.wait();
}

template<>
void OpenCLContext::CoercedCopy<float>(const Tensor<CPUContext>& src, Tensor<OpenCLContext>& dst) {
  //dst.mutable_data<float>();
  Copy<float>(src, dst);
  OPENCL_CHECK(cl::flush());
}

}
