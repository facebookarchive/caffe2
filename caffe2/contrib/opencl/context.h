#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

#define CL_HPP_ENABLE_EXCEPTIONS 1
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "libopencl.h"
#include "CL/cl.hpp"

#include <mutex>

#define OPENCL_CHECK(v) do {\
    cl_int _err = (v); \
    if (_err != CL_SUCCESS) { \
      CAFFE_THROW("OpenCL Error:", _err, " on line ", __LINE__);\
    }\
  } while(0)

namespace caffe2 {

struct OpenCLContextSingleton {
 private:
  OpenCLContextSingleton();
  OpenCLContextSingleton(const OpenCLContextSingleton &) = delete;
  OpenCLContextSingleton(OpenCLContextSingleton&&) = delete;
 public:
  static OpenCLContextSingleton& getInstance();
  cl::Platform platform;
  cl::Device device;
  std::vector<cl::Device> devices;
  cl::Context context;
  cl::CommandQueue queue;
};

class OpenCLContext final {
 public:
  explicit OpenCLContext();
  explicit OpenCLContext(const DeviceOption& option) {
    DCHECK_EQ(option.device_type(), OPENCL);
  }
  ~OpenCLContext() {
  }

  static std::pair<void*, MemoryDeleter> New(size_t nbytes);
  static void Delete(void* data);

  void SwitchToDevice(int a, ...){
    auto& ctx = GetSingleton();
    CAFFE_ENFORCE(a < ctx.devices.size());
    ctx.device = ctx.devices[a];
  }

  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  bool FinishDeviceComputation() {
    auto& ctx = GetSingleton();
    ctx.queue.finish();
    return true;
  }
  cl::Kernel BuildKernel(const char* src, std::string additional_options = "", const char* fn_name = "K");

  template <class SrcContext, class DstContext>
  void CopyBytes(size_t nbytes, const void *src, void *dst);

  // For compatibility with old style copy
  template <typename T, class SrcContext, class DstContext>
  inline void Copy(size_t n, const T* src, T* dst) {
    if (std::is_fundamental<T>::value) {
      CopyBytes<SrcContext, DstContext>(n * sizeof(T),
                                     static_cast<const void*>(src),
                                     static_cast<void*>(dst));
    } else {
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  template <typename T_in, typename T_out, class SrcContext, class DstContext>
  inline void Copy(const Tensor<SrcContext>& src, Tensor<DstContext>& dst) {
    dst.Resize(src.dims());
    size_t n = src.size();
    if (std::is_same<T_in, T_out>::value) {
      if (std::is_fundamental<T_in>::value) {
        CopyBytes<SrcContext, DstContext>(n * sizeof(T_in),
                                       static_cast<const void*>(src.template data<T_in>()),
                                       static_cast<void*>(dst.template mutable_data<T_out>()));
      } else {
        for (int i = 0; i < n; ++i) {
          dst.template mutable_data<T_out>()[i] = src.template data<T_in>()[i];
        }
      }
    } else {
      CAFFE_THROW("This Copy requires specialization.");
    }
  }

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(const Tensor<SrcContext>& src, Tensor<DstContext>& dst) {
    Copy<T, T>(src, dst);
  }

  static struct OpenCLContextSingleton& GetSingleton();
};

typedef Tensor<OpenCLContext> TensorCL;
std::string BuildArgumentList(std::vector<std::pair<std::string, std::string>> args);

} // namespace caffe2

