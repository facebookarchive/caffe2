#ifndef CUDA_H
#define CUDA_H

#ifdef WITH_CUDA
  #include "caffe2/core/context_gpu.h"
#endif

namespace caffe2 {

#ifdef WITH_CUDA
static const int cuda_multipier = 10;
#else
static const int cuda_multipier = 1;
#endif

bool setupCUDA() {
  DeviceOption option;
  option.set_device_type(CUDA);
#ifdef WITH_CUDA
  new CUDAContext(option);
  std::cout << std::endl << "using CUDA" << std::endl;
  return true;
#else
  return false;
#endif
}

void set_device_cuda_model(NetDef &model) {
#ifdef WITH_CUDA
  model.mutable_device_option()->set_device_type(CUDA);
#endif
}

TensorCPU get_tensor_blob(const Blob &blob) {
#ifdef WITH_CUDA
  if (blob.IsType<TensorCUDA>()) {
    return TensorCPU(blob.Get<TensorCUDA>());
  }
#endif
  return blob.Get<TensorCPU>();
}

void set_tensor_blob(Blob &blob, const TensorCPU &value) {
#ifdef WITH_CUDA
  if (blob.IsType<TensorCUDA>()) {
    auto tensor = blob.GetMutable<TensorCUDA>();
    tensor->ResizeLike(value);
    tensor->ShareData(value);
    return;
  }
#endif
  auto tensor = blob.GetMutable<TensorCPU>();
  tensor->ResizeLike(value);
  tensor->ShareData(value);
}

}  // namespace caffe2

#endif  // CUDA_H
