#pragma once

#include <functional>
#include <string>

namespace onnx_caffe2 {

enum class DeviceType {CPU=0, CUDA=1};

struct Device {
  Device(const std::string& spec);
  DeviceType type;
  int device_id{-1};
};

}

namespace std {
template <> struct hash<onnx_caffe2::DeviceType> {
  std::size_t operator()(const onnx_caffe2::DeviceType &k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std
