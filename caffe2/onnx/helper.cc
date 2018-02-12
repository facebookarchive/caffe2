#include "helper.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace onnx_caffe2 {

std::unordered_set<std::string> DummyName::used_names_;
size_t DummyName::counter_ = 0;

std::string DummyName::NewDummyName() {
  while (true) {
    std::string name = caffe2::MakeString("OC2_DUMMY_", counter_++);
    if (not used_names_.count(name)) {
      used_names_.insert(name);
      return name;
    }
  }
}

std::string
DummyName::Reset(const std::unordered_set<std::string> &used_names) {
  used_names_ = used_names;
  counter_ = 0;
  return "";
}

bool Caffe2RegisteredOps::inited_ = false;
std::set<std::string> Caffe2RegisteredOps::ops_;

void Caffe2RegisteredOps::Init() {
  // CPU operators
  for (const auto &name : caffe2::CPUOperatorRegistry()->Keys()) {
    ops_.insert(name);
  }
  // CUDA operators
  for (const auto &name : caffe2::CUDAOperatorRegistry()->Keys()) {
    ops_.insert(name);
  }

  inited_ = true;
}


bool Caffe2RegisteredOps::IsOperator(const std::string& op_type) {
  // pull in all the operators upon first invocation
  if (not inited_) {
    Init();
  }

  return ops_.count(caffe2::OpRegistryKey(op_type, "DEFAULT"));
}

}
