#include "helper.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace onnx_caffe2 {

std::unordered_set<std::string> DummyName::used_names_;
size_t DummyName::counter_ = 0;

std::string DummyName::NewDummyName() {
  while (true) {
    std::string name = caffe2::MakeString("OC2_DUMMY_", counter_++);
    if (!used_names_.count(name)) {
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

}
