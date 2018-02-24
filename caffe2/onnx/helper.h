#pragma once

#include <set>
#include <string>
#include <unordered_set>

namespace onnx_caffe2 {
class DummyName {
  public:
    static std::string NewDummyName();

    static std::string Reset(const std::unordered_set<std::string>& used_names);

    static void AddName(const std::string& new_used) {
      used_names_.insert(new_used);
    }

   private:
    static std::unordered_set<std::string> used_names_;
    static size_t counter_;
};

}
