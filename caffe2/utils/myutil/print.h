#ifndef PRINT_H
#define PRINT_H

#include "caffe2/core/net.h"

namespace caffe2 {

// Proto

template<typename T> void print(const std::vector<T> vector, const std::string &name = "") {
  if (name.length() > 0) std::cout << name << ": ";
  for (auto &v: vector) {
    std::cout << v << ' ';
  }
  if (name.length() > 0) std::cout << std::endl;
}

void print(const OperatorDef &def) {
  std::cout << "op {" << std::endl;
  for (const auto &input: def.input()) {
    std::cout << "  input: " << '"' << input << '"' << std::endl;
  }
  for (const auto &output: def.output()) {
    std::cout << "  output: " << '"' << output << '"' << std::endl;
  }
  std::cout << "  name: " << '"' << def.name() << '"' << std::endl;
  std::cout << "  type: " << '"' << def.type() << '"' << std::endl;
  if (def.arg_size()) {
    for (const auto &arg: def.arg()) {
      std::cout << "  arg {" << std::endl;
      std::cout << "    name: " << '"' << arg.name() << '"' << std::endl;
      if (arg.has_f()) std::cout << "    f: " << arg.f() << std::endl;
      if (arg.has_i()) std::cout << "    i: " << arg.i() << std::endl;
      if (arg.has_s()) std::cout << "    s: " << '"' << arg.s() << '"' << std::endl;

      if (arg.ints().size() < 10) for (const auto &v: arg.ints()) std::cout << "    ints: " << v << std::endl;
      else std::cout << "    ints: #" << arg.ints().size() << std::endl;
      if (arg.floats().size() < 10) for (const auto &v: arg.floats()) std::cout << "    floats: " << v << std::endl;
      else std::cout << "    floats: #" << arg.floats().size() << std::endl;
      if (arg.strings().size() < 10) for (const auto &v: arg.strings()) std::cout << "    strings: " << '"' << v << '"' << std::endl;
      else std::cout << "    strings: #" << arg.strings().size() << std::endl;
      std::cout << "  }" << std::endl;
    }
  }
  if (def.engine() != "") {
    std::cout << "  engine: " << '"' << def.engine() << '"' << std::endl;
  }
  if (def.has_device_option()) {
    std::cout << "device_option {" << std::endl;
    std::cout << "  device_type: " << '"' << def.device_option().device_type() << '"' << std::endl;
    std::cout << "  cuda_gpu_id: " << '"' << def.device_option().cuda_gpu_id() << '"' << std::endl;
    std::cout << "}" << std::endl;
  }
  if (def.is_gradient_op()) {
    std::cout << "  is_gradient_op: true" << std::endl;
  }
  std::cout << '}' << std::endl;
}

void print(const NetDef &def) {
  // To just dump the whole thing, use protobuf directly:
  // #include "google/protobuf/io/zero_copy_stream_impl.h"
  // #include "google/protobuf/text_format.h"
  // google::protobuf::io::OstreamOutputStream stream(&std::cout);
  // google::protobuf::TextFormat::Print(init_net, &stream);

  std::cout << "name: " << '"' << def.name() << '"' << std::endl;
  for (const auto &op: def.op()) {
    print(op);
  }
  if (def.has_device_option()) {
    std::cout << "device_option {" << std::endl;
    std::cout << "  device_type: " << '"' << def.device_option().device_type() << '"' << std::endl;
    std::cout << "  cuda_gpu_id: " << '"' << def.device_option().cuda_gpu_id() << '"' << std::endl;
    std::cout << "}" << std::endl;
  }
  if (def.has_num_workers()) {
    std::cout << "num_workers: " << def.num_workers() << std::endl;
  }
  for (const auto &input: def.external_input()) {
    std::cout << "external_input: " << '"' << input << '"' << std::endl;
  }
  for (const auto &output: def.external_output()) {
    std::cout << "external_output: " << '"' << output << '"' << std::endl;
  }
}

template<typename T, typename C>
void printType(const Tensor<C> &tensor, const std::string &name, int max) {
  const auto& data = tensor.template data<T>();
  if (name.length() > 0) std::cout << name << "(" << tensor.dims() << "): ";
  for (auto i = 0; i < (tensor.size() > max ? max : tensor.size()); ++i) {
    std::cout << (float)data[i] << ' ';
  }
  if (tensor.size() > max) {
    std::cout << "... (" << *std::min_element(data, data + tensor.size()) << "," << *std::max_element(data, data + tensor.size()) << ")";
  }
  if (name.length() > 0) std::cout << std::endl;
}

template<typename C>
void print(const Tensor<C> &tensor, const std::string &name = "", int max = 100) {
  if (tensor.template IsType<float>()) {
    return printType<float>(tensor, name, max);
  }
  if (tensor.template IsType<int>()) {
    return printType<int>(tensor, name, max);
  }
  if (tensor.template IsType<uint8_t>()) {
    return printType<uint8_t>(tensor, name, max);
  }
  if (tensor.template IsType<int8_t>()) {
    return printType<int8_t>(tensor, name, max);
  }
  std::cout << name << "?" << std::endl;
}

void print(const Blob &blob, const std::string &name = "") {
  print(blob.Get<Tensor<CPUContext>>(), name);
}

template<typename C>
void printBest(const Tensor<C> &tensor, const char **classes, const std::string &name = "") {
    // sort top results
  const auto &probs = tensor.template data<float>();
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0; i < tensor.size(); i++) {
    if (probs[i] > 0.01) {
      pairs.push_back(std::make_pair(probs[i] * 100, i));
    }
  }
  std:sort(pairs.begin(), pairs.end());

  // show results
  if (name.length() > 0) std::cout << name << ": " << std::endl;
  for (auto pair: pairs) {
    std::cout << pair.first << "% '" << classes[pair.second] << "' (" << pair.second << ")" << std::endl;
  }
}


// Join

template<typename T>
std::string join_values(const T &values, const std::string &separator = " ", const std::string &prefix = "[", const std::string &suffix = "]", int collapse = 64) {
  std::stringstream stream;
  if (values.size() > collapse) {
    stream << "[#" << values.size() << "]";
  } else {
    bool is_next = false;
    for (const auto &v: values) {
      if (is_next) {
        stream << separator;
      } else {
        stream << prefix;
        is_next = true;
      }
      stream << v;
    }
    if (is_next) {
        stream << suffix;
    }
  }
  return stream.str();
}

std::string join_op(const OperatorDef &def) {
  std::stringstream stream;
  stream << "op: " << def.type();
  if (def.name().size()) {
    stream << " " << '"' << def.name() << '"';
  }
  if (def.input_size() || def.output_size()) {
    stream << " " << join_values(def.input(), " ", "(", ")") << "->" << join_values(def.output(), " ", "(", ")");
  }
  if (def.arg_size()) {
    for (const auto &arg: def.arg()) {
      stream << " " << arg.name() << ":";
      if (arg.has_f()) {
        stream << arg.f();
      } else if (arg.has_i()) {
        stream << arg.i();
      } else if (arg.has_s()) {
        stream << arg.s();
      } else {
        stream << join_values(arg.ints());
        stream << join_values(arg.floats());
        stream << join_values(arg.strings());
      }
    }
  }
  if (def.has_engine()) {
    stream << " engine:" << def.engine();
  }
  if (def.has_device_option() && def.device_option().has_device_type()) {
    stream << " device_type:" << def.device_option().device_type();
  }
  if (def.has_device_option() && def.device_option().has_cuda_gpu_id()) {
    stream << " cuda_gpu_id:" << def.device_option().cuda_gpu_id();
  }
  if (def.has_is_gradient_op()) {
    stream << " is_gradient_op:true";
  }
  stream << std::endl;
  return stream.str();
}

std::string join_net(const NetDef &def) {
  std::stringstream stream;
  stream << "net: ------------- " << def.name() << " -------------" << std::endl;
  for (const auto &op: def.op()) {
    stream << join_op(op);
  }
  if (def.has_device_option() && def.device_option().has_device_type()) {
    stream << "device_type:" << def.device_option().device_type() << std::endl;
  }
  if (def.has_device_option() && def.device_option().has_cuda_gpu_id()) {
    stream << "cuda_gpu_id:" << def.device_option().cuda_gpu_id() << std::endl;
  }
  if (def.has_num_workers()) {
    stream << "num_workers: " << def.num_workers() << std::endl;
  }
  if (def.external_input_size()) {
    stream << "external_input: " << join_values(def.external_input()) << std::endl;
  }
  if (def.external_output_size()) {
    stream << "external_output: " << join_values(def.external_output()) << std::endl;
  }
  return stream.str();
}


}  // namespace caffe2

// TensorProto_DataType_UNDEFINED 0
// TensorProto_DataType_FLOAT     1
// TensorProto_DataType_INT32     2
// TensorProto_DataType_BYTE      3
// TensorProto_DataType_STRING    4
// TensorProto_DataType_BOOL      5
// TensorProto_DataType_UINT8     6
// TensorProto_DataType_INT8      7
// TensorProto_DataType_UINT16    8
// TensorProto_DataType_INT16     9
// TensorProto_DataType_INT64    10
// TensorProto_DataType_FLOAT16  12
// TensorProto_DataType_DOUBLE   13

#endif  // PRINT_H
