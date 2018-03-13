// Note(jiayq): the import_array function is done inside
// caffe2_python.cc. Read
// http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.

#define NO_IMPORT_ARRAY

#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "caffe2/trt/tensorrt_tranformer.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace caffe2 {
namespace python {

REGISTER_CUDA_OPERATOR(Python, GPUFallbackOp<PythonOp<CPUContext, false>>);
REGISTER_CUDA_OPERATOR(
    PythonGradient,
    GPUFallbackOp<PythonGradientOp<CPUContext, false>>);

REGISTER_CUDA_OPERATOR(PythonDLPack, PythonOp<CUDAContext, true>);
REGISTER_CUDA_OPERATOR(
    PythonDLPackGradient,
    PythonGradientOp<CUDAContext, true>);

REGISTER_BLOB_FETCHER((TypeMeta::Id<TensorCUDA>()), TensorFetcher<CUDAContext>);
REGISTER_BLOB_FEEDER(CUDA, TensorFeeder<CUDAContext>);

namespace py = pybind11;

static bool ParseProtobufFromLargeString(const string& str, Message* proto) {
  ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  // Set PlanDef message size limit to 1G.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

void addCUDAGlobalMethods(py::module& m) {
  m.def("num_cuda_devices", &NumCudaDevices);
  m.def("get_cuda_version", &CudaVersion);
  m.def("get_cudnn_version", &cudnnCompiledVersion);
  m.def("get_cuda_peer_access_pattern", []() {
    std::vector<std::vector<bool>> pattern;
    CAFFE_ENFORCE(caffe2::GetCudaPeerAccessPattern(&pattern));
    return pattern;
  });
  m.def("get_device_properties", [](int deviceid) {
    auto& prop = GetDeviceProperty(deviceid);
    std::map<std::string, py::object> obj;
    obj["name"] = py::cast(prop.name);
    obj["major"] = py::cast(prop.major);
    obj["minor"] = py::cast(prop.minor);
    obj["totalGlobalMem"] = py::cast(prop.totalGlobalMem);
    return obj;
  });
  m.def(
      "onnx_to_trt_op",
      [](const py::bytes& onnx_model_str,
         const std::unordered_map<std::string, std::vector<int>>&
             output_size_hints) -> py::bytes {
        TensorRTTransformer t;
        auto op_def =
            t.BuildTrtOp(onnx_model_str.cast<std::string>(), output_size_hints);
        std::string out;
        op_def.SerializeToString(&out);
        return py::bytes(out);
      });
  m.def(
      "transform_trt",
      [](const py::bytes& init_net_str,
         const py::bytes& pred_net_str,
         const std::unordered_map<std::string, std::vector<int>>& shapes)
          -> std::vector<py::bytes> {
        caffe2::NetDef init_net;
        if(!ParseProtobufFromLargeString(
            init_net_str.cast<std::string>(), &init_net)) {
          LOG(ERROR) << "broken init_net protobuf";
        }
        caffe2::NetDef pred_net;
        if(!ParseProtobufFromLargeString(
            pred_net_str.cast<std::string>(), &pred_net)) {
          LOG(ERROR) << "broken pred_net protobuf";
        }
        std::unordered_map<std::string, TensorShape> tensor_shapes;
        for (const auto& it: shapes) {
          tensor_shapes.emplace(
              it.first, CreateTensorShape(it.second, TensorProto::FLOAT));
        }
        TensorRTTransformer ts;
        //ts.LoadNets(init_net, pred_net);
        ts.TransformSimple(&init_net, &pred_net, tensor_shapes);
        std::string init_net_str2;
        std::string pred_net_str2;
        init_net.SerializeToString(&init_net_str2);
        pred_net.SerializeToString(&pred_net_str2);
        return {py::bytes(init_net_str2), py::bytes(pred_net_str2)};
      });
};

void addCUDAObjectMethods(py::module& m) {
  py::class_<DLPackWrapper<CUDAContext>>(m, "DLPackTensorCUDA")
      .def_property_readonly(
          "data",
          [](DLPackWrapper<CUDAContext>* t) -> py::object {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                CUDA,
                "Expected CUDA device option for CUDA tensor");

            return t->data();
          },
          "Return DLPack tensor with tensor's data.")
      .def(
          "feed",
          [](DLPackWrapper<CUDAContext>* t, py::object obj) {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                CUDA,
                "Expected CUDA device option for CUDA tensor");
            t->feed(obj);
          },
          "Copy data from given DLPack tensor into this tensor.")
      .def_property_readonly(
          "_shape",
          [](const DLPackWrapper<CUDAContext>& t) { return t.tensor->dims(); })
      .def(
          "_reshape",
          [](DLPackWrapper<CUDAContext>* t, std::vector<TIndex> dims) {
            t->tensor->Resize(dims);
          });
}

PYBIND11_MODULE(caffe2_pybind11_state_gpu, m) {
  m.doc() = "pybind11 stateful interface to Caffe2 workspaces - GPU edition";

  addGlobalMethods(m);
  addCUDAGlobalMethods(m);
  addObjectMethods(m);
  addCUDAObjectMethods(m);
}
} // namespace python
} // namespace caffe2
