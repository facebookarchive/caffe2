#pragma once

#include "caffe2/proto/caffe2.pb.h"
#include "onnx/onnx_pb.h"
#include "device.h"
#include "backend_rep.h"

#include <google/protobuf/text_format.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace onnx_caffe2 {

struct Caffe2Ops {
  ::google::protobuf::RepeatedPtrField<caffe2::OperatorDef> init_ops;
  ::google::protobuf::RepeatedPtrField<caffe2::OperatorDef> ops;
  ::google::protobuf::RepeatedPtrField<std::string> interface_blobs;
};

// A convenient class to query attributes of a NodeProto. Note that the
// NodeProto can not be modified during the query of OnnxAttributes object
class OnnxAttributes {
public:
  OnnxAttributes(const onnx::NodeProto &node);

  bool HasAttribute(const std::string& key) const {
    return onnx_attrs_.count(key);
  }

  onnx::AttributeProto* AddRewrittenAttibute(const std::string& key) {
    auto tmp = rewritten_onnx_attrs_.emplace(key, onnx::AttributeProto());
    auto& attr = tmp.first->second;
    attr.set_name(key);
    return &attr;
  }

  ::google::protobuf::RepeatedPtrField<caffe2::Argument>
  OnnxAttrToCaffe2Arg(std::function<std::string(const std::string &)> mapper) const;

  // Get attribute given attribute name, specialied on data type T. Note that
  // the return value is copied
  template <typename T> T get(const std::string &key) const;

  template <typename T>
  T get(const std::string& key, const T& default_value) const {
    if (onnx_attrs_.count(key)) {
      return get<T>(key);
    } else {
      return default_value;
    }
  }

private:
  std::unordered_map<std::string, const onnx::AttributeProto*> onnx_attrs_;
  std::unordered_map<std::string, onnx::AttributeProto> rewritten_onnx_attrs_;
};

template <> int64_t OnnxAttributes::get(const std::string &key) const;
template <> float OnnxAttributes::get(const std::string &key) const;

template <>
::google::protobuf::RepeatedPtrField<std::string>
OnnxAttributes::get(const std::string &key) const;

template <>
::google::protobuf::RepeatedField<::google::protobuf::int64>
OnnxAttributes::get(const std::string &key) const;

template <>
const onnx::TensorProto *OnnxAttributes::get(const std::string &key) const;

// convenient class for onnx node
struct OnnxNode {
  OnnxNode(const onnx::NodeProto& node_in):node(node_in), attributes(node_in) {}

  const onnx::NodeProto& node;

  OnnxAttributes attributes;
};

class Caffe2Backend {
public:
  Caffe2BackendRep* Prepare(const std::string &onnx_model_str,
                           const std::string &device,
                           const std::vector<Caffe2Ops> &extras);

  Caffe2Ops ConvertNode(const std::string& node_str, int opset_version);
private:
  using SpecialOpConverter = Caffe2Ops (Caffe2Backend::*)(
      const onnx::ModelProto &, const onnx::ModelProto &, OnnxNode *, int);

  void OnnxToCaffe2(caffe2::NetDef *init_net, caffe2::NetDef *pred_net,
                    const onnx::ModelProto &onnx_model,
                    const std::string &device, int opset_version,
                    bool include_initializers,
                    const std::vector<Caffe2Ops> &extras);

  Caffe2Ops OnnxNodeToCaffe2Ops(const onnx::ModelProto &init_model,
                                const onnx::ModelProto &pred_model,
                                OnnxNode *onnx_node, int opset_version);

  std::unordered_set<std::string>
  AllNamesInGraph(const onnx::GraphProto &graph);

  void InplaceRewrite(onnx::GraphProto *graph);

  std::unordered_map<std::string, std::string>
  InplaceRewrite(::google::protobuf::RepeatedPtrField<onnx::NodeProto> *nodes);

  void BuildTensorFillingOp(caffe2::OperatorDef *c2_op,
                            const onnx::TensorProto &onnx_tensor,
                            const std::string &name = "");

  Caffe2Ops CommonOnnxNodeToCaffe2Ops(const onnx::ModelProto &init_model,
                                     const onnx::ModelProto &pred_model,
                                     OnnxNode *onnx_node,
                                     int opset_version);

  Caffe2Ops CreateConstant(const onnx::ModelProto &init_model,
                           const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                           int opset_version);

  Caffe2Ops CreateConvePoolOpBase(const onnx::ModelProto &init_model,
                                  const onnx::ModelProto &pred_model,
                                  OnnxNode *onnx_node, int opset_version);

  Caffe2Ops CreateReshape(const onnx::ModelProto &init_model,
                          const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                          int opset_version);

  Caffe2Ops CreateGather(const onnx::ModelProto &init_model,
                         const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                         int opset_version);

  Caffe2Ops CreateGemm(const onnx::ModelProto &init_model,
                       const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                       int opset_version);

  Caffe2Ops CreatePad(const onnx::ModelProto &init_model,
                      const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                      int opset_version);

  Caffe2Ops CreateConcat(const onnx::ModelProto &init_model,
                         const onnx::ModelProto &pred_model,
                         OnnxNode *onnx_node, int opset_version);

  Caffe2Ops CreateLogSoftmax(const onnx::ModelProto &init_model,
                             const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                             int opset_version);

  Caffe2Ops CreateSlice(const onnx::ModelProto &init_model,
                        const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                        int opset_version);

  Caffe2Ops CreateSqrt(const onnx::ModelProto &init_model,
                        const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                        int opset_version);

  Caffe2Ops CreateReciprocal(const onnx::ModelProto &init_model,
                        const onnx::ModelProto &pred_model, OnnxNode *onnx_node,
                        int opset_version);

  const static std::unordered_set<std::string> kRNNOperators_;
  const static std::unordered_map<std::string, int> kBrokenOperators_;
  const static std::unordered_map<std::string, std::string> kRenamedOperators_;
  const static std::unordered_map<std::string, std::string> kRenamedAttrs_;
  const static std::unordered_map<std::string,
                                  std::unordered_map<std::string, std::string>>
      kPerOpRenamedAttrs_;
  const static std::unordered_map<std::string, Caffe2Backend::SpecialOpConverter>
      kSpecialOperators_;
};

}
