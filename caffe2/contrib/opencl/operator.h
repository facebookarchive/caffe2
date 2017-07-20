#ifndef CAFFE2_CONTRIB_OPENCL_OPERATOR_H_
#define CAFFE2_CONTRIB_OPENCL_OPERATOR_H_

#include "caffe2/core/common.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

CAFFE_DECLARE_REGISTRY(
    OpenCLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_OPENCL_OPERATOR_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(OpenCLOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_OPENCL_OPERATOR(name, ...) \
  CAFFE_REGISTER_CLASS(OpenCLOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_OPENCL_OPERATOR_STR(str_name, ...) \
  CAFFE_REGISTER_TYPED_CLASS(OpenCLOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_OPENCL_OPERATOR_WITH_ENGINE(name, engine, ...) \
  CAFFE_REGISTER_CLASS(OpenCLOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

} // namespace caffe2

#endif // CAFFE2_CONTRIB_OPENCL_OPERATOR_H_
