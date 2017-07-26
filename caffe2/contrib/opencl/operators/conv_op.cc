#include "conv_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(Conv, ConvOp<float>);
REGISTER_OPENCL_OPERATOR(ConvHalf, ConvOp<cl_half>);

} // namespace
} // namespace caffe2
