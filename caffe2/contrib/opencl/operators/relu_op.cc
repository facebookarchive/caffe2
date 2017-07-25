#include "relu_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(Relu, ReluOp<float, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(ReluHalf, ReluOp<cl_half, OpenCLContext>);

} // namespace
} // namespace caffe2
