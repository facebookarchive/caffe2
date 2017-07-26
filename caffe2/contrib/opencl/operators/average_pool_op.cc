#include "average_pool_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(AveragePool, AveragePoolOp<float, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(AveragePoolHalf, AveragePoolOp<cl_half, OpenCLContext>);

} // namespace
} // namespace caffe2



