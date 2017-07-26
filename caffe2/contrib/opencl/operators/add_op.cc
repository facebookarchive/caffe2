#include "add_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(Add, AddOp<float, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(AddHalf, AddOp<cl_half, OpenCLContext>);

} // namespace
} // namespace caffe2

