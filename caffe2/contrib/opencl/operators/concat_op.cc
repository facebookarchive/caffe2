#include "concat_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(Concat, ConcatOp<float, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(ConcatHalf, ConcatOp<cl_half, OpenCLContext>);

} // namespace
} // namespace caffe2


