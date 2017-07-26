#include "spatial_batch_norm_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(SpatialBN, SpatialBNOp<float, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(SpatialBNHalf, SpatialBNOp<cl_half, OpenCLContext>);

} // namespace
} // namespace caffe2
