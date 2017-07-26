#include "channel_shuffle_op.h"
#include "operator.h"

namespace caffe2 {
namespace {

REGISTER_OPENCL_OPERATOR(ChannelShuffle, ChannelShuffleOp<float, OpenCLContext>);
REGISTER_OPENCL_OPERATOR(ChannelShuffleHalf, ChannelShuffleOp<cl_half, OpenCLContext>);

} // namespace
} // namespace caffe2

