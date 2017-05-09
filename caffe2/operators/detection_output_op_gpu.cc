#include "caffe2/operators/detection_output_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
REGISTER_CUDA_OPERATOR(DetectionOutput, DetectionOutputOp<float, CUDAContext>);
} // namespace

} // namespace caffe2
