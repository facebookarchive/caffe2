
#include "caffe2/mkl/operators/operator_fallback_mkl.h"
#include "caffe2/operators/softmax_op.h"


//can add more non-MKL operators if needed
namespace caffe2 {

REGISTER_MKL_OPERATOR(Softmax, mkl::MKLFallbackOp<SoftmaxOp<float, CPUContext>, SkipIndices<0>>);

}  // namespace caffe2




