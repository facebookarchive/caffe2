#include "caffe2/operators/numpy_batch_matmul_op.h"
#include "caffe2/core/operator_schema.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(NumpyBatchMatMul, NumpyBatchMatMulOp<CPUContext>);

OPERATOR_SCHEMA(NumpyBatchMatMul)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Batch matrix multiplication with the same semantics as https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.

This supports arbitrary dimensional inputs (1- to N-D) and broadcasting batch dimensions.
)DOC")
    .Input(0, "A", "tensor of shape (dim0, dim1 ... M, K)")
    .Input(1, "B", "tensor of shpae (dim0, dim2 ... K, N)")
    .Output(0, "Y", "tensor of shape (dim0, dim1 ... M, N)")
    .Arg(
        "trans_a",
        "Pass 1 to transpose the last two dimensions of A before "
        "doing multiplication")
    .Arg(
        "trans_b",
        "Pass 1 to transpose the last two dimensions of B before "
        "doing multiplication");

} // namespace caffe2
