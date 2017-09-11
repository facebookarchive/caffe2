#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/recurrent_network_op.h"

namespace caffe2 {

namespace detail {

template <typename T, typename Context>
void initializeRecurrentInput(
    const RecurrentInput& rc,
    int32_t seqLen,
    int32_t batchSize,
    Workspace* ws,
    Context* context);

namespace {

template <typename T>
__global__
void initRecurrentInput_kernel(
    size_t stateSize,
    const T* input,
    T* state) {
  // index into appropriate target buffer
  const int block_id = blockIdx.x;
  T* state_local = state + block_id*stateSize;

  // copy
  for (int idx=threadIdx.x; idx < stateSize; idx+=blockDim.x) {
    state_local[idx] = input[idx];
  }
}


}; // namespace

template <>
void initializeRecurrentInput<float,CUDAContext>(
    const RecurrentInput& rc,
    int32_t seqLen,
    int32_t batchSize,
    Workspace* ws,
    CUDAContext* context) {
  auto stateBlob = ws->GetBlob(rc.state);
  CAFFE_ENFORCE(stateBlob);
  auto* state = stateBlob->GetMutable<Tensor<CUDAContext>>();

  auto inputBlob = ws->GetBlob(rc.input);
  CAFFE_ENFORCE(inputBlob);
  const auto& input = inputBlob->Get<Tensor<CUDAContext>>();
  CAFFE_ENFORCE_GE(input.ndim(), 1, rc.input);
  CAFFE_ENFORCE_LE(input.ndim(), 3, rc.input);

  const auto stateSize = input.dim(input.ndim() - 1);
  // States at [0, ..., T] (inclusive)
  state->Resize(seqLen + 1, batchSize, stateSize);

  if (input.ndim() == 3) {
    CAFFE_ENFORCE_EQ(input.dim(0), 1, rc.input);
  }
  if (input.ndim() >= 2) {
    CAFFE_ENFORCE_EQ(input.dim(input.ndim() - 2), batchSize, rc.input);
    context->Copy<float, CUDAContext, CUDAContext>(
        batchSize * stateSize,
        input.data<float>(),
        state->mutable_data<float>());
  } else {
    initRecurrentInput_kernel<float><<<batchSize, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
        stateSize,
        input.data<float>(),
        state->mutable_data<float>());
  }
}
template <>
void initializeRecurrentInput<float16,CUDAContext>(
    const RecurrentInput& rc,
    int32_t seqLen,
    int32_t batchSize,
    Workspace* ws,
    CUDAContext* context) {
  auto stateBlob = ws->GetBlob(rc.state);
  CAFFE_ENFORCE(stateBlob);
  auto* state = stateBlob->GetMutable<Tensor<CUDAContext>>();

  auto inputBlob = ws->GetBlob(rc.input);
  CAFFE_ENFORCE(inputBlob);
  const auto& input = inputBlob->Get<Tensor<CUDAContext>>();
  CAFFE_ENFORCE_GE(input.ndim(), 1, rc.input);
  CAFFE_ENFORCE_LE(input.ndim(), 3, rc.input);

  const auto stateSize = input.dim(input.ndim() - 1);
  // States at [0, ..., T] (inclusive)
  state->Resize(seqLen + 1, batchSize, stateSize);

  if (input.ndim() == 3) {
    CAFFE_ENFORCE_EQ(input.dim(0), 1, rc.input);
  }
  if (input.ndim() >= 2) {
    CAFFE_ENFORCE_EQ(input.dim(input.ndim() - 2), batchSize, rc.input);
    context->Copy<float16, CUDAContext, CUDAContext>(
        batchSize * stateSize,
        input.data<float16>(),
        state->mutable_data<float16>());
  } else {
    initRecurrentInput_kernel<float16><<<batchSize, CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
        stateSize,
        input.data<float16>(),
        state->mutable_data<float16>());
  }
}

}; // namespace detail

template <>
bool RecurrentNetworkOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

template <>
bool RecurrentNetworkGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

template <>
bool AccumulateInputGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(1));
}

template <>
bool RNNApplyLinkOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(1));
}

REGISTER_CUDA_OPERATOR(
    RecurrentNetwork,
    RecurrentNetworkOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RecurrentNetworkGradient,
    RecurrentNetworkGradientOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    rnn_internal_accumulate_gradient_input,
    AccumulateInputGradientOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    rnn_internal_apply_link,
    RNNApplyLinkOp<CUDAContext>);


} // namespace caffe2
