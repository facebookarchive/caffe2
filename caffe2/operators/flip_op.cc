#include "caffe2/operators/flip_op.h"

namespace caffe2 {

  template <>
  template <typename T>
  bool FlipOp<CPUContext>::DoRunWithType() {
    const auto& input = Input(0);
    auto* output = Output(0);
    size_t count = input.size();
    int num_axes = axes_.size();
    CAFFE_ENFORCE(OperatorBase::HasArgument("axes"), "argument axes is missing");
    const T* from_data = input.template data<T>();
    T* to_data = output->template mutable_data<T>();
    auto in_dims = input.dims();

    // Measure amount of contiguous data we can copy at once
    // Suppose input.dims()=(N,C,H,W),
    //   if axes=(1,) or (0,1) then blocksize = H * W
    //   if axes=(2,) or (1,2) or (0,1,2) then blocksize = W
    //   if axes=(3,) or (2,3) or (1,2,3) or (0,1,2,3) then blocksize = 1
    // Calculate stride
    //   if axes=(1,) or (1,2) or (1,2,3) then stride = C or C * H or C * H * W
    //   if axes=(2,) or (2,3) then stride = H or H * W
    //   if axes=(3,) then stride = W
    TIndex blocksize = 1;
    TIndex stride = 1;
    for (int i = input.ndim() - 1; i >= 0; --i) {
      if (axes_[num_axes - 1] < i) {
        blocksize *= in_dims[i];
      }
      else if (axes_[0] <= i) {
        stride *= in_dims[i];
      }
      else {
        break;
      }
    }

    // Now, for every stride, reverse data in blocksize
    // Branch here to avoid branching within the loop
    if (blocksize > 1) {
      for (size_t index = 0; index < (count / blocksize); index += stride) {
        for (size_t i = 0; i < stride; i++) {
          memcpy(
            to_data + blocksize * (index + i),
            from_data + blocksize * (index + stride - 1 - i),
            blocksize * sizeof(T));
        }
      }
    }
    else {
      for (size_t index = 0; index < count; index += stride) {
        for (size_t i = 0; i < stride; i++) {
          *(to_data + index + i) = *(from_data + index + stride - 1 - i);
        }
      }
    }

    return true;
  }

  namespace {
    REGISTER_CPU_OPERATOR(Flip, FlipOp<CPUContext>);

    OPERATOR_SCHEMA(Flip)
      .NumInputs(1)
      .NumOutputs(1)
      .IdenticalTypeAndShapeOfInput(0)
      .SetDoc(R"DOC(
Flip the input tensor similar to numpy.flip. For example, when axes=(3,), 
given an input tensor M of shape (N, C, H, W), the output will be 
similar as numpy.flip(M, 3). And when axes=(2,), the output will be
similar as numpy.flip(M, 2).
)DOC")
      .Arg(
        "axes",
        "A list of integers. Mandatory."
        "Flip the axes according to the values given.")
      .Input(0, "data", "An input tensor.")
      .Output(0, "flipped", "Flipped output tensor.");

    class GetFlipGradient : public GradientMakerBase {
      using GradientMakerBase::GradientMakerBase;
      vector<OperatorDef> GetGradientDefs() override {
        auto ops = SingleGradientDef(
          "Flip", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        ops[0].mutable_arg()->CopyFrom(Def().arg());
        return ops;
      }
    };
    REGISTER_GRADIENT(Flip, GetFlipGradient);
  } // namespace
} // namespace caffe2
