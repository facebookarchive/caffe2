#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(ConvGradient).NumInputs(2, 3).NumOutputs(1, 3);
class GetConvGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 2);

    ArgumentHelper argsHelper(def_);

    auto compute_dX = !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

    auto engine = def_.engine();

    // Use seperate gradient ops
    if (engine == "CUDNN" && def_.device_option().device_type() == CUDA) {
      vector<OperatorDef> grad_defs{};

      // dW
      OperatorDef wgrad_def = CreateOperatorDef(
          "ConvFilterGradient",
          "",
          vector<string>{I(0), GO(0), I(1)}, // {X, dY, W}
          vector<string>{GI(1)}); // {dW}
      grad_defs.push_back(wgrad_def);

      // dX - can be disabled
      if (compute_dX) {
        OperatorDef dgrad_def = CreateOperatorDef(
            "ConvDataGradient",
            "",
            vector<string>{I(0), I(1), GO(0)}, // {X, W, dY}
            vector<string>{GI(0)}); // {dX}
        grad_defs.push_back(dgrad_def);
      }

      // db
      if (def_.input_size() > 2) {
        OperatorDef bgrad_def = CreateOperatorDef(
            "ConvBiasGradient",
            "",
            vector<string>{GO(0)}, // {dY}
            vector<string>{GI(2)}); // {db}
        grad_defs.push_back(bgrad_def);
      }
      return grad_defs;
    } else {
      if (def_.input_size() == 3) {
        if (compute_dX) {
          return SingleGradientDef(
              "ConvGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1), GI(2), GI(0)});
        } else {
          return SingleGradientDef(
              "ConvGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1), GI(2)});
        }
      } else {
        if (compute_dX) {
          return SingleGradientDef(
              "ConvGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1), GI(0)},
              vector<Argument>{MakeArgument<int>("no_bias", 1)});
        } else {
          return SingleGradientDef(
              "ConvGradient",
              "",
              vector<string>{I(0), I(1), GO(0)},
              vector<string>{GI(1)},
              vector<Argument>{MakeArgument<int>("no_bias", 1)});
        }
      }
    }
  }
};
REGISTER_GRADIENT(Conv, GetConvGradient);

} // namespace
} // namespace caffe2
