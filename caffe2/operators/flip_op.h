#ifndef CAFFE2_OPERATORS_FLIP_OP_H_
#define CAFFE2_OPERATORS_FLIP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

  template <class Context>
  class FlipOp final : public Operator<Context> {
  public:
    USE_OPERATOR_CONTEXT_FUNCTIONS;
    USE_DISPATCH_HELPER;
    FlipOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
      axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {
      // We will check the legality of axes_: it should be continuous and
      // monotonous increasing between 0 and X.ndim().
      // legal: (1,), (2,3) and (0,1,2)
      // illegal: (-1,0,1), (1,3) and (4,) when only 4 dimension
      CAFFE_ENFORCE(OperatorBase::HasArgument("axes"), "Argument `axes` is missing");
      CAFFE_ENFORCE(axes_.size() > 0, "Argument `axes` is missing");
      CAFFE_ENFORCE(axes_[0] >= 0, "Argument `axes` has invalid dimension:", axes_[0]);
      for (int i = 1; i < axes_.size(); ++i) {
        CAFFE_ENFORCE(axes_[i] == axes_[0] + i, "Argument `axes` has invalid dimension:", axes_[i]);
      }
      //CAFFE_ENFORCE(axes_[axes_.size()-1] < X.ndim(), "Argument `axes` has invalid dimension:", axes_[axes_.size()-1]);
    }
    ~FlipOp() {}

    bool RunOnDevice() override {
      const auto& X = Input(0);
      auto* Y = Output(0);
      Y->ResizeLike(X);
      // Do the actual flip, which is implemented in DoRunWithType().
      return DispatchHelper<TensorTypes<float, double, int, long>>::call(
        this, Input(0));
    }

  protected:
    template <typename T>
    bool DoRunWithType();

    std::vector<int> axes_;
  };

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FLIP_OP_H_
