#ifndef NORM_OP_H_
#define NORM_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class NormOp final : public Operator<Context> {
	public:
		NormOp(const OperatorDef& operator_def, Workspace* ws)
			: Operator<Context>(operator_def, ws),
				across_spatial_(OperatorBase::GetSingleArgument<bool>("across_spatial", false)),
				order_(StringToStorageOrder(
							OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
				eps_(OperatorBase::GetSingleArgument<float>("eps", 10e-10)),
				channel_shared_(OperatorBase::GetSingleArgument<bool>("channel_shared", false)) {
			CAFFE_ENFORCE_EQ(order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
		}
		USE_OPERATOR_CONTEXT_FUNCTIONS;
		bool RunOnDevice() override;
	
	protected:
		bool across_spatial_;
		StorageOrder order_;
		float eps_;
		bool channel_shared_;
		Tensor<Context> buffer_, buffer_channel_, buffer_spatial_;
		Tensor<Context> norm_;
		Tensor<Context> sum_channel_multiplier_, sum_spatial_multiplier_;
};


} // namepsace caffe2



#endif // NORM_OP_H_
