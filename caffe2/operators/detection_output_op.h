#ifndef DETECTION_OUTPUT_OP_H_
#define DETECTION_OUTPUT_OP_H_

#include "caffe/proto/caffe.pb.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/bbox_util.h"

namespace caffe2 {
template <typename T, class Context>
class DetectionOutputOp final : public Operator<Context> {
	public:
		DetectionOutputOp(const OperatorDef& operator_def, Workspace* ws)
			: Operator<Context>(operator_def, ws),
				num_classes_(OperatorBase::GetSingleArgument<int>("num_classes", -1)),
				share_location_(OperatorBase::GetSingleArgument<bool>("share_location", true)),
				background_label_id_(OperatorBase::GetSingleArgument<int>("background_label_id", 0)),
				nms_threshold_(OperatorBase::GetSingleArgument<float>("nms_threshold", 0.3)),
				top_k_(OperatorBase::GetSingleArgument<int>("top_k", -1)),
				eta_(OperatorBase::GetSingleArgument<float>("eta", 1.0)),
				variance_encoded_in_target_(OperatorBase::GetSingleArgument<bool>("variance_encoded_in_target", false)),
				keep_top_k_(OperatorBase::GetSingleArgument<int>("keep_top_k", -1)),
				confidence_threshold_(OperatorBase::GetSingleArgument<float>("confidence_threshold", 0.00999999977648)),
				code_t_(OperatorBase::GetSingleArgument<int>("code_type", 2))	{
			
			CAFFE_ENFORCE_GT(num_classes_, 0);
			CAFFE_ENFORCE_GE(nms_threshold_, 0.);
			CAFFE_ENFORCE_GT(eta_, 0.);
			CAFFE_ENFORCE_LE(eta_, 1.);
			num_loc_classes_ = share_location_ ? 1 : num_classes_;
			switch(code_t_) {
			case 1:
				code_type_ = caffe::PriorBoxParameter_CodeType_CORNER;
				break;
			case 2:
				code_type_ = caffe::PriorBoxParameter_CodeType_CENTER_SIZE;
				break;
			case 3:
				code_type_ = caffe::PriorBoxParameter_CodeType_CORNER_SIZE;
				break;
			default:
				LOG(FATAL) << "Unknown CodeType";
			}
		}
		USE_OPERATOR_CONTEXT_FUNCTIONS;
		bool RunOnDevice() override;
	
	protected:
		vector<NormalizedBBox> prior_bboxes;
		int num_loc_classes_;
		bool share_location_;
		int num_classes_;
		int background_label_id_;
		CodeType code_type_;
		int code_t_;
		bool variance_encoded_in_target_;
		float confidence_threshold_;
		float nms_threshold_;
		float eta_;
		int top_k_;
		int keep_top_k_;
    
    Tensor<Context> bbox_preds_;
    Tensor<Context> bbox_permute_;
    Tensor<Context> conf_permute_;
};

} // namespace caffe2



#endif
