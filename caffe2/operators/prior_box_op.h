#ifndef PRIOR_BOX_OP_H_
#define PRIOR_BOX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class PriorBoxOp final : public Operator<Context> {
	public:
		PriorBoxOp(const OperatorDef& operator_def, Workspace* ws)
			: Operator<Context>(operator_def, ws),
				min_sizes_(OperatorBase::GetRepeatedArgument<float>("min_sizes")),
				max_sizes_(OperatorBase::GetRepeatedArgument<float>("max_sizes")),
				aspect_ratios_(OperatorBase::GetRepeatedArgument<float>("aspect_ratios")),
				flip_(OperatorBase::GetSingleArgument<bool>("flip", true)),
				clip_(OperatorBase::GetSingleArgument<bool>("clip", false)),
				variance_(OperatorBase::GetRepeatedArgument<float>("variance")),
				img_size_(OperatorBase::GetSingleArgument<int>("img_size", 0)),
				img_w_(OperatorBase::GetSingleArgument<int>("img_w", 0)),
				img_h_(OperatorBase::GetSingleArgument<int>("img_h", 0)),
				step_(OperatorBase::GetSingleArgument<float>("step", 0.)),
				step_h_(OperatorBase::GetSingleArgument<float>("step_h", 0.)),
				step_w_(OperatorBase::GetSingleArgument<float>("step_w", 0.)),
				offset_(OperatorBase::GetSingleArgument<float>("offset", 0.5)),
				order_(StringToStorageOrder(
					OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
			
			// check order
			CAFFE_ENFORCE_EQ(order_, 
						StorageOrder::NCHW, "Only NCHW order is supported right now.");
			
			// check min_sizes
			CAFFE_ENFORCE_GT(min_sizes_.size(), 0);
			for(int i = 0; i < min_sizes_.size(); i++) {
				CAFFE_ENFORCE_GT(min_sizes_[i], 0);
			}

			// set aspts_
			aspts_.clear();
			aspts_.push_back(1.);
			for(int i = 0; i < aspect_ratios_.size(); i++) {
				float ar = aspect_ratios_[i];
				bool already_exist = false;
				for(int j = 0; j < aspts_.size(); j++) {
					if (fabs(ar - aspts_[j]) < 1e-6) {
						already_exist = true;
						break;
					}
				}
				if (!already_exist) {
					aspts_.push_back(ar);
					if (flip_) {
						aspts_.push_back(1./ar);
					}
				}
			}

			// set num_priors and check max_sizes
			num_priors_ = aspts_.size() * min_sizes_.size();
			if(max_sizes_.size() > 0) {
				CAFFE_ENFORCE_EQ(max_sizes_.size(), min_sizes_.size());
				for(int i = 0; i < max_sizes_.size(); i++) {
					CAFFE_ENFORCE_GT(max_sizes_[i], min_sizes_[i]);
					num_priors_ += 1;
				}
			}

			// check and set variance
			if(variance_.size() > 1) {
				CAFFE_ENFORCE_EQ(variance_.size(), 4);
				for(int i = 0; i < variance_.size(); i++) {
					CAFFE_ENFORCE_GT(variance_[i], 0);
				}
			} else if(variance_.size() == 1) {
				CAFFE_ENFORCE_GT(variance_[0], 0);
			} else {
				variance_.push_back(0.1);
			}

			// check img_size and set img_h_,img_w_
			if(img_h_ != 0 || img_w_ != 0) {
				CAFFE_ENFORCE_EQ(img_size_, 0);
				CAFFE_ENFORCE_GT(img_h_, 0);
				CAFFE_ENFORCE_GT(img_w_, 0);
			} else if(img_size_ != 0) {
				CAFFE_ENFORCE_GT(img_size_, 0);
				img_h_ = img_size_;
				img_w_ = img_size_;
			} else {
				img_h_ = 0;
				img_w_ = 0;
			}

			// check step and set step_h_, step_w_
			if(step_h_ != 0. || step_w_ != 0.) {
				CAFFE_ENFORCE_EQ(step_, 0.);
				CAFFE_ENFORCE_GT(step_h_, 0.);
				CAFFE_ENFORCE_GT(step_w_, 0.);
			} else if(step_ != 0.) {
				CAFFE_ENFORCE_GT(step_, 0.);
				step_h_ = step_;
				step_w_ = step_;
			} else {
				step_h_ = 0.;
				step_w_ = 0.;
			}

		}
		USE_OPERATOR_CONTEXT_FUNCTIONS;
		bool RunOnDevice() override;

	protected:
		vector<float> min_sizes_;
		vector<float> max_sizes_;
		vector<float> aspect_ratios_;
		vector<float> aspts_;

		bool flip_;
		int num_priors_;
		bool clip_;
		vector<float> variance_;

		int img_size_;
		int img_w_;
		int img_h_;
		float step_;
		float step_w_;
		float step_h_;
		float offset_;
		StorageOrder order_;
};

} // namespace caffe2

#endif // PRIOR_BOX_OP_H_
