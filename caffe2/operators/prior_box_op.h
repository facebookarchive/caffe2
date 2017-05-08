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

template <typename T, class Context>
bool PriorBoxOp<T, Context>::RunOnDevice() {
	const auto& X = Input(0);
	const auto& data = Input(1);
	auto* Y = OperatorBase::Output<TensorCPU>(0);
	
	const int layer_height = X.dim(2);
	const int layer_width = X.dim(3);
	int img_height, img_width;
	if (img_h_ == 0 || img_w_ == 0) {
		img_height = data.dim(2);
		img_width = data.dim(3);
	} else {
		img_height = img_h_;
		img_width = img_w_;
	}
	float step_w, step_h;
	if (step_w_ == 0 || step_h_ == 0) {
		step_h = static_cast<float>(img_height) / layer_height;
		step_w = static_cast<float>(img_width) / layer_width;
	} else {
		step_h = step_h_;
		step_w = step_w_;
	}
	int dim = layer_height * layer_width * num_priors_ * 4;
	Y->Resize(1, 2, dim);

	int idx = 0;
	T* top_data = Y->template mutable_data<T>();
	for(int h = 0; h < layer_height; ++h) {
		for(int w = 0; w < layer_width; ++w) {
			float center_x = (w + offset_) * step_w;
			float center_y = (h + offset_) * step_h;
			float box_width, box_height;
			for(int s = 0; s < min_sizes_.size(); ++s) {
				int min_size_ = min_sizes_[s];
				box_width = box_height = min_size_;
				top_data[idx++] = (center_x - box_width / 2.) / img_width;
				top_data[idx++] = (center_y - box_height / 2.) / img_height;
				top_data[idx++] = (center_x + box_width / 2.) / img_width;
				top_data[idx++] = (center_y + box_height / 2.) / img_height;

				if (max_sizes_.size() > 0) {
					CAFFE_ENFORCE_EQ(min_sizes_.size(), max_sizes_.size());
					int max_size_ = max_sizes_[s];
					box_width = box_height = sqrt(min_size_ * max_size_);
					top_data[idx++] = (center_x - box_width / 2.) / img_width;
					top_data[idx++] = (center_y - box_height / 2.) / img_height;
					top_data[idx++] = (center_x + box_width / 2.) / img_width;
					top_data[idx++] = (center_y + box_height / 2.) / img_height;
				}
				for (int r = 0; r < aspts_.size(); ++r) {
					float ar = aspts_[r];
					if (fabs(ar - 1.) < 1e-6) {
						continue;
					}
					box_width = min_size_ * sqrt(ar);
					box_height = min_size_ / sqrt(ar);
					top_data[idx++] = (center_x - box_width / 2.) / img_width;
					top_data[idx++] = (center_y - box_height / 2.) / img_height;
					top_data[idx++] = (center_x + box_width / 2.) / img_width;
					top_data[idx++] = (center_y + box_height / 2.) / img_height;
				}
			}
		}
	}
	if (clip_) {
		for (int d = 0; d < dim; ++d) {
			top_data[d] = std::min<T>(std::max<T>(top_data[d], 0.), 1.);
		}
	}

	top_data += dim;
	if (variance_.size() == 1) {
		math::Set<T, Context>(dim, float(variance_[0]), top_data, &context_);
	} else {
		int count = 0;
		for (int h = 0; h < layer_height; ++h) {
			for (int w = 0; w < layer_width; ++w) {
				for (int i = 0; i < num_priors_; ++i) {
					for (int j = 0; j < 4; ++j) {
						top_data[count] = variance_[j];
						++count;
					}
				}
			}
		}
	}
	return true;
}


} // namespace caffe2

#endif // PRIOR_BOX_OP_H_
