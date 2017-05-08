#include "caffe2/operators/prior_box_op.h"

namespace caffe2 {

template<>
bool PriorBoxOp<float, CPUContext>::RunOnDevice() {
	const auto& X = Input(0);
	const auto& data = Input(1);
	auto* Y = Output(0);
	
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
	float* top_data = Y->mutable_data<float>();
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
			top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
		}
	}

	top_data += dim;
	if (variance_.size() == 1) {
		math::Set<float, CPUContext>(dim, float(variance_[0]), top_data, &context_);
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
}

namespace {
REGISTER_CPU_OPERATOR(PriorBox, PriorBoxOp<float, CPUContext>);
NO_GRADIENT(PriorBox);

OPERATOR_SCHEMA(PriorBox)
	.NumInputs(2)
	.NumOutputs(1)
	.TensorInferenceFunction(
		[](const OperatorDef& def, const vector<TensorShape>& in) {
			ArgumentHelper helper(def);
			const StorageOrder order = StringToStorageOrder(
					helper.GetSingleArgument<string>("order", "NCHW"));
			const TensorShape &X = in[0];
			int layer_height = X.dims(2); int layer_width = X.dims(3);
			
			vector<float> aspects = helper.GetRepeatedArgument<float>("aspect_ratios");
			vector<float> min_sizes = helper.GetRepeatedArgument<float>("min_sizes");
			vector<float> aspts;
			aspts.push_back(1);
			bool flip = helper.GetSingleArgument<bool>("flip", true);
			for(int i = 0; i < aspects.size(); i++) {
				float ar = aspects[i];
				bool already_exist = false;
				for(int j = 0; j < aspts.size(); j++) {
					if (fabs(ar - aspts[j]) < 1e-6) {
						already_exist = true;
						break;
					}
				}
				if(!already_exist) {
					aspts.push_back(ar);
					if(flip) {
						aspts.push_back(1./ar);
					}
				}
			}

			int num_priors_ = aspts.size() * min_sizes.size();
			vector<float> max_sizes = helper.GetRepeatedArgument<float>("max_sizes");
			for(int i = 0; i < max_sizes.size(); i++)
				num_priors_ += 1;

			TensorShape Y = CreateTensorShape(
					vector<int>({1, 2, layer_width * layer_height * num_priors_ * 4}),
					X.data_type());
			return vector<TensorShape>({Y});
		})
	.SetDoc(R"DOC(PriorBoxLayer in SSD)DOC")
	.Arg("min_sizes", "repeated min_sizes")
	.Arg("max_sizes", "optional max_sizes")
	.Arg("aspect_ratios", "repeated aspect ratios")
	.Arg("flip", "1 / ar")
	.Arg("clip", "clip box")
	.Arg("variance", "prior variance")
	.Arg("img_size", "image size")
	.Arg("img_w", "image width")
	.Arg("img_h", "image height")
	.Arg("step", "feature stride")
	.Arg("step_h", "step height")
	.Arg("step_w", "step width")
	.Arg("offset", "offset")
	.Arg("order", "NCHW")
	.Input(0, "X", "NCHW tensor")
	.Input(1, "data", "NCHW input tensor")
	.Output(0, "Y", "prior boxes");

} // namespace


} // namespace caffe2
