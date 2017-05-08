#include "caffe2/operators/prior_box_op.h"

namespace caffe2 {

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
