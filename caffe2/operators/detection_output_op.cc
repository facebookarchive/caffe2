#include "caffe2/operators/detection_output_op.h"

namespace caffe2 {

template <>
bool DetectionOutputOp<float, CPUContext>::RunOnDevice() {
	const auto& loc = Input(0);
	const auto& conf = Input(1); 

  // TODO(ky): 
  // DetectionOutputOp must run in CUDA devs
  // because I don't find a way in CPUContext 
  // to access loc and conf which are in CUDAContext.
  // So you will get a error when runing in CPU.
  // Please run in CUDAContext!!!
  //


  //Tensor<CPUContext> loc, conf;
  //loc.CopyFrom(loc_, &context_);
  //conf.CopyFrom(conf_, &context_);
	const auto& prior = OperatorBase::Input<TensorCPU>(2);
	auto* Y = OperatorBase::Output<TensorCPU>(0);

	const float* loc_data = loc.template data<float>();
	const float* conf_data = conf.template data<float>();
	const float* prior_data = prior.template data<float>();

	const int num = loc.dim(0);
	int num_priors_ = prior.dim(2) / 4;
	CAFFE_ENFORCE_EQ(num_priors_ * num_loc_classes_ * 4, loc.dim(1));
	CAFFE_ENFORCE_EQ(num_priors_ * num_classes_, conf.dim(1));
	
	// Retrieve all location predictions.
	vector<LabelBBox> all_loc_preds;
	GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
										share_location_, &all_loc_preds);
	
	// Retrieve all confidences.
	vector<map<int, vector<float> > > all_conf_scores;
	GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
											&all_conf_scores);

	// Retrieve all prior bboxes. It is same within a batch since we assume all
	// images in a batch are of same dimension.
	vector<NormalizedBBox> prior_bboxes;
	vector<vector<float> > prior_variances;
	GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

	// Decode all loc predictions to bboxes.
	vector<LabelBBox> all_decode_bboxes;
	const bool clip_bbox = false;
	DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
									share_location_, num_loc_classes_, background_label_id_,
									code_type_, variance_encoded_in_target_, clip_bbox,
									&all_decode_bboxes);
	
	int num_kept = 0;
	vector<map<int, vector<int> > > all_indices;
	for(int i = 0; i < num; i++) {
		const LabelBBox& decode_bboxes = all_decode_bboxes[i];
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		map<int, vector<int> > indices;
		int num_det = 0;
		for (int c = 0; c < num_classes_; ++c) {
			if (c == background_label_id_) {
				continue;
			}
			if (conf_scores.find(c) == conf_scores.end()) {
				LOG(FATAL) << "Could not find confidence predictions for label " << c;
			}
			const vector<float>& scores = conf_scores.find(c)->second;
			int label = share_location_ ? -1 : c;
			if (decode_bboxes.find(label) == decode_bboxes.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for label " << label;
				continue;
			}
			const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
			ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
					top_k_, &(indices[c]));
			num_det += indices[c].size();
		}
		if (keep_top_k_ > -1 && num_det > keep_top_k_) {
			vector<pair<float, pair<int, int> > > score_index_pairs;
			for (map<int, vector<int> >::iterator it = indices.begin();
					it != indices.end(); ++it) {
				int label = it->first;
				const vector<int>& label_indices = it->second;
				if (conf_scores.find(label) == conf_scores.end()) {
					LOG(FATAL) << "Could not find location predictions for " << label;
					continue;
				}
				const vector<float>& scores = conf_scores.find(label)->second;
				for (int j = 0; j < label_indices.size(); ++j) {
					int idx = label_indices[j];
					CAFFE_ENFORCE_LT(idx, scores.size());
					score_index_pairs.push_back(std::make_pair(
								scores[idx], std::make_pair(label, idx)));
				}
			}
			// Keep top k results per image.
			std::sort(score_index_pairs.begin(), score_index_pairs.end(),
					SortScorePairDescend<pair<int, int> >);
			score_index_pairs.resize(keep_top_k_);
			// Store the new indices.
			map<int, vector<int> > new_indices;
			for (int j = 0; j < score_index_pairs.size(); ++j) {
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			}
			all_indices.push_back(new_indices);
			num_kept += keep_top_k_;
		} else {
			all_indices.push_back(indices);
			num_kept += num_det;
		}
	}

	float* top_data;
	if(num_kept == 0) {
		LOG(INFO) << "Couldn't find any detections";
		Y->Resize(1,1,num,7);
		top_data = Y->template mutable_data<float>();
		math::Set<float, CPUContext>(num*7, float(-1), top_data, &context_);
		// Generate fake results per image.
		for (int i = 0; i < num; ++i) {
			top_data[0] = i;
			top_data += 7;
		}
	} else {
		Y->Resize(1,1,num_kept,7);
		top_data = Y->template mutable_data<float>();
	}

	int count = 0;
	for (int i = 0; i < num; ++i) {
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		const LabelBBox& decode_bboxes = all_decode_bboxes[i];
		for (map<int, vector<int> >::iterator it = all_indices[i].begin();
				it != all_indices[i].end(); ++it) {
			int label = it->first;
			if (conf_scores.find(label) == conf_scores.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find confidence predictions for " << label;
				continue;
			}
			const vector<float>& scores = conf_scores.find(label)->second;
			int loc_label = share_location_ ? -1 : label;
			if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for " << loc_label;
				continue;
			}
			const vector<NormalizedBBox>& bboxes =
				decode_bboxes.find(loc_label)->second;
			vector<int>& indices = it->second;

			for (int j = 0; j < indices.size(); ++j) {
				int idx = indices[j];
				top_data[count * 7] = i;
				top_data[count * 7 + 1] = label;
				top_data[count * 7 + 2] = scores[idx];
				const NormalizedBBox& bbox = bboxes[idx];
				top_data[count * 7 + 3] = bbox.xmin();
				top_data[count * 7 + 4] = bbox.ymin();
				top_data[count * 7 + 5] = bbox.xmax();
				top_data[count * 7 + 6] = bbox.ymax();
				++count;
			}
		}
	}
	
	return true;
}

namespace {
REGISTER_CPU_OPERATOR(DetectionOutput, DetectionOutputOp<float, CPUContext>);
NO_GRADIENT(DetectionOutput);

OPERATOR_SCHEMA(DetectionOutput)
	.NumInputs(3)
	.NumOutputs(1)
	.SetDoc(R"DOC(DetectionOutputLayer in SSD)DOC")
	.Arg("num_classes", "num_classes")
	.Arg("share_location", "share_location")
	.Arg("background_label_id", "background_label_id")
	.Arg("nms_threshold", "nms_threshold")
	.Arg("top_k", "top_k")
	.Arg("eta", "eta")
	.Arg("code_type", "code_type")
	.Arg("variance_encoded_in_target", "variance_encoded_in_target")
	.Arg("keep_top_k", "keep_top_k")
	.Arg("confidence_threshold", "confidence_threshold")
	.Input(0, "loc", "loc")
	.Input(1, "conf", "conf")
	.Input(2, "prior", "prior")
	.Output(0, "Y", "detection results");
} // namespace 

} // namespace caffe2
