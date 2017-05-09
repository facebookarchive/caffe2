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
};

template <typename T, class Context>
bool DetectionOutputOp<T, Context>::RunOnDevice() {
	const auto& loc_ = Input(0);
	const auto& conf_ = Input(1);
	Tensor<CPUContext> loc, conf;
	loc.CopyFrom(loc_, &context_);
	conf.CopyFrom(conf_, &context_);

	const auto& prior = OperatorBase::Input<TensorCPU>(2);
	auto* Y = OperatorBase::Output<TensorCPU>(0);

	const T* loc_data = loc.template data<T>();
	const T* conf_data = conf.template data<T>();
	const T* prior_data = prior.template data<T>();

	const int num = loc_.dim(0);
	int num_priors_ = prior.dim(2) / 4;
	CAFFE_ENFORCE_EQ(num_priors_ * num_loc_classes_ * 4, loc_.dim(1));
	CAFFE_ENFORCE_EQ(num_priors_ * num_classes_, conf_.dim(1));
	
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

	T* top_data;
	if(num_kept == 0) {
		LOG(INFO) << "Couldn't find any detections";
		Y->Resize(1,1,num,7);
		top_data = Y->template mutable_data<T>();
		math::Set<T, Context>(num*7, float(-1), top_data, &context_);
		// Generate fake results per image.
		for (int i = 0; i < num; ++i) {
			top_data[0] = i;
			top_data += 7;
		}
	} else {
		Y->Resize(1,1,num_kept,7);
		top_data = Y->template mutable_data<T>();
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

} // namespace caffe2



#endif
