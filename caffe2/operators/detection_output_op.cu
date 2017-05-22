#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/detection_output_op.h"

namespace caffe2 {

template <typename Dtype>
__global__ void DecodeBBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int i = index % 4;
    const int c = (index / 4) % num_loc_classes;
    const int d = (index / 4 / num_loc_classes) % num_priors;
    if (!share_location && c == background_label_id) {
      // Ignore background class if not share_location.
      return;
    }
    const int pi = d * 4;
    const int vi = pi + num_priors * 4;
    if (code_type == caffe::PriorBoxParameter_CodeType_CORNER) {
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index];
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
      }
    } else if (code_type == caffe::PriorBoxParameter_CodeType_CENTER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      const Dtype xmin = loc_data[index - i];
      const Dtype ymin = loc_data[index - i + 1];
      const Dtype xmax = loc_data[index - i + 2];
      const Dtype ymax = loc_data[index - i + 3];

      Dtype decode_bbox_center_x, decode_bbox_center_y;
      Dtype decode_bbox_width, decode_bbox_height;
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to retore the offset
        // predictions.
        decode_bbox_center_x = xmin * prior_width + prior_center_x;
        decode_bbox_center_y = ymin * prior_height + prior_center_y;
        decode_bbox_width = exp(xmax) * prior_width;
        decode_bbox_height = exp(ymax) * prior_height;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        decode_bbox_center_x =
          prior_data[vi] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y =
          prior_data[vi + 1] * ymin * prior_height + prior_center_y;
        decode_bbox_width =
          exp(prior_data[vi + 2] * xmax) * prior_width;
        decode_bbox_height =
          exp(prior_data[vi + 3] * ymax) * prior_height;
      }

      switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
      }
    } else if (code_type == caffe::PriorBoxParameter_CodeType_CORNER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      Dtype p_size;
      if (i == 0 || i == 2) {
        p_size = prior_width;
      } else {
        p_size = prior_height;
      }
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
      }
    } else {
      // Unknown code type.
    }
    if (clip_bbox) {
      bbox_data[index] = max(min(bbox_data[index], Dtype(1.)), Dtype(0.));
    }
  }
}

template <typename Dtype>
__global__ void PermuteDataKernel(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int i = index % num_dim;
    const int c = (index / num_dim) % num_classes;
    const int d = (index / num_dim / num_classes) % num_data;
    const int n = index / num_dim / num_classes / num_data;
    const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
    new_data[new_index] = data[index];
  }
}


template<>
bool DetectionOutputOp<float, CUDAContext>::RunOnDevice() {
  auto& loc = Input(0);
  auto& conf = Input(1);
  auto& prior_ = OperatorBase::Input<TensorCPU>(2);
	auto* Y = OperatorBase::Output<TensorCPU>(0);

  Tensor<CUDAContext> prior;
  prior.CopyFrom(prior_, &context_);

  const float* loc_data = loc.data<float>();
	const float* conf_data = conf.data<float>();
  const float* prior_data = prior.data<float>();
  const int num = loc.dim(0);
  
  bbox_preds_.ResizeLike(loc);
  if (!share_location_) {
    bbox_permute_.ResizeLike(loc);
  }
  conf_permute_.ResizeLike(conf);
  
  float* bbox_data = bbox_preds_.mutable_data<float>();
  const int loc_count = bbox_preds_.size();
  const bool clip_bbox = false;
	const int num_priors_ = prior.dim(2) / 4;
	CAFFE_ENFORCE_EQ(num_priors_ * num_loc_classes_ * 4, loc.dim(1));
	CAFFE_ENFORCE_EQ(num_priors_ * num_classes_, conf.dim(1));
  
  
	DecodeBBoxesKernel<float> <<<CAFFE_GET_BLOCKS(loc_count),
															 CAFFE_CUDA_NUM_THREADS, 0,
															 context_.cuda_stream()>>> 
	(loc_count, loc_data, prior_data, code_type_, variance_encoded_in_target_,
   num_priors_, share_location_, num_loc_classes_, background_label_id_,
	 clip_bbox, bbox_data);
  
  
	const float* bbox_cpu_data = (float*)malloc(bbox_permute_.size()*sizeof(float));
	Tensor<CPUContext> bbox_device_data;
	if (!share_location_) {
		float* bbox_permute_data = bbox_permute_.mutable_data<float>();
		PermuteDataKernel<float> <<<CAFFE_GET_BLOCKS(loc_count),
																CAFFE_CUDA_NUM_THREADS, 0,
																context_.cuda_stream()>>>
		(loc_count, bbox_data, num_loc_classes_, num_priors_, 4, bbox_permute_data);
		bbox_device_data.CopyFrom(bbox_permute_, &context_);
		bbox_cpu_data = bbox_device_data.data<float>();
	} else {
		bbox_device_data.CopyFrom(bbox_preds_, &context_);
		bbox_cpu_data = bbox_device_data.data<float>();
	}

	// Retrieve all confidences.
	float* conf_permute_data = conf_permute_.mutable_data<float>();
	const int conf_count = conf.size();
	PermuteDataKernel<float> <<<CAFFE_GET_BLOCKS(conf_count),
															CAFFE_CUDA_NUM_THREADS, 0,
															context_.cuda_stream()>>>
	(conf_count, conf_data, num_classes_, num_priors_, 1, conf_permute_data);
	Tensor<CPUContext> conf_device_data;
	conf_device_data.CopyFrom(conf_permute_, &context_);
	const float* conf_cpu_data = conf_device_data.data<float>();

  // I found CopyFrom use cudaMemcpyAsync, 
  // here in my opinion should wait and check kernal function finished.
  CHECK(context_.FinishDeviceComputation());
	
  int num_kept = 0;
	vector<map<int, vector<int> > > all_indices;
	for (int i = 0; i < num; ++i) {
		map<int, vector<int> > indices;
    int num_det = 0;
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors_ * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      const float* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors_;
      const float* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += c * num_priors_ * 4;
      }
      ApplyNMSFast(cur_bbox_data, cur_conf_data, num_priors_,
          confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors_ + idx];
          score_index_pairs.push_back(std::make_pair(
                  score, std::make_pair(label, idx)));
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
		top_data = Y->mutable_data<float>();
		// here context_ is CUDAContext, but we need CPUContext.
		// So i set 0
		math::Set<float, CPUContext>(num*7, float(-1), top_data, 0);
		for (int i = 0; i < num; ++i) {
			top_data[0] = i;
			top_data += 7;
		}
	} else {
		Y->Resize(1,1,num_kept,7);
		top_data = Y->mutable_data<float>();
	}

	int count = 0;
	for (int i = 0; i < num; ++i) {
		const int conf_idx = i * num_classes_ * num_priors_;
		int bbox_idx;
		if (share_location_) {
			bbox_idx = i * num_priors_ * 4;
		} else {
			bbox_idx = conf_idx * 4;
		}
		for (map<int, vector<int> >::iterator it = all_indices[i].begin();
				it != all_indices[i].end(); ++it) {
			int label = it->first;
			vector<int>& indices = it->second;
			const float* cur_conf_data = conf_cpu_data + conf_idx + label * num_priors_;
			const float* cur_bbox_data = bbox_cpu_data + bbox_idx;
			if (!share_location_) {
				cur_bbox_data += label * num_priors_ * 4;
			}
			for (int j = 0; j < indices.size(); ++j) {
				int idx = indices[j];
				top_data[count * 7] = i;
				top_data[count * 7 + 1] = label;
				top_data[count * 7 + 2] = cur_conf_data[idx];
				for (int k = 0; k < 4; ++k) {
					top_data[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
				}
        count++;
			}
		}
	}

	return true;
}


namespace caffe {

REGISTER_CUDA_OPERATOR(DetectionOutput, DetectionOutputOp<float, CUDAContext>);
} // namespace caffe

} // namespace caffe2

