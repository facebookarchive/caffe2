
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <class SIndex, class Context>
bool SliceImpl(
    Tensor<Context>* output,
    const Tensor<Context>& data,
    const Tensor<Context>& starts,
    const Tensor<Context>& ends,
    Context* context,
    Tensor<Context>* gdata = nullptr,
    const Tensor<Context>* go = nullptr) {
  bool backward = output == nullptr;

  auto* starts_data = starts.template data<SIndex>();
  auto* ends_data = ends.template data<SIndex>();

  CAFFE_ENFORCE_EQ(starts.ndim(), 1);
  CAFFE_ENFORCE_EQ(ends.ndim(), 1);
  CAFFE_ENFORCE_GE(data.ndim(), starts.size());
  CAFFE_ENFORCE_EQ(starts.size(), ends.size());

  std::vector<SIndex> starts_idx(data.ndim());
  std::vector<SIndex> ends_idx(data.ndim());
  std::vector<SIndex> dst_sizes(data.ndim());

  for (int i = 0; i < data.ndim(); ++i) {
    if (i >= starts.size()) {
      starts_idx[i] = 0;
      ends_idx[i] = data.dims()[i];
      continue;
    }
    if (data.dims()[i] > 0) {
      auto start = starts_data[i];
      auto end = ends_data[i];
      if (start < 0) {
        start = data.dims()[i] + 1 + start;
      }
      if (end < 0) {
        end = data.dims()[i] + 1 + end;
      }
      if (start > data.dims()[i]) {
        start = data.dims()[i];
      }
      if (end > data.dims()[i]) {
        end = data.dims()[i];
      }
      CAFFE_ENFORCE_GE(start, 0);
      CAFFE_ENFORCE_GE(end, 0);
      CAFFE_ENFORCE_GE(end, start);
      starts_idx[i] = start;
      ends_idx[i] = end;
      dst_sizes[i] = end - start;
    } else {
      starts_idx[i] = 0;
      ends_idx[i] = 0;
      dst_sizes[i] = 0;
    }
  }

  if (data.size() <= 0) {
    // When the input is empty, we do not need to do copy.
    if (!backward) {
      output->Resize(dst_sizes);
      output->raw_mutable_data(data.meta());
    }
    return true;
  }
  // for now only supports slicing in 1 dimension
  int dim = -1;
  for (int i = 0; i < data.ndim(); ++i) {
    if (starts_idx[i] > 0 || ends_idx[i] < data.dims()[i]) 
      dim = i;
  }
  if (dim == -1) {
    if (!backward) {
      output->CopyFrom(data, context);
    } else {
      gdata->CopyFrom(*go, context);
    }
    return true;
  }
  size_t unit = std::accumulate(
      data.dims().begin() + dim + 1,
      data.dims().end(),
      1,
      std::multiplies<SIndex>());
  size_t num_blocks = std::accumulate(
      dst_sizes.begin(),
      dst_sizes.begin() + dim,
      1,
      std::multiplies<SIndex>());
  if (!backward) {
    output->Resize(dst_sizes);
  } else {
    gdata->ResizeLike(data);
  }

  size_t itemsize = data.meta().itemsize();

  if (!backward) {
    char* src_bytes = (char*)data.raw_data();
    char* dst_bytes = (char*)output->raw_mutable_data(data.meta());

    size_t src_nbytes = data.nbytes();
    size_t dst_nbytes = output->nbytes();

    size_t src_frame_size = unit * data.dims()[dim];
    size_t dst_frame_size = unit * (ends_idx[dim] - starts_idx[dim]);
    size_t src_offset = unit * starts_idx[dim];

    if (num_blocks == 0 || dst_frame_size == 0) {
      return true;
    }

    size_t src_frame_size_bytes = itemsize * src_frame_size;
    size_t dst_frame_size_bytes = itemsize * dst_frame_size;

	std::vector<int> it(dim);
	for(int i = 0 ; i < dim ; i++) it[i] = starts_idx[i];
	while(it[0] < ends_idx[0]) {
		//get the start postion of current frame
		size_t src_frame_idx = 0;
		size_t dst_frame_idx = 0;
		size_t src_basis = 1;
		size_t dst_basis = 1;
		for(int d = dim - 1 ; d >= 0 ; d--) {
			src_frame_idx += it[d] * src_basis;
			dst_frame_idx += (it[d] - starts_idx[d]) * dst_basis;
			src_basis *= data.dims()[d];
			dst_basis *= dst_sizes[d];
		}
		//copy the block in current src frame to dst frame
		char * local_src_offset_bytes = src_bytes + src_frame_idx * src_frame_size_bytes + itemsize * src_offset;
		char * local_dst_offset_bytes = dst_bytes + dst_frame_idx * dst_frame_size_bytes;
		DCHECK_LE(
			static_cast<void*>(local_src_offset_bytes + dst_frame_size_bytes),
			static_cast<void*>(src_bytes + src_nbytes));
		DCHECK_LE(
			static_cast<void*>(local_dst_offset_bytes + dst_frame_size_bytes),
			static_cast<void*>(dst_bytes + dst_nbytes));
		context->template CopyItems<Context, Context>(
			data.meta(),
			dst_frame_size,
			(void*)local_src_offset_bytes,
			(void*)local_dst_offset_bytes);
		//get the index of next frame
		it[dim - 1]++;
		for(int i = dim - 1 ; i > 0 && (it[i] >= ends_idx[i]) ; --i) {
			it[i] = starts_idx[i];
			it[i - 1]++;
		}
	}
  } else {
    char* src_bytes = (char*)go->raw_data();
    char* dst_bytes = (char*)gdata->raw_mutable_data(go->meta());

    size_t src_nbytes = go->nbytes();
    size_t dst_nbytes = gdata->nbytes();

    size_t src_frame_size = unit * (ends_idx[dim] - starts_idx[dim]);
    size_t dst_frame_size = unit * data.dims()[dim];
    size_t dst_offset = unit * starts_idx[dim];

    if (num_blocks == 0 || dst_frame_size == 0) {
      return true;
    }

    size_t src_frame_size_bytes = itemsize * src_frame_size;
    size_t dst_frame_size_bytes = itemsize * dst_frame_size;

    // Zero out gradient blob before copy since we copy in fewer items than
    // there is space for
    math::Set<char, Context>(dst_nbytes, 0, dst_bytes, context);

    // If output tensor is empty, just return zeroed gradient tensor
    if (!src_bytes) {
      return true;
    }

    std::vector<int> it(dim);
	for(int i = 0 ; i < dim ; i++) it[i] = starts_idx[i];
	while(it[0] < ends_idx[0]) {
		//get the start postion of current frame
		size_t src_frame_idx = 0;
		size_t dst_frame_idx = 0;
		size_t src_basis = 1;
		size_t dst_basis = 1;
		for(int d = dim - 1 ; d >= 0 ; d--) {
			src_frame_idx += (it[d] - starts_idx[d]) * src_basis;
			dst_frame_idx += it[d] * dst_basis;
			src_basis *= dst_sizes[d];
			dst_basis *= data.dims()[d];
		}
		//copy current srd frame to the block in dst frame
		char * local_src_offset_bytes = src_bytes + src_frame_idx * src_frame_size_bytes;
		char * local_dst_offset_bytes = dst_bytes + dst_frame_idx * dst_frame_size_bytes + itemsize * dst_offset;
		DCHECK_LE(
			local_src_offset_bytes + src_frame_size_bytes,
			src_bytes + src_nbytes);
		DCHECK_LE(
			local_dst_offset_bytes + src_frame_size_bytes,
			dst_bytes + dst_nbytes);
		context->template CopyItems<Context, Context>(
			go->meta(),
			src_frame_size,
			(void*)local_src_offset_bytes,
			(void*)local_dst_offset_bytes);
		//get the index of next frame
		it[dim - 1]++;
		for(int i = dim - 1 ; i > 0 && (it[i] >= ends_idx[i]) ; --i) {
			it[i] = starts_idx[i];
			it[i - 1]++;
		}		
	}
  }
  return true;
}

} // namespace

template <class SIndex, class Context>
class SliceOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SliceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        starts_(OperatorBase::GetRepeatedArgument<SIndex>("starts")),
        ends_(OperatorBase::GetRepeatedArgument<SIndex>("ends")),
        statically_inited_(false) {}

  bool RunOnDevice() override {
    auto* output = Output(0);
    auto& data = Input(0);

    if (InputSize() > 1) {
      starts_host_.template CopyFrom<Context>(Input(1));
      ends_host_.template CopyFrom<Context>(Input(2));
    } else {
      if (!statically_inited_) {
        CAFFE_ENFORCE(HasArgument("starts"));
        CAFFE_ENFORCE(HasArgument("ends"));
        CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

        starts_host_.Resize(starts_.size());
        ends_host_.Resize(ends_.size());

        memcpy(
            starts_host_.template mutable_data<SIndex>(),
            starts_.data(),
            sizeof(SIndex) * starts_.size());
        memcpy(
            ends_host_.template mutable_data<SIndex>(),
            ends_.data(),
            sizeof(SIndex) * ends_.size());
        statically_inited_ = true;
      }
    }

    return SliceImpl<SIndex, Context>(
        output, data, starts_host_, ends_host_, &context_);
  }

  DISABLE_COPY_AND_ASSIGN(SliceOp);

 private:
  std::vector<SIndex> starts_;
  std::vector<SIndex> ends_;
  bool statically_inited_;
  TensorCPU starts_host_;
  TensorCPU ends_host_;
};

template <class SIndex, class Context>
class SliceGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SliceGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        starts_(OperatorBase::GetRepeatedArgument<SIndex>("starts")),
        ends_(OperatorBase::GetRepeatedArgument<SIndex>("ends")),
        statically_inited_(false) {}

  bool RunOnDevice() override {
    auto* gdata = Output(0);
    auto& data = Input(0);

    if (InputSize() == 4) {
      starts_host_.template CopyFrom<Context>(Input(1));
      ends_host_.template CopyFrom<Context>(Input(2));

      auto& go = Input(3);

      return SliceImpl<SIndex, Context>(
          nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
    } else {
      if (!statically_inited_) {
        CAFFE_ENFORCE(HasArgument("starts"));
        CAFFE_ENFORCE(HasArgument("ends"));
        CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

        starts_host_.Resize(starts_.size());
        ends_host_.Resize(ends_.size());

        memcpy(
            starts_host_.template mutable_data<SIndex>(),
            starts_.data(),
            sizeof(SIndex) * starts_.size());
        memcpy(
            ends_host_.template mutable_data<SIndex>(),
            ends_.data(),
            sizeof(SIndex) * ends_.size());

        statically_inited_ = true;
      }
      auto& go = Input(1);

      return SliceImpl<SIndex, Context>(
          nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
    }
  }

  DISABLE_COPY_AND_ASSIGN(SliceGradientOp);

 private:
  std::vector<SIndex> starts_;
  std::vector<SIndex> ends_;
  bool statically_inited_;
  TensorCPU starts_host_;
  TensorCPU ends_host_;
};
} // namespace caffe2
