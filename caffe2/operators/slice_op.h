
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template<class SIndex, class Context>
void slice_(
    const char* src_bytes,
    char* dst_bytes,
    typename std::vector<SIndex>::const_iterator starts_idx,
    typename std::vector<SIndex>::const_iterator ends_idx,
    typename std::vector<TIndex>::const_iterator src_unit_sizes,
    typename std::vector<TIndex>::const_iterator dst_unit_sizes,
    size_t ndim,
    Context* context,
    const TypeMeta& meta,
    bool backward) {
  DCHECK_LE(*starts_idx, *ends_idx);

  size_t src_unit = meta.itemsize() * *src_unit_sizes;
  size_t dst_unit = meta.itemsize() * *dst_unit_sizes;

  const char* src = src_bytes;
  char* dst = dst_bytes;
  if (backward) {
    dst += *starts_idx * dst_unit;
  } else {
    src += *starts_idx * src_unit;
  }
  size_t num_units = *ends_idx - *starts_idx;

  if (ndim == 0) {
    CAFFE_ENFORCE_EQ(*src_unit_sizes, *dst_unit_sizes);
    context->template CopyItems<Context, Context>(
      meta,
      num_units * *src_unit_sizes,
      src,
      dst
    );
  } else {
    for (size_t i = 0; i != num_units; ++i) {
      slice_<SIndex, Context>(
        src + i * src_unit,
        dst + i * dst_unit,
        starts_idx + 1,
        ends_idx + 1,
        src_unit_sizes + 1,
        dst_unit_sizes + 1,
        ndim - 1,
        context,
        meta,
        backward
      );
    }
  }
}

template<class SIndex>
int _last_nonfull_dimension(
    const std::vector<TIndex>& dims,
    const TensorCPU& starts,
    const TensorCPU& ends
) {
  auto* starts_data = starts.template data<SIndex>();
  auto* ends_data = ends.template data<SIndex>();

  for (int i = dims.size()-1; i >= 0; --i) {
    if (
        dims[i] != 0 &&
        i < starts.size() &&
        (starts_data[i] != 0 || ends_data[i] < dims[i])
    ) {
      return i;
    }
  }
  return -1;
}

template <class SIndex, class Context>
bool SliceImpl(
    Tensor<Context>* output,
    const Tensor<Context>& data,
    const TensorCPU& starts,
    const TensorCPU& ends,
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

  // Compute start and end indices
  const int last_nonfull_dimension = _last_nonfull_dimension<SIndex>(data.dims(), starts, ends);

  // optimize for full-data case
  if (last_nonfull_dimension == -1) {
    if (!backward) {
      output->CopyFrom(data, context);
    } else {
      gdata->CopyFrom(*go, context);
    }
    return true;
  }

  std::vector<SIndex> starts_idx(data.ndim());
  std::vector<SIndex> ends_idx(data.ndim());
  std::vector<SIndex> dst_sizes(data.ndim());

  for (int i = 0; i <= data.ndim(); ++i) {
    if (i >= starts.size()) {
      starts_idx[i] = 0;
      ends_idx[i] = data.dims()[i];
      dst_sizes[i] = data.dims()[i];
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

  // optimize for empty-input case
  if (data.size() <= 0) {
    // When the input is empty, we do not need to do copy.
    if (!backward) {
      output->Resize(dst_sizes);
      output->raw_mutable_data(data.meta());
    }
    return true;
  }

  // postfix product of the dimensions is the unit size for each recursion step
  // precompute it here instead of in each recursion step for performance.
  std::vector<TIndex> src_unit_sizes(data.ndim());
  std::vector<TIndex> dst_unit_sizes(data.ndim());
  src_unit_sizes.back() = 1;
  dst_unit_sizes.back() = 1;
  for (int i = dst_unit_sizes.size()-1; i > 0; --i) {
    src_unit_sizes[i-1] = src_unit_sizes[i] * data.dims()[i];
    dst_unit_sizes[i-1] = dst_unit_sizes[i] * dst_sizes[i];
  }

  char* src_bytes = nullptr;
  char* dst_bytes = nullptr;

  if (!backward) {
    output->Resize(dst_sizes);

    src_bytes = (char*)data.raw_data();
    dst_bytes = (char*)output->raw_mutable_data(data.meta());
  } else {
    gdata->ResizeLike(data);

    src_bytes = (char*)go->raw_data();
    dst_bytes = (char*)gdata->raw_mutable_data(go->meta());

    // Zero out gradient blob before copy since we copy in fewer items than
    // there is space for
    math::Set<char, Context>(gdata->nbytes(), 0, dst_bytes, context);

    // If output tensor is empty, just return zeroed gradient tensor
    if (!src_bytes) {
      return true;
    }

    std::swap(src_unit_sizes, dst_unit_sizes);
  }

  // Do actual slicing, i.e. call the recursive algorithm
  slice_<SIndex, Context>(src_bytes, dst_bytes, starts_idx.begin(), ends_idx.begin(), src_unit_sizes.begin(), dst_unit_sizes.begin(), last_nonfull_dimension, context, data.meta(), backward);

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
