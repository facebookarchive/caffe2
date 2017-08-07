#ifndef MATH_H
#define MATH_H

#include "caffe2/core/tensor.h"

namespace caffe2 {

template<typename C>
void mean_stdev_tensor(const Tensor<C> &tensor, float *mean_out, float *stdev_out) {
  auto data = tensor.template data<float>();
  std::vector<float> values(data, data + tensor.size());
  float sum = std::accumulate(values.begin(), values.end(), 0.0);
  float mean = sum / values.size();
  std::vector<float> diff(values.size());
  std::transform(values.begin(), values.end(), diff.begin(), [mean](float x) { return x - mean; });
  float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  float stdev = std::sqrt(sq_sum / values.size());
  if (mean_out) *mean_out = mean;
  if (stdev_out) *stdev_out = stdev;
}

template<typename C>
void min_max_tensor(const Tensor<C> &tensor, float *min_out, float *max_out) {
  auto data = tensor.template data<float>();
  if (min_out) *min_out = *std::min_element(data, data + tensor.size());
  if (max_out) *max_out = *std::max_element(data, data + tensor.size());
}

template<typename C>
void affine_scale_tensor(Tensor<C> &tensor, float scale, float bias = 0) {
  auto data = tensor.template mutable_data<float>();
  for (auto b = data, e = b + tensor.size(); b != e; b++) {
    *b = *b * scale + bias;
  }
}

template<typename C>
void diag_step_size_tensor(const Tensor<C> &tensor, int *step_out, int *size_out) {
  auto step = 0;
  auto size = tensor.dim(0);
  for (auto d: tensor.dims()) {
    step = step * d + 1;
    if (size > d) size = d;
  }
  if (step_out) *step_out = step;
  if (size_out) *size_out = size;
}

template<typename C>
int offset_in_tensor(const Tensor<C> &tensor, const std::vector<TIndex> &offset) {
  auto off = 0, i = 0;
  for (auto d: tensor.dims()) {
    off = off * d + offset[i++];
  }
  return off;
}

template<typename C>
void get_diagonal_tensor(const Tensor<C> &tensor, Tensor<C> &diagonal, const std::vector<TIndex> &offset) {
  auto off = offset_in_tensor(tensor, offset);
  auto step = 0, size = 0;
  diag_step_size_tensor(tensor, &step, &size);
  diagonal.Resize(size);
  auto data = tensor.template data<float>() + off;
  auto diagonal_data = diagonal.template mutable_data<float>();
  for (auto i = 0; i < size; i++, data += step, diagonal_data++) {
    *diagonal_data = *data;
  }
}

template<typename C>
void set_diagonal_tensor(Tensor<C> &tensor, const Tensor<C> &diagonal, const std::vector<TIndex> &offset) {
  auto off = offset_in_tensor(tensor, offset);
  auto step = 0, size = 0;
  diag_step_size_tensor(tensor, &step, &size);
  auto data = tensor.template mutable_data<float>() + off;
  auto diagonal_data = diagonal.template data<float>();
  for (auto i = 0, s = (int)diagonal.size(); i < size; i++, data += step, diagonal_data++) {
    *data = i < s ? *diagonal_data : 0;
  }
}

template<typename C>
void get_back_mean_tensor(const Tensor<C> &tensor, Tensor<C> &mean, int count = 1) {
  auto dims = tensor.dims();
  auto size = 1;
  while (count--) {
    size *= dims.back();
    dims.pop_back();
  }
  mean.Resize(dims);
  auto data = tensor.template data<float>();
  auto mean_data = mean.template mutable_data<float>();
  auto mean_end = mean_data + mean.size();
  for (auto e = data + tensor.size(); data != e && mean_data != mean_end; mean_data++) {
    auto sum = 0.f;
    for (auto g = data + size; data != g; data++) {
      sum += *data;
    }
    *mean_data = sum / size;
  }
}

template<typename C>
void set_back_mean_tensor(Tensor<C> &tensor, const Tensor<C> &mean, int count = 1) {
  auto dims = tensor.dims();
  auto size = 1;
  while (count--) {
    size *= dims.back();
    dims.pop_back();
  }
  auto data = tensor.template mutable_data<float>();
  auto mean_data = mean.template data<float>();
  auto mean_end = mean_data + mean.size();
  for (auto e = data + tensor.size(); data != e && mean_data != mean_end; mean_data++) {
    for (auto g = data + size; data != g; data++) {
      *data = *mean_data / size;
    }
  }
}

template<typename C>
void get_affine_scale_tensor(const Tensor<C> &tensor, const Tensor<C> &mean, const Tensor<C> &scale, Tensor<C> &transformed, bool inverse = false) {
  auto data = tensor.template data<float>();
  auto size = tensor.size() / tensor.dim(0);
  auto mean_data = mean.template data<float>();
  auto scale_data = scale.template data<float>();
  auto transformed_data = transformed.template mutable_data<float>();
  for (auto e = data + tensor.size(); data != e; mean_data++, scale_data++) {
    for (auto f = data + size; data != f; data++, transformed_data++) {
      if (inverse) {
        *transformed_data = (*data - *mean_data) / (*scale_data + 1e-8);
      } else {
        *transformed_data = *data * *scale_data + *mean_data;
      }
    }
  }
}

template<typename C>
void set_affine_scale_tensor(Tensor<C> &tensor, const Tensor<C> &scale, const Tensor<C> &transformed, bool inverse = false) {
  auto data = tensor.template mutable_data<float>();
  auto size = tensor.size() / tensor.dim(0);
  auto scale_data = scale.template data<float>();
  auto transformed_data = transformed.template data<float>();
  for (auto e = data + tensor.size(); data != e; scale_data++) {
    for (auto f = data + size; data != f; data++, transformed_data++) {
      if (inverse) {
        *data = *transformed_data / (*scale_data + 1e-8);
      } else {
        *data = *transformed_data * *scale_data;
      }
    }
  }
}

template<typename C>
void get_mean_stdev_tensor(const Tensor<C> &tensor, Tensor<C> &mean, Tensor<C> &stdev) {
  auto data = tensor.template data<float>();
  auto size = tensor.size() / tensor.dim(0);
  auto mean_data = mean.template mutable_data<float>();
  auto stdev_data = stdev.template mutable_data<float>();
  for (auto e = data + tensor.size(); data != e; data += size, mean_data++, stdev_data++) {
    auto sum = 0.f;
    for (auto d = data, e = data + size; d != e; d++) {
      sum += *d;
    }
    auto mean = sum / size;
    auto sq_sum = 0.f;
    for (auto d = data, e = data + size; d != e; d++) {
      sq_sum += (*d - mean) * (*d - mean);
    }
    auto stdev = sqrt(sq_sum / size);
    *mean_data = mean;
    *stdev_data = stdev;
  }
}

template<typename C>
float mean_diagonal_tensor(const Tensor<C> &tensor) {
  auto step = 0, size = 0;
  diag_step_size_tensor(tensor, &step, &size);
  auto data = tensor.template data<float>();
  auto sum = 0.f;
  for (auto i = 0; i < size; i++, data += step) {
    sum += *data;
  }
  return sum / size;
}

template<typename C>
void set_diagonal_tensor(Tensor<C> &tensor, float value) {
  auto step = 0, size = 0;
  diag_step_size_tensor(tensor, &step, &size);
  auto data = tensor.template mutable_data<float>();
  for (auto i = 0; i < size; i++, data += step) {
    *data = value;
  }
}

template<typename C>
void add_tensor(Tensor<C> &tensor, const Tensor<C> &tensor_add) {
  CHECK(tensor.size() == tensor_add.size());
  auto data = tensor.template mutable_data<float>();
  auto b_add = tensor_add.template data<float>();
  for (auto b = data, e = b + tensor.size(); b != e; b++, b_add++) {
    *b += *b_add;
  }
}

template<typename C, typename T>
void set_tensor_at(Tensor<C> &tensor, std::vector<int> position, T value) {
  CHECK(tensor.dims().size() == position.size());
  auto pos = 0;
  auto size = 1;
  for (int i = position.size(); i > 0; --i) {
    pos += position[i - 1] * size;
    size *= tensor.dim(i - 1);
  }
  auto data = tensor.template mutable_data<T>();
  data[pos] = value;
}

template<typename C, typename T>
void set_tensor(Tensor<C> &tensor, std::vector<T> values) {
  CHECK(tensor.size() == values.size());
  Tensor<C> t(tensor.dims(), values, NULL);
  tensor.CopyFrom(t);
}

template<typename C>
void clip_tensor(Tensor<C> &tensor, float min, float max) {
  auto data = tensor.template mutable_data<float>();
  for (auto b = data, e = b + tensor.size(); b != e; b++) {
    if (*b < min) {
      *b = min;
    } else if (*b > max) {
      *b = max;
    }
  }
}

}  // namespace caffe2

#endif  // MATH_H
