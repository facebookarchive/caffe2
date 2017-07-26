#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_SPATIAL_BN_IMPL_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_SPATIAL_BN_IMPL_H_

static constexpr const char* kSpatialBN= R"CLC(
__kernel void K(
  global REAL* input,
  const int spatial_size,
  global REAL* scale,
  global REAL* bias,
  global REAL* mean,
  global REAL* var,
  global REAL* output
) {
  const int i = get_global_id(0);
  const int c = get_global_id(1);
  const float new_scale = rsqrt(var[c]) * scale[c];
  const float new_bias = bias[c] - (mean[c] * new_scale);
  output[c * spatial_size + i] = input[c * spatial_size + i];
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_SPATIAL_BN_IMPL_H_

