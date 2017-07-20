#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_UTILS_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_UTILS_H_

static constexpr const char* kHalfToFloat = R"CLC(
kernel void K(
  global const half* X,
  global float* Y
){
  int index = get_global_id(0);
  Y[index] = convert_float(X[index]);
}
)CLC";

static constexpr const char* kFloatToHalf = R"CLC(
kernel void K(
  global const float* X,
  global half* Y
){
  int index = get_global_id(0);
  Y[index] = convert_half(X[index]);
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_UTILS_H_
