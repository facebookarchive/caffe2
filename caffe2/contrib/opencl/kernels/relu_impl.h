#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_RELU_IMPL_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_RELU_IMPL_H_

static constexpr const char* kRelu= R"CLC(
__kernel void K(
  global REAL4* input,
  global REAL4* output
) {
  const int i = get_global_id(0);
  output[i] = input[i] > 0 ? input[i] : 0;
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_RELU_IMPL_H_
