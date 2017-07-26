#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_ADD_IMPL_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_ADD_IMPL_H_

static constexpr const char* kAdd= R"CLC(
__kernel void K(
  global REAL4* inputA,
  global REAL4* inputB,
  global REAL4* output
) {
  const int i = get_global_id(0);
  output[i] = inputA[i] + inputB[i];
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_ADD_IMPL_H_

