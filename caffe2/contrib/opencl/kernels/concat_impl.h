#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_CONCAT_IMPL_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_CONCAT_IMPL_H_

static constexpr const char* kConcat= R"CLC(
__kernel void K(
  global REAL4* inputA,
  global REAL4* inputB,
  global REAL4* output
) {
  const i = get_global_id(0);
  if (i < A_SIZE4) {
    output[i] = inputA[i];
  } else {
    const int j = i - A_SIZE4;
    output[i] = inputB[j];
  }
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_CONCAT_IMPL_H_


