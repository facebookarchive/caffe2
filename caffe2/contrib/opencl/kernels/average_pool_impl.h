#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_AVERAGE_POOL_IMPL_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_AVERAGE_POOL_IMPL_H_

static constexpr const char* kAveragePool= R"CLC(
__kernel void K(
  global REAL* input,
  global REAL* output
) {
  const x = get_global_id(0);
  const y = get_global_id(1);
  const c = get_global_id(2);

  float total = 0;
  for (int i = 0; i < KERNEL; ++i) {
    for (int j = 0; j < KERNEL; ++j) {
      total += input[(c * HEIGHT + y * STRIDE + i) * WIDTH + x * STRIDE + j];
    }
  }
  output[(c * OUT_HEIGHT + y) * OUT_WIDTH + x] = total / (KERNEL * KERNEL);
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_AVERAGE_POOL_IMPL_H_



