#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_CHANNEL_SHUFFLE_IMPL_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_CHANNEL_SHUFFLE_IMPL_H_

static constexpr const char* kChannelShuffle= R"CLC(
__kernel void K(
  global REAL* input,
  global REAL* output
) {
  const int i = get_global_id(0);
  const int out_s = i % SPATIAL;
  const int i_2 = i / SPATIAL;
  const int out_c = i_2 % CHANNEL;

  const int g = out_c % GROUP;
  const int k = out_c / GROUP;
  const int in_c = k + (CHANNEL / GROUP) * g;
  output[out_s + SPATIAL * out_c] = input[out_s + SPATIAL * in_c]; 
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_CHANNEL_SHUFFLE_IMPL_H_

