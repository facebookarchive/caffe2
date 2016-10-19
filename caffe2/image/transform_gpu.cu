#include "caffe2/core/context_gpu.h"
#include "caffe2/image/transform_gpu.h"

/**
 *
 * Copyright (c) 2016, NVIDIA CORPORATION, All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **/

namespace caffe2 {

namespace {

// input in (int8, NHWC), output in (fp32, NCHW)
template <typename In, typename Out>
__global__
void transform_kernel(const int N, const int C, const int H, const int W,
                      const float mean, const float std, const In* in, Out* out) {
  const int n = blockIdx.x;

  const int nStride = C*H*W;

  // pointers to data for this image
  const In* input_ptr = &in[n*nStride];
  Out* output_ptr = &out[n*nStride];

  // either read or write uncoalesced - try reading
  for (int c=0; c < C; ++c) {
    for (int h=threadIdx.y; h < H; h += blockDim.y) {
      for (int w=threadIdx.x; w < W; w += blockDim.x) {
        int in_idx = c + C*w + C*W*h;  // HWC
        int out_idx = c*H*W + h*W + w;  // CHW

        //out[out_idx] = static_cast<Out>(
        //                static_cast<In>(in[in_idx]-mean)/std);
        output_ptr[out_idx] = (static_cast<Out>(input_ptr[in_idx])-mean) / std;
      }
    }
  }
}

}

template <typename T_IN, typename T_OUT, class Context>
bool TransformOnGPU(Tensor<Context>& X, Tensor<Context> *Y, T_OUT mean, T_OUT std, Context *context) {};

template <>
bool TransformOnGPU<uint8_t, float, CUDAContext>(Tensor<CUDAContext>& X, Tensor<CUDAContext> *Y, float std, float mean, CUDAContext *context)
{
  // data comes in as NHWC
  const int N = X.dim32(0), C = X.dim32(3), H = X.dim32(1), W = X.dim32(2);
  // data goes out as NCHW
  Y->Resize(std::vector<int>{N,C,H,W});

  auto* input_data = X.data<uint8_t>();
  auto* output_data = Y->mutable_data<float>();

  transform_kernel<uint8_t,float><<<N, dim3(16,16), 0, context->cuda_stream()>>>(N,C,H,W, mean, std, input_data, output_data);
  return true;
}

}  // namespace caffe2
