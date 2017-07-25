#ifndef CAFFE2_CONTRIB_OPENCL_KERNELS_CONV_IMPL_H_
#define CAFFE2_CONTRIB_OPENCL_KERNELS_CONV_IMPL_H_

static constexpr const char* k3x3DW = R"CLC(
__kernel void K(
  global REAL* filter,
  global REAL* input,
  const int H_in,
  const int H_out,
  const int W_in,
  const int W_out,
  global REAL* output) {
#define Y_TILE 4
#define X_TILE 4
#define KERNEL 3
  const int x_base = get_global_id(0) << 2;
  const int y_base = get_global_id(1) << 2;
  const int c = get_global_id(2);
  for (int _y = 0; _y < Y_TILE; ++_y) {
    const int y = _y + y_base;
    if (y >= H_out) return;
    for (int _x = 0; _x < X_TILE; ++_x) {
      const int x = _x + x_base;
      if (x >= W_out) continue;
      float accum = 0;
      for (int i = 0; i < KERNEL; ++i) {
        for (int j = 0; j < KERNEL; ++j) {
          accum += input[(c * H_in + (y * STRIDE + i)) * W_in + (x * STRIDE + j)] *
                   filter[(c * KERNEL + i) * KERNEL + j];
        }
      }
      output[(c * H_out + y) * W_out + x] = accum;
    }
  }
}
)CLC";

static constexpr const char* k1x1Gemm = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// This is taken directly from here:
// https://www.qualcomm.com/news/onq/2016/10/17/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
__kernel void gemm(
  __global const REAL *A,
  const int lda,
  __global const REAL *B,
  const int B_offset,
  __global REAL *C,
  const int ldc,
  const int m,
  const int n,
  const int k
) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  const sampler_t sampler = CLK_ADDRESS_NONE;

  if (((gx << 2) < n) && ((gy << 3) < m)) {
    REAL4 a[8];
    REAL4 b[4];
    REAL4 c[8];

    for (int i = 0; i < 8; i++){
        c[i] = 0.0f;
    }

    int A_y_off = (gy << 3) * lda;

    for (int pos = 0; pos < k; pos += 4) {
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        const int B_off = B_offset + (pos + i) * n + (gx << 2);
        b[i] = vload4(0, B + B_off);
      }

      int A_off = A_y_off + pos;

      #pragma unroll
      for (int i = 0; i < 8; i++) {
        a[i] = vload4(0, A + A_off);
        A_off += lda;
      }

      #pragma unroll
      for (int i = 0; i < 8; i++) {
        c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
      }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int C_offs = ((gy << 3) + i) * ldc + (gx << 2);
      vstore4(c[i], 0, C + C_offs);
    }
  }
}

// https://arxiv.org/pdf/1706.06873.pdf
__kernel void K(
  __global const REAL *A,
  const int lda,
  __global const REAL *B,
  __global REAL *C,
  const int ldc,
  const int m,
  const int n,
  const int k
) {
  const int g = get_global_id(2);
  const int B_offset = g * n * IN_CHANNEL_DIV_G;
  gemm(&A[g * FILTER_DIV_G], lda, B, B_offset, &C[g * ldc * OUT_CHANNEL_DIV_G], ldc, m, n, k);
}
)CLC";

static constexpr const char* kMECGemm = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// This is taken directly from here:
// https://www.qualcomm.com/news/onq/2016/10/17/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
__kernel void gemm(
  __global const REAL *A,
  const int lda,
  __read_only image2d_t Bi,
  __global REAL *C,
  const int ldc,
  const int m,
  const int n,
  const int k
) {
  int gx = get_global_id(0);
  int gy = get_global_id(1);
  const sampler_t sampler = CLK_ADDRESS_NONE;

  if (((gx << 2) < n) && ((gy << 3) < m)) {
    REAL4 a[8];
    REAL4 b[4];
    REAL4 c[8];

    for (int i = 0; i < 8; i++){
        c[i] = 0.0f;
    }

    int A_y_off = (gy << 3) * lda;

    for (int pos = 0; pos < k; pos += 4) {
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        // This access pattern is a perf hit
        b[i] = READ_IMAGE(Bi, sampler, (int2)(gx, pos + i));
      }

      int A_off = A_y_off + pos;

      #pragma unroll
      for (int i = 0; i < 8; i++) {
        a[i] = vload4(0, A + A_off);
        A_off += lda;
      }

      #pragma unroll
      for (int i = 0; i < 8; i++) {
        c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
      }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int C_offs = ((gy << 3) + i) * ldc + (gx << 2);
      vstore4(c[i], 0, C + C_offs);
    }
  }
}

// https://arxiv.org/pdf/1706.06873.pdf
__kernel void K(
  __global const REAL *A,
  const int lda,
  __read_only image2d_t Bi,
  __global REAL *C,
  const int ldc,
  const int m,
  const int n,
  const int k,
  const int iters
) {
  const int i = get_global_id(2);
  gemm(&A[KERNEL * i * IN_CHANNEL], lda, Bi, &C[i * m * OUT_CHANNEL], ldc, m, n, k);
}
)CLC";

// https://arxiv.org/pdf/1706.06873.pdf
static constexpr const char* kMECLowering = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void K(
  constant REAL* input,
  const int H_in,
  const int H_out,
  const int W_in,
  const int W_out,
  global REAL* L
){
  const int x_o = get_global_id(0);
  const int y_i = get_global_id(1);
  const int c = get_global_id(2) << 2;
  #pragma unroll
  for (int j = 0; j < KERNEL; ++j) {
    const REAL4 elem = vload4(0,
      input + IN_CHANNEL * y_i * W_in +
              IN_CHANNEL * x_o + 
              IN_CHANNEL * j + c );
    vstore4(elem, 0,
      L + IN_CHANNEL * KERNEL * H_in * x_o + 
          IN_CHANNEL * KERNEL * y_i +
          IN_CHANNEL * j + c);
  }
}
)CLC";

static constexpr const char* kDirectConv = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void K(
  __read_only image2d_t filter,
  global REAL4* input,
  const int H_in,
  const int H_out,
  const int W_in,
  const int W_out,
  global REAL4* output
){
#define X_TILE 4
#define C_O_TILE 4
#define C_I (IN_CHANNEL >> 2) // adjusted channel in
#define C_O (OUT_CHANNEL >> 2) // adjusted channel out

  const int x_o = get_global_id(0) * X_TILE;
  const int y_o = get_global_id(2);
  const int c_o = get_global_id(1);
  const sampler_t sampler = CLK_ADDRESS_NONE;

  REAL4 value[X_TILE];
  #pragma unroll
  for (int k = 0; k < X_TILE; ++k) {
    value[k] = 0;
  }

  for (int i = 0; i < KERNEL; ++i) {
  const int y_i = y_o + i;

  for (int c_i = 0; c_i < C_I; ++c_i) {

  for (int j = 0; j < KERNEL; ++j) {
  const int x_i = x_o + j;

    REAL4 a[C_O_TILE];
    #pragma unroll
    for (int a_i = 0; a_i < C_O_TILE; ++a_i) {
      const int f_x = KERNEL * C_I * i + C_I * j + c_i;
      const int f_y = (c_o * C_O_TILE) + a_i;
      a[a_i] = READ_IMAGE(filter, sampler, (int2)(f_x, f_y));
    }

    REAL4 b[X_TILE];
    #pragma unroll
    for (int b_i = 0; b_i < X_TILE; ++b_i) {
      b[b_i] = input[ W_in * C_I * y_i +
                             C_I * (x_i + b_i) +
                                   c_i ];
    }

    #pragma unroll
    for (int k = 0; k < X_TILE; ++k) {
      value[k].s0 += dot(b[k], a[0]);
      value[k].s1 += dot(b[k], a[1]);
      value[k].s2 += dot(b[k], a[2]);
      value[k].s3 += dot(b[k], a[3]);
    }

  }

  }
  }

  #pragma unroll
  for (int k = 0; k < X_TILE; ++k) {
    output[ W_out * C_O * y_o +
                    C_O * (x_o + k) +
                          c_o ] = value[k];
  }
}
)CLC";

#endif // CAFFE2_CONTRIB_OPENCL_KERNELS_CONV_IMPL_H_
