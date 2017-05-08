#include "caffe2/operators/norm_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {
namespace {

// divid a matrix with vector
template <typename T>
__global__ void DivBsx(const int nthreads, const T* A,
    const T* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    T* B) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] / v[c];
    } else {
      B[index] = A[index] / v[r];
    }
  }
}

template <typename T>
__global__ void MulBsx(const int nthreads, const T* A,
    const T* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
   	T* B) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] * v[c];
    } else {
      B[index] = A[index] * v[r];
    }
  }
}

REGISTER_CUDA_OPERATOR(Norm, NormOp<float, CUDAContext>);
} // namespace

template <>
bool NormOp<float, CUDAContext>::RunOnDevice() {
	auto& X = Input(0);
	auto& scale = Input(1);
	auto* Y = Output(0);

	int batch_size = X.dim(0);
	int channels = X.dim(1);
	int height = X.dim(2);
	int width = X.dim(3);

	Y->ResizeLike(X);
	buffer_.Resize(1, channels, height, width);
	buffer_channel_.Resize(1, channels, 1, 1);
	buffer_spatial_.Resize(1, 1, height, width);

	if(across_spatial_) {
		norm_.Resize(batch_size, 1, 1, 1);
	} else {
		norm_.Resize(batch_size, 1, height, width);
	}
	sum_channel_multiplier_.Resize(1, channels, 1, 1);
	math::Set<float, CUDAContext>(channels,
			float(1.0), sum_channel_multiplier_.mutable_data<float>(),
			&context_);
	int spatial_dim = height * width;
	sum_spatial_multiplier_.Resize(1, 1, height, width);
	math::Set<float, CUDAContext>(spatial_dim,
			float(1.0), sum_spatial_multiplier_.mutable_data<float>(),
			&context_);

	const float* Xdata = X.data<float>();
	const float* Sdata = scale.data<float>();
	float* buffer_data = buffer_.mutable_data<float>();
	float* norm_data = norm_.mutable_data<float>();
	math::Set<float, CUDAContext>(norm_.size_from_dim(0),
			eps_, norm_data, &context_);
	const float* sum_channel_multiplier = sum_channel_multiplier_.data<float>();
	const float* sum_spatial_multiplier = sum_spatial_multiplier_.data<float>();
	float* Ydata = Y->mutable_data<float>();
	int dim = channels * height * width;

	for(int n = 0; n < batch_size; n++) {
		math::Powx<float, CUDAContext>(dim, Xdata, float(2.0), buffer_data, &context_);
		if(across_spatial_) {
			float accu_sum;
			math::Sum<float, CUDAContext>(dim, buffer_data, &accu_sum, &context_);
			norm_data[n] = pow(accu_sum+eps_, float(0.5));
			math::Scale<float, CUDAContext>(dim, float(1.0 / norm_data[n]), Xdata, Ydata,
					&context_);
		} else {
			math::Gemv<float, CUDAContext>(CblasTrans, channels, spatial_dim,
								float(1.0), buffer_data, sum_channel_multiplier, float(1.0),
								norm_data, &context_);
			math::Powx<float, CUDAContext>(spatial_dim, norm_data, float(0.5),
								norm_data, &context_);
			DivBsx<float> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS, 0, 
				context_.cuda_stream()>>> (
					dim, Xdata, norm_data, channels, spatial_dim, CblasNoTrans, Ydata);
			norm_data += spatial_dim;
		}
		if (channel_shared_) {
			math::Scale<float, CUDAContext>(dim, Sdata[0], Ydata, Ydata, &context_);
		} else {
			MulBsx<float> <<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS, 0, 
				context_.cuda_stream()>>> (
					dim, Ydata, Sdata, channels, spatial_dim, CblasTrans, Ydata);
		}
		Xdata += dim;
		Ydata += dim;
	}
	return true;
}

} // namespace caffe2
