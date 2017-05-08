#include "caffe2/operators/norm_op.h"


namespace caffe2 {

template <>
bool NormOp<float, CPUContext>::RunOnDevice() {
	const auto& X = Input(0);
	const auto& scale = Input(1);
	auto* Y = Output(0);

	int batch_size = X.dim(0);
	int channels = X.dim(1);
	int height = X.dim(2);
	int width = X.dim(3);

	if(channel_shared_) {
		CAFFE_ENFORCE_EQ(scale.dim(0), 1);
	} else {
		CAFFE_ENFORCE_EQ(scale.dim(0), channels);
	}

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
	math::Set<float, CPUContext>(channels, 
			float(1.0), sum_channel_multiplier_.template mutable_data<float>(),
			&context_);
	int spatial_dim = height * width;
	sum_spatial_multiplier_.Resize(1, 1, height, width);
	math::Set<float, CPUContext>(spatial_dim, 
			float(1.0), sum_spatial_multiplier_.template mutable_data<float>(),
			&context_);

	const float* Xdata = X.data<float>();
	const float* Sdata = scale.data<float>();
	float* buffer_data = buffer_.mutable_data<float>();
	float* norm_data = norm_.mutable_data<float>();
	math::Set<float, CPUContext>(norm_.size_from_dim(0),
			eps_, norm_data, &context_);
	const float* sum_channel_multiplier = sum_channel_multiplier_.data<float>();
	const float* sum_spatial_multiplier = sum_spatial_multiplier_.data<float>();
	float* Ydata = Y->mutable_data<float>();
	int dim = channels * height * width;

	for(int n = 0; n < batch_size; n++) {
		math::Sqr<float, CPUContext>(dim, Xdata, buffer_data, &context_);
		if(across_spatial_) {
			float accu_sum;
			math::Sum<float, CPUContext>(dim, buffer_data, &accu_sum, &context_);
			norm_data[n] = pow(accu_sum+eps_, float(0.5));
			math::Scale<float, CPUContext>(dim, float(1.0 / norm_data[n]), Xdata, Ydata, 
					&context_);
		} else {
			math::Gemv<float, CPUContext>(CblasTrans, channels, spatial_dim,
								float(1.0), buffer_data, sum_channel_multiplier, float(1.0),
								norm_data, &context_);
			math::Powx<float, CPUContext>(spatial_dim, norm_data, float(0.5), 
								norm_data, &context_);
			math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
								1, float(1.0), sum_channel_multiplier, norm_data, float(0.0),
								buffer_data, &context_);
			math::Div<float, CPUContext>(dim, Xdata, buffer_data, Ydata, &context_);
			norm_data += spatial_dim;
		}
		if (channel_shared_) {
			math::Scale<float, CPUContext>(dim, Sdata[0], Ydata, Ydata, &context_);
		} else {
			math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
								1, float(1.0), Sdata, sum_spatial_multiplier, float(0.0), buffer_data,
								&context_);
			math::Mul<float, CPUContext>(dim, Ydata, buffer_data, Ydata, &context_);
		}
		Xdata += dim;
		Ydata += dim;
	}
	return true;
}


namespace {

REGISTER_CPU_OPERATOR(Norm, NormOp<float, CPUContext>);
NO_GRADIENT(Norm);

// Input: X, scale
// Output: Y
OPERATOR_SCHEMA(Norm)
		.NumInputs(2)
		.NumOutputs(1)
		.IdenticalTypeAndShape()
		.SetDoc(R"Doc(SSD L2 and Scale layer)Doc")
		.Arg("across_spatial", "default: false")
		.Arg("order", "default: NCHW")
		.Arg("eps", "default: 10e-10")
		.Arg("channel_shared", "default: false")
		.Input(0, "X", "NCHW tensor")
		.Input(1, "scale", "scale factor")
		.Output(0, "Y", "NCHW tensor");

} // namespace


} // namespace caffe2


