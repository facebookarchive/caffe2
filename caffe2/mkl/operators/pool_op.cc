#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"

#include "caffe2/utils/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {

namespace {
// These two classe are just used as template arguments passed to the PoolOp
// template to instantiate the different algorithms.
class AveragePool {};
class MaxPool {};
}  // namespace


namespace mkl {

template <typename T, typename PoolType>
class MKLPoolOp final : public ConvPoolOpBase<MKLContext>{

  public:
    USE_CONV_POOL_BASE_FUNCTIONS(MKLContext);
    
    MKLPoolOp(const OperatorDef &operator_def, Workspace *ws)
        : ConvPoolOpBase<MKLContext>(operator_def, ws) {

      CAFFE_ENFORCE(
            dilation_h_ == 1 && dilation_w_ == 1,
            "Pooling op does not support dilation right now.");
        if (!global_pooling_) {
      CAFFE_ENFORCE(
          pad_t_ < kernel_h_ && pad_b_ < kernel_h_ && pad_l_ < kernel_w_ &&
              pad_r_ < kernel_w_,
          "Pad should be smaller than kernel.");
        }
    }
    
    ~MKLPoolOp() {
        if (workspace_buffer_ != NULL) {
      dnnReleaseBuffer<T>(workspace_buffer_);
      workspace_buffer_ = NULL;
      }
    }

    bool RunOnDeviceWithOrderNCHW() override;
    bool RunOnDeviceWithOrderNHWC() override;

     // Input: X
    // Output: Y
private:
    vector<TIndex> cached_maxpool_input_dims_;
    vector<TIndex> cached_avgpool_input_dims_;   

    LayoutWrapper<T> workspace_layout_;
    T *workspace_buffer_ = nullptr;
    PrimitiveWrapper<T> primitive_;
    MKLMemory<T> buffer_;
    void* resources_[dnnResourceNumber] = {0};
  

};
    

template <>
bool MKLPoolOp<float, MaxPool>::RunOnDeviceWithOrderNCHW() {

    auto& X = OperatorBase::Input<MKLMemory<float>>(0);
    MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);

    if (cached_maxpool_input_dims_ != X.dims())  {
        cached_maxpool_input_dims_ = X.dims();
        
        // We will utilize the SetOutputSize() function in the base class
        // with dummy TensorCPU input and output to calculate the sizes.
        TensorCPU dummy_input(X.dims());
        TensorCPU dummy_output;

        ConvPoolOpBase<MKLContext>::SetOutputSize(dummy_input, &dummy_output, X.dim32(1));
               
        size_t dim = X.ndim();

        CAFFE_ENFORCE(4 == dim);

        int paddings[4] = {-pad_l_, -pad_t_, -pad_r_, -pad_b_};
        size_t strides[2] = {stride_w_, stride_h_};
        size_t kernel_size[2] = {kernel_w_, kernel_h_};
        
        // Create main primitive.
        primitive_.Reset(dnnPoolingCreateForward_F32, 
                            nullptr, 
                            dnnAlgorithmPoolingMax, 
                            X.layout(),
                            kernel_size,
                            strides,
                            paddings,
                            dnnBorderZerosAsymm);

       Y->Reset(dummy_output.dims(), primitive_, dnnResourceDst);
       buffer_.Reset(dummy_output.dims(), primitive_, dnnResourceDst, true); 

       workspace_layout_.Reset(primitive_, dnnResourceWorkspace);
       MKLDNN_SAFE_CALL(mkl::dnnAllocateBuffer<float>((void **)(&workspace_buffer_), workspace_layout_));    

    } 

    // Try to share from the output: this allows us to avoid unnecessary copy
    // operations, if the output is already allocated and is having the same
    // layout as the buffer has.
    buffer_.ShareFrom(*Y);
    resources_[dnnResourceSrc] = X.buffer();
    resources_[dnnResourceDst] = buffer_.buffer();
    resources_[dnnResourceWorkspace] = workspace_buffer_;
    MKLDNN_SAFE_CALL(mkl::dnnExecute<float>(primitive_, resources_));
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    return true;
}

template <>
bool MKLPoolOp<float, AveragePool>::RunOnDeviceWithOrderNCHW() {

    auto& X = OperatorBase::Input<MKLMemory<float>>(0);
    MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);

    if (cached_avgpool_input_dims_ != X.dims())  {
        cached_avgpool_input_dims_ = X.dims();
        
        // We will utilize the SetOutputSize() function in the base class
        // with dummy TensorCPU input and output to calculate the sizes.
        TensorCPU dummy_input(X.dims());
        TensorCPU dummy_output;

        ConvPoolOpBase<MKLContext>::SetOutputSize(dummy_input, &dummy_output, X.dim32(1));
               
        size_t dim = X.ndim();

        CAFFE_ENFORCE(4 == dim);

        int paddings[4] = {-pad_l_, -pad_t_, -pad_r_, -pad_b_};
        size_t strides[2] = {stride_w_, stride_h_};
        size_t kernel_size[2] = {kernel_w_, kernel_h_};
        
        // Create main primitive.
        primitive_.Reset(dnnPoolingCreateForward_F32, 
                            nullptr, 
                            dnnAlgorithmPoolingAvg, 
                            X.layout(),
                            kernel_size,
                            strides,
                            paddings,
                            dnnBorderZerosAsymm);

       Y->Reset(dummy_output.dims(), primitive_, dnnResourceDst);
       buffer_.Reset(dummy_output.dims(), primitive_, dnnResourceDst, true); 

       workspace_layout_.Reset(primitive_, dnnResourceWorkspace);
       MKLDNN_SAFE_CALL(mkl::dnnAllocateBuffer<float>((void **)(&workspace_buffer_), workspace_layout_));
    } 

    // Try to share from the output: this allows us to avoid unnecessary copy
    // operations, if the output is already allocated and is having the same
    // layout as the buffer has.
    buffer_.ShareFrom(*Y);
    resources_[dnnResourceSrc] = X.buffer();
    resources_[dnnResourceDst] = buffer_.buffer();
    resources_[dnnResourceWorkspace] = workspace_buffer_;
    MKLDNN_SAFE_CALL(mkl::dnnExecute<float>(primitive_, resources_));
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    return true;
}


template <>
bool MKLPoolOp<float, MaxPool>::RunOnDeviceWithOrderNHWC() {
    CAFFE_NOT_IMPLEMENTED;
}

template <>
bool MKLPoolOp<float, AveragePool>::RunOnDeviceWithOrderNHWC() {
    CAFFE_NOT_IMPLEMENTED;    
}  



} // namespace mkl


REGISTER_MKL_OPERATOR(AveragePool, mkl::MKLPoolOp<float,  AveragePool>);
REGISTER_MKL_OPERATOR(MaxPool, mkl::MKLPoolOp<float,  MaxPool>);


} // namespace caffe2

#endif  // CAFFE2_HAS_MKL_DNN