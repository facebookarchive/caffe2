
# Accumulate

Accumulate operator accumulates the input tensor to the output tensor. If the output tensor already has the right size, we add to it; otherwise, we first initialize the output tensor to all zeros, and then do accumulation. Any further calls to the operator, given that no one else fiddles with the output in the interim, will do simple accumulations.
Accumulation is done using Axpby operation as shown:  
````
  Y = 1*X + gamma*Y

````
 where X is the input tensor, Y is the output tensor and gamma is the multiplier argument.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`gamma`
</td><td>(float, default 1.0) Accumulation multiplier
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>The input tensor that has to be accumulated to the output tensor. If the output size is not the same as input size, the output tensor is first reshaped and initialized to zero, and only then, accumulation is done.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Accumulated output tensor
</td></tr></table>
### Code
[caffe2/operators/accumulate_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/accumulate_op.cc)
### Devices

- *CPU* `caffe2::AccumulateOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::AccumulateOp<float, caffe2::CUDAContext>`




---


# Accuracy

Accuracy takes two inputs- predictions and labels, and returns a float accuracy value for the batch. Predictions are expected in the form of 2-D tensor containing a batch of scores for various classes, and labels are expected in the  form of 1-D tensor containing true label indices of samples in the batch. If the score for the label index in the predictions is the highest among all classes, it is considered a correct prediction.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`predictions`
</td><td>2-D tensor (Tensor<float>) of size (num_batches x num_classes) containing scores
</td></tr><tr><td>`labels`
</td><td>1-D tensor (Tensor<int>) of size (num_batches) having the indices of true labels
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`accuracy`
</td><td>1-D tensor (Tensor<float>) of size 1 containing accuracy
</td></tr></table>
### Code
[caffe2/operators/accuracy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/accuracy_op.cc)
### Devices

- *CPU* `caffe2::AccuracyOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::AccuracyOp<float, caffe2::CUDAContext>`




---


# Add

Performs element-wise binary addition (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and type as A
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::EigenAddFunctor, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaAddFunctor, caffe2::SameTypeAsInput>`




---


# AddPadding

Given a partitioned tensor T<N, D1..., Dn>, where the partitions are defined as ranges on its outer-most (slowest varying) dimension N, with given range lengths, return a tensor T<N + 2*pad_width, D1 ..., Dn> with paddings added to the start and end of each range.
Optionally, different paddings can be provided for beginning and end. Paddings provided must be a tensor T<D1..., Dn>.
 If no padding is provided, add zero padding.
If no lengths vector is provided, add padding only once, at the start and end of data.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`pad_width`
</td><td>Number of copies of padding to add around each range.
</td></tr><tr><td>`end_pad_width`
</td><td>(Optional) Specifies a different end-padding width.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data_in`
</td><td>(T<N, D1..., Dn>) Input data
</td></tr><tr><td>`lengths`
</td><td>(i64) Num of elements in each range. sum(lengths) = N.
</td></tr><tr><td>`start_padding`
</td><td>T<D1..., Dn> Padding data for range start.
</td></tr><tr><td>`end_padding`
</td><td>T<D1..., Dn> (optional) Padding for range end. If not provided, start_padding is used as end_padding as well.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`data_out`
</td><td>(T<N + 2*pad_width, D1..., Dn>) Padded data.
</td></tr><tr><td>`lengths_out`
</td><td>(i64, optional) Lengths for each padded range.
</td></tr></table>
### Code
[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::AddPaddingOp`




---


# Alias

Makes the output and the input share the same underlying storage.
 WARNING: in general, in caffe2's operator interface different tensors should have different underlying storage, which is the assumption made by components such as the dependency engine and memory optimization. Thus, in normal situations you should not use the AliasOp, especially in a normal forward-backward pass.
 The Alias op is provided so one can achieve true asynchrony, such as Hogwild, in a graph. But make sure you understand all the implications similar to multi-thread computation before you use it explicitly.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>Input tensor whose storage will be shared.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Tensor of same shape as input, sharing its storage.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::AliasOp<caffe2::CPUContext>`

- *GPU* `caffe2::AliasOp<caffe2::CUDAContext>`




---


# Allgather

Does an allgather operation among the nodes.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`comm_world`
</td><td>The common world.
</td></tr><tr><td>`X`
</td><td>A tensor to be allgathered.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>The allgathered tensor, same on all nodes.
</td></tr></table>
### Code
[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)
### Devices

- *CPU* `caffe2::NoDefaultEngineOp<caffe2::CPUContext>`

- *GPU* `caffe2::NoDefaultEngineOp<caffe2::CUDAContext>`




---


# Allreduce

Does an allreduce operation among the nodes. Currently only Sum is supported.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`comm_world`
</td><td>The common world.
</td></tr><tr><td>`X`
</td><td>A tensor to be allreduced.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>The allreduced tensor, same on all nodes.
</td></tr></table>
### Code
[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)
### Devices

- *CPU* `caffe2::NoDefaultEngineOp<caffe2::CPUContext>`

- *GPU* `caffe2::NoDefaultEngineOp<caffe2::CUDAContext>`




---


# And

Performs element-wise logical operation  `and`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<bool>, caffe2::CPUContext, caffe2::NaiveAndFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<bool>, caffe2::CUDAContext, caffe2::CudaAndFunctor, caffe2::FixedType<bool> >`




---


# Append

Append input 2 to the end of input 1.
Input 1 must be the same as output, that is, it is required to be in-place.
Input 1 may have to be re-allocated in order for accommodate to the new size.
Currently, an exponential growth ratio is used in order to ensure amortized constant time complexity.
All except the outer-most dimension must be the same between input 1 and 2.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`dataset`
</td><td>The tensor to be appended to.
</td></tr><tr><td>`new_data`
</td><td>Tensor to append to the end of dataset.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`dataset`
</td><td>Same as input 0, representing the mutated tensor.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::AppendOp<caffe2::CPUContext>`




---


# AtomicAppend
No documentation yet.

### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::AtomicAppendOp<caffe2::CPUContext>`




---


# AtomicFetchAdd

Given a mutex and two int32 scalar tensors, performs an atomic fetch add by mutating the first argument and adding it to the second input argument. Returns the updated integer and the value prior to the update.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`mutex_ptr`
</td><td>Blob containing to a unique_ptr<mutex>
</td></tr><tr><td>`mut_value`
</td><td>Value to be mutated after the sum.
</td></tr><tr><td>`increment`
</td><td>Value to add to the first operand.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`mut_value`
</td><td>Mutated value after sum. Usually same as input 1.
</td></tr><tr><td>`fetched_value`
</td><td>Value of the first operand before sum.
</td></tr></table>
### Code
[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::AtomicFetchAddOp`




---


# AveragePool

AveragePool consumes an input blob X and applies average pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Average pooling consisting of averaging all values of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. 
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
</td></tr></table>
### Code
[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)
### Devices

- *CPU* `caffe2::PoolOp<float, caffe2::CPUContext, caffe2::(anonymous namespace)::AveragePool>`

- *GPU* `caffe2::PoolOp<float, caffe2::CUDAContext, caffe2::(anonymous namespace)::AveragePool>`



### Engines
`CUDNN` on *CUDA*

---


# AveragePoolGradient
No documentation yet.

### Code
[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)
### Devices

- *CPU* `caffe2::PoolGradientOp<float, caffe2::CPUContext, caffe2::(anonymous namespace)::AveragePool>`

- *GPU* `caffe2::PoolGradientOp<float, caffe2::CUDAContext, caffe2::(anonymous namespace)::AveragePool>`



### Engines
`CUDNN` on *CUDA*

---


# AveragedLoss

AveragedLoss takes in a 1-D tensor as input and returns a single output float value which represents the average of input data (average of the losses).

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>The input data as Tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The output tensor of size 1 containing the averaged value.
</td></tr></table>
### Code
[caffe2/operators/loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc)
### Devices

- *CPU* `caffe2::AveragedLoss<float, caffe2::CPUContext>`

- *GPU* `caffe2::AveragedLoss<float, caffe2::CUDAContext>`




---


# AveragedLossGradient
No documentation yet.

### Code
[caffe2/operators/loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc)
### Devices

- *CPU* `caffe2::AveragedLossGradient<float, caffe2::CPUContext>`

- *GPU* `caffe2::AveragedLossGradientGPUSpecialization`




---


# BatchMatMul

Batch Matrix multiplication Yi = Ai * Bi, where A has size (C x M x K), B has size (C x K x N) where C is the batch size and i ranges from 0 to C-1.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`trans_a`
</td><td>Pass 1 to transpose A before multiplication
</td></tr><tr><td>`trans_b`
</td><td>Pass 1 to transpose B before multiplication
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>3D matrix of size (C x M x K)
</td></tr><tr><td>`B`
</td><td>3D matrix of size (C x K x N)
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>3D matrix of size (C x M x N)
</td></tr></table>
### Code
[caffe2/operators/batch_matmul_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/batch_matmul_op.cc)
### Devices

- *CPU* `caffe2::BatchMatMulOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`

- *GPU* `caffe2::BatchMatMulOp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`




---


# BatchToSpace

 BatchToSpace for 4-D tensors of type T.
 Rearranges (permutes) data from batch into blocks of spatial data, followed by cropping. This is the reverse transformation of SpaceToBatch. More specifically, this op outputs a copy of the input tensor where values from the batch dimension are moved in spatial blocks to the height and width dimensions, followed by cropping along the height and width dimensions.
 
### Code
[caffe2/operators/space_batch_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/space_batch_op.cc)
### Devices

- *CPU* `caffe2::BatchToSpaceOp<caffe2::CPUContext>`

- *GPU* `caffe2::BatchToSpaceOp<caffe2::CUDAContext>`




---


# BooleanMask

Given a data 1D tensor and a mask (boolean) tensor of same shape, returns a tensor containing only the elements corresponding to positions where the mask is true.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>The 1D, original data tensor.
</td></tr><tr><td>`mask`
</td><td>A tensor of bools of same shape as `data`.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`masked_data`
</td><td>A tensor of same type as `data`.
</td></tr></table>
### Code
[caffe2/operators/boolean_mask_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/boolean_mask_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::BooleanMaskOp<caffe2::CPUContext>`




---


# BooleanMaskLengths

Given a tensor of int32 segment lengths and a mask (boolean) tensor, return the segment lengths of a corresponding segmented tensor after BooleanMask is applied.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>A 1D int32 tensor representing segment lengths.
</td></tr><tr><td>`mask`
</td><td>A 1D bool tensor of values to keep.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`masked_lengths`
</td><td>Segment lengths of a masked tensor.
</td></tr></table>
### Code
[caffe2/operators/boolean_mask_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/boolean_mask_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::BooleanMaskLengthsOp<caffe2::CPUContext>`




---


# Broadcast

Does a broadcast operation from the root node to every other node. The tensor on each node should have been pre-created with the same shape and data type.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`root`
</td><td>(int, default 0) the root to run broadcast from.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`comm_world`
</td><td>The common world.
</td></tr><tr><td>`X`
</td><td>A tensor to be broadcasted.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>In-place as input 1.
</td></tr></table>
### Code
[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)
### Devices

- *CPU* `caffe2::NoDefaultEngineOp<caffe2::CPUContext>`

- *GPU* `caffe2::NoDefaultEngineOp<caffe2::CUDAContext>`




---


# Cast

The operator casts the elements of a given input tensor to a data type specified by the 'to' argument and returns an output tensor of the same size in the converted type. The 'to' argument must be one of the data types specified in the 'DataType' enum field in the TensorProto message. If the 'to' argument is not provided or is not one of the enumerated types in DataType, Caffe2 throws an Enforce error.
 NOTE: Casting to and from strings is not supported yet.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`to`
</td><td>The data type to which the elements of the input tensor are cast.Strictly must be one of the types from DataType enum in TensorProto
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>Input tensor to be cast.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Output tensor with the same shape as input with type specified by the 'to' argument
</td></tr></table>
### Code
[caffe2/operators/cast_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cast_op.cc)
### Devices

- *CPU* `caffe2::CastOp<caffe2::CPUContext>`

- *GPU* `caffe2::CastOp<caffe2::CUDAContext>`




---


# CheckAtomicBool
Copy the value of a atomic<bool> to a bool
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`atomic_bool`
</td><td>Blob containing a unique_ptr<atomic<bool>>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`value`
</td><td>Copy of the value for the atomic<bool>
</td></tr></table>
### Code
[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::CheckAtomicBoolOp`




---


# CheckCounterDone

If the internal count value <= 0, outputs true, otherwise outputs false, 
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`counter`
</td><td>A blob pointing to an instance of a counter.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`done`
</td><td>true if the internal count is zero or negative.
</td></tr></table>
### Code
[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)
### Devices

- *CPU* `caffe2::CheckCounterDoneOp<long, caffe2::CPUContext>`

- *GPU* `caffe2::CheckCounterDoneOp<long, caffe2::CUDAContext>`




---


# CheckDatasetConsistency

Checks that the given data fields represents a consistent dataset unther the schema specified by the  `fields`  argument. Operator fails if the fields are not consistent. If data is consistent, each field's data can be safely appended to an existing dataset, keeping it consistent.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`fields`
</td><td>List of strings representing the string names in the formatspecified in the doc for CreateTreeCursor.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`field_0`
</td><td>Data for field 0.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::CheckDatasetConsistencyOp`




---


# Clip

Clip operator limits the given input within an interval. The interval is specified with arguments 'min' and 'max'. They default to numeric_limits::min() and numeric_limits::max() respectively. The clipping operation can be done in in-place fashion too, where the input and output blobs are the same.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`min`
</td><td>Minimum value, under which element is replaced by min
</td></tr><tr><td>`max`
</td><td>Maximum value, above which element is replaced by max
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>Input tensor (Tensor<float>) containing elements to beclipped
</td></tr><tr><td>`output`
</td><td>Output tensor (Tensor<float>) containing clippedinput elements
</td></tr></table>
### Code
[caffe2/operators/clip_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/clip_op.cc)
### Devices

- *CPU* `caffe2::ClipOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ClipOp<float, caffe2::CUDAContext>`




---


# ClipGradient
No documentation yet.

### Code
[caffe2/operators/clip_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/clip_op.cc)
### Devices

- *CPU* `caffe2::ClipGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ClipGradientOp<float, caffe2::CUDAContext>`




---


# Col2Im
No documentation yet.

### Code
[caffe2/operators/im2col_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/im2col_op.cc)
### Devices

- *CPU* `caffe2::Col2ImOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::Col2ImOp<float, caffe2::CUDAContext>`




---


# CollectTensor

Collect tensor into tensor vector by reservoir sampling, argument num_to_collect indicates the max number of tensors that will be collcted. The first half of the inputs are tensor vectors, which are also the outputs. The second half of the inputs are the tensors to be collected into each vector (in the same order). The input tensors are collected in all-or-none manner. If they are collected, they will be placed at the same index in the output vectors.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`num_to_collect`
</td><td>The max number of tensors to collect
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::CollectTensorOp<caffe2::CPUContext>`




---


# ComputeOffset

Compute the offsets matrix given cursor and data blobs. Need to be ran at beginning or after reseting cursor  Input(0) is a blob pointing to a TreeCursor, and [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 ComputeOffset is thread safe.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`cursor`
</td><td>A blob containing a pointer to the cursor.
</td></tr><tr><td>`dataset_field_0`
</td><td>First dataset field
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`field_0`
</td><td>Tensor containing offset info for this chunk.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::ComputeOffsetOp`




---


# Concat
Concatenate a list of tensors into a single tensor.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`axis`
</td><td>Which axis to concat on
</td></tr><tr><td>`order`
</td><td>Either NHWC or HCWH, will concat on C axis
</td></tr></table>
### Code
[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)
### Devices

- *CPU* `caffe2::ConcatOp<caffe2::CPUContext>`

- *GPU* `caffe2::ConcatOp<caffe2::CUDAContext>`




---


# ConcatTensorVector

Concat Tensors in the std::unique_ptr<std::vector<Tensor> > along the first dimension.
    
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`vector of Tensor`
</td><td>std::unique_ptr<std::vector<Tensor> >
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`tensor`
</td><td>tensor after concatenating
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::ConcatTensorVectorOp<caffe2::CPUContext>`




---


# ConditionalSetAtomicBool

 
````
    Set an atomic<bool> to true if the given condition bool variable is true

````
     
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`atomic_bool`
</td><td>Blob containing a unique_ptr<atomic<bool>>
</td></tr><tr><td>`condition`
</td><td>Blob containing a bool
</td></tr></table>
### Code
[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::ConditionalSetAtomicBoolOp`




---


# ConstantFill

The operator fills the elements of the output tensor with a constant value specified by the 'value' argument.
 The data type is specified by the 'dtype' argument. The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the TensorProto message. If the 'dtype' argument is not provided, the data type of 'value' is used.
 The output tensor shape is specified by the 'shape' argument. If the number of input is 1, the shape will be identical to that of the input at run time with optional additional dimensions appended at the end as specified by 'extra_shape' argument. In that case the 'shape' argument should not be set.
 NOTE: Currently, it supports data type of float, int32, int64, and bool.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`value`
</td><td>The value for the elements of the output tensor.
</td></tr><tr><td>`dtype`
</td><td>The data type for the elements of the output tensor.Strictly must be one of the types from DataType enum in TensorProto.
</td></tr><tr><td>`shape`
</td><td>The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
</td></tr><tr><td>`extra_shape`
</td><td>The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>Input tensor (optional) to provide shape information.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Output tensor of constant values specified by 'value'argument and its type is specified by the 'dtype' argument
</td></tr></table>
### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::ConstantFillOp<caffe2::CPUContext>`

- *GPU* `caffe2::ConstantFillOp<caffe2::CUDAContext>`




---


# Conv

The convolution operator consumes an input vector, the filter blob and the bias blob and computes the output. Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is convolved with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: conv_op_impl.h is the templated implementation of the conv_op.h file, which is why they are separate files.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints.
</td></tr><tr><td>`filter`
</td><td>The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
</td></tr><tr><td>`bias`
</td><td>The 1D bias blob that is added through the convolution; has size (M).
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.
</td></tr></table>
### Code
[caffe2/operators/conv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_op.cc)
### Devices

- *CPU* `caffe2::ConvOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ConvOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*`EIGEN` on *CPU*`NNPACK` on *CPU*`MKLDNN` on *CPU*

---


# ConvGradient
No documentation yet.

### Code
[caffe2/operators/conv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_op.cc)
### Devices

- *CPU* `caffe2::ConvGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ConvGradientOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*

---


# ConvTranspose

 
````
    The transposed convolution consumes an input vector, the filter blob, and
    the bias blob, and computes the output. Note that other parameters, such as
    the stride and kernel size, or the pads' sizes in each direction are not
    necessary for input because they are provided by the
    ConvTransposeUnpoolOpBase operator. Various dimension checks are done
    implicitly, and the sizes are specified in the Input docs for this operator.
    As is expected, the filter is deconvolved with a subset of the
    image and the bias is added; this is done throughout the image data and the
    output is computed. As a side note on the implementation layout:
    conv_transpose_op_impl.h is the templated implementation of the
    conv_transpose_op.h file, which is why they are separate files.

````
   
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints.
</td></tr><tr><td>`filter`
</td><td>The filter blob that will be used in the transposed convolution; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
</td></tr><tr><td>`bias`
</td><td>The 1D bias blob that is added through the convolution;has size (C)
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output data blob that contains the result of the transposed convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.
</td></tr></table>
### Code
[caffe2/operators/conv_transpose_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_transpose_op.cc)
### Devices

- *CPU* `caffe2::ConvTransposeOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ConvTransposeOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*

---


# ConvTransposeGradient
No documentation yet.

### Code
[caffe2/operators/conv_transpose_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_transpose_op.cc)
### Devices

- *CPU* `caffe2::ConvTransposeGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ConvTransposeGradientOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*

---


# Copy
Copy input tensor into output, potentially across devices.
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>The input tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Tensor that will contain a copy of the input.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::CopyOp<caffe2::CPUContext, caffe2::CPUContext, caffe2::CPUContext>`

- *GPU* `caffe2::CopyOp<caffe2::CUDAContext, caffe2::CUDAContext, caffe2::CUDAContext>`




---


# CopyFromCPUInput

Take a CPU input tensor and copy it to an output in the current Context (GPU or CPU). This may involves cross-device MemCpy.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>The input CPU tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>either a TensorCUDA or a TensorCPU
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::CopyOp<caffe2::CPUContext, caffe2::CPUContext, caffe2::CPUContext>`

- *GPU* `caffe2::CopyOp<caffe2::CUDAContext, caffe2::CUDAContext, caffe2::CPUContext>`




---


# CosineEmbeddingCriterion

CosineEmbeddingCriterion takes two inputs: the similarity value and the label, and computes the elementwise criterion output as  output = 1 - s,  
````
              if y == 1

````
   
````
        max(0, s - margin),  if y == -1

````

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`S`
</td><td>The cosine similarity as a 1-dim TensorCPU.
</td></tr><tr><td>`Y`
</td><td>The label as a 1-dim TensorCPU with int value of 1 or -1.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`loss`
</td><td>The output loss with the same dimensionality as S.
</td></tr></table>
### Code
[caffe2/operators/cosine_embedding_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cosine_embedding_criterion_op.cc)
### Devices

- *CPU* `caffe2::CosineEmbeddingCriterionOp<caffe2::CPUContext>`

- *GPU* `caffe2::CosineEmbeddingCriterionOp<caffe2::CUDAContext>`




---


# CosineEmbeddingCriterionGradient
No documentation yet.

### Code
[caffe2/operators/cosine_embedding_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cosine_embedding_criterion_op.cc)
### Devices

- *CPU* `caffe2::CosineEmbeddingCriterionGradientOp<caffe2::CPUContext>`

- *GPU* `caffe2::CosineEmbeddingCriterionGradientOp<caffe2::CUDAContext>`




---


# CosineSimilarity

 
````
  Given two input float tensors X, Y, and produces one output float tensor
  of the cosine similarity between X and Y.

````
   
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D input tensor
</td></tr></table>
### Code
[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)
### Devices

- *CPU* `caffe2::CosineSimilarityOp<float, caffe2::CPUContext>`




---


# CosineSimilarityGradient
No documentation yet.

### Code
[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)
### Devices

- *CPU* `caffe2::CosineSimilarityGradientOp<float, caffe2::CPUContext>`




---


# CountDown

If the internal count value > 0, decreases count value by 1 and outputs false, otherwise outputs true.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`counter`
</td><td>A blob pointing to an instance of a counter.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`done`
</td><td>false unless the internal count is zero.
</td></tr></table>
### Code
[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)
### Devices

- *CPU* `caffe2::CountDownOp<long, caffe2::CPUContext>`

- *GPU* `caffe2::CountDownOp<long, caffe2::CUDAContext>`




---


# CountUp

Increases count value by 1 and outputs the previous value atomically 
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`counter`
</td><td>A blob pointing to an instance of a counter.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`previous_count`
</td><td>count value BEFORE this operation
</td></tr></table>
### Code
[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)
### Devices

- *CPU* `caffe2::CountUpOp<long, caffe2::CPUContext>`

- *GPU* `caffe2::CountUpOp<long, caffe2::CUDAContext>`




---


# CreateAtomicBool
Create an unique_ptr blob to hold a atomic<bool>
### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`atomic_bool`
</td><td>Blob containing a unique_ptr<atomic<bool>>
</td></tr></table>
### Code
[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::CreateAtomicBoolOp`




---


# CreateCommonWorld

Creates a common world for communication operators.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`size`
</td><td>(int) size of the common world.
</td></tr><tr><td>`rank`
</td><td>(int) rank of this node in the common world.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`kv_handler`
</td><td>Key/value handler for rendezvous (optional).
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`comm_world`
</td><td>A common world for distributed messaging.
</td></tr></table>
### Code
[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)
### Devices

- *CPU* `caffe2::NoDefaultEngineOp<caffe2::CPUContext>`

- *GPU* `caffe2::NoDefaultEngineOp<caffe2::CUDAContext>`




---


# CreateCounter

Creates a count-down counter with initial value specified by the 'init_count' argument.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`init_count`
</td><td>Initial count for the counter, must be >= 0.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`counter`
</td><td>A blob pointing to an instance of a new counter.
</td></tr></table>
### Code
[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)
### Devices

- *CPU* `caffe2::CreateCounterOp<long, caffe2::CPUContext>`

- *GPU* `caffe2::CreateCounterOp<long, caffe2::CUDAContext>`




---


# CreateMutex
Creates an unlocked mutex and returns it in a unique_ptr blob.
### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`mutex_ptr`
</td><td>Blob containing a std::unique_ptr<mutex>.
</td></tr></table>
### Code
[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::CreateMutexOp`




---


# CreateQPSMetric

CreateQPSMetric operator create a blob that will store state that is required for computing QPSMetric. The only output of the operator will have blob with QPSMetricState as an output.

### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Blob with QPSMetricState
</td></tr></table>
### Code
[caffe2/operators/metrics_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/metrics_ops.cc)
### Devices

- *CPU* `caffe2::CreateQPSMetricOp`




---


# CreateTensorVector
Create a std::unique_ptr<std::vector<Tensor> >
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::CreateTensorVectorOp<caffe2::CPUContext>`




---


# CreateTextFileReader
Create a text file reader. Fields are delimited by <TAB>.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`filename`
</td><td>Path to the file.
</td></tr><tr><td>`num_pases`
</td><td>Number of passes over the file.
</td></tr><tr><td>`field_types`
</td><td>List with type of each field. Type enum is found at core.DataType.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>Pointer to the created TextFileReaderInstance.
</td></tr></table>
### Code
[caffe2/operators/text_file_reader.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/text_file_reader.cc)
### Devices

- *CPU* `caffe2::CreateTextFileReaderOp`




---


# CreateTreeCursor

Creates a cursor to iterate through a list of tensors, where some of those tensors contains the lengths in a nested schema. The schema is determined by the  `fields`  arguments.
 For example, to represent the following schema:   
````
  Struct(
      a=Int(),
      b=List(List(Int),
      c=List(
          Struct(

````
   
````
            c1=String,

````
   
````
            c2=List(Int),
          ),
      ),
  )


````
 the field list will be:  
````
  [
      "a",
      "b:lengths",
      "b:values:lengths",
      "b:values:values",
      "c:lengths",
      "c:c1",
      "c:c2:lengths",
      "c:c2:values",
  ]


````
 And for the following instance of the struct:   
````
  Struct(
      a=3,
      b=[[4, 5], [6, 7, 8], [], [9]],
      c=[
          Struct(c1='alex', c2=[10, 11]),
          Struct(c1='bob', c2=[12]),
      ],
  )


````
 The values of the fields will be:  
````
  {
      "a": [3],
      "b:lengths": [4],
      "b:values:lengths": [2, 3, 0, 1],
      "b:values:values": [4, 5, 6, 7, 8, 9],
      "c:lengths": [2],
      "c:c1": ["alex", "bob"],
      "c:c2:lengths": [2, 1],
      "c:c2:values", [10, 11, 12],
  }


````
 In general, every field name in the format "{prefix}:lengths" defines a domain "{prefix}", and every subsequent field in the format "{prefx}:{field}" will be in that domain, and the length of the domain is provided for each entry of the parent domain. In the example, "b:lengths" defines a domain of length 4, so every field under domain "b" will have 4 entries.
The "lengths" field for a given domain must appear before any reference to that domain.
 Returns a pointer to an instance of the Cursor, which keeps the current offset on each of the domains defined by  `fields` . Cursor also ensures thread-safety such that ReadNextBatch and ResetCursor can be used safely in parallel.
 A cursor does not contain data per se, so calls to ReadNextBatch actually need to pass a list of blobs containing the data to read for each one of the fields.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`fields`
</td><td>A list of strings each one representing a field of the dataset.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`cursor`
</td><td>A blob pointing to an instance of a new TreeCursor.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::CreateTreeCursorOp`




---


# CrossEntropy

Operator computes the cross entropy between the input and the label set. In  practice, it is most commonly used at the end of models, after the SoftMax  operator and before the AveragedLoss operator. Note that CrossEntropy  assumes that the soft labels provided is a 2D array of size N x D  (batch size x number of classes). Each entry in the 2D label corresponds to  the soft label for the input, where each element represents the correct  probability of the class being selected. As such, each element must be between  0 and 1, and all elements in an entry must sum to 1. The formula used is:   
````
                Y[i] = sum_j (label[i][j] * log(X[i][j]))


````
  where (i, j) is the classifier's prediction of the jth class (the correct one),  and i is the batch size. Each log has a lower limit for numerical stability.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input blob from the previous layer, which is almost always the result of a softmax operation; X is a 2D array of size N x D, where N is the batch size and D is the number of classes
</td></tr><tr><td>`label`
</td><td>Blob containing the labels used to compare the input
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output blob after the cross entropy computation
</td></tr></table>
### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::CrossEntropyOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::GPUFallbackOp<caffe2::CrossEntropyOp<float, caffe2::CPUContext>, caffe2::SkipIndices<> >`




---


# CrossEntropyGradient
No documentation yet.

### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::CrossEntropyGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::GPUFallbackOp<caffe2::CrossEntropyGradientOp<float, caffe2::CPUContext>, caffe2::SkipIndices<> >`




---


# DepthConcat
Backward compatible operator name for Concat.
### Code
[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)
### Devices

- *CPU* `caffe2::ConcatOp<caffe2::CPUContext>`

- *GPU* `caffe2::ConcatOp<caffe2::CUDAContext>`




---


# DepthSplit
Backward compatible operator name for Split.
### Code
[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)
### Devices

- *CPU* `caffe2::SplitOp<caffe2::CPUContext>`

- *GPU* `caffe2::SplitOp<caffe2::CUDAContext>`




---


# Div

Performs element-wise binary division (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and type as A
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::EigenDivFunctor, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaDivFunctor, caffe2::SameTypeAsInput>`




---


# DivGradient
No documentation yet.

### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::DivGradientOp<float, caffe2::CPUContext>`




---


# DotProduct

 
````
  Given two input float tensors X, Y, and produces one output float tensor
  of the dot product between X and Y.

````
   
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D input tensor
</td></tr></table>
### Code
[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)
### Devices

- *CPU* `caffe2::DotProductOp<float, caffe2::CPUContext>`




---


# DotProductGradient
No documentation yet.

### Code
[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)
### Devices

- *CPU* `caffe2::DotProductGradientOp<float, caffe2::CPUContext>`




---


# Dropout

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs, output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in test mode or not, the output Y will either be a random dropout, or a simple copy of the input. Note that our implementation of Dropout does scaling in the training phase, so during testing nothing needs to be done.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`ratio`
</td><td>(float, default 0.5) the ratio of random dropout
</td></tr><tr><td>`is_test`
</td><td>(int, default 0) if nonzero, run dropout in test mode where the output is simply Y = X.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>The input data as Tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The output.
</td></tr><tr><td>`mask`
</td><td>The output mask. If is_test is nonzero, this output is not filled.
</td></tr></table>
### Code
[caffe2/operators/dropout_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dropout_op.cc)
### Devices

- *CPU* `caffe2::DropoutOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::DropoutOp<float, caffe2::CUDAContext>`




---


# DropoutGrad
No documentation yet.

### Code
[caffe2/operators/dropout_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dropout_op.cc)
### Devices

- *CPU* `caffe2::DropoutGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::DropoutGradientOp<float, caffe2::CUDAContext>`




---


# EQ

Performs element-wise comparison  `==`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long>, caffe2::CPUContext, caffe2::NaiveEQFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long>, caffe2::CUDAContext, caffe2::CudaEQFunctor, caffe2::FixedType<bool> >`




---


# Elu

 Elu takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the function  `f(x) = alpha * (exp(x) - 1.) for x < 0` ,  `f(x) = x for x >= 0` ., is applied to the tensor elementwise.
 
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D input tensor
</td></tr></table>
### Code
[caffe2/operators/elu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elu_op.cc)
### Devices

- *CPU* `caffe2::EluOp<float, caffe2::CPUContext>`




---


# EluGradient

EluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.

### Code
[caffe2/operators/elu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elu_op.cc)
### Devices

- *CPU* `caffe2::EluGradientOp<float, caffe2::CPUContext>`




---


# EnsureCPUOutput

Take an input tensor in the current Context (GPU or CPU) and create an output which is always a TensorCPU. This may involves cross-device MemCpy.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>The input CUDA or CPU tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>TensorCPU that is a copy of the input.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::CopyOp<caffe2::CPUContext, caffe2::CPUContext, caffe2::CPUContext>`

- *GPU* `caffe2::CopyOp<caffe2::CUDAContext, caffe2::CPUContext, caffe2::CUDAContext>`




---


# Exp

Calculates the exponential of the given input tensor, element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>Input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The exponential of the input tensor computed element-wise
</td></tr></table>
### Code
[caffe2/operators/exp_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/exp_op.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CPUContext, caffe2::WithDefaultConstructor<caffe2::ExpCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CUDAContext, caffe2::WithDefaultConstructor<caffe2::ExpCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# ExpandDims

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument  `dims` , a list of dimensions that will be inserted.
Dimension indices in  `dims`  are as seen in the output tensor. For example:   
````
  Given a tensor such that tensor.Shape() = [3, 4, 5], then
  ExpandDims(tensor, dims=[0, 4]).Shape() == [1, 3, 4, 5, 1])


````
 If the same blob is provided in input and output, the operation is copy-free.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>Original tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`expanded`
</td><td>Reshaped tensor with same data as input.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::ExpandDimsOp<caffe2::CPUContext>`

- *GPU* `caffe2::ExpandDimsOp<caffe2::CUDAContext>`




---


# ExtendTensor

Extend input 0 if necessary based on max element in input 1.
Input 0 must be the same as output, that is, it is required to be in-place.
Input 0 may have to be re-allocated in order for accommodate to the new size.
Currently, an exponential growth ratio is used in order to ensure amortized constant time complexity.
All except the outer-most dimension must be the same between input 0 and 1.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tensor`
</td><td>The tensor to be extended.
</td></tr><tr><td>`new_indices`
</td><td>The size of tensor will be extended based on max element in new_indices.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`extended_tensor`
</td><td>Same as input 0, representing the mutated tensor.
</td></tr></table>
### Code
[caffe2/operators/extend_tensor_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/extend_tensor_op.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::ExtendTensorOp<caffe2::CPUContext>`




---


# FC

Computes the result of passing an input vector X into a fully connected layer with 2D weight matrix W and 1D bias vector b.
 The layer computes Y = X * W + b, where X has size (M x K), W has size (K x N), b has size (N), and Y has size (M x N), where M is the batch size. Even though b is 1D, it is resized to size (M x N) implicitly and added to each vector in the batch. These dimensions must be matched correctly, or else the operator will throw errors.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`axis`
</td><td>(int32_t) default to 1; describes the axis of the inputs; defaults to one because the 0th axis most likely describes the batch_size
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>2D input of size (MxK) data
</td></tr><tr><td>`W`
</td><td>2D blob of size (KxN) containing fully connected weight matrix
</td></tr><tr><td>`b`
</td><td>1D blob containing bias vector
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D output tensor
</td></tr></table>
### Code
[caffe2/operators/fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)
### Devices

- *CPU* `caffe2::FullyConnectedOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`

- *GPU* `caffe2::FullyConnectedOp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`



### Engines
`NERVANA` on *CUDA*`PACKED` on *CPU*

---


# FCGradient
No documentation yet.

### Code
[caffe2/operators/fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)
### Devices

- *CPU* `caffe2::FullyConnectedGradientOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`

- *GPU* `caffe2::FullyConnectedGradientOp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`



### Engines
`NERVANA` on *CUDA*

---


# FeedBlob

FeedBlobs the content of the blobs. The input and output blobs should be one-to-one inplace.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`value`
</td><td>(string) if provided then we will use this string as the value for theprovided output tensor
</td></tr></table>
### Code
[caffe2/operators/feed_blob_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/feed_blob_op.cc)
### Devices

- *CPU* `caffe2::FeedBlobOp<caffe2::CPUContext>`




---


# FindDuplicateElements

Shrink the data tensor by removing data blocks with given zero-based indices in the outermost dimension of the tensor. Indices are not assumed in any order or unique but with the range [0, blocks_size). Indices could be empty.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>a 1-D tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`indices`
</td><td>indices of duplicate elements in data, excluding first occurrences.
</td></tr></table>
### Code
[caffe2/operators/find_duplicate_elements_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.cc)
### Devices

- *CPU* `caffe2::FindDuplicateElementsOp<caffe2::CPUContext>`




---


# Flatten

Flattens the input tensor into a 2D matrix, keeping the first dimension unchanged.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>A tensor of rank >= 2.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>A tensor of rank 2 with the contents of the input tensor, with first dimension equal first dimension of input, and remaining input dimensions flatenned into the inner dimension of the output.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::FlattenOp<caffe2::CPUContext>`

- *GPU* `caffe2::FlattenOp<caffe2::CUDAContext>`




---


# FlattenToVec

Flattens the input tensor into a 1D vector.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>A tensor of rank >= 1.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>A tensor of rank 1 with the contents of the input tensor
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::FlattenToVecOp<caffe2::CPUContext>`

- *GPU* `caffe2::FlattenToVecOp<caffe2::CUDAContext>`




---


# FloatToHalf
No documentation yet.

### Code
[caffe2/operators/half_float_ops.cu](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/half_float_ops.cu)
### Devices

- *GPU* `caffe2::FloatToHalfCUDA`




---


# Free

Frees the content of the blobs. The input and output blobs should be one-to-one inplace.
### Code
[caffe2/operators/free_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/free_op.cc)
### Devices

- *CPU* `caffe2::FreeOp<caffe2::CPUContext>`

- *GPU* `caffe2::FreeOp<caffe2::CUDAContext>`




---


# GE

Performs element-wise comparison  `>=`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::NaiveGEFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaGEFunctor, caffe2::FixedType<bool> >`




---


# GT

Performs element-wise comparison  `>`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::NaiveGTFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaGTFunctor, caffe2::FixedType<bool> >`




---


# Gather

Given DATA tensor of rank r >= 1, and INDICES tensor of rank q, gather entries of the outer-most dimension of DATA indexed by INDICES, and concatenate them in an output tensor of rank q + (r - 1).
 Example:  
````
  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  INDICES = [
      [0, 1],
      [1, 2],
  ]
  OUTPUT = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]

````

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Tensor of rank r >= 1.
</td></tr><tr><td>`INDICES`
</td><td>Tensor of int32/int64 indices, of any rank q.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Tensor of rank q + (r - 1).
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::GatherOp<caffe2::CPUContext>`




---


# GatherPadding

Gather the sum of start and end paddings in a padded input sequence. Used in order to compute the gradients of AddPadding w.r.t the padding tensors.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`pad_width`
</td><td>Outer-size of padding present around each range.
</td></tr><tr><td>`end_pad_width`
</td><td>(Optional) Specifies a different end-padding width.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data_in`
</td><td>T<N, D1..., Dn> Padded input data
</td></tr><tr><td>`lengths`
</td><td>(i64) Num of elements in each range. sum(lengths) = N. If not provided, considers all data as a single segment.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`padding_sum`
</td><td>Sum of all start paddings, or of all paddings if end_padding_sum is not provided.
</td></tr><tr><td>`end_padding_sum`
</td><td>T<D1..., Dn> Sum of all end paddings, if provided.
</td></tr></table>
### Code
[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::GatherPaddingOp`




---


# GatherRanges

Given DATA tensor of rank 1, and RANGES tensor of rank 3, gather corresponding ranges into a 1-D tensor OUTPUT.
 RANGES dimentions description: 1: represents list of examples within a batch 2: represents list features 3: two values which are start and length or a range (to be applied on DATA)  Another output LENGTHS represents each example length within OUTPUT  Example:  
````
  DATA  = [1, 2, 3, 4, 5, 6]
  RANGES = [
    [
      [0, 1],
      [2, 2],
    ],
    [
      [4, 1],
      [5, 1],
    ]
  ]
  OUTPUT = [1, 3, 4, 5, 6]
  LENGTHS = [3, 2]

````

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Tensor of rank 1.
</td></tr><tr><td>`RANGES`
</td><td>Tensor of int32/int64 ranges, of dims (N, M, 2). Where N is number of examples and M is a size of each example. Last dimention represents a range in the format (start, lengths)
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>1-D tensor of size sum of range lengths
</td></tr><tr><td>`LENGTHS`
</td><td>1-D tensor of size N with lengths over gathered data for each row in a batch. sum(LENGTHS) == OUTPUT.size()
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::GatherRangesOp<caffe2::CPUContext>`




---


# GaussianFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::GaussianFillOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::GaussianFillOp<float, caffe2::CUDAContext>`




---


# GetAllBlobNames

Return a 1D tensor of strings containing the names of each blob in the active workspace.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`include_shared`
</td><td>(bool, default true) Whether to include blobs inherited from parent workspaces.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`blob_names`
</td><td>1D tensor of strings containing blob names.
</td></tr></table>
### Code
[caffe2/operators/workspace_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/workspace_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::GetAllBlobNamesOp`




---


# GivenTensorFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::GivenTensorFillOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::GivenTensorFillOp<float, caffe2::CUDAContext>`




---


# GivenTensorInt64Fill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::GivenTensorFillOp<long, caffe2::CPUContext>`




---


# GivenTensorIntFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::GivenTensorFillOp<int, caffe2::CPUContext>`




---


# GivenTensorStringFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::GivenTensorFillOp<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> >, caffe2::CPUContext>`




---


# HSoftmax

Hierarchical softmax is an operator which approximates the softmax operator while giving significant training speed gains and reasonably comparable performance. In this operator, instead of calculating the probabilities of all the classes, we calculate the probability of each step in the path from root to the target word in the hierarchy.
 The operator takes a 2-D tensor (Tensor<float>) containing a batch of layers, a set of parameters represented by the weight matrix and bias terms, and a 1-D tensor (Tensor<int>) holding labels, or the indices of the target class. The hierarchy has to be specified as an argument to the operator.
 The operator returns a 1-D tensor holding the computed log probability of the target class and a 2-D tensor of intermediate outputs (from the weight matrix and softmax from each step in the path from root to target class) which will be used by the gradient operator to compute gradients for all samples in the batch.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`hierarchy`
</td><td>Serialized HierarchyProto string containing list of vocabulary words and their paths from root of hierarchy to the leaf
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data from previous layer
</td></tr><tr><td>`W`
</td><td>2D blob containing 'stacked' fully connected weight matrices. Each node in the hierarchy contributes one FC weight matrix if it has children nodes. Dimension is N*D, D is input dimension of data (X), N is sum of all output dimensions, or total number of nodes (excl root)
</td></tr><tr><td>`b`
</td><td>1D blob with N parameters
</td></tr><tr><td>`labels`
</td><td>int word_id of the target word
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1-D of log probability outputs, one per sample
</td></tr><tr><td>`intermediate_output`
</td><td>Extra blob to store the intermediate FC and softmax outputs for each node in the hierarchical path of a word. The outputs from samples are stored in consecutive blocks in the forward pass and are used in reverse order in the backward gradientOp pass
</td></tr></table>
### Code
[caffe2/operators/h_softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/h_softmax_op.cc)
### Devices

- *CPU* `caffe2::HSoftmaxOp<float, caffe2::CPUContext>`




---


# HSoftmaxGradient
No documentation yet.

### Code
[caffe2/operators/h_softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/h_softmax_op.cc)
### Devices

- *CPU* `caffe2::HSoftmaxGradientOp<float, caffe2::CPUContext>`




---


# HSoftmaxSearch

 
````
  HSoftmaxSearch is an operator to generate the most possible paths given a
  well-trained model and input vector. Greedy algorithm is used for pruning the
  search tree.

````
   
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`tree`
</td><td>Serialized TreeProto string containing a tree including all intermidate nodes and leafs. All nodes must have names for correct outputs
</td></tr><tr><td>`beam`
</td><td>beam used for pruning tree. The pruning algorithm is that only children, whose score is smaller than parent's score puls beam, will be propagated. 
</td></tr><tr><td>`topN`
</td><td>Number of nodes in outputs
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data from previous layer
</td></tr><tr><td>`W`
</td><td>The matrix trained from Softmax Ops
</td></tr><tr><td>`b`
</td><td>The bias traiend from Softmax Ops
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y_names`
</td><td>The name of selected nodes and leafs. For nodes, it will be the name defined in the tree. For leafs, it will be the index of the word in the tree.
</td></tr><tr><td>`Y_scores`
</td><td>The corresponding scores of Y_names
</td></tr></table>
### Code
[caffe2/operators/h_softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/h_softmax_op.cc)
### Devices

- *CPU* `caffe2::HSoftmaxSearchOp<float, caffe2::CPUContext>`




---


# HalfToFloat
No documentation yet.

### Code
[caffe2/operators/half_float_ops.cu](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/half_float_ops.cu)
### Devices

- *GPU* `caffe2::HalfToFloatCUDA`




---


# HasElements
Returns true iff the input tensor has size > 0
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tensor`
</td><td>Tensor of any type.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`has_elements`
</td><td>Scalar bool tensor. True if input is not empty.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::HasElementsOp<caffe2::CPUContext>`




---


# Im2Col
The Im2Col operator from Matlab.
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>4-tensor in NCHW or NHWC.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>4-tensor. For NCHW: N x (C x kH x kW) x outH x outW.For NHWC: N x outH x outW x (kH x kW x C
</td></tr></table>
### Code
[caffe2/operators/im2col_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/im2col_op.cc)
### Devices

- *CPU* `caffe2::Im2ColOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::Im2ColOp<float, caffe2::CUDAContext>`




---


# IndexFreeze

Freezes the given index, disallowing creation of new index entries.
Should not be called concurrently with IndexGet.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handle`
</td><td>Pointer to an Index instance.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexFreezeOp`




---


# IndexGet

Given an index handle and a tensor of keys, return an Int tensor of same shape containing the indices for each of the keys. If the index is frozen, unknown entries are given index 0. Otherwise, new entries are added into the index.
If an insert is necessary but max_elements has been reached, fail.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handle`
</td><td>Pointer to an Index instance.
</td></tr><tr><td>`keys`
</td><td>Tensor of keys to be looked up.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`indices`
</td><td>Indices for each of the keys.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexGetOp`




---


# IndexLoad

Loads the index from the given 1-D tensor. Elements in the tensor will be given consecutive indexes starting at 1. Fails if tensor contains repeated elements.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`skip_first_entry`
</td><td>If set, skips the first entry of the tensor. This allows to load tensors that are aligned with an embedding, where the first entry corresponds to the default 0 index entry.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handle`
</td><td>Pointer to an Index instance.
</td></tr><tr><td>`items`
</td><td>1-D tensor with elements starting with index 1.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexLoadOp`




---


# IndexSize

Returns the number of entries currently present in the index.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handle`
</td><td>Pointer to an Index instance.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`items`
</td><td>Scalar int64 tensor with number of entries.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexSizeOp`




---


# IndexStore

Stores the keys of this index in a 1-D tensor. Since element 0 is reserved for unknowns, the first element of the output tensor will be element of index 1.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handle`
</td><td>Pointer to an Index instance.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`items`
</td><td>1-D tensor with elements starting with index 1.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexStoreOp`




---


# InstanceNorm

Carries out instance normalization as described in the paper  [https://arxiv.org/abs/1607.08022.](https://arxiv.org/abs/1607.08022.)  Depending on the mode it is being run, there are multiple cases for the number of outputs, which we list below:  Output case #1: Y (train mode type 1, and test mode) Output case #2: Y, saved_mean, saved_inv_var  
````
                (train mode type 2)


````
 For training mode, type 2 is faster in the sense that for the backward pass, it is able to reuse the saved mean and inv_var in the gradient computation.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`is_test`
</td><td>If set to nonzero, run spatial batch normalization in test mode.
</td></tr><tr><td>`epsilon`
</td><td>The epsilon value to use to avoid division by zero.
</td></tr><tr><td>`order`
</td><td>A StorageOrder string.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>The input 4-dimensional tensor of shape NCHW or NHWC depending on the order parameter.
</td></tr><tr><td>`scale`
</td><td>The input 1-dimensional scale tensor of size C.
</td></tr><tr><td>`bias`
</td><td>The input 1-dimensional bias tensor of size C.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>The output 4-dimensional tensor of the same shape as X.
</td></tr><tr><td>`saved_mean`
</td><td>Optional saved mean used during training to speed up gradient computation. Should not be used for testing.
</td></tr><tr><td>`saved_inv_var`
</td><td>Optional saved inverse variance used during training to speed up gradient computation. Should not be used for testing.
</td></tr></table>
### Code
[caffe2/operators/instance_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc)
### Devices

- *CPU* `caffe2::InstanceNormOp<caffe2::CPUContext>`




---


# IntIndexCreate

Creates a dictionary that maps int32 keys to consecutive integers from 1 to max_elements. Zero is reserved for unknown keys.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`max_elements`
</td><td>Max number of elements, including the zero entry.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>Pointer to an Index instance.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexCreateOp<int>`




---


# IsEmpty
Returns true iff the input tensor has size == 0
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tensor`
</td><td>Tensor of any type.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`is_empty`
</td><td>Scalar bool tensor. True if input is empty.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::IsEmptyOp<caffe2::CPUContext>`




---


# LE

Performs element-wise comparison  `<=`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::NaiveLEFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaLEFunctor, caffe2::FixedType<bool> >`




---


# LRN
No documentation yet.

### Code
[caffe2/operators/local_response_normalization_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/local_response_normalization_op.cc)
### Devices

- *CPU* `caffe2::LRNOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::LRNOp<float, caffe2::CUDAContext>`




---


# LRNGradient
No documentation yet.

### Code
[caffe2/operators/local_response_normalization_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/local_response_normalization_op.cc)
### Devices

- *CPU* `caffe2::LRNGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::LRNGradientOp<float, caffe2::CUDAContext>`




---


# LSTMUnit

 LSTMUnit computes the activations of a standard LSTM (without peephole connections), in a sequence-length aware fashion.
 Concretely, given the (fused) inputs X (TxNxD), the previous cell state (NxD), and the sequence lengths (N), computes the LSTM activations, avoiding computation if the input is invalid (as in, the value at X{t][n] >= seqLengths[n].
 
### Code
[caffe2/operators/lstm_unit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lstm_unit_op.cc)
### Devices

- *CPU* `caffe2::LSTMUnitOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::LSTMUnitOp<float, caffe2::CUDAContext>`




---


# LSTMUnitGradient
No documentation yet.

### Code
[caffe2/operators/lstm_unit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lstm_unit_op.cc)
### Devices

- *CPU* `caffe2::LSTMUnitGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::LSTMUnitGradientOp<float, caffe2::CUDAContext>`




---


# LT

Performs element-wise comparison  `<`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::NaiveLTFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaLTFunctor, caffe2::FixedType<bool> >`




---


# LabelCrossEntropy

Operator computes the cross entropy between the input and the label set. In  practice, it is most commonly used at the end of models, after the SoftMax  operator and before the AveragedLoss operator. Note that LabelCrossEntropy  assumes that the label provided is either a 1D array of size N (batch size), or  a 2D array of size N x 1 (batch size). Each entry in the label vector indicates  which is the correct class; as such, each entry must be between 0 and D - 1,  inclusive, where D is the total number of classes. The formula used is:   
````
                            Y[i] = -log(X[i][j])


````
  where (i, j) is the classifier's prediction of the jth class (the correct one),  and i is the batch size. Each log has a lower limit for numerical stability.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input blob from the previous layer, which is almost always the result of a softmax operation; X is a 2D array of size N x D, where N is the batch size and D is the number of classes
</td></tr><tr><td>`label`
</td><td>Blob containing the labels used to compare the input
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output blob after the cross entropy computation
</td></tr></table>
### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::LabelCrossEntropyOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::LabelCrossEntropyOp<float, caffe2::CUDAContext>`




---


# LabelCrossEntropyGradient
No documentation yet.

### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::LabelCrossEntropyGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::LabelCrossEntropyGradientOp<float, caffe2::CUDAContext>`




---


# LengthsPartition

LengthsPartition splits the input int tensor into multiple ones according to the second tensor. The first dimension is expected to be the tensor that describes lengths of the elements.
 Takes the second input and partitions it to shards according to the remainder of values modulo the number of partitions. It requires the second tensor to be a 1D-tensor of the integral type. The first tensor should be 1D-tensor of int32 that would represent the lengths of the elements in the input. The number of partitions is derived as (num_output / num_input).
 If additional inputs are present they must have the same shape as the first input, optionally with extra trailing dimensions. They will be partitioned accordingly to the first input.
 Optional arg 'pack_first_input' transforms the first tensor values as X_ij / num_partitions.
 Outputs are ordered as X_0_part_0, X_1_part_0, ..., X_N-1_part_0, X_0_part_1, ..., X_N-1_part_K-1 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`pack_first_input`
</td><td>(int, default 0) If set, the operator transforms the first tensor values as floor(X_ij / num_partitions)
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>Input tensor containing data to be partitioned. The number of input tensors might be greater than 1 but must have the same shape as the previous tensors.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`partitions`
</td><td>Output Partitions. The number of output tensors has to be a multiple of the number of input tensors.
</td></tr></table>
### Code
[caffe2/operators/partition_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/partition_ops.cc)
### Devices

- *CPU* `caffe2::LengthsPartitionOp`




---


# LengthsRangeFill

Convert a length vector to a range sequene. For example, input=[4,3,1], the output would be [0,1,2,3,0,1,2,0].

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>1D tensor of int32 or int64 segment lengths.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`range_sequence`
</td><td>1D tensor whose size is the sum of `lengths`
</td></tr></table>
### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::LengthsRangeFillOp<caffe2::CPUContext>`




---


# LengthsSum

Applies 'Sum' to each segment of the input tensor. Segments are defined by their LENGTHS.
 LENGTHS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 For example LENGTHS = [2, 1] stands for segments DATA[0..1] and DATA[2]  The first dimension of the output is equal to the number of input segments, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor, slices of which are aggregated.
</td></tr><tr><td>`LENGTHS`
</td><td>Vector with the same sum of elements as the first dimension of DATA
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of len(LENGTHS) 
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsOp<float, int, caffe2::CPUContext, caffe2::SumReducer<float, caffe2::CPUContext>, false>`




---


# LengthsSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsGradientOp<float, int, caffe2::CPUContext, caffe2::SumReducerGradient<float, caffe2::CPUContext> >`




---


# LengthsToRanges

Given a vector of segment lengths, calculates offsets of each segment and packs them next to the lengths. For the input vector of length N the output is a Nx2 matrix with (offset, lengths) packaged for each segment. Output is going to have the same type as input. For long tensors explicit casting from int32 to int64 might be necessary prior to this op.
 For example,  `[1, 3, 0, 2]`  transforms into  `[[0, 1], [1, 3], [4, 0], [4, 2]]` .

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>1D tensor of int32 or int64 segment lengths.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`ranges`
</td><td>2D tensor of shape len(lengths) X 2 and the same type as `lengths`
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::LengthsToRangesOp<caffe2::CPUContext>`




---


# LengthsToSegmentIds

Given a vector of segment lengths, returns a zero-based, consecutive vector of segment_ids. For example, [1, 3, 0, 2] will produce [0, 1, 1, 1, 3, 3].
In general, the inverse operation is SegmentIdsToLengths. Notice though that trailing empty sequence lengths can't be properly recovered from segment ids.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>1D tensor of int32 or int64 segment lengths.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`segment_ids`
</td><td>1D tensor of length `sum(lengths)`
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::LengthsToSegmentIdsOp<caffe2::CPUContext>`




---


# LengthsToShape
No documentation yet.

### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::LengthsToShapeOp<caffe2::CPUContext>`




---


# LengthsWeightedSum

Applies 'WeightedSum' to each segment of the input tensor. Segments are defined by their LENGTHS.
 LENGTHS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 For example LENGTHS = [2, 1] stands for segments DATA[0..1] and DATA[2]  The first dimension of the output is equal to the number of input segments, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor for the summation
</td></tr><tr><td>`SCALARS`
</td><td>Scalar multipliers for the input slices. Must be a vector with the length matching the first dimension of DATA
</td></tr><tr><td>`LENGTHS`
</td><td>Vector with the same sum of elements as the first dimension of DATA
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of len(LENGTHS) 
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducer<float, caffe2::CPUContext>, false>`




---


# LengthsWeightedSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsGradientOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducerGradient<float, caffe2::CPUContext> >`




---


# Load

The Load operator loads a set of serialized blobs from a db. It takes no input and [0, infinity) number of outputs, using the db keys to match the db entries with the outputs.
 If an input is passed, then it is assumed that that input blob is a DBReader to load from, and we ignore the db and db_type arguments.
 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`absolute_path`
</td><td>(int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace.
</td></tr><tr><td>`db`
</td><td>(string) the path to the db to load.
</td></tr><tr><td>`db_type`
</td><td>(string) the type of the db.
</td></tr><tr><td>`keep_device`
</td><td>(int, default 0) if nonzero, the blobs are loaded into the device that is specified in the serialized BlobProto. Otherwise, the device will be set as the one that the Load operator is being run under.
</td></tr><tr><td>`load_all`
</td><td>(int, default 0) if nonzero, will load all blobs pointed to by the db to the workspace overwriting/creating blobs as needed.
</td></tr></table>
### Code
[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)
### Devices

- *CPU* `caffe2::LoadOp<caffe2::CPUContext>`

- *GPU* `caffe2::LoadOp<caffe2::CUDAContext>`




---


# LongIndexCreate

Creates a dictionary that maps int64 keys to consecutive integers from 1 to max_elements. Zero is reserved for unknown keys.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`max_elements`
</td><td>Max number of elements, including the zero entry.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>Pointer to an Index instance.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexCreateOp<long>`




---


# LpPool

 LpPool consumes an input blob X and applies L-p pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. L-p pooling consisting of taking the L-p norm of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.
   
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. 
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output data tensor from L-p pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
</td></tr></table>
### Code
[caffe2/operators/lp_pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lp_pool_op.cc)
### Devices

- *CPU* `caffe2::PoolOp<float, caffe2::CPUContext, caffe2::LpPool>`

- *GPU* `caffe2::PoolOp<float, caffe2::CUDAContext, caffe2::(anonymous namespace)::LpPool>`




---


# LpPoolGradient
No documentation yet.

### Code
[caffe2/operators/lp_pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lp_pool_op.cc)
### Devices

- *CPU* `caffe2::PoolGradientOp<float, caffe2::CPUContext, caffe2::LpPool>`

- *GPU* `caffe2::PoolGradientOp<float, caffe2::CUDAContext, caffe2::(anonymous namespace)::LpPool>`




---


# MSRAFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::MSRAFillOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::MSRAFillOp<float, caffe2::CUDAContext>`




---


# MakeTwoClass

Given a vector of probabilities, this operator transforms this into a 2-column  matrix with complimentary probabilities for binary classification. In explicit  terms, given the vector X, the output Y is vstack(1 - X, X).
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input vector of probabilities
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>2-column matrix with complimentary probabilities of X for binary classification
</td></tr></table>
### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::MakeTwoClassOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::MakeTwoClassOp<float, caffe2::CUDAContext>`




---


# MakeTwoClassGradient
No documentation yet.

### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::MakeTwoClassGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::MakeTwoClassGradientOp<float, caffe2::CUDAContext>`




---


# MarginRankingCriterion

MarginRankingCriterion takes two input data X1 (Tensor<float>), X2 (Tensor<float>), and label Y (Tensor<int>) to produce the loss (Tensor<float>) where the loss function, loss(X1, X2, Y) = max(0, -Y * (X1 - X2) + margin), is applied to the tensor elementwise.
 If y == 1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for y == -1.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X1`
</td><td>The left input vector as a 1-dim TensorCPU.
</td></tr><tr><td>`X2`
</td><td>The right input vector as a 1-dim TensorCPU.
</td></tr><tr><td>`Y`
</td><td>The label as a 1-dim TensorCPU with int value of 1 or -1.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`loss`
</td><td>The output loss with the same dimensionality as X1.
</td></tr></table>
### Code
[caffe2/operators/margin_ranking_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/margin_ranking_criterion_op.cc)
### Devices

- *CPU* `caffe2::MarginRankingCriterionOp<caffe2::CPUContext>`

- *GPU* `caffe2::MarginRankingCriterionOp<caffe2::CUDAContext>`




---


# MarginRankingCriterionGradient

MarginRankingCriterionGradient takes both X1, X2, Y and dY and uses them to update dX1, and dX2 according to the chain rule and derivatives of the loss function.

### Code
[caffe2/operators/margin_ranking_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/margin_ranking_criterion_op.cc)
### Devices

- *CPU* `caffe2::MarginRankingCriterionGradientOp<caffe2::CPUContext>`

- *GPU* `caffe2::MarginRankingCriterionGradientOp<caffe2::CUDAContext>`




---


# MatMul

Matrix multiplication Y = A * B, where A has size (M x K), B has size (K x N).

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`trans_a`
</td><td>Pass 1 to transpose A before multiplication
</td></tr><tr><td>`trans_b`
</td><td>Pass 1 to transpose B before multiplication
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>2D matrix of size (M x K)
</td></tr><tr><td>`B`
</td><td>2D matrix of size (K x N)
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D product
</td></tr></table>
### Code
[caffe2/operators/matmul_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/matmul_op.cc)
### Devices

- *CPU* `caffe2::MatMulOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`

- *GPU* `caffe2::MatMulOp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`




---


# Max

Element-wise max of each of the input tensors. The first input tensor can be used in-place as the output tensor, in which case the max will be done in place and results will be accumulated in input0. All inputs and outputs must have the same shape and data type.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data_0`
</td><td>First of the input tensors. Can be inplace.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`max`
</td><td>Output tensor. Same dimension as inputs.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::MaxOp<float, caffe2::CPUContext>`




---


# MaxGradient
No documentation yet.

### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::MaxGradientOp<float, caffe2::CPUContext>`




---


# MaxPool

MaxPool consumes an input blob X and applies max pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Max pooling consisting of taking the maximumvalue of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. 
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
</td></tr></table>
### Code
[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)
### Devices

- *CPU* `caffe2::PoolOp<float, caffe2::CPUContext, caffe2::(anonymous namespace)::MaxPool>`

- *GPU* `caffe2::PoolOp<float, caffe2::CUDAContext, caffe2::(anonymous namespace)::MaxPool>`



### Engines
`NVRTC` on *CUDA*`CUDNN` on *CUDA*`NNPACK` on *CPU*

---


# MaxPoolGradient
No documentation yet.

### Code
[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)
### Devices

- *CPU* `caffe2::PoolGradientOp<float, caffe2::CPUContext, caffe2::(anonymous namespace)::MaxPool>`

- *GPU* `caffe2::PoolGradientOp<float, caffe2::CUDAContext, caffe2::(anonymous namespace)::MaxPool>`



### Engines
`CUDNN` on *CUDA*`NVRTC` on *CUDA*

---


# Mul

Performs element-wise binary multiplication (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and type as A
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::EigenMulFunctor, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaMulFunctor, caffe2::SameTypeAsInput>`




---


# MultiClassAccuracy

Respectively compute accuracy score for each class given a number of instances and predicted scores of each class for each instance.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`prediction`
</td><td>2-D float tensor (N,D,) of predicted scores of each class for each data. N is the number of instances, i.e., batch size. D is number of possible classes/labels.
</td></tr><tr><td>`labels`
</td><td>1-D int tensor (N,) of labels for each instance.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`accuracies`
</td><td>1-D float tensor (D,) of accuracy for each class. If a class has no instance in the batch, its accuracy score is set to zero.
</td></tr><tr><td>`amounts`
</td><td>1-D int tensor (D,) of number of instances for each class in the batch.
</td></tr></table>
### Code
[caffe2/operators/multi_class_accuracy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/multi_class_accuracy_op.cc)
### Devices

- *CPU* `caffe2::MultiClassAccuracyOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::MultiClassAccuracyOp<float, caffe2::CUDAContext>`




---


# NCHW2NHWC

The operator switches the order of data in a tensor from NCHW- sample index N, channels C, height H and width W, to the NHWC order.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>The input data (Tensor<float>) in the NCHW order.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The output tensor (Tensor<float>) in the NHWC order.
</td></tr></table>
### Code
[caffe2/operators/order_switch_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/order_switch_ops.cc)
### Devices

- *CPU* `caffe2::NCHW2NHWCOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::NCHW2NHWCOp<float, caffe2::CUDAContext>`




---


# NHWC2NCHW

The operator switches the order of data in a tensor from NHWC- sample index N, height H, width H and channels C, to the NCHW order.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>The input data (Tensor<float>) in the NHWC order.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The output tensor (Tensor<float>) in the NCHW order.
</td></tr></table>
### Code
[caffe2/operators/order_switch_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/order_switch_ops.cc)
### Devices

- *CPU* `caffe2::NHWC2NCHWOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::NHWC2NCHWOp<float, caffe2::CUDAContext>`




---


# Negative

Computes the element-wise negative of the input.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D input tensor
</td></tr></table>
### Code
[caffe2/operators/negative_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/negative_op.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float, double, int, long>, caffe2::CPUContext, caffe2::WithDefaultConstructor<caffe2::NegativeCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float, double, int, long>, caffe2::CUDAContext, caffe2::WithDefaultConstructor<caffe2::NegativeCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# Normalize

Given a matrix, apply L2-normalization along the last dimension.

### Code
[caffe2/operators/normalize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/normalize_op.cc)
### Devices

- *CPU* `caffe2::NormalizeOp<float, caffe2::CPUContext>`




---


# NormalizeGradient
No documentation yet.

### Code
[caffe2/operators/normalize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/normalize_op.cc)
### Devices

- *CPU* `caffe2::NormalizeGradientOp<float, caffe2::CPUContext>`




---


# Not
Performs element-wise negation.
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input tensor of type `bool`.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output tensor of type `bool`.
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<bool>, caffe2::CPUContext, caffe2::WithDefaultConstructor<caffe2::NotFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<bool>, caffe2::CUDAContext, caffe2::WithDefaultConstructor<caffe2::CudaNotFunctor>, caffe2::SameTypeAsInput>`




---


# OneHot

Given a sequence of indices, one for each example in a batch, returns a matrix where each inner dimension has the size of the index and has 1.0 in the index active in the given example, and 0.0 everywhere else.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`indices`
</td><td>The active index for each example in the batch.
</td></tr><tr><td>`index_size_tensor`
</td><td>Scalar with the size of the index.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`one_hots`
</td><td>Matrix of size len(indices) x index_size
</td></tr></table>
### Code
[caffe2/operators/one_hot_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::OneHotOp`




---


# Or

Performs element-wise logical operation  `or`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<bool>, caffe2::CPUContext, caffe2::NaiveOrFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<bool>, caffe2::CUDAContext, caffe2::CudaOrFunctor, caffe2::FixedType<bool> >`




---


# PRelu

 PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one output data (Tensor<T>) where the function  `f(x) = slope * x for x < 0` ,  `f(x) = x for x >= 0` ., is applied to the data tensor elementwise.
 
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>`Slope`
</td><td>1D slope tensor. If `Slope` is of size 1, the value is sharedacross different channels
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D input tensor
</td></tr></table>
### Code
[caffe2/operators/prelu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/prelu_op.cc)
### Devices

- *CPU* `caffe2::PReluOp<float, caffe2::CPUContext>`




---


# PReluGradient

 PReluGradient takes both Y and dY and uses this to update dX and dW according to the chain rule and derivatives of the rectified linear function.
 
### Code
[caffe2/operators/prelu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/prelu_op.cc)
### Devices

- *CPU* `caffe2::PReluGradientOp<float, caffe2::CPUContext>`




---


# PackSegments
Map N dim tensor to N+1 dim based on length blob. Sequences that     are shorter than the longest sequence are padded with zeros.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`pad_minf`
</td><td>Padding number in the packed segments. Use true to pad     -infinity, otherwise pad zeros
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>1-d int/long tensor contains the length in each of the output.
</td></tr><tr><td>`tensor`
</td><td>N dim Tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`packed_tensor`
</td><td>N + 1 dim Tesorwhere dim(1) is the max length, dim(0) is the batch size.
</td></tr></table>
### Code
[caffe2/operators/pack_segments.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pack_segments.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::PackSegmentsOp<caffe2::CPUContext>`




---


# PadEmptySamples

Pad empty field given lengths and index features,  Input(0) is a blob pointing to the lengths of samples in one batch, [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the features.
 PadEmptySamples is thread safe.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>A blob containing a pointer to the lengths.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`out_lengths`
</td><td>Tensor containing lengths with empty sample padded.
</td></tr></table>
### Code
[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::PadEmptySamplesOp`




---


# PadImage

PadImage pads values around the boundary of an image according to the pad values and stride sizes defined by the ConvPoolOpBase operator.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. 
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output data tensor from padding the H and W dimensions on the tensor. Dimensions will vary based on various pad and stride sizes.
</td></tr></table>
### Code
[caffe2/operators/pad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pad_op.cc)
### Devices

- *CPU* `caffe2::PadImageOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::PadImageOp<float, caffe2::CUDAContext>`




---


# PadImageGradient
No documentation yet.

### Code
[caffe2/operators/pad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pad_op.cc)
### Devices

- *CPU* `caffe2::PadImageGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::PadImageGradientOp<float, caffe2::CUDAContext>`




---


# Partition

Splits the input int tensor into multiple ones according to the first tensor.
 Takes the first input and partitions it to shards according to the remainder of values modulo the number of partitions. It requires that the first tensor is of integral type. The number of partitions is derived as (num_output / num_input).
 If additional inputs are present they must have the same shape as the first input, optionally with extra trailing dimensions. They will be partitioned accordingly to the first input.
 Optional arg 'pack_first_input' transforms the first tensor values as X_ij / num_partitions.
 Outputs are ordered as X_0_part_0, X_1_part_0, ..., X_N-1_part_0, X_0_part_1, ..., X_N-1_part_K-1 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`pack_first_input`
</td><td>(int, default 0) If set, the operator transforms the first tensor values as floor(X_ij / num_partitions)
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>Input tensor containing data to be partitioned. The number of input tensors might be greater than 1 but must have the same shape as the previous tensors.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`partitions`
</td><td>Output Partitions. The number of output tensors has to be a multiple of the number of input tensors.
</td></tr></table>
### Code
[caffe2/operators/partition_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/partition_ops.cc)
### Devices

- *CPU* `caffe2::PartitionOp`




---


# Perplexity

Perplexity calculates how well a probability distribution predicts a sample.
Perplexity takes a 1-D tensor containing a batch of probabilities. Each value in the tensor belongs to a different sample and represents the probability of the model predicting the true label for that sample. The operator returns a single (float) perplexity value for the batch.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`probabilities`
</td><td>The input data as Tensor. It contains a batch oftrue label or target probabilities
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The output- a single (float) perplexity value for the batch
</td></tr></table>
### Code
[caffe2/operators/perplexity_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/perplexity_op.cc)
### Devices

- *CPU* `caffe2::PerplexityOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::PerplexityOp<float, caffe2::CUDAContext>`




---


# Print
Logs shape and contents of input tensor to stderr or to a file.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`to_file`
</td><td>(bool) if 1, saves contents to the root folder of the current workspace, appending the tensor contents to a file named after the blob name. Otherwise, logs to stderr.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tensor`
</td><td>The tensor to print.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::PrintOp<caffe2::CPUContext>`

- *GPU* `caffe2::PrintOp<caffe2::CUDAContext>`




---


# QPSMetric

QPSMetric operator syncronously updates metric storedcreate a blob that will store state that is required for computing QPSMetric. The only output of the operator will have blob with QPSMetricState as an output.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`QPS_METRIC_STATE`
</td><td>Input Blob QPSMetricState, that needs to be updated
</td></tr><tr><td>`INPUT_BATCH`
</td><td>Input Blob containing a tensor with batch of the examples. First dimension of the batch will be used to get the number of examples in the batch.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Blob with QPSMetricState
</td></tr></table>
### Code
[caffe2/operators/metrics_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/metrics_ops.cc)
### Devices

- *CPU* `caffe2::QPSMetricOp`




---


# QPSMetricReport

QPSMetricReport operator that syncronously consumes the QPSMetricState blob and reports the information about QPS.

### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Blob with QPSMetricState
</td></tr></table>
### Code
[caffe2/operators/metrics_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/metrics_ops.cc)
### Devices

- *CPU* `caffe2::QPSMetricReportOp`




---


# RangeFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::RangeFillOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::RangeFillOp<float, caffe2::CUDAContext>`




---


# ReadNextBatch

Read the next batch of examples out of the given cursor and data blobs.
 Input(0) is a blob pointing to a TreeCursor, and [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 ReadNextBatch is thread safe.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`batch_size`
</td><td>Number of top-level entries to read.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`cursor`
</td><td>A blob containing a pointer to the cursor.
</td></tr><tr><td>`dataset_field_0`
</td><td>First dataset field
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`field_0`
</td><td>Tensor containing the next batch for field 0.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::ReadNextBatchOp`




---


# ReadRandomBatch

Read the next batch of examples out of the given cursor, idx blob, offset matrix and data blobs.
 Input(0) is a blob pointing to a TreeCursor, Input(1) is a blob pointing to the shuffled idx Input(2) is a blob pointing to the offset matrix and [Input(3),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 ReadRandomBatch is thread safe.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`batch_size`
</td><td>Number of top-level entries to read.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`cursor`
</td><td>A blob containing a pointer to the cursor.
</td></tr><tr><td>`idx`
</td><td>idx with a shuffled order.
</td></tr><tr><td>`offsetsmat`
</td><td>offset matrix containing length offset info.
</td></tr><tr><td>`dataset_field_0`
</td><td>First dataset field
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`field_0`
</td><td>Tensor containing the next batch for field 0.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::ReadRandomBatchOp`




---


# ReceiveTensor

Receives the tensor from another node.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`src`
</td><td>(int) he rank to receive the tensor from.
</td></tr><tr><td>`tag`
</td><td>(int) a tag to receive the tensor with.
</td></tr><tr><td>`raw_buffer`
</td><td>(bool) if set, only send the content and assume that the receiver has already known the tensor's shape and information.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`comm_world`
</td><td>The common world.
</td></tr><tr><td>`Y`
</td><td>In-place output. If raw_buffer is specified, Y should have pre-allocated data and type..
</td></tr><tr><td>`src`
</td><td>An int CPUtensor of size 1 specifying the rank. If given, this overrides the 'from' argument of the op.
</td></tr><tr><td>`tag`
</td><td>An int CPUtensor of size 1 specifying the tag to send the tensor with. This overrides the 'tag' argument of the op.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>The received tensor.
</td></tr><tr><td>`src`
</td><td>The sender that sent the message as a CPUTensor of size 1 and of type int.
</td></tr><tr><td>`tag`
</td><td>The tag that the message is sent with as a CPUTensor of size 1 and of type int.
</td></tr></table>
### Code
[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)
### Devices

- *CPU* `caffe2::NoDefaultEngineOp<caffe2::CPUContext>`

- *GPU* `caffe2::NoDefaultEngineOp<caffe2::CUDAContext>`




---


# RecurrentNetwork

 Run the input network in a recurrent fashion. This can be used to implement fairly general recurrent neural networks (RNNs).
 The operator proceeds as follows.
 - First, initialized the states from the input recurrent states - For each timestep T, apply the links (that map offsets from input/output  
````
  tensors into the inputs/outputs for the `step` network)

````
 - Finally, alias the recurrent states to the specified output blobs.
 This is a fairly special-case meta-operator, and so the implementation is somewhat complex. It trades of generality (and frankly usability) against performance and control (compared to e.g. TF dynamic_rnn, Theano scan, etc).
 See the usage examples for a flavor of how to use it.
 
### Code
[caffe2/operators/recurrent_network_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/recurrent_network_op.cc)
### Devices

- *CPU* `caffe2::RecurrentNetworkOp<float, caffe2::CPUContext>`




---


# RecurrentNetworkGradient
No documentation yet.

### Code
[caffe2/operators/recurrent_network_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/recurrent_network_op.cc)
### Devices

- *CPU* `caffe2::RecurrentNetworkGradientOp<float, caffe2::CPUContext>`




---


# Reduce

Does a reduce operation from every node to the root node. Currently only Sum is supported.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`root`
</td><td>(int, default 0) the root to run reduce into.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`comm_world`
</td><td>The common world.
</td></tr><tr><td>`X`
</td><td>A tensor to be reduced.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>The reduced result on root, not set for other nodes.
</td></tr></table>
### Code
[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)
### Devices

- *CPU* `caffe2::NoDefaultEngineOp<caffe2::CPUContext>`

- *GPU* `caffe2::NoDefaultEngineOp<caffe2::CUDAContext>`




---


# ReduceFrontSum

Reduces the input tensor along the first dimension of the input tensor by applying 'Sum'. This op acts in a similar way to SortedSegmentSum and UnsortedSegmentSum but as if all input slices belong to a single segment.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor to be reduced on the first dimension
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated tensor
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractReduceFrontOp<float, caffe2::CPUContext, caffe2::SumReducer<float, caffe2::CPUContext> >`




---


# ReduceFrontSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractReduceFrontGradientOp<float, caffe2::CPUContext, caffe2::SumReducerGradient<float, caffe2::CPUContext> >`




---


# ReduceFrontWeightedSum

Reduces the input tensor along the first dimension of the input tensor by applying 'WeightedSum'. This op acts in a similar way to SortedSegmentWeightedSum and UnsortedSegmentWeightedSum but as if all input slices belong to a single segment.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor for the summation
</td></tr><tr><td>`SCALARS`
</td><td>Scalar multipliers for the input slices. Must be a vector with the length matching the first dimension of DATA
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated tensor
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractReduceFrontOp<float, caffe2::CPUContext, caffe2::WeightedSumReducer<float, caffe2::CPUContext> >`




---


# ReduceFrontWeightedSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractReduceFrontGradientOp<float, caffe2::CPUContext, caffe2::WeightedSumReducerGradient<float, caffe2::CPUContext> >`




---


# Relu

Relu takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to the tensor elementwise.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D input tensor
</td></tr></table>
### Code
[caffe2/operators/relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/relu_op.cc)
### Devices

- *CPU* `caffe2::ReluOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ReluOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*

---


# ReluGradient

ReluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.

### Code
[caffe2/operators/relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/relu_op.cc)
### Devices

- *CPU* `caffe2::ReluGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ReluGradientOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*

---


# RemoveDataBlocks

Shrink the data tensor by removing data blocks with given zero-based indices in the outermost dimension of the tensor. Indices are not assumed in any order or unique but with the range [0, blocks_size). Indices could be empty.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>a N-D data tensor, N >= 1
</td></tr><tr><td>`indices`
</td><td>zero-based indices of blocks to be removed
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`shrunk data`
</td><td>data after removing data blocks indexed by 'indices'
</td></tr></table>
### Code
[caffe2/operators/remove_data_blocks_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/remove_data_blocks_op.cc)
### Devices

- *CPU* `caffe2::RemoveDataBlocksOp<caffe2::CPUContext>`




---


# RemovePadding

Remove padding around the edges of each segment of the input data. This is the reverse opration of AddPadding, and uses the same arguments and conventions for input and output data format.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`pad_width`
</td><td>Outer-size of padding to remove around each range.
</td></tr><tr><td>`end_pad_width`
</td><td>(Optional) Specifies a different end-padding width.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data_in`
</td><td>T<N, D1..., Dn> Input data
</td></tr><tr><td>`lengths`
</td><td>(i64) Num of elements in each range. sum(lengths) = N. If not provided, considers all data as a single segment.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`data_out`
</td><td>(T<N - 2*pad_width, D1..., Dn>) Unpadded data.
</td></tr><tr><td>`lengths_out`
</td><td>(i64, optional) Lengths for each unpadded range.
</td></tr></table>
### Code
[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::RemovePaddingOp`




---


# ResetCounter

Resets a count-down counter with initial value specified by the 'init_count' argument.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`init_count`
</td><td>Resets counter to this value, must be >= 0.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`counter`
</td><td>A blob pointing to an instance of a new counter.
</td></tr></table>
### Code
[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)
### Devices

- *CPU* `caffe2::ResetCounterOp<long, caffe2::CPUContext>`

- *GPU* `caffe2::ResetCounterOp<long, caffe2::CUDAContext>`




---


# ResetCursor

Resets the offsets for the given TreeCursor. This operation is thread safe.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`cursor`
</td><td>A blob containing a pointer to the cursor.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::ResetCursorOp`




---


# Reshape

Reshape the input tensor similar to numpy.reshape.
 It takes a tensor as input and an optional tensor specifying the new shape.
When the second input is absent, an extra argument  `shape`  must be specified.
It outputs the reshaped tensor as well as the original shape.
 At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions. A dimension could also be 0, in which case the actual dimension value is going to be copied from the input tensor.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`shape`
</td><td>New shape
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>An input tensor.
</td></tr><tr><td>`new_shape`
</td><td>New shape.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`reshaped`
</td><td>Reshaped data.
</td></tr><tr><td>`old_shape`
</td><td>Original shape.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::ReshapeOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ReshapeOp<float, caffe2::CUDAContext>`




---


# ResizeLike

Produces tensor condaining data of first input and shape of second input.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>Tensor whose data will be copied into the output.
</td></tr><tr><td>`shape_tensor`
</td><td>Tensor whose shape will be applied to output.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Tensor with data of input 0 and shape of input 1.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::ResizeLikeOp<caffe2::CPUContext>`

- *GPU* `caffe2::ResizeLikeOp<caffe2::CUDAContext>`




---


# RetrieveCount

Retrieve the current value from the counter.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`counter`
</td><td>A blob pointing to an instance of a counter.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`count`
</td><td>current count value.
</td></tr></table>
### Code
[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)
### Devices

- *CPU* `caffe2::RetrieveCountOp<long, caffe2::CPUContext>`

- *GPU* `caffe2::RetrieveCountOp<long, caffe2::CUDAContext>`




---


# ReversePackedSegs

Reverse segments in a 3-D tensor (lengths, segments, embeddings,), leaving paddings unchanged. This operator is used to reverse input of a recurrent neural network to make it a BRNN.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>a 3-D (lengths, segments, embeddings,) tensor.
</td></tr><tr><td>`lengths`
</td><td>length of each segment.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`reversed data`
</td><td>a (lengths, segments, embeddings,) tensor with each segment reversedand paddings unchanged.
</td></tr></table>
### Code
[caffe2/operators/reverse_packed_segs_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reverse_packed_segs_op.cc)
### Devices

- *CPU* `caffe2::ReversePackedSegsOp<caffe2::CPUContext>`




---


# Save

The Save operator saves a set of blobs to a db. It takes [1, infinity) number of inputs and has no output. The contents of the inputs are written into the db specified by the arguments.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`absolute_path`
</td><td>(int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace.
</td></tr><tr><td>`db`
</td><td>(string) the path to the db to load.
</td></tr><tr><td>`db_type`
</td><td>(string) the type of the db.
</td></tr></table>
### Code
[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)
### Devices

- *CPU* `caffe2::SaveOp<caffe2::CPUContext>`

- *GPU* `caffe2::SaveOp<caffe2::CUDAContext>`




---


# Scale

Scale takes one input data (Tensor<float>) and produces one output data (Tensor<float>) whose value is the input data tensor scaled element-wise.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`scale`
</td><td>(float, default 1.0) the scale to apply.
</td></tr></table>
### Code
[caffe2/operators/scale_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/scale_op.cc)
### Devices

- *CPU* `caffe2::ScaleOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::ScaleOp<float, caffe2::CUDAContext>`




---


# ScatterAssign

Update slices of the tensor in-place by overriding current value.
 Note: The op pretty much ignores the exact shapes of the input arguments and cares only about sizes. It's done for performance consideration to avoid unnecessary reshapes. Only first dimension of X_0 is important, let's call it N. If M is the total size of X_0 and K is the size of INDICES then X_i is assumed to be of shape K x (M / N) regardless of the real shape.
 Note: Each update in INDICES is applied independently which means that if duplicated elements are present in INDICES arbitrary one will win.
 Currently only works on CPU because of access to INDICES.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Tensor to be updated.
</td></tr><tr><td>`INDICES`
</td><td>1-D list of indices on the first dimensionof X_0 that need to be updated
</td></tr><tr><td>`SLICES`
</td><td>Update slices, with shape len(INDICES) + shape(X_0)[1:]
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Has to be exactly the same tensor as the input 0
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::ScatterAssignOp<float, caffe2::CPUContext>`




---


# ScatterWeightedSum

Similar to WeightedSum, computes the weighted sum of several tensors, with the difference that inputs are sliced tensors. The first tensor has to be in-place and only slices of it on the first dimension as indexed by INDICES will be updated.
 Note: The op pretty much ignores the exact shapes of the input arguments and cares only about sizes. It's done for performance consideration to avoid unnecessary reshapes. Only first dimension of X_0 is important, let's call it N. If M is the total size of X_0 and K is the size of INDICES then X_i is assumed to be of shape K x (M / N) regardless of the real shape.
 Note: Each update in INDICES is applied independently which means that if duplicated elements are present in INDICES the corresponding slice of X_0 will be scaled multiple times. Manual collapsing of INDICES is required beforehand if necessary.
 Note: Updates are applied sequentially by inputs which might have undesired consequences if the input tensor is accessed concurrently by different op (e.g. when doing Hogwild). Other threads might see intermediate results even on individual slice level, e.g. X_0 scaled by weight_0 but without any updates applied.
 Currently only works on CPU because of access to INDICES.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X_0`
</td><td>Tensor to be updated.
</td></tr><tr><td>`Weight_0`
</td><td>Scalar weight for X_0, applied only to slices affected.
</td></tr><tr><td>`INDICES`
</td><td>1-D list of indices on the first dimension of X_0 that need to be updated
</td></tr><tr><td>`X_1`
</td><td>Update slices, with shape len(INDICES) + shape(X_0)[1:]
</td></tr><tr><td>`Weight_1`
</td><td>Scalar weight for X_1 update
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`X_0`
</td><td>Has to be exactly the same tensor as the input 0
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::ScatterWeightedSumOp<float, caffe2::CPUContext>`




---


# SegmentIdsToLengthWeights
 Similar as SegmentIdsToLengths but output vector of segment weights derived by lengths. i.e 1/pow(length, power) 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`power`
</td><td>n of 1/pow(length,n) for normalization
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`segment_ids`
</td><td>1-D int32_t or int64_t tensor of segment ids
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`a vector of weights`
</td><td>1-D float tensor of segment weights by length
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::SegmentIdsToLengthWeightsOp<caffe2::CPUContext>`




---


# SegmentIdsToLengths

Transfers a vector of segment ids to a vector of segment lengths. This operation supports non-consecutive segment ids. Segments not appearing in the input vector will have length 0. If the second input is provided, the number of segments = the size of its first dimension. Otherwise, the number of segments = the last index in the first input vector + 1.
 In general, for consecutive, zero-based segment IDs, this is the inverse operation of LengthsToSegmentIds, except that a vector of segment IDs cannot represent empty segments at the end (if the second input is absent).

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`segment_ids`
</td><td>1-D int32_t or int64_t tensor of segment ids
</td></tr><tr><td>`data (optional)`
</td><td>if provided, number of segments = the size of its first dimension
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>1-D int64_t tensor of segment lengths
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::SegmentIdsToLengthsOp<caffe2::CPUContext>`




---


# SegmentIdsToRanges

Transfers a vector of segment ids to a vector of segment ranges. This operation supports non-consecutive segment ids. Segments not appearing in the input vector will have length 0. If the second input is provided, the number of segments = the size of its first dimension. Otherwise, the number of segments = the last index in the first input vector + 1.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`segment_ids`
</td><td>1-D int32_t or int64_t tensor of segment ids
</td></tr><tr><td>`data (optional)`
</td><td>if provided, number of segments = the size of its first dimension
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>1-D int64_t tensor of segment lengths
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::SegmentIdsToRangesOp<caffe2::CPUContext>`




---


# SegmentOneHot

Given a sequence of indices, segmented by the lengths tensor, returns a matrix that has the elements in each sequence set to 1.0, and 0.0 everywhere else.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>Size of each segment.
</td></tr><tr><td>`indices`
</td><td>Active indices, of size sum(lengths)
</td></tr><tr><td>`index_size_tensor`
</td><td>Size of the index
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`one_hots`
</td><td>Matrix of size len(lengths) x index_size
</td></tr></table>
### Code
[caffe2/operators/one_hot_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::SegmentOneHotOp`




---


# SendTensor

Sends the tensor to another node.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`dst`
</td><td>The rank to send the tensor to.
</td></tr><tr><td>`tag`
</td><td>(int) a tag to send the tensor with.
</td></tr><tr><td>`raw_buffer`
</td><td>(bool) if set, only send the content and assume that the receiver has already known the tensor's shape and information.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`comm_world`
</td><td>The common world.
</td></tr><tr><td>`X`
</td><td>A tensor to be allgathered.
</td></tr><tr><td>`dst`
</td><td>An int CPUtensor of size 1 specifying the rank. If given, this overrides the 'to' argument of the op.
</td></tr><tr><td>`tag`
</td><td>An int CPUtensor of size 1 specifying the tag to send the tensor with. This overrides the 'tag' argument of the op.
</td></tr></table>
### Code
[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)
### Devices

- *CPU* `caffe2::NoDefaultEngineOp<caffe2::CPUContext>`

- *GPU* `caffe2::NoDefaultEngineOp<caffe2::CUDAContext>`




---


# Shape
Produce a 1D int64 tensor with the shape of the input tensor.
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::ShapeOp<caffe2::CPUContext>`




---


# Sigmoid

Sigmoid takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the tensor elementwise.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D output tensor
</td></tr></table>
### Code
[caffe2/operators/sigmoid_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sigmoid_op.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CPUContext, caffe2::WithDefaultConstructor<caffe2::SigmoidCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CUDAContext, caffe2::WithDefaultConstructor<caffe2::SigmoidCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# SigmoidCrossEntropyWithLogits

Given two matrices logits and targets, of same shape, (batch_size, num_classes), computes the sigmoid cross entropy between the two.
Returns a tensor of shape (batch_size,) of losses for each example.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`logits`
</td><td>matrix of logits for each example and class.
</td></tr><tr><td>`targets`
</td><td>matrix of targets, same shape as logits.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`xentropy`
</td><td>Vector with the total xentropy for each example.
</td></tr></table>
### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::SigmoidCrossEntropyWithLogitsOp<float, caffe2::CPUContext>`




---


# SigmoidCrossEntropyWithLogitsGradient
No documentation yet.

### Code
[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)
### Devices

- *CPU* `caffe2::SigmoidCrossEntropyWithLogitsGradientOp<float, caffe2::CPUContext>`




---


# SigmoidGradient

SigmoidGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the sigmoid function.

### Code
[caffe2/operators/sigmoid_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sigmoid_op.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<float>, caffe2::CPUContext, caffe2::WithoutBroadcast<caffe2::SigmoidGradientCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<float>, caffe2::CUDAContext, caffe2::WithoutBroadcast<caffe2::SigmoidGradientCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# Slice

Produces a slice of the input tensor. Currently, only slicing in a single dimension is supported.
Slices are passed as 2 1D vectors with starting and end indices for each dimension of the input  `data`  tensor. End indices are non-inclusive. If a negative value is passed for any of the start or end indices, it represent number of elements before the end of that dimension.
 Example:   
````
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 3]

  result = [
      [2, 3],
      [6, 7],
  ]

````

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>Tensor of data to extract slices from.
</td></tr><tr><td>`starts`
</td><td>1D tensor: start-indices for each dimension of data.
</td></tr><tr><td>`ends`
</td><td>1D tensor: end-indices for each dimension of data.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Sliced data tensor.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::SliceOp<int, caffe2::CPUContext>`




---


# Snapshot

The Snapshot operator is similar to the Save operator, but allows one to save to db every few iterations, with a db name that is appended with the iteration count. It takes [1, infinity) number of inputs and has no output. The first input has to be a TensorCPU of type int and has size 1 (i.e. the iteration counter). This is determined whether we need to do snapshotting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`absolute_path`
</td><td>(int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace.
</td></tr><tr><td>`db`
</td><td>(string) a template string that one can combine with the iteration to create the final db name. For example, "/home/lonestarr/checkpoint_%08d.db"
</td></tr><tr><td>`db_type`
</td><td>(string) the type of the db.
</td></tr><tr><td>`every`
</td><td>(int, default 1) the snapshotting is carried out when (iter mod every) is zero.
</td></tr></table>
### Code
[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)
### Devices

- *CPU* `caffe2::SnapshotOp<caffe2::CPUContext>`

- *GPU* `caffe2::SnapshotOp<caffe2::CUDAContext>`




---


# Softmax

The operator computes the softmax normalized values for each layer in the batch  of the given input. The input is a 2-D tensor (Tensor<float>) of size (batch_size x input_feature_dimensions). The output tensor has the same shape and contains the softmax normalized values of the corresponding input.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>The input data as 2-D Tensor<float>.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The softmax normalized output values with the same shape as input tensor.
</td></tr></table>
### Code
[caffe2/operators/softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_op.cc)
### Devices

- *CPU* `caffe2::SoftmaxOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SoftmaxOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*

---


# SoftmaxGradient
No documentation yet.

### Code
[caffe2/operators/softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_op.cc)
### Devices

- *CPU* `caffe2::SoftmaxGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SoftmaxGradientOp<float, caffe2::CUDAContext>`



### Engines
`CUDNN` on *CUDA*

---


# SoftmaxWithLoss

Combined Softmax and Cross-Entropy loss operator.
The operator computes the softmax normalized values for each layer in the batch of the given input, after which cross-entropy loss is computed. This operator is numerically more stable than separate Softmax and CrossEntropy ops.
The inputs are a 2-D tensor (Tensor<float>) of size (batch_size x input_feature_dimensions) and tensor of labels (ground truth).
Output is tensor with the probability for each label for each example (N x D) and averaged loss (scalar). Use parameter spatial=1 to enable spatial softmax.
Spatial softmax also supports special \"don't care\" label (-1) that is ignored when computing the loss.
 Optional third input blob can be used to weight the samples for the loss.
For the spatial version, weighting is by x,y position of the input.

### Code
[caffe2/operators/softmax_with_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_with_loss_op.cc)
### Devices

- *CPU* `caffe2::SoftmaxWithLossOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SoftmaxWithLossOp<float, caffe2::CUDAContext>`




---


# SoftmaxWithLossGradient
No documentation yet.

### Code
[caffe2/operators/softmax_with_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_with_loss_op.cc)
### Devices

- *CPU* `caffe2::SoftmaxWithLossGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SoftmaxWithLossGradientOp<float, caffe2::CUDAContext>`




---


# Softsign

Calculates the softsign (x/1+|x|) of the given input tensor element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>1-D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The softsign (x/1+|x|) values of the input tensor computed element-wise
</td></tr></table>
### Code
[caffe2/operators/softsign_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softsign_op.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CPUContext, caffe2::WithDefaultConstructor<caffe2::SoftsignCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CUDAContext, caffe2::WithDefaultConstructor<caffe2::SoftsignCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# SoftsignGradient

Calculates the softsign gradient (sgn(x)/(1+|x|)^2) of the given input tensor element-wise.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>1-D input tensor
</td></tr><tr><td>`input`
</td><td>1-D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The softsign gradient (sgn(x)/(1+|x|)^2) values of the input tensor computed element-wise
</td></tr></table>
### Code
[caffe2/operators/softsign_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softsign_op.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<float>, caffe2::CPUContext, caffe2::WithoutBroadcast<caffe2::SoftsignGradientCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<float>, caffe2::CUDAContext, caffe2::WithoutBroadcast<caffe2::SoftsignGradientCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# SortAndShuffle

Compute the sorted indices given a field index to sort by and break the sorted indices into chunks of shuffle_size * batch_size and shuffle each chunk, finally we shuffle between batches. If sort_by_field_idx is -1 we skip sort.
 For example, we have data sorted as 1,2,3,4,5,6,7,8,9,10,11,12  and batchSize = 2 and shuffleSize = 3, when we shuffle we get: [3,1,4,6,5,2] [12,10,11,8,9,7]  After this we will shuffle among different batches with size 2 [3,1],[4,6],[5,2],[12,10],[11,8],[9,7]  We may end up with something like [9,7],[5,2],[12,10],[4,6],[3,1],[11,8]  Input(0) is a blob pointing to a TreeCursor, and [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 SortAndShuffle is thread safe.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`cursor`
</td><td>A blob containing a pointer to the cursor.
</td></tr><tr><td>`dataset_field_0`
</td><td>First dataset field
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`indices`
</td><td>Tensor containing sorted indices.
</td></tr></table>
### Code
[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::SortAndShuffleOp`




---


# SortedSegmentRangeLogMeanExp

Applies 'LogMeanExp' to each segment of input tensor. In order to allow for more efficient implementation of 'LogMeanExp', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 LogMeanExp computes the element-wise log of the mean of exponentials of input slices. Operation doesn't change the shape of individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor to be aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeOp<float, int, caffe2::CPUContext, caffe2::LogMeanExpRangeReducer<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeLogMeanExpGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeGradientOp<float, int, caffe2::CPUContext, caffe2::LogMeanExpRangeReducerGradient<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeLogSumExp

Applies 'LogSumExp' to each segment of input tensor. In order to allow for more efficient implementation of 'LogSumExp', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 LogSumExp computes the element-wise log of the sum of exponentials of input slices. Operation doesn't change the shape of individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor to be aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeOp<float, int, caffe2::CPUContext, caffe2::LogSumExpRangeReducer<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeLogSumExpGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeGradientOp<float, int, caffe2::CPUContext, caffe2::LogSumExpRangeReducerGradient<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeMax

Applies 'Max' to each segment of input tensor. In order to allow for more efficient implementation of 'Max', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Max computation is done element-wise, so that each element of the output slice corresponds to the max value of the respective elements in the input slices. Operation doesn't change the shape of individual blocks. This implementation imitates torch nn.Max operator. If the maximum value occurs more than once, the operator will return the first occurence of value. When computing the gradient using the backward propagation, the gradient input corresponding to the first occurence of the maximum value will be used.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor to be aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeOp<float, int, caffe2::CPUContext, caffe2::MaxRangeReducer<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeMaxGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeGradientOp<float, int, caffe2::CPUContext, caffe2::MaxRangeReducerGradient<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeMean

Applies 'Mean' to each segment of input tensor. In order to allow for more efficient implementation of 'Mean', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Mean computation is done element-wise, so that each element of the output slice corresponds to the average value of the respective elements in the input slices. Operation doesn't change the shape of individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor to be aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeOp<float, int, caffe2::CPUContext, caffe2::MeanRangeReducer<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeMeanGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeGradientOp<float, int, caffe2::CPUContext, caffe2::MeanRangeReducerGradient<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeSum

Applies 'Sum' to each segment of input tensor. In order to allow for more efficient implementation of 'Sum', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor to be aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeOp<float, int, caffe2::CPUContext, caffe2::SumRangeReducer<float, caffe2::CPUContext> >`




---


# SortedSegmentRangeSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentRangeGradientOp<float, int, caffe2::CPUContext, caffe2::SumRangeReducerGradient<float, caffe2::CPUContext> >`




---


# SortedSegmentSum

Applies 'Sum' to each segment of input tensor. Segments need to be sorted and contiguous. See also UnsortedSegmentSum that doesn't have this requirement.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor, slices of which are aggregated.
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of K (the number of segments).
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentOp<float, int, caffe2::CPUContext, caffe2::SumReducer<float, caffe2::CPUContext>, false>`




---


# SortedSegmentSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::SumReducerGradient<float, caffe2::CPUContext> >`




---


# SortedSegmentWeightedSum

Applies 'WeightedSum' to each segment of input tensor. Segments need to be sorted and contiguous. See also UnsortedSegmentWeightedSum that doesn't have this requirement.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor for the summation
</td></tr><tr><td>`SCALARS`
</td><td>Scalar multipliers for the input slices. Must be a vector with the length matching the first dimension of DATA
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of K (the number of segments).
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducer<float, caffe2::CPUContext>, false>`




---


# SortedSegmentWeightedSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducerGradient<float, caffe2::CPUContext> >`




---


# SpaceToBatch

 SpaceToBatch for 4-D tensors of type T.
 Zero-pads and then rearranges (permutes) blocks of spatial data into batch. More specifically, this op outputs a copy of the input tensor where values from the height and width dimensions are moved to the batch dimension. After the zero-padding, both height and width of the input must be divisible by the block size.
 
### Code
[caffe2/operators/space_batch_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/space_batch_op.cc)
### Devices

- *CPU* `caffe2::SpaceToBatchOp<caffe2::CPUContext>`

- *GPU* `caffe2::SpaceToBatchOp<caffe2::CUDAContext>`




---


# SparseLengthsSum

Pulls in slices of the input tensor, groups them into segments and applies 'Sum' to each segment. Segments are defined by their LENGTHS.
 This op is basically Gather and LengthsSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 LENGTHS is a vector that defines slice sizes by first dimention of DATA. Values belonging to the same segment are aggregated together. sum(LENGTHS) has to match INDICES size.
 The first dimension of the output is equal to the number of input segment, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor, slices of which are aggregated.
</td></tr><tr><td>`INDICES`
</td><td>Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
</td></tr><tr><td>`LENGTHS`
</td><td>Non negative vector with sum of elements equal to INDICES length
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of K (the number of segments).
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsOp<float, int, caffe2::CPUContext, caffe2::SumReducer<float, caffe2::CPUContext>, true>`




---


# SparseLengthsSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsGradientOp<float, int, caffe2::CPUContext, caffe2::SumReducerGradient<float, caffe2::CPUContext> >`




---


# SparseLengthsWeightedSum

Pulls in slices of the input tensor, groups them into segments and applies 'WeightedSum' to each segment. Segments are defined by their LENGTHS.
 This op is basically Gather and LengthsWeightedSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 LENGTHS is a vector that defines slice sizes by first dimention of DATA. Values belonging to the same segment are aggregated together. sum(LENGTHS) has to match INDICES size.
 The first dimension of the output is equal to the number of input segment, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor for the summation
</td></tr><tr><td>`SCALARS`
</td><td>Scalar multipliers for the input slices. Must be a vector with the length matching the first dimension of DATA
</td></tr><tr><td>`INDICES`
</td><td>Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
</td></tr><tr><td>`LENGTHS`
</td><td>Non negative vector with sum of elements equal to INDICES length
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of K (the number of segments).
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducer<float, caffe2::CPUContext>, true>`




---


# SparseLengthsWeightedSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractLengthsGradientOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducerGradient<float, caffe2::CPUContext> >`




---


# SparseSortedSegmentSum

Pulls in slices of the input tensor, groups them into segments and applies 'Sum' to each segment. Segments need to be sorted and contiguous. See also SparseUnsortedSegmentSum that doesn't have this requirement.
 This op is basically Gather and SortedSegmentSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor, slices of which are aggregated.
</td></tr><tr><td>`INDICES`
</td><td>Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as INDICES and values in the range 0..K-1 and in increasing order that maps each slice of DATA referenced by INDICES to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of K (the number of segments).
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentOp<float, int, caffe2::CPUContext, caffe2::SumReducer<float, caffe2::CPUContext>, true>`




---


# SparseSortedSegmentSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::SumReducerGradient<float, caffe2::CPUContext> >`




---


# SparseSortedSegmentWeightedSum

Pulls in slices of the input tensor, groups them into segments and applies 'WeightedSum' to each segment. Segments need to be sorted and contiguous. See also SparseUnsortedSegmentWeightedSum that doesn't have this requirement.
 This op is basically Gather and SortedSegmentWeightedSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor for the summation
</td></tr><tr><td>`SCALARS`
</td><td>Scalar multipliers for the input slices. Must be a vector with the length matching the first dimension of DATA
</td></tr><tr><td>`INDICES`
</td><td>Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Vector with the same length as INDICES and values in the range 0..K-1 and in increasing order that maps each slice of DATA referenced by INDICES to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of K (the number of segments).
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducer<float, caffe2::CPUContext>, true>`




---


# SparseSortedSegmentWeightedSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractSortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducerGradient<float, caffe2::CPUContext> >`




---


# SparseToDenseMask

Convert sparse representations to dense with given indices.
 Transforms a sparse representation of map<id, value> represented as  `indices`  vector and  `values`  tensor into a compacted tensor where the first dimension corresponds to each id provided in mask argument. Missing values are filled with the value of  `default_value` . After running this op:   ```  output[j, :] = values[i] # where mask[j] == indices[i] output[j, ...] = default_value # when mask[j] doesn't appear in indices  ```   If  `lengths`  is provided and not empty, and extra "batch" dimension is prepended to the output.
  `values`  and  `default_value`  can have additional matching dimensions, operation is performed on the entire subtensor in thise case.
 For example, if  `lengths`  is supplied and  `values`  is 1-D vector of floats and  `default_value`  is a float scalar, the output is going to be a float matrix of size  `len(lengths) X len(mask)`  
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`mask`
</td><td>list(int) argument with desired ids on the 'dense' output dimension
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`indices`
</td><td>1-D int32/int64 tensor of concatenated ids of data
</td></tr><tr><td>`values`
</td><td>Data tensor, first dimension has to match `indices`
</td></tr><tr><td>`default_value`
</td><td>Default value for the output if the id is not present in `indices`. Must have the same type as `values` and the same shape, but without the first dimension
</td></tr><tr><td>`lengths`
</td><td>Optional lengths to represent a batch of `indices` and `values`.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Output tensor of the same type as `values` of shape `[len(lengths), len(mask)] + shape(default_value)` (if `lengths` is not provided the first dimension is omitted)
</td></tr></table>
### Code
[caffe2/operators/sparse_to_dense_mask_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sparse_to_dense_mask_op.cc)
### Devices

- *CPU* `caffe2::SparseToDenseMaskOp`




---


# SparseUnsortedSegmentSum

Pulls in slices of the input tensor, groups them into segments and applies 'Sum' to each segment. Segments ids can appear in arbitrary order (unlike in SparseSortedSegmentSum).
 This op is basically Gather and UnsortedSegmentSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor, slices of which are aggregated.
</td></tr><tr><td>`INDICES`
</td><td>Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Integer vector with the same length as INDICES that maps each slice of DATA referenced by INDICES to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of equal to the number of segments.
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentOp<float, int, caffe2::CPUContext, caffe2::SumReducer<float, caffe2::CPUContext>, true>`




---


# SparseUnsortedSegmentSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::SumReducerGradient<float, caffe2::CPUContext> >`




---


# SparseUnsortedSegmentWeightedSum

Pulls in slices of the input tensor, groups them into segments and applies 'WeightedSum' to each segment. Segments ids can appear in arbitrary order (unlike in SparseSortedSegmentWeightedSum).
 This op is basically Gather and UnsortedSegmentWeightedSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor for the summation
</td></tr><tr><td>`SCALARS`
</td><td>Scalar multipliers for the input slices. Must be a vector with the length matching the first dimension of DATA
</td></tr><tr><td>`INDICES`
</td><td>Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Integer vector with the same length as INDICES that maps each slice of DATA referenced by INDICES to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of equal to the number of segments.
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducer<float, caffe2::CPUContext>, true>`




---


# SparseUnsortedSegmentWeightedSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducerGradient<float, caffe2::CPUContext> >`




---


# SpatialBN

Carries out spatial batch normalization as described in the paper  [https://arxiv.org/abs/1502.03167.](https://arxiv.org/abs/1502.03167.)  Depending on the mode it is being run, there are multiple cases for the number of outputs, which we list below:  Output case #1: Y, mean, var, saved_mean, saved_var  
````
                (training mode)

````
 Output case #2: Y (test mode) 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`is_test`
</td><td>If set to nonzero, run spatial batch normalization in test mode.
</td></tr><tr><td>`epsilon`
</td><td>The epsilon value to use to avoid division by zero.
</td></tr><tr><td>`order`
</td><td>A StorageOrder string.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>The input 4-dimensional tensor of shape NCHW or NHWC depending on the order parameter.
</td></tr><tr><td>`scale`
</td><td>The scale as a 1-dimensional tensor of size C to be applied to the output.
</td></tr><tr><td>`bias`
</td><td>The bias as a 1-dimensional tensor of size C to be applied to the output.
</td></tr><tr><td>`mean`
</td><td>The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size C.
</td></tr><tr><td>`var`
</td><td>The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size C.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>The output 4-dimensional tensor of the same shape as X.
</td></tr><tr><td>`mean`
</td><td>The running mean after the spatial BN operator. Must be in-place with the input mean. Should not be used for testing.
</td></tr><tr><td>`var`
</td><td>The running variance after the spatial BN operator. Must be in-place with the input var. Should not be used for testing.
</td></tr><tr><td>`saved_mean`
</td><td>Saved mean used during training to speed up gradient computation. Should not be used for testing.
</td></tr><tr><td>`saved_var`
</td><td>Saved variance used during training to speed up gradient computation. Should not be used for testing.
</td></tr></table>
### Code
[caffe2/operators/spatial_batch_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/spatial_batch_norm_op.cc)
### Devices

- *CPU* `caffe2::SpatialBNOp<caffe2::CPUContext>`

- *GPU* `caffe2::CudnnSpatialBNOp<float>`



### Engines
`CUDNN` on *CUDA*

---


# SpatialBNGradient
No documentation yet.

### Code
[caffe2/operators/spatial_batch_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/spatial_batch_norm_op.cc)
### Devices

- *CPU* `caffe2::SpatialBNGradientOp<caffe2::CPUContext>`

- *GPU* `caffe2::CudnnSpatialBNGradientOp<float>`



### Engines
`CUDNN` on *CUDA*

---


# Split
Split a tensor into a list of tensors.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`axis`
</td><td>Which axis to split on
</td></tr><tr><td>`order`
</td><td>Either NHWC or NCWH, will split on C axis
</td></tr></table>
### Code
[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)
### Devices

- *CPU* `caffe2::SplitOp<caffe2::CPUContext>`

- *GPU* `caffe2::SplitOp<caffe2::CUDAContext>`




---


# SquareRootDivide

Given DATA tensor with first dimention N and SCALE vector of the same size N produces an output tensor with same dimensions as DATA. Which consists of DATA slices. i-th slice is divided by sqrt(SCALE[i]) elementwise. If SCALE[i] == 0 output slice is identical to the input one (no scaling)  Example:   
````
  Data = [
    [1.0, 2.0],
    [3.0, 4.0]
  ]

  SCALE = [4, 9]

  OUTPUT = [
    [2.0, 4.0],
    [9.0, 12.0]
  ]


````

### Code
[caffe2/operators/square_root_divide_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/square_root_divide_op.cc)
### Devices

- *CPU* `caffe2::SquareRootDivideOp<int, caffe2::CPUContext>`




---


# SquaredL2Distance

 
````
  Given two input float tensors X, Y, and produces one output float tensor
  of the L2 difference between X and Y that is computed as ||(X - Y)^2 / 2||.

````
   
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>1D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>1D input tensor
</td></tr></table>
### Code
[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)
### Devices

- *CPU* `caffe2::SquaredL2DistanceOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SquaredL2DistanceOp<float, caffe2::CUDAContext>`




---


# SquaredL2DistanceGradient
No documentation yet.

### Code
[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)
### Devices

- *CPU* `caffe2::SquaredL2DistanceGradientOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SquaredL2DistanceGradientOp<float, caffe2::CUDAContext>`




---


# Squeeze

Remove single-dimensional entries from the shape of a tensor.
Takes a 
````
  parameter `dims` with a list of dimension to squeeze.

````
 If the same blob is provided in input and output, the operation is copy-free.
This is the exact inverse operation of ExpandDims given the same  `dims`  arg.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>Tensors with at least max(dims) dimensions.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`squeezed`
</td><td>Reshaped tensor with same data as input.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::SqueezeOp<caffe2::CPUContext>`

- *GPU* `caffe2::SqueezeOp<caffe2::CUDAContext>`




---


# StopGradient

StopGradient is a helper operator that does no actual numerical computation, and in the gradient computation phase stops the gradient from being computed through it.

### Code
[caffe2/operators/stop_gradient.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stop_gradient.cc)
### Devices

- *CPU* `caffe2::StopGradientOp<caffe2::CPUContext>`

- *GPU* `caffe2::StopGradientOp<caffe2::CUDAContext>`




---


# StringEndsWith

Performs the ends-with check on each string in the input tensor.
Returns tensor of boolean of the same dimension of input.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`suffix`
</td><td>The suffix to check input strings against.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`strings`
</td><td>Tensor of std::string.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`bools`
</td><td>Tensor of bools of same shape as input.
</td></tr></table>
### Code
[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >, caffe2::CPUContext, caffe2::ForEach<caffe2::(anonymous namespace)::EndsWith>, caffe2::FixedType<bool> >`




---


# StringIndexCreate

Creates a dictionary that maps string keys to consecutive integers from 1 to max_elements. Zero is reserved for unknown keys.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`max_elements`
</td><td>Max number of elements, including the zero entry.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handle`
</td><td>Pointer to an Index instance.
</td></tr></table>
### Code
[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)
### Devices

- *CPU* `caffe2::IndexCreateOp<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >`




---


# StringPrefix

Computes the element-wise string prefix of the string tensor.
Input strings that are shorter than prefix length will be returned unchanged.
NOTE: Prefix is computed on number of bytes, which may lead to wrong behavior and potentially invalid strings for variable-length encodings such as utf-8.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`length`
</td><td>Maximum size of the prefix, in bytes.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`strings`
</td><td>Tensor of std::string.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`prefixes`
</td><td>Tensor of std::string containing prefixes for each input.
</td></tr></table>
### Code
[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >, caffe2::CPUContext, caffe2::ForEach<caffe2::(anonymous namespace)::Prefix>, caffe2::FixedType<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > > >`




---


# StringStartsWith

Performs the starts-with check on each string in the input tensor.
Returns tensor of boolean of the same dimension of input.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`prefix`
</td><td>The prefix to check input strings against.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`strings`
</td><td>Tensor of std::string.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`bools`
</td><td>Tensor of bools of same shape as input.
</td></tr></table>
### Code
[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >, caffe2::CPUContext, caffe2::ForEach<caffe2::(anonymous namespace)::StartsWith>, caffe2::FixedType<bool> >`




---


# StringSuffix

Computes the element-wise string suffix of the string tensor.
Input strings that are shorter than suffix length will be returned unchanged.
NOTE: Prefix is computed on number of bytes, which may lead to wrong behavior and potentially invalid strings for variable-length encodings such as utf-8.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`length`
</td><td>Maximum size of the suffix, in bytes.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`strings`
</td><td>Tensor of std::string.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`suffixes`
</td><td>Tensor of std::string containing suffixes for each output.
</td></tr></table>
### Code
[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >, caffe2::CPUContext, caffe2::ForEach<caffe2::(anonymous namespace)::Suffix>, caffe2::FixedType<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > > >`




---


# Sub

Performs element-wise binary subtraction (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand, should share the type with the second operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and type as A
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CPUContext, caffe2::EigenSubFunctor, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<int, long, float, double>, caffe2::CUDAContext, caffe2::CudaSubFunctor, caffe2::SameTypeAsInput>`




---


# Sum

Element-wise sum of each of the input tensors. The first input tensor can be used in-place as the output tensor, in which case the sum will be done in place and results will be accumulated in input0. All inputs and outputs must have the same shape and data type.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data_0`
</td><td>First of the input tensors. Can be inplace.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`sum`
</td><td>Output tensor. Same dimension as inputs.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::SumOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SumOp<float, caffe2::CUDAContext>`




---


# SumInt
No documentation yet.

### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::SumOp<int, caffe2::CPUContext>`




---


# Summarize

Summarize computes four statistics of the input tensor (Tensor<float>)- min, max, mean and standard deviation. The output will be written to a 1-D tensor of size 4 if an output tensor is provided. Else, if the argument 'to_file' is greater than 0, the values are written to a log file in the root folder.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`to_file`
</td><td>(int, default 0) flag to indicate if the summarized statistics have to be written to a log file.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>The input data as Tensor<float>.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>1-D tensor (Tensor<float>) of size 4 containing min, max, mean and standard deviation
</td></tr></table>
### Code
[caffe2/operators/summarize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/summarize_op.cc)
### Devices

- *CPU* `caffe2::SummarizeOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::SummarizeOp<float, caffe2::CUDAContext>`




---


# TT

The TT-layer serves as a low-rank decomposition of a fully connected layer. The inputs are the same as to a fully connected layer, but the number of parameters are greatly reduced and forward computation time can be drastically reduced especially for layers with large weight matrices. The multiplication is computed as a product of the input vector with each of the cores that make up the TT layer. Given the input sizes (inp_sizes), output sizes(out_sizes), and the ranks of each of the cores (tt_ranks), the ith core will have size:   
````
    inp_sizes[i] * tt_ranks[i] * tt_ranks[i + 1] * out_sizes[i].


````
 The complexity of the computation is dictated by the sizes of inp_sizes, out_sizes, and tt_ranks, where there is the trade off between accuracy of the low-rank decomposition and the speed of the computation.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`inp_sizes`
</td><td>(int[]) Input sizes of cores. Indicates the input size of the individual cores; the size of the input vector X must match the product of the inp_sizes array.
</td></tr><tr><td>`out_sizes`
</td><td>(int[]) Output sizes of cores. Indicates the output size of the individual cores; the size of the output vector Y must match the product of the out_sizes array.
</td></tr><tr><td>`tt_ranks`
</td><td>(int[]) Ranks of cores. Indicates the ranks of the individual cores; lower rank means larger compression, faster computation but reduce accuracy.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`X`
</td><td>Input tensor from previous layer with size (M x K), where M is the batch size and K is the input size.
</td></tr><tr><td>`b`
</td><td>1D blob containing the bias vector
</td></tr><tr><td>`cores`
</td><td>1D blob containing each individual cores with sizes specified above.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`Y`
</td><td>Output tensor from previous layer with size (M x N), where M is the batch size and N is the output size.
</td></tr></table>
### Code
[caffe2/operators/tt_linear_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tt_linear_op.cc)
### Devices

- *CPU* `caffe2::TTLinearOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`




---


# Tanh

Calculates the hyperbolic tangent of the given input tensor element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`input`
</td><td>1-D input tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The hyperbolic tangent values of the input tensor computed element-wise
</td></tr></table>
### Code
[caffe2/operators/tanh_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tanh_op.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CPUContext, caffe2::WithDefaultConstructor<caffe2::TanhCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<float>, caffe2::CUDAContext, caffe2::WithDefaultConstructor<caffe2::TanhCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# TanhGradient
No documentation yet.

### Code
[caffe2/operators/tanh_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tanh_op.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<float>, caffe2::CPUContext, caffe2::WithoutBroadcast<caffe2::TanhGradientCPUFunctor>, caffe2::SameTypeAsInput>`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<float>, caffe2::CUDAContext, caffe2::WithoutBroadcast<caffe2::TanhGradientCUDAFunctor>, caffe2::SameTypeAsInput>`




---


# TensorProtosDBInput

TensorProtosDBInput is a simple input operator that basically reads things from a db where each key-value pair stores an index as key, and a TensorProtos object as value. These TensorProtos objects should have the same size, and they will be grouped into batches of the given size. The DB Reader is provided as input to the operator and it returns as many output tensors as the size of the TensorProtos object. Each output will simply be a tensor containing a batch of data with size specified by the 'batch_size' argument containing data from the corresponding index in the TensorProtos objects in the DB.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`batch_size`
</td><td>(int, default 0) the number of samples in a batch. The default value of 0 means that the operator will attempt to insert the entire data in a single output blob.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>A pre-initialized DB reader. Typically, this is obtained by calling CreateDB operator with a db_name and a db_type. The resulting output blob is a DB Reader tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>The output tensor in which the batches of data are returned. The number of output tensors is equal to the size of (number of TensorProto's in) the TensorProtos objects stored in the DB as values. Each output tensor will be of size specified by the 'batch_size' argument of the operator
</td></tr></table>
### Code
[caffe2/operators/tensor_protos_db_input.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tensor_protos_db_input.cc)
### Devices

- *CPU* `caffe2::TensorProtosDBInput<caffe2::CPUContext>`

- *GPU* `caffe2::TensorProtosDBInput<caffe2::CUDAContext>`




---


# TextFileReaderRead
Read a batch of rows from the given text file reader instance. Expects the number of fields to be equal to the number of outputs. Each output is a 1D tensor containing the values for the given field for each row. When end of file is reached, returns empty tensors.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`batch_size`
</td><td>Maximum number of rows to read.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>Pointer to an existing TextFileReaderInstance.
</td></tr></table>
### Code
[caffe2/operators/text_file_reader.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/text_file_reader.cc)
### Devices

- *CPU* `caffe2::TextFileReaderReadOp`




---


# Transpose

Transpose the input tensor similar to numpy.transpose. For example, when axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape will be (2, 1, 3).

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`axes`
</td><td>A list of integers. By default, reverse the dimensions, otherwise permute the axes according to the values given.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>An input tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`transposed`
</td><td>Transposed output.
</td></tr></table>
### Code
[caffe2/operators/transpose_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/transpose_op.cc)
### Devices

- *CPU* `caffe2::TransposeOp<caffe2::CPUContext>`

- *GPU* `caffe2::TransposeOp<caffe2::CUDAContext>`




---


# UniformFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::UniformFillOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::UniformFillOp<float, caffe2::CUDAContext>`




---


# UniformIntFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::UniformFillOp<int, caffe2::CPUContext>`

- *GPU* `caffe2::UniformFillOp<int, caffe2::CUDAContext>`




---


# Unique

Deduplicates input indices vector and optionally produces reverse remapping.
There's no guarantees on the ordering of the output indices.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`indices`
</td><td>1D tensor of int32 or int64 indices.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`unique_indices`
</td><td>1D tensor of deduped entries.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::UniqueOp<caffe2::CPUContext>`




---


# UnpackSegments
Map N+1 dim tensor to N dim based on length blob
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>1-d int/long tensor contains the length in each of the input.
</td></tr><tr><td>`tensor`
</td><td>N+1 dim Tensor.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`packed_tensor`
</td><td>N dim Tesor
</td></tr></table>
### Code
[caffe2/operators/pack_segments.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pack_segments.cc)
### Devices

- *CPU* `caffe2::(anonymous namespace)::UnpackSegmentsOp<caffe2::CPUContext>`




---


# UnsortedSegmentSum

Applies 'Sum' to each segment of input tensor. Segments ids can appear in arbitrary order (unlike in SortedSegmentSum).
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`num_segments`
</td><td>Optional int argument specifying the number of output segments and thus the first dimension of the output
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor, slices of which are aggregated.
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Integer vector with the same length as the first dimension of DATA that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of equal to the number of segments.
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentOp<float, int, caffe2::CPUContext, caffe2::SumReducer<float, caffe2::CPUContext>, false>`




---


# UnsortedSegmentSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::SumReducerGradient<float, caffe2::CPUContext> >`




---


# UnsortedSegmentWeightedSum

Applies 'WeightedSum' to each segment of input tensor. Segments ids can appear in arbitrary order (unlike in SortedSegmentWeightedSum).
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`num_segments`
</td><td>Optional int argument specifying the number of output segments and thus the first dimension of the output
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`DATA`
</td><td>Input tensor for the summation
</td></tr><tr><td>`SCALARS`
</td><td>Scalar multipliers for the input slices. Must be a vector with the length matching the first dimension of DATA
</td></tr><tr><td>`SEGMENT_IDS`
</td><td>Integer vector with the same length as the first dimension of DATA that maps each slice of DATA to one of the segments
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`OUTPUT`
</td><td>Aggregated output tensor. Has the first dimension of equal to the number of segments.
</td></tr></table>
### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducer<float, caffe2::CPUContext>, false>`




---


# UnsortedSegmentWeightedSumGradient
No documentation yet.

### Code
[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)
### Devices

- *CPU* `caffe2::AbstractUnsortedSegmentGradientOp<float, int, caffe2::CPUContext, caffe2::WeightedSumReducerGradient<float, caffe2::CPUContext> >`




---


# WallClockTime
Time since epoch in nanoseconds.
### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`time`
</td><td>The time in nanoseconds.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::WallClockTimeOp<caffe2::CPUContext>`




---


# WeightedSum

Element-wise weighted sum of several data, weight tensor pairs.
Input should be in the form X_0, weight_0, X_1, weight_1, ... where X_i all have the same shape, and weight_i are size 1 tensors that specifies the weight of each vector. Note that if one wants to do in-place computation, it could only be done with X_0 also as the output, but not other X_i.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`weight_0`
</td><td>Weight of the first input in the sum.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Result containing weighted elem-wise sum of inputs.
</td></tr></table>
### Code
[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)
### Devices

- *CPU* `caffe2::WeightedSumOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::WeightedSumOp<float, caffe2::CUDAContext>`




---


# XavierFill
No documentation yet.

### Code
[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)
### Devices

- *CPU* `caffe2::XavierFillOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::XavierFillOp<float, caffe2::CUDAContext>`




---


# Xor

Performs element-wise logical operation  `xor`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   
````
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0


````
 Argument  `broadcast=1`  needs to be passed to enable broadcasting.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`broadcast`
</td><td>Pass 1 to enable broadcasting
</td></tr><tr><td>`axis`
</td><td>If set, defines the broadcast dimensions. See doc for details.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>First operand.
</td></tr><tr><td>`B`
</td><td>Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>Result, has same dimensions and A and type `bool`
</td></tr></table>
### Code
[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)
### Devices

- *CPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<bool>, caffe2::CPUContext, caffe2::NaiveXorFunctor, caffe2::FixedType<bool> >`

- *GPU* `caffe2::BinaryElementwiseOp<caffe2::TensorTypes<bool>, caffe2::CUDAContext, caffe2::CudaXorFunctor, caffe2::FixedType<bool> >`




---


# Adagrad

 Computes the AdaGrad update for an input gradient and accumulated history. Concretely, given inputs (param, grad, history, learning_rate), computes   
````
    new_history = history + square(grad)
    new_grad = learning_rate * grad / (sqrt(new_history) + epsilon)
    new_param = param + new_grad

````
 and returns (new_param, new_history).
 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`epsilon`
</td><td>Default 1e-5
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`param`
</td><td>Parameters to be updated
</td></tr><tr><td>`moment`
</td><td>Moment history
</td></tr><tr><td>`grad`
</td><td>Gradient computed
</td></tr><tr><td>`lr`
</td><td>learning rate
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output_param`
</td><td>Updated parameters
</td></tr><tr><td>`output_moment`
</td><td>Updated moment
</td></tr></table>
### Code
[caffe2/sgd/adagrad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adagrad_op.cc)
### Devices

- *CPU* `caffe2::AdagradOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::AdagradOp<float, caffe2::CUDAContext>`



### Engines
`SIMD` on *CPU*

---


# Adam

 Computes the Adam update ( [https://arxiv.org/abs/1412.6980)](https://arxiv.org/abs/1412.6980))  for an input gradient and momentum parameters. Concretely, given inputs (param, m1, m2, grad, lr, iters),   
````
    t = iters + 1
    corrected_local_rate = lr * sqrt(1 - power(beta2, t)) /
      (1 - power(beta1, t))
    m1_o = (beta1 * m1) + (1 - beta1) * grad
    m2_o = (beta2 * m2) + (1 - beta2) * np.square(grad)
    grad_o = corrected_local_rate * m1_o / \
        (sqrt(m2_o) + epsilon)
    param_o = param + grad_o


````
 and returns (param_o, m1_o, m2_o)  
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`beta1`
</td><td>Default 0.9
</td></tr><tr><td>`beta2`
</td><td>Default 0.999
</td></tr><tr><td>`epsilon`
</td><td>Default 1e-5
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`param`
</td><td>Parameters to be updated
</td></tr><tr><td>`moment_1`
</td><td>First moment history
</td></tr><tr><td>`moment_2`
</td><td>Second moment history
</td></tr><tr><td>`grad`
</td><td>Gradient computed
</td></tr><tr><td>`lr`
</td><td>learning rate
</td></tr><tr><td>`iter`
</td><td>iteration number
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output_param`
</td><td>Updated parameters
</td></tr><tr><td>`output_moment_1`
</td><td>Updated first moment
</td></tr><tr><td>`output_moment_2`
</td><td>Updated second moment
</td></tr></table>
### Code
[caffe2/sgd/adam_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adam_op.cc)
### Devices

- *CPU* `caffe2::AdamOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::AdamOp<float, caffe2::CUDAContext>`




---


# AdsNNPreper
No documentation yet.

### Code
[caffe2/fb/data/AdsNNPreperOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/AdsNNPreperOp.cpp)
### Devices

- *CPU* `caffe2::fb::AdsNNPreperOp`




---


# AlignLabels
No documentation yet.

### Code
[caffe2/fb/text/ops/LabelOps.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/LabelOps.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::AlignLabelsOp`




---


# ArraySelect
No documentation yet.

### Code
[caffe2/fb/distribute/ops/ServiceDiscoveryOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ServiceDiscoveryOp.cpp)
### Devices

- *CPU* `caffe2::fb::ArraySelectOp`




---


# AtomicIter

Similar to Iter, but takes a mutex as the first input to make sure that updates are carried out atomically. This can be used in e.g. Hogwild sgd algorithms.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`mutex`
</td><td>The mutex used to do atomic increment.
</td></tr><tr><td>`iter`
</td><td>The iter counter as an int64_t TensorCPU.
</td></tr></table>
### Code
[caffe2/sgd/iter_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/iter_op.cc)
### Devices

- *CPU* `caffe2::AtomicIterOp<caffe2::CPUContext>`

- *GPU* `caffe2::AtomicIterOp<caffe2::CUDAContext>`




---


# CapitalizationType
No documentation yet.

### Code
[caffe2/fb/text/ops/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/string_ops.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >, caffe2::CPUContext, caffe2::ForEach<caffe2::fb::(anonymous namespace)::CapitalizationType>, caffe2::FixedType<int> >`




---


# CloseBlobsQueue
No documentation yet.

### Code
[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)
### Devices

- *CPU* `caffe2::CloseBlobsQueueOp<caffe2::CPUContext>`

- *GPU* `caffe2::CloseBlobsQueueOp<caffe2::CUDAContext>`




---


# ConcatStringArray
Merge string scalars to a 1D string tensor
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>first string scalar
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`array`
</td><td>1D string tensro with all of the strings
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::ConcatStringArrayOp`




---


# CreateBlobsQueue
No documentation yet.

### Code
[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)
### Devices

- *CPU* `caffe2::CreateBlobsQueueOp<caffe2::CPUContext>`

- *GPU* `caffe2::CreateBlobsQueueOp<caffe2::CUDAContext>`




---


# CreateDB
No documentation yet.

### Code
[caffe2/db/create_db_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/db/create_db_op.cc)
### Devices

- *CPU* `caffe2::CreateDBOp<caffe2::CPUContext>`

- *GPU* `caffe2::CreateDBOp<caffe2::CUDAContext>`




---


# CreateHiveReader
Create a hive reader instance.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`namespace`
</td><td>Required: Namespace of the Hive table to read from.
</td></tr><tr><td>`tablename`
</td><td>Required: Name of the table name to read from.
</td></tr><tr><td>`partitions`
</td><td>Required for partitioned tables: Partitions to read.
</td></tr><tr><td>`columns`
</td><td>(default: all cols) If set, read only the given columns.
</td></tr><tr><td>`num_passes`
</td><td>Number of passes on the data.
</td></tr><tr><td>`batch_size`
</td><td>Number of rows per batch.
</td></tr><tr><td>`shard_ids`
</td><td>If set along with num_shards, read only a subset of the data.
</td></tr><tr><td>`num_shards`
</td><td>How many shards to split the data into. Used along with shard_ids.
</td></tr><tr><td>`num_splits`
</td><td>(default: one split per file) Number of splits. Defines the number of splits to break down the data into. Each split has to be read serially, so in order to get better parallelism for smaller tables, it may be possible to get better speed by manually setting num_splits.
</td></tr><tr><td>`queue_size`
</td><td>Number of batches to keep in the reading queue.
</td></tr><tr><td>`num_threads`
</td><td>Number of parallel readers.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`split_queue`
</td><td>(Optional) Pointer to queue of splits to read.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>Pointer to the hive reader instance.
</td></tr></table>
### Code
[caffe2/fb/data/HiveReader.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveReader.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::CreateHiveReaderOp`




---


# CreateHiveWriter

Operator to create a Hive Writer Instance object. Takes namespace, tablename and parttion as input to write to.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`namespace`
</td><td>
</td></tr><tr><td>`tablename`
</td><td>
</td></tr><tr><td>`partition`
</td><td>
</td></tr><tr><td>`num_splits`
</td><td>1
</td></tr></table>
### Code
[caffe2/fb/data/HiveWriter.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveWriter.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::CreateHiveWriterOp`




---


# DequeueBlobs
No documentation yet.

### Code
[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)
### Devices

- *CPU* `caffe2::DequeueBlobsOp<caffe2::CPUContext>`

- *GPU* `caffe2::DequeueBlobsOp<caffe2::CUDAContext>`




---


# EmbeddingReader
No documentation yet.

### Code
[caffe2/fb/embnn/ops/EmbnnReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/embnn/ops/EmbnnReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::EmbeddingReaderOp`




---


# EmbnnReader
No documentation yet.

### Code
[caffe2/fb/embnn/ops/EmbnnReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/embnn/ops/EmbnnReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::EmbnnReaderOp`




---


# EmbnnReaderCreate
No documentation yet.

### Code
[caffe2/fb/embnn/ops/EmbnnReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/embnn/ops/EmbnnReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::EmbnnReaderCreateOp`




---


# EmbnnSingleReader
No documentation yet.

### Code
[caffe2/fb/embnn/ops/EmbnnReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/embnn/ops/EmbnnReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::EmbnnSingleReaderOp`




---


# EnqueueBlobs
No documentation yet.

### Code
[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)
### Devices

- *CPU* `caffe2::EnqueueBlobsOp<caffe2::CPUContext>`

- *GPU* `caffe2::EnqueueBlobsOp<caffe2::CUDAContext>`




---


# FoldCase
No documentation yet.

### Code
[caffe2/fb/text/ops/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/string_ops.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >, caffe2::CPUContext, caffe2::ForEach<caffe2::fb::(anonymous namespace)::UnicodeOp<caffe2::fb::(anonymous namespace)::FoldCase> >, caffe2::FixedType<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > > >`




---


# Ftrl
No documentation yet.

### Code
[caffe2/sgd/ftrl_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/ftrl_op.cc)
### Devices

- *CPU* `caffe2::FtrlOp<float, caffe2::CPUContext>`



### Engines
`SIMD` on *CPU*

---


# GetHiveTableColumns
Return a list of names for each output tensor returned by HiveReaderRead for the given hive table.
### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`blob_names`
</td><td>1D string tensor with names of each blob returned from HiveReaderRead.
</td></tr><tr><td>`blob_types`
</td><td>1D string tensor with type of each blob returned from HiveReaderRead.
</td></tr></table>
### Code
[caffe2/fb/data/HiveReader.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveReader.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::GetHiveTableColumnsOp`




---


# HashSharding

Generates shard id for ids.
 shard_id = hash(id) % shards 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`remove_dup`
</td><td>remove duplicate ids
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`id`
</td><td>tensor of ids
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`shard_id`
</td><td>shard id for each id
</td></tr><tr><td>`sizes`
</td><td>size of each shard
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::HashShardingOp`




---


# HiveReaderEnqueueSplits
Enqueue the list of splits for this hive reader. Requires the queue to be large enough to hold all splits; otherwise, fails.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`namespace`
</td><td>Required: Namespace of the Hive table to read from.
</td></tr><tr><td>`tablename`
</td><td>Required: Name of the table name to read from.
</td></tr><tr><td>`partitions`
</td><td>Required for partitioned tables: Partitions to read.
</td></tr><tr><td>`num_splits`
</td><td>See doc for CreateHiveReader.
</td></tr><tr><td>`close_queue`
</td><td>Whether to close the queue at the end.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`queue`
</td><td>Pointer to the queue.
</td></tr></table>
### Code
[caffe2/fb/data/HiveReader.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveReader.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::HiveReaderEnqueueSplitsOp`




---


# HiveReaderGetSplits
Return a tensor with list of splits for this hive reader.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`namespace`
</td><td>Required: Namespace of the Hive table to read from.
</td></tr><tr><td>`tablename`
</td><td>Required: Name of the table name to read from.
</td></tr><tr><td>`partitions`
</td><td>Required for partitioned tables: Partitions to read.
</td></tr><tr><td>`num_splits`
</td><td>See doc for CreateHiveReader.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`splits`
</td><td>1D tensor with list of splits to read.
</td></tr></table>
### Code
[caffe2/fb/data/HiveReader.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveReader.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::HiveReaderGetSplitsOp`




---


# HiveReaderRead
Read a batch from a HiveReader instance. Batches are read as columns returned as a list of 1D tensors depending on the hive table schema. Call GetHiveTableColumns operator to obtain the list of names for each of the output tensors for a given table. 
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>Pointer to the HiveReader instance.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`blobs`
</td><td>Blob references to the columns being read
</td></tr></table>
### Code
[caffe2/fb/data/HiveReader.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveReader.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::HiveReaderReadOp`




---


# HiveWriterCommit

Operator to commit the write to Hive. Takes Hive Writer instance as input.
    
### Code
[caffe2/fb/data/HiveWriter.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveWriter.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::HiveWriterCommitOp`




---


# HiveWriterWrite

Operator to write to Hive. Takes Hive Writer instance as input and arbitrary number of column tensors to write. Will write one row at a time.
    
### Code
[caffe2/fb/data/HiveWriter.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/HiveWriter.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::HiveWriterWriteOp`




---


# ImageInput
No documentation yet.

### Code
[caffe2/image/image_input_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/image/image_input_op.cc)
### Devices

- *CPU* `caffe2::ImageInputOp<caffe2::CPUContext>`

- *GPU* `caffe2::ImageInputOp<caffe2::CUDAContext>`




---


# IndexCount

Accumulate counting tensor given new indices input,  Input(0) is a blob containing all the past countings, Input(1) is a blob containing new incoming indices, Input(2) is mutex,  IndexCount is thread safe.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`countings`
</td><td>A blob containing a pointer to countings.
</td></tr><tr><td>`indices`
</td><td>A blob containing a pointer to the indices.
</td></tr><tr><td>`mutex`
</td><td>mutex to keep thread safe.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`new_countings`
</td><td>Tensor containing a pointer to new countings.
</td></tr></table>
### Code
[caffe2/fb/text/ops/indexcount_ops.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/indexcount_ops.cpp)
### Devices

- *CPU* `caffe2::(anonymous namespace)::IndexCountOp`




---


# IndexTopK

Pad empty field given lengths and features,  Input(0) is a blob pointing to the countings,  
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`topK`
</td><td>Number of indices we are going to keep.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`countings`
</td><td>A blob containing a pointer to the countings.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`out_indices`
</td><td>Tensor containing the indices of topK countings.
</td></tr></table>
### Code
[caffe2/fb/text/ops/indexcount_ops.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/indexcount_ops.cpp)
### Devices

- *CPU* `caffe2::(anonymous namespace)::IndexTopKOp`




---


# IntShift
No documentation yet.

### Code
[caffe2/fb/embnn/ops/EmbnnReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/embnn/ops/EmbnnReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::IntShiftOp`




---


# Iter

Stores a singe integer, that gets incremented on each call to Run().
Useful for tracking the iteration count during SGD, for example.

### Code
[caffe2/sgd/iter_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/iter_op.cc)
### Devices

- *CPU* `caffe2::IterOp<caffe2::CPUContext>`

- *GPU* `caffe2::IterOp<caffe2::CUDAContext>`




---


# KVAtomicValue

If the key is not set, server will init it with 0. Then it will add add_value and return the current value. The operation is atomic. It can be used for simple task distribution. But be aware, it will not return 0.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`add_value`
</td><td>the integer value will be added
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`value`
</td><td>the current value of that counter
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVAtomicValueOp`




---


# KVCheckSignal

Check whether is the key set in the server or not. This op will not block the caller.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`status`
</td><td>return true if the key is set
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVCheckSignalOp`




---


# KVGetBlob

Get a blob value from the KVServer. The caller will be blocked until that blob is set. KVCheckSignal could be used for non-blocking check.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>content of that blob
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVGetBlobOp`




---


# KVHandlerClose

Close the KVHandler. This will unblock any local threads that is blocked by the handler.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>the same as input
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVHandlerCloseOp`




---


# KVHandlerCreate

Create a unique_ptr<StoreHandler>.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`server_id`
</td><td>The ID of this machine. Server will use the ID to find handler address to callback.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server_address`
</td><td>The socket address from SimpleKVServer
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>`address`
</td><td>The socket address of this handler
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVHandlerCreateOp`




---


# KVServerClose

Close the KVServer.
This is not required since the destructor of SimpleKVServer will close it too.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server_in`
</td><td>unique_ptr<SimpleKVServer>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`server_out`
</td><td>the same as input
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVServerCloseOp`




---


# KVServerCreate
Create a unique_ptr<SimpleKVServer>
### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<SimpleKVServer>
</td></tr><tr><td>`address`
</td><td>Socket address for the server. Client should use the address to connect to the server.
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVServerCreateOp`




---


# KVServerInit

Pass all of the handler addresses from the client side to the server to initialize the communication channel. The server is ready to be used after this op finishes.
 Input(0) is the unique pointer of SimpleKVServer.
[Input(1),...,Input(num_machines)] is all of the handler addresses from the client side.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<SimpleKVServer>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVServerInitOp`




---


# KVServerReset

Reset KV server, removing all blobs and counters, and sending signal to all handlers to reset themselves.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<SimpleKVServer>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVServerResetOp`




---


# KVSetBlob

Pass a blob to KVServer. The key is blob's name, and the value is the data in that blob.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>`data`
</td><td>data blob
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVSetBlobOp`




---


# KVSetSignal

This is similar to KVSetBlobOp. Since it is just a signal, the content will be empty. A real blob is not required.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`blob_name`
</td><td>key for the signal
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVSetSignalOp`




---


# KVWaitSignal

Check whether is the key set in the server. Block the caller until it is set.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>`names_blob`
</td><td>(optional) Names of blobs to wait for.
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/KVStoreOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/KVStoreOp.cpp)
### Devices

- *CPU* `caffe2::fb::KVWaitSignalOp`




---


# LearningRate
No documentation yet.

### Code
[caffe2/sgd/learning_rate_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/learning_rate_op.cc)
### Devices

- *CPU* `caffe2::LearningRateOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::LearningRateOp<float, caffe2::CUDAContext>`




---


# LogScoreEstimator

After a LogScoreRatioEstimator object is created using LogScoreEstimatorCreate, this operators updates each of the metrics. Given the estimator, the current prediction and the label, various metrics are computed at a frequency dictated by logging_frequency. No outputs are produced, as the LogScoreRatioEstimator object itself is updated. This operator will also do basic defensive checks to ensure that the labels and predictions have the same sizes. An example usage:   
````
      net.LogScoreEstimator([est, loss.predictions, loss.label], [])


````
 where the loss object's predictions and labels are passed to the operator.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`logging_frequency`
</td><td>Indicates the frequency with which the metrics are updated; optional parameter.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`ESTIMATOR`
</td><td>Input Blob containing the LogScoreRatioEstimator object, to be modified and updated as each prediction and label pair's metrics are computed.
</td></tr><tr><td>`PREDICTION`
</td><td>Input Blob containing the predictions, or outputs, of the model.
</td></tr><tr><td>`LABEL`
</td><td>Input Blob containing the labels.
</td></tr></table>
### Code
[caffe2/fb/metrics/LogScoreEstimatorOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/LogScoreEstimatorOp.cpp)
### Devices

- *CPU* `caffe2::fb::MetricsEstimatorOp<caffe2::fb::LogScoreEstimator, int>`




---


# LogScoreEstimatorCreate

The LogScoreEstimator set of operators (LogScoreEstimatorCreate, LogScoreEstimator, LogScoreEstimatorReport) is used to monitor important metrics during training of various Caffe2 models, such as MLP and sparse MLPs.
This operator is the starting point; no inputs are needed explicitly, and the output is the pointer to the LogScoreEstimator object. An example of the usage is as follows:   
````
    est = net.LogScoreEstimatorCreate([], ['stats'], name='stats', lastN=10)


````
 where the name of the blobs are provided. Use the LogScoreEstimator operator " "to update each of the individual metrics, and use the LogScoreEstimatorReport " "operator to output and print these results for viewing or graphing.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`lastN`
</td><td>LogScoreEstimator computes aggregate statistics over a sliding window and lastN dictates how many of data points are aggregated in computing the metrics. Set lastN to 0 disable this functionality; note that setting lastN to a nonzero value can make the operator much slower.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`est`
</td><td>Returns the LogScoreRatioEstimator object, which is the wrapper for the corresponding C++ object
</td></tr></table>
### Code
[caffe2/fb/metrics/LogScoreEstimatorOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/LogScoreEstimatorOp.cpp)
### Devices

- *CPU* `caffe2::fb::MetricsEstimatorCreateOp<caffe2::fb::LogScoreEstimator>`




---


# LogScoreEstimatorReport

This operator is used to report and output the updated metrics after the LogScoreEstimator operator is used. More explicitly, the JSON stats are computed and reported. Two output blobs are computed containing the various metrics computed, and these blobs can be interrogated for individual statistics. An example usage is as such:   
````
              lifetime_stats_blob, lastn_stats_blob = \
                  net.LogScoreEstimatorReport(
                      [est],
                      ['lifetime_stats', 'lastn_stats'],
                      logging_frequency=1)


````
 Here, the logging frequency is again passed as an argument to the operator.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`axis`
</td><td>
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`ESTIMATOR`
</td><td>Input Blob containing an updated 
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`lifetime_stats_blob`
</td><td>An output Blob that contains the metrics for the lifetime of statistics.
</td></tr><tr><td>`lastn_stats_blob`
</td><td>An output Blob that contains the metrics for the last n predictions.
</td></tr></table>
### Code
[caffe2/fb/metrics/LogScoreEstimatorOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/LogScoreEstimatorOp.cpp)
### Devices

- *CPU* `caffe2::fb::MetricsEstimatorReportOp<caffe2::fb::LogScoreEstimator>`




---


# MSEEstimator

After a MSEEstimator object is created using MSEEstimatorCreate, this operators updates each of the metrics. Given the estimator, the current prediction and the label, various metrics are computed at a frequency dictated by logging_frequency. No outputs are produced, as the MSEEstimator object itself is updated. This operator will also do basic defensive checks to ensure that the labels and predictions have the same sizes. An example usage:   
````
      net.MSEEstimator([est, loss.predictions, loss.label], [])


````
 where the loss object's predictions and labels are passed to the operator.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`logging_frequency`
</td><td>Indicates the frequency with which the metrics are updated; optional parameter.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`ESTIMATOR`
</td><td>Input Blob containing the MSEEstimator object, to be modified and updated as each prediction and label pair's metrics are computed.
</td></tr><tr><td>`PREDICTION`
</td><td>Input Blob containing the predictions, or outputs, of the model.
</td></tr><tr><td>`LABEL`
</td><td>Input Blob containing the labels.
</td></tr><tr><td>`WEIGHTS`
</td><td>Optional Input Blob containing the example weights.
</td></tr></table>
### Code
[caffe2/fb/metrics/MSEEstimatorOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/MSEEstimatorOp.cpp)
### Devices

- *CPU* `caffe2::fb::MetricsEstimatorOp<caffe2::fb::MSEEstimator, float>`




---


# MSEEstimatorCreate

The MSEEstimator set of operators (MSEEstimatorCreate, MSEEstimator, MSEEstimatorReport) is used to monitor important metrics during training of various Caffe2 models, such as MLP and sparse MLPs.
This operator is the starting point; no inputs are needed explicitly, and the output is the pointer to the MSEEstimator object. An example of the usage is as follows:   
````
    est = net.MSEEstimatorCreate([], ['stats'], name='stats')


````
 where the name of the blobs are provided. Use the MSEEstimator operator to update each of the individual metrics, and use the MSEEstimatorReport operator to output and print these results for viewing or graphing.

### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`est`
</td><td>Returns the MSEEstimator object, which is the wrapper for the corresponding C++ object
</td></tr></table>
### Code
[caffe2/fb/metrics/MSEEstimatorOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/MSEEstimatorOp.cpp)
### Devices

- *CPU* `caffe2::fb::MetricsEstimatorCreateOp<caffe2::fb::MSEEstimator>`




---


# MSEEstimatorReport

This operator is used to report and output the updated metrics after the MSEEstimator operator is used. More explicitly, the JSON stats are computed and reported. One output blob is computed containing the various metrics computed, and this blob can be interrogated for individual statistics. An example usage is as such:   
````
              lifetime_stats_blob = \
                  net.MSEEstimatorReport(
                      [est],
                      ['lifetime_stats'],
                      logging_frequency=1)


````
 Here, the logging frequency is again passed as an argument to the operator.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`axis`
</td><td>
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`ESTIMATOR`
</td><td>Input Blob containing an updated 
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`lifetime_stats_blob`
</td><td>An output Blob that contains the metrics for the lifetime of statistics.
</td></tr></table>
### Code
[caffe2/fb/metrics/MSEEstimatorOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/MSEEstimatorOp.cpp)
### Devices

- *CPU* `caffe2::fb::MetricsEstimatorReportOp<caffe2::fb::MSEEstimator>`




---


# MomentumSGD

 Computes a momentum SGD update for an input gradient and momentum parameters. Concretely, given inputs (grad, m, lr) and parameters (momentum, nesterov), computes:   
````
    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        return (adjusted_gradient, adjusted_gradient)
    else:
        m_new = momentum * m + lr * grad
        return ((1 + momentum) * m_new - momentum * m, m_new)



````

### Code
[caffe2/sgd/momentum_sgd_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/momentum_sgd_op.cc)
### Devices

- *CPU* `caffe2::MomentumSGDOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::MomentumSGDOp<float, caffe2::CUDAContext>`




---


# MultiLabelMetric

Generate accuracy and confusion matrix for multi label classification problem.

### Code
[caffe2/fb/metrics/MetricsOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/MetricsOp.cpp)
### Devices

- *CPU* `caffe2::fb::MultiLabelMetricOp`




---


# NNLoaderCreate
No documentation yet.

### Code
[caffe2/fb/data/NNLoaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/NNLoaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::NNLoaderCreateOp<caffe2::CPUContext>`

- *GPU* `caffe2::fb::NNLoaderCreateOp<caffe2::CUDAContext>`




---


# NNLoaderHasNoData
No documentation yet.

### Code
[caffe2/fb/data/NNLoaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/NNLoaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::NNLoaderHasNoDataOp<caffe2::CPUContext>`

- *GPU* `caffe2::fb::NNLoaderHasNoDataOp<caffe2::CUDAContext>`




---


# NNLoaderRead
No documentation yet.

### Code
[caffe2/fb/data/NNLoaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/NNLoaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::NNLoaderReadOp`

- *GPU* `caffe2::GPUFallbackOp<caffe2::fb::NNLoaderReadOp, caffe2::SkipIndices<> >`




---


# PSClientCreate
Create PS Client
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`client_id`
</td><td>server send responses based on the client id, so it must be unique
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`context`
</td><td>zmq context which will be used by PSClient
</td></tr><tr><td>`address_in`
</td><td>input addresses from all servers
</td></tr><tr><td>`address_out`
</td><td>output addresses from all servers
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSClientCreateOp`




---


# PSClientSend
Send request to parameter server
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`server_id`
</td><td>target PS server id
</td></tr><tr><td>`param_ids`
</td><td>list of param ids
</td></tr><tr><td>`shard_ids`
</td><td>list of shard ids
</td></tr><tr><td>`request_type`
</td><td>request type
</td></tr><tr><td>`use_tracker`
</td><td>whether to use tracker to track request
</td></tr><tr><td>`freq`
</td><td>frequency of actually sending data to avoid using executiong step
</td></tr><tr><td>`skip_empty`
</td><td>send empty request or not
</td></tr><tr><td>`compression_codec`
</td><td>compression codec for data
</td></tr><tr><td>`compression_level`
</td><td>compression level for data
</td></tr><tr><td>`reply_in_one`
</td><td>server replies all shards together or not
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`client`
</td><td>unique_ptr<PSClient>
</td></tr><tr><td>`tracker_group`
</td><td>shared_ptr<TrackerGroup>
</td></tr><tr><td>`tracker_version`
</td><td>version of the tracker
</td></tr><tr><td>`data`
</td><td>first data blob
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSClientSendOp`




---


# PSServerAddHandler
Add a handler to the server for (param, shard)
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`param_id`
</td><td>parameter id
</td></tr><tr><td>`shard_id`
</td><td>shard id for the parameter
</td></tr><tr><td>`buffer_size`
</td><td>size of the queue
</td></tr><tr><td>`support_types`
</td><td>request types defined in ps.thrift::RequestType
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<PSServer>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>shared_ptr<PSTaskHandler>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSServerAddHandlerOp`




---


# PSServerCreate
Create PS server
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`context`
</td><td>unique pointer of zmq::context_t
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<PSServer>
</td></tr><tr><td>`address_in`
</td><td>socket address for input messages
</td></tr><tr><td>`address_out`
</td><td>socket address for output messages
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSServerCreateOp`




---


# PSServerDequeueTask

Dequeue a task from the queue for the handler.
 Output(0) is task pointer.
Output(1) is the overall stop condition.
The rest of the outputs are booleans representing whether the handler for certain request type should be triggered. The index is consistent with the support_types passed by AddHandler.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`handler`
</td><td>shared_ptr<PSTaskHandler>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`task`
</td><td>unique_ptr<Task>
</td></tr><tr><td>`queue_status`
</td><td>whether queue is empty and closed
</td></tr><tr><td>`request_type_status`
</td><td>whether the task is set as certain request type
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSServerDequeueTaskOp`




---


# PSServerExpandTask
Extract input data blobs from task
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<PSServer>
</td></tr><tr><td>`task`
</td><td>unique_ptr<PSTask>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>first data blob
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSServerExpandTaskOp`




---


# PSServerReplyTask

Respond to the request with data blobs.
If the request is set with ReplyInOne, the op will not reply immediately but just decrease the counter unless it is the last one handling the request.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`compression_codec`
</td><td>compression codec for response
</td></tr><tr><td>`compression_level`
</td><td>compression level for response
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<PSServer>
</td></tr><tr><td>`task`
</td><td>unique_ptr<PSTask>
</td></tr><tr><td>`data`
</td><td>first data blob
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSServerReplyTaskOp`




---


# PSServerStart
Start PS Server. Need to add handlers before starting server.
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<PSServer>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSServerStartOp`




---


# PSServerStop
Stop PS Server
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`server`
</td><td>unique_ptr<PSServer>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSServerStopOp`




---


# PSTrackerCreate
Generate a new version from tracker group
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tracker_group`
</td><td>shared_ptr<TrackerGroup>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`tracker_version`
</td><td>a scalar represents tracker version
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSTrackerCreateOp`




---


# PSTrackerExpand
Extract data blobs from tracker(Response)
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tracker_group`
</td><td>shared_ptr<TrackerGroup>
</td></tr><tr><td>`tracker`
</td><td>shared_ptr<Tracker>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>first data blob
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSTrackerExpandOp`




---


# PSTrackerGroupCreate

Create a TrackerGroup which represents the schema for requests.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`param_ids`
</td><td>parameter ids
</td></tr><tr><td>`shard_ids`
</td><td>shard ids
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`client`
</td><td>unique_ptr<PSClient>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`tracker_group`
</td><td>shared_ptr<TrackerGroup>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSTrackerGroupCreateOp`




---


# PSTrackerGroupExtractLatest
Extract latest tracker and remove all older ones
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tracker_group`
</td><td>shared_ptr<TrackerGroup>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`tracker`
</td><td>shared_ptr<Tracker>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSTrackerGroupExtractLatestOp`




---


# PSTrackerGroupExtractVersion
Wait and return tracker for given version
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tracker_group`
</td><td>shared_ptr<TrackerGroup>
</td></tr><tr><td>`tracker_version`
</td><td>version for the tracker
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`None`
</td><td>
</td></tr><tr><td>`tracker`
</td><td>shared_ptr<Tracker>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::PSTrackerGroupExtractVersionOp`




---


# PackedFC

Computes the result of passing an input vector X into a fully connected layer with 2D weight matrix W and 1D bias vector b. This is essentially the same as the FC operator but allows one to pack the weight matrix for more efficient inference. See the schema for the FC op for details.
 Unlike many other operators in Caffe2, this operator is stateful: it assumes that the input weight matrix W never changes, so it is only suitable for inference time when the weight matrix never gets updated by any other ops.
Due to performance considerations, this is not checked in non-debug builds.

### Code
[caffe2/mkl/operators/packed_fc_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/mkl/operators/packed_fc_op.cc)
### Devices

- *CPU* `caffe2::mkl::PackedFCOp`




---


# ParseLabels
No documentation yet.

### Code
[caffe2/fb/text/ops/LabelOps.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/LabelOps.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::ParseLabelsOp`




---


# Python
No documentation yet.

### Code
[caffe2/python/pybind_state.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc)
### Devices

- *CPU* `caffe2::python::PythonOp`

- *GPU* `caffe2::GPUFallbackOp<caffe2::python::PythonOp, caffe2::SkipIndices<> >`




---


# PythonGradient
No documentation yet.

### Code
[caffe2/python/pybind_state.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc)
### Devices

- *CPU* `caffe2::python::PythonGradientOp`

- *GPU* `caffe2::GPUFallbackOp<caffe2::python::PythonGradientOp, caffe2::SkipIndices<> >`




---


# ResultWriter

Generate accuracy and confusion matrix for single label classification problem.

### Code
[caffe2/fb/embnn/ops/ResultWriterOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/embnn/ops/ResultWriterOp.cpp)
### Devices

- *CPU* `caffe2::fb::ResultWriterOp`




---


# RmsProp

 Computes the RMSProp update ( [http://www.cs.toronto.edu/](http://www.cs.toronto.edu/) ~tijmen/csc321/slides/lecture_slides_lec6.pdf).
Concretely, given inputs (grad, mean_squares, mom, lr), computes:   
````
    mean_squares_o = mean_squares + (1 - decay) * (squaare(grad) - mean_squares)
    mom_o = momentum * mom + lr * grad / sqrt(epsilon + mean_squares_o)
    grad_o = mom_o


````
 returns (grad_o, mean_squares_o, mom_o).
 
### Code
[caffe2/sgd/rmsprop_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/rmsprop_op.cc)
### Devices

- *CPU* `caffe2::RmsPropOp<float, caffe2::CPUContext>`

- *GPU* `caffe2::RmsPropOp<float, caffe2::CUDAContext>`




---


# SafeDequeueBlobs

Dequeue the blobs from queue. When the queue is closed and empty, the output status will be set to true which can be used as exit criteria for execution step.
The 1st input is the queue and the last output is the status. The rest are data blobs.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`queue`
</td><td>The shared pointer for the BlobsQueue
</td></tr></table>
### Code
[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)
### Devices

- *CPU* `caffe2::SafeDequeueBlobsOp<caffe2::CPUContext>`




---


# SafeEnqueueBlobs

Enqueue the blobs into queue. When the queue is closed and full, the output status will be set to true which can be used as exit criteria for execution step.
The 1st input is the queue and the last output is the status. The rest are data blobs.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`queue`
</td><td>The shared pointer for the BlobsQueue
</td></tr></table>
### Code
[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)
### Devices

- *CPU* `caffe2::SafeEnqueueBlobsOp<caffe2::CPUContext>`




---


# ServiceDiscovery

Create a global barrier for all of the machines and exchange addresses. Every caller needs to pass a address in and it will give back all of the addresses.
It is assuming that the number of machines is known ahead.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`shard_id`
</td><td>The ID for the caller. It should be in the range of[0, total_shards)
</td></tr><tr><td>`total_shards`
</td><td>Total number of machines will participate.
</td></tr><tr><td>`identity`
</td><td>The ID of the task, it usually is flow task id.
</td></tr><tr><td>`timestamp`
</td><td>The pair of identity and timestamp will be used for distinguishing different tasks. It was designed for task retries, but in fact, the retires will use the same timestamp. This is more useful for distinguish different local test task right now.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`address_in`
</td><td>one single string for the address of caller
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`svc_dsc_holder`
</td><td>Holder keeping service discovery alive.
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ServiceDiscoveryOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ServiceDiscoveryOp.cpp)
### Devices

- *CPU* `caffe2::fb::ServiceDiscoveryOp`




---


# ShardingSplit
Split a tensor based on the given shard ids
### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data`
</td><td>data blob
</td></tr><tr><td>`shard_id`
</td><td>shard ids generated by HashSharding
</td></tr><tr><td>`sizes`
</td><td>size of each shard generated by HashSharding
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>output for the first shard
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/PSOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/PSOp.cpp)
### Devices

- *CPU* `caffe2::fb::ShardingSplitOp`




---


# SigridTransformedCreate
No documentation yet.

### Code
[caffe2/fb/data/SigridLoaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/SigridLoaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::SigridTransformedCreateOp`




---


# SigridTransformedRead
No documentation yet.

### Code
[caffe2/fb/data/SigridLoaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/data/SigridLoaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::SigridTransformedReadOp`




---


# SigridTransforms

Invokes Sigrid Transformation layer given the inputs and returns outputs.
 This op operates on batches of examples and uses standard way of encoding Sigrid features as a bunch of Caffe2 tensors described below.
 Because the op doesn't have any information about feature types necessary information needs to be provided via  `input_spec`  JSON argument.
 `input_spec`  must be an array of objects where each object describes a feature block. Feature block must have  `type`  field describing sigrid's data type,  `name` / `names`  argument listing feature names and optional  `cardinality` / `cardinalities`  argument providing features' max value. Example:   ```  [  
````
  {"type": "FLOAT", "name": "A"},
  {"type": "INT", "names": ["B", "C"]},
  {"type": "ID_LIST", "names": ["D", "E"], "cardinality": 10},
  {"type": "ID_SCORE_LIST", "names": ["F", "G"], "cardinalities": [100, 200]},

````
 ]  ```   WARNING:  `cardinalities`  argument is easy to mis-use as it doesn't enforce data correctness. It's not recommended for direct usage and is intended for the occassion when the list of transforms needs to be split between two different SigridTransformOps. Please use Sigrid's MapTransform or IdentityTransform instead.
  ** Beforehand initialization **   Above argument can be big in size (transforms config) thus if the same config is shared across multiple ops it might be useful to create the transformation layer only once to save on both NetDef size and runtime memory.
 SigridTransformsCreate op can be used for this purpose. It takes the same arguments as described above and produces a single Blob containing "transforms instance". That instance can be passed as the very first arguments to SigridTransforms op preceding all of the input tensors.
  ** Simple output format **   Some of the results from the transformation can be returned from the operator.
Desired features, organized in groups are specified in  `output_spec`  argument.
Features in each group must be of the same data type. For example:   ```  [  
````
  "A",
  ["D", "E"],
  ["G"],

````
 ]  ```    ** Binarized output format **   Alternative output format is "binarized features" suitable for passing to a single lookup table (e.g. for LR implementation).
 Instead of or in addition to  `output_spec` , a list-of-strings argument  `output_binarized_features`  can be provided. All listed features must have finite cardinality so that they can be compacted into a single range.
 The compaction procedure is similar to BaseTransformationProcessor's in Sigrid: ranges of all features are concatenated together and the list of (index, weight) is returned.
 If  `output_binarized_features`  is set, the first 3 outputs would be filled as:   
````
  # 0: `indices` - int64 vector of indices of non-zero elements in concatenated

````
   
````
                  features ranges. Lists of indices are concatenated for the

````
   
````
                  entire input batch
  # 1: `weights` - float vector of values of non-zero elements in concatenated

````
   
````
                  features ranges. Has the same lengths as `indices`
  # 2: `lengths` - int32 vector denoting number of non-zero elements in each

````
   
````
                  example. `sum(lengths) = len(indices) = len(weights)`)


````
 `BATCH` ** Features encoding format **   Only 4 basic Sigrid types supported for now. Several features of the same type for a batch of Sigrid examples can be represented as 1-3 tensors in Caffe2. A batch of   examples with several features of a single type is represented as follows depending on the type:  - FLOAT: a float matrix of shape BATCH x NUM_FEATURES - INT: an int64 matrix of shape BATCH x NUM_FEATURES - ID_LIST: two tensors in the following order:   
````
  -- int32 tensor `ranges` of shape BATCH x NUM_FEATURES x 2 that represents
      for each feature offset in the second tensor and length of the list.

````
   
````
  -- int64 vector `data` of length `sum(len(list_i_j))` with concatenation
      of all value lists

````
 - ID_SCORE_LIST: three tensors in the following order:   
````
  -- int32 tensor `ranges` of shape BATCH x NUM_FEATURES x 2 that represents
      for each feature offset in the second tensor and length of the list.

````
   
````
  -- int64 vector `ids` of length `sum(len(list_i_j))` with concatenation
      of all keys from value lists

````
   
````
  -- float vector `scores` of length `sum(len(list_i_j))` with concatenation
      of all scores from value lists


````
 More docs on representation:  [https://our.intern.facebook.com/intern/dex/caffe2-operators/](https://our.intern.facebook.com/intern/dex/caffe2-operators/) #sparse-operations  
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`input_spec`
</td><td>string with JSON-serialized array describing feature space and feature blocks
</td></tr><tr><td>`transforms_config`
</td><td>JSON-serialized Thrift struct sigrid::TransformsConfig
</td></tr><tr><td>`output_spec`
</td><td>string with JSON array or array representing desired output feature blocks.
</td></tr><tr><td>`output_binarized_features`
</td><td>list of strings with the names of features to be returned in binarized format (aka Sigrid trainer format)
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`instance (optional)`
</td><td>Initialized SigridTransforms blob created beforehand by SigridTransformsCreate. If provided the op shouldn't have any of the arguments like `transforms_config` - they should be passed to SigridTransformsCreate instead
</td></tr></table>
### Code
[caffe2/fb/transforms/SigridTransformsOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/transforms/SigridTransformsOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::SigridTransformsOp`




---


# SigridTransformsCreate
Allows to pre-initialize transforms to save memory. Receives identical arguments to SigridTransforms operator, see its docs for details
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`input_spec`
</td><td>see SigridTransforms docs
</td></tr><tr><td>`transforms_config`
</td><td>see SigridTransforms docs
</td></tr><tr><td>`output_spec`
</td><td>see SigridTransforms docs
</td></tr><tr><td>`output_binarized_features`
</td><td>see SigridTransforms docs
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`instance`
</td><td>Initialized SigridTransforms blob that can be passed to SigridTransforms op as the first input.
</td></tr></table>
### Code
[caffe2/fb/transforms/SigridTransformsOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/transforms/SigridTransformsOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::SigridTransformsCreateOp`




---


# SingleLabelMetric

Generate accuracy and confusion matrix for single label classification problem.

### Code
[caffe2/fb/metrics/MetricsOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/metrics/MetricsOp.cpp)
### Devices

- *CPU* `caffe2::fb::SingleLabelMetricOp`




---


# SparseAdagrad

 Given inputs (param, history, indices, grad, lr), runs the dense AdaGrad update on (param, grad, history[indices], lr), and returns (new_param, new_history) as in the dense case.
 
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`epsilon`
</td><td>Default 1e-5
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`param`
</td><td>Parameters to be updated
</td></tr><tr><td>`moment`
</td><td>Moment history
</td></tr><tr><td>`indices`
</td><td>Sparse indices
</td></tr><tr><td>`grad`
</td><td>Gradient computed
</td></tr><tr><td>`lr`
</td><td>learning rate
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output_param`
</td><td>Updated parameters
</td></tr><tr><td>`output_moment_1`
</td><td>Updated moment
</td></tr></table>
### Code
[caffe2/sgd/adagrad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adagrad_op.cc)
### Devices

- *CPU* `caffe2::SparseAdagradOp<float, caffe2::CPUContext>`



### Engines
`SIMD` on *CPU*

---


# SparseAdam

 Computes the Adam Update for the sparse case.
Given inputs (param, moment1, moment2, indices, grad, lr, iter), runs the dense Adam on on (param, moment1[indices], momemnt2[indices], lr, iter) and returns (new_param, new_moment1, new_moment2) as in dense case  
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`beta1`
</td><td>Default 0.9
</td></tr><tr><td>`beta2`
</td><td>Default 0.999
</td></tr><tr><td>`epsilon`
</td><td>Default 1e-5
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`param`
</td><td>Parameters to be updated
</td></tr><tr><td>`moment_1`
</td><td>First moment history
</td></tr><tr><td>`moment_2`
</td><td>Second moment history
</td></tr><tr><td>`indices`
</td><td>Sparse indices
</td></tr><tr><td>`grad`
</td><td>Gradient computed
</td></tr><tr><td>`lr`
</td><td>learning rate
</td></tr><tr><td>`iter`
</td><td>iteration number
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output_param`
</td><td>Updated parameters
</td></tr><tr><td>`output_moment_1`
</td><td>Updated first moment
</td></tr><tr><td>`output_moment_2`
</td><td>Updated second moment
</td></tr></table>
### Code
[caffe2/sgd/adam_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adam_op.cc)
### Devices

- *CPU* `caffe2::SparseAdamOp<float, caffe2::CPUContext>`




---


# SparseFtrl
No documentation yet.

### Code
[caffe2/sgd/ftrl_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/ftrl_op.cc)
### Devices

- *CPU* `caffe2::SparseFtrlOp<float>`



### Engines
`SIMD` on *CPU*

---


# SplitString
No documentation yet.

### Code
[caffe2/fb/text/ops/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/string_ops.cc)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::SplitStringOp`




---


# Stats
No documentation yet.

### Code
[caffe2/fb/embnn/ops/UtilOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/embnn/ops/UtilOp.cpp)
### Devices

- *CPU* `caffe2::fb::StatsOp<float, caffe2::CPUContext>`




---


# TextDatasetRead
No documentation yet.

### Code
[caffe2/fb/text/TextReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/TextReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::TextDatasetReadOp`




---


# TextDatasetReset
No documentation yet.

### Code
[caffe2/fb/text/TextReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/TextReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::TextDatasetResetOp`




---


# TextReaderCreate
No documentation yet.

### Code
[caffe2/fb/text/TextReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/TextReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::TextReaderCreateOp`




---


# TextReaderInitLookup
No documentation yet.

### Code
[caffe2/fb/text/TextReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/TextReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::TextReaderInitLookupOp`




---


# TextReaderReadAll
No documentation yet.

### Code
[caffe2/fb/text/TextReaderOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/TextReaderOp.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::TextReaderReadAllOp`




---


# TokenShape
No documentation yet.

### Code
[caffe2/fb/text/ops/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/string_ops.cc)
### Devices

- *CPU* `caffe2::UnaryElementwiseWithArgsOp<caffe2::TensorTypes<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > >, caffe2::CPUContext, caffe2::ForEach<caffe2::fb::(anonymous namespace)::TokenShape>, caffe2::FixedType<std::basic_fbstring<char, std::char_traits<char>, std::allocator<char>, std::fbstring_core<char> > > >`




---


# Tokenize

Tokenize each string in the input vector. Tokens are returned in a flat vector, along with the number of tokens for each of the input strings, in the same order. Optionally, some tokenizers also provide the byte-range in the input string for each one of the tokens.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`tokenizer`
</td><td>Pointer to an Tokenizer instance.
</td></tr><tr><td>`texts`
</td><td>1-D tensor of utf-8 encoded strings to tokenize.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`lengths`
</td><td>int32 tensor of same size as input. Contains the number of tokens in each of the input texts.
</td></tr><tr><td>`tokens`
</td><td>1-D tensor of utf-8 encoded tokens, size sum(lengths).
</td></tr><tr><td>`byte_ranges`
</td><td>(optional) int32[sum(lengths) x 2] with [start, end) byte indices in the input string for each of the tokens produced. Notice that byte indices do not always correspond to character indices in utf-8.
</td></tr></table>
### Code
[caffe2/fb/text/ops/TokenizerOps.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/TokenizerOps.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::TokenizeOp`




---


# TokenizerCreate

TokenizerCreate create a tokenizer used by TokenizeOp. Configuration can be passed in json_config, as a json-encoded TokenizerConfig.
Refer to aml/text/tokenizers/if/tokenizer_config.thrift for the config schema.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`json_config`
</td><td>(optional) Json-encoded TokenizerConfig struct.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`tokenizer`
</td><td>Pointer to an Tokenizer instance
</td></tr></table>
### Code
[caffe2/fb/text/ops/TokenizerOps.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/text/ops/TokenizerOps.cpp)
### Devices

- *CPU* `caffe2::fb::(anonymous namespace)::TokenizerCreateOp`




---


# TrainerExampleLabelGen
No documentation yet.

### Code
[caffe2/fb/distribute/ops/TrainerExampleOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/TrainerExampleOp.cpp)
### Devices

- *CPU* `caffe2::fb::TrainerExampleLabelGenOp`




---


# WeightedSampling
Sample an item based on provided weights.
### Interface
<table><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`index`
</td><td>The index of the sampled item.
</td></tr></table>
### Code
[caffe2/fb/operators/random_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/operators/random_ops.cc)
### Devices

- *CPU* `caffe2::WeightedSamplingOp<caffe2::CPUContext>`




---


# ZMQBind
Create a ZMQ socket and bind with an address
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`type`
</td><td>ZMQ socket type
</td></tr><tr><td>`send_high_water_mark`
</td><td>Limit number of outgoing messages.
</td></tr><tr><td>`recv_high_water_mark`
</td><td>Limit number of incoming messages.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`context`
</td><td>unique_ptr<zmq::context_t>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`socket`
</td><td>unique_ptr<zmq::socket_t>
</td></tr><tr><td>`address`
</td><td>address of this socket
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQBindOp`




---


# ZMQCompress
Serialize, compress tensor and put it into IOBuf
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`compression_codec`
</td><td>defined by folly::io::CodecType
</td></tr><tr><td>`compression_level`
</td><td>defined by folly::io::COMPRESSION_LEVEL_*
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQCompressOp`




---


# ZMQConnect
Create a ZMQ socket and connect to one or more adddresses.
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`type`
</td><td>ZMQ socket type
</td></tr><tr><td>`send_high_water_mark`
</td><td>Limit number of outgoing messages.
</td></tr><tr><td>`recv_high_water_mark`
</td><td>Limit number of incoming messages.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`context`
</td><td>unique_ptr<zmq::context_t>
</td></tr><tr><td>`address_1`
</td><td>1st address that this socket should connect to
</td></tr><tr><td>`address_2`
</td><td>2nd address to connect to
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`socket`
</td><td>unique_ptr<zmq::socket_t>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQConnectOp`




---


# ZMQContextCreate
Create a ZMQ Context
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`num_threads`
</td><td>number of threads that will be used in zmq context
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`context`
</td><td>unique_ptr<zmq::context_t>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQContextCreateOp`




---


# ZMQDecompress
Decompress, deserialize and recover tensor from IOBuf
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`compression_codec`
</td><td>defined by folly::io::CodecType
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQDecompressOp`




---


# ZMQRawRecv

Recv raw content through zmq socket.
 Outputs are IOBuf pointers which will hold raw content.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`socket`
</td><td>unique_ptr<zmq::socket_t>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQRawRecvOp`




---


# ZMQRawSend

Send raw content through zmq socket.
 Input(0) is the socket pointer.
The rest of the inputs are IOBuf pointers which will be sent through network.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`socket`
</td><td>unique_ptr<zmq::socket_t>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQRawSendOp`




---


# ZMQRecv

Recv the blobs through the given socket.
 Input(0) is the socket pointer.
The outputs are the blobs that received from the communication. The order of the blobs are the same as it was sent.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`socket`
</td><td>unique_ptr<zmq::socket_t>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQRecvOp`




---


# ZMQSend

Send the blobs through the given socket.
 Input(0) is the socket pointer.
The rest of the inputs are the blobs to be sent.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`socket_in`
</td><td>unique_ptr<zmq::socket_t>
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQSendOp`




---


# ZMQSinglePoll

Check whether can we read or write to the given socket. If it is blocked for certain time, check whether all of the signals are set. Return true if all of the signals are set.
 This is used for ending things properly. For example, if the socket is for reading training data from remote machine. It is hard to know when to stop reading. Even that we can send a special data through the same channel, since the order of the messages is not guaranteed, the reading procedure can not be stopped immediately. It is still necessary to drain the queues.
 To end communication properly, we need to set a unique signal on both side if it exits due to the other reason. On the receiver side, after the queue is drained, it will trigger the timeout and it will figure out that the signals are set. On the sender side, if the queue is full and reciver are all gone, the timeout will also be triggered and it will figure out the status similarly.
 The timeout should not be too small so that it quries KVStore too frequently which is wasting communication time. Usually this is the final exit, we don't need to worry too much about the timeout. But if it is inside a loop and exits very frequently, this would not be a good solution.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`signal_blobs`
</td><td>the singal blob names will be checked
</td></tr><tr><td>`poll_in`
</td><td>set to non-zero value if the socket is for reading
</td></tr><tr><td>`timeout`
</td><td>timeout by ms
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`data_socket`
</td><td>unique_ptr<zmq::socket_t>
</td></tr><tr><td>`kv_handler`
</td><td>unique_ptr<StoreHandler>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`should_stop`
</td><td>return true if timeout is triggered and all signals are set
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQSinglePollOp`




---


# ZMQSocketClose

Close a ZMQ socket, wait for time specified by LINGER option, then drop all of the messages in the queue.

### Interface
<table><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`socket_in`
</td><td>unique_ptr<zmq::socket_t>
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`socket_out`
</td><td>the same socket for in-place change
</td></tr></table>
### Code
[caffe2/fb/distribute/ops/ZeroMQOp.cpp](https://github.com/caffe2/caffe2/blob/master/caffe2/fb/distribute/ops/ZeroMQOp.cpp)
### Devices

- *CPU* `caffe2::fb::ZMQSocketCloseOp`




---


# NCCLAllGather
No documentation yet.

### Code
[caffe2/contrib/nccl/cuda_nccl_op_gpu.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/contrib/nccl/cuda_nccl_op_gpu.cc)
### Devices

- *GPU* `caffe2::NCCLAllGatherOp<float>`




---


# NCCLAllreduce
No documentation yet.

### Code
[caffe2/contrib/nccl/cuda_nccl_op_gpu.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/contrib/nccl/cuda_nccl_op_gpu.cc)
### Devices

- *GPU* `caffe2::NCCLAllreduceOp<float>`




---


# NCCLBroadcast
No documentation yet.

### Code
[caffe2/contrib/nccl/cuda_nccl_op_gpu.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/contrib/nccl/cuda_nccl_op_gpu.cc)
### Devices

- *GPU* `caffe2::NCCLBroadcastOp<float>`




---


# NCCLReduce
No documentation yet.

### Code
[caffe2/contrib/nccl/cuda_nccl_op_gpu.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/contrib/nccl/cuda_nccl_op_gpu.cc)
### Devices

- *GPU* `caffe2::NCCLReduceOp<float>`




---


# FCGradient_Decomp
No documentation yet.

### Code
[caffe2/experiments/operators/fully_connected_op_decomposition.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/fully_connected_op_decomposition.cc)
### Devices

- *CPU* `caffe2::FullyConnectedDecompGradientOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`

- *GPU* `caffe2::FullyConnectedDecompGradientOp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`




---


# FCGradient_Prune
No documentation yet.

### Code
[caffe2/experiments/operators/fully_connected_op_prune.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/fully_connected_op_prune.cc)
### Devices

- *CPU* `caffe2::FullyConnectedPruneGradientOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`




---


# FC_Decomp
No documentation yet.

### Code
[caffe2/experiments/operators/fully_connected_op_decomposition.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/fully_connected_op_decomposition.cc)
### Devices

- *CPU* `caffe2::FullyConnectedOpDecomp<float, caffe2::CPUContext, caffe2::DefaultEngine>`




---


# FC_Prune
No documentation yet.

### Code
[caffe2/experiments/operators/fully_connected_op_prune.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/fully_connected_op_prune.cc)
### Devices

- *CPU* `caffe2::FullyConnectedOpPrune<float, caffe2::CPUContext, caffe2::DefaultEngine>`




---


# FC_Sparse
No documentation yet.

### Code
[caffe2/experiments/operators/fully_connected_op_sparse.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/fully_connected_op_sparse.cc)
### Devices

- *CPU* `caffe2::FullyConnectedOp_SPARSE<float, caffe2::CPUContext, caffe2::DefaultEngine>`




---


# FunHash

This layer compresses a fully-connected layer for sparse inputs via hashing.
It takes four required inputs and an optional fifth input.
The first three inputs  `scalars` ,  `indices` , and  `segment_ids`  are the sparse segmented representation of sparse data, which are the same as the last three inputs of the  `SparseSortedSegmentWeightedSum`  operator. If the argument  `num_segments`  is specified, it would be used as the first dimension for the output; otherwise it would be derived from the maximum segment ID.
 The fourth input is a 1D weight vector. Each entry of the fully-connected layer would be randomly mapped from one of the entries in this vector.
 When the optional fifth input vector is present, each weight of the fully-connected layer would be the linear combination of K entries randomly mapped from the weight vector, provided the input (length-K vector) serves as the coefficients.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`num_outputs`
</td><td>Number of outputs
</td></tr><tr><td>`num_segments`
</td><td>Number of segments
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`scalars`
</td><td>Values of the non-zero entries of the sparse data.
</td></tr><tr><td>`indices`
</td><td>Indices to the non-zero valued features.
</td></tr><tr><td>`segment_ids`
</td><td>Segment IDs corresponding to the non-zero entries.
</td></tr><tr><td>`weight`
</td><td>Weight vector
</td></tr><tr><td>`alpha`
</td><td>Optional coefficients for linear combination of hashed weights.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Output tensor with the first dimension equal to the number of segments.
</td></tr></table>
### Code
[caffe2/experiments/operators/funhash_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/funhash_op.cc)
### Devices

- *CPU* `caffe2::FunHashOp<float, caffe2::CPUContext>`




---


# FunHashGradient
No documentation yet.

### Code
[caffe2/experiments/operators/funhash_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/funhash_op.cc)
### Devices

- *CPU* `caffe2::FunHashGradientOp<float, caffe2::CPUContext>`




---


# SparseFunHash

This layer compresses a fully-connected layer for sparse inputs via hashing.
It takes four required inputs and an option fifth input.
The first three inputs  `scalars` ,  `indices` , and  `segment_ids`  are the sparse segmented representation of sparse data, which are the same as the last three inputs of the  `SparseSortedSegmentWeightedSum`  operator. If the argument  `num_segments`  is specified, it would be used as the first dimension for the output; otherwise it would be derived from the maximum segment ID.
 The fourth input is a 1D weight vector. Each entry of the fully-connected layer would be randomly mapped from one of the entries in this vector.
 When the optional fifth input vector is present, each weight of the fully-connected layer would be the linear combination of K entries randomly mapped from the weight vector, provided the input (length-K vector) serves as the coefficients.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`num_outputs`
</td><td>Number of outputs
</td></tr><tr><td>`num_segments`
</td><td>Number of segments
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`scalars`
</td><td>Values of the non-zero entries of the sparse data.
</td></tr><tr><td>`indices`
</td><td>Indices to the non-zero valued features.
</td></tr><tr><td>`segment_ids`
</td><td>Segment IDs corresponding to the non-zero entries.
</td></tr><tr><td>`weight`
</td><td>Weight vector
</td></tr><tr><td>`alpha`
</td><td>Optional coefficients for linear combination of hashed weights.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`output`
</td><td>Output tensor with the first dimension equal to the number of segments.
</td></tr></table>
### Code
[caffe2/experiments/operators/sparse_funhash_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/sparse_funhash_op.cc)
### Devices

- *CPU* `caffe2::SparseFunHashOp<float, caffe2::CPUContext>`




---


# SparseFunHashGradient
No documentation yet.

### Code
[caffe2/experiments/operators/sparse_funhash_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/sparse_funhash_op.cc)
### Devices

- *CPU* `caffe2::SparseFunHashGradientOp<float, caffe2::CPUContext>`




---


# SparseMatrixReshape

Compute the indices of the reshaped sparse matrix.
 It takes two 1D tensors as input: the column indices (in int64) and the row indices (in int), which correspond to  `INDICES`  and  `SEGMENT_IDS`  in  `SparseSortedSegment`  family.
It outputs the corresponding reshaped column and row indices.
 Two arguments are required: an argument  `old_shape`  specifies the original shape of the matrix, and  `new_shape`  specifies the new shape.
One of the dimension in  `old_shape`  and  `new_shape`  can be -1.
The valid combinations are listed below, where p, q, r, s are strictly positive integers.
 old_shape=(p, q) new_shape=(r, s)  old_shape=(p, q) new_shape=(-1, s)  old_shape=(p, q) new_shape=(r, -1)  old_shape=(-1, q) new_shape=(-1, s)  Note that only the first dimension in  `old_shape`  can be -1. In that case the second dimension in  `new_shape`  must NOT be -1.

### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`old_shape`
</td><td>Old shape.
</td></tr><tr><td>`new_shape`
</td><td>New shape.
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`old_col`
</td><td>Original column indices.
</td></tr><tr><td>`old_row`
</td><td>Original row indices.
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`new_col`
</td><td>New column indices.
</td></tr><tr><td>`new_row`
</td><td>New row indices.
</td></tr></table>
### Code
[caffe2/experiments/operators/sparse_matrix_reshape_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/sparse_matrix_reshape_op.cc)
### Devices

- *CPU* `caffe2::SparseMatrixReshapeOp<caffe2::CPUContext>`




---


# TTContraction

Tensor contraction C = A * B
### Interface
<table><tr><td>*Arguments*
</td><td>
</td></tr><tr><td>`K`
</td><td>i_{k-1} * r_k
</td></tr><tr><td>`M`
</td><td>r_{k-1} * o_{k-1}
</td></tr><tr><td>`N`
</td><td>o_k
</td></tr><tr><td>*Inputs*
</td><td>
</td></tr><tr><td>`A`
</td><td>2D matrix of size (K x M)
</td></tr><tr><td>`B`
</td><td>tensor
</td></tr><tr><td>*Outputs*
</td><td>
</td></tr><tr><td>`C`
</td><td>contracted tensor
</td></tr></table>
### Code
[caffe2/experiments/operators/tt_contraction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/tt_contraction_op.cc)
### Devices

- *CPU* `caffe2::TTContractionOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`

- *GPU* `caffe2::TTContractionOp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`




---


# TTContractionGradient
No documentation yet.

### Code
[caffe2/experiments/operators/tt_contraction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/tt_contraction_op.cc)
### Devices

- *CPU* `caffe2::TTContractionGradientOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`

- *GPU* `caffe2::TTContractionGradientOp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`




---


# TTPad
No documentation yet.

### Code
[caffe2/experiments/operators/tt_pad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/tt_pad_op.cc)
### Devices

- *CPU* `caffe2::TTPadOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`




---


# TTPadGradient
No documentation yet.

### Code
[caffe2/experiments/operators/tt_pad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/experiments/operators/tt_pad_op.cc)
### Devices

- *CPU* `caffe2::TTPadGradientOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`




---


# CopyCPUToGPU
No schema documented yet.

### Devices

- *GPU* `caffe2::CopyOp<caffe2::CUDAContext, caffe2::CUDAContext, caffe2::CPUContext>`



# CopyGPUToCPU
No schema documented yet.

### Devices

- *GPU* `caffe2::CopyOp<caffe2::CUDAContext, caffe2::CPUContext, caffe2::CUDAContext>`



# FC_Dcomp
No schema documented yet.

### Devices

- *GPU* `caffe2::FullyConnectedOpDecomp<float, caffe2::CUDAContext, caffe2::DefaultEngine>`



# ReluFp16
No schema documented yet.

### Devices

- *GPU* `caffe2::ReluOp<caffe2::__f16, caffe2::CUDAContext>`



# ReluFp16Gradient
No schema documented yet.

### Devices

- *GPU* `caffe2::ReluGradientOp<caffe2::__f16, caffe2::CUDAContext>`



# TTLinearGradient
No schema documented yet.

### Devices

- *CPU* `caffe2::TTLinearGradientOp<float, caffe2::CPUContext, caffe2::DefaultEngine>`



