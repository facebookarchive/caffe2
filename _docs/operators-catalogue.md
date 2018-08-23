---
docid: operators-catalog
title: Operators Catalog
layout: operators
permalink: /docs/operators-catalogue.html
---
* TOC
{:toc}


## APMeter


APMeter computes Average Precision for binary or multi-class classification.
It takes two inputs: prediction scores P of size (n_samples x n_classes), and true labels Y of size (n_samples x n_classes). It returns a single float number per class for the average precision of that class.



### Interface


---------- | ----------
*Arguments* | 
`buffer_size` | (int32_t) indicates how many predictions should the op buffer. defaults to 1000
*Inputs* | 
`predictions` | 2-D tensor (Tensor<float>) of size (num_samples xnum_classes) containing prediction scores
`labels` | 2-D tensor (Tensor<int>) of size (num_samples) containing true labels for each sample
*Outputs* | 
`AP` | 1-D tensor (Tensor<float>) of size num_classes containing average precision for each class


### Code


[caffe2/operators/apmeter_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/apmeter_op.cc)

---



## Abs


Calculates the absolute value of the given input tensor, element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The absolute value of the input tensor computed element-wise


### Code


[caffe2/operators/abs_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/abs_op.cc)

---



## AbsGradient

No documentation yet.


### Code


[caffe2/operators/abs_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/abs_op.cc)

---



## Accumulate


Accumulate operator accumulates the input tensor to the output tensor. If the output tensor already has the right size, we add to it; otherwise, we first initialize the output tensor to all zeros, and then do accumulation. Any further calls to the operator, given that no one else fiddles with the output in the interim, will do simple accumulations.
Accumulation is done using Axpby operation as shown:  

```
  Y = 1*X + gamma*Y
```

 where X is the input tensor, Y is the output tensor and gamma is the multiplier argument.



### Interface


---------- | ----------
*Arguments* | 
`gamma` | (float, default 1.0) Accumulation multiplier
*Inputs* | 
`input` | The input tensor that has to be accumulated to the output tensor. If the output size is not the same as input size, the output tensor is first reshaped and initialized to zero, and only then, accumulation is done.
*Outputs* | 
`output` | Accumulated output tensor


### Code


[caffe2/operators/accumulate_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/accumulate_op.cc)

---



## AccumulateHistogram


This operator calculate thes histogram of values in input tensor.
There're 2 outputs, one for histogram of current input tensor, and another for histogram of the all input tensors accumulated through history.
The output would contain num_buckets + 2 values. index[1 ... num_buckets] for values in [lower_bound, upper_bound) interval. And the rest 2 for values smaller than lower_bound or greater than upper_bound respectively.



### Interface


---------- | ----------
*Arguments* | 
`lower_bound` | the lower bound value
`upper_bound` | the upper bound value
`num_buckets` | number of buckets to use in [lower_bound, upper_bound)
*Inputs* | 
`X` | Input tensor.
*Outputs* | 
`CurHist` | Output histogram of the current tensor.
`AccHist` | Accumulated histogram of the history tensor.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## Accuracy


Accuracy takes two inputs- predictions and labels, and returns a float accuracy value for the batch. Predictions are expected in the form of 2-D tensor containing a batch of scores for various classes, and labels are expected in the  form of 1-D tensor containing true label indices of samples in the batch. If the score for the label index in the predictions is the highest among all classes, it is considered a correct prediction.



### Interface


---------- | ----------
*Arguments* | 
`top_k` | Count as correct by comparing the true label to the top k scoring classes (default 1: only compare to the top scoring class i.e. argmax)
*Inputs* | 
`predictions` | 2-D tensor (Tensor<float>) of size (num_batches x num_classes) containing scores
`labels` | 1-D tensor (Tensor<int>) of size (num_batches) having the indices of true labels
*Outputs* | 
`accuracy` | 1-D tensor (Tensor<float>) of size 1 containing accuracy


### Code


[caffe2/operators/accuracy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/accuracy_op.cc)

---



## Adagrad


 Computes the AdaGrad update for an input gradient and accumulated history. Concretely, given inputs (param, grad, moment, learning_rate), computes   

```
    new_moment = moment + square(grad)
    new_grad = learning_rate * grad / (sqrt(new_moment) + epsilon)
    new_param = param + new_grad
```

 and returns (new_param, new_moment).
 


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | Default 1e-5
`decay` | Default 1. If it is in (0, 1), the gradient square sum is decayed by this factor.
*Inputs* | 
`param` | Parameters to be updated
`moment` | Moment history
`grad` | Gradient computed
`lr` | learning rate
*Outputs* | 
`output_param` | Updated parameters
`output_moment` | Updated moment


### Code


[caffe2/sgd/adagrad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adagrad_op.cc)

---



## Adam


 Computes the Adam update ( [https://arxiv.org/abs/1412.6980)](https://arxiv.org/abs/1412.6980))  for an input gradient and momentum parameters. Concretely, given inputs (param, m1, m2, grad, lr, iters),   

```
    t = iters + 1
    corrected_local_rate = lr * sqrt(1 - power(beta2, t)) /
      (1 - power(beta1, t))
    m1_o = (beta1 * m1) + (1 - beta1) * grad
    m2_o = (beta2 * m2) + (1 - beta2) * np.square(grad)
    grad_o = corrected_local_rate * m1_o / \
        (sqrt(m2_o) + epsilon)
    param_o = param + grad_o

```

 and returns (param_o, m1_o, m2_o)  


### Interface


---------- | ----------
*Arguments* | 
`beta1` | Default 0.9
`beta2` | Default 0.999
`epsilon` | Default 1e-5
*Inputs* | 
`param` | Parameters to be updated
`moment_1` | First moment history
`moment_2` | Second moment history
`grad` | Gradient computed
`lr` | learning rate
`iter` | iteration number
*Outputs* | 
`output_param` | Updated parameters
`output_moment_1` | Updated first moment
`output_moment_2` | Updated second moment


### Code


[caffe2/sgd/adam_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adam_op.cc)

---



## Add


Performs element-wise binary addition (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and type as A


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## AddPadding


Given a partitioned tensor T<N, D1..., Dn>, where the partitions are defined as ranges on its outer-most (slowest varying) dimension N, with given range lengths, return a tensor T<N + 2*padding_width, D1 ..., Dn> with paddings added to the start and end of each range.
Optionally, different paddings can be provided for beginning and end. Paddings provided must be a tensor T<D1..., Dn>.
 If no padding is provided, add zero padding.
If no lengths vector is provided, add padding only once, at the start and end of data.



### Interface


---------- | ----------
*Arguments* | 
`padding_width` | Number of copies of padding to add around each range.
`end_padding_width` | (Optional) Specifies a different end-padding width.
*Inputs* | 
`data_in` | (T<N, D1..., Dn>) Input data
`lengths` | (i64) Num of elements in each range. sum(lengths) = N.
`start_padding` | T<D1..., Dn> Padding data for range start.
`end_padding` | T<D1..., Dn> (optional) Padding for range end. If not provided, start_padding is used as end_padding as well.
*Outputs* | 
`data_out` | (T<N + 2*padding_width, D1..., Dn>) Padded data.
`lengths_out` | (i64, optional) Lengths for each padded range.


### Code


[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)

---



## Alias


Makes the output and the input share the same underlying storage.
 WARNING: in general, in caffe2's operator interface different tensors should have different underlying storage, which is the assumption made by components such as the dependency engine and memory optimization. Thus, in normal situations you should not use the AliasOp, especially in a normal forward-backward pass.
 The Alias op is provided so one can achieve true asynchrony, such as Hogwild, in a graph. But make sure you understand all the implications similar to multi-thread computation before you use it explicitly.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor whose storage will be shared.
*Outputs* | 
`output` | Tensor of same shape as input, sharing its storage.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## Allgather


Does an allgather operation among the nodes.



### Interface


---------- | ----------
*Inputs* | 
`comm_world` | The common world.
`X` | A tensor to be allgathered.
*Outputs* | 
`Y` | The allgathered tensor, same on all nodes.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## Allreduce


Does an allreduce operation among the nodes. Currently only Sum is supported.



### Interface


---------- | ----------
*Inputs* | 
`comm_world` | The common world.
`X` | A tensor to be allreduced.
*Outputs* | 
`Y` | The allreduced tensor, same on all nodes.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## And


Performs element-wise logical operation  `and`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## Append


Append input 2 to the end of input 1.
Input 1 must be the same as output, that is, it is required to be in-place.
Input 1 may have to be re-allocated in order for accommodate to the new size.
Currently, an exponential growth ratio is used in order to ensure amortized constant time complexity.
All except the outer-most dimension must be the same between input 1 and 2.



### Interface


---------- | ----------
*Inputs* | 
`dataset` | The tensor to be appended to.
`new_data` | Tensor to append to the end of dataset.
*Outputs* | 
`dataset` | Same as input 0, representing the mutated tensor.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## ArgMax


Retrive the argmax of the axis dimension. Given an input tensor of shape [a_0, a_1, ..., a_{n-1}] and two arguments axis as int and keepdims as bool, returns one output: - Index tensor which contains the indices of the largest element. It has the  

```
  same dims as X.dims() with the dimension along axis equals 1 when
  keepdims == true otherwise removed.
```

     


### Interface


---------- | ----------
*Arguments* | 
`axis` | The axis to get argmax.
`keepdims` | Whether to keep the axis dim in the output.
*Inputs* | 
`X` | Tenor of shape [a_0, a_1, ..., a_{n-1}].
*Outputs* | 
`Indices` | Tensor of indices for the largest values.


### Code


[caffe2/operators/arg_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/arg_ops.cc)

---



## ArgMin


Retrive the argmin of the axis dimension. Given an input tensor of shape [a_0, a_1, ..., a_{n-1}] and two arguments axis as int and keepdims as bool, returns one output: - Index tensor which contains the indices of the largest element. It has the  

```
  same dims as X.dims() with the dimension along axis equals 1 when
  keepdims == true otherwise removed.
```

     


### Interface


---------- | ----------
*Arguments* | 
`axis` | The axis to get argmin.
`keepdims` | Whether to keep the axis dim in the output.
*Inputs* | 
`X` | Tenor of shape [a_0, a_1, ..., a_{n-1}].
*Outputs* | 
`Indices` | Tensor of indices for the largest values.


### Code


[caffe2/operators/arg_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/arg_ops.cc)

---



## Assert


Assertion op. Takes in a tensor of bools, ints, longs, or long longs and checks if all values are true when coerced into a boolean. In other words, for non-bool types this asserts that all values in the tensor are non-zero.
	


### Interface


---------- | ----------
*Arguments* | 
`error_msg` | An error message to print when the assert fails.


### Code


[caffe2/operators/assert_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/assert_op.cc)

---



## AtomicAppend

No documentation yet.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## AtomicFetchAdd


Given a mutex and two int32 scalar tensors, performs an atomic fetch add by mutating the first argument and adding it to the second input argument. Returns the updated integer and the value prior to the update.



### Interface


---------- | ----------
*Inputs* | 
`mutex_ptr` | Blob containing to a unique_ptr<mutex>
`mut_value` | Value to be mutated after the sum.
`increment` | Value to add to the first operand.
*Outputs* | 
`mut_value` | Mutated value after sum. Usually same as input 1.
`fetched_value` | Value of the first operand before sum.


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)

---



## AtomicIter


Similar to Iter, but takes a mutex as the first input to make sure that updates are carried out atomically. This can be used in e.g. Hogwild sgd algorithms.



### Interface


---------- | ----------
*Inputs* | 
`mutex` | The mutex used to do atomic increment.
`iter` | The iter counter as an int64_t TensorCPU.


### Code


[caffe2/sgd/iter_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/iter_op.cc)

---



## AveragePool

AveragePool  consumes an input blob X and applies average pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Average pooling consisting of averaging all values of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool1D

AveragePool1D  consumes an input blob X and applies average pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Average pooling consisting of averaging all values of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool1DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragePool2D

AveragePool2D  consumes an input blob X and applies average pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Average pooling consisting of averaging all values of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool2DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragePool3D

AveragePool3D  consumes an input blob X and applies average pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Average pooling consisting of averaging all values of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool3DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragePoolGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragedLoss


AveragedLoss takes in a 1-D tensor as input and returns a single output float value which represents the average of input data (average of the losses).



### Interface


---------- | ----------
*Inputs* | 
`input` | The input data as Tensor
*Outputs* | 
`output` | The output tensor of size 1 containing the averaged value.


### Code


[caffe2/operators/loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc)

---



## AveragedLossGradient

No documentation yet.


### Code


[caffe2/operators/loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc)

---



## BBoxTransform


Transform proposal bounding boxes to target bounding box using bounding box  

```
    regression deltas.
```




### Interface


---------- | ----------
*Arguments* | 
`weights` | vector<float> weights [wx, wy, ww, wh] for the deltas
`apply_scale` | bool (default true), transform the boxes to the scaled image space after applying the bbox deltas.Set to false to match the detectron code, set to true for keypoint models and for backward compatibility
`correct_transform_coords` | bool (default false), Correct bounding box transform coordates, see bbox_transform() in boxes.py Set to true to match the detectron code, set to false for backward compatibility
*Inputs* | 
`rois` | Bounding box proposals in pixel coordinates, Size (M, 4), format [x1, y1, x2, y2], orSize (M, 5), format [batch_index, x1, y1, x2, y2]. If proposals from multiple images in a batch are present, they should be grouped sequentially and in incremental order.
`deltas` | bounding box translations and scales,size (M, 4*K), format [dx, dy, dw, dh], K = # classes
`im_info` | Image dimensions, size (batch_size, 3), format [img_height, img_width, img_scale]
*Outputs* | 
`box_out` | Pixel coordinates of the transformed bounding boxes,Size (M, 4*K), format [x1, y1, x2, y2]
`roi_batch_splits` | Tensor of shape (batch_size) with each element denoting the number of RoIs belonging to the corresponding image in batch


### Code


[caffe2/operators/bbox_transform_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/bbox_transform_op.cc)

---



## BRGNCHWCToPackedInt8BGRAStylizerDeprocess

No documentation yet.


### Code


[caffe2/operators/stylizer_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stylizer_ops.cc)

---



## Barrier


Does a barrier operation among the nodes.



### Interface


---------- | ----------
*Inputs* | 
`comm_world` | The common world.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## BatchBoxCox


Input  `data`  is a N * D matrix. Apply box-cox transform for each column.
 `lambda1`  and  `lambda2`  is of size D that defines the hyper-parameters for the transform of each column  `x`  of the input  `data` :   

```
    ln(x + lambda2), if lambda1 == 0
    ((x + lambda2)^lambda1 - 1)/lambda1, if lambda1 != 0

```




### Interface


---------- | ----------
*Inputs* | 
`data` | input float or double N * D matrix
`lambda1` | tensor of size D with the same type as data
`lambda2` | tensor of size D with the same type as data
*Outputs* | 
`output` | output matrix that applied box-cox transform


### Code


[caffe2/operators/batch_box_cox_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/batch_box_cox_op.cc)

---



## BatchBucketOneHot


Input is a matrix tensor. Its first dimension is the batch size. For each column, bucketize it based on the boundary values and then do one hot encoding. The  `lengths`  specifies the number of boundary values for each column. The final number of buckets is this number plus 1. This would also be the expanded feature size.  `boundaries`  specifies all the boundary values.
Note that each bucket is right-inclusive. That is, given boundary values [b1, b2, b3], the buckets are defined as (-int, b1], (b1, b2], (b2, b3], (b3, inf).
For example   

```
  If data = [[2, 3], [4, 1], [2, 5]], lengths = [2, 3],
  and boundaries = [0.1, 2.5, 1, 3.1, 4.5], then

  output = [[0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]]

```




### Interface


---------- | ----------
*Inputs* | 
`data` | input tensor matrix
`lengths` | the size is the same as the width of the `data`
`boundaries` | bucket boundaries
*Outputs* | 
`output` | output matrix that expands each input column with one hot encodingbased on the bucketization


### Code


[caffe2/operators/one_hot_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc)

---



## BatchDenseToSparse


This Op is a inverse of BatchSparseToDenseOp.
Basically, given a  `lengths`  vector, a  `indices`  vector, and a dense matrix  `dense` , output  `value`  vector so that, along with  `lengths`  vector and  `indices`  vector, forms a sparse representation of the dense matrix.
 A sparse matrix is represented by  `lengths`  vector,  `indices`  vector, and  `values`  vector. Each element in  `lengths`  vector (lengths[ `i` ]) represents the number of indices in this batch (batch  `i` ).
With in each batch,  `indices`  should not have duplicate number.
 For example, with input:   

```
  lengths = [2, 3, 1]
  indices = [0, 1, 2, 3, 4, 5]
  output = [[6, 7, 0, 0, 0,  0],
            [0, 0, 8, 9, 10, 0],
            [0, 0, 0, 0, 0, 11]]

```

 The output is:   

```
  values = [6, 7, 8, 9, 10, 11]

```

 after running this operator.



### Interface


---------- | ----------
*Inputs* | 
`lengths` | Flatten lengths, Used to break down indices into per batch indices
`indices` | Flatten indices, tensor of total size = \sum lengths, containing the indices 
`dense` | dense 2-D tensor, first dim = len(lengths), last dim > Any(indices)
*Outputs* | 
`values` | Values, tensor of the same size as `indices` and same data type as dense tensor.


### Code


[caffe2/operators/batch_sparse_to_dense_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/batch_sparse_to_dense_op.cc)

---



## BatchGather


Batch gather operation, first dimension in DATA is the batch size.
Given DATA tensor of rank r >= 2, and INDICES tensor of rank q >= 1, gather entries of the outer-most dimension of DATA indexed by INDICES, and concatenate them in an output tensor of rank (q - 1) + (r - 1).
 Example:  

```
  DATA  = [
      [1.0, 1.2, 2.4, 4.5],
      [2.3, 3.4, 3.6, 2.3],
      [4.5, 5.7, 1.2, 4.5],
  ]
  INDICES = [
      [0, 2],
  ]
  OUTPUT = [
      [1.0, 2.4],
      [2.3, 3.6],
      [4.5, 1.2],
  ]
```




### Interface


---------- | ----------
*Inputs* | 
`DATA` | Tensor of rank r >= 2.
`INDICES` | Tensor of int32/int64 indices, of any rank q.
*Outputs* | 
`OUTPUT` | Tensor of rank (q - 1) + (r - 1).


### Code


[caffe2/operators/batch_gather_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/batch_gather_ops.cc)

---



## BatchGatherGradient

No documentation yet.


### Code


[caffe2/operators/batch_gather_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/batch_gather_ops.cc)

---



## BatchMatMul


Batch Matrix multiplication Yi = Ai  * Bi, where A has shape (dim0, dim1, ... M, K), B has shape (dim0, dim1, ... K, N), Y has shape (dim0, dim1, ... M, N) and i ranges from 0 to (dim0 *  dim1 ...) - 1. rank(A) == rank(B) >= 2. In case of A and B being two diemnsional, it behaves like normal matrix multiplication.



### Interface


---------- | ----------
*Arguments* | 
`trans_a` | Pass 1 to transpose the last two dimensions of A before doing multiplication
`trans_b` | Pass 1 to transpose the last two dimensions of B before doing multiplication
`broadcast` | Pass 1 to allow broadcasting of dimensions. Behavior is the same as numpy.matmul. Gradient is currently not supported when running in broadcast mode.
*Inputs* | 
`A` | tensor of shape (dim0, dim1 ... M, K)
`B` | tensor of shpae (dim0, dim2 ... K, N)
*Outputs* | 
`Y` | tensor of shape (dim0, dim1 ... M, N)


### Code


[caffe2/operators/batch_matmul_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/batch_matmul_op.cc)

---



## BatchOneHot


Input is a matrix tensor. Its first dimension is the batch size. Expand each column of it using one hot encoding. The  `lengths`  specifies the size of each column after encoding, and the  `values`  is the dictionary value of one-hot encoding for each column. For example   

```
  If data = [[2, 3], [4, 1], [2, 5]], lengths = [2, 3],
  and values = [2, 4, 1, 3, 5], then

  output = [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 0, 1]]
```




### Interface


---------- | ----------
*Inputs* | 
`data` | input tensor matrix
`lengths` | the size is the same as the width of the `data`
`values` | one hot encoding dictionary values
*Outputs* | 
`output` | output matrix that expands each input column with one hot encoding


### Code


[caffe2/operators/one_hot_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc)

---



## BatchSparseToDense


Convert sparse matrix representation into dense matrix.
 A sparse matrix is represented by  `lengths`  vector,  `indices`  vector, and  `values`  vector. Each element in  `lengths`  vector (lengths[ `i` ]) represents the number of indices in this batch (batch  `i` ).
With in each batch,  `indices`  should not have duplicate number.
 For example, with input:   

```
  lengths = [2, 3, 1]
  indices = [0, 1, 2, 3, 4, 5]
  values =  [6, 7, 8, 9, 10, 11]
  dense_dim = 6
  default_value = 0

```

 The output is:   

```
  output = [[6, 7, 0, 0, 0,  0],
            [0, 0, 8, 9, 10, 0],
            [0, 0, 0, 0, 0, 11]]

```

 after running this operator.



### Interface


---------- | ----------
*Arguments* | 
`dense_last_dim` | Optional, output dense last dimension. If both this argument and output_shape_inference are set, it should be consistent with output_shape_inference's last dim
`default_value` | Optional, missing values are filled with this value.default_value = 0 when not set
*Inputs* | 
`lengths` | Flatten tensor, used to break down indices and values into per batch indices and values.
`indices` | Flatten tensor of total size = \sum lengths, containing the indices 
`values` | Data tensor, dimension has to match `indices`
`output_shape_inference` | Optional, a dense tensor whose shape define the output shape
*Outputs* | 
`dense` | 2-D dense tensor, with 1st dim = len(lengths), 2nd dim = dense_last_dimin the arg list, the tensor is of the same data type as `values`.Missing values are filled with default_value


### Code


[caffe2/operators/batch_sparse_to_dense_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/batch_sparse_to_dense_op.cc)

---



## BatchToSpace


 BatchToSpace for 4-D tensors of type T.
 Rearranges (permutes) data from batch into blocks of spatial data, followed by cropping. This is the reverse transformation of SpaceToBatch. More specifically, this op outputs a copy of the input tensor where values from the batch dimension are moved in spatial blocks to the height and width dimensions, followed by cropping along the height and width dimensions.
 


### Code


[caffe2/operators/space_batch_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/space_batch_op.cc)

---



## BernoulliJSD


Computes the Jensen-Shannon divergence (JSD) between two Bernoulli distributions where each is parametrized by a single probability.



### Interface


---------- | ----------
*Inputs* | 
`T` | array of probabilities for target
*Outputs* | 
`L` | array of JSD losses


### Code


[caffe2/operators/jsd_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/jsd_op.cc)

---



## BernoulliJSDGradient

No documentation yet.


### Code


[caffe2/operators/jsd_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/jsd_op.cc)

---



## BooleanMask


Given a data tensor and a 1D boolean mask tensor, returns a tensor containing only the elements corresponding to positions where the mask is true.



### Interface


---------- | ----------
*Inputs* | 
`data` | The 1D, original data tensor.
`mask` | A tensor of bools of same shape as `data`.
*Outputs* | 
`masked_data` | A tensor of same type as `data`.
`masked_indices` | A tensor for indices.


### Code


[caffe2/operators/boolean_mask_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/boolean_mask_ops.cc)

---



## BooleanMaskLengths


Given a tensor of int32 segment lengths and a mask (boolean) tensor, return the segment lengths of a corresponding segmented tensor after BooleanMask is applied.



### Interface


---------- | ----------
*Inputs* | 
`lengths` | A 1D int32 tensor representing segment lengths.
`mask` | A 1D bool tensor of values to keep.
*Outputs* | 
`masked_lengths` | Segment lengths of a masked tensor.


### Code


[caffe2/operators/boolean_mask_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/boolean_mask_ops.cc)

---



## BooleanUnmask


Given a series of mask and values, reconstruct values together according to masks.
 A comprehensive example:  

```
  mask1   = True, False, True, False, False
  values1 = 1.0, 3.0
  mask2   = False, True, False, False, False
  values2 = 2.0
  mask3   = False, False, False, True, True
  values3 = 4.0, 5.0

```

 Reconstruct by:  

```
  output = net.BooleanUnmask([mask1, values1, mask2, values2, mask3, values3], ["output"])

```

 We get:  

```
  output = 1.0, 2.0, 3.0, 4.0, 5.0

```

 Note that for all mask positions, there must be at least one True. If for a field there are multiple True's, we will accept the first value. For example:   Example 1:  

```
  mask1   = True, False
  values1 = 1.0
  mask2   = False, False
  values2 =

```

 This is not allowed:  

```
  output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"])

```

 Example 2:  

```
  mask1   = True, False
  values1 = 1.0
  mask2   = True, True
  values2 = 2.0, 2.0

  output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"])

```

 We get:  

```
  output = 1.0, 2.0
```




### Interface


---------- | ----------
*Outputs* | 
`unmasked_data` | The final reconstructed unmasked data


### Code


[caffe2/operators/boolean_unmask_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/boolean_unmask_ops.cc)

---



## BoxWithNMSLimit


Apply NMS to each class (except background) and limit the number of returned boxes.



### Interface


---------- | ----------
*Arguments* | 
`score_thresh` | (float) TEST.SCORE_THRESH
`nms` | (float) TEST.NMS
`detections_per_im` | (int) TEST.DEECTIONS_PER_IM
`soft_nms_enabled` | (bool) TEST.SOFT_NMS.ENABLED
`soft_nms_method` | (string) TEST.SOFT_NMS.METHOD
`soft_nms_sigma` | (float) TEST.SOFT_NMS.SIGMA
`soft_nms_min_score_thres` | (float) Lower bound on updated scores to discard boxes
*Inputs* | 
`scores` | Scores, size (count, num_classes)
`boxes` | Bounding box for each class, size (count, num_classes * 4)
`batch_splits` | Tensor of shape (batch_size) with each element denoting the number of RoIs/boxes belonging to the corresponding image in batch. Sum should add up to total count of scores/boxes.
*Outputs* | 
`scores` | Filtered scores, size (n)
`boxes` | Filtered boxes, size (n, 4)
`classes` | Class id for each filtered score/box, size (n)
`batch_splits` | Output batch splits for scores/boxes after applying NMS
`keeps` | Optional filtered indices, size (n)
`keeps_size` | Optional number of filtered indices per class, size (num_classes)


### Code


[caffe2/operators/box_with_nms_limit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/box_with_nms_limit_op.cc)

---



## Broadcast


Does a broadcast operation from the root node to every other node. The tensor on each node should have been pre-created with the same shape and data type.



### Interface


---------- | ----------
*Arguments* | 
`root` | (int, default 0) the root to run broadcast from.
*Inputs* | 
`comm_world` | The common world.
`X` | A tensor to be broadcasted.
*Outputs* | 
`X` | In-place as input 1.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## Cast


The operator casts the elements of a given input tensor to a data type specified by the 'to' argument and returns an output tensor of the same size in the converted type. The 'to' argument must be one of the data types specified in the 'DataType' enum field in the TensorProto message. If the 'to' argument is not provided or is not one of the enumerated types in DataType, Caffe2 throws an Enforce error.
 NOTE: Casting to and from strings is not supported yet.



### Interface


---------- | ----------
*Arguments* | 
`to` | The data type to which the elements of the input tensor are cast.Strictly must be one of the types from DataType enum in TensorProto
*Inputs* | 
`input` | Input tensor to be cast.
*Outputs* | 
`output` | Output tensor with the same shape as input with type specified by the 'to' argument


### Code


[caffe2/operators/cast_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cast_op.cc)

---



## Ceil


Ceil takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the ceil function, y = ceil(x), is applied to the tensor elementwise. Currently supports only float32.



### Interface


---------- | ----------
*Inputs* | 
`X` | ND input tensor
*Outputs* | 
`Y` | ND input tensor


### Code


[caffe2/operators/ceil_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/ceil_op.cc)

---



## ChannelBackpropStats


Given an input tensor in NCHW format, the gradient for the output of SpatialBN and the per-channel mean and inverse std var vectors for the input, computes the per-channel bias and scale gradient to be used during the backward pass for subsequent spatial batch normalization gradient calculation. Typically, the results of this op are subsequently reduced over multiple devices to obtain statistics over a larger batch size in cases where the batch size for a single model copy is too low to yield the full benefit of batch normalization. The resulting bias and scale can then be plugged back into SpatialBNGradient to get results over the larger batch size 


### Interface


---------- | ----------
*Inputs* | 
`X` | The input 4-dimensional tensor of shape NCHW
`mean` | The mean saved from the forward pass as a 1-dimensional tensor of size C.
`inv_std` | The saved inverse standard deviation as a 1-dimensional tensor of size C.
`output_grad` | Gradient for the output layer of SpatialBN, here used as input because we are on the backward pass
*Outputs* | 
`scale_grad` | Gradient for the scale vector
`bias_grad` | Gradient for the bias vector


### Code


[caffe2/operators/channel_backprop_stats_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/channel_backprop_stats_op.cc)

---



## ChannelShuffle

No documentation yet.


### Code


[caffe2/operators/channel_shuffle_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/channel_shuffle_op.cc)

---



## ChannelShuffleGradient

No documentation yet.


### Code


[caffe2/operators/channel_shuffle_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/channel_shuffle_op.cc)

---



## ChannelStats


Given an input tensor in NCHW format, computes the sum of all elements per channel and the sum of all elements squared per channel. These values can be reduced across multiple batches and used to obtain the mean and variance across the full set of batches. Using the new mean and variance as input to SpatialBN has the effect of changing the batch size over which SpatialBN is applied.



### Interface


---------- | ----------
*Inputs* | 
`X` | The input 4-dimensional tensor of shape NCHW
*Outputs* | 
`sum` | The output 1-dimensional tensor of size C containing the sum of elements of X per channel.
`sumsq` | The output 1-dimensional tensor of size C containing the sum of elements squared per channel.


### Code


[caffe2/operators/channel_stats_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/channel_stats_op.cc)

---



## CheckAtomicBool

Copy the value of an atomic<bool> to a bool


### Interface


---------- | ----------
*Inputs* | 
`atomic_bool` | Blob containing a unique_ptr<atomic<bool>>
*Outputs* | 
`value` | Copy of the value for the atomic<bool>


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)

---



## CheckCounterDone


If the internal count value <= 0, outputs true, otherwise outputs false, 


### Interface


---------- | ----------
*Inputs* | 
`counter` | A blob pointing to an instance of a counter.
*Outputs* | 
`done` | true if the internal count is zero or negative.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)

---



## CheckDatasetConsistency


Checks that the given data fields represents a consistent dataset under the schema specified by the  `fields`  argument. Operator fails if the fields are not consistent. If data is consistent, each field's data can be safely appended to an existing dataset, keeping it consistent.



### Interface


---------- | ----------
*Arguments* | 
`fields` | List of strings representing the string names in the formatspecified in the doc for CreateTreeCursor.
*Inputs* | 
`field_0` | Data for field 0.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## Checkpoint


The Checkpoint operator is similar to the Save operator, but allows one to save to db every few iterations, with a db name that is appended with the iteration count. It takes [1, infinity) number of inputs and has no output. The first input has to be a TensorCPU of type int and has size 1 (i.e. the iteration counter). This is determined whether we need to do checkpointing.



### Interface


---------- | ----------
*Arguments* | 
`absolute_path` | (int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace.
`db` | (string) a template string that one can combine with the iteration to create the final db name. For example, "/home/lonestarr/checkpoint_%08d.db"
`db_type` | (string) the type of the db.
`every` | (int, default 1) the checkpointing is carried out when (iter mod every) is zero.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)

---



## Clip


Clip operator limits the given input within an interval. The interval is specified with arguments 'min' and 'max'. They default to numeric_limits::lowest() and numeric_limits::max() respectively. The clipping operation can be done in in-place fashion too, where the input and output blobs are the same.



### Interface


---------- | ----------
*Arguments* | 
`min` | Minimum value, under which element is replaced by min
`max` | Maximum value, above which element is replaced by max
*Inputs* | 
`input` | Input tensor (Tensor<float>) containing elements to beclipped
`output` | Output tensor (Tensor<float>) containing clippedinput elements


### Code


[caffe2/operators/clip_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/clip_op.cc)

---



## ClipGradient

No documentation yet.


### Code


[caffe2/operators/clip_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/clip_op.cc)

---



## ClipTensorByScaling


 

```
    Clips the input tensor by scaling based on the input value and the threshold.
    The value is usually the (pre-computed) norm of the tensor. If the value is
    larger than the threshold, scaling would be performed in this way:

          tensor *= (threshold / value).

    This op could be used for gradient clipping.
```




### Interface


---------- | ----------
*Arguments* | 
`threshold` | Threshold to determine whether to scale down the tensor
*Inputs* | 
`input_tensor` | Tensor of floats to be clipped.
`val` | Value to be compared against the threshold
*Outputs* | 
`clipped` | Tensor of floats, which is the same size as the input tensor, representing the clipped tensor.


### Code


[caffe2/sgd/clip_tensor_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/clip_tensor_op.cc)

---



## CloneCommonWorld


Clones existing common world.



### Interface


---------- | ----------
*Inputs* | 
`existing_comm_world` | Existing common world to clone.
*Outputs* | 
`comm_world` | A common world for collective operations.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## CloseBlobsQueue

No documentation yet.


### Code


[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)

---



## CloseRebatchingQueue


Closes the Queue.



### Interface


---------- | ----------
*Inputs* | 
`queue` | object representing the queue


### Code


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/rebatching_queue_ops.cc)

---



## Col2Im

No documentation yet.


### Code


[caffe2/operators/im2col_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/im2col_op.cc)

---



## CollectAndDistributeFpnRpnProposals


Merge RPN proposals generated at multiple FPN levels and then distribute those proposals to their appropriate FPN levels for Faster RCNN.
An anchor at one FPN level may predict an RoI that will map to another level, hence the need to redistribute the proposals.
 Only inference is supported. To train, please use the original Python operator in Detectron.
 Inputs and outputs are examples only; if min/max levels change, the number of inputs and outputs, as well as their level numbering, will change.



### Interface


---------- | ----------
*Arguments* | 
`roi_canonical_scale` | (int) ROI_CANONICAL_SCALE
`roi_canonical_level` | (int) ROI_CANONICAL_LEVEL
`roi_max_level` | (int) ROI_MAX_LEVEL
`roi_min_level` | (int) ROI_MIN_LEVEL
`rpn_max_level` | (int) RPN_MAX_LEVEL
`rpn_min_level` | (int) RPN_MIN_LEVEL
`rpn_post_nms_topN` | (int) RPN_POST_NMS_TOP_N
*Inputs* | 
`rpn_rois_fpn2` | RPN proposals for FPN level 2, size (n x 5), format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn3` | RPN proposals for FPN level 3, size (n x 5), format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn4` | RPN proposals for FPN level 4, size (n x 5), format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn5` | RPN proposals for FPN level 5, size (n x 5), format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn6` | RPN proposals for FPN level 6, size (n x 5), format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_roi_probs_fpn2` | RPN objectness probabilities for FPN level 2, size (n). See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn3` | RPN objectness probabilities for FPN level 3, size (n). See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn4` | RPN objectness probabilities for FPN level 4, size (n). See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn5` | RPN objectness probabilities for FPN level 5, size (n). See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn6` | RPN objectness probabilities for FPN level 6, size (n). See rpn_roi_probs documentation from GenerateProposals.
*Outputs* | 
`rois` | Top proposals limited to rpn_post_nms_topN total, size (n x 5), format (image_index, x1, y1, x2, y2)
`rois_fpn2` | RPN proposals for ROI level 2, size (n x 5), format (image_index, x1, y1, x2, y2)
`rois_fpn3` | RPN proposals for ROI level 3, size (n x 5), format (image_index, x1, y1, x2, y2)
`rois_fpn4` | RPN proposals for ROI level 4, size (n x 5), format (image_index, x1, y1, x2, y2)
`rois_fpn5` | RPN proposals for ROI level 5, size (n x 5), format (image_index, x1, y1, x2, y2)
`rois_idx_restore` | Permutation on the concatenation of all rois_fpni, i=min...max, such that when applied the RPN RoIs are restored to their original order in the input blobs.


### Code


[caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.cc)

---



## CollectTensor


Collect tensor into tensor vector by reservoir sampling, argument num_to_collect indicates the max number of tensors that will be collected. The first half of the inputs are tensor vectors, which are also the outputs. The second half of the inputs are the tensors to be collected into each vector (in the same order). The input tensors are collected in all-or-none manner. If they are collected, they will be placed at the same index in the output vectors.



### Interface


---------- | ----------
*Arguments* | 
`num_to_collect` | The max number of tensors to collect


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## ColwiseMax

Compute column-wise max reduction of the input tensor.


### Interface


---------- | ----------
*Inputs* | 
`X` | A tenosr of dimensions batch_size x M x N to compute colwise-max.
*Outputs* | 
`Y` | batch_size x N column-max results matrix.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_ops.cc)

---



## ColwiseMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_ops.cc)

---



## ComputeOffset


Compute the offsets matrix given cursor and data blobs. Need to be ran at beginning or after reseting cursor  Input(0) is a blob pointing to a TreeCursor, and [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 ComputeOffset is thread safe.



### Interface


---------- | ----------
*Inputs* | 
`cursor` | A blob containing a pointer to the cursor.
`dataset_field_0` | First dataset field
*Outputs* | 
`field_0` | Tensor containing offset info for this chunk.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## Concat

Concatenate a list of tensors into a single tensor


### Interface


---------- | ----------
*Arguments* | 
`axis` | Which axis to concat on
`order` | Either NHWC or NCHW, will concat on C axis, defaults to NCHW
`add_axis` | Pass 1 to add the axis specified in arg 'axis' to all input tensors
*Outputs* | 
`concat_result` | Concatenated tensor
`split_info` | The dimensions of the inputs.


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)

---



## ConcatTensorVector


Concat Tensors in the std::unique_ptr<std::vector<Tensor> > along the first dimension.
    


### Interface


---------- | ----------
*Inputs* | 
`vector of Tensor` | std::unique_ptr<std::vector<Tensor> >
*Outputs* | 
`tensor` | tensor after concatenating


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## Conditional


Given a 1-D tensor of boolean values, apply conditional operator along the first dimension of DataT and DataF and return DataO. Note, DataT and DataF must have the exact same shape and type.



### Interface


---------- | ----------
*Inputs* | 
`Condition` | Boolean tensor to select DataT or DataF
`DataT` | Data to use when True
`DataF` | Data to use when False
*Outputs* | 
`DataO` | Output data after applying ConditionalOp


### Code


[caffe2/operators/conditional_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conditional_op.cc)

---



## ConditionalSetAtomicBool


Set an atomic<bool> to true if the given condition bool variable is true     


### Interface


---------- | ----------
*Inputs* | 
`atomic_bool` | Blob containing a unique_ptr<atomic<bool>>
`condition` | Blob containing a bool


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)

---



## ConstantFill


The operator fills the elements of the output tensor with a constant value specified by the 'value' argument.
 The data type is specified by the 'dtype' argument. The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the TensorProto message. If the 'dtype' argument is not provided, the data type of 'value' is used.
 The output tensor shape is specified by the 'shape' argument. If the number of input is 1, the shape will be identical to that of the input at run time with optional additional dimensions appended at the end as specified by 'extra_shape' argument. In that case the 'shape' argument should not be set.
 If input_as_shape is set to true, then the input should be a 1D tensor containing the desired output shape (the dimensions specified in extra_shape will also be appended)  NOTE: Currently, it supports data type of float, int32, int64, and bool.



### Interface


---------- | ----------
*Arguments* | 
`value` | The value for the elements of the output tensor.
`dtype` | The data type for the elements of the output tensor.Strictly must be one of the types from DataType enum in TensorProto.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape.  First input must be in CPU context.
*Inputs* | 
`input` | Input tensor (optional) to provide shape information.
*Outputs* | 
`output` | Output tensor of constant values specified by 'value'argument and its type is specified by the 'dtype' argument


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## Conv


The convolution operator consumes an input vector, a filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is convolved with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: conv_op_impl.h is the templated implementation of the conv_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints. 
`filter` | The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the convolution; has size (M).
*Outputs* | 
`Y` | Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/conv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_op.cc)

---



## Conv1D


The convolution operator consumes an input vector, a 1D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is convolved with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: conv_op_impl.h is the templated implementation of the conv_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints. 
`filter` | The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the convolution; has size (M).
*Outputs* | 
`Y` | Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/conv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_op.cc)

---



## Conv1DGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## Conv2D


The convolution operator consumes an input vector, a 2D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is convolved with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: conv_op_impl.h is the templated implementation of the conv_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints. 
`filter` | The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the convolution; has size (M).
*Outputs* | 
`Y` | Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/conv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_op.cc)

---



## Conv2DGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## Conv3D


The convolution operator consumes an input vector, a 3D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is convolved with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: conv_op_impl.h is the templated implementation of the conv_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints. 
`filter` | The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the convolution; has size (M).
*Outputs* | 
`Y` | Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/conv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_op.cc)

---



## Conv3DGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## ConvGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## ConvTranspose


The transposed convolution consumes an input vector, the filter blob, and the bias blob, and computes the output. Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvTransposeUnpoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator.
As is expected, the filter is deconvolved with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: conv_transpose_op_impl.h is the templated implementation of the conv_transpose_op.h file, which is why they are separate files.
  


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints.
`filter` | The filter blob that will be used in the transposed convolution; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the convolution;has size (C). Optional, if not passed, will treat it as all 0.
*Outputs* | 
`Y` | Output data blob that contains the result of the transposed convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/conv_transpose_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_transpose_op.cc)

---



## ConvTransposeGradient

No documentation yet.


### Code


[caffe2/operators/conv_transpose_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_transpose_gradient_op.cc)

---



## Copy

Copy input tensor into output, potentially across devices.


### Interface


---------- | ----------
*Inputs* | 
`input` | The input tensor.
*Outputs* | 
`output` | Tensor that will contain a copy of the input.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## CopyFromCPUInput


Take a CPU input tensor and copy it to an output in the current Context (GPU or CPU). This may involves cross-device MemCpy.



### Interface


---------- | ----------
*Inputs* | 
`input` | The input CPU tensor.
*Outputs* | 
`output` | either a TensorCUDA or a TensorCPU


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## CopyOnDeviceLike

Copy input tensor into output to the specific device.


### Interface


---------- | ----------
*Inputs* | 
`input` | The input tensor.
`dst` | Tensor, on which device the copy will be performed.
*Outputs* | 
`output` | Tensor that will contain a copy of the input.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## Cos


Calculates the cosine of the given input tensor, element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The cosine of the input tensor computed element-wise


### Code


[caffe2/operators/cos_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cos_op.cc)

---



## CosGradient

No documentation yet.


### Code


[caffe2/operators/cos_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cos_op.cc)

---



## CosineEmbeddingCriterion


CosineEmbeddingCriterion takes two inputs: the similarity value and the label, and computes the elementwise criterion output as   

```
  output = 1 - s,               if y == 1
```

   

```
          max(0, s - margin),  if y == -1
```




### Interface


---------- | ----------
*Inputs* | 
`S` | The cosine similarity as a 1-dim TensorCPU.
`Y` | The label as a 1-dim TensorCPU with int value of 1 or -1.
*Outputs* | 
`loss` | The output loss with the same dimensionality as S.


### Code


[caffe2/operators/cosine_embedding_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cosine_embedding_criterion_op.cc)

---



## CosineEmbeddingCriterionGradient

No documentation yet.


### Code


[caffe2/operators/cosine_embedding_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cosine_embedding_criterion_op.cc)

---



## CosineSimilarity


Given two input float tensors X, Y, and produces one output float tensor of the cosine similarity between X and Y.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D or 2D input tensor
`Y` | 1D or 2D input tensor (must have the same shape as X)
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## CosineSimilarityGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## CountDown


If the internal count value > 0, decreases count value by 1 and outputs false, otherwise outputs true.



### Interface


---------- | ----------
*Inputs* | 
`counter` | A blob pointing to an instance of a counter.
*Outputs* | 
`done` | false unless the internal count is zero.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)

---



## CountUp


Increases count value by 1 and outputs the previous value atomically 


### Interface


---------- | ----------
*Inputs* | 
`counter` | A blob pointing to an instance of a counter.
*Outputs* | 
`previous_count` | count value BEFORE this operation


### Code


[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)

---



## CpuUtilizationReport

Report the delta in max CPU utilization observed so far in the             plan


### Interface


---------- | ----------
*Arguments* | 
`stats_name` | String name of the stat entry holding CPU utilization
*Inputs* | 
`utilization` | Delta in max CPU utilization observed, in percentage as a float value


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## CreateAtomicBool

Create an unique_ptr blob to hold an atomic<bool>


### Interface


---------- | ----------
*Outputs* | 
`atomic_bool` | Blob containing a unique_ptr<atomic<bool>>


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)

---



## CreateBlobsQueue

No documentation yet.


### Code


[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)

---



## CreateBlobsQueueDB

Create a DBReader from a BlobsQueue


### Interface


---------- | ----------
*Arguments* | 
`key_blob_index` | (default: -1 (no key)) index of blob for DB key in the BlobsQueue.
`value_blob_index` | (default: 0) index of blob for DB value in the BlobsQueue.
`timeout_secs` | (default: 0.0 (no timeout)) Timeout in seconds for reading from the BlobsQueue.
*Inputs* | 
`queue` | The shared pointer to a queue containing Blobs.
*Outputs* | 
`reader` | The DBReader for the given BlobsQueue


### Code


[caffe2/queue/blobs_queue_db.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/blobs_queue_db.cc)

---



## CreateCommonWorld


Creates a common world for communication operators.



### Interface


---------- | ----------
*Arguments* | 
`size` | (int) size of the common world.
`rank` | (int) rank of this node in the common world.
*Inputs* | 
`kv_handler` | Key/value handler for rendezvous (optional).
*Outputs* | 
`comm_world` | A common world for collective operations.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## CreateCounter


Creates a count-down counter with initial value specified by the 'init_count' argument.



### Interface


---------- | ----------
*Arguments* | 
`init_count` | Initial count for the counter, must be >= 0.
*Outputs* | 
`counter` | A blob pointing to an instance of a new counter.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)

---



## CreateDB

No documentation yet.


### Code


[caffe2/db/create_db_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/db/create_db_op.cc)

---



## CreateMap

Create an empty map blob


### Interface


---------- | ----------
*Arguments* | 
`key_dtype` | Key's TensorProto::DataType (default INT32)
`value_dtype` | Value's TensorProto::DataType (default INT32)
*Outputs* | 
`map blob` | Blob reference to the map


### Code


[caffe2/operators/map_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/map_ops.cc)

---



## CreateMutex

Creates an unlocked mutex and returns it in a unique_ptr blob.


### Interface


---------- | ----------
*Outputs* | 
`mutex_ptr` | Blob containing a std::unique_ptr<mutex>.


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/atomic_ops.cc)

---



## CreateRebatchingQueue


Creates the Queue.



### Interface


---------- | ----------
*Arguments* | 
`num_blobs` | Number of input tensors the queue will support
`capacity` | Maximal number of elements the queue can hold at any given point
*Outputs* | 
`queue` | object representing the queue


### Code


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/rebatching_queue_ops.cc)

---



## CreateScope


'CreateScope' operator initializes and outputs empty scope that is used by Do operator to store local blobs     


### Code


[caffe2/operators/create_scope_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/create_scope_op.cc)

---



## CreateTensorVector

Create a std::unique_ptr<std::vector<Tensor> >


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## CreateTextFileReader

Create a text file reader. Fields are delimited by <TAB>.


### Interface


---------- | ----------
*Arguments* | 
`filename` | Path to the file.
`num_passes` | Number of passes over the file.
`field_types` | List with type of each field. Type enum is found at core.DataType.
*Outputs* | 
`handler` | Pointer to the created TextFileReaderInstance.


### Code


[caffe2/operators/text_file_reader.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/text_file_reader.cc)

---



## CreateTreeCursor


Creates a cursor to iterate through a list of tensors, where some of those tensors contains the lengths in a nested schema. The schema is determined by the  `fields`  arguments.
 For example, to represent the following schema:   

```
  Struct(
      a=Int(),
      b=List(List(Int),
      c=List(
          Struct(
```

   

```
            c1=String,
```

   

```
            c2=List(Int),
          ),
      ),
  )

```

 the field list will be:  

```
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

```

 And for the following instance of the struct:   

```
  Struct(
      a=3,
      b=[[4, 5], [6, 7, 8], [], [9]],
      c=[
          Struct(c1='alex', c2=[10, 11]),
          Struct(c1='bob', c2=[12]),
      ],
  )

```

 The values of the fields will be:  

```
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

```

 In general, every field name in the format "{prefix}:lengths" defines a domain "{prefix}", and every subsequent field in the format "{prefix}:{field}" will be in that domain, and the length of the domain is provided for each entry of the parent domain. In the example, "b:lengths" defines a domain of length 4, so every field under domain "b" will have 4 entries.
The "lengths" field for a given domain must appear before any reference to that domain.
 Returns a pointer to an instance of the Cursor, which keeps the current offset on each of the domains defined by  `fields` . Cursor also ensures thread-safety such that ReadNextBatch and ResetCursor can be used safely in parallel.
 A cursor does not contain data per se, so calls to ReadNextBatch actually need to pass a list of blobs containing the data to read for each one of the fields.



### Interface


---------- | ----------
*Arguments* | 
`fields` | A list of strings each one representing a field of the dataset.
*Outputs* | 
`cursor` | A blob pointing to an instance of a new TreeCursor.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## CrossEntropy


Operator computes the cross entropy between the input and the label set. In  practice, it is most commonly used at the end of models, after the SoftMax  operator and before the AveragedLoss operator. Note that CrossEntropy  assumes that the soft labels provided is a 2D array of size N x D  (batch size x number of classes). Each entry in the 2D label corresponds to  the soft label for the input, where each element represents the correct  probability of the class being selected. As such, each element must be between  0 and 1, and all elements in an entry must sum to 1. The formula used is:   

```
                Y[i] = sum_j (label[i][j] * log(X[i][j]))

```

  where (i, j) is the classifier's prediction of the jth class (the correct one),  and i is the batch size. Each log has a lower limit for numerical stability.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input blob from the previous layer, which is almost always the result of a softmax operation; X is a 2D array of size N x D, where N is the batch size and D is the number of classes
`label` | Blob containing the labels used to compare the input
*Outputs* | 
`Y` | Output blob after the cross entropy computation


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## CrossEntropyGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## DBExists


Checks if the DB exists.



### Interface


---------- | ----------
*Arguments* | 
`absolute_path` | (int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace.
`db_name` | (string) the path to the db to load.
`db_type` | (string) the type of the db.
*Outputs* | 
`exists` | A scalar bool Tensor.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)

---



## DepthConcat

Backward compatible operator name for Concat.


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)

---



## DepthSplit

Backward compatible operator name for Split.


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)

---



## DequeueBlobs


 

```
  Dequeue the blobs from queue.
```

   


### Interface


---------- | ----------
*Arguments* | 
`timeout_secs` | Timeout in secs, default: no timeout
*Inputs* | 
`queue` | The shared pointer for the BlobsQueue
*Outputs* | 
`blob` | The blob to store the dequeued data


### Code


[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)

---



## DequeueRebatchingQueue


Dequeue Tensors from the Queue.
If the Queue is closed this might return less elements than asked.
If num_elements > 1 the returned elements will be concatenated into one tensor per component.



### Interface


---------- | ----------
*Arguments* | 
`num_elements` | Number of elements to dequeue. By default we dequeue one element.
*Inputs* | 
`rebatching_queue` | object representing the queue
`tensor` | First tensor to enqueue


### Code


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/rebatching_queue_ops.cc)

---



## DestroyCommonWorld

Closes all connections managed by a common world.


### Interface


---------- | ----------
*Inputs* | 
`common_world` | The common world to be destroyed.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## DiagonalFill


The operator fills the diagonal elements of the output tensor (>= 2D) with a constant value specified by the 'value' argument, and others 0. If number of dimensions of the output tensor is greater than 2, all dimensions must be equal.
 The data type is specified by the 'dtype' argument. The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the TensorProto message. If the 'dtype' argument is not provided, the data type of 'value' is used.
 The output tensor shape is specified by the 'shape' argument. If the number of input is 1, the shape will be identical to that of the input at run time with optional additional dimensions appended at the end as specified by 'extra_shape' argument. In that case the 'shape' argument should not be set.
 If input_as_shape is set to true, then the input should be a 1D tensor containing the desired output shape (the dimensions specified in extra_shape will also be appended)  NOTE: Currently, it supports data type of float, int32, int64, and bool.



### Interface


---------- | ----------
*Arguments* | 
`value` | The value for the elements of the output tensor.
`dtype` | The data type for the elements of the output tensor.Strictly must be one of the types from DataType enum in TensorProto.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape
*Inputs* | 
`input` | Input tensor (optional) to provide shape information.
*Outputs* | 
`output` | Output tensorargument and its type is specified by the 'dtype' argument


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## Div


Performs element-wise binary division (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and type as A


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## DivGradient

No documentation yet.


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## Do


'Do' control operator, executes a subnet in a separate workspace.
Last blobs in the input and output lists should be the same blob created with CreateScope op. Arguments 'inner_blobs' and 'outer_blobs_idx' provide a mapping between selected inner blob names and corresponding outer blob indices.
    


### Interface


---------- | ----------
*Arguments* | 
`net` | Subnet with blob bindings
`inner_blobs` | List of inner net blob names to bind to outer workspace
`outer_blobs_idx` | Indices of corresponding outer workspace blobs, in order: operator inputs, operator outputs (skipping workspace blobs)
`saved_fwd_blobs` | List of blobs from the forward Do operator workspace needed in backward pass, used in gradient Do operator
`reuse_workspace` | Whether to reuse workspace or create a new one in a given scope


### Code


[caffe2/operators/do_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/do_op.cc)

---



## DotProduct


Given two input float tensors X, Y, and produces one output float tensor of the dot product between X and Y.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D or 2D input tensor
`Y` | 1D or 2D input tensor (must have the same shape as X)
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## DotProductGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## DotProductWithPadding


Given two input float tensors X, Y with different shapes and produces one output float tensor of the dot product between X and Y. We currently support two kinds of strategies to achieve this. Before doing normal dot_product 1) pad the smaller tensor (using pad_value) to the same shape as the other one.
2) replicate the smaller tensor to the same shape as the other one. Note the first dimension of X, Y must be equal. Only the second dimension of X or Y can be padded.



### Interface


---------- | ----------
*Arguments* | 
`pad_value` | the padding value for tensors with smaller dimension
`replicate` | whether to replicate the smaller tensor or not
*Inputs* | 
`X` | 1D or 2D input tensor
`Y` | 1D or 2D input tensor
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## DotProductWithPaddingGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## Dropout


Dropout takes one input data (Tensor<float>) and produces two Tensor outputs, output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in test mode or not, the output Y will either be a random dropout, or a simple copy of the input. Note that our implementation of Dropout does scaling in the training phase, so during testing nothing needs to be done.



### Interface


---------- | ----------
*Arguments* | 
`ratio` | (float, default 0.5) the ratio of random dropout
`is_test` | (int) if nonzero, run dropout in test mode where the output is simply Y = X.
*Inputs* | 
`data` | The input data as Tensor.
*Outputs* | 
`output` | The output.
`mask` | The output mask. If is_test is nonzero, this output is not filled.


### Code


[caffe2/operators/dropout_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dropout_op.cc)

---



## DropoutGrad

No documentation yet.


### Code


[caffe2/operators/dropout_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dropout_op.cc)

---



## EQ


Performs element-wise equality comparison  `==`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## ElementwiseLinear


Given inputs X of size (N x D), w of size D and b of size D, the op computes Y of size (N X D) where Y_{nd} = X_{nd} * w_d + b_d   


### Interface


---------- | ----------
*Arguments* | 
`axis` | default to 1; describes the axis of the inputs; defaults to one because the 0th axis most likely describes the batch_size
*Inputs* | 
`X` | 2D input tensor of size (N X D) data
`w` | 1D scaling factors of size D
`b` | 1D biases of size D
*Outputs* | 
`Y` | 2D output tensor


### Code


[caffe2/operators/elementwise_linear_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_linear_op.cc)

---



## ElementwiseLinearGradient

No documentation yet.


### Code


[caffe2/operators/elementwise_linear_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_linear_op.cc)

---



## Elu


 Elu takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the function  `f(x) = alpha * (exp(x) - 1.) for x < 0` ,  `f(x) = x for x >= 0` ., is applied to the tensor elementwise.
 


### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/elu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elu_op.cc)

---



## EluGradient


EluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.



### Code


[caffe2/operators/elu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elu_op.cc)

---



## EnqueueBlobs

No documentation yet.


### Code


[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)

---



## EnqueueRebatchingQueue


Enqueues Tensors into the queue.
Number of input tensors should be equal to the number of components passed during creation of the queue.
If the Queue is closed this operation will fail.
If enqueue_batch argument is set. We will split the input tensors by the first dimension to produce single queue elements.



### Interface


---------- | ----------
*Arguments* | 
`enqueue_batch` | Are we enqueuing a batch or just a single element.         By default we enqueue single element.
*Inputs* | 
`queue` | object representing the queue
`tensor` | First tensor to enque. 


### Code


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/rebatching_queue_ops.cc)

---



## EnsureCPUOutput


Take an input tensor in the current Context (GPU or CPU) and create an output which is always a TensorCPU. This may involves cross-device MemCpy.



### Interface


---------- | ----------
*Inputs* | 
`input` | The input CUDA or CPU tensor.
*Outputs* | 
`output` | TensorCPU that is a copy of the input.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## EnsureDense


This operator converts dense or sparse gradients to dense ones.
Therefore, sparse gradient can be back propagated to Operators that consume dense gradients only (e.g., FCGradient).
 The operator's behaviors:  - In forward, simply pass in place or copy input to the output.
- In backward, if the gradient passed-in is sparse gradient, change it to dense gradient in linear time; otherwise, simply pass the dense gradient.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensors.
*Outputs* | 
`output` | Output tensor. Same dimension as inputs.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## Exp


Calculates the exponential of the given input tensor, element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The exponential of the input tensor computed element-wise


### Code


[caffe2/operators/exp_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/exp_op.cc)

---



## ExpandDims


Insert single-dimensional entries to the shape of a tensor.
Takes one required argument  `dims` , a list of dimensions that will be inserted.
Dimension indices in  `dims`  are as seen in the output tensor. For example:   

```
  Given a tensor such that tensor.Shape() = [3, 4, 5], then
  ExpandDims(tensor, dims=[0, 4]).Shape() == [1, 3, 4, 5, 1])

```

 If the same blob is provided in input and output, the operation is copy-free.



### Interface


---------- | ----------
*Inputs* | 
`data` | Original tensor
*Outputs* | 
`expanded` | Reshaped tensor with same data as input.


### Code


[caffe2/operators/expand_squeeze_dims_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/expand_squeeze_dims_op.cc)

---



## ExtendTensor


Extend input 0 if necessary based on max element in input 1.
Input 0 must be the same as output, that is, it is required to be in-place.
Input 0 may have to be re-allocated in order for accommodate to the new size.
Currently, an exponential growth ratio is used in order to ensure amortized constant time complexity.
All except the outer-most dimension must be the same between input 0 and 1.



### Interface


---------- | ----------
*Inputs* | 
`tensor` | The tensor to be extended.
`new_indices` | The size of tensor will be extended based on max element in new_indices.
*Outputs* | 
`extended_tensor` | Same as input 0, representing the mutated tensor.


### Code


[caffe2/operators/extend_tensor_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/extend_tensor_op.cc)

---



## FC


Computes the result of passing an input vector X into a fully connected layer with 2D weight matrix W and 1D bias vector b. That is, the layer computes Y = X * W^T + b, where X has size (M x K), W has size (N x K), b has size (N), and Y has size (M x N), where M is often the batch size.
  NOTE: X does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor X \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}] where a_i \in N+ and k is the axis provided, then X will be coerced into a 2-dimensional tensor with dimensions [a_0  * ... *  a_{k-1}, a_k  * ... *  a_{n-1}]. For the default case where axis=1, this means the X tensor will be coerced into a 2D tensor of dimensions [a_0, a_1  * ... *  a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = M and a_1  * ... *  a_{n-1} = K.
Lastly, even though b is a 1D vector of size N, it is copied/resized to be size (M x N) implicitly and added to each vector in the batch.
Each of these dimensions must be matched correctly, or else the operator will throw errors.



### Interface


---------- | ----------
*Arguments* | 
`axis` | (int32_t) default to 1; describes the axis of the inputs; defaults to one because the 0th axis most likely describes the batch_size
`axis_w` | (int32_t) default to 1; describes the axis of the weight matrix W; defaults to one because the 0th axis most likely describes the batch_size
`float16_compute` | Whether to use float-16 compute kernel
*Inputs* | 
`X` | input tensor that's coerced into a 2D matrix of size (MxK) as described above
`W` | A tensor that is coerced into a 2D blob of size (KxN) containing fully connected weight matrix
`b` | 1D blob containing bias vector
*Outputs* | 
`Y` | 2D output tensor


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)

---



## FCGradient

No documentation yet.


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)

---



## FCTransposed


Same as FC, but weight matrix is supposed to be already pretransposed.
FCTransposed stands for calling blass with no noTrans, noTrans 


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)

---



## FCTransposedGradient

No documentation yet.


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)

---



## FeedBlob


FeedBlobs the content of the blobs. The input and output blobs should be one-to-one inplace.


### Interface


---------- | ----------
*Arguments* | 
`value` | (string) if provided then we will use this string as the value for theprovided output tensor


### Code


[caffe2/operators/feed_blob_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/feed_blob_op.cc)

---



## FileStoreHandlerCreate


Creates a unique_ptr<StoreHandler> that uses the filesystem as backing store (typically a filesystem shared between many nodes, such as NFS).
This store handler is not built to be fast. Its recommended use is for integration tests and prototypes where extra dependencies are cumbersome. Use an ephemeral path to ensure multiple processes or runs don't interfere.



### Interface


---------- | ----------
*Arguments* | 
`path` | base path used by the FileStoreHandler
`prefix` | prefix for all keys used by this store
*Outputs* | 
`handler` | unique_ptr<StoreHandler>


### Code


[caffe2/distributed/file_store_handler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/distributed/file_store_handler_op.cc)

---



## Find


Finds elements of second input from first input, outputting the last (max) index for each query.
If query not find, inserts missing_value.
See IndexGet() for a version that modifies the index when values are not found.



### Interface


---------- | ----------
*Arguments* | 
`missing_value` | Placeholder for items that are not found
*Inputs* | 
`index` | Index (integers)
`query` | Needles / query
*Outputs* | 
`query_indices` | Indices of the needles in index or 'missing value'


### Code


[caffe2/operators/find_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_op.cc)

---



## FindDuplicateElements


Shrink the data tensor by removing data blocks with given zero-based indices in the outermost dimension of the tensor. Indices are not assumed in any order or unique but with the range [0, blocks_size). Indices could be empty.
  


### Interface


---------- | ----------
*Inputs* | 
`data` | a 1-D tensor.
*Outputs* | 
`indices` | indices of duplicate elements in data, excluding first occurrences.


### Code


[caffe2/operators/find_duplicate_elements_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.cc)

---



## Flatten


Flattens the input tensor into a 2D matrix. If input tensor has shape (d_0, d_1, ... d_n) then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn) 


### Interface


---------- | ----------
*Arguments* | 
`axis` | (Default to 1) Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output
*Inputs* | 
`input` | A tensor of rank >= axis.
*Outputs* | 
`output` | A 2D tensor with the contents of the input tensor, with input dimensions up to axis flattened to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output.


### Code


[caffe2/operators/flatten_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/flatten_op.cc)

---



## FlattenToVec


Flattens the input tensor into a 1D vector.



### Interface


---------- | ----------
*Inputs* | 
`input` | A tensor of rank >= 1.
*Outputs* | 
`output` | A tensor of rank 1 with the contents of the input tensor


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## FlexibleTopK


Given two tensors: X and K, retrieve the top K[..., 1] elements from X on the last dimension.
X is an input tensor of shape [a_1, a_2, ..., a_n, r].
K is an input tensor of shape [a_1, a_2, ..., a_n, 1], where for each element, r >= K[..., 1] > 0 Output two outputs: -Flatten values tensor of shape [ \sum_i K[i, 1] ] which contains the values of  the top K[..., 1] 

```
  elements along the last dimension
```

 -Flatten indices tensor of shape [ \sum_i K[i, 1] ] which contains the indices  of the top K[..., 1] 

```
  elements, flatten indices from the input tensor).
```

 These two outputs should be used with the input K, so that we know which indices in X are picked.
 Given two equivalent values, this operator uses the indices along the last dim- ension as a tiebreaker. That is, the element with the lower index will appear first.
    


### Interface


---------- | ----------
*Inputs* | 
`X` | Tensor of shape [a_1, a_2, ..., a_n, r]
`K` | Tensor of shape [a_1, a_2, ..., a_n, 1]
*Outputs* | 
`Flatten values` | Tensor of shape [ \sum_i K[i, 1] ] containing top K[..., 1] values from the input tensor
`Flatten indices` | Tensor of shape [ \sum_i K[i, 1] ] containing the indices into the flatten input


### Code


[caffe2/operators/flexible_top_k.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/flexible_top_k.cc)

---



## FlexibleTopKGradient

No documentation yet.


### Code


[caffe2/operators/flexible_top_k.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/flexible_top_k.cc)

---



## FloatToFused8BitRowwiseQuantized


Applies 8-bit row-wise quantization by determining the range (maximum - minimum) and offset (minimum value) of each row in the input matrix, and then scaling each element to an 8-bit number between 0 and 255. To later de-quantize values, the scale (range / 255) and offset (bias) are stored alongside the data. More precisely, the first 4 bytes of each row in the output matrix are a 32-bit float storing the scale, the next 4 bytes store the bias as a 32-bit float, and all remaining bytes in the row encode single quantized values.) 


### Interface


---------- | ----------
*Inputs* | 
`input` | Float32 input data
*Outputs* | 
`output` | Fused scale, bias and quantized data


### Code


[caffe2/operators/fused_rowwise_8bit_conversion_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fused_rowwise_8bit_conversion_ops.cc)

---



## FloatToRowwiseQuantized8Bits


This operator applies 8Bit row-wise quantization to input tensor and returns quantized tensor. Row wise quantization of input tensor is the following process. We take tensor of size (m_1, m_2,...,m_n), n >= 2, reshape it into matrix of size (m_1, m_2 x... x m_n) and apply row-wise quantization. After this, we compute scale_i= (min_i - max_i) / 255 and 

```
  bias_i = min_i for
```

 i-th row r_i of reshaped matrix, where min_i and max_i -- 

```
  minimum
```

 and maximum elements of i-th row, and quantize each element r_{ij} as 0 <= round(r_ij - bias_i) / scale_i) < 256. Instead of input tensor we obtain uint8 tensor and auxiliary information as scale and bias to restore input tensor (with losses).



### Interface


---------- | ----------
*Inputs* | 
`input` | input
*Outputs* | 
`quantized_input` | quantized_input
`scale_bias` | Matrix of floats, each row r_i of which stores a pair s_i, b_i


### Code


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## Floor


Floor takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the floor function, y = floor(x), is applied to the tensor elementwise. Currently supports only float32.



### Interface


---------- | ----------
*Inputs* | 
`X` | ND input tensor
*Outputs* | 
`Y` | ND input tensor


### Code


[caffe2/operators/floor_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/floor_op.cc)

---



## Free


Frees the content of the blobs. The input and output blobs should be one-to-one inplace.


### Code


[caffe2/operators/free_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/free_op.cc)

---



## Ftrl

No documentation yet.


### Code


[caffe2/sgd/ftrl_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/ftrl_op.cc)

---



## Fused8BitRowwiseQuantizedToFloat


De-quantizes the result of the FloatToFused8BitRowwiseQuantized operator. The input is expected to encode the scale as a 32-bit float in the second to the last 4 bytes of each row, followed by the bias as a 32-bit float in the next 4 bytes, and the quantized values in the preceding bytes of the row. The output is a matrix containing only the values, but de-quantized. De-quantization is performed by multiplying each value by its row's scale and bias parameters. The de-quantized values will thus not be exactly equal to the original, un-quantized floating point values.



### Interface


---------- | ----------
*Inputs* | 
`scale_bias_quantized_input` | Fused scale, bias and quantized data
*Outputs* | 
`float_input` | Float32 data


### Code


[caffe2/operators/fused_rowwise_8bit_conversion_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fused_rowwise_8bit_conversion_ops.cc)

---



## GE


Performs element-wise greater or equal than comparison  `>=`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## GRUUnit


GRUUnit computes the activations of a standard GRU, in a sequence-length aware fashion.
 Concretely, given the (fused) inputs X (TxNxD), the previous hidden state (NxD), and the sequence lengths (N), computes the GRU activations, avoiding computation if the input is invalid (as in, the value at X[t][n] >= seqLengths[n].
 


### Interface


---------- | ----------
*Arguments* | 
`drop_states` | Bool to determine if hidden state is zeroes or passed along for timesteps past the given sequence_length.
`sequence_lengths` | When false, the sequence lengths input is left out, and all following inputs are shifted left by one.
*Outputs* | 
`hidden` | The new GRU hidden state calculated by this op.


### Code


[caffe2/operators/gru_unit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gru_unit_op.cc)

---



## GRUUnitGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`sequence_lengths` | When false, the sequence lengths input is left out, and all following inputs are shifted left by one.


### Code


[caffe2/operators/gru_unit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gru_unit_op.cc)

---



## GT


Performs element-wise greater than comparison  `>`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## Gather


Given DATA tensor of rank r >= 1, and INDICES tensor of rank q, gather entries of the outer-most dimension of DATA indexed by INDICES, and concatenate them in an output tensor of rank q + (r - 1).
 Example:  

```
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
```




### Interface


---------- | ----------
*Inputs* | 
`DATA` | Tensor of rank r >= 1.
`INDICES` | Tensor of int32/int64 indices, of any rank q.
*Outputs* | 
`OUTPUT` | Tensor of rank q + (r - 1).


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## GatherByKey


Inverse operation of Partition.
 Takes the original, full 'keys' tensor followed by sharded value tensors, and returns the full value tensor, combined using the same hash used in Partition.



### Interface


---------- | ----------
*Inputs* | 
`keys` | The first input is the full keys tensor (same as the first input of Partition).
`sharded_values` | Subsequented inputs are sharded values tensors.
*Outputs* | 
`values` | Reconstructed values tensor.


### Code


[caffe2/operators/partition_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/partition_ops.cc)

---



## GatherFused8BitRowwise


Perform the same operation as Gather, but operating on 8-bit rowwise quantized matrices with fused storage (where each row stores quantized values, and then the scale and offset).
DATA needs to have rank 2 and INDICES needs to have rank 1.



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor with rank 2 obtained with operator FloatToFused8BitRowwiseQuantized
`INDICES` | Integer vector containing indices of the first dimension of DATA forthe rows that are being gathered
*Outputs* | 
`OUTPUT` | output


### Code


[caffe2/operators/gather_fused_8bit_rowwise_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_fused_8bit_rowwise_op.cc)

---



## GatherPadding


Gather the sum of start and end paddings in a padded input sequence. Used in order to compute the gradients of AddPadding w.r.t the padding tensors.



### Interface


---------- | ----------
*Arguments* | 
`padding_width` | Outer-size of padding present around each range.
`end_padding_width` | (Optional) Specifies a different end-padding width.
*Inputs* | 
`data_in` | T<N, D1..., Dn> Padded input data
`lengths` | (i64) Num of elements in each range. sum(lengths) = N. If not provided, considers all data as a single segment.
*Outputs* | 
`padding_sum` | Sum of all start paddings, or of all paddings if end_padding_sum is not provided.
`end_padding_sum` | T<D1..., Dn> Sum of all end paddings, if provided.


### Code


[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)

---



## GatherRanges


Given DATA tensor of rank 1, and RANGES tensor of rank 3, gather corresponding ranges into a 1-D tensor OUTPUT.
 RANGES dimentions description: 1: represents list of examples within a batch 2: represents list features 3: two values which are start and length or a range (to be applied on DATA)  Another output LENGTHS represents each example length within OUTPUT  Example:  

```
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
```




### Interface


---------- | ----------
*Inputs* | 
`DATA` | Tensor of rank 1.
`RANGES` | Tensor of int32/int64 ranges, of dims (N, M, 2). Where N is number of examples and M is a size of each example. Last dimension represents a range in the format (start, lengths)
*Outputs* | 
`OUTPUT` | 1-D tensor of size sum of range lengths
`LENGTHS` | 1-D tensor of size N with lengths over gathered data for each row in a batch. sum(LENGTHS) == OUTPUT.size()


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## GatherRangesToDense


Given DATA tensor of rank 1, and RANGES tensor of rank 3, gather values corresponding to each range into a separate output tensor. If the optional input KEY tensor is also given, the output will be sorted by KEY for each example.
 RANGES dimensions description: 1: represents list of examples within a batch 2: represents list features 3: two values which are start and length or a range (to be applied on DATA)  Each feature has fixed lengths which are passed as lengths argument and a separate tensor will be produced for each feature.
i.e. DATA.dim(1) = len(lengths) = NumOuptuts.
 Missing features (represented by empty ranges) filled with default_value.
 Example 1:  

```
  DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
  RANGES = [
    [
      [2, 4],
      [0, 2],
    ],
    [
      [0, 0],
      [6, 2],
    ]
  ]
  lengths = [4, 2]
  OUTPUT[0] = [[3, 4, 5, 6], [0, 0, 0, 0]]
  OUTPUT[1] = [[1, 2], [7, 8]]

```

 Example 2 (with KEY): DATA 

```
  = [1, 2, 3, 4, 5, 6, 7, 8]
```

 KEY  

```
  = [0, 1, 3, 2, 1, 0, 1, 0]
```

 RANGES = [  

```
  [
    [2, 4],
    [0, 2],
  ],
  [
    [0, 0],
    [6, 2],
  ]
```

 ] lengths = [4, 2] OUTPUT[0] = [[6, 5, 4, 3], [0, 0, 0, 0]] OUTPUT[1] = [[1, 2], [8, 7]]  Contrast Example 2 with Example 1. For each data point per feature, the values are sorted by the corresponding KEY.



### Interface


---------- | ----------
*Arguments* | 
`lengths` | Expected lengths for ranges
*Inputs* | 
`DATA` | Tensor of rank 1.
`RANGES` | Tensor of int32/int64 ranges, of dims (N, M, 2). Where N is number of examples and M is a size of each example. Last dimention represents a range in the format (start, lengths)
`KEY` | Tensor of rank 1 and type int64.
*Outputs* | 
`OUTPUT` | 1-D tensor of size sum of range lengths


### Code


[caffe2/operators/gather_ranges_to_dense_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_ranges_to_dense_op.cc)

---



## GaussianFill

No documentation yet.


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## GenerateProposals


Generate bounding box proposals for Faster RCNN. The propoasls are generated for a list of images based on image score 'score', bounding box regression result 'deltas' as well as predefined bounding box shapes 'anchors'. Greedy non-maximum suppression is applied to generate the final bounding boxes.



### Interface


---------- | ----------
*Arguments* | 
`spatial_scale` | (float) spatial scale
`pre_nms_topN` | (int) RPN_PRE_NMS_TOP_N
`post_nms_topN` | (int) RPN_POST_NMS_TOP_N
`nms_thresh` | (float) RPN_NMS_THRESH
`min_size` | (float) RPN_MIN_SIZE
*Inputs* | 
`scores` | Scores from conv layer, size (img_count, A, H, W)
`bbox_deltas` | Bounding box deltas from conv layer, size (img_count, 4 * A, H, W)
`im_info` | Image info, size (img_count, 3), format (height, width, scale)
`anchors` | Bounding box anchors, size (A, 4)
*Outputs* | 
`rois` | Proposals, size (n x 5), format (image_index, x1, y1, x2, y2)
`rois_probs` | scores of proposals, size (n)


### Code


[caffe2/operators/generate_proposals_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/generate_proposals_op.cc)

---



## GenerateProposalsCPP

No documentation yet.


### Code


[caffe2/operators/generate_proposals_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/generate_proposals_op.cc)

---



## GetAllBlobNames


Return a 1D tensor of strings containing the names of each blob in the active workspace.



### Interface


---------- | ----------
*Arguments* | 
`include_shared` | (bool, default true) Whether to include blobs inherited from parent workspaces.
*Outputs* | 
`blob_names` | 1D tensor of strings containing blob names.


### Code


[caffe2/operators/workspace_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/workspace_ops.cc)

---



## GivenTensorBoolFill

No documentation yet.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorDoubleFill

No documentation yet.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorFill

No documentation yet.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorInt64Fill

No documentation yet.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorIntFill

No documentation yet.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorStringFill

No documentation yet.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## Glu


Applies gated linear unit to the input Tensor X. The output Y is half the size of the input X, so if the shape of X is [d1, d2, ..., N] shape of Y will be [d1, d2, ..., dn/2] and Y(:dn-1, i) = GLU(X(:dn-1, i), X(:dn-1, i+N/2)) = X(dn-1, i) * sigmoid(X(dn-1, i+N/2)) 


### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D output tensor


### Code


[caffe2/operators/glu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/glu_op.cc)

---



## HSoftmax


Hierarchical softmax is an operator which approximates the softmax operator while giving significant training speed gains and reasonably comparable performance. In this operator, instead of calculating the probabilities of all the classes, we calculate the probability of each step in the path from root to the target word in the hierarchy.
 The operator takes a 2-D tensor (Tensor<float>) containing a batch of layers, a set of parameters represented by the weight matrix and bias terms, and a 1-D tensor (Tensor<int>) holding labels, or the indices of the target class. The hierarchy has to be specified as an argument to the operator.
 The operator returns a 1-D tensor holding the computed log probability of the target class and a 2-D tensor of intermediate outputs (from the weight matrix and softmax from each step in the path from root to target class) which will be used by the gradient operator to compute gradients for all samples in the batch.



### Interface


---------- | ----------
*Arguments* | 
`hierarchy` | Serialized HierarchyProto string containing list of vocabulary words and their paths from root of hierarchy to the leaf
*Inputs* | 
`X` | Input data from previous layer
`W` | 2D blob containing 'stacked' fully connected weight matrices. Each node in the hierarchy contributes one FC weight matrix if it has children nodes. Dimension is N*D, D is input dimension of data (X), N is sum of all output dimensions, or total number of nodes (excl root)
`b` | 1D blob with N parameters
`labels` | int word_id of the target word
*Outputs* | 
`Y` | 1-D of log probability outputs, one per sample
`intermediate_output` | Extra blob to store the intermediate FC and softmax outputs for each node in the hierarchical path of a word. The outputs from samples are stored in consecutive blocks in the forward pass and are used in reverse order in the backward gradientOp pass


### Code


[caffe2/operators/h_softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/h_softmax_op.cc)

---



## HSoftmaxGradient

No documentation yet.


### Code


[caffe2/operators/h_softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/h_softmax_op.cc)

---



## HSoftmaxSearch


HSoftmaxSearch is an operator to generate the most possible paths given a well-trained model and input vector. Greedy algorithm is used for pruning the search tree.



### Interface


---------- | ----------
*Arguments* | 
`tree` | Serialized TreeProto string containing a tree including all intermidate nodes and leafs. All nodes must have names for correct outputs
`beam` | beam used for pruning tree. The pruning algorithm is that only children, whose score is smaller than parent's score puls beam, will be propagated. 
`topN` | Number of nodes in outputs
*Inputs* | 
`X` | Input data from previous layer
`W` | The matrix trained from Softmax Ops
`b` | The bias traiend from Softmax Ops
*Outputs* | 
`Y_names` | The name of selected nodes and leafs. For nodes, it will be the name defined in the tree. For leafs, it will be the index of the word in the tree.
`Y_scores` | The corresponding scores of Y_names


### Code


[caffe2/operators/h_softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/h_softmax_op.cc)

---



## HasElements

Returns true iff the input tensor has size > 0


### Interface


---------- | ----------
*Inputs* | 
`tensor` | Tensor of any type.
*Outputs* | 
`has_elements` | Scalar bool tensor. True if input is not empty.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## HasScope


Checks whether scope blob has any saved scopes left     


### Code


[caffe2/operators/create_scope_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/create_scope_op.cc)

---



## HuffmanTreeHierarchy


HuffmanTreeHierarchy is an operator to generate huffman tree hierarchy given the input labels. It returns the tree as seralized HierarchyProto 


### Interface


---------- | ----------
*Arguments* | 
`num_classes` | The number of classes used to build the hierarchy.
*Inputs* | 
`Labels` | The labels vector
*Outputs* | 
`Hierarch` | Huffman coding hierarchy of the labels


### Code


[caffe2/operators/h_softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/h_softmax_op.cc)

---



## If


'If' control operator, first input is a scalar boolean blob that stores condition value. Accepts 'then_net' (required) and 'else_net' (optional) arguments for 'then' and 'else' subnets respectively. Subnets are executed in the same workspace as 'If'.
    


### Interface


---------- | ----------
*Arguments* | 
`then_net` | Net executed when condition is true
`else_net` | Net executed when condition is false (optional)
*Inputs* | 
`condition` | Scalar boolean condition


### Code


[caffe2/operators/if_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/if_op.cc)

---



## Im2Col

The Im2Col operator from Matlab.


### Interface


---------- | ----------
*Inputs* | 
`X` | 4-tensor in NCHW or NHWC.
*Outputs* | 
`Y` | 4-tensor. For NCHW: N x (C x kH x kW) x outH x outW.For NHWC: N x outH x outW x (kH x kW x C


### Code


[caffe2/operators/im2col_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/im2col_op.cc)

---



## ImageInput


Imports and processes images from a database. For each run of the operator, batch_size images will be processed. GPUs can optionally be used for part of the processing.
 The following transformations are applied to the image  

```
  - A bounding box is applied to the initial image (optional)
  - The image is rescaled either up or down (with the scale argument) or
    just up (with the minsize argument)
  - The image is randomly cropped (crop size is passed as an argument but
    the location of the crop is random except if is_test is passed in which case
    the image in cropped at the center)
  - The image is normalized. Each of its color channels can have separate
    normalization values

```

 The dimension of the output image will always be cropxcrop 


### Interface


---------- | ----------
*Arguments* | 
`batch_size` | Number of images to output for each run of the operator. Must be 1 or greater
`color` | Number of color channels (1 or 3). Defaults to 1
`color_jitter` | Whether or not to do color jitter. Defaults to 0
`img_saturation` | Image saturation scale used in color jittering. Defaults to 0.4
`img_brightness` | Image brightness scale used in color jittering. Defaults to 0.4
`img_contrast` | Image contrast scale used in color jittering. Defaults to 0.4
`color_lighting` | Whether or not to do color lighting. Defaults to 0
`color_lighting_std` | Std of normal distribution where color lighting scaling factor is sampled. Defaults to 0.1
`scale_jitter_type` | Type 0: No scale jittering Type 1: Inception-style scale jittering
`label_type` | Type 0: single integer label for multi-class classification. Type 1: sparse active label indices for multi-label classification. Type 2: dense label embedding vector for label embedding regression
`scale` | Scale the size of the smallest dimension of the image to this. Scale and minsize are mutually exclusive. Must be larger than crop
`minsize` | Scale the size of the smallest dimension of the image to this only if the size is initially smaller. Scale and minsize are mutually exclusive. Must be larger than crop.
`warp` | If 1, both dimensions of the image will be set to minsize or scale; otherwise, the other dimension is proportionally scaled. Defaults to 0
`crop` | Size to crop the image to. Must be provided
`mirror` | Whether or not to mirror the image. Defaults to 0
`mean` | Mean by which to normalize color channels. Defaults to 0.
`mean_per_channel` | Vector of means per color channel  (1 or 3 elements). Defaults to mean argument. Channel order BGR
`std` | Standard deviation by which to normalize color channels. Defaults to 1.
`std_per_channel` | Vector of standard dev. per color channel  (1 or 3 elements). Defaults to std argument. Channel order is BGR
`bounding_ymin` | Bounding box coordinate. Defaults to -1 (none)
`bounding_xmin` | Bounding box coordinate. Defaults to -1 (none)
`bounding_height` | Bounding box coordinate. Defaults to -1 (none)
`bounding_width` | Bounding box coordinate. Defaults to -1 (none)
`is_test` | Set to 1 to do deterministic cropping. Defaults to 0
`use_caffe_datum` | 1 if the input is in Caffe format. Defaults to 0
`use_gpu_transform` | 1 if GPU acceleration should be used. Defaults to 0. Can only be 1 in a CUDAContext
`decode_threads` | Number of CPU decode/transform threads. Defaults to 4
`output_type` | If gpu_transform, can set to FLOAT or FLOAT16.
`db` | Name of the database (if not passed as input)
`db_type` | Type of database (if not passed as input). Defaults to leveldb
`output_sizes` | The sizes of any outputs besides the data and label (should have a number of elements equal to the number of additional outputs)
`random_scale` | [min, max] shortest-side desired for image resize. Defaults to [-1, -1] or no random resize desired.
*Inputs* | 
`reader` | The input reader (a db::DBReader)
*Outputs* | 
`data` | Tensor containing the images
`label` | Tensor containing the labels
`additional outputs` | Any outputs after the first 2 will be Tensors read from the input TensorProtos


### Code


[caffe2/image/image_input_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/image/image_input_op.cc)

---



## IndexFreeze


Freezes the given index, disallowing creation of new index entries.
Should not be called concurrently with IndexGet.



### Interface


---------- | ----------
*Inputs* | 
`handle` | Pointer to an Index instance.
*Outputs* | 
`handle` | The input handle.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## IndexGet


Given an index handle and a tensor of keys, return an Int tensor of same shape containing the indices for each of the keys. If the index is frozen, unknown entries are given index 0. Otherwise, new entries are added into the index.
If an insert is necessary but max_elements has been reached, fail.



### Interface


---------- | ----------
*Inputs* | 
`handle` | Pointer to an Index instance.
`keys` | Tensor of keys to be looked up.
*Outputs* | 
`indices` | Indices for each of the keys.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## IndexHash


This operator translates a list of indices into a list of hashed indices.
A seed can be fed as an argument to change the behavior of the hash function.
If a modulo is specified, all the hashed indices will be modulo the specified number. All input and output indices are enforced to be positive.



### Interface


---------- | ----------
*Arguments* | 
`seed` | seed for the hash function
`modulo` | must be > 0, hashed ids will be modulo this number
*Inputs* | 
`Indices` | Input feature indices.
*Outputs* | 
`HashedIndices` | Hashed feature indices.


### Code


[caffe2/operators/index_hash_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_hash_ops.cc)

---



## IndexLoad


Loads the index from the given 1-D tensor. Elements in the tensor will be given consecutive indexes starting at 1. Fails if tensor contains repeated elements.



### Interface


---------- | ----------
*Arguments* | 
`skip_first_entry` | If set, skips the first entry of the tensor. This allows to load tensors that are aligned with an embedding, where the first entry corresponds to the default 0 index entry.
*Inputs* | 
`handle` | Pointer to an Index instance.
`items` | 1-D tensor with elements starting with index 1.
*Outputs* | 
`handle` | The input handle.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## IndexSize


Returns the number of entries currently present in the index.



### Interface


---------- | ----------
*Inputs* | 
`handle` | Pointer to an Index instance.
*Outputs* | 
`items` | Scalar int64 tensor with number of entries.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## IndexStore


Stores the keys of this index in a 1-D tensor. Since element 0 is reserved for unknowns, the first element of the output tensor will be element of index 1.



### Interface


---------- | ----------
*Inputs* | 
`handle` | Pointer to an Index instance.
*Outputs* | 
`items` | 1-D tensor with elements starting with index 1.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## InstanceNorm


Carries out instance normalization as described in the paper  [https://arxiv.org/abs/1607.08022.](https://arxiv.org/abs/1607.08022.)  Depending on the mode it is being run, there are multiple cases for the number of outputs, which we list below:   

```
  * Output case #1: output
  * Output case #2: output, saved_mean
    - don't use, doesn't make sense but won't crash
  * Output case #3: output, saved_mean, saved_inv_stdev
    - Makes sense for training only

```

 For training mode, type 3 is faster in the sense that for the backward pass, it is able to reuse the saved mean and inv_stdev in the gradient computation.



### Interface


---------- | ----------
*Arguments* | 
`epsilon` | The epsilon value to use to avoid division by zero.
`order` | A StorageOrder string.
*Inputs* | 
`input` | The input 4-dimensional tensor of shape NCHW or NHWC depending on the order parameter.
`scale` | The input 1-dimensional scale tensor of size C.
`bias` | The input 1-dimensional bias tensor of size C.
*Outputs* | 
`output` | The output 4-dimensional tensor of the same shape as input.
`saved_mean` | Optional saved mean used during training to speed up gradient computation. Should not be used for testing.
`saved_inv_stdev` | Optional saved inverse stdev used during training to speed up gradient computation. Should not be used for testing.


### Code


[caffe2/operators/instance_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc)

---



## InstanceNormGradient

No documentation yet.


### Code


[caffe2/operators/instance_norm_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_gradient_op.cc)

---



## IntIndexCreate


Creates a dictionary that maps int32 keys to consecutive integers from 1 to max_elements. Zero is reserved for unknown keys.



### Interface


---------- | ----------
*Arguments* | 
`max_elements` | Max number of elements, including the zero entry.
*Outputs* | 
`handler` | Pointer to an Index instance.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## IsEmpty

Returns true iff the input tensor has size == 0


### Interface


---------- | ----------
*Inputs* | 
`tensor` | Tensor of any type.
*Outputs* | 
`is_empty` | Scalar bool tensor. True if input is empty.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## IsMemberOf


IsMemberOf takes input data (Tensor<T>) and a list of values as argument, and produces one output data (Tensor<bool>) where the function  `f(x) = x in values` , is applied to the data tensor elementwise.



### Interface


---------- | ----------
*Arguments* | 
`value` | Declare one value for the membership test.
`dtype` | The data type for the elements of the output tensor.Strictly must be one of the types from DataType enum in TensorProto.
*Inputs* | 
`X` | Input tensor of any shape
*Outputs* | 
`Y` | Output tensor (same size as X containing booleans)


### Code


[caffe2/operators/elementwise_logical_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.cc)

---



## Iter


Stores a singe integer, that gets incremented on each call to Run().
Useful for tracking the iteration count during SGD, for example.



### Code


[caffe2/sgd/iter_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/iter_op.cc)

---



## KeySplit

No documentation yet.


### Code


[caffe2/operators/key_split_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/key_split_ops.cc)

---



## KeyValueToMap

Convert key and value blob pairs into a map blob


### Interface


---------- | ----------
*Inputs* | 
`key blob` | Blob reference to the key
`value blob` | Blob reference to the value
*Outputs* | 
`map blob` | Blob reference to the map


### Code


[caffe2/operators/map_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/map_ops.cc)

---



## L1Distance


Given two input float tensors X, Y, and produces one output float tensor of the L1 difference between X and Y, computed as L1(x,y) = sum over |x-y| 


### Interface


---------- | ----------
*Inputs* | 
`X` | 1D or 2D input tensor
`Y` | 1D or 2D input tensor (must have the same shape as X)
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## L1DistanceGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## LC


The locally connected operator consumes an input vector, a filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC1D


The locally connected operator consumes an input vector, a 1D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC1DGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC2D


The locally connected operator consumes an input vector, a 2D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC2DGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC3D


The locally connected operator consumes an input vector, a 3D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC3DGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LCGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LE


Performs element-wise less or equal than comparison  `<=`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## LRN

No documentation yet.


### Code


[caffe2/operators/local_response_normalization_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/local_response_normalization_op.cc)

---



## LRNGradient

No documentation yet.


### Code


[caffe2/operators/local_response_normalization_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/local_response_normalization_op.cc)

---



## LSTMUnit


LSTMUnit computes the activations of a standard LSTM (without peephole connections), in a sequence-length aware fashion.
 Concretely, given the (fused) inputs X (TxNxD), the previous cell state (NxD), and the sequence lengths (N), computes the LSTM activations, avoiding computation if the input is invalid (as in, the value at X{t][n] >= seqLengths[n].
 


### Interface


---------- | ----------
*Arguments* | 
`forget_bias` | Bias term to add in while calculating forget gate
`sequence_lengths` | When false, the sequence lengths input is left out, and all following inputs are shifted left by one.


### Code


[caffe2/operators/lstm_unit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lstm_unit_op.cc)

---



## LSTMUnitGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`sequence_lengths` | When false, the sequence lengths input is left out, and all following inputs are shifted left by one.


### Code


[caffe2/operators/lstm_unit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lstm_unit_op.cc)

---



## LT


Performs element-wise less than comparison  `<`  (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## LabelCrossEntropy


Operator computes the cross entropy between the input and the label set. In  practice, it is most commonly used at the end of models, after the SoftMax  operator and before the AveragedLoss operator. Note that LabelCrossEntropy  assumes that the label provided is either a 1D array of size N (batch size), or  a 2D array of size N x 1 (batch size). Each entry in the label vector indicates  which is the correct class; as such, each entry must be between 0 and D - 1,  inclusive, where D is the total number of classes. The formula used is:   

```
                            Y[i] = -log(X[i][j])

```

  where (i, j) is the classifier's prediction of the jth class (the correct one),  and i is the batch size. Each log has a lower limit for numerical stability.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input blob from the previous layer, which is almost always the result of a softmax operation; X is a 2D array of size N x D, where N is the batch size and D is the number of classes
`label` | Blob containing the labels used to compare the input
*Outputs* | 
`Y` | Output blob after the cross entropy computation


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## LabelCrossEntropyGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## LambdaRankNdcg


It implements the LambdaRank as appeared in Wu, Qiang, et al. "Adapting boosting for information retrieval measures." Information Retrieval 13.3 (2010): 254-270.
 This method heuristically optimizes the NDCG.



### Code


[caffe2/operators/listwise_l2r_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/listwise_l2r_op.cc)

---



## LambdaRankNdcgGradient

No documentation yet.


### Code


[caffe2/operators/listwise_l2r_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/listwise_l2r_op.cc)

---



## Lars


Implement Layer-wise Adaptive Rate Scaling (LARS) as in  [https://arxiv.org/abs/1708.03888.](https://arxiv.org/abs/1708.03888.)  Without weight decay, given a global learning rate lr, parameter tensor X and its gradient dX, the local learning rate for X will be   

```
    local_lr = lr * norm(X) / ( norm(dX) + offset * norm(X) )

```

   

```
            = lr  / ( norm(dX) / norm(X) + offset ),

```

 where offset is a preset hyper-parameter to avoid numerical issue.
In this implementation, we uses l2 norm and output the rescaling factor   

```
    1 / ( norm(dX) / norm(X) + offset ).

```




### Interface


---------- | ----------
*Arguments* | 
`offset` | rescaling offset parameter
*Inputs* | 
`X` | Parameter tensor
`dX` | Gradient tensor
*Outputs* | 
`lr_rescale` | Local learning rate rescaling factor


### Code


[caffe2/sgd/lars_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/lars_op.cc)

---



## LastNWindowCollector


Collect the last N rows from input data. The purpose is to keep track of data accross batches, so for example suppose the LastNWindowCollector is called successively with the following input data   

```
  [1, 2, 3, 4]
  [5, 6, 7]
  [8, 9, 10, 11]

```

 And the number of items is set to 6, then the output after the 3rd call will contain the following elements:   

```
  [6, 7, 8, 9, 10, 11]

```

 No guarantee is made on the ordering of elements in input. So a valid value for output could have been   

```
  [11, 10, 9, 8, 7, 6]

```

 Also, this method works for any order tensor, treating the first dimension as input rows and keeping the last N rows seen as input. So for instance:   

```
  [[1, 2], [2, 3], [3, 4], [4, 5]]
  [[5, 6], [6, 7], [7, 8]]
  [[8, 9], [9, 10], [10, 11], [11, 12]]

```

 A possible output would be   

```
  [[6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12]]

```

 This is not thread safe unless a mutex is given.



### Interface


---------- | ----------
*Arguments* | 
`num_to_collect` | The number of random samples to append for each positive samples
*Inputs* | 
`last-N buffer` | The buffer for last-N record. Should be initialized to empty tensor
`next cursor` | The cursor pointing to the next position that should be replaced. Should be initialized to 0.
`DATA` | tensor to collect from
`MUTEX` | (optional) mutex to use to make this thread-safe
`NUM_VISITED` | 
*Outputs* | 
`last-N buffer` | Data stored in sessions
`next cursor` | Updated input cursor
`NUM_VISITED` | number of records seen so far


### Code


[caffe2/operators/last_n_window_collector.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/last_n_window_collector.cc)

---



## LayerNorm


Computes layer normalization as described in  [https://arxiv.org/pdf/1607.06450.pdf.](https://arxiv.org/pdf/1607.06450.pdf.) 
Given an input vector x \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}], this op treats dimensions a_k through a_{n-1} as feature vectors. For each feature vector, the op contains the mean and standard deviation. Then, it returns the normalized values (with respect to the feature vector).
 Note that this op does not contain the scale an bias terms described in the paper. Simply follow this op with an FC op to add those. Concretely, this op implements:  h = \frac{1}{\sigma}(a - \mu) where \mu = \frac{1}{H}\sum_{i=1}^{H} a_i and \sigma = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_i - \mu)^2} where H is the number of hidden units (i.e. product of dimensions from 'axis' to the end.) 


### Interface


---------- | ----------
*Arguments* | 
`axis` | (int) default to 1; Describes axis of the inputs. Defaults to one because the 0th axis most likely describes the batch size
`epsilon` | (float) default to 0.001. Small value to be added to the stdev when dividing out by that value. This prevents division by zero.
*Inputs* | 
`input` | Input tensor which layer normalization will be applied to
*Outputs* | 
`output` | Normalized values
`mean` | Mean values for each feature vector
`stddev` | Standard deviations for each feature vector


### Code


[caffe2/operators/layer_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/layer_norm_op.cc)

---



## LayerNormGradient

No documentation yet.


### Code


[caffe2/operators/layer_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/layer_norm_op.cc)

---



## LeakyRelu


LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one output data (Tensor<T>) where the function  `f(x) = alpha * x for x < 0` ,  `f(x) = x for x >= 0` , is applied to the data tensor elementwise.



### Interface


---------- | ----------
*Arguments* | 
`alpha` | Coefficient of leakage, default value is 0.01
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/leaky_relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/leaky_relu_op.cc)

---



## LeakyReluGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`alpha` | Coefficient of leakage


### Code


[caffe2/operators/leaky_relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/leaky_relu_op.cc)

---



## LearningRate


Learning rate is a decreasing function of time. With low learning rates the improvements will be linear. With high learning rates they will start to look more exponential. Learning rate is controlled by the following arguments:   Required:  

```
  `iterations`
  `base_lr`: base learning rate
  `policy`: this controls how the learning rate is applied, options are:
    `fixed`
    `step`: uses `stepsize`, `gamma`
    `exp`: uses `gamma`
    `inv`: uses `gamma`, `power`
    `linearWarmup`: uses `start_multiplier`, `num_iter`
    `constantWarmup`: uses `multiplier`, `num_iter`
    `alter`: uses  `active_first`, `active_period`, `inactive_period`
    `hill`: uses those in both `linearWarmup` and `inv`, plus `end_multiplier`


```

 Optional:  

```
  `stepsize`: defaults to 0
  `gamma`: defaults to 0
  `power`: defaults to 0
  `num_iter`: defaults to 0
  `start_multiplier`: defaults to 0
  `multiplier`: defaults to 0.5


```

 Usage:  

```
  train_net.LearningRate(*iterations*, "*label*", base_lr=*float*,
```

   

```
                        policy="policy_name", stepsize=*int*, gamma=*float*)


```

 Example usage:  

```
  train_net.LearningRate(200, "LR", base_lr=-0.1,
```

   

```
                        policy="step", stepsize=20, gamma=0.9)
```




### Interface


---------- | ----------
*Arguments* | 
`base_lr` | (float, required) base learning rate
`policy` | (float, default 1.0) strategy for gamma enforcement
`power` | (float, default 1.0) used only for inv policy type
`gamma` | (float, default 1.0) momentum of change
`stepsize` | (float, default 1.0) sampling rate on iterations
`active_first` | (boolean, default True) in alter policy
`active_period` | (int64_t, required) in alter policy
`inactive_period` | (int64_t, required) in alter policy
`max_iter` | (int, default -1) maximum iterations in this training run
`num_iter` | (int, default 0) number of iterations over which to warmup lr
`start_multiplier` | (float, default 0) starting multiplier for learning rate
`end_multiplier` | (float, default 0) end multiplier for learning rate
`multiplier` | (float, default 0.5) constant multiplier for learning rate
*Inputs* | 
`input` | description needed
*Outputs* | 
`output` | description needed


### Code


[caffe2/sgd/learning_rate_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/learning_rate_op.cc)

---



## LengthsGather


Gather items from sparse tensor. Sparse tensor is described by items and lengths. This operator gathers items corresponding to lengths at the given indices. This deliberately doesn't return lengths of OUTPUTS so that both lists and maps can be supported without special cases. If you need lengths tensor for  OUTPUT, use  `Gather` .
 Example:  

```
  ITEMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  LENGTHS = [0, 2, 3, 1, 4]
  INDICES = [0, 2, 4]

  OUTPUT = [2, 3, 4, 6, 7, 8, 9]
```




### Interface


---------- | ----------
*Inputs* | 
`ITEMS` | items tensor
`LENGTHS` | lengths tensor
`INDICES` | indices into LENGTHS where items should be gathered
*Outputs* | 
`OUTPUT` | 1-D tensor containing gathered items


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsIndicesInGradientSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMax


Applies 'Max' to each segment of the input tensor. Segments are defined by their LENGTHS.
 LENGTHS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 For example LENGTHS = [2, 1] stands for segments DATA[0..1] and DATA[2]  The first dimension of the output is equal to the number of input segments, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Max computes the element-wise max of the input slices. Operation doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of len(LENGTHS) 


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMaxWithMainInputAndForwardOutputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMean


Applies 'Mean' to each segment of the input tensor. Segments are defined by their LENGTHS.
 LENGTHS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 For example LENGTHS = [2, 1] stands for segments DATA[0..1] and DATA[2]  The first dimension of the output is equal to the number of input segments, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Mean computes the element-wise mean of the input slices. Operation doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of len(LENGTHS) 


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsPartition


LengthsPartition splits the input int tensor into multiple ones according to the second tensor. The first dimension is expected to be the tensor that describes lengths of the elements.
 Takes the second input and partitions it to shards according to the remainder of values modulo the number of partitions. It requires the second tensor to be a 1D-tensor of the integral type. The first tensor should be 1D-tensor of int32 that would represent the lengths of the elements in the input. The number of partitions is derived as (num_output / num_input).
 If additional inputs are present they must have the same shape as the first input, optionally with extra trailing dimensions. They will be partitioned accordingly to the first input.
 Optional arg 'pack_first_input' transforms the first tensor values as X_ij / num_partitions.
 Outputs are ordered as X_0_part_0, X_1_part_0, ..., X_N-1_part_0, X_0_part_1, ..., X_N-1_part_K-1 


### Interface


---------- | ----------
*Arguments* | 
`pack_first_input` | (int, default 0) If set, the operator transforms the first tensor values as floor(X_ij / num_partitions)
*Inputs* | 
`input` | Input tensor containing data to be partitioned. The number of input tensors might be greater than 1 but must have the same shape as the previous tensors.
*Outputs* | 
`partitions` | Output Partitions. The number of output tensors has to be a multiple of the number of input tensors.


### Code


[caffe2/operators/partition_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/partition_ops.cc)

---



## LengthsRangeFill


Convert a length vector to a range sequence. For example, input=[4,3,1], the output would be [0,1,2,3,0,1,2,0].



### Interface


---------- | ----------
*Inputs* | 
`lengths` | 1D tensor of int32 or int64 segment lengths.
*Outputs* | 
`range_sequence` | 1D tensor whose size is the sum of `lengths`


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## LengthsSum


Applies 'Sum' to each segment of the input tensor. Segments are defined by their LENGTHS.
 LENGTHS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 For example LENGTHS = [2, 1] stands for segments DATA[0..1] and DATA[2]  The first dimension of the output is equal to the number of input segments, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of len(LENGTHS) 


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsTile


Given DATA tensor of rank r >= 1, and LENGTHS tensor of rank 1, duplicate each entry of the outer-most dimension of DATA according to LENGTHS, and concatenate them in an output tensor of rank r.
 Example:  

```
  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
      [6.8, 7.9],
  ]
  LENGTHS = [0, 1, 3, 2]
  OUTPUT = [
      [2.3, 3.4],
      [4.5, 5.7],
      [4.5, 5.7],
      [4.5, 5.7],
      [6.8, 7.9],
      [6.8, 7.9],
  ]
```




### Interface


---------- | ----------
*Inputs* | 
`DATA` | Tensor of rank r >= 1. First dimension must be equal to the size of lengths
`LENGTHS` | Tensor of int32 lengths of rank 1
*Outputs* | 
`OUTPUT` | Tensor of rank r


### Code


[caffe2/operators/lengths_tile_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_tile_op.cc)

---



## LengthsToRanges


Given a vector of segment lengths, calculates offsets of each segment and packs them next to the lengths. For the input vector of length N the output is a Nx2 matrix with (offset, lengths) packaged for each segment.
 For example,  `[1, 3, 0, 2]`  transforms into  `[[0, 1], [1, 3], [4, 0], [4, 2]]` .



### Interface


---------- | ----------
*Inputs* | 
`lengths` | 1D tensor of int32 segment lengths.
*Outputs* | 
`ranges` | 2D tensor of shape len(lengths) X 2 and the same type as `lengths`


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsToSegmentIds


Given a vector of segment lengths, returns a zero-based, consecutive vector of segment_ids. For example, [1, 3, 0, 2] will produce [0, 1, 1, 1, 3, 3].
In general, the inverse operation is SegmentIdsToLengths. Notice though that trailing empty sequence lengths can't be properly recovered from segment ids.



### Interface


---------- | ----------
*Inputs* | 
`lengths` | 1D tensor of int32 or int64 segment lengths.
*Outputs* | 
`segment_ids` | 1D tensor of length `sum(lengths)`


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsToShape

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsToWeights


Similar as LengthsToSegmentIds but output vector of segment weights derived by lengths. i.e 1/pow(length, power) 


### Interface


---------- | ----------
*Arguments* | 
`power` | n of 1/pow(length,n) for normalization
*Inputs* | 
`lengths` | 1-D int32_t or int64_t tensor of lengths
*Outputs* | 
`a vector of weights` | 1-D float tensor of weights by length


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsTopK


Apply TopK to each segment of the input tensor, where segments are defined by their LENGTHS, and concatenate them in an output tensor of shape=(SIZE(LENGTHs), k). In case there's less than k values in a segment, the output value will be padded by 0, and the corresponding output indices will be padded by -1.



### Interface


---------- | ----------
*Arguments* | 
`k` | the number of top values to return for each segment, if the number of values is smaller than k, the values would be padded with 0 and indices would be padded with -1.
*Inputs* | 
`DATA` | Tensor of rank 1. First dimension must be equal to the sum of lengths
`LENGTHS` | Tensor of int32 lengths of rank 1
*Outputs* | 
`TopKValue` | Output top k elements for each segment, withshape=(SIZE(lengths), k)
`TopKIndices` | Output indices in DATA corresponding to value in TopKValue


### Code


[caffe2/operators/lengths_top_k_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_top_k_op.cc)

---



## LengthsTopKGradient

No documentation yet.


### Code


[caffe2/operators/lengths_top_k_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_top_k_op.cc)

---



## LengthsWeightedSum


Applies 'WeightedSum' to each segment of the input tensor. Segments are defined by their LENGTHS.
 LENGTHS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 For example LENGTHS = [2, 1] stands for segments DATA[0..1] and DATA[2]  The first dimension of the output is equal to the number of input segments, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`grad_on_weights` | Produce also gradient for `weights`. For now it's only supported in `Lengths`-based operators
*Inputs* | 
`DATA` | Input tensor for the summation
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the number of slices
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of len(LENGTHS) 


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsWeightedSumWithMainInputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## Load


The Load operator loads a set of serialized blobs from a db or multiple dbs. It takes [0, infinity) number of inputs and [0, infinity) number of outputs, using the db keys to match the db entries with the outputs.
 If at least one input is passed, then it is assumed that that input blobs are a set of DBReaders to load from. Otherwise the db or dbs argument is used to load blobs from one single db or multiple dbs respectively. db_type argument is used to specify the type of the input db/dbs.



### Interface


---------- | ----------
*Arguments* | 
`absolute_path` | (int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace.
`add_prefix` | (string, default="") blobs will be prefixed with this when loading.Useful for avoiding collisions with blobs existing in the workspace.The output blob names specified to this op should include this prefix.
`strip_prefix` | (string, default="") characters in the provided blob  names that match strip_prefix will be removed prior to loading. Also, characters that precede strip_prefix will be removed. Useful  for removing device scope from blob names.
`db` | (string) the path to the db to load.
`dbs` | (list of strings) the paths to the dbs to load. This is used for loading blobs from multiple databases. If it is set, argument in "db" will be ignored.
`db_type` | (string) the type of the db.
`keep_device` | (int, default 0) if nonzero, the blobs are loaded into the device that is specified in the serialized BlobProto. Otherwise, the device will be set as the one that the Load operator is being run under.
`load_all` | (int, default 0) if nonzero, will load all blobs pointed to by the db to the workspace overwriting/creating blobs as needed.
`allow_incomplete` | (bool, default false) if true, will allow not loading all the output blobs specified in the outputs
`source_blob_names` | (list of strings) if set, used instead of output blob names, to specify which blobs in the db shall be loaded. Must be the same length as number of output blobs.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)

---



## Log


Calculates the natural log of the given input tensor, element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The natural log of the input tensor computed element-wise


### Code


[caffe2/operators/log_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/log_op.cc)

---



## Logit


Elementwise logit transform: logit(x) = log(x / (1 - x)), where x is the input data clampped in (eps, 1-eps).



### Interface


---------- | ----------
*Arguments* | 
`eps (optional)` | small positive epsilon value, the default is 1e-6.
*Inputs* | 
`X` | input float tensor
*Outputs* | 
`Y` | output float tensor


### Code


[caffe2/operators/logit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/logit_op.cc)

---



## LogitGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`eps` | small positive epsilon value, the default is 1e-6.
*Inputs* | 
`X` | input float tensor
`dY` | input float tensor
*Outputs* | 
`dX` | output float tensor


### Code


[caffe2/operators/logit_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/logit_op.cc)

---



## LongIndexCreate


Creates a dictionary that maps int64 keys to consecutive integers from 1 to max_elements. Zero is reserved for unknown keys.



### Interface


---------- | ----------
*Arguments* | 
`max_elements` | Max number of elements, including the zero entry.
*Outputs* | 
`handler` | Pointer to an Index instance.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## LpNorm


Given one input float tensor X, and produces one output float tensor of the Lp norm of tensor X, computed as Lp(x) = sum over |x^p|, in which p is either 1 or 2(currently only supports l1 and l2 norm), determined by the argument p.



### Interface


---------- | ----------
*Arguments* | 
`p` | Order of the norm in p-norm
`average` | whehther we calculate norm or averaged_norm.The Lp_averaged_norm(x) is defined asLp_averaged_norm(x) = LpNorm(x) / size(x)
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/lpnorm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lpnorm_op.cc)

---



## LpNormGradient


Given one input float tensor X, derivative dout, and produces one output float tensor dX. dX is the derivative of the Lp norm of tensor X, computed as dx = d(sum over |x^p|)/dx, in which p is either 1 or 2(currently only supports l1 and l2 norm) determined by the argument p.



### Interface


---------- | ----------
*Arguments* | 
`p` | Order of the norm in p-norm
`average` | whehther we calculate norm or averaged_norm.The Lp_averaged_norm(x) is defined asLp_averaged_normgradient(x) = LpNormGradient(x) / size(x)
*Inputs* | 
`X` | 1D input tensor
`dout` | 1D input tensor
*Outputs* | 
`dx` | 1D output tensor


### Code


[caffe2/operators/lpnorm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lpnorm_op.cc)

---



## LpPool


LpPool consumes an input blob X and applies L-p pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. L-p pooling consisting of taking the L-p norm of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. 
*Outputs* | 
`Y` | Output data tensor from L-p pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/lp_pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lp_pool_op.cc)

---



## LpPoolGradient

No documentation yet.


### Code


[caffe2/operators/lp_pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lp_pool_op.cc)

---



## MSRAFill

No documentation yet.


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## MakeTwoClass


Given a vector of probabilities, this operator transforms this into a 2-column  matrix with complimentary probabilities for binary classification. In explicit  terms, given the vector X, the output Y is vstack(1 - X, X).
  


### Interface


---------- | ----------
*Inputs* | 
`X` | Input vector of probabilities
*Outputs* | 
`Y` | 2-column matrix with complimentary probabilities of X for binary classification


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## MakeTwoClassGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## MapToKeyValue

Convert a map blob into key and value blob pairs


### Interface


---------- | ----------
*Inputs* | 
`map blob` | Blob reference to the map
*Outputs* | 
`key blob` | Blob reference to the key
`value blob` | Blob reference to the value


### Code


[caffe2/operators/map_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/map_ops.cc)

---



## MarginRankingCriterion


MarginRankingCriterion takes two input data X1 (Tensor<float>), X2 (Tensor<float>), and label Y (Tensor<int>) to produce the loss (Tensor<float>) where the loss function, loss(X1, X2, Y) = max(0, -Y * (X1 - X2) + margin), is applied to the tensor elementwise.
 If y == 1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for y == -1.



### Interface


---------- | ----------
*Inputs* | 
`X1` | The left input vector as a 1-dim TensorCPU.
`X2` | The right input vector as a 1-dim TensorCPU.
`Y` | The label as a 1-dim TensorCPU with int value of 1 or -1.
*Outputs* | 
`loss` | The output loss with the same dimensionality as X1.


### Code


[caffe2/operators/margin_ranking_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/margin_ranking_criterion_op.cc)

---



## MarginRankingCriterionGradient


MarginRankingCriterionGradient takes both X1, X2, Y and dY and uses them to update dX1, and dX2 according to the chain rule and derivatives of the loss function.



### Code


[caffe2/operators/margin_ranking_criterion_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/margin_ranking_criterion_op.cc)

---



## MatMul


Matrix multiplication Y = A * B, where A has size (M x K), B has size (K x N), and Y will have a size (M x N).



### Interface


---------- | ----------
*Arguments* | 
`axis_a` | Exclusive axis that divides the first and second dimension of matrix A, default to 1
`axis_b` | Exclusive axis that divides the first and second dimension of matrix B, default to 1
`trans_a` | Pass 1 to transpose A before multiplication and after the dimension adjustment using axis_a
`trans_b` | Pass 1 to transpose B before multiplication and after the dimension adjustment using axis_b
*Inputs* | 
`A` | 2D matrix of size (M x K)
`B` | 2D matrix of size (K x N)
*Outputs* | 
`Y` | 2D matrix of size (M x N)


### Code


[caffe2/operators/matmul_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/matmul_op.cc)

---



## Max


Element-wise max of each of the input tensors. The first input tensor can be used in-place as the output tensor, in which case the max will be done in place and results will be accumulated in input0. All inputs and outputs must have the same shape and data type.



### Interface


---------- | ----------
*Inputs* | 
`data_0` | First of the input tensors. Can be inplace.
*Outputs* | 
`max` | Output tensor. Same dimension as inputs.


### Code


[caffe2/operators/minmax_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/minmax_ops.cc)

---



## MaxGradient

No documentation yet.


### Code


[caffe2/operators/minmax_gradient_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/minmax_gradient_ops.cc)

---



## MaxPool

MaxPool  consumes an input blob X and applies max pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Max pooling consisting of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool1D

MaxPool1D  consumes an input blob X and applies max pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Max pooling consisting of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool1DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## MaxPool2D

MaxPool2D  consumes an input blob X and applies max pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Max pooling consisting of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool2DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## MaxPool3D

MaxPool3D  consumes an input blob X and applies max pooling across the the blob according to kernel sizes, stride sizes, and pad lengths defined by the ConvPoolOpBase operator. Max pooling consisting of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob Y for further processing.



### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case.
*Outputs* | 
`Y` | Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.


### Code


[caffe2/operators/pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool3DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## MaxPoolGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## Mean


Element-wise mean of each of the input tensors. The first input tensor can be used in-place as the output tensor, in which case the mean will be done in place and results will be accumulated in input0. All inputs and outputs must have the same shape and data type.



### Interface


---------- | ----------
*Inputs* | 
`data_0` | First of the input tensors. Can be inplace.
*Outputs* | 
`mean` | Output tensor. Same dimension as inputs.


### Code


[caffe2/operators/mean_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/mean_op.cc)

---



## MeanGradient

No documentation yet.


### Code


[caffe2/operators/mean_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/mean_op.cc)

---



## MergeDim


Merge first two dimensions in a single dimension with size dim(0) * dim(1).



### Interface


---------- | ----------
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`reshaped` | Reshaped tensor.


### Code


[caffe2/operators/prepend_dim_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/prepend_dim_op.cc)

---



## MergeIdLists


MergeIdLists: Merge multiple ID_LISTs into a single ID_LIST.
 An ID_LIST is a list of IDs (may be ints, often longs) that represents a single feature. As described in  [https://caffe2.ai/docs/sparse-operations.html,](https://caffe2.ai/docs/sparse-operations.html,)  a batch of ID_LIST examples is represented as a pair of lengths and values where the  `lengths`  (int32) segment the  `values`  or ids (int32/int64) into examples.
 Given multiple inputs of the form lengths_0, values_0, lengths_1, values_1, ...
which correspond to lengths and values of ID_LISTs of different features, this operator produces a merged ID_LIST that combines the ID_LIST features. The final merged output is described by a lengths and values vector.
 WARNING: The merge makes no guarantee about the relative order of ID_LISTs within a batch. This can be an issue if ID_LIST are order sensitive.



### Interface


---------- | ----------
*Inputs* | 
`lengths_0` | Lengths of the ID_LISTs batch for first feature
`values_0` | Values of the ID_LISTs batch for first feature
*Outputs* | 
`merged_lengths` | Lengths of the merged ID_LISTs batch
`merged_values` | Values of the merged ID_LISTs batch


### Code


[caffe2/operators/merge_id_lists_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/merge_id_lists_op.cc)

---



## Min


Element-wise min of each of the input tensors. The first input tensor can be used in-place as the output tensor, in which case the min will be done in place and results will be accumulated in input0. All inputs and outputs must have the same shape and data type.



### Interface


---------- | ----------
*Inputs* | 
`data_0` | First of the input tensors. Can be inplace.
*Outputs* | 
`min` | Output tensor. Same dimension as inputs.


### Code


[caffe2/operators/minmax_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/minmax_ops.cc)

---



## MinGradient

No documentation yet.


### Code


[caffe2/operators/minmax_gradient_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/minmax_gradient_ops.cc)

---



## Mod


Elementwise modulo operation. Each element in the output is the modulo result of the corresponding elment in the input data. The divisor of the modulo is provided by the operator argument  `divisor` .



### Interface


---------- | ----------
*Arguments* | 
`divisor` | The divisor of the modulo operation. Must >= 1
`sign_follow_divisor` | The sign of output follows Dividend if set to `false`.           Otherwise follows Divisor
*Inputs* | 
`data` | input int32 or int64 data
*Outputs* | 
`output` | output of data with modulo operation applied


### Code


[caffe2/operators/mod_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/mod_op.cc)

---



## MomentumSGD


 Computes a momentum SGD update for an input gradient and momentum parameters. Concretely, given inputs (grad, m, lr) and parameters (momentum, nesterov), computes:   

```
    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        return (adjusted_gradient, adjusted_gradient)
    else:
        m_new = momentum * m + lr * grad
        return ((1 + momentum) * m_new - momentum * m, m_new)

```

 Output is (grad, momentum)  Note the difference to MomemtumSGDUpdate, which actually performs the parameter update (and is thus faster).



### Code


[caffe2/sgd/momentum_sgd_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/momentum_sgd_op.cc)

---



## MomentumSGDUpdate


 Performs a momentum SGD update for an input gradient and momentum parameters. Concretely, given inputs (grad, m, lr, param) and arguments (momentum, nesterov), computes:   

```
    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        param = param - adjusted_gradient
        return (adjusted_gradient, adjusted_gradient, param)
    else:
        m_new = momentum * m + lr * grad
        param = param - ((1 + momentum) * m_new - momentum * m),
        return ((1 + momentum) * m_new - momentum * m, m_new, param)

```

 Output is (grad, momentum, parameter).
 Note the difference to MomentumSGD, which returns a new gradient but does not perform the parameter update.
 


### Code


[caffe2/sgd/momentum_sgd_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/momentum_sgd_op.cc)

---



## Mul


Performs element-wise binary multiplication (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and type as A


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## MultiClassAccuracy


Respectively compute accuracy score for each class given a number of instances and predicted scores of each class for each instance.



### Interface


---------- | ----------
*Inputs* | 
`prediction` | 2-D float tensor (N,D,) of predicted scores of each class for each data. N is the number of instances, i.e., batch size. D is number of possible classes/labels.
`labels` | 1-D int tensor (N,) of labels for each instance.
*Outputs* | 
`accuracies` | 1-D float tensor (D,) of accuracy for each class. If a class has no instance in the batch, its accuracy score is set to zero.
`amounts` | 1-D int tensor (D,) of number of instances for each class in the batch.


### Code


[caffe2/operators/multi_class_accuracy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/multi_class_accuracy_op.cc)

---



## NCHW2NHWC


The operator switches the order of data in a tensor from NCHW- sample index N, channels C, height H and width W, to the NHWC order.



### Interface


---------- | ----------
*Inputs* | 
`data` | The input data (Tensor<float>) in the NCHW order.
*Outputs* | 
`output` | The output tensor (Tensor<float>) in the NHWC order.


### Code


[caffe2/operators/order_switch_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/order_switch_ops.cc)

---



## NGramFromCategorical

No documentation yet.


### Code


[caffe2/operators/ngram_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/ngram_ops.cc)

---



## NHWC2NCHW


The operator switches the order of data in a tensor from NHWC- sample index N, height H, width H and channels C, to the NCHW order.



### Interface


---------- | ----------
*Inputs* | 
`data` | The input data (Tensor<float>) in the NHWC order.
*Outputs* | 
`output` | The output tensor (Tensor<float>) in the NCHW order.


### Code


[caffe2/operators/order_switch_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/order_switch_ops.cc)

---



## NanCheck

Identity operator, but checks all values for nan or inf


### Interface


---------- | ----------
*Inputs* | 
`tensor` | Tensor to check for nan/inf
*Outputs* | 
`output` | Tensor to copy input into if no NaNs or inf. Can be in-place


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## NegateGradient


NegagteGradient operator in forward pass simply copies input to the output, and in backward pass, flips the sign of the output gradient 


### Code


[caffe2/operators/negate_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/negate_gradient_op.cc)

---



## Negative


Computes the element-wise negative of the input.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/negative_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/negative_op.cc)

---



## Normalize


Given a matrix, apply L2-normalization along the specified dimension.



### Interface


---------- | ----------
*Arguments* | 
`axis` | axis to normalize


### Code


[caffe2/operators/normalize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/normalize_op.cc)

---



## NormalizeGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`axis` | axis to normalize


### Code


[caffe2/operators/normalize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/normalize_op.cc)

---



## NormalizeL1


Given a matrix, apply L1-normalization along the specified axis.



### Interface


---------- | ----------
*Arguments* | 
`axis` | axis to normalize


### Code


[caffe2/operators/normalize_l1_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/normalize_l1_op.cc)

---



## NormalizePlanarYUV

No documentation yet.


### Code


[caffe2/operators/norm_planar_yuv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/norm_planar_yuv_op.cc)

---



## Not

Performs element-wise negation.


### Interface


---------- | ----------
*Inputs* | 
`X` | Input tensor of type `bool`.
*Outputs* | 
`Y` | Output tensor of type `bool`.


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## ONNXWhile


 *** EXPERIMENTAL. This operator is a work-in-progress. No assumption should be made about the stability or correctness of this op. ** *  Generic Looping construct confirming to the ONNX Loop operator spec. This loop has multiple termination conditions:  1. Trip count. Iteration count specified at runtime. Set by specifying the  

```
    input M. Optional. Set to empty string to omit. Note that a static trip
    count (specified at graph construction time) can be specified by passing
    in a constant node for input M.
```

 2. Loop termination condition. This is an input to the op that determines  

```
    whether to run the first interation and also a loop-carried dependency for
    the body graph. The body graph must yield a value for the condition
    variable, whether this input is provided or not.

```

 This table summarizes the operating modes of this operator with equivalent C-style code:  Operator inputs defined as (max_trip_count, condition_var). Omitted optional inputs are represented as empty string. Concretely, in this caffe2 op an input is marked as omitted by setting its 'has_{name}' argument to False.
  

```
    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }
```

     


### Interface


---------- | ----------
*Arguments* | 
`loop_net` | Net executed on each iteration
*Inputs* | 
`condition` | Scalar boolean condition


### Code


[caffe2/operators/onnx_while_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/onnx_while_op.cc)

---



## OneHot


Given a sequence of indices, one for each example in a batch, returns a matrix where each inner dimension has the size of the index and has 1.0 in the index active in the given example, and 0.0 everywhere else.



### Interface


---------- | ----------
*Inputs* | 
`indices` | The active index for each example in the batch.
`index_size_tensor` | Scalar with the size of the index. Must be in CPU context
*Outputs* | 
`one_hots` | Matrix of size len(indices) x index_size


### Code


[caffe2/operators/one_hot_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc)

---



## Or


Performs element-wise logical operation  `or`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## PRelu


 PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one output data (Tensor<T>) where the function  `f(x) = slope * x for x < 0` ,  `f(x) = x for x >= 0` ., is applied to the data tensor elementwise.
 


### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
`Slope` | 1D slope tensor. If `Slope` is of size 1, the value is sharedacross different channels
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/prelu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/prelu_op.cc)

---



## PReluGradient


 PReluGradient takes both Y and dY and uses this to update dX and dW according to the chain rule and derivatives of the rectified linear function.
 


### Code


[caffe2/operators/prelu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/prelu_op.cc)

---



## PackRNNSequence


Pack values based on the length blob. Each number from length blob represents the corresponding values that need to be packed. The dimension for each pack is the same as the maximum number from the length blob (padding with zero is implemented for smaller length value). The overall output dimension is: T  * N *  D, where T is the max number of lengths, N is the size of lengths, and D is the dimension of each feature value. The following example shows the input and output of this operator:   Given:  

```
  values = [v1, v2, v3, v4, v5, v6, v7, v8]
  lengths = [2, 3, 1, 2];


```

 Output:  

```
  output = [
    [v1, v3, v6, v7],
    [v2, v4, 0,  v8],
    [0,  v5, 0,  0 ],
  ]


```

 One application for this operator is the transfer data into the format that is used for RNN models. Note that the gradient operator of PackRNNSequence is UnpackRNNSequence.



### Interface


---------- | ----------
*Inputs* | 
`values` | Data tensor, contains a sequence of features
`lengths` | lengths with each number representing the pack size.
*Outputs* | 
`output` | Output tensor after packing


### Code


[caffe2/operators/pack_rnn_sequence_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pack_rnn_sequence_op.cc)

---



## PackRecords


Given a dataset under a schema specified by the  `fields`  argument will pack all the input tensors into one, where each tensor element represents a row of data (batch of size 1). This format allows easier use with the rest of Caffe2 operators.



### Interface


---------- | ----------
*Arguments* | 
`fields` | List of strings representing the string names in the formatspecified in the doc for CreateTreeCursor.
*Outputs* | 
`tensor` | One dimensional tensor having a complex type of SharedTensorVectorPtr. In order to reverse it back to the original input it has to be inserted into UnPackRecordsOp.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## PackSegments

Map N dim tensor to N+1 dim based on length blob. Sequences that     are shorter than the longest sequence are padded with zeros.


### Interface


---------- | ----------
*Arguments* | 
`pad_minf` | Padding number in the packed segments. Use true to pad     -infinity, otherwise pad zeros
`return_presence_mask` | bool whether to return presence mask, false by default
*Inputs* | 
`lengths` | 1-d int/long tensor contains the length in each of the output.
`tensor` | N dim Tensor.
*Outputs* | 
`packed_tensor` | N + 1 dim Tensorwhere dim(1) is the max length, dim(0) is the batch size.
`presence_mask` | 2 dim boolean tensor, false where packed_tensor is padded, true otherwise.


### Code


[caffe2/operators/pack_segments.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pack_segments.cc)

---



## PackedInt8BGRANHWCToNCHWCStylizerPreprocess

No documentation yet.


### Code


[caffe2/operators/stylizer_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stylizer_ops.cc)

---



## PadEmptySamples


Pad empty field given lengths and index features,  Input(0) is a blob pointing to the lengths of samples in one batch, [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the features.
 PadEmptySamples is thread safe.



### Interface


---------- | ----------
*Inputs* | 
`lengths` | A blob containing a pointer to the lengths.
*Outputs* | 
`out_lengths` | Tensor containing lengths with empty sample padded.


### Code


[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)

---



## PadImage


PadImage pads values around the boundary of an image according to the pad values and stride sizes defined by the ConvPoolOpBase operator.
  


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. 
*Outputs* | 
`Y` | Output data tensor from padding the H and W dimensions on the tensor. Dimensions will vary based on various pad and stride sizes.


### Code


[caffe2/operators/pad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pad_op.cc)

---



## PadImageGradient

No documentation yet.


### Code


[caffe2/operators/pad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pad_op.cc)

---



## PairWiseLoss


Operator computes the pair wise loss between all pairs within a batch  using the logit loss function on the difference in scores between pairs 


### Interface


---------- | ----------
*Inputs* | 
`X` | Input blob from the previous layer, which is almost always the result of a softmax operation; X is a 2D array of size N x 1where N is the batch size. For more info: D. Sculley, Large Scale Learning to Rank. https://www.eecs.tufts.edu/~dsculley/papers/large-scale-rank.pdf
`label` | Blob containing the labels used to compare the input
`lengths` | Optional input blob that contains the lengthsof multiple sessions. The summation of this blob must be equalto the size of blob X. If lengths blob is provided, the outputblob has the same size as lengths blob, and the cross entropyis computed within each session.
*Outputs* | 
`Y` | Output blob after the cross entropy computation


### Code


[caffe2/operators/rank_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/rank_loss_op.cc)

---



## PairWiseLossGradient

No documentation yet.


### Code


[caffe2/operators/rank_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/rank_loss_op.cc)

---



## Partition


Splits the input int tensor into multiple ones according to the first tensor.
 Takes the first input and partitions it to shards according to the remainder of values modulo the number of partitions. It requires that the first tensor is of integral type. The number of partitions is derived as (num_output / num_input).
 If additional inputs are present they must have the same shape as the first input, optionally with extra trailing dimensions. They will be partitioned accordingly to the first input.
 Optional arg 'pack_first_input' transforms the first tensor values as X_ij / num_partitions.
 Outputs are ordered as X_0_part_0, X_1_part_0, ..., X_N-1_part_0, X_0_part_1, ..., X_N-1_part_K-1 


### Interface


---------- | ----------
*Arguments* | 
`pack_first_input` | (int, default 0) If set, the operator transforms the first tensor values as floor(X_ij / num_partitions)
*Inputs* | 
`input` | Input tensor containing data to be partitioned. The number of input tensors might be greater than 1 but must have the same shape as the previous tensors.
*Outputs* | 
`partitions` | Output Partitions. The number of output tensors has to be a multiple of the number of input tensors.


### Code


[caffe2/operators/partition_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/partition_ops.cc)

---



## Percentile


 

```
    This operator is used to find percentile representations for raw values, given a sample
    set of raw values, labeled with their corresponding percentiles from the same distribution.
    In particular, this operator takes as input a tensor of floats to find the percentile values
    for, a 2D tensor of floats, where the first column of the tensor represents sampled values,
    and the second column represents the percentile labels, and a tensor  of integers lengths.

    This lengths tensor is used because the operator works on multiple sets of raw values at the same time. For
    example, for an input:
    original_values=[[3, 5, 3],[5, 1, 6]], lengths = [2, 1, 1], value_to_pct = [[3, 0.2], [5, 0.5], [1, 0.3], [3. 0.6]]

    Our operator expects that each column i of the input tensor is sampled from distribution i. Lengths tells
    us that the first two elements in value_to_pct are sampled from distribution 1, the next is from distribution two,
    and the last is from distribution 3. We expect the output of our operator to give us [[0.2, 1.0, 0.6], [0.5, 0.3, 1.0]].

    To calculate the percentile of an element, we check to see if its value is already mapped to
    a percentile in value_to_pct. If so, we return that value. If not, we linearly interpolate between
    the two closest values in value_to_pct. If the value is larger than all values in value_to_pct, we
    return 1. If it's smaller than all the values, we return 0.

```




### Interface


---------- | ----------
*Inputs* | 
`original_values` | Input 2D tensor of floats, representing the original, raw data to calculate percentiles for.
`value_to_pct` | Sorted 2D tensor, with 2 columns. Each element in the first column is a float representing the raw value of a sample. Its corresponding element in the next column represents the percentile it maps to.
`lengths` | 1D tensor, representing the length of each distribution. We expect that the sum of elements of this tensor is equal to the total length of value_to_pct.
*Outputs* | 
`percentile_values` | 1D tensor of floats, with the same dimensions as the flattened input tensor. Each element of this tensor, percentile_values[i], corresponds to the percentile calculated for original_values[i].


### Code


[caffe2/operators/percentile_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/percentile_op.cc)

---



## Perplexity


Perplexity calculates how well a probability distribution predicts a sample.
Perplexity takes a 1-D tensor containing a batch of probabilities. Each value in the tensor belongs to a different sample and represents the probability of the model predicting the true label for that sample. The operator returns a single (float) perplexity value for the batch.



### Interface


---------- | ----------
*Inputs* | 
`probabilities` | The input data as Tensor. It contains a batch oftrue label or target probabilities
*Outputs* | 
`output` | The output- a single (float) perplexity value for the batch


### Code


[caffe2/operators/perplexity_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/perplexity_op.cc)

---



## PiecewiseLinearTransform


PiecewiseLinearTransform takes inputs -- predictions, a 2-D or 1-D tensor (Tensor<float>) of size (batch_size x prediction_dimensions). The piecewise linear functions are stored in bounds, slopes and intercepts. The output tensor has the same shape of input  `predictions`  and contains the predictions transformed by the piecewise linear functions. Each column of predictions has its own piecewise linear transformation functions. Therefore the size of piecewise function parameters are pieces x prediction_dimensions, except for binary predictions where only the positive prediction needs them. Note that in each piece, low bound is excluded while high bound is included. Also the piecewise linear function must be continuous.
 Notes - If the input is binary predictions (Nx2 or Nx1 tensor), set the binary arg to true so that one group of piecewise linear functions is needed (see details below).
- The transform parameters (bounds, slopes, intercepts) can be passed either through args or through input blobs.
- If we have multiple groups of piecewise linear functions, each group has the same number of pieces.
- If a prediction is out of the bounds, it is capped to the smallest or largest bound.



### Interface


---------- | ----------
*Arguments* | 
`bounds` | 1-D vector of size (prediction_dimensions x (pieces+1)) contain the upper bounds of each piece of linear function. One special case is the first bound is the lower bound of whole piecewise function and we treat it the same as the left most functions. (bounds, slopes, intercepts) can be passed through either arg or input blobs.
`slopes` | 1-D vector of size (prediction_dimensions x pieces) containing the slopes of linear function
`intercepts` | 1-D vector of size (prediction_dimensions x pieces) containing the intercepts of linear function
`binary` | If set true, we assume the input is a Nx1 or Nx2 tensor. If it is Nx1 tensor, it is positive predictions. If the input is Nx2 tensor, its first column is negative predictions and second column is positive and negative + positive = 1. We just need one group of piecewise linear functions for the positive predictions.
*Inputs* | 
`predictions` | 2-D tensor (Tensor<float>) of size (num_batches x num_classes) containing scores
`bounds (optional)` | See bounds in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.
`slopes (optional)` | See slopes in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.
`intercepts (optional)` | See intercepts in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.
*Outputs* | 
`transforms` | 2-D tensor (Tensor<float>) of size (num_batches x num_classes) containing transformed predictions


### Code


[caffe2/operators/piecewise_linear_transform_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/piecewise_linear_transform_op.cc)

---



## Pow


Pow takes input data (Tensor<T>) and an argument exponent, which can be a scalar or another tensor. It produces one output data (Tensor<T>), where the function  `f(x) = x^exponent`  is applied to the data tensor elementwise.



### Interface


---------- | ----------
*Arguments* | 
`exponent` | The exponent of the power function.
*Inputs* | 
`X` | Input tensor of any shape
`exponent` | The exponent of the power function.
*Outputs* | 
`Y` | Output tensor (same size as X)


### Code


[caffe2/operators/pow_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pow_op.cc)

---



## PrependDim


Reshape the tensor by prepending a dimension of fixed size and dividing the size of the next dimension by that amount.



### Interface


---------- | ----------
*Arguments* | 
`dim_size` | Size of the dimension to prepend.
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`reshaped` | Reshaped tensor.


### Code


[caffe2/operators/prepend_dim_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/prepend_dim_op.cc)

---



## Print

Logs shape and contents of input tensor to stderr or to a file.


### Interface


---------- | ----------
*Arguments* | 
`to_file` | (bool) if 1, saves contents to the root folder of the current workspace, appending the tensor contents to a file named after the blob name. Otherwise, logs to stderr.
*Inputs* | 
`tensor` | The tensor to print.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## Python

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc)

---



## PythonDLPack

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc)

---



## PythonDLPackGradient

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc)

---



## PythonGradient

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/python/pybind_state.cc)

---



## QuantDecode


Decode inputs using codebook. This is a general LUT operator that returns tensors with values from codebook (input 0) based on given indices in codes (input 1 ~ n).
  Example:   Input:  

```
  codebook = [1.5, 2.5, 3.5]
  codes_0 = [0, 1, 1, 2]
  codes_1 = [2, 0, 0]


```

 Output:  

```
  decoded_0 = [1.5, 2.5, 2.5, 3.5]
  decoded_1 = [3.5, 1.5, 1.5]
```




### Interface


---------- | ----------
*Inputs* | 
`codebook` | Codebook in 1d tensor (float)
`codes_0` | Encoded codes 0 (uint8/uint16/int32)
`codes_1` | Encoded codes 1 if existed (uint8/uint16/int32)
`codes_n` | Encoded codes n if existed (uint8/uint16/int32)
*Outputs* | 
`decoded_0` | Decoded tensor for codes_0 (float)
`decoded_1` | Decoded tensor for codes_1 (float)
`decoded_n` | Decoded tensor for codes_n (float)


### Code


[caffe2/operators/quant_decode_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/quant_decode_op.cc)

---



## QuantDecodeGradient

No documentation yet.


### Code


[caffe2/operators/quant_decode_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/quant_decode_op.cc)

---



## RMACRegions


Computes a fixed-grid of RMAC region coordinates at various levels as described in  [https://arxiv.org/abs/1511.05879.](https://arxiv.org/abs/1511.05879.) 



### Interface


---------- | ----------
*Arguments* | 
`scales` | Number of scales to sample regions at.
`overlap` | Overlap between consecutive regions.
*Inputs* | 
`X` | The input 4D tensor of shape NCHW.
*Outputs* | 
`RMAC_REGIONS` | The output RMAC regions for all items in the batch. Tensor of shape (N x 5) following the ROIPoolOp format - each row is of the format (batch_index x1 y1 x2 y2) where x1, y1, x2, y2 are the region co-ordinates. Each region is repeated N times corresponding to each item in the batch.


### Code


[caffe2/operators/rmac_regions_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/rmac_regions_op.cc)

---



## Range

Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop). When called with a single value, this will return  `[0, v]`  with the result type inferred from the input types.


### Interface


---------- | ----------
*Inputs* | 
`start` | Optional scalar Tensor with the start of the interval (inclusive).
`stop` | scalar Tensor with the end of the interval (exclusive)
`step` | Optional scalar Tensor with spacing between values.
*Outputs* | 
`output` | 1D tensor of same type as inputs that contains the sequence.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## RangeFill

No documentation yet.


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## ReadNextBatch


Read the next batch of examples out of the given cursor and data blobs.
 Input(0) is a blob pointing to a TreeCursor, and [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 ReadNextBatch is thread safe.



### Interface


---------- | ----------
*Arguments* | 
`batch_size` | Number of top-level entries to read.
*Inputs* | 
`cursor` | A blob containing a pointer to the cursor.
`dataset_field_0` | First dataset field
*Outputs* | 
`field_0` | Tensor containing the next batch for field 0.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## ReadRandomBatch


Read the next batch of examples out of the given cursor, idx blob, offset matrix and data blobs.
 Input(0) is a blob pointing to a TreeCursor, Input(1) is a blob pointing to the shuffled idx Input(2) is a blob pointing to the offset matrix and [Input(3),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 ReadRandomBatch is thread safe.



### Interface


---------- | ----------
*Arguments* | 
`batch_size` | Number of top-level entries to read.
`loop_over` | (bool) Repeat the dataset indefinitely
*Inputs* | 
`cursor` | A blob containing a pointer to the cursor.
`idx` | idx with a shuffled order.
`offsetsmat` | offset matrix containing length offset info.
`dataset_field_0` | First dataset field
*Outputs* | 
`field_0` | Tensor containing the next batch for field 0.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## ReceiveTensor


Receives the tensor from another node.



### Interface


---------- | ----------
*Arguments* | 
`src` | (int) he rank to receive the tensor from.
`tag` | (int) a tag to receive the tensor with.
`raw_buffer` | (bool) if set, only send the content and assume that the receiver has already known the tensor's shape and information.
*Inputs* | 
`comm_world` | The common world.
`Y` | In-place output. If raw_buffer is specified, Y should have pre-allocated data and type..
`src` | An int CPUtensor of size 1 specifying the rank. If given, this overrides the 'from' argument of the op.
`tag` | An int CPUtensor of size 1 specifying the tag to send the tensor with. This overrides the 'tag' argument of the op.
*Outputs* | 
`Y` | The received tensor.
`src` | The sender that sent the message as a CPUTensor of size 1 and of type int.
`tag` | The tag that the message is sent with as a CPUTensor of size 1 and of type int.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## RecurrentNetwork


Run the input network in a recurrent fashion. This can be used to implement fairly general recurrent neural networks (RNNs).
 The operator proceeds as follows.
 - First, initialized the states from the input recurrent states - For each timestep T, apply the links (that map offsets from input/output tensors into the inputs/outputs for the  `step`  network) - Finally, alias the recurrent states to the specified output blobs.
 This is a fairly special-case meta-operator, and so the implementation is somewhat complex. It trades of generality (and frankly usability) against performance and control (compared to e.g. TF dynamic_rnn, Theano scan, etc).
 See the usage examples for a flavor of how to use it.



### Code


[caffe2/operators/recurrent_network_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/recurrent_network_op.cc)

---



## RecurrentNetworkBlobFetcher


Retrieves blobs from scratch workspaces (which contain intermediate recurrent network computation for each timestep) and puts them in the global workspace under CPUContext.



### Interface


---------- | ----------
*Arguments* | 
`prefix` | Prefix string to prepend extracted blobs.
*Inputs* | 
`ScratchWorkspaceBlob` | Name of scratch workspace blob returned by recurrent network.
*Outputs* | 
`blob_names` | 1D tensor of strings containing extracted blob names.


### Code


[caffe2/operators/recurrent_network_blob_fetcher_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/recurrent_network_blob_fetcher_op.cc)

---



## RecurrentNetworkGradient

No documentation yet.


### Code


[caffe2/operators/recurrent_network_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/recurrent_network_op.cc)

---



## Reduce


Does a reduce operation from every node to the root node. Currently only Sum is supported.



### Interface


---------- | ----------
*Arguments* | 
`root` | (int, default 0) the root to run reduce into.
*Inputs* | 
`comm_world` | The common world.
`X` | A tensor to be reduced.
*Outputs* | 
`Y` | The reduced result on root, not set for other nodes.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## ReduceBackMax


Reduces the input tensor along the last dimension of the input tensor by applying 'Max'. When lengths is given, max is only computed with subsets of elements correspondingly.



### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | Number of dimensions to reduce
*Inputs* | 
`data_in` | (T<D1..., Dn>) Input data.
`lengths` | Num of elements in each sample, should have size D1 x D2 x ... x D(n-1).


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackMean


Reduces the input tensor along the last dimension of the input tensor by applying 'Mean'. When lengths is given, mean is only computed with subsets of elements correspondingly.



### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | Number of dimensions to reduce.
*Inputs* | 
`data_in` | (T<D1..., Dn>) Input data.
`lengths` | Num of elements in each sample, should have size D1 x D2 x ... x D(n-1).


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackMeanGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackSum


Reduces the input tensor along the last dimension of the input tensor by applying 'Sum'. 

```
  When lengths is given, sum is only computed
```

 with subsets of elements correspondingly.



### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | Number of dimensions to reduce.
*Inputs* | 
`data_in` | (T<D1..., Dn>) Input data.
`lengths` | Num of elements in each sample, should have size D1 x D2 x ... x D(n-1).


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackSumGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMax


Reduces the input tensor along the first dimension of the input tensor by applying 'Max'. When lengths is given, max is only computed with subsets of elements correspondingly.



### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | Number of dimensions to reduce
*Inputs* | 
`data_in` | (T<D1..., Dn>) Input data.
`lengths` | Num of elements in each sample, should have size D2 x D3 ... x Dn.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMean


Reduces the input tensor along the first dimension of the input tensor by applying 'Mean'. When lengths is given, mean is only computed with subsets of elements correspondingly.



### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | Number of dimensions to reduce.
*Inputs* | 
`data_in` | (T<D1..., Dn>) Input data.
`lengths` | Num of elements in each sample, should have size D2 x D3 x ... x Dn.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMeanGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontSum


Reduces the input tensor along the first dimension of the input tensor by applying 'Sum'. 

```
  When lengths is given, sum is only computed
```

 with subsets of elements correspondingly.



### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | Number of dimensions to reduce.
*Inputs* | 
`data_in` | (T<D1..., Dn>) Input data.
`lengths` | Num of elements in each sample, should have size D2 x D3 x ... x Dn.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontSumGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontWeightedSum


Reduces the input tensor along the first dimension of the input tensor by applying 'WeightedSum'. This op acts in a similar way to SortedSegmentWeightedSum and UnsortedSegmentWeightedSum but as if all input slices belong to a single segment.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`grad_on_weights` | Produce also gradient for `weights`. For now it's only supported in `Lengths`-based operators
*Inputs* | 
`DATA` | Input tensor for the summation
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the number of slices
*Outputs* | 
`OUTPUT` | Aggregated tensor


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## ReduceFrontWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## ReduceMean


 

```
      Computes the mean of the input tensor's element along the provided axes.
      The resulted tensor has the same rank as the input if keepdims equal 1.
      If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
```

     


### Interface


---------- | ----------
*Arguments* | 
`axes` | A list of integers, along which to reduce.
`keepdims` | Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`reduced` | Reduced output tensor.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceScatter


Does reduce-scatter operation among the nodes. Currently only Sum is supported.



### Interface


---------- | ----------
*Inputs* | 
`comm_world` | The common world.
`X` | A tensor to be reduce-scattered.
*Outputs* | 
`Y` | The reduced tensor, scattered on all nodes.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## ReduceSum


 

```
  Computes the sum of the input tensor's element along the provided axes.
  The resulted tensor has the same rank as the input if keepdims equal 1.
  If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
```




### Interface


---------- | ----------
*Arguments* | 
`axes` | A list of integers, along which to reduce.
`keepdims` | Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`reduced` | Reduced output tensor.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceTailSum


Reduce the tailing dimensions 


### Interface


---------- | ----------
*Inputs* | 
`mat` | The matrix
*Outputs* | 
`output` | Output


### Code


[caffe2/operators/rowmul_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/rowmul_op.cc)

---



## Relu


Relu takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to the tensor elementwise.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/relu_op.cc)

---



## ReluGradient


ReluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.



### Code


[caffe2/operators/relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/relu_op.cc)

---



## RemoveDataBlocks


Shrink the data tensor by removing data blocks with given zero-based indices in the outermost dimension of the tensor. Indices are not assumed in any order or unique but with the range [0, blocks_size). Indices could be empty.
  


### Interface


---------- | ----------
*Inputs* | 
`data` | a N-D data tensor, N >= 1
`indices` | zero-based indices of blocks to be removed
*Outputs* | 
`shrunk data` | data after removing data blocks indexed by 'indices'


### Code


[caffe2/operators/remove_data_blocks_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/remove_data_blocks_op.cc)

---



## RemovePadding


Remove padding around the edges of each segment of the input data. This is the reverse opration of AddPadding, and uses the same arguments and conventions for input and output data format.



### Interface


---------- | ----------
*Arguments* | 
`padding_width` | Outer-size of padding to remove around each range.
`end_padding_width` | (Optional) Specifies a different end-padding width.
*Inputs* | 
`data_in` | T<N, D1..., Dn> Input data
`lengths` | (i64) Num of elements in each range. sum(lengths) = N. If not provided, considers all data as a single segment.
*Outputs* | 
`data_out` | (T<N - 2*padding_width, D1..., Dn>) Unpadded data.
`lengths_out` | (i64, optional) Lengths for each unpadded range.


### Code


[caffe2/operators/sequence_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sequence_ops.cc)

---



## ReplaceNaN


Replace the NaN (not a number) element in the input tensor with argument  `value`  


### Interface


---------- | ----------
*Arguments* | 
`value (optional)` | the value to replace NaN, the default is 0
*Inputs* | 
`input` | Input tensor
`output` | Output tensor


### Code


[caffe2/operators/replace_nan_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/replace_nan_op.cc)

---



## ReservoirSampling


Collect  `DATA`  tensor into  `RESERVOIR`  of size  `num_to_collect` .  `DATA`  is assumed to be a batch.
 In case where 'objects' may be repeated in data and you only want at most one instance of each 'object' in the reservoir,  `OBJECT_ID`  can be given for deduplication. If  `OBJECT_ID`  is given, then you also need to supply additional book-keeping tensors. See input blob documentation for details.
 This operator is thread-safe.



### Interface


---------- | ----------
*Arguments* | 
`num_to_collect` | The number of random samples to append for each positive samples
*Inputs* | 
`RESERVOIR` | The reservoir; should be initialized to empty tensor
`NUM_VISITED` | Number of examples seen so far; should be initialized to 0
`DATA` | Tensor to collect from. The first dimension is assumed to be batch size. If the object to be collected is represented by multiple tensors, use `PackRecords` to pack them into single tensor.
`MUTEX` | Mutex to prevent data race
`OBJECT_ID` | (Optional, int64) If provided, used for deduplicating object in the reservoir
`OBJECT_TO_POS_MAP_IN` | (Optional) Auxillary bookkeeping map. This should be created from  `CreateMap` with keys of type int64 and values of type int32
`POS_TO_OBJECT_IN` | (Optional) Tensor of type int64 used for bookkeeping in deduplication
*Outputs* | 
`RESERVOIR` | Same as the input
`NUM_VISITED` | Same as the input
`OBJECT_TO_POS_MAP` | (Optional) Same as the input
`POS_TO_OBJECT` | (Optional) Same as the input


### Code


[caffe2/operators/reservoir_sampling.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reservoir_sampling.cc)

---



## ResetCounter


Resets a count-down counter with initial value specified by the 'init_count' argument.



### Interface


---------- | ----------
*Arguments* | 
`init_count` | Resets counter to this value, must be >= 0.
*Inputs* | 
`counter` | A blob pointing to an instance of a new counter.
*Outputs* | 
`previous_value` | (optional) Previous value of the counter.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)

---



## ResetCursor


Resets the offsets for the given TreeCursor. This operation is thread safe.



### Interface


---------- | ----------
*Inputs* | 
`cursor` | A blob containing a pointer to the cursor.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## Reshape


Reshape the input tensor similar to numpy.reshape.
 It takes a tensor as input and an optional tensor specifying the new shape.
When the second input is absent, an extra argument  `shape`  must be specified.
It outputs the reshaped tensor as well as the original shape.
 At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions. A dimension could also be 0, in which case the actual dimension value is going to be copied from the input tensor.



### Interface


---------- | ----------
*Arguments* | 
`shape` | New shape
*Inputs* | 
`data` | An input tensor.
`new_shape` | New shape.
*Outputs* | 
`reshaped` | Reshaped data.
`old_shape` | Original shape.


### Code


[caffe2/operators/reshape_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reshape_op.cc)

---



## ResizeLike


Produces tensor containing data of first input and shape of second input.



### Interface


---------- | ----------
*Inputs* | 
`data` | Tensor whose data will be copied into the output.
`shape_tensor` | Tensor whose shape will be applied to output.
*Outputs* | 
`output` | Tensor with data of input 0 and shape of input 1.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## ResizeNearest


Resizes the spatial dimensions of the input using nearest neighbor interpolation. The  `width_scale`  and  `height_scale`  arguments control the size of the output, which is given by: output_width = floor(input_width  * width_scale) output_height = floor(output_height *  height_scale) 


### Interface


---------- | ----------
*Arguments* | 
`width_scale` | Scale along width dimension
`height_scale` | Scale along height dimension
*Inputs* | 
`X` | Input tensor
*Outputs* | 
`Y` | Output tensor


### Code


[caffe2/operators/resize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/resize_op.cc)

---



## ResizeNearestGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`width_scale` | Scale along width dimension
`height_scale` | Scale along height dimension


### Code


[caffe2/operators/resize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/resize_op.cc)

---



## RetrieveCount


Retrieve the current value from the counter.



### Interface


---------- | ----------
*Inputs* | 
`counter` | A blob pointing to an instance of a counter.
*Outputs* | 
`count` | current count value.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/counter_ops.cc)

---



## ReversePackedSegs


Reverse segments in a 3-D tensor (lengths, segments, embeddings,), leaving paddings unchanged. This operator is used to reverse input of a recurrent neural network to make it a BRNN.
  


### Interface


---------- | ----------
*Inputs* | 
`data` | a 3-D (lengths, segments, embeddings,) tensor.
`lengths` | length of each segment.
*Outputs* | 
`reversed data` | a (lengths, segments, embeddings,) tensor with each segment reversedand paddings unchanged.


### Code


[caffe2/operators/reverse_packed_segs_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reverse_packed_segs_op.cc)

---



## RmsProp


Computes the RMSProp update ( [http://www.cs.toronto.edu/](http://www.cs.toronto.edu/) ~tijmen/csc321/slides/lecture_slides_lec6.pdf).
Concretely, given inputs (grad, mean_squares, mom, lr), computes:   

```
    mean_squares_o = mean_squares + (1 - decay) * (square(grad) - mean_squares)
    mom_o = momentum * mom + lr * grad / sqrt(epsilon + mean_squares_o)
    grad_o = mom_o

```

 Returns (grad_o, mean_squares_o, mom_o).



### Code


[caffe2/sgd/rmsprop_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/rmsprop_op.cc)

---



## RoIAlign


Region of Interest (RoI) align operation as used in Mask R-CNN.



### Interface


---------- | ----------
*Arguments* | 
`spatial_scale` | (float) default 1.0; Spatial scale of the input feature map X relative to the input image. E.g., 0.0625 if X has a stride of 16 w.r.t. the input image.
`pooled_h` | (int) default 1; Pooled output Y's height.
`pooled_w` | (int) default 1; Pooled output Y's width.
`sampling_ratio` | (int) default -1; number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If <= 0, then an adaptive number of grid points are used (computed as ceil(roi_width / pooled_w), and likewise for height).
*Inputs* | 
`X` | 4D feature map input of shape (N, C, H, W).
`RoIs` | 2D input of shape (R, 4 or 5) specifying R RoIs representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI coordinates are in the coordinate system of the input image. For inputs corresponding to a single image, batch index can be excluded to have just 4 columns.
*Outputs* | 
`Y` | 4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element is a pooled feature map cooresponding to the r-th RoI.


### Code


[caffe2/operators/roi_align_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_op.cc)

---



## RoIAlignGradient

No documentation yet.


### Interface


---------- | ----------
*Inputs* | 
`X` | See RoIPoolF.
`RoIs` | See RoIPoolF.
`dY` | Gradient of forward output 0 (Y)
*Outputs* | 
`dX` | Gradient of forward input 0 (X)


### Code


[caffe2/operators/roi_align_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_gradient_op.cc)

---



## RoIPool


Carries out ROI Pooling for Faster-RCNN.
Depending on the mode, there are multiple output cases:   

```
  Output case #1: Y, argmaxes (train mode)
  Output case #2: Y           (test mode)
```




### Interface


---------- | ----------
*Arguments* | 
`is_test` | If set, run in test mode and skip computation of argmaxes (used for gradient computation). Only one output tensor is produced. (Default: false).
`order` | A StorageOrder string (Default: "NCHW").
`pooled_h` | The pooled output height (Default: 1).
`pooled_w` | The pooled output width (Default: 1).
`spatial_scale` | Multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling (Default: 1.0).
*Inputs* | 
`X` | The input 4-D tensor of data. Only NCHW order is currently supported.
`rois` | RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].
*Outputs* | 
`Y` | RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_h, pooled_w).
`argmaxes` | Argmaxes corresponding to indices in X used for gradient computation. Only output if arg "is_test" is false.


### Code


[caffe2/operators/roi_pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_pool_op.cc)

---



## RoIPoolGradient

No documentation yet.


### Code


[caffe2/operators/roi_pool_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_pool_op.cc)

---



## RowMul


Given a matrix A and column vector w, the output is the multiplication of row i of A and element i of w, e.g. C[i][j] = A[i][j] * w[i]. This operator should be deprecated when the gradient operator of Mul with broadcast is implemented.



### Interface


---------- | ----------
*Inputs* | 
`mat` | The matrix
`w` | The column vector
*Outputs* | 
`output` | Output


### Code


[caffe2/operators/rowmul_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/rowmul_op.cc)

---



## RowWiseArgMax


 

```
    Given a 2D (N X D) input tensor, this operator returns a 2D (N X 1) output
    tensor with with the index of the maximum value in each row. If there are
    duplicate max values in a row the index of the first occurence is returned.
```

     


### Interface


---------- | ----------
*Inputs* | 
`X` | 2D (N X D) input tensor
*Outputs* | 
`Z` | 2D (N X 1) output tensor


### Code


[caffe2/operators/arg_max_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/arg_max_op.cc)

---



## RowWiseSparseAdagrad


 Given inputs (param, moment, indices, grad, lr), runs a modified sparse Adagrad update on (param, grad, moment[indices], lr), and returns (new_param, new_momwnr), where moment is a 1D tensor with length equal to the number of rows in param: shape(moment) == shape(param)[0]. Each element of moment is applied to an entire row of param, and the new moment is calculated by adding the average squared sum of gradients across each row. Note that indices must also be a 1D tensor indexing into the rows of param.
 


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | Default 1e-5
*Inputs* | 
`param` | Parameters to be updated
`moment` | Moment history
`indices` | Sparse indices
`grad` | Gradient computed
`lr` | learning rate
*Outputs* | 
`output_param` | Updated parameters
`output_moment_1` | Updated moment


### Code


[caffe2/sgd/adagrad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adagrad_op.cc)

---



## RowWiseSparseAdam


 Computes a modified Adam Update for the sparse case.
Given inputs (param, moment1, moment2, indices, grad, lr, iter), runs the Adam update on (param, moment1[indices], moment2[indices], lr, iter) and returns (new_param, new_moment1, new_moment2), where moment2 is a 1D tensor with length equal to the number of rows in param: shape(moment2) == shape(param)[0]. Each element of 

```
  moment2 is
```

 applied to an entire row of param, and the new moment2 values are calculated by averaging across the row.
 


### Interface


---------- | ----------
*Arguments* | 
`beta1` | Default 0.9
`beta2` | Default 0.999
`epsilon` | Default 1e-5
*Inputs* | 
`param` | Parameters to be updated
`moment_1` | First moment history
`moment_2` | Second moment history
`indices` | Sparse indices
`grad` | Gradient computed
`lr` | learning rate
`iter` | iteration number
*Outputs* | 
`output_param` | Updated parameters
`output_moment_1` | Updated first moment
`output_moment_2` | Updated second moment


### Code


[caffe2/sgd/adam_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adam_op.cc)

---



## Rowwise8BitQuantizedToFloat


Given uint8 tensor, quantized using 8bit row-wise quantization, and auxiliary scales and biases, this operator restores float tensor in the following way. We take input 8bits tensor of size 

```
  (m_1, m_2, ..., m_n), n >= 2, reshape it  into matrix of size
```

 (m_1, m_2 x... x m_n). We compute element r_{ij} of output matrix as r_{ij} * s_i + b_i and after this we reshape this output matrix into output tensor of size (m_1, m_2, ..., m_n).



### Interface


---------- | ----------
*Inputs* | 
`quantized_input` | quantized_input
`scale_bias` | Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row
*Outputs* | 
`None` | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## RowwiseMax

Compute row-wise max reduction of the input tensor.


### Interface


---------- | ----------
*Inputs* | 
`X` | A tenosr of dimensions batch_size x M x N to compute rowwise-max.
*Outputs* | 
`Y` | batch_size x M rowwise-max results matrix.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_ops.cc)

---



## RowwiseMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_ops.cc)

---



## SafeDequeueBlobs


Dequeue the blobs from queue. When the queue is closed and empty, the output status will be set to true which can be used as exit criteria for execution step.
The 1st input is the queue and the last output is the status. The rest are data blobs.



### Interface


---------- | ----------
*Arguments* | 
`num_records` | (default 1) If > 1, multiple records will be dequeued and tensors for each column will be concatenated. This requires all tensors in the records to be at least 1D, and to have the same inner dimensions.
*Inputs* | 
`queue` | The shared pointer for the BlobsQueue
*Outputs* | 
`blob` | The blob to store the dequeued data
`status` | Is set to 0/1 depending on the success of dequeue


### Code


[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)

---



## SafeEnqueueBlobs


Enqueue the blobs into queue. When the queue is closed and full, the output status will be set to true which can be used as exit criteria for execution step.
The 1st input is the queue and the last output is the status. The rest are data blobs.



### Interface


---------- | ----------
*Inputs* | 
`queue` | The shared pointer for the BlobsQueue


### Code


[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)

---



## Save


The Save operator saves a set of blobs to a db. It takes [1, infinity) number of inputs and has no output. The contents of the inputs are written into the db specified by the arguments.



### Interface


---------- | ----------
*Arguments* | 
`absolute_path` | (int, default 0) if set, use the db path directly and do not prepend the current root folder of the workspace.
`strip_prefix` | (string, default="") characters in the provided blob  names that match strip_prefix will be removed prior to saving. Also, characters that precede strip_prefix will be removed. Useful  for removing device scope from blob names.
`blob_name_overrides` | (list of strings) if set, used instead of original blob names. Must be the same length as number of blobs.
`db` | (string) the path to the db to load.
`db_type` | (string) the type of the db.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)

---



## Scale


Scale takes one input data (Tensor<float>) and produces one output data (Tensor<float>) whose value is the input data tensor scaled element-wise.



### Interface


---------- | ----------
*Arguments* | 
`scale` | (float, default 1.0) the scale to apply.


### Code


[caffe2/operators/scale_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/scale_op.cc)

---



## ScatterAssign


Update slices of the tensor in-place by overriding current value.
 Note: The op pretty much ignores the exact shapes of the input arguments and cares only about sizes. It's done for performance consideration to avoid unnecessary reshapes. Only first dimension of X_0 is important, let's call it N. If M is the total size of X_0 and K is the size of INDICES then X_i is assumed to be of shape K x (M / N) regardless of the real shape.
 Note: Each update in INDICES is applied independently which means that if duplicated elements are present in INDICES arbitrary one will win.
 Currently only works on CPU because of access to INDICES.



### Interface


---------- | ----------
*Inputs* | 
`DATA` | Tensor to be updated.
`INDICES` | 1-D list of indices on the first dimensionof X_0 that need to be updated
`SLICES` | Update slices, with shape len(INDICES) + shape(X_0)[1:]
*Outputs* | 
`DATA` | Has to be exactly the same tensor as the input 0


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## ScatterWeightedSum


Similar to WeightedSum, computes the weighted sum of several tensors, with the difference that inputs are sliced tensors. The first tensor has to be in-place and only slices of it on the first dimension as indexed by INDICES will be updated.
 Note: The op pretty much ignores the exact shapes of the input arguments and cares only about sizes. It's done for performance consideration to avoid unnecessary reshapes. Only first dimension of X_0 is important, let's call it N. If M is the total size of X_0 and K is the size of INDICES then X_i is assumed to be of shape K x (M / N) regardless of the real shape.
 Note: Each update in INDICES is applied independently which means that if duplicated elements are present in INDICES the corresponding slice of X_0 will be scaled multiple times. Manual collapsing of INDICES is required beforehand if necessary.
 Note: Updates are applied sequentially by inputs which might have undesired consequences if the input tensor is accessed concurrently by different op (e.g. when doing Hogwild). Other threads might see intermediate results even on individual slice level, e.g. X_0 scaled by weight_0 but without any updates applied.
 Currently only works on CPU because of access to INDICES.



### Interface


---------- | ----------
*Inputs* | 
`X_0` | Tensor to be updated.
`Weight_0` | Scalar weight for X_0, applied only to slices affected.
`INDICES` | 1-D list of indices on the first dimension of X_0 that need to be updated
`X_1` | Update slices, with shape len(INDICES) + shape(X_0)[1:]
`Weight_1` | Scalar weight for X_1 update
*Outputs* | 
`X_0` | Has to be exactly the same tensor as the input 0


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## SegmentIdsToLengths


Transfers a vector of segment ids to a vector of segment lengths. This operation supports non-consecutive segment ids. Segments not appearing in the input vector will have length 0. If the second input is provided, the number of segments = the size of its first dimension. Otherwise, the number of segments = the last index in the first input vector + 1.
 In general, for consecutive, zero-based segment IDs, this is the inverse operation of LengthsToSegmentIds, except that a vector of segment IDs cannot represent empty segments at the end (if the second input is absent).



### Interface


---------- | ----------
*Inputs* | 
`segment_ids` | 1-D int32_t or int64_t tensor of segment ids
`data (optional)` | if provided, number of segments = the size of its first dimension
*Outputs* | 
`lengths` | 1-D int64_t tensor of segment lengths


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## SegmentIdsToRanges


Transfers a vector of segment ids to a vector of segment ranges. This operation supports non-consecutive segment ids. Segments not appearing in the input vector will have length 0. If the second input is provided, the number of segments = the size of its first dimension. Otherwise, the number of segments = the last index in the first input vector + 1.



### Interface


---------- | ----------
*Inputs* | 
`segment_ids` | 1-D int32_t or int64_t tensor of segment ids
`data (optional)` | if provided, number of segments = the size of its first dimension
*Outputs* | 
`lengths` | 1-D int64_t tensor of segment lengths


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## SegmentOneHot


Given a sequence of indices, segmented by the lengths tensor, returns a matrix that has the elements in each sequence set to 1.0, and 0.0 everywhere else.



### Interface


---------- | ----------
*Inputs* | 
`lengths` | Size of each segment.
`indices` | Active indices, of size sum(lengths)
`index_size_tensor` | Size of the index
*Outputs* | 
`one_hots` | Matrix of size len(lengths) x index_size


### Code


[caffe2/operators/one_hot_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc)

---



## Selu


Selu takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the function, y = scale *(alpha_* e^x-alpha_ if x < 0 else x), is applied to the tensor elementwise.



### Interface


---------- | ----------
*Arguments* | 
`alpha` | (float) default to 1.6732~; affects the activation function itself. This should go with the weight initialization in the paper.  See https://arxiv.org/abs/1706.02515 
`scale` | (float) default to 1.0507~; affects the activation function itself.
*Inputs* | 
`X` | input tensor
*Outputs* | 
`Y` | input tensor


### Code


[caffe2/operators/selu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/selu_op.cc)

---



## SeluGradient


SeluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the selu function.



### Interface


---------- | ----------
*Arguments* | 
`alpha` | (float) default to 1.6732~; affects the activation function itself.This should go with the weight initialization in the paper.  See https://arxiv.org/abs/1706.02515 
`scale` | (float) default to 1.0507~; affects the activation function itself.
*Inputs* | 
`Y` | input tensor
`dY` | input tensor


### Code


[caffe2/operators/selu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/selu_op.cc)

---



## SendTensor


Sends the tensor to another node.



### Interface


---------- | ----------
*Arguments* | 
`dst` | The rank to send the tensor to.
`tag` | (int) a tag to send the tensor with.
`raw_buffer` | (bool) if set, only send the content and assume that the receiver has already known the tensor's shape and information.
*Inputs* | 
`comm_world` | The common world.
`X` | A tensor to be allgathered.
`dst` | An int CPUtensor of size 1 specifying the rank. If given, this overrides the 'to' argument of the op.
`tag` | An int CPUtensor of size 1 specifying the tag to send the tensor with. This overrides the 'tag' argument of the op.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/communicator_op.cc)

---



## SequenceMask


Mask op designed for use in attention mechanisms for sequence modeling tasks.
Supports batching: given batch_dim, collapses dims 0 through batch_dim into a single dimension, e.g. if tensor dims are [4,2,1,3,4] and batch_dim=2, first collapse tensor to [4 *2* 1,3,4], then mask each batch [i,:,:].
  Two current operating modes:   1) Given a 2D input tensor and 1D tensor of sequence lengths, for each row i in the input tensor, set elements in that row to -inf if their column index j >= sequence_lengths[i]. This mode takes two inputs and argument mode = 'sequence'   2) Triangular mask. Given row index i and column index j, set elements to -inf given the following conditions:   

```
      mode='upper', x_ij = -inf if j < i
      mode='lower', x_ij = -inf if j > i
      mode='upperdiag', x_ij = -inf if j <= i
      mode='lowerdiag', x_ij = -inf if j >= i

```

 This mode takes one input.
  3) Window Mask. Given a 2D input tensor and 1D tensor of window centers, for each row i in the input tensor, set elements in that row to -inf if their column index j outside [center - radius, center + radius].
This mode takes two inputs and argument mode = 'sequence'.
Argument 'radius' should be provided.



### Interface


---------- | ----------
*Arguments* | 
`mode` | (string) Mode selection. Possible values: 'sequence', 'upper', 'lower', 'upperdiag', 'lowerdiag'
`axis` | (int) Beginning axis of row elements. All dimensions to the left will be treated as row indices and those to the right (inclusive) will be treated as column indices in the 2D mask
`grad` | (bool) operate in gradient mode
`radius` | (int) radius of windows in window mode
`batch` | (int) batch dimension of tensor (optional)
`repeat_from_axis` | (int) used when mask should be repeated for one or more data dimensions (beginning at this axis).  (currently only supported for sequence mode without batch argument)
*Inputs* | 
`input` | Tensor to apply masking to
`sequence_lengths` | 1D Tensor of sequence lengths for mode #1
*Outputs* | 
`masked_tensor` | Input tensor with masking applied


### Code


[caffe2/operators/boolean_mask_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/boolean_mask_ops.cc)

---



## Shape

Produce a 1D int64 tensor with the shape of the input tensor.


### Code


[caffe2/operators/shape_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/shape_op.cc)

---



## Sigmoid


Sigmoid takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the tensor elementwise.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D output tensor


### Code


[caffe2/operators/sigmoid_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sigmoid_op.cc)

---



## SigmoidCrossEntropyWithLogits


Given two matrices logits and targets, of same shape, (batch_size, num_classes), computes the sigmoid cross entropy between the two.
Returns a tensor of shape (batch_size,) of losses for each example.



### Interface


---------- | ----------
*Inputs* | 
`logits` | matrix of logits for each example and class.
`targets` | matrix of targets, same shape as logits.
*Outputs* | 
`xentropy` | Vector with the total xentropy for each example.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## SigmoidCrossEntropyWithLogitsGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## SigmoidGradient


SigmoidGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the sigmoid function.



### Code


[caffe2/operators/sigmoid_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sigmoid_op.cc)

---



## Sign

Computes sign for each element of the input: -1, 0 or 1.


### Code


[caffe2/operators/math_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/math_ops.cc)

---



## Sin


Calculates the sine of the given input tensor, element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The sine of the input tensor computed element-wise


### Code


[caffe2/operators/sin_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sin_op.cc)

---



## SinGradient

No documentation yet.


### Code


[caffe2/operators/sin_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sin_op.cc)

---



## SinusoidPositionEncoding


Calculates a sinusoid position encoding tensor as described in  [https://arxiv.org/abs/1706.03762.](https://arxiv.org/abs/1706.03762.)  Takes a 2-D tensor (of size M x K) of positions as input, the embedding size as an argument, and outputs a position encoding tensor of size (M x K x embedding_size). Here M is typically the max sequence length and K is typically the batch size.
The input tensor must satisfy input[m, 0] == input[m, k] for all k.
 Encoded as amplitude  * SIN(pos/alpha^(i/embedding_size)) if i is even, else amplitude *  COS(pos/alpha^(i/embedding_size)). Here, pos is the position, alpha and amplitude are tuning parameters, i is the current dimension for the embedding, and embedding_size is the number of total dimensions in the embedding.



### Interface


---------- | ----------
*Arguments* | 
`embedding_size` | Desired embedding size/number of dimensions -- defaults to 100
`alpha` | Sinusoid tuning parameter -- defaults to 10000
`amplitude` | Amplitude of Sin/Cos output
*Inputs* | 
`positions` | 2-D tensor of positions to be encoded
*Outputs* | 
`output` | 3-D tensor representing the positional encoding


### Code


[caffe2/operators/sinusoid_position_encoding_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sinusoid_position_encoding_op.cc)

---



## Size

Return a 1D tensor of type int64 that contains the number of elements of the input tensor


### Interface


---------- | ----------
*Inputs* | 
`tensor` | Tensor to calculate number of elements
*Outputs* | 
`output` | 1D tensor of type int64 that contains the number of elements in the input tensor.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## Slice


Produces a slice of the input tensor. Currently, only slicing in a single dimension is supported.
Slices are passed as 2 1D vectors or as two keyword argument lists with starting and end indices for each dimension of the input  `data`  tensor. If a negative value is passed for any of the start or end indices, it represents the number of elements before the end of that dimension. End indices are non-inclusive unless negative (end index -1 means up to and including the last element).
  Example:   

```
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
```




### Interface


---------- | ----------
*Arguments* | 
`starts` | List of starting indices
`ends` | List of ending indices
*Inputs* | 
`data` | Tensor of data to extract slices from.
`starts` | 1D tensor: start-indices for each dimension of data.
`ends` | 1D tensor: end-indices for each dimension of data.
*Outputs* | 
`output` | Sliced data tensor.


### Code


[caffe2/operators/slice_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/slice_op.cc)

---



## SliceGradient

No documentation yet.


### Code


[caffe2/operators/slice_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/slice_op.cc)

---



## Snapshot

No documentation yet.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/load_save_op.cc)

---



## Softmax


The operator computes the softmax normalized values for each layer in the batch  of the given input. The input is a 2-D tensor (Tensor<float>) of size (batch_size x input_feature_dimensions). The output tensor has the same shape and contains the softmax normalized values of the corresponding input.
 X does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor X \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is the axis provided, then X will be coerced into a 2-dimensional tensor with dimensions [a_0  * ... *  a_{k-1}, a_k  * ... *  a_{n-1}]. For the default case where axis=1, this means the X tensor will be coerced into a 2D tensor of dimensions [a_0, a_1  * ... *  a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1  * ... *  a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator will throw errors.



### Interface


---------- | ----------
*Arguments* | 
`axis` | (int) default to 1; describes the axis of the inputs when coerced to 2D; defaults to one because the 0th axis most likely describes the batch_size
*Inputs* | 
`input` | The input tensor that's coerced into a 2D matrix of size (NxD) as described above.
*Outputs* | 
`output` | The softmax normalized output values with the same shape as input tensor.


### Code


[caffe2/operators/softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_op.cc)

---



## SoftmaxGradient

No documentation yet.


### Code


[caffe2/operators/softmax_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_op.cc)

---



## SoftmaxWithLoss


Combined Softmax and Cross-Entropy loss operator.
The operator computes the softmax normalized values for each layer in the batch of the given input, after which cross-entropy loss is computed. This operator is numerically more stable than separate Softmax and CrossEntropy ops.
The inputs are a 2-D tensor (Tensor<float>) of size (batch_size x input_feature_dimensions) and tensor of labels (ground truth).
Output is tensor with the probability for each label for each example (N x D) and averaged loss (scalar).
Use parameter label_prob=1 to enable inputting labels as a probability distribution.
Optional third input blob can be used to weight the samples for the loss.



### Interface


---------- | ----------
*Inputs* | 
`logits` | Unscaled log probabilities
`labels` | Ground truth
`weight_tensor` | Optional blob to be used to weight the samples for the loss.
*Outputs* | 
`softmax` | Tensor with softmax cross entropy loss
`loss` | Average loss


### Code


[caffe2/operators/softmax_with_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_with_loss_op.cc)

---



## SoftmaxWithLossGradient

No documentation yet.


### Code


[caffe2/operators/softmax_with_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softmax_with_loss_op.cc)

---



## Softplus


Softplus takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to the tensor elementwise.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/softplus_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softplus_op.cc)

---



## SoftplusGradient

No documentation yet.


### Code


[caffe2/operators/softplus_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softplus_op.cc)

---



## Softsign


Calculates the softsign (x/1+|x|) of the given input tensor element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.



### Interface


---------- | ----------
*Inputs* | 
`input` | 1-D input tensor
*Outputs* | 
`output` | The softsign (x/1+|x|) values of the input tensor computed element-wise


### Code


[caffe2/operators/softsign_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softsign_op.cc)

---



## SoftsignGradient


Calculates the softsign gradient (sgn(x)/(1+|x|)^2) of the given input tensor element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | 1-D input tensor
`input` | 1-D input tensor
*Outputs* | 
`output` | The softsign gradient (sgn(x)/(1+|x|)^2) values of the input tensor computed element-wise


### Code


[caffe2/operators/softsign_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/softsign_op.cc)

---



## SortAndShuffle


Compute the sorted indices given a field index to sort by and break the sorted indices into chunks of shuffle_size * batch_size and shuffle each chunk, finally we shuffle between batches. If sort_by_field_idx is -1 we skip sort.
 For example, we have data sorted as 1,2,3,4,5,6,7,8,9,10,11,12  and batchSize = 2 and shuffleSize = 3, when we shuffle we get: [3,1,4,6,5,2] [12,10,11,8,9,7]  After this we will shuffle among different batches with size 2 [3,1],[4,6],[5,2],[12,10],[11,8],[9,7]  We may end up with something like [9,7],[5,2],[12,10],[4,6],[3,1],[11,8]  Input(0) is a blob pointing to a TreeCursor, and [Input(1),... Input(num_fields)] a list of tensors containing the data for each field of the dataset.
 SortAndShuffle is thread safe.



### Interface


---------- | ----------
*Inputs* | 
`cursor` | A blob containing a pointer to the cursor.
`dataset_field_0` | First dataset field
*Outputs* | 
`indices` | Tensor containing sorted indices.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## SortedSegmentMean


Applies 'Mean' to each segment of input tensor. Segments need to be sorted and contiguous. See also UnsortedSegmentMean that doesn't have this requirement.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Mean computes the element-wise mean of the input slices. Operation doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeLogMeanExp


Applies 'LogMeanExp' to each segment of input tensor. In order to allow for more efficient implementation of 'LogMeanExp', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 LogMeanExp computes the element-wise log of the mean of exponentials of input slices. Operation doesn't change the shape of individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor to be aggregated
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeLogMeanExpGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeLogSumExp


Applies 'LogSumExp' to each segment of input tensor. In order to allow for more efficient implementation of 'LogSumExp', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 LogSumExp computes the element-wise log of the sum of exponentials of input slices. Operation doesn't change the shape of individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor to be aggregated
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeLogSumExpGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeMax


Applies 'Max' to each segment of input tensor. In order to allow for more efficient implementation of 'Max', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Max computation is done element-wise, so that each element of the output slice corresponds to the max value of the respective elements in the input slices. Operation doesn't change the shape of individual blocks. This implementation imitates torch nn.Max operator. If the maximum value occurs more than once, the operator will return the first occurence of value. When computing the gradient using the backward propagation, the gradient input corresponding to the first occurence of the maximum value will be used.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor to be aggregated
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeMaxGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeMean


Applies 'Mean' to each segment of input tensor. In order to allow for more efficient implementation of 'Mean', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Mean computation is done element-wise, so that each element of the output slice corresponds to the average value of the respective elements in the input slices. Operation doesn't change the shape of individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor to be aggregated
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeSum


Applies 'Sum' to each segment of input tensor. In order to allow for more efficient implementation of 'Sum', the input segments have to be contiguous and non-empty.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor to be aggregated
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated tensor with the first dimension of K and the other dimentsions inherited from DATA


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentSum


Applies 'Sum' to each segment of input tensor. Segments need to be sorted and contiguous. See also UnsortedSegmentSum that doesn't have this requirement.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentWeightedSum


Applies 'WeightedSum' to each segment of input tensor. Segments need to be sorted and contiguous. See also UnsortedSegmentWeightedSum that doesn't have this requirement.
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`grad_on_weights` | Produce also gradient for `weights`. For now it's only supported in `Lengths`-based operators
*Inputs* | 
`DATA` | Input tensor for the summation
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the number of slices
`SEGMENT_IDS` | Vector with the same length as the first dimension of DATA and values in the range 0..K-1 and in increasing order that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SpaceToBatch


 SpaceToBatch for 4-D tensors of type T.
 Zero-pads and then rearranges (permutes) blocks of spatial data into batch. More specifically, this op outputs a copy of the input tensor where values from the height and width dimensions are moved to the batch dimension. After the zero-padding, both height and width of the input must be divisible by the block size.
 


### Code


[caffe2/operators/space_batch_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/space_batch_op.cc)

---



## SparseAdagrad


 Given inputs (param, moment, indices, grad, lr), runs the dense AdaGrad update on (param, grad, moment[indices], lr), and returns (new_param, new_moment) as in the dense case.
 


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | Default 1e-5
*Inputs* | 
`param` | Parameters to be updated
`moment` | Moment history
`indices` | Sparse indices
`grad` | Gradient computed
`lr` | learning rate
*Outputs* | 
`output_param` | Updated parameters
`output_moment_1` | Updated moment


### Code


[caffe2/sgd/adagrad_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adagrad_op.cc)

---



## SparseAdam


 Computes the Adam Update for the sparse case.
Given inputs (param, moment1, moment2, indices, grad, lr, iter), runs the dense Adam on (param, moment1[indices], momemnt2[indices], lr, iter) and returns (new_param, new_moment1, new_moment2) as in dense case  


### Interface


---------- | ----------
*Arguments* | 
`beta1` | Default 0.9
`beta2` | Default 0.999
`epsilon` | Default 1e-5
*Inputs* | 
`param` | Parameters to be updated
`moment_1` | First moment history
`moment_2` | Second moment history
`indices` | Sparse indices
`grad` | Gradient computed
`lr` | learning rate
`iter` | iteration number
*Outputs* | 
`output_param` | Updated parameters
`output_moment_1` | Updated first moment
`output_moment_2` | Updated second moment


### Code


[caffe2/sgd/adam_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/adam_op.cc)

---



## SparseFtrl

No documentation yet.


### Code


[caffe2/sgd/ftrl_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/ftrl_op.cc)

---



## SparseLengthsIndicesInGradientSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsIndicesInGradientWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsMean


Pulls in slices of the input tensor, groups them into segments and applies 'Mean' to each segment. Segments are defined by their LENGTHS.
 This op is basically Gather and LengthsMean fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 LENGTHS is a vector that defines slice sizes by first dimention of DATA. Values belonging to the same segment are aggregated together. sum(LENGTHS) has to match INDICES size.
 The first dimension of the output is equal to the number of input segment, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Mean computes the element-wise mean of the input slices. Operation doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Non negative vector with sum of elements equal to INDICES length
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsMean8BitsRowwise


Variation of SparseLengthsMean operator, where DATA is stored using 8bits. DATA was quantized with 8Bit row-wise quantization (see doc to FloatToRowwiseQuantized8Bits operator). To restore DATA from 8Bit, we use additional input that stores scales and biases.



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
`scale_bias` | Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## SparseLengthsMeanFused8BitRowwise


Performs the same operation as SparseLengthsMean, but operating on 8-bit rowwise quantized matrices with fused storage (where each row stores quantized values, and then 4-byte scale and 4-byte bias).



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToFused8BitRowwiseQuantized
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc)

---



## SparseLengthsMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsSum


Pulls in slices of the input tensor, groups them into segments and applies 'Sum' to each segment. Segments are defined by their LENGTHS.
 This op is basically Gather and LengthsSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 LENGTHS is a vector that defines slice sizes by first dimention of DATA. Values belonging to the same segment are aggregated together. sum(LENGTHS) has to match INDICES size.
 The first dimension of the output is equal to the number of input segment, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Non negative vector with sum of elements equal to INDICES length
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsSum8BitsRowwise


Variation of SparseLengthsSum operator, where DATA is stored using 8bits. DATA was quantized with 8Bit row-wise quantization (see doc to FloatToRowwiseQuantized8Bits operator). To restore DATA from 8Bit, we use additional input that stores scales and biases.



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
`scale_bias` | Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## SparseLengthsSumFused8BitRowwise


Performs the same operation as SparseLengthsSum, but operating on 8-bit rowwise quantized matrices with fused storage (where each row stores quantized values, and then 4-byte scale and 4-byte bias).



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToFused8BitRowwiseQuantized
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc)

---



## SparseLengthsSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsWeightedMean8BitsRowwise


Variation of SparseLengthsWeightedMean operator, where DATA is stored using 8bits. DATA was quantized with 8Bit row-wise quantization (see doc to FloatToRowwiseQuantized8Bits operator). To restore DATA from 8Bit, we use additional input that stores scales and biases.



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the length of INDICES
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
`scale_bias` | Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## SparseLengthsWeightedSum


Pulls in slices of the input tensor, groups them into segments and applies 'WeightedSum' to each segment. Segments are defined by their LENGTHS.
 This op is basically Gather and LengthsWeightedSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 LENGTHS is a vector that defines slice sizes by first dimention of DATA. Values belonging to the same segment are aggregated together. sum(LENGTHS) has to match INDICES size.
 The first dimension of the output is equal to the number of input segment, i.e.  `len(LENGTHS)` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`grad_on_weights` | Produce also gradient for `weights`. For now it's only supported in `Lengths`-based operators
*Inputs* | 
`DATA` | Input tensor for the summation
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the number of slices
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Non negative vector with sum of elements equal to INDICES length
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsWeightedSum8BitsRowwise


Variation of SparseLengthsWeightedSum operator, where DATA is stored using 8bits. DATA was quantized with 8Bit row-wise quantization (see doc to FloatToRowwiseQuantized8Bits operator). To restore DATA from 8Bit, we use additional input that stores scales and biases.



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the length of INDICES
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
`scale_bias` | Matrix of floats, each row r_i of which stores a pair s_i, b_i -- scale and bias for i-th row
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## SparseLengthsWeightedSumFused8BitRowwise


Performs the same operation as SparseLengthsWeightedSum, but operating on 8-bit rowwise quantized matrices with fused storage (where each row stores quantized values, and then 4-byte scale and 4-byte bias).



### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToFused8BitRowwiseQuantized
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
`WEIGHTS` | Vector of weights to scale rows of DATA with before reduction
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc)

---



## SparseLengthsWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsWeightedSumWithMainInputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseMomentumSGDUpdate


 Performs a momentum SGD update analogous to MomentumSGDUpdate, but using a GradientSlice and indices into the full param and momentum tables. Both param and momentum should be in-place (corresponding inputs and outputs should be the same blobs).
   


### Interface


---------- | ----------
*Arguments* | 
`momentum` | Momentum hyperparameter.
`nesterov` | (boolean) Whether to use Nesterov Accelerated Gradient.
*Inputs* | 
`grad` | GradientSlice with gradients for updated indices.
`moment` | Momentum blob, same shape as param.
`lr` | Learning rate.
`param` | Full parameter blob.
`indices` | Indices (in first dimension of param) where updates are performed.
*Outputs* | 
`output_grad` | Adjusted gradient.
`output_moment` | Updated momentum.
`output_param` | Updated parameter


### Code


[caffe2/sgd/momentum_sgd_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/momentum_sgd_op.cc)

---



## SparseNormalize


Given a sparse matrix, apply max_norm or constant_norm sparse regularization.



### Interface


---------- | ----------
*Arguments* | 
`use_max_norm` | A bool variable to control whether to use max norm     or constant norm. When use_max_norm = false, constant norm is used so that     all the embedding vectors are scaled to have a L2 norm equals to A     (see blow arugment norm=A). If use_max_norm = true,     max norm is used so that embedding is scaled so that its l2 norm is no larger     than A. If an embedding's norm is less than A originally,     the embedding is left unchanged.    The default is True.
`norm` | L2 norm of the embedding. The default is 1.0.
*Inputs* | 
`param` | Parameters to be normalized
`indices` | Sparse indices
`grad` | Gradient computed
*Outputs* | 
`output_param` | Normalized parameters


### Code


[caffe2/operators/sparse_normalize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sparse_normalize_op.cc)

---



## SparseSortedSegmentMean


Pulls in slices of the input tensor, groups them into segments and applies 'Mean' to each segment. Segments need to be sorted and contiguous. See also SparseUnsortedSegmentMean that doesn't have this requirement.
 This op is basically Gather and SortedSegmentMean fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Mean computes the element-wise mean of the input slices. Operation doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`SEGMENT_IDS` | Vector with the same length as INDICES and values in the range 0..K-1 and in increasing order that maps each slice of DATA referenced by INDICES to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentSum


Pulls in slices of the input tensor, groups them into segments and applies 'Sum' to each segment. Segments need to be sorted and contiguous. See also SparseUnsortedSegmentSum that doesn't have this requirement.
 This op is basically Gather and SortedSegmentSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`SEGMENT_IDS` | Vector with the same length as INDICES and values in the range 0..K-1 and in increasing order that maps each slice of DATA referenced by INDICES to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentWeightedSum


Pulls in slices of the input tensor, groups them into segments and applies 'WeightedSum' to each segment. Segments need to be sorted and contiguous. See also SparseUnsortedSegmentWeightedSum that doesn't have this requirement.
 This op is basically Gather and SortedSegmentWeightedSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 The first dimension of the output is equal to the number of input segments, i.e.  `SEGMENT_IDS[-1]+1` . Other dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`grad_on_weights` | Produce also gradient for `weights`. For now it's only supported in `Lengths`-based operators
*Inputs* | 
`DATA` | Input tensor for the summation
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the number of slices
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`SEGMENT_IDS` | Vector with the same length as INDICES and values in the range 0..K-1 and in increasing order that maps each slice of DATA referenced by INDICES to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of K (the number of segments).


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseToDense


Convert sparse representations to dense with given indices.
 Transforms a sparse representation of map<id, value> represented as  `indices`  vector and  `values`  tensor into a compacted tensor where the first dimension is determined by the first dimension of the 3rd input if it is given or the max index. Missing values are filled with zeros.
 The op supports duplicated indices and performs summation over corresponding values. This behavior is useful for converting GradientSlices into dense representation.
 After running this op:   

```
  output[indices[i], :] += values[i]  # sum over all indices[i] equal to the index
  output[j, ...] = 0 if j not in indices
```




### Interface


---------- | ----------
*Inputs* | 
`indices` | 1-D int32/int64 tensor of concatenated ids of data
`values` | Data tensor, first dimension has to match `indices`, basic numeric types are supported
`data_to_infer_dim` | Optional: if provided, the first dimension of output is the first dimension of this tensor.
*Outputs* | 
`output` | Output tensor of the same type as `values` of shape `[len(lengths), len(mask)] + shape(default_value)` (if `lengths` is not provided the first dimension is omitted)


### Code


[caffe2/operators/sparse_to_dense_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sparse_to_dense_op.cc)

---



## SparseToDenseMask


Convert sparse representations to dense with given indices.
 Transforms a sparse representation of map<id, value> represented as  `indices`  vector and  `values`  tensor into a compacted tensor where the first dimension corresponds to each id provided in mask argument. Missing values are filled with the value of  `default_value` . After running this op:   

```
  output[j, :] = values[i] # where mask[j] == indices[i]
  output[j, ...] = default_value # when mask[j] doesn't appear in indices

```

 If  `lengths`  is provided and not empty, and extra "batch" dimension is prepended to the output.
  `values`  and  `default_value`  can have additional matching dimensions, operation is performed on the entire subtensor in thise case.
 For example, if  `lengths`  is supplied and  `values`  is 1-D vector of floats and  `default_value`  is a float scalar, the output is going to be a float matrix of size  `len(lengths) X len(mask)`  


### Interface


---------- | ----------
*Arguments* | 
`mask` | list(int) argument with desired ids on the 'dense' output dimension
`return_presence_mask` | bool whether to return presence mask, false by default
*Inputs* | 
`indices` | 1-D int32/int64 tensor of concatenated ids of data
`values` | Data tensor, first dimension has to match `indices`
`default_value` | Default value for the output if the id is not present in `indices`. Must have the same type as `values` and the same shape, but without the first dimension
`lengths` | Optional lengths to represent a batch of `indices` and `values`.
*Outputs* | 
`output` | Output tensor of the same type as `values` of shape `[len(lengths), len(mask)] + shape(default_value)` (if `lengths` is not provided the first dimension is omitted)
`presence_mask` | Bool tensor of shape `[len(lengths), len(mask)]` (if `lengths` is not provided the first dimension is omitted). True when a value for given id was present, false otherwise.


### Code


[caffe2/operators/sparse_to_dense_mask_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sparse_to_dense_mask_op.cc)

---



## SparseToDenseMaskGradient


The output is the gradient of the input value from SparseToDenseMask. The gradient for default_value has not been implemented.



### Code


[caffe2/operators/sparse_to_dense_mask_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sparse_to_dense_mask_op.cc)

---



## SparseUnsortedSegmentMean


Pulls in slices of the input tensor, groups them into segments and applies 'Mean' to each segment. Segments ids can appear in arbitrary order (unlike in SparseSortedSegmentMean).
 This op is basically Gather and UnsortedSegmentMean fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Mean computes the element-wise mean of the input slices. Operation doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`SEGMENT_IDS` | Integer vector with the same length as INDICES that maps each slice of DATA referenced by INDICES to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of equal to the number of segments.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentSum


Pulls in slices of the input tensor, groups them into segments and applies 'Sum' to each segment. Segments ids can appear in arbitrary order (unlike in SparseSortedSegmentSum).
 This op is basically Gather and UnsortedSegmentSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`SEGMENT_IDS` | Integer vector with the same length as INDICES that maps each slice of DATA referenced by INDICES to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of equal to the number of segments.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentWeightedSum


Pulls in slices of the input tensor, groups them into segments and applies 'WeightedSum' to each segment. Segments ids can appear in arbitrary order (unlike in SparseSortedSegmentWeightedSum).
 This op is basically Gather and UnsortedSegmentWeightedSum fused together.
 INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. INDICES represent which slices of DATA need to be pulled in.
 SEGMENT_IDS is a vector that maps each referenced slice of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together. SEGMENT_IDS should have the same dimension as INDICES.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`grad_on_weights` | Produce also gradient for `weights`. For now it's only supported in `Lengths`-based operators
*Inputs* | 
`DATA` | Input tensor for the summation
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the number of slices
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`SEGMENT_IDS` | Integer vector with the same length as INDICES that maps each slice of DATA referenced by INDICES to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of equal to the number of segments.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SpatialBN


Carries out spatial batch normalization as described in the paper  [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)  . Depending on the mode it is being run, there are multiple cases for the number of outputs, which we list below:   Output case #1:  

```
  Y, mean, var, saved_mean, saved_var (training mode)


```

 Output case #2:  

```
  Y (test mode)
```




### Interface


---------- | ----------
*Arguments* | 
`is_test` | If set to nonzero, run spatial batch normalization in test mode.
`epsilon` | The epsilon value to use to avoid division by zero.
`order` | A StorageOrder string.
`momentum` | Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum)
`num_batches` | (Optional) Specifies the number of batches to apply normalization on. Requires specifying the optional sums and sumsq inputs that provide statistics across multiple batches from which mean and variance can be determined.
*Inputs* | 
`X` | The input 4-dimensional tensor of shape NCHW or NHWC depending on the order parameter.
`scale` | The scale as a 1-dimensional tensor of size C to be applied to the output.
`bias` | The bias as a 1-dimensional tensor of size C to be applied to the output.
`mean` | The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size C.
`var` | The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size C.
`sums` | (optional) Per-channel sums of elements to be used to determine the mean and variance for this batch
`sumsq` | (optional) Per-channel sum of elements squared per channel to be used to determine the variance for this batch
*Outputs* | 
`Y` | The output 4-dimensional tensor of the same shape as X.
`mean` | The running mean after the spatial BN operator. Must be in-place with the input mean. Should not be used for testing.
`var` | The running variance after the spatial BN operator. Must be in-place with the input var. Should not be used for testing.
`saved_mean` | Saved mean used during training to speed up gradient computation. Should not be used for testing.
`saved_var` | Saved variance used during training to speed up gradient computation. Should not be used for testing.


### Code


[caffe2/operators/spatial_batch_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/spatial_batch_norm_op.cc)

---



## SpatialBNGradient

No documentation yet.


### Code


[caffe2/operators/spatial_batch_norm_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/spatial_batch_norm_gradient_op.cc)

---



## SpatialSoftmaxWithLoss


Combined Spatial Softmax and Cross-Entropy loss operator.
Similar to SoftmaxWithLoss, this operator computes the spatial softmax normalized values for each layer in the batch of the given input, after which cross-entropy loss is computed. This operator is numerically more stable than separate Softmax and CrossEntropy ops. The inputs are a 2-D tensor (Tensor<float>) of size (batch_size x input_feature_dimensions) and tensor of labels (ground truth).
Output is tensor with the probability for each label in a pixel for each example (N x D x W x H) and averaged loss (scalar).
For spatial softmax, weighting is by x,y position of the input.



### Interface


---------- | ----------
*Inputs* | 
`logits` | Unscaled log probabilities
`labels` | Ground truth
`weight_tensor` | Optional blob to be used to weight the samples for the loss. With        spatial set, weighting is by x,y of the input
*Outputs* | 
`softmax` | Tensor with softmax cross entropy loss
`loss` | Average loss


### Code


[caffe2/operators/spatial_softmax_with_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/spatial_softmax_with_loss_op.cc)

---



## SpatialSoftmaxWithLossGradient

No documentation yet.


### Code


[caffe2/operators/spatial_softmax_with_loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/spatial_softmax_with_loss_op.cc)

---



## Split


Split a tensor into a list of tensors, along the specified 'axis'. The lengths of the split can be specified using argument 'split' or optional second input blob to the operator. Otherwise, the tensor is split to equal sized parts.



### Interface


---------- | ----------
*Arguments* | 
`axis` | Which axis to split on
`split` | length of each output
`order` | Either NHWC or NCWH, will split on C axis, defaults to NCHW
*Inputs* | 
`input` | The tensor to split
`split` | Optional list of output lengths (see also arg 'split')


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/concat_split_op.cc)

---



## Sqr

Square (x^2) the elements of the input


### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | Squared elements of the input


### Code


[caffe2/operators/math_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/math_ops.cc)

---



## Sqrt


Computes the element-wise sqrt of the input.



### Interface


---------- | ----------
*Inputs* | 
`X` | ND input tensor
*Outputs* | 
`Y` | ND input tensor


### Code


[caffe2/operators/sqrt_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/sqrt_op.cc)

---



## SquareRootDivide


Given DATA tensor with first dimension N and SCALE vector of the same size N produces an output tensor with same dimensions as DATA. Which consists of DATA slices. i-th slice is divided by sqrt(SCALE[i]) elementwise. If SCALE[i] == 0 output slice is identical to the input one (no scaling)  Example:   

```
  Data = [
    [2.0, 4.0],
    [9.0, 12.0]
  ]

  SCALE = [4, 9]

  OUTPUT = [
    [1.0, 2.0],
    [3.0, 4.0]
  ]

```




### Code


[caffe2/operators/square_root_divide_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/square_root_divide_op.cc)

---



## SquaredL2Distance


Given two input float tensors X, Y, and produces one output float tensor of the L2 difference between X and Y that is computed as ||(X - Y)^2 / 2||.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D or 2D input tensor
`Y` | 1D or 2D input tensor (must have the same shape as X)
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## SquaredL2DistanceGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/distance_op.cc)

---



## Squeeze


Remove single-dimensional entries from the shape of a tensor.
Takes a parameter  `dims`  with a list of dimension to squeeze.
If the same blob is provided in input and output, the operation is copy-free.
This is the exact inverse operation of ExpandDims given the same  `dims`  arg.



### Interface


---------- | ----------
*Inputs* | 
`data` | Tensors with at least max(dims) dimensions.
*Outputs* | 
`squeezed` | Reshaped tensor with same data as input.


### Code


[caffe2/operators/expand_squeeze_dims_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/expand_squeeze_dims_op.cc)

---



## StatRegistryCreate


Create a StatRegistry object that will contain a map of performance counters keyed by name. A StatRegistry is used to gather and retrieve performance counts throughout the caffe2 codebase.



### Interface


---------- | ----------
*Outputs* | 
`handle` | A Blob pointing to the newly created StatRegistry.


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## StatRegistryExport

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`reset` | (default true) Whether to atomically reset the counters afterwards.
*Inputs* | 
`handle` | If provided, export values from given StatRegistry.Otherwise, export values from the global singleton StatRegistry.
*Outputs* | 
`keys` | 1D string tensor with exported key names
`values` | 1D int64 tensor with exported values
`timestamps` | The unix timestamp at counter retrieval.


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## StatRegistryUpdate


Update the given StatRegistry, or the global StatRegistry, with the values of counters for the given keys.



### Interface


---------- | ----------
*Inputs* | 
`keys` | 1D string tensor with the key names to update.
`values` | 1D int64 tensor with the values to update.
`handle` | If provided, update the given StatRegistry. Otherwise, update the global singleton.


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## StopGradient


StopGradient is a helper operator that does no actual numerical computation, and in the gradient computation phase stops the gradient from being computed through it.



### Code


[caffe2/operators/stop_gradient.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stop_gradient.cc)

---



## StoreAdd


Add a value to a remote counter. If the key is not set, the store initializes it to 0 and then performs the add operation. The operation returns the resulting counter value.



### Interface


---------- | ----------
*Arguments* | 
`blob_name` | key of the counter (required)
`add_value` | value that is added (optional, default: 1)
*Inputs* | 
`handler` | unique_ptr<StoreHandler>
*Outputs* | 
`value` | the current value of the counter


### Code


[caffe2/distributed/store_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/distributed/store_ops.cc)

---



## StoreGet


Get a blob from a store. The key is the output blob's name. The key can be overridden by specifying the 'blob_name' argument.



### Interface


---------- | ----------
*Arguments* | 
`blob_name` | alternative key for the blob (optional)
*Inputs* | 
`handler` | unique_ptr<StoreHandler>
*Outputs* | 
`data` | data blob


### Code


[caffe2/distributed/store_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/distributed/store_ops.cc)

---



## StoreSet


Set a blob in a store. The key is the input blob's name and the value is the data in that blob. The key can be overridden by specifying the 'blob_name' argument.



### Interface


---------- | ----------
*Arguments* | 
`blob_name` | alternative key for the blob (optional)
*Inputs* | 
`handler` | unique_ptr<StoreHandler>
`data` | data blob


### Code


[caffe2/distributed/store_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/distributed/store_ops.cc)

---



## StoreWait


Wait for the specified blob names to be set. The blob names can be passed either as an input blob with blob names or as an argument.



### Interface


---------- | ----------
*Arguments* | 
`blob_names` | names of the blobs to wait for (optional)
*Inputs* | 
`handler` | unique_ptr<StoreHandler>
`names` | names of the blobs to wait for (optional)


### Code


[caffe2/distributed/store_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/distributed/store_ops.cc)

---



## StringEndsWith


Performs the ends-with check on each string in the input tensor.
Returns tensor of boolean of the same dimension of input.



### Interface


---------- | ----------
*Arguments* | 
`suffix` | The suffix to check input strings against.
*Inputs* | 
`strings` | Tensor of std::string.
*Outputs* | 
`bools` | Tensor of bools of same shape as input.


### Code


[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)

---



## StringIndexCreate


Creates a dictionary that maps string keys to consecutive integers from 1 to max_elements. Zero is reserved for unknown keys.



### Interface


---------- | ----------
*Arguments* | 
`max_elements` | Max number of elements, including the zero entry.
*Outputs* | 
`handle` | Pointer to an Index instance.


### Code


[caffe2/operators/index_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/index_ops.cc)

---



## StringJoin


Takes a 1-D or a 2-D tensor as input and joins elements in each row with the provided delimiter. Output is a 1-D tensor of size equal to the first dimension of the input. Each element in the output tensor is a string of concatenated elements corresponding to each row in the input tensor. For 1-D input, each element is treated as a row.



### Interface


---------- | ----------
*Arguments* | 
`delimiter` | Delimiter for join (Default: ",").
`axis` | Axis for the join (either 0 or 1)
*Inputs* | 
`input` | 1-D or 2-D tensor
*Outputs* | 
`strings` | 1-D tensor of strings created by joining row elements from the input tensor.


### Code


[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)

---



## StringPrefix


Computes the element-wise string prefix of the string tensor.
Input strings that are shorter than prefix length will be returned unchanged.
NOTE: Prefix is computed on number of bytes, which may lead to wrong behavior and potentially invalid strings for variable-length encodings such as utf-8.



### Interface


---------- | ----------
*Arguments* | 
`length` | Maximum size of the prefix, in bytes.
*Inputs* | 
`strings` | Tensor of std::string.
*Outputs* | 
`prefixes` | Tensor of std::string containing prefixes for each input.


### Code


[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)

---



## StringStartsWith


Performs the starts-with check on each string in the input tensor.
Returns tensor of boolean of the same dimension of input.



### Interface


---------- | ----------
*Arguments* | 
`prefix` | The prefix to check input strings against.
*Inputs* | 
`strings` | Tensor of std::string.
*Outputs* | 
`bools` | Tensor of bools of same shape as input.


### Code


[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)

---



## StringSuffix


Computes the element-wise string suffix of the string tensor.
Input strings that are shorter than suffix length will be returned unchanged.
NOTE: Prefix is computed on number of bytes, which may lead to wrong behavior and potentially invalid strings for variable-length encodings such as utf-8.



### Interface


---------- | ----------
*Arguments* | 
`length` | Maximum size of the suffix, in bytes.
*Inputs* | 
`strings` | Tensor of std::string.
*Outputs* | 
`suffixes` | Tensor of std::string containing suffixes for each output.


### Code


[caffe2/operators/string_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/string_ops.cc)

---



## Sub


Performs element-wise binary subtraction (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and type as A


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## Sum


Element-wise sum of each of the input tensors. The first input tensor can be used in-place as the output tensor, in which case the sum will be done in place and results will be accumulated in input0. All inputs and outputs must have the same shape and data type.



### Interface


---------- | ----------
*Inputs* | 
`data_0` | First of the input tensors. Can be inplace.
*Outputs* | 
`sum` | Output tensor. Same dimension as inputs.


### Code


[caffe2/operators/elementwise_sum_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_sum_op.cc)

---



## SumElements

Sums the elements of the input tensor.


### Interface


---------- | ----------
*Arguments* | 
`average` | whether to average or not
*Inputs* | 
`X` | Tensor to sum up
*Outputs* | 
`sum` | Scalar sum


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_ops.cc)

---



## SumElementsGradient

No documentation yet.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_ops.cc)

---



## SumInt

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## SumReduceLike


SumReduceLike operator takes 2 tensors as input. It performs reduce sum to the first input so that the output looks like the second one.
It assumes that the first input has more dimensions than the second, and the dimensions of the second input is the contiguous subset of the dimensions of the first.
For example, the following tensor shapes are supported:   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 2, 5), shape(B) = (2), with axis=0
```

     


### Interface


---------- | ----------
*Arguments* | 
`axis` | If set, defines the starting dimension for reduction. Args `axis` and `axis_str` cannot be used simultaneously.
`axis_str` | If set, it could only be N or C or H or W. `order` arg should also be provided. It defines the reduction dimensions on NCHW or NHWC. Args `axis` and `axis_str` cannot be used simultaneously.
`order` | Either NHWC or HCWH
*Inputs* | 
`A` | First operand, should share the type with the second operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and type as B


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## SumSqrElements

Sums the squares elements of the input tensor.


### Interface


---------- | ----------
*Arguments* | 
`average` | whether to average or not
*Inputs* | 
`X` | Tensor to sum up
*Outputs* | 
`sum` | Scalar sum of squares


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/reduction_ops.cc)

---



## Summarize


Summarize computes four statistics of the input tensor (Tensor<float>)- min, max, mean and standard deviation. The output will be written to a 1-D tensor of size 4 if an output tensor is provided. Else, if the argument 'to_file' is greater than 0, the values are written to a log file in the root folder.



### Interface


---------- | ----------
*Arguments* | 
`to_file` | (int, default 0) flag to indicate if the summarized statistics have to be written to a log file.
*Inputs* | 
`data` | The input data as Tensor<float>.
*Outputs* | 
`output` | 1-D tensor (Tensor<float>) of size 4 containing min, max, mean and standard deviation


### Code


[caffe2/operators/summarize_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/summarize_op.cc)

---



## Swish


Swish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the swish function, y = x / (1 + exp(-x)), is applied to the tensor elementwise.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D output tensor


### Code


[caffe2/operators/swish_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/swish_op.cc)

---



## SwishGradient


SwishGradient takes X, Y and dY and uses this to update dX according to the chain rule and derivatives of the swish function.



### Code


[caffe2/operators/swish_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/swish_op.cc)

---



## TT


The TT-layer serves as a low-rank decomposition of a fully connected layer. The inputs are the same as to a fully connected layer, but the number of parameters are greatly reduced and forward computation time can be drastically reduced especially for layers with large weight matrices. The multiplication is computed as a product of the input vector with each of the cores that make up the TT layer. Given the input sizes (inp_sizes), output sizes(out_sizes), and the ranks of each of the cores (tt_ranks), the ith core will have size:   

```
    inp_sizes[i] * tt_ranks[i] * tt_ranks[i + 1] * out_sizes[i].

```

 The complexity of the computation is dictated by the sizes of inp_sizes, out_sizes, and tt_ranks, where there is the trade off between accuracy of the low-rank decomposition and the speed of the computation.



### Interface


---------- | ----------
*Arguments* | 
`inp_sizes` | (int[]) Input sizes of cores. Indicates the input size of the individual cores; the size of the input vector X must match the product of the inp_sizes array.
`out_sizes` | (int[]) Output sizes of cores. Indicates the output size of the individual cores; the size of the output vector Y must match the product of the out_sizes array.
`tt_ranks` | (int[]) Ranks of cores. Indicates the ranks of the individual cores; lower rank means larger compression, faster computation but reduce accuracy.
*Inputs* | 
`X` | Input tensor from previous layer with size (M x K), where M is the batch size and K is the input size.
`b` | 1D blob containing the bias vector
`cores` | 1D blob containing each individual cores with sizes specified above.
*Outputs* | 
`Y` | Output tensor from previous layer with size (M x N), where M is the batch size and N is the output size.


### Code


[caffe2/operators/tt_linear_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tt_linear_op.cc)

---



## TTLinearGradient

No documentation yet.


### Code


[caffe2/operators/tt_linear_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tt_linear_op.cc)

---



## Tanh


Calculates the hyperbolic tangent of the given input tensor element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.



### Interface


---------- | ----------
*Inputs* | 
`input` | 1-D input tensor
*Outputs* | 
`output` | The hyperbolic tangent values of the input tensor computed element-wise


### Code


[caffe2/operators/tanh_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tanh_op.cc)

---



## TanhGradient

No documentation yet.


### Code


[caffe2/operators/tanh_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tanh_op.cc)

---



## TensorProtosDBInput


TensorProtosDBInput is a simple input operator that basically reads things from a db where each key-value pair stores an index as key, and a TensorProtos object as value. These TensorProtos objects should have the same size, and they will be grouped into batches of the given size. The DB Reader is provided as input to the operator and it returns as many output tensors as the size of the TensorProtos object. Each output will simply be a tensor containing a batch of data with size specified by the 'batch_size' argument containing data from the corresponding index in the TensorProtos objects in the DB.



### Interface


---------- | ----------
*Arguments* | 
`batch_size` | (int, default 0) the number of samples in a batch. The default value of 0 means that the operator will attempt to insert the entire data in a single output blob.
*Inputs* | 
`data` | A pre-initialized DB reader. Typically, this is obtained by calling CreateDB operator with a db_name and a db_type. The resulting output blob is a DB Reader tensor
*Outputs* | 
`output` | The output tensor in which the batches of data are returned. The number of output tensors is equal to the size of (number of TensorProto's in) the TensorProtos objects stored in the DB as values. Each output tensor will be of size specified by the 'batch_size' argument of the operator


### Code


[caffe2/operators/tensor_protos_db_input.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tensor_protos_db_input.cc)

---



## TensorVectorSize

Get the size of the input vector


### Interface


---------- | ----------
*Inputs* | 
`tensor vector` | std::unique_ptr<std::vector<Tensor> >
*Outputs* | 
`size` | int32_t size


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## TextFileReaderRead

Read a batch of rows from the given text file reader instance. Expects the number of fields to be equal to the number of outputs. Each output is a 1D tensor containing the values for the given field for each row. When end of file is reached, returns empty tensors.


### Interface


---------- | ----------
*Arguments* | 
`batch_size` | Maximum number of rows to read.
*Inputs* | 
`handler` | Pointer to an existing TextFileReaderInstance.


### Code


[caffe2/operators/text_file_reader.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/text_file_reader.cc)

---



## ThresholdedRelu


ThresholdedRelu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, y = x for x > alpha, y = 0 otherwise, is applied to the tensor elementwise.



### Interface


---------- | ----------
*Arguments* | 
`alpha` | (float) defaults to 1.0.
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/thresholded_relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/thresholded_relu_op.cc)

---



## ThresholdedReluGradient


ThresholdedReluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.



### Code


[caffe2/operators/thresholded_relu_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/thresholded_relu_op.cc)

---



## Tile


Constructs a tensor by tiling a given tensor along a specified axis.
 This operation creates a new tensor by replicating the input tensor 'tiles' times along dimension 'axis'. The output tensor's 'axis'th dimension has input.dims(axis) * tiles elements, and the values of input are replicated 'tiles' times along the 'axis'th dimension.
For example, tiling [[a b c d]] by tile=2, axis=0 produces [[a b c d], [a b c d]].



### Interface


---------- | ----------
*Arguments* | 
`tiles` | Number of replicas
`axis` | Axis to replicate along
*Inputs* | 
`data` | The input tensor.
`tiles` | (optional) Number of replicas (overrides argument)
`axis` | (optional) Axis to replicate along (overrides argument)
*Outputs* | 
`tiled_data` | Tensor that will contain input replicated along the given axis.


### Code


[caffe2/operators/tile_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tile_op.cc)

---



## TileGradient

No documentation yet.


### Code


[caffe2/operators/tile_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/tile_op.cc)

---



## TimerBegin


Start a wallclock timer, returning a pointer to it.
The timer is stopped by calling TimerEnd


### Interface


---------- | ----------
*Arguments* | 
`counter_name` | Name of the timer. If not provided, use output name.
*Outputs* | 
`timer` | Pointer to timer, to be passed to TimerEnd.


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## TimerEnd

Stop a timer started with TimerBegin, publishing a CAFFE_EVENT


### Interface


---------- | ----------
*Inputs* | 
`timer` | Pointer to timer, obtained from TimerBegin.


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## TimerGet

Queries the current time of a timer in nanos


### Interface


---------- | ----------
*Inputs* | 
`timer` | Pointer to timer, obtained from TimerBegin.
*Outputs* | 
`nanos` | nanoseconds in int64


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## TimerGetAndEnd

Queries the current time of a timer in nanos, stops the timer             publishing a CAFFE_EVENT


### Interface


---------- | ----------
*Inputs* | 
`timer` | Pointer to timer, obtained from TimerBegin.
*Outputs* | 
`nanos` | nanoseconds in int64


### Code


[caffe2/operators/stats_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/stats_ops.cc)

---



## TopK


Retrieve the top-K elements for the last dimension. Given an input tensor of shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs: -Value tensor of shape [a_1, a_2, ..., a_n, k] which contains the values of  the top k elements along the last dimension -Index tensor of shape [a_1, a_2, ..., a_n, k] which contains the indices  of the top k elements (original indices from the input tensor).
 Given two equivalent values, this operator uses the indices along the last dim- ension as a tiebreaker. That is, the element with the lower index will appear first.
    


### Interface


---------- | ----------
*Arguments* | 
`k` | Number of top elements to retrieve
*Inputs* | 
`X` | Tensor of shape [a_1, a_2, ..., a_n, r]
*Outputs* | 
`Values` | Tensor of shape [a_1, a_2, ..., a_n, k] containing top K values from the input tensor
`Indices` | Tensor of shape [a_1, a_2, ..., a_n, k] containing the corresponding input tensor indices for the top K values.
`Flatten indices` | Tensor of shape [a_1 * a_2 * ... * a_n * k] containing the indices into the flatten input


### Code


[caffe2/operators/top_k.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/top_k.cc)

---



## TopKGradient

No documentation yet.


### Code


[caffe2/operators/top_k.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/top_k.cc)

---



## Transpose


Transpose the input tensor similar to numpy.transpose. For example, when axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape will be (2, 1, 3).



### Interface


---------- | ----------
*Arguments* | 
`axes` | A list of integers. By default, reverse the dimensions, otherwise permute the axes according to the values given.
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`transposed` | Transposed output.


### Code


[caffe2/operators/transpose_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/transpose_op.cc)

---



## TrimDataset


Trim the given dataset inplace, given the dataset blobs and the field specs.
Trimming happens such that the dataset will contain the largest possible number of records that is a multiple of the 'multiple_of' argument.



### Interface


---------- | ----------
*Arguments* | 
`fields` | List of strings representing the string names in the formatspecified in the doc for CreateTreeCursor.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## UnPackRecords


Given a packed dataset (packed by the PackRecordsOp) and the  `fields`  argument describing the datasets schema returns the original dataset format. Number of returned tensors is equal to the number of fields in the  `fields`  argument.
 The first input is the packed tensor to be unpacked. Optionally, you can provide prototype tensors to give the expected shapes of the output tensors. This is helpful when you expected to unpack empty tensor, e.g., output of a sampling process.



### Interface


---------- | ----------
*Arguments* | 
`fields` | List of strings representing the string names in the formatspecified in the doc for CreateTreeCursor.
*Inputs* | 
`packed_tensor` | The tensor to be unpacked


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/dataset_ops.cc)

---



## UniformFill


Fill the output tensor with FLOAT samples from uniform distribution [min, max].
 The range can be defined either by arguments or input blobs. If the range is given by input blobs, you also need to give the shape as input. When the range is given as arguments, this operator enforces min <= max. When the range is given as inputs, the constraint is not enforced. When MAX < MIN, the first dimension of the output is set to 0. This behavior is allowed so that dynamically sampling indices into a dynamically sized tensor is possible.
 The shape of the output can be given as argument or input.



### Interface


---------- | ----------
*Arguments* | 
`min` | minimum value, inclusive
`max` | maximum value, inclusive
`shape` | shape of the output, do not set when input_as_shape=1
`input_as_shape` | set to 1 to use the first input as shape. First input must be in CPU context.
*Inputs* | 
`SHAPE` | 1-D tensor of the shape of the output, must be used with input_as_shape
`MIN` | scalar blob of mininum value
`MAX` | scalar blob of maximum value
*Outputs* | 
`OUTPUT` | output tensor


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## UniformIntFill


Like  `UniformFill`  but fill with INT32.



### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## Unique


Deduplicates input indices vector and optionally produces reverse remapping.
There's no guarantees on the ordering of the output indices.



### Interface


---------- | ----------
*Inputs* | 
`indices` | 1D tensor of int32 or int64 indices.
*Outputs* | 
`unique_indices` | 1D tensor of deduped entries.
`remapping` | (optional) mapping from `indices` to `unique_indices`. This has the same shape as `indices`. Its elements are the indices into `unique_indices` such that `Gather(['unique_indices', 'remapping'])` yields `indices`.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## UniqueUniformFill


Fill the output tensor with uniform samples between min and max (inclusive).
If the second input is given, its elements will be excluded from uniform sampling. Using the second input will require you to provide shape via the first input.



### Interface


---------- | ----------
*Arguments* | 
`min` | Minimum value, inclusive
`max` | Maximum value, inclusive
`dtype` | The data type for the elements of the output tensor.Strictly must be one of the types from DataType enum in TensorProto.This only supports INT32 and INT64 now. If not set, assume INT32
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob. Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape. First input must be in CPU context.
*Inputs* | 
`input` | Input tensor to provide shape information
`avoid` | (optional) Avoid elements in this tensor. Elements must be unique.
*Outputs* | 
`output` | Output tensor of unique uniform samples


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## UnpackRNNSequence


This is the reverse operator for PackRNNSequence. It maps the packed values back to sequence values based on the length blob. Each number from length blob represents the corresponding values that has been grouped. The dimension for each pack is the same as the maximum number from the length blob (padding with zero was implemented for smaller length value). The overall output dimension is: M * D, where M is the sum of lengths, and D is the dimension of each feature value. The following example shows the input and output of this operator:   Given:  

```
  values = [
    [v1, v3, v6, v7],
    [v2, v4, 0,  v8],
    [0,  v5, 0,  0 ],
  ]
  lengths = [2, 3, 1, 2]


```

 Output:  

```
  output = [v1, v2, v3, v4, v5, v6, v7, v8];


```

 One application for this operator is the transfer data from the format of RNN back to sequence values. Note that the gradient operator of UnpackRNNSequence is PackRNNSequence.



### Interface


---------- | ----------
*Inputs* | 
`values` | Data tensor, contains the packed features
`lengths` | lengths with each number representing the pack size.
*Outputs* | 
`output` | Output tensor before packing


### Code


[caffe2/operators/pack_rnn_sequence_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pack_rnn_sequence_op.cc)

---



## UnpackSegments

Map N+1 dim tensor to N dim based on length blob


### Interface


---------- | ----------
*Inputs* | 
`lengths` | 1-d int/long tensor contains the length in each of the input.
`tensor` | N+1 dim Tensor.
*Outputs* | 
`packed_tensor` | N dim Tensor


### Code


[caffe2/operators/pack_segments.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/pack_segments.cc)

---



## UnsafeCoalesce


Coalesce the N inputs into N outputs and a single coalesced output blob.
 This allows operations that operate over multiple small kernels (e.g.
biases in a deep CNN) to be coalesced into a single larger operation, amortizing the kernel launch overhead, synchronization costs for distributed computation, etc.
 The operator:  - computes the total size of the coalesced blob by summing the input sizes - allocates the coalesced output blob as the total size - copies the input vectors into the coalesced blob, at the correct offset.
- aliases each Output(i) to- point into the coalesced blob, at the corresponding offset for Input(i).
 This is 'unsafe' as the output vectors are aliased, so use with caution.
 


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## UnsortedSegmentMean


Applies 'Mean' to each segment of input tensor. Segments ids can appear in arbitrary order (unlike in SortedSegmentMean).
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Mean computes the element-wise mean of the input slices. Operation doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`num_segments` | Optional int argument specifying the number of output segments and thus the first dimension of the output
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`SEGMENT_IDS` | Integer vector with the same length as the first dimension of DATA that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of equal to the number of segments.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentSum


Applies 'Sum' to each segment of input tensor. Segments ids can appear in arbitrary order (unlike in SortedSegmentSum).
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`num_segments` | Optional int argument specifying the number of output segments and thus the first dimension of the output
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`SEGMENT_IDS` | Integer vector with the same length as the first dimension of DATA that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of equal to the number of segments.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentWeightedSum


Applies 'WeightedSum' to each segment of input tensor. Segments ids can appear in arbitrary order (unlike in SortedSegmentWeightedSum).
 SEGMENT_IDS is a vector that maps each of the first dimension slices of the DATA to a particular group (segment). Values belonging to the same segment are aggregated together.
 If  `num_segments`  argument is passed it would be used as a first dimension for the output. Otherwise, it'd be dynamically calculated from as the max value of SEGMENT_IDS plus one. Other output dimensions are inherited from the input tensor.
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  


### Interface


---------- | ----------
*Arguments* | 
`num_segments` | Optional int argument specifying the number of output segments and thus the first dimension of the output
`grad_on_weights` | Produce also gradient for `weights`. For now it's only supported in `Lengths`-based operators
*Inputs* | 
`DATA` | Input tensor for the summation
`SCALARS` | Scalar multipliers for the input slices. Must be a vector with the length matching the number of slices
`SEGMENT_IDS` | Integer vector with the same length as the first dimension of DATA that maps each slice of DATA to one of the segments
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of equal to the number of segments.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## VariableLengthSequencePadding


Super special-case operator. Used to pad a tensor to mimic pytorch's pad_packed_sequence.
 Given an input tensor INPUT of size NxBxM and an input tensor LENS of size B, where  N = maximum sequence length B = batch size M = hidden size  set each element of INPUT to zero if it is is past the end of the corresponding sequence (i.e. if LENS[j] > i for an index (i,j,k)).
 


### Code


[caffe2/operators/variable_length_sequence_padding.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/variable_length_sequence_padding.cc)

---



## WallClockTime

Time since epoch in nanoseconds.


### Interface


---------- | ----------
*Outputs* | 
`time` | The time in nanoseconds.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## WeightedMultiSampling


The operator performs sampling based on the input sampling weights.
All weights are cummulative probability thus sorted. The output is a 1-D tensor (Tensor<int>). If two inputs are given, the second input is used to provide shape of the output sample tensor. Otherwise, we use argument  `num_samples`  to determine the number of samples to generate.



### Interface


---------- | ----------
*Arguments* | 
`num_samples` | number of samples to sample from the input data
*Inputs* | 
`sampling_cdf` | An optional 1-D Tensor<float>.Input cumulative sampling probability (such as [0.2, 0.5, 0.8, 1.5]). All weights must be non-negative numbers. Note that the last value of CDF is not necessary 1. If the last value is not 1, all values in sampling_cdf will be scaled by this number.
`shape_tensor (optional)` | Tensor whose shape will be applied to output.
*Outputs* | 
`sampled_indexes` | The output tensor contains indices sampled from distribution givenby the weight vector in the input tensorThe output is a 1-D Tensor<int> of size determined by argument`num_samples` or the second input tensor.


### Code


[caffe2/operators/weighted_multi_sampling_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/weighted_multi_sampling_op.cc)

---



## WeightedSample


The operator performs sampling based on the input sampling weights for each batch. All weights must be non-negative numbers.
The input is a 2-D tensor (Tensor<float>) of size (batch_size x weights_dim).
For each batch, an index is randomly sampled from the distribution given by the weights of the corresponding batch.
The output is a 1-D tensor (Tensor<int>) of size (batch_size x 1) and contains the index(es) of the sampled output.



### Interface


---------- | ----------
*Inputs* | 
`sampling_weights` | A 2-D Tensor<float> of size (batch_size x weights_dim).All weights must be non-negative numbers.
`sampling_values` | An optional 2-D Tensor<float> of size (batch_size x weights_dim).Its values correspond to the sampling weights.
*Outputs* | 
`sampled_indexes` | The output tensor contains index(es) sampled from distribution givenby the weight vector(s) in the input tensorThe output is a 1-D Tensor<int> of size (batch_size x 1)
`sampled_values` | The output tensor contains value(s) selected by the sampled index(es)It is a 1-D Tensor<float> of size (batch_size x 1)


### Code


[caffe2/operators/weighted_sample_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/weighted_sample_op.cc)

---



## WeightedSampleDequeueBlobs


Dequeue the blobs from multiple queues. When one of queues is closed and empty, the output status will be set to true which can be used as exit criteria for execution step.
The 1st input is the queue and the last output is the status. The rest are data blobs.



### Interface


---------- | ----------
*Arguments* | 
`weights` | Weights for sampling from multiple queues
`table_idx_blob` | The index of the blob (among the output blob list) that will be used to store the index of the table chosen to read the current batch.


### Code


[caffe2/queue/queue_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/queue/queue_ops.cc)

---



## WeightedSigmoidCrossEntropyWithLogits


Given three matrices: logits, targets, weights, all of the same shape, (batch_size, num_classes), computes the weighted sigmoid cross entropy between logits and targets. Specifically, at each position r,c, this computes weights[r, c] * crossentropy(sigmoid(logits[r, c]), targets[r, c]), and then averages over each row.
Returns a tensor of shape (batch_size,) of losses for each example.



### Interface


---------- | ----------
*Inputs* | 
`logits` | matrix of logits for each example and class.
`targets` | matrix of targets, same shape as logits.
`weights` | matrix of weights, same shape as logits.
*Outputs* | 
`xentropy` | Vector with the total xentropy for each example.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## WeightedSigmoidCrossEntropyWithLogitsGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## WeightedSum


Element-wise weighted sum of several data, weight tensor pairs.
Input should be in the form X_0, weight_0, X_1, weight_1, ... where X_i all have the same shape, and weight_i are size 1 tensors that specifies the weight of each vector. Note that if one wants to do in-place computation, it could only be done with X_0 also as the output, but not other X_i.



### Interface


---------- | ----------
*Inputs* | 
`weight_0` | Weight of the first input in the sum.
*Outputs* | 
`output` | Result containing weighted elem-wise sum of inputs.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## WeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)

---



## Where


Operator Where takes three input data (Tensor<bool>, Tensor<T>, Tensor<T>) and produces one output data (Tensor<T>) where z = c ? x : y is applied elementwise.



### Interface


---------- | ----------
*Inputs* | 
`C` | input tensor containing booleans
`X` | input tensor
`Y` | input tensor
*Outputs* | 
`Z` | output tensor


### Code


[caffe2/operators/elementwise_logical_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.cc)

---



## While


'While' control operator, first input is a scalar boolean blob that stores loop's condition value. Accepts 'loop_net' (required) and 'cond_net' (optional) arguments for loop's body and condition subnets respectively. If condition subnet is specified, it is executed before the first and after each iteration. Subnets are executed in the same workspace as 'While'.
    


### Interface


---------- | ----------
*Arguments* | 
`loop_net` | Net executed on each iteration
`cond_net` | Net to (re)compute condition value
*Inputs* | 
`condition` | Scalar boolean condition


### Code


[caffe2/operators/while_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/while_op.cc)

---



## XavierFill

No documentation yet.


### Code


[caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)

---



## Xor


Performs element-wise logical operation  `xor`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

```

 Argument  `broadcast=1`  needs to be passed to enable broadcasting.



### Interface


---------- | ----------
*Arguments* | 
`broadcast` | Pass 1 to enable broadcasting
`axis` | If set, defines the broadcast dimensions. See doc for details.
*Inputs* | 
`A` | First operand.
`B` | Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | Result, has same dimensions and A and type `bool`


### Code


[caffe2/operators/elementwise_op_schema.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_op_schema.cc)

---



## YellowFin


 Computes the YellowFin update ( [https://arxiv.org/abs/1706.03471)](https://arxiv.org/abs/1706.03471))  and performs momentum SGD optimization step. lr and mu are not being shared between parameters. curv_win, g_avg, g2_avg and scalars_memory are just auxiliary memory for computing moving averages (see the publication). Takes arguments beta: coefficient for moving averages, curv_win_width: timeframe when average squared gradient is being stored, epsilon: for numerical purposes, nesterov and zero_debias for debias of moving average.
 


### Interface


---------- | ----------
*Arguments* | 
`beta` | Default 0.999
`curv_win_width` | Default 20
`epsilon` | Default 1e-6
`nesterov` | Default false
`zero_debias` | Default true
*Inputs* | 
`param` | Parameters to be updated
`moment` | Momentum
`lr` | Learning rate
`mu` | Momentum coefficient
`curv_win` | Memory for latest curvature ranges
`g_avg` | Moving average of gradient
`g2_avg` | Moving average of squared gradient
`scalars_memory` | Memory for stateful scalars
`grad` | Gradient computed
`iter` | Iteration number
*Outputs* | 
`output_param` | Parameters to be updated
`output_moment` | Momentum
`output_lr` | Output learning rate
`output_mu` | Output momentum coefficient
`output_curv_win` | Output memory for latest curvature ranges
`output_g_avg` | Output moving average of gradient
`output_g2_avg` | Output moving average of squared gradient
`output_scalars_memory` | Output memory for stateful scalars


### Code


[caffe2/sgd/yellowfin_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/sgd/yellowfin_op.cc)

---



## ZeroGradient


ZeroGradient operators doesn't produce any output blobs. One can use this operator to produce 0 gradient for the input blob.



### Code


[caffe2/operators/zero_gradient_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/zero_gradient_op.cc)

---



## rnn_internal_accumulate_gradient_input


Internal RNN operator.



### Code


[caffe2/operators/recurrent_network_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/recurrent_network_op.cc)

---



## rnn_internal_apply_link


Internal RNN operator.



### Code


[caffe2/operators/recurrent_network_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/recurrent_network_op.cc)

---

