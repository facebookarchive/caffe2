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
`labels` | 2-D tensor (Tensor<float>) of size (num_samples) containing true labels for each sample
*Outputs* | 
`AP` | 1-D tensor (Tensor<float>) of size num_classes containing average precision for each class


### Code


[caffe2/operators/apmeter_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/apmeter_op.cc)

---



## Abs


Calculates the absolute value of the given input tensor, element-wise.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Abs",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.randn(5).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [ 0.3005476  

```
  1.551666   -1.3591481   0.39191285 -0.21866608]
```

 Y: [0.3005476 

```
  1.551666   1.3591481  0.39191285 0.21866608]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor<float>)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Absolute value of input element-wise.


### Code


[caffe2/operators/abs_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc)

---



## AbsGradient

No documentation yet.


### Code


[caffe2/operators/abs_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/abs_op.cc)

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


[caffe2/operators/accumulate_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/accumulate_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## Accuracy


Accuracy takes two inputs- predictions and labels, and returns a float accuracy value for the batch. Predictions are expected in the form of 2-D tensor containing a batch of scores for various classes, and labels are expected in the  form of 1-D tensor containing true label indices of samples in the batch. If the score for the label index in the predictions is the highest among all classes, it is considered a correct prediction.



### Interface


---------- | ----------
*Arguments* | 
`top_k` | Count as correct by comparing the true label to the top k scoring classes (default 1: only compare to the top scoring class i.e. argmax)
*Inputs* | 
`predictions` | 2-D tensor (Tensor<float>) of size (num_batches x num_classes) containing scores
`labels` | 1-D tensor (Tensor<float>) of size (num_batches) having the indices of true labels
*Outputs* | 
`accuracy` | 1-D tensor (Tensor<float>) of size 1 containing accuracy


### Code


[caffe2/operators/accuracy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/accuracy_op.cc)

---



## Acos


Calculates the arccosine of the given input tensor, element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The arccosine of the input tensor computed element-wise


### Code


[caffe2/operators/acos_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/acos_op.cc)

---



## AcosGradient

No documentation yet.


### Code


[caffe2/operators/acos_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/acos_op.cc)

---



## Add


Performs element-wise binary addition (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "Add",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([[1,2],[3,4]])) workspace.FeedBlob("B", np.array([[5,6],[7,8]])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [[1 2]  [3 4]] B: [[5 6]  [7 8]] C: [[ 6 

```
  8]
```

  [10 12]]   ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting
`axis` | *(type: int; default: -1)* Axis to concatenate on.
*Inputs* | 
`A` | *(type: Tensor`<float>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size as A.
*Outputs* | 
`C` | *(type: Tensor`<float>`)* Output tensor with same dimensions and type as A.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## AddGradient

No documentation yet.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## AddPadding


Given a partitioned tensor $T<N, D_1, ..., D_n>$, where the partitions are defined as ranges on its outer-most (slowest varying) dimension $N$, return a tensor $T<(N + 2 * padding\_width), D_1, ..., D_n>$ with paddings added to the start and end of each range.
 Optionally, different paddings can be provided for beginning and end.
Paddings provided must be a tensor $T<D_1, ..., D_n>$. If no padding is provided, add zero padding. If no lengths vector is provided, add padding only once, at the start and end of data.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "AddPadding",
    ["X", "lengths"],
    ["Y", "lengths_out"],
    padding_width=1

```

 )  workspace.FeedBlob("X", (np.random.rand(3,2,2).astype(np.float32))) workspace.FeedBlob("lengths", np.array([3]).astype(np.int32))  print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y")) print("lengths_out:", workspace.FetchBlob("lengths_out"))  ```    **Result**    ```  X: [[[0.2531572 

```
  0.4588472 ]
  [0.45140603 0.61161053]]

```

  [[0.92500854 0.8045306 ]  

```
  [0.03356671 0.30233648]]

```

  [[0.4660227 

```
  0.6287745 ]
  [0.79372746 0.08609265]]]
```

 Y: [[[0.  

```
        0.        ]
  [0.         0.        ]]

```

  [[0.2531572 

```
  0.4588472 ]
  [0.45140603 0.61161053]]

```

  [[0.92500854 0.8045306 ]  

```
  [0.03356671 0.30233648]]

```

  [[0.4660227 

```
  0.6287745 ]
  [0.79372746 0.08609265]]

```

  [[0.  

```
        0.        ]
  [0.         0.        ]]]
```

 lengths_out: [5]  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`padding_width` | *(type: int)* Number of copies of padding to add around each range.
`end_padding_width` | *(type: int)* [OPTIONAL] Specifies a different end-padding width. If this is not set, will use same as `padding_width`.
*Inputs* | 
`data_in` | *(type: Tensor)* Input data ($T<N, D_1, ..., D_n>$).
`lengths` | *(type: Tensor`<int>`)* Number of elements in each range. sum(lengths) = N.
`start_padding` | *(type: Tensor`<int>`)* [OPTIONAL] Padding data for range start ($T<D_1, ..., D_n>$).
`end_padding` | *(type: Tensor`<int>`)* [OPTIONAL] Padding for range end. If not provided, `start_padding` is used ($T<D_1, ..., D_n>$).
*Outputs* | 
`data_out` | *(type: Tensor)* Padded data tensor ($T<N + 2*padding_width, D_1, ..., D_n>$).
`lengths_out` | *(type: Tensor`<int>`)* [OPTIONAL] Lengths for each padded range.


### Code


[caffe2/operators/sequence_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc)

---



## AffineChannel


Applies a separate affine transformation to each channel of the input. Useful for replacing spatial batch norm with its equivalent fixed transformation.



### Interface


---------- | ----------
*Inputs* | 
`X` | Feature map input with order NCHW or NHWC.
`scale` | 1D input of shape (C); the c-th element is the scale factor of the affine transformation for the c-th channel of the input.
`bias` | 1D input of shape (C); the c-th element is the bias of the affine transformation for the c-th channel of the input.
*Outputs* | 
`Y` | Output with the same order of Input.


### Code


[caffe2/operators/affine_channel_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/affine_channel_op.cc)

---



## AffineChannelGradient

No documentation yet.


### Code


[caffe2/operators/affine_channel_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/affine_channel_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

---



## And


Performs element-wise logical operation  **and**  (with limited broadcast support).
Both input operands should be of type  `bool` .
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "And",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5)) workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5)) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A:  [[ True False False]  [False 

```
  True False]
```

  [False False 

```
  True]]
```

 B:  [[ True False 

```
  True]
```

  [False False False]  [False False False]] C:  [[ True False False]  [False False False]  [False False False]]   ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor of booleans. Has same dimensions as input `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## Append


Append input  `B`  to the end of input  `A` .
 - It is required that this operation run in-place, meaning that the input  `A`  blob must match the output blob.
- All except the outer-most dimension must be the same between  `A`  and  `B` .
- Input  `A`  may have to be re-allocated in order for accommodate to the new size. Currently, an exponential growth ratio is used in order to ensure amortized constant time complexity.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Append",
    ["A", "B"],
    ["A"],
```

 )  workspace.FeedBlob("A", np.random.randint(10, size=(1,3,3))) workspace.FeedBlob("B", np.random.randint(10, size=(2,3,3))) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("A:", workspace.FetchBlob("A"))   ```    **Result**    ```   A: [[[3 8 7]  

```
  [1 6 6]
  [5 0 6]]]
```

 B: [[[4 3 1]  

```
  [7 9 6]
  [9 4 5]]

```

  [[7 7 4]  

```
  [9 8 7]
  [1 6 6]]]
```

 A: [[[3 8 7]  

```
  [1 6 6]
  [5 0 6]]

```

  [[4 3 1]  

```
  [7 9 6]
  [9 4 5]]

```

  [[7 7 4]  

```
  [9 8 7]
  [1 6 6]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`A` | (*Tensor*): base input tensor of shape $(N, d_1, d_2, ..., d_n)$
`B` | (*Tensor*): second input tensor of shape $(M, d_1, d_2, ..., d_n)$ to be appended to the base
*Outputs* | 
`A` | (*Tensor*): output tensor of shape $(N+M, d_1, d_2, ..., d_n)$


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## ArgMax


Retrieve the argmax of an axis dimension specified by the  `axis`  argument. Given an input tensor and two arguments ( `axis`  and  `keepdims` ), returns a tensor containing the indices of the largest element along the given axis. If the  `keepdims`  arg is  *True*  (default), the shape of the output tensor matches the input tensor except the  `axis`  dimension equals 1. Else, the  `axis`  dimension of the output tensor is removed.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ArgMax",
    ["X"],
    ["Indices"],
    axis=2,
    keepdims=False
```

 )  workspace.FeedBlob("X", (np.random.randint(10, size=(3,3,3))).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Indices:", workspace.FetchBlob("Indices"))   ```    **Result**    ```  X: [[[4. 9. 6.]  

```
  [6. 6. 1.]
  [9. 5. 4.]]

```

  [[6. 7. 4.]  

```
  [7. 9. 1.]
  [3. 2. 8.]]

```

  [[3. 4. 6.]  

```
  [5. 2. 7.]
  [1. 5. 7.]]]
```

 Indices: [[1 0 0]  [1 1 2]  [2 2 2]]   ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`axis` | *(type: int; default: -1)* The axis to get argmax.
`keepdims` | *(type: bool; default: True)* If True (default), the output tensor shape will match the input tensor shape except the `axis` dimension equals 1. Else, the `axis` dimension of the output tensor is removed.
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Indices` | *(type: Tensor`<float>`)* Tensor of indices for the largest values.


### Code


[caffe2/operators/arg_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc)

---



## ArgMin


Retrieve the argmin of an axis dimension specified by the  `axis`  argument. Given an input tensor and two arguments ( `axis`  and  `keepdims` ), returns a tensor containing the indices of the smallest element along the given axis. If the  `keepdims`  arg is  *True*  (default), the shape of the output tensor matches the input tensor except the  `axis`  dimension equals 1. Else, the  `axis`  dimension of the output tensor is removed.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ArgMin",
    ["X"],
    ["Indices"],
    axis=1
```

 )  workspace.FeedBlob("X", (np.random.randint(10, size=(5,5))).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Indices:", workspace.FetchBlob("Indices"))   ```    **Result**    ```   X: [[9. 4. 6. 4. 1.]  

```
  [5. 9. 8. 3. 4.]
  [6. 1. 0. 2. 9.]
  [7. 8. 2. 4. 9.]
  [3. 9. 4. 9. 4.]]
```

 Indices: [[4]  

```
  [3]
  [2]
  [2]
  [0]]

```

 ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`axis` | *(type: int; default: -1)* The axis to get argmin.
`keepdims` | *(type: bool; default: True)* If True (default), the output tensor shape will match the input tensor shape except the `axis` dimension equals 1. Else, the `axis` dimension of the output tensor is removed.
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Indices` | *(type: Tensor`<float>`)* Tensor of indices for the smallest values.


### Code


[caffe2/operators/arg_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/arg_ops.cc)

---



## Asin


Calculates the arcsine of the given input tensor, element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The arcsine of the input tensor computed element-wise


### Code


[caffe2/operators/asin_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/asin_op.cc)

---



## AsinGradient

No documentation yet.


### Code


[caffe2/operators/asin_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/asin_op.cc)

---



## Assert


Takes in a tensor of type  *bool* ,  *int* ,  *long* , or  *long long*  and checks if all values are True when coerced into a boolean. In other words, for non-bool types this asserts that all values in the tensor are non-zero. If a value is False after coerced into a boolean, the operator throws an error. Else, if all values are True, nothing is returned. For tracability, a custom error message can be set using the  `error_msg`  arguement.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/assert_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/assert_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Assert",
    ["A"],
    [],
    error_msg="Failed assertion from Assert operator"
```

 )  workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.int32)) print("A:", workspace.FetchBlob("A")) try:  

```
    workspace.RunOperatorOnce(op)
```

 except RuntimeError:  

```
    print("Assertion Failed!")
```

 else:  

```
    print("Assertion Passed!")

```

 ```    **Result**   ```  A: [[7 5 6]  [1 2 4]  [5 3 7]] Assertion Passed!  ```  </details>  	


### Interface


---------- | ----------
*Arguments* | 
`error_msg` | (*string*): custom error message to be thrown when the input does not pass assertion
*Inputs* | 
`X` | (*Tensor*): input tensor


### Code


[caffe2/operators/assert_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/assert_op.cc)

---



## Atan


Calculates the arctangent of the given input tensor, element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The arctangent of the input tensor computed element-wise


### Code


[caffe2/operators/atan_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/atan_op.cc)

---



## AtanGradient

No documentation yet.


### Code


[caffe2/operators/atan_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/atan_op.cc)

---



## AtomicAppend

No documentation yet.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/atomic_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/atomic_ops.cc)

---



## AveragePool

AveragePool  consumes an input blob and applies average pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists of taking the average value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "AveragePool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-0.2883434  

```
  0.43498734  0.05417408  1.912558    0.09390241
    -0.33173105]
```

   

```
  [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
```

   

```
    0.9989975 ]
```

   

```
  [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
```

   

```
    0.33195344]
```

   

```
  [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
```

   

```
    0.2586746 ]
```

   

```
  [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
    -0.97614396]
```

   

```
  [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
    -0.37085298]]]]

```

 Y:  [[[[0.7462672 

```
  0.83399826 0.2948959 ]
```

   

```
  [0.4843537  0.3506009  0.35500962]
```

   

```
  [0.9251013  0.19026303 0.13366827]]]]
```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool1D

AveragePool1D  consumes an input blob and applies average pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists of taking the average value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "AveragePool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-0.2883434  

```
  0.43498734  0.05417408  1.912558    0.09390241
    -0.33173105]
```

   

```
  [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
```

   

```
    0.9989975 ]
```

   

```
  [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
```

   

```
    0.33195344]
```

   

```
  [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
```

   

```
    0.2586746 ]
```

   

```
  [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
    -0.97614396]
```

   

```
  [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
    -0.37085298]]]]

```

 Y:  [[[[0.7462672 

```
  0.83399826 0.2948959 ]
```

   

```
  [0.4843537  0.3506009  0.35500962]
```

   

```
  [0.9251013  0.19026303 0.13366827]]]]
```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool1DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragePool2D

AveragePool2D  consumes an input blob and applies average pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists of taking the average value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "AveragePool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-0.2883434  

```
  0.43498734  0.05417408  1.912558    0.09390241
    -0.33173105]
```

   

```
  [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
```

   

```
    0.9989975 ]
```

   

```
  [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
```

   

```
    0.33195344]
```

   

```
  [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
```

   

```
    0.2586746 ]
```

   

```
  [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
    -0.97614396]
```

   

```
  [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
    -0.37085298]]]]

```

 Y:  [[[[0.7462672 

```
  0.83399826 0.2948959 ]
```

   

```
  [0.4843537  0.3506009  0.35500962]
```

   

```
  [0.9251013  0.19026303 0.13366827]]]]
```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool2DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragePool3D

AveragePool3D  consumes an input blob and applies average pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists of taking the average value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "AveragePool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-0.2883434  

```
  0.43498734  0.05417408  1.912558    0.09390241
    -0.33173105]
```

   

```
  [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
```

   

```
    0.9989975 ]
```

   

```
  [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
```

   

```
    0.33195344]
```

   

```
  [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
```

   

```
    0.2586746 ]
```

   

```
  [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
    -0.97614396]
```

   

```
  [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
    -0.37085298]]]]

```

 Y:  [[[[0.7462672 

```
  0.83399826 0.2948959 ]
```

   

```
  [0.4843537  0.3506009  0.35500962]
```

   

```
  [0.9251013  0.19026303 0.13366827]]]]
```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## AveragePool3DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragePoolGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## AveragedLoss


The  *AveragedLoss*  op takes a single 1-D input tensor  *input*  and returns a single output float value  *output* . The output represents the average of the values in  *input* . This op is commonly used for averaging losses, hence the name, however it does not exclusively operate on losses.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "AveragedLoss",
    ["input"],
    ["output"],
```

 )  workspace.FeedBlob("input", np.array([8, 10, 12]).astype(np.float32)) print("input:\n", workspace.FetchBlob("input"))  workspace.RunOperatorOnce(op) print("output: \n", workspace.FetchBlob("output"))   ```    **Result**    ```   input:  [ 8. 10. 12.] output:  10.0   ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`input` | The input data as Tensor
*Outputs* | 
`output` | The output tensor of size 1 containing the averaged value.


### Code


[caffe2/operators/loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/loss_op.cc)

---



## AveragedLossGradient

No documentation yet.


### Code


[caffe2/operators/loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/loss_op.cc)

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
`rotated` | bool (default false). If true, then boxes (rois and deltas) include angle info to handle rotation. The format will be [ctr_x, ctr_y, width, height, angle (in degrees)].
`angle_bound_on` | bool (default true). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi].
`angle_bound_lo` | int (default -90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi].
`angle_bound_hi` | int (default 90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi].
`clip_angle_thresh` | float (default 1.0 degrees). For RRPN, clip almost horizontal boxes within this threshold of tolerance for backward compatibility. Set to negative value for no clipping.
*Inputs* | 
`rois` | Bounding box proposals in pixel coordinates, Size (M, 4), format [x1, y1, x2, y2], orSize (M, 5), format [batch_index, x1, y1, x2, y2]. If proposals from multiple images in a batch are present, they should be grouped sequentially and in incremental order.For rotated boxes, this would have an additional angle (in degrees) in the format [<optionaal_batch_id>, ctr_x, ctr_y, w, h, angle].
`deltas` | bounding box translations and scales,size (M, 4*K), format [dx, dy, dw, dh], K = # classes. For rotated boxes, size (M, 5*K, format [dx, dy, dw, dh, da].
`im_info` | Image dimensions, size (batch_size, 3), format [img_height, img_width, img_scale]
*Outputs* | 
`box_out` | Pixel coordinates of the transformed bounding boxes,Size (M, 4*K), format [x1, y1, x2, y2]. For rotated boxes, size (M, 5*K), format [ctr_x, ctr_y, w, h, angle].
`roi_batch_splits` | Tensor of shape (batch_size) with each element denoting the number of RoIs belonging to the corresponding image in batch


### Code


[caffe2/operators/bbox_transform_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/bbox_transform_op.cc)

---



## BRGNCHWCToPackedInt8BGRAStylizerDeprocess

No documentation yet.


### Code


[caffe2/operators/stylizer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stylizer_ops.cc)

---



## Barrier


Does a barrier operation among the nodes.



### Interface


---------- | ----------
*Inputs* | 
`comm_world` | The common world.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

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


[caffe2/operators/batch_box_cox_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_box_cox_op.cc)

---



## BatchBucketOneHot


Input is a matrix tensor. Its first dimension is the batch size. For each column, bucketize it based on the boundary values and then do one hot encoding. The  `lengths`  specifies the number of boundary values for each column. The final number of buckets is this number plus 1. This would also be the expanded feature size.  `boundaries`  specifies all the boundary values.
Note that each bucket is right-inclusive. That is, given boundary values [b1, b2, b3], the buckets are defined as (-int, b1], (b1, b2], (b2, b3], (b3, inf).
For example   

```
  data = [[2, 3], [4, 1], [2, 5]], lengths = [2, 3],
  If boundaries = [0.1, 2.5, 1, 3.1, 4.5], then
  output = [[0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]]

  If boundaries = [0.1, 2.5, 1, 1, 3.1], then
  output = [[0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1]]

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


[caffe2/operators/one_hot_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/one_hot_ops.cc)

---



## BatchBucketize


Bucketize the float_features into sparse features.
The float_features is a N * D tensor where N is the batch_size, and D is the feature_dim.
The indices is a 1D tensor containing the indices of the features that need to be bucketized.
The lengths is a 1D tensor that splits the following 'boundaries' argument.
The boundaries is a 1D tensor containing the border list for each feature.
 With in each batch,  `indices`  should not have duplicate number, and the number of elements in  `indices`  should be less than or euqal to  `D` .
Each element in  `lengths`  vector (lengths[ `i` ]) represents the number of boundaries in the sub border list.
The sum of all elements in  `lengths`  must be equal to the size of 

```
  `boundaries`.
```

 If lengths[0] = 2, the first sub border list is [0.5, 1.0], which separate the value to (-inf, 0.5], (0,5, 1.0], (1.0, inf). The bucketized feature will have three possible values (i.e. 0, 1, 2).
  For example, with input:   

```
  float_features = [[1.42, 2.07, 3.19, 0.55, 4.32],
                    [4.57, 2.30, 0.84, 4.48, 3.09],
                    [0.89, 0.26, 2.41, 0.47, 1.05],
                    [0.03, 2.97, 2.43, 4.36, 3.11],
                    [2.74, 5.77, 0.90, 2.63, 0.38]]
  indices = [0, 1, 4]
  lengths = [2, 3, 1]
  boundaries =  [0.5, 1.0, 1.5, 2.5, 3.5, 2.5]

```

 The output is:   

```
  output =[[2, 1, 1],
```

   

```
          [2, 1, 1],
```

   

```
          [1, 0, 0],
```

   

```
          [0, 2, 1],
```

   

```
          [2, 3, 0]]

```

 after running this operator.



### Interface


---------- | ----------
*Inputs* | 
`float_features` | 2-D dense tensor, the second dimension must be greater or equal to the indices dimension
`indices` | Flatten tensor, containing the indices of `float_features` to be bucketized. The datatype must be int32.
`lengths` | Flatten tensor, the size must be equal to that of `indices`. The datatype must be int32.
`boundaries` | Flatten tensor, dimension has to match the sum of lengths
*Outputs* | 
`bucktized_feat` | 2-D dense tensor, with 1st dim = float_features.dim(0), 2nd dim = size(indices)in the arg list, the tensor is of the same data type as `feature`.


### Code


[caffe2/operators/batch_bucketize_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_bucketize_op.cc)

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


[caffe2/operators/batch_sparse_to_dense_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_sparse_to_dense_op.cc)

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


[caffe2/operators/batch_gather_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_gather_ops.cc)

---



## BatchGatherGradient

No documentation yet.


### Code


[caffe2/operators/batch_gather_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_gather_ops.cc)

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


[caffe2/operators/batch_matmul_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_matmul_op.cc)

---



## BatchMoments

No documentation yet.


### Code


[caffe2/operators/batch_moments_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_moments_op.cc)

---



## BatchMomentsGradient

No documentation yet.


### Code


[caffe2/operators/batch_moments_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_moments_op.cc)

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


[caffe2/operators/one_hot_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/one_hot_ops.cc)

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


[caffe2/operators/batch_sparse_to_dense_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/batch_sparse_to_dense_op.cc)

---



## BatchToSpace


Rearranges (permutes) data from batch into blocks of spatial data, followed by cropping. This is the reverse transformation of  `SpaceToBatch` . More specifically, this op outputs a copy of the input tensor where values from the batch dimension are moved in spatial blocks to the height and width dimensions, followed by cropping along the height and width dimensions. Only "NCHW" order is currently supported.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "BatchToSpace",
    ["X"],
    ["Y"],
    pad=3
```

 )  workspace.FeedBlob("X", np.random.rand(10,3,32,32).astype(np.float32)) print("X.shape:", workspace.FetchBlob("X").shape) workspace.RunOperatorOnce(op) print("Y.shape:", workspace.FetchBlob("Y").shape)   ```    **Result**    ```   X.shape: (10, 3, 32, 32) Y.shape: (2, 3, 58, 58)   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`pad` | (*int*): exclusive axis that divides the first and second dimension of matrix `A` (default=0)
`block_size` | (*int*): height/width of spatial blocks to be moved (default=2)
`order` | (*string*): order of dimensions of input and output blobs; only "NCHW" order is currently supported (default="NCHW")
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor (NCHW order)
*Outputs* | 
`Y` | (*Tensor`<float>`*): output tensor (NCHW order)


### Code


[caffe2/operators/space_batch_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc)

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


[caffe2/operators/jsd_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/jsd_op.cc)

---



## BernoulliJSDGradient

No documentation yet.


### Code


[caffe2/operators/jsd_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/jsd_op.cc)

---



## BisectPercentile


 

```
    This operator is to map raw feature values into the percentile
    representations based on Bisection for more than one feature.

    The input is the bath of input feature values, with the size of (batch_size,
    num_feature), where num_feature = F (F >= 1).

    For each feature, we also need additional information regarding the feature
    value distribution.
    There are several vectors to keep data to percentile mappping information
    as arguments (context):
    1. feature raw values (R)
    2. feature percentile mapping (P)
    3. feature percentile lower bound (L)
    4. feature percentile upper bound (U)

    A toy example:
    Suppose the sampled data distribution is as follows:
    1, 1, 2, 2, 2, 2, 2, 2, 3, 4
    We have the mapping vectors as follows:
    R = [1, 2, 3, 4]
    P = [0.15, 0.55, 0.9, 1.0]
    L = [0.1, 0.3, 0.9, 1.0]
    U = [0.2, 0.8, 0.9, 1.0]
    Where P is computed as (L + U) / 2.

    For a given list of feature values, X = [x_0, x_1, ..., x_i, ...], for each
    feature value (x_i) we first apply bisection to find the right index (t),
    such that R[t] <= x_i < R[t+1].
    If x_i = R[t], P[t] is returned;
    otherwise, the interpolation is apply by (R[t], R[t+1]) and (U[t] and L[t]).

    As there are F features (F >= 1), we concate all the R_f, P_f, L_f, and
    U_f for each feature f and use an additional input length to keep track of
    the number of points for each set of raw feature value to percentile mapping.
    For example, there are two features:
    R_1 =[0.1, 0.4, 0.5];
    R_2 = [0.3, 1.2];
    We will build R = [0.1, 0.4, 0.5, 0.3, 1.2]; besides, we have
    lengths = [3, 2]
    to indicate the boundries of the percentile information.

```




### Interface


---------- | ----------
*Arguments* | 
`percentile_raw` | 1D tensor, which is the concatenation of all sorted raw feature values for all features.
`percentile_mapping` | 1D tensor. There is one-one mapping between percentile_mapping and percentile_raw such that each element in percentile_mapping corresponds to the percentile value of the corresponding raw feature value.
`percentile_lower` | 1D tensor. There is one-one mapping between percentile_upper and percentile_raw such that each element in percentile_mapping corresponds to the percentile lower bound of the corresponding raw feature value.
`percentile_upper` | 1D tensor. There is one-one mapping between percentile_upper and percentile_raw such that each element in percentile_mapping corresponds to the percentile upper bound of the corresponding raw feature value.
`lengths` | 1D tensor. There is one-one mapping between percentile_upper and percentile_raw such that each element in percentile_mapping corresponds to the percentile upper bound of the corresponding raw feature value.
*Inputs* | 
`raw_values` | Input 2D tensor of floats of size (N, D), where N is the batch size and D is the feature dimension.
*Outputs* | 
`percentile` | 2D tensor of output with the same dimensions as the input raw_values.


### Code


[caffe2/operators/bisect_percentile_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/bisect_percentile_op.cc)

---



## BitwiseAnd


Performs element-wise bitwise operation  `bitwise_and`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 ```  Argument `broadcast=1` needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)   


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor)* First operand.
`B` | *(type: Tensor)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor)* Output tensor. Has same dimensions as input `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## BitwiseOr


Performs element-wise bitwise operation  `bitwise_or`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 ```  Argument `broadcast=1` needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)   


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor)* First operand.
`B` | *(type: Tensor)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor)* Output tensor. Has same dimensions as input `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## BitwiseXor


Performs element-wise bitwise operation  `bitwise_xor`  (with limited broadcast support).
Both input operands should be of type  `bool` .
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 ```  Argument `broadcast=1` needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)   


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor)* First operand.
`B` | *(type: Tensor)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor)* Output tensor. Has same dimensions as input `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## BooleanMask


Given a 1D  `data`  tensor and a boolean  `mask`  tensor of the same shape, returns a  `masked_data`  tensor containing only the elements corresponding to positions where the  `mask`  is True, and a  `masked_indices`  tensor containing the indices of the True elements.
  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "BooleanMask",
    ["data", "mask"],
    ["masked_data", "masked_indices"]
```

 )  workspace.FeedBlob("data", np.array([1,2,3,4,5,6])) workspace.FeedBlob("mask", np.array([True,False,False,True,True,False])) print("data:", workspace.FetchBlob("data")) print("mask:", workspace.FetchBlob("mask")) workspace.RunOperatorOnce(op) print("masked_data:", workspace.FetchBlob("masked_data")) print("masked_indices:", workspace.FetchBlob("masked_indices"))   ```    **Result**    ```   data: [1 2 3 4 5 6] mask: [ True False False 

```
  True  True False]
```

 masked_data: [1 4 5] masked_indices: [0 3 4]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`data` | (*Tensor*): 1D input tensor
`mask` | (*Tensor`<bool>`*): tensor of bools which determines the input elements that will be left in the `masked_data` output tensor; same shape as `data`
*Outputs* | 
`masked_data` | (*Tensor*): 1D tensor of same type as `data` input that contains the masked input tensor
`masked_indices` | (*Tensor`<int>`*): 1D tensor of indices of the True elements in the `mask` tensor


### Code


[caffe2/operators/boolean_mask_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc)

---



## BooleanMaskLengths


Given a tensor of int32  `lengths`  tensor representing segment lengths and a  `mask`  (boolean) tensor, return the segment lengths of the corresponding segmented tensor after  **BooleanMask**  is applied.
 If  `lengths`  tensor is $[a_1, a_2, ..., a_n]$, then length of  `mask`  tensor must be $a_1 + a_2 + ... + a_n$.
  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "BooleanMaskLengths",
    ["lengths", "mask"],
    ["masked_lengths"]
```

 )  workspace.FeedBlob("lengths", np.array([1,3,2], dtype=np.int32)) workspace.FeedBlob("mask", np.array([False,True,True,False,True,True])) print("lengths:", workspace.FetchBlob("lengths")) print("mask:", workspace.FetchBlob("mask")) workspace.RunOperatorOnce(op) print("masked_lengths:", workspace.FetchBlob("masked_lengths"))   ```    **Result**    ```   lengths: [1 3 2] mask: [False 

```
  True  True False  True  True]
```

 masked_lengths: [0 2 2]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`lengths` | (*Tensor`<int>`*): input tensor containing segment lengths
`mask` | (*Tensor`<bool>`*): A 1D bool tensor of values to keep.
*Outputs* | 
`masked_lengths` | (*Tensor`<int>`*): 1D tensor of same type as inputs that contains the sequence


### Code


[caffe2/operators/boolean_mask_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc)

---



## BooleanUnmask


Given a series of masks and values, reconstruct values together according to masks. A comprehensive example:  ```  mask1  

```
  = True, False, True, False, False
```

 values1 = 1.0, 3.0 mask2  

```
  = False, True, False, False, False
```

 values2 = 2.0 mask3  

```
  = False, False, False, True, True
```

 values3 = 4.0, 5.0  ```   Reconstruct by:   ```  output = net.BooleanUnmask([mask1, values1, mask2, values2, mask3, values3], ["output"]) output = 1.0, 2.0, 3.0, 4.0, 5.0  ```   Note that for all mask positions, there must be at least one True. This is not allowed:   ```  mask1  

```
  = True, False
```

 values1 = 1.0 mask2  

```
  = False, False
```

 values2 =  output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"])  ```   If there are multiple True values for a field, we accept the first value, and no longer expect a value for that location:   ```  mask1  

```
  = True, False
```

 values1 = 1.0 mask2  

```
  = True, True
```

 values2 = 2.0  output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"]) output = 1.0, 2.0  ```    ***  Note that we alternate  `data`  and  `mask`  inputs  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_unmask_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_unmask_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "BooleanUnmask",
    ["mask1", "data1", "mask2", "data2"],
    ["unmasked_data"]
```

 )  workspace.FeedBlob("mask1", np.array([True,False,False,True,True,False])) workspace.FeedBlob("data1", np.array([1,4,5])) workspace.FeedBlob("mask2", np.array([False,True,True,False,False,True])) workspace.FeedBlob("data2", np.array([2,3,6]))  print("data1:", workspace.FetchBlob("data1")) print("mask1:", workspace.FetchBlob("mask1")) print("data2:", workspace.FetchBlob("data2")) print("mask2:", workspace.FetchBlob("mask2")) workspace.RunOperatorOnce(op) print("unmasked_data:", workspace.FetchBlob("unmasked_data"))   ```    **Result**    ```   data1: [1 4 5] mask1: [ True False False 

```
  True  True False]
```

 data2: [2 3 6] mask2: [False 

```
  True  True False False  True]
```

 unmasked_data: [1 2 3 4 5 6]   ```   </details> 


### Interface


---------- | ----------
*Inputs* | 
`data` | (*Tensor*): 1D input tensor(s)
`mask` | (*Tensor`<bool>`*): 1D boolean mask tensor(s)
*Outputs* | 
`unmasked_data` | (*Tensor*): 1D tensor of same type as `data` input that contains the unmasked input tensor


### Code


[caffe2/operators/boolean_unmask_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_unmask_ops.cc)

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
`rotated` | bool (default false). If true, then boxes (rois and deltas) include angle info to handle rotation. The format will be [ctr_x, ctr_y, width, height, angle (in degrees)].
*Inputs* | 
`scores` | Scores, size (count, num_classes)
`boxes` | Bounding box for each class, size (count, num_classes * 4). For rotated boxes, this would have an additional angle (in degrees) in the format [<optionaal_batch_id>, ctr_x, ctr_y, w, h, angle]. Size: (count, num_classes * 5).
`batch_splits` | Tensor of shape (batch_size) with each element denoting the number of RoIs/boxes belonging to the corresponding image in batch. Sum should add up to total count of scores/boxes.
*Outputs* | 
`scores` | Filtered scores, size (n)
`boxes` | Filtered boxes, size (n, 4). For rotated boxes, size (n, 5), format [ctr_x, ctr_y, w, h, angle].
`classes` | Class id for each filtered score/box, size (n)
`batch_splits` | Output batch splits for scores/boxes after applying NMS
`keeps` | Optional filtered indices, size (n)
`keeps_size` | Optional number of filtered indices per class, size (num_classes)


### Code


[caffe2/operators/box_with_nms_limit_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/box_with_nms_limit_op.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

---



## ByteWeightDequant

No documentation yet.


### Code


[caffe2/operators/byte_weight_dequant_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/byte_weight_dequant_op.cc)

---



## CTCBeamSearchDecoder

Prefix beam search decoder for connectionist temporal classification.


### Interface


---------- | ----------
*Arguments* | 
`beam_width` | Maximum number of candidates to carry over to next activation step.
`prune_threshold` | Probability threshold below which outputs are ignored.
*Inputs* | 
`INPUTS` | 3D float Tensor sized [max_activation_length, batch_size, alphabet_size] of network logits (before softmax application).
`SEQ_LEN` | (optional) 1D int vector containing sequence lengths, having size [batch_size] seq_len will be set to max_time if not provided.
*Outputs* | 
`OUTPUT_LEN` | Output_len matrix size (batch_size). Each index stores final output length of its corresponding batch item.
`VALUES` | Values vector, size (total_decoded_outputs). The flattened vector of final output sequences, in batch order.


### Code


[caffe2/operators/ctc_beam_search_decoder_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ctc_beam_search_decoder_op.cc)

---



## CTCGreedyDecoder

Greedy decoder for connectionist temporal classification.


### Interface


---------- | ----------
*Arguments* | 
`merge_repeated` | When merge_repeated is true, merge repeated classes in output.
*Inputs* | 
`INPUTS` | 3D float Tensor sized [max_time, batch_size, num_classes]
`SEQ_LEN` | (optional) 1D int vector containing sequence lengths, having size [batch_size]seq_len will be set to max_time if not provided
*Outputs* | 
`OUTPUT_LEN` | Output_len matrix size (batch). The row store: [decoded_length]
`VALUES` | Values vector, size (total_decoded_outputs). The vector stores the decoded classes


### Code


[caffe2/operators/ctc_greedy_decoder_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ctc_greedy_decoder_op.cc)

---



## Cast


Casts the elements of a given input tensor to a data type specified by the  `to`  argument and returns an output tensor of the same size in the converted type.
The  `to`  argument must be one of the data types specified in the  *DataType*  enum field in the TensorProto message (see below). If the  `to`  argument is not provided or is not one of the enumerated types in  *DataType* , Caffe2 throws an Enforce error.
 NOTE: Casting to and from strings is not supported yet.
 TensorProto  *DataType*  field:  ```  message TensorProto {  

```
  ...
  enum DataType {
    UNDEFINED = 0;
    FLOAT = 1;  // float
    INT32 = 2;  // int
    BYTE = 3;  // BYTE, when deserialized, is going to be restored as uint8.
    STRING = 4;  // string
    BOOL = 5;  // bool
    UINT8 = 6;  // uint8_t
    INT8 = 7;  // int8_t
    UINT16 = 8;  // uint16_t
    INT16 = 9;  // int16_t
    INT64 = 10;  // int64_t
    FLOAT16 = 12;  // caffe2::__f16, caffe2::float16
    DOUBLE = 13;  // double
  }
```

 

```
    "Cast",
    ["X"],
    ["Y"],
    to=2
```

 ```   Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cast_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cast_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32)*10) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))  ```    **Result**    ```  X: [[9.436466  

```
  5.8529844  0.54932857]
```

  [1.1583444 

```
  2.9936118  0.22950427]
```

  [3.9143739 

```
  3.4040766  8.905341  ]]
```

 Y: [[9 5 0]  [1 2 0]  [3 3 8]]  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`to` | *(type: int)* Data type to which the elements of the input tensor are cast. Strictly must be one of the types from *DataType* enum in TensorProto.
*Inputs* | 
`X` | *(type: Tensor)* Input tensor to be cast.
*Outputs* | 
`Y` | *(type: Tensor`<'to' type>`)* Output tensor with the same shape as input with type specified by the `to` argument.


### Code


[caffe2/operators/cast_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cast_op.cc)

---



## Cbrt

No documentation yet.


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor calculated as the cbrt of the input tensor, element-wise.


### Code


[caffe2/operators/cbrt_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cbrt_op.cc)

---



## CbrtGradient

No documentation yet.


### Code


[caffe2/operators/cbrt_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cbrt_op.cc)

---



## Ceil


Element-wise application of the ceil function ($y=ceil(x)$) to the input tensor  `X` . Output tensor shape is the same as the input tensor.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ceil_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ceil_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Ceil",
    ["X"],
    ["X"],
```

 )  workspace.FeedBlob("X", (np.random.uniform(-10, 10, (5,5))).astype(np.float32)) print("X before running op:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("X after running op:", workspace.FetchBlob("X"))   ```    **Result**    ```   X before running op: [[ 8.44598 

```
    -6.5098248  -2.2993476  -7.6859694   0.58566964]
```

  [-7.846551  

```
  -0.03689406  6.9362907  -4.0521703   4.4969673 ]
```

  [ 0.33355865 -7.895527  

```
  -8.393201    9.374202   -2.3930092 ]
```

  [-6.3061996  

```
  3.1403487   3.782099   -8.516556   -2.8387244 ]
```

  [-2.0164998  

```
  4.7663913  -3.422966    0.3636999   8.75713   ]]
```

 X after running op: [[ 9. -6. -2. -7. 

```
  1.]
```

  [-7. -0. 

```
  7. -4.  5.]
```

  [ 1. -7. -8. 10. -2.]  [-6. 

```
  4.  4. -8. -2.]
```

  [-2. 

```
  5. -3.  1.  9.]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.


### Code


[caffe2/operators/ceil_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ceil_op.cc)

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


[caffe2/operators/channel_backprop_stats_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/channel_backprop_stats_op.cc)

---



## ChannelShuffle

No documentation yet.


### Code


[caffe2/operators/channel_shuffle_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/channel_shuffle_op.cc)

---



## ChannelShuffleGradient

No documentation yet.


### Code


[caffe2/operators/channel_shuffle_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/channel_shuffle_op.cc)

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


[caffe2/operators/channel_stats_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/channel_stats_op.cc)

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


[caffe2/operators/atomic_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/atomic_ops.cc)

---



## CheckCounterDone


If the internal count value <= 0, outputs true, otherwise outputs false.
  

```
  Github Links:
  - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc


```

 <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  createcounter_op = core.CreateOperator(  

```
    "CreateCounter",
    [],
    ["counter"],
    init_count=5
```

 )  retrievecount_op = core.CreateOperator(  

```
    "RetrieveCount",
    ["counter"],
    ["count"]
```

 )  checkcounterdone_op = core.CreateOperator(  

```
    "CheckCounterDone",
    ["counter"],
    ["done"]
```

 )  countup_op = core.CreateOperator(  

```
    "CountUp",
    ["counter"],
    ["previous_count"],
```

 )  countdown_op = core.CreateOperator(  

```
    "CountDown",
    ["counter"],
    ["done"],
```

 )  resetcounter_op = core.CreateOperator(  

```
    "ResetCounter",
    ["counter"],
    ["previous_count"],
    init_count=3
```

 )   # Create counter workspace.RunOperatorOnce(createcounter_op) print("'counter' pointer:", workspace.FetchBlob("counter"))   # Retrieve initial counter value workspace.RunOperatorOnce(retrievecount_op) print("Initial 'count':", workspace.FetchBlob("count"))   # Check if counter is done workspace.RunOperatorOnce(checkcounterdone_op) print("Initial 'done' value:", workspace.FetchBlob("done"))   # Test CountUp operator print("\nTesting CountUp operator...") for i in range(5):  

```
    workspace.RunOperatorOnce(countup_op)
    print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

```

 workspace.RunOperatorOnce(retrievecount_op) print("'count' value after CountUp test:", workspace.FetchBlob("count"))   # Test CountDown operator print("\nTesting CountDown operator...") for i in range(11):  

```
    workspace.RunOperatorOnce(countdown_op)
    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))
```

 ```    **Result**   ``` 'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
Initial 'count': 5 Initial 'done' value: False  Testing CountUp operator...
'previous_count' after CountUp: 5 'previous_count' after CountUp: 6 'previous_count' after CountUp: 7 'previous_count' after CountUp: 8 'previous_count' after CountUp: 9 'count' value after CountUp test: 10  Testing CountDown operator...
'count' value after CountDown: 9	'done' value: False 'count' value after CountDown: 8	'done' value: False 'count' value after CountDown: 7	'done' value: False 'count' value after CountDown: 6	'done' value: False 'count' value after CountDown: 5	'done' value: False 'count' value after CountDown: 4	'done' value: False 'count' value after CountDown: 3	'done' value: False 'count' value after CountDown: 2	'done' value: False 'count' value after CountDown: 1	'done' value: False 'count' value after CountDown: 0	'done' value: False 'count' value after CountDown: -1	'done' value: True ```  </details>  


### Interface


---------- | ----------
*Inputs* | 
`counter` | *(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.
*Outputs* | 
`done` | *(type: bool)* True if the internal count is zero or negative, otherwise False.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)

---



## Clip


This operator limits the given input within an interval. The interval is specified by the  `min`  and  `max`  arguments. They default to  *numeric_limits::lowest()*  and  *numeric_limits::max()*  respectively. The clipping operation can be done in an in-place fashion by using the same output blob as the input blob.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/clip_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/clip_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Clip",
    ["X"],
    ["Y"],
    min=20.0,
    max=60.0

```

 )  workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```  X: [[45. 16. 59. 99. 48.]  [12. 44. 46. 82. 28.]  [ 1. 91. 18. 

```
  9. 71.]
```

  [24. 37. 61. 12. 81.]  [36. 38. 30. 84. 40.]] Y: [[45. 20. 59. 60. 48.]  [20. 44. 46. 60. 28.]  [20. 60. 20. 20. 60.]  [24. 37. 60. 20. 60.]  [36. 38. 30. 60. 40.]]  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`min` | *(type: float)* Minimum value, under which element is replaced by min (default=*numeric_limits::lowest()*).
`max` | *(type: float)* Maximum value, under which element is replaced by max (default=*numeric_limits::max()*).
*Inputs* | 
`X` | *(Tensor`<float>`)* Input tensor within range [*numeric_limits::lowest()*, *numeric_limits::max()*].
*Outputs* | 
`Y` | *(Tensor`<float>`)* Output tensor clipped within range [`min`, `max`].


### Code


[caffe2/operators/clip_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/clip_op.cc)

---



## ClipGradient

No documentation yet.


### Code


[caffe2/operators/clip_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/clip_op.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

---



## Col2Im

No documentation yet.


### Code


[caffe2/operators/im2col_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/im2col_op.cc)

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
`rpn_rois_fpn2` | RPN proposals for FPN level 2, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn3` | RPN proposals for FPN level 3, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn4` | RPN proposals for FPN level 4, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn5` | RPN proposals for FPN level 5, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_rois_fpn6` | RPN proposals for FPN level 6, format (image_index, x1, y1, x2, y2). See rpn_rois documentation from GenerateProposals.
`rpn_roi_probs_fpn2` | RPN objectness probabilities for FPN level 2. See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn3` | RPN objectness probabilities for FPN level 3. See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn4` | RPN objectness probabilities for FPN level 4. See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn5` | RPN objectness probabilities for FPN level 5. See rpn_roi_probs documentation from GenerateProposals.
`rpn_roi_probs_fpn6` | RPN objectness probabilities for FPN level 6. See rpn_roi_probs documentation from GenerateProposals.
*Outputs* | 
`rois` | Top proposals limited to rpn_post_nms_topN total, format (image_index, x1, y1, x2, y2)
`rois_fpn2` | RPN proposals for ROI level 2, format (image_index, x1, y1, x2, y2)
`rois_fpn3` | RPN proposals for ROI level 3, format (image_index, x1, y1, x2, y2)
`rois_fpn4` | RPN proposals for ROI level 4, format (image_index, x1, y1, x2, y2)
`rois_fpn5` | RPN proposals for ROI level 5, format (image_index, x1, y1, x2, y2)
`rois_idx_restore` | Permutation on the concatenation of all rois_fpni, i=min...max, such that when applied the RPN RoIs are restored to their original order in the input blobs.


### Code


[caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/collect_and_distribute_fpn_rpn_proposals_op.cc)

---



## CollectTensor


Collect tensor into tensor vector by reservoir sampling, argument num_to_collect indicates the max number of tensors that will be collected. The first half of the inputs are tensor vectors, which are also the outputs. The second half of the inputs are the tensors to be collected into each vector (in the same order). The input tensors are collected in all-or-none manner. If they are collected, they will be placed at the same index in the output vectors.



### Interface


---------- | ----------
*Arguments* | 
`num_to_collect` | The max number of tensors to collect


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## ColwiseMax


Compute column-wise max reduction of the input tensor. This op takes one input, $X$, of shape $BxMxN$, where $B$ is the batch size, $M$ is number of rows, and $N$ is number of columns. The output of this op, $Y$, is a matrix of shape $BxN$, with one row for each element of the batch, and the same number of columns as the input tensor.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ColwiseMax",
    ["X"],
    ["Y"]
```

 )  # Create X, simulating a batch of 2, 4x4 matricies X = np.random.randint(0,high=20,size=(2,4,4)) print("X:\n",X)  # Feed X into workspace workspace.FeedBlob("X", X.astype(np.float32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[[17 15 

```
  2  6]
  [ 8 12  6  0]
  [ 6  9  7  3]
  [ 4 13 16 13]]

```

  [[ 0 

```
  3  4 12]
  [18  1 17 12]
  [ 7 17 13 14]
  [12 17  2  1]]]
```

 Y:  [[17. 15. 16. 13.]  [18. 17. 17. 14.]]   ```   </details>      


### Interface


---------- | ----------
*Inputs* | 
`X` | A tensor of dimensions $B x M x N$ to compute columnwise-max. Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of each element of the batch, respectively.
*Outputs* | 
`Y` | The output tensor of shape $B x N$, where each row represents the column-wise maximums for that element of the input batch.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

---



## ColwiseMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## Concat


Concatenate a list of tensors into a single tensor. Similar functionality to Numpy's [concatenate]( [https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html))  function. The  `axis`  argument specifies what axis along which the arrays will be concatenated.
When set to non-zero (default=0), the  `add_axis`  argument adds the axis specified in  `axis`  to all input tensors.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Concat",
    ["X1",  "X2"],
    ["Y", "split_info"],
    axis=0
```

 )  workspace.FeedBlob("X1", np.array([[1,2],[3,4]])) workspace.FeedBlob("X2", np.array([[5,6]])) print("X1:", workspace.FetchBlob("X1")) print("X2:", workspace.FetchBlob("X2")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y")) print("split_info:", workspace.FetchBlob("split_info"))   ```    **Result**    ```   X1: [[1 2]  [3 4]] X2: [[5 6]] Y: [[1 2]  [3 4]  [5 6]] split_info: [2 1]   ```   </details>  <details>  <summary> <b>Example 2</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Concat",
    ["X1",  "X2"],
    ["Y", "split_info"],
    add_axis=1,
    axis=3
```

 )  workspace.FeedBlob("X1", np.random.randint(10, size=(1, 1, 5, 5))) # NCHW workspace.FeedBlob("X2", np.random.randint(10, size=(1, 1, 5, 5))) # NCHW print("X1:", workspace.FetchBlob("X1")) print("X2:", workspace.FetchBlob("X2")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y")) print("split_info:", workspace.FetchBlob("split_info"))   ```    **Result**    ```   X1: [[[[1 8 3 9 0]   

```
  [6 4 6 5 6]
```

   

```
  [3 9 1 9 9]
```

   

```
  [5 1 0 7 7]
```

   

```
  [9 4 0 0 9]]]]
```

 X2: [[[[7 0 2 6 1]   

```
  [3 9 4 0 3]
```

   

```
  [5 3 8 9 4]
```

   

```
  [3 4 2 1 0]
```

   

```
  [0 8 8 8 1]]]]
```

 Y: [[[[[1 8 3 9 0]  

```
    [7 0 2 6 1]]

```

   

```
  [[6 4 6 5 6]
    [3 9 4 0 3]]

```

   

```
  [[3 9 1 9 9]
    [5 3 8 9 4]]

```

   

```
  [[5 1 0 7 7]
    [3 4 2 1 0]]

```

   

```
  [[9 4 0 0 9]
    [0 8 8 8 1]]]]]
```

 split_info: [1 1]   ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`axis` | *(type: int; default: -1)* Axis to concatenate on.
`order` | *(type: string; default='NCHW')* Order of blob dimensions. Concats on the C dimension.
`add_axis` | *(type: int)* Pass non-zero integer to add the axis specified in `axis` to all input tensors.
*Inputs* | 
`X1, X2, ...` | *(type: Tensor`<float>`)* List of input tensors.
*Outputs* | 
`concat_result` | *(type: Tensor`<float>`)* Concatenated tensor.
`split_info` | *(type: Tensor`<int>`)* The dimensions of the inputs.


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/conditional_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conditional_op.cc)

---



## ConditionalSetAtomicBool


Set an atomic<bool> to true if the given condition bool variable is true     


### Interface


---------- | ----------
*Inputs* | 
`atomic_bool` | Blob containing a unique_ptr<atomic<bool>>
`condition` | Blob containing a bool


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/atomic_ops.cc)

---



## ConstantFill


This operator fills the elements of the output tensor with a constant value specified by the  `value`  argument.
 - The data type is specified by the  `dtype`  argument  - Currently, the data types supported are  *float* ,  *int32* ,  *int64* , and  *bool*   - If the  `dtype`  argument is not provided, the data type of  `value`  is used  - The output tensor shape is either specified by the  `shape`  argument or will match the shape of the input tensor if one is provided (if an input tensor is provided, a shape argument should not be set)  - Optional additional dimensions can be appended at the end as specified by  `extra_shape`  argument  - If  `input_as_shape`  is set to True, the input should be a 1D tensor containing the desired output shape (the dimensions specified in  `extra_shape`  will also be appended)  When specifying  `dtype`  argument, use the integer keys from the  *DataType*  enum in TensorProto:   ```  message TensorProto {  

```
  ...
  enum DataType {
    UNDEFINED = 0;
    FLOAT = 1;  // float
    INT32 = 2;  // int
    BYTE = 3;  // BYTE, when deserialized, is going to be restored as uint8.
    STRING = 4;  // string
    BOOL = 5;  // bool
    UINT8 = 6;  // uint8_t
    INT8 = 7;  // int8_t
    UINT16 = 8;  // uint16_t
    INT16 = 9;  // int16_t
    INT64 = 10;  // int64_t
    FLOAT16 = 12;  // caffe2::__f16, caffe2::float16
    DOUBLE = 13;  // double
  }
```

 

```
    "ConstantFill",
    [],
    ["Y"],
    shape=(1,5,5)
```

 ```   Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))  ```    **Result**    ```  Y: [[[0. 0. 0. 0. 0.]  

```
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0.]]]
```

 

```
    "ConstantFill",
    ["X"],
    ["Y"],
    value=4.0,
    dtype=1,
    extra_shape=(1,2)
```

 ```  </details>  <details> <summary> <b>Example 2</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("X", (np.random.randint(100, size=(3,3))).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))  ```    **Result**    ```  X: [[86. 30. 84.]  [34. 51. 

```
  9.]
```

  [29. 86. 59.]] Y: [[[[4. 4.]]   

```
  [[4. 4.]]

  [[4. 4.]]]


```

  [[[4. 4.]]   

```
  [[4. 4.]]

  [[4. 4.]]]


```

  [[[4. 4.]]   

```
  [[4. 4.]]

  [[4. 4.]]]]
```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`value` | *(type: primitive; default: 0.0f) value to populate output tensor with.
`dtype` | *(type: int)* The data type for the elements of the output tensor. Strictly must be one of the types from *DataType* enum in TensorProto.
`shape` | *(type: int | Tuple(int))* Shape of the output tensor. Cannot pass an input blob and this arg at the same time.
`extra_shape` | *(type: int | Tuple(int))* Additional dimensions appended at the end of the shape indicated by the input blob. Cannot set thisargument when there is no input blob.
`input_as_shape` | *(type: int | Tuple(int))* 1D tensor containing the desired output shape. First input must be in CPU context.
*Inputs* | 
`X` | *(type: Tensor)* [OPTIONAL] Input tensor to provide shape information.
*Outputs* | 
`Y` | *(type: Tensor)* Output tensor of constant values.


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

---



## Conv


The convolution operator consumes an input vector, a filter blob and a bias blob and computes the output.  The Conv2D operator computes a 2D convolution operation over an input blob $(X)$, with a filter blob $(filter)$ and a bias blob $(bias)$, and outputs a single output blob $(Y)$. Although there are several options for order, the convention is that the input $(X)$ is a blob of shape $(N,C_{in},H_{in},W_{in})$ and the output $(Y)$ is a blob of shape $(N,C_{out},H_{out},W_{out})$. Here, $N$ is the batch size, $C$ is the number of channels, $H$ is the spatial height, and $W$ is the spatial width. For example, if your input data was a batch of five, 100x120pixel RGB images, $X$ would have shape $(5,3,120,100)$.
 The $filter$ input blob may contain multiple filters and has shape $(M, C_{in}, K_H, K_W)$. Here, $M$ is the number of individual filters contained in the blob, $C_{in}$ is the number of channels of each filter (by convention in 2D convolution it is the same as the number of channels in the input), $K_H$ is the spatial height of the kernel, and $K_W$ is the spatial width of the kernel. The $bias$ blob is a vector of length $M$, where there is one bias for each filter in the $filter$ blob.
 Given the shape of the input blob and the filter blob, we can calculate the shape of the output blob as follows. The number of items in the batch $N$ will stay the same. The number of channels in the output will equal the number of kernels in the filter blob, so $C_{out} = M.$ With stride and pad defined below, the spatial height and width of the output ($H_{out}$ and $W_{out}$) are calculated as  $$H_{out} = \left \lfloor{\frac{H_{in} - K_H + 2 *pad}{stride}+1}\right \rfloor$$   $$W_{out} = \left \lfloor{\frac{W_{in} - K_W + 2* pad}{stride}+1}\right \rfloor$$   Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Conv",
    ["X", "filter", "bias"],
    ["Y"],
    kernel=5,
    pad=1,
    stride=2
```

 )  # Create X: (N,C,H,W) data = np.random.randn(1,1,8,8).astype(np.float32) print("Data shape: ",data.shape)  # Create W: (M,C,Kh,Kw) filters = np.random.randn(3,1,5,5).astype(np.float32) print("Filter shape: ",filters.shape)  # Create b: M bias = np.array([1.,1.,1.]).astype(np.float32) print("Bias shape: ",bias.shape)  # Put the inputs into the workspace workspace.FeedBlob("X", data) workspace.FeedBlob("filter", filters) workspace.FeedBlob("bias", bias)  # Run the operator workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   Data shape: 

```
  (1, 1, 8, 8)
```

 Filter shape: 

```
  (3, 1, 5, 5)
```

 Bias shape: 

```
  (3,)
```

 Y:  [[[[ 

```
  0.6406407    0.8620521    0.56461596]
```

   

```
  [ -1.5042953   -0.79549205 -10.683343  ]
```

   

```
  [ -0.5240259    3.4538248   -3.9564204 ]]

  [[  0.6876496    4.8328524   -1.9525816 ]
```

   

```
  [  1.2995434   -2.3895378    7.2670045 ]
```

   

```
  [  3.9929862    1.8126237    5.4699917 ]]

  [[  3.55949      4.7934155    0.76086235]
```

   

```
  [  3.9588015   -1.3251319    4.413117  ]
```

   

```
  [ -1.5296054   -1.4924102   -3.2552304 ]]]]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be convolved with the kernels in the filter blob.
`filter` | The filter blob, of shape $(M, C_{in}, K_H, K_W)$, containing the filters to be convolved with the data.
`bias` | The bias blob, of length $M$, containing the biases for the convolution, one bias per filter.
*Outputs* | 
`Y` | Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the convolution.


### Code


[caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)

---



## Conv1D


The convolution operator consumes an input vector, a 1D filter blob and a bias blob and computes the output.  The Conv2D operator computes a 2D convolution operation over an input blob $(X)$, with a filter blob $(filter)$ and a bias blob $(bias)$, and outputs a single output blob $(Y)$. Although there are several options for order, the convention is that the input $(X)$ is a blob of shape $(N,C_{in},H_{in},W_{in})$ and the output $(Y)$ is a blob of shape $(N,C_{out},H_{out},W_{out})$. Here, $N$ is the batch size, $C$ is the number of channels, $H$ is the spatial height, and $W$ is the spatial width. For example, if your input data was a batch of five, 100x120pixel RGB images, $X$ would have shape $(5,3,120,100)$.
 The $filter$ input blob may contain multiple filters and has shape $(M, C_{in}, K_H, K_W)$. Here, $M$ is the number of individual filters contained in the blob, $C_{in}$ is the number of channels of each filter (by convention in 2D convolution it is the same as the number of channels in the input), $K_H$ is the spatial height of the kernel, and $K_W$ is the spatial width of the kernel. The $bias$ blob is a vector of length $M$, where there is one bias for each filter in the $filter$ blob.
 Given the shape of the input blob and the filter blob, we can calculate the shape of the output blob as follows. The number of items in the batch $N$ will stay the same. The number of channels in the output will equal the number of kernels in the filter blob, so $C_{out} = M.$ With stride and pad defined below, the spatial height and width of the output ($H_{out}$ and $W_{out}$) are calculated as  $$H_{out} = \left \lfloor{\frac{H_{in} - K_H + 2 *pad}{stride}+1}\right \rfloor$$   $$W_{out} = \left \lfloor{\frac{W_{in} - K_W + 2* pad}{stride}+1}\right \rfloor$$   Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Conv",
    ["X", "filter", "bias"],
    ["Y"],
    kernel=5,
    pad=1,
    stride=2
```

 )  # Create X: (N,C,H,W) data = np.random.randn(1,1,8,8).astype(np.float32) print("Data shape: ",data.shape)  # Create W: (M,C,Kh,Kw) filters = np.random.randn(3,1,5,5).astype(np.float32) print("Filter shape: ",filters.shape)  # Create b: M bias = np.array([1.,1.,1.]).astype(np.float32) print("Bias shape: ",bias.shape)  # Put the inputs into the workspace workspace.FeedBlob("X", data) workspace.FeedBlob("filter", filters) workspace.FeedBlob("bias", bias)  # Run the operator workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   Data shape: 

```
  (1, 1, 8, 8)
```

 Filter shape: 

```
  (3, 1, 5, 5)
```

 Bias shape: 

```
  (3,)
```

 Y:  [[[[ 

```
  0.6406407    0.8620521    0.56461596]
```

   

```
  [ -1.5042953   -0.79549205 -10.683343  ]
```

   

```
  [ -0.5240259    3.4538248   -3.9564204 ]]

  [[  0.6876496    4.8328524   -1.9525816 ]
```

   

```
  [  1.2995434   -2.3895378    7.2670045 ]
```

   

```
  [  3.9929862    1.8126237    5.4699917 ]]

  [[  3.55949      4.7934155    0.76086235]
```

   

```
  [  3.9588015   -1.3251319    4.413117  ]
```

   

```
  [ -1.5296054   -1.4924102   -3.2552304 ]]]]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be convolved with the kernels in the filter blob.
`filter` | The filter blob, of shape $(M, C_{in}, K_H, K_W)$, containing the filters to be convolved with the data.
`bias` | The bias blob, of length $M$, containing the biases for the convolution, one bias per filter.
*Outputs* | 
`Y` | Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the convolution.


### Code


[caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)

---



## Conv1DGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## Conv2D


The convolution operator consumes an input vector, a 2D filter blob and a bias blob and computes the output.  The Conv2D operator computes a 2D convolution operation over an input blob $(X)$, with a filter blob $(filter)$ and a bias blob $(bias)$, and outputs a single output blob $(Y)$. Although there are several options for order, the convention is that the input $(X)$ is a blob of shape $(N,C_{in},H_{in},W_{in})$ and the output $(Y)$ is a blob of shape $(N,C_{out},H_{out},W_{out})$. Here, $N$ is the batch size, $C$ is the number of channels, $H$ is the spatial height, and $W$ is the spatial width. For example, if your input data was a batch of five, 100x120pixel RGB images, $X$ would have shape $(5,3,120,100)$.
 The $filter$ input blob may contain multiple filters and has shape $(M, C_{in}, K_H, K_W)$. Here, $M$ is the number of individual filters contained in the blob, $C_{in}$ is the number of channels of each filter (by convention in 2D convolution it is the same as the number of channels in the input), $K_H$ is the spatial height of the kernel, and $K_W$ is the spatial width of the kernel. The $bias$ blob is a vector of length $M$, where there is one bias for each filter in the $filter$ blob.
 Given the shape of the input blob and the filter blob, we can calculate the shape of the output blob as follows. The number of items in the batch $N$ will stay the same. The number of channels in the output will equal the number of kernels in the filter blob, so $C_{out} = M.$ With stride and pad defined below, the spatial height and width of the output ($H_{out}$ and $W_{out}$) are calculated as  $$H_{out} = \left \lfloor{\frac{H_{in} - K_H + 2 *pad}{stride}+1}\right \rfloor$$   $$W_{out} = \left \lfloor{\frac{W_{in} - K_W + 2* pad}{stride}+1}\right \rfloor$$   Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Conv",
    ["X", "filter", "bias"],
    ["Y"],
    kernel=5,
    pad=1,
    stride=2
```

 )  # Create X: (N,C,H,W) data = np.random.randn(1,1,8,8).astype(np.float32) print("Data shape: ",data.shape)  # Create W: (M,C,Kh,Kw) filters = np.random.randn(3,1,5,5).astype(np.float32) print("Filter shape: ",filters.shape)  # Create b: M bias = np.array([1.,1.,1.]).astype(np.float32) print("Bias shape: ",bias.shape)  # Put the inputs into the workspace workspace.FeedBlob("X", data) workspace.FeedBlob("filter", filters) workspace.FeedBlob("bias", bias)  # Run the operator workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   Data shape: 

```
  (1, 1, 8, 8)
```

 Filter shape: 

```
  (3, 1, 5, 5)
```

 Bias shape: 

```
  (3,)
```

 Y:  [[[[ 

```
  0.6406407    0.8620521    0.56461596]
```

   

```
  [ -1.5042953   -0.79549205 -10.683343  ]
```

   

```
  [ -0.5240259    3.4538248   -3.9564204 ]]

  [[  0.6876496    4.8328524   -1.9525816 ]
```

   

```
  [  1.2995434   -2.3895378    7.2670045 ]
```

   

```
  [  3.9929862    1.8126237    5.4699917 ]]

  [[  3.55949      4.7934155    0.76086235]
```

   

```
  [  3.9588015   -1.3251319    4.413117  ]
```

   

```
  [ -1.5296054   -1.4924102   -3.2552304 ]]]]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be convolved with the kernels in the filter blob.
`filter` | The filter blob, of shape $(M, C_{in}, K_H, K_W)$, containing the filters to be convolved with the data.
`bias` | The bias blob, of length $M$, containing the biases for the convolution, one bias per filter.
*Outputs* | 
`Y` | Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the convolution.


### Code


[caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)

---



## Conv2DGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## Conv3D


The convolution operator consumes an input vector, a 3D filter blob and a bias blob and computes the output.  The Conv2D operator computes a 2D convolution operation over an input blob $(X)$, with a filter blob $(filter)$ and a bias blob $(bias)$, and outputs a single output blob $(Y)$. Although there are several options for order, the convention is that the input $(X)$ is a blob of shape $(N,C_{in},H_{in},W_{in})$ and the output $(Y)$ is a blob of shape $(N,C_{out},H_{out},W_{out})$. Here, $N$ is the batch size, $C$ is the number of channels, $H$ is the spatial height, and $W$ is the spatial width. For example, if your input data was a batch of five, 100x120pixel RGB images, $X$ would have shape $(5,3,120,100)$.
 The $filter$ input blob may contain multiple filters and has shape $(M, C_{in}, K_H, K_W)$. Here, $M$ is the number of individual filters contained in the blob, $C_{in}$ is the number of channels of each filter (by convention in 2D convolution it is the same as the number of channels in the input), $K_H$ is the spatial height of the kernel, and $K_W$ is the spatial width of the kernel. The $bias$ blob is a vector of length $M$, where there is one bias for each filter in the $filter$ blob.
 Given the shape of the input blob and the filter blob, we can calculate the shape of the output blob as follows. The number of items in the batch $N$ will stay the same. The number of channels in the output will equal the number of kernels in the filter blob, so $C_{out} = M.$ With stride and pad defined below, the spatial height and width of the output ($H_{out}$ and $W_{out}$) are calculated as  $$H_{out} = \left \lfloor{\frac{H_{in} - K_H + 2 *pad}{stride}+1}\right \rfloor$$   $$W_{out} = \left \lfloor{\frac{W_{in} - K_W + 2* pad}{stride}+1}\right \rfloor$$   Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Conv",
    ["X", "filter", "bias"],
    ["Y"],
    kernel=5,
    pad=1,
    stride=2
```

 )  # Create X: (N,C,H,W) data = np.random.randn(1,1,8,8).astype(np.float32) print("Data shape: ",data.shape)  # Create W: (M,C,Kh,Kw) filters = np.random.randn(3,1,5,5).astype(np.float32) print("Filter shape: ",filters.shape)  # Create b: M bias = np.array([1.,1.,1.]).astype(np.float32) print("Bias shape: ",bias.shape)  # Put the inputs into the workspace workspace.FeedBlob("X", data) workspace.FeedBlob("filter", filters) workspace.FeedBlob("bias", bias)  # Run the operator workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   Data shape: 

```
  (1, 1, 8, 8)
```

 Filter shape: 

```
  (3, 1, 5, 5)
```

 Bias shape: 

```
  (3,)
```

 Y:  [[[[ 

```
  0.6406407    0.8620521    0.56461596]
```

   

```
  [ -1.5042953   -0.79549205 -10.683343  ]
```

   

```
  [ -0.5240259    3.4538248   -3.9564204 ]]

  [[  0.6876496    4.8328524   -1.9525816 ]
```

   

```
  [  1.2995434   -2.3895378    7.2670045 ]
```

   

```
  [  3.9929862    1.8126237    5.4699917 ]]

  [[  3.55949      4.7934155    0.76086235]
```

   

```
  [  3.9588015   -1.3251319    4.413117  ]
```

   

```
  [ -1.5296054   -1.4924102   -3.2552304 ]]]]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be convolved with the kernels in the filter blob.
`filter` | The filter blob, of shape $(M, C_{in}, K_H, K_W)$, containing the filters to be convolved with the data.
`bias` | The bias blob, of length $M$, containing the biases for the convolution, one bias per filter.
*Outputs* | 
`Y` | Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the convolution.


### Code


[caffe2/operators/conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc)

---



## Conv3DGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## ConvGradient

No documentation yet.


### Code


[caffe2/operators/conv_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_gradient_op.cc)

---



## ConvTranspose


The ConvTranspose op takes an input data tensor $X$, an input weight tensor $filter$, and optionally an input bias tensor $bias$. It then computes the transposed convolution, sometimes referred to as deconvolution, and produces a single output tensor $Y$. The hyperparameters of the op such as kernel size, stride, and padding are specified as args. At each stride, the filter is deconvolved with a subset of $X$ and the $bias$ is added. This is done throughout the input data until the output computation is complete.
 The output shapes are computed as follows. The number of channels in the output feature map is the number of kernels specified in the filter blob. The spatial height and width are computed as:  $$H_{out} = (H_{in}-1) *strides[0] - 2* pads[0] + kernels[0]$$   $$W_{out} = (W_{in}-1) *strides[1] - 2* pads[1] + kernels[1]$$  Note on the implementation layout: conv_transpose_op_impl.h is the templated implementation of the conv_transpose_op.h file, which is why they are separate files. Also, in the implementation this operator inherits from the  *ConvTransposeUnpoolOpBase*  operator.
 Github Links: -  [https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_op.h](https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_op.h)  -  [https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_op.cc](https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_op.cc)  -  [https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_unpool_op_base.h](https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_unpool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ConvTranspose",
    ["X", "filter", "bias"],
    ["Y"],
    kernels=[2,2],
    pads=[4,4,4,4],
    strides=[2,2]
```

 )  # Create X: (N,C,H,W) data = np.random.randn(2,3,5,5).astype(np.float32) print("Data shape: ",data.shape)  # Create filter: (M,C,Kh,Kw) filters = np.random.randn(3,1,2,2).astype(np.float32) print("Filter shape: ",filters.shape)  # Create b: M bias = np.array([1.]).astype(np.float32) print("Bias shape: ",bias.shape)  # Put the inputs into the workspace workspace.FeedBlob("X", data) workspace.FeedBlob("filter", filters) workspace.FeedBlob("bias", bias)  # Run the operator workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   Data shape: 

```
  (2, 3, 5, 5)
```

 Filter shape: 

```
  (3, 1, 2, 2)
```

 Bias shape: 

```
  (1,)
```

 Y:  [[[[0.53606427 0.5775447 ]   

```
  [0.40148795 1.5188271 ]]]


```

  [[[1.9903406 

```
  3.2794335 ]
```

   

```
  [0.09960175 0.31917763]]]]

```

 ```   </details>    


### Interface


---------- | ----------
*Arguments* | 
`legacy_pad` | *(type: int; optional)* Should the legacy padding be VALID or SAME. When used, pads should not be used.
`kernels` | *(type: [int]; default: [])* Desired kernel size. If left at default the kernel size will be inferred from the input $filter$ blob.
`strides` | *(type: [int]; default: [])* Controls the stride of the kernel as it traverses the input blob.
`pads` | *(type: [int]; default: [])* Controls the amount of padding applied to the input feature map before computation.
`adjs` | *(type: [int]; default: [])*
`order` | *(type: string; default: "NCHW")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is "NHWC".
`shared_buffer` | *(type: int; default: 0)*
`no_bias` | *(type: bool; default: False)* 
*Inputs* | 
`X` | Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be operated on.
`filter` | The filter blob, of shape $(M, C_{out}, K_H, K_W)$, containing the filters to be used in the transposed convolution.
`bias` | The bias blob, of length $C_{out}$, containing the biases for the operation, one bias per output channel. If not passed, biases assumed to be zeros.
*Outputs* | 
`Y` | Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the operation.


### Code


[caffe2/operators/conv_transpose_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_transpose_op.cc)

---



## ConvTransposeGradient

No documentation yet.


### Code


[caffe2/operators/conv_transpose_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_transpose_gradient_op.cc)

---



## Copy


Copy input tensor into output, potentially across devices.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Copy",
    ["input"],
    ["output"]
```

 )  workspace.FeedBlob("input", np.random.rand(3,3)) print("input:", workspace.FetchBlob("input")) workspace.RunOperatorOnce(op) print("output:", workspace.FetchBlob("output"))   ```    **Result**    ```   input: [[0.16826761 0.68168217 0.55196001]  [0.19735483 0.34837823 0.69015595]  [0.09448514 0.57390828 0.37097193]] output: [[0.16826761 0.68168217 0.55196001]  [0.19735483 0.34837823 0.69015595]  [0.09448514 0.57390828 0.37097193]]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`input` | (*Tensor*): input tensor to copy
*Outputs* | 
`output` | (*Tensor*): copy of input tensor


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## CopyCPUToGPU


Copy tensor for CPU to GPU context. Must be run under GPU device option.



### Interface


---------- | ----------
*Inputs* | 
`input` | The input tensor.
*Outputs* | 
`output` | Tensor that will contain a copy of the input.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## CopyGPUToCPU


Copy tensor for GPU to CPU context. Must be run under GPU device option.



### Interface


---------- | ----------
*Inputs* | 
`input` | The input tensor.
*Outputs* | 
`output` | Tensor that will contain a copy of the input.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## Cos


Calculates the cosine of the given input tensor, element-wise.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cos_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cos_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Cos",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.rand(5).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [0.6816719 

```
  0.76771533 0.933932   0.01404487 0.11862425]
```

 Y: [0.7765203 

```
  0.71949923 0.5946774  0.99990135 0.9929724 ]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor calculated as the cosine of the input tensor, element-wise.


### Code


[caffe2/operators/cos_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cos_op.cc)

---



## CosGradient

No documentation yet.


### Code


[caffe2/operators/cos_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cos_op.cc)

---



## Cosh


Calculates the hyperbolic cosine of the given input tensor, element-wise.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosh_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Cosh",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.rand(5).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [0.66423494 0.32074615 0.81523746 0.90423071 0.39275789] Y: [1.22883528 1.05188156 1.35112322 1.43744212 1.07812598]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The hyperbolic cosine values of the input tensor, computed element-wise


### Code


[caffe2/operators/cosh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosh_op.cc)

---



## CoshGradient

No documentation yet.


### Code


[caffe2/operators/cosh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosh_op.cc)

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


[caffe2/operators/cosine_embedding_criterion_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosine_embedding_criterion_op.cc)

---



## CosineEmbeddingCriterionGradient

No documentation yet.


### Code


[caffe2/operators/cosine_embedding_criterion_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cosine_embedding_criterion_op.cc)

---



## CosineSimilarity


This op takes two input float tensors of the same size, $X$ and $Y$, and produces one output float tensor , $Z$, calculated as the cosine similarity between $X$ and $Y$. Recall, the cosine similarity between two tensors $X$ and $Y$ is defined as:  $$\mathbf{Z}=CosineSimilarity(\mathbf{X},\mathbf{Y}) = \frac{\mathbf{X}\cdot\mathbf{Y}}{\|\mathbf{X}\|\|\mathbf{Y}\|} = \frac{\sum_n^{i=1}X_iY_i}{\sqrt{\sum_n^{i=1}X_i^2}\sqrt{\sum_n^{i=1}Y_i^2}}$$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "CosineSimilarity",
    ["X", "Y"],
    ["Z"]
```

 )  # Create X X = np.random.randn(3, 3) print("X:\n",X)  # Create Y Y = np.random.randn(3, 3) print("Y:\n",Y)  # Feed X & Y into workspace workspace.FeedBlob("X", X.astype(np.float32)) workspace.FeedBlob("Y", Y.astype(np.float32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Z:\n", workspace.FetchBlob("Z"))   ```    **Result**    ```   X:  [[-0.42635564 -0.23831588 -0.25515547]  [ 1.43914719 -1.05613228 

```
  1.01717373]
```

  [ 0.06883105 

```
  0.33386519 -1.46648334]]
```

 Y:  [[-0.90648691 -0.14241514 -1.1070837 ]  [ 0.92152729 -0.28115511 -0.17756722]  [-0.88394254 

```
  1.34654037 -0.80080998]]
```

 Z:  [-1.7849885e-23 

```
  1.7849885e-23 -1.0842022e-07]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | 1D or 2D input tensor
`Y` | 1D or 2D input tensor (must have the same shape as X)
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## CosineSimilarityGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## CountDown


If the internal count value > 0, decreases count value by 1 and outputs False, otherwise outputs True.
  

```
  Github Links:
  - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc


```

 <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  createcounter_op = core.CreateOperator(  

```
    "CreateCounter",
    [],
    ["counter"],
    init_count=5
```

 )  retrievecount_op = core.CreateOperator(  

```
    "RetrieveCount",
    ["counter"],
    ["count"]
```

 )  checkcounterdone_op = core.CreateOperator(  

```
    "CheckCounterDone",
    ["counter"],
    ["done"]
```

 )  countup_op = core.CreateOperator(  

```
    "CountUp",
    ["counter"],
    ["previous_count"],
```

 )  countdown_op = core.CreateOperator(  

```
    "CountDown",
    ["counter"],
    ["done"],
```

 )  resetcounter_op = core.CreateOperator(  

```
    "ResetCounter",
    ["counter"],
    ["previous_count"],
    init_count=3
```

 )   # Create counter workspace.RunOperatorOnce(createcounter_op) print("'counter' pointer:", workspace.FetchBlob("counter"))   # Retrieve initial counter value workspace.RunOperatorOnce(retrievecount_op) print("Initial 'count':", workspace.FetchBlob("count"))   # Check if counter is done workspace.RunOperatorOnce(checkcounterdone_op) print("Initial 'done' value:", workspace.FetchBlob("done"))   # Test CountUp operator print("\nTesting CountUp operator...") for i in range(5):  

```
    workspace.RunOperatorOnce(countup_op)
    print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

```

 workspace.RunOperatorOnce(retrievecount_op) print("'count' value after CountUp test:", workspace.FetchBlob("count"))   # Test CountDown operator print("\nTesting CountDown operator...") for i in range(11):  

```
    workspace.RunOperatorOnce(countdown_op)
    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))
```

 ```    **Result**   ``` 'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
Initial 'count': 5 Initial 'done' value: False  Testing CountUp operator...
'previous_count' after CountUp: 5 'previous_count' after CountUp: 6 'previous_count' after CountUp: 7 'previous_count' after CountUp: 8 'previous_count' after CountUp: 9 'count' value after CountUp test: 10  Testing CountDown operator...
'count' value after CountDown: 9	'done' value: False 'count' value after CountDown: 8	'done' value: False 'count' value after CountDown: 7	'done' value: False 'count' value after CountDown: 6	'done' value: False 'count' value after CountDown: 5	'done' value: False 'count' value after CountDown: 4	'done' value: False 'count' value after CountDown: 3	'done' value: False 'count' value after CountDown: 2	'done' value: False 'count' value after CountDown: 1	'done' value: False 'count' value after CountDown: 0	'done' value: False 'count' value after CountDown: -1	'done' value: True ```  </details>  


### Interface


---------- | ----------
*Inputs* | 
`counter` | *(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.
*Outputs* | 
`done` | *(type: bool)* False unless the internal count is zero.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc)

---



## CountUp


Increases count value by 1 and outputs the previous value atomically.
  

```
  Github Links:
  - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc


```

 <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  createcounter_op = core.CreateOperator(  

```
    "CreateCounter",
    [],
    ["counter"],
    init_count=5
```

 )  retrievecount_op = core.CreateOperator(  

```
    "RetrieveCount",
    ["counter"],
    ["count"]
```

 )  checkcounterdone_op = core.CreateOperator(  

```
    "CheckCounterDone",
    ["counter"],
    ["done"]
```

 )  countup_op = core.CreateOperator(  

```
    "CountUp",
    ["counter"],
    ["previous_count"],
```

 )  countdown_op = core.CreateOperator(  

```
    "CountDown",
    ["counter"],
    ["done"],
```

 )  resetcounter_op = core.CreateOperator(  

```
    "ResetCounter",
    ["counter"],
    ["previous_count"],
    init_count=3
```

 )   # Create counter workspace.RunOperatorOnce(createcounter_op) print("'counter' pointer:", workspace.FetchBlob("counter"))   # Retrieve initial counter value workspace.RunOperatorOnce(retrievecount_op) print("Initial 'count':", workspace.FetchBlob("count"))   # Check if counter is done workspace.RunOperatorOnce(checkcounterdone_op) print("Initial 'done' value:", workspace.FetchBlob("done"))   # Test CountUp operator print("\nTesting CountUp operator...") for i in range(5):  

```
    workspace.RunOperatorOnce(countup_op)
    print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

```

 workspace.RunOperatorOnce(retrievecount_op) print("'count' value after CountUp test:", workspace.FetchBlob("count"))   # Test CountDown operator print("\nTesting CountDown operator...") for i in range(11):  

```
    workspace.RunOperatorOnce(countdown_op)
    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))
```

 ```    **Result**   ``` 'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
Initial 'count': 5 Initial 'done' value: False  Testing CountUp operator...
'previous_count' after CountUp: 5 'previous_count' after CountUp: 6 'previous_count' after CountUp: 7 'previous_count' after CountUp: 8 'previous_count' after CountUp: 9 'count' value after CountUp test: 10  Testing CountDown operator...
'count' value after CountDown: 9	'done' value: False 'count' value after CountDown: 8	'done' value: False 'count' value after CountDown: 7	'done' value: False 'count' value after CountDown: 6	'done' value: False 'count' value after CountDown: 5	'done' value: False 'count' value after CountDown: 4	'done' value: False 'count' value after CountDown: 3	'done' value: False 'count' value after CountDown: 2	'done' value: False 'count' value after CountDown: 1	'done' value: False 'count' value after CountDown: 0	'done' value: False 'count' value after CountDown: -1	'done' value: True ```  </details>  


### Interface


---------- | ----------
*Inputs* | 
`counter` | *(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.
*Outputs* | 
`previous_count` | *(type: int)* Count value BEFORE this operation.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc)

---



## CreateAtomicBool

Create an unique_ptr blob to hold an atomic<bool>


### Interface


---------- | ----------
*Outputs* | 
`atomic_bool` | Blob containing a unique_ptr<atomic<bool>>


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/atomic_ops.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

---



## CreateCounter


Creates a count-down counter with initial value specified by the  `init_count`  argument.
   

```
  Github Links:
  - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc


```

 <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  createcounter_op = core.CreateOperator(  

```
    "CreateCounter",
    [],
    ["counter"],
    init_count=5
```

 )  retrievecount_op = core.CreateOperator(  

```
    "RetrieveCount",
    ["counter"],
    ["count"]
```

 )  checkcounterdone_op = core.CreateOperator(  

```
    "CheckCounterDone",
    ["counter"],
    ["done"]
```

 )  countup_op = core.CreateOperator(  

```
    "CountUp",
    ["counter"],
    ["previous_count"],
```

 )  countdown_op = core.CreateOperator(  

```
    "CountDown",
    ["counter"],
    ["done"],
```

 )  resetcounter_op = core.CreateOperator(  

```
    "ResetCounter",
    ["counter"],
    ["previous_count"],
    init_count=3
```

 )   # Create counter workspace.RunOperatorOnce(createcounter_op) print("'counter' pointer:", workspace.FetchBlob("counter"))   # Retrieve initial counter value workspace.RunOperatorOnce(retrievecount_op) print("Initial 'count':", workspace.FetchBlob("count"))   # Check if counter is done workspace.RunOperatorOnce(checkcounterdone_op) print("Initial 'done' value:", workspace.FetchBlob("done"))   # Test CountUp operator print("\nTesting CountUp operator...") for i in range(5):  

```
    workspace.RunOperatorOnce(countup_op)
    print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

```

 workspace.RunOperatorOnce(retrievecount_op) print("'count' value after CountUp test:", workspace.FetchBlob("count"))   # Test CountDown operator print("\nTesting CountDown operator...") for i in range(11):  

```
    workspace.RunOperatorOnce(countdown_op)
    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))
```

 ```    **Result**   ``` 'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
Initial 'count': 5 Initial 'done' value: False  Testing CountUp operator...
'previous_count' after CountUp: 5 'previous_count' after CountUp: 6 'previous_count' after CountUp: 7 'previous_count' after CountUp: 8 'previous_count' after CountUp: 9 'count' value after CountUp test: 10  Testing CountDown operator...
'count' value after CountDown: 9	'done' value: False 'count' value after CountDown: 8	'done' value: False 'count' value after CountDown: 7	'done' value: False 'count' value after CountDown: 6	'done' value: False 'count' value after CountDown: 5	'done' value: False 'count' value after CountDown: 4	'done' value: False 'count' value after CountDown: 3	'done' value: False 'count' value after CountDown: 2	'done' value: False 'count' value after CountDown: 1	'done' value: False 'count' value after CountDown: 0	'done' value: False 'count' value after CountDown: -1	'done' value: True ```  </details>  


### Interface


---------- | ----------
*Arguments* | 
`init_count` | *(type: int; default: 0)* Initial count for the counter, must be >= 0.
*Outputs* | 
`counter` | *(type: Tensor`<ptr>`)* A blob pointing to an instance of a new counter.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc)

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


[caffe2/operators/map_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/map_ops.cc)

---



## CreateMutex

Creates an unlocked mutex and returns it in a unique_ptr blob.


### Interface


---------- | ----------
*Outputs* | 
`mutex_ptr` | Blob containing a std::unique_ptr<mutex>.


### Code


[caffe2/operators/atomic_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/atomic_ops.cc)

---



## CreateScope


'CreateScope' operator initializes and outputs empty scope that is used by Do operator to store local blobs     


### Code


[caffe2/operators/create_scope_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/create_scope_op.cc)

---



## CreateTensorVector

Create a std::unique_ptr<std::vector<Tensor> >


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/text_file_reader.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/text_file_reader.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## CrossEntropy


This operator computes the cross entropy between a $NxD$ dimensional input data tensor $X$ 

```
  and a $NxD$ dimensional input label tensor $label$. The op produces a single length $N$ output tensor $Y$. Here, $N$ is considered the batch size and $D$ is the size of each element in the batch. In practice, it is most commonly used at the end of models as a part of the loss computation, after the SoftMax operator and before the AveragedLoss operator. The cross entropy operation is defined as follows

```

 $$Y_i = \sum_j (label_{ij} * log(X_{ij}))$$  where ($i$, $j$) is the classifier's prediction of the $j$th class (the correct one), and $i$ is the batch size. Each log has a lower limit for numerical stability.
 Github Links: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "CrossEntropy",
    ["X", "label"],
    ["Y"]
```

 )  # Create X: Sample softmax output for 5-class model X = np.array([[.01, .05, .02, .02, .9],[.03, .1, .42, .05, .4]]) print("X:\n",X)  # Create label: Sample 1-hot ground truth label vectors label = np.array([[0.,0.,0.,0.,1.],[0.,0.,1.,0.,0.]]) print("label:\n",label)  # Feed X & label into workspace workspace.FeedBlob("X", X.astype(np.float32)) workspace.FeedBlob("label", label.astype(np.float32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[0.01 0.05 0.02 0.02 0.9 ]  [0.03 0.1 

```
  0.42 0.05 0.4 ]]
```

 label:  [[0. 0. 0. 0. 1.]  [0. 0. 1. 0. 0.]] Y:  [0.10536055 0.8675006 ]   ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input tensor which is almost always the result of a softmax operation. $X$ is a 2D array of size $NxD$, where $N$ is the batch size and $D$ is the number of classes.
`label` | Blob containing the labels used to compare the input. $label$ is the same shape as $X$.
*Outputs* | 
`Y` | Output blob from the cross entropy computation. $Y$ is 1D length $N$ tensor.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## CrossEntropyGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## Cube

No documentation yet.


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor calculated as the cube of the input tensor, element-wise.


### Code


[caffe2/operators/cube_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cube_op.cc)

---



## CubeGradient

No documentation yet.


### Code


[caffe2/operators/cube_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cube_op.cc)

---



## DBExists


Checks if the db described by the arguments exists.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "DBExists",
    [],
    ["exists"],
    db_name="test_db",
    db_type="leveldb",
```

 )  workspace.RunOperatorOnce(op) print("exists:", workspace.FetchBlob("exists"))   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`absolute_path` | *(type: int; default: 0)* If set to non-zero, save the db directly to the path specified by the `db` arg. If not set (default), prepend the path of the current root folder of the workspace to the path specified by the `db` arg.
`db_name` | *(type: string)* Path to the db in question; see the `absolute_path` arg details for options regarding the current root folder of the workspace.
`db_type` | *(type: string)* Type of db to save (options: "lmdb", "leveldb", "minidb").
*Outputs* | 
`exists` | *(type: Tensor`<bool>`)* Scalar boolean output tensor. True if the db exists, else false.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)

---



## DeformConv


Deformable convolution operator consumes an input vector, the kernel offsets blob, the filter blob and the bias blob and computes the output. Other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is convolved with a subset of the image using the deformed kernel as specified by offsets blob and the bias is added; this is done throughout the image data and the output is computed.
  


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints.
`offset` | Offsets blob that specifies the deformed shape of the kernel; consists of 2d offsets for each kernel element, one full set per each output element; therefore has size (N x 2*kH*kW x H' x W') where N is the batch size, kH and kW are the height and width of the kernel, H' and W' are the output blob dimensions.
`filter` | The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the convolution; has size (M).
*Outputs* | 
`Y` | Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/deform_conv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/deform_conv_op.cc)

---



## DeformConvGradient

No documentation yet.


### Code


[caffe2/operators/deform_conv_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/deform_conv_gradient_op.cc)

---



## DepthConcat

Backward compatible operator name for Concat.


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc)

---



## DepthSplit

Backward compatible operator name for Split.


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc)

---



## DestroyCommonWorld

Closes all connections managed by a common world.


### Interface


---------- | ----------
*Inputs* | 
`common_world` | The common world to be destroyed.


### Code


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

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


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

---



## Div


Performs element-wise binary division (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "Div",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([[18,8],[2,9]])) workspace.FeedBlob("B", np.array([[9,2],[3,2]])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [[18 

```
  8]
```

  [ 2 

```
  9]]
```

 B: [[9 2]  [3 2]] C: [[2 4]  [0 4]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting
`axis` | *(type: int; default: -1)* Axis to concatenate on.
*Inputs* | 
`A` | *(type: Tensor`<float>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size as A.
*Outputs* | 
`C` | *(type: Tensor`<float>`)* Output tensor with same dimensions and type as A.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## DivGradient

No documentation yet.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

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


[caffe2/operators/do_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/do_op.cc)

---



## DotProduct


Computes and outputs the dot product of the two input float tensors  `X`  and  `Y` .
Note that  `X`  and  `Y`  must be either 1D or 2D, and they must be the same shape.
The output tensor is 1D, which represents either the product of each element in a respective dimension if the inputs are 1D, or the sum of the products in a given dimension if the inputs are 2D matrices. Note that the actual dot product is a scalar value, which is effectively the sum of the elements in the 1D output tensor.
 For 1D inputs: Given two vectors $X = [x_0, x_1, x_2]$ and $Y = [y_0, y_1, y_2]$; $Z = [x_0  * y_0, x_1 *  y_1, x_2 * y_2]$  For 2D inputs: Given two matrices: $$X = [[x_0^0, x_1^0, x_2^0], \\ [x_0^1, x_1^1, x_2^1], \\ [x_0^2, x_1^2, x_2^2], \\ ..., \\ [x_0^n, x_1^n, x_2^n]]$$  and  $$Y = [[y_0^0, y_1^0, y_2^0], \\ [y_0^1, y_1^1, y_2^1], \\ [y_0^2, y_1^2, y_2^2], \\ ..., \\ [y_0^n, y_1^n, y_2^n]]$$  then  $$Z = 

```
  \biggl[\Big((x_0^0 * y_0^0) + (x_1^0 * y_1^0) + (x_2^0 * y_2^0)\Big), \\ \Big((x_0^1 * y_0^1) + (x_1^1 * y_1^1) + (x_2^1 * y_2^1)\Big), \\ \Big((x_0^2 * y_0^2) + (x_1^2 * y_1^2) + (x_2^2 * y_2^2)\Big), \\ ..., \\ \Big((x_0^n * y_0^n) + (x_1^n * y_1^n) + (x_2^n * y_2^n)\Big)\biggr]$$

```

 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "DotProduct",
    ["X",  "Y"],
    ["Z"]
```

 )  workspace.FeedBlob("X", np.random.randint(20, size=(5)).astype(np.float32)) workspace.FeedBlob("Y", np.random.randint(20, size=(5)).astype(np.float32)) print("X:\n", workspace.FetchBlob("X")) print("Y:\n", workspace.FetchBlob("Y")) workspace.RunOperatorOnce(op) print("Z:\n", workspace.FetchBlob("X"))   workspace.ResetWorkspace() workspace.FeedBlob("X", np.random.randint(10, size=(3,3)).astype(np.float32)) workspace.FeedBlob("Y", np.random.randint(10, size=(3,3)).astype(np.float32)) print("X:\n", workspace.FetchBlob("X")) print("Y:\n", workspace.FetchBlob("Y")) workspace.RunOperatorOnce(op) print("Z:\n", workspace.FetchBlob("Z"))   ```    **Result**    ```   X:  [ 2. 15. 

```
  2.  7. 12.]
```

 Y:  [ 3. 12. 

```
  9.  3. 18.]
```

 Z:  [ 2. 15. 

```
  2.  7. 12.]
```

 X:  [[2. 0. 4.]  [7. 7. 4.]  [7. 9. 9.]] Y:  [[2. 0. 8.]  [9. 6. 1.]  [7. 8. 0.]] Z:  [ 36. 109. 121.]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* 1D or 2D input tensor.
`Y` | *(type: Tensor`<float>`)* 1D or 2D input tensor (must have the same shape as X).
*Outputs* | 
`Z` | *(type: Tensor`<float>`)* 1D output tensor.


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## DotProductGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

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


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## DotProductWithPaddingGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## Dropout


  `Dropout`  takes one input data tensor ( `X` ) and produces two tensor outputs,  `Y`  and  `mask` . If the  `is_test`  argument is zero (default=0), the output  `Y`  will be the input with random elements zeroed. The probability that a given element is zeroed is determined by the  `ratio`  argument.
 If the  `is_test`  argument is set to non-zero, the output  `Y`  is exactly the same as the input  `X` . Note that outputs are scaled by a factor of $\frac{1}{1-ratio}$ during training, so that during test time, we can simply compute an identity function. This scaling is important because we want the output at test time to equal the expected value at training time. Dropout has been proven to be an effective regularization technique to prevent overfitting during training.
  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Dropout",
    ["X"],
    ["Y"] + ["mask"],
    ratio=0.5,
    is_test=0
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(5, 5)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y")) print("mask:", workspace.FetchBlob("mask"))  ```    **Result**    ```  X: [[5. 4. 3. 6. 9.]  [2. 1. 8. 0. 9.]  [7. 3. 0. 6. 3.]  [1. 8. 2. 6. 4.]  [6. 2. 6. 4. 0.]] Y: [[ 0. 

```
  0.  0. 12. 18.]
```

  [ 0. 

```
  0. 16.  0.  0.]
```

  [ 0. 

```
  0.  0. 12.  6.]
```

  [ 0. 

```
  0.  4.  0.  0.]
```

  [12. 

```
  0.  0.  0.  0.]]
```

 mask: [[False False False 

```
  True  True]
```

  [False False 

```
  True  True False]
```

  [False False 

```
  True  True  True]
```

  [False False 

```
  True False False]
```

  [ True False False False False]]  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`ratio` | *(type: float; default: 0.5)* Probability of an element to be zeroed.
`is_test` | *(type: int; default: 0)* If zero (train mode), perform dropout. If non-zero(test mode), Y = X.
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.
`mask` | *(type: Tensor`<bool>`)* The output mask containing boolean values foreach element, signifying which elements are dropped out. If `is_test` isnonzero, this output is not filled.


### Code


[caffe2/operators/dropout_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.cc)

---



## DropoutGrad

No documentation yet.


### Code


[caffe2/operators/dropout_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.cc)

---



## EQ


Performs element-wise equal to comparison  **==**  (with limited broadcast support).
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "EQ",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3])) workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```  A: [ 1 

```
  5  2  9 12  3]
```

 B: [ 1 

```
  3  4  9 12  8]
```

 C: [ True False False 

```
  True  True False]
```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## ElementwiseLinear


This op computes the elementwise linear combination of a batch of input vectors with a weight vector and bias vector. As input, the op takes an input tensor $X$ of shape $NxD$, a weight vector $w$ of length $D$, and a bias vector $b$ of length $D$. Here, $N$ represents the batch size and $D$ represents the length of the feature vectors. The output, $Y$, is a tensor of shape $NxD$ and is calculated as  $$Y_{ij} = X_{ij}w_j + b_j \ for \ i\in{N}, j\in{D}$$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ElementwiseLinear",
    ["X", "w", "b"],
    ["Y"]
```

 )  # Create X X = np.array([[1,2,3,4,5],[6,8,9,16,10]]) print("X:\n",X)  # Create w w = np.array([1,1/2.,1/3.,1/4.,1/5.]) print("w:\n",w)  # Create b b = np.array([1.,1.,1.,1.,1.]) print("b:\n",b)   # Feed X & w & b into workspace workspace.FeedBlob("X", X.astype(np.float32)) workspace.FeedBlob("w", w.astype(np.float32)) workspace.FeedBlob("b", b.astype(np.float32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[ 1 

```
  2  3  4  5]
```

  [ 6 

```
  8  9 16 10]]
```

 w:  [1. 

```
  0.5  0.33333333 0.25  0.2]
```

 b:  [1. 1. 1. 1. 1.] Y:  [[2. 2. 2. 2. 2.]  [7. 5. 4. 5. 3.]]   ```   </details>    


### Interface


---------- | ----------
*Arguments* | 
`axis` | *(type: int; default: 1)* Describes the axis of the inputs; defaults to one because the 0th axis most likely describes the batch size.
*Inputs* | 
`X` | 2D input tensor of size $NxD$. This input represents the input data to be operated on.
`w` | 1D scaling factors, or weights, of size $D$. This input contains the weights that will be multiplied by the data.
`b` | 1D biases of size $D$. This input contains the biases that will be added to the products of the weights and data.
*Outputs* | 
`Y` | 2D output tensor of size $NxD$. Calculated as described above.


### Code


[caffe2/operators/elementwise_linear_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.cc)

---



## ElementwiseLinearGradient

No documentation yet.


### Code


[caffe2/operators/elementwise_linear_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_linear_op.cc)

---



## Elu


 This op implements the exponential linear unit (ELU) activation function as described in [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)]( [https://arxiv.org/abs/1511.07289).](https://arxiv.org/abs/1511.07289).)  The op takes an input tensor $X$ of arbitrary shape, computes the elementwise elu operation, and returns a vector $Y$ of the same shape as output. The alpha parameter may be passed as an argument, but defaults to 1. The elu operation is defined as  $$y=f(x) =\begin{cases}\alpha(e^x-1) & x < 0 \\ x & otherwise\end{cases}$$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Elu",
    ["X"],
    ["Y"],
    alpha=1.1
```

 )  workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32)) print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[ 0.35339102 

```
  1.1860217  -0.10710736]
```

  [-3.1173866 

```
  -0.1889988  -0.20330353]
```

  [ 1.8525308 

```
  -0.368949    0.506277  ]]

```

 Y:  [[ 0.35339102 

```
  1.1860217  -0.11172786]
```

  [-1.0513  

```
    -0.18943374 -0.20236646]
```

  [ 1.8525308 

```
  -0.33939326  0.506277  ]]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`alpha` | *(type: float; default: 1.0)* Defines alpha parameter used in calculation.
*Inputs* | 
`X` | 1D input tensor of data to be operated on.
*Outputs* | 
`Y` | 1D input tensor, calculated as described above.


### Code


[caffe2/operators/elu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.cc)

---



## EluGradient


EluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.



### Code


[caffe2/operators/elu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.cc)

---



## EnforceFinite


Raise if there is NaN or Inf values in the input tensor.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor


### Code


[caffe2/operators/enforce_finite_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/enforce_finite_op.cc)

---



## EnsureCPUOutput


This Op always create TensorCPU output, and may involves cross-device MemCpy.
Under CPU Context, this Op takes TensorCPU as input. Under the CUDA Context, this Op accepts either CUDA or CPU Tensor input.



### Interface


---------- | ----------
*Inputs* | 
`input` | The input CUDA or CPU tensor.
*Outputs* | 
`output` | TensorCPU that is a copy of the input.


### Code


[caffe2/operators/ensure_cpu_output_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ensure_cpu_output_op.cc)

---



## EnsureClipped


Given a tensor, apply clip after gradient is applied; when the param is sparse as indicated by valid indices and grad, in-place is required 


### Interface


---------- | ----------
*Inputs* | 
`param` | Parameters to be normalized
`indices` | Sparse indices, only needed for sparse param
`grad` | Gradient computed, only needed for sparse param
*Outputs* | 
`output_param` | param ensured to be clipped within range


### Code


[caffe2/operators/ensure_clipped_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ensure_clipped_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## Exp


Calculates the exponential of the given input tensor ($exp(x)$), element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/exp_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/exp_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Exp",
    ["X"],
    ["X"],
```

 )  workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32)) print("X before running op:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("X after running op:", workspace.FetchBlob("X"))   ```    **Result**    ```   X before running op: [[0.5821691 

```
  0.07719802 0.50159824]
```

  [0.40952456 0.36788362 0.84887683]  [0.02472685 0.65730894 0.9066397 ]] X after running op: [[1.7899168 1.080256 

```
  1.6513585]
```

  [1.5061016 1.4446739 2.3370204]  [1.0250351 1.9295927 2.4759884]]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* The exponential of the input tensor computed element-wise.


### Code


[caffe2/operators/exp_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/exp_op.cc)

---



## Expand


	Broadcast the input tensor to a materialized new tensor using given shape.
	Broadcast rule is similar to "numpy.array(input) * numpy.ones(shape)": 	Dimensions are right alignment; 	Two corresponding dimensions must have the same value, or one of them 	equals to 1.
 

```
        In order to align with PyTorch's `expand`, `shape` is allowed to have entries
        equal to -1, which means to preserve the size of the corresponding dimension
        in `X` (so it's actually equivalent to equal to 1).
```




### Interface


---------- | ----------
*Inputs* | 
`X` | (*Tensor`<NumericType>`*): input tensor
`shape` | (*Tensor`<int>`*): expand shape
*Outputs* | 
`Y` | (*Tensor`<NumericType>`*): expanded tensor


### Code


[caffe2/operators/expand_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_op.cc)

---



## ExpandDims


The  *ExpandDims*  op inserts single-dimensional entries into the shape of the input tensor  *data,*  and produces a single output tensor  *expanded* . The op also takes an argument  *dims*  with a list of dimensions for where to add the single dimensional entries. If the same blob is provided as input and output, the operation is copy-free. This is the exact inverse operation of  *Squeeze* .
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ExpandDims",
    ["data"],
    ["expanded"],
    dims=[0,1],
```

 )  workspace.FeedBlob("data", np.zeros((100,100)).astype(np.float32)) print("data.shape:", workspace.FetchBlob("data").shape)  workspace.RunOperatorOnce(op) print("expanded.shape:", workspace.FetchBlob("expanded").shape)   ```    **Result**    ```   data.shape: (100, 100) expanded.shape: (1, 1, 100, 100)   ```   </details>    


### Interface


---------- | ----------
*Arguments* | 
`dims` | *(type: [int])* List of dimensions of *data* to add single dimensional entry.
*Inputs* | 
`data` | Input tensor of data to be operated on.
*Outputs* | 
`expanded` | Reshaped tensor with same data as input.


### Code


[caffe2/operators/expand_squeeze_dims_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc)

---



## ExpandGradient

No documentation yet.


### Code


[caffe2/operators/expand_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_op.cc)

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


[caffe2/operators/extend_tensor_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/extend_tensor_op.cc)

---



## FC


The FC operator computes an output $(Y)$ as a linear combination of the input data blob $(X)$ with a weight blob $(W)$ and bias blob $(b)$. More formally,  $$Y = XW^T+b$$  Here, $X$ is a matrix of shape $(M,K)$, $W$ is a matrix of shape $(N,K)$, $b$ is a vector of length $N$, and $Y$ is a matrix of shape $(M,N)$. $N$ can be thought of as the number of nodes in the layer, $M$ is the batch size, and $K$ is the number of features in an input observation.
  *NOTE: $X$ does not need to explicitly be a 2-dimensional matrix, however, if it is not it will be coerced into one. For an arbitrary $n$-dimensional tensor $X$, e.g. $[a_0, a_1, \ldots ,a_{k-1}, a_k, \ldots , a_{n-1}]$, where $a_i$ in $N$, and $k$ is the $axis$ arg provided, then $X$ will be coerced into a 2-dimensional tensor with dimensions $[a_0 *  \ldots  * a_{k-1}, a_k *  \ldots  * a_{n-1}]$. For the default case where axis=1, this means the $X$ tensor will be coerced into a 2D tensor of dimensions $[a_0, a_1 *  \ldots  * a_{n-1}]$, where $a_0$ is often the batch size. In this situation, we must have $a_0 = M$ and $a_1 *  \ldots  * a_{n-1} = K$. Lastly, even though $b$ is a vector of length $N$, it is copied and resized to shape $(M x N)$ implicitly, then added to each vector in the batch.*   Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   # In this example, our batch size is 1 (M=1), the input observation will have #  

```
  6 features (K=6), and the layer will have one hidden node (N=1). The
```

 #  

```
  expected output is Y=7.
```

 workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "FC",
    ["X", "W", "b"],
    ["Y"]
```

 )  # Create X: MxK data = np.array([1,2,3,4,5,6]).astype(np.float32) data = data[np.newaxis,:]  # Create W: NxK weights = np.array(np.array([1,1/2.,1/3.,1/4.,1/5.,1/6.])).astype(np.float32) weights = weights[np.newaxis,:]  # Create b: N bias = np.array([1.]).astype(np.float32)  # Put the inputs into the workspace workspace.FeedBlob("X", data) workspace.FeedBlob("W", weights) workspace.FeedBlob("b", bias)  # Run the operator workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   Y:  [[7.]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`axis` | *(type: int; default: 1)* Describes the axis of the input data $X$. Defaults to one because in the common case when the input $X$ has shape $(M,K)$, the first axis encodes the batch size.
`axis_w` | *(type: int; default: 1)* Describes the axis of the input weight matrix $W$. Defaults to one because the first axis most likely describes the batch_size.
`float16_compute` | *(type: bool; default: False)* Whether to use float-16 compute kernel.
*Inputs* | 
`X` | Input blob to be coerced into a 2D matrix of shape $(M,K)$, where $M$ is the batch size and $K$ is the number of features in a single observation.
`W` | Input blob to be coerced into a 2D matrix of shape $(N,K)$ describing a fully connected weight matrix. Here, $K$ is the number of features in a single observation and $N$ is the number of nodes in the FC layer.
`b` | Input blob containing vector of length $N$ which describes one bias for each node in the layer.
*Outputs* | 
`Y` | Ouput blob containing a 2D output matrix of shape $(M,N)$, where $M$ is the batch size and $N$ is the number of nodes in the layer. The ouput is calculated as $Y=XW^T+b$.


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc)

---



## FCGradient

No documentation yet.


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc)

---



## FCTransposed


Same as FC, but weight matrix is supposed to be already pretransposed.
FCTransposed stands for calling blass with no noTrans, noTrans 


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc)

---



## FCTransposedGradient

No documentation yet.


### Code


[caffe2/operators/fully_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fully_connected_op.cc)

---



## Fail

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## FeedBlob


FeedBlobs the content of the blobs. The input and output blobs should be one-to-one inplace.


### Interface


---------- | ----------
*Arguments* | 
`value` | (string) if provided then we will use this string as the value for theprovided output tensor


### Code


[caffe2/operators/feed_blob_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feed_blob_op.cc)

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


[caffe2/operators/find_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/find_op.cc)

---



## FindDuplicateElements


The  *FindDuplicateElements*  op takes a single 1-D tensor  *data*  as input and returns a single 1-D output tensor  *indices* . The output tensor contains the indices of the duplicate elements of the input, excluding the first occurrences. If all elements of  *data*  are unique,  *indices*  will be empty.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/find_duplicate_elements_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "FindDuplicateElements",
    ["data"],
    ["indices"],
```

 )  workspace.FeedBlob("data", np.array([8,2,1,1,7,8,1]).astype(np.float32)) print("data:\n", workspace.FetchBlob("data"))  workspace.RunOperatorOnce(op) print("indices: \n", workspace.FetchBlob("indices"))   ```    **Result**    ```   data:  [8. 2. 1. 1. 7. 8. 1.] indices:  [3 5 6]   ```   </details>     


### Interface


---------- | ----------
*Inputs* | 
`data` | a 1-D tensor.
*Outputs* | 
`indices` | Indices of duplicate elements in data, excluding first occurrences.


### Code


[caffe2/operators/find_duplicate_elements_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/find_duplicate_elements_op.cc)

---



## Flatten


Flattens the input tensor into a 2D matrix. If input tensor has shape $(d_0, d_1, ..., d_n)$ then the output will have shape $\bigl((d_0  * d_1 *  ...  * d_{(axis-1)}), (d_{axis} *  d_{(axis+1)}  * ... *  d_n)\bigr)$.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flatten_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flatten_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Flatten",
    ["X"],
    ["Y"],
    axis=1
```

 )  workspace.FeedBlob("X", np.random.rand(1,3,2,2)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))  ```    **Result**    ```  X: [[[[0.53432311 0.23734561]   

```
  [0.56481598 0.52152617]]

  [[0.33662627 0.32472711]
```

   

```
  [0.17939016 0.97175851]]

  [[0.87226421 0.49045439]
```

   

```
  [0.92470531 0.30935077]]]]
```

 Y: [[0.53432311 0.23734561 0.56481598 0.52152617 0.33662627 0.32472711  

```
  0.17939016 0.97175851 0.87226421 0.49045439 0.92470531 0.30935077]]
```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`axis` | *(type: int; default: 1)* Indicates up to which input dimensions (exclusive) should be flattened to the outer dimension of the output.
*Inputs* | 
`X` | *(type: Tensor)* Input Tensor of rank >= axis.
*Outputs* | 
`Y` | *(type: Tensor)* A 2D tensor with the contents of the input tensor, with input dimensions up to `axis` flattened to the outer dimension of the output and the remaining input dimensions flattened into the inner dimension of the output.


### Code


[caffe2/operators/flatten_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flatten_op.cc)

---



## FlattenToVec


 The  *FlattenToVec*  op flattens the input tensor into a 1-D vector. The op accepts a single input tensor and returns a single output tensor.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "FlattenToVec",
    ["input"],
    ["output"],
```

 )  workspace.FeedBlob("input", np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]).astype(np.float32)) print("input:\n", workspace.FetchBlob("input"))  workspace.RunOperatorOnce(op) print("output: \n", workspace.FetchBlob("output"))   ```    **Result**    ```   input:  [[ 1. 

```
  2.  3.]
```

  [ 4. 

```
  5.  6.]
```

  [ 7. 

```
  8.  9.]
```

  [10. 11. 12.]] output:  [ 1. 

```
  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`input` | A tensor of rank >= 1.
*Outputs* | 
`output` | A tensor of rank 1 (vector) with the contents of the input tensor.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/flexible_top_k.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flexible_top_k.cc)

---



## FlexibleTopKGradient

No documentation yet.


### Code


[caffe2/operators/flexible_top_k.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/flexible_top_k.cc)

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


[caffe2/operators/fused_rowwise_8bit_conversion_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fused_rowwise_8bit_conversion_ops.cc)

---



## FloatToFusedRandRowwiseQuantized


Applies row-wise stochastic/random quantization by determining the range of each row in the input matrix, and then quantize each element to one of two closest discrete levels by randomly drawing Bernoulli distribution.
The method is extended from TernGrad [1], which randomly quantizes gradients to three levels to reduce communication in distributed training.
The format of each row (x) in the output matrix is [bitwidth][tail][min][max][data]: bitwidth[1 Byte]: bitwidth per data [1, 2, 4 or 8]; tail[1 Byte]: the number of unused buckets [1-8] (One byte is split to 8/bitwidth buckets and each bucket stores one low-precision data in bitwidth bits); min[4 Bytes]: the minimum floating value min(x); max[4 Bytes]: the maximum floating value max(x); data: quantized data.
The quantization is uniform with levels q = min + (max-min)/(2^bitwidth - 1)*[0:1:2^bitwidth].
During stochastic/random quantization x'=Quantize(x), for q_j < x_i <= q_{j+1}, we draw quantization x'_i from Bernoulli distributions with P(x'_i = q_{j+1}) = (x_i - q_j)/(q_{j+1} - q_j), and P(x'_i = q_j) = (q_{j+1} - x_i)/(q_{j+1} - q_j) where x'_i is the quantized value of x_i.
[1] proved E{x'_i}=x_i, which is an unbiased approximation. More details are in the paper.
For example, suppose targeted bitwidth = 2 and x = [0.3, -1.4, -0.6, 0.9, 1.0], then tail = 3, min = -1.4, max = 1.0 and q = [-1.4, -0.6, 0.2, 1.0].
x_1 = 0.3 will be quantized to x'_1 = 0.2 with probability 7/8 and to x'_1 = 1.0 with probability 1/8.
The storage format of quantized data is: [x'_1|x'_3|x'_5|xxx]-[x'_2|x'_4|xxx|xxx].
In general, a input row is split to multiple segments. One segment is a continuous subarray of the row, and its length is the number of bytes storing quantized data in the output matrix.
The b-th bucket of the i-th byte stores the i-th data of the b-th segment of input row.
 [1] Wen, Wei, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li.
"Terngrad: Ternary gradients to reduce communication in distributed deep learning." In Advances in Neural Information Processing Systems, pp. 1508-1518. 2017.
 


### Interface


---------- | ----------
*Arguments* | 
`bitwidth` | How many bits to quantiz per data (defaults to 8).
`random` | random or not (True). False is set up for unittest.
*Inputs* | 
`input` | Float32 input data
*Outputs* | 
`output` | Fused bitwidth, tail, min, max and quantized data


### Code


[caffe2/operators/fused_rowwise_random_quantization_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fused_rowwise_random_quantization_ops.cc)

---



## FloatToHalf

No documentation yet.


### Code


[caffe2/operators/half_float_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/half_float_ops.cc)

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


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## Floor


Element-wise application of the floor function ($y=floor(x)$) to the input tensor  `X` . Output tensor shape is the same as the input tensor. This operator can be used in an in-place fashion by using the same input blob as the output blob.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/floor_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/floor_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Floor",
    ["X"],
    ["X"],
```

 )  workspace.FeedBlob("X", (np.random.uniform(-10, 10, (5,5))).astype(np.float32)) print("X before running op:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("X after running op:", workspace.FetchBlob("X"))   ```    **Result**    ```   X before running op: [[ 3.813361  

```
  -1.319647    5.2089314  -4.931328    0.6218652 ]
```

  [ 7.2757645  

```
  5.5552588   5.785643   -2.4790506  -0.41400087]
```

  [ 1.1541046 

```
  -6.933266    3.3754056   1.6569928  -1.7670316 ]
```

  [-3.4932013  

```
  4.891472    1.5530115  -3.2443287  -4.605099  ]
```

  [-4.574543  

```
  -7.360948    5.91305    -8.196495   -5.357458  ]]
```

 X after running op: [[ 3. -2. 

```
  5. -5.  0.]
```

  [ 7. 

```
  5.  5. -3. -1.]
```

  [ 1. -7. 

```
  3.  1. -2.]
```

  [-4. 

```
  4.  1. -4. -5.]
```

  [-5. -8. 

```
  5. -9. -6.]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.


### Code


[caffe2/operators/floor_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/floor_op.cc)

---



## Free


Frees the content of the blobs. The input and output blobs should be one-to-one inplace.


### Code


[caffe2/operators/free_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/free_op.cc)

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


[caffe2/operators/fused_rowwise_8bit_conversion_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fused_rowwise_8bit_conversion_ops.cc)

---



## FusedRandRowwiseQuantizedToFloat


De-quantizes the result of the FloatToFusedRandRowwiseQuantized operator.
Refer FloatToFusedRandRowwiseQuantized operator for details.



### Interface


---------- | ----------
*Inputs* | 
`quantized_input` | Fused bitwidth, tail, min, max and quantized data
*Outputs* | 
`float_input` | Float32 data


### Code


[caffe2/operators/fused_rowwise_random_quantization_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/fused_rowwise_random_quantization_ops.cc)

---



## GE


Performs element-wise greater or equal than comparison  **>=**  (with limited broadcast support).
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "GE",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3])) workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [ 1 

```
  5  2  9 12  3]
```

 B: [ 1 

```
  3  4  9 12  8]
```

 C: [ True 

```
  True False  True  True False]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

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


[caffe2/operators/gru_unit_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/gru_unit_op.cc)

---



## GRUUnitGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`sequence_lengths` | When false, the sequence lengths input is left out, and all following inputs are shifted left by one.


### Code


[caffe2/operators/gru_unit_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/gru_unit_op.cc)

---



## GT


Performs element-wise greater than comparison  **>**  (with limited broadcast support).
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "GT",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3])) workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [ 1 

```
  5  2  9 12  3]
```

 B: [ 1 

```
  3  4  9 12  8]
```

 C: [False 

```
  True False False False False]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## Gather


 The  *Gather*  op accepts a  *DATA*  tensor of rank $r >= 1$ and  *INDICES*  tensor of rank $q$ as inputs. It then gathers entries of the outer-most dimension of  *DATA* , indexed by  *INDICES* , and concatenate them in an output tensor of rank $q + (r - 1)$.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.cc)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Gather",
    ["DATA", "INDICES"],
    ["OUTPUT"]
```

 ) data = np.array([[1., 1.2],[2.3, 3.4],[4.5, 5.7]]) print("DATA:\n",data)  inds = np.array([[0, 1],[1, 2]]) print("INDICES:\n",inds)  # Feed X into workspace workspace.FeedBlob("DATA", data.astype(np.float32)) workspace.FeedBlob("INDICES", inds.astype(np.int32))  workspace.RunOperatorOnce(op) print("OUTPUT:\n", workspace.FetchBlob("OUTPUT"))   ```    **Result**    ```   DATA:  [[1. 

```
  1.2]
```

  [2.3 3.4]  [4.5 5.7]] INDICES:  [[0 1]  [1 2]] OUTPUT:  [[[1. 

```
  1.2]
  [2.3 3.4]]

```

  [[2.3 3.4]  

```
  [4.5 5.7]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input data tensor of rank $r>=1$
`INDICES` | Input indices tensor of rank $q$. This tensor must contain integers.
*Outputs* | 
`OUTPUT` | Output tensor of rank $q+(r-1)$


### Code


[caffe2/operators/gather_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/gather_op.cc)

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


[caffe2/operators/partition_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/partition_ops.cc)

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


[caffe2/operators/gather_fused_8bit_rowwise_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/gather_fused_8bit_rowwise_op.cc)

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


[caffe2/operators/sequence_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/gather_ranges_to_dense_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/gather_ranges_to_dense_op.cc)

---



## GaussianFill


This op fills an output tensor with samples drawn from a normal distribution specified by the mean and standard deviation arguments. The output tensor shape is specified by the  *shape*  argument. However, if  *input_as_shape*  is set to  *true* , then the  *input*  should be a 1D tensor containing the desired output shape (the dimensions specified in  *extra_shape*  will also be appended). In this case, the  *shape*  argument should  **not**  be set.
  *Note: cannot set the shape argument and pass in an input at the same time.*   Github Links: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "GaussianFill",
    [],
    ["out"],
    shape=[3,3],
    mean=2.0,
    std=1.1
```

 )  workspace.RunOperatorOnce(op) print("Out:\n", workspace.FetchBlob("out"))   ```    **Result**    ```   Out:  [[1.2084167 

```
  2.3336504  2.827349  ]
```

  [2.7108908 

```
  0.9374752  1.7173369 ]
```

  [0.03320992 2.1775863 

```
  1.0894578 ]]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`mean` | *(type: float; default: 0.)* Mean of the distribution to draw from.
`std` | *(type: float; default: 1.)* Standard deviation of the distribution to draw from.
`shape` | *(type: [int])* Desired shape of the *output* tensor.
`extra_shape` | *(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob.
`input_as_shape` | *(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.
*Inputs* | 
`input` | (Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*
*Outputs* | 
`output` | Output tensor of random values drawn from a normal distribution. If the shape argument is set, this is the shape specified, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

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
`correct_transform_coords` | bool (default false), Correct bounding box transform coordates, see bbox_transform() in boxes.py Set to true to match the detectron code, set to false for backward compatibility
`angle_bound_on` | bool (default true). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi].
`angle_bound_lo` | int (default -90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi].
`angle_bound_hi` | int (default 90 degrees). If set, for rotated boxes, angle is normalized to be within [angle_bound_lo, angle_bound_hi].
`clip_angle_thresh` | float (default 1.0 degrees). For RRPN, clip almost horizontal boxes within this threshold of tolerance for backward compatibility. Set to negative value for no clipping.
*Inputs* | 
`scores` | Scores from conv layer, size (img_count, A, H, W)
`bbox_deltas` | Bounding box deltas from conv layer, size (img_count, 4 * A, H, W)
`im_info` | Image info, size (img_count, 3), format (height, width, scale)
`anchors` | Bounding box anchors, size (A, 4)
*Outputs* | 
`rois` | Proposals, size (n x 5), format (image_index, x1, y1, x2, y2)
`rois_probs` | scores of proposals, size (n)


### Code


[caffe2/operators/generate_proposals_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/generate_proposals_op.cc)

---



## GenerateProposalsCPP

No documentation yet.


### Code


[caffe2/operators/generate_proposals_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/generate_proposals_op.cc)

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


[caffe2/operators/workspace_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/workspace_ops.cc)

---



## GetCursorOffset

Get the current offset in the cursor.


### Interface


---------- | ----------
*Inputs* | 
`cursor` | A blob containing a pointer to the cursor.
*Outputs* | 
`offsets` | Tensor containing the offsets for the cursor.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## GetGPUMemoryUsage

Fetches GPU memory stats from CUDAContext. Result is stored  

```
      in output blob with shape (2, num_gpus). First row contains the total
      current memory usage, and the second row the maximum usage during
      this execution.

      NOTE: --caffe2_gpu_memory_tracking flag must be enabled to use this op.
```

     


### Code


[caffe2/operators/mem_query_op.cu](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mem_query_op.cu)

---



## GivenTensorBoolFill

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`values` | The value for the elements of the output tensor.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape. First input must be in CPU context.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorByteStringToUInt8Fill


This op fills a uint8 output tensor with the data specified by the  *value*  argument. The data must previously be serialized as a byte string. The output tensor shape is specified by the  *shape*  argument. Beware, when using this argument  *value*  should have a value for every element of the  *output* , as missing values will not be initialized automatically. If  *input_as_shape*  is set to  *true* , then the  *input*  should be a 1D tensor containing the desired output shape (the dimensions specified in  *extra_shape*  will also be appended). In this case, the  *shape*  argument should  **not**  be set.
 This op allows us to write uint8 tensors to Protobuf as byte strings and read them back as uint8 tensors in order to avoid the Protobuf uint32_t varint encoding size penalty.
 <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  val = np.array([1, 2, 3], dtype=np.uint8) op = core.CreateOperator(  

```
    "GivenTensorByteStringToUInt8Fill",
    [],
    ["out"],
    values=[val.tobytes()],
    shape=val.shape,
```

 )  workspace.RunOperatorOnce(op) print("Out:\n", workspace.FetchBlob("out"))   ```    **Result**    ```   Out:  [1 2 3]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`values` | The value for the elements of the output tensor.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape. First input must be in CPU context.


### Code


[caffe2/operators/given_tensor_byte_string_to_uint8_fill_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/given_tensor_byte_string_to_uint8_fill_op.cc)

---



## GivenTensorDoubleFill

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`values` | The value for the elements of the output tensor.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape. First input must be in CPU context.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorFill


This op fills an output tensor with the data specified by the  *value*  and  *dtype*  arguments. 

```
  The output tensor shape is specified by the *shape* argument. Beware, when using this argument *value* should have a value for every element of the *output*, as missing values will not be initialized automatically. If *input_as_shape* is set to *true*, then the *input* should be a 1D tensor containing the desired output shape (the dimensions specified in *extra_shape* will also be appended). In this case, the *shape* argument should **not** be set.

```

 [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.h) *Note: Do not set the shape argument and pass in an input at the same time.*   Github Links: -   -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/given_tensor_fill_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "GivenTensorFill",
    [],
    ["out"],
    values=[1., 2., 3.],
    shape=[3],
```

 )  workspace.RunOperatorOnce(op) print("Out:\n", workspace.FetchBlob("out"))   ```    **Result**    ```   Out:  [1. 2. 3.]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`values` | *(type depends on dtype, Required=True)* The value of the elements to go in the *output* tensor.
`dtype` | The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto.
`shape` | *(type: [int])* Desired shape of the *output* tensor.
`extra_shape` | *(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob.
`input_as_shape` | *(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.
*Inputs* | 
`input` | (Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*
*Outputs* | 
`output` | Output tensor with desired dimension filled with specified data. If the shape argument is set, this is the shape specified, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorInt64Fill

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`values` | The value for the elements of the output tensor.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape. First input must be in CPU context.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorIntFill

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`values` | The value for the elements of the output tensor.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape. First input must be in CPU context.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/given_tensor_fill_op.cc)

---



## GivenTensorStringFill

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`values` | The value for the elements of the output tensor.
`shape` | The shape of the output tensor.Cannot set the shape argument and pass in an input at the same time.
`extra_shape` | The additional dimensions appended at the end of the shape indicatedby the input blob.Cannot set the extra_shape argument when there is no input blob.
`input_as_shape` | 1D tensor containing the desired output shape. First input must be in CPU context.


### Code


[caffe2/operators/given_tensor_fill_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/given_tensor_fill_op.cc)

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


[caffe2/operators/glu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/glu_op.cc)

---



## GroupNorm


Group Normalization (GN) operation:  [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)  


### Interface


---------- | ----------
*Arguments* | 
`num_groups` | (int) default 32; number of groups used by GN.
`epsilon` | (float) default 1e-5; small constant added to var.
*Inputs* | 
`X` | >=4D feature map input of shape (N, C, H, W) or (N, C, T, H, W)
`gamma` | The scale as a 1-dimensional tensor of size C to be applied to the output.
`beta` | The bias as a 1-dimensional tensor of size C to be applied to the output.
*Outputs* | 
`Y` | The output >=4-dimensional tensor of the same shape as X.
`mean` | The mean of shape (N, G). For backward usage or reference. Cannot be used as activations.
`std` | The std of shape (N, G). For backward usage or reference. Cannot be used as activations.


### Code


[caffe2/operators/group_norm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/group_norm_op.cc)

---



## GroupNormGradient

No documentation yet.


### Code


[caffe2/operators/group_norm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/group_norm_op.cc)

---



## HSoftmax


Hierarchical softmax is an operator which approximates the softmax operator while giving significant training speed gains and reasonably comparable performance. In this operator, instead of calculating the probabilities of all the classes, we calculate the probability of each step in the path from root to the target word in the hierarchy.
 The operator takes a 2-D tensor (Tensor) containing a batch of layers, a set of parameters represented by the weight matrix and bias terms, and a 1-D tensor (Tensor) holding labels, or the indices of the target class. The hierarchy has to be specified as an argument to the operator.
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


[caffe2/operators/h_softmax_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/h_softmax_op.cc)

---



## HSoftmaxGradient

No documentation yet.


### Code


[caffe2/operators/h_softmax_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/h_softmax_op.cc)

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


[caffe2/operators/h_softmax_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/h_softmax_op.cc)

---



## HalfToFloat

No documentation yet.


### Code


[caffe2/operators/half_float_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/half_float_ops.cc)

---



## HasElements


The  *HasElements*  op accepts a single input $tensor$, and produces a single boolean output $has\_elements$. The output is  *True*  if and only if $tensor$ has size > 0. Note, this op is the opposite of the  *IsEmpty*  op.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "HasElements",
    ["tensor"],
    ["has_elements"],
```

 )  # Use a not-empty tensor workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32)) print("tensor:\n", workspace.FetchBlob("tensor"))  workspace.RunOperatorOnce(op) print("has_elements: ", workspace.FetchBlob("has_elements"),"\n")  # Use an empty tensor workspace.FeedBlob("tensor", np.empty(0)) print("tensor:\n", workspace.FetchBlob("tensor"))  workspace.RunOperatorOnce(op) print("has_elements: ", workspace.FetchBlob("has_elements"))   ```    **Result**    ```   tensor:  [[ 0.6116506 

```
  -0.54433197]
```

  [ 0.19406661 -0.7338629 ]] has_elements: 

```
  True

```

 tensor:  [] has_elements: 

```
  False

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`tensor` | Input data tensor to check for elements.
*Outputs* | 
`has_elements` | Output scalar boolean tensor. True if input has size > 0.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## HasScope


Checks whether scope blob has any saved scopes left     


### Code


[caffe2/operators/create_scope_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/create_scope_op.cc)

---



## HeatmapMaxKeypoint

No documentation yet.


### Code


[caffe2/operators/heatmap_max_keypoint_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/heatmap_max_keypoint_op.cc)

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


[caffe2/operators/h_softmax_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/h_softmax_op.cc)

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


[caffe2/operators/if_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/if_op.cc)

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


[caffe2/operators/im2col_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/im2col_op.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

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


[caffe2/operators/index_hash_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_hash_ops.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

---



## InstanceNorm


The  *InstanceNorm*  op applies Instance Normalization over a 4D input as described in [Instance Normalization: The Missing Ingredient for Fast Stylization]( [https://arxiv.org/abs/1607.08022).](https://arxiv.org/abs/1607.08022).) 
 $$output = \frac{input-\mu_{input}}{\sqrt{\sigma_{input}^2} + \epsilon}*scale + bias$$  Notice, two of the outputs are optional so there are three output cases for this op. Case 1: output; Case 2: output, saved_mean; Case 3: output, saved_mean, saved_inv_stdev.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "InstanceNorm",
    ["input", "scale", "bias"],
    ["output"],
    epsilon=1e-5,
```

 )  workspace.FeedBlob("input", np.random.randn(2, 1, 3, 3).astype(np.float32)) print("input:\n", workspace.FetchBlob("input"), "\n")  workspace.FeedBlob("scale", np.array([1.5]).astype(np.float32)) print("scale: ", workspace.FetchBlob("scale"))  workspace.FeedBlob("bias", np.array([1.]).astype(np.float32)) print("bias: ", workspace.FetchBlob("bias"))  workspace.RunOperatorOnce(op) print("output:\n", workspace.FetchBlob("output"))   ```    **Result**    ```   input:  [[[[ 0.97856593 -1.1832817 

```
  -0.2540021 ]
```

   

```
  [-1.3315694  -0.7485018   0.3787225 ]
```

   

```
  [-0.6826597  -1.4637762   0.57116514]]]


```

  [[[-0.44948956 

```
  0.85544354 -0.9315333 ]
```

   

```
  [-0.37202677 -0.22266895 -0.27194235]
```

   

```
  [ 0.4948163  -0.7296504   1.3393803 ]]]]

```

 scale: 

```
  [1.5]
```

 bias: 

```
  [1.]
```

 output:  [[[[ 3.5017493 

```
  -0.3791256   1.2890853 ]
```

   

```
  [-0.6453266   0.40137637  2.4249308 ]
```

   

```
  [ 0.5195738  -0.8826599   2.7703972 ]]]


```

  [[[ 0.12639964 

```
  2.856744   -0.8821926 ]
```

   

```
  [ 0.28847694  0.60098207  0.49788612]
```

   

```
  [ 2.1021945  -0.45978796  3.869297  ]]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | *(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero.
`order` | *(type: string; default: "NCHW")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is "NHWC".
*Inputs* | 
`input` | The input 4-dimensional NCHW tensor to be operated on.
`scale` | The input 1-dimensional scale tensor of size *C*.
`bias` | The input 1-dimensional bias tensor of size *C*.
*Outputs* | 
`output` | The output 4-dimensional tensor of the same shape as input.
`saved_mean` | (Optional) Saved mean used during training to speed up gradient computation. Should not be used for testing.
`saved_inv_stdev` | (Optional) Saved inverse stdev used during training to speed up gradient computation. Should not be used for testing.


### Code


[caffe2/operators/instance_norm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/instance_norm_op.cc)

---



## InstanceNormGradient

No documentation yet.


### Code


[caffe2/operators/instance_norm_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/instance_norm_gradient_op.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

---



## IntegralImage


Computes an integral image, which contains the sum of pixel values within an image vertically and horizontally. This integral image can then be used with other detection and tracking techniques.



### Interface


---------- | ----------
*Inputs* | 
`X` | Images tensor of the form (N, C, H, W)
*Outputs* | 
`Y` | Integrated image of the form (N, C, H+1, W+1)


### Code


[caffe2/operators/integral_image_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/integral_image_op.cc)

---



## IntegralImageGradient

No documentation yet.


### Code


[caffe2/operators/integral_image_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/integral_image_op.cc)

---



## IsEmpty


The  *IsEmpty*  op accepts a single input $tensor$, and produces a single boolean output $is\_empty$. The output is  *True*  if and only if $tensor$ has size == 0.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "IsEmpty",
    ["tensor"],
    ["is_empty"],
```

 )  # Use a not-empty tensor workspace.FeedBlob("tensor", np.random.randn(2, 2).astype(np.float32)) print("tensor:\n", workspace.FetchBlob("tensor"))  workspace.RunOperatorOnce(op) print("is_empty: ", workspace.FetchBlob("is_empty"),"\n")  # Use an empty tensor workspace.FeedBlob("tensor", np.empty(0)) print("tensor:\n", workspace.FetchBlob("tensor"))  workspace.RunOperatorOnce(op) print("is_empty: ", workspace.FetchBlob("is_empty"))   ```    **Result**    ```   tensor:  [[ 0.26018378 

```
  0.6778789 ]
```

  [-1.3097627 

```
  -0.40083608]]
```

 is_empty: 

```
  False

```

 tensor:  [] is_empty: 

```
  True

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`tensor` | Input data tensor to check if empty.
*Outputs* | 
`is_empty` | Output scalar boolean tensor. True if input has size == 0.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## IsMemberOf


The  *IsMemberOf*  op takes an input tensor  *X*  and a list of values as argument, and produces one output data tensor  *Y* . The output tensor is the same shape as  *X*  and contains booleans. The output is calculated as the function  *f(x) = x in value*  and is applied to  *X*  elementwise.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.cc)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/elementwise_logical_ops.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "IsMemberOf",
    ["X"],
    ["Y"],
    value=[0,2,4,6,8],
```

 )  # Use a not-empty tensor workspace.FeedBlob("X", np.array([0,1,2,3,4,5,6,7,8]).astype(np.int32)) print("X:\n", workspace.FetchBlob("X"))  workspace.RunOperatorOnce(op) print("Y: \n", workspace.FetchBlob("Y"))   ```    **Result**    ```  # value=[0,2,4,6,8]  X:  [0 1 2 3 4 5 6 7 8] Y:  [ True False 

```
  True False  True False  True False  True]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`value` | *(type: []; default: -)* List of values to check for membership.
`dtype` | *(type: TensorProto_DataType; default: -)* The data type for the elements of the output tensor. Strictly must be one of the types from DataType enum in TensorProto.
*Inputs* | 
`X` | Input tensor of any shape
*Outputs* | 
`Y` | Output tensor (same size as X containing booleans)


### Code


[caffe2/operators/elementwise_logical_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_logical_ops.cc)

---



## KeySplit

No documentation yet.


### Code


[caffe2/operators/key_split_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/key_split_ops.cc)

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


[caffe2/operators/map_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/map_ops.cc)

---



## L1Distance


Computes the row-wise L1 Distance between the two input tensors $X$ and $Y$, which is defined as  $$L1Distance(\mathbf{x},\mathbf{y}) = \sum_{i}\mid x_i - y_i\mid$$  Note, both inputs must either be 1-dimensional or 2-dimensional and both must have the same shape. The output $Z$ will be 1-dimensional regardless and its length will equal the number of rows in the inputs.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "L1Distance",
    ["X", "Y"],
    ["Z"]
```

 )  # Create X X = 5*np.ones((1, 4)) print("X:\n",X)  # Create Y Y = np.ones((1, 4)) print("Y:\n",Y)  # Feed X & Y into workspace workspace.FeedBlob("X", X.astype(np.float32)) workspace.FeedBlob("Y", Y.astype(np.float32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Z:\n", workspace.FetchBlob("Z"))   ```    **Result**    ```   X:  [[5. 5. 5. 5.]] Y:  [[1. 1. 1. 1.]] Z:  [16.]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | First input tensor. (1D or 2D)
`Y` | Second input tensor. (must have the same shape as $X$)
*Outputs* | 
`Z` | 1D output tensor. One value for each row of the inputs.


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## L1DistanceGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## LC


The locally connected operator consumes an input vector, a filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW) if order == NCHW else (YH * YW * M  * KH * KW * C), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC1D


The locally connected operator consumes an input vector, a 1D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW) if order == NCHW else (YH * YW * M  * KH * KW * C), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC1DGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC2D


The locally connected operator consumes an input vector, a 2D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW) if order == NCHW else (YH * YW * M  * KH * KW * C), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC2DGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC3D


The locally connected operator consumes an input vector, a 3D filter blob and a bias blob and computes the output.  Note that other parameters, such as the stride and kernel size, or the pads' sizes in each direction are not necessary for input because they are provided by the ConvPoolOpBase operator. Various dimension checks are done implicitly, and the sizes are specified in the Input docs for this operator. As is expected, the filter is locally connected with a subset of the image and the bias is added; this is done throughout the image data and the output is computed. As a side note on the implementation layout: locally_connected_op_impl.h is the templated implementation of the locally_connected_op.h file, which is why they are separate files.



### Interface


---------- | ----------
*Inputs* | 
`None` | 
`filter` | The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW) if order == NCHW else (YH * YW * M  * KH * KW * C), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel.
`bias` | The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).
*Outputs* | 
`Y` | Output data blob that contains the result of the locally connected op.The output dimensions are functions of the kernel size, stride size, and pad lengths.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LC3DGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LCGradient

No documentation yet.


### Code


[caffe2/operators/locally_connected_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/locally_connected_op.cc)

---



## LE


Performs element-wise less or equal than comparison  **<=**  (with limited broadcast support).
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "LE",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3])) workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [ 1 

```
  5  2  9 12  3]
```

 B: [ 1 

```
  3  4  9 12  8]
```

 C: [ True False 

```
  True  True  True  True]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## LRN


  `LRN`  applies Local Response Normalization to an input blob. This operation performs a kind of "lateral inhibition" by normalizing over local input regions, where  normalization is applied across channels. This operator is typically used to  normalize an unbounded activation (such as ReLU). The output shape is the same as the input shape. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 The formula for LRN is as follows:  $$b_{c} = a_{c}(bias + \frac{\alpha}{n}\sum_{c'=max(0,c-n/2)}^{min(N-1,c+n/2)} a_{c'}^2 )^{-\beta}$$   Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator("LRN",   

```
    ["X"],
```

   

```
    ["Y", "Y_scale"],
```

   

```
    size=11,
```

   

```
    alpha=0.001,
```

   

```
    beta=0.5,
```

   

```
    bias=2.0,
```

   

```
    order="NHWC"
```

 )  workspace.FeedBlob("X", np.random.randn(1, 6, 6, 1).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y")) print("Y_scale:\n", workspace.FetchBlob("Y_scale"))  ```    **Result**    ```  X:  [[[[ 0.72985137]   

```
  [-0.3753357 ]
```

   

```
  [ 2.7344604 ]
```

   

```
  [-0.5937792 ]
```

   

```
  [ 0.38440478]
```

   

```
  [-2.1659644 ]]

  [[-0.92846817]
```

   

```
  [-0.9996144 ]
```

   

```
  [ 0.212943  ]
```

   

```
  [-1.968045  ]
```

   

```
  [-0.77839696]
```

   

```
  [ 0.45492038]]

  [[-0.11263168]
```

   

```
  [ 1.9901097 ]
```

   

```
  [ 0.19275683]
```

   

```
  [ 0.15630436]
```

   

```
  [ 0.7536298 ]
```

   

```
  [-0.77339894]]

  [[ 0.8353551 ]
```

   

```
  [-0.7784452 ]
```

   

```
  [ 1.779317  ]
```

   

```
  [ 0.22421335]
```

   

```
  [ 1.3846219 ]
```

   

```
  [-3.0546608 ]]

  [[ 0.09977621]
```

   

```
  [ 2.2071757 ]
```

   

```
  [ 0.79971045]
```

   

```
  [ 3.563886  ]
```

   

```
  [-0.7169287 ]
```

   

```
  [ 0.77170426]]

  [[-1.4296649 ]
```

   

```
  [ 0.19181213]
```

   

```
  [ 0.45961624]
```

   

```
  [-1.0201577 ]
```

   

```
  [ 0.62854475]
```

   

```
  [-0.6395456 ]]]] 

```

 Y:  [[[[ 0.5160766 ]   

```
  [-0.26540157]
```

   

```
  [ 1.9332271 ]
```

   

```
  [-0.41986194]
```

   

```
  [ 0.27181432]
```

   

```
  [-1.5314047 ]]

  [[-0.6565133 ]
```

   

```
  [-0.7068181 ]
```

   

```
  [ 0.15057328]
```

   

```
  [-1.3914955 ]
```

   

```
  [-0.5504022 ]
```

   

```
  [ 0.32167578]]

  [[-0.0796426 ]
```

   

```
  [ 1.4070934 ]
```

   

```
  [ 0.13629955]
```

   

```
  [ 0.11052381]
```

   

```
  [ 0.53288984]
```

   

```
  [-0.5468682 ]]

  [[ 0.5906759 ]
```

   

```
  [-0.5504363 ]
```

   

```
  [ 1.2580767 ]
```

   

```
  [ 0.1585426 ]
```

   

```
  [ 0.9790328 ]
```

   

```
  [-2.1595135 ]]

  [[ 0.07055242]
```

   

```
  [ 1.5605361 ]
```

   

```
  [ 0.5654725 ]
```

   

```
  [ 2.5193207 ]
```

   

```
  [-0.50693923]
```

   

```
  [ 0.54567   ]]

  [[-1.0108787 ]
```

   

```
  [ 0.13563155]
```

   

```
  [ 0.3249962 ]
```

   

```
  [-0.72134334]
```

   

```
  [ 0.44444424]
```

   

```
  [-0.45222285]]]]
```

 Y_scale:  [[[[2.0000484]   

```
  [2.0000129]
```

   

```
  [2.0006797]
```

   

```
  [2.000032 ]
```

   

```
  [2.0000134]
```

   

```
  [2.0004265]]

  [[2.0000784]
```

   

```
  [2.0000908]
```

   

```
  [2.000004 ]
```

   

```
  [2.0003521]
```

   

```
  [2.000055 ]
```

   

```
  [2.0000188]]

  [[2.0000012]
```

   

```
  [2.00036  ]
```

   

```
  [2.0000033]
```

   

```
  [2.0000021]
```

   

```
  [2.0000517]
```

   

```
  [2.0000544]]

  [[2.0000634]
```

   

```
  [2.000055 ]
```

   

```
  [2.0002878]
```

   

```
  [2.0000045]
```

   

```
  [2.0001743]
```

   

```
  [2.0008483]]

  [[2.000001 ]
```

   

```
  [2.000443 ]
```

   

```
  [2.0000582]
```

   

```
  [2.0011547]
```

   

```
  [2.0000467]
```

   

```
  [2.0000541]]

  [[2.0001857]
```

   

```
  [2.0000033]
```

   

```
  [2.0000193]
```

   

```
  [2.0000947]
```

   

```
  [2.000036 ]
```

   

```
  [2.0000372]]]]
```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`size` | *(type: int; default: 0)* Amount of neighboring channels to sum over for normalization
`alpha` | *(type: float; default: 0)* Multiplicative (scaling) factor.
`beta` | *(type: float; default: 0)* Exponent.
`bias` | *(type: float; default: 1.0)* Additive factor.
`order` | *(type: float; default: 'NCHW')* Order of blob dimensions.
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor (ReLU output).
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.
`Y_scale` | *(type: Tensor`<float>`)* Output scale.


### Code


[caffe2/operators/local_response_normalization_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.cc)

---



## LRNGradient

No documentation yet.


### Code


[caffe2/operators/local_response_normalization_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.cc)

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


[caffe2/operators/lstm_unit_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lstm_unit_op.cc)

---



## LSTMUnitGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`sequence_lengths` | When false, the sequence lengths input is left out, and all following inputs are shifted left by one.


### Code


[caffe2/operators/lstm_unit_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lstm_unit_op.cc)

---



## LT


Performs element-wise less than comparison  **<**  (with limited broadcast support).
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "LT",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3])) workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [ 1 

```
  5  2  9 12  3]
```

 B: [ 1 

```
  3  4  9 12  8]
```

 C: [False False 

```
  True False False  True]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## LabelCrossEntropy


This operator computes the cross entropy between a $NxD$ dimensional input data tensor $X$ 

```
  and a one dimensional input label tensor $label$. The op produces a single length $N$ output tensor $Y$. Here, $N$ is considered the batch size and $D$ is the size of each element in the batch. In practice, it is most commonly used at the end of models as a part of the loss computation, after the SoftMax operator and before the AveragedLoss operator. The cross entropy operation is defined as follows

```

 $$Y_i = -log(X_{ij})$$  where ($i$, $j$) is the classifier's prediction of the $j$th class (the correct one), and $i$ is the batch size. Each log has a lower limit for numerical stability.
 The difference between  *LabelCrossEntropy*  and  *CrossEntropy*  is how the labels are specified. Here, the labels are a length $N$ list of integers, whereas in CrossEntropy the labels are a $NxD$ dimensional matrix of one hot label vectors. However, the results of computation should be the same, as shown in the two examples where ($i$, $j$) is the classifier's prediction of the $j$th class (the correct one), and $i$ is the batch size. Each log has a lower limit for numerical stability.
 Github Links: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/cross_entropy_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LabelCrossEntropy",
    ["X", "label"],
    ["Y"]
```

 )  # Create X: Sample softmax output for 5-class model X = np.array([[.01, .05, .02, .02, .9],[.03, .1, .42, .05, .4]]) print("X:\n",X)  # Create label: Sample 1-hot ground truth label vectors label = np.array([4,2]) print("label:\n",label)  # Feed X & label into workspace workspace.FeedBlob("X", X.astype(np.float32)) workspace.FeedBlob("label", label.astype(np.int32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[0.01 0.05 0.02 0.02 0.9 ]  [0.03 0.1 

```
  0.42 0.05 0.4 ]]
```

 label:  [4 2] Y:  [0.10536055 0.8675006 ]   ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input tensor which is almost always the result of a softmax operation. $X$ is a 2D array of size $NxD$, where $N$ is the batch size and $D$ is the number of classes.
`label` | Blob containing the labels used to compare the input. $label$ is a length $N$ list of integers, where each element is the integer label for the $n$th element of the batch.
*Outputs* | 
`Y` | Output blob from the cross entropy computation. $Y$ is 1D length $N$ tensor.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## LabelCrossEntropyGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## LambdaRankNdcg


It implements the LambdaRank as appeared in Wu, Qiang, et al. "Adapting boosting for information retrieval measures." Information Retrieval 13.3 (2010): 254-270.
 This method heuristically optimizes the NDCG.



### Code


[caffe2/operators/listwise_l2r_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/listwise_l2r_op.cc)

---



## LambdaRankNdcgGradient

No documentation yet.


### Code


[caffe2/operators/listwise_l2r_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/listwise_l2r_op.cc)

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


[caffe2/operators/last_n_window_collector.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/last_n_window_collector.cc)

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


[caffe2/operators/layer_norm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/layer_norm_op.cc)

---



## LayerNormGradient

No documentation yet.


### Code


[caffe2/operators/layer_norm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/layer_norm_op.cc)

---



## LeakyRelu


The  *LeakyRelu*  op takes one input tensor $X$ and an argument $alpha$, and produces one output tensor $Y$ of the same shape as $X.$ The op performs the element wise leaky relu operation, defined as  $$y=LeakyRelu(x) =\begin{cases}\alpha x & x < 0\\x & otherwise\end{cases}$$  The default value of  *alpha*  is 0.01.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LeakyRelu",
    ["X"],
    ["Y"],
    alpha=0.01
```

 )  workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32)) print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[-0.91060215 

```
  0.09374836  2.1429708 ]
```

  [-0.748983 

```
    0.19164062 -1.5130422 ]
```

  [-0.29539835 -0.8530696  

```
  0.7673204 ]]

```

 Y:  [[-0.00910602 

```
  0.09374836  2.1429708 ]
```

  [-0.00748983 

```
  0.19164062 -0.01513042]
```

  [-0.00295398 -0.0085307  

```
  0.7673204 ]]

```

 ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`alpha` | *(type: float; default: 0.01)* Coefficient of leakage.
*Inputs* | 
`X` | Input tensor of data to be operated on.
*Outputs* | 
`Y` | Output tensor, calculated as described above.


### Code


[caffe2/operators/leaky_relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.cc)

---



## LeakyReluGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`alpha` | Coefficient of leakage


### Code


[caffe2/operators/leaky_relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/leaky_relu_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsIndicesInGradientMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsIndicesInGradientSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMax


Applies 'Max' to each segment of the input tensor. Segments are defined by their  *LENGTHS* .  *LENGTHS*  is a vector that maps each of the slices of  *DATA*  to a particular segment. Values belonging to the same segment are aggregated together and considered for the 'Max' operation.
 For example  *LENGTHS = [2, 1]*  stands for segments  *DATA[0..1]*  and  *DATA[2]*   The sum of elements in  *LENGTHS*  must equal the number of elements in the first dimension of  *DATA* . The length of  *OUTPUT*  is equal to the number of input segments, i.e. len( *LENGTHS* ).
 Max computes the element-wise max of the input slices. Operation doesn't change the shape of the individual blocks.
  The  *LengthsMax*  op takes two inputs  *DATA*  and  *LENGTHS* , and produces a single output  *OUTPUT* . The op finds the maximum value in each of the segments of  *DATA* , where segments are defined by their lengths.
For example, if $DATA = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$ then $OUTPUT = [max([2,4]), max([3,1,2]), max([10])] = [4,3,10]$.
 Github Link: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LengthsMax",
    ["DATA", "LENGTHS"],
    ["OUTPUT"],
```

 )  workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32)) print("DATA:\n", workspace.FetchBlob("DATA"))  workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32)) print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))  workspace.RunOperatorOnce(op) print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))   ```    **Result**    ```   DATA:  [ 2. 

```
  4.  3.  1.  2. 10.]
```

 LENGTHS:  [2 3 1] OUTPUT:  [ 4. 

```
  3. 10.]

```

 ```   </details>     


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of len(LENGTHS) 


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMaxWithMainInputAndForwardOutputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMean


Applies 'Mean' to each segment of the input tensor. Segments are defined by their  *LENGTHS* .  *LENGTHS*  is a vector that maps each of the slices of  *DATA*  to a particular segment. Values belonging to the same segment are aggregated together and considered for the 'Mean' operation.
 For example  *LENGTHS = [2, 1]*  stands for segments  *DATA[0..1]*  and  *DATA[2]*   The sum of elements in  *LENGTHS*  must equal the number of elements in the first dimension of  *DATA* . The length of  *OUTPUT*  is equal to the number of input segments, i.e. len( *LENGTHS* ).
 Mean computes the element-wise mean of the input slices. Operation doesn't change the shape of the individual blocks.
  The  *LengthsMean*  op takes two inputs  *DATA*  and  *LENGTHS* , and produces a single output  *OUTPUT* . The op finds the mean value in each of the segments of  *DATA* , where segments are defined by their lengths.
For example, if $DATA = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$ then $OUTPUT = [mean([2,4]), mean([3,1,2]), mean([10])] = [3,2,10]$.
 Github Link: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LengthsMean",
    ["DATA", "LENGTHS"],
    ["OUTPUT"],
```

 )  workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32)) print("DATA:\n", workspace.FetchBlob("DATA"))  workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32)) print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))  workspace.RunOperatorOnce(op) print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))   ```    **Result**    ```   DATA:  [ 2. 

```
  4.  3.  1.  2. 10.]
```

 LENGTHS:  [2 3 1] OUTPUT:  [ 3. 

```
  2. 10.]

```

 ```   </details>     


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of len(LENGTHS) 


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsPad


Given DATA tensor of rank r >= 1, and LENGTHS tensor of rank 1, pad each segment in DATA with  `value` , so that each segment's length is  `target_length` .
If will throw, if there is segment of length larger than  `target_length` .
 Example:  

```
  DATA  = [
      [2.3, 3.4],
      [4.5, 5.7],
      [6.8, 7.9],
  ]
  LENGTHS = [0, 1, 1, 1]
  and target_length = 2, padding value = -1.0
  OUTPUT = [
    [-1.0, -1.0],
    [-1.0, -1.0],
    [2.3, 3.4],
    [-1.0, -1.0],
    [4.5, 5.7],
    [-1.0, -1.0],
    [6.8, 7.9],
    [-1.0, -1.0],
  ]
```




### Interface


---------- | ----------
*Arguments* | 
`padding_value` | The value to pad the data
`target_length` | The target length of each segment
*Inputs* | 
`DATA` | Tensor of rank r >= 1. First dimension must be equal to the size of lengths
`LENGTHS` | Tensor of int32 lengths of rank 1
*Outputs* | 
`OUTPUT` | Padded DATA tensor


### Code


[caffe2/operators/lengths_pad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_pad_op.cc)

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


[caffe2/operators/partition_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/partition_ops.cc)

---



## LengthsRangeFill


The  *LengthsRangeFill*  op takes a single input  *lengths*  and outputs a single tensor  *range_sequence* . For each element of  *lengths* , the op appends the range(0,lengths) vector to the end of  *range_sequence* . For example, if input=[2,4,1], the output would be [0,1,0,1,2,3,0].
 Github Links: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LengthsRangeFill",
    ["lengths"],
    ["range_sequence"],
```

 )  workspace.FeedBlob("lengths", np.array([2,4,1]).astype(np.int32)) print("lengths:\n", workspace.FetchBlob("lengths"))  workspace.RunOperatorOnce(op) print("range_sequence: \n", workspace.FetchBlob("range_sequence"))   ```    **Result**    ```   lengths:  [2 4 1] range_sequence:  [0 1 0 1 2 3 0]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`lengths` | 1D tensor of int32 or int64 segment lengths.
*Outputs* | 
`range_sequence` | 1D tensor whose size is the sum of *lengths*


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

---



## LengthsSum


Applies 'Sum' to each segment of the input tensor. Segments are defined by their  *LENGTHS* .  *LENGTHS*  is a vector that maps each of the slices of  *DATA*  to a particular segment. Values belonging to the same segment are aggregated together and considered for the 'Sum' operation.
 For example  *LENGTHS = [2, 1]*  stands for segments  *DATA[0..1]*  and  *DATA[2]*   The sum of elements in  *LENGTHS*  must equal the number of elements in the first dimension of  *DATA* . The length of  *OUTPUT*  is equal to the number of input segments, i.e. len( *LENGTHS* ).
 Summation is done element-wise across slices of the input tensor and doesn't change the shape of the individual blocks.
  The  *LengthsSum*  op takes two inputs  *DATA*  and  *LENGTHS* , and produces a single output  *OUTPUT* . The op finds the sum in each of the segments of  *DATA* , where segments are defined by their lengths.
For example, if $DATA = [2,4,3,1,2,10]$ and $LENGTHS = [2,3,1]$ then $OUTPUT = [sum([2,4]), sum([3,1,2]), sum([10])] = [6,6,10]$.
 Github Link: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LengthsSum",
    ["DATA", "LENGTHS"],
    ["OUTPUT"],
```

 )  workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32)) print("DATA:\n", workspace.FetchBlob("DATA"))  workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32)) print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))  workspace.RunOperatorOnce(op) print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))   ```    **Result**    ```   DATA:  [ 2. 

```
  4.  3.  1.  2. 10.]
```

 LENGTHS:  [2 3 1] OUTPUT:  [ 6. 

```
  6. 10.]

```

 ```   </details>     


### Interface


---------- | ----------
*Inputs* | 
`DATA` | Input tensor, slices of which are aggregated.
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`OUTPUT` | Aggregated output tensor. Has the first dimension of len(LENGTHS) 


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/lengths_tile_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_tile_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsToSegmentIds


Given a vector of segment lengths ( *lengths* ) the  *LengthsToSegmentIds*  op returns a zero-based, consecutive vector of segment ids ( *segment_ids* ). For example,  *lengths=[1, 3, 0, 2]*  will produce  *segment_ids=[0, 1, 1, 1, 3, 3]* . In general, the inverse operation is  *SegmentIdsToLengths* . Notice though that trailing empty sequence lengths can't be properly recovered from segment ids.
 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LengthsToSegmentIds",
    ["lengths"],
    ["segment_ids"],
```

 )  workspace.FeedBlob("lengths", np.array([1, 3, 0, 2]).astype(np.int32)) print("lengths:\n", workspace.FetchBlob("lengths"))  workspace.RunOperatorOnce(op) print("segment_ids: \n", workspace.FetchBlob("segment_ids"))   ```    **Result**    ```   lengths:  [1 3 0 2] segment_ids:  [0 1 1 1 3 3]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`lengths` | 1D tensor of int32 or int64 segment lengths.
*Outputs* | 
`segment_ids` | 1D tensor of length *sum(lengths)*


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## LengthsToShape


This operator takes a list of $N$ equal integers as input which represent the lengths of $N$ vectors. The output is the calculated shape of the matrix if the $N$ integers were combined into a single matrix.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LengthsToShape",
    ["X"],
    ["Y"]
```

 )  # Create X: Sample softmax output for 5-class model X = np.array([2,2,2,2,2,2,2,2,2,2]) print("X:\n",X)  # Feed X into workspace workspace.FeedBlob("X", X.astype(np.int32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [2 2 2 2 2 2 2 2 2 2] Y:  [10 

```
  2]

```

 ```   </details>      


### Interface


---------- | ----------
*Inputs* | 
`X` | List, of length $N$, of equal integers representing the lengths of several vectors.
*Outputs* | 
`Y` | Vector of length 2 describing the dimensions of the data if the $N$ vectors from the input were combined to a single matrix.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/lengths_top_k_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_top_k_op.cc)

---



## LengthsTopKGradient

No documentation yet.


### Code


[caffe2/operators/lengths_top_k_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_top_k_op.cc)

---



## LengthsWeightedSum


Applies 'WeightedSum' to each segment of the input tensor. Segments are defined by their  *LENGTHS* .  *LENGTHS*  is a vector that maps each of the slices of  *DATA*  to a particular segment. Values belonging to the same segment are aggregated together and considered for the 'WeightedSum' operation.
 For example  *LENGTHS = [2, 1]*  stands for segments  *DATA[0..1]*  and  *DATA[2]*   The sum of elements in  *LENGTHS*  must equal the number of elements in the first dimension of  *DATA* . The length of  *OUTPUT*  is equal to the number of input segments, i.e. len( *LENGTHS* ).
 Input slices are first scaled by SCALARS and then summed element-wise. It doesn't change the shape of the individual blocks.
  The  *LengthsWeightedSum*  op takes three inputs  *DATA* ,  *LENGTHS* , and  *SCALARS* , and produces a single output  *OUTPUT* . The op finds the weighted sum in each of the segments of  *DATA* , where segments are defined by their lengths. Before calculating the sums, the input  *DATA*  is weighted by the contents of  *SCALARS* .
For example, if $DATA = [2,4,3,1,2,10]$, $SCALARS = [8, 2, 1, 4, 1, 0.6]$, and $LENGTHS = [2,3,1]$, then $OUTPUT = [sum([8 *2,2* 4]), sum([1 *3,4* 1,1 *2]), sum([0.6* 10])] = [24,9,6]$.
 Github Link: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/segment_reduction_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LengthsWeightedSum",
    ["DATA", "SCALARS","LENGTHS"],
    ["OUTPUT"],
```

 )  workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32)) print("DATA:\n", workspace.FetchBlob("DATA"))  workspace.FeedBlob("SCALARS", np.array([8, 2, 1, 4, 1, 0.6]).astype(np.float32)) print("SCALARS:\n", workspace.FetchBlob("SCALARS"))  workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32)) print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))  workspace.RunOperatorOnce(op) print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))   ```    **Result**    ```   DATA:  [ 2. 

```
  4.  3.  1.  2. 10.]
```

 SCALARS:  [8. 

```
  2.  1.  4.  1.  0.6]
```

 LENGTHS:  [2 3 1] OUTPUT:  [24. 

```
  9.  6.]

```

 ```   </details>     


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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## LengthsWeightedSumWithMainInputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## Load


The Load operator loads a set of serialized blobs from a db or multiple dbs. It takes $[0, \infty)$ number of inputs and $[0, \infty)$ number of outputs, using the db keys to match the db entries with the outputs.
 If at least one input is passed, then it is assumed that that input blobs are a set of DBReaders to load from. Otherwise the  `db`  or  `dbs`  argument is used to load blobs from one single db or multiple dbs respectively.  `db_type`  argument is used to specify the type of the input db/dbs.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Load",
    [],
    ["X", "Y"],
    db="test_db",
    db_type="lmdb"
```

 )  workspace.RunOperatorOnce(op) print("X:", workspace.FetchBlob("X")) print("Y:", workspace.FetchBlob("Y"))   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`absolute_path` | *(type: int; default: 0)* If set to non-zero, save the db directly to the path specified by the `db` arg. If not set (default), prepend the path of the current root folder of the workspace to the path specified by the `db` arg.
`add_prefix` | *(type: string, default: "")* Blobs will be prefixed with this when loading. Useful for avoiding collisions with blobs existing in the workspace. The output blob names specified to this op should include this prefix.
`strip_prefix` | *(type: string, default: "")* Characters in the provided blob names that match `strip_prefix` will be removed prior to saving. Also, characters that precede `strip_prefix` will be removed. Useful for removing device scope from blob names.
`db` | *(type: string)* The output path of the db. See the `absolute_path` arg details for options regarding the current root folder of the workspace.
`dbs` | *(type: List(string))* List of paths to dbs to load blobs from. See the `absolute_path` arg details for options regarding the current root folder of the workspace.
`db_type` | (type: string)* Type of db to save (options: "lmdb", "leveldb", "minidb").
`keep_device` | *(type: int; default: 0)* If nonzero, the blobs are loaded into the device that is specified in the serialized `BlobProto`. Otherwise, the device will be set as the one that the `Load` operator is being run under.
`load_all` | *(type: int; default: 0)* If nonzero, will load all blobs pointed to by the db to the workspace overwriting/creating blobs as needed.
`allow_incomplete` | *(type: bool; default: False)* If True, will allow not loading all the output blobs specified in the outputs.
`source_blob_names` | *(type: List(string))* If set, used instead of output blob names to specify which blobs in the db shall be loaded. Must be the same length as number of output blobs.
*Inputs* | 
`X, Y, ...` | *(type: List(DBReader))* [OPTIONAL] List of DBReaders to load from. Can use this instead of the `db`/`dbs` args.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)

---



## Log


Calculates the natural log of the given input tensor ($ln(x)$), element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Log",
    ["X"],
    ["X"],
```

 )  workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32)) print("X before running op:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("X after running op:", workspace.FetchBlob("X"))   ```    **Result**    ```   X before running op: [[0.07341351 0.15404125 0.386613 

```
  ]
```

  [0.34090295 0.99727786 0.24141751]  [0.32016268 0.8724168 

```
  0.93515724]]
```

 X after running op: [[-2.6116474 

```
  -1.8705349  -0.9503311 ]
```

  [-1.0761575 

```
  -0.00272586 -1.4212275 ]
```

  [-1.138926  

```
  -0.13648799 -0.06704059]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor computed as the natural log of the input tensor computed, element-wise.


### Code


[caffe2/operators/log_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/log_op.cc)

---



## LogFatal

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/logit_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/logit_op.cc)

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


[caffe2/operators/logit_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/logit_op.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

---



## LpNorm


This op computes the $L_p$ norm of the one dimensional input tensor $X$, and outputs a one dimensional output tensor $Y$. Here, the $L_p$ norm is calculated as  $$L_p(\mathbf{x}) = \sum_i x_i^p$$  This op supports $p$ values of 1 or 2. If the average argument is set, the norm is calculated as Lp_averaged_norm(x) is defined as Lp_averaged_norm(x) = LpNorm(x) / size(x).
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LpNorm",
    ["X"],
    ["Y"],
    p=2
```

 ) X = np.array([5., 2.]) print("X:\n",X)  # Feed X into workspace workspace.FeedBlob("X", X.astype(np.float32))  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [5. 2.] Y:  [29.]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`p` | *(type: int; default: 2, possible values: {1,2})* Order of the norm in p-norm.
`average` | *(type: bool; default: False)* Whether we calculate norm or averaged_norm.The Lp_averaged_norm(x) is defined as Lp_averaged_norm(x) = LpNorm(x) / size(x)
*Inputs* | 
`X` | 1D Input tensor of data to be operated on.
*Outputs* | 
`Z` | 1D output tensor


### Code


[caffe2/operators/lpnorm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.cc)

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


[caffe2/operators/lpnorm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lpnorm_op.cc)

---



## LpPool


 `LpPool`  consumes an input blob and applies max pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. $L_p$ pooling consists of taking the $L_p$ norm of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lp_pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lp_pool_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "LpPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
    p=2.0
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[[[-1.1113514 

```
  -1.1173418  -0.1504435   0.1327146  -1.2221841  -0.5654315 ]
```

   

```
  [-1.9209646  -0.04675794  0.8604731   1.2042469   0.28154245   0.38656202]
```

   

```
  [-0.8772837  -0.03264008  0.26222762  0.28526652  0.321102    -2.5891325 ]
```

   

```
  [-0.9248281   1.440776   -0.56832    -0.6017927   1.2262512   -2.1443934 ]
```

   

```
  [ 0.5194415  -1.6858683   0.45221648  0.65029615 -0.8574544    0.8121054 ]
```

   

```
  [ 0.25902653  0.4934758   0.49870652 -0.48134378 -0.9178449   -0.07626943]]]]

```

 Y:  [[[[2.4851248 1.49361  

```
  1.4290358]
```

   

```
  [1.9240153 0.9139378 3.5928857]
```

   

```
  [1.8500228 1.0525136 1.4976646]]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`p` | (*float*): type of $L_p$ norm to use (default=2.0)
`kernel` | (*int*): the size of the window to take a max over
`stride` | (*int*): the stride of the window
`pad` | (*int*): implicit zero padding to be added on both sides
`dilation` | (*int*): parameter that controls the stride of elements in the window
`order` | (*string*): order of blob dimensions (default="NCHW")
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
*Outputs* | 
`Y` | (*Tensor`<float>`*): output tensor


### Code


[caffe2/operators/lp_pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lp_pool_op.cc)

---



## LpPoolGradient

No documentation yet.


### Code


[caffe2/operators/lp_pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lp_pool_op.cc)

---



## MSRAFill

No documentation yet.


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

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


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## MakeTwoClassGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

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


[caffe2/operators/map_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/map_ops.cc)

---



## MarginRankingCriterion


MarginRankingCriterion takes two input data X1 (Tensor), X2 (Tensor), and label Y (Tensor) to produce the loss (Tensor) where the loss function, loss(X1, X2, Y) = max(0, -Y * (X1 - X2) + margin), is applied to the tensor elementwise.
 If y == 1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for y == -1.



### Interface


---------- | ----------
*Arguments* | 
`margin` | The margin value as a float. Default is 1.0.
*Inputs* | 
`X1` | The left input vector as a 1-dim TensorCPU.
`X2` | The right input vector as a 1-dim TensorCPU.
`Y` | The label as a 1-dim TensorCPU with int value of 1 or -1.
*Outputs* | 
`loss` | The output loss with the same dimensionality as X1.


### Code


[caffe2/operators/margin_ranking_criterion_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/margin_ranking_criterion_op.cc)

---



## MarginRankingCriterionGradient


MarginRankingCriterionGradient takes both X1, X2, Y and dY and uses them to update dX1, and dX2 according to the chain rule and derivatives of the loss function.



### Code


[caffe2/operators/margin_ranking_criterion_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/margin_ranking_criterion_op.cc)

---



## MatMul


Matrix multiplication $Y = A * B$, where  `A`  has size (M x K),  `B`  has size (K x N), and  `Y`  will have a size (M x N). To transpose  `A`  or  `B`  before multiplication, pass 1 to the  `trans_a`  and/or  `trans_b`  arguments, which separate the first and second dimensions of the respective matrices using  `axis_a`  and  `axis_b` .
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/matmul_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/matmul_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "MatMul",
    ["A", "B"],
    ["Y"],
```

 )  workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.float32)) workspace.FeedBlob("B", np.random.randint(10, size=(3,3)).astype(np.float32)) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))  ```    **Result**    ```  A: [[1. 8. 3.]  [6. 4. 4.]  [5. 4. 7.]] B: [[4. 0. 3.]  [3. 1. 1.]  [8. 5. 8.]] Y: [[52. 23. 35.]  [68. 24. 54.]  [88. 39. 75.]]  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`axis_a` | *(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `A`.
`axis_b` | *(type: int; default: 1)* Exclusive axis that divides the first and second dimension of matrix `B`.
`trans_a` | *(type: int; default: 0)* Pass 1 to transpose `A` before multiplication and after the dimension adjustment using `axis_a`.
`trans_b` | *(type: int; default: 0)* Pass 1 to transpose `B` before multiplication and after the dimension adjustment using `axis_b`.
*Inputs* | 
`A` | *(type: Tensor`<float>`)* 2D matrix of size (M x K).
`B` | *(type: Tensor`<float>`)* 2D matrix of size (K x N).
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* 2D matrix of size (M x N).


### Code


[caffe2/operators/matmul_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/matmul_op.cc)

---



## Max


Element-wise max of an arbitrary number of input tensors. This operation can be performed in-place, by using the first input blob as the output blob. All inputs must have the same shape and data type, and the output will have the same shape as the inputs.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Max",
    ["X", "Y", "Z"],
    ["X"],
```

 )  workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32)) workspace.FeedBlob("Y", (np.random.rand(3,3)).astype(np.float32)) workspace.FeedBlob("Z", (np.random.rand(3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) print("Y:", workspace.FetchBlob("Y")) print("Z:", workspace.FetchBlob("Z")) workspace.RunOperatorOnce(op) print("Max:", workspace.FetchBlob("X"))   ```    **Result**    ```   X: [[0.4496477 

```
  0.07061381 0.7139333 ]
```

  [0.83203 

```
    0.05970785 0.72786295]
```

  [0.75988126 0.04601283 0.32820013]] Y: [[0.05683139 0.16872478 0.671098 

```
  ]
```

  [0.70739156 0.09878621 0.03416285]  [0.34087983 0.94986707 0.67263436]] Z: [[0.48051122 0.07141234 0.85264146]  [0.77086854 0.22082241 0.13154659]  [0.42401117 0.995431  

```
  0.4263775 ]]
```

 Max: [[0.48051122 0.16872478 0.85264146]  [0.83203 

```
    0.22082241 0.72786295]
```

  [0.75988126 0.995431  

```
  0.67263436]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X, Y, ...` | *(type: Tensor`<Ord>`)* List of input tensors with the same shape.
*Outputs* | 
`M` | *(type: Tensor`<Ord>`)* Output tensor with same dimensions as input(s).Contains the maximum valued element at each location.


### Code


[caffe2/operators/minmax_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc)

---



## MaxGradient

No documentation yet.


### Code


[caffe2/operators/minmax_gradient_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_gradient_ops.cc)

---



## MaxPool

MaxPool  consumes an input blob and applies max pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Max pooling consists of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "MaxPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04 

```
  1.1088650e+00
    -2.1476576e+00 -3.5070452e-01]
```

   

```
  [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
```

   

```
    1.2798511e+00 -3.2214901e-01]
```

   

```
  [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
    -9.6424848e-02 -3.9696163e-01]
```

   

```
  [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
    -9.1898698e-01 -1.9609968e-01]
```

   

```
  [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
    -8.1293899e-01 -7.8803545e-01]
```

   

```
  [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
```

   

```
    7.2035044e-01  2.7642491e-01]]]]

```

 Y:  [[[[-0.28534958 

```
  1.108865    1.2798511 ]
```

   

```
  [ 1.6845188  -0.266332   -0.09642485]
```

   

```
  [ 2.4605224   0.13020347  0.72035044]]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool1D

MaxPool1D  consumes an input blob and applies max pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Max pooling consists of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "MaxPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04 

```
  1.1088650e+00
    -2.1476576e+00 -3.5070452e-01]
```

   

```
  [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
```

   

```
    1.2798511e+00 -3.2214901e-01]
```

   

```
  [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
    -9.6424848e-02 -3.9696163e-01]
```

   

```
  [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
    -9.1898698e-01 -1.9609968e-01]
```

   

```
  [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
    -8.1293899e-01 -7.8803545e-01]
```

   

```
  [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
```

   

```
    7.2035044e-01  2.7642491e-01]]]]

```

 Y:  [[[[-0.28534958 

```
  1.108865    1.2798511 ]
```

   

```
  [ 1.6845188  -0.266332   -0.09642485]
```

   

```
  [ 2.4605224   0.13020347  0.72035044]]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool1DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## MaxPool2D

MaxPool2D  consumes an input blob and applies max pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Max pooling consists of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "MaxPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04 

```
  1.1088650e+00
    -2.1476576e+00 -3.5070452e-01]
```

   

```
  [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
```

   

```
    1.2798511e+00 -3.2214901e-01]
```

   

```
  [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
    -9.6424848e-02 -3.9696163e-01]
```

   

```
  [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
    -9.1898698e-01 -1.9609968e-01]
```

   

```
  [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
    -8.1293899e-01 -7.8803545e-01]
```

   

```
  [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
```

   

```
    7.2035044e-01  2.7642491e-01]]]]

```

 Y:  [[[[-0.28534958 

```
  1.108865    1.2798511 ]
```

   

```
  [ 1.6845188  -0.266332   -0.09642485]
```

   

```
  [ 2.4605224   0.13020347  0.72035044]]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool2DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## MaxPool3D

MaxPool3D  consumes an input blob and applies max pooling across the the blob according to kernel sizes, stride sizes, pad lengths and dilation. Max pooling consists of taking the maximum value of a subset of the input tensor according to the kernel size and downsampling the data into the output blob for further processing. The  `brew`  module has a wrapper for this operator for use in a  `ModelHelper`  object.
 Pooling layers reduce the spatial dimensionality of the input blob. Each of the output blob's dimensions will reduce according to:  $$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "MaxPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
```

 )  workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n") workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))  ```    **Result**    ```  X:  [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04 

```
  1.1088650e+00
    -2.1476576e+00 -3.5070452e-01]
```

   

```
  [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
```

   

```
    1.2798511e+00 -3.2214901e-01]
```

   

```
  [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
    -9.6424848e-02 -3.9696163e-01]
```

   

```
  [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
    -9.1898698e-01 -1.9609968e-01]
```

   

```
  [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
    -8.1293899e-01 -7.8803545e-01]
```

   

```
  [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
```

   

```
    7.2035044e-01  2.7642491e-01]]]]

```

 Y:  [[[[-0.28534958 

```
  1.108865    1.2798511 ]
```

   

```
  [ 1.6845188  -0.266332   -0.09642485]
```

   

```
  [ 2.4605224   0.13020347  0.72035044]]]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output data tensor.


### Code


[caffe2/operators/pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc)

---



## MaxPool3DGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## MaxPoolGradient

No documentation yet.


### Code


[caffe2/operators/pool_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_gradient_op.cc)

---



## MaxPoolWithIndex


 

```
    MaxPoolWithIndex consumes an input blob X and applies max pooling across the
    blob according to kernel sizes, stride sizes and pad lengths defined by the
    ConvPoolOpBase operator. It also produces an explicit mask that defines the
    location that all maximum values were found, which is re-used in the
    gradient pass. This op is deterministic.
```

   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. 
*Outputs* | 
`Y` | Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
`Index` | Mask of location indices of the found maximum values,  used in the gradient operator to accumulate dY values to the appropriate locations in Y


### Code


[caffe2/operators/max_pool_with_index.cu](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/max_pool_with_index.cu)

---



## MaxPoolWithIndexGradient

No documentation yet.


### Code


[caffe2/operators/max_pool_with_index.cu](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/max_pool_with_index.cu)

---



## Mean


Element-wise mean of an arbitrary number of input tensors. This operation can be performed in-place, by using the first input blob as the output blob. All inputs must have the same shape and data type, and the output will have the same shape as the inputs.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mean_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mean_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Mean",
    ["X", "Y", "Z"],
    ["X"],
```

 )  workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32)) workspace.FeedBlob("Y", (np.random.rand(3,3)).astype(np.float32)) workspace.FeedBlob("Z", (np.random.rand(3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) print("Y:", workspace.FetchBlob("Y")) print("Z:", workspace.FetchBlob("Z")) workspace.RunOperatorOnce(op) print("Mean:", workspace.FetchBlob("X"))   ```    **Result**    ```   X: [[0.6035237 

```
  0.5305746  0.6298913 ]
```

  [0.9169737 

```
  0.01280353 0.16286302]
```

  [0.6017664 

```
  0.9946255  0.05128575]]
```

 Y: [[0.07544111 0.45371833 0.08460239]  [0.9708728 

```
  0.7422064  0.7933344 ]
```

  [0.97671497 0.3411384 

```
  0.73818344]]
```

 Z: [[0.08837954 0.90187573 0.46734726]  [0.6308827 

```
  0.8719029  0.39888734]
```

  [0.90059936 0.92883426 0.5695987 ]] Mean: [[0.25578147 0.6287229 

```
  0.39394698]
```

  [0.8395764 

```
  0.5423043  0.45169494]
```

  [0.8263602 

```
  0.75486606 0.45302266]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X, Y, ...` | *(type: Tensor`<Ord>`)* List of input tensors with the same shape.
*Outputs* | 
`M` | *(type: Tensor`<Ord>`)* Output tensor with the same dimensions as inputs. Contains the mean values of the input tensors calculated element-wise.


### Code


[caffe2/operators/mean_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mean_op.cc)

---



## MeanGradient

No documentation yet.


### Code


[caffe2/operators/mean_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mean_op.cc)

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


[caffe2/operators/prepend_dim_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prepend_dim_op.cc)

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


[caffe2/operators/merge_id_lists_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/merge_id_lists_op.cc)

---



## MergeMultiListFeatureTensors

Merge given multi-feature tensors with list features into one.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`in1_keys` | .keys
`in1_values_lengths` | .values.lengths
`in1_values_values` | .values.values
*Outputs* | 
`out_lengths` | .lengths
`out_keys` | .keys
`out_values_lengths` | .values.lengths
`out_values_values` | .values.values


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeMultiListFeatureTensorsGradient

Explode given multi-feature tensors with list features into many.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`in1_values_lengths` | .values.lengths
`out_values_values_grad` | .values.values_grad
*Outputs* | 
`in1_values_values_grad` | .values.values_grad


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeMultiMapFeatureTensors

Merge given multi-feature tensors with map features into one.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`in1_keys` | .keys
`in1_values_lengths` | .values.lengths
`in1_values_keys` | .values.keys
`in1_values_values` | .values.values
*Outputs* | 
`out_lengths` | .lengths
`out_keys` | .keys
`out_values_lengths` | .values_lengths
`out_values_keys` | .values.keys
`out_values_values` | .values.values


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeMultiMapFeatureTensorsGradient

Explode given multi-feature tensors with map features into many.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`in1_values_lengths` | .values.lengths
`out_values_values_grad` | .values.values_grad
*Outputs* | 
`in1_values_values_grad` | .values.values_grad


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeMultiScalarFeatureTensors

Merge given multi-feature tensors with scalar features into one.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`in1_keys` | .keys
`in1_values` | .values
*Outputs* | 
`out_lengths` | .lengths
`out_keys` | .keys
`out_values` | .values


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeMultiScalarFeatureTensorsGradient

Explode given multi-feature tensors with scalar features into many.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`out_values_grad` | .values_grad
*Outputs* | 
`in1_values_grad` | .values_grad


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeSingleListFeatureTensors

Merge given single-feature tensors with list features into one multi-feature tensor.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Arguments* | 
`feature_ids` | feature ids
*Inputs* | 
`in1_lengths` | .lengths
`in1_values` | .values
`in1_presence` | .presence
*Outputs* | 
`out_lengths` | .lengths
`out_keys` | .keys
`out_values_lengths` | .values.lengths
`out_values_values` | .values.values


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeSingleListFeatureTensorsGradient

Explode multi-feature tensors with list features into single-feature tensors.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`in1_presence` | .presence
`out_values_values` | .values.values_grad
*Outputs* | 
`out1_values` | .values_grad


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeSingleMapFeatureTensors

Merge given single-feature tensors with map features into one multi-feature tensor.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Arguments* | 
`feature_ids` | feature ids
*Inputs* | 
`in1_lengths` | .lengths
`in1_keys` | .keys
`in1_values` | .values
`in1_presence` | .presence
*Outputs* | 
`out_lengths` | .lengths
`out_keys` | .keys
`out_values_lengths` | .values.lengths
`out_values_keys` | .values.keys
`out_values_values` | .values.values


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeSingleMapFeatureTensorsGradient

Explode given multi-feature tensors with map features into multiple single-feature tensor.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_lengths` | .lengths
`in1_presence` | .presence
`out_values_values_grad` | .values.values_grad
*Outputs* | 
`in1_values_grad` | .values_grad


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeSingleScalarFeatureTensors

Merge given single-feature tensors with scalar features into one multi-feature tensor.
 

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Arguments* | 
`feature_ids` | feature ids
*Inputs* | 
`in1` | 
`in1_presence` | .presence
*Outputs* | 
`out_lengths` | .lengths
`out_keys` | .keys
`out_values` | .values


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## MergeSingleScalarFeatureTensorsGradient

Explode multi-feature tensor of scalar features into one or moresingle-feature tensors  

```
  Single-feature representation:
  - scalar features:
    <feature full name> T
  - list features:
    <feature full name>.lengths int32
    <feature full name>.values T
  - map features:
    <feature full name>.lengths int32
    <feature full name>.keys K
    <feature full name>.values V

  Missing values are set to zero, and value presence flag is set accordingly:
    <feature full name>.presence bool

  Multi-feature representation:
  - scalar features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values T
  - list features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.values T
  - map features:
    <feature type>.lengths int32
    <feature type>.keys int64
    <feature type>.values.lengths int32
    <feature type>.values.keys K
    <feature type>.values.values V

  You can read more about representing batches of lists and maps here:
  https://our.intern.facebook.com/intern/dex/caffe2/sparse-operations/
```




### Interface


---------- | ----------
*Inputs* | 
`in1_presence` | .presence
`.values_grad` | .values_grad
*Outputs* | 
`in1_grad` | _grad of inputs


### Code


[caffe2/operators/feature_maps_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/feature_maps_ops.cc)

---



## Min


Element-wise min of an arbitrary number of input tensors. This operation can be performed in-place, by using the first input blob as the output blob. All inputs must have the same shape and data type, and the output will have the same shape as the inputs.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Min",
    ["X", "Y", "Z"],
    ["X"],
```

 )  workspace.FeedBlob("X", (np.random.rand(2,2)).astype(np.float32)) workspace.FeedBlob("Y", (np.random.rand(2,2)).astype(np.float32)) workspace.FeedBlob("Z", (np.random.rand(2,2)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) print("Y:", workspace.FetchBlob("Y")) print("Z:", workspace.FetchBlob("Z")) workspace.RunOperatorOnce(op) print("Min:", workspace.FetchBlob("X"))   ```    **Result**    ```   X: [[0.32731926 0.4939747 ]  [0.29242373 0.43460014]] Y: [[0.40928316 0.916115 

```
  ]
```

  [0.77526504 0.29339448]] Z: [[0.7899794 

```
  0.90335774]
```

  [0.82599413 0.2843068 ]] Min: [[0.32731926 0.4939747 ]  [0.29242373 0.2843068 ]]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X, Y, ...` | *(type: Tensor`<Ord>`)* List of input tensors with the same shape.
*Outputs* | 
`M` | *(type: Tensor`<Ord>`)* Output tensor with same dimensions as input(s).Contains the minimum valued element at each location.


### Code


[caffe2/operators/minmax_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_ops.cc)

---



## MinGradient

No documentation yet.


### Code


[caffe2/operators/minmax_gradient_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/minmax_gradient_ops.cc)

---



## Mod


Element-wise modulo operation. Each element in the output is the modulo result of the corresponding element in the input data. The divisor of the modulo is provided by the  `divisor`  argument.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mod_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mod_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Mod",
    ["X"],
    ["Y"],
    divisor=10
```

 )  workspace.FeedBlob("X", (np.random.randint(100, size=(5,5)))) print("X before running op:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("X after running op:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X before running op: [[56 22 43 13 60]  [ 4 55 58 10 45]  [64 66 

```
  4  3 66]
```

  [10 36 47 52 78]  [91 

```
  4 36 47 95]]
```

 X after running op: [[6 2 3 3 0]  [4 5 8 0 5]  [4 6 4 3 6]  [0 6 7 2 8]  [1 4 6 7 5]]    ```    </details>  


### Interface


---------- | ----------
*Arguments* | 
`divisor` | *(type: int; default: 0)* Divisor of the modulo operation (must be >= 1).
`sign_follow_divisor` | *(type: bool; default: False)* If true, sign of output matches divisor, else if false, sign follows dividend.
*Inputs* | 
`X` | *(type: Tensor`<int>`)* Input tensor with int32 or int64 data.
*Outputs* | 
`Y` | *(type: Tensor`<int>`)* Output tensor of data with modulo operation applied.


### Code


[caffe2/operators/mod_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/mod_op.cc)

---



## Moments


 

```
  Computes the mean and variance of the input tensor's element along the
  provided axes. The resulted tensor has the same rank as the input if keepdims
  equals True.
  If keepdims equals False, then the resulted tensor have the reduced dimension
  pruned.
```




### Interface


---------- | ----------
*Arguments* | 
`axes` | A list of integers, along which to reduce. If axes is not provided, the op computes the element-wise mean and variance.
`keepdims` | Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`mean` | Reduced mean tensor.
`variance` | Reduced variance tensor.


### Code


[caffe2/operators/moments_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/moments_op.cc)

---



## MomentsGradient

No documentation yet.


### Code


[caffe2/operators/moments_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/moments_op.cc)

---



## Mul


Performs element-wise binary multiplication (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "Mul",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([[1,2],[3,4]])) workspace.FeedBlob("B", np.array([[5,6],[7,8]])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [[1 2]  [3 4]] B: [[5 6]  [7 8]] C: [[ 5 12]  [21 32]]   ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting
`axis` | *(type: int; default: -1)* Axis to concatenate on.
*Inputs* | 
`A` | *(type: Tensor`<float>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size as A.
*Outputs* | 
`C` | *(type: Tensor`<float>`)* Output tensor with same dimensions and type as A.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## MulGradient

No documentation yet.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

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


[caffe2/operators/multi_class_accuracy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/multi_class_accuracy_op.cc)

---



## NCHW2NHWC


The operator switches the order of data in a tensor from NCHW- sample index N, channels C, height H and width W, to the NHWC order (this is for 2D images).
In general, this operator switches the order of data in a tensor from N C H_1 ... H_k to N H_1 ... H_k C for k-dimensional features, and currently supports k=1, 2, and 3.



### Interface


---------- | ----------
*Inputs* | 
`data` | The input data (Tensor) in the NCHW order.
*Outputs* | 
`output` | The output tensor (Tensor) in the NHWC order.


### Code


[caffe2/operators/order_switch_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/order_switch_ops.cc)

---



## NE


Performs element-wise not equal to comparison  **!=**  (with limited broadcast support).
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "NE",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3])) workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```  A: [ 1 

```
  5  2  9 12  3]
```

 B: [ 1 

```
  3  4  9 12  8]
```

 C: [False 

```
  True  True False False  True]
```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## NGramFromCategorical

No documentation yet.


### Code


[caffe2/operators/ngram_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/ngram_ops.cc)

---



## NHWC2NCHW


The operator switches the order of data in a tensor from NHWC- sample index N, height H, width H and channels C, to the NCHW order (this is for 2D images).
In general, this operator switches the order of data in a tensor from N H_1 ...
H_k C to N C H_1 ... H_k for k-dimensional features, and currently supports k=1, 2, and 3.



### Interface


---------- | ----------
*Inputs* | 
`data` | The input data (Tensor) in the NHWC order.
*Outputs* | 
`output` | The output tensor (Tensor) in the NCHW order.


### Code


[caffe2/operators/order_switch_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/order_switch_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## NegateGradient


NegagteGradient operator in forward pass simply copies input to the output, and in backward pass, flips the sign of the output gradient 


### Code


[caffe2/operators/negate_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/negate_gradient_op.cc)

---



## Negative


Computes the element-wise negative of the input.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/negative_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/negative_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Negative",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", (np.random.rand(3,3).astype(np.float32))) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))  ```    **Result**    ```  X: [[0.83296907 0.61407167 0.32562155]  [0.59304523 0.03111175 0.29365504]  [0.09478621 0.5424558 

```
  0.73940724]]
```

 Y: [[-0.83296907 -0.61407167 -0.32562155]  [-0.59304523 -0.03111175 -0.29365504]  [-0.09478621 -0.5424558 

```
  -0.73940724]]
```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* 1D input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* 1D output tensor.


### Code


[caffe2/operators/negative_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/negative_op.cc)

---



## Normalize


Given a matrix, apply L2-normalization along the specified dimension.



### Interface


---------- | ----------
*Arguments* | 
`axis` | axis to normalize


### Code


[caffe2/operators/normalize_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/normalize_op.cc)

---



## NormalizeGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`axis` | axis to normalize


### Code


[caffe2/operators/normalize_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/normalize_op.cc)

---



## NormalizeL1


Given a matrix, apply L1-normalization along the specified axis.



### Interface


---------- | ----------
*Arguments* | 
`axis` | axis to normalize


### Code


[caffe2/operators/normalize_l1_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/normalize_l1_op.cc)

---



## NormalizePlanarYUV

No documentation yet.


### Code


[caffe2/operators/norm_planar_yuv_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/norm_planar_yuv_op.cc)

---



## Not


Performs element-wise negation on input tensor  `X` .
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator( "Not", ["X"], ["Y"], )  workspace.FeedBlob("X", (np.random.rand(3, 3) > 0.5)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[ True False False] [False False False] [ True 

```
  True  True]]
```

 Y: [[False 

```
  True  True]
```

 [ True 

```
  True  True]
```

 [False False False]]   ```   </details>      


### Interface


---------- | ----------
*Inputs* | 
`X` | *(Tensor`<bool>`)* Input tensor.
*Outputs* | 
`Y` | *(Tensor`<bool>`)* Negated output tensor.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## NumpyTile

No documentation yet.


### Interface


---------- | ----------
*Inputs* | 
`data` | The input tensor.
`repeats` | 1-D Tensor specifying how many times to repeat each axis.
*Outputs* | 
`tiled_data` | Tensor that will contain input replicated along the given axis.


### Code


[caffe2/operators/numpy_tile_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/numpy_tile_op.cc)

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
`body` | Net executed on each iteration
`has_trip_count` | Whether to use the trip count input
`has_cond` | Whether to use the condition input
`save_scopes` | Whether to save the scopes across iterations, as in for backprop
`disable_scopes` | Do not create new scopes. Use this only if you're certain there will be no name collision, for example if you're converting from a fully-SSA IR
*Inputs* | 
`max_trip_count` | Number of iterations to go out to. Used if the flag has_trip_count is True.
`first_iter_condition` | Dynamic condition value for the first iteration. For all subsequent iterations, the condition from the body graph is used. This input is used if the flag has_cond is true.


### Code


[caffe2/operators/onnx_while_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/onnx_while_op.cc)

---



## OneHot


The  *OneHot*  op accepts two inputs  *indices*  and  *index_size_tensor* , and produces a single output  *one_hots* . 

```
  For each index in *indices* the op creates a one-hot row in *one_hots* of length *index_size_tensor* where all entries are zero except the entry at the index is 1. The size of *one_hots* is *len(indices)* x *index_size_tensor*.

```

 Github Links:  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "OneHot",
    ["indices", "index_size_tensor"],
    ["one_hots"],
```

 )  workspace.FeedBlob("indices", np.array([0,1,2,3,4]).astype(np.long)) print("indices:\n", workspace.FetchBlob("indices"))  workspace.FeedBlob("index_size_tensor", np.array([5]).astype(np.long)) print("index_size_tensor:\n", workspace.FetchBlob("index_size_tensor"))  workspace.RunOperatorOnce(op) print("one_hots: \n", workspace.FetchBlob("one_hots"))   ```    **Result**    ```   indices:  [0 1 2 3 4] index_size_tensor:  [5] one_hots:  [[1. 0. 0. 0. 0.]  [0. 1. 0. 0. 0.]  [0. 0. 1. 0. 0.]  [0. 0. 0. 1. 0.]  [0. 0. 0. 0. 1.]]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`indices` | The active index for each example in the batch.
`index_size_tensor` | Scalar with the size of the index. Must be in CPU context
*Outputs* | 
`one_hots` | Matrix of size len(indices) x index_size


### Code


[caffe2/operators/one_hot_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/one_hot_ops.cc)

---



## Onnxifi


 

```
    The Onnxifi operator is a black-box operator to lower the computation to Onnxifi backend
```

     


### Interface


---------- | ----------
*Arguments* | 
`onnx_model` | (string default="") Serialized ONNX model to be converted to backend representation
`initializers` | Initialization pair indicating the mapping of the name between NetDef and ONNX model


### Code


[caffe2/operators/onnxifi_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/onnxifi_op.cc)

---



## Or


Performs element-wise logical operation  **or**  (with limited broadcast support).
Both input operands should be of type  `bool` .
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "Or",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5)) workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5)) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [[False 

```
  True  True]
```

  [False 

```
  True  True]
```

  [ True 

```
  True  True]]
```

 B: [[False 

```
  True False]
```

  [ True 

```
  True  True]
```

  [False 

```
  True False]]
```

 C: [[False 

```
  True  True]
```

  [ True 

```
  True  True]
```

  [ True 

```
  True  True]]

```

 ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor of booleans. Has same dimensions as input `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## PRelu


 The  *PRelu*  op takes input data tensor $X$, an input slope tensor $slope$, and produces one output tensor $Y$ of the same shape as $X.$ The op performs the element wise  *PRelu*  operation, defined as  $$y=prelu(x) =\begin{cases}slope * x & x < 0\\x & otherwise\end{cases}$$  Note, is slope is size 1, the value is shared across the channels, otherwise $X$ and $slope$ must be the same shape. See [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification]( [https://arxiv.org/abs/1502.01852)](https://arxiv.org/abs/1502.01852))  for more information.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "PRelu",
    ["X","Slope"],
    ["Y"],
```

 )  workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32)) print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.FeedBlob("Slope", np.array([0.1]).astype(np.float32)) print("Slope:\n", workspace.FetchBlob("Slope"), "\n")  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[ 0.3957382 

```
  -0.19725518 -0.26991343]
```

  [ 1.5513182 

```
  -0.27427664 -0.14584002]
```

  [-0.4121164  

```
  0.9292345   0.96426094]]

```

 Slope:  [0.1]  Y:  [[ 0.3957382 

```
  -0.01972552 -0.02699134]
```

  [ 1.5513182 

```
  -0.02742766 -0.014584  ]
```

  [-0.04121164 

```
  0.9292345   0.96426094]]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | Input tensor of data to be operated on.
`Slope` | 1D input slope tensor. If `Slope` is of size 1, the value is shared across different channels
*Outputs* | 
`Y` | Output tensor, with same shape as $X$.


### Code


[caffe2/operators/prelu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.cc)

---



## PReluGradient


 PReluGradient takes both Y and dY and uses this to update dX and dW according to the chain rule and derivatives of the rectified linear function.
 


### Code


[caffe2/operators/prelu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prelu_op.cc)

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


[caffe2/operators/pack_rnn_sequence_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pack_rnn_sequence_op.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## PackSegments

Map N dim tensor to N+1 dim based on length blob. Sequences that     are shorter than the longest sequence are padded with zeros.


### Interface


---------- | ----------
*Arguments* | 
`max_length` | The pre-defined max_length for the packed segments
`pad_minf` | Padding number in the packed segments. Use true to pad     -infinity, otherwise pad zeros
`return_presence_mask` | bool whether to return presence mask, false by default
*Inputs* | 
`lengths` | 1-d int/long tensor contains the length in each of the output.
`tensor` | N dim Tensor.
*Outputs* | 
`packed_tensor` | N + 1 dim Tensorwhere dim(1) is the max length, dim(0) is the batch size.
`presence_mask` | 2 dim boolean tensor, false where packed_tensor is padded, true otherwise.


### Code


[caffe2/operators/pack_segments.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pack_segments.cc)

---



## PackedInt8BGRANHWCToNCHWCStylizerPreprocess

No documentation yet.


### Code


[caffe2/operators/stylizer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stylizer_ops.cc)

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


[caffe2/operators/sequence_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc)

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


[caffe2/operators/pad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pad_op.cc)

---



## PadImageGradient

No documentation yet.


### Code


[caffe2/operators/pad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pad_op.cc)

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


[caffe2/operators/rank_loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rank_loss_op.cc)

---



## PairWiseLossGradient

No documentation yet.


### Code


[caffe2/operators/rank_loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rank_loss_op.cc)

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


[caffe2/operators/partition_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/partition_ops.cc)

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


[caffe2/operators/percentile_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/percentile_op.cc)

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


[caffe2/operators/perplexity_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/perplexity_op.cc)

---



## PiecewiseLinearTransform


PiecewiseLinearTransform takes inputs -- predictions, a 2-D or 1-D tensor (Tensor) of size (batch_size x prediction_dimensions). The piecewise linear functions are stored in bounds, slopes and intercepts. The output tensor has the same shape of input  `predictions`  and contains the predictions transformed by the piecewise linear functions. Each column of predictions has its own piecewise linear transformation functions. Therefore the size of piecewise function parameters are pieces x prediction_dimensions, except for binary predictions where only the positive prediction needs them. Note that in each piece, low bound is excluded while high bound is included. Also the piecewise linear function must be continuous.
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
`predictions` | 2-D tensor (Tensor) of size (num_batches x num_classes) containing scores
`bounds (optional)` | See bounds in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.
`slopes (optional)` | See slopes in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.
`intercepts (optional)` | See intercepts in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.
*Outputs* | 
`transforms` | 2-D tensor (Tensor) of size (num_batches x num_classes) containing transformed predictions


### Code


[caffe2/operators/piecewise_linear_transform_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/piecewise_linear_transform_op.cc)

---



## Pow


The  *Pow*  op takes an input data tensor $X$ and an exponent parameter  *exponent* , which can be a scalar or another tensor. As output, it produces a single output data tensor $Y$, where the function $f(x) = x^{exponent}$ has been applied to $X$ elementwise.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Pow",
    ["X", "exponent"],
    ["Y"],
    broadcast=1
```

 )  workspace.FeedBlob("X", np.array([1,2,3,4,5,6]).astype(np.float32)) print("X: ", workspace.FetchBlob("X"))  workspace.FeedBlob("exponent", np.array([2]).astype(np.float32)) print("exponent: ", workspace.FetchBlob("exponent"))  workspace.RunOperatorOnce(op) print("Y: ", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: 

```
  [1. 2. 3. 4. 5. 6.]
```

 exponent: 

```
  [2.]
```

 Y: 

```
  [ 1.  4.  9. 16. 25. 36.]

```

 ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`exponent` | The exponent of the power function. Do not use if setting exponent via input.
`axis` | *(type: int; default: -1)*
`broadcast` | *(type: bool; default: False)*
*Inputs* | 
`X` | Input data blob to be operated on.
`exponent` | Exponent blob containing the exponent(s) for calculation. Do not use if setting exponent via argument.
*Outputs* | 
`Y` | Output data blob with the same shape as the input.


### Code


[caffe2/operators/pow_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.cc)

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


[caffe2/operators/prepend_dim_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/prepend_dim_op.cc)

---



## Print

Logs shape and contents of input tensor to stderr or to a file.


### Interface


---------- | ----------
*Arguments* | 
`to_file` | (bool) if 1, saves contents to the root folder of the current workspace, appending the tensor contents to a file named after the blob name. Otherwise, logs to stderr.
`limit` | (int, default 0) If set, prints the first `limit` elements of tensor. If 0, prints the first `k_limit_default`(1000) elements of tensor
`every_n` | (int, default 1) Print tensor every `every_n` runs
*Inputs* | 
`tensor` | The tensor to print.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/quant_decode_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/quant_decode_op.cc)

---



## QuantDecodeGradient

No documentation yet.


### Code


[caffe2/operators/quant_decode_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/quant_decode_op.cc)

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


[caffe2/operators/rmac_regions_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rmac_regions_op.cc)

---



## Range


Generates an output tensor within the half-open interval $[start, stop)$ (the interval including start but excluding stop).
- The  `start`  input is optional, and defaults to 0 when not set.
- The  `step`  input is optional, and defaults to 1 when not set.
- The type of the  `output`  tensor is determined by the types of inputs used.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Range",
    ["start", "stop", "step"],
    ["output"]
```

 )  workspace.FeedBlob("start", np.array(4, dtype=np.int32)) workspace.FeedBlob("stop", np.array(17, dtype=np.int32)) workspace.FeedBlob("step", np.array(2, dtype=np.int32)) print("start:", workspace.FetchBlob("start")) print("stop:", workspace.FetchBlob("stop")) print("step:", workspace.FetchBlob("step")) workspace.RunOperatorOnce(op) print("output:", workspace.FetchBlob("output"))   ```    **Result**    ```   start: 4 stop: 17 step: 2 output: [ 4 

```
  6  8 10 12 14 16]

```

 ```   </details>         


### Interface


---------- | ----------
*Inputs* | 
`start` | (*Tensor*): [OPTIONAL] scalar tensor containing the start of the interval (inclusive) (default=0)
`stop` | (*Tensor*): scalar tensor containing the end of the interval (exclusive)
`step` | (*Tensor*): [OPTIONAL] scalar tensor specifying the spacing between values (default=1)
*Outputs* | 
`output` | (*Tensor*): 1D tensor of same type as inputs that contains the sequence


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## RangeFill

No documentation yet.


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

---



## Reciprocal


Performs element-wise reciprocal ($\1/x$) of input tensor $X$.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reciprocal_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reciprocal_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Reciprocal",
    ["X"],
    ["Y"],
```

 )  workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[8. 3. 3.]  [4. 0. 0.]  [1. 2. 5.]] Y: [[0.125 0.3333333 

```
  0.3333333 ]
```

  [0.25 

```
  inf        inf       ]
```

  [1  

```
    0.5        0.2       ]]

```

 ```   </details> 


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.


### Code


[caffe2/operators/reciprocal_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reciprocal_op.cc)

---



## ReciprocalGradient

No documentation yet.


### Code


[caffe2/operators/reciprocal_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reciprocal_op.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

---



## ReduceBackMax


Reduces the input tensor along the last dimension of the by applying  **max** .
 Can reduce more than one of the "last" dimensions by setting  `num_reduce_dim` .
 A second (optional) input,  `lengths` , can be passed, which enforces that only a subset of the elements are considered in the max operation.
- If input tensor  `X`  has shape $(d_0, d_1, d_2, ..., d_n)$,  `lengths`  must have shape $(d_0  * d_1 *  d_2  * ... *  d_{n-1})$.
- The values of the  `lengths`  tensor determine how many of the values to consider for each vector in the $d_{n-1}$ dimension.
 For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1]$, then $Y = [max(1,5), max(4,1,8), max(2)] = [5, 8, 2]$   Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceBackMax",
    ["X"],
    ["Y"],
    num_reduce_dim=2
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[[2. 5. 1.]   

```
  [6. 1. 9.]
```

   

```
  [8. 5. 9.]]

  [[5. 7. 8.]
```

   

```
  [9. 9. 6.]
```

   

```
  [6. 5. 0.]]]]
```

 Y: [[9. 9.]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | (*int*): number of dimensions to reduce (default=1)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
`lengths` | (*Tensor`<int>`*): number of elements in each sample
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackMean


Reduces the input tensor along the last dimension of the by applying  **mean** .
 Can reduce more than one of the "last" dimensions by setting  `num_reduce_dim` .
 A second (optional) input,  `lengths` , can be passed, which enforces that only a subset of the elements are considered in the mean operation.
- If input tensor  `X`  has shape $(d_0, d_1, d_2, ..., d_n)$,  `lengths`  must have shape $(d_0  * d_1 *  d_2  * ... *  d_{n-1})$.
- The values of the  `lengths`  tensor determine how many of the values to consider for each vector in the $d_{n-1}$ dimension.
 For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1]$, then $Y = [mean(1,5), mean(4,1,8), mean(2)] = [3, 4.333, 2]$   Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceBackMean",
    ["X"],
    ["Y"],
    num_reduce_dim=2
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[[5. 9. 0.]   

```
  [8. 4. 0.]
```

   

```
  [2. 2. 4.]]

  [[9. 0. 9.]
```

   

```
  [7. 9. 7.]
```

   

```
  [1. 0. 2.]]]]
```

 Y: [[3.7777777 4.888889 ]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | (*int*): number of dimensions to reduce (default=1)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
`lengths` | (*Tensor`<int>`*): number of elements in each sample
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackMeanGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackSum


Reduces the input tensor along the last dimension of the by applying  **sum** .
 Can reduce more than one of the "last" dimensions by setting  `num_reduce_dim` .
 A second (optional) input,  `lengths` , can be passed, which enforces that only a subset of the elements are considered in the sum operation.
- If input tensor  `X`  has shape $(d_0, d_1, d_2, ..., d_n)$,  `lengths`  must have shape $(d_0  * d_1 *  d_2  * ... *  d_{n-1})$.
- The values of the  `lengths`  tensor determine how many of the values to consider for each vector in the $d_{n-1}$ dimension.
 For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1]$, then $Y = [sum(1,5), sum(4,1,8), sum(2)] = [6, 13, 2]$   Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceBackSum",
    ["X"],
    ["Y"],
    num_reduce_dim=2
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[[2. 7. 7.]   

```
  [1. 1. 0.]
```

   

```
  [9. 7. 2.]]

  [[6. 6. 4.]
```

   

```
  [1. 2. 6.]
```

   

```
  [6. 6. 3.]]]]
```

 Y: [[36. 40.]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | (*int*): number of dimensions to reduce (default=1)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
`lengths` | (*Tensor`<int>`*): number of elements in each sample
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceBackSumGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMax


Reduces the input tensor along the last dimension of the by applying  **max** .
 Can reduce more than one of the "first" dimensions by setting  `num_reduce_dim` .
 A second (optional) input,  `lengths` , can be passed, which enforces that only a subset of the elements are considered in the max operation.
- If input tensor  `X`  has shape $(d_0, d_1, d_2, ..., d_n)$,  `lengths`  must have shape $(d_1  * d_2 *  ... * d_{n})$.
- The values of the  `lengths`  tensor determine how many of the values to consider for each vector in the $d_{0}$ dimension.
 For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1,2]$, then $Y = [max(1,4), max(5,1,7), max(2), max(9,2)] = [4, 7, 2, 9]$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceFrontMax",
    ["X"],
    ["Y"],
    num_reduce_dim=2
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[2. 8. 1.]  

```
  [9. 6. 6.]
  [7. 7. 0.]]

```

  [[4. 3. 9.]  

```
  [9. 2. 7.]
  [6. 4. 7.]]]
```

 Y: [9. 8. 9.]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | (*int*): number of dimensions to reduce (default=1)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
`lengths` | (*Tensor`<int>`*): number of elements in each sample
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMean


Reduces the input tensor along the last dimension of the by applying  **mean** .
 Can reduce more than one of the "first" dimensions by setting  `num_reduce_dim` .
 A second (optional) input,  `lengths` , can be passed, which enforces that only a subset of the elements are considered in the mean operation.
- If input tensor  `X`  has shape $(d_0, d_1, d_2, ..., d_n)$,  `lengths`  must have shape $(d_1  * d_2 *  ... * d_{n})$.
- The values of the  `lengths`  tensor determine how many of the values to consider for each vector in the $d_{0}$ dimension.
 For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1,2]$, then $Y = [mean(1,4), mean(5,1,7), mean(2), mean(9,2)] = [2.5, 4.333, 2, 5.5]$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceFrontMean",
    ["X"],
    ["Y"],
    num_reduce_dim=2
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[5. 0. 9.]  

```
  [4. 1. 1.]
  [9. 0. 8.]]

```

  [[2. 6. 7.]  

```
  [6. 2. 6.]
  [0. 4. 5.]]]
```

 Y: [4.3333335 

```
    2.1666667     6.]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | (*int*): number of dimensions to reduce (default=1)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
`lengths` | (*Tensor`<int>`*): number of elements in each sample
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontMeanGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontSum


Reduces the input tensor along the last dimension of the by applying  **sum** .
 Can reduce more than one of the "first" dimensions by setting  `num_reduce_dim` .
 A second (optional) input,  `lengths` , can be passed, which enforces that only a subset of the elements are considered in the sum operation.
- If input tensor  `X`  has shape $(d_0, d_1, d_2, ..., d_n)$,  `lengths`  must have shape $(d_1  * d_2 *  ... * d_{n})$.
- The values of the  `lengths`  tensor determine how many of the values to consider for each vector in the $d_{0}$ dimension.
 For example, if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$ and $lengths = [2,3,1,2]$, then $Y = [sum(1,4), sum(5,1,7), sum(2), sum(9,2)] = [2.5, 4.333, 2, 5.5]$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceFrontSum",
    ["X"],
    ["Y"],
    num_reduce_dim=2
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[4. 1. 1.]  

```
  [0. 6. 7.]
  [7. 8. 6.]]

```

  [[5. 7. 7.]  

```
  [0. 1. 6.]
  [2. 9. 0.]]]
```

 Y: [18. 32. 27.]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`num_reduce_dims` | (*int*): number of dimensions to reduce (default=1)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
`lengths` | (*Tensor`<int>`*): number of elements in each sample
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

---



## ReduceFrontSumGradient

No documentation yet.


### Code


[caffe2/operators/reduction_front_back_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_front_back_ops.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## ReduceFrontWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## ReduceL1


Computes the  **L1 norm**  of the input tensor's elements along the provided  `axes` . The resulting tensor has the same rank as the input if the  `keepdims`  argument equals 1 (default). If  `keepdims`  is set to 0, then the  `axes`  dimensions are pruned.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceL1",
    ["X"],
    ["Y"],
    axes=(0,1),
    keepdims=0
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[[ 2. 

```
  7.  6.  4.  5.]
```

   

```
  [ 2.  1.  9.  8.  7.]
```

   

```
  [ 4.  9.  1.  0.  0.]
```

   

```
  [ 6.  4.  0.  8.  1.]
```

   

```
  [ 1.  7.  1.  0.  2.]]

  [[ 5.  8.  1.  7.  7.]
```

   

```
  [ 4.  5.  6.  5.  4.]
```

   

```
  [ 1.  9.  6.  6.  3.]
```

   

```
  [ 6.  6.  8.  8.  4.]
```

   

```
  [ 2.  3.  5.  8.  1.]]]]

```

 Y: [[ 

```
  7.  15.   7.  11.  12.]
```

  [ 

```
  6.   6.  15.  13.  11.]
```

  [ 

```
  5.  18.   7.   6.   3.]
```

  [ 12. 

```
  10.   8.  16.   5.]
```

  [ 

```
  3.  10.   6.   8.   3.]]

```

 ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`axes` | (*Tuple(int)*): list of axes to reduce
`keepdims` | (*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceL1Gradient

No documentation yet.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceL2


Computes the  **L2 norm**  of the input tensor's elements along the provided  `axes` . The resulting tensor has the same rank as the input if the  `keepdims`  argument equals 1 (default). If  `keepdims`  is set to 0, then the  `axes`  dimensions are pruned.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceL2",
    ["X"],
    ["Y"],
    axes=(0,1),
    keepdims=0
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[[ 8. 

```
  0.  2.  5.  1.]
```

   

```
  [ 1.  3.  0.  4.  0.]
```

   

```
  [ 1.  3.  6.  7.  7.]
```

   

```
  [ 6.  9.  8.  4.  6.]
```

   

```
  [ 6.  1.  5.  7.  3.]]

  [[ 2.  4.  6.  2.  8.]
```

   

```
  [ 1.  1.  8.  0.  8.]
```

   

```
  [ 5.  9.  0.  3.  2.]
```

   

```
  [ 1.  7.  3.  7.  3.]
```

   

```
  [ 6.  8.  9.  8.  7.]]]]

```

 Y: [[ 

```
  8.24621105   4.           6.3245554    5.38516474   8.06225777]
```

  [ 

```
  1.41421354   3.1622777    8.           4.           8.        ]
```

  [ 

```
  5.09901953   9.48683262   6.           7.6157732    7.28010988]
```

  [ 

```
  6.08276272  11.40175438   8.54400349   8.06225777   6.70820379]
```

  [ 

```
  8.48528099   8.06225777  10.29563046  10.63014603   7.6157732 ]]

```

 ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`axes` | (*Tuple(int)*): list of axes to reduce
`keepdims` | (*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceL2Gradient

No documentation yet.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceMax


 

```
  Computes the max of the input tensor's element along the provided axes.
  The resulted tensor has the same rank as the input if keepdims equal True.
  If keepdims equal false, then the resulted tensor have the reduced dimension
  pruned.
```




### Interface


---------- | ----------
*Arguments* | 
`axes` | A list of integers, along which to reduce.
`keepdims` | Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`reduced` | Reduced output tensor.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceMean


Computes the  **mean**  of the input tensor's elements along the provided  `axes` . The resulting tensor has the same rank as the input if the  `keepdims`  argument equals 1 (default). If  `keepdims`  is set to 0, then the  `axes`  dimensions are pruned.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceMean",
    ["X"],
    ["Y"],
    axes=(0,1),
    keepdims=0
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[[9. 0. 3. 6. 0.]   

```
  [3. 4. 5. 0. 9.]
```

   

```
  [6. 9. 1. 1. 5.]
```

   

```
  [6. 2. 3. 7. 7.]
```

   

```
  [3. 1. 1. 0. 1.]]

  [[4. 3. 9. 8. 1.]
```

   

```
  [8. 2. 0. 4. 0.]
```

   

```
  [8. 9. 9. 0. 2.]
```

   

```
  [7. 2. 5. 8. 9.]
```

   

```
  [5. 9. 1. 9. 0.]]]]
```

 Y: [[6.5 1.5 6. 

```
  7.  0.5]
```

  [5.5 3. 

```
  2.5 2.  4.5]
```

  [7. 

```
  9.  5.  0.5 3.5]
```

  [6.5 2. 

```
  4.  7.5 8. ]
```

  [4. 

```
  5.  1.  4.5 0.5]]

```

 ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`axes` | (*Tuple(int)*): list of axes to reduce
`keepdims` | (*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceMeanGradient

No documentation yet.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceMin


 

```
  Computes the min of the input tensor's element along the provided axes.
  The resulted tensor has the same rank as the input if keepdims equal True.
  If keepdims equal false, then the resulted tensor have the reduced dimension
  pruned.
```




### Interface


---------- | ----------
*Arguments* | 
`axes` | A list of integers, along which to reduce.
`keepdims` | Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).
*Inputs* | 
`data` | An input tensor.
*Outputs* | 
`reduced` | Reduced output tensor.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceMinGradient

No documentation yet.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

---



## ReduceSum


Computes the  **sum**  of the input tensor's elements along the provided  `axes` . The resulting tensor has the same rank as the input if the  `keepdims`  argument equals 1 (default). If  `keepdims`  is set to 0, then the  `axes`  dimensions are pruned.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "ReduceSum",
    ["X"],
    ["Y"],
    axes=(0,1),
    keepdims=0
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[[[5. 3. 7. 9. 5.]   

```
  [4. 5. 1. 8. 3.]
```

   

```
  [1. 0. 9. 7. 6.]
```

   

```
  [7. 5. 0. 3. 1.]
```

   

```
  [6. 4. 4. 8. 3.]]

  [[8. 9. 6. 7. 7.]
```

   

```
  [5. 5. 4. 7. 0.]
```

   

```
  [9. 7. 6. 6. 7.]
```

   

```
  [7. 5. 2. 4. 2.]
```

   

```
  [4. 5. 1. 9. 4.]]]]
```

 Y: [[13. 12. 13. 16. 12.]  [ 9. 10. 

```
  5. 15.  3.]
```

  [10. 

```
  7. 15. 13. 13.]
```

  [14. 10. 

```
  2.  7.  3.]
```

  [10. 

```
  9.  5. 17.  7.]]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`axes` | (*Tuple(int)*): list of axes to reduce
`keepdims` | (*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor
*Outputs* | 
`Y` | (*Tensor`<float>`*): reduced tensor


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

---



## ReduceSumGradient

No documentation yet.


### Code


[caffe2/operators/reduce_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc)

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


[caffe2/operators/rowmul_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rowmul_op.cc)

---



## Relu


Applies rectified linear unit operation to the input data element-wise. The Relu operation takes one input $X$, produces one output $Y$, and is defined as:  $$Y = max(0,X)$$  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
  "Relu",
  ["X"],
  ["Y"]
  )

```

 workspace.FeedBlob("X", np.random.randn(4, 4).astype(np.float32)) # NCHW print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[-1.4655551  

```
  0.64575136  0.7921748   0.4150579 ]
```

  [ 0.41085166 -0.2837964  

```
  0.9881425  -1.9300346 ]
```

  [ 0.39705405 

```
  0.44639114  0.9940703   0.2926532 ]
```

  [-0.6726489  

```
  0.01330667  1.101319    0.33858967]]

```

 Y:  [[0.  

```
        0.64575136 0.7921748  0.4150579 ]
```

  [0.41085166 0.  

```
        0.9881425  0.        ]
```

  [0.39705405 0.44639114 0.9940703 

```
  0.2926532 ]
```

  [0.  

```
        0.01330667 1.101319   0.33858967]]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D output tensor with same shape as input


### Code


[caffe2/operators/relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cc)

---



## ReluGradient


ReluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.



### Code


[caffe2/operators/relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cc)

---



## ReluN


Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, y = min(max(0, x), n), is applied to the tensor elementwise.



### Interface


---------- | ----------
*Arguments* | 
`n` | the cap of output
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D input tensor


### Code


[caffe2/operators/relu_n_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_n_op.cc)

---



## ReluNGradient


ReluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.



### Interface


---------- | ----------
*Arguments* | 
`n` | the cap of forward op output


### Code


[caffe2/operators/relu_n_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_n_op.cc)

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


[caffe2/operators/remove_data_blocks_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/remove_data_blocks_op.cc)

---



## RemovePadding


Remove padding around the edges of each segment of the input data. This is the reverse operation of  **AddPadding** , and uses the same arguments and conventions for input and output data format.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  addpad_op = core.CreateOperator(  

```
    "AddPadding",
    ["X", "lengths_add"],
    ["Y", "lengths_out_add"],
    padding_width=1
```

 )  rmpad_op = core.CreateOperator(  

```
    "RemovePadding",
    ["Y", "lengths_rm"],
    ["Z", "lengths_out_rm"],
    padding_width=1
```

 )  workspace.FeedBlob("X", (np.random.randint(20, size=(3,5)))) workspace.FeedBlob("lengths_add", np.array([3]).astype(np.int32)) workspace.FeedBlob("lengths_rm", np.array([5]).astype(np.int32))  print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(addpad_op) print("Y:", workspace.FetchBlob("Y")) print("lengths_out_add:", workspace.FetchBlob("lengths_out_add"))  workspace.RunOperatorOnce(rmpad_op) print("Z:", workspace.FetchBlob("Z")) print("lengths_out_rm:", workspace.FetchBlob("lengths_out_rm"))  ```    **Result**    ```  X: [[17 19 

```
  1  9  1]
```

  [19 

```
  3  5 19  1]
```

  [16 

```
  0  0  0  4]]
```

 Y: [[ 0 

```
  0  0  0  0]
```

  [17 19 

```
  1  9  1]
```

  [19 

```
  3  5 19  1]
```

  [16 

```
  0  0  0  4]
```

  [ 0 

```
  0  0  0  0]]
```

 lengths_out_add: [5] Z: [[17 19 

```
  1  9  1]
```

  [19 

```
  3  5 19  1]
```

  [16 

```
  0  0  0  4]]
```

 lengths_out_rm: [3]  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`padding_width` | *(type: int)* Outer-size of padding to remove around each range.
`end_padding_width` | *(type: int)* [OPTIONAL] Specifies a different end-padding width. If this is not set, will use same as `padding_width`.
*Inputs* | 
`data_in` | Input tensor ($T<N, D_1, ..., D_n>$).
`lengths` | *(type: Tensor`<int>`)* Number of elements in each range. sum(lengths) = N. If not provided, considers all data as a single segment.
*Outputs* | 
`data_out` | *(type: Tensor)* Padded data tensor ($T<N + 2*padding_width, D_1, ..., D_n>$).
`lengths_out` | *(type: Tensor`<int>`)* [OPTIONAL] Lengths for each padded range.


### Code


[caffe2/operators/sequence_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc)

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


[caffe2/operators/replace_nan_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/replace_nan_op.cc)

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


[caffe2/operators/reservoir_sampling.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reservoir_sampling.cc)

---



## ResetCounter


Resets a count-down counter with initial value specified by the  `init_count`  argument.
  

```
  Github Links:
  - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc


```

 <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  createcounter_op = core.CreateOperator(  

```
    "CreateCounter",
    [],
    ["counter"],
    init_count=5
```

 )  retrievecount_op = core.CreateOperator(  

```
    "RetrieveCount",
    ["counter"],
    ["count"]
```

 )  checkcounterdone_op = core.CreateOperator(  

```
    "CheckCounterDone",
    ["counter"],
    ["done"]
```

 )  countup_op = core.CreateOperator(  

```
    "CountUp",
    ["counter"],
    ["previous_count"],
```

 )  countdown_op = core.CreateOperator(  

```
    "CountDown",
    ["counter"],
    ["done"],
```

 )  resetcounter_op = core.CreateOperator(  

```
    "ResetCounter",
    ["counter"],
    ["previous_count"],
    init_count=3
```

 )   # Create counter workspace.RunOperatorOnce(createcounter_op) print("'counter' pointer:", workspace.FetchBlob("counter"))   # Retrieve initial counter value workspace.RunOperatorOnce(retrievecount_op) print("Initial 'count':", workspace.FetchBlob("count"))   # Check if counter is done workspace.RunOperatorOnce(checkcounterdone_op) print("Initial 'done' value:", workspace.FetchBlob("done"))   # Test CountUp operator print("\nTesting CountUp operator...") for i in range(5):  

```
    workspace.RunOperatorOnce(countup_op)
    print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

```

 workspace.RunOperatorOnce(retrievecount_op) print("'count' value after CountUp test:", workspace.FetchBlob("count"))   # Test CountDown operator print("\nTesting CountDown operator...") for i in range(11):  

```
    workspace.RunOperatorOnce(countdown_op)
    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))
```

 ```    **Result**   ``` 'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
Initial 'count': 5 Initial 'done' value: False  Testing CountUp operator...
'previous_count' after CountUp: 5 'previous_count' after CountUp: 6 'previous_count' after CountUp: 7 'previous_count' after CountUp: 8 'previous_count' after CountUp: 9 'count' value after CountUp test: 10  Testing CountDown operator...
'count' value after CountDown: 9	'done' value: False 'count' value after CountDown: 8	'done' value: False 'count' value after CountDown: 7	'done' value: False 'count' value after CountDown: 6	'done' value: False 'count' value after CountDown: 5	'done' value: False 'count' value after CountDown: 4	'done' value: False 'count' value after CountDown: 3	'done' value: False 'count' value after CountDown: 2	'done' value: False 'count' value after CountDown: 1	'done' value: False 'count' value after CountDown: 0	'done' value: False 'count' value after CountDown: -1	'done' value: True ```  </details>  


### Interface


---------- | ----------
*Arguments* | 
`init_count` | *(type: int; default: 0)* Resets counter to this value, must be >= 0.
*Inputs* | 
`counter` | *(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.
*Outputs* | 
`previous_value` | *(type: int)* [OPTIONAL] count value BEFORE this operation.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc)

---



## ResetCursor


Resets the offsets for the given TreeCursor. This operation is thread safe.



### Interface


---------- | ----------
*Inputs* | 
`cursor` | A blob containing a pointer to the cursor.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## Reshape


Reshape the input tensor similar to numpy's [reshape]( [https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html).](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html).) 
 Takes a tensor as input and an optional tensor specifying the new shape. When the second input is absent, an extra argument shape must be specified. Outputs the reshaped tensor as well as the original shape.
 At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions. A dimension could also be 0, in which case the actual dimension value is going to be copied from the input tensor.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reshape_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reshape_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Reshape",
    ["data"],
    ["reshaped", "old_shape"],
    shape=(3,2)
```

 )  workspace.FeedBlob("data", (np.random.randint(100, size=(6)))) print("data:", workspace.FetchBlob("data")) workspace.RunOperatorOnce(op) print("reshaped:", workspace.FetchBlob("reshaped")) print("old_shape:", workspace.FetchBlob("old_shape"))  ```    **Result**    ```  data: [86 60 85 96 

```
  7 37]
```

 reshaped: [[86 60]  

```
          [85 96]
          [ 7 37]]
```

 old_shape: [6]  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`shape` | *(type: Tuple(int))* New shape. Do not set if using `new_shape` input.
*Inputs* | 
`data` | *(type: Tensor)* Input tensor.
`new_shape` | *(type: Tensor`<int>`)* [OPTIONAL] Tensor containing new shape.
*Outputs* | 
`reshaped` | *(type: Tensor)* Reshaped output tensor.
`old_shape` | *(type: Tensor`<int>`)* Tensor containing old shape of `data`.


### Code


[caffe2/operators/reshape_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reshape_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/resize_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/resize_op.cc)

---



## ResizeNearestGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`width_scale` | Scale along width dimension
`height_scale` | Scale along height dimension


### Code


[caffe2/operators/resize_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/resize_op.cc)

---



## RetrieveCount


Retrieve the current value from the counter as an integer.
  

```
  Github Links:
  - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc


```

 <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  createcounter_op = core.CreateOperator(  

```
    "CreateCounter",
    [],
    ["counter"],
    init_count=5
```

 )  retrievecount_op = core.CreateOperator(  

```
    "RetrieveCount",
    ["counter"],
    ["count"]
```

 )  checkcounterdone_op = core.CreateOperator(  

```
    "CheckCounterDone",
    ["counter"],
    ["done"]
```

 )  countup_op = core.CreateOperator(  

```
    "CountUp",
    ["counter"],
    ["previous_count"],
```

 )  countdown_op = core.CreateOperator(  

```
    "CountDown",
    ["counter"],
    ["done"],
```

 )  resetcounter_op = core.CreateOperator(  

```
    "ResetCounter",
    ["counter"],
    ["previous_count"],
    init_count=3
```

 )   # Create counter workspace.RunOperatorOnce(createcounter_op) print("'counter' pointer:", workspace.FetchBlob("counter"))   # Retrieve initial counter value workspace.RunOperatorOnce(retrievecount_op) print("Initial 'count':", workspace.FetchBlob("count"))   # Check if counter is done workspace.RunOperatorOnce(checkcounterdone_op) print("Initial 'done' value:", workspace.FetchBlob("done"))   # Test CountUp operator print("\nTesting CountUp operator...") for i in range(5):  

```
    workspace.RunOperatorOnce(countup_op)
    print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

```

 workspace.RunOperatorOnce(retrievecount_op) print("'count' value after CountUp test:", workspace.FetchBlob("count"))   # Test CountDown operator print("\nTesting CountDown operator...") for i in range(11):  

```
    workspace.RunOperatorOnce(countdown_op)
    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))
```

 ```    **Result**   ``` 'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
Initial 'count': 5 Initial 'done' value: False  Testing CountUp operator...
'previous_count' after CountUp: 5 'previous_count' after CountUp: 6 'previous_count' after CountUp: 7 'previous_count' after CountUp: 8 'previous_count' after CountUp: 9 'count' value after CountUp test: 10  Testing CountDown operator...
'count' value after CountDown: 9	'done' value: False 'count' value after CountDown: 8	'done' value: False 'count' value after CountDown: 7	'done' value: False 'count' value after CountDown: 6	'done' value: False 'count' value after CountDown: 5	'done' value: False 'count' value after CountDown: 4	'done' value: False 'count' value after CountDown: 3	'done' value: False 'count' value after CountDown: 2	'done' value: False 'count' value after CountDown: 1	'done' value: False 'count' value after CountDown: 0	'done' value: False 'count' value after CountDown: -1	'done' value: True ```  </details>  


### Interface


---------- | ----------
*Inputs* | 
`counter` | *(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.
*Outputs* | 
`count` | *(type: int)* Current count value.


### Code


[caffe2/operators/counter_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc)

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


[caffe2/operators/reverse_packed_segs_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reverse_packed_segs_op.cc)

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


[caffe2/operators/roi_align_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_align_op.cc)

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


[caffe2/operators/roi_align_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_align_gradient_op.cc)

---



## RoIAlignRotated


Similar to RoIAlign but can handle rotated region proposals.
Based on  [https://arxiv.org/abs/1703.01086.](https://arxiv.org/abs/1703.01086.) 



### Interface


---------- | ----------
*Arguments* | 
`spatial_scale` | (float) default 1.0; Spatial scale of the input feature map X relative to the input image. E.g., 0.0625 if X has a stride of 16 w.r.t. the input image.
`pooled_h` | (int) default 1; Pooled output Y's height.
`pooled_w` | (int) default 1; Pooled output Y's width.
`sampling_ratio` | (int) default -1; number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If <= 0, then an adaptive number of grid points are used (computed as ceil(roi_width / pooled_w), and likewise for height).
*Inputs* | 
`X` | 4D feature map input of shape (N, C, H, W).
`RoIs` | 2D input of shape (R, 5 or 6) specifying R RoIs representing: batch index in [0, N - 1], center_x, center_y, width, height, angle. The RoI coordinates are in the coordinate system of the input image. `angle` should be specified in degrees and represents the RoI rotated counter-clockwise. For inputs corresponding to a single image, batch index can be excluded to have just 5 columns.
*Outputs* | 
`Y` | 4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element is a pooled feature map cooresponding to the r-th RoI.


### Code


[caffe2/operators/roi_align_rotated_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_align_rotated_op.cc)

---



## RoIAlignRotatedGradient

No documentation yet.


### Interface


---------- | ----------
*Inputs* | 
`X` | See RoIAlignRotated.
`RoIs` | See RoIAlignRotated.
`dY` | Gradient of forward output 0 (Y)
*Outputs* | 
`dX` | Gradient of forward input 0 (X)


### Code


[caffe2/operators/roi_align_rotated_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_align_rotated_gradient_op.cc)

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


[caffe2/operators/roi_pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_pool_op.cc)

---



## RoIPoolGradient

No documentation yet.


### Code


[caffe2/operators/roi_pool_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_pool_op.cc)

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


[caffe2/operators/rowmul_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rowmul_op.cc)

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


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

---



## RowwiseMax


Compute row-wise max reduction of the input tensor. This op takes one input, $X$, of shape $BxMxN$, where $B$ is the batch size, $M$ is number of rows, and $N$ is number of columns. The output of this op, $Y$, is a matrix of shape $BxM$, with one row for each element of the batch, and the same number of columns as the number of rows of the input tensor.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "RowwiseMax",
    ["X"],
    ["Y"]
```

 )  # Create X, simulating a batch of 2, 4x4 matricies X = np.random.randint(0,high=20,size=(2,4,4)) print("X:\n",X)  # Feed X into workspace workspace.FeedBlob("X", X.astype(np.float32))  # Run op workspace.RunOperatorOnce(op)  # Collect Output print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[[ 5 12 10 

```
  1]
  [ 4 16  2 15]
  [ 5 11 12 15]
  [15  4 17 19]]

```

  [[16 

```
  5  5 13]
  [17  2  1 17]
  [18  3 19  5]
  [14 16 10 16]]]
```

 Y:  [[12. 16. 15. 19.]  [16. 17. 19. 16.]]   ```   </details>      


### Interface


---------- | ----------
*Inputs* | 
`X` | A tensor of dimensions $B x M x N$ to compute rowwise-max. Here, $B$ is batch size, and $M$ and $N$ are the number of rows and columns of each element of the batch, respectively.
*Outputs* | 
`Y` | The output tensor of shape $B x M$, where each row represents the row-wise maximums for that element of the input batch.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

---



## RowwiseMaxGradient

No documentation yet.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

---



## Rsqrt

Computes the element-wise rsqrt of the input.


### Interface


---------- | ----------
*Inputs* | 
`X` | ND input tensor
*Outputs* | 
`Y` | ND output tensor


### Code


[caffe2/operators/rsqrt_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rsqrt_op.cc)

---



## RsqrtGradient

No documentation yet.


### Code


[caffe2/operators/rsqrt_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rsqrt_op.cc)

---



## Save


Saves a set of blobs to a db. It takes $[1, \infty)$ number of inputs and has no output. The contents of the inputs are written into the db using the settings specified by the arguments.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Save",
    ["X", "Y", "Z"],
    [],
    db="test_db2",
    db_type="leveldb",
    blob_name_overrides=["x_scores", "y_scores", "z_scores"]
```

 )  workspace.FeedBlob("X", np.random.randint(20, size=(5,5))) workspace.FeedBlob("Y", np.random.randint(20, size=(5,5))) workspace.FeedBlob("Z", np.random.randint(20, size=(5,5))) workspace.RunOperatorOnce(op)   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`absolute_path` | *(type: int; default: 0)* If set to non-zero, save the db directly to the path specified by the `db` arg. If not set (default), prepend the path of the current root folder of the workspace to the path specified by the `db` arg.
`strip_prefix` | *(type: string, default: "")* Characters in the provided blob names that match `strip_prefix` will be removed prior to saving. Also, characters that precede `strip_prefix` will be removed. Useful for removing device scope from blob names.
`blob_name_overrides` | *(List(string))* If set, used as blob names instead of original blob names. Must be same length as number of blobs.
`db` | *(type: string)* The output path of the db. See the `absolute_path` arg details for options regarding the current root folder of the workspace.
`db_type` | *(type: string)* Type of db to save (options: "lmdb", "leveldb", "minidb").
*Inputs* | 
`X` | *(type: Tensor)* Input tensor(s).


### Code


[caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)

---



## Scale


Scale takes one input data (Tensor) and produces one output data (Tensor) whose value is the input data tensor scaled element-wise.



### Interface


---------- | ----------
*Arguments* | 
`scale` | (float, default 1.0) the scale to apply.


### Code


[caffe2/operators/scale_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/scale_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/one_hot_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/one_hot_ops.cc)

---



## Selu


 The  *Selu*  op takes one input tensor $X$, an argument $alpha$, an argument $scale$, and produces one output tensor $Y$ of the same shape as $X.$ The op performs the element wise  *Selu*  operation, defined as  $$y=selu(x) =\begin{cases}scale (\alpha e^{x} - \alpha) & x < 0\\scale  * x & otherwise\end{cases}$$  The default value of * alpha * is 1.6732632423543772848170429916717 and the default value of * scale* is 1.0507009873554804934193349852946. See [Self-Normalizing Neural Networks]( [https://arxiv.org/abs/1706.02515)](https://arxiv.org/abs/1706.02515))  for more information.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Selu",
    ["X"],
    ["Y"],
```

 )  workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32)) print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[ 1.1613879 

```
  -0.27111396 -1.2076733 ]
```

  [ 1.3442237 

```
  -1.0701777   1.2070968 ]
```

  [ 0.23810555 

```
  0.9740916  -1.7872391 ]]

```

 Y:  [[ 1.2202715 

```
  -0.4174965  -1.2326177 ]
```

  [ 1.4123772 

```
  -1.1551634   1.2682979 ]
```

  [ 0.25017774 

```
  1.023479   -1.4637551 ]]

```

 ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`alpha` | *(type: float; default: 1.673263~)* Alpha constant in equation.
`scale` | *(type: float; default: 1.050700~; must be > 1.0)* Scale constant in equation.
*Inputs* | 
`X` | Input tensor of data to be operated on.
*Outputs* | 
`Y` | Output tensor with same shape as input.


### Code


[caffe2/operators/selu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.cc)

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


[caffe2/operators/selu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/selu_op.cc)

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


[caffe2/operators/communicator_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/communicator_op.cc)

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


[caffe2/operators/boolean_mask_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc)

---



## Shape


Produce a 1D int64 tensor with the shape of the input tensor.
If called with an optional argument  `axes` , the result will only contain the dimensions of specified axes.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/shape_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/shape_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Shape",
    ["X"],
    ["shape"],
```

 )  workspace.FeedBlob("X", (np.random.randint(10, size=(2,3)))) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("shape:", workspace.FetchBlob("shape"))   ```    **Result**    ```   X: [[3 2 5]  [5 7 3]] shape: [2 3]   ```   </details>        


### Interface


---------- | ----------
*Arguments* | 
`axes` | *(type: int[])* Array of interested axes.If given, this operator only returns the dimensions of the given axes.Otherwise, the operator returns the dimensions of all axes.
*Inputs* | 
`X` | *(type: Tensor)* Input tensor.
*Outputs* | 
`shape` | *(type: Tensor)* Output tensor containing shape of input tensor.


### Code


[caffe2/operators/shape_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/shape_op.cc)

---



## Sigmoid


Apply the Sigmoid function element-wise to the input tensor. This is often used as a non-linear activation function in a neural network. The sigmoid function is defined as:  $$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$  Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Sigmoid",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.randn(5).astype(np.float32)) print("input:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("sigmoid:", workspace.FetchBlob("Y"))   ```    **Result**    ```   input: [ 1.5744036  

```
  0.31632107  1.7842269   1.4450722  -2.1726978 ]
```

 sigmoid: [0.8284105 

```
  0.57842743 0.85621804 0.80923885 0.10222916]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.


### Code


[caffe2/operators/sigmoid_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc)

---



## SigmoidCrossEntropyWithLogits


Given two matrices logits and targets, of same shape, (batch_size, num_classes), computes the sigmoid cross entropy between the two.
Returns a tensor of shape (batch_size,) of losses for each example.



### Interface


---------- | ----------
*Arguments* | 
`log_D_trick` | 
default is false; if enabled, will use the log d trick to avoid the vanishing
gradients early on; see Goodfellow et. al (2014)

`unjoined_lr_loss` | 
default is false; if enabled, the model will be allowed to train on an unjoined
dataset, where some examples might be false negative and might appear
in the dataset later as (true) positive example.

*Inputs* | 
`logits` | matrix of logits for each example and class.
`targets` | matrix of targets, same shape as logits.
*Outputs* | 
`xentropy` | Vector with the total xentropy for each example.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## SigmoidCrossEntropyWithLogitsGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## SigmoidGradient


SigmoidGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the sigmoid function.



### Code


[caffe2/operators/sigmoid_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc)

---



## Sign


Computes sign for each element of the input: -1, 0 or 1.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator( "Sign", ["X"], ["Y"], )  workspace.FeedBlob("X", (np.random.rand(3, 3).astype(np.float32) - np.random.rand(3, 3).astype(np.float32))) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[ 0.02816287 

```
  0.22408086 -0.30342305]
```

 [-0.18481976 

```
  0.03948995  0.39698976]
```

 [-0.63304734 -0.6919183 

```
  -0.31524038]]
```

 Y: [[ 1. 

```
  1. -1.]
```

 [-1. 

```
  1.  1.]
```

 [-1. -1. -1.]]   ```   </details>      


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## Sin


Calculates the sine of the given input tensor, element-wise.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sin_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sin_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Sin",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.rand(5).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [0.8466114 

```
  0.1803606  0.5601509  0.04959291 0.64770824]
```

 Y: [0.74903965 0.17938434 0.5313141 

```
  0.04957259 0.60336035]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor calculated as the sine of the input tensor, element-wise.


### Code


[caffe2/operators/sin_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sin_op.cc)

---



## SinGradient

No documentation yet.


### Code


[caffe2/operators/sin_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sin_op.cc)

---



## Sinh


Calculates the hyperbolic sine of the given input tensor, element-wise.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinh_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Sinh",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.rand(5).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [0.98907769 0.52907848 0.03216429 0.94983935 0.47881418] Y: [1.15841695 0.5541099 

```
  0.03216984 1.09924557 0.49732079]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The hyperbolic sine values of the input tensor, computed element-wise


### Code


[caffe2/operators/sinh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinh_op.cc)

---



## SinhGradient

No documentation yet.


### Code


[caffe2/operators/sinh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinh_op.cc)

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


[caffe2/operators/sinusoid_position_encoding_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinusoid_position_encoding_op.cc)

---



## Size


Return a 1D tensor of type  *int64*  that contains the number of elements of the input tensor.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Size",
    ["X"],
    ["size"],
```

 )  workspace.FeedBlob("X", (np.random.randint(10, size=(3,3)))) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("size:", workspace.FetchBlob("size"))  workspace.ResetWorkspace()  workspace.FeedBlob("X", (np.random.rand(6,4))) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("size:", workspace.FetchBlob("size"))   ```    **Result**    ```   X: [[3 7 0]  [0 1 6]  [5 0 8]] size: 9 X: [[0.92017884 0.32115368 0.68692035 0.64135016]  [0.8723328 

```
  0.77830265 0.80688656 0.25524236]
```

  [0.37970216 0.76407047 0.85689564 0.30692883]  [0.69352573 0.42531502 0.16415212 0.59209324]  [0.52684188 0.37094846 0.60670079 0.6489272 ]  [0.94715906 0.34800557 0.61898769 0.28947359]] size: 24   ```   </details>        


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor)* Input tensor to calculate number of elements.
*Outputs* | 
`size` | *(type: Tensor)* 1D tensor of type int64 that contains the number of elements in the input tensor *X*.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## Slice


Produces a slice of the input tensor.
 - Currently, only slicing in a single dimension is supported.
 - Start and end indices are either passed as two 1D input tensors or using the  `starts`  and  `ends`  arguments.
 - If a negative value is passed for any of the start or end indices, it represents the number of elements before the end of that dimension. End indices are non-inclusive unless negative (end index -1 means up to and including the last element).
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/slice_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/slice_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Slice",
    ["X"],
    ["Y"],
    starts=(0,1),
    ends=(-1,3)
```

 )  workspace.FeedBlob("X", np.array([[1,2,3,4],[5,6,7,8]])) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[1 2 3 4]  [5 6 7 8]] Y: [[2 3]  [6 7]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`starts` | (*Tuple(int)*): list of starting indices
`ends` | (*Tuple(int)*): list of ending indices
*Inputs* | 
`X` | (*Tensor*): tensor to extract slices from
`starts` | (*Tensor`<int>`*): 1D tensor of start-indices for each dimension of data
`ends` | (*Tensor`<int>`*): 1D tensor of end-indices for each dimension of data
*Outputs* | 
`Y` | (*Tensor*): sliced output tensor


### Code


[caffe2/operators/slice_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/slice_op.cc)

---



## SliceGradient

No documentation yet.


### Code


[caffe2/operators/slice_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/slice_op.cc)

---



## Snapshot

No documentation yet.


### Code


[caffe2/operators/load_save_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc)

---



## Softmax


 Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1. The softmax operator is typically the last layer in a classifier network, as its output can be interpreted as confidence probabilities of an input belonging to each class. The input is a 2-D tensor (Tensor) of size (batch_size x input_feature_dimensions). The output tensor has the same shape and contains the softmax normalized values of the corresponding input. The softmax function is defined as follows:  $$softmax(x_i) = \frac{\exp(x_i)}{\sum_{j} \exp(x_j)}$$  The input does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor  `X`  in $[a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}]$, where k is the  `axis`  provided, then  `X`  will be coerced into a 2-dimensional tensor with dimensions $[(a_0  * ... *  a_{k-1}), (a_k  * ... *  a_{n-1})]$. For the default case where  `axis` =1, the  `X`  tensor will be coerced into a 2D tensor of dimensions $[a_0, (a_1  * ... *  a_{n-1})]$, where $a_0$ is often the batch size. In this situation, we must have $a_0 = N$ and $a_1  * ... *  a_{n-1} = D$. Each of these dimensions must be matched correctly, or else the operator will throw errors.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Softmax",
    ["X"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.randn(1, 5).astype(np.float32)) print("input:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("softmax:", workspace.FetchBlob("Y"))   ```    **Result**    ```  input: [[ 0.0417839  

```
  0.61960053 -0.23150268 -0.64389366 -3.0000346 ]]
```

 softmax: [[0.24422921 0.43525138 0.18582782 0.12303016 0.01166145]]   ```   </details>    


### Interface


---------- | ----------
*Arguments* | 
`axis` | *(type: int; default: 1)* Axis of the inputs when coerced to 2D matrix.
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input tensor that's coerced into a 2D matrix of size (NxD) as described above.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* The softmax normalized output tensor with the same shape as input tensor.


### Code


[caffe2/operators/softmax_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.cc)

---



## SoftmaxGradient

No documentation yet.


### Code


[caffe2/operators/softmax_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.cc)

---



## SoftmaxWithLoss


Combined Softmax and Cross-Entropy loss operator. The operator first computes the softmax normalized values for each layer in the batch of the given input, then computes cross-entropy loss. This operator is numerically more stable than separate  `Softmax`  and  `CrossEntropy`  ops. The inputs are a 2-D tensor  `logits`  of size (batch_size x input_feature_dimensions), which represents the unscaled log probabilities, and a 1-dimensional integer  `labels`  tensor for ground truth. An optional third input blob ( `weight_tensor` ) can be used to weight the samples for the loss, which is useful if the training set is unbalanced. This operator outputs a  `softmax`  tensor which contains the probability for each label for each example (same shape is  `logits`  input), and a scalar  `loss`  value, which is the averaged cross-entropy loss between the softmax probabilities and the ground truth values. Use parameter  `label_prob` =1 to enable inputting labels as a probability distribution.
 Softmax cross-entropy loss function:  $$loss(x, class) = -\log{\biggl(\frac{\exp(x[class])}{\sum_{j} \exp(x[j])}\biggr)} = -x[class] + \log{\biggl(\sum_{j} \exp(x[j])\biggr)}$$  or if the  `weight_tensor`  has been passed:  $$loss(x, class) = weight[class]\biggl(-x[class] + \log{\biggl(\sum_{j} \exp(x[j])\biggr)}\biggr)$$  The  `logits`  input does not need to explicitly be a 2D vector; rather, it will be coerced into one. For an arbitrary n-dimensional tensor  `X`  in $[a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}]$, where k is the  `axis`  provided, then  `X`  will be coerced into a 2-dimensional tensor with dimensions $[(a_0  * ... *  a_{k-1}), (a_k  * ... *  a_{n-1})]$. For the default case where  `axis` =1, the  `X`  tensor will be coerced into a 2D tensor of dimensions $[a_0, (a_1  * ... *  a_{n-1})]$, where $a_0$ is often the batch size. In this situation, we must have $a_0 = N$ and $a_1  * ... *  a_{n-1} = D$. Each of these dimensions must be matched correctly, or else the operator will throw errors.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_with_loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_with_loss_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "SoftmaxWithLoss",
    ["logits", "labels"],
    ["softmax", "avgloss"]
```

 )  workspace.FeedBlob("logits", np.random.randn(1, 5).astype(np.float32)) workspace.FeedBlob("labels", np.asarray([4]).astype(np.int32)) print("logits:", workspace.FetchBlob("logits")) print("labels:", workspace.FetchBlob("labels")) workspace.RunOperatorOnce(op) print("softmax:", workspace.FetchBlob("softmax")) print("avgloss:", workspace.FetchBlob("avgloss"))   ```    **Result**    ```   logits: [[-0.3429451 

```
  -0.80375195  0.23104447  1.4569176  -0.5268362 ]]
```

 labels: [4] softmax: [[0.09721052 0.0613179 

```
  0.17258129 0.58800864 0.0808817 ]]
```

 avgloss: 2.5147676   ```   </details>  <details>  <summary> <b>Example 2</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "SoftmaxWithLoss",
    ["logits", "labels"],
    ["softmax", "avgloss"],
    scale=5.0
```

 )  workspace.FeedBlob("logits", np.asarray([[.1, .4, .7, 1.5, .2]]).astype(np.float32)) workspace.FeedBlob("labels", np.asarray([4]).astype(np.int32)) print("logits:", workspace.FetchBlob("logits")) print("labels:", workspace.FetchBlob("labels")) workspace.RunOperatorOnce(op) print("softmax:", workspace.FetchBlob("softmax")) print("avgloss:", workspace.FetchBlob("avgloss"))   ```    **Result**    ```   logits: [[0.1 0.4 0.7 1.5 0.2]] labels: [4] softmax: [[0.10715417 0.144643  

```
  0.19524762 0.4345316  0.11842369]]
```

 avgloss: 10.667433   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`label_prob` | *(type: int; default: 0)* Setting to 1 enables inputting labels as probability distribution.
`axis` | *(type: int; default: 1)* Axis of the inputs when coerced to 2D.
`scale` | *(type: float)* Average loss output scaling factor (must be >= 0).
`order` | *(type: string; default: 'NCHW')* Order of blob dimensions (only 'NCHW' is supported currently).
*Inputs* | 
`logits` | *(type: Tensor`<float>`)* Input tensor.
`labels` | *(type: Tensor`<float>`)* Ground truth label tensor.
`weight_tensor` | *(type: Tensor`<float>`)* [OPTIONAL] Blob used to weight the samples for the loss.
*Outputs* | 
`softmax` | *(type: Tensor`<float>`)* Softmax output tensor.
`loss` | *(type: float)* Averaged cross-entropy loss output.


### Code


[caffe2/operators/softmax_with_loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_with_loss_op.cc)

---



## SoftmaxWithLossGradient

No documentation yet.


### Code


[caffe2/operators/softmax_with_loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_with_loss_op.cc)

---



## Softplus


Softplus takes one input data tensor $X$ and produces one output data tensor $Y,$ where the softplus function, $y = ln(e^x + 1)$, is applied to $X$ elementwise.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Softplus",
    ["X"],
    ["Y"],
```

 )  workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32)) print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[-0.5380011  

```
  0.65190786  0.55673236]
```

  [-0.16272168 

```
  0.5451048   0.30880353]
```

  [-0.76606876 -0.6238556 

```
  -0.40444514]]

```

 Y:  [[0.4598992 

```
  1.0713093  1.0097669 ]
```

  [0.61509246 1.0023911 

```
  0.8594219 ]
```

  [0.38174385 0.42909983 0.5112337 ]]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`X` | Input data blob to be operated on.
*Outputs* | 
`Y` | Output data blob with same shape as input.


### Code


[caffe2/operators/softplus_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.cc)

---



## SoftplusGradient

No documentation yet.


### Code


[caffe2/operators/softplus_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softplus_op.cc)

---



## Softsign


 *Softsign*  takes one input data tensor $X$ and produces one output data $Y,$ where the softsign function, $y = \frac{x}{1+ |x|}$, is applied to $X$ elementwise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softsign_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softsign_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Softsign",
    ["X"],
    ["Y"],
```

 )  workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32)) print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.RunOperatorOnce(op) print("Y:\n", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[-1.3060539  

```
  0.7242748  -1.9907674 ]
```

  [-0.64802396 -0.03244735 

```
  0.7455406 ]
```

  [-0.298492  

```
  -0.5774271   2.8364444 ]]

```

 Y:  [[-0.5663588  

```
  0.420046   -0.6656376 ]
```

  [-0.39321268 -0.03142761 

```
  0.4271116 ]
```

  [-0.2298759 

```
  -0.36605626  0.739342  ]]

```

 ```   </details>   


### Interface


---------- | ----------
*Inputs* | 
`input` | Input data blob to be operated on.
*Outputs* | 
`output` | Output data blob with same shape as input


### Code


[caffe2/operators/softsign_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softsign_op.cc)

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


[caffe2/operators/softsign_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softsign_op.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeLogMeanExpGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeLogSumExpGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeMaxGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentRangeSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SpaceToBatch


Zero-pads and then rearranges (permutes) blocks of spatial data into batch. More specifically, this op outputs a copy of the input tensor where values from the height and width dimensions are moved to the batch dimension. After the zero-padding is according to the  `pad`  argument, both height and width of the input must be divisible by the  `block_size` . Only "NCHW" order is currently supported.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "SpaceToBatch",
    ["X"],
    ["Y"],
    pad=2,
    block_size=3
```

 )  workspace.FeedBlob("X", np.random.rand(1,3,5,5).astype(np.float32)) print("X.shape:", workspace.FetchBlob("X").shape) workspace.RunOperatorOnce(op) print("Y.shape:", workspace.FetchBlob("Y").shape)   ```    **Result**    ```   X.shape: (1, 3, 5, 5) Y.shape: (9, 3, 3, 3)   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`pad` | (*int*): exclusive axis that divides the first and second dimension of matrix `A` (default=0)
`block_size` | (*int*): height/width of spatial blocks to be moved (default=2)
`order` | (*string*): order of dimensions of input and output blobs; only "NCHW" order is currently supported (default="NCHW")
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor (NCHW order)
*Outputs* | 
`Y` | (*Tensor`<float>`*): output tensor (NCHW order)


### Code


[caffe2/operators/space_batch_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc)

---



## SparseLengthsIndicesInGradientMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsIndicesInGradientSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsIndicesInGradientWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/lengths_reducer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_ops.cc)

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


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

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


[caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc)

---



## SparseLengthsMeanGradient

No documentation yet.


### Code


[caffe2/operators/lengths_reducer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_ops.cc)

---



## SparseLengthsPositionalWeightedSum


Variation of SparseLengthsWeightedSum operator, where, for each row, weights are accessed by indices [0..L-1], where L is the length of given row.
This is basically a fused operator of LengthsRangeFill + Gather + SparseWeightedSum 


### Interface


---------- | ----------
*Inputs* | 
`DATA` | uint8 tensor obtained with operator FloatToRowwiseQuantized8Bits
`WEIGHT` | Scalar multipliers for the input slices. Must be a vector with the length matching the length of DATA
`INDICES` | Integer vector containing indices of the first dimension of DATA for the slices that are being aggregated
`LENGTHS` | Vector with the same sum of elements as the first dimension of DATA
*Outputs* | 
`output` | output


### Code


[caffe2/operators/lengths_reducer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_ops.cc)

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


[caffe2/operators/lengths_reducer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_ops.cc)

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


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

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


[caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc)

---



## SparseLengthsSumGradient

No documentation yet.


### Code


[caffe2/operators/lengths_reducer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_ops.cc)

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


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

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


[caffe2/operators/lengths_reducer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_ops.cc)

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


[caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_rowwise_8bit_ops.cc)

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


[caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_fused_8bit_rowwise_ops.cc)

---



## SparseLengthsWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/lengths_reducer_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/lengths_reducer_ops.cc)

---



## SparseLengthsWeightedSumWithMainInputGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/sparse_normalize_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sparse_normalize_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseSortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/sparse_to_dense_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sparse_to_dense_op.cc)

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


[caffe2/operators/sparse_to_dense_mask_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sparse_to_dense_mask_op.cc)

---



## SparseToDenseMaskGradient


The output is the gradient of the input value from SparseToDenseMask. The gradient for default_value has not been implemented.



### Code


[caffe2/operators/sparse_to_dense_mask_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sparse_to_dense_mask_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SparseUnsortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## SpatialBN


Applies spatial batch normalization to the input tensor as described in the original paper, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift]( [https://arxiv.org/abs/1502.03167).](https://arxiv.org/abs/1502.03167).)  Be aware, this operator has two different output sets, depending on the value of  *is_test* . According to the paper, the primary operation of spatial batch normalization is:  $$Y = \frac{X - \mu_x}{\sqrt{\sigma^2_{x} + \epsilon}} *\gamma + b$$  In the equation, $\mu_x$ is the * mean *, $X$ is the input data, $\sigma^2_{x}$ is the * var *, $\epsilon$ is * epsilon *, $\gamma$ is the * scale *, $b$ is the * bias *, and $Y$ is the output data. The * momentum * arg also affects this calculation in the computation of the running mean and variance. The influence of * momentum * is as follows:  $$running\_mean = running\_mean *  momentum + mean  * (1 - momentum)$$  $$running\_var = running\_var *  momentum + var  * (1 - momentum)$$  Output when is_test = 0 (train mode): * Y, mean, var, saved_mean, saved_var *  Output when is_test = 1 (test mode): * Y*  Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.cc)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.h)   


### Interface


---------- | ----------
*Arguments* | 
`is_test` | *(type: int; default: 0)* If set to nonzero, run spatial batch normalization in test mode.
`epsilon` | *(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero.
`order` | *(type: string; default: "NCHW")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is "NHWC".
`momentum` | *(type: float; default: 0.9)* Factor used in computing the running mean and variance. e.g., running_mean = running_mean x momentum + mean x (1 - momentum)
`num_batches` | *(type: int; default: 1)* Specifies the number of batches to apply normalization on. Requires specifying the optional sums and sumsq inputs that provide statistics across multiple batches from which mean and variance can be determined.
*Inputs* | 
`X` | The input 4-dimensional tensor of shape $NCHW$ or $NHWC$ depending on the order parameter.
`scale` | The scale as a 1-dimensional tensor of size $C$ to be applied to the output.
`bias` | The bias as a 1-dimensional tensor of size $C$ to be applied to the output.
`mean` | The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size $C$.
`var` | The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size $C$.
`sums` | *(optional)* Per-channel sums of elements to be used to determine the mean and variance for this batch.
`sumsq` | *(optional)* Per-channel sum of elements squared per channel to be used to determine the variance for this batch.
*Outputs* | 
`Y` | The output 4-dimensional tensor of the same shape as $X$.
`mean` | The running mean after the spatial BN operator. Must be in-place with the input *mean*. Should not be used for testing.
`var` | The running variance after the spatial BN operator. Must be in-place with the input *var*. Should not be used for testing.
`saved_mean` | Saved mean used during training to speed up gradient computation. Should not be used for testing.
`saved_var` | Saved variance used during training to speed up gradient computation. Should not be used for testing.


### Code


[caffe2/operators/spatial_batch_norm_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_op.cc)

---



## SpatialBNGradient

No documentation yet.


### Code


[caffe2/operators/spatial_batch_norm_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_batch_norm_gradient_op.cc)

---



## SpatialSoftmaxWithLoss


Combined Spatial Softmax and Cross-Entropy loss operator.
Similar to SoftmaxWithLoss, this operator computes the spatial softmax normalized values for each layer in the batch of the given input, after which cross-entropy loss is computed. This operator is numerically more stable than separate Softmax and CrossEntropy ops. The inputs are a 2-D tensor (Tensor) of size (batch_size x input_feature_dimensions) and tensor of labels (ground truth).
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


[caffe2/operators/spatial_softmax_with_loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_softmax_with_loss_op.cc)

---



## SpatialSoftmaxWithLossGradient

No documentation yet.


### Code


[caffe2/operators/spatial_softmax_with_loss_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/spatial_softmax_with_loss_op.cc)

---



## Split


Split an  `input`  tensor into a list of tensors, along the axis specified by the  `axis`  dimension. The lengths of the split can be specified using argument  `split`  or optional second input blob to the operator. Otherwise, the tensor is split to equal sized parts.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Split",
    ["input"],
    ["output_0","output_1","output_2"],
    split=(3,2,4),
    axis=0
```

 )  workspace.FeedBlob("input", np.random.randint(10, size=(9))) print("input:", workspace.FetchBlob("input")) workspace.RunOperatorOnce(op) print("output_0:", workspace.FetchBlob("output_0")) print("output_1:", workspace.FetchBlob("output_1")) print("output_2:", workspace.FetchBlob("output_2"))   ```    **Result**    ```   input: [2 2 6 6 6 0 5 7 4] output_0: [2 2 6] output_1: [6 6] output_2: [0 5 7 4]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`axis` | (*int*): axis to split on
`split` | (*Tuple(int)*): length of each output
`order` | (*string*): order of dimensions of input and output blobs; either "NCHW" or "NHWC"
*Inputs* | 
`input` | (*Tensor*): tensor to split
`split` | (*Tensor`<int>`*): [OPTIONAL] list of output lengths (see also arg `split`)
*Outputs* | 
`[output_0, output_1, ...]` | (*Tensor*): output tensor


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc)

---



## SplitByLengths


Split a tensor into a list of tensors, given a lengths input, along the specified 'axis'. If  `K`  outputs are provided, the op assumes  `len(lengths) % K == 0` .
The  `input`  will be split into  `K`  parts. Each part of length  `sum(lengths[i*k:i*k+k))`


### Interface


---------- | ----------
*Arguments* | 
`axis` | Which axis to split on
`order` | Either NHWC or NCWH, will split on C axis, defaults to NCHW
*Inputs* | 
`input` | The tensor to split
`legnths` | The tensor `l_i` indicates the logic block of input.


### Code


[caffe2/operators/concat_split_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/concat_split_op.cc)

---



## Sqr


Performs element-wise squaring ($x^2$) of input tensor.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqr_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqr_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Sqr",
    ["X"],
    ["Y"],
```

 )  workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[4. 6. 2.]  [0. 1. 6.]  [9. 2. 7.]] Y: [[16. 36. 

```
  4.]
```

  [ 0. 

```
  1. 36.]
```

  [81. 

```
  4. 49.]]

```

 ```   </details>      


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.


### Code


[caffe2/operators/sqr_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqr_op.cc)

---



## Sqrt


Performs element-wise square-root ($\sqrt{x}$) of input tensor $X$.
 Github Link: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqrt_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqrt_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Sqrt",
    ["X"],
    ["Y"],
```

 )  workspace.FeedBlob("X", (np.random.randint(10, size=(3,3))).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[8. 3. 3.]  [4. 0. 0.]  [1. 2. 5.]] Y: [[2.8284268 

```
  1.7320508  1.7320508 ]
```

  [1.9999999 

```
  0.         0.        ]
```

  [0.99999994 1.4142134 

```
  2.236068  ]]

```

 ```   </details> 


### Interface


---------- | ----------
*Inputs* | 
`X` | *(type: Tensor`<float>`)* Input data tensor.
*Outputs* | 
`Y` | *(type: Tensor`<float>`)* Output tensor.


### Code


[caffe2/operators/sqrt_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqrt_op.cc)

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


[caffe2/operators/square_root_divide_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/square_root_divide_op.cc)

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


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## SquaredL2DistanceGradient

No documentation yet.


### Code


[caffe2/operators/distance_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc)

---



## Squeeze


The  *Squeeze*  op removes single-dimensional entries from the shape of the input tensor  *data,*  and produces a single output tensor  *squeezed* . The op also takes an argument  *dims*  with a list of dimensions to squeeze. If the same blob is provided as input and output, the operation is copy-free. This is the exact inverse operation of  *ExpandDims*  given the same  *dims*  argument.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.h)  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Squeeze",
    ["data"],
    ["squeezed"],
    dims=[0,1],
```

 )  workspace.FeedBlob("data", np.zeros((1,1,100,100)).astype(np.float32)) print("data.shape:", workspace.FetchBlob("data").shape)  workspace.RunOperatorOnce(op) print("squeezed.shape:", workspace.FetchBlob("squeezed").shape)   ```    **Result**    ```   data.shape: (1, 1, 100, 100) squeezed.shape: (100, 100)   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`dims` | *(type: [int])* List of dimensions of *data* to squeeze out.
*Inputs* | 
`data` | Input tensor of data to be operated on.
*Outputs* | 
`squeezed` | Reshaped tensor with same data as input.


### Code


[caffe2/operators/expand_squeeze_dims_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_squeeze_dims_op.cc)

---



## StatRegistryCreate


Create a StatRegistry object that will contain a map of performance counters keyed by name. A StatRegistry is used to gather and retrieve performance counts throughout the caffe2 codebase.



### Interface


---------- | ----------
*Outputs* | 
`handle` | A Blob pointing to the newly created StatRegistry.


### Code


[caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)

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


[caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)

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


[caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)

---



## StopGradient


StopGradient is a helper operator that does no actual numerical computation, and in the gradient computation phase stops the gradient from being computed through it.



### Code


[caffe2/operators/stop_gradient.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stop_gradient.cc)

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


[caffe2/operators/string_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/string_ops.cc)

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


[caffe2/operators/index_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/index_ops.cc)

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


[caffe2/operators/string_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/string_ops.cc)

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


[caffe2/operators/string_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/string_ops.cc)

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


[caffe2/operators/string_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/string_ops.cc)

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


[caffe2/operators/string_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/string_ops.cc)

---



## StumpFunc


Converts each input element into either high_ or low_value based on the given threshold.



### Interface


---------- | ----------
*Inputs* | 
`X` | tensor of float
*Outputs* | 
`Y` | tensor of float


### Code


[caffe2/operators/stump_func_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stump_func_op.cc)

---



## StumpFuncIndex


Split the elemnts and return the indices based on the given threshold.



### Interface


---------- | ----------
*Inputs* | 
`X` | tensor of float
*Outputs* | 
`Index_Low` | tensor of int64 indices for elements below/equal threshold
`Index_High` | tensor of int64 indices for elements above threshold


### Code


[caffe2/operators/stump_func_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stump_func_op.cc)

---



## Sub


Performs element-wise binary subtraction (with limited broadcast support).
 If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "Sub",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", np.array([[10,12],[4,14]])) workspace.FeedBlob("B", np.array([[5,16],[1,19]])) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [[10 12]  [ 4 14]] B: [[ 5 16]  [ 1 19]] C: [[ 5 -4]  [ 3 -5]]   ```   </details>   


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting
`axis` | *(type: int; default: -1)* Axis to concatenate on.
*Inputs* | 
`A` | *(type: Tensor`<float>`)* First operand, should share the type with the second operand.
`B` | *(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size as A.
*Outputs* | 
`C` | *(type: Tensor`<float>`)* Output tensor with same dimensions and type as A.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## SubGradient

No documentation yet.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## Sum


Element-wise sum of each of the input tensors. The first input tensor can be used in-place as the output tensor, in which case the sum will be done in place and results will be accumulated the first input tensor. All inputs and outputs must have the same shape and data type.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_sum_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_sum_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Sum",
    ["A",  "B"],
    ["C"],
```

 )  workspace.FeedBlob("A", np.array([[1,2],[3,4]]).astype(np.float32)) workspace.FeedBlob("B", np.array([[5,6],[7,8]]).astype(np.float32)) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("A"))   ```    **Result**    ```   A: [[1. 2.]  [3. 4.]] B: [[5. 6.]  [7. 8.]] C: [[1. 2.]  [3. 4.]]   ```   </details>  <details>  <summary> <b>Example 2</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Sum",
    ["A",  "B"],
    ["A"],  # inplace
```

 )  workspace.FeedBlob("A", np.array([[1,2,5],[8,3,4]]).astype(np.float32)) workspace.FeedBlob("B", np.array([[9,5,6],[6,7,8]]).astype(np.float32)) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("A after Sum:", workspace.FetchBlob("A"))   ```    **Result**    ```   A: [[1. 2. 5.]  [8. 3. 4.]] B: [[9. 5. 6.]  [6. 7. 8.]] A after Sum: [[10. 

```
  7. 11.]
```

  [14. 10. 12.]]   ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`A` | *(type: Tensor`<float>`)* First tensor to be added element-wise.
`B` | *(type: Tensor`<float>`)* Second tensor to be added element-wise.
*Outputs* | 
`C` | *(type: Tensor`<float>`)* Sum of A and B.


### Code


[caffe2/operators/elementwise_sum_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_sum_op.cc)

---



## SumElements


Sums the elements of the input tensor. Tensor type must be float32.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  sum_op = core.CreateOperator(  

```
    "SumElements",
    ["X"],
    ["Y"]
```

 )  avg_op = core.CreateOperator(  

```
    "SumElements",
    ["X"],
    ["Y"],
    average=True
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(3,3)).astype(np.float32)) print("X:\n", workspace.FetchBlob("X")) workspace.RunOperatorOnce(sum_op) print("Y (sum_op):", workspace.FetchBlob("Y")) workspace.RunOperatorOnce(avg_op) print("Y (avg_op):", workspace.FetchBlob("Y"))   ```    **Result**    ```   X:  [[7. 2. 5.]  [9. 4. 2.]  [1. 2. 5.]] Y (sum_op): 37.0 Y (avg_op): 4.111111   ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`average` | (*bool*): set to True to compute the average of the elements rather than the sum
*Inputs* | 
`X` | (*Tensor`<float>`*): blob pointing to an instance of a counter
*Outputs* | 
`sum` | (*Tensor`<float>`*): Scalar tensor containing the sum (or average)


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

---



## SumElementsGradient

No documentation yet.


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

---



## SumElementsInt

Sums the integer elements of the input tensor.


### Interface


---------- | ----------
*Inputs* | 
`X` | Tensor to sum up
*Outputs* | 
`sum` | Scalar sum


### Code


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

---



## SumInt

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

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


[caffe2/operators/reduction_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduction_ops.cc)

---



## Summarize


Summarize computes four statistics of the input tensor (Tensor)- min, max, mean and standard deviation. The output will be written to a 1-D tensor of size 4 if an output tensor is provided. Else, if the argument 'to_file' is greater than 0, the values are written to a log file in the root folder.



### Interface


---------- | ----------
*Arguments* | 
`to_file` | (int, default 0) flag to indicate if the summarized statistics have to be written to a log file.
*Inputs* | 
`data` | The input data as Tensor.
*Outputs* | 
`output` | 1-D tensor (Tensor) of size 4 containing min, max, mean and standard deviation


### Code


[caffe2/operators/summarize_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/summarize_op.cc)

---



## Swish


Swish takes one input data (Tensor) and produces one output data (Tensor) where the swish function, y = x / (1 + exp(-x)), is applied to the tensor elementwise.



### Interface


---------- | ----------
*Inputs* | 
`X` | 1D input tensor
*Outputs* | 
`Y` | 1D output tensor


### Code


[caffe2/operators/swish_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/swish_op.cc)

---



## SwishGradient


SwishGradient takes X, Y and dY and uses this to update dX according to the chain rule and derivatives of the swish function.



### Code


[caffe2/operators/swish_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/swish_op.cc)

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


[caffe2/operators/tt_linear_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tt_linear_op.cc)

---



## TTLinearGradient

No documentation yet.


### Code


[caffe2/operators/tt_linear_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tt_linear_op.cc)

---



## Tan


Calculates the tangent of the given input tensor, element-wise.



### Interface


---------- | ----------
*Inputs* | 
`input` | Input tensor
*Outputs* | 
`output` | The tangent of the input tensor computed element-wise


### Code


[caffe2/operators/tan_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tan_op.cc)

---



## TanGradient

No documentation yet.


### Code


[caffe2/operators/tan_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tan_op.cc)

---



## Tanh


Calculates the hyperbolic tangent of the given input tensor element-wise. This operation can be done in an in-place fashion too, by providing the same input and output blobs.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tanh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tanh_op.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Tanh",
    ["X"],
    ["X"],
```

 )  workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32)) print("X:\n", workspace.FetchBlob("X"), "\n")  workspace.RunOperatorOnce(op) print("X:\n", workspace.FetchBlob("X"))   ```    **Result**    ```   X:  [[ 2.032603  

```
  -2.3556721  -0.14955314]
```

  [ 0.39309832 -1.1020128 

```
  -0.92951244]
```

  [-0.62815386 

```
  0.21342885  1.4002231 ]]

```

 X:  [[ 0.9662601 

```
  -0.982175   -0.14844811]
```

  [ 0.3740282 

```
  -0.8012209  -0.73036647]
```

  [-0.55677974 

```
  0.21024609  0.8853999 ]]

```

 ```   </details>  


### Interface


---------- | ----------
*Inputs* | 
`input` | 1-D input tensor
*Outputs* | 
`output` | The hyperbolic tangent values of the input tensor, computed element-wise


### Code


[caffe2/operators/tanh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tanh_op.cc)

---



## TanhGradient

No documentation yet.


### Code


[caffe2/operators/tanh_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tanh_op.cc)

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


[caffe2/operators/tensor_protos_db_input.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tensor_protos_db_input.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/text_file_reader.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/text_file_reader.cc)

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


[caffe2/operators/thresholded_relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/thresholded_relu_op.cc)

---



## ThresholdedReluGradient


ThresholdedReluGradient takes both Y and dY and uses this to update dX according to the chain rule and derivatives of the rectified linear function.



### Code


[caffe2/operators/thresholded_relu_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/thresholded_relu_op.cc)

---



## ThrowChildThreadException

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## ThrowException

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## Tile


Constructs a tensor by tiling a given tensor along a specified axis. This operation creates a new tensor by replicating the input tensor a number of times specified by the  `tiles`  argument along the  `axis`  dimension. The output tensor's  `axis`  dimension has $(X.dims(axis) * tiles)$ elements.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Tile",
    ["X", "tiles", "axis"],
    ["Y"]
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(5,5))) workspace.FeedBlob("tiles", np.array([5]).astype(np.int32)) workspace.FeedBlob("axis", np.array([1]).astype(np.int32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Y:", workspace.FetchBlob("Y"))   ```    **Result**    ```   X: [[9 1 7 1 3]  [2 3 6 2 5]  [0 9 2 6 4]  [5 8 1 5 9]  [2 0 1 3 7]] Y: [[9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3]  [2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5]  [0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4]  [5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9]  [2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`tiles` | (*int*): number of replicas
`axis` | (*int*): axis to replicate along
*Inputs* | 
`X` | (*Tensor*): input tensor
`tiles` | (*Tensor`<int>`*): [OPTIONAL] number of replicas (overrides `tiles` argument)
`axis` | (*Tensor`<int>`*): [OPTIONAL] axis to replicate along (overrides `axis` argument)
*Outputs* | 
`Y` | (*Tensor*): output tensor


### Code


[caffe2/operators/tile_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cc)

---



## TileGradient

No documentation yet.


### Code


[caffe2/operators/tile_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cc)

---



## TimerBegin


Start a wallclock timer, returning a scalar tensor containing a pointer to it. The timer is stopped by calling  **TimerEnd** .
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)       


### Interface


---------- | ----------
*Arguments* | 
`counter_name` | (*str*): name of the timer object; if not set use output name
*Outputs* | 
`timer` | (*Tensor`<ptr>`*): pointer to a timer object


### Code


[caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)

---



## TimerEnd


Stop a timer started with  **TimerBegin** . Publishes a CAFFE_EVENT.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)       


### Interface


---------- | ----------
*Inputs* | 
`timer` | (*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op


### Code


[caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)

---



## TimerGet


Queries the current time of a timer object in nanoseconds.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)       


### Interface


---------- | ----------
*Inputs* | 
`timer` | (*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op
*Outputs* | 
`nanos` | (*Tensor`<int64>`*): scalar containing time in nanoseconds


### Code


[caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)

---



## TimerGetAndEnd


Queries the current time of a timer in nanos, stops the timer publishing a CAFFE_EVENT.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  timerbegin_op = core.CreateOperator(  

```
    "TimerBegin",
    [],
    ["timer"]
```

 )  timerget_op = core.CreateOperator(  

```
    "TimerGet",
    ["timer"],
    ["nanos"]
```

 )  timerend_op = core.CreateOperator(  

```
    "TimerEnd",
    ["timer"],
    []
```

 )  timergetandend_op = core.CreateOperator(  

```
    "TimerGetAndEnd",
    ["timer"],
    ["nanos"]
```

 )  # Test TimerBegin/TimerGet/TimerEnd workspace.RunOperatorOnce(timerbegin_op) print("timer:", workspace.FetchBlob("timer")) workspace.RunOperatorOnce(timerget_op) print("nanos:", workspace.FetchBlob("nanos")) workspace.RunOperatorOnce(timerend_op)   # Test TimerBegin/TimerGetAndEnd workspace.RunOperatorOnce(timerbegin_op) print("timer:", workspace.FetchBlob("timer")) workspace.RunOperatorOnce(timergetandend_op) print("nanos:", workspace.FetchBlob("nanos"))   ```    **Result**    ```   timer: b'timer, a C++ native class of type caffe2::TimerInstance *.' nanos: 361140 timer: b'timer, a C++ native class of type caffe2::TimerInstance* .' nanos: [252250]   ```   </details>        


### Interface


---------- | ----------
*Inputs* | 
`timer` | (*Tensor`<ptr>`*): pointer to a timer object; obtained from **TimerBegin** op
*Outputs* | 
`nanos` | (*Tensor`<int64>`*): scalar tensor containing time in nanoseconds


### Code


[caffe2/operators/stats_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/stats_ops.cc)

---



## TopK


Retrieve the top-K elements of the last dimension. Given an input tensor of shape $(a_1, a_2, ..., a_n, r)$ and integer argument  `k` , return up to three outputs:  1. Value tensor of shape $(a_1, a_2, ..., a_n, k)$ which contains the values of the top k elements along the last dimension 2. Index tensor of shape $(a_1, a_2, ..., a_n, k)$ which contains the indices of the top k elements (original indices from the input tensor).
3. [OPTIONAL] Flattened index tensor of shape $(a_1  * a_2 *  ...  * a_n *  k,)$.
 Given two equivalent values, this operator uses the indices along the last dimension as a tiebreaker. That is, the element with the lower index will appear first.
 Github Links: -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k.cc)    <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "TopK",
    ["X"],
    ["Values", "Indices", "Flattened_indices"],
    k=2
```

 )  workspace.FeedBlob("X", np.random.randint(10, size=(3,3,3)).astype(np.float32)) print("X:", workspace.FetchBlob("X")) workspace.RunOperatorOnce(op) print("Values:", workspace.FetchBlob("Values")) print("Indices:", workspace.FetchBlob("Indices")) print("Flattened_indices:", workspace.FetchBlob("Flattened_indices"))   ```    **Result**    ```   X: [[[6. 7. 0.]  

```
  [8. 7. 7.]
  [1. 5. 6.]]

```

  [[0. 6. 1.]  

```
  [2. 8. 4.]
  [1. 2. 9.]]

```

  [[4. 3. 7.]  

```
  [0. 1. 7.]
  [0. 1. 8.]]]
```

 Values: [[[7. 6.]  

```
  [8. 7.]
  [6. 5.]]

```

  [[6. 1.]  

```
  [8. 4.]
  [9. 2.]]

```

  [[7. 4.]  

```
  [7. 1.]
  [8. 1.]]]
```

 Indices: [[[1 0]  

```
  [0 1]
  [2 1]]

```

  [[1 2]  

```
  [1 2]
  [2 1]]

```

  [[2 0]  

```
  [2 1]
  [2 1]]]
```

 Flattened_indices: [ 1 

```
  0  3  4  8  7 10 11 13 14 17 16 20 18 23 22 26 25]

```

 ```   </details>    


### Interface


---------- | ----------
*Arguments* | 
`k` | (*int*): number of top elements to retrieve
*Inputs* | 
`X` | (*Tensor`<float>`*): input tensor of shape $(a_1, a_2, ..., a_n, r)$
*Outputs* | 
`Values` | (*Tensor`<float>`*): output tensor of shape $(a_1, a_2, ..., a_n, k)$
`Indices` | (*Tensor`<int>`*): tensor of indices of shape $(a_1, a_2, ..., a_n, k)$; indices values refer to each element's index in the last dimension of the `X` input tensor
`Flattened_indices` | (*Tensor`<int>`*): tensor of indices of shape $(a_1 * a_2 * ... * a_n * k,)$; indices values refer to each element's index in the flattened input tensor `X`


### Code


[caffe2/operators/top_k.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k.cc)

---



## TopKGradient

No documentation yet.


### Code


[caffe2/operators/top_k.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k.cc)

---



## Transpose


Transpose the input tensor by permuting the axes of the input according to the  `axes`  argument. Similar to numpy's [transpose]( [https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html))  function.
 For example, when axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape will be (2, 1, 3).
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/transpose_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/transpose_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```  workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "Transpose",
    ["X"],
    ["Y"],
    axes=(0,3,1,2)
```

 )  x = np.random.rand(1,32,32,3) workspace.FeedBlob("X", x) print("X.shape (NHWC order):", workspace.FetchBlob("X").shape) workspace.RunOperatorOnce(op) print("Y.shape (NCHW order):", workspace.FetchBlob("Y").shape)  ```    **Result**    ```  X.shape (NHWC order): (1, 32, 32, 3) Y.shape (NCHW order): (1, 3, 32, 32)  ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`axes` | *(type: Tuple(int))* Order to permute axes of input tensor. Reverses the dimensions by default.
*Inputs* | 
`X` | *(type: Tensor)* Input tensor.
*Outputs* | 
`Y` | *(type: Tensor)* Transposed output.


### Code


[caffe2/operators/transpose_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/transpose_op.cc)

---



## TrimDataset


Trim the given dataset inplace, given the dataset blobs and the field specs.
Trimming happens such that the dataset will contain the largest possible number of records that is a multiple of the 'multiple_of' argument.



### Interface


---------- | ----------
*Arguments* | 
`fields` | List of strings representing the string names in the formatspecified in the doc for CreateTreeCursor.


### Code


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

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


[caffe2/operators/dataset_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc)

---



## UniformFill


Fill the output tensor with float samples from uniform distribution [ `min` ,  `max` ].
 - The range can be defined either by arguments or input blobs.  `min`  and  `max`  are inclusive.
 

```
    - If the range is given by input blobs, you also need to give the shape as input.
    - When the range is given as arguments, this operator enforces min <= max. When the range is given as inputs, the constraint is not enforced.
    - When the range is given as inputs and max < min, the first dimension of the output is set to 0. This behavior is allowed so that dynamically sampling indices into a dynamically sized tensor is possible.
```

 - The shape of the output can be given as argument or input.
 Github Links: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op_1 = core.CreateOperator(  

```
    "UniformFill",
    [],
    ["output"],
    min=5.5,
    max=10.5,
    shape=(3,3)
```

 )  op_2 = core.CreateOperator(  

```
    "UniformFill",
    ["shape", "min", "max"],
    ["output"],
    input_as_shape=1
```

 )  # Test arg-based op workspace.RunOperatorOnce(op_1) print("output (op_1):\n", workspace.FetchBlob("output"))  # Test input-based op workspace.ResetWorkspace() workspace.FeedBlob("shape", np.array([5,5])) workspace.FeedBlob("min", np.array(13.8, dtype=np.float32)) workspace.FeedBlob("max", np.array(19.3, dtype=np.float32)) workspace.RunOperatorOnce(op_2) print("output (op_2):\n", workspace.FetchBlob("output"))   ```    **Result**    ```   output (op_1):  [[8.894862 

```
  8.225005  6.7890406]
```

  [9.588293 

```
  7.1072135 7.7234955]
```

  [8.210596 

```
  6.0202913 9.665462 ]]
```

 output (op_2):  [[18.965155 15.603871 15.038921 17.14872 

```
  18.134571]
```

  [18.84237 

```
  17.845276 19.214737 16.970337 15.494069]
```

  [18.754795 16.724329 15.311974 16.962536 18.60965 ]  [15.186268 15.264773 18.73341 

```
  19.077969 14.237255]
```

  [15.917589 15.844325 16.248466 17.006554 17.502048]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`min` | (*float*): minimum value, inclusive
`max` | (*float*): maximum value, inclusive
`shape` | (*Tuple(int)*): shape of the output, do not set when `input_as_shape`=1
`input_as_shape` | (*int*): set to 1 to use the first input as shape; `shape` input must be in CPU context
*Inputs* | 
`shape` | (*Tensor`<int>`*): 1-D tensor of the shape of the output, must be used with `input_as_shape` argument
`min` | (*Tensor`<float>`*): scalar tensor containing minimum value, inclusive
`max` | (*Tensor`<float>`*): scalar tensor containing maximum value, inclusive
*Outputs* | 
`output` | (*Tensor`<float>`*): filled output tensor


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

---



## UniformIntFill


Fill the output tensor with int32 samples from uniform distribution [ `min` ,  `max` ].
 - The range can be defined either by arguments or input blobs.  `min`  and  `max`  are inclusive.
 

```
    - If the range is given by input blobs, you also need to give the shape as input.
    - When the range is given as arguments, this operator enforces min <= max. When the range is given as inputs, the constraint is not enforced.
    - When the range is given as inputs and max < min, the first dimension of the output is set to 0. This behavior is allowed so that dynamically sampling indices into a dynamically sized tensor is possible.
```

 - The shape of the output can be given as argument or input.
 Github Links: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op_1 = core.CreateOperator(  

```
    "UniformIntFill",
    [],
    ["output"],
    min=5,
    max=10,
    shape=(3,3)
```

 )  op_2 = core.CreateOperator(  

```
    "UniformIntFill",
    ["shape", "min", "max"],
    ["output"],
    input_as_shape=1
```

 )  # Test arg-based op workspace.RunOperatorOnce(op_1) print("output (op_1):\n", workspace.FetchBlob("output"))  # Test input-based op workspace.ResetWorkspace() workspace.FeedBlob("shape", np.array([5,5])) workspace.FeedBlob("min", np.array(13, dtype=np.int32)) workspace.FeedBlob("max", np.array(19, dtype=np.int32)) workspace.RunOperatorOnce(op_2) print("output (op_2):\n", workspace.FetchBlob("output"))   ```    **Result**    ```   output (op_1):  [[ 6 10 

```
  7]
```

  [ 5 10 

```
  6]
```

  [ 7 

```
  5 10]]
```

 output (op_2):  [[19 13 15 13 13]  [14 17 14 15 15]  [17 14 19 13 13]  [17 18 16 13 18]  [14 15 16 18 16]]   ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`min` | (*int*): minimum value, inclusive
`max` | (*int*): maximum value, inclusive
`shape` | (*Tuple(int)*): shape of the output, do not set when `input_as_shape`=1
`input_as_shape` | (*int*): set to 1 to use the first input as shape; `shape` input must be in CPU context
*Inputs* | 
`shape` | (*Tensor`<int>`*): 1-D tensor of the shape of the output, must be used with `input_as_shape` argument
`min` | (*Tensor`<int>`*): scalar tensor containing minimum value, inclusive
`max` | (*Tensor`<int>`*): scalar tensor containing maximum value, inclusive
*Outputs* | 
`output` | (*Tensor`<int>`*): filled output tensor


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

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


[caffe2/operators/unique_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/unique_ops.cc)

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


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

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


[caffe2/operators/pack_rnn_sequence_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pack_rnn_sequence_op.cc)

---



## UnpackSegments

Map N+1 dim tensor to N dim based on length blob


### Interface


---------- | ----------
*Arguments* | 
`max_length` | The pre-defined max_length for the packed segments
*Inputs* | 
`lengths` | 1-d int/long tensor contains the length in each of the input.
`tensor` | N+1 dim Tensor.
*Outputs* | 
`packed_tensor` | N dim Tensor


### Code


[caffe2/operators/pack_segments.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pack_segments.cc)

---



## UnsafeCoalesce


Coalesce the N inputs into N outputs and a single coalesced output blob.
 This allows operations that operate over multiple small kernels (e.g.
biases in a deep CNN) to be coalesced into a single larger operation, amortizing the kernel launch overhead, synchronization costs for distributed computation, etc.
 The operator:  - computes the total size of the coalesced blob by summing the input sizes - allocates the coalesced output blob as the total size - copies the input vectors into the coalesced blob, at the correct offset.
- aliases each Output(i) to- point into the coalesced blob, at the corresponding offset for Input(i).
 This is 'unsafe' as the output vectors are aliased, so use with caution.
 


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentMeanGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

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


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UnsortedSegmentWeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/segment_reduction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/segment_reduction_op.cc)

---



## UpsampleBilinear


Resizes the spatial dimensions of the input using bilinear interpolation. The  `width_scale`  and  `height_scale`  arguments control the size of the output, which is given by: output_width = floor(input_width  * width_scale) output_height = floor(output_height *  height_scale) 


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


[caffe2/operators/upsample_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/upsample_op.cc)

---



## UpsampleBilinearGradient

No documentation yet.


### Interface


---------- | ----------
*Arguments* | 
`width_scale` | Scale along width dimension
`height_scale` | Scale along height dimension


### Code


[caffe2/operators/upsample_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/upsample_op.cc)

---



## VariableLengthSequencePadding


Super special-case operator. Used to pad a tensor to mimic pytorch's pad_packed_sequence.
 Given an input tensor INPUT of size NxBxM and an input tensor LENS of size B, where  N = maximum sequence length B = batch size M = hidden size  set each element of INPUT to zero if it is is past the end of the corresponding sequence (i.e. if LENS[j] > i for an index (i,j,k)).
 


### Code


[caffe2/operators/variable_length_sequence_padding.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/variable_length_sequence_padding.cc)

---



## WallClockTime

Time since epoch in nanoseconds.


### Interface


---------- | ----------
*Outputs* | 
`time` | The time in nanoseconds.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## WeightedMultiSampling


The operator performs sampling based on the input sampling weights.
All weights are cummulative probability thus sorted. The output is a 1-D tensor (Tensor). If two inputs are given, the second input is used to provide shape of the output sample tensor. Otherwise, we use argument  `num_samples`  to determine the number of samples to generate.



### Interface


---------- | ----------
*Arguments* | 
`num_samples` | number of samples to sample from the input data
*Inputs* | 
`sampling_cdf` | An optional 1-D Tensor.Input cumulative sampling probability (such as [0.2, 0.5, 0.8, 1.5]). All weights must be non-negative numbers. Note that the last value of CDF is not necessary 1. If the last value is not 1, all values in sampling_cdf will be scaled by this number.
`shape_tensor (optional)` | Tensor whose shape will be applied to output.
*Outputs* | 
`sampled_indexes` | The output tensor contains indices sampled from distribution givenby the weight vector in the input tensorThe output is a 1-D Tensor of size determined by argument`num_samples` or the second input tensor.


### Code


[caffe2/operators/weighted_multi_sampling_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/weighted_multi_sampling_op.cc)

---



## WeightedSample


The operator performs sampling based on the input sampling weights for each batch. All weights must be non-negative numbers.
The input is a 2-D tensor (Tensor) of size (batch_size x weights_dim).
For each batch, an index is randomly sampled from the distribution given by the weights of the corresponding batch.
The output is a 1-D tensor (Tensor) of size (batch_size x 1) and contains the index(es) of the sampled output.



### Interface


---------- | ----------
*Inputs* | 
`sampling_weights` | A 2-D Tensor of size (batch_size x weights_dim).All weights must be non-negative numbers.
`sampling_values` | An optional 2-D Tensor of size (batch_size x weights_dim).Its values correspond to the sampling weights.
*Outputs* | 
`sampled_indexes` | The output tensor contains index(es) sampled from distribution givenby the weight vector(s) in the input tensorThe output is a 1-D Tensor of size (batch_size x 1)
`sampled_values` | The output tensor contains value(s) selected by the sampled index(es)It is a 1-D Tensor of size (batch_size x 1)


### Code


[caffe2/operators/weighted_sample_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/weighted_sample_op.cc)

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


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

---



## WeightedSigmoidCrossEntropyWithLogitsGradient

No documentation yet.


### Code


[caffe2/operators/cross_entropy_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cross_entropy_op.cc)

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


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## WeightedSumGradient

No documentation yet.


### Code


[caffe2/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/utility_ops.cc)

---



## Where


Operator Where takes three input data (Tensor, Tensor, Tensor) and produces one output data (Tensor) where z = c ? x : y is applied elementwise.



### Interface


---------- | ----------
*Inputs* | 
`C` | input tensor containing booleans
`X` | input tensor
`Y` | input tensor
*Outputs* | 
`Z` | output tensor


### Code


[caffe2/operators/elementwise_logical_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_logical_ops.cc)

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


[caffe2/operators/while_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/while_op.cc)

---



## XavierFill


This op fills an output tensor with values sampled from a uniform distribution with the range determined by the desired shape of the output. Rather, than specifying the range of values manually, the novelty of Xavier Fill is that it automatically scales the range of the distribution it draws from based on the size of the desired output tensor. For more information check out the paper [Understanding the difficulty of training deep feedforward neural networks]( [http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).)  The output tensor shape is specified by the  *shape*  argument. However, if  *input_as_shape*  is set to  *true* , then the  *input*  should be a 1D tensor containing the desired output shape (the dimensions specified in  *extra_shape*  will also be appended). In this case, the  *shape*  argument should  **not**  be set.
  *Note: Do not set the shape argument and pass in an input at the same time.*   Github Links: -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h)  -  [https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc)   <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  

```
    "XavierFill",
    [],
    ["out"],
    shape=[3,3],
```

 )  workspace.RunOperatorOnce(op) print("Out:\n", workspace.FetchBlob("out"))   ```    **Result**    ```   Out:  [[-0.8412168  

```
  0.33207083 -0.88418937]
```

  [ 0.43059897 -0.8340702  

```
  0.07781601]
```

  [ 0.93261135 -0.24542928 -0.3980782 ]]   ```   </details>  


### Interface


---------- | ----------
*Arguments* | 
`shape` | *(type: [int])* Desired shape of the *output* tensor.
`extra_shape` | *(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob.
`input_as_shape` | *(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.
*Inputs* | 
`input` | (Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*
*Outputs* | 
`output` | Output tensor of random values drawn from an automatically scaled uniform distribution, based on the size of the output tensor. If the shape argument is set, this is the shape specified by the shape argument, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.


### Code


[caffe2/operators/filler_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.cc)

---



## Xor


Performs element-wise logical operation  **xor**  (with limited broadcast support).
Both input operands should be of type  `bool` .
  If necessary the right-hand-side argument will be broadcasted to match the shape of left-hand-side argument. When broadcasting is specified, the second tensor can either be of size 1 (a scalar value), or having its shape as a contiguous subset of the first tensor's shape. The starting of the mutually equal shape is specified by the argument "axis", and if it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 For example, the following tensor shapes are supported (with broadcast=1):  ```   

```
  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
```

 

```
    "Xor",
    ["A",  "B"],
    ["C"],
```

 ```  Argument  `broadcast=1`  needs to be passed to enable broadcasting.
 Github Links:  -  [https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_op_schema.cc)      <details>  <summary> <b>Example</b> </summary>   **Code**    ```   workspace.ResetWorkspace()  op = core.CreateOperator(  )  workspace.FeedBlob("A", (np.random.rand(3, 3) > 0.5)) workspace.FeedBlob("B", (np.random.rand(3, 3) > 0.5)) print("A:", workspace.FetchBlob("A")) print("B:", workspace.FetchBlob("B")) workspace.RunOperatorOnce(op) print("C:", workspace.FetchBlob("C"))   ```    **Result**    ```   A: [[ True 

```
  True  True]
```

  [False False 

```
  True]
```

  [False 

```
  True False]]
```

 B: [[False False False]  [ True 

```
  True  True]
```

  [False False False]] C: [[ True 

```
  True  True]
```

  [ True 

```
  True False]
```

  [False 

```
  True False]]

```

 ```   </details>      


### Interface


---------- | ----------
*Arguments* | 
`broadcast` | *(type: int; default: 0)* Pass 1 to enable broadcasting.
`axis` | *(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.
*Inputs* | 
`A` | *(type: Tensor`<bool>`)* First operand.
`B` | *(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.
*Outputs* | 
`C` | *(type: Tensor`<bool>`)* Output tensor of booleans. Has same dimensions as input `A`.


### Code


[caffe2/operators/elementwise_ops_schema.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc)

---



## ZeroGradient


ZeroGradient operators doesn't produce any output blobs. One can use this operator to produce 0 gradient for the input blob.



### Code


[caffe2/operators/zero_gradient_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/zero_gradient_op.cc)

---



## Adadelta


 Computes the AdaDelta update ( [https://arxiv.org/abs/1212.5701)](https://arxiv.org/abs/1212.5701))  for an input gradient and accumulated history of squared gradients. Concretely, given inputs (param, moment, moment_delta, grad, learning_rate), computes:   

```
    new_moment = moment * decay + square(grad) * (1 - decay)
    new_grad = sqrt(moment_delta + epsilon) / sqrt(new_moment + epsilon) * grad
    new_param = param + learning_rate * new_grad
    new_moment_delta = moment_delta * decay + square(new_grad) * (1 - decay)

```

 and returns (new_param, new_moment, new_moment_delta).
 


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | Default 1e-5
`decay` | Default 0.95, the squared gradient sum is decayed by this factor.
*Inputs* | 
`param` | Parameters to be updated
`moment` | Average of squared gradients
`moment_delta` | Average of squared parameter updates
`grad` | Gradient computed
`lr` | Learning rate
*Outputs* | 
`output_param` | Updated parameters
`output_moment` | Updated average squared gradient
`output_moment_delta` | Updated average of squared parameter updates


### Code


[caffe2/sgd/adadelta_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adadelta_op.cc)

---



## Adagrad


 Computes the AdaGrad update for an input gradient and accumulated history. Concretely, given inputs (param, grad, moment, learning_rate), computes   

```
    new_moment = moment + square(grad)
    effective_lr = learning_rate / (sqrt(new_moment) + epsilon)
    update = learning_rate * grad / (sqrt(new_moment) + epsilon)
    new_param = param + update
```

 and returns (new_param, new_moment).
 Optionally returns effective_lr and update as well.
 


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
`output_effective_lr` | (optional) Effective learning rate
`output_update` | (optional) Actual update that is applied.


### Code


[caffe2/sgd/adagrad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adagrad_op.cc)

---



## Adam


 Computes the Adam update ( [https://arxiv.org/abs/1412.6980)](https://arxiv.org/abs/1412.6980))  for an input gradient and momentum parameters. Concretely, given inputs (param, m1, m2, grad, lr, iters),   

```
    t = iters + 1
    correction_multiplier = sqrt(1 - power(beta2, t)) /
      (1 - power(beta1, t))
    m1_o = (beta1 * m1) + (1 - beta1) * grad
    m2_o = (beta2 * m2) + (1 - beta2) * np.square(grad)
    grad_o = correction_multiplier * m1_o / \
        (sqrt(m2_o) + epsilon)
    param_o = param + lr * grad_o

```

 and returns (param_o, m1_o, m2_o, grad_o), in which grad_o is an optional output  


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
`output_grad` | Effective grad


### Code


[caffe2/sgd/adam_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adam_op.cc)

---



## AnyNotEq


Return true if any of the input elements is not equal to the target value.
Otherwise return false.



### Interface


---------- | ----------
*Inputs* | 
`Input` | tensor of int64
*Outputs* | 
`Output` | scalar of bool


### Code


[caffe2/fb/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/utility_ops.cc)

---



## AtomicIter


Similar to Iter, but takes a mutex as the first input to make sure that updates are carried out atomically. This can be used in e.g. Hogwild sgd algorithms.



### Interface


---------- | ----------
*Inputs* | 
`mutex` | The mutex used to do atomic increment.
`iter` | The iter counter as an int64_t TensorCPU.


### Code


[caffe2/sgd/iter_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/iter_op.cc)

---



## ClipTensorByScaling


 

```
    Clips the input tensor by scaling based on the input value and the threshold.
    The value is usually the (pre-computed) norm of the tensor. If the value is
    larger than the threshold, scaling would be performed in this way:

          tensor *= (threshold / value).

    An optional input called additional_threshold can be provided which
    will scale the original threshold before it is used. That is,
    the final threshold will become threshold * additional_threshold.
    This op could be used for gradient clipping.
```




### Interface


---------- | ----------
*Arguments* | 
`threshold` | Threshold to determine whether to scale down the tensor
*Inputs* | 
`input_tensor` | Tensor of floats to be clipped.
`val` | Value to be compared against the threshold
`additional_threshold` | An optional additonal threshold to scale the orignal threshold
*Outputs* | 
`clipped` | Tensor of floats, which is the same size as the input tensor, representing the clipped tensor.


### Code


[caffe2/sgd/clip_tensor_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/clip_tensor_op.cc)

---



## CloseBlobsQueue

No documentation yet.


### Code


[caffe2/queue/queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/queue_ops.cc)

---



## CloseRebatchingQueue


Closes the Queue.



### Interface


---------- | ----------
*Inputs* | 
`queue` | object representing the queue


### Code


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/rebatching_queue_ops.cc)

---



## CreateBlobsQueue

No documentation yet.


### Code


[caffe2/queue/queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/queue_ops.cc)

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


[caffe2/queue/blobs_queue_db.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/blobs_queue_db.cc)

---



## CreateDB

No documentation yet.


### Code


[caffe2/db/create_db_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/db/create_db_op.cc)

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


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/rebatching_queue_ops.cc)

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


[caffe2/queue/queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/queue_ops.cc)

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


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/rebatching_queue_ops.cc)

---



## EnqueueBlobs

No documentation yet.


### Code


[caffe2/queue/queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/queue_ops.cc)

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


[caffe2/queue/rebatching_queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/rebatching_queue_ops.cc)

---



## FP16MomentumSGDUpdate


 Computes the momentum SGD update similarly to the MomentumSGDUpdateOp, however this op also performs the weight decay update at the same time, thus making it more efficient.
 This op is also functionally equivalent to the FP32MomentumSGDUpdateOp, however it expects FP16 data and performs its updates in either FP16 precision (default), or FP32 precision if the 'fp32_update' flag is set to True.
 


### Code


[caffe2/sgd/fp16_momentum_sgd_op.cu](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/fp16_momentum_sgd_op.cu)

---



## FP32MomentumSGDUpdate


 Computes the momentum SGD update similarly to the MomentumSGDUpdateOp, however this op also performs the weight decay update at the same time, thus making it more efficient.
 This op is also functionally equivalent to the FP16MomentumSGDUpdateOp, however it expects FP32 data and performs its updates in FP32 precision.
 


### Code


[caffe2/sgd/fp32_momentum_sgd_op.cu](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/fp32_momentum_sgd_op.cu)

---



## FbFCPacked

Same as FC,       but the weight is prepacked as a fbgemm::PackedGemmMatrix<TW>


### Code


[caffe2/fb/fbgemm/fb_fc_packed_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/fbgemm/fb_fc_packed_op.cc)

---



## FbGemmPack

Prepack weight for fbgemm


### Interface


---------- | ----------
*Inputs* | 
`X` | row major format weight matrix
*Outputs* | 
`Y` | Block row major packed format weight matrix


### Code


[caffe2/fb/fbgemm/fb_gemm_pack.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/fbgemm/fb_gemm_pack.cc)

---



## FilterExampleIds


This op allows us to do example splitting with a subset of batch. The first input (when full_batch_input=False) is the indices of the examples in the subset, and from the 2nd input, they are dense blobs of label values (for different tasks) in boolean (true means the label exists and false means absent). we filter the examples by checking for each of them if they have at least one is true. For example, if input is: [1, 2, 4, 5] (selecting 1,2,4,5 th examples from the batch), [1, 1, 1, 1, 0, 0] (their labels of the first task), and [1, 1, 1, 1, 0, 1] (their labels of the second task), and full_batch_input=False (use only partial batch as source for filtering) then the output is: [1, 2, 5] because the label value for 4 in both tasks are false; when setting full_batch_input, the first input is not needed since it is inferred as 0, ..., batch_size-1   


### Interface


---------- | ----------
*Arguments* | 
`full_batch_input` | boolean, if false, input 0 shall be the (absolute) index (in batch) of the input, otherwise, assume the ids to be filtered is the full batch
*Outputs* | 
`rel_idx` | the (relative) index of the output
`abs_idx` | the (absolute) index (in batch) of the output


### Code


[caffe2/fb/operators/label_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/label_ops.cc)

---



## FilterSparseLabels


Given a string for index_to_tasks, whose format is like 'index_1:task_id_1,task_id_2;index_2:task_id_3', input labels are filtered accordingly.
    


### Interface


---------- | ----------
*Arguments* | 
`index_to_tasks` | string representation of map from index to tasks.
*Inputs* | 
`lengths` | Length of keys/values for each example.
`keys` | List of keys.
`values` | List of labels.
`index` | Index used for selecting a filter.
*Outputs* | 
`filtered_lengths` | Length of flitered keys/values for each example.
`filtered_keys` | List of filtered keys.
`filtered_values` | List of filtered labels.


### Code


[caffe2/fb/operators/label_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/label_ops.cc)

---



## Ftrl

No documentation yet.


### Code


[caffe2/sgd/ftrl_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/ftrl_op.cc)

---



## GFtrl

No documentation yet.


### Code


[caffe2/sgd/gftrl_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/gftrl_op.cc)

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


[caffe2/image/image_input_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/image/image_input_op.cc)

---



## Iter


Stores a singe integer, that gets incremented on each call to Run().
Useful for tracking the iteration count during SGD, for example.



### Code


[caffe2/sgd/iter_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/iter_op.cc)

---



## Lars


Implement Layer-wise Adaptive Rate Scaling (LARS) with clipping. Before adding weight decay, given a parameter tensor X and its gradient dX, the local learning rate for X will be  local_lr = trust  * norm(X) / ( norm(dX) + wd *  norm(X) + offset * norm(X) )   

```
      = trust / ( norm(dX) / norm(X) + wd + offset ),

```

 where offset is a preset hyper-parameter to avoid numerical issue and trust indicates how much we trust the layer to change its parameters during one update.
In this implementation, we uses l2 norm and the computed local learning rate is clipped based on the upper bound lr_max and the lower bound lr_min:  local_lr = min(local_lr, lr_max) and local_lr = max(local_lr, lr_min)  


### Interface


---------- | ----------
*Arguments* | 
`offset` | rescaling offset parameter
`lr_min` | minimum learning rate for clipping
*Inputs* | 
`X` | Parameter tensor
`dX` | Gradient tensor
`wd` | Weight decay
`trust` | Trust
`lr_max` | Upper bound of learning rate
*Outputs* | 
`lr_rescaled` | Rescaled local learning rate


### Code


[caffe2/sgd/lars_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/lars_op.cc)

---



## LearningRate


Learning rate is a decreasing function of time. With low learning rates the improvements will be linear. With high learning rates they will start to look more exponential. Learning rate is controlled by the following arguments:   Required:   `iterations`    `base_lr` : base learning rate   `policy` : this controls how the learning rate is applied, options are:   

```
  `fixed`
```

   

```
  `step`: uses `stepsize`, `gamma`
```

   

```
  `exp`: uses `gamma`
```

   

```
  `inv`: uses `gamma`, `power`
```

   

```
  `linearWarmup`: uses `start_multiplier`, `num_iter`
```

   

```
  `constantWarmup`: uses `multiplier`, `num_iter`
```

   

```
  `alter`: uses  `active_first`, `active_period`, `inactive_period`
```

   

```
  `hill`: uses those in both `linearWarmup` and `inv`, plus `end_multiplier`
```

   

```
  `composite`: uses `sub_policy_num_iters` and additional args with format
```

   

```
  sub_policy_{sub_policy_index}_{sub_policy_arg}, for example:
```

   

```
  sub_policy_0_policy: "exp", sub_policy_0_gamma: 0.99,
```

   

```
  sub_policy_0_lr_scale: 1.2
```

   

```
  sub_policy_0_policy: "fixed", sub_policy_0_lr_scale: 1.0
```

   

```
  sub_policy_num_iters: [1000, 1000]

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
`sub_policy_num_iters` | (int array, default empty) number of iterations for each sub learning rate policy in composite policy
*Inputs* | 
`input` | description needed
*Outputs* | 
`output` | description needed


### Code


[caffe2/sgd/learning_rate_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/learning_rate_op.cc)

---



## LearningRateAdaption


 

```
      Learning Rate Adaption is an operation that perform one iteration of
      gradient descent based on learning rate:
        lr(k) = lr(k-1) - lr_alpha * df(k-1)/dlr,
      where df(k-1)/dlr is the gradient of objective function f on lr, and
      lr_alpha is a learning rate hyperparameter. It can be prove that
      df(k-1)/dlr equals INNERPRODUCT(grad(k-1), -grad(k-2)), where grad(k-1) is
      the grad of f(k-1) on parameters. When the argument
      "normalized_lr_adaption" is false, we simply perform the
      following update:
      lr(k) = lr(k-1) - lr_alpha * INNERPRODUCT(grad(k-1), grad(k-2)).
      If we set "normalized_lr_adaption" to be true, we do not directly apply
      INNERPRODUCT(grad(k-1), -grad(k-2)) as the grad. Instead, we perform the
      following update:
      lr(k) = lr(k-1) + lr_alpha * cosineSimilarity(grad(k-1), grad(k-2)).
```




### Interface


---------- | ----------
*Arguments* | 
`lr_alpha` | the learning rate for performing gradient descent on learning rate lr
`normalized_lr_adaption` | whether to apply normalized lr adaption or not
*Inputs* | 
`lr` | Learning rate
`grad` | Gradient computed
`effgrad` | The effective grad
*Outputs* | 
`output_lr` | Updated learning rate


### Code


[caffe2/sgd/learning_rate_adaption_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/learning_rate_adaption_op.cc)

---



## MergeExampleIds


Given a variable number of tensors of integers (example ids) as input, and argument merge_by_labels as a list of integers, it produces another of tensor of integers which are elements merged from those selected tensors.
    


### Interface


---------- | ----------
*Arguments* | 
`merge_by_labels` | list(int)


### Code


[caffe2/fb/operators/label_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/label_ops.cc)

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


[caffe2/sgd/momentum_sgd_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/momentum_sgd_op.cc)

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


[caffe2/sgd/momentum_sgd_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/momentum_sgd_op.cc)

---



## PSCall

Triggers an asynchronous call to a Parameter Server


### Interface


---------- | ----------
*Arguments* | 
`server_id` | Target PS server ID
`param_id` | Parameter ID
`shard_id` | Shard ID
`request_type` | Request type
`compression_codec` | Compression codec for data
`compression_level` | Compression level for data
`use_const_ref` | Do not consume input.
*Inputs* | 
`client` | unique_ptr<PSClient> to use


### Code


[caffe2/fb/async/pscall_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/async/pscall_op.cc)

---



## PackedFC


Computes the result of passing an input vector X into a fully connected layer with 2D weight matrix W and 1D bias vector b. This is essentially the same as the FC operator but allows one to pack the weight matrix for more efficient inference. See the schema for the FC op for details.
 Unlike many other operators in Caffe2, this operator is stateful: it assumes that the input weight matrix W never changes, so it is only suitable for inference time when the weight matrix never gets updated by any other ops.
Due to performance considerations, this is not checked in non-debug builds.



### Code


[caffe2/mkl/operators/packed_fc_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/mkl/operators/packed_fc_op.cc)

---



## Python

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/python/pybind_state.cc)

---



## PythonDLPack

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/python/pybind_state.cc)

---



## PythonDLPackGradient

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/python/pybind_state.cc)

---



## PythonGradient

No documentation yet.


### Code


[caffe2/python/pybind_state.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/python/pybind_state.cc)

---



## RecurrentNetwork


Run the input network in a recurrent fashion. This can be used to implement fairly general recurrent neural networks (RNNs).
 The operator proceeds as follows.
 - First, initialized the states from the input recurrent states - For each timestep T, apply the links (that map offsets from input/output tensors into the inputs/outputs for the  `step`  network) - Finally, alias the recurrent states to the specified output blobs.
 This is a fairly special-case meta-operator, and so the implementation is somewhat complex. It trades of generality (and frankly usability) against performance and control (compared to e.g. TF dynamic_rnn, Theano scan, etc).
 See the usage examples for a flavor of how to use it.



### Code


[caffe2/operators/rnn/recurrent_network_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rnn/recurrent_network_op.cc)

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


[caffe2/operators/rnn/recurrent_network_blob_fetcher_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rnn/recurrent_network_blob_fetcher_op.cc)

---



## RecurrentNetworkGradient

No documentation yet.


### Code


[caffe2/operators/rnn/recurrent_network_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rnn/recurrent_network_op.cc)

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


[caffe2/sgd/rmsprop_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/rmsprop_op.cc)

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


[caffe2/sgd/adagrad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adagrad_op.cc)

---



## RowWiseSparseAdam


  

```
    Computes a modified Adam Update for the sparse case.
    Given inputs (param, moment1, moment2, indices, grad, lr, iter), runs the
    Adam update on (param, moment1[indices], moment2[indices], lr, iter) and returns
    (new_param, new_moment1, new_moment2), where moment2 is a 1D tensor
    with length equal to the number of rows in param:
    shape(moment2) == shape(param)[0]. Each element of  moment2 is
    applied to an entire row of param, and the new moment2 values are
    calculated by averaging across the row.

```

     


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


[caffe2/sgd/adam_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adam_op.cc)

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


[caffe2/queue/queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/queue_ops.cc)

---



## SafeEnqueueBlobs


Enqueue the blobs into queue. When the queue is closed and full, the output status will be set to true which can be used as exit criteria for execution step.
The 1st input is the queue and the last output is the status. The rest are data blobs.



### Interface


---------- | ----------
*Inputs* | 
`queue` | The shared pointer for the BlobsQueue


### Code


[caffe2/queue/queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/queue_ops.cc)

---



## SparseAdadelta


 Given inputs (param, moment, moment_delta, indices, grad, lr), runs the dense AdaDelta update on (param, grad, moment[indices],  moment_delta[indices], lr), and returns (new_param, new_moment,  new_moment_delta) as in the dense case.
 


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | Default 1e-5
`decay` | Default 0.95, the squared gradient sum is decayed by this factor.
*Inputs* | 
`param` | Parameters to be updated
`moment` | Average of squared gradients
`moment_delta` | Average of squared parameter updates
`indices` | Sparse indices
`grad` | Gradient computed
`lr` | learning rate
*Outputs* | 
`output_param` | Updated parameters
`output_moment` | Updated average squared gradient
`output_moment_delta` | Updated average of squared parameter updates


### Code


[caffe2/sgd/adadelta_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adadelta_op.cc)

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


[caffe2/sgd/adagrad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adagrad_op.cc)

---



## SparseAdam


  

```
    Computes the Adam Update for the sparse case.
    Given inputs (param, moment1, moment2, indices, grad, lr, iter), runs the dense
    Adam on (param, moment1[indices], momemnt2[indices], lr, iter) and returns
    (new_param, new_moment1, new_moment2) as in dense case

```

     


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


[caffe2/sgd/adam_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/adam_op.cc)

---



## SparseFtrl

No documentation yet.


### Code


[caffe2/sgd/ftrl_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/ftrl_op.cc)

---



## SparseLabelSplit


Suppose the maximum of label index is r. This operator has 2r + 1 1-D outputs.
0<= i < r, output[i] contains the label_values of labels with label_index=i (original order is kept).
r<= i < 2r, output[i] contains the corresponding example_ids for output[i-r].
output[2r] (optional) keeps an offset map that is useful for the gradient computation.
Specifically, this map keeps track of the ordering of examples in the expert inputs.



### Interface


---------- | ----------
*Arguments* | 
`num_labels` | Optional; Number of label tasks
*Inputs* | 
`length` | A Nx1 int32 tensor. Sum of its values needs to bethe same as the size of label_index and label_value
`label_index.` | A Mx1 int64 tensor.
`label_value.` | A Mx1 float tensor.


### Code


[caffe2/fb/operators/label_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/label_ops.cc)

---



## SparseLabelSplitGradient

No documentation yet.


### Code


[caffe2/fb/operators/label_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/label_ops.cc)

---



## SparseLabelToBool


Similar to SparseLabelToDense, except now we only cares about wether the label of each example is missing (false) or not (true). The input thus is only len and key, and the output is bool tensors for each label.
  


### Code


[caffe2/fb/operators/label_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/label_ops.cc)

---



## SparseLabelToDense

Converts multi-labels in sparse segment into dense labels.


### Code


[caffe2/fb/operators/label_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/label_ops.cc)

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


[caffe2/sgd/momentum_sgd_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/momentum_sgd_op.cc)

---



## SparseWngrad


 This operator implement the optimization algorithm in  [https://arxiv.org/abs/1803.02865](https://arxiv.org/abs/1803.02865)  by Wu, Ward and Bottou.
Given inputs (param, seq_b, indices, grad, lr), runs the dense WnGrad update on (param, grad, seq_b, lr), and returns (new_param, new_seq_b) as in the dense case.
 


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | Default 1e-5
*Inputs* | 
`param` | Parameters to be updated
`seq_b` | seq_b history
`indices` | Sparse indices
`grad` | Gradient computed
`lr` | learning rate
*Outputs* | 
`output_param` | Updated parameters
`output_seq_b` | Updated seq_b


### Code


[caffe2/sgd/wngrad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/wngrad_op.cc)

---



## SwapYouMightGetFired

Hacky swap contents of two input blobs. Things will blow up     to a billion pieces. You might get fired for using this. XD


### Code


[caffe2/fb/operators/utility_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/fb/operators/utility_ops.cc)

---



## VideoInput

No documentation yet.


### Code


[caffe2/video/video_input_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/video/video_input_op.cc)

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


[caffe2/queue/queue_ops.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/queue/queue_ops.cc)

---



## Wngrad


 Computes the WnGrad update for an input gradient and accumulated history. This operator implement the optimization algorithm in  [https://arxiv.org/abs/1803.02865](https://arxiv.org/abs/1803.02865)  by Wu, Ward and Bottou.
Concretely, given inputs (param, grad, seq_b, learning_rate), computes   

```
    new_seq_b = seq_b + 1 / seq_b * norm(grad)^2
    effective_lr = learning_rate / (new_seq_b + epsilon)
    update = learning_rate * grad / (new_seq_b + epsilon)
    new_param = param + update
```

 and returns (new_param, new_seq_b).
 Optionally returns effective_lr and update as well.
 


### Interface


---------- | ----------
*Arguments* | 
`epsilon` | Default 1e-5
*Inputs* | 
`param` | Parameters to be updated
`seq_b` | Seq_b history
`grad` | Gradient computed
`lr` | learning rate
*Outputs* | 
`output_param` | Updated parameters
`output_seq_b` | Updated seq_b
`output_effective_lr` | (optional) Effective learning rate
`output_update` | (optional) Actual update that is applied.


### Code


[caffe2/sgd/wngrad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/wngrad_op.cc)

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


[caffe2/sgd/yellowfin_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/sgd/yellowfin_op.cc)

---



## rnn_internal_accumulate_gradient_input


Internal RNN operator.



### Code


[caffe2/operators/rnn/recurrent_network_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rnn/recurrent_network_op.cc)

---



## rnn_internal_apply_link


Internal RNN operator.



### Code


[caffe2/operators/rnn/recurrent_network_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/rnn/recurrent_network_op.cc)

---



## ATen

No documentation yet.


### Code


[caffe2/contrib/aten/aten_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/contrib/aten/aten_op.cc)

---



## QuantDecompZstd


 Decompress a set of tensors that are compressed using zstd.
 The data can be compressed using mutils.compress_data_list(), see  quant_decomp_op_test.py for an example.
 The number of outputs depended on the input.
 


### Interface


---------- | ----------
*Inputs* | 
`compressed` | Compressed data in 1d tensor (uint8_t), or 0d tensor with one element in string type.The data is compressed using mutils.compress_data_list().
*Outputs* | 
`output0` | Decompressed data 0
`output1` | Decompressed data 1 if existed
`output2` | Decompressed data 2 if existed
`outputn` | Decompressed data n if existed


### Code


[caffe2/share/contrib/zstd/quant_decomp_zstd_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/share/contrib/zstd/quant_decomp_zstd_op.cc)

---



## FCGradient_Decomp

No documentation yet.


### Code


[caffe2/experiments/operators/fully_connected_op_decomposition.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/fully_connected_op_decomposition.cc)

---



## FCGradient_Prune

No documentation yet.


### Code


[caffe2/experiments/operators/fully_connected_op_prune.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/fully_connected_op_prune.cc)

---



## FC_Decomp

No documentation yet.


### Code


[caffe2/experiments/operators/fully_connected_op_decomposition.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/fully_connected_op_decomposition.cc)

---



## FC_Prune

No documentation yet.


### Code


[caffe2/experiments/operators/fully_connected_op_prune.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/fully_connected_op_prune.cc)

---



## FC_Sparse

No documentation yet.


### Code


[caffe2/experiments/operators/fully_connected_op_sparse.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/fully_connected_op_sparse.cc)

---



## FunHash


This layer compresses a fully-connected layer for sparse inputs via hashing.
It takes four required inputs and an optional fifth input.
The first three inputs  `scalars` ,  `indices` , and  `segment_ids`  are the sparse segmented representation of sparse data, which are the same as the last three inputs of the  `SparseSortedSegmentWeightedSum`  operator. If the argument  `num_segments`  is specified, it would be used as the first dimension for the output; otherwise it would be derived from the maximum segment ID.
 The fourth input is a 1D weight vector. Each entry of the fully-connected layer would be randomly mapped from one of the entries in this vector.
 When the optional fifth input vector is present, each weight of the fully-connected layer would be the linear combination of K entries randomly mapped from the weight vector, provided the input (length-K vector) serves as the coefficients.



### Interface


---------- | ----------
*Arguments* | 
`num_outputs` | Number of outputs
`num_segments` | Number of segments
*Inputs* | 
`scalars` | Values of the non-zero entries of the sparse data.
`indices` | Indices to the non-zero valued features.
`segment_ids` | Segment IDs corresponding to the non-zero entries.
`weight` | Weight vector
`alpha` | Optional coefficients for linear combination of hashed weights.
*Outputs* | 
`output` | Output tensor with the first dimension equal to the number of segments.


### Code


[caffe2/experiments/operators/funhash_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/funhash_op.cc)

---



## FunHashGradient

No documentation yet.


### Code


[caffe2/experiments/operators/funhash_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/funhash_op.cc)

---



## SparseFunHash


This layer compresses a fully-connected layer for sparse inputs via hashing.
It takes four required inputs and an option fifth input.
The first three inputs  `scalars` ,  `indices` , and  `segment_ids`  are the sparse segmented representation of sparse data, which are the same as the last three inputs of the  `SparseSortedSegmentWeightedSum`  operator. If the argument  `num_segments`  is specified, it would be used as the first dimension for the output; otherwise it would be derived from the maximum segment ID.
 The fourth input is a 1D weight vector. Each entry of the fully-connected layer would be randomly mapped from one of the entries in this vector.
 When the optional fifth input vector is present, each weight of the fully-connected layer would be the linear combination of K entries randomly mapped from the weight vector, provided the input (length-K vector) serves as the coefficients.



### Interface


---------- | ----------
*Arguments* | 
`num_outputs` | Number of outputs
`num_segments` | Number of segments
*Inputs* | 
`scalars` | Values of the non-zero entries of the sparse data.
`indices` | Indices to the non-zero valued features.
`segment_ids` | Segment IDs corresponding to the non-zero entries.
`weight` | Weight vector
`alpha` | Optional coefficients for linear combination of hashed weights.
*Outputs* | 
`output` | Output tensor with the first dimension equal to the number of segments.


### Code


[caffe2/experiments/operators/sparse_funhash_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/sparse_funhash_op.cc)

---



## SparseFunHashGradient

No documentation yet.


### Code


[caffe2/experiments/operators/sparse_funhash_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/sparse_funhash_op.cc)

---



## SparseMatrixReshape


Compute the indices of the reshaped sparse matrix.
 It takes two 1D tensors as input: the column indices (in int64) and the row indices (in int), which correspond to  `INDICES`  and  `SEGMENT_IDS`  in  `SparseSortedSegment`  family.
It outputs the corresponding reshaped column and row indices.
 Two arguments are required: an argument  `old_shape`  specifies the original shape of the matrix, and  `new_shape`  specifies the new shape.
One of the dimension in  `old_shape`  and  `new_shape`  can be -1.
The valid combinations are listed below, where p, q, r, s are strictly positive integers.
 old_shape=(p, q) new_shape=(r, s)  old_shape=(p, q) new_shape=(-1, s)  old_shape=(p, q) new_shape=(r, -1)  old_shape=(-1, q) new_shape=(-1, s)  Note that only the first dimension in  `old_shape`  can be -1. In that case the second dimension in  `new_shape`  must NOT be -1.



### Interface


---------- | ----------
*Arguments* | 
`old_shape` | Old shape.
`new_shape` | New shape.
*Inputs* | 
`old_col` | Original column indices.
`old_row` | Original row indices.
*Outputs* | 
`new_col` | New column indices.
`new_row` | New row indices.


### Code


[caffe2/experiments/operators/sparse_matrix_reshape_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/sparse_matrix_reshape_op.cc)

---



## TTContraction


Tensor contraction C = A * B 


### Interface


---------- | ----------
*Arguments* | 
`K` | i_{k-1} * r_k
`M` | r_{k-1} * o_{k-1}
`N` | o_k
*Inputs* | 
`A` | 2D matrix of size (K x M)
`B` | tensor
*Outputs* | 
`C` | contracted tensor


### Code


[caffe2/experiments/operators/tt_contraction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/tt_contraction_op.cc)

---



## TTContractionGradient

No documentation yet.


### Code


[caffe2/experiments/operators/tt_contraction_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/tt_contraction_op.cc)

---



## TTPad

No documentation yet.


### Code


[caffe2/experiments/operators/tt_pad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/tt_pad_op.cc)

---



## TTPadGradient

No documentation yet.


### Code


[caffe2/experiments/operators/tt_pad_op.cc](https://github.com/pytorch/pytorch/blob/master/caffe2/experiments/operators/tt_pad_op.cc)

---



## C10Add_DontUseThisOpYet

No schema documented yet.


## C10AveragedLoss_DontUseThisOpYet

No schema documented yet.


## C10BatchGather_DontUseThisOpYet

No schema documented yet.


## C10BatchMatMul_DontUseThisOpYet

No schema documented yet.


## C10Cast_DontUseThisOpYet

No schema documented yet.


## C10Concat_DontUseThisOpYet

No schema documented yet.


## C10ConstantFill_DontUseThisOpYet

No schema documented yet.


## C10EnforceFinite_DontUseThisOpYet

No schema documented yet.


## C10ExpandDims_DontUseThisOpYet

No schema documented yet.


## C10FC_DontUseThisOpYet

No schema documented yet.


## C10Flatten_DontUseThisOpYet

No schema documented yet.


## C10GivenTensorFill_DontUseThisOpYet

No schema documented yet.


## C10GivenTensorInt64Fill_DontUseThisOpYet

No schema documented yet.


## C10GivenTensorIntFill_DontUseThisOpYet

No schema documented yet.


## C10Mul_DontUseThisOpYet

No schema documented yet.


## C10Relu_DontUseThisOpYet

No schema documented yet.


## C10SigmoidCrossEntropyWithLogits_DontUseThisOpYet

No schema documented yet.


## C10Sigmoid_DontUseThisOpYet

No schema documented yet.


## C10SparseLengthsSum_DontUseThisOpYet

No schema documented yet.


## C10StopGradient_DontUseThisOpYet

No schema documented yet.


## C10UniformFill_DontUseThisOpYet

No schema documented yet.


## SparseLengthsMax

No schema documented yet.
