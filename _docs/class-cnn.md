---
docid: cnn
title: CNN Class
layout: docs
permalink: /docs/cnn.html
---

## CNNModelHelper
**Code:** [cnn.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/cnn.py) 
| **API Docs:** [cnn.CNNModelHelper](/doxygen-python/html/classcnn_1_1CNNModelHelper.html)

`CNNModelHelper` is a helper class so you can write CNN models more easily, without having to manually define parameter initializations and operators separately. You will find many built-in helper functions as well as automatic support for a collection of operators that are listed below.

### Example Usage

```python
# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = cnn.CNNModelHelper(name="my first net")
fc_1 = m.FC("data", "fc1", dim_in=100, dim_out=10)
pred = m.Sigmoid(fc_1, "pred")
[softmax, loss] = m.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])
```

Arguments |
|-------
order="NCHW",	|	ws_nbytes_limit=None, init_params=True,
name=None,	|	skip_sparse_optim=False,
use_cudnn=True,	|	param_model=None
cudnn_exhaustive_search=False,	|

Functions |
|-------
Accuracy	|	GetWeights
AddWeightDecay	|	ImageInput
AveragePool	|	InstanceNorm
Concat	|	Iter
Conv	|	LRN
ConvTranspose	|	LSTM
DepthConcat	|	MaxPool
Dropout	|	PackedFC
FC	|	PadImage
FC_Decomp	|	PRelu
FC_Prune	|	Relu
FC_Sparse	|	SpatialBN
GetBiases	|	Sum
GroupConv	|	Transpose

Operators |
|-------
Accuracy	|	NCCLAllreduce
Adam	|	NHWC2NCHW
Add	|	PackSegments
Adagrad	|	Print
SparseAdagrad	|	PRelu
AveragedLoss	|	Scale
Cast	|	ScatterWeightedSum
Checkpoint	|	Sigmoid
ConstantFill	|	SortedSegmentSum
Copy	|	Snapshot # Note: snapshot is deprecated use Checkpoint
CopyGPUToCPU	|	Softmax
CopyCPUToGPU	|	SoftmaxWithLoss
DequeueBlobs	|	SquaredL2Distance
EnsureCPUOutput	|	Squeeze
Flatten	|	StopGradient
FlattenToVec	|	Summarize
LabelCrossEntropy	|	Tanh
LearningRate	|	UnpackSegments
MakeTwoClass	|	WeightedSum
MatMul	|
