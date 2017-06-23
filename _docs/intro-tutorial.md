---
docid: intro-tutorial
title: Intro Tutorial
layout: docs
permalink: /docs/intro-tutorial.html
---

## Caffe2 Concepts
Below you can learn more about the main concepts of Caffe2 that are crucial for understanding and developing Caffe2 models.

### Blobs and Workspace, Tensors
Data in Caffe2 is organized as blobs. Blob is just a named chunk of data in memory. Most blobs contain a tensor (think multidimensional array), and in python they are translated to numpy arrays (numpy is a popular numerical library for python and is already installed as a prerequisite with Caffe2).

[Workspace](workspace.html) stores all the blobs. Following example shows how to feed blobs into `workspace` and fetch them. Workspaces initialize themselves the moment you start using them.

```py
from caffe2.python import workspace, model_helper
import numpy as np
# Create random tensor of three dimensions
x = np.random.rand(4, 3, 2)
print(x)
print(x.shape)

workspace.FeedBlob("my_x", x)

x2 = workspace.FetchBlob("my_x")
print(x2)
```

### Nets and Operators
The fundamental object of Caffe2 is a net (short for network). Net is a graph of operators, and each operator takes a set of input blobs and produces one or more output blobs.

In the code block below we will create a super simple model. It will have these components:

* One fully-connected layer (FC)
  * a Sigmoid activation with a Softmax
  * a CrossEntropy loss

Composing nets directly is quite tedious, so it is better to use *model helpers* that are python classes that aid in creating the nets. Even though we call it and pass in a single name "my first net", `ModelHelper` will create two interrelated nets:

1. one that initializes the parameters (ref. init_net)
2. one that runs the actual training (ref. exec_net)

```py
# Create the input data
data = np.random.rand(16, 100).astype(np.float32)

# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

```

We created some random data and random labels and then fed those as blobs into the workspace.

```py
# Create model using a model helper
m = model_helper.ModelHelper(name="my first net")
```

You've now used the `model_helper` to created the two nets we mentioned earlier (init_net, exec_net). We plan to add a fully-connected layer using the FC operator in this model next, but first we need to do some prep work by creating some random fills that the FC Op expects. Next we can add the Ops and use the weights and bias blobs we created, calling them by name when we invoke the FC Op.

```py
weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])
```

In Caffe2 the FC Op takes in an input blob (our data), weights, and bias. Weights and bias using either `XavierFill` or `ConstantFill` will both take an empty array, name, and shape (as `shape=[output, input]`).

```py
fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
pred = m.net.Sigmoid(fc_1, "pred")
[softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])
```

Reviewing the code blocks above:

First, we created the input data and label blobs in memory (in practice, you would be loading data from a input data source such as database -- more about that later). Note that the data and label blobs have first dimension '16'; this is because the input to the model is a mini-batch of 16 samples at a time. Many Caffe2 operators can be accessed directly through `ModelHelper` and can handle a mini-batch of input a time. Check [ModelHelper's Operator List](workspace.html#cnnmodelhelper) for more details.

Second, we create a model by defining a bunch of operators: [FC](operators-catalogue.html#fc), [Sigmoid](operators-catalogue.html#sigmoidgradient) and [SoftmaxWithLoss](operators-catalogue.html#softmaxwithloss). *Note:* at this point, the operators are not executed, you are just creating the definition of the model.

Model helper will create two nets: `m.param_init_net` which is a net you run only once. It will initialize all the parameter blobs such as weights for the FC layer. The actual training is done by executing `m.net`. This is transparent to you and happens automatically.

The net definition is stored in a protobuf structure (see Google's Protobuffer documentation to learn more; protobuffers are equivalent to Thrift structs). You can easily inspect it by calling `net.Proto()`:

```python
print(str(m.net.Proto()))
```

The output should look like:

```json
name: "my first net"
op {
  input: "data"
  input: "fc_w"
  input: "fc_b"
  output: "fc1"
  name: ""
  type: "FC"
}
op {
  input: "fc1"
  output: "pred"
  name: ""
  type: "Sigmoid"
}
op {
  input: "pred"
  input: "label"
  output: "softmax"
  output: "loss"
  name: ""
  type: "SoftmaxWithLoss"
}
external_input: "data"
external_input: "fc_w"
external_input: "fc_b"
external_input: "label"
```

You also should have a look at the param initialization net:

```python
print(str(m.param_init_net.Proto()))
```

You can see how there are two operators that create random fill for the weight and bias blobs of the FC operator.

This is the primary idea of Caffe2 API: use Python to conveniently compose nets to train your model, pass those nets to C++ code as serialized protobuffers, and then let the C++ code run the nets with full performance.

### Executing
Now when we have the model training operators defined, we can start to run it to train our model.

First, we run only once the param initialization:

```py
workspace.RunNetOnce(m.param_init_net)
```

Note, as usual, this will actually pass the protobuffer of the `param_init_net` down to the C++ runtime for execution.

Then we create the actual training Net:

```py
workspace.CreateNet(m.net)
```

We create it once and then we can efficiently run it multiple times:

```python
# Run 100 x 10 iterations
for j in range(0, 100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)

    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)

    workspace.RunNet(m.name, 10)   # run for 10 times
```

Note how we refer to the network name in `RunNet()`. Since the net was created inside workspace, we don't need to pass the net definition again.

After execution, you can inspect the results stored in the output blobs (that contain tensors i.e numpy arrays):

```python
print(workspace.FetchBlob("softmax"))
print(workspace.FetchBlob("loss"))
```

### Backward pass
This net only contains the forward pass, thus is not learning anything. The backward pass is created by creating the gradient operators for each operator in the forward pass.

If you care to follow this example yourself, then try the following steps an examine the results!

Insert following before you call `RunNetOnce()`:

```python
m.AddGradientOperators([loss])
```

Examine the protobuf output:

```python
print(str(m.net.Proto()))
```

This concludes the overview, but there's a lot more to be found in the [tutorials](tutorials.html).
