---
docid: learn-more
title: Learn More
layout: docs
permalink: /docs/learn-more.html
---

Caffe2 is a machine learning framework enabling simple and flexible deep learning. Building on the original [Caffe](caffe.berkeleyvision.org), Caffe2 is designed with expression, speed, and modularity in mind, and allows a more flexible way to organize computation.

## What Is Deep Learning?

Deep learning is one of the latest advances in Artificial Intelligence (AI) and computer science in general. In a nutshell, deep learning extracts and transforms features through the processing of subsequent layers of data, where each successive layer uses the output from the previous layer as input. In many ways, it is the next generation of machine learning and often works hand-in-hand with existing machine learning processing.

To better understand what Caffe2 is and how you can use it, below are a few examples of machine learning and deep learning in practice today.

> Check out other [applications of deep learning](/docs/applications-of-deep-learning) as well.

### Examples of Deep Learning

Want to see some examples of how deep learning works without doing all of the setup? Try out some demos:

### [Caffe Neural Network for Image Classification](http://demo.caffe.berkeleyvision.org/classify_url?imageurl=http%3A%2F%2Fi1.kym-cdn.com%2Fentries%2Ficons%2Foriginal%2F000%2F014%2F959%2FScreenshot_116.png)

![screenshot of CNN demo page](/static/images/CNN-demo.png)

### [Portrait Matcher](http://zeus.robots.ox.ac.uk/facepainting/)

![screenshot of the portrait matcher demo page](/static/images/portrait-matcher-demo.png)

## Caffe2 Philosophy

The philosophy of Caffe2 is the same as [Caffe](http://caffe.berkeleyvision.org/tutorial/#philosophy), and the principles that direct its development can be summed up in six points:

* **Expression**: models and optimizations are defined as plaintext schemas instead of code.
* **Speed**: for research and industry alike speed is crucial for state-of-the-art models and massive data.
* **Modularity**: new tasks and settings require flexibility and extension.
* **Openness**: scientific and applied progress call for common code, reference models, and reproducibility.
* **Community**: academic research, startup prototypes, and industrial applications all share strength by joint discussion and development in a BSD-2 project.

## Why Use Caffe2?

Deep Learning has the potential to bring breakthroughs in machine learning and artificial intelligence. Caffe2 aims to provide an easy and straightforward way for developers to experiment with deep learning first hand.

> In some cases you may want to use existing models and skip the whole "learning" step and get familiar with the utility and effectiveness of deep learning before trying train your own model.

## Working with Caffe2

First, make sure you have [installed Caffe2](/docs/getting-started).

When you are first getting started with deep learning and Caffe2, it will help to understand the workflow of how you will create and deploy your deep learning application. There are two primary stages for working with a deep learning application built with Caffe2:

1. Create your model, which will learn from your inputs and information (classifiers) about the inputs and expected outputs.
2. Run the finished model elsewhere. e.g., on a smart phone, or as sub-component of a platform or a larger app.

Creating the model usually takes some significant processing power and time. While you can get away with just using your laptop's CPU to create and train your deep learning neural network with Caffe2, you may find that for more complicated models with a lot of inputs that this takes too long. Fortunately, you can use the power of GPUs to massively speed up this process. One method of development is to work with a subset of data on your standard PC or small cloud instance, but then run the training of the model with all of the data on a cloud instance with large GPU capacity.

Running the model ends up being relatively lightweight in the sense that even if you took millions of images as inputs, the output that is used when running is much smaller. For example, using 50000 images as inputs might have been several GBs of data, but the output model might only be 200 MB.

## Caffe2 Concepts
Below you can learn more about the main concepts of Caffe2 that are crucial for understanding and developing Caffe2 models.

### Blobs and Workspace, Tensors
Data in Caffe2 is organized as blobs. Blob is just a named chunk of data in memory. Most blobs contain a tensor (think multidimensional array), and in python they are translated to numpy arrays (numpy is a popular numerical library for python and is already installed as a prerequisite with Caffe2).

[Workspace](workspace.html) stores all the blobs. Following example shows how to feed blobs into `workspace` and fetch them. Workspaces initialize themselves the moment you start using them.

```python
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

Composing nets directly is quite tedious, so it is better to use *model helpers* that are python classes that aid in creating the nets. Even though we call it and pass in a single name "my first net", `CNNModelHelper` will create two interrelated nets:

1. one that initializes the parameters (ref. init_net)
2. one that runs the actual training (ref. exec_net)

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

Reviewing the code block above:

First, we created the input data and label blobs in memory (in practice, you would be loading data from a input data source such as database -- more about that later). Note that the data and label blobs have first dimension '16'; this is because the input to the model is a mini-batch of 16 samples at a time. Many Caffe2 operators can be accessed directly through `CNNModelHelper` and can handle a mini-batch of input a time. Check [CNNModelHelper's Operator List](workspace.html#cnnmodelhelper) for more details.

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
  input: "fc1_w"
  input: "fc1_b"
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
external_input: "fc1_w"
external_input: "fc1_b"
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

```python
workspace.RunNetOnce(m.param_init_net)
```

Note, as usual, this will actually pass the protobuffer of the `param_init_net` down to the C++ runtime for execution.

Then we create the actual training Net:

```python
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
