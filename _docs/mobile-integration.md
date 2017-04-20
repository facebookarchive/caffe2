---
docid: mobile-integration
title: Integrating Caffe2 on iOS/Android
layout: docs
permalink: /docs/mobile-integration.html
---

Caffe2 is optimized for mobile integrations, flexibility, easy updates, and running models on lower powered devices. In this guide we will describe what you need to know to implement Caffe2 in your mobile project.

### Demo Camera Project

If you would like to see a working Caffe2 implementation on mobile, currently Android only, check out this demo project.

[AI Camera Demo](AI-Camera-demo-android.html)

### High level summary

*   Distribute (Asset Pipeline, Mobile Config, etc) the models to devices.
*   Instantiate a caffe2::Predictor instance (iOS) or Caffe2 instance (Android) to expose the model to your code.
*   Pass inputs to the model, get outputs back.

### Key objects

*   caffe2::NetDef - (typically binary-serialized) Google Protobuf instance that encapsulates the computation graph and the pretrained weights.
*   caffe2::Predictor - stateful class that is instantiated with an "initialization" NetDef and a "predict" NetDef, and executes the "predict" NetDef with the input and returns the output.

### Library layout

Caffe2 is composed of:

*   A core library, composed of the Workspace, Blob, Net, and Operator classes.
*   An operator library, a range of Operator implementations (such as convolution, etc)

It's pure C++, with the only non-optional dependencies being:

*   Google Protobuf (the lite version, ~300kb)
*   Eigen, a BLAS (on Android) is required for certain primitives, and a vectorized vector/matrix manipulation library, and Eigen is the fastest benchmarked on ARM.

For some use cases you can also bundle NNPACK, which specifically optimizes convolutions on ARM. It's optional (but recommended).

Error handling is by throwing exceptions, typically caffe2::EnforceNotMet, which inherits from std::exception.

#### Intuitive overview

A model consists of two parts - a set of weights (typically floating-point numbers) that represent the learned parameters (updated during training), and a set of 'operations' that form a computation graph that represent how to combine the input data (that varies with each graph pass) with the learned parameters (constant with each graph pass). The parameters (and intermediate states in the computation graph live in a Workspace, which is essentially a `std::unordered_map<string, Blob>`, where a Blob represents an arbitrary typed pointer, typically a TensorCPU, which is an \*n-\*dimensional array (a la Python's numpy ndarray, Torch's Tensor, etc).

The core class is caffe2::Predictor, which exposes the constructor:

    Predictor(const NetDef& init_net, const NetDef& predict_net)

where the two `NetDef` inputs are Google Protocol Buffer objects that represent the two computation graphs described above - the init_net typically runs a set of operations that deserialize weights into the Workspace, and the `predict_net` specifies how to execute the computation graph for each input.

#### Usage considerations

The Predictor is a stateful class - typically the flow would be to instantiate the class once and reuse it for multiple requests. The setup overhead is either trivial or non-trivial, depending on the use case. The constructor does the following:

*   Constructs the workspace object
*   Executes the `init_net`, allocating memory and setting the values of the parameters.
*   Constructs the `predict_net` (mapping a `caffe2::NetDef` to a `caffe2::NetBase` instance (typically `caffe2::SimpleNet`)).

One key point is that all the initialization is in a sense “statically” verifiable - if the constructor fails (by throwing an exception) on one machine, then it will *always* fail on *every* machine. Before exporting the `NetDef` instances, verify that the Net construction can execute correctly.

#### Performance considerations

Currently Caffe2 is optimized for ARM CPUs with NEON (basically any ARM CPU since 2012). Perhaps surprisingly, ARM CPUs outperform the on-board GPUs (our NNPACK ARM CPU implementation outperforms Apple's MPSCNNConvolution for all devices except the iPhone 7). There are other advantages to offloading compute onto the GPU/DSP, and it's an active work in progress to expose these in Caffe2.

For a convolutional implementation, it is recommended to use NNPACK since that's substantially faster (~2x-3x) than the standard `im2col/sgemm` implementation used in most frameworks. Setting `OperatorDef::engine` to NNPACK is recommended here. Example:

```
def pick_engines(net):
    net = copy.deepcopy(net)
    for op in net.op:
        if op.type == "Conv":
            op.engine = "NNPACK"
        if op.type == "ConvTranspose":
            op.engine = "BLOCK"
    return net
```

For non-convolutional (e.g. ranking) workloads, the key computational primitive are often fully-connected layers (e.g. FullyConnectedOp in Caffe2, InnerProductLayer in Caffe, nn.Linear in Torch). For these use cases, you can fall back to a BLAS library, specifically Accelerate on iOS and Eigen on Android.

#### Memory considerations

The model for memory usage of an instantiated and run Predictor is that it's the sum of the size of the weights and the total size of the activations. There is no 'static' memory allocated, all allocations are tied to the Workspace instance owned by the Predictor, so there should be no memory impact after all Predictor instances are deleted.

It's recommended before exporting to run something like:

```
def optimize_net(net):
    optimization = memonger.optimize_interference(
        net,
        [b for b in net.external_input] +
        [b for b in net.external_output])
    try:
        # This can fail if the blobs aren't in the workspace.'
        stats = memonger.compute_statistics(optimization.assignments)
        print("Memory saving: {:.2f}%".format(
            float(stats.optimized_nbytes) / stats.baseline_nbytes * 100))
    except Exception as e:
        print(e)
    return pick_engines(share_conv_buffers(rename_blobs(optimization.net)))
```


This will automatically share activations where valid in the topological ordering of the graph (see [Predictor](https://github.com/facebook/fb-caffe-exts#predictor) for a more detailed discussion).

#### Startup considerations on iOS

Caffe2 uses a registry pattern for registering operator classes. The macros are in the operator registry section of core Operator, [operator.h](https://github.com/caffe2/caffe2/blob/master/caffe2/core/operator.h):

```
// The operator registry. Since we are not expecting a great number of devices,
// we will simply have an if-then type command and allocate the actual
// generation to device-specific registerers.
// Note that although we have CUDA and CUDNN here, the registerers themselves do
// not depend on specific cuda or cudnn libraries. This means that we will be
// able to compile it even when there is no cuda available - we simply do not
// link any cuda or cudnn operators.
CAFFE_DECLARE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_CPU_OPERATOR_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(CPUOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR(name, ...) \
  CAFFE_REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR_STR(str_name, ...) \
  CAFFE_REGISTER_TYPED_CLASS(CPUOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_CPU_OPERATOR_WITH_ENGINE(name, engine, ...) \
  CAFFE_REGISTER_CLASS(CPUOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)
```

  and used by, for example, [conv_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_op.cc):

```
REGISTER_CPU_OPERATOR(Conv, ConvOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>);
```
