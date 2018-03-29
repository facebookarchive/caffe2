---
docid: caffe-migration
title: What is Caffe2?
layout: docs
permalink: /docs/caffe-migration.html
---

Caffe2 is a deep learning framework that provides an easy and straightforward way for you to experiment with deep learning and leverage community contributions of new models and algorithms. You can bring your creations to scale using the power of GPUs in the cloud or to the masses on mobile with Caffe2's cross-platform libraries.

Some of the most commonly asked questions about Caffe2 are:

> What does Caffe2 do well? How is it different from Caffe or other deep learning frameworks?

Modularity and being designed for both scale and mobile deployments are the high-level answers to the first question. In many ways Caffe2 is an un-framework because it is so flexible and modular.

## How Does Caffe Compare to Caffe2?

The original Caffe framework was useful for large-scale product use cases, especially with its unparalleled performance and well tested C++ codebase. Caffe has some design choices that are inherited from its original use case: conventional CNN applications. As new computation patterns have emerged, especially distributed computation, mobile, reduced precision computation, and more non-vision use cases, its design has shown some limitations.

Caffe2 improves Caffe 1.0 in a series of directions:

* first-class support for large-scale distributed training
* mobile deployment
* new hardware support (in addition to CPU and CUDA)
* flexibility for future directions such as quantized computation
* stress tested by the vast scale of Facebook applications

## What's New in Caffe2?

One of the basic units of computation in Caffe2 are the `Operators`. You can think of these as a more flexible version of the layers from Caffe. Caffe2 comes with over 400 different operators and provides guidance for the community to create and contribute to this growing resource. For more information, check out [operators information](operators.html) and run through the [intro tutorial](intro-tutorial.html).

## Caffe to Caffe2

### Converting from Caffe

Converting your models from original Caffe is relatively easy. We provide a tutorial below that will convert your caffemodel, but you will still need to verify that the accuracy and loss rates are within range or better.

#### Getting Caffe1 Models for Translation to Caffe2

Here you can find a tutorial with examples of downloading models from Caffe's original repository that you can use with the Caffe2 translator. Skip this if you're starting from scratch and just want to learn Caffe2.

[Browse the IPython Tutorial](https://github.com/caffe2/tutorials/blob/master/Getting_Caffe1_Models_for_Translation.ipynb)

#### Converting Models from Caffe to Caffe2

 We have provided a command line python script tailor made for this purpose. It is found in Caffe2's [python folder](https://github.com/caffe2/caffe2/tree/master/caffe2/python).

* [caffe_translator.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/caffe_translator.py) - This script has built-in translators for common layers. The tutorial mentioned above implements this same script, so it may be helpful to review the tutorial to see how the script can be utilized. You can also call the script directly from command line.
* [caffe_translator_test.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/caffe_translator_test.py) - This a large test that goes through the translation of the BVLC caffenet model, runs an example through the whole model, and verifies numerically that all the results look right. In default, it is disabled unless you explicitly want to run it.

Usage is:

```
python -m caffe2.python.caffe_translator deploy.prototxt pretrained.caffemodel
```

## Torch to Caffe2

### How is Caffe2 different from PyTorch?

Caffe2 is built to excel at mobile and at large scale deployments. While it is new in Caffe2 to support multi-GPU, bringing Torch and Caffe2 together with the same level of GPU support, Caffe2 is built to excel at utilizing both multiple GPUs on a single-host and multiple hosts with GPUs. PyTorch is great for research, experimentation and trying out exotic neural networks, while Caffe2 is headed towards supporting more industrial-strength applications with a heavy focus on mobile. This is not to say that PyTorch doesn't do mobile or doesn't scale or that you can't use Caffe2 with some awesome new paradigm of neural network, we're just highlighting some of the current characteristics and directions for these two projects. We plan to have plenty of interoperability and methods of converting back and forth so you can experience the best of both worlds.

### Converting from Torch

This can currently be accomplished in a two step process where you convert to Caffe first and then to Caffe2 as described above.

[Github torch2caffe](https://github.com/facebook/fb-caffe-exts#torch2caffe)

## Troubleshooting

Some older versions of Caffe produced models that are not convertible by this translator. Generally speaking you can manage this with a custom script that stuffs your layers into [blobs](/doxygen-python/html/namespaceworkspace.html#a34cb41f806c820ea5ce1876ee3aa29f0) and you label them.

[Utilize the appropriate upgrade binary from Caffe's tools](https://github.com/BVLC/caffe/tree/master/tools) to upgrade to the latest version of Caffe, then try our Caffe to Caffe2 translator.
