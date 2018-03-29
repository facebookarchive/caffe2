---
docid: sync-sgd
title: Synchronous SGD
layout: docs
permalink: /docs/SynchronousSGD.html
---

There are multiple ways to utilize multiple GPUs or machines to train models. Synchronous SGD, using Caffe2's data parallel model, is the simplest and easiest to understand: each GPU will execute exactly same code to run their share of the mini-batch. Between mini-batches, we average the gradients of each GPU and each GPU executes the parameter update in exactly the same way. At any point in time the parameters have same values on each GPU. Another way to understand Synchronous SGD is that it allows increasing the mini-batch size. Using 8 GPUS to run a batch of 32 each is equivalent to one GPU running a mini-batch of 256.

## Programming Guide

**Example code:**  

* [data_parallel_model_test.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/data_parallel_model_test.py) has a simple 2-GPU model.
* For a more complex model, see the example [Resnet-50 trainer for ImageNet](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py).

Parallelizing a model is done by module [caffe2.python.data_parallel_model](/doxygen-python/html/namespacedata__parallel__model.html). The model must be created using a ModelHelper, such as [model_helper.ModelHelper](https://github.com/caffe2/caffe2/blob/master/caffe2/python/model_helper.py).

For a full-length tutorial building ResNet-50 for a single GPU, then using `Parallelize_GPU` for multiple GPU check out this [tutorial](https://github.com/caffe2/tutorials/blob/master/Multi-GPU_Training.ipynb)
Here is example from the [Resnet-50 example code](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py):

```python
from caffe2.python import data_parallel_model, model_helper

train_model = model_helper.ModelHelper(name="resnet50")

data_parallel_model.Parallelize_GPU(
      train_model,
      input_builder_fun=add_image_input,
      forward_pass_builder_fun=create_resnet50_model_ops,
      param_update_builder_fun=add_parameter_update_ops,
      devices=gpus,  # list of integers such as [0, 1, 2, 3]
      optimize_gradient_memory=False/True,
  )
```

The key is to split your model creation code to three functions. These functions construct the operators like you would do without parallelization.

 - `input_builder_fun`: creates the operators to provide input to the network. *Note*: be careful that each GPU reads unique data (they should not read the same exact data)! Typically they should share the same Reader to prevent this, or the data should be batched in such way that each Reader is provided unique data. **Signature**: `function(model)`
 - `forward_pass_builder_fun`: this function adds the operators, layers to the network. It should return a list of loss-blobs that are used for computing the loss gradient. This function is also passed an [internally calculated loss_scale](https://github.com/caffe2/caffe2/blob/master/caffe2/python/data_parallel_model.py#L79) parameter that is used to scale your loss to normalize for the number of GPUs. **Signature**: `function(model, loss_scale)`
 - `param_update_builder_fun`: this function adds the operators for applying the gradient update to parameters. For example, a simple SGD update, a momentum parameter update. You should also instantiate the Learning Rate and Iteration blobs here. You can set this function to None if you are not doing learning but only forward pass. **Signature**: `function(model)`
 - `optimize_gradient_memory`: if enabled, [memonger](/doxygen-python/html/namespacememonger.html) module is used to optimize memory usage of gradient operators by sharing blobs when possible. This can save significant amount of memory, and may help you run larger batches.

### Notes

- Do not access the `model_helper.params` directly! Instead use `model_helper.GetParams()`, which only returns the parameters for the current GPU.

### Implementation Notes

Under the hood, Caffe2 uses `DeviceScope` and `NameScope` to distinguish parameters for each GPU. Each parameter is prefixed with a namescope such as "gpu_0/" or "gpu_5/". Each blob created by the functions above is assigned to the correct GPU by `DeviceScope` set by the `data_parallel_model.Parallelize_GPU` function. To checkpoint the model, only pickup parameters prefixed with "gpu_0/" by calling `model.GetParams("gpu_0")`. We use CUDA NCCL-ops to synchronize parameters between machines.

### Performance

Performance will depend on the model, but for Resnet-50, we get ~7x speedup on 8 [M40 GPUs](http://www.nvidia.com/object/tesla-m40.html) over 1 GPU.

### Further Reading & Examples

[Gloo](https://github.com/facebookincubator/gloo) is a Facebook Incubator project that helps manage multi-host, multi-GPU machine learning applications.

[Resnet-50 example code](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py) contains example code using `rendezvous` which is a feature not specifically utilized in this synch SGD example, but is present in the [data_parallel_model module](/doxygen-python/html/namespacedata__parallel__model.html) that it used.

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) is the source research for Resnet-50, wherein they explore the results of building deeper and deeper networks, to over 1,000 layers, using residual learning on the ImageNet dataset. Resnet-50 is their residual network variation using 50 layers that performed quite well with the task of object detection, classification, and localization.
