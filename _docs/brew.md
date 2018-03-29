---
docid: brew
title: Brewing Models
layout: docs
permalink: /docs/brew.html
---

> `brew` is Caffe2's new API for building models. The `CNNModelHelper` filled this role in the past, but since Caffe2 has expanded well beyond excelling at CNNs it made sense to provide a `ModelHelper` object that is more generic. You may notice that the new `ModelHelper` has much the same functionality as `CNNModelHelper`. `brew` wraps the new `ModelHelper` making building models even easier than before.

## Model Building and Brew's Helper Functions

In this overview we will introduce  [brew](https://github.com/caffe2/caffe2/blob/master/caffe2/python/brew.py), a lightweight collection of helper functions to help you build your model. We will start with explaining the key concepts of Ops versus Helper Functions. Then we will show `brew` usage, how it acts as an interface to the `ModelHelper` object, the and `arg_scope` syntax sugar. Finally we discuss the motivation of introducing `brew`.

## Concepts: Ops vs Helper Functions

Before we dig into `brew` we should review some conventions in Caffe2 and how layers of a neural network are represented. Deep learning networks in Caffe2 are built up with operators. Typically these operators are written in C++ for maximum performance. Caffe2 also provides a Python API that wraps these C++ operators, so you can more flexibly experiment and prototype. In Caffe2, operators are always presented in a CamelCase fashion, whereas Python helper functions with a similar name are in lowercase. Examples of this are to follow.

### Ops

We often refer to operators as an "Op" or a collection of operators as "Ops". For example, the `FC` Op represents a Fully-Connected operator that has weighted connections to every neuron in the previous layer and to every neuron on the next layer. For example, you can create an `FC` Op with:

```py
model.net.FC([blob_in, weights, bias], blob_out)
```

Or you can create a `Copy` Op with:

```py
model.net.Copy(blob_in, blob_out)
```

> A list of Operators handled by `ModelHelper` is at the bottom of this document. The 29 Ops that are most commonly used are currently included. This is a subset of the 400+ Ops Caffe2 has at the time of this writing.

It should also be noted that you can also create an operator without annotating `net`. For example, just like in the previous example where we created a `Copy` Op, we can use the following code to create a `Copy` operator on `model.net`:

```py
model.Copy(blob_in, blob_out)
```

### Helper Functions

Building your model/network using merely single operators could be painstaking since you will have to do parameter initialization, device/engine choice all by yourself (but this is also why Caffe2 is so fast!). For example, to build an FC layer you have several lines of code to prepare `weight` and `bias`, which are then fed to the Op.

**This is the longer, manual way:**

```py
model = model_helper.ModelHelper(name="train")
# initialize your weight
weight = model.param_init_net.XavierFill(
    [],
    blob_out + '_w',
    shape=[dim_out, dim_in],
    **kwargs, # maybe indicating weight should be on GPU here
)
# initialize your bias
bias = model.param_init_net.ConstantFill(
    [],
    blob_out + '_b',
    shape=[dim_out, ],
    **kwargs,
)
# finally building FC
model.net.FC([blob_in, weights, bias], blob_out, **kwargs)
```

Luckily Caffe2 helper functions are here to help. Helper functions are wrapper functions that create a complete layer for a model. The helper function will typically handle parameter initialization, operator definition, and engine selection. Caffe2 default helper functions are named in Python PEP8 function convention. For example, using [python/helpers/fc.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/helpers/fc.py), implementing an `FC` Op via the helper function `fc` is much simpler:

**An easier way using a helper function:**

```py
fcLayer = fc(model, blob_in, blob_out, **kwargs) # returns a blob reference
```

> Some helper functions build much more than 1 operator. For example, the LSTM function in [python/rnn_cell.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/rnn_cell.py) is helping you building a whole LSTM unit in your network.

Check out the [repo](https://github.com/caffe2/caffe2/tree/master/caffe2/python/helpers) for more cool helper functions!

## brew

Now that you've been introduced to Ops and Helper Functions, let's cover how `brew` can make model building even easier. `brew` is a smart collection of helper functions. You can use all Caffe2 awesome helper functions with a single import of brew module. You can now add a FC layer using:

```py
from caffe2.python import brew

brew.fc(model, blob_in, blob_out, ...)
```

That's pretty much the same as using the helper function directly, however `brew` really starts to shine once your models get more complicated. The following is a LeNet model building example, extracted from the [MNIST tutorial](https://github.com/caffe2/tutorials/blob/master/MNIST.ipynb).

```py
from caffe2.python import brew

def AddLeNetModel(model, data):
    conv1 = brew.conv(model, data, 'conv1', 1, 20, 5)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    conv2 = brew.conv(model, pool1, 'conv2', 20, 50, 5)
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    fc3 = brew.fc(model, pool2, 'fc3', 50 * 4 * 4, 500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
```

Each layer is created using `brew`, which in turn is using its operator hooks to instantiate each Op.

### arg_scope

`arg_scope` is a syntax sugar for you to set default helper function argument values within its context. For example, say you want to experiment with different weight initializations in your ResNet-150 training script. You can either:

```py
# change all weight_init here
brew.conv(model, ..., weight_init=('XavierFill', {}),...)
...
# repeat 150 times
...
brew.conv(model, ..., weight_init=('XavierFill', {}),...)
```

Or with the help of `arg_scope`, you can

```py
with brew.arg_scope([brew.conv], weight_init=('XavierFill', {})):
     brew.conv(model, ...) # no weight_init needed here!
     brew.conv(model, ...)
     ...
```

### Custom Helper Function

As you use `brew` more often and find a need for implementing an Op not currently covered by `brew` you will want to write your own helper function. You can register your helper function to brew to enjoy unified management and syntax sugar.

Simply define your new helper function, register it with `brew` using the `.Register` function, then call it with `brew.new_helper_function`.

```py
def my_super_layer(model, blob_in, blob_out, **kwargs):
"""
   100x faster, awesome code that you'll share one day.
"""

brew.Register(my_super_layer)
brew.my_super_layer(model, blob_in, blob_out)
```

If you think your helper function might be helpful to the rest of the Caffe2 community, remember to share it, and create a pull request.

### Caffe2 Default Helper Functions

To get more details about each of these functions, visit the [Operators Catalogue](operators-catalogue).

* [accuracy](operators-catalogue.html#accuracy)
* [add_weight_decay](operators-catalogue.html#add_weight_decay)
* [average_pool](operators-catalogue.html#average_pool)
* [concat](operators-catalogue.html#concat)
* [conv](operators-catalogue.html#conv)
* [conv_nd](operators-catalogue.html#conv_nd)
* [conv_transpose](operators-catalogue.html#conv_transpose)
* [depth_concat](operators-catalogue.html#depth_concat)
* [dropout](operators-catalogue.html#dropout)
* [fc](operators-catalogue.html#fc)
* [fc_decomp](operators-catalogue.html#fc_decomp)
* [fc_prune](operators-catalogue.html#fc_prune)
* [fc_sparse](operators-catalogue.html#fc_sparse)
* [group_conv](operators-catalogue.html#group_conv)
* [group_conv_deprecated](operators-catalogue.html#group_conv_deprecated)
* [image_input](operators-catalogue.html#image_input)
* [instance_norm](operators-catalogue.html#instance_norm)
* [iter](operators-catalogue.html#iter)
* [lrn](operators-catalogue.html#lrn)
* [max_pool](operators-catalogue.html#max_pool)
* [max_pool_with_index](operators-catalogue.html#max_pool_with_index)
* [packed_fc](operators-catalogue.html#packed_fc)
* [prelu](operators-catalogue.html#prelu)
* [softmax](operators-catalogue.html#softmax)
* [spatial_bn](operators-catalogue.html#spatial_bn)
* [relu](operators-catalogue.html#relu)
* [sum](operators-catalogue.html#sum)
* [transpose](operators-catalogue.html#transpose)
* [video_input](operators-catalogue.html#video_input)

## Motivation for brew

Thanks for reading a whole overview on `brew`! Congratulations, you are finally here! Long story short, we want to separate model building process and model storage. In our view, `ModelHelper` class should only contain network definition and parameter information. The `brew` module will have the functions to build network and initialize parameters.

Compared with previous gigantic `CNNModelHelper` that is doing both model storage and model building, the `ModelHelper` + `brew` way of model building is much more modularized and easier to extend. In terms of naming, it is also much less confusing as the Caffe2 family supports a variety of networks, including MLP, RNN and CNN. We hope this tutorial will help your model building to be faster and easier while also getting to know Caffe2 in more depth. There is a detailed example of brew usage in [python/brew_test.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/brew_test.py). If you have any question about brew, please feel free to contact us and ask a question in an Issue on the repo. Thank you again for embracing the new `brew` API.
