---
docid: help
title: Help
layout: docs
permalink: /docs/help.html
---

## Python API Errors

### Gradient Operator Checks

Example error message:

> "Gradient operator needs output "gpu_0/tmp2" at version 0, but currently we have version 1"

This is also generally seen as some variation of:

> Gradient name X is expected to correspond to version Y of X, but currently we have version Z.

or

> Gradient output X is expected to correspond to version Y of X, but currently we have version Z.

or

> Gradient input X is expected to correspond to version Y of X, but currently we have version Z.

Check if your model uses in-place blobs and that blob is passed forward to another operator. It cannot keep track of the input needed for the gradient for the first operator. If you attempt a backward pass, it needs to have the original input to create the gradients. This message basically tells you that you should not do an in-place op for "gpu0/tmp2". Once that is addressed, you should be able to generate the backward pass.

[Source code details](http://betadocs.caffe2.ai/doxygen-python/html/core_8py_source.html#l00410)

## Operator Usage

### What's the Difference between net.Sum() and net.Add()?

> `net.Sum()` allows multiple inputs, but does not allow broadcasting.

> `net.Add()` allows only two inputs, but the second one can be broadcasted to match the shape of the first one.
