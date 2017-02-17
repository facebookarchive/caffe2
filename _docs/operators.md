---
docid: operators
title: Operators Overview
layout: docs
permalink: /docs/operators.html
---

One of basic units of computation in Caffe2 are the `Operators`. Each operator contains the logic necessary to compute the output given the appropriate number and types of inputs and parameters. The overall difference between operators' functionality in Caffe and Caffe2 is illustrated in the following graphic, respectively:

![operators comparison](../static/images/operators-comparison.png)

As a result, for example, in the Fully Connected operator, each of the input X, bias b, and the weight matrix W must be provided, and a single output will be computed.

## Operators Catalogue

For a full listing of Caffe2 Operators, refer to the Operators Catalogue:

* [Operators Catalogue](operators-catalogue.html)

## Writing Your Own Custom Operators

Fantastic idea! Write custom operators and [share](https://github.com/caffe2/caffe2/issues) them with the community! Refer to the guide on writing operators:

* [Guide for creating your own operators](custom-operators.html)

## Sparse Operations

Caffe2 provides support for representing sparse features and performing corresponding operations on segments of tensors. Refer to the guide on sparse operators:

* [Guide for sparse operations](sparse-operations.html)
