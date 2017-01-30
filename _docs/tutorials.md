---
docid: tutorials
title: Tutorials
layout: docs
permalink: /docs/tutorials.html
---

We'd love to start by saying that we really appreciate your interest in Caffe2, and hope this will be a high-performance framework for your machine learning product uses. Caffe2 is intended to be modular and facilitate fast prototyping of ideas and experiments in deep learning. Given this modularity, note that once you have a model defined, and you are interested in gaining additional performance and scalability, you are able to use pure C++ to deploy such models without having to use Python in your final product. Also, as the community develops enhanced and high-performance modules you are able to easily swap these modules into your Caffe2 project.

## Pick Your Path

1. Make my own neural network!
2. Use one off the shelf!

If you chose 1 then keep reading. You'll need some background in neural networking first. Have that dialed in already? Skip ahead to the [Tour of Components](http://localhost:8000/tutorials.html#caffe2-tutorials-tour-of-caffe-components) below. Need a primer or a refresher? Some resources are listed below.

If you chose 2, then you will want to jump down to the [IPython notebook tutorials](http://localhost:8000/tutorials.html#caffe2-tutorials-ipython-notebook-tutorials), where several examples are using pre-trained models and will show you how to get a demo project up and running in minutes. Want to have image classification in your Android or iOS app? It's pretty much plug-n-play with Android Studio or Xcode.

### New to deep learning:

A broad introduction is given in the free online draft of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. In particular the chapters on using neural nets and how backpropagation works are helpful if you are new to the subject.

For an exposition of neural networks in circuits and code, check out [Understanding Neural Networks from a Programmer’s Perspective](http://karpathy.github.io/neuralnets/) by Andrej Karpathy (Stanford).

### Experienced researchers in some facet of machine learning:

The [Tutorial on Deep Learning for Vision](https://sites.google.com/site/deeplearningcvpr2014/) from CVPR ‘14 is a good companion tutorial for researchers. Once you have the framework and practice foundations from the Caffe tutorial, explore the fundamental ideas and advanced research directions in the CVPR ‘14 tutorial.

These recent academic tutorials cover deep learning for researchers in machine learning and vision:

* [Deep Learning Tutorial](http://www.cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf) by Yann LeCun (NYU, Facebook) and Marc’Aurelio Ranzato (Facebook). ICML 2013 tutorial.
* [LISA Deep Learning Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf) by the LISA Lab directed by Yoshua Bengio (U. Montréal).

## Tour of Caffe Components  

* [Nets, Layers, and Blobs](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html): the anatomy of a Caffe model
* [Forward / Backward](http://caffe.berkeleyvision.org/tutorial/forward_backward.html): the essential computations of layered compositional models
* [Loss](http://caffe.berkeleyvision.org/tutorial/loss.html): the task to be learned is defined by the loss
* [Solver](http://caffe.berkeleyvision.org/tutorial/solver.html): the solver coordinates model optimization
* [Layer Catalogue](http://caffe.berkeleyvision.org/tutorial/layers.html): **DEPRECATED** in favor of Operators. However, the original Caffe coverage of layers is quite good and worth looking at. Below in the operators section we will talk more about how Caffe2 compares with its earlier version. The layer is the fundamental unit of modeling and computation – Caffe’s catalogue includes layers for state-of-the-art models
* [Interfaces](http://caffe.berkeleyvision.org/tutorial/interfaces.html): command line, Python, and MATLAB Caffe
* [Data](http://caffe.berkeleyvision.org/tutorial/data.html): how to *caffeinate* data for model input

### Operators

**New in Caffe2!**
One of basic units of computation in Caffe2 are the [Operators](operators.html). Each operator contains the logic necessary to compute the output given the appropriate number and types of inputs and parameters. The overall difference between operators' functionality in Caffe and Caffe2 is illustrated in the following graphic, respectively:

![operators comparison](/static/images/operators-comparison.png)

As a result, for example, in the FC operator, each of the input X, bias b, and the weight matrix W must be provided, and a single output will be computed.

#### Writing Your Own Operators
Fantastic idea! Write custom operators and share them with the community! Here is a [guide for creating your own operators](operators_custom.html).

## IPython Notebook Tutorials

These IPython notebook tutorials we have provided below will guide you through the Caffe2 Python interface. Some tutorials have been generously provided by the Caffe community and we welcome more contributions of this kind to help others get ramped up more quickly and to try out the many different uses of Caffe2.

The iPython notebook tutorials can be browsed or downloaded using the links below each tutorial's title.

You may browse these ipynb files on Github directly and this is the preferred route if you just want to look at the code and try it out for yourself.

However, it is recommended to run them in Jupyter Notebook and take advantage of their interactivity. Installation instructions below will show you how to do this. Skip this part if you want to jump right into the tutorial descriptions below.

### Installation

To run the tutorials you'll need Python 2.7, [ipython-notebooks](http://jupyter.org/install.html) and [matplotlib](http://matplotlib.org/users/installing.html), which can be installed on with:

#### MacOSx via Brew & pip

  ```
  brew install matplotlib --with-python3
  pip install ipython notebook
  ```

#### Anaconda

Anaconda comes with iPython notebook, so you'll only need to install matplotlib.

  ```
  conda install matplotlib
  ```

#### pip

  ```
  pip install matplotlib
  pip install ipython notebook
  ```

Instructions on how to setup Jupyter Notebook, which is the latest, greatest way to use and create interactive code notebooks (ipynb files) is found at [http://jupyter.org](http://jupyter.org/install.html).

Note: if you've already successfully installed Caffe2 with Anaconda Python, then great news! You already have Jupyter Notebook. Starting it is easy:

  ```
  jupyter notebook
  ```

When your browser opens with your local Jupyter server (default is http://localhost:8888), browse to the Caffe2 repository and look for them in */docs/tutorials*. Opening them this way will launch their interactive features.

### Basics

[Browse](https://github.com/caffe2/caffe2/blob/documentation/tutorials/basics.ipynb) | [Download](tutorials/basics.ipynb)

This tutorial introduces a few basic Caffe2 components:

* Workspaces
* Operators
* Nets

### Toy Regression - Plotting Lines & Random Data

[Browse](https://github.com/caffe2/caffe2/blob/documentation/tutorials/toy_regression.ipynb) | [Download](tutorials/toy_regression.ipynb)

This tutorial shows how to use more Caffe2 features with simple linear regression as the theme.

* generate some sample random data as the input for the model
* create a network with this data
* automatically train the model
* review stochastic gradient descent results and changes to your ground truth parameters as the network learned

### MNIST - Handwriting Recognition

[Browse](https://github.com/caffe2/caffe2/blob/documentation/tutorials/MNIST.ipynb) | [Download](tutorials/MNIST.ipynb)

This tutorial creates a small convolutional neural network (CNN) that can identify handwriting. The train and test the CNN, we use handwriting imagery from the MNIST dataset. This is a collection of 60,000 images of 500 different people's handwriting that is used for training your CNN. Another set of 10,000 test images (different from the training images) is used to test the accuracy of the resulting CNN.

### Immediate Outputs - Experimental

[Browse](https://github.com/caffe2/caffe2/blob/documentation/tutorials/immediate.ipynb) | [Download](tutorials/immediate.ipynb)

Explores an experimental Caffe2 feature that allows you inspect intermediate outputs as you go. It will run corresponding operators as you write them.

### Converting Models from Caffe to Caffe2

A tutorial for converting your old Caffe models or for any of the models found in the Caffe Model Zoo is provided in the following Jupyter notebook found at `docs/tutorials/Caffe_translator.ipynb` or you can browse and download them here:

[Browse](https://github.com/caffe2/caffe2/blob/documentation/tutorials/Caffe_translator.ipynb) | [Download](tutorials/Caffe_translator.ipynb)
