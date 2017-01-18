# Caffe2 Tutorials

The tutorials can be found in the repository under [/caffe2/python/tutorial/](/caffe2/python/tutorial/index.md)

You may view these ipynb files on github directly, however they are in the IPython Notebook format so you may want to run them in Jupyter Notebook and take advantage of their interactivity.

[Prerequisites](#prerequisites) |
[Basics](#basics) |
[Toy regression](#toy-regression) |
[Handwriting recognition](#mnist) |
[Immediate processing](#immediate)

## Prerequisites

You will need the following for these tutorials.

* Python 2 environment
* matplotlib
* Jupyter notebook

Instructions on how to setup Jupyter Notebook, which is the latest, greatest way to use and create interactive code notebooks (ipynb files) is found at [http://jupyter.org](http://jupyter.org/install.html).

Note: if you've already successfully installed Caffe2 with Anaconda Python, then great news! You already have Jupyter Notebook. Starting it is easy:

```
$ jupyter notebook
```
When your browser opens with your local Jupyter server (default is http://localhost:8888), browse to the Caffe2 repository that you already downloaded or cloned and find the tutorial files `/caffe2/python/tutorial/`. Opening them this way will launch their interactive features.

### Installation

#### Anaconda

```
$ conda install matplotlib
```

#### pip

```
$ pip install matplotlib
```

## [Basics](../caffe2/python/tutorial/Tutorial_1_Basics.ipynb)

### Description
This tutorial introduces a few basic Caffe2 components:

* Workspaces
* Operators
* Nets

## [Toy Regression](Tutorial_2_toy_regression.ipynb)

### Description
This tutorial shows how to use more Caffe2 features with simple linear regression as the theme.

* generate some sample random data as the input for the model
* create a network with this data
* automatically train the model
* review stochastic gradient descent results and changes to your ground truth parameters as the network learned

## [MNIST](Tutorial_3_MNIST.ipynb)

### Description
This tutorial creates a small convolutional neural network (CNN) that can identify handwriting. The train and test the CNN, we use handwriting imagery from the MNIST dataset. This is a collection of 60,000 images of 500 different people's handwriting that is used for training your CNN. Another set of 10,000 test images (different from the training images) is used to test the accuracy of the resulting CNN.

## [Immediate](Tutorial_4_immediate.ipynb)

### Description
Explores an experimental Caffe2 feature that allows you inspect intermediate outputs as you go. It will run corresponding operators as you write them.
