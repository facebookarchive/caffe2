# Caffe2 Tutorials

  We'd love to start by saying that we really appreciate your interest in Caffe2, and hope this will be a high-performance framework for your machine learning product uses.

  These ipython notebook tutorials will guide you through the Caffe2 Python interface. Note that once we have a model defined, one is able to use pure C++ to deploy such models, without having to use Python in products.

  The iPython notebook tutorials can be [browsed](https://github.com/aaronmarkham/caffe2/tree/master/docs/tutorials) or downloaded using the links below each tutorial's title.

  You may browse these ipynb files on Github directly and this is the preferred route if you just want to look at the code and try it out for yourself.

  However, it is recommended to run them in Jupyter Notebook and take advantage of their interactivity. Installation instructions below will show you how to do this. Skip this part if you want to jump right into the tutorial descriptions below.

## Installation

  To run the tutorials you'll need Python 2.7, [ipython-notebooks](http://jupyter.org/install.html) and [matplotlib](http://matplotlib.org/users/installing.html), which can be installed on with:

### MacOSx via Brew & pip

  ```
  brew install matplotlib --with-python3
  pip install ipython notebook
  ```

### Anaconda

Anaconda comes with iPython notebook, so you'll only need to install matplotlib.

  ```
  conda install matplotlib
  ```

### pip

  ```
  pip install matplotlib
  pip install ipython notebook
  ```

Instructions on how to setup Jupyter Notebook, which is the latest, greatest way to use and create interactive code notebooks (ipynb files) is found at [http://jupyter.org](http://jupyter.org/install.html).

Note: if you've already successfully installed Caffe2 with Anaconda Python, then great news! You already have Jupyter Notebook. Starting it is easy:

```
$ jupyter notebook
```

When your browser opens with your local Jupyter server (default is http://localhost:8888), browse to the Caffe2 repository and look for them in */docs/tutorials*. Opening them this way will launch their interactive features.

## Tutorials

### Basics

[Browse](https://github.com/aaronmarkham/caffe2/blob/master/docs/tutorials/basics.ipynb) | [Download](tutorials/basics.ipynb)

This tutorial introduces a few basic Caffe2 components:

* Workspaces
* Operators
* Nets

### Toy Regression - Plotting Lines & Random Data

[Browse](https://github.com/aaronmarkham/caffe2/blob/master/docs/tutorials/toy_regression.ipynb) | [Download](tutorials/toy_regression.ipynb)

This tutorial shows how to use more Caffe2 features with simple linear regression as the theme.

* generate some sample random data as the input for the model
* create a network with this data
* automatically train the model
* review stochastic gradient descent results and changes to your ground truth parameters as the network learned

### MNIST - Handwriting Recognition

[Browse](https://github.com/aaronmarkham/caffe2/blob/master/docs/tutorials/MNIST.ipynb) | [Download](tutorials/MNIST.ipynb)

This tutorial creates a small convolutional neural network (CNN) that can identify handwriting. The train and test the CNN, we use handwriting imagery from the MNIST dataset. This is a collection of 60,000 images of 500 different people's handwriting that is used for training your CNN. Another set of 10,000 test images (different from the training images) is used to test the accuracy of the resulting CNN.

### Immediate Outputs - Experimental

[Browse](https://github.com/aaronmarkham/caffe2/blob/master/docs/tutorials/immediate.ipynb) | [Download](tutorials/immediate.ipynb)

Explores an experimental Caffe2 feature that allows you inspect intermediate outputs as you go. It will run corresponding operators as you write them.
