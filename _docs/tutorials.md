---
docid: tutorials
title: Caffe2 Tutorials
layout: docs
permalink: /docs/tutorials.html
---

  We'd love to start by saying that we really appreciate your interest in Caffe2, and hope this will be a high-performance framework for your machine learning product uses. Caffe2 is intended to be modular and facilitate fast prototyping of ideas and experiments in deep learning. Given this modularity, note that once you have a model defined, and you are interested in gaining additional performance and scalability, you are able to use pure C++ to deploy such models without having to use Python in your final product. Also, as the community develops enhanced and high-performance modules you are able to easily swap these modules into your Caffe2 project.

## Pick Your Path

1. Make my own neural network!
2. Use one off the shelf!

If you chose 1 then keep reading. You'll need some background in neural networking first. Have that dialed in already? Skip ahead to the [Tour of Components](tutorials.html#tour-of-caffe-components) below. Need a primer or a refresher? Some resources are listed below.

If you chose 2, then you will want to jump down to the [IPython notebook tutorials](tutorials.html#ipython-notebook-tutorials), where several examples are using pre-trained models and will show you how to get a demo project up and running in minutes. Want to have image classification in your Android or iOS app? It's pretty much plug-n-play with Android Studio or Xcode.

### New to deep learning

A broad introduction is given in the free online draft of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. In particular the chapters on using neural nets and how backpropagation works are helpful if you are new to the subject.

For an exposition of neural networks in circuits and code, check out [Understanding Neural Networks from a Programmer’s Perspective](http://karpathy.github.io/neuralnets/) by Andrej Karpathy (Stanford).

### Experienced researchers in some facet of machine learning

The [Tutorial on Deep Learning for Vision](https://sites.google.com/site/deeplearningcvpr2014/) from CVPR ‘14 is a good companion tutorial for researchers. Once you have the framework and practice foundations from the Caffe tutorial, explore the fundamental ideas and advanced research directions in the CVPR ‘14 tutorial.

These recent academic tutorials cover deep learning for researchers in machine learning and vision:

* [Deep Learning Tutorial](http://www.cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf) by Yann LeCun (NYU, Facebook) and Marc’Aurelio Ranzato (Facebook). ICML 2013 tutorial.
* [LISA Deep Learning Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf) by the LISA Lab directed by Yoshua Bengio (U. Montréal).

## Tour of Caffe Components  

### C++ implementation

* gpu.h: needs documentation
* db.h: needs documentation


### Python implementation

* TensorProtosDBInput: needs documentation


### Operators

One of basic units of computation in Caffe2 are the [Operators](/docs/operators).

#### Writing Your Own Operators

Fantastic idea! Write custom operators and share them with the community! Refer to the guide on writing operators:

* [Guide for creating your own operators](/docs/custom-operators)

## IPython Notebook Tutorials

These IPython notebook tutorials we have provided below will guide you through the Caffe2 Python interface. Some tutorials have been generously provided by the Caffe community and we welcome more contributions of this kind to help others get ramped up more quickly and to try out the many different uses of Caffe2.

The iPython notebook tutorials can be browsed or downloaded using the links below each tutorial's title.

You may browse these ipynb files on Github directly and this is the preferred route if you just want to look at the code and try it out for yourself.

However, it is recommended to run them in Jupyter Notebook and take advantage of their interactivity. Installation instructions below will show you how to do this. Skip this part if you want to jump right into the tutorial descriptions below.

### Installation

To run the tutorials you'll need Python 2.7, [ipython-notebooks](http://jupyter.org/install.html) and [matplotlib](http://matplotlib.org/users/installing.html), which can be installed on with:

#### MacOSx via Brew & pip

```bash
brew install matplotlib --with-python3
pip install ipython notebook
pip install scikit-image
```

#### Anaconda

Anaconda comes with iPython notebook, so you'll only need to install matplotlib.

```bash
conda install matplotlib
conda install scikit-image
```

#### pip

```bash
pip install matplotlib
pip install ipython notebook
pip install scikit-image
```

Instructions on how to setup Jupyter Notebook, which is the latest, greatest way to use and create interactive code notebooks (ipynb files) is found at [http://jupyter.org](http://jupyter.org/install.html).

Note: if you've already successfully installed Caffe2 with Anaconda Python, then great news! You already have Jupyter Notebook. Starting it is easy:

```
jupyter notebook
```

Or you can run the shell script included in the tutorial folder:

```bash
./start_ipython_notebook.sh
```

When your browser opens with your local Jupyter server (default is http://localhost:8888), browse to the Caffe2 repository and look for them in the [`tutorials`](/tutorials) directory. Opening them this way will launch their interactive features.

### Beginners and New to Caffe2

#### Models and Datasets - a Primer

New to Caffe and Deep Learning? Start here and find out more about the different models and datasets available to you.

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/Models_and_Datasets.ipynb) | [Download](../tutorials/Models_and_Datasets.ipynb)

#### Getting Caffe1 Models for Translation to Caffe2

Here you can find a tutorial with examples of downloading models from Caffe's original repository that you can use with the Caffe2 translator.

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/Getting_Caffe1_Models_for_Translation.ipynb) | [Download](../tutorials/Getting_Caffe1_Models_for_Translation.ipynb)

#### Converting Models from Caffe to Caffe2

A tutorial for converting your old Caffe models or for any of the models found in the Caffe Model Zoo is provided in the following Jupyter notebook found at `docs/tutorials/Caffe_translator.ipynb` or you can browse and download them here:

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/Caffe_translator.ipynb) | [Download](/tutorials/Caffe_translator.ipynb)

#### Basics of Caffe2 - Workspaces, Operators, and Nets

This tutorial introduces a few basic Caffe2 components:

* Workspaces
* Operators
* Nets

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/basics.ipynb) | [Download](/tutorials/basics.ipynb)

#### Toy Regression - Plotting Lines & Random Data

This tutorial shows how to use more Caffe2 features with simple linear regression as the theme.

* generate some sample random data as the input for the model
* create a network with this data
* automatically train the model
* review stochastic gradient descent results and changes to your ground truth parameters as the network learned

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/toy_regression.ipynb) | [Download](/tutorials/toy_regression.ipynb)

#### Image Pre-Processing Pipeline

Learn how to get your images ready for ingestion into pre-trained models or as test images against other datasets. From cell phones to web cams to new medical imagery you will want to consider your image ingestion pipeline and what conversions are necessary for both speed and accuracy during any kind of image classification.

* resizing
* rescaling
* HWC to CHW
* RGB to BRG
* image prep for Caffe2 ingestion

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/Image_Pre-Processing_Pipeline.ipynb) | [Download](../tutorials/Image_Pre-Processing_Pipeline.ipynb)

#### Loading Pre-trained Models

Take advantage of the Model Zoo and grab some pre-trained models and take them for a test drive. This tutorial has a set of different models that are ready to go and will show you the basic steps for prepping them and firing up your neural net. Then you can throw some images or other tests at them and see how they perform.

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/Loading_Pretrained_Models.ipynb) | [Download](../tutorials/Loading_Pretrained_Models.ipynb)

### Creating a Convolutional Neural Network from Scratch

#### MNIST - Handwriting Recognition

This tutorial creates a small convolutional neural network (CNN) that can identify handwriting. The train and test the CNN, we use handwriting imagery from the MNIST dataset. This is a collection of 60,000 images of 500 different people's handwriting that is used for training your CNN. Another set of 10,000 test images (different from the training images) is used to test the accuracy of the resulting CNN.

[Browse](https://github.com/caffe2/caffe2/blob/gh-pages/tutorials/MNIST.ipynb) | [Download](../tutorials/MNIST.ipynb)

### Write Your Own Tutorial!

Have a great tutorial that you've created or have some ideas? Let's chat about it. Create an [Issue](https://github.com/caffe2/caffe2/issues) and post a link to your tutorial or post your idea.
