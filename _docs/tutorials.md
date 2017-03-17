---
docid: tutorials
title: Caffe2 Tutorials Overview
layout: docs
permalink: /docs/tutorials.html
---

  We'd love to start by saying that we really appreciate your interest in Caffe2, and hope this will be a high-performance framework for your machine learning product uses. Caffe2 is intended to be modular and facilitate fast prototyping of ideas and experiments in deep learning. Given this modularity, note that once you have a model defined, and you are interested in gaining additional performance and scalability, you are able to use pure C++ to deploy such models without having to use Python in your final product. Also, as the community develops enhanced and high-performance modules you are able to easily swap these modules into your Caffe2 project.

## Pick Your Path

1. [Use a pre-trained neural network off the shelf!](tutorials.html#null__beginners-and-new-to-caffe2) (Easy)
2. [Make my own neural network!](tutorials.html#null__creating-a-convolutional-neural-network-from-scratch) (Intermediate)
3. [Mobile First! I want to make an app that uses deep learning!](AI-Camera-demo-android) (Advanced)

If you chose 1, click the link to where several examples are using pre-trained models and we will show you how to get a demo project up and running in minutes.

If you chose 2 then you'll need some background in neural networking first. Have that dialed in already? Skip ahead to the link. Need a primer or a refresher? Some resources are listed below.

If you chose 3, click the link to discover how to have image classification in your Android or iOS app. It's pretty much plug-n-play with Android Studio or Xcode, but you'll need to integrate directly with Caffe2's C++ hooks.

### New to deep learning

A broad introduction is given in the free online draft of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. In particular the chapters on using neural nets and how backpropagation works are helpful if you are new to the subject.

For an exposition of neural networks in circuits and code, check out [Understanding Neural Networks from a Programmer’s Perspective](http://karpathy.github.io/neuralnets/) by Andrej Karpathy (Stanford).

### Experienced researchers in some facet of machine learning

The [Tutorial on Deep Learning for Vision](https://sites.google.com/site/deeplearningcvpr2014/) from CVPR ‘14 is a good companion tutorial for researchers. Once you have the framework and practice foundations from the Caffe tutorial, explore the fundamental ideas and advanced research directions in the CVPR ‘14 tutorial.

These recent academic tutorials cover deep learning for researchers in machine learning and vision:

* [Deep Learning Tutorial](http://www.cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf) by Yann LeCun (NYU, Facebook) and Marc’Aurelio Ranzato (Facebook). ICML 2013 tutorial.
* [LISA Deep Learning Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf) by the LISA Lab directed by Yoshua Bengio (U. Montréal).

## IPython Notebook Tutorials and Example Scripts

The IPython notebook tutorials and example scripts we have provided below will guide you through the Caffe2 Python interface. Some tutorials have been generously provided by the Caffe community and we welcome more contributions of this kind to help others get ramped up more quickly and to try out the many different uses of Caffe2. The iPython notebook tutorials can be browsed or downloaded using the links below each tutorial's title. You may browse these ipynb files on Github directly and this is the preferred route if you just want to look at the code and try it out for yourself. However, it is recommended to run them in Jupyter Notebook and take advantage of their interactivity. Installation instructions below will show you how to do this. Skip this part if you want to jump right into the tutorial descriptions below.

There are example scripts that can be found in [/caffe2/python/examples](https://github.com/caffe2/caffe2/tree/master/caffe2/python/examples) that are also great resources for starting off on a project using Caffe2.

* [char_rnn.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/char_rnn.py): generate a recurrent convolution neural network that will sample text that you input and randomly generate text of a similar style
* [lmdb_create_example.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/lmdb_create_example.py): create an lmdb database of random image data and labels that can be used a skeleton to write your own data import
* [resnet50_trainer.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py): parallelized multi-GPU distributed trainer for Resnet 50. Can be used to train on imagenet data, for example
* [seq2seq.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/seq2seq.py): create a specialized RNN that handles lines of text for projects such as language translation
* [seq2seq_util.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/seq2seq_util.py): utility functions for the sequence to sequence example script


### Beginners

#### [Models and Datasets - a Primer](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Models_and_Datasets.ipynb)

New to Caffe and Deep Learning? Start here and find out more about the different models and datasets available to you.



#### [Loading Pre-trained Models](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Loading_Pretrained_Models.ipynb)

Take advantage of the Model Zoo and grab some pre-trained models and take them for a test drive. This tutorial has a set of different models that are ready to go and will show you the basic steps for prepping them and firing up your neural net. Then you can throw some images or other tests at them and see how they perform.



### New to Caffe2

You also may want to review the [Intro Tutorial](intro-tutorial) before starting this notebook.

#### [Basics of Caffe2 - Workspaces, Operators, and Nets](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/basics.ipynb)

This tutorial introduces a few basic Caffe2 components:

* Workspaces
* Operators
* Nets



#### [Toy Regression - Plotting Lines & Random Data](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/toy_regression.ipynb)

This tutorial shows how to use more Caffe2 features with simple linear regression as the theme.

* generate some sample random data as the input for the model
* create a network with this data
* automatically train the model
* review stochastic gradient descent results and changes to your ground truth parameters as the network learned



#### [Image Pre-Processing Pipeline](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Image_Pre-Processing_Pipeline.ipynb)

Learn how to get your images ready for ingestion into pre-trained models or as test images against other datasets. From cell phones to web cams to new medical imagery you will want to consider your image ingestion pipeline and what conversions are necessary for both speed and accuracy during any kind of image classification.

* resizing
* rescaling
* HWC to CHW
* RGB to BRG
* image prep for Caffe2 ingestion




### Creating a Convolutional Neural Network from Scratch

#### [MNIST - Handwriting Recognition](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb)

This tutorial creates a small convolutional neural network (CNN) that can identify handwriting. The train and test the CNN, we use handwriting imagery from the MNIST dataset. This is a collection of 60,000 images of 500 different people's handwriting that is used for training your CNN. Another set of 10,000 test images (different from the training images) is used to test the accuracy of the resulting CNN.

#### [Create Your Own Dataset](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/create_your_own_dataset.ipynb)

Try your hand at importing and massaging data so it can be used in Caffe2. This tutorial uses the Iris dataset.


### Write Your Own Tutorial!

Have a great tutorial that you've created or have some ideas? Let's chat about it. Create an [Issue](https://github.com/caffe2/caffe2/issues) and post a link to your tutorial or post your idea.


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


### Tutorials Installation

To run the tutorials you'll need Python 2.7, [ipython-notebooks](http://jupyter.org/install.html) and [matplotlib](http://matplotlib.org/users/installing.html), which can be installed on with:

#### MacOSx via Brew & pip

```bash
brew install matplotlib
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
