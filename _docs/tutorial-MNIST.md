---
docid: mnist
title: MNIST - Create a CNN from Scratch
layout: docs
permalink: /docs/tutorial-MNIST.html
---

This tutorial creates a small convolutional neural network (CNN) that can identify handwriting. The train and test the CNN, we use handwriting imagery from the MNIST dataset. This is a collection of 60,000 images of 500 different people's handwriting that is used for training your CNN. Another set of 10,000 test images (different from the training images) is used to test the accuracy of the resulting CNN.

[Browse the IPython Tutorial](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb)


We will use the cnn model helper - that helps us to deal with parameter initializations naturally.

First, let's import the necessities.


```python
%matplotlib inline
from matplotlib import pyplot
import numpy as np
import os
import shutil


from caffe2.python import core, cnn, net_drawer, workspace, visualize

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
# set this where the root of caffe2 is installed
caffe2_root = "~/caffe2"
print("Necessities imported!")
```

We will track statistics during the training time and store these on disk in a local folder. We need to set up a data folder for the data and a root folder for the stats. You should already have these folders, and in the data folder the MNIST dataset should be setup as a leveldb database for both the training set and the test set for this tutorial.

If these folders are missing then you will need to [download the MNIST dataset](Models_and_Datasets.ipynb), g/unzip the dataset and labels, then find the binaries in `/caffe2/build/caffe2/binaries/` or in `/usr/local/binaries/` and run the following, however the code block below will attempt to do this for you, so try that first.

```
./make_mnist_db --channel_first --db leveldb --image_file ~/Downloads/train-images-idx3-ubyte --label_file ~/Downloads/train-labels-idx1-ubyte --output_file ~/caffe2/caffe2/python/tutorials/tutorial_data/mnist/mnist-train-nchw-leveldb

./make_mnist_db --channel_first --db leveldb --image_file ~/Downloads/t10k-images-idx3-ubyte --label_file ~/Downloads/t10k-labels-idx1-ubyte --output_file ~/caffe2/caffe2/python/tutorials/tutorial_data/mnist/mnist-test-nchw-leveldb
```


```python
# This section preps your image and test set in a leveldb
# if you didn't download the dataset yet go back to Models and Datasets and get it there
current_folder = os.getcwd()

data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
image_file_train = os.path.join(data_folder, "train-images-idx3-ubyte")
label_file_train = os.path.join(data_folder, "train-labels-idx1-ubyte")
image_file_test = os.path.join(data_folder, "t10k-images-idx3-ubyte")
label_file_test = os.path.join(data_folder, "t10k-labels-idx1-ubyte")

# Get the dataset if it is missing
def DownloadDataset(url, path):
    import requests, zipfile, StringIO
    print "Downloading... ", url, " to ", path
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(label_file_train):
    DownloadDataset("https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip", data_folder)

def GenerateDB(image, label, name):
    name = os.path.join(data_folder, name)
    print 'DB name: ', name
    syscall = "/usr/local/binaries/make_mnist_db --channel_first --db leveldb --image_file " + image + " --label_file " + label + " --output_file " + name
    print "Creating database with: ", syscall
    os.system(syscall)

# (Re)generate the leveldb database (known to get corrupted...)
GenerateDB(image_file_train, label_file_train, "mnist-train-nchw-leveldb")
GenerateDB(image_file_test, label_file_test, "mnist-test-nchw-leveldb")


if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old workspace files...")
    shutil.rmtree(root_folder)

os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)

print("training data folder:"+data_folder)
print("workspace root folder:"+root_folder)
```

We will be using the `CNNModelHelper`, which has a set of wrapper functions that automatically separates the parameter intialization and the actual computation into two networks. Under the hood, a `CNNModelHelper` object has two underlying nets, `param_init_net` and `net`, that keeps record of the initialization network and the main network respectively.

For the sake of modularity, we will separate the model to multiple different parts:

    (1) The data input part (AddInput function)
    (2) The main computation part (AddLeNetModel function)
    (3) The training part - adding gradient operators, update, etc. (AddTrainingOperators function)
    (4) The bookkeeping part, where we just print out statistics for inspection. (AddBookkeepingOperators function)

`AddInput` will load the data from a DB. We store MNIST data in pixel values, so after batching this will give us:

    - data with shape `(batch_size, num_channels, width, height)`
        - in this case `[batch_size, 1, 28, 28]` of data type *uint8*
    - label with shape `[batch_size]` of data type *int*

Since we are going to do float computations, we will cast the data to the *float* data type.
For better numerical stability, instead of representing data in [0, 255] range, we will scale them down to [0, 1].
Note that we are doing in-place computation for this operator: we don't need the pre-scale data.
Now, when computing the backward pass, we will not need the gradient computation for the backward pass. `StopGradient` does exactly that: in the forward pass it does nothing and in the backward pass all it does is to tell the gradient generator "the gradient does not need to pass through me".



```python
def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label
print("Input function created.")
```

    Input function created.


At this point we need to take a look at the predictions coming out of the network at convert them into probabilities.
"What's the probability that this number we're looking at is a 5", or "is this a 7", and so forth.

The results will be conformed into a range between 0 and 1 such that the closer you are to 1 the more likely the number matches the prediction. The process that we can use to do this is available in LeNet and will provide us the *softmax* prediction. The `AddLeNetModel` function below will output the `softmax`. However, in this case, it does much more than the softmax - it is the computed model with its convoluted layers, as well as the softmax.

TODO: include image of the model below


```python
def AddLeNetModel(model, data):
    conv1 = model.Conv(data, 'conv1', 1, 20, 5)
    pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
    conv2 = model.Conv(pool1, 'conv2', 20, 50, 5)
    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
    fc3 = model.FC(pool2, 'fc3', 50 * 4 * 4, 500)
    fc3 = model.Relu(fc3, fc3)
    pred = model.FC(fc3, 'pred', 500, 10)
    softmax = model.Softmax(pred, 'softmax')
    return softmax
print("Model function created.")
```

    Model function created.


The `AddAccuracy` function below adds an accuracy operator to the model. We will use this in the next function to keep track of the model's accuracy.


```python
def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy
print("Accuracy function created.")
```

    Accuracy function created.


The next function, `AddTrainingOperators`, adds training operators to the model.

In the first step, we apply an Operator, `LabelCrossEntropy`, that computes the cross entropy between the input and the label set. This operator is almost always used after getting a softmax and before computing the model's loss. It's going to take in the `[softmax, label]` array along with a label, `'xent'` for "Cross Entropy".

    xent = model.LabelCrossEntropy([softmax, label], 'xent')

`AveragedLoss` will take in the cross entropy and return the average of the losses found in the cross entropy.

    loss = model.AveragedLoss(xent, "loss")

For bookkeeping purposes, we will also compute the accuracy of the model by invoking the AddAccuracy function like so:

    AddAccuracy(model, softmax, label)

The next line is the key part of the training model: we add all the gradient operators to the model. The gradient is computed with respect to the loss that we computed above.

    model.AddGradientOperators([loss])


The next handful of lines support a very simple stochastic gradient descent.
--- TODO(jiayq): We are working on wrapping these SGD operations in a cleaner fashion, and we will update this when it is ready. For now, you can see how we basically express the SGD algorithms with basic operators.
It isn't necessary to fully understand this part at the moment, but we'll walk you through the process anyway.

We start with `model.Iter`, a counter for the number of iterations we run in the training.

    ITER = model.Iter("iter")

We do a simple learning rate schedule where lr = base_lr * (t ^ gamma)
Note that we are doing minimization, so the base_lr is negative so we are going the DOWNHILL direction.

    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
ONE is a constant value that is used in the gradient update. We only need to create it once, so it is explicitly placed in param_init_net.

    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

Now, for each parameter, we do the gradient updates. Note how we get the gradient of each parameter - CNNModelHelper keeps track of that. The update is a simple weighted sum: param = param + param_grad * LR

    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)        

We will need to checkpoint the parameters of the model periodically. This is achieved via the Checkpoint operator. It also takes in a parameter "every" so that we don't checkpoint way too often. In this case, we will say let's checkpoint every 20 iterations, which should probably be fine.

    model.Checkpoint([ITER] + model.params, [],
                   db="mnist_lenet_checkpoint_%05d.leveldb",
                   db_type="leveldb", every=20)


```python
def AddTrainingOperators(model, softmax, label):
    # something very important happens here
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    ITER = model.Iter("iter")
    # set the learning rate schedule
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - CNNModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    # let's checkpoint every 20 iterations, which should probably be fine.
    # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    model.Checkpoint([ITER] + model.params, [],
                   db="mnist_lenet_checkpoint_%05d.leveldb",
                   db_type="leveldb", every=20)
print("Training function created.")
```

    Training function created.


The following function, `AddBookkeepingOperations`, adds a few bookkeeping operators that we can inspect later. These operators do not affect the training procedure: they only collect statistics and prints them to file or to logs.


```python
def AddBookkeepingOperators(model):
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.
print("Bookkeeping function created")
```

Now, let's actually create the models for training and testing. If you are seeing WARNING messages below, don't be alarmed. The functions we established earlier are now going to be executed. Remember the four steps that we're doing:

    (1) data input  
    (2) main computation
    (3) training
    (4) bookkeeping

Before we can do the data input though we need to define our training model. We will basically need every piece of the components we defined above. In this example, we're using NCHW storage order on the mnist_train dataset.


```python
train_model = cnn.CNNModelHelper(order="NCHW", name="mnist_train")
data, label = AddInput(
    train_model, batch_size=64,
    db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
    db_type='leveldb')
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)
AddBookkeepingOperators(train_model)

# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel
# part, and an accuracy part. Note that init_params is set False because
# we will be using the parameters obtained from the train model.
test_model = cnn.CNNModelHelper(
    order="NCHW", name="mnist_test", init_params=False)
data, label = AddInput(
    test_model, batch_size=100,
    db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'),
    db_type='leveldb')
softmax = AddLeNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

# Deployment model. We simply need the main LeNetModel part.
deploy_model = cnn.CNNModelHelper(
    order="NCHW", name="mnist_deploy", init_params=False)
AddLeNetModel(deploy_model, "data")
# You may wonder what happens with the param_init_net part of the deploy_model.
# No, we will not use them, since during deployment time we will not randomly
# initialize the parameters, but load the parameters from the db.

print('Created training and deploy models.')
```


Now, let's take a look what the training and deploy models look like, using the simple graph visualization tool that Caffe2 has. If the following command fails for you, it might be because that the machine you run on does not have graphviz installed. You can usually install that by:

```sudo yum install graphviz```

If the graph looks too small, right click and open the image in a new tab for better inspection.


```python
from IPython import display
graph = net_drawer.GetPydotGraph(train_model.net.Proto().op, "mnist", rankdir="LR")
display.Image(graph.create_png(), width=800)
```


![graph](../static/images/tutorial-mnist1.png)



Now, the graph above shows everything that is happening in the training phase: the white nodes are the blobs, and the green rectangular nodes are the operators being run. You may have noticed the massive parallel lines like train tracks: these are dependencies from the blobs generated in the forward pass to their backward operators.

Let's display the graph in a more minimal way by showing only the necessary dependencies and only showing the operators. If you read carefully, you can see that the left half of the graph is the forward pass, the right half of the graph is the backward pass, and on the very right there are a set of parameter update and summarization operators.


```python
graph = net_drawer.GetPydotGraphMinimal(
    train_model.net.Proto().op, "mnist", rankdir="LR", minimal_dependency=True)
display.Image(graph.create_png(), width=800)
```




![graph](../static/images/tutorial-mnist2.png)



Now, when we run the network, one way is to directly run it from Python. Remember as we are running the network, we can periodically pull blobs from the network - Let's first show how we do this.

Before, that, let's re-iterate the fact that, the CNNModelHelper class has not executed anything yet. All it does is to *declare* the network, which is basically creating the protocol buffers. For example, we will show a portion of the serialized protocol buffer for the training models' param init net.


```python
print(str(train_model.param_init_net.Proto())[:400] + '\n...')
```

We will also dump all the protocol buffers to disk so you can easily inspect them. As you may have noticed, these protocol buffers are much like the old good caffe's network definitions.


```python
with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol buffers files have been created in your root folder: "+root_folder)
```

Next we will run the training procedure. We will drive all the computation in Python here, however you can also write a plan out to disk so that you can completely train stuff in C++.  We'll leave discussion on that route for another tutorial.

Please note that this process will take a while to run. Keep an eye on the asterisk (In [\*]) or other IPython indicators that the code block is still running.

First we must initialize the network with:

    workspace.RunNetOnce(train_model.param_init_net)

Since we are going to run the main network multiple times, we first create the network which puts the actual network generated from the protobuf into the workspace.

    workspace.CreateNet(train_model.net)

We will set the number of iterations that we'll run the network to 200 and create two numpy arrays to record the accuracy and loss for each iteration.

    total_iters = 200
    accuracy = np.zeros(total_iters)
    loss = np.zeros(total_iters)

With the network and tracking of accuracy and loss setup we can now loop the 200 interations calling `workspace.RunNet` and passing the name of the network `train_model.net.Proto().name`. On each iteration we calculate the accuracy and loss with `workspace.FetchBlob('accuracy')` and `workspace.FetchBlob('loss')`.

    for i in range(total_iters):
        workspace.RunNet(train_model.net.Proto().name)
        accuracy[i] = workspace.FetchBlob('accuracy')
        loss[i] = workspace.FetchBlob('loss')

Finally, we can plot the results using `pyplot`.


```python
# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net)
# set the number of iterations and track the accuracy & loss
total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Now, we will manually run the network for 200 iterations.
for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
# After the execution is done, let's plot the values.
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
```

![graph](../static/images/tutorial-mnist3.png)


Now we can sample some of the data and predictions.


```python
# Let's look at some of the data.
pyplot.figure()
data = workspace.FetchBlob('data')
_ = visualize.NCHW.ShowMultiple(data)
pyplot.figure()
softmax = workspace.FetchBlob('softmax')
_ = pyplot.plot(softmax[0], 'ro')
pyplot.title('Prediction for the first image')
```

![numbers](../static/images/tutorial-mnist4.png)



![chart](../static/images/tutorial-mnist5.png)


Remember that we created the test net? We will run the test pass and report the test accuracy here. Note that although test_model will be using the parameters obtained from train_model, test_model.param_init_net must still be run to initialize the input data.
In this run, we only need to track the accuracy and we're also only going to run 100 iterations.


```python
# run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)
test_accuracy = np.zeros(100)
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
# After the execution is done, let's plot the values.
pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
print('test_accuracy: %f' % test_accuracy.mean())
```

    test_accuracy: 0.946700



![chart](../static/images/tutorial-mnist6.png)


This concludes the MNIST tutorial. We hope this tutorial highlighted some of Caffe2's features and how easy it is to create a simple CNN.
