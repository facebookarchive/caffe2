---
docid: zoo
title: Caffe2 Model Zoo
layout: docs
permalink: /docs/zoo.html
---
![model zoo](../static/images/landing-puzzle.png)
[Caffe2's Model Zoo](https://github.com/caffe2/caffe2/wiki/Model-Zoo) is maintained by project contributors on the [Github wiki](https://github.com/caffe2/caffe2/wiki/Model-Zoo). Head over there for the full list.

## Model Zoo Overview

If you want to get your hands on pre-trained models, you are in the right place! One of the greatest things about Caffe was the vibrant community of developers and researchers that shared their work in the [original Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). In [Caffe2's Model Zoo](https://github.com/caffe2/caffe2/wiki/Model-Zoo) you will not only find a selection of original Caffe models, but we also provide a set of models here that are ready for use with Caffe2.

You can use these models to quickly build demo applications and explore deep learning capabilities without doing any time-consuming and resource-intensive training. You can recreate and evaluate the results from other's projects, hack together new uses, or improve upon the previously posted models. This is the place for sharing too. If you've created something interesting, please create a Github [Issue](https://github.com/caffe2/caffe2/issues) and share your project, so after a community evaluation, we can share it here in the Model Zoo!

### Compatibility

Caffe2 utilizes a newer format, usually found in the protobuf `.pb` file format, so original `.caffemodel` files will require conversion. Several Caffe models have been ported to Caffe2 for you. A tutorial and sample code is also provided so that you may convert any Caffe model to the new Caffe2 format on your own.

#### Converting Models from Caffe to Caffe2

If you have existing Caffe models or have been using Caffe and want a quick jumpstart, checkout the [Caffe Migration](caffe-migration.html) to start.

### Submitting Your Model to Caffe2's Model Zoo

Please file an [Issue](https://github.com/caffe2/caffe2/issues) to have your project and related models added to the zoo, or if you're already a contributor to the project you can add your entry directly to the [wiki page](https://github.com/caffe2/caffe2/wiki/Model-Zoo).

### Downloading and Importing Caffe2 Models

Loading up a pre-trained model to do things like [predictions such as object detection](tutorial-loading-pre-trained-models.html) is very simple in Caffe2. You need two files: 1) a protobuf that defines the network, and 2) a protobuf that has all of the network weights. The first is generally referred to as the predict_net and the second the init_net. The predict net is small, and the the init_net is usually quite large. Below are two protobuf files that are used to run the Squeezenet model. Click the icon to download them.

|Protobuf file | Download |
|-----|-----|
| predict_net.pb | [![download predict_net.pb](../static/images/download-c2.png)](https://s3.amazonaws.com/caffe2/models/squeezenet/predict_net.pb) |
| init_net.pb | [![download predict_net.pb](../static/images/download-c2.png)](https://s3.amazonaws.com/caffe2/models/squeezenet/init_net.pb) |

Using them from within Python is easy. Just open the files by passing in the path to them where you see `path_to_INIT_NET` and `path_to_PREDICT_NET` respectively. You'll read them into `init_net` and  `predict_net`, then you'll spawn a new Caffe2 workspace automatically when you call `workspace.Predictor`. This call is a wrapper directly to the C++ API `Predictor` and all you need to pass it is the two protobuf files you just opened.

```python
with open(path_to_INIT_NET) as f:
    init_net = f.read()
with open(path_to_PREDICT_NET) as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)
```

From here you can do a simple prediction. Continuing with the Squeezenet example, we can pass in a [prepped image](tutorial-image-pre-processing.html) and get a answer back with the results - which may contain an accurate detection of an object in the image.

```
results = p.run([img])
```

### Model Downloader Module

A great new model downloading and importing feature has been added to Caffe2. It is simple to use, and allows you to setup and run a pre-trained model very quickly. It has an `--install` or `-i` argument that will install the model as a python module. If you don't use the install switch, it simply downloads the model into your current directory.

For example, downloading and installing the squeezenet model:

```bash
python -m caffe2.python.models.download --install squeezenet
```

Then you can use python to import the model directly as a module. If you have trouble, try running this with `sudo` and/or forcing the PYTHONPATH for `sudo` with `sudo PYTHONPATH=/usr/local python -m caffe2.python.models.download --install`

```python
from caffe2.python import workspace
from caffe2.python.models import squeezenet as mynet
init_net = mynet.init_net
predict_net = mynet.predict_net
# you must name it something
predict_net.name = "squeezenet_predict"
workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)
p = workspace.Predictor(init_net.SerializeToString(), predict_net.SerializeToString())
```

This yields `init_net` and `predict_net` fully parsed protobufs that are ready to be loaded into a net, and within a few lines of Python you've instantiated a neural network from a pre-trained model.
