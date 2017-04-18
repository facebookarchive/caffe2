---
docid: zoo
title: Caffe2 Model Zoo
layout: docs
permalink: /docs/zoo.html
---

## Model Zoo Overview

If you want to get your hands on pre-trained models, you are in the right place! One of the greatest things about Caffe was the vibrant community of developers and researchers that shared their work in the [original Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). Below you will not only find a selection of original Caffe models, but we also provide a set of models here that are ready for use with Caffe2.

You can use these models to quickly build demo applications and explore deep learning capabilities without doing any time-consuming and resource-intensive training. You can recreate and evaluate the results from other's projects, hack together new uses, or improve upon the previously posted models. This is the place for sharing too. If you've created something interesting, please create a Github [Issue](https://github.com/caffe2/caffe2/issues) and share your project, so after a community evaluation, we can share it here in the Model Zoo!

### Compatibility

Caffe2 utilizes a newer format, usually found in the protobuf `.pb` file format, so original `.caffemodel` files will require conversion. Several Caffe models have been ported to Caffe2 for you. A tutorial and sample code is also provided so that you may convert any Caffe model to the new Caffe2 format on your own.

#### Converting Models from Caffe to Caffe2

If you have existing Caffe models or have been using Caffe and want a quick jumpstart, checkout the [Caffe Migration](caffe-migration.html) to start.

### Submitting Your Model to Caffe2's Model Zoo

Please file an [Issue](https://github.com/caffe2/caffe2/issues) to have your project and related models added to the zoo, or if you're already a contributor to the project you can add your entry directly to the [wiki page](https://github.com/caffe2/caffe2/wiki/Model-Zoo).

### Downloading and Importing Caffe2 Models

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

### Image Classification

| Name | Type | Dataset | Caffe Model | Caffe2 Model |
|------|------|---------|-------------|--------------|
| [Squeezenet](https://github.com/caffe2/models/tree/master/squeezenet) | image classification | ImageNet > AlexNet |  | [![caffemodel](../static/images/download-c2.png)](https://github.com/caffe2/models/tree/master/squeezenet) |
| [BVLC AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) | image classification | ImageNet > AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel) | [![caffemodel](../static/images/download-c2.png)](https://github.com/caffe2/models/tree/master/bvlc_alexnet) |
| [BVLC CaffeNet Model](https://github.com/BVLC/caffe/blob/80f44100e19fd371ff55beb3ec2ad5919fb6ac43/models/bvlc_reference_caffenet/readme.md) | image classification | ImageNet > AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) | [![caffemodel](../static/images/download-c2.png)](https://github.com/caffe2/models/tree/master/bvlc_reference_caffenet) |
| [BVLC GoogleNet Model](https://github.com/BVLC/caffe/blob/80f44100e19fd371ff55beb3ec2ad5919fb6ac43/models/bvlc_googlenet/readme.md) | image classification | ILSVRC 2014 > GoogleNet/Inception | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) | [![caffemodel](../static/images/download-c2.png)](https://github.com/caffe2/models/tree/master/bvlc_googlenet) |
| [VGG Team ILSVRC14 16-layer](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) | image classification | ILSVRC 2014 > Very Deep CNN | [![caffemodel](../static/images/download-c1.png)](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) | |
| [VGG Team ILSVRC14 19-layer](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) | image classification | ILSVRC 2014 > Very Deep CNN | [![caffemodel](../static/images/download-c1.png)](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel) | |
| [Network in Network ImageNet Model](https://gist.github.com/mavenlin/d802a5849de39225bcc6) | small image classification dataset and fast | ImageNet | [![caffemodel](../static/images/download-c1.png)](https://www.dropbox.com/s/0cidxafrb2wuwxw/nin_imagenet.caffemodel?dl=1) | |
| [Network in Network CIFAR10 Model](https://gist.github.com/mavenlin/e56253735ef32c3c296d) | tiny image classification dataset and fast | CIFAR-10 | [![caffemodel](../static/images/download-c1.png)](https://www.dropbox.com/s/blrajqirr1p31v0/cifar10_nin.caffemodel?dl=1) | |
| [FCN-32s PASCAL](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s) | image classification | ILSVRC14 VGG-16 | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/fcn32s-heavy-pascal.caffemodel) | |
| [FCN-16s-PASCAL](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn16s) | image classification | ILSVRC14 VGG-16 | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/fcn16s-heavy-pascal.caffemodel) | |
| [FCN-8s PASCAL](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s) | image classification | ILSVRC14 VGG-16 | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel) | |
| [FCN-8s PASCAL at-once](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce) | image classification | ILSVRC14 VGG-16 | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/fcn8s-atonce-pascal.caffemodel) | |
| [FCN-AlexNet PASCAL](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn-alexnet) | image classification | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/fcn-alexnet-pascal.caffemodel) | |
| [Places-CNN model](http://places.csail.mit.edu/) | places | Places205-AlexNet | [tar.gz](http://places.csail.mit.edu/model/placesCNN_upgraded.tar.gz) | |
| [Flower Power](http://jimgoo.com/flower-power/) | flowers | Oxford 102 | [![caffemodel](../static/images/download-c1.png)](https://s3.amazonaws.com/jgoode/cannaid/bvlc_reference_caffenet.caffemodel) | |
| [ImageNet ILSVRC13 RCNN](https://github.com/BVLC/caffe/blob/80f44100e19fd371ff55beb3ec2ad5919fb6ac43/models/bvlc_reference_rcnn_ilsvrc13/readme.md) | image classification | ILSVRC13 | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/bvlc_reference_rcnn_ilsvrc13.caffemodel) | [![caffemodel](../static/images/download-c2.png)](https://github.com/caffe2/models/tree/master/bvlc_reference_rcnn_ilsvrc13) |

### Image Segmentation

| Name | Type | Dataset | Caffe Model | Caffe2 Model |
|------|------|---------|-------------|--------------|
| [FCN-32s NYUDv2 Color](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/nyud-fcn32s-color) | image segmentation | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/nyud-fcn32s-color-heavy.caffemodel) | |
| [FCN-32s NYUDv2 HHA](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/nyud-fcn32s-hha) | image segmentation | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/nyud-fcn32s-hha-heavy.caffemodel) | |
| [FCN-32s NYUDv2 Early Color-Depth](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/nyud-fcn32s-color-d) | image segmentation | AlexNet | n/a | |
| [FCN-32s NYUDv2 Late Color-HHA](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/nyud-fcn32s-color-hha) | image segmentation | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/nyud-fcn32s-color-hha-heavy.caffemodel) | |
| [FCN-32s SIFT Flow](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/siftflow-fcn32s) | image segmentation | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/siftflow-fcn32s-heavy.caffemodel) | |
| [FCN-16s SIFT Flow](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/siftflow-fcn16s) | image segmentation | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/siftflow-fcn16s-heavy.caffemodel) | |
| [FCN-8s SIFT Flow](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/siftflow-fcn8s) | image segmentation | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/siftflow-fcn8s-heavy.caffemodel) | |
| [CRF-RNN PASCAL](https://github.com/torrvision/crfasrnn) | image segmentation, object classification | PASCAL | [![caffemodel](../static/images/download-c1.png)](http://goo.gl/j7PrPZ) | |

### Object and Scene Labeling

| Name | Type | Dataset | Caffe Model | Caffe2 Model |
|------|------|---------|-------------|--------------|
| [FCN-32s PASCAL-Context](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/pascalcontext-fcn32s) | object and scene labeling | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/pascalcontext-fcn32s-heavy.caffemodel) | |
| [FCN-16s PASCAL-Context](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/pascalcontext-fcn16s) | object and scene labeling | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/pascalcontext-fcn16s-heavy.caffemodel) | |
| [FCN-8s PASCAL-Context](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/pascalcontext-fcn8s) | object and scene labeling | AlexNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/pascalcontext-fcn8s-heavy.caffemodel) | |
| [Salient Object Subitizing](http://www.cs.bu.edu/groups/ivc/Subitizing/) | object detection | various | | |
| [GoogLeNet_cars on car model classification](https://gist.github.com/bogger/b90eb88e31cd745525ae) | cars | CompCars | [![caffemodel](../static/images/download-c1.png)](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/googlenet_finetune_web_car_iter_10000.caffemodel) | |

### Style

| Name | Type | Dataset | Caffe Model | Caffe2 Model |
|------|------|---------|-------------|--------------|
| [Finetuning CaffeNet on Flickr Style](https://gist.github.com/sergeyk/034c6ac3865563b69e60) | image style | CaffeNet | [![caffemodel](../static/images/download-c1.png)](http://dl.caffe.berkeleyvision.org/finetune_flickr_style.caffemodel) | |

### Faces

| Name | Type | Dataset | Caffe Model | Caffe2 Model |
|------|------|---------|-------------|--------------|
| [Models for Age and Gender Classification](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/) | faces > age, gender | OUI-Adience Face | [zip](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/cnn_age_gender_models_and_data.0.0.2.zip) | |
| [VGG Face CNN descriptor](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) | faces > celebrities | VGG-Face | [tar.gz](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz) | |
| [Yearbook Photo](https://gist.github.com/katerakelly/842f948d568d7f1f0044) | faces > yearbook | VGG-16 + train_val,solver | [![caffemodel](../static/images/download-c1.png)](https://www.dropbox.com/s/6bbbckxwa14ainq/yearbook_cleaned.caffemodel?dl=0) | |
| [Emotion Recognition](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/) | faces > emotion | | [zip](https://dl.dropboxusercontent.com/u/38822310/DemoDir.zip) | |
| [Facial Landmark Detection](http://www.openu.ac.il/home/hassner/projects/tcnn_landmarks/) | faces > landmark | | [![caffemodel](../static/images/download-c1.png)](https://github.com/ishay2b/VanillaCNN/blob/master/ZOO/vanillaCNN.caffemodel) | |

### Video Processing

| Name | Type | Dataset | Caffe Model | Caffe2 Model |
|------|------|---------|-------------|--------------|
| [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html) | scene segmentation: cars, buildings, pedestrians, etc | | | |
| [Translating Videos to Natural Language](https://www.cs.utexas.edu/~vsub/naacl15_project.html) | CNN + RNN, video input, text output | | [![caffemodel](../static/images/download-c1.png)](https://www.dropbox.com/s/edbd49n4hhr7d7x/naacl15_pool_vgg_fc7_mean_fac2.caffemodel?dl=1) | |
| [Sequence to Sequence - Video to Text](https://vsubhashini.github.io/s2vt.html) | CNN + RNN, video input, text output | S2VT_VGG, S2VT_vocabulary | [![caffemodel](../static/images/download-c1.png)](https://www.dropbox.com/s/wn6k2oqurxzt6e2/s2s_vgg_pstream_allvocab_fac2_iter_16000.caffemodel?dl=1) | |
