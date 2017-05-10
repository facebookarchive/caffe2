# SSD: Single Shot MultiBox Object Detector

SSD is an unified framework for object detection with a single network.

### Disclaimer
This is a re-implementation of original SSD which is based on caffe. The official
repository is available [here](https://github.com/weiliu89/caffe/tree/ssd).
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).

This example is intended for SSD detection in caffe2. Converting the caffe trained model to caffe2. 
However, due to different implementation details, the results might differ slightly.

### Timeline
- [x] VGGNet_VOC0712_SSD_300x300_ft


### Getting started
* Clone my caffe2 & Build Caffe2: Follow the official instructions.

### Models
- [VGGNet_VOC0712_SSD_300x300_ft](https://pan.baidu.com/s/1gfceC6Z)

### VGGNet_VOC0712_SSD_300x300_ft
After download the models_VGGNet_VOC0712_SSD_300x300_ft.tar.gz model, 
put it to {CAFFE2_HOME/caffe2/python/examples/ssd} and extract it:
* Run
```
cd {CAFFE2_HOME/caffe2/python}
cp models_VGGNet_VOC0712_SSD_300x300_ft.tar.gz ./examples/ssd
tar -xvf examples/ssd/models_VGGNet_VOC0712_SSD_300x300_ft.tar.gz
# Get the prototxt and caffemodel
```
* Convert deploy.prototxt and VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel
to caffe2.
