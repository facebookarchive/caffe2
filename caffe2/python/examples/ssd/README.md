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
- [x] VGG_VOC0712Plus_SSD_300x300_ft

### Getting started
* Clone my caffe2 & Build Caffe2: Follow the official instructions.

### Models
- [VGGNet_VOC0712_SSD_300x300_ft](https://pan.baidu.com/s/1gfceC6Z)
- [VGG_VOC0712Plus_SSD_300x300_ft](https://pan.baidu.com/s/1i5iMl17)

### Example
After download the original SSD model, put it to `{CAFFE2_HOME}/caffe2/python/examples/ssd` and extract it:
1. Run
```
# change folder
cd {CAFFE2_HOME}/caffe2/python/examples/ssd
# copy model
cp {Download}/models_VGGNet_VOC0712_SSD_300x300_ft.tar.gz .
# get the prototxt and caffemodel
tar -xvf models_VGGNet_VOC0712_SSD_300x300_ft.tar.gz
```
2. Convert deploy.prototxt and VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel
to caffe2.
```
# change to {CAFFE2_HOME}/caffe2/python
cd ../../
# converting to caffe2 deploy.pb and model.pb
python caffe_translator.py 
	examples/ssd/models/VGGNet/VOC0712/SSD_300x300_ft/deploy.prototxt 
	examples/ssd/models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel 
	--init_net examples/ssd/VGG_VOC0712_SSD_300x300_ft_iter_120000_model.pb 
	--predict_net examples/ssd/VGG_VOC0712_SSD_300x300_ft_iter_120000_deploy.pb
```
3. Check out `examples/ssd/visualize_caffe2_implementation_det.ipynb` on how to detect objects using a caffe2 SSD model.
Check out `examples/ssd/visualize_caffe_implementation_det.ipynb` to compare the result of caffe with caffe2.
