import sys
import numpy as np
sys.path.insert(0, '/data2/obj_detect/ssd/caffe/python')
import caffe
ssd_pt = '/data_shared/obj_det_models/ssd_leader_board/VGG_VOC0712_SSD_300x300_ft/deploy.prototxt'
ssd_md = '/data_shared/obj_det_models/ssd_leader_board/VGG_VOC0712_SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'
ssd_img = '/data2/obj_detect/ssd/caffe/examples/images/fish-bike.jpg'

net = caffe.Net(ssd_pt, ssd_md, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)
image = caffe.io.load_image(ssd_img)
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

detections = net.forward()['detection_out']


np.save('./detections.npy', detections)
np.save('./input.npy', transformed_image)


