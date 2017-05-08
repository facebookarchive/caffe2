import sys
import numpy as np
sys.path.insert(0, '/data2/obj_detect/ssd/caffe/python')
import caffe
ssd_pt = '/home/ky/obj_detect/ssd/py-ssd/models/debug_prior_layer/debug.prototxt'
ssd_md = '/home/ky/obj_detect/ssd/caffe/pretrain/models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'
ssd_net = caffe.Net(ssd_pt, ssd_md, caffe.TEST)
conv4_norm_blob = np.random.rand(1, 512, 38, 38).astype(np.float32)
data_blob = np.random.rand(1, 3, 300, 300).astype(np.float32)

ssd_net.blobs['data'].data[...] = data_blob
ssd_net.blobs['conv4_norm'].data[...] = conv4_norm_blob

ssd_net.forward()
prior_data = ssd_net.blobs['conv4_3_norm_mbox_priorbox'].data
np.save('./prior.npy', prior_data)


