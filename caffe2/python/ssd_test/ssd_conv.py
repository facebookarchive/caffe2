import sys
import numpy as np
sys.path.insert(0, '/data2/obj_detect/ssd/caffe/python')
import caffe
ssd_pt = '/data2/obj_detect/learn/pybind11/learn/caffe2/caffe2/python/ssd_test/debug_ssd_conv_layer/debug.prototxt'
ssd_md = '/home/ky/obj_detect/ssd/caffe/pretrain/models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'
ssd_net = caffe.Net(ssd_pt, ssd_md, caffe.TEST)
data_blob = np.random.rand(1, 3, 300, 300).astype(np.float32)

ssd_net.blobs['data'].data[...] = data_blob

ssd_net.forward()

ty = ['_w', '_b']

for blob_name in ssd_net.blobs:
    blob = ssd_net.blobs[blob_name].data
    np.save('./{}.npy'.format(blob_name), blob)
    
    if blob_name in ssd_net.params:
        params = ssd_net.params[blob_name]
        for i in range(len(params)):
            param = params[i].data
            np.save('./{}.npy'.format(blob_name+ty[i]), param)


