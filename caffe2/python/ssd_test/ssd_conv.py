import sys
import numpy as np
sys.path.insert(0, '/data2/obj_detect/ssd/caffe/python')
import caffe
ssd_pt = '/data_shared/obj_det_models/ssd_leader_board/VGG_VOC0712_SSD_300x300_ft/deploy.prototxt'
ssd_md = '/data_shared/obj_det_models/ssd_leader_board/VGG_VOC0712_SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'
ssd_net = caffe.Net(ssd_pt, ssd_md, caffe.TEST)
#data_blob = np.random.rand(1, 3, 300, 300).astype(np.float32)
data_blob = np.load('./input.npy')
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


