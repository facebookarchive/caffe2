
import sys
import numpy as np
'''
sys.path.insert(0, '/data2/obj_detect/ssd/caffe/python')
import caffe
ssd_pt = '/home/ky/obj_detect/ssd/py-ssd/models/debug_norm_layer/debug.prototxt'
ssd_md = '/home/ky/obj_detect/ssd/caffe/pretrain/models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'
ssd_net = caffe.Net(ssd_pt, ssd_md, caffe.TEST)
data_blob = np.random.rand(1, 512, 38, 38).astype(np.float32)
ssd_net.blobs['data'].data[...] = data_blob
ssd_scale = ssd_net.params['conv4_3_norm'][0].data
ssd_net.forward()
norm_data = ssd_net.blobs['conv4_3_norm'].data
np.save('./data.npy', data_blob)
np.save('./scale.npy', ssd_scale)
np.save('./conv4_3_norm_out.npy', norm_data)
'''


from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

device_opts = caffe2_pb2.DeviceOption()
device_opts.device_type = caffe2_pb2.CUDA
device_opts.cuda_gpu_id = 0


net = core.Net("norm_test")
net.Norm(["data", "scale"], "norm_out", name="norm_layer")
net.RunAllOnGPU(gpu_id=0)
data = np.load('./data.npy')
scale = np.load('./scale.npy')
out = np.load('conv4_3_norm_out.npy')
workspace.FeedBlob("data", data, device_opts)
workspace.FeedBlob("scale", scale, device_opts)
'''
workspace.CreateNet(net.Proto())

start = time.time()
for i in range(1000):
    workspace.RunNet("norm_test", 1)
end = time.time()
print (end - start) / 1000
caffe2_out = workspace.FetchBlob("norm_out")

print np.allclose(caffe2_out, out)
'''
print net.Proto()

