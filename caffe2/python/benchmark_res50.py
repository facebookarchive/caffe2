import caffe2.python.models.resnet as resnet
from caffe2.python import cnn
from caffe2.python import workspace
import caffe2.python._import_c_extension as C
import numpy as np
from caffe2.proto import caffe2_pb2
import time

test_model = cnn.CNNModelHelper(
            order="NCHW",
            name="resnet50_test",
            use_cudnn=True,
            cudnn_exhaustive_search=True,
            ws_nbytes_limit = 512*1024*1024
        )
[softmax, loss] = resnet.create_resnet50(test_model, "data",
		num_input_channels=3, num_labels=1000, label="label", no_bias=True)

device_opts = caffe2_pb2.DeviceOption()
device_opts.device_type = caffe2_pb2.CUDA
device_opts.cuda_gpu_id = 0

net_def = test_model.net.Proto()
net_def.device_option.CopyFrom(device_opts)
test_model.param_init_net.RunAllOnGPU(gpu_id=0, use_cudnn=True)

workspace.CreateBlob("data")
workspace.CreateBlob("label")

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(net_def)

workspace.FeedBlob('data', np.random.rand(100, 3, 224, 224).astype(np.float32),
        device_opts)
workspace.FeedBlob('label', np.ones([100, ], dtype=np.int32), device_opts)

#start = time.time()
#for i in range(1000):
#    workspace.RunNet(net_def.name, 1)
#end = time.time()
#print('epoch time: {}'.format((end- start) / 1000))

C.benchmark_net(net_def.name, 0, 1000, True)
