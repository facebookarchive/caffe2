from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import numpy as np

i = np.random.rand(1, 18, 18, 4).astype(np.float32)
k = np.random.rand(4, 3, 3, 4).astype(np.float32)
b = np.random.rand(4).astype(np.float32)
workspace.FeedBlob('input', i)
workspace.FeedBlob('kernel', k)
workspace.FeedBlob('bias', b)

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
 ['input'],
 ['cl_input'],
)
op = core.CreateOperator("Conv",
 ['cl_input', 'kernel', 'bias'],
 ['cl_output'],
 kernel=3,
 order="NHWC",
 device_option=device_option,
)
op_copy2 = core.CreateOperator("CopyFromOpenCL",
 ['cl_output'],
 ['output'],
 #device_option=device_option,
)
workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy2)

ref_op = core.CreateOperator("Conv",
  ['input', 'kernel', 'bias'],
  ['ref_output'],
  order='NHWC',
  kernel=3,
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
for i in range(len(test_out)):
  if abs(test_out[i] - ref_out[i]) > 1:
    print(test_out[i], ref_out[i], i)
    exit(1)
print("success!")
