from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import numpy as np
import math

in_c = 32
out_c = 32
group = 2
kern = 1
spatial = 16

#inp = np.random.rand(1, 8, 18, 18).astype(np.float32)
#k = np.random.rand(8, 8, 1, 1).astype(np.float32)


inp = np.array(range(in_c * spatial * spatial)).astype(np.float32)
#inp = np.array([2.0 for _ in range(in_c * spatial * spatial)]).astype(np.float32)
inp = inp.reshape(1,in_c,spatial,spatial)

k = np.array(range(out_c / group * in_c / group * kern * kern)).astype(np.float32)
#k = np.array([1.0 for _ in range(in_c * in_c * kern * kern)]).astype(np.float32)
k = k.reshape(out_c / group, in_c / group, kern, kern)

b = np.zeros(out_c / group).astype(np.float32)#np.random.rand(16).astype(np.float32)
workspace.FeedBlob('input', inp)
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
 kernel=1,
 order="NCHW",
 group=group,
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
  order='NCHW',
  group=group,
  kernel=1,
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > 1 or math.isnan(test_out[i]):
    print(test_out[i], ref_out[i], i, "ERROR")
    if i > 20:
      exit(1)
print("success!")
