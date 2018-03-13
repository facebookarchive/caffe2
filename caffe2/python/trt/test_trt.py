# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import onnx
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info, make_model
from caffe2.python.onnx.helper import c2_native_run_net, c2_native_run_op
from onnx.backend.base import namedtupledict
import caffe2.python.onnx.backend as c2
import caffe2.python.onnx.frontend as c2_front
from caffe2.python.onnx.workspace import Workspace
import numpy as np
import os.path
import json

import caffe2.python._import_c_extension as C

# Note that ONNX-TRT enforce an NCHW input!!!!

def dim_values_to_list(dim_values):
    return [x.dim_value for x in dim_values]

def get_output_shapes(output_value_infos):
    names = [x.name for x in output_value_infos]
    shapes = [dim_values_to_list(x.type.tensor_type.shape.dim) for x in output_value_infos]
    return dict(zip(names, shapes))

def print_net(net):
    for i in net.external_input:
        print("Input: {}".format(i))
    for i in net.external_output:
        print("Output: {}".format(i))
    for op in net.op:
        print("Op {}".format(op.type))
        for x in op.input:
            print("  input: {}".format(x))
        for y in op.output:
            print("  output: {}".format(y))

def test_relu_graph():
    X = np.random.randn(1, 1, 3, 2).astype(np.float32)
    node_def = make_node("Relu", ["X"], ["Y"])
    Y_c2 = c2.run_node(node_def, {"X": X})
    graph_def = make_graph(
        [node_def],
        name="test",
        inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 3, 2])],
        outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1, 3, 2])])
    model_def = make_model(graph_def, producer_name='relu-test')
    op_outputs = [x.name for x in model_def.graph.output]
    #print("Onnx Model: {}".format(model_def))
    get_output_shapes(graph_def.output)
    trt_str = C.onnx_to_trt_op(model_def.SerializeToString(), get_output_shapes(graph_def.output))
    op = caffe2_pb2.OperatorDef()
    op.ParseFromString(trt_str)
    device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
    op.device_option.CopyFrom(device_option)
    #print("{}".format(op))
    Y_trt = None
    with Workspace(), core.DeviceScope(device_option):  # temporary!
        workspace.FeedBlob("X", X)
        workspace.RunOperatorsOnce([op])
        output_values = [workspace.FetchBlob(name) for name in op_outputs]
        Y_trt = namedtupledict('Outputs', op_outputs)(*output_values)
    np.testing.assert_almost_equal(Y_c2, Y_trt)

def test_resnet50():
    input_blob_dims = (1, 3, 224, 224)
    model_def = onnx.load('/home/yinghai/.onnx/models/resnet50/model.onnx')
    op_inputs = [x.name for x in model_def.graph.input]
    op_outputs = [x.name for x in model_def.graph.output]
    print("Inputs: {}".format(op_inputs))
    print("Outputs: {}".format(op_outputs))
    n, c, h, w = input_blob_dims
    data = np.random.randn(n, c, h, w).astype(np.float32)
    Y_c2 = c2.run_model(model_def, {op_inputs[0]: data})
    trt_str = C.onnx_to_trt_op(model_def.SerializeToString(), get_output_shapes(model_def.graph.output))
    op = caffe2_pb2.OperatorDef()
    op.ParseFromString(trt_str)
    device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
    op.device_option.CopyFrom(device_option)
    Y_trt = None
    with Workspace(), core.DeviceScope(device_option):  # temporary!
        workspace.FeedBlob(op_inputs[0], data)
        workspace.RunOperatorsOnce([op])
        output_values = [workspace.FetchBlob(name) for name in op_outputs]
        Y_trt = namedtupledict('Outputs', op_outputs)(*output_values)
    np.testing.assert_almost_equal(Y_c2, Y_trt)

def infer_shape(init_net, pred_net, inputs):
    ws, outputs = c2_native_run_net(init_net, pred_net, inputs)
    hints = {}
    for op in pred_net.op:
        for o in op.output:
            if o not in hints:
                blob = ws.FetchBlob(o)
                if hasattr(blob, 'shape'):
                    hints[o] = blob.shape
        for i in op.input:
            if i not in hints:
                blob = ws.FetchBlob(i)
                if hasattr(blob, 'shape'):
                    hints[i] = blob.shape

    #print("Shapes: {}".format(hints))
    return hints

def add_head_tail(pred_net):
    # Add head
    head = caffe2_pb2.OperatorDef()
    head.type = "Copy"
    head.input.append("real_data_0")
    head.output.append("gpu_0/data_0")
    dummy = caffe2_pb2.NetDef()
    dummy.op.extend(pred_net.op)
    del pred_net.op[:]
    pred_net.op.extend([head])
    pred_net.op.extend(dummy.op)
    pred_net.external_input[0] = "real_data_0"

    # Add tail
    tail = caffe2_pb2.OperatorDef()
    tail.type = "Copy"
    tail.input.append("gpu_0/softmax_1")
    tail.output.append("real_softmax_1")
    pred_net.op.extend([tail])
    pred_net.external_output[0] = "real_softmax_1"
    #print_net(pred_net)

def test_resnet50_cut():
    init_net = caffe2_pb2.NetDef()
    model_path = '/home/yinghai/.caffe2/models/resnet50/'
    with open(os.path.join(model_path + 'init_net.pb'), 'rb') as f:
        init_net.ParseFromString(f.read())
    pred_net = caffe2_pb2.NetDef()
    with open(os.path.join(model_path + 'predict_net.pb'), 'rb') as f:
        pred_net.ParseFromString(f.read())
    print("Loaded resnet model: {}, {}".format(init_net.name, pred_net.name))
    c2_front.ssa_rewrite(pred_net, init_net, value_info=json.load(open(os.path.join(model_path, 'value_info.json'))))
    add_head_tail(pred_net)
    input_blob_dims = (1, 3, 224, 224)
    n, c, h, w = input_blob_dims
    data = np.random.randn(n, c, h, w).astype(np.float32)
    input_name = "real_data_0"
    shape_hints = infer_shape(init_net, pred_net, {input_name: data})
    shape_hints[input_name] = input_blob_dims

    device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
    init_net.device_option.CopyFrom(device_option)
    pred_net.device_option.CopyFrom(device_option)
    for op in pred_net.op:
        op.device_option.CopyFrom(device_option)
    net_outputs = pred_net.external_output
    Y_c2 = None
    with Workspace(), core.DeviceScope(device_option):  # temporary!
        workspace.FeedBlob(input_name, data)
        workspace.RunNetOnce(init_net)
        workspace.RunNetOnce(pred_net)
        output_values = [workspace.FetchBlob(name) for name in net_outputs]
        Y_c2 = namedtupledict('Outputs', net_outputs)(*output_values)

    # Cut the graph
    init_net_str, pred_net_str = C.transform_trt(init_net.SerializeToString(), pred_net.SerializeToString(), shape_hints)
    init_net_cut = caffe2_pb2.NetDef()
    init_net_cut.ParseFromString(init_net_str)
    pred_net_cut = caffe2_pb2.NetDef()
    pred_net_cut.ParseFromString(pred_net_str)
    del init_net_str, pred_net_str
    print_net(pred_net_cut)
    Y_trt = None
    with Workspace(), core.DeviceScope(device_option):  # temporary!
        workspace.FeedBlob(input_name, data)
        workspace.RunNetOnce(init_net_cut)
        workspace.RunNetOnce(pred_net_cut)
        output_values = [workspace.FetchBlob(name) for name in net_outputs]
        Y_trt = namedtupledict('Outputs', net_outputs)(*output_values)
    np.testing.assert_almost_equal(Y_c2, Y_trt)

def test_detectron():
    init_net = caffe2_pb2.NetDef()
    model_path = '/home/yinghai/.caffe2/models/detectron/e2e_faster_rcnn_R-50-C4_1x/'
    with open(os.path.join(model_path + 'init_net.pb'), 'rb') as f:
        init_net.ParseFromString(f.read())
    pred_net = caffe2_pb2.NetDef()
    with open(os.path.join(model_path + 'predict_net.pb'), 'rb') as f:
        pred_net.ParseFromString(f.read())
    print("Loaded detectron model: {}, {}".format(init_net.name, pred_net.name))
    #c2_front.ssa_rewrite(pred_net, init_net, value_info=json.load(open(os.path.join(model_path, 'value_info.json'))))
    input_blob_dims = (1, 3, 800, 800)
    for i in pred_net.external_input:
        print("Input: {}".format(i))
    #for i in pred_net.external_output:
    #    print("Output: {}".format(i))
    #for op in pred_net.op:
    #    print("Saw {}".format(op.type))
    #    for x in op.input:
    #        print("  input: {}".format(x))
    #    for y in op.output:
    #        print("  output: {}".format(y))
    n, c, h, w = input_blob_dims
    data = np.random.randn(n, c, h, w).astype(np.float32)
    im_info = np.random.randn(n, 3).astype(np.float32)
    ws, outputs = c2_native_run_net(init_net, pred_net, {"data":data, "im_info":im_info})

if __name__ == '__main__':
    #test_relu_graph()
    #test_resnet50()
    test_resnet50_cut()
    #test_detectron()
