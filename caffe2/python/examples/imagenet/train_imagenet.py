from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import cnn, core, workspace, dyndep, net_drawer
from caffe2.proto import caffe2_pb2

from Networks import *
from ResNet import *
from Inception_v3 import *

import numpy as np
 
import argparse
import time
from timeit import default_timer as timer 
 
##
 #
 # Copyright (c) 2016, NVIDIA CORPORATION, All rights reserved
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #
 # 1. Redistributions of source code must retain the above copyright notice, this
 #    list of conditions and the following disclaimer.
 # 2. Redistributions in binary form must reproduce the above copyright notice,
 #    this list of conditions and the following disclaimer in the documentation
 #    and/or other materials provided with the distribution.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##

dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/nccl:nccl_ops')
 
def getNet(net_type):
    nets = {
        'AlexNet' : AlexNet,
        'AlexNetBN' : AlexNetBN,
        'OverFeat' : OverFeat,
        'VGGA' : VGGA,
        'Inception' : Inception,
        'ResNet' : ResNet,
        'ResNet18' : ResNetHelper(18),
        'ResNet34' : ResNetHelper(34),
        'ResNet50' : ResNetHelper(50),
        'ResNet101' : ResNetHelper(101),
        'ResNet1523' : ResNetHelper(152),
        'InceptionV3' : Inception_v3
    }

    return nets[net_type]

def addData(model, reader, batch_size, crop_size, mirror=True):
    data, label = model.ImageInput(
        [reader], ["data", "label"],
        use_gpu_transform=True,
        batch_size=batch_size, use_caffe_datum=True,
        mean=128., std=128., scale=256, crop=crop_size, mirror=1,
    )
    return data, label

def addAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")

    return accuracy

def generateTestModel(args, num_gpus, crop_size, mirror=False):
    # Don't init params, will reference from GPU0 training net
    test_model = cnn.CNNModelHelper(
        "NCHW", "imagenet_test", use_cudnn=True, cudnn_exhaustive_search=False,
        ws_nbytes_limit=8192, init_params=False)

    reader = test_model.CreateDB(
        "reader",
        db=args.test_db,
        db_type=args.test_db_type)

    # always weak scale testing
    batch_size = args.test_batch_size

    for i in range(num_gpus):
        with core.NameScope("gpu_{}".format(i)):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):

                # Base net
                net = getNet(args.net)
                data, label = addData(test_model, reader, batch_size=batch_size, crop_size=net.CropSize(), mirror=mirror)
                pred = net.Net(test_model, data, test_phase=True)
                xent = test_model.LabelCrossEntropy([pred, label], "xent")
                test_loss = test_model.AveragedLoss(xent, "test_loss")
                
                # Test-specific
                accuracy = addAccuracy(test_model, softmax=pred, label=label)

    return test_model

def generateTrainModel(args, num_gpus, crop_size, mirror):
    train_model = cnn.CNNModelHelper(
        "NCHW", "imagenet_train", use_cudnn=True, cudnn_exhaustive_search=True,
        ws_nbytes_limit=64*1024*1024)
    reader = train_model.CreateDB(
        "reader",
        db=args.db,
        db_type=args.db_type)
    all_loss_gradients = {}

    # handle strong / weak scaling
    batch_size = args.batch_size if args.scaling == 'weak' else args.batch_size / num_gpus

    for i in range(num_gpus):
        with core.NameScope("gpu_{}".format(i)):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):

                # initialize the net on this GPU
                net = getNet(args.net)
                data, label = addData(train_model, reader, batch_size=batch_size, crop_size=net.CropSize(), mirror=True)
                pred = net.Net(train_model, data)

                # Training-specific ops
                xent = train_model.LabelCrossEntropy([pred, label], "xent")
                loss = train_model.AveragedLoss(xent, "loss")
                # Since we are doing multiple GPU, we will need to explicitly
                # create the loss gradients on each GPU.
                one = train_model.ConstantFill(loss, ["loss_grad"], value=1.0)
                all_loss_gradients[str(loss)] = str(one)
 
    train_model.AddGradientOperators(all_loss_gradients)
 
    # After the gradients, we will add an NCCLAllreduce for all the parameters.
    # Also add NCCLBroadcast for initial parameters
    all_params = train_model.params
    assert len(all_params) % num_gpus == 0, "This should not happen."
    params_per_gpu = len(all_params) // num_gpus
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        if num_gpus > 1:
            for i in reversed(range(params_per_gpu)):
                gradients = [
                    train_model.param_to_grad[p] for p in all_params[i::params_per_gpu]
                ]
                train_model.NCCLAllreduce(gradients, gradients)

            # broadcast parameters
            for i in range(params_per_gpu):
                params = all_params[i::params_per_gpu]

                train_model.param_init_net.NCCLBroadcast(params, params)
 
    # add basic training iter
    for i in range(num_gpus):
        with core.NameScope("gpu_{}".format(i)):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                ITER = train_model.Iter("iter")
                LR = train_model.LearningRate(ITER, "LR", base_lr=-0.01, policy="step", stepsize=1, gamma=0.999)
                ONE = train_model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

                # all parameters on this GPU
                for param in all_params[i*params_per_gpu:(i+1)*params_per_gpu]:
                    # print(param)
                    # relevant gradient
                    param_grad = train_model.param_to_grad[param]
                    param_mom = train_model.param_init_net.ConstantFill([param], param_grad+"_mom", value=0.)
                    train_model.MomentumSGD([param_grad, param_mom, LR], [param_grad, param_mom], momentum=0.9, nesterov=False)
                    train_model.WeightedSum([param, ONE, param_grad, ONE], param)

    train_model.net.Proto().type = 'dag'
    train_model.net.Proto().num_workers = num_gpus * 3
 
    # print(train_model.net.Proto())

    return train_model

def parseArgs():
    parser = argparse.ArgumentParser(
        description="image throughput benchmark.")
    parser.add_argument(
        "--db", type=str,
        default="/data/imagenet-compressed/256px/ilsvrc12_train_lmdb",
        help="The input db path.")
    parser.add_argument(
        "--db_type", type=str, default="lmdb",
        help="The input db type.")
    parser.add_argument(
        "--test_db", type=str,
        default="/data/imagenet-compressed/256px/ilsvrc12_val_lmdb",
        help="The input db path.")
    parser.add_argument(
        "--test_db_type", type=str, default="lmdb",
        help="The input db type.")
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="The batch size.")
    parser.add_argument(
        "--test_batch_size", type=int, default=50,
        help="The batch size.")
    parser.add_argument(
        "--caffe2_log_level", type=int, default=0,
        help="The log level of caffe2.")
    parser.add_argument(
        "--gpus", type=int, default=-1,
        help="Number of GPUs to use")
    parser.add_argument(
        "--save_graph", action="store_true",
        help="Save pdf of graph to disk")
    parser.add_argument(
        "--net", type=str, default="AlexNet",
        help="Network type to train")
    parser.add_argument(
        "--scaling", choices=['strong', 'weak'], type=str,
        default='weak',
        help="Type of scaling used")
    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    workspace.GlobalInit(
        ["caffe2", "-caffe2_log_level={}".format(args.caffe2_log_level)])
 
    num_gpus = workspace.NumCudaDevices() if (args.gpus <= 0) else args.gpus

    # generate the training model on all GPUs
    train_model = generateTrainModel(args, num_gpus=num_gpus, crop_size=227, mirror=True)
 
    #generate the test model on GPU 0
    test_model = generateTestModel(args, num_gpus=num_gpus, crop_size=227, mirror=False)

    # draw the model.
    with open("imagenet_multi_gpu.pbtxt", "w") as fid:
        fid.write(str(train_model.net.Proto()))

    if args.save_graph:
        graph = net_drawer.GetPydotGraphMinimal(train_model.net.Proto().op, "imagenet")
        graph.write_pdf("imagenet_multi_gpu.pdf")
 
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    num_iter = 2500
    test_interval = 500
    print_interval = 100

    aggregate_batch_size = args.batch_size*num_gpus if args.scaling == 'weak' else args.batch_size
    batch_size_per_gpu = args.batch_size if args.scaling == 'weak' else args.batch_size / num_gpus

    print('\nTotal batchsize: {}, {} per GPU'.format(aggregate_batch_size, batch_size_per_gpu))
    print('\n==== Starting Optimization Loop ({0} iterations) ====\n'.format(num_iter))
    accumulated_time = 0.
    for i in range(num_iter+1):
        start = timer()
        workspace.RunNet(train_model.net.Proto().name)
        end = timer()
        accumulated_time += end-start

        # run test net, track Accuracy
        acc_str = ""
        if (i % test_interval == 0 and i != 0):
            workspace.RunNet(test_model.net.Proto().name)
            loss_sum = 0.
            acc_sum = 0.
            for g in range(num_gpus):
                acc_sum += workspace.FetchBlob("gpu_{}/accuracy".format(g))
                loss_sum += workspace.FetchBlob("gpu_{}/test_loss".format(g))
            acc_str = ", test loss: {0:4f}, test accuracy: {1:.3f}".format(float(loss_sum / num_gpus), float(acc_sum))
            # print("iter: {0:8d} test loss: {1:4f}, test accuracy: {2:4f}".format(i, float(loss_sum / num_gpus), float(acc_sum)))

        # track Loss
        if (i % print_interval == 0):
            loss = workspace.FetchBlob("gpu_0/loss")
            time_per_iter = accumulated_time / print_interval
            accumulated_time = 0.
            samples_per_second = aggregate_batch_size / time_per_iter
            output_str = "[{timestamp}] iter: {0:8d}, loss: {1:4f}, {2:.0f} images/second".format(i, float(loss), float(samples_per_second), timestamp=time.strftime('%H:%M:%S'))

            output_str += acc_str
            print(output_str)

    workspace.ResetWorkspace()
 
if __name__ == '__main__':
    main()
