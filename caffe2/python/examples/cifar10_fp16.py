import numpy as np
import os
import time
from timeit import default_timer as timer

from caffe2.python import brew, core, cnn, net_drawer, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python.model_helper import ModelHelper

def AddInput(model, batch_size, db, db_type, param_type="float", is_test=False):
    """Adds the data input part."""
    reader = model.CreateDB(
        "reader_"+db,
        db=db,
        db_type=db_type)

    # Note: Using C2 datum for this test
    data, label = model.ImageInput(
        [reader], ["data", "label"],
        use_gpu_transform=True,
        batch_size=batch_size, use_caffe_datum=True,
        mean=128., scale=32, crop=32, mirror=1, color=3,
        is_test=is_test, output_type=param_type
    )
    data = model.StopGradient(data, data)

    return data, label

# Move conv1 to fp16 only
def AddCifar10Model(model, data, param_type):

    with brew.arg_scope([brew.conv, brew.fc], dtype="float16"):
        conv1 = brew.conv(model, data, 'conv1', 3, 32, 5,
                weight_init=('GaussianFill', {'std':0.0001, 'mean' : 0.0}), pad=2)
        pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2) #14 or 15
        relu1 = brew.relu(model, pool1, "relu1")
        conv2 = brew.conv(model, relu1, 'conv2', 32, 32, 5, weight_init=('GaussianFill', {'std' : 0.01}), pad=2)
        conv2 = brew.relu(model, conv2, conv2)
        pool2 = brew.average_pool(model, conv2, 'pool2', kernel=3, stride=2) #5 or 6
        conv3 = brew.conv(model, pool2, 'conv3', 32, 64, 5, weight_init=('GaussianFill', {'std' : 0.01}), pad=2)
        conv3 = brew.relu(model, conv3, conv3)
        pool3 = brew.average_pool(model, conv3, 'pool3', kernel=3, stride=2)
        fc1 = brew.fc(model, pool3, 'fc1', 64 * 3 * 3, 64, weight_init=('GaussianFill', {'std' : 0.1}))
        fc2 = brew.fc(model, fc1, 'fc2', 64, 10, weight_init=('GaussianFill', {'std' : 0.1}))

    # Final cast out -> fp32
    if param_type == "float16":
        fc2 = model.HalfToFloat(fc2, "fc2_fp32")

    softmax = model.Softmax(fc2, 'softmax')

    return softmax

def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""

    accuracy = model.Accuracy([softmax, label], "accuracy", top_k=1)
    return accuracy

def AddTrainingOperators(model, softmax, label, using_snapshot, use_fp16=False):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    loss = model.AveragedLoss(xent, "loss")

    AddAccuracy(model, softmax, label)

    print("adding gradient ops")
    model.AddGradientOperators([loss])


    ITER = brew.iter(model, "ITER")
    LR = model.LearningRate(
        "ITER", "LR", base_lr=0.001, policy="step", stepsize=4000, gamma=0.1)
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

    brew.add_weight_decay(model, 0.004)

    for param in model.params:
        param_grad = model.param_to_grad[param]
        # Check if we need to use the full precision version of parameters for update
        if param in model.param_to_float_copy:
            param_fp32 = model.param_to_float_copy[param]
            # Initialize the momentum from the full fp32 version of the parameters
            param_mom = model.param_init_net.ConstantFill([param_fp32], param_grad+'_mom', value=0.)

            # cast fp16 gradients -> fp32 and update
            grad_fp32 = model.HalfToFloat(param_grad, param_grad+'_fp32')
            model.MomentumSGDUpdate([grad_fp32, param_mom, LR, param_fp32], [grad_fp32, param_mom, param_fp32],
                                    momentum=0.9, nesterov=False)
            # update the fp16 copy of params
            param = model.FloatToHalf(param_fp32, param)
        else:
            param_mom = model.param_init_net.ConstantFill([param], param_grad+'_mom', value=0.)
            model.MomentumSGDUpdate([param_grad, param_mom, LR, param], [param_grad, param_mom, param], momentum=0.9, nesterov=False)

def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)

def createTrainModel(param_type="float"):
    order_scope = { 'order' : 'NCHW' }
    train_model = ModelHelper(name="cifar10_quick_train", arg_scope=order_scope)

    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        data, label = AddInput(
            train_model, batch_size=100,
            db='cifar10_train_lmdb',
            db_type='lmdb', param_type=param_type
            )

        softmax = AddCifar10Model(train_model, data, param_type)

        AddTrainingOperators(train_model, softmax, label, param_type)
        # AddBookkeepingOperators(train_model)

    train_model.net.Proto().type = 'dag'
    train_model.net.Proto().num_workers = 3

    return train_model

def createTestModel(param_type="float"):
    order_scope = { 'order' : 'NCHW' }
    test_model = ModelHelper(
        name="cifar10_quick_test", init_params=False, arg_scope=order_scope)

    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
        data, label = AddInput(
            test_model, batch_size=100,
            db='cifar10_test_lmdb',
            db_type='lmdb',
            param_type=param_type,
            is_test=True)

        softmax = AddCifar10Model(test_model, data, param_type)
        AddAccuracy(test_model, softmax, label)

    test_model.net.Proto().type = 'dag'
    test_model.net.Proto().num_workers = 3

    return test_model

def train_test(train_model, test_model):

    with open("cifar.pbtxt", "w") as fid:
        fid.write(str(train_model.Proto()))
    with open("cifar_init.pbtxt","w") as fid:
        fid.write(str(train_model.param_init_net.Proto()))

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    total_iters = 9001
    display = 1000
    test_interval = 500
    test_iter = 100

    loss = np.zeros(total_iters)
    lr = np.zeros(total_iters)

    accumulated_time = 0.
    for i in range(total_iters):
        start = timer()
        workspace.RunNet(train_model.net.Proto().name)
        end = timer()
        accumulated_time = end - start
        loss[i] = workspace.FetchBlob('loss')
        lr[i] = -workspace.FetchBlob('LR')
        if(i % display == 0):
            time_per_iter = accumulated_time / display
            samples_per_sec= 100 / time_per_iter
            accumulated_time = 0.
            print("Iteration {:4d}, lr = {:8f}, samples / second = {:4f}".format(i, lr[i], samples_per_sec))
            print("    Train net: loss = {:6f}".format(loss[i]))

        if(i % test_interval == 0 and i > 0):
            test_accuracy = np.zeros(test_iter)
            test_loss = np.zeros(test_iter)
            for j in range(test_iter):
                workspace.RunNet(test_model.net.Proto().name)
                test_accuracy[j] = workspace.FetchBlob('accuracy')
            print("Iteration {:4d}, lr = {:8f}".format(i, lr[i]))
            print("    Test net: accuracy = {:4f}".format(test_accuracy.mean()))


def main():
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

    data_folder = '/home/slayton/git/caffe/examples/cifar10'
    stats_folder = data_folder+'/stats'
    workspace.ResetWorkspace(stats_folder)

    param_type = "float16"
    train_model = createTrainModel(param_type)
    test_model = createTestModel(param_type)

    train_test(train_model, test_model)

if __name__ == '__main__':
    main()
