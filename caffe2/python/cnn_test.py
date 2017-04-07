import os
import random
import shutil
import tempfile
import unittest

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import cnn, core, data_parallel_model, test_util, workspace


@unittest.skipIf(workspace.NumCudaDevices() == 0, 'No GPUs')
class TestSaveLoad(test_util.TestCase):
    """Test CNNModelHelper.{Load,Save}* functions."""

    def setUp(self):
        super(TestSaveLoad, self).setUp()
        self.snapshot_dir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestSaveLoad, self).tearDown()
        if os.path.exists(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir)

    @staticmethod
    def AddInputOps(model):
        pass

    @staticmethod
    def AddForwardPassOps(model, loss_scale):
        fc = model.FC('data', 'fc', 10, 4)
        softmax = model.Softmax(fc, 'softmax')
        xent = model.LabelCrossEntropy([softmax, 'label'], 'xent')
        accuracy = model.Accuracy([softmax, 'label'], 'accuracy')
        loss = model.AveragedLoss(xent, 'loss')
        return [loss]

    @staticmethod
    def AddParamUpdateOps(model):
        _iter = model.Iter('iter')
        lr = model.LearningRate([_iter], 'lr', base_lr=0.1,
                                policy='step', stepsize=1, gamma=0.999)
        for param in model.GetParams():
            grad = model.param_to_grad[param]
            mom = model.param_init_net.ConstantFill(
                param, grad + '_momentum', value=0.)
            model.solverstate_params.append(mom)
            model.net.MomentumSGDUpdate(
                [grad, mom, lr, param],
                [grad, mom, param], momentum=0.9, nesterov=False)

    @classmethod
    def CreateTrainModel(cls):
        model = cnn.CNNModelHelper()
        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=cls.AddInputOps,
            forward_pass_builder_fun=cls.AddForwardPassOps,
            param_update_builder_fun=cls.AddParamUpdateOps,
            devices=[0])
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)
        return model

    @classmethod
    def CreateTestModel(cls):
        model = cnn.CNNModelHelper(init_params=False)
        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=cls.AddInputOps,
            forward_pass_builder_fun=cls.AddForwardPassOps,
            param_update_builder_fun=None,
            devices=[0])
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)
        return model

    @staticmethod
    def Run(model, data, labels):
        for batch, label in zip(data, labels):
            workspace.FeedBlob('data', data)
            workspace.FeedBlob('label', label)
            workspace.RunNet(model.net.Proto().name)
        return {
            'iter': workspace.FetchBlob('gpu_0/iter'),
            'lr': workspace.FetchBlob('gpu_0/lr'),
            'loss': workspace.FetchBlob('gpu_0/loss'),
            'accuracy': workspace.FetchBlob('gpu_0/accuracy'),
        }

    def test_save_load(self):
        # Random data (Batch, N, features)
        data = np.random.rand(100, 10, 10).astype(np.float32)

        # Labels are [0,3], data is shifted by label
        labels = np.zeros((100, 10), dtype=np.int32)
        for b in range(data.shape[0]):
            for n in range(data.shape[1]):
                cls = random.randrange(4)
                labels[b][n] = cls
                data[b][n] += cls

        # Split up the data
        data_train1 = data[:40]
        labels_train1 = labels[:40]
        data_train2 = data[40:80]
        labels_train2 = labels[40:80]
        data_test = data[80:]
        labels_test = labels[80:]

        # Create the data/label blobs before creating any net
        device = core.DeviceOption(caffe2_pb2.CUDA, 0)
        workspace.FeedBlob('gpu_0/data', data[0], device)
        workspace.FeedBlob('gpu_0/label', labels[0], device)

        # Create the models
        train_model = self.CreateTrainModel()
        test_model = self.CreateTestModel()

        # Train on 40 batches
        train_results_A = self.Run(train_model, data_train1, labels_train1)

        # Save state
        params_file = os.path.join(self.snapshot_dir, 'params.minidb')
        solverstate_file = os.path.join(self.snapshot_dir, 'solverstate.minidb')
        train_model.SaveParamBlobs(
            db=params_file, db_type='minidb',
            strip_prefix='/', absolute_path=True)
        train_model.SaveSolverstateBlobs(
            db=solverstate_file, db_type='minidb',
            strip_prefix='/', absolute_path=True)

        # Continue training for another 40 batches
        train_results_B1 = self.Run(train_model, data_train2, labels_train2)
        test_results_B1 = self.Run(test_model, data_test, labels_test)

        # Load state
        train_model.LoadParamBlobs(
            db=params_file, db_type='minidb',
            add_prefix='gpu_0/', keep_device=True, absolute_path=True)
        train_model.LoadSolverstateBlobs(
            db=solverstate_file, db_type='minidb',
            add_prefix='gpu_0/', keep_device=True, absolute_path=True)

        # Train again on the second set of 40 batches
        train_results_B2 = self.Run(train_model, data_train2, labels_train2)
        test_results_B2 = self.Run(test_model, data_test, labels_test)

        self.assertEquals(train_results_B1, train_results_B2)
        self.assertEquals(test_results_B1, test_results_B2)
