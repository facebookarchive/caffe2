from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import cnn, core, workspace, test_util


@unittest.skipIf(not workspace.C.has_mkldnn, "Skipping as we do not have mkldnn.")
class TestMKLBasic(test_util.TestCase):
    def testAlexNet(self):
        bs = 1
        X = np.random.rand(bs, 3, 224, 224).astype(np.float32) - 0.5
        W1 = np.random.rand(64, 3, 11, 11).astype(np.float32) - 0.5
        b1 = np.random.rand(64).astype(np.float32) - 0.5
        
        W2 = np.random.rand(192, 64, 5, 5).astype(np.float32) - 0.5
        b2 = np.random.rand(192).astype(np.float32) - 0.5

        W3 = np.random.rand(384, 192, 3, 3).astype(np.float32) - 0.5
        b3 = np.random.rand(384).astype(np.float32) - 0.5

        W4 = np.random.rand(256, 384, 3, 3).astype(np.float32) - 0.5
        b4 = np.random.rand(256).astype(np.float32) - 0.5

        W5 = np.random.rand(256, 256, 3, 3).astype(np.float32) - 0.5
        b5 = np.random.rand(256).astype(np.float32) - 0.5

        Wfc1 = np.random.rand(4096, 256*6*6).astype(np.float32) - 0.5
        bfc1 = np.random.rand(4096).astype(np.float32) - 0.5
        
        Wfc2 = np.random.rand(4096, 4096).astype(np.float32) - 0.5
        bfc2 = np.random.rand(4096).astype(np.float32) - 0.5

        Wfc3 = np.random.rand(1000, 4096).astype(np.float32) - 0.5
        bfc3 = np.random.rand(1000).astype(np.float32) - 0.5

        #label = np.arange(1, 1000, 1).astype(np.int32)
        #label = np.array[1000].astype(np.int32)
        #label = np.array([1, 2, 3, 4]).astype(np.int32)
        label = np.random.randint(low=0, high=1000, size=(bs,)).astype(np.int32)


        mkl_do = core.DeviceOption(caffe2_pb2.MKLDNN)

        # feed cpu
        '''w = {w1, w2, w3, w4}
        b = {b1, b2, b3, b4}
        wfc = {Wfc1, Wfc2, Wfc3}
        bfc = {bfc1, bfc2, bfc3}'''

        workspace.FeedBlob("X", X)
        workspace.FeedBlob("W1", W1)
        workspace.FeedBlob("b1", b1)
        
        workspace.FeedBlob("W2", W2)
        workspace.FeedBlob("b2", b2)
        workspace.FeedBlob("W3", W3)
        workspace.FeedBlob("b3", b3)
        workspace.FeedBlob("W4", W4)
        workspace.FeedBlob("b4", b4)
        workspace.FeedBlob("W5", W5)
        workspace.FeedBlob("b5", b5)

        workspace.FeedBlob("Wfc1", Wfc1)
        workspace.FeedBlob("bfc1", bfc1)        
        workspace.FeedBlob("Wfc2", Wfc2)
        workspace.FeedBlob("bfc2", bfc2)
        workspace.FeedBlob("Wfc3", Wfc3)
        workspace.FeedBlob("bfc3", bfc3)
        
        workspace.FeedBlob("label", label)

        #feed mkl
        workspace.FeedBlob("Xmkl", X, device_option=mkl_do)
        workspace.FeedBlob("W1mkl", W1, device_option=mkl_do)
        workspace.FeedBlob("b1mkl", b1, device_option=mkl_do)
        
        workspace.FeedBlob("W2mkl", W2, device_option=mkl_do)
        workspace.FeedBlob("b2mkl", b2, device_option=mkl_do)
        workspace.FeedBlob("W3mkl", W3, device_option=mkl_do)
        workspace.FeedBlob("b3mkl", b3, device_option=mkl_do)
        workspace.FeedBlob("W4mkl", W4, device_option=mkl_do)
        workspace.FeedBlob("b4mkl", b4, device_option=mkl_do)
        workspace.FeedBlob("W5mkl", W5, device_option=mkl_do)
        workspace.FeedBlob("b5mkl", b5, device_option=mkl_do)

        workspace.FeedBlob("Wfc1mkl", Wfc1, device_option=mkl_do)
        workspace.FeedBlob("bfc1mkl", bfc1, device_option=mkl_do)        
        workspace.FeedBlob("Wfc2mkl", Wfc2, device_option=mkl_do)
        workspace.FeedBlob("bfc2mkl", bfc2, device_option=mkl_do)
        workspace.FeedBlob("Wfc3mkl", Wfc3, device_option=mkl_do)
        workspace.FeedBlob("bfc3mkl", bfc3, device_option=mkl_do)
        
        #workspace.FeedBlob("labelmkl", label, device_option=mkl_do)

        net = core.Net("test")
        
        net.Conv(["X", "W1", "b1"], "C1", pad=2, stride=4, kernel=11)
        net.Relu("C1", "R1")
        net.MaxPool("R1", "P1", stride=2, kernel=3)
        net.Conv(["P1", "W2", "b2"], "C2", pad=2, kernel=5)
        net.Relu("C2", "R2")
        net.MaxPool("R2", "P2", stride=2, kernel=3)
        net.Conv(["P2", "W3", "b3"], "C3", pad=1, kernel=3)
        net.Relu("C3", "R3")
        net.Conv(["R3", "W4", "b4"], "C4", pad=1, kernel=3)
        net.Relu("C4", "R4")
        net.Conv(["R4", "W5", "b5"], "C5", pad=1, kernel=3)
        net.Relu("C5", "R5")
        net.MaxPool("R5", "P3", stride=2, kernel=3)
        net.FC(["P3","Wfc1", "bfc1"], "fc1")
        net.Relu("fc1", "R6")
        net.FC(["R6","Wfc2", "bfc2"], "fc2")
        net.Relu("fc2", "R7")
        net.FC(["R7","Wfc3", "bfc3"], "fc3")
        net.Softmax("fc3", "pred")
        net.LabelCrossEntropy(["pred", "label"], "xent")
        net.AveragedLoss("xent", "loss")
        
        #mkl net
        net.Conv(["Xmkl", "W1mkl", "b1mkl"], "C1mkl", pad=2, stride=4, kernel=11, device_option=mkl_do)
        net.Relu("C1mkl", "R1mkl", device_option=mkl_do)
        net.MaxPool("R1mkl", "P1mkl", stride=2, kernel=3, device_option=mkl_do)
        net.Conv(["P1mkl", "W2mkl", "b2mkl"], "C2mkl", pad=2, kernel=5, device_option=mkl_do)
        net.Relu("C2mkl", "R2mkl", device_option=mkl_do)
        net.MaxPool("R2mkl", "P2mkl", stride=2, kernel=3, device_option=mkl_do)
        net.Conv(["P2mkl", "W3mkl", "b3mkl"], "C3mkl", pad=1, kernel=3, device_option=mkl_do)
        net.Relu("C3mkl", "R3mkl", device_option=mkl_do)
        net.Conv(["R3mkl", "W4mkl", "b4mkl"], "C4mkl", pad=1, kernel=3, device_option=mkl_do)
        net.Relu("C4mkl", "R4mkl", device_option=mkl_do)
        net.Conv(["R4mkl", "W5mkl", "b5mkl"], "C5mkl", pad=1, kernel=3, device_option=mkl_do)
        net.Relu("C5mkl", "R5mkl", device_option=mkl_do)
        net.MaxPool("R5mkl", "P3mkl", stride=2, kernel=3, device_option=mkl_do)
        net.FC(["P3mkl","Wfc1mkl", "bfc1mkl"], "fc1mkl", device_option=mkl_do)
        net.Relu("fc1mkl", "R6mkl", device_option=mkl_do)
        net.FC(["R6mkl","Wfc2mkl", "bfc2mkl"], "fc2mkl", device_option=mkl_do)
        net.Relu("fc2mkl", "R7mkl", device_option=mkl_do)
        net.FC(["R7mkl","Wfc3mkl", "bfc3mkl"], "fc3mkl", device_option=mkl_do)
        net.Softmax("fc3mkl", "pred_mkl", device_option=mkl_do)
        net.LabelCrossEntropy(["pred_mkl", "label"], "xent_mkl", device_option=mkl_do)
        net.AveragedLoss("xent_mkl", "loss_mkl", device_option=mkl_do)
        
        #print("Current blobs in the workspace: {}".format(workspace.Blobs()))

       
        workspace.CreateNet(net)
        workspace.RunNet(net)

        # makes sure that the results are good.
	print("CPU loss {}, MKL loss {}.".format(workspace.FetchBlob("loss"), workspace.FetchBlob("loss_mkl")))	
        '''np.testing.assert_allclose(
            workspace.FetchBlob("fc3"),
            workspace.FetchBlob("fc3mkl"),
            atol=1e-1,
            rtol=1e-1)'''
            
        runtime = workspace.BenchmarkNet(net.Proto().name, 1, 100, False)
    

if __name__ == '__main__':
    unittest.main()
