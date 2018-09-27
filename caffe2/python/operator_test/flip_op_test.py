from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np
from caffe2.proto import caffe2_pb2

import unittest


class TestFlip(hu.HypothesisTestCase):

    @given(N=st.sampled_from([2,3]),
           C=st.sampled_from([3,5]),
           H=st.sampled_from([5,8]),
           W=st.sampled_from([8,13]),
           engine=st.sampled_from([None]), # CPU only by now
           **hu.gcs)
    def test_flip(self, N, C, H, W, engine, gc, dc):
        X = np.random.rand(N, C, H, W).astype(np.float32)

        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(3,), engine=engine)
        
        def ref_fliplr(X):
            return [np.flip(X, 3)]

        self.assertReferenceChecks(
            device_option=core.DeviceOption(caffe2_pb2.CPU, 0), # CPU only by now
            op=op,
            inputs=[X],
            reference=ref_fliplr,
        )

    @given(N=st.sampled_from([2,3]),
           C=st.sampled_from([3,5]),
           H=st.sampled_from([5,8]),
           W=st.sampled_from([8,13]),
           engine=st.sampled_from([None]), # CPU only by now
           **hu.gcs)
    def test_flip2(self, N, C, H, W, engine, gc, dc):
        X = np.random.rand(N, C, H, W).astype(np.float32)

        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(2,), engine=engine)


        def ref_flipud(X):
            return [np.flip(X, 2)]

        self.assertReferenceChecks(
            device_option=core.DeviceOption(caffe2_pb2.CPU, 0), # CPU only by now
            op=op,
            inputs=[X],
            reference=ref_flipud,
        )

    @given(N=st.sampled_from([2,3]),
           C=st.sampled_from([3,5]),
           H=st.sampled_from([5,8]),
           W=st.sampled_from([8,13]),
           engine=st.sampled_from([None]), # CPU only by now
           **hu.gcs)
    def test_flip3(self, N, C, H, W, engine, gc, dc):
        X = np.random.rand(N, C, H, W).astype(np.float32)

        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(2,3), engine=engine)


        def ref_flipud(X):
            return [np.flip(np.flip(X, 3), 2)]

        self.assertReferenceChecks(
            device_option=core.DeviceOption(caffe2_pb2.CPU, 0), # CPU only by now
            op=op,
            inputs=[X],
            reference=ref_flipud,
        )

    @given(N=st.sampled_from([2,3]),
           C=st.sampled_from([3,5]),
           H=st.sampled_from([5,8]),
           W=st.sampled_from([8,13]),
           engine=st.sampled_from([None]), # CPU only by now
           **hu.gcs)
    def test_flip4(self, N, C, H, W, engine, gc, dc):
        X = np.random.rand(N, C, H, W).astype(np.float32)

        op = core.CreateOperator("Flip", ["X"], ["Y"], axes=(0,1,2), engine=engine)


        def ref_flipud(X):
            return [np.flip(np.flip(np.flip(X, 2), 1), 0)]

        self.assertReferenceChecks(
            device_option=core.DeviceOption(caffe2_pb2.CPU, 0), # CPU only by now
            op=op,
            inputs=[X],
            reference=ref_flipud,
        )

if __name__ == "__main__":
    unittest.main()
