from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestFcOperator(hu.HypothesisTestCase):

    @given(n=st.integers(1, 5), m=st.integers(0, 5),
           k=st.integers(1, 5),
           multi_dim=st.sampled_from([True, False]),
           **hu.gcs)
    def test_fc(self, n, m, k, multi_dim, gc, dc):
        X = np.random.rand(m, k).astype(np.float32) - 0.5
        if multi_dim:
            W = np.random.rand(n, k, 1, 1).astype(np.float32) - 0.5
        else:
            W = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        def fc_op(X, W, b):
            return [np.dot(X, W.reshape(n, k).transpose()) + b.reshape(n)]

        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            'out'
        )

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W, b],
            reference=fc_op,
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, W, b], [0])
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, W, b], 0, [0])
        # Gradient check wrt W
        self.assertGradientChecks(gc, op, [X, W, b], 1, [0])
        # Gradient check wrt b
        self.assertGradientChecks(gc, op, [X, W, b], 2, [0])

    @given(n=st.integers(1, 5),
           m=st.integers(1, 5),
           k=st.integers(1, 5),
           tensor_core=st.booleans(),
           **hu.gcs_gpu_only)
    def test_fc_fp16(self, n, m, k, tensor_core, gc, dc):
        # Need to ensure TensorCore kernels can be called
        if tensor_core:
            m *= 8
            k *= 8
            n *= 8

        X = np.random.rand(m, k).astype(np.float32) - 0.5
        W = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5

        def fc_op(X, W, b):
            return [np.dot(X, W.transpose()) + b]

        engine = 'TENSORCORE' if tensor_core else ''
        X = X.astype(np.float16)
        W = W.astype(np.float16)
        b = b.astype(np.float16)
        op = core.CreateOperator(
            'FC',
            ['X', 'W', 'b'],
            'out',
            engine=engine
        )

        threshold = 5e-1
        stepsize = 1.

        # Check against numpy reference
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, W, b],
            reference=fc_op,
            threshold=threshold
        )
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, [X, W, b], [0], threshold=threshold)
        # Gradient check wrt X
        self.assertGradientChecks(gc, op, [X, W, b], 0, [0], threshold=threshold, stepsize=stepsize)
        # Gradient check wrt W
        self.assertGradientChecks(gc, op, [X, W, b], 1, [0], threshold=threshold, stepsize=stepsize)
        # Gradient check wrt b
        self.assertGradientChecks(gc, op, [X, W, b], 2, [0], threshold=threshold, stepsize=stepsize)


if __name__ == "__main__":
    import unittest
    unittest.main()
