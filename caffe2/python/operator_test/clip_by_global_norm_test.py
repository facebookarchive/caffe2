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

# Copyright (c) 2018, NVIDIA CORPORATION, All rights reserved
# Distributed under 2-clause BSD license; see accompanying LICENSE file
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import math

from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu


class TestClipByGlobalNorm(hu.HypothesisTestCase):
    @staticmethod
    def _dtype_conversion(X, dtype, gc, dc):
        """ClipByGradientNormOp only supports fp16 with CUDA."""
        if dtype == np.float16:
            assume(gc.device_type == caffe2_pb2.CUDA)
            dc = [d for d in dc if d.device_type == caffe2_pb2.CUDA]
            X = [x.astype(dtype) for x in X]
        return X, dc


    @given(X=hu.tensors(max_n=10, varying_shape=True),
           clip_norm=st.floats(min_value=1.0, max_value=10.0),
           scale=st.floats(min_value=1.0/1024, max_value=2.0),
           inplace=st.booleans(),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    @settings(max_examples=30)
    def test_clip_by_global_norm(self, X, clip_norm, scale, inplace, dtype, gc, dc):

        def clip_by_global_norm_ref(*inputs):
            X_scaled = [np.dot(x, scale) for x in inputs]
            norm_sqr = lambda x: np.sum(np.square(x))
            global_norm = math.sqrt(sum([norm_sqr(x.astype(np.float32)) for x in X_scaled])) 
            clip_ratio = clip_norm / max(global_norm, clip_norm)
            X_clipped = [np.dot(x.astype(np.float32), clip_ratio) for x in X_scaled]
            return X_clipped

        inputs = ["X_{}".format(i) for i in range(len(X))]
        outputs = ["Y_{}".format(i) for i in range(len(X))] \
            if not inplace else inputs
        op = core.CreateOperator(
            "ClipByGlobalNorm",
            inputs, outputs,
            clip_norm=clip_norm,
            scale=scale)

        X, dc = self._dtype_conversion(X, dtype, gc, dc)
        atol=1e-8 if dtype==np.float32 else 1e-3
        self.assertReferenceChecks(gc, op, X, clip_by_global_norm_ref, atol=atol)
        # Check over multiple devices
        self.assertDeviceChecks(dc, op, X, [i for i in range(len(X))])


if __name__ == "__main__":
    import unittest
    unittest.main()
