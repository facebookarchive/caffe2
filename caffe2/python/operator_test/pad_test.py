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

import numpy as np
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from caffe2.proto import caffe2_pb2
from hypothesis import assume, given


class TestPad(hu.HypothesisTestCase):
    @given(pad_t=st.integers(-2, 0),
           pad_l=st.integers(-2, 0),
           pad_b=st.integers(-2, 0),
           pad_r=st.integers(-2, 0),
           mode=st.sampled_from(["constant", "reflect", "edge"]),
           size_w=st.integers(5, 8),
           size_h=st.integers(5, 8),
           size_c=st.integers(1, 4),
           size_n=st.integers(1, 4),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_crop(self,
                  pad_t, pad_l, pad_b, pad_r,
                  mode,
                  size_w, size_h, size_c, size_n,
                  dtype,
                  gc, dc):
        if dtype == np.float16:
            # fp16 is only supported with CUDA
            assume(gc.device_type == caffe2_pb2.CUDA)
            dc = [d for d in dc if d.device_type == caffe2_pb2.CUDA]

        op = core.CreateOperator(
            "PadImage",
            ["X"],
            ["Y"],
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
            mode=mode,
        )
        X = np.random.rand(
            size_n, size_c, size_h, size_w).astype(dtype)

        def ref(X):
            return (X[:, :, -pad_t:pad_b or None, -pad_l:pad_r or None],)

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        if dtype != np.float16:  # not ready yet
            self.assertGradientChecks(gc, op, [X], 0, [0])

    @given(pad_t=st.integers(0, 4),
           pad_l=st.integers(0, 4),
           pad_b=st.integers(0, 4),
           pad_r=st.integers(0, 4),
           mode=st.sampled_from(["constant", "reflect", "edge"]),
           size_w=st.integers(5, 8),
           size_h=st.integers(5, 8),
           size_c=st.integers(1, 4),
           size_n=st.integers(1, 4),
           dtype=st.sampled_from([np.float32, np.float16]),
           **hu.gcs)
    def test_pads(self,
                  pad_t, pad_l, pad_b, pad_r,
                  mode,
                  size_w, size_h, size_c, size_n,
                  dtype,
                  gc, dc):
        if dtype == np.float16:
            # fp16 is only supported with CUDA
            assume(gc.device_type == caffe2_pb2.CUDA)
            dc = [d for d in dc if d.device_type == caffe2_pb2.CUDA]

        op = core.CreateOperator(
            "PadImage",
            ["X"],
            ["Y"],
            pad_t=pad_t,
            pad_l=pad_l,
            pad_b=pad_b,
            pad_r=pad_r,
            mode=mode,
        )
        X = np.random.rand(
            size_n, size_c, size_h, size_w).astype(dtype)

        def ref(X):
            return (np.pad(X, [(0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)], mode=mode),)

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        if dtype != np.float16:  # not ready yet
            self.assertGradientChecks(gc, op, [X], 0, [0])


if __name__ == "__main__":
    unittest.main()
