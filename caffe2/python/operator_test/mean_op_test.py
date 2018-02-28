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

from caffe2.python import core
from hypothesis import given
import hypothesis.strategies as st
import caffe2.python.hypothesis_test_util as hu
import numpy as np

import unittest


class TestMean(hu.HypothesisTestCase):
    @given(
        inputs=hu.tensors(n=2, min_dim=2, max_dim=2),
        engine=st.sampled_from(["", "CUDNN"]),
        **hu.gcs)
    def test_mean(self, inputs, gc, dc, engine):
        X, Y = inputs

        def mean_ref(X, Y):
            return np.mean([X, Y], axis=0)

        op = core.CreateOperator("Mean", ["X", "Y"], ["Result"], engine=engine)
        self.ws.create_blob("X").feed(X)
        self.ws.create_blob("Y").feed(Y)
        self.ws.run(op)

        op_output = self.ws.blobs["Result"].fetch()
        ref_output = mean_ref(X, Y)
        np.testing.assert_allclose(op_output, ref_output, atol=1e-4, rtol=1e-4)

        self.assertDeviceChecks(dc, op, [X, Y], [0])


if __name__ == "__main__":
    unittest.main()
