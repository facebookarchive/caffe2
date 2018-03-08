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
import itertools as it
import functools
import caffe2.proto.caffe2_pb2 as pb2

import unittest


class TestReduceSum(hu.HypothesisTestCase):
    @given(
        d0=st.integers(1, 5),
        d1=st.integers(1, 5),
        d2=st.integers(1, 5),
        d3=st.integers(1, 5),
        keepdims=st.integers(0, 1),
        seed=st.integers(0, 2**32 - 1),
        **hu.gcs_cpu_only)
    def test_reduce_sum(self, d0, d1, d2, d3, keepdims, seed, gc, dc):
        np.random.seed(seed)

        def reduce_sum_ref(data, axis, keepdims):
            return [np.sum(data, axis=axis, keepdims=keepdims)]

        # all combinations of reduced axes
        for n in range(1, 5):
            for axes in it.combinations(range(4), n):
                data = np.random.randn(d0, d1, d2, d3).astype(np.float32)

                args = []
                arg = pb2.Argument()
                arg.name = "axes"
                arg.ints.extend(axes)
                args.append(arg)

                arg = pb2.Argument()
                arg.name = "keepdims"
                arg.i = keepdims
                args.append(arg)

                op = core.CreateOperator(
                    "ReduceSum",
                    ["data"],
                    ["Y"],
                    arg=args,
                )

                self.assertReferenceChecks(gc, op, [data],
                                           functools.partial(
                                               reduce_sum_ref,
                                               axis=axes,
                                               keepdims=keepdims))


if __name__ == "__main__":
    unittest.main()
