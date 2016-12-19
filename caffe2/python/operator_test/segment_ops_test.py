from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from functools import partial
from hypothesis import given

import caffe2.python.hypothesis_test_util as hu
import numpy as np


class TesterBase:
    def segment_reduce_op(self, data, segment_ids, reducer, indices=None):
        segments = self.split(data, segment_ids, indices)
        output = np.zeros((len(segments), ) + data.shape[1:])
        for i, segment in enumerate(segments):
            if len(segment) > 0:
                output[i] = reducer(segment)
            else:
                output[i] = 0.0
        return output

    def segment_reduce_grad_op(
        self,
        data,
        segment_ids,
        reducer_grad,
        grad_out,
        output,
        indices=None
    ):
        segments = self.split(data, segment_ids, indices)
        segment_grads = [
            reducer_grad(grad_out[i], [output[i]], [segment])
            for i, segment in enumerate(segments)
        ]
        return self.unsplit(data.shape[1:], segment_grads, segment_ids)

    def test(self, prefix, input_strategy, refs):
        tester = self

        @given(X=input_strategy, **hu.gcs_cpu_only)
        def test_segment_ops(self, X, gc, dc):
            for op_name, ref, grad_ref in refs:
                inputs = ['input%d' % i for i in range(0, len(X))]
                op = core.CreateOperator(prefix + op_name, inputs, ['output'])

                def seg_reduce(data, *args):
                    indices, segments = (
                        args if len(args) == 2 else (None, args[0])
                    )
                    out = tester.segment_reduce_op(
                        data=data,
                        segment_ids=segments,
                        indices=indices,
                        reducer=ref
                    )
                    return (out, )

                def seg_reduce_grad(grad_out, outputs, inputs):
                    data = inputs[0]
                    args = inputs[1:]
                    indices, segments = (
                        args if len(args) == 2 else (None, args[0])
                    )
                    # grad r.t. data
                    grad_val = tester.segment_reduce_grad_op(
                        data, segments, grad_ref, grad_out, outputs[0], indices
                    )
                    # if sparse, include indices along with data gradient
                    data_grad_slice = (
                        (grad_val, indices) if indices is not None else grad_val
                    )
                    # other inputs don't have gradient
                    return (data_grad_slice, ) + (None, ) * (len(inputs) - 1)

                self.assertReferenceChecks(
                    device_option=gc,
                    op=op,
                    inputs=X,
                    reference=seg_reduce,
                    output_to_grad='output',
                    grad_reference=seg_reduce_grad,
                )

        return test_segment_ops


class SegmentsTester(TesterBase):
    def split(self, data, segment_ids, indices=None):
        """
        Given:
          data[M1 x M2 x ... x Md]
                          the input data
          indices[N]      the index of each entry of segment_ids into data,
                          where 0 <= index[i] < M1,
                          with default indices=[0,1,...N]
          segment_ids[N]  the segment_id for each entry of indices,

        returns K outputs, each one containing data entries corresponding
        to one of the segments present in `segment_ids`.
        """
        if segment_ids.size == 0:
            return []
        K = max(segment_ids) + 1
        outputs = [
            np.zeros(
                (np.count_nonzero(segment_ids == seg_id), ) + data.shape[1:],
                dtype=data.dtype
            ) for seg_id in range(0, K)
        ]
        counts = np.zeros(K, dtype=int)
        for i, seg_id in enumerate(segment_ids):
            data_idx = i if indices is None else indices[i]
            outputs[seg_id][counts[seg_id]] = data[data_idx]
            counts[seg_id] += 1
        return outputs

    def unsplit(self, extra_shape, inputs, segment_ids):
        """ Inverse operation to `split`, with indices=None """
        output = np.zeros((len(segment_ids), ) + extra_shape)
        if len(segment_ids) == 0:
            return output
        K = max(segment_ids) + 1
        counts = np.zeros(K, dtype=int)
        for i, seg_id in enumerate(segment_ids):
            output[i] = inputs[seg_id][counts[seg_id]]
            counts[seg_id] += 1
        return output


class LengthsTester(TesterBase):
    def split(self, data, lengths, indices=None):
        K = len(lengths)
        outputs = [
            np.zeros((lengths[seg_id], ) + data.shape[1:],
                     dtype=data.dtype) for seg_id in range(0, K)
        ]
        start = 0
        for i in range(0, K):
            for j in range(0, lengths[i]):
                data_index = start + j
                if indices is not None:
                    data_index = indices[data_index]
                outputs[i][j] = data[data_index]
            start += lengths[i]
        return outputs

    def unsplit(self, extra_shape, inputs, lengths):
        N = sum(lengths)
        output = np.zeros((N, ) + extra_shape)
        K = len(lengths)
        assert len(inputs) == K
        current = 0
        for i in range(0, K):
            for j in range(0, lengths[i]):
                output[current] = inputs[i][j]
                current += 1
        return output


def sum_grad(grad_out, outputs, inputs):
    return np.repeat(
        np.expand_dims(grad_out, axis=0),
        inputs[0].shape[0],
        axis=0
    )


def logsumexp(x):
    return np.log(np.sum(np.exp(x), axis=0))


def logsumexp_grad(grad_out, outputs, inputs):
    sum_exps = np.sum(np.exp(inputs[0]), axis=0)
    return np.repeat(
        np.expand_dims(grad_out / sum_exps, 0),
        inputs[0].shape[0],
        axis=0
    ) * np.exp(inputs[0])


def logmeanexp(x):
    return np.log(np.mean(np.exp(x), axis=0))


def mean(x):
    return np.mean(x, axis=0)


def mean_grad(grad_out, outputs, inputs):
    return np.repeat(
        np.expand_dims(grad_out / inputs[0].shape[0], 0),
        inputs[0].shape[0],
        axis=0
    )


def max_fwd(x):
    return np.amax(x, axis=0)


def max_grad(grad_out, outputs, inputs):
    flat_inputs = inputs[0].flatten()
    flat_outputs = np.array(outputs[0]).flatten()
    flat_grad_in = np.zeros(flat_inputs.shape)
    flat_grad_out = np.array(grad_out).flatten()
    blocks = inputs[0].shape[0]
    block_size = flat_inputs.shape[0] // blocks

    for i in range(block_size):
        out_grad = flat_grad_out[i]
        out = flat_outputs[i]
        for j in range(blocks):
            idx = j * block_size + i
            if out == flat_inputs[idx]:
                flat_grad_in[idx] = out_grad
                break

    return np.resize(flat_grad_in, inputs[0].shape)


REFERENCES_ALL = [
    ('Sum', partial(np.sum, axis=0), sum_grad),
    ('Mean', partial(np.mean, axis=0), mean_grad),
]

REFERENCES_SORTED = [
    ('RangeSum', partial(np.sum, axis=0), sum_grad),
    ('RangeLogSumExp', logsumexp, logsumexp_grad),
    # gradient is the same as sum
    ('RangeLogMeanExp', logmeanexp, logsumexp_grad),
    ('RangeMean', mean, mean_grad),
    ('RangeMax', max_fwd, max_grad),
]


class TestSegmentOps(hu.HypothesisTestCase):
    def test_sorted_segment_ops(self):
        SegmentsTester().test(
            'SortedSegment',
            hu.segmented_tensor(
                dtype=np.float32,
                is_sorted=True,
                allow_empty=True
            ),
            REFERENCES_ALL + REFERENCES_SORTED
        )(self)

    def test_unsorted_segment_ops(self):
        SegmentsTester().test(
            'UnsortedSegment',
            hu.segmented_tensor(
                dtype=np.float32,
                is_sorted=False,
                allow_empty=True
            ),
            REFERENCES_ALL
        )(self)

    def test_sparse_sorted_segment_ops(self):
        SegmentsTester().test(
            'SparseSortedSegment',
            hu.sparse_segmented_tensor(
                dtype=np.float32,
                is_sorted=True,
                allow_empty=True
            ),
            REFERENCES_ALL
        )(self)

    def test_sparse_unsorted_segment_ops(self):
        SegmentsTester().test(
            'SparseUnsortedSegment',
            hu.sparse_segmented_tensor(
                dtype=np.float32,
                is_sorted=False,
                allow_empty=True
            ),
            REFERENCES_ALL
        )(self)

    def test_lengths_ops(self):
        LengthsTester().test(
            'Lengths',
            hu.lengths_tensor(
                dtype=np.float32,
                min_value=1,
                max_value=10,
                allow_empty=True
            ),
            REFERENCES_ALL
        )(self)

    def test_sparse_lengths_ops(self):
        LengthsTester().test(
            'SparseLengths',
            hu.sparse_lengths_tensor(
                dtype=np.float32,
                min_value=1,
                max_value=10,
                allow_empty=True
            ),
            REFERENCES_ALL
        )(self)

if __name__ == "__main__":
    import unittest
    unittest.main()
