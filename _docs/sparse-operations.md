---
docid: sparse-operations
title: Sparse Operations
layout: docs
permalink: /docs/sparse-operations.html
---

Caffe2 provides support for representing sparse features and performing corresponding operations on segments of tensors.

## Representations

Segmented tensors naturally arise when handling a batch of data of varying dimension. Consider having a batch of training example with a sparse feature, e.g. list of PAGE_IDs. It’s natural to represent those features as a list of int64s for each example. For example, let’s say there are 3 examples in a batch with the following feature values:

```
ex1 = {1,2,3}
ex2 = {2,4,6,7}
ex3 = {3,6}
batch = {ex1, ex2, ex3}
```

There are several possible choices for representing a batch of several example in a single tensor: Values and lengths represents a batch as two tensors - one holding concatenated feature values and another having the number of feature values for each example. For matrices it roughly corresponds to CSR (compressed sparse row) format but with lengths instead of offsets.

```
values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
#          \_____/  \________/  \__/
lengths =    [3,        4,       2]
```

Segment IDs also concatenates values together but has the second vector of the same length as the first dimension of the main tensor. Each element of the `segment_ids` maps corresponding slice of the main tensor to one of the examples (called segments in this case). Usually segment ids are sorted:

```
values      = [1, 2, 3, 2, 4, 6, 7, 3, 6]
segment_ids = [0, 0, 0, 1, 1, 1, 1, 2, 2]
```

However they can be arbitrary ordered which is useful in some use cases. This representation is called unsorted segment IDS, e.g.:

```
values      = [4, 1, 3, 6, 3, 2, 7, 2, 6]
segment_ids = [1, 0, 2, 1, 0, 1, 1, 0, 2]
```

Padded representation stacks examples along the first dimension (e.g. rows in a matrix) and uses a filler value to make them of equal length. Assuming `-1` as a filler the above example looks as follows:

```
padded = [[1,  2,  3, -1],
          [2,  4,  6,  7],
          [3,  6, -1, -1]]
```
Sparse tensor comes from interpreting values as indices in some big sparse matrix. It’s usually a very inefficient representation for practical purposes, but often is a semantical meaning of how features are used. In above example:

```
sparse_matrix = [[0, 1, 1, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 1, 0, 1, 1],
                 [0, 0, 0, 1, 0, 0, 1, 0]]
#                -------------------------
# ids             0  1  2  3  4  5  6  7
```

Caffe2 uses values with lengths and segment ids representations and provides necessary ops to convert between two.

### More complex examples

Sparse features with values get naturally represented in either way by separating feature ids and corresponding values.

```
# batch has 2 examples with a single feature of PAGE_ID and corresponding score
ex1 = {1: 0.4, 3: 0.7}
ex2 = {2: 0.5, 3: 0.5, 5: 0.1}
battch = {ex1, ex2}
# values with lengths
values      = [  1,   3,   2,   3,   5]
scores      = [0.4, 0.7, 0.5, 0.5, 0.1]
lengths     = [2, 3]
# or alternatively with segment_ids
values      = [  1,   3,   2,   3,   5]
scores      = [0.4, 0.7, 0.5, 0.5, 0.1]
segment_ids = [  0,   0,   1,   1,   1]
```

Segmented representations can be nested. One often use case is when each example has features of multiple types, each in turn being a list of ids (`LIST<MAP<INT, LIST<INT>>>` in other words).

```
# Assume feature types are defined somewhere as enum {PAGE_ID=1, APP_ID=2, POST_ID=3}
ex1 = {1: {10, 11}, 3: {101}}
ex2 = {1: {11}, 2: {50}, 3: {102, 103}}
batch = {ex1, ex2}# values with lengths
values          = [10, 11, 101, 11, 50, 102, 103]
#                  \____/  \_/  \_/ \_/ \______/
values_lengths  = [   2,    1,   1,  1,    2]
keys            = [   1,    3,   1,  2,    3]
#                  \_________/  \__________/
example_lengths = [     2,            3]
```

## Operators overview

Caffe2 provides some of the utility ops to manipulate above representations and a bunch of operators taking sparse representations, mostly various reductions.

### Converting between representations

* `LengthsToSegmentIds` takes a vector of lengths and produces a sorted segment vector that has the length of sum of the input vector and each segment id is replicated corresponding number of times. Simple implementation in python would be `[id for _ in range(x) for id,x in enumerate(lengths)]`
* `SegmentIdsToLengthsOp` is the reverse operation going from sorted segment ids back to the length vector.

### Basic sparse operations

The most basic operation acting on sparse ids is `Gather` which effectively performs embedding lookup by pulling slices of data tensor referenced by indices. Corresponding update operation is `ScatterAssign` that replaces slices of the data tensor as referenced by indices.

### Reduction operators overview

Most of the segment-based operations combined two parts: choice of representation and reduction function. Since it’s usually more computationally efficient to fuse both parts together, there is almost full cartesian product between retrieval options and reduction functions. The name of the operator is usually a concatenation of two parts, e.g. `SortedSegment + Sum = SortedSegmentSum`. Examples above were describing one-dimensional input tensors. In case the inputs are multi-dimensional, all segmentation acts on the first dimension only. For example, for `SortedSegmentSum`:

```
data = [[1, 4],
        [3, 2],
        [8, 1],
        [9, 4],
        [5, 8]])
segment_ids = [0, 0, 0, 1, 1]
SortedSegmentSum([data, segment_ids]) ->
  [[  9.   7.]
   [ 14.  12.]]
```

### Sorted segment reduction ops

Group of `SortedSegment*` ops take an input tensor and segment ids vector mapping each slice of the first dimension to a segment and perform particular aggregation operation. The `segment_ids` tensor should be the size of the first dimension, `d0`, with consecutive IDs in the range 0 to k, where k<d0. In particular, a segmentation of a matrix tensor is a mapping of rows to segments. The first dimension of the output tensor is going to have dimension of k+1. Reduction ops include:

* `SortedSegmentSum` - element-wise addition within segment
* `SortedSegmentWeightedSum` - as above, but applies a scalar to each of the slices

Some of the reduction functions act more efficiently on sorted segments and thus have special implementations with Range in name. Thus the only representation supported is sorted segments. Ops include:

* `SortedSegmentRangeLogSumExp` - logarithm of sums of exponents for each segment
* `SortedSegmentRangeMean` - average of values for each segment

### Unsorted segment reduction ops

`UnsortedSegment*` ops have similar interface to `SortedSegment*` ones but don’t require the segment ids to appear in the increasing order. As an optimization, the total number of segments can be passed as`num_segments` argument. Otherwise, it would be determined as `max(segment_ids)+1`. The actual ops are just corresponding equivalents for their sorted versions:

* `UnsortedSegmentSum`
* `UnsortedSegmentWeightedSum`

### Fused sparse reduction ops

It’s fairly common to combine sparse table lookup with further reduction, for example averaging the embeddings for multiple sparse features. In Caffe2 it can be implemented by combination of `Gather` and `SortedSegment*`. We provide a fused operator that combines them and supplies more efficient implementation. Each of the `SortedSegment*` and `UnsortedSegment*` ops have a fused equivalent with Sparse prefix. For example, `Gather + SortedSegmentSum` can be replaced with `SparseSortedSegmentSum`.

### Other fused sparse operations

There is a number of operators fuse `Gather` or `ScatterUpdate` with an actual operator for efficiency purposes. That’s the case for update operators used in optimizers, e.g. `SparseFtrl`, `SparseAdagrad`, `ScatterWeightedSum`.
