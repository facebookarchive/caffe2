---
docid: distributed-training
title: Distributed Training
layout: docs
permalink: /docs/distributed-training.html
---
One of Caffe2's most significant features is easy, built-in distributed training. This means that you can very quickly scale up or down without refactoring your design.

## Scaling Out with Multi-GPU Machines

Training huge datasets like ImageNet's 14 million images is possible within minutes or even seconds. Thinking bigger, this means even now you'll be able to process tens of thousands of images per second and continue to scale this upward. Recent results for training ImageNet with Caffe2 are shown below.

![chart images per second](/static/images/infograph-chart-ips.png)

These speeds were tested with [NVIDIA Tesla P100 cards](NVIDIA Tesla P100 cards) for ResNet-50. You can add GPUs in a single machine, or scale horizontally with additional machines, each with one or more GPUs. As seen in the chart below, scaling is slightly sublinear due to some small overhead.

![chart speedup](/static/images/infograph-chart-speedup.png)

For a deeper dive and examples of distributed training, check out [SynchronousSGD](sync-sgd), where you'll be taught the programming principles for using Caffe's [data_parallel_model](https://github.com/caffe2/caffe2/blob/master/caffe2/python/data_parallel_model.py).

## Under the Hood

* [Gloo](https://github.com/facebookincubator/gloo): Caffe2 leverages, Gloo, a communications library for multi-machine training.
* [NCCL](https://github.com/nvidia/nccl): Caffe2 also utilize's NVIDIA's NCCL for multi-GPU communications.
* [Redis](https://redis.io/) To facilitate management of nodes in distributed training, Caffe can use a simple NFS share between nodes, or you can provide a Redis server to handle the nodes' communications.

For an example of distributed training with Caffe2 you can run the [resnet50_trainer](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py) script on a single GPU machine. The defaults assume that you've already loaded the training data into a [lmdb database](https://symas.com/offerings/lightning-memory-mapped-database/), but you have the additional option of using [LevelDB](https://github.com/google/leveldb). A guide for using the script is below.

## Try Distributed Training

We're assuming that you've successfully build Caffe2 and that you have a system with at least one GPU, but preferably more to test out the distributed features.

First get yourself an training image database ready in [lmdb database](https://symas.com/offerings/lightning-memory-mapped-database/) or [LevelDB](https://github.com/google/leveldb) format. You can browse and download a variety of [datasets here](datasets).

[resnet50_trainer.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py) script default output:

### Usage

```
usage: resnet50_trainer.py [-h] --train_data TRAIN_DATA
                           [--test_data TEST_DATA] [--db_type DB_TYPE]
                           [--gpus GPUS] [--num_gpus NUM_GPUS]
                           [--num_channels NUM_CHANNELS]
                           [--image_size IMAGE_SIZE] [--num_labels NUM_LABELS]
                           [--batch_size BATCH_SIZE] [--epoch_size EPOCH_SIZE]
                           [--num_epochs NUM_EPOCHS]
                           [--base_learning_rate BASE_LEARNING_RATE]
                           [--weight_decay WEIGHT_DECAY]
                           [--num_shards NUM_SHARDS] [--shard_id SHARD_ID]
                           [--run_id RUN_ID] [--redis_host REDIS_HOST]
                           [--redis_port REDIS_PORT]
                           [--file_store_path FILE_STORE_PATH]
```

|Required||
|-------|--------|
| `--train_data` | the path to the database of training data |

|Optional||
|-------|--------|
| `--test_data` | the path to the database of test data |
| `--db_type` | either `lmdb` or `leveldb`, defaults to `lmdb` |
| `--gpus`| a list of GPU IDs, where 0 would be the first GPU device #, comma separated |
| `--num_gpus` | an integer for the total number of GPUs; alternative to using a list with `gpus` |
| `--num_channels` | number of color channels, defaults to 3 |
| `--image_size` | the height or width in pixels of the input images, assumes they're square, defaults to 227, might not handle small sizes |
| `--num_labels` | number of labels, defaults to 1000 |
| `--batch_size` | batch size, total over all GPUs, defaults to 32, expand as you increase GPUs |
| `--epoch_size` | number of images per [epoch](https://deeplearning4j.org/glossary#epoch-vs-iteration), defaults to 1.5MM (1500000), definitely change this |
| `--num_epochs` | number of [epochs](https://deeplearning4j.org/glossary#epoch-vs-iteration) |
| `--base_learning_rate` | initial learning rate, defaults to 0.1 (based on 256 global batch size) |
| `--weight_decay` | weight decay (L2 regularization) |
| `--num_shards` | number of machines in a distributed run, defaults to 1 |
| `--shard_id` | shard/node id, defaults to 0, next node would be 1, and so forth |
| `--run_id RUN_ID` | unique run identifier, e.g. uuid |
| `--redis_host` | host of Redis server (for rendezvous) |
| `--redis_port` | Redis port for rendezvous |
| `--file_store_path` | alternative to Redis, (NFS) path to shared directory to use for rendezvous temp files to coordinate between each shard/node |

Arguments for preliminary testing:

1. `--train_data` <path to db> (required)
2. `--db_type` <lmbd or leveldb> (default=lmdb)
3. `--num_gpus` <#> (use this instead of listing out each one with `--gpus`)
4. `--batch_size` <multiples of 32> (default=32)
5. `--test_data` <path to db> (optional)

The only required parameter is the training database. You can try that out first with no other parameters if you have your training set already in lmdb.

```
python resnet50_trainer.py --train_data <location of lmdb training database>
```

Using LevelDB:

```
python resnet50_trainer.py --train_data <location of leveldb training database> --db_type leveldb
```

The script uses a default batch size of 32. When using 2 GPUs you want to increase the batch size according to the number of GPUs, so that you're using as much of the memory on the GPU as possible. In the case of using 2 GPUs as in the example below, we double the batch size to 64:

```
python resnet50_trainer.py --train_data <location of lmdb training database> --num_gpus 2 --batch_size 64
```

You will notice that when you add the second GPU and double the batch size the number of iterations per epoch is half.

Using `nvidia-smi` you can examine the GPUs' current status and see if you're properly maxing it out in each run. Try running `watch -n1 nvidia-smi` to continuously report the status while you run different experiments.

If you add the `--test_data` parameter you will get occasional test runs intermingled which can provide a nice metric on how well the neural network is doing at that time. It'll give you accuracy numbers and help you assess convergence.

### Logging

As you run the script and training progresses, you will notice log files are deposited in the same folder. The naming convention will give you an idea of what that particular run's parameters were. For example:

```
resnet50_gpu2_b64_L1000_lr0.10_v2_20170411_141617.log
```

You can infer from this filename that the parameters were: `--gpus 2`, `--batch_size 64`, `num_labels 1000`, `--base_learning_rate 0.10`, followed by a timestamp.

When opening the log file you will find the parameters used for that run, a header to let you know what the comma separated values represent, and finally the log data.

The list of values recorded in the log:
time_spent,cumulative_time_spent,input_count,cumulative_input_count,cumulative_batch_count,inputs_per_sec,accuracy,epoch,learning_rate,loss,test_accuracy

`test_accuracy` will be set to -1 if you don't use test data, otherwise it will populate with an accuracy number.

### Conclusion

There are many other parameters you can tinker with using this script. We look forward to hearing from you about your experiments and to see your models appear on [Caffe2's Model Zoo](zoo)!
