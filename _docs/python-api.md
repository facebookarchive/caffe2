---
docid: python-api
title: Python API
layout: docs
permalink: /docs/python-api.html
---
# namespace `activation_ops_test` {#namespaceactivation__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`activation_ops_test::TestActivations`](#classactivation__ops__test_1_1_test_activations)    |# class `activation_ops_test::TestActivations` {#classactivation__ops__test_1_1_test_activations}

```
class activation_ops_test::TestActivations
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_elu(self,`[`X`](#classactivation__ops__test_1_1_test_activations_1a63b6ee0db3af0feeb2abc9045c80f014)`,`[`alpha`](#classactivation__ops__test_1_1_test_activations_1ac20799e6d6e0fb79c099b20137f69b3d)`,`[`inplace`](#classactivation__ops__test_1_1_test_activations_1aeada65ac3279af7ea1f96ad588306fe2)`,gc,dc)` |
`public def test_prelu(self,`[`X`](#classactivation__ops__test_1_1_test_activations_1a63b6ee0db3af0feeb2abc9045c80f014)`,`[`alpha`](#classactivation__ops__test_1_1_test_activations_1ac20799e6d6e0fb79c099b20137f69b3d)`,`[`inplace`](#classactivation__ops__test_1_1_test_activations_1aeada65ac3279af7ea1f96ad588306fe2)`,`[`shared`](#classactivation__ops__test_1_1_test_activations_1a2889e01666a53ba47255094084906f01)`,`[`order`](#classactivation__ops__test_1_1_test_activations_1ac29060925698f858c666911c85a55d1e)`,gc,dc)` |
`public def test_leaky_relu(self,`[`X`](#classactivation__ops__test_1_1_test_activations_1a63b6ee0db3af0feeb2abc9045c80f014)`,`[`alpha`](#classactivation__ops__test_1_1_test_activations_1ac20799e6d6e0fb79c099b20137f69b3d)`,`[`inplace`](#classactivation__ops__test_1_1_test_activations_1aeada65ac3279af7ea1f96ad588306fe2)`,gc,dc)` |

## Members

#### `public def test_elu(self,`[`X`](#classactivation__ops__test_1_1_test_activations_1a63b6ee0db3af0feeb2abc9045c80f014)`,`[`alpha`](#classactivation__ops__test_1_1_test_activations_1ac20799e6d6e0fb79c099b20137f69b3d)`,`[`inplace`](#classactivation__ops__test_1_1_test_activations_1aeada65ac3279af7ea1f96ad588306fe2)`,gc,dc)` {#classactivation__ops__test_1_1_test_activations_1a5b56552e7e01be01f4b26db19f412afa}





#### `public def test_prelu(self,`[`X`](#classactivation__ops__test_1_1_test_activations_1a63b6ee0db3af0feeb2abc9045c80f014)`,`[`alpha`](#classactivation__ops__test_1_1_test_activations_1ac20799e6d6e0fb79c099b20137f69b3d)`,`[`inplace`](#classactivation__ops__test_1_1_test_activations_1aeada65ac3279af7ea1f96ad588306fe2)`,`[`shared`](#classactivation__ops__test_1_1_test_activations_1a2889e01666a53ba47255094084906f01)`,`[`order`](#classactivation__ops__test_1_1_test_activations_1ac29060925698f858c666911c85a55d1e)`,gc,dc)` {#classactivation__ops__test_1_1_test_activations_1a7b72aef69b684eab935efee438cb3cd6}





#### `public def test_leaky_relu(self,`[`X`](#classactivation__ops__test_1_1_test_activations_1a63b6ee0db3af0feeb2abc9045c80f014)`,`[`alpha`](#classactivation__ops__test_1_1_test_activations_1ac20799e6d6e0fb79c099b20137f69b3d)`,`[`inplace`](#classactivation__ops__test_1_1_test_activations_1aeada65ac3279af7ea1f96ad588306fe2)`,gc,dc)` {#classactivation__ops__test_1_1_test_activations_1a2c35be5db20dd433d300ab9186b6eb49}





# namespace `atomic_ops_test` {#namespaceatomic__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`atomic_ops_test::TestAtomicOps`](#classatomic__ops__test_1_1_test_atomic_ops)    |# class `atomic_ops_test::TestAtomicOps` {#classatomic__ops__test_1_1_test_atomic_ops}

```
class atomic_ops_test::TestAtomicOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_atomic_ops(self)` |

## Members

#### `public def test_atomic_ops(self)` {#classatomic__ops__test_1_1_test_atomic_ops_1a1a0d33d3a94d5547d989c2894e053dc2}



Test that both countdown and checksum are update atomically by having
cowntdown count from 20k to 0 from parallel the workers and updating
the checksum to the value fetched. If operations are trully atomic,
each value from 1 to 20k should be fetched exactly once from the
countdown, and fed exactly once to the checksum, such that at the end
checksum must contain the exact value of sum[i=0..20000](i).

# namespace `attention` {#namespaceattention}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`attention::AttentionType`](#classattention_1_1_attention_type)    |
# class `attention::AttentionType` {#classattention_1_1_attention_type}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# namespace `caffe2::python` {#namespacecaffe2_1_1python}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`caffe2::python::BlobFeederBase`](#classcaffe2_1_1python_1_1_blob_feeder_base)    |
`class `[`caffe2::python::BlobFetcherBase`](#classcaffe2_1_1python_1_1_blob_fetcher_base)    |
`class `[`caffe2::python::PythonGradientOp`](#classcaffe2_1_1python_1_1_python_gradient_op)    |
`class `[`caffe2::python::PythonOp`](#classcaffe2_1_1python_1_1_python_op)    |
`class `[`caffe2::python::PythonOpBase`](#classcaffe2_1_1python_1_1_python_op_base)    |
`class `[`caffe2::python::StringFetcher`](#classcaffe2_1_1python_1_1_string_fetcher)    |
`class `[`caffe2::python::TensorFeeder`](#classcaffe2_1_1python_1_1_tensor_feeder)    |
`class `[`caffe2::python::TensorFetcher`](#classcaffe2_1_1python_1_1_tensor_fetcher)    |
`struct `[`caffe2::python::GetPythonGradient`](#structcaffe2_1_1python_1_1_get_python_gradient)    |
# class `caffe2::python::BlobFeederBase` {#classcaffe2_1_1python_1_1_blob_feeder_base}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public virtual  ~BlobFeederBase()` |
`public void Feed(const DeviceOption & option,PyArrayObject * array,Blob * blob)` |

## Members

#### `public virtual  ~BlobFeederBase()` {#classcaffe2_1_1python_1_1_blob_feeder_base_1a3552391a25b52d7826101d0f5018b271}





#### `public void Feed(const DeviceOption & option,PyArrayObject * array,Blob * blob)` {#classcaffe2_1_1python_1_1_blob_feeder_base_1ade827da9dd851838c6f7a2b3af214764}





# class `caffe2::python::BlobFetcherBase` {#classcaffe2_1_1python_1_1_blob_fetcher_base}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public virtual  ~BlobFetcherBase()` |
`public pybind11::object Fetch(const Blob & blob)` |

## Members

#### `public virtual  ~BlobFetcherBase()` {#classcaffe2_1_1python_1_1_blob_fetcher_base_1a99f4d1b806aebbcd9d89a04702901015}





#### `public pybind11::object Fetch(const Blob & blob)` {#classcaffe2_1_1python_1_1_blob_fetcher_base_1a7f5cd9ef39edbf577a366bb49fb9aae2}





# class `caffe2::python::PythonGradientOp` {#classcaffe2_1_1python_1_1_python_gradient_op}

```
class caffe2::python::PythonGradientOp
  : public caffe2::python::PythonOpBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`protected virtual const `[`python_detail::Func`](#structcaffe2_1_1python_1_1python__detail_1_1_func)` & getFunc()` |

## Members

#### `protected virtual const `[`python_detail::Func`](#structcaffe2_1_1python_1_1python__detail_1_1_func)` & getFunc()` {#classcaffe2_1_1python_1_1_python_gradient_op_1abc53345b8a66ebd807b5bd35330fee53}





# class `caffe2::python::PythonOp` {#classcaffe2_1_1python_1_1_python_op}

```
class caffe2::python::PythonOp
  : public caffe2::python::PythonOpBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`protected virtual const `[`python_detail::Func`](#structcaffe2_1_1python_1_1python__detail_1_1_func)` & getFunc()` |

## Members

#### `protected virtual const `[`python_detail::Func`](#structcaffe2_1_1python_1_1python__detail_1_1_func)` & getFunc()` {#classcaffe2_1_1python_1_1_python_op_1a99747ecc593d5eb25809fc06b91133f9}





# class `caffe2::python::PythonOpBase` {#classcaffe2_1_1python_1_1_python_op_base}

```
class caffe2::python::PythonOpBase
  : public Operator< CPUContext >
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline  PythonOpBase(const OperatorDef & operator_def,Workspace * ws)` |
`public bool RunOnDevice()` |
`protected Workspace * ws_` |
`protected const `[`python_detail::Func`](#structcaffe2_1_1python_1_1python__detail_1_1_func)` & getFunc()` |

## Members

#### `public inline  PythonOpBase(const OperatorDef & operator_def,Workspace * ws)` {#classcaffe2_1_1python_1_1_python_op_base_1a18d0e365c118ce6e0877b1ee0eec9ec6}





#### `public bool RunOnDevice()` {#classcaffe2_1_1python_1_1_python_op_base_1a799e19b302bbd79f5f92f6f59538ae1f}





#### `protected Workspace * ws_` {#classcaffe2_1_1python_1_1_python_op_base_1aead89a647c2de1a94f64012ad785546d}





#### `protected const `[`python_detail::Func`](#structcaffe2_1_1python_1_1python__detail_1_1_func)` & getFunc()` {#classcaffe2_1_1python_1_1_python_op_base_1a9c6919afb155da1ae62ec81e5a73c757}





# class `caffe2::python::StringFetcher` {#classcaffe2_1_1python_1_1_string_fetcher}

```
class caffe2::python::StringFetcher
  : public caffe2::python::BlobFetcherBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual py::object Fetch(const Blob & blob)` |

## Members

#### `public inline virtual py::object Fetch(const Blob & blob)` {#classcaffe2_1_1python_1_1_string_fetcher_1a1ccc149ee08f44b365b91ade5dc68ce6}





# class `caffe2::python::TensorFeeder` {#classcaffe2_1_1python_1_1_tensor_feeder}

```
class caffe2::python::TensorFeeder
  : public caffe2::python::BlobFeederBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline void FeedTensor(const DeviceOption & option,PyArrayObject * original_array,Tensor< Context > * tensor)` |
`public inline virtual void Feed(const DeviceOption & option,PyArrayObject * original_array,Blob * blob)` |

## Members

#### `public inline void FeedTensor(const DeviceOption & option,PyArrayObject * original_array,Tensor< Context > * tensor)` {#classcaffe2_1_1python_1_1_tensor_feeder_1acac5d75f2bd4bf0ae00ed0b52998e892}





#### `public inline virtual void Feed(const DeviceOption & option,PyArrayObject * original_array,Blob * blob)` {#classcaffe2_1_1python_1_1_tensor_feeder_1a272a7aecfecb623da69d0b9a0b28d5c0}





# class `caffe2::python::TensorFetcher` {#classcaffe2_1_1python_1_1_tensor_fetcher}

```
class caffe2::python::TensorFetcher
  : public caffe2::python::BlobFetcherBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual pybind11::object Fetch(const Blob & blob)` |
`public inline bool NeedsCopy(const TypeMeta & meta) const` |
`public inline `[`FetchedBlob`](#structcaffe2_1_1python_1_1_blob_fetcher_base_1_1_fetched_blob)` FetchTensor(const Tensor< Context > & tensor,bool force_copy)` |

## Members

#### `public inline virtual pybind11::object Fetch(const Blob & blob)` {#classcaffe2_1_1python_1_1_tensor_fetcher_1aa9a9bb270e3ba8f372baaf4e9e2ba5f7}





#### `public inline bool NeedsCopy(const TypeMeta & meta) const` {#classcaffe2_1_1python_1_1_tensor_fetcher_1a764ff62d8f7414b6f4bc84422e9c7200}





#### `public inline `[`FetchedBlob`](#structcaffe2_1_1python_1_1_blob_fetcher_base_1_1_fetched_blob)` FetchTensor(const Tensor< Context > & tensor,bool force_copy)` {#classcaffe2_1_1python_1_1_tensor_fetcher_1a44227576989ead4b026e3b0f3718b1a7}





# struct `caffe2::python::GetPythonGradient` {#structcaffe2_1_1python_1_1_get_python_gradient}

```
struct caffe2::python::GetPythonGradient
  : public GradientMakerBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline std::vector< OperatorDef > GetGradientDefs()` |

## Members

#### `public inline std::vector< OperatorDef > GetGradientDefs()` {#structcaffe2_1_1python_1_1_get_python_gradient_1ab958e33f5b2160081aa5e5696f122228}





# namespace `caffe2::python::python_detail` {#namespacecaffe2_1_1python_1_1python__detail}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`struct `[`caffe2::python::python_detail::Func`](#structcaffe2_1_1python_1_1python__detail_1_1_func)    |
# struct `caffe2::python::python_detail::Func` {#structcaffe2_1_1python_1_1python__detail_1_1_func}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public py::object py_func` |
`public bool needs_workspace` |

## Members

#### `public py::object py_func` {#structcaffe2_1_1python_1_1python__detail_1_1_func_1a4f92bc992f73953c57fe9d46a7d2534e}





#### `public bool needs_workspace` {#structcaffe2_1_1python_1_1python__detail_1_1_func_1a9481bdfcd13b3c9df32da7bfa75b0f57}





# namespace `caffe_translator` {#namespacecaffe__translator}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`caffe_translator::TranslatorRegistry`](#classcaffe__translator_1_1_translator_registry)    |
# class `caffe_translator::TranslatorRegistry` {#classcaffe__translator_1_1_translator_registry}

```
class caffe_translator::TranslatorRegistry
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def Register(cls,op_name)` |
`public def TranslateLayer(cls,layer,pretrained_blobs,`[`is_test`](#namespacecaffe__translator_1a93b532edd28d0a024bcadc2aeca11eae)`)` |
`public def TranslateModel(cls,caffe_net,pretrained_net,`[`is_test`](#namespacecaffe__translator_1a93b532edd28d0a024bcadc2aeca11eae)`,net_state)` |

## Members

#### `public def Register(cls,op_name)` {#classcaffe__translator_1_1_translator_registry_1a755d05297bd1b41291ad5b04729ded69}



A decorator for registering gradient mappings.

#### `public def TranslateLayer(cls,layer,pretrained_blobs,`[`is_test`](#namespacecaffe__translator_1a93b532edd28d0a024bcadc2aeca11eae)`)` {#classcaffe__translator_1_1_translator_registry_1ade5f3ea0aa9688cb3a11000c5aa4daa7}





#### `public def TranslateModel(cls,caffe_net,pretrained_net,`[`is_test`](#namespacecaffe__translator_1a93b532edd28d0a024bcadc2aeca11eae)`,net_state)` {#classcaffe__translator_1_1_translator_registry_1a87d803831fd7654565aac961bdbed4ba}





# namespace `caffe_translator_test` {#namespacecaffe__translator__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`caffe_translator_test::TestNumericalEquivalence`](#classcaffe__translator__test_1_1_test_numerical_equivalence)    |
# class `caffe_translator_test::TestNumericalEquivalence` {#classcaffe__translator__test_1_1_test_numerical_equivalence}

```
class caffe_translator_test::TestNumericalEquivalence
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testBlobs(self)` |

## Members

#### `public def testBlobs(self)` {#classcaffe__translator__test_1_1_test_numerical_equivalence_1a1316d1a83213e294c6c94e16263e1b00}





# namespace `char_rnn` {#namespacechar__rnn}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`char_rnn::CharRNN`](#classchar__rnn_1_1_char_r_n_n)    |
# class `char_rnn::CharRNN` {#classchar__rnn_1_1_char_r_n_n}

```
class char_rnn::CharRNN
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  seq_length` |
`public  batch_size` |
`public  iters_to_report` |
`public  hidden_size` |
`public  text` |
`public  vocab` |
`public  char_to_idx` |
`public  idx_to_char` |
`public  D` |
`public  cell_state` |
`public  forward_net` |
`public  model` |
`public  predictions` |
`public  loss` |
`public  prepare_state` |
`public def __init__(self,args)` |
`public def CreateModel(self)` |
`public def TrainModel(self)` |
`public def GenerateText(self,num_characters,ch)` |

## Members

#### `public  seq_length` {#classchar__rnn_1_1_char_r_n_n_1af34ec9d2c44cc91c031eb7a39544b11c}





#### `public  batch_size` {#classchar__rnn_1_1_char_r_n_n_1a9ec010c6c97e9bf93bef3bce46f491bb}





#### `public  iters_to_report` {#classchar__rnn_1_1_char_r_n_n_1a5098281080a6cd03e88e9ca267c7df28}





#### `public  hidden_size` {#classchar__rnn_1_1_char_r_n_n_1a3f0e50bd524b4a4b7d0fc1bd4437d8de}





#### `public  text` {#classchar__rnn_1_1_char_r_n_n_1acf84ce05e9549f746c27d83c4e40979f}





#### `public  vocab` {#classchar__rnn_1_1_char_r_n_n_1a7e3749753f603cc00eb15bbdc0e53231}





#### `public  char_to_idx` {#classchar__rnn_1_1_char_r_n_n_1ad5dd626868e535a97af6b02b39324737}





#### `public  idx_to_char` {#classchar__rnn_1_1_char_r_n_n_1a0a4185eb8cfd73f596fd3677e20a5cd0}





#### `public  D` {#classchar__rnn_1_1_char_r_n_n_1af43e5228ec903a9b058ec3c370b5ab5b}





#### `public  cell_state` {#classchar__rnn_1_1_char_r_n_n_1a3517b230dadc67f4a6b3e5c6d5ed2192}





#### `public  forward_net` {#classchar__rnn_1_1_char_r_n_n_1ae8c8b68bc9fe0394ac75dfba521ac334}





#### `public  model` {#classchar__rnn_1_1_char_r_n_n_1a86255cb9ff64c02417737a4163d03466}





#### `public  predictions` {#classchar__rnn_1_1_char_r_n_n_1a17285e2627928188087403a6dcd7dfb4}





#### `public  loss` {#classchar__rnn_1_1_char_r_n_n_1ad03bddc81ddf0a32ff606ffafcd02ff9}





#### `public  prepare_state` {#classchar__rnn_1_1_char_r_n_n_1ac54026468f41eb034c064ba48f319b16}





#### `public def __init__(self,args)` {#classchar__rnn_1_1_char_r_n_n_1acea435efa72c66ac2f9550f62a7b2b81}





#### `public def CreateModel(self)` {#classchar__rnn_1_1_char_r_n_n_1afbdfdbf392ebf49314d6b8c927114f9d}





#### `public def TrainModel(self)` {#classchar__rnn_1_1_char_r_n_n_1a771321548e4f76ab8f5ae72a95c2e0dc}





#### `public def GenerateText(self,num_characters,ch)` {#classchar__rnn_1_1_char_r_n_n_1a66cc63340d151f66f2159194d3b61d44}





# namespace `checkpoint` {#namespacecheckpoint}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`checkpoint::CheckpointManager`](#classcheckpoint_1_1_checkpoint_manager)    |
`class `[`checkpoint::Job`](#classcheckpoint_1_1_job)    |
`class `[`checkpoint::JobRunner`](#classcheckpoint_1_1_job_runner)    |
`class `[`checkpoint::MultiNodeCheckpointManager`](#classcheckpoint_1_1_multi_node_checkpoint_manager)    |
# class `checkpoint::CheckpointManager` {#classcheckpoint_1_1_checkpoint_manager}

```
class checkpoint::CheckpointManager
  : public object
```  



Controls saving and loading of workspaces on every epoch boundary of a job.
If a CheckpointManager instance is passed to JobRunner, then JobRunner will
call `init`, `read` and `save` at different moments in between epoch runs.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,db,db_type)` |
`public def init(self,nodes,retrieve_from_epoch)` |
`public def blob_list(self)` |
`public def load(self,epoch)` |
`public def save(self,epoch)` |

## Members

#### `public def __init__(self,db,db_type)` {#classcheckpoint_1_1_checkpoint_manager_1ad1621ed573f0d83fd38e5ec54c1f5150}





#### `public def init(self,nodes,retrieve_from_epoch)` {#classcheckpoint_1_1_checkpoint_manager_1a589fc2e02123b4245fc1321aa7c3c1db}



Build a Task that will be run once after the job's `init_group` is run.
This task will determine which blobs need to be checkpointed.
If retrieve_from_epoch is not None, then the checkpoint metadata is
retrieved from a previously saved checkpoint.

#### `public def blob_list(self)` {#classcheckpoint_1_1_checkpoint_manager_1a34446adfe329fe0b4dcc233f11837f24}





#### `public def load(self,epoch)` {#classcheckpoint_1_1_checkpoint_manager_1a29e6cdabb9f9290be74d9e8075d3d091}



Build a Task that will be run by JobRunner when the job is to be
resumed from a given epoch. This task will run a Load op that will
load and deserialize all relevant blobs from a persistent storage.

#### `public def save(self,epoch)` {#classcheckpoint_1_1_checkpoint_manager_1ac066d395b434f37e394a249e0c2bc47f}



Build a Task that is run once after `init_group` and after each
epoch is run. This will execute a Save ops to serialize and persist
blobs present in the global workspaace.

# class `checkpoint::Job` {#classcheckpoint_1_1_job}

```
class checkpoint::Job
  : public object
```  



A Job defines three TaskGroups: the `init_group`, the `epoch_group` and the
`exit_group` which will be run by a JobRunner.

The `init_group` will be run only once at startup. Its role is to
initialize globally persistent blobs such as model weights, accumulators
and data file lists.

The `epoch_group` will be run in a loop after init_group. The loop will
exit when any of the stop signals added with `add_stop_signal` is True
at the end of an epoch.

The `exit_group` will be run only once at the very end of the job, when one
of the stopping criterias for `epoch_group` was met. The role of this group
is save the results of training in the end of the job.

Jobs are context-driven, so that Tasks can be added to the active Job
without having to explicitly pass the job object around.

Example of usage:

def build_reader(partitions):
    with Job.current().init_group:
        reader = HiveReader(init_reader, ..., partitions)
        Task(step=init_reader)
    with Job.current().epoch_group:
        limited_reader = ReaderWithLimit(reader, num_iter=10000)
        data_queue = pipe(limited_reader, num_threads=8)
        Job.current().add_stop_signal(limited_reader.data_finished())
    return data_queue

def build_hogwild_trainer(reader, model):
    with Job.current().init_group:
        Task(step=model.param_init_net)
    with Job.current().epoch_group:
        pipe(reader, processor=model, num_threads=8)
    with Job.current().exit_group:
        Task(step=model.save_model_net)

with Job() as job:
    reader = build_reader(partitions)
    model = build_model(params)
    build_hogwild_trainer(reader, model)

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  init_group` |
`public  epoch_group` |
`public  exit_group` |
`public  stop_signals` |
`public def __init__(self,`[`init_group`](#classcheckpoint_1_1_job_1a6d470176f954046c8a62e44fe274066c)`,`[`epoch_group`](#classcheckpoint_1_1_job_1afb4a3db96ab2e13e47fc09790b047905)`,`[`exit_group`](#classcheckpoint_1_1_job_1af5da626c0af3ae55021fa60438fb6f8a)`,`[`stop_signals`](#classcheckpoint_1_1_job_1aa06f043289b429fd2c0df6c1f814ff40)`,`[`nodes_to_checkpoint`](#classcheckpoint_1_1_job_1a27e69378740d3040ffc8105907b9d3c8)`)` |
`public def nodes_to_checkpoint(self)` |
`public def compile(self,session_class)` |
`public def __enter__(self)` |
`public def __exit__(self,args)` |
`public def add_stop_signal(self,output)` |

## Members

#### `public  init_group` {#classcheckpoint_1_1_job_1a6d470176f954046c8a62e44fe274066c}





#### `public  epoch_group` {#classcheckpoint_1_1_job_1afb4a3db96ab2e13e47fc09790b047905}





#### `public  exit_group` {#classcheckpoint_1_1_job_1af5da626c0af3ae55021fa60438fb6f8a}





#### `public  stop_signals` {#classcheckpoint_1_1_job_1aa06f043289b429fd2c0df6c1f814ff40}





#### `public def __init__(self,`[`init_group`](#classcheckpoint_1_1_job_1a6d470176f954046c8a62e44fe274066c)`,`[`epoch_group`](#classcheckpoint_1_1_job_1afb4a3db96ab2e13e47fc09790b047905)`,`[`exit_group`](#classcheckpoint_1_1_job_1af5da626c0af3ae55021fa60438fb6f8a)`,`[`stop_signals`](#classcheckpoint_1_1_job_1aa06f043289b429fd2c0df6c1f814ff40)`,`[`nodes_to_checkpoint`](#classcheckpoint_1_1_job_1a27e69378740d3040ffc8105907b9d3c8)`)` {#classcheckpoint_1_1_job_1a9288de1911adc330fba72603923313cb}





#### `public def nodes_to_checkpoint(self)` {#classcheckpoint_1_1_job_1a27e69378740d3040ffc8105907b9d3c8}





#### `public def compile(self,session_class)` {#classcheckpoint_1_1_job_1ae750cf25cb587641d7443b33607fc666}





#### `public def __enter__(self)` {#classcheckpoint_1_1_job_1a0fbf0d78f3a4afabd96a38a3c72fbeac}





#### `public def __exit__(self,args)` {#classcheckpoint_1_1_job_1adfaaca2bcc688e993453b646796dc9c6}





#### `public def add_stop_signal(self,output)` {#classcheckpoint_1_1_job_1a15bc67c2f2eaa602478e61f254bde8a8}





# class `checkpoint::JobRunner` {#classcheckpoint_1_1_job_runner}

```
class checkpoint::JobRunner
  : public object
```  



Implement the runtime logic for jobs with checkpointing at the level of
epoch. Can be used to run either single-host or distributed jobs. Job
runner is a callable to be called once from the client, passing a Session
as argument. This call will block until the Job execution is complete.

If a checkpoint_manager is passed, checkpoints will be taken after
initialization and after each epoch execution. If, in addition,
`resume_from_epoch` is an epoch number, the corresponding checkpoint will
be loaded and job execution will continue from the given epoch. In
this case, the job's init_group will not be run.

Refer to checkpoint_test.py for an example.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  resume_from_epoch` |
`public  checkpoint` |
`public  job` |
`public def __init__(self,`[`job`](#classcheckpoint_1_1_job_runner_1ae1b644f9df16fda12929c286ba44fe42)`,checkpoint_manager,`[`resume_from_epoch`](#classcheckpoint_1_1_job_runner_1acb408839a20cac7811bb79f26f6259df)`)` |
`public def __call__(self,client)` |

## Members

#### `public  resume_from_epoch` {#classcheckpoint_1_1_job_runner_1acb408839a20cac7811bb79f26f6259df}





#### `public  checkpoint` {#classcheckpoint_1_1_job_runner_1a48b331656a4f19648dc72895fba895bb}





#### `public  job` {#classcheckpoint_1_1_job_runner_1ae1b644f9df16fda12929c286ba44fe42}





#### `public def __init__(self,`[`job`](#classcheckpoint_1_1_job_runner_1ae1b644f9df16fda12929c286ba44fe42)`,checkpoint_manager,`[`resume_from_epoch`](#classcheckpoint_1_1_job_runner_1acb408839a20cac7811bb79f26f6259df)`)` {#classcheckpoint_1_1_job_runner_1a118344727ddee33f73db975d9ecbc878}





#### `public def __call__(self,client)` {#classcheckpoint_1_1_job_runner_1a18cfad046233b6e11744c0f77d6dd39d}





# class `checkpoint::MultiNodeCheckpointManager` {#classcheckpoint_1_1_multi_node_checkpoint_manager}

```
class checkpoint::MultiNodeCheckpointManager
  : public object
```  



Coordinates checkpointing and checkpointing across multiple nodes.
Each of `init`, `load` and `save` will build TaskGroups which will
trigger checkpointing on each of the nodes involved in a distributed job.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,db_prefix,db_type,node_manager_class)` |
`public def init(self,nodes,retrieve_from_epoch)` |
`public def load(self,epoch)` |
`public def save(self,epoch)` |

## Members

#### `public def __init__(self,db_prefix,db_type,node_manager_class)` {#classcheckpoint_1_1_multi_node_checkpoint_manager_1a5d28b4b8538e5520a591449fc7ac482c}





#### `public def init(self,nodes,retrieve_from_epoch)` {#classcheckpoint_1_1_multi_node_checkpoint_manager_1a88939d5486ffcbf0bc4c79f4e120f9a0}





#### `public def load(self,epoch)` {#classcheckpoint_1_1_multi_node_checkpoint_manager_1a6f3f7a77783bfa8c76d573cac9755aa6}





#### `public def save(self,epoch)` {#classcheckpoint_1_1_multi_node_checkpoint_manager_1adc8ed2a690251ffd359388d276f394a3}





# namespace `checkpoint_test` {#namespacecheckpoint__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`checkpoint_test::CheckpointTest`](#classcheckpoint__test_1_1_checkpoint_test)    |
`class `[`checkpoint_test::TestCheckpoint`](#classcheckpoint__test_1_1_test_checkpoint)    |
# class `checkpoint_test::CheckpointTest` {#classcheckpoint__test_1_1_checkpoint_test}

```
class checkpoint_test::CheckpointTest
  : public TestCase
```  



A simple test case to make sure that the checkpoint behavior is correct.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testCheckpoint(self)` |

## Members

#### `public def testCheckpoint(self)` {#classcheckpoint__test_1_1_checkpoint_test_1a647170f92e62ffdebb5756763f6b3d60}





# class `checkpoint_test::TestCheckpoint` {#classcheckpoint__test_1_1_test_checkpoint}

```
class checkpoint_test::TestCheckpoint
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def run_with(self,builder)` |
`public def test_single_checkpoint(self)` |

## Members

#### `public def run_with(self,builder)` {#classcheckpoint__test_1_1_test_checkpoint_1aa03abd01d52614c38eed5fb7e51b03e8}





#### `public def test_single_checkpoint(self)` {#classcheckpoint__test_1_1_test_checkpoint_1af2c649b2812e3521d5aa3c3ee4c57255}





# namespace `cnn` {#namespacecnn}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`cnn::CNNModelHelper`](#classcnn_1_1_c_n_n_model_helper)    |
# class `cnn::CNNModelHelper` {#classcnn_1_1_c_n_n_model_helper}

```
class cnn::CNNModelHelper
  : public ModelHelperBase
```  



A helper model so we can write CNN models more easily, without having to
manually define parameter initializations and operators separately.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  weights` |
`public  biases` |
`public  order` |
`public  use_cudnn` |
`public  cudnn_exhaustive_search` |
`public  ws_nbytes_limit` |
`public def __init__(self,`[`order`](#classcnn_1_1_c_n_n_model_helper_1afe7f82eb7d6660a11e55d8ddae979d6f)`,name,`[`use_cudnn`](#classcnn_1_1_c_n_n_model_helper_1afbd329e6417894a39a5317b158c0ab25)`,`[`cudnn_exhaustive_search`](#classcnn_1_1_c_n_n_model_helper_1ac340457d0b8151ba4b11a11a6a9bb8e3)`,`[`ws_nbytes_limit`](#classcnn_1_1_c_n_n_model_helper_1af83e7c7093cfc8030ed294291eb7b1dd)`,init_params,skip_sparse_optim,param_model)` |
`public def GetWeights(self,namescope)` |
`public def GetBiases(self,namescope)` |
`public def ImageInput(self,blob_in,blob_out,use_gpu_transform,kwargs)` |
`public def Conv(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,group,transform_inputs,kwargs)` |
`public def ConvTranspose(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,kwargs)` |
`public def GroupConv(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,group,kwargs)` |
`public def GroupConv_Deprecated(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,group,kwargs)` |
`public def FC(self,args,kwargs)` |
`public def PackedFC(self,args,kwargs)` |
`public def FC_Decomp(self,blob_in,blob_out,dim_in,dim_out,rank_approx,weight_init,bias_init,kwargs)` |
`public def FC_Prune(self,blob_in,blob_out,dim_in,dim_out,weight_init,bias_init,mask_init,threshold,need_compress_rate,comp_lb,kwargs)` |
`public def FC_Sparse(self,blob_in,blob_out,w_csr,iw,jw,bias,kwargs)` |
`public def LRN(self,blob_in,blob_out,kwargs)` |
`public def Dropout(self,blob_in,blob_out,kwargs)` |
`public def MaxPool(self,blob_in,blob_out,kwargs)` |
`public def AveragePool(self,blob_in,blob_out,kwargs)` |
`public def Concat(self,blobs_in,blob_out,kwargs)` |
`public def DepthConcat(self,blobs_in,blob_out,kwargs)` |
`public def PRelu(self,blob_in,blob_out,num_channels,slope_init,kwargs)` |
`public def Relu(self,blob_in,blob_out,kwargs)` |
`public def Transpose(self,blob_in,blob_out,kwargs)` |
`public def Sum(self,blob_in,blob_out,kwargs)` |
`public def InstanceNorm(self,blob_in,blob_out,dim_in,kwargs)` |
`public def SpatialBN(self,blob_in,blob_out,dim_in,kwargs)` |
`public def Iter(self,blob_out,kwargs)` |
`public def Accuracy(self,blob_in,blob_out,kwargs)` |
`public def PadImage(self,blob_in,blob_out,kwargs)` |
`public def XavierInit(self)` |
`public def ConstantInit(self,value)` |
`public def MSRAInit(self)` |
`public def ZeroInit(self)` |
`public def AddWeightDecay(self,weight_decay)` |
`public def CPU(self)` |
`public def GPU(self,gpu_id)` |

## Members

#### `public  weights` {#classcnn_1_1_c_n_n_model_helper_1a00347863bafe6617d993f2b7ae4a905f}





#### `public  biases` {#classcnn_1_1_c_n_n_model_helper_1a6e7c33cbe747909dfb313fc3491d2721}





#### `public  order` {#classcnn_1_1_c_n_n_model_helper_1afe7f82eb7d6660a11e55d8ddae979d6f}





#### `public  use_cudnn` {#classcnn_1_1_c_n_n_model_helper_1afbd329e6417894a39a5317b158c0ab25}





#### `public  cudnn_exhaustive_search` {#classcnn_1_1_c_n_n_model_helper_1ac340457d0b8151ba4b11a11a6a9bb8e3}





#### `public  ws_nbytes_limit` {#classcnn_1_1_c_n_n_model_helper_1af83e7c7093cfc8030ed294291eb7b1dd}





#### `public def __init__(self,`[`order`](#classcnn_1_1_c_n_n_model_helper_1afe7f82eb7d6660a11e55d8ddae979d6f)`,name,`[`use_cudnn`](#classcnn_1_1_c_n_n_model_helper_1afbd329e6417894a39a5317b158c0ab25)`,`[`cudnn_exhaustive_search`](#classcnn_1_1_c_n_n_model_helper_1ac340457d0b8151ba4b11a11a6a9bb8e3)`,`[`ws_nbytes_limit`](#classcnn_1_1_c_n_n_model_helper_1af83e7c7093cfc8030ed294291eb7b1dd)`,init_params,skip_sparse_optim,param_model)` {#classcnn_1_1_c_n_n_model_helper_1a06a7f9614789d607c48ef9a480d9b9f7}





#### `public def GetWeights(self,namescope)` {#classcnn_1_1_c_n_n_model_helper_1adff567260f914575c62c199a9da350aa}





#### `public def GetBiases(self,namescope)` {#classcnn_1_1_c_n_n_model_helper_1a26d7d384fb2987cefd7052d32dbe3460}





#### `public def ImageInput(self,blob_in,blob_out,use_gpu_transform,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a4da84bc6aad9b78509bb200a4638a060}



Image Input.

#### `public def Conv(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,group,transform_inputs,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a27163c7d1c44d563baafa11de0476be6}



Convolution. We intentionally do not provide odd kernel/stride/pad
settings in order to discourage the use of odd cases.

#### `public def ConvTranspose(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a73d2419aa859e072d3638aebb89e0a7f}



ConvTranspose.

#### `public def GroupConv(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,group,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a782f95f79b7c969c6af16d5d1ccb2e85}



Group Convolution.

This is essentially the same as Conv with a group argument passed in.
We specialize this for backward interface compatibility.

#### `public def GroupConv_Deprecated(self,blob_in,blob_out,dim_in,dim_out,kernel,weight_init,bias_init,group,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a960e96e236c0c5a451e3dd8f18154588}



GroupConvolution's deprecated interface.

This is used to simulate a group convolution via split and concat. You
should always use the new group convolution in your new code.

#### `public def FC(self,args,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a3701d59dd231976c04beaffc2327b125}





#### `public def PackedFC(self,args,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1ab96ef0197626e08b6384a33f394f0536}





#### `public def FC_Decomp(self,blob_in,blob_out,dim_in,dim_out,rank_approx,weight_init,bias_init,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1ab23c26f2acbec7347e685b304668e98e}



FC_Decomp version
Here we assume that the rank of original input is bigger than 5.

#### `public def FC_Prune(self,blob_in,blob_out,dim_in,dim_out,weight_init,bias_init,mask_init,threshold,need_compress_rate,comp_lb,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a1d693db1d1adcae7543ae13ef02b2528}



FC_Prune version
Runnable so far. Great!:)

#### `public def FC_Sparse(self,blob_in,blob_out,w_csr,iw,jw,bias,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1aeb7d743e67e77dc9529fded01fbfcae8}



FC_Sparse: Only takes in alocated weights

#### `public def LRN(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1af53dd06f9f5b6d12f92b32b6bb8699cb}



LRN

#### `public def Dropout(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a65ebeb3a3cde74999c7ccafd01a63385}



Dropout

#### `public def MaxPool(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a56f170e4ced5ae9a566a87f54907af8c}



Max pooling

#### `public def AveragePool(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1ace33e54a068d5a9ec66d47ae874f3e34}



Average pooling

#### `public def Concat(self,blobs_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1aee52ef5a308ce56f9131005193f01752}



Depth Concat.

#### `public def DepthConcat(self,blobs_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a98bb571a96184678b460d49e3873237c}



The old depth concat function - we should move to use concat.

#### `public def PRelu(self,blob_in,blob_out,num_channels,slope_init,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a518251874d418f2371af73e94e23a903}



PRelu

#### `public def Relu(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a7019b1ee8ec49e55b9cb9032d17559e3}



Relu.

#### `public def Transpose(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a16cb74bff5718c66c86ec5a1bffde574}



Transpose.

#### `public def Sum(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1ab01f9186a346a3d46c297f3adf422a91}



Sum

#### `public def InstanceNorm(self,blob_in,blob_out,dim_in,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1a3040c698d3d26aa9db1c5ccca222d426}





#### `public def SpatialBN(self,blob_in,blob_out,dim_in,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1ad924333d71f6c3686866b565b7cdd70d}





#### `public def Iter(self,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1af4637179bbb0b2121c13da490552a93f}





#### `public def Accuracy(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1ae40795ba00fa4b20c066f625c837d65c}





#### `public def PadImage(self,blob_in,blob_out,kwargs)` {#classcnn_1_1_c_n_n_model_helper_1ae7fe68424c9cd4c8c04b67f6fc8db36f}





#### `public def XavierInit(self)` {#classcnn_1_1_c_n_n_model_helper_1ae961797007a0242bd5b88d317baa882c}





#### `public def ConstantInit(self,value)` {#classcnn_1_1_c_n_n_model_helper_1aab0ccf634dba1ec4d6c645b37a4dc828}





#### `public def MSRAInit(self)` {#classcnn_1_1_c_n_n_model_helper_1a5851d3dcd63514a09b4e3425ae1e2038}





#### `public def ZeroInit(self)` {#classcnn_1_1_c_n_n_model_helper_1a688d427e6923759c3c2f43f5e78990f4}





#### `public def AddWeightDecay(self,weight_decay)` {#classcnn_1_1_c_n_n_model_helper_1a26379c6a454f79a5bbdc3f907cdea046}



Adds a decay to weights in the model.

This is a form of L2 regularization.

Args:
    weight_decay: strength of the regularization

#### `public def CPU(self)` {#classcnn_1_1_c_n_n_model_helper_1adbdc8d2a131f16656a28b0f02bdd124b}





#### `public def GPU(self,gpu_id)` {#classcnn_1_1_c_n_n_model_helper_1a0c818ac77a152d047ca08d1e2ca87260}





# namespace `context` {#namespacecontext}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`context::ContextInfo`](#classcontext_1_1_context_info)    |
`class `[`context::ContextManager`](#classcontext_1_1_context_manager)    |
`class `[`context::define_context`](#classcontext_1_1define__context)    |
# class `context::ContextInfo` {#classcontext_1_1_context_info}

```
class context::ContextInfo
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  cls` |
`public  allow_default` |
`public  arg_name` |
`public def __init__(self,`[`cls`](#classcontext_1_1_context_info_1a8c672f1587d32ce5a726903855fbe267)`,`[`allow_default`](#classcontext_1_1_context_info_1a301916b858e60170e6bb6f923fd18048)`,`[`arg_name`](#classcontext_1_1_context_info_1a31e3b63641385fb1e1efb90ede433197)`)` |
`public def enter(self,value)` |
`public def exit(self,value)` |
`public def get_active(self,required)` |

## Members

#### `public  cls` {#classcontext_1_1_context_info_1a8c672f1587d32ce5a726903855fbe267}





#### `public  allow_default` {#classcontext_1_1_context_info_1a301916b858e60170e6bb6f923fd18048}





#### `public  arg_name` {#classcontext_1_1_context_info_1a31e3b63641385fb1e1efb90ede433197}





#### `public def __init__(self,`[`cls`](#classcontext_1_1_context_info_1a8c672f1587d32ce5a726903855fbe267)`,`[`allow_default`](#classcontext_1_1_context_info_1a301916b858e60170e6bb6f923fd18048)`,`[`arg_name`](#classcontext_1_1_context_info_1a31e3b63641385fb1e1efb90ede433197)`)` {#classcontext_1_1_context_info_1ab083045400ddaf32e0fa47fc2f41b22d}





#### `public def enter(self,value)` {#classcontext_1_1_context_info_1a2608b9079e3e295d241c585975fc0f82}





#### `public def exit(self,value)` {#classcontext_1_1_context_info_1a9f9d4bc47d74886de6de6a5b8397818a}





#### `public def get_active(self,required)` {#classcontext_1_1_context_info_1afcb07fddf1bf8640b924ed1784877e6a}





# class `context::ContextManager` {#classcontext_1_1_context_manager}

```
class context::ContextManager
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self)` |
`public def register(self,ctx_info)` |
`public def get(self,cls)` |

## Members

#### `public def __init__(self)` {#classcontext_1_1_context_manager_1ad712e6e4fb1c4459ae4233b336547608}





#### `public def register(self,ctx_info)` {#classcontext_1_1_context_manager_1ac6ffcb182cd92f0031faca58d9ac07f5}





#### `public def get(self,cls)` {#classcontext_1_1_context_manager_1a98c81d49cdfd530ed8e1ea03009b01be}





# class `context::define_context` {#classcontext_1_1define__context}

```
class context::define_context
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  arg_name` |
`public  allow_default` |
`public  current` |
`public def __init__(self,`[`arg_name`](#classcontext_1_1define__context_1ad4a943afbb3ed24f45152fbd2669d901)`,`[`allow_default`](#classcontext_1_1define__context_1a2c09abb6f8618116589232311eae5487)`)` |
`public def __call__(self,cls)` |

## Members

#### `public  arg_name` {#classcontext_1_1define__context_1ad4a943afbb3ed24f45152fbd2669d901}





#### `public  allow_default` {#classcontext_1_1define__context_1a2c09abb6f8618116589232311eae5487}





#### `public  current` {#classcontext_1_1define__context_1a5ee535b4dac91d105f82c8598ca09ede}





#### `public def __init__(self,`[`arg_name`](#classcontext_1_1define__context_1ad4a943afbb3ed24f45152fbd2669d901)`,`[`allow_default`](#classcontext_1_1define__context_1a2c09abb6f8618116589232311eae5487)`)` {#classcontext_1_1define__context_1a8084f8ee76a253784c3a25faa5716556}





#### `public def __call__(self,cls)` {#classcontext_1_1define__context_1aedd18e7ee1a953e4902e27f165f29e6a}





# namespace `context_test` {#namespacecontext__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`context_test::MyContext`](#classcontext__test_1_1_my_context)    |
`class `[`context_test::TestContext`](#classcontext__test_1_1_test_context)    |
# class `context_test::MyContext` {#classcontext__test_1_1_my_context}

```
class context_test::MyContext
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `context_test::TestContext` {#classcontext__test_1_1_test_context}

```
class context_test::TestContext
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def use_my_context(self)` |
`public def testMultiThreaded(self)` |

## Members

#### `public def use_my_context(self)` {#classcontext__test_1_1_test_context_1aaae020d5f20e6af7f5cae07c09899c33}





#### `public def testMultiThreaded(self)` {#classcontext__test_1_1_test_context_1abb58f1dc8b59bf5f872ecb97e97f62e7}





# namespace `control_test` {#namespacecontrol__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`control_test::TestControl`](#classcontrol__test_1_1_test_control)    |
# class `control_test::TestControl` {#classcontrol__test_1_1_test_control}

```
class control_test::TestControl
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  N_` |
`public  init_net_` |
`public  cnt_net_` |
`public  cnt_2_net_` |
`public  cond_net_` |
`public  not_cond_net_` |
`public  true_cond_net_` |
`public  false_cond_net_` |
`public  idle_net_` |
`public def setUp(self)` |
`public def CheckNetOutput(self,nets_and_expects)` |
`public def CheckNetAllOutput(self,net,expects)` |
`public def BuildAndRunPlan(self,step)` |
`public def ForLoopTest(self,nets_or_steps)` |
`public def testForLoopWithNets(self)` |
`public def testForLoopWithStep(self)` |
`public def WhileLoopTest(self,nets_or_steps)` |
`public def testWhileLoopWithNet(self)` |
`public def testWhileLoopWithStep(self)` |
`public def UntilLoopTest(self,nets_or_steps)` |
`public def testUntilLoopWithNet(self)` |
`public def testUntilLoopWithStep(self)` |
`public def DoWhileLoopTest(self,nets_or_steps)` |
`public def testDoWhileLoopWithNet(self)` |
`public def testDoWhileLoopWithStep(self)` |
`public def DoUntilLoopTest(self,nets_or_steps)` |
`public def testDoUntilLoopWithNet(self)` |
`public def testDoUntilLoopWithStep(self)` |
`public def IfCondTest(self,cond_net,expect,cond_on_blob)` |
`public def testIfCondTrueOnNet(self)` |
`public def testIfCondTrueOnBlob(self)` |
`public def testIfCondFalseOnNet(self)` |
`public def testIfCondFalseOnBlob(self)` |
`public def IfElseCondTest(self,cond_net,cond_value,expect,cond_on_blob)` |
`public def testIfElseCondTrueOnNet(self)` |
`public def testIfElseCondTrueOnBlob(self)` |
`public def testIfElseCondFalseOnNet(self)` |
`public def testIfElseCondFalseOnBlob(self)` |
`public def IfNotCondTest(self,cond_net,expect,cond_on_blob)` |
`public def testIfNotCondTrueOnNet(self)` |
`public def testIfNotCondTrueOnBlob(self)` |
`public def testIfNotCondFalseOnNet(self)` |
`public def testIfNotCondFalseOnBlob(self)` |
`public def IfNotElseCondTest(self,cond_net,cond_value,expect,cond_on_blob)` |
`public def testIfNotElseCondTrueOnNet(self)` |
`public def testIfNotElseCondTrueOnBlob(self)` |
`public def testIfNotElseCondFalseOnNet(self)` |
`public def testIfNotElseCondFalseOnBlob(self)` |
`public def testSwitch(self)` |
`public def testSwitchNot(self)` |
`public def testBoolNet(self)` |
`public def testCombineConditions(self)` |
`public def testMergeConditionNets(self)` |

## Members

#### `public  N_` {#classcontrol__test_1_1_test_control_1a39d81187285ba1c87728a76c242f90e6}





#### `public  init_net_` {#classcontrol__test_1_1_test_control_1a2f6d0fcf8446567bf1c99668b79d5977}





#### `public  cnt_net_` {#classcontrol__test_1_1_test_control_1aee2889dcafbefb32fe479b8277cd4aae}





#### `public  cnt_2_net_` {#classcontrol__test_1_1_test_control_1acb845b67f24a2a84ce66e61add29f3c4}





#### `public  cond_net_` {#classcontrol__test_1_1_test_control_1a4f5b532c3f26883427a8f30720c57f94}





#### `public  not_cond_net_` {#classcontrol__test_1_1_test_control_1ad30c9167ed98fafbe8f2f739a43ad2a7}





#### `public  true_cond_net_` {#classcontrol__test_1_1_test_control_1a3c3c90d47802fee609c291234c09fefe}





#### `public  false_cond_net_` {#classcontrol__test_1_1_test_control_1ad20473ed4b409d9c4e70d8346a77d6a4}





#### `public  idle_net_` {#classcontrol__test_1_1_test_control_1aa795978628ab70e539a6d2d08c1dc8e9}





#### `public def setUp(self)` {#classcontrol__test_1_1_test_control_1af84c67049a8b312127247e1ea0fe4348}





#### `public def CheckNetOutput(self,nets_and_expects)` {#classcontrol__test_1_1_test_control_1a967c71a611252d551a91dd95bcae834b}



Check the net output is expected
nets_and_expects is a list of tuples (net, expect)

#### `public def CheckNetAllOutput(self,net,expects)` {#classcontrol__test_1_1_test_control_1af2e39a7bc0865e8cbb52013fb6a2af8e}



Check the net output is expected
expects is a list of bools.

#### `public def BuildAndRunPlan(self,step)` {#classcontrol__test_1_1_test_control_1a3937e9e081cf05d80e5ec9c2cd0bb9c4}





#### `public def ForLoopTest(self,nets_or_steps)` {#classcontrol__test_1_1_test_control_1afe9d625367ed2e7c21462cf7506e208b}





#### `public def testForLoopWithNets(self)` {#classcontrol__test_1_1_test_control_1af4bc01d5c91cee76e140d9b80b54b67c}





#### `public def testForLoopWithStep(self)` {#classcontrol__test_1_1_test_control_1a712e4774c2f027c36c1887057d9f659d}





#### `public def WhileLoopTest(self,nets_or_steps)` {#classcontrol__test_1_1_test_control_1a0dd0eccf028305f2a5d657560d78f779}





#### `public def testWhileLoopWithNet(self)` {#classcontrol__test_1_1_test_control_1ad89351cfd1d5466e63c283cabe97b50a}





#### `public def testWhileLoopWithStep(self)` {#classcontrol__test_1_1_test_control_1a99339c4db05b9c161f21a12458a052fc}





#### `public def UntilLoopTest(self,nets_or_steps)` {#classcontrol__test_1_1_test_control_1a1377b83dfd45a4f472c703289c260170}





#### `public def testUntilLoopWithNet(self)` {#classcontrol__test_1_1_test_control_1a7f66c41a10dc701febbb273de10f1363}





#### `public def testUntilLoopWithStep(self)` {#classcontrol__test_1_1_test_control_1a055e032255bbb5ff0dde47e2499d7a31}





#### `public def DoWhileLoopTest(self,nets_or_steps)` {#classcontrol__test_1_1_test_control_1add4505d5b90a7b04af449af26a5d5466}





#### `public def testDoWhileLoopWithNet(self)` {#classcontrol__test_1_1_test_control_1aff22d6a7fffb1b91cde4bb29c0945e72}





#### `public def testDoWhileLoopWithStep(self)` {#classcontrol__test_1_1_test_control_1a48c5ac6a0741096a33aeefaeca0a3371}





#### `public def DoUntilLoopTest(self,nets_or_steps)` {#classcontrol__test_1_1_test_control_1ac872ff5466ee215e53d975602aae2b18}





#### `public def testDoUntilLoopWithNet(self)` {#classcontrol__test_1_1_test_control_1a54014d22cc7e467100e83dabdcef5450}





#### `public def testDoUntilLoopWithStep(self)` {#classcontrol__test_1_1_test_control_1acace0eab93aa82484a5366519472edc4}





#### `public def IfCondTest(self,cond_net,expect,cond_on_blob)` {#classcontrol__test_1_1_test_control_1a6c4cb88d5efb2d26f28e839536a47b42}





#### `public def testIfCondTrueOnNet(self)` {#classcontrol__test_1_1_test_control_1ae595181d1658cfb9f8126558e0b1603d}





#### `public def testIfCondTrueOnBlob(self)` {#classcontrol__test_1_1_test_control_1aa429ab41d03b1fbf89cab5bef6ebb7e9}





#### `public def testIfCondFalseOnNet(self)` {#classcontrol__test_1_1_test_control_1a434e8feab9cb55d1531cc39482277bac}





#### `public def testIfCondFalseOnBlob(self)` {#classcontrol__test_1_1_test_control_1acbd634c83a37df810c1e5ad38ee7e5b6}





#### `public def IfElseCondTest(self,cond_net,cond_value,expect,cond_on_blob)` {#classcontrol__test_1_1_test_control_1a34686eae3001ef6bf1aa095ab5fda898}





#### `public def testIfElseCondTrueOnNet(self)` {#classcontrol__test_1_1_test_control_1a326ce4bd0f0ea6415a78e3763887d2a2}





#### `public def testIfElseCondTrueOnBlob(self)` {#classcontrol__test_1_1_test_control_1a8ebcb3c34e42de4d14abc6442003b09a}





#### `public def testIfElseCondFalseOnNet(self)` {#classcontrol__test_1_1_test_control_1a15d7e4a591859eaf1f142faf9193d1bf}





#### `public def testIfElseCondFalseOnBlob(self)` {#classcontrol__test_1_1_test_control_1a14947282a3b248bfb46a264a1ea8cc79}





#### `public def IfNotCondTest(self,cond_net,expect,cond_on_blob)` {#classcontrol__test_1_1_test_control_1a558929a2524313544a2a4e48119f08fe}





#### `public def testIfNotCondTrueOnNet(self)` {#classcontrol__test_1_1_test_control_1a9eeea9ad6e2e5941ae24fef12a551590}





#### `public def testIfNotCondTrueOnBlob(self)` {#classcontrol__test_1_1_test_control_1adbf362a9d0e12ce600b6b2391f5ca224}





#### `public def testIfNotCondFalseOnNet(self)` {#classcontrol__test_1_1_test_control_1a584e15c42b03118d6268c353ac6519d6}





#### `public def testIfNotCondFalseOnBlob(self)` {#classcontrol__test_1_1_test_control_1ac2bc1501c5d8683b53e0dbcaa58eae2e}





#### `public def IfNotElseCondTest(self,cond_net,cond_value,expect,cond_on_blob)` {#classcontrol__test_1_1_test_control_1a66c462da4d26b8da919ba713d0f715aa}





#### `public def testIfNotElseCondTrueOnNet(self)` {#classcontrol__test_1_1_test_control_1aa43a88f34c6852ab021c3b92e42e2380}





#### `public def testIfNotElseCondTrueOnBlob(self)` {#classcontrol__test_1_1_test_control_1a11221c530861151715317a23f67d91d2}





#### `public def testIfNotElseCondFalseOnNet(self)` {#classcontrol__test_1_1_test_control_1aa49f0689afb419a4995075597f3bc7c4}





#### `public def testIfNotElseCondFalseOnBlob(self)` {#classcontrol__test_1_1_test_control_1a7e30ddb4c320de1fa25eb14e68530c2d}





#### `public def testSwitch(self)` {#classcontrol__test_1_1_test_control_1a3bbbbbc91c6c7d2b0ad5e36dfb666e9c}





#### `public def testSwitchNot(self)` {#classcontrol__test_1_1_test_control_1a85aa53c018bc14964514b43f3837ab38}





#### `public def testBoolNet(self)` {#classcontrol__test_1_1_test_control_1a336d8db02c85807bb51273258de0ab1c}





#### `public def testCombineConditions(self)` {#classcontrol__test_1_1_test_control_1adf43c13fccd90ff1133dda8edd612b3f}





#### `public def testMergeConditionNets(self)` {#classcontrol__test_1_1_test_control_1a6902cdb94023f081856ce8e403cc5bb5}





# namespace `conv_test` {#namespaceconv__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`conv_test::TestConvolution`](#classconv__test_1_1_test_convolution)    |
# class `conv_test::TestConvolution` {#classconv__test_1_1_test_convolution}

```
class conv_test::TestConvolution
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_convolution_separate_stride_pad_gradients(self,`[`stride_h`](#classconv__test_1_1_test_convolution_1afafcd9deaebaa93113259c53d24d2a56)`,`[`stride_w`](#classconv__test_1_1_test_convolution_1a58b264fd434f368e135a9bda0b34f9df)`,`[`pad_t`](#classconv__test_1_1_test_convolution_1ac2bc3aa191eeabc331026d9fba09ef1d)`,`[`pad_l`](#classconv__test_1_1_test_convolution_1a12afd1d51634ff70494c130d30c687db)`,`[`pad_b`](#classconv__test_1_1_test_convolution_1a46fb6d5854f8e60896a1e04ba45912bd)`,`[`pad_r`](#classconv__test_1_1_test_convolution_1abcc443a6f488855c481d38fac6fd8eaa)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`order`](#classconv__test_1_1_test_convolution_1ac8ebcd5b68e60080bd85753d25e5d84a)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`,`[`shared_buffer`](#classconv__test_1_1_test_convolution_1a9902765b2fcc086dc89c11e60f0fcbb3)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` |
`public def test_convolution_separate_stride_pad_layout(self,`[`stride_h`](#classconv__test_1_1_test_convolution_1afafcd9deaebaa93113259c53d24d2a56)`,`[`stride_w`](#classconv__test_1_1_test_convolution_1a58b264fd434f368e135a9bda0b34f9df)`,`[`pad_t`](#classconv__test_1_1_test_convolution_1ac2bc3aa191eeabc331026d9fba09ef1d)`,`[`pad_l`](#classconv__test_1_1_test_convolution_1a12afd1d51634ff70494c130d30c687db)`,`[`pad_b`](#classconv__test_1_1_test_convolution_1a46fb6d5854f8e60896a1e04ba45912bd)`,`[`pad_r`](#classconv__test_1_1_test_convolution_1abcc443a6f488855c481d38fac6fd8eaa)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` |
`public def test_convolution_gradients(self,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`order`](#classconv__test_1_1_test_convolution_1ac8ebcd5b68e60080bd85753d25e5d84a)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` |
`public def test_1d_convlution_nchw(self,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` |
`public def test_3d_convlution_nchw(self,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` |
`public def test_convolution_layout(self,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` |
`public def test_convolution_sync(self,`[`net_type`](#classconv__test_1_1_test_convolution_1a586ddee33931c8ba2e13d0b02f809e7a)`,`[`num_workers`](#classconv__test_1_1_test_convolution_1a83eaab199b29897dea909f2ca8fe56c8)`,`[`do`](#classconv__test_1_1_test_convolution_1af4a0b54e748a52508c5c75b34eede360)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`)` |

## Members

#### `public def test_convolution_separate_stride_pad_gradients(self,`[`stride_h`](#classconv__test_1_1_test_convolution_1afafcd9deaebaa93113259c53d24d2a56)`,`[`stride_w`](#classconv__test_1_1_test_convolution_1a58b264fd434f368e135a9bda0b34f9df)`,`[`pad_t`](#classconv__test_1_1_test_convolution_1ac2bc3aa191eeabc331026d9fba09ef1d)`,`[`pad_l`](#classconv__test_1_1_test_convolution_1a12afd1d51634ff70494c130d30c687db)`,`[`pad_b`](#classconv__test_1_1_test_convolution_1a46fb6d5854f8e60896a1e04ba45912bd)`,`[`pad_r`](#classconv__test_1_1_test_convolution_1abcc443a6f488855c481d38fac6fd8eaa)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`order`](#classconv__test_1_1_test_convolution_1ac8ebcd5b68e60080bd85753d25e5d84a)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`,`[`shared_buffer`](#classconv__test_1_1_test_convolution_1a9902765b2fcc086dc89c11e60f0fcbb3)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` {#classconv__test_1_1_test_convolution_1ada0d62edc51ba6ad750bbb3bd6a277d7}





#### `public def test_convolution_separate_stride_pad_layout(self,`[`stride_h`](#classconv__test_1_1_test_convolution_1afafcd9deaebaa93113259c53d24d2a56)`,`[`stride_w`](#classconv__test_1_1_test_convolution_1a58b264fd434f368e135a9bda0b34f9df)`,`[`pad_t`](#classconv__test_1_1_test_convolution_1ac2bc3aa191eeabc331026d9fba09ef1d)`,`[`pad_l`](#classconv__test_1_1_test_convolution_1a12afd1d51634ff70494c130d30c687db)`,`[`pad_b`](#classconv__test_1_1_test_convolution_1a46fb6d5854f8e60896a1e04ba45912bd)`,`[`pad_r`](#classconv__test_1_1_test_convolution_1abcc443a6f488855c481d38fac6fd8eaa)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` {#classconv__test_1_1_test_convolution_1ad2f28db7ec39cd90ffb37d262c74ea35}





#### `public def test_convolution_gradients(self,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`order`](#classconv__test_1_1_test_convolution_1ac8ebcd5b68e60080bd85753d25e5d84a)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` {#classconv__test_1_1_test_convolution_1a4d30cb97747ffc25d83786e40a9ff307}





#### `public def test_1d_convlution_nchw(self,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` {#classconv__test_1_1_test_convolution_1af945fc34571b78d3c391463ec6965d81}





#### `public def test_3d_convlution_nchw(self,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` {#classconv__test_1_1_test_convolution_1a5b9f3c3c7dfba9b4b2a1c93cf47a935d}





#### `public def test_convolution_layout(self,`[`stride`](#classconv__test_1_1_test_convolution_1a1977bfb839e6a9ffd9d7509e62ff24aa)`,`[`pad`](#classconv__test_1_1_test_convolution_1a83f1dc17b014d5e79abd64c206f2bd2e)`,`[`kernel`](#classconv__test_1_1_test_convolution_1ac21eeec54f9ca36e12071292a47b6ff5)`,`[`dilation`](#classconv__test_1_1_test_convolution_1a40a9f506da8ce86069639fe2cabdda12)`,`[`size`](#classconv__test_1_1_test_convolution_1a199169268acb42a9b0b878a716cae4b6)`,`[`input_channels`](#classconv__test_1_1_test_convolution_1a0f8ef947a70d989a3bc88a56d6842c84)`,`[`output_channels`](#classconv__test_1_1_test_convolution_1aca334403da7215a908ff9a5f5503d2ec)`,`[`batch_size`](#classconv__test_1_1_test_convolution_1a2354081941b8b87ab61626b10afceeff)`,`[`use_bias`](#classconv__test_1_1_test_convolution_1ae14ecb85957c18fef4d27c24085e798e)`,gc,dc)` {#classconv__test_1_1_test_convolution_1a852e34fbea57d3746675d96e36a73aa3}





#### `public def test_convolution_sync(self,`[`net_type`](#classconv__test_1_1_test_convolution_1a586ddee33931c8ba2e13d0b02f809e7a)`,`[`num_workers`](#classconv__test_1_1_test_convolution_1a83eaab199b29897dea909f2ca8fe56c8)`,`[`do`](#classconv__test_1_1_test_convolution_1af4a0b54e748a52508c5c75b34eede360)`,`[`engine`](#classconv__test_1_1_test_convolution_1a5684aa01a5860487512729c3f728a54a)`)` {#classconv__test_1_1_test_convolution_1a99872a484bfdbc55eeac74b54513147c}





# namespace `conv_transpose_test` {#namespaceconv__transpose__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`conv_transpose_test::TestConvolutionTranspose`](#classconv__transpose__test_1_1_test_convolution_transpose)    |
# class `conv_transpose_test::TestConvolutionTranspose` {#classconv__transpose__test_1_1_test_convolution_transpose}

```
class conv_transpose_test::TestConvolutionTranspose
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_convolution_transpose_layout(self,`[`stride`](#classconv__transpose__test_1_1_test_convolution_transpose_1a8e9fe1f03a25c4dd85bf144a4d7c8dc2)`,`[`pad`](#classconv__transpose__test_1_1_test_convolution_transpose_1a342f6b81c783a263b70a2471bb200fbd)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj`](#classconv__transpose__test_1_1_test_convolution_transpose_1afc7581d0c041d694e3d0a9b1bc08c790)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,`[`shared_buffer`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7a1a92ae6f04b418dade17e6649e96ea)`,gc,dc)` |
`public def test_convolution_transpose_separate_stride_pad_adj_layout(self,`[`stride_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5ef04ac962f966f71f9ba0bf6a59501a)`,`[`stride_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1ad0453f3d1a62871ceec81f5555a183df)`,`[`pad_t`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7bc2a5c5f7adc55d99a41f554231ee50)`,`[`pad_l`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0d8c2f26cc669499c9e9fe358f40954a)`,`[`pad_b`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7abe4791046feea73ab8a8cc6865ed90)`,`[`pad_r`](#classconv__transpose__test_1_1_test_convolution_transpose_1a1a86164897744efccdc5057bd666cefb)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a9b21589e93d70ebda55f0475ca29e249)`,`[`adj_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5afff37b8de7f85db6d020abdca84a19)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,gc,dc)` |
`public def test_convolution_transpose_gradients(self,`[`stride`](#classconv__transpose__test_1_1_test_convolution_transpose_1a8e9fe1f03a25c4dd85bf144a4d7c8dc2)`,`[`pad`](#classconv__transpose__test_1_1_test_convolution_transpose_1a342f6b81c783a263b70a2471bb200fbd)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj`](#classconv__transpose__test_1_1_test_convolution_transpose_1afc7581d0c041d694e3d0a9b1bc08c790)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`order`](#classconv__transpose__test_1_1_test_convolution_transpose_1abfe1dd33276e5a6db311ca618da2a94e)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,gc,dc)` |
`public def test_convolution_transpose_separate_stride_pad_adj_gradient(self,`[`stride_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5ef04ac962f966f71f9ba0bf6a59501a)`,`[`stride_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1ad0453f3d1a62871ceec81f5555a183df)`,`[`pad_t`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7bc2a5c5f7adc55d99a41f554231ee50)`,`[`pad_l`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0d8c2f26cc669499c9e9fe358f40954a)`,`[`pad_b`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7abe4791046feea73ab8a8cc6865ed90)`,`[`pad_r`](#classconv__transpose__test_1_1_test_convolution_transpose_1a1a86164897744efccdc5057bd666cefb)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a9b21589e93d70ebda55f0475ca29e249)`,`[`adj_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5afff37b8de7f85db6d020abdca84a19)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`order`](#classconv__transpose__test_1_1_test_convolution_transpose_1abfe1dd33276e5a6db311ca618da2a94e)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,gc,dc)` |

## Members

#### `public def test_convolution_transpose_layout(self,`[`stride`](#classconv__transpose__test_1_1_test_convolution_transpose_1a8e9fe1f03a25c4dd85bf144a4d7c8dc2)`,`[`pad`](#classconv__transpose__test_1_1_test_convolution_transpose_1a342f6b81c783a263b70a2471bb200fbd)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj`](#classconv__transpose__test_1_1_test_convolution_transpose_1afc7581d0c041d694e3d0a9b1bc08c790)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,`[`shared_buffer`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7a1a92ae6f04b418dade17e6649e96ea)`,gc,dc)` {#classconv__transpose__test_1_1_test_convolution_transpose_1adfbc15a8eea2679234823dce980b0f55}





#### `public def test_convolution_transpose_separate_stride_pad_adj_layout(self,`[`stride_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5ef04ac962f966f71f9ba0bf6a59501a)`,`[`stride_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1ad0453f3d1a62871ceec81f5555a183df)`,`[`pad_t`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7bc2a5c5f7adc55d99a41f554231ee50)`,`[`pad_l`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0d8c2f26cc669499c9e9fe358f40954a)`,`[`pad_b`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7abe4791046feea73ab8a8cc6865ed90)`,`[`pad_r`](#classconv__transpose__test_1_1_test_convolution_transpose_1a1a86164897744efccdc5057bd666cefb)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a9b21589e93d70ebda55f0475ca29e249)`,`[`adj_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5afff37b8de7f85db6d020abdca84a19)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,gc,dc)` {#classconv__transpose__test_1_1_test_convolution_transpose_1aadba06f43543213417157ff707dcb6ad}





#### `public def test_convolution_transpose_gradients(self,`[`stride`](#classconv__transpose__test_1_1_test_convolution_transpose_1a8e9fe1f03a25c4dd85bf144a4d7c8dc2)`,`[`pad`](#classconv__transpose__test_1_1_test_convolution_transpose_1a342f6b81c783a263b70a2471bb200fbd)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj`](#classconv__transpose__test_1_1_test_convolution_transpose_1afc7581d0c041d694e3d0a9b1bc08c790)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`order`](#classconv__transpose__test_1_1_test_convolution_transpose_1abfe1dd33276e5a6db311ca618da2a94e)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,gc,dc)` {#classconv__transpose__test_1_1_test_convolution_transpose_1a26916ae4dc09f19df8414c6b00e56a37}





#### `public def test_convolution_transpose_separate_stride_pad_adj_gradient(self,`[`stride_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5ef04ac962f966f71f9ba0bf6a59501a)`,`[`stride_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1ad0453f3d1a62871ceec81f5555a183df)`,`[`pad_t`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7bc2a5c5f7adc55d99a41f554231ee50)`,`[`pad_l`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0d8c2f26cc669499c9e9fe358f40954a)`,`[`pad_b`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7abe4791046feea73ab8a8cc6865ed90)`,`[`pad_r`](#classconv__transpose__test_1_1_test_convolution_transpose_1a1a86164897744efccdc5057bd666cefb)`,`[`kernel`](#classconv__transpose__test_1_1_test_convolution_transpose_1a92c29215d00be63be80562d640344630)`,`[`adj_h`](#classconv__transpose__test_1_1_test_convolution_transpose_1a9b21589e93d70ebda55f0475ca29e249)`,`[`adj_w`](#classconv__transpose__test_1_1_test_convolution_transpose_1a5afff37b8de7f85db6d020abdca84a19)`,`[`size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0ba1c41433e0e6444810597ffdeea3f9)`,`[`input_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a7b867ff2152cf0b4f5966fb84b15e4b5)`,`[`output_channels`](#classconv__transpose__test_1_1_test_convolution_transpose_1a0170c5e174f2d9f26ac78e4a6e436d4d)`,`[`batch_size`](#classconv__transpose__test_1_1_test_convolution_transpose_1a54575a9a0151b3dfaf18684876c94b1f)`,`[`order`](#classconv__transpose__test_1_1_test_convolution_transpose_1abfe1dd33276e5a6db311ca618da2a94e)`,`[`engine`](#classconv__transpose__test_1_1_test_convolution_transpose_1ae319bba99a852573cd187543745cad64)`,gc,dc)` {#classconv__transpose__test_1_1_test_convolution_transpose_1aeedcef1b5a24164002b70da81a9620d6}





# namespace `convnet_benchmarks_test` {#namespaceconvnet__benchmarks__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`convnet_benchmarks_test::TestConvnetBenchmarks`](#classconvnet__benchmarks__test_1_1_test_convnet_benchmarks)    |
# class `convnet_benchmarks_test::TestConvnetBenchmarks` {#classconvnet__benchmarks__test_1_1_test_convnet_benchmarks}

```
class convnet_benchmarks_test::TestConvnetBenchmarks
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testConvnetBenchmarks(self)` |

## Members

#### `public def testConvnetBenchmarks(self)` {#classconvnet__benchmarks__test_1_1_test_convnet_benchmarks_1ac3f15478172e1ad89def6bd699c5c13a}





# namespace `copy_ops_test` {#namespacecopy__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`copy_ops_test::CopyOpsTest`](#classcopy__ops__test_1_1_copy_ops_test)    |
# class `copy_ops_test::CopyOpsTest` {#classcopy__ops__test_1_1_copy_ops_test}

```
class copy_ops_test::CopyOpsTest
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def run_test_copy_gradient(self,device_opt)` |
`public def test_copy_gradient_cpu(self)` |
`public def test_copy_gradient_gpu(self)` |
`public def test_copy_gradient_multiple_gpus(self)` |
`public def test_cpu2gpu_gpu2cpu_gradients(self)` |

## Members

#### `public def run_test_copy_gradient(self,device_opt)` {#classcopy__ops__test_1_1_copy_ops_test_1a47e75f56340f6c988f0971159b8aa1d7}





#### `public def test_copy_gradient_cpu(self)` {#classcopy__ops__test_1_1_copy_ops_test_1a94bfc18e38ac122ce2241e5e9a9293ad}





#### `public def test_copy_gradient_gpu(self)` {#classcopy__ops__test_1_1_copy_ops_test_1a4db1a321daefb022cfec6c3b20648b18}





#### `public def test_copy_gradient_multiple_gpus(self)` {#classcopy__ops__test_1_1_copy_ops_test_1a5b579ca807f36cf6a7915654bed488b3}





#### `public def test_cpu2gpu_gpu2cpu_gradients(self)` {#classcopy__ops__test_1_1_copy_ops_test_1a486b15562343f91836947db20ef1064f}





# namespace `core` {#namespacecore}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`core::BlobReference`](#classcore_1_1_blob_reference)    |
`class `[`core::DataType`](#classcore_1_1_data_type)    |
`class `[`core::ExecutionStep`](#classcore_1_1_execution_step)    |
`class `[`core::GradientRegistry`](#classcore_1_1_gradient_registry)    |
`class `[`core::IR`](#classcore_1_1_i_r)    |
`class `[`core::Net`](#classcore_1_1_net)    |
`class `[`core::Plan`](#classcore_1_1_plan)    |
# class `core::BlobReference` {#classcore_1_1_blob_reference}

```
class core::BlobReference
  : public object
```  



A wrapper around a blob in a net.

BlobReference gives us a way to refer to the network that the blob is
generated from. Note that blobs are, essentially, just strings in the
current workspace.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  meta` |
`public def __init__(self,name,net)` |
`public def __hash__(self)` |
`public def __eq__(self,other)` |
`public def __ne__(self,other)` |
`public def __str__(self)` |
`public def __repr__(self)` |
`public def __add__(self,other)` |
`public def __radd__(self,other)` |
`public def Net(self)` |
`public def GetNameScope(self)` |
`public def __getattr__(self,op_type)` |

## Members

#### `public  meta` {#classcore_1_1_blob_reference_1a3275563c827e17d32ce9e03a903ce6ba}





#### `public def __init__(self,name,net)` {#classcore_1_1_blob_reference_1a640d2608dd846636ec684c792905c10a}



Initializes a blob reference.

Note that this does not prepends the namescope. If needed, use
ScopedBlobReference() to prepend the existing namespace.

#### `public def __hash__(self)` {#classcore_1_1_blob_reference_1a28142363c69497a4206fd573b29ca2f5}





#### `public def __eq__(self,other)` {#classcore_1_1_blob_reference_1a9fdc02793948d730b55cc45af3cc32e1}





#### `public def __ne__(self,other)` {#classcore_1_1_blob_reference_1a6fd4be73c80688ed08622d3ef0eb0e1e}





#### `public def __str__(self)` {#classcore_1_1_blob_reference_1a7210df8eb96809e874c94f062309f6b8}





#### `public def __repr__(self)` {#classcore_1_1_blob_reference_1ac4b27f570c0f6b575463fdf4e9d0d839}





#### `public def __add__(self,other)` {#classcore_1_1_blob_reference_1a039e632271fde21cfb21dc25e9f67502}





#### `public def __radd__(self,other)` {#classcore_1_1_blob_reference_1a4fc84a172c70b7ccc263d6c963bdbaa0}





#### `public def Net(self)` {#classcore_1_1_blob_reference_1a467ef92d6b71a4e201a3cfaa33a77cc9}





#### `public def GetNameScope(self)` {#classcore_1_1_blob_reference_1aec47fedae4213f43432ff500ba1dc4d0}





#### `public def __getattr__(self,op_type)` {#classcore_1_1_blob_reference_1acce5e1da50581ce1ade246a9d1cd0e83}



A wrapper allowing one to initiate operators from a blob reference.

Example: for a blob reference b that comes from network n, doing
    b.Relu(...)
is equivalent to doing
    net.Relu([b], ...)

# class `core::DataType` {#classcore_1_1_data_type}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `core::ExecutionStep` {#classcore_1_1_execution_step}

```
class core::ExecutionStep
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,name,nets,num_iter)` |
`public def get_net(self,name)` |
`public def Name(self)` |
`public def __str__(self)` |
`public def Proto(self)` |
`public def HasNets(self)` |
`public def HasSubsteps(self)` |
`public def Nets(self)` |
`public def Substeps(self)` |
`public def SetIter(self,num_iter)` |
`public def SetOnlyOnce(self,only_once)` |
`public def SetShouldStopBlob(self,should_stop_blob)` |
`public def RunEveryMillis(self,interval)` |
`public def SetReportNet(self,report_net,report_interval)` |
`public def AddSubstep(self,substep)` |
`public def SetConcurrentSubsteps(self,concurrent_substeps)` |
`public def AddNet(self,net)` |
`public def get_all_attributes(self,name)` |

## Members

#### `public def __init__(self,name,nets,num_iter)` {#classcore_1_1_execution_step_1ac332824c1df7adcd987fe24ae13fbc79}





#### `public def get_net(self,name)` {#classcore_1_1_execution_step_1a539450df2e3565ba46f49ed0a8e3e6fa}





#### `public def Name(self)` {#classcore_1_1_execution_step_1a996006c3ca33372d00de36814b8eafb9}





#### `public def __str__(self)` {#classcore_1_1_execution_step_1a02d4bad765f97ae83d861c4c8567fa5f}





#### `public def Proto(self)` {#classcore_1_1_execution_step_1ad3d977d5dfd18ca69c02ecbb6d8852e1}





#### `public def HasNets(self)` {#classcore_1_1_execution_step_1acc51723b0a73f00a873490396c169d20}





#### `public def HasSubsteps(self)` {#classcore_1_1_execution_step_1a3757f5875d41e8fe5c0393312e8de79c}





#### `public def Nets(self)` {#classcore_1_1_execution_step_1a2b8c7a5b7b0784820a90787fbb554b27}





#### `public def Substeps(self)` {#classcore_1_1_execution_step_1aab89388731255c15d7d901f42d1b15ba}





#### `public def SetIter(self,num_iter)` {#classcore_1_1_execution_step_1a897c9d9fa7369253ab4e0f0f1c804576}





#### `public def SetOnlyOnce(self,only_once)` {#classcore_1_1_execution_step_1a56a61743ab1886d9333682a09e561cf5}





#### `public def SetShouldStopBlob(self,should_stop_blob)` {#classcore_1_1_execution_step_1a576477ac4f71e65eac39eeb77efb4630}





#### `public def RunEveryMillis(self,interval)` {#classcore_1_1_execution_step_1a24f3ba35a5014a9df0c6dc26da86778c}



Run this step every interval millisecods, as long as its
siblings are still running. It is guaranteed that, after all
siblings finish, this step will run at least one.

This property is ignored for top-level ExecutionSteps.

#### `public def SetReportNet(self,report_net,report_interval)` {#classcore_1_1_execution_step_1a49a77533154befc82738d0b9b9d14c81}



DEPRECATED. Use RunEveryMillis instead.

#### `public def AddSubstep(self,substep)` {#classcore_1_1_execution_step_1a529d982d22f5a57073390158f9fc9bb7}





#### `public def SetConcurrentSubsteps(self,concurrent_substeps)` {#classcore_1_1_execution_step_1a22445a73f04bc82f94cb23610da1dd84}





#### `public def AddNet(self,net)` {#classcore_1_1_execution_step_1ad8868db186b8bab1260669d913ec7163}





#### `public def get_all_attributes(self,name)` {#classcore_1_1_execution_step_1a82b3270332a3d632065b5e9b2144edcc}



Return the list of all attributes under the given `name`, present in
all of the nets used in this execution step and its children.

# class `core::GradientRegistry` {#classcore_1_1_gradient_registry}

```
class core::GradientRegistry
  : public object
```  



GradientRegistry holds the mapping from operators to their gradients.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def RegisterGradient(cls,op_type)` |
`public def GetGradientForOp(cls,op,g_output)` |
`public def GetBackwardPass(cls,operators,ys)` |

## Members

#### `public def RegisterGradient(cls,op_type)` {#classcore_1_1_gradient_registry_1ac78dbd0e9c63985475d6ca72a5cacd4e}



A decorator for registering gradient mappings.

#### `public def GetGradientForOp(cls,op,g_output)` {#classcore_1_1_gradient_registry_1a05ddc48a38abb50f99dd4303bea6c12e}





#### `public def GetBackwardPass(cls,operators,ys)` {#classcore_1_1_gradient_registry_1a706fc62ec597b5e6c49569f9cb68bfe7}



Gets the backward pass for the list of operators.

Args:
    operators: a list of operators constituting the forward pass.
    ys: a list or a dictionary specifying what blobs we want to compute
derivatives of. If the input is a list, we will automatically
generate their gradients with all-one values; if the input is a
dictionary, for any dictionary entries that are not None, we'll
take the corresponding blobs as their gradients; for all those
that are None, we will auto-fill them with 1.
Returns:
    gradient_ops: a list of gradient operators to run.
    all_input_to_grads: a map from input to their corresponding
gradients.

# class `core::IR` {#classcore_1_1_i_r}

```
class core::IR
  : public object
```  



A simple IR class to keep track of all intermediate representations used
in the gradient computation.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  ssa` |
`public  input_usages` |
`public  frontier` |
`public  gradient_frontier` |
`public  gradient_generators` |
`public def __init__(self,operators)` |
`public def Play(self,op)` |
`public def CheckGradientOperatorInput(self,grad_op_input,g_output,fwd_op_idx,locally_generated_blobs)` |
`public def AppendSparseGenerators(self,sparse_generators)` |
`public def BuildGradientGenerators(self,fwd_op_idx,gradient_ops,g_output,g_input)` |
`public def DoGradientAccumulation(self,fwd_op_idx)` |
`public def GetBackwardPass(self,ys)` |

## Members

#### `public  ssa` {#classcore_1_1_i_r_1a7d3c0e121a0414bf6cde99cbf3e4a934}





#### `public  input_usages` {#classcore_1_1_i_r_1aa8a0b78cafa5e52bf7e6bb442e3fe696}





#### `public  frontier` {#classcore_1_1_i_r_1a609f2288e6bba939f7f3bbf3badb3133}





#### `public  gradient_frontier` {#classcore_1_1_i_r_1a621d9860fb5f7b08bf6982c96e894593}





#### `public  gradient_generators` {#classcore_1_1_i_r_1a20a9ab521c4d4554f0b50cbf93f60119}





#### `public def __init__(self,operators)` {#classcore_1_1_i_r_1a88033193fee257bc79d92d65efab7e6c}





#### `public def Play(self,op)` {#classcore_1_1_i_r_1a596b5653e73bc18c631431ae63c3fc10}



"Adds an op to the current IR, and update the internal states to
reflect the blobs and versions after the execution of the op.

#### `public def CheckGradientOperatorInput(self,grad_op_input,g_output,fwd_op_idx,locally_generated_blobs)` {#classcore_1_1_i_r_1a9c8ef53797f3a0601e7d741927851122}



Checks if the gradient operators can be correctly carried out.

#### `public def AppendSparseGenerators(self,sparse_generators)` {#classcore_1_1_i_r_1a076c61035d2825b393bea069160b389e}





#### `public def BuildGradientGenerators(self,fwd_op_idx,gradient_ops,g_output,g_input)` {#classcore_1_1_i_r_1a2fe337e136e7a0c259eaa7a466c18b97}



Updates gradient_generators and gradient_frontier

#### `public def DoGradientAccumulation(self,fwd_op_idx)` {#classcore_1_1_i_r_1a507a82eb945ef3ab9878472a29acf244}



For each input name in the forward op, check if we will need to
add gradient accumulation. If so, do gradient accumulation and return
the list of gradient operators.

The criteria for doing gradient accumulation is:
(1) the specific input version has been used by multiple operators.
(2) the current fwd_op_idx is the first to use that input, i.e. in the
    backward pass, is the last to optionally generate the gradient for
    the op.
(3) For the operators that used the input, their gradient operators
    have generated more than 1 gradient.

When accumulating operators, our current solution is to rename all the
created gradients with an internal intermediate name, and then add a
Sum() operator that adds up all the gradients. This may use more memory
due to intermediate storage, but is usually the fastest approach as one
can do one single sum for multiple intermediate gradients.

#### `public def GetBackwardPass(self,ys)` {#classcore_1_1_i_r_1af64e8de2e61caa941bccf6dfb1ecfb76}



Gets the backward pass that computes the derivatives of given blobs.

Inputs:
  ys: a list or a dictionary specifying what blobs we want to compute
      derivatives of. If the input is a list, we will automatically
      generate their gradients with all-one values; if the input is a
      dictionary, for any dictionary entries that are not None, we will
      take the corresponding blobs as their gradients; for all those
      that are None, we will auto-fill them with 1.

# class `core::Net` {#classcore_1_1_net}

```
class core::Net
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,name_or_proto)` |
`public def AppendNet(self,net)` |
`public def LogInfo(self,msg_or_blobs)` |
`public def add_attribute(self,name,obj)` |
`public def get_attributes(self,name)` |
`public def set_rand_seed(self,seed,sequence_seed,seed_on_op_def)` |
`public def Name(self)` |
`public def __str__(self)` |
`public def Const(self,array,blob_out,dtype)` |
`public def BlobIsDefined(self,blob)` |
`public def UsesBlob(self,blob)` |
`public def GetBlobRef(self,blob_name)` |
`public def Clone(self,name,blob_remap,op_id_mask,remap_funcs,keep_schema)` |
`public def ClonePartial(self,name,inputs,outputs,remap_funcs)` |
`public def Proto(self)` |
`public def NextScopedBlob(self,prefix)` |
`public def NextBlob(self,prefix)` |
`public def NextName(self,prefix,output_id)` |
`public def AddGradientOperators(self,ys,skip)` |
`public def AddExternalInput(self,inputs)` |
`public def AddExternalOutput(self,outputs)` |
`public def AddScopedExternalInputs(self,inputs)` |
`public def AddScopedExternalOutputs(self,outputs)` |
`public def external_inputs(self)` |
`public def external_outputs(self)` |
`public def set_input_record(self,`[`input_record`](#classcore_1_1_net_1a72f411999e5c4060b872fe33ce0af05d)`)` |
`public def set_output_record(self,record)` |
`public def AppendOutputRecordField(self,field_name,record)` |
`public def input_record(self)` |
`public def output_record(self)` |
`public def AddExternalInputs(self,inputs)` |
`public def AddExternalOutputs(self,outputs)` |
`public def DeduplicateGradientSlices(self,g,aggregator)` |
`public def RunAllOnGPU(self,gpu_id,use_cudnn)` |
`public def __getattr__(self,op_type)` |
`public def Python(self,f,grad_f,pass_workspace)` |

## Members

#### `public def __init__(self,name_or_proto)` {#classcore_1_1_net_1a9f6d3e34217fc0324ce8fefcd7d39549}



Create a Net.
Args:
    name_or_proto:  If a NetDef is provided, clone it. Otherwise,
            create an empty net with the given name.

#### `public def AppendNet(self,net)` {#classcore_1_1_net_1a431aa2378e74547410c2edcc08ad789e}





#### `public def LogInfo(self,msg_or_blobs)` {#classcore_1_1_net_1ae75ef724fda39c963ac36eaa8fedecff}





#### `public def add_attribute(self,name,obj)` {#classcore_1_1_net_1a57e16fd50058bb7dad87e5e1b00e0472}



Add `obj` to the list of attributes in this net under the given `name`.
Attributes are user-defined objects and have no pre-defined semantics.

#### `public def get_attributes(self,name)` {#classcore_1_1_net_1a83127fbbc42d1dd6095cae912ffcd801}



Returns the list of attributes in this net for a given `name`.
Attributes are user-defined objects added with `add_attribute'.

#### `public def set_rand_seed(self,seed,sequence_seed,seed_on_op_def)` {#classcore_1_1_net_1a9432c6758f6ae1dc4123da30c494267b}



Adds a random seed to each op in the net.
If sequence_seed is set, the i-th op has rand_seed=`seed + i`
If seed_on_op_def is set, the op rand_seed=hash(str(op))
sequence_seed and seed_on_op_def cannot be both set to True.

#### `public def Name(self)` {#classcore_1_1_net_1a46f5f9d76d344b0ce1ba56a5e2d30207}





#### `public def __str__(self)` {#classcore_1_1_net_1aedf27d311dcc4a727ccb0386547a5e93}





#### `public def Const(self,array,blob_out,dtype)` {#classcore_1_1_net_1aca89f7c4b2e8e2cac73a00625cbb2d75}





#### `public def BlobIsDefined(self,blob)` {#classcore_1_1_net_1abccc9ed5d30c5e85590b94c189b8eb5f}



Returns true if the given BlobReference is produced as output of
an operator in this net, or if it is provided as an external input.

#### `public def UsesBlob(self,blob)` {#classcore_1_1_net_1ab412fdb18c6f7aa655f6098139d4e84d}



Returns true iff the given BlobReference is used by any operator
or this net, or if it is one of the external inputs of the net.

#### `public def GetBlobRef(self,blob_name)` {#classcore_1_1_net_1a4426babfd48d88042aa123bb022cf9f8}



Given the name of a blob produced by this net, return a BlobReference
to it. If the blob is not produced by any op in this net,
raises KeyError.

#### `public def Clone(self,name,blob_remap,op_id_mask,remap_funcs,keep_schema)` {#classcore_1_1_net_1a3281e3cc6f095229c7863fcd6c62fd3f}



Clone this net.
Args:
    name:        name of the cloned net
    blob_remap:  optional map with list of blob names to replace
    op_id_mask:  optional list of operator indices to include in
         the cloned net. If not provided, all ops are included.

#### `public def ClonePartial(self,name,inputs,outputs,remap_funcs)` {#classcore_1_1_net_1a97e540af146e766ee704b0a05fcb08f2}



Clone this net, including only ops that are necessary in order to
compute `outputs` given `inputs`. Return references to the cloned
outputs. Internal blobs (blobs that are produced and consumed inside
the net but not used as outputs) will be remapped to avoid name
conflict.

Args:
    name:    the name of the cloned net
    inputs:  map where the keys correspond to BlobReferences in the
     original net, and the values correspond to external inputs
     in the partially cloned net. If `inputs` is a list, don't
     remap input names.
    outputs: outputs to be produced by the cloned net.

Returns:
    Tuple (new_net, new_outputs)
new_net:       a new Net object.
new_outputs:   list of BlobReferences corresponding to the
               outputs produced by new_net.

#### `public def Proto(self)` {#classcore_1_1_net_1a14fea745967c8177b271a61ccfe81b75}





#### `public def NextScopedBlob(self,prefix)` {#classcore_1_1_net_1abb0495bd3a568a20a1fa3035bf11e470}



Return the blob that has not been defined or registered in the
current net. It returns `ScopedBlobReference(prefix)`, if it's valid,
otherwise `ScopedBlobReference(prefix) + '_auto_' + ?`. Different calls
is guaranteed to return blob with different names.

#### `public def NextBlob(self,prefix)` {#classcore_1_1_net_1a071e2cb04040122401f47aab1710af69}



Return the blob that has not been defined or registered in the
current net. It returns `BlobReference(prefix)`, if it's valid,
otherwise `BlobReference(prefix) + '_auto_' + ?`. Different calls
is guaranteed to return blob with different names.

#### `public def NextName(self,prefix,output_id)` {#classcore_1_1_net_1ad97867443ef6c922b32a7ddb62b19c79}



Returns the next name to be used, if you do not want to explicitly
name your blob. [Deprecated, use NextBlob, NextScopedBlob instead]

#### `public def AddGradientOperators(self,ys,skip)` {#classcore_1_1_net_1a62e9740947d7ef041a932c42d9aeab54}



Add the gradient for operators in the net.

Inputs:
  ys: a list or a dictionary specifying what blobs we want to compute
      derivatives of. If the input is a list, we will automatically
      generate their gradients with all-one values; if the input is a
      dictionary, for any dictionary entries that are not None, we will
      take the corresponding blobs as their gradients; for all those
      that are None, we will auto-fill them with 1.
  skip: skips the first n operators. This is provided mainly because a
      lot of nets may use the first few operators for data generation
      like stuff which really do not need to have gradients.

Outputs:
  returns a map from the blob name in the input network to a blob
  containing gradient or a GradientSlice in case of sparse gradient

Currently, this is hard-coded for float operators if there are branches
(i.e. a blob is used as input to multiple operators). This is because
the gradient accumulation (Sum) is float only right now.

#### `public def AddExternalInput(self,inputs)` {#classcore_1_1_net_1a2e0accc1c4361b682ab1df39a9397ca0}





#### `public def AddExternalOutput(self,outputs)` {#classcore_1_1_net_1aa1365a10e86dfa11180fe9d7e4b81b24}





#### `public def AddScopedExternalInputs(self,inputs)` {#classcore_1_1_net_1a03ef219b092bedb4a47f6b4caa2612f0}





#### `public def AddScopedExternalOutputs(self,outputs)` {#classcore_1_1_net_1a2a3256edd76eabe32f3b2062cf9268ff}





#### `public def external_inputs(self)` {#classcore_1_1_net_1a5f01a6ca87d4a219a16a2dfdae2fc7db}





#### `public def external_outputs(self)` {#classcore_1_1_net_1aba685e0bee40b0289e5e6be6defeebe4}





#### `public def set_input_record(self,`[`input_record`](#classcore_1_1_net_1a72f411999e5c4060b872fe33ce0af05d)`)` {#classcore_1_1_net_1ad61d7177e84a41bdf2c928fcb2cfe557}





#### `public def set_output_record(self,record)` {#classcore_1_1_net_1a590dfbd15788069b949a1a202ae48b2c}





#### `public def AppendOutputRecordField(self,field_name,record)` {#classcore_1_1_net_1a3943e942d6f272fc08216b8c029417c1}





#### `public def input_record(self)` {#classcore_1_1_net_1a72f411999e5c4060b872fe33ce0af05d}





#### `public def output_record(self)` {#classcore_1_1_net_1a60cf5c177579f957e46b0ad618fa18ea}





#### `public def AddExternalInputs(self,inputs)` {#classcore_1_1_net_1a0e39cf9a717915e42386969a3133a36a}





#### `public def AddExternalOutputs(self,outputs)` {#classcore_1_1_net_1ab94a4ea87500e67d58bce9ad8c6d2b38}





#### `public def DeduplicateGradientSlices(self,g,aggregator)` {#classcore_1_1_net_1afa29ddd8c096d859e1894def8889deee}





#### `public def RunAllOnGPU(self,gpu_id,use_cudnn)` {#classcore_1_1_net_1a3fa040cf7337c4a818925dd83d32d2cf}



A convenient function to run everything on the GPU.

#### `public def __getattr__(self,op_type)` {#classcore_1_1_net_1a50fd7c755ea87c06fcf2887e1e9b3bcc}





#### `public def Python(self,f,grad_f,pass_workspace)` {#classcore_1_1_net_1a2a5ab3b04de28feef0a9f36b3b4a444c}



Registers and returns a python operator.

`f` and `f_grad` can be one of the following:
    - a function with signature (inputs, outputs), where inputs and
      outputs are a list of CPUTensor objects. This function will be
      called from C++ everytime the operator is executed.
    - a tuple (func, args, kwargs), here `func` is a callable, args is
      an argument list, and kwargs is a dict list. The call:
  f = func(*args, kwargs)
      will be performed locally at node initialization time, on all of
      the nodes of the job, returning `f`, a callable that will be used
      as the python operator function to be called during Net execution.
      This is to be used when using python operator in a distributed
      context, and allows to create and keep local python state across
      calls to the operator.

If `pass_workspace` is True, the signature is changed to
(inputs, outputs, workspace) where `workspace` is the workspace the op
is going to run on. This is potentially dangerous (as the op can
manipulate the workspace directly), use on your own risk.

# class `core::Plan` {#classcore_1_1_plan}

```
class core::Plan
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,name_or_step)` |
`public def __str__(self)` |
`public def Proto(self)` |
`public def AddNets(self,nets)` |
`public def Nets(self)` |
`public def AddStep(self,step)` |
`public def get_all_attributes(self,name)` |

## Members

#### `public def __init__(self,name_or_step)` {#classcore_1_1_plan_1aa35729dadf142fb20e4965e9e804e41e}





#### `public def __str__(self)` {#classcore_1_1_plan_1aeaf612b77e81b817a8b76fd2a7d86fb9}





#### `public def Proto(self)` {#classcore_1_1_plan_1a9b365f97d710c546ac4be367ade74583}





#### `public def AddNets(self,nets)` {#classcore_1_1_plan_1a5a73d6e6130efcde509425fbd639bace}





#### `public def Nets(self)` {#classcore_1_1_plan_1a56fb576ee34dc6f7fd119c74396533af}





#### `public def AddStep(self,step)` {#classcore_1_1_plan_1a8970d914a65f7da5c990f60bff3860e1}





#### `public def get_all_attributes(self,name)` {#classcore_1_1_plan_1ad90a0adf6f3edead2cf0d3b7b0300a5d}



Return the list of all attributes under the given `name`, present in
all of the nets used in this plan.

# namespace `core_gradients_test` {#namespacecore__gradients__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`core_gradients_test::TestGradientCalculation`](#classcore__gradients__test_1_1_test_gradient_calculation)    |
`class `[`core_gradients_test::TestGradientsAccumulationWithNoGradientOps`](#classcore__gradients__test_1_1_test_gradients_accumulation_with_no_gradient_ops)    |
`class `[`core_gradients_test::TestGradientsAccumulationWithPassThroughGradients`](#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients)    |
`class `[`core_gradients_test::TestSparseGradientsAccumulation`](#classcore__gradients__test_1_1_test_sparse_gradients_accumulation)    |
# class `core_gradients_test::TestGradientCalculation` {#classcore__gradients__test_1_1_test_gradient_calculation}

```
class core_gradients_test::TestGradientCalculation
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testDirect(self,`[`device_option`](#classcore__gradients__test_1_1_test_gradient_calculation_1a4124ee867057d65f5504d657fff86e6a)`)` |
`public def testDirectImplicitGradientSource(self)` |
`public def testDoesNotGenerateUnnecessaryGradients(self)` |
`public def testDirectButNoOutputGradientGiven(self)` |
`public def testDirectInPlace(self)` |
`public def testUseOutput(self)` |
`public def testUseOutputInPlace(self)` |
`public def testUseOutputButOutputHasBeenChanged(self)` |
`public def testUseInput(self)` |
`public def testUseInputButInputHasBeenChanged(self)` |
`public def testMultiUseInput(self,`[`device_option`](#classcore__gradients__test_1_1_test_gradient_calculation_1a4124ee867057d65f5504d657fff86e6a)`)` |
`public def testMultiUseInputButWithNoGradient(self)` |
`public def testMultiUseInputAndMultipleVersions(self)` |
`public def testMultiUseInputAndMultipleVersionsBig(self)` |
`public def testGradientMappingUsingSumOp(self)` |
`public def testGradientCalculationWithPrint(self)` |
`public def testStopGradient(self)` |
`public def testStopGradientInplace(self)` |
`public def testStopGradientWithMultiUseOperators(self)` |

## Members

#### `public def testDirect(self,`[`device_option`](#classcore__gradients__test_1_1_test_gradient_calculation_1a4124ee867057d65f5504d657fff86e6a)`)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a7cb16508994acd2afbf9625d9ab63020}





#### `public def testDirectImplicitGradientSource(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a2c4f0022a4adb5b72b1070599085ffa3}





#### `public def testDoesNotGenerateUnnecessaryGradients(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a428c502dc0df397de467f8d85c74f37e}





#### `public def testDirectButNoOutputGradientGiven(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1acaf4169921d3fc7f3c8b63d8e976f699}





#### `public def testDirectInPlace(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a8dca671b431b4e949f2d1336f622fc1a}





#### `public def testUseOutput(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a21037e7907e4009a8aafb610f480f2c7}





#### `public def testUseOutputInPlace(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a4fcf61478197650a29741770d9321132}





#### `public def testUseOutputButOutputHasBeenChanged(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a854ba52510b87441fcf424c883f477a3}





#### `public def testUseInput(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a56d925e5e00680bbe67a0da6cecbdd63}





#### `public def testUseInputButInputHasBeenChanged(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a79dcb3eb96a2794e20b7e2aa6a62f6ab}



Test gradient for the following case:

in -> out, with UseInput
in -> in

Since we overwrite in in op#1, but in will be needed by the gradient
calculation of op#0, the gradient registry should raise an error.

#### `public def testMultiUseInput(self,`[`device_option`](#classcore__gradients__test_1_1_test_gradient_calculation_1a4124ee867057d65f5504d657fff86e6a)`)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a63405e3156723f134638b150d0905ae0}



Test gradient for the following case:

in -> hidden1
in -> hidden2
hidden1, hidden2 -> out

#### `public def testMultiUseInputButWithNoGradient(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a8567a4d70bc871ec7a35903ff0d4aaee}



Test gradient for the following case:

in -> hidden1
in -(no gradient)-> hidden2
hidden1, hidden2 -> out

#### `public def testMultiUseInputAndMultipleVersions(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1ae6b6fe9bb9c60e8b2eb561118874a362}



Test gradient for the following case:

in -> in
in -> hidden1, hidden2
hidden1, hidden2 -> out

#### `public def testMultiUseInputAndMultipleVersionsBig(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a074f4523743618d5208c14d5f38e2671}



Test gradient for the following case:

in -> in
in -> hidden1, hidden2
hidden1, hidden2 -> in
in -> hidden3, hidden4, hidden5
hidden3, hidden4, hidden5 -> out

#### `public def testGradientMappingUsingSumOp(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1aebe8b67ab71050b98736d0e45776b431}



Since Sum is used in accumulating gradients, we will test if
it is OK to also explicitly use it in the graph.

#### `public def testGradientCalculationWithPrint(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a9039047272efe89c0dba44febc4c5238}



Test a common use case where we have Print in the forward pass.

#### `public def testStopGradient(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a29d96d02100dcd8c5cf04d647b882792}





#### `public def testStopGradientInplace(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1a356011270f7e9ab30f0162974b48a029}





#### `public def testStopGradientWithMultiUseOperators(self)` {#classcore__gradients__test_1_1_test_gradient_calculation_1aa711294003afe8f7b0986aec13dd6354}





# class `core_gradients_test::TestGradientsAccumulationWithNoGradientOps` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_no_gradient_ops}

```
class core_gradients_test::TestGradientsAccumulationWithNoGradientOps
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testNormalAccumulation(self)` |
`public def testAccumulationWithNoGradientBranch(self)` |

## Members

#### `public def testNormalAccumulation(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_no_gradient_ops_1ae08899cd3848cff0e3c773a13bdb83c8}





#### `public def testAccumulationWithNoGradientBranch(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_no_gradient_ops_1a596b51b9169f655d117c095c8c390733}





# class `core_gradients_test::TestGradientsAccumulationWithPassThroughGradients` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients}

```
class core_gradients_test::TestGradientsAccumulationWithPassThroughGradients
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testAddOpInMiddle(self)` |
`public def testSubOpInMiddle(self)` |
`public def testAddOpAtLeaf(self)` |
`public def testSubOpAtLeaf(self)` |
`public def testMultiLayerAddOps(self)` |
`public def testMultiLayerSubOps(self)` |

## Members

#### `public def testAddOpInMiddle(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients_1abd68fb41d48f74c2db53381a3a3c0fb8}





#### `public def testSubOpInMiddle(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients_1a53ab5f66bb9dbc16b846a649a3a3ed1a}





#### `public def testAddOpAtLeaf(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients_1a95205a40a10a23a158eebc4745e79686}





#### `public def testSubOpAtLeaf(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients_1a2d421b9f3b6e8de7beb2882877f1d28e}





#### `public def testMultiLayerAddOps(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients_1ad9eca27199cbf0221fceb465001e0e04}





#### `public def testMultiLayerSubOps(self)` {#classcore__gradients__test_1_1_test_gradients_accumulation_with_pass_through_gradients_1a25aca3696f1d769f5e4aa1c5e9920441}





# class `core_gradients_test::TestSparseGradientsAccumulation` {#classcore__gradients__test_1_1_test_sparse_gradients_accumulation}

```
class core_gradients_test::TestSparseGradientsAccumulation
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testSparseAccumulationWithValues(self)` |
`public def testSparseGradientToDense(self)` |
`public def testSparseAccumulationWithIndicesAndValues(self)` |

## Members

#### `public def testSparseAccumulationWithValues(self)` {#classcore__gradients__test_1_1_test_sparse_gradients_accumulation_1a05e38cb02ac5b22fb702531f2f326520}





#### `public def testSparseGradientToDense(self)` {#classcore__gradients__test_1_1_test_sparse_gradients_accumulation_1abe02a7ceb1fd3c6fcd9c6e7f236960e1}





#### `public def testSparseAccumulationWithIndicesAndValues(self)` {#classcore__gradients__test_1_1_test_sparse_gradients_accumulation_1a3cb2e3d8ce53c5db3e648c9781f0d39d}





# namespace `core_test` {#namespacecore__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`core_test::TestAutoNaming`](#classcore__test_1_1_test_auto_naming)    |
`class `[`core_test::TestCloneNet`](#classcore__test_1_1_test_clone_net)    |
`class `[`core_test::TestCreateOperator`](#classcore__test_1_1_test_create_operator)    |
`class `[`core_test::TestScopes`](#classcore__test_1_1_test_scopes)    |
# class `core_test::TestAutoNaming` {#classcore__test_1_1_test_auto_naming}

```
class core_test::TestAutoNaming
  : public test_util.TestCase
```  



Test that operators are named with different names, and that automatically
named blob names don't clash intra or inter networks.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_next_blob(self)` |
`public def test_auto_naming(self)` |

## Members

#### `public def test_next_blob(self)` {#classcore__test_1_1_test_auto_naming_1afaedebaf7041797ecde35abb7fc5ddc7}





#### `public def test_auto_naming(self)` {#classcore__test_1_1_test_auto_naming_1ae09606426c2c60342b84270616e90330}





# class `core_test::TestCloneNet` {#classcore__test_1_1_test_clone_net}

```
class core_test::TestCloneNet
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testPartialClone(self)` |

## Members

#### `public def testPartialClone(self)` {#classcore__test_1_1_test_clone_net_1af6e84f871481041b70de2a52bcf3b716}





# class `core_test::TestCreateOperator` {#classcore__test_1_1_test_create_operator}

```
class core_test::TestCreateOperator
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testCreate(self)` |
`public def testCreateWithNoneKwarg(self)` |

## Members

#### `public def testCreate(self)` {#classcore__test_1_1_test_create_operator_1a23fb48a378ebdbf51f6c835bebace5c6}





#### `public def testCreateWithNoneKwarg(self)` {#classcore__test_1_1_test_create_operator_1a186c6adebdb56f733a2ef3f5c4dffcba}





# class `core_test::TestScopes` {#classcore__test_1_1_test_scopes}

```
class core_test::TestScopes
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testBlobReferenceIsIndependentFromNameScope(self)` |
`public def testNameScopeWithOp(self)` |
`public def testNameScopeWithReset(self)` |
`public def testDeviceScope(self)` |
`public def testNameAndDeviceScopeTogether(self)` |

## Members

#### `public def testBlobReferenceIsIndependentFromNameScope(self)` {#classcore__test_1_1_test_scopes_1ac746e55428a8208e897217898e96c2f3}





#### `public def testNameScopeWithOp(self)` {#classcore__test_1_1_test_scopes_1a203eea94b05f6e0bbdaf06fb775da31d}





#### `public def testNameScopeWithReset(self)` {#classcore__test_1_1_test_scopes_1a0231b7000d2194a9d1e65dd7a441e6e4}





#### `public def testDeviceScope(self)` {#classcore__test_1_1_test_scopes_1ab5c1d3a83bd3f20916b79fd42e0649b1}





#### `public def testNameAndDeviceScopeTogether(self)` {#classcore__test_1_1_test_scopes_1a7abe10d2acc1b0e833403eff441a0bb7}





# namespace `cosine_embedding_criterion_op_test` {#namespacecosine__embedding__criterion__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`cosine_embedding_criterion_op_test::TestCosineEmbeddingCriterion`](#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion)    |
# class `cosine_embedding_criterion_op_test::TestCosineEmbeddingCriterion` {#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion}

```
class cosine_embedding_criterion_op_test::TestCosineEmbeddingCriterion
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_cosine_embedding_criterion(self,`[`N`](#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion_1a407c6d013294b1d860f32e43f859a90c)`,`[`seed`](#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion_1a3c31d641c71c3a76d59e01369786e4e1)`,`[`margin`](#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion_1a4c79fb4e52c49b0f5c8f03a9e6b6634b)`,gc,dc)` |

## Members

#### `public def test_cosine_embedding_criterion(self,`[`N`](#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion_1a407c6d013294b1d860f32e43f859a90c)`,`[`seed`](#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion_1a3c31d641c71c3a76d59e01369786e4e1)`,`[`margin`](#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion_1a4c79fb4e52c49b0f5c8f03a9e6b6634b)`,gc,dc)` {#classcosine__embedding__criterion__op__test_1_1_test_cosine_embedding_criterion_1a19a282678f10d8b5b6244f0ce01b9983}





# namespace `counter_ops_test` {#namespacecounter__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`counter_ops_test::TestCounterOps`](#classcounter__ops__test_1_1_test_counter_ops)    |
# class `counter_ops_test::TestCounterOps` {#classcounter__ops__test_1_1_test_counter_ops}

```
class counter_ops_test::TestCounterOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_counter_ops(self)` |

## Members

#### `public def test_counter_ops(self)` {#classcounter__ops__test_1_1_test_counter_ops_1a0b5fb6337e1326ad4e343df55599c0e9}





# namespace `cross_entropy_ops_test` {#namespacecross__entropy__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`cross_entropy_ops_test::TestCrossEntropyOps`](#classcross__entropy__ops__test_1_1_test_cross_entropy_ops)    |
# class `cross_entropy_ops_test::TestCrossEntropyOps` {#classcross__entropy__ops__test_1_1_test_cross_entropy_ops}

```
class cross_entropy_ops_test::TestCrossEntropyOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_sigmoid_cross_entropy_with_logits(self,`[`inputs`](#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1a0092246b7cbfb8dd83475012e3c22d49)`)` |
`public def test_soft_label_cross_entropy(self,`[`n`](#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1af0337ccb74f5ff705704e91d9f6f7205)`,`[`b`](#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1a51f9cdc767787eeb1f42779eb8c5af91)`,gc,dc)` |

## Members

#### `public def test_sigmoid_cross_entropy_with_logits(self,`[`inputs`](#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1a0092246b7cbfb8dd83475012e3c22d49)`)` {#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1af674143da08cb074515def76c8ddd9fe}





#### `public def test_soft_label_cross_entropy(self,`[`n`](#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1af0337ccb74f5ff705704e91d9f6f7205)`,`[`b`](#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1a51f9cdc767787eeb1f42779eb8c5af91)`,gc,dc)` {#classcross__entropy__ops__test_1_1_test_cross_entropy_ops_1a482822717059e0eac129ac0d76c63448}





# namespace `data_parallel_model_test` {#namespacedata__parallel__model__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`data_parallel_model_test::GPUDataParallelModelTest`](#classdata__parallel__model__test_1_1_g_p_u_data_parallel_model_test)    |
`class `[`data_parallel_model_test::RecurrentNetworkParallelTest`](#classdata__parallel__model__test_1_1_recurrent_network_parallel_test)    |
`class `[`data_parallel_model_test::SparseDataParallelModelTest`](#classdata__parallel__model__test_1_1_sparse_data_parallel_model_test)    |
# class `data_parallel_model_test::GPUDataParallelModelTest` {#classdata__parallel__model__test_1_1_g_p_u_data_parallel_model_test}

```
class data_parallel_model_test::GPUDataParallelModelTest
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def run_model(self,gpu_devices)` |
`public def test_equiv(self)` |

## Members

#### `public def run_model(self,gpu_devices)` {#classdata__parallel__model__test_1_1_g_p_u_data_parallel_model_test_1a4f9edec74587155d8f5c03bd26961ce1}



Helper function for test_equiv

#### `public def test_equiv(self)` {#classdata__parallel__model__test_1_1_g_p_u_data_parallel_model_test_1a6689d14d93be1d1c6e7d9bc3e88ce192}



Test that the model produces exactly same results given
total batchsize, independent of number of GPUs.

# class `data_parallel_model_test::RecurrentNetworkParallelTest` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test}

```
class data_parallel_model_test::RecurrentNetworkParallelTest
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  T` |
`public  batch_size` |
`public  input_dim` |
`public  hidden_dim` |
`public  batch_per_device` |
`public def run_model(self,gpu_devices)` |
`public def test_equiv_recurrent(self)` |

## Members

#### `public  T` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test_1a47d5cba8d6bcd5701587f118090b37e1}





#### `public  batch_size` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test_1acf035d0d958a06c61b24eef1e4c869e9}





#### `public  input_dim` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test_1ab687e63e54ef9da777885dc0ac9f6528}





#### `public  hidden_dim` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test_1a772e4f4e7f97c0f87f9fb8bd4e211c43}





#### `public  batch_per_device` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test_1a7a6a2dc4f5728d339fbf26101c077d4d}





#### `public def run_model(self,gpu_devices)` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test_1a0c4fcc26efbe2f73b9ca77accce8a726}



Helper function for test_equiv

#### `public def test_equiv_recurrent(self)` {#classdata__parallel__model__test_1_1_recurrent_network_parallel_test_1a0febcaff88c07a9a503edb26f191cdc8}



Test that the model produces exactly same results given
total batchsize, independent of number of GPUs.

# class `data_parallel_model_test::SparseDataParallelModelTest` {#classdata__parallel__model__test_1_1_sparse_data_parallel_model_test}

```
class data_parallel_model_test::SparseDataParallelModelTest
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  ITER` |
`public  LR` |
`public  vecs` |
`public def run_model(self,V,gpu_devices)` |
`public def test_equiv_sparse(self)` |

## Members

#### `public  ITER` {#classdata__parallel__model__test_1_1_sparse_data_parallel_model_test_1ac60fff2352c7e6161aee691b378fe5ae}





#### `public  LR` {#classdata__parallel__model__test_1_1_sparse_data_parallel_model_test_1a9e77f60ff455a7cd2b67ae07d8497a18}





#### `public  vecs` {#classdata__parallel__model__test_1_1_sparse_data_parallel_model_test_1a83e26574509841cf5da9c9a070ffba8e}





#### `public def run_model(self,V,gpu_devices)` {#classdata__parallel__model__test_1_1_sparse_data_parallel_model_test_1a3d91280738288ad97d73a3de7a6aa163}



Helper function for test_equiv

#### `public def test_equiv_sparse(self)` {#classdata__parallel__model__test_1_1_sparse_data_parallel_model_test_1ad82b4c57baad3f9748ce71427cdf0280}



Test that the model produces exactly same results given
    total batchsize, independent of number of GPUs.

# namespace `data_workers` {#namespacedata__workers}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`data_workers::DataInputCoordinator`](#classdata__workers_1_1_data_input_coordinator)    |
`class `[`data_workers::GlobalCoordinator`](#classdata__workers_1_1_global_coordinator)    |
# class `data_workers::DataInputCoordinator` {#classdata__workers_1_1_data_input_coordinator}

```
class data_workers::DataInputCoordinator
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,net,input_blob_names,batch_size,device_option,namescope,input_source_name,queue)` |
`public def is_active(self)` |
`public def put(self,chunk)` |

## Members

#### `public def __init__(self,net,input_blob_names,batch_size,device_option,namescope,input_source_name,queue)` {#classdata__workers_1_1_data_input_coordinator_1a3ff2f8697f570e25d5293457c9292e51}





#### `public def is_active(self)` {#classdata__workers_1_1_data_input_coordinator_1a75959077167c75835785c0b5853e55d8}





#### `public def put(self,chunk)` {#classdata__workers_1_1_data_input_coordinator_1af53e24f8733f1a125c4fb4961e281bc0}





# class `data_workers::GlobalCoordinator` {#classdata__workers_1_1_global_coordinator}

```
class data_workers::GlobalCoordinator
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self)` |
`public def add(self,coordinator)` |
`public def get_queue(self,queue_name,max_buffered_batches)` |
`public def start(self)` |
`public def stop(self)` |
`public def register_shutdown_handler(self)` |

## Members

#### `public def __init__(self)` {#classdata__workers_1_1_global_coordinator_1ac53aaaa9ea3145a22c7000faa3296839}





#### `public def add(self,coordinator)` {#classdata__workers_1_1_global_coordinator_1a2f9d3d81a61c54ffa0049e926777386a}





#### `public def get_queue(self,queue_name,max_buffered_batches)` {#classdata__workers_1_1_global_coordinator_1aeb37cfadabd426dca4ebffe5cc7e30af}





#### `public def start(self)` {#classdata__workers_1_1_global_coordinator_1ad431d06758e5aa597e2afec49aa573dc}





#### `public def stop(self)` {#classdata__workers_1_1_global_coordinator_1a5794f95bcbe891eb6fdfe16b9b7aef64}





#### `public def register_shutdown_handler(self)` {#classdata__workers_1_1_global_coordinator_1ace266371e0e843bf7e897286bef26d92}





# namespace `data_workers_test` {#namespacedata__workers__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`data_workers_test::DataWorkersTest`](#classdata__workers__test_1_1_data_workers_test)    |
# class `data_workers_test::DataWorkersTest` {#classdata__workers__test_1_1_data_workers_test}

```
class data_workers_test::DataWorkersTest
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testNonParallelModel(self)` |
`public def testGracefulShutdown(self)` |

## Members

#### `public def testNonParallelModel(self)` {#classdata__workers__test_1_1_data_workers_test_1ad4a7774503ad58f1686300f214349765}





#### `public def testGracefulShutdown(self)` {#classdata__workers__test_1_1_data_workers_test_1a9d8e2218ef5160fa4b068199046ab7fe}





# namespace `dataio` {#namespacedataio}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dataio::CounterReader`](#classdataio_1_1_counter_reader)    |
`class `[`dataio::Pipe`](#classdataio_1_1_pipe)    |
`class `[`dataio::PipedReaderBuilder`](#classdataio_1_1_piped_reader_builder)    |
`class `[`dataio::Reader`](#classdataio_1_1_reader)    |
`class `[`dataio::ReaderBuilder`](#classdataio_1_1_reader_builder)    |
`class `[`dataio::ReaderWithLimit`](#classdataio_1_1_reader_with_limit)    |
`class `[`dataio::Writer`](#classdataio_1_1_writer)    |
# class `dataio::CounterReader` {#classdataio_1_1_counter_reader}

```
class dataio::CounterReader
  : public dataio.Reader
```  



Reader that produces increasing integers.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  counter` |
`public  should_stop` |
`public def __init__(self)` |
`public def setup_ex(self,global_init_net,global_finish_net)` |
`public def read_ex(self,local_init_net,local_finish_net)` |

## Members

#### `public  counter` {#classdataio_1_1_counter_reader_1a3e7b2233e429a7463d7ca9c806f0e57e}





#### `public  should_stop` {#classdataio_1_1_counter_reader_1afd4d8f176ffb3a163a098e09422888e8}





#### `public def __init__(self)` {#classdataio_1_1_counter_reader_1afac9752faf6ef3ba7b5135aaeeab417c}





#### `public def setup_ex(self,global_init_net,global_finish_net)` {#classdataio_1_1_counter_reader_1a511712e0f1fc9abe9da5f3d8a974df44}





#### `public def read_ex(self,local_init_net,local_finish_net)` {#classdataio_1_1_counter_reader_1a26ea1068880e8d08c1e5683d626d06e9}





# class `dataio::Pipe` {#classdataio_1_1_pipe}

```
class dataio::Pipe
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,`[`schema`](#classdataio_1_1_pipe_1aa62b258d75b2f34388cd4ccc8b7e20e3)`,obj_key)` |
`public def schema(self)` |
`public def setup(self,global_init_net)` |
`public def reader(self)` |
`public def writer(self)` |
`public def num_readers(self)` |
`public def num_writers(self)` |

## Members

#### `public def __init__(self,`[`schema`](#classdataio_1_1_pipe_1aa62b258d75b2f34388cd4ccc8b7e20e3)`,obj_key)` {#classdataio_1_1_pipe_1ab40b54f0987d0cfbee52a3d5eb8c7900}





#### `public def schema(self)` {#classdataio_1_1_pipe_1aa62b258d75b2f34388cd4ccc8b7e20e3}





#### `public def setup(self,global_init_net)` {#classdataio_1_1_pipe_1a9ea83de7cdc69c9407b27f35e59258ca}





#### `public def reader(self)` {#classdataio_1_1_pipe_1a61c845aefd673296ee03e1288c07d69d}





#### `public def writer(self)` {#classdataio_1_1_pipe_1a4c4b4e775b3b4d8c5040ca6d54a5b4ed}





#### `public def num_readers(self)` {#classdataio_1_1_pipe_1a977b4ab1d08856601da7a13091dfbc77}





#### `public def num_writers(self)` {#classdataio_1_1_pipe_1a6d3537cf666547b2de5ec48809387c2d}





# class `dataio::PipedReaderBuilder` {#classdataio_1_1_piped_reader_builder}

```
class dataio::PipedReaderBuilder
  : public dataio.ReaderBuilder
```  



ReaderBuilder that modifies underlying builder by calling `piper`
function on each new reader produced, and return the result of
the function. This way, it is possible to append data processing
pipelines that will be replicated for each reader that gets created.

E.g.:

PipedReaderBuilder(
    ReaderBuilder(...),
    lambda reader: pipe(reader, processor=my_proc))

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,builder,piper)` |
`public def schema(self)` |
`public def enqueue_splits(self,net,split_queue)` |
`public def splits(self,net)` |
`public def new_reader(self,split_queue)` |

## Members

#### `public def __init__(self,builder,piper)` {#classdataio_1_1_piped_reader_builder_1acda9a1dfc262666ddc4ca7299b0f11fd}





#### `public def schema(self)` {#classdataio_1_1_piped_reader_builder_1a562f3ebe02b49ac5f85581ba0b64f1ae}





#### `public def enqueue_splits(self,net,split_queue)` {#classdataio_1_1_piped_reader_builder_1a91c1753937031c025db6397bbc320ae1}





#### `public def splits(self,net)` {#classdataio_1_1_piped_reader_builder_1a3d81f63da9d498813d3f7ea6173931aa}





#### `public def new_reader(self,split_queue)` {#classdataio_1_1_piped_reader_builder_1ad87e8bb464cff4c5aeff62ffe7ea82ea}





# class `dataio::Reader` {#classdataio_1_1_reader}

```
class dataio::Reader
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,`[`schema`](#classdataio_1_1_reader_1a92e993974271957bbca19fc2ca818dd6)`)` |
`public def schema(self)` |
`public def setup_ex(self,init_net,finish_net)` |
`public def read_ex(self,local_init_net,local_finish_net)` |
`public def read_record_ex(self,local_init_net,local_finish_net)` |
`public def read(self,read_net)` |
`public def reset(self,net)` |
`public def read_record(self,read_net)` |
`public def execution_step(self,reader_net_name,external_should_stop)` |

## Members

#### `public def __init__(self,`[`schema`](#classdataio_1_1_reader_1a92e993974271957bbca19fc2ca818dd6)`)` {#classdataio_1_1_reader_1aade68bcdf5f5efb6af7e9cebd6a88933}





#### `public def schema(self)` {#classdataio_1_1_reader_1a92e993974271957bbca19fc2ca818dd6}



Return the schema associated with the Reader

#### `public def setup_ex(self,init_net,finish_net)` {#classdataio_1_1_reader_1a5461f2c38da9a2ee4f7cd96d6e609849}



Nets to be executed once at startup and finish.
   Experimental extension. Don't use yet

#### `public def read_ex(self,local_init_net,local_finish_net)` {#classdataio_1_1_reader_1a5179c5354531efed68c6bbcd7cb2a8f7}



Experimental extension to the interface. Don't use yet

#### `public def read_record_ex(self,local_init_net,local_finish_net)` {#classdataio_1_1_reader_1acd753bbab113f4f937f0cd6e6d043636}



Experimental extension to the interface. Don't use yet

#### `public def read(self,read_net)` {#classdataio_1_1_reader_1a0a2d930b88a799400966b05bf67a14ab}



Add operations to read_net that will read the read batch of data
and return a list of BlobReference representing the blobs that will
contain the batches produced.

Operations added to `read_net` must be thread safe and atomic, that is,
it should be possible to clone `read_net` and run multiple instances of
it in parallel.

Args:
    read_net: the net that will be appended with read operations

Returns:
    A tuple (should_stop, fields), with:

should_stop: BlobReference pointing to a boolean scalar
             blob that indicates whether the read operation
             was succesfull or whether the end of data has
             been reached.
fields: A tuple of BlobReference containing the latest batch
        of data that was read.

#### `public def reset(self,net)` {#classdataio_1_1_reader_1a82de8d1221c15af96a7d3ba29b748cf2}



Append operations to `net` that will reset the reader.

This can be used to read the data multiple times.
Not all readers support this operation.

#### `public def read_record(self,read_net)` {#classdataio_1_1_reader_1a55e368d4babd4f10516f36fb23742433}





#### `public def execution_step(self,reader_net_name,external_should_stop)` {#classdataio_1_1_reader_1af34001b65f5f7755d60499fbe1ab9708}



Create an execution step with a net containing read operators.

The execution step will contain a `stop_blob` that knows how to stop
the execution loop when end of data was reached.

E.g.:

    read_step, fields = reader.execution_step()
    consume_net = core.Net('consume')
    consume_net.Print(fields[0], [])
    p = core.Plan('reader')
    p.AddStep(read_step.AddNet(consume_net))
    core.RunPlan(p)

Args:

    reader_net_name: (optional) the name of the reader_net to be
             created. The execution step will
             be named accordingly.

Returns:
    A tuple (read_step, fields), with:

read_step: A newly created execution step containing a net with
           read operations. The step will have `stop_blob` set,
           in order to stop the loop on end of data.
fields: A tuple of BlobReference containing the latest batch
        of data that was read.

# class `dataio::ReaderBuilder` {#classdataio_1_1_reader_builder}

```
class dataio::ReaderBuilder
  : public object
```  



Allow usage of a reader in distributed fashion.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def schema(self)` |
`public def enqueue_splits(self,net,split_queue)` |
`public def splits(self,net)` |
`public def new_reader(self,split_queue)` |

## Members

#### `public def schema(self)` {#classdataio_1_1_reader_builder_1ab40d1c13edf38cdddcfe62106c917112}





#### `public def enqueue_splits(self,net,split_queue)` {#classdataio_1_1_reader_builder_1aead4b61235540071648fa43c5441070b}





#### `public def splits(self,net)` {#classdataio_1_1_reader_builder_1a7611317b6d86f599b3bbb6613d330c7b}





#### `public def new_reader(self,split_queue)` {#classdataio_1_1_reader_builder_1a5de781938ac50c63edd04f02f1428197}





# class `dataio::ReaderWithLimit` {#classdataio_1_1_reader_with_limit}

```
class dataio::ReaderWithLimit
  : public dataio.Reader
```  



Reader that stops after `num_iter` calls.

If num_iter is None it becomes just a simple reader that exports a global
flag for "out of data".

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  reader` |
`public  counter` |
`public  num_iter` |
`public def __init__(self,`[`reader`](#classdataio_1_1_reader_with_limit_1a1bb42d5e1f605fcc6c3b50f021449963)`,`[`num_iter`](#classdataio_1_1_reader_with_limit_1aa5df59315e0a2d8463fb17bd58c9b4e9)`)` |
`public def setup_ex(self,global_init_net,global_finish_net)` |
`public def read_ex(self,local_init_net,local_finish_net)` |
`public def data_finished(self)` |

## Members

#### `public  reader` {#classdataio_1_1_reader_with_limit_1a1bb42d5e1f605fcc6c3b50f021449963}





#### `public  counter` {#classdataio_1_1_reader_with_limit_1a8defcb6442b317b800a842c3806e5127}





#### `public  num_iter` {#classdataio_1_1_reader_with_limit_1aa5df59315e0a2d8463fb17bd58c9b4e9}





#### `public def __init__(self,`[`reader`](#classdataio_1_1_reader_with_limit_1a1bb42d5e1f605fcc6c3b50f021449963)`,`[`num_iter`](#classdataio_1_1_reader_with_limit_1aa5df59315e0a2d8463fb17bd58c9b4e9)`)` {#classdataio_1_1_reader_with_limit_1a5ac04b7da8f151b8576a7d302bf86f06}





#### `public def setup_ex(self,global_init_net,global_finish_net)` {#classdataio_1_1_reader_with_limit_1ae4321a4fd2d3d874ad777ff0f2f77b54}





#### `public def read_ex(self,local_init_net,local_finish_net)` {#classdataio_1_1_reader_with_limit_1a83c9211ceb0832c367cc79f54885dd64}



1. check if we reached number of iterations and populate the same
should_stop blob

#### `public def data_finished(self)` {#classdataio_1_1_reader_with_limit_1a64bdc5c22b0f96502c0f81b01ab7a6e0}



Return a blob that can be checked after the end of the reading task,
which will contain a scalar float indicating whether the underlying
reader has been exhausted (True) or whether we stopped because reached
the limit of iterations (False).

# class `dataio::Writer` {#classdataio_1_1_writer}

```
class dataio::Writer
  : public object
```  



Writer is a abstract class to be implemented in order to provide
operations capable of feeding a data stream or a dataset.

A Writer must implement 2 operations:
`write`, which adds operations to a net that write the write batch of
data, and `commit`, which adds operations to a net in order to indicate
that no more data will be written.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def schema(self)` |
`public def write(self,writer_net,fields)` |
`public def write_record(self,writer_net,fields)` |
`public def setup_ex(self,init_net,finish_net)` |
`public def write_ex(self,fields,local_init_net,local_finish_net,stop_blob)` |
`public def write_record_ex(self,fields,local_init_net,local_finish_net,stop_blob)` |
`public def commit(self,finish_net)` |

## Members

#### `public def schema(self)` {#classdataio_1_1_writer_1a57ffd8095e8484ded13a983c88f9b523}





#### `public def write(self,writer_net,fields)` {#classdataio_1_1_writer_1a68f495e53a5f368f67757a3a3bf7971c}



Add operations to `writer_net` that write the next batch of data.

Operations added to the net must be thread-safe and unique, that is:
multiple writers must be able to write to the dataset in parallel.

Args:
    fields: a tuple of BlobReference containing the batch of data to
    write.

#### `public def write_record(self,writer_net,fields)` {#classdataio_1_1_writer_1a1fdfe47dcc8117819457cf26c871aa8c}





#### `public def setup_ex(self,init_net,finish_net)` {#classdataio_1_1_writer_1a58e5ca55f4ea25cdf47435d0dd5e2622}



Experimental, don't use yet

#### `public def write_ex(self,fields,local_init_net,local_finish_net,stop_blob)` {#classdataio_1_1_writer_1adfd82dd55c5ef4aadb58f0e8428f49e3}



Experimental extension to the interface. Don't use yet

#### `public def write_record_ex(self,fields,local_init_net,local_finish_net,stop_blob)` {#classdataio_1_1_writer_1aeb060c0bcc66ba47d959f5d3f19952b7}



Experimental extension to the interface. Don't use yet.

#### `public def commit(self,finish_net)` {#classdataio_1_1_writer_1a890dbfb569d06efc9d92520f2e0aa662}



Add operations to `finish_net` that signal end of data.

This must be implemented by all Writers, but may be no-op for some
of them.

# namespace `dataio_test` {#namespacedataio__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dataio_test::TestReaderWithLimit`](#classdataio__test_1_1_test_reader_with_limit)    |
# class `dataio_test::TestReaderWithLimit` {#classdataio__test_1_1_test_reader_with_limit}

```
class dataio_test::TestReaderWithLimit
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_reader_with_limit(self)` |

## Members

#### `public def test_reader_with_limit(self)` {#classdataio__test_1_1_test_reader_with_limit_1aaef12cc824f9485daa7c635d28d25488}





# namespace `dataset` {#namespacedataset}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dataset::_DatasetRandomReader`](#classdataset_1_1___dataset_random_reader)    |
`class `[`dataset::_DatasetReader`](#classdataset_1_1___dataset_reader)    |
`class `[`dataset::_DatasetWriter`](#classdataset_1_1___dataset_writer)    |
`class `[`dataset::Dataset`](#classdataset_1_1_dataset)    |
# class `dataset::_DatasetRandomReader` {#classdataset_1_1___dataset_random_reader}

```
class dataset::_DatasetRandomReader
  : public Reader
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  dataset` |
`public  cursor` |
`public  name` |
`public  indices` |
`public  batch_size` |
`public  offsets` |
`public def __init__(self,`[`dataset`](#classdataset_1_1___dataset_random_reader_1a9283b981821ffe90e7d59209ff974c0f)`,`[`name`](#classdataset_1_1___dataset_random_reader_1aa8f9e92ad00fad45745576934ba89d54)`,`[`indices`](#classdataset_1_1___dataset_random_reader_1a436941659fe57fa32cea3c45973f4753)`,`[`batch_size`](#classdataset_1_1___dataset_random_reader_1ab575e3612f8a96037086ab64081ba4d1)`)` |
`public def setup_ex(self,init_net,exit_net)` |
`public def reset(self,net)` |
`public def computeoffset(self,net)` |
`public def sort_and_shuffle(self,net,sort_by_field,shuffle_size,`[`batch_size`](#classdataset_1_1___dataset_random_reader_1ab575e3612f8a96037086ab64081ba4d1)`)` |
`public def read(self,read_net)` |

## Members

#### `public  dataset` {#classdataset_1_1___dataset_random_reader_1a9283b981821ffe90e7d59209ff974c0f}





#### `public  cursor` {#classdataset_1_1___dataset_random_reader_1ad7a1b2cbae354705e8bb403133311e6b}





#### `public  name` {#classdataset_1_1___dataset_random_reader_1aa8f9e92ad00fad45745576934ba89d54}





#### `public  indices` {#classdataset_1_1___dataset_random_reader_1a436941659fe57fa32cea3c45973f4753}





#### `public  batch_size` {#classdataset_1_1___dataset_random_reader_1ab575e3612f8a96037086ab64081ba4d1}





#### `public  offsets` {#classdataset_1_1___dataset_random_reader_1a7e4fd99b143f4cb943f1cfa5af42351c}





#### `public def __init__(self,`[`dataset`](#classdataset_1_1___dataset_random_reader_1a9283b981821ffe90e7d59209ff974c0f)`,`[`name`](#classdataset_1_1___dataset_random_reader_1aa8f9e92ad00fad45745576934ba89d54)`,`[`indices`](#classdataset_1_1___dataset_random_reader_1a436941659fe57fa32cea3c45973f4753)`,`[`batch_size`](#classdataset_1_1___dataset_random_reader_1ab575e3612f8a96037086ab64081ba4d1)`)` {#classdataset_1_1___dataset_random_reader_1ac366b6145cfab4104b28f12773a44c99}



Don't call this directly. Instead, use dataset.random_reader()

#### `public def setup_ex(self,init_net,exit_net)` {#classdataset_1_1___dataset_random_reader_1aea521290c65a636a768fd126a1b1e13b}





#### `public def reset(self,net)` {#classdataset_1_1___dataset_random_reader_1a7332af56cdcbe5a580d92ba61c5359b6}





#### `public def computeoffset(self,net)` {#classdataset_1_1___dataset_random_reader_1a7365ad083a92512d9b1f00fdb1eb1ddd}





#### `public def sort_and_shuffle(self,net,sort_by_field,shuffle_size,`[`batch_size`](#classdataset_1_1___dataset_random_reader_1ab575e3612f8a96037086ab64081ba4d1)`)` {#classdataset_1_1___dataset_random_reader_1ab571cc0d5cda11e6d4489ca31321836f}





#### `public def read(self,read_net)` {#classdataset_1_1___dataset_random_reader_1a3994cf08a815ab2356adfbeda8e43dd6}





# class `dataset::_DatasetReader` {#classdataset_1_1___dataset_reader}

```
class dataset::_DatasetReader
  : public Reader
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  dataset` |
`public  name` |
`public  batch_size` |
`public  cursor` |
`public def __init__(self,`[`dataset`](#classdataset_1_1___dataset_reader_1a0318685ab2c7f108f00f0ce08775d772)`,`[`name`](#classdataset_1_1___dataset_reader_1a52f18700ed9ef418369d90dfe9133694)`,`[`batch_size`](#classdataset_1_1___dataset_reader_1a3e40f6d82d90baae594fa1da5ae8dc4e)`)` |
`public def setup_ex(self,init_net,exit_net)` |
`public def read(self,read_net)` |
`public def reset(self,net)` |

## Members

#### `public  dataset` {#classdataset_1_1___dataset_reader_1a0318685ab2c7f108f00f0ce08775d772}





#### `public  name` {#classdataset_1_1___dataset_reader_1a52f18700ed9ef418369d90dfe9133694}





#### `public  batch_size` {#classdataset_1_1___dataset_reader_1a3e40f6d82d90baae594fa1da5ae8dc4e}





#### `public  cursor` {#classdataset_1_1___dataset_reader_1ab7f03597c4669233641aafa50b877a8b}





#### `public def __init__(self,`[`dataset`](#classdataset_1_1___dataset_reader_1a0318685ab2c7f108f00f0ce08775d772)`,`[`name`](#classdataset_1_1___dataset_reader_1a52f18700ed9ef418369d90dfe9133694)`,`[`batch_size`](#classdataset_1_1___dataset_reader_1a3e40f6d82d90baae594fa1da5ae8dc4e)`)` {#classdataset_1_1___dataset_reader_1a4ee4b857f94a3884a7aa16de7f174126}



Don't call this directly. Instead, use dataset.reader()

#### `public def setup_ex(self,init_net,exit_net)` {#classdataset_1_1___dataset_reader_1aab8e4301acfb2befbab784635fc0aade}





#### `public def read(self,read_net)` {#classdataset_1_1___dataset_reader_1a6f738a780f3ebb3260f6f31d2adb7ddf}





#### `public def reset(self,net)` {#classdataset_1_1___dataset_reader_1a9363b4307b9db5b06184b434f0f0110d}





# class `dataset::_DatasetWriter` {#classdataset_1_1___dataset_writer}

```
class dataset::_DatasetWriter
  : public Writer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  mutex` |
`public def __init__(self,content)` |
`public def setup_ex(self,init_net,exit_net)` |
`public def write(self,writer_net,fields)` |
`public def commit(self,finish_net)` |

## Members

#### `public  mutex` {#classdataset_1_1___dataset_writer_1a1beec5184256bd7c075bf3ee8045215b}





#### `public def __init__(self,content)` {#classdataset_1_1___dataset_writer_1a78daed72fc2d153b7c9b768a347ff6f8}



Don't call this directly. Use dataset.writer() instead.

#### `public def setup_ex(self,init_net,exit_net)` {#classdataset_1_1___dataset_writer_1a96bf5fda5dddd08d79b2a3eadfccd620}





#### `public def write(self,writer_net,fields)` {#classdataset_1_1___dataset_writer_1a5955232d2ec3920f18cbe150c3f8e499}



Add operations to `net` that append the blobs in `fields` to the end
of the dataset. An additional operator will also be added that checks
the consistency of the data in `fields` against the dataset schema.

Args:
    writer_net: The net that will contain the Append operators.
    fields: A list of BlobReference to be appeneded to this dataset.

#### `public def commit(self,finish_net)` {#classdataset_1_1___dataset_writer_1a09732cee232bacf68d158890e14f5118}



Commit is a no-op for an in-memory dataset.

# class `dataset::Dataset` {#classdataset_1_1_dataset}

```
class dataset::Dataset
  : public object
```  



Represents an in-memory dataset with fixed schema.

Use this to store and iterate through datasets with complex schema that
fit in memory.

Iterating through entries of this dataset is very fast since the dataset
is stored as a set of native Caffe2 tensors, thus no type conversion or
deserialization is necessary.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  schema` |
`public  fields` |
`public  field_types` |
`public  name` |
`public  field_blobs` |
`public def __init__(self,`[`fields`](#classdataset_1_1_dataset_1a07d5bdbf33a51e55f080e1bf08dfb308)`,`[`name`](#classdataset_1_1_dataset_1a08099d5ecf992a1c57078c456ba26f26)`)` |
`public def init_empty(self,init_net)` |
`public def init_from_dataframe(self,net,dataframe)` |
`public def get_blobs(self)` |
`public def content(self)` |
`public def field_names(self)` |
`public def field_types(self)` |
`public def reader(self,init_net,cursor_name,batch_size)` |
`public def random_reader(self,init_net,indices,cursor_name,batch_size)` |
`public def writer(self,init_net)` |

## Members

#### `public  schema` {#classdataset_1_1_dataset_1a5bdc4a139cb5d3c6fc07a150fd6a5486}





#### `public  fields` {#classdataset_1_1_dataset_1a07d5bdbf33a51e55f080e1bf08dfb308}





#### `public  field_types` {#classdataset_1_1_dataset_1a9fbe106fbff808f65da69cf0711fda38}





#### `public  name` {#classdataset_1_1_dataset_1a08099d5ecf992a1c57078c456ba26f26}





#### `public  field_blobs` {#classdataset_1_1_dataset_1aea7d9009b1d4d336b693ff1c4025aaaa}





#### `public def __init__(self,`[`fields`](#classdataset_1_1_dataset_1a07d5bdbf33a51e55f080e1bf08dfb308)`,`[`name`](#classdataset_1_1_dataset_1a08099d5ecf992a1c57078c456ba26f26)`)` {#classdataset_1_1_dataset_1a61647faeb8a1f8a3a8861453f20771f1}



Create an un-initialized dataset with schema provided by `fields`.

Before this dataset can be used, it must be initialized, either by
`init_empty` or `init_from_dataframe`.

Args:
    fields: either a schema.Struct or a list of field names in a format
    compatible with the one described in schema.py.
    name: optional name to prepend to blobs that will store the data.

#### `public def init_empty(self,init_net)` {#classdataset_1_1_dataset_1ab3465dcbe1e2fd4c51294e814d6a1e0d}



Initialize the blobs for this dataset with empty values.

Empty arrays will be immediately fed into the current workspace,
and `init_net` will take those blobs as external inputs.

#### `public def init_from_dataframe(self,net,dataframe)` {#classdataset_1_1_dataset_1a9fb8b5d289d586f1e8c5e565b7a362a2}



Initialize the blobs for this dataset from a Pandas dataframe.

Each column of the dataframe will be immediately fed into the current
workspace, and the `net` will take this blobs as external inputs.

#### `public def get_blobs(self)` {#classdataset_1_1_dataset_1a6f69f0863dad5413532f940156de7b43}



Return the list of BlobReference pointing to the blobs that contain
the data for this dataset.

#### `public def content(self)` {#classdataset_1_1_dataset_1ab6ceb96921c923d77d66b9a2e2df51fa}



Return a Record of BlobReferences pointing to the full content of
this dataset.

#### `public def field_names(self)` {#classdataset_1_1_dataset_1a5724593b268669c7241f0571bd42fba4}



Return the list of field names for this dataset.

#### `public def field_types(self)` {#classdataset_1_1_dataset_1a8dd6065c197fd9c80d6e4eda8845ab6c}



Return the list of field dtypes for this dataset.

If a list of strings, not a schema.Struct, was passed to the
constructor, this will return a list of dtype(np.void).

#### `public def reader(self,init_net,cursor_name,batch_size)` {#classdataset_1_1_dataset_1a985fd3ba198df10ff681f6a6d0302d2b}



Create a Reader object that is used to iterate through the dataset.

This will append operations to `init_net` that create a TreeCursor,
used to iterate through the data.

NOTE: Currently, it is not safe to append to a dataset while reading.

Args:
    init_net: net that will be run once to create the cursor.
    cursor_name: optional name for the blob containing a pointer
         to the cursor.
    batch_size: how many samples to read per iteration.

Returns:
    A _DatasetReader that can be used to create operators that will
    iterate through the dataset.

#### `public def random_reader(self,init_net,indices,cursor_name,batch_size)` {#classdataset_1_1_dataset_1a242776a266e2a74a710632ccd56cd2f0}



Create a Reader object that is used to iterate through the dataset.

NOTE: The reader order depends on the order in indices.

Args:
    init_net: net that will be run once to create the cursor.
    indices: blob of reading order
    cursor_name: optional name for the blob containing a pointer
         to the cursor.
    batch_size: how many samples to read per iteration.

Returns:
    A DatasetReader that can be used to create operators that will
    iterate through the dataset according to indices.

#### `public def writer(self,init_net)` {#classdataset_1_1_dataset_1a6d904ee9d2be89745bf4b5e68132a86e}



Create a Writer that can be used to append entries into the dataset.

NOTE: Currently, it is not safe to append to a dataset
      while reading from it.
NOTE: Currently implementation of writer is not thread safe.
      TODO: fixme

Args:
    init_net: net that will be run once in order to create the writer.
      (currently not used)

# namespace `dataset_ops_test` {#namespacedataset__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dataset_ops_test::TestDatasetOps`](#classdataset__ops__test_1_1_test_dataset_ops)    |
# class `dataset_ops_test::TestDatasetOps` {#classdataset__ops__test_1_1_test_dataset_ops}

```
class dataset_ops_test::TestDatasetOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_dataset_ops(self)` |
`public def test_last_n_window_ops(self)` |
`public def test_collect_tensor_ops(self)` |

## Members

#### `public def test_dataset_ops(self)` {#classdataset__ops__test_1_1_test_dataset_ops_1a402390250b2801620fd0e15e07d07218}



1. Defining the schema of our dataset.

This example schema could represent, for example, a search query log.

#### `public def test_last_n_window_ops(self)` {#classdataset__ops__test_1_1_test_dataset_ops_1a0741796aee494632fd13fa551dc4aeb2}





#### `public def test_collect_tensor_ops(self)` {#classdataset__ops__test_1_1_test_dataset_ops_1a13ada61d529bed3f88c658374deb9c8a}





# namespace `db_test` {#namespacedb__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`db_test::TestDB`](#classdb__test_1_1_test_d_b)    |
# class `db_test::TestDB` {#classdb__test_1_1_test_d_b}

```
class db_test::TestDB
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  file_name` |
`public  data` |
`public def setUp(self)` |
`public def testSimple(self)` |

## Members

#### `public  file_name` {#classdb__test_1_1_test_d_b_1a78a0e4874c84ae9c3adde10e53835426}





#### `public  data` {#classdb__test_1_1_test_d_b_1a4304e30b35921ca37ce814eec61dd9be}





#### `public def setUp(self)` {#classdb__test_1_1_test_d_b_1ab4f3c889bbbc441e77fac80ecefde4d2}





#### `public def testSimple(self)` {#classdb__test_1_1_test_d_b_1a0ff374cdc33a0ed23a91e64b64fcebaf}





# namespace `device_checker` {#namespacedevice__checker}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`device_checker::DeviceChecker`](#classdevice__checker_1_1_device_checker)    |
# class `device_checker::DeviceChecker` {#classdevice__checker_1_1_device_checker}

```
class device_checker::DeviceChecker
  : public object
```  



A device checker in Python to check consistency across multiple devices.

This is not the most efficient way to check devices, as the Python interface
will involve a lot of copy back and forth operations. Use at your own risk.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,threshold,device_options)` |
`public def CheckSimple(self,op,inputs,outputs_to_check,input_device_options)` |
`public def CheckNet(self,net,inputs,blobs_to_check,ignore)` |

## Members

#### `public def __init__(self,threshold,device_options)` {#classdevice__checker_1_1_device_checker_1a698209b2e1708eb7349b424fd0c8e512}





#### `public def CheckSimple(self,op,inputs,outputs_to_check,input_device_options)` {#classdevice__checker_1_1_device_checker_1ab9c002843e4f4e76f65a84fc45f8a990}



Checks the operator with different device implementations.

Inputs:
  op: the operator to be checked.
  inputs: the input data in numpy arrays.
  outputs_to_check: the outputs to check between devices.
  input_device_options: a mapping from input name to a device to use
    (instead of self._device_options)
Outputs:
  boolean: True if it passes, False if it does not pass.

#### `public def CheckNet(self,net,inputs,blobs_to_check,ignore)` {#classdevice__checker_1_1_device_checker_1ab1801eb71622ecc86b8d1d31ff8dc237}



Checks a network by inspecting all of its intermediate results, and
see if things match.

# namespace `duplicate_operands_test` {#namespaceduplicate__operands__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`duplicate_operands_test::TestDuplicateOperands`](#classduplicate__operands__test_1_1_test_duplicate_operands)    |
# class `duplicate_operands_test::TestDuplicateOperands` {#classduplicate__operands__test_1_1_test_duplicate_operands}

```
class duplicate_operands_test::TestDuplicateOperands
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_duplicate_operands(self)` |

## Members

#### `public def test_duplicate_operands(self)` {#classduplicate__operands__test_1_1_test_duplicate_operands_1a041e801cc4790663c9551395e33c767b}





# namespace `elementwise_op_broadcast_test` {#namespaceelementwise__op__broadcast__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`elementwise_op_broadcast_test::TestElementwiseBroadcast`](#classelementwise__op__broadcast__test_1_1_test_elementwise_broadcast)    |
# class `elementwise_op_broadcast_test::TestElementwiseBroadcast` {#classelementwise__op__broadcast__test_1_1_test_elementwise_broadcast}

```
class elementwise_op_broadcast_test::TestElementwiseBroadcast
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_broadcast(self,gc,dc)` |
`public def test_broadcast_scalar(self,gc,dc)` |
`public def test_semantic_broadcast(self,gc,dc)` |

## Members

#### `public def test_broadcast(self,gc,dc)` {#classelementwise__op__broadcast__test_1_1_test_elementwise_broadcast_1ab55a4900d15ee4fa6cade7760db7bcef}





#### `public def test_broadcast_scalar(self,gc,dc)` {#classelementwise__op__broadcast__test_1_1_test_elementwise_broadcast_1a9a22a5bcb4166b87653f4cdec0c70d91}





#### `public def test_semantic_broadcast(self,gc,dc)` {#classelementwise__op__broadcast__test_1_1_test_elementwise_broadcast_1a9df44bae0a42f4e24b2e764926871ebe}





# namespace `elementwise_ops_test` {#namespaceelementwise__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`elementwise_ops_test::TestElementwiseOps`](#classelementwise__ops__test_1_1_test_elementwise_ops)    |
# class `elementwise_ops_test::TestElementwiseOps` {#classelementwise__ops__test_1_1_test_elementwise_ops}

```
class elementwise_ops_test::TestElementwiseOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_div(self,`[`n`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a894d25ac41746ff2e303681c72dad196)`,`[`m`](#classelementwise__ops__test_1_1_test_elementwise_ops_1aa0ce11505aae85f8615d4c8d9af6b136)`,`[`d`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a73b0e0d2c22cdace9ad21e0852843562)`,gc,dc)` |
`public def test_log(self,`[`n`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a894d25ac41746ff2e303681c72dad196)`,`[`m`](#classelementwise__ops__test_1_1_test_elementwise_ops_1aa0ce11505aae85f8615d4c8d9af6b136)`,gc,dc)` |
`public def test_sqr(self,`[`n`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a894d25ac41746ff2e303681c72dad196)`,`[`m`](#classelementwise__ops__test_1_1_test_elementwise_ops_1aa0ce11505aae85f8615d4c8d9af6b136)`,gc,dc)` |

## Members

#### `public def test_div(self,`[`n`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a894d25ac41746ff2e303681c72dad196)`,`[`m`](#classelementwise__ops__test_1_1_test_elementwise_ops_1aa0ce11505aae85f8615d4c8d9af6b136)`,`[`d`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a73b0e0d2c22cdace9ad21e0852843562)`,gc,dc)` {#classelementwise__ops__test_1_1_test_elementwise_ops_1a627cb870b6691c4e3a247b05bf82fe2c}





#### `public def test_log(self,`[`n`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a894d25ac41746ff2e303681c72dad196)`,`[`m`](#classelementwise__ops__test_1_1_test_elementwise_ops_1aa0ce11505aae85f8615d4c8d9af6b136)`,gc,dc)` {#classelementwise__ops__test_1_1_test_elementwise_ops_1ac4d344788329350248ce6e8c72d71ce4}





#### `public def test_sqr(self,`[`n`](#classelementwise__ops__test_1_1_test_elementwise_ops_1a894d25ac41746ff2e303681c72dad196)`,`[`m`](#classelementwise__ops__test_1_1_test_elementwise_ops_1aa0ce11505aae85f8615d4c8d9af6b136)`,gc,dc)` {#classelementwise__ops__test_1_1_test_elementwise_ops_1accb3cce075e1b31fd3a6575d8c1789e2}





# namespace `emptysample_ops_test` {#namespaceemptysample__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`emptysample_ops_test::TestEmptySampleOps`](#classemptysample__ops__test_1_1_test_empty_sample_ops)    |
# class `emptysample_ops_test::TestEmptySampleOps` {#classemptysample__ops__test_1_1_test_empty_sample_ops}

```
class emptysample_ops_test::TestEmptySampleOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_emptysample(self)` |

## Members

#### `public def test_emptysample(self)` {#classemptysample__ops__test_1_1_test_empty_sample_ops_1ace4e3ab0dd1ac4b10c9d103be0bc16a1}





# namespace `experiment_util` {#namespaceexperiment__util}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`experiment_util::ExternalLogger`](#classexperiment__util_1_1_external_logger)    |
`class `[`experiment_util::ModelTrainerLog`](#classexperiment__util_1_1_model_trainer_log)    |
# class `experiment_util::ExternalLogger` {#classexperiment__util_1_1_external_logger}

```
class experiment_util::ExternalLogger
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def set_runtime_args(self,runtime_args)` |
`public def log(self,log_dict)` |

## Members

#### `public def set_runtime_args(self,runtime_args)` {#classexperiment__util_1_1_external_logger_1a6958df366dcf4503578e4fee11340f52}



Set runtime arguments for the logger.
    runtime_args: dict of runtime arguments.

#### `public def log(self,log_dict)` {#classexperiment__util_1_1_external_logger_1a176c4229f6f12e67690186ade43c80c0}



log a dict of key/values to an external destination
    log_dict: input dict

# class `experiment_util::ModelTrainerLog` {#classexperiment__util_1_1_model_trainer_log}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  experiment_id` |
`public  filename` |
`public  headers` |
`public  start_time` |
`public  last_time` |
`public  last_input_count` |
`public  external_loggers` |
`public def __init__(self,expname,runtime_args,`[`external_loggers`](#classexperiment__util_1_1_model_trainer_log_1a8d8a7b43183105700fb490db67b5a5eb)`)` |
`public def logstr(self,str)` |
`public def log(self,input_count,batch_count,additional_values)` |

## Members

#### `public  experiment_id` {#classexperiment__util_1_1_model_trainer_log_1a88444505eaa1c9d549efa4a8f315133e}





#### `public  filename` {#classexperiment__util_1_1_model_trainer_log_1a56f16d06d6e81866e0cfc475088454ec}





#### `public  headers` {#classexperiment__util_1_1_model_trainer_log_1ae5b3230a6fa2fc411c7f49b47c2f3e7e}





#### `public  start_time` {#classexperiment__util_1_1_model_trainer_log_1a8d7188d9083a72bd406d1dec4670fad7}





#### `public  last_time` {#classexperiment__util_1_1_model_trainer_log_1af0aadf430c7ef29920e40b0b46f1971c}





#### `public  last_input_count` {#classexperiment__util_1_1_model_trainer_log_1ad9d96c946468eb339c7e7f448e7d1c7b}





#### `public  external_loggers` {#classexperiment__util_1_1_model_trainer_log_1a8d8a7b43183105700fb490db67b5a5eb}





#### `public def __init__(self,expname,runtime_args,`[`external_loggers`](#classexperiment__util_1_1_model_trainer_log_1a8d8a7b43183105700fb490db67b5a5eb)`)` {#classexperiment__util_1_1_model_trainer_log_1a5b0fbf151401881a09e43c08076a40f1}





#### `public def logstr(self,str)` {#classexperiment__util_1_1_model_trainer_log_1a3d58b4239c18e5aded6cdf431d38070f}





#### `public def log(self,input_count,batch_count,additional_values)` {#classexperiment__util_1_1_model_trainer_log_1a6fbc296be50336565ecc25e54779f8b9}





# namespace `extend_tensor_op_test` {#namespaceextend__tensor__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`extend_tensor_op_test::TestExtendTensorOp`](#classextend__tensor__op__test_1_1_test_extend_tensor_op)    |
# class `extend_tensor_op_test::TestExtendTensorOp` {#classextend__tensor__op__test_1_1_test_extend_tensor_op}

```
class extend_tensor_op_test::TestExtendTensorOp
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_extend_tensor(self)` |
`public def test_counting(self)` |

## Members

#### `public def test_extend_tensor(self)` {#classextend__tensor__op__test_1_1_test_extend_tensor_op_1ae8518def5c26304ac7d641b2979f841a}





#### `public def test_counting(self)` {#classextend__tensor__op__test_1_1_test_extend_tensor_op_1a073bb77f9d29e1161a2fd196ee161a66}





# namespace `fc_operator_test` {#namespacefc__operator__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`fc_operator_test::TestFcOperator`](#classfc__operator__test_1_1_test_fc_operator)    |
# class `fc_operator_test::TestFcOperator` {#classfc__operator__test_1_1_test_fc_operator}

```
class fc_operator_test::TestFcOperator
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_fc(self,`[`n`](#classfc__operator__test_1_1_test_fc_operator_1ae579d52d9c26bf7e216c710949ca02e2)`,`[`m`](#classfc__operator__test_1_1_test_fc_operator_1a4c17cc9fc095f8ce6a2a2bca762cac56)`,`[`k`](#classfc__operator__test_1_1_test_fc_operator_1aa5aba0d3cbb216c6e820918259b34388)`,gc,dc)` |

## Members

#### `public def test_fc(self,`[`n`](#classfc__operator__test_1_1_test_fc_operator_1ae579d52d9c26bf7e216c710949ca02e2)`,`[`m`](#classfc__operator__test_1_1_test_fc_operator_1a4c17cc9fc095f8ce6a2a2bca762cac56)`,`[`k`](#classfc__operator__test_1_1_test_fc_operator_1aa5aba0d3cbb216c6e820918259b34388)`,gc,dc)` {#classfc__operator__test_1_1_test_fc_operator_1a83aad8dde30f20dacb882bec4d321d5e}





# namespace `filler_ops_test` {#namespacefiller__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`filler_ops_test::TestFillerOperator`](#classfiller__ops__test_1_1_test_filler_operator)    |
# class `filler_ops_test::TestFillerOperator` {#classfiller__ops__test_1_1_test_filler_operator}

```
class filler_ops_test::TestFillerOperator
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_shape_error(self,gc,dc)` |
`public def test_gaussian_fill_op(self,gc,dc)` |
`public def test_msra_fill_op(self,gc,dc)` |

## Members

#### `public def test_shape_error(self,gc,dc)` {#classfiller__ops__test_1_1_test_filler_operator_1a6f5a2fb1791c70282fcc1099ba8cf46a}





#### `public def test_gaussian_fill_op(self,gc,dc)` {#classfiller__ops__test_1_1_test_filler_operator_1a2381bdfbff34e3cd3c1f457141e7c021}





#### `public def test_msra_fill_op(self,gc,dc)` {#classfiller__ops__test_1_1_test_filler_operator_1ae321d6bbc2d0885fd933dea5aca684b0}





# namespace `formatter` {#namespaceformatter}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`formatter::Formatter`](#classformatter_1_1_formatter)    |
`class `[`formatter::Markdown`](#classformatter_1_1_markdown)    |
# class `formatter::Formatter` {#classformatter_1_1_formatter}

```
class formatter::Formatter
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  content` |
`public def __init__(self)` |
`public def clone(self)` |
`public def dump(self)` |
`public def parseAndAdd(self,text)` |
`public def addRaw(self,text)` |
`public def addLine(self,text)` |
`public def addLinebreak(self)` |
`public def addHeader(self,text)` |
`public def addEmphasis(self,text)` |
`public def addList(self,textList)` |
`public def addLink(self,text,url)` |
`public def addCode(self,text)` |
`public def addCodeLink(self,text)` |
`public def addTable(self,table)` |
`public def addBreak(self)` |

## Members

#### `public  content` {#classformatter_1_1_formatter_1acefae5c828ca5601c6b1262f59b5ffe4}





#### `public def __init__(self)` {#classformatter_1_1_formatter_1ab2093fe878f7abc82a91ed98f553c26b}





#### `public def clone(self)` {#classformatter_1_1_formatter_1a353915c086f86945dc60930e4bee4876}





#### `public def dump(self)` {#classformatter_1_1_formatter_1aa1bdd20fec33c21d0db0bb947488f6e7}





#### `public def parseAndAdd(self,text)` {#classformatter_1_1_formatter_1a3cf38933a4ffef36febbac017c60282b}





#### `public def addRaw(self,text)` {#classformatter_1_1_formatter_1a7c97ae835341f55b7d88fa6566880bb7}





#### `public def addLine(self,text)` {#classformatter_1_1_formatter_1a04c92ecdc14393ca164ddf3b01b5e7ce}





#### `public def addLinebreak(self)` {#classformatter_1_1_formatter_1aa6d25bacc66552d95a2a5b676fc207c9}





#### `public def addHeader(self,text)` {#classformatter_1_1_formatter_1a7802e8a55a3dd109b7e6f2d92ce4c670}





#### `public def addEmphasis(self,text)` {#classformatter_1_1_formatter_1add20b71f3840c41d95575117ea087d0a}





#### `public def addList(self,textList)` {#classformatter_1_1_formatter_1a2f9ec666fd05ed9d4ee7d20f381c2e5e}





#### `public def addLink(self,text,url)` {#classformatter_1_1_formatter_1a534e7926957c7f2ac0cc4bd4eb298a23}





#### `public def addCode(self,text)` {#classformatter_1_1_formatter_1a7de7cc51b70e9ef70e63914b2aa53980}





#### `public def addCodeLink(self,text)` {#classformatter_1_1_formatter_1af8293f1f29de97895a9e9ae62f65dc99}





#### `public def addTable(self,table)` {#classformatter_1_1_formatter_1a1f07564cd3aa6bc15c667abc26902164}





#### `public def addBreak(self)` {#classformatter_1_1_formatter_1a5eb4f8d6c61f99f9fcb301bc67efc685}





# class `formatter::Markdown` {#classformatter_1_1_markdown}

```
class formatter::Markdown
  : public formatter.Formatter
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def addRaw(self,text)` |
`public def addLine(self,text,new_line)` |
`public def addLinebreak(self)` |
`public def addHeader(self,text,h)` |
`public def addEmphasis(self,text,s)` |
`public def addList(self,textList)` |
`public def addLink(self,text,url)` |
`public def addCodeLink(self,path,options)` |
`public def addCode(self,text,inline)` |
`public def addTable(self,table,noTitle)` |
`public def addBreak(self)` |

## Members

#### `public def addRaw(self,text)` {#classformatter_1_1_markdown_1aac7aa9d735646056b539eeb5ffeaa339}





#### `public def addLine(self,text,new_line)` {#classformatter_1_1_markdown_1af369e031530eadc4c2eb5977b241d36b}





#### `public def addLinebreak(self)` {#classformatter_1_1_markdown_1a7507e9793511687c2f9fb0f957d9e0e7}





#### `public def addHeader(self,text,h)` {#classformatter_1_1_markdown_1ac176aeae684d6be8b8ef7c050c6a134c}





#### `public def addEmphasis(self,text,s)` {#classformatter_1_1_markdown_1af215dec5af85abfce71233bb85634a98}





#### `public def addList(self,textList)` {#classformatter_1_1_markdown_1add17da2da8e9cbcb3a27e73aa3a88035}





#### `public def addLink(self,text,url)` {#classformatter_1_1_markdown_1aec925b6bb0a6d4119c91930eedee9216}





#### `public def addCodeLink(self,path,options)` {#classformatter_1_1_markdown_1abcef06ce43be2674c6b0fd849bb2f41c}





#### `public def addCode(self,text,inline)` {#classformatter_1_1_markdown_1a32c2a9484626ec8ddb24507b495bf5bf}





#### `public def addTable(self,table,noTitle)` {#classformatter_1_1_markdown_1a235b862ba49678b7563cf7687cbf201e}





#### `public def addBreak(self)` {#classformatter_1_1_markdown_1a3bfc5047aa04f33128907d58a793a91a}





# namespace `gather_ops_test` {#namespacegather__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`gather_ops_test::TestGatherOps`](#classgather__ops__test_1_1_test_gather_ops)    |
# class `gather_ops_test::TestGatherOps` {#classgather__ops__test_1_1_test_gather_ops}

```
class gather_ops_test::TestGatherOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_gather_ops(self)` |

## Members

#### `public def test_gather_ops(self)` {#classgather__ops__test_1_1_test_gather_ops_1ab953b5f91d0ac514b2f134b82921976f}





# namespace `gather_ranges_op_test` {#namespacegather__ranges__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`gather_ranges_op_test::TestGatherRanges`](#classgather__ranges__op__test_1_1_test_gather_ranges)    |
# class `gather_ranges_op_test::TestGatherRanges` {#classgather__ranges__op__test_1_1_test_gather_ranges}

```
class gather_ranges_op_test::TestGatherRanges
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_gather_ranges(self,`[`boarders_and_data`](#classgather__ranges__op__test_1_1_test_gather_ranges_1ae2ed18fd5b989f42ff835af5695a862f)`,gc,dc)` |

## Members

#### `public def test_gather_ranges(self,`[`boarders_and_data`](#classgather__ranges__op__test_1_1_test_gather_ranges_1ae2ed18fd5b989f42ff835af5695a862f)`,gc,dc)` {#classgather__ranges__op__test_1_1_test_gather_ranges_1ab1fa7cb451fedd973f8d0ea5e10a07c0}





# namespace `generator` {#namespacegenerator}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`generator::DocGenerator`](#classgenerator_1_1_doc_generator)    |
`class `[`generator::DocUploader`](#classgenerator_1_1_doc_uploader)    |
`class `[`generator::OpDocGenerator`](#classgenerator_1_1_op_doc_generator)    |
`class `[`generator::OperatorDoc`](#classgenerator_1_1_operator_doc)    |
`class `[`generator::OperatorEngine`](#classgenerator_1_1_operator_engine)    |
# class `generator::DocGenerator` {#classgenerator_1_1_doc_generator}

```
class generator::DocGenerator
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  formatter` |
`public  uploader` |
`public  content_body` |
`public def __init__(self,`[`formatter`](#classgenerator_1_1_doc_generator_1aa01019648077e4d6974da9efee305292)`,`[`uploader`](#classgenerator_1_1_doc_generator_1aaeae6a60aca5cdf71a9a9fd59e272e89)`)` |
`public def create_body(self)` |
`public def update(self)` |

## Members

#### `public  formatter` {#classgenerator_1_1_doc_generator_1aa01019648077e4d6974da9efee305292}





#### `public  uploader` {#classgenerator_1_1_doc_generator_1aaeae6a60aca5cdf71a9a9fd59e272e89}





#### `public  content_body` {#classgenerator_1_1_doc_generator_1a36cbd81bdf264dd2144823e3a8109a59}





#### `public def __init__(self,`[`formatter`](#classgenerator_1_1_doc_generator_1aa01019648077e4d6974da9efee305292)`,`[`uploader`](#classgenerator_1_1_doc_generator_1aaeae6a60aca5cdf71a9a9fd59e272e89)`)` {#classgenerator_1_1_doc_generator_1ac28f0ae0369f31d53ded5a3cafcb554a}





#### `public def create_body(self)` {#classgenerator_1_1_doc_generator_1af2c5361055ce3cdd2c2f349fac8d4bc1}





#### `public def update(self)` {#classgenerator_1_1_doc_generator_1ab817445f3f5be2c4c94ed20e612e635f}





# class `generator::DocUploader` {#classgenerator_1_1_doc_uploader}

```
class generator::DocUploader
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self)` |
`public def upload(self,text)` |

## Members

#### `public def __init__(self)` {#classgenerator_1_1_doc_uploader_1a3ab120f996755867e69aee5187b73617}





#### `public def upload(self,text)` {#classgenerator_1_1_doc_uploader_1af1a3095852d4be68558f6fcaa346efea}





# class `generator::OpDocGenerator` {#classgenerator_1_1_op_doc_generator}

```
class generator::OpDocGenerator
  : public generator.DocGenerator
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  operators` |
`public  engines` |
`public def getOperatorDoc(self,name,schema,priority)` |
`public def getOperatorEngine(self,name)` |
`public def getOperators(self)` |
`public def createBody(self)` |

## Members

#### `public  operators` {#classgenerator_1_1_op_doc_generator_1a9edba8d98e294756631e5203487271ee}





#### `public  engines` {#classgenerator_1_1_op_doc_generator_1a9d6e9cf30ca2f17bbd2ad1a5f58e4fa1}





#### `public def getOperatorDoc(self,name,schema,priority)` {#classgenerator_1_1_op_doc_generator_1a0bd380fd4e0f4e79fc60a229dc99a36f}





#### `public def getOperatorEngine(self,name)` {#classgenerator_1_1_op_doc_generator_1a06adb72b4be11ab42333f3a67dcf1ff1}





#### `public def getOperators(self)` {#classgenerator_1_1_op_doc_generator_1ac6cbff1fba6610004042a6baf2e54420}





#### `public def createBody(self)` {#classgenerator_1_1_op_doc_generator_1afda3bbfefcae5ac491e3207e79947c55}





# class `generator::OperatorDoc` {#classgenerator_1_1_operator_doc}

```
class generator::OperatorDoc
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  name` |
`public  schema` |
`public  priority` |
`public  engines` |
`public def __init__(self,`[`name`](#classgenerator_1_1_operator_doc_1a8dabd3cbce97329795fe5b46713e5175)`,`[`schema`](#classgenerator_1_1_operator_doc_1a6736d603e51a115f78a3d5e4ca45b902)`,`[`priority`](#classgenerator_1_1_operator_doc_1a59e7f93faa5c5a87051b1c21bd94b86d)`)` |
`public def addEngines(self,`[`engines`](#classgenerator_1_1_operator_doc_1af5650f9917e544e98c34770b9e108cbd)`)` |
`public def generateDoc(self,formatter)` |
`public def generateTable(self,formatter,tuples,title_row,title)` |
`public def generateInterface(self,formatter)` |
`public def generateCodeLink(self,formatter)` |
`public def getInfo(self,formatter,`[`name`](#classgenerator_1_1_operator_doc_1a8dabd3cbce97329795fe5b46713e5175)`,impl)` |
`public def generateDevices(self,formatter)` |
`public def generateEngines(self,formatter)` |
`public def generateSchema(self,formatter)` |

## Members

#### `public  name` {#classgenerator_1_1_operator_doc_1a8dabd3cbce97329795fe5b46713e5175}





#### `public  schema` {#classgenerator_1_1_operator_doc_1a6736d603e51a115f78a3d5e4ca45b902}





#### `public  priority` {#classgenerator_1_1_operator_doc_1a59e7f93faa5c5a87051b1c21bd94b86d}





#### `public  engines` {#classgenerator_1_1_operator_doc_1af5650f9917e544e98c34770b9e108cbd}





#### `public def __init__(self,`[`name`](#classgenerator_1_1_operator_doc_1a8dabd3cbce97329795fe5b46713e5175)`,`[`schema`](#classgenerator_1_1_operator_doc_1a6736d603e51a115f78a3d5e4ca45b902)`,`[`priority`](#classgenerator_1_1_operator_doc_1a59e7f93faa5c5a87051b1c21bd94b86d)`)` {#classgenerator_1_1_operator_doc_1a33c9bdbde03528a466c22d0e30b54d5f}





#### `public def addEngines(self,`[`engines`](#classgenerator_1_1_operator_doc_1af5650f9917e544e98c34770b9e108cbd)`)` {#classgenerator_1_1_operator_doc_1a013570c7f1cea4dc369e6a678f58b7dd}





#### `public def generateDoc(self,formatter)` {#classgenerator_1_1_operator_doc_1a9041572b1a16147c64efbf84b1678783}





#### `public def generateTable(self,formatter,tuples,title_row,title)` {#classgenerator_1_1_operator_doc_1a3a6011ceeaa84fb4dc2c13ed043ccd89}





#### `public def generateInterface(self,formatter)` {#classgenerator_1_1_operator_doc_1abd2088aae5b50f439fb0307d0735f237}





#### `public def generateCodeLink(self,formatter)` {#classgenerator_1_1_operator_doc_1aeb83a71389c13909234cf3b2038ae10c}





#### `public def getInfo(self,formatter,`[`name`](#classgenerator_1_1_operator_doc_1a8dabd3cbce97329795fe5b46713e5175)`,impl)` {#classgenerator_1_1_operator_doc_1ae2c321f738348b529ab6b235486d554c}





#### `public def generateDevices(self,formatter)` {#classgenerator_1_1_operator_doc_1a131b67ccd7552ed2bf2abcd4bd6be801}





#### `public def generateEngines(self,formatter)` {#classgenerator_1_1_operator_doc_1ab1a9c00e998ec7f2e5b06d8678db2269}





#### `public def generateSchema(self,formatter)` {#classgenerator_1_1_operator_doc_1a68679480a1b66fdc67395f7e1ebe09e9}





# class `generator::OperatorEngine` {#classgenerator_1_1_operator_engine}

```
class generator::OperatorEngine
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  op_name` |
`public  engine` |
`public def __init__(self,name)` |
`public def getDeviceImpl(self)` |
`public def generateDoc(self,formatter)` |

## Members

#### `public  op_name` {#classgenerator_1_1_operator_engine_1a2afe6a2804dca638afc7ae8b5c667477}





#### `public  engine` {#classgenerator_1_1_operator_engine_1ac12280824180af738f7466aa3805c136}





#### `public def __init__(self,name)` {#classgenerator_1_1_operator_engine_1a566344095271221a43f34878a17719e5}





#### `public def getDeviceImpl(self)` {#classgenerator_1_1_operator_engine_1a6481c6b8e0933ceca53135d41b96a7d2}





#### `public def generateDoc(self,formatter)` {#classgenerator_1_1_operator_engine_1a99f7acf6ec436e805de6e1b2dc953ba9}





# namespace `github` {#namespacegithub}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`github::GHMarkdown`](#classgithub_1_1_g_h_markdown)    |
`class `[`github::GHOpDocGenerator`](#classgithub_1_1_g_h_op_doc_generator)    |
`class `[`github::GHOpDocUploader`](#classgithub_1_1_g_h_op_doc_uploader)    |
`class `[`github::GHOperatorDoc`](#classgithub_1_1_g_h_operator_doc)    |
`class `[`github::GHOperatorEngine`](#classgithub_1_1_g_h_operator_engine)    |
# class `github::GHMarkdown` {#classgithub_1_1_g_h_markdown}

```
class github::GHMarkdown
  : public Markdown
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def addHeader(self,text,h)` |
`public def addTable(self,table,noTitle)` |

## Members

#### `public def addHeader(self,text,h)` {#classgithub_1_1_g_h_markdown_1aec2c9c47201161d67fcfca0aae6d1dbc}





#### `public def addTable(self,table,noTitle)` {#classgithub_1_1_g_h_markdown_1ab631c1f3a6e95aad32b32d8e27139a9b}





# class `github::GHOpDocGenerator` {#classgithub_1_1_g_h_op_doc_generator}

```
class github::GHOpDocGenerator
  : public OpDocGenerator
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def getOperatorDoc(self,name,schema,priority)` |
`public def getOperatorEngine(self,name)` |

## Members

#### `public def getOperatorDoc(self,name,schema,priority)` {#classgithub_1_1_g_h_op_doc_generator_1a6aea71d9fe63359709b1d23864472d44}





#### `public def getOperatorEngine(self,name)` {#classgithub_1_1_g_h_op_doc_generator_1afbbbdd5e90d8f30c7ad2bd0a4d82df74}





# class `github::GHOpDocUploader` {#classgithub_1_1_g_h_op_doc_uploader}

```
class github::GHOpDocUploader
  : public DocUploader
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self)` |
`public def upload(self,content_body)` |

## Members

#### `public def __init__(self)` {#classgithub_1_1_g_h_op_doc_uploader_1a6acf38d927282fb998acfa178d0f9857}





#### `public def upload(self,content_body)` {#classgithub_1_1_g_h_op_doc_uploader_1ada89903fd39d2bc0830fc980c6169c59}





# class `github::GHOperatorDoc` {#classgithub_1_1_g_h_operator_doc}

```
class github::GHOperatorDoc
  : public OperatorDoc
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def generateCodeLink(self,formatter)` |
`public def getInfo(self,formatter,name,impl)` |

## Members

#### `public def generateCodeLink(self,formatter)` {#classgithub_1_1_g_h_operator_doc_1a8e25624ce8e2bdde5decc9131e412fb6}





#### `public def getInfo(self,formatter,name,impl)` {#classgithub_1_1_g_h_operator_doc_1a29bec8b7607c2c3f49c3db8b466c1cbd}





# class `github::GHOperatorEngine` {#classgithub_1_1_g_h_operator_engine}

```
class github::GHOperatorEngine
  : public OperatorEngine
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def generateDoc(self,formatter,schema)` |

## Members

#### `public def generateDoc(self,formatter,schema)` {#classgithub_1_1_g_h_operator_engine_1ad9ddde0f1b8abc79e4d987686ca3b241}





# namespace `given_tensor_fill_op_test` {#namespacegiven__tensor__fill__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`given_tensor_fill_op_test::TestGivenTensorFillOps`](#classgiven__tensor__fill__op__test_1_1_test_given_tensor_fill_ops)    |
# class `given_tensor_fill_op_test::TestGivenTensorFillOps` {#classgiven__tensor__fill__op__test_1_1_test_given_tensor_fill_ops}

```
class given_tensor_fill_op_test::TestGivenTensorFillOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_given_tensor_fill(self,`[`X`](#classgiven__tensor__fill__op__test_1_1_test_given_tensor_fill_ops_1a1eb1ccfe1d5e67324b38b7cd7c4251e4)`,`[`t`](#classgiven__tensor__fill__op__test_1_1_test_given_tensor_fill_ops_1adfdc69b809eda70c77d4fad0a9a7b1df)`,gc,dc)` |

## Members

#### `public def test_given_tensor_fill(self,`[`X`](#classgiven__tensor__fill__op__test_1_1_test_given_tensor_fill_ops_1a1eb1ccfe1d5e67324b38b7cd7c4251e4)`,`[`t`](#classgiven__tensor__fill__op__test_1_1_test_given_tensor_fill_ops_1adfdc69b809eda70c77d4fad0a9a7b1df)`,gc,dc)` {#classgiven__tensor__fill__op__test_1_1_test_given_tensor_fill_ops_1a01508cd2adf3a22c2f26e30fe4ce72b0}





# namespace `gradient_check_test` {#namespacegradient__check__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`gradient_check_test::TestConcat`](#classgradient__check__test_1_1_test_concat)    |
`class `[`gradient_check_test::TestExp`](#classgradient__check__test_1_1_test_exp)    |
`class `[`gradient_check_test::TestFlatten`](#classgradient__check__test_1_1_test_flatten)    |
`class `[`gradient_check_test::TestLRN`](#classgradient__check__test_1_1_test_l_r_n)    |
`class `[`gradient_check_test::TestMakeTwoClass`](#classgradient__check__test_1_1_test_make_two_class)    |
`class `[`gradient_check_test::TestRelu`](#classgradient__check__test_1_1_test_relu)    |
`class `[`gradient_check_test::TestSigmoid`](#classgradient__check__test_1_1_test_sigmoid)    |
`class `[`gradient_check_test::TestSum`](#classgradient__check__test_1_1_test_sum)    |
`class `[`gradient_check_test::TestTanh`](#classgradient__check__test_1_1_test_tanh)    |
# class `gradient_check_test::TestConcat` {#classgradient__check__test_1_1_test_concat}

```
class gradient_check_test::TestConcat
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testConcatNHWC(self)` |
`public def testConcatNCHW(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_concat_1aab778507966e16896eeba4e2a3b4bf26}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_concat_1a425aa964e3af98b5c228bf61d7cf091a}





#### `public def testConcatNHWC(self)` {#classgradient__check__test_1_1_test_concat_1ae421bfc1ca5b5f66a3ca258eb40f248c}





#### `public def testConcatNCHW(self)` {#classgradient__check__test_1_1_test_concat_1a0c53e9e491421c6c860127dced795606}





# class `gradient_check_test::TestExp` {#classgradient__check__test_1_1_test_exp}

```
class gradient_check_test::TestExp
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testExp(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_exp_1a4da327af524a65fd2bd3f71838948d64}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_exp_1a1d3c94b5c66864060a4181c43fe4cbac}





#### `public def testExp(self)` {#classgradient__check__test_1_1_test_exp_1ad2a510af3f62d8d5f3d08cd17a03770d}





# class `gradient_check_test::TestFlatten` {#classgradient__check__test_1_1_test_flatten}

```
class gradient_check_test::TestFlatten
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testFlatten(self)` |

## Members

#### `public def testFlatten(self)` {#classgradient__check__test_1_1_test_flatten_1a16cad692a45761e47bf4dabacb5eca81}





# class `gradient_check_test::TestLRN` {#classgradient__check__test_1_1_test_l_r_n}

```
class gradient_check_test::TestLRN
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testLRN(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_l_r_n_1a80c2b3a286ca72e499e712091e1de84f}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_l_r_n_1a55ae45b2bd2937a9001e98996af94c80}





#### `public def testLRN(self)` {#classgradient__check__test_1_1_test_l_r_n_1afa99ad0c402295c466e5a98d3a7e8c0b}





# class `gradient_check_test::TestMakeTwoClass` {#classgradient__check__test_1_1_test_make_two_class}

```
class gradient_check_test::TestMakeTwoClass
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testMakeTwoClass(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_make_two_class_1a665eb92939f7ffb7fbd3f371e5741acf}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_make_two_class_1ac5fe90cd5bfe2b413bfbb6d3173a69d2}





#### `public def testMakeTwoClass(self)` {#classgradient__check__test_1_1_test_make_two_class_1a136dea1bb66c76d15a7f7dd0f2f29670}





# class `gradient_check_test::TestRelu` {#classgradient__check__test_1_1_test_relu}

```
class gradient_check_test::TestRelu
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testRelu(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_relu_1af890c9e8512814d53006a238f6b72ac3}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_relu_1ab68117397ec5765cf409d0c916199ef5}





#### `public def testRelu(self)` {#classgradient__check__test_1_1_test_relu_1a0d676d580ca852311eaf7c0fc6a57576}





# class `gradient_check_test::TestSigmoid` {#classgradient__check__test_1_1_test_sigmoid}

```
class gradient_check_test::TestSigmoid
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testSigmoid(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_sigmoid_1a45a86f6bbaad570d499d0a6079f91bea}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_sigmoid_1a7d3438904bf682044a80cd81b523fcb6}





#### `public def testSigmoid(self)` {#classgradient__check__test_1_1_test_sigmoid_1a3fbc272e2af96e7ce4c6521d5df2a68e}





# class `gradient_check_test::TestSum` {#classgradient__check__test_1_1_test_sum}

```
class gradient_check_test::TestSum
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testSum(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_sum_1aa17b33c9e918d1d380edd52aa246cb9c}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_sum_1a1796f25e83a537092e646767d4d9fd4b}





#### `public def testSum(self)` {#classgradient__check__test_1_1_test_sum_1afc2223ee8a53e633adba3ddeff90496a}





# class `gradient_check_test::TestTanh` {#classgradient__check__test_1_1_test_tanh}

```
class gradient_check_test::TestTanh
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  test_configs` |
`public def setUp(self)` |
`public def testTanh(self)` |

## Members

#### `public  test_configs` {#classgradient__check__test_1_1_test_tanh_1af5d3ebf84e40439b258443e5889f2d39}





#### `public def setUp(self)` {#classgradient__check__test_1_1_test_tanh_1ae4164d5cb012314fce6fe39bd9cdcc65}





#### `public def testTanh(self)` {#classgradient__check__test_1_1_test_tanh_1afa17751ab3a509ffaf10121bfc6f84c9}





# namespace `gradient_checker` {#namespacegradient__checker}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`gradient_checker::GradientChecker`](#classgradient__checker_1_1_gradient_checker)    |
# class `gradient_checker::GradientChecker` {#classgradient__checker_1_1_gradient_checker}




A gradient checker in Python.

This is not the most efficient way to check gradients, as the Python
interface will involve a lot of copy back and forth operations. Use at your
own risk.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,stepsize,threshold,device_option,workspace_name)` |
`public def GetLossAndGrad(self,op,grad_ops,x,input_name,grad_name,outputs_with_grads)` |
`public def CheckSimple(self,op,inputs,input_to_check,outputs_with_grads,grad_ops,input_device_options)` |

## Members

#### `public def __init__(self,stepsize,threshold,device_option,workspace_name)` {#classgradient__checker_1_1_gradient_checker_1a71be1145db755fe27c37378072f59144}





#### `public def GetLossAndGrad(self,op,grad_ops,x,input_name,grad_name,outputs_with_grads)` {#classgradient__checker_1_1_gradient_checker_1adb58e3241f184c5d6ef6ab68174c79b4}





#### `public def CheckSimple(self,op,inputs,input_to_check,outputs_with_grads,grad_ops,input_device_options)` {#classgradient__checker_1_1_gradient_checker_1ab4d8baef0e381a6fc72af61c4aa90981}



Checks the operator in a very simple fashion by stacking a sum of
squares on the top.

Inputs:
  op: the operator to be checked.
  inputs: the input data in numpy arrays.
  input_to_check: an index specifying which input blob we should
      check.
  outputs_with_grads: indices specifying which output blobs will we
      need to check gradients with. For these outputs, we will collect a
      squared sum and also feed in their gradients.
  grad_operator: the gradient operator. If not given, we will get the
      gradient operator from the gradient registry.
  input_device_options: an optional mapping from input names to
      DeviceOptions (to override the default DeviceOption)
Outputs:
  boolean: True if it passes, False if it does not pass.

# namespace `group_conv_test` {#namespacegroup__conv__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`group_conv_test::TestGroupConvolution`](#classgroup__conv__test_1_1_test_group_convolution)    |
# class `group_conv_test::TestGroupConvolution` {#classgroup__conv__test_1_1_test_group_convolution}

```
class group_conv_test::TestGroupConvolution
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_group_convolution(self,`[`stride`](#classgroup__conv__test_1_1_test_group_convolution_1a99f39daf472f403797306bb23c75949d)`,`[`pad`](#classgroup__conv__test_1_1_test_group_convolution_1a14c1219c270f3c123893f751019ac54d)`,`[`kernel`](#classgroup__conv__test_1_1_test_group_convolution_1a0a9472ae755bcf59b7dc4b9b2890c657)`,`[`size`](#classgroup__conv__test_1_1_test_group_convolution_1a6e6ac78358bcd7cf8ad14bbcb986764f)`,`[`group`](#classgroup__conv__test_1_1_test_group_convolution_1a9e306a42f5f79e1bb622b24b38cd04ed)`,`[`input_channels_per_group`](#classgroup__conv__test_1_1_test_group_convolution_1afd6e36ab4150a58b1f8b0ac0f37a6920)`,`[`output_channels_per_group`](#classgroup__conv__test_1_1_test_group_convolution_1acffcc72ec63acf9389f457cdcc8bf6aa)`,`[`batch_size`](#classgroup__conv__test_1_1_test_group_convolution_1a2667e3df7477b6a795fb81aaff627ea0)`,`[`order`](#classgroup__conv__test_1_1_test_group_convolution_1a593a80c7244f7cd6ccf7231811f91f21)`,`[`engine`](#classgroup__conv__test_1_1_test_group_convolution_1a992d7c6095729ab39a8f3927e71c457e)`,`[`use_bias`](#classgroup__conv__test_1_1_test_group_convolution_1acc7eb0d596f4780b611fc70eaf5c1a3c)`,gc,dc)` |

## Members

#### `public def test_group_convolution(self,`[`stride`](#classgroup__conv__test_1_1_test_group_convolution_1a99f39daf472f403797306bb23c75949d)`,`[`pad`](#classgroup__conv__test_1_1_test_group_convolution_1a14c1219c270f3c123893f751019ac54d)`,`[`kernel`](#classgroup__conv__test_1_1_test_group_convolution_1a0a9472ae755bcf59b7dc4b9b2890c657)`,`[`size`](#classgroup__conv__test_1_1_test_group_convolution_1a6e6ac78358bcd7cf8ad14bbcb986764f)`,`[`group`](#classgroup__conv__test_1_1_test_group_convolution_1a9e306a42f5f79e1bb622b24b38cd04ed)`,`[`input_channels_per_group`](#classgroup__conv__test_1_1_test_group_convolution_1afd6e36ab4150a58b1f8b0ac0f37a6920)`,`[`output_channels_per_group`](#classgroup__conv__test_1_1_test_group_convolution_1acffcc72ec63acf9389f457cdcc8bf6aa)`,`[`batch_size`](#classgroup__conv__test_1_1_test_group_convolution_1a2667e3df7477b6a795fb81aaff627ea0)`,`[`order`](#classgroup__conv__test_1_1_test_group_convolution_1a593a80c7244f7cd6ccf7231811f91f21)`,`[`engine`](#classgroup__conv__test_1_1_test_group_convolution_1a992d7c6095729ab39a8f3927e71c457e)`,`[`use_bias`](#classgroup__conv__test_1_1_test_group_convolution_1acc7eb0d596f4780b611fc70eaf5c1a3c)`,gc,dc)` {#classgroup__conv__test_1_1_test_group_convolution_1ab8b81329bbd9bf8434386cde153746fc}





# namespace `hsm_test` {#namespacehsm__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`hsm_test::TestHsm`](#classhsm__test_1_1_test_hsm)    |
# class `hsm_test::TestHsm` {#classhsm__test_1_1_test_hsm}

```
class hsm_test::TestHsm
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_hsm_search(self)` |
`public def test_hsm_run_once(self)` |
`public def test_hsm_forward(self)` |
`public def test_hsm_gradient(self,gc,dc)` |
`public def test_huffman_tree_hierarchy(self)` |

## Members

#### `public def test_hsm_search(self)` {#classhsm__test_1_1_test_hsm_1a1db1bdc920e945457d3ffbf9edf09fd4}





#### `public def test_hsm_run_once(self)` {#classhsm__test_1_1_test_hsm_1a316f2d04c3ed3832b12843758fc9ae1f}





#### `public def test_hsm_forward(self)` {#classhsm__test_1_1_test_hsm_1a2257162483577b028d39ac9db3c3d77a}





#### `public def test_hsm_gradient(self,gc,dc)` {#classhsm__test_1_1_test_hsm_1a5858e6ed4c829062422f3b2d7f4c18c3}





#### `public def test_huffman_tree_hierarchy(self)` {#classhsm__test_1_1_test_hsm_1a3c7c2a71ddc2f1a6c7befa11c9fae4e5}





# namespace `hypothesis_test` {#namespacehypothesis__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`hypothesis_test::TestOperators`](#classhypothesis__test_1_1_test_operators)    |
# class `hypothesis_test::TestOperators` {#classhypothesis__test_1_1_test_operators}

```
class hypothesis_test::TestOperators
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_comparison_ops(self)` |
`public def test_sum(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,gc,dc)` |
`public def test_row_mul(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,gc,dc)` |
`public def test_max(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,gc,dc)` |
`public def test_add(self)` |
`public def test_sub(self)` |
`public def test_mul(self)` |
`public def test_div(self)` |
`public def test_negative(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,gc,dc)` |
`public def test_tanh(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` |
`public def test_averaged_loss(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` |
`public def test_softsign(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`inplace`](#classhypothesis__test_1_1_test_operators_1a8eb35cfea0c1d5a4e7fae8356f29a83c)`,gc,dc)` |
`public def test_random_seed_behaviour(self,`[`device_options`](#classhypothesis__test_1_1_test_operators_1ae95bb9b87caed0508ba02a24e1c6444c)`,`[`set_seed`](#classhypothesis__test_1_1_test_operators_1a05f4b0fc7578aca8f97c567100a8315d)`)` |
`public def test_fully_connected_axis(self,`[`axis`](#classhypothesis__test_1_1_test_operators_1aad5a01e245c2fcacdcd261a1b8281298)`,`[`num_output`](#classhypothesis__test_1_1_test_operators_1a4780247fd8895624a771646b9e404744)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` |
`public def test_recurrent(self,`[`hidden_size`](#classhypothesis__test_1_1_test_operators_1a89fb658f9d1695184216cc28ffdfe3a0)`,`[`num_layers`](#classhypothesis__test_1_1_test_operators_1a0bfdb03aff98e1f7536b170d0d2e2eb3)`,`[`bidirectional`](#classhypothesis__test_1_1_test_operators_1a8b1d2557d83cd743da32e4d3e5d18b1a)`,`[`rnn_mode`](#classhypothesis__test_1_1_test_operators_1a1a3a31bae1aced2766836c92e7e86d14)`,`[`input_mode`](#classhypothesis__test_1_1_test_operators_1a86d69243a1e087a146b8bcd761407345)`,`[`dropout`](#classhypothesis__test_1_1_test_operators_1af4727052035691716355b537ac6cb115)`,`[`T`](#classhypothesis__test_1_1_test_operators_1a82465b23ff9e2b193cd52ab36c328271)`,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`D`](#classhypothesis__test_1_1_test_operators_1a0cbb751d0f8274f77fdc56a6351eadef)`)` |
`public def test_depth_concat(self,`[`ndim`](#classhypothesis__test_1_1_test_operators_1a06761bc90a2a57bac1567735f20187b8)`,`[`axis`](#classhypothesis__test_1_1_test_operators_1aad5a01e245c2fcacdcd261a1b8281298)`,`[`num_inputs`](#classhypothesis__test_1_1_test_operators_1a030da36492c2b0002fdb9012c1391047)`,gc,dc)` |
`public def test_depth_concat_with_order(self,`[`num_inputs`](#classhypothesis__test_1_1_test_operators_1a030da36492c2b0002fdb9012c1391047)`,`[`order`](#classhypothesis__test_1_1_test_operators_1aca3a92172de68033b9dd36a4c13be70e)`,gc,dc)` |
`public def test_last_n_windows(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` |
`public def test_im2col_layout(self,`[`batch_size`](#classhypothesis__test_1_1_test_operators_1abc071725fde190b50919cec9a8d2ad87)`,`[`stride`](#classhypothesis__test_1_1_test_operators_1a40beeccce6374e75f9bb63e95179c875)`,`[`pad`](#classhypothesis__test_1_1_test_operators_1a08a11466ee9db6402605d2515da1afe0)`,`[`kernel`](#classhypothesis__test_1_1_test_operators_1a21bcf3f31395aefbc3974e4fc016205f)`,`[`dilation`](#classhypothesis__test_1_1_test_operators_1ae50d6d8f7fe57180ac563b3db47a99a0)`,`[`size`](#classhypothesis__test_1_1_test_operators_1ab1950d1cf42201078915caa97c49f8a8)`,`[`channels`](#classhypothesis__test_1_1_test_operators_1a319a7d42a6b4a73d37b093d60ebb6dd6)`,gc,dc)` |
`public def test_print(self,`[`dtype`](#classhypothesis__test_1_1_test_operators_1a572c477934efea7aeb3a6e79899e36b4)`)` |
`public def test_momentum_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`momentum`](#classhypothesis__test_1_1_test_operators_1aa387ea2193f592eaddadca5a2908a2c7)`,`[`nesterov`](#classhypothesis__test_1_1_test_operators_1a42d40a65d0bcb0f66c220104a49f3ada)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,gc,dc)` |
`public def test_rmsprop_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`decay`](#classhypothesis__test_1_1_test_operators_1a52d2981374fafa13412afb275831663f)`,`[`momentum`](#classhypothesis__test_1_1_test_operators_1aa387ea2193f592eaddadca5a2908a2c7)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` |
`public def test_adagrad_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` |
`public def test_sparse_adagrad_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` |
`public def test_adam_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`beta1`](#classhypothesis__test_1_1_test_operators_1aabbdeef6c39248b70dddb56bd2d00cc2)`,`[`beta2`](#classhypothesis__test_1_1_test_operators_1a125a42b09f9376d8856f5448ad2b73bd)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`iters`](#classhypothesis__test_1_1_test_operators_1aaa8c7cc2894468c5e1634c6404158f06)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` |
`public def test_sparse_adam_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`beta1`](#classhypothesis__test_1_1_test_operators_1aabbdeef6c39248b70dddb56bd2d00cc2)`,`[`beta2`](#classhypothesis__test_1_1_test_operators_1a125a42b09f9376d8856f5448ad2b73bd)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`iters`](#classhypothesis__test_1_1_test_operators_1aaa8c7cc2894468c5e1634c6404158f06)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` |
`public def test_ftrl_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`alpha`](#classhypothesis__test_1_1_test_operators_1a1f93052ae687dd8e3583c7e03ffea198)`,`[`beta`](#classhypothesis__test_1_1_test_operators_1a3e8b6d5e9634fd3d60d4d9f0fb554cb1)`,`[`lambda1`](#classhypothesis__test_1_1_test_operators_1ab4377980b1f3ef8077cc0f67ab728690)`,`[`lambda2`](#classhypothesis__test_1_1_test_operators_1a3e0ae53fba58450a84c60dabdb675dc5)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` |
`public def test_sparse_ftrl_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`alpha`](#classhypothesis__test_1_1_test_operators_1a1f93052ae687dd8e3583c7e03ffea198)`,`[`beta`](#classhypothesis__test_1_1_test_operators_1a3e8b6d5e9634fd3d60d4d9f0fb554cb1)`,`[`lambda1`](#classhypothesis__test_1_1_test_operators_1ab4377980b1f3ef8077cc0f67ab728690)`,`[`lambda2`](#classhypothesis__test_1_1_test_operators_1a3e0ae53fba58450a84c60dabdb675dc5)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` |
`public def test_unique(self,`[`input`](#classhypothesis__test_1_1_test_operators_1af0d1492ab5e9f99a9d5946eb6d699669)`,`[`with_remapping`](#classhypothesis__test_1_1_test_operators_1a75652dabd84a0e3d2c22624286f33af9)`,gc,dc)` |
`public def test_accuracy(self,`[`prediction`](#classhypothesis__test_1_1_test_operators_1a1e1e0e7525ff869df40d41510a1af379)`,`[`labels`](#classhypothesis__test_1_1_test_operators_1a21cad4bb848051cc14ee8cf1ef3a7fbe)`,`[`top_k`](#classhypothesis__test_1_1_test_operators_1a28fcae426b01516e1ea31005efea5e1a)`,gc,dc)` |
`public def test_perplexity(self,`[`target_probabilities`](#classhypothesis__test_1_1_test_operators_1a7257d8a4e0443f72aa73b19972c81725)`,gc,dc)` |
`public def test_lengths_to_segment_ids(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` |
`public def test_lengths_range_fill(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` |
`public def test_segment_ids_to_ranges(self,gc,dc)` |
`public def test_lengths_to_ranges(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` |
`public def test_multi_class_accuracy(self,`[`prediction`](#classhypothesis__test_1_1_test_operators_1a1e1e0e7525ff869df40d41510a1af379)`,`[`labels`](#classhypothesis__test_1_1_test_operators_1a21cad4bb848051cc14ee8cf1ef3a7fbe)`,gc,dc)` |
`public def test_segment_ids_to_lengths(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` |
`public def test_lengths_to_weights(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,`[`power`](#classhypothesis__test_1_1_test_operators_1a1922ea7b9ac960a5275a72a9967d3bcd)`,gc,dc)` |
`public def test_exp(self,`[`input_tensor`](#classhypothesis__test_1_1_test_operators_1ad412cb3dc73895171394acda4661b6a4)`,gc,dc)` |
`public def test_log(self,`[`input_tensor`](#classhypothesis__test_1_1_test_operators_1ad412cb3dc73895171394acda4661b6a4)`,gc,dc)` |
`public def test_blobs_queue_threading(self,`[`num_threads`](#classhypothesis__test_1_1_test_operators_1a81d4d8380244d2ad8f8db816682e8f5a)`,`[`num_elements`](#classhypothesis__test_1_1_test_operators_1ae156a1e77934d779f0c3c6c27020c51a)`,`[`capacity`](#classhypothesis__test_1_1_test_operators_1aa0323a39246db6359ba2df9bad2ae513)`,`[`num_blobs`](#classhypothesis__test_1_1_test_operators_1ae47e70d274e38d23383598acd7943855)`,`[`do`](#classhypothesis__test_1_1_test_operators_1ac846ad74b49f8aaf63fcb6d6935adb83)`)` |
`public def test_safe_blobs_queue(self,`[`num_producers`](#classhypothesis__test_1_1_test_operators_1a5fe484c9b67ff4eee345b3ff34b36182)`,`[`num_consumers`](#classhypothesis__test_1_1_test_operators_1a5966b498a103073a04819dc66149ac27)`,`[`capacity`](#classhypothesis__test_1_1_test_operators_1aa0323a39246db6359ba2df9bad2ae513)`,`[`num_blobs`](#classhypothesis__test_1_1_test_operators_1ae47e70d274e38d23383598acd7943855)`,`[`do`](#classhypothesis__test_1_1_test_operators_1ac846ad74b49f8aaf63fcb6d6935adb83)`)` |
`public def test_squeeze_expand_dims(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,gc,dc)` |
`public def test_tt_layer(self,gc,dc)` |
`public def test_dag_net_forking(self,`[`net_type`](#classhypothesis__test_1_1_test_operators_1a9444a24af661469f015a97c3f8dc086d)`,`[`num_workers`](#classhypothesis__test_1_1_test_operators_1a3e7247400b8044b8cb39b3099e35745c)`,`[`do`](#classhypothesis__test_1_1_test_operators_1ac846ad74b49f8aaf63fcb6d6935adb83)`)` |
`public def test_slice(self,`[`input`](#classhypothesis__test_1_1_test_operators_1af0d1492ab5e9f99a9d5946eb6d699669)`,`[`slice_dim`](#classhypothesis__test_1_1_test_operators_1ac5923c92d00255fd5e2ecb91dec50b14)`,`[`a`](#classhypothesis__test_1_1_test_operators_1a5b6a31cbe0877ebee52415a69326e4ea)`,`[`b`](#classhypothesis__test_1_1_test_operators_1a257994c43db22b6bdda76c1bce527202)`,`[`is_empty`](#classhypothesis__test_1_1_test_operators_1a32dcafc5f283e03da05084c559b43458)`,gc,dc)` |
`public def test_shape(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,gc,dc)` |
`public def test_has_elements(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,gc,dc)` |
`public def test_should_stop_as_criteria_net_execution_step(self,`[`initial_iters`](#classhypothesis__test_1_1_test_operators_1a84f6247f9cb9cbc7cac95286734316e1)`,`[`max_iters`](#classhypothesis__test_1_1_test_operators_1ae8c76049dc68273492217c0a8a71b572)`)` |
`public def test_disabled_execution_step(self)` |
`public def test_iter_count_with_execution_step(self,`[`initial_iters`](#classhypothesis__test_1_1_test_operators_1a84f6247f9cb9cbc7cac95286734316e1)`,`[`num_iters`](#classhypothesis__test_1_1_test_operators_1aca1fe3431802315c7b9b103ab9f2f7ee)`)` |
`public def test_atomic_iter_with_concurrent_steps(self,`[`initial_iters`](#classhypothesis__test_1_1_test_operators_1a84f6247f9cb9cbc7cac95286734316e1)`,`[`num_iters`](#classhypothesis__test_1_1_test_operators_1aca1fe3431802315c7b9b103ab9f2f7ee)`,`[`num_nets`](#classhypothesis__test_1_1_test_operators_1afd4ceeedc3db9dcb8188cf5f3a475abe)`)` |
`public def test_cast(self,`[`a`](#classhypothesis__test_1_1_test_operators_1a5b6a31cbe0877ebee52415a69326e4ea)`,`[`src`](#classhypothesis__test_1_1_test_operators_1aac888985ad9d1baae0520c069b3cf094)`,`[`dst`](#classhypothesis__test_1_1_test_operators_1aaf64c11880a5fc334d57da551cf05afd)`,`[`use_name`](#classhypothesis__test_1_1_test_operators_1a780479771c854f1b40b771c9fe292199)`,gc,dc)` |
`public def test_constant_fill(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,`[`has_input`](#classhypothesis__test_1_1_test_operators_1a242040cac136c63c8942d150189f0f77)`,`[`has_extra_shape`](#classhypothesis__test_1_1_test_operators_1af1883166455e8a3252a1deb4d6cd71ee)`,`[`extra_shape`](#classhypothesis__test_1_1_test_operators_1aa600249650bd9a8f158dea5e561b8b37)`,gc,dc)` |
`public def test_elman_recurrent_network(self,`[`t`](#classhypothesis__test_1_1_test_operators_1a8ec8ec4ddf432a9a80170daf155d8629)`,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`d`](#classhypothesis__test_1_1_test_operators_1a14c0c7bd41bb72699a09435431e04278)`)` |
`public def test_space_to_batch(self,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`c`](#classhypothesis__test_1_1_test_operators_1ab6f36f6f856f329477c55f53a6d6854e)`,`[`h`](#classhypothesis__test_1_1_test_operators_1ad402e3ae0a60680e368617a43c06049d)`,`[`w`](#classhypothesis__test_1_1_test_operators_1adb429f55d13513a9b728f7e37b7b0fd0)`,`[`pad`](#classhypothesis__test_1_1_test_operators_1a08a11466ee9db6402605d2515da1afe0)`,`[`block_size`](#classhypothesis__test_1_1_test_operators_1a9bae577213382305214db77244cefb89)`,gc,dc)` |
`public def test_batch_to_space(self,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`c`](#classhypothesis__test_1_1_test_operators_1ab6f36f6f856f329477c55f53a6d6854e)`,`[`h`](#classhypothesis__test_1_1_test_operators_1ad402e3ae0a60680e368617a43c06049d)`,`[`w`](#classhypothesis__test_1_1_test_operators_1adb429f55d13513a9b728f7e37b7b0fd0)`,`[`pad`](#classhypothesis__test_1_1_test_operators_1a08a11466ee9db6402605d2515da1afe0)`,`[`block_size`](#classhypothesis__test_1_1_test_operators_1a9bae577213382305214db77244cefb89)`,gc,dc)` |
`public def test_scale(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`scale`](#classhypothesis__test_1_1_test_operators_1af37e1c2535000ccca296b4c2b58e8dd8)`,gc,dc)` |
`public def test_transpose(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`seed`](#classhypothesis__test_1_1_test_operators_1aebb3d0a121e007ed568f7745e4e19ab7)`,`[`null_axes`](#classhypothesis__test_1_1_test_operators_1aaba4f253e2681964e7286e072c0c4e19)`,gc,dc)` |
`public def test_string_serde(self,`[`s`](#classhypothesis__test_1_1_test_operators_1aaaf9358977d8afe2245433b66b2876e9)`)` |
`public def test_distances(self,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`dim`](#classhypothesis__test_1_1_test_operators_1ad1f80d4497f466262ea2a1b7ce08f0c5)`,gc,dc)` |
`public def test_pad_image(self,`[`pad_t`](#classhypothesis__test_1_1_test_operators_1a0736b68a00e84c9a16a05cdf8208f439)`,`[`pad_l`](#classhypothesis__test_1_1_test_operators_1aa59a255034c8194278da39805690abc0)`,`[`pad_b`](#classhypothesis__test_1_1_test_operators_1a3b4aa30fc5f1b380bad0d0040974b4b7)`,`[`pad_r`](#classhypothesis__test_1_1_test_operators_1a9ef299f9742931f6facb2e24a7713490)`,`[`size`](#classhypothesis__test_1_1_test_operators_1ab1950d1cf42201078915caa97c49f8a8)`,`[`input_channels`](#classhypothesis__test_1_1_test_operators_1a19266f5d65d40e444e218d75c6b019e2)`,`[`batch_size`](#classhypothesis__test_1_1_test_operators_1abc071725fde190b50919cec9a8d2ad87)`,`[`order`](#classhypothesis__test_1_1_test_operators_1aca3a92172de68033b9dd36a4c13be70e)`,`[`mode`](#classhypothesis__test_1_1_test_operators_1afa31b7474606401d3b2e1e3544526eb9)`,gc,dc)` |
`public def test_instance_norm(self,`[`size`](#classhypothesis__test_1_1_test_operators_1ab1950d1cf42201078915caa97c49f8a8)`,`[`input_channels`](#classhypothesis__test_1_1_test_operators_1a19266f5d65d40e444e218d75c6b019e2)`,`[`batch_size`](#classhypothesis__test_1_1_test_operators_1abc071725fde190b50919cec9a8d2ad87)`,`[`order`](#classhypothesis__test_1_1_test_operators_1aca3a92172de68033b9dd36a4c13be70e)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` |
`public def test_unsafe_coalesce(self,`[`sizes`](#classhypothesis__test_1_1_test_operators_1ae8a6f8691b5bb00b83c3c16c5860726a)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,gc,dc)` |
`public def test_normalize(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` |
`public def test_sparse_to_dense(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`extra_dim`](#classhypothesis__test_1_1_test_operators_1a6c880b0c3b157745bc440bdcd786b579)`,gc,dc)` |
`public def test_dot_product(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,gc,dc)` |
`public def test_dot_product_with_paddding(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,`[`K`](#classhypothesis__test_1_1_test_operators_1a21e8f2e180f95644af139f51361e83a9)`,`[`pad_value`](#classhypothesis__test_1_1_test_operators_1a2b246511527c4a32655593446772848e)`,gc,dc)` |
`public def test_dot_product_with_rep_paddding(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,`[`pad_value`](#classhypothesis__test_1_1_test_operators_1a2b246511527c4a32655593446772848e)`,gc,dc)` |
`public def test_ensure_dense(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,gc,dc)` |
`public def test_accumulate_histogram_op(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,`[`num_buckets`](#classhypothesis__test_1_1_test_operators_1a3d5a3cea377bcea9ab4d9d9cf4af52eb)`,gc,dc)` |

## Members

#### `public def test_comparison_ops(self)` {#classhypothesis__test_1_1_test_operators_1a07ea39f47cb767b268d59fbd16890a9e}





#### `public def test_sum(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a43b562cc53113eb6eb1e969fc0dd721c}





#### `public def test_row_mul(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ad65e56de5a9a4e19aa09c96eeff50527}





#### `public def test_max(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ad335ad5864e1e1a8b452b57f4e4f0b29}





#### `public def test_add(self)` {#classhypothesis__test_1_1_test_operators_1abe144b7d85a7478ddd477295ae6490cf}





#### `public def test_sub(self)` {#classhypothesis__test_1_1_test_operators_1ac15f7b343ad6518a090dcc75b2a606e5}





#### `public def test_mul(self)` {#classhypothesis__test_1_1_test_operators_1a6956d341482e98453182a745ab77750d}





#### `public def test_div(self)` {#classhypothesis__test_1_1_test_operators_1a73e9d578fd3da4008c20223013a24d10}





#### `public def test_negative(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a2f27d49570a52165f23ad0b9d1b4e998}





#### `public def test_tanh(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a65f146a4a93684ca82389daeb51754e4}





#### `public def test_averaged_loss(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ac7f2b5a4d2f466aa51200850483a48f4}





#### `public def test_softsign(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`inplace`](#classhypothesis__test_1_1_test_operators_1a8eb35cfea0c1d5a4e7fae8356f29a83c)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a3e0fb7464e29e2d6d4fe9c625eba8899}





#### `public def test_random_seed_behaviour(self,`[`device_options`](#classhypothesis__test_1_1_test_operators_1ae95bb9b87caed0508ba02a24e1c6444c)`,`[`set_seed`](#classhypothesis__test_1_1_test_operators_1a05f4b0fc7578aca8f97c567100a8315d)`)` {#classhypothesis__test_1_1_test_operators_1a641f3f117fbcb7f0917a9ad43e6e6c43}





#### `public def test_fully_connected_axis(self,`[`axis`](#classhypothesis__test_1_1_test_operators_1aad5a01e245c2fcacdcd261a1b8281298)`,`[`num_output`](#classhypothesis__test_1_1_test_operators_1a4780247fd8895624a771646b9e404744)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a8d19b42c4d807e459b98772bd4ae2e34}





#### `public def test_recurrent(self,`[`hidden_size`](#classhypothesis__test_1_1_test_operators_1a89fb658f9d1695184216cc28ffdfe3a0)`,`[`num_layers`](#classhypothesis__test_1_1_test_operators_1a0bfdb03aff98e1f7536b170d0d2e2eb3)`,`[`bidirectional`](#classhypothesis__test_1_1_test_operators_1a8b1d2557d83cd743da32e4d3e5d18b1a)`,`[`rnn_mode`](#classhypothesis__test_1_1_test_operators_1a1a3a31bae1aced2766836c92e7e86d14)`,`[`input_mode`](#classhypothesis__test_1_1_test_operators_1a86d69243a1e087a146b8bcd761407345)`,`[`dropout`](#classhypothesis__test_1_1_test_operators_1af4727052035691716355b537ac6cb115)`,`[`T`](#classhypothesis__test_1_1_test_operators_1a82465b23ff9e2b193cd52ab36c328271)`,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`D`](#classhypothesis__test_1_1_test_operators_1a0cbb751d0f8274f77fdc56a6351eadef)`)` {#classhypothesis__test_1_1_test_operators_1af5e05ffd1c7df3bb68ed48b491d5d5f5}





#### `public def test_depth_concat(self,`[`ndim`](#classhypothesis__test_1_1_test_operators_1a06761bc90a2a57bac1567735f20187b8)`,`[`axis`](#classhypothesis__test_1_1_test_operators_1aad5a01e245c2fcacdcd261a1b8281298)`,`[`num_inputs`](#classhypothesis__test_1_1_test_operators_1a030da36492c2b0002fdb9012c1391047)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a0663ff811471a0d813ed593ab090e73d}





#### `public def test_depth_concat_with_order(self,`[`num_inputs`](#classhypothesis__test_1_1_test_operators_1a030da36492c2b0002fdb9012c1391047)`,`[`order`](#classhypothesis__test_1_1_test_operators_1aca3a92172de68033b9dd36a4c13be70e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ac47d9e271ad8af779d95d1f3567113d9}





#### `public def test_last_n_windows(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a708d00d445e26a31d3b227ccdc7850a8}





#### `public def test_im2col_layout(self,`[`batch_size`](#classhypothesis__test_1_1_test_operators_1abc071725fde190b50919cec9a8d2ad87)`,`[`stride`](#classhypothesis__test_1_1_test_operators_1a40beeccce6374e75f9bb63e95179c875)`,`[`pad`](#classhypothesis__test_1_1_test_operators_1a08a11466ee9db6402605d2515da1afe0)`,`[`kernel`](#classhypothesis__test_1_1_test_operators_1a21bcf3f31395aefbc3974e4fc016205f)`,`[`dilation`](#classhypothesis__test_1_1_test_operators_1ae50d6d8f7fe57180ac563b3db47a99a0)`,`[`size`](#classhypothesis__test_1_1_test_operators_1ab1950d1cf42201078915caa97c49f8a8)`,`[`channels`](#classhypothesis__test_1_1_test_operators_1a319a7d42a6b4a73d37b093d60ebb6dd6)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a9e690d948d667f3d904f35589a645c5e}





#### `public def test_print(self,`[`dtype`](#classhypothesis__test_1_1_test_operators_1a572c477934efea7aeb3a6e79899e36b4)`)` {#classhypothesis__test_1_1_test_operators_1a45d580d636c74065818ccdb0164ed92a}





#### `public def test_momentum_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`momentum`](#classhypothesis__test_1_1_test_operators_1aa387ea2193f592eaddadca5a2908a2c7)`,`[`nesterov`](#classhypothesis__test_1_1_test_operators_1a42d40a65d0bcb0f66c220104a49f3ada)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ac4a63f5355e9fa2089a3fcf42b1162d1}





#### `public def test_rmsprop_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`decay`](#classhypothesis__test_1_1_test_operators_1a52d2981374fafa13412afb275831663f)`,`[`momentum`](#classhypothesis__test_1_1_test_operators_1aa387ea2193f592eaddadca5a2908a2c7)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a6f1f0aca6813f558148dbb1d19764c54}





#### `public def test_adagrad_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a7a3a19c0ed11ac03f3b4f678aa424d57}





#### `public def test_sparse_adagrad_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a5dfe0b9b9a2dd6ea9bf81f9932833115}





#### `public def test_adam_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`beta1`](#classhypothesis__test_1_1_test_operators_1aabbdeef6c39248b70dddb56bd2d00cc2)`,`[`beta2`](#classhypothesis__test_1_1_test_operators_1a125a42b09f9376d8856f5448ad2b73bd)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`iters`](#classhypothesis__test_1_1_test_operators_1aaa8c7cc2894468c5e1634c6404158f06)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1afe014ad5614478a005de7923a9e1e45c}





#### `public def test_sparse_adam_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`beta1`](#classhypothesis__test_1_1_test_operators_1aabbdeef6c39248b70dddb56bd2d00cc2)`,`[`beta2`](#classhypothesis__test_1_1_test_operators_1a125a42b09f9376d8856f5448ad2b73bd)`,`[`lr`](#classhypothesis__test_1_1_test_operators_1a41761a7c7740b8ec4b1db20711a2f572)`,`[`iters`](#classhypothesis__test_1_1_test_operators_1aaa8c7cc2894468c5e1634c6404158f06)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1aee128913792c1042fa2e5e70042b960b}





#### `public def test_ftrl_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`alpha`](#classhypothesis__test_1_1_test_operators_1a1f93052ae687dd8e3583c7e03ffea198)`,`[`beta`](#classhypothesis__test_1_1_test_operators_1a3e8b6d5e9634fd3d60d4d9f0fb554cb1)`,`[`lambda1`](#classhypothesis__test_1_1_test_operators_1ab4377980b1f3ef8077cc0f67ab728690)`,`[`lambda2`](#classhypothesis__test_1_1_test_operators_1a3e0ae53fba58450a84c60dabdb675dc5)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ac9909997c145d3fda27a7dfb440054e8}





#### `public def test_sparse_ftrl_sgd(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,`[`alpha`](#classhypothesis__test_1_1_test_operators_1a1f93052ae687dd8e3583c7e03ffea198)`,`[`beta`](#classhypothesis__test_1_1_test_operators_1a3e8b6d5e9634fd3d60d4d9f0fb554cb1)`,`[`lambda1`](#classhypothesis__test_1_1_test_operators_1ab4377980b1f3ef8077cc0f67ab728690)`,`[`lambda2`](#classhypothesis__test_1_1_test_operators_1a3e0ae53fba58450a84c60dabdb675dc5)`,`[`engine`](#classhypothesis__test_1_1_test_operators_1a300a6a773e56b336e080d28be91761a7)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a7e0f4c7520f1f3177e62eecaa54b4c49}





#### `public def test_unique(self,`[`input`](#classhypothesis__test_1_1_test_operators_1af0d1492ab5e9f99a9d5946eb6d699669)`,`[`with_remapping`](#classhypothesis__test_1_1_test_operators_1a75652dabd84a0e3d2c22624286f33af9)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1af059b0a75c46e30edaeaf904cfae8e32}





#### `public def test_accuracy(self,`[`prediction`](#classhypothesis__test_1_1_test_operators_1a1e1e0e7525ff869df40d41510a1af379)`,`[`labels`](#classhypothesis__test_1_1_test_operators_1a21cad4bb848051cc14ee8cf1ef3a7fbe)`,`[`top_k`](#classhypothesis__test_1_1_test_operators_1a28fcae426b01516e1ea31005efea5e1a)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ae93cc5c400b776af7556a18a96c92c15}





#### `public def test_perplexity(self,`[`target_probabilities`](#classhypothesis__test_1_1_test_operators_1a7257d8a4e0443f72aa73b19972c81725)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ac8f8af180f2df1fd5a84a619aee70203}





#### `public def test_lengths_to_segment_ids(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ae930a79911993e9890b77ccb5b16448e}





#### `public def test_lengths_range_fill(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ab5187813527415e4c21fd8af6ef33c87}





#### `public def test_segment_ids_to_ranges(self,gc,dc)` {#classhypothesis__test_1_1_test_operators_1afee540d551aa4928437196f73b095a40}





#### `public def test_lengths_to_ranges(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a319bffeec5e5a4cb27753db396ea027d}





#### `public def test_multi_class_accuracy(self,`[`prediction`](#classhypothesis__test_1_1_test_operators_1a1e1e0e7525ff869df40d41510a1af379)`,`[`labels`](#classhypothesis__test_1_1_test_operators_1a21cad4bb848051cc14ee8cf1ef3a7fbe)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a638eebe49a1867277a6153b3ac8e8918}





#### `public def test_segment_ids_to_lengths(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a002783a960d7f64a57e217eb64f619d8}





#### `public def test_lengths_to_weights(self,`[`lengths`](#classhypothesis__test_1_1_test_operators_1af2713261e18dfd9797878b9aed74d311)`,`[`power`](#classhypothesis__test_1_1_test_operators_1a1922ea7b9ac960a5275a72a9967d3bcd)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1aa388f17aae0c4ee788bc40f1f32e01f8}





#### `public def test_exp(self,`[`input_tensor`](#classhypothesis__test_1_1_test_operators_1ad412cb3dc73895171394acda4661b6a4)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a000a7e7c04b5d8b12a87b4812562478e}





#### `public def test_log(self,`[`input_tensor`](#classhypothesis__test_1_1_test_operators_1ad412cb3dc73895171394acda4661b6a4)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ab1330aa295a3a5ae089c195d267a8d00}





#### `public def test_blobs_queue_threading(self,`[`num_threads`](#classhypothesis__test_1_1_test_operators_1a81d4d8380244d2ad8f8db816682e8f5a)`,`[`num_elements`](#classhypothesis__test_1_1_test_operators_1ae156a1e77934d779f0c3c6c27020c51a)`,`[`capacity`](#classhypothesis__test_1_1_test_operators_1aa0323a39246db6359ba2df9bad2ae513)`,`[`num_blobs`](#classhypothesis__test_1_1_test_operators_1ae47e70d274e38d23383598acd7943855)`,`[`do`](#classhypothesis__test_1_1_test_operators_1ac846ad74b49f8aaf63fcb6d6935adb83)`)` {#classhypothesis__test_1_1_test_operators_1a452573d22e123ad19c2f27c36e82b4cb}



- Construct matrices of size N x D
- Start K threads
- Push all N rows into the queue of capacity C
- Pull all N rows out of the queue.
- Verify that the output matrices are permutation of the rows of the
  original matrices.

#### `public def test_safe_blobs_queue(self,`[`num_producers`](#classhypothesis__test_1_1_test_operators_1a5fe484c9b67ff4eee345b3ff34b36182)`,`[`num_consumers`](#classhypothesis__test_1_1_test_operators_1a5966b498a103073a04819dc66149ac27)`,`[`capacity`](#classhypothesis__test_1_1_test_operators_1aa0323a39246db6359ba2df9bad2ae513)`,`[`num_blobs`](#classhypothesis__test_1_1_test_operators_1ae47e70d274e38d23383598acd7943855)`,`[`do`](#classhypothesis__test_1_1_test_operators_1ac846ad74b49f8aaf63fcb6d6935adb83)`)` {#classhypothesis__test_1_1_test_operators_1acd3779a7cc5c6867434fefc76aa2839a}





#### `public def test_squeeze_expand_dims(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ade11e0cd5c8d814035f1500e620a4afa}





#### `public def test_tt_layer(self,gc,dc)` {#classhypothesis__test_1_1_test_operators_1abe1805d48c961c5671acaf9a1b4fa511}





#### `public def test_dag_net_forking(self,`[`net_type`](#classhypothesis__test_1_1_test_operators_1a9444a24af661469f015a97c3f8dc086d)`,`[`num_workers`](#classhypothesis__test_1_1_test_operators_1a3e7247400b8044b8cb39b3099e35745c)`,`[`do`](#classhypothesis__test_1_1_test_operators_1ac846ad74b49f8aaf63fcb6d6935adb83)`)` {#classhypothesis__test_1_1_test_operators_1a83edcc7e3ca0e26111aa1ae4615adfdb}





#### `public def test_slice(self,`[`input`](#classhypothesis__test_1_1_test_operators_1af0d1492ab5e9f99a9d5946eb6d699669)`,`[`slice_dim`](#classhypothesis__test_1_1_test_operators_1ac5923c92d00255fd5e2ecb91dec50b14)`,`[`a`](#classhypothesis__test_1_1_test_operators_1a5b6a31cbe0877ebee52415a69326e4ea)`,`[`b`](#classhypothesis__test_1_1_test_operators_1a257994c43db22b6bdda76c1bce527202)`,`[`is_empty`](#classhypothesis__test_1_1_test_operators_1a32dcafc5f283e03da05084c559b43458)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a6d816f05476079f5f95c157580b73ab0}





#### `public def test_shape(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1afcfdcf4c74c06c804f4c60b6c24bba9c}





#### `public def test_has_elements(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1abc3f9a94c358272b0bdfa0530ab661a3}





#### `public def test_should_stop_as_criteria_net_execution_step(self,`[`initial_iters`](#classhypothesis__test_1_1_test_operators_1a84f6247f9cb9cbc7cac95286734316e1)`,`[`max_iters`](#classhypothesis__test_1_1_test_operators_1ae8c76049dc68273492217c0a8a71b572)`)` {#classhypothesis__test_1_1_test_operators_1a4a2a5e44a767c96e2705b97ed5139c4b}





#### `public def test_disabled_execution_step(self)` {#classhypothesis__test_1_1_test_operators_1a9ffb37f9b8ae35587b21f43670149748}





#### `public def test_iter_count_with_execution_step(self,`[`initial_iters`](#classhypothesis__test_1_1_test_operators_1a84f6247f9cb9cbc7cac95286734316e1)`,`[`num_iters`](#classhypothesis__test_1_1_test_operators_1aca1fe3431802315c7b9b103ab9f2f7ee)`)` {#classhypothesis__test_1_1_test_operators_1aa8c178c39d17ff91aa7a8f1ba8188c72}





#### `public def test_atomic_iter_with_concurrent_steps(self,`[`initial_iters`](#classhypothesis__test_1_1_test_operators_1a84f6247f9cb9cbc7cac95286734316e1)`,`[`num_iters`](#classhypothesis__test_1_1_test_operators_1aca1fe3431802315c7b9b103ab9f2f7ee)`,`[`num_nets`](#classhypothesis__test_1_1_test_operators_1afd4ceeedc3db9dcb8188cf5f3a475abe)`)` {#classhypothesis__test_1_1_test_operators_1a3bf246b0a14c927e3956ce04355f8abf}





#### `public def test_cast(self,`[`a`](#classhypothesis__test_1_1_test_operators_1a5b6a31cbe0877ebee52415a69326e4ea)`,`[`src`](#classhypothesis__test_1_1_test_operators_1aac888985ad9d1baae0520c069b3cf094)`,`[`dst`](#classhypothesis__test_1_1_test_operators_1aaf64c11880a5fc334d57da551cf05afd)`,`[`use_name`](#classhypothesis__test_1_1_test_operators_1a780479771c854f1b40b771c9fe292199)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a56762276acef0f33e7b2a5fe1231c62f}





#### `public def test_constant_fill(self,`[`data`](#classhypothesis__test_1_1_test_operators_1a1cac41037609041e377d0ecb3db8fda8)`,`[`has_input`](#classhypothesis__test_1_1_test_operators_1a242040cac136c63c8942d150189f0f77)`,`[`has_extra_shape`](#classhypothesis__test_1_1_test_operators_1af1883166455e8a3252a1deb4d6cd71ee)`,`[`extra_shape`](#classhypothesis__test_1_1_test_operators_1aa600249650bd9a8f158dea5e561b8b37)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ab22819b5187ba021804cf7fc98381f23}





#### `public def test_elman_recurrent_network(self,`[`t`](#classhypothesis__test_1_1_test_operators_1a8ec8ec4ddf432a9a80170daf155d8629)`,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`d`](#classhypothesis__test_1_1_test_operators_1a14c0c7bd41bb72699a09435431e04278)`)` {#classhypothesis__test_1_1_test_operators_1ab1770d973d1eac58f21818e5438617d0}





#### `public def test_space_to_batch(self,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`c`](#classhypothesis__test_1_1_test_operators_1ab6f36f6f856f329477c55f53a6d6854e)`,`[`h`](#classhypothesis__test_1_1_test_operators_1ad402e3ae0a60680e368617a43c06049d)`,`[`w`](#classhypothesis__test_1_1_test_operators_1adb429f55d13513a9b728f7e37b7b0fd0)`,`[`pad`](#classhypothesis__test_1_1_test_operators_1a08a11466ee9db6402605d2515da1afe0)`,`[`block_size`](#classhypothesis__test_1_1_test_operators_1a9bae577213382305214db77244cefb89)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a62eee593059344bcf7a5443958fb0f2d}





#### `public def test_batch_to_space(self,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`c`](#classhypothesis__test_1_1_test_operators_1ab6f36f6f856f329477c55f53a6d6854e)`,`[`h`](#classhypothesis__test_1_1_test_operators_1ad402e3ae0a60680e368617a43c06049d)`,`[`w`](#classhypothesis__test_1_1_test_operators_1adb429f55d13513a9b728f7e37b7b0fd0)`,`[`pad`](#classhypothesis__test_1_1_test_operators_1a08a11466ee9db6402605d2515da1afe0)`,`[`block_size`](#classhypothesis__test_1_1_test_operators_1a9bae577213382305214db77244cefb89)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1af6965e23a4903e6e4bc9aa3dcd20c12a}





#### `public def test_scale(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,`[`scale`](#classhypothesis__test_1_1_test_operators_1af37e1c2535000ccca296b4c2b58e8dd8)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ae23200e6c68ef88b89079a0f28060864}





#### `public def test_transpose(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`seed`](#classhypothesis__test_1_1_test_operators_1aebb3d0a121e007ed568f7745e4e19ab7)`,`[`null_axes`](#classhypothesis__test_1_1_test_operators_1aaba4f253e2681964e7286e072c0c4e19)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ae1d9c4e261e66f47167db533d153d42c}





#### `public def test_string_serde(self,`[`s`](#classhypothesis__test_1_1_test_operators_1aaaf9358977d8afe2245433b66b2876e9)`)` {#classhypothesis__test_1_1_test_operators_1ae313bb979d29c9b2da141dc2d19bfae1}





#### `public def test_distances(self,`[`n`](#classhypothesis__test_1_1_test_operators_1a25c2f626f99bc8e358859bffed125856)`,`[`dim`](#classhypothesis__test_1_1_test_operators_1ad1f80d4497f466262ea2a1b7ce08f0c5)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a064292e839f4d08b74e393e7418fe398}





#### `public def test_pad_image(self,`[`pad_t`](#classhypothesis__test_1_1_test_operators_1a0736b68a00e84c9a16a05cdf8208f439)`,`[`pad_l`](#classhypothesis__test_1_1_test_operators_1aa59a255034c8194278da39805690abc0)`,`[`pad_b`](#classhypothesis__test_1_1_test_operators_1a3b4aa30fc5f1b380bad0d0040974b4b7)`,`[`pad_r`](#classhypothesis__test_1_1_test_operators_1a9ef299f9742931f6facb2e24a7713490)`,`[`size`](#classhypothesis__test_1_1_test_operators_1ab1950d1cf42201078915caa97c49f8a8)`,`[`input_channels`](#classhypothesis__test_1_1_test_operators_1a19266f5d65d40e444e218d75c6b019e2)`,`[`batch_size`](#classhypothesis__test_1_1_test_operators_1abc071725fde190b50919cec9a8d2ad87)`,`[`order`](#classhypothesis__test_1_1_test_operators_1aca3a92172de68033b9dd36a4c13be70e)`,`[`mode`](#classhypothesis__test_1_1_test_operators_1afa31b7474606401d3b2e1e3544526eb9)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a581ae9f8b30b428e2165c4d73c9a5d2c}





#### `public def test_instance_norm(self,`[`size`](#classhypothesis__test_1_1_test_operators_1ab1950d1cf42201078915caa97c49f8a8)`,`[`input_channels`](#classhypothesis__test_1_1_test_operators_1a19266f5d65d40e444e218d75c6b019e2)`,`[`batch_size`](#classhypothesis__test_1_1_test_operators_1abc071725fde190b50919cec9a8d2ad87)`,`[`order`](#classhypothesis__test_1_1_test_operators_1aca3a92172de68033b9dd36a4c13be70e)`,`[`epsilon`](#classhypothesis__test_1_1_test_operators_1addb5b68ada1724dd3323ba7eb15fb01e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1aef0a35b3e7a57f01cadf358169188fa6}





#### `public def test_unsafe_coalesce(self,`[`sizes`](#classhypothesis__test_1_1_test_operators_1ae8a6f8691b5bb00b83c3c16c5860726a)`,`[`in_place`](#classhypothesis__test_1_1_test_operators_1a3b2e2433d7fa0ec9e85e95e3fae23d4b)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a9448399824df5e2eb5948c847b4f306a}





#### `public def test_normalize(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a1ee9551985bd424fe71f7063393edbb8}





#### `public def test_sparse_to_dense(self,`[`X`](#classhypothesis__test_1_1_test_operators_1a782b4ac4496217bbca8f174720c4620e)`,`[`extra_dim`](#classhypothesis__test_1_1_test_operators_1a6c880b0c3b157745bc440bdcd786b579)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a7d12961153dc24a8f90ca043f04c4a13}





#### `public def test_dot_product(self,`[`inputs`](#classhypothesis__test_1_1_test_operators_1a7162511aec941a7459bbd1e52ad374f5)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a66306925e7ebda48035d9df1327d7676}





#### `public def test_dot_product_with_paddding(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,`[`K`](#classhypothesis__test_1_1_test_operators_1a21e8f2e180f95644af139f51361e83a9)`,`[`pad_value`](#classhypothesis__test_1_1_test_operators_1a2b246511527c4a32655593446772848e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1acf3d408adb2bfc25b22123de563d0bfc}





#### `public def test_dot_product_with_rep_paddding(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,`[`pad_value`](#classhypothesis__test_1_1_test_operators_1a2b246511527c4a32655593446772848e)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a827d0a0719392644df56bffc379da623}





#### `public def test_ensure_dense(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1ac8751bb2d30602ad9955e1564a6f1188}





#### `public def test_accumulate_histogram_op(self,`[`N`](#classhypothesis__test_1_1_test_operators_1a77e17493fba108db0d4be6f5910a0339)`,`[`M`](#classhypothesis__test_1_1_test_operators_1a8647a5325a9e8e83bcad2639b3562616)`,`[`num_buckets`](#classhypothesis__test_1_1_test_operators_1a3d5a3cea377bcea9ab4d9d9cf4af52eb)`,gc,dc)` {#classhypothesis__test_1_1_test_operators_1a44571eda9403f0b16d3dc0d15bbb328c}





# namespace `hypothesis_test_util` {#namespacehypothesis__test__util}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`hypothesis_test_util::HypothesisTestCase`](#classhypothesis__test__util_1_1_hypothesis_test_case)    |
# class `hypothesis_test_util::HypothesisTestCase` {#classhypothesis__test__util_1_1_hypothesis_test_case}

```
class hypothesis_test_util::HypothesisTestCase
  : public test_util.TestCase
```  



A unittest.TestCase subclass with some helper functions for
utilizing the `hypothesis` (hypothesis.readthedocs.io) library.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def assertDeviceChecks(self,`[`device_options`](#namespacehypothesis__test__util_1a469ff94bc7a0fc3c317401a1ff957100)`,op,inputs,outputs_to_check,input_device_options,threshold)` |
`public def assertGradientChecks(self,device_option,op,inputs,outputs_to_check,outputs_with_grads,grad_ops,threshold,stepsize,input_device_options)` |
`public def assertReferenceChecks(self,device_option,op,inputs,reference,input_device_options,threshold,output_to_grad,grad_reference,atol,outputs_to_check)` |
`public def assertValidationChecks(self,device_option,op,inputs,validator,input_device_options,as_kwargs)` |

## Members

#### `public def assertDeviceChecks(self,`[`device_options`](#namespacehypothesis__test__util_1a469ff94bc7a0fc3c317401a1ff957100)`,op,inputs,outputs_to_check,input_device_options,threshold)` {#classhypothesis__test__util_1_1_hypothesis_test_case_1a8e5a5ea6c1ca06b2ee50a56d10b0a0d5}



Asserts that the operator computes the same outputs, regardless of
which device it is executed on.

Useful for checking the consistency of GPU and CPU
implementations of operators.

Usage example:

    @given(inputs=hu.tensors(n=2), in_place=st.booleans(), **hu.gcs)
    def test_sum(self, inputs, in_place, gc, dc):
op = core.CreateOperator("Sum", ["X1", "X2"],
                                ["Y" if not in_place else "X1"])
X1, X2 = inputs
self.assertDeviceChecks(dc, op, [X1, X2], [0])

#### `public def assertGradientChecks(self,device_option,op,inputs,outputs_to_check,outputs_with_grads,grad_ops,threshold,stepsize,input_device_options)` {#classhypothesis__test__util_1_1_hypothesis_test_case_1a898f5288da927a19f596ec212e065017}



Implements a standard numerical gradient checker for the operator
in question.

Useful for checking the consistency of the forward and
backward implementations of operators.

Usage example:

    @given(inputs=hu.tensors(n=2), in_place=st.booleans(), **hu.gcs)
    def test_sum(self, inputs, in_place, gc, dc):
op = core.CreateOperator("Sum", ["X1", "X2"],
                                ["Y" if not in_place else "X1"])
X1, X2 = inputs
self.assertGradientChecks(gc, op, [X1, X2], 0, [0])

#### `public def assertReferenceChecks(self,device_option,op,inputs,reference,input_device_options,threshold,output_to_grad,grad_reference,atol,outputs_to_check)` {#classhypothesis__test__util_1_1_hypothesis_test_case_1a26430279848a5a6689f63432979bd61a}



This runs the reference Python function implementation
(effectively calling `reference(*inputs)`, and compares that
to the output of output, with an absolute/relative tolerance
given by the `threshold` parameter.

Useful for checking the implementation matches the Python
(typically NumPy) implementation of the same functionality.

Usage example:

    @given(X=hu.tensor(), inplace=st.booleans(), **hu.gcs)
    def test_softsign(self, X, inplace, gc, dc):
op = core.CreateOperator(
    "Softsign", ["X"], ["X" if inplace else "Y"])

def softsign(X):
    return (X / (1 + np.abs(X)),)

self.assertReferenceChecks(gc, op, [X], softsign)

#### `public def assertValidationChecks(self,device_option,op,inputs,validator,input_device_options,as_kwargs)` {#classhypothesis__test__util_1_1_hypothesis_test_case_1af11736004a4627799bf4eb63686797b3}





# namespace `index_ops_test` {#namespaceindex__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`index_ops_test::TestIndexOps`](#classindex__ops__test_1_1_test_index_ops)    |
# class `index_ops_test::TestIndexOps` {#classindex__ops__test_1_1_test_index_ops}

```
class index_ops_test::TestIndexOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_string_index_ops(self)` |
`public def test_int_index_ops(self)` |
`public def test_long_index_ops(self)` |

## Members

#### `public def test_string_index_ops(self)` {#classindex__ops__test_1_1_test_index_ops_1a73e077112eb4eb9d721c3d591f0af8d8}





#### `public def test_int_index_ops(self)` {#classindex__ops__test_1_1_test_index_ops_1aa0bc0d248df0265088691707c7525d2a}





#### `public def test_long_index_ops(self)` {#classindex__ops__test_1_1_test_index_ops_1a2b7c8b6430583236c56a6c25c225059a}





# namespace `instance_norm_test` {#namespaceinstance__norm__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`instance_norm_test::TestInstanceNorm`](#classinstance__norm__test_1_1_test_instance_norm)    |
# class `instance_norm_test::TestInstanceNorm` {#classinstance__norm__test_1_1_test_instance_norm}

```
class instance_norm_test::TestInstanceNorm
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_instance_norm_gradients(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`)` |
`public def test_instance_norm_layout(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`)` |
`public def test_instance_norm_reference_check(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`,`[`inplace`](#classinstance__norm__test_1_1_test_instance_norm_1a42528d74365ee9bc0df751adc4ba1af3)`)` |
`public def test_instance_norm_device_check(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`)` |
`public def test_instance_norm_cnn_helper(self,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`,`[`is_test`](#classinstance__norm__test_1_1_test_instance_norm_1a09420b36a35fed5a87b01fce018373af)`)` |

## Members

#### `public def test_instance_norm_gradients(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`)` {#classinstance__norm__test_1_1_test_instance_norm_1a0aaec54c364dd4845753557f8c271852}





#### `public def test_instance_norm_layout(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`)` {#classinstance__norm__test_1_1_test_instance_norm_1a69d262292c7a2fb2c52e00a3fa88ad92}





#### `public def test_instance_norm_reference_check(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`,`[`inplace`](#classinstance__norm__test_1_1_test_instance_norm_1a42528d74365ee9bc0df751adc4ba1af3)`)` {#classinstance__norm__test_1_1_test_instance_norm_1a7e287db754acb9f5d523763be69b9e46}





#### `public def test_instance_norm_device_check(self,`[`gc`](#classinstance__norm__test_1_1_test_instance_norm_1a3604aa25e678a100049e4ddad2857f8f)`,`[`dc`](#classinstance__norm__test_1_1_test_instance_norm_1a533f5b1de13260b71b8db1ee04fa7f92)`,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`store_mean`](#classinstance__norm__test_1_1_test_instance_norm_1a2f6721823e9c1eacfabb2cea030c01ce)`,`[`store_inv_stdev`](#classinstance__norm__test_1_1_test_instance_norm_1a10f46329e09a9d360fc4c2752bc94a4d)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`)` {#classinstance__norm__test_1_1_test_instance_norm_1a2e886aff07cf1805dd4028f47b7a84cd}





#### `public def test_instance_norm_cnn_helper(self,`[`N`](#classinstance__norm__test_1_1_test_instance_norm_1a21fee01e480a3b403a47077ee2c358fd)`,`[`C`](#classinstance__norm__test_1_1_test_instance_norm_1a49567e17e351a7fcfb12794c13eff9c0)`,`[`H`](#classinstance__norm__test_1_1_test_instance_norm_1ad278203b8b77183df42011a2e6f82cf7)`,`[`W`](#classinstance__norm__test_1_1_test_instance_norm_1a683069c0b961b858f9b97fe741c34293)`,`[`order`](#classinstance__norm__test_1_1_test_instance_norm_1a2c13a76032818238f6b8da51365f089a)`,`[`epsilon`](#classinstance__norm__test_1_1_test_instance_norm_1a8c3ba1bcc5399ddf6a341609e3dadd41)`,`[`seed`](#classinstance__norm__test_1_1_test_instance_norm_1aac9c3508f45845ad83864be919968087)`,`[`is_test`](#classinstance__norm__test_1_1_test_instance_norm_1a09420b36a35fed5a87b01fce018373af)`)` {#classinstance__norm__test_1_1_test_instance_norm_1a6edd05d9c8367a07b9cfda50e5d462b3}





# namespace `introspect_vis` {#namespaceintrospect__vis}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`introspect_vis::IntrospectVisData`](#classintrospect__vis_1_1_introspect_vis_data)    |
# class `introspect_vis::IntrospectVisData` {#classintrospect__vis_1_1_introspect_vis_data}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  inputs` |
`public  model_name` |
`public  max_num_instances` |
`public  count` |
`public  instances` |
`public  labels` |
`public  is_multilabel` |
`public  conv_groups` |
`public  neuron_groups` |
`public  selections` |
`public  summaries` |
`public  neuron_summaries` |
`public def __init__(self,`[`inputs`](#classintrospect__vis_1_1_introspect_vis_data_1a3e9924aa09a3016b0a3ea01b6e93f4dd)`,`[`model_name`](#classintrospect__vis_1_1_introspect_vis_data_1ae46d3a377784dd0f0792b5d32907a644)`,first_outputs,meta_info,lab_arr)` |
`public def getInstanceActivations(self,outputs)` |
`public def getInstanceConvActivations(self,outputs)` |
`public def updateNeuronSummaries(self,activations,true_idxs,model_specific)` |
`public def appendInstance(self,instance)` |
`public def processInstance(self,idx,`[`labels`](#classintrospect__vis_1_1_introspect_vis_data_1a3893ea11c213e458b53aad9e82c5d327)`,scores,outputs,model_specific)` |
`public def updateArrangements(self)` |
`public def postprocess(self,filepath)` |

## Members

#### `public  inputs` {#classintrospect__vis_1_1_introspect_vis_data_1a3e9924aa09a3016b0a3ea01b6e93f4dd}





#### `public  model_name` {#classintrospect__vis_1_1_introspect_vis_data_1ae46d3a377784dd0f0792b5d32907a644}





#### `public  max_num_instances` {#classintrospect__vis_1_1_introspect_vis_data_1a9785eabb01c8a23d94eafe2a1c38d3a7}





#### `public  count` {#classintrospect__vis_1_1_introspect_vis_data_1a400930d8d4577d81c2996fa5aefbdb2e}





#### `public  instances` {#classintrospect__vis_1_1_introspect_vis_data_1a781468655a87a42f5ef1f8ae33215248}





#### `public  labels` {#classintrospect__vis_1_1_introspect_vis_data_1a3893ea11c213e458b53aad9e82c5d327}





#### `public  is_multilabel` {#classintrospect__vis_1_1_introspect_vis_data_1a01e1f80ddba360afb81b7d1a874bd5d2}





#### `public  conv_groups` {#classintrospect__vis_1_1_introspect_vis_data_1ae26919c5d2712080ca2199541e0d7603}





#### `public  neuron_groups` {#classintrospect__vis_1_1_introspect_vis_data_1adca1fc915fd9ee112788245c1f2d7d93}





#### `public  selections` {#classintrospect__vis_1_1_introspect_vis_data_1aba8dc80c2e35b3d3e8ee8592a45f56b1}





#### `public  summaries` {#classintrospect__vis_1_1_introspect_vis_data_1af480d9fd38ca6d9ee7ea8847e7525d32}





#### `public  neuron_summaries` {#classintrospect__vis_1_1_introspect_vis_data_1afa6632b861d37699df149722683c139b}





#### `public def __init__(self,`[`inputs`](#classintrospect__vis_1_1_introspect_vis_data_1a3e9924aa09a3016b0a3ea01b6e93f4dd)`,`[`model_name`](#classintrospect__vis_1_1_introspect_vis_data_1ae46d3a377784dd0f0792b5d32907a644)`,first_outputs,meta_info,lab_arr)` {#classintrospect__vis_1_1_introspect_vis_data_1a66aaff807cb9b765a4ae3ed1d3750157}





#### `public def getInstanceActivations(self,outputs)` {#classintrospect__vis_1_1_introspect_vis_data_1ae964e062173c598813d02b6b83002c00}





#### `public def getInstanceConvActivations(self,outputs)` {#classintrospect__vis_1_1_introspect_vis_data_1a6b1783c0f8cd94ee433028f9ff2aede6}





#### `public def updateNeuronSummaries(self,activations,true_idxs,model_specific)` {#classintrospect__vis_1_1_introspect_vis_data_1a2da7eb826e00a6d48cc478b3f77f8eac}





#### `public def appendInstance(self,instance)` {#classintrospect__vis_1_1_introspect_vis_data_1addae9ddd8721774a0a92a0fc71698041}





#### `public def processInstance(self,idx,`[`labels`](#classintrospect__vis_1_1_introspect_vis_data_1a3893ea11c213e458b53aad9e82c5d327)`,scores,outputs,model_specific)` {#classintrospect__vis_1_1_introspect_vis_data_1a9ddabad9f8bfbeef2ac41d44e8de1d4e}





#### `public def updateArrangements(self)` {#classintrospect__vis_1_1_introspect_vis_data_1ad40fee754bc999e05db1f7cac6f37c44}





#### `public def postprocess(self,filepath)` {#classintrospect__vis_1_1_introspect_vis_data_1a8ae902914c8ef2baa9e0efdb08510dc4}





# namespace `layer_model_helper` {#namespacelayer__model__helper}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layer_model_helper::LayerModelHelper`](#classlayer__model__helper_1_1_layer_model_helper)    |
# class `layer_model_helper::LayerModelHelper` {#classlayer__model__helper_1_1_layer_model_helper}

```
class layer_model_helper::LayerModelHelper
  : public model_helper.ModelHelperBase
```  



Model helper for building models on top of layers abstractions.

Each layer is the abstraction that is higher level than Operator. Layer
is responsible for ownership of it's own parameters and can easily be
instantiated in multiple nets possible with different sets of ops.
As an example: one can easily instantiate predict and train nets from
the same set of layers, where predict net will have subset of the
operators from train net.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  param_to_optim` |
`public  param_init_net` |
`public  global_constants` |
`public  global_constant_initializers` |
`public def __init__(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,`[`input_feature_schema`](#classlayer__model__helper_1_1_layer_model_helper_1ac876d888aad5637d4179aaaab08e5ad2)`,`[`trainer_extra_schema`](#classlayer__model__helper_1_1_layer_model_helper_1a02897edcb85550d4a41450d623e098fe)`)` |
`public def add_metric_field(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,value)` |
`public def add_global_constant(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,array,dtype,initializer)` |
`public def create_init_net(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`)` |
`public def next_layer_name(self,prefix)` |
`public def add_layer(self,layer)` |
`public def get_parameter_blobs(self)` |
`public def default_optimizer(self)` |
`public def default_optimizer(self,optimizer)` |
`public def input_feature_schema(self)` |
`public def trainer_extra_schema(self)` |
`public def metrics_schema(self)` |
`public def output_schema(self)` |
`public def output_schema(self,schema)` |
`public def loss(self)` |
`public def loss(self,loss)` |
`public def __getattr__(self,layer)` |
`public def layers(self)` |
`public def SgdOptim(self,base_lr,policy,kwargs)` |
`public def AdagradOptim(self,alpha,epsilon,kwargs)` |
`public def FtrlOptim(self,alpha,beta,lambda1,lambda2,kwargs)` |
`public def Adagrad(self,`[`net`](#classmodel__helper_1_1_model_helper_base_1a9dde9ad0ccd1d0004788b52c08297b2a)`,`[`param_init_net`](#classlayer__model__helper_1_1_layer_model_helper_1a8b6b5d8674815865800a7dff562bd167)`,param,grad,alpha,epsilon,sparse_dedup_aggregator,engine)` |
`public def Ftrl(self,`[`net`](#classmodel__helper_1_1_model_helper_base_1a9dde9ad0ccd1d0004788b52c08297b2a)`,`[`param_init_net`](#classlayer__model__helper_1_1_layer_model_helper_1a8b6b5d8674815865800a7dff562bd167)`,param,grad,alpha,beta,lambda1,lambda2,sparse_dedup_aggregator,engine)` |
`public def Sgd(self,`[`net`](#classmodel__helper_1_1_model_helper_base_1a9dde9ad0ccd1d0004788b52c08297b2a)`,`[`param_init_net`](#classlayer__model__helper_1_1_layer_model_helper_1a8b6b5d8674815865800a7dff562bd167)`,param,grad,base_lr,policy,momentum,kwargs)` |

## Members

#### `public  param_to_optim` {#classlayer__model__helper_1_1_layer_model_helper_1a5d464c8c3e75d5e142e4392b1d528d87}





#### `public  param_init_net` {#classlayer__model__helper_1_1_layer_model_helper_1a8b6b5d8674815865800a7dff562bd167}





#### `public  global_constants` {#classlayer__model__helper_1_1_layer_model_helper_1a9fc7acb8599d4155276ca9f5f6daa6df}





#### `public  global_constant_initializers` {#classlayer__model__helper_1_1_layer_model_helper_1a10e4bd95e6dcc4948d6ee986de880e80}





#### `public def __init__(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,`[`input_feature_schema`](#classlayer__model__helper_1_1_layer_model_helper_1ac876d888aad5637d4179aaaab08e5ad2)`,`[`trainer_extra_schema`](#classlayer__model__helper_1_1_layer_model_helper_1a02897edcb85550d4a41450d623e098fe)`)` {#classlayer__model__helper_1_1_layer_model_helper_1a786d1c306f4e0cad6bb85aa3d471cd41}





#### `public def add_metric_field(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,value)` {#classlayer__model__helper_1_1_layer_model_helper_1a761feecf930824c5e098bc07d3d0a6b1}





#### `public def add_global_constant(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,array,dtype,initializer)` {#classlayer__model__helper_1_1_layer_model_helper_1aa100e7bb6623c536d553b66974b294d7}





#### `public def create_init_net(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`)` {#classlayer__model__helper_1_1_layer_model_helper_1ae37c317d54ca80795dabc5b54bfdedfb}





#### `public def next_layer_name(self,prefix)` {#classlayer__model__helper_1_1_layer_model_helper_1aa36f7571550fc7c7f6d489e714619db3}





#### `public def add_layer(self,layer)` {#classlayer__model__helper_1_1_layer_model_helper_1a3cf0f25fb6466324128cd255c0a5d5bb}





#### `public def get_parameter_blobs(self)` {#classlayer__model__helper_1_1_layer_model_helper_1a620d6051c00ad0323ad5d7482128ff92}





#### `public def default_optimizer(self)` {#classlayer__model__helper_1_1_layer_model_helper_1ab2660204f44ee48708a7bab5fce55577}





#### `public def default_optimizer(self,optimizer)` {#classlayer__model__helper_1_1_layer_model_helper_1a9c1fabc2e2e6b15f9ce42604c8a6831c}





#### `public def input_feature_schema(self)` {#classlayer__model__helper_1_1_layer_model_helper_1ac876d888aad5637d4179aaaab08e5ad2}





#### `public def trainer_extra_schema(self)` {#classlayer__model__helper_1_1_layer_model_helper_1a02897edcb85550d4a41450d623e098fe}





#### `public def metrics_schema(self)` {#classlayer__model__helper_1_1_layer_model_helper_1adb3051e367d0c10992071b3bff003bbf}



Returns the schema that represents model output that should be used for
metric reporting.

During the training/evaluation this schema will be appended to the
schema that represents model output.

#### `public def output_schema(self)` {#classlayer__model__helper_1_1_layer_model_helper_1a82f150bbe6dd0d12425051ec46625b2d}





#### `public def output_schema(self,schema)` {#classlayer__model__helper_1_1_layer_model_helper_1a8bc838863fd82e5ea50cd3c85febf6b2}





#### `public def loss(self)` {#classlayer__model__helper_1_1_layer_model_helper_1addef7e5f0cd00fdd40709c14aaf96d1a}





#### `public def loss(self,loss)` {#classlayer__model__helper_1_1_layer_model_helper_1a20c0d8b78b6a182cade5bc3269f6d54d}





#### `public def __getattr__(self,layer)` {#classlayer__model__helper_1_1_layer_model_helper_1a7b101ef5b68590c5372c5b8f05f6f432}





#### `public def layers(self)` {#classlayer__model__helper_1_1_layer_model_helper_1a847dc4eadaa1538bdef83a87a80da1e9}





#### `public def SgdOptim(self,base_lr,policy,kwargs)` {#classlayer__model__helper_1_1_layer_model_helper_1aded9f5d2d9a99b4894e50a8538f63f1d}





#### `public def AdagradOptim(self,alpha,epsilon,kwargs)` {#classlayer__model__helper_1_1_layer_model_helper_1acdfd4b82794788ce5e03e6dd0b854baf}





#### `public def FtrlOptim(self,alpha,beta,lambda1,lambda2,kwargs)` {#classlayer__model__helper_1_1_layer_model_helper_1ad6a43d8af46ec24f2c9e9428ac8580ba}





#### `public def Adagrad(self,`[`net`](#classmodel__helper_1_1_model_helper_base_1a9dde9ad0ccd1d0004788b52c08297b2a)`,`[`param_init_net`](#classlayer__model__helper_1_1_layer_model_helper_1a8b6b5d8674815865800a7dff562bd167)`,param,grad,alpha,epsilon,sparse_dedup_aggregator,engine)` {#classlayer__model__helper_1_1_layer_model_helper_1ac2a9f2f14d8b5ed8c67b7b250d349ca2}





#### `public def Ftrl(self,`[`net`](#classmodel__helper_1_1_model_helper_base_1a9dde9ad0ccd1d0004788b52c08297b2a)`,`[`param_init_net`](#classlayer__model__helper_1_1_layer_model_helper_1a8b6b5d8674815865800a7dff562bd167)`,param,grad,alpha,beta,lambda1,lambda2,sparse_dedup_aggregator,engine)` {#classlayer__model__helper_1_1_layer_model_helper_1ada1aca8e60b89ef961f32174289c2405}





#### `public def Sgd(self,`[`net`](#classmodel__helper_1_1_model_helper_base_1a9dde9ad0ccd1d0004788b52c08297b2a)`,`[`param_init_net`](#classlayer__model__helper_1_1_layer_model_helper_1a8b6b5d8674815865800a7dff562bd167)`,param,grad,base_lr,policy,momentum,kwargs)` {#classlayer__model__helper_1_1_layer_model_helper_1a590875f0eae1be271d364f0f6cd9dd41}





# namespace `layers::batch_distill_lr_loss` {#namespacelayers_1_1batch__distill__lr__loss}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::batch_distill_lr_loss::BatchDistillLRLoss`](#classlayers_1_1batch__distill__lr__loss_1_1_batch_distill_l_r_loss)    |
# class `layers::batch_distill_lr_loss::BatchDistillLRLoss` {#classlayers_1_1batch__distill__lr__loss_1_1_batch_distill_l_r_loss}

```
class layers::batch_distill_lr_loss::BatchDistillLRLoss
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,teacherWeight,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1batch__distill__lr__loss_1_1_batch_distill_l_r_loss_1a049857c2f31608a0fbd285ce0545a10b}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,teacherWeight,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1batch__distill__lr__loss_1_1_batch_distill_l_r_loss_1abb5d2003450912b93d17ec101f34d688}





#### `public def add_ops(self,net)` {#classlayers_1_1batch__distill__lr__loss_1_1_batch_distill_l_r_loss_1a7add973dbf9e64911170e8ef10b47169}





# namespace `layers::batch_lr_loss` {#namespacelayers_1_1batch__lr__loss}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::batch_lr_loss::BatchLRLoss`](#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss)    |
# class `layers::batch_lr_loss::BatchLRLoss` {#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss}

```
class layers::batch_lr_loss::BatchLRLoss
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  average_loss` |
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`average_loss`](#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss_1a97e0166fd5726a70739d86191f5b0414)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  average_loss` {#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss_1a97e0166fd5726a70739d86191f5b0414}





#### `public  output_schema` {#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss_1a217e1c071e8a154e98224375d371e7f2}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`average_loss`](#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss_1a97e0166fd5726a70739d86191f5b0414)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss_1aab11cc2e8345287b6bde8132f358407f}





#### `public def add_ops(self,net)` {#classlayers_1_1batch__lr__loss_1_1_batch_l_r_loss_1a7da8b03907e55cb0f13407c3f693c26d}





# namespace `layers::batch_mse_loss` {#namespacelayers_1_1batch__mse__loss}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::batch_mse_loss::BatchMSELoss`](#classlayers_1_1batch__mse__loss_1_1_batch_m_s_e_loss)    |
# class `layers::batch_mse_loss::BatchMSELoss` {#classlayers_1_1batch__mse__loss_1_1_batch_m_s_e_loss}

```
class layers::batch_mse_loss::BatchMSELoss
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1batch__mse__loss_1_1_batch_m_s_e_loss_1ac151f5628361e7a65cf1996fb520f38c}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1batch__mse__loss_1_1_batch_m_s_e_loss_1a28b730201985e18cf6d34899fc99d7ab}





#### `public def add_ops(self,net)` {#classlayers_1_1batch__mse__loss_1_1_batch_m_s_e_loss_1a6920cb08467fc5ab339bd1b3184453a9}





# namespace `layers::batch_sigmoid_cross_entropy_loss` {#namespacelayers_1_1batch__sigmoid__cross__entropy__loss}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::batch_sigmoid_cross_entropy_loss::BatchSigmoidCrossEntropyLoss`](#classlayers_1_1batch__sigmoid__cross__entropy__loss_1_1_batch_sigmoid_cross_entropy_loss)    |
# class `layers::batch_sigmoid_cross_entropy_loss::BatchSigmoidCrossEntropyLoss` {#classlayers_1_1batch__sigmoid__cross__entropy__loss_1_1_batch_sigmoid_cross_entropy_loss}

```
class layers::batch_sigmoid_cross_entropy_loss::BatchSigmoidCrossEntropyLoss
  : public ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public def __init__(self,model,input_record,name,kwargs)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1batch__sigmoid__cross__entropy__loss_1_1_batch_sigmoid_cross_entropy_loss_1aa9d79d89cdf3e9a3d3247af589e51979}





#### `public def __init__(self,model,input_record,name,kwargs)` {#classlayers_1_1batch__sigmoid__cross__entropy__loss_1_1_batch_sigmoid_cross_entropy_loss_1a42de704cecda211d56cbc289bcecc73a}





#### `public def add_ops(self,net)` {#classlayers_1_1batch__sigmoid__cross__entropy__loss_1_1_batch_sigmoid_cross_entropy_loss_1af2586f8673276c4fa6744dfe15ccb012}





# namespace `layers::batch_softmax_loss` {#namespacelayers_1_1batch__softmax__loss}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::batch_softmax_loss::BatchSoftmaxLoss`](#classlayers_1_1batch__softmax__loss_1_1_batch_softmax_loss)    |
# class `layers::batch_softmax_loss::BatchSoftmaxLoss` {#classlayers_1_1batch__softmax__loss_1_1_batch_softmax_loss}

```
class layers::batch_softmax_loss::BatchSoftmaxLoss
  : public ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public def __init__(self,model,input_record,name,kwargs)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1batch__softmax__loss_1_1_batch_softmax_loss_1a6fd839f56ffd4c69077c8e3f9bbb9c0e}





#### `public def __init__(self,model,input_record,name,kwargs)` {#classlayers_1_1batch__softmax__loss_1_1_batch_softmax_loss_1a17c1c1a5863aa0d31ea624547802e717}





#### `public def add_ops(self,net)` {#classlayers_1_1batch__softmax__loss_1_1_batch_softmax_loss_1ac01c95770daae77c5317e59b86ca6ad2}





# namespace `layers::concat` {#namespacelayers_1_1concat}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::concat::Concat`](#classlayers_1_1concat_1_1_concat)    |
# class `layers::concat::Concat` {#classlayers_1_1concat_1_1_concat}

```
class layers::concat::Concat
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  axis` |
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`axis`](#classlayers_1_1concat_1_1_concat_1aec3be3b4411c1dd44a701cf3daff2112)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  axis` {#classlayers_1_1concat_1_1_concat_1aec3be3b4411c1dd44a701cf3daff2112}





#### `public  output_schema` {#classlayers_1_1concat_1_1_concat_1ae016c7b09da7d5b6a9175ed0ae997ebf}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`axis`](#classlayers_1_1concat_1_1_concat_1aec3be3b4411c1dd44a701cf3daff2112)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1concat_1_1_concat_1ab2c1a89b84e0ccc4aefbc1ac8c132d28}





#### `public def add_ops(self,net)` {#classlayers_1_1concat_1_1_concat_1a754e915dda0b1e98e7f1eda704d7bc45}





# namespace `layers::dot_product` {#namespacelayers_1_1dot__product}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::dot_product::DotProduct`](#classlayers_1_1dot__product_1_1_dot_product)    |
# class `layers::dot_product::DotProduct` {#classlayers_1_1dot__product_1_1_dot_product}

```
class layers::dot_product::DotProduct
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1dot__product_1_1_dot_product_1aae1f6957fd80a04c3339b5fc51dd7613}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1dot__product_1_1_dot_product_1a73537c9959575635869d184fff44f6f2}





#### `public def add_ops(self,net)` {#classlayers_1_1dot__product_1_1_dot_product_1aae58db4b78e3b0c40a2d96d66377d615}





# namespace `layers::expand_dims` {#namespacelayers_1_1expand__dims}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::expand_dims::ExpandDims`](#classlayers_1_1expand__dims_1_1_expand_dims)    |
# class `layers::expand_dims::ExpandDims` {#classlayers_1_1expand__dims_1_1_expand_dims}

```
class layers::expand_dims::ExpandDims
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  dims` |
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`dims`](#classlayers_1_1expand__dims_1_1_expand_dims_1ad2414d5e45b7a1f4fd6f3bb255bcaed2)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  dims` {#classlayers_1_1expand__dims_1_1_expand_dims_1ad2414d5e45b7a1f4fd6f3bb255bcaed2}





#### `public  output_schema` {#classlayers_1_1expand__dims_1_1_expand_dims_1a4356ec5ab5fcb1558e2780a7fd719609}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`dims`](#classlayers_1_1expand__dims_1_1_expand_dims_1ad2414d5e45b7a1f4fd6f3bb255bcaed2)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1expand__dims_1_1_expand_dims_1a3840ebc4542e48461a227de41cf23ccb}





#### `public def add_ops(self,net)` {#classlayers_1_1expand__dims_1_1_expand_dims_1ab63cc9dfee78aea8a1bc244c4eb06835}





# namespace `layers::fc` {#namespacelayers_1_1fc}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::fc::FC`](#classlayers_1_1fc_1_1_f_c)    |
# class `layers::fc::FC` {#classlayers_1_1fc_1_1_f_c}

```
class layers::fc::FC
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public  w` |
`public  b` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,output_dims,weight_init,bias_init,weight_optim,bias_optim,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1fc_1_1_f_c_1adc040d7078c5e2f50ff13f8cd61c8ffa}





#### `public  w` {#classlayers_1_1fc_1_1_f_c_1aa20f057d448ce5e43e23a7f8be27ff1e}





#### `public  b` {#classlayers_1_1fc_1_1_f_c_1a7de9174b86e93a9d25ab8b038bfdc982}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,output_dims,weight_init,bias_init,weight_optim,bias_optim,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1fc_1_1_f_c_1a61362f597d896937065753a3406c290a}





#### `public def add_ops(self,net)` {#classlayers_1_1fc_1_1_f_c_1aef131f3054f4aa3722407c527bac98b9}





# namespace `layers::fc_without_bias` {#namespacelayers_1_1fc__without__bias}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::fc_without_bias::FCWithoutBias`](#classlayers_1_1fc__without__bias_1_1_f_c_without_bias)    |
# class `layers::fc_without_bias::FCWithoutBias` {#classlayers_1_1fc__without__bias_1_1_f_c_without_bias}

```
class layers::fc_without_bias::FCWithoutBias
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public  w` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,output_dims,weight_init,weight_optim,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1fc__without__bias_1_1_f_c_without_bias_1a25cbba060655064e4103167666556b0b}





#### `public  w` {#classlayers_1_1fc__without__bias_1_1_f_c_without_bias_1a59a1a29bab0dfb8d183f0fade24b2151}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,output_dims,weight_init,weight_optim,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1fc__without__bias_1_1_f_c_without_bias_1a0aafae6d6019225518f43a97067d533a}





#### `public def add_ops(self,net)` {#classlayers_1_1fc__without__bias_1_1_f_c_without_bias_1a15cc9194a5467885a985e101bb0dae61}





# namespace `layers::functional` {#namespacelayers_1_1functional}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::functional::Functional`](#classlayers_1_1functional_1_1_functional)    |
# class `layers::functional::Functional` {#classlayers_1_1functional_1_1_functional}

```
class layers::functional::Functional
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,num_outputs,function,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  output_schema` {#classlayers_1_1functional_1_1_functional_1ab705db02cd34321ba51ea57520107e83}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,num_outputs,function,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1functional_1_1_functional_1a0377980b435a7a95241aedc4244c47f4}





#### `public def add_ops(self,net)` {#classlayers_1_1functional_1_1_functional_1a92f11a9076208d072e58bb5e5d76bc55}





# namespace `layers::layers` {#namespacelayers_1_1layers}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::layers::InstantiationContext`](#classlayers_1_1layers_1_1_instantiation_context)    |
`class `[`layers::layers::ModelLayer`](#classlayers_1_1layers_1_1_model_layer)    |
# class `layers::layers::InstantiationContext` {#classlayers_1_1layers_1_1_instantiation_context}

```
class layers::layers::InstantiationContext
  : public object
```  



List of contexts where layer could be instantitated

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `layers::layers::ModelLayer` {#classlayers_1_1layers_1_1_model_layer}

```
class layers::layers::ModelLayer
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  name` |
`public  model` |
`public  kwargs` |
`public  input_record` |
`public  request_only` |
`public  output_schema` |
`public  tags` |
`public  params` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,prefix,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`tags`](#classlayers_1_1layers_1_1_model_layer_1a7f29e4d433fd07509fbfc877447b0bde)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def get_type(self)` |
`public def get_output_schema(self)` |
`public def get_parameters(self)` |
`public def get_fp16_compatible_parameters(self)` |
`public def get_memory_usage(self)` |
`public def add_operators(self,net,init_net,context)` |
`public def add_ops(self,net)` |
`public def add_train_ops(self,net)` |

## Members

#### `public  name` {#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb}





#### `public  model` {#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e}





#### `public  kwargs` {#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8}





#### `public  input_record` {#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3}





#### `public  request_only` {#classlayers_1_1layers_1_1_model_layer_1aee9d665511df11182681d8892fa3e483}





#### `public  output_schema` {#classlayers_1_1layers_1_1_model_layer_1ada9f75368d0bab0d77b80f28687c727b}





#### `public  tags` {#classlayers_1_1layers_1_1_model_layer_1a7f29e4d433fd07509fbfc877447b0bde}





#### `public  params` {#classlayers_1_1layers_1_1_model_layer_1aa67c0a9daddfbaa02546d98fafd9f6de}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,prefix,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`tags`](#classlayers_1_1layers_1_1_model_layer_1a7f29e4d433fd07509fbfc877447b0bde)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1layers_1_1_model_layer_1a216ba07713bb1ab601ab099626dc5458}





#### `public def get_type(self)` {#classlayers_1_1layers_1_1_model_layer_1a6ab4a7db02535bfb9a9b4f000215ffc9}





#### `public def get_output_schema(self)` {#classlayers_1_1layers_1_1_model_layer_1a5be5644fbeaac5b03d2cc9ca21342128}





#### `public def get_parameters(self)` {#classlayers_1_1layers_1_1_model_layer_1ac6f55aeb14b1c7b740aac76a796e334c}





#### `public def get_fp16_compatible_parameters(self)` {#classlayers_1_1layers_1_1_model_layer_1adb385e96825c76f6efdf044bead5de5a}



Return a subset of parameters which can be converted to fp16

#### `public def get_memory_usage(self)` {#classlayers_1_1layers_1_1_model_layer_1abd7f3dd9befc869397bff27ad630eb34}





#### `public def add_operators(self,net,init_net,context)` {#classlayers_1_1layers_1_1_model_layer_1a3eea0a6e0679ec18e984f0488cc43df9}





#### `public def add_ops(self,net)` {#classlayers_1_1layers_1_1_model_layer_1ac4489c98a8cd62239cb6b823409635af}





#### `public def add_train_ops(self,net)` {#classlayers_1_1layers_1_1_model_layer_1a53d71e91afdce7eaedec427981f08bd9}





# namespace `layers::sparse_lookup` {#namespacelayers_1_1sparse__lookup}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::sparse_lookup::SparseLookup`](#classlayers_1_1sparse__lookup_1_1_sparse_lookup)    |
# class `layers::sparse_lookup::SparseLookup` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup}

```
class layers::sparse_lookup::SparseLookup
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  reducer` |
`public  output_schema` |
`public  shape` |
`public  weight_init` |
`public  w` |
`public  pos_w` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,inner_shape,`[`reducer`](#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1adb1b83fdaba1028ec22f78ac7f5b1de4)`,`[`weight_init`](#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a26b8d1b22349636e7bfade4745c03879)`,weight_optim,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def get_memory_usage(self)` |
`public def get_fp16_compatible_parameters(self)` |
`public def add_ops(self,net)` |

## Members

#### `public  reducer` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1adb1b83fdaba1028ec22f78ac7f5b1de4}





#### `public  output_schema` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a79fccee9c5be716ce70876c95e7eb76e}





#### `public  shape` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a85b05350b222cebe676170b72fa091b4}





#### `public  weight_init` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a26b8d1b22349636e7bfade4745c03879}





#### `public  w` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a28a5a7be9f834292609b37f51ff9c082}





#### `public  pos_w` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a4bfa797e6b8b2d4164484635ca25ebe5}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,inner_shape,`[`reducer`](#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1adb1b83fdaba1028ec22f78ac7f5b1de4)`,`[`weight_init`](#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a26b8d1b22349636e7bfade4745c03879)`,weight_optim,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a4f48adde4fdf01dc59c7a03b97974714}





#### `public def get_memory_usage(self)` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1ac1d71a9c048d1ddfe63f958a44bf9c94}





#### `public def get_fp16_compatible_parameters(self)` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a2b87af0566e19f2eb7daff8e42d51ec5}





#### `public def add_ops(self,net)` {#classlayers_1_1sparse__lookup_1_1_sparse_lookup_1a8f9d1c0d456c2033baa43f35e7689647}





# namespace `layers::sparse_to_dense` {#namespacelayers_1_1sparse__to__dense}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::sparse_to_dense::SparseToDense`](#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense)    |
# class `layers::sparse_to_dense::SparseToDense` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense}

```
class layers::sparse_to_dense::SparseToDense
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  input_specs` |
`public  output_schema` |
`public  zero` |
`public  zero_range` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`input_specs`](#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a8f846dd222d67716170444931820104d)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |
`public def get_metadata(self)` |

## Members

#### `public  input_specs` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a8f846dd222d67716170444931820104d}





#### `public  output_schema` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a3d8e932cc5c77af4dc2fe67b270817c5}





#### `public  zero` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a2c29de4a56783896d59ba82f6dc85434}





#### `public  zero_range` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a7dc119791267bcd8a4a79b5ed5b20e0d}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,`[`input_specs`](#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a8f846dd222d67716170444931820104d)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a56afd0ef20c4c4aa84daf49d259b08c8}



`input_specs` follows the format of FeatureSpec from schema. To be more
precise it's a namedtuple that should have:
    'feature_type', 'feature_names', 'feature_ids'

#### `public def add_ops(self,net)` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1ad19fa48da3b5617344edb06f4a4674da}





#### `public def get_metadata(self)` {#classlayers_1_1sparse__to__dense_1_1_sparse_to_dense_1a43c146b39200a00513cd4b96897db6b1}





# namespace `layers::split` {#namespacelayers_1_1split}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::split::Split`](#classlayers_1_1split_1_1_split)    |
# class `layers::split::Split` {#classlayers_1_1split_1_1_split}

```
class layers::split::Split
  : public layers.layers.ModelLayer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  axis` |
`public  output_schema` |
`public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,num_splits,`[`axis`](#classlayers_1_1split_1_1_split_1a66e011a2174096d088121c7113d24d3c)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` |
`public def add_ops(self,net)` |

## Members

#### `public  axis` {#classlayers_1_1split_1_1_split_1a66e011a2174096d088121c7113d24d3c}





#### `public  output_schema` {#classlayers_1_1split_1_1_split_1a0b51aaf56a3ed1e72f7875c62123f1a0}





#### `public def __init__(self,`[`model`](#classlayers_1_1layers_1_1_model_layer_1ab2781c1f055d9170c9fc39070428c65e)`,`[`input_record`](#classlayers_1_1layers_1_1_model_layer_1ae27261a71b2abe9afd6911f125da01e3)`,num_splits,`[`axis`](#classlayers_1_1split_1_1_split_1a66e011a2174096d088121c7113d24d3c)`,`[`name`](#classlayers_1_1layers_1_1_model_layer_1a8cdb3e22ef4fa6d50b9ce768e9b2bebb)`,`[`kwargs`](#classlayers_1_1layers_1_1_model_layer_1a676ef4fa1cce9c05173a74b79711d2f8)`)` {#classlayers_1_1split_1_1_split_1a192992c1ef20248ab13f25ff83b92162}





#### `public def add_ops(self,net)` {#classlayers_1_1split_1_1_split_1a7eec0d427adb54d659810678c31cea32}





# namespace `layers::tags` {#namespacelayers_1_1tags}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers::tags::TagContext`](#classlayers_1_1tags_1_1_tag_context)    |
`class `[`layers::tags::Tags`](#classlayers_1_1tags_1_1_tags)    |
# class `layers::tags::TagContext` {#classlayers_1_1tags_1_1_tag_context}

```
class layers::tags::TagContext
  : public object
```  



Scope driven way to provide tags to the layers.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  tags` |
`public def __init__(self,`[`tags`](#classlayers_1_1tags_1_1_tag_context_1ae6c0ed4413d61ae8ef9600e849c21c46)`)` |
`public def add_tags(self,`[`tags`](#classlayers_1_1tags_1_1_tag_context_1ae6c0ed4413d61ae8ef9600e849c21c46)`)` |
`public def remove_tags(self,`[`tags`](#classlayers_1_1tags_1_1_tag_context_1ae6c0ed4413d61ae8ef9600e849c21c46)`)` |

## Members

#### `public  tags` {#classlayers_1_1tags_1_1_tag_context_1ae6c0ed4413d61ae8ef9600e849c21c46}





#### `public def __init__(self,`[`tags`](#classlayers_1_1tags_1_1_tag_context_1ae6c0ed4413d61ae8ef9600e849c21c46)`)` {#classlayers_1_1tags_1_1_tag_context_1ae12d46e822dfc99b215b548c7a97814f}





#### `public def add_tags(self,`[`tags`](#classlayers_1_1tags_1_1_tag_context_1ae6c0ed4413d61ae8ef9600e849c21c46)`)` {#classlayers_1_1tags_1_1_tag_context_1ad897dc5ed9567a2c8106292121802442}





#### `public def remove_tags(self,`[`tags`](#classlayers_1_1tags_1_1_tag_context_1ae6c0ed4413d61ae8ef9600e849c21c46)`)` {#classlayers_1_1tags_1_1_tag_context_1acabba6ec5298fc97f218ae46aec1bd1c}





# class `layers::tags::Tags` {#classlayers_1_1tags_1_1_tags}

```
class layers::tags::Tags
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  tags` |
`public def __init__(self,`[`tags`](#classlayers_1_1tags_1_1_tags_1ad4f3062ce0265696997f4b7351f93dea)`)` |
`public def __enter__(self)` |
`public def __exit__(self,type,value,traceback)` |

## Members

#### `public  tags` {#classlayers_1_1tags_1_1_tags_1ad4f3062ce0265696997f4b7351f93dea}





#### `public def __init__(self,`[`tags`](#classlayers_1_1tags_1_1_tags_1ad4f3062ce0265696997f4b7351f93dea)`)` {#classlayers_1_1tags_1_1_tags_1a4aea2ee7f116c803d186af21a50ddb1d}





#### `public def __enter__(self)` {#classlayers_1_1tags_1_1_tags_1ad551f7248dd180496717e76e411975e9}





#### `public def __exit__(self,type,value,traceback)` {#classlayers_1_1tags_1_1_tags_1a8291218c3eaec93158acd24043d851b2}





# namespace `layers_test` {#namespacelayers__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`layers_test::TestLayers`](#classlayers__test_1_1_test_layers)    |
# class `layers_test::TestLayers` {#classlayers__test_1_1_test_layers}

```
class layers_test::TestLayers
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  model` |
`public def setUp(self)` |
`public def new_record(self,schema_obj)` |
`public def get_training_nets(self)` |
`public def get_predict_net(self)` |
`public def assertBlobsEqual(self,spec_blobs,op_blobs)` |
`public def assertNetContainOps(self,net,op_specs)` |
`public def testFCWithoutBias(self)` |
`public def testBatchSigmoidCrossEntropyLoss(self)` |
`public def testBatchSoftmaxLoss(self)` |
`public def testFunctionalLayer(self)` |
`public def testFunctionalLayerHelper(self)` |
`public def testFunctionalLayerHelperAutoInference(self)` |
`public def testFunctionalLayerHelperAutoInferenceScalar(self)` |

## Members

#### `public  model` {#classlayers__test_1_1_test_layers_1a22c11f3ad6ec5a029d4c670af9017aaf}





#### `public def setUp(self)` {#classlayers__test_1_1_test_layers_1ac3116a459832a6dfea4852ac32274f1b}





#### `public def new_record(self,schema_obj)` {#classlayers__test_1_1_test_layers_1a5865d7fb138121523c5e75b5ced0438e}





#### `public def get_training_nets(self)` {#classlayers__test_1_1_test_layers_1ac5c01d29fc01619ee6017238ffeb33cf}



We don't use
layer_model_instantiator.generate_training_nets_forward_only()
here because it includes initialization of global constants, which make
testing tricky

#### `public def get_predict_net(self)` {#classlayers__test_1_1_test_layers_1aef53e7e3e227631f4be7c0c781ed820a}





#### `public def assertBlobsEqual(self,spec_blobs,op_blobs)` {#classlayers__test_1_1_test_layers_1a2ce9347880f020ab7d8db6e230f0b458}



spec_blobs can either be None or a list of blob names. If it's None,
then no assertion is performed. The elements of the list can be None,
in that case, it means that position will not be checked.

#### `public def assertNetContainOps(self,net,op_specs)` {#classlayers__test_1_1_test_layers_1a343d0bbcd6ebfcc8eb934d0742a03067}



Given a net and a list of OpSpec's, check that the net match the spec

#### `public def testFCWithoutBias(self)` {#classlayers__test_1_1_test_layers_1a517473613e776c047107231a00e673c3}





#### `public def testBatchSigmoidCrossEntropyLoss(self)` {#classlayers__test_1_1_test_layers_1a9c054f16427e958bdff27e1a11469d8a}





#### `public def testBatchSoftmaxLoss(self)` {#classlayers__test_1_1_test_layers_1ab114da7ee9f4118fe4ea2d8be6790ae1}





#### `public def testFunctionalLayer(self)` {#classlayers__test_1_1_test_layers_1a76eb7a839b37414cafc7e20534362f37}





#### `public def testFunctionalLayerHelper(self)` {#classlayers__test_1_1_test_layers_1ad92dc6ababc4f40309e0dd944d7ecb75}





#### `public def testFunctionalLayerHelperAutoInference(self)` {#classlayers__test_1_1_test_layers_1a40f4311e2416019c84dae0255d01f98b}





#### `public def testFunctionalLayerHelperAutoInferenceScalar(self)` {#classlayers__test_1_1_test_layers_1a6f578c675c0f3fa159255b04becc6bce}





# namespace `load_save_test` {#namespaceload__save__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`load_save_test::TestLoadSave`](#classload__save__test_1_1_test_load_save)    |
`class `[`load_save_test::TestLoadSaveBase`](#classload__save__test_1_1_test_load_save_base)    |
# class `load_save_test::TestLoadSave` {#classload__save__test_1_1_test_load_save}

```
class load_save_test::TestLoadSave
  : public load_save_test.TestLoadSaveBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testLoadSave(self)` |
`public def testRepeatedArgs(self)` |
`public def testLoadExcessblobs(self)` |
`public def testTruncatedFile(self)` |
`public def testBlobNameOverrides(self)` |
`public def testMissingFile(self)` |

## Members

#### `public def testLoadSave(self)` {#classload__save__test_1_1_test_load_save_1abbe88107df502d6a6ae0535b0bc6c199}





#### `public def testRepeatedArgs(self)` {#classload__save__test_1_1_test_load_save_1a91c3cbdb25e4965f8acec3341da1aa19}





#### `public def testLoadExcessblobs(self)` {#classload__save__test_1_1_test_load_save_1a948781e7d017af34a2a58cd31cc26dd3}





#### `public def testTruncatedFile(self)` {#classload__save__test_1_1_test_load_save_1a27607414eeae592752d2c0a715e2a1fe}





#### `public def testBlobNameOverrides(self)` {#classload__save__test_1_1_test_load_save_1a744112c9aca8fb6889acbd8e20027077}





#### `public def testMissingFile(self)` {#classload__save__test_1_1_test_load_save_1a4d36cb073a707d79ce0c45f27dda5aad}





# class `load_save_test::TestLoadSaveBase` {#classload__save__test_1_1_test_load_save_base}

```
class load_save_test::TestLoadSaveBase
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,methodName,db_type)` |
`public def load_save(self,`[`src_device_type`](#classload__save__test_1_1_test_load_save_base_1ad0b5eea57075eb4df848e23a91527233)`,`[`src_gpu_id`](#classload__save__test_1_1_test_load_save_base_1a69f28406980cbdae79839feb9b329591)`,`[`dst_device_type`](#classload__save__test_1_1_test_load_save_base_1ad4e442c3c4c9a15a73a2d5294e32e01b)`,`[`dst_gpu_id`](#classload__save__test_1_1_test_load_save_base_1a7451d12493bc6761cf8fd7669b3f6f81)`)` |
`public def saveFile(self,tmp_folder,db_type)` |

## Members

#### `public def __init__(self,methodName,db_type)` {#classload__save__test_1_1_test_load_save_base_1af333e1a24899be48856f18d965094428}





#### `public def load_save(self,`[`src_device_type`](#classload__save__test_1_1_test_load_save_base_1ad0b5eea57075eb4df848e23a91527233)`,`[`src_gpu_id`](#classload__save__test_1_1_test_load_save_base_1a69f28406980cbdae79839feb9b329591)`,`[`dst_device_type`](#classload__save__test_1_1_test_load_save_base_1ad4e442c3c4c9a15a73a2d5294e32e01b)`,`[`dst_gpu_id`](#classload__save__test_1_1_test_load_save_base_1a7451d12493bc6761cf8fd7669b3f6f81)`)` {#classload__save__test_1_1_test_load_save_base_1abf0c915fb688283ee866ba7db1b3edc9}





#### `public def saveFile(self,tmp_folder,db_type)` {#classload__save__test_1_1_test_load_save_base_1a8e8852d46381b926596f33ff3ec5d67b}





# namespace `margin_ranking_criterion_op_test` {#namespacemargin__ranking__criterion__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`margin_ranking_criterion_op_test::TestMarginRankingCriterion`](#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion)    |
# class `margin_ranking_criterion_op_test::TestMarginRankingCriterion` {#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion}

```
class margin_ranking_criterion_op_test::TestMarginRankingCriterion
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_margin_ranking_criterion(self,`[`N`](#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion_1acfa8c01e5750969f69b51bae137b96e0)`,`[`seed`](#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion_1a133dae4830fa99da9f76f70fa75fbc4b)`,`[`margin`](#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion_1af23b0981ad9e72496cd878eaf3692b38)`,gc,dc)` |

## Members

#### `public def test_margin_ranking_criterion(self,`[`N`](#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion_1acfa8c01e5750969f69b51bae137b96e0)`,`[`seed`](#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion_1a133dae4830fa99da9f76f70fa75fbc4b)`,`[`margin`](#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion_1af23b0981ad9e72496cd878eaf3692b38)`,gc,dc)` {#classmargin__ranking__criterion__op__test_1_1_test_margin_ranking_criterion_1af5677288cb5c5876a273dac94f726b94}





# namespace `matmul_op_test` {#namespacematmul__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`matmul_op_test::TestBatchMatMul`](#classmatmul__op__test_1_1_test_batch_mat_mul)    |
`class `[`matmul_op_test::TestMatMul`](#classmatmul__op__test_1_1_test_mat_mul)    |
# class `matmul_op_test::TestBatchMatMul` {#classmatmul__op__test_1_1_test_batch_mat_mul}

```
class matmul_op_test::TestBatchMatMul
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_matmul(self,`[`C`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a5c40036d415aa45b53cd591a08bec889)`,`[`M`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a00145a39a98a1ad4183a7d32a5b07ff2)`,`[`K`](#classmatmul__op__test_1_1_test_batch_mat_mul_1abde1e85d6d2affd28842cf6d210a594e)`,`[`N`](#classmatmul__op__test_1_1_test_batch_mat_mul_1ad532ba941a37f2269b120efbbedad782)`,`[`trans_a`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a5e7a484546aeee0f9df19bed6b79efa0)`,`[`trans_b`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a6cc0120e5abe1c94c1895cbbc756f5ba)`,gc,dc)` |

## Members

#### `public def test_matmul(self,`[`C`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a5c40036d415aa45b53cd591a08bec889)`,`[`M`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a00145a39a98a1ad4183a7d32a5b07ff2)`,`[`K`](#classmatmul__op__test_1_1_test_batch_mat_mul_1abde1e85d6d2affd28842cf6d210a594e)`,`[`N`](#classmatmul__op__test_1_1_test_batch_mat_mul_1ad532ba941a37f2269b120efbbedad782)`,`[`trans_a`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a5e7a484546aeee0f9df19bed6b79efa0)`,`[`trans_b`](#classmatmul__op__test_1_1_test_batch_mat_mul_1a6cc0120e5abe1c94c1895cbbc756f5ba)`,gc,dc)` {#classmatmul__op__test_1_1_test_batch_mat_mul_1a5dd01e925a05633f9a9f3cb4367218d4}





# class `matmul_op_test::TestMatMul` {#classmatmul__op__test_1_1_test_mat_mul}

```
class matmul_op_test::TestMatMul
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_matmul(self,`[`M`](#classmatmul__op__test_1_1_test_mat_mul_1a27bb1a3e9d68aff35464f8df522f42f0)`,`[`K`](#classmatmul__op__test_1_1_test_mat_mul_1ac85c7a1c2c490b3d43a65d4235b6697d)`,`[`N`](#classmatmul__op__test_1_1_test_mat_mul_1a7253ce33e075cf76e1b17a945d175aab)`,`[`trans_a`](#classmatmul__op__test_1_1_test_mat_mul_1a719e9a125356f4ad281c7f1571328b12)`,`[`trans_b`](#classmatmul__op__test_1_1_test_mat_mul_1af8259da159cd7071b253efc1fb4e16fb)`,gc,dc)` |

## Members

#### `public def test_matmul(self,`[`M`](#classmatmul__op__test_1_1_test_mat_mul_1a27bb1a3e9d68aff35464f8df522f42f0)`,`[`K`](#classmatmul__op__test_1_1_test_mat_mul_1ac85c7a1c2c490b3d43a65d4235b6697d)`,`[`N`](#classmatmul__op__test_1_1_test_mat_mul_1a7253ce33e075cf76e1b17a945d175aab)`,`[`trans_a`](#classmatmul__op__test_1_1_test_mat_mul_1a719e9a125356f4ad281c7f1571328b12)`,`[`trans_b`](#classmatmul__op__test_1_1_test_mat_mul_1af8259da159cd7071b253efc1fb4e16fb)`,gc,dc)` {#classmatmul__op__test_1_1_test_mat_mul_1a9249fb36aad3f40db9e97d98eb1e12db}





# namespace `memonger_test` {#namespacememonger__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`memonger_test::MemongerTest`](#classmemonger__test_1_1_memonger_test)    |
# class `memonger_test::MemongerTest` {#classmemonger__test_1_1_memonger_test}

```
class memonger_test::MemongerTest
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_simple_memonger(self,`[`input_dim`](#classmemonger__test_1_1_memonger_test_1af3b924ec4bc7c502bd2be3890f91a617)`,`[`output_dim`](#classmemonger__test_1_1_memonger_test_1a593dc93a6ce2d45a7023428683007f12)`,`[`batch_size`](#classmemonger__test_1_1_memonger_test_1ae074dedcaa0758711369e7c52a53c08b)`,`[`do`](#classmemonger__test_1_1_memonger_test_1aff842e4fc7427799309a27df32ee77df)`)` |
`public def test_gradient_optim(self,`[`input_dim`](#classmemonger__test_1_1_memonger_test_1af3b924ec4bc7c502bd2be3890f91a617)`,`[`output_dim`](#classmemonger__test_1_1_memonger_test_1a593dc93a6ce2d45a7023428683007f12)`,`[`batch_size`](#classmemonger__test_1_1_memonger_test_1ae074dedcaa0758711369e7c52a53c08b)`)` |
`public def test_gradient_optim_tree(self,`[`input_dim`](#classmemonger__test_1_1_memonger_test_1af3b924ec4bc7c502bd2be3890f91a617)`,`[`output_dim`](#classmemonger__test_1_1_memonger_test_1a593dc93a6ce2d45a7023428683007f12)`,`[`batch_size`](#classmemonger__test_1_1_memonger_test_1ae074dedcaa0758711369e7c52a53c08b)`)` |

## Members

#### `public def test_simple_memonger(self,`[`input_dim`](#classmemonger__test_1_1_memonger_test_1af3b924ec4bc7c502bd2be3890f91a617)`,`[`output_dim`](#classmemonger__test_1_1_memonger_test_1a593dc93a6ce2d45a7023428683007f12)`,`[`batch_size`](#classmemonger__test_1_1_memonger_test_1ae074dedcaa0758711369e7c52a53c08b)`,`[`do`](#classmemonger__test_1_1_memonger_test_1aff842e4fc7427799309a27df32ee77df)`)` {#classmemonger__test_1_1_memonger_test_1a76901a24c9e01689c75ba3c4ea3d59af}





#### `public def test_gradient_optim(self,`[`input_dim`](#classmemonger__test_1_1_memonger_test_1af3b924ec4bc7c502bd2be3890f91a617)`,`[`output_dim`](#classmemonger__test_1_1_memonger_test_1a593dc93a6ce2d45a7023428683007f12)`,`[`batch_size`](#classmemonger__test_1_1_memonger_test_1ae074dedcaa0758711369e7c52a53c08b)`)` {#classmemonger__test_1_1_memonger_test_1aece5c13b1486f461b62b81e074973d4f}





#### `public def test_gradient_optim_tree(self,`[`input_dim`](#classmemonger__test_1_1_memonger_test_1af3b924ec4bc7c502bd2be3890f91a617)`,`[`output_dim`](#classmemonger__test_1_1_memonger_test_1a593dc93a6ce2d45a7023428683007f12)`,`[`batch_size`](#classmemonger__test_1_1_memonger_test_1ae074dedcaa0758711369e7c52a53c08b)`)` {#classmemonger__test_1_1_memonger_test_1a92b3eefe8ce2929941dc35bfb3cd0efa}





# namespace `mkl_conv_op_test` {#namespacemkl__conv__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`mkl_conv_op_test::MKLConvTest`](#classmkl__conv__op__test_1_1_m_k_l_conv_test)    |
# class `mkl_conv_op_test::MKLConvTest` {#classmkl__conv__op__test_1_1_m_k_l_conv_test}

```
class mkl_conv_op_test::MKLConvTest
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_mkl_convolution(self,`[`stride`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1a011f648b2b86e643cdc97e83a3d40cf3)`,`[`pad`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1a10e787a02963a36a1d20f9c72225e481)`,`[`kernel`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1af7d90ee3fbcaa925f6d15b227ab1f0bd)`,`[`size`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1ae9ab96fc3fbaba156a04889cfa2dea76)`,`[`input_channels`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1a3005ca69d2789c2556f98d5dcb5e4d18)`,`[`output_channels`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1adeab7ce509c839f029d69666f72fdd28)`,`[`batch_size`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1ad8bade8b2da7d18bc20a9cf1044c97a8)`,gc,dc)` |

## Members

#### `public def test_mkl_convolution(self,`[`stride`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1a011f648b2b86e643cdc97e83a3d40cf3)`,`[`pad`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1a10e787a02963a36a1d20f9c72225e481)`,`[`kernel`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1af7d90ee3fbcaa925f6d15b227ab1f0bd)`,`[`size`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1ae9ab96fc3fbaba156a04889cfa2dea76)`,`[`input_channels`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1a3005ca69d2789c2556f98d5dcb5e4d18)`,`[`output_channels`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1adeab7ce509c839f029d69666f72fdd28)`,`[`batch_size`](#classmkl__conv__op__test_1_1_m_k_l_conv_test_1ad8bade8b2da7d18bc20a9cf1044c97a8)`,gc,dc)` {#classmkl__conv__op__test_1_1_m_k_l_conv_test_1a7caf2f4ecc3d4f405d924a382b0985fd}





# namespace `mkl_packed_fc_op_test` {#namespacemkl__packed__fc__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`mkl_packed_fc_op_test::PackedFCTest`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test)    |
# class `mkl_packed_fc_op_test::PackedFCTest` {#classmkl__packed__fc__op__test_1_1_packed_f_c_test}

```
class mkl_packed_fc_op_test::PackedFCTest
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_packed_fc(self,`[`seed`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a3e48d1c434994fa9f09e3fccf36f4971)`,`[`M`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1ac1453ff2ec223357b6971ee58586e679)`,`[`K`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a91a185c6ea8f4581e6870282126d53b1)`,`[`N`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1aca42fc54686219b8f70436e681d83dff)`,gc,dc)` |
`public def test_packed_fc_axis(self,`[`axis`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a4486931393d4af77fb93c05eb75bd2dd)`,`[`num_output`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a5b011673984b3d2a9bff33acec2a04a7)`,gc,dc)` |

## Members

#### `public def test_packed_fc(self,`[`seed`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a3e48d1c434994fa9f09e3fccf36f4971)`,`[`M`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1ac1453ff2ec223357b6971ee58586e679)`,`[`K`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a91a185c6ea8f4581e6870282126d53b1)`,`[`N`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1aca42fc54686219b8f70436e681d83dff)`,gc,dc)` {#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1afc410b8435b3ddb713c9c90749bb7447}





#### `public def test_packed_fc_axis(self,`[`axis`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a4486931393d4af77fb93c05eb75bd2dd)`,`[`num_output`](#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a5b011673984b3d2a9bff33acec2a04a7)`,gc,dc)` {#classmkl__packed__fc__op__test_1_1_packed_f_c_test_1a54b786de46960f0766dcbe5f105b2056}





# namespace `mkl_speed_test` {#namespacemkl__speed__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`mkl_speed_test::TestMKLBasic`](#classmkl__speed__test_1_1_test_m_k_l_basic)    |
# class `mkl_speed_test::TestMKLBasic` {#classmkl__speed__test_1_1_test_m_k_l_basic}

```
class mkl_speed_test::TestMKLBasic
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testReLUSpeed(self)` |
`public def testConvSpeed(self)` |

## Members

#### `public def testReLUSpeed(self)` {#classmkl__speed__test_1_1_test_m_k_l_basic_1ad1e65a1111846c790e4de98a5e0e6f1e}





#### `public def testConvSpeed(self)` {#classmkl__speed__test_1_1_test_m_k_l_basic_1ad4fe42bbef4767f09277d28fe4fffdd6}





# namespace `model_device_test` {#namespacemodel__device__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`model_device_test::TestMiniAlexNet`](#classmodel__device__test_1_1_test_mini_alex_net)    |
# class `model_device_test::TestMiniAlexNet` {#classmodel__device__test_1_1_test_mini_alex_net}

```
class model_device_test::TestMiniAlexNet
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testMiniAlexNetNCHW(self)` |

## Members

#### `public def testMiniAlexNetNCHW(self)` {#classmodel__device__test_1_1_test_mini_alex_net_1a0715b7ffe614c4250fed7251342fa3fd}





# namespace `model_helper` {#namespacemodel__helper}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`model_helper::ModelHelperBase`](#classmodel__helper_1_1_model_helper_base)    |
`class `[`model_helper::ParameterInfo`](#classmodel__helper_1_1_parameter_info)    |
`class `[`model_helper::ParameterType`](#classmodel__helper_1_1_parameter_type)    |
# class `model_helper::ModelHelperBase` {#classmodel__helper_1_1_model_helper_base}

```
class model_helper::ModelHelperBase
  : public object
```  



A helper model so we can write models more easily, without having to
manually define parameter initializations and operators separately.
In order to add support for specific operators, inherit from this class
and add corresponding methods. Operator representing methods should
take care of adding their parameters to params

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  name` |
`public  net` |
`public  param_init_net` |
`public  param_to_grad` |
`public  params` |
`public  computed_params` |
`public  gradient_ops_added` |
`public  init_params` |
`public  allow_not_known_ops` |
`public  skip_sparse_optim` |
`public  grad_map` |
`public def __init__(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,`[`init_params`](#classmodel__helper_1_1_model_helper_base_1a19231ec7dc5927d2cf1d88ac8ceffb3c)`,`[`allow_not_known_ops`](#classmodel__helper_1_1_model_helper_base_1ac07253a0b346d16557d233a79ce7f55f)`,`[`skip_sparse_optim`](#classmodel__helper_1_1_model_helper_base_1aba037083b828218d912e5093334db9c1)`,param_model)` |
`public def add_param(self,param,key,shape,length)` |
`public def param_info(self,grad_type,id)` |
`public def GetParams(self,namescope,top_scope)` |
`public def Proto(self)` |
`public def InitProto(self)` |
`public def RunAllOnGPU(self,args,kwargs)` |
`public def CreateDB(self,blob_out,db,db_type,kwargs)` |
`public def AddGradientOperators(self,args,kwargs)` |
`public def get_param_to_grad(self,`[`params`](#classmodel__helper_1_1_model_helper_base_1a299a7186e14508a789778b759c0b69fb)`)` |
`public def GetOptimizationPairs(self,`[`params`](#classmodel__helper_1_1_model_helper_base_1a299a7186e14508a789778b759c0b69fb)`)` |
`public def GetComputedParams(self,namescope)` |
`public def GetAllParams(self,namescope)` |
`public def TensorProtosDBInput(self,unused_blob_in,blob_out,batch_size,db,db_type,kwargs)` |
`public def AddOperator(self,op_type,inputs,parameters,args,kwargs)` |
`public def GetDevices(self)` |
`public def __getattr__(self,op_type)` |

## Members

#### `public  name` {#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f}





#### `public  net` {#classmodel__helper_1_1_model_helper_base_1a9dde9ad0ccd1d0004788b52c08297b2a}





#### `public  param_init_net` {#classmodel__helper_1_1_model_helper_base_1ae5ca586b69fd0430322271f4a7d4a220}





#### `public  param_to_grad` {#classmodel__helper_1_1_model_helper_base_1a6a6d6c760037741dc72dd8a37fdd44ba}





#### `public  params` {#classmodel__helper_1_1_model_helper_base_1a299a7186e14508a789778b759c0b69fb}





#### `public  computed_params` {#classmodel__helper_1_1_model_helper_base_1a389be70b851b15abffe7f024723e613d}





#### `public  gradient_ops_added` {#classmodel__helper_1_1_model_helper_base_1a48c15d20838858079daa61797fc3521d}





#### `public  init_params` {#classmodel__helper_1_1_model_helper_base_1a19231ec7dc5927d2cf1d88ac8ceffb3c}





#### `public  allow_not_known_ops` {#classmodel__helper_1_1_model_helper_base_1ac07253a0b346d16557d233a79ce7f55f}





#### `public  skip_sparse_optim` {#classmodel__helper_1_1_model_helper_base_1aba037083b828218d912e5093334db9c1}





#### `public  grad_map` {#classmodel__helper_1_1_model_helper_base_1a9727196b7ff08abc7ee4fccaf195c5e0}





#### `public def __init__(self,`[`name`](#classmodel__helper_1_1_model_helper_base_1aa6e79b7aa97180ad75bd36aa4a35d04f)`,`[`init_params`](#classmodel__helper_1_1_model_helper_base_1a19231ec7dc5927d2cf1d88ac8ceffb3c)`,`[`allow_not_known_ops`](#classmodel__helper_1_1_model_helper_base_1ac07253a0b346d16557d233a79ce7f55f)`,`[`skip_sparse_optim`](#classmodel__helper_1_1_model_helper_base_1aba037083b828218d912e5093334db9c1)`,param_model)` {#classmodel__helper_1_1_model_helper_base_1ad521c84315759391e77fa7a8fedb336c}





#### `public def add_param(self,param,key,shape,length)` {#classmodel__helper_1_1_model_helper_base_1aa0d09a54d8ce88791e919afa1ecd05e0}





#### `public def param_info(self,grad_type,id)` {#classmodel__helper_1_1_model_helper_base_1a99546ab421432b7ee2d927a6274557a4}





#### `public def GetParams(self,namescope,top_scope)` {#classmodel__helper_1_1_model_helper_base_1a6f35f9a99be28a8e6e8c17c11576e9a0}



Returns the params in current namescope

#### `public def Proto(self)` {#classmodel__helper_1_1_model_helper_base_1a050e5bab27e601a0f4f724893e3e0b57}





#### `public def InitProto(self)` {#classmodel__helper_1_1_model_helper_base_1a4ace44908b4ae4fa3a46ff2543e3cd22}





#### `public def RunAllOnGPU(self,args,kwargs)` {#classmodel__helper_1_1_model_helper_base_1a2cdc3a54e91b90ef96fd00aa7aaddfd2}





#### `public def CreateDB(self,blob_out,db,db_type,kwargs)` {#classmodel__helper_1_1_model_helper_base_1a61f252549563017530ebb35ee7e93557}





#### `public def AddGradientOperators(self,args,kwargs)` {#classmodel__helper_1_1_model_helper_base_1aef1a5acd56c37b1f5e69817a089ebd49}





#### `public def get_param_to_grad(self,`[`params`](#classmodel__helper_1_1_model_helper_base_1a299a7186e14508a789778b759c0b69fb)`)` {#classmodel__helper_1_1_model_helper_base_1af5db74a6453273130b36145f7647a73f}



Given a list of parameters returns a dict from a parameter
to a corresponding gradient

#### `public def GetOptimizationPairs(self,`[`params`](#classmodel__helper_1_1_model_helper_base_1a299a7186e14508a789778b759c0b69fb)`)` {#classmodel__helper_1_1_model_helper_base_1a18eeecdb52a2b8e9d109298de2e57f9f}



Returns a map for param => grad.
If params is not specified, all parameters will be considered.

#### `public def GetComputedParams(self,namescope)` {#classmodel__helper_1_1_model_helper_base_1ac99c89c0e752eaaafa9f407659ec5ffa}



Returns the computed params in current namescope. 'Computed params'
are such parameters that are not optimized via gradient descent but are
directly computed from data, such as the running mean and variance
of Spatial Batch Normalization.

#### `public def GetAllParams(self,namescope)` {#classmodel__helper_1_1_model_helper_base_1ab498776bed23f98dcb30831582f7dfaf}





#### `public def TensorProtosDBInput(self,unused_blob_in,blob_out,batch_size,db,db_type,kwargs)` {#classmodel__helper_1_1_model_helper_base_1ae09d760d09e02954387533e85768e445}



TensorProtosDBInput.

#### `public def AddOperator(self,op_type,inputs,parameters,args,kwargs)` {#classmodel__helper_1_1_model_helper_base_1a2cd406d0725daafd0a263b41670711fe}



Adds an operator to a model. Use parameters list
to specify which operator inputs are model parameters to be
optimized.

Example of usage:

model.SparseLengthsSum(
     [embedding, indices, lengths],
     parameters=[embedding],
)

Here embedding is a parameter to be optimized while indices
and lengths are not.

#### `public def GetDevices(self)` {#classmodel__helper_1_1_model_helper_base_1ab326350bfbaebb821ba1c40292a43ab0}





#### `public def __getattr__(self,op_type)` {#classmodel__helper_1_1_model_helper_base_1aba08619eb1a0ff1398c11caebdeffa47}



Catch-all for all other operators, mostly those without params.

# class `model_helper::ParameterInfo` {#classmodel__helper_1_1_parameter_info}

```
class model_helper::ParameterInfo
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  param_id` |
`public  name` |
`public  blob` |
`public  key` |
`public  shape` |
`public  size` |
`public  length` |
`public  grad` |
`public def __init__(self,`[`param_id`](#classmodel__helper_1_1_parameter_info_1a56224e527066c587abbc8263993bdd34)`,param,`[`key`](#classmodel__helper_1_1_parameter_info_1a3470042ed54702de64abed35da82ed65)`,`[`shape`](#classmodel__helper_1_1_parameter_info_1ab90f465d9eb0cf7be3c35c3a5b5c5684)`,`[`length`](#classmodel__helper_1_1_parameter_info_1a41516f889a52e32f3255c262659bb03f)`)` |
`public def grad_type(self)` |
`public def cloned_init_net(self)` |
`public def __str__(self)` |

## Members

#### `public  param_id` {#classmodel__helper_1_1_parameter_info_1a56224e527066c587abbc8263993bdd34}





#### `public  name` {#classmodel__helper_1_1_parameter_info_1a1a2b114e019e06f99aa187ac5e744fb4}





#### `public  blob` {#classmodel__helper_1_1_parameter_info_1ab0069ec1e89b50188c83594d6802c089}





#### `public  key` {#classmodel__helper_1_1_parameter_info_1a3470042ed54702de64abed35da82ed65}





#### `public  shape` {#classmodel__helper_1_1_parameter_info_1ab90f465d9eb0cf7be3c35c3a5b5c5684}





#### `public  size` {#classmodel__helper_1_1_parameter_info_1a0ed3e8b31ecbebe8ba6739fea03a1199}





#### `public  length` {#classmodel__helper_1_1_parameter_info_1a41516f889a52e32f3255c262659bb03f}





#### `public  grad` {#classmodel__helper_1_1_parameter_info_1a1a1be290761fc92ad7b324b9b3d7627b}





#### `public def __init__(self,`[`param_id`](#classmodel__helper_1_1_parameter_info_1a56224e527066c587abbc8263993bdd34)`,param,`[`key`](#classmodel__helper_1_1_parameter_info_1a3470042ed54702de64abed35da82ed65)`,`[`shape`](#classmodel__helper_1_1_parameter_info_1ab90f465d9eb0cf7be3c35c3a5b5c5684)`,`[`length`](#classmodel__helper_1_1_parameter_info_1a41516f889a52e32f3255c262659bb03f)`)` {#classmodel__helper_1_1_parameter_info_1add816cf2890dd429f537bae1b316e07e}





#### `public def grad_type(self)` {#classmodel__helper_1_1_parameter_info_1aef22a3833f3933006dfe929087a9a12a}





#### `public def cloned_init_net(self)` {#classmodel__helper_1_1_parameter_info_1a2a1da58e3910017eb79a8d1670c18742}





#### `public def __str__(self)` {#classmodel__helper_1_1_parameter_info_1a27db8ef62ff4dbcde17bee824085b199}





# class `model_helper::ParameterType` {#classmodel__helper_1_1_parameter_type}

```
class model_helper::ParameterType
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# namespace `momentum_sgd_test` {#namespacemomentum__sgd__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`momentum_sgd_test::TestMomentumSGD`](#classmomentum__sgd__test_1_1_test_momentum_s_g_d)    |
# class `momentum_sgd_test::TestMomentumSGD` {#classmomentum__sgd__test_1_1_test_momentum_s_g_d}

```
class momentum_sgd_test::TestMomentumSGD
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_momentum_sgd(self,`[`n`](#classmomentum__sgd__test_1_1_test_momentum_s_g_d_1a871c27d14a2f322f98d1e58d384250dc)`,gc,dc)` |

## Members

#### `public def test_momentum_sgd(self,`[`n`](#classmomentum__sgd__test_1_1_test_momentum_s_g_d_1a871c27d14a2f322f98d1e58d384250dc)`,gc,dc)` {#classmomentum__sgd__test_1_1_test_momentum_s_g_d_1a3d4642d4fb570d5665f3c0e71cbbce8d}





# namespace `mpi_test` {#namespacempi__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`mpi_test::TestMPI`](#classmpi__test_1_1_test_m_p_i)    |
# class `mpi_test::TestMPI` {#classmpi__test_1_1_test_m_p_i}

```
class mpi_test::TestMPI
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_broadcast(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`root`](#classmpi__test_1_1_test_m_p_i_1a88703412ed37785dc15249b3cfdbec64)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,gc,dc)` |
`public def test_reduce(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`root`](#classmpi__test_1_1_test_m_p_i_1a88703412ed37785dc15249b3cfdbec64)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,gc,dc)` |
`public def test_allreduce(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`root`](#classmpi__test_1_1_test_m_p_i_1a88703412ed37785dc15249b3cfdbec64)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,`[`inplace`](#classmpi__test_1_1_test_m_p_i_1a69f93fdc151e08c9dcc0bebd6c0890b4)`,gc,dc)` |
`public def test_sendrecv(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,`[`specify_send_blob`](#classmpi__test_1_1_test_m_p_i_1aa57e93c7592b6812b79767eca8b727e5)`,`[`specify_recv_blob`](#classmpi__test_1_1_test_m_p_i_1a522ddbcfab6e7a07754910b5d4a7040f)`,gc,dc)` |

## Members

#### `public def test_broadcast(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`root`](#classmpi__test_1_1_test_m_p_i_1a88703412ed37785dc15249b3cfdbec64)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,gc,dc)` {#classmpi__test_1_1_test_m_p_i_1a1a8d78d14a161cfaaae0822564a066f3}





#### `public def test_reduce(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`root`](#classmpi__test_1_1_test_m_p_i_1a88703412ed37785dc15249b3cfdbec64)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,gc,dc)` {#classmpi__test_1_1_test_m_p_i_1ad7447d5e4cbfbd37b4898cca26769e86}





#### `public def test_allreduce(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`root`](#classmpi__test_1_1_test_m_p_i_1a88703412ed37785dc15249b3cfdbec64)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,`[`inplace`](#classmpi__test_1_1_test_m_p_i_1a69f93fdc151e08c9dcc0bebd6c0890b4)`,gc,dc)` {#classmpi__test_1_1_test_m_p_i_1a9d02aaa387c0a3d7d830129812abfc3e}





#### `public def test_sendrecv(self,`[`X`](#classmpi__test_1_1_test_m_p_i_1a59d999ee06b9e57c018501c16c662174)`,`[`device_option`](#classmpi__test_1_1_test_m_p_i_1ac1d5d05448bc1f82fd5a42a70eb0843f)`,`[`specify_send_blob`](#classmpi__test_1_1_test_m_p_i_1aa57e93c7592b6812b79767eca8b727e5)`,`[`specify_recv_blob`](#classmpi__test_1_1_test_m_p_i_1a522ddbcfab6e7a07754910b5d4a7040f)`,gc,dc)` {#classmpi__test_1_1_test_m_p_i_1a9beaf7eb245bfb524ff4382032649a74}





# namespace `muji_test` {#namespacemuji__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`muji_test::TestMuji`](#classmuji__test_1_1_test_muji)    |
# class `muji_test::TestMuji` {#classmuji__test_1_1_test_muji}

```
class muji_test::TestMuji
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def RunningAllreduceWithGPUs(self,gpu_ids,allreduce_function)` |
`public def testAllreduceFallback(self)` |
`public def testAllreduceSingleGPU(self)` |
`public def testAllreduceWithTwoGPUs(self)` |
`public def testAllreduceWithFourGPUs(self)` |
`public def testAllreduceWithEightGPUs(self)` |

## Members

#### `public def RunningAllreduceWithGPUs(self,gpu_ids,allreduce_function)` {#classmuji__test_1_1_test_muji_1aae0ded1927bc456784e54ee715701871}



A base function to test different scenarios.

#### `public def testAllreduceFallback(self)` {#classmuji__test_1_1_test_muji_1afc8f2e0e43a856b39329b23f2835d364}





#### `public def testAllreduceSingleGPU(self)` {#classmuji__test_1_1_test_muji_1a2bf3dbfd7e02a15057683cc1664d5515}





#### `public def testAllreduceWithTwoGPUs(self)` {#classmuji__test_1_1_test_muji_1aa97457c4d3aef0c04ca9607b9235decb}





#### `public def testAllreduceWithFourGPUs(self)` {#classmuji__test_1_1_test_muji_1aa739552855952307d227dd690e4c4119}





#### `public def testAllreduceWithEightGPUs(self)` {#classmuji__test_1_1_test_muji_1ac88a0caa8e9f6550f6869b5e0533441e}





# namespace `net_builder` {#namespacenet__builder}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`net_builder::_Loop`](#classnet__builder_1_1___loop)    |
`class `[`net_builder::_ReporterBuilder`](#classnet__builder_1_1___reporter_builder)    |
`class `[`net_builder::_RunIf`](#classnet__builder_1_1___run_if)    |
`class `[`net_builder::_RunOnce`](#classnet__builder_1_1___run_once)    |
`class `[`net_builder::_SetupBuilder`](#classnet__builder_1_1___setup_builder)    |
`class `[`net_builder::_StopGuard`](#classnet__builder_1_1___stop_guard)    |
`class `[`net_builder::NetBuilder`](#classnet__builder_1_1_net_builder)    |
`class `[`net_builder::Operations`](#classnet__builder_1_1_operations)    |
# class `net_builder::_Loop` {#classnet__builder_1_1___loop}

```
class net_builder::_Loop
  : public net_builder.NetBuilder
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,iters,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |
`public def iter(self)` |
`public def __enter__(self)` |
`public def __exit__(self,type,args)` |

## Members

#### `public def __init__(self,iters,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1___loop_1ae01b00be3d43a124e13c4cad1b7f2e80}





#### `public def iter(self)` {#classnet__builder_1_1___loop_1ad3e3cf2cb6e5a680d2695646a4a32b58}





#### `public def __enter__(self)` {#classnet__builder_1_1___loop_1a1fb15009c8974699a63e5e2ad65b77a9}





#### `public def __exit__(self,type,args)` {#classnet__builder_1_1___loop_1a4af845cfa6698ac479384968499b73cc}





# class `net_builder::_ReporterBuilder` {#classnet__builder_1_1___reporter_builder}

```
class net_builder::_ReporterBuilder
  : public net_builder.NetBuilder
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  interval_ms` |
`public def __init__(self,`[`interval_ms`](#classnet__builder_1_1___reporter_builder_1a628dfeb1a63651a12d67f40824474a05)`,net,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |
`public def __exit__(self,etype,args)` |

## Members

#### `public  interval_ms` {#classnet__builder_1_1___reporter_builder_1a628dfeb1a63651a12d67f40824474a05}





#### `public def __init__(self,`[`interval_ms`](#classnet__builder_1_1___reporter_builder_1a628dfeb1a63651a12d67f40824474a05)`,net,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1___reporter_builder_1adff3900f6084fcb54658bf1eb347f416}





#### `public def __exit__(self,etype,args)` {#classnet__builder_1_1___reporter_builder_1ac70a9cd447667f1ad3fbdfce1f18798f}





# class `net_builder::_RunIf` {#classnet__builder_1_1___run_if}

```
class net_builder::_RunIf
  : public net_builder._RunOnce
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,cond_blob,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`,_already_ran)` |
`public def __enter__(self)` |
`public def Elif(self,cond,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |
`public def Else(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |

## Members

#### `public def __init__(self,cond_blob,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`,_already_ran)` {#classnet__builder_1_1___run_if_1a0b211f3ccfe66a660cc95ba26f25d2c0}





#### `public def __enter__(self)` {#classnet__builder_1_1___run_if_1aca66d19341486cf2bad00d948c5820d0}





#### `public def Elif(self,cond,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1___run_if_1a381ad717b0930f692e7ce635673efd55}





#### `public def Else(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1___run_if_1af832e00c9880086df0d8b52e74321129}





# class `net_builder::_RunOnce` {#classnet__builder_1_1___run_once}

```
class net_builder::_RunOnce
  : public net_builder.NetBuilder
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |
`public def __exit__(self,etype,args)` |

## Members

#### `public def __init__(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1___run_once_1a42d0ac7e6086f64044752cedefaa7a6a}





#### `public def __exit__(self,etype,args)` {#classnet__builder_1_1___run_once_1a7a87f3f5ccd5e06225fc082a50482a25}





# class `net_builder::_SetupBuilder` {#classnet__builder_1_1___setup_builder}

```
class net_builder::_SetupBuilder
  : public net_builder.NetBuilder
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  type` |
`public def __init__(self,`[`type`](#classnet__builder_1_1___setup_builder_1a3025a853dca0334bad8de44b7e9239c4)`,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |
`public def setup(self,net)` |
`public def exit(self,net)` |

## Members

#### `public  type` {#classnet__builder_1_1___setup_builder_1a3025a853dca0334bad8de44b7e9239c4}





#### `public def __init__(self,`[`type`](#classnet__builder_1_1___setup_builder_1a3025a853dca0334bad8de44b7e9239c4)`,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1___setup_builder_1a0e0cde6e94560504c997b34c2aa49bcb}





#### `public def setup(self,net)` {#classnet__builder_1_1___setup_builder_1abaf153473e32ad31827a710387f1ee19}





#### `public def exit(self,net)` {#classnet__builder_1_1___setup_builder_1a93e9c36e0f3e88175334e501daf4cf9b}





# class `net_builder::_StopGuard` {#classnet__builder_1_1___stop_guard}

```
class net_builder::_StopGuard
  : public net_builder._RunOnce
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,has_stopped_blob,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |
`public def __enter__(self)` |
`public def __exit__(self,etype,args)` |
`public def has_stopped(self)` |

## Members

#### `public def __init__(self,has_stopped_blob,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1___stop_guard_1ab77267d78a92271fb8a3839e99c58657}





#### `public def __enter__(self)` {#classnet__builder_1_1___stop_guard_1a06da7473dc0e0b9e94022bfa8d4e91d5}





#### `public def __exit__(self,etype,args)` {#classnet__builder_1_1___stop_guard_1a2fd4d5ef4c038a909ec6b2558f20de22}





#### `public def has_stopped(self)` {#classnet__builder_1_1___stop_guard_1ad9fce9f48b7185924bbf781a98047e46}



Return a blob that will be set to scalar bool `True` after
this net builder ran, iff it was halted early.

# class `net_builder::NetBuilder` {#classnet__builder_1_1_net_builder}

```
class net_builder::NetBuilder
  : public object
```  



Scope-driven mechanism for building nets, loops and conditional blocks.
Example:
    from caffe2.python.net_builder import NetBuilder, ops
    with NetBuilder() as nb:
        c = ops.Const(5)
        d = ops.Const(0)
        with ops.loop():
            ops.stop_if(ops.LE([c, ops.Const(0)]))
            ops.Add([c, ops.Const(-1)], [c])
            with ops.If(ops.GE([c, ops.Const(3)])):
                ops.Add([d, ops.Const(10)])
        ops.Print(c, [])
        ops.Print(d, [])
    step = core.to_execution_step(nb)

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  name` |
`public def __init__(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`,_stop_blob_required,_stop_blob,_fullname)` |
`public def stop_blob(self)` |
`public def stop_if(self,blob)` |
`public def add(self,child)` |
`public def current_net(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` |
`public def freeze(self)` |
`public def get(self)` |
`public def __exit__(self,etype,args)` |
`public def __str__(self)` |

## Members

#### `public  name` {#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4}





#### `public def __init__(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`,_stop_blob_required,_stop_blob,_fullname)` {#classnet__builder_1_1_net_builder_1a1347b6036c62ab0f521d11ea448cdd1e}





#### `public def stop_blob(self)` {#classnet__builder_1_1_net_builder_1ab7704cd4edebac1d4710aa69b762ee95}



Returns the BlobReference to the stop_blob of this NetBuilder.
If one is not yet available, creates one.
This function assumes that the stop_blob() will be used immediatelly
in the current net, so it doesn't initialize it if the current net is
the first of the builder.

#### `public def stop_if(self,blob)` {#classnet__builder_1_1_net_builder_1acfe9f21dee6bf9a33ea74b009c8cc1c1}





#### `public def add(self,child)` {#classnet__builder_1_1_net_builder_1ad36819e49043685682d70b17177c104e}





#### `public def current_net(self,`[`name`](#classnet__builder_1_1_net_builder_1a8be90ad5c9f00b1fffaa75f560c23ff4)`)` {#classnet__builder_1_1_net_builder_1ab9761f7bb62ab0e84fbad55829152682}





#### `public def freeze(self)` {#classnet__builder_1_1_net_builder_1a31f53873902df02c8cebc4b55cf72237}





#### `public def get(self)` {#classnet__builder_1_1_net_builder_1a85f4e6b2d2c82d861de7c2ef6974cab8}





#### `public def __exit__(self,etype,args)` {#classnet__builder_1_1_net_builder_1af8a9da8cbf10e82086386fa1377a0973}





#### `public def __str__(self)` {#classnet__builder_1_1_net_builder_1ad02b051cd5eecc690c4a19d8b4a4e604}





# class `net_builder::Operations` {#classnet__builder_1_1_operations}

```
class net_builder::Operations
  : public object
```  



Operations to be used in the context of a NetBuilder.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def net(self,net,name)` |
`public def __getattr__(self,op_type)` |
`public def task_group(self)` |
`public def stop(self)` |
`public def stop_if(self,blob)` |
`public def loop(self,iters,name)` |
`public def stop_guard(self,has_stopped_blob,name)` |
`public def If(self,cond,name)` |
`public def task_init(self)` |
`public def task_exit(self)` |
`public def local_init(self)` |
`public def local_exit(self)` |
`public def task_reporter(self,interval_ms,name)` |
`public def local_reporter(self,interval_ms,name)` |

## Members

#### `public def net(self,net,name)` {#classnet__builder_1_1_operations_1aeb83d8e9dacdbc01f3525031693f7b07}



Retrieves the current net, or add a new net to the builder.
Args:
    net:   If provided, add the given net to the active builder.
   Else, returns the current Net or creates a new one as needed.
    name:  if provided, creates a new Net with given name and makes
   it the new current net of the active builder. Cannot
   be provided if net is provided.

#### `public def __getattr__(self,op_type)` {#classnet__builder_1_1_operations_1a2345e54c342138aa2d34a7ab248f72a8}



Adds an operator call to the currently active Net.

#### `public def task_group(self)` {#classnet__builder_1_1_operations_1a12658024ac3a1d6fe46f81f13951d1ea}



Creates a local task group which will execute as the next step of
the current NetBuilder.

#### `public def stop(self)` {#classnet__builder_1_1_operations_1aeb6b5fe8f8225de32f9c7afa489b053a}



Stop execution of the current execution step.
    Example:
ops.Print(a, 0)
ops.stop()
ops.Print(b, 0)
    In the example, 'b' will never be printed.

#### `public def stop_if(self,blob)` {#classnet__builder_1_1_operations_1ab957fe0ae34c478ec4767b3a5372120a}



Stop execution of the current execution step if the
condition `blob` is met.
    Example:
ops.Print(a, 0)
ops.stop_if(ops.LE([x, ops.Const(0)]))
ops.Print(b, 0)
    In the example, 'b' will only be printed if the value of scalar
    tensor 'x' lower or equal to 0.

#### `public def loop(self,iters,name)` {#classnet__builder_1_1_operations_1a64fcb886af8938ee121d975824eb96be}



Creates a NetBuilder that will execute in a loop as the next step of
the current NetBuilder. If `iters` is provided, the loop will execute
for `iters` iterations and then stop. `iters` can be a constant or a
BlobReference. If `iters` is not provided, the loop will execute
until `ops.stop` or `ops.stop_if` is called.
    Examples:
a = ops.Const(5)
with ops.loop():
    ops.stop_if(ops.LE([a, ops.Const(0)]))
    ops.Print(a, 0)
    ops.Add([a, ops.Const(-1)], [a])
    Above, 'a' will be printed 5 times, with values 5 to 1.

with ops.loop(10) as loop:
    ops.LogInfo(loop.iter())
    This will print the numbers from 0 to 9.

x = ops.Add([ops.Const(10), ops.Const(10)])
with ops.loop(x) as loop:
    ops.LogInfo(loop.iter())
    This will print the numbers from 0 to 19.

#### `public def stop_guard(self,has_stopped_blob,name)` {#classnet__builder_1_1_operations_1a0b635dff17823fa667a4997daf373727}



Creates a NetBuilder that will execute once as the next step of the
current NetBuilder. After execution, a bool tensor will indicate
whether the inner execution was halted with `stop` or `stop_if`.
    Example:
a = ops.Const(True)
with ops.stop_guard() as sg1:
    ops.stop_if(a)
    ops.Print(ops.Const('did not stop'))
b = ops.Const(False)
with ops.stop_guard() as sg2:
    ops.stop_if(b)
    ops.Print(ops.Const('did not stop'))
ops.Print(sg1.has_stopped(), [])
ops.Print(sg2.has_stopped(), [])
    In the example, 'did not stop' will be printed once,
    followed by True and False.

#### `public def If(self,cond,name)` {#classnet__builder_1_1_operations_1a4ddfe5f8132071c35b353f1e7d34114e}



Creates a NetBuilder that will execute once as the next step of the
current NetBuilder if the blob `cond` is True.
    Example:
with ops.If(ops.Const(True)):
    ops.Print(ops.Const('Will print'))
with ops.If(ops.Const(False)):
    ops.Print(ops.Const('Wont print'))
    The example will print 'Will print' once.

#### `public def task_init(self)` {#classnet__builder_1_1_operations_1a4f3857838bf15433ec9f84e289dae6e9}



Defines operations that will be executed once at task startup.
Useful when implementing processors, that don't have access to the Task
top-level structure.
    Example:
def my_processor(rec):
    with ops.task_init():
        one = ops.Const(1)
        two = ops.Const(1)
    return Tuple(
        ops.Add(rec[0](), zero), ops.Add(rec[1](), two))

#### `public def task_exit(self)` {#classnet__builder_1_1_operations_1a3485bf75e086aa9ce769d6230ce3e4dc}



Define operations to be executed at task shutdown.
Useful when implementing processors, that don't have access to the Task
top-level structure.
    Example:
def read_queue(queue):
    with ops.task_exit():
        queue.close(ops.net())
    return queue.read(ops.net())

#### `public def local_init(self)` {#classnet__builder_1_1_operations_1a72885594a27f436a15784616a4418659}



Similar to `task_init`, but executes at TaskGroup's startup instead,
before any task of the group starts executing.

#### `public def local_exit(self)` {#classnet__builder_1_1_operations_1a8cc72f4adf01f185c0d8f5a232eaf789}



Similar to `task_init`, but executes at TaskGroup's exit instead,
after all tasks of the group finished execution.

#### `public def task_reporter(self,interval_ms,name)` {#classnet__builder_1_1_operations_1adf00a36baf1b52b547e066c5b0927b93}



Define operations to be executed at every time interval from
task start-up to finish. These operations are guaranteed to
execute at least once after all other operations of the task are
finished.

    Example:
with ops.task_reporter(interval_ms=10000):
    ops.LogInfo('10s elapsed')

#### `public def local_reporter(self,interval_ms,name)` {#classnet__builder_1_1_operations_1a6f8c3b4574af3cfa41783b0b196251bc}



Similar to task_report, but operations defined within this block
will run repeatedly for as long as any of the tasks in the current
TaskGroup have not finished.

# namespace `net_builder_test` {#namespacenet__builder__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`net_builder_test::TestNetBuilder`](#classnet__builder__test_1_1_test_net_builder)    |
# class `net_builder_test::TestNetBuilder` {#classnet__builder__test_1_1_test_net_builder}

```
class net_builder_test::TestNetBuilder
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_ops(self)` |
`public def test_loops(self)` |
`public def test_setup(self)` |

## Members

#### `public def test_ops(self)` {#classnet__builder__test_1_1_test_net_builder_1a2c3d31f7875a0b561567d2fc66944a5e}





#### `public def test_loops(self)` {#classnet__builder__test_1_1_test_net_builder_1af1b95ced7fe5a3a2b7522ae1b133a0a1}





#### `public def test_setup(self)` {#classnet__builder__test_1_1_test_net_builder_1a902bcbc2d8c060480a5e72f2843304eb}





# namespace `net_printer` {#namespacenet__printer}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`net_printer::Analyzer`](#classnet__printer_1_1_analyzer)    |
`class `[`net_printer::Printer`](#classnet__printer_1_1_printer)    |
`class `[`net_printer::Text`](#classnet__printer_1_1_text)    |
`class `[`net_printer::Visitor`](#classnet__printer_1_1_visitor)    |
# class `net_printer::Analyzer` {#classnet__printer_1_1_analyzer}

```
class net_printer::Analyzer
  : public net_printer.Visitor
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  workspaces` |
`public  workspace_ctx` |
`public def __init__(self)` |
`public def workspace(self)` |
`public def set_workspace(self,node,ws,do_copy)` |
`public def define_blob(self,blob)` |
`public def need_blob(self,blob)` |

## Members

#### `public  workspaces` {#classnet__printer_1_1_analyzer_1a3f39beffd88237628c370ddacde709dc}





#### `public  workspace_ctx` {#classnet__printer_1_1_analyzer_1a492ca012053b176f69bdd28d6737ffb4}





#### `public def __init__(self)` {#classnet__printer_1_1_analyzer_1ab54fb2716a7c2f027429c94e5f6d9f89}





#### `public def workspace(self)` {#classnet__printer_1_1_analyzer_1a1a4047d35a6f9013cc4ff9e7cc7728de}





#### `public def set_workspace(self,node,ws,do_copy)` {#classnet__printer_1_1_analyzer_1acc87f42ae3df94b3b6ffb55ab7e2b10d}





#### `public def define_blob(self,blob)` {#classnet__printer_1_1_analyzer_1a034e66f5417733641cb7eaa77bf6cf56}





#### `public def need_blob(self,blob)` {#classnet__printer_1_1_analyzer_1a620aa119fffbf33eb77019d1bfc96082}





# class `net_printer::Printer` {#classnet__printer_1_1_printer}

```
class net_printer::Printer
  : public net_printer.Visitor
  : public net_printer.Text
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  factor_prefixes` |
`public def __init__(self,`[`factor_prefixes`](#classnet__printer_1_1_printer_1a75fa0cc84d286d0f6474e75b86a46cdb)`)` |

## Members

#### `public  factor_prefixes` {#classnet__printer_1_1_printer_1a75fa0cc84d286d0f6474e75b86a46cdb}





#### `public def __init__(self,`[`factor_prefixes`](#classnet__printer_1_1_printer_1a75fa0cc84d286d0f6474e75b86a46cdb)`)` {#classnet__printer_1_1_printer_1aeb769b856f3bbc767129e771fe9db4f9}





# class `net_printer::Text` {#classnet__printer_1_1_text}

```
class net_printer::Text
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  lines` |
`public def __init__(self)` |
`public def context(self,text)` |
`public def add(self,text)` |
`public def __str__(self)` |

## Members

#### `public  lines` {#classnet__printer_1_1_text_1a849729218bab097310429f2bde256ddb}





#### `public def __init__(self)` {#classnet__printer_1_1_text_1af1002cad53acac0f7ae4834339b08a5e}





#### `public def context(self,text)` {#classnet__printer_1_1_text_1ac68b9144c9942e37b25b9ed2dbe2d73c}





#### `public def add(self,text)` {#classnet__printer_1_1_text_1aa5f27fd5883402b2364e6e80752dca28}





#### `public def __str__(self)` {#classnet__printer_1_1_text_1a1cd234d9095eeab9f267ce6ec5fa2fbf}





# class `net_printer::Visitor` {#classnet__printer_1_1_visitor}

```
class net_printer::Visitor
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  visitors` |
`public def register(cls,Type)` |
`public def __call__(self,obj,args,kwargs)` |

## Members

#### `public  visitors` {#classnet__printer_1_1_visitor_1a6cfcf9c9b9d275e125524fc5e3b34734}





#### `public def register(cls,Type)` {#classnet__printer_1_1_visitor_1ad97907804278b7f2e49b429e2893c726}





#### `public def __call__(self,obj,args,kwargs)` {#classnet__printer_1_1_visitor_1aa8302c5b448273b2c0ce9b0368b3d08b}





# namespace `net_printer_test` {#namespacenet__printer__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`net_printer_test::TestNetPrinter`](#classnet__printer__test_1_1_test_net_printer)    |
# class `net_printer_test::TestNetPrinter` {#classnet__printer__test_1_1_test_net_printer}

```
class net_printer_test::TestNetPrinter
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_print(self)` |
`public def test_valid_job(self)` |
`public def test_undefined_blob(self)` |
`public def test_multiple_definition(self)` |

## Members

#### `public def test_print(self)` {#classnet__printer__test_1_1_test_net_printer_1a0de29b1e30a5b6b6563b26fa33d893c6}





#### `public def test_valid_job(self)` {#classnet__printer__test_1_1_test_net_printer_1ad662bd9a5c6c6ec8e27c7f08ca6e3e6c}





#### `public def test_undefined_blob(self)` {#classnet__printer__test_1_1_test_net_printer_1aee3885861b4efab91ad4981b5628c545}





#### `public def test_multiple_definition(self)` {#classnet__printer__test_1_1_test_net_printer_1a9a5597502c3ad0212477283faf603aba}





# namespace `one_hot_ops_test` {#namespaceone__hot__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`one_hot_ops_test::TestOneHotOps`](#classone__hot__ops__test_1_1_test_one_hot_ops)    |
# class `one_hot_ops_test::TestOneHotOps` {#classone__hot__ops__test_1_1_test_one_hot_ops}

```
class one_hot_ops_test::TestOneHotOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_one_hot(self,`[`hot_indices`](#classone__hot__ops__test_1_1_test_one_hot_ops_1abe4411b84cff317db55ef1eb251e5e23)`,`[`end_padding`](#classone__hot__ops__test_1_1_test_one_hot_ops_1a97467c3d41effd0e7adb01b9396c8cc4)`)` |
`public def test_segment_one_hot(self,`[`hot_indices`](#classone__hot__ops__test_1_1_test_one_hot_ops_1abe4411b84cff317db55ef1eb251e5e23)`)` |

## Members

#### `public def test_one_hot(self,`[`hot_indices`](#classone__hot__ops__test_1_1_test_one_hot_ops_1abe4411b84cff317db55ef1eb251e5e23)`,`[`end_padding`](#classone__hot__ops__test_1_1_test_one_hot_ops_1a97467c3d41effd0e7adb01b9396c8cc4)`)` {#classone__hot__ops__test_1_1_test_one_hot_ops_1aa7c0325c4e681ff8bea8b760da5aaf99}





#### `public def test_segment_one_hot(self,`[`hot_indices`](#classone__hot__ops__test_1_1_test_one_hot_ops_1abe4411b84cff317db55ef1eb251e5e23)`)` {#classone__hot__ops__test_1_1_test_one_hot_ops_1a35bf522569cfba2f053fa6079038b6e8}





# namespace `optimizer` {#namespaceoptimizer}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`optimizer::AdagradOptimizer`](#classoptimizer_1_1_adagrad_optimizer)    |
`class `[`optimizer::AdamOptimizer`](#classoptimizer_1_1_adam_optimizer)    |
`class `[`optimizer::FtrlOptimizer`](#classoptimizer_1_1_ftrl_optimizer)    |
`class `[`optimizer::Optimizer`](#classoptimizer_1_1_optimizer)    |
`class `[`optimizer::SgdOptimizer`](#classoptimizer_1_1_sgd_optimizer)    |
# class `optimizer::AdagradOptimizer` {#classoptimizer_1_1_adagrad_optimizer}

```
class optimizer::AdagradOptimizer
  : public optimizer.Optimizer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  alpha` |
`public  epsilon` |
`public  policy` |
`public  sparse_dedup_aggregator` |
`public  engine` |
`public  init_kwargs` |
`public def __init__(self,`[`alpha`](#classoptimizer_1_1_adagrad_optimizer_1a2484901fee1ad004f47fd62385e1a9da)`,`[`epsilon`](#classoptimizer_1_1_adagrad_optimizer_1a3706bef9d3a49e26f955304de3c49190)`,`[`policy`](#classoptimizer_1_1_adagrad_optimizer_1a301f1a18d32719d095741daf9948b28d)`,`[`sparse_dedup_aggregator`](#classoptimizer_1_1_adagrad_optimizer_1a8ad79cc2a64770c4d570372e400afff2)`,`[`engine`](#classoptimizer_1_1_adagrad_optimizer_1a3267b1cde2da4a20dcd4beadf41126f3)`,kwargs)` |
`public def __call__(self,net,param_init_net,param,grad)` |

## Members

#### `public  alpha` {#classoptimizer_1_1_adagrad_optimizer_1a2484901fee1ad004f47fd62385e1a9da}





#### `public  epsilon` {#classoptimizer_1_1_adagrad_optimizer_1a3706bef9d3a49e26f955304de3c49190}





#### `public  policy` {#classoptimizer_1_1_adagrad_optimizer_1a301f1a18d32719d095741daf9948b28d}





#### `public  sparse_dedup_aggregator` {#classoptimizer_1_1_adagrad_optimizer_1a8ad79cc2a64770c4d570372e400afff2}





#### `public  engine` {#classoptimizer_1_1_adagrad_optimizer_1a3267b1cde2da4a20dcd4beadf41126f3}





#### `public  init_kwargs` {#classoptimizer_1_1_adagrad_optimizer_1a3416404236548c0fdac8c829a09a7d86}





#### `public def __init__(self,`[`alpha`](#classoptimizer_1_1_adagrad_optimizer_1a2484901fee1ad004f47fd62385e1a9da)`,`[`epsilon`](#classoptimizer_1_1_adagrad_optimizer_1a3706bef9d3a49e26f955304de3c49190)`,`[`policy`](#classoptimizer_1_1_adagrad_optimizer_1a301f1a18d32719d095741daf9948b28d)`,`[`sparse_dedup_aggregator`](#classoptimizer_1_1_adagrad_optimizer_1a8ad79cc2a64770c4d570372e400afff2)`,`[`engine`](#classoptimizer_1_1_adagrad_optimizer_1a3267b1cde2da4a20dcd4beadf41126f3)`,kwargs)` {#classoptimizer_1_1_adagrad_optimizer_1aa1ce7c468793540f64bf1604e94aa325}





#### `public def __call__(self,net,param_init_net,param,grad)` {#classoptimizer_1_1_adagrad_optimizer_1a7f225c13977762a83498a3a08c58c700}





# class `optimizer::AdamOptimizer` {#classoptimizer_1_1_adam_optimizer}

```
class optimizer::AdamOptimizer
  : public optimizer.Optimizer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  alpha` |
`public  beta1` |
`public  beta2` |
`public  epsilon` |
`public  policy` |
`public  sparse_dedup_aggregator` |
`public  engine` |
`public  init_kwargs` |
`public def __init__(self,`[`alpha`](#classoptimizer_1_1_adam_optimizer_1a13678a2f7d7816d52bb65ccd56068028)`,`[`beta1`](#classoptimizer_1_1_adam_optimizer_1ac4e7565f63b9082f89e5f328d1941f07)`,`[`beta2`](#classoptimizer_1_1_adam_optimizer_1a3ea4be59f657b960346d0c32848ca875)`,`[`epsilon`](#classoptimizer_1_1_adam_optimizer_1abc77a49ce89d766ef07c7ceb3582a031)`,`[`policy`](#classoptimizer_1_1_adam_optimizer_1aa119b7a6b2a621939d2c4dcf3cd147c2)`,`[`sparse_dedup_aggregator`](#classoptimizer_1_1_adam_optimizer_1a0e617ea127972dc8507ee5cf9db4781b)`,`[`engine`](#classoptimizer_1_1_adam_optimizer_1a04ce6d3314007f90388c5273b4967324)`,kwargs)` |
`public def __call__(self,net,param_init_net,param,grad)` |

## Members

#### `public  alpha` {#classoptimizer_1_1_adam_optimizer_1a13678a2f7d7816d52bb65ccd56068028}





#### `public  beta1` {#classoptimizer_1_1_adam_optimizer_1ac4e7565f63b9082f89e5f328d1941f07}





#### `public  beta2` {#classoptimizer_1_1_adam_optimizer_1a3ea4be59f657b960346d0c32848ca875}





#### `public  epsilon` {#classoptimizer_1_1_adam_optimizer_1abc77a49ce89d766ef07c7ceb3582a031}





#### `public  policy` {#classoptimizer_1_1_adam_optimizer_1aa119b7a6b2a621939d2c4dcf3cd147c2}





#### `public  sparse_dedup_aggregator` {#classoptimizer_1_1_adam_optimizer_1a0e617ea127972dc8507ee5cf9db4781b}





#### `public  engine` {#classoptimizer_1_1_adam_optimizer_1a04ce6d3314007f90388c5273b4967324}





#### `public  init_kwargs` {#classoptimizer_1_1_adam_optimizer_1a3c1d84815ada1419a0b6f7aee7d75e31}





#### `public def __init__(self,`[`alpha`](#classoptimizer_1_1_adam_optimizer_1a13678a2f7d7816d52bb65ccd56068028)`,`[`beta1`](#classoptimizer_1_1_adam_optimizer_1ac4e7565f63b9082f89e5f328d1941f07)`,`[`beta2`](#classoptimizer_1_1_adam_optimizer_1a3ea4be59f657b960346d0c32848ca875)`,`[`epsilon`](#classoptimizer_1_1_adam_optimizer_1abc77a49ce89d766ef07c7ceb3582a031)`,`[`policy`](#classoptimizer_1_1_adam_optimizer_1aa119b7a6b2a621939d2c4dcf3cd147c2)`,`[`sparse_dedup_aggregator`](#classoptimizer_1_1_adam_optimizer_1a0e617ea127972dc8507ee5cf9db4781b)`,`[`engine`](#classoptimizer_1_1_adam_optimizer_1a04ce6d3314007f90388c5273b4967324)`,kwargs)` {#classoptimizer_1_1_adam_optimizer_1a5cf06ca0fba96c5cc2fdfdafe00747fc}





#### `public def __call__(self,net,param_init_net,param,grad)` {#classoptimizer_1_1_adam_optimizer_1a847136a2364b895535302fa7f25984bd}





# class `optimizer::FtrlOptimizer` {#classoptimizer_1_1_ftrl_optimizer}

```
class optimizer::FtrlOptimizer
  : public optimizer.Optimizer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  alpha` |
`public  beta` |
`public  lambda1` |
`public  lambda2` |
`public  sparse_dedup_aggregator` |
`public  engine` |
`public def __init__(self,`[`alpha`](#classoptimizer_1_1_ftrl_optimizer_1a92f97f5fe519c610f7ae40d4d6fe9f02)`,`[`beta`](#classoptimizer_1_1_ftrl_optimizer_1a0354b0725ab9c3397018d8b4063794a9)`,`[`lambda1`](#classoptimizer_1_1_ftrl_optimizer_1a549a35c500de5f6a3a1c8ad11e01f06a)`,`[`lambda2`](#classoptimizer_1_1_ftrl_optimizer_1aee8a71ca1cbe035a609803d9f82553aa)`,`[`sparse_dedup_aggregator`](#classoptimizer_1_1_ftrl_optimizer_1aaa3c49fae44869119d08383e3bfecdc6)`,`[`engine`](#classoptimizer_1_1_ftrl_optimizer_1a905cd4ce70c32ee0ac1d69a3103c1e89)`)` |
`public def __call__(self,net,param_init_net,param,grad)` |

## Members

#### `public  alpha` {#classoptimizer_1_1_ftrl_optimizer_1a92f97f5fe519c610f7ae40d4d6fe9f02}





#### `public  beta` {#classoptimizer_1_1_ftrl_optimizer_1a0354b0725ab9c3397018d8b4063794a9}





#### `public  lambda1` {#classoptimizer_1_1_ftrl_optimizer_1a549a35c500de5f6a3a1c8ad11e01f06a}





#### `public  lambda2` {#classoptimizer_1_1_ftrl_optimizer_1aee8a71ca1cbe035a609803d9f82553aa}





#### `public  sparse_dedup_aggregator` {#classoptimizer_1_1_ftrl_optimizer_1aaa3c49fae44869119d08383e3bfecdc6}





#### `public  engine` {#classoptimizer_1_1_ftrl_optimizer_1a905cd4ce70c32ee0ac1d69a3103c1e89}





#### `public def __init__(self,`[`alpha`](#classoptimizer_1_1_ftrl_optimizer_1a92f97f5fe519c610f7ae40d4d6fe9f02)`,`[`beta`](#classoptimizer_1_1_ftrl_optimizer_1a0354b0725ab9c3397018d8b4063794a9)`,`[`lambda1`](#classoptimizer_1_1_ftrl_optimizer_1a549a35c500de5f6a3a1c8ad11e01f06a)`,`[`lambda2`](#classoptimizer_1_1_ftrl_optimizer_1aee8a71ca1cbe035a609803d9f82553aa)`,`[`sparse_dedup_aggregator`](#classoptimizer_1_1_ftrl_optimizer_1aaa3c49fae44869119d08383e3bfecdc6)`,`[`engine`](#classoptimizer_1_1_ftrl_optimizer_1a905cd4ce70c32ee0ac1d69a3103c1e89)`)` {#classoptimizer_1_1_ftrl_optimizer_1afba5495d94c364ef029f6f6aaa2b567e}





#### `public def __call__(self,net,param_init_net,param,grad)` {#classoptimizer_1_1_ftrl_optimizer_1ab4e35f4af2b0c7a3f78466c8849b0f2a}





# class `optimizer::Optimizer` {#classoptimizer_1_1_optimizer}

```
class optimizer::Optimizer
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self)` |
`public def __call__(self,net,param_init_net,param,grad)` |

## Members

#### `public def __init__(self)` {#classoptimizer_1_1_optimizer_1a4233d665697bc97859078535317a3071}





#### `public def __call__(self,net,param_init_net,param,grad)` {#classoptimizer_1_1_optimizer_1ac65eaba16e186738261550c02f62cabe}





# class `optimizer::SgdOptimizer` {#classoptimizer_1_1_sgd_optimizer}

```
class optimizer::SgdOptimizer
  : public optimizer.Optimizer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  base_learning_rate` |
`public  policy` |
`public  momentum` |
`public  init_kwargs` |
`public def __init__(self,`[`base_learning_rate`](#classoptimizer_1_1_sgd_optimizer_1a4efe795f03f4faa6f1291994bf38ee83)`,`[`policy`](#classoptimizer_1_1_sgd_optimizer_1af83ec1b7926db325778c9347cc5c4e8a)`,`[`momentum`](#classoptimizer_1_1_sgd_optimizer_1a59b097c492c95a54df5ed0c54417d731)`,kwargs)` |
`public def __call__(self,net,param_init_net,param,grad)` |

## Members

#### `public  base_learning_rate` {#classoptimizer_1_1_sgd_optimizer_1a4efe795f03f4faa6f1291994bf38ee83}





#### `public  policy` {#classoptimizer_1_1_sgd_optimizer_1af83ec1b7926db325778c9347cc5c4e8a}





#### `public  momentum` {#classoptimizer_1_1_sgd_optimizer_1a59b097c492c95a54df5ed0c54417d731}





#### `public  init_kwargs` {#classoptimizer_1_1_sgd_optimizer_1a953ee41fc3a2a7bc5d686764c974b91e}





#### `public def __init__(self,`[`base_learning_rate`](#classoptimizer_1_1_sgd_optimizer_1a4efe795f03f4faa6f1291994bf38ee83)`,`[`policy`](#classoptimizer_1_1_sgd_optimizer_1af83ec1b7926db325778c9347cc5c4e8a)`,`[`momentum`](#classoptimizer_1_1_sgd_optimizer_1a59b097c492c95a54df5ed0c54417d731)`,kwargs)` {#classoptimizer_1_1_sgd_optimizer_1a457bae076d7711e00b46fa46c19194b0}





#### `public def __call__(self,net,param_init_net,param,grad)` {#classoptimizer_1_1_sgd_optimizer_1a2b501700e6195ae20a9c989cb9bd9c98}





# namespace `optimizer_test` {#namespaceoptimizer__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`optimizer_test::TestAdagrad`](#classoptimizer__test_1_1_test_adagrad)    |
`class `[`optimizer_test::TestAdam`](#classoptimizer__test_1_1_test_adam)    |
`class `[`optimizer_test::TestFtrl`](#classoptimizer__test_1_1_test_ftrl)    |
`class `[`optimizer_test::TestSgd`](#classoptimizer__test_1_1_test_sgd)    |
# class `optimizer_test::TestAdagrad` {#classoptimizer__test_1_1_test_adagrad}

```
class optimizer_test::TestAdagrad
  : public OptimizerTestBase
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def build_optimizer(self,model)` |

## Members

#### `public def build_optimizer(self,model)` {#classoptimizer__test_1_1_test_adagrad_1a6d1ffd87d2637e7286077b00d1f051b6}





# class `optimizer_test::TestAdam` {#classoptimizer__test_1_1_test_adam}

```
class optimizer_test::TestAdam
  : public OptimizerTestBase
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def build_optimizer(self,model)` |

## Members

#### `public def build_optimizer(self,model)` {#classoptimizer__test_1_1_test_adam_1acc5e8ee89f92be04ae0834462dcb9f85}





# class `optimizer_test::TestFtrl` {#classoptimizer__test_1_1_test_ftrl}

```
class optimizer_test::TestFtrl
  : public OptimizerTestBase
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def build_optimizer(self,model)` |

## Members

#### `public def build_optimizer(self,model)` {#classoptimizer__test_1_1_test_ftrl_1afeaa4a57444c82cf0fe77f50eb532cb8}





# class `optimizer_test::TestSgd` {#classoptimizer__test_1_1_test_sgd}

```
class optimizer_test::TestSgd
  : public OptimizerTestBase
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def build_optimizer(self,model)` |

## Members

#### `public def build_optimizer(self,model)` {#classoptimizer__test_1_1_test_sgd_1a97aa808f0085c71071982c74055a7469}





# namespace `optimizer_test_util` {#namespaceoptimizer__test__util}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`optimizer_test_util::OptimizerTestBase`](#classoptimizer__test__util_1_1_optimizer_test_base)    |
# class `optimizer_test_util::OptimizerTestBase` {#classoptimizer__test__util_1_1_optimizer_test_base}

```
class optimizer_test_util::OptimizerTestBase
  : public object
```  



This is an abstract base class.
Don't inherit from unittest.TestCase, and don't name it 'Test*'.
Do, however, do these things in classes which inherit from this.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testDense(self)` |
`public def testSparse(self)` |

## Members

#### `public def testDense(self)` {#classoptimizer__test__util_1_1_optimizer_test_base_1ad37868d1098e354b5d4b8ad6bff2e3b0}





#### `public def testSparse(self)` {#classoptimizer__test__util_1_1_optimizer_test_base_1a52936fd73172886679f1b4ef3f692aa2}





# namespace `pack_ops_test` {#namespacepack__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`pack_ops_test::TestTensorPackOps`](#classpack__ops__test_1_1_test_tensor_pack_ops)    |
# class `pack_ops_test::TestTensorPackOps` {#classpack__ops__test_1_1_test_tensor_pack_ops}

```
class pack_ops_test::TestTensorPackOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_pack_ops(self)` |
`public def test_pad_minf(self)` |

## Members

#### `public def test_pack_ops(self)` {#classpack__ops__test_1_1_test_tensor_pack_ops_1a28d66e8e1abd84742ff715979ec1da8f}





#### `public def test_pad_minf(self)` {#classpack__ops__test_1_1_test_tensor_pack_ops_1acf2a3601cca2166d8c542fcbbd09b40a}





# namespace `parser` {#namespaceparser}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`parser::Parser`](#classparser_1_1_parser)    |
# class `parser::Parser` {#classparser_1_1_parser}

```
class parser::Parser
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  text` |
`public  lines` |
`public  formatter` |
`public def __init__(self,`[`text`](#classparser_1_1_parser_1aacf1ad73bffd7231436ba8722ad52536)`,`[`formatter`](#classparser_1_1_parser_1a1b513deec36db41c1c204229ad1c586e)`)` |
`public def parseText(self)` |
`public def parse(self)` |

## Members

#### `public  text` {#classparser_1_1_parser_1aacf1ad73bffd7231436ba8722ad52536}





#### `public  lines` {#classparser_1_1_parser_1ade9f7734255f98ac841f310cb7067136}





#### `public  formatter` {#classparser_1_1_parser_1a1b513deec36db41c1c204229ad1c586e}





#### `public def __init__(self,`[`text`](#classparser_1_1_parser_1aacf1ad73bffd7231436ba8722ad52536)`,`[`formatter`](#classparser_1_1_parser_1a1b513deec36db41c1c204229ad1c586e)`)` {#classparser_1_1_parser_1aa80a936591d5da929568b8743d024182}





#### `public def parseText(self)` {#classparser_1_1_parser_1ad73a9ea0ebb1ef542f4b1c3d15f50662}





#### `public def parse(self)` {#classparser_1_1_parser_1a40f8fbbd00980a586139479c0523506a}





# namespace `partition_ops_test` {#namespacepartition__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`partition_ops_test::TestPartitionOps`](#classpartition__ops__test_1_1_test_partition_ops)    |
# class `partition_ops_test::TestPartitionOps` {#classpartition__ops__test_1_1_test_partition_ops}

```
class partition_ops_test::TestPartitionOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_configs(self)` |
`public def testPartition(self)` |
`public def testLengthsPartition(self)` |

## Members

#### `public def test_configs(self)` {#classpartition__ops__test_1_1_test_partition_ops_1a16a2cc4e2140e67af928411f4f3db6fd}





#### `public def testPartition(self)` {#classpartition__ops__test_1_1_test_partition_ops_1aeb18e2c45d7cc41aa5a8d1914552112f}





#### `public def testLengthsPartition(self)` {#classpartition__ops__test_1_1_test_partition_ops_1a3da6158ee742465bc5cf5ad847349d27}





# namespace `piecewise_linear_transform_test` {#namespacepiecewise__linear__transform__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`piecewise_linear_transform_test::TestPiecewiseLinearTransform`](#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform)    |
# class `piecewise_linear_transform_test::TestPiecewiseLinearTransform` {#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform}

```
class piecewise_linear_transform_test::TestPiecewiseLinearTransform
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_piecewise_linear_transform_general(self,`[`n`](#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform_1aebf91bc3360b87305128a4551777b0ac)`,gc,dc)` |
`public def test_piecewise_linear_transform_binary(self,`[`n`](#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform_1aebf91bc3360b87305128a4551777b0ac)`,gc,dc)` |

## Members

#### `public def test_piecewise_linear_transform_general(self,`[`n`](#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform_1aebf91bc3360b87305128a4551777b0ac)`,gc,dc)` {#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform_1a3779208eec5ec26f51f92721f79e4a3f}





#### `public def test_piecewise_linear_transform_binary(self,`[`n`](#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform_1aebf91bc3360b87305128a4551777b0ac)`,gc,dc)` {#classpiecewise__linear__transform__test_1_1_test_piecewise_linear_transform_1a57119e6124c9751592e87270f5725222}





# namespace `pipeline` {#namespacepipeline}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`pipeline::NetProcessor`](#classpipeline_1_1_net_processor)    |
`class `[`pipeline::Output`](#classpipeline_1_1_output)    |
`class `[`pipeline::ProcessingReader`](#classpipeline_1_1_processing_reader)    |
# class `pipeline::NetProcessor` {#classpipeline_1_1_net_processor}

```
class pipeline::NetProcessor
  : public object
```  



Processor that clones a core.Net each time it's called, executing
the cloned net as the processor. It requires the Net to have input
and (optionally) output records set, with net.set_input_record() and
net.set_output_record().

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  name` |
`public  thread_init_nets` |
`public  net` |
`public def __init__(self,`[`net`](#classpipeline_1_1_net_processor_1a9151f57055f203143984dace3aebfa46)`,stop_signal,`[`thread_init_nets`](#classpipeline_1_1_net_processor_1a406dbf60e9d26c4f8e60e75e1a230e85)`,`[`name`](#classpipeline_1_1_net_processor_1abb0751249ef06ddf55b2027f0b2e68a3)`)` |
`public def setup(self,init_net)` |
`public def __call__(self,rec)` |
`public def blob_maps(self)` |

## Members

#### `public  name` {#classpipeline_1_1_net_processor_1abb0751249ef06ddf55b2027f0b2e68a3}





#### `public  thread_init_nets` {#classpipeline_1_1_net_processor_1a406dbf60e9d26c4f8e60e75e1a230e85}





#### `public  net` {#classpipeline_1_1_net_processor_1a9151f57055f203143984dace3aebfa46}





#### `public def __init__(self,`[`net`](#classpipeline_1_1_net_processor_1a9151f57055f203143984dace3aebfa46)`,stop_signal,`[`thread_init_nets`](#classpipeline_1_1_net_processor_1a406dbf60e9d26c4f8e60e75e1a230e85)`,`[`name`](#classpipeline_1_1_net_processor_1abb0751249ef06ddf55b2027f0b2e68a3)`)` {#classpipeline_1_1_net_processor_1aed2e6abf7ee131fe092fcd311c665cfb}





#### `public def setup(self,init_net)` {#classpipeline_1_1_net_processor_1a61714cdb31ac335ed60cfdb2418841f0}





#### `public def __call__(self,rec)` {#classpipeline_1_1_net_processor_1a4ac7866f85bf51563e85d18f00f318a8}





#### `public def blob_maps(self)` {#classpipeline_1_1_net_processor_1a55f44c8f3a1ebce13d9e79c6bc577108}





# class `pipeline::Output` {#classpipeline_1_1_output}

```
class pipeline::Output
  : public object
```  



Represents the result of a processor function. A processor can either
return an Output, or it can return a record, in which case an Output will be
created for it afterwards.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  nets` |
`public  record` |
`public  should_stop` |
`public def __init__(self,`[`nets`](#classpipeline_1_1_output_1a19b923cddcbd0f3194272bdcd8372c6d)`,`[`record`](#classpipeline_1_1_output_1a418ded7fa8c2c79768a88258912260f9)`,`[`should_stop`](#classpipeline_1_1_output_1a271f4e196f0221c074cb1c38bc10cc42)`)` |

## Members

#### `public  nets` {#classpipeline_1_1_output_1a19b923cddcbd0f3194272bdcd8372c6d}





#### `public  record` {#classpipeline_1_1_output_1a418ded7fa8c2c79768a88258912260f9}





#### `public  should_stop` {#classpipeline_1_1_output_1a271f4e196f0221c074cb1c38bc10cc42}





#### `public def __init__(self,`[`nets`](#classpipeline_1_1_output_1a19b923cddcbd0f3194272bdcd8372c6d)`,`[`record`](#classpipeline_1_1_output_1a418ded7fa8c2c79768a88258912260f9)`,`[`should_stop`](#classpipeline_1_1_output_1a271f4e196f0221c074cb1c38bc10cc42)`)` {#classpipeline_1_1_output_1af03e479176663cda92ce56c6dd22939d}





# class `pipeline::ProcessingReader` {#classpipeline_1_1_processing_reader}

```
class pipeline::ProcessingReader
  : public Reader
```  



Reader that reads from a upstream reader, calls the processor, and returns
the processed record.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  reader` |
`public  processor` |
`public def __init__(self,`[`reader`](#classpipeline_1_1_processing_reader_1a97c85a812e3515e7ffdd1f2ae6077f80)`,`[`processor`](#classpipeline_1_1_processing_reader_1ac23eccc085f504356611d4b1df498b3e)`)` |
`public def setup_ex(self,init_net,finish_net)` |
`public def read_ex(self,init_net,exit_net)` |

## Members

#### `public  reader` {#classpipeline_1_1_processing_reader_1a97c85a812e3515e7ffdd1f2ae6077f80}





#### `public  processor` {#classpipeline_1_1_processing_reader_1ac23eccc085f504356611d4b1df498b3e}





#### `public def __init__(self,`[`reader`](#classpipeline_1_1_processing_reader_1a97c85a812e3515e7ffdd1f2ae6077f80)`,`[`processor`](#classpipeline_1_1_processing_reader_1ac23eccc085f504356611d4b1df498b3e)`)` {#classpipeline_1_1_processing_reader_1a2f8642c78bdd902c9581fc3e8e20a71c}





#### `public def setup_ex(self,init_net,finish_net)` {#classpipeline_1_1_processing_reader_1a2ee47745eb1baaeadb53b5f12e9c3777}





#### `public def read_ex(self,init_net,exit_net)` {#classpipeline_1_1_processing_reader_1a9385756a9e82b4ce7d48f623dc8056fa}





# namespace `pooling_test` {#namespacepooling__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`pooling_test::TestPooling`](#classpooling__test_1_1_test_pooling)    |
# class `pooling_test::TestPooling` {#classpooling__test_1_1_test_pooling}

```
class pooling_test::TestPooling
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_pooling_separate_stride_pad(self,`[`stride_h`](#classpooling__test_1_1_test_pooling_1af4101c51f4d73b4cd1f31109de1ac5f4)`,`[`stride_w`](#classpooling__test_1_1_test_pooling_1abc7c4062f202cf958116e0aa7303211e)`,`[`pad_t`](#classpooling__test_1_1_test_pooling_1acd048ecc8c7229c1ba263465b018afd5)`,`[`pad_l`](#classpooling__test_1_1_test_pooling_1a19bdc857f909b9c0b60549a26ced3d53)`,`[`pad_b`](#classpooling__test_1_1_test_pooling_1a64ee810fac0d0be7df933fca5b1861bd)`,`[`pad_r`](#classpooling__test_1_1_test_pooling_1a518fae637e3c19632a7d21d037e12e87)`,`[`kernel`](#classpooling__test_1_1_test_pooling_1a92772182f27d5df9d4fbed45684e7988)`,`[`size`](#classpooling__test_1_1_test_pooling_1a670feeba3e34d08defc83c952c6db92a)`,`[`input_channels`](#classpooling__test_1_1_test_pooling_1a771505a103ee3a7b32d0b5bd86206013)`,`[`batch_size`](#classpooling__test_1_1_test_pooling_1a5ed3601e3ebf77e27bd98a69aabf42bc)`,`[`order`](#classpooling__test_1_1_test_pooling_1adfbbae9236cba0e59ef7af755ea8431d)`,`[`method`](#classpooling__test_1_1_test_pooling_1a12c9273a2bd100dce3858e5760e542d8)`,gc,dc)` |
`public def test_pooling_big_batch(self,gc,dc)` |
`public def test_pooling(self,`[`stride`](#classpooling__test_1_1_test_pooling_1a7cfa48f3aed6860d5df65f6ff0c79cc2)`,`[`pad`](#classpooling__test_1_1_test_pooling_1a4b6af588d49a4136db638436990a6889)`,`[`kernel`](#classpooling__test_1_1_test_pooling_1a92772182f27d5df9d4fbed45684e7988)`,`[`size`](#classpooling__test_1_1_test_pooling_1a670feeba3e34d08defc83c952c6db92a)`,`[`input_channels`](#classpooling__test_1_1_test_pooling_1a771505a103ee3a7b32d0b5bd86206013)`,`[`batch_size`](#classpooling__test_1_1_test_pooling_1a5ed3601e3ebf77e27bd98a69aabf42bc)`,`[`order`](#classpooling__test_1_1_test_pooling_1adfbbae9236cba0e59ef7af755ea8431d)`,`[`method`](#classpooling__test_1_1_test_pooling_1a12c9273a2bd100dce3858e5760e542d8)`,`[`engine`](#classpooling__test_1_1_test_pooling_1abe3e323e2a8c7f0b476aa1636e7822bc)`,gc,dc)` |
`public def test_global_pooling(self,`[`size`](#classpooling__test_1_1_test_pooling_1a670feeba3e34d08defc83c952c6db92a)`,`[`input_channels`](#classpooling__test_1_1_test_pooling_1a771505a103ee3a7b32d0b5bd86206013)`,`[`batch_size`](#classpooling__test_1_1_test_pooling_1a5ed3601e3ebf77e27bd98a69aabf42bc)`,`[`order`](#classpooling__test_1_1_test_pooling_1adfbbae9236cba0e59ef7af755ea8431d)`,`[`method`](#classpooling__test_1_1_test_pooling_1a12c9273a2bd100dce3858e5760e542d8)`,`[`engine`](#classpooling__test_1_1_test_pooling_1abe3e323e2a8c7f0b476aa1636e7822bc)`,gc,dc)` |

## Members

#### `public def test_pooling_separate_stride_pad(self,`[`stride_h`](#classpooling__test_1_1_test_pooling_1af4101c51f4d73b4cd1f31109de1ac5f4)`,`[`stride_w`](#classpooling__test_1_1_test_pooling_1abc7c4062f202cf958116e0aa7303211e)`,`[`pad_t`](#classpooling__test_1_1_test_pooling_1acd048ecc8c7229c1ba263465b018afd5)`,`[`pad_l`](#classpooling__test_1_1_test_pooling_1a19bdc857f909b9c0b60549a26ced3d53)`,`[`pad_b`](#classpooling__test_1_1_test_pooling_1a64ee810fac0d0be7df933fca5b1861bd)`,`[`pad_r`](#classpooling__test_1_1_test_pooling_1a518fae637e3c19632a7d21d037e12e87)`,`[`kernel`](#classpooling__test_1_1_test_pooling_1a92772182f27d5df9d4fbed45684e7988)`,`[`size`](#classpooling__test_1_1_test_pooling_1a670feeba3e34d08defc83c952c6db92a)`,`[`input_channels`](#classpooling__test_1_1_test_pooling_1a771505a103ee3a7b32d0b5bd86206013)`,`[`batch_size`](#classpooling__test_1_1_test_pooling_1a5ed3601e3ebf77e27bd98a69aabf42bc)`,`[`order`](#classpooling__test_1_1_test_pooling_1adfbbae9236cba0e59ef7af755ea8431d)`,`[`method`](#classpooling__test_1_1_test_pooling_1a12c9273a2bd100dce3858e5760e542d8)`,gc,dc)` {#classpooling__test_1_1_test_pooling_1acb78dcd90a741dcd6d50fb21cba14bcf}





#### `public def test_pooling_big_batch(self,gc,dc)` {#classpooling__test_1_1_test_pooling_1a402741b55eccfc94a47903fca32d4e92}





#### `public def test_pooling(self,`[`stride`](#classpooling__test_1_1_test_pooling_1a7cfa48f3aed6860d5df65f6ff0c79cc2)`,`[`pad`](#classpooling__test_1_1_test_pooling_1a4b6af588d49a4136db638436990a6889)`,`[`kernel`](#classpooling__test_1_1_test_pooling_1a92772182f27d5df9d4fbed45684e7988)`,`[`size`](#classpooling__test_1_1_test_pooling_1a670feeba3e34d08defc83c952c6db92a)`,`[`input_channels`](#classpooling__test_1_1_test_pooling_1a771505a103ee3a7b32d0b5bd86206013)`,`[`batch_size`](#classpooling__test_1_1_test_pooling_1a5ed3601e3ebf77e27bd98a69aabf42bc)`,`[`order`](#classpooling__test_1_1_test_pooling_1adfbbae9236cba0e59ef7af755ea8431d)`,`[`method`](#classpooling__test_1_1_test_pooling_1a12c9273a2bd100dce3858e5760e542d8)`,`[`engine`](#classpooling__test_1_1_test_pooling_1abe3e323e2a8c7f0b476aa1636e7822bc)`,gc,dc)` {#classpooling__test_1_1_test_pooling_1a6dcbf687faf0421c90708210f6b82008}





#### `public def test_global_pooling(self,`[`size`](#classpooling__test_1_1_test_pooling_1a670feeba3e34d08defc83c952c6db92a)`,`[`input_channels`](#classpooling__test_1_1_test_pooling_1a771505a103ee3a7b32d0b5bd86206013)`,`[`batch_size`](#classpooling__test_1_1_test_pooling_1a5ed3601e3ebf77e27bd98a69aabf42bc)`,`[`order`](#classpooling__test_1_1_test_pooling_1adfbbae9236cba0e59ef7af755ea8431d)`,`[`method`](#classpooling__test_1_1_test_pooling_1a12c9273a2bd100dce3858e5760e542d8)`,`[`engine`](#classpooling__test_1_1_test_pooling_1abe3e323e2a8c7f0b476aa1636e7822bc)`,gc,dc)` {#classpooling__test_1_1_test_pooling_1a8452cb838f351a5d247655075af3c3f6}





# namespace `pow_op_test` {#namespacepow__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`pow_op_test::TestPowOp`](#classpow__op__test_1_1_test_pow_op)    |
# class `pow_op_test::TestPowOp` {#classpow__op__test_1_1_test_pow_op}

```
class pow_op_test::TestPowOp
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_elementwise_power(self,`[`X`](#classpow__op__test_1_1_test_pow_op_1a45f10fa9f2fc9ac1768aefff42d814a3)`,`[`exponent`](#classpow__op__test_1_1_test_pow_op_1a45862df29a68f33a0fd68207c7542ac4)`,gc,dc)` |

## Members

#### `public def test_elementwise_power(self,`[`X`](#classpow__op__test_1_1_test_pow_op_1a45f10fa9f2fc9ac1768aefff42d814a3)`,`[`exponent`](#classpow__op__test_1_1_test_pow_op_1a45862df29a68f33a0fd68207c7542ac4)`,gc,dc)` {#classpow__op__test_1_1_test_pow_op_1acd29a6bb5973f10cce065f488c66f88e}





# namespace `python_op_test` {#namespacepython__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`python_op_test::PythonOpTest`](#classpython__op__test_1_1_python_op_test)    |
# class `python_op_test::PythonOpTest` {#classpython__op__test_1_1_python_op_test}

```
class python_op_test::PythonOpTest
  : public HypothesisTestCase
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_multithreaded_evaluation_numba_nogil(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`,`[`n`](#classpython__op__test_1_1_python_op_test_1a6be35f7bba1a93c244c1bdb017f25095)`,`[`w`](#classpython__op__test_1_1_python_op_test_1afb6a2355c9e3f4001ed9ca3ce35f5c86)`)` |
`public def test_feed(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` |
`public def test_exception(self)` |
`public def test_feed_with_helper_function(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` |
`public def test_feed_with_gc(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` |
`public def test_reshape(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` |
`public def test_workspace_manipulation(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` |
`public def test_caught_exception_doesnt_terminate(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` |
`public def test_multithreaded_evaluation(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`,`[`n`](#classpython__op__test_1_1_python_op_test_1a6be35f7bba1a93c244c1bdb017f25095)`,`[`w`](#classpython__op__test_1_1_python_op_test_1afb6a2355c9e3f4001ed9ca3ce35f5c86)`)` |
`public def test_gradient(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`,`[`in_place`](#classpython__op__test_1_1_python_op_test_1a9ec9ce2e7808eb8e8fff04fd8df6bcd8)`,gc,dc)` |
`public def test_gradient_multiple(self,`[`inputs`](#classpython__op__test_1_1_python_op_test_1a5d168f5aed0eb9ed642c87a0e56811fb)`,gc,dc)` |

## Members

#### `public def test_multithreaded_evaluation_numba_nogil(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`,`[`n`](#classpython__op__test_1_1_python_op_test_1a6be35f7bba1a93c244c1bdb017f25095)`,`[`w`](#classpython__op__test_1_1_python_op_test_1afb6a2355c9e3f4001ed9ca3ce35f5c86)`)` {#classpython__op__test_1_1_python_op_test_1a5439951de0dbee01b6af510671b665b2}





#### `public def test_feed(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` {#classpython__op__test_1_1_python_op_test_1ac8e194a131359af4a4ddad9d4eb1c36f}





#### `public def test_exception(self)` {#classpython__op__test_1_1_python_op_test_1af504c365d0a66e70c799369cc010c4b6}





#### `public def test_feed_with_helper_function(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` {#classpython__op__test_1_1_python_op_test_1ae422a41d4c844e353d3de8a9c1c4626c}





#### `public def test_feed_with_gc(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` {#classpython__op__test_1_1_python_op_test_1aeb489f55664e37631a6a7c7e6634b9de}





#### `public def test_reshape(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` {#classpython__op__test_1_1_python_op_test_1a4d386ddb31723453122f80ea3b879c37}





#### `public def test_workspace_manipulation(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` {#classpython__op__test_1_1_python_op_test_1adc3a43cc47d635f034f8b33587489bd6}



Verify that python op can manipulate workspace directly

#### `public def test_caught_exception_doesnt_terminate(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`)` {#classpython__op__test_1_1_python_op_test_1a0c9e0d0ab00d7b60295080ad46ad4b9b}





#### `public def test_multithreaded_evaluation(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`,`[`n`](#classpython__op__test_1_1_python_op_test_1a6be35f7bba1a93c244c1bdb017f25095)`,`[`w`](#classpython__op__test_1_1_python_op_test_1afb6a2355c9e3f4001ed9ca3ce35f5c86)`)` {#classpython__op__test_1_1_python_op_test_1a069a70201c293d2d66087648f389453a}





#### `public def test_gradient(self,`[`x`](#classpython__op__test_1_1_python_op_test_1a4c4cd7d8773fd53423db45d6d8064cca)`,`[`in_place`](#classpython__op__test_1_1_python_op_test_1a9ec9ce2e7808eb8e8fff04fd8df6bcd8)`,gc,dc)` {#classpython__op__test_1_1_python_op_test_1a046480c35e2990832c3406f47f515ac8}





#### `public def test_gradient_multiple(self,`[`inputs`](#classpython__op__test_1_1_python_op_test_1a5d168f5aed0eb9ed642c87a0e56811fb)`,gc,dc)` {#classpython__op__test_1_1_python_op_test_1ae46188e2f12f2665c54dd0a993210a6f}





# namespace `queue_util` {#namespacequeue__util}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`queue_util::_QueueReader`](#classqueue__util_1_1___queue_reader)    |
`class `[`queue_util::_QueueWriter`](#classqueue__util_1_1___queue_writer)    |
`class `[`queue_util::Queue`](#classqueue__util_1_1_queue)    |
`class `[`queue_util::QueueWrapper`](#classqueue__util_1_1_queue_wrapper)    |
# class `queue_util::_QueueReader` {#classqueue__util_1_1___queue_reader}

```
class queue_util::_QueueReader
  : public dataio.Reader
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,wrapper)` |
`public def setup_ex(self,init_net,exit_net)` |
`public def read_ex(self,local_init_net,local_finish_net)` |

## Members

#### `public def __init__(self,wrapper)` {#classqueue__util_1_1___queue_reader_1ae907bedb65d45b56a47f341ac5461800}





#### `public def setup_ex(self,init_net,exit_net)` {#classqueue__util_1_1___queue_reader_1a345e52de36e26ad1fb926e0597b66a97}





#### `public def read_ex(self,local_init_net,local_finish_net)` {#classqueue__util_1_1___queue_reader_1a43131136d5681e15f8c1bf21a9dea029}





# class `queue_util::_QueueWriter` {#classqueue__util_1_1___queue_writer}

```
class queue_util::_QueueWriter
  : public dataio.Writer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,wrapper)` |
`public def setup_ex(self,init_net,exit_net)` |
`public def write_ex(self,fields,local_init_net,local_finish_net,status)` |

## Members

#### `public def __init__(self,wrapper)` {#classqueue__util_1_1___queue_writer_1a1cd8a3a6c995ae7ca5b74c868dc603db}





#### `public def setup_ex(self,init_net,exit_net)` {#classqueue__util_1_1___queue_writer_1a90e2956124f8a4ee89089e2672ae7e3c}





#### `public def write_ex(self,fields,local_init_net,local_finish_net,status)` {#classqueue__util_1_1___queue_writer_1a30248d7945aa29aeb975327412e0b43e}





# class `queue_util::Queue` {#classqueue__util_1_1_queue}

```
class queue_util::Queue
  : public queue_util.QueueWrapper
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  capacity` |
`public def __init__(self,`[`capacity`](#classqueue__util_1_1_queue_1ab0c68488fdff9c71f3dfc874a4a4aae1)`,`[`schema`](#classdataio_1_1_pipe_1aa62b258d75b2f34388cd4ccc8b7e20e3)`,name)` |
`public def setup(self,global_init_net)` |

## Members

#### `public  capacity` {#classqueue__util_1_1_queue_1ab0c68488fdff9c71f3dfc874a4a4aae1}





#### `public def __init__(self,`[`capacity`](#classqueue__util_1_1_queue_1ab0c68488fdff9c71f3dfc874a4a4aae1)`,`[`schema`](#classdataio_1_1_pipe_1aa62b258d75b2f34388cd4ccc8b7e20e3)`,name)` {#classqueue__util_1_1_queue_1af39a4692b99a15e61fdd7047b9166cd7}





#### `public def setup(self,global_init_net)` {#classqueue__util_1_1_queue_1a07218c1374ea17bfbd5cc12d82fd603e}





# class `queue_util::QueueWrapper` {#classqueue__util_1_1_queue_wrapper}

```
class queue_util::QueueWrapper
  : public dataio.Pipe
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,handler,`[`schema`](#classdataio_1_1_pipe_1aa62b258d75b2f34388cd4ccc8b7e20e3)`)` |
`public def reader(self)` |
`public def writer(self)` |
`public def queue(self)` |

## Members

#### `public def __init__(self,handler,`[`schema`](#classdataio_1_1_pipe_1aa62b258d75b2f34388cd4ccc8b7e20e3)`)` {#classqueue__util_1_1_queue_wrapper_1ab966e6cf5994f0484da0e7eb29d6e912}





#### `public def reader(self)` {#classqueue__util_1_1_queue_wrapper_1a5200b84a238ac5ca7adcacfff14f4a5a}





#### `public def writer(self)` {#classqueue__util_1_1_queue_wrapper_1a11b7f6eb8be376a812df1e88c003066e}





#### `public def queue(self)` {#classqueue__util_1_1_queue_wrapper_1acd1b00f51e5563b472ba5dce059c8e44}





# namespace `rank_loss_operator_test` {#namespacerank__loss__operator__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`rank_loss_operator_test::TestPairWiseLossOps`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops)    |
# class `rank_loss_operator_test::TestPairWiseLossOps` {#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops}

```
class rank_loss_operator_test::TestPairWiseLossOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_pair_wise_loss_predictions(self,`[`X`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1ad5d8fe2b3d54ff7e1946bdd46e56d7e9)`,`[`label`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1a248cf4e5aa8aa500c1ce9c01e79954e2)`,gc,dc)` |
`public def test_pair_wise_loss_gradient(self,`[`X`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1ad5d8fe2b3d54ff7e1946bdd46e56d7e9)`,`[`label`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1a248cf4e5aa8aa500c1ce9c01e79954e2)`,`[`dY`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1aedd10f5add536721fa73e1a20aed9838)`,gc,dc)` |

## Members

#### `public def test_pair_wise_loss_predictions(self,`[`X`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1ad5d8fe2b3d54ff7e1946bdd46e56d7e9)`,`[`label`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1a248cf4e5aa8aa500c1ce9c01e79954e2)`,gc,dc)` {#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1ab3c8c3d908d407c02d967ea4c121f580}





#### `public def test_pair_wise_loss_gradient(self,`[`X`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1ad5d8fe2b3d54ff7e1946bdd46e56d7e9)`,`[`label`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1a248cf4e5aa8aa500c1ce9c01e79954e2)`,`[`dY`](#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1aedd10f5add536721fa73e1a20aed9838)`,gc,dc)` {#classrank__loss__operator__test_1_1_test_pair_wise_loss_ops_1a3fdc395838d13191c99e616d8bc36352}





# namespace `record_queue` {#namespacerecord__queue}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`record_queue::_QueueReader`](#classrecord__queue_1_1___queue_reader)    |
`class `[`record_queue::_QueueWriter`](#classrecord__queue_1_1___queue_writer)    |
`class `[`record_queue::RecordQueue`](#classrecord__queue_1_1_record_queue)    |
# class `record_queue::_QueueReader` {#classrecord__queue_1_1___queue_reader}

```
class record_queue::_QueueReader
  : public Reader
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  blobs_queue` |
`public  name` |
`public def __init__(self,`[`blobs_queue`](#classrecord__queue_1_1___queue_reader_1ad9e69f779fd81b5db8241615be40a114)`,schema,`[`name`](#classrecord__queue_1_1___queue_reader_1af0bad5d301d2a0917f88bb345971bafe)`)` |
`public def read(self,read_net)` |

## Members

#### `public  blobs_queue` {#classrecord__queue_1_1___queue_reader_1ad9e69f779fd81b5db8241615be40a114}





#### `public  name` {#classrecord__queue_1_1___queue_reader_1af0bad5d301d2a0917f88bb345971bafe}





#### `public def __init__(self,`[`blobs_queue`](#classrecord__queue_1_1___queue_reader_1ad9e69f779fd81b5db8241615be40a114)`,schema,`[`name`](#classrecord__queue_1_1___queue_reader_1af0bad5d301d2a0917f88bb345971bafe)`)` {#classrecord__queue_1_1___queue_reader_1a6f6a5eba50628527ed6fc44cefba394c}



Don't call this directly. Instead, use dataset.reader()

#### `public def read(self,read_net)` {#classrecord__queue_1_1___queue_reader_1ab9d803e56373083aa4698f777b215d16}





# class `record_queue::_QueueWriter` {#classrecord__queue_1_1___queue_writer}

```
class record_queue::_QueueWriter
  : public Writer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  blobs_queue` |
`public  schema` |
`public def __init__(self,`[`blobs_queue`](#classrecord__queue_1_1___queue_writer_1afdb59825b6823b7a0292c1200162bbab)`,`[`schema`](#classrecord__queue_1_1___queue_writer_1a2b711a5598f2258f20019a3df1533475)`)` |
`public def write(self,writer_net,fields)` |

## Members

#### `public  blobs_queue` {#classrecord__queue_1_1___queue_writer_1afdb59825b6823b7a0292c1200162bbab}





#### `public  schema` {#classrecord__queue_1_1___queue_writer_1a2b711a5598f2258f20019a3df1533475}





#### `public def __init__(self,`[`blobs_queue`](#classrecord__queue_1_1___queue_writer_1afdb59825b6823b7a0292c1200162bbab)`,`[`schema`](#classrecord__queue_1_1___queue_writer_1a2b711a5598f2258f20019a3df1533475)`)` {#classrecord__queue_1_1___queue_writer_1aa3cc3d332050be1cf68695341e7b914e}





#### `public def write(self,writer_net,fields)` {#classrecord__queue_1_1___queue_writer_1ac15047a1122edc10fd81040994f7a220}





# class `record_queue::RecordQueue` {#classrecord__queue_1_1_record_queue}

```
class record_queue::RecordQueue
  : public object
```  



The class is used to feed data with some process from a reader into a
    queue and provider a reader interface for data fetching from the queue.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  schema` |
`public  name` |
`public  num_threads` |
`public  blobs_queue` |
`public  writer` |
`public  reader` |
`public  exit_step` |
`public def __init__(self,fields,`[`name`](#classrecord__queue_1_1_record_queue_1ad6e6e70d0b57894db8014c86c8086c59)`,capacity,enforce_unique_name,`[`num_threads`](#classrecord__queue_1_1_record_queue_1a3146bd0b011fb86d4813719bb59c630c)`)` |
`public def build(self,`[`reader`](#classrecord__queue_1_1_record_queue_1a0643248ff0e6fe5e40104c2bb07baaed)`,process)` |

## Members

#### `public  schema` {#classrecord__queue_1_1_record_queue_1a81556ad449c0220c5687dedf97b10d5b}





#### `public  name` {#classrecord__queue_1_1_record_queue_1ad6e6e70d0b57894db8014c86c8086c59}





#### `public  num_threads` {#classrecord__queue_1_1_record_queue_1a3146bd0b011fb86d4813719bb59c630c}





#### `public  blobs_queue` {#classrecord__queue_1_1_record_queue_1abc163c701720d0386ffa62ba320f8ce6}





#### `public  writer` {#classrecord__queue_1_1_record_queue_1aad0f4d3d7cc6523709533ccb69b320e5}





#### `public  reader` {#classrecord__queue_1_1_record_queue_1a0643248ff0e6fe5e40104c2bb07baaed}





#### `public  exit_step` {#classrecord__queue_1_1_record_queue_1af2da84f131d78a93eac0f336a10a346d}





#### `public def __init__(self,fields,`[`name`](#classrecord__queue_1_1_record_queue_1ad6e6e70d0b57894db8014c86c8086c59)`,capacity,enforce_unique_name,`[`num_threads`](#classrecord__queue_1_1_record_queue_1a3146bd0b011fb86d4813719bb59c630c)`)` {#classrecord__queue_1_1_record_queue_1a9e64a48b6fac3912100b46b7662dd570}





#### `public def build(self,`[`reader`](#classrecord__queue_1_1_record_queue_1a0643248ff0e6fe5e40104c2bb07baaed)`,process)` {#classrecord__queue_1_1_record_queue_1a78baf95fed9be3e9a9cd50bf67c5555b}



Build the producer_step to feed data from reader into the queue, and
return the reader interface.
Inputs:
    reader:           read data which will be stored in the queue.
    process:          preprocess data before enqueue.
Outputs:
    reader:           reader to fetch the data from the queue.
    producer_step:    the step insert the data into the queue. Should be
              run with comsume_step together.
    exit_step:        the step to close queue
    schema:           the schema for the reader.

# namespace `record_queue_test` {#namespacerecord__queue__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`record_queue_test::TestRecordQueue`](#classrecord__queue__test_1_1_test_record_queue)    |
# class `record_queue_test::TestRecordQueue` {#classrecord__queue__test_1_1_test_record_queue}

```
class record_queue_test::TestRecordQueue
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_record_queue(self)` |

## Members

#### `public def test_record_queue(self)` {#classrecord__queue__test_1_1_test_record_queue_1ab0832eb916e7fe8935162d650f225293}





# namespace `recurrent_network_test` {#namespacerecurrent__network__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`recurrent_network_test::RecurrentNetworkTest`](#classrecurrent__network__test_1_1_recurrent_network_test)    |
# class `recurrent_network_test::RecurrentNetworkTest` {#classrecurrent__network__test_1_1_recurrent_network_test}

```
class recurrent_network_test::RecurrentNetworkTest
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_lstm(self,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` |
`public def test_milstm(self,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` |
`public def lstm(self,create_lstm,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`,ref,outputs_with_grads)` |
`public def test_sum_mul(self,`[`T`](#classrecurrent__network__test_1_1_recurrent_network_test_1ad53abbf75212a8829037e82f207c7169)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` |
`public def test_mul(self,`[`T`](#classrecurrent__network__test_1_1_recurrent_network_test_1ad53abbf75212a8829037e82f207c7169)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` |
`public def simple_rnn(self,`[`T`](#classrecurrent__network__test_1_1_recurrent_network_test_1ad53abbf75212a8829037e82f207c7169)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`,model,step,input_t,output_t,output_t_prev,input_blob,initial_input_blob)` |
`public def test_lstm_unit_recurrent_network(self,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,dc,gc)` |
`public def test_lstm_with_attention(self,`[`encoder_output_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a559a9bd51a1adf5cd247585af1d891da)`,`[`encoder_output_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1a1acc417cf2aef28872666a0a2d180b28)`,`[`decoder_input_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a31f11ad7c9404425cf9cf7996097976d)`,`[`decoder_state_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1afde38ac4e05c1175c016faa44322a7f6)`,`[`batch_size`](#classrecurrent__network__test_1_1_recurrent_network_test_1a7cd32f3be35858a50add7ebc73badec1)`,gc,dc)` |
`public def test_lstm_with_recurrent_attention(self,`[`encoder_output_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a559a9bd51a1adf5cd247585af1d891da)`,`[`encoder_output_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1a1acc417cf2aef28872666a0a2d180b28)`,`[`decoder_input_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a31f11ad7c9404425cf9cf7996097976d)`,`[`decoder_state_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1afde38ac4e05c1175c016faa44322a7f6)`,`[`batch_size`](#classrecurrent__network__test_1_1_recurrent_network_test_1a7cd32f3be35858a50add7ebc73badec1)`,gc,dc)` |

## Members

#### `public def test_lstm(self,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` {#classrecurrent__network__test_1_1_recurrent_network_test_1aaa0f0ed9367c220200e1088008b0ecd4}





#### `public def test_milstm(self,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` {#classrecurrent__network__test_1_1_recurrent_network_test_1af18f79365c66c157723d42f56e5b5104}





#### `public def lstm(self,create_lstm,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`,ref,outputs_with_grads)` {#classrecurrent__network__test_1_1_recurrent_network_test_1aefaf06fdae2e7695b1edc95677b1c531}





#### `public def test_sum_mul(self,`[`T`](#classrecurrent__network__test_1_1_recurrent_network_test_1ad53abbf75212a8829037e82f207c7169)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` {#classrecurrent__network__test_1_1_recurrent_network_test_1adf962c9bd6684b2b79e84c6835a7e1f0}





#### `public def test_mul(self,`[`T`](#classrecurrent__network__test_1_1_recurrent_network_test_1ad53abbf75212a8829037e82f207c7169)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`)` {#classrecurrent__network__test_1_1_recurrent_network_test_1ade85a84a016809901e00733d755839a6}





#### `public def simple_rnn(self,`[`T`](#classrecurrent__network__test_1_1_recurrent_network_test_1ad53abbf75212a8829037e82f207c7169)`,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`,model,step,input_t,output_t,output_t_prev,input_blob,initial_input_blob)` {#classrecurrent__network__test_1_1_recurrent_network_test_1add8ca80d1a4c72b5131908daf4843d8c}





#### `public def test_lstm_unit_recurrent_network(self,`[`n`](#classrecurrent__network__test_1_1_recurrent_network_test_1a4116c367fa31486c7296ea84f646b0cc)`,`[`d`](#classrecurrent__network__test_1_1_recurrent_network_test_1ae286bf3db63be8595be72fe45fa415f0)`,`[`t`](#classrecurrent__network__test_1_1_recurrent_network_test_1adaaa7494fefb4638321dbe6c1a3f9e61)`,dc,gc)` {#classrecurrent__network__test_1_1_recurrent_network_test_1ab9c2af6d98bf43f837399e686e0aedca}





#### `public def test_lstm_with_attention(self,`[`encoder_output_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a559a9bd51a1adf5cd247585af1d891da)`,`[`encoder_output_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1a1acc417cf2aef28872666a0a2d180b28)`,`[`decoder_input_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a31f11ad7c9404425cf9cf7996097976d)`,`[`decoder_state_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1afde38ac4e05c1175c016faa44322a7f6)`,`[`batch_size`](#classrecurrent__network__test_1_1_recurrent_network_test_1a7cd32f3be35858a50add7ebc73badec1)`,gc,dc)` {#classrecurrent__network__test_1_1_recurrent_network_test_1a3b21e3ceb95e16db52c22acd2462fc42}





#### `public def test_lstm_with_recurrent_attention(self,`[`encoder_output_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a559a9bd51a1adf5cd247585af1d891da)`,`[`encoder_output_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1a1acc417cf2aef28872666a0a2d180b28)`,`[`decoder_input_length`](#classrecurrent__network__test_1_1_recurrent_network_test_1a31f11ad7c9404425cf9cf7996097976d)`,`[`decoder_state_dim`](#classrecurrent__network__test_1_1_recurrent_network_test_1afde38ac4e05c1175c016faa44322a7f6)`,`[`batch_size`](#classrecurrent__network__test_1_1_recurrent_network_test_1a7cd32f3be35858a50add7ebc73badec1)`,gc,dc)` {#classrecurrent__network__test_1_1_recurrent_network_test_1a76f5ff4e059b7a491cb867dae3a3ccf7}





# namespace `reduce_ops_test` {#namespacereduce__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`reduce_ops_test::TestReduceFrontSum`](#classreduce__ops__test_1_1_test_reduce_front_sum)    |
# class `reduce_ops_test::TestReduceFrontSum` {#classreduce__ops__test_1_1_test_reduce_front_sum}

```
class reduce_ops_test::TestReduceFrontSum
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def reduce_op_test(self,op_name,op_ref,in_data,num_reduce_dims,device)` |
`public def test_reduce_front_sum(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,gc,dc)` |
`public def test_reduce_front_mean(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,gc,dc)` |
`public def test_reduce_back_sum(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,dc,gc)` |
`public def test_reduce_back_mean(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,dc,gc)` |

## Members

#### `public def reduce_op_test(self,op_name,op_ref,in_data,num_reduce_dims,device)` {#classreduce__ops__test_1_1_test_reduce_front_sum_1a478826050ce8543d9e065a67d32f431e}





#### `public def test_reduce_front_sum(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,gc,dc)` {#classreduce__ops__test_1_1_test_reduce_front_sum_1a8846cb737b09b0e25768db859fc54ccd}





#### `public def test_reduce_front_mean(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,gc,dc)` {#classreduce__ops__test_1_1_test_reduce_front_sum_1a3eeed84beae2f63f9d86ce5e068d1dc7}





#### `public def test_reduce_back_sum(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,dc,gc)` {#classreduce__ops__test_1_1_test_reduce_front_sum_1a63af876bf75ae58b1c5e66bedf6ed574}





#### `public def test_reduce_back_mean(self,`[`num_reduce_dim`](#classreduce__ops__test_1_1_test_reduce_front_sum_1a6fe4a3c0598b6d9ea2c1efcbc11f58d6)`,dc,gc)` {#classreduce__ops__test_1_1_test_reduce_front_sum_1af02b7c8912a0e3f9e994e25e162359e9}





# namespace `relu_op_test` {#namespacerelu__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`relu_op_test::TestRelu`](#classrelu__op__test_1_1_test_relu)    |
# class `relu_op_test::TestRelu` {#classrelu__op__test_1_1_test_relu}

```
class relu_op_test::TestRelu
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_relu(self,`[`X`](#classrelu__op__test_1_1_test_relu_1a27b8256f1e511435bf1008cd5acaed7d)`,gc,dc)` |

## Members

#### `public def test_relu(self,`[`X`](#classrelu__op__test_1_1_test_relu_1a27b8256f1e511435bf1008cd5acaed7d)`,gc,dc)` {#classrelu__op__test_1_1_test_relu_1ada3a5dd46adddc608daedac586b70263}





# namespace `reshape_ops_test` {#namespacereshape__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`reshape_ops_test::TestLengthsToShapeOps`](#classreshape__ops__test_1_1_test_lengths_to_shape_ops)    |
# class `reshape_ops_test::TestLengthsToShapeOps` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops}

```
class reshape_ops_test::TestLengthsToShapeOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_lengths_to_shape_ops(self)` |
`public def test_reshape_ops(self)` |
`public def test_basic_reshape(self)` |
`public def test_missing_dim(self)` |
`public def test_in_place(self)` |
`public def test_zero_dim(self)` |
`public def test_zero_dim_and_missing_dim(self)` |
`public def test_backprop(self)` |
`public def test_input_shape_changes(self)` |

## Members

#### `public def test_lengths_to_shape_ops(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1a0399711a4811d8331f6ce2517df03b20}





#### `public def test_reshape_ops(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1ad77bf620e51ed334c064e16fcbbaa6c6}





#### `public def test_basic_reshape(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1abeac16bb1f8634c67d020b1a96ce0d6e}





#### `public def test_missing_dim(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1abffe0eb38e03cb0eb446c49cb41f2c91}





#### `public def test_in_place(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1a33726539402d6ea9f1e72946a4dcc9d9}





#### `public def test_zero_dim(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1a0b1aeef719985867c7be7cd3e20f38d0}





#### `public def test_zero_dim_and_missing_dim(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1aad874a6f155d38d89080ec0d7996e6a0}





#### `public def test_backprop(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1ae2f55fd51c92bcbf18f5c790be5bf10e}





#### `public def test_input_shape_changes(self)` {#classreshape__ops__test_1_1_test_lengths_to_shape_ops_1a3c24b88e1789411f21fb4555f80f5694}





# namespace `resize_op_test` {#namespaceresize__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`resize_op_test::TestResize`](#classresize__op__test_1_1_test_resize)    |
# class `resize_op_test::TestResize` {#classresize__op__test_1_1_test_resize}

```
class resize_op_test::TestResize
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_nearest(self,`[`width_scale`](#classresize__op__test_1_1_test_resize_1a5732cec8dab505b29b53579000f91c17)`,`[`height_scale`](#classresize__op__test_1_1_test_resize_1a70b9c337e0408e7a275e70ab783f4a06)`,`[`size_w`](#classresize__op__test_1_1_test_resize_1afac9acafbc2066c66b1669a2752e5cd8)`,`[`size_h`](#classresize__op__test_1_1_test_resize_1a1b725d3a50f385fd0a8e808a72d220ed)`,`[`input_channels`](#classresize__op__test_1_1_test_resize_1ad0d7213bc125e5cca3b79b9fda19652f)`,`[`batch_size`](#classresize__op__test_1_1_test_resize_1a70b788417bcb23196c44a9ad01efebd1)`,gc,dc)` |

## Members

#### `public def test_nearest(self,`[`width_scale`](#classresize__op__test_1_1_test_resize_1a5732cec8dab505b29b53579000f91c17)`,`[`height_scale`](#classresize__op__test_1_1_test_resize_1a70b9c337e0408e7a275e70ab783f4a06)`,`[`size_w`](#classresize__op__test_1_1_test_resize_1afac9acafbc2066c66b1669a2752e5cd8)`,`[`size_h`](#classresize__op__test_1_1_test_resize_1a1b725d3a50f385fd0a8e808a72d220ed)`,`[`input_channels`](#classresize__op__test_1_1_test_resize_1ad0d7213bc125e5cca3b79b9fda19652f)`,`[`batch_size`](#classresize__op__test_1_1_test_resize_1a70b788417bcb23196c44a9ad01efebd1)`,gc,dc)` {#classresize__op__test_1_1_test_resize_1aa4f88393413d06c66d0ad6b7720c6513}





# namespace `resnet` {#namespaceresnet}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`resnet::ResNetBuilder`](#classresnet_1_1_res_net_builder)    |
# class `resnet::ResNetBuilder` {#classresnet_1_1_res_net_builder}




Helper class for constructing residual blocks.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  model` |
`public  comp_count` |
`public  comp_idx` |
`public  prev_blob` |
`public  is_test` |
`public  spatial_bn_mom` |
`public  no_bias` |
`public def __init__(self,`[`model`](#classresnet_1_1_res_net_builder_1a4e71162a7b964002df9f651da210d22a)`,`[`prev_blob`](#classresnet_1_1_res_net_builder_1add333febe7945bc04885315ebdedbaf2)`,`[`no_bias`](#classresnet_1_1_res_net_builder_1a4e183d0b3bc221915daae77ddc67f5f4)`,`[`is_test`](#classresnet_1_1_res_net_builder_1a0d2587be99f08ccbdf75f146ed6e15b8)`,`[`spatial_bn_mom`](#classresnet_1_1_res_net_builder_1af30ff2cbe7a60f99dd284bb31b0e72c5)`)` |
`public def add_conv(self,in_filters,out_filters,kernel,stride,pad)` |
`public def add_relu(self)` |
`public def add_spatial_bn(self,num_filters)` |
`public def add_bottleneck(self,input_filters,base_filters,output_filters,down_sampling,spatial_batch_norm)` |
`public def add_simple_block(self,input_filters,num_filters,down_sampling,spatial_batch_norm)` |

## Members

#### `public  model` {#classresnet_1_1_res_net_builder_1a4e71162a7b964002df9f651da210d22a}





#### `public  comp_count` {#classresnet_1_1_res_net_builder_1ad96e84eed837368b7fa20443e10272e9}





#### `public  comp_idx` {#classresnet_1_1_res_net_builder_1a5a2535316a1f0d6ad1e6a468b4be6e9b}





#### `public  prev_blob` {#classresnet_1_1_res_net_builder_1add333febe7945bc04885315ebdedbaf2}





#### `public  is_test` {#classresnet_1_1_res_net_builder_1a0d2587be99f08ccbdf75f146ed6e15b8}





#### `public  spatial_bn_mom` {#classresnet_1_1_res_net_builder_1af30ff2cbe7a60f99dd284bb31b0e72c5}





#### `public  no_bias` {#classresnet_1_1_res_net_builder_1a4e183d0b3bc221915daae77ddc67f5f4}





#### `public def __init__(self,`[`model`](#classresnet_1_1_res_net_builder_1a4e71162a7b964002df9f651da210d22a)`,`[`prev_blob`](#classresnet_1_1_res_net_builder_1add333febe7945bc04885315ebdedbaf2)`,`[`no_bias`](#classresnet_1_1_res_net_builder_1a4e183d0b3bc221915daae77ddc67f5f4)`,`[`is_test`](#classresnet_1_1_res_net_builder_1a0d2587be99f08ccbdf75f146ed6e15b8)`,`[`spatial_bn_mom`](#classresnet_1_1_res_net_builder_1af30ff2cbe7a60f99dd284bb31b0e72c5)`)` {#classresnet_1_1_res_net_builder_1aa163a356d8006b4da010ca4dfd3707b4}





#### `public def add_conv(self,in_filters,out_filters,kernel,stride,pad)` {#classresnet_1_1_res_net_builder_1a600b49bca29751785118f9f85b6a12cf}





#### `public def add_relu(self)` {#classresnet_1_1_res_net_builder_1a9ae3da12da3b0239c8ff0a2c495e9177}





#### `public def add_spatial_bn(self,num_filters)` {#classresnet_1_1_res_net_builder_1a0312390896562b950569fdaa9ba7a86f}





#### `public def add_bottleneck(self,input_filters,base_filters,output_filters,down_sampling,spatial_batch_norm)` {#classresnet_1_1_res_net_builder_1a8197a53b228cf56f09cbebb7b42549e2}





#### `public def add_simple_block(self,input_filters,num_filters,down_sampling,spatial_batch_norm)` {#classresnet_1_1_res_net_builder_1a36d400f9d432ec92008510726696fe6b}





# namespace `schema` {#namespaceschema}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`schema::_SchemaNode`](#classschema_1_1___schema_node)    |
`class `[`schema::Field`](#classschema_1_1_field)    |
`class `[`schema::List`](#classschema_1_1_list)    |
`class `[`schema::Metadata`](#classschema_1_1_metadata)    |
`class `[`schema::Scalar`](#classschema_1_1_scalar)    |
`class `[`schema::Struct`](#classschema_1_1_struct)    |
# class `schema::_SchemaNode` {#classschema_1_1___schema_node}

```
class schema::_SchemaNode
  : public object
```  



This is a private class used to represent a Schema Node

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  name` |
`public  children` |
`public  type_str` |
`public  field` |
`public  col_blob` |
`public def __init__(self,`[`name`](#classschema_1_1___schema_node_1ae599b93eedb17eb33058dd51e7875129)`,`[`type_str`](#classschema_1_1___schema_node_1af3ab7d3ee2ae8b784a39a5fe9b0f10fe)`)` |
`public def add_child(self,`[`name`](#classschema_1_1___schema_node_1ae599b93eedb17eb33058dd51e7875129)`,`[`type_str`](#classschema_1_1___schema_node_1af3ab7d3ee2ae8b784a39a5fe9b0f10fe)`)` |
`public def get_field(self)` |
`public def print_recursively(self)` |

## Members

#### `public  name` {#classschema_1_1___schema_node_1ae599b93eedb17eb33058dd51e7875129}





#### `public  children` {#classschema_1_1___schema_node_1af6e396195f901cf0e731b315856c71a9}





#### `public  type_str` {#classschema_1_1___schema_node_1af3ab7d3ee2ae8b784a39a5fe9b0f10fe}





#### `public  field` {#classschema_1_1___schema_node_1a64904177907b1df2260ec8dc0562f381}





#### `public  col_blob` {#classschema_1_1___schema_node_1a63965636812c2bea1d572774a6d142a3}





#### `public def __init__(self,`[`name`](#classschema_1_1___schema_node_1ae599b93eedb17eb33058dd51e7875129)`,`[`type_str`](#classschema_1_1___schema_node_1af3ab7d3ee2ae8b784a39a5fe9b0f10fe)`)` {#classschema_1_1___schema_node_1a41a6421289c7260e64f709f0e614a0e5}





#### `public def add_child(self,`[`name`](#classschema_1_1___schema_node_1ae599b93eedb17eb33058dd51e7875129)`,`[`type_str`](#classschema_1_1___schema_node_1af3ab7d3ee2ae8b784a39a5fe9b0f10fe)`)` {#classschema_1_1___schema_node_1a865a1705ffc0f5f09154982a6fa52bee}





#### `public def get_field(self)` {#classschema_1_1___schema_node_1a9330f01eb1d36d58959ae8c8b82b1068}





#### `public def print_recursively(self)` {#classschema_1_1___schema_node_1ae8f29063ad23301a572a21610605131e}





# class `schema::Field` {#classschema_1_1_field}

```
class schema::Field
  : public object
```  



Represents an abstract field type in a dataset.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,children)` |
`public def clone_schema(self)` |
`public def field_names(self)` |
`public def field_types(self)` |
`public def field_metadata(self)` |
`public def field_blobs(self)` |
`public def all_scalars(self)` |
`public def has_blobs(self)` |
`public def clone(self,keep_blobs)` |
`public def slice(self)` |
`public def __eq__(self,other)` |

## Members

#### `public def __init__(self,children)` {#classschema_1_1_field_1a82990f45929dad788cb6a5f654ad2d93}



Derived classes must call this after their initialization.

#### `public def clone_schema(self)` {#classschema_1_1_field_1ab5de5e02c65a2102eea4c74fb32b7fc1}





#### `public def field_names(self)` {#classschema_1_1_field_1a6b7a67ab9bb9f02dfc1877e6b02e229f}



Return the children field names for this field.

#### `public def field_types(self)` {#classschema_1_1_field_1a85561b85e58a598fce0ec338d433d487}



Return the numpy.dtype for each of the children fields.

#### `public def field_metadata(self)` {#classschema_1_1_field_1ad1d17307a4764fff882576fc89de9f31}



Return the Metadata for each of the children fields.

#### `public def field_blobs(self)` {#classschema_1_1_field_1ae6b35ada97c3cef1d9cf5dab918447a4}



Return the list of blobs with contents for this Field.
Values can either be all numpy.ndarray or BlobReference.
If any of the fields doens't have a blob, throws.

#### `public def all_scalars(self)` {#classschema_1_1_field_1aa964d4f1a84c5db9d3364b83abf10de6}



Return the list of all Scalar instances in the Field.
The order is the same as for field_names() or field_blobs()

#### `public def has_blobs(self)` {#classschema_1_1_field_1a3ed8f7784f8879177dc94055f935b10c}



Return True if every scalar of this field has blobs.

#### `public def clone(self,keep_blobs)` {#classschema_1_1_field_1a4b13f579a08faffe21326039a3ff824a}



Clone this Field along with its children.

#### `public def slice(self)` {#classschema_1_1_field_1a98a892aaf9cda03689ce359862dfac5f}



Returns a slice representing the range of field ids that belong to
this field. This slice can be used to index a list of fields.

E.g.:

>>> s = Struct(
>>>     ('a', Scalar()),
>>>     ('b', Struct(
>>>         ('b1', Scalar()),
>>>         ('b2', Scalar()),
>>>     )),
>>>     ('c', Scalar()),
>>> )
>>> field_data = ['da', 'db1', 'db2', 'dc']
>>> field_data[s.b.split()]
['db1', 'db2']

#### `public def __eq__(self,other)` {#classschema_1_1_field_1a5c97cebe25a5c735868f69b0eb089a00}



Equivalance of two schemas

# class `schema::List` {#classschema_1_1_list}

```
class schema::List
  : public schema.Field
```  



Represents a variable-length list.

Values of a list can also be complex fields such as Lists and Structs.
In addition to the fields exposed by its `values` field, a List exposes an
additional `lengths` field, which will contain the size of each list under
the parent domain.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  lengths` |
`public def __init__(self,values,lengths_blob)` |
`public def field_names(self)` |
`public def field_types(self)` |
`public def field_metadata(self)` |
`public def field_blobs(self)` |
`public def all_scalars(self)` |
`public def has_blobs(self)` |
`public def clone(self,keep_blobs)` |
`public def __getattr__(self,item)` |

## Members

#### `public  lengths` {#classschema_1_1_list_1aba966eb7345f87a82d9232cf4816a5b6}





#### `public def __init__(self,values,lengths_blob)` {#classschema_1_1_list_1ace5fcaa64872f7bacb225d996b1926af}





#### `public def field_names(self)` {#classschema_1_1_list_1a3dde023ae5e7f92121759b160ecdf8a6}





#### `public def field_types(self)` {#classschema_1_1_list_1a2b076831c2d6731f6d5e5e8c23d374f6}





#### `public def field_metadata(self)` {#classschema_1_1_list_1a62781cf06bf2e16361568641943c80ef}





#### `public def field_blobs(self)` {#classschema_1_1_list_1acb350802063ad5bbbdbc2e66a75cb3ec}





#### `public def all_scalars(self)` {#classschema_1_1_list_1a70e8597d76b4588a6d5934f39b3c0ed8}





#### `public def has_blobs(self)` {#classschema_1_1_list_1aa64d60089ed66840059bd75753abb745}





#### `public def clone(self,keep_blobs)` {#classschema_1_1_list_1a2e7ff3dec9e30748fd2a14edb09da8b0}





#### `public def __getattr__(self,item)` {#classschema_1_1_list_1a0a37655dbf34acb8842dfe1e38a272c1}



If the value of this list is a struct,
allow to instrospect directly into its fields.

# class `schema::Metadata` {#classschema_1_1_metadata}

```
class schema::Metadata
  : public namedtuple
  : public categorical_limit
  : public expected_value
  : public feature_specs
```  



Represents additional information associated with a scalar in schema.

`categorical_limit` - for fields of integral type that are guaranteed to be
non-negative it specifies the maximum possible value plus one. It's often
used as a size of an embedding table.

`expected_value` - anticipated average value of elements in the field.
Usually makes sense for length fields of lists.

`feature_specs` - information about the features that contained in this
field. For example if field have more then 1 feature it can have list of
feature names contained in this field.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `schema::Scalar` {#classschema_1_1_scalar}

```
class schema::Scalar
  : public schema.Field
```  



Represents a typed scalar or tensor of fixed shape.

A Scalar is a leaf in a schema tree, translating to exactly one tensor in
the dataset's underlying storage.

Usually, the tensor storing the actual values of this field is a 1D tensor,
representing a series of values in its domain. It is possible however to
have higher rank values stored as a Scalar, as long as all entries have
the same shape.

E.g.:

    Scalar(np.float64)

        Scalar field of type float32. Caffe2 will expect readers and
        datasets to expose it as a 1D tensor of doubles (vector), where
        the size of the vector is determined by this fields' domain.

    Scalar((np.int32, 5))

        Tensor field of type int32. Caffe2 will expect readers and
        datasets to implement it as a 2D tensor (matrix) of shape (L, 5),
        where L is determined by this fields' domain.

    Scalar((str, (10, 20)))

        Tensor field of type str. Caffe2 will expect readers and
        datasets to implement it as a 3D tensor of shape (L, 10, 20),
        where L is determined by this fields' domain.

If the field type is unknown at construction time, call Scalar(), that will
default to np.void as its dtype.

It is an error to pass a structured dtype to Scalar, since it would contain
more than one field. Instead, use from_dtype, which will construct
a nested `Struct` field reflecting the given dtype's structure.

A Scalar can also contain a blob, which represents the value of this
Scalar. A blob can be either a numpy.ndarray, in which case it contain the
actual contents of the Scalar, or a BlobReference, which represents a
blob living in a caffe2 Workspace. If blob of different types are passed,
a conversion to numpy.ndarray is attempted.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  dtype` |
`public def __init__(self,`[`dtype`](#classschema_1_1_scalar_1a8ac15b9ed9564e7b1123715c1fde45da)`,blob,`[`metadata`](#classschema_1_1_scalar_1a014b792748e1e54494ce799c3fd7d980)`)` |
`public def field_names(self)` |
`public def field_type(self)` |
`public def field_types(self)` |
`public def field_metadata(self)` |
`public def has_blobs(self)` |
`public def field_blobs(self)` |
`public def all_scalars(self)` |
`public def clone(self,keep_blobs)` |
`public def get(self)` |
`public def __call__(self)` |
`public def metadata(self)` |
`public def set_metadata(self,value)` |
`public def set_value(self,blob)` |
`public def set(self,`[`dtype`](#classschema_1_1_scalar_1a8ac15b9ed9564e7b1123715c1fde45da)`,blob,`[`metadata`](#classschema_1_1_scalar_1a014b792748e1e54494ce799c3fd7d980)`)` |
`public def set_type(self,`[`dtype`](#classschema_1_1_scalar_1a8ac15b9ed9564e7b1123715c1fde45da)`)` |
`public def id(self)` |

## Members

#### `public  dtype` {#classschema_1_1_scalar_1a8ac15b9ed9564e7b1123715c1fde45da}





#### `public def __init__(self,`[`dtype`](#classschema_1_1_scalar_1a8ac15b9ed9564e7b1123715c1fde45da)`,blob,`[`metadata`](#classschema_1_1_scalar_1a014b792748e1e54494ce799c3fd7d980)`)` {#classschema_1_1_scalar_1af0888378aae4b055a0d24c02aeabdb01}





#### `public def field_names(self)` {#classschema_1_1_scalar_1a8b4cde40daceec88c895773f09124e70}





#### `public def field_type(self)` {#classschema_1_1_scalar_1a7b627f31965160b37bbe3819c6ea8182}





#### `public def field_types(self)` {#classschema_1_1_scalar_1afad6ae7b051c665b4595dd69a5b3f034}





#### `public def field_metadata(self)` {#classschema_1_1_scalar_1a7d3ad8eb5d8c92acdcd797d6647db599}





#### `public def has_blobs(self)` {#classschema_1_1_scalar_1a5e354002bef096e01aeeb6d2bef64c92}





#### `public def field_blobs(self)` {#classschema_1_1_scalar_1a50a77bb6a69aaa210e11f5b3543def0c}





#### `public def all_scalars(self)` {#classschema_1_1_scalar_1a2511cb33a1734df1e3c531113ea88c01}





#### `public def clone(self,keep_blobs)` {#classschema_1_1_scalar_1acc7cee1d8f4190301a0fb1faab8077f8}





#### `public def get(self)` {#classschema_1_1_scalar_1a790d0e4ea9edd8b32e2d8b8977b85e5e}



Gets the current blob of this Scalar field.

#### `public def __call__(self)` {#classschema_1_1_scalar_1a9cc8baf595dc665b480d29952a1c22e5}



Shortcut for self.get()

#### `public def metadata(self)` {#classschema_1_1_scalar_1a014b792748e1e54494ce799c3fd7d980}





#### `public def set_metadata(self,value)` {#classschema_1_1_scalar_1ab89570b9283ecaa64c21d04bd5e28fea}





#### `public def set_value(self,blob)` {#classschema_1_1_scalar_1a13411f653e752e0e3dc05a43a41df215}



Sets only the blob field still validating the existing dtype

#### `public def set(self,`[`dtype`](#classschema_1_1_scalar_1a8ac15b9ed9564e7b1123715c1fde45da)`,blob,`[`metadata`](#classschema_1_1_scalar_1a014b792748e1e54494ce799c3fd7d980)`)` {#classschema_1_1_scalar_1ad96c7067e2f1e916bf57e44a0cbf212b}



Set the type and/or blob of this scalar. See __init__ for details.

Args:
    dtype: can be any numpy type. If not provided and `blob` is
   provided, it will be inferred. If no argument is provided,
   this Scalar will be of type np.void.
    blob:  if provided, can be either a BlobReference or a
   numpy.ndarray. If a value of different type is passed,
   a conversion to numpy.ndarray is attempted. Strings aren't
   accepted, since they can be ambiguous. If you want to pass
   a string, to either BlobReference(blob) or np.array(blob).
    metadata: optional instance of Metadata, if provided overrides
      the metadata information of the scalar

#### `public def set_type(self,`[`dtype`](#classschema_1_1_scalar_1a8ac15b9ed9564e7b1123715c1fde45da)`)` {#classschema_1_1_scalar_1a67ba4efb800129933b6a2e780e67e355}





#### `public def id(self)` {#classschema_1_1_scalar_1a0b5e8835afb0e525500ce39b06ba4e11}



Return the zero-indexed position of this scalar field in its schema.
Used in order to index into the field_blob list returned by readers or
accepted by writers.

# class `schema::Struct` {#classschema_1_1_struct}

```
class schema::Struct
  : public schema.Field
```  



Represents a named list of fields sharing the same domain.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  fields` |
`public def __init__(self,`[`fields`](#classschema_1_1_struct_1a00d431e0e10ef8a6eb60e38c1f216a80)`)` |
`public def get_children(self)` |
`public def field_names(self)` |
`public def field_types(self)` |
`public def field_metadata(self)` |
`public def field_blobs(self)` |
`public def all_scalars(self)` |
`public def has_blobs(self)` |
`public def clone(self,keep_blobs)` |
`public def __contains__(self,item)` |
`public def __len__(self)` |
`public def __getitem__(self,item)` |
`public def __getattr__(self,item)` |
`public def __add__(self,other)` |

## Members

#### `public  fields` {#classschema_1_1_struct_1a00d431e0e10ef8a6eb60e38c1f216a80}





#### `public def __init__(self,`[`fields`](#classschema_1_1_struct_1a00d431e0e10ef8a6eb60e38c1f216a80)`)` {#classschema_1_1_struct_1aa87d64972901d0bd715a7413b327988d}



fields is a list of tuples in format of (name, field). The name is
a string of nested name, e.g., `a`, `a:b`, `a:b:c`. For example

Struct(
  ('a', Scalar()),
  ('b:c', Scalar()),
  ('b:d:e', Scalar()),
  ('b', Struct(
    ('f', Scalar()),
  )),
)

is equal to

Struct(
  ('a', Scalar()),
  ('b', Struct(
    ('c', Scalar()),
    ('d', Struct(('e', Scalar()))),
    ('f', Scalar()),
  )),
)

#### `public def get_children(self)` {#classschema_1_1_struct_1ae5a99aab9dbd339335e8f778205b6f9c}





#### `public def field_names(self)` {#classschema_1_1_struct_1aa0eb41a204f18243ed072e19d1d2dc7f}





#### `public def field_types(self)` {#classschema_1_1_struct_1a660cc0372a2b6d708cc79e93e6d7c91d}





#### `public def field_metadata(self)` {#classschema_1_1_struct_1a18cfae15c892a6dc8ace34970bc848da}





#### `public def field_blobs(self)` {#classschema_1_1_struct_1ad31a559d98c8f08c831a4560fa34c128}





#### `public def all_scalars(self)` {#classschema_1_1_struct_1a617b501e8897aedca949e9daec14c4e6}





#### `public def has_blobs(self)` {#classschema_1_1_struct_1a4394efe59e361ad2450dad2dbce11dda}





#### `public def clone(self,keep_blobs)` {#classschema_1_1_struct_1afc88eac3b8243fb55f43c0af501821e1}





#### `public def __contains__(self,item)` {#classschema_1_1_struct_1a172dc892c72187f865bd3fbf5af487e0}





#### `public def __len__(self)` {#classschema_1_1_struct_1af32bbc0721321e7feb7d3f769cfd4f27}





#### `public def __getitem__(self,item)` {#classschema_1_1_struct_1accd55195514a1f9716a8f12fa08c62c0}



item can be a tuple or list of ints or strings, or a single
int or string. String item is a nested field name, e.g., "a", "a:b",
"a:b:c". Int item is the index of a field at the first level of the
Struct.

#### `public def __getattr__(self,item)` {#classschema_1_1_struct_1a476c1494c4eedf76262b5685061d1b72}





#### `public def __add__(self,other)` {#classschema_1_1_struct_1acf5dd93cbf83e3c8b812f0c5974f4333}



Allows to merge fields of two schema.Struct using '+' operator.
If two Struct have common field names, the merge is conducted
recursively. Here are examples:

Example 1
s1 = Struct(('a', Scalar()))
s2 = Struct(('b', Scalar()))
s1 + s2 == Struct(
    ('a', Scalar()),
    ('b', Scalar()),
)

Example 2
s1 = Struct(
    ('a', Scalar()),
    ('b', Struct(('c', Scalar()))),
)
s2 = Struct(('b', Struct(('d', Scalar()))))
s1 + s2 == Struct(
    ('a', Scalar()),
    ('b', Struct(
('c', Scalar()),
('d', Scalar()),
    )),
)

# namespace `schema_test` {#namespaceschema__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`schema_test::TestDB`](#classschema__test_1_1_test_d_b)    |
# class `schema_test::TestDB` {#classschema__test_1_1_test_d_b}

```
class schema_test::TestDB
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testPicklable(self)` |
`public def testNormalizeField(self)` |
`public def testTuple(self)` |
`public def testRawTuple(self)` |
`public def testStructIndexing(self)` |
`public def testPreservesMetadata(self)` |
`public def testDupField(self)` |
`public def testPreservesEmptyFields(self)` |
`public def testStructAddition(self)` |
`public def testStructNestedAddition(self)` |
`public def testGetFieldByNestedName(self)` |
`public def testAddFieldByNestedName(self)` |
`public def testContains(self)` |

## Members

#### `public def testPicklable(self)` {#classschema__test_1_1_test_d_b_1ae70745af28f0d4f62c86e007a6b46a8f}





#### `public def testNormalizeField(self)` {#classschema__test_1_1_test_d_b_1a98ed61d12edf95117f06128061661c1a}





#### `public def testTuple(self)` {#classschema__test_1_1_test_d_b_1a1b4548865721fe7e59473b17680a3712}





#### `public def testRawTuple(self)` {#classschema__test_1_1_test_d_b_1ad522baa20ca763b4320da00b42ae3e13}





#### `public def testStructIndexing(self)` {#classschema__test_1_1_test_d_b_1adbefbb10f9c4ad582603bf2c1299b1f4}





#### `public def testPreservesMetadata(self)` {#classschema__test_1_1_test_d_b_1a2e44d4da802497039ebb6ca446105a5d}





#### `public def testDupField(self)` {#classschema__test_1_1_test_d_b_1a3b7243f88b92b0ad61e9935516a31006}





#### `public def testPreservesEmptyFields(self)` {#classschema__test_1_1_test_d_b_1affc6884761dc7344d83ec630a398fc23}





#### `public def testStructAddition(self)` {#classschema__test_1_1_test_d_b_1ad17c8bbc6b38724e50e16bdc786851e1}





#### `public def testStructNestedAddition(self)` {#classschema__test_1_1_test_d_b_1af0ad1d1b621fe9eba17e31a66c47e1f0}





#### `public def testGetFieldByNestedName(self)` {#classschema__test_1_1_test_d_b_1a17096bd88756576eba83ba9e8dea8b0a}





#### `public def testAddFieldByNestedName(self)` {#classschema__test_1_1_test_d_b_1a9eed95149b91e0bebeb186beb4e8c20f}





#### `public def testContains(self)` {#classschema__test_1_1_test_d_b_1afdd4605646fd29eaa76e6f486eb289e2}





# namespace `scope_test` {#namespacescope__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`scope_test::TestScope`](#classscope__test_1_1_test_scope)    |
# class `scope_test::TestScope` {#classscope__test_1_1_test_scope}

```
class scope_test::TestScope
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testNamescopeBasic(self)` |
`public def testDevicescopeBasic(self)` |
`public def testMultiThreaded(self)` |

## Members

#### `public def testNamescopeBasic(self)` {#classscope__test_1_1_test_scope_1a268d3be58ee05b95b5904271b9702f1f}





#### `public def testDevicescopeBasic(self)` {#classscope__test_1_1_test_scope_1a431c37440ed5a8d57008d4b30c4a2e05}





#### `public def testMultiThreaded(self)` {#classscope__test_1_1_test_scope_1a3395ad921d8bf61826e8ae95b3111142}



Test that name/device scope are properly local to the thread
and don't interfere

# namespace `segment_ops_test` {#namespacesegment__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`segment_ops_test::LengthsTester`](#classsegment__ops__test_1_1_lengths_tester)    |
`class `[`segment_ops_test::SegmentsTester`](#classsegment__ops__test_1_1_segments_tester)    |
`class `[`segment_ops_test::TesterBase`](#classsegment__ops__test_1_1_tester_base)    |
`class `[`segment_ops_test::TestSegmentOps`](#classsegment__ops__test_1_1_test_segment_ops)    |
# class `segment_ops_test::LengthsTester` {#classsegment__ops__test_1_1_lengths_tester}

```
class segment_ops_test::LengthsTester
  : public segment_ops_test.TesterBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def split(self,data,lengths,indices)` |
`public def unsplit(self,extra_shape,inputs,lengths)` |

## Members

#### `public def split(self,data,lengths,indices)` {#classsegment__ops__test_1_1_lengths_tester_1a40f5e31da5b38cbfc9b38a0c424c70ed}





#### `public def unsplit(self,extra_shape,inputs,lengths)` {#classsegment__ops__test_1_1_lengths_tester_1a7528e35626ce35de44ae71fed7f7e76b}





# class `segment_ops_test::SegmentsTester` {#classsegment__ops__test_1_1_segments_tester}

```
class segment_ops_test::SegmentsTester
  : public segment_ops_test.TesterBase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def split(self,data,segment_ids,indices)` |
`public def unsplit(self,extra_shape,inputs,segment_ids)` |

## Members

#### `public def split(self,data,segment_ids,indices)` {#classsegment__ops__test_1_1_segments_tester_1a97bd1cf0b92bd69a91d6e61e5de9618b}



Given:
  data[M1 x M2 x ... x Md]
          the input data
  indices[N]      the index of each entry of segment_ids into data,
          where 0 <= index[i] < M1,
          with default indices=[0,1,...N]
  segment_ids[N]  the segment_id for each entry of indices,

returns K outputs, each one containing data entries corresponding
to one of the segments present in `segment_ids`.

#### `public def unsplit(self,extra_shape,inputs,segment_ids)` {#classsegment__ops__test_1_1_segments_tester_1aeb73cbfcd6be66d3bcfb07afd18be9c2}



Inverse operation to `split`, with indices=None

# class `segment_ops_test::TesterBase` {#classsegment__ops__test_1_1_tester_base}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def segment_reduce_op(self,data,segment_ids,reducer,indices)` |
`public def segment_reduce_grad_op(self,data,segment_ids,reducer_grad,grad_out,output,indices)` |

## Members

#### `public def segment_reduce_op(self,data,segment_ids,reducer,indices)` {#classsegment__ops__test_1_1_tester_base_1a300f8b0362337b50000a6517090e7658}





#### `public def segment_reduce_grad_op(self,data,segment_ids,reducer_grad,grad_out,output,indices)` {#classsegment__ops__test_1_1_tester_base_1a25be91eedc5bae00a23c7551ae7f9134}





# class `segment_ops_test::TestSegmentOps` {#classsegment__ops__test_1_1_test_segment_ops}

```
class segment_ops_test::TestSegmentOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_sorted_segment_ops(self)` |
`public def test_unsorted_segment_ops(self)` |
`public def test_sparse_sorted_segment_ops(self)` |
`public def test_sparse_unsorted_segment_ops(self)` |
`public def test_lengths_ops(self)` |
`public def test_sparse_lengths_ops(self)` |

## Members

#### `public def test_sorted_segment_ops(self)` {#classsegment__ops__test_1_1_test_segment_ops_1a266b34a1909e708f4505615328f890df}





#### `public def test_unsorted_segment_ops(self)` {#classsegment__ops__test_1_1_test_segment_ops_1a179b53316fae40a58ea0feb588af0e4a}





#### `public def test_sparse_sorted_segment_ops(self)` {#classsegment__ops__test_1_1_test_segment_ops_1a9109d1281aa7cc64baf2ba17391d6157}





#### `public def test_sparse_unsorted_segment_ops(self)` {#classsegment__ops__test_1_1_test_segment_ops_1aea7b266dbfb262f3040586e5cde75876}





#### `public def test_lengths_ops(self)` {#classsegment__ops__test_1_1_test_segment_ops_1a64d6953785aaf878c94f7935ee1c010c}





#### `public def test_sparse_lengths_ops(self)` {#classsegment__ops__test_1_1_test_segment_ops_1aaf99713f005860c66efc8a07afaa30bd}





# namespace `seq2seq` {#namespaceseq2seq}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`seq2seq::Seq2SeqModelCaffe2`](#classseq2seq_1_1_seq2_seq_model_caffe2)    |
# class `seq2seq::Seq2SeqModelCaffe2` {#classseq2seq_1_1_seq2_seq_model_caffe2}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  encoder_inputs` |
`public  encoder_lengths` |
`public  decoder_inputs` |
`public  decoder_lengths` |
`public  targets` |
`public  target_weights` |
`public  learning_rate` |
`public  global_step` |
`public  start_time` |
`public  encoder_type` |
`public  total_loss_scalar` |
`public  forward_net` |
`public  model` |
`public  model_params` |
`public  encoder_params` |
`public  source_vocab_size` |
`public  target_vocab_size` |
`public  num_gpus` |
`public  num_cpus` |
`public  batch_size` |
`public def output_projection(self,`[`model`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a7d30d44ddfffb4eb7053c09360d9d26d)`,decoder_outputs,decoder_output_size,`[`target_vocab_size`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a6a845f7374dafea3d21e8b8bbf4af095)`,decoder_softmax_size)` |
`public def __init__(self,`[`model_params`](#classseq2seq_1_1_seq2_seq_model_caffe2_1ad3de9310c7b6d28fc99ede130fea77e3)`,`[`source_vocab_size`](#classseq2seq_1_1_seq2_seq_model_caffe2_1abcb917746774b6eba06c62a97548fd08)`,`[`target_vocab_size`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a6a845f7374dafea3d21e8b8bbf4af095)`,`[`num_gpus`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a7b443112feee1f2f91d080e45f724edc)`,`[`num_cpus`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a7ee0ebef07294dfa4c993a2414ecdf8c)`)` |
`public def __enter__(self)` |
`public def __exit__(self,exc_type,exc_value,traceback)` |
`public def initialize_from_scratch(self)` |
`public def get_current_step(self)` |
`public def inc_current_step(self)` |
`public def step(self,batch,forward_only)` |

## Members

#### `public  encoder_inputs` {#classseq2seq_1_1_seq2_seq_model_caffe2_1ab3ff71c8ee7f094ffd9e3867dbccb378}





#### `public  encoder_lengths` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a52aed2943883bcdeee185cdfe3272e15}





#### `public  decoder_inputs` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a51382c85fd64f2aab045729f8f23853f}





#### `public  decoder_lengths` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a4a886d161e815e086c0bdc193c4c6570}





#### `public  targets` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a124bfb1e07ae17dd8ac187f82600dede}





#### `public  target_weights` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a0463c0464298a5466532605196cc5aaa}





#### `public  learning_rate` {#classseq2seq_1_1_seq2_seq_model_caffe2_1adee1a429b99299aa766ab15073b956c3}





#### `public  global_step` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a94988a2fc934ba4af024d2d43c3cb51f}





#### `public  start_time` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a051ab46c5b761a12a60e5fe39d70b918}





#### `public  encoder_type` {#classseq2seq_1_1_seq2_seq_model_caffe2_1ad16c6ab34ef10e470c14d73ddc5df874}





#### `public  total_loss_scalar` {#classseq2seq_1_1_seq2_seq_model_caffe2_1ab04930acf91f70d1974180e3ab7ebb17}





#### `public  forward_net` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a110ec661e70c4a2b406db86e00a80391}





#### `public  model` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a7d30d44ddfffb4eb7053c09360d9d26d}





#### `public  model_params` {#classseq2seq_1_1_seq2_seq_model_caffe2_1ad3de9310c7b6d28fc99ede130fea77e3}





#### `public  encoder_params` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a51b1c2db1525f3d7f29bbdef28a29d98}





#### `public  source_vocab_size` {#classseq2seq_1_1_seq2_seq_model_caffe2_1abcb917746774b6eba06c62a97548fd08}





#### `public  target_vocab_size` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a6a845f7374dafea3d21e8b8bbf4af095}





#### `public  num_gpus` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a7b443112feee1f2f91d080e45f724edc}





#### `public  num_cpus` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a7ee0ebef07294dfa4c993a2414ecdf8c}





#### `public  batch_size` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a7e6899a5726666c5fd486c43f7dd5a52}





#### `public def output_projection(self,`[`model`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a7d30d44ddfffb4eb7053c09360d9d26d)`,decoder_outputs,decoder_output_size,`[`target_vocab_size`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a6a845f7374dafea3d21e8b8bbf4af095)`,decoder_softmax_size)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a724345a5bdb6a07638e382cfe2d6e05a}





#### `public def __init__(self,`[`model_params`](#classseq2seq_1_1_seq2_seq_model_caffe2_1ad3de9310c7b6d28fc99ede130fea77e3)`,`[`source_vocab_size`](#classseq2seq_1_1_seq2_seq_model_caffe2_1abcb917746774b6eba06c62a97548fd08)`,`[`target_vocab_size`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a6a845f7374dafea3d21e8b8bbf4af095)`,`[`num_gpus`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a7b443112feee1f2f91d080e45f724edc)`,`[`num_cpus`](#classseq2seq_1_1_seq2_seq_model_caffe2_1a7ee0ebef07294dfa4c993a2414ecdf8c)`)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a869156a8a7ed65161c829df7324d8b0a}





#### `public def __enter__(self)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a69f8897ffcd58e153580c14293f37597}





#### `public def __exit__(self,exc_type,exc_value,traceback)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1adc88dc1ed5d17b876568aa3431934af3}





#### `public def initialize_from_scratch(self)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1aaca967ffbd55ac339e3252c76a09b9c8}





#### `public def get_current_step(self)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1aa60cd82a26af9f315a41c0cb529cc99a}





#### `public def inc_current_step(self)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a1abf1aa66d983881b562e23340d25b9d}





#### `public def step(self,batch,forward_only)` {#classseq2seq_1_1_seq2_seq_model_caffe2_1a3d390f372110619febc94c6b151edfb4}





# namespace `seq2seq_util` {#namespaceseq2seq__util}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`seq2seq_util::ModelHelper`](#classseq2seq__util_1_1_model_helper)    |
# class `seq2seq_util::ModelHelper` {#classseq2seq__util_1_1_model_helper}

```
class seq2seq_util::ModelHelper
  : public CNNModelHelper
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  non_trainable_params` |
`public def __init__(self,init_params)` |
`public def AddParam(self,name,init,init_value,trainable)` |

## Members

#### `public  non_trainable_params` {#classseq2seq__util_1_1_model_helper_1a544e838fc7ce222594ef8c9ad5477851}





#### `public def __init__(self,init_params)` {#classseq2seq__util_1_1_model_helper_1a36d7ef841184bc73d6e953f46fab7afd}





#### `public def AddParam(self,name,init,init_value,trainable)` {#classseq2seq__util_1_1_model_helper_1aa649145df9d1176c8c802ac4bdefe024}



Adds a parameter to the model's net and it's initializer if needed

Args:
    init: a tuple (<initialization_op_name>, <initialization_op_kwargs>)
    init_value: int, float or str. Can be used instead of `init` as a
simple constant initializer
    trainable: bool, whether to compute gradient for this param or not

# namespace `sequence_ops_test` {#namespacesequence__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`sequence_ops_test::TestSequenceOps`](#classsequence__ops__test_1_1_test_sequence_ops)    |
# class `sequence_ops_test::TestSequenceOps` {#classsequence__ops__test_1_1_test_sequence_ops}

```
class sequence_ops_test::TestSequenceOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_add_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` |
`public def test_add_zero_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` |
`public def test_add_padding_no_length(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`data`](#classsequence__ops__test_1_1_test_sequence_ops_1a2e0db26793bd423bc63b355c388da4d6)`)` |
`public def test_remove_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` |
`public def test_gather_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` |
`public def test_reverse_packed_segs(self,`[`data`](#classsequence__ops__test_1_1_test_sequence_ops_1a2e0db26793bd423bc63b355c388da4d6)`,gc,dc)` |
`public def test_remove_data_blocks(self,`[`data`](#classsequence__ops__test_1_1_test_sequence_ops_1a2e0db26793bd423bc63b355c388da4d6)`,`[`indices`](#classsequence__ops__test_1_1_test_sequence_ops_1aac38dd109db186474db88e01306392d5)`,gc,dc)` |
`public def test_find_duplicate_elements(self,`[`elements`](#classsequence__ops__test_1_1_test_sequence_ops_1afa46d563e65aac80bb080e87ed8b08ec)`,gc,dc)` |

## Members

#### `public def test_add_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` {#classsequence__ops__test_1_1_test_sequence_ops_1ace936a91a882919490f3139038358532}





#### `public def test_add_zero_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` {#classsequence__ops__test_1_1_test_sequence_ops_1af1e9271ac3a747cf932332b61fb24959}





#### `public def test_add_padding_no_length(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`data`](#classsequence__ops__test_1_1_test_sequence_ops_1a2e0db26793bd423bc63b355c388da4d6)`)` {#classsequence__ops__test_1_1_test_sequence_ops_1a1ca9b1db8c6fe5121d53ab171a0e6856}





#### `public def test_remove_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` {#classsequence__ops__test_1_1_test_sequence_ops_1a58144743187dde937be762a7eef292c8}





#### `public def test_gather_padding(self,`[`start_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a4b217a8c9826173331d2d90012af4efd)`,`[`end_pad_width`](#classsequence__ops__test_1_1_test_sequence_ops_1a1b123b051701d8c2cb9a952ca62bde6f)`,`[`args`](#classsequence__ops__test_1_1_test_sequence_ops_1a5ffd2e76455990e13cf7944688c17df7)`)` {#classsequence__ops__test_1_1_test_sequence_ops_1a7b5d556524ccc89a278daf9957167014}





#### `public def test_reverse_packed_segs(self,`[`data`](#classsequence__ops__test_1_1_test_sequence_ops_1a2e0db26793bd423bc63b355c388da4d6)`,gc,dc)` {#classsequence__ops__test_1_1_test_sequence_ops_1a0bde3c392bea48e5ea55d1b6131453a6}





#### `public def test_remove_data_blocks(self,`[`data`](#classsequence__ops__test_1_1_test_sequence_ops_1a2e0db26793bd423bc63b355c388da4d6)`,`[`indices`](#classsequence__ops__test_1_1_test_sequence_ops_1aac38dd109db186474db88e01306392d5)`,gc,dc)` {#classsequence__ops__test_1_1_test_sequence_ops_1acdef1e74e2edf0c27c8dc1c001e72486}





#### `public def test_find_duplicate_elements(self,`[`elements`](#classsequence__ops__test_1_1_test_sequence_ops_1afa46d563e65aac80bb080e87ed8b08ec)`,gc,dc)` {#classsequence__ops__test_1_1_test_sequence_ops_1ad20fd69252e11d13269122813b8ebd28}





# namespace `session` {#namespacesession}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`session::CompiledRunnable`](#classsession_1_1_compiled_runnable)    |
`class `[`session::LocalSession`](#classsession_1_1_local_session)    |
`class `[`session::Session`](#classsession_1_1_session)    |
# class `session::CompiledRunnable` {#classsession_1_1_compiled_runnable}

```
class session::CompiledRunnable
  : public object
```  



Wrapper for compiled runnable returned from session.compile()

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  obj` |
`public  session_class` |
`public def __init__(self,`[`obj`](#classsession_1_1_compiled_runnable_1a8be9c4fe96f77628072aba2376e19521)`,`[`session_class`](#classsession_1_1_compiled_runnable_1a6249a99009db81ebb05effb32a078aa6)`)` |

## Members

#### `public  obj` {#classsession_1_1_compiled_runnable_1a8be9c4fe96f77628072aba2376e19521}





#### `public  session_class` {#classsession_1_1_compiled_runnable_1a6249a99009db81ebb05effb32a078aa6}





#### `public def __init__(self,`[`obj`](#classsession_1_1_compiled_runnable_1a8be9c4fe96f77628072aba2376e19521)`,`[`session_class`](#classsession_1_1_compiled_runnable_1a6249a99009db81ebb05effb32a078aa6)`)` {#classsession_1_1_compiled_runnable_1a2ebf609d60bf5b5c7f9d5febcee44d9e}





# class `session::LocalSession` {#classsession_1_1_local_session}

```
class session::LocalSession
  : public session.Session
```  



Session that runs in a single node.
Tasks are all remapped to run in parallel in the 'local' node.

Currently, LocalSession runs all parallel tasks in the same workspace,
but this behavior may change in the future. Only tasks pointing to the
same logical node are guaranteed to always run in the same workspace.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,ws)` |

## Members

#### `public def __init__(self,ws)` {#classsession_1_1_local_session_1aac72a23a0eb50229fe0f7d752bbeb49a}





# class `session::Session` {#classsession_1_1_session}

```
class session::Session
  : public object
```  



Allows to run Nets, ExecutionSteps, Plans, Tasks and TaskGroups.
A session can potentially run in multiple nodes concurrently.


Example:
    from core import Net
    from caffe2.python.task import Task, TaskGroup, WorkspaceType

    net = Net('test1')
    net.Add([net.Const(1), net.Const(2)])

    net2 = net.Clone()
    step = core.execution_step('step1', [net2])

    with TaskGroup(WorkspaceType.GLOBAL) as init_tg:
        with Node('node1'):
            n1setup = net.Net('n1setup')
            n1msg = n1setup.Const('Hello from node 1.')
            Task(step=n1setup)

    with TaskGroup() as private_tg:
        with Node('node1'):
            n1 = net.Net('n1')
            n1.Print(n1msg, 0)
            Task(step=n1)
        with Node('node2'):
            n2 = net.Net('n2')
            n2.Print(n2.Const('Hello from node 2.'), 0)
            Task(step=n2)

    session = LocalSession()
    session.run(net)
    session.run(step)
    session.run(init_tg)
    session.run(private_tg)


Global Workspace:
    At the beggining of the session, a global workspace is created and kept
    alive for the duration of the session.


Private Workspace:
    Tasks can be run either directly on the global workspace, or they can
    instantiate a private child workspace that is released after each run.

Blob visibility:
    Tasks running in different nodes in parallel will always run under
    different workspaces, so it must be assumed that they won't be able to
    access each other's blobs. On the other hand, tasks running on the same
    node are guaranteed to run on the same workspace within a run.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self)` |
`public def is_open(self)` |
`public def compile(cls,runnable)` |
`public def run(self,runnable)` |
`public def close(self)` |
`public def fetch_output(self,output)` |
`public def __enter__(self)` |
`public def __exit__(self,ex_type,value,traceback)` |

## Members

#### `public def __init__(self)` {#classsession_1_1_session_1a26c7670854f7e318bfa0d3f5237cdb5c}





#### `public def is_open(self)` {#classsession_1_1_session_1ab2a42250881fea3f1dc2ce64e2893af2}





#### `public def compile(cls,runnable)` {#classsession_1_1_session_1ae5b86b288b4f078d333ddb98f7c4c916}





#### `public def run(self,runnable)` {#classsession_1_1_session_1a49997d5effcfcce56ea72a9bdcea2e0a}





#### `public def close(self)` {#classsession_1_1_session_1a07484b24b9e702ba24d9762730480d08}





#### `public def fetch_output(self,output)` {#classsession_1_1_session_1a142b87a48ce50f7b9b9d0b87682de62f}





#### `public def __enter__(self)` {#classsession_1_1_session_1a91a60469af53b31bb624adc372386ed8}





#### `public def __exit__(self,ex_type,value,traceback)` {#classsession_1_1_session_1ad029ea36fd784f7be73bd212ec55cd93}





# namespace `session_test` {#namespacesession__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`session_test::TestLocalSession`](#classsession__test_1_1_test_local_session)    |
# class `session_test::TestLocalSession` {#classsession__test_1_1_test_local_session}

```
class session_test::TestLocalSession
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_local_session(self)` |

## Members

#### `public def test_local_session(self)` {#classsession__test_1_1_test_local_session_1a4f2ea5b9a28b39df47163617ddc2041e}





# namespace `shape_inference_test` {#namespaceshape__inference__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`shape_inference_test::TestShapeInference`](#classshape__inference__test_1_1_test_shape_inference)    |
# class `shape_inference_test::TestShapeInference` {#classshape__inference__test_1_1_test_shape_inference}

```
class shape_inference_test::TestShapeInference
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testShapeInferenceSimpleFC(self)` |
`public def testShapeInferenceDistances(self)` |
`public def testShapeInferenceConvNet(self)` |
`public def testShapeInferenceTranspose(self)` |
`public def testShapeInferencePad(self)` |
`public def testShapeInferenceTwoClass(self)` |
`public def testShapeInferencePadZero(self)` |
`public def testShapeInferenceMatMul(self)` |
`public def testShapeInferenceSoftmaxWithLoss(self)` |
`public def testShapeInferenceIm2Col(self)` |
`public def testShapeInferenceTile(self)` |
`public def testShapeInferenceFlatten(self)` |
`public def testShapeInferenceReshape(self)` |
`public def testCast(self)` |
`public def InferTensorRunAndCompare(self,model)` |

## Members

#### `public def testShapeInferenceSimpleFC(self)` {#classshape__inference__test_1_1_test_shape_inference_1aecbd91f3b3d38b0e03ca9fd6945035bf}





#### `public def testShapeInferenceDistances(self)` {#classshape__inference__test_1_1_test_shape_inference_1a98c0fa1fa8359d019ae12b95952112c6}





#### `public def testShapeInferenceConvNet(self)` {#classshape__inference__test_1_1_test_shape_inference_1a9e29fb5b35dff70c1af08362163450cf}





#### `public def testShapeInferenceTranspose(self)` {#classshape__inference__test_1_1_test_shape_inference_1ae4277f6800ae9ecd1f5bff373eb6c68a}





#### `public def testShapeInferencePad(self)` {#classshape__inference__test_1_1_test_shape_inference_1a5c603b77fe0991dfe30a65b8f52a8f6d}





#### `public def testShapeInferenceTwoClass(self)` {#classshape__inference__test_1_1_test_shape_inference_1ad43eca3585f3dafc3a609402537c853f}





#### `public def testShapeInferencePadZero(self)` {#classshape__inference__test_1_1_test_shape_inference_1a02130c88f5d2a28d030982fa14abfa19}





#### `public def testShapeInferenceMatMul(self)` {#classshape__inference__test_1_1_test_shape_inference_1a951e85a23d3e282ad2fe798a1f07299d}





#### `public def testShapeInferenceSoftmaxWithLoss(self)` {#classshape__inference__test_1_1_test_shape_inference_1a521e6ee2f5aec8729e1335936afce9e7}





#### `public def testShapeInferenceIm2Col(self)` {#classshape__inference__test_1_1_test_shape_inference_1a66bb4aacd0bdb0a4669487bccfbce506}





#### `public def testShapeInferenceTile(self)` {#classshape__inference__test_1_1_test_shape_inference_1ae75225a1b090cfc850f43043287ecb9c}





#### `public def testShapeInferenceFlatten(self)` {#classshape__inference__test_1_1_test_shape_inference_1a2759cb96825087819066f2646b999c82}





#### `public def testShapeInferenceReshape(self)` {#classshape__inference__test_1_1_test_shape_inference_1a2a981e2b6271b651a7a97fa61b15caca}





#### `public def testCast(self)` {#classshape__inference__test_1_1_test_shape_inference_1aaece04cd0f4e567729a014b09f40abbd}





#### `public def InferTensorRunAndCompare(self,model)` {#classshape__inference__test_1_1_test_shape_inference_1a5004109b0e3d72d672b759827a289dca}



Runs shape inference, and then the model to check
that the inferred shapes agree with the actual ones

# namespace `softmax_ops_test` {#namespacesoftmax__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`softmax_ops_test::TestSoftmaxOps`](#classsoftmax__ops__test_1_1_test_softmax_ops)    |
# class `softmax_ops_test::TestSoftmaxOps` {#classsoftmax__ops__test_1_1_test_softmax_ops}

```
class softmax_ops_test::TestSoftmaxOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_softmax(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` |
`public def test_softmax_axis(self,`[`axis`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ac7326ce1ae576340bf7f50afd856e77a)`,gc,dc)` |
`public def test_softmax_with_loss(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` |
`public def test_softmax_with_loss_label_prob(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` |
`public def test_softmax_with_loss_weighted(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` |
`public def test_softmax_with_loss_label_prob_weighted(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` |
`public def test_spatial_softmax_with_loss(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,`[`weighted`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ad5da73efc331688b529bef7123a5f2bb)`,gc,dc)` |
`public def test_spatial_softmax_with_loss_allignore(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,`[`weighted`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ad5da73efc331688b529bef7123a5f2bb)`,gc,dc)` |
`public def test_softmax_with_loss_zero_weight(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,`[`weighted`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ad5da73efc331688b529bef7123a5f2bb)`,gc,dc)` |
`public def test_compare_cpugpu(self)` |

## Members

#### `public def test_softmax(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1aa40f889c2d820d174e098d13c53b1af4}





#### `public def test_softmax_axis(self,`[`axis`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ac7326ce1ae576340bf7f50afd856e77a)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1a9b66d79e0c50e9956ff5ad107d9c8c9f}





#### `public def test_softmax_with_loss(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1af5157c76ced280a29a4969bc53977a2d}





#### `public def test_softmax_with_loss_label_prob(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1a9d0a0dd4e16a459d5918f0490b42d12c}





#### `public def test_softmax_with_loss_weighted(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1a9219214f2a3032a9e952a6010de322cd}





#### `public def test_softmax_with_loss_label_prob_weighted(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1a5e91e480cc330b472bcd13354ed3fa5d}





#### `public def test_spatial_softmax_with_loss(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,`[`weighted`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ad5da73efc331688b529bef7123a5f2bb)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1ae79b78fa02b3386d263d9b8af1ed6e9b}





#### `public def test_spatial_softmax_with_loss_allignore(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,`[`weighted`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ad5da73efc331688b529bef7123a5f2bb)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1ae8285e5bb410050eb9d16ec151256ce9}





#### `public def test_softmax_with_loss_zero_weight(self,`[`n`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a0a426fd72dffe8648068176067636299)`,`[`D`](#classsoftmax__ops__test_1_1_test_softmax_ops_1a95178eb10592c3bc254e9f4818e1059d)`,`[`weighted`](#classsoftmax__ops__test_1_1_test_softmax_ops_1ad5da73efc331688b529bef7123a5f2bb)`,gc,dc)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1a90241c776268913722ef5e7e00afcd27}





#### `public def test_compare_cpugpu(self)` {#classsoftmax__ops__test_1_1_test_softmax_ops_1a1c3dc0d84c592daebce46ebb6ad1ebf2}



Additional test that checks CPU and GPU returns same values
with larger examples. This is mainly to test the more complex
GPU implementation is correct.

# namespace `sparse_gradient_checker_test` {#namespacesparse__gradient__checker__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`sparse_gradient_checker_test::TestSparseGradient`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient)    |
# class `sparse_gradient_checker_test::TestSparseGradient` {#classsparse__gradient__checker__test_1_1_test_sparse_gradient}

```
class sparse_gradient_checker_test::TestSparseGradient
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_sparse_gradient(self,`[`M`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1a18377978593394467a29e6b361f94b89)`,`[`N`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1a59e8482d48150bed2fb7f7195829ab2e)`,`[`K`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1aaf3f568d800fe169a2e32f877fdab300)`,`[`sparsity`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1ab9e0f8b5d39b6528c27d3d79d52a55bb)`,gc,dc)` |

## Members

#### `public def test_sparse_gradient(self,`[`M`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1a18377978593394467a29e6b361f94b89)`,`[`N`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1a59e8482d48150bed2fb7f7195829ab2e)`,`[`K`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1aaf3f568d800fe169a2e32f877fdab300)`,`[`sparsity`](#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1ab9e0f8b5d39b6528c27d3d79d52a55bb)`,gc,dc)` {#classsparse__gradient__checker__test_1_1_test_sparse_gradient_1a43a362d99909052c9c246e74affddf84}





# namespace `sparse_ops_test` {#namespacesparse__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`sparse_ops_test::TestScatterOps`](#classsparse__ops__test_1_1_test_scatter_ops)    |
# class `sparse_ops_test::TestScatterOps` {#classsparse__ops__test_1_1_test_scatter_ops}

```
class sparse_ops_test::TestScatterOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_configs(self)` |
`public def testScatterWeightedSum(self)` |
`public def testScatterAssign(self)` |

## Members

#### `public def test_configs(self)` {#classsparse__ops__test_1_1_test_scatter_ops_1a31195782ede2fbfd591f7b40b40e260d}





#### `public def testScatterWeightedSum(self)` {#classsparse__ops__test_1_1_test_scatter_ops_1ae33bbc1d2a98c01e9e02dbb7c4253e7d}





#### `public def testScatterAssign(self)` {#classsparse__ops__test_1_1_test_scatter_ops_1a98ab2d6b2edeeb59286c30dfe2c95c09}





# namespace `sparse_to_dense_mask_test` {#namespacesparse__to__dense__mask__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`sparse_to_dense_mask_test::TestSparseToDenseMask`](#classsparse__to__dense__mask__test_1_1_test_sparse_to_dense_mask)    |
# class `sparse_to_dense_mask_test::TestSparseToDenseMask` {#classsparse__to__dense__mask__test_1_1_test_sparse_to_dense_mask}

```
class sparse_to_dense_mask_test::TestSparseToDenseMask
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_sparse_to_dense_mask_float(self)` |
`public def test_sparse_to_dense_mask_subtensor(self)` |
`public def test_sparse_to_dense_mask_string(self)` |
`public def test_sparse_to_dense_mask_empty_lengths(self)` |
`public def test_sparse_to_dense_mask_no_lengths(self)` |

## Members

#### `public def test_sparse_to_dense_mask_float(self)` {#classsparse__to__dense__mask__test_1_1_test_sparse_to_dense_mask_1a75b8a0aaf359df08bb2b1e803e63d9dc}





#### `public def test_sparse_to_dense_mask_subtensor(self)` {#classsparse__to__dense__mask__test_1_1_test_sparse_to_dense_mask_1af13867a0477e2759b8c999047fd0db9b}





#### `public def test_sparse_to_dense_mask_string(self)` {#classsparse__to__dense__mask__test_1_1_test_sparse_to_dense_mask_1acf9f63b812f5dd71a813bf35ff8f9aea}





#### `public def test_sparse_to_dense_mask_empty_lengths(self)` {#classsparse__to__dense__mask__test_1_1_test_sparse_to_dense_mask_1a5a055a4bb0f3e92d3d49236c2d17dafc}





#### `public def test_sparse_to_dense_mask_no_lengths(self)` {#classsparse__to__dense__mask__test_1_1_test_sparse_to_dense_mask_1a7f8532dd3275a1e35be4f5b065000b15}





# namespace `spatial_bn_op_test` {#namespacespatial__bn__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`spatial_bn_op_test::TestSpatialBN`](#classspatial__bn__op__test_1_1_test_spatial_b_n)    |
# class `spatial_bn_op_test::TestSpatialBN` {#classspatial__bn__op__test_1_1_test_spatial_b_n}

```
class spatial_bn_op_test::TestSpatialBN
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_spatialbn_test_mode(self,`[`size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1aa2d831477b0dcc8a84c9e5da3eccf096)`,`[`input_channels`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1acde1d1a630dc10632b3efd9154f5290d)`,`[`batch_size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1afe4499ab62e21cfec699661e285c5b69)`,`[`seed`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a3bf38b6c45e075aeecf8801e0841561d)`,`[`order`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a82ccdb0d20b3efd36f1a60b3790d18c8)`,`[`epsilon`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a6fc1d3a61a3e40b01d0fc0147e311635)`,gc,dc)` |
`public def test_spatialbn_train_mode(self,`[`size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1aa2d831477b0dcc8a84c9e5da3eccf096)`,`[`input_channels`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1acde1d1a630dc10632b3efd9154f5290d)`,`[`batch_size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1afe4499ab62e21cfec699661e285c5b69)`,`[`seed`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a3bf38b6c45e075aeecf8801e0841561d)`,`[`order`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a82ccdb0d20b3efd36f1a60b3790d18c8)`,`[`epsilon`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a6fc1d3a61a3e40b01d0fc0147e311635)`,gc,dc)` |
`public def test_spatialbn_train_mode_gradient_check(self,`[`size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1aa2d831477b0dcc8a84c9e5da3eccf096)`,`[`input_channels`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1acde1d1a630dc10632b3efd9154f5290d)`,`[`batch_size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1afe4499ab62e21cfec699661e285c5b69)`,`[`seed`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a3bf38b6c45e075aeecf8801e0841561d)`,`[`order`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a82ccdb0d20b3efd36f1a60b3790d18c8)`,`[`epsilon`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a6fc1d3a61a3e40b01d0fc0147e311635)`,gc,dc)` |

## Members

#### `public def test_spatialbn_test_mode(self,`[`size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1aa2d831477b0dcc8a84c9e5da3eccf096)`,`[`input_channels`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1acde1d1a630dc10632b3efd9154f5290d)`,`[`batch_size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1afe4499ab62e21cfec699661e285c5b69)`,`[`seed`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a3bf38b6c45e075aeecf8801e0841561d)`,`[`order`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a82ccdb0d20b3efd36f1a60b3790d18c8)`,`[`epsilon`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a6fc1d3a61a3e40b01d0fc0147e311635)`,gc,dc)` {#classspatial__bn__op__test_1_1_test_spatial_b_n_1ad2f7ace44bc0de686beb5dde34fdbada}





#### `public def test_spatialbn_train_mode(self,`[`size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1aa2d831477b0dcc8a84c9e5da3eccf096)`,`[`input_channels`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1acde1d1a630dc10632b3efd9154f5290d)`,`[`batch_size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1afe4499ab62e21cfec699661e285c5b69)`,`[`seed`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a3bf38b6c45e075aeecf8801e0841561d)`,`[`order`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a82ccdb0d20b3efd36f1a60b3790d18c8)`,`[`epsilon`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a6fc1d3a61a3e40b01d0fc0147e311635)`,gc,dc)` {#classspatial__bn__op__test_1_1_test_spatial_b_n_1ab4dc7cbab4d345858fe951dd6c7131d6}





#### `public def test_spatialbn_train_mode_gradient_check(self,`[`size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1aa2d831477b0dcc8a84c9e5da3eccf096)`,`[`input_channels`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1acde1d1a630dc10632b3efd9154f5290d)`,`[`batch_size`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1afe4499ab62e21cfec699661e285c5b69)`,`[`seed`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a3bf38b6c45e075aeecf8801e0841561d)`,`[`order`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a82ccdb0d20b3efd36f1a60b3790d18c8)`,`[`epsilon`](#classspatial__bn__op__test_1_1_test_spatial_b_n_1a6fc1d3a61a3e40b01d0fc0147e311635)`,gc,dc)` {#classspatial__bn__op__test_1_1_test_spatial_b_n_1a5bbbadd8f02b1ade53861ec71a1ed501}





# namespace `square_root_divide_op_test` {#namespacesquare__root__divide__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`square_root_divide_op_test::TestSquareRootDivide`](#classsquare__root__divide__op__test_1_1_test_square_root_divide)    |
# class `square_root_divide_op_test::TestSquareRootDivide` {#classsquare__root__divide__op__test_1_1_test_square_root_divide}

```
class square_root_divide_op_test::TestSquareRootDivide
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_square_root_divide(self,`[`data_and_scale`](#classsquare__root__divide__op__test_1_1_test_square_root_divide_1a830c373ed3b8c21b52fe0d596267a38f)`,gc,dc)` |

## Members

#### `public def test_square_root_divide(self,`[`data_and_scale`](#classsquare__root__divide__op__test_1_1_test_square_root_divide_1a830c373ed3b8c21b52fe0d596267a38f)`,gc,dc)` {#classsquare__root__divide__op__test_1_1_test_square_root_divide_1af17f7b3b4d761d578e9ea81c8883f78d}





# namespace `stats_ops_test` {#namespacestats__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`stats_ops_test::TestCounterOps`](#classstats__ops__test_1_1_test_counter_ops)    |
# class `stats_ops_test::TestCounterOps` {#classstats__ops__test_1_1_test_counter_ops}

```
class stats_ops_test::TestCounterOps
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_stats_ops(self)` |

## Members

#### `public def test_stats_ops(self)` {#classstats__ops__test_1_1_test_counter_ops_1a36fb0b56d46194398d6ed946a3eb40a8}





# namespace `string_ops_test` {#namespacestring__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`string_ops_test::TestStringOps`](#classstring__ops__test_1_1_test_string_ops)    |
# class `string_ops_test::TestStringOps` {#classstring__ops__test_1_1_test_string_ops}

```
class string_ops_test::TestStringOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_string_prefix(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` |
`public def test_string_suffix(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` |
`public def test_string_starts_with(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` |
`public def test_string_ends_with(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` |

## Members

#### `public def test_string_prefix(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` {#classstring__ops__test_1_1_test_string_ops_1a4ac4416135eedf97dd996b083eaa7950}





#### `public def test_string_suffix(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` {#classstring__ops__test_1_1_test_string_ops_1a5491f610db047cc8db6e337315bfe577}





#### `public def test_string_starts_with(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` {#classstring__ops__test_1_1_test_string_ops_1a714fa26e483f933ab3188316d513b219}





#### `public def test_string_ends_with(self,`[`strings`](#classstring__ops__test_1_1_test_string_ops_1ae1a1a07533e10ec6388371da573adebf)`)` {#classstring__ops__test_1_1_test_string_ops_1a354243c8e72ea12e80e49de7a8d45e19}





# namespace `task` {#namespacetask}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`task::Cluster`](#classtask_1_1_cluster)    |
`class `[`task::Node`](#classtask_1_1_node)    |
`class `[`task::SetupNets`](#classtask_1_1_setup_nets)    |
`class `[`task::Task`](#classtask_1_1_task)    |
`class `[`task::TaskGroup`](#classtask_1_1_task_group)    |
`class `[`task::TaskOutput`](#classtask_1_1_task_output)    |
`class `[`task::TaskOutputList`](#classtask_1_1_task_output_list)    |
`class `[`task::WorkspaceType`](#classtask_1_1_workspace_type)    |
# class `task::Cluster` {#classtask_1_1_cluster}

```
class task::Cluster
  : public object
```  



Context that keeps track of all the node names used.
Users shouldn't have to use them directly, since a Cluster is automatically
generated at the first usage of 'Node'.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self)` |
`public def add_node(self,node)` |
`public def nodes(self)` |
`public def node_kwargs(self)` |

## Members

#### `public def __init__(self)` {#classtask_1_1_cluster_1a9cc7ac66a71004d2a37f7a332a1e178a}





#### `public def add_node(self,node)` {#classtask_1_1_cluster_1a4492b65a5ddd676aacefb98bd4f378c2}





#### `public def nodes(self)` {#classtask_1_1_cluster_1a12a84ad15edd26e71295dfbc9c478d58}



Returns the list of unique node names used within this context.

#### `public def node_kwargs(self)` {#classtask_1_1_cluster_1a9875325fda74de872993867cd749984f}





# class `task::Node` {#classtask_1_1_node}

```
class task::Node
  : public object
```  



A Node context is used to indicate that all Tasks instantiated within will
run on the given node name. (Only the name of the node actually counts.)
Example:

    with TaskGroup() as tg:
        with Node('node1'):
            s1 = execution_step(...)
            Task(step=s1)
        with Node('node2'):
            s2 = execution_step(...)
        with Node('node1'):
            s3 = execution_step(...)

    In this example, all three execution steps will run in parallel.
    Moreover, s1 and s3 will run on the same node, and can see each
    others blobs.

    Additionally, a Node can be passed implementation-specific kwargs,
    in order to specify properties of the node. When using AML Flow,
    we currently support:
        resource_requirements: a fblearner.flow.api.ResourceRequirements
                               specifying requirements for this Node.
        flow_returns: a fblearner.flow.api.types.Schema object specifying
                      the output schema of the Flow operator where the
                      Node will run.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,node,`[`kwargs`](#classtask_1_1_node_1a38750a586ffa63aa2b6d312ebf3d3ff1)`)` |
`public def __str__(self)` |
`public def kwargs(self)` |

## Members

#### `public def __init__(self,node,`[`kwargs`](#classtask_1_1_node_1a38750a586ffa63aa2b6d312ebf3d3ff1)`)` {#classtask_1_1_node_1a7af48c59244e202157ab227a398c6196}





#### `public def __str__(self)` {#classtask_1_1_node_1af506014ed282930a41ede08981ed1c83}





#### `public def kwargs(self)` {#classtask_1_1_node_1a38750a586ffa63aa2b6d312ebf3d3ff1}





# class `task::SetupNets` {#classtask_1_1_setup_nets}

```
class task::SetupNets
  : public object
```  



Allow to register a list of nets to be run at initialization
and finalization of Tasks or TaskGroups.
For example, let's say you have the following:

    init_net = core.Net('init')
    my_val = init_net.ConstantFill([], 'my_val', value=0)

    net = core.Net('counter')
    net.Add([my_val, net.Const(1),], [my_val])

    with TaskGroup() as task_group:
        with Node('trainer'):
            my_task = Task(step=[net])

In order to have `init_net` run once before `net` runs for the
first time, you can do one of the following:

    net.add_object(Task.TASK_SETUP, SetupNets([init_net]))

or

    net.add_object(TaskGroup.LOCAL_SETUP, SetupNets([init_net]))

- With Task.TASK_SETUP, init_net will run once at my_task startup.
- With TaskGroup.LOCAL_SETUP, init_net will run once on node 'trainer',
  before any task of the task group is run on that node.

The same SetupNets object can be added to multiple nets. It will only
run once per Task/TaskGroup run.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  init_nets` |
`public  exit_nets` |
`public def __init__(self,`[`init_nets`](#classtask_1_1_setup_nets_1a76d514bfd02e03d7c9f127ba2a7122c9)`,`[`exit_nets`](#classtask_1_1_setup_nets_1aa108e6d4d697d995f1a39cae17c8d6b4)`)` |
`public def setup(self,init_net)` |
`public def exit(self,exit_net)` |

## Members

#### `public  init_nets` {#classtask_1_1_setup_nets_1a76d514bfd02e03d7c9f127ba2a7122c9}





#### `public  exit_nets` {#classtask_1_1_setup_nets_1aa108e6d4d697d995f1a39cae17c8d6b4}





#### `public def __init__(self,`[`init_nets`](#classtask_1_1_setup_nets_1a76d514bfd02e03d7c9f127ba2a7122c9)`,`[`exit_nets`](#classtask_1_1_setup_nets_1aa108e6d4d697d995f1a39cae17c8d6b4)`)` {#classtask_1_1_setup_nets_1a957551467a3bf2a59ce35d15b5f22e7f}





#### `public def setup(self,init_net)` {#classtask_1_1_setup_nets_1ab9abd5a7592ca71fa0f9a7a1fce7c6e6}





#### `public def exit(self,exit_net)` {#classtask_1_1_setup_nets_1af92b021a5fd1db53bef881109c2272cf}





# class `task::Task` {#classtask_1_1_task}

```
class task::Task
  : public object
```  



A Task is composed of an execution step and zero or more outputs.
Tasks are executed in the context of a TaskGroup, which, in turn, can
be run by a Session.

Task outputs are fetched by the session at the end of the run.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  node` |
`public  group` |
`public  name` |
`public def __init__(self,step,`[`outputs`](#classtask_1_1_task_1a7a27b04228308fbd402f6e5620603ac6)`,`[`workspace_type`](#classtask_1_1_task_1a2f6fcaeedd4ee71439e7a887b93a8e6d)`,`[`group`](#classtask_1_1_task_1ad4ff5a07b3e7f01176caab526538f569)`,`[`node`](#classtask_1_1_task_1a50ed3624a5e89098bb670b8cb5972ff1)`,`[`name`](#classtask_1_1_task_1a285810e656a8b4c0c31a61ff216a6aa6)`)` |
`public def __enter__(self)` |
`public def __exit__(self,type,value,traceback)` |
`public def workspace_type(self)` |
`public def add_output(self,output)` |
`public def add_outputs(self,`[`outputs`](#classtask_1_1_task_1a7a27b04228308fbd402f6e5620603ac6)`)` |
`public def set_step(self,step)` |
`public def get_step(self)` |
`public def output_list(self)` |
`public def outputs(self)` |

## Members

#### `public  node` {#classtask_1_1_task_1a50ed3624a5e89098bb670b8cb5972ff1}





#### `public  group` {#classtask_1_1_task_1ad4ff5a07b3e7f01176caab526538f569}





#### `public  name` {#classtask_1_1_task_1a285810e656a8b4c0c31a61ff216a6aa6}





#### `public def __init__(self,step,`[`outputs`](#classtask_1_1_task_1a7a27b04228308fbd402f6e5620603ac6)`,`[`workspace_type`](#classtask_1_1_task_1a2f6fcaeedd4ee71439e7a887b93a8e6d)`,`[`group`](#classtask_1_1_task_1ad4ff5a07b3e7f01176caab526538f569)`,`[`node`](#classtask_1_1_task_1a50ed3624a5e89098bb670b8cb5972ff1)`,`[`name`](#classtask_1_1_task_1a285810e656a8b4c0c31a61ff216a6aa6)`)` {#classtask_1_1_task_1a96e7c28d89a0131c7c1fbed9f7508035}



Instantiate a Task and add it to the current TaskGroup and Node.

#### `public def __enter__(self)` {#classtask_1_1_task_1a78726c6ab0cf67af3debc9a126720765}





#### `public def __exit__(self,type,value,traceback)` {#classtask_1_1_task_1ab5303fcc534e2a1087e1c5ec223020a6}





#### `public def workspace_type(self)` {#classtask_1_1_task_1a2f6fcaeedd4ee71439e7a887b93a8e6d}





#### `public def add_output(self,output)` {#classtask_1_1_task_1a13b9df3efd47460b526d4796b7047319}





#### `public def add_outputs(self,`[`outputs`](#classtask_1_1_task_1a7a27b04228308fbd402f6e5620603ac6)`)` {#classtask_1_1_task_1a72e4547de4bb677fa2410f9292b7c402}





#### `public def set_step(self,step)` {#classtask_1_1_task_1a42f81a806a8ea0b8dfbc9c01d7d4d372}





#### `public def get_step(self)` {#classtask_1_1_task_1a01ef5d76641374cedb1c4d3bae472f58}





#### `public def output_list(self)` {#classtask_1_1_task_1a775c4b18f910ceba0c4782b3a1620408}





#### `public def outputs(self)` {#classtask_1_1_task_1a7a27b04228308fbd402f6e5620603ac6}





# class `task::TaskGroup` {#classtask_1_1_task_group}

```
class task::TaskGroup
  : public object
```  



Context that gathers tasks which will run concurrently, potentially on
multiple nodes. All tasks in the same node will share the same workspace
and thus can share blobs, while tasks running in different nodes won't
be able to directly share data.

All tasks of the task group will start concurrently, and the task group
will finish execution when the last task of the group finishes.

Example:
    # supose that s1 ... s5 are execution steps or nets.
    with TaskGroup() as tg:
        # these tasks go to default node 'local'
        Task(step=s1)
        Task(step=s2)

        with Node('n2'):
            Task(step=s3)
        with Node('n1'):
            Task(step=s4)
        with Node('n2'):
            Task(step=s5)

    # this will run all steps in parallel.
    # s1 and s2 will run at default node 'local'
    # s3 and s5 will run at node 'n2'
    # s4 will run at node 'n1'
    session.run(tg)

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,workspace_type)` |
`public def add(self,task)` |
`public def tasks(self)` |
`public def num_registered_tasks(self)` |
`public def used_nodes(self)` |
`public def report_step(self,step,node,interval_ms)` |
`public def report_net(self,net,node,report_interval)` |
`public def tasks_by_node(self,node_remap)` |
`public def to_task(self,node)` |

## Members

#### `public def __init__(self,workspace_type)` {#classtask_1_1_task_group_1a4ec1d6762343fadbcb0abd7b2c41470c}





#### `public def add(self,task)` {#classtask_1_1_task_group_1a4bf3ad9a2505b9805b40364211d23986}





#### `public def tasks(self)` {#classtask_1_1_task_group_1a447fd85a4320ce441a46ee3b563c9f80}





#### `public def num_registered_tasks(self)` {#classtask_1_1_task_group_1a662a2056e27adf0970336576b737b98d}





#### `public def used_nodes(self)` {#classtask_1_1_task_group_1a27f17c735c64c083e27ffe651cf5b6b2}





#### `public def report_step(self,step,node,interval_ms)` {#classtask_1_1_task_group_1aa46e2369a8c3fef51fe774aad3f6c817}



Add a "report step" to this TaskGroup. This step will run repeatedly
every `interval_ms` milliseconds for the duration of the TaskGroup
execution on each of the nodes. It is guaranteed that this step
will be run at least once after every Task in the node has finished.

#### `public def report_net(self,net,node,report_interval)` {#classtask_1_1_task_group_1a2c135ef9e3e4235c92eaa1001e158134}



DEPRECATED. Use report_step instead.

#### `public def tasks_by_node(self,node_remap)` {#classtask_1_1_task_group_1a784e6622c707bac5fc7208e7ae60ca14}





#### `public def to_task(self,node)` {#classtask_1_1_task_group_1a9f47d44a0c49546af4840329bd90feb6}





# class `task::TaskOutput` {#classtask_1_1_task_output}

```
class task::TaskOutput
  : public object
```  



Represents the output of a task. An output can be a blob,
a list of blob, or a record.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  names` |
`public def __init__(self,`[`names`](#classtask_1_1_task_output_1a8594c6f5dcc63d8bc97b5198ee98db0e)`)` |
`public def set(self,values,_fetch_func)` |
`public def get(self)` |
`public def fetch(self)` |

## Members

#### `public  names` {#classtask_1_1_task_output_1a8594c6f5dcc63d8bc97b5198ee98db0e}





#### `public def __init__(self,`[`names`](#classtask_1_1_task_output_1a8594c6f5dcc63d8bc97b5198ee98db0e)`)` {#classtask_1_1_task_output_1a0e04af3f5e2def2dbea7f568d6837e24}





#### `public def set(self,values,_fetch_func)` {#classtask_1_1_task_output_1aef470a1f95991387605a873a8e5ae724}





#### `public def get(self)` {#classtask_1_1_task_output_1ac1e444f3a59beeee3a4a130ac5535ef1}





#### `public def fetch(self)` {#classtask_1_1_task_output_1a47a0b5e8ecee76f60bc2950fa083b23f}





# class `task::TaskOutputList` {#classtask_1_1_task_output_list}

```
class task::TaskOutputList
  : public object
```  



Keeps a list of outputs for a task

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  outputs` |
`public def __init__(self,`[`outputs`](#classtask_1_1_task_output_list_1ab40de99861e646be2b1f27d625989b23)`)` |
`public def names(self)` |
`public def set_values(self,values,_fetch_func)` |

## Members

#### `public  outputs` {#classtask_1_1_task_output_list_1ab40de99861e646be2b1f27d625989b23}





#### `public def __init__(self,`[`outputs`](#classtask_1_1_task_output_list_1ab40de99861e646be2b1f27d625989b23)`)` {#classtask_1_1_task_output_list_1a10ec348aace19e787bcc4e6798e568e7}





#### `public def names(self)` {#classtask_1_1_task_output_list_1a315290e3e3a1fd5e98f009174f33bc07}



Retrive the output names.
TODO(azzolini): make this schema-based.

#### `public def set_values(self,values,_fetch_func)` {#classtask_1_1_task_output_list_1a26c9a713667bedccc437fd68ca105d3f}





# class `task::WorkspaceType` {#classtask_1_1_workspace_type}

```
class task::WorkspaceType
  : public object
```  



Determines whether tasks of a TaskGroup will run directly at the global
workspace, which is kept alive across runs, or whether a new child
workspace will be created for the run and destroyed afterwards.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# namespace `test_util` {#namespacetest__util}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`test_util::TestCase`](#classtest__util_1_1_test_case)    |
# class `test_util::TestCase` {#classtest__util_1_1_test_case}

```
class test_util::TestCase
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  ws` |
`public def setUpClass(cls)` |
`public def setUp(self)` |
`public def tearDown(self)` |

## Members

#### `public  ws` {#classtest__util_1_1_test_case_1a7aa0826adf8c1249e7e08d2a1cd7f137}





#### `public def setUpClass(cls)` {#classtest__util_1_1_test_case_1a17b661cbd05729928731babf8051a2a8}





#### `public def setUp(self)` {#classtest__util_1_1_test_case_1a11eb6c358e13093508e60e5342b04007}





#### `public def tearDown(self)` {#classtest__util_1_1_test_case_1a89f3023d92f9010425539e556ed649d5}





# namespace `text_file_reader` {#namespacetext__file__reader}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`text_file_reader::TextFileReader`](#classtext__file__reader_1_1_text_file_reader)    |
# class `text_file_reader::TextFileReader` {#classtext__file__reader_1_1_text_file_reader}

```
class text_file_reader::TextFileReader
  : public Reader
```  



Wrapper around operators for reading from text files.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __init__(self,init_net,filename,schema,num_passes,batch_size)` |
`public def read(self,net)` |

## Members

#### `public def __init__(self,init_net,filename,schema,num_passes,batch_size)` {#classtext__file__reader_1_1_text_file_reader_1a244b3c08ffb56280e8e0552514afa954}



Create op for building a TextFileReader instance in the workspace.

Args:
    init_net   : Net that will be run only once at startup.
    filename   : Path to file to read from.
    schema     : schema.Struct representing the schema of the data.
         Currently, only support Struct of strings.
    num_passes : Number of passes over the data.
    batch_size : Number of rows to read at a time.

#### `public def read(self,net)` {#classtext__file__reader_1_1_text_file_reader_1acee9fe684b15556ab5eb8e91fd6382ab}



Create op for reading a batch of rows.

# namespace `text_file_reader_test` {#namespacetext__file__reader__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`text_file_reader_test::TestTextFileReader`](#classtext__file__reader__test_1_1_test_text_file_reader)    |
# class `text_file_reader_test::TestTextFileReader` {#classtext__file__reader__test_1_1_test_text_file_reader}

```
class text_file_reader_test::TestTextFileReader
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_text_file_reader(self)` |

## Members

#### `public def test_text_file_reader(self)` {#classtext__file__reader__test_1_1_test_text_file_reader_1a7817f40d29db3cc17199c4ee0ed8fa8e}





# namespace `tile_op_test` {#namespacetile__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`tile_op_test::TestTile`](#classtile__op__test_1_1_test_tile)    |
# class `tile_op_test::TestTile` {#classtile__op__test_1_1_test_tile}

```
class tile_op_test::TestTile
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_tile(self,`[`M`](#classtile__op__test_1_1_test_tile_1aed0afd85df4316e30c759260981ab9ce)`,`[`K`](#classtile__op__test_1_1_test_tile_1aac6c46803b87dedbbd786edb1ac80526)`,`[`N`](#classtile__op__test_1_1_test_tile_1a4aadab933a39e00cef1baddec3fbb6e2)`,`[`tiles`](#classtile__op__test_1_1_test_tile_1a7930214b4266c1c1df29824d7ed666da)`,`[`axis`](#classtile__op__test_1_1_test_tile_1a9e26ae367acb31a8d5122019c096ba17)`,gc,dc)` |

## Members

#### `public def test_tile(self,`[`M`](#classtile__op__test_1_1_test_tile_1aed0afd85df4316e30c759260981ab9ce)`,`[`K`](#classtile__op__test_1_1_test_tile_1aac6c46803b87dedbbd786edb1ac80526)`,`[`N`](#classtile__op__test_1_1_test_tile_1a4aadab933a39e00cef1baddec3fbb6e2)`,`[`tiles`](#classtile__op__test_1_1_test_tile_1a7930214b4266c1c1df29824d7ed666da)`,`[`axis`](#classtile__op__test_1_1_test_tile_1a9e26ae367acb31a8d5122019c096ba17)`,gc,dc)` {#classtile__op__test_1_1_test_tile_1ad43811003c9c98eed3348b0d7e93d868}





# namespace `timeout_guard` {#namespacetimeout__guard}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`timeout_guard::WatcherThread`](#classtimeout__guard_1_1_watcher_thread)    |
# class `timeout_guard::WatcherThread` {#classtimeout__guard_1_1_watcher_thread}

```
class timeout_guard::WatcherThread
  : public Thread
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  timeout_secs` |
`public  completed` |
`public  condition` |
`public  daemon` |
`public  caller_thread` |
`public def __init__(self,`[`timeout_secs`](#classtimeout__guard_1_1_watcher_thread_1ad71f833c03d8dd8db4e6c7f17d12b3c9)`)` |
`public def run(self)` |

## Members

#### `public  timeout_secs` {#classtimeout__guard_1_1_watcher_thread_1ad71f833c03d8dd8db4e6c7f17d12b3c9}





#### `public  completed` {#classtimeout__guard_1_1_watcher_thread_1a91b9789732466480d49198ccdafbff9e}





#### `public  condition` {#classtimeout__guard_1_1_watcher_thread_1a5c2ed54304d910c40c6a565c5bd3a953}





#### `public  daemon` {#classtimeout__guard_1_1_watcher_thread_1aa73f43381b5bc660ab04c8df95a60d5c}





#### `public  caller_thread` {#classtimeout__guard_1_1_watcher_thread_1aefb7d97a7badc0fb5cb2c11a6dc62670}





#### `public def __init__(self,`[`timeout_secs`](#classtimeout__guard_1_1_watcher_thread_1ad71f833c03d8dd8db4e6c7f17d12b3c9)`)` {#classtimeout__guard_1_1_watcher_thread_1ace143b85ed51bf6872291fde85b59339}





#### `public def run(self)` {#classtimeout__guard_1_1_watcher_thread_1aeaddd55064b03e2fd39e410524b63e16}





# namespace `top_k_test` {#namespacetop__k__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`top_k_test::TestTopK`](#classtop__k__test_1_1_test_top_k)    |
# class `top_k_test::TestTopK` {#classtop__k__test_1_1_test_top_k}

```
class top_k_test::TestTopK
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_top_k(self,`[`X`](#classtop__k__test_1_1_test_top_k_1a5ebcf1363931062ea6fe71e2191bbf04)`,gc,dc)` |

## Members

#### `public def test_top_k(self,`[`X`](#classtop__k__test_1_1_test_top_k_1a5ebcf1363931062ea6fe71e2191bbf04)`,gc,dc)` {#classtop__k__test_1_1_test_top_k_1adcec928277d04594cfd780e7e397bace}





# namespace `toy_regression_test` {#namespacetoy__regression__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`toy_regression_test::TestToyRegression`](#classtoy__regression__test_1_1_test_toy_regression)    |
# class `toy_regression_test::TestToyRegression` {#classtoy__regression__test_1_1_test_toy_regression}

```
class toy_regression_test::TestToyRegression
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testToyRegression(self)` |

## Members

#### `public def testToyRegression(self)` {#classtoy__regression__test_1_1_test_toy_regression_1ac2c780670ba3580370784a02b117b8f7}



Tests a toy regression end to end.

The test code carries a simple toy regression in the form
    y = 2.0 x1 + 1.5 x2 + 0.5
by randomly generating gaussian inputs and calculating the ground
truth outputs in the net as well. It uses a standard SGD to then
train the parameters.

# namespace `tt_core_test` {#namespacett__core__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`tt_core_test::TestTTSVD`](#classtt__core__test_1_1_test_t_t_s_v_d)    |
# class `tt_core_test::TestTTSVD` {#classtt__core__test_1_1_test_t_t_s_v_d}

```
class tt_core_test::TestTTSVD
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_full_tt_svd(self)` |

## Members

#### `public def test_full_tt_svd(self)` {#classtt__core__test_1_1_test_t_t_s_v_d_1aeb89481e18d3d8a63c38ef179086d4af}





# namespace `unique_uniform_fill_op_test` {#namespaceunique__uniform__fill__op__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`unique_uniform_fill_op_test::TestUniqueUniformFillOp`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op)    |
# class `unique_uniform_fill_op_test::TestUniqueUniformFillOp` {#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op}

```
class unique_uniform_fill_op_test::TestUniqueUniformFillOp
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_unique_uniform_int_fill(self,`[`r`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1a16e84287a7e70632fe9785da4ae84bda)`,`[`avoid`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1a3de44b3c5f923be02bdc2955767e93d8)`,`[`dtypes`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1a8e82f3cb9b13da4302637acff40fa6d1)`,`[`s`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1ac6314dfc7218f3c45611c1431c3208f8)`,gc,dc)` |

## Members

#### `public def test_unique_uniform_int_fill(self,`[`r`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1a16e84287a7e70632fe9785da4ae84bda)`,`[`avoid`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1a3de44b3c5f923be02bdc2955767e93d8)`,`[`dtypes`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1a8e82f3cb9b13da4302637acff40fa6d1)`,`[`s`](#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1ac6314dfc7218f3c45611c1431c3208f8)`,gc,dc)` {#classunique__uniform__fill__op__test_1_1_test_unique_uniform_fill_op_1a5fe230c3b5f9c89d02bd4ccb2b38b1b7}





# namespace `utility_ops_test` {#namespaceutility__ops__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`utility_ops_test::TestUtilityOps`](#classutility__ops__test_1_1_test_utility_ops)    |
# class `utility_ops_test::TestUtilityOps` {#classutility__ops__test_1_1_test_utility_ops}

```
class utility_ops_test::TestUtilityOps
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_elementwise_max(self,`[`n`](#classutility__ops__test_1_1_test_utility_ops_1a7c993d119f43d58f2f532785ed5972a8)`,`[`m`](#classutility__ops__test_1_1_test_utility_ops_1a54b9acac428679d2f97f4a0473f6897b)`,`[`d`](#classutility__ops__test_1_1_test_utility_ops_1ad0cddd1a799a164028982e84f248a7ee)`,gc,dc)` |
`public def test_elementwise_sum(self,`[`n`](#classutility__ops__test_1_1_test_utility_ops_1a7c993d119f43d58f2f532785ed5972a8)`,gc,dc)` |
`public def test_elementwise_avg(self,`[`n`](#classutility__ops__test_1_1_test_utility_ops_1a7c993d119f43d58f2f532785ed5972a8)`,gc,dc)` |

## Members

#### `public def test_elementwise_max(self,`[`n`](#classutility__ops__test_1_1_test_utility_ops_1a7c993d119f43d58f2f532785ed5972a8)`,`[`m`](#classutility__ops__test_1_1_test_utility_ops_1a54b9acac428679d2f97f4a0473f6897b)`,`[`d`](#classutility__ops__test_1_1_test_utility_ops_1ad0cddd1a799a164028982e84f248a7ee)`,gc,dc)` {#classutility__ops__test_1_1_test_utility_ops_1a9cd4aadc19ebdb8820b874557030dc34}





#### `public def test_elementwise_sum(self,`[`n`](#classutility__ops__test_1_1_test_utility_ops_1a7c993d119f43d58f2f532785ed5972a8)`,gc,dc)` {#classutility__ops__test_1_1_test_utility_ops_1ae094be329c809f0d980b10ef59cd47c0}





#### `public def test_elementwise_avg(self,`[`n`](#classutility__ops__test_1_1_test_utility_ops_1a7c993d119f43d58f2f532785ed5972a8)`,gc,dc)` {#classutility__ops__test_1_1_test_utility_ops_1a7038b8e969ea3f3eefebd98071a408de}





# namespace `utils` {#namespaceutils}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`utils::DebugMode`](#classutils_1_1_debug_mode)    |
# class `utils::DebugMode` {#classutils_1_1_debug_mode}

```
class utils::DebugMode
  : public object
```  



This class allows to drop you into an interactive debugger
if there is an unhandled exception in your python script

Example of usage:

def main():
    # your code here
    pass

if __name__ == '__main__':
    from caffe2.python.utils import DebugMode
    DebugMode.run(main)

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def run(cls,func)` |

## Members

#### `public def run(cls,func)` {#classutils_1_1_debug_mode_1ac3b9d8c7497597c4bd8a636f71069a09}





# namespace `visualize` {#namespacevisualize}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`visualize::NCHW`](#classvisualize_1_1_n_c_h_w)    |
`class `[`visualize::NHWC`](#classvisualize_1_1_n_h_w_c)    |
`class `[`visualize::PatchVisualizer`](#classvisualize_1_1_patch_visualizer)    |
# class `visualize::NCHW` {#classvisualize_1_1_n_c_h_w}

```
class visualize::NCHW
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `visualize::NHWC` {#classvisualize_1_1_n_h_w_c}

```
class visualize::NHWC
  : public object
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `visualize::PatchVisualizer` {#classvisualize_1_1_patch_visualizer}

```
class visualize::PatchVisualizer
  : public object
```  



PatchVisualizer visualizes patches.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  gap` |
`public def __init__(self,`[`gap`](#classvisualize_1_1_patch_visualizer_1a7706ed46321cad40417784858820b2ca)`)` |
`public def ShowSingle(self,patch,cmap)` |
`public def ShowMultiple(self,patches,ncols,cmap,bg_func)` |
`public def ShowImages(self,patches,args,kwargs)` |
`public def ShowChannels(self,patch,cmap,bg_func)` |
`public def get_patch_shape(self,patch)` |

## Members

#### `public  gap` {#classvisualize_1_1_patch_visualizer_1a7706ed46321cad40417784858820b2ca}





#### `public def __init__(self,`[`gap`](#classvisualize_1_1_patch_visualizer_1a7706ed46321cad40417784858820b2ca)`)` {#classvisualize_1_1_patch_visualizer_1a39b7c7fbeb0569a7aef78adf23451bd2}





#### `public def ShowSingle(self,patch,cmap)` {#classvisualize_1_1_patch_visualizer_1a78ad7e6d98587aac77a5b88961ab00e8}



Visualizes one single patch.

    The input patch could be a vector (in which case we try to infer the shape
    of the patch), a 2-D matrix, or a 3-D matrix whose 3rd dimension has 3
    channels.

#### `public def ShowMultiple(self,patches,ncols,cmap,bg_func)` {#classvisualize_1_1_patch_visualizer_1a78290245d736f3ec307b4c92ba8b8527}



Visualize multiple patches.

    In the passed in patches matrix, each row is a patch, in the shape of either
    n*n, n*n*1 or n*n*3, either in a flattened format (so patches would be a
    2-D array), or a multi-dimensional tensor. We will try our best to figure
    out automatically the patch size.

#### `public def ShowImages(self,patches,args,kwargs)` {#classvisualize_1_1_patch_visualizer_1a7049dc2c635d2c93a0166fa2e0d637df}



Similar to ShowMultiple, but always normalize the values between 0 and 1
    for better visualization of image-type data.

#### `public def ShowChannels(self,patch,cmap,bg_func)` {#classvisualize_1_1_patch_visualizer_1afbabeeed6779df0720a6bd51df4019f6}



This function shows the channels of a patch.

    The incoming patch should have shape [w, h, num_channels], and each channel
    will be visualized as a separate gray patch.

#### `public def get_patch_shape(self,patch)` {#classvisualize_1_1_patch_visualizer_1af7eda3eaa678f397485f549cb540b600}



Gets the shape of a single patch.

    Basically it tries to interprete the patch as a square, and also check if it
    is in color (3 channels)

# namespace `workspace` {#namespaceworkspace}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`workspace::_BlobDict`](#classworkspace_1_1___blob_dict)    |
# class `workspace::_BlobDict` {#classworkspace_1_1___blob_dict}

```
class workspace::_BlobDict
  : public object
```  



Provides python dict compatible way to do fetching and feeding

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def __getitem__(self,key)` |
`public def __setitem__(self,key,value)` |
`public def __len__(self)` |
`public def __iter__(self)` |
`public def __contains__(self,item)` |

## Members

#### `public def __getitem__(self,key)` {#classworkspace_1_1___blob_dict_1aabe1a599898dd525c9d057ccbbe3a132}





#### `public def __setitem__(self,key,value)` {#classworkspace_1_1___blob_dict_1af486448849930dfe6930c28c00cbcbc0}





#### `public def __len__(self)` {#classworkspace_1_1___blob_dict_1ad68c3d569e157135fef3acad70107aa4}





#### `public def __iter__(self)` {#classworkspace_1_1___blob_dict_1a48fbacbfaa07799bde3f1e8a59284f52}





#### `public def __contains__(self,item)` {#classworkspace_1_1___blob_dict_1a627cdbdb4b75be68a815a31679df1202}





# namespace `workspace_test` {#namespaceworkspace__test}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`workspace_test::TestCppEnforceAsException`](#classworkspace__test_1_1_test_cpp_enforce_as_exception)    |
`class `[`workspace_test::TestCWorkspace`](#classworkspace__test_1_1_test_c_workspace)    |
`class `[`workspace_test::TestImmedibate`](#classworkspace__test_1_1_test_immedibate)    |
`class `[`workspace_test::TestMultiWorkspaces`](#classworkspace__test_1_1_test_multi_workspaces)    |
`class `[`workspace_test::TestPredictor`](#classworkspace__test_1_1_test_predictor)    |
`class `[`workspace_test::TestWorkspace`](#classworkspace__test_1_1_test_workspace)    |
`class `[`workspace_test::TestWorkspaceGPU`](#classworkspace__test_1_1_test_workspace_g_p_u)    |
`class `[`workspace_test::TestWorkspaceMKLDNN`](#classworkspace__test_1_1_test_workspace_m_k_l_d_n_n)    |
# class `workspace_test::TestCppEnforceAsException` {#classworkspace__test_1_1_test_cpp_enforce_as_exception}

```
class workspace_test::TestCppEnforceAsException
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testEnforce(self)` |

## Members

#### `public def testEnforce(self)` {#classworkspace__test_1_1_test_cpp_enforce_as_exception_1a9afd601b2593399cc89921116c595787}





# class `workspace_test::TestCWorkspace` {#classworkspace__test_1_1_test_c_workspace}

```
class workspace_test::TestCWorkspace
  : public HypothesisTestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def test_net_execution(self)` |
`public def test_operator_run(self,`[`name`](#classworkspace__test_1_1_test_c_workspace_1af61a92a392043561f423b78dc4591096)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` |
`public def test_net_run(self,`[`blob_name`](#classworkspace__test_1_1_test_c_workspace_1a828dfb0b6bfde644c1f42d04697edca8)`,`[`net_name`](#classworkspace__test_1_1_test_c_workspace_1afae8a9500e688a693bf052ee2e87de28)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` |
`public def test_plan_run(self,`[`blob_name`](#classworkspace__test_1_1_test_c_workspace_1a828dfb0b6bfde644c1f42d04697edca8)`,`[`plan_name`](#classworkspace__test_1_1_test_c_workspace_1a30f414c39ce217fa9c47a1af9df5bcbb)`,`[`net_name`](#classworkspace__test_1_1_test_c_workspace_1afae8a9500e688a693bf052ee2e87de28)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` |
`public def test_net_create(self,`[`blob_name`](#classworkspace__test_1_1_test_c_workspace_1a828dfb0b6bfde644c1f42d04697edca8)`,`[`net_name`](#classworkspace__test_1_1_test_c_workspace_1afae8a9500e688a693bf052ee2e87de28)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` |
`public def test_array_serde(self,`[`name`](#classworkspace__test_1_1_test_c_workspace_1af61a92a392043561f423b78dc4591096)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`,`[`device_option`](#classworkspace__test_1_1_test_c_workspace_1ade8fa44c7302a21b36a820080cb2bd66)`)` |
`public def test_string_serde(self,`[`name`](#classworkspace__test_1_1_test_c_workspace_1af61a92a392043561f423b78dc4591096)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` |
`public def test_exception(self)` |

## Members

#### `public def test_net_execution(self)` {#classworkspace__test_1_1_test_c_workspace_1add4273381e767467b2101aea3a8e0975}





#### `public def test_operator_run(self,`[`name`](#classworkspace__test_1_1_test_c_workspace_1af61a92a392043561f423b78dc4591096)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` {#classworkspace__test_1_1_test_c_workspace_1a83aab97c9ba09c85e0144c7a2d96f8ae}





#### `public def test_net_run(self,`[`blob_name`](#classworkspace__test_1_1_test_c_workspace_1a828dfb0b6bfde644c1f42d04697edca8)`,`[`net_name`](#classworkspace__test_1_1_test_c_workspace_1afae8a9500e688a693bf052ee2e87de28)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` {#classworkspace__test_1_1_test_c_workspace_1ad50f4ada7401eb3370a9fcf7a61d9fa8}





#### `public def test_plan_run(self,`[`blob_name`](#classworkspace__test_1_1_test_c_workspace_1a828dfb0b6bfde644c1f42d04697edca8)`,`[`plan_name`](#classworkspace__test_1_1_test_c_workspace_1a30f414c39ce217fa9c47a1af9df5bcbb)`,`[`net_name`](#classworkspace__test_1_1_test_c_workspace_1afae8a9500e688a693bf052ee2e87de28)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` {#classworkspace__test_1_1_test_c_workspace_1a5bb412460057b4361529971ff99bd925}





#### `public def test_net_create(self,`[`blob_name`](#classworkspace__test_1_1_test_c_workspace_1a828dfb0b6bfde644c1f42d04697edca8)`,`[`net_name`](#classworkspace__test_1_1_test_c_workspace_1afae8a9500e688a693bf052ee2e87de28)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` {#classworkspace__test_1_1_test_c_workspace_1ab6b6f4eeb7a361bed2f047d3b02a424a}





#### `public def test_array_serde(self,`[`name`](#classworkspace__test_1_1_test_c_workspace_1af61a92a392043561f423b78dc4591096)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`,`[`device_option`](#classworkspace__test_1_1_test_c_workspace_1ade8fa44c7302a21b36a820080cb2bd66)`)` {#classworkspace__test_1_1_test_c_workspace_1a77ea3c025dde76f4284f9d009fefd515}





#### `public def test_string_serde(self,`[`name`](#classworkspace__test_1_1_test_c_workspace_1af61a92a392043561f423b78dc4591096)`,`[`value`](#classworkspace__test_1_1_test_c_workspace_1a9d7d305eb2999232385e9926dc4cef19)`)` {#classworkspace__test_1_1_test_c_workspace_1a73cab7816215aaf3cf08d89a0435bb45}





#### `public def test_exception(self)` {#classworkspace__test_1_1_test_c_workspace_1ace930106b7dffc5da845456c84db566e}





# class `workspace_test::TestImmedibate` {#classworkspace__test_1_1_test_immedibate}

```
class workspace_test::TestImmedibate
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testImmediateEnterExit(self)` |
`public def testImmediateRunsCorrectly(self)` |
`public def testImmediateRootFolder(self)` |

## Members

#### `public def testImmediateEnterExit(self)` {#classworkspace__test_1_1_test_immedibate_1a8040d1a66acb517ad7898a1e8c92db07}





#### `public def testImmediateRunsCorrectly(self)` {#classworkspace__test_1_1_test_immedibate_1a9115d6b8d26cc6d36ef77dd7b5479e6e}





#### `public def testImmediateRootFolder(self)` {#classworkspace__test_1_1_test_immedibate_1abdf9aa7300f7b3e08715b6d188333673}





# class `workspace_test::TestMultiWorkspaces` {#classworkspace__test_1_1_test_multi_workspaces}

```
class workspace_test::TestMultiWorkspaces
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  net` |
`public def setUp(self)` |
`public def testCreateWorkspace(self)` |

## Members

#### `public  net` {#classworkspace__test_1_1_test_multi_workspaces_1a565c7350739a9420a3209b24aef382a0}





#### `public def setUp(self)` {#classworkspace__test_1_1_test_multi_workspaces_1a4c3f61f2f2e574ec62d1a2edd955bf12}





#### `public def testCreateWorkspace(self)` {#classworkspace__test_1_1_test_multi_workspaces_1a9cfa024422747c58b6782b924bd768a4}





# class `workspace_test::TestPredictor` {#classworkspace__test_1_1_test_predictor}

```
class workspace_test::TestPredictor
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  predictor` |
`public def test_predictor_memory_model(self)` |

## Members

#### `public  predictor` {#classworkspace__test_1_1_test_predictor_1a80370b92edfdff238ec6708647c436e8}





#### `public def test_predictor_memory_model(self)` {#classworkspace__test_1_1_test_predictor_1aa915dd68298f89e2a8925442ecf222fa}





# class `workspace_test::TestWorkspace` {#classworkspace__test_1_1_test_workspace}

```
class workspace_test::TestWorkspace
  : public TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  net` |
`public  testblob_ref` |
`public def setUp(self)` |
`public def testRootFolder(self)` |
`public def testWorkspaceHasBlobWithNonexistingName(self)` |
`public def testRunOperatorOnce(self)` |
`public def testRunNetOnce(self)` |
`public def testCurrentWorkspaceWrapper(self)` |
`public def testRunPlan(self)` |
`public def testConstructPlanFromSteps(self)` |
`public def testResetWorkspace(self)` |
`public def testTensorAccess(self)` |
`public def testFetchFeedBlob(self)` |
`public def testFetchFeedBlobViaBlobReference(self)` |
`public def testFetchFeedBlobTypes(self)` |
`public def testFetchFeedBlobBool(self)` |
`public def testFetchFeedBlobZeroDim(self)` |
`public def testFetchFeedLongStringTensor(self)` |
`public def testFetchFeedShortStringTensor(self)` |
`public def testFetchFeedPlainString(self)` |
`public def testFetchBlobs(self)` |
`public def testFetchFeedViaBlobDict(self)` |

## Members

#### `public  net` {#classworkspace__test_1_1_test_workspace_1a34d36b8d52c251efc87642950675b903}





#### `public  testblob_ref` {#classworkspace__test_1_1_test_workspace_1a780ef627fcd148a7720fe0d2ab85e66c}





#### `public def setUp(self)` {#classworkspace__test_1_1_test_workspace_1a15c67420334011655beaa8f0c8478876}





#### `public def testRootFolder(self)` {#classworkspace__test_1_1_test_workspace_1a4f033029a4aba6a0c93fc2fbeb1a8c75}





#### `public def testWorkspaceHasBlobWithNonexistingName(self)` {#classworkspace__test_1_1_test_workspace_1a287d4a8ce0262608dd1663943bedf37b}





#### `public def testRunOperatorOnce(self)` {#classworkspace__test_1_1_test_workspace_1af1478b734fc9393547eaf92aa43f2470}





#### `public def testRunNetOnce(self)` {#classworkspace__test_1_1_test_workspace_1a388160db3a3d1381da3eb1fc76e93182}





#### `public def testCurrentWorkspaceWrapper(self)` {#classworkspace__test_1_1_test_workspace_1a4023f8f792240b93fe25095986dca145}





#### `public def testRunPlan(self)` {#classworkspace__test_1_1_test_workspace_1a3af5ea6c2c9f0be15a1065de59c65347}





#### `public def testConstructPlanFromSteps(self)` {#classworkspace__test_1_1_test_workspace_1a8a1d6f589e7d89d4fa7583ef499c8749}





#### `public def testResetWorkspace(self)` {#classworkspace__test_1_1_test_workspace_1aec2b3ba66b2e2dcbdc5ce3b8931f5586}





#### `public def testTensorAccess(self)` {#classworkspace__test_1_1_test_workspace_1aba2f85c5a29c4b4b0144b0d1656fb894}





#### `public def testFetchFeedBlob(self)` {#classworkspace__test_1_1_test_workspace_1a530539957ff5cb8d233fc5528dca073d}





#### `public def testFetchFeedBlobViaBlobReference(self)` {#classworkspace__test_1_1_test_workspace_1a38b4efd54643453022bc0b9d3b9451c8}





#### `public def testFetchFeedBlobTypes(self)` {#classworkspace__test_1_1_test_workspace_1a6e925fcdc75ae9cb2c38136287189238}





#### `public def testFetchFeedBlobBool(self)` {#classworkspace__test_1_1_test_workspace_1aec1729c9a5852801515cea05cf5054f8}



Special case for bool to ensure coverage of both true and false.

#### `public def testFetchFeedBlobZeroDim(self)` {#classworkspace__test_1_1_test_workspace_1a6c7cc5880aef892e985efd156a84c76b}





#### `public def testFetchFeedLongStringTensor(self)` {#classworkspace__test_1_1_test_workspace_1ae18b4864a0ed71056b7233521ed5c4bd}





#### `public def testFetchFeedShortStringTensor(self)` {#classworkspace__test_1_1_test_workspace_1ac4b1ab367901b9396a912cb036fb922a}





#### `public def testFetchFeedPlainString(self)` {#classworkspace__test_1_1_test_workspace_1a147354df249e1d566a3f9d5143596ded}





#### `public def testFetchBlobs(self)` {#classworkspace__test_1_1_test_workspace_1ab7cadc190b7309697d8f29e713bbc0b3}





#### `public def testFetchFeedViaBlobDict(self)` {#classworkspace__test_1_1_test_workspace_1a13025dbec005624d68b17985ec4da793}





# class `workspace_test::TestWorkspaceGPU` {#classworkspace__test_1_1_test_workspace_g_p_u}

```
class workspace_test::TestWorkspaceGPU
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  net` |
`public def setUp(self)` |
`public def testFetchBlobGPU(self)` |
`public def testDefaultGPUID(self)` |
`public def testGetCudaPeerAccessPattern(self)` |

## Members

#### `public  net` {#classworkspace__test_1_1_test_workspace_g_p_u_1a71550538c352615a5e65725ac79b4706}





#### `public def setUp(self)` {#classworkspace__test_1_1_test_workspace_g_p_u_1a056ff7b6fb246e369f10b28245db0d5d}





#### `public def testFetchBlobGPU(self)` {#classworkspace__test_1_1_test_workspace_g_p_u_1af4c55bdd922e4e0ec4cd741e4738d5b6}





#### `public def testDefaultGPUID(self)` {#classworkspace__test_1_1_test_workspace_g_p_u_1aeeaecfa8015e1b0e97784d8cd2186fa2}





#### `public def testGetCudaPeerAccessPattern(self)` {#classworkspace__test_1_1_test_workspace_g_p_u_1a15134c3d7cd125b26bae172d0fe3e8fc}





# class `workspace_test::TestWorkspaceMKLDNN` {#classworkspace__test_1_1_test_workspace_m_k_l_d_n_n}

```
class workspace_test::TestWorkspaceMKLDNN
  : public test_util.TestCase
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public def testFeedFetchBlobMKLDNN(self)` |

## Members

#### `public def testFeedFetchBlobMKLDNN(self)` {#classworkspace__test_1_1_test_workspace_m_k_l_d_n_n_1ad86f4726cecb7d3971775c01d722173c}





# struct `caffe2::python::BlobFetcherBase::FetchedBlob` {#structcaffe2_1_1python_1_1_blob_fetcher_base_1_1_fetched_blob}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public pybind11::object obj` |
`public bool copied` |

## Members

#### `public pybind11::object obj` {#structcaffe2_1_1python_1_1_blob_fetcher_base_1_1_fetched_blob_1ae37bff156ded1ac3e333f33cdd007cef}





#### `public bool copied` {#structcaffe2_1_1python_1_1_blob_fetcher_base_1_1_fetched_blob_1a604335a6f06c79ae73a4aff9aa5cf24b}
