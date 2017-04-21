---
docid: workspace
title: Workspace Class
layout: docs
permalink: /docs/workspace.html
---
These two classes are highlighted as they're commonly used in examples and tutorials. Workspace is a key component of Caffe2 while [CNNModelHelper](workspace.html#cnnmodelhelper) is useful in quickly creating CNNs.

## Workspace
**Code:** [workspace.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/workspace.py)
| **API Docs:** [Module caffe2.python.workspace](/doxygen-python/html/namespaceworkspace.html#details)

Workspace is a class that holds all the related objects created during runtime:

1. all blobs, and
2. all instantiated networks. It is the owner of all these objects and deals with the scaffolding logistics.

### Example Usage

A workspace is created for you whenever you create nets or handle blobs of data with Caffe2. Calling `workspace` initializes an empty workspace with the given root folder. For any operators that are going to interface with the file system, such as load operators, they will write things under this root folder given by the workspace.

```python
from caffe2.proto.caffe2_pb2 import NetDef
from caffe2.python import workspace

init_net = NetDef()
init_net.ParseFromString(open(protobuf_data))
predict_net = NetDef()
predict_net.ParseFromString(open(protobuf_data))
print predict_net.name //would reveal the name field

workspace.CreateNet(init_net)
workspace.CreateNet(predict_net)
workspace.RunNet(predict_net)
```

You will notice that the CreateNet methods require a net. These are NetDefs that are created from data from datasets, pre-trained models, protobuf data describing the nets and models need to be instantiated as prototype objects inheriting the characteristics from Caffe2's protobuf spec. These will then be managed in the workspace. Also, note that we've loaded the init_net first, which will setup references to blobs that should be filled by predict_net once you load that net.

For more examples of the basics of workspaces, check out the [basics tutorial](../tutorials/Basics.ipynb).

### CreateNet

Creates an empty net unless blobs are passed in.

|Inputs|
|------
| net| required NetDef
| input_blobs

|Outputs|
|------
| net object

```python
workspace.CreateNet(net_def, input_blobs)
```

### FeedBlob

Feeds a blob into the workspace.

|Inputs|
|------
| name| the name of the blob
| arr| either a TensorProto object or a numpy array object to be fed into the workspace
| device_option (optional)| the device option to feed the data with

Returns|
|------
| True or False, stating whether the feed is successful.

```python
workspace.FeedBlob(name, arr, device_option=None)
```

### FetchBlob

Fetches a blob from the workspace.

|Inputs|
|------
| name| the name of the blob - a string or a BlobReference

Returns|
|------
| Fetched blob (numpy array or string) if successful

```python
workspace.FetchBlob(name)
```

### FetchBlobs

Fetches a list of blobs from the workspace.

|Inputs|
|------
| names| list of names of blobs - strings or BlobReferences

Returns|
|------
| list of fetched blobs

```python
workspace.FetchBlob(name)
```

### GetNameScope

Returns the current namescope string. To be used to fetch blobs.

|Outputs|
|------
| namescope

### InferShapesAndTypes

Infers the shapes and types for the specified nets.

|Inputs|
|------
| nets| the list of nets
| blob_dimensions (optional)| a dictionary of blobs and their dimensions. If not specified, the workspace blobs are used.

Returns|
|------
| A tuple of (shapes, types) dictionaries keyed by blob name.

```python
workspace.InferShapesAndTypes(nets, blob_dimensions)
```

### ResetWorkSpace

Resets the workspace, and if root_folder is empty it will keep the current folder setting.

|Inputs|
|------
| root_folder| string

|Outputs|
|------
| workspace object

```python
workspace.ResetWorkspace(root_folder)
```

### RunNet

Runs a given net.

|Inputs|
|------
| name| the name of the net, or a reference to the net.
| num_iter| number of iterations to run, defaults to 1

Returns|
|------
| True or an exception.

```python
workspace.RunNet(name, num_iter)
```

### RunNetOnce

Takes in a net and will run the net one time.

|Inputs|
|------
| net

```python
workspace.RunNetOnce(net)
```

### RunOperatorOnce

Will execute a single operator.

|Inputs|
|------
| operator

```python
workspace.RunOperatorOnce(operator)
```

### RunOperatorsOnce

Will execute a set of operators.

|Inputs|
|------
| operators list

|Outputs|
|------
| Boolean on success
| False if any op fails

```python
workspace.RunOperatorOnce(operators)
```

### RunPlan

Construct a plan of multiple execution steps to run multiple different networks.
Use `RunPlan` to execute this plan.

|Inputs|
|------
| plan_or_step

|Outputs|
|------
| protobuf

```python
workspace.RunPlan(plan_or_step)
```  

### StartMint

Starts a mint instance.
Note: this does not work well under ipython yet. According to [https://github.com/ipython/ipython/issues/5862](https://github.com/ipython/ipython/issues/5862)

|Inputs|
|------
| root_folder| string
| port| int

Output|
|------
| mint instance

```python
workspace.StartMint(root_folder, port)
```

### StringifyBlobName

Returns the name of a blob.

|Inputs|
|------
| name

|Outputs|
|------
| name, "BlobReference"

```python
workspace.StringifyBlobName(name)
```

### StringifyNetName

Returns the name of a net.

|Inputs|
|------
| name

|Outputs|
|------
| name, "Net"

```python
workspace.StringifyNetName(name)
```

### StringifyProto

Stringify a protocol buffer object.

|Inputs|
|------
| obj| a protocol buffer object, or a Pycaffe2 object that has a Proto() function.

|Outputs|
|------
| string| the output protobuf string.

|Raises|
|------
| AttributeError| if the passed in object does not have the right attribute.

```python
workspace.StringifyProto(name)
```
