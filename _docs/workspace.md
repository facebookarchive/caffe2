# Core Caffe2

## Workspace

Workspace is a class that holds all the related objects created during runtime:
(1) all blobs, and
(2) all instantiated networks. It is the owner of all these objects and deals with the scaffolding logistics.

### Example Usage

A workspace is created for you whenever you create nets or handle blobs of data with Caffe2. Calling `workspace` initializes an empty workspace with the given root folder. For any operators that are going to interface with the file system, such as load operators, they will write things under this root folder given by the workspace.

```python
from caffe2.python import workspace
# netdef has references to names of blobs that came from init_net
# if you ran predict_net first bad things
workspace.CreateNet(init_net)
workspace.CreateNet(predict_net)
workspace.RunNet(predict_net)
workspace.ResetWorkspace(root_folder)
```

You will notice that the CreateNet methods require a net. These are NetDefs that are created from data from datasets, pre-trained models, protobuf data describing the nets and models need to be instantiated as prototype objects inheriting the characteristics from Caffe2's protobuf spec. These will then be managed in the workspace.

* protobuf_data may have a `name` field that can be used to later call blobs and nets on the workspace. Otherwise you may define this as described below.

In the code below we create some NetDefs that can be used in the workspace. This is the setup steps that would need to be handled before running the net as shown in the code block above.

```python
from caffe2.proto.caffe2_pb2 import NetDef
i = NetDef()
i.ParseFromString(open(protobuf_data))
p = NetDef()
p.ParseFromString(open(protobuf_data))
print p.name //would reveal the name field
```

For more examples of the basics of workspaces, check out the [basics tutorial](../tutorials/basics.ipynb).

#### CreateNet

Creates an empty net unless blobs are passed in.

Inputs:
  net: required NetDef
  input_blobs

Outputs:
  net object

```python
workspace.CreateNet(net_def, input_blobs)
```

#### FeedBlob

Feeds a blob into the workspace.

Inputs:
  name: the name of the blob.
  arr: either a TensorProto object or a numpy array object to be fed into the workspace.
  device_option (optional): the device option to feed the data with.

Returns:
  True or False, stating whether the feed is successful.

```python
workspace.FeedBlob(name, arr, device_option=None)
```

#### FetchBlob

Fetches a blob from the workspace.

Inputs:
  name: the name of the blob - a string or a BlobReference

Returns:
  Fetched blob (numpy array or string) if successful

```python
workspace.FetchBlob(name)
```

#### FetchBlobs

Fetches a list of blobs from the workspace.

Inputs:
  names: list of names of blobs - strings or BlobReferences

Returns:
  list of fetched blobs

```python
workspace.FetchBlob(name)
```

#### GetNameScope

Returns the current namescope string. To be used to fetch blobs.

Outputs:
  namescope

#### InferShapesAndTypes

Infers the shapes and types for the specified nets.

Inputs:
  nets: the list of nets
  blob_dimensions (optional): a dictionary of blobs and their dimensions. If not specified, the workspace blobs are used.

Returns:
  A tuple of (shapes, types) dictionaries keyed by blob name.

```python
InferShapesAndTypes(nets, blob_dimensions)
```

#### ResetWorkSpace

Resets the workspace, and if root_folder is empty it will keep the current folder setting.

Inputs:
  root_folder: string

Outputs:
  workspace object

```python
workspace.ResetWorkspace(root_folder)
```

#### RunNet

Runs a given net.

Inputs:
  name: the name of the net, or a reference to the net.
  num_iter: number of iterations to run, defaults to 1

Returns:
  True or an exception.

```python
workspace.RunNetOnce(name, num_iter)
```

#### RunNetOnce

Takes in a net and will run the net one time.

Inputs:
  net

```python
workspace.RunNetOnce(net)
```

#### RunOperatorOnce

Will execute a single operator.

Inputs:
  operator

```python
workspace.RunOperatorOnce(operator)
```

#### RunOperatorsOnce

Will execute a set of operators.

Inputs:
  operators list

Outputs:
  Boolean on success
  False if any op fails

```python
workspace.RunOperatorOnce(operators)
```

#### RunPlan

Needs to be documented.

Inputs:
  plan_or_step

Outputs:
  protobuf

```python
workspace.RunPlan(plan_or_step)
```  

#### StartMint

Starts a mint instance.
Note: this does not work well under ipython yet. According to https://github.com/ipython/ipython/issues/5862

Inputs:
  root_folder: string
  port: int

Output:
  mint instance

```python
workspace.StartMint(root_folder, port)
```

#### StringifyBlobName

Returns the name of a blob.

Inputs:
  name

Outputs:
  name, "BlobReference"

```python
workspace.StringifyBlobName(name)
```

#### StringifyNetName

Returns the name of a net.

Inputs:
  name

Outputs:
  name, "Net"

```python
workspace.StringifyNetName(name)
```

#### StringifyProto

Stringify a protocol buffer object.

Inputs:
  obj: a protocol buffer object, or a Pycaffe2 object that has a Proto()
      function.

Outputs:
  string: the output protobuf string.

Raises:
  AttributeError: if the passed in object does not have the right attribute.

```python
workspace.StringifyProto(name)
```
