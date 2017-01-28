# Custom Operators

Did you check out the wide array of [Operators](operators.html) already provided in Caffe2? Still want to roll your own operator? Read on, but don't forget to contribute your fancy new operator back to the project!

## Writing a Basic Operator

Almost every operator will use both a .cc file for the registering of the operator and a .h file for the actual implementation, though this can vary across operators. For example, in some cases, the implementation may be coded in the .cc file. In addition, several operators also have GPU/CUDA implementations, which are stored in .cu files.

We begin with the .cc file. Consider the operator defined in fully_connected_op.cc:

**[fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)**

  ```
  #include "caffe2/operators/fully_connected_op.h"

  namespace caffe2 {
  namespace {

  REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<float, CPUContext>);
  REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<float, CPUContext>);
  ```

  At first, the names of the operators and the corresponding gradient operator is registered with this macro; this binds the function FC whenever used in Python to the FullyConnectedOp operator, where the `float` and `CPUContext` dictate what kind of input type is expected, and what the context is; this value can be either `CPUContext` or `CUDAContext` depending on whether this is used on a CPU or GPU device.

  Now, we consider the operator schema:

**[fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)**

  ```
  OPERATOR_SCHEMA(FC)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  Computes the result of passing an input vector X into a fully connected
  layer with 2D weight matrix W and 1D bias vector b.

  The layer computes Y = X * W + b, where X has size (M x K), W has size (K x N),
  b has size (N), and Y has size (M x N), where M is the batch size. Even though b
  is 1D, it is resized to size (M x N) implicitly and added to each vector in the
  batch. These dimensions must be matched correctly, or else the operator will
  throw errors.
  )DOC")
    .Arg("axis", "(int32_t) default to 1; describes the axis of the inputs; "
    "defaults to one because the 0th axis most likely describes the batch_size")
    .Input(0, "X", "2D input of size (MxK) data")
    .Input(1, "W", "2D blob of size (KxN) containing fully connected weight "
    "matrix")
    .Input(2, "b", "1D blob containing bias vector")
    .Output(0, "Y", "1D output tensor");
    ```

The next part is the **operator schema**. This is where the this operator is told how many inputs and outputs are created, as well as the documentation for it.

`NOTE: Please document your operators as you write new ones that you would like to contribute!`

Note that there can be two kinds of inputs: the Inputs and the Args; the discerning difference is that usually, the Inputs contain the main data used in the operator, such as the weight matrices for a fully connected layer, while the Args are usually auxiliary inputs that are not involved in the raw data manipulation. Take note of how the documentation is added, as this documentation is periodically and automatically parsed by another program to update the [Caffe2 operators documentation](operators.html).

Finally, we address the gradient operators:

**[fully_connected_op.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/operators/fully_connected_op.cc)**

  ```
  OPERATOR_SCHEMA(FCGradient).NumInputs(3).NumOutputs(2, 3);

  class GetFCGradient : public GradientMakerBase {
    using GradientMakerBase::GradientMakerBase;
    vector<OperatorDef> GetGradientDefs() override {
      CHECK_EQ(def_.input_size(), 3);
      return SingleGradientDef(
          "FCGradient", "",
          vector<string>{I(0), I(1), GO(0)},
          vector<string>{GI(1), GI(2), GI(0)});
    }
  };
  REGISTER_GRADIENT(FC, GetFCGradient);
  }  // namespace
  }  // namespace caffe2
  ```

The input and output of GradientOp have to be tagged using the `GradientMakerBase::GetGradientDefs()`. By doing so, we're effectively informing Caffe2 how the inputs and outputs of the gradient operator are related to the corresponding operator. In particular, the first vector tags the inputs of the gradient operator, and the second vector tags the outputs. Note that doc scheme is not necessary for gradient operators usually, unless you see fit.

### Implementation Details
As previously mentioned, most of the implementation details are in header file in the general case. It can be the case that the implementation details are directly placed in the .cc file. For any CUDA implementations, the brunt of the logic and code is in .cu files.

### Unit Testing Caffe2 operators
It is a very good idea to write some unit tests to verify your operator is correctly implemented. There are a few helper libraries provided within Caffe2 to make sure your operator tests have good coverage.

Hypothesis (http://hypothesis.readthedocs.io/) is a very useful library for property-based testing. The key idea here is to express properties of the code under test (e.g. that it passes a gradient check, that it implements a reference function, etc), and then generate random instances and verify they satisfy these properties.

The main functions of interest are exposed on HypothesisTestCase, defined in [caffe2/python/hypothesis_test_util.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/hypothesis_test_util.py).

You should add your unit test to the folder [caffe2/caffe2/python/operator_tests/](https://github.com/caffe2/caffe2/tree/master/caffe2/python/operator_test). In that directory you can find many existing examples to work from.

The key functions are:

* `assertDeviceChecks(devices, op, inputs, outputs)`: This asserts that the operator computes the same outputs, regardless of which device it is executed on.
* `assertGradientChecks(device, op, inputs, output_, outputs_with_grads)`: This implements a standard numerical gradient checker for the operator in question.
* `assertReferenceChecks(device, op, inputs, reference)`: This runs the reference function (effectively calling reference(\*inputs), and comparing that to the output of output.
[hypothesis_test_util.py](https://github.com/caffe2/caffe2/blob/master/caffe2/python/hypothesis_test_util.py)] exposes some useful pre-built samplers.

hu.gcs - a gradient checker device (gc) and device checker devices (dc)
hu.gcs_cpu_only - a gradient checker device (gc) and device checker devices (dc) for CPU-only operators
Examples #
For a simple example.

```
@given(X=hu.tensor(), **hu.gcs)
def test_averaged_loss(self, X, gc, dc):
    op = core.CreateOperator("AveragedLoss", ["X"], ["loss"])
    self.assertDeviceChecks(dc, op, [X], [0])
    self.assertGradientChecks(gc, op, [X], 0, [0])
```

Another example that demonstrates the usage of assertReferenceChecks.

```
@given(inputs=hu.tensors(n=3),
       in_place=st.booleans(),
       beta1=st.floats(min_value=0.1, max_value=0.9),
       beta2=st.floats(min_value=0.1, max_value=0.9),
       lr=st.floats(min_value=0.1, max_value=0.9),
       iters=st.integers(min_value=1, max_value=10000),
       epsilon=st.floats(min_value=1e-5, max_value=1e-2),
       **hu.gcs)
def test_adam(self, inputs, in_place, beta1, beta2, lr, iters, epsilon,
              gc, dc):
    grad, m1, m2 = inputs
    m2 += np.abs(m2) + 0.01
    lr = np.asarray([lr], dtype=np.float32)
    iters = np.asarray([iters], dtype=np.int32)
    op = core.CreateOperator(
        "Adam",
        ["grad", "m1", "m2", "lr", "iters"],
        ["grad" if in_place else "grad_o",
         "m1" if in_place else "m1_o",
         "m2" if in_place else "m2_o"],
        beta1=beta1, beta2=beta2, epsilon=epsilon,
        device_option=gc)
    input_device_options = {"lr": hu.cpu_do, "iters": hu.cpu_do}
    self.assertDeviceChecks(
        dc, op, [grad, m1, m2, lr, iters], [0], input_device_options)

    # Reference
    def adam(grad, m1, m2, lr, iters):
        lr = lr[0]
        iters = iters[0]
        t = iters + 1
        corrected_local_rate = lr * np.sqrt(1. - np.power(beta2, t)) / \
            (1. - np.power(beta1, t))

        m1_o = (beta1 * m1) + (1. - beta1) * grad
        m2_o = (beta2 * m2) + (1. - beta2) * np.square(grad)
        grad_o = corrected_local_rate * m1_o / \
            (np.sqrt(m2_o) + epsilon)
        return (grad_o, m1_o, m2_o)

    self.assertReferenceChecks(gc, op, [grad, m1, m2, lr, iters],
                               adam, input_device_options)
```

For a fancier example that demonstrates drawing more sophisticated elements.

```
@given(prediction=hu.arrays(dims=[10, 3],
                            elements=st.floats(allow_nan=False,
                                               allow_infinity=False,
                                               min_value=0,
                                               max_value=1)),
       labels=hu.arrays(dims=[10],
                        dtype=np.int32,
                        elements=st.integers(min_value=0,
                                             max_value=3 - 1)),
        **hu.gcs)
def test_accuracy(self, prediction, labels, gc, dc):
    op = core.CreateOperator(
        "Accuracy",
        ["prediction", "labels"],
        ["accuracy"]
    )

    def op_ref(prediction, labels):
        N = prediction.shape[0]
        correct = 0
        max_ids = np.argmax(prediction, axis=1)
        for i in range(0, N):
            if max_ids[i] == labels[i]:
                correct += 1
        accuracy = correct / N
        return (accuracy,)

    self.assertReferenceChecks(
        device_option=gc,
        op=op,
        inputs=[prediction, labels],
        reference=op_ref)
```
