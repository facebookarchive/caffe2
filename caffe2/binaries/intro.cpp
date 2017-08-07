#include "caffe2/core/init.h"
#include "caffe2/core/operator_gradient.h"

#include "caffe2/utils/myutil/print.h"

namespace caffe2 {

void run() {
  std::cout << std::endl;
  std::cout << "## Caffe2 Intro Tutorial ##" << std::endl;
  std::cout << std::endl;

  // >>> import numpy as np
  Workspace workspace;

  // >>> x = np.random.rand(4, 3, 2)
  std::vector<float> x(4 * 3 * 2);
  for (auto &v: x) {
    v = (float)rand() / RAND_MAX;
  }

  // >>> print(x)
  print(x, "x");

  // >>> workspace.FeedBlob("my_x", x)
  {
    auto tensor = workspace.CreateBlob("my_x")->GetMutable<TensorCPU>();
    auto value = TensorCPU({ 4, 3, 2 }, x, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> x2 = workspace.FetchBlob("my_x")
  // >>> print(x2)
  {
    const auto blob = workspace.GetBlob("my_x");
    print(*blob, "my_x");
  }

  // >>> data = np.random.rand(16, 100).astype(np.float32)
  std::vector<float> data(16 * 100);
  for (auto& v: data) {
    v = (float)rand() / RAND_MAX;
  }

  // >>> label = (np.random.rand(16) * 10).astype(np.int32)
  std::vector<int> label(16);
  for (auto& v: label) {
    v = 10 * rand() / RAND_MAX;
  }

  // >>> workspace.FeedBlob("data", data)
  {
    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
    auto value = TensorCPU({ 16, 100 }, data, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> workspace.FeedBlob("label", label)
  {
    auto tensor = workspace.CreateBlob("label")->GetMutable<TensorCPU>();
    auto value = TensorCPU({ 16 }, label, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> m = model_helper.ModelHelper(name="my first net")
  NetDef initModel;
  initModel.set_name("my first net_init");
  NetDef predictModel;
  predictModel.set_name("my first net");

  // >>> weight = m.param_initModel.XavierFill([], 'fc_w', shape=[10, 100])
  {
    auto op = initModel.add_op();
    op->set_type("XavierFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(10);
    arg->add_ints(100);
    op->add_output("fc_w");
  }

  // >>> bias = m.param_initModel.ConstantFill([], 'fc_b', shape=[10, ])
  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("shape");
    arg->add_ints(10);
    op->add_output("fc_b");
  }

  std::vector<OperatorDef *> gradient_ops;

  // >>> fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
  {
    auto op = predictModel.add_op();
    op->set_type("FC");
    op->add_input("data");
    op->add_input("fc_w");
    op->add_input("fc_b");
    op->add_output("fc1");
    gradient_ops.push_back(op);
  }

  // >>> pred = m.net.Sigmoid(fc_1, "pred")
  {
    auto op = predictModel.add_op();
    op->set_type("Sigmoid");
    op->add_input("fc1");
    op->add_output("pred");
    gradient_ops.push_back(op);
  }

  // >>> [softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])
  {
    auto op = predictModel.add_op();
    op->set_type("SoftmaxWithLoss");
    op->add_input("pred");
    op->add_input("label");
    op->add_output("softmax");
    op->add_output("loss");
    gradient_ops.push_back(op);
  }

  // >>> m.AddGradientOperators([loss])
  {
    auto op = predictModel.add_op();
    op->set_type("ConstantFill");
    auto arg = op->add_arg();
    arg->set_name("value");
    arg->set_f(1.0);
    op->add_input("loss");
    op->add_output("loss_grad");
    op->set_is_gradient_op(true);
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  for (auto op: gradient_ops) {
    vector<GradientWrapper> output(op->output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op->output(i) + "_grad";
    }
    GradientOpsMeta meta = GetGradientForOp(*op, output);
    auto grad = predictModel.add_op();
    grad->CopyFrom(meta.ops_[0]);
    grad->set_is_gradient_op(true);
  }

  // >>> print(str(m.net.Proto()))
  // std::cout << std::endl;
  // print(predictModel);

  // >>> print(str(m.param_init_net.Proto()))
  // std::cout << std::endl;
  // print(initModel);

  // >>> workspace.RunNetOnce(m.param_init_net)
  auto initNet = CreateNet(initModel, &workspace);
  initNet->Run();

  // >>> workspace.CreateNet(m.net)
  auto predictNet = CreateNet(predictModel, &workspace);

  // >>> for j in range(0, 100):
  for (auto i = 0; i < 100; i++) {

    // >>> data = np.random.rand(16, 100).astype(np.float32)
    std::vector<float> data(16 * 100);
    for (auto& v: data) {
      v = (float)rand() / RAND_MAX;
    }

    // >>> label = (np.random.rand(16) * 10).astype(np.int32)
    std::vector<int> label(16);
    for (auto& v: label) {
      v = 10 * rand() / RAND_MAX;
    }

    // >>> workspace.FeedBlob("data", data)
    {
      auto tensor = workspace.GetBlob("data")->GetMutable<TensorCPU>();
      auto value = TensorCPU({ 16, 100 }, data, NULL);
      tensor->ShareData(value);
    }

    // >>> workspace.FeedBlob("label", label)
    {
      auto tensor = workspace.GetBlob("label")->GetMutable<TensorCPU>();
      auto value = TensorCPU({ 16 }, label, NULL);
      tensor->ShareData(value);
    }

    // >>> workspace.RunNet(m.name, 10)   # run for 10 times
    for (auto j = 0; j < 10; j++) {
      predictNet->Run();
      // std::cout << "step: " << i << " loss: ";
      // print(*(workspace.GetBlob("loss")));
      // std::cout << std::endl;
    }
  }

  std::cout << std::endl;

  // >>> print(workspace.FetchBlob("softmax"))
  print(*(workspace.GetBlob("softmax")), "softmax");

  std::cout << std::endl;

  // >>> print(workspace.FetchBlob("loss"))
  print(*(workspace.GetBlob("loss")), "loss");
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
