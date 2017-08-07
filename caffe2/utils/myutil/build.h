#ifndef BUILD_H
#define BUILD_H

#include "util/operator.h"

#include "caffe2/core/net.h"
#include "caffe2/core/operator_gradient.h"

namespace caffe2 {

static const std::set<std::string> trainable_ops({
  "Add",
  "AffineScale",
  "AveragedLoss",
  "AveragePool",
  "BackMean",
  "Concat",
  "Conv",
  "Diagonal",
  "Dropout",
  "EnsureCPUOutput",
  "FC",
  "LabelCrossEntropy",
  "LRN",
  "MaxPool",
  "Mul",
  "Relu",
  "Reshape",
  "Slice",
  "Softmax",
  "SpatialBN",
  "Sum",
});

static const std::set<std::string> non_trainable_ops({
  "Accuracy",
  "Cout",
  "ConstantFill",
  "TensorProtosDBInput",
});

static const std::set<std::string> non_inplace_ops({
  "Dropout", // TODO: see if "they" fixed dropout on cudnn
});

static const std::map<std::string, std::string> custom_gradient({
  { "EnsureCPUOutput", "CopyFromCPUInput" },
  { "CopyFromCPUInput", "EnsureCPUOutput" },
});

static const std::set<std::string> filler_ops({
  "UniformFill",
  "UniformIntFill",
  "UniqueUniformFill",
  "ConstantFill",
  "GaussianFill",
  "XavierFill",
  "MSRAFill",
  "RangeFill",
  "LengthsRangeFill",
});

static const std::string gradient_suffix("_grad");
static const std::string moment_suffix("_moment");
static const std::string reader_suffix("_reader");
static const std::string iter_name("iter");
static const std::string lr_name("lr");
static const std::string one_name("one");
static const std::string loss_name("loss");
static const std::string label_name("label");
static const std::string xent_name("xent");
static const std::string accuracy_name("accuracy");

// Helpers

void set_device_cpu_op(OperatorDef &op) {
  op.mutable_device_option()->set_device_type(CPU);
}

void set_engine_cudnn_op(OperatorDef &op) {
  op.set_engine("CUDNN");
}

void set_engine_cudnn_net(NetDef &net) {
  for (auto &op: *net.mutable_op()) {
    op.set_engine("CUDNN");
  }
}

void set_fill_to_train(NetDef &model) {
  for (auto &op: *model.mutable_op()) {
    if (op.type() == "GivenTensorFill") {
      op.mutable_arg()->RemoveLast();
      if (op.output(0).find("_w") != std::string::npos) {
        op.set_type("XavierFill");
      }
      if (op.output(0).find("_b") != std::string::npos) {
        op.set_type("ConstantFill");
      }
    }
    op.clear_name();
  }
}

void set_trainable(OperatorDef &op, bool train) {
  if (op.type() == "Dropout") {
    for (auto &arg: *op.mutable_arg()) {
      if (arg.name() == "is_test") {
        arg.set_i(!train);
      }
    }
  }
}

void set_rename_inplace(NetDef &model) {
  std::set<std::string> renames;
  for (auto &op: *model.mutable_op()) {
    if (renames.find(op.input(0)) != renames.end()) {
      op.set_input(0, op.input(0) + "_unique");
    }
    if (renames.find(op.output(0)) != renames.end()) {
      renames.erase(op.output(0));
    }
    if (op.input(0) == op.output(0)) {
      if (non_inplace_ops.find(op.type()) != non_inplace_ops.end()) {
        renames.insert(op.output(0));
        op.set_output(0, op.output(0) + "_unique");
      }
    }
  }
}

OperatorDef *add_gradient_op(NetDef &model, OperatorDef &op) {
  auto grad = model.add_op();
  if (custom_gradient.find(op.type()) == custom_gradient.end()) {
    vector<GradientWrapper> output(op.output_size());
    for (auto i = 0; i < output.size(); i++) {
      output[i].dense_ = op.output(i) + gradient_suffix;
    }
    GradientOpsMeta meta = GetGradientForOp(op, output);
    grad->CopyFrom(meta.ops_[0]);
  } else {
    grad->set_type(custom_gradient.at(op.type()));
    for (auto arg: op.arg()) {
      auto copy = grad->add_arg();
      copy->CopyFrom(arg);
    }
    for (auto output: op.output()) {
      grad->add_input(output + gradient_suffix);
    }
    for (auto input: op.input()) {
      grad->add_output(input + gradient_suffix);
    }
  }
  grad->set_is_gradient_op(true);
  return grad;
}

std::vector<OperatorDef> collect_gradient_ops(NetDef &model) {
  std::set<std::string> external_inputs(model.external_input().begin(), model.external_input().end());
  std::vector<OperatorDef> gradient_ops;
  for (auto &op: model.op()) {
    if (trainable_ops.find(op.type()) != trainable_ops.end()) {
      gradient_ops.push_back(op);
      // std::cout << "type: " << op.type() << std::endl;
    } else if (non_trainable_ops.find(op.type()) == non_trainable_ops.end()) {
      std::cout << "unknown backprop operator type: " << op.type() << std::endl;
    }
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  return gradient_ops;
}

void add_gradient_ops(NetDef &model) {
  for (auto op: collect_gradient_ops(model)) {
    add_gradient_op(model, op);
  }
}

void add_database_ops(NetDef &init_model, NetDef &predict_model, const std::string &name, const std::string &data, const std::string &db, const std::string &db_type, int batch_size) {
  auto reader = name + reader_suffix;
  add_create_db_op(init_model, reader, db_type, db);
  predict_model.add_external_input(reader);
  add_tensor_protos_db_input_op(predict_model, reader, data, label_name, batch_size);
  // add_cout_op(predict_model, data);
  // add_cout_op(predict_model, label_name);
}

void add_test_ops(NetDef &model, const std::string &output) {
  add_accuracy_op(model, output, label_name, accuracy_name);
}

void add_xent_ops(NetDef &model, const std::string &output) {
  add_label_cross_entropy_op(model, output, label_name, xent_name);
  add_averaged_loss(model, xent_name, loss_name);
  add_accuracy_op(model, output, label_name, accuracy_name);
  add_constant_fill_with_op(model, 1.0, loss_name, loss_name + gradient_suffix);
}

std::map<std::string, int> collect_param_sizes(NetDef &model) {
  std::map<std::string, int> sizes;
  for (const auto &op: model.op()) {
    if (filler_ops.find(op.type()) != filler_ops.end()) {
      for (const auto &arg: op.arg()) {
        if (arg.name() == "shape") {
          auto size = 1;
          for (auto i: arg.ints()) {
            size *= i;
          }
          sizes[op.output(0)] = size;
        }
      }
    }
  }
  return sizes;
}

std::vector<std::string> collect_params(NetDef &model) {
  std::vector<std::string> params;
  std::set<std::string> external_inputs(model.external_input().begin(), model.external_input().end());
  for (const auto &op: model.op()) {
    if (trainable_ops.find(op.type()) != trainable_ops.end()) {
      for (const auto &input: op.input()) {
        if (external_inputs.find(input) != external_inputs.end()) {
          params.push_back(input);
        }
      }
    }
  }
  return params;
}

void add_iter_lr_ops(NetDef &init_model, NetDef &predict_model, float base_rate) {
  set_device_cpu_op(*add_constant_fill_int64_op(init_model, { 1 }, 0, iter_name));
  predict_model.add_external_input(iter_name);
  add_iter_op(predict_model, iter_name);
  add_learning_rate_op(predict_model, iter_name, lr_name, base_rate);
}

void add_sgd_ops(NetDef &init_model, NetDef &predict_model) {
  add_constant_fill_float_op(init_model, { 1 }, 1.0, one_name);
  predict_model.add_external_input(one_name);
  for (auto &param: collect_params(predict_model)) {
    add_weighted_sum_op(predict_model, { param, one_name, param + gradient_suffix, lr_name }, param);
  }
}

void add_momentum_ops(NetDef &init_model, NetDef &predict_model) {
  auto sizes = collect_param_sizes(init_model);
  for (auto &param: collect_params(predict_model)) {
    auto size = sizes[param];
    add_constant_fill_float_op(init_model, { size }, 0.0, param + moment_suffix);
    predict_model.add_external_input(param + moment_suffix);
    add_momentum_sgd_op(predict_model, param, param + moment_suffix, param + gradient_suffix, lr_name);
  }
}

void add_adagrad_ops(NetDef &init_model, NetDef &predict_model) {
  auto sizes = collect_param_sizes(init_model);
  for (auto &param: collect_params(predict_model)) {
    auto size = sizes[param];
    add_constant_fill_float_op(init_model, { size }, 0.0, param + moment_suffix);
    predict_model.add_external_input(param + moment_suffix);
    add_adagrad_op(predict_model, param, param + moment_suffix, param + gradient_suffix, lr_name);
  }
}

void add_adam_ops(NetDef &init_model, NetDef &predict_model) {
  auto sizes = collect_param_sizes(init_model);
  for (auto &param: collect_params(predict_model)) {
    auto size = sizes[param];
    std::vector<std::string> moments(2);
    auto i = 0;
    for (auto &moment: moments) {
      moment = param + moment_suffix + "_" + std::to_string(++i);
      add_constant_fill_float_op(init_model, { size }, 0.0, moment);
      predict_model.add_external_input(moment);
    }
    add_adam_op(predict_model, param, moments, param + gradient_suffix, lr_name, iter_name);
  }
}

void add_optimizer_ops(NetDef &init_model, NetDef &predict_model, std::string &optimizer) {
  if (optimizer == "sgd") {
    add_sgd_ops(init_model, predict_model);
  } else if (optimizer == "momentum") {
    add_momentum_ops(init_model, predict_model);
  } else if (optimizer == "adagrad") {
    add_adagrad_ops(init_model, predict_model);
  } else if (optimizer == "adam") {
    add_adam_ops(init_model, predict_model);
  } else {
    LOG(FATAL) << "~ optimizer type not supported: " << optimizer;
  }
}

void add_train_ops(NetDef &init_model, NetDef &predict_model, const std::string &output, float base_rate, std::string &optimizer) {
  add_xent_ops(predict_model, output);
  add_gradient_ops(predict_model);
  add_iter_lr_ops(init_model, predict_model, base_rate);
  add_optimizer_ops(init_model, predict_model, optimizer);
}

}  // namespace caffe2

#endif  // BUILD_H
