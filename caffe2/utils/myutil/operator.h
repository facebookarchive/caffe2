#ifndef OPERATOR_H
#define OPERATOR_H

#include "caffe2/core/operator.h"

namespace caffe2 {

// Operators - Helpers

OperatorDef *add_op(NetDef &model, const std::string &name, const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
  auto op = model.add_op();
  op->set_type(name);
  for (auto input: inputs) {
    op->add_input(input);
  }
  for (auto output: outputs) {
    op->add_output(output);
  }
  return op;
}

Argument *add_arg(OperatorDef &op, const std::string &name, int value) {
  auto arg = op.add_arg();
  arg->set_name(name);
  arg->set_i(value);
  return arg;
}

Argument *add_arg(OperatorDef &op, const std::string &name, std::vector<int> values) {
  auto arg = op.add_arg();
  arg->set_name(name);
  for (auto value: values) {
    arg->add_ints(value);
  }
  return arg;
}

Argument *add_arg(OperatorDef &op, const std::string &name, float value) {
  auto arg = op.add_arg();
  arg->set_name(name);
  arg->set_f(value);
  return arg;
}

Argument *add_arg(OperatorDef &op, const std::string &name, std::vector<float> values) {
  auto arg = op.add_arg();
  arg->set_name(name);
  for (auto value: values) {
    arg->add_floats(value);
  }
  return arg;
}

Argument *add_arg(OperatorDef &op, const std::string &name, const std::string &value) {
  auto arg = op.add_arg();
  arg->set_name(name);
  arg->set_s(value);
  return arg;
}

Argument *add_arg(OperatorDef &op, const std::string &name, const std::vector<std::string> &values) {
  auto arg = op.add_arg();
  arg->set_name(name);
  for (auto value: values) {
    arg->add_strings(value);
  }
  return arg;
}

// Operators - I/O

OperatorDef *add_create_db_op(NetDef &model, const std::string &reader, const std::string &db_type, const std::string &db_path) {
  auto op = model.add_op();
  op->set_type("CreateDB");
  auto arg1 = op->add_arg();
  arg1->set_name("db_type");
  arg1->set_s(db_type);
  auto arg2 = op->add_arg();
  arg2->set_name("db");
  arg2->set_s(db_path);
  op->add_output(reader);
  return op;
}

OperatorDef *add_tensor_protos_db_input_op(NetDef &model, const std::string &reader, const std::string &data, const std::string &label, int batch_size) {
  auto op = model.add_op();
  op->set_type("TensorProtosDBInput");
  auto arg = op->add_arg();
  arg->set_name("batch_size");
  arg->set_i(batch_size);
  op->add_input(reader);
  op->add_output(data);
  op->add_output(label);
  return op;
}

OperatorDef *add_cout_op(NetDef &model, const std::vector<std::string> &params) {
  auto op = model.add_op();
  op->set_type("Cout");
  for (auto param: params) {
    op->add_input(param);
  }
  return op;
}

OperatorDef *add_ensure_cpu_output_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = model.add_op();
  op->set_type("EnsureCPUOutput");
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_copy_from_cpu_input_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = model.add_op();
  op->set_type("CopyFromCPUInput");
  op->add_input(input);
  op->add_output(output);
  return op;
}

// Operators - Initialization

OperatorDef *add_fill_op(NetDef &model, const std::string type, const std::vector<int> &shape, const std::string &param) {
  auto op = model.add_op();
  op->set_type(type);
  auto arg = op->add_arg();
  arg->set_name("shape");
  for (auto dim: shape) {
    arg->add_ints(dim);
  }
  op->add_output(param);
  return op;
}

OperatorDef *add_uniform_fill_float_op(NetDef &model, const std::vector<int> &shape, float min, float max, const std::string &param) {
  auto op = add_fill_op(model, "UniformFill", shape, param);
  auto arg1 = op->add_arg();
  arg1->set_name("min");
  arg1->set_f(min);
  auto arg2 = op->add_arg();
  arg2->set_name("max");
  arg2->set_f(max);
  return op;
}

OperatorDef *add_constant_fill_float_op(NetDef &model, const std::vector<int> &shape, float value, const std::string &param) {
  auto op = add_fill_op(model, "ConstantFill", shape, param);
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  return op;
}

OperatorDef *add_constant_fill_int64_op(NetDef &model, const std::vector<int> &shape, int64_t value, const std::string &param) {
  auto op = add_fill_op(model, "ConstantFill", shape, param);
  auto arg1 = op->add_arg();
  arg1->set_name("value");
  arg1->set_i(value);
  auto arg2 = op->add_arg();
  arg2->set_name("dtype");
  arg2->set_i(TensorProto_DataType_INT64);
  return op;
}

OperatorDef *add_constant_fill_int32_op(NetDef &model, const std::vector<int> &shape, int32_t value, const std::string &param) {
  auto op = add_fill_op(model, "ConstantFill", shape, param);
  auto arg1 = op->add_arg();
  arg1->set_name("value");
  arg1->set_i(value);
  auto arg2 = op->add_arg();
  arg2->set_name("dtype");
  arg2->set_i(TensorProto_DataType_INT32);
  return op;
}

OperatorDef *add_constant_fill_with_op(NetDef &model, float value, const std::string &input, const std::string &output) {
  auto op = model.add_op();
  op->set_type("ConstantFill");
  auto arg = op->add_arg();
  arg->set_name("value");
  arg->set_f(value);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_vector_fill_op(NetDef &model, const std::vector<int> &values, const std::string &name) {
  auto op = model.add_op();
  op->set_type("GivenTensorFill");
  auto arg1 = op->add_arg();
  arg1->set_name("shape");
  arg1->add_ints(values.size());
  auto arg2 = op->add_arg();
  arg2->set_name("values");
  for (auto v: values) {
    arg2->add_ints(v);
  }
  auto arg3 = op->add_arg();
  arg3->set_name("dtype");
  arg3->set_i(TensorProto_DataType_INT32);
  op->add_output(name);
  return op;
}

OperatorDef *add_given_tensor_fill_op(NetDef &model, const TensorCPU &tensor, const std::string &name) {
  auto op = model.add_op();
  op->set_type("GivenTensorFill");
  auto arg1 = op->add_arg();
  arg1->set_name("shape");
  for (auto dim: tensor.dims()) {
    arg1->add_ints(dim);
  }
  auto arg2 = op->add_arg();
  arg2->set_name("values");
  const auto& data = tensor.data<float>();
  for (auto i = 0; i < tensor.size(); ++i) {
    arg2->add_floats(data[i]);
  }
  op->add_output(name);
  return op;
}

// Operators - Prediction

OperatorDef *add_conv_op(NetDef &model, const std::string &input, const std::string &w, const std::string &b, const std::string &output, int stride, int padding, int kernel) {
  auto op = add_op(model, "Conv", { input, w, b }, { output });
  add_arg(*op, "stride", stride);
  add_arg(*op, "pad", padding);
  add_arg(*op, "kernel", kernel);
  return op;
}

OperatorDef *add_relu_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = add_op(model, "Relu", { input }, { output });
  return op;
}

OperatorDef *add_lrn_op(NetDef &model, const std::string &input, const std::string &output, int size, float alpha, float beta, float bias, const std::string &order = "NCHW") {
  auto op = add_op(model, "LRN", { input }, { output, "_" + output + "_scale" });
  add_arg(*op, "size", size);
  add_arg(*op, "alpha", alpha);
  add_arg(*op, "beta", beta);
  add_arg(*op, "bias", bias);
  add_arg(*op, "order", order);
  return op;
}

OperatorDef *add_pool_op(NetDef &model, const std::string &type, const std::string &input, const std::string &output, int stride, int padding, int kernel, const std::string &order = "NCHW") {
  auto op = add_op(model, type, { input }, { output });
  add_arg(*op, "stride", stride);
  add_arg(*op, "pad", padding);
  add_arg(*op, "kernel", kernel);
  add_arg(*op, "order", order);
  add_arg(*op, "legacy_pad", 3);
  return op;
}

OperatorDef *add_max_pool_op(NetDef &model, const std::string &input, const std::string &output, int stride, int padding, int kernel, const std::string &order = "NCHW") {
  auto op = add_pool_op(model, "MaxPool", input, output, stride, padding, kernel, order);
  return op;
}

OperatorDef *add_average_pool_op(NetDef &model, const std::string &input, const std::string &output, int stride, int padding, int kernel, const std::string &order = "NCHW") {
  auto op = add_pool_op(model, "AveragePool", input, output, stride, padding, kernel, order);
  return op;
}

OperatorDef *add_fc_op(NetDef &model, const std::string &input, const std::string &w, const std::string &b, const std::string &output) {
  auto op = add_op(model, "FC", { input, w, b }, { output });
  return op;
}

OperatorDef *add_dropout_op(NetDef &model, const std::string &input, const std::string &output, float ratio) {
  auto op = add_op(model, "Dropout", { input }, { output, "_" + output + "_mask" });
  add_arg(*op, "ratio", ratio);
  add_arg(*op, "is_test", 1); // TODO
  return op;
}

OperatorDef *add_softmax_op(NetDef &model, const std::string &input, const std::string &output) {
  auto op = add_op(model, "Softmax", { input }, { output });
  return op;
}

OperatorDef *add_concat_op(NetDef &model, const std::vector<std::string> &inputs, const std::string &output, const std::string &order = "NCHW") {
  auto op = add_op(model, "Concat", inputs, { output, "_" + output + "_dims" });
  add_arg(*op, "order", order);
  return op;
}

// Operators - Training

OperatorDef *add_accuracy_op(NetDef &model, const std::string &pred, const std::string &label, const std::string &accuracy) {
  auto op = model.add_op();
  op->set_type("Accuracy");
  op->add_input(pred);
  op->add_input(label);
  op->add_output(accuracy);
  return op;
}

OperatorDef *add_label_cross_entropy_op(NetDef &model, const std::string &pred, const std::string &label, const std::string &xent) {
  auto op = model.add_op();
  op->set_type("LabelCrossEntropy");
  op->add_input(pred);
  op->add_input(label);
  op->add_output(xent);
  return op;
}

OperatorDef *add_averaged_loss(NetDef &model, const std::string &input, const std::string &loss) {
  auto op = model.add_op();
  op->set_type("AveragedLoss");
  op->add_input(input);
  op->add_output(loss);
  return op;
}

OperatorDef *add_diagonal_op(NetDef &model, const std::string &input, const std::string &diagonal, const std::vector<int> &offset) {
  auto op = model.add_op();
  op->set_type("Diagonal");
  auto arg = op->add_arg();
  arg->set_name("offset");
  for (auto o: offset) {
    arg->add_ints(o);
  }
  op->add_input(input);
  op->add_output(diagonal);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_back_mean_op(NetDef &model, const std::string &input, const std::string &mean, int count = 1) {
  auto op = model.add_op();
  op->set_type("BackMean");
  auto arg = op->add_arg();
  arg->set_name("count");
  arg->set_i(count);
  op->add_input(input);
  op->add_output(mean);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_mean_stdev_op(NetDef &model, const std::string &input, const std::string &mean, const std::string &scale) {
  auto op = model.add_op();
  op->set_type("MeanStdev");
  op->add_input(input);
  op->add_output(mean);
  op->add_output(scale);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_affine_scale_op(NetDef &model, const std::string &input, const std::string &mean, const std::string &scale, const std::string &transformed, bool inverse = false) {
  auto op = model.add_op();
  op->set_type("AffineScale");
  auto arg = op->add_arg();
  arg->set_name("inverse");
  arg->set_i(inverse);
  op->add_input(input);
  op->add_input(mean);
  op->add_input(scale);
  op->add_output(transformed);
  op->mutable_device_option()->set_device_type(CPU);
  return op;
}

OperatorDef *add_slice_op(NetDef &model, const std::string &input, const std::string &output, const std::vector<std::pair<int, int>> &ranges) {
  auto op = model.add_op();
  op->set_type("Slice");
  auto arg1 = op->add_arg();
  arg1->set_name("starts");
  auto arg2 = op->add_arg();
  arg2->set_name("ends");
  for (auto r: ranges) {
    arg1->add_ints(r.first);
    arg2->add_ints(r.second);
  }
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_reshape_op(NetDef &model, const std::string &input, const std::string &output, const std::vector<int> &shape) {
  auto op = model.add_op();
  op->set_type("Reshape");
  auto arg = op->add_arg();
  arg->set_name("shape");
  for (auto s: shape) {
    arg->add_ints(s);
  }
  op->add_input(input);
  op->add_output(output);
  op->add_output("_");
  return op;
}

OperatorDef *add_weighted_sum_op(NetDef &model, const std::vector<std::string> &inputs, const std::string &sum) {
  auto op = model.add_op();
  op->set_type("WeightedSum");
  for (const auto &input: inputs) {
    op->add_input(input);
  }
  op->add_output(sum);
  return op;
}

OperatorDef *add_momentum_sgd_op(NetDef &model, const std::string &param, const std::string &moment, const std::string &grad, const std::string &lr) {
  auto op = model.add_op();
  op->set_type("MomentumSGDUpdate");
  op->add_input(grad);
  op->add_input(moment);
  op->add_input(lr);
  op->add_input(param);
  op->add_output(grad);
  op->add_output(moment);
  op->add_output(param);
  return op;
}

OperatorDef *add_adagrad_op(NetDef &model, const std::string &param, const std::string &moment, const std::string &grad, const std::string &lr) {
  auto op = model.add_op();
  op->set_type("Adagrad");
  op->add_input(param);
  op->add_input(moment);
  op->add_input(grad);
  op->add_input(lr);
  op->add_output(param);
  op->add_output(moment);
  return op;
}

OperatorDef *add_adam_op(NetDef &model, const std::string &param, const std::vector<std::string> &moments, const std::string &grad, const std::string &lr, const std::string &iter) {
  auto op = model.add_op();
  op->set_type("Adam");
  op->add_input(param);
  for (auto &moment: moments) {
    op->add_input(moment);
  }
  op->add_input(grad);
  op->add_input(lr);
  op->add_input(iter);
  op->add_output(param);
  for (auto &moment: moments) {
    op->add_output(moment);
  }
  return op;
}

OperatorDef *add_scale_op(NetDef &model, const std::string &input, const std::string &output, float scale) {
  auto op = model.add_op();
  op->set_type("Scale");
  auto arg = op->add_arg();
  arg->set_name("scale");
  arg->set_f(scale);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_clip_op(NetDef &model, const std::string &input, const std::string &output, float min, float max) {
  auto op = model.add_op();
  op->set_type("Clip");
  auto arg1 = op->add_arg();
  arg1->set_name("min");
  arg1->set_f(min);
  auto arg2 = op->add_arg();
  arg2->set_name("max");
  arg2->set_f(max);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_cast_op(NetDef &model, const std::string &input, const std::string &output, TensorProto::DataType type) {
  auto op = model.add_op();
  op->set_type("Cast");
  auto arg = op->add_arg();
  arg->set_name("to");
  arg->set_i(type);
  op->add_input(input);
  op->add_output(output);
  return op;
}

OperatorDef *add_iter_op(NetDef &model, const std::string &iter) {
  auto op = model.add_op();
  op->set_type("Iter");
  op->add_input(iter);
  op->add_output(iter);
  return op;
}

OperatorDef *add_learning_rate_op(NetDef &model, const std::string &iter, const std::string &rate, float base_rate) {
  auto op = model.add_op();
  op->set_type("LearningRate");
  auto arg1 = op->add_arg();
  arg1->set_name("policy");
  arg1->set_s("step");
  auto arg2 = op->add_arg();
  arg2->set_name("stepsize");
  arg2->set_i(1);
  auto arg3 = op->add_arg();
  arg3->set_name("base_lr");
  arg3->set_f(-base_rate);
  auto arg4 = op->add_arg();
  arg4->set_name("gamma");
  arg4->set_f(0.999);
  op->add_input(iter);
  op->add_output(rate);
  return op;
}

}  // namespace caffe2

#endif  // OPERATOR_H
