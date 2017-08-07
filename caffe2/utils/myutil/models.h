#ifndef MODELS_H
#define MODELS_H

#include "util/build.h"

#ifdef WITH_CURL
  #include <curl/curl.h>
#endif

namespace caffe2 {

const int bar_length = 64;

const std::map<std::string, std::map<std::string, std::string>> model_lookup {
  { "alexnet", {
    { "res/alexnet_predict_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/predict_net.pb" },
    { "res/alexnet_init_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_alexnet/init_net.pb" }
  }},
  { "googlenet", {
    { "res/googlenet_predict_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/predict_net.pb" },
    { "res/googlenet_init_net.pb", "https://s3.amazonaws.com/caffe2/models/bvlc_googlenet/init_net.pb" },
  }},
  { "squeezenet", {
    { "res/squeezenet_predict_net.pb", "https://s3.amazonaws.com/caffe2/models/squeezenet/predict_net.pb" },
    { "res/squeezenet_init_net.pb", "https://s3.amazonaws.com/caffe2/models/squeezenet/init_net.pb" },
  }},
  { "vgg16", {
    { "res/vgg16_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg16_predict_net.pb" },
    { "res/vgg16_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg16_init_net.pb" }
  }},
  { "vgg19", {
    { "res/vgg19_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg19_predict_net.pb" },
    { "res/vgg19_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/vgg19_init_net.pb" }
  }},
  { "resnet50", {
    { "res/resnet50_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet50_predict_net.pb" },
    { "res/resnet50_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet50_init_net.pb" }
  }},
  { "resnet101", {
    { "res/resnet101_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet101_predict_net.pb" },
    { "res/resnet101_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet101_init_net.pb" }
  }},
  { "resnet152", {
    { "res/resnet152_predict_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet152_predict_net.pb" },
    { "res/resnet152_init_net.pb", "https://github.com/leonardvandriel/caffe2_models/raw/master/model/resnet152_init_net.pb" }
  }}
};

int progress_func(const char* filename, double total_down, double current_down, double total_up, double current_up)
{
  int length = 72;
  if (total_down) {
    int prom = 1000 * current_down / total_down;
    int bar = bar_length * current_down / total_down;
    std::cerr << '\r' << std::string(bar, '#') << std::string(bar_length - bar, ' ') << std::setw(5) << ((float)prom / 10) << "%" << " " << filename << std::flush;
  }
  return 0;
}

bool download(const std::string &filename, const std::string &url) {
#ifdef WITH_CURL
  FILE *fp = fopen(filename.c_str(), "wb");
  CURL *curl = curl_easy_init();
  if (!curl) {
    return false;
  }
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, false);
  curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_func);
  curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, filename.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  CURLcode result = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  fclose(fp);
  std::cerr << '\r' << std::string(filename.length() + bar_length + 7, ' ') << '\r';
  return result == CURLE_OK;
#else
  std::cout << "model download not supported, install cURL" << std::endl;
  return false;
#endif
}

bool ensureFile(const std::string &filename, const std::string &url) {
  if (std::ifstream(filename).good()) {
    return true;
  }
  return download(filename, url);
}

bool ensureModel(const std::string &name) {
  if (model_lookup.find(name) == model_lookup.end()) {
    return false;
  }
  const std::map<std::string, std::string> &pairs = model_lookup.at(name);
  for (const auto &pair: pairs) {
    if(!ensureFile(pair.first, pair.second)) {
      return false;
    }
  }
  return true;
}

// Alexnet

OperatorDef *alexnet_add_conv_ops(NetDef &init_model, NetDef &predict_model, const std::string &prefix, const std::string &input, int in_size, int out_size, int stride, int padding, int kernel, bool group) {
  auto output = "conv" + prefix;
  add_fill_op(init_model, "XavierFill", { out_size, in_size, kernel, kernel }, output + "_w");
  predict_model.add_external_input(output + "_w");
  add_fill_op(init_model, "ConstantFill", { out_size }, output + "_b");
  predict_model.add_external_input(output + "_b");
  auto conv = add_conv_op(predict_model, input, output + "_w", output + "_b", output, stride, padding, kernel);
  if (group) add_arg(*conv, "group", 2);
  return add_relu_op(predict_model, output, output);
}

OperatorDef *alexnet_add_conv_pool(NetDef &init_model, NetDef &predict_model, const std::string &prefix, const std::string &input, int in_size, int out_size, int stride, int padding, int kernel, bool group) {
  auto output = "conv" + prefix;
  auto op = alexnet_add_conv_ops(init_model, predict_model, prefix, input, in_size, out_size, stride, padding, kernel, group);
  add_lrn_op(predict_model, output, "norm" + prefix, 5, 0.0001, 0.75, 1);
  return add_max_pool_op(predict_model, "norm" + prefix, "pool" + prefix, 2, 0, 3);
}

OperatorDef *alexnet_add_fc(NetDef &init_model, NetDef &predict_model, const std::string &prefix, const std::string &input, int in_size, int out_size, bool relu) {
  auto output = "fc" + prefix;
  add_fill_op(init_model, "XavierFill", { out_size, in_size }, output + "_w");
  predict_model.add_external_input(output + "_w");
  add_fill_op(init_model, "ConstantFill", { out_size }, output + "_b");
  predict_model.add_external_input(output + "_b");
  auto op = add_fc_op(predict_model, input, output + "_w", output + "_b", output);
  if (!relu) return op;
  add_relu_op(predict_model, output, output);
  return add_dropout_op(predict_model, output, output, 0.5);
}

void add_alexnet_model(NetDef &init_model, NetDef &predict_model) {
  predict_model.set_name("AlexNet");
  auto input = "data";
  std:string layer = input;
  predict_model.add_external_input(layer);
  layer = alexnet_add_conv_pool(init_model, predict_model, "1", layer, 3, 96, 4, 0, 11, false)->output(0);
  layer = alexnet_add_conv_pool(init_model, predict_model, "2", layer, 48, 256, 1, 2, 5, true)->output(0);
  layer = alexnet_add_conv_ops(init_model, predict_model, "3", layer, 256, 384, 1, 1, 3, false)->output(0);
  layer = alexnet_add_conv_ops(init_model, predict_model, "4", layer, 192, 384, 1, 1, 3, true)->output(0);
  layer = alexnet_add_conv_ops(init_model, predict_model, "5", layer, 192, 256, 1, 1, 3, true)->output(0);
  layer = add_max_pool_op(predict_model, layer, "pool5", 2, 0, 3)->output(0);
  layer = alexnet_add_fc(init_model, predict_model, "6", layer, 9216, 4096, true)->output(0);
  layer = alexnet_add_fc(init_model, predict_model, "7", layer, 4096, 4096, true)->output(0);
  layer = alexnet_add_fc(init_model, predict_model, "8", layer, 4096, 1000, false)->output(0);
  layer = add_softmax_op(predict_model, layer, "prob")->output(0);
  predict_model.add_external_output(layer);
  add_fill_op(init_model, "ConstantFill", { 1 }, input);
}

// GoogleNet

OperatorDef *googlenet_add_conv_ops(NetDef &init_model, NetDef &predict_model, const std::string &input, const std::string &output, int in_size, int out_size, int stride, int padding, int kernel) {
  add_fill_op(init_model, "XavierFill", { out_size, in_size, kernel, kernel }, output + "_w");
  predict_model.add_external_input(output + "_w");
  add_fill_op(init_model, "ConstantFill", { out_size }, output + "_b");
  predict_model.add_external_input(output + "_b");
  add_conv_op(predict_model, input, output + "_w", output + "_b", output, stride, padding, kernel);
  return add_relu_op(predict_model, output, output);
}

OperatorDef *googlenet_add_first(NetDef &init_model, NetDef &predict_model, const std::string &prefix, const std::string &input, int in_size, int out_size) {
  auto output = "conv" + prefix + "/";
  std::string layer = input;
  layer = googlenet_add_conv_ops(init_model, predict_model, layer, output + "7x7_s2", in_size, out_size, 2, 3, 7)->output(0);
  layer = add_max_pool_op(predict_model, layer, "pool" + prefix + "/3x3_s2", 2, 0, 3)->output(0);
  return add_lrn_op(predict_model, layer, "pool1/norm1", 5, 0.0001, 0.75, 1);
}

OperatorDef *googlenet_add_second(NetDef &init_model, NetDef &predict_model, const std::string &prefix, const std::string &input, int in_size, int out_size) {
  auto output = "conv" + prefix + "/3x3";
  std::string layer = input;
  layer = googlenet_add_conv_ops(init_model, predict_model, layer, output + "_reduce", in_size, out_size / 3, 1, 0, 1)->output(0);
  layer = googlenet_add_conv_ops(init_model, predict_model, layer, output, in_size, out_size, 1, 1, 3)->output(0);
  return add_lrn_op(predict_model, layer, "conv2/norm2", 5, 0.0001, 0.75, 1);
}

OperatorDef *googlenet_add_inception(NetDef &init_model, NetDef &predict_model, const std::string &prefix, const std::string &input, std::vector<int> sizes) {
  auto output = "inception_" + prefix + "/";
  std::string layer = input;
  std::vector<std::string> layers;
  for (int i = 0, kernel = 1; i < 3; i++, kernel += 2) {
    auto b = output + std::to_string(kernel) + "x" + std::to_string(kernel);
    if (i) {
      layer = googlenet_add_conv_ops(init_model, predict_model, input, b + "_reduce", sizes[0], sizes[kernel - 1], 1, 0, 1)->output(0);
    }
    layers.push_back(googlenet_add_conv_ops(init_model, predict_model, layer, b, sizes[kernel - 1], sizes[kernel], 1, i, kernel)->output(0));
  }
  layer = add_max_pool_op(predict_model, input, output + "pool", 1, 1, 3)->output(0);
  layers.push_back(googlenet_add_conv_ops(init_model, predict_model, layer, layer + "_proj", sizes[0], sizes[6], 1, 0, 1)->output(0));
  return add_concat_op(predict_model, layers, output + "output");
}

OperatorDef *googlenet_add_fc(NetDef &init_model, NetDef &predict_model, const std::string &prefix, const std::string &input, int in_size, int out_size) {
  auto output = "loss" + prefix + "/classifier";
  add_fill_op(init_model, "XavierFill", { out_size, in_size }, output + "_w");
  predict_model.add_external_input(output + "_w");
  add_fill_op(init_model, "ConstantFill", { out_size }, output + "_b");
  predict_model.add_external_input(output + "_b");
  add_dropout_op(predict_model, input, input, 0.4)->output(0);
  return add_fc_op(predict_model, input, output + "_w", output + "_b", output);
}

void add_googlenet_model(NetDef &init_model, NetDef &predict_model) {
  predict_model.set_name("GoogleNet");
  auto input = "data";
  std:string layer = input;
  predict_model.add_external_input(layer);
  layer = googlenet_add_first(init_model, predict_model, "1", layer, 3, 64)->output(0);
  layer = googlenet_add_second(init_model, predict_model, "2", layer, 64, 192)->output(0);
  layer = add_max_pool_op(predict_model, layer, "pool2/3x3_s2", 2, 0, 3)->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "3a", layer, { 192, 64, 96, 128, 16, 32, 32 })->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "3b", layer, { 256, 128, 128, 192, 32, 96, 64 })->output(0);
  layer = add_max_pool_op(predict_model, layer, "pool3/3x3_s2", 2, 0, 3)->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "4a", layer, { 480, 192, 96, 208, 16, 48, 64 })->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "4b", layer, { 512, 160, 112, 224, 24, 64, 64 })->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "4c", layer, { 512, 128, 128, 256, 24, 64, 64 })->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "4d", layer, { 512, 112, 144, 288, 32, 64, 64 })->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "4e", layer, { 528, 256, 160, 320, 32, 128, 128 })->output(0);
  layer = add_max_pool_op(predict_model, layer, "pool4/3x3_s2", 2, 0, 3)->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "5a", layer, { 832, 256, 160, 320, 32, 128, 128 })->output(0);
  layer = googlenet_add_inception(init_model, predict_model, "5b", layer, { 832, 384, 192, 384, 48, 128, 128 })->output(0);
  layer = add_average_pool_op(predict_model, layer, "pool5/7x7_s1", 1, 0, 7)->output(0);
  layer = googlenet_add_fc(init_model, predict_model, "3", layer, 1024, 1000)->output(0);
  layer = add_softmax_op(predict_model, layer, "prob")->output(0);
  predict_model.add_external_output(layer);
  add_fill_op(init_model, "ConstantFill", { 1 }, input);
}

// All

void add_model(const std::string &name, NetDef &init_model, NetDef &predict_model) {
  if (name == "alexnet") {
    add_alexnet_model(init_model, predict_model);
  } else if (name == "googlenet") {
    add_googlenet_model(init_model, predict_model);
  } else {
    std::cerr << "model " << name << " not implemented";
  }
}

}  // namespace caffe2

#endif  // MODELS_H
