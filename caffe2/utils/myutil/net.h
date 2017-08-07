#ifndef NET_H
#define NET_H

#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/core/db.h"

#include <dirent.h>
#include <sys/stat.h>

#include "util/image.h"
#include "util/cuda.h"
#include "util/print.h"
#include "util/build.h"

namespace caffe2 {

enum {
  kRunTrain = 0,
  kRunValidate = 1,
  kRunTest = 2,
  kRunNum = 3
};

static std::map<int, std::string> name_for_run({
  { kRunTrain, "train" },
  { kRunValidate, "validate" },
  { kRunTest, "test" },
});

static std::map<int, int> percentage_for_run({
  { kRunTest, 10 },
  { kRunValidate, 20 },
  { kRunTrain, 70 },
});

std::string filename_to_key(const std::string &filename) {
  // return filename;
  return std::to_string(std::hash<std::string>{}(filename)) + "_" + filename;
}

std::set<std::string> collect_layers(const NetDef &net, const std::string &layer, bool forward = false) {
  std::map<std::string, std::set<std::string>> lookup;
  for (auto &op: net.op()) {
    for (auto &input: op.input()) {
      for (auto &output: op.output()) {
        lookup[forward ? input : output].insert(forward ? output : input);
      }
    }
  }
  std::set<std::string> result;
  for (std::set<std::string> step({ layer }); step.size();) {
    std::set<std::string> next;
    for (auto &l: step) {
      if (result.find(l) == result.end()) {
        result.insert(l);
        for (auto &n: lookup[l]) {
          next.insert(n);
        }
      }
    }
    step = next;
  }
  return result;
}

void LoadLabels(const std::string &folder, const std::string &path_prefix, std::vector<std::string> &class_labels, std::vector<std::pair<std::string, int>> &image_files) {
  std::cout << "load class labels.." << std::endl;
  auto classes_text_path = path_prefix + "classes.txt";
  ;
  std::ifstream infile(classes_text_path);
  std::string line;
  while (std::getline(infile, line)) {
    if (line.size()) {
      class_labels.push_back(line);
      // std::cout << '.' << line << '.' << std::endl;
    }
  }

  std::cout << "load image folder.." << std::endl;
  auto directory = opendir(folder.c_str());
  CHECK(directory) << "~ no image folder " << folder;
  if (directory) {
    struct stat s;
    struct dirent *entry;
    while ((entry = readdir(directory))) {
      auto class_name = entry->d_name;
      auto class_path = folder + '/' + class_name;
      if (class_name[0] != '.' && class_name[0] != '_' && !stat(class_path.c_str(), &s) && (s.st_mode & S_IFDIR)) {
        auto subdir = opendir(class_path.c_str());
        if (subdir) {
          auto class_index = find(class_labels.begin(), class_labels.end(), class_name) - class_labels.begin();
          if (class_index == class_labels.size()) {
            class_labels.push_back(class_name);
          }
          while ((entry = readdir(subdir))) {
            auto image_file = entry->d_name;
            auto image_path = class_path + '/' + image_file;
            if (image_file[0] != '.' && !stat(image_path.c_str(), &s) && (s.st_mode & S_IFREG)) {
              // std::cout << class_name << ' ' <<  image_path << std::endl;
              image_files.push_back({ image_path, class_index });
            }
          }
          closedir(subdir);
        }
      }
    }
    closedir(directory);
  }
  CHECK(image_files.size()) << "~ no images found in " << folder;
  std::random_shuffle(image_files.begin(), image_files.end());
  std::cout << class_labels.size() << " labels found" << std::endl;
  std::cout << image_files.size() << " images found" << std::endl;

  std::cout << "write class labels.." << std::endl;
  std::ofstream class_file(classes_text_path);
  if (class_file.is_open()) {
    for (auto &label: class_labels) {
      class_file << label << std::endl;
    }
    class_file.close();
  }
  auto classes_header_path = path_prefix + "classes.h";
  std::ofstream labels_file(classes_header_path.c_str());
  if (labels_file.is_open()) {
    labels_file << "const char * retrain_classes[] {";
    bool first = true;
    for (auto &label: class_labels) {
      if (first) {
        first = false;
      } else {
        labels_file << ',';
      }
      labels_file << std::endl << '"' << label << '"';
    }
    labels_file << std::endl << "};" << std::endl;
    labels_file.close();
  }
}

void WriteBatch(Workspace &workspace, NetBase *predict_net, std::string &input_name, std::string &output_name, std::vector<std::pair<std::string, int>> &batch_files, std::unique_ptr<db::DB> *database, int size_to_fit) {
  std::unique_ptr<db::Transaction> transaction[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    transaction[i] = database[i]->NewTransaction();
  }

  std::vector<std::string> filenames;
  for (auto &pair: batch_files) {
    filenames.push_back(pair.first);
  }
  std::vector<int> indices;
  auto input = readImageTensor(filenames, size_to_fit, indices);
  TensorCPU output;
  if (predict_net) {
    set_tensor_blob(*workspace.GetBlob(input_name), input);
    predict_net->Run();
    auto tensor = get_tensor_blob(*workspace.GetBlob(output_name));
    output.ResizeLike(tensor);
    output.ShareData(tensor);
  } else {
    output.ResizeLike(input);
    output.ShareData(input);
  }

  TensorProtos protos;
  TensorProto* data = protos.add_protos();
  TensorProto* label = protos.add_protos();
  data->set_data_type(TensorProto::FLOAT);
  label->set_data_type(TensorProto::INT32);
  label->add_int32_data(0);
  TensorSerializer<CPUContext> serializer;
  std::string value;
  std::vector<TIndex> dims(output.dims().begin() + 1, output.dims().end());
  auto size = output.size() / output.dim(0);
  auto output_data = output.data<float>();
  for (auto i: indices) {
    auto single = TensorCPU(dims, std::vector<float>(output_data, output_data + size), NULL);
    output_data += size;
    data->Clear();
    serializer.Serialize(single, "", data, 0, kDefaultChunkSize);
    label->set_int32_data(0, batch_files[i].second);
    protos.SerializeToString(&value);
    int percentage = 0, p = (int)(rand() * 100.0 / RAND_MAX);
    auto key = filename_to_key(batch_files[i].first);
    for (auto pair: percentage_for_run) {
      percentage += pair.second;
      if (p < percentage) {
        transaction[pair.first]->Put(key, value);
        break;
      }
    }
  }

  for (int i = 0; i < kRunNum; i++) {
    transaction[i]->Commit();
  }
}

void PreProcess(const std::vector<std::pair<std::string, int>> &image_files, const std::string *db_paths, NetDef &init_model, NetDef &predict_model, const std::string &db_type, int batch_size, int size_to_fit) {
  std::cout << "store partial prediction.." << std::endl;
  std::unique_ptr<db::DB> database[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    database[i] = db::CreateDB(db_type, db_paths[i], db::WRITE);
  }
  auto image_count = 0;
  Workspace workspace;
  auto init_net = CreateNet(init_model, &workspace);
  init_net->Run();
  auto predict_net = predict_model.external_input_size() ? CreateNet(predict_model, &workspace) : NULL;
  auto input_name = predict_model.external_input_size() ? predict_model.external_input(0) : "";
  auto output_name = predict_model.external_output_size() ? predict_model.external_output(0) : "";
  std::vector<std::pair<std::string, int>> batch_files;
  for (auto &pair: image_files) {
    auto &filename = pair.first;
    auto class_index = pair.second;
    image_count++;
    auto in_db = false;
    auto key = filename_to_key(filename);
    for (int i = 0; i < kRunNum && !in_db; i++) {
      auto cursor = database[i]->NewCursor();
      cursor->Seek(key);
      in_db |= (cursor->Valid() && cursor->key() == key);
    }
    if (!in_db) {
      batch_files.push_back({ filename, class_index });
    }
    if (batch_files.size() == batch_size) {
      std::cerr << '\r' << std::string(40, ' ') << '\r' << "pre-processing.. " << image_count << '/' << image_files.size() << " " << std::setprecision(3) << ((float)100 * image_count / image_files.size()) << "%" << std::flush;
      WriteBatch(workspace, predict_net ? predict_net.get() : NULL, input_name, output_name, batch_files, database, size_to_fit);
      batch_files.clear();
    }
  }
  if (batch_files.size() > 0) {
    WriteBatch(workspace, predict_net ? predict_net.get() : NULL, input_name, output_name, batch_files, database, size_to_fit);
  }
  for (int i = 0; i < kRunNum; i++) {
    CHECK(database[i]->NewCursor()->Valid()) << "~ database " << name_for_run[i] << " is empty";
  }
  std::cerr << '\r' << std::string(80, ' ') << '\r' << image_files.size() << " images processed" << std::endl;
}

void PreProcess(const std::vector<std::pair<std::string, int>> &image_files, const std::string *db_paths, const std::string &db_type, int size_to_fit) {
  NetDef none;
  PreProcess(image_files, db_paths, none, none, db_type, 64, size_to_fit);
}

void SplitModel(NetDef &base_init_model, NetDef &base_predict_model, const std::string &layer, NetDef &first_init_model, NetDef &first_predict_model, NetDef &second_init_model, NetDef &second_predict_model, bool force_cpu, bool inclusive = true) {
  std::cout << "split model.." << std::endl;
  std::set<std::string> static_inputs = collect_layers(base_predict_model, layer);

  // copy operators
  for (const auto &op: base_init_model.op()) {
    auto is_first = (static_inputs.find(op.output(0)) != static_inputs.end());
    auto new_op = (is_first ? first_init_model : second_init_model).add_op();
    new_op->CopyFrom(op);
  }
  for (const auto &op: base_predict_model.op()) {
    auto is_first = (static_inputs.find(op.output(0)) != static_inputs.end() && (inclusive || op.input(0) != op.output(0)));
    auto new_op = (is_first ? first_predict_model : second_predict_model).add_op();
    new_op->CopyFrom(op);
    if (!force_cpu) {
      set_engine_cudnn_op(*new_op);
    }
  }

  // copy externals
  if (first_predict_model.op().size()) {
    // first_predict_model.add_external_input(base_predict_model.external_input(0));
  }
  if (second_predict_model.op().size()) {
    // second_predict_model.add_external_input(layer);
  }
  for (const auto &output: base_init_model.external_output()) {
    auto is_first = (static_inputs.find(output) != static_inputs.end());
    if (is_first) {
      first_init_model.add_external_output(output);
    } else {
      second_init_model.add_external_output(output);
    }
  }
  for (const auto &input: base_predict_model.external_input()) {
    auto is_first = (static_inputs.find(input) != static_inputs.end());
    if (is_first) {
      first_predict_model.add_external_input(input);
    } else {
      second_predict_model.add_external_input(input);
    }
  }
  if (first_predict_model.op().size()) {
    first_predict_model.add_external_output(layer);
  }
  if (second_predict_model.op().size()) {
    second_predict_model.add_external_output(base_predict_model.external_output(0));
  }

  if (base_init_model.has_name()) {
    if (!first_init_model.has_name()) {
      first_init_model.set_name(base_init_model.name() + "_first");
    }
    if (!second_init_model.has_name()) {
      second_init_model.set_name(base_init_model.name() + "_second");
    }
  }
  if (base_predict_model.has_name()) {
    if (!first_predict_model.has_name()) {
      first_predict_model.set_name(base_predict_model.name() + "_first");
    }
    if (!second_predict_model.has_name()) {
      second_predict_model.set_name(base_predict_model.name() + "_second");
    }
  }
}

void TrainModel(NetDef &base_init_model, NetDef &base_predict_model, const std::string &layer, int out_size, NetDef &train_init_model, NetDef &train_predict_model, float base_rate, std::string &optimizer) {
  std::string last_w, last_b;
  for (const auto &op: base_predict_model.op()) {
    auto new_op = train_predict_model.add_op();
    new_op->CopyFrom(op);
    set_trainable(*new_op, true);
    if (op.type() == "FC") {
      last_w = op.input(1);
      last_b = op.input(2);
    }
  }
  set_rename_inplace(train_predict_model);
  for (const auto &op: base_init_model.op()) {
    auto &output = op.output(0);
    auto init_op = train_init_model.add_op();
    bool uniform = (output.find("_b") != std::string::npos);
    init_op->set_type(uniform ? "ConstantFill" : "XavierFill");
    for (const auto &arg: op.arg()) {
      if (arg.name() == "shape") {
        auto init_arg = init_op->add_arg();
        init_arg->set_name("shape");
        if (output == last_w) {
          init_arg->add_ints(out_size);
          init_arg->add_ints(arg.ints(1));
        } else if (output == last_b) {
          init_arg->add_ints(out_size);
        } else {
          init_arg->CopyFrom(arg);
        }
      }
    }
    init_op->add_output(output);
  }
  for (const auto &input: base_predict_model.external_input()) {
    train_predict_model.add_external_input(input);
  }
  for (const auto &output: base_predict_model.external_output()) {
    train_predict_model.add_external_output(output);
  }
  // auto op = train_init_model.add_op();
  // op->set_type("ConstantFill");
  // auto arg = op->add_arg();
  // arg->set_name("shape");
  // arg->add_ints(1);
  // op->add_output(layer);
  add_train_ops(train_init_model, train_predict_model, train_predict_model.external_output(0), base_rate, optimizer);
}

void TestModel(NetDef &base_predict_model, NetDef &test_predict_model) {
  for (const auto &op: base_predict_model.op()) {
    auto new_op = test_predict_model.add_op();
    new_op->CopyFrom(op);
    set_trainable(*new_op, false);
  }
  for (const auto &input: base_predict_model.external_input()) {
    test_predict_model.add_external_input(input);
  }
  for (const auto &output: base_predict_model.external_output()) {
    test_predict_model.add_external_output(output);
  }
  add_test_ops(test_predict_model, test_predict_model.external_output(0));
}

void CheckLayerAvailable(NetDef &model, const std::string &layer)
    {
    std::vector<std::pair<std::string, std::string>> available_layers({ { model.external_input(0), "Input" } });
    auto layer_found = (model.external_input(0) == layer);
    for (const auto &op: model.op()) {
      if (op.input(0) != op.output(0)) {
        available_layers.push_back({ op.output(0), op.type() });
        layer_found |= (op.output(0) == layer);
      }
    }
    if (!layer_found) {
      std::cout << "available layers:" << std::endl;
      for (auto &layer: available_layers) {
        std::cout << "  " << layer.first << " (" << layer.second << ")" << std::endl;
      }
      LOG(FATAL) << "~ no layer with name " << layer << " in model.";
    }
  }

}  // namespace caffe2

#endif  // NET_H
