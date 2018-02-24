/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "workspace_utils.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

std::set<std::string> GetRegisteredOperators() {
  std::set<std::string> all_keys;

  // CPU operators
  for (const auto& name : CPUOperatorRegistry()->Keys()) {
    all_keys.emplace(name);
  }
  // CUDA operators
  for (const auto& name : CUDAOperatorRegistry()->Keys()) {
    all_keys.emplace(name);
  }

  return all_keys;
}
} // namespace caffe2
