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

#include <iostream>
#include <memory>

#include "caffe2/core/module.h"
#include <gtest/gtest.h>
#include "caffe2/core/logging.h"

// An explicitly defined module, testing correctness when we statically link a
// module
CAFFE2_MODULE(caffe2_module_test_static, "Static module for testing.");

namespace caffe2 {

TEST(ModuleTest, StaticModule) {
  const string name = "caffe2_module_test_static";
  const auto& modules = CurrentModules();
  EXPECT_EQ(modules.count(name), 1);
  EXPECT_TRUE(HasModule(name));

  // LoadModule should not raise an error, since the module is already present.
  LoadModule(name);
  // Even a non-existing path should not cause error.
  LoadModule(name, "/does/not/exist.so");
  EXPECT_EQ(modules.count(name), 1);
  EXPECT_TRUE(HasModule(name));
}

#ifdef CAFFE2_BUILD_SHARED_LIBS
TEST(ModuleTest, DynamicModule) {
  const string name = "caffe2_module_test_dynamic";
  const auto& modules = CurrentModules();
  EXPECT_EQ(modules.count(name), 0);
  EXPECT_FALSE(HasModule(name));

  // LoadModule should load the proper module.
  LoadModule(name);
  EXPECT_EQ(modules.count(name), 1);
  EXPECT_TRUE(HasModule(name));
}
#endif

}  // namespace caffe2
