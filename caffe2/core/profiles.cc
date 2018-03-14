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

#include <string>
#include <map>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "caffe2/core/profiles.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/string_utils.h"

#include <cpuinfo.h>

using google::protobuf::RepeatedPtrField;

namespace caffe2 {


namespace {

/**
 * System information needed for matching profiles.
 * Information is queries only once, and then stored in this class.
 */
class SystemInformation {
public:
  SystemInformation() :
    platform(detectPlatform()),
    socName(detectSocName())
  {
  }

  std::string platform;
  std::string socName;

private:
  /**
   * Detects host platform and returns it as a string.
   * Predicate condition PLATFORM_EQUALS is matched against this value.
   * Possible values are:
   *     "Android"
   *     "iOS"
   *     "Windows"
   *     "GNU/Linux"
   *     "macOS"
   *     "Unknown"
   */
  static inline std::string detectPlatform() {
#if defined(__ANDROID__)
    return "Android";
#elif defined(__linux__) && (defined(__GNU_LIBRARY__) || defined(__GLIBC__))
    return "GNU/Linux";
#elif defined(__APPLE__) && TARGET_OS_IPHONE
    return "iOS";
#elif defined(__APPLE__) && TARGET_OS_MAC
    return "macOS";
#elif defined(_WIN32)
    return "Windows";
#else
#warning Platform-specific implementation required
    return "Unknown";
#endif
  }

  /**
   * Detects SoC name and returns it as a string.
   * Predicate condition SOC_STARTSWITH is matched against this value.
   * SoC name is the package name for package #0 reported by cpuinfo. Examples:
   *     "Qualcomm Snapdragon 845"
   *     "Qualcomm MSM8998"
   *     "Samsung Exynos 9810"
   *     "MediaTek MT6771"
   */
  static inline std::string detectSocName() {
    const cpuinfo_package* package = cpuinfo_get_package(0);
    return std::string(package->name,
      strnlen(package->name, CPUINFO_PACKAGE_NAME_MAX));
  }
};

/**
 * Checks if all conditions specified in the predicate are satisfied given
 * the system information. If any predicate is not satisfied, the function
 * skips remaining predicates and exists early.
 *
 * @param sysInfo - cached information about the system (plaftorm, SoC name).
 * @param predicates - a list of predicated to check.
 *
 * @retval true - if the list of predicates is empty or all predicates are
 *    satisfied.
 * @retval false - if the list of predicates is non-empty and any predicate is
 *    not satisfied.
 */
inline bool checkPredicates(
  const SystemInformation& sysInfo,
  const RepeatedPtrField<Profile_Predicate>& predicates)
{
  for (const Profile_Predicate& predicate : predicates) {
    switch (predicate.condition()) {
      case Profile::PLATFORM_EQUALS:
        if ((sysInfo.platform == predicate.value()) != predicate.invert()) {
          return false;
        }
        break;
      case Profile::SOC_STARTSWITH:
        if (startsWith(sysInfo.socName, predicate.value()) !=
            predicate.invert())
        {
          return false;
        }
        break;
    }
  }
  return true;
}

} /* namespace */

void ApplyProfiles(NetDef& netDef) {
  if (netDef.profile_size() == 0) {
    return;
  }

  if (!cpuinfo_initialize()) {
    LOG(ERROR) <<
      "profiles are not applied: could not initialize CPU information";
    netDef.clear_profile();
    return;
  }

  /*
   * Create a map of operator names to their indices in NetDef to speed up
   * applying predicates to the layers.
   */
  std::map<std::string, int> operatorMap;
  for (int i = 0; i < netDef.op_size(); i++) {
    const OperatorDef& op = netDef.op(i);
    if (op.has_name()) {
      operatorMap[op.name()] = i;
    }
  }

  /* Query and cache system information for predicate matching */
  SystemInformation sysInfo;

  for (const Profile& profile : netDef.profile()) {
    if (checkPredicates(sysInfo, profile.predicate())) {
      std::set<std::string> updatedArgNames;
      for (const Argument& arg : profile.arg()) {
        CAFFE_ENFORCE(arg.has_name(),
          "only named arguments are acceptable in a profile");
        updatedArgNames.insert(arg.name());
      }

      for (const std::string& target : profile.target()) {
        const auto operatorIt = operatorMap.find(target);
        if (operatorIt == operatorMap.end()) {
          LOG(WARNING) << "profile processing failed: "
            "target Operator " << target << " does not exist in the graph";
          continue;
        }

        caffe2::OperatorDef* op = netDef.mutable_op(operatorIt->second);
        CAFFE_ENFORCE(op != nullptr, "");

        /* Remove all arguments with names in updatedArgNames */
        google::protobuf::RepeatedPtrField<Argument>* args = op->mutable_arg();
        CAFFE_ENFORCE(args != nullptr, "");
        for (auto it = args->begin(); it != args->end(); ++it) {
          while (it->has_name() &&
              updatedArgNames.find(it->name()) != updatedArgNames.end())
          {
            it = args->erase(it);
            if (it == args->end()) {
              break;
            }
          }
        }

        /* Add arguments specified in the profile */
        for (const Argument& arg : profile.arg()) {
          Argument* addedArg = op->add_arg();
          CAFFE_ENFORCE(addedArg != nullptr, "");
          addedArg->CopyFrom(arg);
        }
      }
    }
  }
}

}
