#include "caffe2/core/net_simple_async.h"
#include "caffe2/core/net.h"

#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/static_tracepoint.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

#ifdef CAFFE2_USE_NVTX
#include <nvToolsExt.h>
#endif

CAFFE2_DECLARE_bool(caffe2_use_nvtx);

namespace caffe2 {

namespace {

using Color = int32_t;
constexpr Color kRunColor = 0x0000CCFF; // blue
constexpr Color kRecordColor = 0x00FF3300; // red
constexpr Color kWaitColor = 0x0066FF33; // green

#ifdef CAFFE2_USE_NVTX

class ProfiledRange {
 public:
  ProfiledRange(const OperatorDef& def, Color color) {
    if (!FLAGS_caffe2_use_nvtx) {
      return;
    }
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = def.type().c_str();
    range_ = nvtxRangeStartEx(&eventAttrib);
    CAFFE_ENFORCE(range_, "Start range is invalid.");
  }

  ~ProfiledRange() {
    if (!FLAGS_caffe2_use_nvtx) {
      return;
    }
    nvtxRangeEnd(range_);
  }

 private:
  nvtxRangeId_t range_ = 0;
  DISABLE_COPY_AND_ASSIGN(ProfiledRange);
};

#else

class ProfiledRange {
 public:
  ProfiledRange(const OperatorDef& def, Color color) {}

 private:
  DISABLE_COPY_AND_ASSIGN(ProfiledRange);
};

#endif // ifdef CAFFE2_USE_NVTX

} // namespace

AsyncSimpleNet::AsyncSimpleNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws) {
  VLOG(1) << "Constructing AsyncSimpleNet " << net_def->name();
  const bool net_def_has_device_option = net_def->has_device_option();
  // Initialize the operators
  const DeviceOption* first_device_option = nullptr;
  const DeviceOption* current_device_option;
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ": "
            << operator_def.type();
    std::unique_ptr<OperatorBase> op{nullptr};
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def->device_option());
      op = CreateOperator(temp_def, ws, idx);
      current_device_option = &net_def->device_option();
    } else {
      op = CreateOperator(operator_def, ws, idx);
      op->set_debug_def(
          std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
      current_device_option = &operator_def.device_option();
    }
    if (!first_device_option) {
      first_device_option = current_device_option;
    } else {
      CAFFE_ENFORCE(
          IsSameDevice(*first_device_option, *current_device_option),
          "AsyncSimpleNet supports only single device networks");
    }
    operators_.emplace_back(std::move(op));
  }
  events_ = {&operators_.back()->event()};
}

bool AsyncSimpleNet::DoRunAsync() {
  StartAllObservers();

  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";
#ifdef CAFFE2_ENABLE_SDT
    const auto& op_name = op->debug_def().name().c_str();
    const auto& op_type = op->debug_def().type().c_str();
    auto* op_ptr = op.get();
    const auto& net_name = name_.c_str();
    CAFFE_SDT(operator_start, net_name, op_name, op_type, op_ptr);
#endif
    ProfiledRange r(op->debug_def(), kRunColor);
    bool res = op->RunAsync();
#ifdef CAFFE2_ENABLE_SDT
    CAFFE_SDT(operator_done, net_name, op_name, op_type, op_ptr);
#endif
    if (!res) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }
  }
  StopAllObservers();
  return true;
}

vector<float> AsyncSimpleNet::TEST_Benchmark(
    const int warmup_runs,
    const int main_runs,
    const bool run_individual) {
  std::cout << "Starting benchmark." << std::endl;
  std::cout << "Running warmup runs." << std::endl;
  CAFFE_ENFORCE(
      warmup_runs >= 0,
      "Number of warm up runs should be non negative, provided ",
      warmup_runs,
      ".");
  for (int i = 0; i < warmup_runs; ++i) {
    CAFFE_ENFORCE(Run(), "Warmup run ", i, " has failed.");
  }

  std::cout << "Main runs." << std::endl;
  CAFFE_ENFORCE(
      main_runs >= 0,
      "Number of main runs should be non negative, provided ",
      main_runs,
      ".");
  Timer timer;
  for (int i = 0; i < main_runs; ++i) {
    CAFFE_ENFORCE(Run(), "Main run ", i, " has failed.");
  }
  auto millis = timer.MilliSeconds();
  std::cout << "Main run finished. Milliseconds per iter: "
            << millis / main_runs
            << ". Iters per second: " << 1000.0 * main_runs / millis << std::endl;

  if (run_individual) {
    std::cout << "AsyncSimpleNet does not do per-op benchmark. To do so, "
                 "switch to a simple net type." << std::endl;
  }
  return vector<float>{millis / main_runs};
}

REGISTER_NET(async_simple, AsyncSimpleNet);

} // namespace caffe2
