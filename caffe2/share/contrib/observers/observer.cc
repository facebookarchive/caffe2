#include "caffe2/share/contrib/observers/observer.h"
#include "caffe2/share/contrib/observers/observer_config.h"

#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include <random>

namespace caffe2 {
namespace {

bool registerGlobalTimeNetObserverCreator(int* /*pargc*/, char*** /*pargv*/) {
  SetGlobalNetObserverCreator([](NetBase* subject) {
      return caffe2::make_unique<TimeNetObserver>(subject);
    });
    return true;
}
}

REGISTER_CAFFE2_EARLY_INIT_FUNCTION(
    registerGlobalTimeNetObserverCreator,
    &registerGlobalTimeNetObserverCreator,
    "Caffe2 net global observer creator");

TimeNetObserver::TimeNetObserver(NetBase* subject_)
    : NetObserver(subject_),
    numRuns_(0) {
}

TimeNetObserver::~TimeNetObserver() {
}

bool TimeNetObserver::Start() {
  // Select whether to log the operator or the net.
  // We have one sample rate for the entire app.
  int netSampleRate = ObserverConfig::getNetSampleRate();
  int operatorNetSampleRatio = ObserverConfig::getOpoeratorNetSampleRatio();
  int skipIters = ObserverConfig::getSkipIters();
  if (skipIters <= numRuns_ &&
      netSampleRate > 0 &&
      rand() % netSampleRate == 0) {
    if (operatorNetSampleRatio > 0 &&
        rand() % operatorNetSampleRatio == 0) {
      logType_ = TimeNetObserver::OPERATOR_DELAY;
    } else {
      logType_ = TimeNetObserver::NET_DELAY;
    }
  } else {
    logType_ = TimeNetObserver::NONE;
  }
  numRuns_++;

  if (logType_ == TimeNetObserver::OPERATOR_DELAY) {
    /* Always recreate new operator  observers
       whenever we measure operator delay */
    const auto& operators = subject_->GetOperators();
    for (auto* op : operators) {
      op->SetObserver(
        caffe2::make_unique<TimeOperatorObserver>(op, this));
    }
  }

  if (logType_ != TimeNetObserver::NONE) {
    /* Only start timer when we need to */
    timer_.Start();
  }
  return true;
}

bool TimeNetObserver::Stop() {
  if (logType_ == TimeNetObserver::NET_DELAY) {
    auto current_run_time = timer_.MilliSeconds();
    ObserverConfig::getReporter()->printNet(subject_, current_run_time);
  } else if (logType_ == TimeNetObserver::OPERATOR_DELAY) {
    auto current_run_time = timer_.MilliSeconds();
    const auto& operators = subject_->GetOperators();
    std::vector<std::pair<std::string, double>> operator_delays;
    for (int idx = 0; idx < operators.size(); ++idx) {
      const auto* op = operators[idx];
      auto name = getObserverName(op, idx);
      double delay = static_cast<const TimeOperatorObserver *>
          (op->GetObserver())->getMilliseconds();
      std::pair<std::string, double> name_delay_pair = {name, delay};
      operator_delays.push_back(name_delay_pair);
    }
    ObserverConfig::getReporter()->printNetWithOperators(subject_, current_run_time,
        operator_delays);
    /* clear all operator delay after use so that we don't spent time
       collecting the operator delay info in later runs */
    for (auto* op : operators) {
      op->RemoveObserver();
    }
  }
  return true;
}

caffe2::string TimeNetObserver::getObserverName(
    const OperatorBase *op, int idx) const {
  string opType = op->has_debug_def() ? op->debug_def().type() : "NO_TYPE";
  string displayName =
      (op->has_debug_def() ? op->debug_def().name().size()
           ? op->debug_def().name()
           : (op->debug_def().output_size() ? op->debug_def().output(0)
                                            : "NO_OUTPUT") : "NO_DEF");
  caffe2::string name = "ID_" + caffe2::to_string(idx) + "_" +
      opType + "_" + displayName;
  return name;
}

TimeOperatorObserver::TimeOperatorObserver(
    OperatorBase* op,
    TimeNetObserver* netObserver)
    : ObserverBase<OperatorBase>(op),
      netObserver_(netObserver),
      milliseconds_(0) {
  CAFFE_ENFORCE(
      netObserver_, "Observers can't operate outside of the net");
}

TimeOperatorObserver::~TimeOperatorObserver() {}

bool TimeOperatorObserver::Start() {
  /* Get the time from the start of the net minus the time spent
     in previous invocations. It is the time spent on other operators.
     This way, when the operator finishes, the time from the start of the net
     minus the time spent in all other operators  is the total time on this
     operator. This is done to avoid saving a timer in each operator */
  milliseconds_ = netObserver_->getTimer().MilliSeconds() - milliseconds_;
  return true;
}

bool TimeOperatorObserver::Stop() {
  /* Time from the start of the net minus the time spent on all other
     operators is the time spent on this operator */
  milliseconds_ = netObserver_->getTimer().MilliSeconds() - milliseconds_;
  return true;
}

double TimeOperatorObserver::getMilliseconds() const {
  return milliseconds_;
}
}
